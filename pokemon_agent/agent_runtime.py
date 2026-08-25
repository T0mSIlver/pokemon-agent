"""Agent workspace, telemetry, and observation runtime for Pokemon Agent.

:class:`AgentRuntime` assembles one observation bundle per turn and owns the
workspace artifacts the dashboard reads. The pieces it composes live next door:
:mod:`pokemon_agent.state_analysis` reads the state dict, :mod:`pokemon_agent.rendering`
draws the annotated frame, :mod:`pokemon_agent.objectives` runs the objective
ladder, and :mod:`pokemon_agent.progress` handles auto-saves and stuck detection.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from pokemon_agent.navigation import LiveNavigationSnapshot
from pokemon_agent.objectives import (
    ObjectiveEngine,
    ObjectiveRecord,
    load_red_objective_packs,
)
from pokemon_agent.progress import ProgressMonitor, slugify
from pokemon_agent.rendering import (
    measure_text,
    normalise_map_grid,
    render_map_inset,
    render_navigation_overlay,
    wrap_text,
)
from pokemon_agent.state_analysis import (
    MOVE_METADATA,
    TYPE_EFFECTIVENESS,
    badge_count,
    bag_item_counts,
    bag_item_names,
    build_battle_guidance,
    build_movement_guidance,
    build_state_delta,
    classify_action_feedback,
    classify_ui_mode,
    classify_ui_state,
    extract_key_state,
    selector_matches,
)

JsonDict = dict[str, Any]

#: Everything this module used to define itself stays importable from here.
#: ``server.py`` and ``pi_supervisor.py`` import from ``agent_runtime``, so the
#: names are the contract even though the implementations moved.
__all__ = [
    "AgentRuntime",
    "JsonDict",
    "MOVE_METADATA",
    "ObjectiveEngine",
    "ObjectiveRecord",
    "TYPE_EFFECTIVENESS",
    "build_movement_guidance",
    "build_state_delta",
    "classify_action_feedback",
    "classify_ui_mode",
    "classify_ui_state",
    "extract_key_state",
    "load_red_objective_packs",
    "render_navigation_overlay",
    "utc_now",
]

# The former private helpers, under their former names, for anything that
# reached past the public surface.
_slugify = slugify
_measure_text = measure_text
_wrap_text = wrap_text
_normalise_map_grid = normalise_map_grid
_render_map_inset = render_map_inset
_selector_matches = selector_matches
_bag_item_counts = bag_item_counts
_bag_item_names = bag_item_names
_badge_count = badge_count


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        os.replace(tmp, str(path))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def _stable_id(*parts: Any) -> str:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return digest


class AgentRuntime:
    """Owns workspace artifacts, telemetry history, and deterministic assist logic."""

    def __init__(
        self,
        *,
        data_dir: Path,
        workspace_dir: Path,
        objective_engine: Optional[ObjectiveEngine] = None,
        history_limit: int = 400,
        visited_lookup: Optional[Callable[[int], set[tuple[int, int]]]] = None,
        map_grid_lookup: Optional[Callable[[int], Optional[JsonDict]]] = None,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.objective_engine = objective_engine or ObjectiveEngine()
        self.history_limit = history_limit
        # SEAM: the server sets this to ExploredMaps.visited (map_id -> absolute
        # tiles the player has stood on) so the overlay can shade walked ground.
        # Left as None the overlay renders exactly as before.
        self.visited_lookup = visited_lookup
        # SEAM: the server sets this to ExploredMaps.grid (map_id -> the whole
        # stored map as width/height plus seen/walkable/walked/warp tile sets)
        # so the overlay can inset a mini-map. Left as None the overlay renders
        # exactly as before.
        self.map_grid_lookup = map_grid_lookup
        self.progress = ProgressMonitor(data_dir=self.data_dir)
        self.event_history: deque[JsonDict] = deque(maxlen=history_limit)
        self.latest_bundle: Optional[JsonDict] = None
        self.live_bundle: Optional[JsonDict] = None
        self.last_state: Optional[JsonDict] = None
        self.dialog_transcript_recent: deque[JsonDict] = deque(maxlen=12)
        self.last_dialog_text = ""
        self.dialog_last_change_at: Optional[str] = None
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.workspace_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_files()

    # The progress monitor owns these three; they stay readable and writable
    # here because they were plain attributes before the split.
    @property
    def recent_trajectory(self) -> deque[JsonDict]:
        return self.progress.recent_trajectory

    @property
    def last_objective_id(self) -> Optional[str]:
        return self.progress.last_objective_id

    @last_objective_id.setter
    def last_objective_id(self, value: Optional[str]) -> None:
        self.progress.last_objective_id = value

    @property
    def action_events_since_objective_change(self) -> int:
        return self.progress.action_events_since_objective_change

    @action_events_since_objective_change.setter
    def action_events_since_objective_change(self, value: int) -> None:
        self.progress.action_events_since_objective_change = value

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "latest_frame": self.workspace_dir / "latest_frame.png",
            "latest_frame_annotated": self.workspace_dir / "latest_frame_annotated.png",
            "live_frame": self.workspace_dir / "live_frame.png",
            "live_frame_annotated": self.workspace_dir / "live_frame_annotated.png",
            "turn_context_json": self.workspace_dir / "turn_context.json",
            "latest_observation_json": self.debug_dir / "latest_observation.json",
            "current_objective_json": self.debug_dir / "current_objective.json",
            "run_log_jsonl": self.debug_dir / "run_log.jsonl",
        }

    def _ensure_workspace_files(self) -> None:
        run_log = self.artifacts["run_log_jsonl"]
        run_log.parent.mkdir(parents=True, exist_ok=True)
        run_log.touch(exist_ok=True)
        for key in ("turn_context_json", "latest_observation_json", "current_objective_json"):
            path = self.artifacts[key]
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")

    def _write_json(self, path: Path, payload: Any) -> None:
        _atomic_write_text(
            path,
            json.dumps(payload, indent=2, sort_keys=False, default=_json_default),
        )

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default) + "\n")

    def _read_json(self, path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return fallback

    def load_turn_context(self) -> JsonDict:
        return self._read_json(self.artifacts["turn_context_json"], {})

    def _record_event(self, event_type: str, payload: JsonDict) -> JsonDict:
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            **payload,
        }
        self.event_history.append(event)
        self._append_jsonl(self.artifacts["run_log_jsonl"], event)
        return event

    def record_external_event(self, event_type: str, payload: JsonDict) -> JsonDict:
        """Record an external API event so dashboard history matches websocket traffic."""
        return self._record_event(event_type, payload)

    def _tail_jsonl(self, path: Path, limit: int) -> list[JsonDict]:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result: list[JsonDict] = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def history(self, limit: int = 200) -> list[JsonDict]:
        if self.event_history:
            return list(self.event_history)[-limit:]
        return self._tail_jsonl(self.artifacts["run_log_jsonl"], limit)

    def _update_dialog_guidance(
        self,
        *,
        screen_text: JsonDict,
        state: JsonDict,
    ) -> tuple[JsonDict, Optional[JsonDict]]:
        dialog = state.get("dialog") or {}
        dialog_active = bool(state.get("dialog_active") or dialog.get("active"))
        text = str(screen_text.get("text") or "").strip()
        changed_event = None
        if (
            dialog_active
            and text
            and not text.startswith("Dialog box visible")
            and text != self.last_dialog_text
        ):
            self.last_dialog_text = text
            self.dialog_last_change_at = utc_now()
            entry = {"timestamp": self.dialog_last_change_at, "text": text}
            self.dialog_transcript_recent.append(entry)
            changed_event = entry
        elif not dialog_active:
            self.last_dialog_text = ""

        return (
            {
                "transcript_recent": [
                    entry["text"] for entry in list(self.dialog_transcript_recent)[-4:]
                ],
                "should_continue": dialog_active,
                "last_change_at": self.dialog_last_change_at,
                "printing": bool(dialog.get("printing")),
                "waiting_for_input": bool(dialog.get("waiting_for_input")),
            },
            changed_event,
        )

    def _latest_observed_frame_artifacts(self) -> JsonDict:
        artifacts = ((self.latest_bundle or {}).get("artifacts") or {}).copy()
        latest_frame = str(artifacts.get("latest_frame") or self.artifacts["latest_frame"])
        latest_frame_annotated = str(
            artifacts.get("latest_frame_annotated") or self.artifacts["latest_frame_annotated"]
        )
        return {
            "latest_frame": latest_frame,
            "latest_frame_annotated": latest_frame_annotated,
        }

    def _observation_frame_paths(self, observation_id: str) -> dict[str, Path]:
        observation_dir = self.workspace_dir / "observations" / observation_id
        observation_dir.mkdir(parents=True, exist_ok=True)
        return {
            "latest_frame": observation_dir / "latest_frame.png",
            "latest_frame_annotated": observation_dir / "latest_frame_annotated.png",
        }

    def _artifact_payload(
        self,
        *,
        latest_frame: Optional[str] = None,
        latest_frame_annotated: Optional[str] = None,
    ) -> JsonDict:
        observed = self._latest_observed_frame_artifacts()
        return {
            "latest_frame": latest_frame or observed["latest_frame"],
            "latest_frame_annotated": latest_frame_annotated or observed["latest_frame_annotated"],
            "live_frame": str(self.artifacts["live_frame"]),
            "live_frame_annotated": str(self.artifacts["live_frame_annotated"]),
            "turn_context_json": str(self.artifacts["turn_context_json"]),
            "latest_observation_json": str(self.artifacts["latest_observation_json"]),
            "current_objective_json": str(self.artifacts["current_objective_json"]),
            "run_log_jsonl": str(self.artifacts["run_log_jsonl"]),
        }

    def _next_observation_id(
        self,
        *,
        generated_at: str,
        reason: str,
        state: JsonDict,
    ) -> str:
        position = (state.get("player") or {}).get("position") or {}
        return "obs-" + _stable_id(
            generated_at,
            reason,
            (state.get("map") or {}).get("map_id"),
            (state.get("map") or {}).get("map_name"),
            position.get("x"),
            position.get("y"),
            (state.get("metadata") or {}).get("frame_count"),
        )

    def _write_turn_context(self, bundle: JsonDict) -> JsonDict:
        """Write the slim display-only turn context. Not a model contract."""
        current = (bundle.get("objective") or {}).get("current") or {}
        state = bundle.get("state") or {}
        player = state.get("player") or {}
        position = player.get("position") or {}
        screen_text = bundle.get("screen_text") or {}
        context = {
            "observation_id": bundle.get("observation_id"),
            "objective": {
                "id": current.get("id"),
                "summary": current.get("summary"),
                "completion_predicate": current.get("completion_predicate"),
            },
            "position": {
                "map_name": (state.get("map") or {}).get("map_name"),
                "x": position.get("x"),
                "y": position.get("y"),
                "facing": player.get("facing"),
            },
            "ui": {
                "mode": screen_text.get("ui_mode"),
                "screen_text": screen_text.get("text"),
            },
        }
        self._write_json(self.artifacts["turn_context_json"], context)
        return context

    def _snapshot_from_navigation_payload(
        self,
        navigation: Optional[JsonDict],
    ) -> Optional[LiveNavigationSnapshot]:
        snapshot = None
        if navigation:
            snapshot_payload = navigation.get("snapshot") or {}
            if snapshot_payload:
                try:
                    snapshot = LiveNavigationSnapshot(
                        map_id=int(snapshot_payload["map_id"]),
                        map_name=str(snapshot_payload["map_name"]),
                        player_position=(
                            int(snapshot_payload["player_position"]["x"]),
                            int(snapshot_payload["player_position"]["y"]),
                        ),
                        facing=str(snapshot_payload.get("facing", "unknown")),
                        tileset=str(snapshot_payload.get("tileset", "UNKNOWN")),
                        window_top_left=(
                            int(snapshot_payload["window_top_left"]["x"]),
                            int(snapshot_payload["window_top_left"]["y"]),
                        ),
                        terrain=list(snapshot_payload.get("terrain", [])),
                        sprite_positions=[
                            (int(item["x"]), int(item["y"]))
                            for item in snapshot_payload.get("sprites", [])
                        ],
                        valid_moves=list(snapshot_payload.get("valid_moves", [])),
                        warps=list(snapshot_payload.get("warps", [])),
                        signs=list(snapshot_payload.get("signs", [])),
                        map_dimensions=snapshot_payload.get("map_dimensions"),
                        interaction=snapshot_payload.get("interaction"),
                    )
                except Exception:  # noqa: BLE001
                    snapshot = None
        return snapshot

    def _visited_tiles(
        self,
        snapshot: Optional[LiveNavigationSnapshot],
    ) -> Optional[set[tuple[int, int]]]:
        if snapshot is None or self.visited_lookup is None:
            return None
        try:
            return {(int(x), int(y)) for x, y in self.visited_lookup(snapshot.map_id)}
        except Exception:  # noqa: BLE001
            return None

    def _map_grid(
        self,
        snapshot: Optional[LiveNavigationSnapshot],
    ) -> Optional[JsonDict]:
        if snapshot is None or self.map_grid_lookup is None:
            return None
        try:
            grid = self.map_grid_lookup(snapshot.map_id)
        except Exception:  # noqa: BLE001
            return None
        return grid if isinstance(grid, dict) else None

    def _coerce_screen_image(self, emulator: Any) -> Image.Image:
        screen = emulator.get_screen()
        if not isinstance(screen, Image.Image):
            screen = Image.fromarray(screen)
        return screen

    def _write_frame_artifacts(
        self,
        *,
        screen: Image.Image,
        annotated: Image.Image,
        frame_path: Path,
        annotated_path: Path,
    ) -> None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        screen.save(buf, format="PNG")
        _atomic_write_bytes(frame_path, buf.getvalue())
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        _atomic_write_bytes(annotated_path, buf.getvalue())

    def _annotate(
        self,
        emulator: Any,
        snapshot: Optional[LiveNavigationSnapshot],
        *,
        objective: Optional[JsonDict],
        goal: Optional[tuple[int, int]],
    ) -> tuple[Image.Image, Image.Image]:
        """Grab the screen once and draw the overlay the model navigates from."""
        screen = self._coerce_screen_image(emulator)
        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=objective,
            goal=goal,
            visited=self._visited_tiles(snapshot),
            map_grid=self._map_grid(snapshot),
        )
        return screen, annotated

    def sync_live_view(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        snapshot = self._snapshot_from_navigation_payload(navigation)
        screen, annotated = self._annotate(
            emulator,
            snapshot,
            objective=current_objective["current"],
            goal=None,
        )
        self._write_frame_artifacts(
            screen=screen,
            annotated=annotated,
            frame_path=self.artifacts["live_frame"],
            annotated_path=self.artifacts["live_frame_annotated"],
        )

        previous_bundle = self.live_bundle or self.latest_bundle or {}
        dialog_active = bool(
            state.get("dialog_active") or (state.get("dialog") or {}).get("active")
        )
        screen_text = self._preserved_screen_text(previous_bundle, state, dialog_active)
        dialog_guidance = (previous_bundle.get("dialog_guidance") or {}).copy()
        dialog_guidance.setdefault("transcript_recent", [])
        dialog_guidance["should_continue"] = dialog_active
        dialog_guidance.setdefault("last_change_at", self.dialog_last_change_at)

        live_bundle = {
            "generated_at": utc_now(),
            "observation_id": (self.latest_bundle or {}).get("observation_id"),
            "reason": "realtime_live_sync",
            "source": "live_sync",
            "artifacts": self._artifact_payload(),
            "state": state,
            "navigation": navigation,
            "screen_text": screen_text,
            "objective": current_objective,
            "recent_action": previous_bundle.get("recent_action") or {},
            "movement_guidance": build_movement_guidance(snapshot=snapshot),
            "dialog_guidance": dialog_guidance,
            "battle_guidance": build_battle_guidance(state, dialog_guidance),
            "state_delta": previous_bundle.get("state_delta")
            or {
                "changed": False,
                "summary": ["Live frame sync only. POST /action to advance the game."],
                "movement": None,
            },
            "stuck": previous_bundle.get("stuck")
            or {
                "level": "clear",
                "reason": "No stuck signal recorded yet.",
                "objective_action_count": 0,
            },
            "workspace_dir": str(self.workspace_dir),
            "turn_context": self.load_turn_context(),
        }

        self.live_bundle = live_bundle
        return {
            "generated_at": live_bundle["generated_at"],
            "source": live_bundle["source"],
            "artifacts": live_bundle["artifacts"],
            "screen_text": live_bundle["screen_text"],
        }

    def _preserved_screen_text(
        self,
        previous_bundle: JsonDict,
        state: JsonDict,
        dialog_active: bool,
    ) -> JsonDict:
        """Carry the last real screen text forward; a live frame reads no new text."""
        previous_screen_text = previous_bundle.get("screen_text") or {}
        preserved_text = ""
        preserved_source = "live_sync"
        if (
            isinstance(previous_screen_text.get("text"), str)
            and previous_screen_text.get("text")
            and bool(previous_screen_text.get("dialog_active")) == dialog_active
        ):
            preserved_text = previous_screen_text["text"]
            preserved_source = "live_sync_cached"
        if not preserved_text:
            preserved_text = "Live frame sync active. POST /action to advance the game."
        return {
            "text": preserved_text,
            "source": preserved_source,
            "ui_mode": classify_ui_mode(state),
            "dialog_active": dialog_active,
        }

    def refresh(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        reason: str,
        source: str,
        requested_actions: Optional[list[str]] = None,
        navigation_plan: Optional[JsonDict] = None,
        navigation_execution: Optional[JsonDict] = None,
        explicit_save: Optional[JsonDict] = None,
    ) -> JsonDict:
        """Produce one observation bundle: frames, objective, deltas, telemetry."""
        current_objective = self.objective_engine.evaluate(state)
        snapshot = self._snapshot_from_navigation_payload(navigation)
        goal = self._goal_from_execution(navigation_execution)

        generated_at = utc_now()
        observation_id = self._next_observation_id(
            generated_at=generated_at,
            reason=reason,
            state=state,
        )
        observation_frames = self._observation_frame_paths(observation_id)
        screen, annotated = self._annotate(
            emulator,
            snapshot,
            objective=current_objective["current"],
            goal=goal,
        )
        for frame_path, annotated_path in (
            (observation_frames["latest_frame"], observation_frames["latest_frame_annotated"]),
            (self.artifacts["latest_frame"], self.artifacts["latest_frame_annotated"]),
        ):
            self._write_frame_artifacts(
                screen=screen,
                annotated=annotated,
                frame_path=frame_path,
                annotated_path=annotated_path,
            )

        screen_text = classify_ui_state(state)
        state_delta = build_state_delta(self.last_state, state)
        action_feedback = classify_action_feedback(
            source=source,
            requested_actions=requested_actions,
            state_before=self.last_state,
            state_after=state,
            state_delta=state_delta,
            navigation_plan=navigation_plan,
            navigation_execution=navigation_execution,
        )
        dialog_guidance, _dialog_change = self._update_dialog_guidance(
            screen_text=screen_text,
            state=state,
        )
        auto_saves = self.progress.maybe_auto_save(
            emulator=emulator,
            state=state,
            objective=current_objective,
            state_delta=state_delta,
            requested_actions=requested_actions,
            source=source,
        )
        if explicit_save:
            auto_saves.append(explicit_save)
        stuck = self.progress.detect_stuck(
            state=state,
            objective=current_objective,
            source=source,
            requested_actions=requested_actions,
        )

        bundle: JsonDict = {
            "generated_at": generated_at,
            "observation_id": observation_id,
            "reason": reason,
            "source": source,
            "artifacts": self._artifact_payload(
                latest_frame=str(observation_frames["latest_frame"]),
                latest_frame_annotated=str(observation_frames["latest_frame_annotated"]),
            ),
            "state": state,
            "navigation": navigation,
            "screen_text": screen_text,
            "objective": current_objective,
            "recent_action": action_feedback,
            "movement_guidance": build_movement_guidance(snapshot=snapshot),
            "dialog_guidance": dialog_guidance,
            "battle_guidance": build_battle_guidance(state, dialog_guidance),
            "state_delta": state_delta,
            "stuck": stuck,
            "workspace_dir": str(self.workspace_dir),
        }
        bundle["turn_context"] = self._write_turn_context(bundle)

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        self._write_json(self.artifacts["latest_observation_json"], bundle)

        events = self._record_observation_events(
            reason=reason,
            source=source,
            objective=current_objective,
            summary=action_feedback["summary"],
            auto_saves=auto_saves,
            stuck=stuck,
        )

        self.latest_bundle = bundle
        self.live_bundle = bundle
        self.last_state = state
        self.progress.note_objective(current_objective["current"]["id"])
        return {"bundle": bundle, "events": events}

    @staticmethod
    def _goal_from_execution(
        navigation_execution: Optional[JsonDict],
    ) -> Optional[tuple[int, int]]:
        target = (navigation_execution or {}).get("target") or {}
        if target.get("x") is None or target.get("y") is None:
            return None
        return (int(target["x"]), int(target["y"]))

    def _record_observation_events(
        self,
        *,
        reason: str,
        source: str,
        objective: JsonDict,
        summary: str,
        auto_saves: list[JsonDict],
        stuck: JsonDict,
    ) -> list[JsonDict]:
        """Append this observation to the run log, in the order the dashboard expects."""
        events: list[JsonDict] = [
            self._record_event(
                "observe",
                {
                    "reason": reason,
                    "source": source,
                    "objective_id": objective["current"]["id"],
                    "summary": summary,
                },
            )
        ]
        if objective["current"]["id"] != self.last_objective_id:
            events.append(
                self._record_event(
                    "objective",
                    {
                        "objective": objective["current"],
                        "progress_percent": objective["progress_percent"],
                    },
                )
            )
        for save_event in auto_saves:
            events.append(self._record_event("save", save_event))
        if stuck["level"] != "clear":
            events.append(self._record_event("stuck", stuck))
        return events

    def dashboard_state(self) -> JsonDict:
        bundle = (
            self.live_bundle
            or self.latest_bundle
            or self._read_json(self.artifacts["latest_observation_json"], {})
        )
        if not bundle:
            return {
                "generated_at": utc_now(),
                "visuals": {},
                "agent_intent": {},
                "world_state": {},
                "memory_and_progress": {},
                "timeline": self.history(50),
            }

        state = bundle.get("state") or {}
        navigation = bundle.get("navigation") or {}
        snapshot = navigation.get("snapshot") or {}
        visual_artifacts = bundle.get("artifacts") or {}
        use_live_frames = (
            bundle.get("source") == "live_sync"
            and visual_artifacts.get("live_frame")
            and visual_artifacts.get("live_frame_annotated")
        )
        return {
            "observation_id": bundle.get("observation_id"),
            "generated_at": bundle.get("generated_at"),
            "visuals": {
                "raw_frame_path": (
                    visual_artifacts.get("live_frame")
                    if use_live_frames
                    else visual_artifacts.get("latest_frame")
                ),
                "annotated_frame_path": (
                    visual_artifacts.get("live_frame_annotated")
                    if use_live_frames
                    else visual_artifacts.get("latest_frame_annotated")
                ),
                "frame_timestamp": bundle.get("generated_at"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "screen_text": bundle.get("screen_text"),
            },
            "agent_intent": {
                "objective": (bundle.get("objective") or {}).get("current") or {},
                "turn_context": bundle.get("turn_context") or self.load_turn_context(),
                "recent_action": bundle.get("recent_action"),
                "movement_guidance": bundle.get("movement_guidance"),
                "dialog_guidance": bundle.get("dialog_guidance"),
                "battle_guidance": bundle.get("battle_guidance"),
                "state_delta": bundle.get("state_delta"),
            },
            "world_state": {
                "map": state.get("map"),
                "player": state.get("player"),
                "party": state.get("party"),
                "battle": state.get("battle"),
                "dialog": state.get("dialog"),
                "interaction": state.get("interaction") or snapshot.get("interaction"),
                "valid_moves": snapshot.get("valid_moves", []),
                "live_ascii": snapshot.get("ascii"),
                "navigation": navigation,
            },
            "memory_and_progress": {
                "progress_percent": (bundle.get("objective") or {}).get("progress_percent"),
                "stuck": bundle.get("stuck"),
                "workspace": {
                    "workspace_dir": bundle.get("workspace_dir"),
                    "turn_context_json": visual_artifacts.get("turn_context_json"),
                },
            },
            "timeline": self.history(80),
            "artifacts": visual_artifacts,
        }
