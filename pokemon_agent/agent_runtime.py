"""Agent workspace, telemetry, and observation runtime for Pokemon Agent."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.navigation import LiveNavigationSnapshot, NavigationStore

JsonDict = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


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


def _truncate_text_block(value: Any, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]..."


@dataclass(slots=True)
class ObjectiveRecord:
    id: str
    title: str
    summary: str
    completion_predicate: str
    failure_hints: list[str]
    save_recommendation: str
    route_hint: str
    priority: int
    current: bool
    completed: bool
    status: str
    progress_percent: int

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class CheckpointRecord:
    id: str
    title: str
    summary: str
    objective_id: str
    map_name: str
    created_at: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class RecoveryCandidate:
    name: str
    path: str
    reason: str
    score: int
    modified_at: str
    source: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@lru_cache(maxsize=1)
def load_red_objective_pack() -> list[JsonDict]:
    path = Path(__file__).parent / "data" / "red_objectives.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_party(party: list[JsonDict]) -> str:
    if not party:
        return "No Pokemon in party."
    summary = []
    for mon in party[:3]:
        summary.append(
            f"{mon.get('nickname') or mon.get('species', 'Unknown')} "
            f"Lv{mon.get('level', '?')} "
            f"HP {mon.get('hp', '?')}/{mon.get('max_hp', '?')}"
        )
    remainder = len(party) - len(summary)
    if remainder > 0:
        summary.append(f"+{remainder} more")
    return "; ".join(summary)


def extract_key_state(state: Optional[JsonDict]) -> JsonDict:
    if not state:
        return {}
    player = state.get("player") or {}
    flags = state.get("flags") or {}
    battle = state.get("battle") or {}
    dialog = state.get("dialog") or {}
    map_info = state.get("map") or {}
    party = state.get("party") or []
    return {
        "map_name": map_info.get("map_name"),
        "map_id": map_info.get("map_id"),
        "position": player.get("position") or {},
        "facing": player.get("facing"),
        "badge_count": player.get("badge_count", flags.get("badge_count", 0)),
        "money": player.get("money"),
        "dialog_active": bool(state.get("dialog_active") or dialog.get("active")),
        "battle_active": bool(battle.get("in_battle")),
        "battle_type": battle.get("type"),
        "has_pokedex": bool(flags.get("has_pokedex")),
        "has_oaks_parcel": bool(flags.get("has_oaks_parcel")),
        "party_summary": [
            {
                "name": mon.get("nickname") or mon.get("species"),
                "hp": mon.get("hp"),
                "max_hp": mon.get("max_hp"),
                "status": mon.get("status"),
                "level": mon.get("level"),
            }
            for mon in party
        ],
    }


def _preferred_direction_hint(text: str) -> Optional[str]:
    lowered = text.lower()
    for direction, words in (
        ("north", ("north", "up")),
        ("south", ("south", "down")),
        ("west", ("west", "left")),
        ("east", ("east", "right")),
    ):
        if any(word in lowered for word in words):
            return direction
    return None


def _progress_amount(direction: str, start: tuple[int, int], coord: tuple[int, int]) -> int:
    if direction == "north":
        return start[1] - coord[1]
    if direction == "south":
        return coord[1] - start[1]
    if direction == "west":
        return start[0] - coord[0]
    if direction == "east":
        return coord[0] - start[0]
    return 0


def build_movement_guidance(
    *,
    state: JsonDict,
    snapshot: Optional[LiveNavigationSnapshot],
    navigation_store: Optional[NavigationStore],
    objective: JsonDict,
) -> JsonDict:
    if snapshot is None:
        return {
            "summary": "Navigation guidance unavailable because no live snapshot was captured.",
            "notes": [],
            "preferred_direction": None,
            "candidate_route": None,
        }

    current = snapshot.player_position
    notes: list[str] = [
        (
            "Use the annotated screenshot as primary evidence. "
            "Use ASCII only as a symbolic collision summary."
        ),
        f"Immediate legal moves: {', '.join(snapshot.valid_moves) or 'none'}.",
    ]
    interaction = snapshot.interaction or {}
    target = interaction.get("target_coord") or {}
    if interaction.get("source") == "blocked_tile":
        notes.append(f"Forward movement is blocked at ({target.get('x')}, {target.get('y')}).")

    preferred_direction = _preferred_direction_hint(objective["current"].get("route_hint", ""))
    if preferred_direction is None:
        preferred_direction = {
            "up": "north",
            "down": "south",
            "left": "west",
            "right": "east",
        }.get(snapshot.facing)

    candidate_route = None
    if navigation_store is not None and preferred_direction is not None:
        location_map = navigation_store.get(snapshot.key)
        if location_map is not None:
            distances = location_map.distance_map(
                current,
                extra_blockers=snapshot.sprite_set,
            )
            candidates = [
                coord
                for coord in distances
                if coord != current and _progress_amount(preferred_direction, current, coord) > 0
            ]
            if candidates:
                best = min(
                    candidates,
                    key=lambda coord: (
                        distances[coord],
                        abs(coord[0] - current[0]),
                        abs(coord[1] - current[1]),
                        coord[1],
                        coord[0],
                    ),
                )
                plan = location_map.plan_route(
                    start=current,
                    goal=best,
                    extra_blockers=snapshot.sprite_set,
                    allow_partial=False,
                )
                candidate_route = {
                    "direction": preferred_direction,
                    "target": {"x": best[0], "y": best[1]},
                    "actions": plan.actions,
                    "steps": len(plan.actions),
                }
                if plan.actions:
                    notes.append(
                        f"Best explored {preferred_direction}-progress route starts with: "
                        f"{', '.join(plan.actions[:4])}."
                    )
                notes.append(
                    f"Nearest explored tile that makes {preferred_direction} progress is "
                    f"({best[0]}, {best[1]})."
                )

    if interaction.get("source") == "blocked_tile" and snapshot.valid_moves:
        sidesteps = [move for move in snapshot.valid_moves if move != snapshot.facing]
        if sidesteps:
            notes.append(f"Because forward is blocked, sidestep first: {', '.join(sidesteps)}.")

    summary = notes[-1] if notes else "Navigation guidance unavailable."
    return {
        "summary": summary,
        "notes": notes,
        "preferred_direction": preferred_direction,
        "candidate_route": candidate_route,
    }


def build_state_delta(before: Optional[JsonDict], after: JsonDict) -> JsonDict:
    current = extract_key_state(after)
    previous = extract_key_state(before)
    if not previous:
        return {
            "changed": True,
            "summary": ["Initial observation snapshot captured."],
            "fields": {"initial": current},
            "movement": None,
        }

    fields: JsonDict = {}
    summary: list[str] = []

    before_map = previous.get("map_name")
    after_map = current.get("map_name")
    if before_map != after_map:
        fields["map"] = {"before": before_map, "after": after_map}
        summary.append(f"Map changed from {before_map or 'unknown'} to {after_map or 'unknown'}.")

    before_pos = previous.get("position") or {}
    after_pos = current.get("position") or {}
    movement = None
    if before_pos != after_pos:
        movement = {
            "before": before_pos,
            "after": after_pos,
            "dx": (after_pos.get("x") or 0) - (before_pos.get("x") or 0),
            "dy": (after_pos.get("y") or 0) - (before_pos.get("y") or 0),
        }
        movement["manhattan"] = abs(movement["dx"]) + abs(movement["dy"])
        fields["position"] = movement
        summary.append(
            "Player position changed "
            f"from ({before_pos.get('x')}, {before_pos.get('y')}) "
            f"to ({after_pos.get('x')}, {after_pos.get('y')})."
        )

    for key, label in (
        ("dialog_active", "Dialog"),
        ("battle_active", "Battle"),
        ("has_pokedex", "Pokedex"),
        ("has_oaks_parcel", "Oak's Parcel"),
    ):
        if previous.get(key) != current.get(key):
            fields[key] = {"before": previous.get(key), "after": current.get(key)}
            state = "enabled" if current.get(key) else "disabled"
            summary.append(f"{label} is now {state}.")

    if previous.get("badge_count") != current.get("badge_count"):
        fields["badge_count"] = {
            "before": previous.get("badge_count"),
            "after": current.get("badge_count"),
        }
        summary.append(
            f"Badge count changed from {previous.get('badge_count', 0)} to "
            f"{current.get('badge_count', 0)}."
        )

    party_changes: list[str] = []
    before_party = {entry.get("name"): entry for entry in previous.get("party_summary", [])}
    for entry in current.get("party_summary", []):
        name = entry.get("name")
        prior = before_party.get(name)
        if not prior:
            party_changes.append(f"{name} joined the party.")
            continue
        if prior.get("hp") != entry.get("hp"):
            party_changes.append(f"{name} HP changed from {prior.get('hp')} to {entry.get('hp')}.")
        if prior.get("status") != entry.get("status"):
            party_changes.append(
                f"{name} status changed from {prior.get('status')} to {entry.get('status')}."
            )
    if party_changes:
        fields["party"] = party_changes
        summary.extend(party_changes[:3])

    changed = bool(fields)
    if not summary:
        summary.append("No observable structured state change.")
    return {
        "changed": changed,
        "summary": summary,
        "fields": fields,
        "movement": movement,
    }


def classify_action_feedback(
    *,
    source: str,
    requested_actions: Optional[list[str]],
    state_before: Optional[JsonDict],
    state_after: JsonDict,
    state_delta: JsonDict,
    navigation_plan: Optional[JsonDict] = None,
    navigation_execution: Optional[JsonDict] = None,
) -> JsonDict:
    requested_actions = requested_actions or []
    tags: list[str] = []
    notes: list[str] = []

    if state_delta.get("fields", {}).get("map"):
        tags.append("map_transition")
        notes.append("Entered a different map.")
    if state_delta.get("movement"):
        tags.append("movement")
        notes.append("Player position changed.")
    if state_delta.get("fields", {}).get("dialog_active"):
        tags.append("dialog_state_change")
        notes.append("Dialog state changed.")
    if state_delta.get("fields", {}).get("battle_active"):
        if (state_before or {}).get("battle", {}).get("in_battle"):
            tags.append("battle_ended")
        elif state_after.get("battle", {}).get("in_battle"):
            tags.append("battle_started")
        notes.append("Battle state changed.")
    if state_delta.get("fields", {}).get("badge_count"):
        tags.append("milestone")
        notes.append("A badge milestone changed.")
    if not state_delta.get("changed") and requested_actions:
        tags.append("no_progress")
        notes.append("Structured state did not change after the requested actions.")

    if source == "navigation" and navigation_execution:
        if navigation_execution.get("success"):
            tags.append("navigation_success")
        else:
            tags.append("navigation_partial")
        notes.append(navigation_execution.get("status", "Navigation result recorded."))
    elif source == "action" and requested_actions:
        notes.append(f"Executed {len(requested_actions)} raw actions.")
    elif source == "observe":
        notes.append("Fresh observation generated for Pi.")

    if not tags:
        tags.append("observe")

    return {
        "source": source,
        "requested_actions": requested_actions,
        "summary": notes[0],
        "notes": notes,
        "tags": tags,
        "navigation_plan": navigation_plan,
        "navigation_execution": navigation_execution,
    }


def classify_ui_mode(state: JsonDict) -> str:
    battle = state.get("battle") or {}
    dialog = state.get("dialog") or {}
    if battle.get("in_battle"):
        return "battle"
    if dialog.get("active") or state.get("dialog_active"):
        return "dialog"
    return "overworld"


def extract_screen_text(image: Image.Image, state: JsonDict) -> JsonDict:
    dialog = state.get("dialog") or {}
    dialog_active = bool(state.get("dialog_active") or dialog.get("active"))
    text = ""
    source = "none"
    note = ""

    crop = image
    if dialog_active:
        crop = image.crop((0, int(image.height * 0.56), image.width, image.height))

    try:
        import pytesseract  # type: ignore[import-not-found]

        scaled = crop.resize((crop.width * 3, crop.height * 3))
        bw = scaled.convert("L").point(lambda px: 255 if px > 120 else 0)
        raw = pytesseract.image_to_string(
            bw,
            config="--psm 6",
        )
        text = re.sub(r"\s+", " ", raw or "").strip()
        if text:
            source = "ocr_dialog" if dialog_active else "ocr_frame"
    except Exception as exc:  # noqa: BLE001
        note = f"OCR unavailable: {type(exc).__name__}"

    if not text and dialog_active:
        waiting = "waiting for input" if dialog.get("waiting_for_input") else "printing"
        text = f"Dialog box visible; OCR text unavailable ({waiting})."
        source = "dialog_state"
    elif not text:
        text = "No readable screen text extracted."
        source = "none"

    return {
        "text": text,
        "source": source,
        "note": note,
        "dialog_active": dialog_active,
        "ui_mode": classify_ui_mode(state),
    }


def render_navigation_overlay(
    image: Image.Image,
    snapshot: Optional[LiveNavigationSnapshot],
    *,
    objective: Optional[JsonDict] = None,
    goal: Optional[tuple[int, int]] = None,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    if not snapshot:
        draw.rectangle((0, 0, canvas.width, 22), fill=(12, 17, 26, 200))
        draw.text((6, 6), "Navigation overlay unavailable", fill=(255, 255, 255, 255), font=font)
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    tile_width = canvas.width / 10
    tile_height = canvas.height / 9

    for local_y, row in enumerate(snapshot.terrain):
        for local_x, tile in enumerate(row):
            left = int(local_x * tile_width)
            top = int(local_y * tile_height)
            right = int((local_x + 1) * tile_width)
            bottom = int((local_y + 1) * tile_height)
            absolute = snapshot.local_to_absolute(local_x, local_y)
            fill = (24, 123, 73, 64) if tile else (180, 58, 58, 80)
            outline = (110, 230, 174, 180) if tile else (255, 120, 120, 180)
            draw.rectangle((left, top, right, bottom), outline=outline, fill=fill, width=1)
            coord_label = f"{absolute[0]},{absolute[1]}"
            draw.text((left + 2, top + 1), coord_label, fill=(255, 255, 255, 220), font=font)

    for sprite_x, sprite_y in snapshot.sprite_positions:
        local = snapshot.absolute_to_local(sprite_x, sprite_y)
        if local is None:
            continue
        left = int(local[0] * tile_width)
        top = int(local[1] * tile_height)
        right = int((local[0] + 1) * tile_width)
        bottom = int((local[1] + 1) * tile_height)
        draw.rectangle((left + 3, top + 3, right - 3, bottom - 3), fill=(255, 174, 66, 180))

    player_left = int(4 * tile_width)
    player_top = int(4 * tile_height)
    player_right = int(5 * tile_width)
    player_bottom = int(5 * tile_height)
    draw.rectangle(
        (player_left + 1, player_top + 1, player_right - 1, player_bottom - 1),
        outline=(55, 208, 255, 255),
        width=3,
    )
    draw.text((player_left + 4, player_top + 5), "P", fill=(55, 208, 255, 255), font=font)

    if goal is not None:
        goal_local = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_local is not None:
            left = int(goal_local[0] * tile_width)
            top = int(goal_local[1] * tile_height)
            right = int((goal_local[0] + 1) * tile_width)
            bottom = int((goal_local[1] + 1) * tile_height)
            draw.rectangle(
                (left + 1, top + 1, right - 1, bottom - 1), outline=(255, 214, 10, 255), width=3
            )
            draw.text((left + 4, top + 5), "G", fill=(255, 214, 10, 255), font=font)

    interaction = snapshot.interaction or {}
    target_coord = interaction.get("target_coord") or {}
    if target_coord.get("x") is not None and target_coord.get("y") is not None:
        local = snapshot.absolute_to_local(int(target_coord["x"]), int(target_coord["y"]))
        if local is not None:
            left = int(local[0] * tile_width)
            top = int(local[1] * tile_height)
            right = int((local[0] + 1) * tile_width)
            bottom = int((local[1] + 1) * tile_height)
            draw.ellipse(
                (left + 4, top + 4, right - 4, bottom - 4), outline=(255, 125, 0, 255), width=3
            )

    legend_height = 38
    draw.rectangle((0, 0, canvas.width, legend_height), fill=(12, 17, 26, 210))
    title = snapshot.map_name
    pos = snapshot.player_position
    subtitle = (
        f"({pos[0]}, {pos[1]}) facing {snapshot.facing} | "
        f"moves: {', '.join(snapshot.valid_moves) or 'none'}"
    )
    objective_line = objective["title"] if objective else "No objective"
    draw.text((6, 4), title, fill=(255, 255, 255, 255), font=font)
    draw.text((6, 16), subtitle, fill=(165, 180, 196, 255), font=font)
    draw.text((6, 28), f"Objective: {objective_line}", fill=(255, 214, 10, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


class ObjectiveEngine:
    """Deterministic Red-first objective progression through Brock."""

    def __init__(self) -> None:
        self.pack = load_red_objective_pack()
        self.by_id = {item["id"]: item for item in self.pack}

    def _current_objective_id(self, state: JsonDict) -> str:
        flags = state.get("flags") or {}
        player = state.get("player") or {}
        battle = state.get("battle") or {}
        map_name = (state.get("map") or {}).get("map_name") or ""
        badge_count = player.get("badge_count", flags.get("badge_count", 0)) or 0
        has_pokedex = bool(flags.get("has_pokedex"))
        has_oaks_parcel = bool(flags.get("has_oaks_parcel"))
        in_battle = bool(battle.get("in_battle"))

        forest_maps = {
            "Route 2",
            "Viridian Forest Gate (S)",
            "Viridian Forest",
            "Viridian Forest Gate (N)",
        }
        if badge_count >= 1:
            return "phase_complete_boulder_badge"
        if map_name == "Pewter Gym" or (map_name == "Pewter City" and in_battle):
            return "defeat_brock"
        if map_name == "Pewter City":
            return "reach_pewter_gym"
        if has_pokedex and map_name in forest_maps:
            return "cross_viridian_forest"
        if has_pokedex:
            return "head_to_viridian_forest"
        if has_oaks_parcel:
            return "return_oaks_parcel"
        if map_name in {"Viridian City", "Viridian Mart", "Route 1"}:
            return "get_oaks_parcel"
        return "leave_pallet_and_get_starter"

    def evaluate(self, state: JsonDict) -> JsonDict:
        current_id = self._current_objective_id(state)
        current_index = next(
            (index for index, item in enumerate(self.pack) if item["id"] == current_id),
            0,
        )
        total_steps = max(len(self.pack) - 1, 1)
        progress_percent = min(100, int((current_index / total_steps) * 100))
        objectives: list[JsonDict] = []
        current_objective: Optional[JsonDict] = None

        for index, item in enumerate(self.pack):
            completed = index < current_index
            current = item["id"] == current_id
            status = "completed" if completed else "current" if current else "pending"
            progress = (
                100
                if completed
                else progress_percent
                if current
                else int((index / total_steps) * 100)
            )
            record = ObjectiveRecord(
                id=item["id"],
                title=item["title"],
                summary=item["summary"],
                completion_predicate=item["completion_predicate"],
                failure_hints=item.get("failure_hints", []),
                save_recommendation=item.get("save_recommendation", ""),
                route_hint=item.get("route_hint", ""),
                priority=index + 1,
                current=current,
                completed=completed,
                status=status,
                progress_percent=progress,
            ).to_dict()
            objectives.append(record)
            if current:
                current_objective = record

        assert current_objective is not None
        return {
            "game": "red",
            "current": current_objective,
            "objectives": objectives,
            "progress_percent": progress_percent,
            "phase_complete": current_id == "phase_complete_boulder_badge",
        }


class AgentRuntime:
    """Owns workspace artifacts, telemetry history, and deterministic assist logic."""

    def __init__(
        self,
        *,
        data_dir: Path,
        workspace_dir: Path,
        objective_engine: Optional[ObjectiveEngine] = None,
        history_limit: int = 400,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.objective_engine = objective_engine or ObjectiveEngine()
        self.history_limit = history_limit
        self.event_history: deque[JsonDict] = deque(maxlen=history_limit)
        self.recent_trajectory: deque[JsonDict] = deque(maxlen=60)
        self.latest_bundle: Optional[JsonDict] = None
        self.live_bundle: Optional[JsonDict] = None
        self.last_state: Optional[JsonDict] = None
        self.last_objective_id: Optional[str] = None
        self.last_turn_plan_hash: Optional[str] = None
        self.checkpoint_ids: set[str] = set()
        self.action_events_since_objective_change = 0
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_files()
        self._load_existing_checkpoint_ids()

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "latest_frame": self.workspace_dir / "latest_frame.png",
            "latest_frame_annotated": self.workspace_dir / "latest_frame_annotated.png",
            "latest_observation_json": self.workspace_dir / "latest_observation.json",
            "latest_observation_md": self.workspace_dir / "latest_observation.md",
            "current_objective_json": self.workspace_dir / "current_objective.json",
            "current_objective_md": self.workspace_dir / "current_objective.md",
            "turn_plan_json": self.workspace_dir / "turn_plan.json",
            "working_memory_md": self.workspace_dir / "working_memory.md",
            "checkpoints_jsonl": self.workspace_dir / "checkpoints.jsonl",
            "knowledge_graph_json": self.workspace_dir / "knowledge_graph.json",
            "recovery_saves_json": self.workspace_dir / "recovery_saves.json",
            "run_log_jsonl": self.workspace_dir / "run_log.jsonl",
        }

    def _ensure_workspace_files(self) -> None:
        for path in (
            self.artifacts["checkpoints_jsonl"],
            self.artifacts["run_log_jsonl"],
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

        defaults: dict[str, Any] = {
            "turn_plan_json": {
                "objective_id": "",
                "summary": "Update this file before each action batch.",
                "planned_actions": [],
                "fallback_actions": [],
                "notes": "",
                "updated_at": "",
            },
            "knowledge_graph_json": {
                "updated_at": "",
                "summary": {"nodes": 0, "edges": 0},
                "nodes": [],
                "edges": [],
            },
            "recovery_saves_json": {
                "updated_at": "",
                "current_recommendation": None,
                "candidates": [],
                "autosave_history": [],
            },
            "latest_observation_json": {},
            "current_objective_json": {},
        }

        for key, payload in defaults.items():
            path = self.artifacts[key]
            if not path.exists():
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if not self.artifacts["working_memory_md"].exists():
            self.artifacts["working_memory_md"].write_text(
                "# Working Memory\n\n"
                "- Update this file with short, current notes for Pi.\n"
                "- Keep notes factual: location, blockers, routes tried, battle plans.\n"
                "- Do not store canonical objective state here;\n"
                "  read current_objective.json instead.\n",
                encoding="utf-8",
            )
        if not self.artifacts["current_objective_md"].exists():
            self.artifacts["current_objective_md"].write_text(
                "Objective state will appear here after the first observation.\n",
                encoding="utf-8",
            )
        if not self.artifacts["latest_observation_md"].exists():
            self.artifacts["latest_observation_md"].write_text(
                "Observation summary will appear here after the first observation.\n",
                encoding="utf-8",
            )

    def _load_existing_checkpoint_ids(self) -> None:
        path = self.artifacts["checkpoints_jsonl"]
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            checkpoint_id = record.get("id")
            if checkpoint_id:
                self.checkpoint_ids.add(checkpoint_id)

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

    def load_turn_plan(self) -> JsonDict:
        return self._read_json(
            self.artifacts["turn_plan_json"],
            {
                "objective_id": "",
                "summary": "",
                "planned_actions": [],
                "fallback_actions": [],
                "notes": "",
                "updated_at": "",
            },
        )

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

    def _candidate_checkpoints(
        self, state: JsonDict, objective: JsonDict
    ) -> list[CheckpointRecord]:
        flags = state.get("flags") or {}
        map_name = (state.get("map") or {}).get("map_name") or "Unknown"
        current_objective_id = objective["current"]["id"]
        records: list[CheckpointRecord] = []

        definitions = [
            (
                bool(flags.get("has_oaks_parcel")),
                "oak_parcel_obtained",
                "Oak's Parcel Obtained",
                "Oak's Parcel is now in the bag.",
            ),
            (
                bool(flags.get("has_pokedex")),
                "pokedex_received",
                "Pokedex Received",
                "Oak's parcel was delivered and the Pokedex was received.",
            ),
            (
                map_name == "Viridian Forest",
                "entered_viridian_forest",
                "Entered Viridian Forest",
                "Reached the forest section of the Brock route.",
            ),
            (
                map_name == "Pewter City",
                "arrived_pewter_city",
                "Arrived In Pewter City",
                "Made it through the forest to Pewter City.",
            ),
            (
                map_name == "Pewter Gym",
                "entered_pewter_gym",
                "Entered Pewter Gym",
                "Ready to challenge Brock.",
            ),
            (
                (flags.get("badge_count") or 0) >= 1,
                "boulder_badge_earned",
                "Boulder Badge Earned",
                "Phase 1 objective pack is complete.",
            ),
        ]

        for condition, checkpoint_id, title, summary in definitions:
            if not condition or checkpoint_id in self.checkpoint_ids:
                continue
            records.append(
                CheckpointRecord(
                    id=checkpoint_id,
                    title=title,
                    summary=summary,
                    objective_id=current_objective_id,
                    map_name=map_name,
                    created_at=utc_now(),
                    metadata={"source": "deterministic_progress"},
                )
            )
        return records

    def _persist_checkpoints(self, checkpoints: Iterable[CheckpointRecord]) -> list[JsonDict]:
        emitted: list[JsonDict] = []
        for checkpoint in checkpoints:
            record = checkpoint.to_dict()
            self._append_jsonl(self.artifacts["checkpoints_jsonl"], record)
            self.checkpoint_ids.add(checkpoint.id)
            emitted.append(record)
        return emitted

    def _build_knowledge_graph(
        self,
        *,
        state: JsonDict,
        navigation: Optional[JsonDict],
        objective: JsonDict,
        navigation_store: Optional[NavigationStore],
    ) -> JsonDict:
        existing = self._read_json(
            self.artifacts["knowledge_graph_json"],
            {
                "updated_at": "",
                "summary": {"nodes": 0, "edges": 0},
                "nodes": [],
                "edges": [],
            },
        )
        nodes_by_id = {
            node["id"]: node
            for node in existing.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        edges_by_key = {
            (edge.get("source"), edge.get("target"), edge.get("type")): edge
            for edge in existing.get("edges", [])
            if isinstance(edge, dict)
        }

        map_info = state.get("map") or {}
        player = state.get("player") or {}
        current_location_id = (
            f"location:{map_info.get('map_id')}:{_slugify(map_info.get('map_name', 'unknown'))}"
        )
        nodes_by_id[current_location_id] = {
            "id": current_location_id,
            "type": "location",
            "label": map_info.get("map_name", "Unknown"),
            "current": True,
            "position": player.get("position"),
        }

        for location in (
            navigation_store.location_maps.values() if navigation_store is not None else []
        ):
            location_id = f"location:{location.map_id}:{_slugify(location.map_name)}"
            node = nodes_by_id.get(location_id, {})
            node.update(
                {
                    "id": location_id,
                    "type": "location",
                    "label": location.map_name,
                    "known_tiles": len(location.tiles),
                    "passable_tiles": location.passable_count(),
                    "blocked_tiles": location.blocked_count(),
                    "current": location_id == current_location_id,
                }
            )
            nodes_by_id[location_id] = node

        current_objective = objective["current"]
        objective_id = f"objective:{current_objective['id']}"
        nodes_by_id[objective_id] = {
            "id": objective_id,
            "type": "objective",
            "label": current_objective["title"],
            "status": current_objective["status"],
            "progress_percent": objective["progress_percent"],
        }
        edges_by_key[(current_location_id, objective_id, "objective_context")] = {
            "source": current_location_id,
            "target": objective_id,
            "type": "objective_context",
        }

        flags = state.get("flags") or {}
        for key, label in (
            ("has_pokedex", "Has Pokedex"),
            ("has_oaks_parcel", "Has Oak's Parcel"),
        ):
            if not flags.get(key):
                continue
            node_id = f"fact:{key}"
            nodes_by_id[node_id] = {
                "id": node_id,
                "type": "fact",
                "label": label,
                "value": True,
            }
            edges_by_key[(current_location_id, node_id, "fact_seen_here")] = {
                "source": current_location_id,
                "target": node_id,
                "type": "fact_seen_here",
            }

        if navigation:
            snapshot = navigation.get("snapshot") or {}
            interaction = snapshot.get("interaction") or {}
            if interaction.get("target_coord"):
                target = interaction["target_coord"]
                node_id = (
                    f"interaction:{interaction.get('kind', 'unknown')}:"
                    f"{target.get('x')}:{target.get('y')}"
                )
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "type": "interaction",
                    "label": interaction.get("kind", "unknown"),
                    "source": interaction.get("source"),
                    "target_coord": target,
                    "reason": interaction.get("reason"),
                }
                edges_by_key[(current_location_id, node_id, "visible_interaction")] = {
                    "source": current_location_id,
                    "target": node_id,
                    "type": "visible_interaction",
                }

            for warp in snapshot.get("warps") or []:
                warp_id = f"warp:{map_info.get('map_id')}:{warp.get('x')}:{warp.get('y')}"
                nodes_by_id[warp_id] = {
                    "id": warp_id,
                    "type": "exit",
                    "label": f"Warp at ({warp.get('x')}, {warp.get('y')})",
                    "target_map_id": warp.get("target_map_id"),
                }
                edges_by_key[(current_location_id, warp_id, "contains_exit")] = {
                    "source": current_location_id,
                    "target": warp_id,
                    "type": "contains_exit",
                }

        graph = {
            "updated_at": utc_now(),
            "summary": {
                "nodes": len(nodes_by_id),
                "edges": len(edges_by_key),
                "current_location": map_info.get("map_name"),
                "current_objective": current_objective["title"],
            },
            "nodes": sorted(nodes_by_id.values(), key=lambda node: node["id"]),
            "edges": sorted(
                edges_by_key.values(),
                key=lambda edge: (edge["source"], edge["type"], edge["target"]),
            ),
        }
        self._write_json(self.artifacts["knowledge_graph_json"], graph)
        return graph

    def _maybe_auto_save(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        objective: JsonDict,
        state_delta: JsonDict,
        checkpoints: list[JsonDict],
        requested_actions: Optional[list[str]],
        source: str,
    ) -> list[JsonDict]:
        triggers: list[str] = []
        if state_delta.get("fields", {}).get("map"):
            triggers.append("map_transition")
        if objective["current"]["id"] != self.last_objective_id:
            triggers.append("objective_change")
        if checkpoints:
            triggers.append("checkpoint")
        if source in {"action", "navigation"} and state.get("battle", {}).get("in_battle"):
            triggers.append("battle_entry")

        if not triggers:
            return []

        saves_dir = self.data_dir / "saves"
        saves_dir.mkdir(parents=True, exist_ok=True)
        created: list[JsonDict] = []
        current_map = (state.get("map") or {}).get("map_name", "unknown")
        for trigger in triggers[:2]:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"auto__{stamp}__{_slugify(trigger)}__{_slugify(current_map)}"
            path = saves_dir / f"{name}.state"
            if path.exists():
                continue
            emulator.save_state(str(path))
            created.append(
                {
                    "name": name,
                    "path": str(path),
                    "reason": trigger,
                    "source": "auto",
                    "notes": [
                        f"source={source}",
                        f"objective={objective['current']['id']}",
                        f"actions={','.join(requested_actions or []) or 'none'}",
                    ],
                }
            )
        return created

    def _build_recovery(self, recent_auto_saves: list[JsonDict]) -> JsonDict:
        current = self._read_json(
            self.artifacts["recovery_saves_json"],
            {
                "updated_at": "",
                "current_recommendation": None,
                "candidates": [],
                "autosave_history": [],
            },
        )
        autosave_history = list(current.get("autosave_history", []))
        autosave_history.extend(recent_auto_saves)
        autosave_history = autosave_history[-50:]

        saves_dir = self.data_dir / "saves"
        candidates: list[RecoveryCandidate] = []
        known_notes = {
            entry.get("name"): entry
            for entry in autosave_history
            if isinstance(entry, dict) and entry.get("name")
        }
        for path in sorted(saves_dir.glob("*.state")):
            meta = known_notes.get(path.stem, {})
            source = meta.get("source", "manual")
            reason = meta.get("reason", "manual_save")
            score = 40
            if reason == "checkpoint":
                score = 95
            elif reason == "objective_change":
                score = 90
            elif reason == "map_transition":
                score = 75
            elif reason == "battle_entry":
                score = 65
            elif source == "manual":
                score = 80
            candidates.append(
                RecoveryCandidate(
                    name=path.stem,
                    path=str(path),
                    reason=reason,
                    score=score,
                    modified_at=datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    source=source,
                    notes=list(meta.get("notes", [])),
                )
            )

        ranked = sorted(
            candidates,
            key=lambda entry: (entry.score, entry.modified_at, entry.name),
            reverse=True,
        )
        payload = {
            "updated_at": utc_now(),
            "current_recommendation": ranked[0].to_dict() if ranked else None,
            "candidates": [entry.to_dict() for entry in ranked[:12]],
            "autosave_history": autosave_history,
        }
        self._write_json(self.artifacts["recovery_saves_json"], payload)
        return payload

    def _detect_stuck(
        self,
        *,
        state: JsonDict,
        objective: JsonDict,
        source: str,
        requested_actions: Optional[list[str]],
    ) -> JsonDict:
        player = state.get("player") or {}
        position = player.get("position") or {}
        signature = {
            "map_name": (state.get("map") or {}).get("map_name"),
            "position": position,
            "dialog_active": bool(
                state.get("dialog_active") or (state.get("dialog") or {}).get("active")
            ),
            "objective_id": objective["current"]["id"],
            "source": source,
            "actions": requested_actions or [],
        }
        self.recent_trajectory.append(signature)

        recent = list(self.recent_trajectory)[-6:]
        no_movement_loop = False
        dialog_loop = False
        objective_timeout = False

        if len(recent) >= 4:
            locations = {
                (
                    item.get("map_name"),
                    json.dumps(item.get("position"), sort_keys=True),
                )
                for item in recent[-4:]
            }
            if len(locations) == 1 and any(item.get("actions") for item in recent[-4:]):
                no_movement_loop = True
            if no_movement_loop and all(item.get("dialog_active") for item in recent[-4:]):
                dialog_loop = True

        if objective["current"]["id"] == self.last_objective_id and source in {
            "action",
            "navigation",
        }:
            self.action_events_since_objective_change += 1
        else:
            self.action_events_since_objective_change = 0
        if self.action_events_since_objective_change >= 12:
            objective_timeout = True

        level = "clear"
        reason = "No stuck pattern detected."
        recommended: list[str] = []
        if dialog_loop:
            level = "warning"
            reason = (
                "Dialog loop detected: repeated actions with the same position and active dialog."
            )
            recommended = ["press_a", "press_b", "re-check latest_frame.png if text is ambiguous"]
        elif no_movement_loop:
            level = "warning"
            reason = "No-movement loop detected: repeated actions without position or map change."
            recommended = [
                "inspect valid_moves",
                "use a shorter action batch",
                "consider a recovery save",
            ]
        if objective_timeout:
            level = "danger" if level == "warning" else "warning"
            reason = "Current objective has seen many action turns without progress."
            recommended = recommended or [
                "review turn_plan.json",
                "reload the top recovery candidate",
            ]

        return {
            "level": level,
            "reason": reason,
            "recommended_actions": recommended,
            "objective_action_count": self.action_events_since_objective_change,
        }

    def _objective_markdown(self, objective: JsonDict) -> str:
        current = objective["current"]
        lines = [
            "# Current Objective",
            "",
            f"- Title: {current['title']}",
            f"- Summary: {current['summary']}",
            f"- Completion predicate: {current['completion_predicate']}",
            f"- Save recommendation: {current['save_recommendation']}",
            f"- Route hint: {current['route_hint']}",
            f"- Progress: {objective['progress_percent']}%",
            "",
            "## Failure Hints",
        ]
        for hint in current.get("failure_hints", []):
            lines.append(f"- {hint}")
        return "\n".join(lines) + "\n"

    def _observation_markdown(
        self,
        *,
        bundle: JsonDict,
    ) -> str:
        objective = bundle["objective"]["current"]
        state = bundle["state"]
        navigation = bundle.get("navigation") or {}
        snapshot = navigation.get("snapshot") or {}
        turn_plan = bundle["turn_plan"]
        feedback = bundle["recent_action"]
        movement_guidance = bundle["movement_guidance"]
        stuck = bundle["stuck"]
        recovery = bundle["recovery"]
        state_delta_summary = (bundle.get("state_delta") or {}).get("summary") or [
            "Live frame sync only. Run /agent/observe for refreshed deltas."
        ]
        lines = [
            "# Vision-First Turn Brief",
            "",
            f"Read first: `{bundle['artifacts']['latest_frame_annotated']}`",
            f"Fallback frame: `{bundle['artifacts']['latest_frame']}`",
            "Mandatory: inspect the annotated frame before choosing actions.",
            "Do not infer terrain from ASCII alone.",
            "",
            f"Objective: {objective['title']}",
            f"Objective summary: {objective['summary']}",
            f"Route hint: {objective['route_hint']}",
            "",
            f"UI mode: {bundle['screen_text']['ui_mode']}",
            f"Map: {(state.get('map') or {}).get('map_name', 'Unknown')}",
            f"Position: {(state.get('player') or {}).get('position')}",
            f"Facing: {(state.get('player') or {}).get('facing')}",
            f"Valid moves: {', '.join(snapshot.get('valid_moves', [])) or 'none'}",
            f"Interaction probe: {(snapshot.get('interaction') or {}).get('reason', 'none')}",
            "ASCII legend: P=player G=goal S=sprite .=passable #=blocked ?=unknown",
            f"Movement guidance: {movement_guidance.get('summary', 'none')}",
            "",
            f"What changed: {' '.join(state_delta_summary[:3])}",
            f"Recent action result: {feedback.get('summary', 'No recent action result.')}",
            f"Planned action batch: {', '.join(turn_plan.get('planned_actions', [])) or 'not set'}",
            f"Fallback: {', '.join(turn_plan.get('fallback_actions', [])) or 'not set'}",
            "",
            f"Screen text: {bundle['screen_text']['text']}",
            "",
            f"Recovery recommendation: "
            f"{(recovery.get('current_recommendation') or {}).get('name', 'none')}",
            f"Stuck signal: {stuck['level']} - {stuck['reason']}",
            "",
            "Navigation notes:",
        ]
        for note in movement_guidance.get("notes", []):
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n"

    def _artifact_payload(self) -> JsonDict:
        return {
            "latest_frame": str(self.artifacts["latest_frame"]),
            "latest_frame_annotated": str(self.artifacts["latest_frame_annotated"]),
            "latest_observation_json": str(self.artifacts["latest_observation_json"]),
            "latest_observation_md": str(self.artifacts["latest_observation_md"]),
            "current_objective_json": str(self.artifacts["current_objective_json"]),
            "current_objective_md": str(self.artifacts["current_objective_md"]),
            "turn_plan_json": str(self.artifacts["turn_plan_json"]),
            "working_memory_md": str(self.artifacts["working_memory_md"]),
            "checkpoints_jsonl": str(self.artifacts["checkpoints_jsonl"]),
            "knowledge_graph_json": str(self.artifacts["knowledge_graph_json"]),
            "recovery_saves_json": str(self.artifacts["recovery_saves_json"]),
            "run_log_jsonl": str(self.artifacts["run_log_jsonl"]),
        }

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
                        map_dimensions=snapshot_payload.get("map_dimensions"),
                        interaction=snapshot_payload.get("interaction"),
                    )
                except Exception:  # noqa: BLE001
                    snapshot = None
        return snapshot

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
    ) -> None:
        self.artifacts["latest_frame"].parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        screen.save(buf, format="PNG")
        _atomic_write_bytes(self.artifacts["latest_frame"], buf.getvalue())
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        _atomic_write_bytes(self.artifacts["latest_frame_annotated"], buf.getvalue())

    def sync_live_view(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        navigation_store: Optional[NavigationStore],
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)
        snapshot = self._snapshot_from_navigation_payload(navigation)
        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=None,
        )
        self._write_frame_artifacts(screen=screen, annotated=annotated)

        previous_bundle = self.live_bundle or self.latest_bundle or {}
        previous_screen_text = previous_bundle.get("screen_text") or {}
        dialog_active = bool(
            state.get("dialog_active") or (state.get("dialog") or {}).get("active")
        )
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
            preserved_text = "Live frame sync active. Run /agent/observe for refreshed OCR."

        live_bundle = {
            "generated_at": utc_now(),
            "reason": "realtime_live_sync",
            "source": "live_sync",
            "artifacts": self._artifact_payload(),
            "state": state,
            "navigation": navigation,
            "screen_text": {
                "text": preserved_text,
                "source": preserved_source,
                "ui_mode": classify_ui_mode(state),
                "dialog_active": dialog_active,
                "note": "Live sync updates visuals and core state between observations.",
            },
            "objective": current_objective,
            "turn_plan": self.load_turn_plan(),
            "recent_action": previous_bundle.get("recent_action") or {},
            "movement_guidance": build_movement_guidance(
                state=state,
                snapshot=snapshot,
                navigation_store=navigation_store,
                objective=current_objective,
            ),
            "state_delta": previous_bundle.get("state_delta")
            or {
                "changed": False,
                "summary": ["Live frame sync only. Run /agent/observe for refreshed deltas."],
                "movement": None,
            },
            "checkpoints": previous_bundle.get("checkpoints")
            or self._tail_jsonl(self.artifacts["checkpoints_jsonl"], 20),
            "knowledge_graph": self._read_json(
                self.artifacts["knowledge_graph_json"],
                previous_bundle.get("knowledge_graph") or {},
            ),
            "recovery": self._read_json(
                self.artifacts["recovery_saves_json"],
                previous_bundle.get("recovery") or {},
            ),
            "stuck": previous_bundle.get("stuck")
            or {
                "level": "clear",
                "reason": "No stuck signal recorded yet.",
                "recommended_actions": [],
                "objective_action_count": 0,
            },
            "workspace_dir": str(self.workspace_dir),
        }

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        _atomic_write_text(
            self.artifacts["current_objective_md"],
            self._objective_markdown(current_objective),
        )
        self._write_json(
            self.artifacts["latest_observation_json"],
            self._compact_observation_payload(live_bundle),
        )
        _atomic_write_text(
            self.artifacts["latest_observation_md"],
            self._observation_markdown(bundle=live_bundle),
        )

        self.live_bundle = live_bundle
        return {
            "generated_at": live_bundle["generated_at"],
            "source": live_bundle["source"],
            "artifacts": live_bundle["artifacts"],
            "screen_text": live_bundle["screen_text"],
        }

    def _compact_observation_payload(self, bundle: JsonDict) -> JsonDict:
        navigation = bundle.get("navigation") or {}
        snapshot = navigation.get("snapshot") or {}
        location_map = navigation.get("location_map") or {}
        objective = bundle.get("objective") or {}
        recovery = bundle.get("recovery") or {}
        artifacts = bundle.get("artifacts") or {}
        current_objective = objective.get("current") or {}
        recent_action = bundle.get("recent_action") or {}
        movement_guidance = bundle.get("movement_guidance") or {}
        state_delta = bundle.get("state_delta") or {}
        candidate_route = movement_guidance.get("candidate_route") or {}
        live_ascii = _truncate_text_block(snapshot.get("ascii"), 900)
        explored_ascii = _truncate_text_block(location_map.get("ascii"), 1400)
        return {
            "generated_at": bundle.get("generated_at"),
            "reason": bundle.get("reason"),
            "source": bundle.get("source"),
            "artifacts": {
                "latest_frame": artifacts.get("latest_frame"),
                "latest_frame_annotated": artifacts.get("latest_frame_annotated"),
                "latest_observation_md": artifacts.get("latest_observation_md"),
                "current_objective_json": artifacts.get("current_objective_json"),
                "current_objective_md": artifacts.get("current_objective_md"),
                "turn_plan_json": artifacts.get("turn_plan_json"),
                "working_memory_md": artifacts.get("working_memory_md"),
                "recovery_saves_json": artifacts.get("recovery_saves_json"),
            },
            "state": extract_key_state(bundle.get("state")),
            "screen_text": {
                "text": _truncate_text_block((bundle.get("screen_text") or {}).get("text"), 420),
                "source": (bundle.get("screen_text") or {}).get("source"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "dialog_active": (bundle.get("screen_text") or {}).get("dialog_active"),
            },
            "objective": {
                "current": {
                    "id": current_objective.get("id"),
                    "title": current_objective.get("title"),
                    "summary": current_objective.get("summary"),
                    "completion_predicate": current_objective.get("completion_predicate"),
                    "save_recommendation": current_objective.get("save_recommendation"),
                    "route_hint": current_objective.get("route_hint"),
                    "failure_hints": (current_objective.get("failure_hints") or [])[:4],
                    "progress_percent": current_objective.get("progress_percent"),
                },
                "progress_percent": objective.get("progress_percent"),
            },
            "turn_plan": {
                "objective_id": (bundle.get("turn_plan") or {}).get("objective_id"),
                "summary": (bundle.get("turn_plan") or {}).get("summary"),
                "planned_actions": ((bundle.get("turn_plan") or {}).get("planned_actions") or [])[
                    :6
                ],
                "fallback_actions": ((bundle.get("turn_plan") or {}).get("fallback_actions") or [])[
                    :6
                ],
                "notes": _truncate_text_block((bundle.get("turn_plan") or {}).get("notes"), 260),
                "updated_at": (bundle.get("turn_plan") or {}).get("updated_at"),
            },
            "recent_action": {
                "source": recent_action.get("source"),
                "summary": recent_action.get("summary"),
                "notes": (recent_action.get("notes") or [])[:4],
                "tags": recent_action.get("tags"),
            },
            "movement_guidance": {
                "summary": movement_guidance.get("summary"),
                "notes": (movement_guidance.get("notes") or [])[:5],
                "preferred_direction": movement_guidance.get("preferred_direction"),
                "candidate_route": (
                    {
                        "direction": candidate_route.get("direction"),
                        "target": candidate_route.get("target"),
                        "steps": candidate_route.get("steps"),
                        "actions": (candidate_route.get("actions") or [])[:6],
                    }
                    if candidate_route
                    else None
                ),
            },
            "state_delta": {
                "changed": state_delta.get("changed"),
                "summary": (state_delta.get("summary") or [])[:4],
                "movement": state_delta.get("movement"),
            },
            "navigation": {
                "valid_moves": snapshot.get("valid_moves", []),
                "interaction": snapshot.get("interaction"),
                "live_ascii": live_ascii,
                "explored_ascii": explored_ascii,
                "ascii_note": (
                    "ASCII is symbolic only and may be truncated. Use the annotated frame first."
                ),
                "window_top_left": snapshot.get("window_top_left"),
                "window_size": snapshot.get("window_size"),
                "bounds": location_map.get("bounds"),
            },
            "checkpoints": [
                {
                    "id": checkpoint.get("id"),
                    "title": checkpoint.get("title"),
                    "created_at": checkpoint.get("created_at"),
                    "objective_id": checkpoint.get("objective_id"),
                    "map_name": checkpoint.get("map_name"),
                }
                for checkpoint in (bundle.get("checkpoints") or [])[-10:]
            ],
            "knowledge_graph_summary": (bundle.get("knowledge_graph") or {}).get("summary"),
            "recovery": {
                "updated_at": recovery.get("updated_at"),
                "current_recommendation": recovery.get("current_recommendation"),
                "candidates": (recovery.get("candidates") or [])[:3],
            },
            "stuck": bundle.get("stuck"),
        }

    def refresh(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        navigation_store: Optional[NavigationStore],
        reason: str,
        source: str,
        requested_actions: Optional[list[str]] = None,
        navigation_plan: Optional[JsonDict] = None,
        navigation_execution: Optional[JsonDict] = None,
        explicit_save: Optional[JsonDict] = None,
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)

        goal = None
        if navigation_execution and navigation_execution.get("target"):
            goal_data = navigation_execution["target"]
            if goal_data.get("x") is not None and goal_data.get("y") is not None:
                goal = (int(goal_data["x"]), int(goal_data["y"]))

        snapshot = self._snapshot_from_navigation_payload(navigation)

        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=goal,
        )
        screen_text = extract_screen_text(screen, state)
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
        movement_guidance = build_movement_guidance(
            state=state,
            snapshot=snapshot,
            navigation_store=navigation_store,
            objective=current_objective,
        )
        turn_plan = self.load_turn_plan()
        turn_plan_hash = json.dumps(turn_plan, sort_keys=True)
        checkpoints = self._persist_checkpoints(
            self._candidate_checkpoints(state, current_objective)
        )
        knowledge_graph = self._build_knowledge_graph(
            state=state,
            navigation=navigation,
            objective=current_objective,
            navigation_store=navigation_store,
        )
        auto_saves = self._maybe_auto_save(
            emulator=emulator,
            state=state,
            objective=current_objective,
            state_delta=state_delta,
            checkpoints=checkpoints,
            requested_actions=requested_actions,
            source=source,
        )
        if explicit_save:
            auto_saves.append(explicit_save)
        recovery = self._build_recovery(auto_saves)
        stuck = self._detect_stuck(
            state=state,
            objective=current_objective,
            source=source,
            requested_actions=requested_actions,
        )

        self._write_frame_artifacts(screen=screen, annotated=annotated)

        bundle = {
            "generated_at": utc_now(),
            "reason": reason,
            "source": source,
            "artifacts": self._artifact_payload(),
            "state": state,
            "navigation": navigation,
            "screen_text": screen_text,
            "objective": current_objective,
            "turn_plan": turn_plan,
            "recent_action": action_feedback,
            "movement_guidance": movement_guidance,
            "state_delta": state_delta,
            "checkpoints": self._tail_jsonl(self.artifacts["checkpoints_jsonl"], 20),
            "knowledge_graph": knowledge_graph,
            "recovery": recovery,
            "stuck": stuck,
            "workspace_dir": str(self.workspace_dir),
        }

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        _atomic_write_text(
            self.artifacts["current_objective_md"],
            self._objective_markdown(current_objective),
        )
        self._write_json(
            self.artifacts["latest_observation_json"],
            self._compact_observation_payload(bundle),
        )
        _atomic_write_text(
            self.artifacts["latest_observation_md"],
            self._observation_markdown(bundle=bundle),
        )

        emitted_events: list[JsonDict] = []
        emitted_events.append(
            self._record_event(
                "observe",
                {
                    "reason": reason,
                    "source": source,
                    "objective_id": current_objective["current"]["id"],
                    "summary": action_feedback["summary"],
                },
            )
        )
        emitted_events.append(
            self._record_event(
                "ocr",
                {
                    "source": screen_text["source"],
                    "text": screen_text["text"],
                },
            )
        )

        if current_objective["current"]["id"] != self.last_objective_id:
            emitted_events.append(
                self._record_event(
                    "objective",
                    {
                        "objective": current_objective["current"],
                        "progress_percent": current_objective["progress_percent"],
                    },
                )
            )

        for checkpoint in checkpoints:
            emitted_events.append(self._record_event("checkpoint", checkpoint))

        emitted_events.append(
            self._record_event(
                "knowledge",
                {
                    "summary": knowledge_graph["summary"],
                },
            )
        )

        for save_event in auto_saves:
            emitted_events.append(self._record_event("save", save_event))

        emitted_events.append(
            self._record_event(
                "recovery",
                {
                    "current_recommendation": recovery.get("current_recommendation"),
                    "candidate_count": len(recovery.get("candidates", [])),
                },
            )
        )

        if stuck["level"] != "clear":
            emitted_events.append(self._record_event("stuck", stuck))

        if turn_plan_hash != self.last_turn_plan_hash:
            emitted_events.append(
                self._record_event(
                    "turn_plan_updated",
                    {
                        "turn_plan": turn_plan,
                    },
                )
            )
            self.last_turn_plan_hash = turn_plan_hash

        self.latest_bundle = bundle
        self.live_bundle = bundle
        self.last_state = state
        self.last_objective_id = current_objective["current"]["id"]
        return {
            "bundle": bundle,
            "events": emitted_events,
        }

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
        current_objective = (bundle.get("objective") or {}).get("current") or {}
        turn_plan = bundle.get("turn_plan") or {}
        recovery = bundle.get("recovery") or {}
        knowledge_graph = bundle.get("knowledge_graph") or {}
        return {
            "generated_at": bundle.get("generated_at"),
            "visuals": {
                "raw_frame_path": (bundle.get("artifacts") or {}).get("latest_frame"),
                "annotated_frame_path": (bundle.get("artifacts") or {}).get(
                    "latest_frame_annotated"
                ),
                "frame_timestamp": bundle.get("generated_at"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "screen_text": bundle.get("screen_text"),
            },
            "agent_intent": {
                "objective": current_objective,
                "turn_plan": turn_plan,
                "recent_action": bundle.get("recent_action"),
                "movement_guidance": bundle.get("movement_guidance"),
                "state_delta": bundle.get("state_delta"),
            },
            "world_state": {
                "map": state.get("map"),
                "player": state.get("player"),
                "party": state.get("party"),
                "battle": state.get("battle"),
                "dialog": state.get("dialog"),
                "interaction": state.get("interaction")
                or (navigation.get("snapshot") or {}).get("interaction"),
                "valid_moves": (navigation.get("snapshot") or {}).get("valid_moves", []),
                "live_ascii": (navigation.get("snapshot") or {}).get("ascii"),
                "explored_ascii": (navigation.get("location_map") or {}).get("ascii"),
                "navigation": navigation,
            },
            "memory_and_progress": {
                "progress_percent": (bundle.get("objective") or {}).get("progress_percent"),
                "checkpoints": bundle.get("checkpoints"),
                "knowledge_graph_summary": knowledge_graph.get("summary"),
                "recovery": recovery,
                "stuck": bundle.get("stuck"),
                "workspace": {
                    "workspace_dir": bundle.get("workspace_dir"),
                    "turn_plan_json": (bundle.get("artifacts") or {}).get("turn_plan_json"),
                    "working_memory_md": (bundle.get("artifacts") or {}).get("working_memory_md"),
                    "latest_observation_md": (bundle.get("artifacts") or {}).get(
                        "latest_observation_md"
                    ),
                },
            },
            "timeline": self.history(80),
            "artifacts": bundle.get("artifacts") or {},
        }
