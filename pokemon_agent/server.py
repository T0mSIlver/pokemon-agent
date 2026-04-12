"""
Pokemon Agent — FastAPI Game Server

Provides HTTP + WebSocket API for controlling a Game Boy / GBA emulator
running a Pokemon ROM, reading game state, and broadcasting events.
"""

import asyncio
import base64
import io
import json
import mimetypes
import re
import time
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from pokemon_agent.agent_runtime import AgentRuntime
from pokemon_agent.harness.contracts import TurnPlanInput
from pokemon_agent.pi_supervisor import PiSupervisor

__version__ = "0.1.0"


def _guess_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(path.name)
    return ct or "application/octet-stream"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GameConfig(BaseModel):
    """Server configuration — set before startup."""

    rom_path: str
    game_type: str = "auto"  # "red", "firered", or "auto"
    port: int = 8765
    data_dir: str = "~/.pokemon-agent"
    load_state: Optional[str] = None  # Save-state name to auto-load on startup
    agent_workspace_dir: Optional[str] = None
    enable_dashboard: bool = True
    realtime: bool = True
    realtime_fps: int = 60
    live_artifact_broadcast_fps: Optional[int] = None


class ActionRequest(BaseModel):
    """Body for POST /action."""

    actions: list[str]


class SaveRequest(BaseModel):
    """Body for POST /save and POST /load."""

    name: str


class NavigationRequest(BaseModel):
    """Body for POST /navigation/path and POST /navigation/navigate."""

    x: int
    y: int
    mode: str = "auto"  # "auto", "screen", or "persistent"


class ObserveRequest(BaseModel):
    """Body for POST /agent/observe."""

    reason: str = "manual_observe"


class PiSupervisorStartRequest(BaseModel):
    """Body for POST /supervisor/start."""

    goal: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    thinking: Optional[str] = None
    auto_continue: bool = True
    max_turns: Optional[int] = None
    continue_delay_seconds: float = 1.0
    skill_path: Optional[str] = None


class PiSupervisorContinueRequest(BaseModel):
    """Body for POST /supervisor/continue."""

    pass


class AgentActRequest(BaseModel):
    """Body for POST /agent/act."""

    branch: Literal["primary", "fallback"] = "primary"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_emulator = None  # Emulator instance
_reader = None  # GameMemoryReader subclass instance
_navigation_store = None  # NavigationStore instance
_runtime: Optional[AgentRuntime] = None
_supervisor: Optional[PiSupervisor] = None
_start_time: float = 0.0
_loop: Optional[asyncio.AbstractEventLoop] = None
_dashboard_dir: Optional[Path] = None
_emulator_lock: Optional[asyncio.Lock] = None
_realtime_task: Optional[asyncio.Task] = None
_realtime_frames_per_second: int = 60
_realtime_enabled: bool = False
_realtime_ticks: int = 0
_realtime_last_tick_at: Optional[float] = None
_live_artifact_task: Optional[asyncio.Task] = None
_live_artifact_frames_per_second: int = 10
_live_artifact_last_sync_at: Optional[float] = None

# WebSocket clients
_ws_clients: Set[WebSocket] = set()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pokemon Agent Server",
    version=__version__,
    description="HTTP + WebSocket API for Pokemon emulator control",
)

# CORS — allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_game_type(rom_path: str) -> str:
    """Pick reader type based on file extension."""
    ext = Path(rom_path).suffix.lower()
    if ext in (".gb", ".gbc"):
        return "red"
    elif ext == ".gba":
        return "firered"
    raise ValueError(f"Unrecognised ROM extension: {ext}")


def _ensure_emulator():
    """Raise 503 if the emulator isn't ready."""
    if _emulator is None:
        raise HTTPException(status_code=503, detail="Emulator not initialised")


async def _run_sync(func, *args, **kwargs):
    """Run a blocking emulator call in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _run_emulator_sync(func, *args, **kwargs):
    """Run a blocking emulator call while holding the emulator lock."""
    if _emulator_lock is None:
        return await _run_sync(func, *args, **kwargs)
    async with _emulator_lock:
        return await _run_sync(func, *args, **kwargs)


async def broadcast(event: dict):
    """Send a JSON event to every connected WebSocket client."""
    dead: list[WebSocket] = []
    payload = json.dumps(event)
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _record_and_broadcast(event_type: str, payload: dict) -> dict:
    event = {"type": event_type, **payload}
    if _runtime is not None:
        event = _runtime.record_external_event(event_type, payload)
    await broadcast(event)
    return event


async def _record_existing_event_and_broadcast(event: dict) -> dict:
    event_type = str(event.get("type") or "event")
    payload = {key: value for key, value in event.items() if key not in {"type", "timestamp"}}
    return await _record_and_broadcast(event_type, payload)


def _get_state_dict() -> dict:
    """Build full game state from the memory reader."""
    from pokemon_agent.state.builder import build_game_state

    state = build_game_state(_reader, frame_count=getattr(_emulator, "frame_count", None))
    dialog = state.get("dialog")
    state["dialog_active"] = bool(isinstance(dialog, dict) and dialog.get("active"))
    try:
        snapshot = _emulator.get_navigation_snapshot(_reader)
    except NotImplementedError:
        return state
    except Exception:
        return state
    state["interaction"] = snapshot.interaction
    return state


def _dialog_is_active(state: dict) -> bool:
    if "dialog_active" in state:
        return bool(state.get("dialog_active"))
    dialog = state.get("dialog")
    return bool(isinstance(dialog, dict) and dialog.get("active"))


def _summarize_navigation_execution(
    snapshot_before,
    snapshot_after,
    plan,
    target: tuple[int, int],
) -> dict:
    start = snapshot_before.player_position
    end = snapshot_after.player_position
    moved = end != start
    reached_target = end == target
    matched_planned_final_position = plan.final_position is not None and end == plan.final_position
    started_at_target = start == target
    success = (
        reached_target
        or (plan.partial and matched_planned_final_position)
        or (started_at_target and not plan.actions)
    )

    before_distance = abs(start[0] - target[0]) + abs(start[1] - target[1])
    after_distance = abs(end[0] - target[0]) + abs(end[1] - target[1])

    if reached_target:
        status = "Reached the requested navigation target."
    elif plan.partial and matched_planned_final_position:
        status = "Reached the planned partial navigation destination."
    elif started_at_target and not plan.actions:
        status = "Already at the requested navigation target."
    elif plan.actions and not moved:
        status = "Executed the navigation plan but the player did not move."
    elif plan.actions and after_distance < before_distance:
        status = "Moved toward the target but did not reach the planned destination."
    elif not plan.actions:
        status = "No navigation actions were executed."
    else:
        status = "Navigation actions executed, but the final position did not match the plan."

    return {
        "success": success,
        "moved": moved,
        "started_at_target": started_at_target,
        "reached_target": reached_target,
        "matched_planned_final_position": matched_planned_final_position,
        "planned_final_position": (
            {"x": plan.final_position[0], "y": plan.final_position[1]}
            if plan.final_position is not None
            else None
        ),
        "actual_position": {"x": end[0], "y": end[1]},
        "target": {"x": target[0], "y": target[1]},
        "remaining_distance_before": before_distance,
        "remaining_distance_after": after_distance,
        "status": status,
    }


def _get_screenshot_bytes() -> bytes:
    """Grab the current frame as PNG bytes."""
    screen = _emulator.get_screen()  # PIL Image or numpy array
    buf = io.BytesIO()
    # If it's a numpy array, convert to PIL first
    try:
        from PIL import Image

        if not isinstance(screen, Image.Image):
            screen = Image.fromarray(screen)
        screen.save(buf, format="PNG")
    except ImportError:
        # Fallback: assume screen already has save()
        screen.save(buf, format="PNG")
    return buf.getvalue()


def _get_dashboard_static_dir() -> Optional[Path]:
    try:
        import pokemon_agent.dashboard as dashboard_mod
    except ImportError:
        return None
    dash_dir = Path(dashboard_mod.__file__).parent / "static"
    if dash_dir.is_dir() and (dash_dir / "index.html").exists():
        return dash_dir
    return None


async def _realtime_emulator_loop() -> None:
    """Advance the emulator at a fixed cadence while the server is idle."""
    global _realtime_ticks, _realtime_last_tick_at
    interval = 1.0 / max(1, _realtime_frames_per_second)
    try:
        while True:
            await asyncio.sleep(interval)
            if _emulator is None:
                continue
            await _run_emulator_sync(_emulator.tick, 1)
            _realtime_ticks += 1
            _realtime_last_tick_at = time.time()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[server] WARNING: Realtime emulator loop stopped: {exc}")


async def _live_artifact_loop() -> None:
    """Refresh live workspace artifacts while realtime emulation is running."""
    global _live_artifact_last_sync_at
    interval = 1.0 / max(1, _live_artifact_frames_per_second)
    try:
        while True:
            await asyncio.sleep(interval)
            if _emulator is None or _runtime is None:
                continue
            payload = await _run_emulator_sync(_sync_live_artifacts_sync)
            if not payload:
                continue
            _live_artifact_last_sync_at = time.time()
            artifacts = payload.get("artifacts") or {}
            generated_at = payload.get("generated_at")
            await broadcast(
                {
                    "type": "screenshot",
                    "data": {
                        "raw_frame_path": artifacts.get("latest_frame"),
                        "annotated_frame_path": artifacts.get("latest_frame_annotated"),
                        "raw_frame_url": _artifact_urls_from_paths(artifacts).get("latest_frame"),
                        "annotated_frame_url": _artifact_urls_from_paths(artifacts).get(
                            "latest_frame_annotated"
                        ),
                        "frame_timestamp": generated_at,
                        "source": payload.get("source"),
                    },
                }
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[server] WARNING: Live artifact loop stopped: {exc}")


def _server_runtime_snapshot() -> dict:
    return {
        "realtime_enabled": _realtime_enabled,
        "realtime_fps": _realtime_frames_per_second,
        "realtime_ticks": _realtime_ticks,
        "realtime_last_tick_at": _realtime_last_tick_at,
        "live_artifact_fps": _live_artifact_frames_per_second if _realtime_enabled else 0,
        "live_artifact_last_sync_at": _live_artifact_last_sync_at,
        "frame_count": getattr(_emulator, "frame_count", None),
    }


def _compact_objective_status(objective: Optional[dict]) -> Optional[dict]:
    if not objective:
        return None
    return {
        "id": objective.get("id"),
        "pack_id": objective.get("pack_id"),
        "title": objective.get("title"),
        "summary": objective.get("summary"),
        "progress_percent": objective.get("progress_percent"),
        "status": objective.get("status"),
        "route_hint": objective.get("route_hint"),
        "completion_predicate": objective.get("completion_predicate"),
        "preferred_landmark_types": objective.get("preferred_landmark_types"),
    }


def _compact_state_snapshot(state: Optional[dict]) -> dict:
    state = state or {}
    player = state.get("player") or {}
    battle = state.get("battle") or {}
    dialog = state.get("dialog") or {}
    flags = state.get("flags") or {}
    position = player.get("position") or {}
    return {
        "map_name": (state.get("map") or {}).get("map_name"),
        "map_id": (state.get("map") or {}).get("map_id"),
        "position": {
            "x": position.get("x"),
            "y": position.get("y"),
        },
        "facing": player.get("facing"),
        "dialog_active": bool(state.get("dialog_active") or dialog.get("active")),
        "battle_active": bool(battle.get("in_battle")),
        "battle_type": battle.get("type"),
        "has_pokedex": bool(flags.get("has_pokedex")),
        "has_oaks_parcel": bool(flags.get("has_oaks_parcel")),
        "badge_count": player.get("badge_count", flags.get("badge_count", 0)),
    }


def _compact_navigation_snapshot(
    navigation: Optional[dict],
    navigation_guidance: Optional[dict] = None,
) -> dict:
    navigation = navigation or {}
    navigation_guidance = navigation_guidance or {}
    snapshot = navigation.get("snapshot") or {}
    location_map = navigation.get("location_map") or {}
    return {
        "valid_moves": snapshot.get("valid_moves", []),
        "interaction": snapshot.get("interaction"),
        "live_ascii": snapshot.get("ascii"),
        "explored_ascii": location_map.get("ascii"),
        "distance_ascii": _truncate_text(navigation_guidance.get("distance_ascii"), 1800),
        "frontiers": (navigation_guidance.get("frontiers") or [])[:8],
        "landmarks": (navigation_guidance.get("landmarks") or [])[:8],
        "route_cards": (navigation_guidance.get("route_cards") or [])[:5],
        "avoidances": (navigation_guidance.get("avoidances") or [])[:5],
        "window_top_left": snapshot.get("window_top_left"),
        "window_size": snapshot.get("window_size"),
        "bounds": location_map.get("bounds"),
    }


def _compact_recovery_summary(recovery: Optional[dict]) -> dict:
    recovery = recovery or {}
    return {
        "current_recommendation": recovery.get("current_recommendation"),
        "candidate_count": len(recovery.get("candidates") or []),
    }


def _compact_supervisor_status(snapshot: Optional[dict]) -> Optional[dict]:
    if not snapshot:
        return None
    return {
        "available": snapshot.get("available"),
        "status": snapshot.get("status"),
        "status_reason": snapshot.get("status_reason"),
        "last_error": snapshot.get("last_error"),
        "session_id": snapshot.get("session_id"),
        "turns_completed": snapshot.get("turns_completed"),
        "model": snapshot.get("model"),
        "provider": snapshot.get("provider"),
        "thinking": snapshot.get("thinking"),
        "goal": snapshot.get("goal"),
    }


def _artifact_urls_from_paths(artifacts: Optional[dict]) -> dict:
    urls: dict[str, str] = {}
    for key in artifacts or {}:
        urls[key] = f"/artifacts/{key}"
    return urls


def _public_artifact_paths(artifacts: Optional[dict]) -> dict:
    allowlist = (
        "latest_frame",
        "latest_frame_annotated",
        "turn_context_json",
        "turn_plan_json",
        "recovery_saves_json",
    )
    return {
        key: value
        for key, value in ((key, (artifacts or {}).get(key)) for key in allowlist)
        if value
    }


def _truncate_text(value: Optional[str], limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]..."


def _compact_screen_text(screen_text: Optional[dict]) -> Optional[dict]:
    if not screen_text:
        return None
    return {
        "text": _truncate_text(screen_text.get("text"), 420),
        "source": screen_text.get("source"),
        "ui_mode": screen_text.get("ui_mode"),
        "dialog_active": screen_text.get("dialog_active"),
    }


def _compact_feedback(feedback: Optional[dict]) -> Optional[dict]:
    if not feedback:
        return None
    return {
        "source": feedback.get("source"),
        "summary": feedback.get("summary"),
        "notes": (feedback.get("notes") or [])[:4],
        "tags": feedback.get("tags"),
    }


def _compact_state_delta_summary(state_delta: Optional[dict]) -> Optional[dict]:
    if not state_delta:
        return None
    return {
        "changed": state_delta.get("changed"),
        "summary": (state_delta.get("summary") or [])[:4],
        "movement": state_delta.get("movement"),
    }


def _compact_memory_snapshot(memory: Optional[dict]) -> Optional[dict]:
    if not memory:
        return None
    return {
        "recent_facts": (memory.get("recent_facts") or [])[:8],
        "current_hypotheses": (memory.get("current_hypotheses") or [])[:4],
        "failed_attempts": (memory.get("failed_attempts") or [])[:5],
        "session_brief_path": memory.get("session_brief_path"),
    }


def _compact_dialog_guidance(dialog: Optional[dict]) -> Optional[dict]:
    if not dialog:
        return None
    return {
        "transcript_recent": (dialog.get("transcript_recent") or [])[:4],
        "should_continue": dialog.get("should_continue"),
        "last_change_at": dialog.get("last_change_at"),
        "printing": dialog.get("printing"),
        "waiting_for_input": dialog.get("waiting_for_input"),
    }


def _compact_battle_guidance(battle: Optional[dict]) -> Optional[dict]:
    if not battle:
        return None
    return {
        "recommended_mode": battle.get("recommended_mode"),
        "recommended_move": battle.get("recommended_move"),
        "reason": battle.get("reason"),
        "safe_short_actions": (battle.get("safe_short_actions") or [])[:4],
    }


def _compact_movement_guidance(movement_guidance: Optional[dict]) -> Optional[dict]:
    if not movement_guidance:
        return None
    candidate_route = movement_guidance.get("candidate_route") or {}
    compact_route = None
    if candidate_route:
        compact_route = {
            "direction": candidate_route.get("direction"),
            "target": candidate_route.get("target"),
            "steps": candidate_route.get("steps"),
            "actions": (candidate_route.get("actions") or [])[:6],
        }
    return {
        "summary": movement_guidance.get("summary"),
        "notes": (movement_guidance.get("notes") or [])[:4],
        "preferred_direction": movement_guidance.get("preferred_direction"),
        "candidate_route": compact_route,
    }


def _compact_turn_plan(turn_plan: Optional[dict]) -> Optional[dict]:
    if not turn_plan:
        return None
    return {
        "objective_id": turn_plan.get("objective_id"),
        "summary": turn_plan.get("summary"),
        "planned_actions": (turn_plan.get("planned_actions") or [])[:6],
        "fallback_actions": (turn_plan.get("fallback_actions") or [])[:6],
        "notes": _truncate_text(turn_plan.get("notes"), 280),
        "updated_at": turn_plan.get("updated_at"),
        "status": turn_plan.get("status") or {},
        "mode": turn_plan.get("mode"),
        "observation_id": turn_plan.get("observation_id"),
    }


def _compact_plan_status(plan_status: Optional[dict]) -> Optional[dict]:
    if not plan_status:
        return None
    return {
        "state": plan_status.get("state"),
        "observation_id": plan_status.get("observation_id"),
        "validated_at": plan_status.get("validated_at"),
        "executed_at": plan_status.get("executed_at"),
        "outcome_checked_at": plan_status.get("outcome_checked_at"),
        "branch_executed": plan_status.get("branch_executed"),
        "reason": _truncate_text(plan_status.get("reason"), 220),
        "last_error": _truncate_text(plan_status.get("last_error"), 220),
    }


def _observe_contract_response(bundle: Optional[dict]) -> dict:
    if not bundle:
        return {}
    artifacts = _public_artifact_paths(bundle.get("artifacts") or {})
    return {
        "observation_id": bundle.get("observation_id"),
        "generated_at": bundle.get("generated_at"),
        "reason": bundle.get("reason"),
        "source": bundle.get("source"),
        "plan_status": _compact_plan_status(bundle.get("plan_status")),
        "artifacts": artifacts,
        "artifact_urls": _artifact_urls_from_paths(artifacts),
    }


def _compact_bundle_response(bundle: Optional[dict]) -> dict:
    if not bundle:
        return {}
    artifacts = _public_artifact_paths(bundle.get("artifacts") or {})
    objective = (bundle.get("objective") or {}).get("current") or {}
    return {
        "observation_id": bundle.get("observation_id"),
        "generated_at": bundle.get("generated_at"),
        "reason": bundle.get("reason"),
        "source": bundle.get("source"),
        "objective": _compact_objective_status(objective),
        "state": _compact_state_snapshot(bundle.get("state")),
        "screen_text": _compact_screen_text(bundle.get("screen_text")),
        "recent_action": _compact_feedback(bundle.get("recent_action")),
        "state_delta": _compact_state_delta_summary(bundle.get("state_delta")),
        "movement_guidance": _compact_movement_guidance(bundle.get("movement_guidance")),
        "navigation": _compact_navigation_snapshot(
            bundle.get("navigation"),
            bundle.get("navigation_guidance"),
        ),
        "memory": _compact_memory_snapshot(bundle.get("memory")),
        "dialog": _compact_dialog_guidance(bundle.get("dialog_guidance")),
        "battle": _compact_battle_guidance(bundle.get("battle_guidance")),
        "turn_plan": _compact_turn_plan(bundle.get("turn_plan")),
        "plan_status": _compact_plan_status(bundle.get("plan_status")),
        "turn_context": bundle.get("turn_context"),
        "stuck": bundle.get("stuck"),
        "recovery": _compact_recovery_summary(bundle.get("recovery")),
        "artifacts": artifacts,
        "artifact_urls": _artifact_urls_from_paths(artifacts),
    }


def _get_navigation_payload_sync(goal: Optional[tuple[int, int]] = None) -> Optional[dict]:
    try:
        snapshot, location_map = _refresh_navigation_state_sync()
    except NotImplementedError:
        return None
    except Exception:
        return None
    return _serialize_navigation(snapshot, location_map, goal=goal)


def _get_live_navigation_payload_sync(goal: Optional[tuple[int, int]] = None) -> Optional[dict]:
    try:
        snapshot = _emulator.get_navigation_snapshot(_reader)
    except NotImplementedError:
        return None
    except Exception:
        return None
    location_map = None
    if _navigation_store is not None:
        location_map = _navigation_store.get(snapshot.key)
    return _serialize_navigation(snapshot, location_map, goal=goal)


def _make_runtime_save_event(name: str, path: Path, source: str, reason: str) -> dict:
    return {
        "name": name,
        "path": str(path),
        "source": source,
        "reason": reason,
        "notes": [],
    }


def _refresh_agent_bundle_sync(
    *,
    reason: str,
    source: str,
    requested_actions: Optional[list[str]] = None,
    navigation_plan: Optional[dict] = None,
    navigation_execution: Optional[dict] = None,
    explicit_save: Optional[dict] = None,
) -> Optional[dict]:
    if _runtime is None:
        return None
    state = _get_state_dict()
    goal = None
    if navigation_execution and navigation_execution.get("target"):
        target = navigation_execution["target"]
        if target.get("x") is not None and target.get("y") is not None:
            goal = (int(target["x"]), int(target["y"]))
    navigation = _get_navigation_payload_sync(goal=goal)
    return _runtime.refresh(
        emulator=_emulator,
        state=state,
        navigation=navigation,
        navigation_store=_navigation_store,
        reason=reason,
        source=source,
        requested_actions=requested_actions,
        navigation_plan=navigation_plan,
        navigation_execution=navigation_execution,
        explicit_save=explicit_save,
    )


async def _refresh_agent_bundle(
    *,
    reason: str,
    source: str,
    requested_actions: Optional[list[str]] = None,
    navigation_plan: Optional[dict] = None,
    navigation_execution: Optional[dict] = None,
    explicit_save: Optional[dict] = None,
) -> Optional[dict]:
    return await _run_emulator_sync(
        _refresh_agent_bundle_sync,
        reason=reason,
        source=source,
        requested_actions=requested_actions,
        navigation_plan=navigation_plan,
        navigation_execution=navigation_execution,
        explicit_save=explicit_save,
    )


def _sync_live_artifacts_sync() -> Optional[dict]:
    if _runtime is None or _emulator is None:
        return None
    state = _get_state_dict()
    navigation = _get_live_navigation_payload_sync()
    return _runtime.sync_live_view(
        emulator=_emulator,
        state=state,
        navigation=navigation,
        navigation_store=_navigation_store,
    )


async def _broadcast_runtime_refresh(result: Optional[dict]) -> None:
    if not result:
        return
    for event in result.get("events", []):
        await broadcast(event)
    bundle = result.get("bundle")
    if bundle:
        artifacts = bundle.get("artifacts") or {}
        await broadcast(
            {
                "type": "screenshot",
                "data": {
                    "raw_frame_path": artifacts.get("latest_frame"),
                    "annotated_frame_path": artifacts.get("latest_frame_annotated"),
                    "raw_frame_url": _artifact_urls_from_paths(artifacts).get("latest_frame"),
                    "annotated_frame_url": _artifact_urls_from_paths(artifacts).get(
                        "latest_frame_annotated"
                    ),
                    "frame_timestamp": bundle.get("generated_at"),
                    "source": bundle.get("source"),
                },
            }
        )


def _navigation_not_supported(exc: Exception) -> HTTPException:
    return HTTPException(status_code=501, detail=f"Navigation unavailable: {exc}")


def _refresh_navigation_state_sync():
    """Read the current live navigation snapshot and merge it into the store."""
    snapshot = _emulator.get_navigation_snapshot(_reader)
    location_map = _navigation_store.update(snapshot)
    return snapshot, location_map


def _plan_navigation_sync(target_x: int, target_y: int, mode: str):
    """Plan a navigation route using either the live or persistent map."""
    snapshot, location_map = _refresh_navigation_state_sync()
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"auto", "screen", "persistent"}:
        raise ValueError(f"Unknown navigation mode: {mode}")

    target_visible = snapshot.absolute_to_local(target_x, target_y) is not None
    if normalized_mode == "screen" or (normalized_mode == "auto" and target_visible):
        plan = _emulator.plan_screen_path(_reader, target_x, target_y)
        plan.visible_target = target_visible

        if normalized_mode == "auto" and target_visible and not plan.reached:
            local_target = snapshot.absolute_to_local(target_x, target_y)
            target_is_walkable = False
            if local_target is not None:
                local_x, local_y = local_target
                target_is_walkable = (
                    bool(snapshot.terrain[local_y][local_x])
                    and (target_x, target_y) not in snapshot.sprite_set
                )

            if target_is_walkable:
                persistent_plan = location_map.plan_route(
                    start=snapshot.player_position,
                    goal=(target_x, target_y),
                    extra_blockers=snapshot.sprite_set,
                )
                persistent_plan.visible_target = True

                if persistent_plan.reached:
                    persistent_plan.status = (
                        "Visible target requires leaving the current screen; "
                        + persistent_plan.status
                    )
                    plan = persistent_plan
                elif not plan.actions and persistent_plan.actions:
                    persistent_plan.status = (
                        "Visible target was not routable on-screen; " + persistent_plan.status
                    )
                    plan = persistent_plan
                elif (
                    plan.final_position is not None
                    and persistent_plan.final_position is not None
                    and persistent_plan.actions
                ):
                    screen_remaining = abs(plan.final_position[0] - target_x) + abs(
                        plan.final_position[1] - target_y
                    )
                    persistent_remaining = abs(persistent_plan.final_position[0] - target_x) + abs(
                        persistent_plan.final_position[1] - target_y
                    )
                    if persistent_remaining < screen_remaining:
                        persistent_plan.status = (
                            "Visible target has a better explored off-screen route; "
                            + persistent_plan.status
                        )
                        plan = persistent_plan
    else:
        plan = location_map.plan_route(
            start=snapshot.player_position,
            goal=(target_x, target_y),
            extra_blockers=snapshot.sprite_set,
        )
        plan.visible_target = target_visible
    return snapshot, location_map, plan


def _serialize_navigation(snapshot, location_map, goal: Optional[tuple[int, int]] = None) -> dict:
    """Convert navigation objects into JSON-safe response data."""
    if location_map is None:
        location_map_payload = {
            "location_key": snapshot.key,
            "map_id": snapshot.map_id,
            "map_name": snapshot.map_name,
            "tileset": snapshot.tileset,
            "updates": 0,
            "known_tiles": 0,
            "known_tile_ids": 0,
            "passable_tiles": 0,
            "blocked_tiles": 0,
            "bounds": None,
            "ascii": "(no explored map data)",
            "ascii_legend": {
                "P": "player",
                "G": "goal",
                "S": "visible sprite blocker",
                ".": "explored passable tile",
                "#": "known blocked tile",
                "?": "unexplored or unknown tile",
            },
        }
    else:
        distances = location_map.distance_map(
            snapshot.player_position,
            extra_blockers=snapshot.sprite_set,
        )
        location_map_payload = location_map.to_dict(
            player=snapshot.player_position,
            goal=goal,
            sprites=snapshot.sprite_positions,
            distances=distances,
        )
    return {
        "snapshot": snapshot.to_dict(goal=goal),
        "location_map": location_map_payload,
    }


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"^(?P<kind>press|walk|hold|wait|a_until_dialog_end)(?:_(?P<rest>.+))?$")


def _execute_action_sync(action_str: str) -> None:
    """Parse and execute a single action string on the emulator.

    Supported formats:
        press_X       — press button X for 10 frames, wait 20 frames
        walk_X        — press direction for 16 frames, wait 8 frames
        hold_X_N      — hold button X for N frames
        wait_N        — tick N frames with no input
        a_until_dialog_end — press A every 30 frames until dialog clears (max 300)
    """
    action_str = action_str.strip().lower()

    if action_str == "a_until_dialog_end":
        for _ in range(10):  # max 300 frames = 10 * 30
            _emulator.press("a")
            _emulator.tick(30)
            # Check dialog flag via reader if available
            try:
                state = _get_state_dict()
                if not state.get("dialog_active", False):
                    break
            except Exception:
                pass
        return

    # Split into tokens
    parts = action_str.split("_")

    if parts[0] == "press" and len(parts) >= 2:
        button = "_".join(parts[1:])
        # Hold button for 8 frames so the game registers the press,
        # then wait 12 frames for the game to process it.
        _emulator.press(button, 8)
        _emulator.tick(12)
        return

    if parts[0] == "walk" and len(parts) >= 2:
        direction = parts[1]
        # Gen 1 movement timing (empirically tested):
        #   - Button must be held >= 4 frames for the game's vblank joypad
        #     poll to register the input reliably.
        #   - wWalkCounter starts at 8, decrements each frame (2 px/frame
        #     = 16 px = 1 tile). Total walk animation = ~16 frames.
        #   - Minimum total frames for a confirmed tile move = 17.
        #   - We use hold=8 + wait=12 = 20 total for a safety margin.
        _emulator.press(direction, 8)
        _emulator.tick(12)
        return

    if parts[0] == "hold" and len(parts) >= 3:
        button = "_".join(parts[1:-1])
        frames = int(parts[-1])
        _emulator.press(button, frames)
        return

    if parts[0] == "wait" and len(parts) == 2:
        frames = int(parts[1])
        _emulator.tick(frames)
        return

    raise ValueError(f"Unknown action format: {action_str}")


def _execute_action_batch_sync(actions: list[str]) -> int:
    executed = 0
    for action_str in actions:
        _execute_action_sync(action_str)
        executed += 1
    return executed


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def configure(config: GameConfig):
    """Set server configuration (call before app startup)."""
    global _config
    _config = config


@app.on_event("startup")
async def _startup():
    global \
        _emulator, \
        _reader, \
        _navigation_store, \
        _runtime, \
        _supervisor, \
        _start_time, \
        _config, \
        _loop, \
        _dashboard_dir, \
        _emulator_lock, \
        _realtime_task, \
        _realtime_frames_per_second, \
        _realtime_enabled, \
        _realtime_ticks, \
        _realtime_last_tick_at, \
        _live_artifact_task, \
        _live_artifact_frames_per_second, \
        _live_artifact_last_sync_at
    _loop = asyncio.get_running_loop()
    _start_time = time.time()
    _emulator_lock = asyncio.Lock()
    _realtime_ticks = 0
    _realtime_last_tick_at = None
    _live_artifact_last_sync_at = None

    if _config is None:
        # Config can be injected via environment or set beforehand
        print("[server] WARNING: No GameConfig set — emulator will NOT start.")
        print("[server] Call server.configure(GameConfig(...)) before startup.")
        return

    rom = Path(_config.rom_path).expanduser().resolve()
    if not rom.exists():
        print(f"[server] ERROR: ROM not found: {rom}")
        return

    # Auto-detect game type
    game_type = _config.game_type
    if game_type == "auto":
        game_type = _detect_game_type(str(rom))
    _config.game_type = game_type

    print(f"[server] Loading ROM: {rom}")
    print(f"[server] Detected game type: {game_type}")

    # Create emulator
    from pokemon_agent.emulator import create_emulator

    _emulator = create_emulator(str(rom))

    # Create memory reader
    if game_type == "red":
        from pokemon_agent.memory.red import PokemonRedReader

        _reader = PokemonRedReader(_emulator)
    elif game_type == "firered":
        from pokemon_agent.memory.firered import PokemonFireRedReader

        _reader = PokemonFireRedReader(_emulator)
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    # Create data directories
    data_dir = Path(_config.data_dir).expanduser().resolve()
    (data_dir / "saves").mkdir(parents=True, exist_ok=True)
    from pokemon_agent.navigation import NavigationStore

    _navigation_store = NavigationStore(data_dir / "navigation_maps.json")
    workspace_dir = (
        Path(_config.agent_workspace_dir).expanduser().resolve()
        if _config.agent_workspace_dir
        else (data_dir / "agent_workspace").resolve()
    )
    _runtime = AgentRuntime(data_dir=data_dir, workspace_dir=workspace_dir)
    _supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url=f"http://127.0.0.1:{_config.port}",
        event_sink=_record_existing_event_and_broadcast,
        stream_sink=broadcast,
    )
    _realtime_frames_per_second = max(1, int(_config.realtime_fps))
    _realtime_enabled = bool(_config.realtime)
    configured_broadcast_fps = getattr(_config, "live_artifact_broadcast_fps", None)
    if configured_broadcast_fps is None:
        _live_artifact_frames_per_second = min(
            _realtime_frames_per_second, max(1, _live_artifact_frames_per_second)
        )
    else:
        _live_artifact_frames_per_second = max(1, int(configured_broadcast_fps))

    if _config.enable_dashboard:
        _dashboard_dir = _get_dashboard_static_dir()
        if _dashboard_dir is not None:
            from fastapi.staticfiles import StaticFiles

            if not any(
                getattr(route, "path", None) == "/dashboard/assets" for route in app.router.routes
            ):
                app.mount(
                    "/dashboard/assets",
                    StaticFiles(directory=str(_dashboard_dir), html=False),
                    name="dashboard-assets",
                )
            print("[server] Dashboard assets mounted at /dashboard/assets")
        else:
            print("[server] Dashboard static files not found — /dashboard unavailable")

    # Auto-load a save state if specified
    if _config.load_state:
        saves_dir = data_dir / "saves"
        state_path = saves_dir / f"{_config.load_state}.state"
        if state_path.exists():
            try:
                _emulator.load_state(str(state_path))
                print(f"[server] Loaded save state: {_config.load_state}")
            except Exception as e:
                print(f"[server] WARNING: Failed to load state '{_config.load_state}': {e}")
        else:
            print(f"[server] WARNING: Save state not found: {state_path}")

    # Seed navigation data when available.
    try:
        _refresh_navigation_state_sync()
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"[server] WARNING: Initial navigation snapshot failed: {e}")

    try:
        result = await _refresh_agent_bundle(reason="startup_refresh", source="observe")
        await _broadcast_runtime_refresh(result)
    except Exception as e:
        print(f"[server] WARNING: Initial agent workspace refresh failed: {e}")

    if _realtime_enabled:
        _realtime_task = asyncio.create_task(_realtime_emulator_loop())
        _live_artifact_task = asyncio.create_task(_live_artifact_loop())
        print(f"[server] Realtime emulation enabled at {_realtime_frames_per_second} FPS")
        print(f"[server] Live artifact sync enabled at {_live_artifact_frames_per_second} FPS")
    else:
        _realtime_task = None
        _live_artifact_task = None
        print("[server] Realtime emulation disabled")

    print(f"[server] Ready — listening on port {_config.port}")
    print(f"[server] Agent workspace: {workspace_dir}")
    print("[server] Endpoints:")
    print("[server]   GET  /          — server info")
    print("[server]   GET  /state     — game state")
    print("[server]   GET  /screenshot — current frame (PNG)")
    print("[server]   POST /agent/observe — refresh the curated turn context")
    print("[server]   POST /agent/plan — validate and persist one turn plan")
    print("[server]   POST /agent/act — execute the validated plan branch")
    print("[server]   POST /action    — execute actions")
    print("[server]   POST /save      — save state")
    print("[server]   POST /load      — load state")
    print("[server]   GET  /saves     — list saves")
    print("[server]   GET  /minimap   — ASCII minimap")
    print("[server]   GET  /navigation/map      — explored navigation map")
    print("[server]   POST /navigation/path     — plan a navigation route")
    print("[server]   POST /navigation/navigate — execute a navigation route")
    print("[server]   GET  /dashboard/state  — aggregated telemetry state")
    print("[server]   GET  /dashboard/history — structured event timeline")
    print("[server]   GET  /supervisor/state — Pi supervisor snapshot")
    print("[server]   POST /supervisor/start — launch a supervised Pi session")
    print("[server]   POST /supervisor/continue — run one more Pi turn")
    print("[server]   POST /supervisor/stop — stop the Pi supervisor")
    print("[server]   GET  /health    — health check")
    print("[server]   WS   /ws        — live events")


@app.on_event("shutdown")
async def _shutdown():
    global _supervisor, _realtime_task, _live_artifact_task
    if _live_artifact_task is not None:
        _live_artifact_task.cancel()
        try:
            await _live_artifact_task
        except asyncio.CancelledError:
            pass
        _live_artifact_task = None
    if _realtime_task is not None:
        _realtime_task.cancel()
        try:
            await _realtime_task
        except asyncio.CancelledError:
            pass
        _realtime_task = None
    if _supervisor is not None:
        if hasattr(_supervisor, "shutdown"):
            await _supervisor.shutdown()
        elif hasattr(_supervisor, "stop"):
            await _supervisor.stop()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    """Server info."""
    return {
        "name": "pokemon-agent",
        "version": __version__,
        "game": _config.game_type if _config else None,
        "rom": _config.rom_path if _config else None,
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "emulator_ready": _emulator is not None,
        "agent_workspace_dir": str(_runtime.workspace_dir) if _runtime else None,
        "dashboard_ready": _dashboard_dir is not None,
        "emulation": _server_runtime_snapshot(),
    }


@app.get("/health")
async def health():
    """Health check."""
    supervisor_snapshot = _supervisor.state_snapshot() if _supervisor is not None else None
    return {
        "status": "ok",
        "emulator_ready": _emulator is not None,
        "agent_workspace_ready": _runtime is not None,
        "dashboard_ready": _dashboard_dir is not None,
        "emulation": _server_runtime_snapshot(),
        "pi_supervisor": _compact_supervisor_status(supervisor_snapshot),
    }


@app.get("/dashboard")
@app.get("/dashboard/")
async def dashboard_index():
    """Serve the telemetry dashboard shell."""
    if _dashboard_dir is None:
        raise HTTPException(
            status_code=404,
            detail="Dashboard static files are not available in this installation.",
        )
    return FileResponse(
        _dashboard_dir / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/artifacts/{artifact_key}")
async def get_workspace_artifact(artifact_key: str):
    """Serve a whitelisted workspace artifact file."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    path = _runtime.artifacts.get(artifact_key)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {artifact_key}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_key}")
    data = path.read_bytes()
    content_type = _guess_content_type(path)
    return Response(content=data, media_type=content_type)


@app.get("/dashboard/state")
async def dashboard_state():
    """Aggregated dashboard state for the operator console."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    payload = _runtime.dashboard_state()
    artifacts = _public_artifact_paths(payload.get("artifacts") or {})
    payload["artifacts"] = artifacts
    payload["artifact_urls"] = _artifact_urls_from_paths(artifacts)
    payload["pi_supervisor"] = _supervisor.state_snapshot() if _supervisor is not None else {}
    payload["server_runtime"] = _server_runtime_snapshot()
    return JSONResponse(content=payload)


@app.get("/dashboard/history")
async def dashboard_history(limit: int = 200):
    """Structured recent dashboard/agent events."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    limit = max(1, min(limit, 1000))
    return {"events": _runtime.history(limit)}


@app.get("/supervisor/state")
async def supervisor_state():
    """Current Pi supervisor status and recent event stream."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    return JSONResponse(content=_compact_supervisor_status(_supervisor.state_snapshot()))


@app.post("/supervisor/start")
async def supervisor_start(req: PiSupervisorStartRequest):
    """Launch Pi under server supervision."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    try:
        state = await _supervisor.start(
            goal=req.goal,
            provider=req.provider,
            model=req.model,
            thinking=req.thinking,
            auto_continue=req.auto_continue,
            max_turns=req.max_turns,
            continue_delay_seconds=req.continue_delay_seconds,
            skill_path=req.skill_path,
        )
        return {"success": True, "supervisor": _compact_supervisor_status(state)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Supervisor start error: {exc}")


@app.post("/supervisor/continue")
async def supervisor_continue(req: PiSupervisorContinueRequest):
    """Run one more Pi turn against the latest session."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    try:
        state = await _supervisor.continue_once()
        return {"success": True, "supervisor": _compact_supervisor_status(state)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Supervisor continue error: {exc}")


@app.post("/supervisor/stop")
async def supervisor_stop():
    """Stop Pi if it is currently running."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    try:
        state = await _supervisor.stop()
        return {"success": True, "supervisor": _compact_supervisor_status(state)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Supervisor stop error: {exc}")


@app.post("/agent/observe")
async def agent_observe(req: Optional[ObserveRequest] = None):
    """Refresh the vision-first workspace bundle and return artifact paths."""
    _ensure_emulator()
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    request = req or ObserveRequest()
    try:
        result = await _refresh_agent_bundle(reason=request.reason, source="observe")
        await _broadcast_runtime_refresh(result)
        if not result:
            raise HTTPException(
                status_code=500, detail="Agent observation refresh returned no data"
            )
        return _observe_contract_response(result["bundle"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent observe error: {e}")


@app.post("/agent/plan")
async def agent_plan(req: TurnPlanInput):
    """Validate and persist one strict turn plan for the latest observation."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    try:
        plan = _runtime.validate_and_store_turn_plan(req.model_dump(mode="json"))
        await _record_and_broadcast(
            "turn_plan_validated",
            {
                "observation_id": plan.observation_id,
                "objective_id": plan.objective_id,
                "intent": plan.intent,
                "mode": plan.mode,
                "status": plan.status.model_dump(mode="json"),
            },
        )
        return {
            "success": True,
            "turn_plan": _compact_turn_plan(_runtime.load_turn_plan()),
            "plan_status": _compact_plan_status(plan.status.model_dump(mode="json")),
            "artifacts": _public_artifact_paths(_runtime._artifact_payload()),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent plan error: {exc}")


@app.post("/agent/act")
async def agent_act(req: Optional[AgentActRequest] = None):
    """Execute the validated primary or fallback branch from turn_plan.json."""
    _ensure_emulator()
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    request = req or AgentActRequest()
    try:
        plan = _runtime.load_turn_plan_model()
        if plan.status.state != "validated":
            raise HTTPException(
                status_code=400,
                detail="No validated turn plan is ready to execute",
            )

        context = _runtime.load_turn_context()
        if not context or plan.observation_id != context.get("observation_id"):
            _runtime.invalidate_turn_plan(
                "A newer observation exists; the validated plan is now stale.",
                state="stale",
            )
            raise HTTPException(status_code=400, detail="Validated turn plan is stale")

        branch = plan.primary_branch if request.branch == "primary" else plan.fallback_branch
        if branch is None:
            raise HTTPException(
                status_code=400,
                detail=f"No {request.branch} branch is available",
            )

        state_before = await _run_emulator_sync(_get_state_dict)
        baseline_map_name = (state_before.get("map") or {}).get("map_name")
        baseline_position = (state_before.get("player") or {}).get("position")

        if branch.kind == "raw_actions":
            requested_actions = list(branch.actions)
            await _record_and_broadcast(
                "action",
                {
                    "actions": requested_actions,
                    "source": "agent_act",
                    "branch": request.branch,
                    "plan_observation_id": plan.observation_id,
                },
            )
            executed = await _run_emulator_sync(_execute_action_batch_sync, requested_actions)
            state_after = await _run_emulator_sync(_get_state_dict)
            navigation_after = await _run_emulator_sync(_get_navigation_payload_sync)
            updated_plan = _runtime.mark_turn_plan_executed(
                branch=request.branch,
                branch_kind=branch.kind,
                requested_actions=requested_actions,
                executed_actions=executed,
                baseline_map_name=baseline_map_name,
                baseline_position=baseline_position,
                summary=f"Executed {executed} raw action(s).",
            )
            await _record_and_broadcast(
                "action_result",
                {
                    "actions": requested_actions,
                    "actions_executed": executed,
                    "source": "agent_act",
                    "branch": request.branch,
                    "state_after": state_after,
                    "navigation_after": navigation_after,
                    "plan_status": updated_plan.status.model_dump(mode="json"),
                },
            )
            return {
                "success": True,
                "branch": request.branch,
                "actions_requested": requested_actions,
                "actions_executed": executed,
                "requires_observe": True,
                "state_after": _compact_state_snapshot(state_after),
                "plan_status": _compact_plan_status(updated_plan.status.model_dump(mode="json")),
            }

        target = (branch.target.x, branch.target.y)
        snapshot_before, location_map_before, nav_plan = await _run_emulator_sync(
            _plan_navigation_sync,
            branch.target.x,
            branch.target.y,
            branch.mode,
        )
        if _dialog_is_active(state_before):
            _runtime.invalidate_turn_plan(
                "Navigation plan became invalid because a dialog or prompt is open.",
            )
            raise HTTPException(
                status_code=400,
                detail="Navigation plan cannot execute while a dialog or prompt is open",
            )

        executed = await _run_emulator_sync(_execute_action_batch_sync, nav_plan.actions)
        state_after = await _run_emulator_sync(_get_state_dict)
        snapshot_after, location_map_after = await _run_emulator_sync(
            _refresh_navigation_state_sync
        )
        navigation_after = _serialize_navigation(snapshot_after, location_map_after, goal=target)
        execution = _summarize_navigation_execution(
            snapshot_before,
            snapshot_after,
            nav_plan,
            target,
        )
        updated_plan = _runtime.mark_turn_plan_executed(
            branch=request.branch,
            branch_kind=branch.kind,
            requested_actions=list(nav_plan.actions),
            executed_actions=executed,
            baseline_map_name=baseline_map_name,
            baseline_position=baseline_position,
            summary=execution["status"],
        )
        await _record_and_broadcast(
            "navigation",
            {
                "target": {"x": branch.target.x, "y": branch.target.y},
                "mode": branch.mode,
                "plan": nav_plan.to_dict(),
                "execution": execution,
                "source": "agent_act",
                "branch": request.branch,
                "state_after": state_after,
                "navigation_after": navigation_after,
                "plan_status": updated_plan.status.model_dump(mode="json"),
            },
        )
        return {
            "success": True,
            "branch": request.branch,
            "actions_requested": list(nav_plan.actions),
            "actions_executed": executed,
            "requires_observe": True,
            "navigation_execution": execution,
            "state_after": _compact_state_snapshot(state_after),
            "plan_status": _compact_plan_status(updated_plan.status.model_dump(mode="json")),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent act error: {exc}")


@app.get("/agent/navigator")
async def agent_navigator():
    """Return a strict JSON navigator view over the latest deterministic guidance."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    try:
        return JSONResponse(content=_runtime.navigator_payload())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent navigator error: {e}")


@app.get("/state")
async def get_state():
    """Full game state JSON."""
    _ensure_emulator()
    try:
        state = await _run_emulator_sync(_get_state_dict)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading state: {e}")


@app.get("/screenshot")
async def screenshot():
    """Current emulator frame as PNG image."""
    _ensure_emulator()
    try:
        png_bytes = await _run_emulator_sync(_get_screenshot_bytes)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.get("/screenshot/base64")
async def screenshot_base64():
    """Current emulator frame as base64-encoded PNG in JSON."""
    _ensure_emulator()
    try:
        png_bytes = await _run_emulator_sync(_get_screenshot_bytes)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {"image": b64, "format": "png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.post("/action")
async def execute_actions(req: ActionRequest):
    """Execute a sequence of game actions."""
    _ensure_emulator()
    try:
        if _runtime is not None:
            _runtime.invalidate_turn_plan(
                "Manual /action invalidated the previously validated plan."
            )
        state_before = await _run_emulator_sync(_get_state_dict)
        await _record_and_broadcast(
            "action",
            {
                "actions": req.actions,
                "state_before": state_before,
            },
        )
        executed = await _run_emulator_sync(_execute_action_batch_sync, req.actions)

        refresh = await _refresh_agent_bundle(
            reason="actions_executed",
            source="action",
            requested_actions=req.actions,
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        state_after = bundle.get("state") or await _run_emulator_sync(_get_state_dict)
        navigation_after = bundle.get("navigation") or await _run_emulator_sync(
            _get_navigation_payload_sync
        )

        await _record_and_broadcast(
            "action_result",
            {
                "actions": req.actions,
                "actions_executed": executed,
                "state_after": state_after,
                "navigation_after": navigation_after,
                "feedback": bundle.get("recent_action"),
                "state_delta": bundle.get("state_delta"),
                "objective_status": (bundle.get("objective") or {}).get("current"),
                "stuck_signal": bundle.get("stuck"),
                "screen_text": bundle.get("screen_text"),
            },
        )

        return {
            "success": True,
            "actions_requested": req.actions,
            "actions_executed": executed,
            **_compact_bundle_response(bundle),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action error: {e}")


@app.post("/save")
async def save_state(req: SaveRequest):
    """Save emulator state to disk."""
    _ensure_emulator()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        saves_dir.mkdir(parents=True, exist_ok=True)
        save_path = saves_dir / f"{req.name}.state"
        await _run_emulator_sync(_emulator.save_state, str(save_path))
        save_event = _make_runtime_save_event(
            req.name,
            save_path,
            source="manual",
            reason="manual_save",
        )
        refresh = await _refresh_agent_bundle(
            reason=f"manual_save:{req.name}",
            source="save",
            explicit_save=save_event,
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        return {
            "success": True,
            "save": {"name": req.name, "path": str(save_path)},
            **_compact_bundle_response(bundle),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {e}")


@app.post("/load")
async def load_state(req: SaveRequest):
    """Load emulator state from disk."""
    _ensure_emulator()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        save_path = saves_dir / f"{req.name}.state"
        if not save_path.exists():
            raise HTTPException(status_code=404, detail=f"Save not found: {req.name}")
        if _runtime is not None:
            _runtime.invalidate_turn_plan("Loading a save invalidated the current plan.")
        await _run_emulator_sync(_emulator.load_state, str(save_path))
        refresh = await _refresh_agent_bundle(
            reason=f"manual_load:{req.name}",
            source="load",
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        state_after = bundle.get("state") or await _run_emulator_sync(_get_state_dict)
        navigation_after = bundle.get("navigation") or await _run_emulator_sync(
            _get_navigation_payload_sync
        )
        await _record_and_broadcast("load", {"name": req.name, "path": str(save_path)})
        await broadcast(
            {
                "type": "state_update",
                "reason": "load",
                "state": state_after,
                "navigation_after": navigation_after,
            }
        )

        return {
            "success": True,
            "save": {"name": req.name, "path": str(save_path)},
            **_compact_bundle_response(bundle),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")


@app.get("/saves")
async def list_saves():
    """List available save-state files."""
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        if not saves_dir.exists():
            return {"saves": []}
        files = sorted(saves_dir.glob("*.state"))
        saves = [
            {
                "name": f.stem,
                "file": f.name,
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in files
        ]
        return {"saves": saves}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing saves: {e}")


@app.get("/minimap")
async def minimap():
    """ASCII explored navigation map for the current location."""
    _ensure_emulator()
    try:
        snapshot, location_map = await _run_emulator_sync(_refresh_navigation_state_sync)
        distances = location_map.distance_map(
            snapshot.player_position,
            extra_blockers=snapshot.sprite_set,
        )
        lines = [
            f"=== {snapshot.map_name} ===",
            f"Player position: {snapshot.player_position}",
            f"Tileset: {snapshot.tileset}",
            "",
            location_map.render_ascii(
                player=snapshot.player_position,
                sprites=snapshot.sprite_positions,
                distances=distances,
            ),
        ]
        return Response(content="\n".join(lines), media_type="text/plain")
    except NotImplementedError as e:
        raise _navigation_not_supported(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Minimap error: {e}")


@app.get("/navigation/map")
async def navigation_map():
    """Return the current live navigation window and explored map."""
    _ensure_emulator()
    try:
        snapshot, location_map = await _run_emulator_sync(_refresh_navigation_state_sync)
        return _compact_navigation_snapshot(_serialize_navigation(snapshot, location_map))
    except NotImplementedError as e:
        raise _navigation_not_supported(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation map error: {e}")


@app.post("/navigation/path")
async def navigation_path(req: NavigationRequest):
    """Plan a navigation route without executing it."""
    _ensure_emulator()
    try:
        snapshot, location_map, plan = await _run_emulator_sync(
            _plan_navigation_sync,
            req.x,
            req.y,
            req.mode,
        )
        payload = _serialize_navigation(snapshot, location_map, goal=(req.x, req.y))
        return {
            "target": {"x": req.x, "y": req.y},
            "plan": plan.to_dict(),
            "navigation": _compact_navigation_snapshot(payload),
        }
    except NotImplementedError as e:
        raise _navigation_not_supported(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation path error: {e}")


@app.post("/navigation/navigate")
async def navigation_navigate(req: NavigationRequest):
    """Plan and execute a navigation route."""
    _ensure_emulator()
    try:
        if _runtime is not None:
            _runtime.invalidate_turn_plan(
                "Manual /navigation/navigate invalidated the previously validated plan."
            )
        snapshot_before, location_map_before, plan = await _run_emulator_sync(
            _plan_navigation_sync,
            req.x,
            req.y,
            req.mode,
        )
        state_before = await _run_emulator_sync(_get_state_dict)
        navigation_before = _serialize_navigation(
            snapshot_before,
            location_map_before,
            goal=(req.x, req.y),
        )

        if _dialog_is_active(state_before):
            execution = {
                "success": False,
                "moved": False,
                "started_at_target": snapshot_before.player_position == (req.x, req.y),
                "reached_target": snapshot_before.player_position == (req.x, req.y),
                "matched_planned_final_position": False,
                "planned_final_position": (
                    {"x": plan.final_position[0], "y": plan.final_position[1]}
                    if plan.final_position is not None
                    else None
                ),
                "actual_position": {
                    "x": snapshot_before.player_position[0],
                    "y": snapshot_before.player_position[1],
                },
                "target": {"x": req.x, "y": req.y},
                "remaining_distance_before": (
                    abs(snapshot_before.player_position[0] - req.x)
                    + abs(snapshot_before.player_position[1] - req.y)
                ),
                "remaining_distance_after": (
                    abs(snapshot_before.player_position[0] - req.x)
                    + abs(snapshot_before.player_position[1] - req.y)
                ),
                "status": "Navigation did not start because a dialog or prompt is open.",
                "blocked_by_dialog": True,
                "suggested_actions": ["press_a"],
            }
            refresh = await _refresh_agent_bundle(
                reason="navigation_blocked_by_dialog",
                source="navigation",
                requested_actions=[],
                navigation_plan=plan.to_dict(),
                navigation_execution=execution,
            )
            await _broadcast_runtime_refresh(refresh)
            bundle = (refresh or {}).get("bundle", {})
            await _record_and_broadcast(
                "navigation",
                {
                    "target": {"x": req.x, "y": req.y},
                    "plan": plan.to_dict(),
                    "execution": execution,
                    "state_after": bundle.get("state") or state_before,
                    "navigation_after": bundle.get("navigation") or navigation_before,
                },
            )
            return {
                "success": False,
                "actions_executed": 0,
                "plan": plan.to_dict(),
                "execution": execution,
                "actions_requested": plan.actions,
                **_compact_bundle_response(bundle),
            }

        executed = await _run_emulator_sync(_execute_action_batch_sync, plan.actions)

        state_after = await _run_emulator_sync(_get_state_dict)
        snapshot_after, location_map_after = await _run_emulator_sync(
            _refresh_navigation_state_sync
        )
        navigation_after = _serialize_navigation(
            snapshot_after,
            location_map_after,
            goal=(req.x, req.y),
        )
        execution = _summarize_navigation_execution(
            snapshot_before,
            snapshot_after,
            plan,
            (req.x, req.y),
        )
        refresh = await _refresh_agent_bundle(
            reason="navigation_executed",
            source="navigation",
            requested_actions=plan.actions,
            navigation_plan=plan.to_dict(),
            navigation_execution=execution,
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        state_after = bundle.get("state") or state_after
        navigation_after = bundle.get("navigation") or navigation_after

        await _record_and_broadcast(
            "navigation",
            {
                "target": {"x": req.x, "y": req.y},
                "plan": plan.to_dict(),
                "execution": execution,
                "actions_executed": executed,
                "state_after": state_after,
                "navigation_after": navigation_after,
            },
        )
        await _record_and_broadcast(
            "action_result",
            {
                "actions": plan.actions,
                "actions_executed": executed,
                "state_after": state_after,
                "navigation_after": navigation_after,
                "feedback": bundle.get("recent_action"),
                "state_delta": bundle.get("state_delta"),
                "objective_status": (bundle.get("objective") or {}).get("current"),
                "stuck_signal": bundle.get("stuck"),
                "screen_text": bundle.get("screen_text"),
            },
        )

        return {
            "success": execution["success"],
            "actions_requested": plan.actions,
            "actions_executed": executed,
            "plan": plan.to_dict(),
            "execution": execution,
            **_compact_bundle_response(bundle),
        }
    except NotImplementedError as e:
        raise _navigation_not_supported(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation execute error: {e}")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live event stream via WebSocket."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send a welcome message
        await ws.send_json(
            {
                "type": "connected",
                "version": __version__,
                "emulator_ready": _emulator is not None,
                "agent_workspace_dir": str(_runtime.workspace_dir) if _runtime else None,
                "emulation": _server_runtime_snapshot(),
            }
        )
        # Keep alive — wait for client messages (or disconnect)
        while True:
            data = await ws.receive_text()
            # Clients can send a "ping" to keep alive
            if data.strip().lower() == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)
