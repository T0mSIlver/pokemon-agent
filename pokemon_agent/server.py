"""
Pokemon Agent — FastAPI Game Server

Provides HTTP + WebSocket API for controlling a Game Boy / GBA emulator
running a Pokemon ROM, reading game state, and broadcasting events.

`POST /action` is the model-facing endpoint: it executes buttons, rewrites the
workspace frames the model looks at, and answers with a handful of fields.
Everything richer than that belongs to the dashboard endpoints.
"""

import asyncio
import base64
import contextlib
import io
import json
import mimetypes
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.routing import Mount

from pokemon_agent.agent_runtime import AgentRuntime
from pokemon_agent.explored_map import ExploredMaps
from pokemon_agent.memory.red import MAP_NAMES, MOVE_NAMES
from pokemon_agent.pi_supervisor import NoLiveSessionError, PiSupervisor

__version__ = "0.1.0"

SCREEN_TEXT_LIMIT = 160


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


class BattleFightRequest(BaseModel):
    """Body for POST /battle/fight."""

    move: str


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


class PiSupervisorSteerRequest(BaseModel):
    """Body for POST /supervisor/steer."""

    message: str


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_emulator = None  # Emulator instance
_reader = None  # GameMemoryReader subclass instance
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

# Persistent explored-map memory. Every navigation snapshot is folded into it;
# the agent reads it back on demand from GET /map. Writing it out on every
# snapshot would mean a disk write per emulated frame, hence the throttle.
_explored_maps: Optional[ExploredMaps] = None
EXPLORED_SAVE_INTERVAL_SECONDS = 10.0
EXPLORED_SAVE_EVERY_RECORDS = 50
_explored_last_save_at: float = 0.0
_explored_records_since_save: int = 0

# WebSocket clients
_ws_clients: Set[WebSocket] = set()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


app = FastAPI(
    title="Pokemon Agent Server",
    version=__version__,
    description="HTTP + WebSocket API for Pokemon emulator control",
    lifespan=_lifespan,
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


def _ensure_runtime() -> AgentRuntime:
    """Return the agent runtime, or raise 503 if it isn't ready."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    return _runtime


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
    battle_menu = _battle_menu_sync(state)
    if battle_menu is not None:
        state["battle_menu"] = battle_menu
    try:
        snapshot = _emulator.get_navigation_snapshot(_reader)
    except NotImplementedError:
        return state
    except Exception:
        return state
    state["interaction"] = snapshot.interaction
    return state


def _battle_menu_sync(state: dict) -> Optional[dict]:
    """Which battle menu is open and what the cursor is on, or None outside battle.

    A menu the agent cannot see is a menu it presses A into blindly. This is the
    one fact the frame shows and the payload used to hide.
    """
    if not ((state.get("battle") or {}).get("in_battle")):
        return None
    read = getattr(_reader, "read_battle_menu", None)
    if read is None:
        return None
    try:
        return read()
    except Exception:  # noqa: BLE001 — perception must never fail a state read
        return None


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
                        "raw_frame_path": artifacts.get("live_frame"),
                        "annotated_frame_path": artifacts.get("live_frame_annotated"),
                        "raw_frame_url": _artifact_urls_from_paths(artifacts).get("live_frame"),
                        "annotated_frame_url": _artifact_urls_from_paths(artifacts).get(
                            "live_frame_annotated"
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


def _runtime_artifact_paths() -> dict:
    """Absolute path per served artifact key, for the supervisor's stream log."""
    if _runtime is None:
        return {}
    return {key: str(path) for key, path in _runtime.artifacts.items()}


async def _critic_context() -> dict:
    """Objective, game state and explored-map summary for the between-session critic.

    Called once when a session starts and once when it ends, so the retrospective
    can show progress rather than just a final position. Never raises.
    """
    payload: dict = {}
    if _runtime is not None:
        with contextlib.suppress(Exception):
            bundle = _runtime.live_bundle or _runtime.latest_bundle or {}
            current = ((bundle.get("objective") or {}).get("current")) or {}
            summary = current.get("summary")
            if summary:
                payload["objective"] = str(summary)
    if _emulator is not None and _reader is not None:
        with contextlib.suppress(Exception):
            payload["game_state"] = await _run_emulator_sync(_get_state_dict)
    if _explored_maps is not None:
        with contextlib.suppress(Exception):
            target = _explored_maps.current_map_id
            if target is not None and _explored_maps.knows(target):
                payload["map_summary"] = _explored_maps.summary(target)
    return payload


#: The whole-map picture. It is served like a frame but the map store writes
#: it, not the frame writer, so it is not in AgentRuntime.artifacts.
MAP_ARTIFACT_KEY = "latest_map"
MAP_ARTIFACT_FILENAME = "latest_map.png"

#: (map id, store revision) of whatever latest_map.png currently shows.
_map_image_state: Optional[tuple] = None


def _map_artifact_path() -> Optional[Path]:
    if _runtime is None:
        return None
    return _runtime.workspace_dir / MAP_ARTIFACT_FILENAME


def _refresh_map_image(map_id: Optional[int] = None) -> Optional[Path]:
    """Redraw latest_map.png, but only when the store has learned something."""
    global _map_image_state
    if _explored_maps is None:
        return None
    path = _map_artifact_path()
    if path is None:
        return None
    target = map_id if map_id is not None else _explored_maps.current_map_id
    if target is None or not _explored_maps.knows(target):
        return None
    state = (target, _explored_maps.revision)
    if state == _map_image_state and path.exists():
        return path
    try:
        written = _explored_maps.write_image(target, path)
    except Exception as exc:  # noqa: BLE001 — a picture must never fail a request
        print(f"[server] WARNING: map image render failed: {exc}")
        return None
    if written is None:
        return None
    _map_image_state = state
    return written


def _artifact_urls_from_paths(artifacts: Optional[dict]) -> dict:
    urls: dict[str, str] = {}
    for key in artifacts or {}:
        urls[key] = f"/artifacts/{key}"
    return urls


def _public_artifact_paths(artifacts: Optional[dict]) -> dict:
    allowlist = (
        "latest_frame",
        "latest_frame_annotated",
        "live_frame",
        "live_frame_annotated",
        "turn_context_json",
    )
    return {
        key: value
        for key, value in ((key, (artifacts or {}).get(key)) for key in allowlist)
        if value
    }


def _warp_step_direction(coord: dict, dimensions: dict) -> Optional[str]:
    """Which way to walk to trigger a warp on the map boundary.

    Boundary warps are the ones that strand an agent: the tile beyond is off-map,
    so the overlay paints it blocked and the model concludes the exit is a wall.
    Interior doors are ambiguous from coordinates alone, so they get no hint.
    """
    x, y = coord.get("x"), coord.get("y")
    width, height = dimensions.get("width"), dimensions.get("height")
    if y == 0:
        return "up"
    if height and y == height - 1:
        return "down"
    if x == 0:
        return "left"
    if width and x == width - 1:
        return "right"
    return None


def _observation_summary(bundle: Optional[dict]) -> dict:
    """Where the player is and what it may do next — the whole model-facing payload.

    Everything else the model needs is in the two workspace frames that the
    runtime refresh has just rewritten.
    """
    bundle = bundle or {}
    state = bundle.get("state") or {}
    player = state.get("player") or {}
    position = player.get("position") or {}
    battle = state.get("battle") or {}
    snapshot = (bundle.get("navigation") or {}).get("snapshot") or {}
    screen_text = bundle.get("screen_text") or {}
    summary = {
        "map": (state.get("map") or {}).get("map_name"),
        "x": position.get("x"),
        "y": position.get("y"),
        "facing": player.get("facing"),
        "moves": list(snapshot.get("valid_moves") or []),
        "mode": screen_text.get("ui_mode"),
        "dialog": bool(state.get("dialog_active") or (state.get("dialog") or {}).get("active")),
        "battle": bool(battle.get("in_battle")),
    }
    # Lead Pokemon HP. Without it the model has to spend a /state call to find out
    # it is about to faint, which it will not do unprompted.
    party = state.get("party") or []
    if party:
        lead = party[0] or {}
        if lead.get("max_hp"):
            summary["hp"] = f"{lead.get('hp')}/{lead.get('max_hp')}"

    # What press_a would hit. "object" is an NPC or item ball, "sign" is readable.
    # Anything else is scenery and not worth a button.
    interaction = snapshot.get("interaction") or state.get("interaction") or {}
    if interaction.get("kind") in ("object", "sign"):
        summary["faces"] = interaction["kind"]

    # Standing on a warp is the one board state where the next step changes maps.
    # Say where it leads and which way to step: at a map edge the destination tile
    # renders as blocked, so "walk into the wall" is unguessable without being told.
    position_pair = (position.get("x"), position.get("y"))
    for warp in snapshot.get("warps") or []:
        coord = warp.get("coord") or warp
        if (coord.get("x"), coord.get("y")) != position_pair:
            continue
        summary["on_warp"] = True
        hint = {}
        target = MAP_NAMES.get(warp.get("target_map_id"))
        if target and target != "???":
            hint["to"] = target
        step = _warp_step_direction(coord, snapshot.get("map_dimensions") or {})
        if step:
            hint["step"] = step
        if hint:
            summary["warp"] = hint
        break

    # A battle screen has no position, no facing and no legal walk directions, and
    # reporting empty ones reads as "nothing is possible". Report the fight instead:
    # who you are fighting and what you can hit them with.
    if summary["battle"]:
        for key in ("x", "y", "facing", "moves", "on_warp", "warp", "faces"):
            summary.pop(key, None)
        enemy = (battle.get("enemy") or {}) if isinstance(battle, dict) else {}
        if enemy.get("species"):
            types = "/".join(enemy.get("types") or [])
            summary["enemy"] = (
                f"{enemy['species']} L{enemy.get('level')} "
                f"{enemy.get('hp')}/{enemy.get('max_hp')}" + (f" ({types})" if types else "")
            )
        if party:
            lead = party[0] or {}
            raw_moves = lead.get("moves") or []
            move_names = [m["name"] for m in raw_moves if isinstance(m, dict) and m.get("name")]
            move_names = move_names or [m for m in raw_moves if isinstance(m, str)]
            if move_names:
                summary["your_moves"] = move_names

        # Which menu is open and which entry A would fire. Perception, not advice:
        # the move cursor remembers where it was last turn and wraps at both ends,
        # so "the cursor starts on the first move" is simply not true.
        menu = state.get("battle_menu") or {}
        if menu.get("menu"):
            summary["menu"] = menu["menu"]
        if menu.get("highlighted"):
            summary["highlighted"] = menu["highlighted"]

    text = str(screen_text.get("text") or "").strip()
    if text:
        summary["screen_text"] = text[:SCREEN_TEXT_LIMIT]
    return summary


#: Two visits is a corridor you walked back down. Three is a loop.
HERE_BEFORE_THRESHOLD = 3


def _annotate_batch_outcome(summary: dict, outcome: Optional[dict]) -> None:
    """Report what the batch achieved, not just where it ended.

    Without this the agent cannot tell a 16-step walk from one step and fifteen
    presses of its face against a tree.
    """
    if not outcome or summary.get("battle"):
        return
    moved = outcome.get("moved")
    if moved is None:  # no walk actions in the batch — nothing to report
        return
    summary["moved"] = moved
    if outcome.get("blocked_after") is not None:
        summary["blocked_after"] = outcome["blocked_after"]


def _annotate_explored_map(summary: dict, bundle: Optional[dict]) -> None:
    """The one thing the frame cannot show: you have stood here before.

    Nothing that steers goes in here. Where to go next is the agent's job; the
    harness gives it perception — the frame, and `GET /map` when it asks.
    """
    if _explored_maps is None or summary.get("battle"):
        return
    snapshot = ((bundle or {}).get("navigation") or {}).get("snapshot") or {}
    position = snapshot.get("player_position") or {}
    map_id, x, y = snapshot.get("map_id"), position.get("x"), position.get("y")
    if map_id is None or x is None or y is None:
        return
    try:
        visits = _explored_maps.visit_count(int(map_id), int(x), int(y))
        if visits >= HERE_BEFORE_THRESHOLD:
            summary["here_before"] = visits - 1
    except Exception as exc:  # noqa: BLE001 — hints must never fail an action
        print(f"[server] WARNING: explored-map hints failed: {exc}")


def _make_runtime_save_event(name: str, path: Path, source: str, reason: str) -> dict:
    return {
        "name": name,
        "path": str(path),
        "source": source,
        "reason": reason,
        "notes": [],
    }


def _navigation_payload_sync() -> Optional[dict]:
    """Serialize the live collision window that the annotated frame overlay draws."""
    if _emulator is None:
        return None
    try:
        snapshot = _emulator.get_navigation_snapshot(_reader)
    except NotImplementedError:
        return None
    except Exception:
        return None
    payload = snapshot.to_dict()
    _record_explored_map(payload)
    return {"snapshot": payload}


def _record_explored_map(snapshot: dict) -> None:
    """Fold one snapshot into the persistent map. Never breaks the caller."""
    global _explored_last_save_at, _explored_records_since_save
    if _explored_maps is None:
        return
    try:
        _explored_maps.record(snapshot)
        _explored_records_since_save += 1
        now = time.monotonic()
        due = (
            _explored_records_since_save >= EXPLORED_SAVE_EVERY_RECORDS
            or now - _explored_last_save_at >= EXPLORED_SAVE_INTERVAL_SECONDS
        )
        if due:
            _explored_maps.save()
            _explored_last_save_at = now
            _explored_records_since_save = 0
        _refresh_map_image()
    except Exception as exc:  # noqa: BLE001 — map memory must never fail a request
        print(f"[server] WARNING: explored-map update failed: {exc}")


def _refresh_agent_bundle_sync(
    *,
    reason: str,
    source: str,
    requested_actions: Optional[list[str]] = None,
    explicit_save: Optional[dict] = None,
) -> Optional[dict]:
    if _runtime is None:
        return None
    return _runtime.refresh(
        emulator=_emulator,
        state=_get_state_dict(),
        navigation=_navigation_payload_sync(),
        reason=reason,
        source=source,
        requested_actions=requested_actions,
        explicit_save=explicit_save,
    )


def _sync_live_artifacts_sync() -> Optional[dict]:
    if _runtime is None or _emulator is None:
        return None
    return _runtime.sync_live_view(
        emulator=_emulator,
        state=_get_state_dict(),
        navigation=_navigation_payload_sync(),
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


async def _refresh_and_broadcast(
    *,
    reason: str,
    source: str,
    requested_actions: Optional[list[str]] = None,
    explicit_save: Optional[dict] = None,
) -> dict:
    """Rewrite the workspace frames, emit the resulting events, return the bundle."""
    result = await _run_emulator_sync(
        _refresh_agent_bundle_sync,
        reason=reason,
        source=source,
        requested_actions=requested_actions,
        explicit_save=explicit_save,
    )
    await _broadcast_runtime_refresh(result)
    return (result or {}).get("bundle") or {}


# A runaway agent-written loop can drive /action thousands of times a minute.
# Normal play sends a couple of batches per minute, so this cap only ever bites
# a loop that is not looking at its own results.
ACTION_RATE_WINDOW_SECONDS = 60.0
ACTION_RATE_MAX_CALLS = 60
_action_call_times: deque[float] = deque(maxlen=ACTION_RATE_MAX_CALLS * 2)


def _check_action_rate() -> None:
    now = time.monotonic()
    while _action_call_times and now - _action_call_times[0] > ACTION_RATE_WINDOW_SECONDS:
        _action_call_times.popleft()
    if len(_action_call_times) >= ACTION_RATE_MAX_CALLS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"More than {ACTION_RATE_MAX_CALLS} action batches in "
                f"{int(ACTION_RATE_WINDOW_SECONDS)}s. You are almost certainly in a loop that "
                "never checks whether the player moved. A blocked move returns the same "
                "position, so a 'walk until position changes' loop never ends. Stop, read a "
                "frame, and pick a different direction."
            ),
        )
    _action_call_times.append(now)


def _reject_unsafe_battle_actions(actions: list[str]) -> None:
    """A battle menu reads as an active dialog, so A-mashing confirms menu entries.

    Pressing A until "the dialog" clears moves through FIGHT -> ITEM -> the bag and
    picks whatever is highlighted, which is indistinguishable from the agent trying
    to use an item on purpose. Make it a refusal with an explanation instead.
    """
    if "a_until_dialog_end" not in actions:
        return
    state = _get_state_dict()
    if not ((state.get("battle") or {}).get("in_battle")):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "a_until_dialog_end is unsafe in battle: the battle menu counts as an open "
            "dialog, so this confirms menu entries and opens the bag. Press A once to "
            "advance battle text, or attack by name with POST /battle/fight — the move "
            "cursor remembers where it was left last turn and wraps, so two A presses "
            "are not 'use my first move'."
        ),
    )


async def _run_actions(actions: list[str], *, source: str, reason: str) -> dict:
    """Execute one batch of actions with the standard before/after bookkeeping.

    Returns the executed count plus the observation bundle produced afterwards.
    """
    _check_action_rate()
    await _run_emulator_sync(_reject_unsafe_battle_actions, actions)
    state_before = await _run_emulator_sync(_get_state_dict)
    await _record_and_broadcast(
        "action",
        {"actions": actions, "source": source, "state_before": state_before},
    )
    outcome = await _run_emulator_sync(_execute_action_batch_sync, actions)
    executed = outcome["executed"]
    bundle = await _refresh_and_broadcast(
        reason=reason,
        source=source,
        requested_actions=actions,
    )
    state_after = bundle.get("state") or await _run_emulator_sync(_get_state_dict)
    await _record_and_broadcast(
        "action_result",
        {
            "actions": actions,
            "actions_executed": executed,
            "source": source,
            "state_after": state_after,
            "feedback": bundle.get("recent_action"),
            "state_delta": bundle.get("state_delta"),
            "objective_status": (bundle.get("objective") or {}).get("current"),
            "stuck_signal": bundle.get("stuck"),
            "screen_text": bundle.get("screen_text"),
        },
    )
    return {"actions_executed": executed, "bundle": bundle, "outcome": outcome}


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------


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


#: Walking is the only action whose success or failure is invisible on screen.
#: A blocked step still "succeeds" — the game just puts you back where you were.
WALK_DIRECTIONS = ("up", "down", "left", "right")

#: Beyond this a position change is a warp or a cutscene, not a step taken.
MAP_TRANSITION_TILES = 4


def _is_walk_action(action_str: str) -> bool:
    parts = action_str.strip().lower().split("_")
    return len(parts) >= 2 and parts[0] == "walk" and parts[1] in WALK_DIRECTIONS


def _player_tile_sync() -> Optional[tuple[int, int]]:
    """Read just the player's tile. One RAM read, not a whole game state."""
    if _reader is None:
        return None
    try:
        position = (_reader.read_player() or {}).get("position") or {}
        x, y = position.get("x"), position.get("y")
    except Exception:  # noqa: BLE001 — sampling must never fail a batch
        return None
    if x is None or y is None:
        return None
    return int(x), int(y)


def _execute_action_batch_sync(actions: list[str]) -> dict:
    """Run a batch, sampling the player's tile around every walk action.

    A blocked move in Gen 1 returns the same position rather than an error, so
    `walk_left` x16 into a tree burns fifteen buttons in silence. Sampling
    between actions is the only way to say which one stopped mattering — and it
    has to be between actions, because a batch that walks right then left nets
    zero movement without ever having been blocked.
    """
    executed = 0
    moved = 0
    blocked_after: Optional[int] = None
    walked_at_all = False
    tile = _player_tile_sync() if any(_is_walk_action(item) for item in actions) else None

    for index, action_str in enumerate(actions, start=1):
        is_walk = _is_walk_action(action_str)
        _execute_action_sync(action_str)
        executed += 1
        if not is_walk or tile is None:
            continue
        walked_at_all = True
        after = _player_tile_sync()
        if after is None:
            continue
        steps = abs(after[0] - tile[0]) + abs(after[1] - tile[1])
        tile = after
        if steps == 0:
            if blocked_after is None and index < len(actions):
                blocked_after = index
            continue
        moved += 1 if steps > MAP_TRANSITION_TILES else steps

    return {
        "executed": executed,
        "moved": moved if walked_at_all else None,
        "blocked_after": blocked_after,
    }


# ---------------------------------------------------------------------------
# Battle sequencing
# ---------------------------------------------------------------------------

#: Normalising the 2x2 top menu onto FIGHT. Safe from any of the four entries
#: because that menu does not wrap: Up on the top row stays on the top row, Left
#: in the left column stays in the left column. Verified from a cursor on RUN.
BATTLE_MENU_NORMALISE = ("press_up", "press_up", "press_left", "press_left")

#: B presses to spend getting back to the top battle menu. A whole battle intro
#: from a cold save takes ten, so this cap only bites when the screen is showing
#: something the command cannot drive at all.
BATTLE_NORMALISE_MAX_PRESSES = 24

#: A presses to spend letting the turn play out before answering.
BATTLE_SETTLE_MAX_PRESSES = 8


def battle_fight_keys(target_index: int, current_index: int, move_count: int) -> list[str]:
    """The exact buttons that select move ``target_index`` and confirm it.

    ``current_index`` is where the move cursor will open — the game remembers it
    from last turn — and the list *wraps at both ends*, so there is no direction
    you can hold to reach a known entry. Down alone therefore covers the whole
    list, and the step count is the only thing that has to be right.
    """
    if move_count <= 0:
        raise ValueError("no moves to choose from")
    if not 0 <= target_index < move_count:
        raise ValueError(f"move index {target_index} is outside 0..{move_count - 1}")
    steps = (target_index - current_index) % move_count
    return [
        *BATTLE_MENU_NORMALISE,  # onto FIGHT, wherever the cursor was
        "press_a",  # open the move list
        *["press_down"] * steps,  # walk it to the move we asked for
        "press_a",  # confirm
    ]


def battle_run_keys() -> list[str]:
    """The exact buttons that flee, from any of the four top-menu entries."""
    return ["press_down", "press_down", "press_right", "press_right", "press_a"]


def _battle_state_sync() -> dict:
    return (_reader.read_battle() if _reader is not None else None) or {}


def _require_battle_sync() -> dict:
    battle = _battle_state_sync()
    if not battle.get("in_battle"):
        raise HTTPException(
            status_code=400,
            detail="Not in a battle. Nothing to attack and nothing to run from.",
        )
    return battle


def _move_name(move_id: int) -> str:
    return MOVE_NAMES.get(move_id, f"???({move_id})")


def _normalise_to_battle_menu_sync() -> int:
    """Press B until the top battle menu is up. Returns the presses spent.

    Two traps are buried here, both of which cost a session to find:

    * ``read_dialog()["active"]`` is **not** a proxy for "the game is waiting for
      my turn". During the battle intro it goes true, then *false* for about a
      second while the sprites slide in, then true again on "Wild X appeared!".
      A loop that waits on the dialog flag exits into the middle of that
      animation, and every button pressed after it is eaten. Watch the menu
      signature instead — that is what ``at_battle_top_menu`` is for.
    * Some ``battle-entry`` save states read ``in_battle: True`` while actually
      sitting in *post*-victory dialogue with a stale ``wIsInBattle``. "In a
      battle" and "the battle menu is up" are two different questions; ask both.

    B is inert on the top battle menu itself, so over-running this loop costs
    nothing: it only advances text and backs out of submenus.
    """
    for pressed in range(BATTLE_NORMALISE_MAX_PRESSES):
        if _reader.at_battle_top_menu():
            return pressed
        _execute_action_sync("press_b")
        _emulator.tick(30)
    if _reader.at_battle_top_menu():
        return BATTLE_NORMALISE_MAX_PRESSES
    raise HTTPException(
        status_code=409,
        detail=(
            f"The battle menu never appeared after {BATTLE_NORMALISE_MAX_PRESSES} B presses, "
            "though the game still reports a battle. Usually that means the fight is already "
            "over and this is the text that follows it — the in-battle flag lags behind the "
            "screen. Press A to clear it, or read the frame."
        ),
    )


def _settle_battle_sync() -> None:
    """Let the turn play out far enough that the answer describes a real screen.

    Stops the moment the top battle menu comes back, so it never presses A on a
    menu — which is the bug this whole endpoint exists to prevent.
    """
    for _ in range(BATTLE_SETTLE_MAX_PRESSES):
        _emulator.tick(30)
        if _reader.at_battle_top_menu():
            return
        if not _battle_state_sync().get("in_battle"):
            return
        _execute_action_sync("press_a")


def resolve_move_name(name: str, moves: list[dict]) -> int:
    """Index of the move called ``name``: exact match first, then unique prefix."""
    known = ", ".join(str(move.get("name")) for move in moves) or "nothing"
    wanted = name.strip().lower()
    if not wanted:
        raise HTTPException(status_code=400, detail=f"No move named. Known moves: {known}.")
    names = [str(move.get("name") or "").lower() for move in moves]
    if wanted in names:
        return names.index(wanted)
    hits = [index for index, candidate in enumerate(names) if candidate.startswith(wanted)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        matched = ", ".join(str(moves[index].get("name")) for index in hits)
        raise HTTPException(
            status_code=400,
            detail=f"{name!r} matches more than one move: {matched}. Spell out more of it.",
        )
    raise HTTPException(status_code=400, detail=f"No move called {name!r}. Known moves: {known}.")


def _battle_fight_sync(name: str) -> dict:
    """Select a move by name and confirm it, then check it really was that move."""
    _require_battle_sync()
    moves = _reader.read_battle_moves()
    if not moves:
        raise HTTPException(
            status_code=409,
            detail="No moves are readable yet — the battle is not far enough along to attack.",
        )
    index = resolve_move_name(name, moves)
    move = moves[index]
    if move.get("pp") == 0:
        listing = ", ".join(f"{other['name']} {other['pp']}PP" for other in moves)
        raise HTTPException(
            status_code=400,
            detail=f"{move['name']} has no PP left. Known moves: {listing}.",
        )

    # Check the cursor *before* confirming: a wrong cursor here costs nothing to
    # fix, whereas a wrong cursor after A has already spent the turn. Hence one
    # retry before the confirming press, and none after it.
    retried = False
    _normalise_to_battle_menu_sync()
    keys = battle_fight_keys(index, _reader.remembered_move_index(), len(moves))
    for key in keys[:-1]:
        _execute_action_sync(key)
    if _reader.selected_move_id() != move["id"]:
        retried = True
        _normalise_to_battle_menu_sync()
        keys = battle_fight_keys(index, _reader.remembered_move_index(), len(moves))
        for key in keys[:-1]:
            _execute_action_sync(key)
        landed = _reader.selected_move_id()
        if landed != move["id"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Could not put the cursor on {move['name']}: it is sitting on "
                    f"{_move_name(landed)} instead. Nothing was confirmed, so no turn "
                    "was spent. Read the frame."
                ),
            )

    _execute_action_sync(keys[-1])
    confirmed = _reader.selected_move_id()
    if confirmed != move["id"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Asked for {move['name']} but the game confirmed {_move_name(confirmed)}. "
                "The turn has already been spent — read the frame before acting again."
            ),
        )
    _settle_battle_sync()
    return {"used": move["name"], "actions": keys, "retried": retried}


def _battle_run_sync() -> dict:
    """Move the cursor to RUN from wherever it is, confirm, and say whether it worked."""
    _require_battle_sync()
    _normalise_to_battle_menu_sync()
    keys = battle_run_keys()
    for key in keys:
        _execute_action_sync(key)
    _settle_battle_sync()
    return {"actions": keys, "fled": not _battle_state_sync().get("in_battle")}


async def _run_battle_sequence(intent: dict, func, *args) -> dict:
    """One battle command, with the bookkeeping an /action batch would get."""
    _check_action_rate()
    state_before = await _run_emulator_sync(_get_state_dict)
    await _record_and_broadcast(
        "action",
        {"actions": [], "source": "battle", "intent": intent, "state_before": state_before},
    )
    outcome = await _run_emulator_sync(func, *args)
    bundle = await _refresh_and_broadcast(
        reason="battle_command",
        source="battle",
        requested_actions=outcome["actions"],
    )
    state_after = bundle.get("state") or await _run_emulator_sync(_get_state_dict)
    await _record_and_broadcast(
        "action_result",
        {
            "actions": outcome["actions"],
            "actions_executed": len(outcome["actions"]),
            "source": "battle",
            "intent": intent,
            "state_after": state_after,
            "screen_text": bundle.get("screen_text"),
        },
    )
    return {"outcome": outcome, "bundle": bundle}


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def configure(config: GameConfig):
    """Set server configuration (call before app startup)."""
    global _config
    _config = config


def _endpoint_banner_lines() -> list[str]:
    """One line per registered route, read off the router so it cannot drift."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    hidden = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or path in hidden:
            continue
        if isinstance(route, Mount):
            verbs = ["GET"]
        elif getattr(route, "methods", None):
            verbs = sorted(set(route.methods) - {"HEAD", "OPTIONS"})
        else:
            verbs = ["WS"]
        text = (getattr(route, "summary", None) or getattr(route, "description", "") or "").strip()
        summary = text.splitlines()[0] if text else ""
        for verb in verbs:
            key = (verb, path)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"[server]   {verb:<4} {path:<24} {summary}".rstrip())
    return lines


async def _startup():
    global \
        _emulator, \
        _reader, \
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
        _live_artifact_last_sync_at, \
        _explored_maps, \
        _map_image_state, \
        _explored_last_save_at, \
        _explored_records_since_save
    _loop = asyncio.get_running_loop()
    _start_time = time.time()
    _emulator_lock = asyncio.Lock()
    _realtime_ticks = 0
    _realtime_last_tick_at = None
    _live_artifact_last_sync_at = None
    _explored_last_save_at = time.monotonic()
    _explored_records_since_save = 0

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
    workspace_dir = (
        Path(_config.agent_workspace_dir).expanduser().resolve()
        if _config.agent_workspace_dir
        else (data_dir / "agent_workspace").resolve()
    )
    # Lives in data_dir, not the workspace, so it outlives both a server restart
    # and the fresh Pi sessions the watchdog starts every ~110k tokens.
    _explored_maps = ExploredMaps(data_dir / "explored_maps.json")
    print(f"[server] Explored maps: {len(_explored_maps.map_ids())} known — {_explored_maps.path}")
    _map_image_state = None
    _runtime = AgentRuntime(
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        # Lets the annotated frame shade the ground already walked.
        visited_lookup=_explored_maps.visited,
    )
    # SEAM: the annotated frame insets a mini-map drawn from these tile sets.
    _runtime.map_grid_lookup = _explored_maps.grid
    _supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url=f"http://127.0.0.1:{_config.port}",
        event_sink=_record_existing_event_and_broadcast,
        stream_sink=broadcast,
        artifact_paths=_runtime_artifact_paths,
        critic_context=_critic_context,
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

    try:
        await _refresh_and_broadcast(reason="startup_refresh", source="observe")
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
    for line in _endpoint_banner_lines():
        print(line)


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
    if _explored_maps is not None:
        _explored_maps.save()


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
    runtime = _ensure_runtime()
    path = runtime.artifacts.get(artifact_key)
    if path is None and artifact_key == MAP_ARTIFACT_KEY:
        path = _refresh_map_image() or _map_artifact_path()
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
    runtime = _ensure_runtime()
    payload = runtime.dashboard_state()
    artifacts = _public_artifact_paths(payload.get("artifacts") or {})
    payload["artifacts"] = artifacts
    payload["artifact_urls"] = _artifact_urls_from_paths(artifacts)
    payload["pi_supervisor"] = _supervisor.state_snapshot() if _supervisor is not None else {}
    payload["server_runtime"] = _server_runtime_snapshot()
    return JSONResponse(content=payload)


@app.get("/dashboard/history")
async def dashboard_history(limit: int = 200):
    """Structured recent dashboard/agent events."""
    runtime = _ensure_runtime()
    limit = max(1, min(limit, 1000))
    return {"events": runtime.history(limit)}


@app.get("/supervisor/state")
async def supervisor_state():
    """Current Pi supervisor status and recent event stream."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    return JSONResponse(content=_compact_supervisor_status(_supervisor.state_snapshot()))


@app.get("/supervisor/stream")
async def supervisor_stream(after: int = 0, limit: int = 1000):
    """Ordered model-time log: tools, thinking, text, prompts and harness events."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    return JSONResponse(content=_supervisor.stream_since(after=after, limit=limit))


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


@app.post("/supervisor/steer")
async def supervisor_steer(req: PiSupervisorSteerRequest):
    """Inject an operator message into the live Pi session.

    400 when the message is empty or over the length cap, 409 when no session is
    live to receive it, 503 when the supervisor was never initialised.
    """
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Pi supervisor is not initialised")
    try:
        entry = await _supervisor.send_operator_message(req.message)
    except HTTPException:
        raise
    except NoLiveSessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Supervisor steer error: {exc}")
    return {
        "success": True,
        "entry": entry,
        "supervisor": _compact_supervisor_status(_supervisor.state_snapshot()),
    }


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


@app.get("/state")
async def get_state():
    """Full game state JSON."""
    _ensure_emulator()
    try:
        state = await _run_emulator_sync(_get_state_dict)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading state: {e}")


@app.get("/map")
async def get_map(map_id: Optional[int] = None):
    """Shape and coverage of a whole map, plus a PNG of it to look at.

    A pull tool for when you are lost: it is never pushed into /action. The
    picture carries the spatial shape; this payload stays small enough to ask
    for often.
    """
    if _explored_maps is None:
        raise HTTPException(status_code=503, detail="Explored-map memory not initialised")
    target = map_id if map_id is not None else _explored_maps.current_map_id
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="No map recorded yet — take an action, then ask again.",
        )
    if not _explored_maps.knows(target):
        known = ", ".join(str(known_id) for known_id in _explored_maps.map_ids()) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"Map {target} has never been visited. Maps recorded so far: {known}.",
        )
    payload = _explored_maps.summary(target)
    image_path = _refresh_map_image(target)
    if image_path is not None:
        payload["image"] = f"/artifacts/{MAP_ARTIFACT_KEY}"
        payload["image_path"] = str(image_path)
    return payload


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
    """Execute game actions, rewrite the workspace frames, report the new position."""
    _ensure_emulator()
    try:
        result = await _run_actions(req.actions, source="action", reason="actions_executed")
        summary = _observation_summary(result["bundle"])
        _annotate_batch_outcome(summary, result.get("outcome"))
        _annotate_explored_map(summary, result["bundle"])
        return {"actions_executed": result["actions_executed"], **summary}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action error: {e}")


@app.post("/battle/fight")
async def battle_fight(req: BattleFightRequest):
    """Attack with a named move, whatever the menu cursor was left on."""
    _ensure_emulator()
    try:
        result = await _run_battle_sequence({"fight": req.move}, _battle_fight_sync, req.move)
        outcome = result["outcome"]
        payload = {"used": outcome["used"], **_observation_summary(result["bundle"])}
        if outcome["retried"]:
            payload["retried"] = True
        return payload
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Battle error: {e}")


@app.post("/battle/run")
async def battle_run():
    """Flee, whatever the menu cursor was left on."""
    _ensure_emulator()
    try:
        result = await _run_battle_sequence({"run": True}, _battle_run_sync)
        return {
            "fled": result["outcome"]["fled"],
            **_observation_summary(result["bundle"]),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Battle error: {e}")


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
        bundle = await _refresh_and_broadcast(
            reason=f"manual_save:{req.name}",
            source="save",
            explicit_save=_make_runtime_save_event(
                req.name,
                save_path,
                source="manual",
                reason="manual_save",
            ),
        )
        return {
            "success": True,
            "save": {"name": req.name, "path": str(save_path)},
            **_observation_summary(bundle),
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
        await _run_emulator_sync(_emulator.load_state, str(save_path))
        bundle = await _refresh_and_broadcast(
            reason=f"manual_load:{req.name}",
            source="load",
        )
        state_after = bundle.get("state") or await _run_emulator_sync(_get_state_dict)
        await _record_and_broadcast("load", {"name": req.name, "path": str(save_path)})
        await broadcast({"type": "state_update", "reason": "load", "state": state_after})
        return {
            "success": True,
            "save": {"name": req.name, "path": str(save_path)},
            **_observation_summary(bundle),
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
