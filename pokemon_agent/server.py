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
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator
from starlette.routing import Mount

from pokemon_agent import capabilities
from pokemon_agent.agent_runtime import AgentRuntime
from pokemon_agent.coordinator import (
    MAX_FRAMES_PER_BATCH,
    ActionLimitError,
    EmulatorCoordinator,
    UnknownActionError,
    presses_for_action,
    validate_action_batch,
)
from pokemon_agent.explored_map import ExploredMaps
from pokemon_agent.guides import GuideLog
from pokemon_agent.intervention_loop import (
    DEFAULT_SLOT_BASE_URL,
    DEFAULT_SLOT_FILENAME,
    DEFAULT_SLOT_MODEL,
    JOURNAL_FILENAME,
    InterventionRunner,
    build_slot_client,
    pi_thinker,
)
from pokemon_agent.memory.red import MAP_NAMES, MOVE_NAMES
from pokemon_agent.milestones import MilestoneTracker
from pokemon_agent.pi_supervisor import NoLiveSessionError, PiSupervisor
from pokemon_agent.pockets import PocketGraph
from pokemon_agent.progress import AUTO_SAVE_PREFIX
from pokemon_agent.run_recorder import RunRecorder, receipt_from_batch
from pokemon_agent.saves import (
    SaveNameError,
    list_save_files,
    resolve_save_path,
    validate_save_name,
)
from pokemon_agent.world import World, reachable_region

__version__ = "0.1.0"

SCREEN_TEXT_LIMIT = 160

#: What the screen reader emits when it can see a dialog box but cannot read any
#: words out of it. `agent_runtime` already filters on this prefix; the action
#: payload did not, so it shipped the placeholder 660 times -- 36,300 bytes
#: restating the `dialog` flag printed beside it.
DIALOG_PLACEHOLDER_PREFIX = "Dialog box visible"

#: Why the payload carries no walk directions on a frame that cannot take a step.
#: `moves` and `run` are answers to "where may I walk *now*", and on these two
#: frames the d-pad never reaches the player: in a battle it drives the battle
#: menu, and under an open box it works the box or is swallowed outright.
#:
#: Measured with Oak's dialog up: the payload offered `moves: ["down"]` and
#: `run down:4`, two `walk_down` actions moved nothing, and the answer came back
#: `moved: 0, blocked_after: 1` -- the harness reporting walkable ground as a
#: wall. Saying which of the two it is costs one line and cannot be misread.
NO_WALK_IN_BATTLE = "no walking in a battle: the d-pad drives the battle menu"
NO_WALK_IN_BOX = "no walking while a box is open: the d-pad works the box, not the player"

#: Why a battle frame reports coordinates but no facing.
#:
#: The coordinates are a fact there -- four wild encounters walked into in Mt.
#: Moon 1F all read the same tile during the fight as the overworld read after
#: fleeing. The facing byte is not: an encounter interrupts the step that started
#: it, so 0xC109 still holds the direction from *before* that step. Two of those
#: four battle frames said "right" and "up" for steps that were walk_up and
#: walk_down, and the overworld came back facing up and down. Reporting the stale
#: value would be the worst of the three things this payload can do with a field.
#: Kept short on purpose: a battle runs several `act` calls and this line is paid
#: on every one of them.
FACING_UNREAD_IN_BATTLE = "facing unread in a battle: the byte is stale from before the encounter"


#: Environment switch for the intervention loop, since the launcher scripts
#: build a GameConfig they do not parameterise.
INTERVENTIONS_ENV_VAR = "POKEMON_AGENT_INTERVENTIONS"
_TRUTHY = {"1", "true", "yes", "on"}


def _interventions_flag(config: Optional["GameConfig"]) -> bool:
    """Whether the harness may interrupt the player. Off unless told otherwise."""

    if config is not None and config.interventions_enabled is not None:
        return bool(config.interventions_enabled)
    return os.environ.get(INTERVENTIONS_ENV_VAR, "").strip().lower() in _TRUTHY


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

    #: Whether the harness may stop the player and think. Off unless something
    #: says otherwise, because firing an intervention swaps the player's whole
    #: KV cache out to disk and a live run is somebody's data. ``None`` defers to
    #: the ``POKEMON_AGENT_INTERVENTIONS`` environment variable, which is what the
    #: launch scripts actually set; the flag is read once, at startup, and can be
    #: turned on for a run already in flight because the run itself is adopted
    #: from disk rather than restarted.
    interventions_enabled: Optional[bool] = None
    slot_base_url: str = DEFAULT_SLOT_BASE_URL
    slot_model: str = DEFAULT_SLOT_MODEL
    slot_filename: str = DEFAULT_SLOT_FILENAME


class ActionRequest(BaseModel):
    """Body for POST /action."""

    actions: list[str]


class SaveRequest(BaseModel):
    """Body for POST /save and POST /load.

    The name is validated here, once, so no endpoint can be written that forgets
    to: a save name is a plain file name and never a path. ``../escaped`` used
    to report success and write outside the saves directory entirely.
    """

    name: str

    @field_validator("name")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        return validate_save_name(value)


class GotoRequest(BaseModel):
    """Body for POST /goto — a map to reach, or a tile on the current map."""

    target: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None


class SimRequest(BaseModel):
    """Body for POST /sim."""

    actions: list[str]


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

# The static map graph behind GET /route and POST /goto, and the record of which
# guide sections the agent chose to open. Both are read-only as far as the
# emulator is concerned, so neither ever waits on the emulator lock.
_world: Optional[World] = None
_guide_log: Optional[GuideLog] = None

#: Buttons sent since startup. A fallback only: once a run is open its own
#: receipts are the press total, and they outlive this process.
_press_count: int = 0

#: The run's receipt writer, and the loop that reads those receipts back to
#: decide whether the player should stop and think. Both are created at startup;
#: the recorder is handed to the supervisor, which owns the run lifecycle.
_run_recorder: Optional[RunRecorder] = None
_interventions: Optional[InterventionRunner] = None
_intervention_task: Optional[asyncio.Task] = None

#: Why the emulator was never created, if it was not. An unsupported ROM is
#: reported here rather than as an ImportError three steps later.
_startup_error: Optional[str] = None

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


#: Games with a working memory reader. FireRed is detected and named but its
#: reader is a stub whose every method raises, so it is not playable.
SUPPORTED_GAME_TYPES = ("red",)

#: Games the extension table names but cannot actually drive.
UNSUPPORTED_GAME_TYPES = {
    "firered": (
        "Pokemon FireRed is not supported: the GBA memory reader is a stub, so the "
        "server would start an emulator it cannot read a single fact out of. Use a "
        "Pokemon Red or Blue .gb/.gbc ROM."
    ),
}


class UnsupportedGameError(ValueError):
    """A ROM this server cannot actually read, named before anything is created."""


def _detect_game_type(rom_path: str) -> str:
    """Pick reader type based on file extension."""
    ext = Path(rom_path).suffix.lower()
    if ext in (".gb", ".gbc"):
        return "red"
    elif ext == ".gba":
        return "firered"
    raise ValueError(f"Unrecognised ROM extension: {ext}")


def _resolve_game_type(rom_path: str, configured: str) -> str:
    """The game type to run, or raise before an emulator exists.

    The reader import used to be the thing that failed, three steps after the
    emulator had already been created and a window opened. One check, up front,
    with one message.
    """
    game_type = _detect_game_type(rom_path) if configured == "auto" else configured
    if game_type in UNSUPPORTED_GAME_TYPES:
        raise UnsupportedGameError(UNSUPPORTED_GAME_TYPES[game_type])
    if game_type not in SUPPORTED_GAME_TYPES:
        supported = ", ".join(SUPPORTED_GAME_TYPES)
        raise UnsupportedGameError(f"Unknown game type: {game_type}. Supported: {supported}.")
    return game_type


def _ensure_emulator():
    """Raise 503 if the emulator isn't ready."""
    if _emulator is None:
        raise HTTPException(status_code=503, detail="Emulator not initialised")


def _ensure_runtime() -> AgentRuntime:
    """Return the agent runtime, or raise 503 if it isn't ready."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    return _runtime


class _ServerOps:
    """The emulator plumbing, resolved from module state at call time.

    Every attribute is a live lookup rather than something captured at
    construction: the server's emulator, reader, runtime and lock are all
    created during startup and replaced by tests, and a coordinator holding
    stale references would drive a machine nobody else is looking at.
    """

    @property
    def lock(self) -> Optional[asyncio.Lock]:
        return _emulator_lock

    @property
    def emulator(self):
        return _emulator

    @property
    def reader(self):
        return _reader

    @property
    def runtime(self) -> Optional[AgentRuntime]:
        return _runtime

    def state_dict(self) -> dict:
        return _get_state_dict()

    def execute_batch(self, actions: list[str]) -> dict:
        """Run the batch and price it, without letting go of the lock.

        ``presses`` and ``milestones`` are what the run's receipt is built from,
        and both have to be read from the machine this batch just left. Sampling
        them after the transaction would describe whatever the next request did.
        """
        before = _press_count
        outcome = dict(_execute_action_batch_sync(actions))
        outcome["presses"] = max(0, _press_count - before)
        outcome["milestones"] = _milestone_ids_sync()
        return outcome

    def reject_unsafe_battle_actions(self, actions: list[str]) -> None:
        _reject_unsafe_battle_actions(actions)

    def refresh_bundle(self, **kwargs) -> Optional[dict]:
        return _refresh_agent_bundle_sync(**kwargs)


#: One coordinator for the process. It owns no state of its own — the lock and
#: the emulator are looked up through the ops object above — so it survives the
#: server being configured, started and restarted underneath it.
_coordinator = EmulatorCoordinator(_ServerOps())


async def _run_emulator_sync(func, *args, **kwargs):
    """Run one blocking emulator call while holding the emulator lock.

    For reads with no follow-up only. Anything that mutates and then observes
    belongs in a coordinator transaction: the gap between two of these calls is
    exactly where a concurrent request used to change the machine underneath.
    """
    return await _coordinator.run(func, *args, **kwargs)


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
    """Where the dashboard shell lives, or None if it is not installed.

    The dashboard package owns both the answer and the routes; this only guards
    the import, so a build without the dashboard still starts a game server.
    """
    try:
        from pokemon_agent.dashboard import dashboard_static_dir
    except ImportError:
        return None
    return dashboard_static_dir()


def _mount_dashboard_routes() -> Optional[Path]:
    """Put the dashboard on this app, and say where its files came from.

    ``mount_dashboard`` registers /dashboard, /dashboard/ and /dashboard/assets
    and skips whatever is already there, so the server no longer keeps a second
    copy of those two routes that could drift from the page they serve.
    """
    try:
        from pokemon_agent.dashboard import mount_dashboard
    except ImportError:
        return None
    return _get_dashboard_static_dir() if mount_dashboard(app) else None


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
    """Fallback for which way to walk off a warp, from map edges alone.

    Only used when the navigation snapshot has no answer. The engine picks
    between two rules and neither is derivable from coordinates, so this guesses
    the boundary case and stays quiet about interior doors.
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


#: Directions, and the (dx, dy) they move. North is up: walk_up decreases y.
_RUNWAY_STEPS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _runway(snapshot: dict) -> dict:
    """How many tiles the player can walk in each direction before something stops it.

    `moves` already says which directions are legal, and the model answers it by
    stepping one tile and asking again: measured over a real session, 400 `act`
    calls moved a median of one tile each. Every one of those tiles costs a tool
    call and its response never leaves the context window, which is what actually
    fills it -- images are 2% of the prompt, tool text is 98%.

    So `moves` is the wrong fact. "You may go left" invites one step. "You may go
    left seven" invites `left:7`. This is the same shape as `faces` and
    `blocked_after`, both of which changed behaviour: a fact about where the
    player is standing, arriving in the payload it already reads, rather than
    advice it has to remember.

    Counted over the live 10x9 collision window only, so the number is small,
    always true, and never a guess about ground the game has not shown us.
    """
    terrain = snapshot.get("terrain")
    if not terrain:
        return {}
    origin = snapshot.get("window_top_left") or {}
    position = snapshot.get("player_position") or {}
    ox, oy = origin.get("x"), origin.get("y")
    px, py = position.get("x"), position.get("y")
    if None in (ox, oy, px, py):
        return {}

    height = len(terrain)
    width = len(terrain[0]) if height else 0
    blocked = {(sprite.get("x"), sprite.get("y")) for sprite in snapshot.get("sprites") or []}

    # Only directions `moves` already calls legal. The raw terrain grid and
    # get_valid_moves disagree on purpose -- the latter also applies the ledge and
    # warp rules -- and a payload that says "you may not go right" beside
    # "right goes 5" is two answers to one question.
    legal = set(snapshot.get("valid_moves") or _RUNWAY_STEPS)

    runway: dict[str, int] = {}
    for direction, (dx, dy) in _RUNWAY_STEPS.items():
        if direction not in legal:
            continue
        steps = 0
        x, y = px, py
        while True:
            x, y = x + dx, y + dy
            col, row = x - ox, y - oy
            if not (0 <= row < height and 0 <= col < width):
                break  # past the window: unknown, not blocked, so stop counting
            if not terrain[row][col] or (x, y) in blocked:
                break
            steps += 1
        if steps:
            runway[direction] = steps
    return runway


#: Destinations to name in the payload. Celadon has ten and a city's tenth
#: doorway is noise, not navigation.
MAX_EXITS = 4

#: And a byte ceiling, because a count alone does not bound the cost: four of the
#: longest map names in the game ("Fuchsia Bill's Grandpa's House" and friends)
#: come to 171 bytes and push the payload from 257 to 439. The whole point of
#: this payload is that it is small enough to read every turn.
MAX_EXITS_BYTES = 120


def _reachable_tiles(snapshot: dict, start: tuple[int, int]) -> Optional[dict]:
    """Tile -> real step count, for everywhere the player can walk on this map.

    None when the map has not been decoded, in which case a caller has nothing
    better than Manhattan distance and should say so rather than pretend.
    """
    if not (snapshot.get("map_terrain") or {}).get("walkable"):
        return None
    try:
        collision = capabilities.collision_from(snapshot, None)
        region = reachable_region(collision, start)
        return dict(region.distance)
    except Exception:  # noqa: BLE001 — an exit list must never fail a request
        return None


def _exits(snapshot: dict) -> dict:
    """Where this map's warps go, nearest tile per destination.

    The agent has spent ten hours inside Mt. Moon. It made two saves on B2F,
    which warps only back to B1F -- the optional fossil room, a dead end for
    progress -- while the way out sits on B1F at (27,3). The harness has known
    that the whole time: `world.route` answers it instantly and `poke route` was
    called once in five hundred steps.

    That is the advisory pattern that has failed every time in this project, so
    this states what exists rather than offering a lookup. It is deliberately
    *not* a recommendation: it says a warp to Route 4 is at (27,3), not that the
    player should take it. Which exit serves the goal stays the model's call, the
    same way `run` reports how far each direction goes without saying which to
    walk.

    Nearest by Manhattan distance, since a warp the player can see the way to is
    worth more than one on the far side of a wall.
    """
    position = snapshot.get("player_position") or {}
    px, py = position.get("x"), position.get("y")
    if px is None or py is None:
        return {}

    # The map's whole warp table, not the handful inside the 10x9 window. The
    # window is why the first version of this was useless: standing on B1F it
    # listed the two staircases in view and omitted the exit twelve tiles north,
    # which is the only one that mattered. Static map data from the pokered
    # decompilation is complete, constant, and already verified against the ROM.
    map_name = snapshot.get("map_name")
    try:
        from pokemon_agent import gamedata

        record = gamedata.world().get(map_name) or {}
        warps = record.get("warps") or []
        connections = record.get("connections") or {}
    except Exception:
        warps = []
        connections = {}
    if not warps:
        warps = [
            {**(w.get("coord") or w), "to_map": MAP_NAMES.get(w.get("target_map_id"))}
            for w in snapshot.get("warps") or []
        ]

    # Which warps the player can actually walk to, from the decoded floor. A
    # warp in another pocket of this map is not an exit, it is a wall with a
    # door painted on it, and ranking by Manhattan distance advertised exactly
    # those: standing on Mt Moon B1F at (26,15) this named the B2F staircase at
    # (21,17), which is in a different pocket and unreachable, and suppressed
    # (13,27), which is in the same pocket and is the way through. Two of the
    # three exits it listed there could not be walked to.
    reachable = _reachable_tiles(snapshot, (px, py))

    best: dict[str, tuple[int, int, int]] = {}
    for warp in warps:
        x, y = warp.get("x"), warp.get("y")
        target = warp.get("to_map")
        if x is None or y is None or not target or target == "???":
            continue
        if reachable is not None and (x, y) not in reachable:
            continue
        # Real steps where the flood measured them, Manhattan only as a
        # fallback for a map nothing has decoded yet.
        distance = reachable[(x, y)] if reachable is not None else abs(x - px) + abs(y - py)
        if target not in best or distance < best[target][0]:
            best[target] = (distance, x, y)

    ranked = sorted(best.items(), key=lambda item: item[1][0])[:MAX_EXITS]
    exits: dict[str, Any] = {}

    # Edge connections first, because they are the ones that matter and the ones
    # a warp list silently omits. Route 4 reaches Cerulean City -- the goal -- by
    # walking off its east edge, not through a door, so the first version of this
    # told the agent about three ways back into Mt. Moon and nothing about the way
    # out. It sat at x=19 of a 90-wide map for thousands of presses.
    for edge, target in sorted((connections or {}).items()):
        if target:
            exits[str(target)] = f"{edge} edge"

    for target, (_, x, y) in ranked:
        candidate = {**exits, target: [x, y]}
        if len(json.dumps(candidate, separators=(",", ":"))) > MAX_EXITS_BYTES:
            break
        exits = candidate
    return exits


def _warp_exit_hint(snapshot: dict, coord: dict) -> dict:
    """Which way to step off the warp under you, and whether it will fire.

    `warp_exit_directions` comes from the engine's own two rules: the overworld
    tilesets check the tile in front against pokered's warp-carpet lists, the
    rest check whether the player faces the edge of the map. Both beat guessing
    from coordinates.

    The armed flag matters more than it looks. Gen 1 only arms a warp when the
    player *walks onto* it, so a loaded save state that starts on a warp tile
    has an exit that silently does nothing until you step off and back on.
    Reporting the direction without that caveat sends the model into a loop
    pressing a button that cannot work.
    """
    hint: dict = {}
    directions = list(snapshot.get("warp_exit_directions") or [])
    if directions:
        hint["step"] = directions[0]
        if len(directions) > 1:
            hint["steps"] = directions
    else:
        step = _warp_step_direction(coord, snapshot.get("map_dimensions") or {})
        if step:
            hint["step"] = step
    if snapshot.get("warp_exit_armed") is False:
        hint["armed"] = False
        hint["note"] = snapshot.get("warp_exit_note") or (
            "This warp is not armed. Step off the tile and back onto it, then take the exit step."
        )
    return hint


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

    # How far each legal direction actually goes, so a clear stretch is one call
    # rather than one call per tile. See _runway.
    runway = _runway(snapshot)
    if runway:
        summary["run"] = runway

    # Which maps this one leads to, and the nearest tile that gets there. See
    # _exits: what exists, not which one to take.
    exits = _exits(snapshot)
    if exits:
        summary["exits"] = exits

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
        hint.update(_warp_exit_hint(snapshot, coord))
        if hint:
            summary["warp"] = hint
        break

    # A battle or an open box is not a frame you can step from, so the fields that
    # answer "where may I walk now" go, and one line says which frame ate them. An
    # empty `moves` list in their place reads as "nothing is possible", which is
    # the failure this project keeps paying for.
    #
    # The coordinates are not one of those fields and are *not* dropped any more.
    # wXCoord and wYCoord are untouched by either frame: four wild encounters
    # walked into in Mt. Moon 1F each read the same tile during the fight that the
    # overworld read back after fleeing. Blanking them made the answer well-formed
    # and hollow -- `Mt Moon 1F (None,None) facing None` -- and the model spent a
    # whole `poke state` call recovering what the answer already knew. One
    # 457-call session shows that `act | state` pair fifteen times.
    if summary["battle"] or summary["dialog"]:
        for key in ("moves", "run"):
            summary.pop(key, None)
        summary["no_walk"] = NO_WALK_IN_BATTLE if summary["battle"] else NO_WALK_IN_BOX

    # In a battle the rest of the stepping fields go too -- they are read off a
    # screen showing the fight -- and the fight takes their place: who you are
    # against and what you can hit them with.
    if summary["battle"]:
        # `exits` goes with them. It is the same class of fact as `run` and
        # `moves` -- an answer to "where may I walk" -- and the strip above
        # forgot it: measured on a Mt Moon B2F encounter, the battle frame
        # printed `exits Mt Moon B1F (25, 9)` one line under `no walking in a
        # battle`, which is the payload contradicting itself. 26 bytes there,
        # 70 on Route 4, on every frame of every fight. The exit has not moved
        # and the overworld answer names it again the moment the fight ends.
        for key in ("on_warp", "warp", "faces", "exits"):
            summary.pop(key, None)
        # Facing goes with them, and says so: see FACING_UNREAD_IN_BATTLE. It is
        # the one field here that a battle frame holds a *wrong* value for.
        summary.pop("facing", None)
        summary["facing_unread"] = FACING_UNREAD_IN_BATTLE
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

    # Real on-screen words only. The reader falls back to a fixed placeholder --
    # "Dialog box visible (waiting for input)." -- when it cannot extract any, and
    # all 660 payloads that ever carried this field carried exactly that string,
    # 36,300 bytes saying what the `dialog` flag beside it already says.
    text = str(screen_text.get("text") or "").strip()
    if text and not text.startswith(DIALOG_PLACEHOLDER_PREFIX):
        summary["screen_text"] = text[:SCREEN_TEXT_LIMIT]
    return summary


#: Two visits is a corridor you walked back down. Three is a loop.
#:
#: Raised from 3 after measuring what it bought: `here_before` was sent 2,339
#: times across the run and referred to again in the agent's next command exactly
#: zero times. It reached 49 on one tile without changing anything. It is the
#: clearest advisory-versus-corrective case in this project -- `run` and `exits`
#: are facts the next command acts on, and this is a number about the past.
#:
#: Not deleted, because it is the only thing in the payload that says "you have
#: been here", and the `circling` detector reads the same store. Raised so it
#: appears when the loop is real rather than on every step of a corridor.
HERE_BEFORE_THRESHOLD = 8


def _annotate_batch_outcome(summary: dict, outcome: Optional[dict]) -> None:
    """Report what the batch achieved, not just where it ended.

    Without this the agent cannot tell a 16-step walk from one step and fifteen
    presses of its face against a tree.
    """
    if not outcome:
        return
    if summary.get("battle"):
        # Including the settle flag, which says nothing on a battle frame: the
        # watchdog watches the walk counter and the sprite step vectors, and an
        # encounter freezes both mid-step, so it gives up on every battle press
        # measured -- the entry frame and all five turns after it. The BATTLE
        # lines already say what this frame is, and the coordinates beside them
        # were measured as true, so the mid-transition caveat would be false.
        return
    # Before anything else, because it says the rest of the answer may not
    # describe a resting frame at all. Every other field here is read off one.
    if outcome.get("settled") is False:
        summary["settled"] = False
    moved = outcome.get("moved")
    if moved is None:  # no walk actions in the batch — nothing to report
        return
    summary["moved"] = moved
    if summary.get("dialog"):
        # A walk that ends under an open box did not hit anything. `blocked_after`
        # is inferred from a position that did not change, and while a box is up
        # nothing moves whatever the ground is: two `walk_down` presses into Oak's
        # dialog came back `blocked_after: 1` about a tile the player walks over
        # every time. `no_walk` beside `moved: 0` says what actually stopped it.
        return
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


def _milestone_ids_sync() -> tuple[str, ...]:
    """Every ladder milestone the game currently satisfies. Never raises.

    One RAM read of the event-flag block plus the badge byte and the bag. It is
    called under the emulator lock at the end of every batch, so a milestone is
    priced at the presses of the batch that actually reached it.
    """
    if _reader is None or not hasattr(_reader, "read_bits"):
        return ()
    try:
        return tuple(sorted(MilestoneTracker(_reader).snapshot()))
    except Exception as exc:  # noqa: BLE001 — the oracle must never fail a batch
        print(f"[server] WARNING: milestone read failed: {exc}")
        return ()


async def _milestone_snapshot() -> frozenset:
    """The milestone oracle as the run recorder wants it: awaited, under the lock."""
    if _emulator is None or _reader is None or not hasattr(_reader, "read_bits"):
        return frozenset()
    return frozenset(await _run_emulator_sync(_milestone_ids_sync))


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


def _check_action_limits(actions: list[str]) -> None:
    """Refuse a batch that would monopolise the emulator, before it starts.

    The caps are enforced here rather than inside the executor because the point
    is to never begin: ``wait_1000000000`` ticked for hundreds of days holding
    the lock, and the client that asked for it had long since timed out.
    """
    try:
        validate_action_batch(actions)
    except ActionLimitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownActionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Receipts and interventions
#
# One receipt per action batch, appended after the emulator lock is released.
# The supervisor decides which run it lands in; this half only ever knows what
# a batch did, which is the half nothing else in the process can see.
# ---------------------------------------------------------------------------


async def _write_receipt(
    *,
    tool: str,
    presses: int,
    bundle: Optional[dict],
    outcome: Optional[dict],
    milestone_ids: tuple = (),
    exit_code: int = 0,
    reloaded: bool = False,
    extra: Optional[dict] = None,
) -> None:
    """Record what one batch cost, then let the intervention loop read it.

    Cost on the hot path: building a dict from values already in hand, plus one
    ``write`` and one ``fsync`` of a few hundred bytes that the recorder hands to
    the default executor, so the event loop is free while the line lands. The
    file is opened ``O_APPEND`` and never re-read, so this does not grow with
    the length of the run.
    """
    recorder = _run_recorder
    if recorder is not None and recorder.run_id is not None:
        try:
            await recorder.append(
                **receipt_from_batch(
                    tool=tool,
                    presses=presses,
                    bundle=bundle,
                    outcome=outcome,
                    milestone_ids=milestone_ids,
                    exit_code=exit_code,
                    reloaded=reloaded,
                    extra=extra,
                )
            )
        except Exception as exc:  # noqa: BLE001 — a lost receipt must not lose the batch
            print(f"[server] WARNING: could not record a receipt: {exc}")
    _schedule_intervention_check(bundle)


def _intervention_state_summary(bundle: Optional[dict]) -> str:
    """The few lines a thinking session gets instead of the player's history."""
    summary = _observation_summary(bundle)
    parts = [f"{key}: {value}" for key, value in summary.items() if value not in (None, "", [])]
    return "\n".join(parts) or "(no observation available)"


def _intervention_milestone_summary() -> str:
    recorder = _run_recorder
    if recorder is None or recorder.run_id is None:
        return ""
    lines = [f"{recorder.total_presses} presses spent so far in run {recorder.run_id}."]
    tail = recorder.attainments[-4:]
    if tail:
        lines.append("Most recent milestones, with what they cost:")
        lines += [f"  {item['label']} at {item['presses']} presses" for item in tail]
    return "\n".join(lines)


def _schedule_intervention_check(bundle: Optional[dict]) -> None:
    """Ask the loop to look at the run, off the request that produced the batch.

    A swap takes the player's whole KV cache to disk and back; making an action
    wait for that would stall the very run it is trying to help. One at a time:
    a batch that arrives while a swap is in flight is skipped, not queued.
    """
    global _intervention_task
    runner, recorder = _interventions, _run_recorder
    if runner is None or recorder is None or not runner.enabled:
        return
    if _intervention_task is not None and not _intervention_task.done():
        return
    _intervention_task = asyncio.create_task(_run_intervention_check(bundle))


async def _run_intervention_check(bundle: Optional[dict]) -> None:
    runner, recorder = _interventions, _run_recorder
    if runner is None or recorder is None:
        return
    try:
        # The party goes in because a receipt has HP and nothing else about the
        # team. `Toothless` needs move PP to see a lead that cannot deal damage,
        # which went unnoticed for 24% of one run's presses precisely because
        # every signal here was about HP.
        state = {"map": _observation_summary(bundle).get("map")}
        party = ((bundle or {}).get("state") or {}).get("party")
        if party:
            state["party"] = party
        await runner.after_batch(
            recorder.recent_receipts(),
            state=state,
            total_presses=recorder.total_presses,
            state_summary=_intervention_state_summary(bundle),
            milestone_summary=_intervention_milestone_summary(),
            observation=bundle,
        )
    except Exception as exc:  # noqa: BLE001 — never let this take the server with it
        print(f"[server] WARNING: intervention check failed: {exc}")


def _intervention_advise(prompt: str) -> str:
    """Run one thinking-mode Pi session. Blocking: it holds the borrowed slot."""
    supervisor = _supervisor
    if supervisor is None or not supervisor.pi_binary:
        raise RuntimeError("Pi executable was not found, so nothing can think.")
    return pi_thinker(
        supervisor.pi_binary,
        provider=supervisor.provider,
        model=supervisor.model,
        cwd=supervisor.workspace_dir,
    )(prompt)


def _build_intervention_runner(data_dir: Path) -> InterventionRunner:
    enabled = _interventions_flag(_config)
    return InterventionRunner(
        enabled=enabled,
        advise=_intervention_advise,
        slot_client=build_slot_client(
            base_url=_config.slot_base_url if _config else DEFAULT_SLOT_BASE_URL,
            model=_config.slot_model if _config else DEFAULT_SLOT_MODEL,
        ),
        slot_filename=_config.slot_filename if _config else DEFAULT_SLOT_FILENAME,
        journal_path=data_dir / JOURNAL_FILENAME,
    )


async def _run_actions(
    actions: list[str], *, source: str, reason: str, rate_check: bool = True
) -> dict:
    """Execute one batch of actions with the standard before/after bookkeeping.

    The batch, the refresh and both state reads happen inside one coordinator
    transaction, so the position reported back is the position this batch
    produced. Events are broadcast afterwards, with the lock already released.
    """
    if rate_check:
        _check_action_rate()
    try:
        _check_action_limits(actions)
        result = await _coordinator.act_and_observe(actions, source=source, reason=reason)
    except Exception as exc:
        # A refused or failed batch is a receipt too. Two of them in a row is
        # exactly what the repeated_failure detector fires on, and a run whose
        # receipts hold only its successes cannot show what it wasted.
        await _write_receipt(
            tool=source,
            presses=0,
            bundle=None,
            outcome=None,
            exit_code=1,
            extra={"error": str(exc)[:200], "actions": list(actions)[:8]},
        )
        raise
    bundle = result["bundle"]
    outcome = result["outcome"]
    executed = result["actions_executed"]

    await _record_and_broadcast(
        "action",
        {"actions": actions, "source": source, "state_before": result["state_before"]},
    )
    await _broadcast_runtime_refresh(result)
    await _record_and_broadcast(
        "action_result",
        {
            "actions": actions,
            "actions_executed": executed,
            "source": source,
            "state_after": result["state_after"],
            "feedback": bundle.get("recent_action"),
            "state_delta": bundle.get("state_delta"),
            "objective_status": (bundle.get("objective") or {}).get("current"),
            "stuck_signal": bundle.get("stuck"),
            "screen_text": bundle.get("screen_text"),
        },
    )
    await _write_receipt(
        tool=source,
        presses=outcome.get("presses") or 0,
        bundle=bundle,
        outcome=outcome,
        milestone_ids=outcome.get("milestones") or (),
    )
    return {"actions_executed": executed, "bundle": bundle, "outcome": outcome}


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------


#: The fixed cadence a press used to use: 8 frames held, 12 waiting. It is a
#: fallback now, kept only for emulators that predate `settle` (the fakes in
#: tests, and FireRed's stub).
LEGACY_PRESS_FRAMES = 8
LEGACY_WAIT_FRAMES = 12


def _press_and_settle_or_wait(button: str) -> bool:
    """Press, then wait for the result to actually be observable.

    Returns whether the game came to rest. False means the frame that follows is
    mid-something and does not describe a resting place — see
    ``_execute_action_batch_sync``, which carries that answer out to the payload.

    A fixed 20-frame wait returns while the game is still moving the player. A
    ledge hop takes about 40 frames, so the old cadence read the mid-air tile
    and recorded it into the explored map as ground the player had walked. It
    also left the emulator frozen mid-animation, where the next input is
    swallowed. `settle` watches the walk counter, the sprite step vectors, the
    ledge and spin flags and the map id instead, so it returns when the game
    hands control back rather than after a guess.
    """
    if hasattr(_emulator, "press_and_settle"):
        return bool(_emulator.press_and_settle(button, LEGACY_PRESS_FRAMES))
    _emulator.press(button, LEGACY_PRESS_FRAMES)
    _emulator.tick(LEGACY_WAIT_FRAMES)
    # An emulator that cannot tell when the game came to rest cannot report that
    # it did not, which is exactly where it stood before settling existed.
    return True


def _execute_action_sync(action_str: str) -> bool:
    """Parse and execute a single action string on the emulator.

    Returns whether the game was at rest afterwards. Only the pressing actions
    can answer no; an explicit `wait_N` is a request for N frames and nothing
    more, so it says nothing about what the game is in the middle of.

    Supported formats:
        press_X       — press button X for 10 frames, wait 20 frames
        walk_X        — press direction for 16 frames, wait 8 frames
        hold_X_N      — hold button X for N frames
        wait_N        — tick N frames with no input
        a_until_dialog_end — press A every 30 frames until dialog clears (max 300)
    """
    global _press_count
    action_str = action_str.strip().lower()
    _press_count += presses_for_action(action_str)

    if action_str == "a_until_dialog_end":
        for _ in range(10):  # max 300 frames = 10 * 30
            _emulator.press("a")
            _press_count += 1
            _emulator.tick(30)
            # Check dialog flag via reader if available
            try:
                state = _get_state_dict()
                if not state.get("dialog_active", False):
                    break
            except Exception:
                pass
        return True

    # Split into tokens
    parts = action_str.split("_")

    if parts[0] == "press" and len(parts) >= 2:
        button = "_".join(parts[1:])
        return _press_and_settle_or_wait(button)

    if parts[0] == "walk" and len(parts) >= 2:
        return _press_and_settle_or_wait(parts[1])

    if parts[0] == "hold" and len(parts) >= 3:
        button = "_".join(parts[1:-1])
        frames = int(parts[-1])
        _emulator.press(button, frames)
        return True

    if parts[0] == "wait" and len(parts) == 2:
        frames = int(parts[1])
        _emulator.tick(frames)
        return True

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
    settled = True
    tile = _player_tile_sync() if any(_is_walk_action(item) for item in actions) else None

    for index, action_str in enumerate(actions, start=1):
        is_walk = _is_walk_action(action_str)
        settled = _execute_action_sync(action_str) and settled
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
        # Whether the game was still at rest when the batch handed back. A frame
        # that is not is the one place the payload can pair one map's name with
        # another map's coordinates: sampled ten frames into a gate warp, the
        # reads say "Route 2 (5,0)" -- Route 2's name with the gate's tile, while
        # the true landing is (3,11). `settle` normally hides that by waiting for
        # the transition to finish; when it gives up, the answer has to say so
        # instead of describing whatever it caught mid-flight. `/load` has
        # reported exactly this since it was written.
        "settled": settled,
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
    """One battle command, with the bookkeeping an /action batch would get.

    The menu walk and the observation of its result are one transaction: every
    cursor read in the sequence has to see the machine the previous press left.
    """
    _check_action_rate()

    def _counted(*call_args) -> dict:
        """The command, priced. A battle spends buttons like any other batch."""
        before = _press_count
        counted = dict(func(*call_args))
        counted["presses"] = max(0, _press_count - before)
        counted["milestones"] = _milestone_ids_sync()
        return counted

    result = await _coordinator.battle_and_observe(
        func=_counted,
        args=args,
        reason="battle_command",
        source="battle",
    )
    outcome = result["outcome"]
    bundle = result["bundle"]

    await _record_and_broadcast(
        "action",
        {
            "actions": [],
            "source": "battle",
            "intent": intent,
            "state_before": result["state_before"],
        },
    )
    await _broadcast_runtime_refresh(result)
    await _record_and_broadcast(
        "action_result",
        {
            "actions": outcome["actions"],
            "actions_executed": len(outcome["actions"]),
            "source": "battle",
            "intent": intent,
            "state_after": result["state_after"],
            "screen_text": bundle.get("screen_text"),
        },
    )
    await _write_receipt(
        tool="battle",
        presses=outcome.get("presses") or 0,
        bundle=bundle,
        outcome=outcome,
        milestone_ids=outcome.get("milestones") or (),
        extra={"intent": intent},
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
        _explored_records_since_save, \
        _world, \
        _guide_log, \
        _press_count, \
        _run_recorder, \
        _interventions, \
        _intervention_task, \
        _startup_error
    _loop = asyncio.get_running_loop()
    _start_time = time.time()
    _emulator_lock = asyncio.Lock()
    # A startup that gives up must not leave the previous run's emulator visible
    # to /health and /action.
    _emulator = None
    _reader = None
    _runtime = None
    _press_count = 0
    _run_recorder = None
    _interventions = None
    _intervention_task = None
    _startup_error = None
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

    # Decide what this ROM is *before* creating anything that would have to be
    # torn down again.
    try:
        game_type = _resolve_game_type(str(rom), _config.game_type)
    except (UnsupportedGameError, ValueError) as exc:
        _startup_error = str(exc)
        print(f"[server] ERROR: {_startup_error}")
        return
    _config.game_type = game_type

    print(f"[server] Loading ROM: {rom}")
    print(f"[server] Detected game type: {game_type}")

    # Create emulator
    from pokemon_agent.emulator import create_emulator

    _emulator = create_emulator(str(rom))

    # Create memory reader
    from pokemon_agent.memory.red import PokemonRedReader

    _reader = PokemonRedReader(_emulator)

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
    # The static map graph and the guide-read log. Both live beside the explored
    # map: they are run memory, not workspace scratch.
    _world = World.load()
    print(f"[server] World graph: {len(_world)} maps — {_world.source or 'not generated yet'}")
    _guide_log = GuideLog(data_dir / "guide_reads.jsonl")
    _map_image_state = None
    _runtime = AgentRuntime(
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        # Lets the annotated frame shade the ground already walked.
        visited_lookup=_explored_maps.visited,
    )
    # SEAM: the annotated frame insets a mini-map drawn from these tile sets.
    _runtime.map_grid_lookup = _explored_maps.grid
    # The scoreboard. The recorder resolves which run this process is joining
    # from a pointer file, so a restart rejoins the playthrough already in
    # progress instead of starting a second one beside it.
    _run_recorder = RunRecorder(
        data_dir,
        milestone_snapshot=_milestone_snapshot,
        start_checkpoint=_config.load_state,
    )
    open_run = _run_recorder.read_pointer()
    print(f"[server] Run receipts: {data_dir / 'runs'} — open run: {open_run or 'none yet'}")
    _interventions = _build_intervention_runner(data_dir)
    _supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url=f"http://127.0.0.1:{_config.port}",
        event_sink=_record_existing_event_and_broadcast,
        stream_sink=broadcast,
        artifact_paths=_runtime_artifact_paths,
        critic_context=_critic_context,
        run_recorder=_run_recorder,
    )
    # Wired after the supervisor exists: an intervention is delivered down the
    # same RPC path POST /supervisor/steer uses, and reported where the run is
    # watched from.
    _interventions.deliver = _supervisor.deliver_intervention
    _interventions.notify = _supervisor.record_intervention
    _supervisor.intervention_state["enabled"] = _interventions.enabled
    print(
        "[server] Interventions: "
        + (
            "ON — the harness may stop the player and think"
            if _interventions.enabled
            else f"off (set {INTERVENTIONS_ENV_VAR}=1 to enable)"
        )
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
        # One mounting path, owned by the dashboard package: the shell at
        # /dashboard and /dashboard/, its assets under /dashboard/assets. The
        # call is idempotent, so a restarted lifespan re-uses what is there.
        _dashboard_dir = _mount_dashboard_routes()
        if _dashboard_dir is not None:
            print("[server] Dashboard mounted at /dashboard (assets under /dashboard/assets)")
        else:
            print("[server] Dashboard static files not found — /dashboard unavailable")

    # Auto-load a save state if specified. Same resolver as POST /load: a name
    # from a config file is no more trusted than a name from the network.
    if _config.load_state:
        try:
            state_path = resolve_save_path(data_dir / "saves", _config.load_state)
        except SaveNameError as exc:
            state_path = None
            print(f"[server] WARNING: Refusing to auto-load '{_config.load_state}': {exc}")
        if state_path is not None and state_path.exists():
            try:
                _emulator.load_state(str(state_path))
                print(f"[server] Loaded save state: {_config.load_state}")
            except Exception as e:
                print(f"[server] WARNING: Failed to load state '{_config.load_state}': {e}")
        elif state_path is not None:
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
    global _supervisor, _realtime_task, _live_artifact_task, _intervention_task
    if _intervention_task is not None:
        _intervention_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _intervention_task
        _intervention_task = None
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
    if _run_recorder is not None:
        # Closes the append handle, not the run: the playthrough outlives this
        # process and the next start adopts it straight back off the pointer.
        _run_recorder.close()


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
        "run": _run_recorder.status() if _run_recorder is not None else None,
        "interventions": _interventions.status() if _interventions is not None else None,
        "startup_error": _startup_error,
    }


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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Supervisor stop error: {exc}")


@app.get("/state")
async def get_state():
    """Full game state JSON."""
    _ensure_emulator()
    try:
        state = await _run_emulator_sync(_get_state_dict)
        return JSONResponse(content=state)
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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


def _saves_dir() -> Path:
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    return Path(_config.data_dir).expanduser().resolve() / "saves"


def _save_path_for(name: str) -> Path:
    """Resolve one save name, or answer 400. The only way to build a save path."""
    try:
        return resolve_save_path(_saves_dir(), name)
    except SaveNameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/save")
async def save_state(req: SaveRequest):
    """Save emulator state to disk."""
    _ensure_emulator()
    saves_dir = _saves_dir()
    saves_dir.mkdir(parents=True, exist_ok=True)
    save_path = _save_path_for(req.name)
    try:
        result = await _coordinator.save_and_observe(
            path=str(save_path),
            reason=f"manual_save:{req.name}",
            explicit_save=_make_runtime_save_event(
                req.name,
                save_path,
                source="manual",
                reason="manual_save",
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {e}")
    await _broadcast_runtime_refresh(result)
    return {
        "success": True,
        "save": {"name": req.name, "path": str(save_path)},
        **_observation_summary(result["bundle"]),
    }


@app.post("/load")
async def load_state(req: SaveRequest):
    """Load emulator state from disk, let it settle, and report where it landed."""
    _ensure_emulator()
    save_path = _save_path_for(req.name)
    if not save_path.exists():
        raise HTTPException(status_code=404, detail=f"Save not found: {req.name}")
    try:
        result = await _coordinator.load_settle_and_observe(
            path=str(save_path),
            reason=f"manual_load:{req.name}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

    bundle = result["bundle"]
    if result.get("settled", True):
        await _broadcast_runtime_refresh(result)
    await _record_and_broadcast("load", {"name": req.name, "path": str(save_path)})
    await broadcast({"type": "state_update", "reason": "load", "state": result["state_after"]})
    # A reload rewinds the game, never the bill. The receipt spends no presses
    # and the running total carries straight over it, so a gym won on the fourth
    # attempt costs what all four attempts cost.
    await _write_receipt(
        tool="load",
        presses=0,
        bundle=bundle,
        outcome=None,
        milestone_ids=await _run_emulator_sync(_milestone_ids_sync),
        reloaded=True,
        extra={"save": req.name},
    )
    payload = {
        "success": True,
        "save": {"name": req.name, "path": str(save_path)},
        **_observation_summary(bundle),
    }
    if not result.get("settled", True):
        # The save was captured mid-transition and the game never came to rest.
        # Nothing was published and nothing was auto-saved: the map store would
        # have taken the *previous* map's geometry and never let go of it.
        payload["settled"] = False
    return payload


#: How many saves to return by default. The full list was 71 kB for 465 files --
#: roughly 18,000 tokens for one `poke saves` call, and it had been 3,802 files
#: before the autosaves were bounded. A save the agent might load is a recent one
#: or one it named; the four hundred harness autosaves behind those are not a
#: menu, they are a backup.
DEFAULT_SAVES_LIMIT = 40


@app.get("/saves")
async def list_saves(limit: int = DEFAULT_SAVES_LIMIT, named: bool = False):
    """List save-state files, newest first.

    `limit=0` returns everything, for an operator who genuinely wants the lot.
    `named=true` drops the harness's own `auto__` checkpoints, which is what a
    caller looking for somewhere to go back to actually wants.
    """
    saves_dir = _saves_dir()
    try:
        files = list_save_files(saves_dir)
        if named:
            files = [f for f in files if not f.name.startswith(AUTO_SAVE_PREFIX)]
        # One stat per file, and a file that vanishes between the glob and the
        # stat drops out of the listing instead of failing it. The autosave
        # pruner unlinks stale `auto__*.state` from an executor thread while
        # this runs on the loop, so the window is open constantly: this 500'd
        # the *entire* listing 23 times in one session, and `poke saves` is how
        # the agent finds anything to load. The pruner already tolerates the
        # same race on its side.
        stated = []
        for found in files:
            try:
                stated.append((found, found.stat()))
            except OSError:
                continue
        stated.sort(key=lambda pair: pair[1].st_mtime, reverse=True)
        total = len(stated)
        shown = stated if limit <= 0 else stated[:limit]
        payload = {
            "saves": [
                {
                    "name": found.stem,
                    "file": found.name,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
                for found, stat in shown
            ],
            "count": total,
            "shown": len(shown),
        }
        if len(shown) < total:
            payload["truncated"] = True
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing saves: {e}")


# ---------------------------------------------------------------------------
# Capabilities: routing, walking, damage, frontier, simulation, guides, progress
#
# Five finished modules answer these; the routes below validate, call one
# service function, and translate its refusal into a status code. The agent's
# CLI is stdlib-only and staged standalone, so HTTP is the only way it can
# reach any of them.
# ---------------------------------------------------------------------------


def _ensure_world() -> World:
    if _world is None:
        raise HTTPException(status_code=503, detail="World graph is not loaded")
    if len(_world) == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "The world graph is empty — pokemon_agent/data/game/world.json has not been "
                "generated. Run: .venv/bin/python scripts/gen_gamedata.py"
            ),
        )
    return _world


def _capability_error(exc: capabilities.CapabilityError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.detail)


def _navigation_snapshot_sync() -> Optional[dict]:
    """The live collision window, without folding it into the explored map."""
    if _emulator is None:
        return None
    try:
        snapshot = _emulator.get_navigation_snapshot(_reader)
    except NotImplementedError:
        return None
    except Exception:  # noqa: BLE001 — perception must never fail a read
        return None
    return snapshot.to_dict()


def _observation_sync() -> dict:
    """State and live collision together, read in one locked pass."""
    return {"state": _get_state_dict(), "navigation": {"snapshot": _navigation_snapshot_sync()}}


def _in_battle_sync() -> bool:
    try:
        return bool((_reader.read_battle() if _reader is not None else {}).get("in_battle"))
    except Exception:  # noqa: BLE001 — a guard must not be the thing that fails
        return False


async def _require_snapshot() -> dict:
    """The live collision window, or a refusal that says why there isn't one.

    A battle screen still yields a snapshot, and it is not a map: the window is
    the fight, so a flood over it reaches one tile and every question answered
    from it comes back confidently wrong. Measured live -- `goto` reported
    `sealed: true, reachable_tiles: 1` and `frontier` reported zero tiles, both
    while a Zubat was on screen. Telling an agent it is walled in when it is
    merely in a battle is worse than telling it nothing.
    """
    _ensure_emulator()
    if await _run_emulator_sync(_in_battle_sync):
        raise HTTPException(
            status_code=409,
            detail=(
                "In a battle, so there is no map to plan over. Finish the fight "
                "or flee, then ask again."
            ),
        )
    snapshot = await _run_emulator_sync(_navigation_snapshot_sync)
    if not snapshot:
        raise HTTPException(
            status_code=503,
            detail="No live collision window right now — the game is not on the overworld.",
        )
    return snapshot


def _explored_grid(map_id: Optional[int]) -> Optional[dict]:
    if _explored_maps is None or map_id is None:
        return None
    try:
        return _explored_maps.grid(int(map_id))
    except Exception:  # noqa: BLE001 — map memory must never fail a request
        return None


def _current_map_name_sync() -> Optional[str]:
    if _reader is None:
        return None
    try:
        return (_reader.read_map_info() or {}).get("map_name")
    except Exception:  # noqa: BLE001
        return None


def _player_coord_sync() -> Optional[tuple[int, int]]:
    """Where the player is standing, for a router that answers per pocket."""
    if _reader is None:
        return None
    try:
        coords = _reader.read_coordinates() or {}
        x, y = coords.get("x"), coords.get("y")
        return (int(x), int(y)) if x is not None and y is not None else None
    except Exception:  # noqa: BLE001
        return None


def _pocket_graph() -> Optional[PocketGraph]:
    """A route graph over pockets, built from whatever the store has decoded.

    Every map the player has stood on contributes its real terrain and its
    ledges; the rest fall back to one pocket each, which is the old behaviour
    for those maps and no worse. Cheap enough to build per request: the pieces
    are flooded lazily and most routes touch a handful of maps.
    """
    if _explored_maps is None:
        return None
    from pokemon_agent import gamedata

    world = gamedata.world()
    by_name = {
        name: map_id
        for map_id, name in ((mid, MAP_NAMES.get(mid)) for mid in _explored_maps.map_ids())
        if name
    }

    def terrain_for(name: str):
        map_id = by_name.get(name)
        return _explored_maps.terrain(map_id) if map_id is not None else None

    def ledges_for(name: str):
        map_id = by_name.get(name)
        return _explored_maps.ledges_for(map_id) if map_id is not None else None

    def connections_for(name: str):
        """The map header's connections, with their offsets, where we have them.

        `gamedata`'s table says which maps touch and never at what offset, so a
        router given only that has to guess which pocket walking off an edge
        lands in. On Route 4's south edge the candidates are sixty tiles apart,
        and the guess produced a route that walked south and back north into a
        different pocket. Stored specs first, the flat table only as a fallback.
        """
        map_id = by_name.get(name)
        stored = _explored_maps.connections_for(map_id) if map_id is not None else {}
        return stored or (world.get(name) or {}).get("connections") or {}

    return PocketGraph(
        lambda name: (world.get(name) or {}).get("warps") or [],
        terrain_for,
        connections_for,
        lambda name: (world.get(name) or {}).get("size"),
        ledges_for,
    )


@app.get("/route")
async def get_route(to: Optional[str] = None):
    """Hops from the current map to another, as a plan rather than as buttons."""
    _ensure_emulator()
    world = _ensure_world()
    current = await _run_emulator_sync(_current_map_name_sync)
    if not current:
        raise HTTPException(status_code=503, detail="The current map is not readable right now.")
    position = await _run_emulator_sync(_player_coord_sync)
    try:
        return capabilities.route_payload(
            world, current, to or "", pockets=_pocket_graph(), at=position
        )
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc


@app.post("/goto")
async def goto(req: GotoRequest):
    """Walk toward a map or a tile, re-planning on live collision each map.

    A hop is a plan, not a guarantee — Route 4 is one map whose halves are
    separated by Mt. Moon — so this stops and says why rather than grinding into
    rock, and it never spends more frames than one action batch may.
    """
    _ensure_emulator()
    # Same reason as `_require_snapshot`: a battle frame is not a map, and
    # planning over one answers "sealed in, one tile reachable".
    if await _run_emulator_sync(_in_battle_sync):
        raise HTTPException(
            status_code=409,
            detail=(
                "In a battle, so there is no map to walk over. Finish the fight "
                "or flee, then ask again."
            ),
        )
    _check_action_rate()
    if req.target and (req.x is not None or req.y is not None):
        raise HTTPException(
            status_code=400, detail="Send either a target map or an x and y, not both."
        )
    target_xy = None
    if req.x is not None or req.y is not None:
        if req.x is None or req.y is None:
            raise HTTPException(status_code=400, detail="A tile target needs both x and y.")
        target_xy = (int(req.x), int(req.y))
    elif not req.target:
        raise HTTPException(status_code=400, detail="Nothing to walk to: send target, or x and y.")
    # Walking to a tile on the current map needs no map graph at all.
    world = _ensure_world() if req.target else World({})

    async def observe() -> dict:
        return capabilities.observation_from_bundle(await _run_emulator_sync(_observation_sync))

    async def act(actions: list[str]) -> dict:
        return await _run_actions(actions, source="goto", reason="goto", rate_check=False)

    try:
        result = await capabilities.walk_to(
            observe=observe,
            act=act,
            world=world,
            explored_grid=_explored_grid,
            target_map=req.target,
            target_xy=target_xy,
            frame_budget=MAX_FRAMES_PER_BATCH,
        )
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc

    bundle = result["bundle"]
    if "screen_text" not in bundle:
        # Nothing was walked, so no batch refreshed the workspace. Answer with a
        # real observation rather than half of one.
        bundle = await _refresh_and_broadcast(reason="goto", source="goto")
    summary = _observation_summary(bundle)
    _annotate_explored_map(summary, bundle)
    payload = {
        "actions_executed": result["actions_executed"],
        **summary,
        "walked": result["walked"],
        "arrived": result["arrived"],
        "stopped_because": result["stopped_because"],
    }
    # What to do instead, when there is something: whether the goal is walled off
    # or merely unseen, and which exits *are* reachable from here. The run spent
    # twelve hours in a sealed pocket of Route 4 whose only ways out led back
    # into Mt. Moon, and every refusal it got named the tile it could not reach
    # rather than the three it could.
    if summary.get("battle"):
        # A wild encounter ended the walk, and everything computed after it was
        # read off a battle frame. Mt. Moon rolls one about every ten steps, so
        # most walks longer than fifteen tiles finish this way -- and each one was
        # reporting `sealed: true, reachable_tiles: 1` about ground the player had
        # just crossed. Say what actually happened instead.
        payload["stopped_because"] = (
            f"walked {result['walked']}, then a wild Pokemon appeared. "
            "Finish the fight or flee, then ask again."
        )
    elif result.get("onward"):
        payload["onward"] = result["onward"]
    return payload


def _calc_inputs_sync() -> dict:
    battle = (_reader.read_battle() if _reader is not None else None) or {}
    party = (_reader.read_party() if _reader is not None else None) or []
    moves: list[dict] = []
    if battle.get("in_battle"):
        try:
            moves = _reader.read_battle_moves() or []
        except Exception:  # noqa: BLE001 — an unreadable move list is not a crash
            moves = []
    return {"battle": battle, "party": party, "moves": moves}


@app.get("/calc")
async def calc():
    """Damage each of the active Pokemon's moves would do, and what it faces back."""
    _ensure_emulator()
    inputs = await _run_emulator_sync(_calc_inputs_sync)
    try:
        return capabilities.calc_payload(inputs["battle"], inputs["party"], inputs["moves"])
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc


@app.get("/frontier")
async def frontier():
    """Reachable ground on this map that has never been stood on, nearest first.

    "Unseen" is *unwalked*, not unrendered: every tile the window has ever shown
    is recorded as seen the moment it is shown, so the useful question is which
    reachable ground the player has not actually been to.
    """
    snapshot = await _require_snapshot()
    grid = _explored_grid(snapshot.get("map_id"))
    walked = (grid or {}).get("walked") or set()
    try:
        return capabilities.frontier_payload(snapshot, grid, walked)
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc


@app.post("/sim")
async def sim(req: SimRequest):
    """Dry-run a plan against live collision. Presses nothing."""
    snapshot = await _require_snapshot()
    try:
        return capabilities.simulate_payload(
            req.actions, snapshot, _explored_grid(snapshot.get("map_id"))
        )
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc


def _record_guide_read(guide: str, slug: str) -> None:
    """Note which section was opened, and where in the run it happened.

    The map and the press count are what make the record answerable later: did
    reading this section change how the segment that followed went?
    """
    if _guide_log is None:
        return
    at_map = None
    if _runtime is not None:
        bundle = _runtime.live_bundle or _runtime.latest_bundle or {}
        at_map = ((bundle.get("state") or {}).get("map") or {}).get("map_name")
    if at_map is None and _explored_maps is not None:
        current = _explored_maps.current_map_id
        at_map = MAP_NAMES.get(current) if current is not None else None
    try:
        _guide_log.record_read(guide, slug, at_map=at_map, presses=_press_count)
    except Exception as exc:  # noqa: BLE001 — telemetry must never fail a read
        print(f"[server] WARNING: guide read not recorded: {exc}")


@app.get("/guide")
async def guide(ref: Optional[str] = None, q: Optional[str] = None):
    """The walkthrough library: an outline, a search, or one section's body."""
    if ref and q:
        raise HTTPException(status_code=400, detail="Send either ref or q, not both.")
    try:
        if ref:
            payload = capabilities.guide_section(ref)
            _record_guide_read(payload["guide"], payload["slug"])
            return payload
        if q:
            return capabilities.guide_search(q)
        return capabilities.guide_outline()
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc


def _milestone_summary_sync() -> dict:
    return MilestoneTracker(_reader).summary()


def _progress_presses() -> int:
    """Buttons spent by the *run*, which outlives this process and its sessions.

    Falls back to the since-startup counter only when no run is open, which is
    the case before a supervisor session has ever started.
    """
    recorder = _run_recorder
    if recorder is not None and recorder.run_id is not None:
        return recorder.total_presses
    return _press_count


@app.get("/progress")
async def progress():
    """How far along the run is, in verified milestones and in buttons spent."""
    _ensure_emulator()
    if _reader is None or not hasattr(_reader, "read_bits"):
        raise HTTPException(
            status_code=503,
            detail="Milestone tracking needs a Pokemon Red memory reader.",
        )
    try:
        summary = await _run_emulator_sync(_milestone_summary_sync)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Milestones unreadable: {exc}") from exc
    payload = capabilities.progress_payload(summary, _progress_presses())
    if _run_recorder is not None:
        # Additive: `presses_to` and `attainments` are the names the dashboard
        # already prefers over the ledger it derives for itself, and `presses`
        # keeps its meaning — it is just the run's number now, not this
        # process's. Empty until a run is open, never absent.
        payload.update(_run_recorder.progress_payload())
    return payload


@app.get("/gamedata/{topic}")
async def game_data(
    topic: str,
    # `map` shadows the builtin inside this function and nowhere else; it is
    # spelled that way because it is the query string the agent types.
    map: Optional[str] = None,
    name: Optional[str] = None,
    limit: Optional[int] = None,
    full: bool = False,
    against: Optional[str] = None,
):
    """The static game database: trainers, encounters, items, shops, species, moves, types.

    Read-only and emulator-free — the ROM is not consulted and nothing is
    pressed, so a script may ask before it commits to a plan. Answers are
    shaped to be printed rather than dumped; see
    :data:`capabilities.GAMEDATA_TOPICS` for the list.
    """
    try:
        return capabilities.gamedata_payload(
            topic,
            map_name=map,
            name=name,
            limit=limit,
            full=full,
            against=against,
        )
    except capabilities.CapabilityError as exc:
        raise _capability_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
