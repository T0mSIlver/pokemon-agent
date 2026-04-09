"""
Pokemon Agent — FastAPI Game Server

Provides HTTP + WebSocket API for controlling a Game Boy / GBA emulator
running a Pokemon ROM, reading game state, and broadcasting events.
"""

import asyncio
import base64
import io
import json
import re
import time
from functools import partial
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from pokemon_agent.agent_runtime import AgentRuntime

__version__ = "0.1.0"

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


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_emulator = None  # Emulator instance
_reader = None  # GameMemoryReader subclass instance
_navigation_store = None  # NavigationStore instance
_runtime: Optional[AgentRuntime] = None
_start_time: float = 0.0
_loop: Optional[asyncio.AbstractEventLoop] = None
_dashboard_dir: Optional[Path] = None

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


def _get_navigation_payload_sync(goal: Optional[tuple[int, int]] = None) -> Optional[dict]:
    try:
        snapshot, location_map = _refresh_navigation_state_sync()
    except NotImplementedError:
        return None
    except Exception:
        return None
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
    return await _run_sync(
        _refresh_agent_bundle_sync,
        reason=reason,
        source=source,
        requested_actions=requested_actions,
        navigation_plan=navigation_plan,
        navigation_execution=navigation_execution,
        explicit_save=explicit_save,
    )


async def _broadcast_runtime_refresh(result: Optional[dict]) -> None:
    if not result:
        return
    for event in result.get("events", []):
        await broadcast(event)
    bundle = result.get("bundle")
    if bundle:
        await broadcast(
            {
                "type": "screenshot",
                "data": {
                    "image": bundle.get("raw_frame_b64"),
                    "annotated_image": bundle.get("annotated_frame_b64"),
                    "format": "png",
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
    distances = location_map.distance_map(
        snapshot.player_position,
        extra_blockers=snapshot.sprite_set,
    )
    return {
        "snapshot": snapshot.to_dict(goal=goal),
        "location_map": location_map.to_dict(
            player=snapshot.player_position,
            goal=goal,
            sprites=snapshot.sprite_positions,
            distances=distances,
        ),
    }


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"^(?P<kind>press|walk|hold|wait|a_until_dialog_end)(?:_(?P<rest>.+))?$")


async def _execute_action(action_str: str) -> None:
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
            await _run_sync(_emulator.press, "a")
            await _run_sync(_emulator.tick, 30)
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
        await _run_sync(_emulator.press, button, 8)
        await _run_sync(_emulator.tick, 12)
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
        await _run_sync(_emulator.press, direction, 8)
        await _run_sync(_emulator.tick, 12)
        return

    if parts[0] == "hold" and len(parts) >= 3:
        button = "_".join(parts[1:-1])
        frames = int(parts[-1])
        await _run_sync(_emulator.press, button, frames)
        return

    if parts[0] == "wait" and len(parts) == 2:
        frames = int(parts[1])
        await _run_sync(_emulator.tick, frames)
        return

    raise ValueError(f"Unknown action format: {action_str}")


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
        _start_time, \
        _config, \
        _loop, \
        _dashboard_dir
    _loop = asyncio.get_running_loop()
    _start_time = time.time()

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

    print(f"[server] Ready — listening on port {_config.port}")
    print(f"[server] Agent workspace: {workspace_dir}")
    print("[server] Endpoints:")
    print("[server]   GET  /          — server info")
    print("[server]   GET  /state     — game state")
    print("[server]   GET  /screenshot — current frame (PNG)")
    print("[server]   POST /agent/observe — refresh vision-first workspace bundle")
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
    print("[server]   GET  /health    — health check")
    print("[server]   WS   /ws        — live events")


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
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "emulator_ready": _emulator is not None,
        "agent_workspace_ready": _runtime is not None,
        "dashboard_ready": _dashboard_dir is not None,
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
    return FileResponse(_dashboard_dir / "index.html")


@app.get("/dashboard/state")
async def dashboard_state():
    """Aggregated dashboard state for the operator console."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    return JSONResponse(content=_runtime.dashboard_state())


@app.get("/dashboard/history")
async def dashboard_history(limit: int = 200):
    """Structured recent dashboard/agent events."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    limit = max(1, min(limit, 1000))
    return {"events": _runtime.history(limit)}


@app.post("/agent/observe")
async def agent_observe(req: ObserveRequest):
    """Refresh the vision-first workspace bundle and return artifact paths."""
    _ensure_emulator()
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialised")
    try:
        result = await _refresh_agent_bundle(reason=req.reason, source="observe")
        await _broadcast_runtime_refresh(result)
        if not result:
            raise HTTPException(
                status_code=500, detail="Agent observation refresh returned no data"
            )
        return result["bundle"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent observe error: {e}")


@app.get("/state")
async def get_state():
    """Full game state JSON."""
    _ensure_emulator()
    try:
        state = await _run_sync(_get_state_dict)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading state: {e}")


@app.get("/screenshot")
async def screenshot():
    """Current emulator frame as PNG image."""
    _ensure_emulator()
    try:
        png_bytes = await _run_sync(_get_screenshot_bytes)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.get("/screenshot/base64")
async def screenshot_base64():
    """Current emulator frame as base64-encoded PNG in JSON."""
    _ensure_emulator()
    try:
        png_bytes = await _run_sync(_get_screenshot_bytes)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {"image": b64, "format": "png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.post("/action")
async def execute_actions(req: ActionRequest):
    """Execute a sequence of game actions."""
    _ensure_emulator()
    try:
        state_before = await _run_sync(_get_state_dict)
        await _record_and_broadcast(
            "action",
            {
                "actions": req.actions,
                "state_before": state_before,
            },
        )
        executed = 0
        for action_str in req.actions:
            await _execute_action(action_str)
            executed += 1

        refresh = await _refresh_agent_bundle(
            reason="actions_executed",
            source="action",
            requested_actions=req.actions,
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        state_after = bundle.get("state") or await _run_sync(_get_state_dict)
        navigation_after = bundle.get("navigation") or _get_navigation_payload_sync()

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
            "actions_executed": executed,
            "state_before": state_before,
            "state_after": state_after,
            "navigation_after": navigation_after,
            "feedback": bundle.get("recent_action"),
            "state_delta": bundle.get("state_delta"),
            "screen_text": bundle.get("screen_text"),
            "objective_status": (bundle.get("objective") or {}).get("current"),
            "stuck_signal": bundle.get("stuck"),
            "artifacts": bundle.get("artifacts"),
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
        await _run_sync(_emulator.save_state, str(save_path))
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
            "path": str(save_path),
            "observation": bundle,
            "artifacts": bundle.get("artifacts"),
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
        await _run_sync(_emulator.load_state, str(save_path))
        refresh = await _refresh_agent_bundle(
            reason=f"manual_load:{req.name}",
            source="load",
        )
        await _broadcast_runtime_refresh(refresh)
        bundle = (refresh or {}).get("bundle", {})
        state_after = bundle.get("state") or await _run_sync(_get_state_dict)
        navigation_after = bundle.get("navigation") or _get_navigation_payload_sync()
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
            "name": req.name,
            "state_after": state_after,
            "navigation_after": navigation_after,
            "feedback": bundle.get("recent_action"),
            "state_delta": bundle.get("state_delta"),
            "screen_text": bundle.get("screen_text"),
            "objective_status": (bundle.get("objective") or {}).get("current"),
            "stuck_signal": bundle.get("stuck"),
            "artifacts": bundle.get("artifacts"),
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
        snapshot, location_map = await _run_sync(_refresh_navigation_state_sync)
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
        snapshot, location_map = await _run_sync(_refresh_navigation_state_sync)
        return _serialize_navigation(snapshot, location_map)
    except NotImplementedError as e:
        raise _navigation_not_supported(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Navigation map error: {e}")


@app.post("/navigation/path")
async def navigation_path(req: NavigationRequest):
    """Plan a navigation route without executing it."""
    _ensure_emulator()
    try:
        snapshot, location_map, plan = await _run_sync(
            _plan_navigation_sync,
            req.x,
            req.y,
            req.mode,
        )
        payload = _serialize_navigation(snapshot, location_map, goal=(req.x, req.y))
        payload["plan"] = plan.to_dict()
        return payload
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
        snapshot_before, location_map_before, plan = await _run_sync(
            _plan_navigation_sync,
            req.x,
            req.y,
            req.mode,
        )
        state_before = await _run_sync(_get_state_dict)
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
                "state_before": state_before,
                "state_after": bundle.get("state") or state_before,
                "navigation_before": navigation_before,
                "navigation_after": bundle.get("navigation") or navigation_before,
                "feedback": bundle.get("recent_action"),
                "state_delta": bundle.get("state_delta"),
                "screen_text": bundle.get("screen_text"),
                "objective_status": (bundle.get("objective") or {}).get("current"),
                "stuck_signal": bundle.get("stuck"),
                "artifacts": bundle.get("artifacts"),
            }

        executed = 0
        for action in plan.actions:
            await _execute_action(action)
            executed += 1

        state_after = await _run_sync(_get_state_dict)
        snapshot_after, location_map_after = await _run_sync(_refresh_navigation_state_sync)
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
            "actions_executed": executed,
            "plan": plan.to_dict(),
            "execution": execution,
            "state_before": state_before,
            "state_after": state_after,
            "navigation_after": navigation_after,
            "navigation_before": navigation_before,
            "feedback": bundle.get("recent_action"),
            "state_delta": bundle.get("state_delta"),
            "screen_text": bundle.get("screen_text"),
            "objective_status": (bundle.get("objective") or {}).get("current"),
            "stuck_signal": bundle.get("stuck"),
            "artifacts": bundle.get("artifacts"),
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
