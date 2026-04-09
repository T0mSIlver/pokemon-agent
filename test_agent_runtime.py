import asyncio
import contextlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops

import pokemon_agent.server as server
from pokemon_agent.agent_runtime import (
    AgentRuntime,
    ObjectiveEngine,
    build_state_delta,
    classify_action_feedback,
    render_navigation_overlay,
)
from pokemon_agent.navigation import LiveNavigationSnapshot, NavigationStore


class FakeEmulator:
    def __init__(self) -> None:
        self.frame_count = 1234
        self.saved_paths: list[str] = []
        self.image = Image.new("RGB", (160, 144), color=(16, 24, 32))

    def get_screen(self):
        return self.image

    def save_state(self, path: str) -> None:
        Path(path).write_bytes(b"state")
        self.saved_paths.append(path)

    def tick(self, frames: int = 1) -> None:
        self.frame_count += frames


class FakeSupervisorAPI:
    def __init__(self) -> None:
        self.started_with = None
        self.continued_with = None
        self.stopped = False

    def state_snapshot(self) -> dict:
        return {
            "available": True,
            "status": "idle",
            "status_reason": "Idle for tests.",
            "next_auto_continue_at": None,
            "config": {
                "auto_continue": True,
                "continue_message": "continue",
                "continue_delay_seconds": 1.0,
            },
            "recent_tools": [],
            "recent_events": [],
            "active_tools": [],
            "stderr_tail": [],
            "transcript": [],
            "default_prompt": "default prompt",
        }

    async def start(self, **kwargs) -> dict:
        self.started_with = kwargs
        snapshot = self.state_snapshot()
        snapshot["status"] = "starting"
        return snapshot

    async def continue_once(self, *, message: str) -> dict:
        self.continued_with = message
        snapshot = self.state_snapshot()
        snapshot["status"] = "starting"
        return snapshot

    async def stop(self) -> dict:
        self.stopped = True
        snapshot = self.state_snapshot()
        snapshot["status"] = "stopped"
        return snapshot

    async def shutdown(self) -> dict:
        return await self.stop()


def make_state(
    *,
    map_name: str = "Pallet Town",
    map_id: int = 0,
    x: int = 5,
    y: int = 6,
    dialog_active: bool = False,
    battle: bool = False,
    has_oaks_parcel: bool = False,
    has_pokedex: bool = False,
    badge_count: int = 0,
) -> dict:
    return {
        "metadata": {"frame_count": 1234},
        "map": {"map_id": map_id, "map_name": map_name},
        "player": {
            "name": "RED",
            "position": {"x": x, "y": y},
            "facing": "up",
            "money": 3000,
            "badge_count": badge_count,
            "badges": [],
        },
        "party": [
            {
                "nickname": "Bulbasaur",
                "species": "Bulbasaur",
                "level": 8,
                "hp": 20,
                "max_hp": 22,
                "status": "OK",
                "moves": [{"name": "Tackle"}],
            }
        ],
        "battle": {"in_battle": battle, "type": "trainer" if battle else "none"},
        "dialog": {
            "active": dialog_active,
            "waiting_for_input": dialog_active,
            "printing": False,
        },
        "dialog_active": dialog_active,
        "flags": {
            "has_oaks_parcel": has_oaks_parcel,
            "has_pokedex": has_pokedex,
            "badge_count": badge_count,
        },
    }


def make_snapshot() -> LiveNavigationSnapshot:
    return LiveNavigationSnapshot(
        map_id=1,
        map_name="Viridian City",
        player_position=(10, 10),
        facing="up",
        tileset="OVERWORLD",
        window_top_left=(6, 6),
        terrain=[[1 for _ in range(10)] for _ in range(9)],
        sprite_positions=[(10, 9)],
        valid_moves=["up", "left", "right"],
        warps=[{"x": 10, "y": 4, "warp_id": 1, "target_map_id": 42}],
        map_dimensions={"width": 20, "height": 18},
        interaction={
            "kind": "object",
            "source": "sprite_direct",
            "reason": "NPC detected in front of the player.",
            "target_coord": {"x": 10, "y": 9},
        },
    )


def make_navigation_payload(store: NavigationStore, snapshot: LiveNavigationSnapshot) -> dict:
    location_map = store.update(snapshot)
    distances = location_map.distance_map(
        snapshot.player_position, extra_blockers=snapshot.sprite_set
    )
    return {
        "snapshot": snapshot.to_dict(),
        "location_map": location_map.to_dict(
            player=snapshot.player_position,
            sprites=snapshot.sprite_positions,
            distances=distances,
        ),
    }


def test_build_state_delta_detects_position_and_dialog_changes():
    before = make_state(map_name="Route 1", x=4, y=5, dialog_active=False)
    after = make_state(map_name="Route 1", x=4, y=6, dialog_active=True)

    delta = build_state_delta(before, after)

    assert delta["changed"] is True
    assert "position" in delta["fields"]
    assert delta["movement"]["dy"] == 1
    assert "dialog_active" in delta["fields"]


def test_classify_action_feedback_marks_no_progress():
    state = make_state()
    delta = build_state_delta(state, state)

    feedback = classify_action_feedback(
        source="action",
        requested_actions=["walk_up"],
        state_before=state,
        state_after=state,
        state_delta=delta,
    )

    assert "no_progress" in feedback["tags"]
    assert feedback["summary"] == "Structured state did not change after the requested actions."


def test_objective_engine_reaches_brock_phase():
    engine = ObjectiveEngine()

    after_pokedex = engine.evaluate(make_state(map_name="Viridian City", has_pokedex=True))
    pewter = engine.evaluate(make_state(map_name="Pewter City", has_pokedex=True))
    brock = engine.evaluate(make_state(map_name="Pewter Gym", has_pokedex=True, battle=True))
    complete = engine.evaluate(make_state(map_name="Pewter City", has_pokedex=True, badge_count=1))

    assert after_pokedex["current"]["id"] == "head_to_viridian_forest"
    assert pewter["current"]["id"] == "reach_pewter_gym"
    assert brock["current"]["id"] == "defeat_brock"
    assert complete["current"]["id"] == "phase_complete_boulder_badge"


def test_render_navigation_overlay_draws_on_image():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()

    overlay = render_navigation_overlay(image, snapshot, objective={"title": "Test Objective"})

    assert overlay.size == image.size
    assert ImageChops.difference(image, overlay).getbbox() is not None


def test_agent_runtime_refresh_writes_workspace_and_dashboard_state(tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    runtime.artifacts["turn_plan_json"].write_text(
        """
{
  "objective_id": "head_to_viridian_forest",
  "summary": "Move toward the forest gate.",
  "planned_actions": ["walk_up"],
  "fallback_actions": ["press_b"],
  "notes": "short probe",
  "updated_at": "2026-04-09T12:00:00Z"
}
""".strip(),
        encoding="utf-8",
    )

    result = runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
        navigation=navigation,
        navigation_store=store,
        reason="test_refresh",
        source="observe",
    )
    bundle = result["bundle"]
    dashboard = runtime.dashboard_state()

    assert Path(bundle["artifacts"]["latest_frame"]).exists()
    assert Path(bundle["artifacts"]["latest_frame_annotated"]).exists()
    assert Path(bundle["artifacts"]["latest_observation_json"]).exists()
    assert dashboard["agent_intent"]["turn_plan"]["summary"] == "Move toward the forest gate."
    assert dashboard["agent_intent"]["movement_guidance"]["notes"]
    assert dashboard["visuals"]["ui_mode"] == "overworld"
    assert dashboard["world_state"]["map"]["map_name"] == "Viridian City"
    assert (
        dashboard["memory_and_progress"]["knowledge_graph_summary"]["current_location"]
        == "Viridian City"
    )
    observation_md = Path(bundle["artifacts"]["latest_observation_md"]).read_text(encoding="utf-8")
    observation_json = json.loads(
        Path(bundle["artifacts"]["latest_observation_json"]).read_text(encoding="utf-8")
    )
    assert "Mandatory: inspect the annotated frame before choosing actions." in observation_md
    assert "Do not infer terrain from ASCII alone." in observation_md
    assert "latest_observation_json" not in observation_json["artifacts"]
    assert "run_log_jsonl" not in observation_json["artifacts"]
    assert "ASCII is symbolic only" in observation_json["navigation"]["ascii_note"]
    assert "fields" not in observation_json["state_delta"]


def test_agent_runtime_detects_repeated_no_movement(tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    stuck_level = "clear"
    for _ in range(4):
        result = runtime.refresh(
            emulator=emulator,
            state=make_state(map_name="Viridian Forest", map_id=50, has_pokedex=True, x=10, y=10),
            navigation=navigation,
            navigation_store=store,
            reason="repeat_action",
            source="action",
            requested_actions=["walk_up"],
        )
        stuck_level = result["bundle"]["stuck"]["level"]

    assert stuck_level in {"warning", "danger"}


def test_dashboard_state_endpoint_uses_runtime_bundle(tmp_path: Path, monkeypatch):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
        navigation=navigation,
        navigation_store=store,
        reason="dashboard_contract",
        source="observe",
    )
    monkeypatch.setattr(server, "_runtime", runtime)
    monkeypatch.setattr(server, "_supervisor", FakeSupervisorAPI())

    with TestClient(server.app) as client:
        response = client.get("/dashboard/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_intent"]["objective"]["title"]
    assert payload["world_state"]["map"]["map_name"] == "Viridian City"
    assert payload["pi_supervisor"]["status"] == "idle"
    assert "realtime_enabled" in payload["server_runtime"]
    assert (
        payload["artifact_urls"]["latest_observation_json"]
        == "/artifacts/latest_observation_json"
    )


def test_agent_observe_endpoint_returns_workspace_bundle(tmp_path: Path, monkeypatch):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    monkeypatch.setattr(server, "_runtime", runtime)
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_navigation_store", store)
    monkeypatch.setattr(
        server,
        "_get_state_dict",
        lambda: make_state(
            map_name="Viridian City",
            map_id=1,
            has_pokedex=True,
            x=10,
            y=10,
        ),
    )
    monkeypatch.setattr(server, "_get_navigation_payload_sync", lambda goal=None: navigation)

    with TestClient(server.app) as client:
        response = client.post("/agent/observe", json={"reason": "contract_test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["reason"] == "contract_test"
    assert payload["artifacts"]["latest_frame"].endswith("latest_frame.png")
    assert Path(payload["artifacts"]["latest_observation_json"]).exists()
    assert "raw_frame_b64" not in payload


def test_artifact_endpoint_serves_workspace_files(tmp_path: Path, monkeypatch):
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    runtime.artifacts["working_memory_md"].write_text("# Notes\n", encoding="utf-8")
    monkeypatch.setattr(server, "_runtime", runtime)

    with TestClient(server.app) as client:
        response = client.get("/artifacts/working_memory_md")

    assert response.status_code == 200
    assert "# Notes" in response.text


def test_supervisor_endpoints_delegate_to_supervisor(monkeypatch):
    fake_supervisor = FakeSupervisorAPI()
    monkeypatch.setattr(server, "_supervisor", fake_supervisor)

    with TestClient(server.app) as client:
        start_response = client.post(
            "/supervisor/start",
            json={
                "prompt": "play",
                "provider": "openai",
                "model": "local-model",
                "thinking": "low",
                "auto_continue": False,
                "continue_message": "continue",
            },
        )
        continue_response = client.post("/supervisor/continue", json={"message": "continue"})
        stop_response = client.post("/supervisor/stop")

    assert start_response.status_code == 200
    assert continue_response.status_code == 200
    assert stop_response.status_code == 200
    assert fake_supervisor.started_with is not None
    assert fake_supervisor.started_with["prompt"] == "play"
    assert fake_supervisor.continued_with == "continue"
    assert fake_supervisor.stopped is True


@pytest.mark.asyncio
async def test_realtime_loop_advances_emulator_frames(monkeypatch):
    emulator = FakeEmulator()
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_emulator_lock", asyncio.Lock())
    monkeypatch.setattr(server, "_realtime_frames_per_second", 120)
    monkeypatch.setattr(server, "_realtime_ticks", 0)
    monkeypatch.setattr(server, "_realtime_last_tick_at", None)

    task = asyncio.create_task(server._realtime_emulator_loop())
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert emulator.frame_count > 1234
    assert server._realtime_ticks > 0


@pytest.mark.asyncio
async def test_live_artifact_loop_refreshes_workspace_and_broadcasts(monkeypatch, tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    events: list[dict] = []

    async def fake_broadcast(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(server, "_runtime", runtime)
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_navigation_store", store)
    monkeypatch.setattr(server, "_emulator_lock", asyncio.Lock())
    monkeypatch.setattr(server, "_live_artifact_frames_per_second", 120)
    monkeypatch.setattr(server, "_live_artifact_last_sync_at", None)
    monkeypatch.setattr(
        server,
        "_get_state_dict",
        lambda: make_state(map_name="Route 1", map_id=12),
    )
    monkeypatch.setattr(server, "_get_live_navigation_payload_sync", lambda goal=None: navigation)
    monkeypatch.setattr(server, "broadcast", fake_broadcast)

    task = asyncio.create_task(server._live_artifact_loop())
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert Path(runtime.artifacts["latest_frame"]).exists()
    assert Path(runtime.artifacts["latest_frame_annotated"]).exists()
    assert server._live_artifact_last_sync_at is not None
    assert any(event.get("type") == "screenshot" for event in events)
    screenshot_events = [event for event in events if event.get("type") == "screenshot"]
    assert screenshot_events[-1]["data"]["source"] == "live_sync"
    assert screenshot_events[-1]["data"]["frame_timestamp"]
