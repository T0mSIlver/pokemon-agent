from pathlib import Path

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
    assert dashboard["visuals"]["ui_mode"] == "overworld"
    assert dashboard["world_state"]["map"]["map_name"] == "Viridian City"
    assert (
        dashboard["memory_and_progress"]["knowledge_graph_summary"]["current_location"]
        == "Viridian City"
    )


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

    with TestClient(server.app) as client:
        response = client.get("/dashboard/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_intent"]["objective"]["title"]
    assert payload["world_state"]["map"]["map_name"] == "Viridian City"


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
