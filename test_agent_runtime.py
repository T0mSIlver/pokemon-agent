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
from pokemon_agent.harness.contracts import TurnPlanInput
from pokemon_agent.harness.planning import (
    build_plan_execution_trace,
    evaluate_plan_outcome,
    mark_plan_executed,
    store_validated_plan,
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
        self.continued = False
        self.stopped = False

    def state_snapshot(self) -> dict:
        return {
            "available": True,
            "status": "idle",
            "status_reason": "Idle for tests.",
            "next_auto_continue_at": None,
            "config": {
                "auto_continue": True,
                "goal": "",
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

    async def continue_once(self) -> dict:
        self.continued = True
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
    bag: list[dict] | None = None,
    enemy_types: list[str] | None = None,
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
                "types": ["Grass", "Poison"],
                "moves": [
                    {"name": "Tackle", "pp": 35},
                    {"name": "Vine Whip", "pp": 10},
                ],
            }
        ],
        "bag": bag if bag is not None else [],
        "battle": {
            "in_battle": battle,
            "type": "trainer" if battle else "none",
            "enemy": (
                {
                    "species": "Geodude",
                    "level": 12,
                    "hp": 30,
                    "max_hp": 30,
                    "status": "OK",
                    "types": enemy_types or ["Rock", "Ground"],
                    "moves": ["Tackle"],
                }
                if battle
                else None
            ),
        },
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
        signs=[{"x": 8, "y": 8, "text_id": 3}],
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
    ss_ticket = engine.evaluate(
        make_state(
            map_name="Cerulean City",
            has_pokedex=True,
            badge_count=1,
            bag=[{"item": "S.S. Ticket", "quantity": 1}],
        )
    )
    cut = engine.evaluate(
        make_state(
            map_name="Vermilion City",
            has_pokedex=True,
            badge_count=1,
            bag=[{"item": "HM01", "quantity": 1}],
        )
    )

    assert after_pokedex["current"]["id"] == "head_to_viridian_forest"
    assert pewter["current"]["id"] == "reach_pewter_gym"
    assert brock["current"]["id"] == "defeat_brock"
    assert complete["current"]["id"] == "cross_mt_moon_to_cerulean"
    assert ss_ticket["current"]["id"] == "head_to_vermilion_with_ticket"
    assert cut["current"]["id"] == "phase_complete_cut_access"


def test_render_navigation_overlay_draws_on_image():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()

    overlay = render_navigation_overlay(image, snapshot, objective={"title": "Test Objective"})

    assert overlay.width > image.width
    assert overlay.height > image.height
    diff = ImageChops.difference(Image.new("RGB", overlay.size, color=(0, 0, 0)), overlay)
    assert diff.getbbox() is not None


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
        json.dumps(
            {
                "version": 1,
                "observation_id": "obs-prev",
                "objective_id": "head_to_viridian_forest",
                "intent": "Move toward the forest gate.",
                "mode": "overworld",
                "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
                "fallback_branch": {"kind": "raw_actions", "actions": ["press_b"]},
                "expected_outcome": {
                    "summary": "Move one tile north.",
                    "position_delta": {"dx": 0, "dy": -1},
                },
                "notes": "short probe",
                "updated_at": "2026-04-09T12:00:00Z",
                "status": {
                    "state": "validated",
                    "observation_id": "obs-prev",
                    "plan_updated_at": "2026-04-09T12:00:00Z",
                    "validated_at": "2026-04-09T12:00:00Z",
                    "reason": "Validated against the latest turn_context.",
                },
            }
        ),
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
    assert Path(bundle["artifacts"]["turn_context_json"]).exists()
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
    turn_context = json.loads(
        Path(bundle["artifacts"]["turn_context_json"]).read_text(encoding="utf-8")
    )
    observation_json = json.loads(
        Path(bundle["artifacts"]["latest_observation_json"]).read_text(encoding="utf-8")
    )
    assert "Mandatory: inspect the annotated frame before choosing actions." in observation_md
    assert "Do not infer terrain from ASCII alone." in observation_md
    assert "latest_observation_json" not in observation_json["artifacts"]
    assert "run_log_jsonl" not in observation_json["artifacts"]
    assert "ASCII is symbolic only" in observation_json["navigation"]["ascii_note"]
    assert observation_json["navigation"]["coordinate_system"] == "map_tile_absolute"
    assert "absolute map tile coordinates" in observation_json["navigation"]["coordinate_note"]
    assert "fields" not in observation_json["state_delta"]
    assert observation_json["navigation"]["route_cards"]
    assert observation_json["navigation"]["frontiers"]
    assert observation_json["navigation"]["landmarks"]
    assert observation_json["navigation"]["distance_ascii"]
    assert turn_context["observation_id"].startswith("obs-")
    assert turn_context["planning"]["observation_id"] == turn_context["observation_id"]
    assert turn_context["planning"]["objective_id"] == turn_context["objective"]["id"]
    assert turn_context["planning"]["branch_templates"]["raw_actions"]["kind"] == "raw_actions"
    assert turn_context["planning"]["expected_outcome_check_fields"] == [
        "map_name",
        "position",
        "position_delta",
        "dialog_active",
        "battle_active",
    ]
    assert turn_context["navigation"]["route_hints"]
    assert turn_context["navigation"]["coordinate_system"] == "map_tile_absolute"
    assert turn_context["plan_status"]["state"] == "stale"
    assert observation_json["memory"]["recent_facts"]
    assert observation_json["memory"]["session_brief_path"].endswith("session_brief.md")
    assert observation_json["dialog"]["should_continue"] is False
    assert observation_json["battle"]["recommended_mode"] == "none"
    assert Path(bundle["artifacts"]["landmarks_json"]).exists()
    assert Path(bundle["artifacts"]["event_memory_jsonl"]).exists()
    assert Path(bundle["artifacts"]["session_brief_md"]).exists()
    assert Path(bundle["artifacts"]["latest_observation_json"]).parent.name == "debug"
    assert Path(bundle["artifacts"]["session_brief_md"]).parent.name == "debug"


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


def test_agent_runtime_moves_debug_artifacts_out_of_workspace_root(tmp_path: Path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    legacy_latest_observation = workspace_dir / "latest_observation.json"
    legacy_working_memory = workspace_dir / "working_memory.md"
    legacy_latest_observation.write_text("{}", encoding="utf-8")
    legacy_working_memory.write_text("# Legacy\n", encoding="utf-8")

    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=workspace_dir,
    )

    assert runtime.artifacts["latest_observation_json"].exists()
    assert runtime.artifacts["working_memory_md"].exists()
    assert runtime.artifacts["latest_observation_json"].parent.name == "debug"
    assert runtime.artifacts["working_memory_md"].parent.name == "debug"
    assert not legacy_latest_observation.exists()
    assert not legacy_working_memory.exists()


def test_evaluate_plan_outcome_uses_execution_baseline_for_position_delta():
    plan = store_validated_plan(
        TurnPlanInput.model_validate(
            {
                "observation_id": "obs-123",
                "objective_id": "get_oaks_parcel",
                "intent": "Step north once.",
                "mode": "overworld",
                "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
                "expected_outcome": {
                    "summary": "Move one tile north.",
                    "position_delta": {"dx": 0, "dy": -1},
                },
            }
        )
    )
    plan = mark_plan_executed(
        plan,
        branch="primary",
        branch_kind="raw_actions",
        requested_actions=["walk_up"],
        executed_actions=1,
        baseline_map_name="Viridian House",
        baseline_position={"x": 2, "y": 7},
        summary="Executed 1 raw action(s).",
    )

    evaluated = evaluate_plan_outcome(
        plan,
        {
            "state": make_state(map_name="Viridian House", map_id=44, x=2, y=6),
            "state_delta": {"movement": {"dx": -19, "dy": -3}},
            "screen_text": {"text": "No readable screen text extracted."},
        },
    )

    assert evaluated.status.state == "matched"
    assert evaluated.status.reason == "Expected outcome matched."


def _build_multi_walk_plan_and_execute(
    *,
    requested: int,
    executed: int,
    baseline_xy: tuple[int, int],
    current_xy: tuple[int, int],
    map_name: str = "Viridian City",
    map_id: int = 1,
):
    plan = store_validated_plan(
        TurnPlanInput.model_validate(
            {
                "observation_id": "obs-batch",
                "objective_id": "get_oaks_parcel",
                "intent": f"Walk left {requested} tiles.",
                "mode": "overworld",
                "primary_branch": {
                    "kind": "raw_actions",
                    "actions": ["walk_left"] * requested,
                },
                "expected_outcome": {
                    "summary": f"Move {requested} tiles left.",
                    "position_delta": {"dx": -requested, "dy": 0},
                },
            }
        )
    )
    plan = mark_plan_executed(
        plan,
        branch="primary",
        branch_kind="raw_actions",
        requested_actions=["walk_left"] * requested,
        executed_actions=executed,
        baseline_map_name=map_name,
        baseline_position={"x": baseline_xy[0], "y": baseline_xy[1]},
        summary=f"Executed {executed}/{requested} raw action(s).",
    )
    bundle = {
        "state": make_state(
            map_name=map_name,
            map_id=map_id,
            x=current_xy[0],
            y=current_xy[1],
        ),
        "state_delta": {
            "movement": {
                "dx": current_xy[0] - baseline_xy[0],
                "dy": current_xy[1] - baseline_xy[1],
            }
        },
        "screen_text": {"text": ""},
    }
    return plan, bundle


def test_evaluate_plan_outcome_marks_blocked_batch_as_partial_not_drifted():
    plan, bundle = _build_multi_walk_plan_and_execute(
        requested=4,
        executed=1,
        baseline_xy=(20, 16),
        current_xy=(19, 16),
    )
    evaluated = evaluate_plan_outcome(plan, bundle)
    assert evaluated.status.state == "partial"
    assert "Batch blocked after 1/4 steps" in evaluated.status.reason


def test_build_plan_execution_trace_exposes_collision_details():
    plan, bundle = _build_multi_walk_plan_and_execute(
        requested=4,
        executed=1,
        baseline_xy=(20, 16),
        current_xy=(19, 16),
    )
    evaluated = evaluate_plan_outcome(plan, bundle)
    trace = build_plan_execution_trace(evaluated, bundle)
    assert trace is not None
    assert "walk_left x4 requested" in trace
    assert "1/4 executed" in trace
    assert "delta=(-1,0)" in trace
    assert "outcome=partial" in trace
    assert "Batch blocked after 1/4 steps" in trace


def test_classify_action_feedback_uses_plan_trace_as_summary():
    state_before = make_state(x=20, y=16)
    state_after = make_state(x=19, y=16)
    delta = build_state_delta(state_before, state_after)
    feedback = classify_action_feedback(
        source="action",
        requested_actions=["walk_left"] * 4,
        state_before=state_before,
        state_after=state_after,
        state_delta=delta,
        plan_execution_trace=(
            "walk_left x4 requested; 1/4 executed; delta=(-1,0); outcome=drifted"
        ),
        plan_state="drifted",
    )
    assert feedback["summary"].startswith("walk_left x4 requested")
    assert feedback["plan_state"] == "drifted"
    assert "plan_drifted" in feedback["tags"]
    assert feedback["notes"][0] == feedback["summary"]
    assert "Player position changed." not in feedback["notes"]


def test_agent_runtime_escalates_stuck_on_repeated_drifts(tmp_path: Path, monkeypatch):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    state_holder = {"x": 16, "y": 16}

    def current_state() -> dict:
        return make_state(
            map_name="Viridian City",
            map_id=1,
            has_pokedex=True,
            x=state_holder["x"],
            y=state_holder["y"],
        )

    for turn in range(4):
        observe = runtime.refresh(
            emulator=emulator,
            state=current_state(),
            navigation=navigation,
            navigation_store=store,
            reason=f"observe_{turn}",
            source="observe",
        )
        observation_id = observe["bundle"]["observation_id"]
        runtime.validate_and_store_turn_plan(
            {
                "observation_id": observation_id,
                "objective_id": observe["bundle"]["objective"]["current"]["id"],
                "intent": "Walk left into the mart.",
                "mode": "overworld",
                "primary_branch": {
                    "kind": "raw_actions",
                    "actions": ["walk_left"],
                },
                "expected_outcome": {
                    "summary": "Move one tile left.",
                    "position": {"x": 15, "y": 16},
                },
            }
        )
        runtime.mark_turn_plan_executed(
            branch="primary",
            branch_kind="raw_actions",
            requested_actions=["walk_left"],
            executed_actions=1,
            baseline_map_name="Viridian City",
            baseline_position={"x": 16, "y": 16},
            summary="Executed 1 raw action(s).",
        )
        result = runtime.refresh(
            emulator=emulator,
            state=current_state(),
            navigation=navigation,
            navigation_store=store,
            reason=f"after_act_{turn}",
            source="action",
            requested_actions=["walk_left"],
        )

    stuck = result["bundle"]["stuck"]
    assert stuck["level"] in {"warning", "danger"}
    assert stuck["drift_count"] >= 3
    assert any("navigation" in action for action in stuck["recommended_actions"])


def test_discover_landmarks_ignores_blocked_tile_interactions(tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = LiveNavigationSnapshot(
        map_id=1,
        map_name="Viridian City",
        player_position=(16, 16),
        facing="left",
        tileset="OVERWORLD",
        window_top_left=(12, 12),
        terrain=[[1 for _ in range(10)] for _ in range(9)],
        sprite_positions=[],
        valid_moves=["up", "down", "right"],
        warps=[],
        signs=[],
        map_dimensions={"width": 20, "height": 18},
        interaction={
            "kind": "background",
            "source": "blocked_tile",
            "reason": "Forward movement is blocked by a non-passable background tile.",
            "target_coord": {"x": 15, "y": 16},
        },
    )
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, x=16, y=16),
        navigation=navigation,
        navigation_store=store,
        reason="probe_wall",
        source="observe",
    )
    landmarks = json.loads(runtime.artifacts["landmarks_json"].read_text(encoding="utf-8"))
    entries = landmarks.get("landmarks") or []
    blocked_entries = [
        entry
        for entry in entries
        if entry.get("coord", {}).get("x") == 15 and entry.get("coord", {}).get("y") == 16
    ]
    assert blocked_entries == []


def test_turn_context_surfaces_warps_with_target_map_names(tmp_path: Path):
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
        reason="warp_context",
        source="observe",
    )
    turn_context = json.loads(runtime.artifacts["turn_context_json"].read_text(encoding="utf-8"))
    warps = turn_context["navigation"]["warps"]
    assert warps, "expected warps to be surfaced in turn_context"
    first_warp = warps[0]
    assert first_warp["coord"] == {"x": 10, "y": 4}
    assert first_warp["target_map_name"] == "Viridian Mart"
    assert first_warp["distance"] == 6


def test_agent_runtime_records_landmarks_and_failure_memory(tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    for _ in range(4):
        runtime.refresh(
            emulator=emulator,
            state=make_state(map_name="Viridian Forest", map_id=50, has_pokedex=True, x=10, y=10),
            navigation=navigation,
            navigation_store=store,
            reason="repeat_action",
            source="action",
            requested_actions=["walk_up"],
        )

    landmarks = json.loads(runtime.artifacts["landmarks_json"].read_text(encoding="utf-8"))
    event_lines = runtime.artifacts["event_memory_jsonl"].read_text(encoding="utf-8").splitlines()
    event_payloads = [json.loads(line) for line in event_lines if line.strip()]
    session_brief = runtime.artifacts["session_brief_md"].read_text(encoding="utf-8")
    observation = json.loads(
        runtime.artifacts["latest_observation_json"].read_text(encoding="utf-8")
    )

    assert any(entry["kind"] == "sign" for entry in landmarks["landmarks"])
    assert any(entry["kind"] == "npc_blocker" for entry in landmarks["landmarks"])
    assert any(entry["kind"] == "failure" for entry in event_payloads)
    assert "Failed Attempts" in session_brief
    assert observation["memory"]["failed_attempts"]


def test_dialog_and_battle_guidance_are_exposed(tmp_path: Path):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )

    result = runtime.refresh(
        emulator=emulator,
        state=make_state(
            map_name="Pewter Gym",
            map_id=53,
            has_pokedex=True,
            badge_count=0,
            battle=True,
            enemy_types=["Rock", "Ground"],
        ),
        navigation=navigation,
        navigation_store=store,
        reason="battle_test",
        source="observe",
    )
    bundle = result["bundle"]

    assert bundle["battle_guidance"]["recommended_mode"] == "select_best_move"
    assert bundle["battle_guidance"]["recommended_move"]["name"] == "Vine Whip"

    dialog_result = runtime.refresh(
        emulator=emulator,
        state=make_state(
            map_name="Oak's Lab",
            map_id=40,
            dialog_active=True,
        ),
        navigation=navigation,
        navigation_store=store,
        reason="dialog_test",
        source="observe",
    )
    dialog_bundle = dialog_result["bundle"]

    assert dialog_bundle["dialog_guidance"]["should_continue"] is True
    assert isinstance(dialog_bundle["dialog_guidance"]["transcript_recent"], list)


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
    assert payload["artifact_urls"]["turn_context_json"] == "/artifacts/turn_context_json"


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
    assert payload["observation_id"].startswith("obs-")
    assert payload["reason"] == "contract_test"
    assert payload["artifacts"]["latest_frame"].endswith("latest_frame.png")
    assert Path(payload["artifacts"]["turn_context_json"]).exists()
    assert payload["plan_status"]["state"] == "awaiting_plan"
    assert "raw_frame_b64" not in payload


def test_agent_observe_endpoint_accepts_empty_body(tmp_path: Path, monkeypatch):
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
        lambda: make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
    )
    monkeypatch.setattr(server, "_get_navigation_payload_sync", lambda goal=None: navigation)

    with TestClient(server.app) as client:
        response = client.post("/agent/observe")

    assert response.status_code == 200
    assert response.json()["reason"] == "manual_observe"


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


def test_agent_plan_endpoint_validates_and_persists_strict_turn_plan(tmp_path: Path, monkeypatch):
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
        lambda: make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
    )
    monkeypatch.setattr(server, "_get_navigation_payload_sync", lambda goal=None: navigation)

    with TestClient(server.app) as client:
        observe = client.post("/agent/observe", json={"reason": "plan_contract"}).json()
        valid_payload = {
            "observation_id": observe["observation_id"],
            "objective_id": "head_to_viridian_forest",
            "intent": "Probe one tile north.",
            "mode": "overworld",
            "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
            "fallback_branch": {"kind": "raw_actions", "actions": ["press_b"]},
            "expected_outcome": {
                "summary": "Move one tile north.",
                "position_delta": {"dx": 0, "dy": -1},
            },
            "notes": "short probe",
        }
        valid_response = client.post("/agent/plan", json=valid_payload)
        invalid_response = client.post(
            "/agent/plan",
            json={**valid_payload, "objective_id": "wrong_objective"},
        )

    assert valid_response.status_code == 200
    assert invalid_response.status_code == 400
    assert valid_response.json()["plan_status"]["state"] == "validated"
    assert runtime.load_turn_plan_model().status.state == "validated"


def test_agent_act_endpoint_executes_validated_plan_and_updates_result_on_observe(
    tmp_path: Path,
    monkeypatch,
):
    emulator = FakeEmulator()
    store = NavigationStore(tmp_path / "navigation.json")
    snapshot = make_snapshot()
    navigation = make_navigation_payload(store, snapshot)
    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
    )
    state_holder = {"x": 10, "y": 10}

    def fake_state() -> dict:
        return make_state(
            map_name="Viridian City",
            map_id=1,
            has_pokedex=True,
            x=state_holder["x"],
            y=state_holder["y"],
        )

    def fake_execute(actions: list[str]) -> int:
        for action in actions:
            if action == "walk_up":
                state_holder["y"] -= 1
        return len(actions)

    monkeypatch.setattr(server, "_runtime", runtime)
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_navigation_store", store)
    monkeypatch.setattr(server, "_get_state_dict", fake_state)
    monkeypatch.setattr(server, "_get_navigation_payload_sync", lambda goal=None: navigation)
    monkeypatch.setattr(server, "_execute_action_batch_sync", fake_execute)

    with TestClient(server.app) as client:
        observe = client.post("/agent/observe", json={"reason": "act_contract"}).json()
        plan_response = client.post(
            "/agent/plan",
            json={
                "observation_id": observe["observation_id"],
                "objective_id": "head_to_viridian_forest",
                "intent": "Step north once.",
                "mode": "overworld",
                "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
                "expected_outcome": {
                    "summary": "Move one tile north.",
                    "position_delta": {"dx": 0, "dy": -1},
                },
            },
        )
        act_response = client.post("/agent/act")
        observe_after = client.post("/agent/observe", json={"reason": "after_act"})

    assert plan_response.status_code == 200
    assert act_response.status_code == 200
    assert act_response.json()["plan_status"]["state"] == "executed_waiting_observe"
    assert act_response.json()["requires_observe"] is True
    assert observe_after.status_code == 200
    assert observe_after.json()["plan_status"]["state"] == "matched"
    assert runtime.load_turn_plan_model().status.state == "matched"


def test_navigator_endpoint_returns_best_route(tmp_path: Path, monkeypatch):
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
        reason="navigator_contract",
        source="observe",
    )
    monkeypatch.setattr(server, "_runtime", runtime)

    with TestClient(server.app) as client:
        response = client.get("/agent/navigator")

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_route"] is not None
    assert payload["alternatives"] is not None


def test_supervisor_endpoints_delegate_to_supervisor(monkeypatch):
    fake_supervisor = FakeSupervisorAPI()
    monkeypatch.setattr(server, "_supervisor", fake_supervisor)

    with TestClient(server.app) as client:
        start_response = client.post(
            "/supervisor/start",
            json={
                "goal": "play",
                "provider": "openai",
                "model": "local-model",
                "thinking": "low",
                "auto_continue": False,
            },
        )
        continue_response = client.post("/supervisor/continue", json={})
        stop_response = client.post("/supervisor/stop")

    assert start_response.status_code == 200
    assert continue_response.status_code == 200
    assert stop_response.status_code == 200
    assert fake_supervisor.started_with is not None
    assert fake_supervisor.started_with["goal"] == "play"
    assert fake_supervisor.continued is True
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
