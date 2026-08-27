import json
from pathlib import Path

from PIL import Image, ImageChops

from pokemon_agent.agent_runtime import (
    AgentRuntime,
    ObjectiveEngine,
    build_movement_guidance,
    build_state_delta,
    classify_action_feedback,
    render_navigation_overlay,
)
from pokemon_agent.navigation import LiveNavigationSnapshot


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


def make_map_grid() -> dict:
    """A small explored map for Viridian City, matching make_snapshot()."""
    seen = {(x, y) for x in range(20) for y in range(18) if x < 14 and y < 13}
    walkable = {tile for tile in seen if (tile[0] + tile[1]) % 4 != 0}
    walked = {(x, 10) for x in range(4, 12)}
    return {
        "width": 20,
        "height": 18,
        "seen": seen,
        "walkable": walkable,
        "walked": walked,
        "warps": {(10, 4)},
    }


def make_navigation_payload(snapshot: LiveNavigationSnapshot) -> dict:
    return {"snapshot": snapshot.to_dict()}


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
    assert cut["current"]["id"] == "phase_complete_cut_access"


#: A run that has just been handed HM01 by the captain, as the milestone oracle
#: reads it: the whole chain that leads there and nothing past it.
REACHED_AT_HM01 = [
    "EVENT_GOT_STARTER",
    "EVENT_BATTLED_RIVAL_IN_OAKS_LAB",
    "EVENT_GOT_OAKS_PARCEL",
    "EVENT_OAK_GOT_PARCEL",
    "EVENT_GOT_POKEDEX",
    "EVENT_GOT_TOWN_MAP",
    "EVENT_BEAT_BROCK",
    "BADGE_BOULDER",
    "EVENT_BEAT_CERULEAN_RIVAL",
    "EVENT_MET_BILL",
    "EVENT_GOT_SS_TICKET",
    "EVENT_RUBBED_CAPTAINS_BACK",
    "EVENT_GOT_HM01",
]


def state_past_cut(milestones=None) -> dict:
    state = make_state(
        map_name="Vermilion City",
        has_pokedex=True,
        badge_count=1,
        bag=[{"item": "HM01", "quantity": 1}],
    )
    if milestones is not None:
        state["milestones"] = milestones
    return state


def test_written_packs_still_own_everything_before_the_handoff():
    engine = ObjectiveEngine()

    # Milestone data present and rich, at a point the packs still cover: the
    # frontier must not get a word in.
    state = make_state(map_name="Pewter City", has_pokedex=True)
    state["milestones"] = REACHED_AT_HM01

    result = engine.evaluate(state)

    assert result["current"]["id"] == "reach_pewter_gym"
    assert result["current"]["pack_id"] == "red_intro_to_brock"
    assert result["phase_complete"] is False


def test_handoff_objective_comes_from_the_live_frontier():
    engine = ObjectiveEngine()

    result = engine.evaluate(state_past_cut(REACHED_AT_HM01))
    current = result["current"]

    assert result["phase_complete"] is True
    assert current["pack_id"] == "milestone_frontier"
    assert current["id"].startswith("milestone_frontier_")
    # Several real options, each a milestone whose prerequisites are met.
    assert "Defeated Misty" in current["summary"]
    assert "Got the Bike Voucher" in current["summary"]
    # Where it is earned comes first, then what it opens.
    assert (
        "Got HM05 Flash [Route 2 Gate, 6 hops] "
        "(opens dark caves, once the Boulder Badge allows it)" in (current["summary"])
    )
    # The map the state says the player is on is the map distances are from.
    assert "map-graph distance from Vermilion City" in current["summary"]
    # Not a promise about geography.
    assert "not a claim that any of them can be reached on foot" in current["summary"]
    # Behind Cut, which needs a badge this run has not got: not on the frontier,
    # so not in the objective.
    assert "Defeated Lt. Surge" not in current["summary"]
    # Exactly one current objective, and the pack rung it replaced is done.
    assert [item["id"] for item in result["objectives"] if item["current"]] == [current["id"]]
    handoff = [item for item in result["objectives"] if item["id"] == "phase_complete_cut_access"]
    assert handoff and handoff[0]["status"] == "completed"
    assert set(current) == set(result["objectives"][0])


def test_a_changed_frontier_changes_the_objective():
    engine = ObjectiveEngine()

    before = engine.evaluate(state_past_cut(REACHED_AT_HM01))["current"]
    after = engine.evaluate(state_past_cut(REACHED_AT_HM01 + ["EVENT_BEAT_MISTY"]))["current"]

    assert before["id"] != after["id"]
    assert "Defeated Misty" not in after["summary"]
    assert "Cascade Badge" in after["summary"]


def test_the_same_frontier_keeps_the_same_objective_id():
    engine = ObjectiveEngine()

    first = engine.evaluate(state_past_cut(REACHED_AT_HM01))["current"]
    # A milestone that was already banked, arriving in a different order.
    shuffled = list(reversed(REACHED_AT_HM01))
    second = engine.evaluate(state_past_cut(shuffled))["current"]

    assert first["id"] == second["id"]


def test_unreadable_milestones_fall_back_to_the_written_objective():
    engine = ObjectiveEngine()

    for milestones in (None, [], (), 17, {"nope": True}, ["NOT_A_MILESTONE"]):
        result = engine.evaluate(state_past_cut(milestones))
        assert result["current"]["id"] == "phase_complete_cut_access", milestones
        assert result["phase_complete"] is True


def test_finished_ladder_falls_back_rather_than_offering_nothing():
    from pokemon_agent.milestones import MILESTONES

    engine = ObjectiveEngine()

    everything = [milestone.id for milestone in MILESTONES]
    result = engine.evaluate(state_past_cut(everything))

    assert result["current"]["id"] == "phase_complete_cut_access"


def test_objective_record_surfaces_only_the_lean_fields():
    engine = ObjectiveEngine()

    current = engine.evaluate(make_state(map_name="Pewter City", has_pokedex=True))["current"]

    assert {"id", "summary", "completion_predicate"}.issubset(current)
    for dropped in ("title", "route_hint", "target_npcs", "progress_percent"):
        assert dropped not in current


def test_render_navigation_overlay_draws_on_image():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()

    overlay = render_navigation_overlay(image, snapshot, objective={"summary": "Test Objective"})

    assert overlay.width > image.width
    assert overlay.height > image.height
    diff = ImageChops.difference(Image.new("RGB", overlay.size, color=(0, 0, 0)), overlay)
    assert diff.getbbox() is not None


def test_render_navigation_overlay_shades_visited_tiles():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}

    plain = render_navigation_overlay(image, snapshot, objective=objective)
    shaded = render_navigation_overlay(
        image,
        snapshot,
        objective=objective,
        visited={(8, 8), (9, 8), (10, 8), (10, 9), (10, 10)},
    )

    assert ImageChops.difference(plain.convert("RGB"), shaded.convert("RGB")).getbbox() is not None


def test_render_navigation_overlay_ignores_visited_tiles_outside_the_window():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}

    plain = render_navigation_overlay(image, snapshot, objective=objective)
    far_away = render_navigation_overlay(
        image,
        snapshot,
        objective=objective,
        visited={(200, 200), (201, 200)},
    )

    assert ImageChops.difference(plain, far_away).getbbox() is None


def test_render_navigation_overlay_without_visited_is_unchanged():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}

    plain = render_navigation_overlay(image, snapshot, objective=objective)
    explicit_none = render_navigation_overlay(image, snapshot, objective=objective, visited=None)
    empty = render_navigation_overlay(image, snapshot, objective=objective, visited=set())

    assert ImageChops.difference(plain, explicit_none).getbbox() is None
    assert ImageChops.difference(plain, empty).getbbox() is None


def test_runtime_visited_lookup_seam_reaches_the_annotated_frame(tmp_path: Path):
    navigation = make_navigation_payload(make_snapshot())
    state = make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10)

    plain_runtime = AgentRuntime(
        data_dir=tmp_path / "data-plain",
        workspace_dir=tmp_path / "workspace-plain",
    )
    shaded_runtime = AgentRuntime(
        data_dir=tmp_path / "data-shaded",
        workspace_dir=tmp_path / "workspace-shaded",
        visited_lookup=lambda map_id: {(8, 8), (9, 8), (10, 8), (10, 9)},
    )

    plain = plain_runtime.refresh(
        emulator=FakeEmulator(),
        state=state,
        navigation=navigation,
        reason="test_refresh",
        source="observe",
    )
    shaded = shaded_runtime.refresh(
        emulator=FakeEmulator(),
        state=state,
        navigation=navigation,
        reason="test_refresh",
        source="observe",
    )

    plain_frame = Image.open(plain["bundle"]["artifacts"]["latest_frame_annotated"]).convert("RGB")
    shaded_frame = Image.open(shaded["bundle"]["artifacts"]["latest_frame_annotated"]).convert(
        "RGB"
    )

    assert ImageChops.difference(plain_frame, shaded_frame).getbbox() is not None


def test_runtime_survives_a_failing_visited_lookup(tmp_path: Path):
    def boom(map_id: int) -> set:
        raise RuntimeError("explored map store unavailable")

    runtime = AgentRuntime(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
        visited_lookup=boom,
    )

    result = runtime.refresh(
        emulator=FakeEmulator(),
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
        navigation=make_navigation_payload(make_snapshot()),
        reason="test_refresh",
        source="observe",
    )

    assert Path(result["bundle"]["artifacts"]["latest_frame_annotated"]).exists()


def test_render_navigation_overlay_insets_the_explored_map():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}

    plain = render_navigation_overlay(image, snapshot, objective=objective)
    with_map = render_navigation_overlay(
        image,
        snapshot,
        objective=objective,
        map_grid=make_map_grid(),
    )

    # The inset lives in a side panel, so the canvas grows to the right and the
    # game window and header columns come out pixel for pixel as before.
    assert with_map.width > plain.width
    assert with_map.height >= plain.height
    unchanged = with_map.crop((0, 0, plain.width, plain.height))
    assert ImageChops.difference(plain, unchanged).getbbox() is None
    panel = with_map.crop((plain.width, 0, with_map.width, with_map.height))
    assert panel.getbbox() is not None
    assert len(panel.getcolors(maxcolors=1 << 20) or ()) > 4


def test_render_navigation_overlay_inset_marks_the_player_in_cyan():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()

    plain = render_navigation_overlay(image, snapshot, objective={"summary": "Test Objective"})
    with_map = render_navigation_overlay(
        image,
        snapshot,
        objective={"summary": "Test Objective"},
        map_grid=make_map_grid(),
    )

    panel = with_map.crop((plain.width, 0, with_map.width, with_map.height))
    cyan = sum(
        count for count, colour in panel.getcolors(maxcolors=1 << 20) if colour == (55, 208, 255)
    )
    # A solid block, not a single pixel: it has to be findable at a glance.
    assert cyan >= 60


def test_render_navigation_overlay_without_a_map_grid_is_unchanged():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}

    plain = render_navigation_overlay(image, snapshot, objective=objective)
    for grid in (
        None,
        {},
        [1, 2, 3],
        {"width": 0, "height": 0, "seen": set()},
        {"width": "wide", "height": None},
        {"width": 20, "height": 18, "seen": set(), "walkable": set()},
    ):
        degraded = render_navigation_overlay(image, snapshot, objective=objective, map_grid=grid)
        assert degraded.size == plain.size
        assert ImageChops.difference(plain, degraded).getbbox() is None


def test_render_navigation_overlay_ignores_out_of_bounds_inset_tiles():
    image = Image.new("RGB", (160, 144), color=(0, 0, 0))
    snapshot = make_snapshot()
    objective = {"summary": "Test Objective"}
    grid = make_map_grid()
    noisy = dict(grid)
    noisy["seen"] = grid["seen"] | {(999, 999), (-4, 2)}

    clean = render_navigation_overlay(image, snapshot, objective=objective, map_grid=grid)
    with_noise = render_navigation_overlay(image, snapshot, objective=objective, map_grid=noisy)

    assert ImageChops.difference(clean, with_noise).getbbox() is None


def test_runtime_map_grid_lookup_seam_reaches_the_annotated_frame(tmp_path: Path):
    navigation = make_navigation_payload(make_snapshot())
    state = make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10)
    grid = make_map_grid()

    plain_runtime = AgentRuntime(
        data_dir=tmp_path / "data-plain",
        workspace_dir=tmp_path / "workspace-plain",
    )
    inset_runtime = AgentRuntime(
        data_dir=tmp_path / "data-inset",
        workspace_dir=tmp_path / "workspace-inset",
        map_grid_lookup=lambda map_id: grid if map_id == 1 else None,
    )

    plain = plain_runtime.refresh(
        emulator=FakeEmulator(),
        state=state,
        navigation=navigation,
        reason="test_refresh",
        source="observe",
    )
    inset = inset_runtime.refresh(
        emulator=FakeEmulator(),
        state=state,
        navigation=navigation,
        reason="test_refresh",
        source="observe",
    )

    plain_frame = Image.open(plain["bundle"]["artifacts"]["latest_frame_annotated"]).convert("RGB")
    inset_frame = Image.open(inset["bundle"]["artifacts"]["latest_frame_annotated"]).convert("RGB")

    assert inset_frame.width > plain_frame.width
    panel = inset_frame.crop((plain_frame.width, 0, inset_frame.width, inset_frame.height))
    colours = {colour for _count, colour in panel.getcolors(maxcolors=1 << 20)}
    assert (55, 208, 255) in colours


def test_runtime_map_grid_lookup_degrades_to_todays_frame(tmp_path: Path):
    def boom(map_id: int) -> dict:
        raise RuntimeError("explored map store unavailable")

    navigation = make_navigation_payload(make_snapshot())
    state = make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10)

    frames = []
    for index, lookup in enumerate((None, boom, lambda map_id: None, lambda map_id: "nonsense")):
        runtime = AgentRuntime(
            data_dir=tmp_path / f"data-{index}",
            workspace_dir=tmp_path / f"workspace-{index}",
            map_grid_lookup=lookup,
        )
        result = runtime.refresh(
            emulator=FakeEmulator(),
            state=state,
            navigation=navigation,
            reason="test_refresh",
            source="observe",
        )
        path = Path(result["bundle"]["artifacts"]["latest_frame_annotated"])
        assert path.exists()
        frames.append(Image.open(path).convert("RGB"))

    for frame in frames[1:]:
        assert ImageChops.difference(frames[0], frame).getbbox() is None


def test_build_movement_guidance_lists_legal_moves():
    guidance = build_movement_guidance(snapshot=make_snapshot())

    assert "up, left, right" in guidance["notes"][0]
    assert guidance["preferred_direction"] == "north"


def test_refresh_writes_frames_and_a_slim_turn_context(tmp_path: Path):
    emulator = FakeEmulator()
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    result = runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
        navigation=make_navigation_payload(make_snapshot()),
        reason="test_refresh",
        source="observe",
    )
    bundle = result["bundle"]

    assert Path(bundle["artifacts"]["latest_frame"]).exists()
    assert Path(bundle["artifacts"]["latest_frame_annotated"]).exists()

    turn_context_path = Path(bundle["artifacts"]["turn_context_json"])
    payload = turn_context_path.read_bytes()
    assert len(payload) < 600
    context = json.loads(payload)
    assert set(context) == {"observation_id", "objective", "position", "ui"}
    assert set(context["objective"]) == {"id", "summary", "completion_predicate"}
    assert context["position"] == {"map_name": "Viridian City", "x": 10, "y": 10, "facing": "up"}
    assert set(context["ui"]) == {"mode", "screen_text"}
    assert context["ui"]["mode"] == "overworld"
    assert context["observation_id"].startswith("obs-")


def test_refresh_produces_one_bundle_and_no_scaffolding_artifacts(tmp_path: Path):
    emulator = FakeEmulator()
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True),
        navigation=make_navigation_payload(make_snapshot()),
        reason="test_refresh",
        source="observe",
    )

    workspace = tmp_path / "workspace"
    assert not (workspace / "debug" / "latest_observation.md").exists()
    assert not (workspace / "debug" / "working_memory.md").exists()
    assert not (workspace / "debug" / "session_brief.md").exists()
    assert not (workspace / "debug" / "knowledge_graph.json").exists()
    assert not (workspace / "debug" / "landmarks.json").exists()
    assert not (workspace / "turn_plan.json").exists()
    assert set(runtime.artifacts) == {
        "latest_frame",
        "latest_frame_annotated",
        "live_frame",
        "live_frame_annotated",
        "turn_context_json",
        "latest_observation_json",
        "current_objective_json",
        "run_log_jsonl",
    }


def test_dashboard_state_exposes_frames_objective_and_stuck(tmp_path: Path):
    emulator = FakeEmulator()
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Viridian City", map_id=1, has_pokedex=True, x=10, y=10),
        navigation=make_navigation_payload(make_snapshot()),
        reason="test_refresh",
        source="observe",
    )
    dashboard = runtime.dashboard_state()

    assert dashboard["visuals"]["ui_mode"] == "overworld"
    assert dashboard["world_state"]["map"]["map_name"] == "Viridian City"
    assert dashboard["agent_intent"]["movement_guidance"]["notes"]
    assert dashboard["memory_and_progress"]["stuck"]["level"] == "clear"


def test_agent_runtime_detects_repeated_no_movement(tmp_path: Path):
    emulator = FakeEmulator()
    navigation = make_navigation_payload(make_snapshot())
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    stuck_level = "clear"
    for _ in range(4):
        result = runtime.refresh(
            emulator=emulator,
            state=make_state(map_name="Viridian Forest", map_id=50, has_pokedex=True, x=10, y=10),
            navigation=navigation,
            reason="repeat_action",
            source="action",
            requested_actions=["walk_up"],
        )
        stuck_level = result["bundle"]["stuck"]["level"]

    assert stuck_level in {"warning", "danger"}


def test_dialog_and_battle_guidance_are_exposed(tmp_path: Path):
    emulator = FakeEmulator()
    navigation = make_navigation_payload(make_snapshot())
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    bundle = runtime.refresh(
        emulator=emulator,
        state=make_state(
            map_name="Pewter Gym",
            map_id=53,
            has_pokedex=True,
            battle=True,
            enemy_types=["Rock", "Ground"],
        ),
        navigation=navigation,
        reason="battle_test",
        source="observe",
    )["bundle"]

    assert bundle["battle_guidance"]["recommended_mode"] == "select_best_move"
    assert bundle["battle_guidance"]["recommended_move"]["name"] == "Vine Whip"

    dialog_bundle = runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Oak's Lab", map_id=40, dialog_active=True),
        navigation=navigation,
        reason="dialog_test",
        source="observe",
    )["bundle"]

    assert dialog_bundle["dialog_guidance"]["should_continue"] is True
    assert isinstance(dialog_bundle["dialog_guidance"]["transcript_recent"], list)


def test_auto_save_still_fires_on_map_transition(tmp_path: Path):
    emulator = FakeEmulator()
    navigation = make_navigation_payload(make_snapshot())
    runtime = AgentRuntime(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")

    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Pallet Town", map_id=0),
        navigation=navigation,
        reason="first",
        source="observe",
    )
    runtime.refresh(
        emulator=emulator,
        state=make_state(map_name="Route 1", map_id=12),
        navigation=navigation,
        reason="moved",
        source="action",
        requested_actions=["walk_up"],
    )

    assert emulator.saved_paths
    assert any("map-transition" in path for path in emulator.saved_paths)
