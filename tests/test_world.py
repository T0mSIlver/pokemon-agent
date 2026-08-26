"""Cross-map routing and the plan simulator.

Everything graph-shaped runs against `tests/fixtures/world_min.json`, a
hand-written Pallet -> Pewter corridor, so these tests pass whether or not the
generated `pokemon_agent/data/game/world.json` exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent.agent_cli import ActionError
from pokemon_agent.navigation import TILE_ID_OFFSET, LiveNavigationSnapshot
from pokemon_agent.world import (
    DEFAULT_WORLD_PATH,
    World,
    frontier,
    frontier_detail,
    movement_edges,
    path_within,
    simulate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "world_min.json"


@pytest.fixture
def world() -> World:
    return World.load(FIXTURE)


# ---------------------------------------------------------------------------
# Collision grids used by the simulator tests
# ---------------------------------------------------------------------------

#: rows[y][x], truthy is passable — `LiveNavigationSnapshot.terrain`'s shape.
ROOM = [
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
]

OPEN_3X3 = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]


# ---------------------------------------------------------------------------
# World graph
# ---------------------------------------------------------------------------


def test_fixture_names_are_real_map_names():
    from pokemon_agent.memory.red import MAP_NAMES

    known = set(MAP_NAMES.values())
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for name, entry in payload["maps"].items():
        assert name in known, name
        for warp in entry["warps"]:
            assert warp["to_map"] in known, warp


def test_map_names_are_sorted_and_complete(world: World):
    names = world.map_names()
    assert names == tuple(sorted(names))
    assert "Pallet Town" in names
    assert "Viridian Forest" in names
    # Route 21 is only ever a connection target in the fixture, so it is a
    # place you can route to but not a map with a record of its own.
    assert "Route 21" not in names


def test_route_pallet_to_pewter_walks_the_corridor_in_order(world: World):
    hops = world.route("Pallet Town", "Pewter City")
    assert hops is not None
    assert [hop.to_map for hop in hops] == [
        "Route 1",
        "Viridian City",
        "Route 2",
        "Pewter City",
    ]
    assert all(hop.kind == "connection" for hop in hops)
    assert all(hop.edge == "north" for hop in hops)
    assert all(hop.at is None for hop in hops)
    # Each hop starts where the previous one landed.
    assert hops[0].from_map == "Pallet Town"
    for earlier, later in zip(hops, hops[1:]):
        assert earlier.to_map == later.from_map
    assert world.distance("Pallet Town", "Pewter City") == 4


def test_route_to_self_is_empty_not_none(world: World):
    assert world.route("Pallet Town", "Pallet Town") == ()
    assert world.distance("Pallet Town", "Pallet Town") == 0


def test_unreachable_pairs_return_none(world: World):
    # Cinnabar Island is in the fixture with no connections and no warps.
    assert world.route("Pallet Town", "Cinnabar Island") is None
    assert world.distance("Pallet Town", "Cinnabar Island") is None
    assert world.route("Pallet Town", "Saffron City") is None
    assert world.route("Nowhere At All", "Pallet Town") is None


def test_route_into_a_building_and_back_out(world: World):
    inward = world.route("Pallet Town", "Red's House 2F")
    assert inward is not None
    assert [hop.kind for hop in inward] == ["warp", "warp"]
    assert inward[0].to_map == "Red's House 1F"
    assert inward[0].at == (5, 5)
    assert inward[0].edge is None
    assert inward[1].to_map == "Red's House 2F"
    assert inward[1].at == (7, 1)

    outward = world.route("Red's House 2F", "Pallet Town")
    assert outward is not None
    assert [hop.to_map for hop in outward] == ["Red's House 1F", "Pallet Town"]
    assert outward[-1].at == (2, 7)


def test_connections_are_two_way_where_the_data_says_so(world: World):
    north = world.neighbours("Pallet Town")
    assert any(hop.to_map == "Route 1" and hop.edge == "north" for hop in north)
    south = world.neighbours("Route 1")
    assert any(hop.to_map == "Pallet Town" and hop.edge == "south" for hop in south)
    assert world.distance("Pewter City", "Pallet Town") == 4


def test_neighbours_keep_hops_to_maps_with_no_record(world: World):
    hops = world.neighbours("Pallet Town")
    assert any(hop.to_map == "Route 21" for hop in hops)
    assert world.neighbours("Route 21") == ()
    assert world.distance("Pallet Town", "Route 21") == 1


def test_forest_route_prefers_the_road_over_the_gates(world: World):
    # Route 2 connects straight to Pewter City, so the forest gates must not
    # win: the shortest route is the road, not the shortcut through the trees.
    hops = world.route("Viridian City", "Pewter City")
    assert hops is not None
    assert len(hops) == 2
    forest = world.route("Route 2", "Viridian Forest")
    assert forest is not None
    assert [hop.to_map for hop in forest] in (
        ["Viridian Forest South Gate", "Viridian Forest"],
        ["Viridian Forest North Gate", "Viridian Forest"],
    )


def test_missing_world_file_loads_empty_and_routes_none(tmp_path: Path):
    world = World.load(tmp_path / "does_not_exist.json")
    assert world.map_names() == ()
    assert world.neighbours("Pallet Town") == ()
    assert world.route("Pallet Town", "Pewter City") is None
    assert world.distance("Pallet Town", "Pewter City") is None


def test_unreadable_world_file_loads_empty(tmp_path: Path):
    broken = tmp_path / "world.json"
    broken.write_text("{not json at all", encoding="utf-8")
    assert World.load(broken).map_names() == ()

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"maps": []}), encoding="utf-8")
    assert World.load(wrong_shape).map_names() == ()


def test_default_load_works_with_or_without_the_generated_file():
    world = World.load()
    if DEFAULT_WORLD_PATH.exists():
        assert "Pallet Town" in world.map_names()
    else:
        assert world.map_names() == ()


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


def test_north_is_up_walk_up_decreases_y():
    result = simulate(["walk_up"], OPEN_3X3, (1, 1), "down")
    assert result.end_pos == (1, 0)
    assert result.end_facing == "up"
    assert result.blocked_at is None

    assert simulate(["down"], OPEN_3X3, (1, 1), "up").end_pos == (1, 2)
    assert simulate(["left"], OPEN_3X3, (1, 1), "up").end_pos == (0, 1)
    assert simulate(["right"], OPEN_3X3, (1, 1), "up").end_pos == (2, 1)


def test_clean_plan_reports_no_block_and_the_right_end_position():
    result = simulate(["up:2", "right"], ROOM, (0, 4), "down")
    assert result.blocked_at is None
    assert result.blocked_by is None
    assert result.warp_at is None
    assert result.end_pos == (1, 2)
    assert result.end_facing == "right"
    assert result.steps_taken == 3
    assert result.trace == ((0, 4), (0, 3), (0, 2), (1, 2))
    assert result.ok


def test_button_presses_cost_a_step_but_move_nothing():
    result = simulate(["a", "up", "b"], ROOM, (0, 4), "down")
    assert result.end_pos == (0, 3)
    assert result.steps_taken == 3
    assert result.blocked_at is None


def test_plan_into_a_wall_reports_the_exact_index():
    result = simulate(["right", "right", "right"], ROOM, (0, 2), "up")
    assert result.blocked_at == 2
    assert result.blocked_by == "wall"
    assert result.end_pos == (2, 2)
    # Pressing into a wall turns the player, exactly as the game does.
    assert result.end_facing == "right"
    assert result.steps_taken == 2
    assert result.trace == ((0, 2), (1, 2), (2, 2))


def test_blocked_index_counts_expanded_repeats():
    result = simulate(["right:4"], ROOM, (0, 2), "up")
    assert result.blocked_at == 2
    assert result.blocked_by == "wall"


def test_walking_off_the_grid_is_an_edge_block():
    result = simulate(["left"], ROOM, (0, 0), "down")
    assert result.blocked_at == 0
    assert result.blocked_by == "edge"
    assert result.end_pos == (0, 0)
    assert result.trace == ((0, 0),)


def test_sprites_block_as_npcs_from_a_live_snapshot():
    snapshot = LiveNavigationSnapshot(
        map_id=1,
        map_name="Viridian City",
        player_position=(12, 12),
        facing="down",
        tileset="OVERWORLD",
        window_top_left=(10, 10),
        terrain=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        sprite_positions=[(12, 11)],
    )
    blocked = simulate(["up"], snapshot, (12, 12), "down")
    assert blocked.blocked_at == 0
    assert blocked.blocked_by == "npc"
    # The window offset is honoured: the same plan sideways walks fine.
    assert simulate(["left"], snapshot, (12, 12), "down").end_pos == (11, 12)


def test_explored_map_grid_shape_is_accepted():
    collision = {
        "width": 3,
        "height": 3,
        "walkable": {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)},
        "sprites": [{"x": 2, "y": 0}],
    }
    assert simulate(["right"], collision, (0, 0), "down").end_pos == (1, 0)
    assert simulate(["right:2"], collision, (0, 0), "down").blocked_by == "npc"
    assert simulate(["down:2", "right"], collision, (0, 0), "down").blocked_at == 2


def test_stepping_onto_a_warp_tile_reports_warp_at():
    result = simulate(["up:3"], ROOM, (0, 4), "down", warps=[(0, 2)])
    assert result.warp_at == 1
    assert result.blocked_at is None
    assert result.end_pos == (0, 2)
    assert result.steps_taken == 2
    # Simulation stops on the warp: past it the player is on another map.
    assert result.trace == ((0, 4), (0, 3), (0, 2))


def test_empty_plan_is_a_no_op_and_bad_tokens_raise():
    result = simulate([], ROOM, (2, 2), "left")
    assert result.end_pos == (2, 2)
    assert result.end_facing == "left"
    assert result.steps_taken == 0
    assert result.trace == ((2, 2),)
    with pytest.raises(ActionError):
        simulate(["moonwalk"], ROOM, (2, 2), "left")


# ---------------------------------------------------------------------------
# path_within
# ---------------------------------------------------------------------------


DETOUR = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]


def test_path_within_goes_around_an_obstacle():
    actions = path_within(DETOUR, (1, 0), (1, 2))
    assert actions is not None
    assert len(actions) == 4  # straight down is walled; around the left end
    assert all(action.startswith("walk_") for action in actions)
    walked = simulate(list(actions), DETOUR, (1, 0), "down")
    assert walked.blocked_at is None
    assert walked.end_pos == (1, 2)


def test_path_within_is_empty_when_already_there():
    assert path_within(DETOUR, (0, 0), (0, 0)) == ()


def test_path_within_returns_none_when_walled_off():
    split = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert path_within(split, (0, 0), (2, 0)) is None
    # A blocked or off-grid target is None, never a path that ends in a wall.
    assert path_within(split, (0, 0), (1, 1)) is None
    assert path_within(split, (0, 0), (9, 9)) is None


def test_path_within_leaves_a_tile_the_grid_calls_blocked():
    # The player can stand on a doorway the collision grid reads as solid;
    # pathing must still walk them off it.
    doorway = [
        [1, 1],
        [0, 1],
    ]
    assert path_within(doorway, (0, 1), (1, 1)) == ("walk_right",)


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------


MAZE = [
    [1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1],
]


def test_frontier_is_nearest_first_and_excludes_seen():
    seen = {(0, 0), (1, 0), (0, 1), (0, 2)}
    tiles = frontier(MAZE, seen, (0, 0))
    assert set(tiles) == {(2, 0), (2, 1), (1, 2), (2, 2)}
    assert tiles[0] == (2, 0)  # two steps away, the closest unseen tile
    assert tiles[-1] == (2, 2)  # four steps away, the furthest
    assert not set(tiles) & seen
    # The x=4 column is walkable but sealed off by the x=3 wall.
    assert not any(x == 4 for x, _ in tiles)


def test_frontier_orders_by_walking_distance_not_straight_line():
    tiles = frontier(MAZE, set(), (0, 0))
    steps = {tile: index for index, tile in enumerate(tiles)}
    # (1, 1) is a wall, so (1, 2) is a four-step walk while (2, 0) is two.
    assert steps[(2, 0)] < steps[(1, 2)]
    assert tiles[0] == (0, 0)


def test_frontier_is_empty_when_everything_is_seen():
    everything = {(x, y) for y in range(3) for x in range(5)}
    assert frontier(MAZE, everything, (0, 0)) == ()


def test_frontier_from_a_live_snapshot_window():
    snapshot = LiveNavigationSnapshot(
        map_id=51,
        map_name="Viridian Forest",
        player_position=(11, 11),
        facing="down",
        tileset="FOREST",
        window_top_left=(10, 10),
        terrain=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        sprite_positions=[(12, 12)],
    )
    tiles = frontier(snapshot, {(11, 11), (11, 10)}, (11, 11))
    assert (11, 11) not in tiles
    assert (11, 10) not in tiles
    assert (12, 12) not in tiles  # an NPC is standing there
    assert tiles[0] in {(10, 11), (12, 11), (11, 12)}
    assert len(tiles) == 6


# ---------------------------------------------------------------------------
# Ledges
# ---------------------------------------------------------------------------

#: pokered `data/tilesets/ledge_tiles.asm`, through the 0x8800 window PyBoy
#: reports background tiles in: stand on 0x2C, press down into 0x37, jump.
LEDGE_TOP = 0x2C + TILE_ID_OFFSET
LEDGE_FACE = 0x37 + TILE_ID_OFFSET
PLAIN = 0x01 + TILE_ID_OFFSET


def ledge_window(player=(10, 10)) -> dict:
    """A 3x5 window at (9, 9) whose middle row is a ledge you can only fall down.

    Rows 9 and 10 are the shelf, row 11 is the ledge face — blocked collision,
    exactly as the game reports it — and rows 12 and 13 are the ground below.
    """
    terrain = [
        [1, 1, 1],  # y = 9
        [1, 1, 1],  # y = 10, the shelf
        [0, 0, 0],  # y = 11, the ledge itself
        [1, 1, 1],  # y = 12, where a jump lands
        [1, 1, 1],  # y = 13
    ]
    tile_ids = {}
    for local_y in range(5):
        for local_x in range(3):
            coord = (9 + local_x, 9 + local_y)
            tile_ids[coord] = {10: LEDGE_TOP, 11: LEDGE_FACE}.get(coord[1], PLAIN)
    return {
        "terrain": terrain,
        "window_top_left": (9, 9),
        "tileset": "OVERWORLD",
        "tile_ids": tile_ids,
        "player_position": player,
    }


def test_a_ledge_is_a_jump_of_two_tiles_and_not_a_wall():
    result = simulate(["down"], ledge_window(), (10, 10), "up")

    assert result.blocked_at is None
    assert result.end_pos == (10, 12)  # two tiles for one press
    assert result.trace == ((10, 10), (10, 12))  # never rests on the ledge
    assert [(hop.index, hop.direction, hop.start, hop.landing) for hop in result.hops] == [
        (0, "down", (10, 10), (10, 12))
    ]
    assert "one way" in result.hops[0].describe()


def test_a_plan_keeps_walking_after_it_jumps():
    result = simulate(["down:3"], ledge_window(), (10, 10), "up")

    assert result.end_pos == (10, 13)
    assert result.steps_taken == 2
    assert result.blocked_at == 2  # the window runs out below y = 13
    assert result.blocked_by == "edge"
    assert len(result.hops) == 1


def test_a_ledge_cannot_be_climbed_back_up():
    result = simulate(["up"], ledge_window(player=(10, 12)), (10, 12), "down")

    assert result.hops == ()
    assert result.blocked_at == 0
    assert result.blocked_by == "wall"
    assert result.end_pos == (10, 12)


def test_the_snapshots_own_ledge_hops_are_enough_without_tile_ids():
    """The HTTP snapshot carries `ledge_hops` and no tile ids, and still jumps."""
    over_the_wire = {
        "terrain": ledge_window()["terrain"],
        "window_top_left": (9, 9),
        "player_position": {"x": 10, "y": 10},
        "ledge_hops": {"down": {"x": 10, "y": 12}},
    }

    result = simulate(["down"], over_the_wire, (10, 10), "up")

    assert result.end_pos == (10, 12)
    assert len(result.hops) == 1


def test_pathing_goes_down_a_ledge_but_never_up_one():
    window = ledge_window()

    assert path_within(window, (10, 10), (10, 13)) == ("walk_down", "walk_down")
    # The only way back is around, and this window has no way around.
    assert path_within(window, (10, 12), (10, 10)) is None
    assert path_within(window, (10, 12), (10, 11)) is None


def test_frontier_leaves_through_a_ledge_and_cannot_come_back_in():
    below = {(10, 12), (10, 13), (9, 12), (9, 13), (11, 12), (11, 13)}

    from_above = set(frontier(ledge_window(), {(10, 10)}, (10, 10)))
    assert below <= from_above  # the ground below is reachable, by jumping

    from_below = set(frontier(ledge_window(player=(10, 12)), {(10, 12)}, (10, 12)))
    assert not any(y <= 11 for _, y in from_below)  # nothing above it is


# ---------------------------------------------------------------------------
# Live window versus remembered map
# ---------------------------------------------------------------------------


def merged(walkable, live, **extra) -> dict:
    grid = {"width": 20, "height": 20, "walkable": set(walkable), "live": set(live)}
    grid.update(extra)
    return grid


def test_a_step_out_of_the_live_window_is_flagged_as_memory():
    row = {(x, 5) for x in range(10)}
    collision = merged(row, {(x, 5) for x in range(4)})

    result = simulate(["right:5"], collision, (0, 5), "right")

    assert result.end_pos == (5, 5)
    assert result.blocked_at is None
    assert result.unverified_from == 3  # the step into (4, 5), which nobody saw
    assert result.certain is False
    # Everything inside the window is a fact and says so.
    assert simulate(["right:2"], collision, (0, 5), "right").certain is True


def test_frontier_says_which_tiles_the_window_vouches_for():
    row = {(x, 5) for x in range(8)}
    collision = merged(row, {(x, 5) for x in range(4)})

    detail = frontier_detail(collision, {(0, 5)}, (0, 5))
    certain = {tile.coord for tile in detail if tile.certain}
    believed = {tile.coord for tile in detail if not tile.certain}

    assert certain == {(1, 5), (2, 5), (3, 5)}
    assert believed == {(4, 5), (5, 5), (6, 5), (7, 5)}
    assert frontier(collision, {(0, 5)}, (0, 5)) == tuple(tile.coord for tile in detail)


def test_a_tile_reached_only_across_memory_is_never_called_a_fact():
    """One remembered tile in the middle makes everything past it a belief."""
    row = {(x, 5) for x in range(6)}
    # The window shows both ends and skips (2, 5) in between.
    collision = merged(row, row - {(2, 5)})

    detail = {tile.coord: tile.certain for tile in frontier_detail(collision, {(0, 5)}, (0, 5))}

    assert detail[(1, 5)] is True
    assert detail[(2, 5)] is False
    assert detail[(3, 5)] is False  # in the window, but only reachable through memory


# ---------------------------------------------------------------------------
# Cave seams and warp exits
# ---------------------------------------------------------------------------

#: pokered `data/tilesets/tile_pair_collisions.asm`: in CAVERN, 0x20 and 0x05
#: are both passable and you cannot step from one to the other.
CAVE_SHELF = 0x20 + TILE_ID_OFFSET
CAVE_FLOOR = 0x05 + TILE_ID_OFFSET


def cave_window(player=(10, 10), tileset: str = "CAVERN") -> dict:
    """A 3x4 window at (9, 9): a shelf over a floor, with a seam between them.

    Every tile is passable collision — the collision map has no idea there are
    two floors here — so nothing but the tile pair says the boundary between
    y = 10 and y = 11 cannot be crossed. This is Mt. Moon's shape, and walking
    straight through it is what `sim` did.
    """
    tile_ids = {
        (9 + local_x, 9 + local_y): CAVE_SHELF if local_y < 2 else CAVE_FLOOR
        for local_y in range(4)
        for local_x in range(3)
    }
    return {
        "terrain": [[1, 1, 1] for _ in range(4)],
        "window_top_left": (9, 9),
        "tileset": tileset,
        "tile_ids": tile_ids,
        "player_position": player,
    }


def test_a_cave_seam_blocks_although_both_of_its_tiles_are_passable():
    result = simulate(["down"], cave_window(), (10, 10), "down")

    assert result.blocked_at == 0
    # Not "wall": the tile below is open ground, and calling it a wall is what
    # sent the agent looking for a way round something it could already see.
    assert result.blocked_by == "tile_pair"
    assert result.end_pos == (10, 10)


def test_a_cave_seam_blocks_from_both_sides():
    """`CheckForTilePairCollisions` tries the pair both ways round."""
    result = simulate(["up"], cave_window(player=(10, 11)), (10, 11), "up")

    assert (result.blocked_at, result.blocked_by) == (0, "tile_pair")


def test_walking_along_one_cave_floor_is_untouched_by_the_seam():
    result = simulate(["right", "left"], cave_window(), (10, 10), "right")

    assert result.blocked_at is None
    assert result.end_pos == (10, 10)


def test_the_same_two_tiles_are_not_a_seam_outside_a_cave():
    """The table has CAVERN and FOREST rows only, and nothing else may fire."""
    result = simulate(["down"], cave_window(tileset="OVERWORLD"), (10, 10), "down")

    assert result.blocked_at is None
    assert result.end_pos == (10, 11)


def test_the_snapshots_own_blocked_pairs_are_enough_without_tile_ids():
    """The HTTP snapshot carries `blocked_pairs` and no tile ids."""
    over_the_wire = {
        "terrain": [[1, 1, 1] for _ in range(4)],
        "window_top_left": (9, 9),
        "player_position": {"x": 10, "y": 10},
        "blocked_pairs": [{"a": {"x": 10, "y": 10}, "b": {"x": 10, "y": 11}}],
    }

    result = simulate(["down"], over_the_wire, (10, 10), "down")

    assert (result.blocked_at, result.blocked_by) == (0, "tile_pair")
    # ... and only that one seam: the tile beside it is still open.
    assert simulate(["right"], over_the_wire, (10, 10), "right").blocked_at is None


def test_pathing_and_frontier_never_cross_a_cave_seam():
    window = cave_window()

    assert path_within(window, (10, 10), (10, 12)) is None
    reachable = set(frontier(window, {(10, 10)}, (10, 10)))
    assert reachable == {(9, 9), (9, 10), (10, 9), (11, 9), (11, 10)}


def warp_exit_window(*, armed: bool) -> dict:
    """Standing on a cave ladder at the bottom edge of the map.

    Collision calls the tile below blocked, because it is off the map. Pressing
    down does not walk into it; it takes the ladder.
    """
    return {
        "terrain": [[1, 1, 1], [0, 0, 0]],
        "window_top_left": (9, 10),
        "player_position": {"x": 10, "y": 10},
        "warp_exit_directions": ["down"],
        "warp_exit_armed": armed,
    }


def test_an_armed_warp_exit_is_a_warp_and_not_a_wall():
    result = simulate(["down"], warp_exit_window(armed=True), (10, 10), "down")

    assert result.warp_at == 0
    assert result.blocked_at is None and result.blocked_by is None
    assert result.steps_taken == 1
    # The plan stops here: past a warp the player is on a map this call has no
    # collision for.
    assert result.end_pos == (10, 10)


def test_an_unarmed_warp_exit_is_still_a_wall():
    """The engine arms a warp when the player walks onto it, not before."""
    result = simulate(["down"], warp_exit_window(armed=False), (10, 10), "down")

    assert result.warp_at is None
    assert (result.blocked_at, result.blocked_by) == (0, "wall")


def test_a_warp_exit_answers_only_the_direction_that_fires_it():
    window = warp_exit_window(armed=True)
    window["terrain"] = [[0, 1, 0], [0, 0, 0]]

    assert simulate(["left"], window, (10, 10), "left").warp_at is None
    assert simulate(["left"], window, (10, 10), "left").blocked_at == 0
    assert simulate(["down"], window, (10, 10), "down").warp_at == 0


def test_a_seam_survives_being_merged_with_the_remembered_map():
    """The shape `/sim` really runs on: window plus store, seams carried along.

    `capabilities.collision_from` folds the window into the explored map and
    hands on what `movement_edges` built, so this is the one path that has to
    keep working — the object the emulator produced is long gone by then.
    """
    snapshot = {
        "terrain": [[1, 1, 1] for _ in range(4)],
        "window_top_left": (9, 9),
        "player_position": {"x": 10, "y": 10},
        # The whole row, as Mt. Moon has it: two floors, and no way between.
        "blocked_pairs": [{"a": {"x": x, "y": 10}, "b": {"x": x, "y": 11}} for x in range(9, 12)],
    }
    remembered = {(x, y) for y in range(9, 13) for x in range(9, 12)}
    merged_map = {
        "width": 20,
        "height": 20,
        "walkable": remembered,
        "live": remembered,
        "ledges": movement_edges(snapshot),
    }

    assert simulate(["down"], merged_map, (10, 10), "down").blocked_by == "tile_pair"
    # A store that remembers walking every tile cannot make two floors one.
    assert path_within(merged_map, (10, 10), (10, 11)) is None
    reachable = set(frontier(merged_map, {(10, 10)}, (10, 10)))
    assert reachable == {(9, 9), (10, 9), (11, 9), (9, 10), (11, 10)}


# ---------------------------------------------------------------------------
# Warps are absorbing, and refused ground stays refused
# ---------------------------------------------------------------------------


#: A corridor with a ladder in the middle of it. Both ends are ordinary ground.
LADDER_CORRIDOR = {
    "width": 7,
    "height": 1,
    "walkable": {(x, 0) for x in range(7)},
    "live": {(x, 0) for x in range(7)},
    "warps": [{"x": 3, "y": 0}],
}


def test_a_route_never_crosses_a_warp_it_was_not_aiming_at():
    """A warp is an edge in with no edge onward, so no plan goes through one.

    Measured on Mt. Moon 1F: the shortest walk from the south entrance (14, 35)
    to the ladder at (5, 5) is 89 steps and crosses the *other* ladder, at
    (17, 11), on step 72. Walked, it spent 72 presses and ended on B1F, and
    every /goto after that asked for (5, 5) on a floor with no such tile — a
    refusal per call, forever, from the wrong map.
    """
    assert path_within(LADDER_CORRIDOR, (0, 0), (6, 0)) is None


def test_the_warp_itself_is_still_somewhere_you_can_walk_to():
    """Absorbing is not forbidden. A ladder is usually exactly where you meant to go."""
    assert path_within(LADDER_CORRIDOR, (0, 0), (3, 0)) == ("walk_right",) * 3


def test_standing_on_a_warp_does_not_strand_you_on_it():
    """The player is often on one: a save loaded in a doorway, a cave mouth."""
    assert path_within(LADDER_CORRIDOR, (3, 0), (0, 0)) == ("walk_left",) * 3


def test_a_refused_tile_is_not_in_the_next_flood():
    """Ground the game would not allow is held out, so the same plan cannot return.

    The planner has no memory between rounds — it recomputes from scratch — so
    without this it rebuilds the identical shortest path, walks into the
    identical NPC and reports the identical refusal. Measured on Mt. Moon 1F at
    40 presses a round, for as many rounds as anything kept asking.
    """
    room = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    assert path_within(room, (0, 1), (2, 1)) == ("walk_up", "walk_right", "walk_right", "walk_down")
    # (1, 0) is where the trainer is standing. Round the other way, then.
    around = path_within(room, (0, 1), (2, 1), refused={(1, 0)})
    assert around == ("walk_down", "walk_right", "walk_right", "walk_up")
    # Both ways out refused is a refusal, not a worse path.
    assert path_within(room, (0, 1), (2, 1), refused={(1, 0), (1, 2)}) is None


def test_refusing_the_tile_you_stand_on_does_not_wall_you_in():
    """A refusal is about somewhere you tried to go, never about where you are."""
    assert path_within([[1, 1, 1]], (0, 0), (2, 0), refused={(0, 0)}) == (
        "walk_right",
        "walk_right",
    )
