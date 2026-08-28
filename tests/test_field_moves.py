"""A small tree is not a wall, and every answer here used to say it was.

Vermilion City is the case that paid for this file. Its gym door sits in a
42-tile pocket, sealed from the 363 tiles the player can reach by exactly one
cuttable tree at (15,18). The blockset calls that tile solid, correctly — it is
solid until Cut is used on it — and `/goto` turned that into "Every tile
bordering the ground you can reach has been looked at and is solid, so this is a
wall and not a gap in the map", with the epistemic framing the rest of the
harness reserves for facts.

One run read that sentence 43 times. It stood in Vermilion City for 4,480 tool
calls, never once opened the gym door seven steps away, and finished 43 hours in
with two badges. Its own notes had guessed right — "gym door tree-blocked? Use
Cut" — and the map answer talked it out of that.

The fixture is Vermilion's decoded floor read off the cartridge, so these run
without the ROM. The live half is in `test_field_moves_live.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent import capabilities, world
from pokemon_agent.navigation import CUT_TREE_TILES

FIXTURES = Path(__file__).parent / "fixtures"

#: The one tree that seals the gym off, and the tile Cut is used from.
TREE = (15, 18)
STAND = (15, 17)
GYM_DOOR = (12, 19)


@pytest.fixture(scope="module")
def vermilion() -> dict:
    payload = json.loads((FIXTURES / "vermilion_city_floor.json").read_text(encoding="utf-8"))
    walkable = {(x, y) for x, y in payload["walkable"]}
    tile_ids = {}
    for key, value in payload["tile_ids"].items():
        x, y = key.split(",")
        tile_ids[(int(x), int(y))] = value
    return {
        "collision": {
            "width": payload["width"],
            "height": payload["height"],
            "walkable": walkable,
            "sprites": [],
            "warps": [GYM_DOOR],
            "live": set(),
            # The whole decoded floor, solid tiles included — which is what
            # `collision_from` builds from ground truth, and what makes the
            # region *sealed* rather than merely unexplored.
            "seen": set(tile_ids),
            "ledges": {},
            "ground_truth": True,
            "tile_ids": tile_ids,
            "tileset": payload["tileset"],
        },
        "player": tuple(payload["player"]),
    }


@pytest.fixture(scope="module")
def region(vermilion) -> world.Region:
    return world.reachable_region(vermilion["collision"], vermilion["player"])


def test_the_gym_really_is_sealed_off(vermilion, region):
    """The premise. Without this the rest of the file is testing nothing."""
    assert GYM_DOOR not in region
    assert len(region.order) == 363
    assert region.sealed, "no unseen ground borders this pocket, so it is not an unlooked-at gap"


def test_the_sealing_tile_is_a_tree_and_the_blockset_calls_it_solid(vermilion):
    collision = vermilion["collision"]
    assert TREE not in collision["walkable"]
    assert collision["tile_ids"][TREE] in CUT_TREE_TILES[collision["tileset"]]


def test_the_seam_names_the_tree(vermilion, region):
    assert capabilities.cut_trees_on_the_seam(vermilion["collision"], region) == [TREE]


def test_a_tree_behind_a_wall_is_not_on_the_seam(vermilion, region):
    """Only trees bordering ground we can reach. The rest are somebody else's."""
    collision = dict(vermilion["collision"])
    ids = dict(collision["tile_ids"])
    tree = next(iter(CUT_TREE_TILES[collision["tileset"]]))
    # (0,0) is a corner of the map the player cannot reach.
    ids[(0, 0)] = tree
    collision["tile_ids"] = ids
    assert (0, 0) not in capabilities.cut_trees_on_the_seam(collision, region)


def test_an_unknown_tileset_claims_no_trees(vermilion, region):
    """A tile id means nothing without the tileset it was read in.

    0x3D is a cuttable tree on the overworld and a floor tile in a gate house;
    the corpus sweep that found this file's cases turned up seven false trees in
    Route 11's gate before the tileset was checked.
    """
    collision = dict(vermilion["collision"], tileset="GATE")
    assert capabilities.cut_trees_on_the_seam(collision, region) == []


def test_cut_plan_stands_next_to_the_tree_and_faces_it(vermilion, region):
    plan = capabilities.cut_plan(vermilion["collision"], region)
    assert plan["tree"] == TREE
    assert plan["stand"] == STAND
    assert plan["facing"] == "down"
    assert plan["steps"] == region.steps_to(STAND)


def test_cut_plan_refuses_a_tile_that_is_not_a_tree(vermilion, region):
    with pytest.raises(capabilities.CapabilityError, match="not a small tree"):
        capabilities.cut_plan(vermilion["collision"], region, at=(0, 0))


def test_cutting_the_tree_joins_the_two_pockets(vermilion):
    """What the cut is worth, measured rather than assumed."""
    collision = dict(vermilion["collision"])
    collision["walkable"] = collision["walkable"] | {TREE}
    after = world.reachable_region(collision, vermilion["player"])
    assert GYM_DOOR in after
    assert len(after.order) == 406, "43 more tiles, and the gym door among them"


def test_the_refusal_names_the_tree_instead_of_claiming_a_wall(vermilion, region):
    stop = capabilities._walled_off(
        world.World({}),
        "the warp to Vermilion Gym at [12, 19]",
        1,
        {"map_name": "Vermilion City", "snapshot": {}},
        vermilion["collision"],
        region,
        lead="not reachable on foot;",
    )
    assert "small tree at [15, 18]" in stop.reason
    assert "poke cut" in stop.reason
    assert "this is a wall" not in stop.reason
    assert stop.onward["kind"] == "behind-a-tree"
    assert stop.onward["cut_trees"] == [[15, 18]]


def test_a_real_wall_is_still_called_a_wall(vermilion):
    """The correction must not turn every refusal into a maybe."""
    collision = dict(vermilion["collision"], tile_ids={})
    region = world.reachable_region(collision, vermilion["player"])
    stop = capabilities._walled_off(
        world.World({}),
        "the warp to Vermilion Gym at [12, 19]",
        1,
        {"map_name": "Vermilion City", "snapshot": {}},
        collision,
        region,
        lead="not reachable on foot;",
    )
    assert "this is a wall and not a gap in the map" in stop.reason
    assert stop.onward["kind"] == "walled-off"


# ---------------------------------------------------------------------------
# Which row of the party submenu holds the move
# ---------------------------------------------------------------------------


def _moves(*names):
    return [{"name": name} for name in names]


def test_field_move_row_counts_only_field_moves():
    assert capabilities.field_move_row(_moves("Cut", "Slash", "Ember", "Leer"), "cut") == 0
    assert capabilities.field_move_row(_moves("Slash", "Cut"), "cut") == 0
    assert capabilities.field_move_row(_moves("Surf", "Cut"), "cut") == 1
    assert capabilities.field_move_row(_moves("Fly", "Surf", "Cut"), "cut") == 2


def test_field_move_row_is_none_when_the_move_is_not_known():
    assert capabilities.field_move_row(_moves("Slash", "Ember"), "cut") is None
    assert capabilities.field_move_row([], "cut") is None


# ---------------------------------------------------------------------------
# What a field move costs, and where that cost is written down
# ---------------------------------------------------------------------------


def test_the_cursor_walker_reports_what_it_spent():
    """Menu presses are buttons. The first version of `poke cut` billed none.

    The walk to the tree is billed by `_run_actions`; the four menu screens
    after it were driven straight through the emulator and appeared on no
    receipt at all — the first four live cuts recorded 20, 4, 40 and 5 presses
    and every one of them was the walk only.
    """
    from pokemon_agent import server

    rows = iter([4, 3, 2, 1])
    pressed: list[str] = []
    original_row, original_press = server._menu_row_sync, server._execute_action_sync
    server._menu_row_sync = lambda: next(rows)
    server._execute_action_sync = lambda action: pressed.append(action)
    try:
        spent = server._walk_cursor_to_sync(1, what="POKEMON")
    finally:
        server._menu_row_sync, server._execute_action_sync = original_row, original_press

    assert pressed == ["press_up"] * 3
    assert spent == 3, "three presses, three billed"


def test_a_cursor_already_on_the_row_costs_nothing():
    from pokemon_agent import server

    original_row, original_press = server._menu_row_sync, server._execute_action_sync
    server._menu_row_sync = lambda: 1
    server._execute_action_sync = lambda action: pytest.fail("pressed a button for nothing")
    try:
        assert server._walk_cursor_to_sync(1, what="POKEMON") == 0
    finally:
        server._menu_row_sync, server._execute_action_sync = original_row, original_press
