"""The persistent explored-map layer: what it remembers, and how it draws it."""

import json

import pytest
from PIL import Image

from pokemon_agent import explored_map
from pokemon_agent.explored_map import ExploredMaps
from pokemon_agent.navigation import LiveNavigationSnapshot

FOREST_ID = 51
FOREST_NAME = "VIRIDIAN FOREST"


def make_snapshot(**overrides):
    """A `LiveNavigationSnapshot.to_dict()`-shaped payload with sensible defaults."""
    player = overrides.pop("player", (5, 5))
    terrain = overrides.pop("terrain", [[1] * 10 for _ in range(9)])
    top_left = overrides.pop("top_left", (player[0] - 4, player[1] - 4))
    dimensions = overrides.pop("dimensions", {"width": 20, "height": 18})
    snapshot = {
        "map_id": overrides.pop("map_id", FOREST_ID),
        "map_name": overrides.pop("map_name", FOREST_NAME),
        "player_position": {"x": player[0], "y": player[1]},
        "window_top_left": {"x": top_left[0], "y": top_left[1]},
        "window_size": {
            "width": len(terrain[0]) if terrain else 0,
            "height": len(terrain),
        },
        "terrain": terrain,
        "sprites": [{"x": x, "y": y} for x, y in overrides.pop("sprites", [])],
        "warps": overrides.pop("warps", []),
        "map_dimensions": dimensions,
    }
    snapshot.update(overrides)
    return snapshot


def store(tmp_path, name="explored_maps.json"):
    return ExploredMaps(tmp_path / name)


def terrain_for(top_left, walls, width=10, height=9):
    """A window of open ground with `walls` (absolute coords) punched out of it."""
    return [
        [0 if (top_left[0] + lx, top_left[1] + ly) in walls else 1 for lx in range(width)]
        for ly in range(height)
    ]


#: Big enough that a tile centre is nowhere near the chrome or the player ring.
TILE_PX = 8

MAP_LEFT_PX = explored_map._LEFT_MARGIN_PX
MAP_TOP_PX = explored_map._TITLE_BAND_PX + explored_map._TICK_BAND_PX


def tile_color(image, x, y, tile=TILE_PX):
    """The colour of tile (x, y), sampled at the middle of its block."""
    return image.getpixel((MAP_LEFT_PX + int((x + 0.5) * tile), MAP_TOP_PX + int((y + 0.5) * tile)))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_a_snapshot_marks_the_window_seen_and_the_player_tile_walked(tmp_path):
    maps = store(tmp_path)

    maps.record(make_snapshot(player=(5, 5)))

    assert maps.visited(FOREST_ID) == {(5, 5)}
    coverage = maps.coverage(FOREST_ID)
    assert coverage == {
        "seen": 90,  # the whole 10x9 window at (1, 1)..(10, 9)
        "walkable_seen": 90,
        "walked": 1,
        "total": 360,
        "percent": 25.0,
    }


def test_a_second_snapshot_extends_coverage_without_losing_the_first(tmp_path):
    maps = store(tmp_path)

    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(player=(14, 5)))

    assert maps.visited(FOREST_ID) == {(5, 5), (14, 5)}
    # 90 tiles from the first window, plus the 81 columns 11..19 the second adds.
    assert maps.coverage(FOREST_ID)["seen"] == 171
    assert maps.player_position(FOREST_ID) == (14, 5)


def test_walls_and_passable_tiles_are_distinguished(tmp_path):
    maps = store(tmp_path)
    terrain = [[1] * 10 for _ in range(9)]
    terrain[0] = [0] * 10  # a solid row of trees along the top of the window

    maps.record(make_snapshot(player=(5, 5), terrain=terrain))

    coverage = maps.coverage(FOREST_ID)
    assert coverage["seen"] == 90
    assert coverage["walkable_seen"] == 80
    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)
    assert {tile_color(image, x, 1) for x in range(1, 11)} == {explored_map.COLOR_WALL}


def test_a_tile_seen_passable_once_is_never_relearned_as_a_wall(tmp_path):
    """A one-frame blocker must not leave a phantom wall behind."""
    maps = store(tmp_path)
    blocked = [[1] * 10 for _ in range(9)]
    blocked[0][0] = 0

    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(player=(5, 5), terrain=blocked))

    assert maps.coverage(FOREST_ID)["walkable_seen"] == 90


def test_a_tile_under_a_sprite_is_not_learned_at_all(tmp_path):
    """An NPC makes its tile read as blocked; that is the NPC, not the map."""
    maps = store(tmp_path)
    terrain = [[1] * 10 for _ in range(9)]
    terrain[0][0] = 0

    maps.record(make_snapshot(player=(5, 5), terrain=terrain, sprites=[(1, 1)]))

    assert maps.coverage(FOREST_ID)["seen"] == 89
    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)
    assert tile_color(image, 1, 1) == explored_map.COLOR_UNKNOWN


def test_a_real_navigation_snapshot_is_ingested(tmp_path):
    """Guards against key drift between navigation.py and this store."""
    maps = store(tmp_path)
    snapshot = LiveNavigationSnapshot(
        map_id=FOREST_ID,
        map_name=FOREST_NAME,
        player_position=(5, 5),
        facing="down",
        tileset="FOREST",
        window_top_left=(1, 1),
        terrain=[[1] * 10 for _ in range(9)],
        warps=[{"x": 3, "y": 3, "warp_id": 0, "target_map_id": 13}],
        map_dimensions={"width": 34, "height": 48},
    )

    maps.record(snapshot.to_dict())

    assert maps.visited(FOREST_ID) == {(5, 5)}
    assert maps.coverage(FOREST_ID)["total"] == 34 * 48
    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)
    assert tile_color(image, 3, 3) == explored_map.COLOR_WARP


def test_snapshots_without_a_map_id_are_ignored(tmp_path):
    maps = store(tmp_path)

    maps.record({})
    maps.record({"map_id": None})

    assert maps.map_ids() == []


def test_each_map_id_is_remembered_separately(tmp_path):
    maps = store(tmp_path)

    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(map_id=1, map_name="VIRIDIAN CITY", player=(9, 9)))

    assert maps.map_ids() == [1, FOREST_ID]
    assert maps.visited(FOREST_ID) == {(5, 5)}
    assert maps.visited(1) == {(9, 9)}
    assert maps.current_map_id == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_map_image_is_one_block_per_tile_plus_chrome(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))

    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)

    assert image.size == (
        MAP_LEFT_PX + (20 * TILE_PX) + explored_map._EDGE_PAD_PX,
        MAP_TOP_PX + (18 * TILE_PX) + explored_map._EDGE_PAD_PX,
    )


@pytest.mark.parametrize(("width", "height"), [(20, 18), (34, 48), (90, 90)])
def test_auto_sizing_keeps_the_long_edge_readable(tmp_path, width, height):
    """34x48 is Viridian Forest; it must fit on screen without a squint."""
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5), dimensions={"width": width, "height": height}))

    image = maps.render_image(FOREST_ID)

    assert 200 <= max(image.size) <= 400


def test_every_kind_of_tile_gets_its_own_colour(tmp_path):
    maps = store(tmp_path)
    walls = {(6, 5)}
    maps.record(
        make_snapshot(
            player=(5, 5),
            terrain=terrain_for((1, 1), walls),
            warps=[{"x": 3, "y": 3}],
        )
    )
    maps.record(make_snapshot(player=(4, 5), terrain=terrain_for((0, 1), walls)))

    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)

    assert tile_color(image, 4, 5) == explored_map.COLOR_PLAYER
    assert tile_color(image, 5, 5) == explored_map.COLOR_WALKED
    assert tile_color(image, 6, 5) == explored_map.COLOR_WALL
    assert tile_color(image, 3, 3) == explored_map.COLOR_WARP
    assert tile_color(image, 2, 2) == explored_map.COLOR_SEEN
    assert tile_color(image, 15, 15) == explored_map.COLOR_UNKNOWN
    probes = [(4, 5), (5, 5), (6, 5), (3, 3), (2, 2), (15, 15)]
    assert len({tile_color(image, x, y) for x, y in probes}) == len(probes)


def test_a_warp_outranks_the_tile_underneath_it(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5), warps=[{"x": 3, "y": 3, "target_map_id": 13}]))

    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)

    assert tile_color(image, 3, 3) == explored_map.COLOR_WARP


def test_the_player_wears_a_ring_so_one_block_is_not_lost_in_the_map(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))

    image = maps.render_image(FOREST_ID, tile_px=TILE_PX)

    assert tile_color(image, 5, 5) == explored_map.COLOR_PLAYER
    ring_left = MAP_LEFT_PX + (5 * TILE_PX) - (2 * TILE_PX)
    ring_middle = MAP_TOP_PX + (5 * TILE_PX) + (TILE_PX // 2)
    assert image.getpixel((ring_left, ring_middle)) == explored_map.COLOR_PLAYER


def test_the_legend_names_a_colour_for_every_kind_of_tile(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))

    assert set(maps.summary(FOREST_ID)["legend"]) == {
        "cyan",
        "purple",
        "dark green",
        "green",
        "red",
        "black",
    }


def test_rendering_rejects_a_map_that_was_never_visited(tmp_path):
    maps = store(tmp_path)

    with pytest.raises(KeyError):
        maps.render_image(999)
    with pytest.raises(KeyError):
        maps.summary(999)
    assert maps.write_image(999, tmp_path / "nope.png") is None
    assert not (tmp_path / "nope.png").exists()


def test_write_image_lands_one_readable_png_and_no_temp_files(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))
    target = tmp_path / "workspace" / "latest_map.png"

    written = maps.write_image(FOREST_ID, target)

    assert written == target
    with Image.open(target) as reopened:
        assert reopened.format == "PNG"
        assert reopened.size == maps.render_image(FOREST_ID).size
    assert [path.name for path in target.parent.iterdir()] == ["latest_map.png"]


def test_the_summary_is_shape_and_counts_with_no_grid_in_it(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5), warps=[{"x": 3, "y": 3}]))

    payload = maps.summary(FOREST_ID)

    assert payload["map_id"] == FOREST_ID
    assert payload["map_name"] == FOREST_NAME
    assert (payload["width"], payload["height"]) == (20, 18)
    assert payload["player"] == {"x": 5, "y": 5}
    assert payload["warps"] == [{"x": 3, "y": 3}]
    assert payload["coverage"]["total"] == 360
    assert "ascii" not in payload


def test_the_revision_only_moves_when_the_map_learns_something(tmp_path):
    """The image is redrawn off this; standing still must not redraw anything."""
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))
    settled = maps.revision

    for _ in range(5):
        maps.record(make_snapshot(player=(5, 5)))
    assert maps.revision == settled

    maps.record(make_snapshot(player=(6, 5)))
    assert maps.revision == settled + 1


# ---------------------------------------------------------------------------
# The grid accessor
# ---------------------------------------------------------------------------


def test_grid_hands_back_the_raw_tile_sets(tmp_path):
    maps = store(tmp_path)
    walls = {(6, 5)}
    maps.record(
        make_snapshot(
            player=(5, 5),
            terrain=terrain_for((1, 1), walls),
            warps=[{"x": 3, "y": 3}],
        )
    )
    maps.record(make_snapshot(player=(4, 5), terrain=terrain_for((0, 1), walls)))

    grid = maps.grid(FOREST_ID)

    assert set(grid) == {"width", "height", "seen", "walkable", "walked", "warps"}
    assert (grid["width"], grid["height"]) == (20, 18)
    assert grid["walked"] == {(5, 5), (4, 5)}
    assert grid["warps"] == {(3, 3)}
    assert (6, 5) in grid["seen"]
    assert (6, 5) not in grid["walkable"]
    assert grid["walked"] <= grid["walkable"] <= grid["seen"]


def test_grid_is_none_for_a_map_that_was_never_recorded(tmp_path):
    assert store(tmp_path).grid(999) is None


def test_grid_hands_out_copies_not_the_live_sets(tmp_path):
    """The overlay folds these into its own drawing; it must not be able to bite."""
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))

    maps.grid(FOREST_ID)["walked"].clear()

    assert maps.visited(FOREST_ID) == {(5, 5)}


# ---------------------------------------------------------------------------
# Nearest unexplored
# ---------------------------------------------------------------------------


def test_unexplored_nearest_finds_the_closest_unwalked_tile(tmp_path):
    maps = store(tmp_path)
    maps.record(
        make_snapshot(
            player=(1, 1),
            terrain=[[1] * 3 for _ in range(3)],
            top_left=(0, 0),
            dimensions={"width": 3, "height": 3},
        )
    )

    assert maps.summary(FOREST_ID)["unexplored_nearest"] == {"x": 1, "y": 0, "distance": 1.0}


def test_unexplored_nearest_is_null_once_everything_walkable_is_walked(tmp_path):
    maps = store(tmp_path)
    corridor = dict(
        terrain=[[1, 1]],
        top_left=(0, 0),
        dimensions={"width": 2, "height": 1},
    )

    maps.record(make_snapshot(player=(0, 0), **corridor))
    assert maps.summary(FOREST_ID)["unexplored_nearest"] is not None

    maps.record(make_snapshot(player=(1, 0), **corridor))
    assert maps.summary(FOREST_ID)["unexplored_nearest"] is None


def test_unexplored_nearest_measures_from_an_explicit_player_position(tmp_path):
    maps = store(tmp_path)
    patch = dict(
        terrain=[[1] * 3 for _ in range(3)],
        top_left=(0, 0),
        dimensions={"width": 3, "height": 3},
    )
    maps.record(make_snapshot(player=(1, 1), **patch))
    maps.record(make_snapshot(player=(2, 2), **patch))

    payload = maps.summary(FOREST_ID, player=(1, 1))

    assert payload["player"] == {"x": 1, "y": 1}  # not the stored (2, 2)
    assert payload["unexplored_nearest"] == {"x": 1, "y": 0, "distance": 1.0}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_exactly(tmp_path):
    maps = store(tmp_path)
    terrain = [[1] * 10 for _ in range(9)]
    terrain[2] = [0] * 10
    maps.record(make_snapshot(player=(5, 5), terrain=terrain, warps=[{"x": 3, "y": 3}]))
    maps.record(make_snapshot(player=(6, 5), terrain=terrain))
    maps.record(make_snapshot(map_id=1, map_name="VIRIDIAN CITY", player=(9, 9)))
    maps.save()

    reloaded = ExploredMaps(maps.path)

    assert reloaded.map_ids() == maps.map_ids()
    assert reloaded.current_map_id == maps.current_map_id
    for map_id in maps.map_ids():
        assert reloaded.visited(map_id) == maps.visited(map_id)
        assert reloaded.coverage(map_id) == maps.coverage(map_id)
        assert reloaded.summary(map_id) == maps.summary(map_id)


def test_saving_leaves_no_temp_files_behind(tmp_path):
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))
    maps.save()
    maps.save()

    assert [path.name for path in tmp_path.iterdir()] == ["explored_maps.json"]
    assert maps.dirty is False


def test_a_full_forest_store_stays_small(tmp_path):
    """A 34x48 map as coordinate pairs would be tens of KB; bitmasks are not."""
    maps = store(tmp_path)
    for y in range(4, 48, 8):
        for x in range(4, 34, 8):
            maps.record(make_snapshot(player=(x, y), dimensions={"width": 34, "height": 48}))
    maps.save()

    assert maps.coverage(FOREST_ID)["seen"] == 34 * 48
    assert maps.path.stat().st_size < 4000


@pytest.mark.parametrize("junk", ["{not json", "[]", '{"maps": 7}', ""])
def test_a_corrupt_store_loads_as_empty_instead_of_raising(tmp_path, junk):
    path = tmp_path / "explored_maps.json"
    path.write_text(junk, encoding="utf-8")

    maps = ExploredMaps(path)

    assert maps.map_ids() == []
    assert maps.current_map_id is None

    # ...and it recovers: the next save overwrites the junk with a valid store.
    maps.record(make_snapshot(player=(5, 5)))
    maps.save()
    assert json.loads(path.read_text(encoding="utf-8"))["maps"]


def test_an_unwritable_path_is_logged_not_raised(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    maps = ExploredMaps(blocker / "explored_maps.json")
    maps.record(make_snapshot(player=(5, 5)))

    maps.save()

    assert "could not save" in capsys.readouterr().out
    assert maps.visited(FOREST_ID) == {(5, 5)}


def test_a_missing_store_starts_empty(tmp_path):
    maps = store(tmp_path)

    assert maps.map_ids() == []
    assert maps.current_map_id is None


# ---------------------------------------------------------------------------
# Visit counts
# ---------------------------------------------------------------------------


def test_a_tile_counts_one_visit_per_arrival(tmp_path):
    maps = store(tmp_path)

    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(player=(6, 5)))
    maps.record(make_snapshot(player=(5, 5)))

    assert maps.visit_count(FOREST_ID, 5, 5) == 2
    assert maps.visit_count(FOREST_ID, 6, 5) == 1
    assert maps.visit_count(FOREST_ID, 9, 9) == 0
    assert maps.visit_count(999, 5, 5) == 0


def test_standing_still_does_not_count_as_visiting_again(tmp_path):
    """The live loop records the same standing position ten times a second."""
    maps = store(tmp_path)

    for _ in range(20):
        maps.record(make_snapshot(player=(5, 5)))

    assert maps.visit_count(FOREST_ID, 5, 5) == 1


def test_coming_back_to_a_map_counts_as_an_arrival(tmp_path):
    """Leaving and returning to the same tile is a revisit, not standing still."""
    maps = store(tmp_path)

    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(map_id=1, map_name="VIRIDIAN CITY", player=(9, 9)))
    maps.record(make_snapshot(player=(5, 5)))

    assert maps.visit_count(FOREST_ID, 5, 5) == 2


def test_visited_still_returns_a_plain_set_of_tiles(tmp_path):
    """agent_runtime shades the frame from this; it must keep its old shape."""
    maps = store(tmp_path)
    maps.record(make_snapshot(player=(5, 5)))
    maps.record(make_snapshot(player=(6, 5)))
    maps.record(make_snapshot(player=(5, 5)))

    visited = maps.visited(FOREST_ID)

    assert isinstance(visited, set)
    assert visited == {(5, 5), (6, 5)}


def test_visit_counts_survive_a_save_and_load(tmp_path):
    maps = store(tmp_path)
    for player in [(5, 5), (6, 5), (5, 5), (6, 5), (5, 5)]:
        maps.record(make_snapshot(player=player))
    maps.save()

    reloaded = ExploredMaps(maps.path)

    assert reloaded.visit_count(FOREST_ID, 5, 5) == 3
    assert reloaded.visit_count(FOREST_ID, 6, 5) == 2
    assert reloaded.visited(FOREST_ID) == maps.visited(FOREST_ID)


def test_a_store_written_before_counts_existed_migrates_to_one_each(tmp_path):
    maps = store(tmp_path)
    for player in [(5, 5), (6, 5), (5, 5)]:
        maps.record(make_snapshot(player=player))
    maps.save()

    legacy = json.loads(maps.path.read_text(encoding="utf-8"))
    assert legacy["maps"][str(FOREST_ID)].pop("visit_counts")  # as an older build wrote it
    maps.path.write_text(json.dumps(legacy), encoding="utf-8")

    reloaded = ExploredMaps(maps.path)

    assert reloaded.visited(FOREST_ID) == {(5, 5), (6, 5)}  # the map memory survives
    assert reloaded.visit_count(FOREST_ID, 5, 5) == 1  # only the tally restarts


# ---------------------------------------------------------------------------
# Compass
# ---------------------------------------------------------------------------
