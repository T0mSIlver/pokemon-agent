"""What /sim, /frontier and /goto are allowed to claim, and on whose authority.

Two sources feed every answer these endpoints give: the live 10x9 collision
window, which is this frame, and the explored-map store, which is memory. They
disagree, and the store has been wrong in a specific way — it has held tiles
nobody can stand on, sampled off a player halfway through a ledge jump. So the
rule is fixed here: inside the window the frame decides, outside it the answer
is a belief and says so.
"""

from __future__ import annotations

from pokemon_agent import capabilities

#: A 10x9 window around a player at (10, 12), the Route 3 pocket's shape: open
#: shelf above, a ledge face along y = 11, the pocket below it.
WINDOW_ORIGIN = (6, 8)


def snapshot(player=(10, 12), ledge_hops=None) -> dict:
    rows = []
    for y in range(8, 17):
        if y == 11 or y >= 14:
            rows.append([0] * 10)
        else:
            rows.append([1] * 10)
    return {
        "map_id": 14,
        "map_name": "Route 3",
        "player_position": {"x": player[0], "y": player[1]},
        "facing": "up",
        "tileset": "OVERWORLD",
        "window_top_left": {"x": WINDOW_ORIGIN[0], "y": WINDOW_ORIGIN[1]},
        "terrain": rows,
        "sprites": [],
        "warps": [],
        "map_dimensions": {"width": 70, "height": 18},
        "ledge_hops": ledge_hops or {},
    }


def window_tiles() -> set:
    return {(WINDOW_ORIGIN[0] + x, WINDOW_ORIGIN[1] + y) for x in range(10) for y in range(9)}


def store(extra_walkable=(), width=70, height=18) -> dict:
    walkable = {(x, y) for x in range(6, 23) for y in (12, 13)}
    walkable |= set(extra_walkable)
    return {
        "width": width,
        "height": height,
        "walkable": walkable,
        "walked": {(10, 12)},
        "seen": set(walkable),
        "warps": set(),
    }


# ---------------------------------------------------------------------------
# collision_from
# ---------------------------------------------------------------------------


def test_the_live_window_overrules_the_store_where_they_overlap():
    remembered = store(extra_walkable=[(10, 11), (11, 11)])  # tiles nobody can stand on

    collision = capabilities.collision_from(snapshot(), remembered)

    assert (10, 11) not in collision["walkable"]
    assert (11, 11) not in collision["walkable"]
    assert collision["live"] == window_tiles()
    # Beyond the window the store is all there is, and it survives untouched.
    assert (20, 12) in collision["walkable"]
    assert (20, 12) not in collision["live"]


def test_a_remembered_tile_the_window_cannot_see_stays_a_belief():
    collision = capabilities.collision_from(snapshot(), store())

    believed = [tile for tile in collision["walkable"] if tile not in collision["live"]]

    assert believed
    assert all(tile[0] > 15 or tile[0] < 6 for tile in believed)


def test_the_tile_a_ledge_jumps_over_is_never_walkable():
    """Even a store that swears somebody stood there. Nobody did — they jumped."""
    remembered = store(extra_walkable=[(10, 11)])
    here = snapshot(player=(10, 10), ledge_hops={"down": {"x": 10, "y": 12}})

    collision = capabilities.collision_from(here, remembered)

    assert (10, 11) not in collision["walkable"]
    assert collision["ledges"][((10, 10), "down")] == (10, 12)


def test_the_basis_line_names_both_halves():
    basis = capabilities.collision_basis(capabilities.collision_from(snapshot(), store()))

    assert "live 90-tile window" in basis
    assert "remembered" in basis
    assert capabilities.collision_basis(capabilities.collision_from(snapshot(), None)) == (
        "the live 90-tile window only"
    )


# ---------------------------------------------------------------------------
# /sim
# ---------------------------------------------------------------------------


def test_sim_answers_a_ledge_with_a_jump_and_says_it_is_one_way():
    here = snapshot(player=(10, 10), ledge_hops={"down": {"x": 10, "y": 12}})

    payload = capabilities.simulate_payload(["down", "down"], here, store())

    assert payload["blocked_at"] is None
    assert payload["end"] == [10, 13]
    assert payload["hops"] == [
        {"at": 0, "direction": "down", "from": [10, 10], "to": [10, 12], "one_way": True}
    ]
    assert "one way" in payload["note"]
    assert payload["certain"] is True


def test_sim_says_where_its_answer_stops_being_this_frame():
    payload = capabilities.simulate_payload(["right:8"], snapshot(), store())

    assert payload["unverified_from"] == 5  # the step into (16, 12), beyond the window
    assert payload["certain"] is False
    assert "remembered map" in payload["note"]
    assert "remembered" in payload["basis"]


# ---------------------------------------------------------------------------
# /frontier
# ---------------------------------------------------------------------------


def test_frontier_never_climbs_the_ledge_the_store_says_it_can():
    remembered = store(extra_walkable=[(x, 11) for x in range(6, 16)])

    payload = capabilities.frontier_payload(snapshot(), remembered, {(10, 12)})
    tiles = [tuple(tile) for tile in payload["tiles"]]

    assert tiles
    assert not [tile for tile in tiles if tile[1] < 12]


def test_frontier_separates_what_it_saw_from_what_it_remembers():
    payload = capabilities.frontier_payload(snapshot(), store(), {(10, 12)})

    confirmed = {tuple(tile) for tile in payload["confirmed"]}
    tiles = {tuple(tile) for tile in payload["tiles"]}

    assert (11, 12) in confirmed  # next door, and the window is showing it
    assert (20, 12) in tiles and (20, 12) not in confirmed  # remembered only
    assert payload["confirmed_count"] + payload["believed_count"] == payload["count"]
    assert payload["believed_count"] > 0
    assert "belief" in payload["note"]
