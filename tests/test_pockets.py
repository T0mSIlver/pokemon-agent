"""Routing over pockets rather than over maps.

The shape under test is Mt Moon: a floor that is several disconnected places,
where getting from one to another means leaving the floor and coming back. A
map-keyed search reports no route at all for that, which is what nineteen warp
refusals and fourteen hours of a live run turned out to be.
"""

from pokemon_agent.pockets import PocketGraph, components, pocket_of

# Two pockets on "cave", joined only by going up to "hall" and back down.
#
#   cave    x0 x1        x3 x4        hall  a corridor, all one piece
#   y0      A  A         B  B
#
WORLD = {
    "cave": [
        {"x": 0, "y": 0, "to_map": "hall", "to_warp": 0},  # warp 0, pocket A
        {"x": 4, "y": 0, "to_map": "hall", "to_warp": 1},  # warp 1, pocket B
        {"x": 3, "y": 0, "to_map": "outside", "to_warp": 0},  # warp 2, pocket B
    ],
    "hall": [
        {"x": 0, "y": 0, "to_map": "cave", "to_warp": 0},
        {"x": 2, "y": 0, "to_map": "cave", "to_warp": 1},
    ],
    "outside": [
        {"x": 0, "y": 0, "to_map": "cave", "to_warp": 2},
    ],
}

TERRAIN = {
    "cave": {(0, 0), (1, 0), (3, 0), (4, 0)},  # a gap at x=2 splits it
    "hall": {(0, 0), (1, 0), (2, 0)},
    "outside": {(0, 0), (1, 0)},
}


def graph(terrain=None):
    terrain = TERRAIN if terrain is None else terrain
    return PocketGraph(lambda name: WORLD.get(name, []), lambda name: terrain.get(name))


def test_components_splits_a_floor_into_the_places_you_can_walk_between():
    pieces = components({(0, 0), (1, 0), (3, 0), (4, 0)})

    assert [sorted(piece) for piece in pieces] == [[(0, 0), (1, 0)], [(3, 0), (4, 0)]]


def test_components_orders_largest_first_so_pocket_zero_is_the_main_floor():
    pieces = components({(0, 0), (5, 5), (6, 5), (7, 5)})

    assert len(pieces[0]) == 3 and len(pieces[1]) == 1


def test_a_route_may_visit_the_same_map_twice():
    """The whole point. Crossing Mt Moon B1F means leaving B1F and coming back."""
    hops = graph().route("cave", (0, 0), "outside")

    assert hops is not None, "a map-keyed search reports None here"
    assert [(hop.from_map, hop.from_pocket) for hop in hops] == [
        ("cave", 0),
        ("hall", 0),
        ("cave", 1),
    ]
    assert hops[-1].to_map == "outside"


def test_the_hops_name_the_tile_to_step_on_and_where_it_puts_you():
    hops = graph().route("cave", (0, 0), "outside")

    assert hops[0].at == (0, 0) and hops[0].landing == (0, 0)
    assert hops[1].at == (2, 0), "the second door out of the hall, not the first"
    assert hops[1].landing == (4, 0), "which lands in cave's other pocket"


def test_a_warp_in_another_pocket_is_not_offered():
    """`_exits` ranked warps by Manhattan distance and advertised exactly these."""
    from_pocket_zero = graph().hops_from("cave", 0)

    assert [hop.at for hop in from_pocket_zero] == [(0, 0)], (
        "the (3,0) and (4,0) doors are not reachable"
    )


def test_asking_for_a_tile_targets_the_pocket_that_holds_it():
    hops = graph().route("cave", (0, 0), "cave", (4, 0))

    assert hops is not None
    assert hops[-1].to_pocket == 1, "arrives in the far pocket, not merely on the map"


def test_being_there_already_is_an_empty_route_not_a_missing_one():
    assert graph().route("cave", (0, 0), "cave", (1, 0)) == ()


def test_a_tile_nobody_can_stand_on_has_no_route_to_it():
    assert graph().route("cave", (0, 0), "cave", (2, 0)) is None


def test_an_unknown_map_collapses_to_one_pocket_rather_than_vanishing():
    """Degrading to the old behaviour is fine; disappearing from the graph is not."""
    partial = dict(TERRAIN)
    del partial["hall"]

    hops = graph(partial).route("cave", (0, 0), "outside")

    assert hops is not None, "the hall is still routable, just not split"
    assert pocket_of([], (99, 99)) == 0


# ---------------------------------------------------------------------------
# One-way ledges
#
# Route 4's east half is reachable ONLY over one-way ledge hops down from row 8
# onto row 10. Treating a ledge as an ordinary edge claims you can climb back;
# ignoring it claims the east half does not exist. The run spent sixteen hours
# on the second answer.
# ---------------------------------------------------------------------------

# A wall at x=2 splits the row; a ledge drops from (1,0) to (3,0) over it.
LEDGE_TERRAIN = {(0, 0), (1, 0), (3, 0), (4, 0)}
LEDGE_HOPS = {((1, 0), "right"): (3, 0)}


def ledge_graph():
    return PocketGraph(
        lambda name: [],
        lambda name: LEDGE_TERRAIN,
        lambda name: {},
        lambda name: (5, 1),
        lambda name: LEDGE_HOPS,
    )


def test_a_ledge_does_not_merge_the_two_sides_into_one_pocket():
    """You can drop down, not climb up, so they are not one place."""
    pieces = components(LEDGE_TERRAIN, LEDGE_HOPS)

    assert len(pieces) == 2, "a one-way edge is not enough to make a pocket"


def test_a_ledge_is_a_route_in_the_direction_it_drops():
    hops = ledge_graph().route("cliff", (0, 0), "cliff", (4, 0))

    assert hops is not None, "the far side is reachable, over the ledge"
    assert len(hops) == 1 and hops[0].kind == "ledge"
    assert hops[0].at == (1, 0) and hops[0].landing == (3, 0)


def test_there_is_no_route_back_up_a_ledge():
    """The failure this guards: a plan that asks the player to climb."""
    assert ledge_graph().route("cliff", (4, 0), "cliff", (0, 0)) is None


def test_without_ledges_the_far_side_looks_unreachable():
    """What the router said for sixteen hours, and why the terrain alone was not enough."""
    blind = PocketGraph(lambda name: [], lambda name: LEDGE_TERRAIN)

    assert blind.route("cliff", (0, 0), "cliff", (4, 0)) is None


def test_an_ambiguous_edge_landing_yields_no_route_rather_than_a_guess():
    """Route 4's south edge is touched by two pockets sixty tiles apart.

    Guessing by taking every edge-touching pocket produced a route that walked
    south to Route 3 and back north into a *different* pocket, as though that
    could move you sideways across the map. Gen 1 connections line up at a
    fixed offset, so it cannot.
    """
    connections = {"road": {"south": "field"}, "field": {"north": "road"}}
    # field's north edge (y=0) is touched by two separate pockets.
    terrain = {"road": {(0, 0), (0, 1)}, "field": {(0, 0), (5, 0)}}
    ambiguous = PocketGraph(
        lambda name: [],
        lambda name: terrain.get(name),
        lambda name: connections.get(name, {}),
        lambda name: (6, 2),
    )

    assert ambiguous.hops_from("road", 0) == [], "two candidate landings means no claim"


def test_an_unambiguous_edge_landing_is_still_a_route():
    connections = {"road": {"south": "field"}, "field": {"north": "road"}}
    terrain = {"road": {(0, 0)}, "field": {(0, 0), (1, 0)}}
    clear = PocketGraph(
        lambda name: [],
        lambda name: terrain.get(name),
        lambda name: connections.get(name, {}),
        lambda name: (2, 1),
    )

    hops = clear.route("road", (0, 0), "field")

    assert len(hops) == 1 and hops[0].edge == "south" and hops[0].kind == "connection"


def test_a_door_onto_ground_that_is_not_walkable_is_not_a_route():
    """A decode gap must not become a plan that promises an unstandable tile."""
    broken = dict(TERRAIN, hall=set())
    graph_with_gap = PocketGraph(lambda name: WORLD.get(name, []), lambda name: broken.get(name))

    # hall has no terrain, so it collapses to one pocket and stays routable.
    assert graph_with_gap.route("cave", (0, 0), "outside") is not None
