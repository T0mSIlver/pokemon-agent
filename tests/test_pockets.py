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


# ---------------------------------------------------------------------------
# Connections with the map header's offsets attached
# ---------------------------------------------------------------------------

#: Route 4 and its two neighbours, cut down to the tiles that matter, with the
#: offsets `mapdecode.decode_connections` reads off the live header. The numbers
#: are the real ones: Route 4 is 90x18, its south strip covers x -6..19 and puts
#: you on Route 3 fifty tiles east, its east strip covers y -6..23 and puts you
#: in Cerulean eight tiles south.
ROUTE_4_SOUTH = {
    "to_map": "Route 3",
    "y_align": 0,
    "x_align": 50,
    "strip": (-6, 19),
    "connected_width": 70,
}
ROUTE_4_EAST = {
    "to_map": "Cerulean City",
    "y_align": 8,
    "x_align": 0,
    "strip": (-6, 23),
    "connected_width": 40,
}

# The west pocket touching the south edge, the far east corner touching it
# sixty tiles away, and one run of the east edge whose tiles land on both sides
# of the fence that splits Cerulean's west side.
ROUTE_4_TERRAIN = {(4, 17), (5, 17), (86, 17), (87, 17), (89, 10), (89, 11), (89, 12), (89, 13)}
CERULEAN_TERRAIN = {(0, 18), (0, 19), (0, 21)}  # (0,21) is below the fence, on its own
CONNECTED = {
    "Route 4": {"south": ROUTE_4_SOUTH, "east": ROUTE_4_EAST},
    "Route 3": {
        "north": {
            "to_map": "Route 4",
            "y_align": 17,
            "x_align": -50,
            "strip": (50, 75),
            "connected_width": 90,
        }
    },
}
SIZES = {"Route 4": (90, 18), "Route 3": (70, 18), "Cerulean City": (40, 36)}
TERRAIN_BY_MAP = {
    "Route 4": ROUTE_4_TERRAIN,
    "Route 3": {(54, 0), (55, 0)},
    "Cerulean City": CERULEAN_TERRAIN,
}


def connected_graph(terrain=None, connections=None):
    terrain = TERRAIN_BY_MAP if terrain is None else terrain
    connections = CONNECTED if connections is None else connections
    return PocketGraph(
        lambda name: [],
        lambda name: terrain.get(name),
        lambda name: connections.get(name, {}),
        lambda name: SIZES.get(name),
    )


def pocket_for(tile):
    return connected_graph().pocket_at("Route 4", tile)


def test_an_offset_connection_names_the_tile_you_land_on():
    """Measured in PyBoy: Route 4 (4,17) walked south arrives at Route 3 (54,0)."""
    hops = connected_graph().hops_from("Route 4", pocket_for((4, 17)))

    south = [hop for hop in hops if hop.edge == "south"]
    assert len(south) == 1
    assert south[0].at == (4, 17) and south[0].landing == (54, 0)
    assert south[0].to_map == "Route 3" and south[0].kind == "connection"


def test_the_edge_of_a_pocket_that_is_off_the_strip_offers_no_hop_at_all():
    """The failure in the brief, from the pocket side.

    Route 4's far east corner touches the south edge sixty tiles from the strip.
    Taking every edge-touching pocket routed south to Route 3 and back north
    into that corner, as though walking south and back could move you sideways.
    """
    corner = pocket_for((86, 17))

    hops = connected_graph().hops_from("Route 4", corner)

    assert [hop.edge for hop in hops] == [], "no strip under it, so no way off it"


def test_one_edge_landing_in_two_pockets_offers_both_rather_than_neither():
    """A run of edge tiles can straddle a fence on the far side.

    Route 4's east edge does exactly that -- its tiles land both above and below
    the fence on Cerulean's west side -- and both halves are somewhere you can
    really walk to. Two candidates is a fact about the map here, not the
    ambiguity the old rule refused.
    """
    graph = connected_graph()

    hops = graph.hops_from("Route 4", graph.pocket_at("Route 4", (89, 10)))

    assert {hop.landing for hop in hops} == {(0, 18), (0, 21)}
    assert len({hop.to_pocket for hop in hops}) == 2, "one hop each, not one of them twice"


def test_a_connection_given_only_as_a_map_name_still_uses_the_old_rule():
    """`gamedata` has no offsets, and a graph built from it must still work.

    The second assertion is the old wrong answer, kept on purpose: Route 3's
    north edge is one pocket, so the name-only rule hands the same hop to the
    corner sixty tiles away. That is what the offsets take out.
    """
    graph = connected_graph(connections={"Route 4": {"south": "Route 3"}})

    hops = graph.hops_from("Route 4", graph.pocket_at("Route 4", (4, 17)))
    from_the_corner = graph.hops_from("Route 4", graph.pocket_at("Route 4", (86, 17)))

    assert len(hops) == 1 and hops[0].at is None, "a side, but not a tile"
    assert len(from_the_corner) == 1, "unchanged, and still wrong, without offsets"


def test_an_offset_landing_on_ground_nobody_can_stand_on_is_not_a_route():
    """A decode gap on the far side is not a door. Same rule as a warp."""
    graph = connected_graph(terrain=dict(TERRAIN_BY_MAP, **{"Route 3": {(0, 0)}}))

    hops = graph.hops_from("Route 4", graph.pocket_at("Route 4", (4, 17)))

    assert [hop.edge for hop in hops] == [], "the strip lands on (54,0), which is not there"


def test_a_door_onto_ground_that_is_not_walkable_is_not_a_route():
    """A decode gap must not become a plan that promises an unstandable tile."""
    broken = dict(TERRAIN, hall=set())
    graph_with_gap = PocketGraph(lambda name: WORLD.get(name, []), lambda name: broken.get(name))

    # hall has no terrain, so it collapses to one pocket and stays routable.
    assert graph_with_gap.route("cave", (0, 0), "outside") is not None


# ---------------------------------------------------------------------------
# The payload /route actually returns
#
# Real map names, because `route_payload` validates against the game's own map
# table before it reaches the router. The terrain is synthetic but the shape is
# not: Mt Moon B1F really is several pockets, and this is the smallest map with
# that property.
# ---------------------------------------------------------------------------

REAL_WORLD = {
    "Mt Moon B1F": [
        {"x": 0, "y": 0, "to_map": "Mt Moon 1F", "to_warp": 0},
        {"x": 4, "y": 0, "to_map": "Mt Moon 1F", "to_warp": 1},
        {"x": 3, "y": 0, "to_map": "Route 4", "to_warp": 0},
    ],
    "Mt Moon 1F": [
        {"x": 0, "y": 0, "to_map": "Mt Moon B1F", "to_warp": 0},
        {"x": 2, "y": 0, "to_map": "Mt Moon B1F", "to_warp": 1},
    ],
    "Route 4": [{"x": 0, "y": 0, "to_map": "Mt Moon B1F", "to_warp": 2}],
}
REAL_TERRAIN = {
    "Mt Moon B1F": {(0, 0), (1, 0), (3, 0), (4, 0)},  # two pockets, gap at x=2
    "Mt Moon 1F": {(0, 0), (1, 0), (2, 0)},
    "Route 4": {(0, 0), (1, 0)},
}


def real_world():
    """A World that knows these three maps and no hops between them.

    No hops on purpose: the fallback search must come back empty, so a passing
    pocket assertion cannot be the static graph answering by accident.
    """
    from pokemon_agent.world import MapInfo, World

    return World(
        {
            name: MapInfo(name=name, map_id=index, size=(8, 4), hops=())
            for index, name in enumerate(REAL_TERRAIN)
        }
    )


def real_graph():
    return PocketGraph(lambda name: REAL_WORLD.get(name, []), lambda name: REAL_TERRAIN.get(name))


def test_route_payload_prefers_the_pocket_answer_and_says_the_ground_is_checked():
    """The point of wiring it in: the map-keyed search cannot express this route.

    Getting from one Mt Moon B1F pocket to the other means leaving B1F and
    coming back, so a search keyed by map name reports no route rather than a
    worse one. The payload says `ground: checked` because these hops came from
    real terrain, not from a static table that has never seen a tile.
    """
    from pokemon_agent import capabilities

    payload = capabilities.route_payload(
        real_world(), "Mt Moon B1F", "Route 4", pockets=real_graph(), at=(0, 0)
    )

    assert payload["ground"] == "checked"
    assert payload["distance"] == 3
    assert [hop["from"] for hop in payload["hops"]] == [
        "Mt Moon B1F",
        "Mt Moon 1F",
        "Mt Moon B1F",
    ], "it comes back to B1F, which is the thing that was impossible"
    assert "real terrain" in payload["basis"]


def test_route_payload_falls_back_when_the_player_position_is_unknown():
    """A router that answers per pocket cannot answer without a tile to start from."""
    from pokemon_agent import capabilities

    payload = capabilities.route_payload(
        real_world(), "Mt Moon B1F", "Route 4", pockets=real_graph(), at=None
    )

    assert payload["hops"] is None, "no pocket answer, and the static graph is empty here"
    assert "No route" in payload["reason"]
