"""The frontier menu, and the distances it now carries.

The failure these tests are the receipt for: a leg that had just won the Cascade
Badge stood in Cerulean City with "Defeated Misty" on a five-item frontier menu
and the gym twenty steps away, and spent 739 presses -- 33% of the leg -- on
Route 4, west of the city, hunting Route 9 and Saffron, which are east and south
and were not on the menu at all. It read ``GET /frontier`` 63 times. The menu
said what was open and nothing about where any of it was.

Each of these was checked by breaking the thing it tests -- sorting the menu by
distance, rendering an unknown location as an unreachable one, dropping the
sentence that says what a hop is, placing Bill from his label -- and confirming
it goes red. A test that passes against the code it was written for and against
the code it was written to prevent is not a test.
"""

from __future__ import annotations

from pokemon_agent import gamedata
from pokemon_agent.milestones import MILESTONES_BY_ID
from pokemon_agent.milestones import frontier as milestone_frontier
from pokemon_agent.objectives import MILESTONE_MAPS, frontier_objective

#: Exactly what RAM reads in ``saves/repeat_guard_deploy.state``: Cerulean City,
#: sixteen rungs, the leg that lost the 739 presses. Frozen here so the case is
#: testable in CI, where there is no ROM.
CERULEAN_16 = (
    "EVENT_GOT_STARTER",
    "EVENT_BATTLED_RIVAL_IN_OAKS_LAB",
    "EVENT_GOT_OAKS_PARCEL",
    "EVENT_OAK_GOT_PARCEL",
    "EVENT_GOT_POKEDEX",
    "EVENT_GOT_TOWN_MAP",
    "EVENT_BEAT_BROCK",
    "BADGE_BOULDER",
    "EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD",
    "EVENT_GOT_HELIX_FOSSIL",
    "EVENT_BEAT_CERULEAN_RIVAL",
    "EVENT_MET_BILL",
    "EVENT_GOT_SS_TICKET",
    "EVENT_RUBBED_CAPTAINS_BACK",
    "EVENT_GOT_HM01",
    "EVENT_SS_ANNE_LEFT",
)

#: The hop counts the packaged graph really answers from Cerulean City, so the
#: fake router and the real one agree about this menu.
CERULEAN_HOPS = {
    ("Cerulean City", "Route 22"): 6,
    ("Cerulean City", "Cerulean Gym"): 1,
    ("Cerulean City", "Pewter Museum 1F"): 4,
    ("Cerulean City", "Pokemon Fan Club"): 5,
    ("Cerulean City", "Route 2 Gate"): 5,
}


class FakeRouter:
    """A map graph of exactly the hop counts a test wants to talk about."""

    def __init__(self, distances):
        self._distances = distances

    def route(self, src: str, dst: str):
        if src == dst:
            return ()
        hops = self._distances.get((src, dst))
        return None if hops is None else tuple(range(hops))


def _summary(reached, **kwargs) -> str:
    record = frontier_objective(reached, priority=1, **kwargs)
    assert record is not None
    return record["summary"]


# ---------------------------------------------------------------------------
# Where a rung is earned
# ---------------------------------------------------------------------------


def test_every_recorded_map_is_a_map_the_graph_knows():
    """A typo here renders as a rung the router can never reach.

    Checked against the generated world rather than a list in this file: the
    milestone ladder and the map graph come from two different generators and
    this is the only place the two are joined.
    """
    known = set(gamedata.map_names())
    for entry in gamedata.world().values():
        for edge in (entry.get("connections") or {}).values():
            if isinstance(edge, str):
                known.add(edge)
        for warp in entry.get("warps") or []:
            target = warp.get("to_map")
            if isinstance(target, str):
                known.add(target)
    assert {rung: where for rung, where in MILESTONE_MAPS.items() if where not in known} == {}


def test_every_recorded_rung_is_a_rung():
    assert set(MILESTONE_MAPS) <= set(MILESTONES_BY_ID)


def test_the_item_rungs_agree_with_the_generated_item_placements():
    """The four rungs the game data can check on its own, checked against it.

    ``items.json`` lists ground items per map. It cannot locate a rung an NPC
    hands over, which is why the rest of :data:`MILESTONE_MAPS` is written out
    by hand -- but where it does know, the hand-written table has to agree with
    it or one of the two is wrong.
    """
    placements: dict[str, set[str]] = {}
    for map_name in gamedata.map_names():
        for entry in gamedata.items(map_name):
            placements.setdefault(str(entry.get("item")), set()).add(map_name)

    for rung, item in (
        ("ITEM_LIFT_KEY", "Lift Key"),
        ("ITEM_SILPH_SCOPE", "Silph Scope"),
        ("ITEM_CARD_KEY", "Card Key"),
        ("ITEM_SECRET_KEY", "Secret Key"),
    ):
        assert placements[item] == {MILESTONE_MAPS[rung]}, rung


def test_the_ss_anne_sailing_has_no_map_and_is_meant_not_to():
    """Absent beats approximate: the ship leaves, it is not somewhere you go."""
    assert "EVENT_SS_ANNE_LEFT" not in MILESTONE_MAPS


def test_no_rung_gets_its_map_from_its_own_label():
    """Bill is met on Route 25 and the flag is set inside his house.

    The rung that proves label text is not a location. Anything reading labels
    would put this one on Route 25 -- the map you walk through -- and would
    place "Got the Bike Voucher" nowhere at all, sounding equally sure of both.
    """
    assert MILESTONES_BY_ID["EVENT_MET_BILL"].label == "Met Bill on Route 25"
    assert MILESTONE_MAPS["EVENT_MET_BILL"] == "Bill's House"


# ---------------------------------------------------------------------------
# The rendered menu
# ---------------------------------------------------------------------------


def test_the_cerulean_menu_says_misty_is_the_near_one():
    """The 739-press case. Every open rung carries a distance and Misty's is 1."""
    summary = _summary(CERULEAN_16, here="Cerulean City", router=FakeRouter(CERULEAN_HOPS))

    assert "Defeated Misty [Cerulean Gym, 1 hop]" in summary
    assert "Beat the rival on Route 22 [Route 22, 6 hops]" in summary
    assert "Got the Old Amber in the Pewter Museum [Pewter Museum 1F, 4 hops]" in summary
    assert "Got the Bike Voucher [Pokemon Fan Club, 5 hops]" in summary
    assert "Got HM05 Flash [Route 2 Gate, 5 hops]" in summary
    # Five open rungs, five answers: none is left without one.
    assert summary.count("[") == 5
    assert "map-graph distance from Cerulean City" in summary


def test_the_menu_stays_in_ladder_order_rather_than_nearest_first():
    """Distance is a column, not a sort key.

    Misty is one hop away and fourth on the ladder among these five. Sorted by
    distance she would be first, and the row at the top of a sorted list reads
    as a recommendation whatever the sentence above it says. The recorded
    decision for this project is that the model orders off the menu and the
    harness only narrows it.
    """
    summary = _summary(CERULEAN_16, here="Cerulean City", router=FakeRouter(CERULEAN_HOPS))

    positions = [summary.index(m.label) for m in milestone_frontier(CERULEAN_16)]
    assert positions == sorted(positions)
    # The nearest option is emphatically not the one printed first.
    assert summary.index("Defeated Misty") > summary.index("Beat the rival on Route 22")
    assert "Ladder order, not a ranking" in summary


def test_an_unknown_location_does_not_render_as_a_distance():
    """A rung with no map on record must not read like a rung far away."""
    reached = tuple(rung for rung in CERULEAN_16 if rung != "EVENT_SS_ANNE_LEFT")
    assert "EVENT_SS_ANNE_LEFT" in {m.id for m in milestone_frontier(reached)}

    summary = _summary(reached, here="Cerulean City", router=FakeRouter(CERULEAN_HOPS))

    assert "The S.S. Anne set sail [no map on record]" in summary
    # Not silently dropped, and not dressed up as a route that could not be found.
    option = summary[summary.index("The S.S. Anne set sail") :].split(";")[0]
    assert "hop" not in option
    assert "no route" not in option


def test_a_map_the_graph_cannot_reach_is_not_a_hop_count():
    """Unreachable and far away are different answers and read differently."""
    summary = _summary(
        ("EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK"),
        here="Cerulean City",
        router=FakeRouter({("Cerulean City", "Cerulean Gym"): None}),
    )

    assert "Defeated Misty [Cerulean Gym, no route the map graph can find]" in summary
    assert "Cerulean Gym, 0 hop" not in summary


def test_standing_on_the_map_says_so_rather_than_zero_hops():
    summary = _summary(
        ("EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK"),
        here="Cerulean Gym",
        router=FakeRouter({}),
    )

    assert "Defeated Misty [Cerulean Gym, the map you are on]" in summary
    assert "0 hop" not in summary


def test_the_hedge_survives_the_distances():
    """A hop count must never become readable as a walkable route.

    The frontier's original promise -- prerequisites met is not reachability --
    is what distances are most likely to quietly undo, so it is asserted next to
    them rather than in a test of its own.
    """
    summary = _summary(CERULEAN_16, here="Cerulean City", router=FakeRouter(CERULEAN_HOPS))

    assert "not a claim that any of them can be reached on foot" in summary
    assert "hops are map-to-map moves, never tiles" in summary
    assert "never a promise the ground in between is walkable" in summary


def test_the_hint_stops_naming_geography_the_menu_now_carries():
    annotated = frontier_objective(
        CERULEAN_16, priority=1, here="Cerulean City", router=FakeRouter(CERULEAN_HOPS)
    )
    plain = frontier_objective(CERULEAN_16, priority=1)

    assert "comes off the map graph" in annotated["failure_hints"][0]
    assert "not geography" in plain["failure_hints"][0]


# ---------------------------------------------------------------------------
# Nothing measured, nothing claimed
# ---------------------------------------------------------------------------


def test_without_a_current_map_the_menu_renders_exactly_as_before():
    """No map, no measurement, and no half-computed brackets either."""
    summary = _summary(CERULEAN_16)

    assert "[" not in summary
    assert "hop" not in summary
    assert "map-graph distance" not in summary
    assert "where you are standing: Beat the rival on Route 22; Defeated Misty" in summary


def test_an_empty_map_graph_is_not_reported_as_a_world_with_no_routes(monkeypatch):
    """A checkout without generated data loses the brackets, not the menu."""
    from pokemon_agent import objectives, world

    objectives._default_router.cache_clear()
    monkeypatch.setattr(world.World, "load", classmethod(lambda cls, path=None: cls({})))
    try:
        summary = _summary(CERULEAN_16, here="Cerulean City")
    finally:
        objectives._default_router.cache_clear()

    assert "no route the map graph can find" not in summary
    assert "[" not in summary


def test_the_packaged_graph_answers_for_the_real_cerulean_frontier():
    """Not a fake router: the graph that actually ships.

    Guards the join between :data:`MILESTONE_MAPS` and ``world.json`` at the one
    moment it matters. If those names stop matching, every bracket on this menu
    silently turns into "no route the map graph can find".
    """
    from pokemon_agent import objectives

    objectives._default_router.cache_clear()
    summary = _summary(CERULEAN_16, here="Cerulean City")

    assert "no route the map graph can find" not in summary
    assert "no map on record" not in summary
    for (_, where), hops in CERULEAN_HOPS.items():
        plural = "" if hops == 1 else "s"
        assert f"[{where}, {hops} hop{plural}]" in summary
