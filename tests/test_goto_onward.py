"""What /goto says when there is no plan, and whether /route said it first.

Two tools answered the same question two ways. `/route Cerulean City` from
Route 4 returned ``distance 1, walk east``; `/goto Cerulean City` from the same
tile walked nothing and said the east was not reachable. Only one of them can
be right, and the emulator settles it: flooding the real ROM from that position
reaches 157 tiles, x 4..19, and there is no way east. Route 4 holds two pockets
of land — the Mt. Moon mouth and the Cerulean half — and the map table's east
connection belongs to the pocket the player is not standing in.

So `/goto` had the right answer and `/route` was overselling, and the fix is not
to loosen `/goto`. It is to make it say *which* kind of no it is answering —
ground that was looked at and is solid, or ground nobody has looked at yet — and
to hand back what is reachable instead of only what is not.

The run's own map made that distinction worth drawing twice over: its Route 4
store believed a corridor out to x = 24 that the map does not have. So the
refusal was right about the east and wrong about being sure, and the answer it
should have given was thirteen steps of walking followed by the same refusal
with the Mt. Moon door named.

The second half of the file is the other way a plan goes missing, and it has
nothing to do with maps. `world.route` searches over map *names*, so it keeps
one edge per pair of maps and drops the rest; Mt. Moon 1F has three ladders down
to B1F and the caller was handed whichever the file listed first. `/goto Mt Moon
B1F` spent a call failing to reach the ladder sixteen tiles away while standing
two tiles above one that works.

Both cases run off a captured fixture rather than the ROM, so they run in CI.
`route4_mt_moon_mouth.json` holds the live window read off PokemonRed.gb at
(15, 11), the run's own map store for Route 4 and Mt Moon 1F at that moment, and
the complete reachable set the emulator flood established.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pokemon_agent import capabilities
from pokemon_agent import world as world_mod
from pokemon_agent.world import Hop, MapInfo, World

FIXTURES = Path(__file__).parent / "fixtures"


def _unpack(rows) -> set[tuple[int, int]]:
    return {(x, y) for y, row in enumerate(rows) for x, flag in enumerate(row) if flag == "1"}


@pytest.fixture(scope="module")
def route4() -> dict:
    payload = json.loads((FIXTURES / "route4_mt_moon_mouth.json").read_text(encoding="utf-8"))
    payload["explored"] = {
        "width": payload["explored"]["width"],
        "height": payload["explored"]["height"],
        "seen": _unpack(payload["explored"]["seen"]),
        "walkable": _unpack(payload["explored"]["walkable"]),
    }
    payload["truth"]["reachable"] = _unpack(payload["truth"]["reachable"])
    return payload


# ---------------------------------------------------------------------------
# A fake overworld, so the walking loop can be driven without an emulator
# ---------------------------------------------------------------------------


class FakeWorld:
    """A walkable set, a player on it, and a store that learns what it is shown.

    Deliberately not the real thing: these cases are about which tile the loop
    chooses and what it says, and a hand-built grid is the only way to put a
    door somewhere awkward on purpose.
    """

    def __init__(
        self,
        walkable: set[tuple[int, int]],
        start: tuple[int, int],
        *,
        width: int,
        height: int,
        map_name: str = "Test Map",
        map_id: int = 1,
        warps: tuple[tuple[int, int], ...] = (),
        remember: bool = True,
    ) -> None:
        self.walkable = set(walkable)
        self.position = start
        self.width, self.height = width, height
        self.map_name, self.map_id = map_name, map_id
        self.warps = list(warps)
        #: Ground the map calls walkable and the game will not let anyone onto:
        #: an NPC standing in a doorway, a tile the decoder got wrong. It reads
        #: as open in every snapshot and refuses every step. This is the gap the
        #: planner has to survive, because no grid can hold it.
        self.refuses: set[tuple[int, int]] = set()
        self.seen: set[tuple[int, int]] = set()
        self.known: set[tuple[int, int]] = set()
        self.remember = remember
        self.batches: list[list[str]] = []
        self._look()

    # -- the world --------------------------------------------------------

    def _look(self) -> None:
        origin = (self.position[0] - 4, self.position[1] - 4)
        for local_y in range(9):
            for local_x in range(10):
                tile = (origin[0] + local_x, origin[1] + local_y)
                if not (0 <= tile[0] < self.width and 0 <= tile[1] < self.height):
                    continue
                self.seen.add(tile)
                if tile in self.walkable:
                    self.known.add(tile)

    def snapshot(self) -> dict:
        origin = (self.position[0] - 4, self.position[1] - 4)
        return {
            "map_name": self.map_name,
            "map_id": self.map_id,
            "player_position": list(self.position),
            "facing": "down",
            "window_top_left": {"x": origin[0], "y": origin[1]},
            "terrain": [
                [1 if (origin[0] + lx, origin[1] + ly) in self.walkable else 0 for lx in range(10)]
                for ly in range(9)
            ],
            "map_dimensions": {"width": self.width, "height": self.height},
            "warps": [{"x": x, "y": y} for x, y in self.warps],
            "sprites": [],
        }

    def bundle(self) -> dict:
        return {
            "state": {
                "player": {"position": list(self.position), "facing": "down"},
                "map": {"map_name": self.map_name, "map_id": self.map_id},
            },
            "navigation": {"snapshot": self.snapshot()},
        }

    def explored_grid(self, map_id):
        if not self.remember:
            return None
        return {
            "width": self.width,
            "height": self.height,
            "seen": set(self.seen),
            "walkable": set(self.known),
        }

    # -- the two coroutines walk_to needs ---------------------------------

    async def observe(self) -> dict:
        return capabilities.observation_from_bundle(self.bundle())

    async def act(self, actions: list[str]) -> dict:
        self.batches.append(list(actions))
        steps = {
            "walk_up": (0, -1),
            "walk_down": (0, 1),
            "walk_left": (-1, 0),
            "walk_right": (1, 0),
        }
        moved = 0
        blocked_after = None
        for index, action in enumerate(actions):
            dx, dy = steps[action]
            target = (self.position[0] + dx, self.position[1] + dy)
            if target not in self.walkable or target in self.refuses:
                blocked_after = index
                break
            self.position = target
            if target in self.warps:
                # Stepping onto a warp leaves the map. Nothing after it is walked.
                self.map_name, self.map_id = f"{self.map_name} (through)", self.map_id + 100
                moved += 1
                break
            self._look()
            moved += 1
        return {
            "actions_executed": len(actions),
            "outcome": {"moved": moved, "blocked_after": blocked_after},
            "bundle": self.bundle(),
        }

    def walk_to(self, **kwargs) -> dict:
        return asyncio.run(
            capabilities.walk_to(
                observe=self.observe,
                act=self.act,
                explored_grid=self.explored_grid,
                **kwargs,
            )
        )


# ---------------------------------------------------------------------------
# The contradiction itself
# ---------------------------------------------------------------------------


def test_the_store_believed_ground_route_4_does_not_have(route4):
    """The run's memory was not merely short of the east — it was wrong about it.

    Flooding the run's own store from (15, 11) walks out to x = 24 along a
    corridor at y = 5..9 that the map does not have; the emulator stops at
    x = 19. So the honest answer at that moment was not "walled off" either.
    It was "your memory says there is a way and it has not been looked at
    lately" — which is a thing you settle by walking at it, not by refusing.
    """
    collision = capabilities.collision_from(route4["snapshot"], route4["explored"])
    region = world_mod.reachable_region(collision, (15, 11))

    assert max(x for x, _ in region.order) == 24
    assert max(x for x, _ in route4["truth"]["reachable"]) == 19
    believed = set(region.order) - route4["truth"]["reachable"]
    assert (24, 6) in believed, "the store holds ground on the far side of the wall"
    assert not region.sealed, "so there is something to go and check"


def test_route_stops_promising_the_hop_it_cannot_check(route4):
    """`/route` still answers one hop east — and now says it checked no ground."""
    world = World.load()
    payload = capabilities.route_payload(world, "Route 4", "Cerulean City")

    assert payload["distance"] == 1
    assert payload["hops"][0]["edge"] == "east"
    assert payload["ground"] == "unchecked"
    assert "which maps touch, never which ground connects" in payload["basis"]
    assert "Route 4" in payload["caveat"] and "pockets of land" in payload["caveat"]


def test_goto_checks_the_belief_and_then_names_the_way_out(route4):
    """The whole answer, end to end, from the tile the two tools disagreed on.

    It walks the thirteen steps that settle the store's claim, the live window
    disproves it, and *then* it refuses — as walled off rather than unreachable,
    because the difference is whether walking at it would ever help. And it
    names the doors that are reachable, because "back into Mt. Moon" is the
    actual answer and nothing in the harness was saying it.
    """
    fake = FakeWorld(
        route4["truth"]["reachable"],
        (15, 11),
        width=90,
        height=18,
        map_name="Route 4",
        map_id=15,
        warps=((11, 5), (18, 5), (24, 5)),
    )
    fake.seen = set(route4["explored"]["seen"])
    fake.known = set(route4["explored"]["walkable"])

    result = fake.walk_to(world=World.load(), target_map="Cerulean City")

    assert result["arrived"] is False
    assert result["walked"] == 13, "it spent thirteen steps disproving its own map"
    assert fake.position == (19, 6), "the east end of the ground that really exists"

    onward = result["onward"]
    assert onward["kind"] == "walled-off"
    assert onward["sealed"] is True
    assert "Every tile bordering" in result["stopped_because"]
    assert result["stopped_because"].startswith("walked 13 toward Cerulean City, then ")

    reachable = {(exit_["to"], tuple(exit_["at"])) for exit_ in onward["exits"]}
    assert ("Mt Moon 1F", (18, 5)) in reachable
    assert ("Mt Moon Pokecenter", (11, 5)) in reachable
    assert ("Route 3", (11, 17)) in reachable
    # The Mt. Moon *exit* warp is on the far side of the wall, so it is not offered.
    assert (24, 5) not in {tuple(exit_["at"]) for exit_ in onward["exits"]}
    assert "Mt Moon 1F at [18, 5], 2 steps" in result["stopped_because"]


# ---------------------------------------------------------------------------
# Unlooked-at ground is not a wall
# ---------------------------------------------------------------------------


def _corridor(length: int) -> set[tuple[int, int]]:
    return {(x, 5) for x in range(length)}


def test_goto_walks_to_the_edge_of_what_it_has_seen_and_says_so():
    """A corridor running east past the window: unknown, not walled off.

    The store holds nothing beyond the window, so BFS has no path east — which
    is a different thing from there being no way east, and answering the second
    is what taught the agent that a whole direction was closed.
    """
    fake = FakeWorld(_corridor(40), (2, 5), width=40, height=11)

    result = fake.walk_to(world=World({}), target_xy=(30, 5))

    assert result["walked"] > 0
    assert result["arrived"] is False
    onward = result["onward"]
    assert onward["kind"] == "unexplored"
    # It stopped on ground it has seen, and names the first tile it has not.
    assert onward["unseen_at"][0] == onward["heading_for"][0] + 1
    assert fake.position == tuple(onward["stopped_at"])
    assert onward["stopped_at"][0] > 2, "it walked east rather than staying put"
    assert "edge of what has been seen and not a wall" in result["stopped_because"]
    assert "Look, then ask again" in result["stopped_because"]


def test_asking_again_gets_further_each_time():
    """One look per call, and each call starts where the last one stopped."""
    fake = FakeWorld(_corridor(40), (2, 5), width=40, height=11)

    first = fake.walk_to(world=World({}), target_xy=(30, 5))
    second = fake.walk_to(world=World({}), target_xy=(30, 5))

    assert second["walked"] > 0
    assert second["onward"]["stopped_at"][0] > first["onward"]["stopped_at"][0]


def test_goto_reaches_a_tile_it_used_to_refuse():
    """The payoff. Looking is what turns the dead end into the rest of the path.

    Ten tiles east, five of them never looked at. The old answer was "no
    walkable path", walked 0; walking to the edge of knowledge puts the target
    inside the window, and the same call finishes the job.
    """
    fake = FakeWorld(_corridor(40), (20, 5), width=40, height=11)
    fake.seen = {tile for tile in fake.seen if tile[0] >= 20}
    fake.known = {tile for tile in fake.known if tile[0] >= 20}

    result = fake.walk_to(world=World({}), target_xy=(30, 5))

    assert result["arrived"] is True
    assert result["onward"] is None
    assert fake.position == (30, 5)


def test_goto_never_walks_backwards_to_look():
    """Unseen ground behind you is not a step toward anything in front."""
    fake = FakeWorld(_corridor(40), (20, 5), width=40, height=11)
    # A five-tile island of memory: unlooked-at ground lies both ways.
    fake.seen = {tile for tile in fake.seen if 18 <= tile[0] <= 22}
    fake.known = {tile for tile in fake.known if 18 <= tile[0] <= 22}

    fake.walk_to(world=World({}), target_xy=(35, 5))

    assert fake.batches
    assert set(fake.batches[0]) == {"walk_right"}


def test_a_walled_tile_target_still_refuses_and_does_not_move():
    """The guarantee that must not loosen: no path invented through rock."""
    walls = {(x, 5) for x in range(11)} | {(x, 5) for x in range(13, 20)}
    fake = FakeWorld(walls, (5, 5), width=20, height=11)
    fake._look()
    fake.seen |= {(x, y) for x in range(20) for y in range(11)}

    result = fake.walk_to(world=World({}), target_xy=(15, 5))

    assert result["walked"] == 0
    assert result["arrived"] is False
    assert "no walkable path" in result["stopped_because"]
    assert result["onward"]["kind"] == "walled-off"
    assert fake.batches == []


# ---------------------------------------------------------------------------
# Three ladders to the same floor
# ---------------------------------------------------------------------------


def test_the_real_mt_moon_map_picks_the_ladder_two_tiles_away(route4):
    """Not in miniature: the run's own Mt Moon 1F store, at the tile it gave up on.

    `/goto Mt Moon B1F` walked twelve, stopped at (17, 9) and said the ladder at
    (5, 5) — sixteen tiles off — was not reachable. It was standing two tiles
    above the ladder at (17, 11) the whole time.
    """
    store = route4["mt_moon_1f"]
    grid = {
        "width": store["width"],
        "height": store["height"],
        "seen": _unpack(store["seen"]),
        "walkable": _unpack(store["walkable"]),
    }
    stopped_at = tuple(store["stopped_at"])
    snapshot = {
        "map_name": "Mt Moon 1F",
        "map_id": 59,
        "player_position": list(stopped_at),
        "facing": "down",
        "terrain": [],  # no live window: the belief it planned on, alone
        "window_top_left": {"x": stopped_at[0] - 4, "y": stopped_at[1] - 4},
        "map_dimensions": {"width": store["width"], "height": store["height"]},
        "warps": [{"x": x, "y": y} for x, y in store["ladders"]],
        "sprites": [],
    }
    observation = {
        "map_name": "Mt Moon 1F",
        "map_id": 59,
        "position": stopped_at,
        "snapshot": snapshot,
        "bundle": {},
    }

    plan, stop = capabilities._leg_for_map(
        World.load(),
        observation,
        capabilities.collision_from(snapshot, grid),
        target_map="Mt Moon B1F",
        target_xy=None,
    )

    assert stop is None, "there was never anything wrong with this map"
    assert plan == ["walk_down", "walk_down"]


def _two_doors_world() -> World:
    """One map with two warps to the same place, the way Mt. Moon 1F has three."""
    hops = (
        Hop(from_map="Cave", to_map="Lower Cave", kind="warp", at=(1, 1), edge=None),
        Hop(from_map="Cave", to_map="Lower Cave", kind="warp", at=(6, 5), edge=None),
    )
    return World(
        {
            "Cave": MapInfo(name="Cave", map_id=1, size=(10, 10), hops=hops),
            "Lower Cave": MapInfo(name="Lower Cave", map_id=2, size=(10, 10), hops=()),
        }
    )


def test_goto_takes_the_nearest_reachable_door_not_the_first_one_listed():
    """Mt. Moon 1F, in miniature.

    The far door is walled off and the near one is two tiles away. `world.route`
    keeps one edge per map pair, so the caller used to be handed whichever the
    file listed first and would spend the whole call failing to reach it.
    """
    room = {(x, 5) for x in range(4, 8)} | {(6, 6)}
    fake = FakeWorld(room, (4, 5), width=10, height=10, map_name="Cave", warps=((1, 1), (6, 5)))
    fake.seen = {(x, y) for x in range(10) for y in range(10)}

    fake.walk_to(world=_two_doors_world(), target_map="Lower Cave")

    assert fake.batches, "it should have walked at the near door"
    assert fake.batches[0] == ["walk_right", "walk_right"]


def test_route_says_how_many_doors_make_the_same_crossing():
    payload = capabilities.route_payload(_two_doors_world(), "Cave", "Lower Cave")

    assert payload["hops"][0]["ways"] == 2
    assert payload["hops"][0]["at_any_of"] == [[1, 1], [6, 5]]


def test_the_refusal_counts_the_doors_it_tried_and_names_them_all():
    """Naming one tile is what made a search problem look like a map problem."""
    unreachable = World(
        {
            "Cave": MapInfo(
                name="Cave",
                map_id=1,
                size=(10, 10),
                hops=(
                    Hop("Cave", "Lower Cave", "warp", (1, 1), None),
                    Hop("Cave", "Lower Cave", "warp", (9, 9), None),
                ),
            ),
            "Lower Cave": MapInfo(name="Lower Cave", map_id=2, size=(10, 10), hops=()),
        }
    )
    room = {(x, 5) for x in range(4, 8)}
    fake = FakeWorld(room, (4, 5), width=10, height=10, map_name="Cave", warps=((1, 1), (9, 9)))
    fake.seen = {(x, y) for x in range(10) for y in range(10)}

    result = fake.walk_to(world=unreachable, target_map="Lower Cave")

    assert result["arrived"] is False
    assert result["onward"]["tried"] == 2
    assert "tried 2 way(s) in" in result["stopped_because"]
    assert "[1, 1], [9, 9]" in result["stopped_because"]


def test_partial_progress_is_reported_as_progress():
    """ "Walked eight, then stopped" is not the same event as "not reachable"."""
    fake = FakeWorld(_corridor(40), (2, 5), width=40, height=11)

    result = fake.walk_to(world=World({}), target_xy=(30, 5))

    assert result["walked"] > 0
    assert result["stopped_because"].startswith(f"walked {result['walked']} toward [30, 5], then ")


# ---------------------------------------------------------------------------
# A plan that failed is not proposed again
# ---------------------------------------------------------------------------


def _knows_the_whole_floor(fake: FakeWorld) -> None:
    """Every tile looked at and every walkable one remembered — a decoded floor.

    This is what `mapdecode` now hands the planner on any map the player is
    standing on, and it is the state requirement three is about: there is no
    unseen ground left, so "walk at the unknown and find out" is not an answer
    and the flood either reaches the tile or it does not.
    """
    fake.seen = {(x, y) for x in range(fake.width) for y in range(fake.height)}
    fake.known = set(fake.walkable)


def _ring(width: int, height: int) -> set[tuple[int, int]]:
    """A rectangular loop of corridor: two ways round between any two tiles."""
    return (
        {(x, 0) for x in range(width)}
        | {(x, height - 1) for x in range(width)}
        | {(0, y) for y in range(height)}
        | {(width - 1, y) for y in range(height)}
    )


def test_a_plan_the_game_refused_is_not_proposed_again():
    """The blocked tile is remembered for the rest of the call, and routed around.

    An NPC standing in a corridor reads as open ground in every snapshot, so
    replanning from scratch produces the identical shortest path, walks into
    the identical NPC and reports the identical refusal. Measured on Mt. Moon
    1F: forty presses a round, at (30, 7), for as many rounds as anything kept
    asking — and every one of those presses landed on ground already walked.

    Here the ring gives a second way round, and one call is enough to find it.
    """
    fake = FakeWorld(_ring(7, 7), (0, 3), width=7, height=7)
    _knows_the_whole_floor(fake)
    fake.refuses = {(0, 1)}  # somebody is standing on the short way north

    result = fake.walk_to(world=World({}), target_xy=(0, 0))

    assert result["arrived"] is True, result["stopped_because"]
    assert fake.position == (0, 0)
    assert len(fake.batches) >= 2, "the first plan was refused, so there was a second"
    assert fake.batches[0] != fake.batches[1], "and it was not the same plan again"
    # Three tiles the short way, twenty-three the long way. It took the long way,
    # which is the only way, rather than the short one over and over.
    assert result["walked"] >= 20


def test_a_refusal_with_no_way_round_is_still_a_refusal():
    """Memory must not turn a wall into wandering. One try, then the same no.

    The guarantee that has to survive every change here: a refusal names what
    IS reachable, and nothing replaces it with a worse plan.
    """
    fake = FakeWorld(_corridor(20), (5, 5), width=20, height=11)
    _knows_the_whole_floor(fake)
    fake.refuses = {(6, 5)}  # the corridor is one wide, so this is the whole east

    result = fake.walk_to(world=World({}), target_xy=(15, 5))

    assert result["arrived"] is False
    assert len(fake.batches) == 1, "tried once, learned, and did not try it again"
    assert result["onward"]["kind"] == "walled-off"
    assert "no walkable path" in result["stopped_because"]
    assert f"tiles are reachable from {list(fake.position)}" in result["stopped_because"]


# ---------------------------------------------------------------------------
# A route that has to start by going the wrong way
# ---------------------------------------------------------------------------


#: A hook. The goal is eighteen tiles WEST; the only route runs east, south,
#: and all the way back west along the bottom — and the one piece of unseen
#: ground the player can reach is south, which is further from the goal than
#: where they are standing.
#:
#: The live window is ten by nine, so from (20, 10) the whole visible pocket is
#: the corridor x = 18..24 and the first four tiles of the leg going south. The
#: rest is ground nobody has looked at, and the only way to any of it is the
#: wrong way.
HOOK = (
    {(x, 10) for x in range(18, 25)}  # the visible corridor
    | {(24, y) for y in range(10, 21)}  # south, off the bottom of the window
    | {(x, 20) for x in range(2, 25)}  # the long way west
    | {(2, y) for y in range(10, 21)}  # and back north to the goal at (2, 10)
)


def test_a_route_that_must_start_by_walking_away_from_the_goal_is_found():
    """Greedy-on-Manhattan cannot take the first step of this route, ever.

    The goal is due west and the only unseen ground reachable from the start is
    due south — five tiles further from the goal than the tile the player is
    standing on. The old scorer asked "does this frontier shrink the distance to
    the goal" and dropped everything answering no, which left nothing to choose
    from: it refused, walked nothing, and refused again the next call and the
    next. That is hill climbing, and a maze is the shape hill climbing cannot
    cross. Cost so far plus distance still to go has no such blind spot — the
    only candidate is the one it picks, whichever way it lies.
    """
    fake = FakeWorld(HOOK, (20, 10), width=40, height=21)

    result = fake.walk_to(world=World({}), target_xy=(2, 10))

    assert fake.batches, "the old scorer had nothing to pick and so walked nothing"
    assert fake.batches[0][0] == "walk_right", "the first step of the only route goes backwards"
    assert result["walked"] > 0

    # And the rest of it, one look per call, until the goal is in the window.
    for _ in range(30):
        if result["arrived"]:
            break
        result = fake.walk_to(world=World({}), target_xy=(2, 10))
    assert result["arrived"] is True, result["stopped_because"]
    assert fake.position == (2, 10)


def test_the_unseen_ground_it_picks_is_the_one_with_least_journey_left():
    """Cost so far plus distance still to go — A*'s estimate, not the last step's gain.

    Both ends of this corridor run out of the window, so both are frontier.
    The goal is east, so the east end wins: not because walking east is
    progress, but because the whole journey through it is shorter.
    """
    fake = FakeWorld(_corridor(40), (20, 5), width=40, height=11)
    fake.seen = {tile for tile in fake.seen if 18 <= tile[0] <= 22}
    fake.known = {tile for tile in fake.known if 18 <= tile[0] <= 22}

    fake.walk_to(world=World({}), target_xy=(35, 5))

    assert set(fake.batches[0]) == {"walk_right"}


# ---------------------------------------------------------------------------
# Nothing routes through a ladder on the way past
# ---------------------------------------------------------------------------


def test_the_walk_does_not_cross_a_ladder_it_was_not_aiming_at():
    """A warp in the corridor is the end of the corridor, not a tile in it.

    Mt. Moon 1F, from the south entrance to the ladder at (5, 5): the shortest
    walk is 89 steps and step 72 lands on the *other* ladder, at (17, 11).
    Walked, that spent 72 presses and ended on B1F, after which every /goto
    asked for (5, 5) on a floor that has no such tile and refused, forever, at
    no cost and no progress. Measured: 80 presses, wrong floor, then a stalemate.
    """
    fake = FakeWorld(_corridor(20), (2, 5), width=20, height=11, warps=((10, 5),))
    _knows_the_whole_floor(fake)

    result = fake.walk_to(world=World({}), target_xy=(15, 5))

    assert result["arrived"] is False
    assert fake.position == (2, 5), "it did not set off down a corridor that ends in a hole"
    assert result["onward"]["kind"] == "walled-off"


def test_a_ladder_you_meant_to_reach_is_still_reachable():
    """Absorbing is not forbidden: a cave ladder is usually the whole point."""
    fake = FakeWorld(_corridor(20), (2, 5), width=20, height=11, warps=((10, 5),))
    _knows_the_whole_floor(fake)

    result = fake.walk_to(world=World({}), target_xy=(10, 5))

    assert fake.position == (10, 5)
    assert result["arrived"] is True
    assert fake.map_name != "Test Map", "and stepping onto it took the ladder down"
