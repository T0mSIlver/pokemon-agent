"""The service layer behind /route, /goto, /calc, /frontier, /sim, /guide and /progress.

Five finished modules already know how to answer these questions — `world`
routes and simulates, `gamedata` does damage, `guides` retrieves, `milestones`
scores. None of them can be imported by the agent's CLI, which is stdlib-only
and staged standalone, so the answers have to travel over HTTP.

Everything here is a plain function over data that was handed in. No FastAPI, no
globals, no emulator: a route function validates, calls one of these, and
translates :class:`CapabilityError` into a status code. The one exception is
:func:`walk_to`, which needs to interleave observing and acting — it takes those
two as coroutines rather than reaching for the emulator itself.
"""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Collection, Optional, Sequence

from pokemon_agent import gamedata, guides
from pokemon_agent import world as world_mod
from pokemon_agent.coordinator import (
    MAX_ACTIONS_PER_BATCH,
    MAX_FRAMES_PER_BATCH,
    batch_within_budget,
    frames_for_action,
)
from pokemon_agent.milestones import MILESTONE_DAG, MILESTONES_BY_ID
from pokemon_agent.pathfinding import DIRECTIONS

Coord = tuple[int, int]


class CapabilityError(Exception):
    """A service refusal that carries the status the route should answer with."""

    status = 400

    def __init__(self, detail: str, status: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status is not None:
            self.status = status


class NotFound(CapabilityError):
    status = 404


class Conflict(CapabilityError):
    status = 409


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------


def _coord(value: Any) -> Optional[Coord]:
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
    else:
        return None
    try:
        return int(x), int(y)
    except (TypeError, ValueError):
        return None


def _all_tiles(truth: dict) -> set[Coord]:
    """Every coordinate on a decoded map, walkable or not.

    A fully decoded floor has no unexplored ground, so `seen` covers all of it.
    That is what lets a refusal say "this is a wall" rather than "nobody has
    looked", which are different answers and were indistinguishable before.
    """
    width, height = int(truth.get("width") or 0), int(truth.get("height") or 0)
    return {(x, y) for y in range(height) for x in range(width)}


def collision_from(snapshot: dict, explored: Optional[dict] = None) -> dict:
    """Live walkability for the current map, in absolute coordinates.

    The live navigation window is only 10x9 tiles — enough to see the next step
    and nothing else — so what the map store has already learned fills in the
    rest. **Precedence is absolute: inside the window the live frame decides,
    including its negatives.** A tile the store calls walkable and the frame
    calls solid is solid; the store is memory, and it has held tiles nobody can
    stand on — the mid-air tile a ledge jump was once sampled on. Beyond the
    window there is nothing to check against, so those tiles travel as `live`
    minus themselves: present in `walkable`, absent from `live`, and every
    answer built on them says so.

    Sprites come from the live window only, because an NPC that stood somewhere
    once is not standing there now.

    `seen` is the third set and the one that was missing. The store records
    every tile a window covered, solid ones included, and dropping that on the
    way in left every consumer unable to tell a wall it has looked at from
    ground nobody has ever been shown. Both read as "not in `walkable`", and
    answering "unreachable" for the second is how a route that had never been
    looked at got reported as a route that does not exist.

    All of that describes the world before `map_terrain`. When the snapshot
    carries the decoded floor it *replaces* the store rather than adding to it,
    because the store's defect was never coverage, it was that it could not
    forget: "a tile seen passable once stays passable", so a tile corrected as
    solid inside the window was walkable again the moment it left. Ground truth
    has no such problem, and it makes every tile `seen`, so nothing downstream
    has to reason about unexplored ground on a map that is fully known.
    """
    truth = snapshot.get("map_terrain") or {}
    ground_truth: set[Coord] = {
        found for found in (_coord(item) for item in truth.get("walkable") or ()) if found
    }

    if ground_truth:
        walkable, seen = set(ground_truth), _all_tiles(truth)
    else:
        walkable = set(explored.get("walkable") or ()) if explored else set()
        seen = set(explored.get("seen") or ()) if explored else set()

    live: set[Coord] = set()
    origin = _coord(snapshot.get("window_top_left")) or (0, 0)
    for local_y, row in enumerate(snapshot.get("terrain") or []):
        for local_x, tile in enumerate(row):
            coord = (origin[0] + local_x, origin[1] + local_y)
            live.add(coord)
            if tile:
                # The frame still knows things the blockset does not: a door or a
                # warp carpet is walkable without being in the collision list.
                walkable.add(coord)
            elif not ground_truth:
                walkable.discard(coord)
            # With ground truth in hand a blocked frame tile is not evidence about
            # terrain. The frame blocks whatever a sprite is standing on, and an
            # NPC is not a wall -- deleting the tile is how a trainer standing in
            # a corridor became a permanent hole in the map. Sprites travel in
            # their own field, where a caller can treat them as the transient
            # thing they are.

    # A ledge's landing is two tiles away because the tile between is one no
    # player can stand on. That holds wherever the ledge was learned, so it
    # outranks a store that remembers standing there.
    #
    # Read from the decoded floor when there is one, because the snapshot only
    # carries the window's tile ids and ledges and seams are map-wide facts.
    # Route 4 has 169 ledge hops and Mt Moon 1F has 131 uncrossable seams; from
    # the window alone the planner saw a handful and read the rest as open
    # corridor. Route 4's east half is reachable *only* over one-way ledges, so
    # with them switched off the road to Cerulean looks like a wall.
    ledges = world_mod.ledge_edges(snapshot)
    if truth.get("tile_ids"):
        map_wide = world_mod.movement_edges(
            {
                "tileset": truth.get("tileset"),
                "tile_ids": truth["tile_ids"],
            }
        )
        map_wide.update(ledges)  # the live frame still wins where they overlap
        map_wide.blocked_pairs |= ledges.blocked_pairs
        map_wide.warp_exits |= ledges.warp_exits
        ledges = map_wide
    for (start, _direction), landing in ledges.items():
        walkable.discard(((start[0] + landing[0]) // 2, (start[1] + landing[1]) // 2))

    dimensions = snapshot.get("map_dimensions") or {}
    width = int(truth.get("width") or dimensions.get("width") or 0)
    height = int(truth.get("height") or dimensions.get("height") or 0)
    if not width or not height:
        width = max((x for x, _ in walkable), default=-1) + 1
        height = max((y for _, y in walkable), default=-1) + 1

    sprites = [found for found in (_coord(item) for item in snapshot.get("sprites") or ()) if found]
    # The window only lists the warps it can see, and the flood needs every one
    # on the floor: a route that crosses an unseen ladder is walked, not
    # planned around. See `world._Grid.is_absorbing`.
    warps = set(warp_coords(snapshot))
    for warp in truth.get("warps") or ():
        found = _coord(warp.get("at") if isinstance(warp, dict) else None) or _coord(warp)
        if found is not None:
            warps.add(found)
    return {
        "width": width,
        "height": height,
        "walkable": walkable,
        "sprites": sprites,
        "warps": sorted(warps),
        "live": live,
        # A tile is looked-at if this frame shows it, if the store wrote it
        # down either way, or if we believe it is ground — you cannot hold a
        # belief about ground nobody ever saw.
        "seen": seen | live | walkable,
        "ledges": ledges,
        # Whether this grid is the decoded floor or an accumulation of windows.
        # Every refusal built on it says which, because "unreachable" from ground
        # truth is a fact and "unreachable" from a mosaic of screens is a guess.
        "ground_truth": bool(ground_truth),
        "tile_ids": world_mod.tile_id_map(truth.get("tile_ids")),
    }


def collision_basis(collision: dict) -> str:
    """One line naming what an answer over this collision map is built from.

    Every consumer prints this, because "reachable" means two different things
    depending on which half of the map it came from, and the agent has no other
    way to tell them apart.
    """
    live = set(collision.get("live") or ())
    remembered = len([tile for tile in collision.get("walkable") or () if tile not in live])
    ledges = len(collision.get("ledges") or ())
    ledge_note = f", {ledges} one-way ledge jump(s) in view" if ledges else ""
    if collision.get("ground_truth"):
        walkable = len(collision.get("walkable") or ())
        return (
            f"this floor's real terrain, all {walkable} walkable tiles of it, decoded "
            f"from the game's own map data (fact, not a guess from what has been "
            f"walked){ledge_note}"
        )
    if not remembered:
        return f"the live {len(live)}-tile window only{ledge_note}"
    return (
        f"the live {len(live)}-tile window (fact, and it overrides the store where they "
        f"overlap) plus {remembered} remembered tiles from the explored map "
        f"(belief — nobody has looked at them this frame){ledge_note}"
    )


def warp_coords(snapshot: dict) -> list[Coord]:
    coords = []
    for warp in snapshot.get("warps") or ():
        found = _coord(warp.get("coord") if isinstance(warp, dict) else None) or _coord(warp)
        if found is not None:
            coords.append(found)
    return coords


def observation_from_bundle(bundle: Optional[dict]) -> dict:
    """The few facts the walking loop needs, pulled out of a runtime bundle."""
    bundle = bundle or {}
    state = bundle.get("state") or {}
    snapshot = ((bundle.get("navigation") or {}).get("snapshot")) or {}
    player = state.get("player") or {}
    position = _coord(player.get("position")) or _coord(snapshot.get("player_position"))
    map_info = state.get("map") or {}
    return {
        "map_name": map_info.get("map_name") or snapshot.get("map_name"),
        "map_id": map_info.get("map_id", snapshot.get("map_id")),
        "position": position,
        "facing": player.get("facing") or snapshot.get("facing"),
        "snapshot": snapshot,
        "bundle": bundle,
    }


# ---------------------------------------------------------------------------
# /route
# ---------------------------------------------------------------------------


def known_map_names(world: world_mod.World) -> dict[str, str]:
    """Every routable map name, keyed by its lowercase form.

    The graph is allowed to be a subgraph: a map named only as a hop target is
    a legal destination even without a record of its own, so targets count too.
    """
    known: dict[str, str] = {}
    for name in world.map_names():
        known.setdefault(name.lower(), name)
        for hop in world.neighbours(name):
            known.setdefault(hop.to_map.lower(), hop.to_map)
    return known


def canonical_map_name(world: world_mod.World, name: Optional[str]) -> str:
    """Match *name* against the graph, ignoring case. Raises 404 if unknown."""
    if not name or not str(name).strip():
        raise NotFound("No map named.")
    found = known_map_names(world).get(str(name).strip().lower())
    if found is None:
        raise NotFound(
            f"No map called {name!r}. Names come from the game's own map table, "
            "for example 'Pewter City' or 'Route 3'."
        )
    return found


#: What every hop in a /route answer is made of, and what none of it is made
#: of. Printed on the payload because the alternative — leaving /goto to be the
#: only one that mentions ground — is what made a one-hop route read as a
#: promise.
ROUTE_BASIS = (
    "the game's own connection and warp tables: which maps touch, never which "
    "ground connects. No tile on any of these maps has been looked at to answer this."
)

#: The failure mode this caveat exists for is not hypothetical. Route 4 holds
#: the Mt. Moon mouth and the Cerulean half as two pockets of land that never
#: join up, so `Route 4 -> Cerulean City, distance 1` is a true statement about
#: the map table and a false one about walking, and only /goto can tell which.
ROUTE_CAVEAT = (
    "Every hop here is unchecked ground. The graph knows two maps touch; it does "
    "not know that the tile you are standing on can reach the seam. One map can "
    "hold several pockets of land that never join up — Route 4's Mt. Moon mouth "
    "and its Cerulean half are two of them — so even a one-hop route can be "
    "unwalkable from where you are. /goto is what finds out, and it answers with "
    "which of the two it hit: ground that was looked at and is solid, or ground "
    "nobody has looked at yet."
)


#: What a pocket route is built from, in one line the agent can read.
POCKET_BASIS = (
    "the real terrain of every map on the way, decoded from the game's own map data, "
    "so each hop names a tile you can actually reach from where you are standing"
)


def _pocket_route_payload(src: str, dst: str, hops: Sequence[Any]) -> dict:
    """A pocket route, rendered the way `route_payload` renders a map route.

    `ground` is "checked" here and "unchecked" below, and the difference is
    real: these hops were found by flooding terrain the game itself supplied,
    and the ones below were found in a static table that has never seen a tile.
    """
    return {
        "from": src,
        "to": dst,
        "distance": len(hops),
        "hops": [
            {
                "from": hop.from_map,
                "to": hop.to_map,
                "kind": hop.kind,
                "at": list(hop.at) if hop.at else None,
                "landing": list(hop.landing) if hop.landing else None,
                "edge": hop.edge,
                "describe": hop.describe(),
            }
            for hop in hops
        ],
        "ground": "checked",
        "basis": POCKET_BASIS,
    }


def route_payload(
    world: world_mod.World,
    source: str,
    target: str,
    *,
    pockets: Optional[Any] = None,
    at: Optional[Coord] = None,
) -> dict:
    """Hops from *source* to *target*, or a reason there are none.

    Hops, not button presses: which buttons cross a map depends on that map's
    live collision, which no static file carries. Every hop carries
    ``ground: "unchecked"`` for the same reason — this function has never seen
    a tile, and a payload that does not say so is read as a guarantee.

    Given a `pockets` graph and the tile the player is on, the answer comes from
    real terrain instead and says ``ground: "checked"``.
    """
    src = canonical_map_name(world, source)
    dst = canonical_map_name(world, target)
    if src == dst:
        return {"from": src, "to": dst, "distance": 0, "hops": [], "basis": ROUTE_BASIS}

    # The pocket router first, when the caller knows where the player is and the
    # store has decoded terrain. It is strictly better informed: it searches
    # (map, pocket) rather than map, so it can express a route that comes back
    # to a map it has already been on -- which is what crossing Mt Moon is, and
    # what the map-keyed search below reports as "no route" rather than finding
    # a worse one.
    if pockets is not None and at is not None:
        found = pockets.route(src, at, dst)
        if found:
            return _pocket_route_payload(src, dst, found)
        if found == ():
            return {"from": src, "to": dst, "distance": 0, "hops": [], "basis": POCKET_BASIS}

    hops = world.route(src, dst)
    if hops is None:
        return {
            "from": src,
            "to": dst,
            "distance": None,
            "hops": None,
            "basis": ROUTE_BASIS,
            "reason": (
                f"No route from {src} to {dst} in the map graph. Either they are not "
                "connected by warps and edges, or the leg between them is not in the "
                "generated world data."
            ),
        }
    return {
        "from": src,
        "to": dst,
        "distance": len(hops),
        "hops": [_route_hop(world, hop) for hop in hops],
        # Not a hedge: the honest state of every hop above. Nothing here was
        # checked against a tile, and /goto is the verb that checks.
        "ground": "unchecked",
        "basis": ROUTE_BASIS,
        "caveat": ROUTE_CAVEAT,
    }


def _route_hop(world: world_mod.World, hop: world_mod.Hop) -> dict:
    """One hop, plus the one thing about it the graph does know and dropped.

    ``ways`` counts the doors that make the same crossing. Mt. Moon 1F has
    three ladders down to B1F and the search keeps whichever the file listed
    first; a reader shown that one tile will walk at that one tile until it
    gives up, which is what happened two tiles from a ladder that works.
    """
    ways = world.hops_between(hop.from_map, hop.to_map)
    payload = {
        "from": hop.from_map,
        "to": hop.to_map,
        "kind": hop.kind,
        "at": list(hop.at) if hop.at is not None else None,
        "edge": hop.edge,
    }
    if len(ways) > 1:
        payload["ways"] = len(ways)
        payload["at_any_of"] = [list(way.at) for way in ways if way.at is not None]
    return payload


# ---------------------------------------------------------------------------
# /goto
# ---------------------------------------------------------------------------

#: How many plan-walk-replan rounds one /goto may take before giving up. A round
#: that makes no progress stops immediately, so this only bounds a route that
#: keeps finding new ground to cross.
MAX_GOTO_ROUNDS = 24

_EDGE_STEPS = {"north": "up", "south": "down", "west": "left", "east": "right"}


def _same_map(left: Optional[str], right: Optional[str]) -> bool:
    return bool(left) and bool(right) and str(left).strip().lower() == str(right).strip().lower()


def _walk_actions(directions: Sequence[str]) -> list[str]:
    return [f"walk_{direction}" for direction in directions]


def plan_within(collision, start: Coord, goal: Coord) -> Optional[list[str]]:
    """Walk actions from *start* to *goal*, entering it from a neighbour if need be.

    A door reads as blocked collision — you walk *into* it, you do not stand on
    it — so a goal that is not walkable is approached from whichever neighbour
    is, and the final step into it is appended.
    """
    direct = world_mod.path_within(collision, start, goal)
    if direct is not None:
        return list(direct)
    for direction, (dx, dy) in DIRECTIONS.items():
        approach = (goal[0] - dx, goal[1] - dy)
        leg = world_mod.path_within(collision, start, approach)
        if leg is None:
            continue
        return [*leg, f"walk_{direction}"]
    return None


def _edge_tiles(collision: dict, edge: str) -> list[Coord]:
    """Every tile along one side of the map, walkable or not."""
    width, height = collision["width"], collision["height"]
    if edge == "north":
        return [(x, 0) for x in range(width)]
    if edge == "south":
        return [(x, height - 1) for x in range(width)]
    if edge == "west":
        return [(0, y) for y in range(height)]
    if edge == "east":
        return [(width - 1, y) for y in range(height)]
    return []


def _cheapest(
    region: world_mod.Region,
    candidates: Sequence[Coord],
    *,
    enter: bool = True,
) -> Optional[tuple[Coord, list[str], int]]:
    """The candidate this region reaches for the fewest steps, and how.

    Cheapest by steps actually walked, over one flood, across every candidate.
    The edge search this replaces sorted by straight-line distance and gave up
    after twelve tries, which on a ninety-tile-wide map means the twelve tiles
    nearest as the crow flies rather than the ones you can get to.

    `enter` walks *into* the goal from a neighbour when the goal itself is not
    standable, which is what a door is. Edges are not doors: you have to be
    standing on the boundary tile before the map will hand you to the next one.
    """
    best: Optional[Coord] = None
    best_cost: Optional[int] = None
    for candidate in candidates:
        cost = region.steps_to(candidate)
        if cost is None and enter:
            costs = [
                found + 1
                for found in (
                    region.steps_to((candidate[0] - dx, candidate[1] - dy))
                    for dx, dy in DIRECTIONS.values()
                )
                if found is not None
            ]
            cost = min(costs) if costs else None
        if cost is None or (best_cost is not None and cost >= best_cost):
            continue
        best, best_cost = candidate, cost
    if best is None or best_cost is None:
        return None
    # One path built, at the end, rather than one per candidate: an edge is up
    # to `width` tiles long and this runs on every replanning round.
    if enter:
        found = region.approach(best)
        if found is None:  # pragma: no cover - a cost implies a path
            return None
        return best, list(found[0]), found[1]
    actions = region.actions_to(best)
    if actions is None:  # pragma: no cover - a cost implies a path
        return None
    return best, list(actions), best_cost


def _remaining(there: Coord, *, toward: object, bounds: Optional[dict] = None) -> int:
    """How far *there* still is from what `toward` means, in tiles.

    A compass edge is a distance along one axis; a tile is Manhattan distance
    to it. Neither is a promise about walking — nothing on a maze is — but it
    orders candidates by how much of the journey is left rather than by how
    much of it the last step covered, which is the whole difference between
    A* and hill climbing.
    """
    if isinstance(toward, str):
        width = int((bounds or {}).get("width") or 0)
        height = int((bounds or {}).get("height") or 0)
        if toward == "north":
            return there[1]
        if toward == "south":
            return max(0, height - 1 - there[1])
        if toward == "west":
            return there[0]
        if toward == "east":
            return max(0, width - 1 - there[0])
        return 0
    goal = _coord(toward)
    if goal is None:
        return 0
    return abs(there[0] - goal[0]) + abs(there[1] - goal[1])


def _toward_the_unseen(
    region: world_mod.Region,
    toward: object,
    *,
    bounds: Optional[dict] = None,
    spent: Collection[Coord] = (),
) -> Optional[tuple[Coord, Coord, list[str]]]:
    """Where to stand to look at the unseen ground most likely to lead onward.

    Returns ``(stand here, the unseen tile beyond, how to get there)``, chosen
    by ``steps walked + tiles still to go`` — what it costs to get there plus
    what is left after. That is A*'s estimate of a whole journey, and picking
    the smallest of it is what lets the answer be a frontier the goal is
    *behind*.

    It used to be ``gain = closer than where I stand?`` with everything scoring
    zero or less thrown away, which is hill climbing on Manhattan distance and
    cannot leave a local minimum by construction. On a maze whose only route
    starts by going the wrong way there was nothing left to choose from, so the
    same short plan came back every round and the same ground got walked again:
    measured at 64% of all presses landing on tiles already walked.

    `spent` is the frontier this call has already gone and looked at. Standing
    on the edge of knowledge and finding it did not help is an answer; going
    back to look at it a second time is not.
    """
    best: Optional[tuple[Coord, Coord, list[str]]] = None
    best_key: Optional[tuple[int, int]] = None
    already = set(spent)
    for stand, unseen in region.edge_of_knowledge:
        if unseen in already or stand in already:
            continue
        steps = region.steps_to(stand)
        if steps is None:  # pragma: no cover - it came out of this region
            continue
        key = (steps + _remaining(unseen, toward=toward, bounds=bounds), steps)
        if best_key is not None and key >= best_key:
            continue
        actions = region.actions_to(stand)
        if actions is None:  # pragma: no cover - it came out of this region
            continue
        best_key, best = key, (stand, unseen, list(actions))
    return best


def _reachable_exits(
    world: world_mod.World,
    here: str,
    collision: dict,
    snapshot: dict,
    region: world_mod.Region,
    limit: int = 6,
) -> list[dict]:
    """The ways off this map that this region can actually get to, cheapest first.

    A refusal that only says "not that way" leaves the agent with nothing to
    do. This is the rest of the answer: from the Mt. Moon mouth on Route 4 the
    east edge is walled off, and these are the three doors and one edge that
    are not.
    """
    hops = world.neighbours(here)
    by_tile = {tuple(hop.at): hop for hop in hops if hop.at is not None}
    exits: list[dict] = []
    for coord in warp_coords(snapshot):
        found = region.approach(coord)
        if found is None:
            continue
        hop = by_tile.get(coord)
        exits.append(
            {
                "kind": "warp",
                "at": list(coord),
                "to": hop.to_map if hop else None,
                "steps": found[1],
            }
        )
    for hop in hops:
        if hop.kind != "connection" or not hop.edge:
            continue
        found = _cheapest(region, _edge_tiles(collision, hop.edge), enter=False)
        if found is None:
            continue
        exits.append(
            {
                "kind": "edge",
                "edge": hop.edge,
                "at": list(found[0]),
                "to": hop.to_map,
                "steps": found[2] + 1,
            }
        )
    exits.sort(key=lambda item: item["steps"])
    return exits[:limit]


def _describe_exits(exits: Sequence[dict]) -> str:
    parts = []
    for item in exits:
        where = f"at {item['at']}"
        if item["kind"] == "edge":
            where = f"off the {item['edge']} edge at {item['at']}"
        parts.append(f"{item.get('to') or 'somewhere unmapped'} {where}, {item['steps']} steps")
    return "; ".join(parts)


class _Stop:
    """Why the walk ended, in a shape and in a sentence.

    Both, because the two travel different distances: `onward` is the structure
    a caller can act on, and `reason` is what survives the trip through /goto's
    three-key HTTP answer.
    """

    def __init__(self, reason: str, onward: Optional[dict] = None) -> None:
        self.reason = reason
        self.onward = onward


def _walled_off(
    world: world_mod.World,
    goal: str,
    tried: int,
    observation: dict,
    collision: dict,
    region: world_mod.Region,
    *,
    lead: str,
) -> _Stop:
    """The refusal for ground that was looked at and is solid."""
    here = observation["map_name"]
    exits: list[dict] = []
    try:
        exits = _reachable_exits(
            world, canonical_map_name(world, here), collision, observation["snapshot"], region
        )
    except NotFound:
        pass
    onward = {
        "kind": "walled-off",
        "goal": goal,
        "tried": tried,
        "reachable_tiles": len(region.order),
        "sealed": region.sealed,
        "exits": exits,
    }
    sealed = (
        "Every tile bordering the ground you can reach has been looked at and is solid, "
        "so this is a wall and not a gap in the map"
        if region.sealed
        else "No unseen ground lies that way either, so there is nothing to walk at and find out"
    )
    onward_note = (
        f" What is reachable from here: {_describe_exits(exits)}."
        if exits
        else " Nothing on this map leads anywhere from here."
    )
    return _Stop(
        f"{lead} {sealed}. {len(region.order)} tiles are reachable from "
        f"{list(region.start)} on {here}.{onward_note}",
        onward,
    )


def _go_look(goal: str, stand: Coord, unseen: Coord, observation: dict, steps: int) -> _Stop:
    """The answer for ground nobody has looked at: go and look at it."""
    onward = {
        "kind": "unexplored",
        "goal": goal,
        "heading_for": list(stand),
        "unseen_at": list(unseen),
        "steps": steps,
    }
    return _Stop(_look_reason(onward, observation), onward)


def _look_reason(onward: dict, observation: dict) -> str:
    """The go-look sentence, rendered against wherever the player actually is.

    Rendered late on purpose. The plan that reaches the edge of knowledge can
    be cut short — a batch runs out, an NPC is standing in it — and a message
    written before the walk claims a tile the player never got to.
    """
    position = observation["position"]
    onward["stopped_at"] = list(position) if position is not None else None
    return (
        f"nothing known reaches {onward['goal']}, but this is the edge of what has been seen "
        f"and not a wall. Stopped at {onward['stopped_at']} on {observation['map_name']}; the "
        f"nearest unlooked-at ground that way is {onward['unseen_at']}, with "
        f"{onward['heading_for']} the last tile before it. Look, then ask again to keep going."
    )


def warp_step_direction(coord: Coord, dimensions: dict) -> Optional[str]:
    """Which way to walk to trigger a boundary warp you are already standing on."""
    width, height = dimensions.get("width"), dimensions.get("height")
    if coord[1] == 0:
        return "up"
    if height and coord[1] == height - 1:
        return "down"
    if coord[0] == 0:
        return "left"
    if width and coord[0] == width - 1:
        return "right"
    return None


def _leg_for_map(
    world: world_mod.World,
    observation: dict,
    collision: dict,
    *,
    target_map: Optional[str],
    target_xy: Optional[Coord],
    refused: Collection[Coord] = (),
    spent: Collection[Coord] = (),
) -> tuple[Optional[list[str]], Optional[_Stop]]:
    """The next batch of walk actions, or the reason there is not one.

    Returns ``(actions, None)`` to keep walking or ``(None, stop)`` to stop.
    Everything here runs off one flood: which of several doors is nearest,
    whether the ground that stopped us was looked at, and where the nearest
    unlooked-at ground in this direction is.

    `refused` and `spent` are this call's memory — ground the game would not
    let the player onto, and frontier the walk has already been to and looked
    at. Both are held out of the flood, so a plan that failed cannot be built
    a second time from the same facts. Without them the planner has no memory
    at all between rounds: it recomputes from scratch, gets the same shortest
    path, walks into the same NPC, and does it again.

    The reachable answer comes first and always. On a floor decoded out of
    WRAM there is no unseen ground at all, so "walk at the unknown and find
    out" is not an answer there — the flood either reaches the tile or the
    tile is behind a wall, and `_walled_off` says which.
    """
    position = observation["position"]
    snapshot = observation["snapshot"]
    region = world_mod.reachable_region(collision, position, refused=refused)

    if target_xy is not None:
        found = _cheapest(region, [target_xy])
        if found is not None:
            return found[1], None
        toward = _toward_the_unseen(region, target_xy, bounds=collision, spent=spent)
        if toward is not None:
            stand, unseen, plan = toward
            return plan, _go_look(str(list(target_xy)), stand, unseen, observation, len(plan))
        return None, _walled_off(
            world,
            str(list(target_xy)),
            1,
            observation,
            collision,
            region,
            lead=(
                f"no walkable path from {list(position)} to {list(target_xy)} on "
                f"{observation['map_name']}, checked against {collision_basis(collision)}."
            ),
        )

    current = observation["map_name"]
    try:
        here = canonical_map_name(world, current)
    except NotFound:
        return None, _Stop(
            f"{current!r} is not a map the route graph knows, so there is nothing to "
            "plan a hop from"
        )
    hops = world.route(here, target_map or "")
    if hops is None:
        return None, _Stop(f"no route from {current} to {target_map}")
    if not hops:
        return [], None
    next_map = hops[0].to_map
    # Every door to the same place, not the first one the file happened to list.
    ways = world.hops_between(here, next_map) or (hops[0],)

    warps = [way for way in ways if way.kind == "warp" and way.at is not None]
    if warps:
        for way in warps:
            if position == tuple(way.at):
                step = warp_step_direction(position, snapshot.get("map_dimensions") or {})
                if step is None:
                    return None, _Stop(
                        f"standing on the warp at {list(position)} on {current} but the map "
                        "did not change; this hop does not lead where the graph says it does"
                    )
                return [f"walk_{step}"], None
        found = _cheapest(region, [tuple(way.at) for way in warps])
        if found is not None:
            return found[1], None

    connections = [way for way in ways if way.kind == "connection" and way.edge]
    edge_goals: list[Coord] = []
    for way in connections:
        edge_goals.extend(_edge_tiles(collision, way.edge or ""))
    if connections:
        found = _cheapest(region, edge_goals, enter=False)
        if found is not None:
            goal, plan, _ = found
            step = _EDGE_STEPS[
                next(way.edge for way in connections if goal in _edge_tiles(collision, way.edge))
            ]
            return [*plan, f"walk_{step}"], None

    if not warps and not connections:
        return None, _Stop(f"hop from {current} to {next_map} has no direction to walk")

    goal_name = _hop_goal_name(current, next_map, warps, connections)
    toward: object
    if connections:
        toward = connections[0].edge
    else:
        toward = tuple(warps[0].at)
    look = _toward_the_unseen(region, toward, bounds=collision, spent=spent)
    if look is not None:
        stand, unseen, plan = look
        return plan, _go_look(goal_name, stand, unseen, observation, len(plan))
    return None, _walled_off(
        world,
        goal_name,
        len(warps) + len(connections),
        observation,
        collision,
        region,
        lead=(
            f"{goal_name} is not reachable on foot from {list(position)} on {current}; "
            f"tried {len(warps) + len(connections)} way(s) in. Checked against "
            f"{collision_basis(collision)}."
        ),
    )


def _hop_goal_name(
    current: str,
    next_map: str,
    warps: Sequence[world_mod.Hop],
    connections: Sequence[world_mod.Hop],
) -> str:
    if connections:
        edges = " or ".join(sorted({str(way.edge) for way in connections}))
        return f"the {edges} edge of {current} toward {next_map}"
    tiles = ", ".join(str(list(way.at)) for way in warps if way.at is not None)
    plural = "warps" if len(warps) > 1 else "warp"
    return f"the {plural} to {next_map} at {tiles}"


def _trail(collision: dict, start: Coord, actions: Sequence[str]) -> list[Coord]:
    """Every tile *actions* means to stand on, starting with *start*.

    Built over the same ledge table the plan was built over, so a jump counts
    as the two tiles it really crosses. Used to name the tile a batch was
    refused on: the plan is a path, the player stopped somewhere on it, and
    the next tile along is the one the game would not allow.
    """
    ledges = world_mod.movement_edges(collision)
    tiles = [start]
    here = start
    for action in actions:
        direction = str(action).replace("walk_", "")
        step = ledges.get((here, direction))
        if step is None:
            delta = DIRECTIONS.get(direction)
            if delta is None:
                break
            step = (here[0] + delta[0], here[1] + delta[1])
        tiles.append(step)
        here = step
    return tiles


def _refused_tile(trail: Sequence[Coord], stopped_at: Optional[Coord]) -> Optional[Coord]:
    """The tile a walk was refused on, or None if the trail does not explain it.

    The player stopped somewhere along the planned path; whatever came next on
    it is what the game would not let them onto. Answering None rather than
    guessing matters: a wrong tile taken out of the flood is a corridor closed
    for the rest of the trip.
    """
    if stopped_at is None:
        return None
    for index in range(len(trail) - 2, -1, -1):
        if trail[index] == stopped_at:
            return trail[index + 1]
    return None


async def walk_to(
    *,
    observe: Callable[[], Awaitable[dict]],
    act: Callable[[list[str]], Awaitable[dict]],
    world: world_mod.World,
    explored_grid: Callable[[Optional[int]], Optional[dict]],
    target_map: Optional[str] = None,
    target_xy: Optional[Coord] = None,
    frame_budget: int = MAX_FRAMES_PER_BATCH,
) -> dict:
    """Walk toward a map or a tile, re-planning on live collision each round.

    Every round reads the live window, plans inside the current map only, walks
    as much of that plan as the frame budget allows, and looks again. A hop is a
    plan, not a guarantee — Route 4 is one map whose halves are separated by
    Mt. Moon — so a plan that cannot be walked stops the whole call and says
    why, rather than grinding into rock.

    There are two ways a plan can fail to exist and this answers them
    differently. Ground that was looked at and is solid is a refusal, and it
    comes with the ways off this map that *are* reachable. Ground nobody has
    looked at is not a refusal: the walk goes to the last tile before the
    unseen ground, stops there, and says what to look at. One such hop per
    call — the point is to hand back a decision, not to wander.

    A plan that just failed is never proposed again inside one call. The tile
    the game refused is remembered for the rest of the walk and held out of
    every later flood, so the next round has to find a different way or admit
    there is not one. Before that memory existed the planner recomputed from
    scratch every round, produced the identical shortest path, walked into the
    identical NPC and reported the identical refusal — measured on Mt. Moon 1F
    at 40 presses per round, forever.

    Never a step onto ground that was not planned over: the frontier tile is
    reached across known walkable tiles like any other goal.
    """
    if target_map is None and target_xy is None:
        raise CapabilityError("Nothing to walk to: send a target map name or an x and y.")
    if target_map is not None:
        target_map = canonical_map_name(world, target_map)

    observation = await observe()
    if observation["position"] is None:
        raise Conflict(
            "The player has no position right now — probably mid-battle or mid-cutscene."
        )

    walked = 0
    executed = 0
    budget = max(0, int(frame_budget))
    stopped = "arrived"
    arrived = False
    onward: Optional[dict] = None
    #: The unseen tile this call set out to look at, once it has picked one.
    looking_at: Optional[Coord] = None
    #: Ground the game refused this call, and frontier it has already looked at.
    #: Both are the memory that stops a failed plan coming back unchanged.
    refused: set[Coord] = set()
    spent: set[Coord] = set()
    origin_map = observation["map_name"]

    for _ in range(MAX_GOTO_ROUNDS):
        if target_xy is not None and observation["position"] == target_xy:
            arrived = True
            break
        if target_map is not None and _same_map(observation["map_name"], target_map):
            arrived = True
            break
        if budget <= 0:
            stopped = f"frame budget of {frame_budget} frames spent"
            break

        collision = collision_from(observation["snapshot"], explored_grid(observation["map_id"]))
        plan, stop = _leg_for_map(
            world,
            observation,
            collision,
            target_map=target_map,
            target_xy=target_xy,
            refused=refused,
            spent=spent,
        )
        heading_for: Optional[Coord] = None
        if plan is None or (stop is not None and not plan):
            stopped = stop.reason if stop is not None else "no plan"
            onward = stop.onward if stop is not None else None
            break
        if stop is not None and stop.onward is not None:
            onward = stop.onward
            stopped = _look_reason(onward, observation)
            if looking_at is not None and observation["position"] == looking_at:
                # Standing on the edge of knowledge it set out for, and there is
                # still no path: the unseen ground is in the window now, so the
                # walk has done its job and the next move is a fresh decision.
                break
            found = _coord(onward.get("heading_for"))
            looking_at = found or looking_at
            heading_for = _coord(onward.get("unseen_at"))
        if not plan:
            arrived = True
            break

        batch = batch_within_budget(plan[:MAX_ACTIONS_PER_BATCH], budget)
        if not batch:
            stopped = f"frame budget of {frame_budget} frames spent"
            break

        was_at = observation["position"]
        trail = _trail(collision, was_at, batch)
        result = await act(batch)
        executed += int(result.get("actions_executed") or 0)
        budget -= sum(frames_for_action(action) for action in batch)
        outcome = result.get("outcome") or {}
        moved = outcome.get("moved") or 0
        walked += int(moved)
        observation = observation_from_bundle(result.get("bundle"))
        if observation["position"] is None:
            stopped = "lost track of the player — a battle or a cutscene interrupted the walk"
            onward = None
            break
        if target_xy is not None and not _same_map(observation["map_name"], origin_map):
            # A warp fired. Nothing routes *through* one, so the only way this
            # happens is that the goal tile was itself a warp — a cave ladder
            # is both — and stepping onto it is arriving on it.
            arrived = trail[-1] == target_xy and len(trail) == len(batch) + 1
            where = f"{observation['map_name']} at {list(observation['position'])}"
            stopped = (
                f"{list(target_xy)} is a warp, so walking onto it left {origin_map}: now on {where}"
                if arrived
                else f"left {origin_map} through a warp and is now on {where}"
            )
            onward = None
            break
        if heading_for is not None and len(batch) == len(plan):
            # Walked the whole way to the edge of knowledge. Looking at it again
            # is not a second answer, so it is out of the running from here on.
            spent.add(heading_for)
        if not moved or outcome.get("blocked_after") is not None:
            blocked_on = _refused_tile(trail, observation["position"])
            if blocked_on is not None and blocked_on not in refused:
                # The one fact this round produced. Keep it, and let the next
                # round plan without it rather than rebuild the same plan.
                refused.add(blocked_on)
                continue
        if not moved:
            stopped = (
                f"blocked after {outcome.get('blocked_after') or 0} of {len(batch)} steps on "
                f"{observation['map_name']} — the way the plan wanted is not walkable"
            )
            onward = None
            break
    else:
        stopped = f"gave up after {MAX_GOTO_ROUNDS} replanning rounds"
        onward = None

    if not arrived and walked:
        # "not reachable" and "walked twelve tiles and then could not continue"
        # are different events, and the agent was being told the first one when
        # the second is what happened.
        goal = target_map or str(list(target_xy or ()))
        stopped = f"walked {walked} toward {goal}, then {stopped}"
    return {
        "walked": walked,
        "arrived": arrived,
        "stopped_because": "arrived" if arrived else stopped,
        "onward": None if arrived else onward,
        "actions_executed": executed,
        "bundle": observation["bundle"],
    }


# ---------------------------------------------------------------------------
# /sim
# ---------------------------------------------------------------------------


def simulate_payload(
    plan: Sequence[str],
    snapshot: dict,
    explored: Optional[dict],
) -> dict:
    """Run *plan* over live collision on paper. Nothing is pressed."""
    position = _coord(snapshot.get("player_position"))
    if position is None:
        raise Conflict("No player position to simulate from.")
    collision = collision_from(snapshot, explored)
    try:
        result = world_mod.simulate(
            list(plan),
            collision,
            position,
            str(snapshot.get("facing") or "down"),
            warp_coords(snapshot),
        )
    except Exception as exc:  # noqa: BLE001 — an unknown token is the caller's mistake
        raise CapabilityError(str(exc)) from exc

    notes = [hop.describe() for hop in result.hops]
    if result.unverified_from is not None:
        notes.append(
            f"step {result.unverified_from} leaves the live window, so everything from "
            "there on is read off the remembered map and may be stale."
        )
    return {
        "end": list(result.end_pos),
        "facing": result.end_facing,
        "steps": result.steps_taken,
        "blocked_at": result.blocked_at,
        "blocked_by": result.blocked_by,
        "warp_at": result.warp_at,
        # A jump is not a block and not a step: one press, two tiles, no way
        # back. Answering "blocked by wall" here is what taught the agent to
        # walk into ledges to find out.
        "hops": [
            {
                "at": hop.index,
                "direction": hop.direction,
                "from": list(hop.start),
                "to": list(hop.landing),
                "one_way": True,
            }
            for hop in result.hops
        ],
        "certain": result.certain,
        "unverified_from": result.unverified_from,
        "basis": collision_basis(collision),
        "note": " ".join(notes) or None,
    }


# ---------------------------------------------------------------------------
# /frontier
# ---------------------------------------------------------------------------


def frontier_payload(snapshot: dict, explored: Optional[dict], seen: Collection[Coord]) -> dict:
    """Reachable-but-unseen tiles on the current map, nearest first.

    Reachable the way the game moves: a ledge is a one-way edge out, never an
    edge back in. Split into what the live window vouches for and what is only
    remembered, because a confidently wrong tile costs more than a missing one —
    three sessions were spent walking at ground that was never reachable.
    """
    position = _coord(snapshot.get("player_position"))
    if position is None:
        raise Conflict("No player position to explore from.")
    collision = collision_from(snapshot, explored)
    detail = world_mod.frontier_detail(collision, seen, position)
    confirmed = [tile.coord for tile in detail if tile.certain]
    believed = len(detail) - len(confirmed)
    note = f"{len(confirmed)} confirmed by the live window"
    if believed:
        note += (
            f"; {believed} reached only across remembered ground, which is a belief and "
            "not a fact — walk it and check rather than trusting it"
        )
    return {
        "map": snapshot.get("map_name"),
        "from": list(position),
        "tiles": [list(tile.coord) for tile in detail],
        "count": len(detail),
        "confirmed": [list(coord) for coord in confirmed],
        "confirmed_count": len(confirmed),
        "believed_count": believed,
        "basis": collision_basis(collision),
        "note": note,
    }


# ---------------------------------------------------------------------------
# /calc
# ---------------------------------------------------------------------------

#: Gen 1 stats are DV- and EV-dependent and neither is in the enemy battle
#: struct the reader exposes, so an unknown stat is estimated from the species'
#: base stat at an average DV. Off by a few points, not by a factor.
DEFAULT_DV = 8


def _estimated_stats(mon: dict) -> dict:
    base = ((gamedata.species(str(mon.get("species") or "")) or {}).get("base")) or {}
    level = max(1, int(mon.get("level") or 1))

    def stat(key: str) -> int:
        return ((int(base.get(key, 0)) + DEFAULT_DV) * 2 * level) // 100 + 5

    return {
        "attack": stat("atk"),
        "defense": stat("def"),
        "speed": stat("spd"),
        "special": stat("spc"),
    }


def _combatant(mon: dict) -> dict:
    """A mon in the shape ``gamedata.damage_range`` wants."""
    stats = dict(mon.get("stats") or {})
    if not {"attack", "defense", "special"} <= set(stats):
        estimated = _estimated_stats(mon)
        estimated.update(stats)
        stats = estimated
    return {
        "species": mon.get("species"),
        "level": int(mon.get("level") or 1),
        "types": list(mon.get("types") or []),
        "stats": stats,
    }


def _damage(attacker: dict, move_name: str, defender: dict) -> Optional[tuple[int, int]]:
    try:
        return gamedata.damage_range(attacker, move_name, defender)
    except KeyError:
        return None


def calc_payload(battle: dict, party: Sequence[dict], moves: Sequence[dict]) -> dict:
    """What each of the active Pokemon's moves would do, and what it faces.

    Damage is the honest Gen 1 range — the game rolls 217..255 out of 255 — so
    the pair is (worst roll, best roll), and ``turns_to_ko`` counts worst rolls.
    """
    if not battle.get("in_battle"):
        raise Conflict("Not in a battle. There is nothing to calculate against.")
    enemy_mon = battle.get("enemy") or {}
    if not enemy_mon.get("species"):
        raise Conflict("The enemy Pokemon is not readable yet — the battle is still starting.")
    if not party:
        raise Conflict("No party Pokemon to attack with.")

    attacker = _combatant(party[0])
    defender = _combatant(enemy_mon)
    enemy_hp = int(enemy_mon.get("hp") or 0)

    entries = []
    for move in moves:
        name = str(move.get("name") or "")
        record = gamedata.move(name)
        if record is None:
            entries.append(
                {
                    "move": name,
                    "type": None,
                    "power": None,
                    "effectiveness": None,
                    "damage": [0, 0],
                    "turns_to_ko": None,
                    "pp": move.get("pp"),
                }
            )
            continue
        rolled = _damage(attacker, name, defender) or (0, 0)
        turns = math.ceil(enemy_hp / rolled[0]) if rolled[0] > 0 and enemy_hp > 0 else None
        entries.append(
            {
                "move": name,
                "type": record["type"],
                "power": record["power"],
                "effectiveness": gamedata.effectiveness(record["type"], defender["types"]),
                "damage": [rolled[0], rolled[1]],
                "turns_to_ko": turns,
                # Without this `calc` ranks a move at 0 PP as the best available
                # and `poke fight` then refuses it. That happened 12 times.
                "pp": move.get("pp"),
            }
        )

    threat = 0
    for name in enemy_mon.get("moves") or ():
        rolled = _damage(defender, str(name), attacker)
        if rolled is not None:
            threat = max(threat, rolled[1])

    return {
        "moves": entries,
        "enemy": {
            "species": enemy_mon.get("species"),
            "level": enemy_mon.get("level"),
            "hp": enemy_hp,
            "types": list(enemy_mon.get("types") or []),
        },
        "threat": threat,
    }


# ---------------------------------------------------------------------------
# /guide
# ---------------------------------------------------------------------------


def guide_outline() -> dict:
    return {"outline": guides.outline()}


def guide_search(query: str, limit: int = 5) -> dict:
    results = guides.search(query, limit=limit)
    return {
        "results": [
            {"ref": section.ref, "title": section.title, "summary": section.summary}
            for section in results
        ]
    }


def split_ref(ref: str) -> tuple[str, str]:
    guide, _, slug = str(ref).partition("/")
    if not guide or not slug:
        raise NotFound(f"{ref!r} is not a guide reference. Use '<guide>/<slug>'.")
    return guide, slug


def guide_section(ref: str) -> dict:
    """One section's body, addressed as ``guide/slug``."""
    guide, slug = split_ref(ref)
    body = guides.read(guide, slug)
    if body is None:
        raise NotFound(f"No guide section at {ref!r}. GET /guide lists every section.")
    title = next(
        (
            section.title
            for section in guides.index()
            if section.guide == guide and section.slug == slug
        ),
        slug,
    )
    return {"guide": guide, "slug": slug, "title": title, "body": body}


# ---------------------------------------------------------------------------
# /progress
# ---------------------------------------------------------------------------


def progress_payload(summary: dict, presses: int) -> dict:
    """The milestone scoreboard, in the currency runs are compared in: buttons.

    ``frontier`` is the same table read forwards instead of backwards: of the
    63 rungs, the few the game will currently let the player attempt. It is a
    menu, not an instruction -- which one to take stays the model's call.
    """
    furthest = summary.get("furthest")
    milestone = MILESTONES_BY_ID.get(furthest) if furthest else None
    return {
        "count": summary.get("count", 0),
        "total": summary.get("total", 0),
        "furthest": furthest,
        "furthest_label": milestone.label if milestone else None,
        "latest": list(summary.get("latest") or []),
        "presses": int(presses),
        "frontier": [
            {
                "id": open_id,
                "label": MILESTONES_BY_ID[open_id].label,
                "gives": list(MILESTONE_DAG[open_id].effects),
            }
            for open_id in (summary.get("frontier") or [])
            if open_id in MILESTONES_BY_ID
        ],
    }


# ---------------------------------------------------------------------------
# /gamedata
#
# 223 maps, 334 trainers, 59 encounter tables, 151 species and 165 moves sit
# under ``pokemon_agent/data/game/``, and until these functions existed none of
# it was reachable from the agent. Every answer is shaped to be *printed*: what
# is in Pewter Gym should cost a few lines of context, not a JSON dump the model
# then has to skim.
# ---------------------------------------------------------------------------

#: What ``GET /gamedata/<topic>`` accepts.
GAMEDATA_TOPICS = ("trainers", "encounters", "species", "move", "items", "shops", "types")

#: Rows returned when the caller does not say. No table in the game data is
#: longer than a couple of dozen rows per map, so this only bites if the
#: generator grows one; the answer still says how many rows were cut.
GAMEDATA_LIMIT = 25
GAMEDATA_MAX_LIMIT = 200


def _gamedata_map_name(name: Optional[str]) -> str:
    """Match *name* against the game's own map table, ignoring case."""
    if not name or not str(name).strip():
        raise CapabilityError("This lookup needs a map, for example ?map=Pewter Gym")
    wanted = str(name).strip().lower()
    for known in gamedata.world():
        if known.lower() == wanted:
            return known
    near = [known for known in gamedata.world() if wanted in known.lower()][:5]
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    raise NotFound(f"No map called {name!r}. Names are the game's own, like 'Route 3'.{hint}")


def _gamedata_limit(limit: Optional[int]) -> int:
    if limit is None:
        return GAMEDATA_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise CapabilityError(f"limit must be a number, not {limit!r}") from None
    if value < 1:
        raise CapabilityError("limit must be at least 1.")
    return min(value, GAMEDATA_MAX_LIMIT)


def _page(rows: Sequence[Any], limit: int) -> tuple[list, dict]:
    """The first *limit* rows, plus the counts that say what was left out."""
    kept = list(rows[:limit])
    counts: dict[str, Any] = {"count": len(rows)}
    if len(rows) > limit:
        counts["shown"] = len(kept)
        counts["truncated"] = True
    return kept, counts


def gamedata_trainers(map_name: Optional[str], limit: Optional[int] = None) -> dict:
    """Who fights you on a map, where they stand, and with what."""
    resolved = _gamedata_map_name(map_name)
    rows = [
        {
            "class": entry.get("trainer_class"),
            "at": [entry.get("x"), entry.get("y")],
            "team": [f"{mon['species']} L{mon['level']}" for mon in entry.get("team") or ()],
        }
        for entry in gamedata.trainers(resolved)
    ]
    kept, counts = _page(rows, _gamedata_limit(limit))
    return {"map": resolved, **counts, "trainers": kept}


def _encounter_table(table: Optional[dict]) -> Optional[dict]:
    """One encounter table, merged per species.

    Route 3 has ten slots and three species. "Pidgey L6-8, 45%" is the fact
    worth carrying; ten rows of the same three names is not.
    """
    if not table:
        return None
    merged: dict[str, dict] = {}
    for slot in table.get("slots") or ():
        row = merged.setdefault(slot["species"], {"levels": [], "chance": 0.0})
        row["levels"].append(int(slot["level"]))
        row["chance"] += float(slot.get("chance") or 0.0)
    levels = [level for row in merged.values() for level in row["levels"]]
    return {
        "rate": table.get("rate"),
        "levels": [min(levels), max(levels)] if levels else None,
        "species": [
            {
                "species": species,
                "levels": [min(row["levels"]), max(row["levels"])],
                "chance": round(row["chance"], 3),
            }
            for species, row in sorted(merged.items(), key=lambda item: -item[1]["chance"])
        ],
    }


def gamedata_encounters(map_name: Optional[str]) -> dict:
    """What is in the grass and what is in the water."""
    resolved = _gamedata_map_name(map_name)
    table = gamedata.encounters(resolved) or {}
    return {
        "map": resolved,
        "grass": _encounter_table(table.get("grass")),
        "water": _encounter_table(table.get("water")),
    }


def _species_name(name: Optional[str]) -> str:
    if not name or not str(name).strip():
        raise CapabilityError("This lookup needs a name, for example ?name=Charmeleon")
    wanted = str(name).strip().lower()
    for known in gamedata.all_species():
        if known.lower() == wanted:
            return known
    near = [known for known in gamedata.all_species() if known.lower().startswith(wanted[:3])][:5]
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    raise NotFound(f"No Pokemon called {name!r}.{hint}")


def gamedata_species(name: Optional[str], full: bool = False) -> dict:
    """One species, as the numbers that decide a fight.

    The learnset is ``[level, move]`` pairs and the TM list is a count, so the
    whole entry stays small enough to print next to a battle. ``full`` adds the
    TM list and the growth rate back for the rare caller that wants them.
    """
    resolved = _species_name(name)
    entry = gamedata.all_species()[resolved]
    payload = {
        "name": resolved,
        "dex": entry.get("dex"),
        "types": list(entry.get("types") or ()),
        "base": entry.get("base"),
        "catch_rate": entry.get("catch_rate"),
        "base_exp": entry.get("base_exp"),
        "evolves": [
            f"{evolution['to']} by {evolution['method']} {evolution.get('param')}".strip()
            for evolution in entry.get("evolutions") or ()
        ],
        "learnset": [[move["level"], move["move"]] for move in entry.get("learnset") or ()],
    }
    tm_hm = list(entry.get("tm_hm") or ())
    if full:
        payload["growth"] = entry.get("growth")
        payload["tm_hm"] = tm_hm
    else:
        payload["tm_hm_count"] = len(tm_hm)
    return payload


def gamedata_move(name: Optional[str]) -> dict:
    """One move: type, power, accuracy, PP, and which stat it attacks with."""
    if not name or not str(name).strip():
        raise CapabilityError("This lookup needs a name, for example ?name=Ember")
    wanted = str(name).strip().lower()
    resolved = next((known for known in gamedata.all_moves() if known.lower() == wanted), None)
    if resolved is None:
        near = [known for known in gamedata.all_moves() if known.lower().startswith(wanted[:3])][:5]
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise NotFound(f"No move called {name!r}.{hint}")
    entry = gamedata.all_moves()[resolved]
    move_type = entry.get("type")
    return {
        "name": resolved,
        "type": move_type,
        "power": entry.get("power"),
        "accuracy": entry.get("accuracy"),
        "pp": entry.get("pp"),
        # Gen 1 splits by type, not by move: every Fire move is special and
        # every Normal move is physical, and that decides which stat it reads.
        "damage_class": "special" if move_type in gamedata.types()["special_types"] else "physical",
        "effect": entry.get("effect"),
    }


def gamedata_items(map_name: Optional[str], limit: Optional[int] = None) -> dict:
    """Item balls and hidden items on a map, with the tile to stand on."""
    resolved = _gamedata_map_name(map_name)
    rows = []
    for entry in gamedata.items(resolved):
        row = {"item": entry.get("item"), "at": [entry.get("x"), entry.get("y")]}
        if entry.get("hidden"):
            # Only worth saying when true: a hidden item needs ITEMFINDER and a
            # press on the tile, a visible one is a ball you walk into.
            row["hidden"] = True
        rows.append(row)
    kept, counts = _page(rows, _gamedata_limit(limit))
    return {"map": resolved, **counts, "items": kept}


def gamedata_shops(map_name: Optional[str]) -> dict:
    """What a mart sells. Empty stock is not an error: most maps sell nothing."""
    resolved = _gamedata_map_name(map_name)
    shop = gamedata.shops(resolved)
    return {"map": resolved, "items": list(shop.get("items") or ()) if shop else None}


def gamedata_types(move_type: Optional[str] = None, against: Optional[str] = None) -> dict:
    """The type chart, as the answer rather than as the table.

    With nothing, the type names. With a move type, what it beats and what it
    bounces off. With defending types too, the one multiplier.
    """
    chart = gamedata.types()
    known_types = list(chart["types"])
    if not move_type:
        return {"types": known_types}
    resolved = _one_type(move_type, known_types)
    defenders = [part.strip() for part in str(against or "").split(",") if part.strip()]
    if defenders:
        canonical = [_one_type(part, known_types) for part in defenders]
        return {
            "type": resolved,
            "against": canonical,
            "multiplier": gamedata.effectiveness(resolved, canonical),
        }
    row = chart["chart"].get(resolved, {})
    return {
        "type": resolved,
        "super_effective": sorted(name for name, value in row.items() if value > 1),
        "not_very_effective": sorted(name for name, value in row.items() if 0 < value < 1),
        "no_effect": sorted(name for name, value in row.items() if value == 0),
    }


def _one_type(name: str, known_types: Sequence[str]) -> str:
    wanted = str(name).strip().lower()
    for known in known_types:
        if known.lower() == wanted:
            return known
    raise NotFound(f"No type called {name!r}. Types: {', '.join(known_types)}.")


def gamedata_payload(
    topic: str,
    *,
    map_name: Optional[str] = None,
    name: Optional[str] = None,
    limit: Optional[int] = None,
    full: bool = False,
    against: Optional[str] = None,
) -> dict:
    """Answer one ``/gamedata/<topic>`` request. The only entry point a route needs."""
    if topic == "trainers":
        return gamedata_trainers(map_name, limit)
    if topic == "encounters":
        return gamedata_encounters(map_name)
    if topic == "items":
        return gamedata_items(map_name, limit)
    if topic == "shops":
        return gamedata_shops(map_name)
    if topic == "species":
        return gamedata_species(name, full)
    if topic == "move":
        return gamedata_move(name)
    if topic == "types":
        return gamedata_types(name, against)
    raise NotFound(f"No game data called {topic!r}. Topics: {', '.join(GAMEDATA_TOPICS)}.")


__all__ = [
    "CapabilityError",
    "Conflict",
    "MAX_GOTO_ROUNDS",
    "NotFound",
    "calc_payload",
    "canonical_map_name",
    "collision_from",
    "collision_basis",
    "frontier_payload",
    "GAMEDATA_TOPICS",
    "gamedata_encounters",
    "gamedata_items",
    "gamedata_move",
    "gamedata_payload",
    "gamedata_shops",
    "gamedata_species",
    "gamedata_trainers",
    "gamedata_types",
    "guide_outline",
    "guide_search",
    "guide_section",
    "known_map_names",
    "observation_from_bundle",
    "plan_within",
    "progress_payload",
    "route_payload",
    "simulate_payload",
    "split_ref",
    "walk_to",
    "warp_coords",
]
