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
from pokemon_agent.milestones import MILESTONES_BY_ID
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


def collision_from(snapshot: dict, explored: Optional[dict] = None) -> dict:
    """Live walkability for the current map, in absolute coordinates.

    The live navigation window is only 10x9 tiles — enough to see the next step
    and nothing else — so what the map store has already learned fills in the
    rest. Where the two disagree the live window wins: it is this frame, and the
    store is memory. Sprites come from the live window only, because an NPC that
    stood somewhere once is not standing there now.
    """
    walkable: set[Coord] = set(explored.get("walkable") or ()) if explored else set()

    origin = _coord(snapshot.get("window_top_left")) or (0, 0)
    for local_y, row in enumerate(snapshot.get("terrain") or []):
        for local_x, tile in enumerate(row):
            coord = (origin[0] + local_x, origin[1] + local_y)
            if tile:
                walkable.add(coord)
            else:
                walkable.discard(coord)

    dimensions = snapshot.get("map_dimensions") or {}
    width = int(dimensions.get("width") or 0)
    height = int(dimensions.get("height") or 0)
    if not width or not height:
        width = max((x for x, _ in walkable), default=-1) + 1
        height = max((y for _, y in walkable), default=-1) + 1

    sprites = [found for found in (_coord(item) for item in snapshot.get("sprites") or ()) if found]
    return {"width": width, "height": height, "walkable": walkable, "sprites": sprites}


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


def route_payload(world: world_mod.World, source: str, target: str) -> dict:
    """Hops from *source* to *target*, or a reason there are none.

    Hops, not button presses: which buttons cross a map depends on that map's
    live collision, which no static file carries.
    """
    src = canonical_map_name(world, source)
    dst = canonical_map_name(world, target)
    if src == dst:
        return {"from": src, "to": dst, "distance": 0, "hops": []}

    hops = world.route(src, dst)
    if hops is None:
        return {
            "from": src,
            "to": dst,
            "distance": None,
            "hops": None,
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
        "hops": [
            {
                "from": hop.from_map,
                "to": hop.to_map,
                "kind": hop.kind,
                "at": list(hop.at) if hop.at is not None else None,
                "edge": hop.edge,
            }
            for hop in hops
        ],
    }


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


def _edge_goal(collision: dict, start: Coord, edge: str) -> Optional[Coord]:
    """The reachable tile on *edge* that is cheapest to walk to, if any."""
    width, height = collision["width"], collision["height"]
    if edge == "north":
        candidates = [(x, 0) for x in range(width)]
    elif edge == "south":
        candidates = [(x, height - 1) for x in range(width)]
    elif edge == "west":
        candidates = [(0, y) for y in range(height)]
    elif edge == "east":
        candidates = [(width - 1, y) for y in range(height)]
    else:
        return None
    walkable = collision["walkable"]
    candidates = [tile for tile in candidates if tile in walkable]
    candidates.sort(key=lambda tile: abs(tile[0] - start[0]) + abs(tile[1] - start[1]))
    for tile in candidates[:12]:
        if world_mod.path_within(collision, start, tile) is not None:
            return tile
    return None


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
) -> tuple[Optional[list[str]], Optional[str]]:
    """The next batch of walk actions, or the reason there is not one.

    Returns ``(actions, None)`` to keep walking or ``(None, reason)`` to stop.
    """
    position = observation["position"]
    snapshot = observation["snapshot"]

    if target_xy is not None:
        plan = plan_within(collision, position, target_xy)
        if plan is None:
            return None, (
                f"no walkable path from {list(position)} to {list(target_xy)} on "
                f"{observation['map_name']}"
            )
        return plan, None

    current = observation["map_name"]
    try:
        here = canonical_map_name(world, current)
    except NotFound:
        return None, (
            f"{current!r} is not a map the route graph knows, so there is nothing to "
            "plan a hop from"
        )
    hops = world.route(here, target_map or "")
    if hops is None:
        return None, f"no route from {current} to {target_map}"
    if not hops:
        return [], None
    hop = hops[0]

    if hop.kind == "warp" and hop.at is not None:
        if position == tuple(hop.at):
            step = warp_step_direction(position, snapshot.get("map_dimensions") or {})
            if step is None:
                return None, (
                    f"standing on the warp at {list(position)} on {current} but the map did "
                    "not change; this hop does not lead where the graph says it does"
                )
            return [f"walk_{step}"], None
        plan = plan_within(collision, position, tuple(hop.at))
        if plan is None:
            return None, (
                f"the warp to {hop.to_map} at {list(hop.at)} is not reachable on foot from "
                f"{list(position)} on {current}. A hop is a plan, not a guarantee — this "
                "half of the map may be walled off from the other."
            )
        return plan, None

    step = _EDGE_STEPS.get(hop.edge or "")
    if step is None:
        return None, f"hop from {current} to {hop.to_map} has no direction to walk"
    goal = _edge_goal(collision, position, hop.edge or "")
    if goal is None:
        return None, (
            f"nothing on the {hop.edge} edge of {current} is reachable from {list(position)}. "
            "A hop is a plan, not a guarantee — this half of the map may be walled off."
        )
    if position == goal:
        return [f"walk_{step}"], None
    plan = plan_within(collision, position, goal)
    if plan is None:
        return None, f"no walkable path to the {hop.edge} edge of {current}"
    return [*plan, f"walk_{step}"], None


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
        plan, reason = _leg_for_map(
            world,
            observation,
            collision,
            target_map=target_map,
            target_xy=target_xy,
        )
        if plan is None:
            stopped = reason or "no plan"
            break
        if not plan:
            arrived = True
            break

        batch = batch_within_budget(plan[:MAX_ACTIONS_PER_BATCH], budget)
        if not batch:
            stopped = f"frame budget of {frame_budget} frames spent"
            break

        result = await act(batch)
        executed += int(result.get("actions_executed") or 0)
        budget -= sum(frames_for_action(action) for action in batch)
        outcome = result.get("outcome") or {}
        moved = outcome.get("moved") or 0
        walked += int(moved)
        observation = observation_from_bundle(result.get("bundle"))
        if observation["position"] is None:
            stopped = "lost track of the player — a battle or a cutscene interrupted the walk"
            break
        if not moved:
            stopped = (
                f"blocked after {outcome.get('blocked_after') or 0} of {len(batch)} steps on "
                f"{observation['map_name']} — the way the plan wanted is not walkable"
            )
            break
    else:
        stopped = f"gave up after {MAX_GOTO_ROUNDS} replanning rounds"

    return {
        "walked": walked,
        "arrived": arrived,
        "stopped_because": "arrived" if arrived else stopped,
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
    return {
        "end": list(result.end_pos),
        "facing": result.end_facing,
        "steps": result.steps_taken,
        "blocked_at": result.blocked_at,
        "blocked_by": result.blocked_by,
        "warp_at": result.warp_at,
    }


# ---------------------------------------------------------------------------
# /frontier
# ---------------------------------------------------------------------------


def frontier_payload(snapshot: dict, explored: Optional[dict], seen: Collection[Coord]) -> dict:
    """Reachable-but-unseen tiles on the current map, nearest first."""
    position = _coord(snapshot.get("player_position"))
    if position is None:
        raise Conflict("No player position to explore from.")
    collision = collision_from(snapshot, explored)
    tiles = world_mod.frontier(collision, seen, position)
    return {
        "map": snapshot.get("map_name"),
        "from": list(position),
        "tiles": [list(tile) for tile in tiles],
        "count": len(tiles),
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
    """The milestone scoreboard, in the currency runs are compared in: buttons."""
    furthest = summary.get("furthest")
    milestone = MILESTONES_BY_ID.get(furthest) if furthest else None
    return {
        "count": summary.get("count", 0),
        "total": summary.get("total", 0),
        "furthest": furthest,
        "furthest_label": milestone.label if milestone else None,
        "latest": list(summary.get("latest") or []),
        "presses": int(presses),
    }


__all__ = [
    "CapabilityError",
    "Conflict",
    "MAX_GOTO_ROUNDS",
    "NotFound",
    "calc_payload",
    "canonical_map_name",
    "collision_from",
    "frontier_payload",
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
