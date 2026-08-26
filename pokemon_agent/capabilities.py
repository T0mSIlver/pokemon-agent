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
    rest. **Precedence is absolute: inside the window the live frame decides,
    including its negatives.** A tile the store calls walkable and the frame
    calls solid is solid; the store is memory, and it has held tiles nobody can
    stand on — the mid-air tile a ledge jump was once sampled on. Beyond the
    window there is nothing to check against, so those tiles travel as `live`
    minus themselves: present in `walkable`, absent from `live`, and every
    answer built on them says so.

    Sprites come from the live window only, because an NPC that stood somewhere
    once is not standing there now.
    """
    walkable: set[Coord] = set(explored.get("walkable") or ()) if explored else set()

    live: set[Coord] = set()
    origin = _coord(snapshot.get("window_top_left")) or (0, 0)
    for local_y, row in enumerate(snapshot.get("terrain") or []):
        for local_x, tile in enumerate(row):
            coord = (origin[0] + local_x, origin[1] + local_y)
            live.add(coord)
            if tile:
                walkable.add(coord)
            else:
                walkable.discard(coord)

    # A ledge's landing is two tiles away because the tile between is one no
    # player can stand on. That holds wherever the ledge was learned, so it
    # outranks a store that remembers standing there.
    ledges = world_mod.ledge_edges(snapshot)
    for (start, _direction), landing in ledges.items():
        walkable.discard(((start[0] + landing[0]) // 2, (start[1] + landing[1]) // 2))

    dimensions = snapshot.get("map_dimensions") or {}
    width = int(dimensions.get("width") or 0)
    height = int(dimensions.get("height") or 0)
    if not width or not height:
        width = max((x for x, _ in walkable), default=-1) + 1
        height = max((y for _, y in walkable), default=-1) + 1

    sprites = [found for found in (_coord(item) for item in snapshot.get("sprites") or ()) if found]
    return {
        "width": width,
        "height": height,
        "walkable": walkable,
        "sprites": sprites,
        "live": live,
        "ledges": ledges,
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
                f"{observation['map_name']}, checked against {collision_basis(collision)}"
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
                "half of the map may be walled off from the other. Checked against "
                f"{collision_basis(collision)}."
            )
        return plan, None

    step = _EDGE_STEPS.get(hop.edge or "")
    if step is None:
        return None, f"hop from {current} to {hop.to_map} has no direction to walk"
    goal = _edge_goal(collision, position, hop.edge or "")
    if goal is None:
        return None, (
            f"nothing on the {hop.edge} edge of {current} is reachable from {list(position)}. "
            "A hop is a plan, not a guarantee — this half of the map may be walled off, and "
            "a ledge you came down is not a way back up. Checked against "
            f"{collision_basis(collision)}."
        )
    if position == goal:
        return [f"walk_{step}"], None
    plan = plan_within(collision, position, goal)
    if plan is None:
        return None, (
            f"no walkable path to the {hop.edge} edge of {current}, checked against "
            f"{collision_basis(collision)}"
        )
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
