"""Cross-map routing and a dry-run simulator for plans.

Two halves that never touch each other:

`World` is the static map graph read from ``data/game/world.json`` — which map
connects to which, by edge or by warp. It answers "how do I get from Pallet Town
to Pewter City" with a list of hops, NOT with button presses: the buttons depend
on per-map collision, which no static file carries, and inventing them would be
worse than admitting we cannot.

`simulate`, `path_within` and `frontier` work inside one map, on the collision
grid the live navigation layer already produces. They let a caller check a plan
before spending it: where it stops, what stopped it, and which unseen ground is
still reachable.

North is up: ``walk_up`` decreases y. Every direction here comes from
`pathfinding.DIRECTIONS`, which is the one place that fact is written down.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .agent_cli import expand_actions
from .pathfinding import DIRECTIONS, directions_to_actions

Coord = Tuple[int, int]

#: Where the generator writes the map graph. Missing is a normal state: the
#: file is generated from the ROM and a checkout may not have it yet.
DEFAULT_WORLD_PATH = Path(__file__).parent / "data" / "game" / "world.json"

#: The order neighbours and BFS expansions are tried in, so every result here
#: is deterministic rather than dict-order luck.
EDGE_ORDER = ("north", "south", "east", "west")


def _log(message: str) -> None:
    print(f"[world] {message}")


# ---------------------------------------------------------------------------
# Map graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hop:
    """One map-to-map transition.

    `at` is the warp tile in `from_map` you must step onto; for an edge
    connection there is no single tile — you walk off that side of the map —
    so `at` is None and `edge` names the side.
    """

    from_map: str
    to_map: str
    kind: str  # "warp" | "connection"
    at: Optional[Coord]  # warp tile in from_map; None for edge connections
    edge: Optional[str]  # "north"|"south"|"east"|"west" for connections; None for warps

    def describe(self) -> str:
        if self.kind == "connection":
            return f"{self.from_map} -> {self.to_map} (walk {self.edge})"
        where = f" at {self.at}" if self.at is not None else ""
        return f"{self.from_map} -> {self.to_map} (warp{where})"


@dataclass(frozen=True)
class MapInfo:
    """One map's static record, as read from the file."""

    name: str
    map_id: Optional[int]
    size: Optional[Coord]
    hops: Tuple[Hop, ...]


def _coord(value: object) -> Optional[Coord]:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
    else:
        return None
    try:
        return int(x), int(y)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _hops_for(name: str, payload: Mapping) -> Tuple[Hop, ...]:
    hops: List[Hop] = []
    connections = payload.get("connections")
    if isinstance(connections, Mapping):
        known = [edge for edge in EDGE_ORDER if edge in connections]
        known += [edge for edge in connections if edge not in EDGE_ORDER]
        for edge in known:
            target = connections.get(edge)
            if not isinstance(target, str) or not target:
                continue
            hops.append(Hop(from_map=name, to_map=target, kind="connection", at=None, edge=edge))
    for warp in payload.get("warps") or []:
        if not isinstance(warp, Mapping):
            continue
        target = warp.get("to_map")
        if not isinstance(target, str) or not target:
            continue
        hops.append(Hop(from_map=name, to_map=target, kind="warp", at=_coord(warp), edge=None))
    return tuple(hops)


class World:
    """The static map graph: maps as nodes, warps and edge connections as hops.

    A hop may name a map that has no record of its own — the file is allowed to
    be a subgraph. Such a map is routable *to* but has no outgoing hops, and
    `map_names` does not list it.
    """

    def __init__(
        self,
        maps: Mapping[str, MapInfo],
        *,
        source: Optional[Path] = None,
        generated_from: str = "",
    ) -> None:
        self._maps: Dict[str, MapInfo] = dict(maps)
        self.source = source
        self.generated_from = generated_from

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"World(maps={len(self._maps)}, source={self.source})"

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "World":
        """Read the graph from `path`, or from the packaged file.

        A missing or unreadable file yields an empty world rather than an
        exception: routing simply answers None until the data lands.
        """
        target = Path(path) if path is not None else DEFAULT_WORLD_PATH
        if not target.exists():
            return cls({}, source=None)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            raw_maps = payload["maps"]
            if not isinstance(raw_maps, Mapping):
                raise TypeError("maps is not an object")
        except Exception as exc:  # noqa: BLE001 - bad data must not break a caller
            _log(f"ignoring unreadable world file {target}: {exc}")
            return cls({}, source=None)

        maps: Dict[str, MapInfo] = {}
        for name, entry in raw_maps.items():
            if not isinstance(entry, Mapping):
                continue
            map_id = entry.get("map_id")
            try:
                map_id = int(map_id) if map_id is not None else None
            except (TypeError, ValueError):
                map_id = None
            maps[str(name)] = MapInfo(
                name=str(name),
                map_id=map_id,
                size=_coord(entry.get("size")),
                hops=_hops_for(str(name), entry),
            )
        return cls(
            maps,
            source=target,
            generated_from=str(payload.get("generated_from") or ""),
        )

    # -- queries ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._maps)

    def __contains__(self, map_name: object) -> bool:
        return map_name in self._maps

    def map_names(self) -> Tuple[str, ...]:
        """Every map with a record of its own, sorted."""
        return tuple(sorted(self._maps))

    def info(self, map_name: str) -> Optional[MapInfo]:
        return self._maps.get(map_name)

    def neighbours(self, map_name: str) -> Tuple[Hop, ...]:
        """Every hop leaving `map_name`; empty for an unknown map."""
        record = self._maps.get(map_name)
        return record.hops if record else ()

    def route(self, src: str, dst: str) -> Optional[Tuple[Hop, ...]]:
        """Fewest hops from `src` to `dst`, or None if there is no way.

        Returns hops, not button presses — the per-map walking between a map's
        entrance and its next exit needs live collision, which this file has
        none of. `src == dst` is an empty tuple, not None.
        """
        if src not in self._maps:
            return None
        if src == dst:
            return ()
        # A map named only as a hop target is a valid destination even without
        # a record of its own, so reachability is checked against the edges.
        previous: Dict[str, Optional[Hop]] = {src: None}
        queue: deque[str] = deque([src])
        while queue:
            current = queue.popleft()
            for hop in self.neighbours(current):
                if hop.to_map in previous:
                    continue
                previous[hop.to_map] = hop
                if hop.to_map == dst:
                    return self._unwind(previous, dst)
                queue.append(hop.to_map)
        return None

    def distance(self, src: str, dst: str) -> Optional[int]:
        """Hop count of the shortest route, or None if unreachable."""
        found = self.route(src, dst)
        return None if found is None else len(found)

    @staticmethod
    def _unwind(previous: Mapping[str, Optional[Hop]], dst: str) -> Tuple[Hop, ...]:
        hops: List[Hop] = []
        cursor: Optional[str] = dst
        while cursor is not None:
            hop = previous.get(cursor)
            if hop is None:
                break
            hops.append(hop)
            cursor = hop.from_map
        hops.reverse()
        return tuple(hops)


# ---------------------------------------------------------------------------
# Collision grids
# ---------------------------------------------------------------------------


class _Grid:
    """One map's walkability, normalised from whatever the caller had.

    The shapes accepted are the ones already in this codebase — nothing new is
    invented here:

    * ``list[list[int]]`` — `LiveNavigationSnapshot.terrain`, row major,
      ``rows[y][x]``, truthy means passable.
    * `LiveNavigationSnapshot` itself — its terrain is a 10x9 window, so the
      window offset (`window_top_left`) is applied and its `sprite_positions`
      become NPC blockers.
    * ``ExploredMaps.grid(map_id)`` — ``{"width", "height", "walkable": set}``,
      absolute coordinates.
    * a bare set of walkable coordinates, bounds taken from its extent.

    Outside the grid is "edge", not "wall": on a live window that means the
    edge of what is *known*, which may well be walkable ground the agent has
    not been shown yet.
    """

    __slots__ = ("width", "height", "origin", "walkable", "npcs")

    def __init__(
        self,
        walkable: Set[Coord],
        *,
        width: int,
        height: int,
        origin: Coord = (0, 0),
        npcs: Collection[Coord] = (),
    ) -> None:
        self.walkable = walkable
        self.width = width
        self.height = height
        self.origin = origin
        self.npcs = set(npcs)

    def in_bounds(self, x: int, y: int) -> bool:
        return (
            self.origin[0] <= x < self.origin[0] + self.width
            and self.origin[1] <= y < self.origin[1] + self.height
        )

    def is_walkable(self, x: int, y: int) -> bool:
        return (x, y) in self.walkable and (x, y) not in self.npcs

    def blocker(self, x: int, y: int) -> Optional[str]:
        """Why (x, y) cannot be entered: "edge", "npc", "wall", or None."""
        if not self.in_bounds(x, y):
            return "edge"
        if (x, y) in self.npcs:
            return "npc"
        if (x, y) not in self.walkable:
            return "wall"
        return None


def _rows_to_grid(rows: Sequence[Sequence[object]], origin: Coord, npcs: Collection[Coord]):
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    walkable: Set[Coord] = set()
    for local_y, row in enumerate(rows):
        for local_x, tile in enumerate(row):
            if tile:
                walkable.add((origin[0] + local_x, origin[1] + local_y))
    return _Grid(walkable, width=width, height=height, origin=origin, npcs=npcs)


def _as_grid(collision: object) -> _Grid:
    """Normalise any of the collision shapes this codebase already uses."""
    if isinstance(collision, _Grid):
        return collision

    # LiveNavigationSnapshot (duck-typed so navigation.py stays uncoupled).
    terrain = getattr(collision, "terrain", None)
    if terrain is not None and not isinstance(collision, Mapping):
        origin = _coord(getattr(collision, "window_top_left", None)) or (0, 0)
        npcs = [
            found
            for found in (_coord(item) for item in getattr(collision, "sprite_positions", ()) or ())
            if found is not None
        ]
        return _rows_to_grid(terrain, origin, npcs)

    if isinstance(collision, Mapping):
        npcs = [
            found
            for found in (
                _coord(item)
                for item in (collision.get("sprites") or collision.get("sprite_positions") or ())
            )
            if found is not None
        ]
        rows = collision.get("terrain")
        if rows is not None:
            origin = _coord(collision.get("window_top_left")) or (0, 0)
            return _rows_to_grid(rows, origin, npcs)
        raw = collision.get("walkable")
        if raw is None:
            raise TypeError("collision mapping needs a 'terrain' or 'walkable' key")
        walkable = {found for found in (_coord(item) for item in raw) if found is not None}
        width = int(collision.get("width") or (max((x for x, _ in walkable), default=-1) + 1))
        height = int(collision.get("height") or (max((y for _, y in walkable), default=-1) + 1))
        return _Grid(walkable, width=width, height=height, npcs=npcs)

    if isinstance(collision, (set, frozenset)):
        walkable = {found for found in (_coord(item) for item in collision) if found is not None}
        width = max((x for x, _ in walkable), default=-1) + 1
        height = max((y for _, y in walkable), default=-1) + 1
        return _Grid(walkable, width=width, height=height)

    if isinstance(collision, Sequence) and not isinstance(collision, (str, bytes)):
        return _rows_to_grid(collision, (0, 0), ())

    raise TypeError(f"unsupported collision shape: {type(collision).__name__}")


# ---------------------------------------------------------------------------
# Plan simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    """What a plan would do, without spending it.

    `blocked_at` and `warp_at` index the *expanded* action list: ``up:4`` is
    four steps, and `press_a` is a step that simply moves nothing. With no
    repeat forms in the plan those indices are the plan's own indices.
    """

    end_pos: Coord
    end_facing: str
    steps_taken: int
    blocked_at: Optional[int]  # index into the plan, None if it ran clean
    blocked_by: Optional[str]  # "wall" | "npc" | "edge"
    warp_at: Optional[int]  # index at which the plan stepped onto a warp tile
    trace: Tuple[Coord, ...]

    @property
    def ok(self) -> bool:
        return self.blocked_at is None


def _direction_of(action: str) -> Optional[str]:
    """The direction a canonical action walks, or None if it does not walk."""
    if action.startswith("walk_"):
        direction = action[len("walk_") :]
        if direction in DIRECTIONS:
            return direction
    return None


def simulate(
    plan: Sequence[str],
    collision,
    start: Coord,
    facing: str,
    warps: Collection[Coord] = (),
) -> SimResult:
    """Walk `plan` over `collision` on paper and report where it ends.

    The plan is in `poke act`'s vocabulary — ``walk_up``, ``up``, ``a``,
    ``up:4`` — and is expanded by the CLI's own parser, so simulator and game
    can never disagree about what a token means. An unknown token raises
    `agent_cli.ActionError`.

    Simulation stops at the first blocked step (its index lands in
    `blocked_at`) and at the first step onto a warp tile (`warp_at`): past a
    warp the player is on another map, whose collision this call does not have.

    Non-walking actions consume a step and change nothing. ``hold_<dir>_N``
    turns the player but its distance is not modelled, so it does not move
    them. Pressing into a wall turns the player to face it, exactly as the game
    does, so `end_facing` is the attempted direction even when blocked.
    """
    grid = _as_grid(collision)
    warp_set = {found for found in (_coord(item) for item in warps) if found is not None}
    actions = expand_actions(list(plan)) if plan else []

    position = (int(start[0]), int(start[1]))
    trace: List[Coord] = [position]
    steps_taken = 0
    blocked_at: Optional[int] = None
    blocked_by: Optional[str] = None
    warp_at: Optional[int] = None

    for index, action in enumerate(actions):
        direction = _direction_of(action)
        if direction is None:
            held = action.split("_")
            if len(held) == 3 and held[0] == "hold" and held[1] in DIRECTIONS:
                facing = held[1]
            steps_taken = index + 1
            continue
        facing = direction
        dx, dy = DIRECTIONS[direction]
        target = (position[0] + dx, position[1] + dy)
        blocker = grid.blocker(*target)
        if blocker is not None:
            blocked_at, blocked_by = index, blocker
            break
        position = target
        trace.append(position)
        steps_taken = index + 1
        if position in warp_set:
            warp_at = index
            break

    return SimResult(
        end_pos=position,
        end_facing=facing,
        steps_taken=steps_taken,
        blocked_at=blocked_at,
        blocked_by=blocked_by,
        warp_at=warp_at,
        trace=tuple(trace),
    )


# ---------------------------------------------------------------------------
# Within-map pathing
# ---------------------------------------------------------------------------


def _bfs(grid: _Grid, start: Coord, *, goal: Optional[Coord] = None):
    """Breadth-first flood from `start`, yielding (tile, previous) in order.

    `start` is entered whether or not it reads as walkable: the player may be
    standing on a warp or a doorway that the grid calls blocked, and refusing
    to path off it would strand them.
    """
    previous: Dict[Coord, Optional[Tuple[Coord, str]]] = {start: None}
    queue: deque[Coord] = deque([start])
    order: List[Coord] = [start]
    while queue:
        current = queue.popleft()
        if goal is not None and current == goal:
            break
        for direction, (dx, dy) in DIRECTIONS.items():
            step = (current[0] + dx, current[1] + dy)
            if step in previous or not grid.is_walkable(*step):
                continue
            previous[step] = (current, direction)
            queue.append(step)
            order.append(step)
    return previous, order


def path_within(collision, start: Coord, target: Coord) -> Optional[Tuple[str, ...]]:
    """Shortest walk from `start` to `target` inside one map, as actions.

    Returns ``('walk_up', 'walk_right', ...)`` — the same strings `poke act`
    takes — or None when `target` is blocked, off the grid, or walled off.
    An empty tuple means you are already there.
    """
    grid = _as_grid(collision)
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))
    if start == target:
        return ()
    if not grid.is_walkable(*target):
        return None

    previous, _ = _bfs(grid, start, goal=target)
    if target not in previous:
        return None

    directions: List[str] = []
    cursor: Optional[Coord] = target
    while cursor is not None:
        step = previous.get(cursor)
        if step is None:
            break
        cursor, direction = step
        directions.append(direction)
    directions.reverse()
    return tuple(directions_to_actions(directions))


def frontier(
    collision,
    seen: Collection[Coord],
    start: Coord,
) -> Tuple[Coord, ...]:
    """Walkable tiles reachable from `start` that are not in `seen`, nearest first.

    "Reachable and unseen" is the whole of maze navigation: it answers "where
    can I still go that I have not been", which an ASCII map of a 10x9 window
    cannot. Nearest first is BFS order — step distance through walkable ground,
    not straight-line distance, so the first result is genuinely the cheapest
    place to go next. One flood over the map, so it costs O(tiles).
    """
    grid = _as_grid(collision)
    start = (int(start[0]), int(start[1]))
    seen_set = {found for found in (_coord(item) for item in seen) if found is not None}
    _, order = _bfs(grid, start)
    return tuple(tile for tile in order if tile not in seen_set and grid.is_walkable(*tile))


def unseen_reachable_count(collision, seen: Collection[Coord], start: Coord) -> int:
    """How much unseen ground is still reachable — the frontier's size."""
    return len(frontier(collision, seen, start))


__all__ = [
    "DEFAULT_WORLD_PATH",
    "Hop",
    "MapInfo",
    "SimResult",
    "World",
    "frontier",
    "path_within",
    "simulate",
    "unseen_reachable_count",
]
