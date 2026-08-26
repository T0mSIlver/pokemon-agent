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

Four rules the grid enforces, all learned the hard way. The first three are
facts about the *seam between two tiles*, which is why none of them can live in
a per-tile walkability grid and all three arrive as edges instead:

* **A ledge is a directed edge.** It reads as blocked collision, but
  `HandleLedges` runs before the collision check, so pressing into it jumps two
  tiles — and there is no way back up. Modelled as a one-way edge, never as an
  open tile, because undirected BFS across one is how a sealed 26-tile pocket
  got reported as an open route.
* **A tile pair collision is a missing edge between two passable tiles.**
  `CheckForTilePairCollisions` refuses the move both ways round, and the table
  it reads has CAVERN and FOREST entries only. So the rule is invisible on the
  overworld and decisive underground: in Mt. Moon it is the entire boundary
  between the upper floor and the lower one, and a simulator that does not know
  it walks straight through a wall the game will not let the player through.
* **A warp exit is an edge that leaves the map.** Pressing it looks exactly
  like walking into a wall — pokered only reaches the warp check *after* the
  move has collided — so a cave ladder at the edge of the map reads as
  "blocked by edge" to anything that stops at collision.
* **The live window outranks the remembered map.** The 10x9 window is this
  frame; the explored-map store is memory, and memory of a tile the player was
  recorded on mid-jump is a tile nobody can stand on. Every tile the window
  covers is a fact; every tile beyond it is a belief, and results say which.

And one rule about the absence of a tile. A grid holds the tiles somebody has
*looked at*; a tile missing from it is either a wall that was looked at and
found solid or ground nobody has ever been shown, and those are not the same
answer. `seen` is what separates them, and `reachable_region` is what reports
the difference: a region whose whole border was looked at is sealed, and a
region that stops at ground nobody has seen is merely unfinished.

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
from .navigation import ledge_hop_allows, ledge_landing, tile_pair_blocked_edges
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

    def hops_between(self, src: str, dst: str) -> Tuple[Hop, ...]:
        """*Every* hop from `src` to `dst`, not just the one `route` picked.

        `route` is a search over map names, so it keeps one edge per pair and
        throws the rest away. That is fine for counting hops and wrong for
        walking them: Mt. Moon 1F has three separate ladders down to B1F, and a
        caller handed only the first spends the whole call failing to reach a
        ladder while standing two tiles from another one.
        """
        return tuple(hop for hop in self.neighbours(src) if hop.to_map == dst)

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

    Two things ride along with walkability. `live` is the set of tiles this
    frame actually showed us, so a caller can separate what it saw from what it
    remembers. `ledges` is a `MovementEdges` holding the three per-seam facts —
    one-way jumps, uncrossable tile pairs, warp exits — because none of them is
    a property of a tile and a grid of tiles has nowhere to put them.
    """

    __slots__ = (
        "width",
        "height",
        "origin",
        "walkable",
        "npcs",
        "live",
        "seen",
        "ledges",
        "seams",
        "warp_exits",
        "warps",
    )

    def __init__(
        self,
        walkable: Set[Coord],
        *,
        width: int,
        height: int,
        origin: Coord = (0, 0),
        npcs: Collection[Coord] = (),
        live: Optional[Collection[Coord]] = None,
        seen: Optional[Collection[Coord]] = None,
        ledges: Optional[Mapping[Tuple[Coord, str], Coord]] = None,
        warps: Collection[Coord] = (),
    ) -> None:
        self.walkable = walkable
        self.width = width
        self.height = height
        self.origin = origin
        self.npcs = set(npcs)
        self.live: Set[Coord] = set(live or ())
        # Looked at, whatever the answer was. Absent a record of it, the only
        # tiles anybody can claim to have looked at are the ones this frame
        # showed and the ones remembered as ground; everything else is a tile
        # nobody has an opinion about, and saying "wall" there is a guess.
        self.seen: Set[Coord] = set(seen) if seen is not None else (self.live | set(walkable))
        self.ledges: Dict[Tuple[Coord, str], Coord] = dict(ledges or {})
        # The other two per-seam facts ride on the same object when there is
        # one, so no caller has to grow an argument to pass them along.
        self.seams: Set[Tuple[Coord, Coord]] = set(getattr(ledges, "blocked_pairs", ()) or ())
        self.warp_exits: Set[Tuple[Coord, str]] = set(getattr(ledges, "warp_exits", ()) or ())
        # Walking onto one of these leaves the map. See `is_absorbing`.
        self.warps: Set[Coord] = set(warps)

    def is_absorbing(self, coord: Coord) -> bool:
        """Whether a walk that reaches *coord* ends there, wherever it was going.

        A warp tile is not a corridor. Gen 1 fires it the moment you step on,
        so a route that crosses one does not continue on the other side of it
        -- it stops, on a different map, wherever that warp leads.

        Measured on Mt. Moon 1F: the shortest walk from the south entrance
        (14, 35) to the ladder at (5, 5) is 89 steps and crosses the *other*
        ladder at (17, 11) on step 72. Walked, it spends 72 presses and ends on
        B1F, and every /goto after that asks for (5, 5) on a floor that has no
        such tile. The tile stays reachable — you may still be going to it —
        but nothing routes through it.
        """
        return coord in self.warps

    def is_live(self, x: int, y: int) -> bool:
        """Whether this tile came from the current frame rather than memory."""
        return (x, y) in self.live

    def is_known(self, x: int, y: int) -> bool:
        """Whether anybody has ever looked at this tile, walkable or not.

        Off the grid counts as known: the map ends there and no amount of
        walking will show us more of it.
        """
        return not self.in_bounds(x, y) or (x, y) in self.seen

    def seam_between(self, here: Coord, there: Coord) -> bool:
        """Whether the tileset refuses the move between two adjacent tiles.

        Symmetric, because `CheckForTilePairCollisions` tries the pair both ways
        round: neither of the two tiles is the one doing the blocking.
        """
        return (here, there) in self.seams or (there, here) in self.seams

    def is_warp_exit(self, coord: Coord, direction: str) -> bool:
        """Whether pressing *direction* here fires the warp under the player.

        Only ever true for a direction ordinary collision already refused —
        pokered reaches its warp check after the move has collided — so this is
        asked *after* `blocker`, never instead of it.
        """
        return (coord, direction) in self.warp_exits

    def ledge_from(self, coord: Coord, direction: str) -> Optional[Coord]:
        """Where pressing *direction* here lands after a ledge jump, if it does.

        One way only: the table is keyed by the direction pressed, and pokered's
        `ledge_tiles.asm` has no upward entry at all, so nothing ever hops back.
        """
        return self.ledges.get((coord, direction))

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


def _rows_to_grid(
    rows: Sequence[Sequence[object]],
    origin: Coord,
    npcs: Collection[Coord],
    ledges: Optional[Mapping[Tuple[Coord, str], Coord]] = None,
    warps: Collection[Coord] = (),
):
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    walkable: Set[Coord] = set()
    live: Set[Coord] = set()
    for local_y, row in enumerate(rows):
        for local_x, tile in enumerate(row):
            coord = (origin[0] + local_x, origin[1] + local_y)
            live.add(coord)
            if tile:
                walkable.add(coord)
    # Rows are somebody showing us tiles, which is the strongest evidence there
    # is: every one of them is a fact about now, walkable or not.
    return _Grid(
        walkable,
        width=width,
        height=height,
        origin=origin,
        npcs=npcs,
        live=live,
        seen=live,
        ledges=ledges,
        warps=warps,
    )


def _attribute(source: object, key: str) -> object:
    """Read *key* off a mapping or an object, whichever the caller handed us."""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _ledge_key(key: object) -> Optional[Tuple[Coord, str]]:
    """Normalise ``((x, y), "down")`` or ``(x, y, "down")`` into one shape."""
    if not isinstance(key, (list, tuple)):
        return None
    if len(key) == 2:
        coord, direction = _coord(key[0]), key[1]
    elif len(key) == 3:
        coord, direction = _coord((key[0], key[1])), key[2]
    else:
        return None
    if coord is None or direction not in DIRECTIONS:
        return None
    return coord, str(direction)


class MovementEdges(Dict[Tuple[Coord, str], Coord]):
    """The one-way ledge jumps, plus the two other facts about seams.

    It *is* the ledge mapping — ``{(tile, direction): landing}`` — and every
    caller that only wants ledges can keep reading it as exactly that. Two more
    sets ride along, because they are the same kind of thing and there is no
    other way to hand them to a grid of tiles:

    * `blocked_pairs` — adjacent tiles the tileset will not let you move
      between, in either direction, however passable both of them look.
    * `warp_exits` — ``(tile, direction)`` where pressing the direction fires
      the warp the player is standing on instead of colliding with a wall.

    They travel here rather than as extra arguments so that a collision map
    built by one module and simulated by another carries all three without
    either module growing a parameter for them.
    """

    __slots__ = ("blocked_pairs", "warp_exits")

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.blocked_pairs: Set[Tuple[Coord, Coord]] = set()
        self.warp_exits: Set[Tuple[Coord, str]] = set()


def _seam_pair(value: object) -> Optional[Tuple[Coord, Coord]]:
    """Normalise ``{"a": ..., "b": ...}`` or ``[(x, y), (x, y)]`` into a pair."""
    if isinstance(value, Mapping):
        first, second = _coord(value.get("a")), _coord(value.get("b"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = _coord(value[0]), _coord(value[1])
    else:
        return None
    if first is None or second is None:
        return None
    return first, second


def tile_id_map(tile_ids: object) -> Dict[Coord, int]:
    """Tile ids as ``{(x, y): id}``, from either shape this codebase carries.

    A coordinate-keyed mapping is what `mapdecode` builds in process. Rows of
    ints, ``rows[y][x]``, is what the same floor looks like once it has been
    through JSON — a tuple key cannot be serialised, so the wire shape is rows.
    Both mean the same map and neither caller should have to know which it got.
    """
    ids: Dict[Coord, int] = {}
    if isinstance(tile_ids, Mapping):
        for key, value in tile_ids.items():
            coord = _coord(key)
            if coord is None:
                continue
            try:
                ids[coord] = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return ids
    if isinstance(tile_ids, Sequence) and not isinstance(tile_ids, (str, bytes)):
        for y, row in enumerate(tile_ids):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                continue
            for x, value in enumerate(row):
                try:
                    ids[(x, y)] = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
    return ids


def movement_edges(source: object) -> MovementEdges:
    """Everything *source* knows about moving between two adjacent tiles.

    Ledge jumps are keyed by the tile you stand on and the direction you press;
    the value is where you land, which is two tiles away and never one. Shapes
    are read in increasing order of authority:

    * a ``ledges`` mapping somebody already built — including a `MovementEdges`,
      whose seams and warp exits come across with it,
    * ``tile_ids`` plus ``tileset`` — every ledge and every uncrossable seam in
      the live window, decided by pokered's own tables through
      `navigation.ledge_hop_allows` and `navigation.tile_pair_blocked_edges`,
    * ``blocked_pairs``, ``ledge_hops`` and ``warp_exit_directions`` — what the
      emulator itself published. The HTTP snapshot carries these and not the
      tile ids, so over the wire they are the only ones there are, and they are
      exactly the ones a plan starts from.

    A warp exit is only taken when the snapshot says it is *armed*: the engine
    arms a warp when the player walks onto it, and an unarmed one really does
    behave like the wall it looks like.
    """
    edges = MovementEdges()

    explicit = _attribute(source, "ledges")
    if isinstance(explicit, Mapping):
        for key, landing in explicit.items():
            found = _ledge_key(key)
            target = _coord(landing)
            if found is not None and target is not None:
                edges[found] = target
        edges.blocked_pairs |= set(getattr(explicit, "blocked_pairs", ()) or ())
        edges.warp_exits |= set(getattr(explicit, "warp_exits", ()) or ())

    tileset = _attribute(source, "tileset")
    ids = tile_id_map(_attribute(source, "tile_ids"))
    if isinstance(tileset, str) and ids:
        for coord, tile in ids.items():
            for direction, (dx, dy) in DIRECTIONS.items():
                neighbour = ids.get((coord[0] + dx, coord[1] + dy))
                if ledge_hop_allows(tileset, direction, tile, neighbour):
                    edges[(coord, direction)] = ledge_landing(coord, direction)
        edges.blocked_pairs |= tile_pair_blocked_edges(tileset, ids)

    for item in _attribute(source, "blocked_pairs") or ():
        pair = _seam_pair(item)
        if pair is not None:
            edges.blocked_pairs.add(pair)

    player = _coord(_attribute(source, "player_position"))
    hops = _attribute(source, "ledge_hops")
    if player is not None and isinstance(hops, Mapping):
        for direction, landing in hops.items():
            if direction not in DIRECTIONS:
                continue
            edges[(player, str(direction))] = _coord(landing) or ledge_landing(
                player, str(direction)
            )
    if player is not None and _attribute(source, "warp_exit_armed"):
        for direction in _attribute(source, "warp_exit_directions") or ():
            if direction in DIRECTIONS:
                edges.warp_exits.add((player, str(direction)))
    return edges


def ledge_edges(source: object) -> MovementEdges:
    """`movement_edges` under the name its first caller knows it by.

    The result is still a mapping of one-way ledge jumps and nothing about
    reading it as one has changed; it just carries the other two per-seam facts
    on the side now.
    """
    return movement_edges(source)


def warp_tiles(source: object) -> Set[Coord]:
    """Every tile on this map that walking onto ends the walk.

    Read off whichever `warps` list the caller happens to carry: the snapshot's
    ``{"x", "y", "warp_id"}`` records, the decoded floor's ``{"at": (x, y)}``
    ones, or bare pairs. All three shapes are already in this codebase and none
    of them was ever consulted by the flood.
    """
    found: Set[Coord] = set()
    for warp in _attribute(source, "warps") or ():
        coord = _coord(warp)
        if coord is None and isinstance(warp, Mapping):
            coord = _coord(warp.get("at")) or _coord(warp.get("coord"))
        if coord is not None:
            found.add(coord)
    return found


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
        return _rows_to_grid(terrain, origin, npcs, ledge_edges(collision), warp_tiles(collision))

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
            return _rows_to_grid(rows, origin, npcs, ledge_edges(collision), warp_tiles(collision))
        raw = collision.get("walkable")
        if raw is None:
            raise TypeError("collision mapping needs a 'terrain' or 'walkable' key")
        walkable = {found for found in (_coord(item) for item in raw) if found is not None}
        width = int(collision.get("width") or (max((x for x, _ in walkable), default=-1) + 1))
        height = int(collision.get("height") or (max((y for _, y in walkable), default=-1) + 1))
        # A merged map has to say which of its tiles the frame actually showed;
        # with no `live` key the whole thing is memory, and is reported as such.
        live = {found for found in (_coord(item) for item in collision.get("live") or ()) if found}
        raw_seen = collision.get("seen")
        seen = (
            {found for found in (_coord(item) for item in raw_seen) if found}
            if raw_seen is not None
            else None
        )
        return _Grid(
            walkable,
            width=width,
            height=height,
            npcs=npcs,
            live=live,
            seen=seen,
            ledges=ledge_edges(collision),
            warps=warp_tiles(collision),
        )

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


#: Only ever used to say which way you cannot come back.
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass(frozen=True)
class LedgeHop:
    """One ledge jump inside a simulated plan.

    A jump is not a step: it crosses two tiles for one button, it is legal even
    though the tile in between reads as blocked, and it cannot be undone. All
    three facts are why "blocked" was the wrong answer for it.
    """

    index: int  # index into the expanded plan
    direction: str
    start: Coord
    landing: Coord  # two tiles away, not one

    def describe(self) -> str:
        return (
            f"step {self.index} (walk_{self.direction}) jumps a ledge from "
            f"{self.start} to {self.landing} — two tiles for one press, and one way: "
            f"you cannot walk back {_OPPOSITE.get(self.direction, 'up')}."
        )


@dataclass(frozen=True)
class SimResult:
    """What a plan would do, without spending it.

    `blocked_at` and `warp_at` index the *expanded* action list: ``up:4`` is
    four steps, and `press_a` is a step that simply moves nothing. With no
    repeat forms in the plan those indices are the plan's own indices.

    `hops` and `unverified_from` are the two things a caller cannot infer from
    an end position: which presses jump a ledge rather than walk, and from which
    press on the answer stops being observation and becomes memory.
    """

    end_pos: Coord
    end_facing: str
    steps_taken: int
    blocked_at: Optional[int]  # index into the plan, None if it ran clean
    blocked_by: Optional[str]  # "wall" | "npc" | "edge" | "tile_pair"
    warp_at: Optional[int]  # index at which the plan stepped onto a warp tile
    trace: Tuple[Coord, ...]
    hops: Tuple[LedgeHop, ...] = ()  # one-way ledge jumps the plan takes
    unverified_from: Optional[int] = None  # first index decided by memory, not this frame

    @property
    def ok(self) -> bool:
        return self.blocked_at is None

    @property
    def certain(self) -> bool:
        """Whether every step was decided by tiles the live window showed."""
        return self.unverified_from is None


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

    A ledge is checked before collision, exactly as `HandleLedges` is in the
    game, so a direction that reads as blocked but is a legal jump moves the
    player two tiles and lands in `hops` rather than in `blocked_at`. A tile
    pair collision is checked *with* collision and reported as
    ``blocked_by="tile_pair"``: both tiles are passable and the step is not,
    which "wall" would have described as terrain nobody can see. A warp exit is
    checked after collision, exactly as `ExtraWarpCheck` is, so a cave ladder
    that ordinary collision refuses lands in `warp_at` and not in `blocked_at`.

    Non-walking actions consume a step and change nothing. ``hold_<dir>_N``
    turns the player but its distance is not modelled, so it does not move
    them. Pressing into a wall turns the player to face it, exactly as the game
    does, so `end_facing` is the attempted direction even when blocked.
    """
    grid = _as_grid(collision)
    warp_set = {found for found in (_coord(item) for item in warps) if found is not None}
    # batch=False: the 40-action and 3600-frame caps exist because the server
    # holds its one emulator lock for a batch it *executes*. A simulation presses
    # nothing and takes no lock. Applying the caps here cost one session 505 of
    # its 549 calls: the model grew a route through Mt. Moon segment by segment,
    # re-simulated it each time, and every attempt past forty actions came back
    # refused -- 41 to 159 actions, 186,737 characters of command and refusal for
    # no answer at all.
    actions = expand_actions(list(plan), batch=False) if plan else []

    position = (int(start[0]), int(start[1]))
    trace: List[Coord] = [position]
    steps_taken = 0
    blocked_at: Optional[int] = None
    blocked_by: Optional[str] = None
    warp_at: Optional[int] = None
    hops: List[LedgeHop] = []
    unverified_from: Optional[int] = None

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

        landing = grid.ledge_from(position, direction)
        if landing is not None:
            hops.append(LedgeHop(index=index, direction=direction, start=position, landing=landing))
            position = landing
            trace.append(position)
            steps_taken = index + 1
            if position in warp_set:
                warp_at = index
                break
            continue

        # A tile the frame did not show is memory, and memory of this map has
        # been wrong before. Say where the answer stopped being observation.
        if unverified_from is None and not grid.is_live(*target):
            unverified_from = index
        blocker = grid.blocker(*target)
        if blocker is None and grid.seam_between(position, target):
            blocker = "tile_pair"
        if blocker is not None:
            # The engine only looks for a warp once the move has collided, so
            # this is the same order and never turns a walkable step into one.
            if grid.is_warp_exit(position, direction):
                warp_at = index
                steps_taken = index + 1
                break
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
        hops=tuple(hops),
        unverified_from=unverified_from,
    )


# ---------------------------------------------------------------------------
# Within-map pathing
# ---------------------------------------------------------------------------


class _Flood:
    """One breadth-first flood: how each tile was reached, and how sure we are.

    `certain` is not a property of a tile but of the whole walk to it. A tile
    the frame is showing us right now, reached only across other tiles the frame
    is showing us, is a fact. One remembered tile anywhere on the way and the
    whole rest of that branch is a belief, because a single stale tile is all it
    takes to invent a corridor — which is exactly what happened on Route 3.
    """

    __slots__ = ("previous", "order", "certain", "distance")

    def __init__(self, start: Coord) -> None:
        self.previous: Dict[Coord, Optional[Tuple[Coord, str]]] = {start: None}
        self.order: List[Coord] = [start]
        # The player is standing on `start`, so it is a fact by definition.
        self.certain: Dict[Coord, bool] = {start: True}
        # Steps, not tiles: a ledge jump crosses two tiles for one press.
        self.distance: Dict[Coord, int] = {start: 0}


def _bfs(
    grid: _Grid,
    start: Coord,
    *,
    goal: Optional[Coord] = None,
    refused: Collection[Coord] = (),
) -> _Flood:
    """Breadth-first flood from `start` over a DIRECTED graph.

    Walking is symmetric; a ledge is not. Pressing a direction that jumps a
    ledge is one edge from here to two tiles away, with no edge back, so a flood
    can leave through a ledge and can never enter through one. Undirected BFS
    over the same tiles is what reported a sealed pocket as an open route.

    A warp is neither: it is an edge *in* with no edge onward. The flood may
    reach one, because it may be where you are going, and it never expands past
    one, because the walk that arrives there is over. See `_Grid.is_absorbing`.

    `refused` is ground this caller has already tried to step onto and been
    refused — an NPC in a doorway, a tile the decoder got wrong. It is left out
    of the flood entirely, so no plan built on this flood can propose it again.

    `start` is entered whether or not it reads as walkable: the player may be
    standing on a warp or a doorway that the grid calls blocked, and refusing
    to path off it would strand them.
    """
    flood = _Flood(start)
    blocked = {found for found in refused if found != start}
    queue: deque[Coord] = deque([start])
    while queue:
        current = queue.popleft()
        if goal is not None and current == goal:
            break
        if current != start and grid.is_absorbing(current):
            continue
        for direction, (dx, dy) in DIRECTIONS.items():
            hop = grid.ledge_from(current, direction)
            step = hop if hop is not None else (current[0] + dx, current[1] + dy)
            if step in flood.previous or step in blocked or not grid.is_walkable(*step):
                continue
            # Two passable tiles the tileset will not let you move between are
            # not connected, and a flood that crosses one invents a route.
            if hop is None and grid.seam_between(current, step):
                continue
            flood.previous[step] = (current, direction)
            flood.certain[step] = flood.certain[current] and grid.is_live(*step)
            flood.distance[step] = flood.distance[current] + 1
            flood.order.append(step)
            queue.append(step)
    return flood


def _actions_from(flood: _Flood, target: Coord) -> Optional[Tuple[str, ...]]:
    """Walk the `previous` chain back from `target` and turn it into actions."""
    if target not in flood.previous:
        return None
    directions: List[str] = []
    cursor: Optional[Coord] = target
    while cursor is not None:
        step = flood.previous.get(cursor)
        if step is None:
            break
        cursor, direction = step
        directions.append(direction)
    directions.reverse()
    return tuple(directions_to_actions(directions))


def path_within(
    collision,
    start: Coord,
    target: Coord,
    *,
    refused: Collection[Coord] = (),
) -> Optional[Tuple[str, ...]]:
    """Shortest walk from `start` to `target` inside one map, as actions.

    Returns ``('walk_up', 'walk_right', ...)`` — the same strings `poke act`
    takes — or None when `target` is blocked, off the grid, or walled off.
    An empty tuple means you are already there.

    `refused` is ground already tried and refused; see `_bfs`.
    """
    grid = _as_grid(collision)
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))
    if start == target:
        return ()
    if not grid.is_walkable(*target):
        return None

    return _actions_from(_bfs(grid, start, goal=target, refused=refused), target)


@dataclass(frozen=True)
class Region:
    """Everywhere one flood can get to, and what stopped it.

    The point of this object is the difference between the two ways a flood
    ends. It ends at a tile somebody looked at and found solid — a wall, and
    walking at it forever will not change that — or it ends at a tile nobody
    has ever been shown, which is not a wall, it is an unanswered question.
    A grid of walkable tiles cannot tell those apart, and a caller that treats
    the second as the first reports "unreachable" for ground it has simply
    never looked at.

    `edge_of_knowledge` is the boundary of the second kind: pairs of (a tile
    you can stand on, the unseen tile next to it), nearest first. `sealed` is
    true when there is no such pair — every way out of this region was looked
    at, and the region really is closed.
    """

    start: Coord
    order: Tuple[Coord, ...]
    distance: Mapping[Coord, int]
    certain: Mapping[Coord, bool]
    edge_of_knowledge: Tuple[Tuple[Coord, Coord], ...]
    _flood: "_Flood"

    @property
    def sealed(self) -> bool:
        """Whether every border of this region is ground somebody looked at."""
        return not self.edge_of_knowledge

    def __contains__(self, coord: object) -> bool:
        return coord in self.distance

    def steps_to(self, coord: Coord) -> Optional[int]:
        return self.distance.get(coord)

    def actions_to(self, coord: Coord) -> Optional[Tuple[str, ...]]:
        """Walk actions from `start` to `coord`, or None if it is not in here."""
        return _actions_from(self._flood, coord)

    def approach(self, goal: Coord) -> Optional[Tuple[Tuple[str, ...], int]]:
        """Actions that end with the player entering `goal`, and their cost.

        A door reads as blocked collision — you walk *into* it, you do not
        stand on it — so a goal this region cannot hold is approached from
        whichever neighbour it can, and the step into it is appended.
        """
        direct = self.actions_to(goal)
        if direct is not None:
            return direct, self.distance[goal]
        best: Optional[Tuple[Tuple[str, ...], int]] = None
        for direction, (dx, dy) in DIRECTIONS.items():
            neighbour = (goal[0] - dx, goal[1] - dy)
            cost = self.distance.get(neighbour)
            if cost is None:
                continue
            if best is not None and cost + 1 >= best[1]:
                continue
            leg = self.actions_to(neighbour)
            if leg is None:  # pragma: no cover - distance implies a path
                continue
            best = ((*leg, f"walk_{direction}"), cost + 1)
        return best


def reachable_region(collision, start: Coord, *, refused: Collection[Coord] = ()) -> Region:
    """Flood from `start` and report where it stopped and why.

    One flood answers every "can I get there from here" a caller has, which is
    the other half of the point: picking the nearest of three ladders used to
    cost three floods, so it was not done, so the first ladder in the file won
    every time.

    `refused` is ground the caller has already been refused on this trip. Pass
    it and the flood cannot rebuild the plan that just failed; see `_bfs`.
    """
    grid = _as_grid(collision)
    start = (int(start[0]), int(start[1]))
    flood = _bfs(grid, start, refused=refused)
    edge: List[Tuple[Coord, Coord]] = []
    for tile in flood.order:
        for dx, dy in DIRECTIONS.values():
            beyond = (tile[0] + dx, tile[1] + dy)
            if beyond in flood.previous or grid.is_known(*beyond):
                continue
            edge.append((tile, beyond))
    return Region(
        start=start,
        order=tuple(flood.order),
        distance=dict(flood.distance),
        certain=dict(flood.certain),
        edge_of_knowledge=tuple(edge),
        _flood=flood,
    )


@dataclass(frozen=True)
class ReachableTile:
    """One frontier tile and whether getting there is a fact or a belief."""

    coord: Coord
    certain: bool  # every tile on the way was in the live window


def frontier_detail(
    collision,
    seen: Collection[Coord],
    start: Coord,
) -> Tuple[ReachableTile, ...]:
    """`frontier`, with each tile labelled fact or belief. Same order."""
    grid = _as_grid(collision)
    start = (int(start[0]), int(start[1]))
    seen_set = {found for found in (_coord(item) for item in seen) if found is not None}
    flood = _bfs(grid, start)
    return tuple(
        ReachableTile(coord=tile, certain=flood.certain.get(tile, False))
        for tile in flood.order
        if tile not in seen_set and grid.is_walkable(*tile)
    )


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

    Reachable means reachable *the way the game moves*: down a ledge counts,
    back up one never does. Use `frontier_detail` when you need to know which of
    these tiles the live window vouches for and which are only remembered.
    """
    return tuple(tile.coord for tile in frontier_detail(collision, seen, start))


def unseen_reachable_count(collision, seen: Collection[Coord], start: Coord) -> int:
    """How much unseen ground is still reachable — the frontier's size."""
    return len(frontier(collision, seen, start))


__all__ = [
    "DEFAULT_WORLD_PATH",
    "Hop",
    "LedgeHop",
    "MapInfo",
    "MovementEdges",
    "ReachableTile",
    "Region",
    "SimResult",
    "World",
    "frontier",
    "frontier_detail",
    "ledge_edges",
    "movement_edges",
    "path_within",
    "reachable_region",
    "simulate",
    "tile_id_map",
    "unseen_reachable_count",
    "warp_tiles",
]
