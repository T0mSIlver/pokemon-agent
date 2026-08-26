"""Routing over pockets of a map, rather than over maps.

`world.route` keys its search by map name and skips any map already on the path.
That is fine where one map is one connected place, and it is wrong wherever a
map is several. Mt Moon B1F is four disconnected pockets, two warps each, and
you cannot walk between them: crossing the floor means climbing to 1F and coming
back down. So the way out of the mountain looks like this,

    Route 4#1     --warp (18, 5)--> Mt Moon 1F#0
    Mt Moon 1F#0  --warp ( 5, 5)--> Mt Moon B1F#1
    Mt Moon B1F#1 --warp (21,17)--> Mt Moon B2F#0
    Mt Moon B2F#0 --warp ( 5, 7)--> Mt Moon B1F#3
    Mt Moon B1F#3 --warp (27, 3)--> Route 4#0

and it visits Mt Moon B1F twice and Route 4 twice. A map-keyed search discards
both, so it does not find a worse route: it reports that there is none. Nineteen
warp refusals with no model in the loop, and fourteen hours of a live run, are
that one line of deduplication.

A map whose terrain is not known collapses to a single pocket, which is exactly
the old behaviour for that map and no worse.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple

from pokemon_agent import mapdecode

Coord = Tuple[int, int]

#: Where terrain is unknown, the whole map is one pocket and this is its index.
WHOLE_MAP = 0


#: Which edge of the destination you arrive on, walking off each edge.
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass(frozen=True)
class PocketHop:
    """One map-to-map move, with the pocket it leaves from and the one it lands in.

    A warp knows the tile you step onto and where it puts you. A connection
    knows the side you walk off, in `edge`, the same split `world.Hop` already
    makes. Where the map header's offsets are available it knows tiles too: one
    tile of the edge that reaches this pocket's landing, and that landing. They
    stay None for a connection given only as a direction and a map name.

    `at` is *a* way off the edge, not the only one or the nearest one. Any edge
    tile of the pocket that lands in the same pocket does as well.
    """

    from_map: str
    from_pocket: int
    to_map: str
    to_pocket: int
    at: Optional[Coord] = None  # the warp tile you step onto, in from_map
    landing: Optional[Coord] = None  # where you appear, in to_map
    edge: Optional[str] = None  # "north"|"south"|"east"|"west" for a connection

    @property
    def kind(self) -> str:
        return "connection" if self.edge else "warp"

    def describe(self) -> str:
        where = f"{self.from_map}#{self.from_pocket} -> {self.to_map}#{self.to_pocket}"
        if self.edge:
            if self.at is None or self.landing is None:
                return f"{where} (walk off the {self.edge} edge)"
            return (
                f"{where} (walk off the {self.edge} edge at {list(self.at)}, "
                f"landing {list(self.landing)})"
            )
        return f"{where} (warp at {list(self.at)}, landing {list(self.landing)})"


def components(walkable: Collection[Coord]) -> List[Set[Coord]]:
    """The walkable set split into pieces you can actually walk between.

    Largest first, so pocket 0 is the main floor wherever there is one and the
    numbering stays stable enough to appear in a message.
    """
    remaining = set(walkable)
    found: List[Set[Coord]] = []
    while remaining:
        start = next(iter(remaining))
        piece, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for step in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if step in remaining and step not in piece:
                    piece.add(step)
                    queue.append(step)
        found.append(piece)
        remaining -= piece
    found.sort(key=len, reverse=True)
    return found


def pocket_of(pieces: Sequence[Set[Coord]], coord: Optional[Coord]) -> Optional[int]:
    """Which pocket a tile is in, or None if it is not walkable.

    With no pieces at all the terrain is unknown, so every tile is in the one
    pocket that stands for the whole map.
    """
    if not pieces:
        return WHOLE_MAP
    if coord is None:
        return None
    for index, piece in enumerate(pieces):
        if coord in piece:
            return index
    return None


class PocketGraph:
    """The warp graph, with one node per pocket instead of one per map.

    `terrain_for` returns a map's walkable tiles, or None when nobody has
    decoded that map yet. Pockets are computed lazily and cached, because most
    routes touch a handful of maps and flooding every map in the game to answer
    one question is what stopped this being done sooner.
    """

    def __init__(
        self,
        warps_for: Callable[[str], Sequence[dict]],
        terrain_for,
        connections_for: Optional[Callable[[str], dict]] = None,
        size_for: Optional[Callable[[str], Optional[Tuple[int, int]]]] = None,
    ):
        self._warps_for = warps_for
        self._terrain_for = terrain_for
        self._connections_for = connections_for or (lambda _name: {})
        self._size_for = size_for or (lambda _name: None)
        self._pieces: Dict[str, List[Set[Coord]]] = {}

    def pieces(self, map_name: str) -> List[Set[Coord]]:
        if map_name not in self._pieces:
            walkable = self._terrain_for(map_name)
            self._pieces[map_name] = components(walkable) if walkable else []
        return self._pieces[map_name]

    def pocket_at(self, map_name: str, coord: Optional[Coord]) -> Optional[int]:
        return pocket_of(self.pieces(map_name), coord)

    def hops_from(self, map_name: str, pocket: int) -> List[PocketHop]:
        """Every warp you can reach on foot from this pocket."""
        out: List[PocketHop] = []
        for warp in self._warps_for(map_name):
            at = _coord(warp)
            target = warp.get("to_map")
            if at is None or not target or target == "???":
                continue
            if self.pocket_at(map_name, at) != pocket:
                continue
            landing = self._landing(target, warp.get("to_warp"))
            if landing is None:
                continue
            to_pocket = self.pocket_at(target, landing)
            if to_pocket is None:
                # The door exists and the far side is not walkable ground. That
                # is a decode gap, not a route, and pretending otherwise is how
                # a plan promises a tile nobody can stand on.
                continue
            out.append(
                PocketHop(
                    from_map=map_name,
                    from_pocket=pocket,
                    to_map=target,
                    to_pocket=to_pocket,
                    at=at,
                    landing=landing,
                )
            )
        out.extend(self._connection_hops(map_name, pocket))
        return out

    def _connection_hops(self, map_name: str, pocket: int) -> List[PocketHop]:
        """Edges you can walk off, from this pocket.

        Route 4 reaches Cerulean by walking off its east side, not through a
        door, so a warp-only graph answers "no route to Cerulean" while standing
        on the road to it. That was the first version of this file.

        A connection given only as a direction and a map name says nothing about
        which tile you land on, nor which pocket that is in when the far edge
        touches more than one. Guessing by taking every edge-touching pocket
        produced this:

            Route 4#1 -> Route 3   (walk off the south edge)
            Route 3   -> Route 4#4 (walk off the north edge)
            Route 4#4 -> Cerulean  (walk off the east edge)

        Route 4's south edge is touched by pocket 1 in the west and pocket 4 in
        the far east corner, and the route hops between them as though walking
        south and back north could move you sixty tiles sideways.

        With offsets it does not have to guess. `mapdecode.decode_connections`
        reads the map header's alignments, so each edge tile of this pocket
        converts to the exact tile it lands on, and the hop names the pocket
        that tile is in. The bogus middle hop above disappears for a better
        reason than caution: Route 4's south strip is 13 blocks at the west
        end, so x 81..89 is off the strip and lands nowhere at all.

        Without offsets the old rule stands -- one candidate pocket or no edge.
        Losing a route is a detour; inventing one is a loop.
        """
        out: List[PocketHop] = []
        for edge, spec in sorted((self._connections_for(map_name) or {}).items()):
            target, offsets = _connection_spec(edge, spec)
            if not target or edge not in OPPOSITE:
                continue
            if not self._touches_edge(map_name, pocket, edge):
                continue
            departures = self._edge_tiles(map_name, pocket, edge)
            if offsets is not None and departures:
                out.extend(self._offset_hops(map_name, pocket, target, offsets, departures))
                continue
            # No offsets, or no terrain to apply them to: the old rule.
            landing_pockets = self._pockets_on_edge(target, OPPOSITE[edge])
            if len(landing_pockets) != 1:
                continue
            out.append(
                PocketHop(
                    from_map=map_name,
                    from_pocket=pocket,
                    to_map=target,
                    to_pocket=landing_pockets[0],
                    edge=edge,
                )
            )
        return out

    def _offset_hops(
        self,
        map_name: str,
        pocket: int,
        target: str,
        offsets: "mapdecode.MapConnection",
        departures: List[Coord],
    ) -> List[PocketHop]:
        """One hop per pocket this pocket's edge tiles actually land in.

        Usually that is one hop. It is none when every tile of the edge is off
        the connection strip, which is the Route 4 east-corner case, and it is
        two when a strip is wide enough to straddle a split on the far side --
        which is a fact about the map, not an ambiguity to bail out on.
        """
        first_landing: Dict[int, Tuple[Coord, Coord]] = {}
        for tile in sorted(departures):
            landing = offsets.landing(tile)
            if landing is None:
                continue
            to_pocket = self.pocket_at(target, landing)
            if to_pocket is None:
                # The strip says you land there and the tile is not walkable.
                # That is a decode gap on the far side, not a route.
                continue
            first_landing.setdefault(to_pocket, (tile, landing))
        return [
            PocketHop(
                from_map=map_name,
                from_pocket=pocket,
                to_map=target,
                to_pocket=to_pocket,
                at=tile,
                landing=landing,
                edge=offsets.direction,
            )
            for to_pocket, (tile, landing) in sorted(first_landing.items())
        ]

    def _edge_tiles(self, map_name: str, pocket: int, edge: str) -> List[Coord]:
        """The tiles of this pocket that sit on that edge; empty if unknown."""
        pieces = self.pieces(map_name)
        if not pieces or pocket >= len(pieces):
            return []
        bounds = self._bounds(map_name)
        return [tile for tile in pieces[pocket] if _on_edge(tile, edge, bounds)]

    def _touches_edge(self, map_name: str, pocket: int, edge: str) -> bool:
        pieces = self.pieces(map_name)
        if not pieces:
            return True  # terrain unknown: do not invent a wall
        if pocket >= len(pieces):
            return False
        return any(_on_edge(tile, edge, self._bounds(map_name)) for tile in pieces[pocket])

    def _pockets_on_edge(self, map_name: str, edge: str) -> List[int]:
        pieces = self.pieces(map_name)
        if not pieces:
            return [WHOLE_MAP]
        bounds = self._bounds(map_name)
        return [
            index
            for index, piece in enumerate(pieces)
            if any(_on_edge(tile, edge, bounds) for tile in piece)
        ]

    def _bounds(self, map_name: str) -> Tuple[int, int]:
        size = self._size_for(map_name)
        if size:
            return int(size[0]), int(size[1])
        pieces = self.pieces(map_name)
        tiles = [tile for piece in pieces for tile in piece]
        width = max((x for x, _ in tiles), default=-1) + 1
        height = max((y for _, y in tiles), default=-1) + 1
        return width, height

    def _landing(self, map_name: str, warp_index) -> Optional[Coord]:
        """Where a warp puts you down, from the destination map's own warp list."""
        if warp_index is None:
            return None
        warps = self._warps_for(map_name)
        try:
            return _coord(warps[int(warp_index)])
        except (IndexError, TypeError, ValueError):
            return None

    def route(
        self,
        src_map: str,
        src_coord: Optional[Coord],
        dst_map: str,
        dst_coord: Optional[Coord] = None,
    ) -> Optional[Tuple[PocketHop, ...]]:
        """Fewest warps from one pocket to another, or None if there is no way.

        Without `dst_coord` any pocket of `dst_map` counts as arrival, which is
        the right reading of "get me to Cerulean" and the wrong one of "get me
        to that tile".
        """
        start_pocket = self.pocket_at(src_map, src_coord)
        if start_pocket is None:
            return None
        target_pocket = self.pocket_at(dst_map, dst_coord) if dst_coord is not None else None
        if dst_coord is not None and target_pocket is None:
            return None

        start = (src_map, start_pocket)
        if start[0] == dst_map and (target_pocket is None or start[1] == target_pocket):
            return ()

        previous: Dict[Tuple[str, int], Optional[PocketHop]] = {start: None}
        queue: deque = deque([start])
        while queue:
            current = queue.popleft()
            for hop in self.hops_from(*current):
                node = (hop.to_map, hop.to_pocket)
                if node in previous:
                    continue
                previous[node] = hop
                if node[0] == dst_map and (target_pocket is None or node[1] == target_pocket):
                    return _unwind(previous, node)
                queue.append(node)
        return None


def _unwind(previous, node) -> Tuple[PocketHop, ...]:
    hops: List[PocketHop] = []
    cursor = node
    while cursor is not None:
        hop = previous.get(cursor)
        if hop is None:
            break
        hops.append(hop)
        cursor = (hop.from_map, hop.from_pocket)
    hops.reverse()
    return tuple(hops)


def _on_edge(tile: Coord, edge: str, bounds: Tuple[int, int]) -> bool:
    width, height = bounds
    x, y = tile
    if edge == "north":
        return y == 0
    if edge == "south":
        return y == height - 1
    if edge == "west":
        return x == 0
    return x == width - 1


def _connection_spec(edge: str, value) -> Tuple[Optional[str], Optional[mapdecode.MapConnection]]:
    """Split a connection into its target map and its offsets, if it has any.

    `gamedata`'s table gives a bare map name, and that is still accepted: this
    graph has to answer for maps nobody has stood on, whose header nobody has
    read. `mapdecode.connection_specs` gives the same name with the alignments
    attached, and only then can a landing tile be named.
    """
    if isinstance(value, str):
        return (value or None), None
    if not isinstance(value, dict):
        return None, None
    target = value.get("to_map") or None
    try:
        return target, mapdecode.MapConnection.from_spec(edge, value)
    except (KeyError, TypeError, ValueError):
        return target, None


def _coord(value) -> Optional[Coord]:
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        return None
    try:
        return int(x), int(y)
    except (TypeError, ValueError):
        return None
