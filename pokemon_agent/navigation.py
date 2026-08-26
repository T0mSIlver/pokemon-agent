"""Live navigation snapshot models for Pokemon Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple

Coord = Tuple[int, int]
MAP_COORDINATE_SYSTEM = "map_tile_absolute"
MAP_COORDINATE_NOTE = (
    "All x/y values are absolute map tile coordinates. "
    "In the annotated frame, columns are x and rows are y."
)

#: PyBoy reports background tiles through the signed 0x8800 addressing window,
#: so every tile id it hands us is the pokered tile id plus this offset.
TILE_ID_OFFSET = 0x100

#: pokered ``data/tilesets/tile_pair_collisions.asm``, both the land and the
#: water table. Two adjacent tiles named here cannot be walked between *in
#: either direction* — ``CheckForTilePairCollisions`` tries the pair both ways
#: round — even though the collision map calls both of them passable. It is a
#: property of the seam between two tiles, not of either tile, which is why no
#: per-tile walkability grid can hold it.
#:
#: The table only has CAVERN and FOREST entries, so this rule is inert on the
#: overworld and decisive inside a cave: in Mt. Moon it is the whole of what
#: separates the upper floor from the lower one.
TILE_PAIR_BLOCKERS: dict[str, set[frozenset[int]]] = {
    "CAVERN": {
        frozenset((288, 261)),
        frozenset((321, 261)),
        frozenset((298, 261)),
        frozenset((261, 289)),
        frozenset((276, 261)),
    },
    "FOREST": {
        frozenset((304, 302)),
        frozenset((338, 302)),
        frozenset((341, 302)),
        frozenset((342, 302)),
        frozenset((288, 302)),
        frozenset((350, 302)),
        frozenset((351, 302)),
        frozenset((276, 302)),
        frozenset((328, 302)),
    },
}


#: pokered ``data/tilesets/ledge_tiles.asm``, keyed by the direction pressed and
#: holding ``(tile the player stands on, tile being jumped)`` pairs. Ledges are
#: directional and OVERWORLD-only: ``engine/overworld/ledges.asm`` returns early
#: for every other tileset, and the table has no upward entry at all.
LEDGE_TILE_PAIRS: dict[str, dict[str, set[tuple[int, int]]]] = {
    "OVERWORLD": {
        "down": {
            (0x2C + TILE_ID_OFFSET, 0x37 + TILE_ID_OFFSET),
            (0x39 + TILE_ID_OFFSET, 0x36 + TILE_ID_OFFSET),
            (0x39 + TILE_ID_OFFSET, 0x37 + TILE_ID_OFFSET),
        },
        "left": {
            (0x2C + TILE_ID_OFFSET, 0x27 + TILE_ID_OFFSET),
            (0x39 + TILE_ID_OFFSET, 0x27 + TILE_ID_OFFSET),
        },
        "right": {
            (0x2C + TILE_ID_OFFSET, 0x0D + TILE_ID_OFFSET),
            (0x2C + TILE_ID_OFFSET, 0x1D + TILE_ID_OFFSET),
            (0x39 + TILE_ID_OFFSET, 0x0D + TILE_ID_OFFSET),
        },
    },
}

#: pokered ``data/tilesets/warp_carpet_tile_ids.asm``. Walking into one of these
#: while standing on a warp entry fires the warp; every other blocked direction
#: is an ordinary wall. Used by ``IsWarpTileInFrontOfPlayer``.
WARP_CARPET_TILES: dict[str, set[int]] = {
    "down": {tile + TILE_ID_OFFSET for tile in (0x01, 0x12, 0x17, 0x3D, 0x04, 0x18, 0x33)},
    "up": {tile + TILE_ID_OFFSET for tile in (0x01, 0x5C)},
    "left": {tile + TILE_ID_OFFSET for tile in (0x1A, 0x4B)},
    "right": {tile + TILE_ID_OFFSET for tile in (0x0F, 0x4E)},
}

#: Tilesets whose warps are checked with ``IsWarpTileInFrontOfPlayer`` rather
#: than ``IsPlayerFacingEdgeOfMap`` (pokered ``ExtraWarpCheck``).
WARP_FRONT_TILE_TILESETS = frozenset({"OVERWORLD", "SHIP", "SHIP_PORT", "PLATEAU"})

#: ``ExtraWarpCheck`` overrides its tileset rule for five maps: Rocket Hideout
#: B1F/B2F/B4F and Rock Tunnel 1F use the front-tile rule despite their indoor
#: tilesets, and S.S. Anne 3F uses the map-edge rule despite the SHIP tileset.
WARP_FRONT_TILE_MAPS = frozenset({0x52, 0xC7, 0xC8, 0xCA})
WARP_MAP_EDGE_MAPS = frozenset({0x61})

MOVE_DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")


def location_key(map_id: int, map_name: str) -> str:
    """Build a stable key for per-location navigation data."""
    return f"{map_id}:{map_name}"


def _coord_dict(coord: Optional[Coord]) -> Optional[Dict[str, int]]:
    if coord is None:
        return None
    return {"x": coord[0], "y": coord[1]}


def terrain_dict(truth: Optional[Mapping[str, object]]) -> Optional[Dict[str, object]]:
    """A decoded floor in a shape `json.dumps` accepts, or None if there is none.

    `mapdecode.decode_map` answers in Python sets and coordinate-keyed dicts,
    and this payload is written to disk and broadcast over a WebSocket, where
    neither survives: a tuple key raises before anything is sent. So walkable
    ground travels as ``[[x, y], ...]`` and tile ids as one row of ints per y,
    which `world.movement_edges` and `capabilities.collision_from` both read.

    Rows rather than a coordinate map because the whole floor now travels and
    the difference is not small: 40x36 Mt. Moon 1F is 1440 tile ids, and as
    ``{"x,y": id}`` pairs that is four times the bytes of the same ints in rows.
    """
    if not truth:
        return None
    width, height = int(truth.get("width") or 0), int(truth.get("height") or 0)
    if not width or not height:
        return None
    tile_ids = truth.get("tile_ids") or {}
    rows: List[List[int]] = []
    if isinstance(tile_ids, Mapping):
        rows = [[int(tile_ids.get((x, y), 0)) for x in range(width)] for y in range(height)]
    elif isinstance(tile_ids, list):
        # Already rows. Serialising a payload twice must not empty it out.
        rows = [[int(tile) for tile in row] for row in tile_ids]
    payload: Dict[str, object] = {
        "map_id": truth.get("map_id"),
        "tileset": truth.get("tileset"),
        "width": width,
        "height": height,
        "walkable": sorted([int(x), int(y)] for x, y in truth.get("walkable") or ()),
        "tile_ids": rows,
    }
    for key in ("warps", "connections"):
        if truth.get(key) is not None:
            payload[key] = truth[key]
    return payload


def _ascii_header(min_x: int, max_x: int) -> str:
    return "     " + "".join(str(x % 10) for x in range(min_x, max_x + 1))


def tile_pair_allows(
    tileset: Optional[str],
    tile_a: Optional[int],
    tile_b: Optional[int],
) -> bool:
    """Return whether movement between two adjacent tiles is allowed."""
    if tileset is None or tile_a is None or tile_b is None:
        return True
    blocked_pairs = TILE_PAIR_BLOCKERS.get(tileset)
    if not blocked_pairs:
        return True
    return frozenset((tile_a, tile_b)) not in blocked_pairs


def tile_pair_blocked_edges(
    tileset: Optional[str],
    tile_ids: Mapping[Coord, int],
) -> Set[Tuple[Coord, Coord]]:
    """Every seam in *tile_ids* the tileset refuses to let you cross.

    Returned as canonical ``(lower, higher)`` coordinate pairs, one per seam,
    because the rule is symmetric: the game tries the pair both ways round, so
    neither tile is the one doing the blocking.

    Only right and down neighbours are visited — that reaches every adjacent
    pair inside the window exactly once. Empty for every tileset with no entry
    in `TILE_PAIR_BLOCKERS`, which is every tileset but CAVERN and FOREST.
    """
    seams: Set[Tuple[Coord, Coord]] = set()
    if not TILE_PAIR_BLOCKERS.get(tileset or ""):
        return seams
    for coord, tile in tile_ids.items():
        x, y = coord
        for neighbour in ((x + 1, y), (x, y + 1)):
            other = tile_ids.get(neighbour)
            if other is None or tile_pair_allows(tileset, tile, other):
                continue
            seams.add((coord, neighbour))
    return seams


def ledge_hop_allows(
    tileset: Optional[str],
    direction: str,
    standing_tile: Optional[int],
    ledge_tile: Optional[int],
) -> bool:
    """Return whether pressing *direction* jumps a ledge instead of colliding.

    A ledge cell reads as blocked in the collision map, but ``HandleLedges``
    runs before the collision check, so the jump wins.
    """
    if tileset is None or standing_tile is None or ledge_tile is None:
        return False
    pairs = LEDGE_TILE_PAIRS.get(tileset, {}).get(direction)
    if not pairs:
        return False
    return (standing_tile, ledge_tile) in pairs


def ledge_landing(coord: Coord, direction: str) -> Coord:
    """Where a ledge hop in *direction* puts the player: two tiles, not one."""
    x, y = coord
    delta = {"up": (0, -2), "down": (0, 2), "left": (-2, 0), "right": (2, 0)}[direction]
    return (x + delta[0], y + delta[1])


def warp_uses_front_tile_rule(tileset: Optional[str], map_id: Optional[int]) -> bool:
    """Which of ``ExtraWarpCheck``'s two rules governs this map's warps."""
    if map_id in WARP_MAP_EDGE_MAPS:
        return False
    if map_id in WARP_FRONT_TILE_MAPS:
        return True
    return tileset in WARP_FRONT_TILE_TILESETS


def facing_edge_of_map(
    coord: Coord,
    direction: str,
    map_dimensions: Optional[Dict[str, int]],
) -> Optional[bool]:
    """Whether the player faces the outer edge of the map, or None if unknown."""
    if not map_dimensions:
        return None
    width = map_dimensions.get("width_tiles", map_dimensions.get("width"))
    height = map_dimensions.get("height_tiles", map_dimensions.get("height"))
    if width is None or height is None:
        return None
    x, y = coord
    if direction == "up":
        return y == 0
    if direction == "down":
        return y == int(height) - 1
    if direction == "left":
        return x == 0
    if direction == "right":
        return x == int(width) - 1
    return None


@dataclass(slots=True)
class LiveNavigationSnapshot:
    """Live navigation state derived from the current emulator frame."""

    map_id: int
    map_name: str
    player_position: Coord
    facing: str
    tileset: str
    window_top_left: Coord
    terrain: List[List[int]]
    sprite_positions: List[Coord] = field(default_factory=list)
    valid_moves: List[str] = field(default_factory=list)
    warps: List[Dict[str, int]] = field(default_factory=list)
    signs: List[Dict[str, int]] = field(default_factory=list)
    map_dimensions: Optional[Dict[str, int]] = None
    tile_ids: Dict[Coord, int] = field(default_factory=dict)
    interaction: Optional[Dict[str, object]] = None
    ledge_hops: Dict[str, Coord] = field(default_factory=dict)
    #: Adjacent tile pairs inside the window that cannot be walked between,
    #: either way, however passable both tiles look. See `TILE_PAIR_BLOCKERS`.
    blocked_pairs: List[Tuple[Coord, Coord]] = field(default_factory=list)
    warp_exit_directions: List[str] = field(default_factory=list)
    warp_exit_armed: bool = False
    warp_exit_note: Optional[str] = None
    #: The whole floor decoded out of WRAM, not the window. `terrain` above is
    #: the 10x9 screen; this is every tile. None on a frame that is not a map --
    #: a battle, a transition -- because an empty floor reads as "walled in"
    #: downstream, which is worse than saying nothing. `mapdecode` has
    #: produced this since it was written and nothing could read it, because it
    #: was assembled in `_movement_components` and then dropped on the way into
    #: this dataclass — it had no field to land in. Every consumer of ground
    #: truth reads `snapshot["map_terrain"]`, so the whole decoded-terrain path
    #: was inert: `collision_from` never took its ground-truth branch and the
    #: explored map never adopted a floor. Measured on `mt_moon_1f_entered`:
    #: 1144 walkable tiles decoded, 26 delivered.
    map_terrain: Optional[Dict[str, object]] = None

    @property
    def key(self) -> str:
        return location_key(self.map_id, self.map_name)

    @property
    def width(self) -> int:
        return len(self.terrain[0]) if self.terrain else 0

    @property
    def height(self) -> int:
        return len(self.terrain)

    @property
    def sprite_set(self) -> set[Coord]:
        return set(self.sprite_positions)

    def absolute_to_local(self, x: int, y: int) -> Optional[Coord]:
        local_x = x - self.window_top_left[0]
        local_y = y - self.window_top_left[1]
        if 0 <= local_x < self.width and 0 <= local_y < self.height:
            return local_x, local_y
        return None

    def local_to_absolute(self, local_x: int, local_y: int) -> Coord:
        return (
            self.window_top_left[0] + local_x,
            self.window_top_left[1] + local_y,
        )

    def render_window_ascii(self, goal: Optional[Coord] = None) -> str:
        """Render the current 9x10 live collision window as ASCII."""
        if not self.terrain:
            return "(no live collision data)"

        goal_local = None
        if goal is not None:
            goal_local = self.absolute_to_local(goal[0], goal[1])

        warp_set: set[Coord] = set()
        for warp in self.warps:
            wx = warp.get("x") if isinstance(warp, dict) else None
            wy = warp.get("y") if isinstance(warp, dict) else None
            if wx is None or wy is None:
                continue
            warp_set.add((int(wx), int(wy)))

        min_x = self.window_top_left[0]
        max_x = self.window_top_left[0] + self.width - 1
        lines = [_ascii_header(min_x, max_x)]
        for local_y, row in enumerate(self.terrain):
            chars: List[str] = []
            for local_x, tile in enumerate(row):
                absolute = self.local_to_absolute(local_x, local_y)
                if (local_x, local_y) == (4, 4):
                    chars.append("P")
                elif goal_local == (local_x, local_y):
                    chars.append("G")
                elif absolute in warp_set:
                    chars.append("W")
                elif absolute in self.sprite_set:
                    chars.append("S")
                elif tile:
                    chars.append(".")
                else:
                    chars.append("#")
            absolute_y = self.window_top_left[1] + local_y
            lines.append(f"{absolute_y:>4} " + "".join(chars))
        return "\n".join(lines)

    def to_dict(self, goal: Optional[Coord] = None) -> Dict[str, object]:
        return {
            "location_key": self.key,
            "map_id": self.map_id,
            "map_name": self.map_name,
            "coordinate_system": MAP_COORDINATE_SYSTEM,
            "coordinate_note": MAP_COORDINATE_NOTE,
            "player_position": _coord_dict(self.player_position),
            "facing": self.facing,
            "tileset": self.tileset,
            "window_top_left": _coord_dict(self.window_top_left),
            "window_size": {"width": self.width, "height": self.height},
            "terrain": self.terrain,
            "sprites": [_coord_dict(coord) for coord in self.sprite_positions],
            "valid_moves": self.valid_moves,
            "warps": self.warps,
            "signs": self.signs,
            "map_dimensions": self.map_dimensions,
            "interaction": self.interaction,
            "ledge_hops": {
                direction: _coord_dict(landing)
                for direction, landing in sorted(self.ledge_hops.items())
            },
            "blocked_pairs": [
                {"a": _coord_dict(first), "b": _coord_dict(second)}
                for first, second in self.blocked_pairs
            ],
            "warp_exit_directions": self.warp_exit_directions,
            "warp_exit_armed": self.warp_exit_armed,
            "warp_exit_note": self.warp_exit_note,
            # A whole floor, in a shape json accepts. It is kilobytes, and it
            # is not model-facing: `_observation_summary` picks what the agent
            # reads and never forwards this. Carrying it here rather than
            # threading it past `to_dict` is what stops a consumer silently
            # running on 10x9 windows again.
            "map_terrain": terrain_dict(self.map_terrain),
            "ascii": self.render_window_ascii(goal=goal),
            "ascii_legend": {
                "P": "player",
                "G": "goal",
                "W": "warp tile (step ONTO it, then walk in the exit direction)",
                "S": "visible sprite blocker",
                ".": "passable tile",
                "#": "blocked tile",
            },
            "movement_legend": {
                "ledge_hops": (
                    "One-way ledge jumps. Legal even though the tile reads as blocked, "
                    "and they land two tiles away, not one."
                ),
                "blocked_pairs": (
                    "Seams between two adjacent tiles that cannot be crossed either "
                    "way, even though both tiles are passable and the ASCII map shows "
                    "open floor across them. Cave floors are separated by these."
                ),
                "warp_exit_directions": (
                    "Directions that fire the warp the player is standing on. They are "
                    "not walkable steps, so they are not in valid_moves, and they only "
                    "fire while warp_exit_armed is true."
                ),
            },
        }
