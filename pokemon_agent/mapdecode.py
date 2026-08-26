"""The whole current map's terrain, read out of WRAM rather than off the screen.

Until now the only source of walkability was `game_area_collision()`, the 9x10
window around the player. Everything downstream inherited that: the explored map
was a mosaic of screen scraps, tile-pair seams were known only where the player
was standing, and a tile corrected as solid went back to being walkable as soon
as it left the window. A scripted player handed a *perfect* map of Mt Moon 1F
still could not walk an 89-step leg, because the seams outside the window read
as open corridor.

Gen 1 keeps the loaded map's block layout in `wOverworldMap`, so the whole floor
is already in memory. Each block is 4x4 background tiles and covers 2x2 of the
player's coordinate grid; the engine's collision check looks at the lower-left
tile of each quadrant against the tileset's walkable-tile list.

Two details cost hours to find and are worth stating:

* The collision list pointer is a bank-0 address even though `wTilesetBank`
  names the tileset's *graphics* bank. Reading the list through that bank
  returns plausible garbage: Mt Moon decoded almost correctly and every
  outdoor map decoded to zero walkable tiles.
* Which tile of the block is the collision tile was settled by measurement, not
  by reading the disassembly. Of the sixteen candidates, `row 1, column 0`
  scores 7 misses in 806 tiles the player physically walked and 4 false
  positives in 1,065 tiles it saw and found solid, across two tilesets. The
  next best is off by two orders of magnitude.
* The output has to speak the units the rest of the codebase already speaks.
  The first version returned raw pokered tile ids and a numeric tileset; every
  consumer expects PyBoy tilemap ids (offset by 0x100) and a tileset *name*.
  Nothing raised. `movement_edges` skips its whole ledge and seam block when
  the tileset is not a string, and the tile-pair tables just match nothing. So
  Route 4 reported zero ledges when it has 169, and Mt Moon 1F reported zero
  seams when it has 131. An empty answer that reads as a fact, which is the
  failure this project keeps paying for. Terrain was exact the whole time; the
  units were not.

Verified by walking it. An emulator flood from Route 4 (24,6) -- press a real
button, read wXCoord/wYCoord, repeat -- reaches 467 tiles out to x=89 and steps
off the east edge into Cerulean City. Against that, this decoder's walkable set
has zero false positives and zero false negatives. The two floors of Route 4
are joined only by one-way ledge hops at row 9, which is why they look like
separate pockets until ledges are switched on.

The same header also says where the map's four sides lead and, unlike
`gamedata`'s connection table, at what offset -- see `decode_connections`. That
too was placed by measurement: the connection block lives at 0xD370, not up by
the warps, and every one of the 21 maps reachable from a save state agrees with
`gamedata` on its directions, target maps and connected widths there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

from pokemon_agent.navigation import TILE_ID_OFFSET

Coord = Tuple[int, int]

WCURMAP = 0xD35E
WCURMAPTILESET = 0xD367
WCURMAPHEIGHT = 0xD368  # in blocks
WCURMAPWIDTH = 0xD369  # in blocks
WOVERWORLDMAP = 0xC6E8
WTILESETBANK = 0xD52B
WTILESETBLOCKS = 0xD52C
WTILESETCOLL = 0xD530
WNUMBEROFWARPS = 0xD3AE
WWARPENTRIES = 0xD3AF  # y, x, destination warp id, destination map

#: Which of the four sides have a connected map, as a bitmask.
WMAPCONNECTIONS = 0xD370
CONNECTION_BITS = {"east": 0x01, "west": 0x02, "south": 0x04, "north": 0x08}

#: Each direction owns a fixed 11-byte slot; the engine fills only the sides the
#: bitmask names, so an unlisted slot holds whatever the last map left there.
CONNECTION_SLOTS = {"north": 0xD371, "south": 0xD37C, "west": 0xD387, "east": 0xD392}
CONNECTION_SLOT_SIZE = 11

#: Three blocks of border on every side of the loaded map.
MAP_BORDER = 3

#: A block is 4x4 tiles; the collision tile of each 2x2 quadrant is this one.
COLLISION_ROW = 1
COLLISION_COL = 0

#: An exit warp names LAST_MAP rather than a map, because in-game it returns the
#: player wherever they came from. Statically it is still one fixed door.
LAST_MAP = 0xFF

#: A collision list longer than this means the pointer is wrong, not that the
#: tileset has that many walkable tiles. Fail loudly instead of scanning ROM.
MAX_COLLISION_TILES = 512


class DecodeFailed(RuntimeError):
    """The reads did not describe a map, so nothing here should be believed."""


def decode_map(read_u8: Callable[[int], int], read_rom: Callable[[int, int], int]) -> dict:
    """Walkable terrain and tile ids for the whole loaded map.

    `read_u8` reads WRAM; `read_rom` reads a banked ROM address. The split
    matters because the two pointers this needs live in different places, which
    is the bug described in the module docstring.
    """
    width_blocks, height_blocks = read_u8(WCURMAPWIDTH), read_u8(WCURMAPHEIGHT)
    if not width_blocks or not height_blocks:
        raise DecodeFailed("map dimensions read as zero; no map is loaded")

    bank = read_u8(WTILESETBANK)
    blocks_ptr = _read_u16(read_u8, WTILESETBLOCKS)
    walkable_tiles = _collision_list(read_u8, _read_u16(read_u8, WTILESETCOLL))

    stride = width_blocks + MAP_BORDER * 2
    walkable: Set[Coord] = set()
    tile_ids: Dict[Coord, int] = {}
    blocks: Dict[int, list[int]] = {}

    for block_y in range(height_blocks):
        for block_x in range(width_blocks):
            offset = (block_y + MAP_BORDER) * stride + block_x + MAP_BORDER
            index = read_u8(WOVERWORLDMAP + offset)
            block = blocks.get(index)
            if block is None:
                base = blocks_ptr + index * 16
                block = [read_rom(bank, base + i) for i in range(16)]
                blocks[index] = block
            for sub_y in (0, 1):
                for sub_x in (0, 1):
                    tile = block[(sub_y * 2 + COLLISION_ROW) * 4 + sub_x * 2 + COLLISION_COL]
                    coord = (block_x * 2 + sub_x, block_y * 2 + sub_y)
                    # Shifted into the same space the ledge and tile-pair tables
                    # use. They were written against PyBoy's tilemap, which
                    # offsets every id by 0x100, and a raw pokered id matches
                    # nothing in them -- silently, as an empty set of ledges.
                    tile_ids[coord] = tile + TILE_ID_OFFSET
                    if tile in walkable_tiles:
                        walkable.add(coord)

    # Imported here rather than at module scope: `memory.red` reaches back to
    # `emulator`, which imports this module, and a top-level import closes the
    # cycle.
    from pokemon_agent.memory.red import TILESET_NAMES

    tileset_id = read_u8(WCURMAPTILESET)
    return {
        "map_id": read_u8(WCURMAP),
        # The NAME, because `world.movement_edges` gates its whole ledge and
        # seam block on the tileset being a string and skips it otherwise.
        "tileset": TILESET_NAMES.get(tileset_id, f"UNKNOWN_TILESET({tileset_id})"),
        "tileset_id": tileset_id,
        "width": width_blocks * 2,
        "height": height_blocks * 2,
        "walkable": walkable,
        "tile_ids": tile_ids,
    }


def decode_warps(read_u8: Callable[[int], int]) -> list[dict]:
    """Every warp on the loaded map, with where it leads.

    The destination is a map id and a warp *index* on that map, so the landing
    coordinate is only knowable once the destination map has been decoded too.
    `resolve_last_map` handles the exit warps that name LAST_MAP instead.
    """
    out = []
    for index in range(read_u8(WNUMBEROFWARPS)):
        base = WWARPENTRIES + index * 4
        y, x, dest_warp, dest_map = (read_u8(base + offset) for offset in range(4))
        out.append({"index": index, "at": (x, y), "dest_warp": dest_warp, "dest_map": dest_map})
    return out


def resolve_last_map(map_id: int, warp: dict, warps_by_map: dict[int, list[dict]]):
    """Turn a LAST_MAP destination into a real one, or (None, None).

    An exit warp says "back where you came from", which is not a map. It is
    still one fixed door, so the answer is whichever warp points back at this
    one. Without this a route graph can enter Mt Moon and never leave.
    """
    if warp["dest_map"] != LAST_MAP:
        return warp["dest_map"], warp["dest_warp"]
    for other_id, others in warps_by_map.items():
        for other in others:
            if other["dest_map"] == map_id and other["dest_warp"] == warp["index"]:
                return other_id, other["index"]
    return None, None


@dataclass(frozen=True)
class MapConnection:
    """One side of the loaded map, and where stepping off it puts you.

    The engine does not search for a landing tile: walking off an edge sets one
    coordinate to a stored alignment and shifts the other by a stored delta, so
    a whole edge slides onto the connected map at a fixed offset. Reproducing
    that is the difference between knowing which pocket you arrive in and
    guessing (see `pockets._connection_hops`).

    `strip_from`/`strip_to` are the tiles of *this* map's edge the connection
    actually covers, along the edge's axis (x for north/south, y for east and
    west). They can be negative, because the strip is positioned in the block
    map including its three-block border, and it may start inside that border.
    Outside that range the edge is border blocks and there is nothing to walk
    onto: Route 4's south edge is a 13-block strip at its west end, so its far
    east corner -- x 81..89, sixty tiles away -- connects to nothing at all.
    """

    direction: str
    map_id: int
    y_align: int  # signed
    x_align: int  # signed
    strip_length: int  # in blocks, as stored
    connected_width: int  # in player tiles, doubled from the header's blocks
    strip_from: int  # first tile of this map's edge the strip covers
    strip_to: int  # last, inclusive

    @classmethod
    def from_spec(cls, direction: str, spec: dict) -> "MapConnection":
        """Rebuild one from the plain dict `connection_specs` hands out.

        The routing graph carries these as data through several layers, and
        keeping the arithmetic in one place is worth the round trip.
        """
        strip_from, strip_to = spec["strip"]
        return cls(
            direction=direction,
            map_id=int(spec.get("map_id", 0)),
            y_align=int(spec["y_align"]),
            x_align=int(spec["x_align"]),
            strip_length=int(spec.get("strip_length", 0)),
            connected_width=int(spec["connected_width"]),
            strip_from=int(strip_from),
            strip_to=int(strip_to),
        )

    def landing(self, coord: Coord) -> Optional[Coord]:
        """Where walking off this edge at `coord` puts you, or None.

        None means `coord` is not on the connected strip, so the edge there is
        border and the step is into a wall.

        The connected map's *height* is not in the header, so a landing below
        it is not rejected here. Whoever knows that map's terrain rejects it,
        by finding no walkable tile there.
        """
        x, y = coord
        along = x if self.direction in ("north", "south") else y
        if not self.strip_from <= along <= self.strip_to:
            return None
        if self.direction in ("north", "south"):
            landed = (x + self.x_align, self.y_align)
        else:
            landed = (self.x_align, y + self.y_align)
        if landed[0] < 0 or landed[1] < 0:
            return None
        if landed[0] >= self.connected_width:
            return None
        return landed


def decode_connections(read_u8: Callable[[int], int]) -> Dict[str, MapConnection]:
    """The loaded map's connections, keyed by the edge you walk off.

    Only the sides named by the bitmask are read. The other slots are stale --
    Route 4's north slot still holds Route 3's north connection with a poisoned
    first byte -- so trusting a slot because its map id looks plausible invents
    a connection that is not there.
    """
    mask = read_u8(WMAPCONNECTIONS)
    width_blocks = read_u8(WCURMAPWIDTH)
    stride = width_blocks + MAP_BORDER * 2
    out: Dict[str, MapConnection] = {}
    for direction, bit in CONNECTION_BITS.items():
        if not mask & bit:
            continue
        base = CONNECTION_SLOTS[direction]
        strip_dest = _read_u16(read_u8, base + 3)
        strip_length = read_u8(base + 5)
        # The strip's position on this map is where the engine writes it into
        # the block map, so the offset from wOverworldMap gives the column (for
        # a north/south strip) or the row (east/west) it starts at, border
        # included. Two blocks per player tile.
        offset = strip_dest - WOVERWORLDMAP
        row, column = divmod(offset, stride) if stride else (0, 0)
        start_block = (column if direction in ("north", "south") else row) - MAP_BORDER
        out[direction] = MapConnection(
            direction=direction,
            map_id=read_u8(base),
            y_align=_signed(read_u8(base + 7)),
            x_align=_signed(read_u8(base + 8)),
            strip_length=strip_length,
            connected_width=read_u8(base + 6) * 2,
            strip_from=start_block * 2,
            strip_to=(start_block + strip_length) * 2 - 1,
        )
    return out


def connection_specs(
    connections: Dict[str, MapConnection],
    name_for_id: Callable[[int], Optional[str]],
) -> Dict[str, dict]:
    """Connections in the shape `pockets.PocketGraph` takes for `connections_for`.

    The graph routes over map *names*, so the ids have to be translated here;
    a connection whose id has no name is dropped rather than routed to "???".
    """
    out: Dict[str, dict] = {}
    for direction, connection in connections.items():
        name = name_for_id(connection.map_id)
        if not name:
            continue
        out[direction] = {
            "to_map": name,
            "map_id": connection.map_id,
            "y_align": connection.y_align,
            "x_align": connection.x_align,
            "strip": (connection.strip_from, connection.strip_to),
            "strip_length": connection.strip_length,
            "connected_width": connection.connected_width,
        }
    return out


def _signed(value: int) -> int:
    """An alignment is a two's-complement byte: Route 3 walks north at x - 50."""
    return value - 256 if value > 127 else value


def _read_u16(read_u8: Callable[[int], int], addr: int) -> int:
    return read_u8(addr) | (read_u8(addr + 1) << 8)


def _collision_list(read_u8: Callable[[int], int], pointer: int) -> Set[int]:
    """The tileset's walkable tile ids, terminated by 0xFF.

    Read from bank 0, not from `wTilesetBank`. See the module docstring.
    """
    tiles: Set[int] = set()
    addr = pointer
    while True:
        value = read_u8(addr)
        if value == 0xFF:
            return tiles
        tiles.add(value)
        addr += 1
        if addr - pointer > MAX_COLLISION_TILES:
            raise DecodeFailed(
                f"collision list at 0x{pointer:04X} did not terminate within "
                f"{MAX_COLLISION_TILES} bytes; the pointer is wrong"
            )
