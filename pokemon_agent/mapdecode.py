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
"""

from __future__ import annotations

from typing import Callable, Dict, Set, Tuple

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
                    tile_ids[coord] = tile
                    if tile in walkable_tiles:
                        walkable.add(coord)

    return {
        "map_id": read_u8(WCURMAP),
        "tileset": read_u8(WCURMAPTILESET),
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
