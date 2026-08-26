"""The whole-map terrain decoder.

The fake memory here is a two-block map: one block whose collision tile is
walkable everywhere, one whose is not. That is enough to pin the two things
that were actually wrong in the field -- which bank the collision list is read
from, and which tile of a block the engine checks.
"""

import pytest

from pokemon_agent import mapdecode


def build_memory(
    *,
    width_blocks=2,
    height_blocks=1,
    layout=(0, 1),
    collision_tiles=(0x10,),
    blocks=None,
    bank=0x1B,
    blocks_ptr=0x5000,
    coll_ptr=0x17AC,
):
    """A WRAM/ROM pair shaped like the real thing, with one map loaded."""
    wram = {
        mapdecode.WCURMAP: 59,
        mapdecode.WCURMAPTILESET: 17,
        mapdecode.WCURMAPWIDTH: width_blocks,
        mapdecode.WCURMAPHEIGHT: height_blocks,
        mapdecode.WTILESETBANK: bank,
        mapdecode.WTILESETBLOCKS: blocks_ptr & 0xFF,
        mapdecode.WTILESETBLOCKS + 1: blocks_ptr >> 8,
        mapdecode.WTILESETCOLL: coll_ptr & 0xFF,
        mapdecode.WTILESETCOLL + 1: coll_ptr >> 8,
    }
    for offset, tile in enumerate(list(collision_tiles) + [0xFF]):
        wram[coll_ptr + offset] = tile

    stride = width_blocks + mapdecode.MAP_BORDER * 2
    for block_y in range(height_blocks):
        for block_x in range(width_blocks):
            offset = (block_y + mapdecode.MAP_BORDER) * stride + block_x + mapdecode.MAP_BORDER
            wram[mapdecode.WOVERWORLDMAP + offset] = layout[block_y * width_blocks + block_x]

    # Block 0 is walkable in all four quadrants, block 1 in none, and both carry
    # a distinct decoy in every other position so a wrong tile choice shows up.
    blocks = blocks or {0: [0xAA] * 16, 1: [0xBB] * 16}
    for index, block in blocks.items():
        if index == 0 and block == [0xAA] * 16:
            block = list(block)
            for sub_y in (0, 1):
                for sub_x in (0, 1):
                    block[(sub_y * 2 + 1) * 4 + sub_x * 2] = 0x10
            blocks[index] = block

    rom = {}
    for index, block in blocks.items():
        for i, value in enumerate(block):
            rom[(bank, blocks_ptr + index * 16 + i)] = value

    return (lambda addr: wram.get(addr, 0)), (lambda b, addr: rom.get((b, addr), 0)), wram, rom


def test_decodes_the_whole_map_not_just_a_window():
    read_u8, read_rom, _, _ = build_memory()

    got = mapdecode.decode_map(read_u8, read_rom)

    assert (got["width"], got["height"]) == (4, 2), "two blocks wide is four player tiles"
    # Block 0 fills x 0..1, block 1 fills x 2..3.
    assert got["walkable"] == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert len(got["tile_ids"]) == 8, "every tile has an id, walkable or not"


def test_collision_list_is_read_from_bank_zero():
    """The bug that made every outdoor map decode to zero walkable tiles.

    `wTilesetBank` names the tileset's graphics bank, but the collision list
    pointer is a bank-0 address. Reading it through the graphics bank returns
    garbage that happened to look plausible for one tileset.
    """
    read_u8, read_rom, wram, rom = build_memory()
    # Poison the same address in the tileset's bank. A decoder reading the list
    # from there sees a different, wrong walkable set.
    rom[(0x1B, 0x17AC)] = 0xBB
    rom[(0x1B, 0x17AD)] = 0xFF

    got = mapdecode.decode_map(read_u8, read_rom)

    assert got["walkable"] == {(0, 0), (1, 0), (0, 1), (1, 1)}, "bank 0 is the list that counts"


def test_the_collision_tile_is_the_lower_left_of_each_quadrant():
    """Settled by measurement: 7 misses in 806 walked tiles, next best is far worse."""
    walkable_here = [0x00] * 16
    walkable_here[(0 * 2 + 1) * 4 + 0] = 0x10  # lower-left of the top-left quadrant
    read_u8, read_rom, _, _ = build_memory(width_blocks=1, layout=(0,), blocks={0: walkable_here})

    got = mapdecode.decode_map(read_u8, read_rom)

    assert got["walkable"] == {(0, 0)}, "only the quadrant whose lower-left tile is walkable"


def test_a_collision_list_that_never_terminates_is_an_error_not_a_scan():
    read_u8, read_rom, wram, _ = build_memory()
    del wram[0x17AC + 1]  # drop the 0xFF terminator; the default read returns 0

    with pytest.raises(mapdecode.DecodeFailed, match="the pointer is wrong"):
        mapdecode.decode_map(read_u8, read_rom)


def test_no_map_loaded_is_refused_rather_than_decoded_as_empty():
    read_u8, read_rom, wram, _ = build_memory()
    wram[mapdecode.WCURMAPWIDTH] = 0

    with pytest.raises(mapdecode.DecodeFailed, match="no map is loaded"):
        mapdecode.decode_map(read_u8, read_rom)


# ---------------------------------------------------------------------------
# Warps
# ---------------------------------------------------------------------------


def warp_memory(entries):
    wram = {mapdecode.WNUMBEROFWARPS: len(entries)}
    for index, (x, y, dest_warp, dest_map) in enumerate(entries):
        base = mapdecode.WWARPENTRIES + index * 4
        wram[base], wram[base + 1] = y, x
        wram[base + 2], wram[base + 3] = dest_warp, dest_map
    return lambda addr: wram.get(addr, 0)


def test_warps_carry_destination_map_and_warp_index():
    read_u8 = warp_memory([(5, 5, 2, 59), (17, 11, 0, 61)])

    got = mapdecode.decode_warps(read_u8)

    assert got[0] == {"index": 0, "at": (5, 5), "dest_warp": 2, "dest_map": 59}
    assert got[1]["dest_map"] == 61


def test_an_exit_warp_resolves_to_whatever_points_back_at_it():
    """Without this a route graph can enter Mt Moon and never leave.

    Exit warps name LAST_MAP because in-game they return you where you came
    from. Statically the door is fixed: it is whichever warp points back here.
    """
    warps_by_map = {
        15: [{"index": 0, "at": (18, 5), "dest_warp": 1, "dest_map": 59}],
        59: [
            {"index": 0, "at": (0, 0), "dest_warp": 0, "dest_map": 60},
            {"index": 1, "at": (14, 35), "dest_warp": 0, "dest_map": mapdecode.LAST_MAP},
        ],
    }

    dest_map, dest_warp = mapdecode.resolve_last_map(59, warps_by_map[59][1], warps_by_map)

    assert (dest_map, dest_warp) == (15, 0), "the way out of Mt Moon 1F is Route 4 warp 0"


def test_an_unpaired_exit_warp_resolves_to_nothing_rather_than_guessing():
    warps_by_map = {
        59: [{"index": 0, "at": (1, 1), "dest_warp": 0, "dest_map": mapdecode.LAST_MAP}]
    }

    assert mapdecode.resolve_last_map(59, warps_by_map[59][0], warps_by_map) == (None, None)


def test_an_ordinary_warp_passes_through_resolution_untouched():
    warp = {"index": 0, "at": (5, 5), "dest_warp": 2, "dest_map": 59}

    assert mapdecode.resolve_last_map(60, warp, {}) == (59, 2)
