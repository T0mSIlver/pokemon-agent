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


def test_output_speaks_the_units_every_consumer_already_speaks():
    """The bug that reported zero ledges on a map with 169 of them.

    `movement_edges` skips its entire ledge and seam block unless the tileset is
    a string, and the ledge and tile-pair tables hold PyBoy tilemap ids, which
    are pokered ids plus 0x100. Returning raw ids and a numeric tileset matched
    nothing and raised nothing: Route 4 came back with no ledges, Mt Moon 1F
    with no seams, and both empty answers read as facts.
    """
    from pokemon_agent.navigation import TILE_ID_OFFSET

    read_u8, read_rom, _, _ = build_memory()

    got = mapdecode.decode_map(read_u8, read_rom)

    assert got["tileset"] == "CAVERN", "a name, not the id 17"
    assert got["tileset_id"] == 17, "the raw id is still there for anyone who wants it"
    assert got["tile_ids"][(0, 0)] == 0x10 + TILE_ID_OFFSET
    assert all(tile >= TILE_ID_OFFSET for tile in got["tile_ids"].values())


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


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

#: The bytes at 0xD370 with each map loaded, copied off a save state rather than
#: composed, so a wrong slot address or field order fails these tests the way it
#: would fail the game. Every landing asserted below was also walked in PyBoy,
#: and came out tile for tile.
LIVE_HEADERS = {
    "Route 4": {  # 45 blocks wide; south to Route 3, east to Cerulean City
        "width_blocks": 45,
        "mask": 0x05,
        "slots": {
            "south": [0x0E, 0x6B, 0x42, 0x4C, 0xC9, 0x0D, 0x23, 0x00, 0x32, 0x12, 0xC7],
            "east": [0x03, 0x44, 0x48, 0x18, 0xC7, 0x0F, 0x14, 0x08, 0x00, 0x03, 0xC7],
        },
    },
    "Route 3": {  # 35 blocks wide; north to Route 4, west to Pewter City
        "width_blocks": 35,
        "mask": 0x0A,
        "slots": {
            "north": [0x0F, 0xFA, 0x44, 0x04, 0xC7, 0x0D, 0x2D, 0x11, 0xCE, 0xB4, 0xC8],
            "west": [0x02, 0x0B, 0x46, 0xE8, 0xC6, 0x0F, 0x14, 0x08, 0x27, 0x16, 0xC7],
        },
    },
}


def connection_memory(map_name, extra_slots=()):
    """WRAM as the header sits once `map_name` is loaded."""
    header = LIVE_HEADERS[map_name]
    wram = {
        mapdecode.WCURMAPWIDTH: header["width_blocks"],
        mapdecode.WMAPCONNECTIONS: header["mask"],
    }
    for direction, values in dict(header["slots"], **dict(extra_slots)).items():
        base = mapdecode.CONNECTION_SLOTS[direction]
        for offset, value in enumerate(values):
            wram[base + offset] = value
    return lambda addr: wram.get(addr, 0)


def test_only_the_directions_the_bitmask_names_are_read():
    """Route 4's north slot really does still hold Route 3's connection.

    The four slots are fixed and the engine fills only the sides the mask names,
    so a decoder that trusts a slot because its map id looks plausible invents an
    edge off the top of Route 4 that leads nowhere.
    """
    stale = list(LIVE_HEADERS["Route 3"]["slots"]["north"])
    read_u8 = connection_memory("Route 4", extra_slots=[("north", stale)])

    got = mapdecode.decode_connections(read_u8)

    assert sorted(got) == ["east", "south"], "the mask reads 0x05: east and south"


def test_walking_off_the_south_edge_keeps_x_and_takes_the_stored_row():
    """Measured: Route 4 (9,17) pressed down lands on Route 3 (59,0)."""
    got = mapdecode.decode_connections(connection_memory("Route 4"))["south"]

    assert got.map_id == 14, "Route 3"
    assert got.landing((9, 17)) == (59, 0)
    assert got.landing((19, 17)) == (69, 0), "the whole edge slides by the same offset"


def test_walking_off_the_east_edge_keeps_y_and_takes_the_stored_column():
    """Measured: Route 4 (89,10) pressed right lands on Cerulean City (0,18)."""
    got = mapdecode.decode_connections(connection_memory("Route 4"))["east"]

    assert got.map_id == 3, "Cerulean City"
    assert got.landing((89, 10)) == (0, 18)
    assert got.landing((89, 13)) == (0, 21), "which is a different pocket of Cerulean"


def test_an_alignment_over_127_is_a_step_backwards_rather_than_a_huge_one_forwards():
    """Route 3's north x alignment is 0xCE.

    Read unsigned that is 206 tiles east, off the end of a map 70 wide. Read
    signed it is -50, and Route 3 (59,0) walking north lands on Route 4 (9,17),
    which is where the emulator puts you.
    """
    got = mapdecode.decode_connections(connection_memory("Route 3"))["north"]

    assert (got.x_align, got.y_align) == (-50, 17)
    assert got.landing((59, 0)) == (9, 17)


def test_a_tile_off_the_connection_strip_lands_nowhere():
    """The bogus hop this whole change exists for.

    Route 4's south side is a 13-block strip at the west end. Its far east
    corner, x 81..89, is border, but taking every pocket that touches the edge
    offered `Route 4#4 -> Route 3` from sixty tiles away.
    """
    got = mapdecode.decode_connections(connection_memory("Route 4"))["south"]

    assert (got.strip_from, got.strip_to) == (-6, 19), "13 blocks, from three left of the map"
    assert [got.landing((x, 17)) for x in (86, 87, 88, 89)] == [None] * 4
    assert got.landing((4, 17)) == (54, 0), "the west end of the same edge still lands"


def test_a_landing_past_the_end_of_the_connected_map_is_refused():
    """Route 3 is 70 tiles wide, so nothing may land at x 90 on it."""
    got = mapdecode.decode_connections(connection_memory("Route 4"))["south"]
    over_long = mapdecode.MapConnection(**{**got.__dict__, "strip_to": 200})

    assert over_long.landing((40, 17)) is None, "on the strip, off the map"


def test_connection_specs_translate_ids_into_the_names_a_route_is_keyed_by():
    names = {14: "Route 3", 3: "Cerulean City"}

    got = mapdecode.connection_specs(
        mapdecode.decode_connections(connection_memory("Route 4")), names.get
    )

    assert got["south"]["to_map"] == "Route 3"
    assert mapdecode.MapConnection.from_spec("south", got["south"]).landing((9, 17)) == (59, 0)


def test_a_connection_to_a_map_with_no_name_is_dropped_rather_than_routed_to():
    got = mapdecode.connection_specs(
        mapdecode.decode_connections(connection_memory("Route 4")), lambda _id: None
    )

    assert got == {}


# ---------------------------------------------------------------------------
# Getting the decoded floor to the people who need it
# ---------------------------------------------------------------------------


def test_the_decoded_floor_survives_the_trip_to_a_collision_map():
    """Decoding it was never the problem — delivering it was.

    `map_terrain` was assembled in the emulator's `_movement_components` and
    then dropped: `LiveNavigationSnapshot` had no field for it, so `to_dict`
    could not carry it and every consumer read `snapshot["map_terrain"]` as
    None. Measured on a real save at Mt. Moon 1F's south entrance: 1144
    walkable tiles decoded, 26 delivered. So `collision_from` never once took
    its ground-truth branch and the explored map never adopted a floor —
    /goto planned every route on a 90-tile window of a 40x36 map.

    The wire shape is not the decoder's shape, because this payload is written
    to disk and broadcast: a set and a coordinate-keyed dict both raise before
    anything is sent. So it goes as pairs and rows, and this is the test that
    the two ends agree about that.
    """
    import json

    from pokemon_agent import capabilities
    from pokemon_agent.navigation import LiveNavigationSnapshot, terrain_dict

    read_u8, read_rom, _, _ = build_memory()
    decoded = mapdecode.decode_map(read_u8, read_rom)
    snapshot = LiveNavigationSnapshot(
        map_id=59,
        map_name="Mt Moon 1F",
        player_position=(0, 0),
        facing="down",
        tileset="CAVERN",
        window_top_left=(-4, -4),
        terrain=[[0] * 10 for _ in range(9)],
        map_terrain=terrain_dict(decoded),
    ).to_dict()

    json.dumps(snapshot)  # the payload is broadcast; a tuple key would raise here
    collision = capabilities.collision_from(snapshot, None)

    assert collision["ground_truth"] is True
    assert collision["walkable"] == decoded["walkable"]
    assert (collision["width"], collision["height"]) == (4, 2)
    assert collision["tile_ids"] == decoded["tile_ids"], "and the seams can still be worked out"
