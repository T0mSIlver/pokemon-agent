from pokemon_agent.emulator import _build_interaction_probe
from pokemon_agent.memory.red import ADDR_MAP_HEIGHT, ADDR_MAP_WIDTH, PokemonRedReader
from pokemon_agent.navigation import (
    LEDGE_TILE_PAIRS,
    TILE_ID_OFFSET,
    LiveNavigationSnapshot,
    ledge_hop_allows,
)


def make_snapshot() -> LiveNavigationSnapshot:
    return LiveNavigationSnapshot(
        map_id=1,
        map_name="TEST MAP",
        player_position=(10, 10),
        facing="down",
        tileset="OVERWORLD",
        window_top_left=(8, 9),
        terrain=[
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        sprite_positions=[],
        valid_moves=["up", "left", "right"],
        warps=[],
        map_dimensions={"width": 20, "height": 18},
        tile_ids={
            (8, 9): 11,
            (9, 9): 12,
            (10, 9): 13,
            (8, 10): 21,
            (9, 10): 22,
            (10, 10): 23,
            (8, 11): 31,
            (9, 11): 32,
            (10, 11): 33,
        },
    )


def test_render_window_ascii_marks_warp_tiles_with_W():
    snapshot = make_snapshot()
    snapshot.warps = [{"x": 9, "y": 11, "target_map_id": 5}]

    rendered = snapshot.render_window_ascii()
    legend = snapshot.to_dict().get("ascii_legend") or {}

    assert "W" in rendered
    assert "W" in legend
    assert "step ONTO" in legend["W"].lower() or "warp" in legend["W"].lower()


def test_live_window_ascii_uses_symbols_not_distance_digits():
    snapshot = make_interaction_snapshot(valid_moves=["up", "left", "right"])

    live_ascii = snapshot.render_window_ascii()

    assert live_ascii.splitlines()[0].strip().isdigit()
    assert "P" in live_ascii
    rendered_rows = [line.split(maxsplit=1)[1] for line in live_ascii.splitlines()[1:]]
    assert not any(char.isdigit() for row in rendered_rows for char in row)


def make_interaction_snapshot(sprite_positions=None, valid_moves=None) -> LiveNavigationSnapshot:
    return LiveNavigationSnapshot(
        map_id=1,
        map_name="TEST MAP",
        player_position=(10, 10),
        facing="up",
        tileset="MART",
        window_top_left=(6, 6),
        terrain=[[1 for _ in range(10)] for _ in range(9)],
        sprite_positions=list(sprite_positions or []),
        valid_moves=list(valid_moves or ["down", "left", "right"]),
        warps=[],
        map_dimensions={"width": 20, "height": 18},
    )


def make_tilemap(default=0) -> list[list[int]]:
    return [[default for _ in range(20)] for _ in range(18)]


def test_interaction_probe_detects_direct_object():
    snapshot = make_interaction_snapshot(sprite_positions=[(10, 9)])
    probe = _build_interaction_probe(
        snapshot,
        tilemap=make_tilemap(),
        signs=[],
        talk_over_tiles=set(),
        dialog_active=True,
    )

    assert probe["kind"] == "object"
    assert probe["source"] == "sprite_direct"
    assert probe["distance"] == 1
    assert probe["coordinate_system"] == "map_tile_absolute"
    assert probe["target_coord"] == {"x": 10, "y": 9}
    assert probe["front_tile"]["coord"] == {"x": 10, "y": 9}
    assert "local_coord" not in probe["front_tile"]


def test_interaction_probe_detects_direct_sign():
    snapshot = make_interaction_snapshot()
    probe = _build_interaction_probe(
        snapshot,
        tilemap=make_tilemap(),
        signs=[{"x": 10, "y": 9, "text_id": 7}],
        talk_over_tiles=set(),
        dialog_active=True,
    )

    assert probe["kind"] == "sign"
    assert probe["source"] == "sign_direct"
    assert probe["sign_text_id"] == 7
    assert probe["coordinate_system"] == "map_tile_absolute"
    assert probe["target_coord"] == {"x": 10, "y": 9}


def test_interaction_probe_detects_object_over_counter():
    snapshot = make_interaction_snapshot(sprite_positions=[(10, 8)])
    terrain = [[1 for _ in range(10)] for _ in range(9)]
    terrain[3][4] = 0
    snapshot.terrain = terrain

    tilemap = make_tilemap()
    tilemap[7][8] = 0x18
    probe = _build_interaction_probe(
        snapshot,
        tilemap=tilemap,
        signs=[],
        talk_over_tiles={0x18},
        dialog_active=True,
    )

    assert probe["kind"] == "object"
    assert probe["source"] == "sprite_over_counter"
    assert probe["distance"] == 2
    assert probe["target_coord"] == {"x": 10, "y": 8}


def test_compute_valid_moves_respects_tile_pair_blockers():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain = [[1 for _ in range(10)] for _ in range(9)]
    tilemap = [[0 for _ in range(20)] for _ in range(18)]
    tilemap[9][8] = 302
    tilemap[7][8] = 304

    moves = emu._compute_valid_moves(
        terrain=terrain,
        tilemap=tilemap,
        tileset="FOREST",
        sprites_local=set(),
        player_coords=(10, 10),
        warps=[],
    )

    assert "up" not in moves
    assert {"down", "left", "right"}.issubset(set(moves))


def _warp_scene(front_tile: int) -> tuple[list[list[int]], list[list[int]]]:
    """A player boxed in from below, standing on the map's bottom-edge warp."""
    terrain = [[1 for _ in range(10)] for _ in range(9)]
    for x in range(10):
        terrain[5][x] = 0
    tilemap = [[0 for _ in range(20)] for _ in range(18)]
    tilemap[9][8] = 0x2C + TILE_ID_OFFSET
    tilemap[11][8] = front_tile
    return terrain, tilemap


def test_compute_valid_moves_leaves_warp_directions_to_the_warp_rule():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x12 + TILE_ID_OFFSET)

    moves = emu._compute_valid_moves(
        terrain=terrain,
        tilemap=tilemap,
        tileset="OVERWORLD",
        sprites_local=set(),
        player_coords=(2, 7),
        warps=[{"x": 2, "y": 7, "warp_id": 0, "target_map_id": 40}],
    )

    assert set(moves) == {"up", "left", "right"}


def test_warp_exit_comes_from_the_carpet_tile_in_front_outdoors():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x12 + TILE_ID_OFFSET)

    exits, armed, note = emu._compute_warp_exits(
        terrain=terrain,
        tilemap=tilemap,
        tileset="OVERWORLD",
        map_id=13,
        map_dimensions={"width_tiles": 20, "height_tiles": 72},
        player_coords=(3, 11),
        warps=[{"x": 3, "y": 11, "warp_id": 1, "target_map_id": 47}],
        armed=True,
    )

    assert exits == ["down"]
    assert armed is True
    assert note is None


def test_warp_exit_is_empty_when_no_neighbour_matches():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x50 + TILE_ID_OFFSET)

    exits, armed, note = emu._compute_warp_exits(
        terrain=terrain,
        tilemap=tilemap,
        tileset="OVERWORLD",
        map_id=13,
        map_dimensions={"width_tiles": 20, "height_tiles": 72},
        player_coords=(3, 11),
        warps=[{"x": 3, "y": 11, "warp_id": 1, "target_map_id": 47}],
        armed=True,
    )

    assert exits == []
    assert armed is False
    assert note is not None


def test_warp_exit_indoors_comes_from_the_map_edge():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x50 + TILE_ID_OFFSET)

    exits, armed, _ = emu._compute_warp_exits(
        terrain=terrain,
        tilemap=tilemap,
        tileset="HOUSE",
        map_id=39,
        map_dimensions={"width_tiles": 8, "height_tiles": 8},
        player_coords=(2, 7),
        warps=[{"x": 2, "y": 7, "warp_id": 1, "target_map_id": 255}],
        armed=True,
    )

    assert exits == ["down"]
    assert armed is True


def test_warp_exit_says_so_when_the_map_edge_cannot_be_checked():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x50 + TILE_ID_OFFSET)

    exits, armed, note = emu._compute_warp_exits(
        terrain=terrain,
        tilemap=tilemap,
        tileset="HOUSE",
        map_id=39,
        map_dimensions=None,
        player_coords=(2, 7),
        warps=[{"x": 2, "y": 7, "warp_id": 1, "target_map_id": 255}],
        armed=True,
    )

    assert exits == []
    assert armed is False
    assert "dimensions" in note


def test_warp_exit_reports_the_direction_but_not_armed_after_a_state_load():
    """The engine arms a warp when the player *walks onto* it, so a save state
    dropped on one has the exit direction but no way to fire it yet."""
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain, tilemap = _warp_scene(0x50 + TILE_ID_OFFSET)

    exits, armed, note = emu._compute_warp_exits(
        terrain=terrain,
        tilemap=tilemap,
        tileset="HOUSE",
        map_id=52,
        map_dimensions={"width_tiles": 20, "height_tiles": 8},
        player_coords=(7, 7),
        warps=[{"x": 7, "y": 7, "warp_id": 0, "target_map_id": 53}],
        armed=False,
    )

    assert exits == ["down"]
    assert armed is False
    assert "Step off and back on" in note


def test_ledge_tiles_are_directional_and_overworld_only():
    # pokered's table has no upward ledge and no entry outside OVERWORLD.
    assert ledge_hop_allows("OVERWORLD", "down", 0x39 + TILE_ID_OFFSET, 0x37 + TILE_ID_OFFSET)
    assert not ledge_hop_allows("OVERWORLD", "up", 0x39 + TILE_ID_OFFSET, 0x37 + TILE_ID_OFFSET)
    assert not ledge_hop_allows("FOREST", "down", 0x39 + TILE_ID_OFFSET, 0x37 + TILE_ID_OFFSET)
    assert not ledge_hop_allows("OVERWORLD", "down", 0x39 + TILE_ID_OFFSET, 0x27 + TILE_ID_OFFSET)
    assert ledge_hop_allows("OVERWORLD", "left", 0x2C + TILE_ID_OFFSET, 0x27 + TILE_ID_OFFSET)
    assert ledge_hop_allows("OVERWORLD", "right", 0x2C + TILE_ID_OFFSET, 0x1D + TILE_ID_OFFSET)
    assert not LEDGE_TILE_PAIRS["OVERWORLD"].get("up")


def test_compute_valid_moves_offers_a_ledge_the_collision_map_calls_blocked():
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain = [[1 for _ in range(10)] for _ in range(9)]
    terrain[5][4] = 0  # the ledge itself is never walkable terrain
    tilemap = [[0 for _ in range(20)] for _ in range(18)]
    tilemap[9][8] = 0x39 + TILE_ID_OFFSET
    tilemap[11][8] = 0x37 + TILE_ID_OFFSET

    hops = emu._compute_ledge_hops(tilemap, "OVERWORLD", (3, 46))
    moves = emu._compute_valid_moves(
        terrain=terrain,
        tilemap=tilemap,
        tileset="OVERWORLD",
        sprites_local=set(),
        player_coords=(3, 46),
        warps=[],
        ledge_hops=hops,
    )

    assert "down" in moves
    assert hops == {"down": (3, 48)}


def test_snapshot_publishes_ledge_hops_and_warp_exits():
    snapshot = make_snapshot()
    snapshot.ledge_hops = {"down": (10, 12)}
    snapshot.warp_exit_directions = ["down"]
    snapshot.warp_exit_armed = True

    payload = snapshot.to_dict()

    assert payload["ledge_hops"] == {"down": {"x": 10, "y": 12}}
    assert payload["warp_exit_directions"] == ["down"]
    assert payload["warp_exit_armed"] is True
    assert "valid_moves" in payload["movement_legend"]["warp_exit_directions"]


class FakeMemoryEmulator:
    def __init__(self, values):
        self.values = values

    def read_u8(self, addr):
        return self.values.get(addr, 0)

    def read_range(self, addr, size):
        return bytes(self.values.get(addr + offset, 0) for offset in range(size))


def test_read_map_dimensions_exposes_tile_and_block_units():
    reader = PokemonRedReader(
        FakeMemoryEmulator(
            {
                ADDR_MAP_WIDTH: 10,
                ADDR_MAP_HEIGHT: 9,
            }
        )
    )

    dimensions = reader.read_map_dimensions()

    assert dimensions == {
        "width": 20,
        "height": 18,
        "width_blocks": 10,
        "height_blocks": 9,
        "width_tiles": 20,
        "height_tiles": 18,
    }


def test_read_dialog_treats_visible_window_as_active():
    emu = FakeMemoryEmulator(
        {
            0xD125: 1,
            0xD730: 0,
            0xFF4A: 0,
            0xFF4B: 7,
        }
    )
    reader = PokemonRedReader(emu)

    dialog = reader.read_dialog()

    assert dialog["active"] is True
    assert dialog["window_visible"] is True
    assert dialog["waiting_for_input"] is True
    assert dialog["printing"] is False


def test_read_dialog_inactive_when_window_hidden():
    emu = FakeMemoryEmulator(
        {
            0xD125: 1,
            0xD730: 0,
            0xFF4A: 0x90,
            0xFF4B: 7,
        }
    )
    reader = PokemonRedReader(emu)

    dialog = reader.read_dialog()

    assert dialog["active"] is False
    assert dialog["window_visible"] is False


def test_read_battle_uses_enemy_battle_struct_offsets():
    values = {
        0xD057: 1,  # wild battle
        0xD89D: 177,  # wrong non-active species slot; should be ignored
        0xCFE5 + 0: 36,  # active battle_struct species = Pidgey internal index
        0xCFE5 + 1: 0x00,
        0xCFE5 + 2: 0x14,  # hp = 20
        0xCFE5 + 4: 0x00,  # status = OK
        0xCFE5 + 8: 33,  # Tackle
        0xCFE5 + 9: 28,  # Sand Attack
        0xCFE5 + 14: 5,  # correct battle_struct level
        0xCFE5 + 15: 0x00,
        0xCFE5 + 16: 0x14,  # max_hp = 20
        0xCFE5 + 33: 35,  # wrong old party_struct offset, should be ignored
    }
    reader = PokemonRedReader(FakeMemoryEmulator(values))

    battle = reader.read_battle()

    assert battle["in_battle"] is True
    assert battle["type"] == "wild"
    assert battle["enemy"]["species_id"] == 36
    assert battle["enemy"]["pokedex_id"] == 16
    assert battle["enemy"]["species"] == "Pidgey"
    assert battle["enemy"]["level"] == 5
    assert battle["enemy"]["hp"] == 20
    assert battle["enemy"]["max_hp"] == 20
    assert battle["enemy"]["moves"] == ["Tackle", "Sand Attack"]


def test_read_party_decodes_internal_species_index():
    values = {
        0xD163: 1,  # one party mon
        0xD16B + 0: 176,  # Charmander internal species index
        0xD16B + 1: 0x00,
        0xD16B + 2: 0x15,  # hp = 21
        0xD16B + 4: 0x00,
        0xD16B + 5: 20,
        0xD16B + 6: 20,
        0xD16B + 8: 10,  # Scratch
        0xD16B + 9: 45,  # Growl
        0xD16B + 29: 35,
        0xD16B + 30: 40,
        0xD16B + 33: 6,
        0xD16B + 34: 0x00,
        0xD16B + 35: 0x15,
        0xD2B5: 0x50,  # nickname terminator
    }
    reader = PokemonRedReader(FakeMemoryEmulator(values))

    party = reader.read_party()

    assert len(party) == 1
    assert party[0]["species_id"] == 176
    assert party[0]["pokedex_id"] == 4
    assert party[0]["species"] == "Charmander"
    assert party[0]["level"] == 6


class FakeMemory(dict):
    def __missing__(self, addr):
        return 0


class FakePyBoy:
    """Enough PyBoy to drive settle(): a memory map and a frame counter."""

    def __init__(self, script):
        self.script = script
        self.frame = 0
        self.memory = FakeMemory()

    def tick(self):
        self.frame += 1
        self.memory = FakeMemory(self.script(self.frame))


def make_settle_emulator(script):
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    emu._pyboy = FakePyBoy(script)
    emu._pyboy.tick()
    return emu


def test_settle_returns_once_the_player_holds_still():
    from pokemon_agent.emulator import ADDR_PLAYER_Y, ADDR_WALK_COUNTER

    def script(frame):
        # One tile of walking, then nothing.
        if frame <= 16:
            return {ADDR_WALK_COUNTER: 8 - frame // 2, ADDR_PLAYER_Y: 46}
        return {ADDR_PLAYER_Y: 47}

    emu = make_settle_emulator(script)

    assert emu.settle(max_frames=200, quiet_frames=10) is True
    assert emu._pyboy.frame < 40


def test_settle_gives_up_at_the_frame_cap():
    from pokemon_agent.emulator import ADDR_WALK_COUNTER

    emu = make_settle_emulator(lambda frame: {ADDR_WALK_COUNTER: 4})

    assert emu.settle(max_frames=50, quiet_frames=10) is False
    assert emu._pyboy.frame == 51  # the priming tick plus the cap, and no more


def test_settle_does_not_stop_on_the_intermediate_tile_of_a_ledge_hop():
    from pokemon_agent.emulator import ADDR_MOVEMENT_FLAGS, ADDR_PLAYER_Y

    mid_hop = {ADDR_MOVEMENT_FLAGS: 0x40}  # wMovementFlags bit 6, the ledge hop

    def script(frame):
        # y reaches the intermediate tile early and rests there long enough to
        # look settled; only the ledge flag says otherwise.
        if frame < 10:
            return {ADDR_PLAYER_Y: 46, **mid_hop}
        if frame < 40:
            return {ADDR_PLAYER_Y: 47, **mid_hop}
        return {ADDR_PLAYER_Y: 48}

    emu = make_settle_emulator(script)

    assert emu.settle(max_frames=200, quiet_frames=10) is True
    assert emu.read_u8(ADDR_PLAYER_Y) == 48
