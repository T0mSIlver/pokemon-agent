from pokemon_agent.emulator import _build_interaction_probe
from pokemon_agent.memory.red import ADDR_MAP_HEIGHT, ADDR_MAP_WIDTH, PokemonRedReader
from pokemon_agent.navigation import LiveNavigationSnapshot


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


def test_compute_valid_moves_on_warp_tile_includes_all_directions():
    """Standing on a warp tile must expose every direction so the agent can
    press the one that fires the transition, even when the tile beyond it is
    a wall (e.g. a south-edge doormat warp inside Blue's House)."""
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    terrain = [[1 for _ in range(10)] for _ in range(9)]
    for x in range(10):
        terrain[5][x] = 0
    tilemap = [[0 for _ in range(20)] for _ in range(18)]

    moves = emu._compute_valid_moves(
        terrain=terrain,
        tilemap=tilemap,
        tileset="OVERWORLD",
        sprites_local=set(),
        player_coords=(2, 7),
        warps=[{"x": 2, "y": 7, "warp_id": 0, "target_map_id": 40}],
    )

    assert set(moves) == {"up", "down", "left", "right"}


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
