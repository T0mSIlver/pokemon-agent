from pathlib import Path

import pokemon_agent.server as server
from pokemon_agent.emulator import _build_interaction_probe
from pokemon_agent.memory.red import PokemonRedReader
from pokemon_agent.navigation import (
    LiveNavigationSnapshot,
    LocationNavigationMap,
    NavigationPath,
    NavigationStore,
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


def test_location_map_updates_and_routes_to_known_tile():
    snapshot = make_snapshot()
    location_map = LocationNavigationMap(map_id=1, map_name="TEST MAP")
    location_map.update_from_snapshot(snapshot)

    plan = location_map.plan_route(start=(10, 10), goal=(10, 9))

    assert plan.reached is True
    assert plan.partial is False
    assert plan.directions == ["up"]
    assert plan.actions == ["walk_up"]
    assert plan.final_position == (10, 9)


def test_location_map_routes_adjacent_to_blocked_goal():
    snapshot = make_snapshot()
    location_map = LocationNavigationMap(map_id=1, map_name="TEST MAP")
    location_map.update_from_snapshot(snapshot)

    plan = location_map.plan_route(start=(10, 10), goal=(8, 11))

    assert plan.reached is False
    assert plan.partial is True
    assert plan.final_position in {(8, 10), (9, 11)}
    assert len(plan.actions) == 2


def test_location_map_marks_player_tile_passable_even_if_snapshot_disagrees():
    snapshot = make_snapshot()
    snapshot.terrain[1][2] = 0
    location_map = LocationNavigationMap(map_id=1, map_name="TEST MAP")

    location_map.update_from_snapshot(snapshot)

    assert location_map.tiles[(10, 10)] is True


def test_ascii_maps_use_symbols_not_distance_digits():
    snapshot = make_interaction_snapshot(valid_moves=["up", "left", "right"])
    location_map = LocationNavigationMap(map_id=1, map_name="TEST MAP")
    location_map.update_from_snapshot(snapshot)
    distances = location_map.distance_map(
        snapshot.player_position,
        extra_blockers=snapshot.sprite_set,
    )

    live_ascii = snapshot.render_window_ascii()
    explored_ascii = location_map.render_ascii(
        player=snapshot.player_position,
        sprites=snapshot.sprite_positions,
        distances=distances,
    )

    assert live_ascii.splitlines()[0].strip().isdigit()
    assert explored_ascii.splitlines()[0].strip().isdigit()
    assert "P" in live_ascii
    assert "P" in explored_ascii
    rendered_rows = [line.split(maxsplit=1)[1] for line in explored_ascii.splitlines()[1:]]
    assert not any(char.isdigit() for row in rendered_rows for char in row)


def test_navigation_store_round_trip(tmp_path: Path):
    store_path = tmp_path / "navigation_maps.json"
    store = NavigationStore(store_path)
    snapshot = make_snapshot()
    store.update(snapshot)

    reloaded = NavigationStore(store_path)
    location_map = reloaded.get(snapshot.key)

    assert location_map is not None
    assert location_map.tileset == "OVERWORLD"
    assert location_map.tiles[(10, 10)] is True
    assert location_map.tiles[(8, 11)] is False
    assert location_map.tile_ids[(10, 10)] == 23


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
    assert probe["target_coord"] == {"x": 10, "y": 9}


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


def test_location_map_respects_tile_pair_blockers():
    snapshot = LiveNavigationSnapshot(
        map_id=2,
        map_name="FOREST TEST",
        player_position=(10, 10),
        facing="up",
        tileset="FOREST",
        window_top_left=(9, 9),
        terrain=[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        sprite_positions=[],
        valid_moves=["left", "right", "down"],
        warps=[],
        tile_ids={
            (9, 9): 1,
            (10, 9): 304,
            (11, 9): 1,
            (9, 10): 1,
            (10, 10): 302,
            (11, 10): 1,
            (9, 11): 1,
            (10, 11): 1,
            (11, 11): 1,
        },
    )
    location_map = LocationNavigationMap(map_id=2, map_name="FOREST TEST")
    location_map.update_from_snapshot(snapshot)

    plan = location_map.plan_route(start=(10, 10), goal=(10, 9))

    assert plan.reached is True
    assert plan.directions != ["up"]
    assert len(plan.directions) > 1


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


def test_auto_mode_falls_back_to_persistent_for_visible_offscreen_route(monkeypatch):
    snapshot = make_interaction_snapshot(valid_moves=["down", "left", "right", "up"])
    target = (10, 9)

    class FakeLocationMap:
        def plan_route(self, start, goal, extra_blockers=None):
            assert start == snapshot.player_position
            assert goal == target
            assert extra_blockers == snapshot.sprite_set
            return NavigationPath(
                mode="persistent",
                target=goal,
                reached=True,
                status="Found a route to the explored target tile.",
                directions=["left", "up", "right"],
                final_position=goal,
            )

    class FakeEmulator:
        def plan_screen_path(self, reader, target_x, target_y):
            assert (target_x, target_y) == target
            return NavigationPath(
                mode="screen",
                target=target,
                reached=False,
                status=(
                    "No exact visible route was found; "
                    "routed to the closest visible reachable tile."
                ),
                directions=["left"],
                final_position=(9, 10),
                partial=True,
                visible_target=True,
            )

    monkeypatch.setattr(
        server,
        "_refresh_navigation_state_sync",
        lambda: (snapshot, FakeLocationMap()),
    )
    monkeypatch.setattr(server, "_emulator", FakeEmulator())
    monkeypatch.setattr(server, "_reader", object())

    _snapshot, _location_map, plan = server._plan_navigation_sync(target[0], target[1], "auto")

    assert plan.mode == "persistent"
    assert plan.reached is True
    assert plan.visible_target is True
    assert "leaving the current screen" in plan.status


class FakeMemoryEmulator:
    def __init__(self, values):
        self.values = values

    def read_u8(self, addr):
        return self.values.get(addr, 0)

    def read_range(self, addr, size):
        return bytes(self.values.get(addr + offset, 0) for offset in range(size))


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


def test_navigation_execution_summary_detects_no_movement():
    before = make_snapshot()
    after = make_snapshot()
    plan = NavigationPath(
        mode="persistent",
        target=(10, 9),
        reached=True,
        status="Found a route to the explored target tile.",
        directions=["up"],
        final_position=(10, 9),
    )

    execution = server._summarize_navigation_execution(
        before,
        after,
        plan,
        (10, 9),
    )

    assert execution["success"] is False
    assert execution["moved"] is False
    assert execution["reached_target"] is False
    assert "did not move" in execution["status"]


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
