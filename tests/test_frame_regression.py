"""The annotated frame, pinned byte for byte against the real ROM.

The annotated frame is the model's primary input, so a one-pixel drift is a
silent behaviour change that no assertion about JSON shapes would catch. These
digests were taken from a spread of save states chosen to exercise every branch
of the overlay: outdoor maps and building interiors, battle screens, active
dialog, warps and NPCs in view, an explored map with enough accumulated
territory to fill the mini-map inset, and the degraded path where no collision
window was captured at all.

Skipped entirely when the ROM or pyboy is absent, which is how CI runs.

A digest mismatch means the frame changed. If the change was deliberate, look at
the rendered PNG (the failure message names it), confirm it is what you wanted,
and re-pin. Upgrading Pillow or its default font also moves these numbers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

#: Folded into the explored-map store before any frame is drawn, so the mini-map
#: inset has real territory to show. Order matters: the store is order sensitive.
EXPLORERS = (
    "goredshouse.state",
    "pallet_has_charmander.state",
    "oakslab.state",
    "gotmap.state",
    "viridian_with_hp.state",
    "viridian_healed.state",
    "latest-today.state",
    "route2_north_of_forest.state",
    "final_position.state",
    "entered_viridian_forest.state",
    "forest_mid.state",
    "forest_low_hp_19.state",
    "viridian_forest_maze.state",
    "forest_won_battle_18_40.state",
    "after_rival_forest.state",
    "arrived_pewter_museum.state",
    "forest_gate_north.state",
    "in_forest_gate.state",
    "forest_north_exit_found.state",
    "north_gate_entered.state",
    "at_gym_door_pewter.state",
    "pewter_city_pre_gym.state",
    "pewter_pre_route3.state",
    "pewter_gym_entered.state",
    "pewter_museum_1f.state",
    "healed_in_pewter.state",
    "before_brock.state",
    "onto_route_3.state",
)

#: label -> save state. ``__nosnap`` labels are rendered a second time with no
#: navigation payload, which is the "overlay unavailable" branch.
CASES = {
    "outdoor_pallet_town": "goredshouse.state",
    "outdoor_pallet_charmander": "pallet_has_charmander.state",
    "outdoor_viridian_city": "viridian_healed.state",
    "outdoor_pewter_city_warps": "at_gym_door_pewter.state",
    "outdoor_pewter_city_npc": "pewter_pre_route3.state",
    "outdoor_route2_on_warp": "route2_north_of_forest.state",
    "outdoor_route3_no_warps": "onto_route_3.state",
    "outdoor_route2_south": "final_position.state",
    "inset_forest_maze": "viridian_forest_maze.state",
    "inset_forest_mid": "forest_mid.state",
    "inset_forest_north": "arrived_pewter_museum.state",
    "interior_oaks_lab": "oakslab.state",
    "interior_blues_house": "gotmap.state",
    "interior_pewter_gym": "pewter_gym_entered.state",
    "interior_pewter_museum": "pewter_museum_1f.state",
    "interior_pokecenter_npcs": "healed_in_pewter.state",
    "interior_forest_gate_npcs": "forest_gate_north.state",
    "interior_north_gate_npcs": "north_gate_entered.state",
    "dialog_oaks_lab": "oakdialog.state",
    "dialog_pewter_gym": "beat_brock_boulder_badge.state",
    "dialog_viridian_city": "latest-today.state",
    "battle_route1": "auto__20260409T145331Z__battle-entry__route-1.state",
    "battle_forest_sprites": "auto__20260825T104407Z__battle-entry__viridian-forest.state",
    "battle_forest_dialog": "auto__20260825T093307Z__battle-entry__viridian-forest.state",
    "battle_pewter_gym": "auto__20260825T150536Z__battle-entry__pewter-gym.state",
    "battle_route2": "auto__20260825T130419Z__battle-entry__route-2.state",
}

NO_SNAPSHOT_LABELS = ("outdoor_pallet_town", "battle_route1", "interior_oaks_lab")

#: sha256 of the two PNGs each case writes, by label.
#:
#: Every `annotated` digest moved when the decoded floor started reaching the
#: explored map, and only those: no `raw` frame changed. The mini-map inset used
#: to draw a mosaic of the windows the player happened to have stood in, because
#: `map_terrain` never survived the trip out of the emulator and `adopt_truth`
#: had no caller that reached it. It now draws the map. Pallet Town went from
#: "165 seen" to "360 seen", which is all of it.
EXPECTED = {
    "battle_forest_dialog": {
        "raw": "ae3d4f0b9cae093df8642bf5b728926ee9b10dbdfd2317d73f4c763f6431e87e",
        "annotated": "3fcc8b4f32bb54488d6de89efbc32da6371150f9bb2d7bd63a140caebb36b5f6",
    },
    "battle_forest_sprites": {
        "raw": "a94472e327ed9945ccff835605536991b2d76af29502e3272f72fa1cb804ead9",
        "annotated": "cdfe587211c0149dcdb8123131b935295fe199350006bb8c4bb7016c5209c98e",
    },
    "battle_pewter_gym": {
        "raw": "ede8bc8320992aa513a5ffdcd4fe58ef6a670776f36eb5e370d761a1b7cba188",
        "annotated": "1b9976bacc0db365666808b81d4a936d14a07be4742b8001970a74074a28d5a2",
    },
    "battle_route1": {
        "raw": "3f69b538550d501af84b3ef96ad474ab073cfd8c557056508ffcc56aea3fc0c9",
        "annotated": "1edcc7b9a4f887db7eb549049661118f30cc4b87e6b93c45cbcf47b827e8444e",
    },
    "battle_route1__nosnap": {
        "raw": "3f69b538550d501af84b3ef96ad474ab073cfd8c557056508ffcc56aea3fc0c9",
        "annotated": "b3b0b70fb2500d71f04ed62e18f695856e60eab73d6c3de611ac941bd0108a54",
    },
    "battle_route2": {
        "raw": "677a328545b3a064ac8100e512ddd13e876e9d15aa5b2e3510b6ec38c307324e",
        "annotated": "af6a27aff2c38a930856aed87519f138df9d949f619fe9acee01098a56c79165",
    },
    "dialog_oaks_lab": {
        "raw": "5594e7427de1e25ed51a90e7fdee69b8acfe5592499ac5676056fd8956673b72",
        "annotated": "184d8d65e8a96f0b64f16dc7e40d9821e2bd20bb6193b95e82e41c487f32ca6f",
    },
    "dialog_pewter_gym": {
        "raw": "fa49560a15c71b6a2fc0e80e0ee96e7abe6d4c92ad2c23432766290fd681cb53",
        "annotated": "1e7017fb32b08f44bf8a7a052a0655087d5ce7e7bad483f17021d185dd424b14",
    },
    "dialog_viridian_city": {
        "raw": "8380d26edb13cca078bad53010e163cbab7df02560fc63c085e9e0a30520200b",
        "annotated": "c045a1817bb99f25968b53593d19df2516af7f866bced7bd1bb3fba5a21687c0",
    },
    "inset_forest_maze": {
        "raw": "7d501d471d776679de680531991015c7791748a751476ff4c67a447a16fdf6fb",
        "annotated": "172806b97f10085a8fa292c1af9ee3a6b7e57f7b379da7b3ee170902cc6e5f8c",
    },
    "inset_forest_mid": {
        "raw": "fb311377caf5c30683b9d9629e399fc95b3ae687b0a4d6c0d0c043584793d071",
        "annotated": "5ce2ffb818f7e2dca9b99821231fd52b586896da01ddeb448935c3ddd7d6a32c",
    },
    "inset_forest_north": {
        "raw": "2623ec9e1e7e515eaf873073d986528fca402cac5354b42279b4216a50613a3a",
        "annotated": "28399e0085673fab2b6866b536504ae15924a042dc389abfceaf1e49aae8918e",
    },
    "interior_blues_house": {
        "raw": "5d3921a2ed2f1556c9a5f0c2cc5217bbbc7deeba8d8fccd44087582a122f50ec",
        "annotated": "d6c7dab06e74ad5968bc03e05a9a4a5b53cab6348365fd25918839b19aa777a0",
    },
    "interior_forest_gate_npcs": {
        "raw": "e929e0941a8ff371ba86e18982194eba0b443d18b43d2853f7bbe6ac418975c9",
        "annotated": "79d69dd43f117560b04effa3d8f5ba0da90b63f798666d0b4dbb526cab818bf0",
    },
    "interior_north_gate_npcs": {
        "raw": "2eaf505ee4cdd31eb64400ac3332ec824a23840d4ac72a4d65a4cb6deb938371",
        "annotated": "255f9ba2018d89495551210cc2efbef036b5d9fd4155150fed4f712e6ffb4c2c",
    },
    "interior_oaks_lab": {
        "raw": "9c4698235d465a30133b3bdf0ff3123112670ba31b25e414923b04b38e6c04c6",
        "annotated": "019c0471e0993c73bb54735bc2823d8b08569e75e60bfd546299b7a7ec4115e4",
    },
    "interior_oaks_lab__nosnap": {
        "raw": "9c4698235d465a30133b3bdf0ff3123112670ba31b25e414923b04b38e6c04c6",
        "annotated": "71cbfdc394dc33d92c0942303911a9485085aee0ae348f0b1da594d7d9f59e45",
    },
    "interior_pewter_gym": {
        "raw": "6c15290ec11de0b5eb3bda876193fc1b17718c09febb87274bb36b79594e662f",
        "annotated": "7e8a60cf08dc8f4bbee634a887554ee3cde0222aa56d6f81a463283809acc520",
    },
    "interior_pewter_museum": {
        "raw": "d5f519238eb8888d5449d6ec91e096447dfe8d79bdc87371269c5cf0302a4f11",
        "annotated": "872244e9622e05ee7b7ed8a26aeed32c162bf23c8171412ede54576d8838de2b",
    },
    "interior_pokecenter_npcs": {
        "raw": "9849d5cd3367daa0d1cb8cc6ee2dc5a67c2cd96f5b129194582ad78a6906eb54",
        "annotated": "fdb8ca5605587e49ad6488ba13ee1521c1db08e2fe54efe8c07a76421961e958",
    },
    "outdoor_pallet_charmander": {
        "raw": "302c807de4cc1fbd1540dc88b434126488d778c0841ff86a3147a429a9d05a0c",
        "annotated": "19659e8b680e24c8b38cf210bdea603252923e55cd26eb8411fba27514112b94",
    },
    "outdoor_pallet_town": {
        "raw": "8a67104dc40e9f8ec7dfe8b8b2d7aac6c9f095a4fb7f166532d70f6393be4a7c",
        "annotated": "96502d9eecef53667a88ca0555a8e280aab0638265eabdec006cbce805ef9d13",
    },
    "outdoor_pallet_town__nosnap": {
        "raw": "8a67104dc40e9f8ec7dfe8b8b2d7aac6c9f095a4fb7f166532d70f6393be4a7c",
        "annotated": "14f46703d8b9e01d764c4efd25bc405ea71694c4697a6eb0546d35a3a54ccc37",
    },
    "outdoor_pewter_city_npc": {
        "raw": "e0b97fbd11bad23252138b25958f799c3b7b4256fff442a722f4b919c674d8ed",
        "annotated": "4dc872e44ba3b8a5cab64fbcf2354c04ccb3cf3205a3b734f415f74d6ae376d5",
    },
    "outdoor_pewter_city_warps": {
        "raw": "dd3a3d165e1a85e8597448a277670c9419094b15d7f6913c2345548a7d6d7704",
        "annotated": "aeb3bfa20c5d18c919a4afa1bfdd2b6bc4fb243a2055e44afcc9496afe5c5231",
    },
    "outdoor_route2_on_warp": {
        "raw": "46cec23becb778d63036862d959e98bb9b3556cbc649a0a67167f97317d77806",
        "annotated": "15feb3738c68d5c8712feaaa4078db9de541a34ce5f2f2d9d67d11c568dec05f",
    },
    "outdoor_route2_south": {
        "raw": "0debbf0194730f9ee1fd9facc2df4b91b33057125b8641b340e665999bf9a080",
        "annotated": "a46166af4ddbe63ebea9b7ed30fce633a48fb031b0ca925c472e4f3ae8d9b41c",
    },
    "outdoor_route3_no_warps": {
        "raw": "e20fa6696f4a311f3fc3567a6448c99422c37ea56385cd4320b3396f9eafaaca",
        "annotated": "1c7f97a69983a17560c04e2ab9c519ccb44ebcd3e30bf3e832ed5f3df5f5a7bc",
    },
    "outdoor_viridian_city": {
        "raw": "7afdf6a5a148d046ead1ade3929c63e96cbe9c69f7b0fa7c24861a21ca262774",
        "annotated": "cd77f2ae94e4029ff90eefc821ce536253cf8cb6717d39d28b378368b728cd8f",
    },
}

#: sha256 of the standalone goal-marker overlay.
GOAL_MARKER_DIGEST = "d4c2085762a1d9f62f7d74dfb627c09803495f564aa022af51c74c39615d1f1d"


@pytest.fixture(scope="module")
def emulator():
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator

    emu = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        yield emu
    finally:
        emu.close()


@pytest.fixture(scope="module")
def reader(emulator):
    from pokemon_agent.memory.red import PokemonRedReader

    return PokemonRedReader(emulator)


@pytest.fixture(scope="module")
def explored_maps(emulator, reader, tmp_path_factory):
    from pokemon_agent.explored_map import ExploredMaps

    maps = ExploredMaps(tmp_path_factory.mktemp("explored") / "maps.json")
    for name in EXPLORERS:
        save = SAVES_DIR / name
        if not save.exists():
            continue
        emulator.load_state(str(save))
        emulator.settle()
        maps.record(emulator.get_navigation_snapshot(reader).to_dict())
    return maps


def _load(emulator, reader, name: str) -> tuple[dict, dict]:
    """Restore one save and read back the state and navigation payload the server sends."""
    from pokemon_agent.state.builder import build_game_state

    save = SAVES_DIR / name
    if not save.exists():
        pytest.skip(f"{name} is not present in {SAVES_DIR}")
    emulator.load_state(str(save))
    emulator.settle()
    state = build_game_state(reader, frame_count=0)
    dialog = state.get("dialog")
    state["dialog_active"] = bool(isinstance(dialog, dict) and dialog.get("active"))
    snapshot = emulator.get_navigation_snapshot(reader)
    state["interaction"] = snapshot.interaction
    return state, {"snapshot": snapshot.to_dict()}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_rom
@pytest.mark.parametrize("label", sorted(CASES))
def test_the_annotated_frame_is_byte_for_byte_what_it_was(
    label, emulator, reader, explored_maps, tmp_path
):
    from pokemon_agent.agent_runtime import AgentRuntime

    state, navigation = _load(emulator, reader, CASES[label])

    variants = [("", navigation)]
    if label in NO_SNAPSHOT_LABELS:
        variants.append(("__nosnap", None))

    for suffix, nav in variants:
        workspace = tmp_path / (label + suffix)
        runtime = AgentRuntime(
            data_dir=workspace / "data",
            workspace_dir=workspace / "workspace",
            visited_lookup=explored_maps.visited,
            map_grid_lookup=explored_maps.grid,
        )
        runtime.refresh(
            emulator=emulator,
            state=state,
            navigation=nav,
            reason="frame_regression",
            source="observe",
        )
        expected = EXPECTED[label + suffix]
        for kind, short in (("latest_frame", "raw"), ("latest_frame_annotated", "annotated")):
            path = runtime.artifacts[kind]
            assert _digest(path) == expected[short], (
                f"{label}{suffix} {kind} changed; the rendered PNG is at {path}"
            )


@needs_rom
def test_the_goal_marker_and_the_no_inset_layout_are_unchanged(emulator, reader, tmp_path):
    """The one path the bundle never takes: a goal glyph with no explored map."""
    from PIL import Image

    from pokemon_agent.agent_runtime import render_navigation_overlay

    emulator.load_state(str(SAVES_DIR / "route2_north_of_forest.state"))
    emulator.settle()
    snapshot = emulator.get_navigation_snapshot(reader)
    screen = emulator.get_screen()
    if not isinstance(screen, Image.Image):
        screen = Image.fromarray(screen)

    overlay = render_navigation_overlay(
        screen,
        snapshot,
        objective={"summary": "Reach the Viridian Forest gate."},
        goal=(3, 9),
        visited={(3, 11), (3, 12), (4, 11)},
        map_grid=None,
    )
    path = tmp_path / "goal_marker.png"
    overlay.save(path, format="PNG")

    assert _digest(path) == GOAL_MARKER_DIGEST, (
        f"the goal-marker overlay changed; the rendered PNG is at {path}"
    )
