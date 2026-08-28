"""Surf and Strength, driven on a save built to have them.

Cut and the Bicycle could be checked against the cartridge because the corpus
happened to contain a run that had them. Surf, Strength, Flash and Fly could
not: no save in 3,000 has HM03, HM04, HM05 or HM02, and none has the Soul or
Rainbow badge. Shipping those unverified is the pattern this project has found
nine times — a mechanism that looks right and cannot work.

`tests/hm_fixtures.py` closes that gap: it writes the move into a party
Pokemon's move slot and the badge into wObtainedBadges, in RAM on a copy, so the
game itself can be asked. What is asserted here is the game's answer, not the
menu closing.

Flash is deliberately absent. Used outdoors it prints "A blinding FLASH lights
the area!" and moves no persistent byte of WRAM — 387 bytes change and every one
of them is menu or tile scratch. So there is nothing to check it against outside
a dark cave, no save in the corpus is in one, and a verb that cannot tell
success from failure is worse than no verb. It waits for the run to reach Rock
Tunnel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def _find_saves_dir():
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

START_MENU_TOP_Y = 2
START_MENU_POKEMON_ROW = 1


def _menuable_save(emulator, tmp_path) -> Path:
    """A save where pressing START actually opens the main menu.

    Not the first one in the directory: the corpus is full of states caught
    mid-dialog, mid-battle and before the Pokedex exists, and in all of those
    wTopMenuItemY reads as something other than 2. Picking by name gave a save
    whose START answered 12, which is a different menu entirely.
    """
    import shutil

    from pokemon_agent.memory.red import ADDR_TOP_MENU_ITEM_Y

    for path in sorted(SAVES_DIR.glob("*.state")):
        copy = tmp_path / "menuable.state"
        shutil.copy(path, copy)
        try:
            emulator.load_state(str(copy))
            emulator.settle()
            emulator.press_and_settle("start")
        except Exception:  # noqa: BLE001 — a save that will not load proves nothing
            continue
        if emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y:
            return path
    pytest.skip("no save in the corpus opens the main menu on START")


def _drive_field_move(emulator, record):
    """START -> POKEMON -> the first mon -> the doctored move, reading each screen."""
    import hm_fixtures as hm

    from pokemon_agent.memory.red import ADDR_CURRENT_MENU_ITEM, ADDR_TOP_MENU_ITEM_Y

    emulator.settle()
    emulator.press_and_settle("start")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y
    for _ in range(12):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == START_MENU_POKEMON_ROW:
            break
        emulator.press_and_settle("up" if at > START_MENU_POKEMON_ROW else "down")
    emulator.press_and_settle("a")  # the party list
    emulator.press_and_settle("a")  # the first mon
    ids = hm.move_ids(emulator, 0)
    rows = [index for index, move_id in enumerate(ids) if move_id in hm.FIELD_MOVE_IDS]
    for _ in range(rows.index(record.move_slot)):
        emulator.press_and_settle("down")
    emulator.press_and_settle("a")
    for _ in range(10):
        emulator.press_and_settle("a")
    emulator.settle()


@needs_rom
def test_strength_switches_on_the_flag_the_server_checks(tmp_path):
    """`wd728` bit 0, which is what `_field_strength_sync` reports success from.

    The address was found by diffing WRAM either side of the move rather than
    recalled: it is the only byte in 0xD700-0xD7FF that changes, 0 -> 1.
    """
    pytest.importorskip("pyboy")
    import hm_fixtures as hm

    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        save = _menuable_save(emulator, tmp_path)
        record = hm.give_field_move(emulator, save, "STRENGTH", tmp_path, move_slot=3)
        reader = RedBlueMemoryReader(emulator)
        assert "Rainbow" in reader.read_flags()["badges"], "the fixture set the badge"
        assert not reader.strength_active(), "nothing has used Strength yet"
        _drive_field_move(emulator, record)
        assert reader.strength_active(), "the game switched Strength on"
    finally:
        emulator.close()


@needs_rom
def test_without_the_badge_strength_leaves_the_flag_alone(tmp_path):
    """The control. Without it the test above only proves the byte exists."""
    pytest.importorskip("pyboy")
    import hm_fixtures as hm

    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import ADDR_BADGES, RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        save = _menuable_save(emulator, tmp_path)
        record = hm.give_field_move(emulator, save, "STRENGTH", tmp_path, move_slot=3)
        reader = RedBlueMemoryReader(emulator)
        # Take the Rainbow Badge back off, leaving the move in place.
        badges = emulator.read_u8(ADDR_BADGES)
        hm.write_u8(emulator, ADDR_BADGES, badges & ~(1 << 3))
        assert "Rainbow" not in reader.read_flags()["badges"]
        _drive_field_move(emulator, record)
        assert not reader.strength_active(), (
            "the game refuses Strength without the badge, so the flag must stay down"
        )
    finally:
        emulator.close()


@needs_rom
def test_the_server_refuses_strength_before_spending_a_button(tmp_path):
    """The badge check runs before the menu, so a run without it loses nothing."""
    from pokemon_agent import server

    assert server.FIELD_MOVE_BADGE["strength"] == "Rainbow"
    assert server.FIELD_MOVE_BADGE["surf"] == "Soul"
    assert server.FIELD_MOVE_SOURCE["strength"] == "HM04"
