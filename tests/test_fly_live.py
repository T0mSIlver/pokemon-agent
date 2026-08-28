"""Fly, driven on a save built to have it.

The measurement that motivated this: on one journey, GLM 5.3 Flash spent 2,278
presses where qwen38-27b spent 18,326, and the difference was almost entirely
navigation — 352 `goto` calls against 344 `goto` plus 1,913 raw `action`. Walking
is what this run spends its presses on, and Fly is the one move that removes a
journey rather than shortening it.

Unlike Flash, its effect is unmistakable and checkable: the map changes. What was
not obvious was how to *choose* the destination. `wCurrentMenuItem` does not move
on the town map — measured, it sits at 1 however far the cursor travels — so the
readable thing is the caption the game draws, which says "To PALLET TOWN" and
changes with every press. That is what `_fly_destination_sync` reads.
"""

from __future__ import annotations

import sys
import tempfile
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

START_MENU_POKEMON_ROW = 1


def _open_town_map(emulator, record):
    """START -> POKEMON -> the mon -> FLY, stopping on the town map."""
    import hm_fixtures as hm

    from pokemon_agent.memory.red import ADDR_CURRENT_MENU_ITEM

    emulator.settle()
    emulator.press_and_settle("start")
    for _ in range(12):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == START_MENU_POKEMON_ROW:
            break
        emulator.press_and_settle("up" if at > START_MENU_POKEMON_ROW else "down")
    emulator.press_and_settle("a")
    emulator.press_and_settle("a")
    ids = hm.move_ids(emulator, 0)
    rows = [index for index, move_id in enumerate(ids) if move_id in hm.FIELD_MOVE_IDS]
    for _ in range(rows.index(record.move_slot)):
        emulator.press_and_settle("down")
    emulator.press_and_settle("a")
    emulator.tick(60)


def _flyable_save(emulator, tmp_path):
    """A save outdoors, on foot, holding the badge Fly needs.

    Chosen by property rather than by name. This test used to name
    `pre_cut_restart.state`, which is the file the live run rewrites on every
    restart: the day it was banked in Celadon Gym -- indoors, where the town map
    behaves differently -- all three tests here failed, and none of them had
    anything to do with Fly. A corpus that the run keeps editing cannot be
    addressed by filename.
    """
    import glob
    import shutil

    from pokemon_agent.memory.red import ADDR_WALK_BIKE_SURF, ON_FOOT, RedBlueMemoryReader

    reader = RedBlueMemoryReader(emulator)
    for path in sorted(glob.glob(str(SAVES_DIR / "*.state")), reverse=True):
        copy = tmp_path / "flyable.state"
        shutil.copy(path, copy)
        try:
            emulator.load_state(str(copy))
            emulator.settle()
            flags = reader.read_flags()
            tileset = reader.read_tileset()
        except Exception:  # noqa: BLE001 — a save that will not load proves nothing
            continue
        if (
            tileset == "OVERWORLD"
            and "Thunder" in flags["badges"]
            and emulator.read_u8(ADDR_WALK_BIKE_SURF) == ON_FOOT
        ):
            return path
    pytest.skip("no save in the corpus is outdoors, on foot, with the Thunder Badge")


def _doctored(tmp_path):
    import hm_fixtures as hm

    from pokemon_agent.emulator import PyBoyEmulator

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        save = _flyable_save(emulator, tmp_path)
        record = hm.give_field_move(emulator, save, "FLY", tmp_path, move_slot=3)
    except BaseException:
        emulator.close()
        raise
    return emulator, record


@needs_rom
def test_the_town_map_names_where_the_cursor_is(tmp_path):
    """The caption is the only thing that moves, and it names the destination."""
    pytest.importorskip("pyboy")
    from pokemon_agent.memory.red import ADDR_CURRENT_MENU_ITEM, RedBlueMemoryReader

    emulator, record = _doctored(tmp_path)
    try:
        reader = RedBlueMemoryReader(emulator)
        _open_town_map(emulator, record)

        seen, cursor_values = [], set()
        for _ in range(8):
            text = (reader.read_screen_text() or "").strip()
            assert text.startswith("To "), text
            seen.append(text[3:].strip())
            cursor_values.add(emulator.read_u8(ADDR_CURRENT_MENU_ITEM))
            emulator.press_and_settle("up")
            emulator.tick(20)

        assert len(set(seen)) >= 2, f"the cursor must move: {seen}"
        assert len(cursor_values) == 1, (
            "wCurrentMenuItem does not track this screen, which is why the caption "
            f"is read instead: {cursor_values}"
        )
    finally:
        emulator.close()


@needs_rom
def test_flying_lands_on_the_town_the_caption_named(tmp_path):
    """The claim the verb rests on, pressed rather than reasoned about."""
    pytest.importorskip("pyboy")
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator, record = _doctored(tmp_path)
    try:
        reader = RedBlueMemoryReader(emulator)
        before = reader.read_map_info().get("map_name")
        _open_town_map(emulator, record)

        # Walk to a caption naming somewhere other than where we are.
        chosen = None
        for _ in range(10):
            town = (reader.read_screen_text() or "").strip()[3:].strip()
            if town and town.title() != (before or "").title():
                chosen = town.title()
                break
            emulator.press_and_settle("up")
            emulator.tick(20)
        assert chosen, "the town map offered nowhere to go"

        emulator.press_and_settle("a")
        for _ in range(12):
            emulator.press_and_settle("a")
        emulator.settle()

        landed = (reader.read_map_info().get("map_name") or "").title()
        assert landed != (before or "").title(), "the map must change"
        assert landed == chosen, f"asked for {chosen}, landed on {landed}"
    finally:
        emulator.close()


@needs_rom
def test_the_destination_reader_parses_the_caption(tmp_path):
    """`_fly_destination_sync` is what the server uses; check it on a real screen."""
    pytest.importorskip("pyboy")
    from pokemon_agent import server
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator, record = _doctored(tmp_path)
    try:
        _open_town_map(emulator, record)
        # Both globals: the reader for the caption, and the emulator because the
        # function ticks before reading it -- the caption is a frame behind the
        # press that moved the cursor.
        original_reader, original_emulator = server._reader, server._emulator
        server._reader = RedBlueMemoryReader(emulator)
        server._emulator = emulator
        try:
            town = server._fly_destination_sync()
        finally:
            server._reader, server._emulator = original_reader, original_emulator
        assert town and town == town.title(), town
        assert "To" not in town.split()
    finally:
        emulator.close()


def test_a_full_lap_of_the_town_map_is_bounded():
    """Without a bound, a town that never appears would press forever."""
    from pokemon_agent import server

    assert 8 <= server.MAX_FLY_CURSOR_PRESSES <= 40


def _unused(tmp: Path) -> Path:  # pragma: no cover - keeps tempfile imported
    return Path(tempfile.mkdtemp(dir=tmp))
