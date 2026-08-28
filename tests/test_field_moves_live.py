"""Cut checked against the cartridge, not against a model of it.

`poke cut` drives four menu screens blind-spot free by reading wTopMenuItemY to
tell them apart and wCurrentMenuItem to place the cursor. Those constants are
the whole mechanism, and none of them can be derived — they are what the ROM
happens to do. So they are measured here.

This matters more than the usual constant check. The run this replaces got all
the way to the fourth screen by hand, looked at `CUT STATS SWITCH CANCEL` on a
160x144 display, read it as `FIGHT STATUS SWITCH CANCEL`, concluded that Gen 1
has no field moves outside battle at all, and never tried again in 19,000
further calls. Reading the menu out of memory is the fix for that; a wrong
constant would put it straight back.

Skipped when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

import pytest

from pokemon_agent import capabilities, world
from pokemon_agent.navigation import CUT_TREE_TILES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

#: Menu screens `_field_cut_sync` walks through, by wTopMenuItemY.
START_MENU_TOP_Y = 2
PARTY_MENU_TOP_Y = 1
FIELD_MOVE_MENU_TOP_Y = 10
#: Row of POKEMON in the start menu, with the Pokedex in hand.
START_MENU_POKEMON_ROW = 1
#: And of ITEM, one below it. The bag list it opens has its own anchor.
BAG_MENU_ITEM_ROW = 2
BAG_MENU_TOP_Y = 4


@pytest.fixture(scope="module")
def emulator():
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import PyBoyEmulator

    emu = PyBoyEmulator()
    emu.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        yield emu
    finally:
        emu.close()


def _cuttable_save(emulator, tmp_path) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    """A save whose player can walk up to a tree, and where the tree is.

    Searched rather than named: the corpus is mostly auto-saves, which are
    pruned as a run goes on, so pinning one file would make this test rot.
    """
    from pokemon_agent.memory.red import RedBlueMemoryReader

    reader = RedBlueMemoryReader(emulator)
    for path in sorted(glob.glob(str(SAVES_DIR / "*.state")), reverse=True):
        copy = tmp_path / "probe.state"
        shutil.copy(path, copy)
        try:
            emulator.load_state(str(copy))
            emulator.settle()
            terrain = emulator._read_map_terrain()
            position = reader.read_coordinates()
        except Exception:  # noqa: BLE001 — a save that will not load is not a failure here
            continue
        trees = CUT_TREE_TILES.get(terrain["tileset"], frozenset())
        if not trees or position is None:
            continue
        collision = {
            "width": terrain["width"],
            "height": terrain["height"],
            "walkable": terrain["walkable"],
            "sprites": [],
            "warps": [],
            "live": set(),
            "seen": set(terrain["tile_ids"]),
            "ledges": {},
            "ground_truth": True,
            "tile_ids": terrain["tile_ids"],
            "tileset": terrain["tileset"],
        }
        region = world.reachable_region(collision, position)
        try:
            plan = capabilities.cut_plan(collision, region)
        except capabilities.CapabilityError:
            continue
        if any(
            "cut" == str(move.get("name", "")).lower()
            for mon in reader.read_party() or []
            for move in mon.get("moves") or []
        ):
            return str(copy), plan["tree"], plan["stand"]
    return None


@needs_rom
def test_the_menu_screens_are_told_apart_by_top_menu_item_y(emulator, tmp_path):
    """START, the party list and the field-move submenu, in that order."""
    from pokemon_agent.memory.red import (
        ADDR_CURRENT_MENU_ITEM,
        ADDR_MAX_MENU_ITEM,
        ADDR_TOP_MENU_ITEM_Y,
    )

    found = _cuttable_save(emulator, tmp_path)
    if found is None:
        pytest.skip("no save in the corpus stands near a tree with Cut in the party")
    state, _tree, _stand = found
    emulator.load_state(state)
    emulator.settle()

    emulator.press_and_settle("start")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y

    # The cursor is remembered between openings and it wraps, so it is walked
    # onto the row rather than counted onto it.
    for _ in range(12):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == START_MENU_POKEMON_ROW:
            break
        emulator.press_and_settle("up" if at > START_MENU_POKEMON_ROW else "down")
    assert emulator.read_u8(ADDR_CURRENT_MENU_ITEM) == START_MENU_POKEMON_ROW

    emulator.press_and_settle("a")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == PARTY_MENU_TOP_Y

    emulator.press_and_settle("a")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == FIELD_MOVE_MENU_TOP_Y
    # Field moves, then STATS, SWITCH, CANCEL. `_field_cut_sync` checks this
    # against the moveset before confirming, because a mismatch means the row it
    # worked out is not the row the game drew.
    assert emulator.read_u8(ADDR_MAX_MENU_ITEM) >= 3


@needs_rom
def test_cut_actually_opens_the_tile_the_blockset_calls_solid(emulator, tmp_path):
    """The claim the whole feature rests on, pressed rather than reasoned about."""
    from pokemon_agent.memory.red import ADDR_CURRENT_MENU_ITEM

    found = _cuttable_save(emulator, tmp_path)
    if found is None:
        pytest.skip("no save in the corpus stands near a tree with Cut in the party")
    state, tree, stand = found
    emulator.load_state(state)
    emulator.settle()

    before = emulator._read_map_terrain()
    assert tree not in before["walkable"], "the tree reads as solid before it is cut"
    reachable_before = len(world.reachable_region(_collision(before), stand).order)

    _walk_to(emulator, stand)
    _face(emulator, tree)

    emulator.press_and_settle("start")
    for _ in range(12):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == START_MENU_POKEMON_ROW:
            break
        emulator.press_and_settle("up" if at > START_MENU_POKEMON_ROW else "down")
    emulator.press_and_settle("a")  # the party list
    emulator.press_and_settle("a")  # the first mon, whose row 0 is Cut
    emulator.press_and_settle("a")  # CUT
    for _ in range(8):
        emulator.press_and_settle("a")

    after = emulator._read_map_terrain()
    assert tree in after["walkable"], "the tile the blockset called solid is now walkable"
    assert len(world.reachable_region(_collision(after), stand).order) > reachable_before


def _collision(terrain: dict) -> dict:
    return {
        "width": terrain["width"],
        "height": terrain["height"],
        "walkable": terrain["walkable"],
        "sprites": [],
        "warps": [],
        "live": set(),
        "seen": set(terrain["tile_ids"]),
        "ledges": {},
        "ground_truth": True,
        "tile_ids": terrain["tile_ids"],
        "tileset": terrain["tileset"],
    }


def _walk_to(emulator, target: tuple[int, int]) -> None:
    """Walk the route the flood found. Straight lines run into the scenery."""
    from pokemon_agent.memory.red import RedBlueMemoryReader

    reader = RedBlueMemoryReader(emulator)
    collision = _collision(emulator._read_map_terrain())
    region = world.reachable_region(collision, reader.read_coordinates())
    actions = region.actions_to(target)
    assert actions is not None, f"{target} is not reachable from {reader.read_coordinates()}"
    for action in actions:
        emulator.press_and_settle(action.removeprefix("walk_"))
    assert reader.read_coordinates() == target


def _face(emulator, tree: tuple[int, int]) -> None:
    from pokemon_agent.memory.red import RedBlueMemoryReader

    reader = RedBlueMemoryReader(emulator)
    x, y = reader.read_coordinates()
    direction = (
        "right" if tree[0] > x else "left" if tree[0] < x else "down" if tree[1] > y else "up"
    )
    emulator.press_and_settle(direction)
    assert reader.read_facing().lower() == direction


@needs_rom
def test_the_bag_menu_is_told_apart_and_its_cursor_is_an_absolute_slot(emulator, tmp_path):
    """The bag scrolls, so its cursor is a row on screen plus a scroll offset.

    Counting presses down the list would work only while the bag is shorter than
    the window. The Bicycle is usually the last of a dozen items, which is
    exactly where that assumption breaks.
    """
    import glob
    import shutil

    from pokemon_agent.memory.red import (
        ADDR_CURRENT_MENU_ITEM,
        ADDR_LIST_MENU_ID,
        ADDR_LIST_SCROLL_OFFSET,
        ADDR_TOP_MENU_ITEM_Y,
        ADDR_WALK_BIKE_SURF,
        ITEM_LIST_MENU,
        ON_BIKE,
        ON_FOOT,
        RedBlueMemoryReader,
    )

    reader = RedBlueMemoryReader(emulator)
    found = None
    for path in sorted(glob.glob(str(SAVES_DIR / "*.state")), reverse=True):
        copy = tmp_path / "bike.state"
        shutil.copy(path, copy)
        try:
            emulator.load_state(str(copy))
            emulator.settle()
            bag = reader.read_bag() or []
        except Exception:  # noqa: BLE001 — a save that will not load is not a failure here
            continue
        slot = next((i for i, item in enumerate(bag) if item.get("item") == "Bicycle"), None)
        if slot is not None and emulator.read_u8(ADDR_WALK_BIKE_SURF) == ON_FOOT:
            found = (str(copy), slot, len(bag))
            break
    if found is None:
        pytest.skip("no save in the corpus carries a Bicycle while on foot")
    state, slot, size = found
    emulator.load_state(state)
    emulator.settle()

    emulator.press_and_settle("start")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y
    for _ in range(12):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == BAG_MENU_ITEM_ROW:
            break
        emulator.press_and_settle("up" if at > BAG_MENU_ITEM_ROW else "down")
    emulator.press_and_settle("a")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == BAG_MENU_TOP_Y
    assert emulator.read_u8(ADDR_LIST_MENU_ID) == ITEM_LIST_MENU

    for _ in range(2 * size + 4):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM) + emulator.read_u8(ADDR_LIST_SCROLL_OFFSET)
        if at == slot:
            break
        emulator.press_and_settle("down" if at < slot else "up")
    absolute = emulator.read_u8(ADDR_CURRENT_MENU_ITEM) + emulator.read_u8(ADDR_LIST_SCROLL_OFFSET)
    assert absolute == slot

    emulator.press_and_settle("a")  # USE / TOSS
    emulator.press_and_settle("a")  # USE
    for _ in range(6):
        emulator.press_and_settle("a")
    assert emulator.read_u8(ADDR_WALK_BIKE_SURF) == ON_BIKE
