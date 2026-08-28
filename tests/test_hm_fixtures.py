"""The HM fixture builder, checked against the ROM rather than against itself.

`hm_fixtures` exists so Surf, Strength, Flash and Fly can be driven at all: no
save in the corpus carries HM02..HM05 and none has the Soul or Rainbow badge, so
without a constructed state those four mechanisms would ship unverified. A
fixture builder that writes to the wrong byte would be worse than none -- it
would produce a green test for a mechanism that never fired.

So the writes are read back through the harness's *own* decoder, not through the
offsets that wrote them, and the end of the file drives the real game: the
doctored Surf is selected from the party menu and the game's own
wWalkBikeSurfState is asked whether the player is on the water.

Skipped when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

import glob
from pathlib import Path

import hm_fixtures
import pytest
from hm_fixtures import FIELD_MOVES, NUM_MOVES, OFF_MOVES, OFF_PP

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

#: Menu screens, by wTopMenuItemY. Measured in tests/test_field_moves_live.py.
START_MENU_TOP_Y = 2
PARTY_MENU_TOP_Y = 1

#: The water tile id every overworld-style tileset uses, from pokered
#: `IsNextTileShoreOrWater` (`cp $14`). mapdecode shifts every tile id by 0x100
#: to land in the same space the ledge and tile-pair tables use, so the id the
#: decoded map reports is 0x114.
WATER_TILE = 0x114
#: pokered data/tilesets/water_tilesets.asm -- the tilesets where $14 is water.
#: Without this filter, tile $14 in a HOUSE or POKECENTER tileset (a table, a
#: potted plant) reads as a lake.
WATER_TILESETS = frozenset(
    {"OVERWORLD", "FOREST", "DOJO", "GYM", "SHIP", "SHIP_PORT", "CAVERN", "FACILITY", "PLATEAU"}
)

DIRECTION_STEPS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

#: How far the surf check is willing to walk to reach the shore. Every step is a
#: chance of a wild encounter, and a long hike through grass turns a mechanism
#: check into a coin flip.
MAX_WALK_TO_SHORE = 30
#: How many (stand, facing) pairs to try before calling it a failure.
SHORE_ATTEMPTS = 4


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


@pytest.fixture(scope="module")
def reader(emulator):
    from pokemon_agent.memory.red import RedBlueMemoryReader

    return RedBlueMemoryReader(emulator)


def _corpus() -> list[str]:
    return sorted(glob.glob(str(SAVES_DIR / "*.state")))


@pytest.fixture(scope="module")
def teachable_save(emulator, reader, tmp_path_factory) -> str:
    """A save whose first party member has an empty move slot.

    Searched rather than named: the corpus is mostly autosaves, which a live run
    prunes as it goes, so a pinned filename would rot within the week.
    """
    if SAVES_DIR is None:  # pragma: no cover -- guarded by needs_rom
        pytest.skip("no saves directory")
    scratch = tmp_path_factory.mktemp("teachable")
    for path in _corpus():
        copy = hm_fixtures.copy_state(path, scratch, "probe.state")
        try:
            emulator.load_state(str(copy))
            emulator.settle()
        except Exception:  # noqa: BLE001 -- a save that will not load is not a finding
            continue
        from pokemon_agent.memory.red import ADDR_PARTY_COUNT

        if emulator.read_u8(ADDR_PARTY_COUNT) < 1:
            continue
        if hm_fixtures.free_move_slot(emulator, 0) is None:
            continue
        return path
    pytest.skip("no save in the corpus has a party member with a free move slot")


# ---------------------------------------------------------------------------
# The write lands where the harness's own decoder looks
# ---------------------------------------------------------------------------


@needs_rom
@pytest.mark.parametrize("move_name", sorted(FIELD_MOVES))
def test_the_move_and_the_badge_read_back_through_the_harness(
    emulator, reader, teachable_save, tmp_path, move_name
):
    """read_party() and read_flags() are the proof, not the offsets that wrote."""
    record = hm_fixtures.give_field_move(
        emulator, teachable_save, move_name, tmp_path, give_hm_item=True
    )

    mon = reader.read_party()[record.party_slot]
    taught = [m for m in mon["moves"] if m["id"] == record.move.move_id]
    assert taught, f"{record.move.name} is not in {[m['name'] for m in mon['moves']]}"
    assert len(taught) == 1
    assert taught[0]["name"] == record.move.name
    assert taught[0]["pp"] == record.move.max_pp
    assert taught[0]["pp_up"] == 0

    assert record.move.badge in reader.read_flags()["badges"]
    assert record.move.badge in reader.read_player()["badges"]

    bag = {item["id"]: item for item in reader.read_bag()}
    assert record.hm_item_id in bag
    assert bag[record.hm_item_id]["item"] == record.move.hm_item_name


@needs_rom
def test_the_badge_bit_is_the_one_pokered_checks(emulator, reader, teachable_save, tmp_path):
    """Boulder/Cascade/Thunder/Rainbow/Soul, one bit each, nothing else moved."""
    from pokemon_agent.memory.red import ADDR_BADGES, BADGE_NAMES

    for move_name, spec in sorted(FIELD_MOVES.items()):
        record = hm_fixtures.give_field_move(emulator, teachable_save, move_name, tmp_path)
        after = emulator.read_u8(ADDR_BADGES)
        assert after == record.badges_before | (1 << spec.badge_bit)
        assert BADGE_NAMES[spec.badge_bit] == spec.badge
        # Exactly one bit turned on, and none turned off.
        assert after & record.badges_before == record.badges_before
        assert bin(after ^ record.badges_before).count("1") <= 1


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------


@needs_rom
def test_doctoring_leaves_everything_but_the_one_move_slot_alone(
    emulator, reader, teachable_save, tmp_path
):
    """Species, level, HP, stats and the other three move slots survive."""
    from pokemon_agent.memory.red import ADDR_PARTY_COUNT

    baseline = hm_fixtures.copy_state(teachable_save, tmp_path, "baseline.state")
    emulator.load_state(str(baseline))
    emulator.settle()
    party_before = reader.read_party()
    raw_before = [
        hm_fixtures.raw_party_mon(emulator, slot)
        for slot in range(emulator.read_u8(ADDR_PARTY_COUNT))
    ]

    record = hm_fixtures.give_field_move(emulator, teachable_save, "SURF", tmp_path)
    party_after = reader.read_party()
    raw_after = [
        hm_fixtures.raw_party_mon(emulator, slot)
        for slot in range(emulator.read_u8(ADDR_PARTY_COUNT))
    ]

    assert len(party_after) == len(party_before)
    assert record.replaced_move_id == 0, "the fixture chose an occupied move slot"

    for slot, (before, after) in enumerate(zip(party_before, party_after)):
        for key in ("species", "level", "hp", "max_hp", "status", "types", "stats", "nickname"):
            assert after[key] == before[key], f"slot {slot} {key} changed"
        if slot != record.party_slot:
            assert after["moves"] == before["moves"], f"slot {slot} moves changed"

    # And byte for byte, so that a field the reader does not decode -- DVs,
    # experience, EVs, the catch rate byte -- cannot drift unnoticed.
    changed = record.party_slot
    for slot, (before, after) in enumerate(zip(raw_before, raw_after)):
        if slot != changed:
            assert after == before, f"party slot {slot} changed and should not have"
    was, now = raw_before[changed], raw_after[changed]
    differing = {i for i in range(len(was)) if was[i] != now[i]}
    assert differing == {
        OFF_MOVES + record.move_slot,
        OFF_PP + record.move_slot,
    }, f"unexpected bytes changed in the doctored slot: {sorted(differing)}"

    mon_before, mon_after = party_before[changed], party_after[changed]
    assert len(mon_after["moves"]) == len(mon_before["moves"]) + 1
    kept = [m for m in mon_after["moves"] if m["id"] != record.move.move_id]
    assert kept == mon_before["moves"], "the moves it already knew were disturbed"


@needs_rom
def test_the_save_on_disk_is_never_written(emulator, teachable_save, tmp_path):
    """The corpus is read-only data. Prove it byte for byte."""
    before = hm_fixtures.sha256(teachable_save)
    size_before = Path(teachable_save).stat().st_size

    for move_name in sorted(FIELD_MOVES):
        record = hm_fixtures.give_field_move(
            emulator, teachable_save, move_name, tmp_path, give_hm_item=True
        )
        assert record.state != Path(teachable_save)
        assert record.state.parent == Path(tmp_path)

    assert hm_fixtures.sha256(teachable_save) == before
    assert Path(teachable_save).stat().st_size == size_before


@needs_rom
def test_an_occupied_move_slot_can_be_overwritten_on_purpose(
    emulator, reader, teachable_save, tmp_path
):
    """The default is non-destructive; overwriting has to be asked for."""
    record = hm_fixtures.give_field_move(emulator, teachable_save, "FLY", tmp_path, move_slot=0)
    mon = reader.read_party()[0]
    ids = hm_fixtures.move_ids(emulator, 0)
    assert ids[0] == FIELD_MOVES["FLY"].move_id
    assert record.replaced_move_id != 0
    assert record.notes, "an overwrite should be recorded"
    assert mon["moves"][0]["name"] == "Fly"
    assert len([m for m in mon["moves"] if m["id"] != 0]) == len([i for i in ids if i != 0])


@needs_rom
def test_open_doctored_hands_back_a_loaded_driveable_emulator(teachable_save, tmp_path):
    """The one-call entry point, on its own emulator, driven far enough to matter."""
    pytest.importorskip("pyboy")
    from pokemon_agent.memory.red import ADDR_TOP_MENU_ITEM_Y, RedBlueMemoryReader

    emulator, record = hm_fixtures.open_doctored(
        SAVES_DIR / "PokemonRed.gb",
        teachable_save,
        "flash",  # lower case on purpose: the lookup is case-insensitive
        tmp_path,
        state_name="opened.state",
    )
    try:
        own_reader = RedBlueMemoryReader(emulator)
        assert record.move.name == "Flash"
        assert "Flash" in [m["name"] for m in own_reader.read_party()[0]["moves"]]
        assert "Boulder" in own_reader.read_flags()["badges"]
        emulator.press_and_settle("start")
        assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y
    finally:
        emulator.close()


@needs_rom
def test_a_four_move_pokemon_is_refused_rather_than_silently_clobbered(
    emulator, teachable_save, tmp_path
):
    """A fixture that quietly ate a move would be a fixture nobody could trust."""
    record = hm_fixtures.give_field_move(emulator, teachable_save, "CUT", tmp_path)
    base = hm_fixtures.party_slot_base(record.party_slot)
    for slot in range(NUM_MOVES):
        if hm_fixtures.move_ids(emulator, record.party_slot)[slot] == 0:
            hm_fixtures.write_u8(emulator, base + OFF_MOVES + slot, 33)  # Tackle
            hm_fixtures.write_u8(emulator, base + OFF_PP + slot, 35)
    emulator.save_state(str(tmp_path / "full.state"))

    with pytest.raises(ValueError, match="four moves"):
        hm_fixtures.give_field_move(
            emulator, tmp_path / "full.state", "SURF", tmp_path, state_name="retry.state"
        )


# ---------------------------------------------------------------------------
# End to end: the doctored state actually behaves
# ---------------------------------------------------------------------------


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


def _shore_candidates(emulator, reader) -> list[tuple[tuple[int, int], str, int]]:
    """Reachable tiles that have real water next to them, shortest walk first."""
    from pokemon_agent import world

    terrain = emulator._read_map_terrain()
    if not terrain or terrain["tileset"] not in WATER_TILESETS:
        return []
    water = [c for c, tile in terrain["tile_ids"].items() if tile == WATER_TILE]
    if not water:
        return []
    position = reader.read_coordinates()
    if position is None:
        return []
    region = world.reachable_region(_collision(terrain), position)

    found: list[tuple[tuple[int, int], str, int]] = []
    for tile in water:
        for direction, (dx, dy) in DIRECTION_STEPS.items():
            stand = (tile[0] - dx, tile[1] - dy)
            if stand not in terrain["walkable"]:
                continue
            actions = region.actions_to(stand)
            if actions is None or len(actions) > MAX_WALK_TO_SHORE:
                continue
            found.append((stand, direction, len(actions)))
    found.sort(key=lambda entry: entry[2])
    return found


@pytest.fixture(scope="module")
def surfable_save(emulator, reader, tmp_path_factory):
    """A save standing on dry land within a short walk of open water."""
    if SAVES_DIR is None:  # pragma: no cover -- guarded by needs_rom
        pytest.skip("no saves directory")
    from pokemon_agent.memory.red import ADDR_PARTY_COUNT, ADDR_WALK_BIKE_SURF, ON_FOOT

    scratch = tmp_path_factory.mktemp("surfable")
    for path in _corpus():
        copy = hm_fixtures.copy_state(path, scratch, "probe.state")
        try:
            emulator.load_state(str(copy))
            emulator.settle()
        except Exception:  # noqa: BLE001
            continue
        if emulator.read_u8(ADDR_WALK_BIKE_SURF) != ON_FOOT:
            continue
        if emulator.read_u8(ADDR_PARTY_COUNT) < 1:
            continue
        if hm_fixtures.free_move_slot(emulator, 0) is None:
            continue
        candidates = _shore_candidates(emulator, reader)
        if candidates:
            return path, candidates[:SHORE_ATTEMPTS]
    pytest.skip("no save in the corpus stands within reach of water with a free move slot")


def _walk_to(emulator, reader, target: tuple[int, int]) -> bool:
    from pokemon_agent import world

    collision = _collision(emulator._read_map_terrain())
    region = world.reachable_region(collision, reader.read_coordinates())
    actions = region.actions_to(target)
    if actions is None:
        return False
    for action in actions:
        emulator.press_and_settle(action.removeprefix("walk_"))
    return reader.read_coordinates() == target


def _walk_menu_cursor(emulator, target_row: int, limit: int = 14) -> int:
    from pokemon_agent.memory.red import ADDR_CURRENT_MENU_ITEM

    for _ in range(limit):
        at = emulator.read_u8(ADDR_CURRENT_MENU_ITEM)
        if at == target_row:
            break
        emulator.press_and_settle("up" if at > target_row else "down")
    return emulator.read_u8(ADDR_CURRENT_MENU_ITEM)


def _open_field_move_submenu(emulator, reader, party_slot: int) -> None:
    """START -> POKEMON -> the chosen mon, leaving its submenu open."""
    from pokemon_agent.memory.red import ADDR_TOP_MENU_ITEM_Y

    emulator.press_and_settle("start")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == START_MENU_TOP_Y
    pokemon_row = 1 if reader.read_flags()["has_pokedex"] else 0
    assert _walk_menu_cursor(emulator, pokemon_row) == pokemon_row

    emulator.press_and_settle("a")
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == PARTY_MENU_TOP_Y
    assert _walk_menu_cursor(emulator, party_slot) == party_slot

    emulator.press_and_settle("a")


@needs_rom
def test_the_party_submenu_grows_a_row_for_the_doctored_field_move(
    emulator, reader, teachable_save, tmp_path
):
    """The game itself agrees the write produced a field move.

    The submenu is the field moves the mon knows, then STATS, SWITCH, CANCEL,
    and pokered builds its geometry from that list -- wMaxMenuItem starts at 2
    and gains one per field move, wTopMenuItemY starts at 12 and loses two. If
    the byte had landed anywhere but a move slot, or held a value the game does
    not read as Surf, neither number would move.
    """
    from pokemon_agent.memory.red import ADDR_MAX_MENU_ITEM, ADDR_TOP_MENU_ITEM_Y

    baseline = hm_fixtures.copy_state(teachable_save, tmp_path, "before.state")
    emulator.load_state(str(baseline))
    emulator.settle()
    _open_field_move_submenu(emulator, reader, 0)
    max_before = emulator.read_u8(ADDR_MAX_MENU_ITEM)
    top_y_before = emulator.read_u8(ADDR_TOP_MENU_ITEM_Y)

    record = hm_fixtures.give_field_move(emulator, teachable_save, "SURF", tmp_path)
    assert record.field_moves_after == record.field_moves_before + 1
    _open_field_move_submenu(emulator, reader, record.party_slot)
    max_after = emulator.read_u8(ADDR_MAX_MENU_ITEM)
    top_y_after = emulator.read_u8(ADDR_TOP_MENU_ITEM_Y)

    assert max_after == max_before + 1, "the submenu did not gain a row"
    assert top_y_after == top_y_before - 2, "the submenu did not grow upward by one row"
    assert max_after == record.expected_max_menu_item
    assert top_y_after == record.expected_top_menu_item_y


def _stand_on_the_shore(
    emulator, reader, record, stand: tuple[int, int], direction: str
) -> str | None:
    """Walk to *stand* and face *direction*. Returns why it could not, or None."""
    from pokemon_agent.memory.red import ADDR_BATTLE_TYPE

    if not _walk_to(emulator, reader, stand):
        return f"could not walk to {stand}"
    if emulator.read_u8(ADDR_BATTLE_TYPE) != 0:
        return f"a wild encounter interrupted the walk to {stand}"
    emulator.press_and_settle(direction)
    if reader.read_facing().lower() != direction:
        return f"could not face {direction} at {stand}"
    if reader.read_coordinates() != stand:
        return f"facing {direction} walked off {stand}"
    return None


def _choose_surf_from_the_party_menu(emulator, reader, record) -> None:
    """START -> POKEMON -> the mon -> SURF, then press through whatever it says."""
    from pokemon_agent.memory.red import (
        ADDR_MAX_MENU_ITEM,
        ADDR_TOP_MENU_ITEM_Y,
        ADDR_WALK_BIKE_SURF,
        SURFING,
    )

    _open_field_move_submenu(emulator, reader, record.party_slot)
    assert emulator.read_u8(ADDR_TOP_MENU_ITEM_Y) == record.expected_top_menu_item_y
    assert emulator.read_u8(ADDR_MAX_MENU_ITEM) == record.expected_max_menu_item
    assert _walk_menu_cursor(emulator, record.submenu_row) == record.submenu_row

    emulator.press_and_settle("a")  # SURF
    for _ in range(8):
        if emulator.read_u8(ADDR_WALK_BIKE_SURF) == SURFING:
            return
        emulator.press_and_settle("a")


@needs_rom
def test_the_doctored_surf_actually_puts_the_player_on_the_water(
    emulator, reader, surfable_save, tmp_path
):
    """wWalkBikeSurfState, which nothing but a real Surf can set to 2.

    This is the claim the whole fixture exists to support: a Pokemon given SURF
    in RAM, on a player given the Soul badge in RAM, can be walked to a shore
    and used, and the game moves the player onto a tile the blockset calls
    solid.

    The same run is then replayed with the Soul badge bit cleared and nothing
    else changed. pokered's `.surf` reads wObtainedBadges and refuses before it
    looks at anything else, so that replay must stay on foot -- which is what
    makes the badge write load-bearing rather than decorative.
    """
    from pokemon_agent.memory.red import ADDR_BADGES, ADDR_WALK_BIKE_SURF, ON_FOOT, SURFING

    save_path, candidates = surfable_save
    failures: list[str] = []

    for attempt, (stand, direction, _distance) in enumerate(candidates):
        step = DIRECTION_STEPS[direction]
        water = (stand[0] + step[0], stand[1] + step[1])
        record = hm_fixtures.give_field_move(
            emulator,
            save_path,
            "SURF",
            tmp_path,
            give_hm_item=True,
            state_name=f"surf{attempt}.state",
        )
        assert emulator.read_u8(ADDR_WALK_BIKE_SURF) == ON_FOOT

        blocked = _stand_on_the_shore(emulator, reader, record, stand, direction)
        if blocked:
            failures.append(blocked)
            continue

        _choose_surf_from_the_party_menu(emulator, reader, record)
        if emulator.read_u8(ADDR_WALK_BIKE_SURF) != SURFING:
            failures.append(f"{stand} facing {direction}: still on foot")
            continue

        # Getting on queues a *simulated* step forward, which the engine only
        # runs once the "got on" text is dismissed and the overworld is back in
        # control -- so the state byte flips several presses before the player
        # actually leaves the shore.
        for _ in range(8):
            emulator.settle()
            if reader.read_coordinates() == water:
                break
            emulator.press_and_settle("a")
        assert reader.read_coordinates() == water, (
            "the state byte says surfing but the player never left the shore"
        )
        terrain = emulator._read_map_terrain()
        assert water not in terrain["walkable"], (
            "the tile the player ended up on is not the one the blockset calls solid"
        )

        # The control: identical state, identical inputs, badge bit cleared.
        control = hm_fixtures.give_field_move(
            emulator,
            save_path,
            "SURF",
            tmp_path,
            give_hm_item=True,
            state_name=f"surf{attempt}_nobadge.state",
        )
        hm_fixtures.write_u8(
            emulator, ADDR_BADGES, control.badges_before & ~(1 << control.move.badge_bit)
        )
        assert record.move.badge not in reader.read_flags()["badges"]
        assert _stand_on_the_shore(emulator, reader, control, stand, direction) is None
        _choose_surf_from_the_party_menu(emulator, reader, control)
        assert emulator.read_u8(ADDR_WALK_BIKE_SURF) == ON_FOOT, (
            "Surf worked without the Soul badge, so the badge write proves nothing"
        )
        return

    pytest.fail("the doctored Surf never took: " + "; ".join(failures))
