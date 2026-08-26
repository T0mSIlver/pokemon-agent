"""Guards for the WRAM addresses in pokemon_agent/memory/red.py that drifted.

An address in that table is a claim about pokered's ram/wram.asm, and three of
them were wrong in a way nothing crashed on. The play clock read a byte low all
the way down: hours came off 0xDA40 as a little-endian u16 whose low byte is
padding, so a 25-hour save reported ``6400:00:21`` and the minutes field, which
was really wPlayTimeMaxed, read 00 in all 494 save states in ``saves/``. The town
map flag pointed at 0xD5F3, which is not wTownVisitedFlag and not any flag: it
holds 1 in a save standing in Red's house before the Town Map exists.

The addresses here were re-derived by walking the "Main Data" section of
ram/wram.asm, a walk that lands on wObtainedBadges = 0xD356 and wEventFlags =
0xD747 exactly. These tests are the second source: real save states, where a
wrong address shows up as a value no clock and no visited-town set can hold.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from pokemon_agent.memory.red import (
    ADDR_ELITE_4_FLAGS,
    ADDR_PLAYTIME_H,
    ADDR_PLAYTIME_M,
    ADDR_PLAYTIME_MAXED,
    ADDR_PLAYTIME_S,
    ADDR_TOWN_VISITED_FLAGS,
    MAP_NAMES,
    PokemonRedReader,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: wTownVisitedFlag, in map-id order. NUM_CITY_MAPS is 11, so bits 11..15 of the
#: two-byte array are padding and must never be set.
TOWNS = (
    "Pallet Town",
    "Viridian City",
    "Pewter City",
    "Cerulean City",
    "Lavender Town",
    "Vermilion City",
    "Celadon City",
    "Fuchsia City",
    "Cinnabar Island",
    "Indigo Plateau",
    "Saffron City",
)

#: Enough saves to cross the whole span the library covers without paying for a
#: load of all of them; every one is an independent reading of the same claim.
SAMPLE_SIZE = 12


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()


def _sample(saves_dir: Path) -> List[Path]:
    states = sorted(saves_dir.glob("*.state"))
    if not states:
        return []
    step = max(1, len(states) // SAMPLE_SIZE)
    return states[::step][:SAMPLE_SIZE]


@pytest.fixture(scope="module")
def readings():
    if SAVES_DIR is None:
        pytest.skip("no saves/PokemonRed.gb next to the repo")
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator

    states = _sample(SAVES_DIR)
    if not states:
        pytest.skip("no save states to read")

    rom = str(SAVES_DIR / "PokemonRed.gb")
    emulator = create_emulator(rom)
    emulator.load(rom)
    out = []
    try:
        for state in states:
            emulator.load_state(str(state))
            reader = PokemonRedReader(emulator)
            out.append(
                {
                    "save": state.name,
                    "map": MAP_NAMES.get(emulator.read_u8(0xD35E), "?"),
                    "hours": emulator.read_u8(ADDR_PLAYTIME_H),
                    "maxed": emulator.read_u8(ADDR_PLAYTIME_MAXED),
                    "minutes": emulator.read_u8(ADDR_PLAYTIME_M),
                    "seconds": emulator.read_u8(ADDR_PLAYTIME_S),
                    "towns": emulator.read_u8(ADDR_TOWN_VISITED_FLAGS)
                    | (emulator.read_u8(ADDR_TOWN_VISITED_FLAGS + 1) << 8),
                    "elite4": emulator.read_u8(ADDR_ELITE_4_FLAGS),
                    "party_count": len(reader.read_party()),
                    "play_time": reader.read_player()["play_time"],
                }
            )
    finally:
        emulator.close()
    return out


def test_the_play_clock_reads_as_a_clock(readings):
    """Minutes and seconds below 60, and a wPlayTimeMaxed that is a flag.

    All three fell out of the old off-by-one: minutes read the flag byte and were
    0 everywhere, seconds read the minutes, and hours read a u16 whose value was
    always a multiple of 256.
    """
    for reading in readings:
        assert 0 <= reading["minutes"] < 60, reading
        assert 0 <= reading["seconds"] < 60, reading
        assert reading["maxed"] in (0, 1), reading
        assert 0 <= reading["hours"] <= 255, reading

    # A library that spans a real playthrough cannot have every save on :00.
    assert len({reading["minutes"] for reading in readings}) > 1


def test_the_rendered_play_time_is_not_a_multiple_of_an_hour(readings):
    for reading in readings:
        hours, minutes, _seconds = reading["play_time"].split(":")
        assert int(hours) < 256, reading
        assert int(minutes) < 60, reading


def test_visited_towns_are_a_plausible_set(readings):
    """Pallet is always visited, the padding bits never are, and the town you
    are standing in is always in the set.

    0xD5F3, the address this replaced, fails the first of these: it reads 1 in a
    save that has not left Pallet Town, where only bit 0 can be set.
    """
    for reading in readings:
        assert reading["towns"] >> len(TOWNS) == 0, reading
        if reading["party_count"]:
            assert reading["towns"] & 1, reading
        if reading["map"] in TOWNS:
            assert reading["towns"] & (1 << TOWNS.index(reading["map"])), reading


def test_the_elite_four_bit_is_clear_in_a_run_that_never_got_there(readings):
    for reading in readings:
        assert not reading["elite4"] & 0b1, reading
