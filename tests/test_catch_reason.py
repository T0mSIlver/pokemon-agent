"""The odds were already in the payload. The reason was not.

`_catch_line` has printed ball odds on every wild frame for a while. Measured
over the run it was built for, it did not move the behaviour at all:

    1,944 battle calls:  fight 1,010   run 922   catch 7   item 5

Seven balls thrown, 922 flights, and the run finished 43 hours in with one
Pokemon and three species registered while carrying sixteen unused Poke Balls.
Odds answer "will it work". They never answered "why bother", and the answer
existed — split across two parts of the harness that each held half of it. The
frontier knew Oak's aide wants ten species before he hands over HM05 Flash; the
battle frame knew which species was standing in front of the player. Nothing
put the two in the same sentence.

Flash is not optional for that run: Saffron is sealed by the drink guards, the
way around is Rock Tunnel, and Rock Tunnel without Flash is a dark maze.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_agent import server


def _state(*, species="Nidoran", registered=False, owned=3, balls=True, hp=(20, 30)):
    return {
        "battle": {
            "in_battle": True,
            "type": "wild",
            "enemy": {
                "species": species,
                "pokedex_id": 29,
                "hp": hp[0],
                "max_hp": hp[1],
                "level": 8,
                "catch_rate": 235,
                "registered": registered,
            },
        },
        "flags": {"pokedex_owned": owned, "pokedex_seen": 49},
        "player": {"money": 5388},
        "bag": [{"id": 4, "item": "Poke Ball", "quantity": 16}] if balls else [],
    }


def test_a_new_species_names_the_rung_it_is_short_of():
    line = server._catch_line(_state())
    assert "Poke Ball x16" in line, "the odds stay"
    assert "Nidoran is not in the Pokedex yet" in line
    assert "Got HM05 Flash" in line
    assert "7 more registered opens it" in line


def test_a_species_already_registered_gets_the_odds_and_nothing_else():
    """Not silence — the odds are still the decision. Just no manufactured urgency."""
    line = server._catch_line(_state(registered=True))
    assert "Poke Ball x16" in line
    assert "Pokedex" not in line


def test_an_unreadable_species_claims_nothing():
    """`registered` is None when the dex id could not be read.

    Saying "not in the Pokedex yet" off a failed read would be a made-up reason
    on top of a bad measurement, which is worse than no reason at all.
    """
    line = server._catch_line(_state(registered=None))
    assert "Poke Ball x16" in line
    assert "Pokedex" not in line


def test_the_reason_disappears_once_the_count_is_met():
    line = server._catch_line(_state(owned=10))
    assert "Nidoran is not in the Pokedex yet" in line
    assert "HM05" not in line, "nothing is owed, so nothing is claimed"


def test_a_trainers_pokemon_is_never_offered():
    state = _state()
    state["battle"]["type"] = "trainer"
    assert server._catch_line(state) is None


def test_with_no_balls_the_line_is_still_the_price():
    """The pre-existing behaviour, which this must not have broken."""
    line = server._catch_line(_state(balls=False))
    assert "no balls in the bag" in line
    assert "buys" in line


@pytest.mark.parametrize("owned", [0, 1, 9])
def test_the_shortfall_counts_down(owned):
    line = server._catch_line(_state(owned=owned))
    assert f"{10 - owned} more registered opens it" in line


# ---------------------------------------------------------------------------
# The bit the reason is built on, checked against the cartridge
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir():
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")


@needs_rom
def test_owns_species_agrees_with_the_count_the_harness_already_trusts(tmp_path):
    """`read_flags` counts these bits; `owns_species` reads them one at a time.

    Summing the second has to equal the first, on every save, or the bit order
    is wrong — and a wrong bit order here would put "not in the Pokedex yet"
    beside a species the run already owns, which is a confident wrong answer in
    the payload the model reads most.
    """
    pytest.importorskip("pyboy")
    import glob
    import shutil

    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        checked = 0
        for path in sorted(glob.glob(str(SAVES_DIR / "*.state")))[:8]:
            copy = tmp_path / "probe.state"
            shutil.copy(path, copy)
            try:
                emulator.load_state(str(copy))
                emulator.settle()
                reader = RedBlueMemoryReader(emulator)
                counted = reader.read_flags()["pokedex_owned"]
            except Exception:  # noqa: BLE001 — a save that will not load proves nothing
                continue
            one_at_a_time = sum(1 for dex in range(1, 152) if reader.owns_species(dex))
            assert one_at_a_time == counted, f"{path}: {one_at_a_time} != {counted}"
            for mon in reader.read_party():
                if mon.get("pokedex_id"):
                    assert reader.owns_species(mon["pokedex_id"]), (
                        "a Pokemon in the party must read as registered"
                    )
            checked += 1
        assert checked, "no save in the corpus could be loaded"
    finally:
        emulator.close()


@needs_rom
def test_an_unreadable_dex_number_answers_none_rather_than_false(tmp_path):
    """None and False mean different things here, and only False prints a reason."""
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = RedBlueMemoryReader(emulator)
        assert reader.owns_species(None) is None
        assert reader.owns_species(0) is None
        assert reader.owns_species(152) is None
    finally:
        emulator.close()
