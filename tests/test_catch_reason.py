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


# ---------------------------------------------------------------------------
# The odds themselves, on a frame that cannot answer yet
# ---------------------------------------------------------------------------


def test_a_battle_entry_frame_prints_no_odds_at_all():
    """0 HP out of 36 is the battle starting, not a Pokemon about to faint.

    On a battle-entry frame the enemy's max HP is already written and its
    current HP is still zero. The odds came out "0% now / 100% worn down" — the
    right arithmetic on a number that is not yet true, printed at exactly the
    moment the model decides whether to throw. Four saves in the corpus are in
    that state, all Diglett at 0/36 with a correctly-read catch rate of 255.

    Silence is the honest answer: the frame does not know.
    """
    state = _state()
    state["battle"]["enemy"]["hp"] = 0
    assert server._catch_line(state) is None


def test_a_readable_frame_still_prices_the_ball():
    """The control — without it the test above passes on a broken payload."""
    line = server._catch_line(_state(hp=(20, 30)))
    assert "Poke Ball x16" in line
    assert "%" in line


def test_catch_payload_refuses_a_fainted_enemy():
    from pokemon_agent import capabilities

    battle = {
        "in_battle": True,
        "type": "wild",
        "enemy": {"species": "Diglett", "hp": 0, "max_hp": 36},
    }
    with pytest.raises(capabilities.CapabilityError, match="not readable yet"):
        capabilities.catch_payload(battle, [{"item": "Poke Ball", "quantity": 16}], 255)


@needs_rom
def test_a_pokemon_in_the_party_is_not_always_registered(tmp_path):
    """The obvious invariant is false, and the feature depends on it being false.

    Two saves in this run's corpus hold a Charizard whose Pokedex bit is clear:
    owned reads 2 (Charmander and Charmeleon) with a Charizard standing in slot
    one. Evolution did not register the new species here. The by-species sum
    still equals the count, so the bits are being read correctly — the party is
    simply not the same set as the Pokedex.

    This is why `owns_species` reads the dex and not the party. Oak's aide counts
    the bits, so a Pokemon you are carrying can still be one you are short of,
    and "Charizard is not in the Pokedex yet" is the true statement even with one
    in front of you.
    """
    pytest.importorskip("pyboy")
    import glob
    import shutil

    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        for path in sorted(glob.glob(str(SAVES_DIR / "*.state")))[:8]:
            copy = tmp_path / "probe.state"
            shutil.copy(path, copy)
            try:
                emulator.load_state(str(copy))
                emulator.settle()
                reader = RedBlueMemoryReader(emulator)
                party = reader.read_party()
                counted = reader.read_flags()["pokedex_owned"]
            except Exception:  # noqa: BLE001 — a save that will not load proves nothing
                continue
            carried = [mon["pokedex_id"] for mon in party if mon.get("pokedex_id")]
            if any(not reader.owns_species(dex) for dex in carried):
                # The point of the test: the count is still self-consistent.
                by_species = sum(1 for dex in range(1, 152) if reader.owns_species(dex))
                assert by_species == counted
                return
        pytest.skip("no save in the sampled corpus carries an unregistered party member")
    finally:
        emulator.close()
