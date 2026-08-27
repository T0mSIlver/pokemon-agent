"""The repeat guard proved against the real ROM, not against a model of it.

The rule looks obvious written down and is not: at the Cerulean object the words
in the box are picked at random from four lines, so "the same screen twice" is
the wrong test there; at a Poke Center counter a *legitimate* heal holds the
same screen for nine calls running, so "no new screen once" is the wrong test
everywhere. Both numbers below come from pressing the buttons and reading
``wTileMap`` back.

Skipped entirely when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_agent.repeats import (
    REPEAT_LIMIT,
    RepeatedNoProgress,
    RepeatGuard,
    action_refusal,
    screen_words,
    world_fingerprint,
)
from pokemon_agent.state.builder import build_game_state

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

#: Cerulean City (27,26) facing right: the object the run spent 12,317 presses
#: and 14.6 minutes on, sending `act wait_60 a:38` 302 times. A is the only
#: button it ever pressed there. A named save rather than an autosave, because
#: autosaves rotate out of saves/ and these tests would then skip in silence.
#: Rebuilt by walking to (27,26) on any Cerulean City save and facing right;
#: (27,26) is not reachable in a straight line from the north gate, so it takes
#: a search rather than a run of `walk_right`.
CERULEAN_STATE = "cerulean_object_fixed_point.state"

#: Mt. Moon Poke Center, four steps south of the counter, lead at 10/65 HP —
#: a heal that has not happened yet, so the conversation really does advance.
POKECENTER_STATE = "pre_pokecenter_exit.state"

#: The measured batch, as the server expands it.
CERULEAN_BATCH = ("act", "wait_60", *(["press_a"] * 38))


def _emulator(save_name: str):
    pyboy = pytest.importorskip("pyboy")  # noqa: F841
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    save = SAVES_DIR / save_name
    if not save.exists():
        pytest.skip(f"no {save_name} in {SAVES_DIR}")
    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    emulator.load_state(str(save))
    emulator.tick(6)
    return emulator, RedBlueMemoryReader(emulator)


def _state(reader) -> dict:
    """What the server's `_get_state_dict` hands the guard, minus the trimmings."""
    state = build_game_state(reader)
    state["screen"] = reader.read_screen_text()
    return state


def _one_call(guard, emulator, reader, key, buttons):
    """Send one command the way the server does: check, run, record."""
    guard.check(key)
    before = _state(reader)
    for button in buttons:
        if button == "wait_60":
            emulator.tick(60)
        else:
            emulator.press_and_settle(button, 8)
    after = _state(reader)
    guard.record(key, world_fingerprint(before), world_fingerprint(after), screen_words(after))
    return after


def _drive(guard, emulator, reader, key, buttons, limit=200):
    """Repeat one command until it is refused, or give up. Returns the tally."""
    calls = 0
    presses = 0
    for _ in range(limit):
        try:
            _one_call(guard, emulator, reader, key, buttons)
        except RepeatedNoProgress as exc:
            return calls, presses, exc.detail
        calls += 1
        presses += sum(1 for button in buttons if button != "wait_60")
    return calls, presses, None


@needs_rom
def test_the_words_in_the_box_are_readable_at_all():
    """Everything below rests on this: the harness can see what it is pressing A at.

    ``screen_text`` in every payload the run ever sent was the fixed placeholder
    "Dialog box visible (waiting for input)." — 36,300 bytes across 660 payloads
    saying nothing. ``read_screen_text`` decodes the tile map through the Gen 1
    font table instead, and this is that decode against the object the run got
    stuck on.
    """
    emulator, reader = _emulator(CERULEAN_STATE)
    emulator.press_and_settle("right", 16)
    emulator.press_and_settle("a", 8)

    assert "SLOWBRO" in reader.read_screen_text()


@needs_rom
def test_the_cerulean_object_loop_ends_instead_of_running_forever():
    """The measured episode, replayed. 302 calls and 12,317 presses become a refusal."""
    emulator, reader = _emulator(CERULEAN_STATE)
    guard = RepeatGuard()
    emulator.press_and_settle("right", 16)  # face the object, as the run was

    calls, presses, detail = _drive(
        guard, emulator, reader, CERULEAN_BATCH, ["wait_60"] + ["a"] * 38
    )

    assert detail is not None, "the loop was never refused"
    # Measured: it stops at call 29 for 1,102 presses, against the 302 calls and
    # 12,317 presses the run actually spent — 91% of them saved. The bound is
    # loose because the object answers with one of four lines at random and a
    # line it has not shown yet counts as progress, so the exact call it stops
    # on moves with the RNG.
    assert calls < 60, f"took {calls} calls to notice"
    assert presses < 2400, f"spent {presses} presses"


@needs_rom
def test_the_refusal_names_a_button_that_actually_closes_the_box():
    """`b:2` is the escape, and it has to work from the frame the refusal came from.

    The run pressed B 321 times in 78,795 presses run-wide and zero times in the
    12,317 it spent here, which is the whole reason the way out is spelled out.
    Two presses, because the first only finishes printing the line.
    """
    emulator, reader = _emulator(CERULEAN_STATE)
    guard = RepeatGuard()
    emulator.press_and_settle("right", 16)
    _drive(guard, emulator, reader, CERULEAN_BATCH, ["wait_60"] + ["a"] * 38)
    assert reader.read_dialog()["active"], "the loop should have left a box open"

    detail = action_refusal(guard.streak, dialog=True, blocked_walk=False)
    emulator.press_and_settle("b", 8)
    still_open = reader.read_dialog()["active"]
    emulator.press_and_settle("b", 8)

    assert "poke act b:2" in detail
    assert still_open, "one B finishes the line rather than closing the box"
    assert not reader.read_dialog()["active"]


@needs_rom
def test_the_refusal_does_not_offer_an_escape_that_does_nothing():
    """A direction under an open box is swallowed, so it must not be the way out.

    Measured from the stuck frame: five `left` presses and five `start` presses
    leave the box open, the player on the same tile and the same words on
    screen. Naming either would be a confident wrong answer.
    """
    emulator, reader = _emulator(CERULEAN_STATE)
    guard = RepeatGuard()
    emulator.press_and_settle("right", 16)
    _drive(guard, emulator, reader, CERULEAN_BATCH, ["wait_60"] + ["a"] * 38)
    before = reader.read_coordinates()

    for _ in range(5):
        emulator.press_and_settle("left", 16)

    assert reader.read_dialog()["active"]
    assert reader.read_coordinates() == before
    detail = action_refusal(guard.streak, dialog=True, blocked_walk=False)
    assert "will not do it" in detail


@needs_rom
def test_a_pokecenter_heal_is_never_refused_while_it_is_happening():
    """The negative case, and the one that sets the limit.

    Twenty-five identical `act a:1` calls at a nurse's counter: greeting,
    HEAL/CANCEL, the machine animating, "fighting fit", goodbye. Every screen in
    it is new, except the run of frames the animation holds — and that run is
    the longest legitimate no-progress streak measured anywhere in the game.
    """
    emulator, reader = _emulator(POKECENTER_STATE)
    guard = RepeatGuard()
    for _ in range(6):  # walk up to the counter
        emulator.press_and_settle("up", 16)
    assert reader.read_party()[0]["hp"] == 10

    key = ("act", "press_a")
    longest = 0
    for _ in range(25):
        _one_call(guard, emulator, reader, key, ["a"])
        longest = max(longest, guard.count_for(key))

    assert reader.read_party()[0]["hp"] == 65, "the heal should have gone through"
    assert longest < REPEAT_LIMIT, f"a real heal reached a streak of {longest}"


@needs_rom
def test_the_decoded_words_are_what_keep_a_real_heal_out_of_the_streak():
    """Why the words are in the rule at all, measured both ways on the same heal.

    The durable world does not move for the whole first half of a heal — the
    greeting, the HEAL/CANCEL prompt, the machine animating — so a rule built on
    the world alone counts twelve consecutive "nothing changed" calls through a
    conversation that is plainly advancing. Reading the box instead pulls the
    same twenty calls down to a streak of five, because eleven of them put words
    on screen the agent had not been shown.

    The two numbers are what set the limit at 16: above the one, below anything
    a fixed point can reach.
    """
    emulator, reader = _emulator(POKECENTER_STATE)
    for _ in range(6):
        emulator.press_and_settle("up", 16)

    blind = RepeatGuard()
    reading = RepeatGuard()
    key = ("act", "press_a")
    worst_blind = worst_reading = 0
    for _ in range(20):
        before = _state(reader)
        emulator.press_and_settle("a", 8)
        after = _state(reader)
        fingerprints = (world_fingerprint(before), world_fingerprint(after))
        blind.record(key, *fingerprints, "")  # the same rule with the box unread
        reading.record(key, *fingerprints, screen_words(after))
        worst_blind = max(worst_blind, blind.count_for(key))
        worst_reading = max(worst_reading, reading.count_for(key))

    assert worst_blind > worst_reading * 2
    assert worst_blind >= 12, f"the world alone only reached {worst_blind}"
    assert worst_reading <= 8, f"reading the box still reached {worst_reading}"


@needs_rom
def test_a_pokecenter_counter_is_refused_once_the_heal_has_already_happened():
    """The fourth measured episode: 384 presses and 181 calls A-spamming a nurse.

    The first lap through the conversation is all new words and is left alone.
    Every lap after it is the same four screens the agent has already read, with
    a party that is already at full HP.
    """
    emulator, reader = _emulator(POKECENTER_STATE)
    guard = RepeatGuard()
    for _ in range(6):
        emulator.press_and_settle("up", 16)

    calls, presses, detail = _drive(guard, emulator, reader, ("act", "press_a"), ["a"], limit=200)

    assert detail is not None, "the counter loop was never refused"
    assert reader.read_party()[0]["hp"] == 65, "it should have healed on the way"
    assert calls < 90, f"took {calls} calls to notice, against the run's 181"


@needs_rom
def test_the_clock_running_is_not_mistaken_for_the_game_moving():
    """Ten seconds of game time with no input must leave the fingerprint alone.

    ``player.play_time`` and ``metadata.timestamp`` both advance on their own. A
    fingerprint carrying either never compares equal, so the guard would never
    fire and would never say why.
    """
    emulator, reader = _emulator(CERULEAN_STATE)
    before = _state(reader)
    emulator.tick(600)
    after = _state(reader)

    assert after["player"]["play_time"] != before["player"]["play_time"]
    assert world_fingerprint(before) == world_fingerprint(after)


@needs_rom
def test_a_step_the_player_actually_takes_moves_the_fingerprint():
    emulator, reader = _emulator(CERULEAN_STATE)
    before = _state(reader)
    emulator.press_and_settle("left", 16)
    after = _state(reader)

    assert after["player"]["position"] != before["player"]["position"]
    assert world_fingerprint(before) != world_fingerprint(after)
