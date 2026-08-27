"""The rule that refuses a command already proved to change nothing.

Every number cited here is measured, either from the receipts of the 34-hour run
of 2026-08-25 (``runs/20260825T224823Z-983b/receipts.jsonl``) or from driving the
real ROM; see the module docstring of ``pokemon_agent/repeats.py``. The live
proofs against a save state are in ``tests/test_repeats_live.py``.
"""

import copy

from pokemon_agent.repeats import (
    REPEAT_LIMIT,
    RepeatedNoProgress,
    RepeatGuard,
    Streak,
    action_refusal,
    all_walks_blocked,
    battle_refusal,
    default_refusal,
    looks_like_dialog,
    screen_words,
    sim_refusal,
    world_fingerprint,
)

CERULEAN = ("act", "wait_60", *(["press_a"] * 38))


def state(
    *,
    x=27,
    y=26,
    facing="right",
    hp=95,
    money=9277,
    screen="",
    dialog=False,
    bag=(),
    in_battle=False,
    play_time="37:54:50",
):
    return {
        "metadata": {"timestamp": "2026-08-25T22:48:23+00:00", "frame_count": 1234},
        "player": {
            "position": {"x": x, "y": y},
            "facing": facing,
            "money": money,
            "play_time": play_time,
        },
        "party": [{"species_id": 178, "level": 33, "hp": hp, "max_hp": 95, "status": "OK"}],
        "bag": [{"id": item, "quantity": count} for item, count in bag],
        "battle": {"in_battle": in_battle},
        "map": {"map_id": 3, "map_name": "Cerulean City"},
        "flags": {"badge_count": 2, "pokedex_owned": 2},
        "dialog": {"active": dialog},
        "dialog_active": dialog,
        "screen": screen,
    }


def call(guard, key, before, after=None):
    """One command that the server let through, filed the way the server files it.

    Record only. A refused command never reaches the emulator, so it never
    reaches ``record`` either and the streak stops where the refusal found it.
    """
    after = before if after is None else after
    guard.record(key, world_fingerprint(before), world_fingerprint(after), screen_words(after))


def refused(guard, key, describe=None):
    try:
        guard.check(key, describe=describe)
    except RepeatedNoProgress as exc:
        return exc.detail
    return None


class TestWhenItFires:
    def test_a_second_identical_batch_is_never_touched(self):
        # Two identical batches is how a dialog is advanced, how a corridor is
        # walked, and how a battle is fought. Firing there would break the game.
        guard = RepeatGuard()
        for _ in range(2):
            call(guard, CERULEAN, state())
        assert refused(guard, CERULEAN) is None

    def test_it_stays_quiet_right_up_to_the_limit(self):
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT - 1):
            call(guard, CERULEAN, state())
        assert refused(guard, CERULEAN) is None

    def test_it_refuses_the_call_after_the_limit(self):
        # The measured Cerulean episode sent this batch 302 times for 12,317
        # presses. The 17th is where it stops.
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT):
            call(guard, CERULEAN, state())
        assert refused(guard, CERULEAN) is not None

    def test_the_worst_measured_legitimate_wait_clears_the_limit(self):
        # A full party healing at a Poke Center counter: nine consecutive A
        # presses land on the same frame with the same words while the machine
        # animates one flash per Pokemon. Measured on the ROM; see
        # tests/test_repeats_live.py. The limit has to sit above it.
        guard = RepeatGuard()
        for _ in range(9):
            call(guard, ("act", "press_a"), state(dialog=True, screen="OK. We'll need"))
        assert refused(guard, ("act", "press_a")) is None


class TestWhatCountsAsProgress:
    def test_a_batch_that_moved_the_player_never_counts(self):
        guard = RepeatGuard()
        for step in range(REPEAT_LIMIT * 2):
            before = state(x=27 + step)
            call(guard, ("act", "walk_right"), before, state(x=28 + step))
        assert refused(guard, ("act", "walk_right")) is None

    def test_a_blocked_walk_does(self):
        # A blocked step in Gen 1 returns the same tile rather than an error, so
        # `act up:1` at Vermilion (33,4) reported `moved 0` 362 times running.
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT):
            call(guard, ("act", "walk_up"), state(x=33, y=4))
        assert refused(guard, ("act", "walk_up")) is not None

    def test_words_never_read_before_restart_the_count(self):
        # A real conversation puts something new on screen every press, and a
        # long one runs past the limit without ever repeating itself. Nothing
        # here may be refused however many pages it takes.
        guard = RepeatGuard()
        pages = [f"page {index} of a very long speech" for index in range(REPEAT_LIMIT * 3)]
        for page in pages:
            call(guard, ("act", "press_a"), state(dialog=True, screen=page))
            assert refused(guard, ("act", "press_a")) is None

    def test_words_already_read_do_not(self):
        # Which is the Vermilion Poke Center episode exactly: once the party was
        # healed the agent went round the same conversation 181 more times, and
        # every screen in it was one it had already been shown.
        guard = RepeatGuard()
        lap = ["Welcome to our", "We heal your", "OK. We'll need", "We hope to see"]
        for page in lap:  # the first lap teaches it the words
            call(guard, ("act", "press_a"), state(dialog=True, screen=page))
        assert refused(guard, ("act", "press_a")) is None
        for page in lap * 8:  # every lap after that is a repeat
            call(guard, ("act", "press_a"), state(dialog=True, screen=page))
        assert refused(guard, ("act", "press_a")) is not None

    def test_a_box_that_answers_at_random_still_runs_out_of_new_lines(self):
        # The Cerulean object answers with one of four flavour lines chosen at
        # random, which is why "the same words twice" is the wrong test and
        # "words I have not read before" is the right one: the fourth line is
        # new once and never again.
        guard = RepeatGuard()
        lines = ["ignored orders", "took a snooze", "turned away", "is loafing around"]
        for index in range(REPEAT_LIMIT * 4):
            call(guard, CERULEAN, state(dialog=True, screen=lines[index % 4]))
        assert refused(guard, CERULEAN) is not None

    def test_a_faint_or_a_heal_starts_a_new_streak(self):
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT):
            call(guard, ("act", "press_a"), state(hp=22))
        assert refused(guard, ("act", "press_a")) is not None
        call(guard, ("act", "press_a"), state(hp=22), state(hp=95))
        assert refused(guard, ("act", "press_a")) is None

    def test_an_item_bought_starts_a_new_streak(self):
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT):
            call(guard, ("buy", "poke ball", "1"), state(bag=((4, 1),)))
        assert refused(guard, ("buy", "poke ball", "1")) is not None
        call(
            guard,
            ("buy", "poke ball", "1"),
            state(bag=((4, 1),), money=9277),
            state(bag=((4, 2),), money=9077),
        )
        assert refused(guard, ("buy", "poke ball", "1")) is None


class TestNeverStuck:
    def test_any_other_command_is_still_accepted(self):
        # The one guarantee that matters: a refusal must never leave the agent
        # with no legal move. The block is scoped to the exact command.
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT * 2):
            call(guard, CERULEAN, state())
        assert refused(guard, CERULEAN) is not None
        for escape in (("act", "press_b"), ("act", "walk_left"), ("act", "wait_60"), ("sim",)):
            assert refused(guard, escape) is None

    def test_taking_the_escape_clears_the_block(self):
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT * 2):
            call(guard, CERULEAN, state())
        assert refused(guard, CERULEAN) is not None
        call(guard, ("act", "press_b"), state(dialog=True), state(dialog=False))
        assert refused(guard, CERULEAN) is None

    def test_a_reset_forgets_everything(self):
        guard = RepeatGuard()
        for _ in range(REPEAT_LIMIT * 2):
            call(guard, CERULEAN, state())
        guard.reset()
        assert refused(guard, CERULEAN) is None


class TestTheFingerprint:
    def test_the_clock_is_not_in_it(self):
        # Two fields in the state dict advance with no input at all. A
        # fingerprint carrying either can never compare equal, and a guard that
        # can never compare equal never fires and never says so -- which is the
        # failure mode this test exists to catch.
        earlier = state(play_time="37:54:50")
        later = copy.deepcopy(earlier)
        later["player"]["play_time"] = "38:01:12"
        later["metadata"]["timestamp"] = "2026-08-26T05:12:00+00:00"
        later["metadata"]["frame_count"] = 999999
        assert world_fingerprint(earlier) == world_fingerprint(later)

    def test_everything_a_command_can_win_is_in_it(self):
        base = state()
        for mutate in (
            lambda s: s["player"]["position"].update(x=28),
            lambda s: s["player"].update(facing="left"),
            lambda s: s["player"].update(money=9000),
            lambda s: s["party"][0].update(hp=1),
            lambda s: s["party"][0].update(level=34),
            lambda s: s["bag"].append({"id": 4, "quantity": 1}),
            lambda s: s["flags"].update(badge_count=3),
            lambda s: s["flags"].update(pokedex_owned=3),
            lambda s: s["map"].update(map_id=4),
            lambda s: s["battle"].update(in_battle=True),
        ):
            changed = copy.deepcopy(base)
            mutate(changed)
            assert world_fingerprint(base) != world_fingerprint(changed)

    def test_a_box_opening_and_closing_is_not_in_it(self):
        # Whether a box happens to be open is the transient half of the frame:
        # the Cerulean fixed point flips it twice a second while changing
        # nothing. The words inside it are carried separately.
        assert world_fingerprint(state(dialog=False)) == world_fingerprint(state(dialog=True))

    def test_an_empty_state_does_not_explode(self):
        assert world_fingerprint(None) == world_fingerprint({})
        assert screen_words(None) == ""


class TestTheRefusal:
    def test_it_names_b_because_the_agent_will_not_reach_for_it(self):
        # Run-wide the agent pressed B 321 times in 78,795 presses, and pressed
        # it zero times in the 12,317 it spent at the Cerulean object. A refusal
        # that does not spell out the way back is an advisory.
        streak = Streak(key=CERULEAN, world=(), words={""}, count=REPEAT_LIMIT)
        detail = action_refusal(streak, dialog=True, blocked_walk=False)
        assert "poke act b:2" in detail
        assert "Any other command is accepted" in detail

    def test_each_diagnosis_names_an_escape_that_works_from_that_frame(self):
        # A direction is the way out of a wall and does nothing at all under an
        # open box; offering the wrong one is the confident wrong answer.
        streak = Streak(key=("act", "walk_up"), world=(), words={""}, count=REPEAT_LIMIT)
        blocked = action_refusal(streak, dialog=False, blocked_walk=True)
        toggling = action_refusal(streak, dialog=True, blocked_walk=False)
        waiting = action_refusal(streak, dialog=False, blocked_walk=False)
        assert "blocked" in blocked and "different direction" in blocked
        assert "closes it" in toggling and "poke act b:2" in toggling
        assert "b:2" not in blocked
        assert "wait_60" in waiting

    def test_it_quotes_the_command_back_the_way_it_was_typed(self):
        # Quoting the measured batch token by token is 38 `press_a`s and 300
        # bytes of noise, describing a command the agent never sent.
        streak = Streak(key=CERULEAN, world=(), words={""}, count=REPEAT_LIMIT)
        detail = action_refusal(streak, dialog=True, blocked_walk=False)
        assert "`poke act wait_60 a:38`" in detail
        assert "press_a" not in detail

    def test_it_says_how_many_times(self):
        streak = Streak(key=("act", "walk_up"), world=(), words={""}, count=REPEAT_LIMIT)
        assert "`poke act up`" in action_refusal(streak, dialog=False, blocked_walk=True)
        assert str(REPEAT_LIMIT) in default_refusal(streak)

    def test_the_battle_refusal_names_the_verbs_that_are_not_fleeing(self):
        # Route 6 (1,15): 331 `poke run` calls, 2,405 presses, "could not get
        # away" every time. The escape is a different verb, not a different key.
        streak = Streak(key=("run",), world=(), words={""}, count=REPEAT_LIMIT)
        detail = battle_refusal(streak)
        assert "poke fight" in detail and "poke catch" in detail
        assert "speed roll" in detail

    def test_a_repeated_attack_is_not_told_it_is_fleeing(self):
        streak = Streak(key=("fight", "ember"), world=(), words={""}, count=REPEAT_LIMIT)
        detail = battle_refusal(streak)
        assert "`poke fight ember`" in detail
        assert "speed roll" not in detail
        assert "poke calc" in detail

    def test_the_sim_refusal_says_why_repeating_it_cannot_help(self):
        streak = Streak(key=("sim", "walk_up"), world=(), words={""}, count=REPEAT_LIMIT)
        detail = sim_refusal(streak)
        assert "never touches the game" in detail
        assert "poke act" in detail

    def test_composing_it_costs_nothing_on_the_calls_that_pass(self):
        # `describe` is only ever called on the refusal path.
        guard = RepeatGuard()
        calls = []
        for _ in range(REPEAT_LIMIT):
            guard.check(CERULEAN, describe=lambda streak: calls.append(streak) or "")
            guard.record(CERULEAN, (1,), (1,), "")
        assert calls == []
        refused(guard, CERULEAN, describe=lambda streak: calls.append(streak) or "x")
        assert len(calls) == 1


class TestDiagnosisHelpers:
    def test_a_walk_that_moved_is_not_blocked(self):
        assert not all_walks_blocked(["walk_up"], {"moved": 3})

    def test_a_walk_that_moved_nothing_is(self):
        assert all_walks_blocked(["walk_up"], {"moved": 0})
        assert all_walks_blocked(["walk_up"], {})

    def test_a_batch_with_no_walk_in_it_is_not_a_wall_problem(self):
        assert not all_walks_blocked(["press_a"], {"moved": 0})

    def test_a_box_is_read_from_either_spelling_of_the_flag(self):
        assert looks_like_dialog({"dialog_active": True})
        assert looks_like_dialog({"dialog": {"active": True}})
        assert not looks_like_dialog({})
