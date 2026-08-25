from pokemon_agent.bench.registry import Receipt
from pokemon_agent.interventions import (
    PRIORITY_DANGER,
    PRIORITY_STUCK,
    Circling,
    CommitGate,
    EnteringSegment,
    InterventionPolicy,
    LowHP,
    RepeatedFailure,
    StalledMilestones,
    build_prompt,
    default_detectors,
)


def receipt(seq=0, presses=8, map_name="Route 3", pos=(1, 1), **kw):
    return Receipt(seq=seq, presses=presses, map_name=map_name, pos=pos, **kw)


def walk(count, *, presses=8, start=0, **kw):
    """A batch run that visits a fresh tile each time, so nothing looks stuck."""

    return [receipt(seq=start + i, presses=presses, pos=(i, 0), **kw) for i in range(count)]


class TestStalledMilestones:
    def test_quiet_until_the_press_budget_is_spent(self):
        window = walk(10, presses=8)  # 80 presses
        assert StalledMilestones(presses=800).check(window, {}) is None

    def test_fires_once_nothing_has_been_reached_for_long_enough(self):
        window = walk(120, presses=10)  # 1200 presses, no milestones
        trigger = StalledMilestones(presses=800).check(window, {})
        assert trigger is not None
        assert trigger.priority == PRIORITY_STUCK
        assert trigger.payload["presses_since_milestone"] >= 800

    def test_a_recent_milestone_resets_it(self):
        window = walk(120, presses=10)
        window[-1] = receipt(seq=999, presses=10, milestones_new=("EVENT_BEAT_BROCK",))
        assert StalledMilestones(presses=800).check(window, {}) is None

    def test_an_old_milestone_does_not(self):
        window = [receipt(seq=0, presses=10, milestones_new=("EVENT_BEAT_BROCK",))]
        window += walk(120, presses=10, start=1)
        assert StalledMilestones(presses=800).check(window, {}) is not None


class TestCircling:
    def test_a_straight_line_is_not_circling(self):
        assert Circling(ratio=2.5, min_samples=10).check(walk(40), {}) is None

    def test_pacing_the_same_tiles_is(self):
        window = [receipt(seq=i, pos=(i % 4, 0)) for i in range(60)]
        trigger = Circling(ratio=2.5, min_samples=10).check(window, {})
        assert trigger is not None
        assert trigger.payload["unique"] == 4
        assert trigger.payload["worst"]["count"] == 15

    def test_too_few_samples_to_judge(self):
        window = [receipt(seq=i, pos=(0, 0)) for i in range(5)]
        assert Circling(min_samples=40).check(window, {}) is None

    def test_the_same_tile_on_two_maps_counts_separately(self):
        window = [receipt(seq=i, pos=(0, 0), map_name=f"Route {i}") for i in range(60)]
        assert Circling(ratio=2.5, min_samples=10).check(window, {}) is None


class TestLowHP:
    def test_healthy_party_is_left_alone(self):
        window = [receipt(seq=i, hp=(38, 40)) for i in range(6)]
        assert LowHP().check(window, {}) is None

    def test_sustained_low_hp_fires(self):
        window = [receipt(seq=i, hp=(10, 40)) for i in range(6)]
        trigger = LowHP(fraction=0.35, batches=5).check(window, {})
        assert trigger is not None
        assert trigger.priority == PRIORITY_DANGER
        assert trigger.payload["hp"] == [10, 40]

    def test_one_bad_batch_is_not_enough(self):
        window = [receipt(seq=i, hp=(38, 40)) for i in range(5)]
        window.append(receipt(seq=5, hp=(4, 40)))
        assert LowHP(fraction=0.35, batches=5).check(window, {}) is None

    def test_a_fainted_party_does_not_divide_by_zero(self):
        window = [receipt(seq=i, hp=(0, 0)) for i in range(6)]
        assert LowHP().check(window, {}) is None


class TestRepeatedFailure:
    def test_two_failures_of_the_same_command(self):
        window = [receipt(seq=i, tool="act", exit_code=2) for i in range(2)]
        trigger = RepeatedFailure(times=2).check(window, {})
        assert trigger is not None
        assert trigger.payload["tool"] == "act"

    def test_a_success_in_between_clears_it(self):
        window = [
            receipt(seq=0, tool="act", exit_code=2),
            receipt(seq=1, tool="act", exit_code=0),
        ]
        assert RepeatedFailure(times=2).check(window, {}) is None

    def test_two_different_commands_failing_is_not_a_loop(self):
        window = [
            receipt(seq=0, tool="act", exit_code=2),
            receipt(seq=1, tool="map", exit_code=2),
        ]
        assert RepeatedFailure(times=2).check(window, {}) is None


class TestEnteringSegment:
    def test_first_arrival_fires(self):
        window = walk(3, map_name="Route 3")
        window.append(receipt(seq=9, map_name="Mt Moon 1F"))
        trigger = EnteringSegment(maps=frozenset({"Mt Moon 1F"})).check(window, {})
        assert trigger is not None
        assert trigger.payload["map"] == "Mt Moon 1F"

    def test_it_does_not_fire_again_while_still_there(self):
        window = walk(3, map_name="Mt Moon 1F")
        assert EnteringSegment(maps=frozenset({"Mt Moon 1F"})).check(window, {}) is None

    def test_an_ordinary_map_is_ignored(self):
        window = walk(3, map_name="Route 3")
        assert EnteringSegment(maps=frozenset({"Mt Moon 1F"})).check(window, {}) is None


class TestCommitGate:
    def test_fires_on_a_pending_irreversible_action(self):
        state = {"pending_commit": {"kind": "release", "detail": "Pidgey"}}
        trigger = CommitGate().check([receipt()], state)
        assert trigger is not None
        assert "release" in trigger.reason

    def test_nothing_pending_means_nothing_to_gate(self):
        assert CommitGate().check([receipt()], {}) is None

    def test_an_unlisted_kind_passes_through(self):
        state = {"pending_commit": {"kind": "walk"}}
        assert CommitGate().check([receipt()], state) is None


class TestPolicy:
    def test_it_picks_the_most_urgent_of_several(self):
        window = [receipt(seq=i, pos=(i % 3, 0), hp=(4, 40)) for i in range(60)]
        policy = InterventionPolicy(
            detectors=(Circling(ratio=2.0, min_samples=10), LowHP()),
        )
        trigger = policy.evaluate(window)
        assert trigger is not None
        assert trigger.name == "low_hp"  # danger outranks stuck

    def test_the_cooldown_holds_it_off(self):
        window = [receipt(seq=i, pos=(0, 0), presses=10) for i in range(60)]
        policy = InterventionPolicy(
            detectors=(Circling(ratio=2.0, min_samples=10),),
            cooldown_presses=600,
        )
        first = policy.evaluate(window)
        assert first is not None
        policy.record(first, total_presses=600)
        assert policy.evaluate(window, total_presses=900) is None
        assert policy.evaluate(window, total_presses=1300) is not None

    def test_the_session_budget_is_final(self):
        window = [receipt(seq=i, pos=(0, 0)) for i in range(60)]
        policy = InterventionPolicy(
            detectors=(Circling(ratio=2.0, min_samples=10),),
            cooldown_presses=0,
            max_per_session=2,
        )
        for n in range(2):
            trigger = policy.evaluate(window)
            assert trigger is not None
            policy.record(trigger, total_presses=n)
        assert policy.evaluate(window) is None
        assert policy.remaining() == 0

    def test_a_healthy_run_is_never_interrupted(self):
        window = walk(60, presses=4, hp=(40, 40))
        assert InterventionPolicy().evaluate(window) is None

    def test_default_detectors_all_satisfy_the_protocol(self):
        for detector in default_detectors():
            assert detector.check([], {}) is None


class TestPrompt:
    def test_it_carries_the_reason_the_question_and_the_tail(self):
        window = [
            receipt(seq=1, map_name="Pewter City", pos=(31, 31), blocked_after="up"),
            receipt(seq=2, map_name="Pewter City", pos=(31, 31), moved=0),
        ]
        trigger = LowHP().check([receipt(seq=i, hp=(5, 40)) for i in range(6)], {})
        assert trigger is not None
        prompt = build_prompt(
            trigger,
            state_summary="Pewter City (31,31), Charmander L15 5/40",
            recent=window,
            milestone_summary="12 of 63, furthest: Boulder Badge",
        )
        assert trigger.reason in prompt
        assert trigger.question in prompt
        assert "Charmander L15 5/40" in prompt
        assert "Boulder Badge" in prompt
        assert "blocked after up" in prompt

    def test_it_stays_small_enough_to_be_worth_swapping_for(self):
        window = [receipt(seq=i, pos=(i, 0)) for i in range(500)]
        trigger = StalledMilestones(presses=0).check(window, {})
        assert trigger is not None
        prompt = build_prompt(trigger, state_summary="x", recent=window)
        assert len(prompt) < 4000
