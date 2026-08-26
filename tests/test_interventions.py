from pathlib import Path

from pokemon_agent.bench.registry import Receipt
from pokemon_agent.interventions import (
    FACT_BUDGET_CHARS,
    FACTS_HEADER,
    FACTS_INFERRED_HEADER,
    FACTS_KNOWN_HEADER,
    PRIORITY_DANGER,
    PRIORITY_STUCK,
    Circling,
    CommitGate,
    EnteringSegment,
    Fact,
    InterventionPolicy,
    LowHP,
    MapFacts,
    RepeatedFailure,
    StalledMilestones,
    Toothless,
    Trigger,
    build_prompt,
    check_advice,
    default_detectors,
    format_facts,
    harness_facts,
    refusal_note,
    standing_on,
)
from pokemon_agent.world import World


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


class TestHarnessFacts:
    """The Route 4 firing, and the rules that came out of it.

    A thinking session was asked to help a run at 10/65 HP on Route 4 and
    answered "walk west ~18 tiles to Vermilion City's east gate ... then on to
    Celadon via Route 24". Every geographic claim was wrong and it cost 146
    seconds and 31 presses in the wrong direction, at 10 HP. The harness knew
    the answer the whole time: Cerulean City is one hop east and has a Poke
    Center. These tests pin the facts into the prompt so the next thinker
    cannot be asked to remember them.
    """

    def route4_trigger(self):
        window = [
            receipt(seq=i, presses=8, map_name="Route 4", pos=(60 - i, 5), hp=(10, 65))
            for i in range(6)
        ]
        trigger = LowHP().check(window, {})
        assert trigger is not None
        return trigger, window

    def route4_prompt(self, **kwargs):
        trigger, window = self.route4_trigger()
        return build_prompt(
            trigger,
            state_summary="map: Route 4\nx: 60\ny: 5\nhp: 10/65",
            recent=window,
            **kwargs,
        )

    def test_the_route_4_firing_carries_the_route_the_harness_knew(self):
        prompt = self.route4_prompt(goal="Cross Mt. Moon and emerge into Cerulean City.")

        # The real map, not a remembered one: 90 tiles wide, exits east and south.
        assert "Route 4 is 90 tiles wide and 18 tall" in prompt
        assert "east edge (walk_right) -> Cerulean City" in prompt
        # The healing answer, ranked rather than chosen for it.
        assert "Cerulean Pokecenter 2 hops" in prompt
        assert "Mt Moon Pokecenter 1 hop" in prompt
        # The objective's own destination, routed.
        assert "The objective names Cerulean City: 1 hop from Route 4" in prompt

    def test_the_answer_it_gave_is_not_in_the_facts_it_now_gets(self):
        prompt = self.route4_prompt(goal="Cross Mt. Moon and emerge into Cerulean City.")
        for invented in ("Vermilion", "Celadon", "Route 24"):
            assert invented not in prompt

    def test_the_prompt_says_the_facts_outrank_the_model(self):
        prompt = self.route4_prompt()
        assert "authoritative" in prompt
        assert "recollection of Pokemon Red" in prompt

    def test_hop_counts_are_labelled_graph_distance_and_never_tiles(self):
        """A hop is a plan, not a promise: Route 4's halves do not connect."""

        prompt = self.route4_prompt()
        assert FACTS_INFERRED_HEADER in prompt
        head, tail = prompt.split(FACTS_INFERRED_HEADER, 1)
        # Every hop count lives under the caveat, never in the measured half.
        assert "hops" not in head
        assert "Cerulean Pokecenter 2 hops" in tail

    def test_measured_tiles_and_inferred_hops_are_kept_apart(self):
        trigger, window = self.route4_trigger()
        facts = harness_facts(trigger, recent=window)
        assert [fact.known for fact in facts if "tiles wide" in fact.text] == [True]
        assert [fact.known for fact in facts if "Poke Centers from" in fact.text] == [False]

    def test_poke_centers_are_ranked_and_not_picked(self):
        """Nearest by hop count is not always right — Mt Moon's hop re-enters the cave."""

        maps = MapFacts()
        ranked = maps.poke_centers("Route 4")
        assert [name for _, name, _ in ranked] == [
            "Mt Moon Pokecenter",
            "Cerulean Pokecenter",
            "Pewter Pokecenter",
        ]
        assert [distance for distance, _, _ in ranked] == [1, 2, 3]

    def test_the_ranking_is_by_graph_distance_from_the_map_you_are_on(self):
        """Different map, different answer — and ties are broken by name, not by luck."""

        maps = MapFacts()
        ranked = maps.poke_centers("Pallet Town")
        assert [distance for distance, _, _ in ranked] == [3, 3, 5]
        assert {name for _, name, _ in ranked[:2]} == {
            "Cinnabar Pokecenter",
            "Viridian Pokecenter",
        }

    def test_map_dimensions_come_from_the_data_not_from_a_guess(self):
        maps = MapFacts()
        assert maps.dimensions("Route 4") == (90, 18)
        assert maps.dimensions("Not A Map") is None

    def test_every_exit_is_listed_with_the_button_that_takes_it(self):
        trigger, window = self.route4_trigger()
        exits = next(
            f.text for f in harness_facts(trigger, recent=window) if "Every exit" in f.text
        )
        assert "south edge (walk_down) -> Route 3" in exits
        assert "warp (18,5) -> Mt Moon 1F" in exits

    def test_the_goal_destination_is_the_last_map_the_goal_names(self):
        maps = MapFacts()
        goal = "Leave Pewter, cross Route 3 and Mt. Moon, and emerge into Cerulean City."
        assert maps.find_map(goal) == "Cerulean City"
        assert maps.find_map("nothing here names a map") is None

    def test_a_goal_naming_no_map_gives_no_route_line(self):
        prompt = self.route4_prompt(goal="Win the next fight and keep the party alive.")
        assert "The objective names" not in prompt


class TestFactsFromTheLiveFrame:
    """The half of the facts that needs collision rather than the graph."""

    def observation(self, walked=((2, 1), (3, 1))):
        """A 5x2 room with one wall, as the live window reports it.

        The store remembers walking two of its tiles, which is what makes the
        rest of the room unwalked ground rather than a guess.
        """

        terrain = [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
        return {
            "state": {"map": {"map_name": "Route 4"}, "player": {"position": {"x": 2, "y": 1}}},
            "navigation": {
                "snapshot": {
                    "map_name": "Route 4",
                    "map_id": 15,
                    "terrain": terrain,
                    "window_top_left": {"x": 0, "y": 0},
                    "player_position": {"x": 2, "y": 1},
                    "map_dimensions": {"width": 5, "height": 2},
                    "sprites": [],
                }
            },
            "explored": {"width": 5, "height": 2, "walked": set(walked), "walkable": set()},
        }

    def test_unwalked_reachable_ground_is_counted_from_live_collision(self):
        trigger = LowHP().check(
            [receipt(seq=i, map_name="Route 4", pos=(2, 1), hp=(4, 40)) for i in range(6)], {}
        )
        assert trigger is not None
        facts = harness_facts(trigger, recent=[], observation=self.observation())
        line = next(f.text for f in facts if "never walked on" in f.text)
        # Nine walkable tiles, two of them walked, and the wall is not one of them.
        assert "7 tiles" in line
        assert "nearest (1,1)" in line
        assert "7 confirmed by this frame" in line

    def test_ground_that_is_all_walked_says_so_rather_than_going_quiet(self):
        walked = {(x, y) for y in range(2) for x in range(5)}
        facts = harness_facts(
            Trigger(name="stalled", priority=PRIORITY_STUCK, reason="r", question="q"),
            recent=[],
            observation=self.observation(walked=walked),
        )
        assert any("has already been walked" in fact.text for fact in facts)

    def test_circling_carries_the_tiles_and_the_way_off_them(self):
        window = [receipt(seq=i, map_name="Route 4", pos=(2, 1)) for i in range(50)]
        window += [receipt(seq=100 + i, map_name="Route 4", pos=(3, 1)) for i in range(10)]
        trigger = Circling(min_samples=10).check(window, {})
        assert trigger is not None
        facts = harness_facts(trigger, recent=window, observation=self.observation())
        text = "\n".join(fact.text for fact in facts)
        assert "Tiles you keep standing on: (2,1) x50, (3,1) x10" in text
        assert "Unwalked walkable neighbours of (2,1)" in text
        assert "walk_down to (2,0)" not in text  # (2,0) is the wall
        assert "walk_left to (1,1)" in text

    def test_a_broken_observation_costs_the_live_facts_and_nothing_else(self):
        trigger, window = TestHarnessFacts().route4_trigger()
        facts = harness_facts(trigger, recent=window, observation={"navigation": {"snapshot": 7}})
        assert any("Route 4 is 90 tiles wide" in fact.text for fact in facts)
        assert not any("never walked on" in fact.text for fact in facts)


class TestRepeatedFailureFacts:
    def test_the_error_text_itself_is_carried(self):
        window = [
            receipt(
                seq=i,
                map_name="Route 4",
                tool="poke act",
                exit_code=1,
                extra={"error": "400: walk_left is not a known action", "actions": ["walk_left"]},
            )
            for i in range(2)
        ]
        trigger = RepeatedFailure().check(window, {})
        assert trigger is not None
        prompt = build_prompt(trigger, state_summary="x", recent=window)
        assert "400: walk_left is not a known action" in prompt
        assert "actions sent: walk_left" in prompt

    def test_a_failure_with_no_recorded_error_adds_no_line(self):
        window = [
            receipt(seq=i, map_name="Route 4", tool="poke act", exit_code=1) for i in range(2)
        ]
        trigger = RepeatedFailure().check(window, {})
        assert trigger is not None
        prompt = build_prompt(trigger, state_summary="x", recent=window)
        assert "failed with" not in prompt


class TestFactsDegradeToOmission:
    """A fact that cannot be computed is left out. A wrong one is what cost the run."""

    def empty_maps(self):
        return MapFacts(World.load(Path("/nonexistent/world.json")))

    def test_missing_map_data_leaves_the_prompt_standing(self):
        trigger, window = TestHarnessFacts().route4_trigger()
        maps = self.empty_maps()
        assert maps.available is False
        assert harness_facts(trigger, recent=window, maps=maps) == []

        prompt = build_prompt(trigger, state_summary="map: Route 4", recent=window, maps=maps)
        assert FACTS_HEADER not in prompt
        assert trigger.question in prompt
        assert "map: Route 4" in prompt

    def test_an_unknown_map_name_is_simply_not_described(self):
        window = [receipt(seq=i, map_name="Nowhere", pos=(1, 1), hp=(3, 30)) for i in range(6)]
        trigger = LowHP().check(window, {})
        assert trigger is not None
        assert harness_facts(trigger, recent=window) == []

    def test_no_receipts_and_no_observation_is_not_an_error(self):
        trigger = Trigger(name="stalled", priority=PRIORITY_STUCK, reason="r", question="q")
        assert harness_facts(trigger) == []
        assert format_facts([]) == ""


class TestFactBudget:
    def test_the_block_never_outgrows_its_budget(self):
        facts = [Fact("x" * 120, known=bool(i % 2)) for i in range(40)]
        for budget in (0, 200, 500, 900, FACT_BUDGET_CHARS):
            assert len(format_facts(facts, budget=budget)) <= budget

    def test_facts_are_dropped_from_the_end_not_the_front(self):
        facts = [Fact("first fact"), Fact("second fact"), Fact("third fact")]
        rendered = format_facts(facts, budget=len(FACTS_HEADER) + len(FACTS_KNOWN_HEADER) + 40)
        assert "first fact" in rendered
        assert "third fact" not in rendered

    def test_the_route_4_prompt_stays_a_short_problem(self):
        prompt = TestHarnessFacts().route4_prompt(goal="Reach Cerulean City.")
        assert len(prompt) < 3000


# ---------------------------------------------------------------------------
# Toothless: the party cannot deal damage
#
# Measured over one run: 3,285 of 13,601 presses (24%) with the lead holding no
# damaging move, Ember and Rage both at 0 PP, party of one. Nothing noticed,
# because every existing check was about HP. A Pokemon at full health with no
# attack cannot win a battle and cannot flee a trainer.
# ---------------------------------------------------------------------------


def _party(*moves, species="Charmeleon"):
    return {"party": [{"species": species, "moves": list(moves)}]}


def _move(name, power, pp):
    return {"name": name, "power": power, "pp": pp}


def _fighting(n=6):
    return [receipt(seq=i, tool="battle") for i in range(n)]


def test_toothless_fires_when_every_move_is_out_of_pp():
    state = _party(_move("Ember", 40, 0), _move("Rage", 20, 0))
    trigger = Toothless().check(_fighting(), state)
    assert trigger is not None
    assert trigger.priority == PRIORITY_DANGER
    assert "Ember" in trigger.reason and "Rage" in trigger.reason


def test_one_usable_attack_is_enough_to_stay_quiet():
    state = _party(_move("Ember", 40, 0), _move("Scratch", 40, 12))
    assert Toothless().check(_fighting(), state) is None


def test_status_moves_with_pp_do_not_count_as_teeth():
    # Growl has PP and no power. It cannot end a battle.
    state = _party(_move("Ember", 40, 0), _move("Growl", 0, 30))
    assert Toothless().check(_fighting(), state) is not None


def test_it_stays_quiet_when_the_party_is_not_in_harms_way():
    state = _party(_move("Ember", 40, 0))
    assert Toothless().check([], state) is None


def test_a_party_shape_without_pp_is_not_judged():
    # Older payloads carry move names only. Silence beats a guess.
    state = {"party": [{"species": "Charmeleon", "moves": ["Ember", "Rage"]}]}
    assert Toothless().check(_fighting(), state) is None


def test_no_party_no_trigger():
    assert Toothless().check(_fighting(), {}) is None
    assert Toothless().check(_fighting(), {"party": []}) is None


def test_toothless_is_registered_and_outranks_being_lost():
    names = {d.name for d in default_detectors()}
    assert "toothless" in names


# ---------------------------------------------------------------------------
# A standing condition must not mask an episodic one
#
# Measured over one run: all 13 interventions fired on `low_hp`, because HP sat
# at 15-30% for 59% of the run and low_hp outranks circling -- which was itself
# firing in 53 of the windows it lost. The detector aimed at the actual failure
# never got a turn, and 0 milestones came of the 13 that did.
# ---------------------------------------------------------------------------


def _hurt_and_circling(n=60):
    return [receipt(seq=i, pos=(i % 3, 0), hp=(9, 65), presses=8) for i in range(n)]


def test_a_second_look_at_the_same_condition_yields_to_a_fresh_one():
    window = _hurt_and_circling()
    policy = InterventionPolicy(
        detectors=(LowHP(), Circling(ratio=2.0, min_samples=10)),
        cooldown_presses=0,
    )

    first = policy.evaluate(window)
    assert first.name == "low_hp", "danger should win the first time"
    policy.record(first, total_presses=100)

    second = policy.evaluate(window)
    assert second.name == "circling", "the answered condition must stop masking"


def test_an_answered_detector_wins_again_once_its_condition_changed():
    window = _hurt_and_circling()
    policy = InterventionPolicy(
        detectors=(LowHP(), Circling(ratio=2.0, min_samples=10)),
        cooldown_presses=0,
    )
    policy.record(policy.evaluate(window), total_presses=100)
    assert policy.evaluate(window).name == "circling"

    policy.clear_answered("low_hp")
    assert policy.evaluate(window).name == "low_hp"


def test_when_everything_has_been_answered_the_most_urgent_still_wins():
    # Silence would be worse: the run is still in trouble.
    window = _hurt_and_circling()
    policy = InterventionPolicy(
        detectors=(LowHP(), Circling(ratio=2.0, min_samples=10)),
        cooldown_presses=0,
    )
    policy.answered.update({"low_hp", "circling"})
    assert policy.evaluate(window).name == "low_hp"


def test_stalled_fires_inside_a_window_the_policy_actually_passes_it():
    # The threshold is in presses and the window is in receipts. At 800 presses
    # against a median 411-press window, this detector fired in 2% of the windows
    # it was asked about, which is off, not strict.
    policy = InterventionPolicy()
    window = [receipt(seq=i, presses=4, pos=(i, 0)) for i in range(policy.window)]
    spent = sum(r.presses for r in window)
    assert spent >= StalledMilestones().presses, (
        f"a full {policy.window}-receipt window carries {spent} presses, "
        f"under the {StalledMilestones().presses}-press threshold"
    )
    assert StalledMilestones().check(window, {}) is not None


# ---------------------------------------------------------------------------
# Checking the answer, not just grounding the question
#
# Every message quoted below is verbatim from interventions.jsonl of run
# 20260825T224823Z-983b, the only archive of delivered interventions this
# project has. Thirteen were delivered before `harness_facts` existed and
# twelve of those named somewhere unreachable the way they said; twenty-two
# were delivered after it, and none did. So the wrong ones are the fixtures for
# what has to be refused and the right ones are the fixtures for what must not.
# ---------------------------------------------------------------------------


class TestCheckAdviceRefusesWhatTheMapContradicts:
    def test_a_warp_the_map_agrees_with_is_left_alone(self):
        assert check_advice("Step on (11,5) -> Mt Moon Pokecenter.", here="Route 4") == ()

    def test_a_warp_that_goes_somewhere_else_is_caught(self):
        claims = check_advice("Step on (18,5) -> Mt Moon Pokecenter.", here="Route 4")
        assert [claim.kind for claim in claims] == ["warp"]
        assert claims[0].truth == "Route 4 (18,5) goes to Mt Moon 1F"

    def test_a_map_named_in_shorthand_still_holds_its_own_warps(self):
        """Intervention 34: "on B1F, go to warp (27,3) -> Route 4", from B2F.

        True of Mt Moon B1F, which the message called "B1F". Refusing it for the
        abbreviation would be catching a typo, so a warp is checked against the
        maps on the route — the ones named, and the ones one hop off them.
        """

        assert check_advice("On B1F, go to warp (27,3) → Route 4.", here="Mt Moon B2F") == ()

    def test_a_warp_on_a_map_further_along_the_route_still_counts(self):
        """Advice is a plan, not a sentence about one map.

        Intervention 19 was written from inside Mt Moon 1F and said "on Route 4,
        go to the warp at (11,5), which leads to the Mt Moon Pokecenter". That is
        true of Route 4, and refusing it would be a bug in the checker rather
        than a catch: a coordinate is checked against every map the message
        names, not only against the one the player happens to be standing on.
        """

        text = (
            "Walk south 9 tiles to the exit at (14,35) — it warps to Route 4. "
            "On Route 4, go to the warp at (11,5), which leads to the Mt Moon Pokecenter."
        )
        assert check_advice(text, here="Mt Moon 1F") == ()

    def test_a_coordinate_off_every_named_map_is_caught(self):
        claims = check_advice("Head for (99,99) and press A.", here="Mt Moon 1F")
        assert [claim.kind for claim in claims] == ["bounds"]
        assert "Mt Moon 1F is 40x36" in claims[0].truth

    def test_a_coordinate_that_fits_any_named_map_is_left_alone(self):
        # (60,4) is off Mt Moon 1F and well inside Route 3, which the text names.
        assert check_advice("On Route 3, walk to (60,4).", here="Mt Moon 1F") == ()

    def test_the_wrong_edge_is_caught(self):
        claims = check_advice("Walk west to Cerulean City.", here="Route 4")
        assert [claim.kind for claim in claims] == ["edge"]
        assert "off the east edge of Route 4" in claims[0].truth

    def test_the_right_edge_is_left_alone(self):
        assert check_advice("Walk east to Cerulean City.", here="Route 4") == ()
        assert check_advice("Press down to Route 3.", here="Route 4") == ()

    def test_a_compass_word_not_aimed_at_a_map_is_not_a_claim(self):
        """The regression that killed the first version of this rule.

        Taking every compass word with a map name inside sixty characters of it
        read intervention 23 — "only 'down' and 'left' are available, so the
        east exit to Cerulean City is impossible right now" — as a claim that
        Cerulean is west of Route 4, and refused thirteen of the twenty-two
        messages that were right. A direction binds to a destination only where
        it is written as one.
        """

        text = (
            "Only 'down' and 'left' are available, so the east exit to "
            "Cerulean City (70 tiles east) is impossible right now."
        )
        assert check_advice(text, here="Route 4") == ()

    def test_a_destination_reached_by_warp_makes_no_compass_claim(self):
        """Mt Moon 1F leaves to Route 4 by warp, so no direction can be wrong."""

        assert check_advice("Head down to Route 4.", here="Mt Moon 1F") == ()

    def test_somewhere_too_far_to_walk_to_in_one_message_is_caught(self):
        """Intervention 9, written from inside Mt Moon 1F with the party at 10 HP.

        Pallet Town is seven warps away and has no Pokemon Center in it. The
        player followed the advice.
        """

        text = "You emerge on Route 1. Turn WEST and walk about 45 tiles to Pallet Town."
        claims = check_advice(text, here="Mt Moon 1F")
        assert {claim.kind for claim in claims} == {"exit", "distance"}
        assert "7 hops" in next(c.truth for c in claims if c.said == "Pallet Town")

    def test_the_hop_ceiling_is_the_callers_to_lift(self):
        text = "Head for Pallet Town."
        assert check_advice(text, here="Mt Moon 1F") != ()
        assert check_advice(text, here="Mt Moon 1F", max_hops=99) == ()

    def test_a_name_that_means_several_maps_is_not_a_checkable_claim(self):
        """ "Mt. Moon" is four maps. Refusing shorthand is not catching an error."""

        assert check_advice("Step on (18,5) -> Mt. Moon.", here="Route 4") == ()

    def test_spelling_is_normalised_but_ambiguity_is_not(self):
        maps = MapFacts()
        assert maps.resolve("Mt. Moon Poke Center") == "Mt Moon Pokecenter"
        assert maps.resolve("Mt Moon") is None
        assert maps.resolve("nowhere at all") is None

    def test_every_map_a_message_names_is_found_not_just_the_last(self):
        maps = MapFacts()
        text = "Leave Mt Moon 1F for Route 4, then east into Cerulean City."
        assert maps.names_in(text) == ("Mt Moon 1F", "Route 4", "Cerulean City")
        # find_map answers "what is this about", which is the other question.
        assert maps.find_map(text) == "Cerulean City"

    def test_a_name_inside_a_longer_one_is_not_a_match(self):
        assert "Route 2" not in MapFacts().names_in("Walk east along Route 24.")


class TestCheckAdviceOnTheArchive:
    """The messages themselves, refused or delivered as the map data decides."""

    def test_the_message_that_invented_a_door_out_of_mt_moon(self):
        """Intervention 2. Mt Moon 1F's only exits are two warps to Route 4."""

        text = (
            "From (25,10) press down 6 times to reach (25,16). From (25,16), follow it "
            "toward the south side of Mt Moon 1F; the door to Route 2 is there. Once "
            "outside on Route 2, face west and walk to Viridian City."
        )
        claims = check_advice(text, here="Mt Moon 1F")
        assert {claim.kind for claim in claims} == {"exit", "distance"}
        assert {"Route 2", "Viridian City"} <= {claim.said for claim in claims}
        assert any("nowhere else" in claim.truth for claim in claims)

    def test_the_message_that_sent_the_run_to_pallet_town_from_two_floors_down(self):
        """Intervention 13. Pallet Town is eight warps from Mt Moon B2F."""

        text = (
            "Walk north the full length of Route 2 (~48 tiles), then continue north "
            "onto Route 1 (~20 tiles) into Pallet Town."
        )
        assert {claim.said for claim in check_advice(text, here="Mt Moon B2F")} == {
            "Route 2",
            "Route 1",
            "Pallet Town",
        }

    def test_the_grounded_message_that_named_three_real_warps(self):
        """Intervention 29, and the shape every message after the facts landed had.

        The list is verbatim, and it is why the warp pattern refuses to read
        across a "(" or a ":": three tiles sharing one destination phrase is not
        a claim about the first of them.
        """

        text = (
            "Do not step on (11,5), (18,5), (24,5): they warp into Mt Moon/Pokecenter. "
            "Keep pressing right until you leave the east edge; that exits to "
            "Cerulean City, about 70 tiles from x=19."
        )
        assert check_advice(text, here="Route 4") == ()

    def test_the_grounded_message_that_routed_out_of_b1f(self):
        """Intervention 35: (27,3) really is Mt Moon B1F's door to Route 4."""

        text = (
            "From (24,22): right 3 tiles to (27,22), up 19 tiles to (27,3). That tile "
            "warps to Route 4. On Route 4: go east to the map edge to Cerulean City."
        )
        assert check_advice(text, here="Mt Moon B1F") == ()


class TestCheckAdviceDegradesToDisprovingNothing:
    """Same rule as the facts: what cannot be computed is left out, never guessed."""

    def test_no_map_data_disproves_nothing(self):
        maps = MapFacts(World.load(Path("/nonexistent/world.json")))
        assert check_advice("Walk west to Cerulean City.", here="Route 4", maps=maps) == ()

    def test_a_map_the_harness_cannot_name_disproves_nothing(self):
        assert check_advice("Walk west to Cerulean City.", here="") == ()
        assert check_advice("", here="Route 4") == ()

    def test_an_unreachable_map_is_a_claim_no_ceiling_forgives(self):
        # The Colosseum is link-cable-only: no warp or edge in the game reaches
        # it, so no hop ceiling makes naming it a walk the player can take.
        claims = check_advice("Go to the Colosseum.", here="Route 4", max_hops=99)
        assert [claim.kind for claim in claims] == ["distance"]
        assert "cannot reach it" in claims[0].truth


class TestStandingOn:
    def test_the_live_frame_wins_over_the_receipts(self):
        observation = {"state": {"map": {"map_name": "Route 4"}}}
        assert standing_on(observation, [receipt(seq=0, map_name="Route 3")]) == "Route 4"

    def test_the_receipts_answer_when_there_is_no_frame(self):
        assert standing_on(None, [receipt(seq=0, map_name="Route 3")]) == "Route 3"

    def test_neither_source_knowing_is_not_an_error(self):
        assert standing_on(None, []) == ""


def test_a_refusal_note_names_what_the_map_says_instead():
    note = refusal_note(check_advice("Walk west to Cerulean City.", here="Route 4"))
    assert note.startswith("map data contradicts: ")
    assert "east edge of Route 4" in note
