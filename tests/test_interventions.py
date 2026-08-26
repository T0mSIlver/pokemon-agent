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
    Trigger,
    build_prompt,
    default_detectors,
    format_facts,
    harness_facts,
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
