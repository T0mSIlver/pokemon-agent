"""The detector and the skill text that stop a run walking the map by hand.

Two models ran the same leg -- Route 1 to the Old Amber in the Pewter Museum --
off the same save through the same harness, and called ``goto`` almost exactly
as often, 352 and 344 times. The one that also sent 1,913 raw ``act`` batches on
top of them spent 18,326 presses against 2,278, bumped walls 3,004 times against
235, and fought 254 wild battles against 17.

What it was reading when it did that is the ``run`` line, which the server
counts inside the 10x9 live collision window only and therefore never prints
above 5. In that leg 1,124 of the 1,459 one-direction ``act`` calls sent exactly
the number the previous answer had printed. The skill text called ``run`` "how
many tiles each direction goes before something stops you" and offered
``left:7`` as an example, which that field cannot produce.

So there are two things to hold: the text has to say what the number is, and the
harness has to be able to see the shape when it happens again.
"""

from __future__ import annotations

import json
from pathlib import Path

from pokemon_agent.bench.registry import Receipt
from pokemon_agent.interventions import (
    LIVE_WINDOW_RUNWAY,
    PRIORITY_STUCK,
    Circling,
    HandWalking,
    InterventionPolicy,
    default_detectors,
)

FIXTURES = Path(__file__).parent / "fixtures"
SKILL = Path(__file__).resolve().parents[1] / "skill" / "SKILL.md"


def batches(count, *, tiles, tool="action", start=0, map_name="Route 3"):
    """``count`` receipts that each moved ``tiles`` tiles onto fresh ground."""

    return [
        Receipt(
            seq=start + i,
            presses=tiles,
            map_name=map_name,
            pos=(i, 0),
            moved=tiles,
            tool=tool,
        )
        for i in range(count)
    ]


def replay(name):
    """A real 120-receipt window off one of the two runs."""

    path = FIXTURES / name
    return [Receipt.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


class TestHandWalking:
    def test_fires_on_a_window_of_screen_length_batches_with_no_goto(self):
        trigger = HandWalking().check(batches(25, tiles=4), {})
        assert trigger is not None
        assert trigger.priority == PRIORITY_STUCK
        assert trigger.payload["batches"] == 25
        assert trigger.payload["tiles_per_batch"] == 4.0
        assert "goto" in trigger.question

    def test_one_goto_in_the_window_is_enough_to_stay_quiet(self):
        window = batches(25, tiles=4)
        window[7] = Receipt(seq=7, presses=30, map_name="Route 3", moved=30, tool="goto")
        assert HandWalking().check(window, {}) is None

    def test_long_batches_are_not_hand_walking(self):
        # Crossing a room already mapped, eight tiles at a time. Nothing here is
        # wasteful and the detector must not claim it is.
        assert HandWalking().check(batches(40, tiles=8), {}) is None

    def test_a_room_crossing_is_too_short_to_judge(self):
        assert HandWalking().check(batches(19, tiles=3), {}) is None

    def test_batches_that_moved_nothing_do_not_count_as_walking(self):
        window = [
            Receipt(seq=i, presses=4, map_name="Pewter Museum 1F", moved=0, tool="action")
            for i in range(60)
        ]
        assert HandWalking().check(window, {}) is None

    def test_the_ceiling_is_the_window_the_player_can_see(self):
        # 5 is what `run` can print at most, because the live collision window is
        # 10 wide and 9 tall with the player at the centre. A batch longer than
        # that saw ground the last answer did not show, so it is not this shape.
        assert LIVE_WINDOW_RUNWAY == 5
        assert HandWalking().check(batches(30, tiles=LIVE_WINDOW_RUNWAY), {}) is not None
        assert HandWalking().check(batches(30, tiles=LIVE_WINDOW_RUNWAY + 1), {}) is None

    def test_it_is_registered_and_yields_to_circling_on_a_tie(self):
        detectors = default_detectors()
        assert "hand_walking" in {d.name for d in detectors}
        names = [d.name for d in detectors]
        assert names.index("circling") < names.index("hand_walking"), (
            "evaluate takes the first maximum, so circling must come first to win a tie"
        )

    def test_the_policy_delivers_it(self):
        policy = InterventionPolicy(detectors=(Circling(), HandWalking()), cooldown_presses=0)
        trigger = policy.evaluate(batches(30, tiles=4))
        assert trigger is not None and trigger.name == "hand_walking"


class TestAgainstTheRunsItWasMeasuredOn:
    """Proof it fires, on the receipts that motivated it, and stays quiet on the
    receipts of the run that did the same journey with the pathfinder."""

    def test_it_fires_on_the_hand_walked_window(self):
        trigger = HandWalking().check(replay("hand_walked_window.jsonl"), {})
        assert trigger is not None
        assert trigger.payload["batches"] >= 100
        assert trigger.payload["tiles_per_batch"] <= LIVE_WINDOW_RUNWAY
        assert trigger.payload["map"] == "Mt Moon 1F"

    def test_it_stays_quiet_on_the_pathfound_window(self):
        window = replay("pathfound_window.jsonl")
        walking = [r for r in window if r.tool == "action" and (r.moved or 0) > 0]
        assert len(walking) >= HandWalking().min_batches, (
            "the negative case has to be a window with real walking in it"
        )
        assert HandWalking().check(window, {}) is None


class TestWhatTheSkillSaysAboutRun:
    skill = SKILL.read_text()

    def test_run_is_described_as_the_visible_window_not_the_map(self):
        skill = self.skill
        line = next(line for line in skill.splitlines() if line.startswith("- `run` is"))
        assert "window" in line
        assert "10x9" in line

    def test_it_no_longer_offers_a_runway_the_field_cannot_print(self):
        # `run` is counted inside a 10-wide, 9-tall window centred on the player,
        # so 5 is its ceiling. The old text used `left:7` as its worked example.
        assert "`left:7`" not in self.skill

    def test_crossing_further_than_the_screen_is_gotos_job(self):
        assert "longer than the screen belongs to `poke goto`" in self.skill

    def test_an_encounter_stopping_goto_is_named_as_a_pause(self):
        # 78 of the 113 `goto` calls that did not arrive in the measured leg had
        # stopped for a wild Pokemon. A model that reads those as failures learns
        # to distrust the one verb that crosses a map.
        section = self.skill.split("## Getting somewhere", 1)[1].split("## When you are lost", 1)[0]
        assert "wild Pokemon appeared" in section
        assert "does not resume itself" in section
