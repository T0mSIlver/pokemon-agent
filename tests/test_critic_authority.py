"""Two ways the critic told the next session something that was not true.

Measured over one 112-session run: 99 retrospectives delivered, of which 19
contained a false statement and 20 were byte-identical repeats of the one
before. Both classes are addressed here; neither is about the critic reasoning
badly, and both are about what the harness told it to believe.

**The map graph was labelled authoritative.** `world.json` holds which maps
touch. It has no representation of a cuttable tree, a guard who wants a drink,
a badge lock or a boulder — `grep` for any of those across the map layer returns
nothing. The critic prompt nonetheless said "If the map data says an exit is
north, it is north, however confidently the session argued otherwise", and
`direction_claims` printed "WRONG. The map data says …" over the agent's own
observations. Eight retrospectives in a row told the agent its correct note
about the Vermilion gym tree was "a belief, not a finding"; the tree was real,
and the run spent 4,480 tool calls in that city without opening the door.

**A failed critic re-served the last handoff.** The next session was then told,
in the present tense, about a session two back. Twenty of the ninety-nine were
that; they produced 4 milestones against a run rate of 12 in 112.
"""

from __future__ import annotations

from pathlib import Path

from pokemon_agent import critic


class TestTheGraphNoLongerOverrulesObservation:
    def test_the_prompt_no_longer_tells_the_critic_the_graph_always_wins(self):
        source = Path(critic.__file__).read_text(encoding="utf-8")
        assert "however\nconfidently the session argued otherwise" not in source
        assert "If the map data says an exit is north, it is north" not in source

    def test_the_prompt_names_what_the_graph_cannot_see(self):
        source = Path(critic.__file__).read_text(encoding="utf-8")
        for blocker in ("cuttable tree", "badge lock", "boulder"):
            assert blocker in source, blocker

    def test_a_disagreement_is_reported_as_something_to_check(self):
        """Not as a verdict against the agent.

        `direction_claims` takes the narration and the map the session started
        on, and routes each compass claim against the graph itself.
        """
        # A claim the graph disagrees with: Lavender is east and north of
        # Vermilion, not west. The old wording answered "WRONG."
        rows = critic.direction_claims("go west to Lavender Town", "Vermilion City")
        joined = " ".join(rows)
        assert rows, "the claim should be checked at all"
        assert "WRONG" not in joined
        assert "a thing to check rather than a thing the agent got wrong" in joined
        assert "not whether the way is walkable" in joined

    def test_a_claim_the_graph_agrees_with_is_still_confirmed(self):
        """The hedge must not turn every check into a shrug."""
        rows = critic.direction_claims("head south to Pewter City", "Cerulean City")
        assert rows and "agrees with the map data" in rows[0]


class TestRetiringAStaleHandoff:
    def test_a_handoff_is_moved_aside_rather_than_re_served(self, tmp_path: Path):
        critic.write_handoff(tmp_path, "the retrospective for session 40")
        assert critic.read_handoff(tmp_path)

        assert critic.retire_handoff(tmp_path) is True

        assert critic.read_handoff(tmp_path) == "", "session 41 must be told nothing"
        stale = tmp_path / critic.HANDOFF_STALE_FILENAME
        assert stale.is_file(), "and the post-mortem must survive"
        assert "session 40" in stale.read_text(encoding="utf-8")

    def test_retiring_nothing_is_not_an_error(self, tmp_path: Path):
        assert critic.retire_handoff(tmp_path) is False

    def test_the_next_good_critique_still_lands(self, tmp_path: Path):
        """Retiring must not poison the file for the session that succeeds."""
        critic.write_handoff(tmp_path, "stale one")
        critic.retire_handoff(tmp_path)
        critic.write_handoff(tmp_path, "a fresh retrospective")
        assert "fresh retrospective" in critic.read_handoff(tmp_path)
