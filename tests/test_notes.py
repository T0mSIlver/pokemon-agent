"""The harness's block in ``NOTES.md``.

Every test that matters here is a test that the model's own text survived. The
bug being fixed is a stale block; the bug that would be worse is a lost note.
"""

from __future__ import annotations

import pytest

from pokemon_agent import notes
from pokemon_agent.critic import read_notes

MODEL_TEXT = """## Your notes

Tried the S.S. Anne bow twice, the guard turns me back without the ticket.
Vermilion gym door is at (14,18) and the trash-can puzzle resets on a wrong pick.
Open question: does Bill's house need Cut?
"""

STATE = {
    "player": {
        "money": 7038,
        "badges": ["Boulder", "Cascade"],
        "position": {"x": 14, "y": 18},
        "facing": "down",
    },
    "map": {"map_id": 5, "map_name": "Vermilion City"},
    "party": [
        {
            "species": "CHARMELEON",
            "level": 32,
            "hp": 39,
            "max_hp": 92,
            "status": "OK",
            "types": ["Fire"],
            "moves": [
                {"name": "Rage", "pp": 14},
                {"name": "Growl", "pp": 40},
                {"name": "Ember", "pp": 25},
                {"name": "Cut", "pp": 30},
            ],
        }
    ],
    "bag": [{"item": "POKE BALL", "quantity": 5}, {"item": "POTION", "quantity": 3}],
    "flags": {"badges": ["Boulder", "Cascade"]},
}

PROGRESS = {"count": 16, "total": 58, "furthest": "The S.S. Anne set sail"}


def refresh(tmp_path, state=STATE, progress=PROGRESS):
    notes.refresh_notes(tmp_path, state=state, progress=progress)
    return notes.notes_path(tmp_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# What the block says
# ---------------------------------------------------------------------------


def test_block_carries_every_harness_owned_field():
    body = "\n".join(notes.state_block_lines(STATE, PROGRESS))
    assert "Vermilion City (14,18) facing down" in body
    assert "$7038" in body
    assert "Boulder, Cascade" in body
    assert "CHARMELEON L32 39/92 HP Fire" in body
    assert "Cut 30/30" in body
    assert "POKE BALL x5, POTION x3" in body
    assert "16/58, furthest: The S.S. Anne set sail" in body


def test_moves_carry_pp_out_of_the_move_table():
    """The failure this exists for: notes said Leer, the party held Cut."""

    body = "\n".join(notes.state_block_lines(STATE, PROGRESS))
    assert "Rage 14/20" in body
    assert "Leer" not in body


def test_pp_ups_never_produce_a_denominator_below_the_numerator():
    state = {"party": [{"species": "X", "level": 5, "moves": [{"name": "Ember", "pp": 33}]}]}
    assert "Ember 33/33" in "\n".join(notes.state_block_lines(state))


def test_unknown_move_keeps_the_count_it_read():
    state = {"party": [{"species": "X", "level": 5, "moves": [{"name": "???(212)", "pp": 7}]}]}
    assert "???(212) 7" in "\n".join(notes.state_block_lines(state))


def test_every_field_is_present_even_with_nothing_read():
    body = "\n".join(notes.state_block_lines(None, None))
    for field in ("- Where:", "- Money:", "- Badges:", "- Party:", "- Bag:", "- Milestones:"):
        assert field in body
    assert body.count(notes.UNREAD) >= 3


def test_facing_is_not_reported_in_a_battle():
    state = dict(STATE, battle={"in_battle": True})
    assert "in a battle, facing unread" in "\n".join(notes.state_block_lines(state))


def test_block_says_it_is_not_the_models_to_write():
    block = notes.state_block(STATE, PROGRESS)
    assert block.startswith(notes.BLOCK_BEGIN)
    assert block.rstrip().endswith(notes.BLOCK_END)
    assert "not yours" in block


def test_block_has_no_timestamp_so_two_renders_match():
    assert notes.state_block(STATE, PROGRESS) == notes.state_block(STATE, PROGRESS)


# ---------------------------------------------------------------------------
# Splicing: the model's text survives every shape
# ---------------------------------------------------------------------------


def test_models_notes_survive_a_rewrite(tmp_path):
    path = notes.notes_path(tmp_path)
    path.write_text(notes.state_block({}, None) + "\n" + MODEL_TEXT, encoding="utf-8")
    after = refresh(tmp_path)
    assert MODEL_TEXT in after
    assert "Vermilion City (14,18)" in after
    assert after.count(notes.BLOCK_BEGIN) == 1


def test_two_rewrites_in_a_row_are_identical(tmp_path):
    path = notes.notes_path(tmp_path)
    path.write_text(MODEL_TEXT, encoding="utf-8")
    first = refresh(tmp_path)
    second = refresh(tmp_path)
    assert first == second


def test_a_stale_block_is_replaced_not_appended(tmp_path):
    path = notes.notes_path(tmp_path)
    stale = notes.state_block(
        {"map": {"map_name": "Pewter City"}, "player": {"position": {"x": 1, "y": 2}}}, None
    )
    path.write_text(stale + "\n" + MODEL_TEXT, encoding="utf-8")
    after = refresh(tmp_path)
    assert "Pewter City" not in after
    assert after.count(notes.BLOCK_BEGIN) == 1
    assert MODEL_TEXT in after


def test_missing_file_is_created_with_a_section_for_the_model(tmp_path):
    workspace = tmp_path / "does-not-exist-yet"
    after = refresh(workspace)
    assert after.startswith(notes.BLOCK_BEGIN)
    assert notes.SEED_BODY in after
    assert refresh(workspace) == after


def test_empty_file_is_filled_without_losing_anything(tmp_path):
    notes.notes_path(tmp_path).write_text("", encoding="utf-8")
    after = refresh(tmp_path)
    assert notes.SEED_BODY in after
    assert refresh(tmp_path) == after


def test_whitespace_only_file_is_treated_as_empty(tmp_path):
    notes.notes_path(tmp_path).write_text("\n\n   \n", encoding="utf-8")
    after = refresh(tmp_path)
    assert notes.SEED_BODY in after
    assert refresh(tmp_path) == after


def test_file_with_no_delimiters_keeps_all_of_its_text(tmp_path):
    notes.notes_path(tmp_path).write_text(MODEL_TEXT, encoding="utf-8")
    after = refresh(tmp_path)
    assert MODEL_TEXT in after
    assert after.startswith(notes.BLOCK_BEGIN)


def test_file_with_no_trailing_newline_survives(tmp_path):
    notes.notes_path(tmp_path).write_text("one last thought", encoding="utf-8")
    after = refresh(tmp_path)
    assert "one last thought" in after
    assert after.endswith("\n")
    assert refresh(tmp_path) == after


def test_opening_delimiter_with_no_closing_one_deletes_nothing(tmp_path):
    """No way to know where the old block ended, so nothing is cut."""

    orphan = f"{notes.BLOCK_BEGIN}\nhalf a block\n{MODEL_TEXT}"
    notes.notes_path(tmp_path).write_text(orphan, encoding="utf-8")
    after = refresh(tmp_path)
    assert "half a block" in after
    assert MODEL_TEXT in after
    # And it is byte-stable from the next pass on.
    assert refresh(tmp_path) == after


def test_closing_delimiter_with_no_opening_one_deletes_nothing(tmp_path):
    orphan = f"{MODEL_TEXT}\n{notes.BLOCK_END}\n"
    notes.notes_path(tmp_path).write_text(orphan, encoding="utf-8")
    after = refresh(tmp_path)
    assert MODEL_TEXT in after
    assert refresh(tmp_path) == after


def test_a_duplicated_block_lower_down_is_left_to_the_model(tmp_path):
    """Only the pair at the top is the harness's; the copy below is model text."""

    copied = notes.state_block({"map": {"map_name": "Pewter City"}}, None)
    notes.notes_path(tmp_path).write_text(
        notes.state_block({}, None) + "\n" + MODEL_TEXT + "\n" + copied,
        encoding="utf-8",
    )
    after = refresh(tmp_path)
    assert after.count(notes.BLOCK_BEGIN) == 2
    assert MODEL_TEXT in after
    assert after.startswith(notes.BLOCK_BEGIN + "\n" + notes.BLOCK_TITLE)
    assert refresh(tmp_path) == after


def test_a_model_edit_inside_the_block_is_discarded_and_nothing_else_is(tmp_path):
    tampered = notes.state_block({"map": {"map_name": "Pewter City"}}, None).replace(
        "- Bag: empty", "- Bag: MASTER BALL x99\n- I moved this line here myself"
    )
    notes.notes_path(tmp_path).write_text(tampered + "\n" + MODEL_TEXT, encoding="utf-8")
    after = refresh(tmp_path)
    assert "MASTER BALL" not in after
    assert "I moved this line here myself" not in after
    assert MODEL_TEXT in after


def test_undecodable_file_is_refused_rather_than_overwritten(tmp_path):
    path = notes.notes_path(tmp_path)
    path.write_bytes(b"\xff\xfe not utf-8 at all")
    with pytest.raises(UnicodeDecodeError):
        notes.refresh_notes(tmp_path, state=STATE, progress=PROGRESS)
    assert path.read_bytes() == b"\xff\xfe not utf-8 at all"


def test_no_temp_file_is_left_behind(tmp_path):
    refresh(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [notes.NOTES_FILENAME]


# ---------------------------------------------------------------------------
# Stripping: the critic gets claims, not measurements
# ---------------------------------------------------------------------------


def test_strip_removes_the_block_and_keeps_the_rest():
    text = notes.state_block(STATE, PROGRESS) + "\n" + MODEL_TEXT
    stripped = notes.strip_state_block(text)
    assert "Vermilion City" not in stripped
    assert stripped == MODEL_TEXT


def test_strip_is_a_no_op_without_delimiters():
    assert notes.strip_state_block(MODEL_TEXT) == MODEL_TEXT
    assert notes.strip_state_block("") == ""


def test_critic_reads_only_the_models_half(tmp_path):
    refresh(tmp_path)
    notes.notes_path(tmp_path).write_text(
        notes.notes_path(tmp_path).read_text(encoding="utf-8") + MODEL_TEXT,
        encoding="utf-8",
    )
    seen = read_notes(tmp_path)
    assert "Vermilion City" not in seen
    assert "trash-can puzzle" in seen
    assert notes.BLOCK_BEGIN not in seen


# ---------------------------------------------------------------------------
# The supervisor's two calls
# ---------------------------------------------------------------------------


class _Facts:
    done_count = 16
    rung_done = "The S.S. Anne set sail"


def _supervisor(tmp_path):
    from pokemon_agent.pi_supervisor import PiSupervisor

    return PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:9",
        pi_binary="/nonexistent/pi",
    )


def test_supervisor_writes_the_block_from_the_context_it_already_has(tmp_path):
    supervisor = _supervisor(tmp_path)
    supervisor._refresh_notes({"game_state": STATE}, _Facts())

    written = (supervisor.workspace_dir / notes.NOTES_FILENAME).read_text(encoding="utf-8")
    assert "Vermilion City (14,18)" in written
    assert "16/58, furthest: The S.S. Anne set sail" in written


def test_supervisor_takes_the_milestone_score_off_the_facts(tmp_path):
    supervisor = _supervisor(tmp_path)
    assert supervisor._notes_progress(_Facts()) == {
        "count": 16,
        "furthest": "The S.S. Anne set sail",
    }
    assert supervisor._notes_progress(None) is None
    assert supervisor._notes_progress(object()) is None


def test_a_failed_refresh_never_raises_and_names_the_exception_type(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)

    def boom(*args, **kwargs):
        raise TypeError("unsupported operand")

    monkeypatch.setattr(notes, "refresh_notes", boom)
    supervisor._refresh_notes({"game_state": STATE}, _Facts())

    event = supervisor.recent_events[-1]
    assert event["type"] == "pi_notes_refresh_failed"
    # The `/route` fallback that swallowed a TypeError cost an hour. Name it.
    assert "TypeError" in event["summary"]
    assert "unsupported operand" in event["summary"]


def test_a_missing_context_still_writes_a_block(tmp_path):
    supervisor = _supervisor(tmp_path)
    supervisor._refresh_notes(None, None)

    written = (supervisor.workspace_dir / notes.NOTES_FILENAME).read_text(encoding="utf-8")
    assert written.startswith(notes.BLOCK_BEGIN)
    assert f"- Milestones: {notes.UNREAD}" in written
