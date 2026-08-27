import json
import os
from pathlib import Path

import pytest

from pokemon_agent.bench.registry import RunRegistry
from pokemon_agent.critic import (
    AUTO_SAVE_PREFIX,
    CLAIMS_HEADING,
    CRITIC_RAW_FILENAME,
    CRITIC_RAW_PREVIOUS_FILENAME,
    DEBUG_DIRNAME,
    DIGEST_CHAR_BUDGET,
    DIGEST_TOKEN_BUDGET,
    EXPLORED_STORE_FILENAME,
    FACTS_DIGEST_HEADING,
    HANDOFF_FILENAME,
    HANDOFF_PREVIOUS_FILENAME,
    MAX_HANDOFF_WORDS,
    NARRATION_HEADING,
    NEXT_GOAL_LABEL,
    NO_TEXT_ERROR,
    SALVAGED_REASONING_NOTICE,
    SAVES_DIRNAME,
    TARGET_HANDOFF_WORDS,
    CriticResult,
    DigestInput,
    Intel,
    SessionFacts,
    ToolCall,
    build_critic_command,
    build_digest,
    build_prompt,
    call_lines,
    cap_words,
    check_next_goal,
    classify_call,
    collect_intel,
    collect_session_facts,
    compute_behaviour_stats,
    coordinate_claims,
    coverage_lines,
    describe_no_text,
    direction_claims,
    estimate_tokens,
    handoff_body,
    handoff_path,
    ladder_position,
    list_named_saves,
    map_brief,
    mentioned_map,
    milestone_frontier,
    narration_lines,
    nearest_pokecenter,
    parse_actions,
    parse_final_text,
    parse_next_goal,
    read_handoff,
    read_session_mark,
    repeat_lines,
    route_text,
    run_critic,
    session_mark_path,
    strike_false_claims,
    tail_words,
    tool_calls_from_stream,
    trend_lines,
    untried_lines,
    waste_lines,
    write_handoff,
    write_session_mark,
)
from pokemon_agent.pi_supervisor import iter_jsonl_records

FAKE_PRINT_PI = """#!/usr/bin/env python3
# Minimal stand-in for `pi --mode json --print`: argv in, a JSON event stream out.
# It streams deltas before the finished message, the way the real one does, and can
# stop on `length` with no text at all - the failure that started all of this.

import json
import pathlib
import sys

PARAMS = __PARAMS__

argv = sys.argv[1:]
workspace = pathlib.Path.cwd()
(workspace / "critic_argv.json").write_text(json.dumps(argv), encoding="utf-8")
with (workspace / "critic_calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")


def opt(name):
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


config = dict(PARAMS)
config.update(PARAMS["overrides"].get(opt("--thinking") or "", {}))

if config["hang"]:
    import time

    while True:
        time.sleep(1)

if config["exit_code"]:
    sys.stderr.write("provider refused the request\\n")
    raise SystemExit(config["exit_code"])


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


def deltas(kind, body):
    for index in range(0, len(body), 8):
        emit(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": kind, "delta": body[index : index + 8]},
            }
        )


emit({"type": "session", "version": 3, "id": "critic-fake"})
emit({"type": "agent_start"})
emit({"type": "turn_start"})
emit(
    {
        "type": "message_start",
        "message": {"role": "user", "content": [{"type": "text", "text": "digest"}]},
    }
)

reasoning = config["thinking"]
text = config["text"]
deltas("thinking_delta", reasoning)
deltas("text_delta", text)

content = []
if reasoning:
    content.append({"type": "thinking", "thinking": reasoning})
if text:
    content.append({"type": "text", "text": text})

if content or config["stop_reason"] or config["usage"]:
    message = {"role": "assistant", "content": content}
    if config["stop_reason"]:
        message["stopReason"] = config["stop_reason"]
    if config["usage"]:
        message["usage"] = config["usage"]
    emit({"type": "message_end", "message": message})
    if text:
        emit(
            {
                "type": "turn_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                "toolResults": [],
            }
        )

emit({"type": "agent_end", "messages": [], "willRetry": False})
emit({"type": "agent_settled"})
"""


def make_fake_print_pi(
    tmp_path: Path,
    *,
    text: str = "Retrospective: you rammed the north wall 38 times.",
    thinking: str = "Counting the wall-rams.",
    stop_reason: str = "",
    usage: dict | None = None,
    overrides: dict | None = None,
    exit_code: int = 0,
    hang: bool = False,
) -> Path:
    # ``overrides`` swaps the answer per ``--thinking`` level, so a retry can differ.
    params = {
        "text": text,
        "thinking": thinking,
        "stop_reason": stop_reason,
        "usage": usage or {},
        "overrides": overrides or {},
        "exit_code": exit_code,
        "hang": hang,
    }
    digest = abs(hash(json.dumps(params, sort_keys=True))) % 1000000
    script = tmp_path / f"fake-pi-print-{digest}"
    script.write_text(FAKE_PRINT_PI.replace("__PARAMS__", repr(params)), encoding="utf-8")
    script.chmod(0o755)
    return script


def critic_calls(workspace: Path) -> list[list[str]]:
    # Every argv the fake pi was invoked with, oldest first.
    log = workspace / "critic_calls.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


def raw_stream(workspace: Path, filename: str = CRITIC_RAW_FILENAME) -> str:
    return (workspace / DEBUG_DIRNAME / filename).read_text(encoding="utf-8")


def action_call(
    actions: list[str],
    *,
    comment: str = "walk north",
    x: int = 5,
    y: int = 6,
    moved: int | None = 4,
    blocked_after: int | None = None,
    battle: bool = False,
) -> ToolCall:
    command = f"# {comment}\n./poke act " + " ".join(actions)
    response: dict = {"x": x, "y": y, "facing": "up", "hp": "18/22"}
    if moved is not None:
        response["moved"] = moved
    if blocked_after is not None:
        response["blocked_after"] = blocked_after
    if battle:
        response = {"mode": "battle", "battle": True, "hp": "18/22"}
    return ToolCall(
        name="bash",
        headline=comment,
        command=command,
        result_summary=f"x={x} y={y} facing=up",
        result_full=json.dumps({"output": json.dumps(response), "exitCode": 0}, indent=2),
    )


def sample_calls() -> list[ToolCall]:
    calls = [
        action_call(["walk_up"] * 8, comment="long run north", moved=2, blocked_after=3),
        action_call(["walk_up"] * 8, comment="try north again", moved=0, blocked_after=1),
        action_call(["walk_left", "walk_left"], comment="go west instead", x=3, y=6, moved=2),
        action_call(["press_a"], comment="talk to the sign", x=3, y=6, moved=None),
        action_call(["press_a"], comment="fight it", battle=True),
        ToolCall(
            name="read",
            headline="read latest_frame_annotated.png",
            path="/ws/latest_frame_annotated.png",
            result_summary="image 12.0 KB",
        ),
        ToolCall(
            name="bash",
            headline="check the whole map",
            command="# check the whole map\ncurl -sS http://localhost:8765/map",
            result_summary='{"map_id": 12}',
        ),
        ToolCall(
            name="write",
            headline="write NOTES.md",
            path="/ws/NOTES.md",
            result_summary="ok",
            is_error=True,
        ),
    ]
    # Walk the same tile twice more so the revisit counter has something to say.
    calls.append(action_call(["walk_right"], comment="back east", x=5, y=6, moved=1))
    calls.append(action_call(["walk_left"], comment="back west", x=5, y=6, moved=1))
    return calls


# ---------------------------------------------------------------------------
# Reading a session back
# ---------------------------------------------------------------------------


def test_stream_tool_entries_become_tool_calls():
    entries = [
        {"kind": "user", "text": "continue"},
        {
            "kind": "tool",
            "state": "error",
            "tool": {
                "name": "bash",
                "headline": "walk north",
                "command": "# walk north\ncurl /action",
                "result_summary": "boom",
                "result_full": "boom",
            },
        },
        {"kind": "tool", "state": "ok", "tool": None},
    ]

    calls = tool_calls_from_stream(entries)

    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].is_error is True
    assert calls[0].comment == "walk north"


def test_actions_and_classification_are_read_off_the_command():
    call = action_call(["walk_up", "walk_up", "press_a"])

    assert parse_actions(call.command) == ["walk_up", "walk_up", "press_a"]
    assert classify_call(call) == "action"
    assert classify_call(ToolCall(name="read", path="/ws/latest_frame.png")) == "frame_read"
    assert classify_call(ToolCall(name="read", path="/ws/NOTES.md")) == "file_read"
    assert classify_call(ToolCall(name="bash", command="curl /map")) == "map"
    assert classify_call(ToolCall(name="bash", command="ls")) == "bash"


def test_action_lines_show_the_buttons_and_the_outcome_not_the_curl_line():
    lines = call_lines(
        [
            action_call(
                ["walk_up"] * 8,
                comment="head north out of the clearing",
                moved=2,
                blocked_after=3,
            ),
            ToolCall(
                name="read",
                headline="read latest_map.png",
                path="/ws/latest_map.png",
                result_summary="image 6.5 KB",
            ),
        ]
    )

    assert lines[0] == (
        "- head north out of the clearing [walk_up x8]"
        "  ->  x=5 y=6 facing=up moved=2 blocked_after=3 hp=18/22"
    )
    assert lines[1] == "- read latest_map.png  ->  image 6.5 KB"
    assert all(len(line) <= 200 for line in lines)


def test_a_batch_the_cli_rejects_contributes_no_actions():
    """`poke` refuses the whole batch on the first bad token and sends nothing."""
    rejected = ToolCall(name="bash", command="./poke act up typo down")

    assert parse_actions(rejected.command) == []
    assert parse_actions("./poke act up:0") == []
    assert parse_actions("./poke act --json up") == []
    assert parse_actions("./poke act") == []
    stats = compute_behaviour_stats([rejected])
    assert (stats["action_batches"], stats["total_buttons"]) == (1, 0)


def test_global_options_before_the_subcommand_are_not_mistaken_for_actions():
    assert parse_actions("poke --port 9000 act up up") == ["walk_up", "walk_up"]
    assert parse_actions("./poke --url http://box:1/ act a:2") == ["press_a", "press_a"]
    assert parse_actions("./poke act --port 9000 down") == ["walk_down"]
    assert classify_call(ToolCall(name="bash", command="poke --port 9000 act up")) == "action"
    assert classify_call(ToolCall(name="bash", command="poke --port 9000 state")) == "state"


def test_action_lines_fall_back_to_a_label_when_the_agent_wrote_no_comment():
    call = action_call(["walk_left", "walk_left", "press_a"], comment="")

    assert call_lines([call])[0].startswith("- action [walk_left x2, press_a]  ->  x=5 y=6")


def test_behaviour_stats_count_wall_rams_revisits_and_the_tool_mix():
    stats = compute_behaviour_stats(sample_calls())

    assert stats["action_batches"] == 7
    assert stats["total_buttons"] == 22
    assert stats["average_batch_size"] == 3.1
    assert stats["blocked_batches"] == 2
    assert stats["blocked_fraction"] == pytest.approx(2 / 7, abs=0.001)
    assert stats["moved_zero_batches"] == 1
    assert stats["battles"] == 1
    assert stats["distinct_tiles"] == 2
    assert stats["positions_sampled"] == 6
    assert stats["top_tiles"][0] == {"x": 5, "y": 6, "visits": 4}
    assert stats["tool_mix"]["action"] == 7
    assert stats["tool_mix"]["frame_read"] == 1
    assert stats["tool_mix"]["map"] == 1
    assert stats["tool_errors"] == 1


def test_narration_is_deduplicated_and_sampled_across_the_run():
    calls = [action_call(["walk_up"], comment=f"step {index}") for index in range(120)]
    calls.insert(3, action_call(["walk_up"], comment="step 2"))

    lines = narration_lines(calls, limit=10)

    assert len(lines) == 10
    assert lines[0] == "step 0"
    assert len(set(lines)) == 10


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_digest_carries_the_stats_the_state_and_the_map(tmp_path: Path):
    digest = build_digest(
        DigestInput(
            goal="Reach Viridian City.",
            objective="Deliver Oak's parcel",
            turns_completed=4,
            status="completed",
            status_reason="Token budget reached (110123/110000 context tokens).",
            session_tokens=110_123,
            start_state={
                "map": {"map_name": "PALLET TOWN", "map_id": 0},
                "player": {"position": {"x": 5, "y": 6}, "badges": [], "money": 3000},
                "party": [{"species": "CHARMANDER", "level": 5, "hp": 19, "max_hp": 19}],
            },
            final_state={
                "map": {"map_name": "ROUTE 1", "map_id": 12},
                "player": {"position": {"x": 9, "y": 21}, "badges": [], "money": 3000},
                "party": [{"species": "CHARMANDER", "level": 8, "hp": 12, "max_hp": 26}],
            },
            map_summary={
                "map_id": 12,
                "map_name": "ROUTE 1",
                "width": 20,
                "height": 36,
                "coverage": {"seen": 240, "walked": 130, "total": 720},
                "warps": [{"x": 9, "y": 0}, {"x": 10, "y": 35}],
                "unexplored_nearest": {"x": 4, "y": 18},
            },
            notes="# Notes\nHead north on Route 1.",
            calls=sample_calls(),
        )
    )

    assert "Goal: Reach Viridian City." in digest
    assert "Objective: Deliver Oak's parcel" in digest
    assert "Context tokens used: 110123" in digest
    # start vs end, so progress is visible
    assert "PALLET TOWN" in digest
    assert "ROUTE 1" in digest
    assert "CHARMANDER L5" in digest
    assert "CHARMANDER L8" in digest
    # the hard behavioural stats
    assert "/action batches: 7" in digest
    assert "Buttons sent: 22 (average batch 3.1)" in digest
    assert "blocked_after): 2 of 7 (29%)" in digest
    assert "Batches that ended with moved=0: 1" in digest
    assert "Battles entered: 1" in digest
    assert "2 distinct out of 6 positions sampled" in digest
    assert "(5,6)x4" in digest
    assert "action=7" in digest
    # explored map, narration, tool log and notes
    assert "Coverage: seen=240, total=720, walked=130" in digest
    assert "Warps: (9,0), (10,35)" in digest
    assert "long run north" in digest
    assert "Head north on Route 1." in digest


def test_digest_stays_under_the_token_budget_for_a_huge_session():
    calls = [
        action_call(["walk_up"] * 8, comment=f"step {index} of a very long grind north")
        for index in range(4000)
    ]

    digest = build_digest(
        DigestInput(goal="Grind.", notes="n" * 20_000, calls=calls),
    )

    assert len(digest) <= DIGEST_CHAR_BUDGET
    assert estimate_tokens(digest) <= DIGEST_TOKEN_BUDGET
    # trimming never costs the numbers
    assert "/action batches: 4000" in digest


def test_prompt_states_the_word_cap_and_the_reader():
    prompt = build_prompt("# Finished session digest\n")

    assert f"{MAX_HANDOFF_WORDS} is the hard ceiling" in prompt
    assert f"about {TARGET_HANDOFF_WORDS} words" in prompt
    assert "no reasoning" in prompt
    assert "# Finished session digest" in prompt


# ---------------------------------------------------------------------------
# The pi invocation
# ---------------------------------------------------------------------------


def test_command_is_a_one_shot_print_run_with_read_only_tools(tmp_path: Path):
    image = tmp_path / "latest_map.png"
    image.write_bytes(b"map")

    command = build_critic_command(
        "/usr/bin/pi",
        prompt="review this",
        provider="llamacpp",
        model="qwen38-27b",
        images=[image],
    )

    assert command == [
        "/usr/bin/pi",
        "--mode",
        "json",
        "--print",
        "--thinking",
        "xhigh",
        "--provider",
        "llamacpp",
        "--model",
        "qwen38-27b",
        "-ne",
        "-ns",
        "-nc",
        "-np",
        "--no-themes",
        "--offline",
        "--no-session",
        "--tools",
        "read",
        f"@{image}",
        "review this",
    ]


def test_final_text_is_the_last_assistant_message():
    stream = "\n".join(
        [
            json.dumps({"type": "session", "id": "x"}),
            "not json at all",
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "the digest"}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hidden"},
                            {"type": "text", "text": "first pass"},
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "final answer"}],
                    },
                }
            ),
        ]
    )

    assert parse_final_text(stream) == "final answer"
    assert parse_final_text("") == ""


def test_a_handoff_under_the_cap_is_left_exactly_as_written():
    assert cap_words("one two three", limit=5) == "one two three"
    whole = "First point. Second point. Third point, the one that matters."
    assert cap_words(whole, limit=MAX_HANDOFF_WORDS) == whole
    assert "truncated" not in cap_words(whole, limit=MAX_HANDOFF_WORDS)


def test_the_word_cap_cuts_at_a_sentence_boundary_and_says_so():
    text = "Alpha one two three. Beta four five six. Gamma seven eight nine."
    capped = cap_words(text, limit=6)

    # Cut back to the end of the first sentence, not chopped at the sixth word.
    assert capped.startswith("Alpha one two three.")
    assert "Beta" not in capped.split("_[truncated")[0]
    assert "truncated at the last sentence end" in capped
    assert "8 more words not shown" in capped
    assert not capped.rstrip().endswith("...")


def test_the_word_cap_falls_back_to_a_paragraph_break():
    # The half-written second paragraph has no sentence end to cut at, so the break
    # above it is the last boundary that fits.
    text = "One two three.\n\nFour five six seven eight nine ten."
    capped = cap_words(text, limit=6)

    assert capped.startswith("One two three.")
    assert "Four" not in capped
    assert "truncated at the last paragraph break" in capped


def test_a_run_on_with_no_boundary_admits_the_cut_is_mid_sentence():
    capped = cap_words(" ".join(["word"] * 400), limit=10)

    assert capped.startswith("word word")
    assert "no sentence break found" in capped
    assert "390 more words not shown" in capped


def test_the_reasoning_tail_starts_on_a_whole_sentence():
    text = "Alpha one two. Beta three four. Gamma five six. Delta seven eight."
    tail = tail_words(text, limit=6)

    assert tail.startswith("_[truncated:")
    assert "picking up at the next sentence" in tail
    assert tail.endswith("Delta seven eight.")
    # Never opens mid-clause: whatever survives begins a sentence of its own.
    body = tail.split("\n\n", 1)[1]
    assert body.startswith("Gamma") or body.startswith("Delta")


# ---------------------------------------------------------------------------
# The next-session goal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        (
            "Some retrospective.\n\nNEXT GOAL: Walk north from Pewter to Route 3.",
            "Walk north from Pewter to Route 3",
        ),
        (
            "Some retrospective.\n**NEXT GOAL:** Heal at the Viridian Center, then Route 2.",
            "Heal at the Viridian Center, then Route 2",
        ),
        ("- next goal — Buy 10 Potions before Mt Moon.", "Buy 10 Potions before Mt Moon"),
        (
            "#### Next Goal:\n\nBeat Misty for the Cascade Badge.",
            "Beat Misty for the Cascade Badge",
        ),
        # Restated later in the reply: the last one is the one it settled on.
        (
            "NEXT GOAL: Go to Pewter.\nOn reflection.\nNEXT GOAL: Go to Cerulean instead.",
            "Go to Cerulean instead",
        ),
    ],
    ids=["plain", "bold", "bullet-dash", "heading-then-line", "restated"],
)
def test_the_next_goal_survives_however_the_critic_decorated_it(reply: str, expected: str):
    assert parse_next_goal(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "A retrospective with no goal line at all.",
        "NEXT GOAL:",
        "NEXT GOAL: ok",
        "NEXT GOAL: :::",
        "NEXT GOAL: " + "filler " * 200,
        "The next goal is something we should think about later.",
        "",
    ],
    ids=["absent", "empty", "too-short", "punctuation", "a-whole-plan", "prose", "nothing"],
)
def test_a_goal_the_parser_cannot_trust_is_no_goal_at_all(reply: str):
    assert parse_next_goal(reply) == ""


def test_the_prompt_asks_for_the_next_goal_in_a_parseable_shape():
    prompt = build_prompt("# Finished session digest\n")

    assert f"{NEXT_GOAL_LABEL}: <one short imperative sentence" in prompt
    assert "already achieved is finished" in prompt
    # The retry has to ask for it too, or a salvaged session loses its goal.
    assert NEXT_GOAL_LABEL in build_prompt("# digest\n", immediate=True)


@pytest.mark.asyncio
async def test_a_critique_carries_its_next_goal_off_the_run(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="You won the Boulder Badge.\n\nNEXT GOAL: Head north through Route 3 to Mt Moon.",
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )

    assert result.ok is True
    assert result.next_goal == "Head north through Route 3 to Mt Moon"
    # The line stays in the handoff too; the next session reads it as an instruction.
    assert "NEXT GOAL:" in read_handoff(workspace)


@pytest.mark.asyncio
async def test_a_critique_without_a_goal_line_reports_no_next_goal(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(tmp_path, text="You won the Boulder Badge. Nothing more to add.")

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )

    assert result.ok is True
    assert result.next_goal == ""


@pytest.mark.asyncio
async def test_a_long_retrospective_reaches_the_handoff_whole(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Long enough that the old 300-word cap would have chopped it mid-sentence,
    # and still inside the current one, which is what the ceiling is for.
    body = "\n\n".join(
        f"{index}. Mistake {index}: you rammed the wall at (5,6) {index} times."
        for index in range(20)
    )
    critique = f"{body}\n\nNEXT GOAL: Take the west path out of Viridian Forest."
    assert 200 < len(critique.split()) <= MAX_HANDOFF_WORDS
    fake_pi = make_fake_print_pi(tmp_path, text=critique)

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )

    assert result.ok is True
    assert result.text == critique
    assert "truncated" not in result.text
    assert read_handoff(workspace) == critique
    assert result.next_goal == "Take the west path out of Viridian Forest"


# ---------------------------------------------------------------------------
# Handoff files
# ---------------------------------------------------------------------------


def test_writing_a_handoff_rotates_the_previous_one(tmp_path: Path):
    write_handoff(tmp_path, "first critique")
    write_handoff(tmp_path, "second critique")

    assert read_handoff(tmp_path) == "second critique"
    assert (tmp_path / HANDOFF_PREVIOUS_FILENAME).read_text(encoding="utf-8").strip() == (
        "first critique"
    )
    assert read_handoff(tmp_path / "nowhere") == ""


# ---------------------------------------------------------------------------
# Running the critic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_writes_the_handoff_and_attaches_the_images(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "latest_map.png").write_bytes(b"map")
    (workspace / "latest_frame_annotated.png").write_bytes(b"frame")
    write_handoff(workspace, "the previous critique")
    fake_pi = make_fake_print_pi(tmp_path, text="You rammed the north wall in 38% of batches.")

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n\n## Session\nGoal: test\n",
        provider="llamacpp",
        model="qwen38-27b",
        timeout_seconds=20,
    )

    assert result.ok is True
    assert result.text == "You rammed the north wall in 38% of batches."
    assert result.error is None
    assert result.digest_tokens > 0
    assert result.handoff_path == str(workspace / HANDOFF_FILENAME)
    assert read_handoff(workspace) == "You rammed the north wall in 38% of batches."
    assert (workspace / HANDOFF_PREVIOUS_FILENAME).read_text(encoding="utf-8").strip() == (
        "the previous critique"
    )

    argv = json.loads((workspace / "critic_argv.json").read_text(encoding="utf-8"))
    assert f"@{workspace / 'latest_map.png'}" in argv
    assert f"@{workspace / 'latest_frame_annotated.png'}" in argv
    assert argv[-1].startswith("You are reviewing a finished session")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": "", "thinking": ""},
        {"exit_code": 3},
        {"hang": True},
    ],
    ids=["empty-output", "non-zero-exit", "timeout"],
)
async def test_a_failing_critic_keeps_the_previous_handoff(tmp_path: Path, kwargs: dict):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_handoff(workspace, "the previous critique")
    fake_pi = make_fake_print_pi(tmp_path, **kwargs)

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=1.5,
    )

    assert isinstance(result, CriticResult)
    assert result.ok is False
    assert result.error
    assert read_handoff(workspace) == "the previous critique"
    assert not (workspace / HANDOFF_PREVIOUS_FILENAME).exists()


@pytest.mark.asyncio
async def test_a_missing_pi_binary_is_reported_not_raised(tmp_path: Path):
    result = await run_critic(
        pi_binary=None,
        workspace_dir=tmp_path,
        digest="# Finished session digest\n",
    )

    assert result.ok is False
    assert "not found" in (result.error or "")


@pytest.mark.asyncio
async def test_an_unlaunchable_critic_is_reported_not_raised(tmp_path: Path):
    result = await run_critic(
        pi_binary=str(tmp_path / "does-not-exist"),
        workspace_dir=tmp_path,
        digest="# Finished session digest\n",
    )

    assert result.ok is False
    assert "Could not launch" in (result.error or "")


# ---------------------------------------------------------------------------
# Watching the critic run, and surviving a critic that never answers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_critic_streams_its_reasoning_and_its_answer_as_events(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="Take the west path from (5,6).",
        thinking="Counting the wall-rams: 38 of them.",
    )
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
        event_sink=sink,
    )

    assert result.ok is True
    kinds = [event["type"] for event in events]
    assert kinds[0] == "attempt_start"
    assert events[0]["thinking"] == "xhigh"
    assert kinds.count("thinking_end") == 1
    assert kinds.count("text_end") == 1
    assert kinds.index("thinking_delta") < kinds.index("text_delta") < kinds.index("text_end")
    reasoning = "".join(event["delta"] for event in events if event["type"] == "thinking_delta")
    assert reasoning == "Counting the wall-rams: 38 of them."
    said = next(event["text"] for event in events if event["type"] == "text_end")
    assert said == "Take the west path from (5,6)."


@pytest.mark.asyncio
async def test_the_raw_event_stream_is_written_and_rotated(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = make_fake_print_pi(tmp_path, text="first critique")
    second = make_fake_print_pi(tmp_path, text="second critique")

    await run_critic(
        pi_binary=str(first),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )
    result = await run_critic(
        pi_binary=str(second),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )

    assert result.raw_path == str(workspace / DEBUG_DIRNAME / CRITIC_RAW_FILENAME)
    assert "second critique" in raw_stream(workspace)
    assert "first critique" in raw_stream(workspace, CRITIC_RAW_PREVIOUS_FILENAME)
    # Still strict JSONL, so a post-mortem can just parse it.
    assert iter_jsonl_records(raw_stream(workspace))[0]["type"] == "session"


@pytest.mark.asyncio
async def test_stderr_is_kept_beside_the_event_stream(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(tmp_path, exit_code=3)

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
    )

    assert result.ok is False
    assert "provider refused the request" in raw_stream(workspace)


@pytest.mark.asyncio
async def test_a_run_that_reasons_past_its_ceiling_reports_why_it_stopped(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_handoff(workspace, "the previous critique")
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="",
        thinking="",
        stop_reason="length",
        usage={"input": 2303, "output": 16384},
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
        retry_enabled=False,
    )

    assert result.ok is False
    assert result.error == "Critic produced no text (stopReason=length, output=16384)."
    assert result.stop_reason == "length"
    assert result.usage == {"input": 2303, "output": 16384}
    assert [attempt["thinking"] for attempt in result.attempts] == ["xhigh"]
    assert read_handoff(workspace) == "the previous critique"


def test_describe_no_text_says_as_much_as_the_stream_told_it():
    assert describe_no_text(None, None) == NO_TEXT_ERROR
    assert describe_no_text("length", {"input": 2303, "output": 16384}) == (
        "Critic produced no text (stopReason=length, output=16384)."
    )
    assert describe_no_text(None, {"outputTokens": 12}) == "Critic produced no text (output=12)."
    assert describe_no_text("toolUse", None) == "Critic produced no text (stopReason=toolUse)."


@pytest.mark.asyncio
async def test_truncated_reasoning_is_salvaged_into_a_marked_handoff(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_handoff(workspace, "the previous critique")
    # Long enough to still overrun the raised cap, so the tail path is exercised.
    reasoning = ". ".join(f"thought{index}" for index in range(2000)) + "."
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="",
        thinking=reasoning,
        stop_reason="length",
        usage={"output": 16384},
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
        retry_enabled=False,
    )

    assert result.ok is True
    assert result.salvaged is True
    assert "Critic produced no text (stopReason=length, output=16384)." in (result.error or "")
    # The salvage lands on disk for a post-mortem, marked for what it is.
    written = handoff_path(workspace).read_text(encoding="utf-8")
    assert written.startswith(SALVAGED_REASONING_NOTICE)
    # The tail of the reasoning is the part that was still being written.
    assert "thought1999" in written
    assert "thought0." not in written
    assert "truncated" in written
    # But it is NOT what the next session reads. See read_handoff: the live one
    # was 1,648 bytes of the critic counting words at itself, inside a first
    # message of about 2,100.
    assert read_handoff(workspace) == ""
    assert len(written.split()) <= MAX_HANDOFF_WORDS
    assert (workspace / HANDOFF_PREVIOUS_FILENAME).read_text(encoding="utf-8").strip() == (
        "the previous critique"
    )


@pytest.mark.asyncio
async def test_an_empty_first_pass_buys_one_cheaper_retry(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="",
        thinking="",
        stop_reason="length",
        usage={"output": 16384},
        overrides={"medium": {"text": "Take the west path from (5,6).", "stop_reason": ""}},
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=20,
        retry_min_seconds=0.0,
    )

    assert result.ok is True
    assert result.salvaged is False
    assert result.text == "Take the west path from (5,6)."
    assert read_handoff(workspace) == "Take the west path from (5,6)."
    assert [attempt["thinking"] for attempt in result.attempts] == ["xhigh", "medium"]

    calls = critic_calls(workspace)
    assert len(calls) == 2
    assert calls[1][calls[1].index("--thinking") + 1] == "medium"
    assert "This is the second attempt." in calls[1][-1]
    assert f"{MAX_HANDOFF_WORDS} is the hard ceiling" in calls[1][-1]
    # Both attempts survive on disk: the retry rotated the first one, it did not eat it.
    assert "16384" in raw_stream(workspace, CRITIC_RAW_PREVIOUS_FILENAME)


@pytest.mark.asyncio
async def test_a_first_pass_that_times_out_still_buys_the_retry(tmp_path: Path):
    """The case the retry exists for, and the one it could not reach.

    The first pass used to be handed the whole budget, so a pass that ran out
    of time had by definition spent everything and the remainder the retry is
    sized against was nothing. The retry only ever fired when the first pass
    failed *fast* — an empty answer, a non-zero exit — which is not why it was
    written. Capping the first pass below the total is what makes it reachable.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(
        tmp_path,
        hang=True,
        overrides={"medium": {"hang": False, "text": "Take the west path from (5,6)."}},
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        timeout_seconds=6.0,
        first_attempt_seconds=1.0,
        retry_min_seconds=0.5,
    )

    assert [attempt["thinking"] for attempt in result.attempts] == ["xhigh", "medium"]
    assert result.text == "Take the west path from (5,6)."
    assert result.ok is True
    # The whole budget was never spent: the cap is what left room to retry.
    assert result.duration_seconds < 6.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"retry_enabled": False}, {"timeout_seconds": 1.0}],
    ids=["disabled", "out-of-time"],
)
async def test_the_retry_is_skipped_when_disabled_or_out_of_time(tmp_path: Path, kwargs: dict):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_pi = make_fake_print_pi(
        tmp_path,
        text="",
        thinking="",
        overrides={"medium": {"text": "a retry answer"}},
    )

    result = await run_critic(
        pi_binary=str(fake_pi),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        **{"timeout_seconds": 20, **kwargs},
    )

    assert result.ok is False
    assert len(critic_calls(workspace)) == 1
    assert len(result.attempts) == 1
    assert read_handoff(workspace) == ""


def test_the_prompt_asks_for_the_answer_before_any_elaboration():
    prompt = build_prompt("# Finished session digest\n")

    assert "Write the retrospective first." in prompt
    assert prompt.index("Write the retrospective first.") < prompt.index("Below is a digest")
    assert f"{MAX_HANDOFF_WORDS} is the hard ceiling" in prompt
    assert "cite the digest" in prompt

    retry = build_prompt("# Finished session digest\n", immediate=True)
    assert "This is the second attempt." in retry
    assert "Do not deliberate" in retry
    assert f"{MAX_HANDOFF_WORDS} is the hard ceiling" in retry


# ---------------------------------------------------------------------------
# Ground truth off the receipts
#
# The critic used to be told only what the model said it did. One retrospective
# on disk read the session's own narration and told the next session to go and
# beat a gym leader the run had already beaten. These numbers come off
# receipts.jsonl, which no agent writes to.
# ---------------------------------------------------------------------------


def record_run(data_dir: Path, receipts: list[dict], *, run_id: str = "") -> str:
    """Write a run the way a live session would, and hand back its id."""

    registry = RunRegistry(data_dir)
    run_id = run_id or registry.start_run(
        harness_sha="", config_hash="", start_checkpoint=None, goal="Cross Route 3.", model="fake"
    )
    for seq, payload in enumerate(receipts):
        registry.append(run_id, {"seq": seq, **payload})
    registry.close_all()
    return run_id


def pocket_receipts(start: float) -> list[dict]:
    """The Route 3 pocket: 26 tiles, hundreds of presses, no progress."""

    rows: list[dict] = [
        {
            "t": start - 1,
            "presses": 0,
            "tool": "run_start",
            "milestone_count": 8,
            "baseline_milestones": ["BADGE_BOULDER", "EVENT_BEAT_BROCK"],
        }
    ]
    for index in range(8):
        rows.append(
            {
                "t": start + index,
                "presses": 40,
                "tool": "action",
                "map": "Route 3",
                "pos": [22, 12],
                "moved": 0,
                "hp": [62, 62],
                "party_size": 1,
            }
        )
    rows.append(
        {
            "t": start + 9,
            "presses": 32,
            "tool": "goto",
            "map": "Route 3",
            "pos": [22, 8],
            "moved": 32,
            "hp": [62, 62],
            "party_size": 1,
        }
    )
    return rows


def test_the_facts_are_read_off_the_receipts_not_the_narration(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))

    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=1000.0, session_index=3)

    assert facts is not None
    assert facts.total_presses == 352
    assert facts.session_presses == 352
    assert facts.session_batches == 9
    assert facts.blocked_batches == 8
    assert facts.unique_positions == 2
    assert facts.position_samples == 9
    assert facts.hot_map == "Route 3"
    assert facts.hot_pos == (22, 12)
    assert facts.hot_visits == 8
    assert facts.ended_map == "Route 3"
    assert facts.ended_pos == (22, 8)
    assert facts.ended_hp == (62, 62)
    assert facts.tool_mix == (("action", 8), ("goto", 1))
    # The baseline is what stops a retrospective ordering a badge it already has.
    assert facts.done == ("BADGE_BOULDER", "EVENT_BEAT_BROCK")
    assert facts.gained == ()
    assert "run_start" not in dict(facts.tool_mix)


def test_a_session_that_pressed_nothing_says_so_rather_than_claiming_the_run(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))

    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=9_999.0)

    assert facts is not None
    # The run's own total is still the run's own total: only the slice is empty.
    assert facts.total_presses == 352
    assert facts.session_presses == 0
    assert facts.session_batches == 0
    assert "0 presses over 0 batches" in "\n".join(facts.lines())


def test_a_run_that_cannot_be_read_is_a_run_with_no_facts(tmp_path: Path):
    assert collect_session_facts(data_dir=tmp_path, run_id="20990101T000000Z-dead") is None
    assert collect_session_facts(data_dir=tmp_path, run_id="") is None
    assert collect_session_facts(data_dir=None, run_id="anything") is None


def test_only_the_saves_the_agent_named_are_worth_a_token(tmp_path: Path):
    saves = tmp_path / SAVES_DIRNAME
    saves.mkdir()
    for index, name in enumerate(["pewter_start", "before_mt_moon", "route3_ledge"]):
        path = saves / f"{name}.state"
        path.write_bytes(b"x")
        os.utime(path, (1000 + index, 1000 + index))
    for index in range(50):
        auto = saves / f"{AUTO_SAVE_PREFIX}{index:06d}.state"
        auto.write_bytes(b"x")
        os.utime(auto, (2000 + index, 2000 + index))
    (saves / "notes.txt").write_text("not a save", encoding="utf-8")

    names = list_named_saves(tmp_path, limit=2)

    # Newest first, autosaves excluded however new they are: 3,322 of them sat on
    # disk beside `pewter_start` and none was a place the agent chose to return to.
    assert names == ("route3_ledge", "before_mt_moon")
    assert list_named_saves(tmp_path / "nowhere") == ()
    assert list_named_saves(tmp_path, limit=0) == ()


def test_the_session_mark_survives_a_process_that_did_not(tmp_path: Path):
    write_session_mark(tmp_path, run_id="20260825T224823Z-983b", started_t=1234.5, session_index=4)

    mark = read_session_mark(tmp_path)
    assert mark["run_id"] == "20260825T224823Z-983b"
    assert mark["started_t"] == 1234.5
    assert mark["session_index"] == 4
    # And the history nothing else on disk keeps: `run_start` is written once per
    # run, and a run is many sessions long, so without this the receipts cannot
    # be cut into sessions and no comparison between them is possible.
    assert mark["history"] == [
        {"run_id": "20260825T224823Z-983b", "started_t": 1234.5, "session_index": 4}
    ]
    assert read_session_mark(tmp_path / "nowhere") == {}

    session_mark_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert read_session_mark(tmp_path) == {}


def test_the_facts_block_is_worth_its_tokens(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))
    saves = data_dir / SAVES_DIRNAME
    saves.mkdir(parents=True)
    for name in ["pewter_start", "before_mt_moon", "route3_ledge", "brock_beaten"]:
        (saves / f"{name}.state").write_bytes(b"x")

    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=1000.0)
    block = facts.render()

    # The whole point is that this is cheap. A briefing is a regression.
    #
    # The ceiling moved from 200 tokens to 320 to buy three lines of static map
    # data - every exit off the map it is standing on, the routed way to the map
    # the goal names, and where to heal when the party is hurt. It is paid for by
    # cutting the raw milestone-id list for the ladder's own labels, folding the
    # whiteout counter into the walking line, and dropping the `action xN` count
    # that the press total above it already implies.
    assert estimate_tokens(block) < 320
    assert len(block.splitlines()) <= 12
    # And that the next session can act on every line of it.
    assert "Most revisited tile: Route 3 (22,12), stood on 8 times." in block
    assert "`./poke load <name>`" in block
    # Ladder labels, not event ids: "BADGE_BOULDER" is not an instruction.
    assert "highest rung: Boulder Badge" in block
    assert "Next rung: Beat the Super Nerd guarding the Mt. Moon fossils" in block
    # And the geography, off the world file rather than off the model. No legend
    # here, because nothing on this line is starred: measured on the live run's
    # session 6 the same line carried "(* = never stepped on)" over an exits
    # string with no `*` in it, 22 bytes teaching a notation it does not use.
    assert "Every way off Route 3: walk north -> Route 4" in block
    assert "never stepped on" not in block


def test_the_digest_puts_the_receipts_above_the_models_own_account(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))
    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=1000.0)

    digest = build_digest(
        DigestInput(goal="Cross Route 3.", notes="n" * 20_000, calls=[], facts=facts)
    )

    assert FACTS_DIGEST_HEADING in digest
    assert digest.index(FACTS_DIGEST_HEADING) < digest.index("What it did (measured")
    # Trimming eats the narration and the notes; it never eats the facts.
    assert "Most revisited tile: Route 3 (22,12)" in digest
    assert len(digest) <= DIGEST_CHAR_BUDGET


def test_the_critic_is_told_the_receipts_win(tmp_path: Path):
    prompt = build_prompt("# Finished session digest\n")

    assert "the measurement wins and a claim it contradicts is a claim you must not make" in prompt
    assert "handed to the next agent verbatim" in prompt
    # NOTES.md is the model talking about itself, and one retrospective on disk
    # repeated "machine INACCESSIBLE (confirmed)" out of it as though that
    # settled anything. The agent healed at that machine 26 seconds later.
    assert "Never restate a claim as a fact" in prompt
    assert "do not name it" in prompt


def test_a_receipt_written_the_instant_a_session_began_belongs_to_it(tmp_path: Path):
    """Receipts round `t` to the millisecond; the mark keeps the raw clock.

    Without a tick of slack the first receipt of a session rounds to just before
    the session started and drops out of its own slice, which shows up as a
    session that pressed fewer buttons than it did.
    """

    data_dir = tmp_path / "data"
    started = 1000.001_3
    # Written after the session began, and rounds to before it: 1000.001 < 1000.0013.
    first = round(started + 0.000_1, 3)
    assert first < started
    run_id = record_run(
        data_dir,
        [
            {"t": first, "presses": 7, "tool": "action", "map": "Route 3"},
            {"t": round(started + 5, 3), "presses": 3, "tool": "action", "map": "Route 3"},
        ],
    )

    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=started)

    assert facts is not None
    assert facts.session_presses == 10


# ---------------------------------------------------------------------------
# Static world intelligence
#
# The player is on a minimum-context diet; the critic is not. These are the
# lookups it gets that the session never had, and every one of them exists
# because a transcript on disk shows the model getting that exact fact wrong
# from memory and then acting on it.
# ---------------------------------------------------------------------------


def test_a_map_brief_is_the_game_data_not_a_recollection():
    brief = map_brief("Route 3")

    assert brief.known
    assert brief.map_id == 14
    assert brief.size == (70, 18)
    assert ("north", "Route 4") in brief.connections
    assert ("west", "Pewter City") in brief.connections
    assert brief.trainers  # eight of them, and the session walked past every one
    assert brief.encounter_rate == 20
    assert map_brief("Kalos").known is False
    assert map_brief("").lines() == []


def test_the_exits_line_stars_the_ways_out_that_were_never_taken():
    brief = map_brief("Mt Moon B1F")

    exits = brief.exits(stood_on=[(25, 9)])

    # 720 presses went into a pocket whose way out was a warp tile the agent
    # never stepped on, and nothing it could read said which tiles those were.
    assert "(25,9) -> Mt Moon 1F" in exits
    assert "(27,3)* -> Route 4" in exits


def test_a_route_is_hops_off_the_map_graph():
    assert route_text("Mt Moon B1F", "Route 4") == "warp (27,3) to Route 4"
    assert route_text("Route 4", "Cerulean City") == "walk east to Cerulean City"
    assert route_text("Route 3", "Route 3") == ""
    assert route_text("Route 3", "Nowhere") == ""


def test_the_nearest_place_to_heal_is_looked_up_not_remembered():
    """A model with no ground truth invented a Poke Center in the wrong city."""

    name, hops = nearest_pokecenter("Route 3")

    # Two hops north-then-warp, not the Pewter one the route came from.
    assert name == "Mt Moon Pokecenter"
    assert hops == 2
    assert nearest_pokecenter("") == ("", 0)


@pytest.mark.parametrize(
    "text,expected",
    [
        # The destination is the end of the journey, not the start of it.
        ("walk out of Route 4 and on to Cerulean City", "Cerulean City"),
        # Longer names win a tie, so a building is never read as its town.
        ("heal at the Cerulean Pokecenter", "Cerulean Pokecenter"),
        # Bounded, so a longer route number is not found inside a shorter one.
        ("head for Route 44", ""),
        ("no map here at all", ""),
    ],
)
def test_the_map_a_goal_is_aiming_at(text: str, expected: str):
    assert mentioned_map(text, exclude=["Route 4"]) == expected


def test_the_next_rung_is_the_one_above_the_highest_reached():
    """Not the lowest unreached one: several rungs are optional.

    A run that walked past the Route 22 rival has not left work behind it, and
    sending it back for that battle is the same wrong instruction as telling it
    to beat a gym leader it has already beaten.
    """

    done = ["EVENT_GOT_STARTER", "EVENT_BEAT_BROCK", "BADGE_BOULDER"]

    reached, upcoming = ladder_position(done)

    assert reached == "Boulder Badge"
    assert upcoming == "Beat the Super Nerd guarding the Mt. Moon fossils"
    assert ladder_position([])[0] == ""
    assert ladder_position(["not-a-milestone"]) == ("", "Chose a starter Pokemon")


# ---------------------------------------------------------------------------
# Claims, checked
# ---------------------------------------------------------------------------


def test_a_coordinate_the_model_wrote_about_is_answered_by_the_map_file():
    """One handoff called the B1F ladders the cave mouth and the Route 4 doors
    ladders. Every coordinate in it was in the world file, saying the opposite."""

    text = "the cave mouth out to Route 4 is (25,9); (27,3) descends to B1F; (99,99) is the wall"

    rows = coordinate_claims(text, "Mt Moon B1F")

    assert "- (25,9) IS a warp on Mt Moon B1F: it leads to Mt Moon 1F." in rows
    assert "- (27,3) IS a warp on Mt Moon B1F: it leads to Route 4." in rows
    assert "- (99,99) is outside Mt Moon B1F, which is 28x28." in rows


def test_a_coordinate_from_another_map_is_named_as_such():
    rows = coordinate_claims("the exit was (3,7)", "Mt Moon B1F", ["Mt Moon Pokecenter"])

    assert rows == [
        "- (3,7) is not a warp on Mt Moon B1F, but it is one on "
        "Mt Moon Pokecenter, leading to Route 4."
    ]
    # Ordinary ground says nothing. A line per tile would bury the ones that matter.
    assert coordinate_claims("standing at (12,12)", "Mt Moon B1F") == []


def test_a_direction_the_model_wrote_is_checked_against_the_graph():
    """The claim that cost this run 5,618 presses, and the one that did not."""

    wrong = direction_claims("head south on Route 3 to Cerulean City", "Mt Moon B1F")
    right = direction_claims("walk east from Route 4 into Cerulean City", "Route 4")

    assert wrong == [
        '- "south ... Cerulean City": WRONG. The map data says warp (27,3) to '
        "Route 4, then walk east to Cerulean City."
    ]
    assert right == [
        '- "east ... Cerulean City": agrees with the map data (walk east to Cerulean City).'
    ]
    assert direction_claims("go west, young man", "Route 3") == []


def test_the_narration_and_the_notes_are_headed_as_claims():
    """`machine INACCESSIBLE (confirmed)` was copied out of NOTES.md into a
    handoff as ground truth. The agent healed at that machine 26 seconds later."""

    digest = build_digest(
        DigestInput(notes="machine INACCESSIBLE (confirmed)", calls=sample_calls())
    )

    assert "CLAIMS, not facts, and unverified" in digest
    assert digest.count("CLAIMS, not facts, and unverified") == 2


# ---------------------------------------------------------------------------
# The goal the next session is started on
# ---------------------------------------------------------------------------


def test_a_goal_that_points_the_wrong_way_is_thrown_away():
    """The critic's goal is not advice: it is the next session's first line, and
    the session acts on it in preference to a correct tool result three seconds
    old. This exact goal cost the run 5,618 presses."""

    goal, problem = check_next_goal(
        "head south on Route 3 to Cerulean City", from_map="Mt Moon B1F"
    )

    assert goal == ""
    assert problem == (
        'says "south" about Cerulean City, but the map graph says warp (27,3) to '
        "Route 4, then walk east to Cerulean City"
    )


def test_a_goal_naming_somewhere_unreachable_is_thrown_away():
    goal, problem = check_next_goal("get to Kalos", from_map="Route 3")
    assert (goal, problem) == ("get to Kalos", "")

    goal, problem = check_next_goal("get to Cinnabar Gym", from_map="Trade Center")
    assert goal == ""
    assert "cannot reach" in problem


def test_a_goal_the_graph_agrees_with_or_cannot_check_is_left_alone():
    routed = "take warp (27,3) to Route 4, then walk east to Cerulean City"
    assert check_next_goal(routed, from_map="Mt Moon B1F") == (routed, "")

    # No map named is nothing to check, and the fallback is not free.
    local = "Press A on every sign in this room"
    assert check_next_goal(local, from_map="Mt Moon B1F") == (local, "")
    assert check_next_goal("", from_map="Mt Moon B1F") == ("", "")
    assert check_next_goal("anything", from_map="") == ("anything", "")


@pytest.mark.asyncio
async def test_the_critic_drops_its_own_goal_when_the_map_graph_refuses_it(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = make_fake_print_pi(
        tmp_path,
        text="You walked in circles.\n\nNEXT GOAL: head south on Route 3 to Cerulean City",
    )

    result = await run_critic(
        pi_binary=str(script),
        workspace_dir=workspace,
        digest="# Finished session digest\n",
        from_map="Mt Moon B1F",
    )

    # The handoff still lands - the prose is worth keeping - but the goal does not
    # reach the next session, and the operator is told exactly why.
    assert result.ok
    assert result.next_goal == ""
    assert "south" in result.next_goal_rejected
    assert "Dropped the NEXT GOAL" in (result.error or "")
    assert read_handoff(workspace).startswith("You walked in circles.")


# ---------------------------------------------------------------------------
# Session intelligence
# ---------------------------------------------------------------------------


def test_the_presses_are_split_into_buckets_the_agent_never_sees(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))
    receipts = RunRegistry(data_dir).load(run_id).receipts

    rows = waste_lines(receipts, since_t=1000.0)

    # One batch onto a new tile, seven more into the wall, then a `goto` that
    # actually went somewhere. Four fifths of the session bought nothing.
    assert rows[0] == "- 352 presses this session: blocked 280 (80%), productive 72 (20%)."
    assert rows[1].startswith("- Route 3: 352 presses, 80% of them revisiting")
    assert waste_lines((), since_t=None) == []


def test_the_commands_it_repeated_are_counted_because_it_never_notices():
    """SKILL.md says stop after three. One session sent thirteen in a row."""

    calls = [action_call(["walk_right"], comment=f"probe {index}") for index in range(13)]
    calls.append(action_call(["walk_up"], comment="try north"))

    rows = repeat_lines(calls)

    assert "`act walk_right` x13" in rows[0]
    assert rows[1] == "- Longest identical run: `act walk_right` 13 times back to back."
    assert repeat_lines([]) == []


def test_sessions_are_compared_so_again_is_a_measurable_word(tmp_path: Path):
    data_dir = tmp_path / "data"
    receipts = [
        {"t": 100.0, "presses": 20, "tool": "action", "map": "Route 3", "pos": [1, 1], "moved": 4},
        {"t": 101.0, "presses": 20, "tool": "action", "map": "Route 3", "pos": [2, 1], "moved": 4},
        # Second session: same ground, twice the buttons, nothing new.
        {"t": 200.0, "presses": 40, "tool": "action", "map": "Route 3", "pos": [1, 1], "moved": 0},
        {"t": 201.0, "presses": 40, "tool": "action", "map": "Route 3", "pos": [2, 1], "moved": 0},
    ]
    run_id = record_run(data_dir, receipts)
    loaded = RunRegistry(data_dir).load(run_id).receipts

    rows = trend_lines(loaded, [(1, 100.0), (2, 200.0)])

    assert rows[0].startswith("- s1: 40 presses, 2 tiles it had never stood on (20.0 presses each)")
    assert "- s2: 80 presses, 0 tiles it had never stood on" in rows[1]
    assert "100% of batches moved nothing" in rows[1]
    # One session is not a comparison.
    assert trend_lines(loaded, [(1, 100.0)]) == []


def test_the_verbs_it_never_called_are_named(tmp_path: Path):
    """`progress` was called zero times across nine sessions; naming `saves` in a
    handoff is what got `./poke load` called for the first time ever."""

    rows = untried_lines([action_call(["walk_up"], comment="north")])

    assert "- Verbs it never called this session: " in rows[0]
    for verb in ("progress", "saves", "load", "goto", "frontier"):
        assert verb in rows[0]
    assert rows[1] == "- Verbs it did call: act x1."


def test_the_explored_store_says_what_is_reachable_and_never_walked(tmp_path: Path):
    store = tmp_path / EXPLORED_STORE_FILENAME
    # A 4x1 corridor of Route 3, two tiles of which were actually walked.
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "current_map_id": 14,
                "maps": {
                    "14": {
                        "map_name": "Route 3",
                        "width": 4,
                        "height": 1,
                        "seen": ["f"],
                        "walkable": ["f"],
                        "walked": ["c"],
                        "visits": {"0,0": 2, "1,0": 1},
                        "player": [1, 0],
                        "warps": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = coverage_lines(
        tmp_path,
        map_name="Route 3",
        map_id=14,
        player=(1, 0),
        warps=[(3, 0, "Route 4")],
    )

    # The store repairs the corridor up to Route 3's real size as it loads it,
    # so the totals are the map's; the counts of what was seen are the store's.
    assert rows[0].startswith("- Route 3 explored: 4 of ")
    assert rows[0].endswith("4 walkable, 2 actually walked on.")
    assert "Reachable from (1,0) and never walked on: 2 tiles" in rows[1]
    assert "(2,0), (3,0)" in rows[1]
    assert rows[2] == "- Warp tiles on this map it never stood on: (3,0)->Route 4."
    assert coverage_lines(None, map_name="Route 3", map_id=14, player=None) == []


def test_intel_is_collected_block_by_block_and_never_raises(tmp_path: Path):
    """A dead block, never a dead critic: this must not be why a session cannot
    start, whatever it is handed."""

    intel = collect_intel(
        data_dir=tmp_path / "nowhere",
        run_id="20990101T000000Z-dead",
        since_t=None,
        facts=None,
        calls=[],
        goal="",
    )

    assert isinstance(intel, Intel)
    assert not intel


def test_the_digest_puts_the_measurements_above_the_models_account(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_id = record_run(data_dir, pocket_receipts(1000.0))
    facts = collect_session_facts(data_dir=data_dir, run_id=run_id, since_t=1000.0)
    intel = collect_intel(
        data_dir=data_dir,
        run_id=run_id,
        since_t=1000.0,
        facts=facts,
        calls=sample_calls(),
        goal="Reach Cerulean City",
        notes="the way out is west to Cerulean City, past the ladder at (99,99)",
        session_starts=[(1, 900.0), (2, 1000.0)],
    )

    digest = build_digest(
        DigestInput(goal="Cross Route 3.", calls=sample_calls(), facts=facts, intel=intel)
    )

    assert "Where it is, from the game's own map data (authoritative)" in digest
    assert "walk north -> Route 4" in digest
    assert (
        "Route to Cerulean City: walk north to Route 4, then walk east to Cerulean City"
    ) in digest
    assert CLAIMS_HEADING in digest
    assert "(99,99) is outside Route 3, which is 70x18." in digest
    assert '"west ... Cerulean City": WRONG.' in digest
    assert digest.index(FACTS_DIGEST_HEADING) < digest.index(CLAIMS_HEADING)
    assert digest.index(CLAIMS_HEADING) < digest.index(NARRATION_HEADING)


def test_the_critic_is_told_it_knows_more_than_the_agent_did():
    prompt = build_prompt("# Finished session digest\n")

    assert "You are given more than the agent had" in prompt
    assert "same thing last session and it did not work" in prompt
    assert "checked against the map graph before it is used" in prompt


# ---------------------------------------------------------------------------
# `handoff_body`: the retrospective minus the goal that travels on its own
#
# `parse_next_goal` lifts the line out, `check_next_goal` rules on it, and the
# supervisor puts the survivor at the top of the first user message. Leaving it
# in the file is right -- it is the post-mortem record -- and sending it a
# second time is wrong three different ways, all of them measured on the live
# run: duplicated when accepted, contradicting when an operator goal is in
# force, and re-delivered when the check threw it away.
# ---------------------------------------------------------------------------


def test_handoff_body_drops_the_goal_line_and_keeps_the_retrospective():
    text = (
        "**Most costly mistake:** 408 `poke sim` calls re-tested the x=20 wall.\n"
        "\n"
        "**Do instead:** `poke goto 20 6`.\n"
        "\n"
        "NEXT GOAL: Walk east from (19,6) to Cerulean City."
    )

    body = handoff_body(text)

    assert "NEXT GOAL" not in body
    assert "408 `poke sim` calls" in body
    assert "`poke goto 20 6`" in body
    # Measured on the live run's HANDOFF.md: 1,094 bytes down to 1,008.
    assert len(body) < len(text)


def test_handoff_body_drops_a_goal_written_under_its_own_label():
    """`parse_next_goal` accepts the split form, so this has to drop both lines.

    Dropping the label alone would leave the goal behind as a bare imperative
    sentence at the end of the retrospective, which is the same instruction with
    nothing left to mark it as one.
    """
    text = "You re-swept Mt Moon B1F.\n\nNEXT GOAL:\nWalk east to Cerulean City."

    body = handoff_body(text)

    assert body == "You re-swept Mt Moon B1F."


def test_handoff_body_leaves_a_retrospective_that_never_named_a_goal_alone():
    text = "**Most costly mistake:** you re-swept B1F.\n\n**Do instead:** take (27,3)."

    assert handoff_body(text) == text


def test_handoff_body_keeps_the_truncation_marker():
    """A shortened handoff has to keep saying that it was shortened."""
    text = "The tail of some reasoning.\n\n_[truncated: 8593 earlier words not shown]_"

    assert "_[truncated" in handoff_body(text)


def test_handoff_body_survives_an_empty_handoff():
    assert handoff_body("") == ""
    assert handoff_body("NEXT GOAL: Walk east.") == ""


def test_the_exits_line_prints_the_star_legend_only_when_a_star_is_there():
    """22 bytes teaching a notation the sentence next to it does not use.

    Measured on the live run's session 6: "Every way off Route 3 (* = never
    stepped on): walk north -> Route 4; walk west -> Pewter City." Nothing was
    starred, and the legend was paid for anyway, once per session.
    """
    facts = SessionFacts(exits="walk north -> Route 4", ended_map="Route 3")
    plain = "\n".join(facts.lines())

    starred = SessionFacts(exits="walk north -> Route 4*", ended_map="Route 3")
    marked = "\n".join(starred.lines())

    assert "Every way off Route 3: walk north -> Route 4." in plain
    assert "never stepped on" not in plain
    assert "Every way off Route 3 (* = never stepped on): walk north -> Route 4*." in marked


# ---------------------------------------------------------------------------
# What the next session is allowed to read
# ---------------------------------------------------------------------------


def test_a_salvaged_reasoning_tail_is_not_handed_to_the_next_session(tmp_path):
    """Measured live: 1,648 bytes of the critic counting words at itself.

    "Most(1) costly(2) mistake(3)..." arrived as the retrospective in a first
    message of about 2,100 bytes, so most of a new session's handoff was a
    transcript of the critic failing to write one. It stays in HANDOFF.md for
    the post-mortem; it does not go to the model. The deterministic ground-truth
    block beside it already carries the run's real facts.
    """
    write_handoff(
        tmp_path,
        SALVAGED_REASONING_NOTICE + "\n\nRecount the final version. Most(1) costly(2)",
    )

    assert read_handoff(tmp_path) == ""
    assert "Most(1)" in handoff_path(tmp_path).read_text(encoding="utf-8")


def test_an_ordinary_retrospective_still_comes_through(tmp_path):
    write_handoff(tmp_path, "Walked into the Route 4 wall again. Take the Mt Moon route.")

    assert "Route 4 wall" in read_handoff(tmp_path)


def test_the_word_ceiling_applies_when_reading_not_only_when_writing(tmp_path):
    """The cap lived only on the write path, so anything hand-edited went in whole."""
    handoff_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    handoff_path(tmp_path).write_text(
        " ".join(f"word{n}" for n in range(MAX_HANDOFF_WORDS * 3)), encoding="utf-8"
    )

    got = read_handoff(tmp_path)

    # cap_words appends its own truncation notice, which is words too.
    assert len(got.split()) <= MAX_HANDOFF_WORDS + 20, "the ceiling is applied on read"


# The body of the handoff, not only its goal line
# ---------------------------------------------------------------------------


def test_a_sentence_the_map_data_contradicts_is_struck_from_the_handoff():
    """`check_next_goal` gatekeeps one line; the body was delivered verbatim.

    The body becomes the next session's first user message, which is to say a
    model reads it as ground truth and then spends a session acting on it. The
    same map data that rejects a goal rejects a sentence.
    """

    text = (
        "You spent 61% of the session in battle. "
        "Exit Mt Moon 1F west to Route 2 and walk to Viridian City to heal.\n"
        "- The Ember PP ran out at batch 40."
    )

    body, struck = strike_false_claims(text, from_map="Mt Moon 1F")

    assert "61% of the session in battle" in body
    assert "Ember PP ran out" in body
    assert "Viridian City" not in body
    assert struck == ["Exit Mt Moon 1F west to Route 2 and walk to Viridian City to heal."]


def test_a_handoff_the_map_data_agrees_with_is_written_untouched():
    text = "Take warp (27,3) to Route 4, then walk east to Cerulean City."
    assert strike_false_claims(text, from_map="Mt Moon B1F") == (text, [])


def test_naming_the_run_s_destination_is_not_a_false_claim():
    """No hop ceiling on a handoff.

    The ceiling in `check_advice` is for a message steering the next few hundred
    presses, where naming somewhere eight warps off is the tell. A retrospective
    naming the run's destination is doing its job.
    """

    text = "The run is heading for Cerulean City and is four maps short of it."
    assert strike_false_claims(text, from_map="Pallet Town") == (text, [])


def test_nothing_to_check_against_leaves_the_handoff_alone():
    text = "Walk west to Cerulean City."
    assert strike_false_claims(text, from_map="") == (text, [])
    assert strike_false_claims("", from_map="Route 4") == ("", [])


def test_the_open_milestones_are_the_dag_frontier_not_one_guessed_rung():
    """The ladder is a line and the game is not.

    On the session this was added for, the run was standing in Mt Moon with the
    Boulder Badge won. `ladder_position` answered "Beat the rival on Route 22" —
    four maps behind it, optional, and already walked past. The DAG knows what
    is actually open, including the floor the run was standing on.
    """

    done = ["EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK", "BADGE_BOULDER"]

    open_now = milestone_frontier(done)

    assert "Beat the Super Nerd guarding the Mt. Moon fossils" in open_now
    assert len(open_now) > 1, "a frontier of one is a ladder with extra steps"
    assert milestone_frontier([]) == ("Chose a starter Pokemon",)


def test_the_frontier_reaches_the_digest_the_critic_reads():
    facts = SessionFacts(
        run_id="r",
        session_index=1,
        done=("BADGE_BOULDER",),
        done_count=1,
        frontier=("Beat the Super Nerd guarding the Mt. Moon fossils", "Defeated Misty"),
    )

    rendered = "\n".join(facts.lines())

    assert "Open now (every milestone whose prerequisites are already met)" in rendered
    assert "Defeated Misty" in rendered
