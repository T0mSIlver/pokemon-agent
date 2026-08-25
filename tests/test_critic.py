import json
from pathlib import Path

import pytest

from pokemon_agent.critic import (
    CRITIC_RAW_FILENAME,
    CRITIC_RAW_PREVIOUS_FILENAME,
    DEBUG_DIRNAME,
    DIGEST_CHAR_BUDGET,
    HANDOFF_FILENAME,
    HANDOFF_PREVIOUS_FILENAME,
    MAX_HANDOFF_WORDS,
    NEXT_GOAL_LABEL,
    NO_TEXT_ERROR,
    SALVAGED_REASONING_NOTICE,
    TARGET_HANDOFF_WORDS,
    CriticResult,
    DigestInput,
    ToolCall,
    build_critic_command,
    build_digest,
    build_prompt,
    call_lines,
    cap_words,
    classify_call,
    compute_behaviour_stats,
    describe_no_text,
    estimate_tokens,
    narration_lines,
    parse_actions,
    parse_final_text,
    parse_next_goal,
    read_handoff,
    run_critic,
    tail_words,
    tool_calls_from_stream,
    write_handoff,
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
    assert estimate_tokens(digest) <= 12_000
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
    # Comfortably past the old 300-word cap, comfortably inside the new one.
    body = "\n\n".join(
        f"{index}. Mistake {index}: you rammed the wall at (5,6) {index} times."
        for index in range(60)
    )
    critique = f"{body}\n\nNEXT GOAL: Take the west path out of Viridian Forest."
    assert 300 < len(critique.split()) < MAX_HANDOFF_WORDS
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
    handoff = read_handoff(workspace)
    assert handoff.startswith(SALVAGED_REASONING_NOTICE)
    # The tail of the reasoning is the part that was still being written.
    assert "thought1999" in handoff
    assert "thought0." not in handoff
    assert "truncated" in handoff
    assert len(handoff.split()) <= MAX_HANDOFF_WORDS
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
