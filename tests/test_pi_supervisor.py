import asyncio
import base64
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from pokemon_agent import pi_supervisor as pi_supervisor_module
from pokemon_agent.bench.registry import STATUS_FINISHED, RunRegistry
from pokemon_agent.critic import (
    CRITIC_RAW_FILENAME,
    DEBUG_DIRNAME,
    DIGEST_CHAR_BUDGET,
    FACTS_DIGEST_HEADING,
    FACTS_HEADING,
    HANDOFF_FILENAME,
    HANDOFF_HEADING,
    HANDOFF_PREVIOUS_FILENAME,
    HANDOFF_STALE_FILENAME,
    MAX_HANDOFF_WORDS,
    SALVAGED_REASONING_NOTICE,
    handoff_body,
    handoff_path,
    read_handoff,
    write_handoff,
)
from pokemon_agent.pi_supervisor import (
    CONTINUE_MESSAGE,
    CRITIC_STREAM_PREFIX,
    FALLBACK_GOAL,
    GOAL_SOURCE_CRITIC,
    GOAL_SOURCE_FALLBACK,
    GOAL_SOURCE_OBJECTIVE,
    GOAL_SOURCE_OPERATOR,
    INTERVENTION_MESSAGE_LIMIT,
    INTERVENTION_STREAM_SOURCE,
    OPERATOR_MESSAGE_LIMIT,
    OPERATOR_STREAM_SOURCE,
    ORPHAN_TOOL_RESULT_TEXT,
    STREAM_ENTRY_CAP,
    TOOL_RESULT_LIMIT,
    NoLiveSessionError,
    PiSupervisor,
    command_headline,
    find_orphaned_tool_calls,
    iter_jsonl_records,
    parse_model_limits_output,
    payload_text,
    repair_orphaned_tool_calls,
)
from pokemon_agent.run_recorder import RUN_POINTER_FILENAME, RunRecorder

#: The real thing, captured before the autouse fixture below stubs it out, so
#: the two tests that are about the workspace interpreter can still call it.
STAGE_WORKSPACE_VENV = PiSupervisor._stage_workspace_venv


@pytest.fixture(autouse=True)
def no_real_workspace_venv(monkeypatch):
    """Staging builds a real venv; no test in this file wants one.

    Every session start stages the workspace, and a venv with Pillow and numpy
    in it is a hundred megabytes and several seconds. Multiplied by the starts
    in this file that filled the disk the suite runs on.
    """
    monkeypatch.setattr(PiSupervisor, "_stage_workspace_venv", lambda self: None)


FAKE_RPC_SERVER = '''#!/usr/bin/env python3
"""Minimal stand-in for `pi --mode rpc`: JSON commands in, JSON events out."""

import json
import pathlib
import sys
import time

PARAMS = __PARAMS__

argv = sys.argv[1:]


def opt(name):
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


if "--list-models" in argv:
    provider = opt("--provider") or "llamacpp"
    model = opt("--list-models") or "fake-model"
    print("provider  model  context  max-out  thinking  images")
    print(provider + "  " + model + "  262.1K   65.5K    yes       yes")
    raise SystemExit(0)

if "--print" in argv:
    # The between-session critic: one-shot print mode, no session directory.
    pathlib.Path.cwd().joinpath("critic_argv.json").write_text(
        json.dumps(argv), encoding="utf-8"
    )
    with pathlib.Path.cwd().joinpath("critic_calls.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(argv) + "\\n")
    critique = PARAMS["critique"]
    if critique is None:
        sys.stderr.write("no critique configured\\n")
        raise SystemExit(3)
    time.sleep(PARAMS["critique_delay"])

    def emit_print(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    def print_deltas(kind, body):
        for index in range(0, len(body), 8):
            emit_print(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": kind, "delta": body[index : index + 8]},
                }
            )

    emit_print({"type": "agent_start"})
    reasoning = PARAMS["critique_thinking"]
    print_deltas("thinking_delta", reasoning)
    print_deltas("text_delta", critique)
    content = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    if critique:
        content.append({"type": "text", "text": critique})
    if content or PARAMS["critique_stop_reason"] or PARAMS["critique_usage"]:
        message = {"role": "assistant", "content": content}
        if PARAMS["critique_stop_reason"]:
            message["stopReason"] = PARAMS["critique_stop_reason"]
        if PARAMS["critique_usage"]:
            message["usage"] = PARAMS["critique_usage"]
        emit_print({"type": "message_end", "message": message})
    emit_print({"type": "agent_settled"})
    raise SystemExit(0)

session_dir = pathlib.Path(opt("--session-dir"))
session_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = session_dir.parent
session_id = opt("--session-id") or "session-fake"
session_file = session_dir / (session_id + ".jsonl")
command_log = workspace_dir / "rpc_commands.jsonl"

if not session_file.exists():
    session_file.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": "2026-01-01T00:00:00Z",
                "cwd": str(workspace_dir),
            }
        )
        + "\\n",
        encoding="utf-8",
    )

state = {"auto_compaction": True, "turn": 0, "entry": 0}


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


def log_command(command):
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(command) + "\\n")


def append_entry(message):
    state["entry"] += 1
    entry_id = "entry-%04d" % state["entry"]
    with session_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "message",
                    "id": entry_id,
                    "parentId": None,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": message,
                }
            )
            + "\\n"
        )


def run_turn():
    state["turn"] += 1
    text = "Initial turn." if state["turn"] == 1 else "Continued turn."
    emit({"type": "agent_start"})
    emit({"type": "turn_start"})
    emit({"type": "message_start", "message": {"role": "assistant", "content": []}})
    emit(
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "thinking_delta",
                "delta": "Inspecting the frame.",
            },
        }
    )

    if PARAMS["hang_mid_tool"]:
        append_entry(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tool-orphan",
                        "name": "bash",
                        "arguments": {"command": "sleep 600"},
                    }
                ],
                "stopReason": "toolUse",
            }
        )
        emit(
            {
                "type": "tool_execution_start",
                "toolCallId": "tool-orphan",
                "toolName": "bash",
                "args": {"command": "sleep 600"},
            }
        )
        while True:
            time.sleep(1)

    if PARAMS["orphan_tool_call"]:
        append_entry(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tool-orphan",
                        "name": "bash",
                        "arguments": {"command": "echo hi"},
                    }
                ],
                "stopReason": "toolUse",
            }
        )

    if PARAMS["action_tool"]:
        body = json.dumps({"actions": ["walk_up", "walk_up"]})
        command = "# walk north\\ncurl -sS -X POST http://localhost:$PORT/action -d " + repr(body)
        response = json.dumps({"x": 5, "y": 6, "facing": "up", "moved": 0, "blocked_after": 1})
        emit(
            {
                "type": "tool_execution_start",
                "toolCallId": "tool-action-%d" % state["turn"],
                "toolName": "bash",
                "args": {"command": command},
            }
        )
        emit(
            {
                "type": "tool_execution_end",
                "toolCallId": "tool-action-%d" % state["turn"],
                "toolName": "bash",
                "result": {"output": response},
                "isError": False,
            }
        )

    if PARAMS["include_write_tool"]:
        plan = {
            "objective_id": "get_oaks_parcel",
            "summary": "Move north carefully.",
            "planned_actions": ["walk_up"],
            "fallback_actions": ["walk_left"],
        }
        emit(
            {
                "type": "tool_execution_start",
                "toolCallId": "tool-%d" % state["turn"],
                "toolName": "write",
                "args": {
                    "path": str(workspace_dir / "turn_plan.json"),
                    "content": json.dumps(plan),
                },
            }
        )
        emit(
            {
                "type": "tool_execution_end",
                "toolCallId": "tool-%d" % state["turn"],
                "toolName": "write",
                "result": {"ok": True},
                "isError": False,
            }
        )

    emit(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": text},
        }
    )
    emit(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspecting the frame."},
                    {"type": "text", "text": text},
                ],
                "usage": {
                    "input": 3200,
                    "output": 260,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 3460,
                },
            },
        }
    )
    emit(
        {
            "type": "turn_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "toolResults": [],
        }
    )
    emit({"type": "agent_end", "messages": [], "willRetry": True})
    emit({"type": "agent_end", "messages": [], "willRetry": False})
    if PARAMS["settle"]:
        emit({"type": "agent_settled"})


while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    command = json.loads(line)
    log_command(command)
    kind = command.get("type")
    request_id = command.get("id")

    if kind == "set_auto_compaction":
        state["auto_compaction"] = bool(command.get("enabled"))
        emit(
            {
                "id": request_id,
                "type": "response",
                "command": "set_auto_compaction",
                "success": True,
            }
        )
        continue

    if kind == "get_session_stats":
        emit(
            {
                "id": request_id,
                "type": "response",
                "command": "get_session_stats",
                "success": True,
                "data": {
                    "sessionFile": str(session_file),
                    "sessionId": session_id,
                    "tokens": {
                        "input": 3200,
                        "output": 260,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "total": 3460,
                    },
                    "cost": 0.0,
                    "contextUsage": {
                        "tokens": PARAMS["context_tokens"],
                        "contextWindow": 262144,
                        "percent": 2,
                    },
                },
            }
        )
        continue

    if kind == "get_state":
        emit(
            {
                "id": request_id,
                "type": "response",
                "command": "get_state",
                "success": True,
                "data": {
                    "isStreaming": PARAMS["is_streaming"],
                    "messageCount": state["turn"],
                },
            }
        )
        continue

    if kind == "prompt" and command.get("streamingBehavior"):
        # A queued message: accepted for the running agent, no new turn here.
        emit({"id": request_id, "type": "response", "command": "prompt", "success": True})
        continue

    if kind == "prompt":
        emit({"id": request_id, "type": "response", "command": "prompt", "success": True})
        run_turn()
        continue

    emit({"id": request_id, "type": "response", "command": kind, "success": True})
'''


def make_fake_rpc_server(
    tmp_path: Path,
    *,
    include_write_tool: bool = True,
    hang_mid_tool: bool = False,
    context_tokens: int = 4200,
    settle: bool = True,
    orphan_tool_call: bool = False,
    action_tool: bool = False,
    is_streaming: bool = True,
    critique: Optional[str] = None,
    critique_delay: float = 0.0,
    critique_thinking: str = "",
    critique_stop_reason: str = "",
    critique_usage: Optional[dict] = None,
) -> Path:
    """Build a stand-in RPC server.

    ``settle=False`` reproduces the long single turn seen in real runs: the fake
    streams a whole turn, never emits ``agent_settled``, and keeps servicing
    stdin so ``get_session_stats`` still answers mid-turn.

    ``critique`` is what the same binary answers in ``--print`` mode, where the
    supervisor runs its between-session critic: ``None`` fails with status 3,
    ``""`` returns an empty stream, and any text is the retrospective.
    """

    script = tmp_path / "fake-pi-rpc"
    params = {
        "include_write_tool": include_write_tool,
        "hang_mid_tool": hang_mid_tool,
        "context_tokens": context_tokens,
        "settle": settle,
        "orphan_tool_call": orphan_tool_call,
        "action_tool": action_tool,
        "is_streaming": is_streaming,
        "critique": critique,
        "critique_delay": critique_delay,
        "critique_thinking": critique_thinking,
        "critique_stop_reason": critique_stop_reason,
        "critique_usage": critique_usage or {},
    }
    script.write_text(
        FAKE_RPC_SERVER.replace("__PARAMS__", repr(params)),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def read_commands(workspace_dir: Path) -> list[dict]:
    log = workspace_dir / "rpc_commands.jsonl"
    if not log.is_file():
        return []
    return iter_jsonl_records(log.read_text(encoding="utf-8"))


def prompt_commands(workspace_dir: Path) -> list[dict]:
    return [command for command in read_commands(workspace_dir) if command.get("type") == "prompt"]


def write_frames(workspace_dir: Path) -> tuple[Path, Path]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    annotated = workspace_dir / "latest_frame_annotated.png"
    raw = workspace_dir / "latest_frame.png"
    annotated.write_bytes(b"annotated-frame")
    raw.write_bytes(b"raw-frame")
    return annotated, raw


async def wait_for(predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_launch_command_uses_rpc_mode_flags(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        provider="llamacpp",
        model="fake-model",
        thinking="medium",
        auto_continue=False,
    )
    await supervisor.wait_until_idle(timeout=10)

    launch = next(event for event in events if event["type"] == "pi_session_launch")
    snapshot = supervisor.state_snapshot()
    assert launch["command_preview"] == [
        str(fake_pi),
        "--mode",
        "rpc",
        "--system-prompt",
        str(supervisor.skill_path),
        "--tools",
        "read,bash,edit,write",
        "--session-id",
        snapshot["session_id"],
        "--session-dir",
        str(workspace_dir.resolve() / "pi-session"),
        "-ne",
        "-ns",
        "-nc",
        "-np",
        "--no-themes",
        "--offline",
        "--provider",
        "llamacpp",
        "--model",
        "fake-model",
        "--thinking",
        "medium",
    ]
    assert "--mode json" not in " ".join(launch["command_preview"])
    assert "--print" not in launch["command_preview"]
    assert snapshot["model_limits"]["context_window_tokens"] == 262100


@pytest.mark.asyncio
async def test_auto_compaction_is_disabled_before_the_first_prompt(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    commands = read_commands(workspace_dir)
    assert commands[0]["type"] == "set_auto_compaction"
    assert commands[0]["enabled"] is False
    assert commands[1]["type"] == "prompt"


@pytest.mark.asyncio
async def test_initial_prompt_carries_goal_text_and_both_frames(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    annotated, raw = write_frames(workspace_dir)

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        stream_sink=stream,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    prompts = prompt_commands(workspace_dir)
    assert len(prompts) == 1
    assert prompts[0]["message"] == "Reach the next checkpoint."
    images = prompts[0]["images"]
    assert [image["mimeType"] for image in images] == ["image/png", "image/png"]
    assert images[0]["type"] == "image"
    assert images[0]["data"] == base64.b64encode(annotated.read_bytes()).decode("ascii")
    assert images[1]["data"] == base64.b64encode(raw.read_bytes()).decode("ascii")

    prompt_event = next(event for event in streamed if event["type"] == "pi_prompt_sent")
    assert prompt_event["attachments"] == [str(annotated), str(raw)]


@pytest.mark.asyncio
async def test_follow_up_prompt_is_bare_continue_without_images(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    write_frames(workspace_dir)

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        stream_sink=stream,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        max_turns=3,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=15)

    prompts = prompt_commands(workspace_dir)
    assert len(prompts) == 3
    assert prompts[1]["message"] == CONTINUE_MESSAGE == "continue"
    assert "images" not in prompts[1]
    assert "images" not in prompts[2]

    prompt_events = [event for event in streamed if event["type"] == "pi_prompt_sent"]
    assert prompt_events[1]["attachments"] == []
    assert prompt_events[1]["resume"] is True

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 3
    assert snapshot["continue_count"] == 2


@pytest.mark.asyncio
async def test_one_turn_completes_per_agent_settled_not_per_agent_end(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    # The fake emits two agent_end events (one with willRetry) before a single settle.
    assert len([event for event in events if event["type"] == "pi_agent_end"]) == 2
    assert len([event for event in events if event["type"] == "pi_agent_settled"]) == 1
    assert len([event for event in events if event["type"] == "pi_turn_end"]) == 1
    assert supervisor.state_snapshot()["turns_completed"] == 1


@pytest.mark.asyncio
async def test_transcript_tool_and_usage_telemetry(tmp_path: Path):
    events: list[dict] = []
    streamed: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)

    async def sink(event: dict) -> None:
        events.append(event)

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        stream_sink=stream,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 1
    assert snapshot["goal"] == "Reach the next checkpoint."
    assert snapshot["last_assistant_text"] == "Initial turn."
    assert "Inspecting the frame." in snapshot["last_assistant_thinking"]
    assert snapshot["turn_plan_preview"]["payload"]["objective_id"] == "get_oaks_parcel"
    assert snapshot["recent_tools"][-1]["tool_name"] == "write"
    assert "get_oaks_parcel" in snapshot["recent_tools"][-1]["args"]
    assert '"ok": true' in snapshot["recent_tools"][-1]["result"]
    assert any(
        entry["direction"] == "outbound" and entry["role"] == "user"
        for entry in snapshot["transcript"]
    )
    assert any(
        entry["channel"] == "assistant" and entry["role"] == "assistant"
        for entry in snapshot["transcript"]
    )

    counts = snapshot["counts"]
    assert counts["tool_calls"] == 1
    assert counts["thinking_blocks"] == 1
    assert counts["assistant_messages"] == 1
    assert counts["user_messages"] == 0

    assert snapshot["last_message_usage"]["output"] == 260
    assert snapshot["session_usage"]["input"] == 3200
    assert snapshot["context_usage"]["tokens"] == 4200
    assert snapshot["context_usage"]["contextWindow"] == 262144
    assert snapshot["session_usage"]["totalTokens"] == 4200

    assert snapshot["stderr_tail"] == []
    assert any(event["type"] == "pi_turn_end" for event in events)
    assert any(event["type"] == "pi_text_delta" for event in streamed)
    assert any(event["type"] == "pi_prompt_sent" for event in streamed)


@pytest.mark.asyncio
async def test_supervisor_retains_full_tool_history_for_session(tmp_path: Path):
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
    )

    for index in range(30):
        tool_call_id = f"tool-{index}"
        await supervisor._handle_event(
            {
                "type": "tool_execution_start",
                "toolCallId": tool_call_id,
                "toolName": "bash",
                "args": {"command": f"echo {index}"},
            }
        )
        await supervisor._handle_event(
            {
                "type": "tool_execution_end",
                "toolCallId": tool_call_id,
                "toolName": "bash",
                "result": {"ok": True, "index": index},
                "isError": False,
            }
        )

    snapshot = supervisor.state_snapshot()
    assert snapshot["counts"]["tool_calls"] == 30
    assert len(snapshot["recent_tools"]) == 30
    assert snapshot["recent_tools"][0]["tool_call_id"] == "tool-0"
    assert snapshot["recent_tools"][-1]["tool_call_id"] == "tool-29"


@pytest.mark.asyncio
async def test_supervisor_stages_poke_cli_in_workspace(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    # A workspace outlives a session, so the helpers the CLI replaced can still
    # be sitting in it from an older run.
    for stale in ("agent_curl.sh", "act"):
        (workspace_dir / stale).write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    # `poke` takes bare action arguments. Hand-built curl JSON was losing ~40% of
    # the agent's actions to a dropped closing quote; this form has nothing to
    # misquote, so it must always be staged and executable.
    poke = workspace_dir / "poke"
    assert poke.is_file()
    assert poke.stat().st_mode & 0o111, "poke must be executable"
    assert "def cmd_act(" in poke.read_text(encoding="utf-8")

    # Leaving the old helpers behind would let the quoting failure come back.
    assert not (workspace_dir / "agent_curl.sh").exists()
    assert not (workspace_dir / "act").exists()


@pytest.mark.asyncio
async def test_supervisor_stops_auto_continue_after_idle_turns(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, include_write_tool=False)

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
        max_idle_turns=2,
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=15)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "stuck"
    assert "no tool calls" in snapshot["status_reason"]
    assert snapshot["turns_completed"] == 2
    assert snapshot["counts"]["tool_calls"] == 0
    stuck_event = next(event for event in events if event["type"] == "pi_supervisor_stuck")
    assert stuck_event["idle_turns"] == 2


@pytest.mark.asyncio
async def test_token_budget_ends_the_run(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, context_tokens=120_000)

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
        token_budget=110_000,
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=10)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert "Token budget reached" in snapshot["status_reason"]
    assert snapshot["turns_completed"] == 1
    assert any(event["type"] == "pi_token_budget_reached" for event in events)


@pytest.mark.asyncio
async def test_objective_callback_ends_the_run(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path)

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )
    supervisor.set_objective_complete(lambda: True)

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=10)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["status_reason"] == "Objective reported complete."
    assert snapshot["turns_completed"] == 1
    assert any(event["type"] == "pi_objective_complete" for event in events)


async def live_supervisor(tmp_path: Path, **kwargs) -> PiSupervisor:
    """A supervisor parked mid-turn: the fake streams a turn and never settles."""

    fake_pi = make_fake_rpc_server(tmp_path, settle=False, **kwargs)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
        stats_poll_seconds=0,
    )
    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    assert await wait_for(lambda: supervisor.last_assistant_text == "Initial turn.")
    assert supervisor.state_snapshot()["status"] == "running"
    return supervisor


def operator_entries(supervisor: PiSupervisor) -> list[dict]:
    return [
        entry
        for entry in supervisor.stream_entries
        if entry["kind"] == "user" and entry["source"] == OPERATOR_STREAM_SOURCE
    ]


@pytest.mark.asyncio
async def test_operator_message_steers_the_live_session(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path)
    workspace_dir = tmp_path / "workspace"

    record = await supervisor.send_operator_message("  The ledge is one tile left.  ")

    steer = prompt_commands(workspace_dir)[-1]
    assert steer["message"] == "The ledge is one tile left."
    assert steer["streamingBehavior"] == "steer"
    # The opening goal prompt is untouched: the steer is a second, separate send.
    assert len(prompt_commands(workspace_dir)) == 2

    assert record["text"] == "The ledge is one tile left."
    assert record["streaming_behavior"] == "steer"
    assert record["ts"]

    entries = operator_entries(supervisor)
    assert len(entries) == 1
    assert entries[0]["text"] == "The ledge is one tile left."
    assert entries[0]["seq"] == record["seq"]
    # The harness's own prompt sits in the same log but is not marked operator.
    harness_prompts = [
        entry
        for entry in supervisor.stream_entries
        if entry["kind"] == "user" and entry["source"] == "agent"
    ]
    assert harness_prompts
    assert entries[0]["seq"] > harness_prompts[0]["seq"]

    snapshot = supervisor.state_snapshot()
    assert snapshot["operator_messages"] == [record]

    await supervisor.stop()


@pytest.mark.asyncio
async def test_operator_message_falls_back_to_follow_up_when_pi_is_not_streaming(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path, is_streaming=False)
    workspace_dir = tmp_path / "workspace"

    record = await supervisor.send_operator_message("Heal at the centre first.")

    assert prompt_commands(workspace_dir)[-1]["streamingBehavior"] == "followUp"
    assert record["streaming_behavior"] == "followUp"

    await supervisor.stop()


@pytest.mark.asyncio
async def test_operator_message_is_streamed_to_the_dashboard(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, settle=False)

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        stream_sink=stream,
        pi_binary=str(fake_pi),
        stats_poll_seconds=0,
    )
    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    assert await wait_for(lambda: supervisor.state_snapshot()["status"] == "running")

    await supervisor.send_operator_message("Talk to the guard.")

    pushed = [
        event["entry"]
        for event in streamed
        if event.get("type") == "pi_stream_entry"
        and event["entry"]["source"] == OPERATOR_STREAM_SOURCE
    ]
    assert [entry["text"] for entry in pushed] == ["Talk to the guard."]

    await supervisor.stop()


@pytest.mark.asyncio
async def test_operator_message_is_refused_without_a_live_session(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    with pytest.raises(NoLiveSessionError) as idle:
        await supervisor.send_operator_message("Go north.")
    assert "idle" in str(idle.value)
    assert supervisor.stream_entries == []
    assert supervisor.state_snapshot()["operator_messages"] == []

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)
    await supervisor.stop()

    with pytest.raises(NoLiveSessionError) as stopped:
        await supervisor.send_operator_message("Go north.")
    assert "steer" in str(stopped.value)
    assert operator_entries(supervisor) == []


@pytest.mark.asyncio
async def test_operator_message_is_refused_during_the_critique(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )
    supervisor.status = "critiquing"

    with pytest.raises(NoLiveSessionError) as refused:
        await supervisor.send_operator_message("Go north.")
    assert "critique" in str(refused.value)


@pytest.mark.asyncio
async def test_operator_message_rejects_empty_and_over_long_input(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path)
    workspace_dir = tmp_path / "workspace"
    before = len(prompt_commands(workspace_dir))

    for blank in ("", "   ", "\n\t "):
        with pytest.raises(ValueError, match="empty"):
            await supervisor.send_operator_message(blank)

    with pytest.raises(ValueError, match=str(OPERATOR_MESSAGE_LIMIT)):
        await supervisor.send_operator_message("x" * (OPERATOR_MESSAGE_LIMIT + 1))

    # A message exactly at the cap is fine.
    await supervisor.send_operator_message("y" * OPERATOR_MESSAGE_LIMIT)

    assert len(prompt_commands(workspace_dir)) == before + 1
    assert [entry["text"] for entry in operator_entries(supervisor)] == [
        "y" * OPERATOR_MESSAGE_LIMIT
    ]

    await supervisor.stop()


@pytest.mark.asyncio
async def test_operator_message_history_is_capped_and_ordered(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path)

    for index in range(3):
        await supervisor.send_operator_message(f"note {index}")

    history = supervisor.state_snapshot()["operator_messages"]
    assert [record["text"] for record in history] == ["note 0", "note 1", "note 2"]
    assert all(record["ts"] for record in history)
    assert [record["seq"] for record in history] == sorted(record["seq"] for record in history)

    await supervisor.stop()


@pytest.mark.asyncio
async def test_stats_poll_reports_usage_mid_turn_without_a_settle(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, settle=False)
    workspace_dir = tmp_path / "workspace"

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        stream_sink=stream,
        pi_binary=str(fake_pi),
        stats_poll_seconds=0.05,
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    assert await wait_for(lambda: supervisor.state_snapshot()["context_usage"] is not None)

    snapshot = supervisor.state_snapshot()
    # Still mid-turn: no settle has arrived, so the old post-settle refresh never ran.
    assert snapshot["status"] == "running"
    assert snapshot["turns_completed"] == 0
    assert snapshot["context_usage"]["tokens"] == 4200
    assert snapshot["context_usage"]["contextWindow"] == 262144
    assert snapshot["session_usage"]["input"] == 3200
    assert snapshot["session_usage"]["totalTokens"] == 4200
    assert snapshot["config"]["stats_poll_seconds"] == 0.05

    stats_commands = [
        command
        for command in read_commands(workspace_dir)
        if command.get("type") == "get_session_stats"
    ]
    assert stats_commands
    # Poll requests must not collide with the in-flight prompt's request id.
    stats_ids = {command["id"] for command in stats_commands}
    prompt_ids = {command["id"] for command in prompt_commands(workspace_dir)}
    assert stats_ids.isdisjoint(prompt_ids)
    assert len(stats_ids) == len(stats_commands)

    assert any(event["type"] == "pi_session_stats" for event in streamed)

    await supervisor.stop()


@pytest.mark.asyncio
async def test_stats_poll_enforces_the_token_budget_mid_turn(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(
        tmp_path,
        settle=False,
        orphan_tool_call=True,
        context_tokens=134_871,
    )
    workspace_dir = tmp_path / "workspace"

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
        token_budget=110_000,
        stats_poll_seconds=0.05,
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=15)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["status_reason"] == "Token budget reached (134871/110000 context tokens)."
    # The turn never settled, so the run stopped without completing it.
    assert snapshot["turns_completed"] == 0
    assert len(prompt_commands(workspace_dir)) == 1
    assert supervisor._stats_task is None

    budget_event = next(event for event in events if event["type"] == "pi_token_budget_reached")
    assert budget_event["token_budget"] == 110_000
    assert budget_event["context_usage"]["tokens"] == 134_871

    # The mid-turn stop still repairs the tool call the kill left open.
    assert snapshot["counts"]["repaired_tool_calls"] == 1
    entries = iter_jsonl_records(Path(snapshot["session_file"]).read_text(encoding="utf-8"))
    assert find_orphaned_tool_calls(entries) == []
    assert entries[-1]["message"]["toolCallId"] == "tool-orphan"


@pytest.mark.asyncio
async def test_stats_poll_task_is_cancelled_on_shutdown(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, settle=False)
    workspace_dir = tmp_path / "workspace"
    loop = asyncio.get_running_loop()
    reported: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))

    try:
        supervisor = PiSupervisor(
            workspace_dir=workspace_dir,
            server_url="http://127.0.0.1:8765",
            pi_binary=str(fake_pi),
            stats_poll_seconds=0.05,
        )

        await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
        assert await wait_for(lambda: supervisor.state_snapshot()["context_usage"] is not None)

        poll_task = supervisor._stats_task
        assert poll_task is not None
        assert not poll_task.done()

        snapshot = await supervisor.stop()
        assert snapshot["status"] == "stopped"
        assert supervisor._stats_task is None
        assert poll_task.done()
        assert supervisor._pending_responses == {}

        gc.collect()
        await asyncio.sleep(0.05)
        assert reported == []
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("interval", [0, None])
@pytest.mark.asyncio
async def test_stats_poll_is_disabled_when_the_interval_is_falsy(tmp_path: Path, interval):
    fake_pi = make_fake_rpc_server(tmp_path, settle=False)
    workspace_dir = tmp_path / "workspace"

    assert (
        PiSupervisor(
            workspace_dir=tmp_path / "default-workspace",
            server_url="http://127.0.0.1:8765",
        ).stats_poll_seconds
        == 30.0
    )

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
        stats_poll_seconds=interval,
    )
    assert supervisor.stats_poll_seconds == 0.0

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    assert await wait_for(lambda: supervisor.state_snapshot()["counts"]["tool_calls"] >= 1)
    await asyncio.sleep(0.3)

    assert supervisor._stats_task is None
    assert supervisor.state_snapshot()["context_usage"] is None
    assert not [
        command
        for command in read_commands(workspace_dir)
        if command.get("type") == "get_session_stats"
    ]

    await supervisor.stop()


def test_find_orphaned_tool_calls_ignores_matched_results() -> None:
    entries = [
        {"type": "session", "version": 3, "id": "abc"},
        {
            "type": "message",
            "id": "e1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {}},
                    {"type": "toolCall", "id": "call-2", "name": "read", "arguments": {}},
                ],
            },
        },
        {
            "type": "message",
            "id": "e2",
            "message": {"role": "toolResult", "toolCallId": "call-1", "toolName": "bash"},
        },
    ]

    assert find_orphaned_tool_calls(entries) == [("call-2", "read")]


def test_repair_orphaned_tool_calls_appends_synthetic_error_results(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "version": 3, "id": "abc", "cwd": str(tmp_path)}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "e1",
                        "parentId": None,
                        "message": {"role": "user", "content": "go"},
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "e2",
                        "parentId": "e1",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call-1",
                                    "name": "bash",
                                    "arguments": {"command": "ls"},
                                }
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert repair_orphaned_tool_calls(session_file) == ["call-1"]

    entries = iter_jsonl_records(session_file.read_text(encoding="utf-8"))
    appended = entries[-1]
    assert appended["parentId"] == "e2"
    assert appended["message"]["role"] == "toolResult"
    assert appended["message"]["toolCallId"] == "call-1"
    assert appended["message"]["toolName"] == "bash"
    assert appended["message"]["isError"] is True
    assert appended["message"]["content"][0]["text"] == ORPHAN_TOOL_RESULT_TEXT

    # The history is now well formed, so a second pass is a no-op.
    assert repair_orphaned_tool_calls(session_file) == []
    assert find_orphaned_tool_calls(entries + [appended]) == []


@pytest.mark.asyncio
async def test_stop_repairs_tool_calls_left_open_by_the_kill(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, hang_mid_tool=True)
    workspace_dir = tmp_path / "workspace"

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    assert await wait_for(lambda: supervisor.state_snapshot()["counts"]["tool_calls"] >= 1)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "running"
    assert snapshot["turns_completed"] == 0

    snapshot = await supervisor.stop()
    assert snapshot["status"] == "stopped"
    assert snapshot["counts"]["repaired_tool_calls"] == 1

    session_file = Path(snapshot["session_file"])
    entries = iter_jsonl_records(session_file.read_text(encoding="utf-8"))
    assert find_orphaned_tool_calls(entries) == []
    assert entries[-1]["message"]["toolCallId"] == "tool-orphan"
    assert entries[-1]["message"]["isError"] is True


def test_iter_jsonl_records_splits_on_newline_only() -> None:
    payload = json.dumps({"type": "message", "text": "line one\u2028line two\u2029end"})
    # U+2028/U+2029 are legal inside a JSON string, and str.splitlines() would cut the
    # record in three; splitting on "\n" alone keeps it intact.
    hazard = json.dumps({"type": "hazard", "text": "a\u2028b\u2029c"}, ensure_ascii=False)
    assert len(hazard.splitlines()) == 3
    assert len(iter_jsonl_records(hazard + "\n")) == 1

    records = iter_jsonl_records(payload + "\r\n" + json.dumps({"type": "second"}) + "\n")

    assert len(records) == 2
    assert records[0]["text"] == "line one\u2028line two\u2029end"
    assert records[1]["type"] == "second"


def test_parse_model_limits_output_extracts_context_window() -> None:
    parsed = parse_model_limits_output(
        "provider  model           context  max-out  thinking  images\n"
        "llamacpp  gemma4-26b-a4b  262.1K   65.5K    yes       yes\n",
        provider="llamacpp",
        model="gemma4-26b-a4b",
    )

    assert parsed is not None
    assert parsed["provider"] == "llamacpp"
    assert parsed["model"] == "gemma4-26b-a4b"
    assert parsed["context_window"] == "262.1K"
    assert parsed["context_window_tokens"] == 262100
    assert parsed["max_output_tokens"] == 65500


# ----------------------------------------------------------------------
# Ordered stream log
# ----------------------------------------------------------------------

CURL_COMMAND = (
    "curl -sS -X POST http://localhost:$PORT/action "
    "-H 'Content-Type: application/json' "
    '-d \'{"actions": ["walk_up","walk_right"]}\''
)


def make_supervisor(tmp_path: Path, **kwargs) -> PiSupervisor:
    return PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        **kwargs,
    )


async def run_tool(
    supervisor: PiSupervisor,
    *,
    tool_call_id: str,
    tool_name: str,
    args: dict,
    result=None,
    is_error: bool = False,
    finish: bool = True,
) -> dict:
    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
        }
    )
    if finish:
        await supervisor._handle_event(
            {
                "type": "tool_execution_end",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "result": result,
                "isError": is_error,
            }
        )
    return last_tool_entry(supervisor)


def stream_entries(supervisor: PiSupervisor, kind: str) -> list[dict]:
    return [entry for entry in supervisor.stream_entries if entry["kind"] == kind]


def last_tool_entry(supervisor: PiSupervisor) -> dict:
    return stream_entries(supervisor, "tool")[-1]


def system_labels(supervisor: PiSupervisor) -> list[str]:
    return [entry["system"]["label"] for entry in stream_entries(supervisor, "system")]


def test_command_headline_reads_a_single_line_comment() -> None:
    command = "# Blocked. Try going up first then right\n" + CURL_COMMAND

    assert command_headline(command) == "Blocked. Try going up first then right"


def test_command_headline_joins_a_multi_line_comment_block() -> None:
    command = (
        "# Blocked by the ledge.\n"
        "#\n"
        "# Step up, then right, then re-read the frame.\n"
        + CURL_COMMAND
        + "\n# trailing comment that is not part of the headline"
    )

    assert command_headline(command) == (
        "Blocked by the ledge. Step up, then right, then re-read the frame."
    )


def test_command_headline_falls_back_to_the_first_command_line() -> None:
    assert command_headline("\n\necho ready\necho done") == "echo ready"

    headline = command_headline(CURL_COMMAND)
    assert len(headline) <= 120
    assert headline.startswith("curl -sS -X POST http://localhost:$PORT/action")
    assert headline.endswith("\u2026")


@pytest.mark.asyncio
async def test_bash_stream_entry_keeps_the_command_verbatim(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)
    command = "# Blocked. Try going up first then right\n" + CURL_COMMAND

    entry = await run_tool(
        supervisor,
        tool_call_id="tool-bash",
        tool_name="bash",
        args={"command": command},
        result='{"actions_executed": 2, "map": "PALLET TOWN", "x": 11, "y": 34, '
        '"facing": "up", "hp": "15/30"}',
    )

    assert entry["kind"] == "tool"
    assert entry["tool"]["name"] == "bash"
    assert entry["tool"]["headline"] == "Blocked. Try going up first then right"
    assert entry["tool"]["command"] == command
    assert entry["tool"]["path"] is None
    assert entry["tool"]["result_summary"] == "x=11 y=34 facing=up hp=15/30"
    assert '"actions_executed"' in entry["tool"]["result_full"]


@pytest.mark.asyncio
async def test_bash_result_summary_falls_back_to_the_first_output_line(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)

    entry = await run_tool(
        supervisor,
        tool_call_id="tool-bash",
        tool_name="bash",
        args={"command": "ls -la"},
        result="total 8\ndrwxr-xr-x  2 dev dev 4096 workspace",
    )

    assert entry["tool"]["headline"] == "ls -la"
    assert entry["tool"]["result_summary"] == "total 8"


@pytest.mark.asyncio
async def test_file_tool_entries_carry_a_verb_headline_and_absolute_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path)
    annotated, _raw = write_frames(workspace)

    read_entry = await run_tool(
        supervisor,
        tool_call_id="tool-read",
        tool_name="read",
        args={"path": str(annotated)},
        result={"type": "image", "data": "..."},
    )
    write_entry = await run_tool(
        supervisor,
        tool_call_id="tool-write",
        tool_name="write",
        args={"path": "turn_plan.json", "content": "{}"},
        result={"ok": True},
    )

    assert read_entry["tool"]["headline"] == "read latest_frame_annotated.png"
    assert read_entry["tool"]["path"] == str(annotated.resolve())
    assert write_entry["tool"]["headline"] == "write turn_plan.json"
    assert write_entry["tool"]["path"] == str((workspace / "turn_plan.json").resolve())


@pytest.mark.asyncio
async def test_image_artifact_is_set_only_for_image_artifact_reads(tmp_path: Path):
    workspace = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path)
    annotated, _raw = write_frames(workspace)
    annotated.write_bytes(b"p" * 4200)
    context = workspace / "turn_context.json"
    context.write_text("{}", encoding="utf-8")
    stray = workspace / "notes.png"
    stray.write_bytes(b"png")

    frame_entry = await run_tool(
        supervisor,
        tool_call_id="tool-frame",
        tool_name="read",
        args={"path": str(annotated)},
        result={"type": "image", "data": "..."},
    )
    context_entry = await run_tool(
        supervisor,
        tool_call_id="tool-context",
        tool_name="read",
        args={"path": str(context)},
        result="{}",
    )
    stray_entry = await run_tool(
        supervisor,
        tool_call_id="tool-stray",
        tool_name="read",
        args={"path": str(stray)},
        result={"type": "image", "data": "..."},
    )

    assert frame_entry["tool"]["image_artifact"] == "latest_frame_annotated"
    assert frame_entry["tool"]["result_summary"] == "image 4.1 KB"
    assert context_entry["tool"]["image_artifact"] is None
    assert context_entry["tool"]["result_summary"] == "{}"
    assert stray_entry["tool"]["image_artifact"] is None


@pytest.mark.asyncio
async def test_tool_entry_goes_running_then_ok_under_one_seq(tmp_path: Path):
    streamed: list[dict] = []

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = make_supervisor(tmp_path, stream_sink=stream)

    await run_tool(
        supervisor,
        tool_call_id="tool-bash",
        tool_name="bash",
        args={"command": "# Look around\necho hi"},
        result="hi",
        finish=False,
    )
    running = last_tool_entry(supervisor)
    assert running["state"] == "running"
    assert running["tool"]["headline"] == "Look around"
    assert running["tool"]["result_summary"] == ""
    running_seq = running["seq"]

    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "tool-bash",
            "toolName": "bash",
            "result": "hi",
            "isError": False,
        }
    )

    assert len(stream_entries(supervisor, "tool")) == 1
    finished = last_tool_entry(supervisor)
    assert finished["seq"] == running_seq
    assert finished["state"] == "ok"
    assert finished["tool"]["result_summary"] == "hi"
    assert finished["tool"]["duration_ms"] is not None

    pushed = [event for event in streamed if event["type"] == "pi_stream_entry"]
    assert [event["entry"]["seq"] for event in pushed] == [running_seq, running_seq]
    assert [event["entry"]["state"] for event in pushed] == ["running", "ok"]


@pytest.mark.asyncio
async def test_failed_tool_call_reports_the_error_first_line(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)

    entry = await run_tool(
        supervisor,
        tool_call_id="tool-bash",
        tool_name="bash",
        args={"command": "curl http://localhost:1/action"},
        result="curl: (7) Failed to connect to localhost port 1\nmore detail here",
        is_error=True,
    )

    assert entry["state"] == "error"
    assert entry["tool"]["result_summary"] == "curl: (7) Failed to connect to localhost port 1"
    assert "more detail here" in entry["tool"]["result_full"]


@pytest.mark.asyncio
async def test_thinking_entry_grows_across_deltas_under_one_seq(tmp_path: Path):
    streamed: list[dict] = []

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = make_supervisor(tmp_path, stream_sink=stream)
    await supervisor._handle_event(
        {"type": "message_start", "message": {"role": "assistant", "content": []}}
    )
    for delta in ("Blocked ", "by the ", "ledge."):
        await supervisor._handle_event(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "thinking_delta", "delta": delta},
            }
        )

    thinking = stream_entries(supervisor, "thinking")
    assert len(thinking) == 1
    assert thinking[0]["state"] == "running"
    assert thinking[0]["text"] == "Blocked by the ledge."

    pushed = [
        event["entry"]
        for event in streamed
        if event["type"] == "pi_stream_entry" and event["entry"]["kind"] == "thinking"
    ]
    assert {entry["seq"] for entry in pushed} == {thinking[0]["seq"]}
    assert [entry["text"] for entry in pushed] == [
        "Blocked",
        "Blocked by the",
        "Blocked by the ledge.",
    ]

    await supervisor._handle_event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Blocked by the ledge."},
                    {"type": "text", "text": "Going up then right."},
                ],
            },
        }
    )

    assert len(stream_entries(supervisor, "thinking")) == 1
    assert stream_entries(supervisor, "thinking")[0]["state"] == "ok"
    assert stream_entries(supervisor, "text")[-1]["text"] == "Going up then right."
    assert any(event["type"] == "pi_thinking_delta" for event in streamed)


@pytest.mark.asyncio
async def test_stream_seq_is_monotonic_across_kinds(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi))

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    entries = supervisor.state_snapshot()["stream"]
    assert [entry["seq"] for entry in entries] == list(range(1, len(entries) + 1))
    assert {entry["kind"] for entry in entries} == {"system", "user", "thinking", "tool", "text"}
    assert all(entry["ts"].endswith("+00:00") for entry in entries)
    assert all(entry["state"] in {"running", "ok", "error"} for entry in entries)
    assert not [entry for entry in entries if entry["state"] == "running"]


def test_stream_cap_drops_the_oldest_without_renumbering(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)
    overflow = 120

    for index in range(STREAM_ENTRY_CAP + overflow):
        supervisor._new_stream_entry("text", text=f"entry {index}")

    entries = supervisor.state_snapshot()["stream"]
    assert len(entries) == STREAM_ENTRY_CAP
    assert entries[0]["seq"] == overflow + 1
    assert entries[-1]["seq"] == STREAM_ENTRY_CAP + overflow
    assert entries[0]["text"] == f"entry {overflow}"
    assert supervisor.stream_since(after=0, limit=5)["next_seq"] == overflow + 5


def test_stream_since_pages_from_a_seq(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)
    supervisor.session_id = "session-1"
    for index in range(5):
        supervisor._new_stream_entry("text", text=f"entry {index}")

    page = supervisor.stream_since(after=2, limit=2)

    assert [entry["seq"] for entry in page["entries"]] == [3, 4]
    assert page["next_seq"] == 4
    assert page["session_id"] == "session-1"
    assert supervisor.stream_since(after=5)["entries"] == []
    assert supervisor.stream_since(after=5)["next_seq"] == 5


@pytest.mark.asyncio
async def test_system_entries_mark_turn_boundaries_and_the_token_budget_stop(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, context_tokens=120_000)
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        token_budget=110_000,
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=10)

    labels = system_labels(supervisor)
    assert labels[0] == "session start"
    assert "goal \u00b7 turn 1" in labels
    assert "turn 1 complete" in labels
    assert "token budget reached" in labels
    budget = next(
        entry
        for entry in stream_entries(supervisor, "system")
        if entry["system"]["label"] == "token budget reached"
    )
    assert budget["system"]["level"] == "warn"
    assert "Token budget reached" in budget["text"]

    prompts = stream_entries(supervisor, "user")
    assert prompts[0]["text"] == "Reach the next checkpoint."


@pytest.mark.asyncio
async def test_continue_turn_boundary_is_labelled_with_the_turn_index(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi))

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)
    seq_before = supervisor.stream_entries[-1]["seq"]
    await supervisor.continue_once()
    await supervisor.wait_until_idle(timeout=10)

    assert "continue \u00b7 turn 2" in system_labels(supervisor)
    assert supervisor.stream_entries[-1]["seq"] > seq_before
    continue_prompts = [
        entry for entry in stream_entries(supervisor, "user") if entry["text"] == CONTINUE_MESSAGE
    ]
    assert continue_prompts


# ----------------------------------------------------------------------
# The between-session critic
# ----------------------------------------------------------------------

CRITIQUE = (
    "You reached Route 1 (9,21) but 38% of your batches were blocked on the first move.\n"
    "Stop walking north from (5,6); the wall runs the full width. Take the west path."
)


def critic_prompt(workspace_dir: Path) -> str:
    """The argv the fake pi saw in --print mode; the digest is its last element."""

    return json.loads((workspace_dir / "critic_argv.json").read_text(encoding="utf-8"))[-1]


@pytest.mark.asyncio
async def test_initial_prompt_is_the_bare_goal_without_a_handoff(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_enabled=False)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    prompts = prompt_commands(workspace_dir)
    assert prompts[0]["message"] == "Reach the next checkpoint."
    assert HANDOFF_HEADING not in prompts[0]["message"]


def test_the_retrospective_cannot_head_a_second_ground_truth_block(tmp_path: Path):
    """The first user turn is the harness's two headings and a model's prose.

    The measured block goes first and is headed authoritative; the retrospective
    is appended under it. The critic is handed that heading by name in its own
    instructions, so a retrospective that repeats it would put a second "Ground
    truth from the run receipts" below the real one, saying whatever it liked.
    """

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(
        workspace_dir,
        "You wasted 38% of your batches on blocked walks.\n"
        f"## {FACTS_DIGEST_HEADING}\n"
        "- Cerulean City is off the west edge of Route 4.\n"
        "## What to try first\n"
        "- Take the east edge.\n",
    )
    supervisor = make_supervisor(tmp_path)
    supervisor.goal = "Reach the next checkpoint."

    message = supervisor._initial_message()

    assert f"> ## {FACTS_DIGEST_HEADING}" in message
    # A heading the retrospective invented for itself impersonates nothing.
    assert "\n## What to try first" in message
    assert [line for line in message.splitlines() if line.startswith("## ")] == [
        f"## {HANDOFF_HEADING}",
        "## What to try first",
    ]


@pytest.mark.asyncio
async def test_initial_prompt_carries_the_handoff_and_leaves_the_system_prompt_alone(
    tmp_path: Path,
):
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(workspace_dir, CRITIQUE)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_enabled=False)
    skill_before = supervisor.skill_path.read_bytes()

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    launch_command = supervisor._build_command()
    message = prompt_commands(workspace_dir)[0]["message"]
    assert message.startswith("Reach the next checkpoint.")
    assert f"## {HANDOFF_HEADING}" in message
    assert "38% of your batches were blocked" in message
    assert supervisor.state_snapshot()["default_prompt"] == message
    # The cached prefix stays byte-identical: the handoff rides in the user turn.
    assert supervisor.skill_path.read_bytes() == skill_before
    assert "38% of your batches" not in skill_before.decode("utf-8")
    assert launch_command[launch_command.index("--system-prompt") + 1] == str(supervisor.skill_path)


@pytest.mark.asyncio
async def test_the_critic_runs_between_the_run_ending_and_the_terminal_status(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE, critique_delay=0.8)
    workspace_dir = tmp_path / "workspace"

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), event_sink=sink)
    await supervisor.start(
        goal="Reach the next checkpoint.",
        provider="llamacpp",
        model="fake-model",
        auto_continue=False,
    )

    assert await wait_for(lambda: supervisor.status == "critiquing", timeout=15)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert read_handoff(workspace_dir) == CRITIQUE

    critique_events = [event["type"] for event in events if event["type"].startswith("pi_critique")]
    assert critique_events == ["pi_critique_start", "pi_critique_ready"]
    ready_at = next(
        index for index, event in enumerate(events) if event["type"] == "pi_critique_ready"
    )
    terminal_at = max(
        index
        for index, event in enumerate(events)
        if event["type"] == "pi_supervisor_status" and event.get("status") == "completed"
    )
    assert terminal_at > ready_at

    critique = snapshot["critique"]
    assert critique["enabled"] is True
    assert critique["text"] == CRITIQUE
    assert critique["at"]
    assert critique["duration_seconds"] >= 0.8
    assert critique["digest_tokens"] > 0
    assert critique["error"] is None
    assert critique["handoff_path"] == str(workspace_dir.resolve() / HANDOFF_FILENAME)
    assert "critique start" in system_labels(supervisor)
    assert "critique ready" in system_labels(supervisor)

    argv = json.loads((workspace_dir / "critic_argv.json").read_text(encoding="utf-8"))
    assert argv[:5] == ["--mode", "json", "--print", "--thinking", "xhigh"]
    assert "--no-session" in argv
    assert argv[argv.index("--tools") + 1] == "read"
    assert argv[argv.index("--model") + 1] == "fake-model"
    assert argv[-1].startswith("You are reviewing a finished session")


@pytest.mark.asyncio
async def test_the_digest_the_critic_reads_carries_the_session_stats(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE, action_tool=True)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        critic_context=lambda: {
            "objective": "Deliver Oak's parcel",
            "game_state": {
                "map": {"map_name": "ROUTE 1", "map_id": 12},
                "player": {"position": {"x": 9, "y": 21}, "badges": [], "money": 3000},
                "party": [{"species": "CHARMANDER", "level": 8, "hp": 12, "max_hp": 26}],
            },
            "map_summary": {
                "map_id": 12,
                "map_name": "ROUTE 1",
                "width": 20,
                "height": 36,
                "coverage": {"seen": 240, "walked": 130, "total": 720},
                "warps": [{"x": 9, "y": 0}],
            },
        },
    )
    supervisor.workspace_dir.mkdir(parents=True, exist_ok=True)
    (supervisor.workspace_dir / "NOTES.md").write_text("Head north on Route 1.", encoding="utf-8")

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    digest = critic_prompt(workspace_dir)
    assert "Objective: Deliver Oak's parcel" in digest
    assert "Session ended: completed" in digest
    assert "ROUTE 1" in digest
    assert "CHARMANDER L8" in digest
    assert "/action batches: 1" in digest
    assert "Buttons sent: 2 (average batch 2.0)" in digest
    assert "blocked_after): 1 of 1 (100%)" in digest
    assert "Batches that ended with moved=0: 1" in digest
    assert "Coverage: seen=240, total=720, walked=130" in digest
    assert "Warps: (9,0)" in digest
    assert "walk north" in digest
    assert "Head north on Route 1." in digest
    # The raw session JSONL is ~1 MB; the digest has to stay in the token budget.
    assert len(digest) <= DIGEST_CHAR_BUDGET


@pytest.mark.asyncio
async def test_the_critic_can_be_disabled(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_enabled=False)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["critique"]["enabled"] is False
    assert snapshot["critique"]["text"] is None
    assert snapshot["config"]["critic_enabled"] is False
    assert not (workspace_dir / HANDOFF_FILENAME).exists()
    assert not (workspace_dir / "critic_argv.json").exists()
    assert "critique start" not in system_labels(supervisor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "critique_kwargs",
    [
        {"critique": None},
        {"critique": ""},
        {"critique": CRITIQUE, "critique_delay": 5.0},
    ],
    ids=["non-zero-exit", "empty-output", "timeout"],
)
async def test_a_failing_critic_retires_the_old_handoff_and_still_ends_the_run(
    tmp_path: Path,
    critique_kwargs: dict,
):
    fake_pi = make_fake_rpc_server(tmp_path, **critique_kwargs)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(workspace_dir, "the previous critique")
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_timeout_seconds=1.0)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["status_reason"] == "Pi completed one turn."
    assert snapshot["critique"]["text"] is None
    assert snapshot["critique"]["error"]
    # Retired, not re-served. Keeping it meant the next session was told about
    # the session before last in the present tense, with its goal line silently
    # reverting to the generic run objective. Measured over one 112-session run:
    # 20 of 99 retrospectives were byte-identical repeats of the one before, and
    # those sessions earned 4 milestones against a run rate of 12 in 112. A
    # session told nothing still reads the deterministic facts block beside it.
    assert read_handoff(workspace_dir) == ""
    assert (workspace_dir / HANDOFF_STALE_FILENAME).is_file(), "the post-mortem survives"
    assert not (workspace_dir / HANDOFF_PREVIOUS_FILENAME).exists()
    assert "critique failed" in system_labels(supervisor)


def test_the_watchdog_treats_critiquing_as_busy(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "keep_run_alive.sh"
    text = script.read_text(encoding="utf-8")
    busy_arm = next(
        line.strip()
        for line in text.splitlines()
        if "critiquing" in line and line.strip().endswith(")")
    )
    assert busy_arm == "running|starting|critiquing)"

    # Run the real case arm, so this fails if the pattern stops matching.
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'for status in running starting critiquing completed stopped stuck ""; do',
                '  case "$status" in',
                f'    {busy_arm} echo "busy:$status" ;;',
                '    "") echo "unknown" ;;',
                '    *) echo "start:$status" ;;',
                "  esac",
                "done",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(probe)], capture_output=True, text=True, check=True)

    assert result.stdout.split() == [
        "busy:running",
        "busy:starting",
        "busy:critiquing",
        "start:completed",
        "start:stopped",
        "start:stuck",
        "unknown",
    ]


# ----------------------------------------------------------------------
# Watching the critic, and surviving one that never answers
# ----------------------------------------------------------------------

CRITIC_REASONING = "Counting the wall-rams: 38 of them, every one from (5,6) heading north."


def critic_entries(supervisor: PiSupervisor, kind: str) -> list[dict]:
    return [
        entry
        for entry in supervisor.stream_entries
        if entry["kind"] == kind and entry.get("source") == "critic"
    ]


@pytest.mark.asyncio
async def test_the_critic_narrates_into_the_stream_as_it_thinks(tmp_path: Path):
    pushed: list[dict] = []

    async def stream_sink(event: dict) -> None:
        if event.get("type") == "pi_stream_entry":
            pushed.append(event["entry"])

    fake_pi = make_fake_rpc_server(
        tmp_path,
        critique=CRITIQUE,
        critique_thinking=CRITIC_REASONING,
    )
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), stream_sink=stream_sink)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    thinking = critic_entries(supervisor, "thinking")
    said = critic_entries(supervisor, "text")
    assert len(thinking) == 1
    assert len(said) == 1
    # Attributable at a glance: the critic's narration is labelled, the agent's is not.
    assert thinking[0]["text"].startswith(CRITIC_STREAM_PREFIX)
    assert said[0]["text"].startswith(CRITIC_STREAM_PREFIX)
    assert "38 of them" in thinking[0]["text"]
    assert "38% of your batches" in said[0]["text"]
    assert thinking[0]["state"] == "ok"
    for entry in stream_entries(supervisor, "thinking"):
        if entry.get("source") != "critic":
            assert not entry["text"].startswith(CRITIC_STREAM_PREFIX)

    # One entry, grown in place by the deltas rather than one entry per delta.
    grew = [
        entry for entry in pushed if entry["kind"] == "thinking" and entry.get("source") == "critic"
    ]
    assert len(grew) > 1
    assert {entry["seq"] for entry in grew} == {thinking[0]["seq"]}
    assert [len(entry["text"]) for entry in grew] == sorted(len(e["text"]) for e in grew)


@pytest.mark.asyncio
async def test_the_heartbeat_mutates_one_system_entry_instead_of_spamming(tmp_path: Path):
    pushed: list[dict] = []

    async def stream_sink(event: dict) -> None:
        if event.get("type") == "pi_stream_entry":
            pushed.append(event["entry"])

    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE, critique_delay=0.5)
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        stream_sink=stream_sink,
        critic_heartbeat_seconds=0.05,
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    beats = critic_entries(supervisor, "system")
    assert len(beats) == 1
    assert beats[0]["system"]["label"].startswith("critique ran ")

    ticks = [
        entry for entry in pushed if entry["kind"] == "system" and entry.get("source") == "critic"
    ]
    assert len(ticks) >= 3
    assert {entry["seq"] for entry in ticks} == {beats[0]["seq"]}
    assert any(entry["system"]["label"].startswith("critique running · ") for entry in ticks)
    # The dividers the operator already relies on are still there.
    assert "critique start" in system_labels(supervisor)
    assert "critique ready" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_a_critic_that_hits_its_output_ceiling_says_so_in_the_snapshot(tmp_path: Path):
    fake_pi = make_fake_rpc_server(
        tmp_path,
        critique="",
        critique_stop_reason="length",
        critique_usage={"input": 2303, "output": 16384},
    )
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(workspace_dir, "the previous critique")
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        critic_retry_enabled=False,
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    critique = snapshot["critique"]
    assert snapshot["status"] == "completed"
    assert critique["text"] is None
    assert critique["error"] == "Critic produced no text (stopReason=length, output=16384)."
    assert critique["stop_reason"] == "length"
    assert critique["usage"] == {"input": 2303, "output": 16384}
    assert critique["salvaged"] is False
    assert critique["raw_path"] == str(
        workspace_dir.resolve() / DEBUG_DIRNAME / CRITIC_RAW_FILENAME
    )
    assert [attempt["thinking"] for attempt in critique["attempts"]] == ["xhigh"]
    assert snapshot["config"]["critic_retry_enabled"] is False
    assert snapshot["config"]["critic_retry_thinking"] == "medium"
    # The artefact that makes the next one a two-minute diagnosis.
    raw = Path(critique["raw_path"]).read_text(encoding="utf-8")
    assert '"stopReason": "length"' in raw
    # Hitting the output ceiling is a failed pass like any other: the stale
    # handoff goes aside rather than to the next session.
    assert read_handoff(workspace_dir) == ""
    assert "critique failed" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_truncated_reasoning_still_leaves_the_next_session_a_handoff(tmp_path: Path):
    reasoning = " ".join(f"thought{index}" for index in range(500))
    fake_pi = make_fake_rpc_server(
        tmp_path,
        critique="",
        critique_thinking=reasoning,
        critique_stop_reason="length",
        critique_usage={"output": 16384},
    )
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_retry_enabled=False)
    skill_before = supervisor.skill_path.read_bytes()

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["critique"]["salvaged"] is True
    # The salvage lands on disk, marked. It is not what the next session reads:
    # see critic.read_handoff, where the live one was 1,648 bytes of the critic
    # counting words at itself.
    written = handoff_path(workspace_dir).read_text(encoding="utf-8")
    assert written.startswith(SALVAGED_REASONING_NOTICE)
    assert read_handoff(workspace_dir) == ""
    assert "thought499" in written
    assert len(written.split()) <= MAX_HANDOFF_WORDS
    assert "critique salvaged" in system_labels(supervisor)
    assert supervisor.skill_path.read_bytes() == skill_before


@pytest.mark.asyncio
async def test_a_retry_seals_the_first_attempts_narration(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)

    await supervisor._on_critic_event({"type": "attempt_start", "attempt": 1, "thinking": "xhigh"})
    await supervisor._on_critic_event({"type": "thinking_delta", "delta": "reasoning "})
    await supervisor._on_critic_event({"type": "thinking_delta", "delta": "forever"})
    await supervisor._on_critic_event({"type": "attempt_start", "attempt": 2, "thinking": "medium"})
    await supervisor._on_critic_event({"type": "text_end", "text": "the short answer"})

    thinking = critic_entries(supervisor, "thinking")
    assert len(thinking) == 1
    assert thinking[0]["text"] == CRITIC_STREAM_PREFIX + "reasoning forever"
    assert thinking[0]["state"] == "ok"
    said = critic_entries(supervisor, "text")
    assert [entry["text"] for entry in said] == [CRITIC_STREAM_PREFIX + "the short answer"]
    assert "critique retry · thinking medium" in system_labels(supervisor)


# ----------------------------------------------------------------------
# Nothing the operator can read is silently shortened
# ----------------------------------------------------------------------


def test_a_payload_past_the_ceiling_says_how_much_it_dropped():
    assert payload_text("small enough") == "small enough"

    clipped = payload_text("y" * (TOOL_RESULT_LIMIT + 500))

    assert clipped.startswith("y" * 1000)
    assert "[truncated: 500 more characters not shown]" in clipped
    # The old marker said nothing about what was lost.
    assert "...[truncated]..." not in clipped


@pytest.mark.asyncio
async def test_a_long_thinking_block_survives_into_the_stream_and_transcript(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)
    # Far past the 8k ceiling that used to cut critic reasoning off mid-thought.
    thinking = ("The wall at (5,6) runs the full width of the room. " * 800).strip()

    await supervisor._handle_event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": thinking}],
            },
        }
    )

    entry = stream_entries(supervisor, "thinking")[-1]
    assert entry["text"] == thinking
    assert "truncated" not in entry["text"]

    transcript = supervisor.state_snapshot()["transcript"][-1]
    assert transcript["content"] == thinking
    # The preview is still a preview: short, and never the only copy.
    assert len(transcript["preview"]) < 250


@pytest.mark.asyncio
async def test_a_long_critique_reaches_the_snapshot_and_the_stream_whole(tmp_path: Path):
    body = "\n\n".join(
        f"{index}. Mistake {index}: you rammed the wall at (5,6) {index} times."
        for index in range(20)
    )
    critique = f"{body}\n\nNEXT GOAL: Take the west path out of Viridian Forest."
    # A retrospective that fills its budget still reaches the dashboard whole:
    # the word cap is on what the critic is asked for, not on what is displayed.
    assert 200 < len(critique.split()) <= MAX_HANDOFF_WORDS
    fake_pi = make_fake_rpc_server(tmp_path, critique=critique)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi))

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["critique"]["text"] == critique
    assert "truncated" not in snapshot["critique"]["text"]
    assert read_handoff(workspace_dir) == critique

    ready = [
        entry
        for entry in stream_entries(supervisor, "system")
        if entry["system"]["label"] == "critique ready"
    ]
    assert ready and ready[0]["text"] == critique
    assert "next goal" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_a_salvaged_critique_reaches_the_snapshot_whole(tmp_path: Path):
    reasoning = ". ".join(f"thought{index}" for index in range(2000)) + "."
    fake_pi = make_fake_rpc_server(
        tmp_path,
        critique="",
        critique_thinking=reasoning,
        critique_stop_reason="length",
        critique_usage={"output": 16384},
    )
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_retry_enabled=False)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    text = snapshot["critique"]["text"]
    assert snapshot["critique"]["salvaged"] is True
    # Whole in the snapshot for a post-mortem; withheld from the next session.
    assert text == handoff_path(workspace_dir).read_text(encoding="utf-8").strip()
    assert read_handoff(workspace_dir) == ""
    # The salvaged tail is marked, cut at a boundary, and never trails a bare "...".
    assert "truncated" in text
    assert "picking up at the next sentence" in text
    assert text.rstrip().endswith("thought1999.")
    assert len(text.split()) <= MAX_HANDOFF_WORDS


# ----------------------------------------------------------------------
# Which goal the next session gets
# ----------------------------------------------------------------------

CRITIQUE_WITH_GOAL = (
    "You won the Boulder Badge at Pewter Gym after 41 turns.\n"
    "Stop re-entering the gym; the badge check already reported it.\n\n"
    "NEXT GOAL: Leave Pewter east onto Route 3 and reach Mt Moon."
)
CRITIC_GOAL = "Leave Pewter east onto Route 3 and reach Mt Moon"


def test_goal_precedence_runs_operator_then_critic_then_objective(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)
    supervisor._session_start_context = {"objective": "Collect all eight badges."}

    assert supervisor._resolve_goal("Beat Brock.") == ("Beat Brock.", GOAL_SOURCE_OPERATOR)
    # No operator goal and no critic goal: the objective engine has the floor.
    assert supervisor._resolve_goal("   ") == (
        "Collect all eight badges.",
        GOAL_SOURCE_OBJECTIVE,
    )

    supervisor.critic_next_goal = CRITIC_GOAL
    assert supervisor._resolve_goal("") == (CRITIC_GOAL, GOAL_SOURCE_CRITIC)
    # An operator who does supply one still outranks the critic.
    assert supervisor._resolve_goal("Beat Brock.") == ("Beat Brock.", GOAL_SOURCE_OPERATOR)

    supervisor.critic_next_goal = ""
    supervisor._session_start_context = {}
    assert supervisor._resolve_goal("") == (FALLBACK_GOAL, GOAL_SOURCE_FALLBACK)


@pytest.mark.asyncio
async def test_a_finished_operator_goal_is_not_re_pinned_on_the_next_start(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE_WITH_GOAL)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        critic_context=lambda: {"objective": "Collect all eight badges."},
    )

    await supervisor.start(goal="Beat Brock in the Pewter Gym.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    first = supervisor.state_snapshot()
    assert first["goal"] == "Beat Brock in the Pewter Gym."
    assert first["goal_source"] == GOAL_SOURCE_OPERATOR
    assert first["critique"]["next_goal"] == CRITIC_GOAL

    # The watchdog's restart names no goal. The finished one must not come back.
    await supervisor.start(auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    second = supervisor.state_snapshot()
    assert second["operator_goal"] == ""
    assert second["goal"] == CRITIC_GOAL
    assert second["goal_source"] == GOAL_SOURCE_CRITIC
    assert prompt_commands(workspace_dir)[-1]["message"].startswith(CRITIC_GOAL)


@pytest.mark.asyncio
async def test_a_next_goal_the_parser_rejects_falls_through_to_the_objective(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, critique="A retrospective.\n\nNEXT GOAL: ok")
    supervisor = make_supervisor(
        tmp_path,
        pi_binary=str(fake_pi),
        critic_context=lambda: {"objective": "Collect all eight badges."},
    )

    await supervisor.start(goal="Beat Brock in the Pewter Gym.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)
    assert supervisor.state_snapshot()["critique"]["next_goal"] == ""

    await supervisor.start(auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["goal"] == "Collect all eight badges."
    assert snapshot["goal_source"] == GOAL_SOURCE_OBJECTIVE


# ---------------------------------------------------------------------------
# The run the sessions add up to
#
# A session dies on the token budget every half hour and the watchdog starts
# another. What is being scored is the playthrough, so the supervisor adopts the
# open run at every start instead of opening a second one beside it.
# ---------------------------------------------------------------------------


def recorded_supervisor(tmp_path: Path, **kwargs) -> tuple[PiSupervisor, RunRecorder]:
    fake_pi = make_fake_rpc_server(tmp_path, **kwargs)
    recorder = RunRecorder(tmp_path / "data")
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
        stats_poll_seconds=0,
        run_recorder=recorder,
    )
    return supervisor, recorder


@pytest.mark.asyncio
async def test_a_run_spans_several_sessions_and_the_presses_carry_over(tmp_path: Path):
    supervisor, recorder = recorded_supervisor(tmp_path)

    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)
    first_run = supervisor.run_id
    assert first_run
    assert supervisor.run_adopted is False
    await recorder.append(tool="action", presses=40, map_name="Route 1")
    await recorder.append(tool="action", presses=17, map_name="Route 1")

    # The budget tripped; the watchdog POSTs /supervisor/start again.
    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    assert supervisor.run_id == first_run
    assert supervisor.run_adopted is True
    assert recorder.total_presses == 57
    assert RunRegistry(tmp_path / "data").load_meta(first_run).status == "running"


@pytest.mark.asyncio
async def test_the_run_snapshot_says_what_the_playthrough_has_cost(tmp_path: Path):
    supervisor, recorder = recorded_supervisor(tmp_path)
    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)
    await recorder.append(tool="action", presses=12)

    run = supervisor.state_snapshot()["run"]

    assert run["run_id"] == supervisor.run_id
    assert run["presses"] == 12
    assert run["sessions"] == 1
    assert run["adopted"] is False


@pytest.mark.asyncio
async def test_a_session_start_announces_the_run_in_the_stream(tmp_path: Path):
    supervisor, _ = recorded_supervisor(tmp_path)
    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    assert "run started" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_reaching_the_objective_is_the_one_thing_that_closes_a_run(tmp_path: Path):
    supervisor, recorder = recorded_supervisor(tmp_path)
    supervisor.set_objective_complete(lambda: True)

    await supervisor.start(goal="Reach Pewter.", auto_continue=True, continue_delay_seconds=0)
    await supervisor.wait_until_idle(timeout=25)

    assert supervisor.run_id is None
    assert recorder.run_id is None
    assert not (tmp_path / "data" / "runs" / RUN_POINTER_FILENAME).exists()
    finished = RunRegistry(tmp_path / "data").list_runs()[-1]
    assert finished.status == STATUS_FINISHED
    assert finished.finish_reason == "objective complete"


@pytest.mark.asyncio
async def test_a_recorder_that_cannot_open_a_run_does_not_stop_the_session(tmp_path: Path):
    class Broken(RunRecorder):
        async def begin_session(self, **kwargs):
            raise OSError("read-only file system")

    fake_pi = make_fake_rpc_server(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
        stats_poll_seconds=0,
        run_recorder=Broken(tmp_path / "data"),
    )

    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    assert supervisor.state_snapshot()["status"] == "completed"
    assert "read-only file system" in (supervisor.run_error or "")


@pytest.mark.asyncio
async def test_a_supervisor_with_no_recorder_still_plays(tmp_path: Path):
    supervisor = make_supervisor(tmp_path, pi_binary=str(make_fake_rpc_server(tmp_path)))

    await supervisor.start(goal="Reach Pewter.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    assert supervisor.state_snapshot()["run"]["run_id"] is None


# ---------------------------------------------------------------------------
# Delivering an intervention
# ---------------------------------------------------------------------------


def intervention_entries(supervisor: PiSupervisor) -> list[dict]:
    return [
        entry
        for entry in supervisor.stream_entries
        if entry["kind"] == "user" and entry["source"] == INTERVENTION_STREAM_SOURCE
    ]


@pytest.mark.asyncio
async def test_an_intervention_is_delivered_down_the_steer_path(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path)
    workspace_dir = tmp_path / "workspace"

    record = await supervisor.deliver_intervention("Walk left four tiles, then up two.")

    steer = prompt_commands(workspace_dir)[-1]
    assert steer["message"] == "Walk left four tiles, then up two."
    assert steer["streamingBehavior"] == "steer"
    assert record["source"] == INTERVENTION_STREAM_SOURCE

    entries = intervention_entries(supervisor)
    assert [entry["text"] for entry in entries] == ["Walk left four tiles, then up two."]
    # Nobody typed it, so it is not filed among the operator's own messages.
    assert operator_entries(supervisor) == []

    await supervisor.stop()


@pytest.mark.asyncio
async def test_an_intervention_gets_more_room_than_a_human_steer(tmp_path: Path):
    supervisor = await live_supervisor(tmp_path)
    plan = "x" * (OPERATOR_MESSAGE_LIMIT + 100)

    with pytest.raises(ValueError):
        await supervisor.send_operator_message(plan)

    record = await supervisor.deliver_intervention(plan)
    assert record["text"] == plan

    with pytest.raises(ValueError):
        await supervisor.deliver_intervention("x" * (INTERVENTION_MESSAGE_LIMIT + 1))

    await supervisor.stop()


@pytest.mark.asyncio
async def test_an_intervention_cannot_land_without_a_live_session(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)

    with pytest.raises(NoLiveSessionError):
        await supervisor.deliver_intervention("Walk left.")


@pytest.mark.asyncio
async def test_a_fired_intervention_shows_up_in_the_supervisor_state(tmp_path: Path):
    supervisor = make_supervisor(tmp_path)

    await supervisor.record_intervention(
        {
            "at": "2026-08-26T09:00:00Z",
            "trigger": "circling",
            "reason": "112 positions across 40 tiles.",
            "answer": "Head south to the gate.",
            "delivered": True,
            "status": {"enabled": True, "fired": 1, "delivered": 1, "slot_lost": None},
        }
    )

    state = supervisor.state_snapshot()["interventions"]
    assert state["enabled"] is True
    assert state["fired"] == 1
    assert state["last"]["trigger"] == "circling"
    assert state["last"]["delivered"] is True
    assert "intervention: circling" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_a_lost_slot_is_the_loudest_thing_the_supervisor_reports(tmp_path: Path):
    """The player's whole context is a file on the model box. Nothing swallows that."""
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = make_supervisor(tmp_path, event_sink=sink)

    await supervisor.record_intervention(
        {
            "at": "2026-08-26T09:00:00Z",
            "trigger": "stalled",
            "error": "could not restore slot 0",
            "status": {
                "enabled": False,
                "fired": 1,
                "delivered": 0,
                "disabled_reason": "interventions are off for the rest of this session",
                "slot_lost": {
                    "at": "2026-08-26T09:00:00Z",
                    "filename": "player.bin",
                    "message": "could not restore slot 0 from 'player.bin'",
                },
            },
        }
    )

    state = supervisor.state_snapshot()["interventions"]
    assert state["slot_lost"]["filename"] == "player.bin"
    assert state["disabled_reason"]
    assert "could not restore slot 0" in supervisor.state_snapshot()["last_error"]

    errors = [
        entry
        for entry in supervisor.stream_entries
        if (entry.get("system") or {}).get("level") == "error"
    ]
    assert any("slot lost" in (entry["system"]["label"]) for entry in errors)
    assert any(event["type"] == "pi_intervention_slot_lost" for event in events)


# ---------------------------------------------------------------------------
# Workspace staging
#
# The workspace outlives a session, so everything staged into it has to be
# refreshed on the way in and everything retired has to be removed — a stale
# copy keeps working, which is exactly how a deleted contract comes back.
# ---------------------------------------------------------------------------


def staged(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    supervisor = PiSupervisor(workspace_dir=workspace, server_url="http://localhost:1")
    supervisor._stage_workspace_helpers()
    return workspace


def test_staging_puts_the_cli_the_client_and_the_wrapper_in_the_workspace(tmp_path):
    workspace = staged(tmp_path)

    for name in ("poke", "poke.py", "py"):
        staged_file = workspace / name
        assert staged_file.is_file(), name
        assert os.access(staged_file, os.X_OK), name

    repo_root = Path(__file__).resolve().parents[1]
    assert (workspace / "poke").read_bytes() == (
        repo_root / "pokemon_agent" / "agent_cli.py"
    ).read_bytes()
    assert (workspace / "poke.py").read_bytes() == (
        repo_root / "pokemon_agent" / "agent_api.py"
    ).read_bytes()


def test_the_staged_client_imports_under_a_plain_interpreter(tmp_path):
    """`import poke` has to work from the workspace, with no package on the path."""
    workspace = staged(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import poke; print(poke.limits()['max_actions_per_batch'], poke.client())",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PORT": "4242"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "40 Client('http://localhost:4242')"


def test_staging_removes_helpers_and_contracts_that_were_retired(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    for stale in PiSupervisor.STALE_WORKSPACE_HELPERS:
        (workspace / stale).write_text("stale")

    staged(tmp_path)

    for stale in PiSupervisor.STALE_WORKSPACE_HELPERS:
        assert not (workspace / stale).exists(), stale
    assert "turn_plan.json" in PiSupervisor.STALE_WORKSPACE_HELPERS


def test_staging_makes_the_skills_directory_and_explains_it_once(tmp_path):
    workspace = staged(tmp_path)
    readme = workspace / "skills" / "README.md"

    assert readme.is_file()
    assert "survive" in readme.read_text()
    assert "./py" in readme.read_text()

    # Everything under skills/ belongs to the model, including the README once
    # it has edited it, and its own scripts and notes.
    readme.write_text("mine now")
    (workspace / "skills" / "cross_mt_moon.py").write_text("import poke")
    (workspace / "NOTES.md").write_text("what I learned")

    staged(tmp_path)

    assert readme.read_text() == "mine now"
    assert (workspace / "skills" / "cross_mt_moon.py").read_text() == "import poke"
    assert (workspace / "NOTES.md").read_text() == "what I learned"


def fake_venv_builder(workspace: Path, calls: list, fail_install: bool = False):
    """Stand in for `python -m venv` and `pip install`, without either."""

    def run(command, **kwargs):
        calls.append(list(command))
        if "venv" in command:
            (workspace / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (workspace / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
        elif fail_install:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_the_workspace_interpreter_is_built_once_and_never_rebuilt(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    supervisor = PiSupervisor(workspace_dir=workspace, server_url="http://localhost:1")
    calls: list[list[str]] = []
    monkeypatch.setattr(pi_supervisor_module.subprocess, "run", fake_venv_builder(workspace, calls))

    STAGE_WORKSPACE_VENV(supervisor)

    assert len(calls) == 2
    assert calls[0][1:3] == ["-m", "venv"]
    assert "pillow" in calls[1] and "numpy" in calls[1]
    assert (workspace / ".venv" / PiSupervisor.WORKSPACE_VENV_STAMP).exists()

    # A session start must not pay for the venv twice.
    STAGE_WORKSPACE_VENV(supervisor)
    assert len(calls) == 2


def test_a_venv_whose_packages_never_landed_is_finished_next_session(tmp_path, monkeypatch):
    """A `bin/python` is not proof: pip is the half that matters and can fail."""
    workspace = tmp_path / "workspace"
    supervisor = PiSupervisor(workspace_dir=workspace, server_url="http://localhost:1")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pi_supervisor_module.subprocess,
        "run",
        fake_venv_builder(workspace, calls, fail_install=True),
    )

    STAGE_WORKSPACE_VENV(supervisor)
    assert (workspace / ".venv" / "bin" / "python").exists()
    assert not (workspace / ".venv" / PiSupervisor.WORKSPACE_VENV_STAMP).exists()

    calls.clear()
    monkeypatch.setattr(pi_supervisor_module.subprocess, "run", fake_venv_builder(workspace, calls))
    STAGE_WORKSPACE_VENV(supervisor)

    # Only the install is retried; the interpreter is already there.
    assert len(calls) == 1
    assert "pillow" in calls[0]
    assert (workspace / ".venv" / PiSupervisor.WORKSPACE_VENV_STAMP).exists()


def test_a_workspace_interpreter_that_will_not_build_does_not_stop_the_run(
    tmp_path, monkeypatch, capsys
):
    supervisor = PiSupervisor(workspace_dir=tmp_path / "workspace", server_url="http://localhost:1")

    def explode(command, **kwargs):
        raise OSError("no python here")

    monkeypatch.setattr(pi_supervisor_module.subprocess, "run", explode)

    STAGE_WORKSPACE_VENV(supervisor)

    assert "no python here" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Between-session handoff
#
# Nine sessions and four rotations on disk produced no HANDOFF.md and no
# critic_last.jsonl. Every one of those rotations went through `stop()`, which
# cancelled the critic on the reasoning that an operator stop has to return
# promptly - and `/supervisor/start` refuses while a session is live, so there is
# no rotation that does not go through `stop()` first. The tests below pin every
# path a session can end on, because the bug was never in the critic.
# ---------------------------------------------------------------------------


async def end_by_completing(supervisor: PiSupervisor) -> None:
    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=30)


async def end_by_operator_stop(supervisor: PiSupervisor) -> None:
    await supervisor.start(
        goal="Reach the next checkpoint.", auto_continue=True, continue_delay_seconds=5
    )
    assert await wait_for(lambda: supervisor.status == "running", timeout=15)
    await supervisor.stop()
    await supervisor.wait_until_idle(timeout=30)


async def end_by_idle_breaker(supervisor: PiSupervisor) -> None:
    supervisor.max_idle_turns = 1
    await supervisor.start(
        goal="Reach the next checkpoint.", auto_continue=True, continue_delay_seconds=0
    )
    await supervisor.wait_until_idle(timeout=30)


async def end_by_token_budget(supervisor: PiSupervisor) -> None:
    supervisor.token_budget = 110_000
    await supervisor.start(
        goal="Reach the next checkpoint.", auto_continue=True, continue_delay_seconds=0
    )
    await supervisor.wait_until_idle(timeout=30)


#: Every way a session ends: how to drive it there, the fake-pi flags that get it
#: there, and the terminal status it lands on.
ENDING_PATHS = {
    "completed": (end_by_completing, {}, "completed"),
    "operator-stop": (end_by_operator_stop, {"settle": False}, "stopped"),
    "idle-breaker": (end_by_idle_breaker, {"include_write_tool": False}, "stuck"),
    "token-budget": (end_by_token_budget, {"context_tokens": 120_000}, "completed"),
    "crash": (end_by_completing, {}, "error"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(ENDING_PATHS))
async def test_every_way_a_session_ends_leaves_a_retrospective(
    tmp_path: Path, monkeypatch, path: str
):
    drive, flags, expected_status = ENDING_PATHS[path]
    events: list[dict] = []
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE_WITH_GOAL, **flags)
    workspace_dir = tmp_path / "workspace"

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = make_supervisor(
        tmp_path, pi_binary=str(fake_pi), event_sink=sink, stats_poll_seconds=0
    )
    if path == "crash":
        # A session that ends by raising is still a session that ended.
        async def explode(self) -> bool:
            raise RuntimeError("pi died mid-turn")

        monkeypatch.setattr(PiSupervisor, "_await_settle", explode)

    await drive(supervisor)

    assert supervisor.status == expected_status
    assert read_handoff(workspace_dir) == CRITIQUE_WITH_GOAL
    assert supervisor.critic_next_goal == CRITIC_GOAL
    assert [event["type"] for event in events if event["type"].startswith("pi_critique")] == [
        "pi_critique_start",
        "pi_critique_ready",
    ]
    # The raw stream is kept whatever happened, so a bad critique is debuggable.
    assert (workspace_dir / DEBUG_DIRNAME / CRITIC_RAW_FILENAME).is_file()


@pytest.mark.asyncio
async def test_an_operator_stop_returns_promptly_and_critiques_in_the_background(tmp_path: Path):
    """Promptness and the retrospective were traded against each other for nothing.

    `stop()` returns on a shielded wait, so the run loop stays alive under
    `critiquing`, and the watchdog - which already treats `critiquing` as busy -
    waits for it instead of starting a session over the top of it.
    """

    fake_pi = make_fake_rpc_server(
        tmp_path, critique=CRITIQUE_WITH_GOAL, critique_delay=8.0, settle=False
    )
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), stats_poll_seconds=0)

    await supervisor.start(
        goal="Reach the next checkpoint.", auto_continue=True, continue_delay_seconds=5
    )
    assert await wait_for(lambda: supervisor.status == "running", timeout=15)

    started = time.monotonic()
    snapshot = await supervisor.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 8.0
    assert snapshot["status"] == "critiquing"
    assert supervisor.is_running is True

    await supervisor.wait_until_idle(timeout=40)
    assert supervisor.status == "stopped"
    assert read_handoff(workspace_dir) == CRITIQUE_WITH_GOAL


@pytest.mark.asyncio
async def test_a_server_shutdown_does_not_start_a_critic_it_cannot_finish(tmp_path: Path):
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE_WITH_GOAL, settle=False)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), stats_poll_seconds=0)

    await supervisor.start(
        goal="Reach the next checkpoint.", auto_continue=True, continue_delay_seconds=5
    )
    assert await wait_for(lambda: supervisor.status == "running", timeout=15)

    await supervisor.shutdown()
    await supervisor.wait_until_idle(timeout=20)

    assert supervisor.status == "stopped"
    assert not (workspace_dir / HANDOFF_FILENAME).exists()
    assert "critique start" not in system_labels(supervisor)


@pytest.mark.asyncio
async def test_a_critic_that_overruns_its_own_budget_is_abandoned(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pi_supervisor_module, "CRITIC_OVERRUN_GRACE_SECONDS", 0.5)
    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE_WITH_GOAL)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_timeout_seconds=0.5)

    async def never_returns(**kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(pi_supervisor_module._critic_module(), "run_critic", never_returns)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert "overran" in snapshot["critique"]["error"]
    assert "critique failed" in system_labels(supervisor)


@pytest.mark.asyncio
async def test_a_failed_critique_does_not_stop_the_next_session_from_starting(tmp_path: Path):
    # `critique=None` exits non-zero: the retrospective is lost, the run is not.
    fake_pi = make_fake_rpc_server(tmp_path, critique=None)
    workspace_dir = tmp_path / "workspace"
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_timeout_seconds=2.0)

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)
    assert supervisor.state_snapshot()["critique"]["error"]

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=25)

    assert supervisor.status == "completed"
    assert len(prompt_commands(workspace_dir)) >= 2


# ---------------------------------------------------------------------------
# What the handoff carries
# ---------------------------------------------------------------------------


async def record_a_stuck_session(recorder: RunRecorder) -> None:
    """Receipts shaped like the Route 3 pocket: many presses, almost no ground."""

    await recorder.append(
        tool="action",
        presses=40,
        map_name="Route 3",
        pos=(22, 12),
        moved=0,
        hp=(62, 62),
        party_size=1,
    )
    for _ in range(4):
        await recorder.append(
            tool="action",
            presses=20,
            map_name="Route 3",
            pos=(22, 12),
            moved=0,
            hp=(62, 62),
            party_size=1,
        )
    await recorder.append(
        tool="goto",
        presses=32,
        map_name="Route 3",
        pos=(22, 8),
        moved=32,
        hp=(62, 62),
        party_size=1,
    )


@pytest.mark.asyncio
async def test_the_first_message_carries_ground_truth_off_the_receipts(tmp_path: Path):
    supervisor, recorder = recorded_supervisor(tmp_path, critique=CRITIQUE_WITH_GOAL)
    workspace_dir = tmp_path / "workspace"
    saves = tmp_path / "data" / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    (saves / "pewter_start.state").write_bytes(b"x")
    (saves / "auto__000123.state").write_bytes(b"x")

    await supervisor.start(goal="Cross Route 3.", auto_continue=False)
    await recorder.append(tool="action", presses=1, milestone_ids={"BADGE_BOULDER"})
    await record_a_stuck_session(recorder)
    await supervisor.wait_until_idle(timeout=30)

    await supervisor.start(auto_continue=False)
    await supervisor.wait_until_idle(timeout=30)

    message = prompt_commands(workspace_dir)[-1]["message"]
    assert message.startswith(CRITIC_GOAL)
    assert f"## {FACTS_HEADING}" in message
    assert f"Run {supervisor.run_id}, after session 1" in message
    assert "153 presses spent on it so far" in message
    # What the run already has, so the next goal is never one it has already met -
    # as the ladder's own label, which is an instruction, rather than an event id.
    assert "Done (1), highest rung: Boulder Badge" in message
    assert "Next rung: Beat the Super Nerd guarding the Mt. Moon fossils" in message
    # Where it got stuck, and how badly.
    assert "Most revisited tile: Route 3 (22,12), stood on 5 times" in message
    assert "5 of 7 batches (71%) moved nothing" in message
    # Which verbs it reached for besides walking, and by omission which it did not.
    assert "Verbs beyond walking: goto x1" in message
    # Where the exits are, off the world file: every false belief in these
    # transcripts has been a compass direction the model invented for itself.
    # No star legend, because no exit here is starred: measured on the live run's
    # session 6, that legend cost 22 bytes over a line with no `*` in it.
    assert "Every way off Route 3: walk north -> Route 4" in message
    # Named so a branch can be recognised, but not offered. This line used to
    # end "loadable with `./poke load <name>`" and sat at the top of every
    # session, advertising the run's most expensive behaviour: 143 loads and at
    # least thirteen rungs handed back, Misty and the Cascade Badge each bought
    # three times.
    assert "Saves on disk, newest first: pewter_start." in message
    assert "poke load" not in message
    assert "auto__" not in message
    # The critic's own words come after the facts, under their own heading.
    assert message.index(FACTS_HEADING) < message.index(HANDOFF_HEADING)
    # Everything the critic wrote except its NEXT GOAL line, which is already
    # the first line of this message. See `critic.handoff_body`.
    assert handoff_body(CRITIQUE_WITH_GOAL) in message
    assert "NEXT GOAL" not in message


@pytest.mark.asyncio
async def test_ground_truth_survives_the_restart_that_takes_the_critic_with_it(tmp_path: Path):
    """The second reason `HANDOFF.md` was missing, and the one `stop()` cannot fix.

    When the server process dies mid-session, the run loop's teardown never runs
    and no critic ever starts. A fresh `PiSupervisor` over the same workspace is
    exactly what comes back up. The facts are read at *start*, off the receipts
    and a mark file, so they cross that boundary even though nothing in memory
    does.
    """

    fake_pi = make_fake_rpc_server(tmp_path, critique=CRITIQUE_WITH_GOAL)
    workspace_dir = tmp_path / "workspace"

    def build() -> PiSupervisor:
        return PiSupervisor(
            workspace_dir=workspace_dir,
            server_url="http://127.0.0.1:8765",
            pi_binary=str(fake_pi),
            stats_poll_seconds=0,
            critic_enabled=False,
            run_recorder=RunRecorder(tmp_path / "data"),
        )

    killed = build()
    await killed.start(goal="Cross Route 3.", auto_continue=False)
    await killed.wait_until_idle(timeout=30)
    await killed.run_recorder.append(
        tool="action", presses=40, map_name="Route 3", pos=(22, 12), moved=0
    )
    assert not (workspace_dir / HANDOFF_FILENAME).exists()

    restarted = build()
    await restarted.start(goal="Cross Route 3.", auto_continue=False)
    await restarted.wait_until_idle(timeout=30)

    message = prompt_commands(workspace_dir)[-1]["message"]
    assert f"## {FACTS_HEADING}" in message
    assert "40 presses over 1 batch, ended on Route 3 (22,12)" in message
    assert HANDOFF_HEADING not in message


@pytest.mark.asyncio
async def test_the_digest_the_critic_reads_is_grounded_in_the_receipts(tmp_path: Path):
    supervisor, recorder = recorded_supervisor(tmp_path, critique=CRITIQUE_WITH_GOAL, settle=False)
    workspace_dir = tmp_path / "workspace"

    await supervisor.start(goal="Cross Route 3.", auto_continue=True, continue_delay_seconds=5)
    assert await wait_for(lambda: supervisor.status == "running", timeout=15)
    await record_a_stuck_session(recorder)
    await supervisor.stop()
    await supervisor.wait_until_idle(timeout=40)

    digest = critic_prompt(workspace_dir)
    assert FACTS_DIGEST_HEADING in digest
    assert "Most revisited tile: Route 3 (22,12), stood on 5 times" in digest
    assert "authoritative - do not contradict these" in digest
    # The facts sit above the model's own account of the same session.
    assert digest.index(FACTS_DIGEST_HEADING) < digest.index("What it did (measured")


@pytest.mark.asyncio
async def test_the_digest_carries_intelligence_the_session_never_had(tmp_path: Path):
    """The critic pays for its context once; the player pays for its own on all
    five hundred turns. So the critic is told what the session could not afford
    to be: the map's real exits, where the presses went, and what it repeated."""

    supervisor, recorder = recorded_supervisor(tmp_path, critique=CRITIQUE_WITH_GOAL, settle=False)
    workspace_dir = tmp_path / "workspace"

    await supervisor.start(goal="Cross Route 3.", auto_continue=True, continue_delay_seconds=5)
    assert await wait_for(lambda: supervisor.status == "running", timeout=15)
    await record_a_stuck_session(recorder)
    await supervisor.stop()
    await supervisor.wait_until_idle(timeout=40)

    digest = critic_prompt(workspace_dir)
    # Geography off the world file, not off the model's memory of Kanto.
    assert "walk north -> Route 4" in digest
    # Presses bucketed the way `scope waste` buckets them.
    assert "presses this session:" in digest
    # Trainers standing on it, which is a reason to walk somewhere on purpose.
    assert "Trainers standing here:" in digest
    # And the measurements sit above the model's own account, not beside it.
    assert digest.index("game's own map data") < digest.index("## Session")


@pytest.mark.asyncio
async def test_the_first_message_does_not_hand_the_goal_over_twice(tmp_path: Path):
    """The goal is block one. The handoff's own NEXT GOAL line is the same words.

    Measured on the live run's HANDOFF.md, the line came to 83 bytes and said
    exactly what the message already opened with -- once as the instruction and
    once as the last thing the model reads before it acts.
    """
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(workspace_dir, CRITIQUE_WITH_GOAL)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_enabled=False)

    await supervisor.start(
        goal="Leave Pewter east onto Route 3 and reach Mt Moon.", auto_continue=False
    )
    await supervisor.wait_until_idle(timeout=10)

    message = prompt_commands(workspace_dir)[0]["message"]

    assert message.startswith("Leave Pewter east onto Route 3 and reach Mt Moon.")
    assert "NEXT GOAL" not in message
    assert message.count("Leave Pewter east onto Route 3 and reach Mt Moon.") == 1
    # Everything the critic actually wrote still arrives.
    assert "Stop re-entering the gym" in message
    # And the file keeps the whole record for the post-mortem.
    assert "NEXT GOAL:" in read_handoff(workspace_dir)


@pytest.mark.asyncio
async def test_an_operator_goal_is_not_contradicted_by_the_handoffs_own_goal(tmp_path: Path):
    """Two instructions in one message is worse than a duplicated one.

    An operator goal outranks the critic's, and the critic's used to arrive
    anyway at the bottom of the same message. `check_next_goal` throws a goal
    away for naming an unroutable map or a direction the map graph contradicts,
    and that rejected goal took the same route back in; its docstring prices one
    such goal at 5,618 presses.
    """
    fake_pi = make_fake_rpc_server(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    write_handoff(workspace_dir, CRITIQUE_WITH_GOAL)
    supervisor = make_supervisor(tmp_path, pi_binary=str(fake_pi), critic_enabled=False)

    await supervisor.start(goal="Heal at the Pewter Pokecenter first.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=10)

    message = prompt_commands(workspace_dir)[0]["message"]

    assert message.startswith("Heal at the Pewter Pokecenter first.")
    assert "Leave Pewter east onto Route 3" not in message
