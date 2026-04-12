from pathlib import Path

import pytest

from pokemon_agent.pi_supervisor import PiSupervisor, parse_model_limits_output


def make_fake_pi_script(tmp_path: Path, *, include_turn_end: bool = True) -> Path:
    script = tmp_path / "fake-pi"
    template = """#!/usr/bin/env python3
import json
import pathlib
import sys

resume = "--continue" in sys.argv or "--session" in sys.argv
session_dir = pathlib.Path(sys.argv[sys.argv.index("--session-dir") + 1])
session_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = session_dir.parent
session_file = session_dir / "session-123.jsonl"
if not session_file.exists():
    session_file.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "session-123",
                "timestamp": "2026-01-01T00:00:00Z",
                "cwd": str(workspace_dir),
            }
        )
        + "\\n",
        encoding="utf-8",
    )
message = "Continued turn." if resume else "Initial turn."

events = [
    {"type": "session", "id": "session-123", "cwd": str(workspace_dir)},
    {"type": "agent_start"},
    {"type": "turn_start", "turnIndex": 1},
    {"type": "message_start", "message": {"role": "assistant", "content": []}},
    {
        "type": "message_update",
        "message": {"role": "assistant", "content": []},
        "assistantMessageEvent": {"type": "thinking_delta", "delta": "Inspecting the frame."},
    },
    {
        "type": "tool_execution_start",
        "toolCallId": "tool-1",
        "toolName": "write",
        "args": {
            "path": str(workspace_dir / "turn_plan.json"),
            "content": json.dumps(
                {
                    "objective_id": "get_oaks_parcel",
                    "summary": "Move north carefully.",
                    "planned_actions": ["walk_up"],
                    "fallback_actions": ["walk_left"],
                }
            ),
        },
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "tool-1",
        "toolName": "write",
        "result": {"ok": True},
        "isError": False,
    },
    {
        "type": "message_update",
        "message": {"role": "assistant", "content": []},
        "assistantMessageEvent": {"type": "text_delta", "delta": message},
    },
    {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "Inspecting the frame."},
                {"type": "text", "text": message},
            ],
            "usage": {
                "input": 3200,
                "output": 260,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 3460,
            },
        },
    },
    {"type": "agent_end", "messages": []},
]

if __INCLUDE_TURN_END__:
    events.insert(
        -1,
        {
            "type": "turn_end",
            "turnIndex": 1,
            "message": {"role": "assistant", "content": [{"type": "text", "text": message}]},
            "toolResults": [],
        },
    )

for event in events:
    print(json.dumps(event), flush=True)
"""
    script.write_text(
        template.replace("__INCLUDE_TURN_END__", repr(include_turn_end)),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


@pytest.mark.asyncio
async def test_pi_supervisor_tracks_turn_output_and_tools(tmp_path: Path):
    events: list[dict] = []
    streamed: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)

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
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 1
    assert snapshot["session_id"] == "session-123"
    assert snapshot["goal"] == "Reach the next checkpoint."
    assert snapshot["last_assistant_text"] == "Initial turn."
    assert "Inspecting the frame." in snapshot["last_assistant_thinking"]
    assert snapshot["turn_plan_preview"]["payload"]["objective_id"] == "get_oaks_parcel"
    assert snapshot["recent_tools"][-1]["tool_name"] == "write"
    assert '"objective_id": "get_oaks_parcel"' in snapshot["recent_tools"][-1]["args"]
    assert '"ok": true' in snapshot["recent_tools"][-1]["result"]
    assert any(
        entry["direction"] == "outbound" and entry["role"] == "user"
        for entry in snapshot["transcript"]
    )
    assert any(
        entry["channel"] == "assistant" and entry["role"] == "assistant"
        for entry in snapshot["transcript"]
    )
    assert any(event["type"] == "pi_turn_end" for event in events)
    assert any(event["type"] == "pi_text_delta" for event in streamed)
    assert any(event["type"] == "pi_prompt_sent" for event in streamed)


@pytest.mark.asyncio
async def test_pi_supervisor_attaches_latest_frame_pngs_to_turn_prompt(tmp_path: Path):
    events: list[dict] = []
    streamed: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    annotated = workspace_dir / "latest_frame_annotated.png"
    raw = workspace_dir / "latest_frame.png"
    annotated.write_bytes(b"annotated-frame")
    raw.write_bytes(b"raw-frame")

    async def sink(event: dict) -> None:
        events.append(event)

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        stream_sink=stream,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    launch_event = next(event for event in events if event["type"] == "pi_turn_launch")
    assert f"@{annotated}" in launch_event["command_preview"]
    assert f"@{raw}" in launch_event["command_preview"]

    prompt_event = next(event for event in streamed if event["type"] == "pi_prompt_sent")
    assert prompt_event["attachments"] == [str(annotated), str(raw)]


@pytest.mark.asyncio
async def test_pi_supervisor_auto_continue_uses_exact_session_file(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)
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
        auto_continue=True,
        max_turns=2,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=5)

    launch_events = [event for event in events if event["type"] == "pi_turn_launch"]
    expected_session_file = workspace_dir / "pi-session" / "session-123.jsonl"

    assert len(launch_events) == 2
    assert "--session" not in launch_events[0]["command_preview"]
    assert "--session" in launch_events[1]["command_preview"]
    assert str(expected_session_file) in launch_events[1]["command_preview"]

    snapshot = supervisor.state_snapshot()
    assert snapshot["session_file"] == str(expected_session_file)


@pytest.mark.asyncio
async def test_pi_supervisor_can_continue_existing_session(tmp_path: Path):
    fake_pi = make_fake_pi_script(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)
    await supervisor.continue_once()
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["turns_completed"] == 2
    assert snapshot["last_assistant_text"] == "Continued turn."
    assert snapshot["status"] == "completed"


@pytest.mark.asyncio
async def test_pi_supervisor_continue_snapshot_clears_live_stream_fields(tmp_path: Path):
    fake_pi = make_fake_pi_script(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    snapshot = await supervisor.continue_once()

    assert snapshot["status"] == "starting"
    assert snapshot["current_assistant_text"] == ""
    assert snapshot["current_assistant_thinking"] == ""
    assert snapshot["active_tools"] == []

    await supervisor.wait_until_idle(timeout=5)


@pytest.mark.asyncio
async def test_pi_supervisor_auto_continue_schedules_and_runs_next_turn(tmp_path: Path):
    events: list[dict] = []
    streamed: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)

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

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        max_turns=2,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 2
    assert snapshot["continue_count"] == 1
    assert any(event["type"] == "pi_auto_continue_scheduled" for event in events)
    assert any(
        event["type"] == "pi_prompt_sent" and event.get("resume") is True for event in streamed
    )


@pytest.mark.asyncio
async def test_pi_supervisor_synthesizes_turn_completion_when_pi_omits_turn_end(
    tmp_path: Path,
):
    events: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path, include_turn_end=False)

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        max_turns=2,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 2
    assert snapshot["last_assistant_text"] == "Continued turn."
    assert any(
        event["type"] == "pi_turn_end" and event.get("synthetic") is True for event in events
    )


@pytest.mark.asyncio
async def test_pi_supervisor_tracks_usage_and_counts(tmp_path: Path):
    fake_pi = make_fake_pi_script(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(
        goal="Reach the next checkpoint.",
        auto_continue=True,
        max_turns=2,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    counts = snapshot["counts"]
    assert counts["assistant_messages"] == 2
    assert counts["thinking_blocks"] == 2
    assert counts["tool_calls"] == 2
    assert counts["user_messages"] == 0
    usage = snapshot["session_usage"]
    assert usage is not None
    assert usage["totalTokens"] == 3460
    assert snapshot["last_message_usage"]["output"] == 260
    assert snapshot["compaction"]["tokens_before"] is None


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
