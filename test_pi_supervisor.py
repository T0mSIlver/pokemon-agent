import asyncio
from pathlib import Path

import pytest

from pokemon_agent.harness.prompting import CONTINUE_PROMPT
from pokemon_agent.pi_supervisor import PiSupervisor, parse_model_limits_output


def make_fake_pi_script(
    tmp_path: Path,
    *,
    include_turn_end: bool = True,
    include_frame_read: bool = False,
    include_write_tool: bool = True,
    linger_after_events: bool = False,
) -> Path:
    script = tmp_path / "fake-pi"
    template = """#!/usr/bin/env python3
import json
import pathlib
import sys
import time

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

vision_events = []
if __INCLUDE_FRAME_READ__:
    vision_events = [
        {
            "type": "tool_execution_start",
            "toolCallId": "tool-read-1",
            "toolName": "read",
            "args": {
                "path": str(workspace_dir / "latest_frame_annotated.png"),
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "tool-read-1",
            "toolName": "read",
            "result": {"ok": True},
            "isError": False,
        },
    ]

tool_events = []
if __INCLUDE_WRITE_TOOL__:
    tool_events = [
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
    ]

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
    *vision_events,
    *tool_events,
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

if __LINGER_AFTER_EVENTS__:
    while True:
        time.sleep(1)
"""
    script.write_text(
        template.replace("__INCLUDE_TURN_END__", repr(include_turn_end))
        .replace("__INCLUDE_FRAME_READ__", repr(include_frame_read))
        .replace("__INCLUDE_WRITE_TOOL__", repr(include_write_tool))
        .replace("__LINGER_AFTER_EVENTS__", repr(linger_after_events)),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def make_fake_cycle_pi_script(
    tmp_path: Path,
    *,
    include_frame_read: bool = True,
) -> Path:
    script = tmp_path / "fake-pi-cycle"
    template = """#!/usr/bin/env python3
import json
import pathlib
import sys
import time

session_dir = pathlib.Path(sys.argv[sys.argv.index("--session-dir") + 1])
session_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = session_dir.parent
session_file = session_dir / "session-cycle.jsonl"
if not session_file.exists():
    session_file.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "session-cycle",
                "timestamp": "2026-01-01T00:00:00Z",
                "cwd": str(workspace_dir),
            }
        )
        + "\\n",
        encoding="utf-8",
    )

observe_result = {
    "observation_id": "obs-cycle-1",
    "artifacts": {
        "latest_frame": str(workspace_dir / "latest_frame.png"),
        "latest_frame_annotated": str(workspace_dir / "latest_frame_annotated.png"),
        "turn_context_json": str(workspace_dir / "turn_context.json"),
        "turn_plan_json": str(workspace_dir / "turn_plan.json"),
    },
}

events = [
    {"type": "session", "id": "session-cycle", "cwd": str(workspace_dir)},
    {
        "type": "tool_execution_start",
        "toolCallId": "observe-1",
        "toolName": "bash",
        "args": {"command": "curl -s http://127.0.0.1:8765/agent/observe"},
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "observe-1",
        "toolName": "bash",
        "result": observe_result,
        "isError": False,
    },
]

if __INCLUDE_FRAME_READ__:
    events.extend(
        [
            {
                "type": "tool_execution_start",
                "toolCallId": "frame-1",
                "toolName": "read",
                "args": {"path": str(workspace_dir / "latest_frame_annotated.png")},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "frame-1",
                "toolName": "read",
                "result": {"ok": True},
                "isError": False,
            },
        ]
    )

events.extend(
    [
        {
            "type": "tool_execution_start",
            "toolCallId": "plan-1",
            "toolName": "bash",
            "args": {"command": "PORT=8765 bash agent_curl.sh /agent/plan <<'JSON'\\n{}\\nJSON"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "plan-1",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "act-1",
            "toolName": "bash",
            "args": {"command": "PORT=8765 bash agent_curl.sh /agent/act <<'JSON'\\n{}\\nJSON"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "act-1",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        },
        {
            "type": "tool_execution_start",
            "toolCallId": "observe-2",
            "toolName": "bash",
            "args": {"command": "curl -s http://127.0.0.1:8765/agent/observe"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "observe-2",
            "toolName": "bash",
            "result": observe_result,
            "isError": False,
        },
    ]
)

for event in events:
    print(json.dumps(event), flush=True)
    time.sleep(0.02)

while True:
    time.sleep(1)
"""
    script.write_text(
        template.replace("__INCLUDE_FRAME_READ__", repr(include_frame_read)),
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
    assert any(event["type"] == "pi_turn_end" for event in events)
    assert any(event["type"] == "pi_text_delta" for event in streamed)
    assert any(event["type"] == "pi_prompt_sent" for event in streamed)


@pytest.mark.asyncio
async def test_pi_supervisor_retains_full_tool_history_for_session(tmp_path: Path):
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
async def test_pi_supervisor_stages_agent_curl_in_workspace(tmp_path: Path):
    fake_pi = make_fake_pi_script(tmp_path)
    workspace_dir = tmp_path / "workspace"

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    staged = workspace_dir / "agent_curl.sh"
    assert staged.is_file()
    assert "PORT:-8765" in staged.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_pi_supervisor_flags_turns_that_skip_annotated_frame_reads(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "latest_frame_annotated.png").write_bytes(b"annotated-frame")
    (workspace_dir / "latest_frame.png").write_bytes(b"raw-frame")

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    vision = snapshot["vision"]["last_turn"]
    assert vision["annotated_available"] is True
    assert vision["annotated_read"] is False
    assert vision["compliant"] is False
    assert snapshot["vision"]["violations"] == 1
    violation_event = next(event for event in events if event["type"] == "pi_turn_vision_violation")
    assert "latest_frame_annotated.png" in violation_event["summary"]


@pytest.mark.asyncio
async def test_pi_supervisor_records_successful_annotated_frame_reads(tmp_path: Path):
    events: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path, include_frame_read=True)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "latest_frame_annotated.png").write_bytes(b"annotated-frame")
    (workspace_dir / "latest_frame.png").write_bytes(b"raw-frame")

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    vision = snapshot["vision"]["last_turn"]
    assert vision["annotated_read"] is True
    assert vision["used_vision"] is True
    assert vision["compliant"] is True
    assert snapshot["vision"]["violations"] == 0
    assert any(event["type"] == "pi_vision_read" for event in events)


@pytest.mark.asyncio
async def test_pi_supervisor_preserves_vision_state_during_gameplay_cycle(tmp_path: Path):
    events: list[dict] = []
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "latest_frame_annotated.png").write_bytes(b"annotated-frame")
    (workspace_dir / "latest_frame.png").write_bytes(b"raw-frame")

    async def sink(event: dict) -> None:
        events.append(event)

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        event_sink=sink,
    )

    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "observe-1",
            "toolName": "bash",
            "args": {"command": "curl -s http://127.0.0.1:8765/agent/observe"},
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "observe-1",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "frame-1",
            "toolName": "read",
            "args": {"path": str(workspace_dir / "latest_frame_annotated.png")},
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "frame-1",
            "toolName": "read",
            "result": {"ok": True},
            "isError": False,
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "plan-1",
            "toolName": "bash",
            "args": {"command": "PORT=8765 bash agent_curl.sh /agent/plan <<'JSON'\\n{}\\nJSON"},
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "plan-1",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "act-1",
            "toolName": "bash",
            "args": {"command": "PORT=8765 bash agent_curl.sh /agent/act <<'JSON'\\n{}\\nJSON"},
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "act-1",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "observe-2",
            "toolName": "bash",
            "args": {"command": "curl -s http://127.0.0.1:8765/agent/observe"},
        }
    )
    await supervisor._handle_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "observe-2",
            "toolName": "bash",
            "result": {"success": True},
            "isError": False,
        }
    )

    snapshot = supervisor.state_snapshot()
    vision = snapshot["vision"]["current_turn"]
    assert snapshot["turns_completed"] == 0
    assert vision["annotated_read"] is True
    assert vision["used_vision"] is True
    assert not any(event["type"] == "pi_turn_end" for event in events)


@pytest.mark.asyncio
async def test_pi_supervisor_keeps_running_until_agent_stops(tmp_path: Path):
    fake_pi = make_fake_cycle_pi_script(tmp_path, include_frame_read=True)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "latest_frame_annotated.png").write_bytes(b"annotated-frame")
    (workspace_dir / "latest_frame.png").write_bytes(b"raw-frame")

    supervisor = PiSupervisor(
        workspace_dir=workspace_dir,
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    for _ in range(20):
        if supervisor.state_snapshot()["vision"]["current_turn"]["annotated_read"]:
            break
        await asyncio.sleep(0.05)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "running"
    assert snapshot["turns_completed"] == 0
    assert snapshot["vision"]["current_turn"]["annotated_read"] is True

    await supervisor.stop()
    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "stopped"


@pytest.mark.asyncio
async def test_pi_supervisor_auto_continue_warns_after_vision_violation(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_pi_script(tmp_path)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "latest_frame_annotated.png").write_bytes(b"annotated-frame")
    (workspace_dir / "latest_frame.png").write_bytes(b"raw-frame")

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
        max_turns=2,
        continue_delay_seconds=0,
    )
    await supervisor.wait_until_idle(timeout=5)

    prompt_events = [event for event in streamed if event["type"] == "pi_prompt_sent"]
    assert len(prompt_events) == 2
    assert "violated the frame-inspection policy" in prompt_events[1]["prompt"]
    assert (
        "Do not call /agent/plan or /agent/act until both frame reads succeed."
        in prompt_events[1]["prompt"]
    )


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
async def test_pi_supervisor_completes_text_only_turn_without_tool_calls(tmp_path: Path):
    fake_pi = make_fake_pi_script(
        tmp_path,
        include_turn_end=False,
        include_write_tool=False,
        linger_after_events=True,
    )
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(goal="Reach the next checkpoint.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 1
    assert snapshot["last_assistant_text"] == "Initial turn."
    assert snapshot["counts"]["tool_calls"] == 0
    assert snapshot["recent_tools"] == []


@pytest.mark.asyncio
async def test_pi_supervisor_auto_continue_advances_after_text_only_message(tmp_path: Path):
    streamed: list[dict] = []
    fake_pi = make_fake_pi_script(
        tmp_path,
        include_turn_end=False,
        include_write_tool=False,
        linger_after_events=True,
    )

    async def stream(event: dict) -> None:
        streamed.append(event)

    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
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

    prompt_events = [event for event in streamed if event["type"] == "pi_prompt_sent"]
    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 2
    assert snapshot["continue_count"] == 1
    assert snapshot["counts"]["tool_calls"] == 0
    assert snapshot["last_assistant_text"] == "Continued turn."
    assert len(prompt_events) == 2
    assert prompt_events[1]["resume"] is True
    assert prompt_events[1]["prompt"] == CONTINUE_PROMPT


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


def test_continue_prompt_mentions_dialog_action_without_single_cycle_language() -> None:
    assert "a_until_dialog_end" in CONTINUE_PROMPT
    assert "one gameplay cycle" not in CONTINUE_PROMPT
    assert "recent_action.plan_state" not in CONTINUE_PROMPT
