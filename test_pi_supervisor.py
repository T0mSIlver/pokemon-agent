from pathlib import Path

import pytest

from pokemon_agent.pi_supervisor import PiSupervisor


def make_fake_pi_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake-pi"
    script.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

resume = "--continue" in sys.argv
session_dir = pathlib.Path(sys.argv[sys.argv.index("--session-dir") + 1])
session_dir.mkdir(parents=True, exist_ok=True)
workspace_dir = session_dir.parent
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
        },
    },
    {
        "type": "turn_end",
        "turnIndex": 1,
        "message": {"role": "assistant", "content": [{"type": "text", "text": message}]},
        "toolResults": [],
    },
    {"type": "agent_end", "messages": []},
]

for event in events:
    print(json.dumps(event), flush=True)
""",
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

    await supervisor.start(prompt="Play the game.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["turns_completed"] == 1
    assert snapshot["session_id"] == "session-123"
    assert snapshot["last_assistant_text"] == "Initial turn."
    assert "Inspecting the frame." in snapshot["last_assistant_thinking"]
    assert snapshot["turn_plan_preview"]["payload"]["objective_id"] == "get_oaks_parcel"
    assert snapshot["recent_tools"][-1]["tool_name"] == "write"
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
async def test_pi_supervisor_can_continue_existing_session(tmp_path: Path):
    fake_pi = make_fake_pi_script(tmp_path)
    supervisor = PiSupervisor(
        workspace_dir=tmp_path / "workspace",
        server_url="http://127.0.0.1:8765",
        pi_binary=str(fake_pi),
    )

    await supervisor.start(prompt="Play the game.", auto_continue=False)
    await supervisor.wait_until_idle(timeout=5)
    await supervisor.continue_once(message="continue")
    await supervisor.wait_until_idle(timeout=5)

    snapshot = supervisor.state_snapshot()
    assert snapshot["turns_completed"] == 2
    assert snapshot["last_assistant_text"] == "Continued turn."
    assert snapshot["status"] == "completed"


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
        prompt="Play the game.",
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
