"""Async Pi supervisor driving a single long-lived ``pi --mode rpc`` process."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shutil
import time
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union
from urllib.parse import urlparse

from pokemon_agent.agent_runtime import utc_now
from pokemon_agent.run_recorder import RunRecorder

JsonDict = dict[str, Any]
EventSink = Callable[[JsonDict], Awaitable[None]]
StreamSink = Callable[[JsonDict], Awaitable[None]]
ObjectiveCheck = Callable[[], Union[bool, Awaitable[bool]]]
ArtifactPathProvider = Callable[[], dict[str, Any]]
CriticContextProvider = Callable[[], Union[dict[str, Any], Awaitable[dict[str, Any]]]]

DEFAULT_TOOLS = ["read", "bash", "edit", "write"]
FRAME_IMAGE_FILES = ("latest_frame_annotated.png", "latest_frame.png")
STREAM_ENTRY_CAP = 5000

# Text the operator can open in the dashboard is never silently shortened. Short
# previews stay short - a headline has one line to work with - but every preview
# ships beside the whole text in the same payload, so expanding always reaches all
# of it. The ceilings below are only there so one runaway stdout cannot pin the
# browser, and when one does bite, TRUNCATION_NOTE says how much is missing.
STREAM_HEADLINE_LIMIT = 120
STREAM_COMMENT_HEADLINE_LIMIT = 240
#: Whole thinking / text / critique bodies on stream entries. The old 8k ceiling cut
#: xhigh critic reasoning off part-way through, which is exactly the loss to avoid.
STREAM_TEXT_LIMIT = 120_000
#: Whole tool results, carried on ``result_full`` and expandable in the dashboard.
TOOL_RESULT_LIMIT = 200_000
#: Transcript entries keep a one-line ``preview`` and the whole thing in ``content``.
TRANSCRIPT_TEXT_LIMIT = 120_000
#: One stdout / stderr line. Unbounded at the source, so this cap has to stay; the
#: marker makes the loss visible instead of silent.
STDERR_LINE_LIMIT = 20_000
#: Preview of an stderr line in ``stderr_tail``. The transcript holds it whole.
STDERR_TAIL_PREVIEW = 400
#: Error strings on the snapshot. An error is read closely, not skimmed, so this is
#: generous: a truncated stack trace costs more to debug than it saves in bytes.
ERROR_TEXT_LIMIT = 4_000
#: Appended by :func:`_clip_text` when a ceiling bites, in place of a bare "...".
TRUNCATION_NOTE = "\n\n…[truncated: {dropped:,} more characters not shown]"
STREAM_ENTRY_EVENT = "pi_stream_entry"
STREAM_DEFAULT_PAGE = 1000
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
FILE_TOOL_NAMES = frozenset({"read", "write", "edit"})
DEFAULT_ARTIFACT_FILES = {
    "latest_frame": "latest_frame.png",
    "latest_frame_annotated": "latest_frame_annotated.png",
    "live_frame": "live_frame.png",
    "live_frame_annotated": "live_frame_annotated.png",
    "turn_context_json": "turn_context.json",
}
DEFAULT_TOKEN_BUDGET = 110_000
DEFAULT_STATS_POLL_SECONDS = 30.0
CONTINUE_MESSAGE = "continue"
FALLBACK_GOAL = "Begin."

# Where the goal for a session came from, most specific first. Reported on the
# snapshot so the dashboard can say why the agent was told to do what it was told.
GOAL_SOURCE_OPERATOR = "operator"
GOAL_SOURCE_CRITIC = "critic"
GOAL_SOURCE_OBJECTIVE = "objective"
GOAL_SOURCE_FALLBACK = "fallback"
_STATUS_STREAM_LEVELS = {"error": "error", "stuck": "warn"}
STREAM_CHUNK_SIZE = 65536
#: Marks the between-session critic's own narration in the shared stream log.
CRITIC_STREAM_PREFIX = "[critic] "
CRITIC_HEARTBEAT_SECONDS = 15.0
ORPHAN_TOOL_RESULT_TEXT = (
    "Tool call interrupted: the supervisor stopped Pi before this tool returned a result."
)
#: Longest operator message accepted by :meth:`PiSupervisor.send_operator_message`.
#: A nudge, not an essay: anything longer is a new goal and belongs in a restart.
OPERATOR_MESSAGE_LIMIT = 400
#: How many recent operator messages ``state_snapshot`` reports to the dashboard.
OPERATOR_MESSAGE_HISTORY = 20
#: ``source`` marker on stream entries a human typed, as opposed to the harness.
OPERATOR_STREAM_SOURCE = "operator"
#: ``source`` marker on the one instruction an intervention hands the player. It
#: arrives down the operator path but nobody typed it, and the transcript has to
#: be able to tell those apart afterwards.
INTERVENTION_STREAM_SOURCE = "intervention"
#: An intervention's answer is a plan for a segment, not a nudge, so it gets more
#: room than a human's steer. Still one instruction, still capped.
INTERVENTION_MESSAGE_LIMIT = 1200
#: Statuses where a ``pi --mode rpc`` process is live enough to take a message.
STEERABLE_STATUSES = frozenset({"starting", "running"})


class NoLiveSessionError(RuntimeError):
    """Raised when an operator message arrives with no live Pi session to carry it."""


def _critic_module():
    """Deferred import: ``pokemon_agent.critic`` imports this module for its parsers."""

    from pokemon_agent import critic

    return critic


async def iter_stream_lines(stream: Any):
    """Yield strict JSONL records: split on ``\\n`` only, tolerating a trailing ``\\r``."""

    buffer = b""
    while True:
        chunk = await stream.read(STREAM_CHUNK_SIZE)
        if not chunk:
            break
        buffer += chunk
        while True:
            index = buffer.find(b"\n")
            if index == -1:
                break
            raw, buffer = buffer[:index], buffer[index + 1 :]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            yield raw.decode("utf-8", errors="replace")
    if buffer:
        if buffer.endswith(b"\r"):
            buffer = buffer[:-1]
        yield buffer.decode("utf-8", errors="replace")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_skill_path() -> Path:
    return _repo_root() / "skill" / "SKILL.md"


def _server_port(server_url: str) -> Optional[int]:
    try:
        parsed = urlparse(server_url)
    except Exception:  # noqa: BLE001
        return None
    return parsed.port


def _truncate(value: str, limit: int = 320) -> str:
    """A one-line preview. Only ever for a headline, label or summary field.

    Never the sole copy of anything: whatever this shortens must also travel whole,
    in the same payload, on a field the dashboard can expand.
    """

    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _clip_text(value: str, limit: int = TRANSCRIPT_TEXT_LIMIT) -> str:
    """The whole text, unless it is absurd - and then it says what it dropped."""

    text = value.strip()
    if len(text) <= limit:
        return text
    kept = text[:limit].rstrip()
    return kept + TRUNCATION_NOTE.format(dropped=len(text) - len(kept))


def _utc_after(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))).isoformat()


def _collect_text(value: Any) -> list[str]:
    parts: list[str] = []
    if value is None:
        return parts
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            parts.append(stripped)
        return parts
    if isinstance(value, list):
        for item in value:
            parts.extend(_collect_text(item))
        return parts
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("delta")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                else:
                    parts.extend(_collect_text(item))
        for key in ("text", "delta", "message", "reason", "output"):
            child = value.get(key)
            if isinstance(child, (str, list, dict)):
                parts.extend(_collect_text(child))
        return parts
    return parts


def extract_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    text_parts: list[str] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def extract_message_thinking(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    thinking_parts: list[str] = []
    for item in message.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "thinking":
            text = item.get("thinking") or item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                thinking_parts.append(text.strip())
    return "\n".join(thinking_parts).strip()


def preview_payload(value: Any, limit: int = 260) -> str:
    """Headline-length preview of a payload. :func:`payload_text` carries it whole."""

    text = "\n".join(_collect_text(value)).strip()
    if not text:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            text = repr(value)
    return _truncate(text, limit)


def payload_text(value: Any, limit: int = TOOL_RESULT_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)
        except TypeError:
            text = repr(value)
    return _clip_text(text, limit)


def parse_compact_token_count(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*([\d.]+)\s*([KMB])?\s*$", value)
    if not match:
        return None
    amount = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = 1
    if suffix == "K":
        multiplier = 1_000
    elif suffix == "M":
        multiplier = 1_000_000
    elif suffix == "B":
        multiplier = 1_000_000_000
    return int(amount * multiplier)


def normalize_model_lookup(
    provider: Optional[str],
    model: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    chosen_provider = (provider or "").strip() or None
    chosen_model = (model or "").strip()
    if not chosen_model:
        return chosen_provider, None
    if "/" in chosen_model:
        inferred_provider, chosen_model = chosen_model.split("/", 1)
        chosen_provider = chosen_provider or inferred_provider.strip() or None
    if ":" in chosen_model:
        base_model, maybe_thinking = chosen_model.rsplit(":", 1)
        if maybe_thinking in {"off", "minimal", "low", "medium", "high", "xhigh"}:
            chosen_model = base_model
    chosen_model = chosen_model.strip() or None
    return chosen_provider, chosen_model


def parse_model_limits_output(
    output: str,
    *,
    provider: Optional[str],
    model: Optional[str],
) -> Optional[JsonDict]:
    requested_provider, requested_model = normalize_model_lookup(provider, model)
    rows: list[JsonDict] = []
    for raw_line in output.split("\n"):
        line = raw_line.strip()
        if not line or line.lower().startswith("provider "):
            continue
        parts = [part for part in re.split(r"\s{2,}|\t+", line) if part]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "provider": parts[0],
                "model": parts[1],
                "context_window": parts[2],
                "context_window_tokens": parse_compact_token_count(parts[2]),
                "max_output": parts[3],
                "max_output_tokens": parse_compact_token_count(parts[3]),
                "thinking": parts[4],
                "images": parts[5],
            }
        )

    if not rows:
        return None

    if requested_model:
        for row in rows:
            if row["model"] == requested_model and (
                not requested_provider or row["provider"] == requested_provider
            ):
                return row

    if requested_provider:
        for row in rows:
            if row["provider"] == requested_provider:
                return row

    return rows[0]


def extract_leading_comment(command: str) -> str:
    """Join the ``#`` comment block the model writes above a bash command."""

    lines = command.split("\n")
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and lines[index].lstrip().startswith("#!"):
        index += 1
    parts: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("#"):
            break
        text = stripped.lstrip("#").strip()
        if text:
            parts.append(text)
        index += 1
    return " ".join(parts)


def first_nonempty_line(text: str) -> str:
    for raw in text.split("\n"):
        line = raw.strip()
        if line:
            return line
    return ""


def command_headline(command: str) -> str:
    comment = extract_leading_comment(command)
    if comment:
        return _truncate(comment, STREAM_COMMENT_HEADLINE_LIMIT)
    return _truncate(first_nonempty_line(command), STREAM_HEADLINE_LIMIT)


def format_byte_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    kilobytes = size / 1024
    if kilobytes < 1024:
        return f"{kilobytes:.1f} KB"
    return f"{kilobytes / 1024:.1f} MB"


def tool_output_text(value: Any) -> str:
    """Best-effort plain text for a tool result, for line/JSON summarising."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("output", "stdout", "text", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    collected = "\n".join(_collect_text(value)).strip()
    if collected:
        return collected
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return repr(value)


def summarize_position_json(text: str) -> Optional[str]:
    """Turn an ``/action`` style JSON body into ``x=11 y=34 facing=up hp=15/30``."""

    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not any(key in payload for key in ("x", "y", "facing")):
        return None
    parts = [
        f"{key}={payload[key]}"
        for key in ("x", "y", "facing", "hp")
        if payload.get(key) is not None
    ]
    return " ".join(parts) or None


def extract_file_hint(args: Any) -> Optional[str]:
    if not isinstance(args, dict):
        return None
    for key in ("path", "filePath", "file_path", "target_path", "targetPath"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_turn_plan_candidate(args: Any) -> Optional[JsonDict]:
    if not isinstance(args, dict):
        return None
    path = extract_file_hint(args)
    if not path or not path.endswith("turn_plan.json"):
        return None
    for key in ("content", "text", "newText", "new_text"):
        raw = args.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Unparseable, so the operator has to read it: keep it whole.
                return {"raw": _clip_text(raw, ERROR_TEXT_LIMIT)}
            if isinstance(parsed, dict):
                return parsed
    return {"path": path}


def iter_jsonl_records(text: str) -> list[JsonDict]:
    """Split strict JSONL on ``\\n`` only, never on U+2028/U+2029/\\x0b/\\x0c."""

    records: list[JsonDict] = []
    for raw in text.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def find_orphaned_tool_calls(entries: list[JsonDict]) -> list[tuple[str, str]]:
    """Return ``(tool_call_id, tool_name)`` pairs that never received a tool result."""

    pending: dict[str, str] = {}
    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                call_id = block.get("id")
                if isinstance(call_id, str) and call_id:
                    pending[call_id] = str(block.get("name") or "unknown")
        elif role == "toolResult":
            call_id = message.get("toolCallId")
            if isinstance(call_id, str):
                pending.pop(call_id, None)
    return list(pending.items())


def repair_orphaned_tool_calls(session_file: Path) -> list[str]:
    """Append synthetic error tool results for tool calls that never got one.

    Killing Pi mid-turn leaves assistant messages whose ``toolCall`` blocks have no
    matching ``toolResult``. Replaying that history re-sends the malformed pair to the
    provider on every later turn, so close the holes on disk before resuming.
    """

    try:
        text = session_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    entries = iter_jsonl_records(text)
    orphans = find_orphaned_tool_calls(entries)
    if not orphans:
        return []

    parent_id: Optional[str] = None
    for entry in entries:
        entry_id = entry.get("id")
        if entry.get("type") != "session" and isinstance(entry_id, str):
            parent_id = entry_id

    now_iso = utc_now()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    appended: list[str] = []
    lines: list[str] = []
    for call_id, tool_name in orphans:
        entry_id = uuid.uuid4().hex[:8]
        lines.append(
            json.dumps(
                {
                    "type": "message",
                    "id": entry_id,
                    "parentId": parent_id,
                    "timestamp": now_iso,
                    "message": {
                        "role": "toolResult",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "content": [{"type": "text", "text": ORPHAN_TOOL_RESULT_TEXT}],
                        "isError": True,
                        "timestamp": now_ms,
                    },
                },
                ensure_ascii=False,
            )
        )
        parent_id = entry_id
        appended.append(call_id)

    prefix = "" if not text or text.endswith("\n") else "\n"
    try:
        with session_file.open("a", encoding="utf-8") as handle:
            handle.write(prefix + "\n".join(lines) + "\n")
    except OSError:
        return []
    return appended


class PiSupervisor:
    """Drives one long-lived ``pi --mode rpc`` process and exposes dashboard telemetry."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        server_url: str,
        event_sink: Optional[EventSink] = None,
        stream_sink: Optional[StreamSink] = None,
        repo_root: Optional[Path] = None,
        pi_binary: Optional[str] = None,
        max_idle_turns: int = 3,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        stats_poll_seconds: Optional[float] = DEFAULT_STATS_POLL_SECONDS,
        objective_complete: Optional[ObjectiveCheck] = None,
        artifact_paths: Optional[ArtifactPathProvider] = None,
        critic_enabled: bool = True,
        critic_context: Optional[CriticContextProvider] = None,
        critic_timeout_seconds: Optional[float] = None,
        critic_thinking: Optional[str] = None,
        critic_retry_enabled: bool = True,
        critic_retry_thinking: Optional[str] = None,
        critic_heartbeat_seconds: float = CRITIC_HEARTBEAT_SECONDS,
        run_recorder: Optional[RunRecorder] = None,
    ) -> None:
        self.max_idle_turns = max_idle_turns if max_idle_turns and max_idle_turns > 0 else 0
        self.token_budget = token_budget if token_budget and token_budget > 0 else 0
        self.stats_poll_seconds = (
            float(stats_poll_seconds) if stats_poll_seconds and stats_poll_seconds > 0 else 0.0
        )
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.server_url = server_url
        self.repo_root = (repo_root or _repo_root()).expanduser().resolve()
        self.skill_path = default_skill_path().expanduser().resolve()
        self.pi_binary = pi_binary or shutil.which("pi")
        self.event_sink = event_sink
        self.stream_sink = stream_sink
        self.objective_complete = objective_complete
        self.artifact_paths = artifact_paths
        #: The scoreboard's writer. The supervisor owns the run *lifecycle*: it is
        #: the only thing that knows a playthrough has begun and what it is for.
        #: The server owns the receipts, because only it sees an action batch.
        self.run_recorder = run_recorder
        self.run_id: Optional[str] = None
        self.run_adopted: bool = False
        self.run_error: Optional[str] = None
        #: Last thing the intervention loop did, mirrored here so a swap of the
        #: player's KV cache is visible where the run is watched from.
        self.intervention_state: JsonDict = {
            "enabled": False,
            "fired": 0,
            "delivered": 0,
            "last": None,
            "slot_lost": None,
            "disabled_reason": None,
        }
        self.session_dir = self.workspace_dir / "pi-session"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        critic = _critic_module()
        self.critic_enabled = bool(critic_enabled)
        self.critic_context = critic_context
        self.critic_timeout_seconds = float(
            critic_timeout_seconds
            if critic_timeout_seconds and critic_timeout_seconds > 0
            else critic.DEFAULT_CRITIC_TIMEOUT_SECONDS
        )
        self.critic_thinking = critic_thinking or critic.DEFAULT_CRITIC_THINKING
        self.critic_retry_enabled = bool(critic_retry_enabled)
        self.critic_retry_thinking = critic_retry_thinking or critic.DEFAULT_CRITIC_RETRY_THINKING
        self.critic_heartbeat_seconds = max(0.0, float(critic_heartbeat_seconds or 0.0))
        self.last_critique: Optional[str] = None
        self.last_critique_at: Optional[str] = None
        self.last_critique_seconds: Optional[float] = None
        self.last_critique_error: Optional[str] = None
        self.last_critique_tokens: Optional[int] = None
        self.last_critique_stop_reason: Optional[str] = None
        self.last_critique_usage: Optional[JsonDict] = None
        self.last_critique_salvaged: bool = False
        self.last_critique_raw_path: Optional[str] = None
        self.last_critique_attempts: list[JsonDict] = []

        self.status = "idle"
        self.status_reason = "Pi supervisor is idle."
        self.available = self.pi_binary is not None
        self.last_error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.last_event_at: Optional[str] = None
        self.last_turn_started_at: Optional[str] = None
        self.last_turn_completed_at: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_file: Optional[Path] = None
        self.current_pid: Optional[int] = None
        self.turns_completed = 0
        self.continue_count = 0
        self.auto_continue = False
        self.goal = ""
        #: What the operator handed to :meth:`start`, kept apart from the resolved
        #: goal so a restart without one can fall back instead of re-pinning it.
        self.operator_goal = ""
        #: The forward-looking goal the last critique wrote, for the next session.
        self.critic_next_goal = ""
        self.goal_source = GOAL_SOURCE_FALLBACK
        self.continue_delay_seconds = 1.0
        self.max_turns: Optional[int] = None
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.thinking: Optional[str] = None
        self.current_prompt = ""
        self.last_prompt = ""
        self.default_prompt = FALLBACK_GOAL
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.last_assistant_text = ""
        self.last_assistant_thinking = ""
        self.latest_turn_summary = ""
        self.current_tool_calls: dict[str, JsonDict] = {}
        self.recent_tools: list[JsonDict] = []
        self.recent_events: deque[JsonDict] = deque(maxlen=120)
        self.stderr_tail: deque[str] = deque(maxlen=30)
        self.transcript: deque[JsonDict] = deque(maxlen=160)
        self.stream_entries: list[JsonDict] = []
        self.operator_messages: deque[JsonDict] = deque(maxlen=OPERATOR_MESSAGE_HISTORY)
        self.turn_plan_preview: Optional[JsonDict] = None
        self.next_auto_continue_at: Optional[str] = None
        self.session_usage: Optional[JsonDict] = None
        self.last_message_usage: Optional[JsonDict] = None
        self.context_usage: Optional[JsonDict] = None
        self.model_limits: Optional[JsonDict] = None
        self.tool_call_count: int = 0
        self.thinking_block_count: int = 0
        self.assistant_message_count: int = 0
        self.user_message_count: int = 0
        self.repaired_tool_calls: int = 0
        self.last_compaction_tokens_before: Optional[int] = None
        self.last_compaction_tokens_after: Optional[int] = None
        self.last_compaction_at: Optional[str] = None
        self._pending_thinking_in_message: bool = False
        self._model_limits_cache: dict[str, Optional[JsonDict]] = {}
        self._stream_seq: int = 0
        self._stream_by_seq: dict[int, JsonDict] = {}
        self._stream_text_buffers: dict[int, str] = {}
        self._tool_stream_seq: dict[str, int] = {}
        self._tool_started_monotonic: dict[str, float] = {}
        self._active_thinking_seq: Optional[int] = None
        self._active_text_seq: Optional[int] = None
        self._message_saw_thinking_delta: bool = False
        self._message_saw_text_delta: bool = False

        self._task: Optional[asyncio.Task[None]] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_tasks: list[asyncio.Task[None]] = []
        self._stats_task: Optional[asyncio.Task[None]] = None
        self._pending_responses: dict[str, asyncio.Future[JsonDict]] = {}
        self._request_counter = 0
        self._settled_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._stop_requested = False
        self._budget_stop_requested = False
        self._current_turn_completed = False
        self._current_cycle_has_tool_call = False
        self._consecutive_idle_turns = 0
        self._session_start_context: JsonDict = {}
        self._critic_process: Optional[asyncio.subprocess.Process] = None
        self._critic_cancelled = False
        self._critic_thinking_seq: Optional[int] = None
        self._critic_text_seq: Optional[int] = None
        self._critic_heartbeat_seq: Optional[int] = None
        self._critic_heartbeat_task: Optional[asyncio.Task[None]] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def set_objective_complete(self, callback: Optional[ObjectiveCheck]) -> None:
        """Register the predicate the server uses to declare the objective finished."""

        self.objective_complete = callback

    def _config_snapshot(self) -> JsonDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking": self.thinking,
            "auto_continue": self.auto_continue,
            "goal": self.goal,
            "goal_source": self.goal_source,
            "continue_delay_seconds": self.continue_delay_seconds,
            "max_turns": self.max_turns,
            "skill_path": str(self.skill_path),
            "session_dir": str(self.session_dir),
            "server_url": self.server_url,
            "tools": DEFAULT_TOOLS,
            "max_idle_turns": self.max_idle_turns,
            "token_budget": self.token_budget,
            "stats_poll_seconds": self.stats_poll_seconds,
            "critic_enabled": self.critic_enabled,
            "critic_timeout_seconds": self.critic_timeout_seconds,
            "critic_thinking": self.critic_thinking,
            "critic_retry_enabled": self.critic_retry_enabled,
            "critic_retry_thinking": self.critic_retry_thinking,
        }

    def state_snapshot(self) -> JsonDict:
        return {
            "available": self.available,
            "pi_binary": self.pi_binary,
            "status": self.status,
            "status_reason": self.status_reason,
            "provider": self.provider,
            "model": self.model,
            "thinking": self.thinking,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "last_turn_started_at": self.last_turn_started_at,
            "last_turn_completed_at": self.last_turn_completed_at,
            "session_id": self.session_id,
            "session_file": str(self.session_file) if self.session_file else None,
            "session_dir": str(self.session_dir),
            "skill_path": str(self.skill_path),
            "server_url": self.server_url,
            "current_pid": self.current_pid,
            "turns_completed": self.turns_completed,
            "continue_count": self.continue_count,
            "goal": self.goal,
            "goal_source": self.goal_source,
            "operator_goal": self.operator_goal,
            "critic_next_goal": self.critic_next_goal,
            "current_prompt": self.current_prompt,
            "last_prompt": self.last_prompt,
            "default_prompt": self.default_prompt,
            "current_assistant_text": self.current_assistant_text,
            "current_assistant_thinking": self.current_assistant_thinking,
            "last_assistant_text": self.last_assistant_text,
            "last_assistant_thinking": self.last_assistant_thinking,
            "latest_turn_summary": self.latest_turn_summary,
            "active_tools": list(self.current_tool_calls.values()),
            "recent_tools": list(self.recent_tools),
            "recent_events": list(self.recent_events),
            "stderr_tail": list(self.stderr_tail),
            "transcript": list(self.transcript),
            "stream": list(self.stream_entries),
            "operator_messages": list(self.operator_messages),
            "turn_plan_preview": self.turn_plan_preview,
            "next_auto_continue_at": self.next_auto_continue_at,
            "session_usage": self.session_usage,
            "last_message_usage": self.last_message_usage,
            "context_usage": self.context_usage,
            "model_limits": self.model_limits,
            "counts": {
                "tool_calls": self.tool_call_count,
                "thinking_blocks": self.thinking_block_count,
                "assistant_messages": self.assistant_message_count,
                "user_messages": self.user_message_count,
                "repaired_tool_calls": self.repaired_tool_calls,
            },
            "compaction": {
                "tokens_before": self.last_compaction_tokens_before,
                "tokens_after": self.last_compaction_tokens_after,
                "at": self.last_compaction_at,
            },
            "critique": {
                "enabled": self.critic_enabled,
                "text": self.last_critique,
                "at": self.last_critique_at,
                "duration_seconds": self.last_critique_seconds,
                "digest_tokens": self.last_critique_tokens,
                "error": self.last_critique_error,
                "stop_reason": self.last_critique_stop_reason,
                "usage": self.last_critique_usage,
                "salvaged": self.last_critique_salvaged,
                "next_goal": self.critic_next_goal,
                "raw_path": self.last_critique_raw_path,
                "attempts": list(self.last_critique_attempts),
                "handoff_path": str(_critic_module().handoff_path(self.workspace_dir)),
            },
            "run": self.run_snapshot(),
            "interventions": dict(self.intervention_state),
            "config": self._config_snapshot(),
        }

    def run_snapshot(self) -> JsonDict:
        """Which run the sessions are adding up to, and what it has cost."""

        recorder = self.run_recorder
        payload: JsonDict = {
            "run_id": self.run_id,
            "adopted": self.run_adopted,
            "error": self.run_error,
        }
        if recorder is not None:
            payload.update(recorder.status())
        return payload

    # ------------------------------------------------------------------
    # Ordered stream log
    # ------------------------------------------------------------------

    def stream_since(
        self,
        after: int = 0,
        limit: int = STREAM_DEFAULT_PAGE,
    ) -> JsonDict:
        """Return stream entries newer than ``after``, oldest first."""

        after = max(0, int(after))
        limit = max(1, min(int(limit), STREAM_ENTRY_CAP))
        entries = [entry for entry in self.stream_entries if entry["seq"] > after][:limit]
        next_seq = entries[-1]["seq"] if entries else after
        return {
            "entries": deepcopy(entries),
            "next_seq": next_seq,
            "session_id": self.session_id,
        }

    def _reset_stream(self) -> None:
        self.stream_entries.clear()
        self._stream_by_seq.clear()
        self._stream_text_buffers.clear()
        self._tool_stream_seq.clear()
        self._tool_started_monotonic.clear()
        self._stream_seq = 0
        self._active_thinking_seq = None
        self._active_text_seq = None
        self._critic_thinking_seq = None
        self._critic_text_seq = None
        self._critic_heartbeat_seq = None
        self._message_saw_thinking_delta = False
        self._message_saw_text_delta = False

    def _new_stream_entry(
        self,
        kind: str,
        *,
        state: str = "ok",
        text: str = "",
        tool: Optional[JsonDict] = None,
        system: Optional[JsonDict] = None,
        source: str = "agent",
    ) -> JsonDict:
        self._stream_seq += 1
        entry: JsonDict = {
            "seq": self._stream_seq,
            "ts": utc_now(),
            "kind": kind,
            "state": state,
            "source": source,
            "text": text,
            "tool": tool,
            "system": system,
        }
        self.stream_entries.append(entry)
        self._stream_by_seq[entry["seq"]] = entry
        overflow = len(self.stream_entries) - STREAM_ENTRY_CAP
        if overflow > 0:
            for dropped in self.stream_entries[:overflow]:
                self._stream_by_seq.pop(dropped["seq"], None)
                self._stream_text_buffers.pop(dropped["seq"], None)
            del self.stream_entries[:overflow]
        self.last_event_at = entry["ts"]
        return entry

    async def _emit_stream_entry(self, entry: Optional[JsonDict]) -> None:
        if entry is None or self.stream_sink is None:
            return
        await self.stream_sink({"type": STREAM_ENTRY_EVENT, "entry": deepcopy(entry)})

    async def _push_stream_user(self, text: str, *, source: str = "agent") -> JsonDict:
        entry = self._new_stream_entry(
            "user",
            state="ok",
            text=_clip_text(text, STREAM_TEXT_LIMIT),
            source=source,
        )
        await self._emit_stream_entry(entry)
        return entry

    async def _push_stream_system(
        self,
        label: str,
        *,
        text: str = "",
        level: str = "info",
    ) -> JsonDict:
        entry = self._new_stream_entry(
            "system",
            state="error" if level == "error" else "ok",
            text=_clip_text(text or label, STREAM_TEXT_LIMIT),
            system={"label": _truncate(label, 160), "level": level},
        )
        await self._emit_stream_entry(entry)
        return entry

    # -- artifact lookup ------------------------------------------------

    def _artifact_path_map(self) -> dict[str, Path]:
        raw: Any = None
        provider = self.artifact_paths
        if callable(provider):
            try:
                raw = provider()
            except Exception:  # noqa: BLE001
                raw = None
        elif isinstance(provider, dict):
            raw = provider
        if not isinstance(raw, dict) or not raw:
            raw = {
                key: self.workspace_dir / filename
                for key, filename in DEFAULT_ARTIFACT_FILES.items()
            }
        resolved: dict[str, Path] = {}
        for key, value in raw.items():
            if not isinstance(value, (str, Path)):
                continue
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                resolved[str(key)] = Path(value).expanduser().resolve()
        return resolved

    def _resolve_workspace_path(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.workspace_dir / path
        try:
            return path.resolve()
        except (OSError, RuntimeError):
            return path

    def _artifact_key_for_path(self, path: Path) -> Optional[str]:
        for key, candidate in self._artifact_path_map().items():
            if candidate == path:
                return key
        return None

    # -- tool payloads --------------------------------------------------

    def _build_tool_payload(self, tool_name: Any, args: Any) -> JsonDict:
        name = str(tool_name or "tool").strip() or "tool"
        lowered = name.lower()
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str):
            command = None
        raw_path = extract_file_hint(args)
        path: Optional[Path] = self._resolve_workspace_path(raw_path) if raw_path else None

        if lowered == "bash" and command and command.strip():
            headline = command_headline(command)
        elif path is not None:
            verb = lowered if lowered in FILE_TOOL_NAMES else name
            headline = f"{verb} {path.name}"
        elif command and command.strip():
            headline = command_headline(command)
        else:
            preview = preview_payload(args, STREAM_HEADLINE_LIMIT) if args else ""
            headline = _truncate(f"{name} {preview}".strip(), STREAM_HEADLINE_LIMIT)

        image_artifact: Optional[str] = None
        if lowered == "read" and path is not None and path.suffix.lower() in IMAGE_SUFFIXES:
            image_artifact = self._artifact_key_for_path(path)

        return {
            "name": name,
            "headline": headline,
            "command": command,
            "path": str(path) if path is not None else None,
            "image_artifact": image_artifact,
            "result_summary": "",
            "result_full": "",
            "duration_ms": None,
        }

    def _fill_tool_result(self, tool: JsonDict, result: Any, *, is_error: bool) -> None:
        text = tool_output_text(result)
        tool["result_full"] = payload_text(result)
        if is_error:
            tool["result_summary"] = (
                _truncate(first_nonempty_line(text), STREAM_HEADLINE_LIMIT) or "Tool call failed."
            )
            return
        name = str(tool.get("name") or "").lower()
        raw_path = tool.get("path")
        if name == "read" and isinstance(raw_path, str):
            path = Path(raw_path)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                size: Optional[int] = None
                with contextlib.suppress(OSError):
                    size = path.stat().st_size
                tool["result_summary"] = (
                    f"image {format_byte_size(size)}" if size is not None else "image"
                )
                return
        if name == "bash":
            position = summarize_position_json(text)
            if position:
                tool["result_summary"] = position
                return
        tool["result_summary"] = _truncate(first_nonempty_line(text), STREAM_HEADLINE_LIMIT)

    async def _start_tool_stream_entry(
        self,
        tool_call_id: Any,
        tool_name: Any,
        args: Any,
    ) -> JsonDict:
        entry = self._new_stream_entry(
            "tool",
            state="running",
            tool=self._build_tool_payload(tool_name, args),
        )
        key = str(tool_call_id) if tool_call_id else f"anon-{entry['seq']}"
        self._tool_stream_seq[key] = entry["seq"]
        self._tool_started_monotonic[key] = time.monotonic()
        await self._emit_stream_entry(entry)
        return entry

    async def _seal_active_narration(self) -> None:
        """A tool call breaks the narration: later deltas start their own entry."""

        for attribute in ("_active_thinking_seq", "_active_text_seq"):
            seq = getattr(self, attribute)
            setattr(self, attribute, None)
            if seq is None:
                continue
            self._stream_text_buffers.pop(seq, None)
            entry = self._stream_by_seq.get(seq)
            if entry is None or entry.get("state") != "running":
                continue
            entry["state"] = "ok"
            await self._emit_stream_entry(entry)

    async def _finish_tool_stream_entry(
        self,
        tool_call_id: Any,
        tool_name: Any,
        result: Any,
        *,
        is_error: bool,
    ) -> Optional[JsonDict]:
        key = str(tool_call_id) if tool_call_id else None
        seq = self._tool_stream_seq.pop(key, None) if key else None
        started = self._tool_started_monotonic.pop(key, None) if key else None
        entry = self._stream_by_seq.get(seq) if seq is not None else None
        if entry is None:
            entry = self._new_stream_entry(
                "tool",
                state="running",
                tool=self._build_tool_payload(tool_name, None),
            )
        tool = entry.get("tool") or {}
        self._fill_tool_result(tool, result, is_error=is_error)
        if started is not None:
            tool["duration_ms"] = int(max(0.0, time.monotonic() - started) * 1000)
        entry["tool"] = tool
        entry["state"] = "error" if is_error else "ok"
        await self._emit_stream_entry(entry)
        return entry

    def _append_stream_text(self, seq: Optional[int], delta: str) -> Optional[JsonDict]:
        if seq is None:
            return None
        entry = self._stream_by_seq.get(seq)
        if entry is None:
            return None
        buffered = self._stream_text_buffers.get(seq, "") + delta
        self._stream_text_buffers[seq] = buffered
        entry["text"] = _clip_text(buffered, STREAM_TEXT_LIMIT)
        return entry

    async def _stream_delta(self, kind: str, delta: str) -> None:
        attribute = "_active_thinking_seq" if kind == "thinking" else "_active_text_seq"
        seen = "_message_saw_thinking_delta" if kind == "thinking" else "_message_saw_text_delta"
        seq = getattr(self, attribute)
        if seq is None or seq not in self._stream_by_seq:
            entry = self._new_stream_entry(kind, state="running")
            seq = entry["seq"]
            setattr(self, attribute, seq)
            self._stream_text_buffers[seq] = ""
        setattr(self, seen, True)
        await self._emit_stream_entry(self._append_stream_text(seq, delta))

    async def _finalize_stream_text(self, kind: str, final_text: str) -> None:
        attribute = "_active_thinking_seq" if kind == "thinking" else "_active_text_seq"
        seen = "_message_saw_thinking_delta" if kind == "thinking" else "_message_saw_text_delta"
        seq = getattr(self, attribute)
        setattr(self, attribute, None)
        saw_delta = getattr(self, seen)
        setattr(self, seen, False)
        if seq is not None:
            entry = self._stream_by_seq.get(seq)
            self._stream_text_buffers.pop(seq, None)
            if entry is not None:
                if final_text and not entry["text"]:
                    entry["text"] = _clip_text(final_text, STREAM_TEXT_LIMIT)
                entry["state"] = "ok"
                await self._emit_stream_entry(entry)
                return
        if saw_delta or not final_text:
            return
        entry = self._new_stream_entry(
            kind,
            state="ok",
            text=_clip_text(final_text, STREAM_TEXT_LIMIT),
        )
        await self._emit_stream_entry(entry)

    async def _close_open_stream_entries(self) -> None:
        """Mark anything still ``running`` when the process goes away."""

        for key in list(self._tool_stream_seq):
            seq = self._tool_stream_seq.pop(key)
            started = self._tool_started_monotonic.pop(key, None)
            entry = self._stream_by_seq.get(seq)
            if entry is None or entry.get("state") != "running":
                continue
            tool = entry.get("tool") or {}
            tool["result_summary"] = _truncate(ORPHAN_TOOL_RESULT_TEXT, STREAM_HEADLINE_LIMIT)
            tool["result_full"] = ORPHAN_TOOL_RESULT_TEXT
            if started is not None:
                tool["duration_ms"] = int(max(0.0, time.monotonic() - started) * 1000)
            entry["tool"] = tool
            entry["state"] = "error"
            await self._emit_stream_entry(entry)
        for attribute in ("_active_thinking_seq", "_active_text_seq"):
            seq = getattr(self, attribute)
            setattr(self, attribute, None)
            entry = self._stream_by_seq.get(seq) if seq is not None else None
            if entry is not None and entry.get("state") == "running":
                entry["state"] = "ok"
                await self._emit_stream_entry(entry)
        self._message_saw_thinking_delta = False
        self._message_saw_text_delta = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        goal: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        auto_continue: bool = True,
        max_turns: Optional[int] = None,
        continue_delay_seconds: float = 1.0,
        skill_path: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> JsonDict:
        if self.is_running:
            raise ValueError("Pi supervisor is already running.")
        if not self.available or not self.pi_binary:
            raise ValueError("Pi executable was not found on PATH.")

        chosen_skill = Path(skill_path).expanduser().resolve() if skill_path else self.skill_path
        if not chosen_skill.exists():
            raise ValueError(f"Pi system prompt not found: {chosen_skill}")

        self.skill_path = chosen_skill
        self.provider = provider or None
        self.model = model or None
        self.thinking = thinking or None
        self.auto_continue = bool(auto_continue)
        self.operator_goal = (goal or "").strip()
        self.max_turns = max_turns if max_turns and max_turns > 0 else None
        self.continue_delay_seconds = max(0.0, float(continue_delay_seconds))
        if token_budget is not None:
            self.token_budget = token_budget if token_budget > 0 else 0
        self._stage_workspace_helpers()
        self._session_start_context = await self._collect_critic_context()
        self.goal, self.goal_source = self._resolve_goal(self.operator_goal)

        initial_prompt = self._initial_message()
        self.default_prompt = initial_prompt

        self.status = "starting"
        self.status_reason = "Starting a fresh Pi RPC session."
        self.last_error = None
        self.started_at = utc_now()
        self.last_event_at = self.started_at
        self.current_prompt = initial_prompt
        self.last_prompt = initial_prompt
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.last_assistant_text = ""
        self.last_assistant_thinking = ""
        self.latest_turn_summary = ""
        self.current_tool_calls.clear()
        self.recent_tools.clear()
        self.recent_events.clear()
        self.stderr_tail.clear()
        self.transcript.clear()
        self.operator_messages.clear()
        self._reset_stream()
        self.turn_plan_preview = None
        self.turns_completed = 0
        self.continue_count = 0
        self.session_id = str(uuid.uuid4())
        self.session_file = None
        self.current_pid = None
        self.next_auto_continue_at = None
        self.session_usage = None
        self.last_message_usage = None
        self.context_usage = None
        self.model_limits = None
        self.tool_call_count = 0
        self.thinking_block_count = 0
        self.assistant_message_count = 0
        self.user_message_count = 0
        self.repaired_tool_calls = 0
        self.last_compaction_tokens_before = None
        self.last_compaction_tokens_after = None
        self.last_compaction_at = None
        self._pending_thinking_in_message = False
        self._consecutive_idle_turns = 0
        self._stop_requested = False
        self._budget_stop_requested = False
        self._critic_cancelled = False
        await self._refresh_model_limits()

        await self._emit_major(
            "pi_supervisor_status",
            {
                "status": self.status,
                "summary": self.status_reason,
                "config": self._config_snapshot(),
            },
        )
        await self._push_stream_system(
            "session start",
            text=self.goal or FALLBACK_GOAL,
        )
        # After the stream reset, so the run the session joined is the first
        # thing in the log a watcher scrolls to.
        await self._begin_run()
        self._task = asyncio.create_task(self._run_loop(resume=False, force_single_turn=False))
        return self.state_snapshot()

    async def continue_once(self) -> JsonDict:
        if self.is_running:
            raise ValueError("Pi supervisor is already running.")
        if not self.session_id:
            raise ValueError("Pi supervisor has no previous session to continue.")
        self.current_prompt = CONTINUE_MESSAGE
        self.last_prompt = CONTINUE_MESSAGE
        self.status = "starting"
        self.status_reason = "Continuing the existing Pi session."
        self.last_error = None
        self.last_event_at = utc_now()
        self.next_auto_continue_at = None
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.current_tool_calls.clear()
        self._stop_requested = False
        self._budget_stop_requested = False
        self._critic_cancelled = False
        self._stage_workspace_helpers()
        await self._refresh_model_limits()
        await self._begin_run()
        await self._emit_major(
            "pi_supervisor_status",
            {
                "status": self.status,
                "summary": self.status_reason,
                "config": self._config_snapshot(),
            },
        )
        self._task = asyncio.create_task(self._run_loop(resume=True, force_single_turn=True))
        return self.state_snapshot()

    async def stop(self) -> JsonDict:
        self._stop_requested = True
        # An operator stop must return promptly, so it also cancels the critic:
        # a retrospective is worth minutes only when the watchdog is restarting.
        self._critic_cancelled = True
        await self._abort_critic()
        if self.is_running:
            self.status = "stopping"
            self.status_reason = "Stop requested by operator."
            await self._emit_major(
                "pi_supervisor_status",
                {
                    "status": self.status,
                    "summary": self.status_reason,
                },
            )
        await self._shutdown_process()
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=6)
        if self.status not in {"stopped", "completed", "error", "stuck"}:
            self.status = "stopped"
            self.status_reason = "Pi supervisor stopped."
        self.current_pid = None
        self.next_auto_continue_at = None
        await self._push_stream_system("session stop", text=self.status_reason)
        return self.state_snapshot()

    # ------------------------------------------------------------------
    # Run lifecycle
    #
    # A session is not a run. The token budget kills a session every ~30 minutes
    # and the watchdog POSTs /supervisor/start for the next one; the playthrough
    # those sessions add up to is the thing worth scoring, so every start adopts
    # the open run rather than beginning a new one. Only the objective being
    # reached closes it.
    # ------------------------------------------------------------------

    async def _begin_run(self) -> None:
        """Attach this session to the open run, or open one. Never raises."""

        recorder = self.run_recorder
        if recorder is None:
            return
        try:
            handle = await recorder.begin_session(
                goal=self.goal,
                model=self.model or "",
                config={
                    "provider": self.provider,
                    "model": self.model,
                    "thinking": self.thinking,
                    "token_budget": self.token_budget,
                    "max_idle_turns": self.max_idle_turns,
                    "skill_path": str(self.skill_path),
                },
            )
        except Exception as exc:  # noqa: BLE001 — a run that cannot be scored still plays
            self.run_error = _truncate(str(exc), ERROR_TEXT_LIMIT)
            self._push_recent_event("pi_run_record_failed", self.run_error)
            return
        self.run_id = handle.run_id
        self.run_adopted = handle.adopted
        self.run_error = None
        verb = "resumed" if handle.adopted else "started"
        await self._push_stream_system(
            f"run {verb}",
            text=(
                f"{handle.run_id} — session {handle.sessions}, "
                f"{handle.total_presses:,} presses so far."
            ),
        )

    async def _finish_run(self, reason: str) -> None:
        """Close the run for good. Only the objective being reached gets here."""

        recorder = self.run_recorder
        if recorder is None or self.run_id is None:
            return
        try:
            await recorder.finish_run(reason)
        except Exception as exc:  # noqa: BLE001
            self.run_error = _truncate(str(exc), ERROR_TEXT_LIMIT)
            return
        await self._push_stream_system("run finished", text=f"{self.run_id} — {reason}")
        self.run_id = None
        self.run_adopted = False

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    async def record_intervention(self, payload: JsonDict) -> None:
        """Mirror one intervention into the session stream and the state.

        A slot that could not be restored is the loudest thing this harness can
        report: the player's whole context is a file on the model box and the
        run is playing without it. It goes in as an error and stays in the state
        snapshot, which is what ``/health`` reads.
        """

        raw_status = payload.get("status")
        status: JsonDict = raw_status if isinstance(raw_status, dict) else {}
        self.intervention_state = {
            "enabled": bool(status.get("enabled")),
            "fired": int(status.get("fired") or 0),
            "delivered": int(status.get("delivered") or 0),
            "last": {
                "at": payload.get("at"),
                "trigger": payload.get("trigger"),
                "reason": payload.get("reason"),
                "answer": payload.get("answer"),
                "delivered": bool(payload.get("delivered")),
                "error": payload.get("error"),
            },
            "slot_lost": status.get("slot_lost"),
            "disabled_reason": status.get("disabled_reason"),
        }
        slot_lost = status.get("slot_lost")
        if slot_lost:
            self.last_error = _truncate(str(slot_lost.get("message") or ""), ERROR_TEXT_LIMIT)
            await self._push_stream_system(
                "intervention slot lost",
                text=str(slot_lost.get("message") or ""),
                level="error",
            )
            await self._emit_major(
                "pi_intervention_slot_lost",
                {"summary": self.last_error, "filename": slot_lost.get("filename")},
            )
            return
        headline = f"intervention: {payload.get('trigger') or 'unknown'}"
        if payload.get("error"):
            await self._push_stream_system(
                f"{headline} failed", text=str(payload["error"]), level="warn"
            )
            return
        await self._push_stream_system(headline, text=str(payload.get("reason") or ""))

    async def deliver_intervention(self, message: str) -> JsonDict:
        """Hand a thinking session's answer to the live player.

        The same RPC path ``POST /supervisor/steer`` uses, marked so the
        transcript can tell a harness instruction from a human one.
        """

        return await self.send_operator_message(
            message,
            source=INTERVENTION_STREAM_SOURCE,
            limit=INTERVENTION_MESSAGE_LIMIT,
        )

    def live_session_refusal(self) -> Optional[str]:
        """Why an operator message cannot land right now, or ``None`` when it can."""

        if self.status == "critiquing":
            return (
                "Pi is writing its between-session critique, not playing. "
                "Start or continue a session before sending a message."
            )
        if self.status not in STEERABLE_STATUSES:
            return (
                f"Pi supervisor is {self.status}, so there is no live session to steer. "
                "Start or continue a session first."
            )
        process = self._process
        if process is None or process.returncode is not None:
            return "The Pi RPC process is not running, so there is no live session to steer."
        return None

    async def _steering_behavior(self) -> str:
        """``steer`` mid-turn, ``followUp`` when Pi is between turns.

        A bare prompt is rejected by pi while the agent streams, so the choice is
        never "no behavior": only whether the message cuts in at the next tool-call
        boundary or waits for the turn to end. When ``get_state`` does not answer,
        assume streaming, since ``steer`` is accepted in both states.
        """

        response = await self._send_command({"type": "get_state"}, timeout=10.0)
        data = response.get("data") if isinstance(response, dict) else None
        if isinstance(data, dict) and data.get("isStreaming") is False:
            return "followUp"
        return "steer"

    async def send_operator_message(
        self,
        message: str,
        *,
        source: str = OPERATOR_STREAM_SOURCE,
        limit: int = OPERATOR_MESSAGE_LIMIT,
    ) -> JsonDict:
        """Inject a message into the live session.

        ``source`` and ``limit`` exist so the intervention loop can travel this
        same path — one RPC prompt, one stream entry, one transcript line —
        without pretending a harness instruction was typed by a person.

        Raises ``ValueError`` for input the game loop should not carry and
        ``NoLiveSessionError`` when nothing is live to receive it.
        """

        text = (message or "").strip()
        if not text:
            raise ValueError("Operator message is empty.")
        if len(text) > limit:
            raise ValueError(f"Operator message is {len(text)} characters; the limit is {limit}.")
        refusal = self.live_session_refusal()
        if refusal:
            raise NoLiveSessionError(refusal)

        behavior = await self._steering_behavior()
        command: JsonDict = {
            "type": "prompt",
            "message": text,
            "streamingBehavior": behavior,
        }
        response = await self._send_command(command, timeout=30.0)
        if isinstance(response, dict) and response.get("success") is False:
            raise RuntimeError(f"Pi rejected the operator message: {response.get('error')}")

        entry = await self._push_stream_user(text, source=source)
        self._append_transcript(
            direction="outbound",
            role="user",
            channel=source,
            content=text,
            meta={"streaming_behavior": behavior},
        )
        self.user_message_count += 1
        record: JsonDict = {
            "seq": entry["seq"],
            "ts": entry["ts"],
            "text": entry["text"],
            "source": source,
            "streaming_behavior": behavior,
        }
        self.operator_messages.append(record)
        return dict(record)

    async def shutdown(self) -> None:
        await self.stop()

    async def wait_until_idle(self, timeout: float = 30.0) -> None:
        task = self._task
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def _run_loop(self, *, resume: bool, force_single_turn: bool) -> None:
        try:
            if resume:
                self._repair_session_file()
            await self._launch_process()
            await self._send_command({"type": "set_auto_compaction", "enabled": False})
            await self._send_prompt(
                CONTINUE_MESSAGE if resume else self._initial_message(),
                with_images=not resume,
                resume=resume,
            )

            while True:
                if not await self._await_settle():
                    self._apply_exit_status()
                    break

                await self._complete_turn(
                    summary_text=self.current_assistant_text or self.last_assistant_text or None
                )
                await self._refresh_session_stats()

                if self._budget_stop_requested:
                    break
                if self._stop_requested:
                    self.status = "stopped"
                    self.status_reason = "Pi supervisor stopped."
                    break
                if await self._objective_is_complete():
                    self.status = "completed"
                    self.status_reason = "Objective reported complete."
                    await self._finish_run("objective complete")
                    await self._emit_major(
                        "pi_objective_complete",
                        {
                            "summary": self.status_reason,
                            "turns_completed": self.turns_completed,
                        },
                    )
                    break
                if force_single_turn:
                    self.status = "completed"
                    self.status_reason = "Pi completed one manual continue turn."
                    break
                if not self.auto_continue:
                    self.status = "completed"
                    self.status_reason = "Pi completed one turn."
                    break
                if self.max_turns is not None and self.turns_completed >= self.max_turns:
                    self.status = "completed"
                    self.status_reason = f"Reached max turns ({self.max_turns})."
                    break
                if self._token_budget_exhausted():
                    await self._announce_token_budget_stop()
                    break
                if self._current_cycle_has_tool_call:
                    self._consecutive_idle_turns = 0
                else:
                    self._consecutive_idle_turns += 1
                if self.max_idle_turns and self._consecutive_idle_turns >= self.max_idle_turns:
                    self.status = "stuck"
                    self.status_reason = (
                        f"Stopping auto-continue: {self._consecutive_idle_turns} turns in a row "
                        "produced no tool calls."
                    )
                    await self._emit_major(
                        "pi_supervisor_stuck",
                        {
                            "summary": self.status_reason,
                            "idle_turns": self._consecutive_idle_turns,
                            "turns_completed": self.turns_completed,
                            "last_turn_summary": self.latest_turn_summary,
                        },
                    )
                    await self._push_stream_system(
                        "no tool calls",
                        text=self.status_reason,
                        level="warn",
                    )
                    break

                self.continue_count += 1
                self.next_auto_continue_at = _utc_after(self.continue_delay_seconds)
                self.status = "running"
                self.status_reason = (
                    f"Auto-continue scheduled in {self.continue_delay_seconds:.1f}s."
                )
                await self._emit_major(
                    "pi_auto_continue_scheduled",
                    {
                        "summary": self.status_reason,
                        "goal": self.goal,
                        "continue_delay_seconds": self.continue_delay_seconds,
                        "next_auto_continue_at": self.next_auto_continue_at,
                        "next_turn_index": self.turns_completed + 1,
                    },
                )
                if self.continue_delay_seconds:
                    await asyncio.sleep(self.continue_delay_seconds)
                if self._budget_stop_requested:
                    break
                if self._stop_requested:
                    self.status = "stopped"
                    self.status_reason = "Pi supervisor stopped."
                    break
                await self._send_prompt(CONTINUE_MESSAGE, with_images=False, resume=True)
        except asyncio.CancelledError:
            self.status = "stopped"
            self.status_reason = "Pi supervisor task was cancelled."
            raise
        except Exception as exc:  # noqa: BLE001
            self.status = "error"
            self.status_reason = "Pi supervisor encountered an error."
            self.last_error = str(exc)
            await self._emit_major("pi_supervisor_error", {"summary": str(exc)})
            await self._push_stream_system("supervisor error", text=str(exc), level="error")
        finally:
            await self._shutdown_process()
            self.current_pid = None
            self.next_auto_continue_at = None
            # The watchdog starts the next session the moment the status turns
            # terminal, so the critique has to finish first, under `critiquing`.
            terminal_status, terminal_reason = self.status, self.status_reason
            await self._run_critic_pass(terminal_status, terminal_reason)
            self.status, self.status_reason = terminal_status, terminal_reason
            await self._emit_major(
                "pi_supervisor_status",
                {
                    "status": self.status,
                    "summary": self.status_reason,
                    "last_error": self.last_error,
                    "turns_completed": self.turns_completed,
                },
            )
            await self._push_stream_system(
                f"session {self.status}",
                text=self.status_reason,
                level=_STATUS_STREAM_LEVELS.get(self.status, "info"),
            )
            self._task = None

    def _apply_exit_status(self) -> None:
        process = self._process
        returncode = process.returncode if process is not None else None
        if self._budget_stop_requested:
            # The poll already set the terminal status and asked the process to exit.
            return
        if self._stop_requested:
            self.status = "stopped"
            self.status_reason = "Pi supervisor stopped."
            return
        if returncode in (0, None):
            self.status = "completed"
            self.status_reason = "Pi exited after finishing the session."
            return
        stderr_preview = "\n".join(self.stderr_tail).strip()
        raise RuntimeError(
            f"Pi exited with status {returncode}."
            + (f" stderr: {stderr_preview}" if stderr_preview else "")
        )

    async def _objective_is_complete(self) -> bool:
        callback = self.objective_complete
        if callback is None:
            return False
        try:
            result = callback()
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                result = await result
        except Exception as exc:  # noqa: BLE001
            self._push_recent_event(
                "pi_objective_check_failed", _truncate(str(exc), ERROR_TEXT_LIMIT)
            )
            return False
        return bool(result)

    def _token_budget_exhausted(self) -> bool:
        if not self.token_budget:
            return False
        used = (self.context_usage or {}).get("tokens")
        if not isinstance(used, (int, float)):
            return False
        return int(used) >= self.token_budget

    # ------------------------------------------------------------------
    # Process and RPC plumbing
    # ------------------------------------------------------------------

    def _resolve_goal(self, requested: str) -> tuple[str, str]:
        """The goal for the session about to start, and where it came from.

        Precedence, most specific first:

        1. an explicit operator goal handed to :meth:`start`;
        2. the ``NEXT GOAL`` line the last critique wrote;
        3. the objective engine's current objective;
        4. :data:`FALLBACK_GOAL`.

        An operator goal is deliberately not sticky. The watchdog starts a fresh
        session every time the budget trips, and a goal that survived those restarts
        on its own outlived its own completion: the run kept being told to win a
        badge it had already won. A restart that supplies no goal drops through to
        the critic's, which is rewritten after every session and cannot go stale.
        """

        operator = (requested or "").strip()
        if operator:
            return operator, GOAL_SOURCE_OPERATOR
        from_critic = (self.critic_next_goal or "").strip()
        if from_critic:
            return from_critic, GOAL_SOURCE_CRITIC
        objective = str((self._session_start_context or {}).get("objective") or "").strip()
        if objective:
            return objective, GOAL_SOURCE_OBJECTIVE
        return FALLBACK_GOAL, GOAL_SOURCE_FALLBACK

    def _initial_message(self) -> str:
        """Goal first, then last session's retrospective when the critic left one.

        This is the user turn, never the system prompt: ``skill/SKILL.md`` is the
        cached prefix and has to stay byte-identical across sessions.
        """

        critic = _critic_module()
        goal = self.goal.strip() or FALLBACK_GOAL
        handoff = critic.read_handoff(self.workspace_dir)
        if not handoff:
            return goal
        return f"{goal}\n\n## {critic.HANDOFF_HEADING}\n\n{handoff}"

    # ------------------------------------------------------------------
    # Between-session critique
    # ------------------------------------------------------------------

    async def _collect_critic_context(self) -> JsonDict:
        """Ask the server for the objective, game state and explored-map summary."""

        provider = self.critic_context
        if provider is None:
            return {}
        try:
            result = provider()
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                result = await result
        except Exception as exc:  # noqa: BLE001 — context is a nicety, never a blocker
            self._push_recent_event(
                "pi_critic_context_failed", _truncate(str(exc), ERROR_TEXT_LIMIT)
            )
            return {}
        return result if isinstance(result, dict) else {}

    def _build_critic_digest(self, end_context: JsonDict, status: str, reason: str) -> str:
        critic = _critic_module()
        start_context = self._session_start_context or {}
        return critic.build_digest(
            critic.DigestInput(
                goal=self.goal,
                objective=str(end_context.get("objective") or ""),
                turns_completed=self.turns_completed,
                status=status,
                status_reason=reason,
                session_tokens=(self.context_usage or {}).get("tokens"),
                start_state=start_context.get("game_state"),
                final_state=end_context.get("game_state"),
                map_summary=end_context.get("map_summary"),
                notes=critic.read_notes(self.workspace_dir),
                calls=critic.tool_calls_from_stream(self.stream_entries),
            )
        )

    async def _abort_critic(self) -> None:
        process = self._critic_process
        self._critic_process = None
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    # -- the critic's own live narration --------------------------------

    @staticmethod
    def _critic_seq_attribute(kind: str) -> str:
        return "_critic_thinking_seq" if kind == "thinking" else "_critic_text_seq"

    async def _on_critic_event(self, event: JsonDict) -> None:
        """Turn one critic event into a stream entry, as the critic emits it."""

        kind = event.get("type")
        if kind == "attempt_start":
            # A retry starts its own entries, and never leaves the last ones running.
            await self._close_critic_stream_entries()
            if int(event.get("attempt") or 1) > 1:
                await self._push_stream_system(
                    f"critique retry · thinking {event.get('thinking')}",
                    text="The first critic pass reached no answer. Retrying, briefly.",
                    level="warn",
                )
            return
        if kind in ("thinking_delta", "text_delta"):
            await self._critic_stream_delta(kind.split("_")[0], str(event.get("delta") or ""))
            return
        if kind in ("thinking_end", "text_end"):
            await self._critic_finalize_stream(kind.split("_")[0], str(event.get("text") or ""))

    async def _critic_stream_delta(self, kind: str, delta: str) -> None:
        if not delta:
            return
        attribute = self._critic_seq_attribute(kind)
        seq = getattr(self, attribute)
        if seq is None or seq not in self._stream_by_seq:
            entry = self._new_stream_entry(
                kind,
                state="running",
                text=CRITIC_STREAM_PREFIX,
                source="critic",
            )
            seq = entry["seq"]
            setattr(self, attribute, seq)
            self._stream_text_buffers[seq] = CRITIC_STREAM_PREFIX
        await self._emit_stream_entry(self._append_stream_text(seq, delta))

    async def _critic_finalize_stream(self, kind: str, final_text: str) -> None:
        attribute = self._critic_seq_attribute(kind)
        seq = getattr(self, attribute)
        setattr(self, attribute, None)
        entry = self._stream_by_seq.get(seq) if seq is not None else None
        if entry is not None:
            self._stream_text_buffers.pop(seq, None)
            if final_text:
                entry["text"] = _clip_text(CRITIC_STREAM_PREFIX + final_text, STREAM_TEXT_LIMIT)
            entry["state"] = "ok"
            await self._emit_stream_entry(entry)
            return
        if not final_text:
            return
        entry = self._new_stream_entry(
            kind,
            text=_clip_text(CRITIC_STREAM_PREFIX + final_text, STREAM_TEXT_LIMIT),
            source="critic",
        )
        await self._emit_stream_entry(entry)

    async def _close_critic_stream_entries(self) -> None:
        """Nothing the critic left half-written stays ``running`` after it exits."""

        for kind in ("thinking", "text"):
            attribute = self._critic_seq_attribute(kind)
            seq = getattr(self, attribute)
            setattr(self, attribute, None)
            if seq is None:
                continue
            self._stream_text_buffers.pop(seq, None)
            entry = self._stream_by_seq.get(seq)
            if entry is not None and entry.get("state") == "running":
                entry["state"] = "ok"
                await self._emit_stream_entry(entry)

    async def _critic_heartbeat_loop(self, started: float) -> None:
        """One system entry, mutated in place, so a long critique never looks hung."""

        interval = self.critic_heartbeat_seconds
        if interval <= 0:
            return
        while True:
            await asyncio.sleep(interval)
            await self._tick_critic_heartbeat(started)

    async def _tick_critic_heartbeat(self, started: float) -> None:
        elapsed = int(max(0.0, time.monotonic() - started))
        label = f"critique running · {elapsed}s"
        text = f"The critic has been working for {elapsed}s."
        seq = self._critic_heartbeat_seq
        entry = self._stream_by_seq.get(seq) if seq is not None else None
        if entry is None:
            entry = self._new_stream_entry(
                "system",
                text=text,
                system={"label": label, "level": "info"},
                source="critic",
            )
            self._critic_heartbeat_seq = entry["seq"]
        else:
            entry["ts"] = utc_now()
            entry["text"] = text
            entry["system"] = {"label": label, "level": "info"}
        await self._emit_stream_entry(entry)

    async def _stop_critic_heartbeat(self, started: float) -> None:
        task = self._critic_heartbeat_task
        self._critic_heartbeat_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        seq = self._critic_heartbeat_seq
        self._critic_heartbeat_seq = None
        entry = self._stream_by_seq.get(seq) if seq is not None else None
        if entry is None:
            return
        elapsed = int(max(0.0, time.monotonic() - started))
        entry["text"] = f"The critic ran for {elapsed}s."
        entry["system"] = {"label": f"critique ran {elapsed}s", "level": "info"}
        await self._emit_stream_entry(entry)

    async def _run_critic_pass(self, terminal_status: str, terminal_reason: str) -> None:
        """Review the finished session and leave a handoff. Never raises, never blocks."""

        if not self.critic_enabled or self._critic_cancelled or not self.pi_binary:
            return
        critic = _critic_module()
        self.status = "critiquing"
        self.status_reason = "Reviewing the finished session for the next one."
        await self._emit_major(
            "pi_critique_start",
            {"summary": self.status_reason, "status": self.status},
        )
        await self._push_stream_system("critique start", text=self.status_reason)

        self.last_critique_error = None
        self.last_critique_stop_reason = None
        self.last_critique_usage = None
        self.last_critique_salvaged = False
        self.last_critique_raw_path = None
        self.last_critique_attempts = []
        self._critic_thinking_seq = None
        self._critic_text_seq = None
        self._critic_heartbeat_seq = None
        started = time.monotonic()
        self._critic_heartbeat_task = asyncio.create_task(self._critic_heartbeat_loop(started))
        try:
            end_context = await self._collect_critic_context()
            result = await critic.run_critic(
                pi_binary=self.pi_binary,
                workspace_dir=self.workspace_dir,
                digest=self._build_critic_digest(end_context, terminal_status, terminal_reason),
                provider=self.provider,
                model=self.model,
                thinking=self.critic_thinking,
                timeout_seconds=self.critic_timeout_seconds,
                retry_enabled=self.critic_retry_enabled,
                retry_thinking=self.critic_retry_thinking,
                event_sink=self._on_critic_event,
                process_sink=lambda process: setattr(self, "_critic_process", process),
            )
        except asyncio.CancelledError:
            await self._abort_critic()
            raise
        except Exception as exc:  # noqa: BLE001 — a bad critique must not wedge the loop
            self.last_critique_error = _truncate(str(exc), ERROR_TEXT_LIMIT)
            await self._push_stream_system(
                "critique failed",
                text=self.last_critique_error,
                level="warn",
            )
            return
        finally:
            self._critic_process = None
            await self._stop_critic_heartbeat(started)
            await self._close_critic_stream_entries()

        self.last_critique_at = utc_now()
        self.last_critique_seconds = result.duration_seconds
        self.last_critique_tokens = result.digest_tokens
        self.last_critique_stop_reason = result.stop_reason
        self.last_critique_usage = result.usage
        self.last_critique_raw_path = result.raw_path
        self.last_critique_attempts = list(result.attempts)
        self.last_critique_salvaged = bool(result.salvaged)
        if not result.ok:
            self.last_critique_error = _truncate(
                result.error or "Critic produced nothing.", ERROR_TEXT_LIMIT
            )
            await self._emit_major(
                "pi_critique_failed",
                {
                    "summary": self.last_critique_error,
                    "duration_seconds": result.duration_seconds,
                    "stop_reason": result.stop_reason,
                    "usage": result.usage,
                    "raw_path": result.raw_path,
                },
            )
            await self._push_stream_system(
                "critique failed",
                text=f"{self.last_critique_error} Keeping the previous handoff.",
                level="warn",
            )
            return

        self.last_critique = result.text
        # Replaced outright, never merged: a next goal from two sessions ago is as
        # stale as the operator goal this precedence chain exists to retire.
        self.critic_next_goal = (result.next_goal or "").strip()
        self.last_critique_error = (
            _truncate(result.error, ERROR_TEXT_LIMIT) if result.error else None
        )
        await self._emit_major(
            "pi_critique_ready",
            {
                "summary": _truncate(result.text, 220),
                "duration_seconds": result.duration_seconds,
                "digest_tokens": result.digest_tokens,
                "handoff_path": result.handoff_path,
                "salvaged": result.salvaged,
                "next_goal": self.critic_next_goal,
                "stop_reason": result.stop_reason,
                "usage": result.usage,
                "raw_path": result.raw_path,
            },
        )
        await self._push_stream_system(
            "critique salvaged" if result.salvaged else "critique ready",
            text=result.text,
            level="warn" if result.salvaged else "info",
        )
        if self.critic_next_goal:
            await self._push_stream_system(
                "next goal",
                text=self.critic_next_goal,
            )

    # `poke` exists because hand-built curl JSON was losing roughly 40% of the
    # agent's actions to a single dropped closing quote. Bare arguments cannot be
    # misquoted: the CLI has no JSON and no string literal to truncate.
    WORKSPACE_HELPERS = {"poke": Path("pokemon_agent") / "agent_cli.py"}

    #: Earlier helpers, removed. A workspace outlives a session, so a stale copy
    #: would keep working and the quoting failure would come back with it.
    STALE_WORKSPACE_HELPERS = ("agent_curl.sh", "act")

    def _stage_workspace_helpers(self) -> None:
        for name, relative_source in self.WORKSPACE_HELPERS.items():
            source = self.repo_root / relative_source
            if not source.is_file():
                raise FileNotFoundError(f"Missing helper script: {source}")
            destination = self.workspace_dir / name
            shutil.copy2(source, destination)
            destination.chmod(0o755)
        for stale in self.STALE_WORKSPACE_HELPERS:
            (self.workspace_dir / stale).unlink(missing_ok=True)

    def _build_command(self) -> list[str]:
        assert self.pi_binary is not None
        assert self.session_id is not None
        command = [
            self.pi_binary,
            "--mode",
            "rpc",
            "--system-prompt",
            str(self.skill_path),
            "--tools",
            ",".join(DEFAULT_TOOLS),
            "--session-id",
            self.session_id,
            "--session-dir",
            str(self.session_dir),
            "-ne",
            "-ns",
            "-nc",
            "-np",
            "--no-themes",
            "--offline",
        ]
        if self.provider:
            command.extend(["--provider", self.provider])
        if self.model:
            command.extend(["--model", self.model])
        if self.thinking:
            command.extend(["--thinking", self.thinking])
        return command

    async def _launch_process(self) -> None:
        command = self._build_command()
        self._settled_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._pending_responses.clear()
        self.status = "running"
        self.status_reason = "Pi RPC session is live."
        self.next_auto_continue_at = None
        await self._emit_major(
            "pi_session_launch",
            {
                "summary": f"Launching Pi RPC session {self.session_id}.",
                "command_preview": command,
                "session_id": self.session_id,
            },
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.workspace_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                **(
                    {"PORT": str(port)}
                    if (port := _server_port(self.server_url)) is not None
                    else {}
                ),
            },
        )
        self._process = process
        self.current_pid = process.pid
        self.session_file = self._resolve_session_file()
        self._reader_tasks = [
            asyncio.create_task(self._read_stdout(process)),
            asyncio.create_task(self._read_stderr(process)),
            asyncio.create_task(self._watch_exit(process)),
        ]
        self._stats_task = self._start_stats_poll(process)
        self._push_recent_event(
            "pi_session",
            f"Session {self.session_id or 'unknown'} started (pid {process.pid}).",
        )

    async def _watch_exit(self, process: asyncio.subprocess.Process) -> None:
        await process.wait()
        self._exit_event.set()
        for future in self._pending_responses.values():
            if not future.done():
                future.set_exception(RuntimeError("Pi exited before responding."))
        self._pending_responses.clear()

    async def _terminate_process(self) -> None:
        """Close stdin and, if needed, signal the RPC process until it exits."""

        process = self._process
        if process is None:
            return
        if process.returncode is None and process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        if process.returncode is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=4)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=3)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)

    async def _shutdown_process(self) -> None:
        await self._close_open_stream_entries()
        await self._cancel_stats_poll()
        await self._terminate_process()
        for task in self._reader_tasks:
            task.cancel()
        for task in self._reader_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._reader_tasks = []
        self._process = None
        self.current_pid = None
        self._repair_session_file()

    def _repair_session_file(self) -> None:
        session_file = self._resolve_session_file()
        if session_file is None:
            return
        repaired = repair_orphaned_tool_calls(session_file)
        if not repaired:
            return
        self.repaired_tool_calls += len(repaired)
        self._push_recent_event(
            "pi_tool_calls_repaired",
            f"Wrote {len(repaired)} synthetic tool results for interrupted tool calls.",
            {"tool_call_ids": repaired},
        )

    def _resolve_session_file(self) -> Optional[Path]:
        if self.session_file is not None and self.session_file.is_file():
            return self.session_file
        if not self.session_id:
            return None
        try:
            candidates = sorted(
                self.session_dir.glob("*.jsonl"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return None
        for candidate in candidates:
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    header = json.loads(handle.readline())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if header.get("type") == "session" and header.get("id") == self.session_id:
                self.session_file = candidate.resolve()
                return self.session_file
        return None

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def _send_command(
        self,
        command: JsonDict,
        *,
        timeout: float = 30.0,
    ) -> Optional[JsonDict]:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise RuntimeError("Pi RPC process is not running.")
        request_id = self._next_request_id()
        payload = {"id": request_id, **command}
        future: asyncio.Future[JsonDict] = asyncio.get_running_loop().create_future()
        self._pending_responses[request_id] = future
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            process.stdin.write(line.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            self._pending_responses.pop(request_id, None)
            raise RuntimeError(f"Failed to write RPC command: {exc}") from exc
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, RuntimeError):
            return None
        finally:
            self._pending_responses.pop(request_id, None)
            if future.done() and not future.cancelled():
                # Consume any exception so a cancelled caller never leaves it
                # to surface as an unretrieved-future error on the event loop.
                future.exception()

    async def _send_prompt(self, message: str, *, with_images: bool, resume: bool) -> None:
        attachment_paths = self._frame_paths() if with_images else []
        images = [self._encode_image(path) for path in attachment_paths]

        self.current_prompt = message
        self.last_prompt = message
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.current_tool_calls.clear()
        self._current_turn_completed = False
        self._current_cycle_has_tool_call = False
        self._active_thinking_seq = None
        self._active_text_seq = None
        self._message_saw_thinking_delta = False
        self._message_saw_text_delta = False
        self._settled_event.clear()
        self.last_turn_started_at = utc_now()
        self.last_event_at = self.last_turn_started_at

        prompt_entry = self._append_transcript(
            direction="outbound",
            role="user",
            channel="prompt",
            content=message,
            meta={
                "resume": resume,
                "attachments": [str(path) for path in attachment_paths],
            },
        )
        await self._emit_stream(
            "pi_prompt_sent",
            {
                "prompt": prompt_entry["content"],
                "attachments": prompt_entry["meta"]["attachments"],
                "images": len(images),
                "resume": resume,
                "session_id": self.session_id,
            },
        )
        await self._emit_stream("pi_transcript", {"entry": prompt_entry})
        await self._push_stream_system(
            f"{'continue' if resume else 'goal'} \u00b7 turn {self.turns_completed + 1}",
            text=message,
        )
        await self._push_stream_user(message)

        command: JsonDict = {"type": "prompt", "message": message}
        if images:
            command["images"] = images
        response = await self._send_command(command, timeout=60.0)
        if response is not None and response.get("success") is False:
            raise RuntimeError(f"Pi rejected the prompt: {response.get('error')}")

    def _frame_paths(self) -> list[Path]:
        paths: list[Path] = []
        for filename in FRAME_IMAGE_FILES:
            candidate = self.workspace_dir / filename
            if candidate.is_file():
                paths.append(candidate)
        return paths

    @staticmethod
    def _encode_image(path: Path) -> JsonDict:
        return {
            "type": "image",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "mimeType": "image/png",
        }

    async def _await_settle(self) -> bool:
        settle_task = asyncio.create_task(self._settled_event.wait())
        exit_task = asyncio.create_task(self._exit_event.wait())
        try:
            await asyncio.wait({settle_task, exit_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (settle_task, exit_task):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._settled_event.is_set():
            self._settled_event.clear()
            return True
        return False

    async def _refresh_session_stats(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            response = await self._send_command({"type": "get_session_stats"}, timeout=15.0)
        except RuntimeError:
            return
        if not isinstance(response, dict) or response.get("success") is not True:
            return
        data = response.get("data")
        if not isinstance(data, dict):
            return
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            total = tokens.get("total", tokens.get("totalTokens"))
            self.session_usage = {
                "input": tokens.get("input"),
                "output": tokens.get("output"),
                "cacheRead": tokens.get("cacheRead"),
                "cacheWrite": tokens.get("cacheWrite"),
                "totalTokens": int(total) if isinstance(total, (int, float)) else None,
                "updated_at": utc_now(),
            }
        context_usage = data.get("contextUsage")
        if isinstance(context_usage, dict):
            self.context_usage = {**context_usage, "updated_at": utc_now()}
            used = context_usage.get("tokens")
            if isinstance(used, (int, float)):
                self.session_usage = {
                    **(self.session_usage or {}),
                    "totalTokens": int(used),
                    "updated_at": utc_now(),
                }

    def _start_stats_poll(
        self, process: asyncio.subprocess.Process
    ) -> Optional[asyncio.Task[None]]:
        interval = self.stats_poll_seconds
        if not interval:
            return None
        return asyncio.create_task(self._stats_poll_loop(process, interval))

    async def _cancel_stats_poll(self) -> None:
        task = self._stats_task
        self._stats_task = None
        if task is None or task.done():
            return
        if task is asyncio.current_task():
            # The poll never tears itself down; leave it to unwind on its own.
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _stats_poll_loop(
        self,
        process: asyncio.subprocess.Process,
        interval: float,
    ) -> None:
        """Sample session stats on a timer so telemetry survives an hour-long turn."""

        try:
            while True:
                await asyncio.sleep(interval)
                if self._stop_requested or self._budget_stop_requested:
                    return
                if self._process is not process or process.returncode is not None:
                    return
                await self._refresh_session_stats()
                await self._emit_stream(
                    "pi_session_stats",
                    {
                        "source": "poll",
                        "session_usage": self.session_usage,
                        "context_usage": self.context_usage,
                        "token_budget": self.token_budget,
                    },
                )
                if self._token_budget_exhausted():
                    await self._stop_for_token_budget()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._push_recent_event("pi_stats_poll_failed", _truncate(str(exc), ERROR_TEXT_LIMIT))

    async def _announce_token_budget_stop(self) -> None:
        used = (self.context_usage or {}).get("tokens")
        self.status = "completed"
        self.status_reason = f"Token budget reached ({used}/{self.token_budget} context tokens)."
        await self._emit_major(
            "pi_token_budget_reached",
            {
                "summary": self.status_reason,
                "token_budget": self.token_budget,
                "context_usage": self.context_usage,
            },
        )
        await self._push_stream_system(
            "token budget reached",
            text=self.status_reason,
            level="warn",
        )

    async def _stop_for_token_budget(self) -> None:
        """End the run from the poll, mid-turn, without waiting for a settle."""

        self._budget_stop_requested = True
        await self._announce_token_budget_stop()
        # Only the process is torn down here: the run loop wakes on the exit event
        # and its own teardown cancels this task and repairs orphaned tool calls.
        await self._terminate_process()

    # ------------------------------------------------------------------
    # Stream readers
    # ------------------------------------------------------------------

    async def _read_stdout(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        async for line in self._iter_lines(process.stdout):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                clipped = _clip_text(raw_line, STDERR_LINE_LIMIT)
                self.stderr_tail.append(_truncate(clipped, STDERR_TAIL_PREVIEW))
                entry = self._append_transcript(
                    direction="system",
                    role="system",
                    channel="stdout_parse_error",
                    content=clipped,
                )
                await self._emit_stream("pi_stdout_parse_error", {"line": entry["content"]})
                await self._emit_stream("pi_transcript", {"entry": entry})
                await self._push_stream_system(
                    "stdout parse error",
                    text=entry["content"],
                    level="error",
                )
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "response":
                self._resolve_response(payload)
                continue
            await self._handle_event(payload)

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        async for line in self._iter_lines(process.stderr):
            text = line.strip()
            if not text:
                continue
            clipped = _clip_text(text, STDERR_LINE_LIMIT)
            self.stderr_tail.append(_truncate(clipped, STDERR_TAIL_PREVIEW))
            entry = self._append_transcript(
                direction="system",
                role="system",
                channel="stderr",
                content=clipped,
            )
            await self._emit_stream("pi_stderr", {"text": entry["content"]})
            await self._emit_stream("pi_transcript", {"entry": entry})
            await self._push_stream_system("stderr", text=entry["content"], level="warn")

    @staticmethod
    def _iter_lines(stream: asyncio.StreamReader):
        return iter_stream_lines(stream)

    def _resolve_response(self, payload: JsonDict) -> None:
        request_id = payload.get("id")
        future = self._pending_responses.pop(request_id, None) if request_id else None
        if future is not None and not future.done():
            future.set_result(payload)
        if payload.get("success") is False:
            self._push_recent_event(
                "pi_rpc_error",
                _truncate(f"{payload.get('command')}: {payload.get('error')}", 200),
                payload,
            )

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _refresh_turn_plan_preview_from_workspace(self) -> None:
        path = self.workspace_dir / "turn_plan.json"
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self.turn_plan_preview = {
            "source": "workspace_turn_plan",
            "updated_at": utc_now(),
            "payload": payload,
        }

    async def _refresh_model_limits(self) -> None:
        chosen_provider, chosen_model = normalize_model_lookup(self.provider, self.model)
        if not self.pi_binary or not chosen_model:
            self.model_limits = None
            return

        cache_key = f"{chosen_provider or ''}::{chosen_model}"
        if cache_key in self._model_limits_cache:
            cached = self._model_limits_cache[cache_key]
            self.model_limits = dict(cached) if isinstance(cached, dict) else None
            return

        command = [self.pi_binary]
        if chosen_provider:
            command.extend(["--provider", chosen_provider])
        command.extend(["--list-models", chosen_model])

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=8)
        except (OSError, asyncio.TimeoutError):
            self._model_limits_cache[cache_key] = None
            self.model_limits = None
            return

        if process.returncode != 0:
            self._model_limits_cache[cache_key] = None
            self.model_limits = None
            return

        parsed = parse_model_limits_output(
            stdout.decode("utf-8", errors="replace"),
            provider=chosen_provider,
            model=chosen_model,
        )
        self._model_limits_cache[cache_key] = parsed
        self.model_limits = dict(parsed) if isinstance(parsed, dict) else None

    def _push_recent_event(
        self,
        event_type: str,
        summary: str,
        payload: Optional[JsonDict] = None,
    ) -> None:
        self.recent_events.append(
            {
                "type": event_type,
                "timestamp": utc_now(),
                "summary": summary,
                "payload": payload or {},
            }
        )
        self.last_event_at = utc_now()

    async def _emit_major(self, event_type: str, payload: JsonDict) -> None:
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            **payload,
        }
        self._push_recent_event(event_type, payload.get("summary") or event_type, payload)
        if self.event_sink is not None:
            await self.event_sink(event)

    async def _emit_stream(self, event_type: str, payload: JsonDict) -> None:
        if self.stream_sink is None:
            return
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            **payload,
        }
        await self.stream_sink(event)

    def _append_transcript(
        self,
        *,
        direction: str,
        role: str,
        channel: str,
        content: str,
        meta: Optional[JsonDict] = None,
        status: str = "info",
    ) -> JsonDict:
        entry = {
            "timestamp": utc_now(),
            "direction": direction,
            "role": role,
            "channel": channel,
            # `preview` is for a collapsed row; `content` is what the operator opens.
            "content": _clip_text(content, TRANSCRIPT_TEXT_LIMIT),
            "preview": _truncate(content, 220),
            "status": status,
            "meta": meta or {},
        }
        self.transcript.append(entry)
        self.last_event_at = entry["timestamp"]
        return entry

    async def _complete_turn(
        self,
        *,
        summary_text: Optional[str],
        tool_result_count: int = 0,
    ) -> None:
        if self._current_turn_completed:
            return
        self._current_turn_completed = True
        self.turns_completed += 1
        self.last_turn_completed_at = utc_now()
        self.latest_turn_summary = _truncate(summary_text or "Pi completed a turn.", 220)
        await self._emit_major(
            "pi_turn_end",
            {
                "summary": self.latest_turn_summary,
                "turns_completed": self.turns_completed,
                "tool_result_count": tool_result_count,
                "had_tool_calls": self._current_cycle_has_tool_call,
            },
        )
        await self._push_stream_system(
            f"turn {self.turns_completed} complete",
            text=self.latest_turn_summary,
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: JsonDict) -> None:
        event_type = event.get("type")
        self.last_event_at = utc_now()

        if event_type == "model_change":
            self.provider = event.get("provider") or self.provider
            self.model = event.get("modelId") or event.get("model") or self.model
            await self._refresh_model_limits()
            self._push_recent_event(
                "pi_model_change",
                f"Model {(self.model or 'unknown')} active.",
                {
                    "provider": self.provider,
                    "model": self.model,
                },
            )
            return

        if event_type == "thinking_level_change":
            self.thinking = event.get("thinkingLevel") or self.thinking
            self._push_recent_event(
                "pi_thinking_level_change",
                f"Thinking {(self.thinking or 'default')}.",
                {"thinking": self.thinking},
            )
            return

        if event_type == "agent_start":
            await self._emit_major("pi_agent_start", {"summary": "Pi agent run started."})
            return

        if event_type == "agent_end":
            summary = (
                self.current_assistant_text
                or self.last_assistant_text
                or self.latest_turn_summary
                or "Pi agent run ended."
            )
            await self._emit_major(
                "pi_agent_end",
                {
                    "summary": summary,
                    "will_retry": bool(event.get("willRetry")),
                },
            )
            return

        if event_type == "agent_settled":
            await self._emit_major("pi_agent_settled", {"summary": "Pi agent run settled."})
            self._settled_event.set()
            return

        if event_type == "turn_start":
            self.last_turn_started_at = utc_now()
            await self._emit_major("pi_turn_start", {"summary": "Pi turn started."})
            return

        if event_type == "turn_end":
            message_text = extract_message_text(event.get("message"))
            thinking_text = extract_message_thinking(event.get("message"))
            if message_text:
                self.last_assistant_text = message_text
            if thinking_text:
                self.last_assistant_thinking = thinking_text
            self._refresh_turn_plan_preview_from_workspace()
            return

        if event_type == "message_start":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                self.current_assistant_text = ""
                self.current_assistant_thinking = ""
                self.assistant_message_count += 1
                self._pending_thinking_in_message = False
                self._active_thinking_seq = None
                self._active_text_seq = None
                self._message_saw_thinking_delta = False
                self._message_saw_text_delta = False
            elif role == "user":
                self.user_message_count += 1
            return

        if event_type == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            assistant_type = assistant_event.get("type")
            delta = assistant_event.get("delta")
            if assistant_type == "text_delta" and isinstance(delta, str):
                self.current_assistant_text += delta
                self._push_recent_event(
                    "pi_text_delta",
                    _truncate(delta, 120),
                    {"delta": _truncate(delta, 240)},
                )
                await self._emit_stream(
                    "pi_text_delta",
                    {
                        "delta": delta,
                        "text": _clip_text(self.current_assistant_text, 4000),
                    },
                )
                await self._stream_delta("text", delta)
            elif assistant_type == "thinking_delta" and isinstance(delta, str):
                if not self._pending_thinking_in_message:
                    self.thinking_block_count += 1
                    self._pending_thinking_in_message = True
                self.current_assistant_thinking += delta
                self._push_recent_event(
                    "pi_thinking_delta",
                    _truncate(delta, 120),
                    {"delta": _truncate(delta, 240)},
                )
                await self._emit_stream(
                    "pi_thinking_delta",
                    {
                        "delta": delta,
                        "thinking": _clip_text(self.current_assistant_thinking, 4000),
                    },
                )
                await self._stream_delta("thinking", delta)
            else:
                summary = assistant_type or "message_update"
                self._push_recent_event("pi_message_update", summary, assistant_event)
            return

        if event_type == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                return
            final_text = extract_message_text(message) or self.current_assistant_text
            final_thinking = extract_message_thinking(message) or self.current_assistant_thinking
            if final_thinking and not self._pending_thinking_in_message:
                self.thinking_block_count += 1
            self._pending_thinking_in_message = False
            usage = message.get("usage")
            if isinstance(usage, dict):
                self.last_message_usage = usage
            if final_text:
                self.last_assistant_text = final_text
                self.current_assistant_text = final_text
                text_entry = self._append_transcript(
                    direction="inbound",
                    role="assistant",
                    channel="assistant",
                    content=final_text,
                )
                await self._emit_stream("pi_transcript", {"entry": text_entry})
            if final_thinking:
                self.last_assistant_thinking = final_thinking
                self.current_assistant_thinking = final_thinking
                thinking_entry = self._append_transcript(
                    direction="inbound",
                    role="assistant_thinking",
                    channel="thinking",
                    content=final_thinking,
                )
                await self._emit_stream("pi_transcript", {"entry": thinking_entry})
            await self._finalize_stream_text("thinking", final_thinking)
            await self._finalize_stream_text("text", final_text)
            self._refresh_turn_plan_preview_from_workspace()
            await self._emit_major(
                "pi_message_end",
                {
                    "summary": _truncate(final_text or "Assistant message completed.", 220),
                    "usage": self.last_message_usage,
                },
            )
            return

        if event_type == "tool_execution_start":
            args = event.get("args") or {}
            file_hint = extract_file_hint(args)
            summary = event.get("toolName", "tool")
            if file_hint:
                summary = f"{summary}: {file_hint}"
            entry = {
                "tool_call_id": event.get("toolCallId"),
                "tool_name": event.get("toolName"),
                "summary": summary,
                "file_hint": file_hint,
                "args_preview": preview_payload(args),
                "args": payload_text(args),
                "started_at": utc_now(),
                "status": "running",
                "result": "",
                "result_preview": "",
            }
            self._current_cycle_has_tool_call = True
            self.current_tool_calls[event.get("toolCallId", summary)] = entry
            self.tool_call_count += 1
            await self._seal_active_narration()
            stream_entry = await self._start_tool_stream_entry(
                event.get("toolCallId"),
                event.get("toolName"),
                args,
            )
            entry["stream_seq"] = stream_entry["seq"]
            turn_plan = extract_turn_plan_candidate(args)
            if turn_plan is not None:
                self.turn_plan_preview = {
                    "source": "pi_tool_write",
                    "updated_at": utc_now(),
                    "payload": turn_plan,
                }
            await self._emit_major(
                "pi_tool_start",
                {
                    "summary": summary,
                    "tool_name": event.get("toolName"),
                    "args_preview": entry["args_preview"],
                },
            )
            return

        if event_type == "tool_execution_update":
            tool_call_id = event.get("toolCallId")
            entry = self.current_tool_calls.get(tool_call_id)
            if entry is not None:
                entry["result"] = payload_text(event.get("partialResult"))
                entry["result_preview"] = preview_payload(event.get("partialResult"))
            self._push_recent_event(
                "pi_tool_update",
                event.get("toolName", "tool_update"),
                {
                    "tool_name": event.get("toolName"),
                    "partial_result_preview": preview_payload(event.get("partialResult")),
                },
            )
            return

        if event_type == "tool_execution_end":
            tool_call_id = event.get("toolCallId")
            entry = self.current_tool_calls.pop(tool_call_id, None) or {
                "tool_call_id": tool_call_id,
                "tool_name": event.get("toolName"),
                "summary": event.get("toolName", "tool"),
                "file_hint": None,
                "args": "",
                "args_preview": "",
                "started_at": utc_now(),
                "result": "",
            }
            entry["status"] = "error" if event.get("isError") else "completed"
            entry["finished_at"] = utc_now()
            entry["result"] = payload_text(event.get("result"))
            entry["result_preview"] = preview_payload(event.get("result"))
            self.recent_tools.append(entry)
            self._refresh_turn_plan_preview_from_workspace()
            summary = entry["summary"]
            if entry["status"] == "error":
                summary = f"{summary} failed"
            stream_entry = await self._finish_tool_stream_entry(
                tool_call_id,
                event.get("toolName") or entry.get("tool_name"),
                event.get("result"),
                is_error=bool(event.get("isError")),
            )
            if stream_entry is not None:
                entry["stream_seq"] = stream_entry["seq"]
            await self._emit_major(
                "pi_tool_end",
                {
                    "summary": summary,
                    "tool_name": entry.get("tool_name"),
                    "is_error": bool(event.get("isError")),
                    "result_preview": entry["result_preview"],
                },
            )
            return

        if event_type == "queue_update":
            await self._emit_major(
                "pi_queue_update",
                {
                    "summary": "Pi queue updated.",
                    "steering_count": len(event.get("steering") or []),
                    "follow_up_count": len(event.get("followUp") or []),
                },
            )
            return

        if event_type == "compaction_start":
            tokens_before = event.get("tokensBefore")
            if isinstance(tokens_before, (int, float)):
                self.last_compaction_tokens_before = int(tokens_before)
            self.last_compaction_at = utc_now()
            await self._emit_major(
                "pi_compaction_start",
                {
                    "summary": f"Compaction started ({event.get('reason', 'unknown')}).",
                    "tokens_before": self.last_compaction_tokens_before,
                },
            )
            await self._push_stream_system(
                "compaction start",
                text=f"Compaction started ({event.get('reason', 'unknown')}).",
            )
            return

        if event_type == "compaction_end":
            summary = f"Compaction finished ({event.get('reason', 'unknown')})."
            if event.get("aborted"):
                summary = f"Compaction aborted ({event.get('reason', 'unknown')})."
            result = event.get("result")
            tokens_after = event.get("tokensAfter")
            if tokens_after is None and isinstance(result, dict):
                tokens_after = result.get("estimatedTokensAfter")
            if isinstance(tokens_after, (int, float)):
                self.last_compaction_tokens_after = int(tokens_after)
                self.session_usage = {
                    **(self.session_usage or {}),
                    "totalTokens": int(tokens_after),
                    "updated_at": utc_now(),
                    "after_compaction": True,
                }
            self.last_compaction_at = utc_now()
            await self._emit_major(
                "pi_compaction_end",
                {
                    "summary": summary,
                    "tokens_after": self.last_compaction_tokens_after,
                },
            )
            await self._push_stream_system("compaction end", text=summary)
            return

        if event_type == "auto_retry_start":
            retry_summary = (
                f"Auto-retry {event.get('attempt')}/{event.get('maxAttempts')} "
                f"after error: {_truncate(str(event.get('errorMessage', 'unknown')), 180)}"
            )
            await self._emit_major("pi_auto_retry_start", {"summary": retry_summary})
            await self._push_stream_system(
                f"auto-retry {event.get('attempt')}/{event.get('maxAttempts')}",
                text=retry_summary,
                level="warn",
            )
            return

        if event_type == "auto_retry_end":
            succeeded = bool(event.get("success"))
            retry_summary = (
                "Auto-retry succeeded."
                if succeeded
                else _truncate(f"Auto-retry failed: {event.get('finalError', 'unknown')}", 220)
            )
            await self._emit_major("pi_auto_retry_end", {"summary": retry_summary})
            await self._push_stream_system(
                "auto-retry done" if succeeded else "auto-retry failed",
                text=retry_summary,
                level="info" if succeeded else "error",
            )
            return

        self._push_recent_event(
            f"pi_{event_type or 'event'}",
            preview_payload(event, 180),
            event,
        )
