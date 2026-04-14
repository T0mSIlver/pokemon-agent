"""Async Pi supervisor for turn-based session control and dashboard telemetry."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

from pokemon_agent.agent_runtime import utc_now
from pokemon_agent.harness.prompting import (
    CONTINUE_PROMPT,
    continue_supervisor_prompt,
    default_supervisor_prompt,
)

JsonDict = dict[str, Any]
EventSink = Callable[[JsonDict], Awaitable[None]]
StreamSink = Callable[[JsonDict], Awaitable[None]]

DEFAULT_TOOLS = ["read", "bash", "edit", "write", "grep", "find", "ls"]
VISION_ATTACHMENT_FILES = ("latest_frame_annotated.png", "latest_frame.png")
ANNOTATED_FRAME_NAME = "latest_frame_annotated.png"
RAW_FRAME_NAME = "latest_frame.png"


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
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _clip_text(value: str, limit: int = 6000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]..."


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
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                thinking_parts.append(text.strip())
    return "\n".join(thinking_parts).strip()


def preview_payload(value: Any, limit: int = 260) -> str:
    text = "\n".join(_collect_text(value)).strip()
    if not text:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            text = repr(value)
    return _truncate(text, limit)


def payload_text(value: Any, limit: int = 20000) -> str:
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
    for raw_line in output.splitlines():
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
                return {"raw": _truncate(raw, 500)}
            if isinstance(parsed, dict):
                return parsed
    return {"path": path}


class PiSupervisor:
    """Launches Pi turn-by-turn and exposes a dashboard-friendly state snapshot."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        server_url: str,
        event_sink: Optional[EventSink] = None,
        stream_sink: Optional[StreamSink] = None,
        repo_root: Optional[Path] = None,
        pi_binary: Optional[str] = None,
    ) -> None:
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.server_url = server_url
        self.repo_root = (repo_root or _repo_root()).expanduser().resolve()
        self.skill_path = default_skill_path().expanduser().resolve()
        self.pi_binary = pi_binary or shutil.which("pi")
        self.event_sink = event_sink
        self.stream_sink = stream_sink
        self.session_dir = self.workspace_dir / "pi-session"
        self.session_dir.mkdir(parents=True, exist_ok=True)

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
        self.continue_delay_seconds = 1.0
        self.max_turns: Optional[int] = None
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.thinking: Optional[str] = None
        self.current_prompt = ""
        self.last_prompt = ""
        self.default_prompt = default_supervisor_prompt(
            server_url=self.server_url,
            workspace_dir=self.workspace_dir,
            goal="",
        )
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
        self.turn_plan_preview: Optional[JsonDict] = None
        self.next_auto_continue_at: Optional[str] = None
        self.session_usage: Optional[JsonDict] = None
        self.last_message_usage: Optional[JsonDict] = None
        self.model_limits: Optional[JsonDict] = None
        self.tool_call_count: int = 0
        self.thinking_block_count: int = 0
        self.assistant_message_count: int = 0
        self.user_message_count: int = 0
        self.last_compaction_tokens_before: Optional[int] = None
        self.last_compaction_tokens_after: Optional[int] = None
        self.last_compaction_at: Optional[str] = None
        self._pending_thinking_in_message: bool = False
        self._model_limits_cache: dict[str, Optional[JsonDict]] = {}
        self.current_turn_vision: JsonDict = self._new_turn_vision([])
        self.last_turn_vision: JsonDict = self._new_turn_vision([])
        self.vision_violation_count: int = 0

        self._task: Optional[asyncio.Task[None]] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stop_requested = False
        self._current_turn_completed = False

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _config_snapshot(self) -> JsonDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking": self.thinking,
            "auto_continue": self.auto_continue,
            "goal": self.goal,
            "continue_delay_seconds": self.continue_delay_seconds,
            "max_turns": self.max_turns,
            "skill_path": str(self.skill_path),
            "session_dir": str(self.session_dir),
            "server_url": self.server_url,
            "tools": DEFAULT_TOOLS,
        }

    def _new_turn_vision(self, attachment_paths: list[Path]) -> JsonDict:
        annotated_path = next(
            (str(path) for path in attachment_paths if path.name == ANNOTATED_FRAME_NAME),
            None,
        )
        raw_path = next((str(path) for path in attachment_paths if path.name == RAW_FRAME_NAME), None)
        return {
            "annotated_path": annotated_path,
            "annotated_available": annotated_path is not None,
            "annotated_read": False,
            "raw_path": raw_path,
            "raw_available": raw_path is not None,
            "raw_read": False,
            "read_sequence": [],
            "used_vision": False,
            "compliant": None,
            "violation_reason": "",
        }

    def _record_turn_vision_read(self, path: str) -> None:
        if not path:
            return
        filename = Path(path).name
        if filename not in {ANNOTATED_FRAME_NAME, RAW_FRAME_NAME}:
            return
        sequence = self.current_turn_vision.setdefault("read_sequence", [])
        if filename == ANNOTATED_FRAME_NAME and not self.current_turn_vision.get("annotated_read"):
            self.current_turn_vision["annotated_read"] = True
            sequence.append(path)
        elif filename == RAW_FRAME_NAME and not self.current_turn_vision.get("raw_read"):
            self.current_turn_vision["raw_read"] = True
            sequence.append(path)
        self.current_turn_vision["used_vision"] = bool(
            self.current_turn_vision.get("annotated_read") or self.current_turn_vision.get("raw_read")
        )

    def _finalize_turn_vision(self) -> JsonDict:
        vision = dict(self.current_turn_vision)
        annotated_available = bool(vision.get("annotated_available"))
        raw_available = bool(vision.get("raw_available"))
        annotated_read = bool(vision.get("annotated_read"))
        raw_read = bool(vision.get("raw_read"))
        vision["used_vision"] = annotated_read or raw_read

        violation_reason = ""
        if annotated_available and not annotated_read:
            violation_reason = f"{ANNOTATED_FRAME_NAME} was attached but never read"
        elif not annotated_available and raw_available and not raw_read:
            violation_reason = f"{RAW_FRAME_NAME} was attached but no frame was read"

        vision["violation_reason"] = violation_reason
        vision["compliant"] = not violation_reason
        return vision

    def _current_continue_prompt(self) -> str:
        reason = ""
        if self.last_turn_vision.get("compliant") is False:
            reason = str(self.last_turn_vision.get("violation_reason") or "")
        return continue_supervisor_prompt(vision_violation_reason=reason)

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
            "turn_plan_preview": self.turn_plan_preview,
            "next_auto_continue_at": self.next_auto_continue_at,
            "session_usage": self.session_usage,
            "last_message_usage": self.last_message_usage,
            "model_limits": self.model_limits,
            "counts": {
                "tool_calls": self.tool_call_count,
                "thinking_blocks": self.thinking_block_count,
                "assistant_messages": self.assistant_message_count,
                "user_messages": self.user_message_count,
            },
            "compaction": {
                "tokens_before": self.last_compaction_tokens_before,
                "tokens_after": self.last_compaction_tokens_after,
                "at": self.last_compaction_at,
            },
            "vision": {
                "current_turn": self.current_turn_vision,
                "last_turn": self.last_turn_vision,
                "violations": self.vision_violation_count,
            },
            "config": self._config_snapshot(),
        }

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
    ) -> JsonDict:
        if self.is_running:
            raise ValueError("Pi supervisor is already running.")
        if not self.available or not self.pi_binary:
            raise ValueError("Pi executable was not found on PATH.")

        chosen_skill = Path(skill_path).expanduser().resolve() if skill_path else self.skill_path
        if not chosen_skill.exists():
            raise ValueError(f"Pi skill not found: {chosen_skill}")

        self.skill_path = chosen_skill
        self.provider = provider or None
        self.model = model or None
        self.thinking = thinking or None
        self.auto_continue = bool(auto_continue)
        self.goal = (goal or "").strip()
        self.max_turns = max_turns if max_turns and max_turns > 0 else None
        self.continue_delay_seconds = max(0.0, float(continue_delay_seconds))
        self.default_prompt = default_supervisor_prompt(
            server_url=self.server_url,
            workspace_dir=self.workspace_dir,
            goal=self.goal,
        )
        initial_prompt = self.default_prompt

        self.status = "starting"
        self.status_reason = "Starting a fresh Pi session."
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
        self.turn_plan_preview = None
        self.turns_completed = 0
        self.continue_count = 0
        self.session_id = None
        self.session_file = None
        self.current_pid = None
        self.next_auto_continue_at = None
        self.session_usage = None
        self.last_message_usage = None
        self.model_limits = None
        self.tool_call_count = 0
        self.thinking_block_count = 0
        self.assistant_message_count = 0
        self.user_message_count = 0
        self.last_compaction_tokens_before = None
        self.last_compaction_tokens_after = None
        self.last_compaction_at = None
        self._pending_thinking_in_message = False
        self.current_turn_vision = self._new_turn_vision([])
        self.last_turn_vision = self._new_turn_vision([])
        self.vision_violation_count = 0
        self._stop_requested = False
        await self._refresh_model_limits()

        await self._emit_major(
            "pi_supervisor_status",
            {
                "status": self.status,
                "summary": self.status_reason,
                "config": self._config_snapshot(),
            },
        )
        self._task = asyncio.create_task(
            self._run_loop(initial_prompt=initial_prompt, resume=False)
        )
        return self.state_snapshot()

    async def continue_once(self) -> JsonDict:
        if self.is_running:
            raise ValueError("Pi supervisor is already running.")
        if not self.session_id:
            raise ValueError("Pi supervisor has no previous session to continue.")
        self.current_prompt = self._current_continue_prompt()
        self.last_prompt = self.current_prompt
        self.status = "starting"
        self.status_reason = "Continuing the existing Pi session."
        self.last_error = None
        self.last_event_at = utc_now()
        self.next_auto_continue_at = None
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.current_tool_calls.clear()
        self.current_turn_vision = self._new_turn_vision(self._vision_attachment_paths())
        await self._refresh_model_limits()
        await self._emit_major(
            "pi_supervisor_status",
            {
                "status": self.status,
                "summary": self.status_reason,
                "config": self._config_snapshot(),
            },
        )
        self._task = asyncio.create_task(
            self._run_loop(initial_prompt=self.current_prompt, resume=True, force_single_turn=True)
        )
        return self.state_snapshot()

    async def stop(self) -> JsonDict:
        self._stop_requested = True
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
        process = self._process
        if process and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=4)
            if process.returncode is None:
                process.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=2)
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=6)
        if self.status not in {"stopped", "completed", "error"}:
            self.status = "stopped"
            self.status_reason = "Pi supervisor stopped."
        self.current_pid = None
        self.next_auto_continue_at = None
        return self.state_snapshot()

    async def shutdown(self) -> None:
        await self.stop()

    async def wait_until_idle(self, timeout: float = 30.0) -> None:
        task = self._task
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def _run_loop(
        self,
        *,
        initial_prompt: str,
        resume: bool,
        force_single_turn: bool = False,
    ) -> None:
        current_prompt = initial_prompt
        use_continue = resume
        try:
            while True:
                await self._run_turn(prompt=current_prompt, resume=use_continue)
                if self._stop_requested:
                    self.status = "stopped"
                    self.status_reason = "Pi supervisor stopped."
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
                self.continue_count += 1
                current_prompt = self._current_continue_prompt()
                use_continue = True
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
                await asyncio.sleep(self.continue_delay_seconds)
        except asyncio.CancelledError:
            self.status = "stopped"
            self.status_reason = "Pi supervisor task was cancelled."
            raise
        except Exception as exc:  # noqa: BLE001
            self.status = "error"
            self.status_reason = "Pi supervisor encountered an error."
            self.last_error = str(exc)
            await self._emit_major(
                "pi_supervisor_error",
                {
                    "summary": str(exc),
                },
            )
        finally:
            self.current_pid = None
            self._process = None
            self.next_auto_continue_at = None
            await self._emit_major(
                "pi_supervisor_status",
                {
                    "status": self.status,
                    "summary": self.status_reason,
                    "last_error": self.last_error,
                    "turns_completed": self.turns_completed,
                },
            )
            self._task = None

    async def _run_turn(self, *, prompt: str, resume: bool) -> None:
        attachment_paths = self._vision_attachment_paths()
        session_file = self._resolve_session_file() if resume else None
        command = self._build_command(
            prompt=prompt,
            resume=resume,
            attachment_paths=attachment_paths,
            session_file=session_file,
        )
        self.status = "running"
        self.status_reason = "Pi is processing a turn."
        self.next_auto_continue_at = None
        self._current_turn_completed = False
        self.current_assistant_text = ""
        self.current_assistant_thinking = ""
        self.current_tool_calls.clear()
        self.current_turn_vision = self._new_turn_vision(attachment_paths)
        self.last_event_at = utc_now()
        prompt_entry = self._append_transcript(
            direction="outbound",
            role="user",
            channel="prompt",
            content=prompt,
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
                "resume": resume,
                "session_id": self.session_id,
            },
        )
        await self._emit_stream("pi_transcript", {"entry": prompt_entry})
        await self._emit_major(
            "pi_turn_launch",
            {
                "summary": f"Launching Pi turn with prompt: {_truncate(prompt, 140)}",
                "command_preview": command,
                "resume": resume,
            },
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.workspace_dir),
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

        stdout_task = asyncio.create_task(self._read_stdout(process))
        stderr_task = asyncio.create_task(self._read_stderr(process))
        returncode = await process.wait()
        await stdout_task
        await stderr_task
        if returncode == 0 and not self._stop_requested and not self._current_turn_completed:
            await self._complete_turn(
                summary_text=self.current_assistant_text or self.last_assistant_text or None,
                tool_result_count=0,
                synthetic=True,
            )
        self.current_pid = None
        self._process = None
        if returncode != 0 and not self._stop_requested:
            stderr_preview = "\n".join(self.stderr_tail).strip()
            raise RuntimeError(
                f"Pi exited with status {returncode}."
                + (f" stderr: {stderr_preview}" if stderr_preview else "")
            )

    def _vision_attachment_paths(self) -> list[Path]:
        attachment_paths: list[Path] = []
        for filename in VISION_ATTACHMENT_FILES:
            candidate = self.workspace_dir / filename
            if candidate.is_file():
                attachment_paths.append(candidate)
        return attachment_paths

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
                header = json.loads(candidate.read_text(encoding="utf-8").splitlines()[0])
            except (IndexError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if header.get("type") == "session" and header.get("id") == self.session_id:
                self.session_file = candidate.resolve()
                return self.session_file
        return None

    def _build_command(
        self,
        *,
        prompt: str,
        resume: bool,
        attachment_paths: Optional[list[Path]] = None,
        session_file: Optional[Path] = None,
    ) -> list[str]:
        assert self.pi_binary is not None
        command = [
            self.pi_binary,
            "--mode",
            "json",
            "--print",
            "--session-dir",
            str(self.session_dir),
            "--skill",
            str(self.skill_path),
            "--tools",
            ",".join(DEFAULT_TOOLS),
        ]
        if session_file is not None:
            command.extend(["--session", str(session_file)])
        elif resume:
            command.append("--continue")
        if self.provider:
            command.extend(["--provider", self.provider])
        if self.model:
            command.extend(["--model", self.model])
        if self.thinking:
            command.extend(["--thinking", self.thinking])
        for path in attachment_paths or []:
            command.append(f"@{path}")
        command.append(prompt)
        return command

    async def _read_stdout(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            raw_line = line.decode("utf-8", errors="replace").strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                clipped = _clip_text(raw_line, 1200)
                self.stderr_tail.append(_truncate(clipped, 240))
                entry = self._append_transcript(
                    direction="system",
                    role="system",
                    channel="stdout_parse_error",
                    content=clipped,
                )
                await self._emit_stream(
                    "pi_stdout_parse_error",
                    {
                        "line": entry["content"],
                    },
                )
                await self._emit_stream("pi_transcript", {"entry": entry})
                continue
            await self._handle_event(payload)

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            clipped = _clip_text(text, 1200)
            self.stderr_tail.append(_truncate(clipped, 240))
            entry = self._append_transcript(
                direction="system",
                role="system",
                channel="stderr",
                content=clipped,
            )
            await self._emit_stream(
                "pi_stderr",
                {
                    "text": entry["content"],
                },
            )
            await self._emit_stream("pi_transcript", {"entry": entry})

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
            "content": _clip_text(content),
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
        synthetic: bool = False,
    ) -> None:
        if self._current_turn_completed:
            return
        self._current_turn_completed = True
        self.turns_completed += 1
        self.last_turn_completed_at = utc_now()
        self.latest_turn_summary = _truncate(summary_text or "Pi completed a turn.", 220)
        vision = self._finalize_turn_vision()
        self.last_turn_vision = vision
        if not vision.get("compliant"):
            self.vision_violation_count += 1
            self.latest_turn_summary = _truncate(
                f"{self.latest_turn_summary} Vision policy violated: {vision['violation_reason']}.",
                220,
            )
        payload: JsonDict = {
            "summary": self.latest_turn_summary,
            "turns_completed": self.turns_completed,
            "tool_result_count": tool_result_count,
            "vision": vision,
        }
        if synthetic:
            payload["synthetic"] = True
            payload["summary"] = _truncate(
                self.latest_turn_summary + " (turn_end missing from Pi event stream)",
                260,
            )
        await self._emit_major("pi_turn_end", payload)
        if not vision.get("compliant"):
            await self._emit_major(
                "pi_turn_vision_violation",
                {
                    "summary": vision["violation_reason"],
                    "vision": vision,
                    "turns_completed": self.turns_completed,
                },
            )

    async def _handle_event(self, event: JsonDict) -> None:
        event_type = event.get("type")
        self.last_event_at = utc_now()
        if event_type == "session":
            self.session_id = event.get("id")
            self.session_file = self._resolve_session_file()
            self._push_recent_event(
                "pi_session",
                f"Session {self.session_id or 'unknown'} started.",
            )
            return

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
            await self._emit_major("pi_agent_start", {"summary": "Pi agent turn started."})
            return

        if event_type == "agent_end":
            summary = (
                self.current_assistant_text
                or self.last_assistant_text
                or self.latest_turn_summary
                or "Pi agent turn ended."
            )
            await self._emit_major("pi_agent_end", {"summary": summary})
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
            summary_text = message_text or self.current_assistant_text or self.last_assistant_text
            await self._complete_turn(
                summary_text=summary_text,
                tool_result_count=len(event.get("toolResults") or []),
            )
            return

        if event_type == "message_start":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                self.current_assistant_text = ""
                self.current_assistant_thinking = ""
                self.assistant_message_count += 1
                self._pending_thinking_in_message = False
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
                total_tokens = usage.get("totalTokens")
                if isinstance(total_tokens, (int, float)):
                    self.session_usage = {
                        "input": usage.get("input"),
                        "output": usage.get("output"),
                        "cacheRead": usage.get("cacheRead"),
                        "cacheWrite": usage.get("cacheWrite"),
                        "totalTokens": int(total_tokens),
                        "updated_at": utc_now(),
                    }
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
            self.current_tool_calls[event.get("toolCallId", summary)] = entry
            self.tool_call_count += 1
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
            if entry["status"] == "completed" and entry.get("tool_name") == "read":
                file_hint = entry.get("file_hint")
                if isinstance(file_hint, str):
                    self._record_turn_vision_read(file_hint)
            self.recent_tools.append(entry)
            self._refresh_turn_plan_preview_from_workspace()
            summary = entry["summary"]
            if entry["status"] == "error":
                summary = f"{summary} failed"
            await self._emit_major(
                "pi_tool_end",
                {
                    "summary": summary,
                    "tool_name": entry.get("tool_name"),
                    "is_error": bool(event.get("isError")),
                    "result_preview": entry["result_preview"],
                },
            )
            if entry["status"] == "completed" and entry.get("tool_name") == "read":
                file_hint = entry.get("file_hint")
                if isinstance(file_hint, str) and Path(file_hint).name in {
                    ANNOTATED_FRAME_NAME,
                    RAW_FRAME_NAME,
                }:
                    await self._emit_major(
                        "pi_vision_read",
                        {
                            "summary": f"Vision read: {Path(file_hint).name}",
                            "path": file_hint,
                            "vision": dict(self.current_turn_vision),
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
            return

        if event_type == "compaction_end":
            summary = f"Compaction finished ({event.get('reason', 'unknown')})."
            if event.get("aborted"):
                summary = f"Compaction aborted ({event.get('reason', 'unknown')})."
            tokens_after = event.get("tokensAfter")
            if isinstance(tokens_after, (int, float)):
                self.last_compaction_tokens_after = int(tokens_after)
                if self.session_usage is None:
                    self.session_usage = {}
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
            return

        if event_type == "auto_retry_start":
            await self._emit_major(
                "pi_auto_retry_start",
                {
                    "summary": (
                        f"Auto-retry {event.get('attempt')}/{event.get('maxAttempts')} "
                        f"after error: {_truncate(str(event.get('errorMessage', 'unknown')), 180)}"
                    ),
                },
            )
            return

        if event_type == "auto_retry_end":
            await self._emit_major(
                "pi_auto_retry_end",
                {
                    "summary": (
                        "Auto-retry succeeded."
                        if event.get("success")
                        else _truncate(
                            f"Auto-retry failed: {event.get('finalError', 'unknown')}",
                            220,
                        )
                    ),
                },
            )
            return

        self._push_recent_event(
            f"pi_{event_type or 'event'}",
            preview_payload(event, 180),
            event,
        )
