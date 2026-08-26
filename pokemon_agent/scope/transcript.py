"""Reading a Pi session transcript into something countable.

The file on disk is one JSON object per line, appended while the agent plays:
a ``session`` header, ``model_change`` / ``thinking_level_change`` markers, then
alternating ``message`` records — an assistant message carrying ``toolCall``
blocks and a usage report, followed by a ``toolResult`` message carrying the
output. Images ride inline as base64 in either direction.

Two facts shape this module. First, the newest file is being written to right
now, so the reader takes one snapshot of the bytes and treats a final unparseable
line as the writer mid-``write`` rather than as corruption. Second, nothing
downstream ever wants the base64 — so image blocks are reduced to their
dimensions and byte count at parse time and the payload is dropped on the floor.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

#: Segment separators in a bash command line, honoured outside quotes only.
_SEPARATORS = ";\n"
_PAIRED_SEPARATORS = ("&&", "||", "|", "&")

#: ``./poke frontier``, ``poke act up``, ``bash poke state`` — the verb is the
#: token after the program, and that is the unit the harness ships features in.
_POKE_RE = re.compile(r"(?:^|[\s;&|()])(?:\./)?poke\s+([a-z][a-z0-9_-]*)")

#: Leading ``VAR=value`` assignments, including ``p="$(cat x)"``, which name no
#: program at all and would otherwise be mistaken for one.
_ASSIGN_PREFIX_RE = re.compile(r"^\s*(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)\s*)+")

#: Programs that only get somewhere else. The interesting command is the next
#: one along, and 156 bash calls reading ``!cd`` would say nothing at all.
_PASSTHROUGH_PROGRAMS = frozenset({"cd", "export", "set", "source", ".", "unset", "pushd"})

#: Interpreters worth naming their script after: ``!python3`` is a hundred
#: different commands, ``!python3 harness_post.py`` is one.
_INTERPRETERS = frozenset({"bash", "sh", "zsh", "python", "python3", "node", "uv"})

#: Verbs the harness has shipped as advice the model is free to ignore. Whether
#: they are ever called is the question this whole package exists to answer.
ADVISORY_VERBS: tuple[str, ...] = (
    "route",
    "goto",
    "calc",
    "sim",
    "frontier",
    "guide",
    "progress",
)


def parse_timestamp(value: Any) -> Optional[float]:
    """Epoch seconds from either an ISO-8601 string or a millisecond integer."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        # Transcript message stamps are milliseconds; run ids are seconds.
        return number / 1000.0 if number > 1e11 else number
    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def iter_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """``(objects, unparseable_lines)`` from a file that may be growing.

    One ``read_bytes`` is the snapshot. A trailing line that does not parse is
    the writer caught mid-append and is not counted against the file; anything
    unparseable earlier is real damage and is counted.
    """

    try:
        raw = path.read_bytes()
    except OSError:
        return [], 0
    lines = raw.split(b"\n")
    objects: list[dict[str, Any]] = []
    corrupt = 0
    last = len(lines) - 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            if index != last:
                corrupt += 1
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        elif index != last:
            corrupt += 1
    return objects, corrupt


# -- images -------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(data_b64: str) -> Optional[tuple[int, int]]:
    """``(width, height)`` from the first few base64 characters of a PNG.

    The IHDR chunk holds both in bytes 16..24, so 32 base64 characters are
    enough and the megabyte behind them is never decoded.
    """

    head = data_b64[:32]
    if len(head) < 32:
        return None
    try:
        blob = base64.b64decode(head, validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(blob) < 24 or not blob.startswith(_PNG_SIGNATURE):
        return None
    try:
        width, height = struct.unpack(">II", blob[16:24])
    except struct.error:
        return None
    if not (0 < width <= 1 << 16 and 0 < height <= 1 << 16):
        return None
    return int(width), int(height)


def base64_bytes(data_b64: str) -> int:
    """Decoded size of a base64 payload, without decoding it."""

    length = len(data_b64)
    if length == 0:
        return 0
    padding = len(data_b64) - len(data_b64.rstrip("="))
    return max(0, (length // 4) * 3 - padding)


@dataclass(frozen=True)
class ImageRef:
    """One image that crossed the wire, minus the megabyte of base64."""

    mime_type: str
    nbytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    #: "toolResult" when the model read a frame, "user" when one was attached.
    origin: str = "toolResult"

    @property
    def pixels(self) -> Optional[int]:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height


def _image_ref(block: dict[str, Any], origin: str) -> ImageRef:
    data = block.get("data")
    data = data if isinstance(data, str) else ""
    dims = png_dimensions(data)
    return ImageRef(
        mime_type=str(block.get("mimeType") or ""),
        nbytes=base64_bytes(data),
        width=dims[0] if dims else None,
        height=dims[1] if dims else None,
        origin=origin,
    )


# -- commands -----------------------------------------------------------------


def split_segments(command: str) -> list[str]:
    """Break a bash line into its top-level commands, respecting quotes.

    Good enough is the goal: a python heredoc full of semicolons must not turn
    into forty phantom commands, and ``./poke state; ./poke route X`` must turn
    into two.
    """

    segments: list[str] = []
    buffer: list[str] = []
    quote: Optional[str] = None
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if quote is not None:
            buffer.append(char)
            if char == "\\" and quote == '"' and index + 1 < length:
                buffer.append(command[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            buffer.append(char)
            index += 1
            continue
        matched = next(
            (pair for pair in _PAIRED_SEPARATORS if command.startswith(pair, index)), None
        )
        if matched is not None:
            segments.append("".join(buffer))
            buffer = []
            index += len(matched)
            continue
        if char in _SEPARATORS:
            segments.append("".join(buffer))
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    segments.append("".join(buffer))
    return [segment.strip() for segment in segments if segment.strip()]


def program_name(segment: str) -> str:
    """The executable a segment runs, or ``""`` if it runs nothing.

    A comment, a bare assignment and an empty segment all name no program, and
    each of them would otherwise show up in the histogram as a program called
    ``#``, ``p=`` or the empty string.
    """

    body = _ASSIGN_PREFIX_RE.sub("", segment).strip()
    if not body or body.startswith("#"):
        return ""
    tokens = body.split()
    name = tokens[0].split("/")[-1].strip("(){}`\"'")
    if name in _INTERPRETERS and len(tokens) > 1:
        argument = tokens[1]
        detail = argument if argument.startswith("-") else argument.split("/")[-1]
        return f"{name} {detail}"
    return name


def bash_program(command: str) -> str:
    """What a bash line is actually *for*, skipping the getting-there."""

    for segment in split_segments(command):
        name = program_name(segment)
        if not name or name.split()[0] in _PASSTHROUGH_PROGRAMS:
            continue
        return name
    return "bash"


def poke_verbs(command: str) -> list[str]:
    """Every ``poke <verb>`` this bash line invokes, in order."""

    return [match.group(1) for match in _POKE_RE.finditer(command)]


@dataclass(frozen=True)
class Call:
    """One tool call and the result that came back for it."""

    step: int
    tool: str
    command: str = ""
    #: What the call is counted as: ``poke <verb>``, ``!<program>`` for other
    #: bash, or the bare tool name for everything that is not bash.
    label: str = ""
    #: The label plus its arguments, which is what a loop repeats.
    signature: str = ""
    kind: str = "tool"  # "poke" | "bash" | "tool"
    result_text: str = ""
    result_bytes: int = 0
    is_error: bool = False
    images: tuple[ImageRef, ...] = ()
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return max(0.0, self.ended_at - self.started_at)

    @property
    def result_json(self) -> Optional[dict[str, Any]]:
        """``./poke act`` answers with one JSON object; most verbs do not."""

        text = self.result_text.strip()
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text.split("\n", 1)[0])
        except (json.JSONDecodeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class Usage:
    """What the provider reported it charged for one assistant message."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0

    @property
    def prompt(self) -> int:
        """Everything the model had to read: fresh input plus cache hits.

        This is the occupancy of the context window at that step, and it is a
        reported number rather than an estimate of one.
        """

        return self.input + self.cache_read

    @classmethod
    def from_dict(cls, payload: Any) -> "Usage":
        if not isinstance(payload, dict):
            return cls()

        def number(key: str) -> int:
            value = payload.get(key)
            return (
                int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
            )

        return cls(
            input=number("input"),
            output=number("output"),
            cache_read=number("cacheRead"),
            cache_write=number("cacheWrite"),
            total=number("totalTokens"),
        )


@dataclass
class Step:
    """One assistant message and everything that came back from it."""

    index: int
    at: Optional[float] = None
    usage: Usage = field(default_factory=Usage)
    text: str = ""
    thinking_chars: int = 0
    calls: list[Call] = field(default_factory=list)
    #: Images attached to the *user* message that preceded this step, if any.
    prompt_images: tuple[ImageRef, ...] = ()

    @property
    def images(self) -> list[ImageRef]:
        out = list(self.prompt_images)
        for call in self.calls:
            out.extend(call.images)
        return out


@dataclass(frozen=True)
class Injection:
    """A user-role message that arrived after the session was already playing.

    The goal prompt is the first one and is not an injection. Everything after
    it was pushed in from outside while the model was mid-run: an intervention's
    advice, an operator nudge, or the harness's own ``continue``. They are the
    only record on disk that an intervention happened at all, so they are kept
    with the step they landed after.
    """

    #: Index of the next assistant step, i.e. the first one that could react.
    step: int
    at: Optional[float]
    text: str

    @property
    def is_continue(self) -> bool:
        """``continue`` is the harness restarting a stalled turn, not advice."""

        return self.text.strip().lower() in {"continue", "continue.", "go on"}

    @property
    def headline(self) -> str:
        for line in self.text.splitlines():
            if line.strip():
                return line.strip()
        return ""


@dataclass
class Session:
    """A whole transcript, parsed."""

    path: Path
    session_id: str = ""
    started_at: Optional[float] = None
    model: str = ""
    provider: str = ""
    thinking_level: str = ""
    goal: str = ""
    steps: list[Step] = field(default_factory=list)
    #: User messages that arrived after the first one, oldest first.
    injections: list[Injection] = field(default_factory=list)
    corrupt_lines: int = 0
    truncated_tail: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def short_id(self) -> str:
        return self.session_id[:8] if self.session_id else self.path.stem[:8]

    @property
    def ended_at(self) -> Optional[float]:
        for step in reversed(self.steps):
            for call in reversed(step.calls):
                if call.ended_at is not None:
                    return call.ended_at
            if step.at is not None:
                return step.at
        return self.started_at

    @property
    def elapsed(self) -> float:
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    @property
    def calls(self) -> list[Call]:
        return [call for step in self.steps for call in step.calls]


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _describe(tool: str, arguments: dict[str, Any]) -> tuple[str, str, str, str]:
    """``(command, label, signature, kind)`` for one tool call."""

    if tool == "bash":
        command = str(arguments.get("command") or "")
        verbs = poke_verbs(command)
        if verbs:
            label = f"poke {verbs[0]}"
            signature = " ; ".join(f"poke {verb}" for verb in verbs)
            # Keep the arguments of a single-verb call: ``act up`` and ``act
            # down`` alternating is a loop, ``act`` alone is just movement.
            if len(verbs) == 1:
                for segment in split_segments(command):
                    found = poke_verbs(segment)
                    if found:
                        tokens = segment.replace("./poke", "poke").split()
                        start = tokens.index("poke") if "poke" in tokens else 0
                        signature = " ".join(tokens[start : start + 5]).lower()
                        break
            return command, label, signature, "poke"
        label = f"!{bash_program(command)}"
        return command, label, label, "bash"
    if tool in {"read", "write", "edit"}:
        path = str(arguments.get("path") or "")
        return path, tool, f"{tool} {Path(path).name}" if path else tool, "tool"
    if tool == "grep":
        return str(arguments.get("pattern") or ""), tool, tool, "tool"
    return "", tool or "?", tool or "?", "tool"


def parse_session(path: Path) -> Session:
    """Read one transcript off disk. Never raises on a malformed file."""

    objects, corrupt = iter_jsonl(path)
    session = Session(path=path, corrupt_lines=corrupt)

    steps: list[Step] = []
    by_call_id: dict[str, Call] = {}
    pending_images: list[ImageRef] = []

    for payload in objects:
        kind = payload.get("type")
        if kind == "session":
            session.session_id = str(payload.get("id") or "")
            session.started_at = parse_timestamp(payload.get("timestamp"))
            continue
        if kind == "model_change":
            session.provider = str(payload.get("provider") or "")
            session.model = str(payload.get("modelId") or "")
            continue
        if kind == "thinking_level_change":
            session.thinking_level = str(payload.get("thinkingLevel") or "")
            continue
        if kind != "message":
            continue

        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        blocks = _content_blocks(message)
        at = parse_timestamp(payload.get("timestamp"))

        if role == "user":
            texts: list[str] = []
            for block in blocks:
                if block.get("type") == "image":
                    pending_images.append(_image_ref(block, "user"))
                elif block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
            body = "\n".join(part for part in texts if part).strip()
            if body and not session.goal:
                session.goal = body
            elif body:
                session.injections.append(Injection(step=len(steps), at=at, text=body))
            continue

        if role == "assistant":
            step = Step(
                index=len(steps),
                at=at,
                usage=Usage.from_dict(message.get("usage")),
                prompt_images=tuple(pending_images),
            )
            pending_images = []
            texts: list[str] = []
            for block in blocks:
                block_type = block.get("type")
                if block_type == "text":
                    texts.append(str(block.get("text") or ""))
                elif block_type == "thinking":
                    step.thinking_chars += len(
                        str(block.get("thinking") or block.get("text") or "")
                    )
                elif block_type == "toolCall":
                    tool = str(block.get("name") or "")
                    arguments = block.get("arguments")
                    arguments = arguments if isinstance(arguments, dict) else {}
                    command, label, signature, call_kind = _describe(tool, arguments)
                    call = Call(
                        step=step.index,
                        tool=tool,
                        command=command,
                        label=label,
                        signature=signature,
                        kind=call_kind,
                        started_at=at,
                    )
                    step.calls.append(call)
                    call_id = block.get("id")
                    if isinstance(call_id, str) and call_id:
                        by_call_id[call_id] = call
            step.text = "\n".join(part for part in texts if part).strip()
            steps.append(step)
            continue

        if role == "toolResult":
            call_id = message.get("toolCallId")
            call = by_call_id.get(call_id) if isinstance(call_id, str) else None
            texts: list[str] = []
            images: list[ImageRef] = []
            for block in blocks:
                if block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
                elif block.get("type") == "image":
                    images.append(_image_ref(block, "toolResult"))
            text = "\n".join(texts)
            if call is None:
                continue
            updated = Call(
                step=call.step,
                tool=call.tool,
                command=call.command,
                label=call.label,
                signature=call.signature,
                kind=call.kind,
                result_text=text,
                result_bytes=len(text.encode("utf-8", "replace")),
                is_error=bool(message.get("isError")),
                images=tuple(images),
                started_at=call.started_at,
                ended_at=at,
            )
            step_calls = steps[call.step].calls
            for position, existing in enumerate(step_calls):
                if existing is call:
                    step_calls[position] = updated
                    break
            if isinstance(call_id, str):
                by_call_id[call_id] = updated

    session.steps = steps
    if session.started_at is None and steps:
        session.started_at = steps[0].at
    return session


def load_sessions(paths: Iterable[Path]) -> list[Session]:
    return [parse_session(path) for path in paths]


def median(values: Sequence[float]) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0
