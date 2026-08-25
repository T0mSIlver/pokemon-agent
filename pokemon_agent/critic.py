"""One-shot retrospective run between sessions.

The watchdog starts a fresh Pi session every time a run ends, and the emulator
keeps the game while the context is thrown away. This module spends one
``pi --print`` call at ``--thinking xhigh`` on reading what the finished session
actually did, and leaves a short ``HANDOFF.md`` for the next thinking-off agent.

Everything here is best-effort. A critic that errors, times out or says nothing
leaves the previous handoff in place and must never hold up the next session.

The run is streamed rather than buffered: events reach the supervisor as pi emits
them, the raw stream lands in ``<workspace>/debug/critic_last.jsonl`` whatever
happens, and a pass that reasons past its output ceiling is retried once, cheaper,
before its reasoning tail is salvaged into the handoff.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator, Optional

from pokemon_agent.agent_cli import ActionError, expand_actions
from pokemon_agent.pi_supervisor import (
    IMAGE_SUFFIXES,
    extract_leading_comment,
    extract_message_text,
    extract_message_thinking,
    first_nonempty_line,
    iter_jsonl_records,
    iter_stream_lines,
)

JsonDict = dict[str, Any]
#: Awaited with each critic event as it arrives, so a long critique is watchable.
CriticEventSink = Callable[[JsonDict], Awaitable[None]]

DEFAULT_CRITIC_TIMEOUT_SECONDS = 600.0
DEFAULT_CRITIC_THINKING = "xhigh"
#: The cheaper level the retry runs at when the first pass never reaches an answer.
DEFAULT_CRITIC_RETRY_THINKING = "medium"
#: A retry is only worth starting with at least this much of the budget left.
CRITIC_RETRY_MIN_SECONDS = 45.0
CRITIC_TOOLS = ["read"]

HANDOFF_FILENAME = "HANDOFF.md"
HANDOFF_PREVIOUS_FILENAME = "HANDOFF.prev.md"
HANDOFF_HEADING = "Retrospective from your previous session"
NOTES_FILENAME = "NOTES.md"

#: The critic's raw event stream, kept across one rotation, for post-mortems.
DEBUG_DIRNAME = "debug"
CRITIC_RAW_FILENAME = "critic_last.jsonl"
CRITIC_RAW_PREVIOUS_FILENAME = "critic_prev.jsonl"

NO_TEXT_ERROR = "Critic produced no text."
SALVAGED_REASONING_NOTICE = (
    "**Salvaged from the critic's truncated reasoning.** It ran out of output budget before it "
    "wrote an answer, so this is the tail of what it was thinking, not a finished retrospective."
)
SALVAGED_ANSWER_NOTICE = (
    "**Salvaged from a truncated critic reply.** The run was cut off part-way through the "
    "retrospective below."
)

#: Images the critic gets to look at, in the order pi receives them.
CRITIC_IMAGE_FILES = ("latest_map.png", "latest_frame_annotated.png")

#: Hard ceiling on the handoff, in words. The next session reads this text exactly
#: once, against a 110k-token context budget: 900 words is roughly 1,200 tokens, or
#: about 1% of that budget. The old 300-word cap saved ~800 tokens and paid for it
#: by chopping the retrospective mid-sentence, which routinely lost the concrete
#: "do this differently next time" list at the end - the single most valuable thing
#: the critic produces. Spending the tokens is the better trade every time.
MAX_HANDOFF_WORDS = 900
#: What the critic is *asked* for. The ceiling above is a backstop, not the target,
#: so a critic that lands slightly long still gets to finish its own last sentence.
TARGET_HANDOFF_WORDS = 500

#: Appended when the cap does bite, in place of a bare "...", so a shortened handoff
#: says that it was shortened and how much is missing.
TRUNCATION_MARKER = "\n\n_[truncated at the last {boundary}: {dropped} more words not shown]_"
#: The same idea for a reasoning tail, where the *start* is what got cut away.
TAIL_TRUNCATION_MARKER = "_[truncated: {dropped} earlier words not shown; picking up {boundary}]_"

#: The line the critic ends on, naming the goal for the session after it.
NEXT_GOAL_LABEL = "NEXT GOAL"
#: Longest next-goal line accepted. A goal is one instruction, not a plan; anything
#: longer than this is the critic wandering off format, and is dropped.
MAX_NEXT_GOAL_CHARS = 240
#: Shorter than this is not an instruction ("done", "n/a", a stray colon).
MIN_NEXT_GOAL_CHARS = 8

#: Whole-digest ceiling. ~4 chars per token is the usual rough conversion.
DIGEST_TOKEN_BUDGET = 12_000
CHARS_PER_TOKEN = 4
DIGEST_CHAR_BUDGET = DIGEST_TOKEN_BUDGET * CHARS_PER_TOKEN

RECENT_TOOL_CALLS = 60
MAX_NARRATION_LINES = 40
NOTES_CHAR_LIMIT = 4_000
TOP_TILES = 6
CALL_LINE_LIMIT = 200
#: Only the head of a tool result is scanned for the `/action` response body.
RESULT_SCAN_LIMIT = 8_000

_ACTIONS_RE = re.compile(r'"actions"\s*:\s*(\[[^\]]*\])')
#: ``./poke act up up a`` — the CLI that replaced hand-written curl. Anchored so
#: prose like "# poke around the sign" is not read as a command.
_POKE_RE = re.compile(r"(?m)(?:\./poke|(?:^|[;&|])\s*poke)\s+([^\n|;&]*)")
#: The options `poke` accepts on either side of its subcommand. Each takes a
#: value, so `poke --port 9000 act up` still runs `act` with one button.
_POKE_GLOBAL_OPTIONS = frozenset({"--port", "--url"})
_ACTION_RESPONSE_KEYS = ("moved", "blocked_after", "facing", "on_warp", "mode", "battle")


def estimate_tokens(text: str) -> int:
    """Rough token count for a prompt. Deliberately crude and slightly pessimistic."""

    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Reading the session back
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool call as the supervisor's stream log recorded it."""

    name: str
    headline: str = ""
    command: str = ""
    path: str = ""
    result_summary: str = ""
    result_full: str = ""
    is_error: bool = False

    @property
    def comment(self) -> str:
        return extract_leading_comment(self.command) if self.command else ""


def tool_calls_from_stream(entries: Iterable[JsonDict]) -> list[ToolCall]:
    """Pull the tool entries out of ``PiSupervisor.stream_entries``."""

    calls: list[ToolCall] = []
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("kind") != "tool":
            continue
        tool = entry.get("tool")
        if not isinstance(tool, dict):
            continue
        calls.append(
            ToolCall(
                name=str(tool.get("name") or "tool"),
                headline=str(tool.get("headline") or ""),
                command=str(tool.get("command") or ""),
                path=str(tool.get("path") or ""),
                result_summary=str(tool.get("result_summary") or ""),
                result_full=str(tool.get("result_full") or ""),
                is_error=entry.get("state") == "error",
            )
        )
    return calls


def iter_json_objects(text: str, limit: int = RESULT_SCAN_LIMIT) -> Iterator[JsonDict]:
    """Yield every balanced ``{...}`` object found in the head of ``text``."""

    head = text[:limit]
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(head):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    payload = json.loads(head[start : index + 1])
                    if isinstance(payload, dict):
                        yield payload
                start = -1


def action_response(result_text: str) -> Optional[JsonDict]:
    """Find the ``/action`` response body inside a bash tool result.

    Pi wraps a bash result in its own envelope, so the body may be the whole
    text, a JSON-encoded string field, or one object among several.
    """

    if not result_text:
        return None
    for payload in iter_json_objects(result_text):
        if any(key in payload for key in _ACTION_RESPONSE_KEYS):
            return payload
        for value in payload.values():
            if not isinstance(value, str) or "{" not in value:
                continue
            for nested in iter_json_objects(value):
                if any(key in nested for key in _ACTION_RESPONSE_KEYS):
                    return nested
    return None


def _drop_global_options(tokens: list[str]) -> list[str]:
    """The tokens argparse would leave behind after `poke`'s own options."""

    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name, has_value, _ = token.partition("=")
        if name in _POKE_GLOBAL_OPTIONS:
            index += 1 if has_value else 2
            continue
        kept.append(token)
        index += 1
    return kept


def poke_subcommand(command: str) -> Optional[tuple[str, list[str]]]:
    """``(subcommand, arguments)`` for a ``poke`` call, or None."""

    match = _POKE_RE.search(command or "")
    if not match:
        return None
    tokens = _drop_global_options(match.group(1).split())
    if not tokens:
        return None
    return tokens[0], tokens[1:]


def parse_actions(command: str) -> list[str]:
    """The button list a tool call sent, whichever form it used."""

    call = poke_subcommand(command)
    if call and call[0] == "act":
        # The CLI expands the batch or refuses it whole: one bad token and no
        # request is ever sent, so nothing in it reached the game. Expanding it
        # here through the CLI's own function is the only way the two agree.
        try:
            return expand_actions(call[1])
        except ActionError:
            return []
    match = _ACTIONS_RE.search(command or "")
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if isinstance(item, str)]


def classify_call(call: ToolCall) -> str:
    """Bucket a tool call for the tool-mix table."""

    name = call.name.lower()
    if name == "read":
        suffix = Path(call.path).suffix.lower() if call.path else ""
        return "frame_read" if suffix in IMAGE_SUFFIXES else "file_read"
    if name != "bash":
        return name
    command = call.command or ""
    poke = poke_subcommand(command)
    if poke:
        return {
            "act": "action",
            "map": "map",
            "state": "state",
            "save": "save",
            "load": "save",
            "saves": "save",
        }.get(poke[0], "bash")
    if "/action" in command:
        return "action"
    if "/map" in command or "latest_map" in command:
        return "map"
    if "/state" in command:
        return "state"
    if "/save" in command or "/load" in command or "/saves" in command:
        return "save"
    return "bash"


def compute_behaviour_stats(calls: list[ToolCall]) -> JsonDict:
    """Hard numbers about how the session played, not how it described itself."""

    mix: Counter[str] = Counter()
    batches = 0
    buttons = 0
    blocked = 0
    moved_zero = 0
    battles = 0
    in_battle = False
    errors = 0
    positions: list[tuple[int, int]] = []

    for call in calls:
        kind = classify_call(call)
        mix[kind] += 1
        if call.is_error:
            errors += 1
        if kind != "action":
            continue
        batches += 1
        buttons += len(parse_actions(call.command))
        payload = action_response(call.result_full) or {}
        if payload.get("blocked_after") is not None:
            blocked += 1
        moved = payload.get("moved")
        if isinstance(moved, (int, float)) and int(moved) == 0:
            moved_zero += 1
        fighting = bool(payload.get("battle"))
        if fighting and not in_battle:
            battles += 1
        in_battle = fighting
        x, y = payload.get("x"), payload.get("y")
        if isinstance(x, int) and isinstance(y, int):
            positions.append((x, y))

    tiles = Counter(positions)
    return {
        "tool_calls": len(calls),
        "tool_errors": errors,
        "action_batches": batches,
        "total_buttons": buttons,
        "average_batch_size": round(buttons / batches, 1) if batches else 0.0,
        "blocked_batches": blocked,
        "blocked_fraction": round(blocked / batches, 3) if batches else 0.0,
        "moved_zero_batches": moved_zero,
        "battles": battles,
        "positions_sampled": len(positions),
        "distinct_tiles": len(tiles),
        "top_tiles": [
            {"x": x, "y": y, "visits": count} for (x, y), count in tiles.most_common(TOP_TILES)
        ],
        "tool_mix": dict(sorted(mix.items(), key=lambda item: (-item[1], item[0]))),
    }


def narration_lines(calls: list[ToolCall], limit: int = MAX_NARRATION_LINES) -> list[str]:
    """The ``#`` comments the agent wrote above its bash calls, sampled evenly."""

    comments: list[str] = []
    for call in calls:
        comment = call.comment.strip()
        if comment and (not comments or comments[-1] != comment):
            comments.append(comment)
    if len(comments) <= limit:
        return comments
    # Sample across the whole run so the arc survives, not just the tail.
    step = len(comments) / limit
    return [comments[min(len(comments) - 1, int(index * step))] for index in range(limit)]


def compact_actions(actions: list[str]) -> str:
    """``walk_up x4, press_a`` — the buttons, without the curl boilerplate."""

    runs: list[list[Any]] = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])
    return ", ".join(name if count == 1 else f"{name} x{count}" for name, count in runs)


def summarize_action_response(payload: JsonDict) -> str:
    """The fields that say whether a batch worked, in one line."""

    parts = [
        f"{key}={payload[key]}"
        for key in ("x", "y", "facing", "moved", "blocked_after", "hp", "here_before")
        if payload.get(key) is not None
    ]
    if payload.get("battle"):
        parts.append("battle")
    if payload.get("on_warp"):
        parts.append("on_warp")
    return " ".join(parts)


def call_lines(calls: list[ToolCall], limit: int = RECENT_TOOL_CALLS) -> list[str]:
    """One line per tool call: what it tried, and what came back.

    An ``/action`` call is rendered as its buttons and its outcome rather than
    its curl line, which is identical every time and says nothing.
    """

    lines: list[str] = []
    for call in calls[-limit:]:
        marker = "!" if call.is_error else "-"
        if classify_call(call) == "action":
            buttons = compact_actions(parse_actions(call.command))
            head = call.comment or "action"
            if buttons:
                head = f"{head} [{buttons}]"
            payload = action_response(call.result_full)
            result = summarize_action_response(payload) if payload else call.result_summary
        else:
            head = call.headline or first_nonempty_line(call.command) or call.name
            result = call.result_summary or ("error" if call.is_error else "")
        line = f"{marker} {head[: CALL_LINE_LIMIT - 60]}"
        if result:
            line = f"{line}  ->  {result}"
        lines.append(line[:CALL_LINE_LIMIT])
    return lines


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def format_game_state(state: Optional[JsonDict]) -> list[str]:
    """Map, position, party, badges and money - the things progress shows up in."""

    if not isinstance(state, dict) or not state:
        return ["unknown"]
    lines: list[str] = []
    map_info = state.get("map") or {}
    if isinstance(map_info, dict) and map_info.get("map_name"):
        lines.append(f"Map: {map_info.get('map_name')} (id={map_info.get('map_id')})")
    player = state.get("player") or {}
    if isinstance(player, dict) and player:
        position = player.get("position") or {}
        lines.append(f"Position: ({position.get('x')}, {position.get('y')})")
        badges = player.get("badges") or []
        lines.append(f"Badges: {', '.join(badges) if badges else 'none'}")
        money = player.get("money")
        if money is not None:
            lines.append(f"Money: ${money}")
    party = state.get("party") or []
    if isinstance(party, list) and party:
        for index, mon in enumerate(party, start=1):
            if not isinstance(mon, dict):
                continue
            lines.append(
                f"Party {index}: {mon.get('species', '?')} "
                f"L{mon.get('level', '?')} HP {mon.get('hp', '?')}/{mon.get('max_hp', '?')}"
            )
    return lines or ["unknown"]


def format_map_summary(summary: Optional[JsonDict]) -> list[str]:
    """``ExploredMaps.summary()`` as a few lines: shape, coverage, warps."""

    if not isinstance(summary, dict) or not summary:
        return ["no explored-map record for this map"]
    lines = [
        f"Map: {summary.get('map_name')} (id={summary.get('map_id')}) "
        f"{summary.get('width')}x{summary.get('height')}"
    ]
    coverage = summary.get("coverage")
    if isinstance(coverage, dict):
        lines.append(
            "Coverage: " + ", ".join(f"{key}={value}" for key, value in sorted(coverage.items()))
        )
    warps = summary.get("warps") or []
    if warps:
        rendered = ", ".join(f"({warp.get('x')},{warp.get('y')})" for warp in warps[:16])
        lines.append(f"Warps: {rendered}")
    nearest = summary.get("unexplored_nearest")
    if nearest:
        lines.append(f"Nearest unexplored: {json.dumps(nearest, sort_keys=True)}")
    return lines


def _section(title: str, body: Iterable[str]) -> str:
    lines = [line for line in body if line]
    if not lines:
        return ""
    return f"## {title}\n" + "\n".join(lines)


def format_stats(stats: JsonDict) -> list[str]:
    batches = stats.get("action_batches") or 0
    blocked_pct = int(round((stats.get("blocked_fraction") or 0.0) * 100))
    lines = [
        f"- Tool calls: {stats.get('tool_calls')} ({stats.get('tool_errors')} returned an error)",
        f"- /action batches: {batches}",
        f"- Buttons sent: {stats.get('total_buttons')} "
        f"(average batch {stats.get('average_batch_size')})",
        "- Batches that hit something and stopped early (blocked_after): "
        f"{stats.get('blocked_batches')} of {batches} ({blocked_pct}%)",
        f"- Batches that ended with moved=0: {stats.get('moved_zero_batches')}",
        f"- Battles entered: {stats.get('battles')}",
        f"- Tiles: {stats.get('distinct_tiles')} distinct out of "
        f"{stats.get('positions_sampled')} positions sampled",
    ]
    top = stats.get("top_tiles") or []
    if top:
        rendered = ", ".join(
            f"({tile['x']},{tile['y']})x{tile['visits']}" for tile in top if isinstance(tile, dict)
        )
        lines.append(f"- Most revisited tiles: {rendered}")
    mix = stats.get("tool_mix") or {}
    if mix:
        rendered = ", ".join(f"{key}={value}" for key, value in mix.items())
        lines.append(f"- Tool mix: {rendered}")
    return lines


def read_notes(workspace_dir: Path, limit: int = NOTES_CHAR_LIMIT) -> str:
    path = Path(workspace_dir) / NOTES_FILENAME
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


@dataclass
class DigestInput:
    """Everything the digest can draw on. Every field is optional."""

    goal: str = ""
    objective: str = ""
    turns_completed: int = 0
    status: str = ""
    status_reason: str = ""
    session_tokens: Optional[int] = None
    start_state: Optional[JsonDict] = None
    final_state: Optional[JsonDict] = None
    map_summary: Optional[JsonDict] = None
    notes: str = ""
    calls: list[ToolCall] = field(default_factory=list)


def build_digest(data: DigestInput, *, char_budget: int = DIGEST_CHAR_BUDGET) -> str:
    """Render a bounded digest of the finished session.

    The raw session JSONL is around a megabyte; this is the ~10k-token version
    of it, and it is trimmed from the least useful end until it fits.
    """

    stats = compute_behaviour_stats(data.calls)
    header = [
        f"Goal: {data.goal or 'not set'}",
        f"Objective: {data.objective or 'unknown'}",
        f"Session ended: {data.status or 'unknown'} - {data.status_reason or 'no reason given'}",
        f"Turns completed: {data.turns_completed}",
    ]
    if data.session_tokens is not None:
        header.append(f"Context tokens used: {data.session_tokens}")

    narration = narration_lines(data.calls)
    recent = call_lines(data.calls)

    def render(narration_rows: list[str], recent_rows: list[str], notes: str) -> str:
        sections = [
            "# Finished session digest",
            _section("Session", header),
            _section("Game state at the start of the session", format_game_state(data.start_state)),
            _section("Game state now", format_game_state(data.final_state)),
            _section("What it did (measured, not reported)", format_stats(stats)),
            _section("Explored-map coverage", format_map_summary(data.map_summary)),
            _section("Narration the agent wrote, oldest first", narration_rows),
            _section(f"Last {len(recent_rows)} tool calls, oldest first", recent_rows),
            _section("NOTES.md as the agent left it", [notes] if notes else []),
        ]
        return "\n\n".join(section for section in sections if section).strip() + "\n"

    notes = data.notes
    digest = render(narration, recent, notes)
    # Trim from the cheapest end first: the head of the log, then narration, then notes.
    while len(digest) > char_budget and len(recent) > 10:
        recent = recent[len(recent) // 4 :]
        digest = render(narration, recent, notes)
    while len(digest) > char_budget and len(narration) > 5:
        narration = narration[len(narration) // 4 :]
        digest = render(narration, recent, notes)
    if len(digest) > char_budget and notes:
        notes = notes[: max(0, len(notes) - (len(digest) - char_budget))]
        digest = render(narration, recent, notes)
    if len(digest) > char_budget:
        digest = digest[:char_budget].rstrip() + "\n"
    return digest


CRITIC_INSTRUCTIONS = f"""\
You are reviewing a finished session of an agent playing Pokemon Red through an HTTP harness.
The session is over. A fresh session with an empty context is about to start on the same save.

Write the retrospective first. The first characters of your reply are the first characters of
the retrospective itself: do not think out loud in the reply, do not restate the digest, do not
describe your approach. Anything you want to add comes after the retrospective, never before it.

Below is a digest of what the finished session did: its goal, the game state before and after,
measured statistics from its own tool calls, the map it explored, the narration it wrote, its
last tool calls, and the notes file it maintains.

The retrospective is the next agent's first instruction. Cover, in this order and with no
preamble:

1. What the last session actually achieved, in one or two lines.
2. The specific mistakes it made, each cited with evidence from the numbers above.
3. Concrete, checkable things to do differently next session.
4. Anything learned about this map worth carrying forward: layout, exits, where encounters are.

Hard constraints:
- Aim for about {TARGET_HANDOFF_WORDS} words; {MAX_HANDOFF_WORDS} is the hard ceiling. Finish
  every sentence you start - a point cut off half-way is worth less than one you left out.
- Everything must be specific to THIS session and cite the digest. No generic Pokemon advice.
- Do not restate the rules of the harness. The next agent already has them in its system prompt.
- The reader is a fast model with no reasoning that acts on your words immediately, so a vague
  instruction is worse than no instruction. Name coordinates, directions, maps and counts.
- Output the retrospective itself as plain markdown. No preamble, no sign-off, no code fences.

Finish with the goal for the next session, as the very last line, alone, exactly this shape:

{NEXT_GOAL_LABEL}: <one short imperative sentence naming the single thing to do first>

Judge that goal against the state AFTER this session, not the goal it was given. A goal the
session already achieved is finished: name what comes next instead of repeating it. Keep it
under {MAX_NEXT_GOAL_CHARS} characters, on one line, with nothing after it.
"""

#: Appended on the retry, where the first pass reasoned past its output ceiling.
IMMEDIATE_INSTRUCTIONS = f"""\
This is the second attempt. The first one spent its whole output budget reasoning and never
wrote an answer. Do not deliberate: start typing the retrospective now, keep it to a handful of
lines, and stop. Half a retrospective delivered beats a whole one that never arrives. Write the
{NEXT_GOAL_LABEL} line even if you write nothing else.
"""


def build_prompt(digest: str, *, immediate: bool = False) -> str:
    instructions = CRITIC_INSTRUCTIONS
    if immediate:
        instructions = f"{instructions}\n{IMMEDIATE_INSTRUCTIONS}"
    return f"{instructions}\n---\n\n{digest}"


# ---------------------------------------------------------------------------
# Running pi
# ---------------------------------------------------------------------------


def critic_image_paths(workspace_dir: Path) -> list[Path]:
    """The map picture and the latest annotated frame, when they are on disk."""

    paths: list[Path] = []
    for filename in CRITIC_IMAGE_FILES:
        candidate = Path(workspace_dir) / filename
        if candidate.is_file():
            paths.append(candidate)
    return paths


def build_critic_command(
    pi_binary: str,
    *,
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    thinking: str = DEFAULT_CRITIC_THINKING,
    images: Optional[list[Path]] = None,
) -> list[str]:
    """The one-shot print-mode invocation. No session, and ``read`` as the only tool."""

    command = [pi_binary, "--mode", "json", "--print", "--thinking", thinking]
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    command.extend(
        [
            "-ne",
            "-ns",
            "-nc",
            "-np",
            "--no-themes",
            "--offline",
            "--no-session",
            "--tools",
            ",".join(CRITIC_TOOLS),
        ]
    )
    command.extend(f"@{path}" for path in images or [])
    command.append(prompt)
    return command


def parse_final_text(stdout: str) -> str:
    """The last assistant text in a ``--mode json`` event stream."""

    texts: list[str] = []
    for record in iter_jsonl_records(stdout):
        if record.get("type") not in {"message_end", "turn_end"}:
            continue
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = extract_message_text(message)
        if text:
            texts.append(text)
    return texts[-1].strip() if texts else ""


def usage_output_tokens(usage: Any) -> Optional[int]:
    """Output tokens out of an assistant ``usage`` block, whatever it calls the field."""

    if not isinstance(usage, dict):
        return None
    for key in ("output", "outputTokens", "output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def describe_no_text(stop_reason: Optional[str], usage: Any) -> str:
    """``Critic produced no text (stopReason=length, output=16384).``

    Six minutes of decode with nothing to show for it is only diagnosable if the
    reason it stopped travels with the failure.
    """

    detail = []
    if stop_reason:
        detail.append(f"stopReason={stop_reason}")
    tokens = usage_output_tokens(usage)
    if tokens is not None:
        detail.append(f"output={tokens}")
    if not detail:
        return NO_TEXT_ERROR
    return f"Critic produced no text ({', '.join(detail)})."


def critic_debug_dir(workspace_dir: Path) -> Path:
    return Path(workspace_dir) / DEBUG_DIRNAME


def write_raw_output(
    workspace_dir: Path,
    stdout_lines: list[str],
    stderr_lines: Optional[list[str]] = None,
) -> Optional[str]:
    """Keep the critic's raw event stream, rotating the previous run out of the way.

    This is the artefact a post-mortem starts from, so it is written for every
    attempt, including the ones that timed out or said nothing.
    """

    body = "".join(f"{line}\n" for line in stdout_lines)
    if stderr_lines:
        body += json.dumps({"type": "stderr", "lines": stderr_lines}, ensure_ascii=False) + "\n"
    directory = critic_debug_dir(workspace_dir)
    current = directory / CRITIC_RAW_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if current.is_file():
            os.replace(current, directory / CRITIC_RAW_PREVIOUS_FILENAME)
        current.write_text(body, encoding="utf-8")
    except OSError:
        return None
    return str(current)


_WORD_RE = re.compile(r"\S+")
#: A sentence ends at .!? plus any closing quote or bracket, followed by whitespace
#: or the end of the text. "3." in a numbered list is followed by a space too, so a
#: single leading digit before the stop does not count.
_SENTENCE_END_RE = re.compile(r"(?<!\b\d)[.!?][\"'’”)\]]*(?=\s|$)")


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in _WORD_RE.finditer(text)]


def _last_boundary(head: str) -> tuple[int, str]:
    """Offset just past the last paragraph break or sentence end in ``head``.

    Returns ``(0, "")`` when ``head`` holds neither, which is the run-on case: the
    caller falls back to a word cut and says so in its marker.
    """

    paragraph = head.rfind("\n\n")
    sentences = [match.end() for match in _SENTENCE_END_RE.finditer(head)]
    sentence = sentences[-1] if sentences else -1
    if paragraph >= 0 and paragraph >= sentence:
        return paragraph, "paragraph break"
    if sentence >= 0:
        return sentence, "sentence end"
    return 0, ""


def cap_words(text: str, limit: int = MAX_HANDOFF_WORDS) -> str:
    """``text`` shortened to ``limit`` words, never mid-sentence.

    Over the cap, the text is cut back to the last paragraph break or sentence end
    that fits, and an explicit marker says how many words were dropped. A bare "..."
    used to hide this, and the operator read a retrospective that stopped mid-clause.
    """

    spans = _word_spans(text)
    if len(spans) <= limit:
        return text.strip()
    head = text[: spans[limit - 1][1]]
    cut, boundary = _last_boundary(head)
    if cut <= 0:
        # One unbroken run of words: cut at the limit and admit it in the marker.
        cut, boundary = len(head), "word (no sentence break found)"
    kept = text[:cut].strip()
    dropped = len(_word_spans(text)) - len(_word_spans(kept))
    if dropped <= 0:
        return kept
    return kept + TRUNCATION_MARKER.format(boundary=boundary, dropped=dropped)


def tail_words(text: str, limit: int = MAX_HANDOFF_WORDS) -> str:
    """The last ``limit`` words. Reasoning is truncated at the end, so its tail is the news.

    The tail is nudged forward to the next sentence or paragraph start so it never
    opens mid-clause, and the marker names how many words were dropped ahead of it.
    """

    spans = _word_spans(text)
    if len(spans) <= limit:
        return text.strip()
    start = spans[-limit][0]
    rest = text[start:]
    paragraph = rest.find("\n\n")
    sentence = _SENTENCE_END_RE.search(rest)
    candidates: list[tuple[int, str]] = []
    if paragraph >= 0:
        candidates.append((paragraph + 2, "at the next paragraph"))
    if sentence is not None:
        candidates.append((sentence.end(), "at the next sentence"))
    offset, boundary = min(candidates, default=(0, "mid-sentence"))
    kept = rest[offset:].strip() or rest.strip()
    dropped = len(spans) - len(_word_spans(kept))
    marker = TAIL_TRUNCATION_MARKER.format(boundary=boundary, dropped=max(0, dropped))
    return f"{marker}\n\n{kept}"


#: ``NEXT GOAL: ...`` however the model decorated it - a bullet, a heading, bold
#: markers, an em dash for the colon. The label must still be the first words on the
#: line, so prose that merely mentions a next goal is not mistaken for the marker.
_NEXT_GOAL_RE = re.compile(
    r"^[^0-9A-Za-z]{0,8}" + NEXT_GOAL_LABEL.replace(" ", r"\s+") + r"\b[^0-9A-Za-z]{0,8}(.*)$",
    re.IGNORECASE,
)
_GOAL_EDGE_CHARS = "*_`#>-–— \t\"'“”‘’:."


def _clean_goal_line(raw: str) -> str:
    """One tidy imperative line, or "" when what the critic wrote is not usable."""

    text = " ".join(str(raw or "").split()).strip(_GOAL_EDGE_CHARS).strip()
    text = " ".join(text.split())
    if len(text) > MAX_NEXT_GOAL_CHARS:
        # Overlong is usually the critic pasting a plan onto the line. Its first
        # sentence is normally the actual goal; if even that runs long, drop it and
        # let the caller fall back rather than pin a paragraph as the goal.
        sentence = _SENTENCE_END_RE.search(text)
        text = text[: sentence.end()].strip() if sentence else ""
    if len(text) < MIN_NEXT_GOAL_CHARS or len(text) > MAX_NEXT_GOAL_CHARS:
        return ""
    if not any(character.isalpha() for character in text):
        return ""
    return text


def parse_next_goal(text: str) -> str:
    """The critic's goal for the session after next, or "" when it wrote none.

    Deliberately forgiving about decoration and deliberately strict about shape: a
    goal the caller cannot trust is worse than no goal, because the caller has a
    working fallback and no way to tell a mangled line from a real one.
    """

    lines = (text or "").splitlines()
    found = ""
    for index, line in enumerate(lines):
        match = _NEXT_GOAL_RE.match(line)
        if match is None:
            continue
        candidate = _clean_goal_line(match.group(1))
        if not candidate:
            # The label sat alone on its line; the goal is the next thing written.
            for follower in lines[index + 1 :]:
                candidate = _clean_goal_line(follower)
                if candidate:
                    break
        if candidate:
            found = candidate  # A restated goal later in the reply wins.
    return found


def handoff_path(workspace_dir: Path) -> Path:
    return Path(workspace_dir) / HANDOFF_FILENAME


def read_handoff(workspace_dir: Path) -> str:
    try:
        return handoff_path(workspace_dir).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def write_handoff(workspace_dir: Path, text: str) -> Path:
    """Replace ``HANDOFF.md``, keeping the outgoing one as ``HANDOFF.prev.md``.

    The new file is staged first, so a failed write never destroys both copies.
    """

    directory = Path(workspace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / HANDOFF_FILENAME
    staged = directory / f".{HANDOFF_FILENAME}.new"
    staged.write_text(text.strip() + "\n", encoding="utf-8")
    if current.is_file():
        os.replace(current, directory / HANDOFF_PREVIOUS_FILENAME)
    os.replace(staged, current)
    return current


@dataclass
class CriticAttempt:
    """One pi invocation: what it said, why it stopped, and where its raw log went."""

    thinking: str = DEFAULT_CRITIC_THINKING
    text: str = ""
    reasoning: str = ""
    stop_reason: Optional[str] = None
    usage: Optional[JsonDict] = None
    error: Optional[str] = None
    returncode: Optional[int] = None
    raw_path: Optional[str] = None
    duration_seconds: float = 0.0
    command: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.text) and self.error is None

    @property
    def diagnosis(self) -> str:
        return self.error or describe_no_text(self.stop_reason, self.usage)

    def summary(self) -> JsonDict:
        return {
            "thinking": self.thinking,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "error": self.error,
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "text_chars": len(self.text),
            "reasoning_chars": len(self.reasoning),
            "raw_path": self.raw_path,
        }


class CriticStream:
    """Folds the critic's JSONL stdout into an attempt, record by record as it lands.

    Buffering the whole run and parsing it at exit is what made a six-minute
    critique invisible; every record is handed to ``sink`` the moment it arrives.
    """

    def __init__(self, attempt: CriticAttempt, sink: Optional[CriticEventSink] = None) -> None:
        self.attempt = attempt
        self._sink = sink
        self._delta_text = ""
        self._delta_reasoning = ""
        self._final_text = ""
        self._final_reasoning = ""
        self._ended: set[tuple[str, str]] = set()

    async def feed(self, line: str) -> None:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(record, dict):
            await self.handle(record)

    async def handle(self, record: JsonDict) -> None:
        kind = record.get("type")
        if kind == "message_update":
            await self._handle_delta(record.get("assistantMessageEvent"))
            return
        if kind not in {"message_end", "turn_end"}:
            return
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return
        stop_reason = message.get("stopReason") or record.get("stopReason")
        if isinstance(stop_reason, str) and stop_reason:
            self.attempt.stop_reason = stop_reason
        usage = message.get("usage")
        if isinstance(usage, dict) and usage:
            self.attempt.usage = usage
        reasoning = extract_message_thinking(message)
        if reasoning:
            self._final_reasoning = reasoning
            await self._end("thinking", reasoning)
        text = extract_message_text(message)
        if text:
            self._final_text = text
            await self._end("text", text)
        self.close()

    async def _handle_delta(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        if event.get("type") == "thinking_delta":
            self._delta_reasoning += delta
            await self._emit({"type": "thinking_delta", "delta": delta})
        elif event.get("type") == "text_delta":
            self._delta_text += delta
            await self._emit({"type": "text_delta", "delta": delta})

    async def _end(self, kind: str, text: str) -> None:
        # message_end and turn_end repeat the same message; announce it once.
        key = (kind, text)
        if key in self._ended:
            return
        self._ended.add(key)
        await self._emit({"type": f"{kind}_end", "text": text})

    async def _emit(self, event: JsonDict) -> None:
        if self._sink is None:
            return
        with contextlib.suppress(Exception):
            await self._sink(event)

    def close(self) -> None:
        """Resolve the attempt: a finished message wins, the deltas are the fallback."""

        self.attempt.text = (self._final_text or self._delta_text).strip()
        self.attempt.reasoning = (self._final_reasoning or self._delta_reasoning).strip()


def _kill(process: Any) -> None:
    if getattr(process, "returncode", 0) is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()


async def run_attempt(
    *,
    command: list[str],
    thinking: str,
    workspace: Path,
    timeout_seconds: float,
    event_sink: Optional[CriticEventSink] = None,
    process_sink: Optional[Callable[[Any], None]] = None,
) -> CriticAttempt:
    """One pi run, streamed. Reports every failure as an attempt, never as an exception."""

    attempt = CriticAttempt(thinking=thinking, command=list(command))
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(workspace),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        attempt.error = f"Could not launch the critic: {exc}"
        attempt.duration_seconds = round(time.monotonic() - started, 2)
        return attempt

    if process_sink is not None:
        with contextlib.suppress(Exception):
            process_sink(process)

    stream = CriticStream(attempt, event_sink)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    async def pump_stdout() -> None:
        async for line in iter_stream_lines(process.stdout):
            stdout_lines.append(line)
            await stream.feed(line)

    async def pump_stderr() -> None:
        async for line in iter_stream_lines(process.stderr):
            stderr_lines.append(line)

    tasks = [asyncio.create_task(pump_stdout()), asyncio.create_task(pump_stderr())]
    try:
        _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        if pending:
            attempt.error = f"Critic timed out after {timeout_seconds:.0f}s."
        else:
            for task in tasks:
                failure = task.exception()
                if failure is not None:
                    raise failure
            remaining = max(1.0, timeout_seconds - (time.monotonic() - started))
            try:
                attempt.returncode = await asyncio.wait_for(process.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                attempt.error = f"Critic timed out after {timeout_seconds:.0f}s."
    except asyncio.CancelledError:
        _kill(process)
        raise
    except Exception as exc:  # noqa: BLE001 - a critique must never break the run loop
        attempt.error = f"Critic failed: {exc}"
    finally:
        _kill(process)
        for task in tasks:
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await process.wait()

    stream.close()
    attempt.duration_seconds = round(time.monotonic() - started, 2)
    attempt.raw_path = write_raw_output(workspace, stdout_lines, stderr_lines)
    if not attempt.text:
        # Belt and braces: re-read the whole stream for a shape the live parse missed.
        attempt.text = parse_final_text("\n".join(stdout_lines))
    if attempt.error is None and attempt.returncode not in (0, None):
        detail = " ".join(stderr_lines[-3:])[:400]
        attempt.error = f"Critic exited with status {attempt.returncode}. {detail}".strip()
    return attempt


#: Words to leave for whichever truncation marker gets appended.
_MARKER_WORD_ALLOWANCE = 16


def _salvaged(notice: str, body: str, *, tail: bool) -> str:
    """Notice plus body, with the pair - not just the body - under the word cap."""

    # The notice and the truncation marker both come out of the same word cap.
    budget = max(1, MAX_HANDOFF_WORDS - len(notice.split()) - _MARKER_WORD_ALLOWANCE)
    trimmed = tail_words(body, budget) if tail else cap_words(body, budget)
    return f"{notice}\n\n{trimmed}"


def salvage(attempts: list[CriticAttempt]) -> str:
    """The best partial thing any attempt left behind, marked for what it is.

    A cut-off answer beats a reasoning tail; a reasoning tail beats throwing six
    minutes of decode away and handing the next session nothing.
    """

    partial = next((attempt.text for attempt in reversed(attempts) if attempt.text), "")
    if partial:
        return _salvaged(SALVAGED_ANSWER_NOTICE, partial, tail=False)
    reasoning = max((attempt.reasoning for attempt in attempts), key=len, default="")
    if not reasoning:
        return ""
    return _salvaged(SALVAGED_REASONING_NOTICE, reasoning, tail=True)


@dataclass
class CriticResult:
    """What one critic pass produced. ``ok`` is false for every failure mode.

    ``ok`` with ``salvaged`` set means a handoff was written from a truncated
    reply or from the tail of the reasoning; ``error`` still says what went wrong.
    """

    ok: bool
    text: str = ""
    #: The critic's ``NEXT GOAL:`` line, or "" when it wrote none the parser trusts.
    #: The supervisor uses it only when the operator has not named a goal of their own.
    next_goal: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    digest: str = ""
    digest_tokens: int = 0
    command: list[str] = field(default_factory=list)
    handoff_path: Optional[str] = None
    salvaged: bool = False
    stop_reason: Optional[str] = None
    usage: Optional[JsonDict] = None
    raw_path: Optional[str] = None
    attempts: list[JsonDict] = field(default_factory=list)


async def run_critic(
    *,
    pi_binary: Optional[str],
    workspace_dir: Path,
    digest: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    thinking: str = DEFAULT_CRITIC_THINKING,
    timeout_seconds: float = DEFAULT_CRITIC_TIMEOUT_SECONDS,
    include_images: bool = True,
    retry_enabled: bool = True,
    retry_thinking: str = DEFAULT_CRITIC_RETRY_THINKING,
    retry_min_seconds: float = CRITIC_RETRY_MIN_SECONDS,
    event_sink: Optional[CriticEventSink] = None,
    process_sink: Optional[Callable[[Any], None]] = None,
) -> CriticResult:
    """Run the critic and write ``HANDOFF.md``. Never raises.

    ``event_sink`` is awaited with the critic's own events as they arrive
    (``attempt_start``, ``thinking_delta``, ``text_delta``, ``thinking_end``,
    ``text_end``), so the supervisor can render the critique while it runs.
    ``process_sink`` is handed the subprocess as soon as it exists, so the
    supervisor can kill an in-flight critic when the operator stops the run.

    An attempt that produces no usable text buys one cheaper retry, as long as
    the time budget still allows it.
    """

    workspace = Path(workspace_dir)
    tokens = estimate_tokens(build_prompt(digest))
    if not pi_binary:
        return CriticResult(
            ok=False,
            error="Pi executable was not found.",
            digest=digest,
            digest_tokens=tokens,
        )

    images = critic_image_paths(workspace) if include_images else []
    started = time.monotonic()
    attempts: list[CriticAttempt] = []

    async def attempt_once(level: str, *, immediate: bool, budget: float) -> CriticAttempt:
        command = build_critic_command(
            pi_binary,
            prompt=build_prompt(digest, immediate=immediate),
            provider=provider,
            model=model,
            thinking=level,
            images=images,
        )
        if event_sink is not None:
            with contextlib.suppress(Exception):
                await event_sink(
                    {"type": "attempt_start", "attempt": len(attempts) + 1, "thinking": level}
                )
        attempt = await run_attempt(
            command=command,
            thinking=level,
            workspace=workspace,
            timeout_seconds=budget,
            event_sink=event_sink,
            process_sink=process_sink,
        )
        attempts.append(attempt)
        return attempt

    attempt = await attempt_once(thinking, immediate=False, budget=timeout_seconds)
    if not attempt.usable and retry_enabled:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining >= retry_min_seconds:
            await attempt_once(retry_thinking, immediate=True, budget=remaining)

    def latest(attribute: str) -> Any:
        return next(
            (value for value in (getattr(one, attribute) for one in reversed(attempts)) if value),
            None,
        )

    def finish(**fields: Any) -> CriticResult:
        return CriticResult(
            duration_seconds=round(time.monotonic() - started, 2),
            digest=digest,
            digest_tokens=tokens,
            command=attempts[-1].command if attempts else [],
            stop_reason=latest("stop_reason"),
            usage=latest("usage"),
            raw_path=latest("raw_path"),
            attempts=[one.summary() for one in attempts],
            **fields,
        )

    error = attempts[0].diagnosis
    if len(attempts) > 1:
        error = f"{error} Retry at {attempts[-1].thinking}: {attempts[-1].diagnosis}"

    usable = next((one for one in reversed(attempts) if one.usable), None)
    if usable is not None:
        text, salvaged, error = cap_words(usable.text), False, None
    else:
        text, salvaged = salvage(attempts), True
        if not text:
            return finish(ok=False, error=error)
        error = f"{error} Salvaged a partial handoff instead."

    result = finish(
        ok=True,
        text=text,
        next_goal=parse_next_goal(text),
        error=error,
        salvaged=salvaged,
    )
    try:
        result.handoff_path = str(write_handoff(workspace, text))
    except OSError as exc:
        return finish(ok=False, text=text, error=f"Could not write {HANDOFF_FILENAME}: {exc}")
    return result
