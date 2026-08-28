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
from typing import Any, Awaitable, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from pokemon_agent import notes as notes_module
from pokemon_agent.agent_cli import ActionError, expand_actions
from pokemon_agent.bench.metrics import compute as compute_run_metrics
from pokemon_agent.bench.metrics import whiteout_events
from pokemon_agent.bench.registry import RunRegistry
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
#: What the first pass may spend, so a retry can still be afforded after it.
#: It used to be handed the whole budget, which made the retry unreachable in
#: the one case it exists for: a first pass that times out has by definition
#: spent everything, leaving a remainder of nothing to retry out of. The retry
#: only ever ran when the first pass failed *fast*. A fallback that cannot fire
#: is the shape of bug this project keeps finding, so the cap is explicit.
CRITIC_FIRST_ATTEMPT_SECONDS = 430.0
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

#: Hard ceiling on the critic's prose, in words.
#:
#: This was 900 - about 1,200 tokens - back when the retrospective was the whole
#: handoff and its opening paragraph had to carry what the session achieved. It no
#: longer does: :class:`SessionFacts` states that from the receipts, in a form the
#: critic cannot get wrong, and rides above the prose in the same message. What is
#: left for the critic is the part a counter cannot produce - which mistake cost
#: the most and what to do instead - and that is a few paragraphs, not an essay.
#:
#: The first principle is minimum context: the agent builds its own, and every word
#: here is a word it did not spend looking at the game. 260 words is roughly 350
#: tokens, and the whole handoff lands around 500.
MAX_HANDOFF_WORDS = 260
#: What the critic is *asked* for. The ceiling above is a backstop, not the target,
#: so a critic that lands slightly long still gets to finish its own last sentence.
TARGET_HANDOFF_WORDS = 160

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
#:
#: The player's budget and the critic's are opposites. The player re-reads its
#: context on all five hundred turns of a session, so every token it is given it
#: pays for five hundred times. The critic reads once, at high effort, on a
#: context that is thrown away when it finishes, so a digest three times larger
#: costs one prompt. A five-hundred-turn session is a megabyte of JSONL; at
#: twelve thousand tokens the critic was shown its last sixty tool calls out of
#: eight hundred and had to infer the rest from narration. It no longer has to.
DIGEST_TOKEN_BUDGET = 40_000
CHARS_PER_TOKEN = 4
DIGEST_CHAR_BUDGET = DIGEST_TOKEN_BUDGET * CHARS_PER_TOKEN

#: Tool calls and narration lines shown. Both are sampled across the whole
#: session rather than taken off the end, so widening them widens the arc the
#: critic can see and not just its tail.
RECENT_TOOL_CALLS = 500
MAX_NARRATION_LINES = 140
NOTES_CHAR_LIMIT = 6_000
TOP_TILES = 6
CALL_LINE_LIMIT = 220
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
    """The half of ``NOTES.md`` the model wrote, which is the half that is a claim.

    The harness owns a delimited block at the top of that file and rewrites it
    from the game — see :mod:`pokemon_agent.notes`. It is dropped here, because
    every heading this text appears under says "CLAIMS, not facts, and
    unverified", and filing a measurement under that heading is how a
    measurement gets argued with.
    """

    path = Path(workspace_dir) / NOTES_FILENAME
    try:
        text = notes_module.strip_state_block(path.read_text(encoding="utf-8")).strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


# ---------------------------------------------------------------------------
# Static world intelligence
#
# The player runs on a minimum-context diet because it pays for every token again
# on each of five hundred turns. The critic runs once, on a context it throws
# away, so the asymmetry is the point: it can be told far more than the agent it
# is reviewing, as long as what it hands back stays short.
#
# This block is the part the transcripts said was missing. Nine sessions of
# reasoning contain no case of the harness telling the agent something false
# about the game, and a dozen cases of the agent inventing a compass direction
# and then acting on it against a correct tool result. A critic with no geography
# of its own can only repeat whichever of those it read. So it gets the real one:
# ``pokemon_agent/data/game/world.json`` holds all 223 maps with their
# dimensions, their edge connections and every warp's destination, and none of it
# is a recollection.
#
# Everything here answers with nothing rather than raising. A critic that cannot
# start is worse than a critic with less to say.
# ---------------------------------------------------------------------------

#: Warps named for one map before the rest collapse into a count.
MAX_BRIEF_WARPS = 12
#: Trainers and items named. Both are reasons to walk somewhere on purpose.
MAX_BRIEF_TRAINERS = 8
MAX_BRIEF_ITEMS = 8
#: Encounter slots named, commonest first, so a level check has something to read.
MAX_BRIEF_ENCOUNTERS = 5
#: Hops printed for one route. Longer than this and the next session will not
#: finish it anyway, so the tail is a count.
MAX_ROUTE_HOPS = 8
#: What a healing map is called in the world data.
POKECENTER_MARK = "Pokecenter"

#: Process-lifetime memo for everything read off disk or introspected once:
#: the map graph, the game data, the milestone ladder, the CLI's verb list and
#: ``scope``'s bucket classifier. All of it is static for the run's lifetime,
#: and the critic is on the path between one session ending and the next
#: starting, so none of it is worth reading twice.
_LOOKUP_CACHE: dict[str, Any] = {}


def _game_data() -> Any:
    """``pokemon_agent.gamedata``, or ``None`` when the generated files are absent."""

    if "gamedata" not in _LOOKUP_CACHE:
        try:
            from pokemon_agent import gamedata

            gamedata.world()
        except Exception:  # noqa: BLE001 — ungenerated data is missing data, not an error
            _LOOKUP_CACHE["gamedata"] = None
        else:
            _LOOKUP_CACHE["gamedata"] = gamedata
    return _LOOKUP_CACHE["gamedata"]


def world_graph() -> Any:
    """The static map graph, loaded once. ``None`` when it cannot be read."""

    if "world" not in _LOOKUP_CACHE:
        try:
            from pokemon_agent.world import World

            world = World.load()
        except Exception:  # noqa: BLE001
            world = None
        _LOOKUP_CACHE["world"] = world if world is not None and len(world) else None
    return _LOOKUP_CACHE["world"]


def known_map_names() -> tuple[str, ...]:
    """Every map name the game data knows, longest first for greedy matching."""

    if "names" not in _LOOKUP_CACHE:
        data = _game_data()
        names = tuple(data.map_names()) if data is not None else ()
        _LOOKUP_CACHE["names"] = tuple(sorted(names, key=len, reverse=True))
    return _LOOKUP_CACHE["names"]


def mentioned_map(text: str, exclude: Iterable[str] = ()) -> str:
    """The map a goal is aiming at, or "".

    The *last* map named wins, because a goal is written as a journey and the
    destination is the end of it: "walk from Mt Moon B1F out to Route 4 and on to
    Cerulean City" is a goal about Cerulean. Ties go to the longer name, so
    "Cerulean Pokecenter" is never read as "Cerulean City", and every match is
    bounded, so "Route 4" is never found inside "Route 44".

    ``exclude`` drops maps that are not destinations - the one already stood on,
    most of all, because routing a map to itself is a line with nothing in it.
    """

    haystack = (text or "").lower()
    if not haystack:
        return ""
    skip = {one.lower() for one in exclude}
    best, best_key = "", (-1, -1)
    for name in known_map_names():
        needle = name.lower()
        if needle in skip:
            continue
        start = haystack.find(needle)
        while start != -1:
            before = haystack[start - 1] if start else " "
            after = haystack[start + len(needle) : start + len(needle) + 1] or " "
            if not before.isalnum() and not after.isalnum() and (start, len(needle)) > best_key:
                best, best_key = name, (start, len(needle))
            start = haystack.find(needle, start + 1)
    return best


def _hop_text(hop: Any) -> str:
    """One map-to-map transition as an instruction rather than a record."""

    if getattr(hop, "kind", "") == "connection":
        return f"walk {hop.edge} to {hop.to_map}"
    at = getattr(hop, "at", None)
    where = f" ({at[0]},{at[1]})" if at is not None else ""
    return f"warp{where} to {hop.to_map}"


def route_text(source: str, target: str) -> str:
    """``walk north to Route 4, then ...`` — the map graph's answer, not a memory.

    Empty when either end is unknown or nothing connects them, which is itself
    worth knowing: it means the next leg is not a walk.
    """

    world = world_graph()
    if world is None or not source or not target or source == target:
        return ""
    try:
        hops = world.route(source, target)
    except Exception:  # noqa: BLE001
        return ""
    if not hops:
        return ""
    named = [_hop_text(hop) for hop in hops[:MAX_ROUTE_HOPS]]
    rest = len(hops) - len(named)
    return ", then ".join(named) + (f", then {rest} more hops" if rest > 0 else "")


def nearest_pokecenter(source: str) -> tuple[str, int]:
    """The closest map you can heal on, and how many hops away it is.

    A thinking model with no ground truth invented a Poke Center in the wrong
    city and sent a party at 10 HP backwards to reach it. This is the same
    question, answered off the map graph instead.
    """

    world = world_graph()
    if world is None or not source:
        return "", 0
    best, best_distance = "", -1
    for name in world.map_names():
        if POKECENTER_MARK not in name:
            continue
        try:
            distance = world.distance(source, name)
        except Exception:  # noqa: BLE001
            distance = None
        if distance is None:
            continue
        if best_distance < 0 or distance < best_distance:
            best, best_distance = name, distance
    return (best, best_distance) if best else ("", 0)


@dataclass(frozen=True)
class MapBrief:
    """One map as the game defines it, not as the agent remembers it."""

    name: str = ""
    map_id: Optional[int] = None
    size: Optional[tuple[int, int]] = None
    #: ``("north", "Route 4")`` — walk off that edge and you are on that map.
    connections: tuple[tuple[str, str], ...] = ()
    #: ``(x, y, destination)``. A destination of "" is one the game picks at runtime.
    warps: tuple[tuple[int, int, str], ...] = ()
    trainers: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    encounter_rate: int = 0
    encounters: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return bool(self.name)

    def exits(self, stood_on: Iterable[tuple[int, int]] = ()) -> str:
        """Every way off this map in one line, starring the ones never used.

        The star is the whole value of the line. Three sessions and 720 presses
        went into a pocket whose way out was a warp tile the agent never stepped
        on, and no report it could read said which tiles those were. A star is a
        character; spelling it out costs three tokens on a line that repeats it
        eight times, and this line ships in the next session's first message.
        """

        visited = set(stood_on)
        parts = [f"walk {edge} -> {target}" for edge, target in self.connections]
        for x, y, target in self.warps[:MAX_BRIEF_WARPS]:
            mark = "" if (x, y) in visited else "*"
            parts.append(f"({x},{y}){mark} -> {target or 'runtime-chosen'}")
        rest = len(self.warps) - min(len(self.warps), MAX_BRIEF_WARPS)
        if rest > 0:
            parts.append(f"+{rest} more warps")
        return "; ".join(parts) if parts else "none in the map data"

    def lines(self, stood_on: Iterable[tuple[int, int]] = ()) -> list[str]:
        """The whole brief, for the digest rather than the handoff."""

        if not self.known:
            return []
        shape = f"{self.size[0]}x{self.size[1]} tiles" if self.size else "size unknown"
        rows = [f"- {self.name} (id={self.map_id}) {shape}.", f"- Exits: {self.exits(stood_on)}"]
        if self.trainers:
            rows.append(f"- Trainers standing here: {'; '.join(self.trainers)}.")
        if self.items:
            rows.append(f"- Items on the ground: {'; '.join(self.items)}.")
        if self.encounters:
            rows.append(
                f"- Wild encounters ({self.encounter_rate}/256 per step): "
                + ", ".join(self.encounters)
                + "."
            )
        return rows


def _trainer_text(entry: Mapping[str, Any]) -> str:
    team = entry.get("team")
    levels = ""
    if isinstance(team, list) and team:
        levels = " " + "/".join(
            f"{one.get('species')} L{one.get('level')}" for one in team[:3] if isinstance(one, dict)
        )
    return f"{entry.get('trainer_class', '?')} at ({entry.get('x')},{entry.get('y')}){levels}"


def _item_text(entry: Mapping[str, Any]) -> str:
    kind = " (hidden)" if entry.get("hidden") else ""
    return f"{entry.get('item', '?')}{kind} at ({entry.get('x')},{entry.get('y')})"


def _brief_size(size: Any) -> Optional[tuple[int, int]]:
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None
    return None


def map_brief(map_name: str) -> MapBrief:
    """Everything the generated game data holds about one map. Never raises."""

    data = _game_data()
    if data is None or not map_name:
        return MapBrief()
    try:
        entry = data.world().get(map_name)
    except Exception:  # noqa: BLE001
        entry = None
    if not isinstance(entry, Mapping):
        return MapBrief()

    def safely(call: Callable[[], Any], default: Any) -> Any:
        try:
            return call() or default
        except Exception:  # noqa: BLE001
            return default

    warps: list[tuple[int, int, str]] = []
    for warp in entry.get("warps") or []:
        if not isinstance(warp, Mapping):
            continue
        try:
            warps.append((int(warp["x"]), int(warp["y"]), str(warp.get("to_map") or "")))
        except (KeyError, TypeError, ValueError):
            continue

    trainers = safely(lambda: data.trainers(map_name), [])
    items = safely(lambda: data.items(map_name), [])
    encounters = safely(lambda: data.encounters(map_name), {})
    grass = encounters.get("grass") if isinstance(encounters, Mapping) else None
    slots = grass.get("slots") if isinstance(grass, Mapping) else None

    return MapBrief(
        name=map_name,
        map_id=entry.get("map_id"),
        size=_brief_size(entry.get("size")),
        connections=tuple(
            (str(edge), str(target))
            for edge, target in (entry.get("connections") or {}).items()
            if target
        ),
        warps=tuple(warps),
        trainers=tuple(
            _trainer_text(one) for one in trainers[:MAX_BRIEF_TRAINERS] if isinstance(one, Mapping)
        ),
        items=tuple(_item_text(one) for one in items[:MAX_BRIEF_ITEMS] if isinstance(one, Mapping)),
        encounter_rate=int(grass.get("rate") or 0) if isinstance(grass, Mapping) else 0,
        encounters=tuple(
            f"{one.get('species')} L{one.get('level')}"
            f" {round(100 * float(one.get('chance') or 0))}%"
            for one in (slots or [])[:MAX_BRIEF_ENCOUNTERS]
            if isinstance(one, Mapping)
        ),
    )


def ladder_labels() -> tuple[dict[str, str], tuple[str, ...]]:
    """``{milestone id: label}`` and the ladder in order. Empty when unscored."""

    if "ladder" not in _LOOKUP_CACHE:
        try:
            from pokemon_agent.bench.metrics import load_ladder

            entries = load_ladder()
        except Exception:  # noqa: BLE001
            entries = {}
        ranked = sorted(
            (entry for entry in entries.values() if entry.ladder_index is not None),
            key=lambda entry: entry.ladder_index or 0,
        )
        _LOOKUP_CACHE["ladder"] = (
            {entry.milestone_id: entry.label for entry in entries.values()},
            tuple(entry.milestone_id for entry in ranked),
        )
    return _LOOKUP_CACHE["ladder"]


def _live_or_reached(
    live_milestones: Optional[Iterable[str]],
    reached: Sequence[str],
) -> tuple[tuple[str, ...], bool]:
    """``(what the game holds now, whether RAM said so)``.

    Falls back to *reached* -- the receipts' running union -- and says so, so a
    caller with no RAM reading still gets the block it has always got rather
    than a blank one. The fallback is a high-water mark; see
    :func:`collect_session_facts`.

    A list with no recognisable milestone in it is a failed read, not a fresh
    game. The two are identical from here, and the difference matters: the
    frontier of nothing is "go and get a starter", which is a confident lie to
    print over a run that is nineteen hours past its starter. Same rule
    :func:`pokemon_agent.objectives.frontier_objective` applies to the same
    field, for the same reason.

    Ordered highest rung first, matching what ``reached`` already does, so the
    fallback render -- the first few ids, when the ladder has no label -- names
    the same end of the run either way.
    """

    if live_milestones is None:
        return tuple(reached), False
    labels, ordered = ladder_labels()
    try:
        held = {str(item) for item in live_milestones}
    except TypeError:  # a milestones field that is not iterable
        return tuple(reached), False
    if not any(milestone_id in labels for milestone_id in held):
        return tuple(reached), False
    ranks = {identifier: rank for rank, identifier in enumerate(ordered)}
    known = [identifier for identifier in held if identifier in labels]
    known.sort(key=lambda identifier: (-ranks.get(identifier, -1), identifier))
    return tuple(known), True


#: Enough to say which jobs are open without turning the digest into a plan.
MAX_FRONTIER_SHOWN = 6


def milestone_frontier(done: Iterable[str], limit: int = MAX_FRONTIER_SHOWN) -> tuple[str, ...]:
    """Labels of every milestone whose prerequisites the run has already met.

    :func:`ladder_position` answers with one rung because the ladder is a list,
    and a list has to pick. The DAG does not: it knows the fossil pair excludes
    each other, that the Route 22 rival is optional, and that three jobs can be
    open at once. Handing the critic the one next rung has been handing it a
    guess, and on this run the guess pointed four maps backwards.

    Empty when the DAG cannot be read, which drops the line and nothing else.
    """

    try:
        from pokemon_agent.milestones import frontier as dag_frontier
    except Exception:  # noqa: BLE001
        return ()
    try:
        return tuple(node.label or node.id for node in dag_frontier(list(done))[:limit])
    except Exception:  # noqa: BLE001
        return ()


def ladder_position(done: Iterable[str]) -> tuple[str, str]:
    """``(highest rung reached, next rung to reach)`` as labels, either may be "".

    The ladder is the run's actual objective and the receipts say exactly how far
    up it the run is. A retrospective that names the next rung cannot repeat the
    failure where one told a session to beat a gym leader it had already beaten.

    "Next" means the next rung *above* the highest one reached, not the lowest
    unreached one. Several rungs are optional and a run that walked past one has
    not left work behind it; sending it back for the rival battle on Route 22
    would be the same wrong instruction in the other direction.
    """

    labels, ordered = ladder_labels()
    reached = {identifier for identifier in done if identifier in labels}
    ranks = {identifier: rank for rank, identifier in enumerate(ordered)}
    highest = max((ranks[one] for one in reached if one in ranks), default=-1)
    top = next((one for one in ordered if ranks[one] == highest), "")
    upcoming = next(
        (one for one in ordered if ranks[one] > highest and one not in reached),
        "",
    )
    return labels.get(top, ""), labels.get(upcoming, "")


# ---------------------------------------------------------------------------
# Ground truth off the run receipts
#
# Everything above this line is the model's account of the session, read back out
# of its own transcript. Everything below is the harness's account, read out of
# ``<data_dir>/runs/<run_id>/receipts.jsonl``, which the server writes after each
# action batch and the agent never touches.
#
# The distinction is the point. Ask a model what it achieved and you get an answer
# shaped like an achievement: one real retrospective on disk told the next session
# to go and beat a gym leader the run had already beaten, because nothing in the
# digest said the badge was won. Counts are not opinions, so the critic is handed
# them and told not to argue with them, and the next session is handed them
# whether or not a critic ever ran.
# ---------------------------------------------------------------------------

#: Where the harness records which run a session belonged to and when it began, so
#: the session after it can still slice the receipts when the server was killed in
#: between and no in-memory state survived.
SESSION_MARK_FILENAME = "session_mark.json"

#: ``tool`` on the bookkeeping receipt a run opens with. It spent no buttons, so
#: it is not a thing the agent reached for and does not belong in the tool mix.
RUN_START_TOOL = "run_start"

SAVES_DIRNAME = "saves"
#: The harness writes one of these per batch. There are thousands and none of them
#: is a place the agent chose to come back to, so none is worth a token.
AUTO_SAVE_PREFIX = "auto__"
SAVE_SUFFIX = ".state"
#: Named saves offered to the next session, newest first. Four, not six: the
#: point of the line is that the escape hatch exists at all - `./poke saves` had
#: never been called once - and the two oldest names cost twelve tokens of a
#: message that now also has to carry the map's exits.
MAX_HANDOFF_SAVES = 4
#: Milestone ids named before the rest collapse into "+n more".
MAX_HANDOFF_MILESTONES = 4
#: Tools named in the mix line.
MAX_HANDOFF_TOOLS = 4
#: Below this a revisited tile is just walking, not a trap.
HOTSPOT_MIN_VISITS = 5
#: At or below this share of max HP the handoff spends a line on where to heal.
HURT_HP_FRACTION = 0.4
#: Receipts store ``t`` rounded to the millisecond, and the mark stores the raw
#: clock, so a receipt written microseconds after a session began can round to
#: just before it and fall out of its own session. Widen the slice by one tick.
RECEIPT_TIME_EPSILON = 0.001

FACTS_HEADING = "Ground truth from the run receipts"
FACTS_DIGEST_HEADING = f"{FACTS_HEADING} (authoritative - do not contradict these)"

#: The two blocks the model wrote itself. Both are headed as claims, because one
#: retrospective on disk copied "machine INACCESSIBLE (confirmed)" out of NOTES.md
#: and handed it on as fact; the agent healed at that machine 26 seconds later.
NARRATION_HEADING = "Narration the agent wrote, oldest first - CLAIMS, not facts, and unverified"
NOTES_HEADING = "NOTES.md as the agent left it - CLAIMS, not facts, and unverified"


def quoted_lines(text: str) -> list[str]:
    """*text* as markdown blockquote lines, so it cannot close its own section.

    Every section of the digest is a ``## `` heading and its body, and one body
    is a whole document the model wrote: ``NOTES.md``. Markdown does not nest,
    so the first ``## `` line inside those notes ends the section that called
    them claims, and everything after it reads as a section of the digest — a
    peer of "Ground truth from the run receipts (authoritative)". The seeded
    notes file opens with ``## Your notes``, so this is not a corner case; it is
    what the section looks like on every run that has notes at all.

    Blockquoting is the cheapest correct fix: not a byte of the model's text is
    lost or reordered, and no line of it starts a heading any more. A blank line
    keeps its paragraph break as a bare ``>``, because :func:`_section` drops
    empty strings.
    """

    return [f"> {line}" if line.strip() else ">" for line in (text or "").splitlines()]


def _plural(count: int, noun: str, plural: str = "") -> str:
    return f"{count:,} " + (noun if count == 1 else (plural or noun + "s"))


def _position(pos: Optional[tuple[int, int]]) -> str:
    return f"({pos[0]},{pos[1]})" if pos else "?"


@dataclass(frozen=True)
class SessionFacts:
    """What the receipts say the run has cost and the last session did with it.

    Every field is safe on an empty run: a run whose first session is still
    starting has one receipt in it, and the next session must still get a usable
    first message out of that.
    """

    run_id: str = ""
    session_index: int = 0
    total_presses: int = 0
    #: Milestone ids the game holds *now*, highest rung first. Read off RAM when
    #: the caller had a reading; only then does it fall. Without one it is the
    #: receipts' union of baseline and earned, which is a high-water mark -- see
    #: :data:`peak_count` and :func:`collect_session_facts`.
    done: tuple[str, ...] = ()
    done_count: int = 0
    #: The most milestones the run ever held at once, from the receipts. Equal to
    #: ``done_count`` on a run that never handed a rung back, and larger when a
    #: reload did. The gap is worth saying out loud: the run has already paid for
    #: those rungs and the next session has to earn them a second time.
    peak_count: int = 0
    #: Whether :data:`done` came from a RAM reading rather than from the receipts.
    live: bool = False
    #: Milestones the finished session earned. Usually empty, and that is the point.
    gained: tuple[str, ...] = ()

    session_presses: int = 0
    session_batches: int = 0
    blocked_batches: int = 0
    position_samples: int = 0
    unique_positions: int = 0
    ended_map: str = ""
    ended_pos: Optional[tuple[int, int]] = None
    ended_hp: Optional[tuple[int, int]] = None
    party_size: int = 0
    whiteouts: int = 0
    reloads: int = 0
    hot_map: str = ""
    hot_pos: Optional[tuple[int, int]] = None
    hot_visits: int = 0
    tool_mix: tuple[tuple[str, int], ...] = ()
    saves: tuple[str, ...] = ()

    # -- static ground truth, from the generated game data rather than the run --
    #: Label of the highest ladder rung the run has reached, and of the next one.
    rung_done: str = ""
    rung_next: str = ""
    #: Every milestone whose prerequisites are already met, from the DAG in
    #: :mod:`pokemon_agent.milestones`. The ladder is a line and the game is
    #: not: on the session this was added for, ``rung_next`` read "Beat the
    #: rival on Route 22" — four maps behind the run, optional, and skipped —
    #: while the DAG had the Super Nerd holding the fossils on the floor the
    #: run was actually standing on. One next rung is a guess about which of
    #: several open jobs matters; the frontier is the list.
    frontier: tuple[str, ...] = ()
    #: Every way off the map the session ended on, marking the ones never used.
    exits: str = ""
    #: Where the goal is, as hops on the map graph, and what the goal was read as.
    route_target: str = ""
    route: str = ""
    #: The way to the nearest healing map, carried only when the party is hurt.
    heal_target: str = ""
    heal_route: str = ""

    @property
    def presses_per_new_tile(self) -> Optional[float]:
        if not self.unique_positions or not self.session_presses:
            return None
        return round(self.session_presses / self.unique_positions, 1)

    def lines(self) -> list[str]:
        """The block as bullets. A row with nothing to say is not written."""

        rows = [
            f"- Run {self.run_id or 'unknown'}, after session {self.session_index or 1}. "
            f"{_plural(self.total_presses, 'press', 'presses')} spent on it so far."
        ]
        if self.done:
            # The ladder's own labels, not the raw event ids. Four ids cost thirty
            # tokens and told the next session nothing it could act on; the rung
            # above the highest one reached is the whole of what it needed.
            reached = (
                f"highest rung: {self.rung_done}"
                if self.rung_done
                else ", ".join(self.done[:MAX_HANDOFF_MILESTONES])
            )
            gained = (
                "Gained last session: " + ", ".join(self.gained)
                if self.gained
                else "Nothing new last session"
            )
            upcoming = f" Next rung: {self.rung_next}." if self.rung_next else ""
            # A reload that lands on an earlier branch hands rungs back. The run
            # still paid for them, so the peak is the honest bill -- but the next
            # session is playing the branch the game is actually on, and telling
            # it otherwise is how a model was once told to ride a bicycle the
            # cartridge did not have.
            lost = (
                f" The run held {self.peak_count} at its peak and gave "
                f"{self.peak_count - self.done_count} back to a reload; those are "
                "not in the game now."
                if self.live and self.peak_count > self.done_count
                else ""
            )
            rows.append(f"- Done ({self.done_count}), {reached}. {gained}.{upcoming}{lost}")
        if self.frontier:
            rows.append(
                "- Open now (every milestone whose prerequisites are already met): "
                + "; ".join(self.frontier)
                + "."
            )

        tail = f"ended on {self.ended_map or '?'} {_position(self.ended_pos)}"
        if self.ended_hp:
            tail += f", HP {self.ended_hp[0]}/{self.ended_hp[1]}, party {self.party_size}"
        rows.append(
            f"- Last session: {_plural(self.session_presses, 'press', 'presses')} over "
            f"{_plural(self.session_batches, 'batch', 'batches')}, {tail}."
        )

        if self.unique_positions:
            rate = self.presses_per_new_tile
            blocked = ""
            if self.session_batches:
                share = round(100 * self.blocked_batches / self.session_batches)
                blocked = (
                    f"; {self.blocked_batches:,} of {self.session_batches:,} "
                    f"batches ({share}%) moved nothing"
                )
            # Whiteouts and reloads used to have a line of their own and were
            # zero on almost every session that ever wrote one. They ride along
            # here instead, and say nothing at all when there is nothing to say.
            trouble = ""
            if self.whiteouts or self.reloads:
                trouble = f"; {self.whiteouts} whiteouts, {self.reloads} save reloads"
            rows.append(
                f"- Walking: {self.unique_positions:,} distinct tiles from "
                f"{self.position_samples:,} samples"
                + (f" ({rate} presses per new tile)" if rate else "")
                + blocked
                + trouble
                + "."
            )

        if self.hot_pos and self.hot_visits >= HOTSPOT_MIN_VISITS:
            rows.append(
                f"- Most revisited tile: {self.hot_map or '?'} {_position(self.hot_pos)}, "
                f"stood on {self.hot_visits:,} times."
            )

        # The two lines that exist because every false belief in the transcripts
        # was a compass direction the model invented and then acted on.
        if self.exits:
            # The legend only when a `*` is there to explain. Measured on the
            # live run's session 6, the line read "Every way off Route 3 (* =
            # never stepped on): walk north -> Route 4; walk west -> Pewter
            # City." with nothing starred: 22 bytes teaching a notation the
            # sentence does not use.
            legend = " (* = never stepped on)" if "*" in self.exits else ""
            rows.append(f"- Every way off {self.ended_map or 'this map'}{legend}: {self.exits}.")
        if self.route:
            rows.append(f"- Map graph says the way to {self.route_target}: {self.route}.")
        if self.heal_route:
            rows.append(f"- Party is hurt. Nearest heal is {self.heal_target}: {self.heal_route}.")

        # `action` is every batch of buttons and the line above already counted
        # those, so it is a fifth of this line saying nothing. What is worth a
        # token is which of the other verbs it reached for at all.
        beyond = [(name, count) for name, count in self.tool_mix if name != "action"]
        if beyond:
            rendered = ", ".join(f"{name} x{count}" for name, count in beyond)
            rows.append(f"- Verbs beyond walking: {rendered}.")

        if self.saves:
            # Named, not offered. This line used to end "loadable with `./poke
            # load <name>`" and sat at the top of every session, which is the
            # harness advertising the single most expensive behaviour in the
            # run: 143 loads, at least thirteen rungs handed back, and the same
            # two — Misty and the Cascade Badge — bought three times over. The
            # names are still worth having, because a save is how a branch is
            # recognised; the invitation is not.
            rows.append(
                "- Saves on disk, newest first: "
                + ", ".join(self.saves)
                + ". Loading one rewinds the game to it and keeps the bill; walking is usually "
                "cheaper than the milestones a reload hands back."
            )
        return rows

    def render(self, heading: str = FACTS_HEADING) -> str:
        """The block as it reaches the next session, under the same heading level
        as the retrospective it sits above."""

        rows = self.lines()
        return (f"## {heading}\n" + "\n".join(rows)) if rows else ""


def list_named_saves(data_dir: Optional[Path], limit: int = MAX_HANDOFF_SAVES) -> tuple[str, ...]:
    """The newest saves the agent named itself, newest first.

    A save it never lists is a save it never loads. ``./poke saves`` was called
    zero times across the nine sessions this was built from, while ``pewter_start``
    sat on disk through three sessions of the trap it would have escaped.
    """

    if data_dir is None or limit <= 0:
        return ()
    try:
        entries = [
            (entry.stat().st_mtime, entry.stem)
            for entry in (Path(data_dir) / SAVES_DIRNAME).iterdir()
            if entry.suffix == SAVE_SUFFIX and not entry.name.startswith(AUTO_SAVE_PREFIX)
        ]
    except OSError:
        return ()
    entries.sort(key=lambda item: (-item[0], item[1]))
    return tuple(name for _, name in entries[:limit])


def collect_session_facts(
    *,
    data_dir: Optional[Path],
    run_id: Optional[str],
    since_t: Optional[float] = None,
    session_index: int = 0,
    saves_limit: int = MAX_HANDOFF_SAVES,
    goal: str = "",
    live_milestones: Optional[Iterable[str]] = None,
) -> Optional[SessionFacts]:
    """Read the run back off disk. ``None`` when there is nothing to read.

    ``since_t`` splits the run into "the session that just ended" and everything
    before it. It is a wall clock rather than a sequence number on purpose: the
    sequence counter lives in a process a crash takes with it, and the mark file
    on disk does not.

    ``live_milestones`` is what the game holds right now, as
    :class:`~pokemon_agent.milestones.MilestoneTracker` read it off RAM. Pass it
    whenever the caller has one. Without it the only source here is the receipts,
    and the receipts only ever *add* milestones: a rung is written the batch it
    fires and nothing writes the subtraction when a reload lands on a branch that
    never had it. Every current-state claim below -- how many are done, the
    highest rung, the next rung, what is open now -- is therefore a high-water
    mark unless RAM supplied it. On the run this argument was added for the gap
    was 21 against 18, and the block claiming 21 opened the next session's first
    user message: it named "Got the Bicycle" as the highest rung to a game with no
    bicycle, and put Lt. Surge and the Rocket Hideout on a frontier computed from
    three rungs the cartridge had handed back.

    What stays on the receipts is what the run *did*: presses spent, milestones
    gained last session, whiteouts, reloads, tiles walked. Those are history and
    a reload does not undo them.
    """

    if data_dir is None or not run_id:
        return None
    try:
        record = RunRegistry(Path(data_dir)).load(run_id)
    except Exception:  # noqa: BLE001 — an unreadable run is a run with no facts
        return None
    receipts = tuple(record.receipts)
    if not receipts:
        return None

    metrics = compute_run_metrics(record)
    baseline: list[str] = []
    for receipt in receipts:
        raw = receipt.extra.get("baseline_milestones")
        if isinstance(raw, list):
            baseline = [str(item) for item in raw]
            break

    # An empty slice means the mark is newer than every receipt — a session that
    # pressed nothing. Reporting the whole run there would be a lie; reporting
    # zero is the truth, so the slice stands as it is.
    cutoff = None if since_t is None else since_t - RECEIPT_TIME_EPSILON
    session = [one for one in receipts if cutoff is None or one.t >= cutoff]
    # Newest first: the last rung reached is the one that says where the run is.
    earned = [item.milestone_id for item in reversed(metrics.attainments)]

    visits: Counter[tuple[str, int, int]] = Counter()
    tools: Counter[str] = Counter()
    gained: list[str] = []
    session_presses = batches = blocked = samples = reloads = party_size = 0
    # Rising edges, not flagged frames — see `whiteout_events`. The handoff line
    # this feeds told one session it had whited out 40 times when it had whited
    # out 19, which is the sort of number a next session plans around.
    whiteouts = whiteout_events(session)
    ended_map = ""
    ended_pos: Optional[tuple[int, int]] = None
    ended_hp: Optional[tuple[int, int]] = None
    for receipt in session:
        session_presses += receipt.presses
        gained.extend(receipt.milestones_new)
        if receipt.is_action_batch:
            batches += 1
            if receipt.moved == 0:
                blocked += 1
        if receipt.tool and receipt.tool != RUN_START_TOOL:
            tools[receipt.tool] += 1
        reloads += int(receipt.reloaded)
        if receipt.pos is not None:
            samples += 1
            visits[(receipt.map_name or "?", receipt.pos[0], receipt.pos[1])] += 1
            ended_map, ended_pos = receipt.map_name, receipt.pos
        if receipt.hp is not None:
            ended_hp = receipt.hp
        if receipt.party_size:
            party_size = receipt.party_size

    hot_map, hot_pos, hot_visits = "", None, 0
    if visits:
        (hot_map, x, y), hot_visits = visits.most_common(1)[0]
        hot_pos = (x, y)

    # Static ground truth about where the run ended up. It costs no model call
    # and no server call, so it reaches the next session even when the critic
    # never ran at all - which is the path nine sessions on disk actually took.
    reached = tuple(dict.fromkeys([*earned, *baseline]))
    done, live = _live_or_reached(live_milestones, reached)
    rung_done, rung_next = ladder_position(done)
    open_now = milestone_frontier(done)
    brief = map_brief(ended_map)
    # Every tile of this map the *run* ever stood on, not just this session: a
    # warp the run walked through six hours ago is not a warp it has never used.
    stood_on = {
        receipt.pos
        for receipt in receipts
        if receipt.pos is not None and receipt.map_name == ended_map
    }
    target = mentioned_map(goal, exclude=[ended_map])
    route = route_text(ended_map, target) if target else ""
    exits = brief.exits(stood_on) if brief.known else ""
    if route and ", then " not in route and target in exits:
        # One hop to somewhere the exits line already names, in the same words.
        # The second copy is fifteen tokens of the next session's first message.
        route = ""
    # A party this low is one wild Zubat from a whiteout, and the last time a
    # model was left to work out where to heal on its own it invented a Poke
    # Center in the wrong city and walked there at 10 HP.
    heal_target = heal_route = ""
    if ended_hp and ended_hp[1] and ended_hp[0] / ended_hp[1] <= HURT_HP_FRACTION:
        heal_target, _ = nearest_pokecenter(ended_map)
        heal_route = route_text(ended_map, heal_target) if heal_target else ""
        if not heal_route:
            heal_target = ""

    return SessionFacts(
        run_id=record.run_id,
        session_index=session_index or 1,
        total_presses=metrics.total_presses,
        done=done,
        done_count=len(done),
        peak_count=len(reached),
        live=live,
        gained=tuple(dict.fromkeys(gained)),
        session_presses=session_presses,
        session_batches=batches,
        blocked_batches=blocked,
        position_samples=samples,
        unique_positions=len(visits),
        ended_map=ended_map,
        ended_pos=ended_pos,
        ended_hp=ended_hp,
        party_size=party_size,
        whiteouts=whiteouts,
        reloads=reloads,
        hot_map=hot_map,
        hot_pos=hot_pos,
        hot_visits=hot_visits,
        tool_mix=tuple(tools.most_common(MAX_HANDOFF_TOOLS)),
        saves=list_named_saves(data_dir, saves_limit),
        rung_done=rung_done,
        rung_next=rung_next,
        frontier=open_now,
        heal_target=heal_target,
        heal_route=heal_route,
        exits=exits,
        route_target=target if route else "",
        route=route,
    )


# ---------------------------------------------------------------------------
# Session intelligence
#
# The facts block above is what the *next* session is told. This block is what
# the *critic* is told, and it is deliberately much larger: measurements the
# player never sees, about the session it just finished and the ones before it.
#
# Four things the transcripts said a retrospective could not be written without:
# where the presses actually went (a histogram, not an impression), which
# commands were repeated verbatim (thirteen identical `act right` calls in a
# row, never once mentioned by the model), how this session compares with the
# last few (three consecutive sessions repeated the same trapped walk), and what
# was never reached for at all (`progress` at zero calls across nine sessions,
# `saves` at zero while the save that would have escaped sat on disk).
#
# Every collector here returns an empty list rather than raising.
# ---------------------------------------------------------------------------

#: Sessions compared, newest last. Beyond this the table stops being readable.
MAX_TREND_SESSIONS = 6
#: Session start clocks kept in the mark file. One more than the table needs, so
#: the oldest slice still has a boundary to end at.
MAX_MARK_HISTORY = MAX_TREND_SESSIONS + 2
#: Repeated commands named before the tail collapses into a count.
MAX_REPEATED_COMMANDS = 8
#: A command has to be repeated at least this often before it is a habit.
MIN_REPEATS = 3
#: Reachable-but-never-walked tiles named, nearest first.
MAX_FRONTIER_TILES = 10
#: Maps given a coverage line, busiest first.
MAX_COVERAGE_MAPS = 6
#: Guide sections offered for the current map.
MAX_GUIDE_HITS = 4

EXPLORED_STORE_FILENAME = "explored_maps.json"

#: The verbs a stuck session most needs and least often reaches for. Sourced from
#: the CLI parser when it can be, so a verb added there shows up here without an
#: edit; the literal list is the fallback for when that introspection breaks.
FALLBACK_VERBS = (
    "act",
    "buy",
    "calc",
    "catch",
    "fight",
    "frame",
    "frontier",
    "goto",
    "guide",
    "health",
    "load",
    "map",
    "progress",
    "route",
    "run",
    "save",
    "saves",
    "sim",
    "state",
)


def cli_verbs() -> tuple[str, ...]:
    """Every ``./poke`` subcommand, off the parser that defines them."""

    if "verbs" not in _LOOKUP_CACHE:
        found: tuple[str, ...] = ()
        try:
            from pokemon_agent.agent_cli import build_parser

            for action in build_parser()._subparsers._group_actions:  # noqa: SLF001
                choices = getattr(action, "choices", None)
                if choices:
                    found = tuple(sorted(choices))
                    break
        except Exception:  # noqa: BLE001 — a parser that moved is not a reason to fail
            found = ()
        _LOOKUP_CACHE["verbs"] = found or FALLBACK_VERBS
    return _LOOKUP_CACHE["verbs"]


def called_verbs(calls: Iterable[ToolCall]) -> Counter:
    """How many times the session ran each ``./poke`` subcommand."""

    counts: Counter[str] = Counter()
    for call in calls:
        parsed = poke_subcommand(call.command or "")
        if parsed:
            counts[parsed[0]] += 1
    return counts


def receipts_for(data_dir: Optional[Path], run_id: Optional[str]) -> tuple[Any, ...]:
    """Every receipt in a run, or ``()``. The same read the facts block does."""

    if data_dir is None or not run_id:
        return ()
    try:
        return tuple(RunRegistry(Path(data_dir)).load(run_id).receipts)
    except Exception:  # noqa: BLE001
        return ()


def _bucket_classifier() -> Optional[Callable[..., str]]:
    """``scope.analysis.classify_receipt``, or ``None``.

    ``scope`` is another agent's module and this one only reads it. Importing it
    lazily means a rename there costs the critic a section, not a session.
    """

    if "classify" not in _LOOKUP_CACHE:
        try:
            from pokemon_agent.scope.analysis import classify_receipt
        except Exception:  # noqa: BLE001
            classify_receipt = None  # type: ignore[assignment]
        _LOOKUP_CACHE["classify"] = classify_receipt
    return _LOOKUP_CACHE["classify"]


def waste_lines(receipts: Sequence[Any], since_t: Optional[float]) -> list[str]:
    """Where the session's presses went, by bucket and by map.

    The buckets are ``scope``'s, so an operator reading ``scope waste`` and a
    critic reading this see the same split of the same presses. ``seen`` is
    seeded from every receipt before the session, which is what makes
    "productive" mean *new ground this session*, not new ground ever.
    """

    classify = _bucket_classifier()
    if classify is None or not receipts:
        return []
    cutoff = None if since_t is None else since_t - RECEIPT_TIME_EPSILON
    seen: set[tuple[str, int, int]] = set()
    presses: Counter[str] = Counter()
    per_map: dict[str, Counter[str]] = {}
    for receipt in receipts:
        tile = (
            (receipt.map_name or "?", receipt.pos[0], receipt.pos[1])
            if receipt.pos is not None
            else None
        )
        in_session = cutoff is None or receipt.t >= cutoff
        if receipt.presses > 0 and in_session:
            try:
                bucket = classify(receipt, seen, None)
            except Exception:  # noqa: BLE001
                bucket = "revisit"
            presses[bucket] += receipt.presses
            per_map.setdefault(receipt.map_name or "?", Counter())[bucket] += receipt.presses
        if tile is not None:
            seen.add(tile)

    total = sum(presses.values())
    if not total:
        return []
    rendered = ", ".join(
        f"{bucket} {count:,} ({round(100 * count / total)}%)"
        for bucket, count in presses.most_common()
    )
    rows = [f"- {total:,} presses this session: {rendered}."]
    for name, counts in sorted(per_map.items(), key=lambda item: -sum(item[1].values()))[
        :MAX_COVERAGE_MAPS
    ]:
        subtotal = sum(counts.values())
        wasted = counts.get("revisit", 0) + counts.get("blocked", 0)
        rows.append(
            f"- {name}: {subtotal:,} presses, "
            f"{round(100 * wasted / subtotal) if subtotal else 0}% of them revisiting "
            f"ground it had already stood on or walking into a wall."
        )
    return rows


def repeat_lines(calls: Sequence[ToolCall]) -> list[str]:
    """Commands the session sent over and over, and its longest identical run.

    SKILL.md says "if the same action fails three times, stop repeating it". One
    session on disk sent thirteen consecutive identical `./poke act right` calls
    and its own narration never mentions it. The model cannot see this about
    itself; the critic can be handed it.
    """

    commands = [
        " ".join(f"{verb} {' '.join(rest)}".split())
        for verb, rest in (
            parsed for parsed in (poke_subcommand(call.command or "") for call in calls) if parsed
        )
    ]
    if not commands:
        return []
    counts = Counter(commands)
    repeated = [(text, count) for text, count in counts.most_common() if count >= MIN_REPEATS]

    longest_run, longest_text, run, previous = 0, "", 0, None
    for command in commands:
        run = run + 1 if command == previous else 1
        previous = command
        if run > longest_run:
            longest_run, longest_text = run, command

    pairs: Counter[tuple[str, str]] = Counter(zip(commands, commands[1:]))
    rows: list[str] = []
    if repeated:
        rendered = ", ".join(
            f"`{text}` x{count}" for text, count in repeated[:MAX_REPEATED_COMMANDS]
        )
        rows.append(
            f"- {len(commands):,} poke commands, {len(counts):,} of them distinct. "
            f"Sent most often: {rendered}."
        )
    if longest_run >= MIN_REPEATS:
        rows.append(f"- Longest identical run: `{longest_text}` {longest_run} times back to back.")
    cycles = [(pair, count) for pair, count in pairs.most_common(3) if count >= MIN_REPEATS]
    if cycles:
        rendered = ", ".join(f"`{one}` -> `{two}` x{count}" for (one, two), count in cycles)
        rows.append(f"- Two-step cycles it kept re-entering: {rendered}.")
    return rows


def coverage_lines(
    data_dir: Optional[Path],
    *,
    map_name: str,
    map_id: Optional[int],
    player: Optional[tuple[int, int]],
    warps: Iterable[tuple[int, int, str]] = (),
) -> list[str]:
    """What the explored-map store knows about the map the session ended on.

    Coverage is the cheap half; the expensive and useful half is the frontier —
    walkable ground reachable from where it is standing that it has never walked
    on. A session that reports "nowhere left to go" with sixty reachable unwalked
    tiles under its feet has said something the receipts can contradict.
    """

    if data_dir is None or map_id is None:
        return []
    try:
        from pokemon_agent.explored_map import ExploredMaps

        store = ExploredMaps(Path(data_dir) / EXPLORED_STORE_FILENAME)
        if not store.knows(map_id):
            return []
        grid = store.grid(map_id)
        counts = store.coverage(map_id)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(grid, dict) or not isinstance(counts, dict):
        return []

    rows = [
        f"- {map_name or map_id} explored: {counts.get('seen', 0)} of {counts.get('total', 0)} "
        f"tiles seen ({counts.get('percent', 0)}%), {counts.get('walkable_seen', 0)} walkable, "
        f"{counts.get('walked', 0)} actually walked on."
    ]
    walked = grid.get("walked") or set()
    origin = player if player is not None else store.player_position(map_id)
    if origin is not None:
        try:
            from pokemon_agent.world import frontier

            reachable = frontier(grid, walked, origin)
        except Exception:  # noqa: BLE001
            reachable = ()
        if reachable:
            nearest = ", ".join(f"({x},{y})" for x, y in reachable[:MAX_FRONTIER_TILES])
            rows.append(
                f"- Reachable from ({origin[0]},{origin[1]}) and never walked on: "
                f"{len(reachable)} tiles. Nearest first: {nearest}."
            )
        else:
            rows.append(
                f"- Nothing walkable and unwalked is reachable from "
                f"({origin[0]},{origin[1]}). Every way on from here is a warp or an edge."
            )
    missed = [(x, y, target) for x, y, target in warps if (x, y) not in walked]
    if missed:
        rendered = ", ".join(f"({x},{y})->{target or '?'}" for x, y, target in missed)
        rows.append(f"- Warp tiles on this map it never stood on: {rendered}.")
    return rows


@dataclass(frozen=True)
class SessionSlice:
    """One session of a run, measured off the receipts alone."""

    index: int
    presses: int = 0
    batches: int = 0
    blocked: int = 0
    new_tiles: int = 0
    milestones: int = 0
    ended_map: str = ""

    @property
    def presses_per_new_tile(self) -> Optional[float]:
        if not self.new_tiles or not self.presses:
            return None
        return round(self.presses / self.new_tiles, 1)

    def line(self) -> str:
        rate = self.presses_per_new_tile
        blocked = round(100 * self.blocked / self.batches) if self.batches else 0
        return (
            f"- s{self.index}: {self.presses:,} presses, {self.new_tiles} tiles it had never "
            f"stood on ({rate if rate is not None else 'n/a'} presses each), {blocked}% of "
            f"batches moved nothing, +{self.milestones} milestones, ended on "
            f"{self.ended_map or '?'}."
        )


def session_slices(receipts: Sequence[Any], starts: Sequence[tuple[int, float]]) -> list[Any]:
    """Split a run's receipts by session start time and measure each slice.

    ``starts`` comes off the session mark, which the harness stamps at every
    session start and a crash cannot lose. The receipts themselves carry no
    session marker: ``run_start`` is written once per *run*, and a run is nine
    sessions long.
    """

    ordered = sorted(starts, key=lambda item: item[1])
    if not ordered or not receipts:
        return []
    seen: set[tuple[str, int, int]] = set()
    slices: list[SessionSlice] = []
    bounds = [
        (index, start, ordered[position + 1][1] if position + 1 < len(ordered) else float("inf"))
        for position, (index, start) in enumerate(ordered)
    ]
    for index, start, stop in bounds:
        presses = batches = blocked = fresh = milestones = 0
        ended = ""
        for receipt in receipts:
            tile = (
                (receipt.map_name or "?", receipt.pos[0], receipt.pos[1])
                if receipt.pos is not None
                else None
            )
            inside = start - RECEIPT_TIME_EPSILON <= receipt.t < stop
            if inside:
                presses += receipt.presses
                milestones += len(receipt.milestones_new)
                if receipt.is_action_batch:
                    batches += 1
                    if receipt.moved == 0:
                        blocked += 1
                if tile is not None:
                    ended = receipt.map_name or ended
                    if tile not in seen:
                        fresh += 1
            if tile is not None and receipt.t < stop:
                seen.add(tile)
        slices.append(
            SessionSlice(
                index=index,
                presses=presses,
                batches=batches,
                blocked=blocked,
                new_tiles=fresh,
                milestones=milestones,
                ended_map=ended,
            )
        )
    return slices[-MAX_TREND_SESSIONS:]


def trend_lines(receipts: Sequence[Any], starts: Sequence[tuple[int, float]]) -> list[str]:
    """The last few sessions side by side, so "again" is a measurable word.

    A retrospective that can say "you did this same thing last session and it did
    not work" is worth more than one describing a single session in isolation.
    """

    slices = session_slices(receipts, starts)
    if len(slices) < 2:
        return []
    rows = [one.line() for one in slices]
    latest, previous = slices[-1], slices[-2]
    if latest.presses_per_new_tile and previous.presses_per_new_tile:
        change = latest.presses_per_new_tile - previous.presses_per_new_tile
        direction = "worse" if change > 0 else "better"
        rows.append(
            f"- Presses per new tile went {direction} than last session "
            f"({previous.presses_per_new_tile} -> {latest.presses_per_new_tile})."
        )
    return rows


def guide_lines(map_name: str, read_slugs: Iterable[str]) -> list[str]:
    """Guide sections about this map, and whether the session opened them."""

    if not map_name:
        return []
    try:
        from pokemon_agent import guides

        hits = guides.search(map_name, limit=MAX_GUIDE_HITS)
    except Exception:  # noqa: BLE001
        return []
    already = {str(one).lower() for one in read_slugs}
    unread = [
        f"`./poke guide {section.guide}/{section.slug}`"
        for section in hits
        if f"{section.guide}/{section.slug}".lower() not in already
    ]
    if not unread:
        return []
    return [f"- Guide sections about {map_name} it did not open: {', '.join(unread)}."]


def untried_lines(
    calls: Sequence[ToolCall],
    *,
    map_name: str = "",
    facts: Optional[Any] = None,
) -> list[str]:
    """Verbs the session never called, and guide sections it never opened.

    Across the nine sessions this was built from, ``progress`` was called zero
    times, ``saves`` zero times and ``load`` zero times, while a save named
    ``pewter_start`` sat on disk through three sessions of the trap it would have
    escaped. Naming the untried verb in the handoff is what got ``load`` called
    for the first time.
    """

    counts = called_verbs(calls)
    if not counts:
        # A session with no parsed commands at all has nothing to say here, and
        # "it never called any of the seventeen verbs" is a sentence, not a fact.
        return []
    rows: list[str] = []
    never = [verb for verb in cli_verbs() if not counts.get(verb)]
    if never:
        rows.append(f"- Verbs it never called this session: {', '.join(never)}.")
    if counts:
        rendered = ", ".join(f"{verb} x{count}" for verb, count in counts.most_common(10))
        rows.append(f"- Verbs it did call: {rendered}.")
    if facts is not None and getattr(facts, "saves", ()):
        loaded = counts.get("load", 0)
        rows.append(
            f"- {len(facts.saves)} named saves are on disk and `./poke load` was called "
            f"{loaded} time(s) this session."
        )
    read_slugs = [
        rest[0]
        for verb, rest in (
            parsed for parsed in (poke_subcommand(call.command or "") for call in calls) if parsed
        )
        if verb == "guide" and rest
    ]
    rows.extend(guide_lines(map_name, read_slugs))
    return rows


# ---------------------------------------------------------------------------
# Claims, checked
#
# A retrospective on disk copied "machine INACCESSIBLE (confirmed)" out of the
# agent's own NOTES.md and handed it to the next session as ground truth. The
# agent healed 10 -> 65 HP at that machine twenty-six seconds later. The word
# "confirmed" in a model's notes confirms nothing; it is a claim, and the only
# thing that settles a claim is a measurement or a lookup.
#
# So every checkable claim the model wrote gets checked here, before the critic
# reads it, against the same generated map data the game was built from. A
# coordinate is either a warp or it is not. A direction is either the edge the
# map graph names or it is wrong.
# ---------------------------------------------------------------------------

#: Coordinates and directions checked. Both are bounded: the point is to catch
#: the load-bearing claim, not to audit every sentence.
MAX_CHECKED_COORDS = 14
MAX_CHECKED_DIRECTIONS = 8
#: How close a compass word has to sit to a map name to be read as a claim about
#: reaching that map, rather than a step inside the map already stood on.
DIRECTION_WINDOW = 40

COMPASS = ("north", "south", "east", "west")

_COORD_RE = re.compile(r"\((\d{1,3})\s*,\s*(\d{1,3})\)")
_COMPASS_RE = re.compile(r"(?i)\b(north|south|east|west)\b")

CLAIMS_HEADING = "Claims the agent made, checked against the map data"


def coordinate_claims(
    text: str,
    map_name: str,
    elsewhere: Iterable[str] = (),
    limit: int = MAX_CHECKED_COORDS,
) -> list[str]:
    """Every ``(x,y)`` the model wrote about, answered from the map data.

    One handoff on disk named the three B1F ladders as "the cave mouth out to
    Route 4" and the two actual Route 4 doors as "descend to B1F - do not take
    them". Every coordinate in it was in the world file, and every one of them
    said the opposite.

    Only the three answers worth a token are written: this tile is a warp, this
    tile is a warp but on a different map the run has been on, this tile is not
    on the map at all. A tile that is simply ordinary ground says nothing, and a
    line per ordinary tile would bury the three that matter.
    """

    brief = map_brief(map_name)
    if not brief.known or not text:
        return []
    warps = {(x, y): target for x, y, target in brief.warps}
    others = {
        (x, y): (name, target)
        for name in elsewhere
        if name and name != map_name
        for x, y, target in map_brief(name).warps
    }
    width, height = brief.size or (0, 0)
    rows: list[str] = []
    for found in dict.fromkeys(_COORD_RE.findall(text)):
        x, y = int(found[0]), int(found[1])
        if (x, y) in warps:
            rows.append(f"- ({x},{y}) IS a warp on {map_name}: it leads to {warps[(x, y)]}.")
        elif width and height and not (0 <= x < width and 0 <= y < height):
            rows.append(f"- ({x},{y}) is outside {map_name}, which is {width}x{height}.")
        elif (x, y) in others:
            name, target = others[(x, y)]
            rows.append(
                f"- ({x},{y}) is not a warp on {map_name}, but it is one on {name}, "
                f"leading to {target}."
            )
        if len(rows) >= limit:
            break
    return rows


def direction_claims(text: str, source: str, limit: int = MAX_CHECKED_DIRECTIONS) -> list[str]:
    """Compass words written next to a map name, checked against the map graph.

    This is the failure the transcripts are full of: the agent writes a bearing
    from memory, acts on it, and overrides a correct tool result to do it. The
    check is mechanical - the graph either has a connection on that edge or it
    does not.
    """

    if not text or not source:
        return []
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for found in _COMPASS_RE.finditer(text):
        word = found.group(1).lower()
        window = text[max(0, found.start() - DIRECTION_WINDOW) : found.end() + DIRECTION_WINDOW]
        target = mentioned_map(window, exclude=[source])
        if not target or (word, target) in seen:
            continue
        seen.add((word, target))
        route = route_text(source, target)
        if not route:
            rows.append(
                f'- "{word} ... {target}": there is no route from {source} to {target} '
                f"in the map data at all."
            )
        elif f"walk {word}" in route:
            rows.append(f'- "{word} ... {target}": agrees with the map data ({route}).')
        else:
            rows.append(
                f'- "{word} ... {target}": the map graph instead says {route}. The graph '
                f"knows which maps touch, not whether the way is walkable, so a disagreement "
                f"is a thing to check rather than a thing the agent got wrong."
            )
        if len(rows) >= limit:
            break
    return rows


def claim_lines(text: str, map_name: str, elsewhere: Iterable[str] = ()) -> list[str]:
    """Every checkable claim in one body of the model's own prose."""

    rows = coordinate_claims(text, map_name, elsewhere)
    rows.extend(direction_claims(text, map_name))
    return rows


@dataclass(frozen=True)
class Intel:
    """Everything measured that the finished session was never shown."""

    geography: tuple[str, ...] = ()
    coverage: tuple[str, ...] = ()
    waste: tuple[str, ...] = ()
    repeats: tuple[str, ...] = ()
    trend: tuple[str, ...] = ()
    untried: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()

    def sections(self) -> list[tuple[str, tuple[str, ...]]]:
        """Heading and body per block, in the order the critic should read them."""

        return [
            (
                "Where it is, from the map graph -- which maps touch, never "
                "whether the way is open",
                self.geography,
            ),
            ("The map it is standing in, from the explored-map store", self.coverage),
            ("Where the presses went (measured, bucketed)", self.waste),
            ("Commands it repeated", self.repeats),
            ("How this session compares with the ones before it", self.trend),
            ("What it never reached for", self.untried),
            (CLAIMS_HEADING, self.claims),
        ]

    def __bool__(self) -> bool:
        return any(body for _, body in self.sections())


def geography_lines(
    *,
    map_name: str,
    targets: Iterable[str] = (),
    stood_on: Iterable[tuple[int, int]] = (),
    visited_maps: Iterable[str] = (),
) -> list[str]:
    """The static truth about where the run is and how to leave it.

    This is the block that exists because of one specific failure: a model with
    no ground truth invented a Poke Center in the wrong city and sent a party at
    10 HP backwards to reach it. Every line here comes out of the generated world
    data, so the critic never has to take the session's word for a direction.
    """

    brief = map_brief(map_name)
    if not brief.known:
        return []
    rows = brief.lines(stood_on)
    center, hops = nearest_pokecenter(map_name)
    if center:
        route = route_text(map_name, center)
        rows.append(
            f"- Nearest place to heal: {center}, {hops} hop(s) away"
            + (f" — {route}." if route else ".")
        )
    for target in dict.fromkeys(one for one in targets if one and one != map_name):
        route = route_text(map_name, target)
        rows.append(
            f"- Route to {target}: {route}."
            if route
            else f"- There is no route from {map_name} to {target} in the map data."
        )
    seen = set(visited_maps)
    reachable = [target for _, target in brief.connections]
    reachable += [target for _, _, target in brief.warps if target]
    unvisited = [name for name in dict.fromkeys(reachable) if name not in seen]
    if unvisited:
        rows.append(f"- Maps one hop away this run has never been on: {', '.join(unvisited)}.")
    return rows


def collect_intel(
    *,
    data_dir: Optional[Path],
    run_id: Optional[str] = None,
    since_t: Optional[float] = None,
    facts: Optional[Any] = None,
    calls: Sequence[ToolCall] = (),
    goal: str = "",
    objective: str = "",
    notes: str = "",
    session_starts: Sequence[tuple[int, float]] = (),
) -> Intel:
    """Assemble every measurement the critic gets and the player never did.

    Each block is collected independently and a block that raises is simply an
    empty block: the critic must never be the reason a session fails to start.
    """

    map_name = getattr(facts, "ended_map", "") or ""
    player = getattr(facts, "ended_pos", None)
    receipts = receipts_for(data_dir, run_id)
    stood_on = {
        receipt.pos
        for receipt in receipts
        if receipt.pos is not None and (receipt.map_name or "") == map_name
    }
    visited_maps = {receipt.map_name for receipt in receipts if receipt.map_name}
    brief = map_brief(map_name)
    targets = [mentioned_map(goal), mentioned_map(objective)]

    def guarded(call: Callable[[], list[str]]) -> tuple[str, ...]:
        try:
            return tuple(call())
        except Exception:  # noqa: BLE001 — one dead block, not a dead critic
            return ()

    return Intel(
        geography=guarded(
            lambda: geography_lines(
                map_name=map_name,
                targets=targets,
                stood_on=stood_on,
                visited_maps=visited_maps,
            )
        ),
        coverage=guarded(
            lambda: coverage_lines(
                data_dir,
                map_name=map_name,
                map_id=brief.map_id,
                player=player,
                warps=brief.warps,
            )
        ),
        waste=guarded(lambda: waste_lines(receipts, since_t)),
        repeats=guarded(lambda: repeat_lines(calls)),
        trend=guarded(lambda: trend_lines(receipts, session_starts)),
        untried=guarded(lambda: untried_lines(calls, map_name=map_name, facts=facts)),
        claims=guarded(
            lambda: claim_lines(
                "\n".join([notes, *narration_lines(list(calls))]),
                map_name,
                visited_maps,
            )
        ),
    )


def session_mark_path(workspace_dir: Path) -> Path:
    return critic_debug_dir(workspace_dir) / SESSION_MARK_FILENAME


def read_session_mark(workspace_dir: Path) -> JsonDict:
    """What the previous session stamped about itself, or ``{}``."""

    try:
        payload = json.loads(session_mark_path(workspace_dir).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def session_starts(mark: JsonDict, run_id: Optional[str] = None) -> tuple[tuple[int, float], ...]:
    """``(session index, start clock)`` per session in the mark, oldest first.

    Filtered to one run: sessions from a run that has been closed and replaced
    cannot be compared with sessions from this one, because the receipts they
    would be measured against belong to a different file.
    """

    rows: list[tuple[int, float]] = []
    for entry in mark.get("history") or []:
        if not isinstance(entry, dict):
            continue
        started = entry.get("started_t")
        if not isinstance(started, (int, float)):
            continue
        if run_id and str(entry.get("run_id") or "") not in {"", run_id}:
            continue
        rows.append((int(entry.get("session_index") or 0), float(started)))
    return tuple(sorted(rows, key=lambda item: item[1]))


def write_session_mark(
    workspace_dir: Path, *, run_id: Optional[str], started_t: float, session_index: int
) -> None:
    """Stamp this session into the workspace. Best effort, never raises.

    The mark also keeps the last few sessions' start clocks, because nothing else
    on disk does. ``run_start`` is written once per *run* and a run is many
    sessions long, so without this history the receipts cannot be cut into
    sessions and "you did this same thing last session" is unmeasurable.
    """

    previous = read_session_mark(workspace_dir)
    entry = {
        "run_id": run_id or "",
        "started_t": float(started_t),
        "session_index": int(session_index),
    }
    history = [one for one in (previous.get("history") or []) if isinstance(one, dict)]
    if not history and isinstance(previous.get("started_t"), (int, float)):
        # First write after the upgrade: the mark already on disk is one session.
        history = [
            {
                "run_id": str(previous.get("run_id") or ""),
                "started_t": float(previous["started_t"]),
                "session_index": int(previous.get("session_index") or 0),
            }
        ]
    history = [one for one in history if one.get("started_t") != entry["started_t"]]
    history.append(entry)
    payload = {**entry, "history": history[-MAX_MARK_HISTORY:]}
    with contextlib.suppress(OSError):
        critic_debug_dir(workspace_dir).mkdir(parents=True, exist_ok=True)
        session_mark_path(workspace_dir).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )


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
    #: Receipts, not recollection. Rendered first and never trimmed.
    facts: Optional[SessionFacts] = None
    #: Measurements and static game data the session itself never saw. A few
    #: hundred tokens, rendered under the facts and never trimmed either.
    intel: Optional[Intel] = None


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

    intel = data.intel or Intel()

    def render(narration_rows: list[str], recent_rows: list[str], notes: str) -> str:
        sections = [
            "# Finished session digest",
            _section(FACTS_DIGEST_HEADING, data.facts.lines() if data.facts else []),
            *(_section(title, body) for title, body in intel.sections()),
            _section("Session", header),
            _section("Game state at the start of the session", format_game_state(data.start_state)),
            _section("Game state now", format_game_state(data.final_state)),
            _section("What it did (measured, not reported)", format_stats(stats)),
            _section("Explored-map coverage", format_map_summary(data.map_summary)),
            _section(NARRATION_HEADING, narration_rows),
            _section(f"Last {len(recent_rows)} tool calls, oldest first", recent_rows),
            _section(NOTES_HEADING, quoted_lines(notes)),
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

You are given more than the agent had. Everything above the "Session" heading is measured or
generated: receipts the server wrote after every batch, the game's own map data, the
explored-map store, and a press-by-press split of where the buttons went. The agent saw none of
it. Use it - most of what a stuck session needs is a number it could not see about itself.

Ground truth beats narration, without exception. Where a measured block and the agent's own
account disagree, the measurement wins and a claim it contradicts is a claim you must not make.
The agent's narration is evidence of what it believed, not of what happened; every false belief
found in these transcripts so far has been a compass direction the model invented and then acted
on against a correct tool result.

The map data is the exception, and it is a large one. The connection and warp tables say which
maps touch. They do not model a cuttable tree, a guard who wants a drink, a badge lock or a
boulder, so an exit they list can be one no player can take. When the agent reports walking at an
exit the graph lists and being stopped, the graph is the thing that is incomplete: it has never
seen a tile. Eight retrospectives in one run told the agent its own correct observation of the
Vermilion gym tree was "a belief, not a finding" because the graph listed the door as a live warp;
the door was behind a tree the whole time, the agent was right, and it spent 4,480 tool calls in
that city without opening it.

The narration and NOTES.md blocks are the agent's claims. They are not evidence. The word
"confirmed" in them confirms nothing: one retrospective repeated "machine INACCESSIBLE
(confirmed)" out of NOTES.md as though it were a finding, and the agent healed at that machine
twenty-six seconds later. Never restate a claim as a fact. If you must mention one, say the
agent believed it, and say what the check above found.

Every direction, coordinate and destination you write must come from the map-data blocks, not
from anything you know about Pokemon Red and not from the agent's account. If a place cannot be
routed to from where the run is standing, do not name it.

The "{FACTS_DIGEST_HEADING}" block is also handed to the next agent verbatim,
above whatever you write, so do not restate what it says.

The retrospective is the next agent's first instruction. Cover, in this order and with no
preamble:

1. The mistake that cost the most presses, cited with numbers from above.
2. Concrete, checkable things to do differently, and what to do instead.
3. Anything learned about this map worth carrying forward: exits, walls, ledges, encounters.

Prefer, in that order: something the session repeated that the comparison block shows it also
repeated last session; an exit, warp or route in the map data it never used; a tile the frontier
says is reachable and unwalked; a verb it never called. A retrospective that says "you did this
same thing last session and it did not work" is worth more than one describing this session
alone.

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

The goal is checked against the map graph before it is used, and one that fails the check is
thrown away. It fails if it names a map with no route from where the run is standing, or if it
names a compass direction next to a map the graph reaches some other way. A goal saying "head
south on Route 3 to Cerulean City" cost this run 5,618 presses; the harness had printed the
correct two hops three seconds earlier. So: name a destination only if the map-data block above
routes to it, and use that block's own wording for the direction.
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


def check_next_goal(goal: str, *, from_map: str) -> tuple[str, str]:
    """``(goal to use, why it was rejected)``. An empty goal is one to drop.

    The critic's goal is not advice, it is the next session's first line, and the
    session acts on it in preference to a correct tool result three seconds old.
    One that said "head south on Route 3 to Cerulean City" cost this run 5,618
    presses while the harness had already printed the right two hops. So a goal
    that names a place gets routed to before it is allowed to leave the building.

    Two rejections, both mechanical:

    * it names a map the graph cannot reach from where the run is standing;
    * it names a compass direction beside a map the graph reaches another way.

    A goal that names no map is not checkable and is left alone - the fallback for
    a dropped goal is the objective engine, which is not free, so this refuses
    only what it can prove wrong.
    """

    text = (goal or "").strip()
    if not text or not from_map:
        return text, ""
    target = mentioned_map(text, exclude=[from_map])
    if not target:
        return text, ""
    route = route_text(from_map, target)
    if not route:
        return "", f"names {target}, which the map graph cannot reach from {from_map}"
    for found in _COMPASS_RE.finditer(text):
        word = found.group(1).lower()
        window = text[max(0, found.start() - DIRECTION_WINDOW) : found.end() + DIRECTION_WINDOW]
        if mentioned_map(window, exclude=[from_map]) != target:
            continue
        if f"walk {word}" not in route:
            return "", (f'says "{word}" about {target}, but the map graph says {route}')
    return text, ""


#: Sentence ends, inside a line. Lines are kept whole so a markdown bullet or
#: heading survives a strike from the middle of the paragraph next to it.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: Further than any two maps in Kanto are apart, so the hop rule never fires on
#: a handoff. See the comment in :func:`strike_false_claims`.
HANDOFF_HOP_CEILING = 999


def strike_false_claims(text: str, *, from_map: str) -> tuple[str, list[str]]:
    """``(handoff to write, sentences struck)``.

    ``check_next_goal`` gatekeeps the one line that becomes the next session's
    goal. Nothing gatekeeps the body, and the body is delivered verbatim as
    that session's first user message — which is to say the retrospective gets
    read as ground truth by a model that will then spend a session acting on
    it. The same map data that rejects a goal rejects a sentence, so it does.

    Struck rather than dropped whole, because a retrospective is not only
    geography: "you spent 61% of the session in battle" is worth keeping even
    in a handoff whose last paragraph invented a door. A body left empty by the
    striking is no handoff at all and the caller is told so with ``""``.
    """

    if not text or not from_map:
        return text, []
    try:
        from pokemon_agent.interventions import check_advice
    except Exception:  # noqa: BLE001 — no map data, nothing to check against
        return text, []
    try:
        # No hop ceiling here. The ceiling in `check_advice` is for a message
        # steering the next few hundred presses, where naming somewhere eight
        # warps off is the tell; a retrospective's job includes naming the run's
        # destination, and Cerulean City is five hops from Pallet Town whether
        # or not the sentence about it is true. Only what the map data
        # contradicts outright gets struck from a handoff.
        claims = check_advice(text, here=from_map, max_hops=HANDOFF_HOP_CEILING)
    except Exception:  # noqa: BLE001
        return text, []
    if not claims:
        return text, []

    # Checked once over the whole body so a sentence keeps the context its
    # neighbours give it, then matched back by the span each claim quoted.
    disproved = [claim.said for claim in claims if claim.said]
    struck: list[str] = []
    kept_lines: list[str] = []
    for line in text.splitlines():
        parts = _SENTENCE_RE.split(line)
        keep = []
        for part in parts:
            if part.strip() and any(said in part for said in disproved):
                struck.append(part.strip())
                continue
            keep.append(part)
        joined = " ".join(piece for piece in keep if piece).strip()
        if joined or not line.strip():
            kept_lines.append(joined if line.strip() else line)
    body = "\n".join(kept_lines).strip()
    return body, struck


def handoff_path(workspace_dir: Path) -> Path:
    return Path(workspace_dir) / HANDOFF_FILENAME


def handoff_body(text: str) -> str:
    """The retrospective without its ``NEXT GOAL:`` line.

    The goal travels on its own: :func:`parse_next_goal` lifts it out,
    :func:`check_next_goal` rules on it, and the supervisor puts the survivor at
    the top of the first user message. The line stays in ``HANDOFF.md`` for the
    post-mortem, and every one of the three ways it can then reach the model
    again is worse than not sending it:

    * accepted -- the same sentence twice in one message, 83 bytes of it, once
      as the instruction and once as the last line the model reads;
    * rejected -- ``check_next_goal`` threw it away for naming an unroutable map
      or a direction the map graph contradicts, and shipping it anyway hands the
      model the exact line the check exists to stop. That check's own docstring
      prices one such goal at 5,618 presses;
    * overridden -- an operator goal is in force and the file ends on a
      different instruction, so the message contradicts itself.

    Everything above the line is left alone, including a truncation marker.
    """

    lines = (text or "").splitlines()
    kept: list[str] = []
    dropping = False
    for line in lines:
        match = _NEXT_GOAL_RE.match(line)
        if match is not None:
            # `parse_next_goal` also accepts the label alone on its line with the
            # goal underneath it, so keep dropping until the goal is gone too.
            dropping = not _clean_goal_line(match.group(1))
            continue
        if dropping:
            if not line.strip():
                continue
            dropping = False
            if _clean_goal_line(line):
                continue
        kept.append(line)
    return "\n".join(kept).strip()


#: The headings the next session's first user message is assembled from. The
#: block under the first one is measured off the receipts; the block under the
#: second is prose a model wrote, and it is appended below the measurements.
FIRST_MESSAGE_HEADINGS = (FACTS_HEADING, HANDOFF_HEADING)


def quote_forged_headings(text: str, headings: Sequence[str] = FIRST_MESSAGE_HEADINGS) -> str:
    """*text* with any line that opens like one of *headings* quoted instead.

    The retrospective is model-written and it is concatenated under the
    harness's own ``## `` headings, one of which announces the block above it as
    authoritative and tells the reader not to contradict it. The critic is
    handed that heading by name in its instructions, so writing it back out is
    not a stretch — and a second "Ground truth from the run receipts" heading,
    below the real one, would be read as more of the same measurements.

    Only a line that opens like one of the harness's own headings is touched:
    a heading the retrospective invented for itself impersonates nothing, and
    quoting every heading would rewrite prose this has no business rewriting.
    Same defence, and the same reason, as the body quoting in
    :func:`pokemon_agent.interventions.revise_advice`.
    """

    def forged(line: str) -> bool:
        bare = line.strip().lstrip("#").strip()
        return any(heading and bare.startswith(heading) for heading in headings)

    return "\n".join(f"> {line}" if forged(line) else line for line in (text or "").splitlines())


def read_handoff(workspace_dir: Path) -> str:
    """The previous session's retrospective, or nothing when there is not one.

    Two things a reader has to be protected from, both measured on the live run.

    A salvaged reasoning tail is not a retrospective. When the critic runs out
    of output budget mid-thought the tail gets kept, honestly labelled, and
    handed to the next session as the last thing it reads before acting. The
    live one was 1,648 bytes of the critic counting words at itself --
    "Most(1) costly(2) mistake(3)..." -- inside a first message that is only
    about 2,100 bytes in total. So the majority of a new session's handoff was
    a transcript of the critic failing to write one.

    It still lands in `HANDOFF.md` for a post-mortem, which is what that file is
    for. It just does not go to the model: the deterministic ground-truth block
    beside it already carries the run's real facts, and no retrospective is a
    smaller lie than a rambling one.

    And the word ceiling only ever existed on the write path, so a hand-edited
    or externally written file went in verbatim at any length. It is applied
    here too.
    """
    try:
        text = handoff_path(workspace_dir).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if text.startswith(SALVAGED_REASONING_NOTICE[:40]):
        return ""
    return cap_words(text)


#: Where a handoff goes when the critic that should have replaced it failed.
#: Kept rather than deleted: it is still the post-mortem of the session it was
#: actually written about.
HANDOFF_STALE_FILENAME = "HANDOFF.stale.md"


def retire_handoff(workspace_dir: Path) -> bool:
    """Move ``HANDOFF.md`` aside so a failed critic serves nothing at all.

    A critic pass that times out used to leave the previous file in place, and
    the next session was then told, in the present tense, about a session two
    back -- with its goal line silently reverting to the generic run objective.
    Measured over one run: 20 of 99 retrospectives were byte-identical repeats
    of the one before, and those sessions earned 4 milestones against a run rate
    of 12 in 112.

    A session told nothing reads the deterministic facts block beside it, which
    is the run's real state. A session told about the wrong session does not
    know it is being misled.
    """
    directory = Path(workspace_dir)
    current = directory / HANDOFF_FILENAME
    if not current.is_file():
        return False
    try:
        os.replace(current, directory / HANDOFF_STALE_FILENAME)
    except OSError:
        return False
    return True


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
    #: The critic's ``NEXT GOAL:`` line, or "" when it wrote none the parser trusts
    #: or the map graph refused the one it wrote. The supervisor uses it only when
    #: the operator has not named a goal of their own.
    next_goal: str = ""
    #: Why a goal the critic did write was thrown away, for the operator to read.
    next_goal_rejected: str = ""
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
    from_map: str = "",
    thinking: str = DEFAULT_CRITIC_THINKING,
    timeout_seconds: float = DEFAULT_CRITIC_TIMEOUT_SECONDS,
    include_images: bool = True,
    retry_enabled: bool = True,
    retry_thinking: str = DEFAULT_CRITIC_RETRY_THINKING,
    retry_min_seconds: float = CRITIC_RETRY_MIN_SECONDS,
    first_attempt_seconds: float = CRITIC_FIRST_ATTEMPT_SECONDS,
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

    first_budget = min(timeout_seconds, first_attempt_seconds) if retry_enabled else timeout_seconds
    attempt = await attempt_once(thinking, immediate=False, budget=first_budget)
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

    goal, rejected = check_next_goal(parse_next_goal(text), from_map=from_map)
    if rejected:
        error = f"{error + ' ' if error else ''}Dropped the NEXT GOAL: it {rejected}."
    text, struck = strike_false_claims(text, from_map=from_map)
    if struck:
        error = (
            f"{error + ' ' if error else ''}Struck {len(struck)} sentence"
            f"{'s' if len(struck) > 1 else ''} the map data contradicts: "
            f"{' | '.join(struck[:3])}"
        )
    if not text:
        return finish(ok=False, error=f"{error + ' ' if error else ''}Nothing left to hand off.")
    result = finish(
        ok=True,
        text=text,
        next_goal=goal,
        next_goal_rejected=rejected,
        error=error,
        salvaged=salvaged,
    )
    try:
        result.handoff_path = str(write_handoff(workspace, text))
    except OSError as exc:
        return finish(ok=False, text=text, error=f"Could not write {HANDOFF_FILENAME}: {exc}")
    return result
