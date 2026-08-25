"""One function per command. Each returns ``(lines, payload)``.

``lines`` is the compact human read — the default, and the one that has to fit
in a screen. ``payload`` is the same content as data for ``--json``. They are
produced together so the two can never drift apart.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from pokemon_agent.bench.registry import Receipt, RunRecord, RunRegistry
from pokemon_agent.scope import analysis, render
from pokemon_agent.scope.discover import Paths, list_sessions, resolve_session
from pokemon_agent.scope.runs import (
    ContextOracle,
    LadderProgress,
    ladder_progress,
    read_action_contexts,
    read_json,
    receipts_between,
    resolve_run_id,
    run_metrics,
)
from pokemon_agent.scope.transcript import Call, Session, parse_session

#: Rows a table prints before it collapses into "and N more". ``--full`` lifts it.
DEFAULT_ROWS = 14
LIVE_COMMANDS = 12

#: Absolute paths in a result are almost all workspace paths, and the leading
#: sixty characters of one are the same sixty characters every time.
_ABSOLUTE_PATH_RE = re.compile(r"(?:/[\w.@+-]+){3,}")


class ScopeError(RuntimeError):
    """Something the user has to fix — a missing workspace, an unknown run."""


@dataclass
class Context:
    """Everything a command might need, loaded once and only when asked for."""

    paths: Paths
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    full: bool = False
    window: int = analysis.DEFAULT_WINDOW

    _session: Optional[Session] = None
    _record: Optional[RunRecord] = None
    _oracle: Optional[ContextOracle] = None

    @property
    def row_limit(self) -> Optional[int]:
        return None if self.full else DEFAULT_ROWS

    def session(self) -> Session:
        if self._session is None:
            path = resolve_session(self.paths.session_dir, self.session_id)
            if path is None:
                where = self.paths.session_dir or "(no workspace found)"
                raise ScopeError(f"no session transcript under {where}")
            self._session = parse_session(path)
        return self._session

    def record(self, run_id: Optional[str] = None) -> RunRecord:
        if run_id is None and self._record is not None:
            return self._record
        if self.paths.data_dir is None:
            raise ScopeError("no run store found; pass --data-dir")
        wanted = resolve_run_id(self.paths.data_dir, run_id or self.run_id)
        if not wanted:
            raise ScopeError(f"no runs under {self.paths.data_dir / 'runs'}")
        try:
            record = RunRegistry(self.paths.data_dir).load(wanted)
        except FileNotFoundError as exc:
            raise ScopeError(str(exc)) from exc
        if run_id is None:
            self._record = record
        return record

    def oracle(self) -> ContextOracle:
        if self._oracle is None:
            self._oracle = ContextOracle(read_action_contexts(self.paths.run_log))
        return self._oracle


# -- shared bits --------------------------------------------------------------


def _press_stats(record: RunRecord) -> tuple[int, float, float]:
    """``(presses, elapsed_seconds, presses_per_minute)`` for a run."""

    metrics = run_metrics(record)
    elapsed = metrics.wall_clock_seconds
    rate = (metrics.total_presses / (elapsed / 60.0)) if elapsed > 0 else 0.0
    return metrics.total_presses, elapsed, round(rate, 1)


def _outcome(call: Call) -> str:
    """One line describing what a call returned."""

    if call.is_error:
        return "ERR " + render.truncate(call.result_text, 52)
    payload = call.result_json
    if payload is not None:
        bits = []
        for key, name in (
            ("moved", "moved"),
            ("here_before", "here"),
            ("actions_executed", "acts"),
        ):
            if key in payload:
                bits.append(f"{name}={payload[key]}")
        for key in ("dialog", "battle"):
            if payload.get(key):
                bits.append(key)
        if bits:
            return " ".join(bits)
    shortened = _ABSOLUTE_PATH_RE.sub(
        lambda match: ".../" + match.group(0).rsplit("/", 1)[-1], call.result_text
    )
    return render.truncate(shortened, 56) or "(no output)"


def _share(part: int, whole: int) -> str:
    return render.pct(part / whole) if whole else "0%"


def _sequence(tokens: tuple[str, ...]) -> str:
    """A loop's commands as one line, with the ``poke`` every token shares dropped."""

    if tokens and all(token.startswith("poke ") for token in tokens):
        tokens = tuple(token[len("poke ") :] for token in tokens)
    return " | ".join(tokens)


# -- live ---------------------------------------------------------------------


def command_live(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """One screen: where the game is, and what the model just did."""

    session = ctx.session()
    record = ctx.record()
    metrics = run_metrics(record)
    ladder = ladder_progress(record)
    presses, elapsed, rate = _press_stats(record)
    observation = read_json(ctx.paths.latest_observation)
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    party = state.get("party") if isinstance(state.get("party"), list) else []
    position = player.get("position") if isinstance(player.get("position"), dict) else {}
    map_block = state.get("map") if isinstance(state.get("map"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    dialog = state.get("dialog") if isinstance(state.get("dialog"), dict) else {}

    party_text = (
        ", ".join(
            f"{member.get('species', '?')} L{member.get('level', '?')} "
            f"{member.get('hp', '?')}/{member.get('max_hp', '?')}"
            for member in party
            if isinstance(member, dict)
        )
        or "(empty)"
    )
    hp_text = "-"
    if party and isinstance(party[0], dict):
        hp_text = f"{party[0].get('hp', '?')}/{party[0].get('max_hp', '?')}"

    mode = (
        "battle" if battle.get("in_battle") else ("dialog" if dialog.get("active") else "overworld")
    )
    last_write = ctx.paths.latest_observation
    age = None
    try:
        if last_write is not None:
            age = time.time() - last_write.stat().st_mtime
    except OSError:
        age = None

    calls = session.calls
    recent = calls[-(LIVE_COMMANDS if not ctx.full else 40) :]
    stuck = analysis.tail_repeat(session)

    lines = [
        f"LIVE  run {record.run_id}  {metrics.status}"
        + (f"  observation {render.human_seconds(age)} old" if age is not None else ""),
        render.kv(
            "session",
            f"{session.short_id}  {session.model or '?'}"
            f"  thinking={session.thinking_level or '?'}  steps {len(session.steps)}",
        ),
        render.kv(
            "where",
            f"{map_block.get('map_name') or '?'}"
            f" ({position.get('x', '?')},{position.get('y', '?')})"
            f" facing {player.get('facing') or '?'}  mode {mode}",
        ),
        render.kv("party", f"{party_text}   hp {hp_text}"),
        render.kv(
            "progress",
            f"ladder {ladder} milestones ({ladder.baseline} from checkpoint)"
            f"   badges {len(player.get('badges') or [])}",
        ),
        render.kv(
            "presses",
            f"{render.thousands(presses)} in {render.human_seconds(elapsed)}"
            f"   {rate}/min   {metrics.receipts} receipts",
        ),
        render.kv(
            "health",
            f"blocked {render.pct(metrics.blocked_rate)}"
            f"   errors {render.pct(metrics.tool_error_rate)}"
            f"   reloads {metrics.reloads}   whiteouts {metrics.whiteouts}",
        ),
        render.kv("goal", render.truncate(metrics.goal or session.goal, 66)),
    ]
    if stuck is not None and stuck.count >= 3:
        lines += [
            "",
            f"STUCK  {_sequence(stuck.tokens)}  repeating x{stuck.count} at the tail",
        ]
    lines += ["", f"last {len(recent)} commands"]
    lines += render.table(
        ["step", "command", "outcome"],
        [
            [str(call.step), render.truncate(call.signature or call.label, 34), _outcome(call)]
            for call in recent
        ],
        align="rll",
    )
    if session.corrupt_lines:
        lines.append(f"note: {session.corrupt_lines} unreadable transcript lines skipped")

    payload = {
        "run_id": record.run_id,
        "status": metrics.status,
        "session_id": session.session_id,
        "model": session.model,
        "thinking_level": session.thinking_level,
        "steps": len(session.steps),
        "map": map_block.get("map_name"),
        "position": [position.get("x"), position.get("y")],
        "mode": mode,
        "hp": hp_text,
        "party": party_text,
        "ladder": {"reached": ladder.reached, "total": ladder.total, "baseline": ladder.baseline},
        "presses": presses,
        "elapsed_seconds": round(elapsed, 1),
        "presses_per_minute": rate,
        "blocked_rate": metrics.blocked_rate,
        "reloads": metrics.reloads,
        "whiteouts": metrics.whiteouts,
        "stuck": stuck.to_dict() if stuck is not None else None,
        "recent": [
            {"step": call.step, "command": call.signature, "outcome": _outcome(call)}
            for call in recent
        ],
    }
    return lines, payload


# -- tools --------------------------------------------------------------------


def command_tools(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """The histogram that decides the next harness change."""

    session = ctx.session()
    report = analysis.tool_report(session)
    rows, hidden = render.capped(report.stats, ctx.row_limit)

    lines = [
        f"TOOLS  session {report.session_id}  {report.steps} steps"
        f"  {report.total_calls} calls  {render.human_seconds(session.elapsed)}",
        "",
    ]
    lines += render.table(
        ["verb/program", "calls", "fail", "med B", "med moved"],
        [
            [
                stat.label,
                str(stat.calls),
                str(stat.failures),
                str(stat.median_result_bytes),
                "-" if stat.median_moved is None else f"{stat.median_moved:g}",
            ]
            for stat in rows
        ],
        align="lrrrr",
    )
    if hidden:
        lines.append(f"  ... and {hidden} more (--full)")
    lines += [
        "",
        "advisory verbs (shipped, optional, historically ignored)",
    ]
    lines += render.table(
        ["verb", "calls", "fail"],
        [[verb, str(calls), str(failures)] for verb, calls, failures in report.advisory],
        align="lrr",
    )
    used = [verb for verb, calls, _ in report.advisory if calls]
    lines.append("  used: " + (", ".join(used) if used else "none of them"))
    lines += [
        "",
        render.kv(
            "bash",
            f"{report.bash_calls} calls"
            f"   ./poke {report.bash_poke_calls}"
            f" ({_share(report.bash_poke_calls, report.bash_calls)})"
            f"   other {report.bash_other_calls}"
            f" ({_share(report.bash_other_calls, report.bash_calls)})",
        ),
    ]
    programs, program_overflow = render.capped(report.other_programs, ctx.row_limit)
    lines.append(
        render.kv(
            "other",
            ", ".join(f"{name.lstrip('!')} {count}" for name, count in programs) or "(none)",
        )
    )
    if program_overflow:
        lines.append(f"          ... and {program_overflow} more (--full)")
    lines.append(
        render.kv(
            "results",
            f"{render.si_bytes(report.result_bytes)} of tool output entered the context",
        )
    )
    lines.append("")
    lines.append("med B = median bytes of the tool result; med moved = median tiles moved.")
    lines.append("A bash line invoking two verbs counts once per verb, so verbs > bash calls.")
    return lines, report.to_dict()


# -- waste --------------------------------------------------------------------


def command_waste(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """Where the presses went."""

    record = ctx.record()
    oracle = ctx.oracle()
    report = analysis.waste_report(record, oracle)
    overall = report.overall

    lines = [
        f"WASTE  run {report.run_id}  {render.thousands(overall.total_presses)} presses"
        f"  {overall.total_batches} batches  {report.milestones} milestones",
        "",
    ]
    lines += render.table(
        ["bucket", "presses", "share", "batches"],
        [
            [
                bucket,
                render.thousands(overall.presses.get(bucket, 0)),
                render.pct(overall.share(bucket)),
                str(overall.batches.get(bucket, 0)),
            ]
            for bucket in analysis.WASTE_BUCKETS
        ],
        align="lrrr",
    )
    maps, hidden = render.capped(report.by_map, ctx.row_limit)
    lines += ["", "per map"]
    lines += render.table(
        ["map", "presses"] + list(analysis.WASTE_BUCKETS),
        [
            [split.name, render.thousands(split.total_presses)]
            + [render.pct(split.share(bucket)) for bucket in analysis.WASTE_BUCKETS]
            for split in maps
        ],
        align="lr" + "r" * len(analysis.WASTE_BUCKETS),
    )
    if hidden:
        lines.append(f"  ... and {hidden} more maps (--full)")
    lines += [
        "",
        "one bucket per batch, first match wins:",
        "  productive  earned a milestone, or ended on a tile never stood on before",
        "  battle/dialog  the game was in one when the batch landed (from run_log)",
        "  blocked     pressed buttons and ended where it started",
        "  revisit     moved, but onto ground already walked",
    ]
    if not report.context_samples:
        lines.append("  (no run_log found: battle and dialog cannot be separated out)")
    if report.unclassified_presses:
        lines.append(f"  {report.unclassified_presses} presses had no position recorded")
    return lines, report.to_dict()


# -- loops --------------------------------------------------------------------


def command_loops(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """Repeated command sequences, longest and most frequent first."""

    session = ctx.session()
    loops = analysis.find_loops(session, limit=40 if ctx.full else 10)
    stuck = analysis.tail_repeat(session)
    total_calls = len(session.calls)

    lines = [
        f"LOOPS  session {session.short_id}  {total_calls} calls  {len(session.steps)} steps",
        "",
    ]
    if not loops:
        lines.append("  no command sequence repeats. Nothing is looping.")
    else:
        lines += render.table(
            ["calls", "reps", "len", "sequence", "steps", "where"],
            [
                [
                    str(loop.covered),
                    f"x{loop.count}",
                    str(loop.length),
                    render.truncate(_sequence(loop.tokens), 50),
                    f"{loop.first_step}-{loop.last_step}",
                    render.truncate(loop.where, 16),
                ]
                for loop in loops
            ],
            align="rrrlll",
        )
        covered = loops[0].covered
        lines.append(
            f"  worst loop covers {covered} of {total_calls} calls"
            f" ({render.pct(covered / total_calls) if total_calls else '0%'} of the session)"
        )
    lines.append("")
    if stuck is not None and stuck.count >= 2:
        lines.append(
            f"right now: {_sequence(stuck.tokens)} repeating x{stuck.count}"
            f" (steps {stuck.first_step}-{stuck.last_step})"
        )
    else:
        lines.append("right now: the tail is not repeating.")
    lines.append("")
    lines.append("reps are non-overlapping; a sub-sequence of an equally frequent longer loop is")
    lines.append("dropped. calls = len x reps, the number of turns the loop consumed.")

    payload = {
        "session_id": session.session_id,
        "calls": total_calls,
        "loops": [loop.to_dict() for loop in loops],
        "tail_repeat": stuck.to_dict() if stuck is not None else None,
    }
    return lines, payload


# -- context ------------------------------------------------------------------


def command_context(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """What is consuming the window, with images as a first-class number."""

    session = ctx.session()
    report = analysis.context_report(session, window=ctx.window)
    curve = report.curve

    lines = [
        f"CONTEXT  session {report.session_id}  {report.model}  {report.steps} steps"
        f"  window {render.thousands(report.window)}",
        "",
        render.kv(
            "prompt",
            f"first {render.thousands(report.first_prompt)}"
            f"   peak {render.thousands(report.peak_prompt)}"
            f" ({render.pct(report.peak_share)} of window) at step {report.peak_step}"
            f"   final {render.thousands(report.final_prompt)}",
            8,
        ),
        render.kv(
            "growth",
            f"{report.median_growth:+g} median, {report.mean_growth:+g} mean tokens per step",
            8,
        ),
        render.kv("output", f"{render.thousands(report.output_tokens)} tokens generated", 8),
        "",
        "prompt tokens per step",
        f"  {render.thousands(min(curve) if curve else 0)} "
        f"{render.sparkline(curve)} {render.thousands(max(curve) if curve else 0)}",
        "",
        "images",
    ]
    lines += render.table(
        ["", ""],
        [
            [
                "sent",
                f"{report.image_count}  ({report.tool_result_images} read as tool results,"
                f" {report.prompt_images} attached to prompts)",
            ],
            ["bytes", f"{render.si_bytes(report.image_bytes)} of base64"],
            ["size", f"{report.common_size} most common"],
            [
                "est cost",
                f"~{render.thousands(report.est_tokens_per_image)} tokens each (median),"
                f" ~{render.thousands(report.image_tokens_total)} in total",
            ],
            [
                "at peak",
                f"~{render.thousands(report.image_tokens_at_peak)} of"
                f" {render.thousands(report.peak_prompt)} prompt tokens are images"
                f" ({render.pct(report.image_share)}),"
                f" ~{render.thousands(report.text_tokens_at_peak)} are text",
            ],
            [
                "measured",
                f"prompt grew {report.measured_with_image:+g} median on the step after an image,"
                f" {report.measured_without_image:+g} without"
                if report.measured_with_image is not None
                and report.measured_without_image is not None
                else "not enough steps to measure",
            ],
            ["verdict", report.verdict],
        ],
        align="ll",
    )
    lines += [
        "",
        "how these were estimated",
        "  prompt/output totals are the provider's own per-message usage",
        "  (input + cacheRead), not an estimate.",
        "  the image share is estimated at width*height/750, with width and height",
        "  read from each PNG's IHDR header (no base64 is ever decoded past 24 bytes).",
        "  'measured' uses neither rule: it compares observed prompt growth after",
        "  steps that carried an image with growth after steps that did not.",
    ]
    return lines, report.to_dict()


# -- session ------------------------------------------------------------------


def command_session(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """A digest of one session: intent, phases, achievement, loss, ending."""

    session = ctx.session()
    try:
        record: Optional[RunRecord] = ctx.record()
    except ScopeError:
        record = None
    receipts: list[Receipt] = (
        receipts_between(record.receipts, session.started_at, session.ended_at) if record else []
    )
    window_presses = sum(receipt.presses for receipt in receipts)
    report = analysis.tool_report(session)
    loops = analysis.find_loops(session, limit=3)
    stuck = analysis.tail_repeat(session)
    ladder = ladder_progress(record) if record else LadderProgress()
    grouped = analysis.phases(session, receipts)
    visible, hidden = render.capped(grouped, None if ctx.full else 10)

    milestones: list[str] = []
    for receipt in receipts:
        milestones.extend(receipt.milestones_new)

    blocked = sum(1 for receipt in receipts if receipt.blocked)
    action_batches = sum(1 for receipt in receipts if receipt.is_action_batch)

    lines = [
        f"SESSION {session.short_id}  {render.stamp(session.started_at)}"
        f" +{render.human_seconds(session.elapsed)}"
        f"  {session.model or '?'}  thinking={session.thinking_level or '?'}",
        render.kv(
            "goal", render.truncate(session.goal or (record.meta.goal if record else ""), 68)
        ),
        render.kv(
            "volume",
            f"{len(session.steps)} steps, {report.total_calls} calls,"
            f" {sum(stat.failures for stat in report.stats)} failed",
        ),
        render.kv(
            "run",
            f"{record.run_id if record else '-'}   {render.thousands(window_presses)} presses"
            f" in this session   ladder {ladder}",
        ),
        "",
        "phases (split on a map change or a 30s stall)",
    ]
    lines += render.table(
        [
            "#",
            "where",
            "why",
            "steps",
            "calls",
            "time",
            "presses",
            "tiles",
            "B/tile",
            "top commands",
        ],
        [
            [
                str(index + 1),
                render.truncate(phase.where or "?", 15),
                phase.boundary,
                f"{phase.first_step}-{phase.last_step}",
                str(phase.calls),
                render.human_seconds(phase.seconds),
                render.thousands(phase.presses),
                str(phase.new_tiles),
                "-"
                if phase.presses_per_new_tile is None
                else render.thousands(phase.presses_per_new_tile),
                render.truncate(
                    ", ".join(f"{name} {count}" for name, count in phase.top_commands), 28
                ),
            ]
            for index, phase in enumerate(visible)
        ],
        align="rllrrrrrrl",
    )
    if hidden:
        lines.append(f"  ... and {hidden} more phases (--full)")

    lines += ["", "achieved"]
    if milestones:
        for milestone in dict.fromkeys(milestones):
            lines.append(f"  {milestone}")
    else:
        lines.append("  no new milestone in this session")

    lines += ["", "lost time"]
    if loops:
        top = loops[0]
        lines.append(
            f"  loop     {render.truncate(_sequence(top.tokens), 44)} x{top.count}"
            f" = {top.covered} calls"
            f" ({render.pct(top.covered / report.total_calls) if report.total_calls else '0%'})"
        )
    if action_batches:
        lines.append(
            f"  blocked  {blocked} of {action_batches} batches ended where they started"
            f" ({render.pct(blocked / action_batches)})"
        )
    worst = max(
        (phase for phase in grouped if phase.presses),
        key=lambda phase: phase.presses_per_new_tile or float("inf"),
        default=None,
    )
    if worst is not None:
        lines.append(
            f"  worst    {worst.where or '?'} (steps {worst.first_step}-{worst.last_step}):"
            f" {render.thousands(worst.presses)} presses bought {worst.new_tiles} new tiles"
            + (
                f" ({worst.presses_per_new_tile} per tile)"
                if worst.presses_per_new_tile is not None
                else ""
            )
        )

    live = False
    try:
        live = (time.time() - session.path.stat().st_mtime) < 120
    except OSError:
        live = False
    last = session.calls[-1] if session.calls else None
    lines += ["", "ended"]
    lines.append(
        f"  {'still running (transcript written in the last 2 min)' if live else 'idle'}"
        + (
            f", stuck in {_sequence(stuck.tokens)} x{stuck.count}"
            if stuck and stuck.count >= 3
            else ""
        )
    )
    if last is not None:
        lines.append(
            f"  last: {render.truncate(last.signature or last.label, 40)} -> {_outcome(last)}"
        )

    payload = {
        "session_id": session.session_id,
        "path": str(session.path),
        "started_at": session.started_at,
        "elapsed_seconds": round(session.elapsed, 1),
        "model": session.model,
        "goal": session.goal,
        "steps": len(session.steps),
        "calls": report.total_calls,
        "run_id": record.run_id if record else None,
        "presses_in_session": window_presses,
        "ladder": {"reached": ladder.reached, "total": ladder.total},
        "phases": [phase.to_dict() for phase in grouped],
        "milestones": list(dict.fromkeys(milestones)),
        "loops": [loop.to_dict() for loop in loops],
        "blocked_batches": blocked,
        "action_batches": action_batches,
        "live": live,
    }
    return lines, payload


# -- timeline -----------------------------------------------------------------


def command_timeline(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """Milestones with cumulative presses and the wall clock between them."""

    record = ctx.record()
    metrics = run_metrics(record)
    ladder = ladder_progress(record)
    rows = analysis.timeline_rows(record)
    visible, hidden = render.capped(rows, None if ctx.full else 30)

    lines = [
        f"TIMELINE  run {record.run_id}  {metrics.status}"
        f"  {render.thousands(metrics.total_presses)} presses"
        f"  ladder {ladder}  {render.human_seconds(metrics.wall_clock_seconds)}",
        "",
    ]
    if ladder.baseline:
        lines.append(f"  start: {ladder.baseline} milestones inherited from the checkpoint")
    if not visible:
        lines.append("  no milestone reached yet in this run")
    else:
        lines += render.table(
            ["presses", "+presses", "at", "+time", "milestone"],
            [
                [
                    render.thousands(row.presses),
                    render.thousands(row.delta_presses),
                    render.human_seconds(row.seconds),
                    render.human_seconds(row.delta_seconds),
                    render.truncate(row.label, 44),
                ]
                for row in visible
            ],
            align="rrrrl",
        )
    if hidden:
        lines.append(f"  ... and {hidden} more (--full)")
    lines += [
        "",
        f"presses are cumulative from the first receipt and never reset on a reload"
        f" ({metrics.reloads} reloads, {metrics.whiteouts} whiteouts so far).",
    ]
    payload = {
        "run_id": record.run_id,
        "status": metrics.status,
        "total_presses": metrics.total_presses,
        "ladder": {"reached": ladder.reached, "total": ladder.total, "baseline": ladder.baseline},
        "wall_clock_seconds": metrics.wall_clock_seconds,
        "milestones": [row.to_dict() for row in rows],
    }
    return lines, payload


# -- diff ---------------------------------------------------------------------


def command_diff(ctx: Context, run_a: str, run_b: str) -> tuple[list[str], dict[str, Any]]:
    """The same metrics side by side, for an A/B test."""

    left = ctx.record(run_a)
    right = ctx.record(run_b)
    rows = analysis.diff_rows(left, right)
    milestones = analysis.diff_milestones(left, right)
    visible, hidden = render.capped(milestones, None if ctx.full else 20)

    def delta(row: analysis.DiffRow) -> str:
        if not isinstance(row.left, (int, float)) or not isinstance(row.right, (int, float)):
            return ""
        if isinstance(row.left, bool) or isinstance(row.right, bool):
            return ""
        difference = row.right - row.left
        if not difference:
            return "same"
        mark = ""
        if row.direction == "lower":
            mark = " better" if difference < 0 else " worse"
        elif row.direction == "higher":
            mark = " better" if difference > 0 else " worse"
        return f"{difference:+,.4g}{mark}"

    lines = [
        f"DIFF  a={left.run_id}  b={right.run_id}",
        "",
    ]
    lines += render.table(
        ["metric", "a", "b", "b-a"],
        [
            [
                row.metric,
                render.thousands(row.left) if row.left is not None else "-",
                render.thousands(row.right) if row.right is not None else "-",
                delta(row),
            ]
            for row in rows
        ],
        align="lrrl",
    )
    lines += ["", "presses to each milestone"]
    if not visible:
        lines.append("  neither run reached a milestone")
    else:
        lines += render.table(
            ["milestone", "a", "b", "b-a"],
            [
                [
                    render.truncate(label, 40),
                    render.thousands(left_presses) if left_presses is not None else "-",
                    render.thousands(right_presses) if right_presses is not None else "-",
                    f"{right_presses - left_presses:+,}"
                    if left_presses is not None and right_presses is not None
                    else "",
                ]
                for label, left_presses, right_presses in visible
            ],
            align="lrrr",
        )
    if hidden:
        lines.append(f"  ... and {hidden} more (--full)")

    payload = {
        "a": left.run_id,
        "b": right.run_id,
        "metrics": [row.to_dict() for row in rows],
        "presses_to": [
            {"milestone": label, "a": left_presses, "b": right_presses}
            for label, left_presses, right_presses in milestones
        ],
    }
    return lines, payload


# -- runs / sessions listing --------------------------------------------------


def command_where(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """What ``scope`` found, and what it would read by default."""

    paths = ctx.paths
    sessions = list_sessions(paths.session_dir)
    runs = RunRegistry(paths.data_dir).list_runs() if paths.data_dir else ()
    lines = [
        "WHERE",
        render.kv("workspace", f"{paths.workspace or '-'}   ({paths.workspace_source})", 11),
        render.kv("data-dir", f"{paths.data_dir or '-'}   ({paths.data_dir_source})", 11),
        "",
        f"sessions ({len(sessions)})",
    ]
    visible, hidden = render.capped(sessions, None if ctx.full else 10)
    lines += render.table(
        ["file", "kB"],
        [
            [
                path.name,
                str(path.stat().st_size // 1024) if path.exists() else "?",
            ]
            for path in visible
        ],
        align="lr",
    )
    if hidden:
        lines.append(f"  ... and {hidden} more (--full)")
    lines += ["", f"runs ({len(runs)})"]
    lines += render.table(
        ["run_id", "status", "receipts", "model"],
        [
            [summary.run_id, summary.status, str(summary.receipt_count), summary.model or "?"]
            for summary in runs[-10:]
        ],
        align="llrl",
    )
    payload = {
        "workspace": str(paths.workspace) if paths.workspace else None,
        "workspace_source": paths.workspace_source,
        "data_dir": str(paths.data_dir) if paths.data_dir else None,
        "data_dir_source": paths.data_dir_source,
        "sessions": [str(path) for path in sessions],
        "runs": [summary.run_id for summary in runs],
    }
    return lines, payload
