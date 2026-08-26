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
from pokemon_agent.scope import analysis, beliefs, progression, render, truth
from pokemon_agent.scope.discover import Paths, list_sessions, resolve_session
from pokemon_agent.scope.runs import (
    ContextOracle,
    LadderProgress,
    ladder_progress,
    read_action_contexts,
    read_events,
    read_json,
    receipts_between,
    resolve_run_id,
    run_metrics,
)
from pokemon_agent.scope.transcript import Call, Session, parse_session, parse_timestamp

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
    #: Presses each side of a split or an intervention gets measured over.
    span: int = progression.DEFAULT_INTERVENTION_SPAN
    #: Where to cut the run for ``split``, e.g. ``press:7000`` or ``save:foo``.
    at: Optional[str] = None

    _session: Optional[Session] = None
    _sessions: Optional[list[Session]] = None
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

    def run_sessions(self) -> list[Session]:
        """Every transcript written while the current run was being played.

        A workspace outlives a run — this one holds transcripts from two — so a
        report about *this* run has to drop the ones whose steps never touched
        it. Filtering on the run's own first and last receipt is the only clock
        both halves of the data share. ``--session`` narrows it to one.
        """

        if self._sessions is not None:
            return self._sessions
        paths = list_sessions(self.paths.session_dir)
        if self.session_id:
            chosen = resolve_session(self.paths.session_dir, self.session_id)
            paths = [chosen] if chosen is not None else []
        sessions = [parse_session(path) for path in paths]
        try:
            low, high = progression.run_window(self.record())
        except ScopeError:
            low, high = float("-inf"), float("inf")
        self._sessions = [session for session in sessions if _overlaps(session, low, high)]
        return self._sessions

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


def _overlaps(session: Session, low: float, high: float) -> bool:
    """Did any part of this transcript happen while the run was being played?"""

    started = session.started_at
    ended = session.ended_at if session.ended_at is not None else started
    if started is None or ended is None:
        return True
    return started <= high and ended >= low


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


# -- claims -------------------------------------------------------------------


def command_claims(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """What the model said was true, checked against the game's own data."""

    sessions = ctx.run_sessions()
    if not sessions:
        raise ScopeError("no transcripts found for this run")
    report = beliefs.claims_report(sessions, limit=30 if ctx.full else 10)

    narration = _share(report.narrating_calls, report.total_calls)
    lines = [
        f"CLAIMS  run {ctx.record().run_id}  {len(sessions)} sessions"
        f"  {report.checked} claims checked  {report.wrong} false",
        "",
        f"narration  {render.thousands(report.narrating_calls)}"
        f" of {render.thousands(report.total_calls)} bash calls ({narration})"
        " carry a # comment; that is the only place this model states a belief",
        "",
    ]
    if not report.tallies:
        lines.append("  nothing checkable was said. No coordinates, no map names.")
    else:
        lines += render.table(
            ["claim", "checked", "true", "false", "wrong", "unchecked"],
            [
                [
                    tally.kind,
                    str(tally.checked),
                    str(tally.true),
                    str(tally.false),
                    render.pct(tally.wrong_share),
                    str(tally.unchecked),
                ]
                for tally in report.tallies
            ],
            align="lrrrrr",
        )
    if report.reasons:
        lines += ["", "why the false ones were false"]
        lines += render.table(
            ["", ""],
            [[why, str(count)] for why, count in report.reasons],
            align="lr",
        )
    if report.offsets:
        repeated = [(offset, count) for offset, count in report.offsets if count > 1]
        visible, hidden = render.capped(repeated or report.offsets, 6)
        lines += ["", "how far off, when it was off (claimed minus actual)"]
        lines += render.table(
            ["", ""],
            [[f"({offset[0]:+d},{offset[1]:+d})", str(count)] for offset, count in visible],
            align="lr",
        )
        if hidden:
            lines.append(f"  ... and {hidden} more distinct offsets")
    worst, hidden = render.capped(report.worst, ctx.row_limit)
    if worst:
        lines += ["", "the biggest misses"]
        lines += render.table(
            ["step", "map", "said", "claimed", "actual", "why"],
            [
                [
                    str(claim.step),
                    render.truncate(claim.map_name, 18),
                    render.truncate(claim.said, 34),
                    str(claim.claimed),
                    str(claim.actual),
                    render.truncate(claim.why, 24),
                ]
                for claim in worst
            ],
            align="rlllll",
        )
        if hidden:
            lines.append(f"  ... and {hidden} more (--full)")
    lines += [
        "",
        "a claim is only counted when the check is exact; ambiguous sentences are",
        "unchecked, never guessed at. position = where it said it was standing before",
        "the call, destination = where it said the call would end, both settled by the",
        f"./poke reply. warp and map are settled against gamedata ({truth.known_maps()} maps).",
    ]
    payload = report.to_dict()
    payload["run_id"] = ctx.record().run_id
    return lines, payload


# -- episodes -----------------------------------------------------------------


def command_episodes(ctx: Context, map_name: Optional[str] = None) -> tuple[list[str], dict]:
    """Presses per stay on a map, and the doors out that were never taken."""

    record = ctx.record()
    report = progression.episode_report(record)
    if map_name:
        return _episode_detail(ctx, report, map_name)

    total = sum(stay.presses for stay in report.stays)
    lines = [
        f"EPISODES  run {record.run_id}  {len(report.episodes)} stays"
        f"  {len(report.stays)} maps  {render.thousands(total)} presses",
        "",
    ]
    stays, hidden = render.capped(report.stays, ctx.row_limit)
    lines += render.table(
        ["map", "stays", "presses", "med", "tiles", "of map", "new/100p"],
        [
            [
                stay.map_name,
                str(stay.episodes),
                render.thousands(stay.presses),
                render.thousands(stay.median_presses),
                str(stay.tiles),
                _share(stay.tiles, stay.map_tiles) if stay.map_tiles else "-",
                f"{stay.yield_per_100:.1f}",
            ]
            for stay in stays
        ],
        align="lrrrrrr",
    )
    if hidden:
        lines.append(f"  ... and {hidden} more maps (--full)")

    pairs, pairs_hidden = render.capped(report.transitions, 8 if not ctx.full else None)
    if pairs:
        lines += ["", "map changes, most travelled first"]
        lines += render.table(
            ["", "", ""],
            [[pair[0], "->  " + pair[1], f"x{count}"] for pair, count in pairs],
            align="llr",
        )
        if pairs_hidden:
            lines.append(f"  ... and {pairs_hidden} more edges (--full)")

    untried, untried_hidden = render.capped(report.untried, ctx.row_limit)
    lines += ["", "exits the game data lists that this run never took"]
    if not untried:
        lines.append("  none: every door out of every visited map was used at least once")
    else:
        lines += render.table(
            ["", "", ""],
            [[pair[0], "->  " + pair[1], ""] for pair in untried],
            align="lll",
        )
        if untried_hidden:
            lines.append(f"  ... and {untried_hidden} more (--full)")
    lines += [
        "",
        "a stay is one unbroken run of receipts on one map. med = median presses per",
        "stay: many short stays on the same map is bouncing, one long stay is a grind.",
        "'of map' is tiles stood on over the tiles the game says the map has.",
        "scope episodes <map> lists every stay on one map.",
    ]
    return lines, report.to_dict()


def _episode_detail(
    ctx: Context, report: progression.EpisodeReport, map_name: str
) -> tuple[list[str], dict[str, Any]]:
    """Every stay on one map, in order."""

    wanted = map_name.lower()
    names = sorted({episode.map_name for episode in report.episodes})
    matches = [name for name in names if wanted in name.lower()]
    if not matches:
        raise ScopeError(
            f"the run never stood on a map matching {map_name!r}; it visited: " + ", ".join(names)
        )
    chosen = matches[0]
    episodes = [episode for episode in report.episodes if episode.map_name == chosen]
    stay = next((item for item in report.stays if item.map_name == chosen), None)

    lines = [
        f"EPISODES  {chosen}  {len(episodes)} stays"
        f"  {render.thousands(stay.presses if stay else 0)} presses"
        f"  {stay.tiles if stay else 0} tiles"
        + (f" of {stay.map_tiles}" if stay and stay.map_tiles else ""),
        "",
    ]
    visible, hidden = render.capped(episodes, None if ctx.full else 20)
    lines += render.table(
        ["#", "seq", "presses", "entry", "exit", "new", "blocked", "left for"],
        [
            [
                str(episode.index),
                f"{episode.first_seq}-{episode.last_seq}",
                render.thousands(episode.presses),
                str(episode.entry or "-"),
                str(episode.exit or "-"),
                str(episode.new_tiles),
                str(episode.blocked_batches),
                render.truncate(episode.went_to or "(still here)", 20),
            ]
            for episode in visible
        ],
        align="rrrlllrl",
    )
    if hidden:
        lines.append(f"  ... and {hidden} more stays (--full)")

    known = truth.map_truth(chosen)
    if known is not None and known.warps:
        lines += ["", f"warps the game data puts on {chosen}"]
        taken = {pair[1] for pair, _ in report.transitions if pair[0] == chosen}
        lines += render.table(
            ["at", "leads to", "went there"],
            [
                [str(warp.at), warp.to_map, "yes" if warp.to_map in taken else "NEVER"]
                for warp in known.warps
            ],
            align="lll",
        )
        lines.append(
            "  'went there' is whether the run ever made that map-to-map move, not"
            " whether it used that exact tile: a receipt records where a warp put the"
        )
        lines.append(
            "  player, never the tile it stepped from. NEVER is therefore certain; yes"
            " may have gone through a different door to the same place."
        )
    barren = [episode for episode in episodes if episode.new_tiles == 0]
    lines += [
        "",
        f"{len(barren)} of {len(episodes)} stays found no ground the run had not already"
        f" stood on ({_share(sum(e.presses for e in barren), stay.presses if stay else 0)}"
        " of this map's presses).",
    ]
    payload = {
        "map": chosen,
        "episodes": [episode.to_dict() for episode in episodes],
        "stay": stay.to_dict() if stay else None,
        "barren_episodes": len(barren),
    }
    return lines, payload


# -- split --------------------------------------------------------------------

#: How a marker names a moment. ``press:`` and ``seq:`` are positions in the
#: run; the rest are events that can be named after the fact, which is the point
#: — a change shipped at 03:14 is remembered as "when interventions went on".
MARKER_HELP = "press:N  seq:N  session:N  map:NAME  save:NAME  intervention:N  time:HH:MM"

#: Batches on the thinner side below which a split is noise. A healthy 400-press
#: window on this run holds 100-130 batches, so a side in the low tens is a
#: handful of mega-batches and a percentage change across it means nothing.
MIN_TRUSTWORTHY_BATCHES = 50


def _marker_time(ctx: Context, marker: str) -> float:
    """Epoch seconds for a marker, or a ScopeError naming what was available."""

    kind, _, value = marker.partition(":")
    kind = kind.strip().lower()
    value = value.strip()
    if not value:
        raise ScopeError(f"a marker needs a value: {MARKER_HELP}")
    record = ctx.record()
    receipts = [receipt for receipt in record.receipts if receipt.t]

    if kind == "press":
        wanted = _as_int(value, marker)
        spent = 0
        for receipt in receipts:
            spent += receipt.presses
            if spent >= wanted:
                return receipt.t
        raise ScopeError(f"the run has only spent {spent} presses, fewer than {wanted}")
    if kind == "seq":
        wanted = _as_int(value, marker)
        for receipt in receipts:
            if receipt.seq >= wanted:
                return receipt.t
        raise ScopeError(f"no receipt with seq >= {wanted}")
    if kind == "session":
        wanted = _as_int(value, marker)
        for receipt in receipts:
            if receipt.extra.get("session") == wanted:
                return receipt.t
        seen = sorted({r.extra.get("session") for r in receipts if r.extra.get("session")})
        raise ScopeError(f"no receipts from session {wanted}; the run has sessions {seen}")
    if kind == "map":
        for receipt in receipts:
            if receipt.map_name.lower() == value.lower():
                return receipt.t
        names = sorted({receipt.map_name for receipt in receipts if receipt.map_name})
        raise ScopeError(f"the run never stood on {value!r}; it visited: " + ", ".join(names))
    if kind == "save":
        saves = read_events(ctx.paths.run_log, "save")
        for at, payload in saves:
            if str(payload.get("name") or "") == value:
                return at
        names = sorted({str(payload.get("name") or "") for _, payload in saves})
        raise ScopeError(
            f"no save called {value!r} in the run log; it holds {len(names)} names"
            + (": " + ", ".join(names[:12]) if names else "")
        )
    if kind == "intervention":
        wanted = _as_int(value, marker)
        found = progression.find_injections(
            ctx.run_sessions(), between=progression.run_window(record)
        )
        if not 1 <= wanted <= len(found):
            raise ScopeError(f"this run has {len(found)} interventions, not a #{wanted}")
        return found[wanted - 1][1].at or 0.0
    if kind == "time":
        return _clock_time(value, receipts)
    raise ScopeError(f"unknown marker {marker!r}. Use one of: {MARKER_HELP}")


def _as_int(value: str, marker: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ScopeError(f"{marker!r} needs a whole number") from exc


def _clock_time(value: str, receipts: list[Receipt]) -> float:
    """``time:14:05`` is today on the run's clock; a full ISO stamp is absolute."""

    if len(value) <= 5 and ":" in value:
        anchor = receipts[0].t if receipts else time.time()
        day = time.localtime(anchor)
        hour, _, minute = value.partition(":")
        try:
            stamp = time.struct_time(
                (day.tm_year, day.tm_mon, day.tm_mday, int(hour), int(minute), 0, 0, 0, -1)
            )
        except ValueError as exc:
            raise ScopeError(f"{value!r} is not a HH:MM time") from exc
        return time.mktime(stamp)
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ScopeError(f"{value!r} is not a time this can read")
    return parsed


def command_split(ctx: Context, marker: Optional[str] = None) -> tuple[list[str], dict[str, Any]]:
    """The same run, before and after one moment in it."""

    marker = marker or ctx.at
    if not marker:
        raise ScopeError(f"split needs a marker: scope split <marker>.  {MARKER_HELP}")
    at = _marker_time(ctx, marker)
    record = ctx.record()
    span = None if ctx.full else ctx.span
    report = progression.split_report(record, marker=marker, at=at, span=span)

    scope_note = (
        f"the {render.thousands(span)} presses either side" if span else "the whole run either side"
    )
    lines = [
        f"SPLIT  run {record.run_id}  at {marker}  ({render.stamp(at)})",
        f"       comparing {scope_note} of that moment",
        "",
    ]
    rows = []
    for name, heading, low, high, change in report.deltas():
        rows.append(
            [
                heading,
                _split_value(name, low),
                _split_value(name, high),
                "-" if change is None else f"{change:+.0%}",
            ]
        )
    lines += render.table(["metric", "before", "after", "change"], rows, align="lrrr")
    lines += [
        "",
        f"before  batches {report.before.first_seq}-{report.before.last_seq}",
        f"after   batches {report.after.first_seq}-{report.after.last_seq}",
    ]
    thin = min(report.before.batches, report.after.batches)
    if thin < MIN_TRUSTWORTHY_BATCHES:
        lines.append(
            f"  CAUTION: the thinner side is {thin} batches. Treat every change above as noise."
        )
    lines += [
        "",
        "new/1k press is tiles stood on for the first time per thousand presses, which is",
        "the only output measure that does not need a milestone to move. --full drops the",
        "press cap and compares the entire run either side instead.",
    ]
    return lines, report.to_dict()


def _split_value(name: str, value: float) -> str:
    if name in progression.SHARE_FIELDS:
        return render.pct(value)
    if float(value).is_integer():
        return render.thousands(int(value))
    return f"{value:,.1f}"


# -- intervene ----------------------------------------------------------------


def command_intervene(ctx: Context) -> tuple[list[str], dict[str, Any]]:
    """Every intervention: what fired it, what it asked for, what changed."""

    record = ctx.record()
    report = progression.intervention_report(record, ctx.run_sessions(), span=ctx.span)

    lines = [
        f"INTERVENE  run {record.run_id}  {len(report.events)} delivered"
        f"  measured over {render.thousands(report.span)} presses each side",
        "",
    ]
    if not report.events:
        lines.append("  no advice was pushed into any session of this run.")
    else:
        visible, hidden = render.capped(report.events, ctx.row_limit)
        lines += render.table(
            ["#", "at", "trigger", "asked", "did it", "hp", "new/1k", "advice"],
            [
                [
                    str(event.index),
                    render.clock(event.at),
                    event.trigger or "?",
                    event.asked_for or "-",
                    _followed_word(event.followed),
                    f"{event.hp_at:.0%}->{event.hp_best_after:.0%}",
                    f"{event.before.new_per_1k:.0f}->{event.after.new_per_1k:.0f}",
                    render.truncate(event.headline, 30),
                ]
                for event in visible
            ],
            align="rlllllll",
        )
        if hidden:
            lines.append(f"  ... and {hidden} more (--full)")

        followed = [event for event in report.events if event.followed is True]
        asked_heal = [event for event in report.events if "heal" in event.headline.lower()]
        healed = [event for event in asked_heal if event.healed]
        better = [
            event for event in report.events if event.after.new_per_1k > event.before.new_per_1k
        ]
        lines += [
            "",
            f"followed   {len(followed)} of {len(report.events)} moved in the direction asked"
            " within the next 6 steps",
            f"healed     {len(healed)} of {len(asked_heal)} that said 'heal' had the party"
            f" back above its starting HP within {render.thousands(report.span)} presses",
            f"new ground {len(better)} of {len(report.events)} covered more fresh tiles"
            " after than before",
            f"milestones {sum(event.after.milestones for event in report.events)} reached in"
            " any of the after-windows",
        ]

    if report.standing:
        lines += [
            "",
            f"how much of the run each detector was already firing on"
            f" ({report.samples} sampled windows)",
        ]
        lines += render.table(
            ["detector", "windows", "share"],
            [[name, str(count), _share(count, report.samples)] for name, count in report.standing],
            align="lrr",
        )
        lines.append("  a detector standing far above the number of interventions delivered is a")
        lines.append("  signal the budget is throwing away, not a signal that is missing.")
    lines += [
        "",
        "the trigger column is replayed, not recorded: the harness's own detectors from",
        "pokemon_agent.interventions are re-run over the receipts up to that moment, so it",
        "is the same rule that fired, evaluated on the same data.",
    ]
    return lines, report.to_dict()


def _followed_word(followed: Optional[bool]) -> str:
    if followed is None:
        return "-"
    return "yes" if followed else "NO"


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
