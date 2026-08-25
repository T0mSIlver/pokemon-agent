"""Plain-text tables for a terminal. No colour codes, no unicode box drawing.

``format_run`` is the scoreboard for one run. ``format_comparison`` is the one
that decides things: runs side by side, one row per milestone, each cell priced
in cumulative button presses with the gap to the best run in the row, so a
harness change is accepted or rejected on a number instead of an impression.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pokemon_agent.bench.metrics import REFERENCE_POINTS, Attainment, RunMetrics

#: How many maps the per-map revisit table shows before it stops.
MAX_MAP_ROWS = 12

EMPTY_CELL = "-"


# ---------------------------------------------------------------------------
# Formatting primitives
# ---------------------------------------------------------------------------


def _count(value: Optional[int]) -> str:
    return EMPTY_CELL if value is None else f"{value:,}"


def _percent(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _ratio(value: float) -> str:
    return f"{value:.2f}x"


def format_duration(seconds: float) -> str:
    """``3h 12m 04s`` — a wall clock a human reads at a glance."""

    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Optional[Sequence[str]] = None,
    gutter: str = "  ",
) -> list[str]:
    """Aligned columns with a dashed rule under the header row.

    ``aligns`` is one of ``"l"``/``"r"`` per column; missing entries are left.
    """

    columns = len(headers)
    alignments = list(aligns or [])
    alignments += ["l"] * (columns - len(alignments))
    widths = [len(str(header)) for header in headers]
    normalized: list[list[str]] = []
    for row in rows:
        cells = [str(cell) for cell in row][:columns]
        cells += [""] * (columns - len(cells))
        normalized.append(cells)
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.rjust(widths[index]) if alignments[index] == "r" else cell.ljust(widths[index])
            for index, cell in enumerate(cells)
        ]
        return gutter.join(parts).rstrip()

    out = [line([str(header) for header in headers]), gutter.join("-" * width for width in widths)]
    if not normalized:
        out.append("(none)")
        return out
    out.extend(line(cells) for cells in normalized)
    return out


def _fields(pairs: Sequence[tuple[str, str]]) -> list[str]:
    """``label : value`` block with the colons lined up."""

    if not pairs:
        return []
    width = max(len(label) for label, _ in pairs)
    return [f"{label.ljust(width)} : {value}" for label, value in pairs]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _milestone_rows(attainments: Sequence[Attainment]) -> list[list[str]]:
    rows: list[list[str]] = []
    previous = 0
    for position, item in enumerate(attainments, start=1):
        elapsed = format_duration(item.seconds) if item.seconds is not None else EMPTY_CELL
        rows.append(
            [
                str(position),
                item.label,
                _count(item.presses),
                f"+{item.presses - previous:,}",
                str(item.seq),
                elapsed,
            ]
        )
        previous = item.presses
    return rows


def format_run(metrics: RunMetrics) -> str:
    """The scoreboard for one run, top to bottom."""

    status = metrics.status or "unknown"
    header = [
        f"RUN {metrics.run_id or '(unnamed)'}",
        "=" * (len(metrics.run_id or "(unnamed)") + 4),
    ]

    milestone_note = f"{metrics.milestones_reached} reached" + (
        f", furthest {metrics.furthest_label}" if metrics.furthest_label else ""
    )
    summary = _fields(
        [
            ("model", metrics.model or EMPTY_CELL),
            ("status", status),
            ("goal", metrics.goal or EMPTY_CELL),
            ("start checkpoint", metrics.start_checkpoint or EMPTY_CELL),
            ("harness sha", metrics.harness_sha or EMPTY_CELL),
            ("config hash", metrics.config_hash or EMPTY_CELL),
            ("wall clock", format_duration(metrics.wall_clock_seconds)),
            ("total presses", _count(metrics.total_presses)),
            ("milestones", milestone_note),
            (
                "presses/milestone",
                f"{metrics.presses_per_milestone:,.1f}"
                if metrics.presses_per_milestone is not None
                else EMPTY_CELL,
            ),
            (
                "blocked rate",
                f"{_percent(metrics.blocked_rate)} "
                f"({metrics.blocked_batches:,} of {metrics.action_batches:,} button batches)",
            ),
            (
                "tool error rate",
                f"{_percent(metrics.tool_error_rate)} "
                f"({metrics.tool_errors:,} of {metrics.tool_calls:,} calls)",
            ),
            (
                "revisit ratio",
                f"{_ratio(metrics.revisit_ratio)} "
                f"({metrics.position_samples:,} samples over "
                f"{metrics.unique_positions:,} tiles)",
            ),
            ("whiteouts", _count(metrics.whiteouts)),
            ("reloads", f"{metrics.reloads:,} (presses are never reset by a reload)"),
            ("receipts", _count(metrics.receipts)),
        ]
    )

    milestones = render_table(
        ["#", "milestone", "presses", "since prev", "seq", "elapsed"],
        _milestone_rows(metrics.attainments),
        aligns=["r", "l", "r", "r", "r", "r"],
    )

    maps = render_table(
        ["map", "samples", "unique", "revisit"],
        [
            [entry.map_name, _count(entry.samples), _count(entry.unique), _ratio(entry.ratio)]
            for entry in metrics.revisit_by_map[:MAX_MAP_ROWS]
        ],
        aligns=["l", "r", "r", "r"],
    )

    blocks = [
        "\n".join(header),
        "\n".join(summary),
        "MILESTONES (cumulative presses at first attainment)\n" + "\n".join(milestones),
        "REVISITS BY MAP\n" + "\n".join(maps),
    ]
    if metrics.corrupt_receipt_lines:
        blocks.append(
            f"note: {metrics.corrupt_receipt_lines} unreadable receipt line(s) were skipped."
        )
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Several runs
# ---------------------------------------------------------------------------


def _comparison_order(runs: Sequence[tuple[str, RunMetrics]]) -> list[tuple[str, str]]:
    """``(milestone_id, label)`` for every milestone any run reached, in ladder order."""

    best_key: dict[str, tuple[int, int, int]] = {}
    labels: dict[str, str] = {}
    for _, metrics in runs:
        for attainment in metrics.attainments:
            key = attainment.sort_key
            known = best_key.get(attainment.milestone_id)
            if known is None or key < known:
                best_key[attainment.milestone_id] = key
            labels.setdefault(attainment.milestone_id, attainment.label)
    ordered = sorted(best_key.items(), key=lambda item: item[1])
    return [(milestone_id, labels.get(milestone_id, milestone_id)) for milestone_id, _ in ordered]


def _cell(presses: Optional[int], best: Optional[int]) -> str:
    """``649 best`` / ``1,608 +959`` / ``-`` — value plus the gap to the best run."""

    if presses is None:
        return EMPTY_CELL
    if best is None or presses == best:
        return f"{presses:,} best"
    return f"{presses:,} +{presses - best:,}"


def format_comparison(runs: Sequence[tuple[str, RunMetrics]]) -> str:
    """Runs side by side, one row per milestone, each cell against the best run."""

    if not runs:
        return "No runs to compare.\n"

    names = [str(name) for name, _ in runs]
    headers = ["milestone", *names]
    aligns = ["l", *["r"] * len(names)]

    rows: list[list[str]] = []
    for milestone_id, label in _comparison_order(runs):
        values = [metrics.presses_for(milestone_id) for _, metrics in runs]
        reached = [value for value in values if value is not None]
        best = min(reached) if reached else None
        rows.append([label, *[_cell(value, best) for value in values]])

    milestone_table = render_table(headers, rows, aligns=aligns)

    def summary_row(label: str, render) -> list[str]:
        return [label, *[render(metrics) for _, metrics in runs]]

    totals = [
        summary_row("total presses", lambda metric: _count(metric.total_presses)),
        summary_row("milestones reached", lambda metric: _count(metric.milestones_reached)),
        summary_row(
            "presses/milestone",
            lambda metric: (
                f"{metric.presses_per_milestone:,.1f}"
                if metric.presses_per_milestone is not None
                else EMPTY_CELL
            ),
        ),
        summary_row("furthest", lambda metric: metric.furthest_label or EMPTY_CELL),
        summary_row("blocked rate", lambda metric: _percent(metric.blocked_rate)),
        summary_row("tool error rate", lambda metric: _percent(metric.tool_error_rate)),
        summary_row("revisit ratio", lambda metric: _ratio(metric.revisit_ratio)),
        summary_row("whiteouts", lambda metric: _count(metric.whiteouts)),
        summary_row("reloads", lambda metric: _count(metric.reloads)),
        summary_row("wall clock", lambda metric: format_duration(metric.wall_clock_seconds)),
    ]
    totals_table = render_table(["metric", *names], totals, aligns=aligns)

    footer = "reference: " + "; ".join(
        f"{label} {presses:,} actions" for label, presses in REFERENCE_POINTS
    )

    blocks = [
        "COMPARISON (cumulative presses at first attainment; +N is the gap to the best run)",
        "\n".join(milestone_table),
        "\n".join(totals_table),
        footer,
    ]
    return "\n\n".join(blocks) + "\n"


def format_run_list(summaries: Sequence) -> str:
    """``python -m pokemon_agent.bench`` with no arguments: what is in the store."""

    import time as _time

    rows = []
    for summary in summaries:
        started = (
            _time.strftime("%Y-%m-%d %H:%M", _time.localtime(summary.started_at))
            if summary.started_at
            else EMPTY_CELL
        )
        rows.append(
            [
                summary.run_id,
                started,
                summary.status,
                _count(summary.receipt_count),
                summary.model or EMPTY_CELL,
                (summary.goal or "")[:48],
            ]
        )
    table = render_table(
        ["run id", "started", "status", "receipts", "model", "goal"],
        rows,
        aligns=["l", "l", "l", "r", "l", "l"],
    )
    return "\n".join(table) + "\n"
