"""Turning the measurements into the fifty lines the supervisor actually reads.

Formatting rules, all of them consequences of the reader being a model with a
budget rather than a person with a scrollbar: no ANSI, no colour, no base64,
fixed-width columns so a table survives being quoted, absolute numbers next to
percentages so neither has to be recomputed, and a hard cap on rows with the
overflow collapsed into a single "and N more" line that ``--full`` lifts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

_SPARK = "▁▂▃▄▅▆▇█"


def thousands(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def human_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def clock(when: Optional[float]) -> str:
    if not when:
        return "-"
    return datetime.fromtimestamp(when).strftime("%H:%M:%S")


def stamp(when: Optional[float]) -> str:
    if not when:
        return "-"
    return datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M")


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def si_bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} kB"
    return f"{count / (1024 * 1024):.1f} MB"


def truncate(text: str, width: int) -> str:
    single = " ".join(str(text).split())
    return single if len(single) <= width else single[: max(0, width - 1)] + "~"


def sparkline(values: Sequence[float], width: int = 48) -> str:
    """A curve in one line of text. Buckets take the maximum, not the mean, so
    a spike is never smoothed away."""

    numbers = [float(value) for value in values]
    if not numbers:
        return ""
    if len(numbers) > width:
        bucket = len(numbers) / width
        numbers = [
            max(
                numbers[
                    int(index * bucket) : max(int((index + 1) * bucket), int(index * bucket) + 1)
                ]
            )
            for index in range(width)
        ]
    low, high = min(numbers), max(numbers)
    span = high - low
    if span <= 0:
        return _SPARK[0] * len(numbers)
    return "".join(
        _SPARK[min(len(_SPARK) - 1, int((n - low) / span * len(_SPARK)))] for n in numbers
    )


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    align: str = "",
    indent: str = "  ",
) -> list[str]:
    """A fixed-width table. ``align`` is one character per column: ``l`` or ``r``."""

    columns = len(headers)
    alignment = (align + "l" * columns)[:columns]
    cells = [[str(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in cells:
        for index, value in enumerate(row[:columns]):
            widths[index] = max(widths[index], len(value))

    def render(row: Sequence[str]) -> str:
        parts = []
        for index in range(columns):
            value = row[index] if index < len(row) else ""
            parts.append(
                value.rjust(widths[index])
                if alignment[index] == "r"
                else value.ljust(widths[index])
            )
        return (indent + "  ".join(parts)).rstrip()

    body = [render(row) for row in cells]
    if not any(str(header).strip() for header in headers):
        return body  # An unlabelled table would only spend a line on blanks.
    return [render([str(header) for header in headers])] + body


def capped(rows: Sequence[Any], limit: Optional[int]) -> tuple[list[Any], int]:
    """``(visible, hidden)`` — the overflow is reported, never silently dropped."""

    if limit is None or len(rows) <= limit:
        return list(rows), 0
    return list(rows[:limit]), len(rows) - limit


def kv(label: str, value: Any, width: int = 9) -> str:
    return f"{label.ljust(width)} {value}"
