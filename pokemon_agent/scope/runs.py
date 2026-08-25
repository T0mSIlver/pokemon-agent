"""The run side of the data: receipts, the ladder, and the harness event log.

Presses come from ``receipts.jsonl`` and nowhere else. The registry already
knows how to read that file while it is being appended to, and
:func:`pokemon_agent.bench.metrics.compute` already knows that a reload rewinds
the game and not the bill, so both are reused rather than reimplemented here.

What this module adds is the two things the receipt does not carry: which rung
of the 63-milestone ladder the run is on, and whether a given batch of presses
was spent in a dialog or a battle. The second comes from ``run_log.jsonl``,
whose ``action_result`` records hold the full post-action state and land on the
same clock as the receipts.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from pokemon_agent.bench.metrics import RunMetrics, compute
from pokemon_agent.bench.registry import Receipt, RunRecord, RunRegistry
from pokemon_agent.scope.transcript import parse_timestamp

CURRENT_FILENAME = "CURRENT"

#: How far a receipt and an ``action_result`` may be apart and still describe
#: the same moment. Measured on a live run: 430 of 431 receipts sat within
#: 0.7 ms of their log record, while consecutive receipts were never closer
#: than 0.18 s apart. A tenth of a second is therefore two orders of magnitude
#: of slack on the match and still cannot reach the neighbouring batch.
_MATCH_TOLERANCE_SECONDS = 0.1


def ladder_ids() -> tuple[str, ...]:
    """Every milestone on the curated ladder, in order. ``()`` if unavailable."""

    try:
        from pokemon_agent import milestones as milestones_module
    except Exception:  # noqa: BLE001 — a missing ladder must not stop a report
        return ()
    out: list[str] = []
    for item in getattr(milestones_module, "MILESTONES", ()) or ():
        identifier = getattr(item, "id", None)
        if isinstance(identifier, str) and identifier:
            out.append(identifier)
    return tuple(out)


def current_run_id(data_dir: Optional[Path]) -> Optional[str]:
    """The run the server has open, per ``runs/CURRENT``."""

    if data_dir is None:
        return None
    marker = data_dir / "runs" / CURRENT_FILENAME
    try:
        name = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def resolve_run_id(data_dir: Optional[Path], wanted: Optional[str]) -> Optional[str]:
    """``wanted``, else the open run, else the most recently started one."""

    if data_dir is None:
        return None
    if wanted:
        return wanted
    open_run = current_run_id(data_dir)
    if open_run and (data_dir / "runs" / open_run / "meta.json").is_file():
        return open_run
    summaries = RunRegistry(data_dir).list_runs()
    return summaries[-1].run_id if summaries else None


def load_run(data_dir: Path, run_id: str) -> RunRecord:
    return RunRegistry(data_dir).load(run_id)


@dataclass(frozen=True)
class LadderProgress:
    """``reached`` of ``total`` rungs of the curated ladder."""

    reached: int = 0
    total: int = 0
    #: Rungs the run inherited from its start checkpoint rather than earning.
    baseline: int = 0

    def __str__(self) -> str:
        return f"{self.reached}/{self.total}" if self.total else str(self.reached)


def ladder_progress(record: RunRecord) -> LadderProgress:
    """How far up the 63-rung ladder the run has got, checkpoint included.

    A run started from a save inherits whatever that save had already done; the
    first receipt records it as ``baseline_milestones``. Counting only what the
    run earned would understate where the game actually is, which is the thing
    the supervisor is looking at.
    """

    rungs = ladder_ids()
    if not rungs:
        return LadderProgress()
    on_ladder = set(rungs)
    baseline: set[str] = set()
    earned: set[str] = set()
    for receipt in record.receipts:
        raw_baseline = receipt.extra.get("baseline_milestones")
        if isinstance(raw_baseline, (list, tuple)):
            baseline.update(str(item) for item in raw_baseline)
        earned.update(receipt.milestones_new)
    reached = (baseline | earned) & on_ladder
    return LadderProgress(
        reached=len(reached),
        total=len(rungs),
        baseline=len(baseline & on_ladder),
    )


def run_metrics(record: RunRecord) -> RunMetrics:
    return compute(record)


# -- the harness event log ----------------------------------------------------


@dataclass(frozen=True)
class ActionContext:
    """The game state right after one batch of presses landed."""

    at: float
    dialog: bool = False
    battle: bool = False
    map_name: str = ""


def read_action_contexts(run_log: Optional[Path]) -> list[ActionContext]:
    """Every ``action_result`` in the run log, oldest first.

    The log is a couple of megabytes and ninety per cent of it is state dumps
    for records this does not want, so lines are filtered on a substring before
    anything is handed to the JSON parser. That turns a full parse of the file
    into a scan plus a couple of hundred small ones.
    """

    if run_log is None:
        return []
    try:
        raw = run_log.read_bytes()
    except OSError:
        return []
    marker = b'"action_result"'
    out: list[ActionContext] = []
    for line in raw.split(b"\n"):
        if marker not in line:
            continue
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("type") != "action_result":
            continue
        at = parse_timestamp(payload.get("timestamp"))
        if at is None:
            continue
        state = payload.get("state_after")
        state = state if isinstance(state, dict) else {}
        dialog = state.get("dialog") if isinstance(state.get("dialog"), dict) else {}
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        map_block = state.get("map") if isinstance(state.get("map"), dict) else {}
        out.append(
            ActionContext(
                at=at,
                dialog=bool(dialog.get("active")),
                battle=bool(battle.get("in_battle")),
                map_name=str(map_block.get("map_name") or ""),
            )
        )
    out.sort(key=lambda item: item.at)
    return out


class ContextOracle:
    """Answers "was the game in a dialog or a battle at time *t*?".

    Matching is nearest-in-time within a few seconds, because the receipt and
    the log record for one action are written in the same breath. Outside that
    window the honest answer is "no idea", which reads as neither.
    """

    def __init__(self, contexts: Iterable[ActionContext]) -> None:
        self._contexts = sorted(contexts, key=lambda item: item.at)
        self._times = [item.at for item in self._contexts]

    def __len__(self) -> int:
        return len(self._contexts)

    def at(self, when: float) -> Optional[ActionContext]:
        if not self._times or not when:
            return None
        index = bisect.bisect_left(self._times, when)
        best: Optional[ActionContext] = None
        best_gap = _MATCH_TOLERANCE_SECONDS
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(self._contexts):
                gap = abs(self._times[candidate] - when)
                if gap <= best_gap:
                    best, best_gap = self._contexts[candidate], gap
        return best


def read_json(path: Optional[Path]) -> dict[str, Any]:
    """A JSON file that may not exist, may be half written, or may be huge."""

    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def receipts_between(
    receipts: Iterable[Receipt], start: Optional[float], end: Optional[float]
) -> list[Receipt]:
    """The receipts written inside a session's wall-clock window."""

    low = start if start is not None else float("-inf")
    high = end if end is not None else float("inf")
    return [receipt for receipt in receipts if receipt.t and low <= receipt.t <= high]
