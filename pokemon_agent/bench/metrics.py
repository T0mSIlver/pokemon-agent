"""Turning a run's receipts into the numbers a harness change is judged on.

The headline is ``presses_to``: the cumulative number of buttons pressed at the
moment each milestone was first reached. It is chosen to sit next to published
work — PokeAgent's best entry reached the first gym in 1,608 actions, its most
efficient in 649 — so it has to mean the same thing they mean: every button the
agent spent, from the first one of the run, counted once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from pokemon_agent.bench.registry import Receipt, RunRecord

#: Published reference points, quoted on the comparison table.
REFERENCE_POINTS: tuple[tuple[str, int], ...] = (
    ("PokeAgent best entry, first gym", 1608),
    ("PokeAgent most efficient, first gym", 649),
)


@dataclass(frozen=True)
class LadderEntry:
    """One rung of the curated milestone ladder, as this module needs it.

    ``ladder_index`` is ``None`` for a milestone the ladder does not rank — the
    contract spells that as ``-1``, and off-ladder events are still scored, just
    ordered by when they were reached.
    """

    milestone_id: str
    label: str
    ladder_index: Optional[int]


def load_ladder() -> dict[str, LadderEntry]:
    """The curated ladder from :mod:`pokemon_agent.milestones`, or ``{}``.

    That module is written by another agent and may not exist yet, may be
    half-written, or may grow fields this one has never heard of. None of that
    is allowed to stop a run from being scored: without it the metrics still
    compute, and milestones simply order by when they were first attained.
    """

    try:
        from pokemon_agent import milestones as milestones_module
    except Exception:  # noqa: BLE001 — an absent or broken ladder is not an error here
        return {}
    entries: dict[str, LadderEntry] = {}
    ladder = getattr(milestones_module, "MILESTONES", ())
    try:
        items = list(ladder)
    except TypeError:
        return {}
    for position, item in enumerate(items):
        identifier = getattr(item, "id", None)
        if not isinstance(identifier, str) or not identifier:
            continue
        index = getattr(item, "ladder_index", None)
        label = getattr(item, "label", None)
        if isinstance(index, int):
            rank = index if index >= 0 else None
        else:
            # No rank of its own: MILESTONES is ordered, so its position is one.
            rank = position
        entries[identifier] = LadderEntry(
            milestone_id=identifier,
            label=str(label) if isinstance(label, str) and label else identifier,
            ladder_index=rank,
        )
    return entries


@dataclass(frozen=True)
class Attainment:
    """The first time a milestone was reached, priced in buttons."""

    milestone_id: str
    label: str
    ladder_index: Optional[int]
    #: Cumulative presses across the whole run, including the batch that got it.
    presses: int
    seq: int
    #: Seconds from the first receipt, or None on a run with no clock.
    seconds: Optional[float] = None

    @property
    def sort_key(self) -> tuple[int, int, int]:
        # Ladder order when the ladder knows the milestone; attainment order for
        # anything it does not (an event id, or a ladder that is not built yet).
        known = 0 if self.ladder_index is not None else 1
        return known, self.ladder_index if self.ladder_index is not None else 0, self.seq


@dataclass(frozen=True)
class MapRevisit:
    """How much of one map's walking was ground already covered."""

    map_name: str
    samples: int
    unique: int

    @property
    def ratio(self) -> float:
        return (self.samples / self.unique) if self.unique else 0.0


@dataclass(frozen=True)
class RunMetrics:
    """Everything computed about one run. Every field is safe on an empty run."""

    run_id: str = ""
    goal: str = ""
    model: str = ""
    status: str = ""
    harness_sha: str = ""
    config_hash: str = ""
    start_checkpoint: Optional[str] = None

    # -- the headline
    presses_to: dict[str, int] = field(default_factory=dict)
    attainments: tuple[Attainment, ...] = ()
    total_presses: int = 0
    milestones_reached: int = 0
    furthest_milestone: Optional[str] = None
    furthest_label: str = ""
    presses_per_milestone: Optional[float] = None

    # -- how the buttons were spent
    receipts: int = 0
    action_batches: int = 0
    blocked_batches: int = 0
    blocked_rate: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    tool_error_rate: float = 0.0

    # -- where they were spent
    position_samples: int = 0
    unique_positions: int = 0
    revisit_ratio: float = 0.0
    revisit_by_map: tuple[MapRevisit, ...] = ()

    # -- how the run went wrong
    whiteouts: int = 0
    reloads: int = 0

    # -- clock
    wall_clock_seconds: float = 0.0
    first_t: Optional[float] = None
    last_t: Optional[float] = None

    #: The tracker's own event total at the last receipt, ladder and all.
    final_milestone_count: int = 0
    corrupt_receipt_lines: int = 0

    def presses_for(self, milestone_id: str) -> Optional[int]:
        return self.presses_to.get(milestone_id)

    def label_for(self, milestone_id: str) -> str:
        for attainment in self.attainments:
            if attainment.milestone_id == milestone_id:
                return attainment.label
        return milestone_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "model": self.model,
            "status": self.status,
            "presses_to": dict(self.presses_to),
            "total_presses": self.total_presses,
            "milestones_reached": self.milestones_reached,
            "furthest_milestone": self.furthest_milestone,
            "presses_per_milestone": self.presses_per_milestone,
            "blocked_rate": self.blocked_rate,
            "tool_error_rate": self.tool_error_rate,
            "revisit_ratio": self.revisit_ratio,
            "revisit_by_map": {entry.map_name: entry.ratio for entry in self.revisit_by_map},
            "whiteouts": self.whiteouts,
            "reloads": self.reloads,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


def _rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def compute(record: RunRecord, *, ladder: Optional[Mapping[str, LadderEntry]] = None) -> RunMetrics:
    """Score one run.

    The press counter is a plain running total over every receipt in file order.
    There is deliberately no branch on ``reloaded`` anywhere in this function:
    reloading a save rewinds the *game*, not the cost of getting there, so a run
    that beats Brock on its fourth attempt reads as the presses of all four
    attempts. Discounting the failed ones would make the number incomparable
    with published action counts, which is the entire point of measuring it.
    """

    rungs = dict(load_ladder() if ladder is None else ladder)
    meta = record.meta
    receipts: tuple[Receipt, ...] = tuple(record.receipts)

    presses_to: dict[str, int] = {}
    attainments: list[Attainment] = []

    cumulative_presses = 0
    action_batches = 0
    blocked_batches = 0
    tool_errors = 0
    whiteouts = 0
    reloads = 0
    final_milestone_count = 0

    samples_by_map: dict[str, int] = {}
    unique_by_map: dict[str, set[tuple[int, int]]] = {}
    position_samples = 0
    unique_positions: set[tuple[str, int, int]] = set()

    first_t: Optional[float] = None
    last_t: Optional[float] = None

    for receipt in receipts:
        # The one line that makes the metric honest. No condition guards it.
        cumulative_presses += receipt.presses

        if receipt.t:
            if first_t is None:
                first_t = receipt.t
            last_t = receipt.t

        if receipt.is_action_batch:
            action_batches += 1
            if receipt.moved == 0:
                blocked_batches += 1
        if receipt.errored:
            tool_errors += 1
        if receipt.whiteout:
            whiteouts += 1
        if receipt.reloaded:
            reloads += 1
        if receipt.milestone_count:
            final_milestone_count = receipt.milestone_count

        if receipt.pos is not None:
            map_name = receipt.map_name or "?"
            position_samples += 1
            samples_by_map[map_name] = samples_by_map.get(map_name, 0) + 1
            unique_by_map.setdefault(map_name, set()).add(receipt.pos)
            unique_positions.add((map_name, receipt.pos[0], receipt.pos[1]))

        for milestone_id in receipt.milestones_new:
            if milestone_id in presses_to:
                continue  # First attainment only: a re-fired event is not progress.
            rung = rungs.get(milestone_id)
            presses_to[milestone_id] = cumulative_presses
            attainments.append(
                Attainment(
                    milestone_id=milestone_id,
                    label=rung.label if rung else milestone_id,
                    ladder_index=rung.ladder_index if rung else None,
                    presses=cumulative_presses,
                    seq=receipt.seq,
                    seconds=(
                        round(receipt.t - first_t, 3) if receipt.t and first_t is not None else None
                    ),
                )
            )

    attainments.sort(key=lambda item: item.sort_key)
    ordered_presses_to = {item.milestone_id: item.presses for item in attainments}

    furthest = attainments[-1] if attainments else None
    milestones_reached = len(ordered_presses_to)

    revisit_by_map = tuple(
        MapRevisit(
            map_name=map_name,
            samples=samples_by_map.get(map_name, 0),
            unique=len(coords),
        )
        for map_name, coords in sorted(
            unique_by_map.items(), key=lambda item: (-samples_by_map.get(item[0], 0), item[0])
        )
    )

    started = meta.started_at or first_t or 0.0
    ended = meta.ended_at or last_t or started
    wall_clock = max(0.0, round(float(ended) - float(started), 3))

    return RunMetrics(
        run_id=meta.run_id,
        goal=meta.goal,
        model=meta.model,
        status=meta.status,
        harness_sha=meta.harness_sha,
        config_hash=meta.config_hash,
        start_checkpoint=meta.start_checkpoint,
        presses_to=ordered_presses_to,
        attainments=tuple(attainments),
        total_presses=cumulative_presses,
        milestones_reached=milestones_reached,
        furthest_milestone=furthest.milestone_id if furthest else None,
        furthest_label=furthest.label if furthest else "",
        presses_per_milestone=(
            round(cumulative_presses / milestones_reached, 1) if milestones_reached else None
        ),
        receipts=len(receipts),
        action_batches=action_batches,
        blocked_batches=blocked_batches,
        blocked_rate=round(_rate(blocked_batches, action_batches), 4),
        tool_calls=len(receipts),
        tool_errors=tool_errors,
        tool_error_rate=round(_rate(tool_errors, len(receipts)), 4),
        position_samples=position_samples,
        unique_positions=len(unique_positions),
        revisit_ratio=round(_rate(position_samples, len(unique_positions)), 3),
        revisit_by_map=revisit_by_map,
        whiteouts=whiteouts,
        reloads=reloads,
        wall_clock_seconds=wall_clock,
        first_t=first_t,
        last_t=last_t,
        final_milestone_count=final_milestone_count,
        corrupt_receipt_lines=record.corrupt_lines,
    )
