"""Three questions about a run in progress that a run-wide total cannot answer.

**Where did the presses on this map actually go?** ``waste`` says Route 3 took
5,391 presses. It cannot say whether that was one crossing or eleven attempts at
the same crossing, and those are different bugs. :func:`episode_report` splits
the run into *episodes* — one unbroken stay on one map — so arrivals, spends and
departures are countable, and adds the doors out of each visited map that the
run never went through, because absence is exactly what a histogram of what
happened cannot show.

**Did a change work?** A harness change ships mid-run, so the before and after
are two stretches of the same run rather than two runs. :func:`split_report`
compares any two time ranges inside one run on the same metrics, split at a
marker that can be named after the event rather than guessed at as a timestamp.

**What did an intervention do?** :func:`intervention_report` finds the advice in
the transcripts, replays the harness's own detectors over the receipts to name
the trigger that fired it, and measures the same stretch of play either side.

Everything is computed from receipts, which are appended while this runs; a
window that is still filling is reported with the sample size next to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from pokemon_agent.bench.registry import Receipt, RunRecord
from pokemon_agent.scope import truth
from pokemon_agent.scope.transcript import Injection, Session, median

# -- episodes -----------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One unbroken stay on one map."""

    index: int
    map_name: str
    first_seq: int
    last_seq: int
    presses: int
    batches: int
    started_at: float = 0.0
    ended_at: float = 0.0
    entry: Optional[tuple[int, int]] = None
    exit: Optional[tuple[int, int]] = None
    #: Distinct tiles stood on during this stay.
    tiles: int = 0
    #: Of those, the ones never stood on before anywhere earlier in the run.
    new_tiles: int = 0
    blocked_batches: int = 0
    milestones: int = 0
    #: The map this stay ended by moving to; ``""`` when it is the last one.
    went_to: str = ""

    @property
    def elapsed(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    @property
    def yield_per_100(self) -> float:
        """New ground per hundred presses — the only output an episode has."""

        return (100.0 * self.new_tiles / self.presses) if self.presses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "map": self.map_name,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "presses": self.presses,
            "batches": self.batches,
            "entry": list(self.entry) if self.entry else None,
            "exit": list(self.exit) if self.exit else None,
            "tiles": self.tiles,
            "new_tiles": self.new_tiles,
            "blocked_batches": self.blocked_batches,
            "milestones": self.milestones,
            "went_to": self.went_to,
            "elapsed_seconds": round(self.elapsed, 1),
            "new_tiles_per_100_presses": round(self.yield_per_100, 2),
        }


@dataclass(frozen=True)
class MapStay:
    """Every episode on one map, summarised."""

    map_name: str
    episodes: int
    presses: int
    tiles: int
    new_tiles: int
    median_presses: float
    milestones: int
    #: Area the game data says the map has, or 0 when it is not known.
    map_tiles: int = 0

    @property
    def yield_per_100(self) -> float:
        return (100.0 * self.new_tiles / self.presses) if self.presses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "episodes": self.episodes,
            "presses": self.presses,
            "tiles": self.tiles,
            "new_tiles": self.new_tiles,
            "median_presses": self.median_presses,
            "milestones": self.milestones,
            "map_tiles": self.map_tiles,
            "new_tiles_per_100_presses": round(self.yield_per_100, 2),
        }


@dataclass(frozen=True)
class EpisodeReport:
    run_id: str
    episodes: tuple[Episode, ...]
    stays: tuple[MapStay, ...]
    #: ``((from, to), times)`` for every map change, most travelled first.
    transitions: tuple[tuple[tuple[str, str], int], ...]
    #: Exits the game data offers from a visited map that the run never took.
    untried: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "stays": [stay.to_dict() for stay in self.stays],
            "transitions": [
                {"from": pair[0], "to": pair[1], "times": count} for pair, count in self.transitions
            ],
            "untried": [{"from": pair[0], "to": pair[1]} for pair in self.untried],
        }


@dataclass
class _OpenEpisode:
    map_name: str
    first_seq: int
    last_seq: int
    started_at: float
    ended_at: float
    presses: int = 0
    batches: int = 0
    blocked: int = 0
    milestones: int = 0
    entry: Optional[tuple[int, int]] = None
    exit: Optional[tuple[int, int]] = None
    new_tiles: int = 0
    tiles: set[tuple[int, int]] = field(default_factory=set)


def episode_report(record: RunRecord) -> EpisodeReport:
    """Split the run into stays on a map, and count what each one bought."""

    seen: set[tuple[str, int, int]] = set()
    episodes: list[Episode] = []
    transitions: dict[tuple[str, str], int] = {}
    open_episode: Optional[_OpenEpisode] = None

    def close(next_map: str) -> None:
        if open_episode is None:
            return
        episodes.append(
            Episode(
                index=len(episodes),
                map_name=open_episode.map_name,
                first_seq=open_episode.first_seq,
                last_seq=open_episode.last_seq,
                presses=open_episode.presses,
                batches=open_episode.batches,
                started_at=open_episode.started_at,
                ended_at=open_episode.ended_at,
                entry=open_episode.entry,
                exit=open_episode.exit,
                tiles=len(open_episode.tiles),
                new_tiles=open_episode.new_tiles,
                blocked_batches=open_episode.blocked,
                milestones=open_episode.milestones,
                went_to=next_map,
            )
        )

    for receipt in record.receipts:
        map_name = receipt.map_name
        if not map_name:
            continue
        if open_episode is not None and open_episode.map_name != map_name:
            pair = (open_episode.map_name, map_name)
            transitions[pair] = transitions.get(pair, 0) + 1
            close(map_name)
            open_episode = None
        if open_episode is None:
            open_episode = _OpenEpisode(
                map_name=map_name,
                first_seq=receipt.seq,
                last_seq=receipt.seq,
                started_at=receipt.t,
                ended_at=receipt.t,
            )
        open_episode.last_seq = receipt.seq
        open_episode.ended_at = receipt.t or open_episode.ended_at
        open_episode.presses += receipt.presses
        if receipt.presses:
            open_episode.batches += 1
            if receipt.moved == 0:
                open_episode.blocked += 1
        open_episode.milestones += len(receipt.milestones_new)
        if receipt.pos is not None:
            position = (receipt.pos[0], receipt.pos[1])
            open_episode.exit = position
            if open_episode.entry is None:
                open_episode.entry = position
            open_episode.tiles.add(position)
            tile = (map_name, position[0], position[1])
            if tile not in seen:
                open_episode.new_tiles += 1
                seen.add(tile)
    close("")

    by_map: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_map.setdefault(episode.map_name, []).append(episode)

    stays = [
        MapStay(
            map_name=map_name,
            episodes=len(group),
            presses=sum(episode.presses for episode in group),
            tiles=len({tile for tile in seen if tile[0] == map_name}),
            new_tiles=sum(episode.new_tiles for episode in group),
            median_presses=median([float(episode.presses) for episode in group]) or 0.0,
            milestones=sum(episode.milestones for episode in group),
            map_tiles=_map_area(map_name),
        )
        for map_name, group in by_map.items()
    ]
    stays.sort(key=lambda stay: (-stay.presses, stay.map_name))

    return EpisodeReport(
        run_id=record.run_id,
        episodes=tuple(episodes),
        stays=tuple(stays),
        transitions=tuple(sorted(transitions.items(), key=lambda item: -item[1])),
        untried=tuple(_untried_edges(set(by_map), transitions)),
    )


def _map_area(map_name: str) -> int:
    known = truth.map_truth(map_name)
    return 0 if known is None else known.width * known.height


def _untried_edges(
    visited: set[str], transitions: dict[tuple[str, str], int]
) -> list[tuple[str, str]]:
    """Doors out of the maps the run stood on that it never went through."""

    out: list[tuple[str, str]] = []
    for map_name in sorted(visited):
        known = truth.map_truth(map_name)
        if known is None:
            continue
        for destination in known.exits:
            if (map_name, destination) not in transitions:
                out.append((map_name, destination))
    return out


# -- windows ------------------------------------------------------------------

#: What a window is judged on. Every one of these is a count or a ratio of
#: counts; none of them needs a paragraph to interpret.
WINDOW_FIELDS: tuple[tuple[str, str], ...] = (
    ("presses", "presses"),
    ("batches", "batches"),
    ("minutes", "minutes"),
    ("presses_per_minute", "presses/min"),
    ("median_batch", "med batch"),
    ("blocked_share", "blocked"),
    ("new_tiles", "new tiles"),
    ("new_per_1k", "new/1k press"),
    ("maps", "maps"),
    ("map_changes", "map changes"),
    ("milestones", "milestones"),
    ("reloads", "reloads"),
    ("median_hp", "med hp"),
)

#: Ratio fields are printed as percentages rather than as decimals.
SHARE_FIELDS = frozenset({"blocked_share", "median_hp"})


@dataclass(frozen=True)
class WindowStats:
    """One stretch of a run, measured."""

    label: str
    presses: int = 0
    batches: int = 0
    minutes: float = 0.0
    median_batch: float = 0.0
    blocked_share: float = 0.0
    new_tiles: int = 0
    maps: int = 0
    map_changes: int = 0
    milestones: int = 0
    reloads: int = 0
    median_hp: float = 0.0
    first_seq: Optional[int] = None
    last_seq: Optional[int] = None

    @property
    def presses_per_minute(self) -> float:
        return (self.presses / self.minutes) if self.minutes else 0.0

    @property
    def new_per_1k(self) -> float:
        return (1000.0 * self.new_tiles / self.presses) if self.presses else 0.0

    def value(self, name: str) -> float:
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        payload = {name: round(self.value(name), 3) for name, _ in WINDOW_FIELDS}
        payload["label"] = self.label
        payload["first_seq"] = self.first_seq
        payload["last_seq"] = self.last_seq
        return payload


def window_stats(label: str, annotated: Sequence[tuple[Receipt, bool]]) -> WindowStats:
    """Measure a window given receipts already tagged with "this tile was new"."""

    acting = [(receipt, fresh) for receipt, fresh in annotated if receipt.presses > 0]
    if not acting:
        return WindowStats(label=label)
    presses = sum(receipt.presses for receipt, _ in acting)
    times = [receipt.t for receipt, _ in acting if receipt.t]
    minutes = ((max(times) - min(times)) / 60.0) if len(times) > 1 else 0.0
    blocked = sum(1 for receipt, _ in acting if receipt.moved == 0)
    maps: list[str] = []
    for receipt, _ in acting:
        if receipt.map_name and (not maps or maps[-1] != receipt.map_name):
            maps.append(receipt.map_name)
    fractions = [
        receipt.hp[0] / receipt.hp[1]
        for receipt, _ in acting
        if receipt.hp is not None and receipt.hp[1] > 0
    ]
    return WindowStats(
        label=label,
        presses=presses,
        batches=len(acting),
        minutes=round(minutes, 2),
        median_batch=median([float(receipt.presses) for receipt, _ in acting]) or 0.0,
        blocked_share=blocked / len(acting),
        new_tiles=sum(1 for _, fresh in acting if fresh),
        maps=len(set(maps)),
        map_changes=max(0, len(maps) - 1),
        milestones=sum(len(receipt.milestones_new) for receipt, _ in acting),
        reloads=sum(1 for receipt, _ in acting if receipt.reloaded),
        median_hp=median(fractions) or 0.0,
        first_seq=acting[0][0].seq,
        last_seq=acting[-1][0].seq,
    )


def annotate(receipts: Iterable[Receipt]) -> list[tuple[Receipt, bool]]:
    """Tag every receipt with whether it landed on ground never stood on before.

    The tag has to be computed from the start of the run or it means nothing: a
    tile is only new once, and a window that begins in the middle cannot know
    what came before it.
    """

    seen: set[tuple[str, int, int]] = set()
    out: list[tuple[Receipt, bool]] = []
    for receipt in receipts:
        fresh = False
        if receipt.pos is not None:
            tile = (receipt.map_name or "?", receipt.pos[0], receipt.pos[1])
            fresh = tile not in seen
            seen.add(tile)
        out.append((receipt, fresh))
    return out


@dataclass(frozen=True)
class SplitReport:
    """Two stretches of one run, side by side."""

    run_id: str
    marker: str
    at: float
    before: WindowStats
    after: WindowStats
    #: Presses each side was limited to, or ``None`` for "everything".
    span: Optional[int]

    def deltas(self) -> list[tuple[str, str, float, float, Optional[float]]]:
        """``(field, heading, before, after, relative change)`` for each metric."""

        rows = []
        for name, heading in WINDOW_FIELDS:
            low, high = self.before.value(name), self.after.value(name)
            change = ((high - low) / low) if low else None
            rows.append((name, heading, low, high, change))
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "marker": self.marker,
            "at": self.at,
            "span": self.span,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


def split_report(
    record: RunRecord,
    *,
    marker: str,
    at: float,
    span: Optional[int] = None,
) -> SplitReport:
    """Compare the run either side of ``at``.

    ``span`` caps each side at that many presses, counted outwards from the
    split. Without it the two sides are the whole run, which is the honest
    default but weights the comparison towards whichever side is longer.
    """

    annotated = annotate(record.receipts)
    before = [item for item in annotated if item[0].t and item[0].t < at]
    after = [item for item in annotated if item[0].t and item[0].t >= at]
    if span is not None:
        before = _tail_presses(before, span)
        after = _head_presses(after, span)
    return SplitReport(
        run_id=record.run_id,
        marker=marker,
        at=at,
        before=window_stats("before", before),
        after=window_stats("after", after),
        span=span,
    )


def _tail_presses(
    annotated: Sequence[tuple[Receipt, bool]], presses: int
) -> list[tuple[Receipt, bool]]:
    out: list[tuple[Receipt, bool]] = []
    spent = 0
    for item in reversed(annotated):
        out.append(item)
        spent += item[0].presses
        if spent >= presses:
            break
    out.reverse()
    return out


def _head_presses(
    annotated: Sequence[tuple[Receipt, bool]], presses: int
) -> list[tuple[Receipt, bool]]:
    out: list[tuple[Receipt, bool]] = []
    spent = 0
    for item in annotated:
        out.append(item)
        spent += item[0].presses
        if spent >= presses:
            break
    return out


# -- interventions ------------------------------------------------------------

#: Presses either side of an intervention that the effect is measured over.
DEFAULT_INTERVENTION_SPAN = 400

#: The words an instruction uses for its first move. Compass and screen
#: directions are both used by the thinking session and mean the same thing to
#: ``poke act``, which only takes the screen ones.
_DIRECTIONS = {
    "up": "up",
    "north": "up",
    "down": "down",
    "south": "down",
    "left": "left",
    "west": "left",
    "right": "right",
    "east": "right",
}


@dataclass(frozen=True)
class InterventionEvent:
    """One piece of advice pushed into a live session, and what came of it."""

    index: int
    at: float
    session: str
    step: int
    #: The detector the harness's own policy would have fired here, replayed.
    trigger: str
    trigger_reason: str
    headline: str
    #: The first direction the advice asks for, when it names one.
    asked_for: str = ""
    #: Whether any of the next few movement calls used it. ``None`` = no ask.
    followed: Optional[bool] = None
    before: WindowStats = field(default_factory=lambda: WindowStats(label="before"))
    after: WindowStats = field(default_factory=lambda: WindowStats(label="after"))
    hp_at: float = 0.0
    hp_best_after: float = 0.0

    @property
    def healed(self) -> bool:
        return self.hp_best_after > self.hp_at + 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at": self.at,
            "session": self.session,
            "step": self.step,
            "trigger": self.trigger,
            "trigger_reason": self.trigger_reason,
            "headline": self.headline,
            "asked_for": self.asked_for,
            "followed": self.followed,
            "hp_at": round(self.hp_at, 3),
            "hp_best_after": round(self.hp_best_after, 3),
            "healed": self.healed,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class InterventionReport:
    run_id: str
    events: tuple[InterventionEvent, ...]
    #: ``(detector, samples it was firing on)`` across the whole run.
    standing: tuple[tuple[str, int], ...]
    #: How many windows were sampled to produce ``standing``.
    samples: int
    span: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "span": self.span,
            "samples": self.samples,
            "events": [event.to_dict() for event in self.events],
            "standing": [
                {
                    "detector": name,
                    "samples": count,
                    "share": round(count / self.samples, 3) if self.samples else 0.0,
                }
                for name, count in self.standing
            ],
        }


def find_injections(
    sessions: Iterable[Session],
    *,
    between: Optional[tuple[float, float]] = None,
) -> list[tuple[Session, Injection]]:
    """Every mid-session message that is advice, oldest first.

    Three things are not advice and are dropped: ``continue``, which is the
    harness restarting a stalled turn; a bare attachment envelope, which is a
    frame being handed over with no words; and anything outside ``between``,
    which is how a workspace holding transcripts from several runs is narrowed
    to the one run being asked about.
    """

    low, high = between if between else (float("-inf"), float("inf"))
    out: list[tuple[Session, Injection]] = []
    for session in sessions:
        for injection in session.injections:
            if injection.at is None or not (low <= injection.at <= high):
                continue
            if injection.is_continue or injection.headline.startswith("<file"):
                continue
            out.append((session, injection))
    out.sort(key=lambda item: item[1].at or 0.0)
    return out


def run_window(record: RunRecord) -> tuple[float, float]:
    """``(first, last)`` receipt timestamps — when this run was being played."""

    times = [receipt.t for receipt in record.receipts if receipt.t]
    return (min(times), max(times)) if times else (float("-inf"), float("inf"))


def _first_ask(text: str) -> str:
    """The first direction the advice asks the player to move in."""

    for match in re.finditer(r"\b([A-Za-z]+)\b", text):
        word = match.group(1).lower()
        if word in _DIRECTIONS:
            return _DIRECTIONS[word]
    return ""


def _followed(session: Session, step: int, direction: str, *, lookahead: int = 6) -> Optional[bool]:
    """Did any of the next few movement calls press that direction?"""

    if not direction:
        return None
    calls = [call for call in session.calls if step <= call.step < step + lookahead]
    movement = [call for call in calls if call.label in {"poke act", "poke goto"}]
    if not movement:
        return None
    return any(direction in call.command.lower() for call in movement)


def replay_triggers(
    receipts: Sequence[Receipt], at: float, *, window: int = 120
) -> tuple[str, str]:
    """Name the detector that would fire on the receipts up to ``at``.

    The harness decides its own interventions from :mod:`pokemon_agent.interventions`,
    and that module is pure — detectors take a window of receipts and return a
    trigger. Replaying them here is therefore the real answer to "why did this
    fire?", not a reconstruction of one. A missing module means the trigger is
    reported as unknown rather than guessed.
    """

    try:
        from pokemon_agent.interventions import default_detectors
    except Exception:  # noqa: BLE001 — the trigger is a label, not the report
        return "", ""
    history = [receipt for receipt in receipts if receipt.t and receipt.t <= at]
    if not history:
        return "", ""
    tail = history[-window:]
    candidates = []
    for detector in default_detectors():
        try:
            trigger = detector.check(tail, {})
        except Exception:  # noqa: BLE001
            continue
        if trigger is not None:
            candidates.append(trigger)
    if not candidates:
        return "", ""
    best = max(candidates, key=lambda trigger: trigger.priority)
    return best.name, best.reason


def _hp_fraction(receipt: Receipt) -> Optional[float]:
    if receipt.hp is None or receipt.hp[1] <= 0:
        return None
    return receipt.hp[0] / receipt.hp[1]


def intervention_report(
    record: RunRecord,
    sessions: Iterable[Session],
    *,
    span: int = DEFAULT_INTERVENTION_SPAN,
) -> InterventionReport:
    """Every intervention, its trigger, and the play either side of it."""

    annotated = annotate(record.receipts)
    receipts = list(record.receipts)
    events: list[InterventionEvent] = []
    found = find_injections(sessions, between=run_window(record))

    for index, (session, injection) in enumerate(found, start=1):
        at = injection.at or 0.0
        before = _tail_presses([item for item in annotated if item[0].t and item[0].t < at], span)
        after = _head_presses([item for item in annotated if item[0].t and item[0].t >= at], span)
        name, reason = replay_triggers(receipts, at)
        hp_now = next(
            (
                value
                for value in (_hp_fraction(item[0]) for item in reversed(before))
                if value is not None
            ),
            0.0,
        )
        hp_best = max(
            (value for value in (_hp_fraction(item[0]) for item in after) if value is not None),
            default=0.0,
        )
        asked = _first_ask(injection.text)
        events.append(
            InterventionEvent(
                index=index,
                at=at,
                session=session.short_id,
                step=injection.step,
                trigger=name,
                trigger_reason=reason,
                headline=injection.headline,
                asked_for=asked,
                followed=_followed(session, injection.step, asked),
                before=window_stats("before", before),
                after=window_stats("after", after),
                hp_at=hp_now,
                hp_best_after=hp_best,
            )
        )

    standing, samples = _standing(receipts)
    return InterventionReport(
        run_id=record.run_id,
        events=tuple(events),
        standing=tuple(standing),
        samples=samples,
        span=span,
    )


def _standing(
    receipts: Sequence[Receipt], *, window: int = 120, stride: int = 20
) -> tuple[list[tuple[str, int]], int]:
    """How often each detector was in a firing state, sampled along the run.

    An intervention budget is a cap on how much of a signal is acted on, so the
    interesting number is not how often one fired but how much of the run it was
    true for. Sampled every ``stride`` batches, because a detector's answer does
    not change between two consecutive presses and evaluating 4,000 windows to
    find that out would be the slowest thing in this package.
    """

    try:
        from pokemon_agent.interventions import default_detectors
    except Exception:  # noqa: BLE001
        return [], 0
    detectors = default_detectors()
    counts: dict[str, int] = {}
    samples = 0
    for end in range(window, len(receipts) + 1, stride):
        samples += 1
        tail = receipts[max(0, end - window) : end]
        for detector in detectors:
            try:
                trigger = detector.check(tail, {})
            except Exception:  # noqa: BLE001
                continue
            if trigger is not None:
                counts[trigger.name] = counts.get(trigger.name, 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1]), samples
