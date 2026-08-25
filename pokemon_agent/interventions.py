"""Decide when to stop the player and think.

The player model runs non-thinking so it stays fast. A second, thinking session
can be swapped in at any point once llama.cpp slot save/restore is available:
the player's KV cache goes to disk, the thinker runs in the freed slot, and the
player's cache comes back. That costs two KV round-trips instead of a full
re-prefill, so an intervention is affordable enough to fire on a rule rather
than only at session boundaries.

Which rule is the whole question, and the answer this module encodes is: the
harness decides, never the model. Every advisory signal this project has handed
the model has been ignored. ``here_before`` reached 49 on a single tile without
changing anything it did. Asking the player to notice it is stuck and call for
help would be the same mistake with more steps, so detectors read receipts and
fire on their own.

Everything here is pure. Detectors take a window of receipts and a state dict
and return a :class:`Trigger` or ``None``; the policy decides which trigger
wins and whether the budget allows it. No I/O, no model calls, no clock.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from pokemon_agent.bench.registry import Receipt

#: Priorities. A higher number wins when two detectors fire on the same batch.
#: Ordered by how badly the run is doing, not by how interesting the answer
#: would be: losing the party ends the run, being lost merely wastes it.
PRIORITY_COMMIT = 40
PRIORITY_DANGER = 30
PRIORITY_STUCK = 20
PRIORITY_REHEARSAL = 10


@dataclass(frozen=True)
class Trigger:
    """One detector's decision that the player should stop and think."""

    name: str
    priority: int
    #: Written for the thinking session to read, so it says what was observed
    #: rather than what to conclude.
    reason: str
    question: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    name: str

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]: ...


def _presses(window: Iterable[Receipt]) -> int:
    return sum(r.presses for r in window)


def _tail(window: Sequence[Receipt], presses: int) -> list[Receipt]:
    """The most recent receipts adding up to ``presses`` buttons, oldest first."""

    out: list[Receipt] = []
    spent = 0
    for receipt in reversed(window):
        out.append(receipt)
        spent += receipt.presses
        if spent >= presses:
            break
    out.reverse()
    return out


@dataclass
class StalledMilestones:
    """No rung of the ladder reached in a long time.

    The clearest evidence a run is not progressing, and it needs no judgement
    about *why*. That is the thinking session's job.
    """

    presses: int = 800
    name: str = "stalled"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        if _presses(window) < self.presses:
            return None
        recent = _tail(window, self.presses)
        if any(r.milestones_new for r in recent):
            return None
        spent = _presses(recent)
        maps = sorted({r.map_name for r in recent if r.map_name})
        return Trigger(
            name=self.name,
            priority=PRIORITY_STUCK,
            reason=(
                f"{spent} button presses since the last milestone. "
                f"Maps visited in that span: {', '.join(maps) or 'unknown'}."
            ),
            question=(
                "Progress has stopped. Work out what is blocking it and give "
                "the next concrete sequence of moves."
            ),
            payload={"presses_since_milestone": spent, "maps": maps},
        )


@dataclass
class Circling:
    """Standing on ground already covered, over and over.

    ``revisit_ratio`` is samples over unique positions. A ratio of 1.0 is a
    straight line; the Pewter failure that motivated this module sat at 2.3
    with one tile stood on 49 times.
    """

    ratio: float = 2.5
    min_samples: int = 40
    name: str = "circling"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        positions = [(r.map_name, r.pos) for r in window if r.pos is not None]
        if len(positions) < self.min_samples:
            return None
        counts = Counter(positions)
        ratio = len(positions) / len(counts)
        if ratio < self.ratio:
            return None
        (worst_map, worst_pos), worst_count = counts.most_common(1)[0]
        return Trigger(
            name=self.name,
            priority=PRIORITY_STUCK,
            reason=(
                f"{len(positions)} positions sampled across {len(counts)} "
                f"distinct tiles (ratio {ratio:.1f}). Most repeated: "
                f"{worst_pos} on {worst_map}, {worst_count} times."
            ),
            question=(
                "The player is walking in circles. Say where it should go "
                "instead and how to get there from where it is standing."
            ),
            payload={
                "ratio": round(ratio, 2),
                "unique": len(counts),
                "worst": {"map": worst_map, "pos": list(worst_pos), "count": worst_count},
            },
        )


@dataclass
class LowHP:
    """Party leader has been hurt for a while and nothing was done about it.

    Whiting out costs money, teleports the player, and has ended several runs
    here. It is also entirely predictable several hundred presses in advance.
    """

    fraction: float = 0.35
    batches: int = 5
    name: str = "low_hp"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        recent = [r for r in window[-self.batches :] if r.hp is not None]
        if len(recent) < self.batches:
            return None
        hurt = [r for r in recent if r.hp[1] > 0 and r.hp[0] / r.hp[1] <= self.fraction]
        if len(hurt) < self.batches:
            return None
        current, maximum = recent[-1].hp
        return Trigger(
            name=self.name,
            priority=PRIORITY_DANGER,
            reason=(
                f"Party leader at {current}/{maximum} HP for {len(hurt)} "
                f"consecutive batches, on {recent[-1].map_name or 'an unknown map'}."
            ),
            question=(
                "The party is close to fainting and the player has not healed. "
                "Decide whether to heal, retreat, or push on, and say how."
            ),
            payload={"hp": [current, maximum], "batches": len(hurt)},
        )


@dataclass
class RepeatedFailure:
    """The same thing failed more than once in a row.

    A model that retries a failing call verbatim will keep retrying it. Two is
    enough evidence; a third attempt teaches nobody anything.
    """

    times: int = 2
    name: str = "repeated_failure"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        if len(window) < self.times:
            return None
        recent = window[-self.times :]
        if not all(r.exit_code != 0 for r in recent):
            return None
        tools = {r.tool for r in recent}
        if len(tools) != 1:
            return None
        tool = recent[-1].tool
        return Trigger(
            name=self.name,
            priority=PRIORITY_DANGER,
            reason=f"`{tool}` failed {len(recent)} times in a row.",
            question=(
                "The same command keeps failing. Work out why from the error "
                "and give a command that will work instead."
            ),
            payload={"tool": tool, "failures": len(recent)},
        )


@dataclass
class EnteringSegment:
    """First arrival somewhere known to be hard.

    Rehearsal rather than rescue. The gyms and the mazes are where runs are
    lost, and they are the places where planning before acting is worth the
    swap.
    """

    maps: frozenset[str] = frozenset()
    name: str = "rehearsal"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        if not window:
            return None
        here = window[-1].map_name
        if here not in self.maps:
            return None
        earlier = {r.map_name for r in window[:-1]}
        if here in earlier:
            return None
        return Trigger(
            name=self.name,
            priority=PRIORITY_REHEARSAL,
            reason=f"First arrival on {here}, which is a known-hard segment.",
            question=(
                f"The player has just entered {here}. Plan this segment before "
                "it starts acting: what to do, in what order, and what to avoid."
            ),
            payload={"map": here},
        )


@dataclass
class CommitGate:
    """About to do something that cannot be undone.

    The receipt carries the intent in ``extra`` because only the caller knows
    an action was irreversible; this detector just notices the flag.
    """

    kinds: frozenset[str] = frozenset({"release", "sell", "use_rare_item", "evolve"})
    name: str = "commit_gate"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        pending = state.get("pending_commit")
        if not isinstance(pending, Mapping):
            return None
        kind = pending.get("kind")
        if kind not in self.kinds:
            return None
        return Trigger(
            name=self.name,
            priority=PRIORITY_COMMIT,
            reason=f"About to {kind}: {pending.get('detail') or 'no detail given'}.",
            question=(
                "This cannot be undone. Say whether to go through with it, and "
                "if not, what to do instead."
            ),
            payload=dict(pending),
        )


DEFAULT_HARD_SEGMENTS = frozenset(
    {
        "Viridian Forest",
        "Mt Moon 1F",
        "Mt Moon B1F",
        "Mt Moon B2F",
        "Pewter Gym",
        "Cerulean Gym",
        "Rock Tunnel 1F",
        "Rock Tunnel B1F",
        "Vermilion Gym",
        "Celadon Gym",
        "Rocket Hideout B4F",
        "Silph Co 11F",
        "Fuchsia Gym",
        "Saffron Gym",
        "Seafoam Islands B4F",
        "Cinnabar Gym",
        "Viridian Gym",
        "Victory Road 2F",
    }
)


def default_detectors() -> tuple[Detector, ...]:
    return (
        CommitGate(),
        LowHP(),
        RepeatedFailure(),
        StalledMilestones(),
        Circling(),
        EnteringSegment(maps=DEFAULT_HARD_SEGMENTS),
    )


@dataclass
class InterventionPolicy:
    """Which trigger wins, and whether we can afford it.

    A swap is cheap but not free, and a thinking session that fires every
    hundred presses would spend more wall clock reasoning than playing. The
    cooldown is measured in button presses rather than seconds so it does not
    drift with how fast the model happens to be running.
    """

    detectors: tuple[Detector, ...] = field(default_factory=default_detectors)
    cooldown_presses: int = 600
    max_per_session: int = 12
    window: int = 120

    fired: list[tuple[int, str]] = field(default_factory=list)

    def evaluate(
        self,
        receipts: Sequence[Receipt],
        state: Optional[Mapping[str, Any]] = None,
        *,
        total_presses: Optional[int] = None,
    ) -> Optional[Trigger]:
        """Return the trigger to act on, or ``None`` to keep playing."""

        state = state or {}
        spent = _presses(receipts) if total_presses is None else total_presses
        if len(self.fired) >= self.max_per_session:
            return None
        if self.fired and spent - self.fired[-1][0] < self.cooldown_presses:
            return None

        window = list(receipts[-self.window :])
        candidates = [
            trigger
            for trigger in (d.check(window, state) for d in self.detectors)
            if trigger is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.priority)

    def record(self, trigger: Trigger, total_presses: int) -> None:
        self.fired.append((total_presses, trigger.name))

    def remaining(self) -> int:
        return max(0, self.max_per_session - len(self.fired))


def build_prompt(
    trigger: Trigger,
    *,
    state_summary: str,
    recent: Sequence[Receipt],
    milestone_summary: str = "",
    limit: int = 20,
) -> str:
    """The whole context a thinking session gets.

    Deliberately small. The point of swapping is that the thinker sees a clean,
    short problem instead of the player's hundred thousand tokens of history.
    """

    lines = [
        "You are advising a model that is playing Pokemon Red and has stopped "
        "making progress. It cannot see this conversation; your answer will be "
        "handed to it as a single instruction.",
        "",
        f"WHY YOU WERE CALLED: {trigger.reason}",
        "",
        "CURRENT STATE",
        state_summary.strip(),
    ]
    if milestone_summary:
        lines += ["", "PROGRESS", milestone_summary.strip()]

    tail = recent[-limit:]
    if tail:
        lines += ["", f"LAST {len(tail)} ACTION BATCHES"]
        for r in tail:
            pos = f"{r.pos}" if r.pos else "?"
            note = []
            if r.blocked_after:
                note.append(f"blocked after {r.blocked_after}")
            if r.moved is not None:
                note.append(f"moved {r.moved}")
            if r.milestones_new:
                note.append("reached " + ", ".join(r.milestones_new))
            if r.exit_code:
                note.append(f"exit {r.exit_code}")
            lines.append(
                f"  {r.seq:>4} {r.map_name or '?':<22} {pos:<10} {r.presses:>3}p  {'; '.join(note)}"
            )

    lines += [
        "",
        "YOUR TASK",
        trigger.question,
        "",
        "Answer in at most 150 words. Be specific about directions and tile "
        "counts. Do not explain your reasoning, give the instruction.",
    ]
    return "\n".join(lines)


__all__ = [
    "Trigger",
    "Detector",
    "StalledMilestones",
    "Circling",
    "LowHP",
    "RepeatedFailure",
    "EnteringSegment",
    "CommitGate",
    "InterventionPolicy",
    "DEFAULT_HARD_SEGMENTS",
    "default_detectors",
    "build_prompt",
    "PRIORITY_COMMIT",
    "PRIORITY_DANGER",
    "PRIORITY_STUCK",
    "PRIORITY_REHEARSAL",
]
