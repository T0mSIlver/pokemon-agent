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

Detection is pure. Detectors take a window of receipts and a state dict and
return a :class:`Trigger` or ``None``; the policy decides which trigger wins
and whether the budget allows it. No I/O, no model calls, no clock.

Prompt building is not, and deliberately so. :func:`harness_facts` reads the
generated map graph and the live collision the harness already has, because a
thinking session asked "where do I go" answers from what it remembers of
Pokemon Red, and what it remembers is wrong often enough to have cost this
project a run. See the comment above :class:`MapFacts` for the firing that
settled it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from pokemon_agent.bench.registry import Receipt
from pokemon_agent.objectives import FRONTIER_PACK_ID

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

    #: The window fallback, in presses, used only when the run cannot say how
    #: long it has really been. It has to fit inside the policy's window, which
    #: is counted in *receipts*: the median 120-receipt window holds 411 presses,
    #: so a threshold above that is a detector switched off rather than a strict
    #: one. That constraint is also why this number alone was never a stall
    #: measurement -- the median gap between milestones in a real run is 2,892
    #: presses, seven times the whole window, so "nothing in the last 400" was
    #: true almost always and fired on 9 of 12 consecutive interventions.
    presses: int = 400

    #: What a stall is when the run *can* say. Set from the same measurement:
    #: the median gap between first-earned milestones is 2,892 presses and the
    #: mean is 4,078, so this fires when the run has spent about a typical
    #: milestone's worth of buttons without reaching one. A detector that is
    #: always true says as little as one that never fires; this one has to be
    #: able to be false.
    run_presses: int = 3000
    name: str = "stalled"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        # The run-wide count when the caller supplies it, because the window
        # cannot reach back far enough to see a real stall.
        since = state.get("presses_since_milestone")
        if isinstance(since, int):
            if since < self.run_presses:
                return None
            maps = sorted({r.map_name for r in window if r.map_name})
            return Trigger(
                name=self.name,
                priority=PRIORITY_STUCK,
                reason=(
                    f"{since} button presses since the last milestone, against a run median of "
                    f"about 2,900 between them. Maps visited recently: "
                    f"{', '.join(maps) or 'unknown'}."
                ),
                question=(
                    "Progress has stopped. Work out what is blocking it and give "
                    "the next concrete sequence of moves."
                ),
                payload={"presses_since_milestone": since, "maps": maps},
            )
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


def _move_power(name: Optional[str]) -> Optional[int]:
    """Base power from the generated move table, or None if it is not a move."""

    if not name:
        return None
    try:
        from pokemon_agent import gamedata

        record = gamedata.move(str(name))
    except Exception:
        return None
    if not record:
        return None
    power = record.get("power")
    return int(power) if power is not None else None


@dataclass
class Toothless:
    """The party cannot deal damage and is still walking into fights.

    Measured over one run: the lead had no damaging move for 3,285 of 13,601
    presses, 24%, with Ember and Rage both at 0 PP and a party of one. Nothing in
    the harness noticed, because every check was about HP. A Pokemon at full
    health with no attack left cannot win a battle and cannot flee a trainer;
    it can only lose slowly.

    The fix is a Poke Center, the same as `low_hp`, which is why this reports the
    cause rather than the remedy. `state` carries the party because the receipt
    schema does not: the caller passes what it read this turn.
    """

    batches: int = 4
    name: str = "toothless"

    def check(self, window: Sequence[Receipt], state: Mapping[str, Any]) -> Optional[Trigger]:
        party = state.get("party")
        if not isinstance(party, Sequence) or not party:
            return None
        lead = party[0] if isinstance(party[0], Mapping) else None
        if not lead:
            return None
        moves = lead.get("moves")
        if not isinstance(moves, Sequence) or not moves:
            return None

        usable = []
        judged = 0
        for move in moves:
            if not isinstance(move, Mapping):
                return None  # names only in this shape; nothing to judge
            pp = move.get("pp")
            if pp is None:
                return None
            # The state payload carries pp but not power, so power comes from the
            # generated move table. Guessing which moves deal damage from their
            # names is exactly the recall this project keeps getting burned by.
            power = move.get("power")
            if power is None:
                power = _move_power(move.get("name"))
            if power is None:
                continue  # unknown move: not evidence either way
            judged += 1
            if power > 0 and pp > 0:
                usable.append(move.get("name"))
        if usable or not judged:
            return None

        # Only worth stopping for if it is still in harm's way.
        fighting = [r for r in window[-self.batches :] if r.tool == "battle" or r.map_name]
        if len(fighting) < self.batches:
            return None

        names = [m.get("name") for m in moves if isinstance(m, Mapping)]
        return Trigger(
            name=self.name,
            priority=PRIORITY_DANGER,
            reason=(
                f"{lead.get('species') or 'The lead'} has no damaging move left: "
                f"{', '.join(str(n) for n in names) or 'no moves'} are all out of PP or "
                f"deal no damage. Party of {len(party)}."
            ),
            question=(
                "The party cannot deal damage, so it cannot win a battle or "
                "escape a trainer. Say how to restore PP and where."
            ),
            payload={
                "species": lead.get("species"),
                "moves": names,
                "party_size": len(party),
            },
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
        Toothless(),
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

    #: Detectors already answered once. They stop winning until the caller says
    #: their condition changed; see the note in :meth:`evaluate`.
    answered: set[str] = field(default_factory=set)

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

        window = list(receipts[-self.window :])
        candidates = [
            trigger
            for trigger in (d.check(window, state) for d in self.detectors)
            if trigger is not None
        ]

        # A detector that has gone quiet is no longer answered: if it comes back
        # it is a new episode and deserves a turn. See the note below for what
        # `answered` is for and what leaving it uncleared cost.
        #
        # This runs before the budget and cooldown gates on purpose. Whether a
        # condition has ended is a fact about the game, not about what we can
        # afford to say, and a condition that stopped and restarted inside one
        # cooldown would otherwise never be seen to have stopped. Detectors are
        # pure functions over a window of receipts, so asking them costs nothing
        # but arithmetic.
        self.answered &= {trigger.name for trigger in candidates}

        if not candidates:
            return None
        if len(self.fired) >= self.max_per_session:
            return None
        if self.fired and spent - self.fired[-1][0] < self.cooldown_presses:
            return None

        # A standing condition must not mask an episodic one. Measured over one
        # run: all 13 interventions fired on `low_hp`, because HP sat at 15-30%
        # for 59% of the run and `low_hp` outranks `circling`, which was firing in
        # 53 of the windows it lost. The detector aimed at the actual failure
        # never got a turn.
        #
        # So a detector that has already been answered stops winning until
        # something changes. Repeating it says nothing new: the thinker was told
        # about the low HP, it answered, and the HP is still low.
        #
        # "Until something changes" was left to the caller, and no caller was
        # ever written -- `clear_answered` had exactly one reference in the
        # whole repo, in a test. So every detector fired ONCE per session and
        # was silenced for the rest of it, whatever happened next. Measured
        # cost: the run spent 12,312 presses, 14.3% of its whole press total,
        # on 324 byte-identical batches at one tile, and `circling` would have
        # fired on that trivially. It had already been answered.
        #
        # A condition that stops and starts again is a new episode, and that
        # needs no caller to adjudicate: the detector itself has gone quiet.
        # The clearing happens above, before the no-candidates return.
        fresh = [t for t in candidates if t.name not in self.answered]
        return max(fresh or candidates, key=lambda t: t.priority)

    def record(self, trigger: Trigger, total_presses: int) -> None:
        self.fired.append((total_presses, trigger.name))
        self.answered.add(trigger.name)

    def clear_answered(self, name: str) -> None:
        """Let a detector win again, once its condition has actually changed.

        The caller decides what "changed" means, because only it can see the
        game: a party that healed, a map that was left, a command that finally
        worked.
        """
        self.answered.discard(name)

    def remaining(self) -> int:
        return max(0, self.max_per_session - len(self.fired))


# ---------------------------------------------------------------------------
# What the harness already knows
# ---------------------------------------------------------------------------
#
# A fact from the harness changes behaviour. A fact recalled by a model is
# frequently wrong, is stated with the same confidence as a true one, and is
# therefore worse than nothing. The critic was fixed this way — it reads run
# receipts instead of asking a model what happened — and the intervention
# prompt is fixed the same way here.
#
# The firing that forced it: `low_hp` at 10/65 on Route 4, answered with "walk
# west ~18 tiles to Vermilion City's east gate, then on to Celadon via Route
# 24". Vermilion is six maps away in the other direction, Route 24 is north of
# Cerulean, Route 4 is 90 tiles wide, and the run spent 146 seconds and 31
# presses walking back toward Mt Moon at 10 HP. The map graph had the answer
# already: Cerulean City is one hop east and has a Poke Center.
#
# Three rules keep this block honest:
#
# * **A hop is a plan, not a promise.** Route 4 is one map whose halves are
#   separated by Mt Moon, so "Cerulean is east" says nothing about whether you
#   can walk east from where you stand. Hop counts are labelled graph distance
#   and printed under INFERRED; tiles measured this frame are printed under
#   KNOWN. The two are never mixed into one sentence.
# * **Rank, do not choose.** The nearest Poke Center by hop count can be the
#   wrong one — Mt Moon Pokecenter is one hop from Route 4 and that hop walks
#   back into the cave the run just escaped. The thinker gets the ranked list
#   with distances and decides; the harness deciding for it is how a routing
#   table starts making strategy.
# * **Omit rather than guess.** Every fact here is dropped silently when the
#   data it needs is absent. A missing line costs a sentence; a wrong line cost
#   this run 31 presses in the wrong direction.

#: How much of the prompt the whole fact block may spend, headers included, in
#: characters — roughly 400 tokens, against a prompt that is otherwise about
#: 300. The value of a thinking session is a clean short problem, not a
#: briefing, so facts past the budget are dropped in the order
#: :func:`harness_facts` returns them rather than the prompt being allowed to
#: grow. Every line of the Route 4 firing this was built for fits inside it.
FACT_BUDGET_CHARS = 1600

#: Caps on the long lists, so one crowded map cannot eat the budget by itself.
#: Every one of them truncates *visibly*: the line that carries a cut list says
#: how many it is showing out of how many there are. A cap that elides in
#: silence under a header calling itself authoritative is worse than no line at
#: all, because the reader is told in the same breath that geography absent
#: from here is not known. S.S. Anne 2F Rooms has twelve warps and the player
#: was standing on the eleventh when this was measured; at six, the one exit
#: that mattered was inside the "and 6 more warps" the sentence swallowed.
MAX_EXITS_SHOWN = 12
MAX_CENTERS_SHOWN = 3
MAX_CENTER_ROUTES = 2
MAX_HOPS_SHOWN = 4
MAX_FRONTIER_SHOWN = 3
MAX_REPEAT_TILES = 3
MAX_ACTIONS_SHOWN = 8
MAX_ERROR_CHARS = 160

#: And a character ceiling on the joined list, because a count alone does not
#: bound the cost: names run from "Route 4" to "Fuchsia Bill's Grandpa's House"
#: and a step count rides on every measured exit. Measured over the whole
#: generated graph, 218 of the 221 maps with exits have twelve or fewer; with
#: every exit measured and a two-digit step count on each — the fattest the
#: line can ever be — 9 of those 221 still spill and are cut, and the cut says
#: so. The rest, which is 96% of the game's maps, carry the word "every"
#: honestly. Against `FACT_BUDGET_CHARS` that costs at worst one INFERRED fact
#: on the most crowded maps, and `format_facts` now names what it dropped.
MAX_EXITS_CHARS = 460

#: The button that leaves a map by each edge. North is up: ``walk_up``
#: decreases y, exactly as in :mod:`pokemon_agent.pathfinding`.
EDGE_BUTTON = {"north": "up", "south": "down", "east": "right", "west": "left"}


@dataclass(frozen=True)
class Fact:
    """One line of ground truth, and whether it was measured or inferred.

    ``known`` is the whole point of the type: a tile count read off this frame
    and a hop count read off the map graph are different kinds of claim, and
    printing them in one list is what lets a reader treat a plan as a promise.
    """

    text: str
    known: bool = True


def _is_poke_center(name: str) -> bool:
    lowered = name.lower()
    return "pokecenter" in lowered or "pokemon center" in lowered


def _standalone(haystack: str, start: int, length: int) -> bool:
    """Whether the slice at ``start`` is a whole word rather than part of one."""

    before = haystack[start - 1] if start else " "
    after = haystack[start + length] if start + length < len(haystack) else " "
    return not before.isalnum() and not after.isalnum()


def _load_world():
    from pokemon_agent.world import World

    return World.load()


class MapFacts:
    """The game's own map data, as answers a prompt can carry.

    Wraps :class:`pokemon_agent.world.World` — the same graph ``/route``
    answers from — and adds the two lookups an intervention needs and the HTTP
    surface has no verb for: which maps are Poke Centers, and which map a goal
    sentence is talking about.

    A missing or unreadable ``world.json`` yields an empty instance whose every
    method answers ``None`` or empty, so a checkout without generated data
    loses these lines from the prompt and nothing else.
    """

    def __init__(self, world: Any = None) -> None:
        self._world = world if world is not None else _load_world()
        self._names: tuple[str, ...] = self._all_names()
        self._centers: tuple[str, ...] = tuple(n for n in self._names if _is_poke_center(n))

    @classmethod
    def default(cls) -> "MapFacts":
        """The packaged map graph, read from disk once per process."""

        global _DEFAULT_MAPS
        if _DEFAULT_MAPS is None:
            _DEFAULT_MAPS = cls()
        return _DEFAULT_MAPS

    def _all_names(self) -> tuple[str, ...]:
        # A map named only as a hop target is a real destination even without a
        # record of its own, so targets count as names too.
        names: set[str] = set(self._world.map_names())
        for name in self._world.map_names():
            for hop in self._world.neighbours(name):
                names.add(hop.to_map)
        return tuple(sorted(names))

    @property
    def available(self) -> bool:
        return bool(self._names)

    def dimensions(self, map_name: Optional[str]) -> Optional[tuple[int, int]]:
        record = self._world.info(map_name) if map_name else None
        return record.size if record is not None else None

    def exits(self, map_name: Optional[str]) -> tuple:
        return self._world.neighbours(map_name) if map_name else ()

    def route(self, source: Optional[str], target: Optional[str]) -> Optional[tuple]:
        if not source or not target:
            return None
        return self._world.route(source, target)

    def reachable_poke_centers(self, source: Optional[str]) -> list[tuple]:
        """*Every* Poke Center the graph can route to from *source*, nearest first.

        The whole list, so a caller that shows three of eleven can say eleven.
        :meth:`poke_centers` is the same answer cut to a length a prompt can
        carry, and it is the cut that has to be declared.
        """

        if not source:
            return []
        ranked: list[tuple[int, str, tuple]] = []
        for name in self._centers:
            hops = self._world.route(source, name)
            if hops is None:
                continue
            ranked.append((len(hops), name, hops))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked

    def poke_centers(self, source: Optional[str], limit: int = MAX_CENTERS_SHOWN) -> list[tuple]:
        """Poke Centers by graph distance from *source*, nearest first.

        Distance is hops in the map graph and nothing more: it does not know
        whether the walk to the first exit is currently possible, which is why
        the caller prints it under INFERRED and hands over the whole ranking
        rather than a single recommendation.
        """

        return self.reachable_poke_centers(source)[:limit]

    def find_map(self, text: str) -> Optional[str]:
        """The last map this text names, or ``None`` if it names none.

        Last rather than first because an objective sentence puts its
        destination at the end — "cross Route 3 and Mt. Moon, and emerge into
        Cerulean City" is about Cerulean City. Whole words only, so a name
        never matches inside a longer one.
        """

        if not text:
            return None
        haystack = text.lower()
        best: Optional[str] = None
        best_at = -1
        for name in self._names:
            lowered = name.lower()
            at = haystack.rfind(lowered)
            while at != -1:
                if _standalone(haystack, at, len(lowered)):
                    if at > best_at or (at == best_at and len(name) > len(best or "")):
                        best, best_at = name, at
                    break
                at = haystack.rfind(lowered, 0, at)
        return best

    def names_in(self, text: str) -> tuple[str, ...]:
        """Every map this text names, in the order they appear.

        :meth:`find_map` answers "what is this sentence about", which is the
        wrong question for a paragraph of advice: a route names the map it
        starts on, the map it passes through and the map it ends on, and a
        checker has to know about all three. Whole words only, same as
        :meth:`find_map`, and a name inside a longer one never counts — "Route 2"
        must not match the "Route 2" inside "Route 24".
        """

        if not text:
            return ()
        haystack = text.lower()
        found: list[tuple[int, str]] = []
        for name in self._names:
            lowered = name.lower()
            at = haystack.find(lowered)
            while at != -1:
                if _standalone(haystack, at, len(lowered)):
                    found.append((at, name))
                    break
                at = haystack.find(lowered, at + 1)
        found.sort()
        return tuple(name for _, name in found)

    def resolve(self, fragment: str) -> Optional[str]:
        """The one map a written name means, or ``None`` when it means several.

        The archive is full of "Mt. Moon", which is four maps, and of "Mt Moon
        Poke Center", which is one map spelled three ways. Punctuation and
        spacing are normalised away; ambiguity is not. A fragment that could be
        several maps returns ``None`` and the caller leaves the claim alone,
        because refusing advice on a name the writer never disambiguated would
        be punishing shorthand rather than catching an error.
        """

        wanted = _normalise_map_name(fragment)
        if not wanted:
            return None
        exact = [name for name in self._names if _normalise_map_name(name) == wanted]
        if len(exact) == 1:
            return exact[0]
        starts = [name for name in self._names if _normalise_map_name(name).startswith(wanted)]
        return starts[0] if len(starts) == 1 else None


def _normalise_map_name(text: str) -> str:
    """A written map name reduced to what two spellings of it share."""

    lowered = (text or "").lower().replace(".", " ").replace("é", "e")
    lowered = lowered.replace("pokemon center", "pokecenter").replace("poke center", "pokecenter")
    return " ".join(lowered.split())


#: Loaded on first use, never at import time.
_DEFAULT_MAPS: Optional[MapFacts] = None


def _describe_hop(hop) -> str:
    """One hop as something a player can press, not as a graph edge."""

    if hop.kind == "connection" and hop.edge:
        button = EDGE_BUTTON.get(hop.edge)
        press = f" (walk_{button})" if button else ""
        return f"{hop.edge} edge{press} -> {hop.to_map}"
    if hop.at is not None:
        return f"warp ({hop.at[0]},{hop.at[1]}) -> {hop.to_map}"
    return f"warp -> {hop.to_map}"


def _describe_route(hops: Sequence[Any]) -> str:
    if not hops:
        return "you are already there"
    shown = [_describe_hop(hop) for hop in hops[:MAX_HOPS_SHOWN]]
    rest = len(hops) - len(shown)
    text = ", then ".join(shown)
    return f"{text}, then {rest} more hop{'s' if rest > 1 else ''}" if rest else text


def _coord_text(coord: Optional[Sequence[int]]) -> str:
    return f"({coord[0]},{coord[1]})" if coord is not None else "?"


def _observed(observation: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Pull the few live things a fact needs out of whatever was handed in.

    Accepts a runtime bundle, a navigation snapshot wrapped in ``snapshot``, or
    a flat snapshot — every shape this codebase already passes around — and
    answers with empty values for anything it cannot find.
    """

    source: Mapping[str, Any] = observation or {}
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, Mapping):
        navigation = source.get("navigation")
        if isinstance(navigation, Mapping):
            snapshot = navigation.get("snapshot")
    if not isinstance(snapshot, Mapping):
        snapshot = source if ("terrain" in source or "player_position" in source) else {}

    state = source.get("state")
    state = state if isinstance(state, Mapping) else {}
    explored = source.get("explored") or source.get("grid")

    # Two things call themselves the goal and they disagree often enough to be
    # worth trying in order: the operator's live instruction to the player, then
    # the objective ladder's own summary of the rung it is on.
    goals: list[str] = []
    goal = source.get("goal")
    if isinstance(goal, str) and goal.strip():
        goals.append(goal.strip())
    objective = source.get("objective")
    if isinstance(objective, Mapping):
        current = objective.get("current")
        # The frontier objective is a menu, not a journey, so the rule that
        # reads a destination out of a goal — the last map it names — has
        # nothing to read. It would return whichever open milestone happens to
        # sort last on the ladder and route to it, which is the harness picking
        # one of five for the model while calling it a fact about the goal.
        if isinstance(current, Mapping) and current.get("summary"):
            if current.get("pack_id") != FRONTIER_PACK_ID:
                goals.append(str(current["summary"]))

    position = None
    player = state.get("player")
    player = player if isinstance(player, Mapping) else {}
    for candidate in (player.get("position"), snapshot.get("player_position")):
        if isinstance(candidate, Mapping) and candidate.get("x") is not None:
            position = (int(candidate["x"]), int(candidate["y"]))
            break
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            position = (int(candidate[0]), int(candidate[1]))
            break

    map_info = state.get("map")
    map_info = map_info if isinstance(map_info, Mapping) else {}
    return {
        "snapshot": snapshot,
        "explored": explored if isinstance(explored, Mapping) else None,
        "goals": goals,
        "position": position,
        "map_name": map_info.get("map_name") or snapshot.get("map_name") or "",
    }


def standing_on(
    observation: Optional[Mapping[str, Any]] = None, recent: Sequence[Receipt] = ()
) -> str:
    """The map the player is on, from the live frame or, failing that, receipts.

    The same two sources :func:`harness_facts` resolves in the same order, named
    so a caller that only needs the map does not have to build a fact list to
    find out. ``""`` when neither source knows, which callers read as "cannot
    check this".
    """

    live = _observed(observation)["map_name"]
    if live:
        return str(live)
    return next((r.map_name for r in reversed(list(recent)) if r.map_name), "")


def _collision(snapshot: Mapping[str, Any], explored: Mapping[str, Any]):
    """Live walkability for this map, or ``None`` if it cannot be built."""

    try:
        from pokemon_agent import capabilities

        return capabilities.collision_from(dict(snapshot), dict(explored))
    except Exception:  # noqa: BLE001 — a fact that cannot be computed is omitted
        return None


def _walked(explored: Optional[Mapping[str, Any]]) -> Optional[set]:
    if not isinstance(explored, Mapping):
        return None
    walked = explored.get("walked")
    if not isinstance(walked, (set, frozenset, list, tuple)):
        return None
    return {tuple(tile) for tile in walked}


def _edge_steps(region, edge: Optional[str], size: Optional[tuple[int, int]]) -> Optional[int]:
    """Steps to the nearest tile on *edge*, which is where you walk off the map.

    An edge connection has no single warp tile — the whole side is the door —
    so its distance is the cheapest tile on that side the flood actually
    reached. Without the map's dimensions there is no way to say which tiles
    those are, and this answers ``None`` rather than picking the extreme of
    whatever the flood happened to cover.
    """

    if not edge or size is None:
        return None
    width, height = int(size[0]), int(size[1])
    limits = {
        "north": ("y", 0),
        "south": ("y", height - 1),
        "west": ("x", 0),
        "east": ("x", width - 1),
    }
    axis = limits.get(edge)
    if axis is None:
        return None
    index = 1 if axis[0] == "y" else 0
    costs = [cost for tile, cost in region.distance.items() if tile[index] == axis[1]]
    return min(costs) if costs else None


def _exit_steps(
    exits: Sequence[Any],
    size: Optional[tuple[int, int]],
    snapshot: Mapping[str, Any],
    explored: Optional[Mapping[str, Any]],
    position: Optional[Sequence[int]],
) -> dict[int, int]:
    """Walking steps from the player to each exit, keyed by index into ``exits``.

    One flood over the same collision :func:`_frontier_fact` is measured on, so
    this is real ground and not a straight line: a warp in another pocket of
    this map is a wall with a door painted on it, and it simply gets no number.
    :meth:`Region.approach` rather than ``steps_to`` because a doorway reads as
    blocked collision — you walk *into* it, you never stand on it.

    An exit missing from the result was not measured. That is a third state, not
    a synonym for unreachable: the flood only crosses ground somebody has looked
    at, so an exit behind unexplored floor is absent from here too, and the
    caller must not print either claim.
    """

    if position is None or not snapshot or not exits:
        return {}
    collision = _collision(snapshot, explored or {})
    if collision is None:
        return {}
    try:
        from pokemon_agent.world import reachable_region

        region = reachable_region(collision, (int(position[0]), int(position[1])))
    except Exception:  # noqa: BLE001 — a fact that cannot be computed is omitted
        return {}
    steps: dict[int, int] = {}
    for index, hop in enumerate(exits):
        if getattr(hop, "at", None) is None:
            cost = _edge_steps(region, getattr(hop, "edge", None), size)
            if cost is not None:
                steps[index] = cost
            continue
        try:
            found = region.approach((int(hop.at[0]), int(hop.at[1])))
        except Exception:  # noqa: BLE001
            continue
        if found is not None:
            steps[index] = found[1]
    return steps


def _exit_order(
    exits: Sequence[Any], steps: Mapping[int, int], position: Optional[Sequence[int]]
) -> tuple[list[int], Optional[int]]:
    """Indices into ``exits``, best first, and which one is under the player.

    The warp you are standing on goes first and nothing outranks it: the next
    button press takes it, so every other exit on the map is a longer answer to
    the same question. After it come the exits a flood measured, nearest first,
    and behind those the ones it could not measure, in the map data's own
    order — which is the order the whole list used to be in, and is meaningless.
    """

    here = (int(position[0]), int(position[1])) if position is not None else None
    underfoot: Optional[int] = None
    if here is not None:
        for index, hop in enumerate(exits):
            at = getattr(hop, "at", None)
            if at is not None and (int(at[0]), int(at[1])) == here:
                underfoot = index
                break
    order = sorted(
        range(len(exits)),
        key=lambda index: (
            index != underfoot,
            index not in steps,
            steps.get(index, 0),
            index,
        ),
    )
    return order, underfoot


def _exits_fact(
    map_name: str,
    exits: Sequence[Any],
    steps: Mapping[int, int],
    position: Optional[Sequence[int]],
) -> Fact:
    """Where this map leads, best first, and honestly about what it left out.

    The sentence used to open "Every exit from ..." and then end "and 6 more
    warps", under a header that tells the reader geography it does not carry is
    not known. On S.S. Anne 2F Rooms — twelve warps, player standing on the
    eleventh — that sentence named six exits, none of them the one underfoot.
    "Every" is now said only when every exit is in the line; a cut list gives
    the true total instead, and "nearest first" is claimed only over the exits a
    flood actually measured, never over the ones it did not.
    """

    order, underfoot = _exit_order(exits, steps, position)
    parts: list[str] = []
    for index in order:
        text = _describe_hop(exits[index])
        if index == underfoot:
            text += " [you are standing on this one]"
        elif index in steps:
            cost = steps[index]
            text += f", {cost} step{'' if cost == 1 else 's'}"
        parts.append(text)

    shown = parts[:MAX_EXITS_SHOWN]
    while len(shown) > 1 and len("; ".join(shown)) > MAX_EXITS_CHARS:
        shown.pop()
    if len(shown) == len(parts):
        lead = f"Every exit from {map_name}"
    else:
        lead = f"{len(shown)} of {len(parts)} exits from {map_name}"
    if steps:
        measured = (
            "nearest first" if len(steps) == len(exits) else "measured exits first, nearest first"
        )
        lead += f" ({measured})"
    return Fact(f"{lead}: {'; '.join(shown)}.")


def _frontier_fact(
    snapshot: Mapping[str, Any],
    explored: Optional[Mapping[str, Any]],
    position: Optional[Sequence[int]],
) -> Optional[Fact]:
    """Reachable ground nobody has walked on, over live collision.

    Answers "where have I not looked" instead of leaving it to be guessed. It
    needs both halves — the frame and the map store — so it is omitted when
    either is missing rather than computed over a 10x9 window and reported as
    if it were the map.
    """

    walked = _walked(explored)
    if not snapshot or walked is None or position is None:
        return None
    collision = _collision(snapshot, explored or {})
    if collision is None:
        return None
    try:
        from pokemon_agent import world as world_mod

        detail = world_mod.frontier_detail(collision, walked, (int(position[0]), int(position[1])))
    except Exception:  # noqa: BLE001
        return None
    if not detail:
        return Fact(
            "Reachable ground on this map you have never walked on: none. "
            "Everything you can reach from here has already been walked."
        )
    confirmed = sum(1 for tile in detail if tile.certain)
    nearest = ", ".join(_coord_text(tile.coord) for tile in detail[:MAX_FRONTIER_SHOWN])
    return Fact(
        f"Reachable ground on this map you have never walked on: {len(detail)} tiles, "
        f"nearest {nearest} ({confirmed} confirmed by this frame, "
        f"{len(detail) - confirmed} only remembered)."
    )


def _repeated_tiles(recent: Sequence[Receipt], map_name: str) -> list[tuple]:
    """Tiles stood on more than once on this map, most stood on first.

    Uncapped, because the caller has to know how many there are before it can
    say honestly how many it is showing. Once only is not "repeated": the
    circling detector reads a whole window across every map, this line reads one
    map, and a player who has just walked onto a fresh one has every tile at x1.
    The captured S.S. Anne payload said "Tiles you keep standing on: (22,15) x1"
    about a tile the player had stood on exactly once.
    """

    counts = Counter(r.pos for r in recent if r.pos is not None and r.map_name == map_name)
    return [(pos, count) for pos, count in counts.most_common() if count > 1]


def _repeat_fact(
    recent: Sequence[Receipt], map_name: str
) -> tuple[Optional[Fact], Optional[tuple]]:
    """The tiles being stood on over and over, and the most stood-on tile.

    The tile comes back separately from the sentence, and survives it: the
    caller's next fact — which step off that tile has never been walked — is
    worth having even on a map where nothing has been stood on twice yet, and
    re-deriving the tile from the sentence would be reading the prompt back.
    """

    counts = Counter(r.pos for r in recent if r.pos is not None and r.map_name == map_name)
    if not counts:
        return None, None
    worst = counts.most_common(1)[0][0]
    repeated = [(pos, count) for pos, count in counts.most_common() if count > 1]
    if not repeated:
        return None, worst
    shown = repeated[:MAX_REPEAT_TILES]
    spread = ", ".join(f"{_coord_text(pos)} x{count}" for pos, count in shown)
    if len(shown) == len(repeated):
        lead = "Tiles you keep standing on"
    else:
        lead = f"{len(shown)} of {len(repeated)} tiles you keep standing on, worst first"
    return Fact(f"{lead}: {spread}."), worst


def _neighbour_fact(
    tile: Optional[Sequence[int]],
    snapshot: Mapping[str, Any],
    explored: Optional[Mapping[str, Any]],
) -> Optional[Fact]:
    """Which step off the most-repeated tile has never been walked."""

    walked = _walked(explored)
    if tile is None or not snapshot or walked is None:
        return None
    collision = _collision(snapshot, explored or {})
    if collision is None:
        return None
    try:
        from pokemon_agent.pathfinding import DIRECTIONS
    except Exception:  # noqa: BLE001
        return None
    walkable = set(collision.get("walkable") or ())
    sprites = set(collision.get("sprites") or ())
    here = (int(tile[0]), int(tile[1]))
    fresh = [
        f"walk_{direction} to {_coord_text((here[0] + dx, here[1] + dy))}"
        for direction, (dx, dy) in DIRECTIONS.items()
        if (here[0] + dx, here[1] + dy) in walkable
        and (here[0] + dx, here[1] + dy) not in sprites
        and (here[0] + dx, here[1] + dy) not in walked
    ]
    if not fresh:
        return Fact(
            f"Every walkable neighbour of {_coord_text(here)} has been walked already, "
            "so the way on is not next to that tile."
        )
    return Fact(f"Unwalked walkable neighbours of {_coord_text(here)}: {'; '.join(fresh)}.")


def _failure_fact(recent: Sequence[Receipt]) -> Optional[Fact]:
    """The error text itself, which is the one thing a retry loop never reads.

    Both halves say when they were cut. A server refusal often puts the part
    that matters last — "walk_left is blocked by a wall, try walk_up" — so an
    error clipped in silence at 160 characters reads as a complete sentence that
    simply did not explain itself.
    """

    for receipt in reversed(recent):
        if not receipt.exit_code:
            continue
        extra = receipt.extra or {}
        error = str(extra.get("error") or "").strip()
        actions = extra.get("actions")
        parts: list[str] = []
        if error:
            tool = receipt.tool or "the command"
            clipped = error[:MAX_ERROR_CHARS]
            cut = (
                f" [first {MAX_ERROR_CHARS} of {len(error)} characters]"
                if len(error) > MAX_ERROR_CHARS
                else ""
            )
            parts.append(f"`{tool}` failed with: {clipped}{cut}")
        if isinstance(actions, (list, tuple)) and actions:
            shown = [str(action) for action in actions[:MAX_ACTIONS_SHOWN]]
            label = (
                "actions sent"
                if len(shown) == len(actions)
                else f"first {len(shown)} of {len(actions)} actions sent"
            )
            parts.append(f"{label}: " + " ".join(shown))
        return Fact(". ".join(parts) + ".") if parts else None
    return None


def harness_facts(
    trigger: Trigger,
    *,
    recent: Sequence[Receipt] = (),
    observation: Optional[Mapping[str, Any]] = None,
    maps: Optional[MapFacts] = None,
    goal: str = "",
) -> list[Fact]:
    """Everything the harness can answer about where the player is, right now.

    Ordered by how much the answer depends on it, because the budget drops from
    the end: where you are, what leads off this map, whatever the trigger makes
    urgent, then the routes and the unwalked ground. Nothing here raises — a
    fact that cannot be computed is simply not in the list.
    """

    maps = maps if maps is not None else MapFacts.default()
    live = _observed(observation)
    map_name = live["map_name"] or next((r.map_name for r in reversed(recent) if r.map_name), "")
    position = live["position"] or next((r.pos for r in reversed(recent) if r.pos), None)
    goals = ([goal] if goal else []) + list(live["goals"])

    known: list[Fact] = []
    inferred: list[Fact] = []

    size = maps.dimensions(map_name)
    if size is not None:
        line = f"{map_name} is {size[0]} tiles wide and {size[1]} tall."
        if position is not None and 0 <= position[0] < size[0] and 0 <= position[1] < size[1]:
            line += (
                f" You are at {_coord_text(position)}: {position[0]} tiles from its west edge, "
                f"{size[0] - 1 - position[0]} from its east, {position[1]} from its north, "
                f"{size[1] - 1 - position[1]} from its south."
            )
        known.append(Fact(line))

    exits = maps.exits(map_name)
    if exits:
        steps = _exit_steps(exits, size, live["snapshot"], live["explored"], position)
        known.append(_exits_fact(map_name, exits, steps, position))

    if trigger.name == "repeated_failure":
        failure = _failure_fact(recent)
        if failure is not None:
            known.append(failure)

    if trigger.name == "circling":
        repeat, worst = _repeat_fact(recent, map_name)
        if repeat is not None:
            known.append(repeat)
        neighbours = _neighbour_fact(worst, live["snapshot"], live["explored"])
        if neighbours is not None:
            known.append(neighbours)

    centers = maps.reachable_poke_centers(map_name)
    if centers:
        shown = centers[:MAX_CENTERS_SHOWN]
        parts = []
        for index, (distance, name, hops) in enumerate(shown):
            plural = "" if distance == 1 else "s"
            detail = f" [{_describe_route(hops)}]" if index < MAX_CENTER_ROUTES else ""
            parts.append(f"{name} {distance} hop{plural}{detail}")
        lead = (
            f"Poke Centers from {map_name}, nearest first"
            if len(shown) == len(centers)
            else f"{len(shown)} of {len(centers)} Poke Centers from {map_name}, nearest first"
        )
        inferred.append(Fact(f"{lead}: {'; '.join(parts)}.", known=False))

    destination = next((found for found in (maps.find_map(text) for text in goals) if found), None)
    if destination:
        hops = maps.route(map_name, destination)
        if hops is not None:
            plural = "" if len(hops) == 1 else "s"
            inferred.append(
                Fact(
                    f"The objective names {destination}: {len(hops)} hop{plural} from "
                    f"{map_name} — {_describe_route(hops)}.",
                    known=False,
                )
            )

    frontier = _frontier_fact(live["snapshot"], live["explored"], position)
    if frontier is not None:
        known.append(frontier)

    return known + inferred


#: Said once, in the prompt, because the failure this block exists to fix was a
#: confident recollection beating a computed answer.
FACTS_HEADER = (
    "HARNESS FACTS — computed just now from the game's own map data and the "
    "live frame. These are authoritative. Your own recollection of Pokemon Red "
    "is not: where it disagrees with these lines, it is wrong. Geography that "
    "is not written here is not known, so do not supply it from memory."
)

#: What separates a tile counted on this frame from a hop counted on the graph.
FACTS_KNOWN_HEADER = "KNOWN — read from the game's data or measured on this frame:"

#: The line that stops a hop count being read as a walking route.
FACTS_INFERRED_HEADER = (
    "INFERRED — map-graph distance. The graph knows which maps touch, not "
    "whether the ground between you and that exit is walkable: one map can be "
    "split into halves that do not connect. A hop count is a plan to check, "
    "never a promise, and never a number of tiles."
)


def format_facts(facts: Sequence[Fact], *, budget: int = FACT_BUDGET_CHARS) -> str:
    """The fact block, split by what was measured and what was inferred.

    ``budget`` is the size of the rendered block, headers and all, and is held
    to exactly: facts are dropped from the end once it is spent, which is the
    whole reason :func:`harness_facts` returns them in the order it does. An
    empty list gives an empty string — no block, no header, no apology.

    A drop is *said*, and the room to say it is reserved before the last fact
    is fitted. The header above this block tells the reader that geography not
    written here is not known; a block that quietly loses its whole INFERRED
    section turns that into "the harness has nothing else", which is a
    different and false claim. One line, and the reader knows to ask.
    """

    known, inferred, dropped = _fit_facts(facts, budget)
    note = ""
    if dropped:
        # Reserve room for the note, then refit inside what is left. Two passes
        # at most: the reservation only grows by the digits in the count. A
        # budget too small to hold both one fact and the note keeps the fact —
        # at that size there is nothing to qualify.
        for _ in range(3):
            candidate = _overflow_note(dropped)
            fitted = _fit_facts(facts, budget - len(candidate) - 2)
            if not (fitted[0] or fitted[1]):
                break
            known, inferred, dropped = fitted
            note = _overflow_note(dropped)
            if candidate == note:
                break
    if not known and not inferred:
        return ""

    lines = [FACTS_HEADER]
    if known:
        lines += ["", FACTS_KNOWN_HEADER]
        lines += [f"  - {_one_line(fact.text)}" for fact in known]
    if inferred:
        lines += ["", FACTS_INFERRED_HEADER]
        lines += [f"  - {_one_line(fact.text)}" for fact in inferred]
    if note:
        lines += ["", note]
    return "\n".join(lines)


def _one_line(text: str) -> str:
    """*text* with every line break flattened into a space.

    A fact is rendered as one bullet under a header that calls this block
    authoritative, and a fact carrying a newline stops being one bullet: the
    tail lands in the left margin, where it reads as another line the harness
    computed. Most facts are built from the map graph and cannot contain one.
    ``_failure_fact`` can — it quotes the server's error text and the action
    strings the player model sent, and a receipt written for a refused batch
    stores those verbatim — so a batch containing "walk_up\\n  - Cerulean City
    is 1 step to the west." puts that sentence under KNOWN in the harness's own
    voice. Flattened here rather than at each fact, because the guarantee is
    about how the block renders and every fact goes through this one place.
    """

    return " ".join((text or "").split())


def _overflow_note(dropped: int) -> str:
    """What the block says when it could not carry everything it computed."""

    if dropped == 1:
        return "(1 further computed fact did not fit in this block and was left out.)"
    return f"({dropped} further computed facts did not fit in this block and were left out.)"


def _fit_facts(facts: Sequence[Fact], budget: int) -> tuple[list[Fact], list[Fact], int]:
    """Split ``facts`` into the two sections at this budget, and count the losses."""

    known: list[Fact] = []
    inferred: list[Fact] = []
    dropped = 0
    spent = len(FACTS_HEADER)
    for fact in facts:
        section = known if fact.known else inferred
        header = 0 if section else len(FACTS_KNOWN_HEADER if fact.known else FACTS_INFERRED_HEADER)
        cost = len(fact.text) + 5 + (header + 2 if header else 0)
        if spent + cost > budget:
            dropped += 1
            continue
        section.append(fact)
        spent += cost
    return known, inferred, dropped


def _indented(text: str) -> list[str]:
    """*text* as lines the harness's own section headers cannot be confused with.

    The state summary is built by the server as ``key: value`` lines, and one of
    those values is ``screen_text``: up to 160 characters decoded straight off
    the tile map, newlines and all. It is the only thing in this prompt the game
    is the author of, and a Pokemon's nickname is part of it, so it is the only
    part a player could have chosen the words of.

    It cannot reproduce ``FACTS_HEADER`` — that is 300 characters with an em
    dash in it, and neither the length nor the character is available to a Gen 1
    dialog box. It can trivially reproduce ``PROGRESS`` or ``YOUR TASK``, which
    are short, plain and upper case, and a two-line dialog already breaks the
    one-key-per-line reading of the block whatever it says. So column zero is
    reserved for the harness: every section header in this prompt starts there
    and nothing quoted from the game does.
    """

    return [f"  {line}" for line in (text or "").strip().splitlines()]


def build_prompt(
    trigger: Trigger,
    *,
    state_summary: str,
    recent: Sequence[Receipt],
    milestone_summary: str = "",
    limit: int = 20,
    observation: Optional[Mapping[str, Any]] = None,
    maps: Optional[MapFacts] = None,
    goal: str = "",
    facts: Optional[Sequence[Fact]] = None,
    fact_budget: int = FACT_BUDGET_CHARS,
) -> str:
    """The whole context a thinking session gets.

    Deliberately small. The point of swapping is that the thinker sees a clean,
    short problem instead of the player's hundred thousand tokens of history —
    and the fact block is capped at ``fact_budget`` characters for the same
    reason, so grounding the prompt cannot quietly turn it into a briefing.

    ``observation`` is whatever live state the caller has: a runtime bundle, a
    navigation snapshot, or nothing at all. With it, the facts include the
    unwalked ground reachable from here; without it, they are the map graph and
    the receipts alone. Passing ``facts`` explicitly skips the computation, and
    passing an empty sequence removes the block entirely.
    """

    if facts is None:
        facts = harness_facts(trigger, recent=recent, observation=observation, maps=maps, goal=goal)

    lines = [
        "You are advising a model that is playing Pokemon Red and has stopped "
        "making progress. It cannot see this conversation; your answer will be "
        "handed to it as a single instruction.",
        "",
        f"WHY YOU WERE CALLED: {trigger.reason}",
        "",
        "CURRENT STATE",
        *_indented(state_summary),
    ]
    block = format_facts(facts, budget=fact_budget)
    if block:
        lines += ["", block]
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
        "Answer in at most 150 words. Not strategy: say where to go and how to "
        "get there. Every claim you make about where something is must come "
        "from the facts above — if they do not cover it, say what to check "
        "rather than recalling it.",
        "",
        # Counted button runs are what this prompt used to ask for, and they are
        # the wrong shape for the failure it is called on. "Press down 8 tiles"
        # dies at the first blocked tile and leaves the player facing a wall
        # with seven presses of plan left; three consecutive interventions gave
        # exactly that advice and the player came out of each one oscillating
        # against a different dead end. `goto` re-plans on live collision, which
        # a counted run cannot, and it was used once in 2,314 presses while the
        # advice channel taught the alternative.
        "The player has `poke goto <map>` and `poke goto <x> <y>`, which walks "
        "toward a target and re-plans against real collision each map, and "
        "stops and says why if it cannot get there. Prefer naming the "
        "destination and letting it walk. Counted runs of presses are for short "
        "moves where you can see the whole path: a blocked tile ends a counted "
        "run and leaves the rest of it wrong.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checking the answer, not just grounding the question.
#
# `harness_facts` grounds the prompt. Nothing used to check the reply, and the
# reply is what reaches the player. Measured over the 35 interventions this
# project has delivered (runs/20260825T224823Z-983b, interventions.jsonl):
#
#   * the 13 delivered before the facts block existed named 12 places that are
#     not reachable the way they said — "exit Mt Moon 1F to Route 2 and walk
#     west to Viridian City" (Mt Moon 1F's only exits are two warps to Route 4),
#     "the Mt Moon B1F elevator" (there is none), "walk south on Route 3 to
#     Viridian City" (Route 3 has no south edge);
#   * the 22 delivered after it named real warps and real edges, every time.
#
# So the facts block did the heavy lifting and this is the backstop for the
# next model that answers from a walkthrough anyway. It refuses only what the
# map data contradicts outright, plus one distance rule that is policy and says
# so, because dropping a correct message costs a detector firing and dropping a
# wrong one saves several hundred presses.
# ---------------------------------------------------------------------------

#: How far ahead one message is allowed to point. Not a truth test — a scope
#: test. Every message delivered after the facts block landed named nothing
#: further than 3 hops off; the ungrounded ones before it averaged 5.4 and
#: reached 8. A message naming somewhere 8 hops away is not steering the next
#: few hundred presses, it is reciting a walkthrough.
MAX_ADVICE_HOPS = 3

#: "(14,35) -> Route 4", "warp (11,5) leads into the Mt Moon Pokecenter".
#: The gap excludes "(" and ":" so a list — "(11,5), (18,5), (24,5): they warp
#: into Mt Moon" — never binds its first tile to the destination of its last.
#: Three tiles and one phrase is not a claim about any one of them.
#:
#: It also stops at a conjunction, because a conjunction hands the verb to a
#: new subject and the tile is no longer what the sentence is about. Both
#: refusals this run were that mistake and both threw away a correct answer:
#: "(20,19) is the loop and the east edge leads to Route 11" was read as a
#: claim that (20,19) leads to Route 11, and refused for it. The east edge does
#: lead to Route 11 — the facts block says so three lines further up. A comma
#: ends the gap for the same reason, except before "which" or "that", where the
#: relative pronoun keeps the tile as the subject: "(11,5), which leads to the
#: Mt Moon Pokecenter" is still a claim about (11,5).
#:
#: Measured over all 57 answers the run has produced: 23 of the 27 matches
#: survive, and the four dropped are the two false refusals plus two fragments
#: that were never warp claims at all ("(12,19) and enter the gym").
_ADVICE_WARP_RE = re.compile(
    r"\((\d+)\s*,\s*(\d+)\)"
    r"(?:,\s*(?:which|that)\b|(?!\b(?:and|but|or|then|while|because)\b)[^.;:,\n(])"
    r"{0,60}?"
    r"(?:->|→|warps?\s+(?:you\s+)?(?:to|into)|leads?\s+(?:to|into)|"
    r"takes\s+you\s+(?:to|into)|enters?)\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z0-9.']*(?:\s+[A-Za-z0-9.'][A-Za-z0-9.']*){0,3})",
    re.IGNORECASE,
)

_ADVICE_COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

#: Leaving this map for a named one: "the door to Route 2 is there", "you
#: emerge on Route 1". Where a map's exits go is the one thing the game data
#: knows completely, so a claim about them is provable either way — which is
#: what makes this the rule a handoff is checked by, where the hop ceiling is
#: lifted and a retrospective is allowed to name the far end of the run.
_ADVICE_EXIT_RE = re.compile(
    r"\b(exits?|exiting|leaves?|leaving|emerges?|outside|door|doorway)\b"
    r"[^.;:\n]{0,30}?\b(?:to|on|onto|into)\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z0-9.']*(?:\s+[A-Za-z0-9.'][A-Za-z0-9.']*){0,3})",
    re.IGNORECASE,
)

#: A compass word aimed at a named map: "continue north toward Viridian City".
#: Deliberately tight. The first version of this rule took every compass word
#: and every map name within 60 characters of it, and on the archive that read
#: "only 'down' and 'left' are available, so the east exit to Cerulean City is
#: impossible" as a claim that Cerulean is west — 13 refusals on 22 messages
#: that were right. Advice is prose; a compass word only binds to a
#: destination when it is written as one, so the preposition and the name have
#: to follow the direction with nothing but the step between them.
_ADVICE_HEADING_RE = re.compile(
    r"\b(north|south|east|west|up|down|left|right)\b"
    r"[^.;:\n]{0,25}?\b(?:to|toward|towards|into|onto)\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z0-9.']*(?:\s+[A-Za-z0-9.'][A-Za-z0-9.']*){0,3})",
    re.IGNORECASE,
)

#: The compass word each direction can be written as. "left" and "west" are the
#: same claim; the model writes both, sometimes in the same sentence.
_COMPASS_EDGE = {
    "north": "north",
    "up": "north",
    "south": "south",
    "down": "south",
    "east": "east",
    "right": "east",
    "west": "west",
    "left": "west",
}


@dataclass(frozen=True)
class FalseClaim:
    """One sentence the map data contradicts, and what it says instead."""

    kind: str  # "warp" | "bounds" | "edge" | "exit" | "distance"
    said: str
    truth: str

    def __str__(self) -> str:
        return f"{self.said} — {self.truth}"


def _named_at(maps: MapFacts, fragment: str) -> Optional[str]:
    """The map a captured fragment starts with, or ``None``.

    The patterns grab a few words after "to" or "->" because a map name is one
    to four of them and the sentence does not stop there: "to Route 2 and walk"
    is what the regex hands over. Longest prefix first, so "Mt Moon Pokecenter"
    is never mistaken for the four maps "Mt Moon" could mean.
    """

    words = (fragment or "").split()
    for count in range(len(words), 0, -1):
        found = maps.resolve(" ".join(words[:count]))
        if found is not None:
            return found
    return None


def _clause_before(text: str, at: int) -> str:
    """The run of text back to the last clause break before *at*.

    A colon is not a break here. "On Route 4: head for the east edge, which
    exits into Cerulean City" is one clause with one subject, and cutting at the
    colon throws away the only word that says which map is being described.
    """

    start = max(text.rfind(mark, 0, at) for mark in (".", ";", "\n"))
    return text[start + 1 : at]


def _warp_targets(maps: MapFacts, map_name: str) -> dict:
    """``{(x, y): {destination, ...}}`` for one map's warps."""

    out: dict = {}
    for hop in maps.exits(map_name):
        if hop.at is not None:
            out.setdefault((int(hop.at[0]), int(hop.at[1])), set()).add(hop.to_map)
    return out


def _edges_to(maps: MapFacts, map_name: str) -> dict:
    """``{neighbour: edge}`` for the maps this one touches along a side."""

    return {
        hop.to_map: hop.edge
        for hop in maps.exits(map_name)
        if hop.kind == "connection" and hop.edge
    }


def check_advice(
    text: str,
    *,
    here: str,
    maps: Optional[MapFacts] = None,
    max_hops: int = MAX_ADVICE_HOPS,
) -> tuple[FalseClaim, ...]:
    """Every claim in *text* the game's own map data disproves.

    *here* is the map the player is standing on. Advice is a plan, not a
    sentence about one map, so a coordinate or a warp is checked against every
    map the text names as well as against *here*: "on Route 4, the warp at
    (11,5) enters the Mt Moon Pokecenter" is true even when it is written by
    someone standing in Mt Moon 1F, and refusing it would be a bug in the
    checker rather than a catch.

    An empty result means nothing was disproved, which is not the same as
    everything being right — a walk this cannot check is a walk it leaves
    alone. Nothing here raises: a checkout without generated map data checks
    nothing and says so by returning ``()``.
    """

    maps = maps if maps is not None else MapFacts.default()
    if not maps.available or not text:
        return ()

    scope = [here] if here else []
    scope += [name for name in maps.names_in(text) if name not in scope]
    if not scope:
        return ()

    claims: list[FalseClaim] = []
    seen: set[tuple[str, str]] = set()

    def add(claim: FalseClaim) -> None:
        key = (claim.kind, claim.said)
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    # A warp tile is checked against the maps on the route, not only the maps
    # the message spelled out in full. Intervention 34 wrote "on B1F, go to
    # warp (27,3) -> Route 4" — true of Mt Moon B1F, which it named as "B1F",
    # and refusing it for the abbreviation would be catching a typo, not an
    # error. One hop off a named map is still somewhere the advice is about.
    reachable = set(scope)
    for name in scope:
        reachable.update(hop.to_map for hop in maps.exits(name))
    warps = {name: _warp_targets(maps, name) for name in sorted(reachable)}

    for match in _ADVICE_WARP_RE.finditer(text):
        tile = (int(match.group(1)), int(match.group(2)))
        destination = _named_at(maps, match.group(3))
        if destination is None:
            continue  # A name that means several maps is not a checkable claim.
        if any(destination in warps[name].get(tile, ()) for name in warps):
            continue
        elsewhere = sorted(
            f"{name} {_coord_text(tile)} goes to {dest}"
            for name in warps
            for dest in warps[name].get(tile, ())
        )
        truth = (
            "; ".join(elsewhere)
            if elsewhere
            else f"no map on this route has a warp on {_coord_text(tile)}"
        )
        add(FalseClaim("warp", match.group(0).strip(), truth))

    sizes = {name: maps.dimensions(name) for name in scope}
    if any(size is not None for size in sizes.values()):
        for match in _ADVICE_COORD_RE.finditer(text):
            tile = (int(match.group(1)), int(match.group(2)))
            fits = [
                name
                for name, size in sizes.items()
                if size is not None and tile[0] < size[0] and tile[1] < size[1]
            ]
            if fits:
                continue
            add(
                FalseClaim(
                    "bounds",
                    _coord_text(tile),
                    "off the edge of every map this message names ("
                    + "; ".join(
                        f"{name} is {size[0]}x{size[1]}"
                        for name, size in sizes.items()
                        if size is not None
                    )
                    + ")",
                )
            )

    # Only the edges of the map the player is standing on. A destination the
    # text merely passes through has its own edges, and judging a sentence
    # about Mt Moon 1F by Cerulean City's edge table is how the first version
    # of this rule refused a dozen correct messages.
    edges = _edges_to(maps, here) if here else {}
    for match in _ADVICE_HEADING_RE.finditer(text):
        edge = _COMPASS_EDGE[match.group(1).lower()]
        named = _named_at(maps, match.group(2))
        # No edge to this map from here means the walk is not the claim being
        # made — you get there by warp, or by way of somewhere else — and a
        # direction that claims nothing cannot be wrong.
        actual = edges.get(named) if named else None
        if actual is None or actual == edge:
            continue
        add(
            FalseClaim(
                "edge",
                match.group(0).strip(),
                f"{named} is off the {actual} edge of {here}, not the {edge}",
            )
        )

    if here:
        onward = {hop.to_map for hop in maps.exits(here)}
        for match in _ADVICE_EXIT_RE.finditer(text):
            named = _named_at(maps, match.group(2))
            if named is None or named == here or named in onward:
                continue
            # Whose exits are being described. "On Route 4: head for the east
            # edge, which exits right into Cerulean City" is a true sentence
            # written by someone standing on Route 3, and judging it by Route
            # 3's exits refuses it. Where the clause names its own map, that
            # map is the subject and this rule has nothing to say.
            clause = _clause_before(text, match.start())
            subject = maps.names_in(clause)
            if subject and subject[-1] != here:
                continue
            add(
                FalseClaim(
                    "exit",
                    match.group(0).strip(),
                    f"{here} leads to {', '.join(sorted(onward))} and nowhere else",
                )
            )

    if here:
        for named in maps.names_in(text):
            if named == here:
                continue
            route = maps.route(here, named)
            if route is None:
                add(
                    FalseClaim(
                        "distance",
                        named,
                        f"the map graph cannot reach it from {here} at all",
                    )
                )
            elif len(route) > max_hops:
                add(
                    FalseClaim(
                        "distance",
                        named,
                        f"{len(route)} hops from {here}, past the {max_hops} "
                        "one message is allowed to point",
                    )
                )

    return tuple(claims)


def refusal_note(claims: Sequence[FalseClaim]) -> str:
    """Why a message was not delivered, short enough to sit in a journal line."""

    return "map data contradicts: " + "; ".join(str(claim) for claim in claims[:4])


# ---------------------------------------------------------------------------
# Striking part of a message instead of dropping all of it
#
# `check_advice` says what is wrong; it does not say what to do about it, and
# the caller did the bluntest possible thing — one disproved claim dropped the
# whole answer. Intervention 61 (2026-08-27T15:08:03Z) is the case that settled
# it: a plan to walk off Route 4's east edge into Cerulean City, correct in
# every particular except the waypoint "(90,14)", one tile past a map that is
# 90 wide. A thinking session was spent, the answer was binned over the
# off-by-one, and the player went on oscillating for another 400 presses.
#
# Interventions are numbered by their line in `interventions.jsonl` here. The
# measurements below are that file at 2026-08-27T15:49Z: 64 firings, 51 of
# which produced an answer, 10 of those with something the map data disproves.
#
# `critic.strike_false_claims` already does the proportionate thing for a
# handoff: cut the sentences, keep the rest. Advice needs two rules a handoff
# does not, and that is why this lives here rather than being borrowed:
#
#   * A plan is enumerated and a hole in it is followed literally. Striking
#     the middle sentence of "3. Press right 24 tiles -> (90,14). You cross the
#     east edge into Cerulean City." leaves a step that promises an arrival and
#     names no button. Whole steps go, never part of one.
#   * A plan that loses most of itself is not that plan minus a detail. Over
#     the 10 answers in the archive the map data contradicts, the survivor is
#     either most of the message (67-91%, eight of them) or a third of it
#     (39% for intervention 7, 35% for 22) — and both of those leftovers tell
#     the player to arrive somewhere the struck part was the route to: "Go to
#     the Pokemon Center in the top of town" with the town gone. Nothing lands
#     between the two groups, so half is the line.
#
# What survives is delivered under a header that quotes what went and what the
# map says instead. That is not politeness: a payload that elides a fact while
# reading as complete is the failure this project has paid for twice, and a
# quoted claim with its refutation beside it is strictly more than the player
# had either way — the original message told it to walk to a tile that does
# not exist, and deletion alone would have left it wondering where step 3 was.
# ---------------------------------------------------------------------------

#: Sentence ends, inside a line. Lines stay whole so a step keeps its number
#: and a paragraph break survives a strike from the line above it. Same rule as
#: `critic._SENTENCE_RE`, for the same reason.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

#: A line the player will follow as one instruction: "3. Press right", "3)
#: Press right", "Step 3: press right", "- press right". The bullet forms need
#: the space, or the markdown the thinker writes in — "**Commit to the east
#: exit**" — reads as a bullet and gets struck whole.
_ENUMERATED_RE = re.compile(r"^\s*(?:(?:step\s+)?(\d+)\s*[.):]|[-*+•]\s)", re.IGNORECASE)

#: How much of a message has to survive the striking for the remainder to be
#: worth delivering. Measured, not chosen: see the comment above.
MIN_SURVIVING_SHARE = 0.5

#: What the header may cost. An answer is cut to ``ANSWER_LIMIT`` — 1200
#: characters — because past that the player skims it, and a header that
#: quotes four long steps back at it would be most of that budget. So the
#: quotes are spelled out until the room runs out and the rest is counted:
#: the line that says which steps are missing is never the part that is
#: dropped, because that is the line the payload would otherwise be lying by.
NOTICE_LIMIT = 500
MAX_NOTICE_QUOTE = 160

#: How many claims a journal line names, the same count :func:`refusal_note`
#: uses for the same reason.
MAX_NOTE_CLAIMS = 4


@dataclass(frozen=True)
class Strike:
    """One piece of a message that was removed, and what disproved it."""

    #: What was cut, as the message wrote it: one sentence, or a whole
    #: enumerated step when the sentence was inside one. A step's number is not
    #: repeated here — it is in ``label``, where the notice can name it.
    text: str
    #: "step 3" where the message numbered it, "" for plain prose.
    label: str
    claims: tuple[FalseClaim, ...] = ()

    def __str__(self) -> str:
        return f"{self.label}: {self.text}" if self.label else self.text


@dataclass(frozen=True)
class Revision:
    """What to hand the player, and what was taken out of it.

    ``text`` empty means deliver nothing — the answer is refused, exactly as it
    was before this existed, and ``refusal`` says why in one journal line.
    """

    text: str
    body: str
    claims: tuple[FalseClaim, ...] = ()
    strikes: tuple[Strike, ...] = ()
    refusal: str = ""

    @property
    def refused(self) -> bool:
        return not self.text

    @property
    def struck(self) -> bool:
        return bool(self.strikes)


def _sentence_spans(line: str) -> list[tuple[int, int]]:
    """``(start, end)`` for each sentence in one line, whitespace-only dropped."""

    spans: list[tuple[int, int]] = []
    start = 0
    for gap in _SENTENCE_END_RE.finditer(line):
        spans.append((start, gap.start()))
        start = gap.end()
    spans.append((start, len(line)))
    return [(a, b) for a, b in spans if line[a:b].strip()]


def _claim_spans(line: str, claims: Sequence[FalseClaim]) -> list[tuple[int, int, FalseClaim]]:
    """Where in *line* each claim's quoted text sits.

    Whole words only. A distance claim says "Route 2", and a message that talks
    about Route 24 is not the message being disproved.
    """

    found: list[tuple[int, int, FalseClaim]] = []
    for claim in claims:
        said = claim.said
        if not said:
            continue
        at = line.find(said)
        while at != -1:
            if _standalone(line, at, len(said)):
                found.append((at, at + len(said), claim))
            at = line.find(said, at + 1)
    return found


#: The opening of the strike header. Deliberately prose and deliberately not a
#: bracketed tag: this line is prepended to text a *model* wrote, and a header
#: shaped like `[harness] ...` is a shape that text could contain too. The
#: player would then have no way to tell the harness's disclosure from a
#: sentence the thinking session made up, which is the exact confusion the
#: disclosure exists to prevent. Anything in the body that opens this way is
#: defanged before the two are joined; see :func:`revise_advice`.
NOTICE_HEADER = "The game's own map data contradicts part of the message below, and it was removed:"


def removal_notice(strikes: Sequence[Strike]) -> str:
    """The header the player reads above a message something was cut from."""

    if not strikes:
        return ""
    lines = [NOTICE_HEADER]
    spelled = 0
    for strike in strikes:
        quote = strike.text
        if len(quote) > MAX_NOTICE_QUOTE:
            quote = quote[:MAX_NOTICE_QUOTE].rstrip() + "…"
        item = [f'- {strike.label + ": " if strike.label else ""}"{quote}"']
        # Two at most. One sentence can be wrong in two ways — a door that is
        # not there, onto a map that is four warps off — and the second is
        # usually the one that says where the player actually is.
        item += [f"  map data: {claim}" for claim in strike.claims[:2]]
        if spelled and sum(len(line) + 1 for line in lines + item) > NOTICE_LIMIT:
            break
        lines += item
        spelled += 1
    if spelled < len(strikes):
        lines.append(f"- and {len(strikes) - spelled} more.")
    # What the reader would otherwise have to work out from the gap: which
    # numbers are missing, and whether anything outside the list went too. A
    # message that reads as whole while a step of it is gone is the elision
    # this project has twice paid for.
    steps = [strike.label for strike in strikes if strike.label]
    loose = len(strikes) - len(steps)
    said = []
    if steps:
        named = steps[0] if len(steps) == 1 else ", ".join(steps[:-1]) + f" or {steps[-1]}"
        said.append(f"has no {named}")
    if loose:
        said.append(f"is missing {loose} sentence{'s' if loose > 1 else ''} of prose")
    lines.append(f"The message below {' and '.join(said)}. Nothing else was changed.")
    return "\n".join(lines)


def strike_note(strikes: Sequence[Strike]) -> str:
    """What was cut, short enough to sit in a journal line beside the answer."""

    # One claim can cost two sentences — "You emerge on Route 1. Turn WEST" is
    # a single claim about both of them — and printing it twice says the
    # thinker was wrong twice.
    reasons = dict.fromkeys(str(claim) for strike in strikes for claim in strike.claims[:1])
    return "map data contradicts, struck: " + "; ".join(list(reasons)[:MAX_NOTE_CLAIMS])


def revise_advice(
    text: str,
    *,
    here: str,
    maps: Optional[MapFacts] = None,
    max_hops: int = MAX_ADVICE_HOPS,
) -> Revision:
    """*text* with what the map data disproves cut out of it, or nothing at all.

    Three outcomes, in the order they are decided:

    * nothing disproved — the message goes through untouched;
    * something disproved and enough left over — the remainder goes through
      under :func:`removal_notice`, which quotes what went;
    * everything else — refused, and ``refusal`` says which way it failed:
      nothing to cut, nothing left, or more than half the message gone. "The
      map contradicted it" no longer distinguishes a message that was binned
      from one that was trimmed, so the journal line has to.
    """

    claims = check_advice(text, here=here, maps=maps, max_hops=max_hops)
    if not claims:
        return Revision(text=text, body=text)

    kept: list[str] = []
    strikes: list[Strike] = []
    for line in text.splitlines():
        hits = _claim_spans(line, claims)
        spans = _sentence_spans(line)
        if not hits or not spans:
            kept.append(line)
            continue
        item = _ENUMERATED_RE.match(line)
        if item:
            # The whole step, so the player never follows a numbered
            # instruction with its verb cut out of it. The number itself moves
            # to the label, where the notice can say which step is missing.
            numbered = item.group(1)
            strikes.append(
                Strike(
                    text=(line[item.end() :] if numbered else line).strip(),
                    label=f"step {numbered}" if numbered else "",
                    claims=tuple(dict.fromkeys(claim for _, _, claim in hits)),
                )
            )
            continue
        survivors: list[str] = []
        for start, end in spans:
            overlapping = tuple(
                dict.fromkeys(claim for at, to, claim in hits if at < end and to > start)
            )
            if overlapping:
                strikes.append(Strike(text=line[start:end].strip(), label="", claims=overlapping))
            else:
                survivors.append(line[start:end])
        remainder = " ".join(piece for piece in survivors if piece.strip()).strip()
        if remainder:
            kept.append(remainder)

    body = "\n".join(kept).strip()
    note = refusal_note(claims)
    if not strikes:
        # The checker quotes what it disproved and a strike has to find that
        # text again to cut it, which fails where the quoting normalised
        # something — a tile written "(99, 99)" is quoted back without the
        # space. A claim with nothing to cut is the old behaviour, unchanged.
        return Revision(
            text="",
            body=body,
            claims=claims,
            refusal=f"{note}; no sentence to strike it from",
        )
    if not body:
        return Revision(
            text="",
            body=body,
            claims=claims,
            strikes=tuple(strikes),
            refusal=f"{note}; nothing left after striking it",
        )
    share = len(body) / len(text.strip() or body)
    if share < MIN_SURVIVING_SHARE:
        return Revision(
            text="",
            body=body,
            claims=claims,
            strikes=tuple(strikes),
            refusal=f"{note}; striking it takes {round((1 - share) * 100)}% of the message",
        )
    # The body is text a model wrote and the header is the harness speaking. If
    # the body can open a line the same way, the player cannot tell which is
    # which, and the one line whose whole job is to be trusted is the one that
    # becomes forgeable. Cheap to prevent, so prevent it.
    safe_body = "\n".join(
        f"> {line}" if line.lstrip().startswith(NOTICE_HEADER[:40]) else line
        for line in body.splitlines()
    )
    return Revision(
        text=f"{removal_notice(strikes)}\n\n{safe_body}",
        body=body,
        claims=claims,
        strikes=tuple(strikes),
    )


__all__ = [
    "Trigger",
    "Detector",
    "StalledMilestones",
    "Circling",
    "LowHP",
    "Toothless",
    "RepeatedFailure",
    "EnteringSegment",
    "CommitGate",
    "InterventionPolicy",
    "DEFAULT_HARD_SEGMENTS",
    "default_detectors",
    "build_prompt",
    "Fact",
    "MapFacts",
    "FalseClaim",
    "check_advice",
    "refusal_note",
    "revise_advice",
    "removal_notice",
    "NOTICE_HEADER",
    "NOTICE_LIMIT",
    "strike_note",
    "Revision",
    "Strike",
    "MIN_SURVIVING_SHARE",
    "MAX_ADVICE_HOPS",
    "standing_on",
    "harness_facts",
    "format_facts",
    "FACT_BUDGET_CHARS",
    "FACTS_HEADER",
    "FACTS_KNOWN_HEADER",
    "FACTS_INFERRED_HEADER",
    "PRIORITY_COMMIT",
    "PRIORITY_DANGER",
    "PRIORITY_STUCK",
    "PRIORITY_REHEARSAL",
]
