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
MAX_EXITS_SHOWN = 6
MAX_CENTERS_SHOWN = 3
MAX_CENTER_ROUTES = 2
MAX_HOPS_SHOWN = 4
MAX_FRONTIER_SHOWN = 3
MAX_ERROR_CHARS = 160

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

    def poke_centers(self, source: Optional[str], limit: int = MAX_CENTERS_SHOWN) -> list[tuple]:
        """Poke Centers by graph distance from *source*, nearest first.

        Distance is hops in the map graph and nothing more: it does not know
        whether the walk to the first exit is currently possible, which is why
        the caller prints it under INFERRED and hands over the whole ranking
        rather than a single recommendation.
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
        return ranked[:limit]

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
        if isinstance(current, Mapping) and current.get("summary"):
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


def _repeated_tiles(recent: Sequence[Receipt], map_name: str, limit: int = 3) -> list[tuple]:
    counts = Counter(r.pos for r in recent if r.pos is not None and r.map_name == map_name)
    return counts.most_common(limit)


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
    """The error text itself, which is the one thing a retry loop never reads."""

    for receipt in reversed(recent):
        if not receipt.exit_code:
            continue
        extra = receipt.extra or {}
        error = str(extra.get("error") or "").strip()
        actions = extra.get("actions")
        parts: list[str] = []
        if error:
            tool = receipt.tool or "the command"
            parts.append(f"`{tool}` failed with: {error[:MAX_ERROR_CHARS]}")
        if isinstance(actions, (list, tuple)) and actions:
            parts.append("actions sent: " + " ".join(str(action) for action in actions[:8]))
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
        shown = [_describe_hop(hop) for hop in exits[:MAX_EXITS_SHOWN]]
        rest = len(exits) - len(shown)
        tail = f", and {rest} more warp{'s' if rest > 1 else ''}" if rest else ""
        known.append(Fact(f"Every exit from {map_name}: {'; '.join(shown)}{tail}."))

    if trigger.name == "repeated_failure":
        failure = _failure_fact(recent)
        if failure is not None:
            known.append(failure)

    if trigger.name == "circling":
        repeated = _repeated_tiles(recent, map_name)
        if repeated:
            spread = ", ".join(f"{_coord_text(pos)} x{count}" for pos, count in repeated)
            known.append(Fact(f"Tiles you keep standing on: {spread}."))
            neighbours = _neighbour_fact(repeated[0][0], live["snapshot"], live["explored"])
            if neighbours is not None:
                known.append(neighbours)

    centers = maps.poke_centers(map_name)
    if centers:
        parts = []
        for index, (distance, name, hops) in enumerate(centers):
            plural = "" if distance == 1 else "s"
            detail = f" [{_describe_route(hops)}]" if index < MAX_CENTER_ROUTES else ""
            parts.append(f"{name} {distance} hop{plural}{detail}")
        inferred.append(
            Fact(f"Poke Centers from {map_name}, nearest first: {'; '.join(parts)}.", known=False)
        )

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
    """

    known: list[Fact] = []
    inferred: list[Fact] = []
    spent = len(FACTS_HEADER)
    for fact in facts:
        section = known if fact.known else inferred
        header = 0 if section else len(FACTS_KNOWN_HEADER if fact.known else FACTS_INFERRED_HEADER)
        cost = len(fact.text) + 5 + (header + 2 if header else 0)
        if spent + cost > budget:
            continue
        section.append(fact)
        spent += cost
    if not known and not inferred:
        return ""

    lines = [FACTS_HEADER]
    if known:
        lines += ["", FACTS_KNOWN_HEADER]
        lines += [f"  - {fact.text}" for fact in known]
    if inferred:
        lines += ["", FACTS_INFERRED_HEADER]
        lines += [f"  - {fact.text}" for fact in inferred]
    return "\n".join(lines)


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
        state_summary.strip(),
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
        "Answer in at most 150 words, as directions the player can press: "
        "which way, how many tiles, which exit. Not strategy. Every claim you "
        "make about where something is must come from the facts above — if "
        "they do not cover it, say what to check rather than recalling it.",
    ]
    return "\n".join(lines)


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
