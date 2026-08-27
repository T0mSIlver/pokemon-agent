"""The deterministic objective ladder and the packs it is loaded from.

The packs are hand-written and finite. When the last of them runs out the
objective stops coming from a file and starts coming from the milestone
frontier -- the rungs of :mod:`pokemon_agent.milestones` whose prerequisites are
already satisfied in RAM. The frontier is presented as a menu, not a plan: the
harness narrows it to what the ladder says is open and the model picks. See
``handoff_to_frontier`` below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Iterable, Optional, Sequence

from pokemon_agent.milestones import MILESTONE_DAG, MILESTONES_BY_ID, Milestone
from pokemon_agent.milestones import frontier as milestone_frontier
from pokemon_agent.state_analysis import selector_matches

JsonDict = dict[str, Any]

#: Pack-objective key that says "this rung is where the written ladder stops".
#: A flag in the pack rather than an id compared in code: the id of the last
#: objective is pack data, and a pack edited without editing this module would
#: silently take the handoff with it.
HANDOFF_KEY = "handoff_to_frontier"

#: ``pack_id`` on the objective built from the live frontier. Not a file.
FRONTIER_PACK_ID = "milestone_frontier"


@dataclass(slots=True)
class ObjectiveRecord:
    pack_id: str
    id: str
    summary: str
    completion_predicate: str
    failure_hints: list[str]
    save_recommendation: str
    priority: int
    current: bool
    completed: bool
    status: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


@lru_cache(maxsize=1)
def load_red_objective_packs() -> list[JsonDict]:
    data_dir = Path(__file__).parent / "data"
    packs: list[JsonDict] = []
    for path in sorted(data_dir.glob("red_objectives_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            packs.append(payload)
    return sorted(packs, key=lambda pack: int(pack.get("order", 0)))


def _frontier_option(milestone: Milestone, open_now: Collection[str]) -> str:
    """One frontier rung as the model reads it: what it is, and what it costs.

    ``effects`` is the DAG's own wording for what the world gains and is often
    empty -- most rungs are a step rather than a key -- so a bare label is the
    normal case, not a gap. ``excludes`` is the one place Red forks for good, and
    a menu that offered both fossils without saying they are the same choice
    would be offering one option too many.
    """
    node = MILESTONE_DAG.get(milestone.id)
    if node is None:
        return milestone.label
    notes = []
    if node.effects:
        notes.append(f"opens {' and '.join(node.effects)}")
    forgone = [
        MILESTONES_BY_ID[other].label
        for other in node.excludes
        if other in open_now and other in MILESTONES_BY_ID
    ]
    if forgone:
        notes.append(f"rules out {' and '.join(forgone)}")
    if not notes:
        return milestone.label
    return f"{milestone.label} ({'; '.join(notes)})"


def _frontier_objective_id(milestone_ids: Sequence[str]) -> str:
    """A stable id for one frontier, changing exactly when the frontier does.

    The id is what the autosave trigger, the stuck counter and the event log
    compare, so it has to move when the goal moves and hold still when it does
    not. A digest of the open rungs is the only thing that does both: a fixed
    string would leave the stuck counter climbing across a goal that had really
    changed, and a counter of reached rungs would move on reads that changed
    nothing about what is open.
    """
    digest = hashlib.sha1("\n".join(milestone_ids).encode("utf-8")).hexdigest()[:8]
    return f"milestone_frontier_{digest}"


def frontier_objective(reached: Optional[Iterable[Any]], *, priority: int) -> Optional[JsonDict]:
    """The live milestone frontier as an objective record, or ``None``.

    ``None`` means there is no answer to give -- unreadable milestone data, or a
    frontier that is empty because the ladder is finished -- and the caller keeps
    the pack objective. Nothing here invents a milestone list: *reached* is
    whatever :class:`~pokemon_agent.milestones.MilestoneTracker` read out of RAM.

    A list with nothing recognisable in it is a failed read, not a fresh game,
    even though the two look identical from here: the server answers an
    unreadable machine with an empty list. The distinction matters because the
    frontier of nothing is "go and get a starter", which would be a confident lie
    printed over a run that is already past HM01.

    The wording is deliberately narrow. The frontier is the set of rungs whose
    ladder prerequisites are met. That is not a claim that any of them can be
    walked to from where the player is standing, and the text must not be
    readable as one.
    """
    try:
        have = [str(item) for item in (reached or ())]
    except TypeError:  # a milestones field that is not iterable
        return None
    if not any(milestone_id in MILESTONES_BY_ID for milestone_id in have):
        return None
    try:
        open_now = milestone_frontier(have)
    except Exception:  # noqa: BLE001 -- the objective must never fail an observation
        return None
    if not open_now:
        return None

    # Every open rung, never a prefix of them. Trimming would be the harness
    # choosing after all, and choosing badly: the frontier arrives in ladder
    # order, so a prefix keeps the shallowest options and drops the deepest --
    # the ones that open the most. The list cannot run away either. Walking the
    # DAG to the end in ladder order, and in 400 random legal orders, it peaks at
    # 13 rungs and averages under five.
    open_ids = {milestone.id for milestone in open_now}
    shown = [_frontier_option(milestone, open_ids) for milestone in open_now]
    summary = (
        "The written objectives end here, so the next goal is yours to pick. "
        "These milestones have every prerequisite the ladder knows about already "
        "met, which is not a claim that any of them can be reached on foot from "
        "where you are standing: " + "; ".join(shown) + "."
    )
    return ObjectiveRecord(
        pack_id=FRONTIER_PACK_ID,
        id=_frontier_objective_id([milestone.id for milestone in open_now]),
        summary=summary,
        completion_predicate=(
            "Any one of the listed milestones reads as set in RAM. poke progress is the check."
        ),
        failure_hints=[
            "The ladder checks prerequisites, not geography. If the one you picked "
            "turns out to be walled off from here, route to it or take another.",
            "poke progress re-reads this list from RAM, so it is current after every batch.",
        ],
        save_recommendation="Save before committing to one of these, and again once it lands.",
        priority=priority,
        current=True,
        completed=False,
        status="current",
    ).to_dict()


class ObjectiveEngine:
    """Deterministic Red-first objective progression across chained packs."""

    def __init__(self) -> None:
        self.packs = load_red_objective_packs()
        self.objectives: list[JsonDict] = []
        for pack in self.packs:
            pack_id = str(pack.get("pack_id") or "unknown_pack")
            for item in pack.get("objectives") or []:
                if not isinstance(item, dict):
                    continue
                merged = dict(item)
                merged["pack_id"] = pack_id
                merged.setdefault("selector", {})
                self.objectives.append(merged)
        self.by_id = {item["id"]: item for item in self.objectives}

    def _current_objective_index(self, state: JsonDict) -> int:
        if not self.objectives:
            return 0
        current_index = 0
        for index, item in enumerate(self.objectives):
            if selector_matches(item.get("selector") or {}, state):
                current_index = index
        return current_index

    def evaluate(self, state: JsonDict) -> JsonDict:
        if not self.objectives:
            empty = ObjectiveRecord(
                pack_id="unknown_pack",
                id="no_objectives_loaded",
                summary="Objective data was not loaded.",
                completion_predicate="N/A",
                failure_hints=[],
                save_recommendation="Manual saves only.",
                priority=1,
                current=True,
                completed=False,
                status="current",
            ).to_dict()
            return {
                "game": "red",
                "current": empty,
                "objectives": [empty],
                "progress_percent": 0,
                "current_pack_id": "unknown_pack",
                "packs": [],
                "phase_complete": False,
            }

        current_index = self._current_objective_index(state)
        current_id = self.objectives[current_index]["id"]
        total_steps = max(len(self.objectives) - 1, 1)
        progress_percent = min(100, int((current_index / total_steps) * 100))
        objectives: list[JsonDict] = []
        current_objective: Optional[JsonDict] = None

        for index, item in enumerate(self.objectives):
            completed = index < current_index
            current = item["id"] == current_id
            record = ObjectiveRecord(
                pack_id=item["pack_id"],
                id=item["id"],
                summary=item["summary"],
                completion_predicate=item["completion_predicate"],
                failure_hints=item.get("failure_hints", []),
                save_recommendation=item.get("save_recommendation", ""),
                priority=index + 1,
                current=current,
                completed=completed,
                status="completed" if completed else "current" if current else "pending",
            ).to_dict()
            objectives.append(record)
            if current:
                current_objective = record

        assert current_objective is not None

        # The written ladder is out of rungs. Hand over to the live frontier if
        # the milestone read came through; if it did not, the pack objective
        # stands, which is what happened before this branch existed.
        handoff = bool(self.objectives[current_index].get(HANDOFF_KEY))
        if handoff:
            live = frontier_objective(state.get("milestones"), priority=len(objectives) + 1)
            if live is not None:
                current_objective["current"] = False
                current_objective["completed"] = True
                current_objective["status"] = "completed"
                objectives.append(live)
                current_objective = live

        return {
            "game": "red",
            "current": current_objective,
            "objectives": objectives,
            "progress_percent": progress_percent,
            "current_pack_id": current_objective["pack_id"],
            "packs": [
                {"pack_id": pack.get("pack_id"), "order": pack.get("order")} for pack in self.packs
            ],
            # The written packs are finished, whether or not the frontier could
            # be read. Same moment the id comparison used to fire on, without
            # the id.
            "phase_complete": handoff,
        }
