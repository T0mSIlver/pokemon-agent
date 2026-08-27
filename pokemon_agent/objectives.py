"""The deterministic objective ladder and the packs it is loaded from.

The packs are hand-written and finite. When the last of them runs out the
objective stops coming from a file and starts coming from the milestone
frontier -- the rungs of :mod:`pokemon_agent.milestones` whose prerequisites are
already satisfied in RAM. The frontier is presented as a menu, not a plan: the
harness narrows it to what the ladder says is open and the model picks. See
``handoff_to_frontier`` below.

The menu says how far away each rung is, because a menu that does not is how a
run loses 739 presses. Standing in Cerulean City with "Defeated Misty" on the
menu and the gym twenty steps away, the model spent 33% of a leg walking Route 4
-- which is the other side of the city -- hunting maps that were not on the
list at all. It read ``GET /frontier`` 63 times and acted on none of it, because
the frontier said what was open and nothing at all about where. See
``MILESTONE_MAPS`` and ``_frontier_option``.
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

# ---------------------------------------------------------------------------
# Where a rung is earned
#
# A milestone is not a map. ``Milestone`` carries an id, a label, a kind and the
# RAM source that proves it -- and nothing about geography, because the ladder
# was generated as a scoreboard and a scoreboard has no coordinates. Neither
# does anything downstream of it: ``red_milestones.json`` has four fields per
# rung and no map, and the generated game data joins to milestones nowhere.
# ``items.json`` is keyed by map but lists only *ground* items, so it can
# confirm four of the fifty-eight and knows nothing about the rest.
#
# So the table below is hand-written from the pokered scripts, to the same
# standard as ``milestones._PLAN``: one entry per rung, the map whose script
# sets that flag, and no entry at all where there is not exactly one such map.
# Two rules keep it from becoming fiction.
#
# * **Never read the label.** "Met Bill on Route 25" is set inside Bill's House,
#   not on Route 25; "Got the Bike Voucher" names no place whatsoever. A rule
#   that guessed a map out of label text would be wrong on the first and silent
#   on the second while sounding equally confident about both.
# * **Absent beats approximate.** ``EVENT_SS_ANNE_LEFT`` is set when the ship
#   sails, which is not somewhere you can walk to, so it has no entry and
#   renders as a rung with no map on record. That is a visibly different answer
#   from "eight hops away", and it has to stay one: a reader who cannot tell
#   "far" from "unknown" is worse off than one who was told nothing.
#
# Checked, not asserted: ``tests/test_objectives.py`` fails if any name here is
# absent from the generated map graph, and cross-checks the four item rungs
# against the ground-item placements in ``items.json``.
MILESTONE_MAPS: dict[str, str] = {
    # Pallet and the road north.
    "EVENT_GOT_STARTER": "Oak's Lab",
    "EVENT_BATTLED_RIVAL_IN_OAKS_LAB": "Oak's Lab",
    "EVENT_GOT_OAKS_PARCEL": "Viridian Mart",
    "EVENT_OAK_GOT_PARCEL": "Oak's Lab",
    "EVENT_GOT_POKEDEX": "Oak's Lab",
    # Daisy is Blue's sister and hands the map over in his house, not in Oak's.
    "EVENT_GOT_TOWN_MAP": "Blue's House",
    "EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE": "Route 22",
    # Oak's aide with HM05 waits in the gate building on Route 2, one floor,
    # not in the trade house next door.
    "EVENT_GOT_HM05": "Route 2 Gate",
    # Pewter.
    "EVENT_BEAT_BROCK": "Pewter Gym",
    "BADGE_BOULDER": "Pewter Gym",
    # The Old Amber comes from the scientist on the museum's ground floor; the
    # fossil display upstairs gives nothing.
    "EVENT_GOT_OLD_AMBER": "Pewter Museum 1F",
    # Mt. Moon. The fossil room and the Super Nerd guarding it are both B2F.
    "EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD": "Mt Moon B2F",
    "EVENT_GOT_DOME_FOSSIL": "Mt Moon B2F",
    "EVENT_GOT_HELIX_FOSSIL": "Mt Moon B2F",
    # Cerulean. The rival waits outside, in front of Nugget Bridge.
    "EVENT_BEAT_CERULEAN_RIVAL": "Cerulean City",
    "EVENT_BEAT_MISTY": "Cerulean Gym",
    "BADGE_CASCADE": "Cerulean Gym",
    "EVENT_GOT_BICYCLE": "Cerulean Bike Shop",
    # Bill is at the end of Route 25 but both flags are set inside his house.
    "EVENT_MET_BILL": "Bill's House",
    "EVENT_GOT_SS_TICKET": "Bill's House",
    # Vermilion. The Fan Club chairman gives the voucher.
    "EVENT_GOT_BIKE_VOUCHER": "Pokemon Fan Club",
    "EVENT_RUBBED_CAPTAINS_BACK": "S.S. Anne Captain's Room",
    "EVENT_GOT_HM01": "S.S. Anne Captain's Room",
    "EVENT_BEAT_LT_SURGE": "Vermilion Gym",
    "BADGE_THUNDER": "Vermilion Gym",
    # Celadon and the hideout under the Game Corner.
    "EVENT_FOUND_ROCKET_HIDEOUT": "Game Corner",
    "ITEM_LIFT_KEY": "Rocket Hideout B4F",
    "EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI": "Rocket Hideout B4F",
    "ITEM_SILPH_SCOPE": "Rocket Hideout B4F",
    "EVENT_BEAT_ERIKA": "Celadon Gym",
    "BADGE_RAINBOW": "Celadon Gym",
    "EVENT_GOT_HM02": "Route 16 Fly House",
    # Lavender.
    "EVENT_BEAT_POKEMON_TOWER_RIVAL": "Pokemon Tower 2F",
    "EVENT_BEAT_GHOST_MAROWAK": "Pokemon Tower 6F",
    "EVENT_RESCUED_MR_FUJI": "Pokemon Tower 7F",
    "EVENT_GOT_POKE_FLUTE": "Mr Fuji's House",
    "EVENT_BEAT_ROUTE12_SNORLAX": "Route 12",
    # Fuchsia and the Safari Zone.
    "EVENT_GOT_HM03": "Safari Zone Secret House",
    "EVENT_GAVE_GOLD_TEETH": "Warden's House",
    "EVENT_GOT_HM04": "Warden's House",
    "EVENT_BEAT_KOGA": "Fuchsia Gym",
    "BADGE_SOUL": "Fuchsia Gym",
    # Saffron and Silph Co.
    "ITEM_CARD_KEY": "Silph Co 5F",
    "EVENT_BEAT_SILPH_CO_RIVAL": "Silph Co 7F",
    "EVENT_BEAT_SILPH_CO_GIOVANNI": "Silph Co 11F",
    "EVENT_GOT_MASTER_BALL": "Silph Co 11F",
    "EVENT_BEAT_SABRINA": "Saffron Gym",
    "BADGE_MARSH": "Saffron Gym",
    # Cinnabar.
    "ITEM_SECRET_KEY": "Pokemon Mansion B1F",
    "EVENT_BEAT_BLAINE": "Cinnabar Gym",
    "BADGE_VOLCANO": "Cinnabar Gym",
    # The last road.
    "EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI": "Viridian Gym",
    "BADGE_EARTH": "Viridian Gym",
    "EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE": "Route 22",
    # Five guards, one badge each, all of them on Route 23.
    "EVENT_PASSED_EARTHBADGE_CHECK": "Route 23",
    "TOWN_INDIGO_PLATEAU": "Indigo Plateau",
    "ELITE_FOUR_CHAMPION": "Hall of Fame",
    # Deliberately absent: EVENT_SS_ANNE_LEFT. The ship sails once you are off
    # it; there is no map you go to in order to make that happen.
}


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


@lru_cache(maxsize=1)
def _default_router() -> Any:
    """The packaged map graph, read once per process.

    :class:`~pokemon_agent.world.World` loads to an empty graph when the
    generated file is missing, and an empty graph routes nowhere, so a checkout
    without game data loses the brackets and keeps the menu.
    """
    from pokemon_agent.world import World

    return World.load()


def _where_note(milestone_id: str, here: Optional[str], router: Any) -> Optional[str]:
    """The bracket after a rung's label: where it is earned, and how far that is.

    Three answers, and they are three because collapsing any two of them loses
    the distinction the reader needs:

    * ``Cerulean Gym, 1 hop`` -- a map on record with a route to it.
    * ``Route 23, no route the map graph can find`` -- a map on record that this
      graph cannot reach from here. Rare, and not the same as far away.
    * ``no map on record`` -- :data:`MILESTONE_MAPS` has no entry, so the
      harness does not know where this one happens and says exactly that
      instead of a number it would have had to invent.

    ``None`` means say nothing at all: no current map, or no graph to ask.

    A hop is a map-to-map move in the static graph and nothing else. It is not
    a tile count, and it is not a promise the ground between the player and the
    next exit is walkable -- Route 4 is one map whose halves are separated by
    Mt. Moon. The sentence that carries these brackets says so once, in
    :func:`frontier_objective`, in the wording
    :mod:`pokemon_agent.interventions` already uses for the same claim.
    """
    if not here or router is None:
        return None
    where = MILESTONE_MAPS.get(milestone_id)
    if where is None:
        return "no map on record"
    route = router.route(here, where)
    if route is None:
        return f"{where}, no route the map graph can find"
    if not route:
        return f"{where}, the map you are on"
    plural = "" if len(route) == 1 else "s"
    return f"{where}, {len(route)} hop{plural}"


def _frontier_option(
    milestone: Milestone,
    open_now: Collection[str],
    here: Optional[str] = None,
    router: Any = None,
) -> str:
    """One frontier rung as the model reads it: what it is, and what it costs.

    ``effects`` is the DAG's own wording for what the world gains and is often
    empty -- most rungs are a step rather than a key -- so a bare label is the
    normal case, not a gap. ``excludes`` is the one place Red forks for good, and
    a menu that offered both fossils without saying they are the same choice
    would be offering one option too many.

    The bracket goes immediately after the label, ahead of the effects, because
    "how far" is the question the menu was failing to answer and it has to be as
    easy to see as the name.
    """
    label = milestone.label
    note = _where_note(milestone.id, here, router)
    if note is not None:
        label = f"{label} [{note}]"
    node = MILESTONE_DAG.get(milestone.id)
    if node is None:
        return label
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
        return label
    return f"{label} ({'; '.join(notes)})"


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


def frontier_objective(
    reached: Optional[Iterable[Any]],
    *,
    priority: int,
    here: Optional[str] = None,
    router: Any = None,
) -> Optional[JsonDict]:
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

    *here* is the map the player is on and *router* anything with
    ``route(src, dst) -> hops | None``; :class:`~pokemon_agent.world.World` is
    one. Given both, each rung carries a bracket saying where it is earned and
    how many map-graph hops away that is. Given neither, the menu renders
    exactly as it did before brackets existed -- which is what a checkout with
    no generated map data gets, and what a caller that cannot read the current
    map gets. A distance is never half-computed: the explanatory sentence and
    the brackets appear together or not at all.
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
    #
    # And never a *reorder* of them. Nearest-first would be the harness making
    # the pick under another name: with one number attached to each row, the row
    # at the top of a sorted list is a recommendation whatever the sentence
    # above it says, and the number it sorts on is the least reliable thing on
    # the row -- a hop count that has not walked an inch of the ground. The
    # recorded decision for this project is that the model orders off the menu
    # and the harness only narrows it. Distance is a column, not a sort key, so
    # the menu stays in ladder order and says so.
    open_ids = {milestone.id for milestone in open_now}
    if router is None and here:
        graph = _default_router()
        # An empty graph routes nowhere, and rendering that as "no route the map
        # graph can find" on every rung would dress a missing file up as a fact
        # about the world.
        router = graph if len(graph) else None
    annotated = bool(here) and router is not None
    shown = [_frontier_option(milestone, open_ids, here, router) for milestone in open_now]
    where_note = (
        (
            f". Each bracket is where the rung is earned and its map-graph distance from "
            f"{here}: hops are map-to-map moves, never tiles, and never a promise the "
            "ground in between is walkable. Ladder order, not a ranking. "
        )
        if annotated
        else ": "
    )
    summary = (
        "The written objectives end here, so the next goal is yours to pick. "
        "These milestones have every prerequisite the ladder knows about already "
        "met, which is not a claim that any of them can be reached on foot from "
        "where you are standing" + where_note + "; ".join(shown) + "."
    )
    return ObjectiveRecord(
        pack_id=FRONTIER_PACK_ID,
        id=_frontier_objective_id([milestone.id for milestone in open_now]),
        summary=summary,
        completion_predicate=(
            "Any one of the listed milestones reads as set in RAM. poke progress is the check."
        ),
        failure_hints=[
            (
                "A hop count comes off the map graph, which knows which maps touch and "
                "not whether you can walk to the exit. Check one with route before you "
                "spend a batch on it."
                if annotated
                else "The ladder checks prerequisites, not geography. If the one you picked "
                "turns out to be walled off from here, route to it or take another."
            ),
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
            # The same map name the selectors read. Absent -- an unreadable
            # machine, a battle frame -- the frontier renders without distances
            # rather than measuring from a guess.
            here = str((state.get("map") or {}).get("map_name") or "") or None
            live = frontier_objective(
                state.get("milestones"), priority=len(objectives) + 1, here=here
            )
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
