"""The party as a fighting unit: what it can beat, and what a learn prompt costs.

Two facts the payload never carried, both measured on the same 33-hour run that
ended with one Charmeleon L33 holding Cut, Growl, Ember and Leer:

* **The fight ahead was never priced.** ``/calc`` prices the Pokemon *in front
  of you*, which is the right table once a fight has started and no help at all
  for deciding whether to start one. That run spent 3,044 presses inside
  Cerulean Gym without a Cascade Badge and whited out 40 times, and no answer it
  ever read named Misty's team or what its own party did to it. It goes in the
  payload rather than behind a verb because the verbs are not called: ``calc``,
  ``route``, ``frontier`` and ``progress`` were each used **zero** times across
  a 457-call session.

* **A move was overwritten and the moveset never recovered.** Cut went over one
  of its attacks, and Gen 1 refuses to delete an HM move afterwards, so that
  slot is spent for the rest of the run. Every frame of the prompt that did it
  was in the payload; none of them said which move was about to go.

Everything here reasons over the whole party, not slot 0. An earlier audit
caught ``calc`` attacking with ``party[0]`` while a different Pokemon was on the
field; "the party" here means every member with HP left, and the best answer to
a given Pokemon may be on any of them.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pokemon_agent import gamedata

JsonDict = Dict[str, Any]

#: Which badge each gym hands out. The leader is entry 0 of the map's trainer
#: table in every one of the eight, so the roster itself is read from pokered
#: rather than typed out here; only this map->badge line is local knowledge.
GYM_BADGE: Dict[str, str] = {
    "Pewter Gym": "Boulder",
    "Cerulean Gym": "Cascade",
    "Vermilion Gym": "Thunder",
    "Celadon Gym": "Rainbow",
    "Fuchsia Gym": "Soul",
    "Saffron Gym": "Marsh",
    "Cinnabar Gym": "Volcano",
    "Viridian Gym": "Earth",
}

#: Phrases from the move-learn flow, decoded off the real screen. Every frame
#: between "CHAR is trying to learn DIG!" and "Which move should be forgotten?"
#: carries one of them, and nothing else in Red says any of them. Captured by
#: driving HM01 and TM28 onto a four-move Charmeleon in PyBoy; see
#: ``tests/test_party.py`` for the frames themselves.
#:
#: Apostrophe-free on purpose: "can't" is one tile in Gen 1 and matching it
#: needs the ligature decoded, so the phrase that catches those two frames is
#: the half of the sentence without it.
LEARN_PROMPT_PHRASES: Tuple[str, ...] = (
    "trying to learn",
    "learn more",
    "than 4 moves",
    "Delete an older",
    "move to make room",
    "be forgotten",
)

#: The frame that actually holds the four move names, with a cursor on one.
FORGET_LIST_PHRASE = "be forgotten"


#: Effects whose move spends a turn going nowhere before it lands. Read off the
#: move's own effect byte in ``moves.json``, so "Dig is 100 power" and "Dig hits
#: every other turn" are the same fact from the same table: 100 power once per
#: two turns is 50 a turn, which is *under* Slash at 70. The payload printed Dig
#: as a strict upgrade over Slash on the other arithmetic, 292 times.
#:
#: Hyper Beam is deliberately not here. Its recharge turn is skipped when it
#: knocks the target out, so halving it would be a claim the data does not make.
TWO_TURN_EFFECTS: Tuple[str, ...] = ("CHARGE_EFFECT", "FLY_EFFECT")


def hm_moves() -> Dict[str, str]:
    """``{"Cut": "HM01", ...}`` -- the five moves Gen 1 will never let go of.

    Read out of ``tms.json`` rather than typed here, so it is the same join that
    turns "HM01 x1" into "Cut". A slot an HM move goes into is spent for the
    rest of the run, which is the most expensive fact in this file and was in
    none of the lines it printed.
    """
    return {move: label for label, move in gamedata.tms().items() if label.startswith("HM")}


def _able(party: Sequence[JsonDict]) -> List[JsonDict]:
    """Party members that can still take a turn. A fainted one answers nothing."""
    return [mon for mon in party or () if isinstance(mon, dict) and (mon.get("hp") or 0) > 0]


def _teachable(party: Sequence[JsonDict]) -> List[JsonDict]:
    """Party members a machine could be taught to, which is all of them.

    Not :func:`_able`. Whether a Pokemon has fainted decides whether it can take
    a turn and has nothing to do with whether it can learn a move -- the Gen 1
    TM screen is happy to teach one at 0 HP. Filtering on HP meant the strongest
    member dropped out of the teaching advice exactly when it had just fainted,
    and the line fell through to whatever was left: measured over one run, 222
    of 292 impressions recommended teaching a level-3 Rattata.
    """
    return [mon for mon in party or () if isinstance(mon, dict)]


def move_names(mon: JsonDict) -> List[str]:
    """The move names on a mon, from either shape the readers produce."""
    names: List[str] = []
    for move in mon.get("moves") or ():
        if isinstance(move, dict) and move.get("name"):
            names.append(str(move["name"]))
        elif isinstance(move, str) and move:
            names.append(move)
    return names


def _has_pp(mon: JsonDict, name: str) -> bool:
    for move in mon.get("moves") or ():
        if isinstance(move, dict) and str(move.get("name")) == name:
            return move.get("pp") != 0
    return True


def damaging_moves(mon: JsonDict) -> List[Tuple[str, int]]:
    """``(name, power)`` for every move on *mon* that can take HP off anything.

    Power, not type: Growl and Leer are the two that were picked 49 times in one
    run's 289 attacks, and what is wrong with them is that their power is zero.
    """
    out: List[Tuple[str, int]] = []
    for name in move_names(mon):
        record = gamedata.move(name)
        if record and record.get("power"):
            out.append((name, int(record["power"])))
    return out


def mon_types(mon: JsonDict) -> List[str]:
    """The mon's types, from the reader's own field or from ``species.json``."""
    if mon.get("types"):
        return [str(kind) for kind in mon["types"]]
    entry = gamedata.species(str(mon.get("species") or "")) or {}
    return [str(kind) for kind in entry.get("types") or ()]


def power_per_turn(mon: JsonDict, name: str) -> int:
    """What one turn of *name* is worth on *mon*, in power.

    Base power is the number every line in this file used to print, and it is
    wrong three ways over. Each correction is a column the game's own tables
    already carry, so every factor is checkable against pokered:

    * **STAB.** ``species.json`` gives the mon's types, ``moves.json`` the
      move's, and Gen 1 multiplies by 3/2 where they meet. Ember on a Charmeleon
      is 60, not 40. Thunderbolt on a Pikachu is 142, not 95 -- and the payload
      offered that machine to a level 3 Rattata instead, 222 times.
    * **Accuracy.** ``moves.json`` carries it. Cut misses one turn in twenty,
      Mega Kick one in four.
    * **The charge turn.** See ``TWO_TURN_EFFECTS``.

    Fixed-damage moves -- the ones whose power byte is 1: Seismic Toss, Dragon
    Rage, Psywave -- answer 0 and stay out of every ranking here rather than
    being compared as though 1 were their power. They still damage, and
    ``damaging_moves`` still lists them.
    """
    record = gamedata.move(name)
    if not record:
        return 0
    power = int(record.get("power") or 0)
    if power <= 1:
        return 0
    value = float(power)
    if record.get("type") in mon_types(mon):
        value *= 1.5
    value *= int(record.get("accuracy") or 100) / 100.0
    if record.get("effect") in TWO_TURN_EFFECTS:
        value /= 2
    return int(value)


def ranked_attacks(mon: JsonDict) -> List[Tuple[str, int]]:
    """``(name, power a turn)`` for the mon's attacks, hardest first."""
    rows = [(name, power_per_turn(mon, name)) for name in move_names(mon)]
    return sorted([row for row in rows if row[1] > 0], key=lambda row: -row[1])


def best_attack(mon: JsonDict) -> Optional[Tuple[str, int]]:
    """The mon's hardest hit per turn, or None when it has none."""
    rows = ranked_attacks(mon)
    return rows[0] if rows else None


def attack_types(mon: JsonDict) -> List[str]:
    """The types this mon can actually take HP off something with."""
    out: List[str] = []
    for name, _ in ranked_attacks(mon):
        kind = (gamedata.move(name) or {}).get("type")
        if kind and str(kind) not in out:
            out.append(str(kind))
    return out


def cheapest_slot(mon: JsonDict) -> Optional[Tuple[str, int, int]]:
    """``(name, power a turn, slot index)`` for the move that costs least to lose.

    Never an HM move: the game refuses to delete those, so naming one as the
    cheapest slot is an instruction the button cannot carry out. Ties go to the
    earlier slot, which is the one fewer presses from where the cursor starts.
    """
    hms = hm_moves()
    rows = [
        (name, power_per_turn(mon, name), index)
        for index, name in enumerate(move_names(mon))
        if name not in hms
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: (row[1], row[2]))


def level_up_move(mon: JsonDict) -> Optional[Tuple[int, str, bool]]:
    """``(level, move, due_now)`` for the move the species is about to teach itself.

    Straight off the ``learnset`` in ``species.json``, which is pokered's table.
    ``due_now`` means the entry sits at exactly this level and the mon does not
    hold it -- the prompt is live or one battle away. A Charizard at L46 without
    Flamethrower is exactly that, and it is what the live party is.

    Entries *below* the level are not reported at all. The game offered them
    once and the run either declined or spent the slot elsewhere -- this party
    passed on Rage at 24 and Scratch before it -- so calling them due would be a
    prompt that is never coming back.
    """
    entry = gamedata.species(str(mon.get("species") or "")) or {}
    level = int(mon.get("level") or 0)
    known = set(move_names(mon))
    rows = [
        (int(row.get("level") or 0), str(row.get("move")))
        for row in entry.get("learnset") or ()
        if str(row.get("move")) not in known and int(row.get("level") or 0) > 1
    ]
    due = [row for row in rows if row[0] == level]
    if due:
        return (due[0][0], due[0][1], True)
    ahead = [row for row in rows if row[0] > level]
    if not ahead:
        return None
    best = min(ahead)
    return (best[0], best[1], False)


def expected_opponent(species: str, level: int) -> Optional[JsonDict]:
    """A trainer's Pokemon in the shape ``gamedata.damage_range`` wants.

    Species and level are exact -- pokered's trainer table is the one the game
    loads -- and the stats are the same DV-8 estimate the harness already falls
    back to for a battler whose live struct is unreadable. Every number derived
    from it is printed with a ``~``.
    """
    entry = gamedata.species(species)
    if not entry:
        return None
    mon = {"species": species, "level": int(level), "types": list(entry.get("types") or [])}
    return {**mon, "stats": gamedata.estimated_stats(mon), "hp": gamedata.estimated_hp(mon)}


def unbeaten_leader(map_name: Optional[str], badges: Iterable[str]) -> Optional[JsonDict]:
    """The gym leader standing on this map whose badge is not in the bag yet.

    Keyed on the map the player is on rather than on "the next gym in the
    ladder": the gyms can be walked in any order the roads allow, and the only
    fight the payload should price is the one in the room.
    """
    badge = GYM_BADGE.get(str(map_name or ""))
    if badge is None or badge in set(badges or ()):
        return None
    roster = gamedata.trainers(str(map_name))
    if not roster:
        return None
    leader = roster[0]
    return {
        "leader": str(leader.get("trainer_class") or "the gym leader"),
        "badge": badge,
        "team": list(leader.get("team") or []),
    }


def _best_answer(able: Sequence[JsonDict], foe: JsonDict) -> Optional[Tuple[int, str, str]]:
    """``(worst-roll damage, mon species, move)`` for the party's hardest hit.

    Every member, every move: the best answer to a Water type is rarely on the
    Pokemon that happens to be in slot 0.
    """
    best: Optional[Tuple[int, str, str]] = None
    for mon in able:
        for name in move_names(mon):
            if not _has_pp(mon, name):
                continue
            try:
                rolled = gamedata.damage_range(mon, name, foe)
            except (KeyError, TypeError):
                continue
            if rolled[0] <= 0:
                continue
            if best is None or rolled[0] > best[0]:
                best = (rolled[0], str(mon.get("species") or "?"), name)
    return best


def leader_outlook(party: Sequence[JsonDict], leader: JsonDict) -> Optional[str]:
    """What this party does to the gym leader's team, one line, before the fight.

    Estimates, and they say so. The point is not the exact turn count: it is
    that "Ember ~7 turns" and "Cut ~3 turns" are different answers and the run
    that lost here kept picking the first one.
    """
    able = _able(party)
    if not able or not leader.get("team"):
        return None
    rows: List[str] = []
    answered = 0
    for member in leader["team"]:
        foe = expected_opponent(str(member.get("species") or ""), int(member.get("level") or 1))
        head = f"{member.get('species')} L{member.get('level')}"
        if foe is None:
            continue
        best = _best_answer(able, foe)
        if best is None:
            rows.append(f"{head} nothing you carry damages it")
            continue
        answered += 1
        damage, who, name = best
        owner = f" ({who})" if len(able) > 1 else ""
        turns = math.ceil(foe["hp"] / damage)
        rows.append(f"{head} best {name}{owner} ~{turns} turn{'' if turns == 1 else 's'}")
    if not rows:
        return None
    line = f"ahead {leader['leader']}: " + " | ".join(rows)
    if answered == 0:
        # The whole point of the field. Said plainly, once, where the decision is.
        line += (
            f"\n  this party cannot win this gym: nothing it carries damages "
            f"anything {leader['leader']} sends out"
        )
    shape = party_shape(able)
    if shape:
        line += f"\n  {shape}"
    return line


def party_shape(able: Sequence[JsonDict]) -> Optional[str]:
    """ "One Pokemon, one attack" -- said only when one of them is true.

    A six-strong party with full movesets needs no line; the run this was
    written for had one Pokemon whose four slots held one attack, and never saw
    either number written down.
    """
    if not able:
        return None
    attacks = {name for mon in able for name, _ in damaging_moves(mon)}
    parts = []
    if len(able) == 1:
        parts.append("1 Pokemon: a faint here is a whiteout")
    if len(attacks) <= 1:
        parts.append(
            f"1 move that damages ({next(iter(attacks))})"
            if attacks
            else "no move that damages anything"
        )
    return "; ".join(parts) if parts else None


#: How many machines one line is allowed to name. Two, not three: each row now
#: carries the learner's level, the move's power a turn, the type it adds and
#: the slot it costs, so a row is roughly twice the length it was and two of
#: them cost what three used to. The third row was also the one that measurably
#: went to the wrong Pokemon -- with the ranking fixed, rows 1 and 2 of the live
#: party are Dig for the L46 Charizard and Thunderbolt for the Pikachu that gets
#: STAB off it, and row 3 was Bubble Beam for a level 3 Rattata.
TEACHABLE_LIMIT = 2


def _slot_cost(mon: JsonDict) -> str:
    """What teaching this mon costs it: a spare slot, or a named move."""
    if len(move_names(mon)) < 4:
        return "a slot is free"
    slot = cheapest_slot(mon)
    if slot is None:
        return "every slot holds an HM move and Gen 1 deletes none of them"
    name, per_turn, _ = slot
    return f"costs the {name} slot" + (" (no damage)" if per_turn == 0 else f" ({per_turn} a turn)")


def _machine_row(label: str, taught: str, record: JsonDict, mon: JsonDict, gained: int) -> str:
    """One machine, as the numbers that decide whether to spend it."""
    species = str(mon.get("species") or "it")
    kind = str(record.get("type"))
    held = ", ".join(f"{name} {value}" for name, value in damaging_moves(mon)) or "nothing"
    best = best_attack(mon)
    if best is None:
        against = "and nothing it carries damages anything"
    elif gained > best[1]:
        against = f"over its best {best[0]} at {best[1]}"
    elif gained == best[1]:
        against = f"level with its best {best[0]} at {best[1]}"
    else:
        against = f"under its best {best[0]} at {best[1]}"
    reasons = [f"{gained} a turn {against}"]
    if record.get("effect") in TWO_TURN_EFFECTS:
        reasons.append(f"{record.get('power')} halved by its 2-turn charge")
    if kind not in attack_types(mon):
        reasons.append(f"adds {kind}, which nothing it carries hits with")
    return (
        f"{label} teaches {taught} ({kind} {record.get('power')}) and {species} can learn it "
        f"(L{mon.get('level')}): "
        + "; ".join(reasons)
        + f"; it attacks with {held}; {_slot_cost(mon)}"
    )


def tm_audit(party: Sequence[JsonDict], bag: Sequence[JsonDict]) -> List[JsonDict]:
    """Every machine in the bag and what became of it. The mechanism, not a mirror.

    ``teachable_tms`` is built out of this list rather than beside it, so a
    machine that never reaches the payload carries a recorded reason instead of
    a silence. That silence is the failure this whole file exists for: TM28 rode
    along in the bag for 60,000 presses and nothing in the harness could say
    whether it had been considered and dropped or never looked at.

    Each row is ``{"item", "teaches", "status", "detail", "mon", "per_turn"}``
    and ``status`` is one of ``named``, ``unknown_move``, ``no_power``,
    ``nobody_can_learn``, ``already_known``, ``no_gain``. Every machine in the
    bag produces exactly one row; nothing else does. ``named`` rows are the
    candidates ``teachable_tms`` then cuts to ``TEACHABLE_LIMIT``, so a row that
    is ``named`` here and absent from the line was crowded out rather than
    rejected.
    """
    able = _teachable(party)
    rows: List[JsonDict] = []
    for entry in bag or ():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("item") or "")
        taught = gamedata.tm_move(label)
        if not taught:
            continue
        row: JsonDict = {
            "item": label,
            "teaches": taught,
            "status": "",
            "detail": "",
            "mon": None,
            "per_turn": 0,
        }
        record = gamedata.move(taught)
        if record is None:
            row["status"] = "unknown_move"
            row["detail"] = f"{taught} is not in moves.json"
            rows.append(row)
            continue
        if not record.get("power"):
            row["status"] = "no_power"
            row["detail"] = f"{taught} deals no damage; this line only ranks attacks"
            rows.append(row)
            continue
        learners = [
            mon
            for mon in able
            if label in ((gamedata.species(str(mon.get("species") or "")) or {}).get("tm_hm") or ())
        ]
        if not learners:
            row["status"] = "nobody_can_learn"
            row["detail"] = "no party member with HP has it in its tm_hm list"
            rows.append(row)
            continue
        if all(taught in move_names(mon) for mon in learners):
            row["status"] = "already_known"
            row["detail"] = f"every learner already carries {taught}"
            rows.append(row)
            continue
        # Ranked by what the move is worth *on the learner*, not by how much it
        # beats what the learner has. The old rule was the difference, which is
        # largest on the weakest Pokemon in the party: it handed Dig,
        # Thunderbolt and Bubble Beam to a level 3 Rattata on 222 of the 292
        # frames it ever printed, while the L46 Charizard that fought every
        # battle of the run was named on 70 and the Pikachu that gets STAB off
        # Thunderbolt on 12. Ties go to the higher level, because between two
        # Pokemon a move is worth the same on, the one that fights is the one
        # carrying the levels.
        candidates = []
        for mon in learners:
            if taught in move_names(mon):
                continue
            gained = power_per_turn(mon, taught)
            best = best_attack(mon)
            new_type = str(record.get("type")) not in attack_types(mon)
            if gained > (best[1] if best else 0) or new_type:
                candidates.append((gained, int(mon.get("level") or 0), mon))
        if not candidates:
            row["status"] = "no_gain"
            row["detail"] = (
                f"{taught} is weaker per turn than what every learner already has, "
                "and adds no type they cannot already hit with"
            )
            rows.append(row)
            continue
        gained, level, mon = max(candidates, key=lambda c: (c[0], c[1]))
        row["status"] = "named"
        row["mon"] = str(mon.get("species") or "it")
        row["per_turn"] = gained
        row["level"] = level
        row["detail"] = _machine_row(label, taught, record, mon, gained)
        rows.append(row)
    return rows


def teachable_tms(party: Sequence[JsonDict], bag: Sequence[JsonDict]) -> Optional[str]:
    """Machines in the bag worth teaching, priced per turn, on the mon that fights.

    The third fact of this file's kind, and the one that cost the most. The run
    this was written for picked up TM28 in Mt. Moon, carried it for roughly
    60,000 presses on a single Charmeleon whose hardest attack was Cut, and
    never taught it. Nothing it read could have told it to: the bag says
    ``TM28 x1``, ``species.json`` says Charmeleon can learn ``TM28``, and until
    ``tms.json`` existed there was nothing anywhere in the harness that said
    TM28 is Dig.

    That join exists now, and the line built on it was still wrong, measurably:
    across a 70-hour run it was printed 292 times, was acted on **zero** times,
    and on 222 of those frames it named a level 3 Rattata. Three things were
    wrong with it and all three are fixed here:

    * it ranked by *base power minus the learner's best*, a difference that is
      always largest on the worst Pokemon in the party. It now ranks by what the
      move is worth on the learner (``power_per_turn``), tie-broken by level;
    * it compared base power, so Dig's 100 read as an upgrade over Slash's 70
      when Dig lands 50 a turn and Slash 70, and Thunderbolt read the same on a
      Pikachu as on a Rattata when STAB makes it 142 against 95;
    * it never priced the cost. Teaching a mon with four moves deletes one, a
      Gen 1 TM is spent doing it, and an HM move can never be deleted at all.
      The slot it would cost is now on the same line as the gain.

    Still deliberately narrow, because a field on every overworld frame is
    expensive: only machines in the bag, only members with HP that pokered says
    can learn them, and only when the move either out-damages what that member
    has per turn or adds a type it cannot hit anything with -- the second is why
    Dig survives the cut on a Charmeleon whose Ember already beats it per turn,
    and Ground on an Electric gym is the reason that matters.

    ``tm_audit`` carries a row for every machine either way, so the machines
    this line drops are droppable-with-a-reason rather than invisible.
    """
    if not _teachable(party) or not bag:
        return None
    rows = [row for row in tm_audit(party, bag) if row["status"] == "named"]
    if not rows:
        return None
    rows.sort(key=lambda row: (-int(row.get("level") or 0), -int(row["per_turn"])))
    return "bag " + " | ".join(str(row["detail"]) for row in rows[:TEACHABLE_LIMIT])


def moveset_gaps(party: Sequence[JsonDict]) -> Optional[str]:
    """The slot the Pokemon that fights is wasting, and the move that would take it.

    ``learn_cost`` speaks only while a replacement prompt is on screen, and
    ``teachable_tms`` only when a machine in the bag happens to fit. Between
    them sits the state this run actually spent its time in: a Charizard
    carrying Leer -- a Defense drop it used 74 times out of 1,436 attacks and
    never once followed up on -- with Ember at 60 a turn beside Slash at 70,
    for tens of thousands of presses, while nothing in any payload said the
    fourth slot was empty of damage.

    Only the highest-level member with HP, because that is the one that fights
    and the only reading of "which one" the state carries; only when it holds a
    slot worth nothing per turn, so a full moveset says nothing at all. Nothing
    here is an instruction: it is the per-turn table, the dead slot, and the
    level-up move that would take it. The bag half of the same decision is
    ``teachable_tms``, which the payload already carries beside this.
    """
    able = _teachable(party)
    if not able:
        return None
    mon = max(able, key=lambda member: int(member.get("level") or 0))
    attacks = ranked_attacks(mon)
    dead = [name for name in move_names(mon) if power_per_turn(mon, name) == 0]
    if not dead or not attacks:
        return None
    species = str(mon.get("species") or "it")
    table = ", ".join(f"{name} {value}" for name, value in attacks)
    plural = "slot" if len(dead) == 1 else "slots"
    line = (
        f"{species} L{mon.get('level')} attacks with {table} a turn and spends "
        f"{len(dead)} {plural} on {', '.join(dead)}, worth nothing"
    )
    upcoming = level_up_move(mon)
    if upcoming:
        level, name, due = upcoming
        record = gamedata.move(name) or {}
        gain = power_per_turn(mon, name)
        when = (
            "is its level-up move at this level and it does not know it"
            if due
            else f"comes at L{level}"
        )
        line += f". {name} ({record.get('type')} {record.get('power')}, {gain} a turn) {when}"
    return line


def is_learn_prompt(screen_text: str) -> bool:
    """Is the game part-way through replacing a move right now."""
    return any(phrase in (screen_text or "") for phrase in LEARN_PROMPT_PHRASES)


def _learning_mon(party: Sequence[JsonDict], prompt: JsonDict) -> Optional[JsonDict]:
    """Which party member the prompt is about.

    The forget list draws the four move names on screen, so when it is up the
    Pokemon is the one whose moves are on it -- a check on ``wWhichPokemon``
    rather than a guess. Before the list appears there is nothing to check
    against and the slot byte is all there is.
    """
    text = str(prompt.get("screen_text") or "")
    matches = [
        mon
        for mon in party
        if isinstance(mon, dict)
        and move_names(mon)
        and all(name.upper() in text for name in move_names(mon))
    ]
    if len(matches) == 1:
        return matches[0]
    slot = prompt.get("slot")
    if isinstance(slot, int) and 0 <= slot < len(party):
        return party[slot]
    return party[0] if party else None


def learn_cost(prompt: Optional[JsonDict], party: Sequence[JsonDict]) -> Optional[str]:
    """What the button about to be pressed would cost the moveset.

    Named before the press, not after: no advice undoes a deleted move, and an
    HM move cannot be deleted at all, so the slot Cut went into is gone for the
    rest of the run.
    """
    if not prompt or not party:
        return None
    mon = _learning_mon(party, prompt)
    if mon is None:
        return None
    species = str(mon.get("species") or "it")
    known = move_names(mon)
    incoming = prompt.get("incoming")
    if incoming and (gamedata.move(str(incoming)) is None or str(incoming) in known):
        # A move id read out of a scratch byte that the screen does not back up.
        # Better to say nothing about it than to name the wrong move.
        incoming = None
    incoming_power = (gamedata.move(str(incoming)) or {}).get("power") if incoming else None
    new_move = f"{incoming} ({incoming_power or 0})" if incoming else "the new move"

    hms = hm_moves()
    slot = cheapest_slot(mon)
    cursor = prompt.get("cursor")
    on_list = FORGET_LIST_PHRASE in str(prompt.get("screen_text") or "")
    if on_list and isinstance(cursor, int) and 0 <= cursor < len(known):
        losing = known[cursor]
        losing_power = (gamedata.move(losing) or {}).get("power") or 0
        kept = [(name, power) for name, power in damaging_moves(mon) if name != losing]
        if incoming and incoming_power:
            kept.append((str(incoming), int(incoming_power)))
        left = ", ".join(f"{name} {power}" for name, power in kept) or "NOTHING that damages"
        line = (
            f"learn A here deletes {losing} ({losing_power}) for {new_move}. "
            f"{species} would be left attacking with {left}"
        )
        # What the button actually does when the cursor is on an HM move: nothing.
        # The frame reads "HM techniques can't be deleted" and the list comes
        # back. Measured: an agent read "A here deletes Cut", believed it, and
        # spent two episodes trying to press past a refusal it had been told was
        # a deletion. The cursor starts on slot 0, which is where Cut usually sits.
        if losing in hms:
            line += (
                f". {losing} is {hms[losing]} and Gen 1 refuses to delete an HM move, "
                "so A here will not take"
            )
        if slot is not None:
            name, per_turn, index = slot
            worth = "does no damage" if per_turn == 0 else f"is worth {per_turn} a turn"
            if index == cursor:
                line += f". {name} is the cheapest slot {species} has"
            else:
                steps = (index - cursor) % max(1, len(known))
                line += f". The cheapest slot is {name}, which {worth}: {steps} down from here"
        return line
    attacking = damaging_moves(mon)
    attacks = ", ".join(f"{name} {power}" for name, power in attacking)
    verb = "does" if len(attacking) == 1 else "do"
    tail = f"only {attacks} {verb} damage" if attacking else "none of them does damage"
    line = (
        f"learn {new_move} replaces one of {species}'s {len(known)} moves "
        f"({', '.join(known)}): {tail}"
    )
    # The name of the slot to spend, rather than two lists to subtract. The
    # payload used to stop at the line above; on the frames it was read, the
    # model twice worked out "replace Leer" from it and twice failed to and
    # abandoned the learn, and once deleted Slash by pressing A on the way past.
    if slot is not None and len(known) >= 4:
        name, per_turn, _ = slot
        worth = "does no damage" if per_turn == 0 else f"is worth {per_turn} a turn"
        line += f"; the cheapest slot is {name}, which {worth}"
    held_hms = [name for name in known if name in hms]
    if held_hms:
        labels = ", ".join(f"{name} is {hms[name]}" for name in held_hms)
        line += f"; {labels}, and Gen 1 deletes no HM move"
    # Same type as something already carried is the other half of the decision:
    # Flamethrower over Ember is not a fifth move, it is the same move harder.
    if incoming:
        kind = (gamedata.move(str(incoming)) or {}).get("type")
        gain = power_per_turn(mon, str(incoming))
        beaten = [
            f"{name} {value}"
            for name, value in ranked_attacks(mon)
            if (gamedata.move(name) or {}).get("type") == kind and value < gain
        ]
        if beaten:
            line += f"; {incoming} is {kind} at {gain} a turn and outclasses {', '.join(beaten)}"
    return line


class SayOnce:
    """Say a fact when it becomes true, not on every frame it stays true.

    The gym outlook is worth roughly 150 bytes and the run it was written for
    spent 3,044 presses in one gym; printed on every frame it would have cost
    more than the whole session's tool text. Keyed on (map, in a battle, the
    text itself), so walking back in, starting a fight, or changing the party
    says it again -- and standing still does not.
    """

    def __init__(self, remember: int = 8) -> None:
        self._seen: deque = deque(maxlen=remember)

    def fresh(self, key: Any) -> bool:
        if key in self._seen:
            return False
        self._seen.append(key)
        return True

    def reset(self) -> None:
        self._seen.clear()
