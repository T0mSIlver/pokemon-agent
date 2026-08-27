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


def _able(party: Sequence[JsonDict]) -> List[JsonDict]:
    """Party members that can still take a turn. A fainted one answers nothing."""
    return [mon for mon in party or () if isinstance(mon, dict) and (mon.get("hp") or 0) > 0]


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


#: How many machines one line is allowed to name. A bag late in the game holds
#: a dozen, and a payload field that grows without bound becomes wallpaper; the
#: rows are sorted by how much power they add, so the cut falls on the least
#: useful ones.
TEACHABLE_LIMIT = 3


def teachable_tms(party: Sequence[JsonDict], bag: Sequence[JsonDict]) -> Optional[str]:
    """Machines in the bag that teach a party member a *stronger* attack than it has.

    The third fact of this file's kind, and the one that cost the most. The run
    this was written for picked up TM28 in Mt. Moon, carried it for roughly
    60,000 presses on a single Charmeleon whose hardest attack was Cut, and
    never taught it. Nothing it read could have told it to: the bag says
    ``TM28 x1``, ``species.json`` says Charmeleon can learn ``TM28``, and until
    ``tms.json`` existed there was nothing anywhere in the harness that said
    TM28 is Dig. The two halves of the sentence were both in the payload and the
    join between them did not exist.

    Deliberately narrow, because a field on every overworld frame is expensive:

    * only machines actually in the bag, and only party members that can learn
      them, both read from pokered rather than assumed;
    * only a *strict* upgrade -- the taught move's power must beat the best
      damaging move that member already has -- so a mon holding its best moveset
      says nothing;
    * a move already known is not an upgrade, whatever its power.

    The type is printed beside the power because power is not the whole reason
    to teach one: Ground on an Electric gym is the reason Dig mattered here, and
    that is the agent's inference to make, not this line's.
    """
    able = _able(party)
    if not able or not bag:
        return None
    machines = []
    for entry in bag or ():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("item") or "")
        taught = gamedata.tm_move(label)
        record = gamedata.move(taught) if taught else None
        if not record or not record.get("power"):
            continue
        machines.append((label, str(taught), int(record["power"]), str(record["type"])))

    rows: List[Tuple[int, str]] = []
    for label, taught, power, kind in machines:
        best: Optional[Tuple[int, JsonDict, int]] = None
        for mon in able:
            entry = gamedata.species(str(mon.get("species") or "")) or {}
            if label not in (entry.get("tm_hm") or ()):
                continue
            if taught in move_names(mon):
                continue
            strongest = max((power for _, power in damaging_moves(mon)), default=0)
            if power <= strongest:
                continue
            if best is None or power - strongest > best[0]:
                best = (power - strongest, mon, strongest)
        if best is None:
            continue
        gain, mon, strongest = best
        species = str(mon.get("species") or "it")
        held = ", ".join(f"{name} {value}" for name, value in damaging_moves(mon)) or "nothing"
        rows.append(
            (
                gain,
                f"{label} teaches {taught} ({kind} {power}) and {species} can learn it; "
                f"it attacks with {held}",
            )
        )
    if not rows:
        return None
    rows.sort(key=lambda row: -row[0])
    return "bag " + " | ".join(line for _, line in rows[:TEACHABLE_LIMIT])


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

    cursor = prompt.get("cursor")
    on_list = FORGET_LIST_PHRASE in str(prompt.get("screen_text") or "")
    if on_list and isinstance(cursor, int) and 0 <= cursor < len(known):
        losing = known[cursor]
        losing_power = (gamedata.move(losing) or {}).get("power") or 0
        kept = [(name, power) for name, power in damaging_moves(mon) if name != losing]
        if incoming and incoming_power:
            kept.append((str(incoming), int(incoming_power)))
        left = ", ".join(f"{name} {power}" for name, power in kept) or "NOTHING that damages"
        return (
            f"learn A here deletes {losing} ({losing_power}) for {new_move}. "
            f"{species} would be left attacking with {left}"
        )
    attacking = damaging_moves(mon)
    attacks = ", ".join(f"{name} {power}" for name, power in attacking)
    verb = "does" if len(attacking) == 1 else "do"
    tail = f"only {attacks} {verb} damage" if attacking else "none of them does damage"
    return (
        f"learn {new_move} replaces one of {species}'s {len(known)} moves "
        f"({', '.join(known)}): {tail}"
    )


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
