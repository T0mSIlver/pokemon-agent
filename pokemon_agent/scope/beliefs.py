"""What the model said it believed, and whether it was true.

The expensive failures in this project have not been the model doing nothing.
They have been the model doing something decisive on a belief that was wrong —
walking four tiles east from a tile it was not standing on, stepping through a
wall it thought was floor, planning around a door that is not there.

Those beliefs are on disk. This model narrates nothing in its assistant text; it
narrates in the ``#`` comment it writes above each bash line, which is part of
the command and is therefore recorded verbatim next to the result:

    # Left 4 to (29,10)
    ./poke act left:4

That is a falsifiable prediction with its own answer key attached. ``./poke act``
replies with the map and tile the player ended on, so the claim and the truth sit
in the same transcript record, and the run store and ``pokemon_agent.gamedata``
supply the rest — where the warps are, which maps exist, how far apart they are.

Three rules keep the verdicts honest:

* A claim is only checked when the check is exact. An ambiguous sentence is
  recorded as *unchecked*, never guessed at, and the unchecked count is printed
  next to the others so the sample is always visible.
* A false claim is classified by *why* it was false, because "walked into a wall"
  and "was wrong about where it stood" are different bugs with different fixes.
* Nothing here parses natural language beyond coordinates and map names. There
  is no model in this loop and no scoring of intent — only claims a table can
  settle.

A fourth rule was learned the expensive way: **before accusing the model, check
that the answer key was read.** This module reported 1,164 of 2,055 position
claims false — 57% — over one 34-hour run, and the number was almost entirely
its own. ``poke act`` stopped printing JSON on 2026-08-26 and the reader here
only knew JSON, so 9,029 of that run's 13,400 position answers were invisible
and every claim after one of them was settled against a tile the player had left
hours earlier. It also read ``poke sim`` waypoints, and sentences about the wall
ahead, as claims about where the player stood. With those three fixed the same
transcripts read 8%; on the calls that actually press a button, 2,503 of 2,708
position claims are right. The payload was never wrong and the model mostly was
not either.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from pokemon_agent.scope import truth
from pokemon_agent.scope.transcript import Call, Session, poke_verbs

Coord = tuple[int, int]

#: ``(12, 3)`` and ``(12,3)`` are the same tile. Three digits is more than any
#: Red map is wide, and refusing four keeps years and press counts out.
_COORD = r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)"
_COORD_RE = re.compile(_COORD)

#: "from (11,6)", "I'm at (20,12)", "stuck at (20,12)" — a claim about where the
#: player is standing *before* the command runs.
_ORIGIN_RE = re.compile(
    r"\b(?:from|i['’]?m\s+at|i\s+am\s+at|currently\s+at|now\s+at|stuck\s+at|standing\s+on)\s*"
    + _COORD,
    re.IGNORECASE,
)

#: "left 4 to (29,10)" — a claim about where the command will end.
_DESTINATION_RE = re.compile(r"\bto\s*" + _COORD, re.IGNORECASE)

#: "warp (3,7)", "the door at (11,5)", "ladder (25,9)".
_FEATURE_RE = re.compile(
    r"\b(warp|door|ladder|stairs|staircase|exit)\b[^().]{0,24}?" + _COORD,
    re.IGNORECASE,
)

#: ``Mt Moon 1F (15,34) facing up`` — how every acting verb has answered since
#: ``poke act`` stopped printing JSON. Not anchored to the start of the line
#: because the model pipes the answer through its own scripts.
_PROSE_POSITION_RE = re.compile(
    r"(?P<map>[A-Za-z][A-Za-z0-9 .'’-]*?)\s*\((?P<x>\d{1,3}),\s*(?P<y>\d{1,3})\)\s+facing\b"
)

#: ``clean: ends at (24, 7)``, ``blocked at step 3 ... stops at (14, 11)`` — a
#: tile ``poke sim`` reported, which is a tile on paper and not one the player
#: is standing on.
_SIM_END_RE = re.compile(r"(?:ends at|stops at)\s*\((\d{1,3}),\s*(\d{1,3})\)")

#: ``from Mt Moon B1F (17,11): clean: ends at ...`` — the tile ``poke sim``
#: walked the plan from, which *is* where the player is standing. Worth reading:
#: a chain of sims answers nothing else, and one such chain ran 538 calls.
_SIM_START_RE = re.compile(
    r"\bfrom\s+(?P<map>[A-Za-z][A-Za-z0-9 .'’-]*?)\s*\((?P<x>\d{1,3}),\s*(?P<y>\d{1,3})\):"
)

#: How many simulated endpoints stay in scope. The model chains sims — sim,
#: read the endpoint, sim the same prefix plus one more leg — and it names the
#: endpoint of a sim a handful of calls back, not one from an hour ago.
_SIM_MEMORY = 8

#: "(22,20) has an NPC on it", "(63,4) is a wall" — a bare leading coordinate
#: followed by a property of *that* tile. The model writes its own tile with
#: "from" or "I'm at"; this shape names the obstacle in front of it, and 176 of
#: the 318 such sentences the checker called false were the model correctly
#: describing the tile it had just been refused.
_TILE_PROPERTY_RE = re.compile(
    r"\b(?:blocked|block|wall|walled|npc|ledge|rock|impassable|trainer|boulder|sign|water)\b",
    re.IGNORECASE,
)

#: The verbs that move the player and answer with where it ended up.
MOVEMENT_VERBS = frozenset({"act", "goto"})

#: A map name handed to ``poke route`` or ``poke goto``. Bare coordinates are a
#: different kind of argument and are skipped by the leading-letter requirement.
_TARGET_RE = re.compile(
    r"poke\s+(route|goto)\s+(\"[^\"]+\"|'[^']+'|(?:[A-Za-z][\w.']*(?:\\?\s+[\w.']+)*))"
)

#: Verdicts. ``unchecked`` is a first-class answer, not a failure to produce one.
TRUE = "true"
FALSE = "false"
UNCHECKED = "unchecked"

#: Why a false claim was false. Ordered by how specific the diagnosis is.
WHY_MAP_CHANGED = "map changed under it"
WHY_BLOCKED = "walked into a wall"
WHY_MISPLACED = "wrong about where it stood"
WHY_NO_SUCH_WARP = "no warp on that tile"
WHY_NO_SUCH_MAP = "no such map"
WHY_OTHER = "other"

#: Claim kinds, in the order the report prints them.
KINDS = ("position", "destination", "warp", "map")


@dataclass(frozen=True)
class Claim:
    """One falsifiable thing the model wrote, and the verdict on it."""

    step: int
    kind: str
    #: The sentence the claim was read out of, trimmed.
    said: str
    #: The map the player was standing on when it was written.
    map_name: str = ""
    claimed: Any = None
    actual: Any = None
    verdict: str = UNCHECKED
    why: str = ""

    @property
    def offset(self) -> Optional[Coord]:
        """``claimed - actual`` for the two coordinate kinds, else ``None``."""

        if not (isinstance(self.claimed, tuple) and isinstance(self.actual, tuple)):
            return None
        if len(self.claimed) != 2 or len(self.actual) != 2:
            return None
        return (self.claimed[0] - self.actual[0], self.claimed[1] - self.actual[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "said": self.said,
            "map": self.map_name,
            "claimed": list(self.claimed) if isinstance(self.claimed, tuple) else self.claimed,
            "actual": list(self.actual) if isinstance(self.actual, tuple) else self.actual,
            "verdict": self.verdict,
            "why": self.why,
            "offset": list(self.offset) if self.offset else None,
        }


def comment_of(command: str) -> str:
    """The ``#`` lines of a bash command, joined into one sentence.

    A trailing ``# comment`` on a line that also runs something is not taken:
    the model writes its reasoning on its own line, and ``echo 'a # b'`` would
    otherwise be read as narration.
    """

    parts = [
        line.strip().lstrip("#").strip()
        for line in command.splitlines()
        if line.strip().startswith("#")
    ]
    return " ".join(part for part in parts if part)


def _coords(text: str) -> list[Coord]:
    return [(int(a), int(b)) for a, b in _COORD_RE.findall(text)]


def _prose_position(text: str) -> Optional[tuple[str, Coord]]:
    """``(map, (x, y))`` off the ``Route 3 (63,0) facing up`` line, if there is one.

    The last match in the output wins, because a line that runs two verbs prints
    two answers and the second one is where the player ended up. ``poke sim``'s
    ``from Mt Moon B1F (17,11):`` counts as one of them: it is the live tile the
    plan was walked from, and during a chain of sims it is the only answer there.

    A match only counts when a real map name sits in front of the tile. The model
    pipes the answer through its own scripts — ``now at Route 4 (13,11) facing
    up``, ``y=0 -> Route 3 (63,0) facing up`` — so the name is taken as the
    longest suffix of the leading text that the game actually has. Insisting on
    that is what keeps ``clean: ends at (24, 7) facing right`` out: ``poke sim``
    answers in the same shape and is reporting a tile on paper, and reading it
    as the live one is the whole failure this reader exists to stop.
    """

    if not truth.known_maps():
        return None
    found: Optional[tuple[str, Coord]] = None
    matches = sorted(
        [*_PROSE_POSITION_RE.finditer(text), *_SIM_START_RE.finditer(text)],
        key=lambda match: match.start(),
    )
    for match in matches:
        words = match.group("map").split()
        for start in range(len(words)):
            name = " ".join(words[start:])
            if truth.map_truth(name) is not None:
                found = (name, (int(match.group("x")), int(match.group("y"))))
                break
    return found


def _result_position(call: Call) -> Optional[tuple[str, Coord]]:
    """``(map, (x, y))`` from a call that answered with the player's state.

    Both answer shapes count. ``poke act`` printed a JSON object until
    2026-08-26 and has printed ``Mt Moon 1F (15,34) facing up`` since, and a
    reader that only knew the first shape went blind the moment the second
    shipped: 9,029 of the 13,400 position answers in the 34-hour run were
    invisible to it. Every claim after one of them was then judged against a
    tile the player had left hours earlier, which is what turned 1,201 correct
    sentences into "wrong about where it stood" — the report's own worst
    example, ``(63,0)`` against an actual ``(15,34)``, was the player standing
    exactly where it said, on Route 3, while the checker still believed Mt Moon.
    """

    payload = call.result_json
    if payload:
        x, y = payload.get("x"), payload.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return str(payload.get("map") or ""), (x, y)
    return _prose_position(call.result_text or "")


def _movement_calls(command: str) -> int:
    return sum(1 for verb in poke_verbs(command) if verb in MOVEMENT_VERBS)


def _requested_presses(command: str) -> Optional[int]:
    """How many buttons ``poke act`` was asked for, if it can be counted."""

    match = re.search(r"poke\s+act\s+([^\n;&|]*)", command)
    if not match:
        return None
    total = 0
    for token in match.group(1).split():
        if token.startswith("-"):
            continue
        head, _, count = token.partition(":")
        if not head.isalpha():
            continue
        if count:
            if not count.isdigit():
                return None
            total += int(count)
        else:
            total += 1
    return total or None


def _classify_destination(
    call: Call,
    before: Optional[tuple[str, Coord]],
    after: tuple[str, Coord],
    claimed: Coord,
) -> str:
    """Why a stated destination did not happen."""

    after_map, after_pos = after
    if before is not None and before[0] and before[0] != after_map:
        return WHY_MAP_CHANGED
    payload = call.result_json or {}
    requested = _requested_presses(call.command)
    moved = payload.get("moved")
    if payload.get("blocked_after") is not None:
        return WHY_BLOCKED
    if isinstance(moved, int) and requested is not None and moved < requested:
        return WHY_BLOCKED
    if before is not None:
        walked = (after_pos[0] - before[1][0], after_pos[1] - before[1][1])
        implied_start = (claimed[0] - walked[0], claimed[1] - walked[1])
        if implied_start != before[1]:
            return WHY_MISPLACED
    return WHY_OTHER


def _target_names(command: str) -> list[str]:
    """Map names passed to ``poke route`` / ``poke goto``, unquoted."""

    out: list[str] = []
    for _, raw in _TARGET_RE.findall(command):
        name = raw.strip().strip("\"'")
        name = name.replace("\\ ", " ").strip()
        # Shell noise: a redirect or a variable is not a claim about a map.
        if not name or name.startswith("$") or any(ch in name for ch in "><|$"):
            continue
        out.append(" ".join(name.split()))
    return out


def extract_claims(session: Session) -> list[Claim]:
    """Every checkable claim in one transcript, in the order it was written."""

    claims: list[Claim] = []
    here: Optional[tuple[str, Coord]] = None
    simulated: deque[Coord] = deque(maxlen=_SIM_MEMORY)

    for call in session.calls:
        if call.tool != "bash":
            continue
        said = comment_of(call.command)
        after = _result_position(call)
        map_now = here[0] if here else (after[0] if after else "")

        for name in _target_names(call.command):
            known = truth.map_truth(name)
            claims.append(
                Claim(
                    step=call.step,
                    kind="map",
                    said=f"{name} is a map",
                    map_name=map_now,
                    claimed=name,
                    actual=None if known is None else name,
                    verdict=FALSE if (known is None and truth.known_maps()) else TRUE,
                    why=WHY_NO_SUCH_MAP if known is None else "",
                )
            )

        if said:
            coords = _coords(said)
            claims.extend(_position_claims(call, said, coords, here, map_now, simulated))
            claims.extend(_warp_claims(call, said, map_now))
            if len(coords) == 1 and _movement_calls(call.command) == 1:
                claims.extend(_destination_claims(call, said, here, after, map_now))
            elif len(coords) > 1:
                claims.append(
                    Claim(
                        step=call.step,
                        kind="destination",
                        said=said,
                        map_name=map_now,
                        claimed=coords[-1],
                        verdict=UNCHECKED,
                        why="multi-leg plan; only the last leg would land here",
                    )
                )

        for x_text, y_text in _SIM_END_RE.findall(call.result_text or ""):
            simulated.append((int(x_text), int(y_text)))
        if after is not None:
            here = after
    return claims


def _position_claims(
    call: Call,
    said: str,
    coords: list[Coord],
    here: Optional[tuple[str, Coord]],
    map_now: str,
    simulated: "deque[Coord]",
) -> list[Claim]:
    match = _ORIGIN_RE.search(said)
    if match:
        claimed = (int(match.group(1)), int(match.group(2)))
    elif said.startswith("(") and coords:
        claimed = coords[0]
        # "(22,20) has an NPC on it" is a true sentence about the tile ahead,
        # written by a model standing next to it. Reading it as "I am at
        # (22,20)" invents a lie the model never told.
        if _TILE_PROPERTY_RE.search(said):
            return [
                Claim(
                    step=call.step,
                    kind="position",
                    said=said,
                    map_name=map_now,
                    claimed=claimed,
                    verdict=UNCHECKED,
                    why="names a property of that tile, not where the player stands",
                )
            ]
    else:
        return []
    # A plan waypoint is not a standing claim. `poke sim` walks a plan on paper
    # from the live tile and answers with where it would stop; the model then
    # sims the same prefix plus one more leg and calls the last endpoint "from
    # (14,8)". 616 of the 701 such sentences in the 34-hour run named a tile a
    # sim had just printed, and the player had not moved for any of them.
    if claimed in simulated and _movement_calls(call.command) == 0:
        return [
            Claim(
                step=call.step,
                kind="position",
                said=said,
                map_name=map_now,
                claimed=claimed,
                verdict=UNCHECKED,
                why="a tile poke sim just reported; a plan waypoint, not a standing claim",
            )
        ]
    if here is None:
        return [
            Claim(
                step=call.step,
                kind="position",
                said=said,
                map_name=map_now,
                claimed=claimed,
                verdict=UNCHECKED,
                why="no position recorded yet",
            )
        ]
    correct = claimed == here[1]
    return [
        Claim(
            step=call.step,
            kind="position",
            said=said,
            map_name=here[0],
            claimed=claimed,
            actual=here[1],
            verdict=TRUE if correct else FALSE,
            why="" if correct else WHY_MISPLACED,
        )
    ]


def _destination_claims(
    call: Call,
    said: str,
    here: Optional[tuple[str, Coord]],
    after: Optional[tuple[str, Coord]],
    map_now: str,
) -> list[Claim]:
    match = _DESTINATION_RE.search(said)
    if not match:
        return []
    claimed = (int(match.group(1)), int(match.group(2)))
    if after is None:
        return [
            Claim(
                step=call.step,
                kind="destination",
                said=said,
                map_name=map_now,
                claimed=claimed,
                verdict=UNCHECKED,
                why="the call did not answer with a position",
            )
        ]
    correct = claimed == after[1]
    return [
        Claim(
            step=call.step,
            kind="destination",
            said=said,
            map_name=after[0],
            claimed=claimed,
            actual=after[1],
            verdict=TRUE if correct else FALSE,
            why="" if correct else _classify_destination(call, here, after, claimed),
        )
    ]


def _warp_claims(call: Call, said: str, map_now: str) -> list[Claim]:
    out: list[Claim] = []
    known = truth.map_truth(map_now)
    for word, x_text, y_text in _FEATURE_RE.findall(said):
        claimed = (int(x_text), int(y_text))
        if known is None:
            out.append(
                Claim(
                    step=call.step,
                    kind="warp",
                    said=said,
                    map_name=map_now,
                    claimed=claimed,
                    verdict=UNCHECKED,
                    why="no game data for this map",
                )
            )
            continue
        warp = known.warp_at(*claimed)
        out.append(
            Claim(
                step=call.step,
                kind="warp",
                said=f"{word.lower()} at {claimed}",
                map_name=map_now,
                claimed=claimed,
                actual=None if warp is None else warp.to_map,
                verdict=TRUE if warp is not None else FALSE,
                why="" if warp is not None else WHY_NO_SUCH_WARP,
            )
        )
    return out


@dataclass(frozen=True)
class KindTally:
    """Verdicts for one kind of claim."""

    kind: str
    true: int = 0
    false: int = 0
    unchecked: int = 0

    @property
    def checked(self) -> int:
        return self.true + self.false

    @property
    def wrong_share(self) -> float:
        return (self.false / self.checked) if self.checked else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "true": self.true,
            "false": self.false,
            "unchecked": self.unchecked,
            "checked": self.checked,
            "wrong_share": round(self.wrong_share, 3),
        }


@dataclass(frozen=True)
class ClaimsReport:
    """Every claim in a run's transcripts, tallied and diagnosed."""

    sessions: tuple[str, ...]
    narrating_calls: int
    total_calls: int
    tallies: tuple[KindTally, ...]
    reasons: tuple[tuple[str, int], ...]
    offsets: tuple[tuple[Coord, int], ...]
    worst: tuple[Claim, ...]

    @property
    def checked(self) -> int:
        return sum(tally.checked for tally in self.tallies)

    @property
    def wrong(self) -> int:
        return sum(tally.false for tally in self.tallies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": list(self.sessions),
            "narrating_calls": self.narrating_calls,
            "total_calls": self.total_calls,
            "checked": self.checked,
            "wrong": self.wrong,
            "by_kind": [tally.to_dict() for tally in self.tallies],
            "reasons": [{"why": why, "count": count} for why, count in self.reasons],
            "offsets": [{"offset": list(offset), "count": count} for offset, count in self.offsets],
            "worst": [claim.to_dict() for claim in self.worst],
        }


def claims_report(sessions: Iterable[Session], *, limit: int = 12) -> ClaimsReport:
    """Tally the claims of one or more transcripts into one verdict."""

    session_list = list(sessions)
    claims: list[Claim] = []
    narrating = 0
    total = 0
    for session in session_list:
        claims.extend(extract_claims(session))
        for call in session.calls:
            if call.tool != "bash":
                continue
            total += 1
            if comment_of(call.command):
                narrating += 1

    counts: dict[str, dict[str, int]] = {kind: {TRUE: 0, FALSE: 0, UNCHECKED: 0} for kind in KINDS}
    reasons: dict[str, int] = {}
    offsets: dict[Coord, int] = {}
    for claim in claims:
        bucket = counts.setdefault(claim.kind, {TRUE: 0, FALSE: 0, UNCHECKED: 0})
        bucket[claim.verdict] = bucket.get(claim.verdict, 0) + 1
        if claim.verdict != FALSE:
            continue
        if claim.why:
            reasons[claim.why] = reasons.get(claim.why, 0) + 1
        shift = claim.offset
        if shift is not None and shift != (0, 0):
            offsets[shift] = offsets.get(shift, 0) + 1

    tallies = tuple(
        KindTally(
            kind=kind,
            true=counts[kind][TRUE],
            false=counts[kind][FALSE],
            unchecked=counts[kind][UNCHECKED],
        )
        for kind in KINDS
        if any(counts.get(kind, {}).values())
    )

    def severity(claim: Claim) -> tuple[int, int]:
        shift = claim.offset
        distance = abs(shift[0]) + abs(shift[1]) if shift else 0
        return (distance, -claim.step)

    worst = sorted(
        (claim for claim in claims if claim.verdict == FALSE), key=severity, reverse=True
    )[:limit]

    return ClaimsReport(
        sessions=tuple(session.short_id for session in session_list),
        narrating_calls=narrating,
        total_calls=total,
        tallies=tallies,
        reasons=tuple(sorted(reasons.items(), key=lambda item: -item[1])),
        offsets=tuple(sorted(offsets.items(), key=lambda item: -item[1])),
        worst=tuple(worst),
    )
