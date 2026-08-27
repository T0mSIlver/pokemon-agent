#!/usr/bin/env python3
"""The game as a Python library, so the agent can plan in code instead of clicks.

``./poke`` drives the game one call at a time. Crossing Mt. Moon that way is
thirty tool calls and thirty observations in the model's context. With this the
same crossing is one script: read the guide section, compute a route, simulate
it, execute it in legal batches, and report five lines.

    import poke

    plan = poke.sim("up:6", "right:3")
    if plan.ok:
        poke.act("up:6", "right:3")

Everything is blocking and synchronous, every refusal is an exception carrying
the server's own words, and every answer is an object with attributes a script
can branch on rather than a dict to spelunk. The raw payload is always on
``.raw`` when something here has shaped away a field you need.

Standalone by design: stdlib only, no package imports. It is copied into the
agent's workspace as ``poke.py`` next to the ``poke`` CLI and imported from
there — run it with the workspace's ``./py`` wrapper, which puts the workspace
on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

DEFAULT_PORT = 8765
TIMEOUT_SECONDS = 120.0

DIRECTIONS = ("up", "down", "left", "right")
BUTTONS = ("a", "b", "start", "select")

#: Every action the server's parser accepts by name.
ACTIONS = (
    *(f"walk_{direction}" for direction in DIRECTIONS),
    *(f"press_{button}" for button in BUTTONS),
    "hold_a_30",
    "wait_60",
    "a_until_dialog_end",
)

#: Short forms, the same vocabulary ``poke act`` takes.
ALIASES = {
    **{direction: f"walk_{direction}" for direction in DIRECTIONS},
    **{button: f"press_{button}" for button in BUTTONS},
    "wait": "wait_60",
    "adialog": "a_until_dialog_end",
}

#: The server's limits, mirrored so a bad plan fails here instead of as a 400.
#: ``tests/test_agent_api.py`` asserts these still equal the CLI's, because the
#: CLI is where they are kept honest against the server.
MAX_REPEAT = 40
MAX_ACTIONS_PER_BATCH = 40
MAX_FRAMES_PER_ACTION = 600
MAX_FRAMES_PER_BATCH = 3600

FRAMES_PER_INPUT = 20
FRAMES_DIALOG_WORST_CASE = 300

#: The server refuses more than 60 action batches a minute, and it counts the
#: CLI's batches and this client's in the same budget. Pacing below its cap
#: means a long walk waits its turn instead of dying on a 429 halfway across a
#: map, and leaves room for a ``./poke`` call in the same minute.
RATE_WINDOW_SECONDS = 60.0
RATE_MAX_BATCHES = 50

#: Longest this client will ever sleep waiting for the rate window to open. A
#: wait longer than the whole window means the clock went backwards or another
#: process is hammering the server; say so rather than hang.
MAX_PACE_WAIT_SECONDS = RATE_WINDOW_SECONDS + 5.0

#: A ceiling on one :func:`walk`. Ten legal batches is most of the way across
#: the biggest map in the game; asking for more is a loop, not a route. This is
#: the number that stops a script becoming the runaway that once sent 5,550
#: calls: a walk cannot exceed it, and the report says what was left unsent.
WALK_MAX_ACTIONS = 400

FRAME_FILES = {
    "raw": "latest_frame.png",
    "annotated": "latest_frame_annotated.png",
}

Tokens = Union[str, Iterable[str]]
Coord = tuple[int, int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PokeError(Exception):
    """Anything this module refuses or the server refuses. Catch this to catch all."""


class ActionError(PokeError):
    """A plan that is wrong before it is sent: a bad name, an illegal batch."""


class ServerError(PokeError):
    """The server said no. The message is the server's own words, verbatim.

    A refusal is information — a 409 says the game is not on the overworld, a
    429 says you are in a loop, a 404 names what does not exist — so it is
    raised rather than swallowed into a None the caller will ignore.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Unreachable(PokeError):
    """Nothing is listening. The server is down or the port is wrong."""


class RateLimited(PokeError):
    """The client's own pacing could not open a slot in a reasonable time."""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def resolve_action(token: str) -> Optional[str]:
    """Canonical action name for one token, or None if it is not an action."""

    name = str(token).strip().lower()
    if name in ALIASES:
        return ALIASES[name]
    if name in ACTIONS:
        return name
    parts = name.split("_")
    if len(parts) == 2 and parts[0] == "wait" and parts[1].isdigit() and int(parts[1]) > 0:
        return name
    if (
        len(parts) == 3
        and parts[0] == "hold"
        and parts[1] in BUTTONS + DIRECTIONS
        and parts[2].isdigit()
        and int(parts[2]) > 0
    ):
        return name
    return None


def frames_for(action: str) -> int:
    """Emulator frames one canonical action costs, for the batch budget."""

    if action == "a_until_dialog_end":
        return FRAMES_DIALOG_WORST_CASE
    parts = action.split("_")
    if parts[0] == "wait":
        return int(parts[1])
    if parts[0] == "hold":
        return int(parts[2])
    return FRAMES_PER_INPUT


def action_help() -> str:
    return (
        "actions: up down left right a b start select wait adialog\n"
        "  long form: " + " ".join(ACTIONS) + "\n"
        "  also wait_N and hold_<button>_N for any frame count\n"
        "  repeat with a colon: 'up:4' is four walk_up"
    )


def _flatten(tokens: Sequence[Tokens]) -> list[str]:
    """``act("up", "a")``, ``act(["up", "a"])`` and ``act(*plan)`` all mean the same.

    So does ``act("up a")``. The CLI takes ``poke sim down:5 right:2`` and the
    shell splits it, so that is the form every example and every habit is in;
    written as one Python string it used to arrive as a single token and raise
    ``unknown action 'down:5 right:2'``. It cost four tracebacks and an
    abandoned probe loop across the sessions. No action name has a space in it,
    so splitting one can never mean anything else.
    """

    flat: list[str] = []
    for token in tokens:
        if isinstance(token, str):
            flat.extend(token.split())
        elif isinstance(token, Iterable):
            for part in token:
                flat.extend(str(part).split())
        else:
            raise ActionError(f"{token!r} is not an action or a list of actions")
    return flat


def expand(*tokens: Tokens, max_repeat: int = MAX_REPEAT) -> list[str]:
    """Tokens as typed into the action list the server expects.

    Names, aliases and the ``name:count`` repeat form. This does *not* check the
    per-batch caps — :func:`chunks` splits a long plan and :func:`Client.act`
    refuses one that is too long for a single batch.
    """

    actions: list[str] = []
    for token in _flatten(tokens):
        name, separator, count_text = str(token).partition(":")
        count = 1
        if separator:
            if not count_text.isdigit() or int(count_text) < 1:
                raise ActionError(f"bad repeat count in {token!r} - 'up:4' is four walk_up")
            count = int(count_text)
            if count > max_repeat:
                raise ActionError(f"{token!r} repeats more than {max_repeat} times")
        action = resolve_action(name)
        if action is None:
            raise ActionError(f"unknown action {name!r}\n{action_help()}")
        frames = frames_for(action)
        if frames > MAX_FRAMES_PER_ACTION:
            raise ActionError(
                f"{token!r} asks for {frames} frames; the limit is "
                f"{MAX_FRAMES_PER_ACTION} ({MAX_FRAMES_PER_ACTION // 60} seconds)"
            )
        actions.extend([action] * count)
    if not actions:
        raise ActionError(f"no actions given\n{action_help()}")
    return actions


def one_batch(*tokens: Tokens) -> list[str]:
    """Expand tokens and refuse anything the server would refuse as one batch."""

    actions = expand(*tokens)
    if len(actions) > MAX_ACTIONS_PER_BATCH:
        raise ActionError(
            f"that batch is {len(actions)} actions; the limit is {MAX_ACTIONS_PER_BATCH}. "
            "Use walk() to send a long path in legal chunks."
        )
    total = sum(frames_for(action) for action in actions)
    if total > MAX_FRAMES_PER_BATCH:
        raise ActionError(
            f"that batch runs {total} frames; the limit is {MAX_FRAMES_PER_BATCH} "
            f"({MAX_FRAMES_PER_BATCH // 60} seconds). Use walk() to split it."
        )
    return actions


def chunks(actions: Sequence[str]) -> list[list[str]]:
    """Split a plan into batches the server will accept, in order.

    Both caps bind: forty actions, and thirty-six hundred frames. A batch of
    forty walks is 800 frames and fits; a batch of forty ``adialog`` would be
    12,000 and does not, so the frame budget closes it early.
    """

    batches: list[list[str]] = []
    current: list[str] = []
    frames = 0
    for action in actions:
        cost = frames_for(action)
        if current and (
            len(current) >= MAX_ACTIONS_PER_BATCH or frames + cost > MAX_FRAMES_PER_BATCH
        ):
            batches.append(current)
            current, frames = [], 0
        current.append(action)
        frames += cost
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# Shaped answers
# ---------------------------------------------------------------------------


def _coord(value: Any) -> Optional[Coord]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _hp_pair(text: Any) -> tuple[Optional[int], Optional[int]]:
    """``"19/36"`` as numbers. The server sends HP as a string to keep it one field."""

    if not isinstance(text, str) or "/" not in text:
        return None, None
    current, _, maximum = text.partition("/")
    try:
        return int(current), int(maximum)
    except ValueError:
        return None, None


@dataclass
class Mon:
    """One Pokemon, yours or theirs."""

    species: Optional[str] = None
    level: Optional[int] = None
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    types: list[str] = field(default_factory=list)
    status: str = "OK"
    moves: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def fainted(self) -> bool:
        return self.hp == 0

    @property
    def hp_fraction(self) -> Optional[float]:
        if not self.max_hp:
            return None
        return (self.hp or 0) / self.max_hp

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "Mon":
        payload = payload or {}
        raw_moves = payload.get("moves") or []
        names = [m["name"] for m in raw_moves if isinstance(m, dict) and m.get("name")]
        return cls(
            species=payload.get("species"),
            level=payload.get("level"),
            hp=payload.get("hp"),
            max_hp=payload.get("max_hp"),
            types=list(payload.get("types") or []),
            status=payload.get("status") or "OK",
            moves=names or [m for m in raw_moves if isinstance(m, str)],
            raw=payload,
        )

    def __str__(self) -> str:
        types = "/".join(self.types) or "?"
        tail = "" if self.status == "OK" else f" {self.status}"
        return f"{self.species} L{self.level} {self.hp}/{self.max_hp} {types}{tail}"


@dataclass
class State:
    """Party, bag, badges and position: everything ``GET /state`` knows."""

    map: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    facing: Optional[str] = None
    money: Optional[int] = None
    badges: list[str] = field(default_factory=list)
    party: list[Mon] = field(default_factory=list)
    bag: dict[str, int] = field(default_factory=dict)
    in_battle: bool = False
    enemy: Optional[Mon] = None
    dialog: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def position(self) -> Optional[Coord]:
        return None if self.x is None or self.y is None else (self.x, self.y)

    @property
    def lead(self) -> Optional[Mon]:
        return self.party[0] if self.party else None

    @property
    def hp(self) -> Optional[int]:
        return self.lead.hp if self.lead else None

    @property
    def max_hp(self) -> Optional[int]:
        return self.lead.max_hp if self.lead else None

    def has(self, item: str) -> int:
        """How many of an item are in the bag, matched without case. 0 for none."""

        wanted = item.strip().lower()
        return next((n for name, n in self.bag.items() if name.lower() == wanted), 0)

    @classmethod
    def from_payload(cls, payload: dict) -> "State":
        player = payload.get("player") or {}
        position = _coord(player.get("position")) or (None, None)
        battle = payload.get("battle") or {}
        return cls(
            map=(payload.get("map") or {}).get("map_name"),
            x=position[0],
            y=position[1],
            facing=player.get("facing"),
            money=player.get("money"),
            badges=list(player.get("badges") or (payload.get("flags") or {}).get("badges") or []),
            party=[Mon.from_payload(mon) for mon in payload.get("party") or []],
            bag={
                entry.get("item"): entry.get("quantity")
                for entry in payload.get("bag") or []
                if entry.get("item")
            },
            in_battle=bool(battle.get("in_battle")),
            enemy=Mon.from_payload(battle.get("enemy")) if battle.get("enemy") else None,
            dialog=bool(
                payload.get("dialog_active") or (payload.get("dialog") or {}).get("active")
            ),
            raw=payload,
        )

    def __str__(self) -> str:
        lines = [f"{self.map} ({self.x},{self.y}) facing {self.facing}"]
        lines.extend(f"  {mon}" for mon in self.party)
        lines.append(f"badges: {', '.join(self.badges) or 'none'}   money: {self.money}")
        items = ", ".join(f"{name} x{count}" for name, count in self.bag.items())
        lines.append(f"bag: {items or 'empty'}")
        if self.in_battle:
            lines.append(f"battle: {self.enemy or 'yes'}")
        return "\n".join(lines)


@dataclass
class Result:
    """What one batch did. The answer every acting call gives back.

    ``moved`` and ``blocked_after`` are the two that tell a sixteen-step walk
    from one step and fifteen presses of your face against a tree; the server
    omits them when the batch had no walks in it, and they are None here.
    """

    actions_executed: int = 0
    map: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    facing: Optional[str] = None
    #: Directions that are walkable from where the player is standing. Empty on a
    #: frame that cannot take a step at all — read ``no_walk`` before concluding
    #: anything from that, because "in a battle" and "walled in" look identical
    #: from an empty list.
    directions: list[str] = field(default_factory=list)
    #: Why there are no directions, when the frame is a battle or an open box.
    no_walk: Optional[str] = None
    mode: Optional[str] = None
    dialog: bool = False
    in_battle: bool = False
    hp: Optional[int] = None
    max_hp: Optional[int] = None
    moved: Optional[int] = None
    blocked_after: Optional[int] = None
    here_before: Optional[int] = None
    #: Set on the one batch that landed after a whiteout, saying where the party
    #: went down, where the game put it, and what the halving cost. It is the
    #: only map change in the game the player did not make, and without this the
    #: payload renders it as an ordinary walk that happens to arrive at full HP.
    whiteout: Optional[str] = None
    on_warp: bool = False
    warp: dict = field(default_factory=dict)
    faces: Optional[str] = None
    screen_text: str = ""
    enemy: Optional[str] = None
    #: The Pokemon on the field on your side, as ``"Charmeleon L25"``. The level
    #: is the half of the fight-or-flee comparison the payload never carried.
    you: Optional[str] = None
    #: The active Pokemon's moves, each priced against what is on the other side:
    #: ``"Ember Fire 12PP 41-49 x2 KO in 1"``. Names alone are not enough to pick.
    battle_moves: list[str] = field(default_factory=list)
    #: The enemy's hardest hit and its name, ``"Leech Life up to 4"``.
    incoming: Optional[str] = None
    #: Set when nothing with PP left does damage: the fight cannot be won and a
    #: trainer cannot be escaped.
    no_damage: Optional[str] = None
    #: Set when the engine has taken the turn — Rage keeps swinging and gives no
    #: menu — so ``fight`` and ``run`` will both refuse until it ends.
    locked_in: Optional[str] = None
    #: On a wild encounter: the balls carried and what each would do, or what a
    #: ball costs when none are carried. ``"Poke Ball x11 36% now / 100% worn
    #: down"``. Never set in a trainer battle, where a ball bounces off.
    catch: Optional[str] = None
    #: Inside a mart: the money, the stock and the prices. Never set anywhere
    #: else, because 211 of the game's 223 maps sell nothing.
    shop: Optional[str] = None
    #: On a map with a nurse: her tile, and what she would fix. Never set
    #: anywhere else, because 210 of the game's 223 maps have no nurse.
    heal: Optional[str] = None
    menu: Optional[str] = None
    highlighted: Optional[str] = None
    #: Only ``goto`` sets these three.
    walked: Optional[int] = None
    arrived: Optional[bool] = None
    stopped_because: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def position(self) -> Optional[Coord]:
        return None if self.x is None or self.y is None else (self.x, self.y)

    @property
    def _where(self) -> str:
        """``(x,y)``, or a phrase saying it was not read. Never ``(None,None)``."""
        position = self.position
        return "(position unread)" if position is None else f"({position[0]},{position[1]})"

    @property
    def blocked(self) -> bool:
        return self.blocked_after is not None

    @property
    def hp_fraction(self) -> Optional[float]:
        if not self.max_hp:
            return None
        return (self.hp or 0) / self.max_hp

    @classmethod
    def from_payload(cls, payload: dict) -> "Result":
        hp, max_hp = _hp_pair(payload.get("hp"))
        return cls(
            actions_executed=payload.get("actions_executed") or 0,
            map=payload.get("map"),
            x=payload.get("x"),
            y=payload.get("y"),
            facing=payload.get("facing"),
            directions=list(payload.get("moves") or []),
            no_walk=payload.get("no_walk"),
            mode=payload.get("mode"),
            dialog=bool(payload.get("dialog")),
            in_battle=bool(payload.get("battle")),
            hp=hp,
            max_hp=max_hp,
            moved=payload.get("moved"),
            blocked_after=payload.get("blocked_after"),
            here_before=payload.get("here_before"),
            whiteout=payload.get("whiteout"),
            on_warp=bool(payload.get("on_warp")),
            warp=payload.get("warp") or {},
            faces=payload.get("faces"),
            screen_text=payload.get("screen_text") or "",
            enemy=payload.get("enemy"),
            you=payload.get("you"),
            battle_moves=list(payload.get("your_moves") or []),
            incoming=payload.get("incoming"),
            no_damage=payload.get("no_damage"),
            locked_in=payload.get("locked_in"),
            catch=payload.get("catch"),
            shop=payload.get("shop"),
            heal=payload.get("heal"),
            menu=payload.get("menu"),
            highlighted=payload.get("highlighted"),
            walked=payload.get("walked"),
            arrived=payload.get("arrived"),
            stopped_because=payload.get("stopped_because"),
            raw=payload,
        )

    def __str__(self) -> str:
        if self.in_battle:
            # A battle payload carries the position now: the coordinates are the
            # tile the player is standing on and will still be standing on when
            # the fight ends. It did not, and one session — unable to find
            # `.position`, since `__dict__` does not list a property and
            # `inspect.getsource` raises on a dataclass `__repr__` — fell back to
            # a regex over this string. The regex returned None the first time a
            # Zubat appeared and its 120-step search stopped at step 6 printing
            # "no pos in: battle vs Zubat...". So this line says where, every
            # time, and says "position unread" rather than printing a None when
            # the server could not read one.
            head = (
                f"battle on {self.map} {self._where} vs {self.enemy or '?'} "
                f"hp {self.hp}/{self.max_hp}"
            )
            # Both of these are the reason a `fight` call is about to be refused.
            # A script that only reads this line has to see them here or it
            # retries the same refusal.
            for note in (self.locked_in, self.no_damage):
                if note:
                    head += f"\n  {note}"
            # The other thing this turn could be spent on. Printed here for the
            # same reason `no_damage` is: a script that only reads this line has
            # to see it, or the ball in the bag stays in the bag.
            if self.catch:
                head += f"\n  catch: {self.catch}"
            return head
        where = f"{self.map} {self._where}" + (f" facing {self.facing}" if self.facing else "")
        moved = "" if self.moved is None else f" moved {self.moved}"
        blocked = "" if self.blocked_after is None else f" blocked after {self.blocked_after}"
        warp = " on a warp" if self.on_warp else ""
        # Both of these go on their own line and last, so a script reading only
        # the first line still gets a well-formed position, and one printing the
        # whole thing cannot miss the reason the map changed under it.
        shop = f"\n  for sale {self.shop}" if self.shop else ""
        heal = f"\n  {self.heal}" if self.heal else ""
        whiteout = f"\n  {self.whiteout}" if self.whiteout else ""
        return where + moved + blocked + warp + shop + heal + whiteout


@dataclass
class Walk:
    """What a chunked long walk actually did, batch by batch.

    A walk stops the moment the board changes under it — a battle, a dialog, a
    new map, a wall — because the rest of a plan written for the old board is
    not worth spending. ``stopped_because`` says which of those happened and
    ``remaining`` is what was never sent.
    """

    plan: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    batches: list[Result] = field(default_factory=list)
    stopped_because: str = "plan finished"
    moved: int = 0

    @property
    def done(self) -> bool:
        return not self.remaining

    @property
    def last(self) -> Optional[Result]:
        return self.batches[-1] if self.batches else None

    @property
    def map(self) -> Optional[str]:
        return self.last.map if self.last else None

    @property
    def position(self) -> Optional[Coord]:
        return self.last.position if self.last else None

    @property
    def in_battle(self) -> bool:
        return bool(self.last and self.last.in_battle)

    @property
    def on_warp(self) -> bool:
        return bool(self.last and self.last.on_warp)

    @property
    def hp(self) -> Optional[int]:
        return self.last.hp if self.last else None

    def __str__(self) -> str:
        return (
            f"{len(self.sent)}/{len(self.plan)} actions in {len(self.batches)} batches, "
            f"moved {self.moved}, stopped: {self.stopped_because}"
            + (f" at {self.last}" if self.last else "")
        )


@dataclass
class SimResult:
    """A plan run against live collision on paper. Nothing was pressed."""

    plan: list[str] = field(default_factory=list)
    end: Optional[Coord] = None
    facing: Optional[str] = None
    steps: int = 0
    blocked_at: Optional[int] = None
    blocked_by: Optional[str] = None
    warp_at: Optional[int] = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the whole plan walks without hitting anything."""

        return self.blocked_at is None

    @property
    def blocked_action(self) -> Optional[str]:
        if self.blocked_at is None or self.blocked_at >= len(self.plan):
            return None
        return self.plan[self.blocked_at]

    @property
    def clear_prefix(self) -> list[str]:
        """The part of the plan that does walk. Sendable as-is when ``ok`` is False."""

        return list(self.plan if self.blocked_at is None else self.plan[: self.blocked_at])

    @classmethod
    def from_payload(cls, plan: Sequence[str], payload: dict) -> "SimResult":
        return cls(
            plan=list(plan),
            end=_coord(payload.get("end")),
            facing=payload.get("facing"),
            steps=payload.get("steps") or 0,
            blocked_at=payload.get("blocked_at"),
            blocked_by=payload.get("blocked_by"),
            warp_at=payload.get("warp_at"),
            raw=payload,
        )

    def __str__(self) -> str:
        if self.ok:
            return f"clear: {self.steps} steps, ends at {self.end} facing {self.facing}"
        return (
            f"blocked at step {self.blocked_at} ({self.blocked_action}) by "
            f"{self.blocked_by}, stops at {self.end}"
        )


@dataclass
class Frontier:
    """Reachable ground on this map nobody has stood on, nearest first.

    Iterates and slices as the tile list, because that is what a caller wants:
    ``for tile in poke.frontier()[:3]``.
    """

    map: Optional[str] = None
    origin: Optional[Coord] = None
    tiles: list[Coord] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[Coord]:
        return iter(self.tiles)

    def __getitem__(self, index):
        return self.tiles[index]

    def __bool__(self) -> bool:
        return bool(self.tiles)

    @classmethod
    def from_payload(cls, payload: dict) -> "Frontier":
        tiles = [found for found in (_coord(tile) for tile in payload.get("tiles") or []) if found]
        return cls(
            map=payload.get("map"),
            origin=_coord(payload.get("from")),
            tiles=tiles,
            raw=payload,
        )

    def __str__(self) -> str:
        return f"{self.map} from {self.origin}: {len(self.tiles)} unseen tiles"


@dataclass
class Hop:
    """One leg of a route: a warp to step on, or a map edge to walk off."""

    from_map: Optional[str] = None
    to_map: Optional[str] = None
    kind: Optional[str] = None
    at: Optional[Coord] = None
    edge: Optional[str] = None

    def __str__(self) -> str:
        where = f" at {self.at}" if self.at else ""
        edge = f" ({self.edge})" if self.edge else ""
        return f"{self.kind}{edge} -> {self.to_map}{where}"


@dataclass
class Route:
    """Which maps lie between here and somewhere else. Hops, never buttons."""

    from_map: Optional[str] = None
    to_map: Optional[str] = None
    hops: Optional[list[Hop]] = None
    reason: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.hops is not None

    @property
    def distance(self) -> Optional[int]:
        return None if self.hops is None else len(self.hops)

    @property
    def next_hop(self) -> Optional[Hop]:
        return self.hops[0] if self.hops else None

    @classmethod
    def from_payload(cls, payload: dict) -> "Route":
        hops = payload.get("hops")
        return cls(
            from_map=payload.get("from"),
            to_map=payload.get("to"),
            hops=(
                None
                if hops is None
                else [
                    Hop(
                        from_map=hop.get("from"),
                        to_map=hop.get("to"),
                        kind=hop.get("kind"),
                        at=_coord(hop.get("at")),
                        edge=hop.get("edge"),
                    )
                    for hop in hops
                ]
            ),
            reason=payload.get("reason"),
            raw=payload,
        )

    def __str__(self) -> str:
        if self.hops is None:
            return self.reason or f"no route from {self.from_map} to {self.to_map}"
        if not self.hops:
            return f"already on {self.to_map}"
        legs = "\n".join(f"  {hop}" for hop in self.hops)
        return f"{self.from_map} to {self.to_map}, {len(self.hops)} hops:\n{legs}"


@dataclass
class MoveDamage:
    """One of your moves, costed against what you are actually fighting."""

    move: str = ""
    damage: tuple[int, int] = (0, 0)
    effectiveness: Optional[float] = None
    turns_to_ko: Optional[int] = None
    pp: Optional[int] = None
    raw: dict = field(default_factory=dict)

    @property
    def low(self) -> int:
        return self.damage[0]

    @property
    def high(self) -> int:
        return self.damage[1]

    @property
    def usable(self) -> bool:
        """Has PP left. A move at 0 PP is a number on a table, not a turn."""
        return self.pp != 0

    def __str__(self) -> str:
        effect = "" if self.effectiveness in (1, 1.0, None) else f" x{self.effectiveness:g}"
        pp = "" if self.pp is None else f" {self.pp}PP"
        if self.pp == 0:
            return f"{self.move}{pp} {self.low}-{self.high} out of PP{effect}"
        ko = f" KO in {self.turns_to_ko}" if self.turns_to_ko else " cannot KO"
        return f"{self.move}{pp} {self.low}-{self.high}{ko}{effect}"


@dataclass
class Calc:
    """The damage table for the battle on screen."""

    enemy: Optional[Mon] = None
    moves: list[MoveDamage] = field(default_factory=list)
    threat: Optional[int] = None
    threat_move: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def best(self) -> Optional[MoveDamage]:
        """Fastest kill, or hardest hit when nothing kills. The obvious pick.

        Only among moves with PP. It used to rank the whole table, so it would
        name a dry Ember as the obvious pick and `poke fight ember` would then
        be refused -- which happened 12 times in one run, and 54 of 106
        auto-saved battle entries had a dry damaging move for it to pick.
        """

        usable = [move for move in self.moves if move.usable]
        if not usable:
            return None
        return min(
            usable,
            key=lambda move: (move.turns_to_ko if move.turns_to_ko else 99, -move.high),
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "Calc":
        enemy = payload.get("enemy") or {}
        return cls(
            enemy=Mon.from_payload(enemy) if enemy else None,
            moves=[
                MoveDamage(
                    move=entry.get("move") or "",
                    damage=tuple(entry.get("damage") or (0, 0)),  # type: ignore[arg-type]
                    effectiveness=entry.get("effectiveness"),
                    turns_to_ko=entry.get("turns_to_ko"),
                    pp=entry.get("pp"),
                    raw=entry,
                )
                for entry in payload.get("moves") or []
            ],
            threat=payload.get("threat"),
            threat_move=payload.get("threat_move"),
            raw=payload,
        )

    def __str__(self) -> str:
        head = f"vs {self.enemy}" if self.enemy else "vs ?"
        lines = "\n".join(f"  {move}" for move in self.moves)
        threat = ""
        if self.threat is not None:
            named = f" ({self.threat_move})" if self.threat_move else ""
            threat = f"\n  worst incoming: {self.threat}{named}"
        return f"{head}\n{lines}{threat}"


@dataclass
class Progress:
    """How far through the game the run is, and what it has cost."""

    count: int = 0
    total: int = 0
    furthest: Optional[str] = None
    furthest_label: Optional[str] = None
    latest: list[str] = field(default_factory=list)
    presses: int = 0
    #: Milestones whose preconditions the game already satisfies, ladder order.
    frontier: list[dict] = field(default_factory=list)
    #: Rungs the run reached and the game no longer holds, a reload having landed
    #: on a branch without them. Empty when nothing was lost; also empty when the
    #: server could not read RAM, which is why the count above is the one to
    #: trust for "how far along am I".
    lost: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict) -> "Progress":
        return cls(
            count=payload.get("count") or 0,
            total=payload.get("total") or 0,
            furthest=payload.get("furthest"),
            furthest_label=payload.get("furthest_label"),
            latest=list(payload.get("latest") or []),
            presses=payload.get("presses") or 0,
            frontier=list(payload.get("frontier") or []),
            lost=list(payload.get("lost") or []),
            raw=payload,
        )

    def __str__(self) -> str:
        head = f"{self.count}/{self.total} milestones, {self.presses} presses"
        if self.furthest_label:
            head += f"\nfurthest: {self.furthest_label}"
        for entry in self.frontier:
            head += f"\nopen: {entry.get('label')}"
        if self.lost:
            names = ", ".join(
                str(item.get("label") or item.get("milestone_id")) for item in self.lost
            )
            head += f"\nreached earlier in this run, not held now: {names}"
        return head


@dataclass
class MapView:
    """Shape and coverage of a whole map, plus the PNG of it on disk."""

    map: Optional[str] = None
    map_id: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seen: Optional[int] = None
    walked: Optional[int] = None
    percent: Optional[float] = None
    player: Optional[Coord] = None
    warps: list[Coord] = field(default_factory=list)
    unexplored_nearest: Optional[Coord] = None
    #: Who is in the room and where — ``[{"service": "heal", "at": [3, 1]}]``.
    #: Empty on the 210 maps with nobody worth naming on them.
    services: list[dict] = field(default_factory=list)
    image_path: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict) -> "MapView":
        coverage = payload.get("coverage") or {}
        warps = [found for found in (_coord(warp) for warp in payload.get("warps") or []) if found]
        return cls(
            map=payload.get("map_name"),
            map_id=payload.get("map_id"),
            width=payload.get("width"),
            height=payload.get("height"),
            seen=coverage.get("seen"),
            walked=coverage.get("walked"),
            percent=coverage.get("percent"),
            player=_coord(payload.get("player")),
            warps=warps,
            unexplored_nearest=_coord(payload.get("unexplored_nearest")),
            services=list(payload.get("services") or []),
            image_path=payload.get("image_path"),
            raw=payload,
        )

    def __str__(self) -> str:
        return (
            f"{self.map} (map {self.map_id}) {self.width}x{self.height}, "
            f"seen {self.seen}/{self.width * self.height if self.width and self.height else '?'} "
            f"({self.percent}%), you at {self.player}"
        )


@dataclass
class GuideHit:
    ref: str = ""
    title: str = ""
    summary: str = ""

    def __str__(self) -> str:
        return f"{self.ref}  {self.summary}"


@dataclass
class GuideSection:
    """One walkthrough section. Prints as its body, so ``print(...)`` is the read."""

    ref: str = ""
    title: str = ""
    body: str = ""
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.body


@dataclass
class Trainer:
    trainer_class: Optional[str] = None
    at: Optional[Coord] = None
    team: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.trainer_class} at {self.at}: {', '.join(self.team)}"


@dataclass
class EncounterSlot:
    species: str = ""
    levels: tuple[int, int] = (0, 0)
    chance: float = 0.0

    def __str__(self) -> str:
        low, high = self.levels
        span = f"L{low}" if low == high else f"L{low}-{high}"
        return f"{self.species} {span} {self.chance:.0%}"


@dataclass
class EncounterTable:
    rate: Optional[int] = None
    levels: Optional[tuple[int, int]] = None
    species: list[EncounterSlot] = field(default_factory=list)

    def __iter__(self) -> Iterator[EncounterSlot]:
        return iter(self.species)

    def __str__(self) -> str:
        return f"rate {self.rate}/256: " + ", ".join(str(slot) for slot in self.species)


@dataclass
class Encounters:
    map: Optional[str] = None
    grass: Optional[EncounterTable] = None
    water: Optional[EncounterTable] = None
    raw: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.grass or self.water)

    def __str__(self) -> str:
        lines = [self.map or "?"]
        if self.grass:
            lines.append(f"  grass: {self.grass}")
        if self.water:
            lines.append(f"  water: {self.water}")
        return "\n".join(lines) if len(lines) > 1 else f"{self.map}: nothing encounterable"


@dataclass
class Species:
    """A species as the numbers that decide a fight."""

    name: str = ""
    dex: Optional[int] = None
    types: list[str] = field(default_factory=list)
    base: dict = field(default_factory=dict)
    catch_rate: Optional[int] = None
    base_exp: Optional[int] = None
    evolves: list[str] = field(default_factory=list)
    #: ``[level, move]`` pairs, in the order the species learns them.
    learnset: list[list] = field(default_factory=list)
    tm_hm: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def learns_by(self, level: int) -> list[str]:
        """Moves this species knows by *level*, latest four being what it has."""

        return [move for at, move in self.learnset if at <= level]

    def __str__(self) -> str:
        stats = " ".join(f"{key} {value}" for key, value in (self.base or {}).items())
        evolves = f", evolves into {'; '.join(self.evolves)}" if self.evolves else ""
        return f"#{self.dex} {self.name} {'/'.join(self.types)}: {stats}{evolves}"


@dataclass
class Move:
    """A move as the numbers a damage calculation needs."""

    name: str = ""
    type: Optional[str] = None
    power: Optional[int] = None
    accuracy: Optional[int] = None
    pp: Optional[int] = None
    damage_class: Optional[str] = None
    effect: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.type}, {self.damage_class}) power {self.power} "
            f"acc {self.accuracy} pp {self.pp}"
        )


@dataclass
class GroundItem:
    item: Optional[str] = None
    at: Optional[Coord] = None
    hidden: bool = False

    def __str__(self) -> str:
        return f"{self.item} at {self.at}" + (" (hidden)" if self.hidden else "")


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class _Guide:
    """``poke.guide`` — the walkthrough library. Nothing is ever pushed at you."""

    def __init__(self, client: "Client") -> None:
        self._client = client

    def outline(self) -> str:
        """Every section there is, one line each."""

        return self._client.get("/guide").get("outline", "")

    def search(self, query: str) -> list[GuideHit]:
        payload = self._client.get("/guide", q=query)
        return [
            GuideHit(
                ref=hit.get("ref", ""), title=hit.get("title", ""), summary=hit.get("summary", "")
            )
            for hit in payload.get("results") or []
        ]

    def read(self, ref: str) -> GuideSection:
        """One section's body, addressed as ``guide/slug``. Raises 404 if unknown."""

        payload = self._client.get("/guide", ref=ref)
        return GuideSection(
            ref=f"{payload.get('guide')}/{payload.get('slug')}",
            title=payload.get("title", ""),
            body=payload.get("body", ""),
            raw=payload,
        )


class _Game:
    """``poke.game`` — the static database: 223 maps, 334 trainers, 151 species.

    Read-only, emulator-free and unrated: asking costs nothing but a round trip,
    so a script can look up what is in Mt. Moon before deciding to go in.
    """

    def __init__(self, client: "Client") -> None:
        self._client = client

    def _fetch(self, topic: str, **params) -> dict:
        return self._client.get(f"/gamedata/{topic}", **params)

    def trainers(self, map_name: str, limit: Optional[int] = None) -> list[Trainer]:
        payload = self._fetch("trainers", map=map_name, limit=limit)
        return [
            Trainer(
                trainer_class=entry.get("class"),
                at=_coord(entry.get("at")),
                team=list(entry.get("team") or []),
            )
            for entry in payload.get("trainers") or []
        ]

    def encounters(self, map_name: str) -> Encounters:
        payload = self._fetch("encounters", map=map_name)
        return Encounters(
            map=payload.get("map"),
            grass=_encounter_table(payload.get("grass")),
            water=_encounter_table(payload.get("water")),
            raw=payload,
        )

    def species(self, name: str, full: bool = False) -> Species:
        payload = self._fetch("species", name=name, full=full or None)
        return Species(
            name=payload.get("name", ""),
            dex=payload.get("dex"),
            types=list(payload.get("types") or []),
            base=payload.get("base") or {},
            catch_rate=payload.get("catch_rate"),
            base_exp=payload.get("base_exp"),
            evolves=list(payload.get("evolves") or []),
            learnset=[list(pair) for pair in payload.get("learnset") or []],
            tm_hm=list(payload.get("tm_hm") or []),
            raw=payload,
        )

    def move(self, name: str) -> Move:
        payload = self._fetch("move", name=name)
        return Move(
            name=payload.get("name", ""),
            type=payload.get("type"),
            power=payload.get("power"),
            accuracy=payload.get("accuracy"),
            pp=payload.get("pp"),
            damage_class=payload.get("damage_class"),
            effect=payload.get("effect"),
            raw=payload,
        )

    def items(self, map_name: str, limit: Optional[int] = None) -> list[GroundItem]:
        payload = self._fetch("items", map=map_name, limit=limit)
        return [
            GroundItem(
                item=entry.get("item"),
                at=_coord(entry.get("at")),
                hidden=bool(entry.get("hidden")),
            )
            for entry in payload.get("items") or []
        ]

    def shops(self, map_name: str) -> dict[str, Optional[int]]:
        """What a mart sells, priced. Empty when the map has no mart — not an error.

        ``{"Poke Ball": 200, "Potion": 300, ...}``, so a shopping list three
        towns ahead is one subtraction rather than a guess. The live payload
        prints the same prices while you are standing in the shop; this is for
        deciding to go.
        """

        answer = self._fetch("shops", map=map_name)
        prices = answer.get("prices") or {}
        return {item: prices.get(item) for item in answer.get("items") or []}

    def types(self, move_type: Optional[str] = None) -> dict:
        """Type names, or what one type beats and bounces off."""

        return self._fetch("types", name=move_type)

    def effectiveness(self, move_type: str, defender_types: Sequence[str]) -> float:
        """The multiplier, e.g. Water on Rock/Ground is 4.0 and Normal on Ghost 0.0."""

        payload = self._fetch("types", name=move_type, against=",".join(defender_types))
        return float(payload.get("multiplier", 1.0))


def _encounter_table(payload: Optional[dict]) -> Optional[EncounterTable]:
    if not payload:
        return None
    levels = _coord(payload.get("levels"))
    return EncounterTable(
        rate=payload.get("rate"),
        levels=levels,
        species=[
            EncounterSlot(
                species=slot.get("species", ""),
                levels=_coord(slot.get("levels")) or (0, 0),
                chance=float(slot.get("chance") or 0.0),
            )
            for slot in payload.get("species") or []
        ],
    )


class Client:
    """A connection to one game server. ``poke.state()`` uses the default one."""

    def __init__(
        self,
        port: Optional[Union[int, str]] = None,
        url: Optional[str] = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        if url:
            self.port = None
            self.url = str(url).rstrip("/")
        else:
            self.port = str(port or os.environ.get("PORT") or DEFAULT_PORT)
            self.url = f"http://localhost:{self.port}"
        self.timeout = float(timeout)
        #: When each acting call went out, for pacing under the server's cap.
        self._sent: deque[float] = deque(maxlen=RATE_MAX_BATCHES * 2)
        self.guide = _Guide(self)
        self.game = _Game(self)

    # -- HTTP ------------------------------------------------------------

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> bytes:
        query = {
            key: ("1" if value is True else str(value))
            for key, value in (params or {}).items()
            if value is not None and value is not False
        }
        if query:
            path = f"{path}?{urllib.parse.urlencode(query)}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise ServerError(error.code, _detail_from(error)) from None
        except urllib.error.URLError as error:
            raise Unreachable(
                f"nothing answering at {self.url}: {getattr(error, 'reason', error)}"
            ) from None
        except OSError as error:
            raise Unreachable(f"nothing answering at {self.url}: {error}") from None

    def _json(self, path: str, **kwargs) -> Any:
        raw = self._request(path, **kwargs)
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except (json.JSONDecodeError, ValueError):
            raise ServerError(0, f"server sent something that is not JSON: {raw[:200]!r}") from None

    def get(self, path: str, **params) -> Any:
        """Raw GET, for an endpoint this client has not grown a method for."""

        return self._json(path, params=params)

    def post(self, path: str, payload: Optional[dict] = None) -> Any:
        """Raw POST. Does *not* pace: use it only for endpoints that press nothing."""

        return self._json(path, method="POST", payload=payload)

    # -- Rate ------------------------------------------------------------

    def _pace(self) -> None:
        """Wait, if waiting is what it takes to stay under the server's cap.

        The server allows 60 action batches a minute and answers a 429 with a
        lecture about loops. Sleeping here instead means a legitimate long walk
        finishes slowly rather than dying halfway across a map, and a runaway
        loop is slowed to a crawl where it is easy to notice.
        """

        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0] > RATE_WINDOW_SECONDS:
                self._sent.popleft()
            if len(self._sent) < RATE_MAX_BATCHES:
                self._sent.append(now)
                return
            wait = RATE_WINDOW_SECONDS - (now - self._sent[0]) + 0.05
            if wait > MAX_PACE_WAIT_SECONDS:
                raise RateLimited(
                    f"waiting {wait:.0f}s for the rate window would be longer than the "
                    f"{RATE_WINDOW_SECONDS:.0f}s window itself. Something else is driving "
                    "this server."
                )
            time.sleep(max(0.0, wait))

    def _act_json(self, path: str, payload: Optional[dict] = None) -> Any:
        self._pace()
        return self._json(path, method="POST", payload=payload)

    # -- Acting ----------------------------------------------------------

    def act(self, *tokens: Tokens) -> Result:
        """Send one batch of actions. ``act("up", "up", "a")``, ``act("right:6")``.

        Refuses a batch the server would refuse, before spending the round trip.
        For anything longer than one batch, use :meth:`walk`.
        """

        actions = one_batch(*tokens)
        return Result.from_payload(self._act_json("/action", {"actions": actions}))

    def walk(
        self,
        *tokens: Tokens,
        stop_on_battle: bool = True,
        stop_on_dialog: bool = True,
        stop_on_map_change: bool = True,
        stop_if_stuck: bool = True,
        max_actions: int = WALK_MAX_ACTIONS,
        start_map: Optional[str] = None,
    ) -> Walk:
        """Send a long path in legal chunks, and stop when the board changes.

        This is the one to reach for when a plan is longer than forty actions.
        It splits the plan across batches that respect both server caps, paces
        itself under the rate limit, and gives up the moment the plan stops
        being about the board it was written for.

        ``walk("up:60", "right:20")`` is one call, four batches, and one report.
        """

        plan = expand(*tokens, max_repeat=max_actions)
        if len(plan) > max_actions:
            raise ActionError(
                f"that walk is {len(plan)} actions; the cap is {max_actions}. Walk part "
                "of it, look at where you ended up, then decide the rest."
            )
        report = Walk(plan=list(plan), remaining=list(plan))
        # Which map the plan was written for. Read once, before anything is
        # pressed: taking it from the first batch's answer would miss the case
        # that matters most, a warp crossed inside that very first batch with
        # the rest of the plan still queued for the map you just left.
        started_on = start_map
        if started_on is None and stop_on_map_change:
            started_on = self.state().map
        for batch in chunks(plan):
            result = Result.from_payload(self._act_json("/action", {"actions": batch}))
            report.batches.append(result)
            report.sent.extend(batch)
            report.remaining = report.remaining[len(batch) :]
            report.moved += result.moved or 0
            if started_on is None:
                started_on = result.map

            reason = None
            if stop_on_battle and result.in_battle:
                reason = "a battle started"
            elif stop_on_map_change and result.map and result.map != started_on:
                reason = f"the map changed to {result.map}"
            elif stop_on_dialog and result.dialog:
                reason = "a dialog opened"
            elif stop_if_stuck and result.blocked_after is not None:
                reason = f"blocked after {result.blocked_after} of {len(batch)} actions"
            if reason:
                report.stopped_because = reason
                return report
        return report

    def fight(self, move: str) -> Result:
        """Attack by name. The server does the menu work; a prefix is enough."""

        return Result.from_payload(self._act_json("/battle/fight", {"move": move}))

    def flee(self) -> Result:
        """Run from the current battle."""

        return Result.from_payload(self._act_json("/battle/run"))

    def catch(self, ball: Optional[str] = None) -> Result:
        """Throw a ball at the wild Pokemon. No ball named throws the weakest carried."""

        return Result.from_payload(self._act_json("/battle/catch", {"ball": ball} if ball else {}))

    def buy(self, item: str, count: int = 1) -> Result:
        """Buy from the mart on this map, walking to the counter first if need be."""

        return Result.from_payload(self._act_json("/mart/buy", {"item": item, "count": int(count)}))

    def heal(self) -> Result:
        """Heal the party at the nurse on this map, walking to her counter first."""

        return Result.from_payload(self._act_json("/pokecenter/heal", {}))

    def goto(self, target: Union[str, Coord], y: Optional[int] = None) -> Result:
        """Walk toward a map or a tile, the server re-planning on each map.

        ``goto("Cerulean City")``, ``goto((12, 4))`` or ``goto(12, 4)``. One call
        is one batch's worth of frames, so a long journey takes several — check
        ``.arrived`` and call again.
        """

        if y is not None:
            payload = {"x": int(target), "y": int(y)}  # type: ignore[arg-type]
        elif isinstance(target, str):
            payload = {"target": target}
        else:
            coord = _coord(target)
            if coord is None:
                raise ActionError(f"{target!r} is not a map name or an (x, y) tile")
            payload = {"x": coord[0], "y": coord[1]}
        return Result.from_payload(self._act_json("/goto", payload))

    # -- Looking ---------------------------------------------------------

    def state(self) -> State:
        """Party, bag, badges and position."""

        return State.from_payload(self._json("/state"))

    def sim(self, *tokens: Tokens) -> SimResult:
        """Try a plan against live collision without spending it. Presses nothing."""

        plan = expand(*tokens, max_repeat=WALK_MAX_ACTIONS)
        return SimResult.from_payload(
            plan, self._json("/sim", method="POST", payload={"actions": plan})
        )

    def frontier(self) -> Frontier:
        """Reachable tiles on this map nobody has stood on, nearest first."""

        return Frontier.from_payload(self._json("/frontier"))

    def route(self, to: str) -> Route:
        """Which maps lie between here and *to*."""

        return Route.from_payload(self._json("/route", params={"to": to}))

    def calc(self) -> Calc:
        """The damage table for the battle on screen. Raises if there is no battle."""

        return Calc.from_payload(self._json("/calc"))

    def map(self, map_id: Optional[int] = None) -> MapView:
        """Coverage and shape of a map, plus the path of a PNG to look at."""

        return MapView.from_payload(self._json("/map", params={"map_id": map_id}))

    def progress(self) -> Progress:
        """Milestones reached and buttons spent."""

        return Progress.from_payload(self._json("/progress"))

    def health(self) -> dict:
        return self._json("/health")

    # -- Saves and frames -------------------------------------------------

    def save(self, name: str) -> Result:
        return Result.from_payload(self._json("/save", method="POST", payload={"name": name}))

    def load(self, name: str, force: bool = False) -> Result:
        """Load a save. Refused if it holds fewer milestones than the live game.

        ``force=True`` overrides that, for a branch that really was lost.
        """

        return Result.from_payload(
            self._json("/load", method="POST", payload={"name": name, "force": bool(force)})
        )

    def saves(self) -> list[str]:
        """Named saves, newest first.

        ``named=true`` for the same reason ``poke saves`` uses it: the plain
        list is the newest forty of everything, and with the 465 saves this run
        has written that is forty ``auto__`` checkpoints and none of the 165
        names anyone would load. ``load()`` still takes an autosave's name.
        """

        payload = self._json("/saves", params={"named": True})
        return [entry.get("name") for entry in payload.get("saves") or []]

    def frames(self) -> dict[str, Path]:
        """Paths of the two workspace frames the runtime keeps rewritten."""

        here = Path(os.environ.get("POKE_WORKSPACE") or Path.cwd())
        if not any((here / name).exists() for name in FRAME_FILES.values()):
            try:
                reported = (self._json("/") or {}).get("agent_workspace_dir")
            except PokeError:
                reported = None
            if reported:
                here = Path(reported)
        return {label: here / name for label, name in FRAME_FILES.items()}

    def screenshot(self, path: Union[str, Path] = "fresh_frame.png") -> Path:
        """Fetch the current frame as a PNG and write it where you asked."""

        target = Path(path).expanduser()
        target.write_bytes(self._request("/screenshot"))
        return target.resolve()

    def limits(self) -> dict:
        """The caps a script should plan inside, so it never has to guess them."""

        return {
            "max_actions_per_batch": MAX_ACTIONS_PER_BATCH,
            "max_frames_per_batch": MAX_FRAMES_PER_BATCH,
            "max_frames_per_action": MAX_FRAMES_PER_ACTION,
            "walk_max_actions": WALK_MAX_ACTIONS,
            "batches_per_minute": RATE_MAX_BATCHES,
            "server_batches_per_minute": 60,
        }

    def __repr__(self) -> str:
        return f"Client({self.url!r})"


def _detail_from(error: urllib.error.HTTPError) -> str:
    """The server writes its 400s and 429s to be read. Keep the words."""

    try:
        body = error.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreadable body must not mask the status
        body = ""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body.strip() or str(error.reason) or f"HTTP {error.code}"
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail)
    return body.strip() or f"HTTP {error.code}"


# ---------------------------------------------------------------------------
# The default client
#
# A script should be able to `import poke` and play, without a setup line. The
# port comes from $PORT, which the harness sets, so the default is right in the
# workspace and `connect()` is there for everywhere else.
# ---------------------------------------------------------------------------

_default: Optional[Client] = None


def client() -> Client:
    """The client the module-level functions use, made on first use."""

    global _default
    if _default is None:
        _default = Client()
    return _default


def connect(port: Optional[Union[int, str]] = None, url: Optional[str] = None) -> Client:
    """Point the module-level functions at a different server."""

    global _default, guide, game
    _default = Client(port=port, url=url)
    guide = _default.guide
    game = _default.game
    return _default


#: ``poke.guide.read(...)`` and ``poke.game.species(...)``. Rebound by connect().
guide = client().guide
game = client().game


def act(*tokens: Tokens) -> Result:
    return client().act(*tokens)


def walk(*tokens: Tokens, **kwargs) -> Walk:
    return client().walk(*tokens, **kwargs)


def fight(move: str) -> Result:
    return client().fight(move)


def flee() -> Result:
    return client().flee()


def catch(ball: Optional[str] = None) -> Result:
    return client().catch(ball)


def buy(item: str, count: int = 1) -> Result:
    return client().buy(item, count)


def heal() -> Result:
    return client().heal()


def goto(target: Union[str, Coord], y: Optional[int] = None) -> Result:
    return client().goto(target, y)


def state() -> State:
    return client().state()


def sim(*tokens: Tokens) -> SimResult:
    return client().sim(*tokens)


def frontier() -> Frontier:
    return client().frontier()


def route(to: str) -> Route:
    return client().route(to)


def calc() -> Calc:
    return client().calc()


# Shadows the builtin for anything that does `from poke import *`, which nothing
# should. `poke.map()` is what a script writes, and the game calls it a map.
def map(map_id: Optional[int] = None) -> MapView:
    return client().map(map_id)


def progress() -> Progress:
    return client().progress()


def health() -> dict:
    return client().health()


def save(name: str) -> Result:
    return client().save(name)


def load(name: str) -> Result:
    return client().load(name)


def saves() -> list[str]:
    return client().saves()


def frames() -> dict[str, Path]:
    return client().frames()


def screenshot(path: Union[str, Path] = "fresh_frame.png") -> Path:
    return client().screenshot(path)


def limits() -> dict:
    return client().limits()


__all__ = [
    "ActionError",
    "Calc",
    "Client",
    "Encounters",
    "Frontier",
    "GroundItem",
    "GuideHit",
    "GuideSection",
    "MapView",
    "Mon",
    "Move",
    "MoveDamage",
    "PokeError",
    "Progress",
    "RateLimited",
    "Result",
    "Route",
    "ServerError",
    "SimResult",
    "Species",
    "State",
    "Trainer",
    "Unreachable",
    "Walk",
    "act",
    "buy",
    "heal",
    "calc",
    "catch",
    "chunks",
    "client",
    "connect",
    "expand",
    "fight",
    "flee",
    "frames",
    "frontier",
    "game",
    "goto",
    "guide",
    "health",
    "limits",
    "load",
    "map",
    "one_batch",
    "progress",
    "route",
    "save",
    "saves",
    "screenshot",
    "sim",
    "state",
    "walk",
]
