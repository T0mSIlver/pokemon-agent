"""Refuse the command that has already been proved to do nothing.

The largest single waste this project has measured is the agent re-sending one
byte-identical command into a frame that cannot answer it. Four episodes in the
34-hour run of 2026-08-25:

===========================  =======  =====  ==========  =============================
Where                        Presses  Calls  Wall clock  The command, repeated
===========================  =======  =====  ==========  =============================
Cerulean City (27,26)         12,317    302    14.6 min  ``act wait_60 a:38``
Route 6 (1,15)                 2,405    331    15.3 min  ``poke run`` -> "could not get away"
Vermilion City (33,4)            368    362    11.8 min  ``act up:1``, ``moved 0`` every time
Vermilion Pokecenter (4,4)       384    181     5.2 min  A-spam at the nurse counter
===========================  =======  =====  ==========  =============================

15,090 presses, 17% of the run, and three of the 49 sessions died inside a loop
like this on the token budget rather than leaving it.

Why the receipt is not enough
-----------------------------
A receipt records map, tile, presses, moved and HP. All 302 Cerulean receipts
are byte-identical -- and so are the receipts of a *legitimate* dialog being
advanced one page at a time, which also ends on the same tile with the same HP
and the same ``dialog`` flag. Measured over the whole run, 99.6% of runs of
consecutive identical receipts are 5 long or shorter and four are 120, 324, 330
and 362; but the receipt cannot say which is which without waiting for the
count, and by then the presses are spent.

What separates them
-------------------
Two things, and both are needed:

* **Nothing durable changed.** :func:`world_fingerprint` is the state that
  survives closing every box: map, tile, facing, money, party, bag, badges,
  whether a battle is on. Deliberately curated rather than a RAM hash --
  ``play_time`` advances every second and a frame hash of the screen differed
  on 20 of 20 batches taken at a *provably static* dialog, so either of those
  as a "did anything change" test never fires at all.
* **No words the model has not already read.** ``read_screen_text`` decodes
  ``wTileMap`` through the Gen 1 font table, so the harness can finally see the
  box it is pressing A at. A dialog that is advancing puts new words on screen
  every press; a dialog that is a fixed point does not, however many times it
  is opened. The Cerulean object is the sharp case: it answers with one of four
  random flavour lines, so "the same words twice" is wrong there and "words I
  have not read before" is right.

The limit
---------
:data:`REPEAT_LIMIT` is 16 because the longest legitimate no-progress run
measured is 9: a full party healing at a Poke Center counter, where the machine
animates one flash per Pokemon and every A press during it lands on the same
frame with the same words. One Pokemon healing is a run of 5. 16 leaves that
worst measured case most of its own length again in headroom, and still spends
at most 16 batches on a loop that spent 302.

The refusal, not a hint
-----------------------
Every advisory this project has handed the model has been ignored -- the
``here_before`` counter reached 49 on one tile and changed nothing. So this is
a 400 that names the way out, and it names ``b`` explicitly because the model
demonstrably will not reach for it: across 78,795 presses in that run it pressed
B 321 times, 0.4%, and pressed it exactly zero times in 12,317 presses at the
Cerulean object. The refusal is scoped to the one command that has been proved
inert; every other command is still accepted, so the agent is never left with
no legal move.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

#: Identical, provably inert calls tolerated before the next one is refused.
#: See the module docstring: the worst measured legitimate run is 9.
REPEAT_LIMIT = 16

#: How many distinct screens to remember per streak. The longest legitimate
#: conversation measured -- the whole Poke Center heal, greeting to goodbye --
#: shows 13, and the run's Vermilion episode was the agent walking round that
#: same conversation forever after the heal had already happened.
WORDS_REMEMBERED = 32

#: A command is keyed by exactly what was asked for, so a refusal never blocks
#: anything but the one call it was proved against.
Key = tuple[str, ...]


class RepeatedNoProgress(Exception):
    """A command that has already changed nothing :data:`REPEAT_LIMIT` times."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def world_fingerprint(state: Optional[Mapping[str, Any]]) -> tuple:
    """The part of the game that survives closing every box.

    Curated, not hashed. Two fields in the state dict change on their own with
    no input at all -- ``metadata.timestamp`` and ``player.play_time`` -- and a
    fingerprint carrying either can never compare equal, which is the silent
    failure mode for a guard like this one: it simply never fires and nobody
    notices. Everything below is a fact about the save, not about the frame.

    The transient half of the frame is left out on purpose. Whether a text box
    happens to be open, which page it is on and what the window Y register says
    are all things a fixed-point dialog changes twice a second while changing
    nothing; the words in that box are carried separately, by the caller.
    """
    state = state or {}
    player = state.get("player") or {}
    position = player.get("position") or {}
    battle = state.get("battle") or {}
    party = state.get("party") or []
    bag = state.get("bag") or []
    flags = state.get("flags") or {}
    enemy = battle.get("enemy") if isinstance(battle.get("enemy"), dict) else {}
    return (
        (state.get("map") or {}).get("map_id"),
        position.get("x"),
        position.get("y"),
        player.get("facing"),
        player.get("money"),
        tuple(
            (
                mon.get("species_id"),
                mon.get("level"),
                mon.get("hp"),
                mon.get("max_hp"),
                mon.get("status"),
                mon.get("experience"),
            )
            for mon in party
            if isinstance(mon, Mapping)
        ),
        tuple(sorted((item.get("id"), item.get("quantity")) for item in bag)),
        flags.get("badge_count"),
        flags.get("pokedex_owned"),
        bool(battle.get("in_battle")),
        (enemy or {}).get("hp"),
    )


def screen_words(state: Optional[Mapping[str, Any]]) -> str:
    """The words on screen, as the harness decoded them off ``wTileMap``.

    Empty on an overworld frame, which is correct and is what makes a blocked
    ``walk_up`` accumulate: there is nothing on screen to be new.
    """
    return str((state or {}).get("screen") or "")


@dataclass
class Streak:
    """One command, repeated, against a world that has not moved."""

    key: Key
    world: tuple
    words: set[str] = field(default_factory=set)
    #: Consecutive calls that changed nothing durable *and* showed nothing new.
    count: int = 0

    def blocked(self) -> bool:
        return self.count >= REPEAT_LIMIT


class RepeatGuard:
    """Tracks the last command and refuses it once it is proved inert.

    One streak, not a history: the moment a different command arrives, or the
    world moves, or a new screen appears, the count starts over. That is what
    keeps the guard from ever wedging the agent -- the escape it names in the
    refusal is itself a different command, so taking it clears the block.
    """

    def __init__(self, limit: int = REPEAT_LIMIT) -> None:
        self._limit = limit
        self._streak: Optional[Streak] = None

    @property
    def streak(self) -> Optional[Streak]:
        return self._streak

    def reset(self) -> None:
        """Forget everything. For a load, a whiteout, or a new run."""
        self._streak = None

    def count_for(self, key: Key) -> int:
        streak = self._streak
        return streak.count if streak is not None and streak.key == tuple(key) else 0

    def check(self, key: Key, *, describe: Optional[Callable[[Streak], str]] = None) -> None:
        """Raise :class:`RepeatedNoProgress` if *key* has been proved inert.

        Called before the emulator is touched, so a refused command costs no
        buttons at all. ``describe`` builds the message and is only called on
        the refusal path, so composing it costs nothing on the 99.99% of calls
        that pass.
        """
        streak = self._streak
        if streak is None or streak.key != tuple(key) or streak.count < self._limit:
            return
        detail = describe(streak) if describe is not None else default_refusal(streak)
        raise RepeatedNoProgress(detail)

    def record(self, key: Key, before: tuple, after: tuple, words: str = "") -> None:
        """File what one call did, and grow or reset the streak accordingly.

        A call counts toward the streak only if all three hold: it is the same
        command as last time, it left the durable world exactly as it found it,
        and the screen it ended on is one this streak has already shown. New
        words are progress -- a multi-page conversation restarts the count on
        every page -- and any durable change starts a new streak from zero.
        """
        key = tuple(key)
        streak = self._streak
        unchanged = before == after
        if streak is not None and streak.key == key and unchanged and streak.world == after:
            if words in streak.words:
                streak.count += 1
            else:
                if len(streak.words) < WORDS_REMEMBERED:
                    streak.words.add(words)
                streak.count = 1
            return
        self._streak = Streak(key=key, world=after, words={words}, count=1 if unchanged else 0)


#: The short names the agent actually types. Quoting a batch back at it as
#: thirty-eight `press_a` tokens would put 300 bytes of noise in the refusal and
#: describe a command it never sent.
_SHORT = {
    "press_a": "a",
    "press_b": "b",
    "press_start": "start",
    "press_select": "select",
    "walk_up": "up",
    "walk_down": "down",
    "walk_left": "left",
    "walk_right": "right",
    "a_until_dialog_end": "adialog",
}


def command_text(key: Key) -> str:
    """The key rendered back the way it was typed: ``wait_60 a:38``."""
    runs: list[list] = []
    for part in key:
        token = _SHORT.get(str(part), str(part))
        if not token or token == "None":
            continue
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    return " ".join(token if count == 1 else f"{token}:{count}" for token, count in runs)


def default_refusal(streak: Streak) -> str:
    """The generic wording, for a command with no diagnosis of its own."""
    return (
        f"`{command_text(streak.key)}` has run {streak.count} times from this frame and "
        "changed nothing: same map, same tile, same party, same bag, and no words on "
        "screen you have not already read. Repeating it again cannot produce a "
        "different answer. Press B to back out of whatever is open (`poke act b`), or "
        "send a different command -- every other command is still accepted, only this "
        "exact one is refused."
    )


#: What to do instead, per diagnosis. Every one of these was measured from the
#: frame it is offered on, because an escape that does not work is worse than no
#: escape at all: standing at the Cerulean object with the box up, `left` and
#: `start` both do nothing at all, five presses each, while `b:2` closes it --
#: the first B finishes printing the line and the second dismisses the box.
_ESCAPE_DIALOG = (
    "Press B until the box closes -- `poke act b:2`, because the first B only finishes "
    "printing the line and the second dismisses it -- and then walk away. A direction on "
    "its own will not do it: the d-pad does not reach the player while a box is open."
)
_ESCAPE_WALL = (
    "Pick a different direction: `poke state` lists the ones that are legal from this "
    "tile and how far each of them goes, and `poke goto <map>` will route round the "
    "obstacle rather than into it."
)
_ESCAPE_WAITING = (
    "If something on screen is still animating, `poke act wait_60` lets it finish without "
    "pressing anything. Otherwise send a different command."
)


def action_refusal(streak: Streak, *, dialog: bool, blocked_walk: bool) -> str:
    """The wording for a repeated ``/action`` batch.

    Three diagnoses with three different escapes, because the measured episodes
    had three different causes: a box toggling, a step into a wall, and buttons
    thrown at an animation. A refusal that names the wrong way out is worse than
    one that names none.
    """
    opening = (
        f"`poke act {command_text(streak.key[1:])}` has run {streak.count} times from this "
        "frame and changed nothing: same tile, same party, same bag, and no words on "
        "screen you have not already read."
    )
    if dialog:
        why = (
            " A press on a box that is waiting for input closes it and the next press "
            "opens it again, so an even number of A presses lands exactly where it "
            "started, however many you send."
        )
        escape = _ESCAPE_DIALOG
    elif blocked_walk:
        why = (
            " Every step in it was blocked. A blocked step in Gen 1 does not fail, it "
            "puts the player back on the tile it started from, so walking harder into "
            "the same wall reads identically to walking into open ground."
        )
        escape = _ESCAPE_WALL
    else:
        why = ""
        escape = _ESCAPE_WAITING
    return (
        f"{opening}{why} {escape} Any other command is accepted; this exact one is not, "
        "until something changes."
    )


def battle_refusal(streak: Streak) -> str:
    """The wording for a repeated battle command that is getting nowhere.

    Branching on the verb, because the three of them fail for three reasons and
    the escape from each is one of the other two.
    """
    verb = streak.key[0] if streak.key else ""
    opening = (
        f"`poke {command_text(streak.key)}` has run {streak.count} times and changed "
        "nothing: the same Pokemon is still out, at the same HP, and neither side has "
        "lost a turn's worth of anything."
    )
    if verb == "run":
        # Route 6 (1,15): 331 calls, 2,405 presses, 15.3 minutes of speed rolls.
        why = (
            " Fleeing is a speed roll against this Pokemon, and a roll that has failed "
            "this often is not going to start passing."
        )
        escape = "Attack it by name (`poke fight <move>`) or throw a ball (`poke catch`)."
    elif verb == "fight":
        why = (
            " A move that takes nothing off it is either not connecting or does no damage "
            "to this type at all; `poke calc` prices every move you have against it."
        )
        escape = "Attack with a different move, throw a ball (`poke catch`), or flee (`poke run`)."
    else:
        why = ""
        escape = (
            "Attack it by name (`poke fight <move>`) or flee (`poke run`) -- `poke calc` "
            "says which of your moves is worth the turn."
        )
    return f"{opening}{why} {escape} Any other battle command is accepted."


def sim_refusal(streak: Streak) -> str:
    """The wording for a plan simulated over and over.

    ``sim`` costs no buttons, which is exactly why it went unnoticed: 2,982
    calls run-wide, 531 of them identical inside one session, and six sessions
    ended inside a loop like this having spent their token budget instead of
    their presses.
    """
    return (
        f"This exact plan has been simulated {streak.count} times and the answer has not "
        "moved. `sim` never touches the game, so nothing that has happened since the "
        "first call could change it. Run the plan (`poke act ...`), simulate a "
        "different one, or read the frame (`poke state`) -- any other command is "
        "accepted."
    )


def looks_like_dialog(state: Optional[Mapping[str, Any]]) -> bool:
    """Whether a box was open on the frame the streak is stuck on."""
    dialog = (state or {}).get("dialog") or {}
    return bool((state or {}).get("dialog_active") or dialog.get("active"))


#: Walks looked at before the circuit breaker below decides they went nowhere.
CYCLE_WINDOW = 24
#: Distinct tiles those walks may end on and still count as going nowhere. Three
#: rather than one because the shape being caught is a lap, not a stuck tile.
CYCLE_TILES = 3
#: And the presses they must have cost. Twenty-four walks ending on three tiles
#: is only damning if buttons were spent doing it; the same window at 1 press a
#: call is a model reading the frame, which is cheap and often right.
CYCLE_PRESSES = 240


#: Stays on a map looked at before the wander detector decides. A "stay" is one
#: unbroken run on one map, however many calls it took.
WANDER_WINDOW = 16
#: How few different maps those stays may cover and still be a wander.
WANDER_MAPS = 4
#: And the buttons they must have cost. All three tuned against a 183,000-press
#: run rather than chosen: at these values it fires 16 times in 36,000 calls,
#: names 49,534 presses, and exactly one of the sixteen lands in a stretch that
#: earned a rung soon after. Loosening any one of them roughly triples the fires
#: and puts a third of them on productive ground.
WANDER_PRESSES = 1600


class Wandering(Exception):
    """A handful of maps walked round and round."""

    def __init__(self, detail: str, maps: Sequence[str], presses: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.maps = tuple(maps)
        self.presses = presses


class WanderGuard:
    """Notices a circuit, which :class:`CycleGuard` structurally cannot.

    `CycleGuard` resets the moment the map changes, on the reasoning that a
    different map is progress by itself. That is true of a journey and false of
    a circuit, and the difference is not "changed map" but "changed map to
    somewhere it has just been".

    The run this was built from spent its first 23.8 hours and 42,000 presses
    walking Mt Moon 1F, B1F, B2F and Route 4 in circles, and its last stall
    doing the same across Pewter, Route 2 and Viridian Forest. Not one call of
    either was a lap on a single map, so nothing in the harness could see them.

    Fires once per window and then forgets, like `CycleGuard`, so the next call
    is always allowed through. Cleared outright by a milestone: a rung is the
    evidence that whatever it was doing worked.
    """

    def __init__(
        self,
        *,
        window: int = WANDER_WINDOW,
        maps: int = WANDER_MAPS,
        presses: int = WANDER_PRESSES,
    ) -> None:
        self._window = window
        self._maps = maps
        self._presses = presses
        self._stays: list[list] = []

    def reset(self) -> None:
        """Forget the window. For a milestone, a load, or a new run."""
        self._stays.clear()

    def record(self, map_name: str, presses: int) -> None:
        """File one call's map and cost, extending the current stay or opening one."""
        if not map_name:
            return
        if not self._stays or self._stays[-1][0] != map_name:
            self._stays.append([map_name, 0])
        self._stays[-1][1] += max(0, int(presses))
        if len(self._stays) > self._window:
            self._stays.pop(0)

    def circuit(self) -> Optional[tuple[list[str], int]]:
        """The maps and the presses, if the window is a circuit. ``None`` if not."""
        if len(self._stays) < self._window:
            return None
        maps = {name for name, _ in self._stays}
        if len(maps) > self._maps:
            return None
        spent = sum(presses for _, presses in self._stays)
        if spent < self._presses:
            return None
        return sorted(maps), spent

    def check(self, describe: Callable[[list[str], int], str]) -> None:
        """Raise once if the window is a circuit, then forget it."""
        found = self.circuit()
        if found is None:
            return
        maps, spent = found
        self.reset()
        raise Wandering(describe(maps, spent), maps, spent)


class WalkingInCircles(Exception):
    """A stretch of walking that keeps arriving where it started."""

    def __init__(self, detail: str, tiles: Sequence[tuple[int, int]], presses: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.tiles = tuple(tiles)
        self.presses = presses


class CycleGuard:
    """Notices a lap, which :class:`RepeatGuard` structurally cannot.

    ``RepeatGuard`` keys on the command, and a lap is not one command. The
    measured case is Route 11 (55,0): 218 consecutive calls, **8,225 presses in
    22 minutes**, every one ending on the same tile, and the streak reset on
    every call because the model kept varying the plan — ``right:5 up:4``,
    ``right:4 up:4``, ``right:3 up:4`` — while walking the identical lap. By
    command it is 6 distinct plans; by outcome it is one, 72 times. 10.6% of
    everything that run spent after its last milestone went through that block,
    and nothing in the harness said a word.

    So this one ignores the command entirely and watches only where walks end.

    It fires **once per window and then forgets**, which is what makes it safe
    to raise rather than annotate. The agent is never left without a legal move:
    the very next call is allowed through whatever it is. The run being fixed
    here already had ``stood here N times before`` appended to 2,023 payloads
    and walked past every one of them, so an annotation is not the instrument.
    """

    def __init__(
        self,
        *,
        window: int = CYCLE_WINDOW,
        tiles: int = CYCLE_TILES,
        presses: int = CYCLE_PRESSES,
    ) -> None:
        self._window = window
        self._tiles = tiles
        self._presses = presses
        self._steps: deque[tuple[str, tuple[int, int], int]] = deque(maxlen=window)

    def reset(self) -> None:
        """Forget the window. For a load, a whiteout, or a new map."""
        self._steps.clear()

    def record(self, map_name: str, position: Optional[tuple[int, int]], presses: int) -> None:
        """File where one walk ended. Only walking calls belong here.

        A battle, a heal or a purchase ends where it started by design, and
        feeding those in would make standing still at a counter look like a lap.
        """
        if not map_name or position is None:
            self.reset()
            return
        if self._steps and self._steps[-1][0] != map_name:
            # A different map is progress by itself: whatever the lap was, it
            # ended. Nothing here is trying to catch a two-map bounce.
            self.reset()
        self._steps.append((map_name, tuple(position), max(0, int(presses))))

    def lap(self) -> Optional[tuple[list[tuple[int, int]], int]]:
        """The tiles and the presses, if the window is a lap. ``None`` if not."""
        if len(self._steps) < self._window:
            return None
        tiles = {step[1] for step in self._steps}
        if len(tiles) > self._tiles:
            return None
        spent = sum(step[2] for step in self._steps)
        if spent < self._presses:
            return None
        return sorted(tiles), spent

    def check(self, describe: Callable[[list[tuple[int, int]], int], str]) -> None:
        """Raise once if the window is a lap, then forget it."""
        found = self.lap()
        if found is None:
            return
        tiles, spent = found
        self.reset()
        raise WalkingInCircles(describe(tiles, spent), tiles, spent)


def all_walks_blocked(actions: Sequence[str], outcome: Optional[Mapping[str, Any]]) -> bool:
    """Whether the batch was walking and moved nothing."""
    if not any(str(action).strip().lower().startswith("walk_") for action in actions):
        return False
    return not (outcome or {}).get("moved")


__all__ = [
    "CYCLE_PRESSES",
    "CYCLE_TILES",
    "CYCLE_WINDOW",
    "CycleGuard",
    "WANDER_MAPS",
    "WANDER_PRESSES",
    "WANDER_WINDOW",
    "Key",
    "REPEAT_LIMIT",
    "RepeatGuard",
    "RepeatedNoProgress",
    "Streak",
    "WORDS_REMEMBERED",
    "action_refusal",
    "all_walks_blocked",
    "battle_refusal",
    "command_text",
    "default_refusal",
    "looks_like_dialog",
    "screen_words",
    "sim_refusal",
    "WalkingInCircles",
    "WanderGuard",
    "Wandering",
    "world_fingerprint",
]
