#!/usr/bin/env python3
"""Regenerate pokemon_agent/data/red_milestones.json from the pokered decomp.

Source of truth for the event bitfield at wEventFlags (0xD747):
  https://raw.githubusercontent.com/pret/pokered/master/constants/event_constants.asm

Same precedent as MAP_NAMES in pokemon_agent/memory/red.py: the upstream files are
parsed once, offline, and the result is checked in. A running emulator must not
depend on network access, and the diff has to be reviewable.

Unlike map_constants.asm, event_constants.asm is *not* a straight line count.
rgbds' const_def/const/const_skip/const_next macros move a counter:

    const_def            counter = 0
    const NAME           NAME = counter; counter += 1
    const_skip [N]       counter += N (N defaults to 1)
    const_next EXPR      counter = EXPR   <- absolute, not relative

Every section header in the file is a `const_next $XXX` that jumps the counter to
the start of that map's slice of the bitfield, so enumerating `const` lines in
order and numbering them 0..506 gives 507 wrong indices. Ask MAP_NAMES how much a
drifted table costs.

The *second* thing this script reads upstream for is which flags the game can
clear again, and that half exists because a hand-written list got it wrong. The
first version of this file carried twenty resettable event names typed out by
hand. The decomp resets thirty-one, plus two whole byte-aligned ranges, and the
eleven that were missed included the last eight rungs of the ladder: Victory
Road's 2F switch (Route23.asm clears it on every map load), all five Elite Four
rooms and the Champion (IndigoPlateauLobby.asm clears them when a failed
challenge sends you back to the lobby; HallOfFame.asm clears them again on the
way in), and the Hall of Fame flag itself (set by hall_of_fame.asm and consumed
by the Pokedex rating in the same cutscene). Finishing the game moved the score
*down* by eight, and 63/63 was not reachable by any route. So the reset set is no
longer typed: it is scanned out of every .asm in the tree, ranges expanded the
way the ResetEventRange macro actually expands them, and any ladder rung that
lands in it fails generation.

Usage:
    .venv/bin/python scripts/gen_milestones.py             # fetch and write
    .venv/bin/python scripts/gen_milestones.py --source D  # read a pokered checkout
    .venv/bin/python scripts/gen_milestones.py --check     # fail if stale
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set, Tuple

SOURCE_URL = "https://raw.githubusercontent.com/pret/pokered/master/constants/event_constants.asm"

#: The whole tree, because the reset sites are spread over ~20 files in scripts/
#: and engine/ and there is no way to know which twenty without reading all of
#: them. One request, unpacked in memory, never touched at runtime.
TREE_URL = "https://codeload.github.com/pret/pokered/tar.gz/refs/heads/master"

#: Where inside the checkout the parse looks. macros/ is included because
#: ResetEventRange's expansion is defined there and worth diffing if it changes.
SOURCE_DIRS = ("constants", "scripts", "engine", "data", "home", "macros")

EVENT_CONSTANTS_PATH = "constants/event_constants.asm"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "pokemon_agent" / "data" / "red_milestones.json"

# DEF NUM_EVENTS EQU const_value, the last line of the upstream file. The counter
# must land exactly here or a const_next was mis-evaluated.
NUM_EVENTS = 0xA00
EXPECTED_EVENT_COUNT = 507

# Four events upstream never bothered to name; the placeholder spells out its own
# bit index in hex. They are free, independent cross-checks on the arithmetic:
# if a const_next were dropped or read as relative, these would stop matching.
SELF_NAMING_EVENTS = ("EVENT_1B8", "EVENT_1BF", "EVENT_2A7", "EVENT_67F")

# Cross-checked a second way, against addresses that predate this script in
# pokemon_agent/memory/red.py (ADDR_OAK_PARCEL/ADDR_POKEDEX_FLAG, both read off a
# running game long before the bitfield was enumerated):
#   EVENT_GOT_POKEDEX      = 37 -> 0xD747 + 37//8 = 0xD74B, bit 5  (ADDR_POKEDEX_FLAG, 0x20)
#   EVENT_GOT_OAKS_PARCEL  = 57 -> 0xD747 + 57//8 = 0xD74E, bit 1  (ADDR_OAK_PARCEL, 0x02)
CROSS_CHECKED_EVENTS = {
    "EVENT_GOT_POKEDEX": 37,
    "EVENT_GOT_OAKS_PARCEL": 57,
    "EVENT_BEAT_BROCK": 119,
}

# Every macro in macros/scripts/events.asm that clears a bit. The named-argument
# ones list their events on the same line; ResetEventRange takes two endpoints and
# is handled separately because it clears whole bytes, not just the named span.
RESET_MACROS = (
    "CheckAndResetEventA",
    "CheckAndResetEvent",
    "ResetEventAfterBranchReuseHL",
    "ResetEventForceReuseHL",
    "ResetEventReuseHL",
    "ResetEvents",
    "ResetEvent",
)

_RESET_LINE_RE = re.compile(r"^\s*(?:%s)\b(?P<args>.*)$" % "|".join(RESET_MACROS))
_RESET_RANGE_RE = re.compile(r"^\s*ResetEventRange\b(?P<args>.*)$")
_EVENT_NAME_RE = re.compile(r"\bEVENT_[A-Z0-9_]+\b")
_MARKER_RE = re.compile(r"^\s*DEF\s+(?P<name>\w+)\s+EQU\s+(?P<expr>.*?)\s*$")

# The scan has to come back with these or it did not really run. Each is a flag a
# human has watched the game clear, spread across four different reset shapes:
# a bare ResetEvent, a CheckAndReset that consumes the flag it tests, a two-name
# ResetEvents, and a ResetEventRange whose named endpoints do not mention it.
RESET_SENTINELS = {
    "EVENT_IN_SAFARI_ZONE": "engine/items/item_effects.asm, ResetEvent",
    "EVENT_HALL_OF_FAME_DEX_RATING": "engine/events/pokedex_rating.asm, CheckAndResetEventA",
    "EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2": "scripts/Route23.asm, ResetEvents",
    "EVENT_BEAT_LANCE": "scripts/HallOfFame.asm, ResetEventRange",
}

#: Below this the scan found a handful of lines and missed the rest.
MIN_RESETTABLE_EVENTS = 30

# ---------------------------------------------------------------------------
# The curated ladder.
#
# Ordered the way a normal playthrough hits it, because the runtime scores
# progress as max(ladder_index) over everything set. Entries are
# (id, label, kind, source):
#   kind "event"   -> source is the EVENT_ name (id and source coincide)
#   kind "badge"   -> source is "badge_bit:N", N = bit in wObtainedBadges (0xD356)
#   kind "item"    -> source is "item_id:N", N = bag item id
#   kind "ram_bit" -> source is "ram_bit:0xADDR:N", one bit of one WRAM byte
#
# Events are preferred over items wherever pokered defines one, because bag
# contents are not monotone: Oak's Parcel and the fossils are consumed when handed
# over, and a player can toss anything. The four item-kind entries are key items
# Gen 1 has no flag for *and* never removes from the bag once given.
#
# ram_bit is the last resort, for the end of the game, where pokered's own event
# flags are all scratch space. Both addresses were resolved by walking the "Main
# Data" section of ram/wram.asm and land on wObtainedBadges = 0xD356 and
# wEventFlags = 0xD747 exactly, which is what makes the walk trustworthy; the
# town-visited byte was then confirmed against 494 save states, where it reads
# Pallet+Viridian before Pewter and Pallet+Viridian+Pewter after, and never
# Cerulean in a run that never got there.
#
# Deliberately not on the ladder:
#   Anything the scan above finds on a ResetEvent* line. The Safari Zone is
#     therefore represented by EVENT_GOT_HM03 (Surf, in the Secret House) rather
#     than EVENT_IN_SAFARI_ZONE; Route 23 by the badge check the guards set
#     rather than by a Victory Road boulder switch, all six of which Route23.asm
#     clears on every map load; and the Elite Four by one wElite4Flags bit rather
#     than by six per-room flags that a failed challenge and then the Hall of
#     Fame itself both wipe.
#   EVENT_GOT_POKEBALLS_FROM_OAK -- OaksLab.asm only reaches it if you have beaten
#     the Route 22 rival *and* your bag holds no Poke Balls. It is clear in save
#     states that are well past Brock, so it would be a rung nobody can stand on.
#   The S.S. Anne rival battle -- pokered gives it no persistent event flag, so
#     seven of the eight rival fights are trackable and that one is not.
# ---------------------------------------------------------------------------
LADDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("EVENT_GOT_STARTER", "Chose a starter Pokemon", "event", "EVENT_GOT_STARTER"),
    (
        "EVENT_BATTLED_RIVAL_IN_OAKS_LAB",
        "Fought the rival in Oak's Lab",
        "event",
        "EVENT_BATTLED_RIVAL_IN_OAKS_LAB",
    ),
    (
        "EVENT_GOT_OAKS_PARCEL",
        "Picked up Oak's Parcel in Viridian",
        "event",
        "EVENT_GOT_OAKS_PARCEL",
    ),
    ("EVENT_OAK_GOT_PARCEL", "Delivered Oak's Parcel", "event", "EVENT_OAK_GOT_PARCEL"),
    ("EVENT_GOT_POKEDEX", "Received the Pokedex", "event", "EVENT_GOT_POKEDEX"),
    ("EVENT_GOT_TOWN_MAP", "Got the Town Map from Daisy", "event", "EVENT_GOT_TOWN_MAP"),
    (
        "EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE",
        "Beat the rival on Route 22",
        "event",
        "EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE",
    ),
    ("EVENT_BEAT_BROCK", "Defeated Brock", "event", "EVENT_BEAT_BROCK"),
    ("BADGE_BOULDER", "Boulder Badge", "badge", "badge_bit:0"),
    (
        "EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD",
        "Beat the Super Nerd guarding the Mt. Moon fossils",
        "event",
        "EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD",
    ),
    ("EVENT_GOT_DOME_FOSSIL", "Took the Dome Fossil", "event", "EVENT_GOT_DOME_FOSSIL"),
    ("EVENT_GOT_HELIX_FOSSIL", "Took the Helix Fossil", "event", "EVENT_GOT_HELIX_FOSSIL"),
    (
        "EVENT_BEAT_CERULEAN_RIVAL",
        "Beat the rival in Cerulean City",
        "event",
        "EVENT_BEAT_CERULEAN_RIVAL",
    ),
    ("EVENT_BEAT_MISTY", "Defeated Misty", "event", "EVENT_BEAT_MISTY"),
    ("BADGE_CASCADE", "Cascade Badge", "badge", "badge_bit:1"),
    ("EVENT_MET_BILL", "Met Bill on Route 25", "event", "EVENT_MET_BILL"),
    ("EVENT_GOT_SS_TICKET", "Got the S.S. Ticket", "event", "EVENT_GOT_SS_TICKET"),
    (
        "EVENT_RUBBED_CAPTAINS_BACK",
        "Cured the S.S. Anne captain",
        "event",
        "EVENT_RUBBED_CAPTAINS_BACK",
    ),
    ("EVENT_GOT_HM01", "Got HM01 Cut", "event", "EVENT_GOT_HM01"),
    ("EVENT_SS_ANNE_LEFT", "The S.S. Anne set sail", "event", "EVENT_SS_ANNE_LEFT"),
    ("EVENT_BEAT_LT_SURGE", "Defeated Lt. Surge", "event", "EVENT_BEAT_LT_SURGE"),
    ("BADGE_THUNDER", "Thunder Badge", "badge", "badge_bit:2"),
    (
        "EVENT_GOT_OLD_AMBER",
        "Got the Old Amber in the Pewter Museum",
        "event",
        "EVENT_GOT_OLD_AMBER",
    ),
    ("EVENT_GOT_BIKE_VOUCHER", "Got the Bike Voucher", "event", "EVENT_GOT_BIKE_VOUCHER"),
    ("EVENT_GOT_BICYCLE", "Got the Bicycle", "event", "EVENT_GOT_BICYCLE"),
    ("EVENT_GOT_HM05", "Got HM05 Flash", "event", "EVENT_GOT_HM05"),
    (
        "EVENT_FOUND_ROCKET_HIDEOUT",
        "Found the Rocket Hideout under the Game Corner",
        "event",
        "EVENT_FOUND_ROCKET_HIDEOUT",
    ),
    ("ITEM_LIFT_KEY", "Have the Lift Key", "item", "item_id:74"),
    (
        "EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI",
        "Defeated Giovanni in the Rocket Hideout",
        "event",
        "EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI",
    ),
    ("ITEM_SILPH_SCOPE", "Have the Silph Scope", "item", "item_id:72"),
    ("EVENT_BEAT_ERIKA", "Defeated Erika", "event", "EVENT_BEAT_ERIKA"),
    ("BADGE_RAINBOW", "Rainbow Badge", "badge", "badge_bit:3"),
    (
        "EVENT_BEAT_POKEMON_TOWER_RIVAL",
        "Beat the rival in Pokemon Tower",
        "event",
        "EVENT_BEAT_POKEMON_TOWER_RIVAL",
    ),
    (
        "EVENT_BEAT_GHOST_MAROWAK",
        "Laid the Marowak ghost to rest",
        "event",
        "EVENT_BEAT_GHOST_MAROWAK",
    ),
    ("EVENT_RESCUED_MR_FUJI", "Rescued Mr. Fuji", "event", "EVENT_RESCUED_MR_FUJI"),
    ("EVENT_GOT_POKE_FLUTE", "Got the Poke Flute", "event", "EVENT_GOT_POKE_FLUTE"),
    (
        "EVENT_BEAT_ROUTE12_SNORLAX",
        "Cleared the Snorlax on Route 12",
        "event",
        "EVENT_BEAT_ROUTE12_SNORLAX",
    ),
    ("EVENT_GOT_HM02", "Got HM02 Fly", "event", "EVENT_GOT_HM02"),
    ("EVENT_GOT_HM03", "Got HM03 Surf in the Safari Zone", "event", "EVENT_GOT_HM03"),
    (
        "EVENT_GAVE_GOLD_TEETH",
        "Returned the Gold Teeth to the Warden",
        "event",
        "EVENT_GAVE_GOLD_TEETH",
    ),
    ("EVENT_GOT_HM04", "Got HM04 Strength", "event", "EVENT_GOT_HM04"),
    ("EVENT_BEAT_KOGA", "Defeated Koga", "event", "EVENT_BEAT_KOGA"),
    ("BADGE_SOUL", "Soul Badge", "badge", "badge_bit:4"),
    ("ITEM_CARD_KEY", "Have the Card Key", "item", "item_id:48"),
    (
        "EVENT_BEAT_SILPH_CO_RIVAL",
        "Beat the rival in Silph Co.",
        "event",
        "EVENT_BEAT_SILPH_CO_RIVAL",
    ),
    (
        "EVENT_BEAT_SILPH_CO_GIOVANNI",
        "Defeated Giovanni in Silph Co.",
        "event",
        "EVENT_BEAT_SILPH_CO_GIOVANNI",
    ),
    ("EVENT_GOT_MASTER_BALL", "Got the Master Ball", "event", "EVENT_GOT_MASTER_BALL"),
    ("EVENT_BEAT_SABRINA", "Defeated Sabrina", "event", "EVENT_BEAT_SABRINA"),
    ("BADGE_MARSH", "Marsh Badge", "badge", "badge_bit:5"),
    ("ITEM_SECRET_KEY", "Have the Secret Key", "item", "item_id:43"),
    ("EVENT_BEAT_BLAINE", "Defeated Blaine", "event", "EVENT_BEAT_BLAINE"),
    ("BADGE_VOLCANO", "Volcano Badge", "badge", "badge_bit:6"),
    (
        "EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI",
        "Defeated Giovanni in the Viridian Gym",
        "event",
        "EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI",
    ),
    ("BADGE_EARTH", "Earth Badge", "badge", "badge_bit:7"),
    (
        "EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE",
        "Beat the rival on Route 22 again",
        "event",
        "EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE",
    ),
    (
        "EVENT_PASSED_EARTHBADGE_CHECK",
        "Passed the last Route 23 badge check",
        "event",
        "EVENT_PASSED_EARTHBADGE_CHECK",
    ),
    # wTownVisitedFlag = 0xD70B, bit 9 = INDIGO_PLATEAU. Set on arrival and never
    # cleared. Victory Road is the only way in on a first visit, so this is the
    # rung that says Victory Road is behind you -- which is what the six boulder
    # switches were supposed to say and cannot, being scratch space Route23.asm
    # zeroes every time the map loads.
    (
        "TOWN_INDIGO_PLATEAU",
        "Cleared Victory Road and reached the Indigo Plateau",
        "ram_bit",
        "ram_bit:0xD70B:9",
    ),
    # wElite4Flags = 0xD734, bit 0 = BIT_UNUSED_BEAT_ELITE_4. HallOfFame.asm sets
    # it on the way in and nothing in the decomp clears it -- the game itself
    # never reads it, which is precisely why it survives. Everything else about
    # the Elite Four is wiped twice over: once by IndigoPlateauLobby.asm when a
    # failed challenge sends you back to the lobby, and again by HallOfFame.asm.
    # So the Elite Four is one rung here, because one bit is all that is left
    # standing afterwards.
    (
        "ELITE_FOUR_CHAMPION",
        "Beat the Elite Four and entered the Hall of Fame",
        "ram_bit",
        "ram_bit:0xD734:0",
    ),
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[+-]|\$[0-9A-Fa-f]+|[0-9]+")


def eval_const_expr(expr: str) -> int:
    """Evaluate an rgbds counter expression, e.g. ``$F0 - 2``.

    Only ``$hex``/decimal literals joined by + and - occur in this file; anything
    richer would need a real expression parser and should fail loudly instead.
    """
    if not _TOKEN_RE.fullmatch("".join(expr.split())) and not _TOKEN_RE.findall(expr):
        raise ValueError(f"unparseable counter expression: {expr!r}")
    total = 0
    sign = 1
    seen = False
    for token in _TOKEN_RE.findall(expr):
        if token == "+":
            sign = 1
        elif token == "-":
            sign = -1
        else:
            total += sign * (int(token[1:], 16) if token.startswith("$") else int(token))
            seen = True
    if not seen:
        raise ValueError(f"unparseable counter expression: {expr!r}")
    return total


def _counter_walk(text: str) -> Iterable[Tuple[int, str, str, int]]:
    """Yield ``(lineno, directive, args, counter)`` for every rgbds line.

    *counter* is the value ``const_value`` holds as the line is reached, which is
    what both the event indices and the ``DEF ... EQU const_value`` range markers
    are read off. One walk, two consumers, no chance of them disagreeing.
    """
    counter = 0
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        directive, args = parts[0], " ".join(parts[1:])
        yield lineno, directive, args, counter
        if directive == "const_def":
            counter = eval_const_expr(args) if args else 0
        elif directive == "const_next":
            counter = eval_const_expr(args)
        elif directive == "const_skip":
            counter += eval_const_expr(args) if args else 1
        elif directive == "const":
            counter += 1
    yield len(text.splitlines()) + 1, "", "", counter


def parse_event_constants(text: str) -> Tuple[Dict[str, int], int]:
    """Return (EVENT_ name -> bit index, final counter value)."""
    events: Dict[str, int] = {}
    counter = 0
    for lineno, directive, args, counter in _counter_walk(text):
        if directive != "const":
            continue
        name = args.split()[0]
        if name in events:
            raise ValueError(f"line {lineno}: duplicate event {name}")
        events[name] = counter
    return events, counter


def parse_range_markers(text: str) -> Dict[str, int]:
    """``DEF NAME EQU const_value``-style labels, e.g. INDIGO_PLATEAU_EVENTS_END.

    ResetEventRange names its endpoints with these rather than with events, so
    without them the two range resets in the decomp cannot be expanded at all.
    """
    markers: Dict[str, int] = {}
    for _lineno, directive, args, counter in _counter_walk(text):
        if directive != "DEF":
            continue
        match = _MARKER_RE.match(f"DEF {args}")
        if match is None or "const_value" not in match.group("expr"):
            continue
        expr = match.group("expr").replace("const_value", str(counter))
        markers[match.group("name")] = eval_const_expr(expr)
    return markers


def range_cleared_bits(start: int, end: int) -> Set[int]:
    """Every bit ``ResetEventRange start, end`` actually clears.

    The macro works a byte at a time (macros/scripts/events.asm): the partial
    start byte from ``start % 8`` up, whole bytes in between, and the partial end
    byte up to ``end % 8``. It therefore reaches past both named endpoints, which
    is how EVENT_AUTOWALKED_INTO_LORELEIS_ROOM ends up cleared by a range whose
    endpoints never mention it.
    """
    start_byte, end_byte = start // 8, end // 8
    if start_byte == end_byte:
        return set(range(start, end + 1))
    cleared = set(range(start, (start_byte + 1) * 8))
    cleared.update(range((start_byte + 1) * 8, end_byte * 8))
    cleared.update(range(end_byte * 8, end + 1))
    return cleared


def scan_resettable(
    sources: Mapping[str, str], events: Mapping[str, int], markers: Mapping[str, int]
) -> Set[str]:
    """Every event name the decomp can clear again, from every .asm in the tree."""
    by_index = {index: name for name, index in events.items()}

    def resolve(token: str) -> int:
        token = token.strip()
        if token in events:
            return events[token]
        if token in markers:
            return markers[token]
        raise SystemExit(f"ResetEventRange endpoint {token!r} resolves to nothing")

    resettable: Set[str] = set()
    for path, text in sorted(sources.items()):
        if path.startswith("macros/"):
            continue  # the macro definitions, not uses
        for raw_line in text.splitlines():
            line = raw_line.split(";", 1)[0]
            range_match = _RESET_RANGE_RE.match(line)
            if range_match is not None:
                args = [a for a in range_match.group("args").split(",") if a.strip()]
                start, end = resolve(args[0]), resolve(args[1])
                resettable.update(
                    by_index[bit] for bit in range_cleared_bits(start, end) if bit in by_index
                )
                continue
            named = _RESET_LINE_RE.match(line)
            if named is not None:
                resettable.update(_EVENT_NAME_RE.findall(named.group("args")))
    return resettable


def validate(events: Dict[str, int], final_counter: int, resettable: Set[str]) -> None:
    if len(events) != EXPECTED_EVENT_COUNT:
        raise SystemExit(f"expected {EXPECTED_EVENT_COUNT} events, parsed {len(events)}")
    if final_counter != NUM_EVENTS:
        raise SystemExit(
            f"counter ended at {final_counter:#x}, upstream NUM_EVENTS is {NUM_EVENTS:#x} "
            "-- a const_next was mis-evaluated"
        )
    for name in SELF_NAMING_EVENTS:
        expected = int(name.removeprefix("EVENT_"), 16)
        if events.get(name) != expected:
            raise SystemExit(f"{name} parsed as {events.get(name)}, its name says {expected}")
    for name, expected in CROSS_CHECKED_EVENTS.items():
        if events.get(name) != expected:
            raise SystemExit(f"{name} parsed as {events.get(name)}, expected {expected}")

    if len(resettable) < MIN_RESETTABLE_EVENTS:
        raise SystemExit(
            f"the reset scan found only {len(resettable)} events; the decomp has at least "
            f"{MIN_RESETTABLE_EVENTS}. A silent miss here is what put eight resettable "
            "flags on the ladder."
        )
    for name, where in RESET_SENTINELS.items():
        if name not in resettable:
            raise SystemExit(f"the reset scan missed {name}, which {where} clears")

    seen: set[str] = set()
    for entry_id, label, kind, source in LADDER:
        if entry_id in seen:
            raise SystemExit(f"duplicate ladder id {entry_id}")
        seen.add(entry_id)
        if not label:
            raise SystemExit(f"ladder entry {entry_id} has no label")
        if kind == "event":
            if source not in events:
                raise SystemExit(f"ladder entry {entry_id} references unknown event {source}")
            if source != entry_id:
                raise SystemExit(f"event ladder entry {entry_id} should be named {source}")
            if source in resettable:
                raise SystemExit(f"ladder entry {entry_id} uses a flag pokered can reset")
        elif kind == "ram_bit":
            _, addr, bit = source.split(":")
            if not 0xC000 <= int(addr, 16) <= 0xDFFF:
                raise SystemExit(f"ladder entry {entry_id} points outside WRAM at {addr}")
            if not 0 <= int(bit) <= 15:
                raise SystemExit(f"ladder entry {entry_id} has out-of-range bit {bit}")
        elif kind == "badge":
            bit = int(source.split(":", 1)[1])
            if not 0 <= bit <= 7:
                raise SystemExit(f"ladder entry {entry_id} has out-of-range badge bit {bit}")
        elif kind == "item":
            item_id = int(source.split(":", 1)[1])
            if not 1 <= item_id <= 255:
                raise SystemExit(f"ladder entry {entry_id} has out-of-range item id {item_id}")
        else:
            raise SystemExit(f"ladder entry {entry_id} has unknown kind {kind!r}")


def build_document(events: Dict[str, int], resettable: Set[str]) -> Dict[str, object]:
    ordered: List[Dict[str, object]] = [
        {"id": name, "bit_index": index}
        for name, index in sorted(events.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    ladder = [
        {"id": entry_id, "label": label, "kind": kind, "source": source}
        for entry_id, label, kind, source in LADDER
    ]
    return {
        "source": SOURCE_URL,
        "source_tree": TREE_URL,
        "flag_base_address": 0xD747,
        "num_event_bits": NUM_EVENTS,
        "events": ordered,
        # Checked in so the invariant "no rung is resettable" is testable without
        # a network round trip, and so a change upstream shows up as a diff.
        "resettable": sorted(resettable),
        "ladder": ladder,
    }


def fetch_sources(source: str | None) -> Dict[str, str]:
    """Every .asm in the decomp, keyed by repo-relative path.

    *source* is a pokered checkout to read instead of downloading. Files outside
    SOURCE_DIRS are skipped: they hold graphics and audio and nothing that sets
    or clears an event.
    """
    if source:
        root = Path(source)
        if not (root / EVENT_CONSTANTS_PATH).exists():
            raise SystemExit(f"{root} is not a pokered checkout ({EVENT_CONSTANTS_PATH} missing)")
        return {
            str(path.relative_to(root)): path.read_text(encoding="utf-8", errors="replace")
            for directory in SOURCE_DIRS
            for path in sorted((root / directory).rglob("*.asm"))
        }

    with urllib.request.urlopen(TREE_URL, timeout=180) as response:  # noqa: S310
        payload = response.read()
    sources: Dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            # The tarball has one top-level directory; strip it.
            relative = member.name.partition("/")[2]
            if not member.isfile() or not relative.endswith(".asm"):
                continue
            if not relative.startswith(tuple(d + "/" for d in SOURCE_DIRS)):
                continue
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - a directory entry lying about itself
                continue
            sources[relative] = handle.read().decode("utf-8", errors="replace")
    if EVENT_CONSTANTS_PATH not in sources:
        raise SystemExit(f"{TREE_URL} yielded no {EVENT_CONSTANTS_PATH}")
    return sources


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="a local pokered checkout instead of fetching")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the checked-in JSON differs from what would be written",
    )
    args = parser.parse_args(argv)

    sources = fetch_sources(args.source)
    constants = sources[EVENT_CONSTANTS_PATH]
    events, final_counter = parse_event_constants(constants)
    resettable = scan_resettable(sources, events, parse_range_markers(constants))
    validate(events, final_counter, resettable)
    document = build_document(events, resettable)
    payload = json.dumps(document, indent=2) + "\n"

    existing = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else None
    if args.check:
        if existing != payload:
            print(f"{OUTPUT_PATH} is stale; rerun scripts/gen_milestones.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT_PATH} is up to date")
        return 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(payload, encoding="utf-8")

    kinds: Dict[str, int] = {}
    for _, _, kind, _ in LADDER:
        kinds[kind] = kinds.get(kind, 0) + 1
    breakdown = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
    state = "unchanged" if existing == payload else "updated"
    print(f"source     {SOURCE_URL}")
    print(f"events     {len(events)} named bits, counter ended at {final_counter:#x}")
    print(f"resettable {len(resettable)} events the decomp can clear again")
    print(f"ladder     {len(LADDER)} entries ({breakdown})")
    print(f"first/last {LADDER[0][0]} -> {LADDER[-1][0]}")
    print(f"wrote      {OUTPUT_PATH.relative_to(REPO_ROOT)} ({state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
