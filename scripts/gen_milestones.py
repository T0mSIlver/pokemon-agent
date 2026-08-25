#!/usr/bin/env python3
"""Regenerate pokemon_agent/data/red_milestones.json from the pokered decomp.

Source of truth for the event bitfield at wEventFlags (0xD747):
  https://raw.githubusercontent.com/pret/pokered/master/constants/event_constants.asm

Same precedent as MAP_NAMES in pokemon_agent/memory/red.py: the upstream file is
parsed once, offline, and the result is checked in. A running emulator must not
depend on network access, and the diff has to be reviewable.

Unlike map_constants.asm, this file is *not* a straight line count. rgbds'
const_def/const/const_skip/const_next macros move a counter:

    const_def            counter = 0
    const NAME           NAME = counter; counter += 1
    const_skip [N]       counter += N (N defaults to 1)
    const_next EXPR      counter = EXPR   <- absolute, not relative

Every section header in the file is a `const_next $XXX` that jumps the counter to
the start of that map's slice of the bitfield, so enumerating `const` lines in
order and numbering them 0..506 gives 507 wrong indices. Ask MAP_NAMES how much a
drifted table costs.

Usage:
    .venv/bin/python scripts/gen_milestones.py            # fetch and write
    .venv/bin/python scripts/gen_milestones.py --source F # parse a local copy
    .venv/bin/python scripts/gen_milestones.py --check    # fail if stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

SOURCE_URL = "https://raw.githubusercontent.com/pret/pokered/master/constants/event_constants.asm"

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

# Events pokered's scripts can *clear* again -- every EVENT_ name that appears
# after a ResetEvent* macro anywhere in the decomp. They are the flags that make a
# ladder non-monotone, so none of them may be used as a rung. Two were caught this
# way: EVENT_IN_SAFARI_ZONE (set only while you are inside) and
# EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH (cleared by VictoryRoad2F.asm when the
# boulder is lifted off).
RESETTABLE_EVENTS = frozenset(
    {
        "EVENT_1ST_LOCK_OPENED",
        "EVENT_2A7",
        "EVENT_2ND_ROUTE22_RIVAL_BATTLE",
        "EVENT_67F",
        "EVENT_BILL_SAID_USE_CELL_SEPARATOR",
        "EVENT_BOUGHT_MUSEUM_TICKET",
        "EVENT_FIGHT_ROUTE12_SNORLAX",
        "EVENT_FIGHT_ROUTE16_SNORLAX",
        "EVENT_IN_PURIFIED_ZONE",
        "EVENT_IN_SAFARI_ZONE",
        "EVENT_IN_SEAFOAM_ISLANDS",
        "EVENT_LAB_STILL_REVIVING_FOSSIL",
        "EVENT_MANSION_SWITCH_ON",
        "EVENT_NUGGET_REWARD_AVAILABLE",
        "EVENT_PIKACHU_FAN_BOAST",
        "EVENT_POKEMON_TOWER_RIVAL_ON_LEFT",
        "EVENT_SAFARI_GAME_OVER",
        "EVENT_SEEL_FAN_BOAST",
        "EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH",
    }
)

# ---------------------------------------------------------------------------
# The curated ladder.
#
# Ordered the way a normal playthrough hits it, because the runtime scores
# progress as max(ladder_index) over everything set. Entries are
# (id, label, kind, source):
#   kind "event" -> source is the EVENT_ name (id and source coincide)
#   kind "badge" -> source is "badge_bit:N", N = bit in wObtainedBadges (0xD356)
#   kind "item"  -> source is "item_id:N", N = bag item id
#
# Events are preferred over items wherever pokered defines one, because bag
# contents are not monotone: Oak's Parcel and the fossils are consumed when handed
# over, and a player can toss anything. The four item-kind entries are key items
# Gen 1 has no flag for *and* never removes from the bag once given.
#
# Deliberately not on the ladder:
#   Anything in RESETTABLE_EVENTS above. The Safari Zone is therefore represented
#     by EVENT_GOT_HM03 (Surf, in the Secret House) rather than EVENT_IN_SAFARI_ZONE,
#     and Victory Road by the 2F boulder switch rather than the 1F one.
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
        "EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2",
        "Opened the Victory Road 2F barrier",
        "event",
        "EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2",
    ),
    (
        "EVENT_AUTOWALKED_INTO_LORELEIS_ROOM",
        "Cleared Victory Road and reached the Elite Four",
        "event",
        "EVENT_AUTOWALKED_INTO_LORELEIS_ROOM",
    ),
    (
        "EVENT_BEAT_LORELEIS_ROOM_TRAINER_0",
        "Defeated Lorelei",
        "event",
        "EVENT_BEAT_LORELEIS_ROOM_TRAINER_0",
    ),
    (
        "EVENT_BEAT_BRUNOS_ROOM_TRAINER_0",
        "Defeated Bruno",
        "event",
        "EVENT_BEAT_BRUNOS_ROOM_TRAINER_0",
    ),
    (
        "EVENT_BEAT_AGATHAS_ROOM_TRAINER_0",
        "Defeated Agatha",
        "event",
        "EVENT_BEAT_AGATHAS_ROOM_TRAINER_0",
    ),
    ("EVENT_BEAT_LANCE", "Defeated Lance", "event", "EVENT_BEAT_LANCE"),
    (
        "EVENT_BEAT_CHAMPION_RIVAL",
        "Defeated the Champion",
        "event",
        "EVENT_BEAT_CHAMPION_RIVAL",
    ),
    (
        "EVENT_HALL_OF_FAME_DEX_RATING",
        "Entered the Hall of Fame",
        "event",
        "EVENT_HALL_OF_FAME_DEX_RATING",
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


def parse_event_constants(text: str) -> Tuple[Dict[str, int], int]:
    """Return (EVENT_ name -> bit index, final counter value)."""
    counter = 0
    events: Dict[str, int] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        directive, args = parts[0], " ".join(parts[1:])
        if directive == "const_def":
            counter = eval_const_expr(args) if args else 0
        elif directive == "const_next":
            counter = eval_const_expr(args)
        elif directive == "const_skip":
            counter += eval_const_expr(args) if args else 1
        elif directive == "const":
            name = parts[1]
            if name in events:
                raise ValueError(f"line {lineno}: duplicate event {name}")
            events[name] = counter
            counter += 1
        # DEF ... EQU lines and anything else only read the counter; ignore them.
    return events, counter


def validate(events: Dict[str, int], final_counter: int) -> None:
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
            if source in RESETTABLE_EVENTS:
                raise SystemExit(f"ladder entry {entry_id} uses a flag pokered can reset")
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


def build_document(events: Dict[str, int]) -> Dict[str, object]:
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
        "flag_base_address": 0xD747,
        "num_event_bits": NUM_EVENTS,
        "events": ordered,
        "ladder": ladder,
    }


def fetch(source: str | None) -> str:
    if source:
        return Path(source).read_text(encoding="utf-8")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310
        return response.read().decode("utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="local event_constants.asm instead of fetching")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the checked-in JSON differs from what would be written",
    )
    args = parser.parse_args(argv)

    events, final_counter = parse_event_constants(fetch(args.source))
    validate(events, final_counter)
    document = build_document(events)
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
    print(f"ladder     {len(LADDER)} entries ({breakdown})")
    print(f"first/last {LADDER[0][0]} -> {LADDER[-1][0]}")
    print(f"wrote      {OUTPUT_PATH.relative_to(REPO_ROOT)} ({state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
