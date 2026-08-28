"""Construct the party/badge state an HM field move needs, in RAM, on a copy.

Why this exists
---------------
This project only trusts a mechanism once it has been driven on the real
cartridge. Cut and the Bicycle could be verified that way because a save in the
corpus happened to have them. The other four field moves cannot: no save in
``saves/`` carries HM02..HM05, and none has the Soul or Rainbow badge, so Surf,
Strength, Flash and Fly would have to ship on reasoning alone -- the exact
"dead mechanism" pattern this repo has already found nine times.

So the state is *built* instead of waited for. A save is copied out of the
corpus, loaded, and two things are poked into WRAM: a move id plus its PP into
one of a party Pokemon's four move slots, and the badge bit the game checks
before it will run that move outside battle. Nothing else is touched, and the
file in ``saves/`` is only ever read.

Everything written here is derived, not guessed:

Party struct offsets -- from pokered ``macros/ram.asm``, ``box_struct`` followed
by ``party_struct``::

    Species db      0
    HP      dw      1..2
    BoxLevel db     3
    Status  db      4
    Type1/Type2 db  5, 6
    CatchRate db    7
    Moves   ds 4    8..11      <- OFF_MOVES
    OTID    dw      12..13
    Exp     ds 3    14..16
    HPExp..SpecialExp dw x5  17..26
    DVs     dw      27..28
    PP      ds 4    29..32     <- OFF_PP
    ; party_struct continues past box_struct:
    Level   db      33
    MaxHP/Attack/Defense/Speed/Special dw x5  34..43

which is 44 bytes = ``red.PARTY_MON_SIZE``, and matches the layout already
documented in ``RedBlueMemoryReader._read_pokemon`` byte for byte. The slot base
is ``red.ADDR_PARTY_DATA + slot * red.PARTY_MON_SIZE``; move slot *i* is
``base + 8 + i`` and its PP counter is ``base + 29 + i``. The top two bits of a
PP byte are PP Ups (``_read_pokemon`` masks with 0x3F and shifts >> 6 for them),
so a freshly written move gets its raw max PP with those bits clear.

Badge bits -- from pokered ``constants/ram_constants.asm``, the ``wObtainedBadges``
bit block: BOULDER 0, CASCADE 1, THUNDER 2, RAINBOW 3, SOUL 4, MARSH 5,
VOLCANO 6, EARTH 7. That is the same order as ``red.BADGE_NAMES``, so the bit
index is just the index into that list.

Which badge gates which move -- from pokered
``engine/menus/start_sub_menus.asm``, ``StartMenu_Pokemon.choseOutOfBattleMove``::

    .fly       bit BIT_THUNDERBADGE, a  / jp z, .newBadgeRequired
    .cut       bit BIT_CASCADEBADGE, a  / jp z, .newBadgeRequired
    .surf      bit BIT_SOULBADGE, a     / jp z, .newBadgeRequired
    .strength  bit BIT_RAINBOWBADGE, a  / jp z, .newBadgeRequired
    .flash     bit BIT_BOULDERBADGE, a  / jp z, .newBadgeRequired

(``a`` is ``wObtainedBadges``, loaded immediately before the jump table.)

PP -- from pokered ``data/moves/moves.asm``: CUT 30, FLY 15, SURF 15,
STRENGTH 15, FLASH 20.

HM item ids -- from pokered ``constants/item_constants.asm``: ``HM01 EQU $C4``
then CUT, FLY, SURF, STRENGTH, FLASH in order, i.e. 196..200. Matches
``red.ITEM_NAMES``.

Field moves -- from pokered ``data/moves/field_moves.asm``: CUT, FLY, SURF,
STRENGTH, FLASH, DIG, TELEPORT, SOFTBOILED (plus one unused entry). The party
submenu lists them in *move-slot* order (``GetMonFieldMoves`` walks the four
move bytes), then STATS, SWITCH, CANCEL, so ``wMaxMenuItem`` is
``2 + len(field moves)`` and ``wTopMenuItemY`` is ``12 - 2 * len(field moves)``.
Both come from ``StartMenu_Pokemon.adjustMenuVariablesLoop``, which starts at
``lb bc, 2, 12`` and does ``inc b`` / ``dec c`` / ``dec c`` per field move.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pokemon_agent.memory.red import (
    ADDR_BADGES,
    ADDR_BAG_COUNT,
    ADDR_BAG_ITEMS,
    ADDR_PARTY_COUNT,
    ADDR_PARTY_DATA,
    BADGE_NAMES,
    BAG_ITEM_CAPACITY,
    PARTY_MON_SIZE,
)

# ---------------------------------------------------------------------------
# Party struct offsets (pokered macros/ram.asm; see module docstring)
# ---------------------------------------------------------------------------

NUM_MOVES = 4

OFF_SPECIES = 0
OFF_HP = 1  # 2 bytes, big-endian
OFF_BOX_LEVEL = 3
OFF_STATUS = 4
OFF_TYPE1 = 5
OFF_TYPE2 = 6
OFF_CATCH_RATE = 7
OFF_MOVES = 8  # 4 bytes, one move id per slot
OFF_OT_ID = 12
OFF_EXP = 14
OFF_DVS = 27
OFF_PP = 29  # 4 bytes; low 6 bits the counter, high 2 bits PP Ups
OFF_LEVEL = 33
OFF_MAX_HP = 34
OFF_ATTACK = 36
OFF_DEFENSE = 38
OFF_SPEED = 40
OFF_SPECIAL = 42

assert OFF_SPECIAL + 2 == PARTY_MON_SIZE, "party struct offsets do not fill 44 bytes"

#: Low six bits of a PP byte. The other two are PP Ups.
PP_MASK = 0x3F

# ---------------------------------------------------------------------------
# Field moves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMove:
    """One HM field move and everything the game checks before running it."""

    name: str
    move_id: int
    max_pp: int
    badge: str
    hm_item_id: int

    @property
    def badge_bit(self) -> int:
        """Bit index in wObtainedBadges, i.e. the position in ``BADGE_NAMES``."""
        return BADGE_NAMES.index(self.badge)

    @property
    def hm_item_name(self) -> str:
        return f"HM{self.hm_item_id - 195:02d}"


#: Keyed by the name `red.MOVE_NAMES` reports, upper-cased. Lookups are
#: case-insensitive so "surf", "Surf" and "SURF" all work.
FIELD_MOVES: Dict[str, FieldMove] = {
    "CUT": FieldMove("Cut", 15, 30, "Cascade", 196),
    "FLY": FieldMove("Fly", 19, 15, "Thunder", 197),
    "SURF": FieldMove("Surf", 57, 15, "Soul", 198),
    "STRENGTH": FieldMove("Strength", 70, 15, "Rainbow", 199),
    "FLASH": FieldMove("Flash", 148, 20, "Boulder", 200),
}

#: Every move that gets its own row in the party submenu, from pokered
#: data/moves/field_moves.asm. DIG/TELEPORT/SOFTBOILED are not HMs but they do
#: take a row, so they count towards wMaxMenuItem.
FIELD_MOVE_IDS = frozenset({15, 19, 57, 70, 148, 91, 100, 135})

#: wMaxMenuItem with no field moves at all: STATS, SWITCH, CANCEL as 0, 1, 2.
SUBMENU_BASE_MAX_ITEM = 2
#: wTopMenuItemY with no field moves; each field move moves it up two rows.
SUBMENU_BASE_TOP_Y = 12


def field_move(name: str) -> FieldMove:
    """Look a field move up by name, case-insensitively."""
    try:
        return FIELD_MOVES[name.strip().upper()]
    except KeyError:
        raise KeyError(
            f"{name!r} is not one of the HM field moves: {', '.join(sorted(FIELD_MOVES))}"
        ) from None


# ---------------------------------------------------------------------------
# Writing bytes
# ---------------------------------------------------------------------------


def write_u8(emulator: Any, addr: int, value: int) -> None:
    """Write one byte of emulated memory.

    ``PyBoyEmulator`` is read-only by design -- it exposes ``read_u8`` and no
    counterpart -- so this reaches through to PyBoy's own ``pyboy.memory``
    mapping, the same object ``PyBoyEmulator.read_u8`` reads from. Doing it here
    rather than adding a writer to the harness keeps the ability to rewrite the
    game's state inside the test tree, where it belongs.
    """
    pyboy = getattr(emulator, "_pyboy", None)
    if pyboy is None:
        raise TypeError(
            f"{type(emulator).__name__} has no PyBoy behind it; "
            "doctoring RAM needs the PyBoy backend"
        )
    pyboy.memory[addr] = value & 0xFF


# ---------------------------------------------------------------------------
# The record of what was changed
# ---------------------------------------------------------------------------


@dataclass
class Doctored:
    """What ``give_field_move`` changed, so a test can assert on all of it."""

    source_state: Path
    state: Path
    move: FieldMove
    party_slot: int
    move_slot: int
    replaced_move_id: int
    replaced_pp: int
    badges_before: int
    badges_after: int
    badge_was_already_set: bool
    #: Row this move occupies in the party field-move submenu, 0-based.
    submenu_row: int
    #: Field moves the chosen Pokemon knew before and after the write.
    field_moves_before: int
    field_moves_after: int
    hm_item_id: Optional[int] = None
    hm_bag_slot: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def expected_max_menu_item(self) -> int:
        """wMaxMenuItem the submenu should report after doctoring."""
        return SUBMENU_BASE_MAX_ITEM + self.field_moves_after

    @property
    def expected_top_menu_item_y(self) -> int:
        """wTopMenuItemY the submenu should report after doctoring."""
        return SUBMENU_BASE_TOP_Y - 2 * self.field_moves_after


# ---------------------------------------------------------------------------
# Reading the party struct straight, without the reader's decoding
# ---------------------------------------------------------------------------


def party_slot_base(slot: int) -> int:
    return ADDR_PARTY_DATA + slot * PARTY_MON_SIZE


def raw_party_mon(emulator: Any, slot: int) -> bytes:
    """The 44 raw bytes of one party slot."""
    return emulator.read_range(party_slot_base(slot), PARTY_MON_SIZE)


def move_ids(emulator: Any, slot: int) -> List[int]:
    data = raw_party_mon(emulator, slot)
    return [data[OFF_MOVES + i] for i in range(NUM_MOVES)]


def free_move_slot(emulator: Any, slot: int) -> Optional[int]:
    """The first empty move slot of a party member, or None if it knows four."""
    for index, move_id in enumerate(move_ids(emulator, slot)):
        if move_id == 0:
            return index
    return None


def count_field_moves(ids: List[int]) -> int:
    return sum(1 for move_id in ids if move_id in FIELD_MOVE_IDS)


# ---------------------------------------------------------------------------
# The doctoring itself
# ---------------------------------------------------------------------------


def copy_state(save_path: str | Path, tmp_path: str | Path, name: str = "doctored.state") -> Path:
    """Copy a save out of the (read-only) corpus into a scratch directory."""
    destination = Path(tmp_path) / name
    shutil.copy(str(save_path), str(destination))
    return destination


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def give_field_move(
    emulator: Any,
    save_path: str | Path,
    move: str,
    tmp_path: str | Path,
    *,
    party_slot: int = 0,
    move_slot: Optional[int] = None,
    give_hm_item: bool = False,
    state_name: str = "doctored.state",
) -> Doctored:
    """Load a copy of *save_path* and teach party member *party_slot* *move*.

    The save file itself is never opened for writing: it is copied into
    *tmp_path* first, and the copy is what gets loaded. Every change after that
    is made in emulated RAM, so nothing on disk changes at all.

    Parameters
    ----------
    emulator:
        A ``PyBoyEmulator`` with the Red ROM already loaded.
    move:
        One of CUT, FLY, SURF, STRENGTH, FLASH (case-insensitive).
    move_slot:
        Which of the four move slots to write into. Default: the first empty
        one, so nothing the Pokemon already knows is lost. Passing an occupied
        slot explicitly is allowed and the displaced move is recorded.
    give_hm_item:
        Also drop the matching HM into the bag. Not needed for the move to
        work -- the badge and the move are what the game checks -- but a bag
        that carries HM03 while a mon knows Surf is a more honest fixture.
    """
    spec = field_move(move)
    source = Path(save_path)
    state = copy_state(source, tmp_path, state_name)

    emulator.load_state(str(state))
    emulator.settle()

    count = emulator.read_u8(ADDR_PARTY_COUNT)
    if not 0 <= party_slot < min(count, 6):
        raise ValueError(f"party slot {party_slot} is empty; the save has {count} Pokemon")

    before = move_ids(emulator, party_slot)
    if move_slot is None:
        free = free_move_slot(emulator, party_slot)
        if free is None:
            raise ValueError(
                f"party slot {party_slot} already knows four moves; "
                "pass move_slot to choose which one to overwrite"
            )
        move_slot = free
    if not 0 <= move_slot < NUM_MOVES:
        raise ValueError(f"move slot {move_slot} is not one of 0..3")

    base = party_slot_base(party_slot)
    raw = raw_party_mon(emulator, party_slot)
    replaced_move_id = raw[OFF_MOVES + move_slot]
    replaced_pp = raw[OFF_PP + move_slot]

    write_u8(emulator, base + OFF_MOVES + move_slot, spec.move_id)
    # PP Ups deliberately cleared: a move the game never taught has never been
    # fed a PP Up, and leaving a previous move's PP Up bits behind would report
    # a PP the move cannot actually have.
    write_u8(emulator, base + OFF_PP + move_slot, spec.max_pp & PP_MASK)

    badges_before = emulator.read_u8(ADDR_BADGES)
    badges_after = badges_before | (1 << spec.badge_bit)
    write_u8(emulator, ADDR_BADGES, badges_after)

    after = move_ids(emulator, party_slot)
    record = Doctored(
        source_state=source,
        state=state,
        move=spec,
        party_slot=party_slot,
        move_slot=move_slot,
        replaced_move_id=replaced_move_id,
        replaced_pp=replaced_pp,
        badges_before=badges_before,
        badges_after=badges_after,
        badge_was_already_set=bool(badges_before & (1 << spec.badge_bit)),
        submenu_row=count_field_moves(after[:move_slot]),
        field_moves_before=count_field_moves(before),
        field_moves_after=count_field_moves(after),
    )
    if replaced_move_id:
        record.notes.append(
            f"move slot {move_slot} held move id {replaced_move_id}; it was overwritten"
        )

    if give_hm_item:
        record.hm_item_id = spec.hm_item_id
        record.hm_bag_slot = add_bag_item(emulator, spec.hm_item_id)

    return record


def add_bag_item(emulator: Any, item_id: int, quantity: int = 1) -> int:
    """Append an item to the bag, keeping the $FF terminator in place.

    The bag is ``wNumBagItems`` (a count) followed by ``wBagItems``, which is
    ``BAG_ITEM_CAPACITY * 2 + 1`` bytes of (id, quantity) pairs closed by $FF --
    ``red.read_bag`` reads exactly that. Returns the bag row the item ended up
    on, which is what a menu-driving test needs.
    """
    count = emulator.read_u8(ADDR_BAG_COUNT)
    for index in range(min(count, BAG_ITEM_CAPACITY)):
        if emulator.read_u8(ADDR_BAG_ITEMS + index * 2) == item_id:
            return index
    if count >= BAG_ITEM_CAPACITY:
        raise ValueError(f"the bag is full ({count} items); cannot add item {item_id}")
    write_u8(emulator, ADDR_BAG_ITEMS + count * 2, item_id)
    write_u8(emulator, ADDR_BAG_ITEMS + count * 2 + 1, quantity)
    write_u8(emulator, ADDR_BAG_ITEMS + (count + 1) * 2, 0xFF)
    write_u8(emulator, ADDR_BAG_COUNT, count + 1)
    return count


def open_doctored(
    rom_path: str | Path,
    save_path: str | Path,
    move: str,
    tmp_path: str | Path,
    **kwargs: Any,
) -> Tuple[Any, Doctored]:
    """Create an emulator, load the ROM, and hand back a doctored, driveable game.

    The caller owns the emulator and must ``close()`` it. Pass an existing
    emulator to ``give_field_move`` instead when one is already open -- loading
    the ROM is the slow part.
    """
    from pokemon_agent.emulator import PyBoyEmulator

    emulator = PyBoyEmulator()
    emulator.load(str(rom_path))
    try:
        record = give_field_move(emulator, save_path, move, tmp_path, **kwargs)
    except Exception:
        emulator.close()
        raise
    return emulator, record
