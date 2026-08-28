"""Guards for the milestone oracle: the parsed event table and the curated ladder.

The event table is *not* positional the way MAP_NAMES is -- pokered's
event_constants.asm moves its counter with `const_next $XXX`, which is an absolute
jump. Count `const` lines instead and all 507 indices are wrong, which would poison
every progress metric downstream. These tests pin the arithmetic against indices
that can be checked a second way, the ladder against its own invariants, and the
whole stack against a real save state when the ROM is present.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest

from pokemon_agent.memory.red import (
    ADDR_BADGES,
    ADDR_BAG_COUNT,
    ADDR_BAG_ITEMS,
    ADDR_ELITE_4_FLAGS,
    ADDR_EVENT_FLAGS,
    ADDR_OAK_PARCEL,
    ADDR_PC_COUNT,
    ADDR_PC_ITEMS,
    ADDR_POKEDEX_FLAG,
    ADDR_TOWN_VISITED_FLAGS,
    ITEM_NAMES,
    PokemonRedReader,
)
from pokemon_agent.milestones import (
    ALL_EVENTS,
    DATA_PATH,
    EVENT_FLAG_BYTES,
    MILESTONE_DAG,
    MILESTONES,
    MILESTONES_BY_ID,
    RESETTABLE_EVENTS,
    MilestoneTracker,
    blocking,
    frontier,
    milestone_for_event,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    path = REPO_ROOT / "scripts" / "gen_milestones.py"
    spec = importlib.util.spec_from_file_location("gen_milestones", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


# ===================================================================
# The parser
# ===================================================================

# Exercises every directive that moves the counter. A naive line count would make
# these 0, 1, 2, 3, 4; the real indices are 0, 3, 5, 16, 30.
PARSER_FIXTURE = """\
; leading comment
	const_def
	const EVENT_A
	const_skip 2
	const EVENT_B
	const_skip
	const EVENT_C

; a section header jumps the counter to an absolute index
	const_next $10
	const EVENT_D
	const_next $20 - 2
	const EVENT_E ; trailing comment
DEF NUM_FIXTURE_EVENTS EQU const_value
"""


def test_const_next_is_an_absolute_jump_not_a_skip():
    events, final = gen.parse_event_constants(PARSER_FIXTURE)
    assert events == {
        "EVENT_A": 0,
        "EVENT_B": 3,
        "EVENT_C": 5,
        "EVENT_D": 0x10,
        "EVENT_E": 0x20 - 2,
    }
    assert final == 0x20 - 1


def test_expression_evaluator_handles_hex_and_subtraction():
    assert gen.eval_const_expr("$F0 - 2") == 0xEE
    assert gen.eval_const_expr("$28") == 0x28
    assert gen.eval_const_expr("16") == 16
    with pytest.raises(ValueError):
        gen.eval_const_expr("NUM_EVENTS")


def test_duplicate_event_names_are_rejected():
    with pytest.raises(ValueError):
        gen.parse_event_constants("\tconst_def\n\tconst EVENT_A\n\tconst EVENT_A\n")


# ===================================================================
# The generated table
# ===================================================================


def test_the_read_window_covers_every_event_bit():
    assert EVENT_FLAG_BYTES * 8 == gen.NUM_EVENTS


def test_every_named_event_is_present_exactly_once():
    assert len(ALL_EVENTS) == gen.EXPECTED_EVENT_COUNT == 507
    assert len(set(ALL_EVENTS.values())) == len(ALL_EVENTS)
    assert all(0 <= index < gen.NUM_EVENTS for index in ALL_EVENTS.values())


@pytest.mark.parametrize("name", gen.SELF_NAMING_EVENTS)
def test_self_naming_events_match_their_own_index(name):
    """EVENT_1B8 and friends spell their bit index in hex.

    Upstream left four bits unnamed, so the placeholder *is* the cross-check: drop
    or misread a single const_next and these stop landing on themselves.
    """
    assert ALL_EVENTS[name] == int(name.removeprefix("EVENT_"), 16)


def test_pokedex_and_parcel_bits_land_on_the_addresses_red_py_already_knew():
    """Second, independent source: two byte/bit pairs that predate this table.

    pokemon_agent/memory/red.py has carried ADDR_POKEDEX_FLAG = 0xD74B bit 5 and
    ADDR_OAK_PARCEL = 0xD74E bit 1 since long before the bitfield was enumerated,
    both read off a running game. If the parse drifted, these would miss.
    """
    for name, addr, mask in (
        ("EVENT_GOT_POKEDEX", ADDR_POKEDEX_FLAG, 0x20),
        ("EVENT_GOT_OAKS_PARCEL", ADDR_OAK_PARCEL, 0x02),
    ):
        index = ALL_EVENTS[name]
        assert ADDR_EVENT_FLAGS + index // 8 == addr
        assert 1 << (index % 8) == mask


def test_beat_brock_bit_matches_the_byte_read_off_the_emulator():
    # 0xD747 + 14 = 0xD755 read 0xC4 in all 444 save states that hold the Boulder
    # Badge, and 0xC4 has bit 7 set. See the module docstring of milestones.py.
    index = ALL_EVENTS["EVENT_BEAT_BROCK"]
    assert (index // 8, index % 8) == (14, 7)


def test_checked_in_json_matches_the_generator_ladder():
    document = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert document["source"] == gen.SOURCE_URL
    assert [(e["id"], e["label"], e["kind"], e["source"]) for e in document["ladder"]] == [
        tuple(entry) for entry in gen.LADDER
    ]


# ===================================================================
# The ladder
# ===================================================================


def test_ladder_size_is_in_the_curated_range():
    assert 50 <= len(MILESTONES) <= 70


def test_ladder_ids_are_unique_and_indices_are_dense_and_ordered():
    assert len({m.id for m in MILESTONES}) == len(MILESTONES)
    assert [m.ladder_index for m in MILESTONES] == list(range(len(MILESTONES)))


def test_every_ladder_source_resolves():
    for milestone in MILESTONES:
        if milestone.kind == "event":
            assert milestone.source in ALL_EVENTS
            assert milestone.id == milestone.source
        elif milestone.kind == "badge":
            assert 0 <= int(milestone.source.split(":")[1]) <= 7
        elif milestone.kind == "item":
            assert int(milestone.source.split(":")[1]) in ITEM_NAMES
        elif milestone.kind == "ram_bit":
            prefix, address, bit = milestone.source.split(":")
            assert prefix == "ram_bit"
            assert 0xC000 <= int(address, 16) <= 0xDFFF
            assert 0 <= int(bit) <= 15
        else:  # pragma: no cover - the assert is the point
            pytest.fail(f"{milestone.id} has unknown kind {milestone.kind!r}")


def test_no_ladder_rung_uses_a_flag_the_game_can_clear():
    """A resettable flag makes the score go backwards.

    EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH looks like progress and is cleared again
    by VictoryRoad2F.asm; EVENT_IN_SAFARI_ZONE is only set while you stand inside.
    """
    for milestone in MILESTONES:
        assert milestone.source not in RESETTABLE_EVENTS


def test_the_reset_set_is_the_one_the_decomp_actually_has():
    """Scanned, not typed. The typed one was eleven names short.

    Each of these is a shape the hand-written list got wrong: a range whose named
    endpoints do not mention the flag (the Elite Four rooms), a CheckAndReset that
    consumes the flag it tests (the Hall of Fame), and a plain two-name ResetEvents
    nobody thought to grep for (the Victory Road 2F switches).
    """
    assert len(RESETTABLE_EVENTS) >= gen.MIN_RESETTABLE_EVENTS
    assert set(gen.RESET_SENTINELS) <= RESETTABLE_EVENTS
    assert RESETTABLE_EVENTS <= set(ALL_EVENTS)


@pytest.mark.parametrize(
    "event_id",
    [
        "EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2",
        "EVENT_AUTOWALKED_INTO_LORELEIS_ROOM",
        "EVENT_BEAT_LORELEIS_ROOM_TRAINER_0",
        "EVENT_BEAT_BRUNOS_ROOM_TRAINER_0",
        "EVENT_BEAT_AGATHAS_ROOM_TRAINER_0",
        "EVENT_BEAT_LANCE",
        "EVENT_BEAT_CHAMPION_RIVAL",
        "EVENT_HALL_OF_FAME_DEX_RATING",
    ],
)
def test_the_eight_rungs_that_could_not_be_banked_are_off_the_ladder(event_id):
    """These were rungs 55..62 and the game clears every one of them.

    Route23.asm zeroes the boulder switches on every map load;
    IndigoPlateauLobby.asm zeroes the Elite Four block when a failed challenge
    sends you back to the lobby; HallOfFame.asm zeroes it again on the way in,
    and pokedex_rating.asm consumes the Hall of Fame flag in the same cutscene
    that sets it. Finishing the game moved the score down by eight and 63/63 was
    unreachable by any route.
    """
    assert event_id in RESETTABLE_EVENTS
    assert event_id not in MILESTONES_BY_ID


def test_reset_range_clears_whole_bytes_not_just_the_named_span():
    """Which is how a range that names Lorelei and Lance also wipes what is between.

    macros/scripts/events.asm masks the partial start byte, zeroes whole bytes in
    the middle, and masks the partial end byte -- so the reach is byte-aligned
    outwards from both endpoints, not the closed interval a reader would assume.
    """
    # Same byte: exactly the closed interval.
    assert gen.range_cleared_bits(9, 11) == {9, 10, 11}
    # Crossing a boundary: from bit 9 up to the end of byte 1, then bits 16..17.
    assert gen.range_cleared_bits(9, 17) == set(range(9, 18))
    # The end byte is cleared from its bit 0, which is below the named endpoint.
    assert 16 in gen.range_cleared_bits(15, 17)


def test_all_eight_badges_appear_in_bit_order():
    bits = [int(m.source.split(":")[1]) for m in MILESTONES if m.kind == "badge"]
    assert bits == list(range(8))


@pytest.mark.parametrize(
    ("leader_event", "badge_id"),
    [
        ("EVENT_BEAT_BROCK", "BADGE_BOULDER"),
        ("EVENT_BEAT_MISTY", "BADGE_CASCADE"),
        ("EVENT_BEAT_LT_SURGE", "BADGE_THUNDER"),
        ("EVENT_BEAT_ERIKA", "BADGE_RAINBOW"),
        ("EVENT_BEAT_KOGA", "BADGE_SOUL"),
        ("EVENT_BEAT_SABRINA", "BADGE_MARSH"),
        ("EVENT_BEAT_BLAINE", "BADGE_VOLCANO"),
        ("EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI", "BADGE_EARTH"),
    ],
)
def test_each_gym_leader_sits_just_before_their_badge(leader_event, badge_id):
    assert (
        MILESTONES_BY_ID[badge_id].ladder_index == MILESTONES_BY_ID[leader_event].ladder_index + 1
    )


def test_the_ladder_ends_at_the_hall_of_fame():
    """One bit, because one bit is all that survives the Hall of Fame.

    wElite4Flags bit 0 is set on the way in and read by nothing in the game,
    which is why it is still there afterwards when the six event flags that
    described the same achievement have all been zeroed.
    """
    assert MILESTONES[-1].id == "ELITE_FOUR_CHAMPION"
    address, bit = MILESTONES[-1].source.split(":")[1:]
    assert (int(address, 16), int(bit)) == (ADDR_ELITE_4_FLAGS, 0)
    assert MILESTONES[-2].id == "TOWN_INDIGO_PLATEAU"


def test_all_five_hms_are_on_the_ladder():
    ids = {m.id for m in MILESTONES}
    assert {f"EVENT_GOT_HM0{n}" for n in range(1, 6)} <= ids


def test_milestone_for_event_marks_off_ladder_events():
    on_ladder = milestone_for_event("EVENT_BEAT_BROCK")
    assert on_ladder.ladder_index == MILESTONES_BY_ID["EVENT_BEAT_BROCK"].ladder_index

    off_ladder = milestone_for_event("EVENT_BEAT_MEWTWO")
    assert off_ladder.ladder_index == -1
    assert off_ladder.kind == "event"

    with pytest.raises(KeyError):
        milestone_for_event("EVENT_NOT_A_REAL_FLAG")


# ===================================================================
# The tracker, against a synthetic RAM image
# ===================================================================


class FakeEmulator:
    """A flat 64K address space, so the *real* reader code runs against it."""

    def __init__(self) -> None:
        self.mem = bytearray(0x10000)

    def read_u8(self, addr: int) -> int:
        return self.mem[addr]

    def read_u16(self, addr: int) -> int:
        return self.mem[addr] | (self.mem[addr + 1] << 8)

    def read_range(self, addr: int, size: int) -> bytes:
        return bytes(self.mem[addr : addr + size])


def make_tracker(
    *,
    events: Iterable[str] = (),
    raw_event_bytes: Mapping[int, int] | None = None,
    badge_bits: Iterable[int] = (),
    items: Sequence[int] = (),
    pc_items: Sequence[int] = (),
    ram_bits: Iterable[str] = (),
) -> MilestoneTracker:
    emulator = FakeEmulator()
    emulator.mem[ADDR_PC_COUNT] = len(pc_items)
    for slot, item_id in enumerate(pc_items):
        emulator.mem[ADDR_PC_ITEMS + slot * 2] = item_id
        emulator.mem[ADDR_PC_ITEMS + slot * 2 + 1] = 1
    emulator.mem[ADDR_PC_ITEMS + len(pc_items) * 2] = 0xFF
    for source in ram_bits:
        _, address, bit = source.split(":")
        index = int(bit)
        emulator.mem[int(address, 16) + index // 8] |= 1 << (index % 8)
    for name in events:
        index = ALL_EVENTS[name]
        emulator.mem[ADDR_EVENT_FLAGS + index // 8] |= 1 << (index % 8)
    for offset, value in (raw_event_bytes or {}).items():
        emulator.mem[ADDR_EVENT_FLAGS + offset] |= value
    for bit in badge_bits:
        emulator.mem[ADDR_BADGES] |= 1 << bit
    emulator.mem[ADDR_BAG_COUNT] = len(items)
    for slot, item_id in enumerate(items):
        emulator.mem[ADDR_BAG_ITEMS + slot * 2] = item_id
        emulator.mem[ADDR_BAG_ITEMS + slot * 2 + 1] = 1
    emulator.mem[ADDR_BAG_ITEMS + len(items) * 2] = 0xFF
    return MilestoneTracker(PokemonRedReader(emulator))


def test_snapshot_reads_all_three_kinds():
    tracker = make_tracker(
        events=["EVENT_GOT_STARTER", "EVENT_BEAT_BROCK"],
        badge_bits=[0],
        items=[74],  # Lift Key
    )
    assert tracker.snapshot() == frozenset(
        {"EVENT_GOT_STARTER", "EVENT_BEAT_BROCK", "BADGE_BOULDER", "ITEM_LIFT_KEY"}
    )


def test_a_key_item_in_the_pc_still_counts():
    """Gen 1 has no key-item pocket, so the Lift Key can be deposited.

    A rung that only read the bag went out again the moment the player tidied
    up, which is the same non-monotonicity a resettable event flag has.
    """
    assert "ITEM_LIFT_KEY" in make_tracker(items=[74]).snapshot()
    assert "ITEM_LIFT_KEY" in make_tracker(pc_items=[74]).snapshot()
    assert "ITEM_LIFT_KEY" not in make_tracker(items=[20], pc_items=[20]).snapshot()


def test_a_ram_bit_milestone_reads_the_byte_its_source_names():
    champion = MILESTONES_BY_ID["ELITE_FOUR_CHAMPION"]
    assert "ELITE_FOUR_CHAMPION" not in make_tracker().snapshot()
    assert "ELITE_FOUR_CHAMPION" in make_tracker(ram_bits=[champion.source]).snapshot()

    # Neighbouring bits in the same byte must not read as the milestone:
    # wElite4Flags bit 1 is BIT_STARTED_ELITE_4, set while a challenge is running
    # and cleared again in the lobby.
    started = MilestoneTracker(PokemonRedReader(FakeEmulator()))
    started.reader.emu.mem[ADDR_ELITE_4_FLAGS] = 0b10
    assert "ELITE_FOUR_CHAMPION" not in started.snapshot()


def test_a_ram_bit_past_the_first_byte_spills_into_the_next_one():
    """Indigo Plateau is bit 9 of an 11-bit array: byte 1, bit 1, LSB first.

    Same order as wEventFlags. Reading it as bit 9 of a single byte, or MSB-first
    within the byte, both land on a different town.
    """
    indigo = MILESTONES_BY_ID["TOWN_INDIGO_PLATEAU"]
    assert indigo.source == f"ram_bit:0x{ADDR_TOWN_VISITED_FLAGS:04X}:9"

    tracker = MilestoneTracker(PokemonRedReader(FakeEmulator()))
    tracker.reader.emu.mem[ADDR_TOWN_VISITED_FLAGS] = 0xFF  # Pallet .. Cinnabar
    assert "TOWN_INDIGO_PLATEAU" not in tracker.snapshot()

    tracker.reader.emu.mem[ADDR_TOWN_VISITED_FLAGS + 1] = 0b10
    assert "TOWN_INDIGO_PLATEAU" in tracker.snapshot()


def test_event_bits_are_little_endian_within_their_byte():
    """Byte 14 of wEventFlags with only the high bit set means Brock, not event 112.

    Written as a literal byte rather than through the index arithmetic, so it fails
    if the reader ever flips to MSB-first.
    """
    tracker = make_tracker(raw_event_bytes={14: 0x80})
    assert "EVENT_BEAT_BROCK" in tracker.snapshot()

    reversed_byte = make_tracker(raw_event_bytes={14: 0x01})
    assert "EVENT_BEAT_BROCK" not in reversed_byte.snapshot()


def test_newly_set_from_empty_previous_returns_everything():
    tracker = make_tracker(events=["EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX"])
    assert tracker.newly_set(frozenset()) == (
        "EVENT_GOT_STARTER",
        "EVENT_GOT_POKEDEX",
    )


def test_newly_set_reports_only_the_difference_in_ladder_order():
    tracker = make_tracker(
        events=["EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK"],
        badge_bits=[0],
    )
    previous = frozenset({"EVENT_GOT_POKEDEX"})
    assert tracker.newly_set(previous) == (
        "EVENT_GOT_STARTER",
        "EVENT_BEAT_BROCK",
        "BADGE_BOULDER",
    )


def test_newly_set_is_empty_when_nothing_moved():
    tracker = make_tracker(events=["EVENT_GOT_STARTER"])
    assert tracker.newly_set(tracker.snapshot()) == ()


def test_newly_set_ignores_milestones_lost_since_previous():
    tracker = make_tracker(events=["EVENT_GOT_STARTER"])
    assert tracker.newly_set(frozenset({"EVENT_BEAT_BROCK"})) == ("EVENT_GOT_STARTER",)


def test_summary_on_a_blank_game():
    summary = make_tracker().summary()
    assert summary == {
        "count": 0,
        "total": len(MILESTONES),
        "furthest": None,
        "furthest_index": -1,
        "latest": [],
        # Nothing done yet leaves exactly one thing that can be done.
        "frontier": ["EVENT_GOT_STARTER"],
    }


def test_summary_reports_the_furthest_rung_not_the_last_one_set():
    tracker = make_tracker(events=["EVENT_BEAT_BROCK", "EVENT_GOT_STARTER"], badge_bits=[0])
    summary = tracker.summary()
    assert summary["count"] == 3
    assert summary["furthest"] == "BADGE_BOULDER"
    assert summary["furthest_index"] == MILESTONES_BY_ID["BADGE_BOULDER"].ladder_index
    assert summary["latest"] == ["EVENT_GOT_STARTER", "EVENT_BEAT_BROCK", "BADGE_BOULDER"]


def test_furthest_index_is_monotone_as_milestones_accumulate():
    reached = ["EVENT_GOT_STARTER"]
    previous = -1
    for name in ("EVENT_GOT_POKEDEX", "EVENT_BEAT_BROCK", "EVENT_BEAT_MISTY"):
        reached.append(name)
        index = make_tracker(events=reached).summary()["furthest_index"]
        assert index > previous
        previous = index


# ===================================================================
# The DAG and its frontier
#
# The ladder was a scoreboard read backwards. Read forwards it is a plan, and
# the reason for writing that down is a measured failure: one 457-call session
# spent 361 calls on `act` and 64 on `state`, called `route`, `frontier`, `calc`
# and `progress` zero times each, and banked none of the 63 milestones in
# fourteen hours. Twenty verbs over an open map is not a menu anyone orders
# from. These tests pin the graph that shortens the menu.
# ===================================================================


def reached_through(last_id: str, *, without: Iterable[str] = ()) -> set[str]:
    """Every ladder id up to and including *last_id*, minus *without*."""
    stop = MILESTONES_BY_ID[last_id].ladder_index
    return {m.id for m in MILESTONES if m.ladder_index <= stop} - set(without)


#: Where the fourteen-hour run actually sat: inside Mt. Moon, one badge, eight
#: rungs banked. Every frontier example below is anchored here so the numbers
#: mean something rather than being invented positions.
AT_MT_MOON = reached_through("BADGE_BOULDER", without=["EVENT_GOT_TOWN_MAP"])


def test_every_ladder_milestone_is_a_node_in_the_graph():
    assert set(MILESTONE_DAG) == {m.id for m in MILESTONES}
    assert all(MILESTONE_DAG[m.id].id == m.id for m in MILESTONES)


def test_no_edge_names_a_milestone_that_is_not_on_the_ladder():
    """A typo in a precondition would silently seal a milestone off forever.

    An unknown id can never appear in a RAM snapshot, so its dependant would
    sit permanently off the frontier with nothing to explain why.
    """
    for node in MILESTONE_DAG.values():
        for edge in node.requires + node.excludes:
            assert edge in MILESTONES_BY_ID, f"{node.id} points at unknown {edge}"


def test_preconditions_only_ever_point_backwards_along_the_ladder():
    """Acyclicity, proved by the cheapest available witness.

    The ladder is already a total order, so if every edge runs from a lower
    index to a higher one the graph cannot contain a cycle -- and ladder order
    is a valid topological order, which is what lets `frontier` be one pass.
    """
    for node in MILESTONE_DAG.values():
        here = MILESTONES_BY_ID[node.id].ladder_index
        for need in node.requires:
            assert MILESTONES_BY_ID[need].ladder_index < here, f"{node.id} <- {need}"


def test_the_graph_has_exactly_one_root():
    roots = [node.id for node in MILESTONE_DAG.values() if not node.requires]
    assert roots == ["EVENT_GOT_STARTER"]


def test_every_milestone_is_reachable_from_the_root():
    """No node is stranded behind a precondition set that can never all hold.

    Walking the graph the way the frontier does -- take everything open, mark
    it done, look again -- has to end with all 63, or some rung is unwinnable.
    Fossils are the one exception the graph knows about, so this walk takes
    both sides of that fork rather than either.
    """
    have: set[str] = set()
    while True:
        opened = [m.id for m in MILESTONES if m.id not in have and not blocking(m.id, have)]
        if not opened:
            break
        have.update(opened)
    assert have == {m.id for m in MILESTONES}


def test_effects_are_written_for_a_reader_not_for_a_parser():
    for node in MILESTONE_DAG.values():
        for effect in node.effects:
            assert effect == effect.strip() and effect
            assert "_" not in effect, f"{node.id} effect reads like an identifier"


# -------------------------------------------------------------------
# The frontier
# -------------------------------------------------------------------


def test_a_milestone_with_unmet_preconditions_is_not_on_the_frontier():
    """The whole point: 63 goals, and the game only permits a few of them.

    Misty is open from Mt. Moon -- the road there is walkable. The Cascade
    Badge is not, because it is downstream of beating her, and neither is
    anything behind Cut. Offering those is how a run spends fourteen hours
    walking toward something the game will not let it do.
    """
    open_ids = {m.id for m in frontier(AT_MT_MOON)}

    assert "EVENT_BEAT_MISTY" in open_ids
    assert "BADGE_CASCADE" not in open_ids
    assert blocking("BADGE_CASCADE", AT_MT_MOON) == ("EVENT_BEAT_MISTY",)
    assert "EVENT_BEAT_ERIKA" not in open_ids  # Celadon is behind the Cut trees
    assert "EVENT_HALL_OF_FAME_DEX_RATING" not in open_ids


def test_the_frontier_shrinks_as_milestones_complete():
    """Doing something that unlocks nothing must leave strictly less to do.

    The Old Amber is that milestone: nothing on the ladder requires it. So it
    is a clean measurement of the frontier as a set, with no unlocking to
    confuse the count.
    """
    before = frontier(AT_MT_MOON)
    after = frontier(AT_MT_MOON | {"EVENT_GOT_OLD_AMBER"})

    assert "EVENT_GOT_OLD_AMBER" in {m.id for m in before}
    assert len(after) == len(before) - 1
    assert {m.id for m in after} == {m.id for m in before} - {"EVENT_GOT_OLD_AMBER"}


def test_the_frontier_is_far_shorter_than_the_ladder_at_every_point():
    """63 rungs is the menu the model collapsed under. This is the bound.

    Walked greedily from a blank game to the Hall of Fame, the frontier never
    exceeds a dozen and mostly sits near five.
    """
    have: set[str] = set()
    sizes = []
    while True:
        open_now = frontier(have)
        if not open_now:
            break
        sizes.append(len(open_now))
        have.add(open_now[0].id)

    assert have == {m.id for m in MILESTONES} - {"EVENT_GOT_HELIX_FOSSIL"}
    assert max(sizes) <= 12
    assert sum(sizes) / len(sizes) < 6


def test_a_milestone_already_reached_is_never_offered_again():
    assert "EVENT_BEAT_BROCK" in AT_MT_MOON
    assert "EVENT_BEAT_BROCK" not in {m.id for m in frontier(AT_MT_MOON)}


def test_the_frontier_comes_back_in_ladder_order():
    indices = [m.ladder_index for m in frontier(AT_MT_MOON)]
    assert indices == sorted(indices)


def test_taking_one_fossil_takes_the_other_off_the_frontier_for_good():
    """Red's one irreversible fork, and the reason `excludes` exists.

    Without it the Helix would stay on the menu for the rest of the run,
    permanently unreachable and permanently advertised -- exactly the lure the
    frontier is meant to remove.
    """
    cleared = AT_MT_MOON | {"EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD"}
    both = {m.id for m in frontier(cleared)}
    assert {"EVENT_GOT_DOME_FOSSIL", "EVENT_GOT_HELIX_FOSSIL"} <= both

    took_dome = {m.id for m in frontier(cleared | {"EVENT_GOT_DOME_FOSSIL"})}
    assert "EVENT_GOT_HELIX_FOSSIL" not in took_dome


def test_a_field_move_needs_the_badge_as_well_as_the_hm():
    """Gen 1 checks the badge, not the move, and Surge's gym door has a tree.

    HM01 alone leaves Surge sealed; the join is what makes this a graph rather
    than a chain, and getting it wrong would offer a gym that cannot be entered.
    """
    have_hm_only = reached_through(
        "EVENT_SS_ANNE_LEFT", without=["EVENT_BEAT_MISTY", "BADGE_CASCADE"]
    )
    assert "EVENT_GOT_HM01" in have_hm_only and "BADGE_CASCADE" not in have_hm_only
    assert blocking("EVENT_BEAT_LT_SURGE", have_hm_only) == ("BADGE_CASCADE",)
    assert "EVENT_BEAT_LT_SURGE" not in {m.id for m in frontier(have_hm_only)}

    assert blocking("EVENT_BEAT_LT_SURGE", have_hm_only | {"BADGE_CASCADE"}) == ()


def test_victory_road_waits_for_all_eight_badges_and_for_strength():
    """Route 23 posts a guard per badge, and the boulders will not move alone.

    Marsh and Volcano are on their own branches -- neither is an ancestor of
    the Earth Badge -- so a chain that only looked at the last badge would let
    the model walk up Route 23 six badges short.
    """
    everything_but_blaine = reached_through(
        "EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE",
        without=["EVENT_BEAT_BLAINE", "BADGE_VOLCANO", "EVENT_GOT_HELIX_FOSSIL"],
    )
    assert blocking("EVENT_PASSED_EARTHBADGE_CHECK", everything_but_blaine) == ("BADGE_VOLCANO",)

    # Strength is not a Route 23 guard; it is the boulders inside Victory Road,
    # so it blocks the plateau rather than the gate.
    at_the_gate = reached_through("EVENT_PASSED_EARTHBADGE_CHECK", without=["EVENT_GOT_HM04"])
    assert blocking("TOWN_INDIGO_PLATEAU", at_the_gate) == ("EVENT_GOT_HM04",)


def test_off_ladder_events_in_a_snapshot_do_not_disturb_the_frontier():
    """`snapshot` only ever returns ladder ids, but callers hand in event sets.

    An unknown id must be ignored rather than raising: a stray flag is not a
    reason to stop answering the only question that shortens the menu.
    """
    noisy = AT_MT_MOON | {"EVENT_BEAT_MEWTWO", "EVENT_IN_SAFARI_ZONE"}
    assert frontier(noisy) == frontier(AT_MT_MOON)


def test_blocking_lists_only_what_is_still_outstanding():
    assert blocking("EVENT_BEAT_GHOST_MAROWAK", set()) == (
        "ITEM_SILPH_SCOPE",
        "EVENT_BEAT_POKEMON_TOWER_RIVAL",
    )
    assert blocking("EVENT_GOT_STARTER", set()) == ()


# -------------------------------------------------------------------
# The frontier is a memory read, not a claim
# -------------------------------------------------------------------


def test_the_tracker_reads_the_frontier_out_of_ram():
    """The loop must never grade its own homework.

    Nothing in the plan table asserts that a milestone happened; the frontier
    moves only because bits moved. Here the Boulder Badge bit and the Brock
    event bit are set in a synthetic address space and the frontier changes as
    a consequence of the read.
    """
    fresh = MilestoneTracker(make_tracker().reader)
    assert [m.id for m in fresh.frontier()] == ["EVENT_GOT_STARTER"]

    tracker = make_tracker(
        events=[
            "EVENT_GOT_STARTER",
            "EVENT_BATTLED_RIVAL_IN_OAKS_LAB",
            "EVENT_GOT_OAKS_PARCEL",
            "EVENT_OAK_GOT_PARCEL",
            "EVENT_GOT_POKEDEX",
            "EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE",
            "EVENT_BEAT_BROCK",
        ],
        badge_bits=[0],
    )
    assert tracker.summary()["count"] == 8
    assert tracker.summary()["frontier"] == [m.id for m in tracker.frontier()]
    assert {m.id for m in tracker.frontier()} == {m.id for m in frontier(AT_MT_MOON)}


def test_an_item_milestone_leaves_the_frontier_when_the_bag_says_so():
    """Item rungs are read from the bag, so the frontier follows the bag.

    The Lift Key is the only kind of postcondition here that a player can drop.
    """
    reached = reached_through("EVENT_FOUND_ROCKET_HIDEOUT", without=["EVENT_GOT_HELIX_FOSSIL"])
    without_key = make_tracker(events=[i for i in reached if i.startswith("EVENT_")])
    assert "ITEM_LIFT_KEY" not in without_key.snapshot()

    with_key = make_tracker(
        events=[i for i in reached if i.startswith("EVENT_")],
        items=[74],
    )
    assert "ITEM_LIFT_KEY" in with_key.snapshot()
    assert "ITEM_LIFT_KEY" not in {m.id for m in with_key.frontier()}


# ===================================================================
# End to end, against a real save state
# ===================================================================


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()

# Named states in saves/ that were swept and confirmed to hold the Boulder Badge.
# (beat_brock_boulder_badge.state is *not* one of them despite its name: it sits in
# the gym with the battle still ahead. Trust the RAM, not the filename.)
POST_BROCK_STATES = (
    "pewter_pre_route3.state",
    "route2_from_pewter.state",
    "pewter_final.state",
)


@pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")
def test_boulder_badge_state_reports_the_brock_milestones():
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator

    state = next(
        (SAVES_DIR / name for name in POST_BROCK_STATES if (SAVES_DIR / name).exists()), None
    )
    if state is None:
        pytest.skip("no post-Brock save state available")

    rom = str(SAVES_DIR / "PokemonRed.gb")
    emulator = create_emulator(rom)
    emulator.load(rom)
    try:
        emulator.load_state(str(state))
        tracker = MilestoneTracker(PokemonRedReader(emulator))
        snapshot = tracker.snapshot()
        summary = tracker.summary()
    finally:
        emulator.close()

    # The badge bit and the event bit are read from two unrelated addresses; both
    # firing on the same state is what confirms the little-endian bit order.
    assert "EVENT_BEAT_BROCK" in snapshot
    assert "BADGE_BOULDER" in snapshot
    assert {"EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX", "EVENT_GOT_OAKS_PARCEL"} <= snapshot
    assert "EVENT_BEAT_MISTY" not in snapshot
    assert "BADGE_CASCADE" not in snapshot
    assert summary["furthest"] == "BADGE_BOULDER"
    assert summary["total"] == len(MILESTONES)


class TestCountedRequirements:
    """Gates that are a number, which the milestone DAG cannot hold as an edge.

    Oak's aide wants ten species registered before he hands over HM05 Flash.
    That is a count, not a rung, so `MILESTONE_DAG` cannot express it — and
    "cannot be an edge" was being read downstream as "has no prerequisite".
    `poke progress` listed Flash as open to a live run holding three species,
    and Flash is the difference between Rock Tunnel and a dark maze, so the run
    kept walking back to Route 2 for it.
    """

    def test_flash_is_short_until_ten_species_are_registered(self):
        from pokemon_agent.milestones import counted_shortfall

        short = counted_shortfall("EVENT_GOT_HM05", {"pokedex_owned": 3})
        assert "10 species registered in the Pokedex" in short
        assert "you have 3" in short

    def test_nothing_is_owed_once_the_count_is_met(self):
        from pokemon_agent.milestones import counted_shortfall

        assert counted_shortfall("EVENT_GOT_HM05", {"pokedex_owned": 10}) == ""
        assert counted_shortfall("EVENT_GOT_HM05", {"pokedex_owned": 40}) == ""

    def test_a_rung_with_no_counted_gate_says_nothing(self):
        from pokemon_agent.milestones import counted_shortfall

        assert counted_shortfall("EVENT_BEAT_LT_SURGE", {"pokedex_owned": 3}) == ""

    def test_unreadable_flags_never_invent_a_requirement(self):
        """A refusal built on a bad read is worse than no refusal."""
        from pokemon_agent.milestones import counted_shortfall

        assert counted_shortfall("EVENT_GOT_HM05", None) == ""
        assert counted_shortfall("EVENT_GOT_HM05", "not a mapping") == ""

    def test_every_counted_requirement_names_a_real_milestone(self):
        from pokemon_agent.milestones import COUNTED_REQUIREMENTS, MILESTONES_BY_ID

        for milestone_id in COUNTED_REQUIREMENTS:
            assert milestone_id in MILESTONES_BY_ID

    def test_the_frontier_carries_the_shortfall(self):
        from pokemon_agent import capabilities

        payload = capabilities.progress_payload(
            {"count": 22, "total": 58, "frontier": ["EVENT_GOT_HM05", "EVENT_BEAT_ERIKA"]},
            1000,
            {"pokedex_owned": 3},
        )
        entries = {entry["id"]: entry for entry in payload["frontier"]}
        assert "10 species" in entries["EVENT_GOT_HM05"]["needs"]
        assert "needs" not in entries["EVENT_BEAT_ERIKA"], "only counted gates carry one"

    def test_the_rung_stays_on_the_frontier(self):
        """Hiding it would be worse: a rung it cannot see is one it cannot aim at."""
        from pokemon_agent import capabilities

        payload = capabilities.progress_payload(
            {"count": 22, "total": 58, "frontier": ["EVENT_GOT_HM05"]}, 1000, {"pokedex_owned": 3}
        )
        assert [entry["id"] for entry in payload["frontier"]] == ["EVENT_GOT_HM05"]
