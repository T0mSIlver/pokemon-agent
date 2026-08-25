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
    ADDR_EVENT_FLAGS,
    ADDR_OAK_PARCEL,
    ADDR_POKEDEX_FLAG,
    ITEM_NAMES,
    PokemonRedReader,
)
from pokemon_agent.milestones import (
    ALL_EVENTS,
    DATA_PATH,
    EVENT_FLAG_BYTES,
    MILESTONES,
    MILESTONES_BY_ID,
    MilestoneTracker,
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
        else:  # pragma: no cover - the assert is the point
            pytest.fail(f"{milestone.id} has unknown kind {milestone.kind!r}")


def test_no_ladder_rung_uses_a_flag_the_game_can_clear():
    """A resettable flag makes the score go backwards.

    EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH looks like progress and is cleared again
    by VictoryRoad2F.asm; EVENT_IN_SAFARI_ZONE is only set while you stand inside.
    """
    for milestone in MILESTONES:
        assert milestone.source not in gen.RESETTABLE_EVENTS


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
    assert MILESTONES[-1].id == "EVENT_HALL_OF_FAME_DEX_RATING"
    assert MILESTONES[-2].id == "EVENT_BEAT_CHAMPION_RIVAL"


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
) -> MilestoneTracker:
    emulator = FakeEmulator()
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
