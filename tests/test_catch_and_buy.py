"""Guards for the two things the harness could not do: throw a ball, buy an item.

The measured failure these exist for is one 33-hour run. It finished with one
Pokemon, 7,198 unspent money, 40 whiteouts, and 501 of 790 battle commands spent
running away — while carrying a Poke Ball it never threw, through hundreds of
wild encounters, and having walked into a Poke Mart and out again.

Nothing in the harness was stopping it and nothing in the harness was telling
it. There was no verb for either, the bag is only reachable from the battle menu
by hand, and `a_until_dialog_end` is *refused* in a battle precisely because
mashing A there opens that bag by accident. No payload it read on a wild
encounter mentioned a ball, and no payload it read inside a mart mentioned a
price. So the tests here cover both halves: the arithmetic that makes the fact
worth printing, and the menu walks that make the verb work twice in a row.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pokemon_agent import capabilities, gamedata
from pokemon_agent.agent_cli import split_buy_tokens
from pokemon_agent.capabilities import BALLS, catch_chance
from pokemon_agent.memory.red import (
    ADDR_ENEMY_CATCH_RATE,
    BALL_ITEM_IDS,
    ITEM_NAMES,
    SPECIES_NAMES,
    PokemonRedReader,
)
from pokemon_agent.server import (
    BATTLE_OPEN_BAG,
    _catch_line,
    _face_toward,
    _shop_line,
    item_list_walk_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ITEM_NAMES_BY_NAME = {name: item_id for item_id, name in ITEM_NAMES.items()}


def wild(species="Oddish", hp=35, max_hp=35, catch_rate=255, status="OK", level=13):
    return {
        "in_battle": True,
        "type": "wild",
        "enemy": {
            "species": species,
            "level": level,
            "hp": hp,
            "max_hp": max_hp,
            "status": status,
            "types": ["Grass", "Poison"],
            "catch_rate": catch_rate,
        },
    }


def bag(**counts):
    return [{"id": 0, "item": name, "quantity": n} for name, n in counts.items()]


# ---------------------------------------------------------------------------
# The catch formula
# ---------------------------------------------------------------------------


def reference_catch_rate(ball, catch_rate, hp, max_hp, bonus):
    """ItemUseBall re-implemented by enumeration, as a second opinion on the closed form.

    Written straight off pokered's control flow — draw Rand1, subtract the
    status bonus, capture outright if that goes negative, fail if what is left
    exceeds the catch rate, then roll Rand2 against the HP term. It counts
    outcomes instead of reasoning about them, so if `catch_chance` has an
    off-by-one in a range or an interval this disagrees.
    """
    spec = BALLS[ball]
    rolls, factor = spec["roll"], spec["factor"]
    caught = 0.0
    for rand1 in range(rolls):
        if rand1 < bonus:
            caught += 1
            continue
        if rand1 - bonus > catch_rate:
            continue
        w = ((max_hp * 255) // factor) // max(hp // 4, 1)
        if w > 255:
            caught += 1
        else:
            caught += (w + 1) / 256
    return caught / rolls


@pytest.mark.parametrize("ball", ["Poke Ball", "Great Ball", "Ultra Ball"])
@pytest.mark.parametrize("catch_rate", [3, 45, 90, 190, 255])
@pytest.mark.parametrize("hp_fraction", [1.0, 0.5, 0.1])
def test_catch_chance_agrees_with_the_engine_enumerated(ball, catch_rate, hp_fraction):
    """The closed form and a brute-force count of the same asm give the same number.

    Exactness is the whole reason the number is worth printing. "Catch rate 255"
    does not separate a full-HP Pidgey, which is better than one throw in three,
    from a full-HP Kadabra, which is worse than one in twenty; a probability
    does, and a probability that is approximately right is a probability the
    agent cannot use to decide between throwing now and hitting it once more.
    """
    max_hp = 60
    hp = max(1, int(max_hp * hp_fraction))
    assert catch_chance(ball, catch_rate, hp, max_hp) == pytest.approx(
        reference_catch_rate(ball, catch_rate, hp, max_hp, 0)
    )


def test_status_is_worth_what_the_engine_subtracts():
    """Sleep and freeze subtract 25 from the first roll; the other ailments subtract 12.

    This is the only Gen 1 catching *strategy* there is, so getting it wrong
    would make the payload advise against the one thing that works.
    """
    for status, bonus in (("SLP(3)", 25), ("FRZ", 25), ("PAR", 12), ("BRN", 12), ("PSN", 12)):
        assert catch_chance("Poke Ball", 90, 40, 60, status) == pytest.approx(
            reference_catch_rate("Poke Ball", 90, 40, 60, bonus)
        )
    assert catch_chance("Poke Ball", 90, 40, 60, "OK") == pytest.approx(
        reference_catch_rate("Poke Ball", 90, 40, 60, 0)
    )
    assert catch_chance("Poke Ball", 90, 40, 60, "SLP(1)") > catch_chance(
        "Poke Ball", 90, 40, 60, "PAR"
    )


def test_wearing_a_pokemon_down_is_what_makes_a_ball_worth_throwing():
    """A third of its HP roughly doubles a middling catch rate.

    The payload prints both numbers side by side for exactly this reason: the
    choice on a wild encounter is not "throw or do not throw", it is "throw now
    or hit it once and then throw", and only the pair answers that.
    """
    full = catch_chance("Poke Ball", 190, 30, 30)
    worn = catch_chance("Poke Ball", 190, 10, 30)
    assert worn > full * 1.8
    # At 1 HP the HP term saturates and drops out entirely, and what is left is
    # the catch-rate gate alone: 191 of 256 first rolls. So wearing something
    # down has a ceiling, and it is the species' own rate — which is why a
    # Poke Ball at an Abra is a bad idea however low its HP goes.
    assert catch_chance("Poke Ball", 190, 1, 30) == pytest.approx(191 / 256)
    assert catch_chance("Poke Ball", 255, 1, 30) == 1.0


def test_a_master_ball_always_works_and_a_great_ball_beats_a_poke_ball():
    """The ladder the default-ball choice is built on, in the order it assumes."""
    assert catch_chance("Master Ball", 3, 100, 100) == 1.0
    target = (45, 40, 60)
    assert catch_chance("Great Ball", *target) > catch_chance("Poke Ball", *target)


def test_a_fainted_or_unreadable_target_is_zero_rather_than_a_divide_by_zero():
    """A battle frame read before the enemy struct is populated must not raise.

    Perception is allowed to say "I cannot see this"; it is never allowed to be
    the thing that fails a state read.
    """
    assert catch_chance("Poke Ball", 255, 0, 0) == 0.0
    assert catch_chance("Poke Ball", 255, 0, 40) == 0.0


def test_an_unknown_ball_is_refused_by_name():
    with pytest.raises(capabilities.NotFound) as caught:
        catch_chance("Lure Ball", 255, 10, 20)
    assert "Poke Ball" in str(caught.value)


# ---------------------------------------------------------------------------
# The payload facts
# ---------------------------------------------------------------------------


def test_a_wild_battle_frame_prices_every_ball_in_the_bag():
    """The odds arrive with the encounter, not when something thinks to ask.

    `poke calc` was called zero times across a 457-call session while the same
    run spent 501 battle commands fleeing, so a fact behind a verb is a fact
    this agent does not have. The ball count is in the line for the same
    reason: without it the next call is a `poke state` to find out whether
    there is anything to throw.
    """
    line = _catch_line({"battle": wild(), "bag": bag(**{"Poke Ball": 11, "Great Ball": 2})})
    assert "Poke Ball x11" in line
    assert "Great Ball x2" in line
    assert "%" in line and "worn down" in line
    assert "poke catch" in line


def test_an_empty_bag_says_what_the_money_buys_instead_of_that_it_is_empty():
    """ "No Poke Balls" was already true and already known. The join is the fact.

    The run had $7,198 and no balls for its whole second half and never put
    those two together, so the line prints the price against the purse rather
    than restating the bag.
    """
    line = _catch_line({"battle": wild(), "bag": [], "player": {"money": 7198}})
    assert "$7198" in line
    assert "35" in line  # 7198 // 200
    assert "Poke Mart" in line


def test_a_trainers_pokemon_gets_no_catch_line_at_all():
    """A ball thrown at a trainer's Pokemon bounces off and the turn is gone.

    Printing odds there would be advice to waste a turn, so the whole line goes
    rather than being softened.
    """
    fight = wild()
    fight["type"] = "trainer"
    assert _catch_line({"battle": fight, "bag": bag(**{"Poke Ball": 5})}) is None


def test_the_ball_price_in_that_line_comes_from_the_shop_table():
    """200 is a number in the game data, not a number typed into a message.

    It appears in a refusal and in a payload line, and the two would drift apart
    the first time either was edited by hand.
    """
    assert capabilities.cheapest_ball_price() == 200
    assert gamedata.shops("Viridian Mart")["prices"]["Poke Ball"] == 200


def test_a_battle_still_starting_gets_no_catch_line():
    """The enemy struct is not populated on the first frames of an encounter.

    A payload built on it there would print a catch rate of 0 as though it were
    a reading, which is worse than printing nothing.
    """
    starting = {"in_battle": True, "type": "wild", "enemy": {}}
    assert _catch_line({"battle": starting, "bag": bag(**{"Poke Ball": 5})}) is None


def test_a_mart_frame_carries_its_stock_its_prices_and_the_money():
    """Standing in a shop is not the same as knowing what is in it.

    The run walked into Vermilion Mart with $7,198 and left with $7,198. This
    line is what makes the second visit different from the first.
    """
    line = _shop_line({"map": {"map_name": "Vermilion Mart"}, "player": {"money": 7198}})
    assert "$7198" in line
    assert "Poke Ball 200" in line
    assert "Super Potion 700" in line
    assert "poke buy" in line


def test_a_map_that_sells_nothing_carries_no_shop_line():
    """211 of the game's 223 maps are not marts, and this costs nothing on them."""
    assert _shop_line({"map": {"map_name": "Route 6"}, "player": {"money": 500}}) is None
    assert _shop_line({"map": {}, "player": {}}) is None


def test_the_catch_line_is_short_enough_to_send_on_every_wild_frame():
    """Tool text is 98% of the model's prompt, so a per-frame line is a per-frame cost.

    Sixty-odd bytes against a ~95 kB median session is the price of the fact
    the whole run was missing; a paragraph would not be.
    """
    line = _catch_line({"battle": wild(), "bag": bag(**{"Poke Ball": 11})})
    assert len(line.encode()) < 80


# ---------------------------------------------------------------------------
# The menu walks
# ---------------------------------------------------------------------------


def test_opening_the_bag_reaches_item_and_never_run():
    """ITEM and RUN share a row. Landing on the wrong one flees the fight.

    Two Downs and two Lefts is the same corner walk `battle_run_keys` uses with
    the column pressed the other way, so it is correct from any of the four
    entries the cursor may have been left on.
    """
    assert BATTLE_OPEN_BAG.count("press_down") == 2
    assert BATTLE_OPEN_BAG.count("press_left") == 2
    assert "press_right" not in BATTLE_OPEN_BAG
    assert BATTLE_OPEN_BAG[-1] == "press_a"


def test_the_bag_walk_goes_back_up_when_the_cursor_opens_below_the_ball():
    """The bag remembers where it was left, exactly as the move list does.

    Measured: a second `poke catch` in the same fight opened the bag already on
    row 1, and a walk that assumed it opens on row 0 pressed Down once more and
    landed on row 2 — one entry past the Poke Ball. The list does not wrap, so
    Down is not a way round; the direction has to be the sign of the difference.
    """
    assert item_list_walk_keys(1, 0) == ["press_down"]
    assert item_list_walk_keys(1, 3) == ["press_up", "press_up"]
    assert item_list_walk_keys(2, 2) == []


def test_the_bag_walk_refuses_a_row_that_is_not_a_row():
    with pytest.raises(ValueError):
        item_list_walk_keys(-1, 0)


def test_the_two_ball_tables_name_the_same_balls_in_the_same_order():
    """`BALLS` prices a throw and `BALL_ITEM_IDS` picks which one to spend.

    They are read by different layers — the odds come from capabilities, the
    default ball from the reader's id list — and a disagreement between them
    would price one ball and throw another, silently.
    """
    assert tuple(spec["id"] for spec in BALLS.values()) == BALL_ITEM_IDS
    assert [ITEM_NAMES[item_id] for item_id in BALL_ITEM_IDS] == list(BALLS)
    assert BALL_ITEM_IDS[0] == ITEM_NAMES_BY_NAME["Poke Ball"]
    assert BALL_ITEM_IDS[-1] == ITEM_NAMES_BY_NAME["Master Ball"]


def test_facing_the_person_behind_the_counter_works_from_either_side():
    """Town marts sit their clerk to the west, the Celadon floors to the north.

    Not named for clerks any more: a Poke Center nurse is reached by the same
    geometry, so the mart's turn-and-talk is now what both verbs use.
    """
    assert _face_toward((0, 5), (2, 5)) == "left"
    assert _face_toward((5, 3), (5, 5)) == "up"
    assert _face_toward((7, 3), (5, 3)) == "right"
    assert _face_toward((5, 8), (5, 5)) == "down"


def test_a_trailing_number_is_the_count_not_part_of_the_item_name():
    """`poke buy poke ball 10` has to mean ten Poke Balls.

    Two argparse positionals cannot split this — `nargs="+"` swallows the
    number — and quoting the item is exactly the thing that lost 260 of 261
    failed tool calls to a dropped apostrophe on an earlier version of this CLI.
    """
    assert split_buy_tokens(["poke", "ball", "10"]) == ("poke ball", 10)
    assert split_buy_tokens(["potion"]) == ("potion", 1)
    assert split_buy_tokens(["TM32"]) == ("TM32", 1)
    assert split_buy_tokens(["super", "potion"]) == ("super potion", 1)


# ---------------------------------------------------------------------------
# The shop table
# ---------------------------------------------------------------------------


def test_every_mart_prices_everything_it_sells():
    """A stocked item with no price is a row the payload silently drops.

    Prices come from two tables in pokered — one per item id, one of nybbles
    for the TMs — and Celadon Mart 2F is the counter that needs both.
    """
    for name in _mart_names():
        shop = gamedata.shops(name)
        prices = shop.get("prices") or {}
        missing = [item for item in shop["items"] if item not in prices]
        assert not missing, f"{name} sells {missing} at no price"


def test_every_counter_names_the_tile_its_clerk_stands_on():
    """Without it, buying starts by finding the till among three identical NPCs.

    The join that produces it is by text id — `CeladonMart2FClerk1Text` against
    `TEXT_CELADONMART2F_CLERK1` — and the generator refuses to build a counter
    it cannot match, so an empty tile here means that join has rotted.
    """
    for name in _mart_names():
        for counter in gamedata.shops(name)["counters"]:
            x, y = counter["at"]
            assert 0 <= x < 64 and 0 <= y < 64, f"{name}: clerk at {counter['at']}"


def test_the_two_floors_with_two_counters_keep_their_stock_apart():
    """Celadon Mart 2F sells healing at one till and TMs at the next one along.

    Per-map stock is not fine-grained enough there, and a walk to the wrong
    clerk buys from the wrong list.
    """
    counters = gamedata.shops("Celadon Dept Store 2F")["counters"]
    assert len(counters) == 2
    first, second = (set(counter["items"]) for counter in counters)
    assert not first & second
    assert counters[0]["at"] != counters[1]["at"]


def _mart_names():
    return [name for name in gamedata.map_names() if gamedata.shops(name)]


# ---------------------------------------------------------------------------
# The catch-rate address, against real save states
# ---------------------------------------------------------------------------


def _find_saves_dir():
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()


@pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")
def test_the_catch_rate_address_reads_the_species_table_on_real_battles():
    """0xD007 holds the catch rate of the Pokemon actually on the field.

    This is the one address the catch odds cannot be computed without, and two
    earlier audits in this project found addresses that were wrong in ways
    nothing crashed on. It was derived twice over from addresses already trusted
    — wEnemyMon at 0xCFE5 plus a 29-byte battle_struct plus five base stats
    lands on it, and one byte further plus an 11-byte nickname lands exactly on
    wBattleMon at 0xD014 — and this is the second source: real encounters, where
    a wrong byte reads as a number no species has.
    """
    pyboy = pytest.importorskip("pyboy")  # noqa: F841
    from pokemon_agent.emulator import PyBoyEmulator

    states = sorted(SAVES_DIR.glob("*battle-entry*.state"))
    if not states:
        pytest.skip("no battle-entry save states to read")
    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        checked = 0
        for state in states[:: max(1, len(states) // 10)][:10]:
            emulator.load_state(str(state))
            emulator.tick(30)
            battle = reader.read_battle()
            enemy = battle.get("enemy") or {}
            species = enemy.get("species")
            if not battle.get("in_battle") or species not in SPECIES_NAMES.values():
                continue
            expected = (gamedata.species(species) or {}).get("catch_rate")
            if expected is None:
                continue
            assert enemy["catch_rate"] == expected, f"{state.name}: {species}"
            assert emulator.read_u8(ADDR_ENEMY_CATCH_RATE) == expected
            checked += 1
        assert checked >= 3, f"only {checked} readable battles among the sampled saves"
    finally:
        emulator.close()


def test_catch_chance_never_leaves_the_unit_interval_on_random_inputs():
    """A probability printed as "137%" is worse than no probability.

    Fuzzed rather than enumerated because the failure would be an interval that
    clips wrong at one end, and the ends are where catch rates actually live: 3
    for the legendaries, 255 for everything on Route 1.
    """
    rng = random.Random(20260827)
    for _ in range(2000):
        max_hp = rng.randint(1, 999)
        chance = catch_chance(
            rng.choice(list(BALLS)),
            rng.randint(0, 255),
            rng.randint(1, max_hp),
            max_hp,
            rng.choice(["OK", "SLP(2)", "PAR", "FRZ", "BRN", "PSN"]),
        )
        assert 0.0 <= chance <= 1.0
