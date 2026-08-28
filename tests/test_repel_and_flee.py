"""Guards for the two ways a wild battle can be not fought: a Repel, or a run.

The measured failure is run ``20260825T224823Z-983b``. Its receipt log is
append-only and was still being written while these were, so every count here is
a floor rather than a total. Over 43,000 receipts it spent over 2,500 ``battle``
commands and 25,900 presses; over 2,300 of those commands were on maps that have
a wild encounter table, and over 1,000 were inside Mt. Moon, whose entire table
is Lv6-11, against a lead that grew from 62 to 157 max HP and was never below it.
It used a Repel zero times. It could not have known to: `_shop_line` prints
"Repel 350" among the Potions on the twelve mart maps and no other payload in the
harness names the item, its price, its effect, or the ones in the bag.

It sent RUN over 1,000 times for over 7,100 presses, and over 370 of those
commands did not end the battle they were sent into. 331 of them went into a
*single* trainer battle on Route 6, at [1,15], at a standing 61/92 HP that never
once moved — because a trainer-battle refusal does not even cost the turn — for
2,337 presses, and then the fight was won with one Rage. No payload had ever said
running is impossible there, and none had ever priced a wild run either.

Every number below is checked against pokered rather than against that story:
``engine/battle/wild_encounters.asm`` for what a Repel suppresses,
``engine/items/item_effects.asm`` for how long, and
``engine/battle/core.asm``'s ``TryRunningFromBattle`` for what a run costs.
"""

from __future__ import annotations

import pytest

from pokemon_agent import gamedata
from pokemon_agent.capabilities import (
    REPELS,
    RUN_ATTEMPT_BONUS,
    NotFound,
    cheapest_repel,
    escape_chance,
    flee_line,
    repel_line,
    repel_payload,
)
from pokemon_agent.memory.red import ITEM_NAMES


def bag(**counts):
    """A bag, keyed by item name with underscores for the spaces."""
    return [
        {
            "id": REPELS.get(name.replace("_", " "), {}).get("id", 0),
            "item": name.replace("_", " "),
            "quantity": n,
        }
        for name, n in counts.items()
    ]


def wild(species="Zubat", speed=55):
    enemy = {"species": species, "stats": {"speed": speed}}
    return {"in_battle": True, "type": "wild", "enemy": enemy}


def me(speed=30):
    return {"stats": {"speed": speed}}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_every_repel_has_the_id_the_game_gives_it():
    """Priced by name and spent by id, exactly as HEALING_ITEMS is, so they must agree.

    A wrong id here would print "Repel" on the frame and then walk the bag
    cursor onto whatever item 30 happens to be.
    """
    for name, spec in REPELS.items():
        assert ITEM_NAMES[spec["id"]] == name


def test_repel_step_counts_are_the_ones_item_effects_asm_loads():
    """``ld b, 100`` / ``ld b, 200`` / ``ld b, 250`` into ``wRepelRemainingSteps``."""
    assert [spec["steps"] for spec in REPELS.values()] == [100, 200, 250]


def test_the_cheapest_repel_is_read_out_of_the_shop_table():
    name, price, steps = cheapest_repel()
    assert (name, steps) == ("Repel", 100)
    priced = {
        item: cost
        for map_name in gamedata.map_names()
        for item, cost in ((gamedata.shops(map_name) or {}).get("prices") or {}).items()
        if item in REPELS
    }
    assert price == min(priced.values()) == priced["Repel"]


# ---------------------------------------------------------------------------
# What a Repel suppresses
# ---------------------------------------------------------------------------


def test_repel_blocks_strictly_below_the_lead_not_at_or_above_it():
    """The pokered rule, which is the one people get wrong.

    ``cp b`` against ``wPartyMon1Level`` then ``jr c`` -- carry only when the
    wild level is *less than* the lead's. A slot exactly at the lead's level
    still appears, and the Repel keeps burning steps while it does.
    """
    # Route 1's table is Lv2-5 and nothing else.
    assert repel_payload("Route 1", 6)["share"] == pytest.approx(1.0, abs=1e-3)
    assert repel_payload("Route 1", 5)["share"] < 1.0  # the Lv5 slot is not below 5
    assert repel_payload("Route 1", 2)["share"] == 0.0  # nothing is below the floor


def test_the_share_is_the_encounter_tables_own_arithmetic():
    """Recomputed from encounters.json rather than typed out, at every level in it.

    `repel_payload` takes a level and not a party on purpose: the engine reads
    ``wPartyMon1Level``, so the caller has to choose slot 1 deliberately and a
    party argument would invite it to hand over the battler instead.
    """
    for map_name in ("Route 1", "Mt Moon 1F", "Diglett's Cave", "Route 4"):
        slots = gamedata.encounters(map_name)["grass"]["slots"]
        total = sum(s["chance"] for s in slots)
        for level in range(1, 40):
            want = sum(s["chance"] for s in slots if s["level"] < level) / total
            assert repel_payload(map_name, level)["share"] == pytest.approx(want)


def test_maps_with_no_wild_table_are_a_refusal_not_a_zero():
    with pytest.raises(NotFound):
        repel_payload("Pewter City", 46)
    with pytest.raises(NotFound):
        repel_payload(None, 46)


def test_the_rate_comes_from_the_encounter_table():
    for map_name in ("Mt Moon 1F", "Route 3", "Diglett's Cave"):
        payload = repel_payload(map_name, 99)
        assert payload["rate"] == gamedata.encounters(map_name)["grass"]["rate"]


# ---------------------------------------------------------------------------
# The line
# ---------------------------------------------------------------------------


def test_the_line_names_the_repel_in_the_bag_and_what_it_stops():
    line = repel_line("Mt Moon 1F", 46, bag(Repel=3))
    assert line is not None
    assert "Repel x3" in line
    assert "100 steps" in line
    assert "L6-11" in line  # Mt Moon 1F's whole table
    assert "L46" in line


def test_the_line_is_silent_where_a_repel_would_do_nothing():
    """Three silences, each for a different reason, and each one load-bearing.

    A payload line that fires on every frame of every map is the thing this
    project deletes, and a Repel that stops nothing is worse than absent: it is
    350 dollars and a menu spent on a fact that was not true.
    """
    assert repel_line("Pewter City", 46, bag(Repel=3)) is None  # no wild table at all
    assert repel_line("Mt Moon 1F", 5, bag(Repel=3)) is None  # every slot at or above
    assert repel_line("Mt Moon 1F", 46, [], money=60) is None  # none held, none affordable


def test_without_one_the_line_is_the_price_beside_the_money():
    line = repel_line("Mt Moon 1F", 46, [], money=7198)
    assert line is not None
    assert "$350" in line and "Poke Mart" in line
    assert "poke act" not in line  # nothing to open the bag for yet


def test_a_partial_table_says_what_still_gets_through():
    """Diglett's Cave runs Lv15-31, so a Lv20 lead buys most of it and not all."""
    line = repel_line("Diglett's Cave", 20, bag(Repel=1))
    assert line is not None
    assert "%" in line and "still comes" in line
    assert "every wild" not in line


def test_a_bag_entry_is_matched_by_id_as_well_as_by_name():
    """`read_bag` names an unknown id ``???(30)``, and the id is still the item.

    The same rule the ball and heal resolvers follow, for the same reason: the
    line is written from the name and the bag cursor is walked by the id.
    """
    unnamed = [{"id": 57, "item": "???(57)", "quantity": 2}]
    line = repel_line("Mt Moon 1F", 46, unnamed)
    assert line is not None and "Max Repel x2" in line and "250 steps" in line


def test_the_line_stays_inside_the_payload_budget():
    """It competes with `catch`, `items` and `moveset` for the same frame.

    120 characters is roughly what the catch line pays for at its worst, and a
    Repel line that ran longer would be buying its space from the ones that
    already move behaviour.
    """
    worst = ""
    for map_name in gamedata.map_names():
        if not (gamedata.encounters(map_name) or {}).get("grass"):
            continue
        for level in range(1, 60):
            for held in (bag(Max_Repel=12), bag(Repel=3), []):
                line = repel_line(map_name, level, held, money=999999) or ""
                worst = max(worst, line, key=len)
    assert 0 < len(worst) <= 160, (len(worst), worst)
    assert len(flee_line(wild(speed=108), me(speed=55))) <= 120


def test_the_expected_count_is_a_ceiling_from_rate_and_share():
    """100 steps at Mt. Moon's 10/256 is 3.9 rolls, and the line may not round up past it."""
    line = repel_line("Mt Moon 1F", 46, bag(Repel=1))
    assert "up to 4 battles" in line
    # Route 3 is 20/256 over the same 100 steps: twice the ceiling.
    assert "up to 8 battles" in repel_line("Route 3", 46, bag(Repel=1))


# ---------------------------------------------------------------------------
# What running costs
# ---------------------------------------------------------------------------


def test_a_tie_on_speed_escapes():
    """``StringCmp`` then ``jr nc``: not-less-than, so equal speeds always escape.

    The decomp's own comment says "greater than" and the code does not. This is
    the assertion that would catch someone rewriting it from the comment.
    """
    assert escape_chance(55, 55) == 1.0
    assert escape_chance(56, 55) == 1.0
    assert escape_chance(54, 55) < 1.0


def test_the_quotient_is_the_engines_arithmetic():
    """(player * 32) // ((enemy >> 2) & 0xFF), then (q + 1) / 256."""
    # 55 * 32 = 1760; 108 >> 2 = 27; 1760 // 27 = 65; (65 + 1) / 256.
    assert escape_chance(55, 108) == pytest.approx(66 / 256)


def test_each_failed_attempt_adds_thirty_out_of_256():
    assert RUN_ATTEMPT_BONUS == 30
    first = escape_chance(55, 108)
    assert escape_chance(55, 108, 2) == pytest.approx(first + 30 / 256)
    assert escape_chance(55, 108, 3) == pytest.approx(first + 60 / 256)
    # And it escapes outright once the addition carries past 255.
    assert escape_chance(55, 108, 8) == 1.0


def test_an_enemy_speed_under_four_escapes_outright():
    """``(enemy >> 2) & 0xFF`` of 0 jumps straight to .canEscape rather than dividing."""
    assert escape_chance(1, 3) == 1.0


def test_an_unreadable_speed_is_none_not_a_number():
    assert escape_chance(None, 55) is None
    assert escape_chance(55, None) is None
    assert escape_chance(0, 55) is None


def test_the_flee_line_prices_a_wild_run_and_names_the_turn_it_costs():
    line = flee_line(wild(speed=108), me(speed=55))
    assert line is not None
    assert "26%" in line  # 66/256
    assert "38%" in line  # the next try, +30/256
    assert "Zubat still attacks" in line


def test_a_certain_escape_does_not_pretend_to_be_a_gamble():
    line = flee_line(wild(speed=55), me(speed=108))
    assert line is not None
    assert "always works" in line
    assert "%" not in line


def test_a_trainer_battle_says_running_is_impossible():
    """The 331-command failure. ``TryRunningFromBattle`` branches to NoRunningText
    before it touches ``wNumRunAttempts``: no roll, no counter, and no turn spent
    either, so the fight does not advance and the loop has nothing to end it."""
    line = flee_line({"in_battle": True, "type": "trainer", "enemy": {}}, me())
    assert line is not None
    assert "refused" in line and "trainer" in line
    assert "%" not in line  # there are no odds to print


def test_the_flee_line_is_absent_off_a_battle_frame_and_on_unreadable_speeds():
    assert flee_line({"in_battle": False}, me()) is None
    assert flee_line({}, me()) is None
    assert flee_line(wild(), {}) is None  # no speed read: no number rather than a wrong one
