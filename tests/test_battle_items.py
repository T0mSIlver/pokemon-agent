"""Guards for the third thing a battle turn can be spent on: an item.

The measured failure these exist for is one 40-hour run. Across all 1,555
`battle` receipts it wrote, the only two intents that ever appear are
``{"run": true}`` (847) and ``{"fight": ...}`` (708). Zero items used, ever, and
party size never left 1 -- while the bag held ten Potions and five Poke Balls,
bought with money the run had gone shopping twice for. It beat Misty on 5/95
having taken 95 -> 77 -> 45 -> 5, with seven Potions in the bag the whole time.

Nothing was refusing it. There was no verb for using an item at all -- the
endpoints are `fight`, `run` and `catch` -- and no payload it ever read on a
battle frame mentioned the bag: `_observation_summary` carries `hp`, and drops
even the `heal` line inside a battle. A model cannot spend a turn on a fact it
is not given through a verb that does not exist.

So the tests here cover the fact: what each carried item would restore, the HP
it would leave you on, and the money-and-price line for a bag that has none.
"""

from __future__ import annotations

import pytest

from pokemon_agent import capabilities, gamedata
from pokemon_agent.capabilities import (
    HEALING_ITEMS,
    MAX_HEAL_ROWS,
    Conflict,
    cheapest_heal,
    heal_item_line,
    heal_item_payload,
)
from pokemon_agent.memory.red import ITEM_NAMES


def mon(hp=5, max_hp=95, species="Charmeleon"):
    return {"species": species, "level": 33, "hp": hp, "max_hp": max_hp}


def bag(**counts):
    """A bag, keyed by item name with underscores for the spaces."""
    return [{"id": 0, "item": name.replace("_", " "), "quantity": n} for name, n in counts.items()]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_every_healing_item_has_the_id_the_game_gives_it():
    """The table is read twice — priced by name, spent by id — so they must agree.

    `_resolve_ball_sync` matches the bag on ids and the item verb has to do the
    same. A name in this table with the wrong id beside it would price a Potion
    on the frame and then walk the bag cursor onto whatever item 21 happens to
    be, which is how a turn gets spent on an Escape Rope.
    """
    for name, spec in HEALING_ITEMS.items():
        assert ITEM_NAMES[spec["id"]] == name


def test_the_table_runs_weakest_first():
    """The order is the choice: a bare `poke item` spends the first row that fits.

    Same ladder as the balls. Out of order, "no item named" would reach for a
    Full Restore to put twenty points back.
    """
    restores = [
        999999 if spec["restores"] is None else spec["restores"] for spec in HEALING_ITEMS.values()
    ]
    assert restores == sorted(restores)


def test_a_full_heal_is_the_absence_of_a_number_not_a_big_one():
    """``None`` restores means "to the maximum", whatever the maximum is.

    Written as 999 it would price a Max Potion on a Chansey the same as on a
    Caterpie, and the resulting-HP column is the whole point of the line.
    """
    assert HEALING_ITEMS["Max Potion"]["restores"] is None
    assert HEALING_ITEMS["Full Restore"]["restores"] is None
    row = heal_item_payload(mon(hp=5, max_hp=95), bag(Max_Potion=1))["items"][0]
    assert (row["to"], row["wasted"]) == (95, 0)


# ---------------------------------------------------------------------------
# What the bag would do
# ---------------------------------------------------------------------------


def test_each_carried_item_is_priced_as_the_hp_it_leaves_you_on():
    """The subtraction, done. "Potion +20" beside "5/95" is the one nobody made."""
    rows = heal_item_payload(mon(hp=5, max_hp=95), bag(Potion=7, Super_Potion=2))["items"]
    assert [(r["item"], r["held"], r["to"]) for r in rows] == [
        ("Potion", 7, 25),
        ("Super Potion", 2, 55),
    ]


def test_healing_never_overshoots_the_maximum_and_says_what_it_throws_away():
    """A Hyper Potion into a six-point gap heals six and wastes 194."""
    row = heal_item_payload(mon(hp=89, max_hp=95), bag(Hyper_Potion=1))["items"][0]
    assert row["to"] == 95
    assert row["wasted"] == 194


def test_the_bag_is_read_for_healing_items_and_nothing_else():
    """The run carried Poke Ball x11, Potion x10, Repel x5. Two of those are noise here."""
    payload = heal_item_payload(mon(), bag(Poke_Ball=11, Potion=10, Repel=5))
    assert [r["item"] for r in payload["items"]] == ["Potion"]


def test_a_bag_with_no_potions_prices_nothing_rather_than_failing():
    payload = heal_item_payload(mon(), bag(Poke_Ball=11, Repel=5))
    assert payload["items"] == []
    assert payload["missing"] == 90


def test_a_fainted_pokemon_is_refused_rather_than_priced():
    """Zero is not "very hurt". A Potion does nothing to it and the turn is still spent."""
    with pytest.raises(Conflict):
        heal_item_payload(mon(hp=0), bag(Potion=7))
    assert heal_item_line(mon(hp=0), bag(Potion=7), money=1437) is None


def test_a_pokemon_the_reader_cannot_see_is_refused_rather_than_divided_by():
    with pytest.raises(Conflict):
        heal_item_payload({"species": "Charmeleon"}, bag(Potion=7))
    assert heal_item_line({}, bag(Potion=7), money=1437) is None


# ---------------------------------------------------------------------------
# The line the frame carries
# ---------------------------------------------------------------------------


def test_the_misty_frame_verbatim():
    """The fight this was written for: 5/95, seven Potions, and nothing said so.

    Pinned as a whole string rather than as its parts, because the parts were
    all individually present in the old run — the HP was on the frame, the bag
    was in `poke state` — and the line joining them was what was missing.
    """
    assert (
        heal_item_line(mon(hp=5, max_hp=95), bag(Potion=7), money=1437)
        == "Potion x7 +20 -> 25/95 — poke item potion"
    )


def test_the_line_ends_in_a_command_that_can_be_run_as_printed():
    """It names the weakest item carried, which is the one a bare call picks anyway."""
    line = heal_item_line(mon(), bag(Super_Potion=3, Potion=1), money=0)
    assert line.endswith("— poke item potion")
    assert heal_item_line(mon(), bag(Super_Potion=3), money=0).endswith("— poke item super potion")


def test_a_pokemon_at_full_hp_gets_no_line_at_all():
    """Nearly every frame is this one, and on all of them the line is wallpaper."""
    assert heal_item_line(mon(hp=95, max_hp=95), bag(Potion=7), money=1437) is None


def test_an_empty_bag_says_what_the_money_buys_instead_of_that_it_is_empty():
    """ "No Potions" was already known. The money beside the price never was.

    Same bargain as the catch line's empty half: the run walked past marts with
    thousands unspent, and no frame it read on a hurt party put the two numbers
    on the same line.
    """
    line = heal_item_line(mon(), bag(Poke_Ball=11), money=1437)
    assert line == "no healing items: $1437 buys Potion x4 at a Poke Mart"


def test_the_price_in_that_line_comes_from_the_shop_table():
    """Read out of the game data, not written down as 300, so a regen keeps it true."""
    item, price = cheapest_heal()
    on_sale = [
        p
        for name in gamedata.map_names()
        for i, p in ((gamedata.shops(name) or {}).get("prices") or {}).items()
        if i in HEALING_ITEMS and p
    ]
    assert price == min(on_sale)
    assert item in HEALING_ITEMS


def test_the_line_is_capped_so_a_full_bag_cannot_grow_it_without_bound():
    """Eight kinds of potion is a legal bag and would be a 200-byte per-frame line."""
    line = heal_item_line(
        mon(),
        bag(Potion=1, Super_Potion=1, Hyper_Potion=1, Max_Potion=1, Full_Restore=1),
        money=0,
    )
    assert line.count(";") == MAX_HEAL_ROWS - 1
    assert "Max Potion" not in line


def test_the_line_is_short_enough_to_send_on_every_hurt_frame():
    """Tool text is most of the model's prompt, so a per-frame line is a per-frame cost.

    Held to the catch line's budget, which is what the same shape of fact costs
    there. The three-row cap above is what keeps it inside this.
    """
    worst = heal_item_line(
        mon(hp=5, max_hp=999), bag(Potion=99, Super_Potion=99, Hyper_Potion=99), money=999999
    )
    assert len(worst.encode()) < 120


def test_the_line_never_mentions_an_item_that_is_not_in_the_bag():
    """The failure mode of a table-driven line: printing the table, not the bag."""
    line = heal_item_line(mon(), bag(Potion=2), money=0)
    for name in HEALING_ITEMS:
        assert (name in line) == (name == "Potion")


def test_the_capabilities_surface_names_the_new_functions():
    """`__all__` is the contract the server imports through."""
    for name in ("HEALING_ITEMS", "cheapest_heal", "heal_item_line", "heal_item_payload"):
        assert name in capabilities.__all__
