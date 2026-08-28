"""What services.json has to name, and what it is allowed to leave out.

The file used to hold thirteen nurses and nothing else, because the generator's
classifier only ever recognised `predef HealParty` written inside the text body
itself. Two whole shapes of script were invisible to it:

* a body that hands its work to a helper label -- `call z, CeladonGymReceiveTM21`
  in Erika's text, `call RedsHouse1FMomHealScript` in Mom's -- because the body
  split closes a body at the next label in column 0;
* every script that gives an item at all, because `call GiveItem` was not in the
  table of things a service does.

Between them that hid all five HMs, the Poke Flute, the Master Ball and Mom's
free heal, which is the only heal in the game before Viridian. This file pins the
rooms whose absence stops a run, and the invariant that every name in the file is
a map the rest of the harness can look up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent import gamedata

SERVICES_JSON = Path(gamedata.__file__).resolve().parent.parent / "data" / "game" / "services.json"


def entries(map_name: str) -> list[dict]:
    return list(gamedata.services(map_name))


def gift_items(map_name: str) -> list[str]:
    return [e["item"] for e in entries(map_name) if e["service"] == "gift"]


# ---------------------------------------------------------------------------
# The rooms that gate the main line
# ---------------------------------------------------------------------------

#: Room -> the item somebody in it hands over, for the gifts a run cannot finish
#: without. Sourced from pokered: each is a `call GiveItem` (or, for Oak's aide,
#: an `hOaksAideRewardItem` load) inside the named NPC's text body.
GATING_GIFTS = {
    # WardensHouse.asm: the Warden takes the Gold Teeth and gives HM04.
    # Strength is what moves the Victory Road boulders.
    "Warden's House": "HM04",
    # MrFujisHouse.asm: the only Poke Flute in the game, and the only way past
    # the two sleeping Snorlax.
    "Mr Fuji's House": "Poke Flute",
    # SilphCo11F.asm: the president's Master Ball, after Giovanni.
    "Silph Co 11F": "Master Ball",
    # Route2Gate.asm: Oak's aide gives HM05 once ten species are seen. Flash is
    # what makes Rock Tunnel navigable.
    "Route 2 Gate": "HM05",
    # SSAnneCaptainsRoom.asm: HM01 Cut, which opens the way out of Vermilion.
    "S.S. Anne Captain's Room": "HM01",
    # Route16FlyHouse.asm / SafariZoneSecretHouse.asm: the other two HMs.
    "Route 16 Fly House": "HM02",
    "Safari Zone Secret House": "HM03",
    # BillsHouse.asm: no S.S. Ticket, no S.S. Anne, no Cut.
    "Bill's House": "S.S. Ticket",
    # PokemonFanClub.asm and BikeShop.asm: the voucher and what it buys.
    "Pokemon Fan Club": "Bike Voucher",
    "Cerulean Bike Shop": "Bicycle",
}


@pytest.mark.parametrize("map_name,item", sorted(GATING_GIFTS.items()))
def test_the_room_that_gates_progress_names_who_hands_the_item_over(map_name, item):
    """Absence here is a run that walks into a locked door with the key nearby."""
    assert item in gift_items(map_name), f"{map_name} no longer offers {item}"


@pytest.mark.parametrize("map_name,item", sorted(GATING_GIFTS.items()))
def test_a_gating_gift_carries_the_tile_the_person_stands_on(map_name, item):
    """An item name with no tile is the same hunt the nurse used to be."""
    gift = next(e for e in entries(map_name) if e.get("item") == item)
    assert gift["service"] == "gift"
    assert isinstance(gift["at"], list) and len(gift["at"]) == 2
    assert all(isinstance(coordinate, int) for coordinate in gift["at"])
    assert gift["text"].startswith("TEXT_")


def test_all_five_hms_are_findable_by_the_room_that_gives_them():
    """Cut, Fly, Surf, Strength and Flash: five rooms, one HM each, no duplicates."""
    found = {
        item: name
        for name in gamedata.map_names()
        for item in gift_items(name)
        if item.startswith("HM")
    }
    assert found == {
        "HM01": "S.S. Anne Captain's Room",
        "HM02": "Route 16 Fly House",
        "HM03": "Safari Zone Secret House",
        "HM04": "Warden's House",
        "HM05": "Route 2 Gate",
    }


# ---------------------------------------------------------------------------
# The heal the generator used to miss
# ---------------------------------------------------------------------------


def test_mom_heals_in_reds_house_and_the_file_says_where_she_stands():
    """RedsHouse1F.asm: her text calls RedsHouse1FMomHealScript -> predef HealParty.

    Pallet Town has no Poke Center. Before Viridian's, she is the only heal in
    the game, and she is free.
    """
    mom = entries("Red's House 1F")
    assert [e["service"] for e in mom] == ["heal"]
    assert mom[0]["at"] == [5, 4]
    assert mom[0]["text"] == "TEXT_REDSHOUSE1F_MOM"


def test_the_eight_gym_leaders_each_hand_over_their_tm():
    """Each leader's TM sits behind a `call z, <Gym>ReceiveTM<nn>` in her text.

    All eight were missing for the same reason as Mom, which is why they are
    asserted together: one classifier bug, one fix, one guard.
    """
    leaders = {
        "Pewter Gym": "TM34",
        "Cerulean Gym": "TM11",
        "Vermilion Gym": "TM24",
        "Celadon Gym": "TM21",
        "Fuchsia Gym": "TM06",
        "Saffron Gym": "TM46",
        "Cinnabar Gym": "TM38",
        "Viridian Gym": "TM27",
    }
    assert {name: gift_items(name) for name in leaders} == {
        name: [tm] for name, tm in leaders.items()
    }


def test_the_celadon_roof_girl_carries_all_three_drinks_she_trades_for():
    """CeladonMartRoof.asm gives TM49, TM48 or TM13 depending on the drink.

    One entry would have been a confident wrong answer about the other two.
    """
    assert gift_items("Celadon Dept Store Roof") == ["TM49", "TM48", "TM13"]


# ---------------------------------------------------------------------------
# Every name in the file has to be a map
# ---------------------------------------------------------------------------


def test_every_map_named_in_services_json_exists_in_world_json():
    """A key the rest of the harness cannot look up is a fact nothing can read."""
    raw = json.loads(SERVICES_JSON.read_text())
    named = {key for key in raw if key != "generated_from"}
    assert named
    assert named <= set(gamedata.world())


def test_every_entry_carries_the_fields_a_reader_assumes():
    raw = json.loads(SERVICES_JSON.read_text())
    for name, rows in raw.items():
        if name == "generated_from":
            continue
        assert rows, name
        for row in rows:
            assert set(row) >= {"service", "text", "at"}, (name, row)
            assert row["service"] in ("heal", "gift"), (name, row)
            assert len(row["at"]) == 2, (name, row)
            if row["service"] == "gift":
                assert row["item"], (name, row)


def test_a_service_tile_is_inside_the_map_it_is_on():
    """A tile past the edge would send a walk into a wall it can never reach."""
    for name in gamedata.map_names():
        size = gamedata.world()[name]["size"]
        for row in entries(name):
            x, y = row["at"]
            assert 0 <= x < size[0] and 0 <= y < size[1], (name, row, size)


def test_the_file_was_generated_from_the_same_commit_as_the_world():
    """Hand-edited entries drift from the maps they name. Nothing here is hand-written."""
    assert gamedata.generated_from("services") == gamedata.generated_from("world")


# ---------------------------------------------------------------------------
# Rooms that are correctly silent
# ---------------------------------------------------------------------------


def test_a_room_that_offers_nothing_carries_nothing():
    """169 of the game's 223 maps. Padding the file would cost every read.

    Viridian School and the Pewter speech house hold two people each whose whole
    script is `text_far`; Diglett's Cave and the north-south Underground Path
    have no object_event at all.
    """
    for name in (
        "Viridian School",
        "Pewter Speech House",
        "Diglett's Cave",
        "Underground Path North South",
        "Route 4",
    ):
        assert gamedata.services(name) == [], name
