"""Guards for the generated game database in pokemon_agent/data/game/.

The data is produced by scripts/gen_gamedata.py from pret/pokered. Two things
can go wrong silently and both are pinned here:

* a map key that is not a name in MAP_NAMES -- pokered calls Pallet Town
  "PalletTown" and "PALLET_TOWN", neither of which the agent's memory reader
  ever produces, so a stray label makes a map's data unreachable forever;
* a mis-parsed table that still looks like plausible JSON. The spot checks
  below (Brock's team, Route 1's two species, three type matchups, one
  hand-computed damage roll) are cheap and catch exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokemon_agent import gamedata
from pokemon_agent.memory.red import MAP_NAMES

DATA_DIR = Path(gamedata.DATA_DIR)

#: Files whose top-level keys are map names.
MAP_KEYED_FILES = ("trainers", "encounters", "items", "shops")
ALL_FILES = ("world", *MAP_KEYED_FILES, "species", "moves", "types")

MAP_NAME_SET = set(MAP_NAMES.values())


def _raw(name: str) -> dict:
    return json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))


# ===================================================================
# Provenance
# ===================================================================


@pytest.mark.parametrize("name", ALL_FILES)
def test_every_file_records_the_upstream_commit(name):
    sha = _raw(name)["generated_from"]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_all_files_come_from_the_same_commit():
    assert len({_raw(name)["generated_from"] for name in ALL_FILES}) == 1


# ===================================================================
# Map names -- the invariant everything else depends on
# ===================================================================


@pytest.mark.parametrize("name", MAP_KEYED_FILES)
def test_map_keyed_files_only_use_real_map_names(name):
    keys = set(_raw(name)) - {"generated_from"}
    assert keys, f"{name}.json is empty"
    assert not keys - MAP_NAME_SET, f"{name}.json has keys that are not in MAP_NAMES"


def test_world_keys_and_map_ids_match_map_names():
    for map_name, entry in gamedata.world().items():
        assert map_name in MAP_NAME_SET
        assert MAP_NAMES[entry["map_id"]] == map_name


def test_connections_and_warps_name_real_maps():
    for map_name, entry in gamedata.world().items():
        for direction, neighbour in entry["connections"].items():
            assert direction in ("north", "south", "east", "west")
            assert neighbour in MAP_NAME_SET, f"{map_name} connects to {neighbour!r}"
        for warp in entry["warps"]:
            if warp["to_map"] is not None:
                assert warp["to_map"] in MAP_NAME_SET, f"{map_name} warps to {warp['to_map']!r}"


def test_warp_indices_resolve_into_the_destination_map():
    world = gamedata.world()
    for map_name, entry in world.items():
        for warp in entry["warps"]:
            if warp["to_map"] is None:
                continue
            if warp["to_map"] not in world:
                # The Silph Co elevator's warps point at an unused map id; the
                # elevator script rewrites them when you pick a floor.
                assert warp["to_map"].startswith("Unused Map "), (
                    f"{map_name} warps to {warp['to_map']!r}, which has no map data"
                )
                continue
            destination = world[warp["to_map"]]
            assert 0 <= warp["to_warp"] < len(destination["warps"]), (
                f"{map_name} -> {warp['to_map']} warp {warp['to_warp']} does not exist"
            )


def test_only_a_few_doors_are_left_unresolved():
    """LAST_MAP doors are resolved by reverse lookup; a handful (Victory Road
    2F and friends, reachable from the floor above and below) genuinely cannot
    be. If this number grows, the reverse lookup broke."""
    unresolved = sum(
        1
        for entry in gamedata.world().values()
        for warp in entry["warps"]
        if warp["to_map"] is None
    )
    assert unresolved <= 8


# ===================================================================
# The world graph
# ===================================================================


def _reachable_from(start: str) -> set:
    world = gamedata.world()
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        entry = world.get(current)
        if entry is None:
            continue
        neighbours = list(entry["connections"].values())
        neighbours += [w["to_map"] for w in entry["warps"] if w["to_map"]]
        for neighbour in neighbours:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def test_the_whole_journey_is_reachable_from_pallet_town():
    reachable = _reachable_from("Pallet Town")
    for destination in (
        "Viridian City",
        "Pewter City",
        "Cerulean City",
        "Celadon City",
        "Fuchsia City",
        "Cinnabar Island",
        "Saffron City",
        "Indigo Plateau",
        "Victory Road 1F",
    ):
        assert destination in reachable, f"{destination} is unreachable from Pallet Town"


def test_pallet_town_is_the_shape_of_pallet_town():
    pallet = gamedata.world()["Pallet Town"]
    assert pallet["map_id"] == 0
    # 10x9 blocks in map_constants.asm, and a block is 2x2 tiles.
    assert pallet["size"] == [20, 18]
    assert pallet["size_blocks"] == [10, 9]
    assert pallet["connections"] == {"north": "Route 1", "south": "Route 21"}
    assert {"x": 5, "y": 5, "to_map": "Red's House 1F", "to_warp": 0} in pallet["warps"]


# ===================================================================
# Trainers, encounters, items, shops
# ===================================================================


def test_brocks_team():
    brock = [t for t in gamedata.trainers("Pewter Gym") if t["trainer_class"] == "Brock"]
    assert len(brock) == 1
    assert brock[0]["team"] == [
        {"species": "Geodude", "level": 12},
        {"species": "Onix", "level": 14},
    ]


def test_trainers_with_no_map_argument_returns_the_whole_table():
    table = gamedata.trainers()
    assert "Pewter Gym" in table
    assert gamedata.trainers("Pallet Town") == []


def test_route_1_only_has_pidgey_and_rattata():
    route_1 = gamedata.encounters("Route 1")
    assert {slot["species"] for slot in route_1["grass"]["slots"]} == {"Pidgey", "Rattata"}
    assert route_1["water"] is None
    assert route_1["grass"]["rate"] == 25
    assert len(route_1["grass"]["slots"]) == 10
    assert sum(slot["chance"] for slot in route_1["grass"]["slots"]) == pytest.approx(1.0, abs=1e-3)


def test_maps_without_wild_pokemon_have_no_encounters():
    assert gamedata.encounters("Pallet Town") is None


def test_viridian_forest_items_include_the_hidden_ones():
    forest = gamedata.items("Viridian Forest")
    assert {"x": 25, "y": 11, "item": "Antidote", "hidden": False} in forest
    assert {"x": 1, "y": 18, "item": "Potion", "hidden": True} in forest


def test_viridian_mart_sells_poke_balls():
    mart = gamedata.shops("Viridian Mart")
    assert mart["items"] == ["Poke Ball", "Antidote", "Parlyz Heal", "Burn Heal"]
    assert gamedata.shops("Pallet Town") is None


# ===================================================================
# Species and moves
# ===================================================================


def test_all_151_species_with_bulbasaur_intact():
    all_species = gamedata.all_species()
    assert len(all_species) == 151
    bulbasaur = gamedata.species("Bulbasaur")
    assert bulbasaur["dex"] == 1
    assert bulbasaur["types"] == ["Grass", "Poison"]
    assert bulbasaur["base"] == {"hp": 45, "atk": 49, "def": 49, "spd": 45, "spc": 65}
    assert bulbasaur["catch_rate"] == 45
    assert bulbasaur["growth"] == "medium_slow"
    assert {"level": 7, "move": "Leech Seed"} in bulbasaur["learnset"]
    assert bulbasaur["evolutions"] == [{"to": "Ivysaur", "method": "level", "param": 16}]
    assert "HM01" in bulbasaur["tm_hm"]


def test_a_mono_type_is_listed_once():
    assert gamedata.species("Charmander")["types"] == ["Fire"]


def test_stone_evolutions_carry_the_stone():
    eevee = gamedata.species("Eevee")
    assert {"to": "Vaporeon", "method": "item", "param": "Water Stone"} in eevee["evolutions"]


def test_tackle():
    # Gen 1 stores accuracy as a byte out of 255; the JSON reports the percentage.
    assert gamedata.move("Tackle") == {
        "id": 33,
        "type": "Normal",
        "power": 35,
        "accuracy": 95,
        "accuracy_byte": 242,
        "pp": 35,
        "effect": "NO_ADDITIONAL_EFFECT",
    }


def test_every_learnset_move_exists():
    moves = gamedata.all_moves()
    for name, entry in gamedata.all_species().items():
        for learn in entry["learnset"]:
            assert learn["move"] in moves, f"{name} learns unknown move {learn['move']!r}"


# ===================================================================
# Types
# ===================================================================


def test_gen_1_type_list():
    listed = gamedata.types()["types"]
    assert "Bird" in listed  # Gen 1 has it
    assert "Dark" not in listed and "Steel" not in listed  # and not these
    assert listed[:3] == ["Normal", "Fighting", "Flying"]


def test_type_chart_matchups():
    assert gamedata.effectiveness("Water", ["Rock"]) == 2.0
    assert gamedata.effectiveness("Water", ["Grass"]) == 0.5
    assert gamedata.effectiveness("Normal", ["Ghost"]) == 0.0
    assert gamedata.effectiveness("Water", ["Rock", "Ground"]) == 4.0
    assert gamedata.effectiveness("Normal", ["Normal"]) == 1.0


# ===================================================================
# Damage
# ===================================================================


def test_damage_range_matches_the_worked_example():
    """Squirtle (L20, Special 40) uses Water Gun on a Geodude (Special 30).

    Water Gun is 40 power and special, so both sides use their Special stat:

        base   = (2 * 20 / 5 + 2) * 40 * 40 / 30 / 50 + 2
               = (10 * 40 * 40) / 30 / 50 + 2 = 16000 / 30 = 533
               = 533 / 50 = 10, + 2 = 12
        STAB   = 12 * 3 / 2 = 18                    (Water on a Water attacker)
        Rock   = 18 * 20 / 10 = 36                  (super effective)
        Ground = 36 * 20 / 10 = 72                  (super effective again)
        min    = 72 * 217 / 255 = 15624 / 255 = 61
        max    = 72 * 255 / 255 = 72
    """
    squirtle = {
        "level": 20,
        "types": ["Water"],
        "stats": {"attack": 30, "defense": 30, "speed": 30, "special": 40},
    }
    geodude = {"types": ["Rock", "Ground"], "stats": {"defense": 50, "special": 30}}
    assert gamedata.damage_range(squirtle, "Water Gun", geodude) == (61, 72)


def test_physical_moves_use_attack_and_defence():
    """Same Squirtle, Tackle (35 power, physical, no STAB) into Defence 50:

    base = (2 * 20 / 5 + 2) * 30 * 35 / 50 / 50 + 2
         = 10 * 30 * 35 = 10500 / 50 = 210, / 50 = 4, + 2 = 6
    Rock resists Normal: 6 * 5 / 10 = 3, Ground is neutral.
    min = 3 * 217 / 255 = 2, max = 3
    """
    squirtle = {
        "level": 20,
        "types": ["Water"],
        "stats": {"attack": 30, "defense": 30, "speed": 30, "special": 40},
    }
    geodude = {"types": ["Rock", "Ground"], "stats": {"defense": 50, "special": 30}}
    assert gamedata.damage_range(squirtle, "Tackle", geodude) == (2, 3)


def test_immunity_and_status_moves_do_nothing():
    attacker = {"level": 50, "types": ["Normal"], "stats": {"attack": 100, "special": 100}}
    ghost = {"types": ["Ghost", "Poison"], "stats": {"defense": 50, "special": 50}}
    assert gamedata.damage_range(attacker, "Tackle", ghost) == (0, 0)
    assert gamedata.damage_range(attacker, "Growl", ghost) == (0, 0)


def test_types_are_looked_up_from_the_species_when_not_given():
    """The memory reader always supplies types, but a caller holding only a
    name should still get the same answer."""
    by_name = gamedata.damage_range(
        {"level": 20, "species": "Squirtle", "stats": {"special": 40}},
        "Water Gun",
        {"species": "Geodude", "stats": {"special": 30}},
    )
    assert by_name == (61, 72)
