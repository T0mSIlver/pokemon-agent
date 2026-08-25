"""Guards for MAP_NAMES, the map id -> name table in pokemon_agent/memory/red.py.

The table is positional: ids come from the order of `map_const` lines in
pret/pokered's constants/map_constants.asm. It has drifted twice, both times
because an entry was missing and every later id silently shifted by one --
Viridian Forest reporting as "Pewter Museum 1F" (884 saves misfiled), then a
Poke Center reporting as "Mt Moon 1F". The objective packs select on map *names*,
so a rename in the table also silently disables objective tracking.

These tests pin all three failure modes: gaps, wrong ids, and dangling selectors.
"""

from __future__ import annotations

from pokemon_agent.agent_runtime import load_red_objective_packs
from pokemon_agent.memory.red import MAP_NAMES

# NUM_MAPS in map_constants.asm is 0xF8: PALLET_TOWN = 0 .. AGATHAS_ROOM = 0xF7.
NUM_MAPS = 248

# Read off the emulator by loading a save and looking at the frame. If one of
# these ever fails, the table has drifted again -- fix the table, not the test.
CONFIRMED_ON_EMULATOR = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    12: "Route 1",
    13: "Route 2",
    37: "Red's House 1F",
    38: "Red's House 2F",
    39: "Blue's House",
    40: "Oak's Lab",
    41: "Viridian Pokecenter",
    42: "Viridian Mart",
    43: "Viridian School",
    44: "Viridian House",
    51: "Viridian Forest",  # wild Kakuna seen here
    56: "Pewter Mart",  # shelves + clerk seen here
    58: "Pewter Pokecenter",  # counter + plants seen here
}

# The two drifts that were found by walking into them, kept as an explicit net.
DRIFT_REGRESSIONS = {
    47: "Viridian Forest North Gate",
    50: "Viridian Forest South Gate",  # missing entry that shifted 51 onwards
    51: "Viridian Forest",
    52: "Pewter Museum 1F",
    55: "Pewter Nidoran House",
    57: "Pewter Speech House",  # missing entry that shifted 58 onwards
    58: "Pewter Pokecenter",
    59: "Mt Moon 1F",
}

# UNUSED_MAP_xx ids, as hex, exactly as they appear in map_constants.asm.
UNUSED_MAP_IDS = {
    0x0B,
    0x69,
    0x6A,
    0x6B,
    0x6D,
    0x6E,
    0x6F,
    0x70,
    0x72,
    0x73,
    0x74,
    0x75,
    0xCC,
    0xCD,
    0xCE,
    0xE7,
    0xED,
    0xEE,
    0xF1,
    0xF2,
    0xF3,
    0xF4,
}


def test_table_covers_every_id_with_no_gaps():
    assert sorted(MAP_NAMES) == list(range(NUM_MAPS))


def test_unused_map_slots_are_present_as_placeholders():
    """The gaps are what caused the drift, so unused ids must stay listed."""
    placeholders = {i for i, name in MAP_NAMES.items() if name.startswith("Unused Map ")}
    assert placeholders == UNUSED_MAP_IDS


def test_confirmed_ids_match_the_emulator():
    for map_id, name in CONFIRMED_ON_EMULATOR.items():
        assert MAP_NAMES[map_id] == name, f"id {map_id} drifted"


def test_known_drifts_stay_fixed():
    for map_id, name in DRIFT_REGRESSIONS.items():
        assert MAP_NAMES[map_id] == name, f"id {map_id} drifted"


def test_no_two_ids_share_a_name():
    """Copy maps (CERULEAN_TRASHED_HOUSE_COPY and friends) keep their "Copy"
    suffix, so the table has no legitimate duplicates and any collision is a
    typo that would make two different places indistinguishable to the agent."""
    by_name: dict[str, list[int]] = {}
    for map_id, name in MAP_NAMES.items():
        by_name.setdefault(name, []).append(map_id)
    assert {n: ids for n, ids in by_name.items() if len(ids) > 1} == {}


def _selector_map_names() -> set[str]:
    names: set[str] = set()
    for pack in load_red_objective_packs():
        for objective in pack.get("objectives") or []:
            selector = objective.get("selector") or {}
            for key in ("map_in", "map_not_in"):
                names.update(selector.get(key) or [])
    return names


def test_objective_packs_only_select_on_real_map_names():
    """Renaming a map silently disables every selector that used the old name,
    which is how objective tracking broke for the forest gates and the Pewter
    houses. Every name a pack selects on has to exist in the table."""
    referenced = _selector_map_names()
    assert referenced, "no map selectors found -- the packs or the loader moved"
    missing = sorted(referenced - set(MAP_NAMES.values()))
    assert not missing, f"objective packs select on maps that do not exist: {missing}"
