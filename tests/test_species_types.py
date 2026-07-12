"""Gen 1 stores a mono-type's single type in both type bytes; state must not repeat it."""

from pokemon_agent.memory.red import INTERNAL_SPECIES_TO_DEX, SPECIES_NAMES, PokemonRedReader

WATER = 0x15
FLYING = 0x02
NORMAL = 0x00


def test_monotype_is_reported_once():
    assert PokemonRedReader._decode_types(WATER, WATER) == ["Water"]


def test_dual_type_keeps_both_in_order():
    assert PokemonRedReader._decode_types(NORMAL, FLYING) == ["Normal", "Flying"]


def test_unknown_type_byte_is_surfaced_not_swallowed():
    assert PokemonRedReader._decode_types(0xEE, 0xEE) == ["???(238)"]


def test_internal_species_indices_map_to_real_species():
    # Squirtle and Bulbasaur are the canonical off-by-table checks: the RAM party
    # struct stores an internal index, not the Pokedex number.
    assert SPECIES_NAMES[INTERNAL_SPECIES_TO_DEX[177]] == "Squirtle"
    assert SPECIES_NAMES[INTERNAL_SPECIES_TO_DEX[153]] == "Bulbasaur"
