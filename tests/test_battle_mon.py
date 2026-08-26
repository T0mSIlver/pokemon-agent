"""The two live ``battle_struct``s, against a synthetic RAM image.

The harness used to read the enemy's HP and types out of ``wEnemyMon`` and then
throw the rest of the struct away, re-deriving both sides' stats from base stats
at an average DV. Measured over 123 auto-saved battle frames from one run, that
guess was wrong for 196 of 492 enemy stat reads and for the player's Attack in
*every single frame* — the Boulder Badge alone puts 50 at 56, and Rage had it at
121 against a party-struct 54. These tests pin the offsets that fixed it.
"""

from pokemon_agent.memory.red import (
    ADDR_BATTLE_MON,
    ADDR_BATTLE_TYPE,
    ADDR_ENEMY_MON,
    ADDR_PLAYER_BATTLE_STATUS_2,
    USED_RAGE_BIT,
    PokemonRedReader,
)

#: Internal species index for Charmeleon, and for Paras.
CHARMELEON = 0xB2
PARAS = 0x6D

FIRE = 0x14
BUG = 0x07
GRASS = 0x16

EMBER = 52
GROWL = 45
SCRATCH = 10


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


def write_battle_struct(
    emulator: FakeEmulator,
    base: int,
    *,
    species: int,
    types: tuple[int, int],
    hp: int,
    max_hp: int,
    level: int,
    moves: tuple[int, ...] = (),
    pp: tuple[int, ...] = (),
    attack: int = 0,
    defense: int = 0,
    speed: int = 0,
    special: int = 0,
) -> None:
    def word(offset: int, value: int) -> None:
        emulator.mem[base + offset] = value >> 8
        emulator.mem[base + offset + 1] = value & 0xFF

    emulator.mem[base] = species
    word(1, hp)
    emulator.mem[base + 5], emulator.mem[base + 6] = types
    for index, move_id in enumerate(moves):
        emulator.mem[base + 8 + index] = move_id
    emulator.mem[base + 14] = level
    word(15, max_hp)
    word(17, attack)
    word(19, defense)
    word(21, speed)
    word(23, special)
    for index, counter in enumerate(pp):
        emulator.mem[base + 25 + index] = counter


def make_reader() -> tuple[PokemonRedReader, FakeEmulator]:
    emulator = FakeEmulator()
    emulator.mem[ADDR_BATTLE_TYPE] = 1  # wild
    write_battle_struct(
        emulator,
        ADDR_BATTLE_MON,
        species=CHARMELEON,
        types=(FIRE, FIRE),
        hp=39,
        max_hp=73,
        level=25,
        moves=(EMBER, GROWL),
        pp=(0, 39),
        attack=56,
        defense=44,
        speed=58,
        special=46,
    )
    write_battle_struct(
        emulator,
        ADDR_ENEMY_MON,
        species=PARAS,
        types=(BUG, GRASS),
        hp=27,
        max_hp=27,
        level=10,
        moves=(SCRATCH,),
        attack=21,
        defense=16,
        speed=11,
        special=18,
    )
    return PokemonRedReader(emulator), emulator


def test_the_enemy_carries_the_stats_the_engine_fights_with():
    reader, _ = make_reader()

    enemy = reader.read_battle()["enemy"]

    assert enemy["species"] == "Paras"
    assert enemy["level"] == 10
    assert enemy["hp"] == 27
    assert enemy["types"] == ["Bug", "Grass"]
    assert enemy["stats"] == {"attack": 21, "defense": 16, "speed": 11, "special": 18}
    # Names, not PP: the enemy's PP is not a fact any decision here turns on.
    assert enemy["moves"] == ["Scratch"]


def test_a_stat_stage_moves_the_number_the_reader_sees():
    # This is the whole reason to read the struct instead of deriving it. Leer
    # cuts Defense to two thirds *in this struct*; a stat re-derived from the
    # species' base stats would report the same number after Leer as before it,
    # which is what the harness did while one run used Leer 41 times.
    reader, emulator = make_reader()
    assert reader.read_battle()["enemy"]["stats"]["defense"] == 16

    emulator.mem[ADDR_ENEMY_MON + 20] = 10

    assert reader.read_battle()["enemy"]["stats"]["defense"] == 10


def test_the_battler_on_your_side_is_read_off_the_field_not_the_party():
    reader, _ = make_reader()

    active = reader.read_battle_mon()

    assert active["species"] == "Charmeleon"
    assert active["level"] == 25
    assert active["hp"] == 39
    # 56, not the 50 the party struct holds: badge boosts land here and nowhere
    # else, and this Attack was 121 in one measured frame with Rage built up.
    assert active["stats"]["attack"] == 56
    assert [move["name"] for move in active["moves"]] == ["Ember", "Growl"]
    assert [move["pp"] for move in active["moves"]] == [0, 39]


def test_rage_is_reported_as_the_thing_that_took_the_turn():
    # D063 bit 6, measured rather than looked up: a Rage confirmed in Mt. Moon
    # B2F left this byte at 0x40 four frames later and the top battle menu never
    # came back for the rest of the fight.
    reader, emulator = make_reader()
    assert reader.battle_lock_in() is None

    emulator.mem[ADDR_PLAYER_BATTLE_STATUS_2] = USED_RAGE_BIT

    assert reader.battle_lock_in() == "Rage"
