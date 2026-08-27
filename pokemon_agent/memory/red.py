"""Pokemon Red / Blue (USA) memory reader.

All RAM addresses come from the *pokered* decomp project
(https://github.com/pret/pokered).  This module targets the
USA Rev-A ROM but most offsets are identical for Rev-0 and Blue.

Gen 1 text uses a custom character encoding (0x50 = terminator,
0x80..0x99 = uppercase A-Z, etc.).  Money is stored as 3-byte BCD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pokemon_agent.memory.reader import GameMemoryReader

# ===================================================================
# RAM addresses (WRAM)
# ===================================================================

# -- Player --
ADDR_PLAYER_NAME = 0xD158  # 11 bytes
ADDR_RIVAL_NAME = 0xD34A  # 11 bytes
ADDR_MONEY = 0xD347  # 3 bytes BCD
ADDR_BADGES = 0xD356  # 1 byte bitmask

# -- Position --
ADDR_MAP_ID = 0xD35E  # current map number (wCurMap)
ADDR_MAP_Y = 0xD361  # player Y on map  (wYCoord)
ADDR_MAP_X = 0xD362  # player X on map  (wXCoord)
ADDR_MAP_BANK = 0xD35E  # same byte is map id
ADDR_MAP_TILESET = 0xD367  # current map tileset (wCurMapTileset)
ADDR_MAP_HEIGHT = 0xD368  # current map height in blocks
ADDR_MAP_WIDTH = 0xD369  # current map width in blocks
ADDR_FACING = 0xC109  # live sprite facing direction
# 0xC109 is the live sprite state that updates as the player moves.

# -- Party --
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES = 0xD164  # 6 bytes + terminator
ADDR_PARTY_DATA = 0xD16B  # 44 bytes per slot × 6
ADDR_PARTY_OT = 0xD273  # 11 bytes per OT × 6
ADDR_PARTY_NICKS = 0xD2B5  # 11 bytes per nick × 6

PARTY_MON_SIZE = 44
BATTLE_MON_SIZE = 29

# -- Bag --
ADDR_BAG_COUNT = 0xD31D
ADDR_BAG_ITEMS = 0xD31E  # pairs (item_id, qty)

BAG_ITEM_CAPACITY = 20

# -- PC items --
ADDR_PC_COUNT = 0xD53A  # wNumBoxItems
ADDR_PC_ITEMS = 0xD53B  # wBoxItems, pairs (item_id, qty)
PC_ITEM_CAPACITY = 50

# -- Item lists: the bag in a battle, and a mart's counter --
# Both are the same widget, so both are read the same way. Every address here
# was diffed off a running game rather than copied from a listing: talking to
# the Vermilion Mart clerk and pressing A on POKE BALL moved 0xCF91 from a stale
# 14 to 4 (POKE_BALL) and 0xCF97 from 1 to 99 (buy up to 99), and 0xD12A read 6
# on a counter that stocks exactly six things. 0xCC36 is the fourth press of
# Down on that list, where the cursor stops and the window scrolls instead.
ADDR_LIST_SCROLL_OFFSET = 0xCC36  # wListScrollOffset — rows scrolled off the top
ADDR_ITEM_LIST = 0xCF7B  # wItemList — count byte, then item ids, $FF-terminated
ADDR_CUR_ITEM = 0xCF91  # wCurItem — the entry A last confirmed
ADDR_LIST_MENU_ID = 0xCF94  # wListMenuID — which list widget is open
ADDR_ITEM_QUANTITY = 0xCF96  # wItemQuantity — the x01 in the quantity roller
ADDR_MAX_ITEM_QUANTITY = 0xCF97  # wMaxItemQuantity
ADDR_LIST_COUNT = 0xD12A  # wListCount — entries in the open list
#: wListMenuID values. 2 is a mart's priced list, 3 the plain bag list.
PRICED_ITEM_LIST_MENU = 2
ITEM_LIST_MENU = 3
#: wTextBoxID while either is up. Not the battle menu's 11.
LIST_MENU_TEXT_BOX_ID = 13
#: wTextBoxID for the BUY/SELL/QUIT frame and for the YES/NO under "That will be
#: $2000. OK?". Both are steps a purchase has to wait for rather than press
#: through: A sent before the prompt is drawn is swallowed, and a run that did
#: that read back an unchanged $7198 and an unchanged bag.
BUY_SELL_QUIT_TEXT_BOX_ID = 14
TWO_OPTION_TEXT_BOX_ID = 20

#: Ball item ids, weakest first. Order is what "throw a ball" resolves to when
#: no ball is named: spend the cheapest thing that can work before the Ultra
#: Ball you cannot buy back, and never spend the Master Ball by accident.
BALL_ITEM_IDS = (4, 3, 2, 1)  # Poke, Great, Ultra, Master

# -- Battle --
ADDR_BATTLE_TYPE = 0xD057  # 0=none, 1=wild, 2=trainer
ADDR_ENEMY_COUNT = 0xD89C
ADDR_ENEMY_SPECIES = 0xD89D
ADDR_ENEMY_DATA = 0xD8A4  # 44 bytes per mon (party struct)
ADDR_ENEMY_MON = 0xCFE5  # active enemy battle mon (live HP/stats)
#: wEnemyMonActualCatchRate — the number ItemUseBall compares its first random
#: roll against, and the only input to a catch the species table cannot supply:
#: the Safari Zone's bait and rocks halve and double *this* byte, not the base
#: rate. Derived twice over from addresses this file already trusts: wEnemyMon
#: (0xCFE5) plus one 29-byte battle_struct plus wEnemyMonBaseStats (5) lands on
#: it, and one byte further plus an 11-byte nickname lands exactly on wBattleMon
#: (0xD014).
ADDR_ENEMY_CATCH_RATE = 0xD007

# -- Battle menus --
# Every address and every literal in this block was read off a running battle and
# then checked against a screenshot of the same frame — a wild Weedle fight and a
# trainer Onix fight, three-move and four-move leads. Do not "correct" them from a
# disassembly listing without re-running that check; a wrong constant here silently
# fires the wrong attack, which is exactly the bug this block exists to kill.
ADDR_TOP_MENU_ITEM_Y = 0xCC24  # wTopMenuItemY — menu row anchor, identifies the menu
ADDR_TOP_MENU_ITEM_X = 0xCC25  # wTopMenuItemX — battle-menu column: 9 left, 15 right
ADDR_CURRENT_MENU_ITEM = 0xCC26  # wCurrentMenuItem — row (top menu) / 1-based entry (moves)
ADDR_MAX_MENU_ITEM = 0xCC28  # wMaxMenuItem
ADDR_PLAYER_MOVE_LIST_INDEX = 0xCC2E  # wPlayerMoveListIndex — 0-based, survives the menu closing
ADDR_PLAYER_SELECTED_MOVE = 0xCCDC  # wPlayerSelectedMove — the move id the turn will actually use
ADDR_PLAYER_BATTLE_STATUS_2 = 0xD063  # wPlayerBattleStatus2
#: Bit 6 of wPlayerBattleStatus2. Measured, not read off a listing: a Rage
#: confirmed in Mt. Moon B2F left D063 at 0x40 four frames later and the top
#: battle menu never returned for the rest of the fight.
USED_RAGE_BIT = 0x40
ADDR_BATTLE_MON = 0xD014  # wBattleMon — the *active* battler, not party slot 0
ADDR_BATTLE_MON_MOVES = 0xD01C  # wBattleMonMoves — the *active* battler, not party slot 0
ADDR_BATTLE_MON_PP = 0xD02D  # wBattleMonPP, low 6 bits are the counter

#: Both battle menus report this text-box id; the row anchor is what tells them apart.
BATTLE_MENU_TEXT_BOX_ID = 11
#: FIGHT/PKMN over ITEM/RUN. Row lives in wCurrentMenuItem, column in wTopMenuItemX.
TOP_MENU_ITEM_Y = 14
TOP_MENU_COLUMN_X = (9, 15)
TOP_MENU_MAX_ITEM = 1
TOP_MENU_ENTRIES = (("FIGHT", "PKMN"), ("ITEM", "RUN"))
#: The move list. wCurrentMenuItem is 1-based here and covers only the real moves,
#: so it reads 1..4 and wraps at both ends.
MOVE_MENU_ITEM_Y = 12
MOVE_MENU_ITEM_X = 5

# -- Dialog --
ADDR_TEXT_BOX_ID = 0xD125  # wTextBoxID
ADDR_JOY_IGNORE = 0xD730  # bit 5 = joypad disabled (in dialogue)
ADDR_TEXT_PROGRESS = 0xC4F2  # approximate; nonzero when text printing
ADDR_WINDOW_Y = 0xFF4A  # hWY / rWY, window Y position (0x90 = hidden)
ADDR_WINDOW_X = 0xFF4B  # hWX / rWX, window X position

# -- The screen, as words --
#
# wTileMap is the 20x18 grid of tiles the game has actually drawn, and Gen 1's
# text encoding *is* its font tile numbering, so the same GEN1_ENCODING that
# decodes a nickname out of the party struct decodes the dialogue box off the
# screen. Verified against the ROM: driving TM28 onto a four-move Charmeleon in
# PyBoy reads back "Which move should / be forgotten?" over "CUT GROWL EMBER
# LEER", exactly as rendered.
#
# The harness has had no on-screen text at all until now -- ``screen_text`` in
# every bundle is the fixed placeholder "Dialog box visible (waiting for
# input)", 36,300 bytes of it across 660 payloads in one run.
ADDR_TILE_MAP = 0xC3A0  # wTileMap
TILE_MAP_WIDTH = 20
TILE_MAP_HEIGHT = 18

# -- Learning a move --
#
# Both routes into a fifth move -- a TM/HM out of the bag and a level-up in
# battle -- go through the same prompt, and this byte holds the move being
# taught for every frame of it. Measured: it reads 91 (Dig) from "Booted up an
# HM!" through "Which move should be forgotten?" while TM28 is being used, and
# still held 15 (Cut) from the last teach before that, so it is scratch and
# means nothing on its own. It is only ever read while the screen says a learn
# is happening, and what it names is checked against the moves the Pokemon
# already knows before it is printed.
ADDR_MOVE_BEING_LEARNED = 0xD0E0
ADDR_WHICH_POKEMON = 0xCF92  # wWhichPokemon, the party slot a menu is acting on

# -- Pokedex --
ADDR_DEX_OWNED = 0xD2F7  # 19 bytes (152 bits, only 151 used)
ADDR_DEX_SEEN = 0xD30A

# -- Play time --
# Every one of these was one byte low, and the whole block read as garbage: hours
# came off 0xDA40 as a little-endian u16 whose low byte is padding, so a 25-hour
# save reported 6400, and "minutes" read wPlayTimeMaxed, which is 0 in every save
# state in saves/ -- 494 of them, all reporting :00. The real layout, walked out
# of pokered's ram/wram.asm from wEventFlags (0xD747) and confirmed against those
# saves, is five consecutive single bytes with a flag in the middle.
ADDR_PLAYTIME_H = 0xDA41  # wPlayTimeHours, 1 byte
ADDR_PLAYTIME_MAXED = 0xDA42  # wPlayTimeMaxed, set once the clock stops at 255:59
ADDR_PLAYTIME_M = 0xDA43  # wPlayTimeMinutes
ADDR_PLAYTIME_S = 0xDA44  # wPlayTimeSeconds
ADDR_PLAYTIME_F = 0xDA45  # wPlayTimeFrames

# -- Event / story flags --
ADDR_EVENT_FLAGS = 0xD747  # large bitfield (wEventFlags)
ADDR_OAK_PARCEL = 0xD74E  # bit 1 = has parcel
ADDR_POKEDEX_FLAG = 0xD74B  # bit 5 = has pokedex

# wTownVisitedFlag: one bit per Fly destination, in map-id order from
# PALLET_TOWN = 0 to SAFFRON_CITY = 10. Set on arrival, never cleared, which is
# what makes ELITE_FOUR's Indigo Plateau bit (9) usable as a milestone. The
# constant here used to read 0xD5F3, which is not this array and not any flag:
# it holds 1 in a save standing in Red's house before the Town Map exists.
ADDR_TOWN_VISITED_FLAGS = 0xD70B  # 2 bytes, 11 bits used

# wElite4Flags. Bit 0 is set by HallOfFame.asm and read by nothing in the game,
# so unlike every event flag in the Indigo Plateau block it survives the reset
# the Hall of Fame performs on its way in.
ADDR_ELITE_4_FLAGS = 0xD734

# -- Warps --
ADDR_WARP_COUNT = 0xD3AE  # current map warp count (wNumberOfWarps)
ADDR_WARP_ENTRIES = 0xD3AF  # current map warp entries (Y, X, warp ID, map ID)

# -- Signs / background events --
ADDR_SIGN_COUNT = 0xD4B0  # current map sign count (wNumSigns)
ADDR_SIGN_COORDS = 0xD4B1  # current map sign coordinates (wSignCoords; Y, X)
ADDR_SIGN_TEXT_IDS = 0xD4D1  # current map sign text ids (wSignTextIDs)
MAX_SIGNS = 16


# ===================================================================
# Gen-1 character encoding table
# ===================================================================


def _build_encoding_table() -> Dict[int, str]:
    """Build the Gen-1 text encoding lookup."""
    t: Dict[int, str] = {}
    # uppercase A-Z: 0x80..0x99
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        t[0x80 + i] = c
    # lowercase a-z: 0xA0..0xB9
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        t[0xA0 + i] = c
    # digits 0-9: 0xF6..0xFF
    for i, c in enumerate("0123456789"):
        t[0xF6 + i] = c
    # The six tiles that sit immediately after lowercase z: one accented vowel
    # and five apostrophe ligatures, each a single tile. Without them "can't"
    # decodes as "can t" with a hole in it and "POKeMON" loses its e -- which is
    # exactly how the move-learn prompt "But, CHAR can't learn more" read on
    # screen before they were added, and why matching it by its words missed a
    # frame of the prompt.
    t[0xBA] = "é"
    t[0xBB] = "'d"
    t[0xBC] = "'l"
    t[0xBD] = "'s"
    t[0xBE] = "'t"
    t[0xBF] = "'v"
    # punctuation / specials
    t[0x7F] = " "
    t[0xE0] = "'"
    t[0xE1] = "P"  # PK
    t[0xE2] = "M"  # MN
    t[0xE3] = "-"
    t[0xE6] = "?"
    t[0xE7] = "!"
    t[0xE8] = "."
    t[0xF0] = "¥"
    t[0xF1] = "×"
    t[0xF3] = "/"
    t[0xF4] = ","
    t[0xF5] = "♀"
    # terminator / newline (handled externally; map for safety)
    t[0x50] = ""
    t[0x4F] = "\n"
    t[0x51] = "\n"
    t[0x55] = "\n"
    return t


GEN1_ENCODING: Dict[int, str] = _build_encoding_table()


# ===================================================================
# Name tables
# ===================================================================

SPECIES_NAMES: Dict[int, str] = {
    0: "MissingNo.",
    1: "Bulbasaur",
    2: "Ivysaur",
    3: "Venusaur",
    4: "Charmander",
    5: "Charmeleon",
    6: "Charizard",
    7: "Squirtle",
    8: "Wartortle",
    9: "Blastoise",
    10: "Caterpie",
    11: "Metapod",
    12: "Butterfree",
    13: "Weedle",
    14: "Kakuna",
    15: "Beedrill",
    16: "Pidgey",
    17: "Pidgeotto",
    18: "Pidgeot",
    19: "Rattata",
    20: "Raticate",
    21: "Spearow",
    22: "Fearow",
    23: "Ekans",
    24: "Arbok",
    25: "Pikachu",
    26: "Raichu",
    27: "Sandshrew",
    28: "Sandslash",
    29: "Nidoran♀",
    30: "Nidorina",
    31: "Nidoqueen",
    32: "Nidoran♂",
    33: "Nidorino",
    34: "Nidoking",
    35: "Clefairy",
    36: "Clefable",
    37: "Vulpix",
    38: "Ninetales",
    39: "Jigglypuff",
    40: "Wigglytuff",
    41: "Zubat",
    42: "Golbat",
    43: "Oddish",
    44: "Gloom",
    45: "Vileplume",
    46: "Paras",
    47: "Parasect",
    48: "Venonat",
    49: "Venomoth",
    50: "Diglett",
    51: "Dugtrio",
    52: "Meowth",
    53: "Persian",
    54: "Psyduck",
    55: "Golduck",
    56: "Mankey",
    57: "Primeape",
    58: "Growlithe",
    59: "Arcanine",
    60: "Poliwag",
    61: "Poliwhirl",
    62: "Poliwrath",
    63: "Abra",
    64: "Kadabra",
    65: "Alakazam",
    66: "Machop",
    67: "Machoke",
    68: "Machamp",
    69: "Bellsprout",
    70: "Weepinbell",
    71: "Victreebel",
    72: "Tentacool",
    73: "Tentacruel",
    74: "Geodude",
    75: "Graveler",
    76: "Golem",
    77: "Ponyta",
    78: "Rapidash",
    79: "Slowpoke",
    80: "Slowbro",
    81: "Magnemite",
    82: "Magneton",
    83: "Farfetch'd",
    84: "Doduo",
    85: "Dodrio",
    86: "Seel",
    87: "Dewgong",
    88: "Grimer",
    89: "Muk",
    90: "Shellder",
    91: "Cloyster",
    92: "Gastly",
    93: "Haunter",
    94: "Gengar",
    95: "Onix",
    96: "Drowzee",
    97: "Hypno",
    98: "Krabby",
    99: "Kingler",
    100: "Voltorb",
    101: "Electrode",
    102: "Exeggcute",
    103: "Exeggutor",
    104: "Cubone",
    105: "Marowak",
    106: "Hitmonlee",
    107: "Hitmonchan",
    108: "Lickitung",
    109: "Koffing",
    110: "Weezing",
    111: "Rhyhorn",
    112: "Rhydon",
    113: "Chansey",
    114: "Tangela",
    115: "Kangaskhan",
    116: "Horsea",
    117: "Seadra",
    118: "Goldeen",
    119: "Seaking",
    120: "Staryu",
    121: "Starmie",
    122: "Mr. Mime",
    123: "Scyther",
    124: "Jynx",
    125: "Electabuzz",
    126: "Magmar",
    127: "Pinsir",
    128: "Tauros",
    129: "Magikarp",
    130: "Gyarados",
    131: "Lapras",
    132: "Ditto",
    133: "Eevee",
    134: "Vaporeon",
    135: "Jolteon",
    136: "Flareon",
    137: "Porygon",
    138: "Omanyte",
    139: "Omastar",
    140: "Kabuto",
    141: "Kabutops",
    142: "Aerodactyl",
    143: "Snorlax",
    144: "Articuno",
    145: "Zapdos",
    146: "Moltres",
    147: "Dratini",
    148: "Dragonair",
    149: "Dragonite",
    150: "Mewtwo",
    151: "Mew",
}

# Gen 1 party and battle structs store an internal species index, not the
# Pokédex number. Translate that internal index through pokered's dex order.
INTERNAL_SPECIES_TO_DEX: Dict[int, int] = {
    1: 112,
    2: 115,
    3: 32,
    4: 35,
    5: 21,
    6: 100,
    7: 34,
    8: 80,
    9: 2,
    10: 103,
    11: 108,
    12: 102,
    13: 88,
    14: 94,
    15: 29,
    16: 31,
    17: 104,
    18: 111,
    19: 131,
    20: 59,
    21: 151,
    22: 130,
    23: 90,
    24: 72,
    25: 92,
    26: 123,
    27: 120,
    28: 9,
    29: 127,
    30: 114,
    31: 0,
    32: 0,
    33: 58,
    34: 95,
    35: 22,
    36: 16,
    37: 79,
    38: 64,
    39: 75,
    40: 113,
    41: 67,
    42: 122,
    43: 106,
    44: 107,
    45: 24,
    46: 47,
    47: 54,
    48: 96,
    49: 76,
    50: 0,
    51: 126,
    52: 0,
    53: 125,
    54: 82,
    55: 109,
    56: 0,
    57: 56,
    58: 86,
    59: 50,
    60: 128,
    61: 0,
    62: 0,
    63: 0,
    64: 83,
    65: 48,
    66: 149,
    67: 0,
    68: 0,
    69: 0,
    70: 84,
    71: 60,
    72: 124,
    73: 146,
    74: 144,
    75: 145,
    76: 132,
    77: 52,
    78: 98,
    79: 0,
    80: 0,
    81: 0,
    82: 37,
    83: 38,
    84: 25,
    85: 26,
    86: 0,
    87: 0,
    88: 147,
    89: 148,
    90: 140,
    91: 141,
    92: 116,
    93: 117,
    94: 0,
    95: 0,
    96: 27,
    97: 28,
    98: 138,
    99: 139,
    100: 39,
    101: 40,
    102: 133,
    103: 136,
    104: 135,
    105: 134,
    106: 66,
    107: 41,
    108: 23,
    109: 46,
    110: 61,
    111: 62,
    112: 13,
    113: 14,
    114: 15,
    115: 0,
    116: 85,
    117: 57,
    118: 51,
    119: 49,
    120: 87,
    121: 0,
    122: 0,
    123: 10,
    124: 11,
    125: 12,
    126: 68,
    127: 0,
    128: 55,
    129: 97,
    130: 42,
    131: 150,
    132: 143,
    133: 129,
    134: 0,
    135: 0,
    136: 89,
    137: 0,
    138: 99,
    139: 91,
    140: 0,
    141: 101,
    142: 36,
    143: 110,
    144: 53,
    145: 105,
    146: 0,
    147: 93,
    148: 63,
    149: 65,
    150: 17,
    151: 18,
    152: 121,
    153: 1,
    154: 3,
    155: 73,
    156: 0,
    157: 118,
    158: 119,
    159: 0,
    160: 0,
    161: 0,
    162: 0,
    163: 77,
    164: 78,
    165: 19,
    166: 20,
    167: 33,
    168: 30,
    169: 74,
    170: 137,
    171: 142,
    172: 0,
    173: 81,
    174: 0,
    175: 0,
    176: 4,
    177: 7,
    178: 5,
    179: 8,
    180: 6,
    181: 0,
    182: 0,
    183: 0,
    184: 0,
    185: 43,
    186: 44,
    187: 45,
    188: 69,
    189: 70,
    190: 71,
}

MOVE_NAMES: Dict[int, str] = {
    0: "(none)",
    1: "Pound",
    2: "Karate Chop",
    3: "Double Slap",
    4: "Comet Punch",
    5: "Mega Punch",
    6: "Pay Day",
    7: "Fire Punch",
    8: "Ice Punch",
    9: "Thunder Punch",
    10: "Scratch",
    11: "Vice Grip",
    12: "Guillotine",
    13: "Razor Wind",
    14: "Swords Dance",
    15: "Cut",
    16: "Gust",
    17: "Wing Attack",
    18: "Whirlwind",
    19: "Fly",
    20: "Bind",
    21: "Slam",
    22: "Vine Whip",
    23: "Stomp",
    24: "Double Kick",
    25: "Mega Kick",
    26: "Jump Kick",
    27: "Rolling Kick",
    28: "Sand Attack",
    29: "Headbutt",
    30: "Horn Attack",
    31: "Fury Attack",
    32: "Horn Drill",
    33: "Tackle",
    34: "Body Slam",
    35: "Wrap",
    36: "Take Down",
    37: "Thrash",
    38: "Double-Edge",
    39: "Tail Whip",
    40: "Poison Sting",
    41: "Twineedle",
    42: "Pin Missile",
    43: "Leer",
    44: "Bite",
    45: "Growl",
    46: "Roar",
    47: "Sing",
    48: "Supersonic",
    49: "Sonic Boom",
    50: "Disable",
    51: "Acid",
    52: "Ember",
    53: "Flamethrower",
    54: "Mist",
    55: "Water Gun",
    56: "Hydro Pump",
    57: "Surf",
    58: "Ice Beam",
    59: "Blizzard",
    60: "Psybeam",
    61: "Bubble Beam",
    62: "Aurora Beam",
    63: "Hyper Beam",
    64: "Peck",
    65: "Drill Peck",
    66: "Submission",
    67: "Low Kick",
    68: "Counter",
    69: "Seismic Toss",
    70: "Strength",
    71: "Absorb",
    72: "Mega Drain",
    73: "Leech Seed",
    74: "Growth",
    75: "Razor Leaf",
    76: "Solar Beam",
    77: "Poison Powder",
    78: "Stun Spore",
    79: "Sleep Powder",
    80: "Petal Dance",
    81: "String Shot",
    82: "Dragon Rage",
    83: "Fire Spin",
    84: "Thunder Shock",
    85: "Thunderbolt",
    86: "Thunder Wave",
    87: "Thunder",
    88: "Rock Throw",
    89: "Earthquake",
    90: "Fissure",
    91: "Dig",
    92: "Toxic",
    93: "Confusion",
    94: "Psychic",
    95: "Hypnosis",
    96: "Meditate",
    97: "Agility",
    98: "Quick Attack",
    99: "Rage",
    100: "Teleport",
    101: "Night Shade",
    102: "Mimic",
    103: "Screech",
    104: "Double Team",
    105: "Recover",
    106: "Harden",
    107: "Minimize",
    108: "Smokescreen",
    109: "Confuse Ray",
    110: "Withdraw",
    111: "Defense Curl",
    112: "Barrier",
    113: "Light Screen",
    114: "Haze",
    115: "Reflect",
    116: "Focus Energy",
    117: "Bide",
    118: "Metronome",
    119: "Mirror Move",
    120: "Self-Destruct",
    121: "Egg Bomb",
    122: "Lick",
    123: "Smog",
    124: "Sludge",
    125: "Bone Club",
    126: "Fire Blast",
    127: "Waterfall",
    128: "Clamp",
    129: "Swift",
    130: "Skull Bash",
    131: "Spike Cannon",
    132: "Constrict",
    133: "Amnesia",
    134: "Kinesis",
    135: "Soft-Boiled",
    136: "High Jump Kick",
    137: "Glare",
    138: "Dream Eater",
    139: "Poison Gas",
    140: "Barrage",
    141: "Leech Life",
    142: "Lovely Kiss",
    143: "Sky Attack",
    144: "Transform",
    145: "Bubble",
    146: "Dizzy Punch",
    147: "Spore",
    148: "Flash",
    149: "Psywave",
    150: "Splash",
    151: "Acid Armor",
    152: "Crabhammer",
    153: "Explosion",
    154: "Fury Swipes",
    155: "Bonemerang",
    156: "Rest",
    157: "Rock Slide",
    158: "Hyper Fang",
    159: "Sharpen",
    160: "Conversion",
    161: "Tri Attack",
    162: "Super Fang",
    163: "Slash",
    164: "Substitute",
    165: "Struggle",
}

TYPE_NAMES: Dict[int, str] = {
    # Gen 1 type ids. 6 is BIRD, a cut type that no obtainable species uses, and
    # leaving it out shifted Bug and Ghost down one: a wild Weedle (Bug/Poison)
    # was being reported as Ghost/Poison, which inverts every matchup the agent
    # reasons about. Psychic/Ice were transposed in the upper block for the same
    # reason.
    0: "Normal",
    1: "Fighting",
    2: "Flying",
    3: "Poison",
    4: "Ground",
    5: "Rock",
    6: "Bird",
    7: "Bug",
    8: "Ghost",
    20: "Fire",
    21: "Water",
    22: "Grass",
    23: "Electric",
    24: "Psychic",
    25: "Ice",
    26: "Dragon",
}

ITEM_NAMES: Dict[int, str] = {
    0: "(none)",
    1: "Master Ball",
    2: "Ultra Ball",
    3: "Great Ball",
    4: "Poke Ball",
    5: "Town Map",
    6: "Bicycle",
    7: "?????",
    8: "Safari Ball",
    9: "Pokedex",
    10: "Moon Stone",
    11: "Antidote",
    12: "Burn Heal",
    13: "Ice Heal",
    14: "Awakening",
    15: "Parlyz Heal",
    16: "Full Restore",
    17: "Max Potion",
    18: "Hyper Potion",
    19: "Super Potion",
    20: "Potion",
    21: "Boulder Badge",
    22: "Cascade Badge",
    23: "Thunder Badge",
    24: "Rainbow Badge",
    25: "Soul Badge",
    26: "Marsh Badge",
    27: "Volcano Badge",
    28: "Earth Badge",
    29: "Escape Rope",
    30: "Repel",
    31: "Old Amber",
    32: "Fire Stone",
    33: "Thunder Stone",
    34: "Water Stone",
    35: "HP Up",
    36: "Protein",
    37: "Iron",
    38: "Carbos",
    39: "Calcium",
    40: "Rare Candy",
    41: "Dome Fossil",
    42: "Helix Fossil",
    43: "Secret Key",
    44: "?????",
    45: "Bike Voucher",
    46: "X Accuracy",
    47: "Leaf Stone",
    48: "Card Key",
    49: "Nugget",
    50: "PP Up",
    51: "Poke Doll",
    52: "Full Heal",
    53: "Revive",
    54: "Max Revive",
    55: "Guard Spec.",
    56: "Super Repel",
    57: "Max Repel",
    58: "Dire Hit",
    59: "Coin",
    60: "Fresh Water",
    61: "Soda Pop",
    62: "Lemonade",
    63: "S.S. Ticket",
    64: "Gold Teeth",
    65: "X Attack",
    66: "X Defend",
    67: "X Speed",
    68: "X Special",
    69: "Coin Case",
    70: "Oak's Parcel",
    71: "Itemfinder",
    72: "Silph Scope",
    73: "Poke Flute",
    74: "Lift Key",
    75: "Exp. All",
    76: "Old Rod",
    77: "Good Rod",
    78: "Super Rod",
    79: "PP Up",
    80: "Ether",
    81: "Max Ether",
    82: "Elixir",
    83: "Max Elixir",
    196: "HM01",
    197: "HM02",
    198: "HM03",
    199: "HM04",
    200: "HM05",
    201: "TM01",
    202: "TM02",
    203: "TM03",
    204: "TM04",
    205: "TM05",
    206: "TM06",
    207: "TM07",
    208: "TM08",
    209: "TM09",
    210: "TM10",
    211: "TM11",
    212: "TM12",
    213: "TM13",
    214: "TM14",
    215: "TM15",
    216: "TM16",
    217: "TM17",
    218: "TM18",
    219: "TM19",
    220: "TM20",
    221: "TM21",
    222: "TM22",
    223: "TM23",
    224: "TM24",
    225: "TM25",
    226: "TM26",
    227: "TM27",
    228: "TM28",
    229: "TM29",
    230: "TM30",
    231: "TM31",
    232: "TM32",
    233: "TM33",
    234: "TM34",
    235: "TM35",
    236: "TM36",
    237: "TM37",
    238: "TM38",
    239: "TM39",
    240: "TM40",
    241: "TM41",
    242: "TM42",
    243: "TM43",
    244: "TM44",
    245: "TM45",
    246: "TM46",
    247: "TM47",
    248: "TM48",
    249: "TM49",
    250: "TM50",
}

# fmt: off
# ---------------------------------------------------------------------------
# Map id -> display name.
#
# Regenerated wholesale from the pret/pokered decompilation, which is the source
# of truth for this table:
#   https://raw.githubusercontent.com/pret/pokered/master/constants/map_constants.asm
#
# Ids are POSITIONAL. Every `map_const` line in that file consumes the next id,
# counting from PALLET_TOWN = 0 to AGATHAS_ROOM = 0xF7 = 247, so a single missing
# entry silently shifts every map after it. That is exactly how this table
# drifted twice: Viridian Forest reported itself as "Pewter Museum 1F" (884 saves
# misfiled), and a Poke Center reported itself as "Mt Moon 1F". The UNUSED_MAP_xx
# placeholders below are therefore load-bearing -- never drop one to tidy up.
#
# Names are produced from the constant by a mechanical rule:
#   * underscores become spaces and each word is Title Case
#   * floor suffixes keep their casing (1F, 2F, B1F, B2F)
#   * SS -> "S.S.", MT -> "Mt", MR -> "Mr", CO -> "Co"
#   * "of" / "the" / "and" stay lower case unless they lead the name
#   * possessive plurals regain their apostrophe (REDS_HOUSE_1F -> "Red's House 1F")
#   * UNUSED_MAP_xx -> "Unused Map xx"
# plus the overrides in MAP_NAME_OVERRIDES.
#
# Checked in as a literal rather than computed at import: a running emulator must
# not depend on network access, and the diff has to be reviewable.
# ---------------------------------------------------------------------------

# The only places where the mechanical rule is overridden, recorded here so the
# table can be regenerated. Keep it short -- each entry is a name someone has to
# maintain by hand forever.
MAP_NAME_OVERRIDES: Dict[str, str] = {
    # Terse constants: "MUSEUM"/"BIKE_SHOP" say nothing about which town they are
    # in, and both the agent and the objective packs select maps by name.
    "MUSEUM_1F": "Pewter Museum 1F",
    "MUSEUM_2F": "Pewter Museum 2F",
    "BIKE_SHOP": "Cerulean Bike Shop",
    # Shorter labels this project already used, confirmed on the emulator.
    "VIRIDIAN_SCHOOL_HOUSE": "Viridian School",
    "VIRIDIAN_NICKNAME_HOUSE": "Viridian House",
    # CELADON_MART_* is the six-floor department store, not a Poke Mart; the
    # existing label keeps it distinct from the real marts in other towns.
    "CELADON_MART_1F": "Celadon Dept Store 1F",
    "CELADON_MART_2F": "Celadon Dept Store 2F",
    "CELADON_MART_3F": "Celadon Dept Store 3F",
    "CELADON_MART_4F": "Celadon Dept Store 4F",
    "CELADON_MART_5F": "Celadon Dept Store 5F",
    "CELADON_MART_ROOF": "Celadon Dept Store Roof",
    "CELADON_MART_ELEVATOR": "Celadon Dept Store Elevator",
}

# Ids confirmed empirically on the emulator (load a save, read the frame):
#   0 Pallet Town, 1 Viridian City, 2 Pewter City, 12 Route 1, 13 Route 2,
#   37-44 the Pallet/Viridian interiors, 51 Viridian Forest (wild Kakuna seen),
#   56 Pewter Mart (shelves + clerk), 58 Pewter Pokecenter (counter + plants).
# tests/test_map_names.py pins those, the no-gaps invariant, and the fact that
# every map name an objective pack selects on actually exists here.
MAP_NAMES: Dict[int, str] = {
    0: "Pallet Town",  # PALLET_TOWN
    1: "Viridian City",  # VIRIDIAN_CITY
    2: "Pewter City",  # PEWTER_CITY
    3: "Cerulean City",  # CERULEAN_CITY
    4: "Lavender Town",  # LAVENDER_TOWN
    5: "Vermilion City",  # VERMILION_CITY
    6: "Celadon City",  # CELADON_CITY
    7: "Fuchsia City",  # FUCHSIA_CITY
    8: "Cinnabar Island",  # CINNABAR_ISLAND
    9: "Indigo Plateau",  # INDIGO_PLATEAU
    10: "Saffron City",  # SAFFRON_CITY
    11: "Unused Map 0B",  # UNUSED_MAP_0B
    12: "Route 1",  # ROUTE_1
    13: "Route 2",  # ROUTE_2
    14: "Route 3",  # ROUTE_3
    15: "Route 4",  # ROUTE_4
    16: "Route 5",  # ROUTE_5
    17: "Route 6",  # ROUTE_6
    18: "Route 7",  # ROUTE_7
    19: "Route 8",  # ROUTE_8
    20: "Route 9",  # ROUTE_9
    21: "Route 10",  # ROUTE_10
    22: "Route 11",  # ROUTE_11
    23: "Route 12",  # ROUTE_12
    24: "Route 13",  # ROUTE_13
    25: "Route 14",  # ROUTE_14
    26: "Route 15",  # ROUTE_15
    27: "Route 16",  # ROUTE_16
    28: "Route 17",  # ROUTE_17
    29: "Route 18",  # ROUTE_18
    30: "Route 19",  # ROUTE_19
    31: "Route 20",  # ROUTE_20
    32: "Route 21",  # ROUTE_21
    33: "Route 22",  # ROUTE_22
    34: "Route 23",  # ROUTE_23
    35: "Route 24",  # ROUTE_24
    36: "Route 25",  # ROUTE_25
    37: "Red's House 1F",  # REDS_HOUSE_1F
    38: "Red's House 2F",  # REDS_HOUSE_2F
    39: "Blue's House",  # BLUES_HOUSE
    40: "Oak's Lab",  # OAKS_LAB
    41: "Viridian Pokecenter",  # VIRIDIAN_POKECENTER
    42: "Viridian Mart",  # VIRIDIAN_MART
    43: "Viridian School",  # VIRIDIAN_SCHOOL_HOUSE
    44: "Viridian House",  # VIRIDIAN_NICKNAME_HOUSE
    45: "Viridian Gym",  # VIRIDIAN_GYM
    46: "Diglett's Cave Route 2",  # DIGLETTS_CAVE_ROUTE_2
    47: "Viridian Forest North Gate",  # VIRIDIAN_FOREST_NORTH_GATE
    48: "Route 2 Trade House",  # ROUTE_2_TRADE_HOUSE
    49: "Route 2 Gate",  # ROUTE_2_GATE
    50: "Viridian Forest South Gate",  # VIRIDIAN_FOREST_SOUTH_GATE
    51: "Viridian Forest",  # VIRIDIAN_FOREST
    52: "Pewter Museum 1F",  # MUSEUM_1F
    53: "Pewter Museum 2F",  # MUSEUM_2F
    54: "Pewter Gym",  # PEWTER_GYM
    55: "Pewter Nidoran House",  # PEWTER_NIDORAN_HOUSE
    56: "Pewter Mart",  # PEWTER_MART
    57: "Pewter Speech House",  # PEWTER_SPEECH_HOUSE
    58: "Pewter Pokecenter",  # PEWTER_POKECENTER
    59: "Mt Moon 1F",  # MT_MOON_1F
    60: "Mt Moon B1F",  # MT_MOON_B1F
    61: "Mt Moon B2F",  # MT_MOON_B2F
    62: "Cerulean Trashed House",  # CERULEAN_TRASHED_HOUSE
    63: "Cerulean Trade House",  # CERULEAN_TRADE_HOUSE
    64: "Cerulean Pokecenter",  # CERULEAN_POKECENTER
    65: "Cerulean Gym",  # CERULEAN_GYM
    66: "Cerulean Bike Shop",  # BIKE_SHOP
    67: "Cerulean Mart",  # CERULEAN_MART
    68: "Mt Moon Pokecenter",  # MT_MOON_POKECENTER
    69: "Cerulean Trashed House Copy",  # CERULEAN_TRASHED_HOUSE_COPY
    70: "Route 5 Gate",  # ROUTE_5_GATE
    71: "Underground Path Route 5",  # UNDERGROUND_PATH_ROUTE_5
    72: "Daycare",  # DAYCARE
    73: "Route 6 Gate",  # ROUTE_6_GATE
    74: "Underground Path Route 6",  # UNDERGROUND_PATH_ROUTE_6
    75: "Underground Path Route 6 Copy",  # UNDERGROUND_PATH_ROUTE_6_COPY
    76: "Route 7 Gate",  # ROUTE_7_GATE
    77: "Underground Path Route 7",  # UNDERGROUND_PATH_ROUTE_7
    78: "Underground Path Route 7 Copy",  # UNDERGROUND_PATH_ROUTE_7_COPY
    79: "Route 8 Gate",  # ROUTE_8_GATE
    80: "Underground Path Route 8",  # UNDERGROUND_PATH_ROUTE_8
    81: "Rock Tunnel Pokecenter",  # ROCK_TUNNEL_POKECENTER
    82: "Rock Tunnel 1F",  # ROCK_TUNNEL_1F
    83: "Power Plant",  # POWER_PLANT
    84: "Route 11 Gate 1F",  # ROUTE_11_GATE_1F
    85: "Diglett's Cave Route 11",  # DIGLETTS_CAVE_ROUTE_11
    86: "Route 11 Gate 2F",  # ROUTE_11_GATE_2F
    87: "Route 12 Gate 1F",  # ROUTE_12_GATE_1F
    88: "Bill's House",  # BILLS_HOUSE
    89: "Vermilion Pokecenter",  # VERMILION_POKECENTER
    90: "Pokemon Fan Club",  # POKEMON_FAN_CLUB
    91: "Vermilion Mart",  # VERMILION_MART
    92: "Vermilion Gym",  # VERMILION_GYM
    93: "Vermilion Pidgey House",  # VERMILION_PIDGEY_HOUSE
    94: "Vermilion Dock",  # VERMILION_DOCK
    95: "S.S. Anne 1F",  # SS_ANNE_1F
    96: "S.S. Anne 2F",  # SS_ANNE_2F
    97: "S.S. Anne 3F",  # SS_ANNE_3F
    98: "S.S. Anne B1F",  # SS_ANNE_B1F
    99: "S.S. Anne Bow",  # SS_ANNE_BOW
    100: "S.S. Anne Kitchen",  # SS_ANNE_KITCHEN
    101: "S.S. Anne Captain's Room",  # SS_ANNE_CAPTAINS_ROOM
    102: "S.S. Anne 1F Rooms",  # SS_ANNE_1F_ROOMS
    103: "S.S. Anne 2F Rooms",  # SS_ANNE_2F_ROOMS
    104: "S.S. Anne B1F Rooms",  # SS_ANNE_B1F_ROOMS
    105: "Unused Map 69",  # UNUSED_MAP_69
    106: "Unused Map 6A",  # UNUSED_MAP_6A
    107: "Unused Map 6B",  # UNUSED_MAP_6B
    108: "Victory Road 1F",  # VICTORY_ROAD_1F
    109: "Unused Map 6D",  # UNUSED_MAP_6D
    110: "Unused Map 6E",  # UNUSED_MAP_6E
    111: "Unused Map 6F",  # UNUSED_MAP_6F
    112: "Unused Map 70",  # UNUSED_MAP_70
    113: "Lance's Room",  # LANCES_ROOM
    114: "Unused Map 72",  # UNUSED_MAP_72
    115: "Unused Map 73",  # UNUSED_MAP_73
    116: "Unused Map 74",  # UNUSED_MAP_74
    117: "Unused Map 75",  # UNUSED_MAP_75
    118: "Hall of Fame",  # HALL_OF_FAME
    119: "Underground Path North South",  # UNDERGROUND_PATH_NORTH_SOUTH
    120: "Champion's Room",  # CHAMPIONS_ROOM
    121: "Underground Path West East",  # UNDERGROUND_PATH_WEST_EAST
    122: "Celadon Dept Store 1F",  # CELADON_MART_1F
    123: "Celadon Dept Store 2F",  # CELADON_MART_2F
    124: "Celadon Dept Store 3F",  # CELADON_MART_3F
    125: "Celadon Dept Store 4F",  # CELADON_MART_4F
    126: "Celadon Dept Store Roof",  # CELADON_MART_ROOF
    127: "Celadon Dept Store Elevator",  # CELADON_MART_ELEVATOR
    128: "Celadon Mansion 1F",  # CELADON_MANSION_1F
    129: "Celadon Mansion 2F",  # CELADON_MANSION_2F
    130: "Celadon Mansion 3F",  # CELADON_MANSION_3F
    131: "Celadon Mansion Roof",  # CELADON_MANSION_ROOF
    132: "Celadon Mansion Roof House",  # CELADON_MANSION_ROOF_HOUSE
    133: "Celadon Pokecenter",  # CELADON_POKECENTER
    134: "Celadon Gym",  # CELADON_GYM
    135: "Game Corner",  # GAME_CORNER
    136: "Celadon Dept Store 5F",  # CELADON_MART_5F
    137: "Game Corner Prize Room",  # GAME_CORNER_PRIZE_ROOM
    138: "Celadon Diner",  # CELADON_DINER
    139: "Celadon Chief House",  # CELADON_CHIEF_HOUSE
    140: "Celadon Hotel",  # CELADON_HOTEL
    141: "Lavender Pokecenter",  # LAVENDER_POKECENTER
    142: "Pokemon Tower 1F",  # POKEMON_TOWER_1F
    143: "Pokemon Tower 2F",  # POKEMON_TOWER_2F
    144: "Pokemon Tower 3F",  # POKEMON_TOWER_3F
    145: "Pokemon Tower 4F",  # POKEMON_TOWER_4F
    146: "Pokemon Tower 5F",  # POKEMON_TOWER_5F
    147: "Pokemon Tower 6F",  # POKEMON_TOWER_6F
    148: "Pokemon Tower 7F",  # POKEMON_TOWER_7F
    149: "Mr Fuji's House",  # MR_FUJIS_HOUSE
    150: "Lavender Mart",  # LAVENDER_MART
    151: "Lavender Cubone House",  # LAVENDER_CUBONE_HOUSE
    152: "Fuchsia Mart",  # FUCHSIA_MART
    153: "Fuchsia Bill's Grandpa's House",  # FUCHSIA_BILLS_GRANDPAS_HOUSE
    154: "Fuchsia Pokecenter",  # FUCHSIA_POKECENTER
    155: "Warden's House",  # WARDENS_HOUSE
    156: "Safari Zone Gate",  # SAFARI_ZONE_GATE
    157: "Fuchsia Gym",  # FUCHSIA_GYM
    158: "Fuchsia Meeting Room",  # FUCHSIA_MEETING_ROOM
    159: "Seafoam Islands B1F",  # SEAFOAM_ISLANDS_B1F
    160: "Seafoam Islands B2F",  # SEAFOAM_ISLANDS_B2F
    161: "Seafoam Islands B3F",  # SEAFOAM_ISLANDS_B3F
    162: "Seafoam Islands B4F",  # SEAFOAM_ISLANDS_B4F
    163: "Vermilion Old Rod House",  # VERMILION_OLD_ROD_HOUSE
    164: "Fuchsia Good Rod House",  # FUCHSIA_GOOD_ROD_HOUSE
    165: "Pokemon Mansion 1F",  # POKEMON_MANSION_1F
    166: "Cinnabar Gym",  # CINNABAR_GYM
    167: "Cinnabar Lab",  # CINNABAR_LAB
    168: "Cinnabar Lab Trade Room",  # CINNABAR_LAB_TRADE_ROOM
    169: "Cinnabar Lab Metronome Room",  # CINNABAR_LAB_METRONOME_ROOM
    170: "Cinnabar Lab Fossil Room",  # CINNABAR_LAB_FOSSIL_ROOM
    171: "Cinnabar Pokecenter",  # CINNABAR_POKECENTER
    172: "Cinnabar Mart",  # CINNABAR_MART
    173: "Cinnabar Mart Copy",  # CINNABAR_MART_COPY
    174: "Indigo Plateau Lobby",  # INDIGO_PLATEAU_LOBBY
    175: "Copycat's House 1F",  # COPYCATS_HOUSE_1F
    176: "Copycat's House 2F",  # COPYCATS_HOUSE_2F
    177: "Fighting Dojo",  # FIGHTING_DOJO
    178: "Saffron Gym",  # SAFFRON_GYM
    179: "Saffron Pidgey House",  # SAFFRON_PIDGEY_HOUSE
    180: "Saffron Mart",  # SAFFRON_MART
    181: "Silph Co 1F",  # SILPH_CO_1F
    182: "Saffron Pokecenter",  # SAFFRON_POKECENTER
    183: "Mr Psychic's House",  # MR_PSYCHICS_HOUSE
    184: "Route 15 Gate 1F",  # ROUTE_15_GATE_1F
    185: "Route 15 Gate 2F",  # ROUTE_15_GATE_2F
    186: "Route 16 Gate 1F",  # ROUTE_16_GATE_1F
    187: "Route 16 Gate 2F",  # ROUTE_16_GATE_2F
    188: "Route 16 Fly House",  # ROUTE_16_FLY_HOUSE
    189: "Route 12 Super Rod House",  # ROUTE_12_SUPER_ROD_HOUSE
    190: "Route 18 Gate 1F",  # ROUTE_18_GATE_1F
    191: "Route 18 Gate 2F",  # ROUTE_18_GATE_2F
    192: "Seafoam Islands 1F",  # SEAFOAM_ISLANDS_1F
    193: "Route 22 Gate",  # ROUTE_22_GATE
    194: "Victory Road 2F",  # VICTORY_ROAD_2F
    195: "Route 12 Gate 2F",  # ROUTE_12_GATE_2F
    196: "Vermilion Trade House",  # VERMILION_TRADE_HOUSE
    197: "Diglett's Cave",  # DIGLETTS_CAVE
    198: "Victory Road 3F",  # VICTORY_ROAD_3F
    199: "Rocket Hideout B1F",  # ROCKET_HIDEOUT_B1F
    200: "Rocket Hideout B2F",  # ROCKET_HIDEOUT_B2F
    201: "Rocket Hideout B3F",  # ROCKET_HIDEOUT_B3F
    202: "Rocket Hideout B4F",  # ROCKET_HIDEOUT_B4F
    203: "Rocket Hideout Elevator",  # ROCKET_HIDEOUT_ELEVATOR
    204: "Unused Map CC",  # UNUSED_MAP_CC
    205: "Unused Map CD",  # UNUSED_MAP_CD
    206: "Unused Map CE",  # UNUSED_MAP_CE
    207: "Silph Co 2F",  # SILPH_CO_2F
    208: "Silph Co 3F",  # SILPH_CO_3F
    209: "Silph Co 4F",  # SILPH_CO_4F
    210: "Silph Co 5F",  # SILPH_CO_5F
    211: "Silph Co 6F",  # SILPH_CO_6F
    212: "Silph Co 7F",  # SILPH_CO_7F
    213: "Silph Co 8F",  # SILPH_CO_8F
    214: "Pokemon Mansion 2F",  # POKEMON_MANSION_2F
    215: "Pokemon Mansion 3F",  # POKEMON_MANSION_3F
    216: "Pokemon Mansion B1F",  # POKEMON_MANSION_B1F
    217: "Safari Zone East",  # SAFARI_ZONE_EAST
    218: "Safari Zone North",  # SAFARI_ZONE_NORTH
    219: "Safari Zone West",  # SAFARI_ZONE_WEST
    220: "Safari Zone Center",  # SAFARI_ZONE_CENTER
    221: "Safari Zone Center Rest House",  # SAFARI_ZONE_CENTER_REST_HOUSE
    222: "Safari Zone Secret House",  # SAFARI_ZONE_SECRET_HOUSE
    223: "Safari Zone West Rest House",  # SAFARI_ZONE_WEST_REST_HOUSE
    224: "Safari Zone East Rest House",  # SAFARI_ZONE_EAST_REST_HOUSE
    225: "Safari Zone North Rest House",  # SAFARI_ZONE_NORTH_REST_HOUSE
    226: "Cerulean Cave 2F",  # CERULEAN_CAVE_2F
    227: "Cerulean Cave B1F",  # CERULEAN_CAVE_B1F
    228: "Cerulean Cave 1F",  # CERULEAN_CAVE_1F
    229: "Name Rater's House",  # NAME_RATERS_HOUSE
    230: "Cerulean Badge House",  # CERULEAN_BADGE_HOUSE
    231: "Unused Map E7",  # UNUSED_MAP_E7
    232: "Rock Tunnel B1F",  # ROCK_TUNNEL_B1F
    233: "Silph Co 9F",  # SILPH_CO_9F
    234: "Silph Co 10F",  # SILPH_CO_10F
    235: "Silph Co 11F",  # SILPH_CO_11F
    236: "Silph Co Elevator",  # SILPH_CO_ELEVATOR
    237: "Unused Map ED",  # UNUSED_MAP_ED
    238: "Unused Map EE",  # UNUSED_MAP_EE
    239: "Trade Center",  # TRADE_CENTER
    240: "Colosseum",  # COLOSSEUM
    241: "Unused Map F1",  # UNUSED_MAP_F1
    242: "Unused Map F2",  # UNUSED_MAP_F2
    243: "Unused Map F3",  # UNUSED_MAP_F3
    244: "Unused Map F4",  # UNUSED_MAP_F4
    245: "Lorelei's Room",  # LORELEIS_ROOM
    246: "Bruno's Room",  # BRUNOS_ROOM
    247: "Agatha's Room",  # AGATHAS_ROOM
}
# fmt: on

_STATUS_TABLE = {
    0: "OK",
    # lower 3 bits = sleep counter (1-7 = asleep)
    # bit 3 = poison, bit 4 = burn, bit 5 = freeze, bit 6 = paralysis
}

FACING_NAMES: Dict[int, str] = {
    0x00: "down",
    0x04: "up",
    0x08: "left",
    0x0C: "right",
}

TILESET_NAMES: Dict[int, str] = {
    0x00: "OVERWORLD",
    0x01: "REDS_HOUSE_1",
    0x02: "MART",
    0x03: "FOREST",
    0x04: "REDS_HOUSE_2",
    0x05: "DOJO",
    0x06: "POKECENTER",
    0x07: "GYM",
    0x08: "HOUSE",
    0x09: "FOREST_GATE",
    0x0A: "MUSEUM",
    0x0B: "UNDERGROUND",
    0x0C: "GATE",
    0x0D: "SHIP",
    0x0E: "SHIP_PORT",
    0x0F: "CEMETERY",
    0x10: "INTERIOR",
    0x11: "CAVERN",
    0x12: "LOBBY",
    0x13: "MANSION",
    0x14: "LAB",
    0x15: "CLUB",
    0x16: "FACILITY",
    0x17: "PLATEAU",
}

TALK_OVER_TILES: Dict[str, set[int]] = {
    "MART": {0x18, 0x19, 0x1E},
    "DOJO": {0x3A},
    "POKECENTER": {0x18, 0x19, 0x1E},
    "GYM": {0x3A},
    "FOREST_GATE": {0x17, 0x32},
    "MUSEUM": {0x17, 0x32},
    "GATE": {0x17, 0x32},
    "CEMETERY": {0x12},
    "LOBBY": {0x15, 0x36},
    "CLUB": {0x07, 0x17},
    "FACILITY": {0x12},
}

BADGE_NAMES = [
    "Boulder",
    "Cascade",
    "Thunder",
    "Rainbow",
    "Soul",
    "Marsh",
    "Volcano",
    "Earth",
]


# ===================================================================
# Reader implementation
# ===================================================================


class RedBlueMemoryReader(GameMemoryReader):
    """Memory reader for *Pokemon Red* and *Pokemon Blue* (USA).

    Parameters
    ----------
    emulator : Emulator
        A loaded PyBoyEmulator running a Red/Blue ROM.
    """

    @property
    def game_name(self) -> str:
        return "Pokemon Red/Blue (USA)"

    # -- helpers --

    def _decode_text(self, addr: int, max_len: int = 11) -> str:
        """Decode a Gen-1 encoded string from RAM."""
        return self.read_string(addr, max_len, GEN1_ENCODING, terminator=0x50)

    def _decode_status(self, status_byte: int) -> str:
        """Return a human-readable status string."""
        if status_byte == 0:
            return "OK"
        parts = []
        sleep = status_byte & 0x07
        if sleep:
            parts.append(f"SLP({sleep})")
        if status_byte & 0x08:
            parts.append("PSN")
        if status_byte & 0x10:
            parts.append("BRN")
        if status_byte & 0x20:
            parts.append("FRZ")
        if status_byte & 0x40:
            parts.append("PAR")
        return "/".join(parts) if parts else "OK"

    @staticmethod
    def _decode_types(type1: int, type2: int) -> List[str]:
        """Decode the two type bytes, collapsing the pair a mono-type stores twice."""
        types = [TYPE_NAMES.get(type1, f"???({type1})")]
        if type2 != type1:
            types.append(TYPE_NAMES.get(type2, f"???({type2})"))
        return types

    def _decode_species(self, species_id: int) -> Dict[str, Any]:
        """Decode a Gen 1 internal species index to stable species metadata."""
        pokedex_id = INTERNAL_SPECIES_TO_DEX.get(species_id)
        if pokedex_id is None:
            return {
                "species_id": species_id,
                "pokedex_id": None,
                "species": f"???({species_id})",
            }
        if pokedex_id == 0:
            return {
                "species_id": species_id,
                "pokedex_id": None,
                "species": "MissingNo.",
            }
        return {
            "species_id": species_id,
            "pokedex_id": pokedex_id,
            "species": SPECIES_NAMES[pokedex_id],
        }

    def _read_pokemon(self, base: int, nick_addr: int) -> Dict[str, Any]:
        """Parse a 44-byte party Pokemon structure at *base*.

        Layout (offsets from base):
          0:  species (1)
          1:  current HP (2, big-endian)
          3:  level (box level, sometimes called 'box level')
          4:  status condition (1)
          5:  type 1 (1)
          6:  type 2 (1)
          7:  catch rate / held item (1)
          8:  move 1 (1)
          9:  move 2 (1)
          10: move 3 (1)
          11: move 4 (1)
          12: OT ID (2, big-endian)
          14: experience (3, big-endian)
          17: HP EV (2, big-endian)
          19: Attack EV (2)
          21: Defense EV (2)
          23: Speed EV (2)
          25: Special EV (2)
          27: IV data (2)
          29: PP move 1 (1)
          30: PP move 2 (1)
          31: PP move 3 (1)
          32: PP move 4 (1)
          ---- party-exclusive fields ----
          33: level (1, actual party level)
          34: max HP (2, big-endian)
          36: attack (2, big-endian)
          38: defense (2, big-endian)
          40: speed (2, big-endian)
          42: special (2, big-endian)
        """
        data = self.emu.read_range(base, PARTY_MON_SIZE)
        species = self._decode_species(data[0])
        nickname = self._decode_text(nick_addr, 11)

        moves = []
        for i in range(4):
            mid = data[8 + i]
            if mid != 0:
                moves.append(
                    {
                        "id": mid,
                        "name": MOVE_NAMES.get(mid, f"???({mid})"),
                        "pp": data[29 + i] & 0x3F,
                        "pp_up": (data[29 + i] >> 6) & 0x03,
                    }
                )

        return {
            **species,
            "nickname": nickname,
            "level": data[33],
            "hp": (data[1] << 8) | data[2],
            "max_hp": (data[34] << 8) | data[35],
            "status": self._decode_status(data[4]),
            "types": self._decode_types(data[5], data[6]),
            "moves": moves,
            "stats": {
                "attack": (data[36] << 8) | data[37],
                "defense": (data[38] << 8) | data[39],
                "speed": (data[40] << 8) | data[41],
                "special": (data[42] << 8) | data[43],
            },
            "ot_id": (data[12] << 8) | data[13],
            "experience": (data[14] << 16) | (data[15] << 8) | data[16],
        }

    def _read_battle_struct(self, base: int) -> Dict[str, Any]:
        """Parse one live 29-byte Gen 1 ``battle_struct`` at *base*.

        Layout (offsets from base), different from the 44-byte party struct::

          0:  species (1)
          1:  current HP (2, big-endian)
          3:  box level (1)
          4:  status (1)
          5:  type 1 (1)
          6:  type 2 (1)
          7:  catch rate (1)
          8:  moves (4)
          12: DVs (2)
          14: level (1)
          15: max HP (2)      17: attack (2)   19: defense (2)
          21: speed (2)       23: special (2)
          25: PP (4)

        The stats block from offset 17 is the one the engine actually fights
        with: it is the party stats *after* badge boosts and after every stat
        stage the fight has applied. Reading it is the difference between a
        damage calculation and a guess. Measured over 123 auto-saved battle
        frames from one run: the player's Attack differed from the party struct
        in **every one** of them (50 -> 56 from the Boulder Badge alone, and
        54 -> 121 once Rage had built up), and re-deriving the enemy's stats
        from base stats at an average DV was wrong for 196 of 492 reads, worst
        case a Geodude whose Defense read 22 when Leer had already cut it to 14.
        """
        data = self.emu.read_range(base, BATTLE_MON_SIZE)

        def word(offset: int) -> int:
            return (data[offset] << 8) | data[offset + 1]

        return {
            **self._decode_species(data[0]),
            "level": data[14],
            "hp": word(1),
            "max_hp": word(15),
            "status": self._decode_status(data[4]),
            "types": self._decode_types(data[5], data[6]),
            "stats": {
                "attack": word(17),
                "defense": word(19),
                "speed": word(21),
                "special": word(23),
            },
        }

    def _read_enemy_battle_mon(self) -> Dict[str, Any]:
        """The enemy side of the field. Moves are names: it has no PP worth showing."""
        data = self.emu.read_range(ADDR_ENEMY_MON, BATTLE_MON_SIZE)
        moves: List[str] = [
            MOVE_NAMES.get(data[8 + index], f"???({data[8 + index]})")
            for index in range(4)
            if data[8 + index] != 0
        ]
        return {
            **self._read_battle_struct(ADDR_ENEMY_MON),
            "moves": moves,
            # Half of whether this one is worth a ball, and free to read here.
            # Behind a separate call it is a number nobody ever asks for.
            "catch_rate": self.enemy_catch_rate(),
        }

    # -- public interface ---------------------------------------------------

    def read_player(self) -> Dict[str, Any]:
        """Read player info: name, money, badges, position, facing, play time."""
        name = self._decode_text(ADDR_PLAYER_NAME, 11)
        rival = self._decode_text(ADDR_RIVAL_NAME, 11)
        money = self.read_bcd(ADDR_MONEY, 3)

        badge_byte = self.emu.read_u8(ADDR_BADGES)
        badge_list = [BADGE_NAMES[i] for i in range(8) if badge_byte & (1 << i)]

        map_y = self.emu.read_u8(ADDR_MAP_Y)
        map_x = self.emu.read_u8(ADDR_MAP_X)
        facing_byte = self.emu.read_u8(ADDR_FACING)
        facing = FACING_NAMES.get(facing_byte, f"unknown(0x{facing_byte:02X})")

        hours = self.emu.read_u8(ADDR_PLAYTIME_H)
        minutes = self.emu.read_u8(ADDR_PLAYTIME_M)
        seconds = self.emu.read_u8(ADDR_PLAYTIME_S)

        return {
            "name": name,
            "rival_name": rival,
            "money": money,
            "badges": badge_list,
            "badge_count": len(badge_list),
            "position": {"y": map_y, "x": map_x},
            "facing": facing,
            "play_time": f"{hours}:{minutes:02d}:{seconds:02d}",
        }

    def read_party(self) -> List[Dict[str, Any]]:
        """Read the player's party (up to 6 Pokemon)."""
        count = self.emu.read_u8(ADDR_PARTY_COUNT)
        count = min(count, 6)
        party: List[Dict[str, Any]] = []
        for i in range(count):
            base = ADDR_PARTY_DATA + i * PARTY_MON_SIZE
            nick_addr = ADDR_PARTY_NICKS + i * 11
            party.append(self._read_pokemon(base, nick_addr))
        return party

    def read_bag(self) -> List[Dict[str, Any]]:
        """Read bag item list."""
        count = self.emu.read_u8(ADDR_BAG_COUNT)
        count = min(count, 20)  # bag max 20 items in Gen 1
        items: List[Dict[str, Any]] = []
        for i in range(count):
            item_id = self.emu.read_u8(ADDR_BAG_ITEMS + i * 2)
            qty = self.emu.read_u8(ADDR_BAG_ITEMS + i * 2 + 1)
            if item_id == 0xFF:  # terminator
                break
            items.append(
                {
                    "id": item_id,
                    "item": ITEM_NAMES.get(item_id, f"???({item_id})"),
                    "quantity": qty,
                }
            )
        return items

    def bag_index_of(self, item_id: int) -> Optional[int]:
        """Where an item sits in the bag list, or None if it is not carried.

        The row number is the target of the cursor walk, and nothing else knows
        that the Town Map sits above the Poke Ball. It is *not* the number of
        Down presses on its own: the list remembers where it was left, so the
        walk is the difference between this row and the one it opens on.
        """
        for index, entry in enumerate(self.read_bag()):
            if entry["id"] == item_id:
                return index
        return None

    def read_list_menu(self) -> Dict[str, Any]:
        """The open item list, as the row the cursor is really on.

        ``index`` is the entry in the *list*, which is what a caller counting
        Down presses needs: the cursor stops at row 2 and the window scrolls
        underneath it, so wCurrentMenuItem alone reads 2 for every entry from
        the third one down.
        """
        menu_id = self.emu.read_u8(ADDR_LIST_MENU_ID)
        row = self.emu.read_u8(ADDR_CURRENT_MENU_ITEM)
        scroll = self.emu.read_u8(ADDR_LIST_SCROLL_OFFSET)
        return {
            "menu_id": menu_id,
            "open": self.emu.read_u8(ADDR_TEXT_BOX_ID) == LIST_MENU_TEXT_BOX_ID
            and menu_id in (PRICED_ITEM_LIST_MENU, ITEM_LIST_MENU),
            "index": row + scroll,
            "count": self.emu.read_u8(ADDR_LIST_COUNT),
        }

    def at_bag_list(self) -> bool:
        """Is the plain item list — the bag — the thing taking input?"""
        menu = self.read_list_menu()
        return bool(menu["open"]) and menu["menu_id"] == ITEM_LIST_MENU

    def at_mart_counter(self) -> bool:
        """Is a mart's priced list — its BUY menu — the thing taking input?"""
        menu = self.read_list_menu()
        return bool(menu["open"]) and menu["menu_id"] == PRICED_ITEM_LIST_MENU

    def selected_item_id(self) -> int:
        """The item id the last A press confirmed on an item list.

        A postcondition for a purchase, where nothing else touches the byte, and
        **not** one for a throw: a successful capture runs the nickname and
        Pokedex routines through the same address, so it holds something
        unrelated by the time the ball has finished wobbling.
        """
        return self.emu.read_u8(ADDR_CUR_ITEM)

    def item_quantity(self) -> int:
        """The number showing in a mart's quantity roller."""
        return self.emu.read_u8(ADDR_ITEM_QUANTITY)

    def at_quantity_roller(self) -> bool:
        """Is the ``x01  $200`` counter the thing Up and Down are driving?

        The roller shares its text box with the list underneath it, so what
        separates them is wMaxItemQuantity: the list leaves it at 1 and opening
        the roller sets it to how many you may buy.
        """
        return (
            self.emu.read_u8(ADDR_TEXT_BOX_ID) == LIST_MENU_TEXT_BOX_ID
            and self.emu.read_u8(ADDR_MAX_ITEM_QUANTITY) > 1
        )

    def at_purchase_prompt(self) -> bool:
        """Is "That will be $N. OK?" up, waiting on YES or NO?"""
        return self.emu.read_u8(ADDR_TEXT_BOX_ID) == TWO_OPTION_TEXT_BOX_ID

    def at_buy_sell_quit(self) -> bool:
        """Is the clerk's BUY / SELL / QUIT menu up, with BUY under the cursor?

        The screen between talking to a clerk and seeing anything for sale. A
        purchase has to wait for it rather than count presses through it: the
        greeting before it is a text box that can take one press or two.
        """
        return self.emu.read_u8(ADDR_TEXT_BOX_ID) == BUY_SELL_QUIT_TEXT_BOX_ID

    def read_shop_list(self) -> List[int]:
        """The item ids a mart counter is offering, in the order it lists them.

        Read off wItemList rather than taken from the static table, because the
        cursor walk has to match the rows actually on screen. Only meaningful
        while a counter is open.
        """
        data = self.emu.read_range(ADDR_ITEM_LIST, 18)
        count = min(data[0], 16)
        ids: List[int] = []
        for value in data[1 : 1 + count]:
            if value == 0xFF:
                break
            ids.append(value)
        return ids

    def enemy_catch_rate(self) -> int:
        """The live catch rate of the Pokemon on the other side.

        The engine's own byte, not the species table's: bait and rocks move it,
        and reading it costs nothing while re-deriving it can be wrong.
        """
        return self.emu.read_u8(ADDR_ENEMY_CATCH_RATE)

    def read_pc_items(self) -> List[Dict[str, Any]]:
        """Read the item PC's list, same shape as :meth:`read_bag`.

        Gen 1 has no key-item pocket, so the Card Key, Lift Key, Silph Scope and
        Secret Key can all be deposited here. A milestone that only looked in the
        bag would un-fire the moment the player tidied up.
        """
        count = min(self.emu.read_u8(ADDR_PC_COUNT), PC_ITEM_CAPACITY)
        items: List[Dict[str, Any]] = []
        for i in range(count):
            item_id = self.emu.read_u8(ADDR_PC_ITEMS + i * 2)
            qty = self.emu.read_u8(ADDR_PC_ITEMS + i * 2 + 1)
            if item_id == 0xFF:  # terminator
                break
            items.append(
                {
                    "id": item_id,
                    "item": ITEM_NAMES.get(item_id, f"???({item_id})"),
                    "quantity": qty,
                }
            )
        return items

    def read_battle(self) -> Dict[str, Any]:
        """Read battle state (whether in battle & enemy info)."""
        battle_type = self.emu.read_u8(ADDR_BATTLE_TYPE)
        type_name = {0: "none", 1: "wild", 2: "trainer"}.get(battle_type, f"unknown({battle_type})")
        result: Dict[str, Any] = {
            "in_battle": battle_type != 0,
            "type": type_name,
        }
        if battle_type != 0:
            result["enemy"] = self._read_enemy_battle_mon()
        return result

    def read_battle_moves(self) -> List[Dict[str, Any]]:
        """Moves of the Pokemon currently *on the field*, in move-list order.

        Read from ``wBattleMon`` rather than party slot 0 so a switched-in Pokemon
        reports its own moves. Verified identical to ``read_party()[0]["moves"]``
        while the lead is out, and identical to what the move list draws.
        """
        ids = self.emu.read_range(ADDR_BATTLE_MON_MOVES, 4)
        pps = self.emu.read_range(ADDR_BATTLE_MON_PP, 4)
        return [
            {"id": move_id, "name": MOVE_NAMES.get(move_id, f"???({move_id})"), "pp": pps[i] & 0x3F}
            for i, move_id in enumerate(ids)
            if move_id != 0
        ]

    def read_battle_mon(self) -> Dict[str, Any]:
        """The Pokemon *on the field* on your side, with the stats the game fights with.

        ``read_party()[0]`` is not this Pokemon after a switch, and its stats are
        not these stats even before one: the party struct holds the unboosted
        numbers, and the engine copies them into ``wBattleMon`` and then applies
        badge boosts and every stat stage of the fight. Damage arithmetic done
        against the party struct is arithmetic about a Pokemon that is not in the
        battle. See :meth:`_read_battle_struct` for what that cost.
        """
        return {**self._read_battle_struct(ADDR_BATTLE_MON), "moves": self.read_battle_moves()}

    def battle_lock_in(self) -> Optional[str]:
        """The move that has taken the turn away, or None if the menu is yours.

        Gen 1 has several states in which the engine chooses for you and the top
        battle menu simply never comes back: Rage, Thrash, Bide, a charging Sky
        Attack, a Hyper Beam recharge. Only Rage is reported here, because Rage
        is the only one this party could produce and this project does not put
        an unmeasured RAM bit in a decision path.

        It is the one worth reporting anyway. One run used Rage 77 times, and
        every battle command after one spent 24 B presses failing to find a menu
        and then got told the fight was probably over — which was not true.
        """
        if self.emu.read_u8(ADDR_PLAYER_BATTLE_STATUS_2) & USED_RAGE_BIT:
            return "Rage"
        return None

    def at_battle_top_menu(self) -> bool:
        """Is the FIGHT/PKMN/ITEM/RUN menu the thing on screen and taking input?

        This is the only trustworthy "the game is waiting for my turn" signal.
        ``read_dialog()["active"]`` is not: during the battle intro it goes true,
        then *false* for about a second while the sprites slide in, then true
        again on "Wild X appeared!". A wait loop on it exits into the animation.
        """
        return (
            self.emu.read_u8(ADDR_TEXT_BOX_ID) == BATTLE_MENU_TEXT_BOX_ID
            and self.emu.read_u8(ADDR_TOP_MENU_ITEM_Y) == TOP_MENU_ITEM_Y
            and self.emu.read_u8(ADDR_TOP_MENU_ITEM_X) in TOP_MENU_COLUMN_X
            and self.emu.read_u8(ADDR_MAX_MENU_ITEM) == TOP_MENU_MAX_ITEM
        )

    def at_battle_move_menu(self) -> bool:
        """Is the move list open?"""
        return (
            self.emu.read_u8(ADDR_TEXT_BOX_ID) == BATTLE_MENU_TEXT_BOX_ID
            and self.emu.read_u8(ADDR_TOP_MENU_ITEM_Y) == MOVE_MENU_ITEM_Y
            and self.emu.read_u8(ADDR_TOP_MENU_ITEM_X) == MOVE_MENU_ITEM_X
        )

    def remembered_move_index(self) -> int:
        """Where the move cursor will be when the move list is next opened.

        The list remembers its previous position, and it wraps at both ends, so
        this is the only way to know what a blind A press would fire.
        """
        return self.emu.read_u8(ADDR_PLAYER_MOVE_LIST_INDEX)

    def selected_move_id(self) -> int:
        """The move id the current turn will use. Tracks the cursor, survives the
        confirming A press — so it is the postcondition worth checking."""
        return self.emu.read_u8(ADDR_PLAYER_SELECTED_MOVE)

    def read_battle_menu(self) -> Dict[str, Any]:
        """Which battle menu is open and which entry the cursor sits on.

        Facts about the screen, nothing more: ``menu`` is ``top``, ``moves`` or
        ``other``, and ``highlighted`` is the entry A would pick right now.
        """
        if self.at_battle_top_menu():
            row = 1 if self.emu.read_u8(ADDR_CURRENT_MENU_ITEM) else 0
            column = 1 if self.emu.read_u8(ADDR_TOP_MENU_ITEM_X) == TOP_MENU_COLUMN_X[1] else 0
            return {"menu": "top", "highlighted": TOP_MENU_ENTRIES[row][column], "index": None}
        if self.at_battle_move_menu():
            index = self.emu.read_u8(ADDR_CURRENT_MENU_ITEM) - 1
            moves = self.read_battle_moves()
            name = moves[index]["name"] if 0 <= index < len(moves) else None
            return {"menu": "moves", "highlighted": name, "index": index if name else None}
        return {"menu": "other", "highlighted": None, "index": None}

    def read_dialog(self) -> Dict[str, Any]:
        """Read dialogue / text box state.

        A Gen 1 prompt has two phases:
        - text is actively printing, during which ``wJoyIgnore`` masks input
        - text is fully drawn and waiting for A/B, during which the dialogue
          window remains visible even if ``wJoyIgnore`` has already been cleared

        ``wTextBoxID`` alone is unreliable because it can remain non-zero after
        the prompt closes, so the active flag is derived from either:
        - ``wJoyIgnore`` bit 5 while text is printing, or
        - the dialogue window still being visible on-screen.
        """
        text_box = self.emu.read_u8(ADDR_TEXT_BOX_ID)
        joy_ignore = self.emu.read_u8(ADDR_JOY_IGNORE)
        window_y = self.emu.read_u8(ADDR_WINDOW_Y)
        window_x = self.emu.read_u8(ADDR_WINDOW_X)
        printing = bool(joy_ignore & 0x20)
        window_visible = window_y < 0x90
        waiting_for_input = window_visible and text_box != 0 and not printing
        in_dialog = printing or waiting_for_input
        return {
            "active": in_dialog,
            "text_box_id": text_box,
            "joy_ignore": joy_ignore,
            "window_visible": window_visible,
            "window_y": window_y,
            "window_x": window_x,
            "printing": printing,
            "waiting_for_input": waiting_for_input,
        }

    def read_screen_text(self) -> str:
        """The words on screen right now, decoded off ``wTileMap``.

        One line per tile row, trailing blanks trimmed, blank rows kept so a
        menu column and the dialogue box under it stay apart. Tiles that are not
        letters -- the map, the sprites, the box borders -- come back as spaces,
        so an overworld frame decodes to nothing and only a box has words in it.
        """
        raw = self.emu.read_range(ADDR_TILE_MAP, TILE_MAP_WIDTH * TILE_MAP_HEIGHT)
        rows = []
        for index in range(TILE_MAP_HEIGHT):
            row = raw[index * TILE_MAP_WIDTH : (index + 1) * TILE_MAP_WIDTH]
            rows.append("".join(GEN1_ENCODING.get(byte, " ") for byte in row).rstrip())
        return "\n".join(rows).strip("\n")

    def read_move_learn(self) -> Optional[Dict[str, Any]]:
        """The move-replacement prompt on screen, or None when there is none.

        Detected from the words themselves rather than from a flag byte: a
        sweep of the whole WRAM block across the flow found no single byte that
        is set for all of it and clear before and after, and the six phrases
        below cover every frame from "CHAR is trying to learn DIG!" to "Which
        move should be forgotten?".

        ``cursor`` is only meaningful on the forget list, where ``wMaxMenuItem``
        reads 3 and ``wCurrentMenuItem`` tracks the highlighted move: pressing
        down took it 0 -> 1 -> 2 against the arrow on screen.
        """
        from pokemon_agent.party import is_learn_prompt

        text = self.read_screen_text()
        if not is_learn_prompt(text):
            return None
        incoming_id = self.emu.read_u8(ADDR_MOVE_BEING_LEARNED)
        return {
            "screen_text": text,
            "incoming": MOVE_NAMES.get(incoming_id),
            "slot": self.emu.read_u8(ADDR_WHICH_POKEMON),
            "cursor": self.emu.read_u8(ADDR_CURRENT_MENU_ITEM),
        }

    def read_coordinates(self) -> tuple[int, int]:
        """Read the player's current map coordinates as ``(x, y)``."""
        return (
            self.emu.read_u8(ADDR_MAP_X),
            self.emu.read_u8(ADDR_MAP_Y),
        )

    def read_facing(self) -> str:
        """Read the player's live facing direction."""
        facing_byte = self.emu.read_u8(ADDR_FACING)
        return FACING_NAMES.get(facing_byte, f"unknown(0x{facing_byte:02X})")

    def read_tileset(self) -> str:
        """Read the current map's tileset."""
        tileset_id = self.emu.read_u8(ADDR_MAP_TILESET)
        return TILESET_NAMES.get(tileset_id, f"UNKNOWN_TILESET({tileset_id})")

    def read_warps(self) -> List[Dict[str, int]]:
        """Read warp entries for the current map."""
        count = self.emu.read_u8(ADDR_WARP_COUNT)
        count = min(count, 32)
        warps: List[Dict[str, int]] = []
        for index in range(count):
            base = ADDR_WARP_ENTRIES + index * 4
            warps.append(
                {
                    "y": self.emu.read_u8(base),
                    "x": self.emu.read_u8(base + 1),
                    "warp_id": self.emu.read_u8(base + 2),
                    "target_map_id": self.emu.read_u8(base + 3),
                }
            )
        return warps

    def read_signs(self) -> List[Dict[str, int]]:
        """Read sign/background-event coordinates for the current map."""
        count = min(self.emu.read_u8(ADDR_SIGN_COUNT), MAX_SIGNS)
        signs: List[Dict[str, int]] = []
        for index in range(count):
            coord_base = ADDR_SIGN_COORDS + index * 2
            signs.append(
                {
                    "y": self.emu.read_u8(coord_base),
                    "x": self.emu.read_u8(coord_base + 1),
                    "text_id": self.emu.read_u8(ADDR_SIGN_TEXT_IDS + index),
                }
            )
        return signs

    def read_talk_over_tiles(self) -> List[int]:
        """Read tiles that permit talking over the front tile, like counters."""
        return sorted(TALK_OVER_TILES.get(self.read_tileset(), set()))

    def read_map_dimensions(self) -> Dict[str, int]:
        """Read current map dimensions with both block and tile units."""
        width_blocks = self.emu.read_u8(ADDR_MAP_WIDTH)
        height_blocks = self.emu.read_u8(ADDR_MAP_HEIGHT)
        width_tiles = width_blocks * 2
        height_tiles = height_blocks * 2
        return {
            "width": width_tiles,
            "height": height_tiles,
            "width_blocks": width_blocks,
            "height_blocks": height_blocks,
            "width_tiles": width_tiles,
            "height_tiles": height_tiles,
        }

    def read_map_info(self) -> Dict[str, Any]:
        """Read current map id and name."""
        map_id = self.emu.read_u8(ADDR_MAP_ID)
        return {
            "map_id": map_id,
            "map_name": MAP_NAMES.get(map_id, f"Unknown Map ({map_id})"),
        }

    def read_flags(self) -> Dict[str, Any]:
        """Read key story / event flags."""
        badges = self.emu.read_u8(ADDR_BADGES)

        # Pokedex count
        owned_bits = self.read_bits(ADDR_DEX_OWNED, 19)
        seen_bits = self.read_bits(ADDR_DEX_SEEN, 19)
        dex_owned = sum(owned_bits[:151])
        dex_seen = sum(seen_bits[:151])

        # Story flags — some common checks
        oak_parcel_byte = self.emu.read_u8(ADDR_OAK_PARCEL)
        pokedex_byte = self.emu.read_u8(ADDR_POKEDEX_FLAG)

        gym_leaders_defeated = [BADGE_NAMES[i] for i in range(8) if badges & (1 << i)]

        return {
            "has_pokedex": bool(pokedex_byte & 0x20),
            "has_oaks_parcel": bool(oak_parcel_byte & 0x02),
            "pokedex_owned": dex_owned,
            "pokedex_seen": dex_seen,
            "badges": gym_leaders_defeated,
            "badge_count": len(gym_leaders_defeated),
        }


# Alias used by server.py and README examples
PokemonRedReader = RedBlueMemoryReader
