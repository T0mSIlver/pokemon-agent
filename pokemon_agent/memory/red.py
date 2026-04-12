"""Pokemon Red / Blue (USA) memory reader.

All RAM addresses come from the *pokered* decomp project
(https://github.com/pret/pokered).  This module targets the
USA Rev-A ROM but most offsets are identical for Rev-0 and Blue.

Gen 1 text uses a custom character encoding (0x50 = terminator,
0x80..0x99 = uppercase A-Z, etc.).  Money is stored as 3-byte BCD.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pokemon_agent.memory.reader import GameMemoryReader

# ===================================================================
# RAM addresses (WRAM)
# ===================================================================

# -- Player --
ADDR_PLAYER_NAME   = 0xD158   # 11 bytes
ADDR_RIVAL_NAME    = 0xD34A   # 11 bytes
ADDR_MONEY         = 0xD347   # 3 bytes BCD
ADDR_BADGES        = 0xD356   # 1 byte bitmask

# -- Position --
ADDR_MAP_ID        = 0xD35E   # current map number (wCurMap)
ADDR_MAP_Y         = 0xD361   # player Y on map  (wYCoord)
ADDR_MAP_X         = 0xD362   # player X on map  (wXCoord)
ADDR_MAP_BANK      = 0xD35E   # same byte is map id
ADDR_MAP_TILESET   = 0xD367   # current map tileset (wCurMapTileset)
ADDR_MAP_HEIGHT    = 0xD368   # current map height in blocks
ADDR_MAP_WIDTH     = 0xD369   # current map width in blocks
ADDR_FACING        = 0xC109   # live sprite facing direction
# 0xC109 is the live sprite state that updates as the player moves.

# -- Party --
ADDR_PARTY_COUNT   = 0xD163
ADDR_PARTY_SPECIES = 0xD164   # 6 bytes + terminator
ADDR_PARTY_DATA    = 0xD16B   # 44 bytes per slot × 6
ADDR_PARTY_OT      = 0xD273   # 11 bytes per OT × 6
ADDR_PARTY_NICKS   = 0xD2B5   # 11 bytes per nick × 6

PARTY_MON_SIZE     = 44
BATTLE_MON_SIZE    = 29

# -- Bag --
ADDR_BAG_COUNT     = 0xD31D
ADDR_BAG_ITEMS     = 0xD31E   # pairs (item_id, qty)

# -- PC items --
ADDR_PC_COUNT      = 0xD53A
ADDR_PC_ITEMS      = 0xD53B

# -- Battle --
ADDR_BATTLE_TYPE   = 0xD057   # 0=none, 1=wild, 2=trainer
ADDR_ENEMY_COUNT   = 0xD89C
ADDR_ENEMY_SPECIES = 0xD89D
ADDR_ENEMY_DATA    = 0xD8A4   # 44 bytes per mon (party struct)
ADDR_ENEMY_MON     = 0xCFE5   # active enemy battle mon (live HP/stats)

# -- Dialog --
ADDR_TEXT_BOX_ID   = 0xD125   # wTextBoxID
ADDR_JOY_IGNORE    = 0xD730   # bit 5 = joypad disabled (in dialogue)
ADDR_TEXT_PROGRESS  = 0xC4F2  # approximate; nonzero when text printing
ADDR_WINDOW_Y      = 0xFF4A   # hWY / rWY, window Y position (0x90 = hidden)
ADDR_WINDOW_X      = 0xFF4B   # hWX / rWX, window X position

# -- Pokedex --
ADDR_DEX_OWNED     = 0xD2F7   # 19 bytes (152 bits, only 151 used)
ADDR_DEX_SEEN      = 0xD30A

# -- Play time --
ADDR_PLAYTIME_H    = 0xDA40   # 2 bytes (little-endian hours)
ADDR_PLAYTIME_M    = 0xDA42   # 1 byte minutes
ADDR_PLAYTIME_S    = 0xDA43   # 1 byte seconds
ADDR_PLAYTIME_F    = 0xDA44   # 1 byte frames

# -- Event / story flags --
ADDR_EVENT_FLAGS   = 0xD747   # large bitfield (wEventFlags)
ADDR_OAK_PARCEL    = 0xD74E   # bit 1 = has parcel
ADDR_POKEDEX_FLAG  = 0xD74B   # bit 5 = has pokedex
ADDR_TOWN_MAP_FLAG = 0xD5F3   # bit 0 = has town map

# -- Warps --
ADDR_WARP_COUNT    = 0xD3AE   # current map warp count (wNumberOfWarps)
ADDR_WARP_ENTRIES  = 0xD3AF   # current map warp entries (Y, X, warp ID, map ID)

# -- Signs / background events --
ADDR_SIGN_COUNT    = 0xD4B0   # current map sign count (wNumSigns)
ADDR_SIGN_COORDS   = 0xD4B1   # current map sign coordinates (wSignCoords; Y, X)
ADDR_SIGN_TEXT_IDS = 0xD4D1   # current map sign text ids (wSignTextIDs)
MAX_SIGNS          = 16


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
    # punctuation / specials
    t[0x7F] = " "
    t[0xE0] = "'"
    t[0xE1] = "P" # PK
    t[0xE2] = "M" # MN
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
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur",
    4: "Charmander", 5: "Charmeleon", 6: "Charizard",
    7: "Squirtle", 8: "Wartortle", 9: "Blastoise",
    10: "Caterpie", 11: "Metapod", 12: "Butterfree",
    13: "Weedle", 14: "Kakuna", 15: "Beedrill",
    16: "Pidgey", 17: "Pidgeotto", 18: "Pidgeot",
    19: "Rattata", 20: "Raticate",
    21: "Spearow", 22: "Fearow",
    23: "Ekans", 24: "Arbok",
    25: "Pikachu", 26: "Raichu",
    27: "Sandshrew", 28: "Sandslash",
    29: "Nidoran♀", 30: "Nidorina", 31: "Nidoqueen",
    32: "Nidoran♂", 33: "Nidorino", 34: "Nidoking",
    35: "Clefairy", 36: "Clefable",
    37: "Vulpix", 38: "Ninetales",
    39: "Jigglypuff", 40: "Wigglytuff",
    41: "Zubat", 42: "Golbat",
    43: "Oddish", 44: "Gloom", 45: "Vileplume",
    46: "Paras", 47: "Parasect",
    48: "Venonat", 49: "Venomoth",
    50: "Diglett", 51: "Dugtrio",
    52: "Meowth", 53: "Persian",
    54: "Psyduck", 55: "Golduck",
    56: "Mankey", 57: "Primeape",
    58: "Growlithe", 59: "Arcanine",
    60: "Poliwag", 61: "Poliwhirl", 62: "Poliwrath",
    63: "Abra", 64: "Kadabra", 65: "Alakazam",
    66: "Machop", 67: "Machoke", 68: "Machamp",
    69: "Bellsprout", 70: "Weepinbell", 71: "Victreebel",
    72: "Tentacool", 73: "Tentacruel",
    74: "Geodude", 75: "Graveler", 76: "Golem",
    77: "Ponyta", 78: "Rapidash",
    79: "Slowpoke", 80: "Slowbro",
    81: "Magnemite", 82: "Magneton",
    83: "Farfetch'd",
    84: "Doduo", 85: "Dodrio",
    86: "Seel", 87: "Dewgong",
    88: "Grimer", 89: "Muk",
    90: "Shellder", 91: "Cloyster",
    92: "Gastly", 93: "Haunter", 94: "Gengar",
    95: "Onix",
    96: "Drowzee", 97: "Hypno",
    98: "Krabby", 99: "Kingler",
    100: "Voltorb", 101: "Electrode",
    102: "Exeggcute", 103: "Exeggutor",
    104: "Cubone", 105: "Marowak",
    106: "Hitmonlee", 107: "Hitmonchan",
    108: "Lickitung",
    109: "Koffing", 110: "Weezing",
    111: "Rhyhorn", 112: "Rhydon",
    113: "Chansey",
    114: "Tangela",
    115: "Kangaskhan",
    116: "Horsea", 117: "Seadra",
    118: "Goldeen", 119: "Seaking",
    120: "Staryu", 121: "Starmie",
    122: "Mr. Mime",
    123: "Scyther",
    124: "Jynx",
    125: "Electabuzz",
    126: "Magmar",
    127: "Pinsir",
    128: "Tauros",
    129: "Magikarp", 130: "Gyarados",
    131: "Lapras",
    132: "Ditto",
    133: "Eevee", 134: "Vaporeon", 135: "Jolteon", 136: "Flareon",
    137: "Porygon",
    138: "Omanyte", 139: "Omastar",
    140: "Kabuto", 141: "Kabutops",
    142: "Aerodactyl",
    143: "Snorlax",
    144: "Articuno", 145: "Zapdos", 146: "Moltres",
    147: "Dratini", 148: "Dragonair", 149: "Dragonite",
    150: "Mewtwo",
    151: "Mew",
}

# Gen 1 party and battle structs store an internal species index, not the
# Pokédex number. Translate that internal index through pokered's dex order.
INTERNAL_SPECIES_TO_DEX: Dict[int, int] = {
    1: 112, 2: 115, 3: 32, 4: 35, 5: 21, 6: 100, 7: 34, 8: 80, 9: 2, 10: 103,
    11: 108, 12: 102, 13: 88, 14: 94, 15: 29, 16: 31, 17: 104, 18: 111,
    19: 131, 20: 59, 21: 151, 22: 130, 23: 90, 24: 72, 25: 92, 26: 123,
    27: 120, 28: 9, 29: 127, 30: 114, 31: 0, 32: 0, 33: 58, 34: 95, 35: 22,
    36: 16, 37: 79, 38: 64, 39: 75, 40: 113, 41: 67, 42: 122, 43: 106,
    44: 107, 45: 24, 46: 47, 47: 54, 48: 96, 49: 76, 50: 0, 51: 126, 52: 0,
    53: 125, 54: 82, 55: 109, 56: 0, 57: 56, 58: 86, 59: 50, 60: 128, 61: 0,
    62: 0, 63: 0, 64: 83, 65: 48, 66: 149, 67: 0, 68: 0, 69: 0, 70: 84,
    71: 60, 72: 124, 73: 146, 74: 144, 75: 145, 76: 132, 77: 52, 78: 98,
    79: 0, 80: 0, 81: 0, 82: 37, 83: 38, 84: 25, 85: 26, 86: 0, 87: 0,
    88: 147, 89: 148, 90: 140, 91: 141, 92: 116, 93: 117, 94: 0, 95: 0,
    96: 27, 97: 28, 98: 138, 99: 139, 100: 39, 101: 40, 102: 133, 103: 136,
    104: 135, 105: 134, 106: 66, 107: 41, 108: 23, 109: 46, 110: 61, 111: 62,
    112: 13, 113: 14, 114: 15, 115: 0, 116: 85, 117: 57, 118: 51, 119: 49,
    120: 87, 121: 0, 122: 0, 123: 10, 124: 11, 125: 12, 126: 68, 127: 0,
    128: 55, 129: 97, 130: 42, 131: 150, 132: 143, 133: 129, 134: 0, 135: 0,
    136: 89, 137: 0, 138: 99, 139: 91, 140: 0, 141: 101, 142: 36, 143: 110,
    144: 53, 145: 105, 146: 0, 147: 93, 148: 63, 149: 65, 150: 17, 151: 18,
    152: 121, 153: 1, 154: 3, 155: 73, 156: 0, 157: 118, 158: 119, 159: 0,
    160: 0, 161: 0, 162: 0, 163: 77, 164: 78, 165: 19, 166: 20, 167: 33,
    168: 30, 169: 74, 170: 137, 171: 142, 172: 0, 173: 81, 174: 0, 175: 0,
    176: 4, 177: 7, 178: 5, 179: 8, 180: 6, 181: 0, 182: 0, 183: 0, 184: 0,
    185: 43, 186: 44, 187: 45, 188: 69, 189: 70, 190: 71,
}

MOVE_NAMES: Dict[int, str] = {
    0: "(none)",
    1: "Pound", 2: "Karate Chop", 3: "Double Slap", 4: "Comet Punch",
    5: "Mega Punch", 6: "Pay Day", 7: "Fire Punch", 8: "Ice Punch",
    9: "Thunder Punch", 10: "Scratch", 11: "Vice Grip", 12: "Guillotine",
    13: "Razor Wind", 14: "Swords Dance", 15: "Cut", 16: "Gust",
    17: "Wing Attack", 18: "Whirlwind", 19: "Fly", 20: "Bind",
    21: "Slam", 22: "Vine Whip", 23: "Stomp", 24: "Double Kick",
    25: "Mega Kick", 26: "Jump Kick", 27: "Rolling Kick", 28: "Sand Attack",
    29: "Headbutt", 30: "Horn Attack", 31: "Fury Attack", 32: "Horn Drill",
    33: "Tackle", 34: "Body Slam", 35: "Wrap", 36: "Take Down",
    37: "Thrash", 38: "Double-Edge", 39: "Tail Whip", 40: "Poison Sting",
    41: "Twineedle", 42: "Pin Missile", 43: "Leer", 44: "Bite",
    45: "Growl", 46: "Roar", 47: "Sing", 48: "Supersonic",
    49: "Sonic Boom", 50: "Disable", 51: "Acid", 52: "Ember",
    53: "Flamethrower", 54: "Mist", 55: "Water Gun", 56: "Hydro Pump",
    57: "Surf", 58: "Ice Beam", 59: "Blizzard", 60: "Psybeam",
    61: "Bubble Beam", 62: "Aurora Beam", 63: "Hyper Beam", 64: "Peck",
    65: "Drill Peck", 66: "Submission", 67: "Low Kick", 68: "Counter",
    69: "Seismic Toss", 70: "Strength", 71: "Absorb", 72: "Mega Drain",
    73: "Leech Seed", 74: "Growth", 75: "Razor Leaf", 76: "Solar Beam",
    77: "Poison Powder", 78: "Stun Spore", 79: "Sleep Powder",
    80: "Petal Dance", 81: "String Shot", 82: "Dragon Rage",
    83: "Fire Spin", 84: "Thunder Shock", 85: "Thunderbolt",
    86: "Thunder Wave", 87: "Thunder", 88: "Rock Throw",
    89: "Earthquake", 90: "Fissure", 91: "Dig", 92: "Toxic",
    93: "Confusion", 94: "Psychic", 95: "Hypnosis", 96: "Meditate",
    97: "Agility", 98: "Quick Attack", 99: "Rage", 100: "Teleport",
    101: "Night Shade", 102: "Mimic", 103: "Screech", 104: "Double Team",
    105: "Recover", 106: "Harden", 107: "Minimize", 108: "Smokescreen",
    109: "Confuse Ray", 110: "Withdraw", 111: "Defense Curl",
    112: "Barrier", 113: "Light Screen", 114: "Haze", 115: "Reflect",
    116: "Focus Energy", 117: "Bide", 118: "Metronome",
    119: "Mirror Move", 120: "Self-Destruct", 121: "Egg Bomb",
    122: "Lick", 123: "Smog", 124: "Sludge", 125: "Bone Club",
    126: "Fire Blast", 127: "Waterfall", 128: "Clamp", 129: "Swift",
    130: "Skull Bash", 131: "Spike Cannon", 132: "Constrict",
    133: "Amnesia", 134: "Kinesis", 135: "Soft-Boiled",
    136: "High Jump Kick", 137: "Glare", 138: "Dream Eater",
    139: "Poison Gas", 140: "Barrage", 141: "Leech Life",
    142: "Lovely Kiss", 143: "Sky Attack", 144: "Transform",
    145: "Bubble", 146: "Dizzy Punch", 147: "Spore",
    148: "Flash", 149: "Psywave", 150: "Splash", 151: "Acid Armor",
    152: "Crabhammer", 153: "Explosion", 154: "Fury Swipes",
    155: "Bonemerang", 156: "Rest", 157: "Rock Slide",
    158: "Hyper Fang", 159: "Sharpen", 160: "Conversion",
    161: "Tri Attack", 162: "Super Fang", 163: "Slash",
    164: "Substitute", 165: "Struggle",
}

TYPE_NAMES: Dict[int, str] = {
    0: "Normal", 1: "Fighting", 2: "Flying", 3: "Poison",
    4: "Ground", 5: "Rock", 6: "Bug", 7: "Ghost",
    # 8 unused
    20: "Fire", 21: "Water", 22: "Grass", 23: "Electric",
    24: "Ice", 25: "Psychic", 26: "Dragon",
}

ITEM_NAMES: Dict[int, str] = {
    0: "(none)",
    1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poke Ball",
    5: "Town Map", 6: "Bicycle", 7: "?????", 8: "Safari Ball",
    9: "Pokedex", 10: "Moon Stone", 11: "Antidote", 12: "Burn Heal",
    13: "Ice Heal", 14: "Awakening", 15: "Parlyz Heal", 16: "Full Restore",
    17: "Max Potion", 18: "Hyper Potion", 19: "Super Potion", 20: "Potion",
    21: "Boulder Badge", 22: "Cascade Badge", 23: "Thunder Badge",
    24: "Rainbow Badge", 25: "Soul Badge", 26: "Marsh Badge",
    27: "Volcano Badge", 28: "Earth Badge",
    29: "Escape Rope", 30: "Repel", 31: "Old Amber",
    32: "Fire Stone", 33: "Thunder Stone", 34: "Water Stone",
    35: "HP Up", 36: "Protein", 37: "Iron", 38: "Carbos",
    39: "Calcium", 40: "Rare Candy",
    41: "Dome Fossil", 42: "Helix Fossil", 43: "Secret Key",
    44: "?????", 45: "Bike Voucher", 46: "X Accuracy",
    47: "Leaf Stone", 48: "Card Key", 49: "Nugget",
    50: "PP Up", 51: "Poke Doll", 52: "Full Heal",
    53: "Revive", 54: "Max Revive", 55: "Guard Spec.",
    56: "Super Repel", 57: "Max Repel", 58: "Dire Hit",
    59: "Coin", 60: "Fresh Water", 61: "Soda Pop", 62: "Lemonade",
    63: "S.S. Ticket", 64: "Gold Teeth", 65: "X Attack",
    66: "X Defend", 67: "X Speed", 68: "X Special",
    69: "Coin Case", 70: "Oak's Parcel", 71: "Itemfinder",
    72: "Silph Scope", 73: "Poke Flute", 74: "Lift Key",
    75: "Exp. All", 76: "Old Rod", 77: "Good Rod", 78: "Super Rod",
    79: "PP Up", 80: "Ether", 81: "Max Ether", 82: "Elixir",
    83: "Max Elixir",
    196: "HM01", 197: "HM02", 198: "HM03", 199: "HM04", 200: "HM05",
    201: "TM01", 202: "TM02", 203: "TM03", 204: "TM04", 205: "TM05",
    206: "TM06", 207: "TM07", 208: "TM08", 209: "TM09", 210: "TM10",
    211: "TM11", 212: "TM12", 213: "TM13", 214: "TM14", 215: "TM15",
    216: "TM16", 217: "TM17", 218: "TM18", 219: "TM19", 220: "TM20",
    221: "TM21", 222: "TM22", 223: "TM23", 224: "TM24", 225: "TM25",
    226: "TM26", 227: "TM27", 228: "TM28", 229: "TM29", 230: "TM30",
    231: "TM31", 232: "TM32", 233: "TM33", 234: "TM34", 235: "TM35",
    236: "TM36", 237: "TM37", 238: "TM38", 239: "TM39", 240: "TM40",
    241: "TM41", 242: "TM42", 243: "TM43", 244: "TM44", 245: "TM45",
    246: "TM46", 247: "TM47", 248: "TM48", 249: "TM49", 250: "TM50",
}

# fmt: off
MAP_NAMES: Dict[int, str] = {
    0: "Pallet Town", 1: "Viridian City", 2: "Pewter City",
    3: "Cerulean City", 4: "Lavender Town", 5: "Vermilion City",
    6: "Celadon City", 7: "Fuchsia City", 8: "Cinnabar Island",
    9: "Indigo Plateau", 10: "Saffron City", 11: "???",
    12: "Route 1", 13: "Route 2", 14: "Route 3", 15: "Route 4",
    16: "Route 5", 17: "Route 6", 18: "Route 7", 19: "Route 8",
    20: "Route 9", 21: "Route 10", 22: "Route 11", 23: "Route 12",
    24: "Route 13", 25: "Route 14", 26: "Route 15", 27: "Route 16",
    28: "Route 17", 29: "Route 18", 30: "Route 19", 31: "Route 20",
    32: "Route 21", 33: "Route 22", 34: "Route 23", 35: "Route 24",
    36: "Route 25",
    37: "Red's House 1F", 38: "Red's House 2F",
    39: "Blue's House", 40: "Oak's Lab",
    41: "Viridian Pokecenter", 42: "Viridian Mart",
    43: "Viridian School", 44: "Viridian House",
    45: "Viridian Gym",
    46: "Digletts Cave (Route 2)", 47: "Viridian Forest Gate (S)",
    48: "Route 2 Trade House", 49: "Route 2 Gate (N)",
    50: "Viridian Forest",
    51: "Pewter Museum 1F", 52: "Pewter Museum 2F",
    53: "Pewter Gym", 54: "Pewter House", 55: "Pewter Mart",
    56: "Pewter Pokecenter",
    57: "Mt Moon 1F", 58: "Mt Moon B1F", 59: "Mt Moon B2F",
    60: "Cerulean House (trashed)", 61: "Cerulean House 2",
    62: "Cerulean Pokecenter", 63: "Cerulean Gym",
    64: "Cerulean Bike Shop", 65: "Cerulean Mart",
    66: "Mt Moon Pokecenter",
    67: "???", 68: "Route 5 Gate", 69: "Underground Path (5-6) Entrance",
    70: "Daycare",
    71: "Route 6 Gate", 72: "Underground Path (5-6) Exit",
    73: "???", 74: "Route 7 Gate",
    75: "Underground Path (7-8) Entrance",
    76: "???",
    77: "Route 8 Gate", 78: "Underground Path (7-8) Exit",
    79: "Rock Tunnel Pokecenter",
    80: "Rock Tunnel 1F", 81: "Power Plant",
    82: "Route 11 Gate 1F", 83: "Digletts Cave (Route 11)",
    84: "Route 11 Gate 2F",
    85: "Route 12 Gate 1F", 86: "Bill's House",
    87: "Vermilion Pokecenter", 88: "Pokemon Fan Club",
    89: "Vermilion Mart", 90: "Vermilion Gym",
    91: "Vermilion House (old rod)", 92: "Vermilion Dock",
    93: "S.S. Anne Exterior", 94: "S.S. Anne 1F Rooms",
    95: "S.S. Anne 2F", 96: "S.S. Anne 2F Rooms",
    97: "S.S. Anne B1F Rooms", 98: "S.S. Anne Bow",
    99: "S.S. Anne Kitchen",  100: "S.S. Anne Captains Room",
    101: "S.S. Anne 1F", 102: "S.S. Anne B1F",
    103: "???",
    104: "???",
    105: "???",
    106: "???",
    107: "???",
    108: "Lavender Pokecenter", 109: "Pokemon Tower 1F",
    110: "Pokemon Tower 2F", 111: "Pokemon Tower 3F",
    112: "Pokemon Tower 4F", 113: "Pokemon Tower 5F",
    114: "Pokemon Tower 6F", 115: "Pokemon Tower 7F",
    116: "Lavender House 1", 117: "Lavender Mart",
    118: "Lavender House 2",
    119: "Celadon Dept Store 1F", 120: "Celadon Dept Store 2F",
    121: "Celadon Dept Store 3F", 122: "Celadon Dept Store 4F",
    123: "Celadon Dept Store Roof", 124: "Celadon Dept Store Elevator",
    125: "Celadon Mansion 1F", 126: "Celadon Mansion 2F",
    127: "Celadon Mansion 3F", 128: "Celadon Mansion Roof",
    129: "Celadon Pokecenter", 130: "Celadon Gym",
    131: "Game Corner", 132: "Celadon Dept Store 5F",
    133: "Game Corner Prize Room",
    134: "Celadon Diner", 135: "Celadon House",
    136: "Celadon Hotel",
    137: "Fuchsia Pokecenter", 138: "Fuchsia Mart",
    139: "Fuchsia House 1", 140: "Fuchsia House 2",
    141: "Safari Zone Gate", 142: "Fuchsia Gym",
    143: "Fuchsia Meeting Room",
    144: "Seafoam Islands B1F", 145: "Seafoam Islands B2F",
    146: "Seafoam Islands B3F", 147: "Seafoam Islands B4F",
    148: "Vermilion House 2 (good rod)",
    149: "Fuchsia House 3 (good rod)", 150: "Mansion 1F",
    151: "Cinnabar Gym", 152: "Cinnabar Lab",
    153: "Cinnabar Lab Trade Room", 154: "Cinnabar Lab Metronome Room",
    155: "Cinnabar Lab Fossil Room",
    156: "Cinnabar Pokecenter", 157: "Cinnabar Mart",
    158: "???",
    159: "Indigo Plateau Lobby", 160: "Copycats House 1F",
    161: "Copycats House 2F",
    162: "Fighting Dojo", 163: "Saffron Gym",
    164: "Saffron House", 165: "Saffron Mart",
    166: "Silph Co 1F", 167: "Silph Co 2F", 168: "Silph Co 3F",
    169: "Silph Co 4F", 170: "Silph Co 5F", 171: "Silph Co 6F",
    172: "Silph Co 7F", 173: "Silph Co 8F", 174: "Silph Co 9F",
    175: "Silph Co 10F", 176: "Silph Co 11F",
    177: "Saffron Pokecenter",
    178: "Mr Psychics House",
    179: "Route 15 Gate 1F", 180: "Route 15 Gate 2F",
    181: "Route 16 Gate 1F", 182: "Route 16 Gate 2F",
    183: "Route 16 Fly House",
    184: "Route 12 House (super rod)",
    185: "Route 18 Gate 1F", 186: "Route 18 Gate 2F",
    187: "Seafoam Islands 1F",
    188: "Route 22 Gate",
    189: "Victory Road 1F",
    190: "Route 12 Gate 2F",
    191: "Vermilion House 3 (diary)",
    192: "Digletts Cave",
    193: "Victory Road 2F",
    194: "Rocket Hideout B1F", 195: "Rocket Hideout B2F",
    196: "Rocket Hideout B3F", 197: "Rocket Hideout B4F",
    198: "Rocket Hideout Elevator",
    199: "???", 200: "???", 201: "???",
    202: "Silph Co Elevator",
    203: "???", 204: "???",
    205: "Trade Center", 206: "Colosseum",
    207: "???", 208: "???",
    209: "Lorelei Room", 210: "Bruno Room",
    211: "Agatha Room", 212: "Lance Room",
    213: "Hall of Fame",
    214: "Underground Path (N-S)", 215: "Champions Room",
    216: "Underground Path (W-E)",
    217: "Cerulean Cave 1F", 218: "Cerulean Cave 2F",
    219: "Cerulean Cave B1F",
    220: "Name Raters House",
    221: "Cerulean House 3",
    222: "???",
    223: "Rock Tunnel B1F",
    224: "Safari Zone East", 225: "Safari Zone North",
    226: "Safari Zone West", 227: "Safari Zone Center",
    228: "Safari Zone Rest House 1", 229: "Safari Zone Secret House",
    230: "Safari Zone Rest House 2", 231: "Safari Zone Rest House 3",
    232: "Safari Zone Rest House 4",
    233: "Unknown Dungeon 2",  # alternate ID
    234: "Unknown Dungeon 3",
    235: "???",
    236: "Pokemon Mansion 2F", 237: "Pokemon Mansion 3F",
    238: "Pokemon Mansion B1F",
    239: "Safari Zone Gate 2",
    240: "Victory Road 3F",
    241: "???",
    242: "???",
    243: "Fighting Dojo 2",
    244: "Indigo Plateau 2",
    245: "???", 246: "???", 247: "???",
    248: "Cerulean Cave 3F",
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
    "Boulder", "Cascade", "Thunder", "Rainbow",
    "Soul", "Marsh", "Volcano", "Earth",
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
                moves.append({
                    "id": mid,
                    "name": MOVE_NAMES.get(mid, f"???({mid})"),
                    "pp": data[29 + i] & 0x3F,
                    "pp_up": (data[29 + i] >> 6) & 0x03,
                })

        return {
            **species,
            "nickname": nickname,
            "level": data[33],
            "hp": (data[1] << 8) | data[2],
            "max_hp": (data[34] << 8) | data[35],
            "status": self._decode_status(data[4]),
            "types": [
                TYPE_NAMES.get(data[5], f"???({data[5]})"),
                TYPE_NAMES.get(data[6], f"???({data[6]})"),
            ],
            "moves": moves,
            "stats": {
                "attack":  (data[36] << 8) | data[37],
                "defense": (data[38] << 8) | data[39],
                "speed":   (data[40] << 8) | data[41],
                "special": (data[42] << 8) | data[43],
            },
            "ot_id": (data[12] << 8) | data[13],
            "experience": (data[14] << 16) | (data[15] << 8) | data[16],
        }

    def _read_enemy_battle_mon(self) -> Dict[str, Any]:
        """Parse the live active enemy battle struct at ``wEnemyMon``.

        ``wEnemyMon`` uses Gen 1's ``battle_struct`` layout:
        - level is at offset 14
        - max HP starts at offset 15
        - PP lives after the stats block

        This is different from the 44-byte party struct used in party data.
        """
        data = self.emu.read_range(ADDR_ENEMY_MON, BATTLE_MON_SIZE)
        species = self._decode_species(data[0])

        moves: List[str] = []
        for index in range(4):
            move_id = data[8 + index]
            if move_id != 0:
                moves.append(MOVE_NAMES.get(move_id, f"???({move_id})"))

        return {
            **species,
            "level": data[14],
            "hp": (data[1] << 8) | data[2],
            "max_hp": (data[15] << 8) | data[16],
            "status": self._decode_status(data[4]),
            "types": [
                TYPE_NAMES.get(data[5], f"???({data[5]})"),
                TYPE_NAMES.get(data[6], f"???({data[6]})"),
            ],
            "moves": moves,
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

        hours = self.emu.read_u16(ADDR_PLAYTIME_H)
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
            items.append({
                "id": item_id,
                "item": ITEM_NAMES.get(item_id, f"???({item_id})"),
                "quantity": qty,
            })
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
        """Read current map dimensions in block coordinates."""
        return {
            "width": self.emu.read_u8(ADDR_MAP_WIDTH),
            "height": self.emu.read_u8(ADDR_MAP_HEIGHT),
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

        gym_leaders_defeated = [
            BADGE_NAMES[i] for i in range(8) if badges & (1 << i)
        ]

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
