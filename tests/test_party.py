"""The party as a fighting unit: the gym ahead, and the move about to be deleted.

Every number here is the one measured on the run that produced this module: one
Charmeleon L33 holding Cut, Growl, Ember and Leer, one Boulder Badge, 40
whiteouts and 3,044 presses inside Cerulean Gym.

The screen-text literals are not invented. They were read off ``wTileMap`` while
PyBoy drove TM28 onto that exact Charmeleon, one frame per A press.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pokemon_agent import gamedata
from pokemon_agent import party as pf

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mon(species, level, types, stats, moves, hp=None, max_hp=None):
    return {
        "species": species,
        "level": level,
        "types": list(types),
        "stats": dict(stats),
        "hp": hp if hp is not None else 95,
        "max_hp": max_hp if max_hp is not None else 95,
        "moves": [
            {"name": name, "pp": pp}
            for name, pp in (
                moves if moves and isinstance(moves[0], tuple) else [(m, 20) for m in moves]
            )
        ],
    }


#: The live party, stats included, as the memory reader read it at 33:15 played.
CHARMELEON = _mon(
    "Charmeleon",
    33,
    ["Fire"],
    {"attack": 66, "defense": 54, "speed": 71, "special": 57},
    [("Cut", 29), ("Growl", 40), ("Ember", 23), ("Leer", 30)],
)

BOULDER_ONLY = ["Boulder"]

# --- the gym ahead ---------------------------------------------------------


def test_the_gym_outlook_names_the_move_that_wins_not_the_move_being_used():
    """Ember was the move it kept picking; Cut is the move that beats Misty.

    Ember is Fire on Water, halved, and takes about seven turns on Starmie. Cut
    is a 50-power Normal move on the same Pokemon and takes three. Both numbers
    were in reach of the harness the whole time and neither was ever printed
    before a fight; the run picked the halved one and lost.
    """
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    line = pf.leader_outlook([CHARMELEON], leader)
    assert "Misty" in line
    assert "Staryu L18" in line and "Starmie L21" in line
    assert "best Cut" in line
    assert "Ember" not in line


def test_a_party_with_no_attack_at_all_is_told_it_cannot_win_the_gym():
    """The failure this field exists for, in its pure form.

    Growl and Leer took 49 of one run's 289 battle turns. A party holding only
    those cannot take a single point of HP off anything Misty owns, and every
    payload it read was silent about that while it walked in forty times.
    """
    status_only = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        ["Growl", "Leer"],
    )
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    line = pf.leader_outlook([status_only], leader)
    assert "nothing you carry damages it" in line
    assert "this party cannot win this gym" in line


def test_one_pokemon_is_reported_as_one_faint_from_a_whiteout():
    """40 whiteouts on a one-Pokemon party, and no payload ever said "one".

    Party size is the fact behind every one of them: with a second Pokemon a
    faint is a switch, and with one it is the walk back from a Poke Center.
    """
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    assert "1 Pokemon" in pf.leader_outlook([CHARMELEON], leader)


def test_the_outlook_reads_every_party_member_not_slot_zero():
    """A prior audit caught `calc` attacking with party[0] while another mon was out.

    The answer to a Water gym is rarely in slot 0. Here slot 0 can only Growl and
    the Pikachu behind it carries the fight, so an outlook that stopped at slot 0
    would report a gym that cannot be won.
    """
    growler = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        ["Growl", "Leer"],
    )
    sparker = _mon(
        "Pikachu",
        25,
        ["Electric"],
        {"attack": 45, "defense": 35, "speed": 70, "special": 45},
        ["Thunder Shock"],
    )
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    line = pf.leader_outlook([growler, sparker], leader)
    assert "best Thunder Shock (Pikachu)" in line
    assert "cannot win this gym" not in line


def test_a_fainted_member_answers_nothing():
    """Its moves are on the struct and it cannot take a turn. Counting them lies."""
    fainted = dict(CHARMELEON, hp=0)
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    assert pf.leader_outlook([fainted], leader) is None


def test_a_move_with_no_pp_is_not_an_answer():
    """`calc` ranked a 0 PP move as the best available 12 times and `fight` refused it.

    The same mistake one room earlier would be worse: it would send the party
    into a gym on a move the game will not let it use.
    """
    dry = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        [("Cut", 0), ("Growl", 40)],
    )
    leader = pf.unbeaten_leader("Cerulean Gym", BOULDER_ONLY)
    assert "nothing you carry damages it" in pf.leader_outlook([dry], leader)


def test_a_gym_whose_badge_is_already_won_says_nothing():
    """Pewter Gym is 3,044 presses of nothing once the Boulder Badge is in the bag."""
    assert pf.unbeaten_leader("Pewter Gym", BOULDER_ONLY) is None


def test_an_ordinary_map_has_no_leader_to_price():
    assert pf.unbeaten_leader("Cerulean City", BOULDER_ONLY) is None
    assert pf.unbeaten_leader(None, BOULDER_ONLY) is None


def test_every_gym_map_resolves_to_a_leader_and_a_team():
    """The map->badge table is local knowledge; the rosters are pokered's.

    If a gym map name ever drifts, this is where it shows up rather than in a
    silent field that stopped appearing.
    """
    for map_name in pf.GYM_BADGE:
        leader = pf.unbeaten_leader(map_name, [])
        assert leader is not None, map_name
        assert leader["team"], map_name


# --- the move about to be deleted ------------------------------------------

#: Read off wTileMap, frame by frame, while TM28 was driven onto the live
#: Charmeleon. Blank rows are the tile rows the game left empty.
TRYING_TO_LEARN = (
    "   CHAR       33\n            ABLE\n\n\n\n\n\n\n\n\n\n\n\n\n CHAR is\n\n trying to learn"
)
CANNOT_LEARN_MORE = (
    "   CHAR       33\n            ABLE\n\n\n\n\n\n\n\n\n\n\n\n\n But, CHAR\n\n can't learn more"
)
MAKE_ROOM_YES_NO = (
    "   CHAR       33\n            ABLE\n\n\n\n\n\n\n                YES\n\n                NO"
    "\n\n\n\n move to make room\n\n for DIG?"
)
FORGET_LIST = (
    "   CHAR       33\n            ABLE\n\n\n\n\n\n\n      CUT\n      GROWL\n      EMBER"
    "\n      LEER\n\n\n Which move should\n\n be forgotten?"
)
ORDINARY_DIALOG = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n TECHNOLOGY IS\n\n INCREDIBLE!"


def _prompt(text, incoming="Dig", cursor=0, slot=0):
    return {"screen_text": text, "incoming": incoming, "cursor": cursor, "slot": slot}


def test_every_frame_of_the_learn_flow_is_recognised():
    """The warning is worthless on the frames it skips.

    An earlier phrase list used "can't learn more", which is one tile in Gen 1
    and decoded as a hole; the frame that said it went unwarned and A on that
    frame walks straight into the deletion.
    """
    for text in (TRYING_TO_LEARN, CANNOT_LEARN_MORE, MAKE_ROOM_YES_NO, FORGET_LIST):
        assert pf.is_learn_prompt(text)


def test_an_ordinary_dialog_is_not_a_learn_prompt():
    assert not pf.is_learn_prompt(ORDINARY_DIALOG)
    assert not pf.is_learn_prompt("")


def test_the_forget_list_names_the_move_the_cursor_would_delete():
    """The button is the whole point. "A here deletes Cut" is what no advice replaces."""
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=0), [CHARMELEON])
    assert "A here deletes Cut (50)" in line
    assert "for Dig (100)" in line
    assert "Ember 40, Dig 100" in line


def test_deleting_the_last_attack_says_the_party_would_be_left_with_nothing():
    """The self-inflicted wound this exists to stop, one move further along.

    Cut went over an attack on the live run and left Charmeleon with one 40-power
    Ember against a Water gym. Delete Ember too and the payload has to say the
    Pokemon can no longer damage anything, before the press, not after.
    """
    one_attack = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        ["Ember", "Growl", "Leer", "Rage"],
    )
    text = FORGET_LIST.replace("CUT", "EMBER").replace(
        "      EMBER\n      LEER", "      LEER\n      RAGE"
    )
    prompt = _prompt(text, incoming="Cut", cursor=0)
    line = pf.learn_cost(prompt, [one_attack])
    assert "A here deletes Ember (40)" in line


def test_the_prompt_before_the_list_names_the_moves_at_risk():
    """No move is chosen yet, so name the four that are and which of them attack."""
    line = pf.learn_cost(_prompt(TRYING_TO_LEARN), [CHARMELEON])
    assert "Cut, Growl, Ember, Leer" in line
    assert "only Cut 50, Ember 40 do damage" in line


def test_an_incoming_move_the_data_does_not_recognise_is_not_named():
    """The move id comes off a scratch byte that keeps its last value forever.

    It read 15 (Cut) on an overworld frame hours after the last teach. It is only
    read while the screen says a learn is happening, and anything it names that
    the Pokemon already knows, or that is not a move, is dropped rather than
    printed as fact.
    """
    stale = pf.learn_cost(_prompt(TRYING_TO_LEARN, incoming="Cut"), [CHARMELEON])
    assert "the new move" in stale
    assert "Cut (" not in stale
    nonsense = pf.learn_cost(_prompt(TRYING_TO_LEARN, incoming="???(0)"), [CHARMELEON])
    assert "the new move" in nonsense


def test_the_learning_pokemon_is_the_one_whose_moves_are_on_screen():
    """A check on wWhichPokemon rather than a guess.

    The forget list draws the four names, so with a party of several the payload
    can prove which member the prompt is about instead of trusting a slot byte.
    """
    other = _mon(
        "Pikachu",
        25,
        ["Electric"],
        {"attack": 45, "defense": 35, "speed": 70, "special": 45},
        ["Thunder Shock", "Quick Attack", "Tail Whip", "Thunder Wave"],
    )
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=2, slot=0), [other, CHARMELEON])
    assert "Charmeleon would be left" in line
    assert "deletes Ember (40)" in line


def test_no_prompt_and_no_party_produce_no_line():
    assert pf.learn_cost(None, [CHARMELEON]) is None
    assert pf.learn_cost(_prompt(FORGET_LIST), []) is None


# --- the machines in the bag -----------------------------------------------

#: The bag that Charmeleon carried into Vermilion, item for item, as `poke state`
#: printed it. TM28 had been in it since Mt. Moon.
VERMILION_BAG = [
    {"item": "Town Map", "quantity": 1},
    {"item": "Poke Ball", "quantity": 11},
    {"item": "TM34", "quantity": 1},
    {"item": "Potion", "quantity": 9},
    {"item": "Helix Fossil", "quantity": 1},
    {"item": "Nugget", "quantity": 1},
    {"item": "S.S. Ticket", "quantity": 1},
    {"item": "TM28", "quantity": 1},
    {"item": "HM01", "quantity": 1},
    {"item": "Super Potion", "quantity": 2},
    {"item": "TM11", "quantity": 1},
    {"item": "Bicycle", "quantity": 1},
]


def test_the_tm_that_was_carried_for_sixty_thousand_presses_is_named():
    """TM28 is Dig, 100 power, and Charmeleon's hardest attack is Cut at 50.

    Both halves were already in the harness -- the bag said TM28, species.json
    said Charmeleon can learn TM28 -- and nothing joined them, so the machine
    rode along unused from Mt. Moon to Vermilion.
    """
    line = pf.teachable_tms([CHARMELEON], VERMILION_BAG)
    assert line is not None
    assert "TM28 teaches Dig (Ground 100)" in line
    assert "Charmeleon can learn it" in line
    # And what it would be traded against, so the cost of the slot is on the
    # same line as the gain.
    assert "Cut 50" in line and "Ember 40" in line


def test_a_machine_the_species_cannot_learn_is_not_offered():
    """TM11 is Bubble Beam and it was in the same bag. Charmeleon cannot learn it."""
    line = pf.teachable_tms([CHARMELEON], VERMILION_BAG)
    assert "TM11" not in line
    assert "Bubble Beam" not in line


def test_a_machine_with_no_power_is_not_an_upgrade():
    """TM34 is Bide, which Charmeleon *can* learn and which deals no damage."""
    line = pf.teachable_tms([CHARMELEON], VERMILION_BAG)
    assert "TM34" not in line and "Bide" not in line


def test_a_move_already_known_is_not_an_upgrade():
    """HM01 is Cut and Charmeleon is holding Cut."""
    assert "HM01" not in pf.teachable_tms([CHARMELEON], VERMILION_BAG)


def test_a_machine_weaker_than_what_is_carried_says_nothing():
    """Once Dig is taught there is nothing left in that bag to say."""
    taught = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        [("Cut", 29), ("Dig", 10), ("Ember", 23), ("Leer", 30)],
    )
    assert pf.teachable_tms([taught], VERMILION_BAG) is None


def test_a_fainted_member_is_still_offered_a_machine():
    """Fainting decides whether it can fight, not whether it can learn.

    The Gen 1 TM screen teaches a Pokemon at 0 HP without complaint, so filtering
    the teaching advice on HP dropped the strongest member exactly when it had
    just fainted and fell through to whatever was left standing. Measured over
    one run: 222 of 292 impressions of this line recommended teaching a level-3
    Rattata, and none of the 292 was ever acted on.
    """
    fainted = dict(CHARMELEON, hp=0)
    assert pf.teachable_tms([fainted], VERMILION_BAG) == pf.teachable_tms(
        [CHARMELEON], VERMILION_BAG
    )


def test_an_empty_bag_or_party_says_nothing():
    assert pf.teachable_tms([CHARMELEON], []) is None
    assert pf.teachable_tms([], VERMILION_BAG) is None


def test_at_most_three_machines_are_named():
    """A late-game bag holds a dozen; a field that grows without bound is wallpaper."""
    weak = _mon(
        "Charmeleon",
        33,
        ["Fire"],
        {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        [("Growl", 40), ("Leer", 30)],
    )
    bag = [{"item": label, "quantity": 1} for label in ("TM01", "TM03", "TM06", "TM28", "TM38")]
    line = pf.teachable_tms([weak], bag)
    assert line.count(" teaches ") == pf.TEACHABLE_LIMIT
    # Sorted by how much power it adds, so the cut falls on the least useful.
    assert "TM38 teaches Fire Blast" in line


# --- how often it is said --------------------------------------------------


def test_a_fact_is_repeated_when_it_becomes_true_again_not_every_frame():
    """3,044 presses in one gym x 113 bytes is 344 kB, on a 95 kB median session.

    So the outlook is keyed on (map, in a battle, the text itself): walking back
    in says it again, a fight starting says it again, standing still does not.
    """
    once = pf.SayOnce()
    assert once.fresh(("Cerulean Gym", False, "line"))
    assert not once.fresh(("Cerulean Gym", False, "line"))
    assert once.fresh(("Cerulean Gym", True, "line"))
    assert not once.fresh(("Cerulean Gym", False, "line"))
    once.reset()
    assert once.fresh(("Cerulean Gym", False, "line"))


# --- against the ROM -------------------------------------------------------


def _find_saves_dir():
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")


@needs_rom
def test_the_screen_decodes_to_the_words_actually_drawn_on_it():
    """wTileMap is the screen, and Gen 1's text encoding is its font numbering.

    Proved by opening the one menu every save can open. POKeDEX is the test for
    the accented tile that used to decode as a hole and took "can't" with it.
    """
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    states = sorted(p for p in SAVES_DIR.glob("*.state"))
    if not states:
        pytest.skip("no save states to open a menu from")
    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        for state in reversed(states):
            emulator.load_state(str(state))
            emulator.tick(60)
            emulator.press("start", 8)
            emulator.tick(60)
            text = RedBlueMemoryReader(emulator).read_screen_text()
            if "ITEM" in text:
                assert "SAVE" in text and "OPTION" in text
                assert "POKéDEX" in text
                return
        pytest.skip("no save opened the start menu")
    finally:
        emulator.close()


@needs_rom
@pytest.mark.skipif(os.environ.get("POKEMON_SKIP_SLOW") == "1", reason="drives ~40 button presses")
def test_a_tm_on_a_full_moveset_is_warned_about_before_the_press():
    """The whole hazard, driven end to end against the real ROM.

    Every frame from "CHAR is trying to learn DIG!" to "Which move should be
    forgotten?" has to carry the warning, and the last of them has to name the
    move under the cursor -- because that is the frame where one A press spends
    a slot that an HM move can never give back.
    """
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    candidates = sorted(SAVES_DIR.glob("*.state"), reverse=True)
    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        for state in candidates:
            emulator.load_state(str(state))
            emulator.tick(60)
            reader = RedBlueMemoryReader(emulator)
            party = reader.read_party()
            bag = reader.read_bag()
            teachable = [
                index
                for index, item in enumerate(bag)
                if str(item.get("item", "")).startswith(("TM", "HM"))
            ]
            if not party or len(party[0].get("moves") or []) < 4 or not teachable:
                continue
            if (reader.read_battle() or {}).get("in_battle") or reader.read_dialog()["active"]:
                continue

            def press(button, wait=60):
                emulator.press(button, 8)
                emulator.tick(wait)

            press("start")
            press("down")
            press("down")
            press("a")  # POKEDEX -> POKEMON -> ITEM -> the bag
            for _ in range(teachable[-1]):
                press("down", wait=20)
            press("a")
            press("a")  # USE
            seen = []
            for _ in range(12):
                press("a")
                line = pf.learn_cost(reader.read_move_learn(), party)
                if line:
                    seen.append(line)
                if "be forgotten" in reader.read_screen_text():
                    break
            if not seen:
                continue  # this save's TM is one the Pokemon cannot learn
            assert all(text.startswith("learn ") for text in seen)
            assert "A here deletes" in seen[-1]
            assert party[0]["moves"][0]["name"] in seen[-1]
            return
        pytest.skip("no save has a teachable TM and a four-move lead")
    finally:
        emulator.close()


@needs_rom
@pytest.mark.skipif(os.environ.get("POKEMON_SKIP_SLOW") == "1", reason="drives ~40 button presses")
def test_the_machine_table_agrees_with_what_the_rom_boots_up():
    """tms.json is generated from pokered; this is the ROM's own answer.

    Booting up a TM puts the move it teaches in ``wMoveNum``, which the reader
    surfaces as ``move_learn["incoming"]``. If the generated table and the ROM
    ever disagreed, the payload would name the wrong move on the one frame where
    naming the wrong move deletes a slot for good.
    """
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    emulator = PyBoyEmulator()
    emulator.load(str(SAVES_DIR / "PokemonRed.gb"))
    checked: list = []
    try:
        for state in sorted(SAVES_DIR.glob("*.state"), reverse=True):
            emulator.load_state(str(state))
            emulator.tick(60)
            reader = RedBlueMemoryReader(emulator)
            party, bag = reader.read_party(), reader.read_bag()
            if not party or (reader.read_battle() or {}).get("in_battle"):
                continue
            if reader.read_dialog()["active"]:
                continue
            learnable = (gamedata.species(str(party[0].get("species") or "")) or {}).get("tm_hm")
            known = set(pf.move_names(party[0]))
            picks = [
                (index, str(item.get("item")))
                for index, item in enumerate(bag)
                if str(item.get("item")) in (learnable or ())
                and gamedata.tm_move(str(item.get("item"))) not in known
            ]
            if not picks:
                continue

            def press(button, times=1, wait=30):
                for _ in range(times):
                    emulator.press(button, 8)
                    emulator.tick(wait)

            def open_the_bag():
                """Both menus remember where they were left, so nothing is counted.

                The start menu is a different length before the Pokedex, and the
                bag list scrolls under a cursor that stops at row 2, so a fixed
                number of Down presses lands somewhere different on every save.
                `at_bag_list` and `read_list_menu` say where it actually is.
                """
                press("start", wait=60)
                for _ in range(8):
                    press("a", wait=60)
                    if reader.at_bag_list():
                        return True
                    press("b", wait=40)
                    press("down", wait=30)
                return False

            for index, label in picks:
                if label in checked:
                    continue
                emulator.load_state(str(state))
                emulator.tick(60)
                assert open_the_bag(), f"{state.name}: never reached the bag"
                for _ in range(len(bag) + 2):
                    here = reader.read_list_menu()["index"]
                    if here == index:
                        break
                    press("down" if here < index else "up", wait=25)
                assert reader.read_list_menu()["index"] == index
                press("a")
                press("a")  # USE
                for _ in range(10):
                    press("a")
                    incoming = (reader.read_move_learn() or {}).get("incoming")
                    if incoming:
                        assert incoming == gamedata.tm_move(label), (
                            f"{label}: the ROM booted up {incoming!r}, tms.json says "
                            f"{gamedata.tm_move(label)!r}"
                        )
                        checked.append(label)
                        break
                else:
                    raise AssertionError(f"{state.name}: {label} booted up nothing")
            # Enough saves to cover a few different machines; the bag of any one
            # of them holds two or three.
            if len(checked) >= 4:
                break
        assert checked, "no save has a machine its lead Pokemon can still learn"
    finally:
        emulator.close()
