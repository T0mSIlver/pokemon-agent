"""Which move to teach, and which to delete: the two decisions, priced.

Every number here was measured on the run in
``runs/20260825T224823Z-983b/receipts.jsonl`` -- 43,312 records, 230,103
presses, 70 hours -- and on the sessions under ``.agent-workspace/pi-session``
that drove it. The two that matter:

* the machine line was printed **292** times and acted on **zero** times, and
  on **222** of those it named a level 3 Rattata while the level 46 Charizard
  that fought every battle of the run was named on 70;
* of 1,436 attacks, **761** were Ember at 40 base power and **74** were Leer,
  which has none, on a Pokemon carrying Slash in the next slot the whole time.

The party those numbers came off is still live on port 8765 and is the
``CHARIZARD`` fixture below, move for move.
"""

from __future__ import annotations

import pytest

from pokemon_agent import gamedata
from pokemon_agent import party as pf


def _mon(species, level, types, moves, hp=100):
    return {
        "species": species,
        "level": level,
        "types": list(types),
        "hp": hp,
        "max_hp": 100,
        "moves": [{"name": name, "pp": 20} for name in moves],
    }


#: The live party on port 8765 at the time of writing, species for species and
#: move for move. One Pokemon does the fighting and five ride along at level 5.
CHARIZARD = _mon("Charizard", 46, ["Fire", "Flying"], ["Cut", "Slash", "Ember", "Leer"])
PIKACHU = _mon("Pikachu", 5, ["Electric"], ["Thunder Shock", "Growl"])
RATTATA = _mon("Rattata", 3, ["Normal"], ["Tackle", "Tail Whip"])
LIVE_PARTY = [CHARIZARD, PIKACHU, RATTATA]

#: The bag beside it, machines only.
LIVE_BAG = [
    {"item": "TM34", "quantity": 1},
    {"item": "TM28", "quantity": 1},
    {"item": "HM01", "quantity": 1},
    {"item": "TM11", "quantity": 1},
    {"item": "TM24", "quantity": 1},
    {"item": "HM05", "quantity": 1},
    {"item": "TM42", "quantity": 1},
]


# --- the data the claims rest on -------------------------------------------


def test_the_game_data_backs_every_number_this_module_prints():
    """No claim here is typed in; each one is a column of pokered's own tables."""
    assert gamedata.move("Dig")["effect"] == "CHARGE_EFFECT"  # so 100 lands per two turns
    assert gamedata.move("Dig")["power"] == 100
    assert gamedata.move("Slash")["power"] == 70
    assert gamedata.move("Leer")["power"] == 0
    assert gamedata.move("Cut")["accuracy"] == 95
    assert gamedata.tm_move("TM28") == "Dig"
    assert gamedata.tm_move("TM24") == "Thunderbolt"
    assert gamedata.tm_move("HM01") == "Cut"
    assert "TM28" in gamedata.species("Charizard")["tm_hm"]
    assert "TM24" not in gamedata.species("Charizard")["tm_hm"]
    assert {"level": 46, "move": "Flamethrower"} in gamedata.species("Charizard")["learnset"]


# --- what a move is worth in a turn ----------------------------------------


def test_stab_accuracy_and_the_charge_turn_are_all_priced():
    """The three corrections, one assertion each, against the raw power beside them.

    The old line compared ``moves.json``'s power column and nothing else, and
    each of these is the difference between that number and what the turn buys.
    """
    # STAB: Ember is Fire on a Fire type, so 40 is 60.
    assert pf.power_per_turn(CHARIZARD, "Ember") == 60
    # Accuracy: Cut misses one turn in twenty, so 50 is 47.
    assert pf.power_per_turn(CHARIZARD, "Cut") == 47
    # The charge turn: Dig's 100 arrives once every two turns.
    assert pf.power_per_turn(CHARIZARD, "Dig") == 50
    # And nothing at all applies to Slash, so it stays 70 -- above Dig.
    assert pf.power_per_turn(CHARIZARD, "Slash") == 70


def test_a_move_that_deals_no_damage_is_worth_nothing_a_turn():
    assert pf.power_per_turn(CHARIZARD, "Leer") == 0
    assert pf.power_per_turn(PIKACHU, "Growl") == 0


def test_fixed_damage_moves_are_not_compared_as_though_their_power_were_one():
    """Seismic Toss deals the user's level; its power byte reads 1 and means nothing."""
    assert gamedata.move("Seismic Toss")["power"] == 1
    assert pf.power_per_turn(CHARIZARD, "Seismic Toss") == 0


def test_the_cheapest_slot_is_never_an_hm_slot():
    """Gen 1 refuses to delete an HM move, so naming one is advice A cannot obey."""
    assert pf.hm_moves()["Cut"] == "HM01"
    name, per_turn, index = pf.cheapest_slot(CHARIZARD)
    assert (name, per_turn, index) == ("Leer", 0, 3)
    hm_only = _mon("Charizard", 46, ["Fire", "Flying"], ["Cut"])
    assert pf.cheapest_slot(hm_only) is None


# --- the machines in the bag -----------------------------------------------


def test_the_machine_goes_to_the_pokemon_that_fights_not_the_weakest_one():
    """The measured bug, in the party it was measured on.

    The old rule ranked by ``taught power - the learner's best``, a difference
    that is always largest on the worst Pokemon in the party. Rattata's best is
    Tackle, so every machine in the bag scored highest on a level 3 Rattata:
    222 of the 292 lines the run ever printed named it, and none was acted on.
    """
    line = pf.teachable_tms(LIVE_PARTY, LIVE_BAG)
    assert "Rattata" not in line
    assert "TM28 teaches Dig (Ground 100) and Charizard can learn it" in line
    assert "TM24 teaches Thunderbolt (Electric 95) and Pikachu can learn it" in line


def test_thunderbolt_goes_to_the_electric_type_that_gets_stab_off_it():
    """95 on a Rattata and 142 on a Pikachu are the same machine and different facts."""
    assert pf.power_per_turn(PIKACHU, "Thunderbolt") == 142
    assert pf.power_per_turn(RATTATA, "Thunderbolt") == 95
    line = pf.teachable_tms([PIKACHU, RATTATA], [{"item": "TM24", "quantity": 1}])
    assert "Pikachu can learn it" in line
    assert "Rattata" not in line


def test_dig_is_not_sold_as_an_upgrade_over_slash():
    """100 beats 70 and 50 a turn does not. Both halves are on the line.

    The reason to teach Dig survives -- it is the only Ground move the party can
    reach, and Ground is what an Electric gym answers to -- but it is stated as
    coverage, which is true, instead of as more damage, which is not.
    """
    line = pf.teachable_tms([CHARIZARD], [{"item": "TM28", "quantity": 1}])
    assert "50 a turn under its best Slash at 70" in line
    assert "halved by its 2-turn charge" in line
    assert "adds Ground" in line


def test_the_line_names_the_level_of_the_pokemon_it_is_talking_about():
    """ "Rattata" and "Rattata L3" are not the same offer, and only one was printed."""
    line = pf.teachable_tms([RATTATA], [{"item": "TM24", "quantity": 1}])
    assert "Rattata can learn it (L3)" in line


def test_teaching_costs_a_slot_and_the_line_says_which_one():
    """A four-move Pokemon pays for a machine with a move, and it never said so."""
    full = pf.teachable_tms([CHARIZARD], [{"item": "TM28", "quantity": 1}])
    assert "costs the Leer slot (no damage)" in full
    spare = pf.teachable_tms([PIKACHU], [{"item": "TM24", "quantity": 1}])
    assert "a slot is free" in spare


def test_every_machine_in_the_bag_is_accounted_for_with_a_reason():
    """The mechanism reports its own failure: no machine can go quiet.

    TM28 rode along in the bag for 60,000 presses and nothing in the harness
    could distinguish "considered and dropped" from "never looked at". Each
    machine now leaves a row saying which.
    """
    rows = {row["item"]: row for row in pf.tm_audit(LIVE_PARTY, LIVE_BAG)}
    assert set(rows) == {"TM34", "TM28", "HM01", "TM11", "TM24", "HM05", "TM42"}
    assert rows["TM34"]["status"] == "no_power"  # Bide
    assert rows["HM05"]["status"] == "no_power"  # Flash
    assert rows["HM01"]["status"] == "already_known"  # Charizard is holding Cut
    assert rows["TM42"]["status"] == "nobody_can_learn"  # Dream Eater
    assert rows["TM28"]["status"] == "named" and rows["TM28"]["mon"] == "Charizard"
    assert rows["TM24"]["status"] == "named" and rows["TM24"]["mon"] == "Pikachu"
    for row in rows.values():
        assert row["detail"], f"{row['item']} left no reason"


def test_a_machine_that_is_neither_stronger_nor_new_coverage_is_dropped_with_a_reason():
    taught = _mon("Charizard", 46, ["Fire", "Flying"], ["Cut", "Slash", "Dig", "Ember"])
    rows = {row["item"]: row for row in pf.tm_audit([taught], [{"item": "TM28", "quantity": 1}])}
    assert rows["TM28"]["status"] == "already_known"
    weaker = _mon("Charizard", 46, ["Fire", "Flying"], ["Slash", "Fire Blast"])
    rows = {row["item"]: row for row in pf.tm_audit([weaker], [{"item": "TM01", "quantity": 1}])}
    assert rows["TM01"]["status"] == "no_gain"  # Mega Punch, 68 a turn, under Slash


# --- the move about to be deleted ------------------------------------------

FORGET_LIST = (
    "\n      CUT\n      SLASH\n      EMBER\n      LEER\n\n Which move should\n\n be forgotten?"
)
TRYING_TO_LEARN = "\n\n\n\n\n\n\n\n CHAR is trying to\n learn FLAMETHROWER!"


def _prompt(text, incoming="Flamethrower", cursor=0, slot=0):
    return {"screen_text": text, "incoming": incoming, "cursor": cursor, "slot": slot}


def test_the_forget_list_names_the_cheapest_slot_and_how_far_away_it_is():
    """The press count is the half the model kept getting wrong.

    One session read "A here deletes Cut", pressed ``down a down a down`` to
    reach Leer, and the first ``a`` deleted Slash. The distance to the slot
    worth nothing is arithmetic the payload can do and the agent was doing by
    eye off a screenshot.
    """
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=0), [CHARIZARD])
    assert "The cheapest slot is Leer, which does no damage: 3 down from here" in line
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=2), [CHARIZARD])
    assert "1 down from here" in line
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=3), [CHARIZARD])
    assert "Leer is the cheapest slot Charizard has" in line


def test_the_list_says_when_the_button_will_not_take():
    """A on an HM slot deletes nothing; the frame says "HM techniques can't be deleted".

    The old line said "A here deletes Cut (50)" and stopped there, which is
    false on that slot, and two sessions spent themselves pressing through a
    refusal they had been told was a deletion.
    """
    line = pf.learn_cost(_prompt(FORGET_LIST, cursor=0), [CHARIZARD])
    assert "Cut is HM01 and Gen 1 refuses to delete an HM move, so A here will not take" in line
    assert "will not take" not in pf.learn_cost(_prompt(FORGET_LIST, cursor=3), [CHARIZARD])


def test_the_prompt_before_the_list_names_the_slot_to_spend():
    """Three damaging moves and four names is a subtraction, and it was left undone.

    Twice a session read this line, could not work out which move to drop, and
    abandoned the learn -- once concluding, wrongly, that "Gen 1 won't let you
    replace a move during a battle".
    """
    line = pf.learn_cost(_prompt(TRYING_TO_LEARN), [CHARIZARD])
    assert "the cheapest slot is Leer, which does no damage" in line
    assert "Cut is HM01, and Gen 1 deletes no HM move" in line


def test_a_move_that_outclasses_one_already_carried_says_so():
    """Flamethrower over Ember is not a fifth attack, it is the same attack harder.

    Fire at 142 a turn against Fire at 60: the slot Ember is in is the one the
    new move is for, and nothing in the payload joined those two lines.
    """
    line = pf.learn_cost(_prompt(TRYING_TO_LEARN), [CHARIZARD])
    assert "Flamethrower is Fire at 142 a turn and outclasses Ember 60" in line


# --- the slot nobody is prompting about ------------------------------------


def test_the_dead_slot_is_named_without_waiting_for_a_prompt():
    """Leer sat in slot four for 70 hours and was used 74 times out of 1,436 attacks.

    ``learn_cost`` speaks only during a replacement prompt and ``teachable_tms``
    only when a machine fits. Between them is where this run actually lived.
    """
    line = pf.moveset_gaps(LIVE_PARTY)
    assert "Charizard L46 attacks with Slash 70, Ember 60, Cut 47 a turn" in line
    assert "spends 1 slot on Leer, worth nothing" in line


def test_the_level_up_move_that_is_due_right_now_is_named():
    """Charizard's L46 move is Flamethrower and the live L46 Charizard lacks it."""
    assert pf.level_up_move(CHARIZARD) == (46, "Flamethrower", True)
    line = pf.moveset_gaps(LIVE_PARTY)
    assert "Flamethrower (Fire 95, 142 a turn) is its level-up move at this level" in line


def test_a_level_up_move_still_ahead_is_named_with_its_level():
    earlier = _mon("Charizard", 39, ["Fire", "Flying"], ["Cut", "Slash", "Ember", "Leer"])
    assert pf.level_up_move(earlier) == (46, "Flamethrower", False)
    assert "Flamethrower (Fire 95, 142 a turn) comes at L46" in pf.moveset_gaps([earlier])


def test_a_move_the_level_is_already_past_is_not_reported_as_due():
    """This party declined Rage at 24 and spent Scratch before it.

    A prompt that has come and gone is not a decision in front of the agent, and
    reporting it as one would be the payload inventing a button to press.
    """
    assert {"level": 24, "move": "Rage"} in gamedata.species("Charizard")["learnset"]
    level, name, due = pf.level_up_move(
        _mon("Charizard", 39, ["Fire", "Flying"], ["Cut", "Slash", "Ember", "Leer"])
    )
    assert (level, name, due) == (46, "Flamethrower", False)


def test_a_full_moveset_says_nothing():
    """The field costs its bytes only on the frames it has something to report."""
    good = _mon("Charizard", 46, ["Fire", "Flying"], ["Cut", "Slash", "Flamethrower", "Dig"])
    assert pf.moveset_gaps([good]) is None
    assert pf.moveset_gaps([]) is None


# ---------------------------------------------------------------------------
# The batch that deleted Slash by accident
# ---------------------------------------------------------------------------


class TestDoubleConfirmDuringAMoveReplacement:
    """`down a down a down` cost a move the session never meant to delete.

    Verbatim from the transcript, having read `learn A here deletes Cut (50) for
    Flamethrower (95)`: "Cut is an HM move, so it can\'t be replaced... I need to
    move down to Leer" -- then `poke act down a down a down`. The first A deleted
    Slash. The session then reasoned "It deleted SLASH, not Leer. That\'s
    unfortunate... But actually, Flamethrower (95) is much stronger than Slash
    (70), so this is a net gain", reading the accident as the plan.

    One A is a confirmation. Two in a batch is a press into a screen the batch
    could not see, and the replacement prompt is the one dialog in Gen 1 that
    cannot be undone.
    """

    def _server(self, monkeypatch, *, learning: bool):
        from pokemon_agent import server

        state = {
            "party": [
                {
                    "species": "Charizard",
                    "level": 46,
                    "hp": 150,
                    "types": ["Fire", "Flying"],
                    "moves": [
                        {"name": "Cut"},
                        {"name": "Slash"},
                        {"name": "Ember"},
                        {"name": "Leer"},
                    ],
                }
            ],
            "battle": {"in_battle": False},
        }
        if learning:
            state["move_learn"] = {"move": "Flamethrower", "cursor": 0}
        monkeypatch.setattr(server, "_get_state_dict", lambda: state)
        return server

    def test_two_confirms_are_refused(self, monkeypatch):
        from fastapi import HTTPException

        server = self._server(monkeypatch, learning=True)
        with pytest.raises(HTTPException) as caught:
            server._reject_unsafe_dialog_actions(
                ["walk_down", "press_a", "walk_down", "press_a", "walk_down"]
            )
        detail = caught.value.detail
        assert "2 A presses in one batch" in detail
        assert "cannot see" in detail

    def test_one_confirm_after_moving_the_cursor_is_allowed(self, monkeypatch):
        """The pattern that worked four times must keep working."""
        server = self._server(monkeypatch, learning=True)
        server._reject_unsafe_dialog_actions(["walk_down", "press_a"])

    def test_two_confirms_outside_a_replacement_are_allowed(self, monkeypatch):
        """Advancing two pages of ordinary dialog is not this failure."""
        server = self._server(monkeypatch, learning=False)
        server._reject_unsafe_dialog_actions(["press_a", "press_a", "press_a"])
