"""GET /gamedata: the static database, shaped to be printed rather than dumped.

223 maps, 334 trainers, 59 encounter tables, 151 species and 165 moves were on
disk and unreachable from the agent. The thing worth pinning is not that the
numbers are right — `test_gamedata.py` does that — but that an answer stays
small enough that asking is cheaper than guessing.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pokemon_agent import capabilities, server

#: A whole answer has to fit in the context a script would spend printing it.
ANSWER_BUDGET_BYTES = 1500


@pytest.fixture(scope="module")
def http():
    # No lifespan: /gamedata reads files, never the emulator, so it answers on a
    # server that has not been configured or started.
    return TestClient(server.app)


def ask(http, topic: str, **params):
    response = http.get(f"/gamedata/{topic}", params=params)
    assert response.status_code == 200, response.text
    assert len(response.content) <= ANSWER_BUDGET_BYTES, (
        f"{topic} answered {len(response.content)}B"
    )
    return response.json()


def test_trainers_say_who_where_and_with_what(http):
    payload = ask(http, "trainers", map="Pewter Gym")

    assert payload["map"] == "Pewter Gym"
    assert payload["count"] == 2
    brock = payload["trainers"][0]
    assert brock["class"] == "Brock"
    assert brock["at"] == [4, 1]
    assert brock["team"] == ["Geodude L12", "Onix L14"]


def test_encounters_merge_the_slots_into_species(http):
    payload = ask(http, "encounters", map="Route 3")

    grass = payload["grass"]
    assert grass["rate"] == 20
    assert grass["levels"] == [3, 8]
    # Ten slots, three species. The ten rows are the game's implementation; the
    # three are the fact a script wants.
    assert [row["species"] for row in grass["species"]] == ["Pidgey", "Spearow", "Jigglypuff"]
    assert grass["species"][0]["levels"] == [6, 8]
    assert sum(row["chance"] for row in grass["species"]) == pytest.approx(1.0, abs=0.01)
    assert payload["water"] is None


def test_a_species_is_small_enough_to_print(http):
    payload = ask(http, "species", name="Charmeleon")

    assert len(json.dumps(payload)) < 500
    assert payload["types"] == ["Fire"]
    assert payload["base"]["atk"] == 64
    assert payload["evolves"] == ["Charizard by level 36"]
    assert [level for level, _ in payload["learnset"]][:3] == [1, 1, 1]
    # The TM list is 24 entries of noise next to a battle, so it is a count.
    assert payload["tm_hm_count"] == 24
    assert "tm_hm" not in payload


def test_full_species_adds_the_tm_list_back(http):
    payload = ask(http, "species", name="Charmeleon", full=True)

    assert "HM01" in payload["tm_hm"]
    assert payload["growth"] == "medium_slow"


def test_a_move_says_which_stat_it_attacks_with(http):
    ember = ask(http, "move", name="ember")

    assert ember["name"] == "Ember"
    assert ember["power"] == 40
    # Gen 1 splits by type, not by move: every Fire move reads Special.
    assert ember["damage_class"] == "special"
    assert ask(http, "move", name="Tackle")["damage_class"] == "physical"


def test_items_carry_the_tile_and_flag_the_hidden_ones(http):
    payload = ask(http, "items", map="Viridian Forest")

    assert payload["count"] == 5
    visible = [row for row in payload["items"] if not row.get("hidden")]
    hidden = [row for row in payload["items"] if row.get("hidden")]
    assert len(visible) == 3 and len(hidden) == 2
    assert visible[0] == {"item": "Antidote", "at": [25, 11]}


def test_shops_list_stock_and_answer_null_for_a_map_with_no_mart(http):
    assert "Poke Ball" in ask(http, "shops", map="Pewter Mart")["items"]
    assert ask(http, "shops", map="Pewter Gym")["items"] is None


def test_types_answer_the_question_rather_than_the_table(http):
    water = ask(http, "types", name="Water")
    assert set(water["super_effective"]) == {"Fire", "Ground", "Rock"}
    assert "Grass" in water["not_very_effective"]

    assert ask(http, "types", name="Normal", against="Ghost")["multiplier"] == 0.0
    assert ask(http, "types", name="Water", against="Rock,Ground")["multiplier"] == 4.0


def test_names_are_matched_without_case(http):
    assert ask(http, "trainers", map="pewter gym")["map"] == "Pewter Gym"
    assert ask(http, "species", name="CHARMELEON")["name"] == "Charmeleon"


def test_an_unknown_name_is_a_404_that_suggests_something(http):
    response = http.get("/gamedata/species", params={"name": "Charmandr"})
    assert response.status_code == 404
    assert "Charmander" in response.json()["detail"]

    response = http.get("/gamedata/trainers", params={"map": "Pewter"})
    assert response.status_code == 404
    assert "Pewter City" in response.json()["detail"]


def test_a_missing_argument_is_a_400_that_shows_the_query(http):
    response = http.get("/gamedata/trainers")
    assert response.status_code == 400
    assert "?map=" in response.json()["detail"]

    response = http.get("/gamedata/species")
    assert response.status_code == 400
    assert "?name=" in response.json()["detail"]


def test_an_unknown_topic_lists_the_topics(http):
    response = http.get("/gamedata/pokedex")
    assert response.status_code == 404
    detail = response.json()["detail"]
    for topic in capabilities.GAMEDATA_TOPICS:
        assert topic in detail


def test_long_tables_are_capped_and_say_so():
    payload = capabilities.gamedata_items("Viridian Forest", limit=2)

    assert payload["count"] == 5
    assert payload["shown"] == 2
    assert payload["truncated"] is True
    assert len(payload["items"]) == 2


def test_a_limit_that_is_not_a_number_is_a_refusal():
    with pytest.raises(capabilities.CapabilityError):
        capabilities.gamedata_items("Viridian Forest", limit="lots")
    with pytest.raises(capabilities.CapabilityError):
        capabilities.gamedata_items("Viridian Forest", limit=0)


def test_every_map_answers_every_map_topic_within_budget():
    """No map in the game produces an answer a script cannot afford to print."""
    from pokemon_agent import gamedata

    for map_name in gamedata.map_names():
        for topic in ("trainers", "encounters", "items", "shops"):
            payload = capabilities.gamedata_payload(topic, map_name=map_name)
            size = len(json.dumps(payload, separators=(",", ":")))
            assert size <= ANSWER_BUDGET_BYTES, f"{topic} for {map_name} is {size}B"


def test_every_species_and_move_answers_within_budget():
    from pokemon_agent import gamedata

    for name in gamedata.all_species():
        size = len(json.dumps(capabilities.gamedata_species(name), separators=(",", ":")))
        assert size <= ANSWER_BUDGET_BYTES, f"{name} is {size}B"
    for name in gamedata.all_moves():
        assert len(json.dumps(capabilities.gamedata_move(name))) <= 400
