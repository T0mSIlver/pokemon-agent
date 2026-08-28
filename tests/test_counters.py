"""Guards for the rooms that hand something over without being a mart.

Counted off one run's own receipts.jsonl: in the 69 minutes after a save reload
took the Bicycle back off it, the run spent 1,233 presses across 543 receipts
inside the Cerulean Bike Shop, 41% of every press in the window, 213 of them
moving nothing, standing on (4,2) 165 times and (4,1) 131 times, holding the
Bike Voucher. The Bicycle is fourteen presses from that room's door.

It is not that the room is hard. The same run traded the voucher correctly at
09:54, from (4,2), in six A presses. What it could not do was find that tile
twice: the failing window reached it, pressed A five separate times, and walked
off on the sixth.

Nothing the harness said about the room helped and one thing it said was false.
`gamedata.services("Cerulean Bike Shop")` was `[]` because the generated
classifier only ever matched `predef HealParty`, so every service in the game is
a nurse. `shop_payload` answered "Cerulean Bike Shop is not a mart. Nothing here
is for sale." -- a confident wrong answer about a room where a major
progression item is traded, which this project ranks below both a refusal and
silence. `poke map` pointed at (4,1) as "nearest unexplored", which is the one
tile up there that faces nobody.

So the tests here cover three things: that the refusal no longer asserts
anything about the room, that the room now carries a fact, and that the fact
matches the running game.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from pokemon_agent import capabilities, gamedata
from pokemon_agent.capabilities import NotFound
from pokemon_agent.server import _counter_line, _counter_stand_tiles

REPO_ROOT = Path(__file__).resolve().parents[1]

BIKE_SHOP = "Cerulean Bike Shop"


def _state(map_name: str, bag: list[str]) -> dict:
    return {
        "map": {"map_name": map_name},
        "bag": [{"item": name, "quantity": 1} for name in bag],
    }


# ---------------------------------------------------------------------------
# The refusal must not assert anything about the room
# ---------------------------------------------------------------------------


def test_a_trading_room_is_never_told_that_nothing_is_for_sale():
    """The whole point. Absence from shops.json is a fact about the table.

    The Bicycle is obtained in this room, so "Nothing here is for sale" was
    false, and the model reading it had no reason to look further.
    """
    with pytest.raises(NotFound) as caught:
        capabilities.shop_payload(BIKE_SHOP, 2977)
    message = str(caught.value)
    assert "Nothing here is for sale" not in message
    assert "nothing" not in message.lower()
    # It still has to refuse, and still has to say what it refused.
    assert BIKE_SHOP in message
    assert "Poke Mart" in message


def test_the_mart_refusal_says_which_table_is_missing_not_what_the_room_holds():
    """Every non-mart map, not just the one that made this matter."""
    for map_name in ("Route 4", "Cerulean City", "Pokemon Fan Club", BIKE_SHOP):
        with pytest.raises(NotFound) as caught:
            capabilities.shop_payload(map_name, 500)
        assert "nothing" not in str(caught.value).lower()


def test_a_real_mart_still_answers_with_its_stock():
    """The refusal wording is the only thing that changed."""
    payload = capabilities.shop_payload("Vermilion Mart", 7198)
    assert payload["money"] == 7198
    assert any(row["item"] == "Poke Ball" for row in payload["stock"])


# ---------------------------------------------------------------------------
# The room now carries a fact
# ---------------------------------------------------------------------------


def test_the_bike_shop_counter_is_known():
    counter = capabilities.counter_at(BIKE_SHOP)
    assert counter is not None
    assert counter["wants"] == "Bike Voucher"
    assert counter["gives"] == "Bicycle"
    assert counter["at"] == [6, 2]
    assert counter["stand"] == [4, 2]
    assert counter["face"] == "right"
    assert counter["presses"] == 6


def test_every_counter_carries_the_four_things_the_line_needs():
    """A half-filled entry would print `facing None` at the model."""
    for name, counter in capabilities.SPECIAL_COUNTERS.items():
        assert set(counter) >= {"service", "at", "stand", "face", "presses", "wants", "gives"}, name
        assert counter["face"] in ("up", "down", "left", "right"), name
        assert counter["presses"] >= 1, name
        assert tuple(counter["stand"]) in _counter_stand_tiles(counter["at"]), name


def test_the_counter_shows_up_as_a_service_so_poke_map_names_it():
    services = capabilities.services_at(BIKE_SHOP)
    assert [entry["service"] for entry in services] == ["trade"]
    assert services[0]["at"] == [6, 2]


def test_the_overlay_and_the_generated_entry_are_one_npc_not_two():
    """Both halves of the harness found this clerk, and they agree on the tile.

    The overlay's [6, 2] was measured by walking to it and pressing A until the
    Bicycle landed in the bag. The generator's came later and independently,
    from `BikeShop.asm`'s own object table, once it learned to read
    `call GiveItem`. Two sources agreeing is worth keeping; naming the same
    person twice on `poke map` is not.

    The overlay wins because it carries what no map table can: which tile to
    stand on, which way to face, and how many presses the conversation takes.
    """
    generated = [entry for entry in gamedata.services(BIKE_SHOP) if entry.get("at") == [6, 2]]
    assert generated, "the generator finds the clerk too, or this test is vacuous"
    assert generated[0]["service"] == "gift"

    merged = capabilities.services_at(BIKE_SHOP)
    assert len(merged) == 1, "one clerk, one entry"
    assert merged[0]["service"] == "trade"
    assert merged[0]["stand"] == [4, 2], "the measured fields survive the merge"
    assert merged[0]["presses"] == 6


def test_the_stand_tile_agrees_with_the_harness_own_counter_convention():
    """A talk-over counter is two tiles out, which is what `poke heal` assumes.

    If the measured tile were not one of the tiles the server would search, the
    two halves of the harness would be describing different rooms.
    """
    counter = capabilities.counter_at(BIKE_SHOP)
    assert tuple(counter["stand"]) in _counter_stand_tiles(counter["at"])


def test_the_overlay_does_not_disturb_the_generated_nurses():
    """Fourteen healers before, fourteen after, and none of them on this map.

    Thirteen until the generator learned to follow a script into the helper it
    calls, which is what had hidden Mom's free heal in Red's House.
    """
    healers = [name for name in gamedata.map_names() if capabilities.healer_at(name) is not None]
    assert len(healers) == 14
    assert capabilities.healer_at(BIKE_SHOP) is None
    assert capabilities.counter_at("Cerulean Pokecenter") is None
    assert capabilities.counter_at("Route 4") is None
    assert capabilities.counter_at(None) is None


# ---------------------------------------------------------------------------
# The line on the frame
# ---------------------------------------------------------------------------


def test_the_frame_names_the_tile_the_direction_the_trade_and_the_press_count():
    """All of it in one line. Any of it missing is a window the run already lost.

    The tile, because it stood on (4,1) 131 times. The facing, because it walks
    in from the north and arrives facing down. The press count, because the one
    failing window pressed A five times and left on the sixth.
    """
    line = _counter_line(_state(BIKE_SHOP, ["Bike Voucher", "Potion"]))
    assert line is not None
    assert "(6,2)" in line  # who
    assert "(4,2)" in line  # where to stand -- not (4,1), which is where it stood
    assert "right" in line  # which way to face
    assert "a:6" in line  # how many times, and in a form `poke act` takes
    assert "Bike Voucher" in line and "Bicycle" in line  # what it is for
    assert "(4,1)" not in line


def test_the_line_goes_once_the_bicycle_is_in_the_bag():
    """A finished room reads as one. See the Mt Moon Center re-walking."""
    assert _counter_line(_state(BIKE_SHOP, ["Bicycle"])) is None


def test_without_the_voucher_the_line_says_so_and_drops_the_tile():
    line = _counter_line(_state(BIKE_SHOP, ["Potion"]))
    assert line is not None
    assert "not carrying" in line
    assert "(4,2)" not in line


def test_no_other_map_carries_the_line():
    for map_name in ("Route 4", "Cerulean City", "Vermilion Mart", "Cerulean Pokecenter"):
        assert _counter_line(_state(map_name, ["Bike Voucher"])) is None
    assert _counter_line({"map": {}}) is None
    assert _counter_line({}) is None


def test_the_line_survives_a_junk_bag():
    """Perception must never fail a state read."""
    assert _counter_line({"map": {"map_name": BIKE_SHOP}, "bag": "junk"}) is not None
    assert _counter_line({"map": {"map_name": BIKE_SHOP}}) is not None


def test_the_line_stays_inside_the_heal_line_budget():
    """Same budget as `_heal_line`, measured the same way, for every counter."""
    for name, counter in capabilities.SPECIAL_COUNTERS.items():
        for bag in ([counter["wants"]], [], ["Potion"]):
            line = _counter_line(_state(name, bag))
            assert line is not None
            assert len(line.encode()) < 100, line


# ---------------------------------------------------------------------------
# Against the real ROM
# ---------------------------------------------------------------------------


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")


@needs_rom
def test_the_recorded_sequence_really_puts_the_bicycle_in_the_bag():
    """Drive it. The tile, facing and press count are measurements, so re-measure.

    Walks to the recorded tile with the same planner the server walks with,
    turns the recorded way, and presses A the recorded number of times.

    The count is an upper bound, not an exact one, and this test is what found
    that out: driven from the door of one save the trade lands on the sixth
    press, and from the tile the corpus happened to offer it lands on the fifth,
    because how much of the clerk's text has already scrolled differs. Extra
    presses hit a closed box and cost nothing; one too few is the failure the
    line exists to stop. So the assertion is "within", and separately that one
    press is not enough -- a count of 1 would pass a bare "within" check while
    telling the model nothing.
    """
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator
    from pokemon_agent.memory.red import MAP_NAMES, PokemonRedReader

    counter = capabilities.counter_at(BIKE_SHOP)
    paths = sorted(SAVES_DIR.glob("*.state"))
    random.Random(20260827).shuffle(paths)
    emulator = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        for path in paths:
            try:
                emulator.load_state(str(path))
            except Exception:  # noqa: BLE001 — a save being written is not a finding
                continue
            emulator.settle()
            if MAP_NAMES.get(reader.read_map_info()["map_id"]) != BIKE_SHOP:
                continue
            carried = {entry["item"] for entry in reader.read_bag()}
            if counter["wants"] not in carried or counter["gives"] in carried:
                continue

            snapshot = emulator.get_navigation_snapshot(reader).to_dict()
            collision = capabilities.collision_from(snapshot)
            start = (reader.read_player()["position"]["x"], reader.read_player()["position"]["y"])
            plan = capabilities.plan_within(collision, start, tuple(counter["stand"]))
            assert plan is not None, f"{counter['stand']} unreachable from {start}"
            for action in plan:
                emulator.press_and_settle(action.removeprefix("walk_"), frames=10)
            position = reader.read_player()["position"]
            assert [position["x"], position["y"]] == counter["stand"]

            emulator.press_and_settle(counter["face"], frames=10)
            assert reader.read_player()["facing"] == counter["face"]

            emulator.press_and_settle("a", frames=6)
            assert counter["gives"] not in {entry["item"] for entry in reader.read_bag()}, (
                "one A press was enough, so the recorded count says nothing"
            )
            for _ in range(counter["presses"] - 1):
                emulator.press_and_settle("a", frames=6)
            bag = {entry["item"] for entry in reader.read_bag()}
            assert counter["gives"] in bag, (
                f"{counter['presses']} A presses from {counter['stand']} facing "
                f"{counter['face']} and still no {counter['gives']}"
            )
            assert counter["wants"] not in bag, "the voucher should have been taken"
            return
        pytest.skip(f"no save in the corpus stands in {BIKE_SHOP} holding the voucher")
    finally:
        emulator.close()


@needs_rom
def test_the_tile_poke_map_recommends_is_the_one_that_does_nothing():
    """(4,1) is walkable, reachable, "nearest unexplored" -- and faces nobody.

    This is why the fact has to name a tile rather than just the room: the two
    candidate tiles are adjacent, both look right, only one of them works, and
    the failing window spent 131 receipts on the wrong one.
    """
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator
    from pokemon_agent.memory.red import MAP_NAMES, PokemonRedReader

    counter = capabilities.counter_at(BIKE_SHOP)
    decoy = (counter["stand"][0], counter["stand"][1] - 1)
    emulator = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        for path in sorted(SAVES_DIR.glob("*.state")):
            try:
                emulator.load_state(str(path))
            except Exception:  # noqa: BLE001 — a save being written is not a finding
                continue
            emulator.settle()
            if MAP_NAMES.get(reader.read_map_info()["map_id"]) != BIKE_SHOP:
                continue
            carried = {entry["item"] for entry in reader.read_bag()}
            if counter["wants"] not in carried or counter["gives"] in carried:
                continue

            snapshot = emulator.get_navigation_snapshot(reader).to_dict()
            collision = capabilities.collision_from(snapshot)
            start = (reader.read_player()["position"]["x"], reader.read_player()["position"]["y"])
            plan = capabilities.plan_within(collision, start, decoy)
            if plan is None:
                pytest.skip(f"{decoy} not reachable from {start} in this save")
            for action in plan:
                emulator.press_and_settle(action.removeprefix("walk_"), frames=10)
            emulator.press_and_settle(counter["face"], frames=10)
            for _ in range(12):
                emulator.press_and_settle("a", frames=6)
            bag = {entry["item"] for entry in reader.read_bag()}
            assert counter["gives"] not in bag, f"{decoy} worked after all -- re-measure"
            assert counter["wants"] in bag
            return
        pytest.skip(f"no save in the corpus stands in {BIKE_SHOP} holding the voucher")
    finally:
        emulator.close()
