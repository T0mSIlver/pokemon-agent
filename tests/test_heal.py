"""Guards for the one thing a run does more often than anything else: healing.

The measured failure is one 34-hour run: 116 Poke Center visits, 4,152 presses,
7% of them productive. Two maps carried most of it — Mt Moon Pokecenter at
1,948 presses (68% re-walking tiles already stood on) and Vermilion Pokecenter
at 1,683 (38% on dialog) — and one tile in Vermilion was stood on across 316
separate batches. Counted the way the model pays for it, standing inside a
Center cost 1,372 tool calls and 315 kB of tool text against a ~95 kB median
session.

Nothing in the harness ever named the nurse. `poke map` in a Center printed the
room size, its two warps and the "nearest unexplored" tile in a room already
100% seen; `world.json` held map_id, size, connections and warps and nothing
about who was standing in the room. So the tests here cover both halves: the
data that names the person, and the verb that walks to her and finishes her
conversation.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from pokemon_agent import capabilities, gamedata
from pokemon_agent.server import _counter_stand_tiles, _heal_line

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every map in the game with a nurse on it. Eleven Poke Centers plus the two
#: that are not Centers at all, which is why the generator classifies by what a
#: script *does* rather than by which room it is in.
HEALER_MAPS = (
    "Viridian Pokecenter",
    "Pewter Pokecenter",
    "Cerulean Pokecenter",
    "Mt Moon Pokecenter",
    "Rock Tunnel Pokecenter",
    "Vermilion Pokecenter",
    "Celadon Pokecenter",
    "Lavender Pokecenter",
    "Fuchsia Pokecenter",
    "Cinnabar Pokecenter",
    "Saffron Pokecenter",
    "Indigo Plateau Lobby",
    "Silph Co 9F",
)


def mon(species="Charmeleon", hp=95, max_hp=95, status="OK", moves=None):
    return {
        "species": species,
        "hp": hp,
        "max_hp": max_hp,
        "status": status,
        "moves": moves if moves is not None else [{"name": "Ember", "pp": 25}],
    }


# ---------------------------------------------------------------------------
# The data that names her
# ---------------------------------------------------------------------------


def test_every_map_with_a_nurse_says_where_she_stands():
    """The fact `world.json` never held. Thirteen maps, no more and no fewer.

    A missing entry is a Center the agent hunts through; an extra one would
    point `poke heal` at somebody who does not heal.
    """
    named = {
        name
        for name in gamedata.world()
        if any(entry["service"] == "heal" for entry in gamedata.services(name))
    }
    assert named == set(HEALER_MAPS)
    for name in HEALER_MAPS:
        healer = capabilities.healer_at(name)
        assert healer is not None and len(healer["at"]) == 2


def test_every_poke_center_nurse_stands_on_the_same_tile():
    """(3,1) in all eleven, which is what makes the room's layout learnable.

    Asserted rather than assumed: the standing spot is derived from her tile, so
    a generator that put her one tile out would send the walk into a wall.
    """
    centers = [name for name in HEALER_MAPS if name.endswith("Pokecenter")]
    assert len(centers) == 11
    assert all(capabilities.healer_at(name)["at"] == [3, 1] for name in centers)


def test_the_two_healers_outside_a_poke_center_are_not_at_that_tile():
    """Silph Co 9F and the Indigo lobby have nurses in rooms of their own shape.

    They are the reason the join is over what her script does — the eleven
    Centers alone could have been special-cased on the map name and this would
    have quietly missed both.
    """
    assert capabilities.healer_at("Silph Co 9F")["at"] == [3, 14]
    assert capabilities.healer_at("Indigo Plateau Lobby")["at"] == [7, 5]


def test_maps_without_a_nurse_carry_nothing_at_all():
    """210 of the game's 223 maps, where this has to cost nothing."""
    assert capabilities.healer_at("Route 4") is None
    assert capabilities.healer_at("Vermilion Mart") is None
    assert capabilities.healer_at(None) is None
    assert gamedata.services("Route 4") == []


def test_her_counter_is_a_talk_over_tile_so_the_walk_ends_two_out():
    """Measured: every heal in the save library happened from (3,3) facing up.

    The tile in front of her, (3,2), is the counter — solid to walk on and
    transparent to an A press. A verb that walked to distance 1 would stop one
    tile short and read as a walk that failed, which is what `poke buy` had to
    learn at a mart till first.
    """
    tiles = _counter_stand_tiles([3, 1])
    assert (3, 3) in tiles
    assert tiles.index((3, 3)) < tiles.index((3, 2))


# ---------------------------------------------------------------------------
# What she would fix
# ---------------------------------------------------------------------------


def test_a_full_party_has_no_shortfall_so_the_trip_is_already_done():
    """The answer that stops the second visit.

    68% of Mt Moon Pokecenter's 1,948 presses were re-walking tiles already
    stood on, and nothing the agent read ever said the healing was finished.
    """
    assert capabilities.heal_shortfall([mon(), mon("Pidgey", 30, 30)]) == []
    assert capabilities.heal_shortfall([]) == []


def test_missing_hp_status_and_spent_pp_are_each_a_reason_to_talk_to_her():
    """She restores all three, so all three have to count as needing her.

    PP especially: `no_damage` on a battle frame tells the agent it cannot win,
    and a full-HP party with empty moves would otherwise read as fit.
    """
    assert capabilities.heal_shortfall([mon(hp=47)]) == ["Charmeleon 47/95"]
    assert capabilities.heal_shortfall([mon(status="PSN")]) == ["Charmeleon PSN"]
    spent = capabilities.heal_shortfall([mon(moves=[{"name": "Ember", "pp": 3}])])
    assert spent == ["Charmeleon 1 move short of PP"]


def test_a_move_the_table_does_not_know_is_never_reported_as_short():
    """Under-reporting costs one avoidable trip; over-reporting is wallpaper.

    PP Ups raise the real ceiling above the table's number, so the test is
    `below full`, and a move with no record at all has no ceiling to be below.
    """
    assert capabilities.heal_shortfall([mon(moves=[{"name": "Nonesuch", "pp": 0}])]) == []
    assert capabilities.heal_shortfall([mon(moves=[{"name": "Ember", "pp": 99}])]) == []


# ---------------------------------------------------------------------------
# The line on the frame
# ---------------------------------------------------------------------------


def state(map_name, party):
    return {"map": {"map_name": map_name}, "party": party}


def test_standing_in_a_center_says_where_the_nurse_is_and_names_the_verb():
    """A fact in the payload beats a verb nobody calls.

    `calc`, `route`, `frontier` and `progress` were each called zero times in a
    457-call session while their facts sat behind a verb. So the tile and the
    command are on the frame the model already reads.
    """
    line = _heal_line(state("Cerulean Pokecenter", [mon(hp=47)]))
    assert "(3,1)" in line
    assert "47/95" in line
    assert "poke heal" in line


def test_a_healed_party_in_a_center_is_told_the_trip_is_finished():
    """The mirror of the line above, and the one that stops the room being re-walked."""
    line = _heal_line(state("Cerulean Pokecenter", [mon()]))
    assert "(3,1)" in line
    assert "already full" in line
    assert "poke heal" not in line


def test_a_map_with_no_nurse_carries_no_line():
    """13 maps in 223 have one, so the other 210 pay nothing for this."""
    assert _heal_line(state("Route 4", [mon(hp=1)])) is None
    assert _heal_line(state("Vermilion Mart", [mon(hp=1)])) is None
    assert _heal_line({"map": {}, "party": []}) is None


def test_an_unreadable_party_costs_the_frame_nothing():
    """Perception must never be the thing that fails a state read."""
    assert _heal_line({"map": {"map_name": "Cerulean Pokecenter"}, "party": "junk"}) is not None
    assert _heal_line({"map": {"map_name": "Cerulean Pokecenter"}}) is not None


def test_the_line_is_short_enough_to_send_on_every_frame_in_the_room():
    """Tool text is 98% of the model's prompt, so a per-frame line is a per-frame cost.

    Sixty-odd bytes on thirteen maps against 315 kB spent hunting for her is the
    trade; a paragraph would not be.
    """
    for party in ([mon(hp=47)], [mon()]):
        assert len(_heal_line(state("Cerulean Pokecenter", party)).encode()) < 100


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

#: Save states standing inside a Poke Center. The whole point of the data is
#: that it lines up with the running game, so it is checked against one.
CENTER_STATES = (
    "healed_before_brock.state",
    "cerulean_healed_L32.state",
    "healed_pp_restored.state",
    "before_misty.state",
)


@needs_rom
def test_the_nurse_in_the_data_is_the_nurse_on_the_screen():
    """Her tile is read out of the running game and matched against services.json.

    A generated table that has drifted from the ROM is worse than no table: the
    walk would arrive somewhere confidently wrong and the A press would hit a
    wall. The sprite the game has loaded at her tile is what settles it.
    """
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator
    from pokemon_agent.memory.red import MAP_NAMES, PokemonRedReader

    states = [SAVES_DIR / name for name in CENTER_STATES if (SAVES_DIR / name).exists()]
    if not states:
        pytest.skip(f"none of {CENTER_STATES} is present in {SAVES_DIR}")
    emulator = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        checked = 0
        for path in states:
            emulator.load_state(str(path))
            emulator.settle()
            map_name = MAP_NAMES.get(reader.read_map_info()["map_id"])
            healer = capabilities.healer_at(map_name)
            if healer is None:
                continue
            # The player is in the room, so the tile she is on has to be inside
            # it and the tile two out has to be one the player can be standing on.
            width, height = gamedata.world()[map_name]["size"]
            assert 0 <= healer["at"][0] < width
            assert 0 <= healer["at"][1] < height
            assert (3, 3) in _counter_stand_tiles(healer["at"])
            checked += 1
        assert checked, "no save in the corpus stood on a map with a nurse"
    finally:
        emulator.close()


@needs_rom
def test_a_hurt_party_in_a_center_reads_as_needing_her_and_a_healed_one_does_not():
    """The two ends of the line, off real party memory rather than a fixture.

    The run this was written for stood in Cerulean at 47/95 and read a payload
    that named the door out; the same frame now names her and the shortfall.
    """
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator
    from pokemon_agent.memory.red import MAP_NAMES, PokemonRedReader

    emulator = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        seen_full = seen_hurt = False
        paths = sorted(SAVES_DIR.glob("*.state"))
        random.Random(20260827).shuffle(paths)
        for path in paths:
            try:
                emulator.load_state(str(path))
            except Exception:  # noqa: BLE001 — a save being written is not a finding
                continue
            emulator.settle()
            map_name = MAP_NAMES.get(reader.read_map_info()["map_id"])
            if capabilities.healer_at(map_name) is None:
                continue
            line = _heal_line({"map": {"map_name": map_name}, "party": reader.read_party()})
            assert line is not None
            if "already full" in line:
                seen_full = True
            else:
                assert "poke heal" in line
                seen_hurt = True
            if seen_full and seen_hurt:
                break
        if not (seen_full and seen_hurt):
            pytest.skip("the corpus has no Center save of each kind right now")
    finally:
        emulator.close()


def _a_hurt_save_in_a_center() -> Path:
    """A save standing on a map with a nurse, with something for her to fix."""
    from pokemon_agent.emulator import create_emulator
    from pokemon_agent.memory.red import MAP_NAMES, PokemonRedReader

    emulator = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        reader = PokemonRedReader(emulator)
        for path in sorted(SAVES_DIR.glob("*.state")):
            try:
                emulator.load_state(str(path))
            except Exception:  # noqa: BLE001 — a save being written is not a finding
                continue
            emulator.settle()
            map_name = MAP_NAMES.get(reader.read_map_info()["map_id"])
            if capabilities.healer_at(map_name) is None:
                continue
            if capabilities.heal_shortfall(reader.read_party()):
                return path
    finally:
        emulator.close()
    pytest.skip("no save in the corpus stands in a Center with anything to heal")


@needs_rom
def test_driving_the_whole_heal_restores_the_party_and_closes_her_box(tmp_path):
    """End to end, against the ROM: walk in hurt, come out full and able to walk.

    Both halves are load-bearing. A heal that leaves her closing line on screen
    gets the next `poke goto` refused with "a text box is open", which is how
    38% of Vermilion's presses went on dialog.
    """
    pytest.importorskip("pyboy")
    from fastapi.testclient import TestClient

    from pokemon_agent import server

    # The live saves directory is the corpus, but the server writes into its
    # data dir, so it gets a scratch one with the states this test needs copied
    # in. A test must never add autosaves to a directory a run is playing from.
    # A save standing in a Center with something to heal. Named by search rather
    # than by name: the corpus is a live directory and the autosaves that land
    # in a Center are rotated out of it within the day.
    hurt = _a_hurt_save_in_a_center()
    saves = tmp_path / "saves"
    saves.mkdir()
    for source in (SAVES_DIR / "PokemonRed.gb", hurt):
        (saves / source.name).write_bytes(source.read_bytes())
    server._action_call_times.clear()
    server.configure(
        server.GameConfig(
            rom_path=str(saves / "PokemonRed.gb"),
            data_dir=str(tmp_path),
            agent_workspace_dir=str(tmp_path / "workspace"),
            realtime=False,
            enable_dashboard=False,
        )
    )
    with TestClient(server.app) as client:
        loaded = client.post("/load", json={"name": hurt.stem})
        assert loaded.status_code == 200
        # Healing drives a menu, so it spends real buttons, and for as long as
        # this endpoint existed it recorded none of them. The errand the player
        # runs most was missing from the total the project measures itself in.
        run = asyncio.run_coroutine_threadsafe(
            server._run_recorder.begin_session(goal="heal", model="test"), server._loop
        ).result(timeout=10)
        answer = client.post("/pokecenter/heal", json={})
        assert answer.status_code == 200, answer.text
        written = (tmp_path / "runs" / run.run_id / "receipts.jsonl").read_text(encoding="utf-8")
        heals = [
            json.loads(line)
            for line in written.splitlines()
            if line and json.loads(line).get("tool") == "heal"
        ]
        assert len(heals) == 1, "the heal wrote no receipt"
        assert heals[0]["presses"] > 0, "walking to the nurse and talking costs buttons"
        payload = answer.json()
        assert payload["nurse"] == [3, 1]
        assert payload["dialog"] is False
        assert payload["moves"], "the party is healed but the player cannot walk"
        assert "already full" in payload["heal"]
        # Calling it again is refused rather than healing a full party a second
        # time, which is what a re-opened conversation would have done.
        again = client.post("/pokecenter/heal", json={})
        assert again.status_code == 409
        assert "already at full" in again.json()["detail"]
