"""The client against a real server, a real emulator and the real ROM.

Everything else about `agent_api` is checked against a stub, which proves the
shapes but not the contract: a stub answers whatever it was told to. This file
starts the actual server on a spare port with its own data directory, loads a
save, and drives the client through the things a script would do — walk, walk
into a wall, simulate, ask what is unexplored, read a guide section, look
something up in the game database, and send a path too long for one batch.

Skipped entirely when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pokemon_agent import agent_api

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()

#: Standing in the Pewter Pokecenter with no dialog open. Indoors, so nothing
#: wild can interrupt a walk and there is no trainer to see the player; walled
#: on two sides, so there is something to walk into.
START_STATE = "before_brock.state"

pytestmark = pytest.mark.skipif(
    SAVES_DIR is None or not (SAVES_DIR / START_STATE).exists(),
    reason="no saves/PokemonRed.gb next to the repo",
)

STARTUP_TIMEOUT_SECONDS = 120.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """A real server on a spare port, over a data directory of its own.

    Its own data directory matters: a live run's saves and explored-map memory
    are somebody's data, and a test must not write into them.
    """
    pytest.importorskip("pyboy")
    data_dir = tmp_path_factory.mktemp("live-data")
    (data_dir / "saves").mkdir()
    shutil.copy2(SAVES_DIR / START_STATE, data_dir / "saves" / "start.state")
    # One harness checkpoint alongside the named save. The live run has 300 of
    # them against 165 names, and newest-first they crowded every name off the
    # list; whether the server really filters them is a contract, not a shape,
    # so it belongs here rather than against a stub.
    shutil.copy2(
        SAVES_DIR / START_STATE,
        data_dir / "saves" / "auto__20260101T000000Z__battle-entry__pewter.state",
    )
    port = _free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pokemon_agent.cli",
            "serve",
            "--rom",
            str(SAVES_DIR / "PokemonRed.gb"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
            "--load-state",
            "start",
            "--no-dashboard",
            "--no-realtime",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = agent_api.Client(port=port)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            if process.poll() is not None:
                pytest.fail(f"server exited early:\n{process.stdout.read()}")
            try:
                if client.health().get("emulator_ready"):
                    break
            except agent_api.PokeError:
                pass
            if time.monotonic() > deadline:
                process.kill()
                pytest.fail("server never became ready")
            time.sleep(0.5)
        yield client
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - only a wedged emulator
            process.kill()


@pytest.fixture()
def board(live):
    """Back to the save, so every test starts from the same tile.

    Reloading is what makes "walking into that wall is blocked" a fact rather
    than a fact about whatever the previous test left behind.
    """
    result = live.load("start")
    assert result.map, result.raw
    walls = [way for way in ("up", "down", "left", "right") if way not in result.directions]
    assert result.directions, f"nowhere to walk from {result}"
    assert walls, f"nothing to walk into from {result}"
    return {"open": result.directions, "walls": walls, "start": result}


def test_the_server_answers_and_the_game_is_on_the_overworld(live):
    state = live.state()

    assert state.map
    assert state.party, "the save should have a party"
    assert state.lead.max_hp
    assert state.position is not None
    assert state.in_battle is False


def test_a_walk_moves_the_player(live, board):
    before = board["start"]

    result = live.act(board["open"][0])

    assert result.actions_executed == 1
    assert result.moved == 1
    assert result.position != before.position
    assert result.map == before.map


def test_a_blocked_walk_says_so_instead_of_moving(live, board):
    wall = board["walls"][0]

    result = live.act(f"{wall}:3")

    assert result.moved == 0
    assert result.blocked is True
    # The first press turns the player to face the wall, which the game counts
    # as a step, so what is blocked is everything after that one.
    assert result.blocked_after < 3
    # Facing still changed, which is why "did the position change" is the wrong
    # question and `moved` is the right one.
    assert result.facing == wall


def test_sim_agrees_with_the_wall_without_pressing_anything(live, board):
    wall = board["walls"][0]
    before = live.state()

    blocked = live.sim(f"{wall}:3")
    clear = live.sim(board["open"][0])

    assert blocked.ok is False
    assert blocked.blocked_at == 0
    assert blocked.blocked_by
    assert clear.ok is True
    assert clear.end != tuple(blocked.end or ())
    # Nothing was pressed: the player is exactly where it was.
    assert live.state().position == before.position


def test_frontier_names_reachable_ground_on_this_map(live):
    frontier = live.frontier()

    assert frontier.map == live.state().map
    assert frontier.origin == live.state().position
    for tile in frontier[:3]:
        assert isinstance(tile, tuple) and len(tile) == 2
    assert len(frontier) == frontier.raw["count"]


def test_a_guide_section_can_be_searched_for_and_read(live):
    hits = live.guide.search("mt moon")

    assert hits, "the guide library should have something about Mt. Moon"
    section = live.guide.read(hits[0].ref)
    assert section.ref == hits[0].ref
    assert len(str(section)) > 100

    with pytest.raises(agent_api.ServerError) as caught:
        live.guide.read("nosuchguide/nosuchsection")
    assert caught.value.status == 404


def test_the_game_database_answers_over_http(live):
    brock = live.game.trainers("Pewter Gym")[0]
    assert brock.trainer_class == "Brock"
    assert brock.team == ["Geodude L12", "Onix L14"]

    charmeleon = live.game.species("charmeleon")
    assert charmeleon.types == ["Fire"]
    assert charmeleon.base["atk"] == 64

    assert live.game.move("Ember").damage_class == "special"
    assert live.game.encounters("Route 3").grass.rate == 20
    assert any(item.hidden for item in live.game.items("Viridian Forest"))
    assert "Poke Ball" in live.game.shops("Pewter Mart")
    assert live.game.effectiveness("Water", ["Rock", "Ground"]) == 4.0

    with pytest.raises(agent_api.ServerError, match="Did you mean"):
        live.game.species("Charmandr")


def test_a_long_plan_is_sent_in_legal_chunks(live):
    """Fifty actions is longer than one batch, so the walker splits it.

    Waits rather than steps, so the split is the only thing under test: a wait
    cannot be blocked, cannot start a battle and cannot leave the map.
    """
    report = live.walk("wait:50")

    sizes = [batch.actions_executed for batch in report.batches]
    assert sizes == [40, 10]
    assert report.done is True
    assert report.remaining == []
    assert report.stopped_because == "plan finished"


def test_a_long_walk_stops_at_the_first_wall_rather_than_grinding(live, board):
    wall = board["walls"][0]

    report = live.walk(f"{wall}:60")

    assert len(report.batches) == 1, "nothing should be spent after the first wall"
    assert report.stopped_because.startswith("blocked after")
    assert report.moved == 0
    assert len(report.remaining) == 20
    for batch in report.batches:
        assert batch.actions_executed <= agent_api.MAX_ACTIONS_PER_BATCH


def test_progress_reports_the_run_in_milestones_and_presses(live):
    progress = live.progress()

    assert progress.total > 0
    # The save is well past Brock, so the ladder has to have noticed something.
    assert 0 < progress.count <= progress.total
    assert progress.furthest_label
    assert progress.presses >= 0


def test_saves_lists_the_names_and_not_the_harness_checkpoints(live):
    """The autosave is still loadable; it is just not what the list is for."""
    names = live.saves()

    assert "start" in names
    assert not [name for name in names if name.startswith("auto__")]
    assert live.load("auto__20260101T000000Z__battle-entry__pewter").map


def test_a_refusal_arrives_as_the_servers_own_words(live):
    """Nothing here is in a battle, so /calc has a reason and says it."""
    with pytest.raises(agent_api.ServerError) as caught:
        live.calc()

    assert caught.value.status in (400, 409)
    assert "battle" in caught.value.detail.lower()
