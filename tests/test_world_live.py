"""The plan simulator checked against the real ROM, not against a model of it.

`sim` is the second most-used verb in the harness and it was wrong about ledges:
it read the collision map, saw a blocked tile, and said "wall" for a direction
the game answers by jumping two tiles. A verifier that is wrong is worse than no
verifier, so its predictions are checked here the only way that settles it —
press the button and look at where the player ended up.

Skipped entirely when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from pokemon_agent import capabilities, world

REPO_ROOT = Path(__file__).resolve().parents[1]

DIRECTIONS = ("up", "down", "left", "right")
ADDR_JOY_IGNORE = 0xCD6B


def _find_saves_dir() -> Path | None:
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = parent / "saves"
        if (candidate / "PokemonRed.gb").exists():
            return candidate
    return None


SAVES_DIR = _find_saves_dir()
needs_rom = pytest.mark.skipif(SAVES_DIR is None, reason="no saves/PokemonRed.gb next to the repo")

#: Route 3, trapped in the strip the agent spent three sessions and 720 button
#: presses inside. The ledge along y = 11 is what put it there.
POCKET_STATE = "route3_stuck_before_interventions.state"
POCKET = (10, 12)
LEDGE_TOP = (10, 10)
LEDGE_LANDING = (10, 12)
#: (10,12) -> (10,10): east to the gap at x = 15, up through it, back west along
#: the shelf. Twelve presses, and not one of them climbs the ledge.
ROUND_THE_LEDGE = ["right"] * 5 + ["up", "up"] + ["left"] * 5

#: Save states where the player is resting on top of a ledge. Standing on one
#: is rare — the agent walks off them immediately — so the sweep is seeded with
#: the ones the corpus has rather than hoping a shuffle turns one up.
LEDGE_STATES = (
    "route2_from_forest.state",
    "route3_progress.state",
    "viridian_healed.state",
)

#: Pressing every direction at every resting position in saves/ takes minutes.
#: This sample is enough to catch a movement rule that has gone wrong; raise it
#: to re-run the whole corpus.
SWEEP_POSITIONS = int(os.environ.get("POKE_SIM_SWEEP_POSITIONS", "40"))


@pytest.fixture(scope="module")
def emulator():
    pytest.importorskip("pyboy")
    from pokemon_agent.emulator import create_emulator

    emu = create_emulator(str(SAVES_DIR / "PokemonRed.gb"))
    try:
        yield emu
    finally:
        emu.close()


@pytest.fixture(scope="module")
def reader(emulator):
    from pokemon_agent.memory.red import PokemonRedReader

    return PokemonRedReader(emulator)


def _is_idle(emulator, reader) -> bool:
    return (
        not reader.read_battle()["in_battle"]
        and not reader.read_dialog()["active"]
        and emulator.read_u8(ADDR_JOY_IGNORE) == 0
    )


def _where(reader) -> tuple[int, tuple[int, int]]:
    return reader.read_map_info()["map_id"], tuple(reader.read_coordinates())


def _walk(emulator, plan) -> None:
    for action in plan:
        emulator.press_and_settle(action.replace("walk_", ""))


@pytest.fixture(scope="module")
def pocket(emulator, reader):
    """The pocket, plus the memory of it the agent would have had.

    Built by walking the strip end to end and recording every window, then
    poisoned the way the real store was poisoned: the old fixed-cadence reader
    sampled the player one tile into a ledge jump and wrote that tile down as
    ground. The phantom corridor out of the pocket was made of exactly those.
    """
    from pokemon_agent.explored_map import ExploredMaps

    state = SAVES_DIR / POCKET_STATE
    if not state.exists():
        pytest.skip(f"{POCKET_STATE} is not in {SAVES_DIR}")

    store = ExploredMaps(Path("/nonexistent/never-written.json"))
    emulator.load_state(str(state))
    emulator.settle()
    if _where(reader) != (14, POCKET) or not _is_idle(emulator, reader):
        pytest.skip("the pocket save no longer starts where it used to")

    store.record(emulator.get_navigation_snapshot(reader).to_dict())
    for direction in ["right"] * 12 + ["left"] * 12:
        emulator.press_and_settle(direction)
        if not _is_idle(emulator, reader):
            pytest.skip("a wild encounter interrupted the walk that builds the map")
        store.record(emulator.get_navigation_snapshot(reader).to_dict())

    grid = store.grid(14)
    for x in range(10, 15):
        grid["walkable"].add((x, 11))
        grid["walked"].add((x, 11))
    return grid


def _restart(emulator, reader):
    emulator.load_state(str(SAVES_DIR / POCKET_STATE))
    emulator.settle()
    assert _where(reader) == (14, POCKET)
    return emulator.get_navigation_snapshot(reader).to_dict()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@needs_rom
def test_simulate_predicts_what_the_emulator_actually_does(emulator, reader):
    """One walk in each direction at many real positions, predicted then pressed.

    Before ledges were modelled this failed on every ledge in the corpus, each
    one reported as a wall the game happily jumps.
    """
    states = sorted(SAVES_DIR.glob("*.state"))
    random.Random(20260826).shuffle(states)
    seeded = [SAVES_DIR / name for name in LEDGE_STATES if (SAVES_DIR / name).exists()]
    states = seeded + [path for path in states if path not in seeded]

    seen: set[tuple] = set()
    checked = 0
    ledge_directions = 0
    mismatches: list[tuple] = []

    for path in states:
        if len(seen) >= SWEEP_POSITIONS:
            break
        try:
            emulator.load_state(str(path))
            emulator.settle()
            if not _is_idle(emulator, reader):
                continue
            start = _where(reader)
            emulator.tick(90)
            if _where(reader) != start or not _is_idle(emulator, reader):
                continue  # a save taken mid-transition, not a resting place
            snapshot = emulator.get_navigation_snapshot(reader)
        except Exception:  # noqa: BLE001 — a save we cannot read is not a finding
            continue
        if snapshot.tileset != "OVERWORLD" or start in seen:
            continue  # ledges are an OVERWORLD rule; the rest is another sweep
        seen.add(start)

        for direction in DIRECTIONS:
            emulator.load_state(str(path))
            emulator.settle()
            predicted = world.simulate([f"walk_{direction}"], snapshot, start[1], snapshot.facing)
            if direction in snapshot.ledge_hops:
                ledge_directions += 1
            emulator.press_and_settle(direction)
            after = _where(reader)
            if after[0] != start[0] or reader.read_battle()["in_battle"]:
                continue  # off this map, or interrupted: not this call's claim
            checked += 1
            if predicted.end_pos != after[1]:
                mismatches.append((path.name, start[1], direction, predicted.end_pos, after[1]))

    assert len(seen) >= min(SWEEP_POSITIONS, 15), "not enough usable save states to sweep"
    assert checked >= 40
    assert ledge_directions, "the sample no longer contains a ledge, so it proves nothing about one"
    assert mismatches == []


@needs_rom
def test_the_ledge_the_transcript_caught_is_a_jump_now(emulator, reader, pocket):
    """`sim down:3` at (10,10) said "blocked by wall". One press moved two tiles."""
    _restart(emulator, reader)
    _walk(emulator, ROUND_THE_LEDGE)
    if _where(reader) != (14, LEDGE_TOP):
        pytest.skip("could not reach the top of the ledge")
    above = emulator.get_navigation_snapshot(reader).to_dict()

    payload = capabilities.simulate_payload(["down", "down", "down"], above, pocket)

    assert payload["hops"] == [
        {
            "at": 0,
            "direction": "down",
            "from": list(LEDGE_TOP),
            "to": list(LEDGE_LANDING),
            "one_way": True,
        }
    ]
    assert payload["blocked_at"] != 0
    assert "one way" in payload["note"]

    emulator.press_and_settle("down")
    assert tuple(reader.read_coordinates()) == LEDGE_LANDING


@needs_rom
def test_a_plan_out_of_the_pocket_walks_the_long_way_and_arrives(emulator, reader, pocket):
    """Up the ledge is not a route, however loudly the remembered map says so."""
    snapshot = _restart(emulator, reader)
    collision = capabilities.collision_from(snapshot, pocket)

    plan = capabilities.plan_within(collision, POCKET, LEDGE_TOP)

    assert plan is not None
    assert plan[0] != "walk_up"  # the phantom the poisoned store offered
    assert len(plan) == len(ROUND_THE_LEDGE)
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == LEDGE_TOP


@needs_rom
def test_a_plan_into_the_pocket_uses_the_ledge_and_arrives(emulator, reader, pocket):
    """Downhill it is a real edge, and two presses instead of thirteen."""
    _restart(emulator, reader)
    _walk(emulator, ROUND_THE_LEDGE)
    if _where(reader) != (14, LEDGE_TOP):
        pytest.skip("could not reach the top of the ledge")
    above = emulator.get_navigation_snapshot(reader).to_dict()

    plan = capabilities.plan_within(capabilities.collision_from(above, pocket), LEDGE_TOP, (10, 13))

    assert plan == ["walk_down", "walk_down"]
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == (10, 13)


@needs_rom
def test_frontier_from_the_pocket_never_stands_on_the_ledge(emulator, reader, pocket):
    snapshot = _restart(emulator, reader)

    payload = capabilities.frontier_payload(snapshot, pocket, pocket["walked"])
    tiles = [tuple(tile) for tile in payload["tiles"]]

    # The poisoned tiles are in the store and in the window, and the window wins.
    assert not [tile for tile in tiles if tile[1] == 11 and 10 <= tile[0] <= 14]
    assert payload["confirmed_count"] + payload["believed_count"] == payload["count"]
    assert payload["confirmed_count"] > 0


@needs_rom
def test_every_northern_tile_frontier_offers_can_really_be_walked_to(emulator, reader, pocket):
    """The pocket was never sealed: there is a gap at x = 15 nobody ever tried.

    So the honest answer from inside it is that ground to the north *is*
    reachable — and the plan that says so has to survive being walked.
    """
    snapshot = _restart(emulator, reader)
    payload = capabilities.frontier_payload(snapshot, pocket, pocket["walked"])
    collision = capabilities.collision_from(snapshot, pocket)
    north = [tuple(tile) for tile in payload["tiles"] if tile[1] < 12]

    assert north, "the gap at x = 15 is open, so north of the ledge is reachable"
    target = north[0]
    plan = capabilities.plan_within(collision, POCKET, target)
    assert plan is not None

    _restart(emulator, reader)
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == target
