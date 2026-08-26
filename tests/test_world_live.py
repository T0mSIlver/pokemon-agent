"""The plan simulator checked against the real ROM, not against a model of it.

`sim` is the second most-used verb in the harness and it was wrong about ledges:
it read the collision map, saw a blocked tile, and said "wall" for a direction
the game answers by jumping two tiles. A verifier that is wrong is worse than no
verifier, so its predictions are checked here the only way that settles it —
press the button and look at where the player ended up.

There are two sweeps, one above ground and one below, because the overworld one
proved less than it looked like it did: two of the rules `simulate` was missing
are inert on the overworld and decide everything in a cave, so 493 clean
overworld directions said nothing at all about Mt. Moon.

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

#: What a live run drops into saves/ as it plays. Named saves are written on
#: purpose and rarely; these arrive on their own, several per minute.
AUTOSAVE_PREFIX = "auto__"

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


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> list[Path]:
    """A frozen copy of the save corpus, in the order the sweeps walk it.

    ``saves/`` is a live directory: the harness is normally playing while these
    tests run, and it writes a fresh ``auto__<timestamp>__….state`` into it
    every time something notable happens. Sweeping that directory made the
    sample change between runs for reasons that had nothing to do with the
    code, and it left `load_state` free to read a file that was being rewritten
    underneath it. So the autosaves are left out and what remains is copied
    once: the corpus one run sweeps cannot change during it.
    """
    if SAVES_DIR is None:  # pragma: no cover — guarded by needs_rom
        pytest.skip("no saves directory")
    frozen = tmp_path_factory.mktemp("corpus")
    states: list[Path] = []
    for path in sorted(SAVES_DIR.glob("*.state")):
        if path.name.startswith(AUTOSAVE_PREFIX):
            continue
        try:
            copy = frozen / path.name
            copy.write_bytes(path.read_bytes())
        except OSError:  # a save being written right now is not a finding
            continue
        states.append(copy)
    random.Random(20260826).shuffle(states)
    return states


def _state(states: list[Path], name: str) -> Path:
    """The frozen copy of one named save, or a skip if the corpus has lost it."""
    for path in states:
        if path.name == name:
            return path
    pytest.skip(f"{name} is not in {SAVES_DIR}")


def _seeded_first(states: list[Path], names) -> list[Path]:
    """The named states first, then the rest, each named one only once.

    A shuffle is not allowed to decide whether the case that caused a bug gets
    covered.
    """
    by_name = {path.name: path for path in states}
    seeded = [by_name[name] for name in names if name in by_name]
    return seeded + [path for path in states if path not in seeded]


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
def pocket_state(corpus) -> Path:
    """The frozen copy of the pocket save, so nothing can rewrite it mid-test."""
    return _state(corpus, POCKET_STATE)


@pytest.fixture(scope="module")
def pocket(emulator, reader, pocket_state):
    """The pocket, plus the memory of it the agent would have had.

    Built by walking the strip end to end and recording every window, then
    poisoned the way the real store was poisoned: the old fixed-cadence reader
    sampled the player one tile into a ledge jump and wrote that tile down as
    ground. The phantom corridor out of the pocket was made of exactly those.
    """
    from pokemon_agent.explored_map import ExploredMaps

    store = ExploredMaps(Path("/nonexistent/never-written.json"))
    emulator.load_state(str(pocket_state))
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


def _restart(emulator, reader, pocket_state):
    emulator.load_state(str(pocket_state))
    emulator.settle()
    assert _where(reader) == (14, POCKET)
    return emulator.get_navigation_snapshot(reader).to_dict()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@needs_rom
def test_simulate_predicts_what_the_emulator_actually_does(emulator, reader, corpus):
    """One walk in each direction at many real positions, predicted then pressed.

    Before ledges were modelled this failed on every ledge in the corpus, each
    one reported as a wall the game happily jumps.
    """
    states = _seeded_first(corpus, LEDGE_STATES)

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
def test_the_ledge_the_transcript_caught_is_a_jump_now(emulator, reader, pocket, pocket_state):
    """`sim down:3` at (10,10) said "blocked by wall". One press moved two tiles."""
    _restart(emulator, reader, pocket_state)
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
def test_a_plan_out_of_the_pocket_walks_the_long_way_and_arrives(
    emulator, reader, pocket, pocket_state
):
    """Up the ledge is not a route, however loudly the remembered map says so."""
    snapshot = _restart(emulator, reader, pocket_state)
    collision = capabilities.collision_from(snapshot, pocket)

    plan = capabilities.plan_within(collision, POCKET, LEDGE_TOP)

    assert plan is not None
    assert plan[0] != "walk_up"  # the phantom the poisoned store offered
    assert len(plan) == len(ROUND_THE_LEDGE)
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == LEDGE_TOP


@needs_rom
def test_a_plan_into_the_pocket_uses_the_ledge_and_arrives(emulator, reader, pocket, pocket_state):
    """Downhill it is a real edge, and two presses instead of thirteen."""
    _restart(emulator, reader, pocket_state)
    _walk(emulator, ROUND_THE_LEDGE)
    if _where(reader) != (14, LEDGE_TOP):
        pytest.skip("could not reach the top of the ledge")
    above = emulator.get_navigation_snapshot(reader).to_dict()

    plan = capabilities.plan_within(capabilities.collision_from(above, pocket), LEDGE_TOP, (10, 13))

    assert plan == ["walk_down", "walk_down"]
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == (10, 13)


@needs_rom
def test_frontier_from_the_pocket_never_stands_on_the_ledge(emulator, reader, pocket, pocket_state):
    snapshot = _restart(emulator, reader, pocket_state)

    payload = capabilities.frontier_payload(snapshot, pocket, pocket["walked"])
    tiles = [tuple(tile) for tile in payload["tiles"]]

    # The poisoned tiles are in the store and in the window, and the window wins.
    assert not [tile for tile in tiles if tile[1] == 11 and 10 <= tile[0] <= 14]
    assert payload["confirmed_count"] + payload["believed_count"] == payload["count"]
    assert payload["confirmed_count"] > 0


@needs_rom
def test_every_northern_tile_frontier_offers_can_really_be_walked_to(
    emulator, reader, pocket, pocket_state
):
    """The pocket was never sealed: there is a gap at x = 15 nobody ever tried.

    So the honest answer from inside it is that ground to the north *is*
    reachable — and the plan that says so has to survive being walked.
    """
    snapshot = _restart(emulator, reader, pocket_state)
    payload = capabilities.frontier_payload(snapshot, pocket, pocket["walked"])
    collision = capabilities.collision_from(snapshot, pocket)
    north = [tuple(tile) for tile in payload["tiles"] if tile[1] < 12]

    assert north, "the gap at x = 15 is open, so north of the ledge is reachable"
    target = north[0]
    plan = capabilities.plan_within(collision, POCKET, target)
    assert plan is not None

    _restart(emulator, reader, pocket_state)
    _walk(emulator, plan)
    assert tuple(reader.read_coordinates()) == target


# ---------------------------------------------------------------------------
# The cave sweep
# ---------------------------------------------------------------------------

#: Every CAVERN save in the corpus. Named, not globbed, and not filtered by
#: tileset at runtime: a sweep that discovers its own subject can quietly stop
#: covering it, which is how caves went unchecked in the first place.
CAVE_STATES = (
    "b1f_before_exits.state",
    "mt_moon_1f_entered.state",
    "mtmoon_1f_explored.state",
    "mtmoon_1f_arrived.state",
    "mt_moon_b1f.state",
    "mt_moon_b1f_top.state",
    "mt_moon_b2f.state",
    "mtmoon_b2f_furthest.state",
    "b1f_retreat.state",
)

#: Mt. Moon B2F (24, 11). Both tiles are plain cave floor, the ASCII window
#: shows open ground below the player, `sim` said the walk was clean and the
#: emulator did not move at all. The tile pair is the whole of the difference.
SEAM_STATE = "b1f_before_exits.state"
SEAM_AT = (24, 11)

#: Mt. Moon 1F (14, 35), standing on the hole down to Route 4 at the bottom
#: edge of the map. Collision refuses the press, so `sim` called it "edge" —
#: which is a wall, at the one tile that is the way out.
LADDER_STATE = "mt_moon_1f_entered.state"
LADDER_AT = (14, 35)
ROUTE_4 = 15

#: A cave save only has one resting position in it, so the sweep walks a short
#: random tour from each to reach more. Prefixes are 0..N-1 presses long.
CAVE_TOURS = int(os.environ.get("POKE_CAVE_SWEEP_TOURS", "10"))
CAVE_SWEEP_POSITIONS = int(os.environ.get("POKE_CAVE_SWEEP_POSITIONS", "40"))


@needs_rom
def test_the_cave_seam_the_agent_walked_into_is_not_called_clean(emulator, reader, corpus):
    """(24, 11) down, the direction the run reported and the sweep reproduced."""
    emulator.load_state(str(_state(corpus, SEAM_STATE)))
    emulator.settle()
    if _where(reader) != (61, SEAM_AT) or not _is_idle(emulator, reader):
        pytest.skip(f"{SEAM_STATE} no longer starts on the seam at {SEAM_AT}")
    snapshot = emulator.get_navigation_snapshot(reader)

    payload = capabilities.simulate_payload(["down", "down", "down"], snapshot.to_dict(), None)

    # The emulator knew all along: the seam is in the snapshot and "down" is
    # not in valid_moves. Only `simulate` did not ask.
    assert (SEAM_AT, (SEAM_AT[0], SEAM_AT[1] + 1)) in snapshot.blocked_pairs
    assert "down" not in snapshot.valid_moves
    assert payload["blocked_at"] == 0
    assert payload["blocked_by"] == "tile_pair"
    assert payload["steps"] == 0

    emulator.press_and_settle("down")
    assert _where(reader) == (61, SEAM_AT)  # and the game agrees


@needs_rom
def test_the_cave_ladder_at_the_edge_of_the_map_is_a_warp_not_a_wall(emulator, reader, corpus):
    """(14, 35) down: one press and the player is on Route 4."""
    emulator.load_state(str(_state(corpus, LADDER_STATE)))
    emulator.settle()
    if _where(reader) != (59, LADDER_AT) or not _is_idle(emulator, reader):
        pytest.skip(f"{LADDER_STATE} no longer starts on the ladder at {LADDER_AT}")
    snapshot = emulator.get_navigation_snapshot(reader)

    payload = capabilities.simulate_payload(["down", "down"], snapshot.to_dict(), None)

    assert snapshot.warp_exit_directions == ["down"] and snapshot.warp_exit_armed
    assert payload["warp_at"] == 0
    assert payload["blocked_at"] is None
    assert payload["blocked_by"] is None

    emulator.press_and_settle("down")
    assert _where(reader)[0] == ROUTE_4


@needs_rom
def test_simulate_predicts_what_the_emulator_does_underground(emulator, reader, corpus, tmp_path):
    """The sweep above, with the tileset filter turned the other way round.

    493 overworld directions found nothing, and two of the three rules
    `simulate` was missing are inert above ground: the tile pair table has
    CAVERN and FOREST rows only, and a cave warp fires on the map edge where a
    door fires on the tile. So this is the same method on the ground where
    those rules are the whole answer. Before they were modelled this sweep
    found seven wrong directions out of 153, on all three floors of Mt. Moon.
    """
    anchor = tmp_path / "anchor.state"
    walker = random.Random(20260826)
    seen: set[tuple] = set()
    checked = 0
    seam_directions = 0
    warp_directions = 0
    mismatches: list[tuple] = []

    for name in CAVE_STATES:
        for length in range(CAVE_TOURS):
            if len(seen) >= CAVE_SWEEP_POSITIONS:
                break
            prefix = [walker.choice(DIRECTIONS) for _ in range(length)]
            try:
                emulator.load_state(str(_state(corpus, name)))
                emulator.settle()
                for direction in prefix:
                    emulator.press_and_settle(direction)
                if not _is_idle(emulator, reader):
                    continue
                start = _where(reader)
                emulator.tick(90)
                if _where(reader) != start or not _is_idle(emulator, reader):
                    continue  # mid-transition, not a resting place
                snapshot = emulator.get_navigation_snapshot(reader)
            except Exception:  # noqa: BLE001 — a save we cannot read is not a finding
                continue
            if snapshot.tileset != "CAVERN" or start in seen:
                continue
            seen.add(start)
            state = snapshot.to_dict()
            # One anchor per position beats replaying the tour four times, and
            # it is the same frame each direction is pressed from.
            emulator.save_state(str(anchor))

            for direction in DIRECTIONS:
                emulator.load_state(str(anchor))
                emulator.settle()
                predicted = capabilities.simulate_payload([f"walk_{direction}"], state, None)
                emulator.press_and_settle(direction)
                after = _where(reader)
                if reader.read_battle()["in_battle"]:
                    continue  # a wild encounter ate the step; it proves nothing
                checked += 1
                if predicted["blocked_by"] == "tile_pair":
                    seam_directions += 1
                if snapshot.warp_exit_armed and direction in snapshot.warp_exit_directions:
                    warp_directions += 1
                if after[0] != start[0]:
                    # The press left the map, so the only right answer is that
                    # it warps. "Blocked by edge" is a wall, at the way out.
                    if predicted["warp_at"] != 0:
                        mismatches.append((name, start[1], direction, f"map {after[0]}", predicted))
                    continue
                if tuple(predicted["end"]) != after[1]:
                    mismatches.append((name, start[1], direction, after[1], predicted))

    assert len(seen) >= min(CAVE_SWEEP_POSITIONS, 20), "not enough cave positions to sweep"
    assert checked >= 60
    # Both guards name a rule, not an outcome: a sample that has stopped
    # containing one of them proves nothing about it however green it is.
    assert seam_directions, "the sample no longer contains a tile pair collision"
    assert warp_directions, "the sample no longer contains an armed cave warp exit"
    assert mismatches == []
