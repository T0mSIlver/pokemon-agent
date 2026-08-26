"""Navigation rules checked against the real ROM, not against a model of it.

Every claim here was wrong at some point in a way that read as plausible: a
ledge that the collision map calls a wall, a warp that only answers to one
direction, a walk observed halfway through its animation. So each one is proved
by pressing the button and looking at what the game did.

Skipped entirely when the ROM or pyboy is absent, which is how CI runs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

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

#: Route 2, standing on top of the ledge at (3, 46) facing the jump. Named
#: saves only: the two autosaves this used to name had both been rotated out of
#: saves/ by the time anyone next ran it, and these tests skipped in silence.
LEDGE_STATES = ("route2_from_forest.state",)
LEDGE_POSITION = (3, 46)
LEDGE_LANDING = (3, 48)

#: Route 2, standing on the warp at (3, 11) into the forest gate (map 47).
WARP_STATES = ("route2_north_of_forest.state",)
WARP_POSITION = (3, 11)
WARP_TARGET_MAP = 47

#: Mt. Moon 1F, a few steps inside the entrance. Wild encounters here land about
#: one step in ten, which is what makes a battle frame cheap to reach on demand.
ENCOUNTER_STATES = ("mtmoon_1f_arrived.state", "mt_moon_1f_entered.state")

#: The full sweep walks every direction at every distinct resting position in
#: saves/, which takes minutes. The default sample is enough to catch a rule
#: that has gone wrong; raise it to re-run the whole corpus.
SWEEP_POSITIONS = int(os.environ.get("POKE_SWEEP_POSITIONS", "40"))


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
    """A frozen copy of the save corpus, in the order the sweep walks it.

    ``saves/`` is a live directory: the harness is normally playing while these
    tests run, and it writes a fresh ``auto__<timestamp>__….state`` into it
    every time something notable happens. Sweeping that directory made the
    sample change between runs for reasons that had nothing to do with the
    code — one autosave written on the morning of 2026-08-26 was enough to turn
    a green sweep red — and it left `load_state` free to read a file that was
    being rewritten underneath it. So the autosaves are left out, and what is
    left is copied once: the corpus one run sweeps cannot change during it.
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
    random.Random(20260825).shuffle(states)
    return states


def _pick(names: tuple[str, ...]) -> Path:
    for name in names:
        path = SAVES_DIR / name
        if path.exists():
            return path
    pytest.skip(f"none of {names} is present in {SAVES_DIR}")


def _is_idle(emulator, reader) -> bool:
    """The overworld is up and taking input from us rather than from a script."""
    return (
        not reader.read_battle()["in_battle"]
        and not reader.read_dialog()["active"]
        and emulator.read_u8(ADDR_JOY_IGNORE) == 0
    )


def _position(reader) -> tuple[int, tuple[int, int]]:
    return reader.read_map_info()["map_id"], reader.read_coordinates()


@needs_rom
def test_the_route_2_ledge_is_offered_as_a_move(emulator, reader):
    emulator.load_state(str(_pick(LEDGE_STATES)))
    emulator.settle()
    assert reader.read_coordinates() == LEDGE_POSITION

    snapshot = emulator.get_navigation_snapshot(reader)

    assert "down" in snapshot.valid_moves
    assert snapshot.ledge_hops == {"down": LEDGE_LANDING}
    # The collision map still calls the ledge a wall; the tile pair overrides it.
    assert snapshot.terrain[5][4] == 0


@needs_rom
def test_one_walk_down_clears_the_ledge_without_resting_on_the_gap(emulator, reader):
    state = str(_pick(LEDGE_STATES))
    emulator.load_state(state)
    emulator.settle()

    assert emulator.press_and_settle("down") is True
    assert reader.read_coordinates() == LEDGE_LANDING

    # The old fixed 20-frame cadence returned here, one tile short, on a tile the
    # player cannot stand on. Recording it poisoned the explored map.
    emulator.load_state(state)
    emulator.settle()
    emulator.press("down", 8)
    emulator.tick(12)
    assert reader.read_coordinates() == (LEDGE_POSITION[0], LEDGE_POSITION[1] + 1)


@needs_rom
def test_the_warp_at_route_2_reports_one_exit_direction_not_four(emulator, reader):
    state = str(_pick(WARP_STATES))
    emulator.load_state(state)
    emulator.settle()
    assert reader.read_coordinates() == WARP_POSITION

    snapshot = emulator.get_navigation_snapshot(reader)

    assert snapshot.warp_exit_directions == ["down"]
    assert snapshot.warp_exit_armed is True
    assert "left" not in snapshot.valid_moves
    assert "right" not in snapshot.valid_moves
    assert "down" not in snapshot.valid_moves  # the warp is not a walkable step
    assert snapshot.valid_moves == ["up"]


@needs_rom
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_each_direction_at_the_warp_does_what_the_snapshot_says(emulator, reader, direction):
    state = str(_pick(WARP_STATES))
    emulator.load_state(state)
    emulator.settle()
    snapshot = emulator.get_navigation_snapshot(reader)
    start = _position(reader)

    emulator.press_and_settle(direction)
    after = _position(reader)

    predicted = direction in snapshot.valid_moves or direction in snapshot.warp_exit_directions
    assert (after != start) is predicted
    if direction in snapshot.warp_exit_directions:
        assert after[0] == WARP_TARGET_MAP


@needs_rom
def test_ordinary_terrain_still_agrees_with_the_emulator(emulator, reader, corpus):
    """Press all four directions at many real positions and compare.

    This is the regression net for the other two fixes: a ledge rule or a warp
    rule that over-fires shows up here as a direction the game refuses.
    """
    states = corpus

    seen: set[tuple[int, tuple[int, int]]] = set()
    mismatches: list[tuple] = []
    ledge_landings: list[tuple] = []

    for path in states:
        if len(seen) >= SWEEP_POSITIONS:
            break
        try:
            emulator.load_state(str(path))
            emulator.settle()
            if not _is_idle(emulator, reader):
                continue
            start = _position(reader)
            emulator.tick(90)
            if _position(reader) != start or not _is_idle(emulator, reader):
                continue  # a snapshot taken mid-transition, not a resting place
        except Exception:  # noqa: BLE001 — a save we cannot read is not a finding
            continue
        if start in seen:
            continue
        seen.add(start)

        emulator.load_state(str(path))
        emulator.settle()
        snapshot = emulator.get_navigation_snapshot(reader)
        expected = set(snapshot.valid_moves)
        if snapshot.warp_exit_armed:
            expected |= set(snapshot.warp_exit_directions)

        for direction in DIRECTIONS:
            emulator.load_state(str(path))
            emulator.settle()
            emulator.press_and_settle(direction)
            after = _position(reader)
            # A wild encounter interrupts the step, so the move was legal even
            # though the coordinates are back where they started.
            moved = after != start or reader.read_battle()["in_battle"]
            if moved != (direction in expected):
                mismatches.append((path.name, start, direction, sorted(expected), after))
            if direction in snapshot.ledge_hops:
                ledge_landings.append((start, direction, snapshot.ledge_hops[direction], after[1]))

    assert len(seen) >= min(SWEEP_POSITIONS, 30), "not enough usable save states to sweep"
    assert mismatches == []
    for start, direction, predicted, landed in ledge_landings:
        assert predicted == landed, f"ledge {direction} at {start}: said {predicted}, got {landed}"


@needs_rom
def test_a_battle_frame_reads_the_tile_the_player_is_standing_on(emulator, reader):
    """The coordinates survive a wild battle, so a battle payload may report them.

    The action payload used to drop x, y and facing on any battle frame, on the
    grounds that a battle screen has no position, and `poke act` rendered
    `Mt Moon 1F (None,None) facing None`. wXCoord and wYCoord are untouched by
    the fight: this walks into an encounter, reads them off the battle frame,
    flees, and reads the same tile back off the overworld.

    Facing is deliberately not compared here — the same measurement run four
    times showed the facing byte on a battle frame is the direction from *before*
    the step the encounter interrupted, which is why the payload refuses to
    report it. See ``server.FACING_UNREAD_IN_BATTLE``.
    """
    emulator.load_state(str(_pick(ENCOUNTER_STATES)))
    emulator.settle()

    for step in range(120):
        if reader.read_battle()["in_battle"]:
            break
        emulator.press_and_settle(DIRECTIONS[step % 4])
    else:
        pytest.skip("no wild encounter in 120 steps — nothing to read a battle frame from")

    on_the_battle_frame = reader.read_coordinates()

    # RUN, and only ever press a direction with the top battle menu confirmed up.
    # A direction pressed anywhere else lands on the overworld and turns the
    # player, which is the comparison this test is trying to make.
    for _ in range(60):
        if not reader.read_battle()["in_battle"]:
            break
        if reader.at_battle_top_menu():
            for button in ("down", "right", "a"):
                emulator.press_and_settle(button)
        else:
            emulator.press_and_settle("b")
    if reader.read_battle()["in_battle"]:
        pytest.skip("could not flee — the comparison needs an overworld frame back")
    emulator.settle()

    assert reader.read_coordinates() == on_the_battle_frame
