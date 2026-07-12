"""Smoke test: the public surface imports cleanly and pathfinding is wired up."""

from pokemon_agent import __version__
from pokemon_agent.emulator import Emulator
from pokemon_agent.memory.reader import GameMemoryReader
from pokemon_agent.pathfinding import find_path, navigate


def test_version_is_exposed():
    assert __version__


def test_core_symbols_import():
    assert Emulator is not None
    assert GameMemoryReader is not None


def test_find_path_returns_steps_toward_the_target():
    # No collision map: the path is unobstructed, so 3 right + 2 down in some order.
    path = find_path((0, 0), (3, 2), None)
    assert path.count("right") == 3
    assert path.count("down") == 2
    assert len(path) == 5


def test_navigate_emits_walk_actions():
    assert navigate((0, 0), (3, 2)) == [
        "walk_down",
        "walk_down",
        "walk_right",
        "walk_right",
        "walk_right",
    ]
