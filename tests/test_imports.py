"""Smoke test: the public surface imports cleanly."""

from pokemon_agent import __version__
from pokemon_agent.emulator import Emulator
from pokemon_agent.memory.reader import GameMemoryReader


def test_version_is_exposed():
    assert __version__


def test_core_symbols_import():
    assert Emulator is not None
    assert GameMemoryReader is not None
