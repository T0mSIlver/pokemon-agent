"""Collision downsampling: 18x20 (PyBoy's upsampled view) -> 9x10 walkable blocks.

`terrain[y][x]` truthy means passable (see Emulator._compute_valid_moves).
"""

import pytest

from pokemon_agent.emulator import PyBoyEmulator


def downsample(matrix):
    # Pure function of its argument -- no emulator state, so skip __init__ (which
    # would demand a ROM).
    emu = PyBoyEmulator.__new__(PyBoyEmulator)
    return emu._downsample_collision(matrix)


def uniform_18x20(value):
    return [[value] * 20 for _ in range(18)]


def test_all_walkable_stays_walkable():
    assert downsample(uniform_18x20(1)) == [[1] * 10 for _ in range(9)]


def test_all_blocked_stays_blocked():
    assert downsample(uniform_18x20(0)) == [[0] * 10 for _ in range(9)]


def test_uniform_blocks_round_trip_exactly():
    # PyBoy replicates each walkable value across a 2x2 block. Downsampling must
    # recover the original 9x10 matrix unchanged.
    native = [[(x + y) % 2 for x in range(10)] for y in range(9)]
    upsampled = [[native[y // 2][x // 2] for x in range(20)] for y in range(18)]
    assert downsample(upsampled) == native


def test_partially_blocked_block_is_not_walkable():
    # The player occupies a whole 16x16 block. If any sub-tile is blocked the block
    # is blocked -- collapsing with any() would fail open and walk the agent into a wall.
    matrix = uniform_18x20(1)
    matrix[0][0] = 0  # one sub-tile of the top-left block
    result = downsample(matrix)
    assert result[0][0] == 0
    assert result[0][1] == 1  # neighbouring blocks untouched


def test_already_downsampled_input_is_normalised_not_rejected():
    assert downsample([[5] * 10 for _ in range(9)]) == [[1] * 10 for _ in range(9)]


def test_unexpected_shape_raises():
    with pytest.raises(ValueError, match="Unexpected collision map shape"):
        downsample([[0] * 5 for _ in range(5)])
