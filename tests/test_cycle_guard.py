"""The lap the repeat guard structurally cannot see.

Route 11 (55,0): 218 consecutive calls, 8,225 presses, 22 minutes, every one
ending on the same tile. `RepeatGuard` never counted past 1 because the model
varied the plan each lap — `right:5 up:4`, `right:4 up:4`, `right:3 up:4` — so
by command it is six distinct plans and by outcome it is one, seventy-two times.
That block alone is 10.6% of everything the run spent after its last milestone.

These cases are the shape of that block, and the shapes it must not be confused
with: a nurse's counter, a conversation, and cheap looking-around.
"""

from __future__ import annotations

import pytest

from pokemon_agent.repeats import (
    CYCLE_PRESSES,
    CYCLE_WINDOW,
    CycleGuard,
    WalkingInCircles,
)


def _describe(tiles, presses):
    return f"{len(tiles)} tiles, {presses} presses"


def _lap(guard, *, map_name="Route 11", tiles=((55, 0),), presses=40, calls=CYCLE_WINDOW):
    for index in range(calls):
        guard.record(map_name, tiles[index % len(tiles)], presses)


def test_the_route_11_block_is_refused():
    """One tile, forty presses a call, twenty-four calls."""
    guard = CycleGuard()
    _lap(guard)
    with pytest.raises(WalkingInCircles) as caught:
        guard.check(_describe)
    assert caught.value.tiles == ((55, 0),)
    assert caught.value.presses == 24 * 40


def test_a_three_tile_lap_is_still_a_lap():
    guard = CycleGuard()
    _lap(guard, tiles=((55, 0), (54, 0), (55, 1)))
    with pytest.raises(WalkingInCircles):
        guard.check(_describe)


def test_it_fires_once_and_then_forgets():
    """The property that makes it safe to raise: the next call is never blocked.

    An agent that has just been refused has to be able to do something, and the
    only thing it can do on a route is walk.
    """
    guard = CycleGuard()
    _lap(guard)
    with pytest.raises(WalkingInCircles):
        guard.check(_describe)
    guard.check(_describe)  # would raise again if the window survived


def test_a_short_stretch_is_not_a_lap():
    guard = CycleGuard()
    _lap(guard, calls=CYCLE_WINDOW - 1)
    guard.check(_describe)


def test_looking_around_cheaply_is_not_a_lap():
    """One press a call on one tile is a model reading the frame, not circling."""
    guard = CycleGuard()
    _lap(guard, presses=1)
    assert guard.lap() is None
    guard.check(_describe)


def test_walking_somewhere_new_is_not_a_lap():
    guard = CycleGuard()
    for step in range(CYCLE_WINDOW):
        guard.record("Route 11", (55 - step, 0), 40)
    guard.check(_describe)


def test_changing_map_starts_over():
    """A different map is progress by itself, whatever the tiles looked like."""
    guard = CycleGuard()
    _lap(guard, calls=CYCLE_WINDOW - 1)
    guard.record("Route 12", (3, 3), 40)
    assert guard.lap() is None


def test_an_unreadable_position_starts_over():
    guard = CycleGuard()
    _lap(guard, calls=CYCLE_WINDOW - 1)
    guard.record("Route 11", None, 40)
    assert guard.lap() is None


def test_the_presses_floor_is_what_it_says():
    """Exactly at the floor it fires; one press under it does not."""
    guard = CycleGuard(window=4, tiles=1, presses=CYCLE_PRESSES)
    for _ in range(4):
        guard.record("Route 11", (55, 0), CYCLE_PRESSES // 4)
    assert guard.lap() == ([(55, 0)], CYCLE_PRESSES)

    lean = CycleGuard(window=4, tiles=1, presses=CYCLE_PRESSES)
    for index in range(4):
        lean.record("Route 11", (55, 0), CYCLE_PRESSES // 4 - (1 if index == 0 else 0))
    assert lean.lap() is None


def test_reset_forgets_everything():
    guard = CycleGuard()
    _lap(guard)
    guard.reset()
    assert guard.lap() is None
