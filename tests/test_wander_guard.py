"""A circuit round four rooms, which the lap detector resets itself out of.

`CycleGuard` forgets everything the moment the map changes, on the reasoning
that a different map is progress by itself. That is true of a journey and false
of a circuit, and the difference is not "changed map" but "changed map to
somewhere it has just been".

The run this was built from spent its first 23.8 hours walking Mt Moon 1F, B1F,
B2F and Route 4 in circles — 42,000 presses, no rung — and its last stall doing
the same across Pewter City, Route 2 and Viridian Forest. Not one call of either
was a lap on a single map, so nothing in the harness could see them.

Thresholds tuned against that run's 36,000 receipts rather than chosen. At
16 stays / 4 maps / 1600 presses it fires 16 times, names 49,534 presses, and
exactly one of the sixteen lands in a stretch that earned a rung soon after.
Loosening any one of them roughly triples the fires and puts a third of them on
productive ground; the tuning table is in the commit that added this.
"""

from __future__ import annotations

import pytest

from pokemon_agent.repeats import (
    WANDER_MAPS,
    WANDER_PRESSES,
    WANDER_WINDOW,
    WanderGuard,
    Wandering,
)

#: The four rooms of the opening stall, in the order it walked them.
MT_MOON = ("Mt Moon 1F", "Mt Moon B1F", "Mt Moon B2F", "Route 4")


def _describe(maps, presses):
    return f"{len(maps)} maps, {presses} presses"


def _walk(guard, maps=MT_MOON, stays=WANDER_WINDOW, presses=None):
    each = WANDER_PRESSES // WANDER_WINDOW + 1 if presses is None else presses
    for index in range(stays):
        guard.record(maps[index % len(maps)], each)


def test_the_mt_moon_circuit_is_refused():
    guard = WanderGuard()
    _walk(guard)
    with pytest.raises(Wandering) as caught:
        guard.check(_describe)
    assert set(caught.value.maps) == set(MT_MOON)
    assert caught.value.presses >= WANDER_PRESSES


def test_a_journey_across_many_maps_is_not_a_circuit():
    """Cerulean to Vermilion is seven rooms in a row and must stay silent."""
    guard = WanderGuard()
    journey = (
        "Cerulean City",
        "Route 5",
        "Route 5 Gate",
        "Underground Path Route 5",
        "Underground Path North South",
        "Route 6 Gate",
        "Route 6",
        "Vermilion City",
        "Vermilion Gym",
        "Route 11",
        "Diglett's Cave Route 11",
        "Diglett's Cave",
        "Diglett's Cave Route 2",
        "Route 2",
        "Viridian City",
        "Route 1",
    )
    for name in journey:
        guard.record(name, 200)
    assert guard.circuit() is None
    guard.check(_describe)


def test_a_cheap_circuit_is_not_refused():
    """Sixteen stays over four rooms that cost almost nothing is looking around."""
    guard = WanderGuard()
    _walk(guard, presses=1)
    assert guard.circuit() is None


def test_one_room_too_many_is_not_a_circuit():
    guard = WanderGuard()
    wider = MT_MOON + ("Route 3", "Pewter City")
    for index in range(WANDER_WINDOW):
        guard.record(wider[index % len(wider)], 200)
    assert guard.circuit() is None


def test_a_stay_is_one_run_not_one_call():
    """Forty calls on four maps is four stays, not forty."""
    guard = WanderGuard()
    for name in MT_MOON:
        for _ in range(10):
            guard.record(name, 50)
    assert len(guard._stays) == len(MT_MOON)
    assert guard.circuit() is None, "four stays is not a window"


def test_it_fires_once_and_then_forgets():
    """The property that makes it safe to raise rather than annotate."""
    guard = WanderGuard()
    _walk(guard)
    with pytest.raises(Wandering):
        guard.check(_describe)
    guard.check(_describe)


def test_reset_forgets_everything():
    """A milestone calls this: a rung is evidence the circuit was going somewhere."""
    guard = WanderGuard()
    _walk(guard)
    guard.reset()
    assert guard.circuit() is None


def test_a_blank_map_is_not_a_stay():
    """Load receipts carry no map, and a hole is not a room."""
    guard = WanderGuard()
    guard.record("", 500)
    assert guard._stays == []


def test_the_window_slides_rather_than_growing():
    guard = WanderGuard()
    _walk(guard, stays=WANDER_WINDOW * 3)
    assert len(guard._stays) == WANDER_WINDOW


@pytest.mark.parametrize("maps", [2, 3, WANDER_MAPS])
def test_any_circuit_up_to_the_limit_counts(maps):
    guard = WanderGuard()
    _walk(guard, maps=MT_MOON[:maps])
    assert guard.circuit() is not None


def test_never_leaving_one_map_is_not_this_guards_case():
    """Sixteen calls on one map is one stay, so the window cannot fill.

    That is the division of labour and not a gap: standing still on a single
    map is a lap, and `CycleGuard` watches tiles, which is the only thing that
    can tell a lap from a long fight in tall grass. Two guards, two shapes.
    """
    guard = WanderGuard()
    _walk(guard, maps=("Mt Moon 1F",), stays=WANDER_WINDOW * 4)
    assert len(guard._stays) == 1
    assert guard.circuit() is None
