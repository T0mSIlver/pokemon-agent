"""Two things a run lost in a maze could not get from the harness.

Mt Moon cost this run 42,000 presses at the start and 7,136 more today, 18% of
everything it spent after the field-move work. Both fixes here come from
watching that, and both hand over a fact the harness already held.

**Which door starts the route.** The frontier menu says where a rung is earned
and how far — `Celadon Gym, 6 hops` — and the wander refusal said "you are
circling" and pointed at `poke progress`. Neither says which of the three
staircases in front of you begins that route, which is the only question a
model lost on Mt Moon 1F actually has.

**Which floor a tile is on.** `poke goto x y` walks the map you are standing on.
Mt Moon's three floors share a coordinate space, so one run asked for the same
tile sixteen times from the wrong floor; every answer was honest and about a
different room than the question. Ten of that stretch's refusals were the repeat
guard catching it.
"""

from __future__ import annotations

import pytest

from pokemon_agent import server


class TestFloorSiblings:
    def test_a_dungeon_floor_knows_its_siblings(self):
        assert server._floor_siblings("Mt Moon 1F") == ["Mt Moon B1F", "Mt Moon B2F"]
        assert server._floor_siblings("Rock Tunnel 1F") == ["Rock Tunnel B1F"]

    def test_a_map_with_no_floors_has_none(self):
        for name in ("Cerulean City", "Route 4", "Viridian Forest", ""):
            assert server._floor_siblings(name) == [], name

    def test_a_lone_floor_is_not_a_sibling_of_itself(self):
        """A building with one floor needs no warning; the note must stay silent."""
        assert server._floor_note("Cerulean City") == ""

    @pytest.mark.parametrize(
        "suffix,is_floor",
        [
            ("1F", True),
            ("B1F", True),
            ("11F", True),
            ("B4F", True),
            ("Gate", False),
            ("F", False),
            ("City", False),
            ("2", False),
            ("BF", False),
        ],
    )
    def test_what_counts_as_a_floor_suffix(self, suffix, is_floor):
        assert server._looks_like_floor(suffix) is is_floor


class TestFloorNote:
    def test_it_names_the_floor_and_the_verb_that_crosses_them(self):
        note = server._floor_note("Mt Moon 1F")
        assert "this is Mt Moon 1F" in note
        assert "Mt Moon B1F" in note and "Mt Moon B2F" in note
        assert "poke goto x y" in note

    def test_a_tall_building_does_not_list_every_floor(self):
        """Silph Co has ten siblings; naming them all is noise on every walk."""
        note = server._floor_note("Silph Co 5F")
        assert "and 7 more" in note
        assert note.count("Silph Co") <= 5


class TestOnwardRoutes:
    def test_it_names_more_than_one_rung(self):
        """Three routes is information. One would be the harness picking the goal,
        which is the thing this project deliberately does not do."""
        assert server.ONWARD_ROUTES_SHOWN >= 2

    @pytest.mark.asyncio
    async def test_it_answers_empty_rather_than_raising_when_the_game_is_unreadable(
        self, monkeypatch
    ):
        """A refusal must never fail on its own advice."""

        async def boom(*args, **kwargs):
            raise RuntimeError("no emulator")

        monkeypatch.setattr(server, "_run_emulator_sync", boom)
        assert await server._onward_sentence() == ""
