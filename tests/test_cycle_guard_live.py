"""The lap detector proved through the HTTP endpoint it is wired into.

`tests/test_cycle_guard.py` exercises :class:`~pokemon_agent.repeats.CycleGuard`
on its own and says nothing about whether anything ever calls it. This file
drives ``POST /action`` and ``POST /goto`` and asserts the 400 comes back, so a
`_check_circling` that stopped being awaited, or a `_record_walk` that stopped
being fed, fails here rather than in the next 8,000-press block.

Everything below was first measured against a real server on a real ROM before
it was written down: a probe on port 8792, Route 11 of a real save, twenty-four
`POST /action` calls of twelve button presses each. The refusal that came back,
verbatim and at status 400::

    24 walks in a row have ended on the same 1 tile — [3, 6] — and cost 277
    presses getting there. Walking is not what is missing. What is reachable
    from here: Diglett's Cave Route 11 at [4, 5], 2 steps; Vermilion City off
    the west edge at [0, 6], 4 steps; Route 11 Gate 1F at [49, 8], 48 steps;
    Route 11 Gate 1F at [49, 9], 49 steps. If every door is one you have
    already been through, the way on is a tool and not a direction: `poke cut`
    opens a small tree, `poke guide` says what this part of the game wants,
    `poke progress` lists what is open.

Two things that probe found and that the assertions below pin:

* The window counts **every** ``/action`` batch, not only walking ones.
  `_record_walk` says "Only walking calls" and the endpoint feeds it
  unconditionally, so twenty-four batches of `press_b` on one tile earn a
  refusal that opens "24 walks in a row". See
  `test_a_batch_with_no_walk_in_it_still_fills_the_window`.
* The refusal's receipt is written through the same recorder as every other
  batch, so it only lands on disk while a run is open. On a bare server — no
  supervisor session — `runs/` is never created and the refusal leaves no
  trace at all. See `test_the_refusal_writes_a_receipt`.
"""

from __future__ import annotations

import json

import pytest
from test_server import (
    FakeEmulator,
    running_server,  # noqa: F401 — same-directory test helper
)

from pokemon_agent import server
from pokemon_agent.repeats import CYCLE_PRESSES, CYCLE_TILES, CYCLE_WINDOW

L, R, U, D = "walk_left", "walk_right", "walk_up", "walk_down"

#: Six plans, one outcome — the shape of the Route 11 block. Every plan ends on
#: the tile it started from and no two consecutive plans are the same command,
#: so `RepeatGuard` (which keys on the command) resets every call and only the
#: lap detector can see anything wrong.
LAP_PLANS = [
    [L, R] * 6,
    [R, L] * 6,
    [L, L, R, R] + [L, R] * 4,
    [R, R, L, L] + [R, L] * 4,
    [D, U] + [L, R] * 5,
    [L, R, D, U] + [L, R] * 4,
]

#: Twelve a call over twenty-four calls is 288, comfortably over the floor.
LAP_PRESSES = len(LAP_PLANS[0])


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A server over an open fake overworld, so a lap is a lap and not a wall."""
    with running_server(tmp_path, monkeypatch, FakeEmulator()) as running:
        yield running


def _post(app, actions):
    return app.http.post("/action", json={"actions": actions})


def _walk_a_lap(app, calls: int = CYCLE_WINDOW) -> list[tuple[int, int]]:
    """Fill the window with `calls` batches that all end where they started."""
    tiles = []
    for index in range(calls):
        response = _post(app, LAP_PLANS[index % len(LAP_PLANS)])
        assert response.status_code == 200, response.json()
        body = response.json()
        tiles.append((body["x"], body["y"]))
    return tiles


def test_the_lap_is_refused_through_the_action_endpoint(app):
    """The 25th call is a 400 that names the tile and the arithmetic."""
    tiles = _walk_a_lap(app)
    assert len(set(tiles)) == 1, tiles

    response = _post(app, LAP_PLANS[0])

    assert response.status_code == 400
    detail = response.json()["detail"]
    x, y = tiles[0]
    assert detail.startswith(
        f"{CYCLE_WINDOW} walks in a row have ended on the same 1 tile — [{x}, {y}] — "
        f"and cost {CYCLE_WINDOW * LAP_PRESSES} presses getting there. "
        "Walking is not what is missing."
    )
    assert detail.endswith(
        "If every door is one you have already been through, the way on is a tool and "
        "not a direction: `poke cut` opens a small tree, `poke guide` says what this "
        "part of the game wants, `poke progress` lists what is open."
    )


def test_the_window_really_is_fed_by_the_endpoint(app):
    """`_record_walk` is what makes the refusal possible; prove it ran.

    Without this the test above could pass on a guard that was primed some
    other way. After twenty-four accepted batches the window is already a lap,
    before anything checks it.
    """
    assert server._cycle_guard.lap() is None
    _walk_a_lap(app)
    found = server._cycle_guard.lap()
    assert found is not None
    tiles, presses = found
    assert len(tiles) == 1
    assert presses == CYCLE_WINDOW * LAP_PRESSES


def test_the_call_after_the_refusal_is_accepted(app):
    """Property (a): it fires once per window, so it can never wedge the agent.

    The agent standing on that tile has nothing to offer but another walk. If
    the refusal survived its own window there would be no legal move left.
    """
    _walk_a_lap(app)
    assert _post(app, LAP_PLANS[0]).status_code == 400

    assert _post(app, LAP_PLANS[1]).status_code == 200
    assert _post(app, LAP_PLANS[2]).status_code == 200


def test_the_refusal_costs_no_presses(app):
    """It is raised before the emulator is touched, so the lap stops there."""
    _walk_a_lap(app)
    spent = app.emulator.frame_count

    assert _post(app, LAP_PLANS[0]).status_code == 400

    assert app.emulator.frame_count == spent


def test_looking_around_cheaply_is_not_refused(app):
    """Property (b): many calls, few presses. Reading a frame is not a lap.

    Thirty single-step batches over two tiles — a full window and a quarter —
    at one press a call. The tiles say lap and the presses say nothing was
    spent, and the presses are what decide it.
    """
    for index in range(CYCLE_WINDOW + 6):
        response = _post(app, [L] if index % 2 == 0 else [R])
        assert response.status_code == 200, response.json()
    assert server._cycle_guard.lap() is None


def test_walking_somewhere_new_is_not_refused(app):
    """Property (b): the presses are spent, but the ground keeps changing.

    Eleven presses a call for twenty-six calls, over a corridor seven tiles
    long: well past the press floor and never a lap, because it never comes
    back to the same handful of tiles.
    """
    drift = R
    tiles = set()
    for _ in range(CYCLE_WINDOW + 2):
        response = _post(app, [R, L] * 5 + [drift])
        assert response.status_code == 200, response.json()
        body = response.json()
        tiles.add((body["x"], body["y"]))
        drift = L if body["x"] >= 11 else (R if body["x"] <= 5 else drift)
    assert len(tiles) > CYCLE_TILES
    assert server._cycle_guard.lap() is None


def test_a_batch_with_no_walk_in_it_does_not_fill_the_window(app):
    """It used to, and the refusal it produced was nonsense.

    Twenty-four batches of `press_b` — no walking action anywhere in them —
    earned a refusal beginning "24 walks in a row have ended on the same 1
    tile", followed by a list of doors to walk through. The endpoint fed
    `_record_walk` every batch it ran, so the window was really "the last two
    dozen /action calls" while its docstring claimed otherwise.

    A-spam that changes nothing is the repeat guard's case, and it says
    something true about a text box. This guard is about laps.
    """
    for index in range(CYCLE_WINDOW * 2):
        plan = ["press_b"] * 12 if index % 2 == 0 else ["press_b"] * 11 + ["press_start"]
        assert _post(app, plan).status_code == 200

    assert _post(app, ["press_b"]).status_code == 200


def test_a_batch_with_no_walk_clears_a_window_it_interrupts(app):
    """Half a lap, a conversation, then the other half is not a lap.

    The plans have to vary, or `RepeatGuard` refuses the identical command at 16
    before this guard has seen 24 of anything — which is itself worth knowing:
    the two guards cover different halves of the same failure, and only the
    varied-plan half needs this one.
    """
    _walk_a_lap(app, calls=CYCLE_WINDOW - 1)

    assert _post(app, ["press_a"]).status_code == 200

    # The window is empty again, so the calls that used to complete it cannot.
    _walk_a_lap(app, calls=CYCLE_WINDOW - 1)


def test_a_goto_is_refused_by_the_same_window(app):
    """Both walking endpoints check it, and one window covers the pair."""
    _walk_a_lap(app)

    response = app.http.post("/goto", json={"x": 5, "y": 5})

    assert response.status_code == 400
    assert response.json()["detail"].startswith(f"{CYCLE_WINDOW} walks in a row")


def _open_a_run(app) -> None:
    """Open a run on the server's own recorder, the way a session would.

    Without this the refusal writes nothing at all: `_write_receipt` returns
    early while `run_id` is None, and only `RunRecorder.begin_session` — which
    only a supervisor session calls — ever sets it. A bare server therefore
    never creates `runs/`, and the live probe's data dir had no `runs/` in it
    after a lap and a refusal.
    """
    recorder = server._run_recorder
    recorder.run_id = recorder.registry.start_run(
        harness_sha="",
        config_hash="",
        model="",
        start_checkpoint=None,
        goal="prove the lap detector writes a receipt",
    )


def test_the_refusal_writes_a_receipt(app):
    """What the receipt carries, read back off disk."""
    _open_a_run(app)
    tiles = _walk_a_lap(app)

    response = _post(app, LAP_PLANS[0])
    assert response.status_code == 400

    receipts_path = server._run_recorder.registry.receipts_path(server._run_recorder.run_id)
    lines = [json.loads(line) for line in receipts_path.read_text().splitlines() if line.strip()]
    assert receipts_path.parent.parent == app.data_dir / "runs"

    refusal = lines[-1]
    assert refusal["tool"] == "action"
    # Nothing was pressed and there is no bundle to read a position out of, so
    # the fixed half of the schema is blank and everything that says what
    # happened rides in the `extra` keys, flattened onto the same line.
    assert refusal["presses"] == 0
    assert refusal["exit"] == 1
    assert refusal["map"] == ""
    assert refusal["pos"] is None
    assert refusal["moved"] is None
    assert refusal["circling"] == [list(tiles[0])]
    assert refusal["circling_presses"] == CYCLE_WINDOW * LAP_PRESSES
    assert refusal["error"] == response.json()["detail"][:200]
    # It is the refused call's own receipt, not the last accepted batch's.
    assert lines[-2]["presses"] > 0


def test_the_press_floor_is_what_stops_a_cheap_window(app):
    """The one number that separates the two halves of property (b)."""
    assert CYCLE_WINDOW * LAP_PRESSES >= CYCLE_PRESSES
    assert CYCLE_WINDOW * 1 < CYCLE_PRESSES
