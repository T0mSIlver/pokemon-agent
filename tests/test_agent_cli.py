"""The agent-facing CLI, driven against a stub server rather than a live game.

Every failure mode here was one the agent used to hit through hand-written curl:
a batch silently lost, a refusal that read as a crash, a server that was not
there. The point of the CLI is that each of those now has one obvious output.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pokemon_agent import agent_cli

BATTLE_GUARD_DETAIL = (
    "a_until_dialog_end is unsafe in battle: the battle menu counts as an open "
    "dialog, so this confirms menu entries and opens the bag. Press A once to "
    "advance battle text, or attack by name with POST /battle/fight — the move "
    "cursor remembers where it was left last turn and wraps, so two A presses "
    "are not 'use my first move'."
)

RATE_LIMIT_DETAIL = (
    "More than 60 action batches in 60s. You are almost certainly in a loop that "
    "never checks whether the player moved. A blocked move returns the same "
    "position, so a 'walk until position changes' loop never ends. Stop, read a "
    "frame, and pick a different direction."
)

STATE_PAYLOAD = {
    "player": {
        "name": "RED",
        "money": 1527,
        "badges": [],
        "position": {"x": 13, "y": 13},
        "facing": "left",
    },
    "party": [
        {
            "species": "Charmander",
            "level": 13,
            "hp": 19,
            "max_hp": 36,
            "status": "OK",
            "types": ["Fire"],
        }
    ],
    "bag": [{"item": "Town Map", "quantity": 1}, {"item": "Poke Ball", "quantity": 2}],
    "battle": {"in_battle": False},
    "dialog": {"active": False},
    "map": {"map_id": 2, "map_name": "Pewter City"},
}

MAP_PAYLOAD = {
    "map_id": 2,
    "map_name": "Pewter City",
    "width": 40,
    "height": 36,
    "coverage": {"seen": 1153, "walked": 200, "total": 1440, "percent": 80.1},
    "warps": [{"x": 2, "y": 7}, {"x": 29, "y": 13}],
    "player": {"x": 13, "y": 13},
    "image_path": "/tmp/latest_map.png",
}


class StubServer:
    """A few canned routes and a record of what the CLI actually sent."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, object]] = {}
        self.requests: list[dict] = []
        self._http: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def route(self, method: str, path: str, payload: object, status: int = 200) -> None:
        self.routes[(method, path)] = (status, payload)

    @property
    def port(self) -> int:
        assert self._http is not None
        return self._http.server_port

    def start(self) -> None:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args, **kwargs):  # keep pytest output clean
                pass

            def _respond(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                stub.requests.append(
                    {
                        "method": method,
                        "path": self.path,
                        "body": json.loads(raw) if raw else None,
                    }
                )
                status, payload = stub.routes.get(
                    (method, self.path),
                    (404, {"detail": f"no stub route for {method} {self.path}"}),
                )
                if isinstance(payload, bytes):
                    body, content_type = payload, "application/octet-stream"
                else:
                    body = json.dumps(payload).encode("utf-8")
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's spelling
                self._respond("GET")

            def do_POST(self):  # noqa: N802
                self._respond("POST")

        self._http = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


@pytest.fixture
def stub():
    server = StubServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


def run(stub: StubServer, *args: str) -> int:
    return agent_cli.main([*args, "--port", str(stub.port)])


def closed_port() -> int:
    """A port nothing is listening on."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_aliases_expand_to_server_action_names():
    assert agent_cli.expand_actions(["up", "down", "left", "right"]) == [
        "walk_up",
        "walk_down",
        "walk_left",
        "walk_right",
    ]
    assert agent_cli.expand_actions(["a", "b", "start", "select"]) == [
        "press_a",
        "press_b",
        "press_start",
        "press_select",
    ]
    assert agent_cli.expand_actions(["wait", "adialog"]) == ["wait_60", "a_until_dialog_end"]


def test_long_names_and_frame_counts_pass_through():
    assert agent_cli.expand_actions(["walk_up", "press_a"]) == ["walk_up", "press_a"]
    assert agent_cli.expand_actions(["wait_120", "hold_b_45"]) == ["wait_120", "hold_b_45"]


def test_repeat_form_expands():
    assert agent_cli.expand_actions(["up:4", "a"]) == [
        "walk_up",
        "walk_up",
        "walk_up",
        "walk_up",
        "press_a",
    ]


def test_repeat_form_rejects_nonsense():
    with pytest.raises(agent_cli.ActionError):
        agent_cli.expand_actions(["up:0"])
    with pytest.raises(agent_cli.ActionError):
        agent_cli.expand_actions(["up:lots"])
    with pytest.raises(agent_cli.ActionError):
        agent_cli.expand_actions([f"up:{agent_cli.MAX_REPEAT + 1}"])


def test_act_sends_expanded_batch_and_prints_response(stub, capsys):
    stub.route("POST", "/action", {"actions_executed": 3, "x": 5, "y": 6, "moved": 2})

    assert run(stub, "act", "up:2", "a") == 0

    assert stub.requests[-1]["body"] == {"actions": ["walk_up", "walk_up", "press_a"]}
    # Prose, not the object: see action_lines. `--json` still prints the payload.
    out = capsys.readouterr().out
    assert "(5,6)" in out
    assert "moved 2" in out


def test_unknown_action_is_refused_before_any_request(stub, capsys):
    stub.route("POST", "/action", {"actions_executed": 1})

    assert run(stub, "act", "jump", "up") != 0

    assert stub.requests == [], "a typo must not reach the game"
    stderr = capsys.readouterr().err
    assert "jump" in stderr
    # The message has to say what *is* valid, or the agent guesses again.
    assert "walk_up" in stderr
    assert "up:4" in stderr


# ---------------------------------------------------------------------------
# Errors the server writes for the agent to read
# ---------------------------------------------------------------------------


def test_battle_guard_detail_is_printed_verbatim(stub, capsys):
    stub.route("POST", "/action", {"detail": BATTLE_GUARD_DETAIL}, status=400)

    assert run(stub, "act", "adialog") == agent_cli.EXIT_HTTP_ERROR

    stderr = capsys.readouterr().err
    assert BATTLE_GUARD_DETAIL in stderr
    assert "Traceback" not in stderr


def test_rate_limit_detail_is_printed_verbatim(stub, capsys):
    stub.route("POST", "/action", {"detail": RATE_LIMIT_DETAIL}, status=429)

    assert run(stub, "act", "up") == agent_cli.EXIT_HTTP_ERROR

    stderr = capsys.readouterr().err
    assert RATE_LIMIT_DETAIL in stderr
    assert "429" in stderr
    assert "Traceback" not in stderr


def test_connection_refused_says_which_port(capsys):
    port = closed_port()

    assert agent_cli.main(["state", "--port", str(port)]) == agent_cli.EXIT_NO_SERVER

    stderr = capsys.readouterr().err
    assert "not answering" in stderr
    assert str(port) in stderr
    assert "Traceback" not in stderr


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_state_summary_is_short_and_scannable(stub, capsys):
    stub.route("GET", "/state", STATE_PAYLOAD)

    assert run(stub, "state") == 0

    out = capsys.readouterr().out
    assert "Pewter City (13,13) facing left" in out
    assert "Charmander L13 19/36 Fire" in out
    assert "badges: none" in out
    assert "1527" in out
    assert "Town Map x1" in out
    assert len(out.strip().splitlines()) <= 8


def test_state_json_is_the_raw_payload(stub, capsys):
    stub.route("GET", "/state", STATE_PAYLOAD)

    assert run(stub, "state", "--json") == 0

    assert json.loads(capsys.readouterr().out) == STATE_PAYLOAD


def test_state_reports_a_battle(stub, capsys):
    payload = dict(STATE_PAYLOAD)
    payload["battle"] = {
        "in_battle": True,
        "enemy": {"species": "Weedle", "level": 3, "hp": 15, "max_hp": 15},
    }
    stub.route("GET", "/state", payload)

    assert run(stub, "state") == 0
    assert "battle: Weedle L3 15/15" in capsys.readouterr().out


def test_state_refuses_the_facing_it_reads_during_a_battle(stub, capsys):
    """The tile is still true in a battle. The direction is not.

    An encounter interrupts the step that started it, so the facing byte holds
    the direction from before that step: two of four battle frames measured in
    Mt. Moon read a direction the overworld disagreed with the moment the fight
    ended. `poke act` refuses it; this is the same frame read a different way.
    """
    payload = dict(STATE_PAYLOAD)
    payload["battle"] = {"in_battle": True, "enemy": {"species": "Weedle"}}
    stub.route("GET", "/state", payload)

    assert run(stub, "state") == 0
    out = capsys.readouterr().out
    assert "Pewter City (13,13) facing unread in a battle" in out
    assert "facing left" not in out


def test_state_names_the_moves_that_have_run_dry(stub, capsys):
    """`poke fight` was refused 12 times for a move with no PP left.

    The payload has carried PP per move all along and nothing printed it, and
    `poke calc` still ranks a 0 PP move as the best one available because its
    own payload does not carry PP at all.
    """
    payload = json.loads(json.dumps(STATE_PAYLOAD))
    payload["party"][0]["moves"] = [
        {"name": "Scratch", "pp": 12},
        {"name": "Ember", "pp": 0},
        {"name": "Rage", "pp": 0},
    ]
    stub.route("GET", "/state", payload)

    assert run(stub, "state") == 0
    assert "no PP: Ember, Rage" in capsys.readouterr().out


def test_state_stays_quiet_while_every_move_still_has_pp(stub, capsys):
    """It costs a line only on the turn it would otherwise cost a wasted call."""
    payload = json.loads(json.dumps(STATE_PAYLOAD))
    payload["party"][0]["moves"] = [{"name": "Scratch", "pp": 12}, {"name": "Ember", "pp": 5}]
    stub.route("GET", "/state", payload)

    assert run(stub, "state") == 0
    assert "no PP" not in capsys.readouterr().out


def test_map_summary_includes_the_png_path(stub, capsys):
    stub.route("GET", "/map", MAP_PAYLOAD)

    assert run(stub, "map") == 0

    out = capsys.readouterr().out
    assert "Pewter City (map 2) 40x36" in out
    assert "seen 1153/1440 (80.1%)" in out
    assert "(2,7)" in out
    assert "png: /tmp/latest_map.png" in out


def test_map_json(stub, capsys):
    stub.route("GET", "/map", MAP_PAYLOAD)

    assert run(stub, "map", "--json") == 0
    assert json.loads(capsys.readouterr().out) == MAP_PAYLOAD


def test_map_not_found_detail_is_surfaced(stub, capsys):
    stub.route("GET", "/map", {"detail": "Map 99 has never been visited."}, status=404)

    assert run(stub, "map") == agent_cli.EXIT_HTTP_ERROR
    assert "Map 99 has never been visited." in capsys.readouterr().err


def test_frame_reports_workspace_paths(stub, tmp_path, monkeypatch, capsys):
    (tmp_path / "latest_frame.png").write_bytes(b"png")
    (tmp_path / "latest_frame_annotated.png").write_bytes(b"png")
    monkeypatch.setenv("POKE_WORKSPACE", str(tmp_path))

    assert run(stub, "frame") == 0

    out = capsys.readouterr().out
    assert str(tmp_path / "latest_frame.png") in out
    assert str(tmp_path / "latest_frame_annotated.png") in out
    assert stub.requests == [], "listing paths must not cost a request"


def test_frame_refresh_writes_a_screenshot(stub, tmp_path, capsys):
    stub.route("GET", "/screenshot", b"\x89PNG-not-really")
    target = tmp_path / "fresh.png"

    assert run(stub, "frame", "--refresh", str(target)) == 0

    assert target.read_bytes() == b"\x89PNG-not-really"
    assert str(target) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Saves
# ---------------------------------------------------------------------------


def test_save_and_load(stub, capsys):
    stub.route("POST", "/save", {"success": True, "save": {"name": "brock", "path": "/s/brock"}})
    stub.route("POST", "/load", {"success": True, "save": {"name": "brock", "path": "/s/brock"}})

    assert run(stub, "save", "brock") == 0
    assert run(stub, "load", "brock") == 0

    assert [request["body"] for request in stub.requests] == [{"name": "brock"}, {"name": "brock"}]
    out = capsys.readouterr().out
    assert "saved brock -> /s/brock" in out
    assert "loaded brock -> /s/brock" in out


def test_load_missing_save_surfaces_the_detail(stub, capsys):
    stub.route("POST", "/load", {"detail": "Save not found: nope"}, status=404)

    assert run(stub, "load", "nope") == agent_cli.EXIT_HTTP_ERROR
    assert "Save not found: nope" in capsys.readouterr().err


def test_saves_asks_for_the_named_ones(stub, capsys):
    """The newest forty of everything is forty autosaves and no answer.

    The harness writes an `auto__` checkpoint on every battle and every map
    change: 300 of the run's 465 saves. Newest-first with the server's default
    limit of 40, every row was one of those and not one of the 165 names a
    caller could act on appeared.
    """
    stub.route(
        "GET",
        "/saves?named=true",
        {
            "saves": [{"name": "brock", "modified": 0}, {"name": "forest", "modified": 0}],
            "count": 2,
        },
    )

    assert run(stub, "saves") == 0

    out = capsys.readouterr().out
    assert "brock" in out
    assert "forest" in out
    assert stub.requests[-1]["path"] == "/saves?named=true"


def test_saves_says_how_many_more_names_it_did_not_show(stub, capsys):
    stub.route(
        "GET",
        "/saves?named=true",
        {"saves": [{"name": "brock", "modified": 0}], "count": 165},
    )

    assert run(stub, "saves") == 0
    assert "and 164 more" in capsys.readouterr().out


def test_saves_says_so_when_empty(stub, capsys):
    stub.route("GET", "/saves?named=true", {"saves": []})

    assert run(stub, "saves") == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("no named saves")
    # The checkpoints are still loadable by name; say so rather than imply they
    # are gone.
    assert "auto__" in out


def test_health(stub, capsys):
    stub.route(
        "GET",
        "/health",
        {"status": "ok", "emulator_ready": True, "agent_workspace_ready": True},
    )

    assert run(stub, "health") == 0
    assert "ok emulator=ready workspace=ready" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Where the server is
# ---------------------------------------------------------------------------


def test_port_environment_variable_is_honoured(stub, monkeypatch, capsys):
    stub.route("GET", "/health", {"status": "ok", "emulator_ready": True})
    monkeypatch.setenv("PORT", str(stub.port))

    assert agent_cli.main(["health"]) == 0
    assert "ok" in capsys.readouterr().out


def test_port_before_the_subcommand_still_wins(stub, capsys):
    stub.route("GET", "/state", STATE_PAYLOAD)

    assert agent_cli.main(["--port", str(stub.port), "state"]) == 0
    assert "Pewter City" in capsys.readouterr().out


def test_url_overrides_port(stub, capsys):
    stub.route("GET", "/health", {"status": "ok", "emulator_ready": True})

    assert agent_cli.main(["health", "--url", f"http://127.0.0.1:{stub.port}/"]) == 0
    assert "ok" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Battle commands
# ---------------------------------------------------------------------------


def test_fight_sends_the_move_name_and_prints_what_happened(stub, capsys):
    stub.route(
        "POST",
        "/battle/fight",
        {"used": "Ember", "battle": True, "hp": "29/32", "menu": "top", "highlighted": "FIGHT"},
    )

    assert run(stub, "fight", "ember") == 0

    assert stub.requests[-1]["path"] == "/battle/fight"
    assert stub.requests[-1]["body"] == {"move": "ember"}
    assert json.loads(capsys.readouterr().out)["used"] == "Ember"


def test_fight_joins_a_multi_word_move(stub):
    stub.route("POST", "/battle/fight", {"used": "Hyper Beam"})

    assert run(stub, "fight", "hyper", "beam") == 0

    assert stub.requests[-1]["body"] == {"move": "hyper beam"}


def test_fight_prints_the_servers_refusal_verbatim(stub, capsys):
    detail = "No move called 'surf'. Known moves: Scratch, Growl, Ember."
    stub.route("POST", "/battle/fight", {"detail": detail}, status=400)

    assert run(stub, "fight", "surf") == agent_cli.EXIT_HTTP_ERROR

    assert detail in capsys.readouterr().err


def test_run_posts_with_no_body_and_prints_the_outcome(stub, capsys):
    stub.route("POST", "/battle/run", {"fled": True, "battle": False, "map": "Route 1"})

    assert run(stub, "run") == 0

    assert stub.requests[-1]["path"] == "/battle/run"
    assert stub.requests[-1]["body"] is None
    assert json.loads(capsys.readouterr().out)["fled"] is True


def test_run_outside_a_battle_reports_the_refusal(stub, capsys):
    stub.route(
        "POST",
        "/battle/run",
        {"detail": "Not in a battle. Nothing to attack and nothing to run from."},
        status=400,
    )

    assert run(stub, "run") == agent_cli.EXIT_HTTP_ERROR

    assert "Not in a battle" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Batch limits
#
# The server holds its single emulator lock for a whole batch, so an uncapped
# frame count takes the game away from everything else. `wait_1000000000` was
# reachable and would have run for hundreds of days.
# ---------------------------------------------------------------------------


def test_a_single_action_cannot_ask_for_an_absurd_frame_count():
    with pytest.raises(agent_cli.ActionError, match="the limit is"):
        agent_cli.expand_actions(["wait_1000000000"])
    with pytest.raises(agent_cli.ActionError, match="the limit is"):
        agent_cli.expand_actions(["hold_up_1000000000"])


def test_frame_counts_inside_the_limit_still_work():
    assert agent_cli.expand_actions(["wait_600"]) == ["wait_600"]
    assert agent_cli.frames_for("wait_600") == 600
    assert agent_cli.frames_for("walk_up") == agent_cli.FRAMES_PER_INPUT


def test_a_batch_longer_than_the_cap_is_refused():
    too_many = ["up"] * (agent_cli.MAX_ACTIONS_PER_BATCH + 1)
    with pytest.raises(agent_cli.ActionError, match="the limit is"):
        agent_cli.expand_actions(too_many)


def test_the_refusal_names_the_edit_rather_than_saying_send_fewer():
    """Send fewer named no number, and one session read it as noise.

    It was growing a route through Mt. Moon segment by segment and re-simulating
    the whole thing; every attempt past forty came back refused and every next
    attempt was longer. 505 of that session's 549 calls were that refusal, 41 to
    159 actions, and not one of them shortened the plan.
    """
    with pytest.raises(agent_cli.ActionError) as caught:
        agent_cli.expand_actions(["up"] * 57)

    message = str(caught.value)
    assert "Drop the last 17" in message
    assert f"send the first {agent_cli.MAX_ACTIONS_PER_BATCH}" in message


def test_a_plan_that_is_only_simulated_is_not_held_to_the_batch_caps():
    """`batch=False` keeps the vocabulary and drops the execution caps.

    The caps exist because the server holds its one emulator lock for the whole
    batch it is executing. Nothing is executed on paper, so nothing is held.
    """
    plan = agent_cli.expand_actions(["up:40", "left:40", "down:40"], batch=False)

    assert len(plan) == 120
    assert plan[0] == "walk_up"
    # A repeat count past MAX_REPEAT is a legal probe on paper too.
    assert len(agent_cli.expand_actions(["up:120"], batch=False)) == 120
    # The frame budget is about emulator time, which a simulation does not spend.
    assert agent_cli.expand_actions(["wait_600"] * 10, batch=False)


def test_even_a_simulated_plan_has_a_ceiling():
    with pytest.raises(agent_cli.ActionError, match="even on paper"):
        agent_cli.expand_actions(["up"] * (agent_cli.MAX_PLAN_ACTIONS + 1), batch=False)


def test_a_simulated_plan_is_still_checked_for_typos():
    with pytest.raises(agent_cli.ActionError, match="unknown action"):
        agent_cli.expand_actions(["sideways"], batch=False)
    with pytest.raises(agent_cli.ActionError, match="the limit is"):
        agent_cli.expand_actions(["wait_1000000000"], batch=False)


def test_a_batch_within_the_action_cap_can_still_bust_the_frame_budget():
    with pytest.raises(agent_cli.ActionError, match="frames"):
        agent_cli.expand_actions(["wait_600"] * 10)


def test_the_caps_match_the_numbers_the_server_enforces():
    assert agent_cli.MAX_ACTIONS_PER_BATCH == 40
    assert agent_cli.MAX_FRAMES_PER_ACTION == 600
    assert agent_cli.MAX_FRAMES_PER_BATCH == 3600


# ---------------------------------------------------------------------------
# Route, goto, calc, frontier, sim, guide, progress
# ---------------------------------------------------------------------------


def test_route_prints_hops_and_never_claims_button_presses(stub, capsys):
    stub.route(
        "GET",
        "/route?to=Cerulean%20City",
        {
            "from": "Pewter City",
            "to": "Cerulean City",
            "distance": 3,
            "hops": [
                {
                    "from": "Pewter City",
                    "to": "Route 3",
                    "kind": "connection",
                    "at": None,
                    "edge": "east",
                },
                {
                    "from": "Route 4",
                    "to": "Cerulean City",
                    "kind": "warp",
                    "at": [12, 8],
                    "edge": None,
                },
            ],
        },
    )
    assert run(stub, "route", "Cerulean", "City") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "2 hops" in out
    assert "Route 3" in out and "(12, 8)" in out


def test_route_says_so_when_there_is_no_way_through(stub, capsys):
    stub.route("GET", "/route?to=Nowhere", {"from": "Pewter City", "to": "Nowhere", "hops": None})
    assert run(stub, "route", "Nowhere") == agent_cli.EXIT_OK
    assert "no route" in capsys.readouterr().out


def test_route_reports_arrival_rather_than_an_empty_list(stub, capsys):
    stub.route(
        "GET",
        "/route?to=Pewter%20City",
        {"from": "Pewter City", "to": "Pewter City", "hops": []},
    )
    assert run(stub, "route", "Pewter", "City") == agent_cli.EXIT_OK
    assert "already on" in capsys.readouterr().out


def test_goto_sends_a_map_name(stub):
    stub.route("POST", "/goto", {"walked": 18, "arrived": True, "stopped_because": "arrived"})
    assert run(stub, "goto", "Cerulean", "City") == agent_cli.EXIT_OK
    assert stub.requests[-1]["body"] == {"target": "Cerulean City"}


def test_goto_sends_coordinates_when_given_a_pair(stub):
    stub.route("POST", "/goto", {"walked": 4, "arrived": True, "stopped_because": "arrived"})
    assert run(stub, "goto", "12,8") == agent_cli.EXIT_OK
    assert stub.requests[-1]["body"] == {"x": 12, "y": 8}


def test_calc_shows_damage_and_what_can_kill_you(stub, capsys):
    stub.route(
        "GET",
        "/calc",
        {
            "enemy": {"species": "Onix", "level": 14, "hp": 43, "types": ["Rock", "Ground"]},
            "moves": [
                {
                    "move": "Bubble",
                    "type": "Water",
                    "power": 20,
                    "effectiveness": 4.0,
                    "damage": [28, 34],
                    "turns_to_ko": 2,
                },
                {
                    "move": "Tackle",
                    "type": "Normal",
                    "power": 35,
                    "effectiveness": 1.0,
                    "damage": [4, 6],
                    "turns_to_ko": None,
                },
            ],
            "threat": 21,
        },
    )
    assert run(stub, "calc") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Onix L14" in out
    assert "x4" in out
    assert "KO in 2" in out
    assert "cannot KO" in out
    assert "worst incoming: 21" in out


def test_sim_names_the_step_that_would_hit_the_wall(stub, capsys):
    stub.route(
        "POST",
        "/sim",
        {
            "end": [12, 8],
            "facing": "up",
            "steps": 3,
            "blocked_at": 3,
            "blocked_by": "wall",
            "warp_at": None,
        },
    )
    assert run(stub, "sim", "up:6") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "blocked at step 3" in out
    assert "walk_up" in out and "wall" in out


def test_sim_reports_a_clean_plan(stub, capsys):
    stub.route(
        "POST",
        "/sim",
        {
            "end": [12, 2],
            "facing": "up",
            "steps": 6,
            "blocked_at": None,
            "blocked_by": None,
            "warp_at": None,
        },
    )
    assert run(stub, "sim", "up:6") == agent_cli.EXIT_OK
    assert "clean" in capsys.readouterr().out


def test_sim_refuses_a_bad_plan_without_asking_the_server(stub):
    assert run(stub, "sim", "sideways") == agent_cli.EXIT_BAD_USAGE
    assert stub.requests == []


def test_sim_sends_a_plan_longer_than_one_batch(stub, capsys):
    """The server simulates from the live tile, so a long plan cannot be split.

    `poke act` can send the first forty and re-plan from where they land. There
    is no equivalent for `poke sim`: it always starts where the player actually
    is, so refusing a long plan left the model with nothing to do but ask again,
    which it did 505 times in one session.
    """
    stub.route("POST", "/sim", {"end": [9, 9], "facing": "up", "blocked_at": None})

    assert run(stub, "sim", "up:40", "left:40", "down:40") == agent_cli.EXIT_OK

    assert len(stub.requests[-1]["body"]["actions"]) == 120
    assert "clean" in capsys.readouterr().out


def test_frontier_truncates_a_long_list_but_says_it_did(stub, capsys):
    stub.route(
        "GET",
        "/frontier",
        {
            "map": "Mt Moon B1F",
            "from": [4, 4],
            "tiles": [[x, 0] for x in range(30)],
            "count": 30,
        },
    )
    assert run(stub, "frontier", "--limit", "5") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "30 unseen" in out
    assert "and 25 more" in out


def test_frontier_says_how_far_to_trust_the_list(stub, capsys):
    """The confidence fields are why 34 of 47 frontier calls asked for --json.

    Then 35 of them piped that to `head`, and the payload puts the tile array
    first: `head -c 400` stops inside it, before `count`, `confirmed_count`,
    `believed_count` and `basis`. The truncation cut off the one thing the JSON
    had been opened for.
    """
    stub.route(
        "GET",
        "/frontier",
        {
            "map": "Mt Moon B1F",
            "from": [4, 4],
            "tiles": [[1, 1], [2, 2], [3, 3]],
            "count": 3,
            "confirmed_count": 2,
            "believed_count": 1,
            "basis": "the live 90-tile window plus the remembered map",
        },
    )

    assert run(stub, "frontier") == agent_cli.EXIT_OK

    out = capsys.readouterr().out
    assert "2 confirmed, 1 believed" in out
    assert "the remembered map" in out


def test_guide_with_no_argument_lists_sections(stub, capsys):
    stub.route("GET", "/guide", {"outline": "speedrun_glitchless/mt-moon  Cross Mt. Moon"})
    assert run(stub, "guide") == agent_cli.EXIT_OK
    assert "mt-moon" in capsys.readouterr().out


def test_guide_reads_one_section_by_reference(stub, capsys):
    stub.route(
        "GET",
        "/guide?ref=speedrun_glitchless/mt-moon",
        {
            "guide": "speedrun_glitchless",
            "slug": "mt-moon",
            "title": "Mt. Moon",
            "body": "Head north from the centre.",
        },
    )
    assert run(stub, "guide", "speedrun_glitchless/mt-moon") == agent_cli.EXIT_OK
    assert "Head north" in capsys.readouterr().out


def test_guide_search_lists_matches(stub, capsys):
    stub.route(
        "GET",
        "/guide?q=mt%20moon",
        {
            "results": [
                {
                    "ref": "speedrun_glitchless/mt-moon",
                    "title": "Mt. Moon",
                    "summary": "Cross to Route 4",
                }
            ]
        },
    )
    assert run(stub, "guide", "-s", "mt", "moon") == agent_cli.EXIT_OK
    assert "Cross to Route 4" in capsys.readouterr().out


def test_progress_reports_the_ladder_and_the_cost(stub, capsys):
    stub.route(
        "GET",
        "/progress",
        {
            "count": 23,
            "total": 63,
            "furthest": "EVENT_BEAT_BROCK",
            "furthest_label": "Defeated Brock",
            "latest": ["Defeated Brock"],
            "presses": 4207,
        },
    )
    assert run(stub, "progress") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "23/63" in out
    assert "4207 presses" in out


def _mt_moon_progress_payload() -> dict:
    """The real payload for where the fourteen-hour run sat: Mt. Moon, one badge."""
    from pokemon_agent import capabilities
    from pokemon_agent.milestones import MILESTONES, frontier

    reached = [m for m in MILESTONES[:9] if m.id != "EVENT_GOT_TOWN_MAP"]
    summary = {
        "count": len(reached),
        "total": len(MILESTONES),
        "furthest": reached[-1].id,
        "latest": [m.id for m in reached[-5:]],
        "frontier": [m.id for m in frontier({m.id for m in reached})],
    }
    return capabilities.progress_payload(summary, 12043)


def test_progress_offers_the_open_milestones_and_says_what_each_opens(stub, capsys):
    """The menu the harness narrows, printed as the choices themselves.

    A run banked zero of 63 milestones in fourteen hours with `progress` called
    zero times, so the verb has to be worth calling: the graph knows which few
    rungs the game will currently permit, and this is where that is said.
    Naming what each one opens is what lets one be preferred over another.
    """
    stub.route("GET", "/progress", _mt_moon_progress_payload())
    assert run(stub, "progress") == agent_cli.EXIT_OK
    out = capsys.readouterr().out

    assert "open now (7), pick one:" in out
    assert "Defeated Misty" in out
    assert "Got HM05 Flash -> dark caves, once the Boulder Badge allows it" in out
    # Behind the Cut trees, so not a choice yet however much it looks like one.
    assert "Erika" not in out


def test_the_whole_progress_answer_still_fits_in_a_few_hundred_bytes(stub, capsys):
    """Tool text is 98% of this model's prompt and is paid again every turn.

    Switching `act` from JSON to prose cut it 52%; a frontier that arrived as a
    JSON array of objects would hand most of that back. The bound is generous
    against the widest point of the graph, which is twelve open rungs.
    """
    stub.route("GET", "/progress", _mt_moon_progress_payload())
    assert run(stub, "progress") == agent_cli.EXIT_OK
    out = capsys.readouterr().out

    assert len(out.encode("utf-8")) < 700
    assert "{" not in out and "[" not in out


# ---------------------------------------------------------------------------
# `act` prints prose, not a JSON object
#
# It was the only acting verb that printed its payload raw: 4,251 of them over
# one run, 960,537 bytes. Tool text is 98% of the model's prompt, so each one is
# paid when it arrives and again on every turn afterwards.
# ---------------------------------------------------------------------------


WALK = {
    "actions_executed": 4,
    "map": "Route 4",
    "x": 19,
    "y": 11,
    "facing": "down",
    "moves": ["up", "down", "left"],
    "run": {"up": 4, "left": 3},
    "exits": {"Cerulean City": "east edge", "Mt Moon 1F": [18, 5]},
    "mode": "overworld",
    "dialog": False,
    "battle": False,
    "hp": "73/73",
    "moved": 4,
}


def test_the_line_carries_where_you_are_and_what_the_batch_did():
    line = agent_cli.action_lines(WALK)
    assert "Route 4 (19,11)" in line
    assert "facing down" in line
    assert "moved 4" in line
    assert "hp 73/73" in line


def test_run_and_exits_survive_the_shaping():
    # These are the two fields the model measurably acts on.
    line = agent_cli.action_lines(WALK)
    assert "up:4" in line and "left:3" in line
    assert "Cerulean City east edge" in line
    assert "Mt Moon 1F (18, 5)" in line


def test_a_blocked_batch_says_which_step_failed():
    line = agent_cli.action_lines({**WALK, "moved": 0, "blocked_after": 1})
    assert "moved 0" in line
    assert "blocked after 1" in line


def test_flags_appear_only_when_they_are_true():
    quiet = agent_cli.action_lines(WALK)
    assert "press a" not in quiet
    assert "warp" not in quiet
    assert "stood here" not in quiet

    loud = agent_cli.action_lines(
        {
            **WALK,
            "faces": "sign",
            "here_before": 9,
            "on_warp": True,
            "warp": {"to": "Route 4", "step": "down"},
        }
    )
    assert "press a" in loud
    assert "stood here 9 times" in loud
    assert "warp to Route 4, step down" in loud


def test_a_battle_says_what_it_is_and_what_you_can_hit_it_with():
    line = agent_cli.action_lines(
        {
            **WALK,
            "battle": True,
            "enemy": "Zubat L7 23/23 (Poison/Flying)",
            "your_moves": ["Ember", "Growl"],
            "menu": "moves",
            "highlighted": "Ember",
        }
    )
    assert "BATTLE vs Zubat L7" in line
    assert "Ember, Growl" in line
    assert "menu moves on Ember" in line


#: What the server sends on a battle frame: the position it can still read, the
#: fight, and a sentence wherever it dropped a field -- the walk directions,
#: which no battle frame can take, and facing, which it holds a stale value for.
BATTLE = {
    "actions_executed": 1,
    "map": "Mt Moon 1F",
    "x": 15,
    "y": 33,
    "facing_unread": "facing unread in a battle: the byte is stale from before the encounter",
    "exits": {"Route 4": [15, 35]},
    "mode": "battle",
    "dialog": True,
    "battle": True,
    "hp": "15/70",
    "no_walk": "no walking in a battle: the d-pad drives the battle menu",
    "enemy": "Geodude L10 28/28 (Rock/Ground)",
    "your_moves": ["Scratch", "Ember"],
    "menu": "other",
}


def test_a_battle_line_carries_the_position_the_payload_knows():
    """Measured live: `Mt Moon 1F (None,None) facing None hp 39/73 exits Mt Mo…`.

    The coordinates were in RAM the whole time -- a battle does not move the
    player -- and this line printing None for them cost a `poke state` call to
    recover, fifteen times in one 457-call session.
    """
    line = agent_cli.action_lines(BATTLE)

    assert "Mt Moon 1F (15,33)" in line
    assert "None" not in line
    # And the field the server refused is named, rather than the line simply
    # ending early as though facing had stopped mattering.
    assert "facing unread in a battle" in line


def test_a_frame_with_no_position_says_so_rather_than_printing_none():
    """When the server really could not read one, say that. A null teaches nothing."""
    line = agent_cli.action_lines({"map": "Mt Moon 1F", "hp": "15/70"})

    assert "Mt Moon 1F (position unread)" in line
    assert "None" not in line
    # No facing either: an unread field is left out, not rendered as a reading.
    assert "facing" not in line


def test_the_reason_there_are_no_walk_directions_is_printed():
    """Otherwise the line looks like an overworld frame with nowhere to go."""
    line = agent_cli.action_lines(BATTLE)

    assert "the d-pad drives the battle menu" in line
    assert "run " not in line


def test_battle_text_is_not_rendered_as_a_menu_with_a_missing_entry():
    """`menu: other` is the reader saying no battle menu is up, so there is no
    cursor to name. It used to print "menu other on None"."""
    line = agent_cli.action_lines(BATTLE)

    assert "no battle menu up" in line
    assert "on None" not in line


def test_a_battle_still_says_which_entry_the_cursor_is_on():
    line = agent_cli.action_lines({**BATTLE, "menu": "moves", "highlighted": "Ember"})

    assert "menu moves on Ember" in line


def test_an_unsettled_frame_warns_before_the_position_is_believed():
    """`settled: false` means the game was still moving when this was read.

    Ten frames into a gate warp the reads say `Route 2 (5,0)`: one map's name
    with another map's coordinates. The line has to arrive before anything read
    off that frame, because everything else on it came from the same read.
    """
    line = agent_cli.action_lines({**WALK, "settled": False})

    assert "still moving" in line
    assert line.splitlines()[1].startswith("the game was still moving")


def test_a_warp_whose_destination_is_unread_is_still_a_warp():
    """ "on a warp to None" reads as a warp that leads nowhere, which is worse
    than saying the destination could not be read."""
    line = agent_cli.action_lines({**WALK, "on_warp": True, "warp": {"step": "up"}})

    assert "on a warp (destination unread), step up" in line
    assert "None" not in line


def test_the_shaped_line_is_much_smaller_than_the_object():
    import json as _json

    raw = len(_json.dumps(WALK, separators=(",", ":")))
    shaped = len(agent_cli.action_lines(WALK))
    assert shaped < raw * 0.6, f"{shaped}B vs {raw}B is not worth the change"


def test_json_is_still_available_for_a_script(stub, capsys):
    stub.route("POST", "/action", WALK)
    assert run(stub, "act", "--json", "up") == agent_cli.EXIT_OK
    out = capsys.readouterr().out
    assert out.strip().startswith("{")
    assert "actions_executed" in out
