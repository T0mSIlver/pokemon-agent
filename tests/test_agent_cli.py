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
    assert json.loads(capsys.readouterr().out) == {
        "actions_executed": 3,
        "x": 5,
        "y": 6,
        "moved": 2,
    }


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


def test_saves_lists_names(stub, capsys):
    stub.route(
        "GET",
        "/saves",
        {"saves": [{"name": "brock", "modified": 0}, {"name": "forest", "modified": 0}]},
    )

    assert run(stub, "saves") == 0

    out = capsys.readouterr().out
    assert "brock" in out
    assert "forest" in out


def test_saves_says_so_when_empty(stub, capsys):
    stub.route("GET", "/saves", {"saves": []})

    assert run(stub, "saves") == 0
    assert capsys.readouterr().out.strip() == "no saves"


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
