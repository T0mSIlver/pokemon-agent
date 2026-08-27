"""The Python client, driven against a stub server rather than a live game.

The live end-to-end proof is `test_agent_api_live.py`; this file pins the parts
that have to be right before a round trip happens at all — the caps, the
chunking, the pacing, and the shape of every answer a script branches on.
"""

from __future__ import annotations

import json
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pokemon_agent import agent_api, agent_cli

STATE_PAYLOAD = {
    "player": {
        "name": "RED",
        "money": 1527,
        "badges": ["Boulder"],
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
            "moves": [{"name": "Ember", "pp": 20}],
        }
    ],
    "bag": [{"item": "Town Map", "quantity": 1}, {"item": "Poke Ball", "quantity": 2}],
    "battle": {"in_battle": False},
    "dialog": {"active": False},
    "map": {"map_id": 2, "map_name": "Pewter City"},
}

WALK_RESULT = {
    "actions_executed": 4,
    "map": "Pewter City",
    "x": 13,
    "y": 9,
    "facing": "up",
    "moves": ["up", "down"],
    "mode": "overworld",
    "dialog": False,
    "battle": False,
    "hp": "19/36",
    "moved": 4,
}


class StubServer:
    """A few canned routes and a record of what the client actually sent."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, object]] = {}
        self.requests: list[dict] = []
        self.default: tuple[int, object] | None = None
        self._http: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def route(self, method: str, path: str, payload: object, status: int = 200) -> None:
        self.routes[(method, path)] = (status, payload)

    def always(self, payload: object, status: int = 200) -> None:
        """Answer every request the same way, whatever the path."""
        self.default = (status, payload)

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
                fallback = stub.default or (
                    404,
                    {"detail": f"no stub route for {method} {self.path}"},
                )
                status, payload = stub.routes.get((method, self.path), fallback)
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

    def action_bodies(self) -> list[list[str]]:
        return [
            request["body"]["actions"]
            for request in self.requests
            if request["path"] == "/action" and request["body"]
        ]


@pytest.fixture
def stub():
    server = StubServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def poke(stub):
    return agent_api.Client(port=stub.port)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_limits_match_the_cli():
    """One set of caps. The CLI is where they are kept honest against the server."""
    for name in (
        "MAX_REPEAT",
        "MAX_ACTIONS_PER_BATCH",
        "MAX_FRAMES_PER_ACTION",
        "MAX_FRAMES_PER_BATCH",
        "FRAMES_PER_INPUT",
        "FRAMES_DIALOG_WORST_CASE",
        "DEFAULT_PORT",
    ):
        assert getattr(agent_api, name) == getattr(agent_cli, name), name
    assert agent_api.ACTIONS == agent_cli.ACTIONS
    assert agent_api.ALIASES == agent_cli.ALIASES


def test_client_paces_below_the_servers_cap():
    assert agent_api.RATE_MAX_BATCHES < 60


def test_expand_takes_the_same_vocabulary_as_the_cli():
    assert agent_api.expand("up", "up", "a") == ["walk_up", "walk_up", "press_a"]
    assert agent_api.expand("right:6") == ["walk_right"] * 6
    assert agent_api.expand(["up", "a"]) == ["walk_up", "press_a"]
    assert agent_api.expand("wait", "adialog") == ["wait_60", "a_until_dialog_end"]


def test_expand_refuses_what_the_cli_refuses():
    for bad in (["up:0"], ["up:lots"], ["sideways"], ["wait_1000000000"]):
        with pytest.raises(agent_api.ActionError):
            agent_api.expand(*bad)


def test_one_batch_refuses_an_illegal_batch_before_sending_it():
    with pytest.raises(agent_api.ActionError, match="the limit is 40"):
        agent_api.one_batch("up:40", "down:40")
    with pytest.raises(agent_api.ActionError, match="the limit is 3600"):
        agent_api.one_batch("adialog:12", "wait_600")


def test_chunks_respect_both_caps():
    walks = agent_api.chunks(agent_api.expand("up:40", "right:40", "down:20"))
    assert [len(batch) for batch in walks] == [40, 40, 20]
    for batch in walks:
        assert sum(agent_api.frames_for(action) for action in batch) <= 3600

    # Forty dialogs would be 12,000 frames, so the frame budget closes each
    # batch long before the action count does.
    dialogs = agent_api.chunks(["a_until_dialog_end"] * 40)
    assert all(len(batch) == 12 for batch in dialogs[:-1])
    for batch in dialogs:
        assert sum(agent_api.frames_for(action) for action in batch) <= 3600


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_a_refusal_is_an_exception_carrying_the_servers_words(stub, poke):
    detail = "You are not in a battle."
    stub.route("GET", "/calc", {"detail": detail}, status=409)

    with pytest.raises(agent_api.ServerError) as caught:
        poke.calc()

    assert caught.value.status == 409
    assert str(caught.value) == detail
    assert caught.value.detail == detail
    assert isinstance(caught.value, agent_api.PokeError)


def test_a_rate_limit_arrives_as_the_servers_lecture(stub, poke):
    stub.route("POST", "/action", {"detail": "More than 60 action batches in 60s."}, status=429)

    with pytest.raises(agent_api.ServerError, match="60 action batches"):
        poke.act("up")


def test_no_server_is_unreachable_not_a_traceback():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = sock.getsockname()[1]

    with pytest.raises(agent_api.Unreachable):
        agent_api.Client(port=closed).state()


# ---------------------------------------------------------------------------
# Looking
# ---------------------------------------------------------------------------


def test_state_is_a_view_not_json_soup(stub, poke):
    stub.route("GET", "/state", STATE_PAYLOAD)

    state = poke.state()

    assert state.map == "Pewter City"
    assert state.position == (13, 13)
    assert state.badges == ["Boulder"]
    assert state.lead.species == "Charmander"
    assert state.lead.moves == ["Ember"]
    assert state.hp == 19 and state.max_hp == 36
    assert state.has("poke ball") == 2
    assert state.has("Full Restore") == 0
    assert state.in_battle is False
    assert state.raw == STATE_PAYLOAD


def test_sim_says_whether_the_plan_walks(stub, poke):
    stub.route(
        "POST",
        "/sim",
        {"end": [13, 7], "facing": "up", "steps": 6, "blocked_at": None, "warp_at": None},
    )
    clean = poke.sim("up:6")
    assert clean.ok is True
    assert clean.end == (13, 7)
    assert clean.blocked_action is None

    stub.route(
        "POST",
        "/sim",
        {"end": [13, 11], "facing": "up", "steps": 2, "blocked_at": 2, "blocked_by": "wall"},
    )
    blocked = poke.sim("up:6")
    assert blocked.ok is False
    assert blocked.blocked_at == 2
    assert blocked.blocked_action == "walk_up"
    assert blocked.clear_prefix == ["walk_up", "walk_up"]


def test_a_plan_written_as_one_string_means_the_same_as_separate_tokens(stub, poke):
    """`poke sim down:5 right:2` is the shell form, so it is the form of habit.

    Written as one Python string it used to arrive as a single token and raise
    `unknown action 'down:5 right:2'`. It cost four tracebacks across the
    sessions and an abandoned per-row probe loop. No action name has a space
    in it, so splitting one can never mean anything else.
    """
    assert agent_api.expand("down:2 right:2") == agent_api.expand("down:2", "right:2")
    assert agent_api.expand(["up a"]) == ["walk_up", "press_a"]

    stub.route("POST", "/sim", {"end": [13, 7], "facing": "up", "steps": 4, "blocked_at": None})
    poke.sim("down:2 right:2")
    assert stub.requests[-1]["body"]["actions"] == [
        "walk_down",
        "walk_down",
        "walk_right",
        "walk_right",
    ]


def test_a_battle_result_says_where_the_player_is_standing():
    """A battle payload carries x and y now: the battle does not move the player.

    One session could not find `.position` — `__dict__` does not list a
    property and `inspect.getsource` raises on a dataclass `__repr__` — and
    fell back to a regex over this string. The regex returned None the first
    time a Zubat appeared and its 120-step search stopped at step 6 printing
    "no pos in: battle vs Zubat...". The coordinates were readable on that
    frame the whole time.
    """
    result = agent_api.Result.from_payload(
        {
            "actions_executed": 1,
            "map": "Mt Moon 1F",
            "x": 15,
            "y": 33,
            "mode": "battle",
            "battle": True,
            "hp": "22/73",
            "enemy": "Zubat L7 23/23 (Poison/Flying)",
            "no_walk": "no walking in a battle: the d-pad drives the battle menu",
        }
    )

    assert result.position == (15, 33)
    text = str(result)
    assert "Mt Moon 1F (15,33)" in text
    assert "Zubat" in text
    assert "None" not in text


def test_a_battle_result_carries_the_priced_moves_and_what_is_coming_back():
    result = agent_api.Result.from_payload(
        {
            "battle": True,
            "map": "Mt Moon 1F",
            "x": 15,
            "y": 33,
            "hp": "39/73",
            "you": "Charmeleon L25",
            "enemy": "Zubat L9 28/28 (Poison/Flying)",
            "your_moves": ["Ember Fire 12PP 41-49 KO in 1", "Growl Normal 40PP no damage"],
            "incoming": "Leech Life up to 1",
        }
    )

    assert result.you == "Charmeleon L25"
    assert result.battle_moves[0].endswith("KO in 1")
    assert result.incoming == "Leech Life up to 1"
    assert result.no_damage is None and result.locked_in is None


def test_a_battle_result_prints_the_two_notes_that_predict_a_refusal():
    """`locked_in` and `no_damage` both mean the next `fight` call is refused.

    A script that reads only `str(result)` has to see them, or it retries the
    same refusal -- which is what a run using Rage 77 times did.
    """
    result = agent_api.Result.from_payload(
        {
            "battle": True,
            "map": "Mt Moon B2F",
            "x": 11,
            "y": 19,
            "hp": "73/73",
            "enemy": "Rattata L13 17/32 (Normal)",
            "locked_in": "Rage has locked this Pokemon in: the game keeps attacking with it.",
        }
    )

    assert "Rage has locked this Pokemon in" in str(result)


def test_a_result_with_no_position_says_so_rather_than_printing_none():
    """A frame whose position could not be read is a refusal, not a coordinate.

    `(None,None)` is a well-formed answer with a hole in it: it reads as a
    reading, and the two things a caller does with it — parse it, or believe
    it — both go wrong.
    """
    result = agent_api.Result.from_payload({"map": "Mt Moon 1F", "battle": True})

    assert "position unread" in str(result)
    assert "None,None" not in str(result)


def test_no_walk_travels_with_the_empty_direction_list():
    """`directions == []` alone cannot tell a battle from being walled in."""
    result = agent_api.Result.from_payload(
        {
            "map": "Oak's Lab",
            "x": 5,
            "y": 3,
            "dialog": True,
            "no_walk": "no walking while a box is open: the d-pad works the box, not the player",
        }
    )

    assert result.directions == []
    assert "the d-pad works the box" in (result.no_walk or "")


def test_saves_lists_the_named_ones(stub, poke):
    """The plain list is the newest forty of everything.

    The harness writes an `auto__` checkpoint on every battle and every map
    change: 300 of the run's 465 saves. Newest-first with the server's default
    limit of 40, every row was one of those and not one of the 165 names a
    script could load appeared.
    """
    stub.route("GET", "/saves?named=1", {"saves": [{"name": "brock"}, {"name": "forest"}]})

    assert poke.saves() == ["brock", "forest"]
    assert stub.requests[-1]["path"] == "/saves?named=1"


def test_frontier_slices_like_the_tile_list_it_is(stub, poke):
    stub.route(
        "GET",
        "/frontier",
        {"map": "Route 3", "from": [4, 4], "tiles": [[5, 4], [6, 4], [7, 4], [8, 4]], "count": 4},
    )

    frontier = poke.frontier()

    assert len(frontier) == 4
    assert frontier[:3] == [(5, 4), (6, 4), (7, 4)]
    assert list(frontier)[0] == (5, 4)
    assert frontier.origin == (4, 4)


def test_route_reports_hops_and_the_reason_when_there_are_none(stub, poke):
    stub.route(
        "GET",
        "/route?to=Cerulean+City",
        {
            "from": "Pewter City",
            "to": "Cerulean City",
            "hops": [{"from": "Pewter City", "to": "Route 3", "kind": "edge", "edge": "east"}],
        },
    )
    route = poke.route("Cerulean City")
    assert route.ok is True
    assert route.distance == 1
    assert route.next_hop.to_map == "Route 3"

    stub.route(
        "GET",
        "/route?to=Nowhere",
        {"from": "Pewter City", "to": "Nowhere", "hops": None, "reason": "not connected"},
    )
    missing = poke.route("Nowhere")
    assert missing.ok is False
    assert "not connected" in str(missing)


def test_calc_picks_the_obvious_move(stub, poke):
    stub.route(
        "GET",
        "/calc",
        {
            "enemy": {"species": "Onix", "level": 14, "hp": 30, "max_hp": 30, "types": ["Rock"]},
            "moves": [
                {"move": "Ember", "damage": [4, 6], "effectiveness": 0.5, "turns_to_ko": None},
                {"move": "Scratch", "damage": [9, 11], "effectiveness": 1, "turns_to_ko": 3},
            ],
            "threat": 12,
        },
    )

    calc = poke.calc()

    assert calc.enemy.species == "Onix"
    assert calc.best.move == "Scratch"
    assert calc.best.high == 11
    assert calc.threat == 12


def test_the_obvious_move_is_never_one_with_no_pp(stub, poke):
    """`best` used to rank the whole table, dry moves included.

    So it named an Ember at 0 PP as the obvious pick and `poke fight ember` was
    then refused -- 12 times in one run, whose auto-saved battle entries had a
    dry damaging move in 54 of 106 frames.
    """
    stub.route(
        "GET",
        "/calc",
        {
            "enemy": {"species": "Zubat", "level": 9, "hp": 28, "types": ["Poison", "Flying"]},
            "moves": [
                {"move": "Ember", "damage": [39, 46], "turns_to_ko": None, "pp": 0},
                {"move": "Rage", "damage": [8, 10], "turns_to_ko": 4, "pp": 20},
                {"move": "Growl", "damage": [0, 0], "turns_to_ko": None, "pp": 39},
            ],
            "threat": 1,
            "threat_move": "Leech Life",
        },
    )

    calc = poke.calc()

    assert calc.best.move == "Rage"
    assert "out of PP" in str(calc.moves[0])
    assert calc.moves[0].usable is False
    assert calc.threat_move == "Leech Life"


def test_a_table_with_nothing_usable_has_no_obvious_move(stub, poke):
    stub.route(
        "GET",
        "/calc",
        {
            "enemy": {"species": "Zubat", "level": 9, "hp": 28, "types": ["Poison", "Flying"]},
            "moves": [{"move": "Ember", "damage": [39, 46], "turns_to_ko": None, "pp": 0}],
            "threat": 1,
        },
    )

    assert poke.calc().best is None


def test_progress_and_map_are_shaped(stub, poke):
    stub.route(
        "GET",
        "/progress",
        {"count": 4, "total": 20, "furthest_label": "Beat Brock", "latest": [], "presses": 900},
    )
    assert poke.progress().count == 4
    assert "Beat Brock" in str(poke.progress())

    stub.route(
        "GET",
        "/map",
        {
            "map_id": 2,
            "map_name": "Pewter City",
            "width": 40,
            "height": 36,
            "coverage": {"seen": 1153, "walked": 200, "total": 1440, "percent": 80.1},
            "warps": [{"x": 2, "y": 7}],
            "player": {"x": 13, "y": 13},
            "image_path": "/tmp/map.png",
        },
    )
    view = poke.map()
    assert view.warps == [(2, 7)]
    assert view.player == (13, 13)
    assert view.percent == 80.1


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


def test_act_sends_one_batch_and_reports_what_it_did(stub, poke):
    stub.route("POST", "/action", {**WALK_RESULT, "blocked_after": 2, "on_warp": True})

    result = poke.act("up:4")

    assert stub.action_bodies() == [["walk_up"] * 4]
    assert result.moved == 4
    assert result.blocked_after == 2
    assert result.blocked is True
    assert result.on_warp is True
    assert result.hp == 19 and result.max_hp == 36
    assert result.in_battle is False
    assert result.position == (13, 9)


def test_act_refuses_a_batch_the_server_would_refuse_without_calling_it(stub, poke):
    with pytest.raises(agent_api.ActionError, match="walk()"):
        poke.act("up:40", "down:40")
    assert stub.requests == []


def test_goto_takes_a_map_or_a_tile(stub, poke):
    stub.route(
        "POST", "/goto", {**WALK_RESULT, "walked": 4, "arrived": True, "stopped_because": ""}
    )

    assert poke.goto("Cerulean City").arrived is True
    assert poke.goto((12, 4)).walked == 4
    assert poke.goto(12, 4).arrived is True

    assert [request["body"] for request in stub.requests] == [
        {"target": "Cerulean City"},
        {"x": 12, "y": 4},
        {"x": 12, "y": 4},
    ]


def test_fight_and_flee_carry_the_move_name(stub, poke):
    stub.route("POST", "/battle/fight", {**WALK_RESULT, "battle": True, "enemy": "Onix L14 8/30"})
    stub.route("POST", "/battle/run", WALK_RESULT)

    assert poke.fight("ember").in_battle is True
    assert poke.flee().in_battle is False
    assert stub.requests[0]["body"] == {"move": "ember"}


# ---------------------------------------------------------------------------
# The chunked walker
# ---------------------------------------------------------------------------


def test_walk_splits_a_long_path_into_legal_batches(stub, poke):
    stub.route("GET", "/state", STATE_PAYLOAD)
    stub.route("POST", "/action", WALK_RESULT)

    report = poke.walk("up:60", "right:30")

    sizes = [len(batch) for batch in stub.action_bodies()]
    assert sizes == [40, 40, 10]
    assert all(size <= agent_api.MAX_ACTIONS_PER_BATCH for size in sizes)
    assert report.done is True
    assert report.remaining == []
    assert len(report.sent) == 90
    assert report.moved == 12  # four per batch, per the stub


def test_walk_stops_when_a_battle_starts(stub, poke):
    stub.route("GET", "/state", STATE_PAYLOAD)
    stub.route("POST", "/action", {**WALK_RESULT, "battle": True, "enemy": "Pidgey L6 14/14"})

    report = poke.walk("up:60")

    assert len(stub.action_bodies()) == 1
    assert report.done is False
    assert len(report.remaining) == 20
    assert report.stopped_because == "a battle started"
    assert report.in_battle is True


def test_walk_stops_when_the_path_is_blocked(stub, poke):
    stub.route("GET", "/state", STATE_PAYLOAD)
    stub.route("POST", "/action", {**WALK_RESULT, "moved": 1, "blocked_after": 1})

    report = poke.walk("up:60")

    assert len(stub.action_bodies()) == 1
    assert "blocked after 1" in report.stopped_because
    assert report.moved == 1


def test_walk_stops_when_the_map_changes_inside_the_first_batch(stub, poke):
    """The case a walk must not miss: a warp crossed with the plan still queued."""
    stub.route("GET", "/state", STATE_PAYLOAD)  # Pewter City, where the plan was written
    stub.route("POST", "/action", {**WALK_RESULT, "map": "Route 3"})

    report = poke.walk("up:60")

    assert len(stub.action_bodies()) == 1
    assert report.stopped_because == "the map changed to Route 3"
    assert len(report.remaining) == 20


def test_walk_takes_a_known_starting_map_without_asking(stub, poke):
    stub.route("POST", "/action", WALK_RESULT)

    report = poke.walk("up:60", start_map="Pewter City")

    assert [request["path"] for request in stub.requests] == ["/action", "/action"]
    assert report.done is True


def test_walk_refuses_to_become_a_runaway_loop(stub, poke):
    with pytest.raises(agent_api.ActionError, match="more than 400 times"):
        poke.walk("up:401")
    with pytest.raises(agent_api.ActionError, match="the cap is 400"):
        poke.walk("up:200", "down:201")
    assert stub.requests == []


def test_walk_paces_itself_under_the_rate_limit(stub, poke, monkeypatch):
    """The client waits its turn rather than earning the server's 429."""
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        # Standing in for the clock: waiting `seconds` is the same thing as the
        # recorded calls being `seconds` further in the past.
        slept.append(seconds)
        poke._sent = deque((sent - seconds for sent in poke._sent), maxlen=poke._sent.maxlen)

    monkeypatch.setattr(agent_api.time, "sleep", fake_sleep)
    monkeypatch.setattr(agent_api, "RATE_MAX_BATCHES", 2)
    stub.route("POST", "/action", WALK_RESULT)

    poke.walk("up:120", start_map="Pewter City")

    assert len(stub.action_bodies()) == 3
    # Two batches went straight out; the third had to wait for the window.
    assert len(slept) == 1
    assert 0 < slept[0] <= agent_api.RATE_WINDOW_SECONDS + 1


def test_pacing_gives_up_rather_than_hanging(poke, monkeypatch):
    monkeypatch.setattr(agent_api, "RATE_MAX_BATCHES", 1)
    monkeypatch.setattr(agent_api, "MAX_PACE_WAIT_SECONDS", 0.0)
    poke._pace()

    with pytest.raises(agent_api.RateLimited):
        poke._pace()


# ---------------------------------------------------------------------------
# Guide and game database
# ---------------------------------------------------------------------------


def test_guide_search_and_read(stub, poke):
    stub.route(
        "GET",
        "/guide?q=mt+moon",
        {"results": [{"ref": "speedrun/mt-moon", "title": "Mt. Moon", "summary": "the cave"}]},
    )
    hits = poke.guide.search("mt moon")
    assert hits[0].ref == "speedrun/mt-moon"

    stub.route(
        "GET",
        "/guide?ref=speedrun%2Fmt-moon",
        {"guide": "speedrun", "slug": "mt-moon", "title": "Mt. Moon", "body": "Go up."},
    )
    section = poke.guide.read("speedrun/mt-moon")
    assert section.title == "Mt. Moon"
    assert str(section) == "Go up."


def test_game_lookups_are_objects(stub, poke):
    stub.route(
        "GET",
        "/gamedata/trainers?map=Pewter+Gym",
        {
            "map": "Pewter Gym",
            "count": 1,
            "trainers": [{"class": "Brock", "at": [4, 1], "team": ["Geodude L12", "Onix L14"]}],
        },
    )
    brock = poke.game.trainers("Pewter Gym")[0]
    assert brock.trainer_class == "Brock"
    assert brock.at == (4, 1)
    assert brock.team == ["Geodude L12", "Onix L14"]

    stub.route(
        "GET",
        "/gamedata/species?name=Charmeleon",
        {
            "name": "Charmeleon",
            "dex": 5,
            "types": ["Fire"],
            "base": {"hp": 58, "atk": 64},
            "learnset": [[1, "Scratch"], [9, "Ember"], [24, "Rage"]],
            "tm_hm_count": 24,
        },
    )
    species = poke.game.species("Charmeleon")
    assert species.types == ["Fire"]
    assert species.learns_by(10) == ["Scratch", "Ember"]

    stub.route(
        "GET",
        "/gamedata/move?name=Ember",
        {
            "name": "Ember",
            "type": "Fire",
            "power": 40,
            "accuracy": 100,
            "pp": 25,
            "damage_class": "special",
        },
    )
    assert poke.game.move("Ember").damage_class == "special"

    stub.route(
        "GET",
        "/gamedata/encounters?map=Route+3",
        {
            "map": "Route 3",
            "grass": {
                "rate": 20,
                "levels": [3, 8],
                "species": [{"species": "Pidgey", "levels": [6, 8], "chance": 0.449}],
            },
            "water": None,
        },
    )
    encounters = poke.game.encounters("Route 3")
    assert encounters.grass.rate == 20
    assert encounters.grass.species[0].species == "Pidgey"
    assert encounters.water is None

    stub.route(
        "GET",
        "/gamedata/items?map=Viridian+Forest",
        {
            "map": "Viridian Forest",
            "count": 1,
            "items": [{"item": "Potion", "at": [1, 18], "hidden": True}],
        },
    )
    assert poke.game.items("Viridian Forest")[0].hidden is True

    stub.route(
        "GET",
        "/gamedata/shops?map=Pewter+Mart",
        {
            "map": "Pewter Mart",
            "items": ["Potion", "Poke Ball"],
            "prices": {"Potion": 300, "Poke Ball": 200},
        },
    )
    # Priced, so "can I afford to stock up" is answerable before walking there.
    assert poke.game.shops("Pewter Mart") == {"Potion": 300, "Poke Ball": 200}

    stub.route(
        "GET",
        "/gamedata/types?name=Water&against=Rock%2CGround",
        {"type": "Water", "against": ["Rock", "Ground"], "multiplier": 4.0},
    )
    assert poke.game.effectiveness("Water", ["Rock", "Ground"]) == 4.0


def test_a_missing_species_is_a_404_with_a_suggestion(stub, poke):
    stub.route(
        "GET",
        "/gamedata/species?name=Charmander2",
        {"detail": "No Pokemon called 'Charmander2'. Did you mean: Charmander?"},
        status=404,
    )

    with pytest.raises(agent_api.ServerError, match="Did you mean: Charmander"):
        poke.game.species("Charmander2")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def test_module_functions_follow_connect(stub):
    stub.route("GET", "/state", STATE_PAYLOAD)
    stub.route("GET", "/guide", {"outline": "everything"})
    previous = agent_api._default
    try:
        agent_api.connect(port=stub.port)
        assert agent_api.state().map == "Pewter City"
        assert agent_api.guide.outline() == "everything"
    finally:
        agent_api._default = previous
        agent_api.guide = agent_api.client().guide
        agent_api.game = agent_api.client().game


def test_default_port_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    assert agent_api.Client().url == "http://localhost:9999"
    monkeypatch.delenv("PORT")
    assert agent_api.Client().url == f"http://localhost:{agent_api.DEFAULT_PORT}"
    assert agent_api.Client(url="http://box:1234/").url == "http://box:1234"
