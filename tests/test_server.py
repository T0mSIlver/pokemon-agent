"""HTTP surface tests: the lean /action contract plus the dashboard endpoints."""

import asyncio
import io
import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pokemon_agent.navigation import LiveNavigationSnapshot

ACTION_KEYS = {
    "actions_executed",
    "map",
    "x",
    "y",
    "facing",
    "moves",
    # How far each legal direction goes. `moves` alone was answered by stepping
    # one tile and asking again, at a measured median of one tile per act call.
    "run",
    "mode",
    "dialog",
    "battle",
    "hp",
}

#: Facts about what the batch did. Nothing that tells the agent where to go.
BATCH_KEYS = {"moved", "blocked_after", "here_before"}

DELETED_ENDPOINTS = [
    ("GET", "/agent/observe"),
    ("POST", "/agent/observe"),
    ("POST", "/agent/plan"),
    ("POST", "/agent/act"),
    ("GET", "/agent/navigator"),
    ("GET", "/minimap"),
    ("GET", "/navigation/map"),
    ("POST", "/navigation/path"),
    ("POST", "/navigation/navigate"),
]

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class FakeEmulator:
    """Enough of the emulator contract for the server to drive a fake overworld."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.x = 5
        self.y = 6
        self.facing = "down"
        self.map_name = "PALLET TOWN"
        self.map_id = 0
        # (map, x, y) -> (map, map_id, x, y). A door the fake overworld honours.
        self.transitions: dict[tuple, tuple] = {}
        # Where walking off the top of the map lands, if anywhere.
        self.north_map: tuple | None = None
        self.dialog_active = False
        self.interaction = None
        self.warps = []
        self.warp_exit_directions: list[str] = []
        # Gen 1 arms a warp only when the player walks onto it, so a fake that
        # was placed on one starts disarmed, exactly like a loaded save state.
        self.warp_armed = False
        self.in_battle = False
        self.battle_after_steps = None  # trip a wild encounter mid-walk
        self.enemy = None
        #: Indices into wEventFlags that read as set, for the milestone ladder.
        self.event_bits: set[int] = set()
        self.pressed: list[str] = []
        self.walls: set[tuple[int, int]] = set()
        # Battle menus, modelled on the real ones: the 2x2 top menu does not wrap,
        # the move list does, and the move cursor remembers where it was left.
        self.battle_menu = "top"
        self.top_row = 0
        self.top_column = 0
        self.move_cursor = 0
        self.selected_move_id = 0
        self.battle_moves = [
            {"id": 10, "name": "Scratch", "pp": 30},
            {"id": 45, "name": "Growl", "pp": 40},
            {"id": 52, "name": "Ember", "pp": 25},
        ]
        self.turn_pending = False
        self.fled = False

    def get_screen(self) -> Image.Image:
        # Vary the pixels with the frame counter so refreshed PNGs differ.
        return Image.new("RGB", (160, 144), color=(self.frame_count % 251, 24, 32))

    def press(self, button: str, frames: int = 1) -> None:
        self.pressed.append(button)
        if self.in_battle:
            self._press_in_battle(button)
            self.frame_count += frames
            return
        delta = DIRECTIONS.get(button)
        if delta is not None:
            self.facing = button
            target = (self.x + delta[0], self.y + delta[1])
            # A blocked step in Gen 1 is not an error: you simply stay put.
            if target not in self.walls:
                self.x, self.y = target
                self._apply_transition()
                if self.battle_after_steps is not None:
                    self.battle_after_steps -= 1
                    if self.battle_after_steps <= 0:
                        self.in_battle = True
                        self.battle_after_steps = None
        self.frame_count += frames

    def _apply_transition(self) -> None:
        """Doors and map edges, as far as the fake overworld models them."""
        landing = self.transitions.get((self.map_name, self.x, self.y))
        if landing is not None:
            self.map_name, self.map_id, self.x, self.y = landing
            return
        if self.y >= 0:
            return
        if self.north_map is None:
            self.y = 0  # the edge of a map with nothing beyond it is a wall
            return
        self.map_name, self.map_id, self.y = self.north_map

    def _press_in_battle(self, button: str) -> None:
        if self.battle_menu == "top":
            if button == "up":
                self.top_row = 0  # the top menu does not wrap
            elif button == "down":
                self.top_row = 1
            elif button == "left":
                self.top_column = 0
            elif button == "right":
                self.top_column = 1
            elif button == "a":
                self._confirm_top_menu()
            return
        if self.battle_menu == "moves":
            count = len(self.battle_moves)
            if button == "down":
                self.move_cursor = (self.move_cursor + 1) % count  # ... but this one does
            elif button == "up":
                self.move_cursor = (self.move_cursor - 1) % count
            elif button == "b":
                self.battle_menu = "top"
            elif button == "a":
                self.battle_menu = "other"
                self.turn_pending = True
            self.selected_move_id = self.battle_moves[self.move_cursor]["id"]
            return
        if button in ("a", "b"):
            self.battle_menu = "top"

    def _confirm_top_menu(self) -> None:
        entry = (("FIGHT", "PKMN"), ("ITEM", "RUN"))[self.top_row][self.top_column]
        if entry == "FIGHT":
            self.battle_menu = "moves"
            self.selected_move_id = self.battle_moves[self.move_cursor]["id"]
        elif entry == "RUN":
            self.in_battle = False
            self.fled = True
            self.battle_menu = "other"
        else:
            self.battle_menu = "other"

    def tick(self, frames: int = 1) -> None:
        if self.turn_pending:
            # The turn resolves and the game hands the menu back for the next one.
            self.turn_pending = False
            self.battle_menu = "top"
            self.top_row = self.top_column = 0
        self.frame_count += frames

    def save_state(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"x": self.x, "y": self.y, "facing": self.facing}),
            encoding="utf-8",
        )

    def load_state(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.x = payload["x"]
        self.y = payload["y"]
        self.facing = payload["facing"]

    def get_navigation_snapshot(self, reader) -> LiveNavigationSnapshot:
        return LiveNavigationSnapshot(
            map_id=self.map_id,
            map_name=self.map_name,
            player_position=(self.x, self.y),
            facing=self.facing,
            tileset="OVERWORLD",
            window_top_left=(self.x - 4, self.y - 4),
            terrain=[
                [0 if (self.x - 4 + lx, self.y - 4 + ly) in self.walls else 1 for lx in range(10)]
                for ly in range(9)
            ],
            sprite_positions=[],
            valid_moves=["up", "left"],
            warps=self.warps,
            signs=[],
            map_dimensions={"width": 20, "height": 18},
            interaction=self.interaction,
            warp_exit_directions=list(self.warp_exit_directions),
            warp_exit_armed=self.warp_armed,
        )


class FakeReader:
    """Memory reader shim that reads straight off the fake emulator."""

    game_name = "Pokemon Red"

    def __init__(self, emulator: FakeEmulator) -> None:
        self.emulator = emulator

    def read_player(self) -> dict:
        return {
            "name": "RED",
            "rival_name": "BLUE",
            "position": {"x": self.emulator.x, "y": self.emulator.y},
            "facing": self.emulator.facing,
            "money": 3000,
            "badge_count": 0,
            "badges": [],
            "play_time": "0:10",
        }

    def read_party(self) -> list[dict]:
        return [
            {
                "nickname": "Bulbasaur",
                "species": "Bulbasaur",
                "level": 8,
                "hp": 20,
                "max_hp": 22,
                "status": "OK",
                "types": ["Grass", "Poison"],
                "moves": [{"name": "Tackle", "pp": 35}],
            }
        ]

    def read_bag(self) -> list[dict]:
        return []

    def read_battle(self) -> dict:
        return {
            "in_battle": self.emulator.in_battle,
            "type": "none",
            "enemy": self.emulator.enemy,
        }

    def read_bits(self, addr: int, size: int) -> list[bool]:
        return [index in self.emulator.event_bits for index in range(size * 8)]

    def read_battle_moves(self) -> list[dict]:
        return list(self.emulator.battle_moves)

    def at_battle_top_menu(self) -> bool:
        return self.emulator.in_battle and self.emulator.battle_menu == "top"

    def at_battle_move_menu(self) -> bool:
        return self.emulator.in_battle and self.emulator.battle_menu == "moves"

    def remembered_move_index(self) -> int:
        return self.emulator.move_cursor

    def selected_move_id(self) -> int:
        return self.emulator.selected_move_id

    def read_battle_menu(self) -> dict:
        if self.at_battle_top_menu():
            entry = (("FIGHT", "PKMN"), ("ITEM", "RUN"))[self.emulator.top_row][
                self.emulator.top_column
            ]
            return {"menu": "top", "highlighted": entry, "index": None}
        if self.at_battle_move_menu():
            index = self.emulator.move_cursor
            return {
                "menu": "moves",
                "highlighted": self.emulator.battle_moves[index]["name"],
                "index": index,
            }
        return {"menu": "other", "highlighted": None, "index": None}

    def read_dialog(self) -> dict:
        return {
            "active": self.emulator.dialog_active,
            "waiting_for_input": self.emulator.dialog_active,
            "printing": False,
        }

    def read_map_info(self) -> dict:
        return {"map_id": self.emulator.map_id, "map_name": self.emulator.map_name}

    def read_flags(self) -> dict:
        return {
            "has_pokedex": False,
            "has_oaks_parcel": False,
            "badge_count": 0,
            "badges": [],
            "pokedex_owned": 0,
            "pokedex_seen": 0,
        }


@contextmanager
def running_server(tmp_path, monkeypatch, emulator):
    """One server lifespan over `tmp_path`; run it twice to simulate a restart."""
    import pokemon_agent.emulator as emulator_mod
    import pokemon_agent.memory.red as red_mod
    from pokemon_agent import server

    monkeypatch.setattr(emulator_mod, "create_emulator", lambda rom_path: emulator)
    monkeypatch.setattr(red_mod, "PokemonRedReader", FakeReader)
    # The action rate limiter is module state; a whole test session of batches
    # would trip it long before any single test misbehaved.
    server._action_call_times.clear()

    rom = tmp_path / "game.gb"
    rom.write_bytes(b"\x00" * 32)
    workspace_dir = tmp_path / "workspace"
    server.configure(
        server.GameConfig(
            rom_path=str(rom),
            data_dir=str(tmp_path / "data"),
            agent_workspace_dir=str(workspace_dir),
            realtime=False,
        )
    )

    with TestClient(server.app) as http:
        yield SimpleNamespace(
            http=http,
            emulator=emulator,
            workspace_dir=workspace_dir,
            data_dir=tmp_path / "data",
            saves_dir=tmp_path / "data" / "saves",
        )


@pytest.fixture()
def server_app(tmp_path, monkeypatch):
    with running_server(tmp_path, monkeypatch, FakeEmulator()) as app:
        yield app


#: A one-tile-wide dead-end corridor running north to south.
CORRIDOR = {(5, y) for y in range(2, 11)}


@pytest.fixture()
def corridor_app(tmp_path, monkeypatch):
    """A map that is walled from the very first frame, so the store learns it.

    Walling the emulator after startup would be too late: the map has already
    recorded that open ground as passable, and passability never un-learns.
    """
    emulator = FakeEmulator()
    emulator.walls = {(x, y) for x in range(20) for y in range(18) if (x, y) not in CORRIDOR}
    with running_server(tmp_path, monkeypatch, emulator) as app:
        yield app


def test_action_returns_a_tiny_payload(server_app):
    response = server_app.http.post("/action", json={"actions": ["walk_up", "walk_left"]})

    assert response.status_code == 200
    assert len(response.content) < 300, response.content
    payload = response.json()
    assert set(payload) == ACTION_KEYS | {"moved"}
    assert payload == {
        "actions_executed": 2,
        "moved": 2,
        "map": "PALLET TOWN",
        "x": 4,
        "y": 5,
        "facing": "left",
        "moves": ["up", "left"],
        "run": {"up": 4, "left": 4},
        "mode": "overworld",
        "dialog": False,
        "battle": False,
        "hp": "20/22",
    }


def test_a_dialog_it_cannot_read_says_so_once_not_twice(server_app):
    # `classify_ui_state` emits a fixed "Dialog box visible (...)" placeholder
    # whenever a box is open, because nothing here reads the words -- the agent
    # is told to look at latest_frame.png for those. All 660 payloads that ever
    # carried `screen_text` carried that placeholder: 36,300 bytes restating the
    # `dialog` flag sitting next to it.
    server_app.emulator.dialog_active = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["mode"] == "dialog"
    assert payload["dialog"] is True
    assert "screen_text" not in payload
    assert set(payload) == ACTION_KEYS


def test_action_refreshes_both_frame_pngs(server_app):
    raw = server_app.workspace_dir / "latest_frame.png"
    annotated = server_app.workspace_dir / "latest_frame_annotated.png"
    before_raw = raw.read_bytes()
    before_annotated = annotated.read_bytes()

    server_app.http.post("/action", json={"actions": ["walk_down"]})

    assert raw.read_bytes() != before_raw
    assert annotated.read_bytes() != before_annotated


def test_action_rejects_an_unknown_action(server_app):
    response = server_app.http.post("/action", json={"actions": ["teleport_to_gym"]})

    assert response.status_code == 400


def test_state_returns_the_full_game_state(server_app):
    state = server_app.http.get("/state").json()

    assert state["map"]["map_name"] == "PALLET TOWN"
    assert state["player"]["position"] == {"x": 5, "y": 6}
    assert state["dialog_active"] is False
    assert state["metadata"]["game"] == "Pokemon Red"


def test_health_reports_readiness(server_app):
    payload = server_app.http.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["emulator_ready"] is True
    assert payload["agent_workspace_ready"] is True
    assert payload["emulation"]["realtime_enabled"] is False


def test_index_reports_the_workspace(server_app):
    payload = server_app.http.get("/").json()

    assert payload["name"] == "pokemon-agent"
    assert payload["game"] == "red"
    assert payload["agent_workspace_dir"] == str(server_app.workspace_dir)


def test_save_load_and_list_round_trip(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})
    saved = server_app.http.post("/save", json={"name": "checkpoint"})
    assert saved.status_code == 200
    assert saved.json()["save"]["name"] == "checkpoint"
    assert (server_app.saves_dir / "checkpoint.state").exists()

    moved = server_app.http.post("/action", json={"actions": ["walk_down", "walk_down"]}).json()
    assert moved["y"] == 7

    loaded = server_app.http.post("/load", json={"name": "checkpoint"})
    assert loaded.status_code == 200
    assert loaded.json()["save"]["name"] == "checkpoint"
    assert (loaded.json()["x"], loaded.json()["y"]) == (5, 5)

    names = [entry["name"] for entry in server_app.http.get("/saves").json()["saves"]]
    assert "checkpoint" in names

    assert server_app.http.post("/load", json={"name": "no_such_save"}).status_code == 404


def test_screenshot_endpoints(server_app):
    png = server_app.http.get("/screenshot")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"

    encoded = server_app.http.get("/screenshot/base64").json()
    assert encoded["format"] == "png"
    assert encoded["image"]


def test_artifacts_serve_the_surviving_keys(server_app):
    for key in ("latest_frame", "latest_frame_annotated", "latest_map"):
        response = server_app.http.get(f"/artifacts/{key}")
        assert response.status_code == 200, key
        assert response.headers["content-type"] == "image/png"

    context = server_app.http.get("/artifacts/turn_context_json")
    assert context.status_code == 200
    assert set(json.loads(context.content)) == {"observation_id", "objective", "position", "ui"}

    for gone in ("turn_plan_json", "recovery_saves_json", "working_memory_md"):
        assert server_app.http.get(f"/artifacts/{gone}").status_code == 404, gone


def test_dashboard_state_keeps_the_keys_app_js_reads(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})
    payload = server_app.http.get("/dashboard/state").json()

    assert {
        "generated_at",
        "visuals",
        "agent_intent",
        "world_state",
        "memory_and_progress",
        "timeline",
        "artifacts",
        "artifact_urls",
        "pi_supervisor",
        "server_runtime",
    }.issubset(payload)
    assert {
        "raw_frame_path",
        "annotated_frame_path",
        "frame_timestamp",
        "ui_mode",
        "screen_text",
    }.issubset(payload["visuals"])
    assert {
        "objective",
        "turn_context",
        "recent_action",
        "movement_guidance",
        "dialog_guidance",
        "battle_guidance",
        "state_delta",
    }.issubset(payload["agent_intent"])
    assert {
        "map",
        "player",
        "party",
        "battle",
        "dialog",
        "interaction",
        "valid_moves",
        "live_ascii",
        "navigation",
    }.issubset(payload["world_state"])
    assert {"progress_percent", "stuck", "workspace"}.issubset(payload["memory_and_progress"])
    assert {"realtime_enabled", "realtime_fps", "live_artifact_fps"}.issubset(
        payload["server_runtime"]
    )
    assert payload["world_state"]["map"]["map_name"] == "PALLET TOWN"
    assert payload["artifact_urls"]["latest_frame_annotated"] == "/artifacts/latest_frame_annotated"
    # Sources for these were deleted with the planning scaffolding.
    assert "turn_plan" not in payload["agent_intent"]
    assert "plan_status" not in payload["agent_intent"]
    assert "recovery" not in payload["memory_and_progress"]


def test_dashboard_history_returns_recorded_events(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    events = server_app.http.get("/dashboard/history?limit=50").json()["events"]

    assert any(event.get("type") == "action" for event in events)
    assert any(event.get("type") == "action_result" for event in events)


def test_supervisor_state_is_reported(server_app):
    payload = server_app.http.get("/supervisor/state").json()

    assert payload["status"] == "idle"
    assert "available" in payload


def test_supervisor_snapshot_keeps_the_legacy_keys_and_adds_the_stream(server_app):
    payload = server_app.http.get("/dashboard/state").json()["pi_supervisor"]

    assert {
        "transcript",
        "recent_tools",
        "recent_events",
        "counts",
        "session_usage",
        "context_usage",
        "config",
        "status",
        "status_reason",
        "model_limits",
        "compaction",
        "stream",
    }.issubset(payload)
    assert payload["stream"] == []


def test_supervisor_steer_returns_the_recorded_entry(server_app, monkeypatch):
    from pokemon_agent import server

    sent = []

    async def fake_send(message):
        sent.append(message)
        return {
            "seq": 12,
            "ts": "2026-01-01T00:00:00Z",
            "text": message,
            "streaming_behavior": "steer",
        }

    monkeypatch.setattr(server._supervisor, "send_operator_message", fake_send)

    response = server_app.http.post("/supervisor/steer", json={"message": "Go north."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["entry"] == {
        "seq": 12,
        "ts": "2026-01-01T00:00:00Z",
        "text": "Go north.",
        "streaming_behavior": "steer",
    }
    assert payload["supervisor"]["status"] == "idle"
    assert sent == ["Go north."]


def test_supervisor_steer_is_409_without_a_live_session(server_app):
    response = server_app.http.post("/supervisor/steer", json={"message": "Go north."})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "idle" in detail
    assert "start or continue" in detail.lower()


@pytest.mark.parametrize("message", ["", "   "])
def test_supervisor_steer_rejects_empty_messages(server_app, message):
    response = server_app.http.post("/supervisor/steer", json={"message": message})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_supervisor_steer_rejects_an_over_long_message(server_app):
    from pokemon_agent.pi_supervisor import OPERATOR_MESSAGE_LIMIT

    response = server_app.http.post(
        "/supervisor/steer",
        json={"message": "x" * (OPERATOR_MESSAGE_LIMIT + 1)},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert str(OPERATOR_MESSAGE_LIMIT) in detail
    assert str(OPERATOR_MESSAGE_LIMIT + 1) in detail


def test_supervisor_steer_does_not_swallow_an_http_exception(server_app, monkeypatch):
    from fastapi import HTTPException

    from pokemon_agent import server

    async def fake_send(message):
        raise HTTPException(status_code=418, detail="Pi is a teapot")

    monkeypatch.setattr(server._supervisor, "send_operator_message", fake_send)

    response = server_app.http.post("/supervisor/steer", json={"message": "Go north."})

    assert response.status_code == 418
    assert response.json()["detail"] == "Pi is a teapot"


def test_supervisor_steer_reports_an_unexpected_failure_as_500(server_app, monkeypatch):
    from pokemon_agent import server

    async def fake_send(message):
        raise OSError("stdin is closed")

    monkeypatch.setattr(server._supervisor, "send_operator_message", fake_send)

    response = server_app.http.post("/supervisor/steer", json={"message": "Go north."})

    assert response.status_code == 500
    assert "stdin is closed" in response.json()["detail"]


def _seed_supervisor_stream(supervisor, frame_path):
    async def seed():
        await supervisor._push_stream_system("session start", text="Reach Viridian City.")
        await supervisor._push_stream_user("Reach Viridian City.")
        await supervisor._handle_event(
            {
                "type": "tool_execution_start",
                "toolCallId": "tool-read",
                "toolName": "read",
                "args": {"path": str(frame_path)},
            }
        )
        await supervisor._handle_event(
            {
                "type": "tool_execution_end",
                "toolCallId": "tool-read",
                "toolName": "read",
                "result": {"type": "image", "data": "..."},
                "isError": False,
            }
        )

    asyncio.run(seed())


def test_supervisor_stream_serves_the_ordered_log(server_app):
    from pokemon_agent import server

    server_app.http.post("/action", json={"actions": ["walk_up"]})
    frame = server_app.workspace_dir / "latest_frame_annotated.png"
    _seed_supervisor_stream(server._supervisor, frame)

    payload = server_app.http.get("/supervisor/stream").json()

    assert set(payload) == {"entries", "next_seq", "session_id"}
    assert [entry["seq"] for entry in payload["entries"]] == [1, 2, 3]
    assert [entry["kind"] for entry in payload["entries"]] == ["system", "user", "tool"]
    assert payload["next_seq"] == 3
    tool = payload["entries"][2]["tool"]
    assert payload["entries"][2]["state"] == "ok"
    assert tool["headline"] == "read latest_frame_annotated.png"
    assert tool["path"] == str(frame.resolve())
    assert tool["image_artifact"] == "latest_frame_annotated"
    assert tool["result_summary"].startswith("image ")


def test_supervisor_stream_pages_from_after(server_app):
    from pokemon_agent import server

    server_app.http.post("/action", json={"actions": ["walk_up"]})
    frame = server_app.workspace_dir / "latest_frame_annotated.png"
    _seed_supervisor_stream(server._supervisor, frame)

    page = server_app.http.get("/supervisor/stream?after=1&limit=1").json()
    assert [entry["seq"] for entry in page["entries"]] == [2]
    assert page["next_seq"] == 2

    tail = server_app.http.get(f"/supervisor/stream?after={page['next_seq']}").json()
    assert [entry["seq"] for entry in tail["entries"]] == [3]
    assert tail["next_seq"] == 3

    caught_up = server_app.http.get("/supervisor/stream?after=3&limit=99999").json()
    assert caught_up["entries"] == []
    assert caught_up["next_seq"] == 3


def test_dashboard_shell_is_served(server_app):
    assert server_app.http.get("/dashboard").status_code == 200


def test_websocket_greets_the_client(server_app):
    with server_app.http.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"
        ws.send_text("ping")
        assert ws.receive_json()["type"] == "pong"


@pytest.mark.parametrize(("method", "path"), DELETED_ENDPOINTS)
def test_deleted_endpoints_are_gone(server_app, method, path):
    response = server_app.http.request(method, path, json={})

    assert response.status_code == 404, f"{method} {path}"


def test_action_reports_faces_when_a_sign_or_object_is_in_front(server_app):
    server_app.emulator.interaction = {"kind": "sign"}

    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert payload["faces"] == "sign"


def test_action_omits_faces_for_scenery(server_app):
    server_app.emulator.interaction = {"kind": "background"}

    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert "faces" not in payload


def test_action_flags_standing_on_a_warp(server_app):
    emulator = server_app.emulator
    emulator.warps = [{"x": emulator.x, "y": emulator.y}]

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["on_warp"] is True


def test_action_reports_lead_pokemon_hp(server_app):
    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert payload["hp"] == "20/22"


def test_action_names_the_warp_destination_and_step_at_a_map_edge(server_app):
    """A boundary warp's exit tile renders as blocked, so the step must be spelled out."""
    emulator = server_app.emulator
    emulator.x, emulator.y = 5, 0  # north edge of the map
    emulator.warps = [{"x": 5, "y": 0, "warp_id": 0, "target_map_id": 13}]

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["on_warp"] is True
    # Placed on the tile rather than walked onto it, so the exit is known but
    # will not fire until the player steps off and back on.
    assert payload["warp"]["to"] == "Route 2"
    assert payload["warp"]["step"] == "up"
    assert payload["warp"]["armed"] is False
    assert "Step off" in payload["warp"]["note"]


def test_action_reports_an_armed_boundary_warp_without_a_caveat(server_app):
    emulator = server_app.emulator
    emulator.x, emulator.y = 5, 0
    emulator.warps = [{"x": 5, "y": 0, "warp_id": 0, "target_map_id": 13}]
    emulator.warp_exit_directions = ["up"]
    emulator.warp_armed = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["warp"] == {"to": "Route 2", "step": "up"}


def test_action_omits_the_step_for_an_interior_warp(server_app):
    emulator = server_app.emulator
    emulator.x, emulator.y = 5, 5  # nowhere near a boundary
    emulator.warps = [{"x": 5, "y": 5, "warp_id": 0, "target_map_id": 13}]

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["on_warp"] is True
    assert "step" not in payload["warp"]
    assert payload["warp"]["to"] == "Route 2"


def test_action_in_battle_reports_the_fight_not_the_overworld(server_app):
    """Position and walk directions are meaningless on a battle screen."""
    emulator = server_app.emulator
    emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["battle"] is True
    for dead in ("x", "y", "facing", "moves", "on_warp", "faces"):
        assert dead not in payload, f"{dead} should not be reported in battle"
    assert payload["hp"] == "20/22"


def test_a_until_dialog_end_is_refused_in_battle(server_app):
    """The battle menu counts as an open dialog, so A-mashing opens the bag."""
    emulator = server_app.emulator
    emulator.in_battle = True

    response = server_app.http.post("/action", json={"actions": ["a_until_dialog_end"]})

    assert response.status_code == 400
    assert "unsafe in battle" in response.json()["detail"]


def test_a_until_dialog_end_is_allowed_outside_battle(server_app):
    response = server_app.http.post("/action", json={"actions": ["a_until_dialog_end"]})

    assert response.status_code == 200


def test_map_summarises_the_current_map_and_points_at_a_picture(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    payload = server_app.http.get("/map").json()

    assert payload["map_id"] == 0
    assert payload["map_name"] == "PALLET TOWN"
    assert (payload["width"], payload["height"]) == (20, 18)
    assert payload["player"] == {"x": 5, "y": 5}
    assert payload["warps"] == []
    assert payload["coverage"]["total"] == 360
    assert payload["coverage"]["walked"] == 2  # the startup tile and the one walked to
    assert payload["image"] == "/artifacts/latest_map"
    assert payload["image_path"] == str(server_app.workspace_dir / "latest_map.png")
    # The picture carries the shape; the payload carries none of it.
    assert "ascii" not in payload
    assert "scale" not in payload


def test_the_frame_overlay_is_wired_to_the_walked_tiles(server_app):
    """agent_runtime shades walked ground through this seam; only the server can join it."""
    from pokemon_agent import server

    server_app.http.post("/action", json={"actions": ["walk_up"]})

    assert server._runtime.visited_lookup is not None
    assert server._runtime.visited_lookup(0) == {(5, 6), (5, 5)}


def test_map_404s_for_a_map_id_that_was_never_visited(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    response = server_app.http.get("/map", params={"map_id": 99})

    assert response.status_code == 404
    assert "never been visited" in response.json()["detail"]


def test_map_serves_the_picture_it_points_at(server_app):
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    payload = server_app.http.get("/map").json()
    response = server_app.http.get(payload["image"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "PNG"
        # 20x18 of Pallet Town, drawn big enough to read at a glance.
        assert 200 <= max(image.size) <= 400
    assert Path(payload["image_path"]).read_bytes() == response.content


def test_the_map_picture_is_redrawn_as_the_agent_walks(server_app):
    server_app.http.get("/map")
    before = (server_app.workspace_dir / "latest_map.png").read_bytes()

    server_app.http.post("/action", json={"actions": ["walk_up", "walk_left"]})
    server_app.http.get("/map")

    assert (server_app.workspace_dir / "latest_map.png").read_bytes() != before


def test_the_mini_map_grid_seam_is_wired(server_app):
    """agent_runtime insets a mini-map drawn from these sets; only the server can join it."""
    from pokemon_agent import server

    server_app.http.post("/action", json={"actions": ["walk_up"]})

    grid = server._runtime.map_grid_lookup(0)
    assert grid["walked"] == {(5, 6), (5, 5)}
    assert (grid["width"], grid["height"]) == (20, 18)
    assert server._runtime.map_grid_lookup(99) is None


def test_action_does_not_gain_a_map_field(server_app):
    """/map is a pull tool — the action response stays as small as it was."""
    first = server_app.http.post("/action", json={"actions": ["walk_up"]})
    server_app.http.get("/map")
    after_reading_the_map = server_app.http.post("/action", json={"actions": ["walk_up"]})

    for probe in (first, after_reading_the_map):
        payload = probe.json()
        assert set(payload) <= ACTION_KEYS | BATCH_KEYS
        assert payload["map"] == "PALLET TOWN"  # the map's name, never a grid
        for leaked in ("ascii", "legend", "coverage", "image", "scale", "width"):
            assert leaked not in payload, leaked
        assert len(probe.content) < 300, probe.content


def test_the_explored_map_survives_a_server_restart(tmp_path, monkeypatch):
    with running_server(tmp_path, monkeypatch, FakeEmulator()) as app:
        app.http.post("/action", json={"actions": ["walk_up", "walk_up"]})
        before = app.http.get("/map").json()["coverage"]
    assert (tmp_path / "data" / "explored_maps.json").exists()

    restarted = FakeEmulator()
    restarted.x, restarted.y = 12, 12
    with running_server(tmp_path, monkeypatch, restarted) as app:
        after = app.http.get("/map").json()["coverage"]

    assert after["walked"] == before["walked"] + 1
    assert after["seen"] > before["seen"] > 90  # more than any one window could show


# ---------------------------------------------------------------------------
# Batch outcome: what the buttons actually achieved
# ---------------------------------------------------------------------------


def test_a_batch_rammed_into_a_wall_reports_that_it_went_nowhere(server_app):
    """walk_up x16 into a tree: fifteen of those buttons did nothing at all."""
    server_app.emulator.walls = {(5, 5)}

    payload = server_app.http.post("/action", json={"actions": ["walk_up"] * 16}).json()

    assert payload["moved"] == 0
    assert payload["blocked_after"] == 1
    assert (payload["x"], payload["y"]) == (5, 6)


def test_a_clear_batch_reports_the_distance_it_covered(server_app):
    payload = server_app.http.post("/action", json={"actions": ["walk_left"] * 4}).json()

    assert payload["moved"] == 4
    assert "blocked_after" not in payload
    assert (payload["x"], payload["y"]) == (1, 6)


def test_a_batch_blocked_partway_reports_where_it_stopped(server_app):
    server_app.emulator.walls = {(5, 3)}

    payload = server_app.http.post("/action", json={"actions": ["walk_up"] * 5}).json()

    assert payload["moved"] == 2  # (5, 6) -> (5, 5) -> (5, 4), then the tree
    assert payload["blocked_after"] == 3
    assert (payload["x"], payload["y"]) == (5, 4)


def test_a_move_blocked_at_the_very_end_wasted_nothing(server_app):
    """blocked_after means 'the rest was wasted'. Nothing followed, so stay quiet."""
    server_app.emulator.walls = {(5, 5)}

    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert payload["moved"] == 0
    assert "blocked_after" not in payload


def test_a_batch_that_never_walks_reports_neither_field(server_app):
    payload = server_app.http.post("/action", json={"actions": ["press_a", "wait_60"]}).json()

    assert "moved" not in payload
    assert "blocked_after" not in payload


def test_a_there_and_back_batch_is_not_mistaken_for_a_blocked_one(server_app):
    """Net displacement is zero, but nothing was blocked — both steps counted."""
    payload = server_app.http.post("/action", json={"actions": ["walk_right", "walk_left"]}).json()

    assert payload["moved"] == 2
    assert "blocked_after" not in payload
    assert (payload["x"], payload["y"]) == (5, 6)


def test_a_batch_in_battle_reports_no_movement(server_app):
    server_app.emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["walk_down"]}).json()

    assert payload["battle"] is True
    for dead in ("moved", "blocked_after", "here_before"):
        assert dead not in payload, dead


# ---------------------------------------------------------------------------
# Revisits
# ---------------------------------------------------------------------------


def step(server_app, direction):
    return server_app.http.post("/action", json={"actions": [f"walk_{direction}"]}).json()


def test_here_before_appears_only_once_a_tile_becomes_a_habit(server_app):
    # The threshold is 8, not 3. At 3 this field was sent 2,339 times across a
    # run and referred to again in the next command exactly zero times -- it
    # reached 49 on one tile and changed nothing -- so it now waits for a loop
    # rather than firing on any corridor walked back down.
    payload = {}
    for i in range(server.HERE_BEFORE_THRESHOLD * 2):
        payload = step(server_app, "up" if i % 2 == 0 else "down")
        if "here_before" in payload:
            break

    assert payload["here_before"] >= server.HERE_BEFORE_THRESHOLD - 1


def test_a_tile_walked_a_couple_of_times_is_not_worth_saying(server_app):
    # A corridor you walk back down is not a loop.
    for _ in range(2):
        assert "here_before" not in step(server_app, "up")
        assert "here_before" not in step(server_app, "down")


def test_standing_still_does_not_inflate_the_revisit_count(server_app):
    for _ in range(5):
        payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()
        assert "here_before" not in payload

    assert server_app.http.get("/map").json()["coverage"]["walked"] == 1


def walk(app, direction, count=1):
    return app.http.post("/action", json={"actions": [f"walk_{direction}"] * count}).json()


def descend_the_corridor(app):
    """Step to the dead end one action at a time, so every tile is recorded walked."""
    for _ in range(4):
        payload = step(app, "down")  # (5, 7), (5, 8), (5, 9), (5, 10)
    return payload


def test_action_never_tells_the_agent_where_to_go(corridor_app):
    """The harness gives perception, not directions; it must not name a heading."""
    payload = descend_the_corridor(corridor_app)

    assert (payload["x"], payload["y"]) == (5, 10)
    assert "unexplored" not in payload
    for value in payload.values():
        assert not isinstance(value, str) or "north" not in value


def test_the_action_payload_stays_small_with_every_field_present(corridor_app):
    descend_the_corridor(corridor_app)
    # Pace it until it is a rut. `here_before` waits for a real loop now, so this
    # walks the corridor rather than assuming a fixed number of passes.
    # An odd number of passes, so it finishes back at the top and the descent
    # below is the same six-step walk the assertions describe.
    for i in range(server.HERE_BEFORE_THRESHOLD * 2 + 1):
        walk(corridor_app, "up" if i % 2 == 0 else "down", 4)

    response = corridor_app.http.post("/action", json={"actions": ["walk_down"] * 6})
    payload = response.json()

    assert payload["moved"] == 4
    assert payload["blocked_after"] == 5  # the corridor ends; steps 5 and 6 were wasted
    assert "here_before" in payload, "every field must be present for this to mean anything"
    assert len(response.content) < 300, response.content


# ---------------------------------------------------------------------------
# Battle commands
# ---------------------------------------------------------------------------


def test_fight_keys_step_down_from_wherever_the_cursor_opens():
    """The move list wraps, so Down alone reaches every entry and the count is
    the only thing that can be wrong."""
    from pokemon_agent import server

    normalise = ["press_up", "press_up", "press_left", "press_left"]

    assert server.battle_fight_keys(0, 0, 4) == [*normalise, "press_a", "press_a"]
    assert server.battle_fight_keys(2, 0, 4) == [
        *normalise,
        "press_a",
        "press_down",
        "press_down",
        "press_a",
    ]
    # Wrapping the short way round: from Leer to Scratch is one Down, not three Ups.
    assert server.battle_fight_keys(0, 3, 4) == [*normalise, "press_a", "press_down", "press_a"]
    assert server.battle_fight_keys(1, 2, 3) == [
        *normalise,
        "press_a",
        "press_down",
        "press_down",
        "press_a",
    ]
    # A one-move Pokemon never needs a step.
    assert server.battle_fight_keys(0, 0, 1) == [*normalise, "press_a", "press_a"]


def test_fight_keys_refuse_an_index_that_is_not_a_move():
    from pokemon_agent import server

    with pytest.raises(ValueError):
        server.battle_fight_keys(3, 0, 3)
    with pytest.raises(ValueError):
        server.battle_fight_keys(-1, 0, 3)
    with pytest.raises(ValueError):
        server.battle_fight_keys(0, 0, 0)


def test_run_keys_reach_run_from_any_top_menu_entry():
    from pokemon_agent import server

    assert server.battle_run_keys() == [
        "press_down",
        "press_down",
        "press_right",
        "press_right",
        "press_a",
    ]


def test_resolve_move_name_takes_case_and_unique_prefixes():
    from fastapi import HTTPException

    from pokemon_agent import server

    moves = [{"name": "Scratch"}, {"name": "Growl"}, {"name": "Ember"}, {"name": "Gust"}]

    assert server.resolve_move_name("ember", moves) == 2
    assert server.resolve_move_name("EMBER", moves) == 2
    assert server.resolve_move_name("emb", moves) == 2
    with pytest.raises(HTTPException) as ambiguous:
        server.resolve_move_name("g", moves)
    assert "Growl" in ambiguous.value.detail and "Gust" in ambiguous.value.detail


def in_battle(server_app, *, cursor=0):
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.battle_menu = "top"
    emulator.move_cursor = cursor
    return emulator


def test_fight_selects_the_named_move_whatever_the_cursor_was_on(server_app):
    emulator = in_battle(server_app, cursor=2)  # cursor left on Ember last turn
    emulator.top_row, emulator.top_column = 1, 1  # ...and parked on RUN

    payload = server_app.http.post("/battle/fight", json={"move": "Scratch"}).json()

    assert payload["used"] == "Scratch"
    assert emulator.selected_move_id == 10
    assert emulator.in_battle is True  # RUN was never confirmed


def test_fight_accepts_a_unique_prefix(server_app):
    emulator = in_battle(server_app)

    payload = server_app.http.post("/battle/fight", json={"move": "emb"}).json()

    assert payload["used"] == "Ember"
    assert emulator.selected_move_id == 52


def test_fight_reports_the_resulting_state_without_a_second_call(server_app):
    in_battle(server_app)

    payload = server_app.http.post("/battle/fight", json={"move": "ember"}).json()

    assert payload["used"] == "Ember"
    assert payload["battle"] is True
    assert payload["hp"] == "20/22"
    assert payload["menu"] == "top"  # the turn resolved and it is our move again


def test_fight_lists_the_real_moves_when_the_name_is_wrong(server_app):
    in_battle(server_app)

    response = server_app.http.post("/battle/fight", json={"move": "Hyper Beam"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Scratch, Growl, Ember" in detail
    assert "Hyper Beam" in detail


def test_fight_refuses_a_move_with_no_pp_and_says_what_is_left(server_app):
    emulator = in_battle(server_app)
    emulator.battle_moves[2]["pp"] = 0

    response = server_app.http.post("/battle/fight", json={"move": "ember"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "no PP left" in detail
    assert "Scratch 30PP" in detail
    assert emulator.selected_move_id == 0  # nothing was pressed


def test_fight_refuses_when_there_is_no_battle(server_app):
    response = server_app.http.post("/battle/fight", json={"move": "ember"})

    assert response.status_code == 400
    assert "Not in a battle" in response.json()["detail"]


def test_fight_gives_up_loudly_when_the_battle_menu_never_appears(server_app):
    emulator = in_battle(server_app)
    emulator.battle_menu = "other"
    # A menu that ignores B is a screen this command cannot drive.
    emulator._press_in_battle = lambda button: None

    response = server_app.http.post("/battle/fight", json={"move": "ember"})

    assert response.status_code == 409
    assert "never appeared" in response.json()["detail"]


def test_run_flees_from_wherever_the_cursor_sits(server_app):
    emulator = in_battle(server_app)
    emulator.battle_menu = "moves"  # cursor abandoned inside the move list

    payload = server_app.http.post("/battle/run").json()

    assert payload["fled"] is True
    assert emulator.fled is True
    assert payload["battle"] is False


def test_run_refuses_when_there_is_no_battle(server_app):
    response = server_app.http.post("/battle/run")

    assert response.status_code == 400
    assert "Not in a battle" in response.json()["detail"]


def test_action_in_battle_reports_which_menu_is_open(server_app):
    """Pressing A blindly fires whatever is highlighted, so say what that is."""
    emulator = in_battle(server_app, cursor=2)
    emulator.battle_menu = "moves"

    payload = server_app.http.post("/action", json={"actions": ["wait_10"]}).json()

    assert payload["menu"] == "moves"
    assert payload["highlighted"] == "Ember"


def test_menu_fields_are_absent_outside_battle(server_app):
    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert "menu" not in payload
    assert "highlighted" not in payload


# ---------------------------------------------------------------------------
# Save names
#
# A save name arrives from the network. Appended straight to the saves
# directory it is not a name at all: `../escaped` used to report success and
# write outside it.
# ---------------------------------------------------------------------------

ESCAPING_NAMES = ["../escaped", "../../escaped", "/tmp/absolute", "a/b", "..", ".", "", "..\\evil"]


@pytest.mark.parametrize("name", ESCAPING_NAMES)
def test_the_model_refuses_a_save_name_that_is_not_a_plain_file_name(name):
    from pydantic import ValidationError

    from pokemon_agent import server

    with pytest.raises(ValidationError):
        server.SaveRequest(name=name)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/absolute", "a/b"])
def test_save_refuses_to_write_outside_the_saves_directory(server_app, name):
    before = {p.name for p in server_app.saves_dir.glob("*.state")}

    response = server_app.http.post("/save", json={"name": name})

    assert response.status_code >= 400
    assert not (server_app.data_dir / "escaped.state").exists()
    assert not (server_app.data_dir.parent / "escaped.state").exists()
    assert {p.name for p in server_app.saves_dir.glob("*.state")} == before


@pytest.mark.parametrize("name", ["../escaped", "/etc/passwd", "a/b"])
def test_load_refuses_to_read_outside_the_saves_directory(server_app, name):
    outside = server_app.data_dir / "escaped.state"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("{}", encoding="utf-8")

    response = server_app.http.post("/load", json={"name": name})

    assert response.status_code >= 400


def test_a_legal_save_name_still_round_trips(server_app):
    name = "auto__20250101T000000Z__map_transition"
    saved = server_app.http.post("/save", json={"name": name})

    assert saved.status_code == 200
    assert (server_app.saves_dir / f"{name}.state").exists()
    assert server_app.http.post("/load", json={"name": name}).status_code == 200


def test_saves_listing_offers_only_names_load_would_accept(server_app):
    server_app.http.post("/save", json={"name": "good"})
    (server_app.saves_dir / "not a name.state").write_text("{}", encoding="utf-8")

    names = [entry["name"] for entry in server_app.http.get("/saves").json()["saves"]]

    assert "good" in names
    assert "not a name" not in names


def test_the_save_path_helper_is_the_only_way_a_path_is_built(tmp_path):
    """Startup auto-load, save, load and listing all resolve through one helper."""
    from pokemon_agent.saves import SaveNameError, resolve_save_path

    saves = tmp_path / "saves"
    saves.mkdir()
    assert resolve_save_path(saves, "ok").parent == saves.resolve()
    for name in ESCAPING_NAMES:
        with pytest.raises(SaveNameError):
            resolve_save_path(saves, name)


# ---------------------------------------------------------------------------
# Action caps
#
# One action used to be able to hold the emulator for hundreds of days.
# ---------------------------------------------------------------------------


def test_the_caps_are_the_numbers_the_cli_mirrors():
    from pokemon_agent import coordinator

    assert coordinator.MAX_ACTIONS_PER_BATCH == 40
    assert coordinator.MAX_FRAMES_PER_ACTION == 600
    assert coordinator.MAX_FRAMES_PER_BATCH == 3600


def test_one_action_may_not_run_the_emulator_forever(server_app):
    response = server_app.http.post("/action", json={"actions": ["wait_1000000000"]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "600" in detail and "1000000000" in detail


def test_a_held_button_is_capped_too(server_app):
    response = server_app.http.post("/action", json={"actions": ["hold_up_100000"]})

    assert response.status_code == 400
    assert "600" in response.json()["detail"]


def test_a_batch_may_not_hold_more_actions_than_the_cap(server_app):
    response = server_app.http.post("/action", json={"actions": ["press_a"] * 41})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "40" in detail and "41" in detail


def test_a_batch_may_not_spend_more_frames_than_the_cap(server_app):
    # Each of these is legal on its own; together they are a minute of emulation.
    response = server_app.http.post("/action", json={"actions": ["wait_600"] * 10})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "3600" in detail and "6000" in detail


def test_an_over_cap_batch_presses_nothing_at_all(server_app):
    before = list(server_app.emulator.pressed)

    server_app.http.post("/action", json={"actions": ["walk_up", "wait_1000000000"]})

    assert server_app.emulator.pressed == before


def test_a_batch_at_the_cap_is_allowed(server_app):
    response = server_app.http.post("/action", json={"actions": ["wait_600"] * 6})

    assert response.status_code == 200


def test_an_unknown_action_is_refused_before_the_batch_starts(server_app):
    before = list(server_app.emulator.pressed)

    response = server_app.http.post("/action", json={"actions": ["walk_up", "teleport"]})

    assert response.status_code == 400
    assert server_app.emulator.pressed == before


# ---------------------------------------------------------------------------
# Transactions
#
# The lock used to serialise calls rather than operations, so a second request
# could mutate the emulator between the first one's mutation and its read.
# ---------------------------------------------------------------------------


async def test_two_concurrent_loads_each_report_their_own_save(tmp_path, monkeypatch):
    """A response naming save A must not carry the map and position of save B."""
    import threading

    from pokemon_agent import server

    saves = tmp_path / "saves"
    saves.mkdir()
    for name in ("A", "B"):
        (saves / f"{name}.state").touch()

    class BlockingEmulator:
        def __init__(self) -> None:
            self.current = None
            self.a_started = threading.Event()
            self.release_a = threading.Event()

        def load_state(self, path):
            name = Path(path).stem
            self.current = name
            if name == "A":  # hold A inside its transaction and let B queue up
                self.a_started.set()
                self.release_a.wait(5)

    emulator = BlockingEmulator()
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_reader", object())
    monkeypatch.setattr(server, "_runtime", None)
    monkeypatch.setattr(server, "_config", SimpleNamespace(data_dir=str(tmp_path)))
    monkeypatch.setattr(server, "_emulator_lock", asyncio.Lock())

    def refresh(**kwargs):
        x = 1 if emulator.current == "A" else 2
        state = {
            "map": {"map_name": emulator.current},
            "player": {"position": {"x": x, "y": 0}},
            "battle": {},
        }
        return {"events": [], "bundle": {"state": state, "navigation": {}, "screen_text": {}}}

    monkeypatch.setattr(server, "_refresh_agent_bundle_sync", refresh)

    task_a = asyncio.create_task(server.load_state(server.SaveRequest(name="A")))
    await asyncio.to_thread(emulator.a_started.wait, 5)
    task_b = asyncio.create_task(server.load_state(server.SaveRequest(name="B")))
    await asyncio.sleep(0.05)
    emulator.release_a.set()
    response_a, response_b = await asyncio.gather(task_a, task_b)

    assert (response_a["save"]["name"], response_a["map"], response_a["x"]) == ("A", "A", 1)
    assert (response_b["save"]["name"], response_b["map"], response_b["x"]) == ("B", "B", 2)


class RecordingOps:
    """The coordinator's plumbing, reduced to an ordered log of what it did."""

    def __init__(self, gate=None) -> None:
        self.lock = asyncio.Lock()
        self.log: list[str] = []
        self.gate = gate
        self.emulator = SimpleNamespace(load_state=self._load, save_state=lambda path: None)
        self.reader = object()
        self.runtime = None
        self.settles = True

    def _load(self, path) -> None:
        self.log.append("load")

    def settle(self, **kwargs) -> bool:
        self.log.append("settle")
        return self.settles

    def state_dict(self) -> dict:
        self.log.append("state")
        return {}

    def reject_unsafe_battle_actions(self, actions) -> None:
        self.log.append("safety")

    def execute_batch(self, actions) -> dict:
        self.log.append(f"execute:{actions[0]}")
        if self.gate is not None and actions[0] == "first":
            self.gate.wait(5)
        return {"executed": len(actions), "moved": None, "blocked_after": None}

    def refresh_bundle(self, **kwargs) -> dict:
        label = (kwargs.get("requested_actions") or ["-"])[0]
        self.log.append(f"refresh:{label}")
        return {"events": [], "bundle": {"state": {"who": label}}}


async def test_an_action_transaction_never_interleaves_with_another():
    """The whole batch-then-observe sequence happens before the next one starts."""
    import threading

    from pokemon_agent.coordinator import EmulatorCoordinator

    gate = threading.Event()
    ops = RecordingOps(gate=gate)
    coordinator = EmulatorCoordinator(ops)

    first = asyncio.create_task(
        coordinator.act_and_observe(["first"], source="action", reason="test")
    )
    await asyncio.sleep(0.05)
    second = asyncio.create_task(
        coordinator.act_and_observe(["second"], source="action", reason="test")
    )
    await asyncio.sleep(0.05)
    gate.set()
    await asyncio.gather(first, second)

    assert ops.log.index("refresh:first") < ops.log.index("execute:second")
    assert ops.log.index("execute:first") < ops.log.index("refresh:first")


async def test_a_load_settles_before_it_observes():
    from pokemon_agent.coordinator import EmulatorCoordinator

    ops = RecordingOps()
    ops.emulator.settle = ops.settle

    await EmulatorCoordinator(ops).load_settle_and_observe(path="x.state", reason="test")

    assert ops.log[:3] == ["load", "settle", "refresh:-"]


async def test_an_unsettled_load_refreshes_nothing():
    """A transition frame that never comes to rest is not worth publishing."""
    from pokemon_agent.coordinator import EmulatorCoordinator

    ops = RecordingOps()
    ops.emulator.settle = ops.settle
    ops.settles = False

    result = await EmulatorCoordinator(ops).load_settle_and_observe(path="x.state", reason="test")

    assert result["settled"] is False
    assert "refresh:-" not in ops.log


class SettlingEmulator(FakeEmulator):
    """A fake that grew the settle() the emulator is being given."""

    def __init__(self, settles: bool = True) -> None:
        super().__init__()
        self.settles = settles
        self.settle_calls: list[tuple] = []
        self.order: list[str] = []

    def load_state(self, path: str) -> None:
        self.order.append("load")
        super().load_state(path)

    def settle(self, *, max_frames: int = 600, quiet_frames: int = 30) -> bool:
        self.order.append("settle")
        self.settle_calls.append((max_frames, quiet_frames))
        return self.settles


def test_load_settles_the_game_before_reporting_it(tmp_path, monkeypatch):
    emulator = SettlingEmulator()
    with running_server(tmp_path, monkeypatch, emulator) as app:
        app.http.post("/save", json={"name": "here"})

        payload = app.http.post("/load", json={"name": "here"}).json()

    assert emulator.settle_calls == [(600, 30)]
    assert emulator.order[:2] == ["load", "settle"]
    assert payload["success"] is True
    assert "settled" not in payload  # a settled load says nothing about settling


def test_an_unsettled_load_publishes_nothing(tmp_path, monkeypatch):
    """Recording a mid-transition frame corrupts the explored map permanently."""
    emulator = SettlingEmulator(settles=False)
    with running_server(tmp_path, monkeypatch, emulator) as app:
        app.http.post("/save", json={"name": "here"})
        annotated = app.workspace_dir / "latest_frame_annotated.png"
        before_frame = annotated.read_bytes()
        before_saves = {p.name for p in app.saves_dir.glob("*.state")}

        payload = app.http.post("/load", json={"name": "here"}).json()

        assert payload["settled"] is False
        assert annotated.read_bytes() == before_frame
        assert {p.name for p in app.saves_dir.glob("*.state")} == before_saves


# ---------------------------------------------------------------------------
# Unsupported games
# ---------------------------------------------------------------------------


def test_a_gba_rom_is_refused_by_name_not_by_import_error():
    from pokemon_agent import server

    with pytest.raises(server.UnsupportedGameError) as excinfo:
        server._resolve_game_type("game.gba", "auto")

    assert "FireRed" in str(excinfo.value)


def test_starting_on_a_gba_rom_creates_no_emulator(tmp_path, monkeypatch):
    import pokemon_agent.emulator as emulator_mod
    from pokemon_agent import server

    def explode(rom_path):
        raise AssertionError("the emulator must not be created for an unsupported game")

    monkeypatch.setattr(emulator_mod, "create_emulator", explode)
    rom = tmp_path / "game.gba"
    rom.write_bytes(b"\x00" * 32)
    server.configure(
        server.GameConfig(
            rom_path=str(rom),
            data_dir=str(tmp_path / "data"),
            agent_workspace_dir=str(tmp_path / "workspace"),
            realtime=False,
        )
    )

    with TestClient(server.app) as http:
        health = http.get("/health").json()
        action = http.post("/action", json={"actions": ["press_a"]})

    assert health["emulator_ready"] is False
    assert "FireRed" in health["startup_error"]
    assert action.status_code == 503


# ---------------------------------------------------------------------------
# GET /route
# ---------------------------------------------------------------------------


def test_route_returns_hops_from_the_current_map(server_app):
    payload = server_app.http.get("/route", params={"to": "Viridian City"}).json()

    assert payload["from"] == "Pallet Town"
    assert payload["to"] == "Viridian City"
    assert payload["distance"] == len(payload["hops"]) == 2
    assert payload["hops"][0] == {
        "from": "Pallet Town",
        "to": "Route 1",
        "kind": "connection",
        "at": None,
        "edge": "north",
    }


def test_route_to_a_warp_names_the_tile_to_step_on(server_app):
    payload = server_app.http.get("/route", params={"to": "Oak's Lab"}).json()

    hop = payload["hops"][-1]
    assert hop["kind"] == "warp"
    assert hop["at"] == [12, 11]


def test_route_to_where_you_already_are_is_no_hops_at_all(server_app):
    payload = server_app.http.get("/route", params={"to": "Pallet Town"}).json()

    assert payload["hops"] == []
    assert payload["distance"] == 0


def test_route_404s_for_a_map_nobody_has_heard_of(server_app):
    response = server_app.http.get("/route", params={"to": "Kanto Airport"})

    assert response.status_code == 404
    assert "Kanto Airport" in response.json()["detail"]


def test_route_says_why_when_there_is_no_way_through():
    """An unreachable destination answers with null hops and a reason."""
    from pokemon_agent.capabilities import route_payload
    from pokemon_agent.world import MapInfo, World

    world = World(
        {
            "Island": MapInfo(name="Island", map_id=1, size=None, hops=()),
            "Mainland": MapInfo(name="Mainland", map_id=2, size=None, hops=()),
        }
    )

    payload = route_payload(world, "Island", "Mainland")

    assert payload["hops"] is None
    assert "No route" in payload["reason"]


# ---------------------------------------------------------------------------
# POST /goto
# ---------------------------------------------------------------------------


def test_goto_walks_to_a_tile_on_the_current_map(server_app):
    payload = server_app.http.post("/goto", json={"x": 5, "y": 2}).json()

    assert payload["arrived"] is True
    assert payload["walked"] == 4
    assert (payload["x"], payload["y"]) == (5, 2)
    assert payload["stopped_because"] == "arrived"


def test_goto_a_tile_it_is_already_on_walks_nothing(server_app):
    payload = server_app.http.post("/goto", json={"x": 5, "y": 6}).json()

    assert payload["arrived"] is True
    assert payload["walked"] == 0
    assert payload["actions_executed"] == 0


def test_goto_looks_at_both_ends_of_the_corridor_before_calling_it_walled_off(corridor_app):
    """The refusal comes after the looking, and it only comes once.

    (12, 6) is seven tiles east of a one-wide corridor, through solid rock. The
    corridor's two ends run out of the live window, so at the start there IS
    unseen ground — just none of it toward the goal. The old planner scored
    every frontier by "does this step shrink Manhattan distance to the goal",
    threw away everything scoring zero or less, and so refused on the first
    call without moving. Correct here by luck, and by construction unable to
    take the first step of any maze route that starts by going the wrong way.

    So the walk now goes and looks: north end, south end, then a refusal that
    is a fact rather than a guess. Twelve presses to settle a nine-tile
    corridor, and — this is the part that matters — it terminates. Each call
    ends somewhere new, and the refusal still names what IS reachable.
    """
    first = corridor_app.http.post("/goto", json={"x": 12, "y": 6}).json()
    assert first["arrived"] is False
    assert (first["x"], first["y"]) == (5, 2), "walked to the north end to look past it"
    assert "edge of what has been seen and not a wall" in first["stopped_because"]

    second = corridor_app.http.post("/goto", json={"x": 12, "y": 6}).json()
    assert second["arrived"] is False
    assert (second["x"], second["y"]) == (5, 10), "and then the south end"

    third = corridor_app.http.post("/goto", json={"x": 12, "y": 6}).json()
    assert third["arrived"] is False
    assert third["walked"] == 0, "nothing left to look at, so nothing left to walk"
    assert "no walkable path" in third["stopped_because"]
    assert third["onward"]["kind"] == "walled-off"


def test_goto_crosses_a_map_edge_toward_a_named_map(tmp_path, monkeypatch):
    emulator = FakeEmulator()
    emulator.y = 4  # close enough that the north edge is inside the live window
    emulator.north_map = ("ROUTE 1", 12, 17)
    with running_server(tmp_path, monkeypatch, emulator) as app:
        payload = app.http.post("/goto", json={"target": "Route 1"}).json()

    assert payload["arrived"] is True
    assert payload["stopped_because"] == "arrived"
    assert payload["map"] == "ROUTE 1"


def test_goto_404s_for_a_map_nobody_has_heard_of(server_app):
    response = server_app.http.post("/goto", json={"target": "Kanto Airport"})

    assert response.status_code == 404


def test_goto_refuses_an_empty_target(server_app):
    assert server_app.http.post("/goto", json={}).status_code == 400
    assert server_app.http.post("/goto", json={"x": 3}).status_code == 400
    assert server_app.http.post("/goto", json={"target": "Route 1", "x": 3}).status_code == 400


# ---------------------------------------------------------------------------
# GET /calc
# ---------------------------------------------------------------------------


PIDGEY = {
    "species": "Pidgey",
    "level": 5,
    "hp": 19,
    "max_hp": 19,
    "types": ["Normal", "Flying"],
    "moves": ["Tackle", "Gust"],
}


def test_calc_is_409_outside_a_battle(server_app):
    response = server_app.http.get("/calc")

    assert response.status_code == 409
    assert "battle" in response.json()["detail"]


def test_calc_reports_a_damage_range_and_a_kill_count_per_move(server_app):
    server_app.emulator.in_battle = True
    server_app.emulator.enemy = PIDGEY

    payload = server_app.http.get("/calc").json()

    by_name = {entry["move"]: entry for entry in payload["moves"]}
    assert set(by_name) == {"Scratch", "Growl", "Ember"}
    ember = by_name["Ember"]
    assert ember["type"] == "Fire"
    assert ember["power"] == 40
    assert ember["effectiveness"] == 1.0
    assert 0 < ember["damage"][0] <= ember["damage"][1]
    assert ember["turns_to_ko"] >= 1
    # A status move does no damage and never kills anything.
    assert by_name["Growl"]["damage"] == [0, 0]
    assert by_name["Growl"]["turns_to_ko"] is None


def test_calc_names_the_enemy_and_the_worst_it_can_do(server_app):
    server_app.emulator.in_battle = True
    server_app.emulator.enemy = PIDGEY

    payload = server_app.http.get("/calc").json()

    assert payload["enemy"] == {
        "species": "Pidgey",
        "level": 5,
        "hp": 19,
        "types": ["Normal", "Flying"],
    }
    assert payload["threat"] > 0


# ---------------------------------------------------------------------------
# GET /frontier
# ---------------------------------------------------------------------------


def test_frontier_lists_reachable_ground_never_stood_on(corridor_app):
    payload = corridor_app.http.get("/frontier").json()

    assert payload["map"] == "PALLET TOWN"
    assert payload["from"] == [5, 6]
    assert payload["count"] == len(payload["tiles"])
    assert [5, 6] not in payload["tiles"]  # you have been here
    assert [5, 5] in payload["tiles"] and [5, 7] in payload["tiles"]
    # The corridor is one tile wide, so nothing beside it is reachable.
    assert all(tile[0] == 5 for tile in payload["tiles"])
    # Nearest first.
    first = payload["tiles"][0]
    assert abs(first[1] - 6) == 1


def test_frontier_shrinks_as_the_ground_is_walked(corridor_app):
    before = corridor_app.http.get("/frontier").json()["count"]

    corridor_app.http.post("/action", json={"actions": ["walk_up", "walk_up"]})

    assert corridor_app.http.get("/frontier").json()["count"] < before


# ---------------------------------------------------------------------------
# POST /sim
# ---------------------------------------------------------------------------


def test_sim_reports_where_a_plan_would_stop(corridor_app):
    payload = corridor_app.http.post("/sim", json={"actions": ["up:6", "right:3"]}).json()

    assert payload["end"] == [5, 2]  # the corridor runs out at y=2
    assert payload["steps"] == 4
    assert payload["blocked_at"] == 4  # index into the *expanded* action list
    assert payload["blocked_by"] == "wall"
    assert payload["facing"] == "up"
    assert payload["warp_at"] is None


def test_sim_presses_nothing(corridor_app):
    before = (corridor_app.emulator.x, corridor_app.emulator.y)
    pressed = list(corridor_app.emulator.pressed)

    corridor_app.http.post("/sim", json={"actions": ["up:6"]})

    assert (corridor_app.emulator.x, corridor_app.emulator.y) == before
    assert corridor_app.emulator.pressed == pressed


def test_sim_refuses_a_plan_it_cannot_read(server_app):
    response = server_app.http.post("/sim", json={"actions": ["teleport"]})

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /guide
# ---------------------------------------------------------------------------


def test_guide_serves_the_outline_by_default(server_app):
    payload = server_app.http.get("/guide").json()

    assert set(payload) == {"outline"}
    assert "read(guide, slug)" in payload["outline"]


def test_guide_serves_one_section_by_reference(server_app):
    from pokemon_agent import guides

    section = guides.index()[0]

    payload = server_app.http.get("/guide", params={"ref": section.ref}).json()

    assert payload["guide"] == section.guide
    assert payload["slug"] == section.slug
    assert payload["title"] == section.title
    assert payload["body"].strip()


def test_reading_a_section_is_recorded_against_the_map_and_the_press_count(server_app):
    from pokemon_agent import guides
    from pokemon_agent import server as server_mod

    section = guides.index()[0]
    server_app.http.post("/action", json={"actions": ["press_a", "press_a"]})

    server_app.http.get("/guide", params={"ref": section.ref})

    reads = server_mod._guide_log.reads()
    assert reads[-1]["guide"] == section.guide
    assert reads[-1]["slug"] == section.slug
    assert reads[-1]["at_map"] == "PALLET TOWN"
    assert reads[-1]["presses"] == 2


def test_guide_404s_for_a_section_that_does_not_exist(server_app):
    assert server_app.http.get("/guide", params={"ref": "nope/nope"}).status_code == 404
    assert server_app.http.get("/guide", params={"ref": "nonsense"}).status_code == 404


def test_guide_searches_by_keyword(server_app):
    payload = server_app.http.get("/guide", params={"q": "brock"}).json()

    assert payload["results"]
    assert set(payload["results"][0]) == {"ref", "title", "summary"}


def test_guide_search_records_nothing(server_app):
    from pokemon_agent import server as server_mod

    before = len(server_mod._guide_log.reads())

    server_app.http.get("/guide", params={"q": "brock"})
    server_app.http.get("/guide")

    assert len(server_mod._guide_log.reads()) == before


# ---------------------------------------------------------------------------
# GET /progress
# ---------------------------------------------------------------------------


def test_progress_counts_milestones_and_buttons(server_app):
    from pokemon_agent.milestones import ALL_EVENTS, MILESTONES

    starter = MILESTONES[0]
    server_app.emulator.event_bits = {ALL_EVENTS[starter.source]}
    server_app.http.post("/action", json={"actions": ["press_a", "walk_up"]})

    payload = server_app.http.get("/progress").json()

    assert payload["count"] == 1
    assert payload["total"] == len(MILESTONES)
    assert payload["furthest"] == starter.id
    assert payload["furthest_label"] == starter.label
    assert payload["latest"] == [starter.id]
    assert payload["presses"] == 2


def test_progress_on_a_fresh_game_is_honest_about_zero(server_app):
    payload = server_app.http.get("/progress").json()

    assert payload == {
        "count": 0,
        "total": payload["total"],
        "furthest": None,
        "furthest_label": None,
        "latest": [],
        "presses": 0,
        # A game where nothing has happened has exactly one thing it can do.
        "frontier": [
            {
                "id": "EVENT_GOT_STARTER",
                "label": "Chose a starter Pokemon",
                "gives": ["a Pokemon of your own"],
            }
        ],
        # The run ledger is always present and empty until a run opens, so the
        # dashboard never has to guess whether the server has an opinion.
        "run_id": None,
        "presses_to": {},
        "attainments": [],
    }


# ---------------------------------------------------------------------------
# Dashboard mounting
#
# The shell and its assets are registered by the dashboard package now, not
# open-coded here as well. A live run is watched through these URLs.
# ---------------------------------------------------------------------------

DASHBOARD_URLS = [
    "/dashboard",
    "/dashboard/",
    "/dashboard/assets/app.js",
    "/dashboard/assets/style.css",
    "/dashboard/state",
    "/dashboard/history",
]


@pytest.mark.parametrize("url", DASHBOARD_URLS)
def test_every_dashboard_url_still_resolves(server_app, url):
    assert server_app.http.get(url).status_code == 200


def test_the_dashboard_shell_is_never_cached(server_app):
    """The shell names its assets with a ?v= token, so a cached shell is stale."""
    response = server_app.http.get("/dashboard")

    assert "no-store" in response.headers["cache-control"]
    assert response.headers["content-type"].startswith("text/html")


def test_the_dashboard_is_mounted_exactly_once(server_app):
    """Two mounting paths could drift; a restarted lifespan must not add a third."""
    from pokemon_agent import server as server_mod

    paths = [getattr(route, "path", None) for route in server_mod.app.router.routes]

    assert paths.count("/dashboard") == 1
    assert paths.count("/dashboard/") == 1
    assert paths.count("/dashboard/assets") == 1


# ---------------------------------------------------------------------------
# Receipts
#
# The scoreboard in pokemon_agent/bench has never been written to by anything.
# These are the tests that it is now: one receipt per batch, appended after the
# emulator lock is released, priced in the buttons the batch actually sent.
# ---------------------------------------------------------------------------


def on_server_loop(coro, timeout: float = 10.0):
    """Await a server coroutine from the test thread, on the server's own loop.

    The TestClient runs the app in a background thread and the emulator lock
    belongs to that loop, so a second loop here would be a different machine.
    """
    from pokemon_agent import server as server_mod

    return asyncio.run_coroutine_threadsafe(coro, server_mod._loop).result(timeout=timeout)


def open_run(goal: str = "Reach Pewter") -> str:
    from pokemon_agent import server as server_mod

    handle = on_server_loop(server_mod._run_recorder.begin_session(goal=goal, model="fake-model"))
    return handle.run_id


def receipts(app, run_id: str) -> list[dict]:
    path = app.data_dir / "runs" / run_id / "receipts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def action_receipts(app, run_id: str) -> list[dict]:
    return [entry for entry in receipts(app, run_id) if entry["tool"] == "action"]


def test_a_run_starts_with_a_header_and_a_baseline_receipt(server_app):
    run_id = open_run()

    meta = json.loads(
        (server_app.data_dir / "runs" / run_id / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["goal"] == "Reach Pewter"
    assert meta["model"] == "fake-model"
    assert meta["status"] == "running"
    assert receipts(server_app, run_id)[0]["tool"] == "run_start"


def test_an_action_batch_leaves_one_receipt_with_every_field_filled(server_app):
    run_id = open_run()

    server_app.http.post("/action", json={"actions": ["walk_up", "walk_up", "press_a"]})

    written = action_receipts(server_app, run_id)
    assert len(written) == 1
    receipt = written[0]
    assert receipt["presses"] == 3
    assert receipt["map"] == "PALLET TOWN"
    assert receipt["pos"] == [5, 4]
    assert receipt["moved"] == 2
    assert receipt["blocked_after"] is None
    assert receipt["hp"] == [20, 22]
    assert receipt["party_size"] == 1
    assert receipt["milestones_new"] == []
    assert receipt["exit"] == 0
    assert receipt["reloaded"] is False
    assert receipt["whiteout"] is False
    assert receipt["t"] > 0


def test_a_batch_that_went_nowhere_says_so_in_its_receipt(server_app):
    run_id = open_run()
    server_app.emulator.walls = {(5, 5)}

    server_app.http.post("/action", json={"actions": ["walk_up", "walk_up"]})

    receipt = action_receipts(server_app, run_id)[-1]
    assert receipt["presses"] == 2
    assert receipt["moved"] == 0
    assert receipt["blocked_after"] == "1"


def test_a_refused_batch_still_leaves_a_receipt(server_app):
    """Two failures in a row is what the repeated_failure detector fires on."""
    run_id = open_run()

    assert server_app.http.post("/action", json={"actions": ["fly_north"]}).status_code == 400

    receipt = action_receipts(server_app, run_id)[-1]
    assert receipt["exit"] == 1
    assert receipt["presses"] == 0
    assert "fly_north" in receipt["error"]


def test_a_milestone_is_priced_at_the_presses_that_reached_it(server_app):
    from pokemon_agent.milestones import ALL_EVENTS, MILESTONES

    run_id = open_run()
    starter = MILESTONES[0]

    server_app.http.post("/action", json={"actions": ["walk_up", "walk_up"]})
    server_app.emulator.event_bits = {ALL_EVENTS[starter.source]}
    server_app.http.post("/action", json={"actions": ["press_a"]})
    server_app.http.post("/action", json={"actions": ["press_a"]})

    written = action_receipts(server_app, run_id)
    assert written[0]["milestones_new"] == []
    assert written[1]["milestones_new"] == [starter.id]
    assert written[2]["milestones_new"] == []  # first attainment only

    payload = server_app.http.get("/progress").json()
    assert payload["run_id"] == run_id
    assert payload["presses_to"] == {starter.id: 3}
    assert payload["attainments"][0]["presses"] == 3
    assert payload["attainments"][0]["label"] == starter.label


def test_a_battle_command_is_priced_like_any_other_batch(server_app):
    run_id = open_run()
    in_battle(server_app)

    server_app.http.post("/battle/run")

    battles = [entry for entry in receipts(server_app, run_id) if entry["tool"] == "battle"]
    assert len(battles) == 1
    assert battles[0]["presses"] > 0
    assert battles[0]["intent"] == {"run": True}


def test_a_reload_leaves_a_receipt_and_never_rewinds_the_bill(server_app):
    """A gym won on the fourth attempt costs what all four attempts cost."""
    run_id = open_run()
    server_app.http.post("/action", json={"actions": ["walk_up", "walk_up"]})
    server_app.http.post("/save", json={"name": "before-brock"})
    server_app.http.post("/action", json={"actions": ["walk_up", "walk_up", "walk_up"]})

    server_app.http.post("/load", json={"name": "before-brock"})
    server_app.http.post("/action", json={"actions": ["press_a"]})

    reloads = [entry for entry in receipts(server_app, run_id) if entry["reloaded"]]
    assert len(reloads) == 1
    assert reloads[0]["presses"] == 0
    assert reloads[0]["tool"] == "load"
    assert server_app.http.get("/progress").json()["presses"] == 6


def test_progress_reports_the_runs_total_and_not_this_processs(server_app):
    from pokemon_agent import server as server_mod

    run_id = open_run()
    # 900 presses this process never saw, from the sessions before it started.
    on_server_loop(server_mod._run_recorder.append(tool="action", presses=900, map_name="Route 3"))
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    payload = server_app.http.get("/progress").json()

    assert payload["presses"] == 901
    assert payload["run_id"] == run_id
    assert server_mod._press_count == 1


def test_progress_keeps_the_field_names_it_already_had(server_app):
    open_run()
    payload = server_app.http.get("/progress").json()

    assert {"count", "total", "furthest", "furthest_label", "latest", "presses"}.issubset(payload)
    assert {"run_id", "presses_to", "attainments"}.issubset(payload)


def test_health_reports_the_run_and_the_intervention_switch(server_app):
    payload = server_app.http.get("/health").json()

    assert payload["run"]["run_id"] is None
    assert payload["interventions"]["enabled"] is False
    assert payload["interventions"]["slot_lost"] is None

    run_id = open_run()
    assert server_app.http.get("/health").json()["run"]["run_id"] == run_id


# ---------------------------------------------------------------------------
# Interventions, through the server
# ---------------------------------------------------------------------------


def wait_for_intervention(timeout: float = 5.0) -> None:
    from pokemon_agent import server as server_mod

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = server_mod._intervention_task
        if task is not None and task.done():
            return
        time.sleep(0.02)
    raise AssertionError("the intervention loop never ran")


def arm_interventions(server_mod, steers: list, answer: str = "Walk left four tiles."):
    """Enable the loop with the model and the slot API faked out entirely."""
    from pokemon_agent.interventions import InterventionPolicy, RepeatedFailure

    runner = server_mod._interventions
    runner.enabled = True
    runner.policy = InterventionPolicy(detectors=(RepeatedFailure(),), cooldown_presses=0)
    runner.slot_client = None  # nothing here may touch the model box
    runner.advise = lambda prompt: answer

    async def deliver(text: str) -> None:
        steers.append(text)

    runner.deliver = deliver
    return runner


def test_the_flag_is_off_and_nothing_fires(server_app):
    from pokemon_agent import server as server_mod

    open_run()
    server_mod._interventions.advise = lambda prompt: pytest.fail("the model was asked to think")

    server_app.http.post("/action", json={"actions": ["fly_north"]})
    server_app.http.post("/action", json={"actions": ["fly_north"]})

    assert server_mod._intervention_task is None
    assert server_mod._interventions.status()["fired"] == 0


def test_a_trigger_steers_the_live_session_exactly_once(server_app):
    from pokemon_agent import server as server_mod

    open_run()
    steers: list[str] = []
    runner = arm_interventions(server_mod, steers)

    server_app.http.post("/action", json={"actions": ["fly_north"]})
    server_app.http.post("/action", json={"actions": ["fly_north"]})
    wait_for_intervention()

    assert steers == ["Walk left four tiles."]
    assert runner.status()["fired"] == 1
    assert runner.status()["delivered"] == 1


def test_a_lost_slot_disables_the_loop_and_shows_up_in_health(server_app):
    from pokemon_agent import server as server_mod
    from pokemon_agent.slots import SlotLost

    open_run()
    steers: list[str] = []
    runner = arm_interventions(server_mod, steers)

    def lose_the_slot(prompt: str) -> str:
        raise SlotLost("could not restore slot 0 from 'player.bin'", "player.bin")

    runner.advise = lose_the_slot

    server_app.http.post("/action", json={"actions": ["fly_north"]})
    server_app.http.post("/action", json={"actions": ["fly_north"]})
    wait_for_intervention()

    health = server_app.http.get("/health").json()["interventions"]
    assert health["slot_lost"]["filename"] == "player.bin"
    assert health["active"] is False
    assert health["disabled_reason"]
    assert steers == []


def test_the_environment_flag_turns_the_loop_on_at_startup(tmp_path, monkeypatch):
    from pokemon_agent import server as server_mod

    monkeypatch.setenv(server_mod.INTERVENTIONS_ENV_VAR, "1")
    with running_server(tmp_path, monkeypatch, FakeEmulator()) as app:
        assert app.http.get("/health").json()["interventions"]["enabled"] is True


def test_the_flag_defaults_to_off_with_no_environment_and_no_config(monkeypatch):
    from pokemon_agent import server as server_mod

    monkeypatch.delenv(server_mod.INTERVENTIONS_ENV_VAR, raising=False)

    assert server_mod._interventions_flag(None) is False
    assert server_mod._interventions_flag(server_mod.GameConfig(rom_path="x.gb")) is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_environment_flag_accepts_the_usual_spellings(monkeypatch, value):
    from pokemon_agent import server as server_mod

    monkeypatch.setenv(server_mod.INTERVENTIONS_ENV_VAR, value)

    assert server_mod._interventions_flag(None) is True


def test_the_config_field_wins_over_the_environment(monkeypatch):
    from pokemon_agent import server as server_mod

    monkeypatch.setenv(server_mod.INTERVENTIONS_ENV_VAR, "1")
    config = server_mod.GameConfig(rom_path="x.gb", interventions_enabled=False)

    assert server_mod._interventions_flag(config) is False


# ---------------------------------------------------------------------------
# `run`: how far each direction goes
#
# `moves` says which directions are legal, and the model answered it by stepping
# one tile and asking again -- 400 act calls at a median of one tile each over a
# real session. Tool text is 98% of the prompt, so each of those tiles is paid for
# twice. `run` turns "you may go left" into "you may go left seven".
# ---------------------------------------------------------------------------


from pokemon_agent import server  # noqa: E402


def _snapshot(terrain, player, origin=(0, 0), sprites=()):
    return {
        "terrain": terrain,
        "player_position": {"x": player[0], "y": player[1]},
        "window_top_left": {"x": origin[0], "y": origin[1]},
        "sprites": [{"x": x, "y": y} for x, y in sprites],
    }


OPEN_ROW = [[1, 1, 1, 1, 1, 1, 1]]


def test_run_counts_open_ground_in_each_direction():
    terrain = [[1] * 7 for _ in range(5)]
    runway = server._runway(_snapshot(terrain, player=(3, 2)))
    assert runway == {"up": 2, "down": 2, "left": 3, "right": 3}


def test_run_stops_at_a_wall():
    terrain = [[1, 1, 0, 1, 1]]
    runway = server._runway(_snapshot(terrain, player=(0, 0)))
    assert runway["right"] == 1


def test_run_omits_a_direction_with_nowhere_to_go():
    terrain = [[0, 1, 0]]
    runway = server._runway(_snapshot(terrain, player=(1, 0)))
    assert "left" not in runway and "right" not in runway


def test_run_stops_at_a_sprite_because_npcs_block():
    terrain = [[1, 1, 1, 1, 1]]
    runway = server._runway(_snapshot(terrain, player=(0, 0), sprites=[(3, 0)]))
    assert runway["right"] == 2


def test_run_respects_the_window_origin():
    terrain = [[1, 1, 1]]
    runway = server._runway(_snapshot(terrain, player=(21, 40), origin=(20, 40)))
    assert runway["right"] == 1


def test_run_never_counts_past_the_window():
    # Outside the live window the game has shown us nothing. Unknown is not
    # walkable, and guessing there is how a confident wrong answer gets made.
    terrain = [[1, 1, 1]]
    runway = server._runway(_snapshot(terrain, player=(2, 0)))
    assert "right" not in runway


def test_run_is_empty_without_a_snapshot():
    assert server._runway({}) == {}
    assert server._runway({"terrain": []}) == {}


def test_run_only_reports_directions_moves_already_calls_legal():
    # The raw terrain grid and get_valid_moves disagree on purpose: the latter
    # also applies the ledge and warp rules. A payload saying "you may not go
    # right" beside "right goes 5" is two answers to one question.
    terrain = [[1] * 7 for _ in range(5)]
    snapshot = _snapshot(terrain, player=(3, 2))
    snapshot["valid_moves"] = ["up", "left"]
    assert server._runway(snapshot) == {"up": 2, "left": 3}


# ---------------------------------------------------------------------------
# `exits`: which maps this one leads to
#
# The agent spent ten hours inside Mt. Moon and made two saves on B2F, which
# warps only back to B1F -- the optional fossil room, a dead end for progress --
# while the way out sat on B1F at (27,3). `poke route` answers that instantly and
# was called once in five hundred steps, which is the advisory pattern that has
# failed every time here.
#
# This states what exists, never which one to take. That stays the model's call,
# the same way `run` says how far each direction goes without saying which to walk.
# ---------------------------------------------------------------------------


def _warp_snapshot(map_name, player, warps):
    return {
        "map_name": map_name,
        "player_position": {"x": player[0], "y": player[1]},
        "warps": [{"coord": {"x": x, "y": y}, "target_map_id": target} for x, y, target in warps],
    }


def test_exits_names_each_destination_once_at_its_nearest_tile():
    # Two staircases to the same floor: the near one is the useful fact.
    snapshot = _warp_snapshot(
        "Unknown Map", player=(10, 10), warps=[(12, 10, 1), (30, 30, 1), (11, 12, 2)]
    )
    exits = server._exits(snapshot)
    assert len(exits) == 2
    assert list(exits.values())[0] == [12, 10]


def test_exits_are_ordered_nearest_first():
    snapshot = _warp_snapshot(
        "Unknown Map", player=(0, 0), warps=[(20, 0, 1), (2, 0, 2), (9, 0, 3)]
    )
    order = [coord[0] for coord in server._exits(snapshot).values()]
    assert order == sorted(order)


def test_exits_is_capped_so_a_hub_map_cannot_flood_the_payload():
    warps = [(i, 0, i + 1) for i in range(12)]
    snapshot = _warp_snapshot("Unknown Map", player=(0, 0), warps=warps)
    assert len(server._exits(snapshot)) <= server.MAX_EXITS


def test_exits_skips_warps_whose_destination_is_unknown():
    snapshot = _warp_snapshot("Unknown Map", player=(0, 0), warps=[(1, 0, 9999)])
    assert server._exits(snapshot) == {}


def test_exits_is_empty_without_a_position():
    assert server._exits({"map_name": "Unknown Map", "warps": []}) == {}
    assert server._exits({}) == {}


def test_exits_respects_a_byte_budget_not_just_a_count():
    # Four of the longest map names come to 171 bytes and would push the action
    # payload from 257 to 439. The payload is small so it can be read every turn.
    import json as _json

    from pokemon_agent import gamedata

    longest = sorted(
        {
            d
            for e in gamedata.world().values()
            for w in (e.get("warps") or [])
            if (d := w.get("to_map"))
        },
        key=len,
        reverse=True,
    )[:6]
    snapshot = {
        "map_name": "Unknown Map",
        "player_position": {"x": 0, "y": 0},
        "warps": [],
    }
    # Feed them through the snapshot fallback path with real names.
    snapshot["warps"] = [
        {"coord": {"x": i, "y": 0}, "target_map_id": None, "to_map": name}
        for i, name in enumerate(longest)
    ]
    exits = server._exits(snapshot)
    assert len(_json.dumps(exits, separators=(",", ":"))) <= server.MAX_EXITS_BYTES


def test_exits_names_edge_connections_not_only_warps():
    # Route 4 reaches Cerulean City -- the goal -- by walking off its east edge,
    # not through a door. The first version of `exits` read warps only, so it
    # listed three ways back into Mt. Moon and nothing about the way out, while
    # the run sat at x=19 of a 90-wide map for thousands of presses.
    snapshot = {
        "map_name": "Route 4",
        "player_position": {"x": 19, "y": 6},
        "warps": [],
    }
    exits = server._exits(snapshot)
    assert exits.get("Cerulean City") == "east edge"
    # And the warps are still there, after the edges.
    assert "Mt Moon 1F" in exits


def test_a_map_with_no_connections_still_lists_its_warps():
    snapshot = {
        "map_name": "Mt Moon B1F",
        "player_position": {"x": 22, "y": 8},
        "warps": [],
    }
    exits = server._exits(snapshot)
    assert exits.get("Route 4") == [27, 3]
    assert not any(isinstance(v, str) for v in exits.values())


# ---------------------------------------------------------------------------
# Payload weight
#
# Tool text is 98% of the model's prompt, so a fat response is paid once when it
# arrives and again on every turn afterwards. Two measured offenders: `/saves`
# returned 71,017 bytes for 465 files, and `/health` reached 21,688 bytes on one
# firing because it inlined a whole intervention answer.
# ---------------------------------------------------------------------------


def test_saves_is_capped_and_says_how_many_it_hid(server_app):
    saves_dir = server_app.saves_dir
    saves_dir.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        (saves_dir / f"auto__{i:03d}.state").write_bytes(b"x")

    payload = server_app.http.get("/saves").json()

    assert payload["shown"] <= server.DEFAULT_SAVES_LIMIT
    assert payload["count"] >= 60
    assert payload["truncated"] is True


def test_a_save_pruned_mid_listing_drops_out_instead_of_500ing_the_lot(server_app, monkeypatch):
    """The autosave pruner deletes files from a thread while this runs on the loop.

    Measured under load: 23 x `GET /saves -> 500`, each because one `auto__`
    file vanished between the glob and the stat. `poke saves` is how the agent
    finds anything to load, so losing the whole listing over one pruned file is
    expensive. The pruner already tolerates this race on its side.
    """
    saves_dir = server_app.saves_dir
    saves_dir.mkdir(parents=True, exist_ok=True)
    (saves_dir / "before_brock.state").write_bytes(b"x")
    doomed = saves_dir / "auto__001.state"
    doomed.write_bytes(b"x")

    real_stat = server.Path.stat

    def vanishing(self, *args, **kwargs):
        if self.name == doomed.name:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(server.Path, "stat", vanishing)
    response = server_app.http.get("/saves")

    assert response.status_code == 200, response.text
    names = {s["name"] for s in response.json()["saves"]}
    assert "before_brock" in names, "the surviving save is still listed"
    assert "auto__001" not in names


def test_saves_can_hide_the_harnesss_own_autosaves(server_app):
    saves_dir = server_app.saves_dir
    saves_dir.mkdir(parents=True, exist_ok=True)
    (saves_dir / "auto__001.state").write_bytes(b"x")
    (saves_dir / "before_brock.state").write_bytes(b"x")

    named = server_app.http.get("/saves?named=true").json()

    names = {s["name"] for s in named["saves"]}
    assert "before_brock" in names
    assert not any(n.startswith("auto__") for n in names)


def test_saves_returns_everything_when_asked(server_app):
    saves_dir = server_app.saves_dir
    saves_dir.mkdir(parents=True, exist_ok=True)
    for i in range(50):
        (saves_dir / f"auto__{i:03d}.state").write_bytes(b"x")

    payload = server_app.http.get("/saves?limit=0").json()

    assert payload["shown"] == payload["count"]
    assert "truncated" not in payload


def test_health_does_not_carry_a_whole_intervention_answer(server_app):
    body = server_app.http.get("/health").content
    assert len(body) < 4096, f"/health is {len(body)} bytes"


# ---------------------------------------------------------------------------
# Navigation refuses to answer from a battle frame
#
# A battle still produces a snapshot and it is not a map -- the window is the
# fight. Measured live, mid-Zubat: `goto` answered `sealed: true,
# reachable_tiles: 1` and `frontier` answered zero tiles. Telling an agent it is
# walled in when it is merely in a battle is worse than telling it nothing.
# ---------------------------------------------------------------------------


def test_navigation_refuses_during_a_battle_instead_of_answering_wrongly(server_app):
    server_app.emulator.in_battle = True

    for method, path, body in (
        ("GET", "/frontier", None),
        ("POST", "/sim", {"actions": ["walk_up"]}),
        ("POST", "/goto", {"x": 1, "y": 1}),
    ):
        response = server_app.http.request(method, path, json=body)
        assert response.status_code == 409, f"{path} answered {response.status_code}"
        assert "battle" in response.json()["detail"].lower()


def test_navigation_answers_again_once_the_battle_ends(server_app):
    server_app.emulator.in_battle = True
    assert server_app.http.get("/frontier").status_code == 409

    server_app.emulator.in_battle = False
    assert server_app.http.get("/frontier").status_code == 200


def test_a_walk_that_ends_in_an_encounter_says_so_rather_than_sealed(server_app):
    # Mt. Moon rolls a wild encounter roughly every ten steps, so most walks
    # longer than fifteen tiles finish in a battle. Everything computed after
    # that point is read off a battle frame, and the old answer was
    # `sealed: true, reachable_tiles: 1` about ground just walked across.
    server_app.emulator.battle_after_steps = 2

    response = server_app.http.post("/goto", json={"x": 5, "y": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["battle"] is True
    assert "wild Pokemon appeared" in payload["stopped_because"]
    assert "onward" not in payload, "a battle frame must not produce a reachability claim"
