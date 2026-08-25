"""HTTP surface tests: the lean /action contract plus the dashboard endpoints."""

import asyncio
import io
import json
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
        self.dialog_active = False
        self.interaction = None
        self.warps = []
        self.in_battle = False
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
        self.frame_count += frames

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
            map_id=0,
            map_name="PALLET TOWN",
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
        return {"in_battle": self.emulator.in_battle, "type": "none", "enemy": None}

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
        return {"map_id": 0, "map_name": "PALLET TOWN"}

    def read_flags(self) -> dict:
        return {
            "has_pokedex": False,
            "has_oaks_parcel": False,
            "badge_count": 0,
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
        "mode": "overworld",
        "dialog": False,
        "battle": False,
        "hp": "20/22",
    }


def test_action_reports_dialog_and_screen_text_only_when_present(server_app):
    server_app.emulator.dialog_active = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert set(payload) == ACTION_KEYS | {"screen_text"}
    assert payload["mode"] == "dialog"
    assert payload["dialog"] is True
    assert payload["screen_text"]


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
    assert payload["warp"] == {"to": "Route 2", "step": "up"}


def test_action_omits_the_step_for_an_interior_warp(server_app):
    emulator = server_app.emulator
    emulator.x, emulator.y = 5, 5  # nowhere near a boundary
    emulator.warps = [{"x": 5, "y": 5, "warp_id": 0, "target_map_id": 13}]

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["on_warp"] is True
    assert payload["warp"] == {"to": "Route 2"}


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
    # The startup refresh already counts as the first arrival at (5, 6).
    assert "here_before" not in step(server_app, "up")  # (5, 5), 1st
    assert "here_before" not in step(server_app, "down")  # (5, 6), 2nd
    assert "here_before" not in step(server_app, "up")  # (5, 5), 2nd

    third_time = step(server_app, "down")  # (5, 6), 3rd

    assert third_time["here_before"] == 2


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
    for direction in ("up", "down", "up"):  # pace the corridor until it is a rut
        walk(corridor_app, direction, 4)

    response = corridor_app.http.post("/action", json={"actions": ["walk_down"] * 6})
    payload = response.json()

    assert payload["moved"] == 4
    assert payload["blocked_after"] == 5  # the corridor ends; steps 5 and 6 were wasted
    assert payload["here_before"] == 2
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
