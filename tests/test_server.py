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

#: What comes back instead on a frame no step can be taken from — a battle or an
#: open box. The two walk fields go and one sentence says which frame took them,
#: because an empty `moves` list is indistinguishable from being walled in.
NO_STEP_KEYS = (ACTION_KEYS - {"moves", "run"}) | {"no_walk"}

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
        #: The words the real reader decodes off wTileMap. Blank on an overworld
        #: frame, exactly as `read_screen_text` answers there.
        self.screen_text = ""
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
        #: Whether RUN gets away. Gen 1 flees on a speed roll, and a Pokemon
        #: faster than yours never passes it: Route 6 (1,15) answered "could not
        #: get away" to 331 consecutive `poke run` calls.
        self.flee_succeeds = True
        #: The move that has taken the turn away, as `battle_lock_in` reports it.
        #: On the real game a Rage means the top menu never comes back.
        self.locked_in: str | None = None
        #: Whether the game comes to rest after a press. See `press_and_settle`.
        self.settles = True
        #: The two readings a whiteout changes and nothing else in this fake
        #: touches: the lead's HP goes to zero and the wallet is halved. Set them
        #: by hand to stage a faint; see `whiteout` below.
        self.hp = 20
        self.money = 3000

    def whiteout(self, map_name: str, x: int, y: int) -> None:
        """What Gen 1 does after the party goes down: heal, halve, teleport."""

        self.hp = 22
        self.money //= 2
        self.map_name, self.x, self.y = map_name, x, y

    def get_screen(self) -> Image.Image:
        # Vary the pixels with the frame counter so refreshed PNGs differ.
        return Image.new("RGB", (160, 144), color=(self.frame_count % 251, 24, 32))

    def press(self, button: str, frames: int = 1) -> None:
        self.pressed.append(button)
        if self.in_battle:
            self._press_in_battle(button)
            self.frame_count += frames
            return
        if self.dialog_active:
            # A d-pad press under an open box never reaches the player: in a text
            # box it is swallowed, in a menu it moves the cursor. Measured on the
            # real ROM -- two `walk_down` actions with Oak's dialog up left the
            # player on (5,3), the tile they started on.
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
            if self.flee_succeeds:
                self.in_battle = False
                self.fled = True
            self.battle_menu = "other"
        else:
            self.battle_menu = "other"

    def press_and_settle(self, button: str, frames: int = 8) -> bool:
        """Press, then let the game come to rest — or answer that it did not.

        The same press-then-wait this fake always did; what is new is the answer.
        `settles = False` is a game that is still moving when the batch hands
        back, which on the real emulator is a warp or a cutscene in progress and
        the one frame where the map name and the coordinates disagree.
        """
        self.press(button, frames)
        self.tick(12)
        return self.settles

    def tick(self, frames: int = 1) -> None:
        if self.turn_pending:
            # The turn resolves and the game hands the menu back for the next one.
            self.turn_pending = False
            self.battle_menu = "top"
            self.top_row = self.top_column = 0
        self.frame_count += frames

    def save_state(self, path: str) -> None:
        # The event flags travel with the state, exactly as they do in a real
        # save: they are what makes one save a step behind another, which is
        # the whole question `POST /load` now has to answer.
        Path(path).write_text(
            json.dumps(
                {
                    "x": self.x,
                    "y": self.y,
                    "facing": self.facing,
                    "event_bits": sorted(self.event_bits),
                    # HP travels too, so a reload can hand resources back — the
                    # thing sixteen of the run's loads were actually for.
                    "hp": self.hp,
                }
            ),
            encoding="utf-8",
        )

    def load_state(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.x = payload["x"]
        self.y = payload["y"]
        self.facing = payload["facing"]
        self.event_bits = set(payload.get("event_bits") or ())
        if payload.get("hp") is not None:
            self.hp = payload["hp"]

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
            "money": self.emulator.money,
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
                "hp": self.emulator.hp,
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

    def read_battle_mon(self) -> dict:
        """The Pokemon on the field, with the stats the engine fights with.

        Deliberately not `read_party()[0]`: on the real reader these are two
        different structs holding two different sets of numbers, and the whole
        point of this method is that the harness stopped confusing them.
        """
        return {
            "species": "Bulbasaur",
            "level": 8,
            "hp": 20,
            "max_hp": 22,
            "status": "OK",
            "types": ["Grass", "Poison"],
            "stats": {"attack": 14, "defense": 13, "speed": 13, "special": 15},
            "moves": list(self.emulator.battle_moves),
        }

    def at_battle_top_menu(self) -> bool:
        return self.emulator.in_battle and self.emulator.battle_menu == "top"

    def battle_lock_in(self) -> str | None:
        return self.emulator.locked_in

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

    def read_screen_text(self) -> str:
        return self.emulator.screen_text

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
        "run": {"up": "4+", "left": "4+"},
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
    assert set(payload) == NO_STEP_KEYS


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
    """Walk directions are meaningless on a battle screen. The position is not."""
    emulator = server_app.emulator
    emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["battle"] is True
    for dead in ("moves", "run", "on_warp", "faces"):
        assert dead not in payload, f"{dead} should not be reported in battle"
    assert payload["hp"] == "20/22"


def test_a_battle_frame_still_says_where_the_player_is_standing(server_app):
    """The coordinates survive a battle, so blanking them threw away a fact.

    Measured on the ROM: four wild encounters walked into in Mt. Moon 1F each
    read the same tile during the fight as the overworld read after fleeing. The
    payload used to drop them anyway, `poke act` rendered
    `Mt Moon 1F (None,None) facing None`, and the model spent a `poke state` call
    getting back what the answer already knew -- fifteen `act | state` pairs in
    one 457-call session.
    """
    emulator = server_app.emulator
    emulator.x, emulator.y = 15, 33
    emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert (payload["x"], payload["y"]) == (15, 33)


def test_a_battle_frame_refuses_facing_instead_of_reporting_a_stale_one(server_app):
    """The one field a battle frame holds a wrong value for.

    An encounter interrupts the step that started it, so the sprite facing byte
    is still the direction from before that step: two of four measured battle
    frames read "right" and "up" for steps that were walk_up and walk_down, and
    the overworld came back facing up and down. A refusal costs one line; a
    confident wrong direction costs a wrong turn.
    """
    server_app.emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert "facing" not in payload
    assert payload["facing_unread"] == server.FACING_UNREAD_IN_BATTLE


def test_a_battle_says_why_it_offers_no_walk_directions(server_app):
    """An empty `moves` list and "you are in a battle" are not the same answer."""
    server_app.emulator.in_battle = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert "moves" not in payload and "run" not in payload
    assert payload["no_walk"] == server.NO_WALK_IN_BATTLE
    assert "battle menu" in payload["no_walk"]


def test_a_battle_frame_prices_every_move_instead_of_naming_it(server_app):
    """The move list used to be four bare names, which is not enough to choose.

    `poke calc` had the numbers and was called **zero** times across one
    457-call session, while the same run spent 501 battle commands fleeing and
    49 of its 289 attacks on Growl and Leer, which deal no damage at all. So the
    numbers moved into the payload the model already reads.
    """
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.enemy = PIDGEY

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    priced = {line.split()[0]: line for line in payload["your_moves"]}
    assert set(priced) == {"Scratch", "Growl", "Ember"}
    assert "Ember Fire 25PP" in priced["Ember"]
    assert "KO in" in priced["Ember"]
    # A status move has no range to report and says so rather than showing 0-0.
    assert priced["Growl"] == "Growl Normal 40PP no damage"
    # And the other half of the stay-or-run decision, beside `hp` in the same
    # answer: the enemy's hardest hit, named.
    assert payload["incoming"].startswith(("Tackle", "Gust"))
    assert "up to" in payload["incoming"]


def test_a_battle_frame_names_your_own_level_beside_the_enemy_s(server_app):
    """The payload carried `hp` and never carried a level.

    So the one comparison that decides whether a fight is winnable -- your level
    against theirs -- could not be made without spending a `poke state` call. The
    run this was measured on kept one Charmeleon at level 25 for seventeen hours
    and whited out twice.
    """
    server_app.emulator.in_battle = True
    server_app.emulator.enemy = PIDGEY

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["you"] == "Bulbasaur L8"
    assert payload["enemy"].startswith("Pidgey L5")


def test_a_battle_frame_marks_the_effectiveness_only_when_it_is_not_one(server_app):
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.enemy = {**PIDGEY, "species": "Paras", "types": ["Bug", "Grass"]}

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    priced = {line.split()[0]: line for line in payload["your_moves"]}
    assert "x4" in priced["Ember"]  # Fire on Bug/Grass
    assert "x" not in priced["Scratch"].removeprefix("Scratch")  # Normal on both


def test_a_battle_frame_says_when_nothing_left_can_deal_damage(server_app):
    """`interventions.Toothless`, said in the payload instead of out of band.

    Of 106 auto-saved battle entries from one run, 46 -- 43% -- had no damaging
    move with PP left on the field. The payload listed four move names and the
    run kept walking into fights it could not win and could not flee.
    """
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.enemy = PIDGEY
    emulator.battle_moves[0]["pp"] = 0  # Scratch
    emulator.battle_moves[2]["pp"] = 0  # Ember, leaving only Growl

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["no_damage"] == server.NO_DAMAGE_IN_BATTLE
    assert "Ember Fire 0PP out of PP" in payload["your_moves"]


def test_a_battle_frame_falls_back_to_bare_names_when_the_numbers_are_unreadable(server_app):
    """A battle still starting has no enemy to price against. Names beat nothing."""
    server_app.emulator.in_battle = True
    server_app.emulator.enemy = {}

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["your_moves"] == ["Scratch", "Growl", "Ember"]
    assert "incoming" not in payload


def test_an_open_box_says_the_d_pad_will_not_reach_the_player(server_app):
    """`moves` under an open box was a fact about the map, read as a fact about now.

    Measured with Oak's dialog up: the payload offered `moves: ["down"]` and
    `run down:4`, and `walk_down` moved nothing at all, because a d-pad press
    under a box works the box.
    """
    server_app.emulator.dialog_active = True

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert "moves" not in payload and "run" not in payload
    assert payload["no_walk"] == server.NO_WALK_IN_BOX
    # The position is still knowable and still reported: only stepping is off.
    assert (payload["x"], payload["y"]) == (5, 6)


def test_a_batch_that_never_came_to_rest_says_so(server_app):
    """A mid-transition frame is the one that pairs two maps in one answer.

    Sampled ten frames into a gate warp on the real ROM, the reads say
    `Route 2 (5,0)` -- Route 2's name with the gate's coordinates, while the tile
    the player actually lands on is (3,11). Every press waits for the game to come
    to rest first, so this is rare; when the wait gives up the answer has to say
    what it is describing rather than let a cutscene frame read as a position.
    `POST /load` has reported exactly this since it was written.
    """
    server_app.emulator.settles = False

    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert payload["settled"] is False


def test_a_battle_frame_is_not_labelled_mid_transition(server_app):
    """The settle watchdog never reports rest in a battle, so the flag says nothing.

    It watches the walk counter and the sprite step vectors, and an encounter
    freezes both mid-step: measured on the ROM it gave up on the entry frame and
    on all five turns after it. Reporting that as "the map and the coordinates
    may not belong together" would contradict the coordinates printed beside it,
    which the same measurement showed are the tile the player is standing on.
    """
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.settles = False

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["battle"] is True
    assert "settled" not in payload


def test_a_settled_batch_does_not_pay_for_the_word(server_app):
    """Which is every batch but the rare one: a field that is always there is read once."""
    payload = server_app.http.post("/action", json={"actions": ["walk_up"]}).json()

    assert "settled" not in payload


def test_a_walk_eaten_by_a_box_is_not_reported_as_blocked_ground(server_app):
    """`blocked_after` is inferred from a position that did not change.

    Under an open box nothing moves whatever the ground is, so the inference is
    wrong every time: two `walk_down` presses into Oak's dialog came back
    `moved: 0, blocked_after: 1` about a tile the player walks over daily. That
    is the harness teaching the agent a wall that is not there.
    """
    server_app.emulator.dialog_active = True

    payload = server_app.http.post("/action", json={"actions": ["walk_down", "walk_down"]}).json()

    assert payload["moved"] == 0
    assert "blocked_after" not in payload
    assert payload["no_walk"] == server.NO_WALK_IN_BOX


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


def faint_then_wake_up(app, *, map_name="VIRIDIAN CITY", x=17, y=9):
    """Play out one whiteout: the party goes down, then the game moves the player.

    Two batches, because that is how it lands on the real game — the faint is
    read off one frame and the teleport off the next.
    """
    app.emulator.hp = 0
    app.http.post("/action", json={"actions": ["press_a"]})
    app.emulator.whiteout(map_name, x, y)
    return app.http.post("/action", json={"actions": ["press_a"]}).json()


def test_the_batch_after_a_whiteout_says_it_was_one(server_app):
    # Measured over 33 hours: 19 whiteouts, and the payload rendered every one
    # as an ordinary map change arriving at full HP. The model always worked out
    # for itself that it had fainted, so the fact that earns its place is the
    # bill: $15,249 halved away, checked by the model after three of nineteen,
    # and one of those three read as "I spent some". It had spent nothing.
    payload = faint_then_wake_up(server_app)

    note = payload["whiteout"]
    assert "PALLET TOWN (5,6)" in note, "where the party went down"
    assert "VIRIDIAN CITY (17,9)" in note, "where the game put it"
    assert "$1,500 of your $3,000" in note, "what the halving cost"


def test_the_whiteout_note_is_news_and_not_a_standing_field(server_app):
    # Read once and cleared. A field on every payload is read on none of them:
    # `here_before` was sent 2,339 times in one run and acted on zero.
    faint_then_wake_up(server_app)

    assert "whiteout" not in step(server_app, "up")


def test_an_ordinary_map_change_carries_no_whiteout_note(server_app):
    server_app.emulator.north_map = ("VIRIDIAN CITY", 1, 17)
    server_app.emulator.y = 0

    assert "whiteout" not in step(server_app, "up")


def test_a_reload_over_a_faint_leaves_no_bill_to_pay(server_app):
    # Loading a save rewinds the money along with everything else, so a note
    # would price a loss that no longer exists. Ten of the nineteen measured
    # whiteouts were undone this way within four receipts of the faint.
    server_app.http.post("/save", json={"name": "before"})
    server_app.emulator.hp = 0
    server_app.http.post("/action", json={"actions": ["press_a"]})

    server_app.http.post("/load", json={"name": "before"})
    server_app.emulator.hp = 22
    server_app.emulator.map_name, server_app.emulator.x = "VIRIDIAN CITY", 17

    assert "whiteout" not in step(server_app, "up")


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


def test_a_locked_in_pokemon_is_refused_before_any_button_is_spent(server_app):
    """Rage takes the turn, and the old refusal said the fight was over instead.

    Measured on the ROM: one Rage in Mt. Moon B2F and the top battle menu never
    returned for the rest of the fight. Every battle command after it spent the
    full 24 B presses hunting for a menu that does not exist and was then told
    "the fight is already over ... press A to clear it", which is both false and
    the wrong thing to do. One run chose Rage 77 times.
    """
    emulator = in_battle(server_app)
    emulator.battle_menu = "other"
    emulator.locked_in = "Rage"
    emulator.pressed.clear()

    response = server_app.http.post("/battle/fight", json={"move": "ember"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Rage has locked this Pokemon in" in detail
    assert "No turn was spent." in detail
    assert emulator.pressed == []


def test_running_is_refused_the_same_way_and_says_so(server_app):
    emulator = in_battle(server_app)
    emulator.battle_menu = "other"
    emulator.locked_in = "Rage"

    response = server_app.http.post("/battle/run", json={})

    assert response.status_code == 409
    assert "no menu" in response.json()["detail"]


def test_a_battle_frame_says_the_turn_has_been_taken_away(server_app):
    emulator = server_app.emulator
    emulator.in_battle = True
    emulator.battle_menu = "other"
    emulator.locked_in = "Rage"

    payload = server_app.http.post("/action", json={"actions": ["press_a"]}).json()

    assert payload["locked_in"] == server.locked_in_note("Rage")


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
    # A number alone is a reason to count HP; the name is a reason to switch.
    assert payload["threat_move"] in PIDGEY["moves"]


def test_calc_will_not_promise_a_kill_from_a_move_with_no_pp(server_app):
    # 54 of 106 auto-saved battle entries from one run had a damaging move at
    # 0 PP, and this table ranked it as the best one available. The damage stays
    # on the row -- it is why restoring the PP is worth the walk -- but a move
    # the game refuses does not kill anything this turn.
    server_app.emulator.in_battle = True
    server_app.emulator.enemy = PIDGEY
    server_app.emulator.battle_moves[2]["pp"] = 0

    entries = server_app.http.get("/calc").json()["moves"]
    ember = {entry["move"]: entry for entry in entries}["Ember"]

    assert ember["pp"] == 0
    assert ember["damage"][0] > 0
    assert ember["turns_to_ko"] is None


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
        # Empty rather than absent, and the difference carries: absent means the
        # prices were never checked against RAM, `[]` means they were and none
        # had been handed back. A reader that cannot tell those apart cannot
        # tell a reconciled ledger from an unreconciled one.
        "lost": [],
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


# ---------------------------------------------------------------------------
# The load guard
#
# `poke load` was used 129 times in one run at 0 presses a call, as a fast-travel
# menu rather than as recovery: seven loads in four minutes across three
# timelines, and the run ended on the pre-Misty branch a badge behind the peak it
# had already reached. A load that would hand milestones back is refused.
# ---------------------------------------------------------------------------


def event_rungs(count: int) -> list:
    """The first *count* ladder rungs the fake's event bits can satisfy."""
    from pokemon_agent.milestones import MILESTONES

    return [rung for rung in MILESTONES if rung.kind == "event"][:count]


def earn(app, *rungs) -> None:
    """Set the event bits for *rungs*, as the game would on reaching them."""
    from pokemon_agent.milestones import ALL_EVENTS

    app.emulator.event_bits |= {ALL_EVENTS[rung.source] for rung in rungs}


def world_of(app) -> tuple:
    return (app.emulator.x, app.emulator.y, frozenset(app.emulator.event_bits))


def test_a_load_that_would_hand_a_milestone_back_is_refused(server_app):
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)

    response = server_app.http.post("/load", json={"name": "before_it"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail == (
        f"Refusing: that save is missing {rung.label}. You would lose it. "
        "To try a route, `poke sim` walks a plan without touching the game; "
        "a reload rewinds it. Load it anyway with --force."
    )


def test_the_refusal_names_every_rung_it_is_protecting(server_app):
    first, second = event_rungs(2)
    server_app.http.post("/save", json={"name": "before_both"})
    earn(server_app, first, second)

    detail = server_app.http.post("/load", json={"name": "before_both"}).json()["detail"]

    assert first.label in detail and second.label in detail
    assert "You would lose them." in detail


def test_a_refused_load_leaves_the_world_exactly_where_it_was(server_app):
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)
    server_app.http.post("/action", json={"actions": ["walk_up", "walk_left"]})
    before = world_of(server_app)

    assert server_app.http.post("/load", json={"name": "before_it"}).status_code == 409

    assert world_of(server_app) == before
    assert server_app.http.get("/progress").json()["count"] == 1


def test_force_loads_the_earlier_branch_anyway(server_app):
    """Recovering a branch that really was lost is the one load that goes back."""
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)

    response = server_app.http.post("/load", json={"name": "before_it", "force": True})

    assert response.status_code == 200
    assert server_app.emulator.event_bits == set()


def test_a_save_level_with_the_game_loads_with_no_extra_friction(server_app):
    rung = event_rungs(1)[0]
    earn(server_app, rung)
    server_app.http.post("/save", json={"name": "after_it"})
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    response = server_app.http.post("/load", json={"name": "after_it"})

    assert response.status_code == 200
    assert server_app.emulator.event_bits != set()


def test_a_save_ahead_of_the_game_loads(server_app):
    """A superset is not a regression: nothing is handed back."""
    rung = event_rungs(1)[0]
    earn(server_app, rung)
    server_app.http.post("/save", json={"name": "ahead"})
    server_app.emulator.event_bits = set()

    assert server_app.http.post("/load", json={"name": "ahead"}).status_code == 200


def test_going_back_to_one_save_too_many_times_is_refused(server_app):
    """Retrying a save is a tactic once or twice; past that it stops paying.

    Measured over the run's 131 loads: the first load of a save is followed by
    a milestone within 3,000 presses 46% of the time, the second 32%, the third
    23%, and from the fourth on it is noise. `before_misty` looked like the
    counter-example at fifteen loads — until the timestamps showed Misty fell
    34,000 presses later in a different episode entirely.
    """

    server_app.http.post("/save", json={"name": "checkpoint"})
    for _ in range(3):
        assert server_app.http.post("/load", json={"name": "checkpoint"}).status_code == 200

    response = server_app.http.post("/load", json={"name": "checkpoint"})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "checkpoint has been loaded 3 times" in detail
    assert "no milestone has been reached since" in detail
    assert "`poke sim` walks a plan" in detail
    assert "--force" in detail


def test_reaching_a_rung_clears_the_count(server_app):
    """A milestone is the evidence that whatever it was doing worked.

    The save is re-taken after the rung so the *other* guard stays out of the
    way: any save older than a milestone is regressive, and this is about the
    counter, not about handing progress back.
    """

    server_app.http.post("/save", json={"name": "checkpoint"})
    for _ in range(3):
        server_app.http.post("/load", json={"name": "checkpoint"})
    earn(server_app, event_rungs(1)[0])
    server_app.http.post("/action", json={"actions": ["walk_up"]})
    server_app.http.post("/save", json={"name": "checkpoint"})

    assert server_app.http.post("/load", json={"name": "checkpoint"}).status_code == 200


def test_a_run_resumed_mid_ladder_still_counts_from_the_first_load(server_app):
    """The rungs a session starts holding are a baseline, not a gain.

    The tests above all begin on an empty ladder, so the first receipt of the
    session reports nothing new and the counter survives. A real session
    resumes from a save holding sixteen rungs, and reading those as *just
    earned* clears the counter on the first load every time. Loading one save
    four times against a live server is what exposed it: the guard stayed
    silent through all four.
    """

    from pokemon_agent import server as server_module

    earn(server_app, *event_rungs(2))
    server_app.http.post("/save", json={"name": "checkpoint"})
    # `/save` writes a receipt, which seeds the baseline and hides the bug. A
    # restarted process has seen no receipt at all when the first load arrives,
    # so put the module back in that state — this is exactly the live sequence.
    server_module._last_milestone_ids = None
    server_module._loads_since_milestone.clear()

    for _ in range(3):
        assert server_app.http.post("/load", json={"name": "checkpoint"}).status_code == 200

    assert server_app.http.post("/load", json={"name": "checkpoint"}).status_code == 409


def test_the_count_is_per_save_not_global(server_app):
    """Three different saves is exploring. The same save four times is not."""
    for name in ("one", "two", "three", "four"):
        server_app.http.post("/save", json={"name": name})
    for name in ("one", "two", "three", "four"):
        assert server_app.http.post("/load", json={"name": name}).status_code == 200


def test_force_goes_back_to_a_save_it_has_worn_out(server_app):
    server_app.http.post("/save", json={"name": "checkpoint"})
    for _ in range(3):
        server_app.http.post("/load", json={"name": "checkpoint"})

    response = server_app.http.post("/load", json={"name": "checkpoint", "force": True})

    assert response.status_code == 200


def test_a_load_records_what_it_handed_back(server_app):
    """The measurement the milestone rule deliberately cannot make.

    Sixteen of the run's loads went back to a full party rather than walk to a
    Poke Center, and every one held the same milestones, so nothing in the
    record showed it. A number first, a rule later.
    """

    run_id = open_run()
    server_app.http.post("/save", json={"name": "rested"})
    server_app.emulator.hp = max(0, server_app.emulator.hp - 12)
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    assert server_app.http.post("/load", json={"name": "rested"}).status_code == 200

    loads = [entry for entry in receipts(server_app, run_id) if entry.get("tool") == "load"]
    assert loads[-1]["restored"]["hp"] == 12


def test_a_load_that_hands_nothing_back_records_nothing(server_app):
    run_id = open_run()
    server_app.http.post("/save", json={"name": "same"})

    assert server_app.http.post("/load", json={"name": "same"}).status_code == 200

    loads = [entry for entry in receipts(server_app, run_id) if entry.get("tool") == "load"]
    assert "restored" not in loads[-1]


def test_a_refused_load_leaves_a_receipt_priced_at_nothing(server_app):
    """The guard is worth exactly the milestones it refuses, which has to be countable."""
    run_id = open_run()
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)

    assert server_app.http.post("/load", json={"name": "before_it"}).status_code == 409

    refusals = [entry for entry in receipts(server_app, run_id) if "load_refused" in entry]
    assert len(refusals) == 1
    entry = refusals[0]
    assert entry["tool"] == "load"
    assert entry["presses"] == 0
    assert entry["exit"] == 1
    assert entry["reloaded"] is False  # nothing was rewound
    assert entry["load_refused"] == [rung.id]
    assert entry["milestones_held"] == 1
    assert rung.label in entry["error"]


def test_a_refused_load_leaves_the_repeat_guard_alone(server_app):
    """The world did not move, so what the guard had proved is still true."""
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)
    _into_a_wall(server_app)
    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 400

    assert server_app.http.post("/load", json={"name": "before_it"}).status_code == 409

    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 400


def test_a_forced_load_backwards_shows_up_as_a_fall_in_what_the_game_holds(server_app):
    """`milestone_count` is a running maximum and never falls. This one does."""
    run_id = open_run()
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)
    server_app.http.post("/action", json={"actions": ["press_a"]})

    server_app.http.post("/load", json={"name": "before_it", "force": True})

    written = receipts(server_app, run_id)
    peak = [entry for entry in written if entry["tool"] == "action"][-1]
    after = [entry for entry in written if entry["reloaded"]][-1]
    assert (peak["milestone_count"], peak["milestones_held"]) == (1, 1)
    assert after["milestone_count"] == 1  # the bill never rewinds
    assert after["milestones_held"] == 0  # the game just did


def test_a_save_leaves_a_receipt_naming_it(server_app):
    """129 load receipts and no save receipts: nothing could pair the two."""
    run_id = open_run()

    server_app.http.post("/save", json={"name": "before_brock"})

    saves = [entry for entry in receipts(server_app, run_id) if entry["tool"] == "save"]
    assert len(saves) == 1
    assert saves[0]["presses"] == 0
    assert saves[0]["save"] == "before_brock"
    assert saves[0]["milestones_held"] == 0


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
    assert runway == {"up": "2+", "down": "2+", "left": "3+", "right": "3+"}


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
    # The terrain ends because the window does, not because of a wall.
    assert runway["right"] == "1+"


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
    assert server._runway(snapshot) == {"up": "2+", "left": "3+"}


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


def _b1f_snapshot(walkable, at):
    """Mt Moon B1F with synthetic terrain but its own real warp table."""
    return {
        "player_position": {"x": at[0], "y": at[1]},
        "map_name": "Mt Moon B1F",
        "map_terrain": {"width": 28, "height": 28, "walkable": walkable, "tile_ids": {}},
    }


def test_exits_names_the_reachable_door_not_the_near_unreachable_one():
    """The measured case: standing on Mt Moon B1F at (26,15), which is four pockets.

    Ranking by Manhattan distance advertised the B2F staircase at (21,17) --
    seven tiles away, in a different pocket, unreachable -- and hid (13,27),
    twenty-five away and the way through. Two of the three exits it listed
    there could not be walked to. This field exists to solve the Mt Moon
    problem and was pointing at the wall.
    """
    # A corridor joining the two warps that really are in one pocket, and
    # (21,17) stranded on an island the way the real floor strands it.
    corridor = {(x, 15) for x in range(13, 27)} | {(x, 27) for x in range(13, 27)}
    corridor |= {(13, y) for y in range(15, 28)}
    island = {(21, 17)}

    exits = server._exits(_b1f_snapshot(corridor | island, (26, 15)))

    assert exits.get("Mt Moon 1F") == [25, 15], "the door in this pocket"
    assert exits.get("Mt Moon B2F") == [13, 27], (
        "the reachable B2F staircase, not the nearer one across a wall"
    )


def test_exits_drops_a_destination_whose_every_door_is_out_of_reach():
    """Naming a door you cannot walk to is worse than naming none: it sends the agent."""
    stranded = {(x, 15) for x in range(24, 28)}  # only (25,15) is reachable

    exits = server._exits(_b1f_snapshot(stranded, (26, 15)))

    assert exits.get("Mt Moon 1F") == [25, 15]
    assert "Mt Moon B2F" not in exits, "every B2F door is in another pocket"


def test_exits_falls_back_to_manhattan_when_the_map_is_not_decoded():
    """Without terrain there is nothing better than distance, and it says so by degrading."""
    snapshot = {"player_position": {"x": 26, "y": 15}, "map_name": "Mt Moon B1F"}

    exits = server._exits(snapshot)

    assert "Mt Moon B2F" in exits, "undecoded: every destination is still offered"
    assert "Mt Moon 1F" in exits


def _battle_bundle(warps):
    """A frame mid-fight, on a map that has somewhere to go once it is over."""
    return {
        "state": {
            "map": {"map_name": "Mt Moon B2F"},
            "player": {"position": {"x": 24, "y": 11}, "facing": "left"},
            "party": [{"hp": 22, "max_hp": 73, "moves": [{"name": "Ember"}]}],
            "battle": {"in_battle": True, "enemy": {"species": "Zubat", "level": 12}},
        },
        "navigation": {"snapshot": _warp_snapshot("Mt Moon B2F", (24, 11), warps)},
    }


def test_a_battle_drops_the_exits_list_with_the_other_walk_facts():
    """`exits` answers "where may I walk", and a battle frame cannot answer it.

    The strip that takes `moves`, `run`, `on_warp`, `warp` and `faces` away in a
    battle forgot this one. Measured on a Mt Moon B2F encounter, the frame
    printed `exits Mt Moon B1F (25, 9)` on the line under `no walking in a
    battle`, which is 26 bytes of the payload contradicting itself -- 70 on
    Route 4, on every frame of every fight. The exit has not moved, and the
    first overworld answer after the fight names it again.
    """
    summary = server._observation_summary(_battle_bundle([(25, 9, 60)]))

    assert summary["battle"] is True
    assert summary["no_walk"] == server.NO_WALK_IN_BATTLE
    assert "exits" not in summary


def test_the_same_frame_out_of_battle_keeps_its_exits():
    """The cut is about the battle, not about the map. Same tile, fight over."""
    bundle = _battle_bundle([(25, 9, 60)])
    bundle["state"]["battle"] = {"in_battle": False}

    summary = server._observation_summary(bundle)

    assert summary["exits"], "the way off the floor is the fact this payload is for"


# ---------------------------------------------------------------------------
# The party against what is ahead, and the move about to be deleted.
# ---------------------------------------------------------------------------

#: The live party at 33 hours played: one Pokemon, four moves, two of which
#: damage anything, one Boulder Badge, standing in the gym it lost 40 times in.
_LIVE_PARTY = [
    {
        "species": "Charmeleon",
        "level": 33,
        "hp": 95,
        "max_hp": 95,
        "types": ["Fire"],
        "stats": {"attack": 66, "defense": 54, "speed": 71, "special": 57},
        "moves": [
            {"name": "Cut", "pp": 29},
            {"name": "Growl", "pp": 40},
            {"name": "Ember", "pp": 23},
            {"name": "Leer", "pp": 30},
        ],
    }
]


def _gym_bundle(map_name="Cerulean Gym", in_battle=False):
    return {
        "state": {
            "map": {"map_name": map_name},
            "player": {"position": {"x": 4, "y": 6}, "facing": "up", "badges": ["Boulder"]},
            "party": _LIVE_PARTY,
            "battle": {"in_battle": in_battle},
        },
        "navigation": {"snapshot": {"map_name": map_name, "player_position": {"x": 4, "y": 6}}},
    }


def test_standing_in_an_unwon_gym_prices_the_leader_against_the_party():
    """3,044 presses in Cerulean Gym and no answer ever named Misty's team.

    `poke calc` prices the Pokemon in front of you, which is the wrong room. The
    fight worth pricing is the one being walked toward, and the verb that could
    have priced it was called zero times in a 457-call session -- so it goes in
    the payload the agent already reads.
    """
    server._ahead_said.reset()
    summary = server._observation_summary(_gym_bundle())
    server._annotate_gym_outlook(summary, _gym_bundle())

    assert "Misty" in summary["ahead"]
    assert "Staryu L18" in summary["ahead"] and "Starmie L21" in summary["ahead"]


def test_the_gym_outlook_is_said_once_and_not_on_every_frame_after():
    """113 bytes x 3,044 presses is 344 kB against a 95 kB median session.

    The whole point of the field is that it arrives at the door. Repeating it on
    every frame inside would cost more than everything else the run reads.
    """
    server._ahead_said.reset()
    first, second = {}, {}
    server._annotate_gym_outlook(first, _gym_bundle())
    server._annotate_gym_outlook(second, _gym_bundle())

    assert "ahead" in first
    assert "ahead" not in second


def test_a_gym_already_won_costs_nothing():
    """Pewter is on the way to everywhere. Its leader is not news any more."""
    server._ahead_said.reset()
    summary = {}
    server._annotate_gym_outlook(summary, _gym_bundle("Pewter Gym"))

    assert "ahead" not in summary


#: The bag the live run walked to Vermilion with. TM28 had been in it since Mt.
#: Moon and was never used.
_LIVE_BAG = [
    {"item": "Town Map", "quantity": 1},
    {"item": "Poke Ball", "quantity": 11},
    {"item": "TM34", "quantity": 1},
    {"item": "Potion", "quantity": 9},
    {"item": "TM28", "quantity": 1},
    {"item": "HM01", "quantity": 1},
    {"item": "TM11", "quantity": 1},
]


def _bag_bundle(map_name="Cerulean City", party=None, bag=None):
    bundle = _gym_bundle(map_name)
    bundle["state"]["party"] = _LIVE_PARTY if party is None else party
    bundle["state"]["bag"] = _LIVE_BAG if bag is None else bag
    return bundle


def test_a_machine_in_the_bag_that_beats_the_moveset_is_named():
    """TM28 rode along for roughly 60,000 presses and no payload named it.

    The bag said "TM28 x1", species.json said Charmeleon can learn TM28, and
    nothing in the harness said TM28 is Dig -- so the one Ground move the run
    was carrying, against the one gym Ground walks through, was never a fact the
    model could reach.
    """
    server._tm_said.reset()
    summary = {}
    server._annotate_teachable_tm(summary, _bag_bundle())

    assert "TM28 teaches Dig (Ground 100)" in summary["tm"]
    assert "Charmeleon can learn it" in summary["tm"]
    # TM11 is Bubble Beam, which Charmeleon cannot learn, and TM34 is Bide,
    # which damages nothing. Neither is an upgrade and neither is named.
    assert "TM11" not in summary["tm"] and "TM34" not in summary["tm"]


def test_the_machine_line_is_said_once_per_room_not_on_every_frame():
    """Same bargain as `ahead`: it arrives when a room is walked into."""
    server._tm_said.reset()
    first, second, elsewhere = {}, {}, {}
    server._annotate_teachable_tm(first, _bag_bundle())
    server._annotate_teachable_tm(second, _bag_bundle())
    server._annotate_teachable_tm(elsewhere, _bag_bundle("Route 5"))

    assert "tm" in first
    assert "tm" not in second
    assert "tm" in elsewhere


def test_a_taught_machine_stops_being_news():
    """Once Dig is on the moveset there is no upgrade left in that bag to name."""
    taught = [dict(_LIVE_PARTY[0], moves=[{"name": "Dig", "pp": 10}, {"name": "Cut", "pp": 29}])]
    server._tm_said.reset()
    summary = {}
    server._annotate_teachable_tm(summary, _bag_bundle(party=taught))

    assert "tm" not in summary


def test_an_unreadable_bag_costs_no_line_and_raises_nothing():
    server._tm_said.reset()
    summary = {}
    server._annotate_teachable_tm(summary, {"state": {"party": _LIVE_PARTY}})
    server._annotate_teachable_tm(summary, None)

    assert "tm" not in summary


def test_a_move_about_to_be_overwritten_is_named_before_the_press():
    """Cut went over an attack on the live run and Gen 1 will not delete an HM.

    No advice undoes that afterwards, so the cost has to be in the payload of
    the frame that still has a B button available.
    """
    bundle = _gym_bundle()
    bundle["state"]["move_learn"] = {
        "screen_text": (
            "      CUT\n      GROWL\n      EMBER\n      LEER\n\n Which move should\n be forgotten?"
        ),
        "incoming": "Dig",
        "cursor": 2,
        "slot": 0,
    }

    summary = server._observation_summary(bundle)

    assert "A here deletes Ember (40)" in summary["learn"]
    assert "Dig (100)" in summary["learn"]


def test_an_ordinary_frame_carries_no_learn_line():
    """It costs nothing on the frames it is absent from, which is nearly all of them."""
    assert "learn" not in server._observation_summary(_gym_bundle())


# ---------------------------------------------------------------------------
# The repeat guard, over HTTP
#
# The rule itself is tested in tests/test_repeats.py and proved against the ROM
# in tests/test_repeats_live.py. These are about the wiring: which endpoints
# carry it, that a refusal costs nothing, and that it never wedges the agent.
# ---------------------------------------------------------------------------

from pokemon_agent.repeats import REPEAT_LIMIT  # noqa: E402


def _into_a_wall(app, times=REPEAT_LIMIT + 1):
    """A step that cannot be taken, sent over and over. Vermilion City (33,4).

    One more call than the limit, because the first one turns the player to face
    the wall and turning is a real change. Everything after it is the same
    command against the same frame.
    """
    app.emulator.walls = {(app.emulator.x, app.emulator.y - 1)}
    last = None
    for _ in range(times):
        last = app.http.post("/action", json={"actions": ["walk_up"]})
    return last


def test_a_batch_proved_inert_is_refused_rather_than_run_again(server_app):
    # The run sent `act up:1` from Vermilion (33,4) 362 times and got `moved 0`
    # 362 times. Nothing about the 363rd could have been different.
    assert _into_a_wall(server_app).status_code == 200

    response = server_app.http.post("/action", json={"actions": ["walk_up"]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "blocked" in detail
    assert "different direction" in detail


def test_the_refusal_spends_no_buttons(server_app):
    _into_a_wall(server_app)
    spent = len(server_app.emulator.pressed)

    server_app.http.post("/action", json={"actions": ["walk_up"]})

    assert len(server_app.emulator.pressed) == spent


def test_a_refused_batch_still_lands_in_the_receipts(server_app):
    # A run whose records hold only what it spent cannot show what it was
    # stopped from spending.
    run_id = open_run()
    _into_a_wall(server_app)
    server_app.http.post("/action", json={"actions": ["walk_up"]})

    refusals = [
        entry for entry in receipts(server_app, run_id) if "repeat_refused" in json.dumps(entry)
    ]
    assert len(refusals) == 1
    assert refusals[0]["presses"] == 0
    assert refusals[0]["exit"] == 1


def test_the_agent_is_never_left_without_a_legal_move(server_app):
    # Every escape the refusal names has to actually work from the frame that
    # produced it, or the guard has replaced a loop with a dead end.
    _into_a_wall(server_app)
    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 400

    for escape in (["press_b"], ["wait_60"], ["walk_down"]):
        assert server_app.http.post("/action", json={"actions": escape}).status_code == 200


def test_a_different_command_clears_the_block(server_app):
    _into_a_wall(server_app)
    server_app.http.post("/action", json={"actions": ["press_b"]})

    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 200


def test_a_walk_that_keeps_moving_is_never_refused(server_app):
    for _ in range(REPEAT_LIMIT * 2):
        response = server_app.http.post("/action", json={"actions": ["walk_left"]})
        if not response.json().get("moved"):
            server_app.emulator.x = 10  # walked into the map edge; step back out
    assert server_app.http.post("/action", json={"actions": ["walk_left"]}).status_code == 200


def test_a_plan_simulated_over_and_over_is_refused_too(server_app):
    # `sim` presses nothing and writes no receipt, which is why 531 identical
    # calls in one session went unnoticed until the session died on its token
    # budget rather than on its press budget.
    for _ in range(REPEAT_LIMIT):
        assert server_app.http.post("/sim", json={"actions": ["walk_up"]}).status_code == 200

    response = server_app.http.post("/sim", json={"actions": ["walk_up"]})

    assert response.status_code == 400
    assert "never touches the game" in response.json()["detail"]
    assert server_app.http.post("/sim", json={"actions": ["walk_down"]}).status_code == 200


def test_a_load_forgets_what_it_had_proved(server_app):
    # A load rewinds the world underneath the guard, so whatever it had proved
    # inert may not be any more.
    server_app.http.post("/save", json={"name": "before_the_wall"})
    _into_a_wall(server_app)
    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 400

    server_app.http.post("/load", json={"name": "before_the_wall"})

    assert server_app.http.post("/action", json={"actions": ["walk_up"]}).status_code == 200


def test_a_repeated_flee_is_refused_with_the_verbs_that_are_not_fleeing(server_app):
    # Route 6 (1,15): 331 `poke run` calls over 15.3 minutes, "could not get
    # away" every one of them. Fleeing is a speed roll; it does not warm up.
    server_app.emulator.enemy = {"species": "Zubat", "level": 12, "hp": 30, "max_hp": 30}
    server_app.emulator.in_battle = True
    server_app.emulator.flee_succeeds = False
    for _ in range(REPEAT_LIMIT):
        assert server_app.http.post("/battle/run").status_code == 200

    response = server_app.http.post("/battle/run")

    assert response.status_code == 400
    assert "poke fight" in response.json()["detail"]


def test_a_dialog_whose_words_keep_changing_is_never_refused(server_app):
    # The decoded box is the only thing that separates a conversation being
    # advanced from one being toggled, and it has to survive both state reads
    # the server makes around a batch or the guard is judging a blank screen.
    server_app.emulator.dialog_active = True
    for page in range(REPEAT_LIMIT * 2):
        server_app.emulator.screen_text = f"page {page} of a very long speech"
        response = server_app.http.post("/action", json={"actions": ["press_a"]})
        assert response.status_code == 200, response.json()


def test_a_dialog_stuck_on_words_already_read_is_refused(server_app):
    server_app.emulator.dialog_active = True
    server_app.emulator.screen_text = "SLOWBRO took a snooze..."
    for _ in range(REPEAT_LIMIT):
        assert server_app.http.post("/action", json={"actions": ["press_a"]}).status_code == 200

    response = server_app.http.post("/action", json={"actions": ["press_a"]})

    assert response.status_code == 400
    assert "poke act b:2" in response.json()["detail"]


# ---------------------------------------------------------------------------
# What the thinking session is told the run has
#
# The recorder's attainments are a lifetime record and never fall, which is
# right for pricing a run and wrong for describing a game. After a reload took
# the Cascade badge back, the block below still told a thinking session the run
# held the Bicycle, and the session answered "The bike is yours; use it on
# Route 5". That was delivered.
# ---------------------------------------------------------------------------


def _recorder_with(attainments, presses=91116):
    from pokemon_agent.run_recorder import RunRecorder

    recorder = RunRecorder.__new__(RunRecorder)
    recorder.run_id = "20260825T224823Z-983b"
    recorder.total_presses = presses
    recorder.attainments = list(attainments)
    return recorder


BIKE_RUN = [
    {"milestone_id": "EVENT_GOT_HM01", "label": "Got HM01 Cut", "presses": 65600},
    {"milestone_id": "EVENT_SS_ANNE_LEFT", "label": "The S.S. Anne set sail", "presses": 65737},
    {"milestone_id": "EVENT_GOT_BIKE_VOUCHER", "label": "Got the Bike Voucher", "presses": 68745},
    {"milestone_id": "EVENT_GOT_BICYCLE", "label": "Got the Bicycle", "presses": 91116},
]


def test_the_thinking_session_is_not_told_about_a_bicycle_the_branch_lost(monkeypatch):
    from pokemon_agent import server as server_module

    monkeypatch.setattr(server_module, "_run_recorder", _recorder_with(BIKE_RUN))
    held = ["EVENT_GOT_HM01", "EVENT_SS_ANNE_LEFT"]

    summary = server_module._intervention_milestone_summary(held)

    # The costs section is what the session reads as "the run has this". The
    # Bicycle may still be named above it, as something reloaded past.
    costs = summary.split("Most recent milestones, with what they cost:")[1]
    assert "Bicycle" not in costs
    assert "Bike Voucher" not in costs
    assert "Got HM01 Cut at 65600 presses" in costs


def test_a_rung_the_run_reached_and_lost_is_named_as_lost(monkeypatch):
    """Better to know the run went backwards than to believe it did not."""
    from pokemon_agent import server as server_module

    monkeypatch.setattr(server_module, "_run_recorder", _recorder_with(BIKE_RUN))

    summary = server_module._intervention_milestone_summary(["EVENT_GOT_HM01"])

    assert "no longer held" in summary
    assert "a save was reloaded past them" in summary
    assert "Got the Bicycle" in summary


def test_the_presses_the_run_spent_are_reported_whatever_it_still_holds(monkeypatch):
    """The bill is the recorder's question and the recorder answers it correctly."""
    from pokemon_agent import server as server_module

    monkeypatch.setattr(server_module, "_run_recorder", _recorder_with(BIKE_RUN))

    assert "91116 presses spent" in server_module._intervention_milestone_summary([])


def test_without_a_live_reading_the_block_is_unchanged(monkeypatch):
    from pokemon_agent import server as server_module

    monkeypatch.setattr(server_module, "_run_recorder", _recorder_with(BIKE_RUN))

    summary = server_module._intervention_milestone_summary(None)

    assert "Got the Bicycle at 91116 presses" in summary
    assert "no longer held" not in summary


# ---------------------------------------------------------------------------
# The verbs that spent buttons and recorded none
#
# `mart_buy` and `pokecenter_heal` both drive a menu, so both press real
# buttons, and neither wrote a receipt. A $3,500 purchase of ten Poke Balls and
# five Potions left no trace in the run at all, and every heal the player ever
# ran was missing from the total this project measures itself in. `sim` presses
# nothing, but wrote a receipt only when refused: 124 refusals were recorded
# against 1,340 actual calls in the leg that won the Cascade badge.
# ---------------------------------------------------------------------------


def test_a_batch_says_how_many_plans_were_walked_on_paper_first(server_app):
    run_id = open_run()

    for _ in range(3):
        assert server_app.http.post("/sim", json={"actions": ["walk_up"]}).status_code == 200
    server_app.http.post("/action", json={"actions": ["walk_left"]})

    entries = [entry for entry in receipts(server_app, run_id) if entry.get("tool") == "action"]
    assert entries[-1]["sims"] == 3


def test_the_sim_count_resets_so_it_is_never_double_counted(server_app):
    run_id = open_run()

    server_app.http.post("/sim", json={"actions": ["walk_up"]})
    server_app.http.post("/action", json={"actions": ["walk_left"]})
    server_app.http.post("/action", json={"actions": ["walk_left"]})

    entries = [entry for entry in receipts(server_app, run_id) if entry.get("tool") == "action"]
    assert entries[-2]["sims"] == 1
    # Absent, not zero: a batch that planned nothing is not a batch this counter
    # never watched.
    assert "sims" not in entries[-1]


# ---------------------------------------------------------------------------
# Planning that never becomes pressing
#
# 2,226 simulations in 671 chains. 484 of those chains are one or two plans —
# a model planning and then acting — and the worst single chain ran 140 with
# nothing pressed in it. Tiles gained per chain stays flat at about four however
# long the chain runs, so past the first couple the answers stop buying
# anything.
# ---------------------------------------------------------------------------


def test_a_seventh_different_plan_without_pressing_anything_is_refused(server_app):
    plans = (
        ["walk_up"],
        ["walk_down"],
        ["walk_left"],
        ["walk_right"],
        ["walk_up", "walk_up"],
        ["walk_down", "walk_down"],
    )
    for plan in plans:
        assert server_app.http.post("/sim", json={"actions": plan}).status_code == 200

    response = server_app.http.post("/sim", json={"actions": ["walk_left", "walk_left"]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "6 different plans" in detail
    assert "Press the first step of the best one" in detail


def test_pressing_something_clears_the_count(server_app):
    plans = (
        ["walk_up"],
        ["walk_down"],
        ["walk_left"],
        ["walk_right"],
        ["walk_up", "walk_up"],
        ["walk_down", "walk_down"],
    )
    for plan in plans:
        server_app.http.post("/sim", json={"actions": plan})
    server_app.http.post("/action", json={"actions": ["walk_left"]})

    after = server_app.http.post("/sim", json={"actions": ["walk_right", "walk_right"]})
    assert after.status_code == 200


def test_the_same_plan_again_is_the_repeat_guards_business_not_this_one(server_app):
    """Both rules stay reachable, and they answer different mistakes.

    Counting repeats toward the cap would hit six long before the repeat guard's
    sixteen, leaving that rule unable to fire at all — the shape of bug this
    project keeps finding in itself. Asking one question sixteen times and
    asking six different ones are not the same error.
    """

    for _ in range(REPEAT_LIMIT):
        assert server_app.http.post("/sim", json={"actions": ["walk_up"]}).status_code == 200

    response = server_app.http.post("/sim", json={"actions": ["walk_up"]})

    assert response.status_code == 400
    # The repeat guard's message, not the cap's.
    assert "different plans" not in response.json()["detail"]


def test_a_loop_that_never_reads_the_refusal_is_throttled(server_app):
    """A refusal assumes a reader; 124 of them arrived in 190 milliseconds.

    Each was a separate `poke sim` process on its own connection, so nothing in
    that loop ever saw the 400 it got back. A rate limit does not care whether
    anyone is reading.
    """

    from pokemon_agent import server as server_module

    server_module._sim_call_times.clear()
    seen = set()
    for i in range(server_module.SIM_RATE_MAX_CALLS + 5):
        # Fresh plan each time so the cap above is not what answers, and the
        # press keeps the distinct-plan set empty.
        server_module._sim_plans_since_press.clear()
        plan = {"actions": ["walk_up"] * (1 + i % 30)}
        code = server_app.http.post("/sim", json=plan).status_code
        seen.add(code)
        if code == 429:
            break

    assert 429 in seen, "the rate limit never fired"


def test_a_forced_load_says_so_in_the_record(server_app):
    """Recovering a lost branch and overriding the guard look identical without it.

    Measured live: `got_bike_voucher` refused at 20:23:20 and the same save
    loaded at 20:24:00, taking the run from 18 milestones to 16. The receipts
    hold the refusal and the success and nothing that says what changed between
    them, so "it forced" stayed an inference about the run's most important
    behaviour.
    """

    run_id = open_run()
    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)

    assert server_app.http.post("/load", json={"name": "before_it"}).status_code == 409
    forced_call = server_app.http.post("/load", json={"name": "before_it", "force": True})
    assert forced_call.status_code == 200

    loads = [e for e in receipts(server_app, run_id) if e.get("tool") == "load"]
    refused, forced = loads[-2], loads[-1]
    assert "load_refused" in refused and "forced" not in refused
    assert forced.get("forced") is True
    # Absent rather than false on an ordinary load, so the key means something
    # every time it appears.
    server_app.http.post("/save", json={"name": "level"})
    assert server_app.http.post("/load", json={"name": "level"}).status_code == 200
    plain = [e for e in receipts(server_app, run_id) if e.get("tool") == "load"][-1]
    assert "forced" not in plain


def test_forcing_past_the_guard_leaves_a_way_back(server_app):
    """The guard banked an undo frame for every load except the destructive ones.

    `snapshot = None if req.force or not held_before else ...` skipped exactly
    the loads it knew would cost something, so the run's only two irreversible
    milestone losses were the only two loads that took no snapshot. Forcing is a
    statement of intent, not a guarantee about the outcome, and both forced
    loads this run were regretted within minutes.
    """

    rung = event_rungs(1)[0]
    server_app.http.post("/save", json={"name": "before_it"})
    earn(server_app, rung)
    server_app.http.post("/action", json={"actions": ["walk_up"]})
    before = world_of(server_app)

    forced = server_app.http.post("/load", json={"name": "before_it", "force": True})
    assert forced.status_code == 200
    undo = forced.json()["undo"]
    assert "undo__before_it" in undo and "poke load" in undo
    assert server_app.http.get("/progress").json()["count"] == 0, "the rung really was given up"

    # And the way back works, which is the whole point of naming it.
    assert server_app.http.post("/load", json={"name": "undo__before_it"}).status_code == 200
    assert world_of(server_app) == before
    assert server_app.http.get("/progress").json()["count"] == 1


def test_an_ordinary_load_keeps_no_undo_frame(server_app):
    """Only a forced load needs one; a refused one is already undone."""
    server_app.http.post("/save", json={"name": "level"})

    answer = server_app.http.post("/load", json={"name": "level"})

    assert answer.status_code == 200
    assert "undo" not in answer.json()
    assert not list((server_app.saves_dir).glob("undo__*.state"))


def test_the_token_budget_is_settable_and_visible(server_app):
    """The one setting that decides when a session dies, and it was invisible.

    110,000 is sized for a 140k-context model. A model with a million-token
    window was retired at 11% of what it could hold, and nothing in any payload
    said what the ceiling was — so the run looked like it stalled rather than
    like it hit a limit meant for something else.
    """

    from pokemon_agent import server as server_module
    from pokemon_agent.pi_supervisor import DEFAULT_TOKEN_BUDGET

    sup = server_module._supervisor
    if sup is None:
        pytest.skip("no supervisor on this app")

    original = sup.token_budget
    try:
        assert sup.state_snapshot()["token_budget"] == DEFAULT_TOKEN_BUDGET

        sup.token_budget = 400_000
        assert sup.state_snapshot()["token_budget"] == 400_000
        # Zero means no ceiling, and has to stay distinguishable from unset.
        sup.token_budget = 0
        assert sup.state_snapshot()["token_budget"] == 0
    finally:
        # The supervisor is a module global shared with every other test in this
        # file. Leaving it mutated is how one assertion here turns into a
        # failure somewhere else entirely, which is exactly the intermittent
        # this file has been showing.
        sup.token_budget = original
