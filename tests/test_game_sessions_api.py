"""/games endpoints: New Game / Load Game, scoped saves, and brain resume."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pokemon_agent import server
from pokemon_agent.sessions import GameSessionManager


class FakeEmulator:
    def __init__(self) -> None:
        self.frame_count = 1234
        self.saved_paths: list[str] = []
        self.loaded_paths: list[str] = []
        self.image = Image.new("RGB", (160, 144), color=(16, 24, 32))

    def get_screen(self):
        return self.image

    def save_state(self, path: str) -> None:
        Path(path).write_bytes(b"state")
        self.saved_paths.append(path)

    def load_state(self, path: str) -> None:
        self.loaded_paths.append(path)

    def tick(self, frames: int = 1) -> None:
        self.frame_count += frames


class FakeSupervisor:
    """Stands in for PiSupervisor where only liveness/identity matter."""

    def __init__(self, *, is_running: bool = False, session_id: str | None = None) -> None:
        self.is_running = is_running
        self.session_id = session_id


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def client(data_dir: Path, monkeypatch):
    """Server with game sessions live, but no ROM: _startup bails out early when
    _config is set after the fact, so the globals we patch here survive."""
    emulator = FakeEmulator()
    sessions = GameSessionManager(data_dir)

    monkeypatch.setattr(
        server,
        "_config",
        server.GameConfig(rom_path="/nonexistent.gb", game_type="red", data_dir=str(data_dir)),
    )
    monkeypatch.setattr(server, "_emulator", emulator)
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_active_session_id", None)
    # There is no ROM, so no memory reader. The suite's convention is to stub the
    # state/navigation reads rather than fake a reader.
    monkeypatch.setattr(server, "_get_state_dict", lambda: {"map": {"map_name": "Pallet Town"}})
    monkeypatch.setattr(server, "_get_navigation_payload_sync", lambda goal=None: None)

    with TestClient(server.app) as test_client:
        test_client.emulator = emulator
        test_client.sessions = sessions
        yield test_client


def test_no_games_to_begin_with(client):
    body = client.get("/games").json()
    assert body["sessions"] == []
    assert body["current"] is None
    assert client.get("/games/current").json()["session"] is None


def test_creating_a_game_activates_it_and_lays_out_its_directories(client, data_dir):
    body = client.post("/games", json={"name": "Nuzlocke"}).json()

    session_id = body["session"]["id"]
    assert body["session"]["name"] == "Nuzlocke"
    assert body["session"]["active"] is True
    assert client.get("/games").json()["current"] == session_id
    assert (data_dir / "games" / session_id / "manifest.json").exists()
    assert (data_dir / "games" / session_id / "workspace").is_dir()


def test_creating_without_activating_leaves_the_current_run_alone(client):
    first = client.post("/games", json={"name": "First"}).json()["session"]["id"]
    client.post("/games", json={"name": "Second", "activate": False})

    assert client.get("/games").json()["current"] == first
    assert len(client.get("/games").json()["sessions"]) == 2


def test_the_runtime_is_rebound_to_the_active_run(client, data_dir):
    session_id = client.post("/games", json={}).json()["session"]["id"]

    # This is the whole mechanism: AgentRuntime auto-saves to data_dir/saves, so
    # scoping the run means handing it the session directory as its data_dir.
    assert server._runtime.data_dir == (data_dir / "games" / session_id).resolve()
    assert (
        server._runtime.workspace_dir == (data_dir / "games" / session_id / "workspace").resolve()
    )


def save_names(client) -> list[str]:
    return [item["name"] for item in client.get("/saves").json()["saves"]]


def test_manual_saves_are_scoped_to_the_active_run(client, data_dir):
    session_id = client.post("/games", json={}).json()["session"]["id"]

    assert client.post("/save", json={"name": "pewter"}).status_code == 200

    assert (data_dir / "games" / session_id / "saves" / "pewter.state").exists()
    assert not (data_dir / "saves" / "pewter.state").exists()
    assert "pewter" in save_names(client)


def test_runtime_auto_saves_are_scoped_to_the_active_run_too(client, data_dir):
    # AgentRuntime writes auto-saves itself, to `data_dir/saves`. If the runtime were
    # not rebound they would leak into the shared pool that /saves lists.
    session_id = client.post("/games", json={}).json()["session"]["id"]

    auto = list((data_dir / "games" / session_id / "saves").glob("auto__*.state"))
    assert auto, "expected the runtime to auto-save into the run"
    assert not list((data_dir / "saves").glob("auto__*.state"))


def test_two_runs_do_not_see_each_others_saves(client):
    first = client.post("/games", json={"name": "First"}).json()["session"]["id"]
    client.post("/save", json={"name": "in-first"})

    client.post("/games", json={"name": "Second"})
    client.post("/save", json={"name": "in-second"})
    assert "in-second" in save_names(client)
    assert "in-first" not in save_names(client)

    client.post(f"/games/{first}/activate", json={"load_latest_save": False})
    assert "in-first" in save_names(client)
    assert "in-second" not in save_names(client)


def test_activating_a_run_loads_its_latest_save(client):
    session_id = client.post("/games", json={}).json()["session"]["id"]
    client.post("/save", json={"name": "pewter"})
    client.post("/games", json={"name": "Other"})  # switch away

    body = client.post(f"/games/{session_id}/activate", json={}).json()

    assert body["loaded_save"] == "pewter"
    assert client.emulator.loaded_paths[-1].endswith(f"{session_id}/saves/pewter.state")


def test_activating_can_skip_loading_the_save(client):
    session_id = client.post("/games", json={}).json()["session"]["id"]
    client.post("/save", json={"name": "pewter"})
    client.post("/games", json={"name": "Other"})

    body = client.post(f"/games/{session_id}/activate", json={"load_latest_save": False}).json()

    assert body["loaded_save"] is None
    assert client.emulator.loaded_paths == []


def test_activating_a_run_with_no_saves_is_fine(client):
    # Created without activating, so nothing has run against it and it has no saves.
    session_id = client.post("/games", json={"activate": False}).json()["session"]["id"]

    body = client.post(f"/games/{session_id}/activate", json={}).json()

    assert body["success"] is True
    assert body["loaded_save"] is None
    assert client.emulator.loaded_paths == []


# ------------------------------------------------------------------- guards


def test_switching_runs_is_refused_while_pi_is_running(client, monkeypatch):
    session_id = client.post("/games", json={}).json()["session"]["id"]
    monkeypatch.setattr(server, "_supervisor", FakeSupervisor(is_running=True))

    # Rebinding under a live Pi turn would leave the subprocess writing into the
    # previous run's workspace.
    assert client.post(f"/games/{session_id}/activate", json={}).status_code == 409
    assert client.post("/games", json={"name": "New"}).status_code == 409
    assert client.delete(f"/games/{session_id}").status_code == 409


def test_creating_without_activating_is_allowed_while_pi_runs(client, monkeypatch):
    client.post("/games", json={})
    monkeypatch.setattr(server, "_supervisor", FakeSupervisor(is_running=True))

    assert client.post("/games", json={"name": "Queued", "activate": False}).status_code == 200


def test_unknown_run_is_a_404(client):
    assert client.post("/games/20260101_000000_abcdef/activate", json={}).status_code == 404
    assert client.delete("/games/20260101_000000_abcdef").status_code == 404


def test_a_traversal_style_id_is_rejected_not_resolved(client):
    assert client.post("/games/..%2F..%2Fetc/activate", json={}).status_code in (400, 404)


# ------------------------------------------------------------------- delete


def test_deleting_the_active_run_falls_back_to_the_shared_layout(client, data_dir):
    session_id = client.post("/games", json={}).json()["session"]["id"]

    body = client.delete(f"/games/{session_id}").json()

    assert body["current"] is None
    assert not (data_dir / "games" / session_id).exists()
    assert server._runtime.data_dir == data_dir.resolve()
    assert client.get("/games/current").json()["session"] is None


# -------------------------------------------------------------- brain resume


def test_the_pi_session_is_written_into_the_manifest(client, data_dir):
    session_id = client.post("/games", json={}).json()["session"]["id"]

    # The supervisor calls its session_sink when Pi reports a session id.
    server._supervisor.session_sink("pi-abc", Path("/tmp/pi-abc.jsonl"))

    manifest = json.loads((data_dir / "games" / session_id / "manifest.json").read_text())
    assert manifest["pi_session_id"] == "pi-abc"
    assert manifest["pi_session_file"] == "/tmp/pi-abc.jsonl"


def test_a_restart_rebinds_to_the_active_run_and_resumes_its_brain(client, data_dir):
    session_id = client.post("/games", json={}).json()["session"]["id"]

    # Pi reports a session, and writes the transcript the supervisor resolves against.
    pi_dir = data_dir / "games" / session_id / "workspace" / "pi-session"
    pi_dir.mkdir(parents=True, exist_ok=True)
    (pi_dir / "pi-abc.jsonl").write_text(json.dumps({"type": "session", "id": "pi-abc"}) + "\n")
    server._supervisor.session_sink("pi-abc", pi_dir / "pi-abc.jsonl")

    # Simulate a restart: rebuild from the persisted current-session pointer alone.
    server._bind_session(GameSessionManager(data_dir).current_id())

    assert server._active_session_id == session_id
    assert server._supervisor.session_id == "pi-abc"  # the brain came back
