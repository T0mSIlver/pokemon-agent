"""Run-scoped game sessions: per-run saves, workspace, and a resumable Pi brain."""

import json

import pytest

from pokemon_agent.sessions import GameSession, GameSessionManager


@pytest.fixture
def manager(tmp_path):
    return GameSessionManager(tmp_path)


def test_create_lays_out_the_run_directory(manager):
    session = manager.create(name="Nuzlocke")

    assert session.name == "Nuzlocke"
    assert manager.manifest_path(session.id).exists()
    assert manager.saves_dir(session.id).is_dir()
    assert manager.workspace_dir(session.id).is_dir()


def test_session_dir_is_the_runtime_data_dir(manager):
    # AgentRuntime writes auto-saves to `data_dir / "saves"`. Handing it the session
    # dir is what scopes those saves to the run, so this relationship must hold.
    session = manager.create()
    assert manager.saves_dir(session.id) == manager.session_dir(session.id) / "saves"
    assert manager.workspace_dir(session.id) == manager.session_dir(session.id) / "workspace"


def test_manifest_round_trips(manager):
    created = manager.create(name="Run A", objective_pack="intro_to_brock")
    loaded = manager.load(created.id)

    assert loaded == created
    assert loaded.objective_pack == "intro_to_brock"


def test_load_of_unknown_session_returns_none(manager):
    assert manager.load("20260101_000000_abcdef") is None
    assert not manager.exists("20260101_000000_abcdef")


def test_corrupt_manifest_is_reported_as_missing_not_raised(manager):
    session = manager.create()
    manager.manifest_path(session.id).write_text("{ truncated")

    assert manager.load(session.id) is None


def test_manifest_write_is_atomic(manager):
    # A torn manifest would strand the run's pi_session_id, so the write must be
    # rename-based -- no .tmp residue, and the manifest always parses.
    session = manager.create()
    manager.record_pi_session(session, "pi-abc", "/tmp/pi-abc.jsonl")

    leftovers = list(manager.session_dir(session.id).glob("*.tmp"))
    assert leftovers == []
    assert json.loads(manager.manifest_path(session.id).read_text())["pi_session_id"] == "pi-abc"


def test_unknown_manifest_fields_are_ignored(manager):
    # Forward compatibility: a manifest written by a newer version must still load.
    session = manager.create()
    payload = json.loads(manager.manifest_path(session.id).read_text())
    payload["some_future_field"] = 42
    manager.manifest_path(session.id).write_text(json.dumps(payload))

    assert manager.load(session.id).id == session.id


@pytest.mark.parametrize("bad_id", ["..", "../escape", "a/b", "", "with space", "nul\x00"])
def test_session_ids_that_could_escape_the_root_are_rejected(manager, bad_id):
    with pytest.raises(ValueError, match="Invalid session id"):
        manager.session_dir(bad_id)


# --------------------------------------------------------------- brain resume


def test_pi_session_is_persisted_so_a_restart_can_resume(manager, tmp_path):
    session = manager.create()
    assert session.pi_session_id is None

    manager.record_pi_session(session, "pi-xyz", tmp_path / "pi-xyz.jsonl")

    # Simulate a server restart: brand new manager over the same data dir.
    reopened = GameSessionManager(tmp_path).load(session.id)
    assert reopened.pi_session_id == "pi-xyz"
    assert reopened.pi_session_file == str(tmp_path / "pi-xyz.jsonl")


def test_recording_a_null_pi_session_clears_the_file(manager):
    session = manager.create()
    manager.record_pi_session(session, "pi-1", "/tmp/pi-1.jsonl")
    manager.record_pi_session(session, None)

    reloaded = manager.load(session.id)
    assert reloaded.pi_session_id is None
    assert reloaded.pi_session_file is None


# ------------------------------------------------------------ active session


def test_there_is_no_current_session_until_one_is_set(manager):
    assert manager.current_id() is None
    assert manager.current() is None


def test_current_session_survives_a_restart(manager, tmp_path):
    session = manager.create(name="Active")
    manager.set_current(session.id)

    assert GameSessionManager(tmp_path).current().name == "Active"


def test_setting_an_unknown_session_current_raises(manager):
    with pytest.raises(ValueError, match="Unknown session"):
        manager.set_current("20260101_000000_abcdef")


def test_a_pointer_to_a_deleted_session_reads_as_no_current_session(manager):
    session = manager.create()
    manager.set_current(session.id)
    manager.delete(session.id)

    assert manager.current_id() is None


# ------------------------------------------------------------------- listing


def test_list_is_newest_updated_first_and_flags_resumability(manager):
    first = manager.create(name="First")
    second = manager.create(name="Second")
    manager.record_pi_session(second, "pi-2")  # touches updated_at

    listing = manager.list()
    assert [item["name"] for item in listing] == ["Second", "First"]
    assert listing[0]["resumable"] is True
    assert listing[1]["resumable"] is False
    assert {item["id"] for item in listing} == {first.id, second.id}


def test_list_skips_directories_without_a_manifest(manager):
    manager.create()
    (manager.root / "not-a-session").mkdir()

    assert len(manager.list()) == 1


def test_saves_are_scoped_to_their_run(manager):
    one = manager.create()
    two = manager.create()
    (manager.saves_dir(one.id) / "pewter.state").write_bytes(b"x")
    (manager.saves_dir(two.id) / "cerulean.state").write_bytes(b"y")

    assert [s["name"] for s in manager.list_saves(one.id)] == ["pewter"]
    assert [s["name"] for s in manager.list_saves(two.id)] == ["cerulean"]
    assert manager.latest_save_path(one.id).name == "pewter.state"


def test_latest_save_path_is_none_when_the_run_has_no_saves(manager):
    assert manager.latest_save_path(manager.create().id) is None


def test_delete_removes_the_whole_run(manager):
    session = manager.create()
    (manager.saves_dir(session.id) / "a.state").write_bytes(b"x")

    assert manager.delete(session.id) is True
    assert not manager.session_dir(session.id).exists()
    assert manager.delete(session.id) is False


# ---------------------------------------------------------------- milestones


def test_milestones_are_newest_first_and_bounded(manager):
    session = manager.create()
    for index in range(105):
        manager.add_milestone(session, f"event {index}")

    reloaded = manager.load(session.id)
    assert len(reloaded.milestones) == 100
    assert reloaded.milestones[0]["description"] == "event 104"


def test_from_dict_tolerates_a_minimal_manifest():
    session = GameSession.from_dict({"id": "x", "name": "y"})
    assert session.game == "red"
    assert session.stats == {"turns": 0, "actions": 0, "saves": 0}


def test_delete_clears_the_current_pointer_instead_of_orphaning_it(manager, tmp_path):
    # current_id() resolves through exists(), so reading it AFTER the rmtree reports
    # None and the pointer would never be cleared.
    session = manager.create()
    manager.set_current(session.id)

    manager.delete(session.id)

    assert not (manager.root / "current.json").exists()
    assert GameSessionManager(tmp_path).current_id() is None


def test_deleting_a_non_current_session_leaves_the_pointer_alone(manager):
    active = manager.create()
    other = manager.create()
    manager.set_current(active.id)

    manager.delete(other.id)

    assert manager.current_id() == active.id
