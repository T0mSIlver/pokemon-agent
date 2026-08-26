import os

from pokemon_agent.progress import ProgressMonitor

# ---------------------------------------------------------------------------
# Autosave retention
#
# One save lands on every map change, objective change and battle, at ~170 KB.
# Unbounded, that reached 3,651 files and 610 MB, filled the disk, and killed a
# live run mid-cave with ENOSPC. Four hours passed before anyone noticed.
# ---------------------------------------------------------------------------


def _auto_save(saves_dir, name, when):
    path = saves_dir / f"auto__{name}.state"
    path.write_bytes(b"x" * 16)
    os.utime(path, (when, when))
    return path


def test_old_autosaves_are_pruned_once_the_limit_is_passed(tmp_path):
    saves = tmp_path / "saves"
    saves.mkdir()
    for i in range(10):
        _auto_save(saves, f"{i:03d}", when=1_000_000 + i)

    monitor = ProgressMonitor(data_dir=tmp_path, auto_save_limit=4)
    monitor._prune_auto_saves(saves)

    left = sorted(p.name for p in saves.glob("auto__*.state"))
    assert len(left) == 4
    # The newest survive, not an arbitrary four.
    assert left == ["auto__006.state", "auto__007.state", "auto__008.state", "auto__009.state"]


def test_pruning_never_touches_a_save_someone_named(tmp_path):
    saves = tmp_path / "saves"
    saves.mkdir()
    for i in range(6):
        _auto_save(saves, f"{i:03d}", when=1_000_000 + i)
    keeper = saves / "before_brock.state"
    keeper.write_bytes(b"precious")

    monitor = ProgressMonitor(data_dir=tmp_path, auto_save_limit=2)
    monitor._prune_auto_saves(saves)

    assert keeper.exists()
    assert keeper.read_bytes() == b"precious"
    assert len(list(saves.glob("auto__*.state"))) == 2


def test_pruning_is_quiet_when_there_is_nothing_to_do(tmp_path):
    saves = tmp_path / "saves"
    saves.mkdir()
    _auto_save(saves, "001", when=1_000_000)

    monitor = ProgressMonitor(data_dir=tmp_path, auto_save_limit=300)
    monitor._prune_auto_saves(saves)

    assert len(list(saves.glob("auto__*.state"))) == 1


def test_pruning_a_missing_directory_does_not_raise(tmp_path):
    monitor = ProgressMonitor(data_dir=tmp_path, auto_save_limit=1)
    monitor._prune_auto_saves(tmp_path / "not-there")
