"""Finding the workspace and the run store without being told where they are.

The server is started with ``--data-dir`` and ``--agent-workspace-dir``, and the
values differ per checkout — a worktree keeps its workspace beside itself while
the runs stay in the main tree. Hard-coding either would make ``scope`` useless
on anyone else's machine, so it asks, in order: the command line, the
environment, the live server's own argv (read out of ``/proc``, which costs
nothing and touches nothing), then the obvious places relative to the cwd.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

#: Matches ``pokemon_agent.bench.registry.DEFAULT_DATA_DIR``.
DEFAULT_DATA_DIR = Path("~/.pokemon-agent")

WORKSPACE_ENV = "POKE_SCOPE_WORKSPACE"
DATA_DIR_ENV = "POKE_SCOPE_DATA_DIR"

#: The workspace subdirectory holding the Pi transcripts.
SESSION_DIRNAME = "pi-session"
#: Where the harness writes its own event log inside the workspace.
DEBUG_DIRNAME = "debug"

_WORKSPACE_FLAG = "--agent-workspace-dir"
_DATA_DIR_FLAG = "--data-dir"
_SERVER_MARKER = "pokemon_agent.cli"


@dataclass(frozen=True)
class Paths:
    """Where the data is, and how each half of it was found."""

    workspace: Optional[Path]
    data_dir: Optional[Path]
    workspace_source: str = "not found"
    data_dir_source: str = "not found"

    @property
    def session_dir(self) -> Optional[Path]:
        return None if self.workspace is None else self.workspace / SESSION_DIRNAME

    @property
    def run_log(self) -> Optional[Path]:
        if self.workspace is None:
            return None
        return self.workspace / DEBUG_DIRNAME / "run_log.jsonl"

    @property
    def latest_observation(self) -> Optional[Path]:
        if self.workspace is None:
            return None
        return self.workspace / DEBUG_DIRNAME / "latest_observation.json"

    @property
    def current_objective(self) -> Optional[Path]:
        if self.workspace is None:
            return None
        return self.workspace / DEBUG_DIRNAME / "current_objective.json"


def looks_like_workspace(path: Path) -> bool:
    """A workspace is the directory the transcripts and the debug log live in."""

    return (path / SESSION_DIRNAME).is_dir() or (path / DEBUG_DIRNAME).is_dir()


def looks_like_data_dir(path: Path) -> bool:
    return (path / "runs").is_dir()


def iter_server_argv(proc_dir: Path = Path("/proc")) -> Iterator[list[str]]:
    """The argv of every live ``pokemon_agent`` server this user can see.

    Reading ``/proc/<pid>/cmdline`` is a read of a virtual file and cannot
    disturb the process. Anything unreadable — another user's process, one that
    exited between the listing and the read — is simply skipped.
    """

    try:
        entries = sorted(proc_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part for part in raw.decode("utf-8", "replace").split("\0") if part]
        if any(_SERVER_MARKER in part for part in argv):
            yield argv


def _flag_value(argv: list[str], flag: str) -> Optional[str]:
    for index, part in enumerate(argv):
        if part == flag and index + 1 < len(argv):
            return argv[index + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None


def live_server_paths(proc_dir: Path = Path("/proc")) -> tuple[Optional[Path], Optional[Path]]:
    """``(workspace, data_dir)`` as the running server was told them, if it runs."""

    for argv in iter_server_argv(proc_dir):
        workspace = _flag_value(argv, _WORKSPACE_FLAG)
        data_dir = _flag_value(argv, _DATA_DIR_FLAG)
        if workspace or data_dir:
            return (
                Path(workspace).expanduser() if workspace else None,
                Path(data_dir).expanduser() if data_dir else None,
            )
    return None, None


def _walk_up(start: Path) -> Iterator[Path]:
    current = start.resolve()
    yield current
    yield from current.parents


def discover(
    workspace: Optional[str | Path] = None,
    data_dir: Optional[str | Path] = None,
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    proc_dir: Path = Path("/proc"),
) -> Paths:
    """Resolve both locations, recording which rule supplied each one."""

    environ = os.environ if env is None else env
    here = Path.cwd() if cwd is None else Path(cwd)

    found_workspace: Optional[Path] = None
    found_data_dir: Optional[Path] = None
    workspace_source = "not found"
    data_dir_source = "not found"

    if workspace:
        found_workspace = Path(workspace).expanduser()
        workspace_source = "--workspace"
    elif environ.get(WORKSPACE_ENV):
        found_workspace = Path(environ[WORKSPACE_ENV]).expanduser()
        workspace_source = f"${WORKSPACE_ENV}"

    if data_dir:
        found_data_dir = Path(data_dir).expanduser()
        data_dir_source = "--data-dir"
    elif environ.get(DATA_DIR_ENV):
        found_data_dir = Path(environ[DATA_DIR_ENV]).expanduser()
        data_dir_source = f"${DATA_DIR_ENV}"

    if found_workspace is None or found_data_dir is None:
        live_workspace, live_data_dir = live_server_paths(proc_dir)
        if found_workspace is None and live_workspace is not None:
            found_workspace, workspace_source = live_workspace, "live server argv"
        if found_data_dir is None and live_data_dir is not None:
            found_data_dir, data_dir_source = live_data_dir, "live server argv"

    if found_data_dir is None:
        for candidate in _walk_up(here):
            if looks_like_data_dir(candidate):
                found_data_dir, data_dir_source = candidate, "cwd"
                break
        else:
            fallback = DEFAULT_DATA_DIR.expanduser()
            if looks_like_data_dir(fallback):
                found_data_dir, data_dir_source = fallback, "default"

    if found_workspace is None:
        candidates: list[Path] = []
        for parent in _walk_up(here):
            candidates.append(parent / ".agent-workspace")
            candidates.append(parent / "agent_workspace")
        if found_data_dir is not None:
            candidates.append(found_data_dir / "agent_workspace")
            candidates.append(found_data_dir / ".agent-workspace")
        for candidate in candidates:
            if looks_like_workspace(candidate):
                found_workspace, workspace_source = candidate, "cwd"
                break

    return Paths(
        workspace=found_workspace,
        data_dir=found_data_dir,
        workspace_source=workspace_source if found_workspace is not None else "not found",
        data_dir_source=data_dir_source if found_data_dir is not None else "not found",
    )


def list_sessions(session_dir: Optional[Path]) -> list[Path]:
    """Transcript files, oldest first. The last one is the live session."""

    if session_dir is None or not session_dir.is_dir():
        return []
    return sorted((p for p in session_dir.glob("*.jsonl") if p.is_file()), key=lambda p: p.name)


def resolve_session(session_dir: Optional[Path], wanted: Optional[str]) -> Optional[Path]:
    """The newest transcript, or the one whose id or filename contains ``wanted``."""

    sessions = list_sessions(session_dir)
    if not sessions:
        return None
    if not wanted:
        return sessions[-1]
    for path in reversed(sessions):
        if wanted in path.name:
            return path
    return None
