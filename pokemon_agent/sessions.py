"""Game sessions — the unit that binds one playthrough together.

A *game session* is a named run that owns everything belonging to that run:

    <data_dir>/games/<session_id>/
        manifest.json          # GameSession.to_dict()
        saves/<name>.state     # emulator save-states for THIS run
        workspace/             # AgentRuntime workspace for THIS run
            debug/             #   checkpoints, landmarks, event memory, ...
            pi-session/        #   Pi's own transcript jsonl files

The directory layout is load-bearing, not cosmetic. ``AgentRuntime`` writes its
auto-saves to ``data_dir / "saves"`` and ``PiSupervisor`` derives its
``pi-session/`` directory from ``workspace_dir``, so pointing a runtime at
``session_dir`` and a supervisor at ``session_dir / "workspace"`` scopes saves,
memory and brain transcript to the run with no further plumbing.

Crucially the manifest persists ``pi_session_id`` / ``pi_session_file``. The
supervisor learns its session id from Pi's own ``session`` event and otherwise
holds it only in memory, so before this a server restart orphaned the brain:
the emulator state survived, the agent's accumulated context did not.

The legacy flat ``<data_dir>/saves/`` directory still works for ad-hoc saves.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_MILESTONES = 100

# Session ids land in filesystem paths and URL path params; keep them boring.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GameSession:
    """Metadata for one playthrough. The on-disk manifest is this, as JSON."""

    id: str
    name: str
    game: str = "red"

    # The Pi brain's continuity across turns *and* across server restarts.
    pi_session_id: Optional[str] = None
    pi_session_file: Optional[str] = None

    objective_pack: Optional[str] = None
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {"turns": 0, "actions": 0, "saves": 0})

    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GameSession":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


class GameSessionManager:
    """Disk-backed CRUD for game sessions under ``<data_dir>/games/``."""

    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir).expanduser().resolve() / "games"
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths

    def session_dir(self, session_id: str) -> Path:
        """Root of one run. Doubles as the AgentRuntime ``data_dir``."""
        return self.root / self._validated(session_id)

    def manifest_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "manifest.json"

    def saves_dir(self, session_id: str) -> Path:
        path = self.session_dir(session_id) / "saves"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def workspace_dir(self, session_id: str) -> Path:
        """AgentRuntime workspace for this run; PiSupervisor hangs pi-session/ off it."""
        path = self.session_dir(session_id) / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _validated(session_id: str) -> str:
        # A session id becomes a path segment. Reject anything that could escape
        # the games/ root ("..", "/", NUL) rather than sanitising it silently.
        if not session_id or not _SAFE_ID.match(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return session_id

    # ---------------------------------------------------------- persistence

    def save(self, session: GameSession) -> GameSession:
        session.updated_at = _now_iso()
        self.session_dir(session.id).mkdir(parents=True, exist_ok=True)
        manifest = self.manifest_path(session.id)
        # Write-then-rename: a crash mid-write must not leave a truncated manifest
        # that would strand the run's pi_session_id.
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session.to_dict(), indent=2))
        tmp.replace(manifest)
        return session

    def load(self, session_id: str) -> Optional[GameSession]:
        manifest = self.manifest_path(session_id)
        if not manifest.exists():
            return None
        try:
            return GameSession.from_dict(json.loads(manifest.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def exists(self, session_id: str) -> bool:
        return self.manifest_path(session_id).exists()

    def create(
        self,
        name: Optional[str] = None,
        game: str = "red",
        objective_pack: Optional[str] = None,
    ) -> GameSession:
        session_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        session = GameSession(
            id=session_id,
            name=name or f"Run {session_id}",
            game=game,
            objective_pack=objective_pack,
        )
        self.saves_dir(session_id)
        self.workspace_dir(session_id)
        return self.save(session)

    def delete(self, session_id: str) -> bool:
        path = self.session_dir(session_id)
        if not path.exists():
            return False
        shutil.rmtree(path)
        if self.current_id() == session_id:
            self.clear_current()
        return True

    def list(self) -> List[Dict[str, Any]]:
        """Summaries of every session, newest-updated first."""
        summaries: List[Dict[str, Any]] = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            session = self.load(path.name)
            if session is None:
                continue
            saves = self.list_saves(session.id)
            summaries.append(
                {
                    "id": session.id,
                    "name": session.name,
                    "game": session.game,
                    "pi_session_id": session.pi_session_id,
                    "resumable": session.pi_session_id is not None,
                    "save_count": len(saves),
                    "latest_save": saves[0]["name"] if saves else None,
                    "turns": session.stats.get("turns", 0),
                    "milestones": len(session.milestones),
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                }
            )
        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return summaries

    # ------------------------------------------------------- active session

    @property
    def _current_pointer(self) -> Path:
        return self.root / "current.json"

    def current_id(self) -> Optional[str]:
        pointer = self._current_pointer
        if not pointer.exists():
            return None
        try:
            session_id = json.loads(pointer.read_text()).get("id")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        # A stale pointer (session deleted out from under us) is not an error.
        if not session_id or not self.exists(session_id):
            return None
        return session_id

    def current(self) -> Optional[GameSession]:
        session_id = self.current_id()
        return self.load(session_id) if session_id else None

    def set_current(self, session_id: str) -> None:
        if not self.exists(session_id):
            raise ValueError(f"Unknown session: {session_id}")
        pointer = self._current_pointer
        tmp = pointer.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"id": session_id}, indent=2))
        tmp.replace(pointer)

    def clear_current(self) -> None:
        self._current_pointer.unlink(missing_ok=True)

    # ------------------------------------------------------------ brain tie

    def record_pi_session(
        self,
        session: GameSession,
        pi_session_id: Optional[str],
        pi_session_file: Optional[Path | str] = None,
    ) -> GameSession:
        """Persist the Pi brain's identity so a restart can resume this run."""
        session.pi_session_id = pi_session_id
        session.pi_session_file = str(pi_session_file) if pi_session_file else None
        return self.save(session)

    # -------------------------------------------------------------- saves

    def list_saves(self, session_id: str) -> List[Dict[str, Any]]:
        """Save-states for this run, newest first."""
        saves = []
        for path in self.saves_dir(session_id).glob("*.state"):
            stat = path.stat()
            saves.append(
                {
                    "name": path.stem,
                    "file": path.name,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        saves.sort(key=lambda item: item["modified"], reverse=True)
        return saves

    def latest_save_path(self, session_id: str) -> Optional[Path]:
        saves = self.list_saves(session_id)
        if not saves:
            return None
        return self.saves_dir(session_id) / saves[0]["file"]

    # ---------------------------------------------------------- milestones

    def add_milestone(
        self, session: GameSession, description: str, category: str = "milestone"
    ) -> GameSession:
        session.milestones.insert(
            0,
            {
                "description": description,
                "category": category,
                "turn": session.stats.get("turns", 0),
                "at": _now_iso(),
            },
        )
        del session.milestones[MAX_MILESTONES:]
        return self.save(session)
