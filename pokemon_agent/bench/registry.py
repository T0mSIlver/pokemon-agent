"""On-disk storage for benchmark runs: one directory per run, one line per batch.

A run is a directory under ``<data_dir>/runs/<run_id>/`` holding two files:

``meta.json``
    Written once at ``start_run`` and rewritten once at ``finish``, always
    atomically (temp file in the same directory, ``fsync``, ``os.replace``), so
    a reader never sees a half-written header. Same pattern as
    ``ExploredMaps.save``.

``receipts.jsonl``
    Append-only. One receipt per agent action batch, appended for days on the
    hot path of a live run, so the write is a single ``write()`` of one complete
    line to a handle opened ``O_APPEND`` — no read-modify-write, no rewrite of
    what is already on disk, and no chance of interleaving with another writer
    mid-line. A crash can therefore only ever lose or truncate the *last* line,
    and :func:`read_receipts` skips a trailing partial line instead of refusing
    to load the run.

The receipt schema is fixed; see :class:`Receipt`.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

SCHEMA_VERSION = 1

RUNS_DIRNAME = "runs"
META_FILENAME = "meta.json"
RECEIPTS_FILENAME = "receipts.jsonl"

STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"

#: How often an append is pushed all the way to the platter. A receipt is a few
#: hundred bytes written once per action batch — seconds apart at best — so the
#: default syncs every one of them and still costs nothing measurable. Raise it
#: if the store lands on a slow disk; the append itself is already atomic.
DEFAULT_FSYNC_EVERY = 1

#: Default store location, matching the ``--data-dir`` the server uses.
DEFAULT_DATA_DIR = Path("~/.pokemon-agent")


def _log(message: str) -> None:
    print(f"[bench] {message}")


def utc_stamp(when: Optional[float] = None) -> str:
    """``20260823T142312Z`` — the run-id prefix, which sorts chronologically."""

    moment = datetime.fromtimestamp(when if when is not None else time.time(), tz=timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` in one step, so a reader sees either version, never both."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_pair(value: Any) -> Optional[tuple[int, int]]:
    """``[12, 8]`` or ``{"x": 12, "y": 8}`` as a coordinate pair, or None."""

    if isinstance(value, Mapping):
        first, second = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value
    else:
        return None
    left, right = _as_optional_int(first), _as_optional_int(second)
    if left is None or right is None:
        return None
    return left, right


def _as_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, (str, int)))


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


@dataclass(frozen=True)
class Receipt:
    """One agent action batch, as it is written to ``receipts.jsonl``.

    ``presses`` is the number of buttons the batch actually sent. It is the
    only quantity the headline metric is built from, and nothing in this module
    or in :mod:`pokemon_agent.bench.metrics` ever discounts it.
    """

    seq: int = 0
    t: float = 0.0
    presses: int = 0
    map_name: str = ""
    pos: Optional[tuple[int, int]] = None
    moved: Optional[int] = None
    blocked_after: Optional[str] = None
    hp: Optional[tuple[int, int]] = None
    party_size: int = 0
    milestones_new: tuple[str, ...] = ()
    milestone_count: int = 0
    tool: str = ""
    exit_code: int = 0
    reloaded: bool = False
    whiteout: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    #: Keys the fixed schema owns; anything else in a receipt lands in ``extra``.
    KNOWN_KEYS = frozenset(
        {
            "seq",
            "t",
            "presses",
            "map",
            "pos",
            "moved",
            "blocked_after",
            "hp",
            "party_size",
            "milestones_new",
            "milestone_count",
            "tool",
            "exit",
            "reloaded",
            "whiteout",
        }
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Receipt":
        """Read a receipt back, tolerating anything a live run managed to write."""

        blocked = payload.get("blocked_after")
        return cls(
            seq=_as_int(payload.get("seq")),
            t=_as_float(payload.get("t")),
            presses=max(0, _as_int(payload.get("presses"))),
            map_name=_as_text(payload.get("map")),
            pos=_as_pair(payload.get("pos")),
            moved=_as_optional_int(payload.get("moved")),
            blocked_after=None if blocked is None else str(blocked),
            hp=_as_pair(payload.get("hp")),
            party_size=_as_int(payload.get("party_size")),
            milestones_new=_as_ids(payload.get("milestones_new")),
            milestone_count=_as_int(payload.get("milestone_count")),
            tool=_as_text(payload.get("tool")),
            exit_code=_as_int(payload.get("exit")),
            reloaded=bool(payload.get("reloaded")),
            whiteout=bool(payload.get("whiteout")),
            extra={key: value for key, value in payload.items() if key not in cls.KNOWN_KEYS},
        )

    def to_dict(self) -> dict[str, Any]:
        """The fixed record, exactly as the schema spells it."""

        payload: dict[str, Any] = {
            "seq": self.seq,
            "t": self.t,
            "presses": self.presses,
            "map": self.map_name,
            "pos": list(self.pos) if self.pos is not None else None,
            "moved": self.moved,
            "blocked_after": self.blocked_after,
            "hp": list(self.hp) if self.hp is not None else None,
            "party_size": self.party_size,
            "milestones_new": list(self.milestones_new),
            "milestone_count": self.milestone_count,
            "tool": self.tool,
            "exit": self.exit_code,
            "reloaded": self.reloaded,
            "whiteout": self.whiteout,
        }
        payload.update(self.extra)
        return payload

    @property
    def is_action_batch(self) -> bool:
        """A batch that sent buttons, and so can be judged on whether it moved."""

        return self.presses > 0

    @property
    def blocked(self) -> bool:
        """Pressed buttons and ended where it started."""

        return self.is_action_batch and self.moved == 0

    @property
    def errored(self) -> bool:
        return self.exit_code != 0


@dataclass(frozen=True)
class RunMeta:
    """The header written at ``start_run`` and closed off at ``finish``."""

    run_id: str
    started_at: float = 0.0
    ended_at: Optional[float] = None
    status: str = STATUS_RUNNING
    harness_sha: str = ""
    config_hash: str = ""
    model: str = ""
    start_checkpoint: Optional[str] = None
    goal: str = ""
    finish_reason: str = ""
    version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, run_id: str = "") -> "RunMeta":
        checkpoint = payload.get("start_checkpoint")
        return cls(
            run_id=_as_text(payload.get("run_id"), run_id) or run_id,
            started_at=_as_float(payload.get("started_at")),
            ended_at=(
                None if payload.get("ended_at") is None else _as_float(payload.get("ended_at"))
            ),
            status=_as_text(payload.get("status"), STATUS_RUNNING) or STATUS_RUNNING,
            harness_sha=_as_text(payload.get("harness_sha")),
            config_hash=_as_text(payload.get("config_hash")),
            model=_as_text(payload.get("model")),
            start_checkpoint=None if checkpoint is None else str(checkpoint),
            goal=_as_text(payload.get("goal")),
            finish_reason=_as_text(payload.get("finish_reason")),
            version=_as_int(payload.get("version"), SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "harness_sha": self.harness_sha,
            "config_hash": self.config_hash,
            "model": self.model,
            "start_checkpoint": self.start_checkpoint,
            "goal": self.goal,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class RunRecord:
    """A whole run, read back off disk: its header and every receipt it wrote."""

    meta: RunMeta
    receipts: tuple[Receipt, ...] = ()
    #: Lines that did not parse — a crash mid-append leaves at most one.
    corrupt_lines: int = 0

    @property
    def run_id(self) -> str:
        return self.meta.run_id

    def __len__(self) -> int:
        return len(self.receipts)


@dataclass(frozen=True)
class RunSummary:
    """One line of ``list_runs``: the header plus how much the run wrote."""

    run_id: str
    started_at: float = 0.0
    ended_at: Optional[float] = None
    status: str = STATUS_RUNNING
    model: str = ""
    goal: str = ""
    finish_reason: str = ""
    receipt_count: int = 0

    @classmethod
    def from_meta(cls, meta: RunMeta, receipt_count: int) -> "RunSummary":
        return cls(
            run_id=meta.run_id,
            started_at=meta.started_at,
            ended_at=meta.ended_at,
            status=meta.status,
            model=meta.model,
            goal=meta.goal,
            finish_reason=meta.finish_reason,
            receipt_count=receipt_count,
        )


def iter_receipt_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every parseable object in a receipts file, skipping what is not.

    A run killed mid-append leaves a truncated final line. That is expected, and
    it must cost the reader that one receipt and nothing else.
    """

    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                yield {}  # Signals a corrupt line to the caller, which counts it.
                continue
            if isinstance(payload, dict):
                yield payload
            else:
                yield {}


def read_receipts(path: Path) -> tuple[tuple[Receipt, ...], int]:
    """``(receipts, corrupt_line_count)`` for one receipts file."""

    receipts: list[Receipt] = []
    corrupt = 0
    for payload in iter_receipt_lines(path):
        if not payload:
            corrupt += 1
            continue
        receipts.append(Receipt.from_dict(payload))
    return tuple(receipts), corrupt


def count_lines(path: Path) -> int:
    """Non-empty line count, read in chunks so a days-long run stays cheap."""

    total = 0
    try:
        with path.open("rb") as handle:
            trailing_newline = True
            while True:
                chunk = handle.read(1 << 16)
                if not chunk:
                    break
                total += chunk.count(b"\n")
                trailing_newline = chunk.endswith(b"\n")
            if not trailing_newline:
                total += 1  # A final line the process never got to terminate.
    except OSError:
        return 0
    return total


class RunRegistry:
    """Creates runs, appends receipts to them, and reads them back.

    One instance can hold several runs open at once; each keeps an ``O_APPEND``
    handle so the hot path is one ``write`` and one ``flush``.
    """

    def __init__(self, data_dir: Path, *, fsync_every: int = DEFAULT_FSYNC_EVERY) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.runs_dir = self.data_dir / RUNS_DIRNAME
        self.fsync_every = max(0, int(fsync_every))
        self._handles: dict[str, Any] = {}
        self._appends_since_sync: dict[str, int] = {}
        self._next_seq: dict[str, int] = {}

    # -- layout ----------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def meta_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / META_FILENAME

    def receipts_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / RECEIPTS_FILENAME

    def exists(self, run_id: str) -> bool:
        return self.meta_path(run_id).is_file()

    # -- creating --------------------------------------------------------

    def _allocate_run_id(self, when: Optional[float] = None) -> str:
        """A fresh id whose lexical order is its chronological order."""

        stamp = utc_stamp(when)
        for _ in range(64):
            candidate = f"{stamp}-{random.randrange(16**4):04x}"
            if not self.run_dir(candidate).exists():
                return candidate
        raise RuntimeError(f"could not allocate a run id under {self.runs_dir}")

    def start_run(
        self,
        *,
        harness_sha: str,
        config_hash: str,
        model: str,
        start_checkpoint: Optional[str],
        goal: str,
        run_id: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> str:
        """Create the run directory and its header. Returns the new run id."""

        started = time.time() if started_at is None else float(started_at)
        new_id = run_id or self._allocate_run_id(started)
        directory = self.run_dir(new_id)
        directory.mkdir(parents=True, exist_ok=True)
        meta = RunMeta(
            run_id=new_id,
            started_at=started,
            status=STATUS_RUNNING,
            harness_sha=str(harness_sha or ""),
            config_hash=str(config_hash or ""),
            model=str(model or ""),
            start_checkpoint=None if start_checkpoint is None else str(start_checkpoint),
            goal=str(goal or ""),
        )
        _atomic_write_text(
            self.meta_path(new_id), json.dumps(meta.to_dict(), indent=2, sort_keys=False) + "\n"
        )
        self.receipts_path(new_id).touch(exist_ok=True)
        self._next_seq[new_id] = 0
        return new_id

    # -- appending -------------------------------------------------------

    def _handle(self, run_id: str):
        handle = self._handles.get(run_id)
        if handle is not None and not handle.closed:
            return handle
        directory = self.run_dir(run_id)
        if not directory.is_dir():
            raise FileNotFoundError(f"no such run: {run_id} (expected {directory})")
        # "a" is O_APPEND: every write lands at the current end of file as one
        # step, so two writers can never interleave inside a line.
        handle = self.receipts_path(run_id).open("a", encoding="utf-8")
        self._handles[run_id] = handle
        self._appends_since_sync.setdefault(run_id, 0)
        return handle

    def _seq_for(self, run_id: str) -> int:
        seq = self._next_seq.get(run_id)
        if seq is None:
            # Resuming a run this process did not start: pick up where it left off.
            seq = count_lines(self.receipts_path(run_id))
        self._next_seq[run_id] = seq + 1
        return seq

    def append(self, run_id: str, receipt: Mapping[str, Any] | Receipt) -> None:
        """Append one receipt. Assigns ``seq`` when the caller left it out."""

        payload = receipt.to_dict() if isinstance(receipt, Receipt) else dict(receipt)
        if payload.get("seq") is None:
            payload["seq"] = self._seq_for(run_id)
        else:
            self._next_seq[run_id] = _as_int(payload["seq"]) + 1
        if payload.get("t") is None:
            payload["t"] = round(time.time(), 3)
        record = Receipt.from_dict(payload)
        # One line, one write. json.dumps cannot emit a bare newline inside a
        # string, so the line is self-delimiting even under a hard kill.
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n"
        handle = self._handle(run_id)
        handle.write(line)
        handle.flush()
        if self.fsync_every:
            pending = self._appends_since_sync.get(run_id, 0) + 1
            if pending >= self.fsync_every:
                pending = 0
                try:
                    os.fsync(handle.fileno())
                except OSError as exc:  # noqa: BLE001 — never break the run loop
                    _log(f"could not fsync receipts for {run_id}: {exc}")
            self._appends_since_sync[run_id] = pending

    # -- closing ---------------------------------------------------------

    def close(self, run_id: str) -> None:
        handle = self._handles.pop(run_id, None)
        self._appends_since_sync.pop(run_id, None)
        if handle is None or handle.closed:
            return
        try:
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            pass
        finally:
            handle.close()

    def close_all(self) -> None:
        for run_id in list(self._handles):
            self.close(run_id)

    def finish(self, run_id: str, reason: str) -> None:
        """Close the receipts file and stamp the header as finished."""

        self.close(run_id)
        meta = self.load_meta(run_id)
        finished = RunMeta(
            run_id=meta.run_id,
            started_at=meta.started_at,
            ended_at=time.time(),
            status=STATUS_FINISHED,
            harness_sha=meta.harness_sha,
            config_hash=meta.config_hash,
            model=meta.model,
            start_checkpoint=meta.start_checkpoint,
            goal=meta.goal,
            finish_reason=str(reason or ""),
            version=meta.version,
        )
        _atomic_write_text(
            self.meta_path(run_id), json.dumps(finished.to_dict(), indent=2, sort_keys=False) + "\n"
        )

    # -- reading ---------------------------------------------------------

    def load_meta(self, run_id: str) -> RunMeta:
        path = self.meta_path(run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise FileNotFoundError(f"unreadable run header {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise FileNotFoundError(f"unreadable run header {path}: not an object")
        return RunMeta.from_dict(payload, run_id=run_id)

    def load(self, run_id: str) -> RunRecord:
        """The whole run: header plus every receipt that parses."""

        meta = self.load_meta(run_id)
        receipts, corrupt = read_receipts(self.receipts_path(run_id))
        return RunRecord(meta=meta, receipts=receipts, corrupt_lines=corrupt)

    def list_runs(self) -> tuple[RunSummary, ...]:
        """Every run in the store, oldest first. Reads headers, not receipts."""

        summaries: list[RunSummary] = []
        try:
            entries = sorted(self.runs_dir.iterdir())
        except OSError:
            return ()
        for directory in entries:
            if not directory.is_dir() or not (directory / META_FILENAME).is_file():
                continue
            try:
                meta = self.load_meta(directory.name)
            except FileNotFoundError as exc:
                _log(f"ignoring {directory.name}: {exc}")
                continue
            summaries.append(RunSummary.from_meta(meta, count_lines(directory / RECEIPTS_FILENAME)))
        summaries.sort(key=lambda summary: (summary.started_at, summary.run_id))
        return tuple(summaries)

    def load_many(self, run_ids: Sequence[str]) -> tuple[RunRecord, ...]:
        return tuple(self.load(run_id) for run_id in run_ids)

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "RunRegistry":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close_all()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close_all()
        except Exception:  # noqa: BLE001
            pass
