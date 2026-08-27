"""The bridge between a live playthrough and ``pokemon_agent.bench``.

:mod:`pokemon_agent.bench` can score a run and print a table; nothing had ever
written it a receipt. This module is what writes them, and it is split the way
the two writers actually fall:

* :meth:`RunRecorder.begin_session` and :meth:`RunRecorder.finish_run` are the
  run lifecycle, called by :class:`~pokemon_agent.pi_supervisor.PiSupervisor`,
  which is the only thing that knows a playthrough has started, what its goal
  is and which model is driving it.
* :meth:`RunRecorder.append` is the receipt, called by
  :mod:`pokemon_agent.server` after an action batch, because only the server
  ever sees a batch and its outcome.

A run is not a session
----------------------

A session dies every ~30 minutes when the token budget trips, and the watchdog
POSTs ``/supervisor/start`` for the next one. A *run* is the playthrough those
sessions add up to, so it is identified by a pointer file,
``<data_dir>/runs/CURRENT``, holding the run id. Every session start reads the
pointer: if it names a run whose header still says ``running``, that run is
adopted and its totals are recovered from its own receipts; otherwise a new run
is created and the pointer is rewritten. A server restart, a crashed session
and a fresh watchdog all take the same path, so presses accumulate across every
one of those boundaries instead of resetting.

The pointer is cleared only by :meth:`finish_run` — reaching the objective, or
an operator saying the playthrough is over. Nothing about a session ending is
allowed to end the run.

Presses never reset
-------------------

:func:`pokemon_agent.bench.metrics.compute` deliberately has no branch on
``reloaded``, and neither does anything here. A reload writes a receipt with
``presses=0`` and ``reloaded=True``; the running total carries straight over it.
A gym won on the fourth attempt costs what all four attempts cost.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from pokemon_agent.bench.metrics import LadderEntry, compute, load_ladder
from pokemon_agent.bench.registry import (
    RUNS_DIRNAME,
    STATUS_RUNNING,
    Receipt,
    RunRegistry,
)
from pokemon_agent.state_analysis import party_is_down

#: The file naming the run that is currently open, next to the run directories.
RUN_POINTER_FILENAME = "CURRENT"

#: How many receipts stay in memory for the intervention detectors to read.
#: :class:`~pokemon_agent.interventions.InterventionPolicy` slices the last 120;
#: double that is enough for every detector and is bounded on a days-long run.
RECENT_RECEIPT_WINDOW = 240

#: Milestone snapshot: the ids the game currently satisfies. Async because the
#: server has to take the emulator lock to read RAM.
MilestoneSnapshot = Callable[[], Awaitable[frozenset]]


def _log(message: str) -> None:
    print(f"[run] {message}")


_LADDER: Optional[dict[str, LadderEntry]] = None


def ladder() -> dict[str, LadderEntry]:
    """The curated milestone ladder, read once. ``{}`` if it will not load."""

    global _LADDER
    if _LADDER is None:
        _LADDER = load_ladder()
    return _LADDER


def _read_ref(git_dir: Path, ref: str) -> str:
    """Resolve one ref name to a sha, loose file first, then ``packed-refs``."""

    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip()[:40]
    except OSError:
        pass
    try:
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha.strip()[:40]
    except OSError:
        pass
    return ""


def harness_sha(repo_root: Optional[Path] = None) -> str:
    """The checked-out commit, read off ``.git`` without running git.

    A receipt store that cannot say which harness produced it cannot be used to
    argue that a harness change helped, so this is worth reading — but it is
    never worth failing a run over, and an export or a tarball has no ``.git``
    at all.

    Linked worktrees keep ``HEAD`` in their own directory and their refs in the
    repository's, so an unresolved ref is retried against ``commondir``. Without
    that every run played from a worktree records a blank sha.
    """

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    git_dir = root / ".git"
    try:
        if git_dir.is_file():  # a linked worktree: ".git" points at the real dir
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir:"):
                git_dir = Path(pointer.split(":", 1)[1].strip())
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not head.startswith("ref:"):
        return head[:40]
    ref = head.split(":", 1)[1].strip()
    sha = _read_ref(git_dir, ref)
    if sha:
        return sha
    try:
        common = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return _read_ref((git_dir / common).resolve(), ref)


def config_hash(payload: Mapping[str, Any]) -> str:
    """A short stable digest of the knobs a run was played with."""

    text = json.dumps(dict(payload), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RunHandle:
    """What :meth:`RunRecorder.begin_session` decided, for the caller to report."""

    run_id: str
    adopted: bool
    sessions: int
    total_presses: int


class RunRecorder:
    """One open run, its receipts, and the totals ``/progress`` reports.

    Holds no lock of its own. Every mutation happens on the event loop, and the
    only blocking work — the append itself — is handed to the default executor
    so a fsync never stalls the loop that is serving the next action.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        registry: Optional[RunRegistry] = None,
        milestone_snapshot: Optional[MilestoneSnapshot] = None,
        start_checkpoint: Optional[str] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.registry = registry if registry is not None else RunRegistry(self.data_dir)
        self.milestone_snapshot = milestone_snapshot
        self.start_checkpoint = start_checkpoint
        self.repo_root = repo_root

        self.run_id: Optional[str] = None
        self.sessions: int = 0
        self.total_presses: int = 0
        self.presses_to: dict[str, int] = {}
        self.attainments: list[dict[str, Any]] = []
        self.milestone_count: int = 0
        #: Milestones the game held at the last receipt that read the oracle, or
        #: ``None`` before any did. Unlike ``milestone_count`` this one falls: a
        #: load onto an earlier branch hands rungs back, and the running maximum
        #: is right not to notice while nothing else is looking.
        self.milestones_held: Optional[int] = None
        self.receipts_written: int = 0
        self.last_error: Optional[str] = None
        self.recent: deque[Receipt] = deque(maxlen=RECENT_RECEIPT_WINDOW)

        #: Wall clock of the run's first receipt, so an attainment can say how
        #: long it took as well as what it cost.
        self._first_t: Optional[float] = None

        #: Milestones already satisfied, so a rung is priced the first time it is
        #: actually reached and a resumed checkpoint does not re-price its history.
        self._known: set[str] = set()

    # -- pointer ---------------------------------------------------------

    @property
    def pointer_path(self) -> Path:
        return self.data_dir / RUNS_DIRNAME / RUN_POINTER_FILENAME

    def read_pointer(self) -> Optional[str]:
        try:
            value = self.pointer_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def _write_pointer(self, run_id: str) -> None:
        path = self.pointer_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(run_id + "\n", encoding="utf-8")
        os.replace(temp, path)

    def _clear_pointer(self) -> None:
        self.pointer_path.unlink(missing_ok=True)

    # -- lifecycle -------------------------------------------------------

    def _adoptable_run(self) -> Optional[str]:
        """The open run the pointer names, or ``None`` to start a fresh one."""

        run_id = self.read_pointer()
        if not run_id or not self.registry.exists(run_id):
            return None
        try:
            meta = self.registry.load_meta(run_id)
        except FileNotFoundError as exc:
            _log(f"pointer names {run_id} but its header is unreadable ({exc}); starting fresh")
            return None
        return run_id if meta.status == STATUS_RUNNING else None

    def _recover_totals(self, run_id: str) -> None:
        """Read an adopted run's own receipts back into the live counters.

        Scoring is :func:`pokemon_agent.bench.metrics.compute`, not a second
        implementation of it, so what ``/progress`` reports mid-run and what the
        bench report prints afterwards cannot drift apart.
        """

        record = self.registry.load(run_id)
        metrics = compute(record)
        self.total_presses = metrics.total_presses
        self.presses_to = dict(metrics.presses_to)
        self.attainments = [
            {
                "milestone_id": item.milestone_id,
                "label": item.label,
                "ladder_index": item.ladder_index,
                "presses": item.presses,
                "seq": item.seq,
                "seconds": item.seconds,
            }
            for item in metrics.attainments
        ]
        self.milestone_count = metrics.final_milestone_count
        self.receipts_written = metrics.receipts
        self._first_t = metrics.first_t
        self._known = set(self.presses_to)
        self.recent.clear()
        self.recent.extend(record.receipts[-RECENT_RECEIPT_WINDOW:])
        if record.corrupt_lines:
            # A hard kill mid-append truncates the final line and nothing else.
            _log(f"{run_id}: skipped {record.corrupt_lines} unparseable receipt line(s)")

    async def begin_session(
        self,
        *,
        goal: str,
        model: str = "",
        config: Optional[Mapping[str, Any]] = None,
    ) -> RunHandle:
        """Open the run this session belongs to, creating it only if needed."""

        run_id = self._adoptable_run()
        adopted = run_id is not None
        if run_id is None:
            run_id = self.registry.start_run(
                harness_sha=harness_sha(self.repo_root),
                config_hash=config_hash(config or {}),
                model=model,
                start_checkpoint=self.start_checkpoint,
                goal=goal,
            )
            self._write_pointer(run_id)
            self.total_presses = 0
            self.presses_to = {}
            self.attainments = []
            self.milestone_count = 0
            self.receipts_written = 0
            self._known = set()
            self._first_t = None
            self.recent.clear()
        else:
            self._recover_totals(run_id)

        self.run_id = run_id
        self.sessions += 1

        # Everything already true when the run opened is history, not progress:
        # a run resumed from a checkpoint that has four badges did not earn them
        # in its first five presses.
        baseline = await self._read_milestones()
        self._known.update(baseline)
        if not adopted:
            await self.append(
                tool="run_start",
                presses=0,
                milestone_ids=baseline,
                extra={"baseline_milestones": sorted(baseline), "session": self.sessions},
            )
        return RunHandle(
            run_id=run_id,
            adopted=adopted,
            sessions=self.sessions,
            total_presses=self.total_presses,
        )

    async def finish_run(self, reason: str) -> Optional[str]:
        """Close the run for good and drop the pointer. Sessions never call this."""

        run_id = self.run_id
        if run_id is None:
            return None
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self.registry.finish, run_id, reason
            )
        except Exception as exc:  # noqa: BLE001 — a bad close must not wedge the loop
            self.last_error = f"could not finish run {run_id}: {exc}"
            _log(self.last_error)
        self._clear_pointer()
        self.run_id = None
        return run_id

    def close(self) -> None:
        """Release the append handles. Leaves the run open, because it is."""

        try:
            self.registry.close_all()
        except Exception as exc:  # noqa: BLE001
            _log(f"could not close receipt handles: {exc}")

    # -- receipts --------------------------------------------------------

    async def _read_milestones(self) -> frozenset:
        snapshot = self.milestone_snapshot
        if snapshot is None:
            return frozenset()
        try:
            return frozenset(await snapshot())
        except Exception as exc:  # noqa: BLE001 — a missing oracle is not a failed run
            _log(f"milestone snapshot failed: {exc}")
            return frozenset()

    def _price(self, milestone_ids: Iterable[str], seq: int, at: float) -> tuple[str, ...]:
        """First attainments in ``milestone_ids``, priced at the running total."""

        rungs = ladder()
        gained: list[str] = []
        for milestone_id in milestone_ids:
            if milestone_id in self._known:
                continue
            self._known.add(milestone_id)
            gained.append(milestone_id)
            rung = rungs.get(milestone_id)
            self.presses_to[milestone_id] = self.total_presses
            self.attainments.append(
                {
                    "milestone_id": milestone_id,
                    "label": rung.label if rung else milestone_id,
                    "ladder_index": rung.ladder_index if rung else None,
                    "presses": self.total_presses,
                    "seq": seq,
                    "seconds": (
                        round(at - self._first_t, 3) if self._first_t is not None else None
                    ),
                }
            )
        return tuple(gained)

    async def append(
        self,
        *,
        tool: str,
        presses: int = 0,
        map_name: str = "",
        pos: Optional[tuple[int, int]] = None,
        moved: Optional[int] = None,
        blocked_after: Optional[Any] = None,
        hp: Optional[tuple[int, int]] = None,
        party_size: int = 0,
        milestone_ids: Optional[Iterable[str]] = None,
        exit_code: int = 0,
        reloaded: bool = False,
        whiteout: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Receipt]:
        """Write one receipt. Never raises, never resets a counter.

        The line is built here and written on the default executor, so the only
        blocking syscall on the path — one ``write`` plus one ``fsync`` of a few
        hundred bytes — happens off the event loop and cannot stall the request
        that follows this batch.
        """

        run_id = self.run_id
        if run_id is None:
            return None

        now = time.time()
        seq = self.receipts_written
        if self._first_t is None:
            self._first_t = now
        # The one line that makes the metric honest: no branch on `reloaded`.
        self.total_presses += max(0, int(presses))
        # `None` is "this caller never read the oracle" and is not the same
        # answer as "the game holds none", so it is kept apart from the empty
        # set all the way onto the receipt.
        live = None if milestone_ids is None else frozenset(str(item) for item in milestone_ids)
        gained = self._price(() if live is None else live, seq, now)
        self.milestone_count = max(self.milestone_count, len(self._known))
        # The live count, beside the running maximum. `_known` accumulates and
        # never shrinks -- that is what prices a rung once and only once -- so it
        # cannot answer "how many does the game hold *now*". Only the ids just
        # read off RAM can, and `_price` has already folded them into `_known`,
        # so the intersection is the live set reconciled against the run's own
        # history. A reload onto an earlier branch shows up here as a fall while
        # `milestone_count` holds its peak, which is what the bill should do.
        held = None if live is None else len(self._known & live)
        if held is not None:
            self.milestones_held = held

        receipt = Receipt(
            seq=seq,
            t=round(now, 3),
            presses=max(0, int(presses)),
            map_name=str(map_name or ""),
            pos=pos,
            moved=moved,
            blocked_after=None if blocked_after is None else str(blocked_after),
            hp=hp,
            party_size=int(party_size or 0),
            milestones_new=gained,
            milestone_count=self.milestone_count,
            milestones_held=held,
            tool=str(tool or ""),
            exit_code=int(exit_code or 0),
            reloaded=bool(reloaded),
            whiteout=bool(whiteout),
            extra=dict(extra or {}),
        )
        self.receipts_written += 1
        self.recent.append(receipt)

        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self.registry.append, run_id, receipt
            )
        except Exception as exc:  # noqa: BLE001 — a lost receipt must not lose the batch
            self.last_error = f"receipt {seq} not written: {exc}"
            _log(self.last_error)
        return receipt

    # -- reading ---------------------------------------------------------

    def recent_receipts(self, limit: int = RECENT_RECEIPT_WINDOW) -> tuple[Receipt, ...]:
        window = list(self.recent)
        return tuple(window[-limit:])

    def progress_payload(self) -> dict[str, Any]:
        """The additive half of ``GET /progress``: what the run has cost so far.

        Field names are the ones the dashboard already prefers over the values it
        derives for itself, so this replaces guesswork rather than adding a rival
        source of truth.
        """

        return {
            "run_id": self.run_id,
            "presses_to": dict(self.presses_to),
            "attainments": [dict(item) for item in self.attainments],
        }

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sessions": self.sessions,
            "presses": self.total_presses,
            "receipts": self.receipts_written,
            "milestones": len(self.presses_to),
            "start_checkpoint": self.start_checkpoint,
            "last_error": self.last_error,
        }


def receipt_from_batch(
    *,
    tool: str,
    presses: int,
    bundle: Optional[Mapping[str, Any]],
    outcome: Optional[Mapping[str, Any]] = None,
    milestone_ids: Optional[Sequence[str]] = None,
    exit_code: int = 0,
    reloaded: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Shape one batch's observation into :meth:`RunRecorder.append` keywords.

    Pure, and tolerant of every field being absent: an action that errored has
    no bundle at all and still has to leave a receipt, because two failures in a
    row is exactly what the ``repeated_failure`` detector fires on.
    """

    state = ((bundle or {}).get("state")) or {}
    player = state.get("player") or {}
    position = player.get("position") or {}
    party = state.get("party") or []
    lead = (party[0] if party else {}) or {}

    pos: Optional[tuple[int, int]] = None
    if position.get("x") is not None and position.get("y") is not None:
        pos = (int(position["x"]), int(position["y"]))

    hp: Optional[tuple[int, int]] = None
    if lead.get("max_hp"):
        hp = (int(lead.get("hp") or 0), int(lead["max_hp"]))

    # Every member down. One definition, shared with the watch that writes the
    # model-facing note, so the receipt and the payload can never disagree about
    # whether a whiteout happened. Note that this flag marks the *frame*, not the
    # event: the party stays down across every batch the faint takes to resolve,
    # which is why anything counting whiteouts has to count rising edges. See
    # `whiteout_events`.
    whiteout = party_is_down(state)

    return {
        "tool": tool,
        "presses": int(presses or 0),
        "map_name": (state.get("map") or {}).get("map_name") or "",
        "pos": pos,
        "moved": (outcome or {}).get("moved"),
        "blocked_after": (outcome or {}).get("blocked_after"),
        "hp": hp,
        "party_size": len(party),
        "milestone_ids": None if milestone_ids is None else tuple(milestone_ids),
        "exit_code": int(exit_code or 0),
        "reloaded": bool(reloaded),
        "whiteout": whiteout,
        "extra": dict(extra or {}),
    }


__all__ = [
    "RECENT_RECEIPT_WINDOW",
    "RUN_POINTER_FILENAME",
    "RunHandle",
    "RunRecorder",
    "config_hash",
    "harness_sha",
    "receipt_from_batch",
]
