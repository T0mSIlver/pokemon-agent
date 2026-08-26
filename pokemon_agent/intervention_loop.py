"""Stop the player, think, hand back one instruction, give the slot back.

:mod:`pokemon_agent.interventions` decides *when* — six detectors and a policy,
all pure. :mod:`pokemon_agent.slots` does the dangerous part — saving the
player's KV cache, freeing the slot and putting the cache back. Neither had a
caller. This module is the caller: it runs after every action batch, asks the
policy, and when the policy says stop it borrows the slot, runs one thinking
session on a small prompt, and delivers the answer to the live session down the
same path ``POST /supervisor/steer`` uses.

Off by default
--------------

A run that has been going for days is somebody's data, and an untested
intervention fired into it costs a swap of the player's entire context at best.
So :class:`InterventionRunner` starts disabled and stays that way unless the
harness is explicitly told otherwise. Enabling it takes effect from the next
batch; nothing about the run has to be restarted or rewound, and a run adopted
mid-flight is scored exactly the same either way.

The failure that matters
------------------------

Between the erase and the restore the player's whole context exists only as a
file on the model box. :func:`pokemon_agent.slots.borrowed_slot` raises
:class:`~pokemon_agent.slots.SlotLost` when it cannot put it back. That is not
an intervention failing, it is the run's memory being on the floor, so it stops
the intervention system for the rest of the session, names the file it is
stranded in, and is reported by ``/health`` and by the supervisor's own state
rather than being logged and forgotten.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from pokemon_agent.bench.registry import Receipt
from pokemon_agent.interventions import (
    Fact,
    InterventionPolicy,
    Trigger,
    build_prompt,
    check_advice,
    harness_facts,
    refusal_note,
    standing_on,
)
from pokemon_agent.slots import SlotClient, SlotError, SlotLost, borrowed_slot

#: The box the player runs on: llama-server in router mode, ``--parallel 1``.
DEFAULT_SLOT_BASE_URL = "http://192.168.1.183:8090"
DEFAULT_SLOT_MODEL = "qwen38-27b"

#: What the player's borrowed cache is written to under ``--slot-save-path``.
DEFAULT_SLOT_FILENAME = "pokemon-player-slot0.bin"

#: The thinking session is the whole point of the swap, so it thinks hard. It is
#: also answering a 20-line prompt, not reading a session, so it stays cheap.
DEFAULT_THINKING = "high"
DEFAULT_TIMEOUT_SECONDS = 300.0

#: An answer is handed to the player as one instruction; past this it stops
#: being an instruction and starts being a wall of text it will skim.
ANSWER_LIMIT = 1200

#: One line per intervention, next to the receipts it was decided from.
JOURNAL_FILENAME = "interventions.jsonl"

#: How many recent interventions the status payload carries.
HISTORY_LIMIT = 20

#: Where a delivered answer shows up in the supervisor's ordered stream.
STREAM_SOURCE = "intervention"

#: How long to wait for the player to stop generating before taking its slot,
#: how many times to try handing it back, and how long to back off between those
#: tries. The restore is the half that must not give up early.
SLOT_WAIT_SECONDS = 300.0
SLOT_RESTORE_ATTEMPTS = 3
SLOT_BACKOFF_SECONDS = 2.0

#: Advice: prompt in, one instruction out. Deliberately synchronous — it runs
#: inside the borrowed slot, on a worker thread, so the restore in
#: ``borrowed_slot``'s ``finally`` cannot be skipped by an await that never
#: resumes.
Advise = Callable[[str], str]
Deliver = Callable[[str], Awaitable[Any]]
Notify = Callable[[Mapping[str, Any]], Awaitable[Any]]

#: The live frame, the explored map and the current goal, for the facts that
#: need more than the receipts. Returning ``None`` is a normal answer: the map
#: graph half of the facts is computed without it.
Observe = Callable[[], Optional[Mapping[str, Any]]]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _log(message: str) -> None:
    print(f"[intervention] {message}")


def build_thinking_command(
    pi_binary: str,
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    thinking: str = DEFAULT_THINKING,
) -> list[str]:
    """One-shot ``pi --print`` in thinking mode, with no tools and no session.

    No tools on purpose: everything the thinker is allowed to know is in the
    prompt, so it cannot spend the borrowed slot reading files.
    """

    command = [pi_binary, "--mode", "json", "--print", "--thinking", thinking]
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    command.extend(
        [
            "-ne",
            "-ns",
            "-nc",
            "-np",
            "--no-themes",
            "--offline",
            "--no-session",
            "--tools",
            "",
            prompt,
        ]
    )
    return command


def pi_thinker(
    pi_binary: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    thinking: str = DEFAULT_THINKING,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: Optional[Path] = None,
) -> Advise:
    """An :data:`Advise` that shells out to ``pi``. Blocking, by design."""

    from pokemon_agent.critic import parse_final_text

    def advise(prompt: str) -> str:
        command = build_thinking_command(
            pi_binary, prompt, provider=provider, model=model, thinking=thinking
        )
        completed = subprocess.run(  # noqa: S603 — argv list, no shell
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd else None,
        )
        answer = parse_final_text(completed.stdout)
        if not answer:
            detail = (completed.stderr or "").strip()[-400:]
            raise RuntimeError(
                f"thinking session produced no text (exit {completed.returncode})"
                + (f": {detail}" if detail else "")
            )
        return answer

    return advise


def live_observation() -> Optional[dict[str, Any]]:
    """The live frame, if this process happens to be the server holding one.

    The loop runs in-process inside the API server, which is the only thing
    that has the emulator, the explored-map store and the current objective.
    The server cannot be imported from here — it imports this module — so this
    reads one that is *already* loaded and takes nothing if there is none. In a
    test or a CLI that is nothing at all; in a live run it is the frame's own
    collision, which is what turns "where have I not looked" from a guess into
    an answer.

    Every step is guarded and every failure is the same failure: no live half,
    map-graph facts only.
    """

    module = sys.modules.get("pokemon_agent.server")
    if module is None:
        return None
    try:
        runtime = getattr(module, "_runtime", None)
        bundle = getattr(runtime, "live_bundle", None) or getattr(runtime, "latest_bundle", None)
        if not isinstance(bundle, Mapping):
            return None
        observation: dict[str, Any] = dict(bundle)

        snapshot = (bundle.get("navigation") or {}).get("snapshot") or {}
        store = getattr(module, "_explored_maps", None)
        map_id = snapshot.get("map_id")
        if store is not None and map_id is not None:
            observation["explored"] = store.grid(int(map_id))

        goal = getattr(getattr(module, "_supervisor", None), "goal", "")
        if isinstance(goal, str) and goal.strip():
            observation["goal"] = goal.strip()
        return observation
    except Exception as exc:  # noqa: BLE001 — facts degrade, the run does not
        _log(f"no live observation for the facts: {type(exc).__name__}: {exc}")
        return None


@dataclass
class InterventionRecord:
    """One firing, and what the run did with it."""

    at: str
    trigger: str
    priority: int
    reason: str
    question: str
    presses_at: int
    payload: dict[str, Any] = field(default_factory=dict)
    #: The harness facts the prompt carried, exactly as the thinker saw them.
    #: An answer that contradicts one of these is a diagnosable failure rather
    #: than a mystery, which is the whole reason they are journalled.
    facts: list[str] = field(default_factory=list)
    answer: str = ""
    delivered: bool = False
    #: Claims :func:`check_advice` disproved against the map data. A non-empty
    #: list means the answer was written but never reached the player, and the
    #: text is still journalled so the refusal can be argued with.
    refused: list[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    slot_saved_tokens: Optional[int] = None
    #: Filled in from the batches that came after: the receipts, presses, tiles
    #: covered and rungs reached once the player had the answer.
    after: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "trigger": self.trigger,
            "priority": self.priority,
            "reason": self.reason,
            "question": self.question,
            "presses_at": self.presses_at,
            "payload": dict(self.payload),
            "facts": list(self.facts),
            "answer": self.answer,
            "delivered": self.delivered,
            "refused": list(self.refused),
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "slot_saved_tokens": self.slot_saved_tokens,
            "after": dict(self.after),
        }


#: How many batches after an intervention are watched to say what it changed.
FOLLOW_UP_BATCHES = 12


class InterventionRunner:
    """The loop: evaluate, borrow, think, steer, record.

    One intervention at a time. A batch that lands while a swap is in flight is
    simply not evaluated: the policy has already recorded the firing, so the
    cooldown is running, and a second detector agreeing with the first is not
    worth a second round-trip of the player's KV cache.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        policy: Optional[InterventionPolicy] = None,
        advise: Optional[Advise] = None,
        deliver: Optional[Deliver] = None,
        slot_client: Optional[SlotClient] = None,
        slot_filename: str = DEFAULT_SLOT_FILENAME,
        slot_wait_seconds: float = SLOT_WAIT_SECONDS,
        slot_restore_attempts: int = SLOT_RESTORE_ATTEMPTS,
        slot_backoff_seconds: float = SLOT_BACKOFF_SECONDS,
        journal_path: Optional[Path] = None,
        notify: Optional[Notify] = None,
        observe: Optional[Observe] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.policy = policy if policy is not None else InterventionPolicy()
        self.advise = advise
        self.deliver = deliver
        self.slot_client = slot_client
        self.slot_filename = slot_filename
        self.slot_wait_seconds = float(slot_wait_seconds)
        self.slot_restore_attempts = max(1, int(slot_restore_attempts))
        self.slot_backoff_seconds = float(slot_backoff_seconds)
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.notify = notify
        # Defaults to whatever the running server has, and to nothing anywhere
        # else. Pass one explicitly to pin it, including ``lambda: None``.
        self.observe: Observe = observe if observe is not None else live_observation

        self.history: deque[InterventionRecord] = deque(maxlen=HISTORY_LIMIT)
        self.fired: int = 0
        self.delivered: int = 0
        #: Answers written but held back because the map data contradicted
        #: them. Counted apart from `failed`: nothing broke, the thinker was
        #: wrong, and the two need different fixes.
        self.refused: int = 0
        self.failed: int = 0
        self.disabled_reason: Optional[str] = None
        self.slot_lost: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self._busy: bool = False
        self._pending: Optional[InterventionRecord] = None
        self._pending_seq: int = 0

    # -- reporting -------------------------------------------------------

    @property
    def active(self) -> bool:
        """Enabled, not disabled by a failure, and able to reach both ends."""

        return bool(self.enabled and self.disabled_reason is None and self.advise and self.deliver)

    #: Fields worth carrying in a liveness check. The full record holds the
    #: prompt, the facts block and the whole answer, which took `/health` to
    #: 21,688 bytes on one firing and left it at 2,531 of 3,256 bytes -- 78% of a
    #: health check -- at rest. Everything omitted here is still on
    #: `/dashboard/state`, in `records()`, and in the journal on disk.
    LAST_SUMMARY_FIELDS = (
        "at",
        "trigger",
        "priority",
        "delivered",
        "presses_at",
        "duration_seconds",
        "error",
    )

    def status(self) -> dict[str, Any]:
        last = self.history[-1] if self.history else None
        if last is not None:
            full = last.to_dict()
            last_summary: Optional[dict[str, Any]] = {
                key: full[key] for key in self.LAST_SUMMARY_FIELDS if key in full
            }
        else:
            last_summary = None
        return {
            "enabled": self.enabled,
            "active": self.active,
            "busy": self._busy,
            "fired": self.fired,
            "delivered": self.delivered,
            "refused": self.refused,
            "failed": self.failed,
            "remaining_this_session": self.policy.remaining(),
            "cooldown_presses": self.policy.cooldown_presses,
            "max_per_session": self.policy.max_per_session,
            "disabled_reason": self.disabled_reason,
            "slot_lost": dict(self.slot_lost) if self.slot_lost else None,
            "last_error": self.last_error,
            "last": last_summary,
        }

    def records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.history]

    # -- the loop --------------------------------------------------------

    async def after_batch(
        self,
        receipts: Sequence[Receipt],
        *,
        state: Optional[Mapping[str, Any]] = None,
        total_presses: Optional[int] = None,
        state_summary: str = "",
        milestone_summary: str = "",
        observation: Optional[Mapping[str, Any]] = None,
        goal: str = "",
    ) -> Optional[InterventionRecord]:
        """Look at what the run just did, and stop it to think if it should.

        Returns the record when one fired, ``None`` every other time — including
        every time the flag is off, which is the common case and costs one
        boolean.
        """

        self._observe_follow_up(receipts, total_presses)
        if not self.active or self._busy:
            return None

        spent = self._total(receipts, total_presses)
        trigger = self.policy.evaluate(receipts, state or {}, total_presses=spent)
        if trigger is None:
            return None

        # Recorded before the swap, not after: the cooldown has to start at the
        # decision, or a slow thinking session lets a second one queue behind it.
        self.policy.record(trigger, spent)
        self.fired += 1
        record = InterventionRecord(
            at=_utc_now(),
            trigger=trigger.name,
            priority=trigger.priority,
            reason=trigger.reason,
            question=trigger.question,
            presses_at=spent,
            payload=dict(trigger.payload),
        )
        self.history.append(record)

        self._busy = True
        started = time.monotonic()
        try:
            await self._run(
                trigger,
                record,
                receipts,
                state_summary,
                milestone_summary,
                observation,
                goal,
            )
        finally:
            self._busy = False
            record.duration_seconds = round(time.monotonic() - started, 2)
            self._pending = record
            self._pending_seq = receipts[-1].seq if receipts else 0
            self._journal(record)
            await self._announce(record)
        return record

    async def _run(
        self,
        trigger: Trigger,
        record: InterventionRecord,
        receipts: Sequence[Receipt],
        state_summary: str,
        milestone_summary: str,
        observation: Optional[Mapping[str, Any]] = None,
        goal: str = "",
    ) -> None:
        facts = self._facts(trigger, receipts, observation, goal)
        record.facts = [fact.text for fact in facts]
        prompt = build_prompt(
            trigger,
            state_summary=state_summary or "(state unavailable)",
            recent=list(receipts),
            milestone_summary=milestone_summary,
            facts=facts,
        )
        try:
            record.answer = await self._think(prompt, record)
        except SlotLost as exc:
            # The player's context is on disk and not in the model. Everything
            # else here is recoverable; this is not, and it never gets swallowed.
            self._lose_slot(exc)
            record.error = str(exc)
            self.failed += 1
            traceback.print_exc()
            return
        except Exception as exc:  # noqa: BLE001 — a bad intervention is not a bad run
            record.error = f"{type(exc).__name__}: {exc}"
            self.last_error = record.error
            self.failed += 1
            _log(f"{trigger.name}: {record.error}")
            return

        answer = (record.answer or "").strip()[:ANSWER_LIMIT]
        record.answer = answer
        if not answer:
            record.error = "thinking session returned nothing"
            self.last_error = record.error
            self.failed += 1
            return

        claims = self._disproved(answer, receipts, observation)
        if claims:
            record.refused = [str(claim) for claim in claims]
            record.error = refusal_note(claims)
            self.refused += 1
            _log(f"{trigger.name}: not delivering — {record.error}")
            return

        try:
            assert self.deliver is not None
            await self.deliver(answer)
        except Exception as exc:  # noqa: BLE001
            record.error = f"could not deliver: {type(exc).__name__}: {exc}"
            self.last_error = record.error
            self.failed += 1
            _log(record.error)
            return
        record.delivered = True
        self.delivered += 1

    def _disproved(
        self,
        answer: str,
        receipts: Sequence[Receipt],
        observation: Optional[Mapping[str, Any]],
    ) -> tuple:
        """Claims in the answer the map data contradicts, if any.

        Grounding the prompt is not the same as grounding the reply. Of the 13
        interventions this project delivered before ``harness_facts`` existed,
        12 named somewhere you cannot get to the way they said — Route 2 from
        inside Mt Moon, an elevator in Mt Moon B1F, Viridian City south along
        Route 3 — and each one cost the player several hundred presses walking
        into a wall it had been told was a door. The facts block stopped that
        happening; this is what catches it the next time a model answers from a
        walkthrough anyway.

        Same rule as :meth:`_facts`: it must never raise. A checker that cannot
        run is a checker that disproves nothing, not one that blocks the run.
        """

        try:
            here = standing_on(observation, receipts)
            if not here:
                return ()
            return check_advice(answer, here=here)
        except Exception as exc:  # noqa: BLE001
            _log(f"could not check the answer: {type(exc).__name__}: {exc}")
            return ()

    def _facts(
        self,
        trigger: Trigger,
        receipts: Sequence[Receipt],
        observation: Optional[Mapping[str, Any]],
        goal: str,
    ) -> list[Fact]:
        """What the harness itself knows about where the player is standing.

        The one thing this must never do is raise: an intervention that dies
        computing its own context is worse than one that fires with a thin
        prompt, and thin is exactly what an empty list produces.
        """

        if observation is None:
            try:
                observation = self.observe()
            except Exception as exc:  # noqa: BLE001
                _log(f"observation failed, facts will use the map graph only: {exc}")
                observation = None
        try:
            return harness_facts(trigger, recent=receipts, observation=observation, goal=goal)
        except Exception as exc:  # noqa: BLE001
            _log(f"could not compute the harness facts: {type(exc).__name__}: {exc}")
            return []

    async def _think(self, prompt: str, record: InterventionRecord) -> str:
        """Run the thinking session, in the player's slot if there is one.

        The borrow and the session are one blocking call on a worker thread, so
        the restore in ``borrowed_slot``'s ``finally`` runs on the same stack
        that took the slot away. Nothing between the erase and the restore is
        allowed to be a coroutine that might never be resumed.
        """

        advise = self.advise
        assert advise is not None
        client = self.slot_client
        if client is None:
            return await asyncio.get_running_loop().run_in_executor(None, advise, prompt)

        filename = self.slot_filename

        def borrow_and_think() -> str:
            with borrowed_slot(
                client,
                filename,
                wait=self.slot_wait_seconds,
                restore_attempts=self.slot_restore_attempts,
                backoff=self.slot_backoff_seconds,
            ) as saved:
                # None when the slot save is unavailable. The thinking session
                # still runs; the player just re-prefills afterwards instead of
                # having its KV cache handed back.
                record.slot_saved_tokens = saved.n_saved if saved else 0
                return advise(prompt)

        return await asyncio.get_running_loop().run_in_executor(None, borrow_and_think)

    # -- failure ---------------------------------------------------------

    def _lose_slot(self, exc: SlotLost) -> None:
        self.slot_lost = {
            "at": _utc_now(),
            "filename": exc.filename,
            "message": str(exc),
        }
        self.disable(
            "The player's KV slot could not be restored after an intervention. "
            f"Its context is on the model box as {exc.filename!r}. Interventions "
            "are off for the rest of this session."
        )

    def disable(self, reason: str) -> None:
        """Stop firing, and say why. Never silently."""

        self.disabled_reason = reason
        self.last_error = reason
        _log(reason)

    def enable(self) -> None:
        """Turn the loop on mid-run. Clears nothing that a failure recorded."""

        self.enabled = True

    # -- bookkeeping -----------------------------------------------------

    @staticmethod
    def _total(receipts: Sequence[Receipt], total_presses: Optional[int]) -> int:
        if total_presses is not None:
            return int(total_presses)
        return sum(receipt.presses for receipt in receipts)

    def _observe_follow_up(self, receipts: Sequence[Receipt], total_presses: Optional[int]) -> None:
        """What the run did after the last answer landed.

        An intervention that is never checked against what happened next is an
        opinion. This is the cheap version of the check: how much the run spent,
        whether it moved, and whether it reached anything.
        """

        record = self._pending
        if record is None or not receipts:
            return
        following = [item for item in receipts if item.seq > self._pending_seq]
        if not following:
            return
        window = following[:FOLLOW_UP_BATCHES]
        record.after = {
            "batches": len(window),
            "presses": sum(item.presses for item in window),
            "moved": sum(item.moved or 0 for item in window),
            "blocked": sum(1 for item in window if item.blocked),
            "milestones": sorted({m for item in window for m in item.milestones_new}),
            "map": window[-1].map_name,
            "presses_total": self._total(receipts, total_presses),
        }
        if len(window) >= FOLLOW_UP_BATCHES:
            self._pending = None

    def _journal(self, record: InterventionRecord) -> None:
        path = self.journal_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError as exc:  # noqa: BLE001 — the journal is a record, not the run
            _log(f"could not write the intervention journal: {exc}")

    async def _announce(self, record: InterventionRecord) -> None:
        sink = self.notify
        if sink is None:
            return
        try:
            await sink({**record.to_dict(), "status": self.status()})
        except Exception as exc:  # noqa: BLE001
            _log(f"could not announce the intervention: {exc}")


def build_slot_client(
    *,
    base_url: str = DEFAULT_SLOT_BASE_URL,
    model: str = DEFAULT_SLOT_MODEL,
) -> SlotClient:
    """The router-mode client for the one slot the player lives in."""

    return SlotClient(base_url=base_url, model=model)


__all__ = [
    "ANSWER_LIMIT",
    "Observe",
    "live_observation",
    "DEFAULT_SLOT_BASE_URL",
    "DEFAULT_SLOT_FILENAME",
    "DEFAULT_SLOT_MODEL",
    "DEFAULT_THINKING",
    "FOLLOW_UP_BATCHES",
    "JOURNAL_FILENAME",
    "STREAM_SOURCE",
    "Advise",
    "Deliver",
    "InterventionRecord",
    "InterventionRunner",
    "Notify",
    "SlotError",
    "SlotLost",
    "build_slot_client",
    "build_thinking_command",
    "pi_thinker",
]
