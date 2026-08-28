"""One lock in front of the emulator, held across whole operations.

The emulator is a single mutable machine. Serialising individual *calls* to it
is not enough: an action batch that mutates under the lock, releases it, and
then reads state back describes whatever the next request left behind. Two
concurrent loads produced exactly that — a response naming save A carrying map
and coordinates from B.

:class:`EmulatorCoordinator` fixes the scope rather than the symptom. Every
public method is one transaction: the lock is taken once and held from the
first state read, through the mutation, the settle, the final state read, the
navigation snapshot, the screenshot and any save the runtime writes. Nothing is
broadcast from inside — a transaction returns the events it produced and the
caller emits them after the lock is gone, so a slow WebSocket client can never
hold the emulator.

The blocking work runs as *one* executor call per transaction, so there is no
window between two thread hops either.

This module also owns the action budget. ``wait_1000000000`` used to be a legal
action: the server would tick a billion frames holding the lock while the client
timed out and the executor kept going. Frames are capped per action and per
batch here, and the caps are named constants because ``poke`` mirrors them.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Action budget
# ---------------------------------------------------------------------------

#: How many actions one batch may carry.
MAX_ACTIONS_PER_BATCH = 40

#: Frames one action may run for. 600 frames is ten seconds of game time.
MAX_FRAMES_PER_ACTION = 600

#: Frames one batch may run for in total. 3600 frames is sixty seconds.
MAX_FRAMES_PER_BATCH = 3600

#: press_X holds for 8 frames then waits 12 for the game to process it.
PRESS_FRAMES = 20

#: walk_X uses the same cadence; Gen 1 needs 17 frames for a confirmed tile move.
WALK_FRAMES = 20

#: a_until_dialog_end presses A every 30 frames, at most ten times.
A_UNTIL_DIALOG_END_FRAMES = 300

WALKABLE_BUTTONS = ("up", "down", "left", "right")


class ActionLimitError(ValueError):
    """A batch that asks for more emulator time than one request may have."""


class UnknownActionError(ValueError):
    """An action string the executor has no format for."""


def frames_for_action(action: str) -> int:
    """How many frames *action* will run the emulator for.

    Raises :class:`UnknownActionError` for anything the executor cannot run, so
    a bad token is refused before any button in the batch is pressed rather
    than halfway through it.
    """
    text = str(action).strip().lower()
    if not text:
        raise UnknownActionError("empty action")
    if text == "a_until_dialog_end":
        return A_UNTIL_DIALOG_END_FRAMES

    parts = text.split("_")
    if parts[0] == "press" and len(parts) >= 2 and parts[1]:
        return PRESS_FRAMES
    if parts[0] == "walk" and len(parts) >= 2 and parts[1]:
        return WALK_FRAMES
    if parts[0] == "hold" and len(parts) >= 3:
        return _frame_count(text, parts[-1])
    if parts[0] == "wait" and len(parts) == 2:
        return _frame_count(text, parts[1])
    raise UnknownActionError(unknown_action(text))


#: Moves that do something outside a battle. Named here so an action that looks
#: like one can be pointed at the verb that exists, rather than bounced with a
#: bare parse error — the run that motivated this sent ``use_cut`` and then
#: ``hm_cut``, was told only "Unknown action format" twice, and never tried to
#: use Cut again in 19,000 further calls.
FIELD_MOVE_WORDS = ("cut", "fly", "surf", "strength", "flash", "dig", "teleport")


def unknown_action(text: str) -> str:
    """The refusal for an action name that is not one, naming the ones that are."""
    hint = ""
    if any(word in text for word in FIELD_MOVE_WORDS):
        hint = (
            " A field move is not a button and not a battle action: `poke cut` cuts a "
            "small tree, walking to it first."
        )
    elif "bike" in text or "bicycle" in text:
        hint = " `poke bike` gets on the Bicycle."
    return (
        f"Unknown action format: {text}. Actions are walk_up/down/left/right, "
        f"press_a/b/start/select, hold_<button>_<frames>, wait_<frames>, and "
        f"a_until_dialog_end.{hint}"
    )


def _frame_count(action: str, raw: str) -> int:
    try:
        frames = int(raw)
    except ValueError:
        raise UnknownActionError(unknown_action(action)) from None
    if frames <= 0:
        raise UnknownActionError(f"{action}: a frame count must be a positive integer")
    return frames


def presses_for_action(action: str) -> int:
    """Buttons *action* sends. The run metric is counted in buttons, not frames."""
    text = str(action).strip().lower()
    if text == "a_until_dialog_end":
        return 0  # counted as it goes: the loop stops as soon as the dialog does
    parts = text.split("_")
    if parts[0] in ("press", "walk", "hold"):
        return 1
    return 0


def validate_action_batch(actions: Sequence[str]) -> int:
    """Check *actions* against every cap and return the frames the batch costs.

    Raises :class:`ActionLimitError` naming the limit that was exceeded and what
    was asked for, or :class:`UnknownActionError` for an unrunnable token.
    """
    count = len(actions)
    if count > MAX_ACTIONS_PER_BATCH:
        raise ActionLimitError(
            f"A batch may hold at most {MAX_ACTIONS_PER_BATCH} actions; this one has "
            f"{count}. Send it in smaller batches and look at the frame in between."
        )
    total = 0
    for action in actions:
        frames = frames_for_action(action)
        if frames > MAX_FRAMES_PER_ACTION:
            raise ActionLimitError(
                f"{str(action).strip().lower()!r} asks for {frames} frames; one action may "
                f"run at most {MAX_FRAMES_PER_ACTION} frames ({MAX_FRAMES_PER_ACTION // 60}s). "
                "The emulator is single-threaded: a long action blocks every other request."
            )
        total += frames
    if total > MAX_FRAMES_PER_BATCH:
        raise ActionLimitError(
            f"This batch asks for {total} frames; one batch may run at most "
            f"{MAX_FRAMES_PER_BATCH} frames ({MAX_FRAMES_PER_BATCH // 60}s). "
            "Split it and check the result in between."
        )
    return total


def batch_within_budget(actions: Sequence[str], budget_frames: int) -> list[str]:
    """The longest prefix of *actions* that fits in *budget_frames* and the caps."""
    chosen: list[str] = []
    spent = 0
    for action in actions:
        if len(chosen) >= MAX_ACTIONS_PER_BATCH:
            break
        frames = frames_for_action(action)
        if spent + frames > min(budget_frames, MAX_FRAMES_PER_BATCH):
            break
        chosen.append(action)
        spent += frames
    return chosen


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

#: Frames a load may spend waiting for the game to stop moving. A save captured
#: mid-transition reports the *previous* map's geometry until the transition
#: finishes, and recording that permanently corrupts the explored map.
SETTLE_MAX_FRAMES = 600
SETTLE_QUIET_FRAMES = 30


class EmulatorCoordinator:
    """Serialises whole operations on the emulator, not individual calls.

    ``ops`` is the plumbing the transactions are built from — the emulator, the
    memory reader, the state read, the action executor and the runtime refresh.
    Every attribute is read at call time, so a caller that swaps the emulator or
    patches a sync helper on the owning module is picked up by the next
    transaction rather than by the next process.
    """

    def __init__(self, ops: Any) -> None:
        self._ops = ops

    # -- plumbing -----------------------------------------------------------

    async def _locked(self, func: Callable, *args, **kwargs):
        """Run one blocking transaction body with the emulator lock held."""
        lock = getattr(self._ops, "lock", None)
        loop = asyncio.get_running_loop()
        call = partial(func, *args, **kwargs)
        if lock is None:  # not started yet — the tests and startup path
            return await loop.run_in_executor(None, call)
        async with lock:
            return await loop.run_in_executor(None, call)

    async def run(self, func: Callable, *args, **kwargs):
        """One blocking emulator call under the lock. For reads with no follow-up."""
        return await self._locked(func, *args, **kwargs)

    def _settle(self) -> bool:
        """Let the game finish whatever it is in the middle of.

        ``Emulator.settle`` is being added alongside this module; until it lands
        an emulator without it is treated as settled, which is exactly the
        behaviour that was there before.
        """
        emulator = self._ops.emulator
        settle = getattr(emulator, "settle", None)
        if settle is None:
            return True
        try:
            return bool(settle(max_frames=SETTLE_MAX_FRAMES, quiet_frames=SETTLE_QUIET_FRAMES))
        except NotImplementedError:
            # A backend that cannot tell when the game has come to rest is no
            # worse off than before settling existed.
            return True
        except TypeError:  # a settle() with a different signature is still a settle
            try:
                return bool(settle())
            except NotImplementedError:
                return True

    def _observe(self, *, reason: str, source: str, **kwargs) -> dict:
        """Refresh the workspace and read the state back, still under the lock."""
        ops = self._ops
        result = ops.refresh_bundle(reason=reason, source=source, **kwargs) or {}
        bundle = result.get("bundle") or {}
        state_after = bundle.get("state")
        if state_after is None:
            # The refresh already read the state; this only runs when there was
            # no runtime to refresh at all, and a state read that fails there
            # must not lose the mutation that has already happened.
            try:
                state_after = ops.state_dict()
            except Exception:  # noqa: BLE001
                state_after = {}
        return {
            "bundle": bundle,
            "events": list(result.get("events") or []),
            "state_after": state_after,
        }

    # -- transactions -------------------------------------------------------

    async def act_and_observe(
        self,
        actions: list[str],
        *,
        source: str,
        reason: str,
    ) -> dict:
        """Execute one batch and read the result back without releasing the lock.

        Holds the lock across: the in-battle safety check, the state read that
        the ``action`` event carries, the batch itself, the workspace refresh
        (final state read, navigation snapshot, explored-map record, both frame
        PNGs, any auto-save) and the state read the ``action_result`` event
        carries.
        """

        def _transaction() -> dict:
            ops = self._ops
            ops.reject_unsafe_battle_actions(actions)
            state_before = ops.state_dict()
            outcome = ops.execute_batch(actions)
            observed = self._observe(reason=reason, source=source, requested_actions=list(actions))
            return {
                "state_before": state_before,
                "outcome": outcome,
                "actions_executed": outcome["executed"],
                **observed,
            }

        return await self._locked(_transaction)

    async def load_settle_and_observe(
        self,
        *,
        path: str,
        reason: str,
        source: str = "load",
    ) -> dict:
        """Restore a save, let it settle, and observe — all under one lock.

        A state captured during a map transition reads as the map it is leaving:
        the wrong dimensions, the wrong warps, the wrong coordinates. Recording
        that is not recoverable, because the explored map takes dimensions and
        warps monotonically. So the transition is allowed to finish first, and
        if it will not finish nothing is published or auto-saved at all.
        """

        def _transaction() -> dict:
            ops = self._ops
            ops.emulator.load_state(str(path))
            settled = self._settle()
            if not settled:
                return {
                    "settled": False,
                    "bundle": {"state": ops.state_dict(), "navigation": {}, "screen_text": {}},
                    "events": [],
                    "state_after": ops.state_dict(),
                }
            return {"settled": True, **self._observe(reason=reason, source=source)}

        return await self._locked(_transaction)

    async def save_and_observe(
        self,
        *,
        path: str,
        reason: str,
        explicit_save: Optional[dict] = None,
        source: str = "save",
    ) -> dict:
        """Write a save file and observe the state it captured, under one lock.

        The observation attached to the response then describes the frame that
        was actually saved, not whatever the next request has already done.
        """

        def _transaction() -> dict:
            self._ops.emulator.save_state(str(path))
            return self._observe(reason=reason, source=source, explicit_save=explicit_save)

        return await self._locked(_transaction)

    async def battle_and_observe(
        self,
        *,
        func: Callable[..., dict],
        args: Iterable[Any] = (),
        reason: str,
        source: str = "battle",
    ) -> dict:
        """Run one battle command and observe it, under one lock.

        A battle command is a menu walk: reading the cursor, pressing, reading
        it again. Every one of those reads has to see the machine the previous
        press left, so the whole sequence is one transaction.
        """

        def _transaction() -> dict:
            ops = self._ops
            state_before = ops.state_dict()
            outcome = func(*args)
            observed = self._observe(
                reason=reason, source=source, requested_actions=list(outcome.get("actions") or [])
            )
            return {"state_before": state_before, "outcome": outcome, **observed}

        return await self._locked(_transaction)

    async def observe_only(self, *, reason: str, source: str) -> dict:
        """Refresh the workspace from the current frame without touching input."""
        return await self._locked(partial(self._observe, reason=reason, source=source))


__all__ = [
    "A_UNTIL_DIALOG_END_FRAMES",
    "ActionLimitError",
    "EmulatorCoordinator",
    "MAX_ACTIONS_PER_BATCH",
    "MAX_FRAMES_PER_ACTION",
    "MAX_FRAMES_PER_BATCH",
    "PRESS_FRAMES",
    "SETTLE_MAX_FRAMES",
    "SETTLE_QUIET_FRAMES",
    "UnknownActionError",
    "WALK_FRAMES",
    "batch_within_budget",
    "frames_for_action",
    "presses_for_action",
    "validate_action_batch",
]
