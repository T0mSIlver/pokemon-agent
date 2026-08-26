"""Stopping the player to think, without a GPU anywhere near the test.

The slot API is a fake that counts save / erase / restore, and the thinking
session is a function that returns a string. What is under test is the loop:
whether it stays quiet when it is told to, whether it fires once when it should,
and what it does when the player's context does not come back.
"""

import json

import pytest

from pokemon_agent.bench.registry import Receipt
from pokemon_agent.intervention_loop import (
    InterventionRunner,
    build_thinking_command,
)
from pokemon_agent.interventions import (
    InterventionPolicy,
    RepeatedFailure,
    StalledMilestones,
)
from pokemon_agent.slots import SaveResult, SlotError, SlotLost

ANSWER = "Walk left four tiles, then up two, and talk to the guard."


class FakeSlot:
    """The one slot the player lives in, as a list of things that happened to it."""

    slot_id = 0

    def __init__(self, *, restore_fails: bool = False, saved_tokens: int = 41_000) -> None:
        self.calls: list[str] = []
        self.restore_fails = restore_fails
        self.saved_tokens = saved_tokens

    def wait_idle(self, timeout: float = 300.0, poll: float = 2.0) -> bool:
        self.calls.append("wait_idle")
        return True

    def save(self, filename: str) -> SaveResult:
        self.calls.append("save")
        return SaveResult(
            filename=filename, n_saved=self.saved_tokens, n_written=1 << 20, save_ms=12.0
        )

    def erase(self) -> int:
        self.calls.append("erase")
        return self.saved_tokens

    def restore(self, filename: str):
        self.calls.append("restore")
        if self.restore_fails:
            raise SlotError("router answered 500")
        raise AssertionError("unreachable: a passing restore is patched in per test")


class RestoringSlot(FakeSlot):
    def restore(self, filename: str):
        from pokemon_agent.slots import RestoreResult

        self.calls.append("restore")
        return RestoreResult(
            filename=filename, n_restored=self.saved_tokens, n_read=1 << 20, restore_ms=9.0
        )


def receipt(seq: int, **kwargs) -> Receipt:
    fields = {
        "seq": seq,
        "t": 1000.0 + seq,
        "presses": kwargs.pop("presses", 10),
        "map_name": kwargs.pop("map_name", "Route 3"),
        "pos": kwargs.pop("pos", (4, 9)),
        "moved": kwargs.pop("moved", 3),
        "party_size": 1,
        "hp": (20, 22),
        "tool": kwargs.pop("tool", "action"),
    }
    fields.update(kwargs)
    return Receipt(**fields)


def failing_pair() -> list[Receipt]:
    """Two identical failures in a row — what `repeated_failure` fires on."""
    return [
        receipt(0),
        receipt(1, exit_code=1, presses=0, moved=None),
        receipt(2, exit_code=1, presses=0, moved=None),
    ]


def make_runner(**kwargs) -> tuple[InterventionRunner, dict]:
    log: dict = {"prompts": [], "steers": [], "notices": []}

    def advise(prompt: str) -> str:
        log["prompts"].append(prompt)
        return ANSWER

    async def deliver(text: str) -> None:
        log["steers"].append(text)

    async def notify(payload) -> None:
        log["notices"].append(dict(payload))

    runner = InterventionRunner(
        enabled=kwargs.pop("enabled", True),
        policy=kwargs.pop(
            "policy",
            InterventionPolicy(detectors=(RepeatedFailure(),), cooldown_presses=100),
        ),
        advise=kwargs.pop("advise", advise),
        deliver=deliver,
        notify=notify,
        slot_restore_attempts=1,
        slot_backoff_seconds=0.0,
        **kwargs,
    )
    return runner, log


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_a_fresh_runner_is_off():
    """A live run is somebody's data. Nothing fires without being told to."""
    assert InterventionRunner().enabled is False
    assert InterventionRunner().active is False


async def test_the_loop_stays_silent_while_the_flag_is_off():
    runner, log = make_runner(enabled=False)

    result = await runner.after_batch(failing_pair(), total_presses=900)

    assert result is None
    assert log == {"prompts": [], "steers": [], "notices": []}
    assert runner.status()["fired"] == 0


async def test_turning_the_flag_on_mid_run_takes_effect_on_the_next_batch():
    runner, log = make_runner(enabled=False)
    await runner.after_batch(failing_pair(), total_presses=900)

    runner.enable()
    await runner.after_batch(failing_pair(), total_presses=900)

    assert log["steers"] == [ANSWER]


async def test_a_runner_with_nowhere_to_send_an_answer_is_not_active():
    runner = InterventionRunner(enabled=True, advise=lambda prompt: ANSWER)

    assert runner.active is False
    assert await runner.after_batch(failing_pair(), total_presses=900) is None


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------


async def test_a_trigger_produces_exactly_one_steer_message():
    runner, log = make_runner()

    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert record is not None
    assert record.trigger == "repeated_failure"
    assert record.delivered is True
    assert log["steers"] == [ANSWER]
    assert len(log["prompts"]) == 1
    assert runner.status()["delivered"] == 1


async def test_the_prompt_carries_the_reason_and_the_recent_batches():
    runner, log = make_runner()

    await runner.after_batch(
        failing_pair(),
        total_presses=900,
        state_summary="map: Route 3\nx: 4",
        milestone_summary="900 presses spent so far.",
    )

    prompt = log["prompts"][0]
    assert "failed 2 times in a row" in prompt
    assert "map: Route 3" in prompt
    assert "900 presses spent so far." in prompt


async def test_the_slot_is_saved_freed_and_handed_back():
    slot = RestoringSlot()
    runner, log = make_runner(slot_client=slot, slot_filename="player.bin")

    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert slot.calls == ["wait_idle", "save", "erase", "restore"]
    assert record.slot_saved_tokens == 41_000
    assert log["steers"] == [ANSWER]


async def test_a_save_that_wrote_nothing_costs_a_reprefill_not_the_intervention():
    # The save is an optimisation. On this box it fails server-side -- the KV
    # cache is quantised to q8_0 and cannot be serialised -- so refusing to
    # intervene without it would mean never intervening at all. Without a save
    # the player re-prefills, which is wall clock, not a lost run.
    slot = RestoringSlot(saved_tokens=0)
    runner, log = make_runner(slot_client=slot)

    await runner.after_batch(failing_pair(), total_presses=900)

    assert log["steers"], "the intervention must still run and deliver"
    # Nothing was stored, so nothing may be given away or restored.
    assert "erase" not in slot.calls
    assert "restore" not in slot.calls
    assert runner.active is True


async def test_a_long_answer_is_cut_to_one_instruction():
    runner, log = make_runner(advise=lambda prompt: "go left. " * 400)

    await runner.after_batch(failing_pair(), total_presses=900)

    assert len(log["steers"][0]) <= 1200


async def test_an_empty_answer_is_not_delivered():
    runner, log = make_runner(advise=lambda prompt: "   ")

    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert log["steers"] == []
    assert record.delivered is False
    assert runner.status()["failed"] == 1


async def test_a_thinking_session_that_blows_up_is_recorded_and_the_run_continues():
    def explode(prompt: str) -> str:
        raise RuntimeError("pi exited 3")

    runner, log = make_runner(advise=explode)

    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert "pi exited 3" in record.error
    assert log["steers"] == []
    assert runner.active is True


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_the_cooldown_holds_between_two_triggers():
    runner, log = make_runner(
        policy=InterventionPolicy(detectors=(RepeatedFailure(),), cooldown_presses=600)
    )

    await runner.after_batch(failing_pair(), total_presses=900)
    await runner.after_batch(failing_pair(), total_presses=1_100)

    assert log["steers"] == [ANSWER]

    await runner.after_batch(failing_pair(), total_presses=1_600)

    assert log["steers"] == [ANSWER, ANSWER]


async def test_the_session_budget_holds():
    runner, log = make_runner(
        policy=InterventionPolicy(
            detectors=(RepeatedFailure(),), cooldown_presses=10, max_per_session=2
        )
    )

    for spent in (900, 1_000, 1_100, 1_200):
        await runner.after_batch(failing_pair(), total_presses=spent)

    assert len(log["steers"]) == 2
    assert runner.status()["remaining_this_session"] == 0


async def test_the_highest_priority_detector_wins():
    """Being lost wastes a run; a failing command is how runs are lost faster."""
    runner, log = make_runner(
        policy=InterventionPolicy(detectors=(StalledMilestones(), RepeatedFailure()))
    )
    window = [receipt(index, presses=40) for index in range(30)] + failing_pair()[1:]

    record = await runner.after_batch(window, total_presses=1_400)

    assert record.trigger == "repeated_failure"


# ---------------------------------------------------------------------------
# Losing the slot
# ---------------------------------------------------------------------------


async def test_a_lost_slot_disables_the_system_and_says_so_loudly():
    """Between the erase and the restore the run's memory is only a file."""
    slot = FakeSlot(restore_fails=True)
    runner, log = make_runner(slot_client=slot, slot_filename="player.bin")

    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert "could not restore" in record.error
    assert runner.slot_lost["filename"] == "player.bin"
    assert runner.active is False
    assert runner.disabled_reason and "player.bin" in runner.disabled_reason

    status = runner.status()
    assert status["slot_lost"]["filename"] == "player.bin"
    assert status["disabled_reason"]
    assert log["notices"][-1]["status"]["slot_lost"]["filename"] == "player.bin"


async def test_nothing_fires_again_after_the_slot_is_lost():
    slot = FakeSlot(restore_fails=True)
    runner, log = make_runner(
        slot_client=slot,
        policy=InterventionPolicy(detectors=(RepeatedFailure(),), cooldown_presses=0),
    )

    await runner.after_batch(failing_pair(), total_presses=900)
    before = list(slot.calls)
    await runner.after_batch(failing_pair(), total_presses=5_000)

    assert slot.calls == before
    assert log["steers"] == []


async def test_slot_lost_is_not_confused_with_an_ordinary_failure():
    assert issubclass(SlotLost, SlotError)

    slot = FakeSlot(restore_fails=True)
    runner, _ = make_runner(slot_client=slot)
    await runner.after_batch(failing_pair(), total_presses=900)
    lost = runner.slot_lost

    other, _ = make_runner(advise=_raise_slot_error)
    await other.after_batch(failing_pair(), total_presses=900)

    assert lost is not None
    assert other.slot_lost is None
    assert other.active is True


def _raise_slot_error(prompt: str) -> str:
    raise SlotError("the router was swapping models")


# ---------------------------------------------------------------------------
# Recording what happened
# ---------------------------------------------------------------------------


async def test_the_run_after_an_intervention_is_recorded_against_it():
    runner, _ = make_runner()
    window = failing_pair()
    record = await runner.after_batch(window, total_presses=900)

    later = window + [receipt(3, presses=12, milestones_new=("EVENT_BEAT_BROCK",))]
    await runner.after_batch(later, total_presses=912)

    assert record.after["batches"] == 1
    assert record.after["presses"] == 12
    assert record.after["milestones"] == ["EVENT_BEAT_BROCK"]


async def test_every_intervention_lands_in_the_journal(tmp_path):
    journal = tmp_path / "interventions.jsonl"
    runner, _ = make_runner(journal_path=journal)

    await runner.after_batch(failing_pair(), total_presses=900)

    line = json.loads(journal.read_text(encoding="utf-8").strip())
    assert line["trigger"] == "repeated_failure"
    assert line["answer"] == ANSWER
    assert line["delivered"] is True
    assert line["presses_at"] == 900


async def test_a_delivery_that_is_refused_is_recorded_not_hidden():
    runner, log = make_runner()

    async def refuse(text: str) -> None:
        raise RuntimeError("no live session to steer")

    runner.deliver = refuse
    record = await runner.after_batch(failing_pair(), total_presses=900)

    assert record.delivered is False
    assert "no live session" in record.error
    assert runner.status()["last_error"]


async def test_the_status_payload_says_what_the_loop_is_doing():
    runner, _ = make_runner()
    await runner.after_batch(failing_pair(), total_presses=900)

    status = runner.status()

    assert status["enabled"] is True
    assert status["fired"] == 1
    assert status["delivered"] == 1
    assert status["busy"] is False
    assert status["last"]["trigger"] == "repeated_failure"


# ---------------------------------------------------------------------------
# The thinking session's command line
# ---------------------------------------------------------------------------


def test_the_thinking_session_runs_one_shot_with_no_tools_and_no_session():
    command = build_thinking_command(
        "/usr/bin/pi", "what now?", provider="llamacpp", model="qwen38-27b", thinking="high"
    )

    assert command[:6] == ["/usr/bin/pi", "--mode", "json", "--print", "--thinking", "high"]
    assert "--no-session" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[-1] == "what now?"


@pytest.mark.parametrize("flag", ["-ne", "-ns", "-nc", "-np", "--offline"])
def test_the_thinking_session_keeps_the_headless_flags(flag):
    assert flag in build_thinking_command("/usr/bin/pi", "what now?")
