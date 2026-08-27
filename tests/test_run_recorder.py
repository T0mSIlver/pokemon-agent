"""The scoreboard's writer: one run, many sessions, and a bill that never resets.

Nothing here needs an emulator, a model or a GPU. The milestone oracle is a
callable the recorder is handed, so a test can say what the game currently
satisfies by assigning to a set.
"""

from pathlib import Path

import pytest

from pokemon_agent.bench.metrics import compute
from pokemon_agent.bench.registry import RECEIPTS_FILENAME, STATUS_FINISHED, RunRegistry
from pokemon_agent.run_recorder import (
    RUN_POINTER_FILENAME,
    RunRecorder,
    config_hash,
    harness_sha,
    receipt_from_batch,
)


class FakeOracle:
    """The milestone reader, as a set somebody else can edit mid-test."""

    def __init__(self, *ids: str) -> None:
        self.ids: set[str] = set(ids)
        self.calls = 0

    async def __call__(self) -> frozenset:
        self.calls += 1
        return frozenset(self.ids)


def make_recorder(tmp_path: Path, oracle: FakeOracle, **kwargs) -> RunRecorder:
    return RunRecorder(tmp_path, milestone_snapshot=oracle, **kwargs)


async def walk(recorder: RunRecorder, presses: int, *, milestones=(), **kwargs):
    return await recorder.append(
        tool="action",
        presses=presses,
        map_name=kwargs.pop("map_name", "Route 1"),
        pos=kwargs.pop("pos", (4, 9)),
        moved=kwargs.pop("moved", presses),
        party_size=1,
        hp=(20, 22),
        milestone_ids=milestones,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


async def test_a_run_spans_several_sessions_and_the_presses_accumulate(tmp_path):
    """The token budget ends a session every half hour. It does not end the run."""
    oracle = FakeOracle()

    first = make_recorder(tmp_path, oracle)
    handle_one = await first.begin_session(goal="Reach Pewter", model="qwen")
    await walk(first, 30)
    await walk(first, 12)
    first.close()

    # The watchdog notices the session died and POSTs /supervisor/start again.
    second = make_recorder(tmp_path, oracle)
    handle_two = await second.begin_session(goal="Reach Pewter", model="qwen")
    await walk(second, 8)

    assert handle_two.run_id == handle_one.run_id
    assert handle_two.adopted is True
    assert second.total_presses == 50
    assert second.sessions == 1  # a fresh recorder counts its own sessions

    scored = compute(RunRegistry(tmp_path).load(handle_one.run_id))
    assert scored.total_presses == 50


async def test_a_new_recorder_starts_a_fresh_run_when_the_last_one_finished(tmp_path):
    oracle = FakeOracle()
    first = make_recorder(tmp_path, oracle)
    opened = await first.begin_session(goal="Reach Pewter", model="qwen")
    await walk(first, 10)
    await first.finish_run("objective complete")

    assert not (tmp_path / "runs" / RUN_POINTER_FILENAME).exists()
    assert RunRegistry(tmp_path).load_meta(opened.run_id).status == STATUS_FINISHED

    second = make_recorder(tmp_path, oracle)
    reopened = await second.begin_session(goal="Reach Cerulean", model="qwen")

    assert reopened.run_id != opened.run_id
    assert reopened.adopted is False
    assert second.total_presses == 0


async def test_the_pointer_survives_a_server_restart_and_names_the_open_run(tmp_path):
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    handle = await recorder.begin_session(goal="Reach Pewter", model="qwen")

    pointer = (tmp_path / "runs" / RUN_POINTER_FILENAME).read_text(encoding="utf-8").strip()

    assert pointer == handle.run_id
    assert make_recorder(tmp_path, oracle).read_pointer() == handle.run_id


async def test_a_session_start_never_finishes_the_run(tmp_path):
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    handle = await recorder.begin_session(goal="Reach Pewter", model="qwen")
    recorder.close()
    await recorder.begin_session(goal="Reach Pewter", model="qwen")

    assert RunRegistry(tmp_path).load_meta(handle.run_id).status == "running"


# ---------------------------------------------------------------------------
# Presses
# ---------------------------------------------------------------------------


async def test_a_reload_does_not_reset_the_press_count(tmp_path):
    """Four attempts at a gym cost what four attempts cost."""
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    await recorder.begin_session(goal="Beat Brock", model="qwen")

    for _ in range(4):
        await walk(recorder, 100)
        await recorder.append(tool="load", presses=0, reloaded=True, milestone_ids=())

    oracle.ids.add("EVENT_BEAT_BROCK")
    await walk(recorder, 40, milestones=("EVENT_BEAT_BROCK",))

    assert recorder.total_presses == 440
    assert recorder.presses_to["EVENT_BEAT_BROCK"] == 440

    scored = compute(RunRegistry(tmp_path).load(recorder.run_id))
    assert scored.total_presses == 440
    assert scored.reloads == 4
    assert scored.presses_to["EVENT_BEAT_BROCK"] == 440


async def test_presses_to_records_the_first_attainment_only(tmp_path):
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    await recorder.begin_session(goal="Reach Pewter", model="qwen")

    await walk(recorder, 25, milestones=("EVENT_GOT_STARTER",))
    await walk(recorder, 25, milestones=("EVENT_GOT_STARTER",))
    await walk(recorder, 25, milestones=("EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX"))

    assert recorder.presses_to == {"EVENT_GOT_STARTER": 25, "EVENT_GOT_POKEDEX": 75}
    assert [item["milestone_id"] for item in recorder.attainments] == [
        "EVENT_GOT_STARTER",
        "EVENT_GOT_POKEDEX",
    ]

    scored = compute(RunRegistry(tmp_path).load(recorder.run_id))
    assert scored.presses_to["EVENT_GOT_STARTER"] == 25


async def test_a_milestone_reached_in_a_later_session_is_priced_across_the_whole_run(tmp_path):
    oracle = FakeOracle()
    first = make_recorder(tmp_path, oracle)
    await first.begin_session(goal="Beat Brock", model="qwen")
    await walk(first, 600)
    first.close()

    second = make_recorder(tmp_path, oracle)
    await second.begin_session(goal="Beat Brock", model="qwen")
    oracle.ids.add("EVENT_BEAT_BROCK")
    await walk(second, 8, milestones=("EVENT_BEAT_BROCK",))

    assert second.presses_to["EVENT_BEAT_BROCK"] == 608


async def test_a_checkpoint_the_run_resumed_from_is_history_not_progress(tmp_path):
    """A run started on a four-badge save did not earn those badges in five presses."""
    oracle = FakeOracle("EVENT_GOT_STARTER", "EVENT_BEAT_BROCK")
    recorder = make_recorder(tmp_path, oracle, start_checkpoint="four-badges")
    await recorder.begin_session(goal="Beat Misty", model="qwen")

    await walk(recorder, 5, milestones=("EVENT_GOT_STARTER", "EVENT_BEAT_BROCK"))

    assert recorder.presses_to == {}

    oracle.ids.add("EVENT_BEAT_MISTY")
    await walk(recorder, 20, milestones=tuple(sorted(oracle.ids)))

    assert recorder.presses_to == {"EVENT_BEAT_MISTY": 25}
    assert RunRegistry(tmp_path).load_meta(recorder.run_id).start_checkpoint == "four-badges"


async def test_the_live_count_falls_when_the_bill_cannot(tmp_path):
    """A reload onto an earlier branch hands rungs back. The bill still holds.

    One run lost a badge and five milestones to a burst of loads and
    `milestone_count` did not move a digit across 19,705 receipts — correctly,
    because it is a running maximum and a gym won on the fourth attempt costs
    what all four attempts cost. `milestones_held` is the number beside it.
    """
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    await recorder.begin_session(goal="Beat Misty", model="qwen")

    await walk(recorder, 40, milestones=("EVENT_GOT_STARTER", "EVENT_BEAT_BROCK"))
    peak = await walk(
        recorder,
        60,
        milestones=("EVENT_GOT_STARTER", "EVENT_BEAT_BROCK", "EVENT_BEAT_MISTY"),
    )
    # `poke load before_misty`, which this run did fifteen times.
    rewound = await recorder.append(
        tool="load",
        presses=0,
        reloaded=True,
        milestone_ids=("EVENT_GOT_STARTER",),
    )

    assert (peak.milestone_count, peak.milestones_held) == (3, 3)
    assert rewound.milestone_count == 3  # what the run has spent, which never rewinds
    assert rewound.milestones_held == 1  # what the game is holding, which just did
    assert compute(RunRegistry(tmp_path).load(recorder.run_id)).final_milestone_count == 3


async def test_a_receipt_that_never_read_the_oracle_claims_nothing_about_it(tmp_path):
    """`None` is not zero. A refused batch reads no RAM and is not a regression."""
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    await recorder.begin_session(goal="Beat Misty", model="qwen")
    await walk(recorder, 10, milestones=("EVENT_GOT_STARTER",))

    refused = await recorder.append(tool="action", presses=0, exit_code=1)

    assert refused.milestones_held is None
    assert refused.milestone_count == 1


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


async def test_receipts_survive_a_crash_partway_through_an_append(tmp_path):
    """A hard kill truncates the final line. It must cost that receipt, and no more."""
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    handle = await recorder.begin_session(goal="Reach Pewter", model="qwen")
    await walk(recorder, 40)
    await walk(recorder, 60)
    recorder.close()

    # The process died with half a line in the pipe.
    receipts = tmp_path / "runs" / handle.run_id / RECEIPTS_FILENAME
    with receipts.open("a", encoding="utf-8") as handle_out:
        handle_out.write('{"seq": 3, "presses": 7, "map": "Route')

    resumed = make_recorder(tmp_path, oracle)
    adopted = await resumed.begin_session(goal="Reach Pewter", model="qwen")

    assert adopted.run_id == handle.run_id
    assert resumed.total_presses == 100  # the complete receipts, all of them
    assert compute(RunRegistry(tmp_path).load(handle.run_id)).corrupt_receipt_lines == 1

    await walk(resumed, 5)
    assert resumed.total_presses == 105


async def test_an_unwritable_store_does_not_take_the_batch_with_it(tmp_path):
    class Exploding(RunRegistry):
        def append(self, run_id, receipt):
            raise OSError("disk full")

    oracle = FakeOracle()
    recorder = RunRecorder(tmp_path, registry=Exploding(tmp_path), milestone_snapshot=oracle)
    await recorder.begin_session(goal="Reach Pewter", model="qwen")

    await walk(recorder, 12)

    assert recorder.total_presses == 12
    assert "disk full" in (recorder.last_error or "")


async def test_appending_without_a_run_is_a_no_op(tmp_path):
    recorder = make_recorder(tmp_path, FakeOracle())

    assert await recorder.append(tool="action", presses=9) is None
    assert recorder.total_presses == 0


# ---------------------------------------------------------------------------
# Shaping a batch into a receipt
# ---------------------------------------------------------------------------


BUNDLE = {
    "state": {
        "map": {"map_name": "Viridian Forest"},
        "player": {"position": {"x": 12, "y": 8}},
        "party": [{"hp": 14, "max_hp": 31}, {"hp": 0, "max_hp": 20}],
    }
}


def test_receipt_from_batch_fills_every_field_the_schema_defines():
    fields = receipt_from_batch(
        tool="action",
        presses=6,
        bundle=BUNDLE,
        outcome={"moved": 4, "blocked_after": 5},
        milestone_ids=("EVENT_GOT_STARTER",),
    )

    assert fields["map_name"] == "Viridian Forest"
    assert fields["pos"] == (12, 8)
    assert fields["hp"] == (14, 31)
    assert fields["party_size"] == 2
    assert fields["moved"] == 4
    assert fields["blocked_after"] == 5
    assert fields["whiteout"] is False


def test_a_batch_with_no_bundle_still_makes_a_receipt():
    """An /action that raised has no observation and still has to leave a mark."""
    fields = receipt_from_batch(tool="action", presses=0, bundle=None, exit_code=1)

    assert fields["exit_code"] == 1
    assert fields["pos"] is None
    assert fields["hp"] is None
    assert fields["party_size"] == 0


def test_a_party_that_is_entirely_down_reads_as_a_whiteout():
    bundle = {"state": {"party": [{"hp": 0, "max_hp": 31}, {"hp": 0, "max_hp": 20}]}}

    assert receipt_from_batch(tool="action", presses=1, bundle=bundle)["whiteout"] is True


def test_one_survivor_is_not_a_whiteout():
    bundle = {"state": {"party": [{"hp": 0, "max_hp": 31}, {"hp": 3, "max_hp": 20}]}}

    assert receipt_from_batch(tool="action", presses=1, bundle=bundle)["whiteout"] is False


def test_config_hash_is_stable_across_key_order():
    assert config_hash({"model": "a", "thinking": "high"}) == config_hash(
        {"thinking": "high", "model": "a"}
    )
    assert config_hash({"model": "a"}) != config_hash({"model": "b"})


# ---------------------------------------------------------------------------
# What /progress reads
# ---------------------------------------------------------------------------


async def test_the_progress_payload_carries_the_run_and_its_ledger(tmp_path):
    oracle = FakeOracle()
    recorder = make_recorder(tmp_path, oracle)
    handle = await recorder.begin_session(goal="Reach Pewter", model="qwen")
    await walk(recorder, 40, milestones=("EVENT_GOT_STARTER",))

    payload = recorder.progress_payload()

    assert payload["run_id"] == handle.run_id
    assert payload["presses_to"] == {"EVENT_GOT_STARTER": 40}
    assert payload["attainments"][0]["presses"] == 40
    assert payload["attainments"][0]["milestone_id"] == "EVENT_GOT_STARTER"


async def test_recent_receipts_are_what_the_detectors_read(tmp_path):
    recorder = make_recorder(tmp_path, FakeOracle())
    await recorder.begin_session(goal="Reach Pewter", model="qwen")
    for _ in range(5):
        await walk(recorder, 4)

    recent = recorder.recent_receipts()

    assert [item.presses for item in recent[-5:]] == [4, 4, 4, 4, 4]
    assert recent[0].tool == "run_start"


@pytest.mark.parametrize("presses", [-5, 0])
async def test_a_batch_that_pressed_nothing_costs_nothing(tmp_path, presses):
    recorder = make_recorder(tmp_path, FakeOracle())
    await recorder.begin_session(goal="Reach Pewter", model="qwen")
    await recorder.append(tool="action", presses=presses)

    assert recorder.total_presses == 0


# ---------------------------------------------------------------------------
# Which harness produced these numbers
# ---------------------------------------------------------------------------


def git_dir(root: Path, head: str) -> Path:
    directory = root / ".git"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "HEAD").write_text(head + "\n", encoding="utf-8")
    return directory


def test_harness_sha_reads_a_detached_head(tmp_path):
    git_dir(tmp_path, "a" * 40)

    assert harness_sha(tmp_path) == "a" * 40


def test_harness_sha_follows_a_loose_ref(tmp_path):
    directory = git_dir(tmp_path, "ref: refs/heads/main")
    (directory / "refs" / "heads").mkdir(parents=True)
    (directory / "refs" / "heads" / "main").write_text("b" * 40, encoding="utf-8")

    assert harness_sha(tmp_path) == "b" * 40


def test_harness_sha_falls_back_to_packed_refs(tmp_path):
    directory = git_dir(tmp_path, "ref: refs/heads/main")
    (directory / "packed-refs").write_text(f"{'c' * 40} refs/heads/main\n", encoding="utf-8")

    assert harness_sha(tmp_path) == "c" * 40


def test_harness_sha_resolves_a_linked_worktree_through_commondir(tmp_path):
    """A worktree keeps HEAD locally and its refs in the repository it links to."""
    real = tmp_path / "repo" / ".git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "refs" / "heads" / "topic").write_text("d" * 40, encoding="utf-8")
    linked = real / "worktrees" / "one"
    linked.mkdir(parents=True)
    (linked / "HEAD").write_text("ref: refs/heads/topic\n", encoding="utf-8")
    (linked / "commondir").write_text("../..\n", encoding="utf-8")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {linked}\n", encoding="utf-8")

    assert harness_sha(checkout) == "d" * 40


def test_harness_sha_is_blank_without_a_git_directory(tmp_path):
    assert harness_sha(tmp_path) == ""
