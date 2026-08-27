"""The measurement layer: what gets written, what gets counted, what gets printed."""

from __future__ import annotations

import json
import sys
import types

import pytest

from pokemon_agent.bench import metrics as metrics_module
from pokemon_agent.bench import registry as registry_module
from pokemon_agent.bench import report as report_module
from pokemon_agent.bench.__main__ import main as bench_main
from pokemon_agent.bench.metrics import LadderEntry, compute
from pokemon_agent.bench.registry import Receipt, RunRegistry
from pokemon_agent.bench.report import format_comparison, format_run, render_table

# The parallel agent owns pokemon_agent.milestones; these ids stand in for it so
# the scoreboard is testable whether or not that module exists yet.
STARTER = "EVENT_GOT_STARTER"
OAK_PARCEL = "EVENT_DELIVERED_OAKS_PARCEL"
BROCK = "EVENT_BEAT_BROCK"

LADDER = {
    STARTER: LadderEntry(STARTER, "Got a starter", 0),
    OAK_PARCEL: LadderEntry(OAK_PARCEL, "Delivered Oak's parcel", 1),
    BROCK: LadderEntry(BROCK, "Beat Brock", 2),
}


def make_registry(tmp_path, **kwargs) -> RunRegistry:
    return RunRegistry(tmp_path / "store", **kwargs)


def start(registry: RunRegistry, **overrides) -> str:
    fields = {
        "harness_sha": "abc1234",
        "config_hash": "cfg-9",
        "model": "test-model",
        "start_checkpoint": "intro_complete",
        "goal": "reach Pewter City",
    }
    fields.update(overrides)
    return registry.start_run(**fields)


def receipt(**overrides) -> dict:
    """A receipt in the fixed schema, with everything defaulted to 'nothing happened'."""

    payload = {
        "t": 1724605812.4,
        "presses": 6,
        "map": "Route 3",
        "pos": [12, 8],
        "moved": 1,
        "blocked_after": None,
        "hp": [15, 40],
        "party_size": 2,
        "milestones_new": [],
        "milestone_count": 0,
        "tool": "act",
        "exit": 0,
        "reloaded": False,
        "whiteout": False,
    }
    payload.update(overrides)
    return payload


def record_from(tmp_path, receipts, **meta_overrides):
    """Push a hand-built receipt list through the registry and read it back."""

    registry = make_registry(tmp_path)
    run_id = start(registry, **meta_overrides)
    for item in receipts:
        registry.append(run_id, item)
    registry.finish(run_id, "test")
    return registry.load(run_id)


def _is_rule(line: str) -> bool:
    return bool(line) and set(line) <= {"-", " "} and "-" in line


def table_block(text: str, marker: str) -> list[str]:
    """The header, rule and rows of the table at or just after the ``marker`` line."""

    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(marker))
    # A table with its own heading line starts on the next line; a bare table
    # (the comparison ones) is matched on its header row itself.
    start_index = index if _is_rule(lines[index + 1]) else index + 1
    block: list[str] = []
    for line in lines[start_index:]:
        if not line.strip():
            break
        block.append(line)
    return block


def assert_columns_aligned(block: list[str]) -> None:
    """Every gutter in the rule line must be blank on every row of the table."""

    assert len(block) >= 2, block
    rule = block[1]
    assert _is_rule(rule), rule
    gutters = [index for index, char in enumerate(rule) if char == " "]
    for line in block:
        assert line == line.rstrip(), f"trailing whitespace: {line!r}"
        if line == "(none)":  # the empty-table sentinel, not a data row
            continue
        for index in gutters:
            assert index >= len(line) or line[index] == " ", (index, line)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_round_trip_start_append_finish_load(tmp_path):
    registry = make_registry(tmp_path)
    run_id = start(registry)

    registry.append(run_id, receipt(presses=6, milestones_new=[STARTER], milestone_count=3))
    registry.append(run_id, receipt(presses=4, map="Route 4", pos=[1, 2], moved=0))
    registry.finish(run_id, "context exhausted")

    record = registry.load(run_id)
    assert record.run_id == run_id
    assert record.meta.status == "finished"
    assert record.meta.finish_reason == "context exhausted"
    assert record.meta.ended_at is not None and record.meta.ended_at >= record.meta.started_at
    assert record.meta.harness_sha == "abc1234"
    assert record.meta.config_hash == "cfg-9"
    assert record.meta.model == "test-model"
    assert record.meta.start_checkpoint == "intro_complete"
    assert record.meta.goal == "reach Pewter City"
    assert record.corrupt_lines == 0

    assert [item.seq for item in record.receipts] == [0, 1]
    first, second = record.receipts
    assert first.presses == 6
    assert first.milestones_new == (STARTER,)
    assert first.pos == (12, 8)
    assert first.hp == (15, 40)
    assert second.map_name == "Route 4"
    assert second.moved == 0


def test_on_disk_layout_is_one_directory_per_run(tmp_path):
    registry = make_registry(tmp_path)
    run_id = start(registry)
    registry.append(run_id, receipt())
    registry.append(run_id, receipt(presses=2))
    registry.close_all()

    run_dir = tmp_path / "store" / "runs" / run_id
    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "receipts.jsonl").is_file()

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id
    assert meta["status"] == "running"

    lines = (run_dir / "receipts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    # The schema is fixed: these keys, with these names.
    assert set(payload) == {
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
        # The live count beside the running maximum. `milestone_count` never
        # falls by design; this one does, which is how a reload onto an earlier
        # branch is visible at all.
        "milestones_held",
        "tool",
        "exit",
        "reloaded",
        "whiteout",
    }


def test_explicit_seq_is_kept_and_continues_the_counter(tmp_path):
    registry = make_registry(tmp_path)
    run_id = start(registry)
    registry.append(run_id, receipt(seq=12))
    registry.append(run_id, receipt())
    record = registry.load(run_id)
    assert [item.seq for item in record.receipts] == [12, 13]


def test_appends_resume_after_the_process_that_started_the_run_dies(tmp_path):
    first = make_registry(tmp_path)
    run_id = start(first)
    first.append(run_id, receipt())
    first.append(run_id, receipt())
    first.close_all()

    # A fresh registry object, as a restarted supervisor would build.
    second = make_registry(tmp_path)
    second.append(run_id, receipt(presses=3))
    record = second.load(run_id)
    assert [item.seq for item in record.receipts] == [0, 1, 2]
    assert record.receipts[-1].presses == 3


def test_a_torn_final_line_costs_one_receipt_and_nothing_else(tmp_path):
    registry = make_registry(tmp_path)
    run_id = start(registry)
    registry.append(run_id, receipt(presses=6))
    registry.append(run_id, receipt(presses=7))
    registry.close_all()

    path = registry.receipts_path(run_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "presses": 9, "map": "Rou')  # killed mid-write

    record = registry.load(run_id)
    assert len(record.receipts) == 2
    assert record.corrupt_lines == 1
    assert compute(record, ladder=LADDER).total_presses == 13


def test_run_ids_sort_chronologically(tmp_path):
    registry = make_registry(tmp_path)
    ids = [
        registry.start_run(
            harness_sha="s",
            config_hash="c",
            model="m",
            start_checkpoint=None,
            goal="g",
            started_at=stamp,
        )
        for stamp in (1_700_000_000.0, 1_700_003_600.0, 1_700_007_200.0)
    ]
    assert ids == sorted(ids)
    assert [summary.run_id for summary in registry.list_runs()] == ids


def test_list_runs_reports_status_and_receipt_count(tmp_path):
    registry = make_registry(tmp_path)
    finished = start(registry, goal="first")
    registry.append(finished, receipt())
    registry.append(finished, receipt())
    registry.finish(finished, "done")
    running = start(registry, goal="second")
    registry.append(running, receipt())

    summaries = {summary.run_id: summary for summary in registry.list_runs()}
    assert summaries[finished].status == "finished"
    assert summaries[finished].receipt_count == 2
    assert summaries[finished].finish_reason == "done"
    assert summaries[running].status == "running"
    assert summaries[running].receipt_count == 1


def test_loading_an_unknown_run_raises(tmp_path):
    registry = make_registry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load("nope")
    with pytest.raises(FileNotFoundError):
        registry.append("nope", receipt())


def test_receipt_round_trips_through_its_own_dict():
    original = Receipt.from_dict(receipt(seq=3, milestones_new=[BROCK], blocked_after="wall"))
    assert Receipt.from_dict(original.to_dict()) == original


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_presses_to_records_first_attainment_only_and_never_regresses(tmp_path):
    record = record_from(
        tmp_path,
        [
            receipt(presses=10),
            receipt(presses=5, milestones_new=[STARTER], milestone_count=1),
            receipt(presses=20),
            # The tracker re-announcing a milestone must not re-price it.
            receipt(presses=100, milestones_new=[STARTER, OAK_PARCEL], milestone_count=2),
            receipt(presses=50, milestones_new=[BROCK], milestone_count=3),
        ],
    )
    result = compute(record, ladder=LADDER)

    assert result.presses_to == {STARTER: 15, OAK_PARCEL: 135, BROCK: 185}
    assert result.total_presses == 185
    assert result.milestones_reached == 3
    assert result.furthest_milestone == BROCK
    assert result.furthest_label == "Beat Brock"
    assert result.presses_per_milestone == pytest.approx(185 / 3, abs=0.05)

    values = [item.presses for item in result.attainments]
    assert values == sorted(values), "presses at attainment must never go backwards"


def test_presses_accumulate_across_a_reload_instead_of_resetting(tmp_path):
    """Beating Brock on the fourth attempt costs all four attempts."""

    receipts = []
    for _ in range(3):
        receipts.append(receipt(presses=400))
        receipts.append(receipt(presses=0, tool="load", reloaded=True, whiteout=True))
    receipts.append(receipt(presses=8, milestones_new=[BROCK], milestone_count=9))

    result = compute(record_from(tmp_path, receipts), ladder=LADDER)

    assert result.reloads == 3
    assert result.whiteouts == 3
    assert result.total_presses == 1208
    assert result.presses_to[BROCK] == 1208, "a reload must never rewind the press counter"


def test_blocked_and_error_rates(tmp_path):
    record = record_from(
        tmp_path,
        [
            receipt(presses=4, moved=1),
            receipt(presses=4, moved=0, blocked_after="up"),
            receipt(presses=4, moved=0, blocked_after="up"),
            receipt(presses=4, moved=2),
            # No buttons sent: a read, not a batch that could have been blocked.
            receipt(presses=0, moved=None, tool="state"),
            receipt(presses=4, moved=1, exit=1, tool="act"),
        ],
    )
    result = compute(record, ladder=LADDER)

    assert result.action_batches == 5
    assert result.blocked_batches == 2
    assert result.blocked_rate == pytest.approx(0.4)
    assert result.tool_calls == 6
    assert result.tool_errors == 1
    assert result.tool_error_rate == pytest.approx(1 / 6, abs=0.001)


def test_revisit_ratio_overall_and_per_map(tmp_path):
    record = record_from(
        tmp_path,
        [
            receipt(map="Route 3", pos=[1, 1]),
            receipt(map="Route 3", pos=[1, 1]),
            receipt(map="Route 3", pos=[2, 1]),
            receipt(map="Mt Moon", pos=[1, 1]),
            receipt(map="Mt Moon", pos=[5, 5]),
            receipt(map="Mt Moon", pos=[5, 5]),
            receipt(map="Mt Moon", pos=[5, 5]),
        ],
    )
    result = compute(record, ladder=LADDER)

    assert result.position_samples == 7
    # (1,1) on Route 3 and (1,1) in Mt Moon are different tiles.
    assert result.unique_positions == 4
    assert result.revisit_ratio == pytest.approx(7 / 4)

    by_map = {entry.map_name: entry for entry in result.revisit_by_map}
    assert by_map["Route 3"].samples == 3 and by_map["Route 3"].unique == 2
    assert by_map["Mt Moon"].ratio == pytest.approx(4 / 2)


def test_wall_clock_and_metadata_survive_the_trip(tmp_path):
    registry = make_registry(tmp_path)
    run_id = registry.start_run(
        harness_sha="deadbee",
        config_hash="cfg",
        model="m",
        start_checkpoint=None,
        goal="beat brock",
        started_at=1_700_000_000.0,
    )
    registry.append(run_id, receipt(t=1_700_000_010.0))
    registry.finish(run_id, "stopped")
    result = compute(registry.load(run_id), ladder=LADDER)

    assert result.goal == "beat brock"
    assert result.harness_sha == "deadbee"
    assert result.status == "finished"
    assert result.wall_clock_seconds > 0


def test_empty_run_scores_zero_without_dividing_by_zero(tmp_path):
    result = compute(record_from(tmp_path, []), ladder=LADDER)

    assert result.total_presses == 0
    assert result.presses_to == {}
    assert result.milestones_reached == 0
    assert result.presses_per_milestone is None
    assert result.blocked_rate == 0.0
    assert result.tool_error_rate == 0.0
    assert result.revisit_ratio == 0.0
    assert result.furthest_milestone is None


def test_milestones_order_by_the_ladder_not_by_when_they_fired(tmp_path):
    record = record_from(
        tmp_path,
        [
            receipt(presses=100, milestones_new=[BROCK]),
            receipt(presses=10, milestones_new=[STARTER]),
        ],
    )
    result = compute(record, ladder=LADDER)
    assert [item.milestone_id for item in result.attainments] == [STARTER, BROCK]
    assert result.presses_to[STARTER] == 110
    assert result.presses_to[BROCK] == 100


def test_unknown_milestone_ids_still_get_scored(tmp_path):
    record = record_from(tmp_path, [receipt(presses=7, milestones_new=["EVENT_MADE_IT_UP"])])
    result = compute(record, ladder=LADDER)
    assert result.presses_to == {"EVENT_MADE_IT_UP": 7}
    assert result.attainments[0].ladder_index is None
    assert result.attainments[0].label == "EVENT_MADE_IT_UP"


def install_milestones(monkeypatch, module):
    """Stand in for ``pokemon_agent.milestones``, present or absent."""

    import pokemon_agent

    if module is None:
        monkeypatch.delattr(pokemon_agent, "milestones", raising=False)
        monkeypatch.setitem(sys.modules, "pokemon_agent.milestones", None)
        return
    monkeypatch.setattr(pokemon_agent, "milestones", module, raising=False)
    monkeypatch.setitem(sys.modules, "pokemon_agent.milestones", module)


def test_load_ladder_tolerates_a_missing_or_broken_milestones_module(monkeypatch):
    install_milestones(monkeypatch, None)
    assert metrics_module.load_ladder() == {}

    broken = types.ModuleType("pokemon_agent.milestones")
    broken.MILESTONES = 17  # not iterable
    install_milestones(monkeypatch, broken)
    assert metrics_module.load_ladder() == {}

    partial = types.ModuleType("pokemon_agent.milestones")
    partial.MILESTONES = (types.SimpleNamespace(label="no id here"),)
    install_milestones(monkeypatch, partial)
    assert metrics_module.load_ladder() == {}


def test_load_ladder_reads_the_contract_shape(monkeypatch):
    rung = types.SimpleNamespace(id=BROCK, label="Beat Brock", kind="event", ladder_index=7)
    off_ladder = types.SimpleNamespace(id="EVENT_ODD", label="Odd", kind="event", ladder_index=-1)
    module = types.ModuleType("pokemon_agent.milestones")
    module.MILESTONES = (rung, off_ladder)
    install_milestones(monkeypatch, module)

    ladder = metrics_module.load_ladder()
    assert ladder[BROCK].ladder_index == 7
    assert ladder[BROCK].label == "Beat Brock"
    # -1 is the contract's "not on the curated ladder": no rung, so no rank.
    assert ladder["EVENT_ODD"].ladder_index is None


def test_the_real_milestone_ladder_is_used_when_it_is_available():
    """The parallel agent's module, if it has landed, must slot straight in."""

    ladder = metrics_module.load_ladder()
    if not ladder:
        pytest.skip("pokemon_agent.milestones is not written yet")
    entry = next(iter(ladder.values()))
    assert isinstance(entry.milestone_id, str) and entry.milestone_id
    assert isinstance(entry.label, str) and entry.label
    assert entry.ladder_index is None or isinstance(entry.ladder_index, int)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_render_table_aligns_and_rules_its_columns():
    lines = render_table(
        ["milestone", "presses"],
        [["Beat Brock", "1,608"], ["Got a starter", "12"]],
        aligns=["l", "r"],
    )
    assert lines == [
        "milestone      presses",
        "-------------  -------",
        "Beat Brock       1,608",
        "Got a starter       12",
    ]


def test_format_run_prints_aligned_tables_and_the_headline(tmp_path):
    record = record_from(
        tmp_path,
        [
            receipt(presses=100, milestones_new=[STARTER], milestone_count=1),
            receipt(presses=8, moved=0, exit=1),
            receipt(presses=1500, milestones_new=[BROCK], milestone_count=2, map="Pewter Gym"),
        ],
    )
    text = format_run(compute(record, ladder=LADDER))

    assert "1,608" in text  # cumulative presses to Brock, in the published currency
    assert "Beat Brock" in text
    assert "presses are never reset by a reload" in text
    assert "\x1b[" not in text  # no colour codes
    assert_columns_aligned(table_block(text, "MILESTONES"))
    assert_columns_aligned(table_block(text, "REVISITS BY MAP"))


def test_format_run_survives_an_empty_run_and_a_run_with_no_milestones(tmp_path):
    empty = format_run(compute(record_from(tmp_path, []), ladder=LADDER))
    assert "(none)" in empty
    assert_columns_aligned(table_block(empty, "MILESTONES"))

    silent = format_run(compute(record_from(tmp_path, [receipt(presses=9)] * 3), ladder=LADDER))
    assert "27" in silent
    assert "(none)" in silent
    assert_columns_aligned(table_block(silent, "MILESTONES"))


def test_format_comparison_puts_runs_side_by_side_with_a_delta(tmp_path):
    fast = compute(
        record_from(tmp_path / "a", [receipt(presses=649, milestones_new=[BROCK])]),
        ladder=LADDER,
    )
    slow = compute(
        record_from(tmp_path / "b", [receipt(presses=1608, milestones_new=[BROCK])]),
        ladder=LADDER,
    )
    nowhere = compute(record_from(tmp_path / "c", [receipt(presses=40)]), ladder=LADDER)

    text = format_comparison([("fast", fast), ("slow", slow), ("nowhere", nowhere)])

    assert "649 best" in text
    assert "1,608 +959" in text  # the whole point: a number to accept or reject on
    assert "PokeAgent best entry, first gym 1,608 actions" in text
    assert "PokeAgent most efficient, first gym 649 actions" in text
    assert_columns_aligned(table_block(text, "milestone "))
    assert_columns_aligned(table_block(text, "metric "))


def test_format_comparison_handles_no_runs_and_milestone_free_runs(tmp_path):
    assert "No runs" in format_comparison([])

    blank = compute(record_from(tmp_path, []), ladder=LADDER)
    text = format_comparison([("one", blank), ("two", blank)])
    assert "(none)" in text
    assert_columns_aligned(table_block(text, "metric "))


def test_format_duration_reads_like_a_clock():
    assert report_module.format_duration(0) == "0s"
    assert report_module.format_duration(75) == "1m 15s"
    assert report_module.format_duration(3 * 3600 + 12 * 60 + 4) == "3h 12m 04s"


# ---------------------------------------------------------------------------
# python -m pokemon_agent.bench
# ---------------------------------------------------------------------------


def test_cli_lists_prints_and_compares(tmp_path, capsys):
    registry = make_registry(tmp_path)
    first = start(registry, goal="one")
    registry.append(first, receipt(presses=649, milestones_new=[BROCK]))
    registry.finish(first, "done")
    second = start(registry, goal="two")
    registry.append(second, receipt(presses=1608, milestones_new=[BROCK]))
    registry.finish(second, "done")
    data_dir = str(registry.data_dir)

    assert bench_main(["--data-dir", data_dir]) == 0
    listing = capsys.readouterr().out
    assert first in listing and second in listing

    assert bench_main([first, "--data-dir", data_dir]) == 0
    assert "RUN " in capsys.readouterr().out

    assert bench_main(["--compare", first, second, "--data-dir", data_dir]) == 0
    comparison = capsys.readouterr().out
    assert "+959" in comparison

    assert bench_main(["missing-run", "--data-dir", data_dir]) == 1
    assert bench_main(["--data-dir", str(tmp_path / "nothing-here")]) == 1


def test_registry_constants_name_the_layout():
    assert registry_module.RUNS_DIRNAME == "runs"
    assert registry_module.META_FILENAME == "meta.json"
    assert registry_module.RECEIPTS_FILENAME == "receipts.jsonl"
