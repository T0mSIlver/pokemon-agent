"""Tests for ``pokemon_agent.scope``.

Two halves. The first builds transcripts and receipt stores in ``tmp_path`` and
pins the behaviour that has to hold on any machine. The second runs every
command against whatever real data is on this disk — eight session transcripts
and a live run, at the time of writing — because a reporting tool that has only
ever seen its own fixtures is a tool that has never been tested. Those tests
skip cleanly where the data is absent.
"""

from __future__ import annotations

import base64
import json
import struct
import time
from pathlib import Path

import pytest

from pokemon_agent.bench.registry import Receipt, RunRecord, RunRegistry
from pokemon_agent.scope import analysis, commands, render
from pokemon_agent.scope.__main__ import main
from pokemon_agent.scope.discover import (
    DATA_DIR_ENV,
    WORKSPACE_ENV,
    discover,
    list_sessions,
    resolve_session,
)
from pokemon_agent.scope.runs import ActionContext, ContextOracle, ladder_progress
from pokemon_agent.scope.transcript import (
    ADVISORY_VERBS,
    base64_bytes,
    bash_program,
    iter_jsonl,
    parse_session,
    png_dimensions,
    poke_verbs,
    split_segments,
)

# -- helpers ------------------------------------------------------------------


def png_b64(width: int, height: int, padding: int = 600) -> str:
    """A byte string that starts like a PNG of the given size."""

    blob = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
        + b"\x00" * padding
    )
    return base64.b64encode(blob).decode("ascii")


class TranscriptBuilder:
    """Writes a transcript the way the Pi supervisor writes one."""

    def __init__(self, path: Path, *, session_id: str = "abcdef01-2222", model: str = "qwen38-27b"):
        self.path = path
        self.lines: list[str] = []
        self.clock = 1_787_698_000.0
        self.call_seq = 0
        self._emit({"type": "session", "id": session_id, "timestamp": self._iso()})
        self._emit({"type": "model_change", "provider": "llamacpp", "modelId": model})
        self._emit({"type": "thinking_level_change", "thinkingLevel": "off"})

    def _iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.clock)) + ".000Z"

    def _emit(self, payload: dict) -> None:
        self.lines.append(json.dumps(payload))
        self.write()

    def write(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")

    def user(self, text: str, images: int = 0) -> "TranscriptBuilder":
        content: list[dict] = [{"type": "text", "text": text}]
        for _ in range(images):
            content.append({"type": "image", "data": png_b64(160, 144), "mimeType": "image/png"})
        self._emit(
            {
                "type": "message",
                "timestamp": self._iso(),
                "message": {"role": "user", "content": content},
            }
        )
        return self

    def turn(
        self,
        command: str,
        result: str = "ok",
        *,
        tool: str = "bash",
        prompt_tokens: int = 1000,
        output_tokens: int = 20,
        images: int = 0,
        is_error: bool = False,
        seconds: float = 1.0,
    ) -> "TranscriptBuilder":
        self.clock += seconds
        self.call_seq += 1
        call_id = f"call{self.call_seq:03d}"
        arguments = {"command": command} if tool == "bash" else {"path": command}
        self._emit(
            {
                "type": "message",
                "timestamp": self._iso(),
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": call_id, "name": tool, "arguments": arguments}
                    ],
                    "usage": {
                        "input": prompt_tokens,
                        "output": output_tokens,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": prompt_tokens + output_tokens,
                    },
                },
            }
        )
        self.clock += seconds
        content: list[dict] = [{"type": "text", "text": result}]
        for _ in range(images):
            content.append({"type": "image", "data": png_b64(160, 144), "mimeType": "image/png"})
        self._emit(
            {
                "type": "message",
                "timestamp": self._iso(),
                "message": {
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "toolName": tool,
                    "content": content,
                    "isError": is_error,
                },
            }
        )
        return self

    def truncate_tail(self, keep: int = 40) -> None:
        """Leave the file as a writer caught mid-append leaves it."""

        text = "\n".join(self.lines) + "\n"
        cut = text.rfind("\n", 0, len(text) - 1)
        self.path.write_text(text[: cut + 1] + text[cut + 1 : cut + 1 + keep], encoding="utf-8")


def act_result(
    x: int, y: int, moved: int, *, here_before: int = 0, map_name: str = "Route 3"
) -> str:
    return json.dumps(
        {
            "actions_executed": 1,
            "map": map_name,
            "x": x,
            "y": y,
            "moved": moved,
            "here_before": here_before,
            "dialog": False,
            "battle": False,
        }
    )


def receipt(**kwargs) -> Receipt:
    payload = {
        "seq": 0,
        "t": 1_787_698_000.0,
        "presses": 1,
        "map": "Route 3",
        "pos": [1, 1],
        "moved": 1,
        "milestones_new": [],
        "tool": "action",
        "exit": 0,
    }
    payload.update(kwargs)
    return Receipt.from_dict(payload)


def make_record(receipts, run_id: str = "run-a", **meta) -> RunRecord:
    from pokemon_agent.bench.registry import RunMeta

    base = {"run_id": run_id, "started_at": 1_787_698_000.0, "status": "running"}
    base.update(meta)
    return RunRecord(meta=RunMeta.from_dict(base), receipts=tuple(receipts))


# -- jsonl reading ------------------------------------------------------------


def test_truncated_final_line_is_the_writer_not_corruption(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    builder.turn("./poke state", "Route 3 (1,1)")
    builder.truncate_tail()
    objects, corrupt = iter_jsonl(path)
    assert corrupt == 0
    session = parse_session(path)
    assert session.corrupt_lines == 0
    assert len(objects) >= 4


def test_damage_before_the_end_is_counted(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"type": "session"}\nnot json at all\n{"type": "message"}\n', encoding="utf-8")
    _, corrupt = iter_jsonl(path)
    assert corrupt == 1


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert iter_jsonl(tmp_path / "nope.jsonl") == ([], 0)


def test_reading_a_growing_file_sees_a_consistent_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(5):
        builder.turn("./poke act up", act_result(1, index, 1))
    before = len(parse_session(path).calls)
    builder.turn("./poke act down", act_result(1, 9, 1))
    builder.truncate_tail()
    after = parse_session(path)
    assert len(after.calls) >= before
    assert after.corrupt_lines == 0


# -- images -------------------------------------------------------------------


def test_png_dimensions_come_from_the_header_only() -> None:
    assert png_dimensions(png_b64(160, 144)) == (160, 144)
    assert png_dimensions(png_b64(640, 576)) == (640, 576)


def test_png_dimensions_reject_non_png_and_short_input() -> None:
    assert png_dimensions(base64.b64encode(b"not a png at all really").decode()) is None
    assert png_dimensions("AAAA") is None


def test_base64_bytes_does_not_decode() -> None:
    blob = b"\x00" * 300
    assert base64_bytes(base64.b64encode(blob).decode()) == 300


def test_parsed_session_never_retains_base64(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    TranscriptBuilder(path).user("go", images=2).turn("./poke frame", "ok", images=1)
    session = parse_session(path)
    images = [image for step in session.steps for image in step.images]
    assert len(images) == 3
    assert all(image.width == 160 and image.height == 144 for image in images)
    assert not any("data" in vars(image) for image in images)


# -- command parsing ----------------------------------------------------------


def test_split_segments_respects_quotes() -> None:
    assert split_segments("./poke state; ./poke route X") == ["./poke state", "./poke route X"]
    assert split_segments("python3 -c 'a; b; c'") == ["python3 -c 'a; b; c'"]
    assert split_segments("a && b || c | d") == ["a", "b", "c", "d"]


def test_poke_verbs_finds_every_invocation() -> None:
    assert poke_verbs("./poke state; echo ---; ./poke route Cerulean City") == ["state", "route"]
    assert poke_verbs("echo poke nothing here") == ["nothing"]
    assert poke_verbs("ls -la") == []


def test_bash_program_skips_the_getting_there() -> None:
    assert bash_program("cd /some/where && bash agent_curl.sh /agent/plan") == "bash agent_curl.sh"
    assert bash_program('# a comment\npython3 -c "print(1)"') == "python3 -c"
    assert bash_program('p="$(cat .ep_plan)"; curl -s http://x') == "curl"
    assert bash_program("/usr/bin/grep -n foo bar") == "grep"
    assert bash_program("   ") == "bash"


def test_a_poke_call_keeps_its_arguments_in_the_signature(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    TranscriptBuilder(path).turn("./poke act up", act_result(1, 1, 1))
    call = parse_session(path).calls[0]
    assert call.label == "poke act"
    assert call.signature == "poke act up"
    assert call.kind == "poke"


# -- tools --------------------------------------------------------------------


def test_tool_report_counts_each_verb_of_a_multi_verb_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    builder.turn("./poke state; ./poke route Cerulean", "two verbs, one bash call")
    builder.turn("curl -s http://127.0.0.1:8765/agent/observe", "{}")
    report = analysis.tool_report(parse_session(path))
    labels = {stat.label: stat.calls for stat in report.stats}
    assert labels["poke state"] == 1
    assert labels["poke route"] == 1
    assert report.bash_calls == 2
    assert report.bash_poke_calls == 1
    assert report.bash_other_calls == 1
    assert report.other_programs == (("!curl", 1),)


def test_tool_report_always_lists_every_advisory_verb(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    TranscriptBuilder(path).turn("./poke act up", act_result(1, 1, 1))
    report = analysis.tool_report(parse_session(path))
    assert tuple(verb for verb, _, _ in report.advisory) == ADVISORY_VERBS
    assert all(calls == 0 for _, calls, _ in report.advisory)


def test_tool_report_medians_and_failures(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    builder.turn("./poke act up", act_result(1, 1, 1))
    builder.turn("./poke act up", act_result(1, 2, 3))
    builder.turn("./poke act up", "boom", is_error=True)
    stat = next(s for s in analysis.tool_report(parse_session(path)).stats if s.label == "poke act")
    assert stat.calls == 3
    assert stat.failures == 1
    assert stat.median_moved == 2  # 1 and 3; the failed call reports no move
    assert stat.median_result_bytes > 0


def test_non_bash_tools_are_reported_under_their_own_name(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    TranscriptBuilder(path).turn("frame.png", "Read image file", tool="read")
    report = analysis.tool_report(parse_session(path))
    assert ("read", 1) in [(stat.label, stat.calls) for stat in report.stats]
    assert report.bash_calls == 0


# -- loops --------------------------------------------------------------------


def _loop_session(tmp_path: Path, commands_list: list[str]) -> Path:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index, command in enumerate(commands_list):
        builder.turn(command, act_result(1, index, 1))
    return path


def test_loops_counts_repeats_without_overlap(tmp_path: Path) -> None:
    path = _loop_session(tmp_path, ["./poke act up", "./poke act down"] * 5)
    loops = analysis.find_loops(parse_session(path))
    assert loops
    top = loops[0]
    assert top.count == 5
    assert top.length == 2
    assert top.covered == 10


def test_loops_treat_a_rotation_as_the_same_cycle(tmp_path: Path) -> None:
    path = _loop_session(tmp_path, ["./poke act up", "./poke act down"] * 6)
    sequences = [tuple(loop.tokens) for loop in analysis.find_loops(parse_session(path))]
    assert len(sequences) == len({analysis._canonical_rotation(seq) for seq in sequences})


def test_loops_report_a_cycle_at_its_true_period(tmp_path: Path) -> None:
    path = _loop_session(tmp_path, ["./poke act up", "./poke act down", "./poke act left"] * 4)
    loops = analysis.find_loops(parse_session(path))
    assert loops[0].length == 3, "a doubled cycle is not a longer loop"
    assert loops[0].count == 4
    assert all(loop.length >= 3 or loop.count > loops[0].count for loop in loops)


def test_a_doubled_cycle_is_not_primitive() -> None:
    assert analysis._is_primitive(("a", "b"))
    assert analysis._is_primitive(("a", "b", "a"))
    assert not analysis._is_primitive(("a", "b", "a", "b"))
    assert not analysis._is_primitive(("a", "a", "a"))


def test_no_loops_in_a_session_that_never_repeats(tmp_path: Path) -> None:
    path = _loop_session(tmp_path, [f"./poke act step{i}" for i in range(10)])
    assert analysis.find_loops(parse_session(path)) == []
    assert analysis.tail_repeat(parse_session(path)) is None


def test_tail_repeat_catches_a_session_stuck_right_now(tmp_path: Path) -> None:
    path = _loop_session(
        tmp_path, ["./poke goto Cerulean"] + ["./poke act up", "./poke act down"] * 4
    )
    stuck = analysis.tail_repeat(parse_session(path))
    assert stuck is not None
    assert stuck.count == 4
    assert stuck.tokens == ("poke act up", "poke act down")


def test_loops_report_where_they_happened(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(6):
        builder.turn("./poke act up", act_result(1, index, 1, map_name="Mt. Moon"))
    loops = analysis.find_loops(parse_session(path))
    assert loops and loops[0].where == "Mt. Moon"


# -- waste --------------------------------------------------------------------


def test_waste_buckets_follow_the_documented_precedence() -> None:
    oracle = ContextOracle(
        [
            ActionContext(at=1_787_698_003.001, dialog=True),
            ActionContext(at=1_787_698_004.001, battle=True),
        ]
    )
    record = make_record(
        [
            receipt(seq=0, t=1_787_698_000.0, pos=[1, 1], moved=1),  # new tile
            receipt(seq=1, t=1_787_698_001.0, pos=[1, 1], moved=1),  # been there
            receipt(seq=2, t=1_787_698_002.0, pos=[1, 1], moved=0),  # blocked
            receipt(seq=3, t=1_787_698_003.0, pos=[1, 1], moved=0),  # dialog wins
            receipt(seq=4, t=1_787_698_004.0, pos=[1, 1], moved=0),  # battle wins
            receipt(seq=5, t=1_787_698_005.0, pos=[9, 9], moved=1, milestones_new=["EVENT_X"]),
        ]
    )
    report = analysis.waste_report(record, oracle)
    assert report.overall.batches == {
        "productive": 2,
        "battle": 1,
        "dialog": 1,
        "blocked": 1,
        "revisit": 1,
    }
    assert report.overall.total_presses == 6
    assert report.milestones == 1


def test_waste_without_a_run_log_still_splits_movement() -> None:
    record = make_record(
        [
            receipt(seq=0, pos=[1, 1], moved=1, presses=4),
            receipt(seq=1, pos=[1, 1], moved=0, presses=6),
        ]
    )
    report = analysis.waste_report(record, None)
    assert report.overall.presses["productive"] == 4
    assert report.overall.presses["blocked"] == 6
    assert report.overall.presses["dialog"] == 0
    assert report.context_samples == 0


def test_waste_splits_per_map() -> None:
    record = make_record(
        [
            receipt(seq=0, map="Route 3", pos=[1, 1], presses=10),
            receipt(seq=1, map="Route 3", pos=[1, 1], presses=10),
            receipt(seq=2, map="Mt. Moon", pos=[2, 2], presses=5),
        ]
    )
    report = analysis.waste_report(record, None)
    names = [split.name for split in report.by_map]
    assert names == ["Route 3", "Mt. Moon"]  # ordered by presses spent
    assert report.by_map[0].total_presses == 20


def test_zero_press_receipts_still_mark_ground_as_walked() -> None:
    record = make_record(
        [
            receipt(seq=0, presses=0, pos=[3, 3], tool="run_start"),
            receipt(seq=1, presses=5, pos=[3, 3], moved=1),
        ]
    )
    report = analysis.waste_report(record, None)
    assert report.overall.presses["revisit"] == 5
    assert report.overall.total_presses == 5


def test_context_oracle_matches_the_same_moment_and_nothing_else() -> None:
    """The tolerance has to be wider than the write gap and narrower than the
    gap between two batches, or a batch inherits its neighbour's state."""

    oracle = ContextOracle([ActionContext(at=1000.0, battle=True)])
    assert oracle.at(1000.001) is not None
    assert oracle.at(1000.18) is None  # the tightest batch spacing seen live
    assert oracle.at(9999.0) is None
    assert oracle.at(0) is None


# -- context ------------------------------------------------------------------


def test_context_uses_reported_usage_not_an_estimate(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(5):
        builder.turn("./poke act up", "ok", prompt_tokens=1000 + index * 100)
    report = analysis.context_report(parse_session(path), window=10_000)
    assert report.first_prompt == 1000
    assert report.peak_prompt == 1400
    assert report.peak_step == 4
    assert report.median_growth == 100.0
    assert report.peak_share == pytest.approx(0.14)


def test_context_counts_images_from_both_directions(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    builder.user("go", images=2)
    builder.turn("./poke frame", "ok", images=1)
    builder.turn("./poke act up", "ok")
    report = analysis.context_report(parse_session(path))
    assert report.image_count == 3
    assert report.prompt_images == 2
    assert report.tool_result_images == 1
    assert report.common_size == "160x144"
    assert report.est_tokens_per_image == analysis.image_tokens(
        next(image for step in parse_session(path).steps for image in step.images)
    )


def test_context_measures_image_cost_from_observed_growth(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    prompt = 1000
    for index in range(8):
        carries_image = index % 2 == 0
        builder.turn("./poke frame", "ok", prompt_tokens=prompt, images=1 if carries_image else 0)
        prompt += 500 if carries_image else 50
    report = analysis.context_report(parse_session(path))
    assert report.measured_with_image == 500.0
    assert report.measured_without_image == 50.0
    assert "images" in report.verdict


def test_context_verdict_clears_frames_when_they_are_cheap(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    prompt = 1000
    for index in range(8):
        builder.turn("./poke frame", "ok", prompt_tokens=prompt, images=1 if index % 2 == 0 else 0)
        prompt += 50
    report = analysis.context_report(parse_session(path))
    assert report.image_share < 0.2
    assert "not what is filling the window" in report.verdict


def test_context_survives_a_session_with_no_usage(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"type": "session", "id": "x"}\n', encoding="utf-8")
    report = analysis.context_report(parse_session(path))
    assert report.peak_prompt == 0
    assert report.verdict == "no images in this session at all"


# -- runs ---------------------------------------------------------------------


def test_ladder_progress_counts_the_checkpoint_it_inherited() -> None:
    from pokemon_agent.scope.runs import ladder_ids

    rungs = ladder_ids()
    assert len(rungs) == 63, "the curated ladder is the denominator scope prints"
    record = make_record(
        [
            receipt(seq=0, presses=0, baseline_milestones=list(rungs[:3])),
            receipt(seq=1, milestones_new=[rungs[7]]),
            receipt(seq=2, milestones_new=[rungs[7], "NOT_ON_THE_LADDER"]),
        ]
    )
    progress = ladder_progress(record)
    assert progress.total == 63
    assert progress.baseline == 3
    assert progress.reached == 4
    assert str(progress) == "4/63"


def test_timeline_rows_carry_deltas() -> None:
    from pokemon_agent.scope.runs import ladder_ids

    rungs = ladder_ids()
    record = make_record(
        [
            receipt(seq=0, t=1000.0, presses=100, milestones_new=[rungs[0]]),
            receipt(seq=1, t=1060.0, presses=250, milestones_new=[rungs[1]]),
        ]
    )
    rows = analysis.timeline_rows(record)
    assert [row.presses for row in rows] == [100, 350]
    assert [row.delta_presses for row in rows] == [100, 250]
    assert rows[1].delta_seconds == 60.0


def test_presses_never_reset_on_a_reload() -> None:
    record = make_record(
        [
            receipt(seq=0, presses=40),
            receipt(seq=1, presses=60, reloaded=True),
            receipt(seq=2, presses=10),
        ]
    )
    from pokemon_agent.scope.runs import run_metrics

    assert run_metrics(record).total_presses == 110


# -- discovery ----------------------------------------------------------------


def _fake_proc(tmp_path: Path, workspace: Path, data_dir: Path) -> Path:
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    argv = [
        ".venv/bin/python",
        "-m",
        "pokemon_agent.cli",
        "serve",
        "--data-dir",
        str(data_dir),
        "--agent-workspace-dir",
        str(workspace),
    ]
    (proc / "4242" / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    (proc / "notapid").mkdir()
    return proc


def test_discovery_reads_the_live_server_argv(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "pi-session").mkdir(parents=True)
    data_dir = tmp_path / "data"
    (data_dir / "runs").mkdir(parents=True)
    paths = discover(cwd=tmp_path, env={}, proc_dir=_fake_proc(tmp_path, workspace, data_dir))
    assert paths.workspace == workspace
    assert paths.data_dir == data_dir
    assert paths.workspace_source == "live server argv"


def test_explicit_arguments_beat_the_live_server(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "pi-session").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    paths = discover(
        other, other, cwd=tmp_path, env={}, proc_dir=_fake_proc(tmp_path, workspace, tmp_path)
    )
    assert paths.workspace == other
    assert paths.workspace_source == "--workspace"
    assert paths.data_dir_source == "--data-dir"


def test_environment_beats_the_live_server(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "pi-session").mkdir(parents=True)
    env = {WORKSPACE_ENV: str(tmp_path / "env-ws"), DATA_DIR_ENV: str(tmp_path / "env-data")}
    paths = discover(cwd=tmp_path, env=env, proc_dir=_fake_proc(tmp_path, workspace, tmp_path))
    assert paths.workspace == tmp_path / "env-ws"
    assert paths.workspace_source == f"${WORKSPACE_ENV}"


def test_discovery_walks_up_from_the_cwd(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".agent-workspace" / "pi-session").mkdir(parents=True)
    (root / "runs").mkdir()
    deep = root / "a" / "b"
    deep.mkdir(parents=True)
    paths = discover(cwd=deep, env={}, proc_dir=tmp_path / "no-proc")
    assert paths.workspace == root / ".agent-workspace"
    assert paths.data_dir == root


def test_discovery_finds_nothing_gracefully(tmp_path: Path) -> None:
    paths = discover(cwd=tmp_path, env={}, proc_dir=tmp_path / "no-proc")
    assert paths.session_dir is None or not paths.session_dir.exists()
    assert paths.workspace_source in {"not found", "cwd"}


def test_resolve_session_prefers_the_newest_then_the_match(tmp_path: Path) -> None:
    session_dir = tmp_path / "pi-session"
    session_dir.mkdir()
    for name in ("2026-01-01T00-00-00Z_aaa.jsonl", "2026-02-01T00-00-00Z_bbb.jsonl"):
        (session_dir / name).write_text("{}\n", encoding="utf-8")
    assert resolve_session(session_dir, None).name.endswith("bbb.jsonl")
    assert resolve_session(session_dir, "aaa").name.endswith("aaa.jsonl")
    assert resolve_session(session_dir, "zzz") is None
    assert len(list_sessions(session_dir)) == 2


# -- rendering ----------------------------------------------------------------


def test_sparkline_is_plain_text_and_bounded() -> None:
    line = render.sparkline(list(range(500)), width=40)
    assert len(line) == 40
    assert "\x1b" not in line


def test_sparkline_handles_flat_and_empty_input() -> None:
    assert render.sparkline([]) == ""
    assert set(render.sparkline([5, 5, 5])) == {"▁"}


def test_table_omits_a_header_row_that_says_nothing() -> None:
    assert len(render.table(["", ""], [["a", "b"]])) == 1
    assert len(render.table(["x", "y"], [["a", "b"]])) == 2


def test_capped_reports_the_overflow() -> None:
    assert render.capped([1, 2, 3, 4], 2) == ([1, 2], 2)
    assert render.capped([1, 2], None) == ([1, 2], 0)


# -- the CLI ------------------------------------------------------------------


def _build_store(tmp_path: Path, run_id: str, presses: list[int], milestone: str) -> Path:
    registry = RunRegistry(tmp_path, fsync_every=0)
    registry.start_run(
        harness_sha="deadbeef",
        config_hash="cfg",
        model="test-model",
        start_checkpoint=None,
        goal="test goal",
        run_id=run_id,
        started_at=1_787_698_000.0,
    )
    for index, count in enumerate(presses):
        registry.append(
            run_id,
            {
                "seq": index,
                "t": 1_787_698_000.0 + index,
                "presses": count,
                "map": "Route 3",
                "pos": [index, 1],
                "moved": 1,
                "milestones_new": [milestone] if index == len(presses) - 1 else [],
                "tool": "action",
                "exit": 0,
            },
        )
    registry.close_all()
    return tmp_path


def test_cli_diff_puts_two_runs_side_by_side(tmp_path: Path, capsys) -> None:
    from pokemon_agent.scope.runs import ladder_ids

    milestone = ladder_ids()[0]
    _build_store(tmp_path, "run-a", [10, 10, 10], milestone)
    _build_store(tmp_path, "run-b", [5, 5, 5], milestone)
    code = main(
        ["diff", "run-a", "run-b", "--data-dir", str(tmp_path), "--workspace", str(tmp_path)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "run-a" in out and "run-b" in out
    assert "presses" in out
    assert "-15 better" in out  # b spent fifteen fewer presses


def test_cli_diff_needs_two_run_ids(tmp_path: Path, capsys) -> None:
    assert main(["diff", "only-one", "--data-dir", str(tmp_path)]) == 2
    assert "two run ids" in capsys.readouterr().err


def test_cli_reports_a_missing_workspace_instead_of_crashing(tmp_path: Path, capsys) -> None:
    code = main(["tools", "--workspace", str(tmp_path / "nowhere"), "--data-dir", str(tmp_path)])
    assert code == 2
    assert "no session transcript" in capsys.readouterr().err


def test_cli_reports_an_empty_run_store(tmp_path: Path, capsys) -> None:
    code = main(["timeline", "--data-dir", str(tmp_path), "--workspace", str(tmp_path)])
    assert code == 2
    assert "no runs" in capsys.readouterr().err


def test_cli_timeline_and_json_on_a_built_store(tmp_path: Path, capsys) -> None:
    from pokemon_agent.scope.runs import ladder_ids

    _build_store(tmp_path, "run-a", [7, 7, 7], ladder_ids()[1])
    assert (
        main(
            [
                "timeline",
                "--run",
                "run-a",
                "--data-dir",
                str(tmp_path),
                "--workspace",
                str(tmp_path),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "21" in text
    assert (
        main(
            [
                "timeline",
                "--run",
                "run-a",
                "--data-dir",
                str(tmp_path),
                "--workspace",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_presses"] == 21
    assert payload["milestones"][0]["presses"] == 21


def test_cli_waste_on_a_built_store(tmp_path: Path, capsys) -> None:
    from pokemon_agent.scope.runs import ladder_ids

    _build_store(tmp_path, "run-a", [4, 4], ladder_ids()[0])
    assert main(["waste", "--data-dir", str(tmp_path), "--workspace", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"]["total_presses"] == 8


def test_current_marker_selects_the_open_run(tmp_path: Path) -> None:
    from pokemon_agent.scope.runs import current_run_id, ladder_ids, resolve_run_id

    _build_store(tmp_path, "run-a", [1], ladder_ids()[0])
    _build_store(tmp_path, "run-b", [1], ladder_ids()[0])
    (tmp_path / "runs" / "CURRENT").write_text("run-a\n", encoding="utf-8")
    assert current_run_id(tmp_path) == "run-a"
    assert resolve_run_id(tmp_path, None) == "run-a"
    assert resolve_run_id(tmp_path, "run-b") == "run-b"


def test_a_stale_current_marker_falls_back_to_the_newest_run(tmp_path: Path) -> None:
    from pokemon_agent.scope.runs import ladder_ids, resolve_run_id

    _build_store(tmp_path, "run-a", [1], ladder_ids()[0])
    (tmp_path / "runs" / "CURRENT").write_text("run-gone\n", encoding="utf-8")
    assert resolve_run_id(tmp_path, None) == "run-a"


def test_phases_split_on_a_map_change(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(4):
        builder.turn("./poke act up", act_result(1, index, 1, map_name="Route 3"))
    for index in range(4):
        builder.turn("./poke act up", act_result(1, index, 1, map_name="Mt. Moon"))
    grouped = analysis.phases(parse_session(path), [])
    assert [phase.where for phase in grouped] == ["Route 3", "Mt. Moon"]
    assert all(phase.boundary == "map" for phase in grouped)


def test_phases_split_on_a_harness_stall(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(4):
        builder.turn("./poke act up", act_result(1, index, 1))
    builder.clock += analysis.STALL_SECONDS + 10
    for index in range(4):
        builder.turn("./poke act up", act_result(1, index, 1))
    grouped = analysis.phases(parse_session(path), [])
    assert len(grouped) == 2
    assert grouped[1].boundary == "stall"


def test_phases_price_a_stretch_in_presses_per_new_tile(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    builder = TranscriptBuilder(path)
    for index in range(4):
        builder.turn("./poke act up", act_result(1, index, 1))
    session = parse_session(path)
    # The window a phase covers runs from its first result to its last.
    start = min(call.ended_at for call in session.calls)
    receipts = [
        receipt(seq=index, t=start + index * 0.1, presses=10, pos=[1, index % 2])
        for index in range(4)
    ]
    phase = analysis.phases(session, receipts)[0]
    assert phase.presses == 40
    assert phase.new_tiles == 2  # (1,0) and (1,1); the rest is ground already walked
    assert phase.presses_per_new_tile == 20.0


def test_cli_session_and_loops_on_a_synthetic_workspace(tmp_path: Path, capsys) -> None:
    from pokemon_agent.scope.runs import ladder_ids

    workspace = tmp_path / "ws"
    (workspace / "pi-session").mkdir(parents=True)
    builder = TranscriptBuilder(workspace / "pi-session" / "2026-01-01T00-00-00Z_s.jsonl")
    builder.user("cross Route 3")
    for index in range(6):
        builder.turn("./poke act up", act_result(1, index, 1))
        builder.turn("./poke act down", act_result(1, index, 1))
    _build_store(tmp_path, "run-a", [1] * 12, ladder_ids()[0])
    common = ["--workspace", str(workspace), "--data-dir", str(tmp_path)]
    assert main(["loops", *common]) == 0
    loops_out = capsys.readouterr().out
    # The ``poke`` every token shares is dropped from the display, not the data.
    assert "act up | act down" in loops_out
    assert "right now:" in loops_out
    assert main(["session", *common]) == 0
    session_out = capsys.readouterr().out
    assert "cross Route 3" in session_out
    assert "phases" in session_out


# -- against the data actually on this disk -----------------------------------

REAL = discover()
HAS_SESSIONS = bool(list_sessions(REAL.session_dir))
HAS_RUNS = REAL.data_dir is not None and (REAL.data_dir / "runs").is_dir()

needs_sessions = pytest.mark.skipif(not HAS_SESSIONS, reason="no session transcripts on this disk")
needs_runs = pytest.mark.skipif(not HAS_RUNS, reason="no run store on this disk")

SESSION_COMMANDS = ("tools", "loops", "context", "session")
RUN_COMMANDS = ("waste", "timeline")

#: A report that does not fit on a screen has failed at its only job.
MAX_LINES = 60


def _looks_like_base64(text: str) -> bool:
    """A long unbroken run of base64 alphabet is an image that escaped."""

    run = 0
    for char in text:
        if char.isalnum() or char in "+/=":
            run += 1
            if run > 120:
                return True
        else:
            run = 0
    return False


@needs_sessions
@pytest.mark.parametrize("command", SESSION_COMMANDS)
def test_real_session_commands_are_short_and_clean(command: str, capsys) -> None:
    assert main([command]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert len(out.splitlines()) <= MAX_LINES, f"{command} printed {len(out.splitlines())} lines"
    assert "\x1b" not in out
    assert not _looks_like_base64(out)


@needs_runs
@pytest.mark.parametrize("command", RUN_COMMANDS)
def test_real_run_commands_are_short_and_clean(command: str, capsys) -> None:
    assert main([command]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= MAX_LINES
    assert "\x1b" not in out


@needs_sessions
@needs_runs
def test_real_live_command(capsys) -> None:
    assert main(["live"]) == 0
    out = capsys.readouterr().out
    assert "LIVE" in out
    assert len(out.splitlines()) <= MAX_LINES


@needs_sessions
@pytest.mark.parametrize("command", SESSION_COMMANDS + ("where",))
def test_real_json_output_is_valid(command: str, capsys) -> None:
    assert main([command, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)


@needs_sessions
def test_every_real_transcript_parses(capsys) -> None:
    for path in list_sessions(REAL.session_dir):
        session = parse_session(path)
        assert session.corrupt_lines == 0, f"{path.name} has damaged lines"
        assert session.steps, f"{path.name} produced no steps"
        report = analysis.tool_report(session)
        assert report.total_calls == len(session.calls)
        assert len(report.advisory) == len(ADVISORY_VERBS)


@needs_sessions
def test_the_live_transcript_is_read_while_it_grows() -> None:
    """The newest file is being appended to; two reads must both succeed."""

    path = list_sessions(REAL.session_dir)[-1]
    first = parse_session(path)
    second = parse_session(path)
    assert first.corrupt_lines == 0 and second.corrupt_lines == 0
    assert len(second.calls) >= len(first.calls)
    assert analysis.find_loops(second) is not None


@needs_runs
def test_the_real_run_reads_back_consistently() -> None:
    from pokemon_agent.scope.runs import resolve_run_id, run_metrics

    run_id = resolve_run_id(REAL.data_dir, None)
    record = RunRegistry(REAL.data_dir).load(run_id)
    metrics = run_metrics(record)
    waste = analysis.waste_report(record, ContextOracle([]))
    assert waste.overall.total_presses == metrics.total_presses
    assert ladder_progress(record).total == 63


@needs_runs
@needs_sessions
def test_reading_the_real_run_log_is_fast() -> None:
    """The run log is megabytes; the substring filter keeps it off the hot path."""

    from pokemon_agent.scope.runs import read_action_contexts

    start = time.perf_counter()
    contexts = read_action_contexts(REAL.run_log)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"reading the run log took {elapsed:.2f}s"
    assert isinstance(contexts, list)


@needs_sessions
def test_a_real_transcript_truncated_mid_line_still_parses(tmp_path: Path) -> None:
    source = list_sessions(REAL.session_dir)[-1]
    raw = source.read_bytes()
    clipped = tmp_path / source.name
    clipped.write_bytes(raw[: int(len(raw) * 0.9)])
    session = parse_session(clipped)
    assert session.corrupt_lines == 0
    assert session.steps
    assert analysis.context_report(session).steps == len(session.steps)


@needs_sessions
def test_commands_module_exposes_a_handler_for_every_cli_command() -> None:
    from pokemon_agent.scope.__main__ import COMMANDS

    for name in COMMANDS:
        if name == "diff":
            assert hasattr(commands, "command_diff")
            continue
        assert hasattr(commands, f"command_{name}"), name
