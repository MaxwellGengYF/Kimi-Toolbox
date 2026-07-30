"""Tests for the P0 benchmark infrastructure (runner, report, attribution, profile)."""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import pytest

pytest.importorskip("tools.bench")

from tools.bench.attribute_failures import (
    attribute_runs,
    build_attribution_report,
    classify_failure_heuristic,
)
from tools.bench.profile import load_profile
from tools.bench.report import (
    aggregate,
    compare_groups,
    count_feature_triggers,
    load_run_records,
    stratified_bootstrap_ci,
)
from tools.bench.run_bench import AgentRunStats, BenchTask, load_tasks, run_bench


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tasks(path: Path) -> list[BenchTask]:
    """Two mock tasks: one whose verify passes, one whose verify fails."""
    ok_cmd = f'"{sys.executable}" -c "import sys; sys.exit(0)"'
    fail_cmd = f'"{sys.executable}" -c "import sys; sys.exit(1)"'
    lines = [
        {"id": "task-ok", "prompt": "do ok", "verify_command": ok_cmd, "timeout_s": 60},
        {"id": "task-fail", "prompt": "do fail", "verify_command": fail_cmd, "timeout_s": 60},
    ]
    path.write_text("\n".join(orjson.dumps(line).decode() for line in lines), encoding="utf-8")
    return load_tasks(path)


def _mock_runner_factory(crash_on: set[tuple[str, int]] | None = None):
    """Agent runner that simulates steps/tokens and writes a wire.jsonl.

    ``crash_on`` holds (prompt, call_index) pairs that should raise.
    """
    crash_on = crash_on or set()
    calls: dict[str, int] = {}

    async def _runner(prompt: str, work_dir: Path, run_dir: Path, timeout_s: int) -> AgentRunStats:
        idx = calls.get(prompt, 0)
        calls[prompt] = idx + 1
        if (prompt, idx) in crash_on:
            raise RuntimeError("simulated agent crash")
        (run_dir / "wire.jsonl").write_text(
            '{"type":"StepBegin","payload":{"n":1}}\n'
            '{"type":"StatusUpdate","payload":{"token_usage":{"input":10,"output":5}}}\n',
            encoding="utf-8",
        )
        return AgentRunStats(steps=3, input_tokens=100, output_tokens=50)

    return _runner


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------


def test_load_tasks_valid(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    assert [t.id for t in tasks] == ["task-ok", "task-fail"]
    assert tasks[0].timeout_s == 60
    assert tasks[0].setup_commands == []


def test_load_tasks_missing_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(orjson.dumps({"id": "x"}).decode(), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_tasks(bad)


def test_load_tasks_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        load_tasks(empty)


# ---------------------------------------------------------------------------
# run_bench: structure, judgment, crash isolation
# ---------------------------------------------------------------------------


async def test_run_bench_structure_and_judgment(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    results = await run_bench(
        tasks, 2, out, _mock_runner_factory(), concurrency=2,
        config_snapshot={"seed": 0},
    )
    assert len(results) == 4

    # Landing structure: result.json + wire.jsonl per run.
    for task in tasks:
        for run_idx in range(2):
            run_dir = out / task.id / str(run_idx)
            assert (run_dir / "result.json").exists()
            assert (run_dir / "wire.jsonl").exists()
            result = orjson.loads((run_dir / "result.json").read_bytes())
            assert result["task_id"] == task.id
            assert result["run_idx"] == run_idx
            assert result["steps"] == 3
            assert result["input_tokens"] == 100
            assert result["config_snapshot"] == {"seed": 0}

    # Judgment: task-ok always solved, task-fail never.
    ok_results = [r for r in results if r.task_id == "task-ok"]
    fail_results = [r for r in results if r.task_id == "task-fail"]
    assert all(r.solved for r in ok_results)
    assert all(not r.solved for r in fail_results)
    assert all(r.error_category == "unsolved" for r in fail_results)


async def test_run_bench_crash_isolation(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    # Crash only the second run of task-ok; everything else must proceed.
    runner = _mock_runner_factory(crash_on={("do ok", 1)})
    results = await run_bench(tasks, 2, out, runner, concurrency=1)

    crashed = [r for r in results if r.task_id == "task-ok" and r.run_idx == 1]
    assert len(crashed) == 1
    assert crashed[0].error_category == "crash"
    assert not crashed[0].solved
    assert (out / "task-ok" / "1" / "crash.txt").exists()

    # The crash did not affect other runs.
    others = [r for r in results if not (r.task_id == "task-ok" and r.run_idx == 1)]
    assert all(r.error_category != "crash" for r in others)
    ok_first = [r for r in results if r.task_id == "task-ok" and r.run_idx == 0]
    assert ok_first[0].solved


async def test_run_bench_setup_failure(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    tasks[0].setup_commands = [f'"{sys.executable}" -c "import sys; sys.exit(3)"']
    out = tmp_path / "bench_runs"
    results = await run_bench(tasks, 1, out, _mock_runner_factory(), concurrency=1)
    setup_failed = [r for r in results if r.task_id == "task-ok"]
    assert setup_failed[0].error_category == "setup_failed"
    assert not setup_failed[0].solved
    assert "setup command failed" in setup_failed[0].verify_output_tail


# ---------------------------------------------------------------------------
# report: aggregation + bootstrap CI + triggers + groups
# ---------------------------------------------------------------------------


async def test_report_aggregation(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    await run_bench(tasks, 5, out, _mock_runner_factory(), concurrency=2)

    records = load_run_records(out)
    assert len(records) == 10

    agg = aggregate(records, runs_dir=out, seed=0)
    assert agg.n_runs == 10
    assert agg.n_tasks == 2
    assert agg.solve_rate == 0.5
    assert 0.0 <= agg.ci_low <= agg.solve_rate <= agg.ci_high <= 1.0
    assert agg.avg_steps == 3.0
    assert agg.avg_input_tokens == 100.0


def test_bootstrap_ci_deterministic() -> None:
    from tools.bench.report import RunRecord

    records = [
        RunRecord(f"t{i}", j, solved=(j % 2 == 0), steps=1, input_tokens=0,
                  output_tokens=0, elapsed_s=0.0, error_category=None)
        for i in range(4)
        for j in range(3)
    ]
    ci1 = stratified_bootstrap_ci(records, n_boot=200, seed=42)
    ci2 = stratified_bootstrap_ci(records, n_boot=200, seed=42)
    assert ci1 == ci2
    assert ci1[0] <= ci1[1]


def test_count_feature_triggers(tmp_path: Path) -> None:
    run_dir = tmp_path / "t1" / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "wire.jsonl").write_text(
        "you are repeating the exact same tool call\nunfinished todolist tasks\n",
        encoding="utf-8",
    )
    counts = count_feature_triggers(tmp_path)
    assert counts["repeat_reminder"] == 1
    assert counts["todo_reminder"] == 1
    assert counts["compact_reminder"] == 0


def test_compare_groups() -> None:
    from tools.bench.report import RunRecord

    def _recs(solved_flags: list[bool]) -> list:
        return [
            RunRecord("t", i, solved=s, steps=1, input_tokens=0,
                      output_tokens=100, elapsed_s=0.0, error_category=None)
            for i, s in enumerate(solved_flags)
        ]

    table = compare_groups({"a": _recs([True, True]), "b": _recs([False, True])}, seed=0)
    assert table["a"]["solve_rate"] == 1.0
    assert table["b"]["solve_rate"] == 0.5
    assert table["a"]["cost_adjusted_solve_rate"] > table["b"]["cost_adjusted_solve_rate"]


# ---------------------------------------------------------------------------
# attribution: heuristic classifier + batch
# ---------------------------------------------------------------------------


def test_classify_failure_heuristic_verification() -> None:
    events = [
        {"type": "x", "payload": "verification failed: test_foo"},
        {"type": "y", "payload": "unfinished `TodoList` tasks remain"},
    ]
    category, confidence, evidence = classify_failure_heuristic(events)
    assert category == "verification"
    assert confidence > 0.5
    assert evidence


def test_classify_failure_heuristic_coherence() -> None:
    events = [
        {"type": "x", "payload": "you are repeating the exact same tool call"},
        {"type": "y", "payload": "same tool called repeatedly with different args"},
    ]
    category, _, _ = classify_failure_heuristic(events)
    assert category == "coherence"


def test_classify_failure_heuristic_empty() -> None:
    category, confidence, _ = classify_failure_heuristic([])
    assert category == "coherence"
    assert confidence == pytest.approx(0.34)


async def test_attribute_runs_batch(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    await run_bench(tasks, 2, out, _mock_runner_factory(), concurrency=1)

    labels = await attribute_runs(out)
    # Only task-fail runs are labeled (2 of them).
    assert len(labels) == 2
    assert all(label.task_id == "task-fail" for label in labels)

    report = build_attribution_report(labels)
    assert report["total_failed_runs"] == 2
    assert report["per_task"]["task-fail"]["total_failures"] == 2
    total = sum(report["overall_categories"].values())
    assert total == 2


async def test_attribute_runs_crash_shortcircuit(tmp_path: Path) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    runner = _mock_runner_factory(crash_on={("do ok", 0)})
    await run_bench(tasks, 1, out, runner, concurrency=1)

    labels = await attribute_runs(out)
    crash_labels = [label for label in labels if label.task_id == "task-ok"]
    assert len(crash_labels) == 1
    assert crash_labels[0].category == "execution"
    assert crash_labels[0].confidence == 1.0


# ---------------------------------------------------------------------------
# profile loading (P10 foundation)
# ---------------------------------------------------------------------------


def test_load_profile_valid(tmp_path: Path) -> None:
    profile = tmp_path / "p.yaml"
    profile.write_text(
        "model: kimi\nruns_per_task: 3\nfeatures:\n  target_churn_enabled: true\n",
        encoding="utf-8",
    )
    data = load_profile(profile)
    assert data["model"] == "kimi"
    assert data["runs_per_task"] == 3
    assert data["features"]["target_churn_enabled"] is True


def test_load_profile_unknown_keys(tmp_path: Path) -> None:
    profile = tmp_path / "p.yaml"
    profile.write_text("bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_profile(profile)


def test_load_profile_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not readable"):
        load_profile(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# P10: profile application + config matrix
# ---------------------------------------------------------------------------


def test_apply_profile_to_config_merges_overrides() -> None:
    from kimi_cli.config import load_config

    from tools.bench.profile import apply_profile_to_config

    profile = {
        "loop_control_overrides": {"target_churn_enabled": False, "target_churn_file_warn": 9},
        "features": {"verification_gate_enabled": False},
    }
    config = apply_profile_to_config(profile, load_config())
    assert config.loop_control.target_churn_enabled is False
    assert config.loop_control.target_churn_file_warn == 9
    assert config.loop_control.verification_gate_enabled is False
    # Untouched keys keep their defaults.
    assert config.loop_control.budget_reminder_enabled is True


def test_apply_profile_to_config_rejects_unknown_fields() -> None:
    from kimi_cli.config import load_config

    from tools.bench.profile import apply_profile_to_config

    with pytest.raises(ValueError, match="unknown LoopControl fields"):
        apply_profile_to_config({"loop_control_overrides": {"nope": 1}}, load_config())


def test_apply_profile_noop_without_overrides() -> None:
    from kimi_cli.config import load_config

    from tools.bench.profile import apply_profile_to_config

    config = load_config()
    result = apply_profile_to_config({}, config)
    assert result is config
    assert result.loop_control.target_churn_enabled is True


async def test_config_matrix_groups_by_signature(tmp_path: Path) -> None:
    from tools.bench.report import build_config_matrix

    tasks = _write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "bench_runs"
    snapshot = {
        "model": "m1",
        "loop_control_overrides": {"target_churn_enabled": True},
    }
    await run_bench(
        tasks, 2, out, _mock_runner_factory(), concurrency=1,
        config_snapshot=snapshot,
    )
    records = load_run_records(out)
    matrix = build_config_matrix(records)
    assert len(matrix) == 1
    signature = next(iter(matrix))
    assert "model=m1" in signature
    assert "target_churn_enabled=True" in signature
    # task-ok solves both runs, task-fail solves none -> 0.5
    assert matrix[signature]["solve_rate"] == 0.5
    assert matrix[signature]["n_runs"] == 4
