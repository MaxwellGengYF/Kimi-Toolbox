"""Tests for best-of-N sampling + selection (P7)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kimix.tools.swarm.best_of_n import (
    AllCandidatesFailedError,
    VerificationRejectedError,
    best_of_n,
    collect_diff,
    create_worker_workspace,
    run_parallel_sample,
    select_best_candidate,
)


def _worker_index(worker_dir: Path) -> int:
    """Extract the worker index from the deterministic workspace prefix."""
    import regex as re

    match = re.search(r"best_of_n_(\d+)_", worker_dir.name)
    assert match, f"unexpected workspace name: {worker_dir.name}"
    return int(match.group(1))


def _make_runner(fail_on: set[int] | None = None):
    """Sample runner that writes a unique file per worker workspace."""
    fail_on = fail_on or set()
    calls: list[Path] = []

    async def _runner(prompt: str, worker_dir: Path) -> tuple[str, int, int]:
        calls.append(worker_dir)
        idx = _worker_index(worker_dir)
        if idx in fail_on:
            raise RuntimeError(f"worker {idx} boom")
        (worker_dir / f"solution_{idx}.txt").write_text(
            f"solution from worker {idx}\n", encoding="utf-8"
        )
        return f"report {idx}", 10 + idx, 100 * (idx + 1)

    return _runner, calls


async def test_parallel_sample_isolation_and_diffs(tmp_path: Path) -> None:
    runner, calls = _make_runner()
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    candidates = await run_parallel_sample("do the task", 4, tmp_path, runner)

    assert len(candidates) == 4
    assert all(c.success for c in candidates)
    assert len(set(calls)) == 4  # 4 distinct workspaces

    # Each candidate produced a diff containing only its own file.
    for c in candidates:
        assert f"solution_{c.index}.txt" in c.diff
        for other in range(4):
            if other != c.index:
                assert f"solution_{other}.txt" not in c.diff


async def test_best_of_n_self_eval_applies_and_verifies(tmp_path: Path) -> None:
    runner, _ = _make_runner()
    verify_calls: list[Path] = []

    async def _selector(prompt: str, review: str) -> int:
        return 2

    async def _verify(work_dir: Path) -> tuple[bool, str]:
        verify_calls.append(work_dir)
        ok = (work_dir / "solution_2.txt").exists()
        return ok, "solution_2 present" if ok else "missing"

    result = await best_of_n(
        "task", tmp_path, runner, _selector, n=4, verify_fn=_verify
    )
    assert result.winner_index == 2
    assert result.verified
    assert verify_calls == [tmp_path]
    # Winner applied to the main workspace; other candidates' files absent.
    assert (tmp_path / "solution_2.txt").read_text(encoding="utf-8") == "solution from worker 2\n"
    assert not (tmp_path / "solution_0.txt").exists()
    assert not (tmp_path / "solution_1.txt").exists()


async def test_best_of_n_all_candidates_failed(tmp_path: Path) -> None:
    runner, _ = _make_runner(fail_on={0, 1, 2, 3})

    async def _selector(prompt: str, review: str) -> int:
        raise AssertionError("selector must not be called")

    with pytest.raises(AllCandidatesFailedError, match="all 4 sampled candidates failed"):
        await best_of_n("task", tmp_path, runner, _selector, n=4)


async def test_best_of_n_degenerate_single_run(tmp_path: Path) -> None:
    runner, _ = _make_runner()
    selector_called = False

    async def _selector(prompt: str, review: str) -> int:
        nonlocal selector_called
        selector_called = True
        return 0

    result = await best_of_n("task", tmp_path, runner, _selector, n=1)
    assert result.winner_index == 0
    assert not selector_called  # no selection overhead for n=1


async def test_best_of_n_verification_rejected(tmp_path: Path) -> None:
    runner, _ = _make_runner()

    async def _selector(prompt: str, review: str) -> int:
        return 1

    async def _verify(work_dir: Path) -> tuple[bool, str]:
        return False, "tests failed"

    with pytest.raises(VerificationRejectedError, match="failed verification"):
        await best_of_n("task", tmp_path, runner, _selector, n=3, verify_fn=_verify)


async def test_majority_vote_selection(tmp_path: Path) -> None:
    runner, _ = _make_runner()
    candidates = await run_parallel_sample("task", 3, tmp_path, runner)

    async def _always_b(prompt: str, pair_text: str) -> int:
        # Pairwise text embeds "Candidate B (#j)" — always pick B.
        import regex as re

        match = re.search(r"Candidate B \(#(\d+)\)", pair_text)
        return int(match.group(1))

    winner, reason = await select_best_candidate(
        "task", candidates, _always_b, strategy="majority"
    )
    assert winner == 2  # candidate 2 beats both 0 and 1
    assert "majority vote" in reason


async def test_select_skips_failed_candidates(tmp_path: Path) -> None:
    runner, _ = _make_runner(fail_on={0, 2})
    candidates = await run_parallel_sample("task", 3, tmp_path, runner)
    assert [c.success for c in candidates] == [False, True, False]

    async def _selector(prompt: str, review: str) -> int:
        return 99  # invalid index -> falls back to the only viable one

    winner, reason = await select_best_candidate("task", candidates, _selector)
    assert winner == 1
    assert "only one viable candidate" in reason


async def test_cancelled_worker_propagates(tmp_path: Path) -> None:
    async def _runner(prompt: str, worker_dir: Path) -> tuple[str, int, int]:
        if _worker_index(worker_dir) == 1:
            raise asyncio.CancelledError()
        return "ok", 1, 1

    with pytest.raises(asyncio.CancelledError):
        await run_parallel_sample("task", 2, tmp_path, _runner)


def test_worker_workspace_copy_mode(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hello\n", encoding="utf-8")
    worker, kind = create_worker_workspace(tmp_path, 0)
    try:
        assert kind == "copy"  # tmp_path is not a git repo
        assert (worker / "file.txt").read_text(encoding="utf-8") == "hello\n"
        # Changes in the worker do not leak into the main workspace.
        (worker / "new.txt").write_text("new\n", encoding="utf-8")
        assert not (tmp_path / "new.txt").exists()
        diff = collect_diff(worker, kind, before_snapshot={})
        assert "new.txt" in diff
    finally:
        from kimix.tools.swarm.best_of_n import cleanup_worker_workspace

        cleanup_worker_workspace(worker, kind, tmp_path)
    assert not worker.exists()
