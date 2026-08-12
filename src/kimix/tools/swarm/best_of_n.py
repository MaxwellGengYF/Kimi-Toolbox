"""Best-of-N sampling + trajectory selection (P7, gap G8).

Extends the swarm concept from "task-decomposition parallelism" to
"same-task multi-sampling + selection":

1. **Sample**: the same task prompt runs in N independent worker
   workspaces (git worktree when inside a git repo, otherwise a temp
   copy), so concurrent writes never pollute each other.
2. **Select**: a selector picks the winning candidate —
   ``self_eval`` (default, one model review call over all candidates)
   or ``majority`` (pairwise comparisons, N >= 3).
3. **Verify**: the winning diff is applied to the main workspace and a
   verification callable runs; failure or an all-failed sample set is an
   explicit error, never a silent accept.

All LLM-facing pieces (worker runner, selector review, verification) are
injectable callables so the machinery is fully testable offline.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SampleCandidate:
    """One sampled run's outcome."""

    index: int
    work_dir: Path
    diff: str = ""
    self_report: str = ""
    steps: int = 0
    output_tokens: int = 0
    success: bool = False
    error: str | None = None


# Runs one sample: (task_prompt, worker_work_dir) -> (self_report, steps, output_tokens).
SampleRunner = Callable[[str, Path], Awaitable[tuple[str, int, int]]]

# Reviews candidates: (task_prompt, formatted_candidates) -> chosen index (0-based).
SelectorFn = Callable[[str, str], Awaitable[int]]

# Verifies the applied result in the main workspace: work_dir -> (ok, detail).
VerifyFn = Callable[[Path], Awaitable[tuple[bool, str]]]


@dataclass(slots=True)
class BestOfNResult:
    """Outcome of the full sample -> select -> verify loop."""

    winner_index: int
    selection_reason: str
    candidates: list[SampleCandidate] = field(default_factory=list)
    verified: bool = False
    verify_detail: str = ""


class AllCandidatesFailedError(RuntimeError):
    """Every sampled worker failed — nothing to select."""


class VerificationRejectedError(RuntimeError):
    """The selected candidate failed post-application verification."""


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------


def _is_git_repo(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


_COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".kimix_cache",
    "bench_runs", ".mypy_cache", ".pytest_cache", ".ruff_cache",
)


def create_worker_workspace(work_dir: Path, index: int) -> tuple[Path, str]:
    """Create an isolated worker workspace.

    Returns ``(worker_path, kind)`` where kind is ``"worktree"`` or ``"copy"``.
    """
    if _is_git_repo(work_dir):
        worker_path = Path(tempfile.mkdtemp(prefix=f"best_of_n_wt_{index}_"))
        # git worktree add requires the target to not exist yet.
        worker_path.rmdir()
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worker_path), "HEAD"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return worker_path, "worktree"
    worker_path = Path(tempfile.mkdtemp(prefix=f"best_of_n_{index}_"))
    shutil.copytree(work_dir, worker_path, dirs_exist_ok=True, ignore=_COPY_IGNORE)
    return worker_path, "copy"


def cleanup_worker_workspace(worker_path: Path, kind: str, main_work_dir: Path) -> None:
    """Remove a worker workspace created by :func:`create_worker_workspace`."""
    try:
        if kind == "worktree":
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worker_path)],
                cwd=str(main_work_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
        if worker_path.exists():
            shutil.rmtree(worker_path, ignore_errors=True)
    except Exception:
        shutil.rmtree(worker_path, ignore_errors=True)


def _snapshot_files(root: Path) -> dict[str, bytes]:
    """Snapshot relative path -> content bytes for diff computation."""
    import xxhash  # noqa: F401 — fast hashing available if needed later

    snapshot: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in (".git", ".venv", "venv", "node_modules", "__pycache__")
        ]
        for filename in filenames:
            full = Path(dirpath) / filename
            rel = str(full.relative_to(root))
            try:
                snapshot[rel] = full.read_bytes()
            except OSError:
                continue
    return snapshot


def collect_diff(worker_path: Path, kind: str, before_snapshot: dict[str, bytes] | None = None) -> str:
    """Compute the worker's changes as a unified diff string."""
    if kind == "worktree":
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(worker_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        diff = result.stdout or ""
        # Include untracked files.
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(worker_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        for rel in (untracked.stdout or "").splitlines():
            rel = rel.strip()
            if not rel:
                continue
            full = worker_path / rel
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            diff += "".join(
                difflib.unified_diff(
                    [], content.splitlines(keepends=True),
                    fromfile="/dev/null", tofile=rel,
                )
            )
        return diff

    # Copy mode: diff before/after snapshots textually.
    before = before_snapshot or {}
    after = _snapshot_files(worker_path)

    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        old = before.get(rel, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        new = after.get(rel, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(old, new, fromfile=f"a/{rel}", tofile=f"b/{rel}")
        )
    return "".join(chunks)


def apply_diff_to_workspace(winner: SampleCandidate, main_work_dir: Path, kind: str) -> None:
    """Apply the winning candidate's changes onto the main workspace.

    For copy-mode workers the worker tree (minus ignored dirs) is copied
    back over the main workspace. For worktree mode the worker's changed
    files are copied back (deletions applied too).
    """
    if kind == "worktree":
        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=str(winner.work_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        names = [n.strip() for n in (changed.stdout or "").splitlines() if n.strip()]
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(winner.work_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        names += [n.strip() for n in (untracked.stdout or "").splitlines() if n.strip()]
        for rel in names:
            src = winner.work_dir / rel
            dst = main_work_dir / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            elif dst.exists():
                dst.unlink()
        return
    shutil.copytree(winner.work_dir, main_work_dir, dirs_exist_ok=True, ignore=_COPY_IGNORE)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def run_parallel_sample(
    task_prompt: str,
    n: int,
    work_dir: Path,
    runner: SampleRunner,
    *,
    max_concurrency: int = 5,
) -> list[SampleCandidate]:
    """Run the same task prompt in N isolated worker workspaces."""
    if n < 1:
        raise ValueError("n must be >= 1")
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(index: int) -> SampleCandidate:
        async with semaphore:
            worker_path, kind = create_worker_workspace(work_dir, index)
            before: dict[str, bytes] | None = None
            if kind == "copy":
                before = _snapshot_files(worker_path)
            candidate = SampleCandidate(index=index, work_dir=worker_path)
            try:
                report, steps, tokens = await runner(task_prompt, worker_path)
                candidate.self_report = report
                candidate.steps = steps
                candidate.output_tokens = tokens
                candidate.diff = collect_diff(worker_path, kind, before)
                candidate.success = True
            except asyncio.CancelledError:
                candidate.error = "cancelled"
                cleanup_worker_workspace(worker_path, kind, work_dir)
                raise
            except Exception as exc:  # noqa: BLE001 — isolation per worker
                candidate.error = f"{type(exc).__name__}: {exc}"
                candidate.success = False
            candidate.work_dir = worker_path
            # Stash kind on the candidate via diff marker for apply step.
            candidate.diff = f"[workspace:{kind}]\n" + candidate.diff
            return candidate

    return await asyncio.gather(*(_one(i) for i in range(n)))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def format_candidates_for_review(candidates: list[SampleCandidate]) -> str:
    """Render candidates (diff + self-report) for a selector review call."""
    parts: list[str] = []
    for c in candidates:
        status = "ok" if c.success else f"failed: {c.error}"
        parts.append(
            f"=== Candidate {c.index} ({status}, {c.steps} steps) ===\n"
            f"Self-report:\n{c.self_report}\n\nDiff:\n{c.diff}"
        )
    return "\n\n".join(parts)


async def select_best_candidate(
    task_prompt: str,
    candidates: list[SampleCandidate],
    selector_fn: SelectorFn,
    *,
    strategy: str = "self_eval",
) -> tuple[int, str]:
    """Pick the winning candidate index.

    ``self_eval``: one review call over all candidates.
    ``majority``: pairwise comparisons aggregated into votes (N >= 3).
    """
    viable = [c for c in candidates if c.success]
    if not viable:
        raise AllCandidatesFailedError(
            f"all {len(candidates)} sampled candidates failed: "
            + "; ".join(f"#{c.index}: {c.error}" for c in candidates)
        )
    if len(viable) == 1:
        return viable[0].index, "only one viable candidate"

    if strategy == "majority" and len(viable) >= 3:
        votes: dict[int, int] = {c.index: 0 for c in viable}
        for i in range(len(viable)):
            for j in range(i + 1, len(viable)):
                a, b = viable[i], viable[j]
                pair_text = (
                    f"=== Candidate A (#{a.index}) ===\n{a.diff}\n"
                    f"=== Candidate B (#{b.index}) ===\n{b.diff}"
                )
                chosen = await selector_fn(task_prompt, pair_text)
                # selector returns the candidate *index* it prefers.
                if chosen == b.index:
                    votes[b.index] += 1
                else:
                    votes[a.index] += 1
        winner = max(votes, key=lambda idx: (votes[idx], -idx))
        return winner, f"majority vote {votes}"

    review_text = format_candidates_for_review(viable)
    chosen = await selector_fn(task_prompt, review_text)
    viable_indices = {c.index for c in viable}
    if chosen not in viable_indices:
        # Defensive: fall back to the first viable candidate.
        return viable[0].index, f"selector returned invalid index {chosen}; fell back"
    return chosen, "self-eval selection"


# ---------------------------------------------------------------------------
# Full loop: sample -> select -> apply -> verify
# ---------------------------------------------------------------------------


async def best_of_n(
    task_prompt: str,
    work_dir: Path,
    runner: SampleRunner,
    selector_fn: SelectorFn,
    *,
    n: int = 4,
    strategy: str = "self_eval",
    verify_fn: VerifyFn | None = None,
    max_concurrency: int = 5,
) -> BestOfNResult:
    """Run the complete best-of-N loop. Never silently accepts failure."""
    if n <= 1:
        # Degenerate: a single plain run, no selection overhead.
        candidates = await run_parallel_sample(
            task_prompt, 1, work_dir, runner, max_concurrency=1
        )
        candidate = candidates[0]
        if not candidate.success:
            raise AllCandidatesFailedError(f"single run failed: {candidate.error}")
        return BestOfNResult(
            winner_index=candidate.index,
            selection_reason="n=1: no selection",
            candidates=candidates,
        )

    candidates = await run_parallel_sample(
        task_prompt, n, work_dir, runner, max_concurrency=max_concurrency
    )
    winner_index, reason = await select_best_candidate(
        task_prompt, candidates, selector_fn, strategy=strategy
    )
    winner = next(c for c in candidates if c.index == winner_index)

    kind = "worktree" if winner.diff.startswith("[workspace:worktree]") else "copy"
    apply_diff_to_workspace(winner, work_dir, kind)

    verified = False
    verify_detail = ""
    if verify_fn is not None:
        ok, verify_detail = await verify_fn(work_dir)
        verified = ok
        if not ok:
            raise VerificationRejectedError(
                f"selected candidate #{winner_index} failed verification: {verify_detail}"
            )

    return BestOfNResult(
        winner_index=winner_index,
        selection_reason=reason,
        candidates=candidates,
        verified=verified,
        verify_detail=verify_detail,
    )
