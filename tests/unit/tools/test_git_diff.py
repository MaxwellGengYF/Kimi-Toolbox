"""Tests for tools/git_diff.py — the agent's diff-review tool.

Regression target: ``git diff -- <file>`` prints NOTHING (exit 0) for an
*untracked* file, so a plain wrapper silently "verifies" files it never
looked at — the agent misreads the empty output as "no changes".  The tool
must never return an empty string for an existing path: untracked files are
rendered as full ``new file`` diffs and unchanged/ignored/missing paths are
labelled explicitly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import git_diff


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo (with one committed file) as the cwd."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.name", "test")
    _run_git(repo, "config", "user.email", "test@test")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-qm", "init")
    monkeypatch.chdir(repo)
    return repo


def test_empty_input_returns_empty() -> None:
    """The only legitimate empty result: nothing was asked for."""
    assert git_diff.get_uncommitted_diff([]) == ""


def test_tracked_modified_shows_diff(repo: Path) -> None:
    (repo / "tracked.txt").write_text("base\nchanged\n", encoding="utf-8")
    out = git_diff.get_uncommitted_diff(["tracked.txt"])
    assert "tracked.txt" in out
    assert "+changed" in out


def test_tracked_unchanged_is_not_silent(repo: Path) -> None:
    """`git diff` on an unchanged tracked file is empty — the tool must say
    so instead of returning empty output."""
    out = git_diff.get_uncommitted_diff(["tracked.txt"])
    assert out.strip(), "an existing path must never produce a silent empty diff"
    assert "no uncommitted changes" in out


def test_untracked_rendered_as_new_file(repo: Path) -> None:
    """Regression (the original bug): `git diff -- <untracked>` is empty; the
    tool renders the file as a full new-file diff instead."""
    (repo / "new.txt").write_text("fresh content\n", encoding="utf-8")
    out = git_diff.get_uncommitted_diff(["new.txt"])
    assert "UNTRACKED (new file)" in out
    assert "+fresh content" in out


def test_ignored_flagged(repo: Path) -> None:
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "app.log").write_text("noise\n", encoding="utf-8")
    out = git_diff.get_uncommitted_diff(["app.log"])
    assert "IGNORED by .gitignore" in out


def test_missing_flagged(repo: Path) -> None:
    out = git_diff.get_uncommitted_diff(["does_not_exist.txt"])
    assert "MISSING" in out


def test_directory_expands_untracked_files(repo: Path) -> None:
    sub = repo / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("hello\n", encoding="utf-8")
    out = git_diff.get_uncommitted_diff(["sub"])
    assert "UNTRACKED (new file)" in out
    assert "+hello" in out


def test_not_a_repo_shows_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a git work tree there is no baseline — the tool still shows the
    file content instead of silently returning empty."""
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "notes.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.chdir(plain)
    out = git_diff.get_uncommitted_diff(["notes.txt"])
    assert "NOT A GIT REPO" in out
    assert "+hello" in out
