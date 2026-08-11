"""Show uncommitted diffs for files, including NEW (untracked) files.

``git diff -- <file>`` prints NOTHING and exits 0 for an *untracked* file, so
a plain wrapper would silently "verify" a file it never looked at — the agent
misreads the empty output as "no changes" and can even conclude its edits
were never applied.  This tool closes that gap:

- every existing target is classified (``tracked`` / ``untracked`` /
  ``ignored`` / ``missing`` / ``not a git repo``) and gets an explicit
  status section — the output is never silently empty for an existing path;
- untracked files are rendered as a full ``new file`` diff
  (``git diff --no-index /dev/null <file>``), so newly created files are
  actually reviewed;
- directories are expanded: tracked modifications plus every untracked
  non-ignored file inside (``git ls-files --others --exclude-standard``);
- ignored files are called out so the agent knows they will not be committed.

Usage::

    uv run tools/git_diff.py <filepath> [<filepath> ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _git(args: list[str]) -> tuple[int, str, str]:
    """Run git with *args* in the current working directory."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
        errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


def _in_repo() -> bool:
    """True when the current working directory is inside a git work tree."""
    rc, _, _ = _git(["rev-parse", "--is-inside-work-tree"])
    return rc == 0


def classify_path(filepath: str, in_repo: bool) -> str:
    """Classify a target's git state: tracked | untracked | ignored | missing | no-repo."""
    path = Path(filepath)
    if not path.exists():
        return "missing"
    if not in_repo:
        return "no-repo"
    rc, _, _ = _git(["ls-files", "--error-unmatch", "--", str(path)])
    if rc == 0:
        return "tracked"
    rc, _, _ = _git(["check-ignore", "-q", "--", str(path)])
    if rc == 0:
        return "ignored"
    return "untracked"


def _new_file_diff(filepath: str) -> str:
    """Render a file as a full ``new file`` diff.

    ``git diff --no-index /dev/null <file>`` exits 1 when differences are
    found (the normal case here); exit 0 means the file is empty.  The path
    is passed through as given (usually repo-relative) so the diff header
    stays readable; absolute inputs still work.
    """
    path = Path(filepath)
    target = str(path) if path.exists() else str(path.absolute())
    rc, out, err = _git(["diff", "--no-index", "--", "/dev/null", target])
    if rc not in (0, 1):
        raise RuntimeError(f"git diff --no-index failed for {filepath}: {err.strip()}")
    return out


def _section(title: str, body: str) -> str:
    text = f"### {title}\n{body}".rstrip()
    return text + "\n" if text else ""


def _diff_for_file(filepath: str, in_repo: bool) -> str:
    state = classify_path(filepath, in_repo)
    if state == "missing":
        return _section(f"{filepath}: MISSING (path does not exist)", "")
    if state == "no-repo":
        # Outside a git work tree there is no baseline to diff against —
        # show the whole file so the agent still sees what it asked to verify.
        return _section(
            f"{filepath}: NOT A GIT REPO (no baseline — showing full content)",
            _new_file_diff(filepath),
        )
    if state == "ignored":
        return _section(
            f"{filepath}: IGNORED by .gitignore (changes will not be committed)", ""
        )
    if state == "untracked":
        return _section(f"{filepath}: UNTRACKED (new file)", _new_file_diff(filepath))
    rc, out, err = _git(["diff", "--", filepath])
    if rc != 0:
        raise RuntimeError(f"git diff failed for {filepath}: {err.strip()}")
    if out.strip():
        return _section(filepath, out)
    return _section(f"{filepath}: no uncommitted changes", "")


def _diff_for_dir(dirpath: str) -> str:
    """Diff a directory: tracked modifications + untracked non-ignored files."""
    parts: list[str] = []
    rc, out, err = _git(["diff", "--", dirpath])
    if rc != 0:
        raise RuntimeError(f"git diff failed for {dirpath}: {err.strip()}")
    if out.strip():
        parts.append(_section(dirpath, out))
    rc, untracked, err = _git(
        ["ls-files", "--others", "--exclude-standard", "--", dirpath]
    )
    if rc != 0:
        raise RuntimeError(f"git ls-files failed for {dirpath}: {err.strip()}")
    for line in untracked.splitlines():
        if not line:
            continue
        parts.append(
            _section(f"{dirpath} -> {line}: UNTRACKED (new file)", _new_file_diff(line))
        )
    rc, _, _ = _git(["check-ignore", "-q", "--", dirpath])
    if rc == 0:
        parts.append(
            _section(f"{dirpath}: IGNORED by .gitignore (changes will not be committed)", "")
        )
    if not parts:
        return _section(f"{dirpath}: no uncommitted changes", "")
    return "\n".join(parts)


def get_uncommitted_diff(filepaths: list[str]) -> str:
    """Return the uncommitted diff for specific files, including untracked files.

    Never silently returns empty for an existing target: every path gets an
    explicit status section (diff / UNTRACKED / IGNORED / MISSING / no
    changes), so an empty result can only mean an empty input list.
    """
    in_repo = _in_repo()
    sections: list[str] = []
    for filepath in filepaths:
        if Path(filepath).is_dir():
            sections.append(_diff_for_dir(filepath))
        else:
            sections.append(_diff_for_file(filepath, in_repo))
    return "\n".join(sections)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run tools/git_diff.py <filepath> [<filepath> ...]")
        sys.exit(1)
    try:
        diff = get_uncommitted_diff(sys.argv[1:])
        print(diff, end="")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
