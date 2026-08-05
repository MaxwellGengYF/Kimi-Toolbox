"""kimix_native.workspace -- workspace kernels: snapshot, diff, changed-files.

Native implementations live in ``runtime_py.workspace`` (compiled kernels, GIL
released). The pure-Python functions below mirror the native behavior exactly so
``use_native('WORKSPACE') is False`` yields identical behavior.
"""

from __future__ import annotations

import difflib
import os

from . import _native, use_native


# ---------------------------------------------------------------------------
# _compat -- exact mirrors of the native reference implementation
# ---------------------------------------------------------------------------


def _compat_snapshot(
    root: str,
    ignore_dirs: tuple[str, ...] = (),
    max_file_bytes: int = 8 * 1024 * 1024,
) -> dict[str, bytes]:
    """Pure-Python directory snapshot."""
    root_path = os.path.abspath(root)
    ignore_set = set(ignore_dirs)
    snapshot: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in ignore_set]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.islink(full):
                continue
            try:
                size = os.path.getsize(full)
                if size > max_file_bytes:
                    continue
                rel = os.path.relpath(full, root_path).replace("\\", "/")
                with open(full, "rb") as f:
                    snapshot[rel] = f.read()
            except OSError:
                continue
    return snapshot


def _lower_extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _is_text(content: bytes, path: str, text_extensions: set[str] | None) -> bool:
    if text_extensions is not None and _lower_extension(path) in text_extensions:
        return True
    return b"\x00" not in content


def _compat_diff_snapshots(
    before: dict[str, bytes],
    after: dict[str, bytes],
    *,
    text_extensions: set[str] | None = None,
    context_lines: int = 3,
) -> bytes:
    """Pure-Python combined unified diff of two snapshots."""
    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        has_old = rel in before
        has_new = rel in after
        if has_old and has_new and before[rel] == after[rel]:
            continue
        old_bytes = before.get(rel, b"")
        new_bytes = after.get(rel, b"")
        content_for_check = new_bytes if rel in after else old_bytes
        if _is_text(content_for_check, rel, text_extensions):
            old_lines = (
                old_bytes.decode("utf-8", errors="replace")
                .splitlines(keepends=True)
            )
            new_lines = (
                new_bytes.decode("utf-8", errors="replace")
                .splitlines(keepends=True)
            )
            if old_lines and not old_lines[-1].endswith("\n"):
                old_lines[-1] += "\n"
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            chunks.extend(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    n=context_lines,
                )
            )
        else:
            chunks.append(f"--- a/{rel}\n")
            chunks.append(f"+++ b/{rel}\n")
            if not has_old and has_new:
                chunks.append("@@ -0,0 +1 @@\n+Binary file\n")
            elif has_old and not has_new:
                chunks.append("@@ -1 +0,0 @@\n-Binary file\n")
            else:
                chunks.append("@@ -1 +1 @@\n-Binary file\n+Binary file\n")
    return "".join(chunks).encode("utf-8")


def _compat_changed_files(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> list[tuple[str, str]]:
    """Pure-Python changed-file list."""
    result: list[tuple[str, str]] = []
    for rel in sorted(set(before) | set(after)):
        has_old = rel in before
        has_new = rel in after
        if has_old and has_new and before[rel] == after[rel]:
            continue
        result.append((rel, rel))
    return result


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


def snapshot(
    root: str,
    ignore_dirs: tuple[str, ...] = (),
    max_file_bytes: int = 8 * 1024 * 1024,
) -> dict[str, bytes]:
    if use_native("WORKSPACE") and _native is not None:
        return _native.workspace.snapshot(root, list(ignore_dirs), max_file_bytes)
    return _compat_snapshot(root, ignore_dirs, max_file_bytes)


def diff_snapshots(
    before: dict[str, bytes],
    after: dict[str, bytes],
    *,
    text_extensions: set[str] | None = None,
    context_lines: int = 3,
) -> bytes:
    if use_native("WORKSPACE") and _native is not None:
        native_exts = set(text_extensions) if text_extensions is not None else None
        return _native.workspace.diff_snapshots(
            before, after, native_exts, context_lines
        )
    return _compat_diff_snapshots(
        before, after, text_extensions=text_extensions, context_lines=context_lines
    )


def changed_files(
    before: dict[str, bytes],
    after: dict[str, bytes],
) -> list[tuple[str, str]]:
    if use_native("WORKSPACE") and _native is not None:
        return _native.workspace.changed_files(before, after)
    return _compat_changed_files(before, after)
