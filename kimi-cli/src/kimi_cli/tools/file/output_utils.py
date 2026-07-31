"""Pure, dependency-free helpers for taming Grep/Glob tool output.

These helpers are intentionally standalone: they never touch a Runtime, the
filesystem, or any third-party library, so they can be unit-tested in
isolation and reused by any tool that emits line-based output.

Three problems they solve (see ``plan.md`` for the full token-saving design):

1. **Folding** — head+tail fold of long line lists instead of blind tail
   dropping, so both the beginning and the end of a result set stay visible
   with an explicit ``… (N lines omitted) …`` marker.
2. **Dedup** — collapse long runs of identical lines into
   ``first-occurrence  (N repeats)``, matching rtk's dedup style so behavior
   is consistent whether the external rtk binary or this local fallback ran.
3. **rtk protocol cleanup** — strip rtk's wrapper header and fold markers out
   of the line stream (they are not real result lines) and surface their
   content as structured metadata instead.
"""

from __future__ import annotations

from typing import Any

import regex as re

#: Default fold budget applied by :func:`fold_lines`.
DEFAULT_MAX_LINES = 200

#: Lines shorter than this are never worth folding; also the default per-line
#: hard cap used by :func:`truncate_line`.
DEFAULT_MAX_LINE_LEN = 500

#: rtk protocol: ``42 matches in 3 files:`` at the top of the output.
_RTK_HEADER_RE = re.compile(r"^(\d+) matches in (\d+) files:$")

#: rtk protocol: per-file fold marker, e.g.
#: ``  +37 more in C:\path\file.py [see remaining: tail -n +26 <log>]``
_RTK_PER_FILE_FOLD_RE = re.compile(
    r"^\s*\+(\d+) more in (.+?) \[see remaining: (.*)\]$"
)

#: rtk protocol: files fold marker, e.g.
#: ``+133 more files [see remaining: tail -n +300 <log>]``
_RTK_FILES_FOLD_RE = re.compile(r"^\s*\+(\d+) more files \[see remaining: (.*)\]$")

#: Inside ``[see remaining: ...]`` rtk emits a tail hint like
#: ``tail -n +26 <log-path>``; parse it into (start_line, log_path).
_RTK_TAIL_HINT_RE = re.compile(r"^tail -n \+(\d+)\s+(\S+)$")

#: The dedup suffix appended by :func:`dedup_lines` (rtk-compatible style).
_REPEATS_SUFFIX_RE = re.compile(r"\s+\(\d+ repeats\)$")


def fold_lines(
    lines: list[str],
    max_lines: int = DEFAULT_MAX_LINES,
    *,
    head: int | None = None,
    tail: int | None = None,
) -> tuple[list[str], int]:
    """Fold *lines* to at most *max_lines* using a head+tail fold.

    When ``len(lines) <= max_lines`` the list is returned unchanged with an
    omitted count of ``0``.  Otherwise the first ``head`` lines and the last
    ``tail`` lines are kept (``head = max(1, max_lines // 2)``,
    ``tail = max_lines - head`` by default) with the marker line
    ``… (N lines omitted) …`` inserted between them.

    A *max_lines* of ``0`` or negative means "unlimited": the list is
    returned unchanged.  Empty and whitespace-only lines are preserved
    verbatim — callers decide whether to strip them.

    Returns:
        ``(folded_lines, omitted_count)``.
    """
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines, 0

    head_count = max(1, max_lines // 2) if head is None else max(0, head)
    tail_count = max_lines - head_count if tail is None else max(0, tail)
    if head_count + tail_count > max_lines:
        # Keep the caller-specified head, cap the tail to the budget.
        tail_count = max(0, max_lines - head_count)

    omitted = len(lines) - (head_count + tail_count)
    if omitted <= 0:
        return lines, 0

    marker = f"… ({omitted} lines omitted) …"
    folded = lines[:head_count] + [marker] + lines[len(lines) - tail_count:]
    return folded, omitted


def dedup_lines(
    lines: list[str],
    *,
    min_repeats: int = 3,
) -> tuple[list[str], int]:
    """Collapse runs of ≥ *min_repeats* identical lines into one + count.

    A run of ``k`` identical consecutive lines (``k >= min_repeats``) becomes
    a single line of the form ``first-occurrence  (k-1 repeats)`` — the same
    marker style rtk uses, so results look consistent whether the external
    rtk binary or this local fallback performed the dedup.

    Returns:
        ``(collapsed_lines, saved_count)`` where *saved_count* is the number
        of lines removed from the input.
    """
    if min_repeats < 2:
        min_repeats = 2
    if len(lines) < 2:
        return lines, 0

    out: list[str] = []
    saved = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        j = i + 1
        while j < n and lines[j] == line:
            j += 1
        run_len = j - i
        if run_len >= min_repeats:
            out.append(f"{line}  ({run_len - 1} repeats)")
            saved += run_len - 1
        else:
            out.extend(lines[i:j])
        i = j
    return out, saved


def truncate_line(line: str, max_len: int = DEFAULT_MAX_LINE_LEN) -> str:
    """Hard-cut *line* to at most *max_len* chars with a ``… [+K chars]`` marker.

    The marker reports how many characters were removed so the model can
    estimate how much information was dropped.  When *max_len* is too small
    to fit the marker the line is cut at *max_len* without a marker.

    Returns:
        The possibly-truncated line.
    """
    if len(line) <= max_len:
        return line

    marker = f"… [+{len(line) - max_len} chars]"
    if len(marker) >= max_len:
        return line[:max_len]
    return line[: max_len - len(marker)] + marker


def _parse_tail_hint(content: str) -> dict[str, Any]:
    """Parse the ``[see remaining: ...]`` payload into metadata fields."""
    m = _RTK_TAIL_HINT_RE.match(content.strip())
    if m:
        return {"start_line": int(m.group(1)), "log": m.group(2)}
    return {"start_line": None, "log": content.strip() or None}


def parse_rtk_rg_output(lines: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Remove rtk protocol lines and return cleaned lines + metadata.

    rtk wraps ``rg`` and injects protocol lines into the output stream:

    - a header ``42 matches in 3 files:`` (content mode) followed by a blank
      line;
    - per-file fold markers ``  +37 more in <path> [see remaining: ...]``;
    - a files fold marker ``+133 more files [see remaining: ...]``.

    These are not real result lines: they pollute the line stream, confuse
    path filters and pagination counters, and leak into the model context.
    This function strips them and returns the remaining lines untouched (the
    order and content of real lines is preserved, including separators and
    blank lines) plus structured metadata.

    Plain rg output (no rtk protocol lines) passes through untouched with an
    empty metadata dict.

    Args:
        lines: Split output lines (no trailing newline).

    Returns:
        ``(cleaned_lines, metadata)`` where *metadata* contains:
        ``total_matches``, ``total_files`` (from the header),
        ``folded_files`` — a list of ``{path, count, log, start_line}`` for
        each per-file fold marker, and ``skipped_files`` / ``skipped_log``
        for the files fold marker.
    """
    metadata: dict[str, Any] = {
        "total_matches": None,
        "total_files": None,
        "folded_files": [],
        "skipped_files": None,
        "skipped_log": None,
    }

    cleaned: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        header = _RTK_HEADER_RE.match(line)
        if header:
            metadata["total_matches"] = int(header.group(1))
            metadata["total_files"] = int(header.group(2))
            # Header is followed by a blank separator line — drop it too.
            if i + 1 < n and not lines[i + 1].strip():
                i += 1
            i += 1
            continue

        per_file = _RTK_PER_FILE_FOLD_RE.match(line)
        if per_file:
            hint = _parse_tail_hint(per_file.group(3))
            metadata["folded_files"].append(
                {
                    "path": per_file.group(2),
                    "count": int(per_file.group(1)),
                    "log": hint["log"],
                    "start_line": hint["start_line"],
                }
            )
            i += 1
            continue

        files_fold = _RTK_FILES_FOLD_RE.match(line)
        if files_fold:
            metadata["skipped_files"] = int(files_fold.group(1))
            hint = _parse_tail_hint(files_fold.group(2))
            metadata["skipped_log"] = hint["log"]
            i += 1
            continue

        cleaned.append(line)
        i += 1

    return cleaned, metadata
