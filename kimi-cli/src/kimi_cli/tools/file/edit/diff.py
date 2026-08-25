"""Unified diff hunk parser and applier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz import fuzz

HunkLineKind = Literal["context", "add", "delete"]


@dataclass
class HunkLine:
    kind: HunkLineKind
    text: str


@dataclass
class DiffHunk:
    start_line: int | None = None
    change_context: str | None = None
    lines: list[HunkLine] = field(default_factory=list)


class ApplyPatchError(Exception):
    """Raised when a patch cannot be applied."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@(.*)$")


def normalize_diff(diff: str) -> str:
    """Strip metadata and normalize a unified diff string."""
    text = diff.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip("\n")
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("diff --git") or line.startswith("index "):
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("*** End of File"):
            continue
        lines.append(line)
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines)


def normalize_create_content(diff: str) -> str:
    """Strip leading '+' prefixes from a create-mode diff body."""
    text = diff.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip("\n")
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def parse_diff_hunks(diff: str) -> list[DiffHunk]:
    """Parse a unified diff string into a list of hunks."""
    text = normalize_diff(diff)
    if not text.strip():
        return []

    hunks: list[DiffHunk] = []
    current: DiffHunk | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            if any(line.kind in ("add", "delete") for line in current.lines):
                hunks.append(current)
            current = None

    for raw_line in text.split("\n"):
        line = raw_line
        stripped = line.strip()

        if stripped.startswith("@@"):
            flush()
            header_match = _HEADER_RE.match(stripped)
            if header_match:
                old_start = int(header_match.group(1))
                new_start = int(header_match.group(2))
                context = header_match.group(3).strip() or None
                current = DiffHunk(start_line=old_start, change_context=context, lines=[])
                if new_start and not old_start:
                    current.start_line = new_start
            else:
                # Bare @@ or anchor-only header: capture text after @@ as context.
                after = stripped[2:].strip()
                if after.startswith("@@"):
                    after = after[2:].strip()
                start_line: int | None = None
                context = None
                if after:
                    # Try to extract a leading line number.
                    m = re.match(r"^(\d+)\b", after)
                    if m:
                        start_line = int(m.group(1))
                        rest = after[m.end() :].strip()
                        if rest:
                            context = rest
                    else:
                        context = after
                current = DiffHunk(start_line=start_line, change_context=context, lines=[])
            continue

        if current is None:
            if stripped == "":
                continue
            # Content outside a hunk header: skip blank lines, reject obvious multi-file markers.
            if stripped.startswith("---") or stripped.startswith("+++"):
                continue
            raise ApplyPatchError(f"Unexpected diff content outside a hunk: {line[:80]!r}")

        if line.startswith("+"):
            current.lines.append(HunkLine(kind="add", text=line[1:]))
        elif line.startswith("-"):
            current.lines.append(HunkLine(kind="delete", text=line[1:]))
        elif line.startswith(" "):
            current.lines.append(HunkLine(kind="context", text=line[1:]))
        elif stripped == "":
            # Blank lines inside a hunk are treated as context with an empty body.
            current.lines.append(HunkLine(kind="context", text=""))
        else:
            # Malformed line inside hunk; treat as end of hunk to be lenient.
            flush()

    flush()

    if not hunks:
        # If there are no proper hunk markers but lines are prefixed with +/-,
        # treat the whole body as a single hunk.
        body: list[HunkLine] = []
        for raw_line in text.split("\n"):
            line = raw_line
            if line.startswith("+"):
                body.append(HunkLine(kind="add", text=line[1:]))
            elif line.startswith("-"):
                body.append(HunkLine(kind="delete", text=line[1:]))
            elif line.startswith(" "):
                body.append(HunkLine(kind="context", text=line[1:]))
            else:
                if body:
                    break
        if body and any(line.kind in ("add", "delete") for line in body):
            hunks.append(DiffHunk(start_line=None, change_context=None, lines=body))

    return hunks


def _hunk_pattern(hunk: DiffHunk) -> list[str]:
    """Return the lines from the hunk that must exist in the original file."""
    return [line.text for line in hunk.lines if line.kind != "add"]


def _count_leading_whitespace(line: str) -> tuple[int, int]:
    """Return (spaces, tabs) counts of leading whitespace."""
    spaces = tabs = 0
    for ch in line:
        if ch == " ":
            spaces += 1
        elif ch == "\t":
            tabs += 1
        else:
            break
    return spaces, tabs


def _infer_indent_adjustment(pattern_lines: list[str], actual_lines: list[str]) -> tuple[int, str]:
    """Infer the leading-whitespace delta and indent character."""
    deltas: list[int] = []
    indent_chars: list[str] = []
    for p, a in zip(pattern_lines, actual_lines):
        if not p.strip() or not a.strip():
            continue
        p_spaces, p_tabs = _count_leading_whitespace(p)
        a_spaces, a_tabs = _count_leading_whitespace(a)
        if p_tabs and not a_tabs and a_spaces:
            # Pattern uses tabs, actual uses spaces: infer tab width.
            if p_tabs and a_spaces % p_tabs == 0:
                deltas.append(a_spaces - p_tabs * (a_spaces // p_tabs))
            else:
                deltas.append(a_spaces - p_spaces)
            indent_chars.append(" ")
        elif p_spaces and not p_spaces and a_tabs:
            deltas.append(a_tabs - p_spaces)
            indent_chars.append("\t")
        else:
            deltas.append(a_spaces - p_spaces)
            indent_chars.append(" ")
    if not deltas:
        return 0, " "
    # Use the most common delta.
    delta = max(set(deltas), key=deltas.count)
    indent_char = max(set(indent_chars), key=indent_chars.count) if indent_chars else " "
    return delta, indent_char


def _apply_indent(line: str, delta: int, indent_char: str) -> str:
    """Apply a leading-whitespace delta to a line."""
    if not line.strip():
        return line
    spaces, tabs = _count_leading_whitespace(line)
    current = spaces + tabs
    new_indent = max(0, current + delta)
    return indent_char * new_indent + line.lstrip()


def _adjust_added_lines(hunk: DiffHunk, pattern_lines: list[str], actual_lines: list[str]) -> list[HunkLine]:
    """Adjust indentation of added lines to match the target file."""
    delta, indent_char = _infer_indent_adjustment(pattern_lines, actual_lines)
    if delta == 0:
        return hunk.lines
    adjusted: list[HunkLine] = []
    for line in hunk.lines:
        if line.kind == "add":
            adjusted.append(HunkLine(kind="add", text=_apply_indent(line.text, delta, indent_char)))
        else:
            adjusted.append(line)
    return adjusted


def _find_exact_matches(original_lines: list[str], pattern: list[str]) -> list[int]:
    """Return all start indices where *pattern* exactly matches *original_lines*."""
    if not pattern:
        return [0]
    matches: list[int] = []
    for i in range(len(original_lines) - len(pattern) + 1):
        if original_lines[i : i + len(pattern)] == pattern:
            matches.append(i)
    return matches


def _find_fuzzy_match(original_lines: list[str], pattern: list[str], threshold: float) -> int | None:
    """Find a unique fuzzy match for *pattern* in *original_lines*."""
    if not pattern:
        return 0
    window_size = len(pattern)
    candidates: list[tuple[int, float]] = []
    for i in range(len(original_lines) - window_size + 1):
        window = original_lines[i : i + window_size]
        score = fuzz.ratio("\n".join(window), "\n".join(pattern)) / 100.0
        if score >= threshold:
            candidates.append((i, score))
    if not candidates:
        return None
    # If the best score is clearly dominant, use it.
    candidates.sort(key=lambda x: x[1], reverse=True)
    if len(candidates) == 1:
        return candidates[0][0]
    if candidates[0][1] - candidates[1][1] >= 0.05:
        return candidates[0][0]
    return None


def apply_diff_hunks(
    content: str,
    hunks: list[DiffHunk],
    *,
    allow_fuzzy: bool = True,
    threshold: float = 0.75,
) -> tuple[str, int | None]:
    """Apply parsed diff hunks to *content* and return the new text plus first changed line.

    The first changed line is 1-based in the resulting content, or None if unchanged.
    """
    original_lines = content.split("\n")
    # Preserve trailing newline semantics: if original ended with newline, we keep that convention.
    ended_with_newline = content.endswith("\n")

    # Work bottom-up so earlier line numbers stay valid.
    sorted_hunks = sorted(hunks, key=lambda h: (h.start_line or 0), reverse=True)
    working_lines = list(original_lines)
    first_changed_line: int | None = None

    for hunk in sorted_hunks:
        pattern = _hunk_pattern(hunk)

        # Try exact match first.
        matches = _find_exact_matches(working_lines, pattern)
        if len(matches) > 1:
            raise ApplyPatchError(
                f"Found multiple matches for hunk at line {hunk.start_line or '?'}; "
                "add more context lines to disambiguate."
            )

        match_index: int | None = matches[0] if matches else None
        if match_index is None and allow_fuzzy:
            match_index = _find_fuzzy_match(working_lines, pattern, threshold)

        if match_index is None:
            context = hunk.change_context or f"line {hunk.start_line or '?'}"
            raise ApplyPatchError(f"No match found for hunk anchored at {context}.")

        # Apply indentation adjustment for added lines.
        actual_lines = working_lines[match_index : match_index + len(pattern)]
        adjusted_hunk = _adjust_added_lines(hunk, pattern, actual_lines)

        # Build replacement lines for the matched block.
        replacement: list[str] = []
        actual_idx = 0
        changed = False
        for line in adjusted_hunk:
            if line.kind == "context":
                replacement.append(actual_lines[actual_idx])
                actual_idx += 1
            elif line.kind == "delete":
                actual_idx += 1
                changed = True
            elif line.kind == "add":
                replacement.append(line.text)
                changed = True

        if changed:
            line_num = match_index + 1
            if first_changed_line is None or line_num < first_changed_line:
                first_changed_line = line_num

        working_lines = working_lines[:match_index] + replacement + working_lines[match_index + len(pattern) :]

    result = "\n".join(working_lines)
    if ended_with_newline and not result.endswith("\n"):
        result += "\n"
    return result, first_changed_line
