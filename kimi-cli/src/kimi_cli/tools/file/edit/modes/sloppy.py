"""Sloppy-mode executor for the multi-mode edit tool."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from kaos.path import KaosPath
from kosong.tooling import ToolError, ToolReturnValue
from rapidfuzz import fuzz

from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.edit.params import EditMode, EditParams
from kimi_cli.tools.file.edit_safety import create_edit_parse_guard
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.path import is_within_directory, kaos_path_from_tool_input

from ..base import BaseEditTool


@dataclass
class SloppyInline:
    """One inline selection `⟪old│new⟫` inside a line."""

    old: str
    new: str


@dataclass
class SloppyOp:
    """One sloppy operation."""

    path: str
    all_match: bool = False
    match_lines: list[str] = field(default_factory=list)
    rewrite_lines: list[str] | None = None
    inline_lines: list[tuple[str, list[SloppyInline]]] = field(default_factory=list)


@dataclass
class _SloppyPrepared:
    op: SloppyOp
    p: KaosPath
    display_path: str
    content: str


# §path or §*path or §* or § (bare)
_SECTION_RE = re.compile(r"^\s*§(\*?)(.*?)\s*$")
# Inline selection using fullwidth pipe │ U+2502 and ⟪ ⟫
_INLINE_RE = re.compile(r"⟪([^│⟫\n]+)│([^⟫\n]*)⟫")


def _split_sections(input_text: str) -> list[list[str]]:
    """Split sloppy input into raw sections starting with §."""
    lines = input_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("§"):
            if current is not None:
                sections.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        sections.append(current)
    return sections


def _parse_op(raw_lines: list[str], default_path: str | None = None) -> SloppyOp:
    """Parse a raw section into a SloppyOp."""
    header_line = raw_lines[0].strip()
    match = _SECTION_RE.match(header_line)
    assert match is not None
    all_match = match.group(1) == "*"
    path = match.group(2).strip()
    if not path:
        path = default_path or ""

    body = [line.rstrip("\r\n") for line in raw_lines[1:]]

    # Detect block rewrite separator "»" on its own line.
    sep_idx = -1
    for i, line in enumerate(body):
        if line.strip() == "»":
            sep_idx = i
            break

    if sep_idx != -1:
        match_lines = body[:sep_idx]
        rewrite_lines = body[sep_idx + 1 :]
        return SloppyOp(
            path=path,
            all_match=all_match,
            match_lines=match_lines,
            rewrite_lines=rewrite_lines,
        )

    # Inline operation: parse selections in each line.
    inline_lines: list[tuple[str, list[SloppyInline]]] = []
    for line in body:
        if line.strip() == "":
            continue
        selections: list[SloppyInline] = []
        remaining = line
        while True:
            m = _INLINE_RE.search(remaining)
            if not m:
                break
            selections.append(SloppyInline(old=m.group(1), new=m.group(2)))
            remaining = remaining[: m.start()] + remaining[m.end() :]
        if selections:
            inline_lines.append((line, selections))
        else:
            inline_lines.append((line, []))

    return SloppyOp(path=path, all_match=all_match, inline_lines=inline_lines)


def parse_sloppy_input(input_text: str) -> list[SloppyOp]:
    """Parse sloppy-mode input into a list of operations."""
    sections = _split_sections(input_text)
    if not sections:
        raise ValueError("No sloppy operations found. Input must start with `§path`.")

    ops: list[SloppyOp] = []
    current_path: str | None = None
    for section in sections:
        op = _parse_op(section, current_path)
        if not op.path:
            raise ValueError("Bare `§` requires a previous section with a path.")
        current_path = op.path
        ops.append(op)
    return ops


def _find_exact_block(content: str, block_lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) character indices where *block_lines* occur in *content*."""
    if not block_lines:
        return None
    needle = "\n".join(block_lines)
    start = content.find(needle)
    if start == -1:
        return None
    return start, start + len(needle)


def _find_fuzzy_block(content: str, block_lines: list[str], threshold: float = 0.75) -> tuple[int, int] | None:
    """Locate *block_lines* in *content* using a line-window fuzzy match."""
    if not block_lines:
        return None
    content_lines = content.splitlines()
    window_size = len(block_lines)
    if window_size > len(content_lines):
        return None
    needle = "\n".join(block_lines)
    best_score = 0.0
    best_idx = 0
    for i in range(len(content_lines) - window_size + 1):
        window = "\n".join(content_lines[i : i + window_size])
        score = fuzz.ratio(window, needle) / 100.0
        if score > best_score:
            best_score = score
            best_idx = i
    if best_score < threshold:
        return None
    # Convert line index to character range.
    start_char = sum(len(l) + 1 for l in content_lines[:best_idx])
    end_char = start_char + sum(len(l) + 1 for l in content_lines[best_idx : best_idx + window_size]) - 1
    return start_char, end_char


def _apply_block_op(content: str, op: SloppyOp) -> str:
    """Apply a MATCH » REWRITE block operation."""
    assert op.rewrite_lines is not None
    match_lines = op.match_lines
    if not match_lines:
        raise ValueError("MATCH block is empty.")

    replacement = "\n".join(op.rewrite_lines)
    found = _find_exact_block(content, match_lines)
    if found is None:
        found = _find_fuzzy_block(content, match_lines)
    if found is None:
        raise ValueError(f"Could not locate MATCH block:\n" + "\n".join(match_lines))

    start, end = found

    # For a pure deletion, swallow one trailing newline to avoid blank lines.
    if not replacement and content[end:end + 1] == "\n":
        end += 1

    if op.all_match:
        # Replace every non-overlapping occurrence.
        result = ""
        prev = 0
        needle = "\n".join(match_lines)
        while True:
            pos = content.find(needle, prev)
            if pos == -1:
                break
            end_pos = pos + len(needle)
            if not replacement and content[end_pos:end_pos + 1] == "\n":
                end_pos += 1
            result += content[prev:pos] + replacement
            prev = end_pos
        result += content[prev:]
        return result

    return content[:start] + replacement + content[end:]


def _apply_inline_op(content: str, op: SloppyOp) -> str:
    """Apply inline selections to the content."""
    if not op.inline_lines:
        return content

    # Collect all selections in order.
    selections: list[SloppyInline] = []
    for _line, sels in op.inline_lines:
        selections.extend(sels)

    if not selections:
        return content

    if op.all_match:
        result = content
        for sel in selections:
            result = result.replace(sel.old, sel.new)
        return result

    for sel in selections:
        if sel.old not in content:
            raise ValueError(f"Could not locate inline selection: {sel.old!r}")
        content = content.replace(sel.old, sel.new, 1)
    return content


class SloppyModeExecutor:
    """Executor for sloppy sparse edits."""

    mode: EditMode = "sloppy"
    description: str = "Edit files using sparse fragment rewrites."

    async def execute(self, tool: BaseEditTool, params: EditParams) -> ToolReturnValue:
        if not params.input:
            return ToolError(message="sloppy mode requires an input payload.", brief="Missing input")

        try:
            ops = parse_sloppy_input(params.input)
        except ValueError as e:
            return ToolError(message=f"Failed to parse sloppy input: {e}", brief="Parse error")

        # Resolve every path up front (atomic preflight).
        prepared: list[_SloppyPrepared] = []
        seen_paths: set[str] = set()
        for op in ops:
            p = kaos_path_from_tool_input(op.path, tool._work_dir)
            display_path = str(p).replace("\\", "/")
            err, _ = await tool._validate_path(p, op.path)
            if err:
                return ToolError(
                    message=f"Invalid path `{display_path}`: {err.message}",
                    brief="Invalid path",
                )
            canonical = str(p.canonical()).replace("\\", "/")
            if canonical in seen_paths and not op.all_match:
                pass
            seen_paths.add(canonical)
            try:
                content = await tool._read_text(p)
            except FileNotFoundError:
                return ToolError(
                    message=f"`{display_path}` does not exist.",
                    brief="File not found",
                )
            prepared.append(_SloppyPrepared(op=op, p=p, display_path=display_path, content=content))

        # Apply all ops atomically: parse + locate first, then compute new contents.
        new_contents: dict[int, tuple[_SloppyPrepared, str]] = {}
        for prep in prepared:
            try:
                if prep.op.rewrite_lines is not None:
                    new_content = _apply_block_op(prep.content, prep.op)
                else:
                    new_content = _apply_inline_op(prep.content, prep.op)
            except ValueError as e:
                return ToolError(
                    message=(
                        f"Sloppy edit failed for `{prep.display_path}`: {e}\n\n"
                        "Corrected payload (re-issue verbatim):\n"
                        f"{params.input}"
                    ),
                    brief="Sloppy edit failed",
                )
            new_contents[id(prep)] = (prep, new_content)

        # Write all files after every op succeeded.
        results: list[str] = []
        for prep, new_content in new_contents.values():
            if new_content == prep.content:
                return ToolError(
                    message=(
                        f"Sloppy edit for `{prep.display_path}` produced no change: "
                        "the anchor already reads as the desired text."
                    ),
                    brief="No change",
                )

            diff_blocks = await build_diff_blocks(prep.display_path, prep.content, new_content)
            action = FileActions.EDIT if tool._is_within_workspace(prep.p) else FileActions.EDIT_OUTSIDE
            approval_result = await tool._approval.request(
                "edit",
                action,
                f"Edit file `{prep.display_path}`"
                + (f" — {params.justification}" if params.justification else ""),
                display=diff_blocks,
            )
            if not approval_result:
                return approval_result.rejection_error()

            if params.allow_conflicts:
                pass
            else:
                markers = ["<<<<<<<", "=======", ">>>>>>>"]
                for i, line in enumerate(prep.content.replace("\r\n", "\n").splitlines(), 1):
                    stripped = line.strip()
                    if stripped in markers or stripped.startswith("<<<<<<< ") or stripped.startswith(">>>>>>> "):
                        return ToolError(
                            message=f"Conflict markers detected in `{prep.display_path}` at line {i}; refusing to edit.",
                            brief="Conflict markers detected",
                        )

            if not tool._session.file_mtime.mark_dirty(str(prep.p)):
                return ToolError(
                    message=(
                        f"`{prep.display_path}` changed externally or was written after the last read. "
                        "Re-read the file and re-issue the edit."
                    ),
                    brief="Stale file",
                )

            await tool._write_text(prep.p, new_content)
            results.append(f"Edited `{prep.display_path}`.")

            guard = create_edit_parse_guard(
                tool._session,
                variant="sloppy",
                arg=params.model_dump(),
            )
            await guard.observe_applied(str(prep.p), prep.content, new_content)
            notes = await guard.finish()
            if notes:
                results[-1] += "\n" + "\n".join(notes)

        return ToolReturnValue(
            is_error=False,
            output="",
            message="\n".join(results),
            display=[],
        )
