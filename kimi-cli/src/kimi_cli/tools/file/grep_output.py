"""Grouped/context output rendering for richer grep (plans/23-grep-rich.md §4.5).

Ports oh-my-pi's ``match-line-format.ts`` (``*N|text`` / `` N|text`` markers)
and the model-facing part of ``grouped-file-output.ts`` (``#``-prefixed file
headers with blank-line separators). Everything is plain text; no TUI
rendering.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "format_match_line",
    "group_lines_by_file",
    "format_grouped_output",
    "group_line_indices_by_blank",
    "should_group",
]


def format_match_line(line_number: int, line: str, is_match: bool) -> str:
    """Render one content line with oh-my-pi's match/context marker.

    Matches get ``*N|text``; context lines get `` N|text`` (a single leading
    space keeps columns aligned). Line numbers are never padded.
    """
    marker = "*" if is_match else " "
    return f"{marker}{line_number}|{line}"


def group_lines_by_file(
    lines: list[str],
    parse: Callable[[str], tuple[str, int, str, bool] | None],
) -> list[tuple[str, list[tuple[int, str, bool]]]]:
    """Group content-mode lines by file in encounter order.

    *parse* is a content-line parser (e.g. ``parse_content_line``) returning
    ``(path, line_no, text, is_match)`` or ``None`` for non-content lines.
    ``--`` separators and gap markers are attached to the group they follow;
    leading non-content lines are dropped.

    Returns ``[(display_path, [(line_no, text, is_match), ...]), ...]`` with
    separators encoded as ``(0, "--", False)`` entries in the body.
    """
    groups: list[tuple[str, list[tuple[int, str, bool]]]] = []
    current_path: str | None = None
    for line in lines:
        parsed = parse(line)
        if parsed is None:
            if current_path is not None and line.strip():
                # Keep "--" separators and gap markers inside the current group.
                groups[-1][1].append((0, line, False))
            continue
        path, line_no, text, is_match = parsed
        if path != current_path:
            current_path = path
            groups.append((path, []))
        groups[-1][1].append((line_no, text, is_match))
    return groups


def format_grouped_output(
    groups: list[tuple[str, list[tuple[int, str, bool]]]],
) -> list[str]:
    """Render grouped output for the model.

    For each file group: a blank separator line between files (none before the
    first), a ``# <display_path>`` header, then the body lines formatted via
    ``format_match_line``. Separator entries (line_no 0) are emitted verbatim.
    """
    out: list[str] = []
    for i, (path, body) in enumerate(groups):
        if i > 0:
            out.append("")
        out.append(f"# {path}")
        for line_no, text, is_match in body:
            if line_no == 0:
                out.append(text)
            else:
                out.append(format_match_line(line_no, text, is_match))
    return out


def group_line_indices_by_blank(raw_lines: list[str]) -> list[list[int]]:
    """Split line indices into groups delimited by blank lines.

    Port of grouped-file-output.ts ``groupLineIndicesByBlank``: returns lists
    of 0-based indices; blank lines themselves are excluded from every group.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for idx, line in enumerate(raw_lines):
        if not line.strip():
            if current:
                groups.append(current)
                current = []
            continue
        current.append(idx)
    if current:
        groups.append(current)
    return groups


def should_group(params: Any, *, has_rich_entries: bool) -> bool:
    """Decide whether grouped rendering applies for this call.

    ``params.grouped`` wins when set (``True``/``False``). When ``None``
    (auto), grouped output activates only for selector/archive entries so
    plain searches keep the legacy byte-identical output.
    """
    grouped = getattr(params, "grouped", None)
    if grouped is not None:
        return bool(grouped)
    return has_rich_entries
