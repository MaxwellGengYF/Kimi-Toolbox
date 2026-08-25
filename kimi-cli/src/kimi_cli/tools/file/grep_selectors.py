"""Selector grammar for richer grep paths (plans/23-grep-rich.md §4.2).

Pure, I/O-free functions implementing the line-range selector grammar shared
by the grep tool:

- ``file.py:50-100`` — 1-based inclusive range.
- ``file.py:50+10`` — 10 lines starting at 50 (50–59).
- ``file.py:301-`` — open-ended (301…EOF); bare ``file.py:301`` behaves the
  same.
- ``file.py:5-16,960-973`` — multi-range, disjoint windows.
- ``file.py:50..100`` — ``..`` alias for ``-``.
- ``:raw`` / ``:conflicts`` — display-mode suffixes with no search meaning
  (treated as unfiltered).

Performance rule: uses the third-party ``regex`` and ``orjson`` packages, not
the stdlib ``re`` / ``json`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import orjson
import regex as re

__all__ = [
    "LineRange",
    "GrepPathSpec",
    "parse_line_range_chunk",
    "parse_line_ranges",
    "is_line_in_ranges",
    "selector_line_ranges",
    "split_path_and_sel",
    "expand_path_entries",
    "merge_ranges_into",
]


@dataclass(frozen=True)
class LineRange:
    """A 1-based inclusive line range. ``end_line=None`` means open-ended."""

    start_line: int
    end_line: int | None = None


@dataclass
class GrepPathSpec:
    """One resolved path entry of a grep call.

    ``original`` is the raw entry as the caller wrote it; ``clean`` is the
    path with the selector peeled off (or the ``archive:member`` form /
    scratch path); ``ranges`` are the parsed line ranges, if any.
    """

    original: str
    clean: str
    literal_filesystem_match: bool = False
    ranges: list[LineRange] | None = None


# Full chunk grammar (case-insensitive): L?N | L?N-M | L?N..M | L?N- | L?N+K
_RANGE_CHUNK_RE = re.compile(
    r"^L?(\d+)(?:(\.\.|[-+])(?:L?(\d+))?)?$", re.IGNORECASE
)

# Loose shape check used by split_path_and_sel to decide whether a trailing
# ':' chunk should be peeled off. Semantic validation happens later via
# parse_line_ranges so invalid selectors surface actionable ToolErrors.
_SELECTOR_CHUNK_SHAPE_RE = re.compile(
    r"^(?:raw|conflicts|L?\d+(?:(?:\.\.|[-+])L?\d*)?)$", re.IGNORECASE
)

# scheme://authority with no path component (e.g. ssh://host:2222) — the
# trailing ":2222" is a port, never a selector.
_SCHEME_AUTHORITY_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/]*$")


def parse_line_range_chunk(chunk: str) -> LineRange | None:
    """Parse one selector chunk into a LineRange.

    Returns ``None`` when the chunk does not match the selector grammar at
    all. Raises ``ValueError`` for grammatically valid but semantically
    invalid chunks (0 start, inverted range, zero/negative count).
    """
    m = _RANGE_CHUNK_RE.match(chunk.strip())
    if m is None:
        return None
    start = int(m.group(1))
    op = m.group(2)
    end_str = m.group(3)

    if start < 1:
        raise ValueError(
            f"Line selector {start} is invalid; lines are 1-indexed. Use :1."
        )

    if op in ("-", ".."):
        end = int(end_str) if end_str is not None else None
        if end is not None and end < start:
            raise ValueError(
                f"Invalid range {start}-{end}: end must be >= start."
            )
        return LineRange(start, end)

    if op == "+":
        count = int(end_str) if end_str is not None else None
        if count is None:
            # "N+" behaves like the open-ended "N-".
            return LineRange(start, None)
        if count < 1:
            raise ValueError(
                f"Invalid range {start}+{count}: count must be >= 1."
            )
        return LineRange(start, start + count - 1)

    # Bare "N" — open-ended from N (same as "N-").
    return LineRange(start, None)


def parse_line_ranges(sel: str) -> list[LineRange] | None:
    """Parse a comma-separated selector into a sorted, merged range list.

    Returns ``None`` when no chunk parses as a selector. Raises ``ValueError``
    when a grammatically valid chunk is semantically invalid. Overlapping and
    adjacent ranges are merged; an open-ended range absorbs all later ranges.
    """
    ranges: list[LineRange] = []
    saw_any = False
    for chunk in sel.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed = parse_line_range_chunk(chunk)
        if parsed is None:
            continue
        saw_any = True
        ranges.append(parsed)
    if not saw_any:
        return None

    ranges.sort(key=lambda r: r.start_line)
    merged: list[LineRange] = []
    for r in ranges:
        last = merged[-1] if merged else None
        if last is not None and last.end_line is None:
            # An open-ended range absorbs everything after it.
            continue
        if last is not None:
            assert last.end_line is not None
            if r.start_line <= last.end_line + 1:
                new_end = (
                    None
                    if r.end_line is None
                    else max(last.end_line, r.end_line)
                )
                merged[-1] = LineRange(last.start_line, new_end)
                continue
        merged.append(r)
    return merged


def is_line_in_ranges(line_number: int, ranges: list[LineRange] | None) -> bool:
    """True when *line_number* is inside any of *ranges* (None = unfiltered)."""
    if ranges is None:
        return True
    for r in ranges:
        if line_number < r.start_line:
            continue
        if r.end_line is None or line_number <= r.end_line:
            return True
    return False


def selector_line_ranges(sel: str | None) -> list[LineRange] | None:
    """Resolve the effective line ranges of a peeled selector string.

    ``raw`` / ``conflicts`` chunks are display modes with no search meaning
    and are skipped. ':' separates chunks (compound selectors like
    ``1-50:raw``); commas stay INSIDE a range-list chunk (``5-16,960-973``).
    Returns the first parseable range list, else ``None`` (unfiltered).
    Raises ``ValueError`` on semantically invalid chunks.
    """
    if not sel:
        return None
    for chunk in sel.split(":"):
        chunk = chunk.strip()
        if not chunk or chunk.lower() in ("raw", "conflicts"):
            continue
        parsed = parse_line_ranges(chunk)
        if parsed is not None:
            return parsed
    return None


def _literal_exists(raw: str) -> bool:
    """True when a real filesystem entry with exactly this name exists."""
    import os

    try:
        return os.path.lexists(raw)
    except (OSError, ValueError):
        return False


def _is_selector_shape(tail: str) -> bool:
    """Loose grammar check for a peelable trailing chunk (no semantics)."""
    tail = tail.strip()
    if not tail:
        return False
    for chunk in tail.split(","):
        chunk = chunk.strip()
        if not chunk or not _SELECTOR_CHUNK_SHAPE_RE.match(chunk):
            return False
    return True


def split_path_and_sel(raw_path: str) -> tuple[str, str | None]:
    """Split ``path[:selector]`` into (path, selector-or-None).

    Guards:
    - Windows drive letters: never peel a tail that would leave a bare drive.
    - ``scheme://authority`` with no path (e.g. ``ssh://host:2222``): the
      trailing colon chunk is a port, not a selector.
    - Literal filesystem match: a real file named ``test:1-2`` outranks the
      selector interpretation (probe the RAW string first).

    At most two chunks are peeled (one range list + one ``raw``/``conflicts``
    display mode); the peeled chunks are rejoined with ':' so callers can pass
    the selector straight to ``selector_line_ranges``.
    """
    if not raw_path or ":" not in raw_path:
        return raw_path, None

    # A real filesystem entry with the raw name wins (issue #4618 parity).
    if _literal_exists(raw_path):
        return raw_path, None

    # ssh://host:2222 (scheme + authority, no path) — the colon chunk is a port.
    if _SCHEME_AUTHORITY_RE.match(raw_path):
        return raw_path, None

    import os

    rest = raw_path
    peeled: list[str] = []
    for _ in range(2):
        idx = rest.rfind(":")
        if idx <= 0:
            break
        head, tail = rest[:idx], rest[idx + 1 :]
        if not _is_selector_shape(tail):
            break
        # Windows drive-letter guard: "C:" must never be left bare.
        drive, after = os.path.splitdrive(head)
        if drive and not after:
            break
        peeled.insert(0, tail)
        rest = head

    if peeled:
        return rest, ":".join(peeled)
    return raw_path, None


def _maybe_json_array(s: str) -> list[str] | None:
    """Parse *s* as a JSON array of strings, or return None."""
    if not s.startswith("["):
        return None
    try:
        parsed = orjson.loads(s)
    except (orjson.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(item, str) for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def expand_path_entries(raw: str | list[str]) -> list[str]:
    """Normalize the ``path`` input to a flat, deduplicated entry list.

    Accepts a list, a JSON-encoded array string (``'["a.py","b.py"]'``), or a
    semicolon-delimited single string (``"src; tests"``). Commas never split
    entries — a comma separates ranges inside a selector.
    """
    if isinstance(raw, list):
        entries = [e.strip() for e in raw if isinstance(e, str) and e.strip()]
    else:
        s = raw.strip()
        if not s:
            return []
        parsed = _maybe_json_array(s)
        if parsed is not None:
            entries = [e.strip() for e in parsed if e.strip()]
        else:
            entries = [p.strip() for p in s.split(";") if p.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for e in entries:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def merge_ranges_into(
    ranges_by_path: dict[str, list[LineRange]],
    abs_key: str,
    ranges: list[LineRange] | None,
) -> None:
    """Append *ranges* to the per-path map (port of mergeRangesInto).

    Duplicates/overlaps are harmless: ``is_line_in_ranges`` scans linearly.
    """
    if not ranges:
        return
    ranges_by_path.setdefault(abs_key, []).extend(ranges)
