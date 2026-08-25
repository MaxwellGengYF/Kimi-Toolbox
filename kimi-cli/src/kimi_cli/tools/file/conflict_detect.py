"""Conflict-marker detection and resolution primitives (plan 24).

Port of oh-my-pi's ``packages/coding-agent/src/tools/conflict-detect.ts``:
strict column-0 git merge-conflict marker scanning, a per-session
``ConflictHistory`` registry with stable ids, ``conflict://`` URI parsing,
``@ours``/``@theirs``/``@base``/``@both`` content-token expansion, and
region splicing with boundary-echo trimming.

The module is pure (no filesystem access except the capped whole-file scan
helper), so read/write/edit can all share it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import regex as re

if TYPE_CHECKING:
    from kimi_cli.session import Session

__all__ = [
    "OURS_PREFIX",
    "BASE_PREFIX",
    "SEPARATOR",
    "THEIRS_PREFIX",
    "SCAN_FILE_DEFAULT_MAX_BYTES",
    "PREVIEW_SIDE_LINES",
    "EchoTrimLimit",
    "ConflictBlock",
    "ConflictEntry",
    "ConflictError",
    "ConflictSplice",
    "ParsedConflictUri",
    "ConflictScanResult",
    "match_marker",
    "is_separator",
    "scan_conflict_lines",
    "scan_file_for_conflicts",
    "find_dangling_openers",
    "ConflictHistory",
    "get_conflict_history",
    "parse_conflict_uri",
    "expand_content_tokens",
    "splice_conflict",
    "conflict_regions_equal",
    "conflict_region_present",
    "render_conflict_region",
    "format_conflict_warning",
    "format_conflict_summary",
    "parse_bulk_directives",
]

OURS_PREFIX = "<<<<<<<"
BASE_PREFIX = "|||||||"
SEPARATOR = "======="
THEIRS_PREFIX = ">>>>>>>"

SCAN_FILE_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
PREVIEW_SIDE_LINES = 6
# Max lines inspected at each boundary when trimming echoed context.
ECHO_TRIM_LIMIT = 12

class ConflictError(Exception):
    """Raised by conflict primitives; callers convert to ToolError."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


Scope = Literal["ours", "theirs", "base"]


@dataclass(frozen=True, slots=True)
class ConflictBlock:
    """One fully-closed conflict marker block."""

    start_line: int  # 1-indexed line of <<<<<<<
    separator_line: int  # 1-indexed line of =======
    end_line: int  # 1-indexed line of >>>>>>>
    base_line: int | None = None  # 1-indexed line of ||||||| (diff3 only)
    ours_label: str | None = None
    base_label: str | None = None
    theirs_label: str | None = None
    ours_lines: tuple[str, ...] = ()  # LF-normalized (no \r)
    base_lines: tuple[str, ...] | None = None
    theirs_lines: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConflictEntry(ConflictBlock):
    """A registered block with a session-scoped id and source path."""

    id: int = 0
    absolute_path: str = ""
    display_path: str = ""


@dataclass(frozen=True, slots=True)
class ConflictSplice:
    """Result of splicing a resolved region back into file text."""

    text: str
    trimmed_leading: int
    trimmed_trailing: int


@dataclass(frozen=True, slots=True)
class ParsedConflictUri:
    id: int | Literal["*"]
    scope: Scope | None = None
    recovered_prefix: str | None = None  # "<file>:conflict://N" cleanup note


@dataclass(frozen=True, slots=True)
class ConflictScanResult:
    blocks: tuple[ConflictBlock, ...] = ()
    scan_truncated: bool = False


# ---------------------------------------------------------------------------
# Marker matching


def _strip_trailing_cr(line: str) -> str:
    return line[:-1] if line.endswith("\r") else line


def match_marker(line: str, prefix: str) -> str | None:
    """Strict column-0 marker match.

    Returns ``""`` for the bare prefix, the label string for
    ``prefix + single space + label``, and ``None`` for non-markers.
    Trailing ``\r`` is stripped first (CRLF checkouts).
    """
    line = _strip_trailing_cr(line)
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix):]
    if not rest:
        return ""
    if rest[0] == " ":
        label = rest[1:]
        # A label is required after the space and must not itself start
        # with a space ("<<<<<<<  two" never matches).
        if label and not label.startswith(" "):
            return label
    return None


def is_separator(line: str) -> bool:
    """The ``=======`` separator matches exactly — no label variant."""
    return _strip_trailing_cr(line) == SEPARATOR


# ---------------------------------------------------------------------------
# Line scanning


def scan_conflict_lines(
    lines: Sequence[str], first_line_number: int = 1
) -> list[ConflictBlock]:
    """State-machine scan (idle -> ours -> base -> theirs).

    Only fully-closed blocks are returned. Malformed sequences reset the
    partial block to idle (port of ``scanConflictLines``).
    """
    blocks: list[ConflictBlock] = []
    state = "idle"
    start_line = 0
    separator_line = 0
    base_line: int | None = None
    ours_label: str | None = None
    base_label: str | None = None
    theirs_label: str | None = None
    ours_buf: list[str] = []
    base_buf: list[str] = []
    theirs_buf: list[str] = []

    def reset() -> None:
        nonlocal state, start_line, separator_line, base_line
        nonlocal ours_label, base_label, theirs_label
        nonlocal ours_buf, base_buf, theirs_buf
        state = "idle"
        start_line = 0
        separator_line = 0
        base_line = None
        ours_label = base_label = theirs_label = None
        ours_buf, base_buf, theirs_buf = [], [], []

    for i, raw in enumerate(lines):
        line_no = first_line_number + i
        line = _strip_trailing_cr(raw)

        if state == "idle":
            label = match_marker(line, OURS_PREFIX)
            if label is not None:
                state = "ours"
                start_line = line_no
                ours_label = label or None
                ours_buf = []
                base_buf = []
                theirs_buf = []
                base_line = None
                separator_line = 0
            continue

        if state == "ours":
            label = match_marker(line, BASE_PREFIX)
            if label is not None:
                state = "base"
                base_line = line_no
                base_label = label or None
                continue
            if is_separator(line):
                state = "theirs"
                separator_line = line_no
                continue
            if match_marker(line, OURS_PREFIX) is not None:
                # Nested opener: restart at this line.
                start_line = line_no
                ours_label = match_marker(line, OURS_PREFIX) or None
                ours_buf = []
                base_buf = []
                theirs_buf = []
                base_line = None
                separator_line = 0
                continue
            ours_buf.append(line)
            continue

        if state == "base":
            if is_separator(line):
                state = "theirs"
                separator_line = line_no
                continue
            # Malformed: ||||||| outside ours or another opener resets.
            if (
                match_marker(line, OURS_PREFIX) is not None
                or match_marker(line, BASE_PREFIX) is not None
            ):
                reset()
                # Re-process this line as a fresh opener from idle.
                label = match_marker(line, OURS_PREFIX)
                if label is not None:
                    state = "ours"
                    start_line = line_no
                    ours_label = label or None
                    ours_buf = []
                continue
            base_buf.append(line)
            continue

        if state == "theirs":
            label = match_marker(line, THEIRS_PREFIX)
            if label is not None:
                theirs_label = label or None
                blocks.append(
                    ConflictBlock(
                        start_line=start_line,
                        separator_line=separator_line,
                        end_line=line_no,
                        base_line=base_line,
                        ours_label=ours_label,
                        base_label=base_label,
                        theirs_label=theirs_label,
                        ours_lines=tuple(ours_buf),
                        base_lines=tuple(base_buf) if base_line else None,
                        theirs_lines=tuple(theirs_buf),
                    )
                )
                reset()
                continue
            # Malformed in theirs phase: opener resets, reprocess.
            if match_marker(line, OURS_PREFIX) is not None:
                reset()
                state = "ours"
                start_line = line_no
                ours_label = match_marker(line, OURS_PREFIX) or None
                ours_buf = []
                continue
            theirs_buf.append(line)
            continue

    # Unclosed blocks at window tail are dropped.
    return blocks


def scan_file_for_conflicts(
    absolute_path: str, max_bytes: int = SCAN_FILE_DEFAULT_MAX_BYTES
) -> ConflictScanResult:
    """Whole-file scan with a byte cap (best-effort, cheap on clean files).

    Reads the first ``max_bytes`` bytes (UTF-8, ``errors='replace'``) and
    scans; a truncated tail can only drop unclosed openers, never invent
    blocks.
    """
    try:
        with open(absolute_path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError as exc:  # pragma: no cover - caller guards
        raise ConflictError(
                f"Cannot scan {absolute_path}: {exc}"
            )
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    blocks = scan_conflict_lines(text.split("\n"), 1)
    return ConflictScanResult(blocks=tuple(blocks), scan_truncated=truncated)


def find_dangling_openers(lines: Sequence[str]) -> list[tuple[int, str]]:
    """Lines holding an unclosed opener (file ends inside a conflict block).

    Returns ``(line_number, marker_line)`` pairs for ``<<<<<<<`` markers
    that were never closed by a matching ``>>>>>>>`` before EOF.
    """
    dangling: list[tuple[int, str]] = []
    state = "idle"
    open_line = 0
    open_text = ""
    for i, raw in enumerate(lines):
        line = _strip_trailing_cr(raw)
        line_no = i + 1
        if state == "idle":
            if match_marker(line, OURS_PREFIX) is not None:
                state = "ours"
                open_line = line_no
                open_text = line
            continue
        if state == "ours":
            if match_marker(line, THEIRS_PREFIX) is not None:
                state = "idle"
            elif match_marker(line, OURS_PREFIX) is not None:
                open_line = line_no
                open_text = line
            continue
    if state == "ours":
        dangling.append((open_line, open_text))
    return dangling


# ---------------------------------------------------------------------------
# Session history


class ConflictHistory:
    """Append-only per-session registry with stable ids."""

    def __init__(self) -> None:
        self._entries: dict[int, ConflictEntry] = {}
        self._next_id = 1
        self._by_key: dict[tuple[str, int], int] = {}

    def register(
        self,
        absolute_path: str,
        display_path: str,
        block: ConflictBlock,
    ) -> ConflictEntry:
        """Register a block; reuses the id for the same path + start_line."""
        key = (absolute_path, block.start_line)
        entry_id = self._by_key.get(key)
        if entry_id is not None and entry_id in self._entries:
            # Overwrite the recorded region (ids survive re-reads).
            old = self._entries[entry_id]
            entry = ConflictEntry(
                start_line=block.start_line,
                separator_line=block.separator_line,
                end_line=block.end_line,
                base_line=block.base_line,
                ours_label=block.ours_label,
                base_label=block.base_label,
                theirs_label=block.theirs_label,
                ours_lines=block.ours_lines,
                base_lines=block.base_lines,
                theirs_lines=block.theirs_lines,
                id=entry_id,
                absolute_path=absolute_path,
                display_path=display_path,
            )
            self._entries[entry_id] = entry
            return entry
        entry_id = self._next_id
        self._next_id += 1
        entry = ConflictEntry(
            start_line=block.start_line,
            separator_line=block.separator_line,
            end_line=block.end_line,
            base_line=block.base_line,
            ours_label=block.ours_label,
            base_label=block.base_label,
            theirs_label=block.theirs_label,
            ours_lines=block.ours_lines,
            base_lines=block.base_lines,
            theirs_lines=block.theirs_lines,
            id=entry_id,
            absolute_path=absolute_path,
            display_path=display_path,
        )
        self._entries[entry_id] = entry
        self._by_key[key] = entry_id
        return entry

    def get(self, entry_id: int) -> ConflictEntry | None:
        return self._entries.get(entry_id)

    def entries(self) -> list[ConflictEntry]:
        return sorted(self._entries.values(), key=lambda e: e.id)

    def invalidate(self, entry_id: int) -> None:
        entry = self._entries.pop(entry_id, None)
        if entry is not None:
            key = (entry.absolute_path, entry.start_line)
            if self._by_key.get(key) == entry_id:
                del self._by_key[key]

    def invalidate_path(self, absolute_path: str) -> None:
        removed = [
            entry_id
            for entry_id, entry in self._entries.items()
            if entry.absolute_path == absolute_path
        ]
        for entry_id in removed:
            self.invalidate(entry_id)


def get_conflict_history(session: Session) -> ConflictHistory:
    """Lazily attach and return the session's ConflictHistory."""
    history = getattr(session, "conflict_history", None)
    if history is None:
        history = ConflictHistory()
        session.conflict_history = history
    return history


# ---------------------------------------------------------------------------
# URI parsing

CONFLICT_URI_RE = re.compile(r"^(?:(.+):)?conflict://(.+)$")


def parse_conflict_uri(raw: str) -> ParsedConflictUri | None:
    """Parse a ``conflict://`` URI.

    Returns ``None`` for non-conflict paths. Raises ``ConflictError`` for a
    well-formed scheme with an invalid id/scope. ``*`` accepts no scope.
    """
    m = CONFLICT_URI_RE.match(raw)
    if m is None:
        return None
    prefix = m.group(1)
    body = m.group(2)

    scope: Scope | None = None
    id_part = body
    if "/" in body:
        id_part, _, scope_part = body.partition("/")
        if scope_part in ("ours", "theirs", "base"):
            scope = scope_part  # type: ignore[assignment]
        else:
            raise ConflictError(
                f"Invalid conflict scope '{scope_part}'. "
                    "Valid scopes: ours, theirs, base."
            )

    if id_part == "*":
        if scope is not None:
            raise ConflictError(
                "conflict://* does not accept a scope — it resolves every registered conflict."
            )
        return ParsedConflictUri(id="*", scope=None, recovered_prefix=prefix)

    try:
        entry_id = int(id_part)
    except ValueError:
        raise ConflictError(
                f"Invalid conflict id '{id_part}' in '{raw}'. "
                "Expected conflict://<N> or conflict://<N>/<ours|theirs|base>."
            )
    if entry_id <= 0:
        raise ConflictError(
                f"Invalid conflict id '{id_part}' — ids start at 1."
            )
    return ParsedConflictUri(id=entry_id, scope=scope, recovered_prefix=prefix)


# ---------------------------------------------------------------------------
# Content token expansion


def expand_content_tokens(content: str, entry: ConflictEntry) -> str:
    """Expand ``@ours``/``@theirs``/``@base``/``@both`` line tokens."""
    out: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "@ours":
            out.extend(entry.ours_lines)
        elif stripped == "@theirs":
            out.extend(entry.theirs_lines)
        elif stripped == "@base":
            if entry.base_lines is None:
                raise ConflictError(
                    f"@base is not available for conflict #{entry.id} — "
                    "it is a 2-way conflict (no ||||||| base section)."
                )
            out.extend(entry.base_lines)
        elif stripped == "@both":
            out.extend(entry.ours_lines)
            out.extend(entry.theirs_lines)
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Splice


def _region_lines(entry: ConflictBlock) -> list[str]:
    return [
        _render_marker(OURS_PREFIX, entry.ours_label),
        *entry.ours_lines,
        *([_render_marker(BASE_PREFIX, entry.base_label)] if entry.base_line else []),
        *(entry.base_lines or []),
        _render_marker(SEPARATOR, None),
        *entry.theirs_lines,
        _render_marker(THEIRS_PREFIX, entry.theirs_label),
    ]


def _render_marker(prefix: str, label: str | None) -> str:
    return prefix if not label else f"{prefix} {label}"


def _block_signature(entry: ConflictBlock) -> list[str]:
    """Label-free marker skeleton used to relocate a shifted block."""
    sig = [OURS_PREFIX, *entry.ours_lines]
    if entry.base_line:
        sig.append(BASE_PREFIX)
        sig.extend(entry.base_lines or ())
    sig.append(SEPARATOR)
    sig.extend(entry.theirs_lines)
    sig.append(THEIRS_PREFIX)
    return sig


def _matches_at(lines: list[str], idx: int, signature: list[str]) -> bool:
    if idx < 0 or idx + len(signature) > len(lines):
        return False
    for offset, expected in enumerate(signature):
        candidate = _strip_trailing_cr(lines[idx + offset])
        if candidate != expected:
            # Marker lines may carry labels; compare prefix for markers.
            if expected in (OURS_PREFIX, BASE_PREFIX, SEPARATOR, THEIRS_PREFIX):
                if match_marker(candidate, expected) is None:
                    return False
            else:
                return False
    return True


def _delimiter_balance(line: str) -> int:
    return line.count("(") - line.count(")") + line.count("[") - line.count("]")


def _echo_prefix_len(replacement: list[str], context: list[str]) -> int:
    """Largest k (<= ECHO_TRIM_LIMIT, keeps >=1 line) with
    replacement[:k] == context[-k:] — forward-order leading echo."""
    limit = min(ECHO_TRIM_LIMIT, len(replacement) - 1, len(context))
    best = 0
    for k in range(1, limit + 1):
        if replacement[:k] == context[-k:]:
            best = k
    return best


def _echo_suffix_len(replacement: list[str], context: list[str]) -> int:
    """Largest k (<= ECHO_TRIM_LIMIT, keeps >=1 line) with
    replacement[len-k:] == context[:k] — forward-order trailing echo."""
    limit = min(ECHO_TRIM_LIMIT, len(replacement) - 1, len(context))
    best = 0
    for k in range(1, limit + 1):
        if replacement[len(replacement) - k:] == context[:k]:
            best = k
    return best


def _trim_echo(
    preceding: list[str], following: list[str], replacement: list[str]
) -> tuple[list[str], int, int]:
    """Trim replacement lines that echo adjacent context (<= ECHO_TRIM_LIMIT).

    Multi-line echoes (2+ consecutive boundary lines) are always trimmed;
    a single-line echo is trimmed only when its delimiter balance is 0
    (self-contained), mirroring oh-my-pi's boundary-echo repair.
    """
    trimmed_leading = 0
    trimmed_trailing = 0
    if len(replacement) <= 1:
        return replacement, 0, 0

    # Leading echo: replacement head equal to lines preceding the region.
    k = _echo_prefix_len(replacement, preceding)
    if k >= 2 or (k == 1 and _delimiter_balance(replacement[0]) == 0):
        replacement = replacement[k:]
        trimmed_leading = k

    # Trailing echo: replacement tail equal to lines after the region.
    k = _echo_suffix_len(replacement, following)
    if k >= 2 or (
        k == 1 and _delimiter_balance(replacement[len(replacement) - 1]) == 0
    ):
        replacement = replacement[: len(replacement) - k]
        trimmed_trailing = k

    return replacement, trimmed_leading, trimmed_trailing


def splice_conflict(
    original_text: str, entry: ConflictEntry, replacement: str
) -> ConflictSplice:
    """Replace the recorded marker region with ``replacement``.

    Locates the region by content anchored at ``entry.start_line``; falls
    back to the nearest occurrence. Trims boundary echo and re-applies
    CRLF when the matched region used it. Raises ``ConflictError`` when the
    recorded block is gone or altered.
    """
    uses_crlf = "\r\n" in original_text
    text_lf = original_text.replace("\r\n", "\n")
    lines = text_lf.split("\n")
    signature = _block_signature(entry)

    anchor = entry.start_line - 1  # 0-indexed
    located: int | None = None
    if _matches_at(lines, anchor, signature):
        located = anchor
    else:
        best_distance = None
        for idx in range(len(lines)):
            if _matches_at(lines, idx, signature):
                distance = abs(idx - anchor)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    located = idx
    if located is None:
        raise ConflictError(
                f"Conflict #{entry.id} no longer matches the recorded block at "
                f"{entry.display_path}:{entry.start_line}. Re-read the file to get a current conflict id."
            )

    region_len = len(signature)
    replacement_lines = [
        _strip_trailing_cr(ln) for ln in replacement.split("\n")
    ] if replacement.strip() or replacement else []
    if replacement == "":
        replacement_lines = []

    preceding = lines[max(0, located - ECHO_TRIM_LIMIT): located]
    following = lines[located + region_len: located + region_len + ECHO_TRIM_LIMIT]
    replacement_lines, trimmed_leading, trimmed_trailing = _trim_echo(
        preceding, following, replacement_lines
    )

    new_lines = lines[:located] + replacement_lines + lines[located + region_len:]
    text = "\n".join(new_lines)
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    return ConflictSplice(
        text=text, trimmed_leading=trimmed_leading, trimmed_trailing=trimmed_trailing
    )


# ---------------------------------------------------------------------------
# Region comparison


def _region_text(entry: ConflictBlock) -> str:
    return "\n".join(_region_lines(entry))


def conflict_regions_equal(a: ConflictBlock, b: ConflictBlock) -> bool:
    return (
        a.start_line == b.start_line
        and a.end_line == b.end_line
        and _region_text(a) == _region_text(b)
    )


def conflict_region_present(content: str, entry: ConflictBlock) -> bool:
    text_lf = content.replace("\r\n", "\n")
    return _region_text(entry) in text_lf


# ---------------------------------------------------------------------------
# Rendering / formatting


def render_conflict_region(
    entry: ConflictEntry, scope: Scope | None
) -> tuple[list[str], int]:
    """``(lines, start_line)`` for ``conflict://N`` reads."""
    if scope is None:
        lines = _region_lines(entry)
        return lines, entry.start_line
    if scope == "ours":
        return list(entry.ours_lines), entry.start_line + 1
    if scope == "theirs":
        return list(entry.theirs_lines), entry.separator_line + 1
    if scope == "base":
        if entry.base_lines is None:
            raise ConflictError(
                f"Conflict #{entry.id} is a 2-way conflict — no base section. "
                    "Use /ours or /theirs."
            )
        assert entry.base_line is not None
        return list(entry.base_lines), entry.base_line + 1
    raise ConflictError(
                f"Unknown conflict scope '{scope}'."
            )


def _preview_lines(lines: tuple[str, ...]) -> list[str]:
    if not lines:
        return ["(empty)"]
    if len(lines) <= PREVIEW_SIDE_LINES:
        return list(lines)
    shown = lines[:PREVIEW_SIDE_LINES]
    return [*shown, f"… ({len(lines) - PREVIEW_SIDE_LINES} more lines)"]


def _collect_labels(entries: Sequence[ConflictEntry]) -> tuple[list[str], list[str]]:
    ours_labels: list[str] = []
    theirs_labels: list[str] = []
    for entry in entries:
        if entry.ours_label and entry.ours_label not in ours_labels:
            ours_labels.append(entry.ours_label)
        if entry.theirs_label and entry.theirs_label not in theirs_labels:
            theirs_labels.append(entry.theirs_label)
    return ours_labels, theirs_labels


def format_conflict_warning(
    entries: Sequence[ConflictEntry],
    *,
    total_in_file: int | None = None,
    display_path: str | None = None,
    scan_truncated: bool = False,
) -> str:
    """Footer appended to read output (port of ``formatConflictWarning``)."""
    if not entries:
        return ""
    count = len(entries)
    total = total_in_file if total_in_file is not None else count
    path_note = f" in {display_path}" if display_path else ""
    lines: list[str] = []
    if total > count:
        lines.append(
            f"⚠ {count} of {total} unresolved conflicts visible in this window"
            f"{path_note} (read `{display_path}:conflicts` for the full list)."
        )
    else:
        plural = "conflict" if count == 1 else "conflicts"
        lines.append(f"⚠ {count} unresolved {plural} detected{path_note}")

    ours_labels, theirs_labels = _collect_labels(entries)
    if ours_labels:
        lines.append(f"- ours = {', '.join(ours_labels)}")
    if theirs_labels:
        lines.append(f"- theirs = {', '.join(theirs_labels)}")
    if scan_truncated:
        lines.append(
            "- note: file scan hit the byte cap; additional conflicts may exist beyond the scanned prefix."
        )
    lines.append(
        "NOTICE: Inspect a block by reading `conflict://<N>` (add `/ours` / `/theirs` / `/base` to render a single side). "
        'Resolve with `write({ path: "conflict://<N>", content })`, or bulk-resolve every registered conflict with '
        '`write({ path: "conflict://*", content })`. Writes replace ONLY the marker block — never repeat the lines '
        "before/after it; they stay in place."
    )
    lines.append(
        "`content` shorthand: a line exactly `@ours` / `@theirs` / `@base` / `@both` expands to that recorded section; "
        "`@both` is ours-then-theirs (additive conflicts only — never for competing edits of the same lines). "
        "Non-token lines pass through verbatim. Keep one side or combine faithfully; never invent content beyond the recorded sides."
    )

    for entry in entries:
        lines.append(f"──── #{entry.id}  L{entry.start_line}-{entry.end_line} ────")
        if entry.base_lines is not None and entry.base_lines == entry.ours_lines:
            lines.append("=== base ≡ ours")
        elif entry.base_lines is not None:
            lines.append("=== base")
            lines.extend(_preview_lines(entry.base_lines))
        if entry.theirs_lines == entry.ours_lines:
            lines.append(">>> theirs ≡ ours")
        else:
            lines.append("<<< ours")
            lines.extend(_preview_lines(entry.ours_lines))
            lines.append(">>> theirs")
            lines.extend(_preview_lines(entry.theirs_lines))
    return "\n".join(lines)


def format_conflict_summary(
    entries: Sequence[ConflictEntry],
    *,
    display_path: str,
    scan_truncated: bool = False,
) -> str:
    """One-line-per-block index for the ``:conflicts`` selector."""
    if not entries:
        return f"No unresolved git merge conflicts in {display_path}."
    count = len(entries)
    plural = "conflict" if count == 1 else "conflicts"
    lines: list[str] = [f"⚠ {count} unresolved {plural} in {display_path}"]
    ours_labels, theirs_labels = _collect_labels(entries)
    if ours_labels:
        lines.append(f"- ours = {', '.join(ours_labels)}")
    if theirs_labels:
        lines.append(f"- theirs = {', '.join(theirs_labels)}")
    if scan_truncated:
        lines.append(
            "- note: file scan hit the byte cap; additional conflicts may exist beyond the scanned prefix."
        )
    lines.append(
        "NOTICE: Bulk-resolve with `write({ path: \"conflict://*\", content })`, or address a single block with "
        '`write({ path: "conflict://<N>", content })`. A line exactly `@ours` / `@theirs` / `@base` / `@both` expands '
        "to that recorded section; non-token lines pass through verbatim."
    )
    for entry in entries:
        suffix = "  (3-way)" if entry.base_line is not None else ""
        lines.append(f"#{entry.id}  L{entry.start_line}-{entry.end_line}{suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bulk directives


_BULK_DIRECTIVE_RE = re.compile(r"^\s*(\d+)\s*:\s*@(ours|theirs|base|both)\s*$")


def parse_bulk_directives(content: str) -> dict[int, str] | None:
    """Parse per-id ``<id>: @side`` directives.

    Returns ``{id: side}`` when every non-empty line is a directive,
    else ``None`` (content applies to every registered entry instead).
    """
    directives: dict[int, str] = {}
    saw_any = False
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        saw_any = True
        m = _BULK_DIRECTIVE_RE.match(raw_line)
        if m is None:
            return None
        directives[int(m.group(1))] = m.group(2)
    return directives if saw_any else None
