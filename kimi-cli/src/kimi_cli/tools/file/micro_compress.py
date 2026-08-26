"""Micro-compression pipeline for smart token reduction.

Implements character / word / line-level transforms that run on tool output
and stale history text to reduce token consumption without destroying
information.  See ``plan.md`` for the full design.

All stages are:

* **Pure functions** — no side effects, easily unit-testable.
* **Deterministic & idempotent** — ``compress(compress(x)) == compress(x)``.
* **Cheap** — O(n) single-pass or near-linear string ops.
* **Independently toggleable** via :class:`MicroCompressConfig`.

The nine stages mirror the catalogue in plan.md §6::

    Stage 1  normalize_encoding       (Class E, lossless)
    Stage 2  strip_control_noise      (Class E, lossless)
    Stage 3  collapse_whitespace      (Class A, lossless-or-annotated)
    Stage 4  fold_per_line_prefix     (Class B, annotated)
    Stage 5  renumber_lines           (Class B3, lossless)
    Stage 6  drop_boilerplate         (Class C, annotated)
    Stage 7  intra_line_dedup         (Class F1, annotated)
    Stage 8  near_duplicate_collapse  (Class F2, annotated)
    Stage 9  elide_low_value_content  (Class D, opt-in/lossy)

Usage::

    from kimi_cli.tools.file.micro_compress import compress, infer_content_kind

    kind = infer_content_kind(path="foo.py")
    compact = compress(text, kind=kind)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import regex as re
import xxhash
from rapidfuzz import fuzz

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MicroCompressConfig:
    """Toggleable configuration for the micro-compression pipeline.

    Defaults match the safe values from plan.md §10.
    """

    enabled: bool = True
    """Master switch."""

    lossless_only: bool = False
    """If *True*, disable annotated stages 4, 6, 7, 8."""

    blank_line_collapse: int = 1
    """Max consecutive blank lines kept (A1). 0 ⇒ remove all blanks."""

    strip_trailing_ws: bool = True
    """Strip trailing whitespace on every line (A2)."""

    common_indent_factor: bool = True
    """Factor out common leading indentation (A3, non-code only)."""

    prefix_fold_min_chars: int = 8
    """Stage 4 — minimum common-prefix length to trigger fold."""

    prefix_fold_min_ratio: float = 0.80
    """Stage 4 — minimum fraction of lines sharing the prefix."""

    prefix_fold_min_lines: int = 20
    """Stage 4 — minimum non-blank line count to consider folding."""

    prefix_fold: bool = True
    """Stage 4 — enable per-line prefix folding (``[prefix: …]``).
    ``lossless_only`` forces this off."""

    banner_drop: bool = True
    """Stage 6 — drop leading boilerplate/banner lines."""

    intra_line_dedup_len: int = 2000
    """Stage 7 — minimum line length to attempt intra-line dedup."""

    intra_line_dedup: bool = True
    """Stage 7 — enable intra-line repetition dedup (``×k [+M chars]``).
    ``lossless_only`` forces this off."""

    near_dup_min_run: int = 4
    """Stage 8 — minimum run length for near-duplicate collapse."""

    near_dup_threshold: int = 90
    """Stage 8 — ``fuzz.ratio`` threshold (0-100) for near-duplicate."""

    near_dup_collapse: bool = True
    """Stage 8 — enable near-duplicate line collapse (``[×k near-dup …]``).
    ``lossless_only`` forces this off.  Disable for search tools (Grep) so
    distinct matches are never hidden."""

    read_compact_code: bool = False
    """Stage 9 — opt-in code content elision (default off)."""


# ---------------------------------------------------------------------------
# Content-kind inference (plan.md §7)
# ---------------------------------------------------------------------------

_CODE_EXTS: set[str] = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".rs", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".java", ".kt", ".kts", ".go", ".rb", ".php", ".swift",
    ".m", ".mm", ".cs", ".scala", ".clj", ".cljs", ".ex", ".exs",
    ".erl", ".hs", ".lua", ".pl", ".pm", ".r", ".jl", ".dart",
    ".vue", ".svelte", ".zig", ".nim", ".v", ".sv", ".ml", ".fs",
}
_PROSE_EXTS: set[str] = {
    ".md", ".markdown", ".txt", ".rst", ".org", ".adoc", ".asciidoc", ".tex",
}
_DATA_EXTS: set[str] = {
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
    ".conf", ".csv", ".tsv", ".properties", ".proto", ".graphql", ".gql",
}
_LOG_EXTS: set[str] = {".log", ".out", ".err"}

_TOOL_KINDS: dict[str, str] = {
    "bash": "log",
    "powershell": "log",
    "pwsh": "log",
    "run": "log",
    "python": "log",
    "fetch_url": "prose",
    "fetchurl": "prose",
    "fetch": "prose",
    "grep": "log",
    "glob": "log",
}


def infer_content_kind(path: str | None = None, tool: str | None = None) -> str:
    """Infer the ``content_kind`` hint (``code | prose | data | log``).

    Extension takes priority; tool name is a fallback.  Unknown → ``log``
    (conservative: only lossless stages run).
    """
    if path:
        ext = Path(path).suffix.lower()
        if ext in _CODE_EXTS:
            return "code"
        if ext in _PROSE_EXTS:
            return "prose"
        if ext in _DATA_EXTS:
            return "data"
        if ext in _LOG_EXTS:
            return "log"
    if tool:
        return _TOOL_KINDS.get(tool.lower(), "log")
    return "log"


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

# Stage 1 — unicode whitespace / zero-width / C0 controls
_UNICODE_WS = re.compile(r"[\u00a0\u0085\u2000-\u200a\u202f\u205f\u3000]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u180e]")
# C0 controls to strip: everything except \t(09) \n(0a) \r(0d — left for Stage 2)
_C0_STRIP = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Stage 2 — ANSI / OSC escape sequences
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_OTHER = re.compile(r"\x1b[@-Z\\^_]")

# Stage 3 — whitespace patterns
# A4 collapses *internal* space runs only — leading indentation is never
# touched (log-kind text can contain code blocks, e.g. ``cat file.py``).
_INTERNAL_SPACE_RUN = re.compile(r"(?<=\S) {3,}(?=\S)")

# Stage 4 — timestamp prefix
_TIMESTAMP_START = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+"
)

# Stage 5 — line-number pattern (ReadFile ``{n:6d}\t``)
_LINENO_RE = re.compile(r"^\s*(\d+)\t")

# Stage 6 — banner detection
# Keyword must be followed by whitespace or EOL (not ``.``/``:``), so grep
# results like ``python.py:12:...`` are never mistaken for banners.
_BANNER_KEYWORDS = re.compile(
    r"^\s*(?:npm|cargo|pip|uv|tsc|yarn|pnpm|deno|bun|python|node|ruby|gem"
    r"|gradle|maven|mvn|make|cmake|dotnet|rustc|gcc|clang)(?:\s|$)",
    re.IGNORECASE,
)
_SYSTEM_META_RE = re.compile(r"^<system>.*</system>$", re.DOTALL)

# Stage 9 — lockfile detection
_LOCKFILE_NAMES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "cargo.lock", "composer.lock", "gemfile.lock", "poetry.lock",
    "go.sum", "uv.lock", "pipfile.lock", "moz-build-lock.json",
}
_LOCKFILE_EXTS: set[str] = {".lock"}

#: Longest common prefix/indent we ever fold (Stage 3/4) and longest repeating
#: unit we scan for (Stage 7). These caps keep the helpers linear in the total
#: characters actually compared instead of O(n²) when the first line is
#: pathologically long (e.g. a 3MB single-line data file matched by Grep). A
#: shared prefix/indent/unit longer than this is never worth folding for token
#: reduction, so truncating the scan is lossless in practice.
_MAX_PREFIX_SCAN = 8192
_MAX_INDENT_SCAN = 8192
_MAX_INTRA_LINE_UNIT = 2048


# ---------------------------------------------------------------------------
# Stage 1 — normalize_encoding  (Class E, lossless)
# ---------------------------------------------------------------------------


def normalize_encoding(text: str) -> str:
    """NFC-normalise, strip BOM/zero-width, normalise unicode whitespace,
    CRLF→LF, and strip C0 control chars (except ``\\t`` ``\\n`` ``\\r``).

    Idempotent and truly lossless.
    """
    text = unicodedata.normalize("NFC", text)
    text = _UNICODE_WS.sub(" ", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n")
    text = _C0_STRIP.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Stage 2 — strip_control_noise  (Class E1/E2, lossless)
# ---------------------------------------------------------------------------


def strip_control_noise(text: str) -> str:
    """Remove ANSI/OSC/DCS escape sequences and collapse CR progress-bar
    chains (keep the last frame only).

    Idempotent and lossless (the removed bytes are terminal-control noise).
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and text.isascii():
        return _NATIVE_TOOLS.compress_strip_control_noise(text)
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_OTHER.sub("", text)
    # CR progress-bar collapse: within each line keep only the segment
    # after the last \r (the final rendered frame).
    if "\r" in text:
        lines = text.split("\n")
        result: list[str] = []
        for line in lines:
            if "\r" in line:
                result.append(line.rsplit("\r", 1)[-1])
            else:
                result.append(line)
        text = "\n".join(result)
    return text



# ---------------------------------------------------------------------------
# Stage 3 — collapse_whitespace  (Class A, lossless-or-annotated)
# ---------------------------------------------------------------------------


def collapse_whitespace(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
) -> str:
    """Collapse redundant whitespace.

    * **A2** strip trailing whitespace (always, if configured).
    * **A1** collapse runs of blank lines to *≤ config.blank_line_collapse*.
    * **A3** factor out common leading indent ≥ 4 cols (non-code, annotated).
    * **A4** collapse internal 3+ space runs to one (prose/log only).
    """
    if config is None:
        config = MicroCompressConfig()

    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and text.isascii():
        return _NATIVE_TOOLS.compress_collapse_whitespace(text, kind, config)
    lines = text.split("\n")
    # A2 — strip trailing whitespace
    # For code kind, only strip trailing spaces (not tabs) so that
    # line-number separator tabs (e.g. ReadFile ``5\tcontent``) survive.
    if config.strip_trailing_ws:
        if kind == "code":
            lines = [ln.rstrip(" ") for ln in lines]
        else:
            lines = [ln.rstrip() for ln in lines]

    # A1 — collapse blank-line runs
    max_blanks = config.blank_line_collapse
    if max_blanks >= 0:
        collapsed: list[str] = []
        blank_run = 0
        for ln in lines:
            if ln.strip() == "":
                blank_run += 1
                if blank_run <= max_blanks:
                    collapsed.append("")
            else:
                blank_run = 0
                collapsed.append(ln)
        lines = collapsed

    # A3 — factor common indentation (non-code, non-lossless-only)
    if (
        config.common_indent_factor
        and kind != "code"
        and not config.lossless_only
    ):
        lines = _factor_common_indent(lines, lines)

    # A4 — collapse internal space runs (prose/log only)
    if kind in ("prose", "log"):
        lines = [_INTERNAL_SPACE_RUN.sub(" ", ln) for ln in lines]

    return "\n".join(lines)


def _factor_common_indent(
    lines: list[str],
    all_lines: list[str],
) -> list[str]:
    """Detect a common leading-whitespace prefix across non-blank lines.

    If the prefix is ≥ 4 characters, factor it out and prepend a marker.
    Returns the (possibly modified) line list.
    """
    non_blank = [ln for ln in lines if ln.strip()]
    if len(non_blank) < 2:
        return lines

    # Linear scan, capped: ``common[:-1]`` in a while-loop is O(n²) when the
    # first line is huge and the next line has no matching prefix (a 3MB
    # single-line match would stall Grep for minutes). A common indent longer
    # than _MAX_INDENT_SCAN chars is meaningless for folding.
    common = non_blank[0][:_MAX_INDENT_SCAN]
    for ln in non_blank[1:]:
        stripped = ln.lstrip(" \t")
        indent = ln[: len(ln) - len(stripped)][:_MAX_INDENT_SCAN]
        n = min(len(common), len(indent))
        i = 0
        while i < n and common[i] == indent[i]:
            i += 1
        common = common[:i]
        if not common:
            break

    if not common or len(common) < 4:
        return lines

    n_cols = len(common)
    prefix_len = len(common)
    new_lines = [
        (ln[prefix_len:] if ln.startswith(common) else ln) for ln in lines
    ]
    return [f"[common-indent: {n_cols} cols removed]"] + new_lines


# ---------------------------------------------------------------------------
# Stage 5 — renumber_lines  (Class B3, lossless)
# ---------------------------------------------------------------------------


def renumber_lines(text: str) -> str:
    """Compact fixed-width line numbers ("  42\\t" -> "42\\t").

    Only fires when *every* substantial line matches ^\\s*\\d+\\t
    (ReadFile-style output).  The bijection is preserved exactly.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and text.isascii():
        return _NATIVE_TOOLS.compress_renumber_lines(text)
    lines = text.split("\n")

    substantial = 0
    numbered = 0
    for ln in lines:
        if ln.strip() == "" or ln.startswith("[") or ln.startswith("…"):
            continue
        substantial += 1
        if _LINENO_RE.match(ln):
            numbered += 1

    if substantial == 0 or numbered < substantial:
        return text

    new_lines: list[str] = []
    for ln in lines:
        m = _LINENO_RE.match(ln)
        if m:
            num = int(m.group(1))
            rest = ln[m.end():]
            new_lines.append(f"{num}\t{rest}")
        else:
            new_lines.append(ln)
    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Stage 4 — fold_per_line_prefix  (Class B, annotated)
# ---------------------------------------------------------------------------


def fold_per_line_prefix(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
) -> str:
    """Fold a common per-line prefix (paths, timestamps) into a single marker.

    * **B1** log-timestamp prefix → ``[ts-prefix folded]``.
    * **B-general** any common prefix ≥ *min_chars* shared by ≥ *min_ratio*
      of lines → ``[prefix: "<prefix>"]``.

    Skipped for ``code`` (indentation is handled by Stage 3).
    """
    if config is None:
        config = MicroCompressConfig()
    if kind == "code" or config.lossless_only or not config.prefix_fold:
        return text

    lines = text.split("\n")
    non_blank = [ln for ln in lines if ln.strip()]
    if len(non_blank) < config.prefix_fold_min_lines:
        return text

    # --- B1: timestamp prefix ---
    ts_count = sum(1 for ln in non_blank if _TIMESTAMP_START.match(ln))
    if ts_count / len(non_blank) >= config.prefix_fold_min_ratio:
        # Strip timestamps, then check for a common post-timestamp prefix
        stripped_lines = []
        for ln in lines:
            m = _TIMESTAMP_START.match(ln)
            stripped_lines.append(ln[m.end():] if m else ln)
        nb_stripped = [
            sl for sl in stripped_lines if sl.strip()
        ]
        common = _longest_common_prefix(nb_stripped)
        if len(common) >= config.prefix_fold_min_chars:
            new_lines = [
                (sl[len(common):] if sl.startswith(common) else sl)
                for sl in stripped_lines
            ]
            return f'[ts-prefix folded, prefix: "{common}"]\n' + "\n".join(
                new_lines
            )
        # Just fold timestamps, no common post-ts prefix
        return "[ts-prefix folded]\n" + "\n".join(stripped_lines)

    # --- B-general: common prefix ---
    common = _longest_common_prefix(non_blank)
    if len(common) < config.prefix_fold_min_chars:
        return text

    sharing = sum(1 for ln in lines if ln.startswith(common))
    if len(lines) == 0 or sharing / len(lines) < config.prefix_fold_min_ratio:
        return text

    new_lines = [
        (ln[len(common):] if ln.startswith(common) else ln) for ln in lines
    ]
    return f'[prefix: "{common}"]\n' + "\n".join(new_lines)


def _longest_common_prefix(strings: list[str]) -> str:
    """Return the longest string that is a prefix of *all* inputs."""
    if not strings:
        return ""
    # Linear scan, capped: the previous ``prefix = prefix[:-1]`` loop is O(n²)
    # when the first string is huge (e.g. a 3MB single-line Grep match) and the
    # next string shares little of it. A shared prefix longer than
    # _MAX_PREFIX_SCAN chars is never worth folding for token reduction, and a
    # truncated common prefix is still a valid (shorter) common prefix.
    prefix = strings[0][:_MAX_PREFIX_SCAN]
    for s in strings[1:]:
        s = s[:_MAX_PREFIX_SCAN]
        n = min(len(prefix), len(s))
        i = 0
        while i < n and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


# ---------------------------------------------------------------------------
# Stage 6 — drop_boilerplate  (Class C, annotated)
# ---------------------------------------------------------------------------


def drop_boilerplate(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
) -> str:
    """Drop leading banner / ASCII-art lines and merge adjacent identical
    ``<system>…</system>`` metadata.

    * **C1/C2** drop leading banner lines → ``[N banner lines dropped]``.
    * **C3** merge adjacent identical ``<system>`` metadata into one.
    """
    if config is None:
        config = MicroCompressConfig()
    # C3 — always merge system metadata (applies to all kinds)
    if config.lossless_only or not config.banner_drop or kind != "log":
        return _merge_system_metadata(text)

    lines = text.split("\n")
    if not lines:
        return text

    # C1/C2 — find leading banner lines (log kind only, per plan §6)
    banner_end = 0
    for i, ln in enumerate(lines):
        if _is_banner_line(ln):
            banner_end = i + 1
        elif ln.strip() == "":
            # Blank lines between banners are part of the banner block
            if banner_end > 0:
                banner_end = i + 1
        else:
            break

    # C3 — always merge system metadata
    if banner_end == 0:
        return _merge_system_metadata(text)

    remaining = lines[banner_end:]
    if not remaining:
        return text  # never drop everything

    result = [f"[{banner_end} banner lines dropped]"] + remaining
    return _merge_system_metadata("\n".join(result))


def _is_banner_line(line: str) -> bool:
    """Heuristic: does *line* look like a tool banner or ASCII-art line?"""
    stripped = line.strip()
    if not stripped:
        return False
    # Known command banner keywords
    if _BANNER_KEYWORDS.match(stripped):
        return True
    # ASCII art: high non-alphanumeric ratio
    if len(stripped) > 4:
        non_alnum = sum(1 for c in stripped if not c.isalnum() and c != " ")
        if non_alnum / len(stripped) > 0.65:
            return True
    return False


def _merge_system_metadata(text: str) -> str:
    """Merge adjacent identical ``<system>…</system>`` lines into one."""
    lines = text.split("\n")
    result: list[str] = []
    prev_meta: str | None = None
    for ln in lines:
        if _SYSTEM_META_RE.match(ln.strip()):
            if ln == prev_meta:
                continue
            prev_meta = ln
        else:
            prev_meta = None
        result.append(ln)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Stage 7 — intra_line_dedup  (Class F1, annotated)
# ---------------------------------------------------------------------------


def intra_line_dedup(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
) -> str:
    """Collapse a single very long line composed of a short repeating unit.

    unit × k → unit ×k [+M chars elided].
    """
    if config is None:
        config = MicroCompressConfig()
    if config.lossless_only or kind == "code" or not config.intra_line_dedup:
        return text
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and text.isascii():
        return _NATIVE_TOOLS.compress_intra_line_dedup(
            text, config.intra_line_dedup_len, _MAX_INTRA_LINE_UNIT
        )
    threshold = config.intra_line_dedup_len
    lines = text.split("\n")

    changed = False
    new_lines: list[str] = []
    for ln in lines:
        if len(ln) > threshold:
            compressed = _compress_repeating_unit(ln)
            if compressed is not ln:
                changed = True
            new_lines.append(compressed)
        else:
            new_lines.append(ln)
    return "\n".join(new_lines) if changed else text


def _compress_repeating_unit(line: str) -> str:
    """If *line* is ``unit × k`` for a short *unit*, return a compact form."""
    n = len(line)
    if n < 6:
        return line
    # Cap the unit length: only *short* repeating units are worth this marker,
    # and scanning every divisor of a 3MB+ line (and doing an O(n) equality per
    # divisor) is wasteful. _MAX_INTRA_LINE_UNIT bounds both the loop and the
    # number of O(n) equality checks.
    max_unit = min(n // 3, _MAX_INTRA_LINE_UNIT)
    for p in range(1, max_unit + 1):
        if n % p != 0:
            continue
        unit = line[:p]
        if unit * (n // p) == line:
            repeats = n // p
            elided = n - p
            marker = f" ×{repeats} [+{elided} chars elided]"
            if len(unit) + len(marker) < n:
                return f"{unit}{marker}"
            return line
    return line


# ---------------------------------------------------------------------------
# Stage 8 — near_duplicate_collapse  (Class F2, annotated)
# ---------------------------------------------------------------------------


def near_duplicate_collapse(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
) -> str:
    """Collapse runs of ≥ *min_run* adjacent near-duplicate lines.

    Lines differing only in a small **contiguous counter** field → first line +
    ``[×k near-dup, fieldN a→b]``.  The ``a→b`` marker is only emitted when the
    changing numeric field forms a contiguous counter sequence (e.g. ``item 5``
    … ``item 14``); near-duplicate lines that differ in *discrete identifiers*
    (e.g. ``P-053: 2 occurrences`` vs ``P-055: 2 occurrences``) are kept as-is,
    because collapsing them behind a range marker would hide distinct values the
    model may need to act on, and a ``a→b`` range would be misleading when the
    intermediate values are absent.
    """
    if config is None:
        config = MicroCompressConfig()
    if (
        config.lossless_only
        or kind == "code"
        or not config.near_dup_collapse
    ):
        return text

    min_run = config.near_dup_min_run
    threshold = config.near_dup_threshold
    if min_run < 2:
        return text

    lines = text.split("\n")
    if len(lines) < min_run:
        return text

    result: list[str] = []
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and fuzz.ratio(lines[i], lines[j]) >= threshold:
            j += 1
        run_len = j - i
        # Skip runs of perfectly identical lines — the existing ``dedup_lines``
        # stage owns those (it reports the count in its own marker format).
        has_distinct = any(ln != lines[i] for ln in lines[i:j])
        if run_len >= min_run and has_distinct and _is_counter_run(lines[i:j]):
            result.append(lines[i])
            result.append(_near_dup_marker(lines[i:j]))
            i = j
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def _is_counter_run(run_lines: list[str]) -> bool:
    """Return True when a near-duplicate run differs only in a contiguous counter.

    A run is a *counter* when every line has the same shape and exactly one
    numeric field varies, and the varying values form a contiguous increasing
    (or decreasing) sequence with step ±1.  This guarantees the ``[×k near-dup,
    fieldN a→b]`` marker is a truthful range rather than a lossy collapse of
    distinct identifiers.

    Runs whose differing field skips values (e.g. ``P-053``/``P-055``/``P-057``)
    or whose variation is non-numeric are NOT counters and are preserved verbatim.
    """
    if len(run_lines) < 2:
        return False
    first_tokens = re.findall(r"\d+", run_lines[0])
    if not first_tokens:
        return False
    n_fields = len(first_tokens)
    # Every line must have the same number of numeric fields so the field
    # positions line up.
    for line in run_lines[1:]:
        if len(re.findall(r"\d+", line)) != n_fields:
            return False
    # Find the single field that varies across the run; all other fields must be
    # constant.  A line that differs from the first in more than one numeric
    # field is a different entry, not a counter increment.
    varying_idx: int | None = None
    for field in range(n_fields):
        values = [int(re.findall(r"\d+", ln)[field]) for ln in run_lines]
        if all(v == values[0] for v in values):
            continue
        if varying_idx is not None:
            # More than one field varies — not a simple counter.
            return False
        varying_idx = field
    if varying_idx is None:
        # No field varies: identical lines, owned by ``dedup_lines``.
        return False
    values = [int(re.findall(r"\d+", ln)[varying_idx]) for ln in run_lines]
    steps = {values[k + 1] - values[k] for k in range(len(values) - 1)}
    # Allow monotonic ±1 steps only; anything else (e.g. 053→055→057) means the
    # values are discrete identifiers and must be preserved.
    return steps in ({1}, {-1})


def _near_dup_marker(run_lines: list[str]) -> str:
    """Build a ``[×k near-dup, fieldN a→b]`` marker for a run."""
    k = len(run_lines) - 1
    first = run_lines[0]
    last = run_lines[-1]
    nums_first = re.findall(r"\d+", first)
    nums_last = re.findall(r"\d+", last)
    field_desc = ""
    if len(nums_first) == len(nums_last) and nums_first:
        for idx, (a, b) in enumerate(zip(nums_first, nums_last)):
            if a != b:
                field_desc = f", field{idx} {a}→{b}"
                break
    return f"[×{k} near-dup{field_desc}]"


# ---------------------------------------------------------------------------
# Stage 9 — elide_low_value_content  (Class D, opt-in, lossy)
# ---------------------------------------------------------------------------


def elide_low_value_content(
    text: str,
    kind: str = "code",
    config: MicroCompressConfig | None = None,
    path: str | None = None,
    active_edit_files: set[str] | None = None,
) -> str:
    """Elide license headers, import clusters, comment blocks, and lockfiles.

    **Opt-in only** (``config.read_compact_code``).  Never fires on the
    file currently being edited.  Replaced content is annotated with a
    descriptive, hash-bearing marker.
    """
    if config is None:
        config = MicroCompressConfig()
    if not config.read_compact_code or kind != "code":
        return text
    if path and active_edit_files and path in active_edit_files:
        return text

    # D4 — lockfile / minified bulk
    if path and _is_lockfile(path):
        lines = text.split("\n")
        n = len(lines)
        h = xxhash.xxh64(text.encode()).hexdigest()
        return f"[{n} lines of generated content, hash=xxh64:{h}]"

    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # D3 — license header (near top of file)
        if i < 30 and _is_license_line(stripped):
            j = i
            while j < len(lines) and _is_license_line(lines[j].strip()):
                j += 1
            if j - i >= 2:
                result.append(f"[{j - i} license lines elided]")
                i = j
                continue

        # D2 — import clusters (Python)
        if _is_import_line(stripped):
            j = i
            while j < len(lines) and _is_import_line(lines[j].strip()):
                j += 1
            if j - i >= 3:
                result.append(f"[{j - i} imports]")
                i = j
                continue

        # D1 — comment blocks
        if _is_comment_line(stripped):
            j = i
            while j < len(lines) and _is_comment_line(lines[j].strip()):
                j += 1
            if j - i >= 3:
                result.append(f"[{j - i} comment lines elided]")
                i = j
                continue

        result.append(line)
        i += 1
    return "\n".join(result)


_LICENSE_KEYWORDS = re.compile(
    r"copyright|licensed|license|warranty|SPDX|apache|mit license|"
    r"bsd|gpl|mozilla|all rights reserved",
    re.IGNORECASE,
)


def _is_license_line(line: str) -> bool:
    if not line:
        return False
    if line.startswith(("#", "//", "/*", "*", ";", "--")):
        return bool(_LICENSE_KEYWORDS.search(line))
    return False


def _is_import_line(line: str) -> bool:
    return bool(
        line.startswith("import ")
        or line.startswith("from ")
        or line.startswith("#include")
        or line.startswith("require(")
        or line.startswith("use ")
    )


def _is_comment_line(line: str) -> bool:
    if not line:
        return False
    return bool(
        line.startswith("#")
        or line.startswith("//")
        or line.startswith("/*")
        or line.startswith("*")
        or line.startswith('"""')
        or line.startswith("'''")
        or line.startswith(";")
    )


def _is_lockfile(path: str) -> bool:
    p = Path(path)
    return p.name.lower() in _LOCKFILE_NAMES or p.suffix.lower() in _LOCKFILE_EXTS


# ---------------------------------------------------------------------------
# Pipeline orchestrator (plan.md §5, Layer 0)
# ---------------------------------------------------------------------------


def compress(
    text: str,
    kind: str = "log",
    config: MicroCompressConfig | None = None,
    path: str | None = None,
    active_edit_files: set[str] | None = None,
) -> str:
    """Run the full micro-compression pipeline on *text*.

    Parameters
    ----------
    text
        Raw text to compress.
    kind
        Content-kind hint (``code | prose | data | log``).  Use
        :func:`infer_content_kind` to determine this.
    config
        Pipeline configuration.  Defaults to :class:`MicroCompressConfig`.
    path
        File path (for Stage 9 lockfile detection and active-edit guard).
    active_edit_files
        Set of paths currently being edited (Stage 9 skips these).

    Returns
    -------
    str
        Compressed text.  Always ≤ *text* in length (barring tiny marker
        overhead on very short inputs).
    """
    if not text or not isinstance(text, str):
        return text
    if config is None:
        config = MicroCompressConfig()
    if not config.enabled:
        return text

    # Stage 1 — lossless encoding normalisation
    text = normalize_encoding(text)

    # Stage 2 — lossless control-noise removal
    text = strip_control_noise(text)

    # Stage 5 — compact line numbers (lossless, before collapse so tabs survive)
    text = renumber_lines(text)

    # Stage 3 — whitespace collapse (lossless + annotated A3/A4)
    text = collapse_whitespace(text, kind, config)

    # Stage 4 — per-line prefix fold (annotated)
    text = fold_per_line_prefix(text, kind, config)

    # Stage 6 — boilerplate/banner drop (annotated)
    text = drop_boilerplate(text, kind, config)

    # Stage 7 — intra-line repetition dedup (annotated)
    text = intra_line_dedup(text, kind, config)

    # Stage 8 — near-duplicate line collapse (annotated)
    text = near_duplicate_collapse(text, kind, config)

    # Stage 9 — opt-in code content elision
    if config.read_compact_code:
        text = elide_low_value_content(
            text, kind, config, path, active_edit_files
        )

    return text


# ---------------------------------------------------------------------------
# Line-list convenience wrapper
# ---------------------------------------------------------------------------


def compress_lines(
    lines: list[str],
    kind: str = "log",
    config: MicroCompressConfig | None = None,
    path: str | None = None,
) -> tuple[list[str], int]:
    """Run :func:`compress` on a list of lines.

    Joins *lines* with ``\n``, runs the full pipeline, splits back into
    lines.  Returns ``(compressed_lines, saved_bytes)`` where
    *saved_bytes* is the character reduction achieved.

    This is the primary integration point for tools (Grep, Glob) that work
    with ``list[str]`` rather than a single text blob.
    """
    if not lines:
        return lines, 0
    original = "\n".join(lines)
    compressed = compress(original, kind=kind, config=config, path=path)
    saved = len(original) - len(compressed)
    if compressed:
        new_lines = compressed.split("\n")
    else:
        new_lines = []
    return new_lines, max(saved, 0)
