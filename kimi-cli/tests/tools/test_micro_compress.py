"""Tests for kimi_cli.tools.file.micro_compress (smart token reduction).

Covers every stage from plan.md §6, the ``infer_content_kind`` inference,
content-kind gating, marker correctness, idempotency and lossless roundtrip
properties, and the end-to-end ``compress()`` orchestrator.
"""

from __future__ import annotations

import pytest

from kimi_cli.tools.file.micro_compress import (
    MicroCompressConfig,
    _MAX_INTRA_LINE_UNIT,
    _MAX_PREFIX_SCAN,
    _compress_repeating_unit,
    _factor_common_indent,
    _longest_common_prefix,
    collapse_whitespace,
    compress,
    compress_lines,
    drop_boilerplate,
    elide_low_value_content,
    fold_per_line_prefix,
    infer_content_kind,
    intra_line_dedup,
    near_duplicate_collapse,
    normalize_encoding,
    renumber_lines,
    strip_control_noise,
)


# ---------------------------------------------------------------------------
# infer_content_kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("foo.py", "code"),
        ("src/bar.ts", "code"),
        ("lib.rs", "code"),
        ("main.c", "code"),
        ("header.h", "code"),
        ("README.md", "prose"),
        ("notes.txt", "prose"),
        ("data.json", "data"),
        ("conf.yaml", "data"),
        ("Cargo.toml", "data"),
        ("app.log", "log"),
        ("build.out", "log"),
        ("unknown.xyz", "log"),  # unknown ext → conservative log
    ],
)
def test_infer_content_kind_by_extension(path: str, expected: str):
    assert infer_content_kind(path=path) == expected


def test_infer_content_kind_by_tool():
    assert infer_content_kind(tool="bash") == "log"
    assert infer_content_kind(tool="python") == "log"
    assert infer_content_kind(tool="fetch_url") == "prose"


def test_infer_content_kind_extension_overrides_tool():
    assert infer_content_kind(path="foo.py", tool="bash") == "code"


def test_infer_content_kind_no_info_defaults_log():
    assert infer_content_kind() == "log"


# ---------------------------------------------------------------------------
# Stage 1 — normalize_encoding
# ---------------------------------------------------------------------------


def test_normalize_encoding_crlf_to_lf():
    assert normalize_encoding("a\r\nb\r\n") == "a\nb\n"


def test_normalize_encoding_strips_bom_and_zero_width():
    text = "\ufeffhello\u200bworld"
    assert normalize_encoding(text) == "helloworld"


def test_normalize_encoding_nbsp_to_space():
    text = "hello\u00a0world"
    assert normalize_encoding(text) == "hello world"


def test_normalize_encoding_em_space_to_space():
    text = "a\u2003b"
    assert normalize_encoding(text) == "a b"


def test_normalize_encoding_strips_c0_controls():
    text = "a\x00b\x07c\x0bd"
    assert normalize_encoding(text) == "abcd"


def test_normalize_encoding_preserves_tab_and_newline():
    text = "a\tb\nc"
    assert normalize_encoding(text) == "a\tb\nc"


def test_normalize_encoding_nfc_normalization():
    # NFD form of 'é' (e + combining accent) → NFC ('é')
    nfd = "e\u0301"
    nfc = "\u00e9"
    assert normalize_encoding(nfd) == nfc


def test_normalize_encoding_idempotent():
    text = "\ufeffhello\u00a0\r\n\u200bworld\x00\n"
    once = normalize_encoding(text)
    twice = normalize_encoding(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Stage 2 — strip_control_noise
# ---------------------------------------------------------------------------


def test_strip_control_noise_ansi_color():
    text = "\x1b[31mred text\x1b[0m"
    assert strip_control_noise(text) == "red text"


def test_strip_control_noise_ansi_cursor():
    text = "\x1b[2J\x1b[Hhello"
    assert strip_control_noise(text) == "hello"


def test_strip_control_noise_osc_sequence():
    text = "\x1b]0;window title\x07hello"
    assert strip_control_noise(text) == "hello"


def test_strip_control_noise_cr_progress_bar():
    text = "a\rb\rc"
    assert strip_control_noise(text) == "c"


def test_strip_control_noise_cr_progress_bar_multiline():
    text = "downloading...\r==>\r===>\ndone"
    assert strip_control_noise(text) == "===>\ndone"


def test_strip_control_noise_no_cr_unchanged():
    text = "plain text\nline2"
    assert strip_control_noise(text) == text


def test_strip_control_noise_idempotent():
    text = "\x1b[32mfoo\x1b[0m\rbar\rbaz"
    once = strip_control_noise(text)
    twice = strip_control_noise(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Stage 3 — collapse_whitespace
# ---------------------------------------------------------------------------


def test_collapse_whitespace_trailing_ws_stripped():
    text = "line1   \nline2\t\n"
    result = collapse_whitespace(text)
    assert "   " not in result
    assert "\t\n" not in result


def test_collapse_whitespace_blank_runs_collapsed():
    text = "a\n\n\n\nb"
    result = collapse_whitespace(text, config=MicroCompressConfig(blank_line_collapse=1))
    assert result == "a\n\nb"


def test_collapse_whitespace_blank_runs_removed_when_zero():
    text = "a\n\n\n\nb"
    result = collapse_whitespace(text, config=MicroCompressConfig(blank_line_collapse=0))
    assert result == "a\nb"


def test_collapse_whitespace_internal_spaces_collapsed_prose():
    text = "a   b   c"
    result = collapse_whitespace(text, kind="prose")
    assert result == "a b c"


def test_collapse_whitespace_internal_spaces_preserved_code():
    text = "a   b   c"
    result = collapse_whitespace(text, kind="code")
    assert result == "a   b   c"


def test_collapse_whitespace_common_indent_factored_non_code():
    text = "    line1\n    line2\n    line3"
    result = collapse_whitespace(text, kind="log")
    assert "[common-indent:" in result
    assert "line1" in result


def test_collapse_whitespace_common_indent_not_factored_code():
    text = "    line1\n    line2\n    line3"
    result = collapse_whitespace(text, kind="code")
    assert "[common-indent:" not in result


def test_collapse_whitespace_common_indent_too_short_skipped():
    # Common indent of only 2 chars (< 4) → not factored
    text = "  a\n  b"
    result = collapse_whitespace(text, kind="log")
    assert "[common-indent:" not in result


def test_collapse_whitespace_single_line_no_common_indent():
    text = "only line"
    result = collapse_whitespace(text, kind="log")
    assert result == "only line"


def test_collapse_whitespace_idempotent():
    text = "line1   \n\n\n  line2\n\n\n"
    once = collapse_whitespace(text, kind="log")
    twice = collapse_whitespace(once, kind="log")
    assert once == twice


# ---------------------------------------------------------------------------
# Stage 5 — renumber_lines
# ---------------------------------------------------------------------------


def test_renumber_lines_fixed_width_compacted():
    text = "     1\tfirst\n     2\tsecond\n    10\ttenth"
    result = renumber_lines(text)
    assert result == "1\tfirst\n2\tsecond\n10\ttenth"


def test_renumber_lines_already_compact_unchanged():
    text = "1\tfirst\n2\tsecond"
    result = renumber_lines(text)
    assert result == text


def test_renumber_lines_non_numbered_unchanged():
    text = "first line\nsecond line"
    result = renumber_lines(text)
    assert result == text


def test_renumber_lines_partial_numbered_unchanged():
    text = "1\tnumbered\nnot numbered"
    result = renumber_lines(text)
    assert result == text  # not all lines numbered → skip


def test_renumber_lines_idempotent():
    text = "     1\ta\n     2\tb\n     3\tc"
    once = renumber_lines(text)
    twice = renumber_lines(once)
    assert once == twice


def test_renumber_lines_preserves_bijection():
    text = "     5\tfive\n    10\tten\n    15\tfifteen"
    result = renumber_lines(text)
    # numbers must still map to correct lines
    assert result == "5\tfive\n10\tten\n15\tfifteen"


# ---------------------------------------------------------------------------
# Stage 4 — fold_per_line_prefix
# ---------------------------------------------------------------------------


def test_fold_per_line_prefix_common_path():
    prefix = "D:\\project\\src\\"
    lines = [f"{prefix}file{i}.py:10:match" for i in range(25)]
    text = "\n".join(lines)
    result = fold_per_line_prefix(text, kind="log")
    assert '[prefix:' in result
    assert prefix not in result.split("\n", 1)[1]  # prefix removed from body


def test_fold_per_line_prefix_timestamp():
    lines = [
        f"2024-01-0{d} 10:00:00.123 INFO message {d}" for d in range(1, 8)
    ] * 4  # 28 lines
    text = "\n".join(lines)
    result = fold_per_line_prefix(text, kind="log")
    assert "[ts-prefix folded" in result


def test_fold_per_line_prefix_skipped_for_code():
    prefix = "        "  # indentation
    lines = [f"{prefix}code_line_{i}" for i in range(25)]
    text = "\n".join(lines)
    result = fold_per_line_prefix(text, kind="code")
    assert result == text


def test_fold_per_line_prefix_too_few_lines():
    lines = ["D:\\path\\a.py:1:x", "D:\\path\\b.py:2:y"]
    text = "\n".join(lines)
    result = fold_per_line_prefix(text, kind="log")
    assert result == text  # too few lines


def test_fold_per_line_prefix_no_common_prefix():
    # Each line starts with a different letter — no shared prefix
    import string
    lines = [f"{ch}_completely_different_line_{i}" for i, ch in enumerate(string.ascii_lowercase)]
    text = "\n".join(lines)
    result = fold_per_line_prefix(text, kind="log")
    assert result == text


def test_fold_per_line_prefix_idempotent():
    prefix = "C:\\workspace\\mod\\"
    lines = [f"{prefix}f{i}.py:1:x" for i in range(25)]
    text = "\n".join(lines)
    once = fold_per_line_prefix(text, kind="log")
    twice = fold_per_line_prefix(once, kind="log")
    assert once == twice


# ---------------------------------------------------------------------------
# Stage 6 — drop_boilerplate
# ---------------------------------------------------------------------------


def test_drop_boilerplate_leading_banners():
    text = "npm version 10.0.0\nnode version 20.0.0\n\nreal output line"
    result = drop_boilerplate(text, kind="log")
    assert "banner lines dropped" in result
    assert "real output line" in result
    assert "npm version" not in result


def test_drop_boilerplate_ascii_art():
    text = "  ___ ___ ___\n | _ | _ | _ |\n |_| |_| |_| |\n\nreal line"
    result = drop_boilerplate(text, kind="log")
    assert "banner lines dropped" in result


def test_drop_boilerplate_no_banner_unchanged():
    text = "line1\nline2\nline3"
    result = drop_boilerplate(text, kind="log")
    assert result == text


def test_drop_boilerplate_merge_system_metadata():
    text = (
        "<system>tool=X id=1</system>\n"
        "<system>tool=X id=1</system>\n"
        "<system>tool=X id=2</system>\n"
        "content"
    )
    result = drop_boilerplate(text, kind="log")
    # Only consecutive identical merges
    lines = result.split("\n")
    # The two identical should be merged into one
    meta_count = sum(1 for ln in lines if "<system>" in ln)
    assert meta_count == 2  # id=1 appears once, id=2 appears once


def test_drop_boilerplate_not_dropping_everything():
    text = "banner line\ncargo build\n"
    result = drop_boilerplate(text, kind="log")
    assert result != ""  # never fully empties


def test_drop_boilerplate_disabled():
    text = "npm version 10\nreal output"
    result = drop_boilerplate(text, kind="log", config=MicroCompressConfig(banner_drop=False))
    assert result == text


# ---------------------------------------------------------------------------
# Stage 7 — intra_line_dedup
# ---------------------------------------------------------------------------


def test_intra_line_dedup_repeating_unit():
    line = "AB" * 1500  # 3000 chars, unit "AB"
    result = intra_line_dedup(line, kind="log")
    assert "×1500" in result
    assert "+2998 chars elided" in result
    assert len(result) < len(line)


def test_intra_line_dedup_single_char_unit():
    line = "x" * 3000
    result = intra_line_dedup(line, kind="log")
    assert "×3000" in result


def test_intra_line_dedup_skipped_for_code():
    line = "AB" * 1500
    result = intra_line_dedup(line, kind="code")
    assert result == line


def test_intra_line_dedup_short_line_unchanged():
    line = "AB" * 10
    result = intra_line_dedup(line, kind="log")
    assert result == line


def test_intra_line_dedup_no_repetition_unchanged():
    line = "a" * 3000 + "b" * 100  # not a clean repetition
    result = intra_line_dedup(line, kind="log")
    assert result == line


def test_intra_line_dedup_multiline():
    long_line = "XY" * 2000
    text = f"normal line\n{long_line}\nanother line"
    result = intra_line_dedup(text, kind="log")
    assert "×2000" in result
    assert "normal line" in result


def test_intra_line_dedup_idempotent():
    line = "AB" * 1500
    once = intra_line_dedup(line, kind="log")
    twice = intra_line_dedup(once, kind="log")
    assert once == twice


# ---------------------------------------------------------------------------
# Stage 8 — near_duplicate_collapse
# ---------------------------------------------------------------------------


def test_near_duplicate_collapse_counter_run():
    lines = [f"Processing item {i} of 100" for i in range(5, 15)]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert "[×" in result
    assert "near-dup" in result


def test_near_duplicate_collapse_with_field_marker():
    lines = [f"Request 5 returned status 200 in {ms}ms" for ms in [10, 11, 12, 13, 14]]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert "field" in result  # should report the changed numeric field


def test_near_duplicate_collapse_too_short_run():
    lines = ["line A", "line A", "line A"]  # only 3, below default 4
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text


def test_near_duplicate_collapse_skipped_for_code():
    lines = [f"x = {i}" for i in range(10)]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="code")
    assert result == text


def test_near_duplicate_collapse_distinct_lines_unchanged():
    lines = ["completely different line one", "another unique line here"]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text


def test_near_duplicate_collapse_idempotent():
    lines = [f"Processing item {i} of 100" for i in range(5, 15)]
    text = "\n".join(lines)
    once = near_duplicate_collapse(text, kind="log")
    twice = near_duplicate_collapse(once, kind="log")
    assert once == twice


def test_near_duplicate_collapse_preserves_distinct_identifier_rows():
    """Distinct identifier rows must survive: the ``a→b`` marker would be a
    misleading range when the intermediate values are absent, and the model
    needs the individual values (e.g. duplicate P-numbers during a merge)."""
    lines = [
        "P-022: 2 occurrences",
        "P-053: 2 occurrences",
        "P-055: 2 occurrences",
        "P-056: 2 occurrences",
        "P-057: 2 occurrences",
    ]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text
    assert "near-dup" not in result


def test_near_duplicate_collapse_preserves_non_contiguous_numeric_ids():
    """Numeric fields that skip values are identifiers, not counters."""
    lines = [f"ID {i} active" for i in [5, 8, 11, 14, 17]]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text


def test_near_duplicate_collapse_counter_decreasing():
    """A monotonic decreasing counter is still a safe range collapse."""
    lines = [f"Countdown {i} seconds" for i in range(10, 2, -1)]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert "near-dup" in result


def test_near_duplicate_collapse_preserves_multiple_varying_fields():
    """Runs where more than one numeric field varies are not a simple counter
    (each line is a distinct entry) and must be preserved."""
    lines = [
        "row 1 of 10",
        "row 2 of 20",
        "row 3 of 30",
        "row 4 of 40",
        "row 5 of 50",
    ]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text


def test_near_duplicate_collapse_preserves_non_numeric_variation():
    """Near-duplicate lines that differ in a non-numeric token (e.g. file
    names) are distinct entries, not counters."""
    lines = [
        "changed: foo.py",
        "changed: bar.py",
        "changed: baz.py",
        "changed: qux.py",
        "changed: quux.py",
    ]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert result == text


# ---------------------------------------------------------------------------
# Stage 9 — elide_low_value_content (opt-in)
# ---------------------------------------------------------------------------


def test_elide_low_value_content_default_off():
    text = "# Copyright 2024\n# Licensed MIT\nimport os\nimport sys\nx = 1"
    result = elide_low_value_content(text, kind="code")
    assert result == text  # default off


def test_elide_low_value_content_license_header():
    lines = ["# Copyright 2024 Example Corp", "# Licensed under MIT License"] + ["code"] * 10
    text = "\n".join(lines)
    result = elide_low_value_content(
        text, kind="code", config=MicroCompressConfig(read_compact_code=True)
    )
    assert "license lines elided" in result


def test_elide_low_value_content_import_cluster():
    lines = [
        "import os",
        "import sys",
        "import json",
        "import re",
        "x = 1",
    ]
    text = "\n".join(lines)
    result = elide_low_value_content(
        text, kind="code", config=MicroCompressConfig(read_compact_code=True)
    )
    assert "[4 imports]" in result


def test_elide_low_value_content_comment_block():
    lines = ["# comment one", "# comment two", "# comment three", "x = 1"]
    text = "\n".join(lines)
    result = elide_low_value_content(
        text, kind="code", config=MicroCompressConfig(read_compact_code=True)
    )
    assert "comment lines elided" in result


def test_elide_low_value_content_lockfile():
    text = "\n".join([f'{{"pkg{i}": "1.0.0"}}' for i in range(100)])
    result = elide_low_value_content(
        text,
        kind="code",
        config=MicroCompressConfig(read_compact_code=True),
        path="package-lock.json",
    )
    assert "lines of generated content" in result
    assert "hash=xxh64:" in result


def test_elide_low_value_content_active_edit_skipped():
    text = "# Copyright 2024\n# License MIT\ncode line"
    result = elide_low_value_content(
        text,
        kind="code",
        config=MicroCompressConfig(read_compact_code=True),
        path="active.py",
        active_edit_files={"active.py"},
    )
    assert result == text  # active edit file skipped


def test_elide_low_value_content_skipped_for_non_code():
    text = "# Copyright\n# License\ntext"
    result = elide_low_value_content(
        text, kind="log", config=MicroCompressConfig(read_compact_code=True)
    )
    assert result == text


def test_elide_low_value_content_inlines_kept():
    # Fewer than threshold imports are kept
    text = "import os\nimport sys\ncode"
    result = elide_low_value_content(
        text, kind="code", config=MicroCompressConfig(read_compact_code=True)
    )
    assert result == text


# ---------------------------------------------------------------------------
# compress() — end-to-end pipeline
# ---------------------------------------------------------------------------


def test_compress_disabled_returns_input():
    text = "  hello  \r\n\n\n  world  "
    result = compress(text, config=MicroCompressConfig(enabled=False))
    assert result == text


def test_compress_empty_string():
    assert compress("") == ""


def test_compress_runs_all_lossless_stages():
    text = "\ufeffhello\x1b[31m\r\n\n\n  world  \r"
    result = compress(text, kind="log")
    # BOM removed
    assert "\ufeff" not in result
    # ANSI removed
    assert "\x1b" not in result
    # CRLF → LF
    assert "\r\n" not in result
    # Trailing whitespace stripped
    assert "  \n" not in result


def test_compress_code_kind_skips_destructive_stages():
    # Code should not get prefix folding or near-dup collapse
    lines = [f"        code_line_{i}" for i in range(25)]
    text = "\n".join(lines)
    result = compress(text, kind="code")
    assert "[prefix:" not in result
    assert "[×" not in result


def test_compress_log_kind_applies_destructive_stages():
    prefix = "C:\\workspace\\src\\"
    lines = [f"{prefix}f{i}.py:1:match" for i in range(25)]
    text = "\n".join(lines)
    result = compress(text, kind="log")
    assert "[prefix:" in result


def test_compress_renumber_when_numbered():
    text = "     1\tline one\n     2\tline two"
    result = compress(text, kind="code")
    assert "1\tline one" in result
    assert "     1\t" not in result


def test_compress_lossless_only_mode():
    config = MicroCompressConfig(lossless_only=True)
    prefix = "C:\\workspace\\src\\"
    lines = [f"{prefix}f{i}.py:1:m" for i in range(25)]
    text = "\n".join(lines)
    result = compress(text, kind="log", config=config)
    # No prefix folding in lossless-only mode
    assert "[prefix:" not in result


def test_compress_idempotent_log():
    text = "\ufefflog line\r\n\n\n  trailing  \n\n\nmore"
    once = compress(text, kind="log")
    twice = compress(once, kind="log")
    assert once == twice


def test_compress_idempotent_code():
    text = "     1\tdef foo():\n     2\t    return 1"
    once = compress(text, kind="code")
    twice = compress(once, kind="code")
    assert once == twice


def test_compress_idempotent_prose():
    text = "Hello   world\r\n\r\n\r\nThis is   prose."
    once = compress(text, kind="prose")
    twice = compress(once, kind="prose")
    assert once == twice


def test_compress_reduces_size_log_output():
    # Typical noisy log with blanks, trailing ws, ANSI
    lines = []
    for i in range(100):
        lines.append(f"\x1b[32m2024-01-01 10:00:{i:02d}.000 INFO processing   \r\n")
    text = "".join(lines)
    result = compress(text, kind="log")
    assert len(result) < len(text)


def test_compress_reduces_size_readfile_numbering():
    lines = [f"{n:6d}\tcode line {n}" for n in range(1, 1001)]
    text = "\n".join(lines)
    result = compress(text, kind="code")
    assert len(result) < len(text)


# ---------------------------------------------------------------------------
# Property tests — idempotency across multiple inputs
# ---------------------------------------------------------------------------


_IDEMPOTENCY_INPUTS = [
    "",  # empty
    "simple text",  # single line
    "line1\nline2\nline3",  # multi-line
    "     1\tnum\n     2\tnum",  # numbered
    "\ufeff\x1b[31mcolored\r\n\r\n\r\n  ws  ",  # noisy
    "A" * 3000,  # long repeating
]


@pytest.mark.parametrize("text", _IDEMPOTENCY_INPUTS)
@pytest.mark.parametrize("kind", ["code", "log", "prose", "data"])
def test_compress_idempotent_parametrized(text: str, kind: str):
    once = compress(text, kind=kind)
    twice = compress(once, kind=kind)
    assert once == twice


# ---------------------------------------------------------------------------
# Lossless roundtrip for stages 1, 2, 3 (A1/A2/A5), 5
# ---------------------------------------------------------------------------


def test_lossless_stages_preserve_content_semantics():
    """Lossless stages only remove noise, never real content."""
    original = (
        "def hello():\n"
        "    print('hi')\n"
        "    return None\n"
    )
    # Through encoding + control + whitespace (without A3/A4 for code)
    text = normalize_encoding(original)
    text = strip_control_noise(text)
    text = collapse_whitespace(text, kind="code")
    text = renumber_lines(text)
    # No content lost
    assert "def hello():" in text
    assert "print('hi')" in text
    assert "return None" in text


def test_lossless_stages_preserve_log_content():
    original = "2024-01-01 INFO starting up\n2024-01-01 INFO running\n"
    text = normalize_encoding(original)
    text = strip_control_noise(text)
    text = collapse_whitespace(text, kind="log")
    assert "starting up" in text
    assert "running" in text


# ---------------------------------------------------------------------------
# Marker correctness
# ---------------------------------------------------------------------------


def test_marker_intra_line_reports_correct_count():
    line = "AB" * 2000  # 4000 chars
    result = intra_line_dedup(line, kind="log")
    assert "×2000" in result
    assert "+3998 chars elided" in result  # 4000 - 2


def test_marker_banner_reports_correct_count():
    text = "npm 10.0.0\nnode v20.0.0\nyarn 1.22.0\nStarting build process"
    result = drop_boilerplate(text, kind="log")
    assert "[3 banner lines dropped]" in result


def test_marker_near_dup_reports_correct_run():
    # Use long lines so single-field change gives high fuzz.ratio
    lines = [f"Processing request batch with id={i} completed successfully" for i in range(10, 20)]
    text = "\n".join(lines)
    result = near_duplicate_collapse(text, kind="log")
    assert "[×9 near-dup" in result


def test_marker_common_indent_reports_cols():
    text = "      line1\n      line2\n      line3"
    result = collapse_whitespace(text, kind="log")
    assert "[common-indent: 6 cols removed]" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compress_none_safe():
    assert compress(None) is None  # type: ignore[arg-type]


def test_compress_single_char():
    result = compress("x", kind="code")
    assert result == "x"


def test_compress_only_whitespace():
    result = compress("   \n\n\n   ", kind="log")
    assert result.strip() == ""


def test_compress_preserves_newline_structure_code():
    text = "def foo():\n    x = 1\n    return x"
    result = compress(text, kind="code")
    # Code should preserve line structure
    assert result.count("\n") == 2


def test_config_defaults_sensible():
    cfg = MicroCompressConfig()
    assert cfg.enabled is True
    assert cfg.lossless_only is False
    assert cfg.read_compact_code is False
    assert cfg.blank_line_collapse == 1
    assert cfg.strip_trailing_ws is True


# ---------------------------------------------------------------------------
# Regression: giant first line must not trigger O(n²) prefix/indent scans
# ---------------------------------------------------------------------------
# A 3MB single-line data file (e.g. models_dev_snapshot.json) matched by Grep
# used to make _factor_common_indent / _longest_common_prefix chop the prefix
# one character at a time (``prefix[:-1]``), stalling the tool for minutes.
# These tests pin the linear, capped behavior.


def test_longest_common_prefix_giant_first_line_is_linear():
    giant = "C:/big.json:1:" + "x" * 500_000
    small = "C:/big.json:2:y"
    # Common prefix is the short path prefix shared by both lines.
    assert _longest_common_prefix([giant, small]) == "C:/big.json:"


def test_longest_common_prefix_capped_at_max_scan():
    # All lines share more than _MAX_PREFIX_SCAN chars; the helper returns a
    # valid (shorter) common prefix instead of scanning the whole string.
    shared = "p" * (_MAX_PREFIX_SCAN + 10_000)
    lines = [shared + "A", shared + "B"]
    result = _longest_common_prefix(lines)
    assert result == shared[:_MAX_PREFIX_SCAN]
    assert all(ln.startswith(result) for ln in lines)


def test_longest_common_prefix_empty_and_single():
    assert _longest_common_prefix([]) == ""
    assert _longest_common_prefix(["abc"]) == "abc"


def test_factor_common_indent_giant_first_line_no_bogus_indent():
    giant = "C:/big.json:1:" + "x" * 500_000
    small = "C:/big.json:2:y"
    lines = [giant, small]
    result = _factor_common_indent(lines, lines)
    # No shared whitespace indent → unchanged, and must not hang.
    assert result == lines


def test_factor_common_indent_real_indent_still_factored():
    giant = "    " + "x" * 500_000
    small = "    y"
    lines = [giant, small]
    result = _factor_common_indent(lines, lines)
    assert result[0].startswith("[common-indent:")
    # The 4-space indent is removed from both lines (giant content first, then
    # the small line).
    assert result[1] == "x" * 500_000
    assert result[2] == "y"


def test_collapse_whitespace_giant_first_line_fast():
    giant = "C:/big.json:1:" + "x" * 500_000
    small = "C:/big.json:2:y"
    text = "\n".join([giant, small])
    result = collapse_whitespace(text, kind="log")
    assert "[common-indent:" not in result
    assert len(result.split("\n")) == 2


def test_fold_per_line_prefix_giant_first_line_still_folds_short_prefix():
    prefix = "C:/dev/Hermes-CN-Core/"
    giant = prefix + "agent/big.json:1:" + "x" * 500_000
    smalls = [f"{prefix}tools/f{i}.py:1:x" for i in range(30)]
    text = "\n".join([giant] + smalls)
    result = fold_per_line_prefix(text, kind="log")
    assert '[prefix: "' in result
    # Body no longer carries the folded prefix on every line.
    body = result.split("\n", 1)[1]
    assert not body.startswith(prefix)


def test_compress_lines_giant_line_completes():
    giant = "C:/big.json:1:" + "x" * 300_000
    lines = [giant] + ["C:/big.json:2:y"] * 25
    out, saved = compress_lines(
        lines,
        kind="log",
        config=MicroCompressConfig(lossless_only=False, near_dup_collapse=False),
    )
    assert isinstance(out, list)
    assert saved >= 0
    assert any(ln for ln in out)


def test_compress_repeating_unit_capped_for_huge_lines():
    # Huge non-repeating line: unit scan must stop at _MAX_INTRA_LINE_UNIT, so
    # this returns unchanged without building O(n) strings for every divisor.
    line = "a" * 200_000 + "b" * 100
    assert _compress_repeating_unit(line) == line


def test_compress_repeating_unit_detects_unit_at_cap_boundary():
    # 2048-char non-periodic unit (sequential ids) so only the full unit
    # matches, not a smaller divisor of the total length.
    unit = "".join(f"{i:04d}" for i in range(512))
    assert len(unit) == _MAX_INTRA_LINE_UNIT
    line = unit * 3
    result = _compress_repeating_unit(line)
    assert "×3" in result
    assert "+4096 chars elided" in result


def test_compress_repeating_unit_detects_short_unit_on_huge_line():
    line = "AB" * 100_000
    result = _compress_repeating_unit(line)
    assert "×100000" in result
