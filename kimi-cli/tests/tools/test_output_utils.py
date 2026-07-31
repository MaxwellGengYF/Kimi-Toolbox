"""Tests for kimi_cli.tools.file.output_utils (pure output-shaping helpers)."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file.output_utils import (
    DEFAULT_MAX_LINES,
    dedup_lines,
    fold_lines,
    parse_rtk_rg_output,
    truncate_line,
)


# ---------------------------------------------------------------------------
# fold_lines
# ---------------------------------------------------------------------------


def test_fold_lines_under_budget_unchanged():
    lines = [f"line{i}" for i in range(10)]
    folded, omitted = fold_lines(lines, max_lines=20)
    assert folded == lines
    assert omitted == 0


def test_fold_lines_exactly_at_budget_unchanged():
    lines = [f"line{i}" for i in range(10)]
    folded, omitted = fold_lines(lines, max_lines=10)
    assert folded == lines
    assert omitted == 0


def test_fold_lines_over_budget_head_tail_marker():
    lines = [f"line{i}" for i in range(10)]
    folded, omitted = fold_lines(lines, max_lines=4)
    # head = 4 // 2 = 2, tail = 2 → marker in the middle
    assert omitted == 6
    assert folded == [
        "line0",
        "line1",
        "… (6 lines omitted) …",
        "line8",
        "line9",
    ]


def test_fold_lines_head_tail_order_preserved():
    lines = [f"line{i}" for i in range(20)]
    folded, omitted = fold_lines(lines, max_lines=6)
    # head = 6 // 2 = 3, tail = 3 → head first, marker, tail last
    assert omitted == 14
    assert folded == [
        "line0",
        "line1",
        "line2",
        "… (14 lines omitted) …",
        "line17",
        "line18",
        "line19",
    ]


def test_fold_lines_default_max_lines():
    lines = [f"line{i}" for i in range(DEFAULT_MAX_LINES + 50)]
    folded, omitted = fold_lines(lines)
    assert omitted == 50
    assert len(folded) == DEFAULT_MAX_LINES + 1  # + marker line
    assert "… (50 lines omitted) …" in folded


def test_fold_lines_odd_max_lines():
    lines = [f"line{i}" for i in range(9)]
    folded, omitted = fold_lines(lines, max_lines=5)
    # head = 5 // 2 = 2, tail = 3
    assert omitted == 4
    assert folded == [
        "line0",
        "line1",
        "… (4 lines omitted) …",
        "line6",
        "line7",
        "line8",
    ]


def test_fold_lines_custom_head_tail():
    lines = [f"line{i}" for i in range(10)]
    folded, omitted = fold_lines(lines, max_lines=5, head=4, tail=1)
    assert omitted == 5
    assert folded == [
        "line0",
        "line1",
        "line2",
        "line3",
        "… (5 lines omitted) …",
        "line9",
    ]


def test_fold_lines_max_lines_zero_unlimited():
    lines = [f"line{i}" for i in range(500)]
    folded, omitted = fold_lines(lines, max_lines=0)
    assert folded == lines
    assert omitted == 0


def test_fold_lines_negative_max_lines_unlimited():
    lines = [f"line{i}" for i in range(10)]
    folded, omitted = fold_lines(lines, max_lines=-5)
    assert folded == lines
    assert omitted == 0


def test_fold_lines_preserves_blank_lines():
    lines = ["a", "", "   ", "b", "c", "d", "e"]
    folded, omitted = fold_lines(lines, max_lines=4)
    assert omitted == 3
    assert folded == ["a", "", "… (3 lines omitted) …", "d", "e"]


def test_fold_lines_empty_input():
    folded, omitted = fold_lines([], max_lines=10)
    assert folded == []
    assert omitted == 0


# ---------------------------------------------------------------------------
# dedup_lines
# ---------------------------------------------------------------------------


def test_dedup_lines_no_repeats_unchanged():
    lines = ["a", "b", "c", "d"]
    out, saved = dedup_lines(lines)
    assert out == lines
    assert saved == 0


def test_dedup_lines_run_collapse():
    lines = ["a", "a", "a", "b", "b", "b"]
    out, saved = dedup_lines(lines)
    assert out == ["a  (2 repeats)", "b  (2 repeats)"]
    assert saved == 4


def test_dedup_lines_below_threshold_unchanged():
    lines = ["a", "a", "b"]
    out, saved = dedup_lines(lines)
    assert out == lines
    assert saved == 0


def test_dedup_lines_threshold_custom():
    lines = ["a", "a", "a", "a", "b", "b"]
    out, saved = dedup_lines(lines, min_repeats=4)
    assert out == ["a  (3 repeats)", "b", "b"]
    assert saved == 3


def test_dedup_lines_non_consecutive_not_merged():
    lines = ["a", "a", "a", "b", "a", "a", "a"]
    out, saved = dedup_lines(lines)
    # Two separate runs of three "a" collapse independently; "b" stays.
    assert out == ["a  (2 repeats)", "b", "a  (2 repeats)"]
    assert saved == 4


def test_dedup_lines_empty_and_single():
    assert dedup_lines([]) == ([], 0)
    assert dedup_lines(["only"]) == (["only"], 0)


def test_dedup_lines_2000_repeats():
    lines = ["match line"] * 2000
    out, saved = dedup_lines(lines)
    assert out == ["match line  (1999 repeats)"]
    assert saved == 1999


def test_dedup_lines_mixed_run_and_normal():
    lines = ["x", "x", "x", "y", "z", "z", "z", "z"]
    out, saved = dedup_lines(lines)
    assert out == ["x  (2 repeats)", "y", "z  (3 repeats)"]
    assert saved == 2 + 3


def test_dedup_lines_min_repeats_one_merges_all():
    lines = ["a", "a", "b"]
    out, saved = dedup_lines(lines, min_repeats=1)
    assert out == ["a  (1 repeats)", "b"]
    assert saved == 1


# ---------------------------------------------------------------------------
# truncate_line
# ---------------------------------------------------------------------------


def test_truncate_line_short_unchanged():
    line = "short line"
    assert truncate_line(line, max_len=100) == line


def test_truncate_line_exact_length_unchanged():
    line = "a" * 50
    assert truncate_line(line, max_len=50) == line


def test_truncate_line_long_with_marker():
    line = "a" * 100
    out = truncate_line(line, max_len=50)
    assert len(out) <= 50
    assert out.startswith("a" * 10)
    assert "… [+50 chars]" in out


def test_truncate_line_marker_reports_removed_chars():
    line = "hello world this is long"  # 24 chars
    out = truncate_line(line, max_len=20)
    assert len(out) <= 20
    assert "… [+4 chars]" in out
    assert out.startswith("hello wo")


def test_truncate_line_tiny_max_len_no_marker():
    line = "a" * 100
    out = truncate_line(line, max_len=3)
    assert out == "aaa"


def test_truncate_line_default_max_len():
    line = "x" * (500 + 100)
    out = truncate_line(line)
    assert len(out) <= 500
    assert "… [+100 chars]" in out


def test_truncate_line_unicode_length():
    line = "é" * 100  # 2 bytes per char but 1 char each
    out = truncate_line(line, max_len=50)
    assert len(out) <= 50
    assert "… [+50 chars]" in out


# ---------------------------------------------------------------------------
# parse_rtk_rg_output
# ---------------------------------------------------------------------------


def test_parse_rtk_header_and_blank_removed():
    lines = [
        "42 matches in 3 files:",
        "",
        "src/a.py:10:match one",
        "src/a.py:12:match two",
        "src/b.py:3:match three",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == [
        "src/a.py:10:match one",
        "src/a.py:12:match two",
        "src/b.py:3:match three",
    ]
    assert meta["total_matches"] == 42
    assert meta["total_files"] == 3


def test_parse_rtk_per_file_fold_removed_and_metadata():
    lines = [
        "src/a.py:10:keep",
        "  +37 more in C:\\dev\\proj\\src\\a.py [see remaining: tail -n +26 C:\\log\\tee.log]",
        "src/b.py:5:keep",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == ["src/a.py:10:keep", "src/b.py:5:keep"]
    assert meta["folded_files"] == [
        {
            "path": "C:\\dev\\proj\\src\\a.py",
            "count": 37,
            "log": "C:\\log\\tee.log",
            "start_line": 26,
        }
    ]


def test_parse_rtk_files_fold_removed():
    lines = [
        "src/a.py:1:keep",
        "+133 more files [see remaining: tail -n +300 C:\\log\\tee2.log]",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == ["src/a.py:1:keep"]
    assert meta["skipped_files"] == 133
    assert meta["skipped_log"] == "C:\\log\\tee2.log"


def test_parse_rtk_full_stream():
    lines = [
        "46 matches in 1 files:",
        "",
        "src/common.py:25:def foo():",
        "  +45 more in src\\common.py [see remaining: tail -n +27 ~/AppData/Local/rtk/tee/123_grep.log]",
        "+9 more files [see remaining: tail -n +31 ~/AppData/Local/rtk/tee/123_grep.log]",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == ["src/common.py:25:def foo():"]
    assert meta["total_matches"] == 46
    assert meta["total_files"] == 1
    assert meta["skipped_files"] == 9
    assert meta["folded_files"][0]["count"] == 45
    assert meta["folded_files"][0]["start_line"] == 27


def test_parse_rtk_plain_rg_passthrough():
    lines = [
        "src/a.py:10:some content",
        "src/b.py:20:more content",
        "src/c.py:30:even more",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == lines
    assert meta == {
        "total_matches": None,
        "total_files": None,
        "folded_files": [],
        "skipped_files": None,
        "skipped_log": None,
    }


def test_parse_rtk_no_false_positive_on_real_paths_with_colons():
    """Windows drive-letter paths must not be mistaken for rtk headers."""
    lines = [
        "C:\\dev\\proj\\src\\a.py:10:1 matches in 0 files: not a header",
        "src/b.py:3:12 matches in 9 files: content",
        "src/c.py:7:+3 more in src\\d.py [see remaining: not a fold]",
    ]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == lines
    assert meta["total_matches"] is None


def test_parse_rtk_blank_separator_without_header_kept():
    lines = ["src/a.py:1:one", "", "src/b.py:2:two"]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == lines
    assert meta["total_matches"] is None


def test_parse_rtk_header_not_first_line_kept_others():
    """A mid-stream header-like line is still parsed (rtk emits it first, but
    be tolerant) and the blank after it is consumed."""
    lines = ["src/a.py:1:one", "12 matches in 2 files:", "", "src/b.py:2:two"]
    cleaned, meta = parse_rtk_rg_output(lines)
    assert cleaned == ["src/a.py:1:one", "src/b.py:2:two"]
    assert meta["total_matches"] == 12


def test_parse_rtk_empty_input():
    cleaned, meta = parse_rtk_rg_output([])
    assert cleaned == []
    assert meta["total_matches"] is None
