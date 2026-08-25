"""Tests for grep_selectors.py (plans/23-grep-rich.md §7)."""

from __future__ import annotations

import os

import pytest

from kimi_cli.tools.file.grep_selectors import (
    GrepPathSpec,
    LineRange,
    expand_path_entries,
    is_line_in_ranges,
    merge_ranges_into,
    parse_line_range_chunk,
    parse_line_ranges,
    selector_line_ranges,
    split_path_and_sel,
)


class TestParseLineRangeChunk:
    def test_simple_range(self):
        assert parse_line_range_chunk("50-100") == LineRange(50, 100)

    def test_plus_count(self):
        # 50+10 = 10 lines starting at 50 -> 50..59
        assert parse_line_range_chunk("50+10") == LineRange(50, 59)

    def test_open_ended_dash(self):
        assert parse_line_range_chunk("301-") == LineRange(301, None)

    def test_bare_number_open_ended(self):
        assert parse_line_range_chunk("301") == LineRange(301, None)

    def test_single_line_range(self):
        assert parse_line_range_chunk("1-1") == LineRange(1, 1)

    def test_L_prefix(self):
        assert parse_line_range_chunk("L42") == LineRange(42, None)
        assert parse_line_range_chunk("L42-L50") == LineRange(42, 50)

    def test_dotdot_alias(self):
        assert parse_line_range_chunk("42..100") == LineRange(42, 100)
        assert parse_line_range_chunk("42..") == LineRange(42, None)

    def test_case_insensitive(self):
        assert parse_line_range_chunk("l42-L50") == LineRange(42, 50)

    def test_whitespace_tolerated(self):
        assert parse_line_range_chunk(" 50-100 ") == LineRange(50, 100)

    def test_zero_start_invalid(self):
        with pytest.raises(ValueError, match="1-indexed"):
            parse_line_range_chunk("0-5")

    def test_inverted_range_invalid(self):
        with pytest.raises(ValueError, match="end must be >= start"):
            parse_line_range_chunk("50-40")

    def test_zero_count_invalid(self):
        with pytest.raises(ValueError, match="count must be >= 1"):
            parse_line_range_chunk("50+0")

    def test_garbage_returns_none(self):
        assert parse_line_range_chunk("garbage") is None
        assert parse_line_range_chunk("") is None

    def test_plus_open_ended(self):
        # "N+" behaves like the open-ended "N-".
        assert parse_line_range_chunk("50+") == LineRange(50, None)


class TestParseLineRanges:
    def test_comma_list(self):
        ranges = parse_line_ranges("5-16,960-973")
        assert ranges == [LineRange(5, 16), LineRange(960, 973)]

    def test_merge_overlapping(self):
        assert parse_line_ranges("1-3,3-5") == [LineRange(1, 5)]

    def test_merge_adjacent(self):
        assert parse_line_ranges("1-3,4-6") == [LineRange(1, 6)]

    def test_open_ended_absorbs(self):
        assert parse_line_ranges("10-,20-30") == [LineRange(10, None)]

    def test_sorts_by_start(self):
        assert parse_line_ranges("20-30,1-5") == [LineRange(1, 5), LineRange(20, 30)]

    def test_no_selector_returns_none(self):
        assert parse_line_ranges("raw") is None
        assert parse_line_ranges("") is None

    def test_invalid_semantics_raises(self):
        with pytest.raises(ValueError):
            parse_line_ranges("50-40")

    def test_whitespace_chunks(self):
        assert parse_line_ranges(" 1-3 , 5-6 ") == [LineRange(1, 3), LineRange(5, 6)]


class TestIsLineInRanges:
    def test_none_unfiltered(self):
        assert is_line_in_ranges(12345, None) is True

    def test_in_range(self):
        ranges = [LineRange(5, 16), LineRange(960, 973)]
        assert is_line_in_ranges(5, ranges) is True
        assert is_line_in_ranges(16, ranges) is True
        assert is_line_in_ranges(965, ranges) is True

    def test_out_of_range(self):
        ranges = [LineRange(5, 16), LineRange(960, 973)]
        assert is_line_in_ranges(4, ranges) is False
        assert is_line_in_ranges(17, ranges) is False
        assert is_line_in_ranges(959, ranges) is False

    def test_open_ended(self):
        assert is_line_in_ranges(10_000_000, [LineRange(301, None)]) is True
        assert is_line_in_ranges(300, [LineRange(301, None)]) is False


class TestSelectorLineRanges:
    def test_none_sel(self):
        assert selector_line_ranges(None) is None
        assert selector_line_ranges("") is None

    def test_raw_unfiltered(self):
        assert selector_line_ranges("raw") is None

    def test_conflicts_unfiltered(self):
        assert selector_line_ranges("conflicts") is None

    def test_raw_with_range_filters(self):
        assert selector_line_ranges("raw:50-100") == [LineRange(50, 100)]
        assert selector_line_ranges("50-100:raw") == [LineRange(50, 100)]

    def test_plain_range(self):
        assert selector_line_ranges("1-5") == [LineRange(1, 5)]

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            selector_line_ranges("50-40")


class TestSplitPathAndSel:
    def test_no_colon(self):
        assert split_path_and_sel("src/app.py") == ("src/app.py", None)

    def test_simple_selector(self):
        assert split_path_and_sel("src/app.py:50-100") == ("src/app.py", "50-100")

    def test_multi_range(self):
        assert split_path_and_sel("a/b.py:5-16,960-973") == (
            "a/b.py",
            "5-16,960-973",
        )

    def test_compound_selector(self):
        assert split_path_and_sel("a/b.py:1-50:raw") == ("a/b.py", "1-50:raw")
        assert split_path_and_sel("a/b.py:raw:1-50") == ("a/b.py", "raw:1-50")

    def test_non_selector_tail_not_peeled(self):
        # archive member / plain path segment: not a selector shape.
        assert split_path_and_sel("bundle.zip:src/foo.ts") == (
            "bundle.zip:src/foo.ts",
            None,
        )

    def test_windows_drive_guard(self):
        # Windows absolute path with selector.
        path, sel = split_path_and_sel(r"C:\dir\f.txt:50-100")
        assert path == r"C:\dir\f.txt"
        assert sel == "50-100"

    def test_bare_drive_not_peeled(self):
        # A bare drive letter must never be left behind.
        path, sel = split_path_and_sel("C:")
        assert (path, sel) == ("C:", None)

    def test_ssh_port_guard(self):
        assert split_path_and_sel("ssh://h:2222") == ("ssh://h:2222", None)

    def test_ssh_with_path_peels(self):
        assert split_path_and_sel("ssh://h/f:1-5") == ("ssh://h/f", "1-5")

    def test_literal_file_preference(self, tmp_path):
        # A real file named "test:1-2" outranks the selector interpretation.
        # On POSIX the filename "test:1-2" is valid.
        literal = tmp_path / "test:1-2"
        try:
            literal.write_text("x")
        except OSError:
            pytest.skip("filesystem does not allow ':' in filenames")
        raw = str(literal)
        assert split_path_and_sel(raw) == (raw, None)

    def test_garbage_tail_not_peeled(self):
        assert split_path_and_sel("src/foo.py:hello world") == (
            "src/foo.py:hello world",
            None,
        )

    def test_empty(self):
        assert split_path_and_sel("") == ("", None)


class TestExpandPathEntries:
    def test_list_passthrough(self):
        assert expand_path_entries(["a.py", "b.py"]) == ["a.py", "b.py"]

    def test_json_array_string(self):
        assert expand_path_entries('["a.py", "b.py"]') == ["a.py", "b.py"]

    def test_semicolon_string(self):
        assert expand_path_entries("src; tests") == ["src", "tests"]

    def test_no_comma_split(self):
        # A comma separates ranges inside a selector — never an entry delimiter.
        assert expand_path_entries("src/a.py:1-2,3-4") == ["src/a.py:1-2,3-4"]

    def test_dedupe(self):
        assert expand_path_entries(["a.py", "a.py", "b.py"]) == ["a.py", "b.py"]

    def test_empty_entries_dropped(self):
        assert expand_path_entries("src; ;tests") == ["src", "tests"]
        assert expand_path_entries("") == []

    def test_json_non_string_items(self):
        # Non-string items fall back to semicolon splitting of the raw string.
        assert expand_path_entries("[1, 2]") == ["[1, 2]"]

    def test_plain_string(self):
        assert expand_path_entries("single.py") == ["single.py"]


class TestMergeRangesInto:
    def test_merge_appends(self):
        m: dict[str, list[LineRange]] = {}
        merge_ranges_into(m, "/a.py", [LineRange(1, 5)])
        merge_ranges_into(m, "/a.py", [LineRange(10, 20)])
        assert m == {"/a.py": [LineRange(1, 5), LineRange(10, 20)]}

    def test_merge_none_noop(self):
        m: dict[str, list[LineRange]] = {}
        merge_ranges_into(m, "/a.py", None)
        assert m == {}


class TestGrepPathSpec:
    def test_defaults(self):
        spec = GrepPathSpec(original="a.py:1-5", clean="a.py")
        assert spec.literal_filesystem_match is False
        assert spec.ranges is None
