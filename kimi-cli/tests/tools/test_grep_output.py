"""Tests for grep_output.py (plans/23-grep-rich.md §7)."""

from __future__ import annotations

from kimi_cli.tools.file.grep_output import (
    format_grouped_output,
    format_match_line,
    group_line_indices_by_blank,
    group_lines_by_file,
    should_group,
)


def _parse(line: str) -> tuple[str, int, str, bool] | None:
    """Mini parser for rg-style content lines (path:LN:text / path-LN-text)."""
    if line == "--":
        return None
    for sep in (":", "-"):
        parts = line.split(sep, 2)
        if len(parts) == 3 and parts[1].isdigit():
            return (parts[0], int(parts[1]), parts[2], sep == ":")
    return None


class TestFormatMatchLine:
    def test_match_marker(self):
        assert format_match_line(12, "x = 1", True) == "*12|x = 1"

    def test_context_marker(self):
        assert format_match_line(13, "y = 2", False) == " 13|y = 2"

    def test_no_padding(self):
        assert format_match_line(3, "a", True) == "*3|a"
        assert format_match_line(1000, "b", False) == " 1000|b"


class TestGroupLinesByFile:
    def test_encounter_order(self):
        lines = [
            "a.py:1:alpha",
            "b.py:2:beta",
            "a.py:2:alpha2",
        ]
        groups = group_lines_by_file(lines, _parse)
        assert [g[0] for g in groups] == ["a.py", "b.py", "a.py"]

    def test_separator_retained_in_group(self):
        lines = [
            "a.py:1:alpha",
            "--",
            "a.py:3:alpha3",
        ]
        groups = group_lines_by_file(lines, _parse)
        assert len(groups) == 1
        body = groups[0][1]
        assert body[1] == (0, "--", False)

    def test_leading_separator_dropped(self):
        lines = ["--", "a.py:1:alpha"]
        groups = group_lines_by_file(lines, _parse)
        assert len(groups) == 1
        assert groups[0][1] == [(1, "alpha", True)]


class TestFormatGroupedOutput:
    def test_headers_and_separators(self):
        groups = [
            ("a.py", [(1, "alpha", True), (2, "ctx", False)]),
            ("b.py", [(3, "beta", True)]),
        ]
        out = format_grouped_output(groups)
        assert out == [
            "# a.py",
            "*1|alpha",
            " 2|ctx",
            "",
            "# b.py",
            "*3|beta",
        ]

    def test_no_blank_before_first(self):
        out = format_grouped_output([("x.py", [(1, "t", True)])])
        assert out == ["# x.py", "*1|t"]

    def test_separator_entry_verbatim(self):
        groups = [("a.py", [(1, "x", True), (0, "--", False), (3, "y", True)])]
        out = format_grouped_output(groups)
        assert out == ["# a.py", "*1|x", "--", "*3|y"]


class TestGroupLineIndicesByBlank:
    def test_basic(self):
        lines = ["# a", "*1|x", "", "# b", "*2|y"]
        assert group_line_indices_by_blank(lines) == [[0, 1], [3, 4]]

    def test_leading_blank(self):
        assert group_line_indices_by_blank(["", "a", "b"]) == [[1, 2]]

    def test_empty(self):
        assert group_line_indices_by_blank([]) == []

    def test_no_blanks(self):
        assert group_line_indices_by_blank(["a", "b"]) == [[0, 1]]


class _P:
    """Minimal params stand-in."""

    def __init__(self, grouped=None):
        self.grouped = grouped


class TestShouldGroup:
    def test_explicit_true(self):
        assert should_group(_P(True), has_rich_entries=False) is True

    def test_explicit_false(self):
        assert should_group(_P(False), has_rich_entries=True) is False

    def test_auto_on_rich(self):
        assert should_group(_P(None), has_rich_entries=True) is True

    def test_auto_off_plain(self):
        assert should_group(_P(None), has_rich_entries=False) is False
