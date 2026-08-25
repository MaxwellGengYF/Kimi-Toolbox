"""Unit tests for kimi_cli.tools.file.conflict_detect (plan 24 M0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from kimi_cli.tools.file.conflict_detect import ConflictError

from kimi_cli.tools.file.conflict_detect import (
    BASE_PREFIX,
    OURS_PREFIX,
    SEPARATOR,
    THEIRS_PREFIX,
    ConflictBlock,
    ConflictEntry,
    ConflictHistory,
    expand_content_tokens,
    find_dangling_openers,
    format_conflict_summary,
    format_conflict_warning,
    get_conflict_history,
    is_separator,
    match_marker,
    parse_bulk_directives,
    parse_conflict_uri,
    render_conflict_region,
    scan_conflict_lines,
    scan_file_for_conflicts,
    splice_conflict,
    conflict_region_present,
    conflict_regions_equal,
)


def _lines(*text: str) -> list[str]:
    return list(text)


# ---------------------------------------------------------------------------
# match_marker


def test_match_marker_bare_prefix() -> None:
    assert match_marker(OURS_PREFIX, OURS_PREFIX) == ""
    assert match_marker(BASE_PREFIX, BASE_PREFIX) == ""
    assert match_marker(SEPARATOR, SEPARATOR) == ""
    assert match_marker(THEIRS_PREFIX, THEIRS_PREFIX) == ""


def test_match_marker_labels() -> None:
    assert match_marker("<<<<<<< HEAD", OURS_PREFIX) == "HEAD"
    assert match_marker(">>>>>>> feature/x", THEIRS_PREFIX) == "feature/x"
    assert match_marker("||||||| base", BASE_PREFIX) == "base"


def test_match_marker_crlf() -> None:
    assert match_marker("<<<<<<< HEAD\r", OURS_PREFIX) == "HEAD"
    assert match_marker(">>>>>>>\r", THEIRS_PREFIX) == ""


def test_match_marker_non_markers() -> None:
    # Bare single characters and spaced variants never match.
    assert match_marker("<", OURS_PREFIX) is None
    assert match_marker("=", SEPARATOR) is None
    assert match_marker(">", THEIRS_PREFIX) is None
    assert match_marker("= foo", SEPARATOR) is None
    # Two spaces after prefix: not a valid label.
    assert match_marker("<<<<<<<  two", OURS_PREFIX) is None
    # Indented marker: not column 0.
    assert match_marker(" <<<<<<< HEAD", OURS_PREFIX) is None
    # Trailing space without label.
    assert match_marker("<<<<<<< ", OURS_PREFIX) is None


def test_is_separator_exact_only() -> None:
    assert is_separator("=======")
    assert is_separator("=======\r")
    assert not is_separator("======= x")
    assert not is_separator("======= ")


# ---------------------------------------------------------------------------
# scan_conflict_lines


def test_scan_two_way_block_with_labels() -> None:
    blocks = scan_conflict_lines(
        _lines(
            "before",
            "<<<<<<< HEAD",
            "ours line",
            "=======",
            "theirs line",
            ">>>>>>> feature/x",
            "after",
        )
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.start_line == 2
    assert b.separator_line == 4
    assert b.end_line == 6
    assert b.base_line is None
    assert b.ours_label == "HEAD"
    assert b.theirs_label == "feature/x"
    assert b.ours_lines == ("ours line",)
    assert b.base_lines is None
    assert b.theirs_lines == ("theirs line",)


def test_scan_bare_markers_no_labels() -> None:
    blocks = scan_conflict_lines(
        _lines("<<<<<<<", "a", "=======", "b", ">>>>>>>")
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.ours_label is None
    assert b.theirs_label is None


def test_scan_diff3_block() -> None:
    blocks = scan_conflict_lines(
        _lines(
            "<<<<<<< HEAD",
            "ours",
            "||||||| base",
            "base",
            "=======",
            "theirs",
            ">>>>>>> other",
        )
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.base_line == 3
    assert b.base_label == "base"
    assert b.base_lines == ("base",)


def test_scan_multiple_blocks_and_offset() -> None:
    blocks = scan_conflict_lines(
        _lines("<<<<<<< a", "x", "=======", "y", ">>>>>>> b"),
        first_line_number=200,
    )
    assert len(blocks) == 1
    assert blocks[0].start_line == 200
    assert blocks[0].end_line == 204


def test_scan_empty_sections() -> None:
    blocks = scan_conflict_lines(_lines("<<<<<<<", "=======", ">>>>>>>"))
    assert len(blocks) == 1
    assert blocks[0].ours_lines == ()
    assert blocks[0].theirs_lines == ()


def test_scan_unclosed_block_dropped() -> None:
    blocks = scan_conflict_lines(
        _lines("<<<<<<< HEAD", "ours", "=======", "theirs")
    )
    assert blocks == []


def test_scan_separator_without_opener_ignored() -> None:
    blocks = scan_conflict_lines(_lines("=======", ">>>>>>> x"))
    assert blocks == []


def test_scan_malformed_base_outside_ours_resets() -> None:
    # ||||||| outside ours phase resets; the later block is still found.
    blocks = scan_conflict_lines(
        _lines(
            "||||||| stray",
            "<<<<<<< HEAD",
            "a",
            "=======",
            "b",
            ">>>>>>> t",
        )
    )
    assert len(blocks) == 1
    assert blocks[0].start_line == 2


def test_scan_nested_opener_restarts() -> None:
    blocks = scan_conflict_lines(
        _lines(
            "<<<<<<< first",
            "<<<<<<< second",
            "a",
            "=======",
            "b",
            ">>>>>>> t",
        )
    )
    assert len(blocks) == 1
    assert blocks[0].start_line == 2
    assert blocks[0].ours_lines == ("a",)


def test_scan_non_marker_lines_not_flagged() -> None:
    blocks = scan_conflict_lines(
        _lines(
            "if a < b:",
            "x = 1 == 2",
            "y = 3 > 2",
            "= foo",
            "<<<<<<<  two spaces",
        )
    )
    assert blocks == []


def test_scan_crlf_lines_match() -> None:
    blocks = scan_conflict_lines(
        _lines("<<<<<<< HEAD\r", "a\r", "=======\r", "b\r", ">>>>>>> t\r")
    )
    assert len(blocks) == 1
    assert blocks[0].ours_lines == ("a",)
    assert blocks[0].theirs_lines == ("b",)


# ---------------------------------------------------------------------------
# scan_file_for_conflicts


def test_scan_file_full(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text(
        "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> t\n", encoding="utf-8"
    )
    result = scan_file_for_conflicts(str(p))
    assert len(result.blocks) == 1
    assert result.scan_truncated is False


def test_scan_file_byte_cap_truncates(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    # First 100 bytes are filler; the conflict lives beyond the cap.
    p.write_text("x" * 200 + "\n<<<<<<< HEAD\na\n=======\nb\n>>>>>>> t\n", encoding="utf-8")
    result = scan_file_for_conflicts(str(p), max_bytes=100)
    assert result.scan_truncated is True
    # Block beyond the cap is not found (only dropped, never invented).
    assert result.blocks == ()


def test_scan_file_unclosed_at_eof_dropped(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("<<<<<<< HEAD\na\n", encoding="utf-8")
    result = scan_file_for_conflicts(str(p))
    assert result.blocks == ()
    assert result.scan_truncated is False


def test_scan_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConflictError):
        scan_file_for_conflicts(str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# find_dangling_openers


def test_find_dangling_openers_at_eof() -> None:
    lines = _lines("text", "<<<<<<< HEAD", "ours")
    dangling = find_dangling_openers(lines)
    assert dangling == [(2, "<<<<<<< HEAD")]


def test_find_dangling_openers_closed() -> None:
    lines = _lines("<<<<<<< HEAD", "a", "=======", "b", ">>>>>>> t")
    assert find_dangling_openers(lines) == []


# ---------------------------------------------------------------------------
# ConflictHistory


def _block(start: int = 2, sep: int = 4, end: int = 6) -> ConflictBlock:
    return ConflictBlock(
        start_line=start,
        separator_line=sep,
        end_line=end,
        ours_lines=("a",),
        theirs_lines=("b",),
    )


def test_history_register_assigns_ids() -> None:
    h = ConflictHistory()
    e1 = h.register("/f.py", "f.py", _block(2, 4, 6))
    e2 = h.register("/f.py", "f.py", _block(10, 12, 14))
    assert e1.id == 1
    assert e2.id == 2
    assert [e.id for e in h.entries()] == [1, 2]


def test_history_register_reuses_id_same_start_line() -> None:
    h = ConflictHistory()
    e1 = h.register("/f.py", "f.py", _block(2, 4, 6))
    e2 = h.register(
        "/f.py",
        "f.py",
        ConflictBlock(
            start_line=2, separator_line=5, end_line=8, ours_lines=("new",), theirs_lines=("x",)
        ),
    )
    assert e2.id == e1.id
    # Overwrite region.
    assert h.get(e1.id).end_line == 8
    assert h.get(e1.id).ours_lines == ("new",)
    assert len(h.entries()) == 1


def test_history_invalidate_and_path() -> None:
    h = ConflictHistory()
    e1 = h.register("/a.py", "a.py", _block(2, 4, 6))
    e2 = h.register("/b.py", "b.py", _block(2, 4, 6))
    h.invalidate(e1.id)
    assert h.get(e1.id) is None
    assert h.get(e2.id) is not None
    h.invalidate_path("/b.py")
    assert h.get(e2.id) is None
    assert h.entries() == []


@dataclass
class _FakeSession:
    conflict_history: ConflictHistory | None = None


def test_get_conflict_history_lazy() -> None:
    s = _FakeSession()
    h1 = get_conflict_history(s)  # type: ignore[arg-type]
    h2 = get_conflict_history(s)  # type: ignore[arg-type]
    assert h1 is h2
    assert s.conflict_history is h1


# ---------------------------------------------------------------------------
# parse_conflict_uri


def test_parse_uri_basic_id() -> None:
    parsed = parse_conflict_uri("conflict://12")
    assert parsed is not None
    assert parsed.id == 12
    assert parsed.scope is None
    assert parsed.recovered_prefix is None


def test_parse_uri_scopes() -> None:
    for scope in ("ours", "theirs", "base"):
        parsed = parse_conflict_uri(f"conflict://3/{scope}")
        assert parsed is not None
        assert parsed.id == 3
        assert parsed.scope == scope


def test_parse_uri_wildcard() -> None:
    parsed = parse_conflict_uri("conflict://*")
    assert parsed is not None
    assert parsed.id == "*"
    assert parsed.scope is None


def test_parse_uri_prefix_recovered() -> None:
    parsed = parse_conflict_uri("src/x.py:conflict://4")
    assert parsed is not None
    assert parsed.id == 4
    assert parsed.recovered_prefix == "src/x.py"


def test_parse_uri_non_conflict_none() -> None:
    assert parse_conflict_uri("src/x.py") is None
    assert parse_conflict_uri("") is None


def test_parse_uri_invalid_id() -> None:
    with pytest.raises(ConflictError):
        parse_conflict_uri("conflict://abc")
    with pytest.raises(ConflictError):
        parse_conflict_uri("conflict://0")


def test_parse_uri_invalid_scope() -> None:
    with pytest.raises(ConflictError):
        parse_conflict_uri("conflict://1/bogus")


def test_parse_uri_wildcard_with_scope_rejected() -> None:
    with pytest.raises(ConflictError):
        parse_conflict_uri("conflict://*/ours")


# ---------------------------------------------------------------------------
# expand_content_tokens


def _entry(
    ours=("OURS",),
    theirs=("THEIRS",),
    base=None,
) -> ConflictEntry:
    return ConflictEntry(
        start_line=2,
        separator_line=4 if base is None else 5,
        end_line=6 if base is None else 8,
        base_line=3 if base is not None else None,
        ours_lines=tuple(ours),
        base_lines=tuple(base) if base is not None else None,
        theirs_lines=tuple(theirs),
        id=1,
        absolute_path="/f.py",
        display_path="f.py",
    )


def test_expand_tokens_basic() -> None:
    out = expand_content_tokens("before\n@ours\nafter", _entry())
    assert out == "before\nOURS\nafter"
    out = expand_content_tokens("@theirs", _entry())
    assert out == "THEIRS"


def test_expand_token_both() -> None:
    out = expand_content_tokens("@both", _entry())
    assert out == "OURS\nTHEIRS"


def test_expand_token_base_two_way_raises() -> None:
    with pytest.raises(ConflictError):
        expand_content_tokens("@base", _entry())


def test_expand_token_base_three_way() -> None:
    out = expand_content_tokens("@base", _entry(base=("BASE",)))
    assert out == "BASE"


def test_expand_tokens_verbatim() -> None:
    out = expand_content_tokens("keep\nme", _entry())
    assert out == "keep\nme"


# ---------------------------------------------------------------------------
# splice_conflict


CONFLICT_TEXT = "a\nb\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> t\nc\nd\n"


def _entry_from_lines() -> ConflictEntry:
    blocks = scan_conflict_lines(CONFLICT_TEXT.split("\n"))
    assert len(blocks) == 1
    return ConflictEntry(
        start_line=blocks[0].start_line,
        separator_line=blocks[0].separator_line,
        end_line=blocks[0].end_line,
        base_line=blocks[0].base_line,
        ours_label=blocks[0].ours_label,
        base_label=blocks[0].base_label,
        theirs_label=blocks[0].theirs_label,
        ours_lines=blocks[0].ours_lines,
        base_lines=blocks[0].base_lines,
        theirs_lines=blocks[0].theirs_lines,
        id=1,
        absolute_path="/f.py",
        display_path="f.py",
    )


def test_splice_exact_ours() -> None:
    entry = _entry_from_lines()
    splice = splice_conflict(CONFLICT_TEXT, entry, "ours")
    assert splice.text == "a\nb\nours\nc\nd\n"
    assert splice.trimmed_leading == 0
    assert splice.trimmed_trailing == 0


def test_splice_replacement_empty_removes_region() -> None:
    entry = _entry_from_lines()
    splice = splice_conflict(CONFLICT_TEXT, entry, "")
    assert splice.text == "a\nb\nc\nd\n"


def test_splice_shifted_anchor_still_locates() -> None:
    entry = _entry_from_lines()
    shifted = "NEW1\nNEW2\n" + CONFLICT_TEXT
    splice = splice_conflict(shifted, entry, "resolved")
    assert splice.text == "NEW1\nNEW2\na\nb\nresolved\nc\nd\n"


def test_splice_missing_block_raises() -> None:
    entry = _entry_from_lines()
    with pytest.raises(ConflictError):
        splice_conflict("no conflicts here\n", entry, "x")


def test_splice_boundary_echo_multi_line_trimmed() -> None:
    # Replacement echoes adjacent context lines; the echo is trimmed.
    entry = _entry_from_lines()
    splice = splice_conflict(CONFLICT_TEXT, entry, "b\nours\nc\nd")
    assert splice.text == "a\nb\nours\nc\nd\n"
    assert splice.trimmed_leading == 1
    assert splice.trimmed_trailing == 2


def test_splice_crlf_roundtrip() -> None:
    entry = _entry_from_lines()
    crlf = CONFLICT_TEXT.replace("\n", "\r\n")
    splice = splice_conflict(crlf, entry, "ours")
    assert splice.text == "a\r\nb\r\nours\r\nc\r\nd\r\n"


# ---------------------------------------------------------------------------
# conflict_regions_equal / present


def test_conflict_regions_equal_and_present() -> None:
    entry = _entry_from_lines()
    other = _entry_from_lines()
    assert conflict_regions_equal(entry, other)
    assert conflict_region_present(CONFLICT_TEXT, entry)
    assert not conflict_region_present("clean\nfile\n", entry)


# ---------------------------------------------------------------------------
# render_conflict_region


def test_render_full_region() -> None:
    entry = _entry_from_lines()
    lines, start = render_conflict_region(entry, None)
    assert start == 3
    assert lines[0] == "<<<<<<< HEAD"
    assert lines[-1] == ">>>>>>> t"
    assert "ours" in lines


def test_render_side_scopes() -> None:
    entry = _entry_from_lines()
    ours, o_start = render_conflict_region(entry, "ours")
    assert ours == ["ours"]
    theirs, t_start = render_conflict_region(entry, "theirs")
    assert theirs == ["theirs"]


def test_render_base_scope_two_way_raises() -> None:
    entry = _entry_from_lines()
    with pytest.raises(ConflictError):
        render_conflict_region(entry, "base")


# ---------------------------------------------------------------------------
# format_conflict_warning / summary


def test_format_conflict_warning_shapes() -> None:
    entry = _entry_from_lines()
    out = format_conflict_warning([entry], display_path="f.py")
    assert "⚠ 1 unresolved conflict detected in f.py" in out
    assert "──── #1  L3-7 ────" in out
    assert "NOTICE" in out
    assert "@ours" in out


def test_format_conflict_warning_partial_window() -> None:
    entry = _entry_from_lines()
    out = format_conflict_warning(
        [entry], total_in_file=3, display_path="f.py"
    )
    assert "1 of 3 unresolved conflicts visible in this window" in out
    assert "f.py:conflicts" in out


def test_format_conflict_warning_scan_truncated_note() -> None:
    entry = _entry_from_lines()
    out = format_conflict_warning([entry], scan_truncated=True)
    assert "byte cap" in out


def test_format_conflict_warning_labels() -> None:
    entry = _entry_from_lines()
    out = format_conflict_warning([entry])
    assert "- ours = HEAD" in out
    assert "- theirs = t" in out


def test_format_conflict_warning_equal_sections_collapse() -> None:
    entry = _entry_from_lines()
    # Make theirs identical to ours.
    same = ConflictEntry(
        start_line=entry.start_line,
        separator_line=entry.separator_line,
        end_line=entry.end_line,
        base_line=None,
        ours_lines=("same",),
        base_lines=None,
        theirs_lines=("same",),
        id=1,
        absolute_path="/f.py",
        display_path="f.py",
    )
    out = format_conflict_warning([same])
    assert ">>> theirs ≡ ours" in out


def test_format_conflict_summary_shapes() -> None:
    entry = _entry_from_lines()
    out = format_conflict_summary([entry], display_path="f.py")
    assert "⚠ 1 unresolved conflict in f.py" in out
    assert "#1  L3-7" in out


def test_format_conflict_summary_empty() -> None:
    out = format_conflict_summary([], display_path="f.py")
    assert out == "No unresolved git merge conflicts in f.py."


def test_format_conflict_summary_three_way_marker() -> None:
    entry = _entry_from_lines()
    three_way = ConflictEntry(
        start_line=3,
        separator_line=5,
        end_line=7,
        base_line=4,
        ours_lines=("a",),
        base_lines=("b",),
        theirs_lines=("c",),
        id=1,
        absolute_path="/f.py",
        display_path="f.py",
    )
    out = format_conflict_summary([three_way], display_path="f.py")
    assert "(3-way)" in out


# ---------------------------------------------------------------------------
# parse_bulk_directives


def test_parse_bulk_directives_valid() -> None:
    d = parse_bulk_directives("1: @ours\n2: @theirs\n3: @both\n")
    assert d == {1: "ours", 2: "theirs", 3: "both"}


def test_parse_bulk_directives_with_blank_lines() -> None:
    d = parse_bulk_directives("1: @ours\n\n2: @theirs\n")
    assert d == {1: "ours", 2: "theirs"}


def test_parse_bulk_directives_non_directive_returns_none() -> None:
    assert parse_bulk_directives("some code\nhere") is None
    assert parse_bulk_directives("1: @ours\nplain line") is None


def test_parse_bulk_directives_empty_returns_none() -> None:
    assert parse_bulk_directives("") is None
    assert parse_bulk_directives("\n\n") is None
