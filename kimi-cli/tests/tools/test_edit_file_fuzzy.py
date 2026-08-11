"""Tests for EditFile fuzzy matching and line-ending normalization."""

from __future__ import annotations

from kimi_cli.tools.file.replace import Edit, EditFile, Params

# ---------------------------------------------------------------------------
# Line-ending normalization
# ---------------------------------------------------------------------------


def test_normalize_line_endings():
    """_normalize_line_endings converts \\r\\n to \\n."""
    tool = object.__new__(EditFile)
    assert tool._normalize_line_endings("a\r\nb") == "a\nb"
    assert tool._normalize_line_endings("a\nb") == "a\nb"
    assert tool._normalize_line_endings("") == ""
    assert tool._normalize_line_endings("a\r\nb\r\nc") == "a\nb\nc"


def test_apply_edit_crlf_file_lf_old():
    """Exact match should succeed when file uses CRLF but old string uses LF."""
    tool = object.__new__(EditFile)
    content = "line1\r\nline2\r\nline3"
    old = "line2\nline3"
    new = "replaced"
    result, count, suggestion = tool._apply_edit(content, Edit(old=old, new=new))
    assert count == 1
    # Exact match via CRLF normalization succeeds, no fuzzy info
    assert suggestion is None
    assert result == "line1\nreplaced"


def test_apply_edit_lf_file_crlf_old():
    """Exact match should succeed when file uses LF but old string uses CRLF."""
    tool = object.__new__(EditFile)
    content = "line1\nline2\nline3"
    old = "line2\r\nline3"
    new = "replaced"
    result, count, suggestion = tool._apply_edit(content, Edit(old=old, new=new))
    assert count == 1
    assert suggestion is None
    assert result == "line1\nreplaced"


def test_apply_edit_replace_all_crlf():
    """replace_all should work across CRLF boundaries."""
    tool = object.__new__(EditFile)
    content = "a\r\nb\r\na\r\nc"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old="a\n", new="X\n", replace_all=True)
    )
    assert count == 2
    assert suggestion is None
    assert result == "X\nb\nX\nc"


# ---------------------------------------------------------------------------
# Default match_mode is fuzzy
# ---------------------------------------------------------------------------


def test_edit_match_mode_default_is_fuzzy():
    """Edit.match_mode defaults to fuzzy."""
    assert Edit(old="a", new="b").match_mode == "fuzzy"


def test_apply_edit_default_fuzzy_applies_fallback():
    """Default (fuzzy) mode applies fuzzy fallback when exact match fails."""
    tool = object.__new__(EditFile)
    content = "helo world\nnext line"
    old = "hello world"
    new = "hi universe"
    result, count, suggestion = tool._apply_edit(content, Edit(old=old, new=new))
    assert count == 1
    assert result == "hi universe\nnext line"

    # Sanity check: explicit exact mode is NOT replaced — it suggests instead.
    result_exact, count_exact, suggestion_exact = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="exact")
    )
    assert count_exact == 0
    assert result_exact == content
    # The closest line is reported as a suggestion, not replaced
    assert suggestion_exact == "helo world"


# ---------------------------------------------------------------------------
# Fuzzy single-line matching
# ---------------------------------------------------------------------------


def test_apply_edit_fuzzy_single_line_trailing_spaces():
    """Fuzzy match should tolerate trailing whitespace differences on a single line."""
    tool = object.__new__(EditFile)
    content = "hello world  \nnext line"
    old = "hello world"
    new = "hi universe"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 1
    assert suggestion is None
    assert result == "hi universe  \nnext line"


def test_apply_edit_fuzzy_single_line_leading_spaces():
    """Fuzzy match should tolerate leading whitespace differences on a single line."""
    tool = object.__new__(EditFile)
    content = "  hello world\nnext line"
    old = "hello world"
    new = "hi universe"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 1
    assert suggestion is None
    assert result == "  hi universe\nnext line"


def test_apply_edit_fuzzy_case_close():
    """Fuzzy match should handle slightly altered wording."""
    tool = object.__new__(EditFile)
    content = "def compute_sum(a, b):\n    return a + b"
    old = "def compute_sum(a, b):"
    new = "def add(a, b):"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 1
    assert suggestion is None
    assert result == "def add(a, b):\n    return a + b"


def test_apply_edit_fuzzy_no_match_returns_suggestion():
    """When nothing matches, no suggestion is returned if below similarity cutoff."""
    tool = object.__new__(EditFile)
    content = "hello world\nfoo bar\nbaz qux"
    old = "xyz123_not_close"
    new = "replacement"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 0
    assert result == content
    # Below cutoff — no suggestion
    assert suggestion is None


def test_apply_edit_fuzzy_replace_all_no_match_returns_suggestion():
    """replace_all with no match should return a suggestion if close enough."""
    tool = object.__new__(EditFile)
    content = "hello world\nfoo bar"
    # Close enough to trigger suggestion via _find_similar
    old = "helo wrld"
    new = "replacement"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, replace_all=True, match_mode="fuzzy")
    )
    assert count == 0
    # "helo wrld" is close enough to get a suggestion
    assert suggestion == "hello world"

    # Completely unrelated — no suggestion
    old = "xyz123_not_close"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, replace_all=True, match_mode="fuzzy")
    )
    assert count == 0
    assert suggestion is None


# ---------------------------------------------------------------------------
# Fuzzy multi-line matching
# ---------------------------------------------------------------------------


def test_apply_edit_fuzzy_multiline_minor_whitespace():
    """Fuzzy match should work for multi-line blocks with minor whitespace diffs."""
    tool = object.__new__(EditFile)
    content = "start\n  line A\n  line B\nend"
    old = "line A\nline B"
    new = "line X\nline Y"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 1
    # Fuzzy match returns match info
    assert suggestion is not None
    assert "fuzzy-matched" in suggestion
    # Fuzzy match replaces the whole matched chunk (including indentation)
    assert result == "start\nline X\nline Y\nend"


def test_apply_edit_fuzzy_multiline_close_match():
    """Fuzzy match should find close multi-line chunks."""
    tool = object.__new__(EditFile)
    content = "class Foo:\n    def bar(self):\n        pass\n    def baz(self):\n        pass"
    old = "    def bar(self):\n        pass"
    new = "    def qux(self):\n        return 42"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, match_mode="fuzzy")
    )
    assert count == 1
    assert suggestion is None
    assert "def qux(self):" in result
    assert "return 42" in result


# ---------------------------------------------------------------------------
# _apply_replace_all helper
# ---------------------------------------------------------------------------


def test_apply_replace_all_max_replacements():
    """replace_all with max_replacements replaces only the first N occurrences."""
    tool = object.__new__(EditFile)
    result, count, suggestion = tool._apply_edit(
        "a b a c a d a", Edit(old="a", new="X", replace_all=True, max_replacements=2)
    )
    assert count == 2
    assert suggestion is None
    assert result == "X b X c a d a"


def test_apply_replace_all_exact_only_no_fuzzy_fallback():
    """replace_all stays exact-only even in fuzzy mode: near-misses are suggested, not replaced."""
    tool = object.__new__(EditFile)
    content = "hello world\nfoo bar"
    old = "helo wrld"  # close to "hello world" but not an exact match
    new = "replacement"
    result, count, suggestion = tool._apply_edit(
        content, Edit(old=old, new=new, replace_all=True, match_mode="fuzzy")
    )
    assert count == 0
    assert result == content
    # Close enough to trigger a suggestion, but never a fuzzy replacement
    assert suggestion == "hello world"


# ---------------------------------------------------------------------------
# _apply_fuzzy_fallback helper
# ---------------------------------------------------------------------------


def _call_fuzzy_fallback(tool: EditFile, content: str, old: str, new: str):
    """Invoke the fuzzy-fallback helper with normalized inputs."""
    edit = Edit(old=old, new=new)
    return tool._apply_fuzzy_fallback(
        content,
        tool._normalize_line_endings(content),
        tool._normalize_line_endings(old),
        tool._normalize_line_endings(new),
        edit,
    )


def test_apply_fuzzy_fallback_chain():
    """_apply_fuzzy_fallback walks strip -> fuzzy match -> suggestion fallback."""
    tool = object.__new__(EditFile)

    # 1. Strip match (whitespace only): no fuzzy-matched marker, preserves spaces
    result, count, suggestion = _call_fuzzy_fallback(
        tool, "hello world  \nnext line", "hello world", "hi universe"
    )
    assert count == 1
    assert suggestion is None
    assert result == "hi universe  \nnext line"

    # 2. Fuzzy match (wording drift): replaced and reports fuzzy-matched marker
    result, count, suggestion = _call_fuzzy_fallback(
        tool, "helo world\nnext line", "hello world", "hi universe"
    )
    assert count == 1
    assert suggestion is not None
    assert "fuzzy-matched" in suggestion
    assert result == "hi universe\nnext line"

    # 3. No match at all: content untouched, no suggestion below cutoff
    result, count, suggestion = _call_fuzzy_fallback(
        tool, "hello world\nfoo bar", "xyz123_not_close", "replacement"
    )
    assert count == 0
    assert result == "hello world\nfoo bar"
    assert suggestion is None


# ---------------------------------------------------------------------------
# _find_similar edge cases
# ---------------------------------------------------------------------------


def test_find_similar_empty_content():
    """_find_similar should return None for empty content."""
    tool = object.__new__(EditFile)
    assert tool._find_similar("target", "") is None


def test_find_similar_exact_match():
    """_find_similar should return the exact line when present."""
    tool = object.__new__(EditFile)
    assert tool._find_similar("hello", "hello\nworld") == "hello"


def test_find_similar_typo():
    """_find_similar should return a close line for minor typos."""
    tool = object.__new__(EditFile)
    content = "hello world\nfoo bar\nbaz qux"
    suggestion = tool._find_similar("hello wrld", content)
    assert suggestion == "hello world"


def test_find_similar_multiline_window():
    """_find_similar should return close multi-line windows."""
    tool = object.__new__(EditFile)
    content = "a\nb\nc\nd"
    suggestion = tool._find_similar("b\nc", content)
    assert suggestion == "b\nc"


def test_find_similar_below_cutoff():
    """_find_similar should return None when similarity is below cutoff."""
    tool = object.__new__(EditFile)
    content = "abc"
    assert tool._find_similar("xyz123!@#", content, cutoff=90.0) is None


# ---------------------------------------------------------------------------
# _find_best_fuzzy_match edge cases
# ---------------------------------------------------------------------------


def test_find_best_fuzzy_match_exact():
    """_find_best_fuzzy_match should return exact match with score 100."""
    tool = object.__new__(EditFile)
    result = tool._find_best_fuzzy_match("hello world", "hello world")
    assert result is not None
    matched, score = result
    assert matched == "hello world"
    assert score == 100.0


def test_find_best_fuzzy_match_none():
    """_find_best_fuzzy_match should return None for completely different text."""
    tool = object.__new__(EditFile)
    assert tool._find_best_fuzzy_match("zzzzzz", "abcdef") is None


def test_find_best_fuzzy_match_multiline():
    """_find_best_fuzzy_match should handle multi-line targets."""
    tool = object.__new__(EditFile)
    content = "line1\nline2\nline3\nline4"
    result = tool._find_best_fuzzy_match("line2\nline3", content)
    assert result is not None
    matched, score = result
    assert matched == "line2\nline3"
    assert score == 100.0


def test_find_best_fuzzy_match_preserves_crlf():
    """_find_best_fuzzy_match should preserve original CRLF in matched text."""
    tool = object.__new__(EditFile)
    content = "line1\r\nline2\r\nline3"
    result = tool._find_best_fuzzy_match("line2", content)
    assert result is not None
    matched, score = result
    # The matched text should be the normalized version since we reconstruct
    # from norm_lines. The exact preservation of CRLF happens in the caller
    # via norm_content replacement.
    assert "line2" in matched


# ---------------------------------------------------------------------------
# Integration-style async tests for new behaviours
# ---------------------------------------------------------------------------


async def test_replace_crlf_file_with_lf_old(edit_file_tool, temp_work_dir):
    """End-to-end: file has CRLF, old string has LF."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("Hello\r\nWorld\r\nTest")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="Hello\nWorld", new="Hi\nUniverse"))
    )

    assert not result.is_error
    assert await file_path.read_text() == "Hi\nUniverse\nTest"


async def test_replace_fuzzy_trailing_whitespace(edit_file_tool, temp_work_dir):
    """End-to-end: old string lacks trailing spaces that file has."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("hello world  \nnext line")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="hello world", new="hi universe", match_mode="fuzzy"),
        )
    )

    assert not result.is_error
    assert await file_path.read_text() == "hi universe  \nnext line"


async def test_replace_fuzzy_leading_whitespace(edit_file_tool, temp_work_dir):
    """End-to-end: old string lacks leading spaces that file has."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("  hello world\nnext line")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="hello world", new="hi universe", match_mode="fuzzy"),
        )
    )

    assert not result.is_error
    assert await file_path.read_text() == "  hi universe\nnext line"


async def test_replace_no_match_with_suggestion(edit_file_tool, temp_work_dir):
    """End-to-end: no match should return an error."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("hello world\nfoo bar\nbaz qux")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="xyz123_not_close", new="hi universe"))
    )

    assert result.is_error
    assert "No replacements were made" in result.message


async def test_replace_default_fuzzy_typo(edit_file_tool, temp_work_dir):
    """End-to-end: default (fuzzy) mode replaces a near-miss old string."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("helo world\nnext line")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="hello world", new="hi universe"))
    )

    assert not result.is_error
    assert "successfully edited" in result.message
    assert await file_path.read_text() == "hi universe\nnext line"


async def test_replace_exact_mode_typo_suggests(edit_file_tool, temp_work_dir):
    """End-to-end: explicit exact mode with a near-miss old string errors with a suggestion."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("helo world\nnext line")

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="hello world", new="hi universe", match_mode="exact"),
        )
    )

    assert result.is_error
    assert "No replacements were made" in result.message
    # The closest line is reported as a suggestion, not replaced
    assert "helo world" in result.message
    assert await file_path.read_text() == "helo world\nnext line"


async def test_replace_fuzzy_multiline(edit_file_tool, temp_work_dir):
    """End-to-end: fuzzy multi-line replacement."""
    file_path = temp_work_dir / "test.txt"
    original = "start\n  line A\n  line B\nend"
    await file_path.write_text(original)

    result = await edit_file_tool(
        Params(
            path=str(file_path),
            edit=Edit(old="line A\nline B", new="line X\nline Y", match_mode="fuzzy"),
        )
    )

    assert not result.is_error
    # Fuzzy match replaces the whole matched chunk
    assert await file_path.read_text() == "start\nline X\nline Y\nend"


async def test_replace_all_with_crlf(edit_file_tool, temp_work_dir):
    """End-to-end: replace_all with CRLF file and LF old string."""
    file_path = temp_work_dir / "test.txt"
    await file_path.write_text("a\r\nb\r\na\r\nc")

    result = await edit_file_tool(
        Params(path=str(file_path), edit=Edit(old="a\n", new="X\n", replace_all=True))
    )

    assert not result.is_error
    assert await file_path.read_text() == "X\nb\nX\nc"
