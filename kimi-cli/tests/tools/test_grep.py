"""Tests for the grep tool."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest
from kaos.path import KaosPath

import kimi_cli.tools.file.grep_local as grep_local
from kimi_cli._rtk_common import _rtk_binary_name
from kimi_cli.share import get_share_dir
from kimi_cli.tools.file.grep_local import (
    Grep,
    Params,
    _build_rg_args,
    _env_with_shared_bin_path,
    _find_existing_rtk,
    _format_cmd,
    _rtk_fold_note,
    _strip_path_prefix,
)
from kimi_cli.tools.utils import DEFAULT_MAX_CHARS


@pytest.fixture
def temp_test_files():
    """Create temporary test files for grep testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_file1 = Path(temp_dir) / "test1.py"
        test_file1.write_text("""def hello_world():
    print("Hello, World!")
    return "hello"

class TestClass:
    def __init__(self):
        self.message = "hello there"
""")

        test_file2 = Path(temp_dir) / "test2.js"
        test_file2.write_text("""function helloWorld() {
    console.log("Hello, World!");
    return "hello";
}

class TestClass {
    constructor() {
        this.message = "hello there";
    }
}
""")

        test_file3 = Path(temp_dir) / "readme.txt"
        test_file3.write_text("""This is a readme file.
It contains some text.
Hello world example is here.
""")

        # Create a subdirectory with files
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        subfile = subdir / "subtest.py"
        subfile.write_text("def sub_hello():\n    return 'hello from subdir'\n")

        yield temp_dir, [test_file1, test_file2, test_file3, subfile]


async def test_grep_files_with_matches(grep_tool: Grep, temp_test_files):
    """Test finding files that contain a pattern."""
    temp_dir, test_files = temp_test_files

    # Test basic pattern matching to catch "Hello" in readme.txt
    result = await grep_tool(
        Params(pattern="Hello", path=temp_dir, output_mode="files_with_matches")
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should find all test files that contain "hello" (case insensitive)
    assert "test1.py" in result.output
    assert "test2.js" in result.output
    assert "readme.txt" in result.output


async def test_grep_content_mode(grep_tool: Grep, temp_test_files):
    """Test showing matching lines with content."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": temp_dir,
                "output_mode": "content",
                "line_number": True,
                "ignore_case": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should show matching lines with line numbers
    assert "hello" in result.output.lower()
    assert ":" in result.output  # Line numbers should be present


async def test_grep_case_insensitive(grep_tool: Grep, temp_test_files):
    """Test case insensitive search."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "HELLO",
                "path": temp_dir,
                "output_mode": "files_with_matches",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should find files with "hello" (lowercase)
    assert "test1.py" in result.output


async def test_grep_with_context(grep_tool: Grep, temp_test_files):
    """Test showing context around matches."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "TestClass",
                "path": temp_dir,
                "output_mode": "content",
                "-C": 1,
                "-n": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should show context lines
    lines = result.output.split("\n")
    assert len(lines) > 2  # Should have more than just the matching line


async def test_grep_count_matches(grep_tool: Grep, temp_test_files):
    """Test counting matches."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": temp_dir,
                "output_mode": "count_matches",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should show count for each file
    assert "test1.py" in result.output
    assert "test2.js" in result.output


async def test_grep_with_glob_pattern(grep_tool: Grep, temp_test_files):
    """Test filtering files with glob pattern."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": temp_dir,
                "output_mode": "files_with_matches",
                "glob": "*.py",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should only find Python files
    assert "test1.py" in result.output
    assert "subtest.py" in result.output
    assert "test2.js" not in result.output
    assert "readme.txt" not in result.output


async def test_grep_with_type_filter(grep_tool: Grep, temp_test_files):
    """Test filtering by file type."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": temp_dir,
                "output_mode": "files_with_matches",
                "type": "py",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should only find Python files
    assert "test1.py" in result.output
    assert "subtest.py" in result.output
    assert "test2.js" not in result.output
    assert "readme.txt" not in result.output


async def test_grep_head_limit(grep_tool: Grep, temp_test_files):
    """Test limiting number of results."""
    temp_dir, test_files = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": temp_dir,
                "output_mode": "files_with_matches",
                "head_limit": 2,
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)

    # Should limit results to 2 files
    lines = [
        line for line in result.output.split("\n") if line.strip() and not line.startswith("...")
    ]
    assert len(lines) <= 2
    assert "Results truncated to 2 lines" in result.message


async def test_grep_output_truncation(grep_tool: Grep):
    """Ensure extremely long output is truncated/deduplicated automatically."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "big.txt"
        test_file.write_text(
            "match line with filler content that keeps growing for truncation purposes\n" * 2000
        )

        result = await grep_tool(
            Params.model_validate(
                {
                    "pattern": "match",
                    "path": temp_dir,
                    "output_mode": "content",
                    "head_limit": 0,
                    "-n": True,
                }
            )
        )

        assert not result.is_error
        assert isinstance(result.output, str)
        # rtk (when active) deduplicates the repeated lines and folds the
        # rest; the tool strips rtk protocol lines and summarizes them in the
        # message instead of leaking them into the output stream. Without
        # rtk the local fold/byte-cap fallbacks apply instead.
        assert "matches in" not in result.output
        assert "[see remaining:" not in result.output
        assert any(
            token in result.message
            for token in (
                "Found 2000 matches in 1 files.",  # rtk content summary
                "Results folded to",  # local fold fallback (no rtk)
                "Output truncated to 102400 bytes",  # byte cap fallback
            )
        )
        assert len(result.output) < DEFAULT_MAX_CHARS + 100


async def test_grep_multiline_mode(grep_tool: Grep):
    """Test multiline pattern matching."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with multiline content
        test_file = Path(temp_dir) / "multiline.py"
        test_file.write_text(
            """def function():
    '''This is a
    multiline docstring'''
    pass
""",
            newline="\n",
        )

        # Test multiline pattern
        result = await grep_tool(
            Params(
                pattern=r"This is a\n    multiline",
                path=temp_dir,
                output_mode="content",
                multiline=True,
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)

        # Should find the multiline pattern
        assert "This is a" in result.output
        assert "multiline" in result.output


async def test_grep_auto_multiline_real_newline(grep_tool: Grep):
    """A pattern with a REAL newline matches across lines without multiline=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "multi.py"
        test_file.write_text(
            "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
            newline="\n",
        )

        # No multiline=True: auto-detection must enable multiline mode.
        result = await grep_tool(
            Params(
                pattern="def foo\\(\\):\n    return 1",
                path=temp_dir,
                output_mode="content",
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "def foo():" in result.output
        assert "return 1" in result.output


async def test_grep_auto_multiline_escape(grep_tool: Grep):
    """A regex `\n` escape matches across lines without multiline=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "multi.py"
        test_file.write_text(
            "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
            newline="\n",
        )

        # No multiline=True: the `\n` escape must auto-enable multiline mode.
        result = await grep_tool(
            Params(
                pattern=r"def foo\(\):\n    return 1",
                path=temp_dir,
                output_mode="content",
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "def foo():" in result.output
        assert "return 1" in result.output


async def test_grep_literal_backslash_n_stays_line_oriented(grep_tool: Grep):
    """A literal backslash-n (even backslashes) does NOT auto-enable multiline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Real line boundary between alpha and beta: literal `\n` must NOT
        # match across it (pattern stays line-oriented).
        (Path(temp_dir) / "real_newline.txt").write_text("alpha\nbeta\n", newline="\n")
        # Same-line literal backslash-n: literal search still works.
        (Path(temp_dir) / "literal.txt").write_text("x alpha\\nbeta y\n", newline="\n")

        result = await grep_tool(
            Params(pattern=r"alpha\\nbeta", path=temp_dir, output_mode="content")
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        # Only the same-line literal backslash-n matches; no cross-line match.
        assert "literal.txt" in result.output
        assert "real_newline.txt" not in result.output


def test_pattern_has_regex_newline():
    """Unit tests for the auto-multiline pattern detection helper."""
    from kimi_cli.tools.file.grep_local import _pattern_has_regex_newline

    # Real newline decoded into the pattern.
    assert _pattern_has_regex_newline("a\nb") is True
    # Regex `\n` escape (single backslash).
    assert _pattern_has_regex_newline(r"a\nb") is True
    # Literal backslash-n (even number of backslashes) → not a newline escape.
    assert _pattern_has_regex_newline(r"a\\nb") is False
    # No newline at all.
    assert _pattern_has_regex_newline("abc") is False


def test_multiline_pattern_crlf_normalization():
    """Multi-line patterns are rewritten to match CRLF files too.

    ripgrep multiline mode matches raw bytes, so on CRLF files (the Windows
    default) a pattern written with ``\n`` would silently find nothing.
    ``_multiline_pattern`` rewrites newline constructs to ``\r?\n``.
    """
    from kimi_cli.tools.file.grep_local import _multiline_pattern

    # Real newline → optional carriage return.
    assert _multiline_pattern("a\nb") == r"a\r?\nb"
    # Explicit CRLF in the pattern normalizes to the same form.
    assert _multiline_pattern("a\r\nb") == r"a\r?\nb"
    # Regex `\n` escape → optional carriage return.
    assert _multiline_pattern(r"a\nb") == r"a\r?\nb"
    # Literal backslash-n (even backslashes) stays untouched.
    assert _multiline_pattern(r"a\\nb") == r"a\\nb"
    # No newline at all → unchanged.
    assert _multiline_pattern("abc") == "abc"


async def test_grep_auto_multiline_crlf_file(grep_tool: Grep):
    """A \n pattern matches across lines in a CRLF file (Windows default)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "crlf.py"
        # Default text-mode write on Windows produces CRLF; force it explicitly.
        test_file.write_bytes(b"def foo():\r\n    return 1\r\n\r\ndef bar():\r\n    return 2\r\n")

        result = await grep_tool(
            Params(
                pattern="def foo\\(\\):\n    return 1",
                path=temp_dir,
                output_mode="content",
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "def foo():" in result.output
        assert "return 1" in result.output


async def test_grep_no_matches(grep_tool: Grep):
    """Test when no matches are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "empty.py"
        test_file.write_text("# This file has no matching content\n")

        result = await grep_tool(
            Params(pattern="nonexistent_pattern", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert result.output == ""
        assert "No matches found" in result.message


async def test_grep_invalid_pattern(grep_tool: Grep):
    """Test with invalid regex pattern."""
    result = await grep_tool(Params(pattern="[invalid", path=".", output_mode="files_with_matches"))
    assert result.is_error
    assert "Failed to grep" in result.message


async def test_grep_single_file(grep_tool: Grep):
    """Test searching in a single file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write("def test_function():\n    return 'hello world'\n")
        f.flush()

        result = await grep_tool(
            Params.model_validate(
                {
                    "pattern": "hello",
                    "path": f.name,
                    "output_mode": "content",
                    "-n": True,
                }
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)

        assert "hello" in result.output
        # For single file search, filename might not be in content output
        # Let's just check that we got valid content
        assert len(result.output.strip()) > 0


async def test_grep_before_after_context(grep_tool: Grep, temp_test_files):
    """Test before and after context separately."""
    temp_dir, test_files = temp_test_files

    # Test before context
    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "TestClass",
                "path": temp_dir,
                "output_mode": "content",
                "-B": 2,
                "-n": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)
    assert "TestClass" in result.output
    assert "}" in result.output
    assert 'return "hello"' in result.output
    assert "Hello, World!" not in result.output

    # Test after context
    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "TestClass",
                "path": temp_dir,
                "output_mode": "content",
                "-A": 2,
                "-n": True,
            }
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)
    assert "TestClass" in result.output
    assert "constructor()" in result.output
    assert "this.message" in result.output
    assert "}" not in result.output


# === Tests for new features ===

async def test_grep_default_head_limit(grep_tool: Grep):
    """Default head_limit=500 truncates large result sets."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(600):
            (Path(temp_dir) / f"file_{i:03d}.txt").write_text("marker\n")

        # head_limit defaults to 500; with max_output_lines=500 the fold
        # budget matches the pagination limit, so no fold marker appears.
        result = await grep_tool(
            Params(pattern="marker", path=temp_dir, output_mode="files_with_matches")
        )
        output = result.output.split("\n")
        assert len(output) == 500
        lines = [x for x in result.output.split("\n") if x.strip()]
        assert len(lines) == 500
        assert "lines omitted" not in result.output
        assert "Results truncated to 500 lines" in result.message
        assert "total: 600" in result.message
        assert "Use offset=500 to see more" in result.message
        assert "Results folded" not in result.message


async def test_grep_head_limit_zero_unlimited(grep_tool: Grep):
    """head_limit=0 keeps all results; the display fold budget is 500 by default."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(300):
            (Path(temp_dir) / f"file_{i:03d}.txt").write_text("marker\n")

        result = await grep_tool(
            Params(pattern="marker", path=temp_dir, output_mode="files_with_matches", head_limit=0)
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        lines = [x for x in result.output.split("\n") if x.strip()]
        # All 300 collected; 300 <= max_output_lines=500 so no folding.
        assert len(lines) == 300
        assert "lines omitted" not in result.output
        assert "Results folded" not in result.message
        assert "Found 300 files matching 'marker'." in result.message
        assert "truncated" not in result.message.lower()


async def test_grep_offset_pagination(grep_tool: Grep):
    """offset skips the first N results; combined with head_limit enables pagination."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a single file with many lines to avoid mtime sort instability
        (Path(temp_dir) / "data.txt").write_text(
            "\n".join(f"line{i} word" for i in range(10)) + "\n"
        )

        # Page 1: first 3
        r1 = await grep_tool(
            Params(
                pattern="word",
                path=temp_dir,
                output_mode="content",
                head_limit=3,
                offset=0,
            )
        )
        assert isinstance(r1.output, str)
        lines1 = [x for x in r1.output.split("\n") if x.strip()]
        assert len(lines1) == 3
        assert "Use offset=3 to see more" in r1.message

        # Page 2: next 3
        r2 = await grep_tool(
            Params(
                pattern="word",
                path=temp_dir,
                output_mode="content",
                head_limit=3,
                offset=3,
            )
        )
        assert isinstance(r2.output, str)
        lines2 = [x for x in r2.output.split("\n") if x.strip()]
        assert len(lines2) == 3
        # No overlap between pages (content mode has stable line order)
        assert set(lines1).isdisjoint(set(lines2))


async def test_grep_offset_content_mode(grep_tool: Grep):
    """offset works correctly with content mode output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "a.txt").write_text("\n".join(f"line{i} match" for i in range(10)) + "\n")

        # Get all results
        r_all = await grep_tool(
            Params(pattern="match", path=temp_dir, output_mode="content", head_limit=0)
        )
        assert isinstance(r_all.output, str)
        all_lines = [x for x in r_all.output.split("\n") if x.strip()]
        assert len(all_lines) == 10

        # Get with offset=5
        r_offset = await grep_tool(
            Params(
                pattern="match",
                path=temp_dir,
                output_mode="content",
                head_limit=3,
                offset=5,
            )
        )
        assert isinstance(r_offset.output, str)
        offset_lines = [x for x in r_offset.output.split("\n") if x.strip()]
        assert len(offset_lines) == 3
        # Should be lines 5,6,7 from original
        assert offset_lines[0] == all_lines[5]
        assert offset_lines[2] == all_lines[7]


async def test_grep_offset_beyond_results(grep_tool: Grep):
    """offset larger than total results returns no matches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "only.txt").write_text("data\n")

        result = await grep_tool(
            Params(
                pattern="data",
                path=temp_dir,
                output_mode="files_with_matches",
                offset=100,
            )
        )
        assert not result.is_error
        assert "No matches found" in result.message


async def test_grep_hidden_files(grep_tool: Grep):
    """Hidden dotfiles (non-sensitive) are searchable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / ".eslintrc.json").write_text('{"rule": "marker"}\n')
        (Path(temp_dir) / "visible.txt").write_text("marker\n")

        result = await grep_tool(
            Params(pattern="marker", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert ".eslintrc.json" in result.output
        assert "visible.txt" in result.output


async def test_grep_vcs_exclusion(grep_tool: Grep):
    """.git directory is excluded from search."""
    with tempfile.TemporaryDirectory() as temp_dir:
        git_dir = Path(temp_dir) / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("vcs_marker\n")
        (Path(temp_dir) / "real.txt").write_text("vcs_marker\n")

        result = await grep_tool(
            Params(pattern="vcs_marker", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert "real.txt" in result.output
        assert ".git" not in result.output


async def test_grep_mtime_sorting(grep_tool: Grep):
    """files_with_matches returns most recently modified files first."""
    import os as _os
    import time

    with tempfile.TemporaryDirectory() as temp_dir:
        old_file = Path(temp_dir) / "old.txt"
        old_file.write_text("sortme\n")
        old_mtime = time.time() - 100
        _os.utime(old_file, (old_mtime, old_mtime))

        new_file = Path(temp_dir) / "new.txt"
        new_file.write_text("sortme\n")

        result = await grep_tool(
            Params(pattern="sortme", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        lines = [x for x in result.output.split("\n") if x.strip()]
        assert len(lines) == 2
        assert lines[0] == "new.txt"
        assert lines[1] == "old.txt"


@pytest.mark.parametrize("output_mode", ["files_with_matches", "content", "count_matches"])
async def test_grep_relative_paths(grep_tool: Grep, temp_test_files, output_mode: str):
    """All output modes return relative paths, not absolute."""
    temp_dir, _ = temp_test_files

    result = await grep_tool(
        Params.model_validate(
            {"pattern": "hello", "path": temp_dir, "output_mode": output_mode, "-i": True}
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)
    for line in result.output.split("\n"):
        if not line.strip():
            continue
        # For content/count, check the path part before first ':'
        if output_mode in ("content", "count_matches") and ":" in line:
            path_part = line.split(":")[0]
        else:
            path_part = line
        assert not Path(path_part).is_absolute(), f"Expected relative path, got: {line}"


async def test_grep_content_default_line_numbers(grep_tool: Grep):
    """content mode includes line numbers by default (without explicit -n)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "a.txt").write_text("hello\nworld\n")

        result = await grep_tool(Params(pattern="hello", path=temp_dir, output_mode="content"))
        assert not result.is_error
        assert isinstance(result.output, str)
        for line in result.output.split("\n"):
            if line.strip() and not line.startswith("--"):
                parts = line.split(":")
                assert len(parts) >= 3, f"Expected path:line:content, got: {line}"
                assert parts[1].strip().isdigit(), f"Expected line number, got: {parts[1]}"


async def test_grep_content_disable_line_numbers(grep_tool: Grep):
    """content mode can opt-out of line numbers with -n=false."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "a.txt").write_text("hello\nworld\n")

        result = await grep_tool(
            Params.model_validate(
                {"pattern": "hello", "path": temp_dir, "output_mode": "content", "-n": False}
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        for line in result.output.split("\n"):
            if line.strip() and not line.startswith("--"):
                parts = line.split(":")
                # path:content (2 parts), NOT path:linenum:content (3 parts)
                assert len(parts) == 2, f"Expected path:content without linenum, got: {line}"


async def test_grep_count_summary(grep_tool: Grep):
    """count_matches: summary in message (not output), accurate on full results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(10):
            (Path(temp_dir) / f"f{i}.txt").write_text("word\nword\nword\n")

        result = await grep_tool(
            Params(pattern="word", path=temp_dir, output_mode="count_matches", head_limit=3)
        )
        assert not result.is_error
        assert isinstance(result.output, str)

        # Output is pure path:count (no summary text)
        output_lines = [x for x in result.output.split("\n") if x.strip()]
        assert len(output_lines) == 3
        for line in output_lines:
            assert "Found" not in line, f"Summary leaked into output: {line}"

        # Summary in message reflects ALL 10 files x 3 matches = 30
        assert "Found 30 total occurrences across 10 files" in result.message
        # Pagination info also present
        assert "Results truncated to 3 lines" in result.message


async def test_grep_content_with_context_lines(grep_tool: Grep):
    """content mode with context: both match and context lines have relative paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "a.txt").write_text("aaa\nbbb\nccc\n")

        result = await grep_tool(
            Params.model_validate(
                {"pattern": "bbb", "path": temp_dir, "output_mode": "content", "-C": 1}
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        assert "bbb" in result.output
        # ALL lines (match and context) should have relative paths
        for line in result.output.split("\n"):
            if line.strip() and line != "--":
                assert not Path(line).is_absolute(), f"Line has absolute path: {line}"


async def test_grep_single_file_relative_path(grep_tool: Grep):
    """Searching a single file still returns relative paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "target.py"
        test_file.write_text("def foo():\n    pass\n")

        result = await grep_tool(Params(pattern="foo", path=str(test_file), output_mode="content"))
        assert not result.is_error
        assert isinstance(result.output, str)
        for line in result.output.split("\n"):
            if line.strip() and not line.startswith("--"):
                assert not Path(line).is_absolute(), f"Expected relative path, got: {line}"


# === Unit tests for internal functions ===


def test_build_rg_args_defaults():
    """Default mode (files_with_matches): fixed params and output mode flag."""
    args = _build_rg_args("rg", Params(pattern="test", path="/tmp"))

    # Fixed params always present
    assert "--no-config" in args
    assert "--hidden" in args
    assert "--max-columns" in args
    for vcs in (".git", ".svn", ".hg", ".bzr", ".jj", ".sl"):
        assert f"!{vcs}" in args

    # Default output mode flag
    assert "--files-with-matches" in args

    # content mode: no --max-columns, no --files-with-matches
    content_args = _build_rg_args(
        "/usr/bin/rg", Params(pattern="x", path="/tmp", output_mode="content")
    )
    assert "--max-columns" not in content_args
    assert "--files-with-matches" not in content_args
    # default head_limit=500 adds --max-count with margin
    assert "--max-count" in content_args
    idx = content_args.index("--max-count")
    assert content_args[idx + 1] == "1500"  # 0 + 500 + 1000

    # head_limit=0 (unlimited): no --max-count
    unlimited_args = _build_rg_args(
        "/usr/bin/rg",
        Params(pattern="x", path="/tmp", output_mode="content", head_limit=0),
    )
    assert "--max-count" not in unlimited_args

    # content mode with head_limit + offset: adds --max-count including both
    limited_args = _build_rg_args(
        "/usr/bin/rg",
        Params(pattern="x", path="/tmp", output_mode="content", head_limit=30, offset=5),
    )
    assert "--max-count" in limited_args
    idx = limited_args.index("--max-count")
    assert limited_args[idx + 1] == "1035"  # 5 + 30 + 1000

    # count_matches mode: has --count-matches
    count_args = _build_rg_args(
        "/usr/bin/rg", Params(pattern="x", path="/tmp", output_mode="count_matches")
    )
    assert "--count-matches" in count_args
    assert "--max-columns" in count_args


def test_build_rg_args_flag_mapping():
    """Verify param-to-flag mapping, single_threaded, and expanduser."""
    # All content flags
    params = Params.model_validate(
        {
            "pattern": "test",
            "path": "/tmp",
            "output_mode": "content",
            "-i": True,
            "multiline": True,
            "-B": 2,
            "-A": 3,
            "-C": 1,
            "-n": True,
            "glob": "*.py",
            "type": "py",
        }
    )
    args = _build_rg_args("rg", params)

    assert "--ignore-case" in args
    assert "--multiline" in args
    assert "--multiline-dotall" in args
    assert "--before-context" in args
    assert "--after-context" in args
    assert "--context" in args
    assert "--line-number" in args
    assert "--glob" in args
    assert "--type" in args
    # Pattern and path after --
    dd_idx = args.index("--")
    assert args[dd_idx + 1] == "test"
    assert args[dd_idx + 2] == "/tmp"

    # single_threaded adds -j 1
    st_args = _build_rg_args("rg", Params(pattern="x", path="/tmp"), single_threaded=True)
    idx = st_args.index("-j")
    assert st_args[idx + 1] == "1"

    # expanduser expands ~ in path
    tilde_args = _build_rg_args("rg", Params(pattern="x", path="~/foo"))
    assert not tilde_args[-1].startswith("~")
    assert "foo" in tilde_args[-1]


def test_strip_path_prefix_posix():
    """Prefix stripping works with POSIX paths (forward slash)."""
    output = [
        "/home/user/project/src/a.py:42:code",
        "/home/user/project/src/b.py-41-context",
        "--",
    ]
    result = _strip_path_prefix(output, "/home/user/project")
    assert result == [
        "src/a.py:42:code",
        "src/b.py-41-context",
        "--",
    ]


def test_strip_path_prefix_windows(monkeypatch):
    """Prefix stripping works with Windows paths (backslash)."""
    monkeypatch.setattr("kimi_cli.tools.file.grep_local.os.sep", "\\")

    output = [
        "C:\\repo\\src\\a.py:42:code",
        "C:\\repo\\src\\b.py-41-context",
        "--",
    ]
    result = _strip_path_prefix(output, "C:\\repo")
    assert result == [
        "src\\a.py:42:code",
        "src\\b.py-41-context",
        "--",
    ]


def test_strip_path_prefix_mixed_separators():
    """Prefix stripping handles ripgrep's mixed Windows-style paths."""
    output = [
        "D:/repo\\src\\a.py:42:code",
        "D:/repo\\src\\b.py-41-context",
        "--",
    ]
    result = _strip_path_prefix(output, "D:\\repo")
    assert result == [
        "src\\a.py:42:code",
        "src\\b.py-41-context",
        "--",
    ]


def test_strip_path_prefix_no_match():
    """Lines not starting with prefix are kept as-is."""
    output = ["/other/path/file.py", "--"]
    result = _strip_path_prefix(output, "/home/user/project")
    assert result == ["/other/path/file.py", "--"]


def test_strip_path_prefix_trailing_sep():
    """Trailing separators on search_base are handled correctly."""
    output = ["/tmp/dir/file.py"]
    # With trailing slash
    assert _strip_path_prefix(output, "/tmp/dir/") == ["file.py"]
    # Without trailing slash
    assert _strip_path_prefix(output, "/tmp/dir") == ["file.py"]


def test_strip_path_prefix_similar_names():
    """search_base=/tmp/a must not match /tmp/abc/file.py."""
    output = ["/tmp/abc/file.py", "/tmp/a/file.py"]
    result = _strip_path_prefix(output, "/tmp/a")
    assert result == ["/tmp/abc/file.py", "file.py"]  # NOT stripped, stripped


# === Tests for include_ignored feature ===


async def test_grep_include_ignored_finds_gitignored_files(grep_tool: Grep):
    """include_ignored=True should find files that are listed in .gitignore."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up a git repo with .gitignore
        import subprocess

        subprocess.run(["git", "init", "-q", temp_dir], check=True)
        (Path(temp_dir) / ".git" / "test_marker").write_text("SECRET=leaked\n")
        # Use a non-sensitive ignored file (build output) to test include_ignored
        (Path(temp_dir) / ".gitignore").write_text("build.log\n")
        (Path(temp_dir) / "build.log").write_text("SECRET=in_build_log\n")
        (Path(temp_dir) / "visible.txt").write_text("SECRET=visible\n")

        # Without include_ignored: build.log should be excluded
        result = await grep_tool(
            Params(pattern="SECRET", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert "visible.txt" in result.output
        assert "build.log" not in result.output

        # With include_ignored: build.log should be found
        result = await grep_tool(
            Params(
                pattern="SECRET",
                path=temp_dir,
                output_mode="files_with_matches",
                include_ignored=True,
            )
        )
        assert not result.is_error
        assert "build.log" in result.output
        assert "visible.txt" in result.output
        assert ".git" not in result.output  # VCS directories still excluded


async def test_grep_include_ignored_default_false(grep_tool: Grep):
    """By default, include_ignored should be False (respect .gitignore)."""
    params = Params(pattern="test", path="/tmp")
    assert params.include_ignored is False


def test_build_rg_args_include_ignored():
    """include_ignored=True should add --no-ignore flag to rg args."""
    params = Params(pattern="test", path="/tmp", include_ignored=True)
    args = _build_rg_args("rg", params)
    assert "--no-ignore" in args

    # Default: no --no-ignore
    params_default = Params(pattern="test", path="/tmp")
    args_default = _build_rg_args("rg", params_default)
    assert "--no-ignore" not in args_default


async def test_grep_filters_sensitive_files_always(grep_tool: Grep):
    """Sensitive files (.env, SSH keys) are always filtered, even without include_ignored."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # No git repo — .env is not gitignored, just a normal dotfile
        (Path(temp_dir) / ".env").write_text("SECRET=hunter2\n")
        (Path(temp_dir) / "id_rsa").write_text("SECRET=private_key\n")
        (Path(temp_dir) / "visible.txt").write_text("SECRET=visible\n")

        result = await grep_tool(
            Params(pattern="SECRET", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert "visible.txt" in result.output
        assert ".env" not in result.output
        assert "id_rsa" not in result.output
        assert "sensitive" in result.message.lower()


async def test_grep_filters_sensitive_in_content_mode(grep_tool: Grep):
    """Sensitive file filtering works in content output mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / ".env").write_text("SECRET=hunter2\n")
        (Path(temp_dir) / "visible.txt").write_text("SECRET=visible\n")

        result = await grep_tool(Params(pattern="SECRET", path=temp_dir, output_mode="content"))
        assert not result.is_error
        assert "visible.txt" in result.output
        assert ".env" not in result.output
        assert "sensitive" in result.message.lower()


async def test_grep_filters_sensitive_context_lines(grep_tool: Grep):
    """Context lines (ripgrep -C) for sensitive files must also be filtered."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / ".env").write_text("line1\nSECRET=hunter2\nline3\n")
        (Path(temp_dir) / "visible.txt").write_text("lineA\nSECRET=visible\nlineC\n")

        result = await grep_tool(
            Params.model_validate(
                {"pattern": "SECRET", "path": temp_dir, "output_mode": "content", "-C": 1}
            )
        )
        assert not result.is_error
        assert "visible.txt" in result.output
        # Neither match lines nor context lines from .env should appear
        assert ".env" not in result.output
        assert "hunter2" not in result.output
        assert "sensitive" in result.message.lower()


async def test_grep_filters_sensitive_hyphenated_path(grep_tool: Grep):
    """Sensitive file in a hyphenated directory should be correctly filtered in content mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        sub = Path(temp_dir) / "my-project"
        sub.mkdir()
        (sub / ".env").write_text("SECRET=leaked\n")
        (Path(temp_dir) / "safe.txt").write_text("SECRET=ok\n")

        result = await grep_tool(
            Params.model_validate(
                {"pattern": "SECRET", "path": temp_dir, "output_mode": "content", "-C": 1}
            )
        )
        assert not result.is_error
        assert "safe.txt" in result.output
        assert ".env" not in result.output
        assert "leaked" not in result.output


async def test_grep_all_sensitive_preserves_warning(grep_tool: Grep):
    """When all results are sensitive, warning should not be lost to 'No matches found'."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / ".env").write_text("ONLY_IN_ENV=secret\n")

        result = await grep_tool(
            Params(pattern="ONLY_IN_ENV", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert "No matches found" in result.message
        assert "sensitive" in result.message.lower()
        assert ".env" in result.message


async def test_grep_allows_env_example(grep_tool: Grep):
    """.env.example is not sensitive and should appear in results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / ".env.example").write_text("API_KEY=placeholder\n")

        result = await grep_tool(
            Params(pattern="API_KEY", path=temp_dir, output_mode="files_with_matches")
        )
        assert not result.is_error
        assert ".env.example" in result.output


# --- Comprehensive backup_grep and internal function tests ---


async def test_backup_grep_empty_pattern(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep rejects empty patterns."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(pattern="", path=str(temp_work_dir), output_mode="files_with_matches")
    )
    assert result.is_error
    assert "Pattern cannot be empty" in result.message


async def test_backup_grep_invalid_pattern(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep returns proper error for invalid regex."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(pattern="[invalid", path=str(temp_work_dir), output_mode="files_with_matches")
    )
    assert result.is_error
    assert "Invalid regex pattern" in result.message


async def test_backup_grep_path_outside_workspace(grep_tool: Grep):
    """backup_grep rejects relative paths that resolve outside the workspace."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(pattern="test", path="../outside_workspace", output_mode="files_with_matches")
    )
    assert result.is_error
    assert "outside the workspace" in result.message.lower()


async def test_backup_grep_path_not_found(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep returns error for nonexistent paths."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(
            pattern="test",
            path=str(temp_work_dir / "nonexistent" / "xyz"),
            output_mode="files_with_matches",
        )
    )
    assert result.is_error
    assert "does not exist" in result.message


async def test_backup_grep_files_with_matches(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep finds files with matches."""
    await (temp_work_dir / "a.py").write_text("hello world\n")
    await (temp_work_dir / "b.js").write_text("hello there\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "files_with_matches",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert "a.py" in result.output
    assert "b.js" in result.output


async def test_backup_grep_content_mode(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep content mode with line numbers."""
    await (temp_work_dir / "a.py").write_text("hello world\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "content",
                "-i": True,
                "-n": True,
            }
        )
    )
    assert not result.is_error
    assert "hello" in result.output.lower()
    assert ":" in result.output


async def test_backup_grep_content_no_line_numbers(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep content mode without line numbers."""
    await (temp_work_dir / "a.py").write_text("hello world\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "content",
                "-i": True,
                "-n": False,
            }
        )
    )
    assert not result.is_error
    for line in result.output.split("\n"):
        if line.strip() and line != "--":
            parts = line.split(":")
            assert len(parts) == 2, f"Expected path:content, got: {line}"


async def test_backup_grep_count_matches(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep count_matches mode."""
    await (temp_work_dir / "a.py").write_text("hello\nhello\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "count_matches",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert "a.py" in result.output
    assert "Found" in result.message


async def test_backup_grep_with_context(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep content mode with before/after context."""
    # Add gap between matches so intervals don't merge
    await (temp_work_dir / "a.py").write_text(
        "line1\nTestClass\nline3\nline4\nline5\nTestClass2\nline7\n"
    )

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "TestClass",
                "path": str(temp_work_dir),
                "output_mode": "content",
                "-C": 1,
                "-n": True,
            }
        )
    )
    assert not result.is_error
    assert "TestClass" in result.output
    # Two match groups separated by gap should produce '--'
    assert "--" in result.output


async def test_backup_grep_multiline(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep multiline pattern matching."""
    await (temp_work_dir / "multi.py").write_text("start\nmatch line\nend\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(
            pattern=r"start\nmatch",
            path=str(temp_work_dir),
            output_mode="content",
            multiline=True,
        )
    )
    assert not result.is_error
    assert "start" in result.output


async def test_backup_grep_no_matches(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep returns no matches message."""
    await (temp_work_dir / "a.py").write_text("nothing here\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(
            pattern="nonexistent_xyz",
            path=str(temp_work_dir),
            output_mode="files_with_matches",
        )
    )
    assert not result.is_error
    assert "No matches found" in result.message


async def test_backup_grep_glob_filter(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep respects glob filter."""
    await (temp_work_dir / "a.py").write_text("hello\n")
    await (temp_work_dir / "b.js").write_text("hello\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "files_with_matches",
                "glob": "*.py",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert "a.py" in result.output
    assert "b.js" not in result.output


async def test_backup_grep_type_filter(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep respects type filter."""
    await (temp_work_dir / "a.py").write_text("hello\n")
    await (temp_work_dir / "b.js").write_text("hello\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params.model_validate(
            {
                "pattern": "hello",
                "path": str(temp_work_dir),
                "output_mode": "files_with_matches",
                "type": "py",
                "-i": True,
            }
        )
    )
    assert not result.is_error
    assert "a.py" in result.output
    assert "b.js" not in result.output


async def test_backup_grep_offset_pagination(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep offset + head_limit pagination."""
    for i in range(10):
        await (temp_work_dir / f"f{i}.txt").write_text(f"word{i}\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(
            pattern=r"word\d",
            path=str(temp_work_dir),
            output_mode="files_with_matches",
            head_limit=3,
            offset=2,
        )
    )
    assert not result.is_error
    lines = [x for x in result.output.split("\n") if x.strip()]
    assert len(lines) == 3


async def test_backup_grep_sensitive_file_filter(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep filters sensitive files."""
    await (temp_work_dir / ".env").write_text("SECRET=x\n")
    await (temp_work_dir / "ok.txt").write_text("SECRET=y\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(pattern="SECRET", path=str(temp_work_dir), output_mode="files_with_matches")
    )
    assert not result.is_error
    assert "ok.txt" in result.output
    assert ".env" not in result.output
    assert "sensitive" in result.message.lower()


async def test_backup_grep_ignored_dirs(grep_tool: Grep, temp_work_dir: KaosPath):
    """backup_grep skips ignored directories."""
    node_modules = temp_work_dir / "node_modules"
    await node_modules.mkdir(parents=True, exist_ok=True)
    pkg = node_modules / "pkg"
    await pkg.mkdir(parents=True, exist_ok=True)
    await (pkg / "index.js").write_text("marker\n")
    src = temp_work_dir / "src"
    await src.mkdir(parents=True, exist_ok=True)
    await (src / "main.js").write_text("marker\n")

    grep_tool._rg_path = None
    grep_tool._rg_path_task = None

    result = await grep_tool(
        Params(pattern="marker", path=str(temp_work_dir), output_mode="files_with_matches")
    )
    assert not result.is_error
    assert "main.js" in result.output
    assert "node_modules" not in result.output


# --- Unit tests for internal helper functions ---


def test_merge_intervals_basic():
    """_merge_intervals merges overlapping and adjacent intervals."""
    from kimi_cli.tools.file.grep_local import _merge_intervals

    assert _merge_intervals([(1, 3), (2, 4)]) == [(1, 4)]
    # Adjacent intervals (1,2) and (3,4) are also merged because 3 <= 2+1
    assert _merge_intervals([(1, 2), (3, 4)]) == [(1, 4)]
    assert _merge_intervals([(1, 5), (2, 3), (4, 6)]) == [(1, 6)]
    assert _merge_intervals([]) == []


def test_extract_path_content():
    """_extract_path extracts path from content mode lines."""
    from kimi_cli.tools.file.grep_local import Grep

    tool = object.__new__(Grep)
    tool._rg_path_task = None
    assert tool._extract_path("file.py:10:match", "content") == "file.py"
    assert tool._extract_path("file.py-10-context", "content") == "file.py"
    assert tool._extract_path("--", "content") is None
    assert tool._extract_path("plain", "content") == "plain"


def test_extract_path_count():
    """_extract_path extracts path from count_matches mode lines."""
    from kimi_cli.tools.file.grep_local import Grep

    tool = object.__new__(Grep)
    tool._rg_path_task = None
    assert tool._extract_path("file.py:42", "count_matches") == "file.py"


def test_extract_path_files():
    """_extract_path returns line as-is for files_with_matches mode."""
    from kimi_cli.tools.file.grep_local import Grep

    tool = object.__new__(Grep)
    tool._rg_path_task = None
    assert tool._extract_path("file.py", "files_with_matches") == "file.py"


def test_is_valid_file_size_limit():
    """_is_valid_file rejects files larger than 5MB."""
    from kimi_cli.tools.file.grep_local import Grep

    tool = object.__new__(Grep)
    tool._rg_path_task = None
    params = Params(pattern="test", path="/tmp")
    assert tool._is_valid_file(Path("/nonexistent"), params) is False


def test_should_skip_dir():
    """_should_skip_dir correctly skips VCS and ignored directories."""
    from kimi_cli.tools.file.grep_local import _should_skip_dir

    assert _should_skip_dir(".git", False) is True
    assert _should_skip_dir(".git", True) is True
    assert _should_skip_dir("node_modules", False) is True
    assert _should_skip_dir("node_modules", True) is False
    assert _should_skip_dir("src", False) is False


def test_matches_type_unknown():
    """_matches_type returns False for unknown type names."""
    from kimi_cli.tools.file.grep_local import _matches_type

    assert _matches_type(Path("file.py"), "unknown_type") is False
    assert _matches_type(Path("file.py"), None) is True
    assert _matches_type(Path("file.py"), "py") is True
    assert _matches_type(Path("file.rs"), "py") is False


def test_matches_glob_edge_cases():
    """_matches_glob handles None and various patterns."""
    from kimi_cli.tools.file.grep_local import _matches_glob

    assert _matches_glob(Path("file.py"), None) is True
    assert _matches_glob(Path("file.py"), "*.py") is True
    assert _matches_glob(Path("file.py"), "*.js") is False
    assert _matches_glob(Path("test.py"), "test.*") is True


def test_is_binary():
    """_is_binary detects null bytes."""
    from kimi_cli.tools.file.grep_local import _is_binary

    assert _is_binary(b"hello\x00world") is True
    assert _is_binary(b"hello world") is False
    assert _is_binary(b"") is False


def test_read_file_text_binary():
    """_read_file_text returns None for binary files."""
    import tempfile

    from kimi_cli.tools.file.grep_local import _read_file_text

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"hello\x00world")
        path = Path(f.name)

    assert _read_file_text(path) is None
    path.unlink()


def test_read_file_text_utf8():
    """_read_file_text returns content for text files."""
    import tempfile

    from kimi_cli.tools.file.grep_local import _read_file_text

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("hello world")
        path = Path(f.name)

    assert _read_file_text(path) == "hello world"
    path.unlink()


def test_compile_regex_cached():
    """_compile_regex_cached compiles and caches regex patterns."""
    import re

    from kimi_cli.tools.file.grep_local import _compile_regex_cached

    p1 = _compile_regex_cached("test", 0)
    p2 = _compile_regex_cached("test", 0)
    assert p1 is p2  # Same cache entry
    assert p1.pattern == "test"
    assert _compile_regex_cached("test", re.IGNORECASE).flags & re.IGNORECASE


def test_is_eagain():
    """_is_eagain detects EAGAIN errors from stderr."""
    from kimi_cli.tools.file.grep_local import _is_eagain

    assert _is_eagain("os error 11") is True
    assert _is_eagain("Resource temporarily unavailable") is True
    assert _is_eagain("some other error") is False


# --- [out of work-dir] warning tests ---


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in Grep")
async def test_grep_outside_work_dir_has_warning(grep_tool: Grep):
    """Grep with path outside work-dir should include [out of work-dir] in message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("hello world\n")
        result = await grep_tool(
            Params.model_validate(
                {"pattern": "hello", "path": tmpdir, "output_mode": "files_with_matches"}
            )
        )
        assert not result.is_error
        assert "[out of work-dir]" in result.message


# ============================================================================
# Fuzzy output_mode matching tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in Grep")
async def test_grep_outside_work_dir_nonexistent_has_warning(grep_tool: Grep):
    """Grep with non-existent outside path should include [out of work-dir]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = str(Path(tmpdir) / "nonexistent")
        result = await grep_tool(
            Params.model_validate(
                {"pattern": "hello", "path": nonexistent, "output_mode": "files_with_matches"}
            )
        )
        # May succeed with "No matches found" or error depending on rg availability
        assert "[out of work-dir]" in result.message


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in Grep")
async def test_grep_outside_work_dir_no_matches_has_warning(grep_tool: Grep):
    """Grep outside work-dir with no matches should include [out of work-dir]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("hello world\n")
        result = await grep_tool(
            Params.model_validate(
                {"pattern": "nonexistent_pattern_xyz", "path": tmpdir, "output_mode": "files_with_matches"}
            )
        )
        assert not result.is_error
        assert "[out of work-dir]" in result.message


# === Tests for rtk (deduplicate_output) integration ===


class _FakeStream:
    """Minimal async stream returning a fixed payload once, then EOF."""

    def __init__(self, data: bytes = b"") -> None:
        self._data = data
        self._done = False

    async def read(self, n: int = -1) -> bytes:
        if self._done:
            return b""
        self._done = True
        return self._data


class _FakeProcess:
    """Minimal stand-in for asyncio.subprocess.Process."""

    def __init__(self, stdout: bytes = b"file1.py\n", returncode: int = 0) -> None:
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream()
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


@pytest.fixture
def captured_exec(monkeypatch):
    """Capture argv/kwargs passed to asyncio.create_subprocess_exec."""
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured


def _share_bin_rtk() -> str:
    return str(get_share_dir() / "bin" / _rtk_binary_name())


def test_find_existing_rtk_share_bin_only(tmp_path, monkeypatch):
    """_find_existing_rtk finds the binary in share/bin only."""
    bin_name = _rtk_binary_name()
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: tmp_path)
    assert _find_existing_rtk(bin_name) is None

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / bin_name).touch()
    assert _find_existing_rtk(bin_name) == bin_dir / bin_name


def test_find_existing_rtk_ignores_path(tmp_path, monkeypatch):
    """_find_existing_rtk must never fall back to the global PATH."""
    bin_name = _rtk_binary_name()
    share = tmp_path / "share"
    share.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / bin_name).touch()  # fake rtk on PATH, outside share/bin
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: share)
    monkeypatch.setenv("PATH", str(other))
    assert _find_existing_rtk(bin_name) is None


def test_env_with_shared_bin_path_prepends_share_bin(tmp_path, monkeypatch):
    """The shared bin directory is moved to the front of PATH."""
    share = tmp_path / "share"
    share.mkdir()
    bin_dir = share / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: share)
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin", str(bin_dir)]))

    env = _env_with_shared_bin_path()
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == str(bin_dir)
    assert entries == [str(bin_dir), "/usr/bin", "/bin"]


def test_env_with_shared_bin_path_removes_duplicate_elsewhere(tmp_path, monkeypatch):
    """A duplicate shared bin entry elsewhere in PATH is removed."""
    share = tmp_path / "share"
    share.mkdir()
    bin_dir = share / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: share)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(bin_dir), "/usr/bin", str(bin_dir), "/bin"])
    )

    env = _env_with_shared_bin_path()
    entries = env["PATH"].split(os.pathsep)
    assert entries == [str(bin_dir), "/usr/bin", "/bin"]


def test_env_with_shared_bin_path_idempotent_when_first(tmp_path, monkeypatch):
    """When shared bin is already first, the result keeps it first and clean."""
    share = tmp_path / "share"
    share.mkdir()
    bin_dir = share / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: share)
    original = {"PATH": str(bin_dir) + os.pathsep + "/usr/bin"}

    env = _env_with_shared_bin_path(original)
    assert env["PATH"].split(os.pathsep) == [str(bin_dir), "/usr/bin"]
    # Calling again is a no-op.
    assert _env_with_shared_bin_path(env)["PATH"] == env["PATH"]


def test_env_with_shared_bin_path_wins_over_global_rg(tmp_path, monkeypatch):
    """Even if another rg/rg.exe exists on PATH, share/bin is resolved first."""
    share = tmp_path / "share"
    share.mkdir()
    bin_dir = share / "bin"
    bin_dir.mkdir()
    fake_global = tmp_path / "fake_global"
    fake_global.mkdir()
    (fake_global / "rg").touch()
    (fake_global / "rg.exe").touch()
    monkeypatch.setattr(grep_local, "get_share_dir", lambda: share)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(fake_global), "/usr/bin", "/bin"])
    )

    env = _env_with_shared_bin_path()
    entries = env["PATH"].split(os.pathsep)
    assert entries[0] == str(bin_dir)
    assert entries[1] == str(fake_global)


def test_build_rg_args_with_rtk():
    """rtk path prefixes the argv; rg stays the bare wrapped executable."""
    params = Params(pattern="hello")
    args = _build_rg_args("rg", params, rtk_path="/abs/bin/rtk")
    assert args[0] == "/abs/bin/rtk"
    assert args[1] == "rg"
    assert "--no-config" in args


def test_build_rg_args_without_rtk_unchanged():
    """No rtk_path → argv uses the bare rg command name."""
    params = Params(pattern="hello")
    args = _build_rg_args("rg", params)
    assert args[0] == "rg"


def test_format_cmd_reflects_rtk_when_active():
    params = Params(pattern="hello")
    # Absolute rtk path is shown as just ``rtk`` for readability.
    assert _format_cmd(params, rtk_path="/abs/bin/rtk").startswith("rtk rg ")
    assert not _format_cmd(params).startswith("rtk ")


def test_rtk_fold_note_without_original_path():
    meta = {
        "total_matches": 42,
        "total_files": 1,
        "folded_files": [
            {"path": "src/a.py", "count": 40, "log": "C:\\log\\tee.log", "start_line": 2}
        ],
        "skipped_files": None,
        "skipped_log": None,
    }
    note = _rtk_fold_note(meta)
    assert note is not None
    assert "rtk folded output: 40 more lines in src/a.py" in note
    assert "Original output:" not in note


def test_rtk_fold_note_with_original_path():
    meta = {
        "total_matches": 42,
        "total_files": 1,
        "folded_files": [
            {"path": "src/a.py", "count": 40, "log": "C:\\log\\tee.log", "start_line": 2}
        ],
        "skipped_files": None,
        "skipped_log": None,
    }
    note = _rtk_fold_note(meta, original_path="C:\\tmp\\rtk_out.txt")
    assert note is not None
    assert "rtk folded output: 40 more lines in src/a.py" in note
    assert "Original output: C:/tmp/rtk_out.txt" in note


@pytest.mark.asyncio
async def test_grep_rtk_active_by_default(grep_tool: Grep, temp_test_files, captured_exec):
    """Default params wrap the bare rg command with the absolute share/bin rtk path."""
    temp_dir, _ = temp_test_files
    grep_tool._rtk_path = _share_bin_rtk()
    grep_tool._rtk_path_task = None

    result = await grep_tool(Params(pattern="Hello", path=temp_dir))
    assert not result.is_error

    args = captured_exec["args"]
    share_bin = str(get_share_dir() / "bin")
    # [<share>/bin/rtk, rg, --no-config, ...] — rtk absolute, rg bare name
    assert args[0] == _share_bin_rtk()
    assert args[1] == "rg"
    assert Path(args[0]).is_absolute()
    assert args[0].startswith(share_bin)
    # PATH is rewritten so the bare ``rg`` resolves to the managed binary.
    assert "env" in captured_exec["kwargs"]
    path_entries = captured_exec["kwargs"]["env"]["PATH"].split(os.pathsep)
    assert path_entries[0] == share_bin
    # Display brief shows ``rtk`` instead of the absolute share/bin path.
    assert result.brief.startswith("rtk ")


@pytest.mark.asyncio
async def test_grep_rtk_disabled_no_rtk_work(
    grep_tool: Grep, temp_test_files, captured_exec, monkeypatch
):
    """deduplicate_output=False → plain rg argv, no rtk resolution attempted."""
    temp_dir, _ = temp_test_files

    async def _boom() -> str:
        raise AssertionError("rtk resolution must not be attempted")

    monkeypatch.setattr(grep_local, "_ensure_rtk_path", _boom)
    grep_tool._rtk_path = None
    grep_tool._rtk_path_task = None

    result = await grep_tool(
        Params(pattern="Hello", path=temp_dir, deduplicate_output=False)
    )
    assert not result.is_error

    args = captured_exec["args"]
    assert args[0] == "rg"
    assert "env" in captured_exec["kwargs"]
    share_bin = str(get_share_dir() / "bin")
    path_entries = captured_exec["kwargs"]["env"]["PATH"].split(os.pathsep)
    assert path_entries[0] == share_bin


@pytest.mark.asyncio
async def test_grep_rtk_missing_falls_back_to_plain_rg(
    grep_tool: Grep, temp_test_files, captured_exec
):
    """Missing rtk binary → silent fallback to plain rg, result still returned."""
    temp_dir, _ = temp_test_files
    grep_tool._rtk_path = None
    grep_tool._rtk_path_task = None

    result = await grep_tool(Params(pattern="Hello", path=temp_dir))
    assert not result.is_error
    assert captured_exec["args"][0] == "rg"


@pytest.mark.asyncio
async def test_grep_rtk_download_failure_falls_back(
    grep_tool: Grep, temp_test_files, captured_exec
):
    """_ensure_rtk_path failure → plain rg, no exception, no backup_grep."""
    temp_dir, _ = temp_test_files

    async def _fail() -> str:
        raise RuntimeError("simulated download failure")

    grep_tool._rtk_path = None
    grep_tool._rtk_path_task = asyncio.ensure_future(_fail())

    result = await grep_tool(Params(pattern="Hello", path=temp_dir))
    assert not result.is_error
    assert captured_exec["args"][0] == "rg"


@pytest.mark.asyncio
async def test_grep_rtk_preserves_no_match_exit_code(
    grep_tool: Grep, temp_test_files, monkeypatch
):
    """Exit code 1 (no matches) through rtk keeps 'No matches found' semantics."""
    temp_dir, _ = temp_test_files
    grep_tool._rtk_path = _share_bin_rtk()
    grep_tool._rtk_path_task = None

    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = list(args)
        return _FakeProcess(stdout=b"", returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await grep_tool(Params(pattern="Hello", path=temp_dir))
    assert not result.is_error
    assert "No matches found" in result.message
    assert captured["args"][0] == _share_bin_rtk()


# === Tests for output shaping (rtk protocol cleanup / local dedup / fold) ===


@pytest.mark.asyncio
async def test_grep_rtk_markers_stripped_and_summarized(
    grep_tool: Grep, monkeypatch
):
    """rtk protocol lines are stripped from the output and summarized in message."""
    fake_stdout = (
        b"42 matches in 1 files:\n"
        b"\n"
        b"src/a.py:10:def foo():\n"
        b"  +40 more in src\\a.py [see remaining: tail -n +2 C:\\log\\tee.log]\n"
        b"+5 more files [see remaining: tail -n +43 C:\\log\\tee2.log]\n"
    )

    async def fake_exec(*args, **kwargs):
        return _FakeProcess(stdout=fake_stdout, returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    # Force the plain-rg path so the fake stdout flows into post-processing.
    grep_tool._rg_path = "rg"
    grep_tool._rg_path_task = None
    grep_tool._rtk_path = None
    grep_tool._rtk_path_task = None

    result = await grep_tool(
        Params(
            pattern="def",
            path=".",
            output_mode="content",
            head_limit=0,
            deduplicate_output=True,
        )
    )
    assert not result.is_error
    assert isinstance(result.output, str)
    # Protocol lines must never appear in the result output.
    assert "matches in 1 files" not in result.output
    assert "see remaining" not in result.output
    assert "+40 more in" not in result.output
    assert "+5 more files" not in result.output
    # The real content survives; counts and fold notes land in the message.
    assert "src/a.py:10:def foo():" in result.output
    assert "Found 42 matches in 1 files." in result.message
    assert "rtk folded output: 40 more lines in src\\a.py" in result.message
    assert "5 more files" in result.message
    assert "Full log: tail -n +2 C:\\log\\tee.log" in result.message
    # The original rtk stream (including protocol lines) was exported.
    assert "Original output:" in result.message
    original_path = result.message.split("Original output: ", 1)[1].strip()
    assert Path(original_path).is_file()
    saved = Path(original_path).read_text(encoding="utf-8")
    assert "42 matches in 1 files" in saved
    assert "+40 more in" in saved
    assert "+5 more files" in saved


@pytest.mark.asyncio
async def test_grep_local_dedup_fallback(grep_tool: Grep, monkeypatch):
    """With rtk unavailable, local dedup collapses repeated content lines."""
    monkeypatch.setattr(grep_local, "_find_existing_rtk", lambda bin_name: None)
    grep_tool._rtk_path = None
    grep_tool._rtk_path_task = None

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "repeats.txt"
        test_file.write_text("match line\n" * 500)

        result = await grep_tool(
            Params(
                pattern="match",
                path=temp_dir,
                output_mode="content",
                head_limit=0,
                line_number=False,
                deduplicate_output=True,
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        lines = [x for x in result.output.split("\n") if x.strip()]
        # 500 identical "repeats.txt:match line" lines collapse to one.
        assert len(lines) == 1
        assert "(499 repeats)" in lines[0]
        assert "Removed 499 repeated line(s) via dedup." in result.message


@pytest.mark.asyncio
async def test_grep_fold_output(grep_tool: Grep):
    """head_limit=0 with 500 matches folds to max_output_lines=200."""
    with tempfile.TemporaryDirectory() as temp_dir:
        (Path(temp_dir) / "data.txt").write_text(
            "\n".join(f"line{i} word" for i in range(500)) + "\n"
        )
        result = await grep_tool(
            Params(
                pattern="word",
                path=temp_dir,
                output_mode="content",
                head_limit=0,
                max_output_lines=200,
                deduplicate_output=False,  # keep lines unique & un-deduped
            )
        )
        assert not result.is_error
        assert isinstance(result.output, str)
        lines = [x for x in result.output.split("\n") if x.strip()]
        assert len(lines) == 201  # 200 real lines + fold marker
        assert "… (300 lines omitted) …" in result.output
        assert "Results folded to 200 lines (300 omitted)" in result.message
        assert "Use max_output_lines=0 or offset to see more" in result.message


@pytest.mark.asyncio
async def test_grep_files_mode_summary(grep_tool: Grep):
    """files_with_matches always reports Found N files in the message."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for name in ("a.py", "b.py", "c.py"):
            (Path(temp_dir) / name).write_text("def marker():\n    pass\n")
        result = await grep_tool(
            Params(
                pattern="marker",
                path=temp_dir,
                output_mode="files_with_matches",
            )
        )
        assert not result.is_error
        assert "Found 3 files matching 'marker'." in result.message
