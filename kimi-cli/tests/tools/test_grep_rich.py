"""Integration tests for plan-23 rich grep (selectors/archive/recorder/grouped).

All pre-existing tests live in test_grep.py (unchanged). These cases exercise
the new behaviors end-to-end through the Grep tool. Note: selector/archive
searches auto-enable grouped output, so body lines are ``*N|text`` /
`` N|text`` under a ``# path`` header.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from kimi_cli.session import Session
from kimi_cli.tools.file.grep_local import Grep, Params
from kimi_cli.tools.file.grep_recorder import get_recorded_grep_files


@pytest.fixture
def sample_dir():
    """Files with SPARSE matches: TODO only on lines 2, 5, 8 (rg --max-count
    must be able to reach every window)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        body = []
        for i in range(1, 11):
            body.append(f"line{i} TODO" if i in (2, 5, 8) else f"line{i}")
        (Path(temp_dir) / "sample.py").write_text("\n".join(body) + "\n")
        (Path(temp_dir) / "other.py").write_text("TODO top\nno match\n")
        yield temp_dir


def _sel(sample_dir: str, name: str, sel: str) -> str:
    return f"{Path(sample_dir).as_posix()}/{name}:{sel}"


async def test_selector_range_content(grep_tool: Grep, sample_dir):
    """file.py:1-6 returns only matches within lines 1-6."""
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "1-6"), output_mode="content")
    )
    assert not result.is_error
    assert "*2|line2 TODO" in result.output
    assert "*5|line5 TODO" in result.output
    assert "line8 TODO" not in result.output


async def test_selector_plus_count(grep_tool: Grep, sample_dir):
    """sample.py:4+2 == lines 4-5 (contains the line-5 match only)."""
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "4+2"), output_mode="content")
    )
    assert not result.is_error
    assert "*5|line5 TODO" in result.output
    assert "line2 TODO" not in result.output
    assert "line8 TODO" not in result.output


async def test_selector_open_ended(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "7-"), output_mode="content")
    )
    assert not result.is_error
    assert "*8|line8 TODO" in result.output
    assert "line2 TODO" not in result.output
    assert "line5 TODO" not in result.output


async def test_selector_multi_range(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=_sel(sample_dir, "sample.py", "1-3,4-6"),
            output_mode="content",
        )
    )
    assert not result.is_error
    assert "line2 TODO" in result.output
    assert "line5 TODO" in result.output
    assert "line8 TODO" not in result.output


async def test_selector_multi_range_disjoint_windows(grep_tool: Grep, sample_dir):
    """1-2,8-9 returns both windows with the middle window excluded."""
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=_sel(sample_dir, "sample.py", "1-2,8-9"),
            output_mode="content",
        )
    )
    assert not result.is_error
    assert "line2 TODO" in result.output
    assert "line8 TODO" in result.output
    assert "line5 TODO" not in result.output


async def test_selector_context_clamped(grep_tool: Grep, sample_dir):
    """Context lines outside the selector window are dropped."""
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=_sel(sample_dir, "other.py", "1-1"),
            output_mode="content",
            context=2,
        )
    )
    assert not result.is_error
    assert "TODO top" in result.output
    assert "no match" not in result.output


async def test_invalid_selector_zero_start(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "0-5"), output_mode="content")
    )
    assert result.is_error
    assert "1-indexed" in result.message


async def test_invalid_selector_inverted(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "50-40"), output_mode="content")
    )
    assert result.is_error
    assert "end must be >= start" in result.message


async def test_selector_on_directory_rejected(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=f"{sample_dir}:1-5", output_mode="content")
    )
    assert result.is_error
    assert "requires a single file" in result.message


async def test_selector_on_glob_rejected(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=f"{sample_dir}/*.py:1-5", output_mode="content")
    )
    assert result.is_error
    assert "not a glob" in result.message


async def test_files_mode_with_selector_rejected(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=_sel(sample_dir, "sample.py", "1-5"),
            output_mode="files_with_matches",
        )
    )
    assert result.is_error
    assert "content" in result.message


async def test_grouped_true_content(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=sample_dir, output_mode="content", grouped=True)
    )
    assert not result.is_error
    assert "# " in result.output
    assert "*1|" in result.output or "*2|" in result.output


async def test_grouped_false_legacy(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=sample_dir, output_mode="content", grouped=False)
    )
    assert not result.is_error
    assert "# " not in result.output
    assert ":1:" in result.output or ":2:" in result.output


async def test_selector_auto_groups(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "2-4"), output_mode="content")
    )
    assert not result.is_error
    assert "# " in result.output


async def test_plain_search_not_grouped(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(pattern="TODO", path=sample_dir, output_mode="content")
    )
    assert not result.is_error
    assert "# " not in result.output


async def test_archive_search(grep_tool: Grep):
    with tempfile.TemporaryDirectory() as temp_dir:
        zpath = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("src/foo.ts", "export const x = 1;\nexport const y = 2;\n")
        rel = f"{zpath.as_posix()}:src/foo.ts"
        result = await grep_tool(
            Params(pattern="export", path=rel, output_mode="content")
        )
        assert not result.is_error
        assert "bundle.zip:src/foo.ts" in result.output
        assert "export const x" in result.output


async def test_archive_with_range(grep_tool: Grep):
    with tempfile.TemporaryDirectory() as temp_dir:
        zpath = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("src/foo.ts", "a\nb\nc\n")
        rel = f"{zpath.as_posix()}:src/foo.ts:2-2"
        result = await grep_tool(
            Params(pattern="[abc]", path=rel, output_mode="content")
        )
        assert not result.is_error
        assert "2|b" in result.output
        assert "1|a" not in result.output


async def test_archive_binary_member_skipped(grep_tool: Grep):
    with tempfile.TemporaryDirectory() as temp_dir:
        zpath = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("img.bin", b"\x00\x01binary")
        rel = f"{zpath.as_posix()}:img.bin"
        result = await grep_tool(
            Params(pattern="binary", path=rel, output_mode="content")
        )
        # all members unreadable -> ToolError with read guidance
        assert result.is_error
        assert "read" in result.message.lower()


async def test_archive_mixed_binary_text(grep_tool: Grep):
    with tempfile.TemporaryDirectory() as temp_dir:
        zpath = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("ok.txt", "hello archive\n")
            zf.writestr("img.bin", b"\x00binary")
        result = await grep_tool(
            Params(
                pattern="hello",
                path=[f"{zpath.as_posix()}:ok.txt", f"{zpath.as_posix()}:img.bin"],
                output_mode="content",
            )
        )
        assert not result.is_error
        assert "hello archive" in result.output
        assert "Skipped archive entries" in result.message


async def test_recorder_persists(session: Session, grep_tool: Grep, sample_dir):
    await grep_tool(Params(pattern="TODO", path=sample_dir, output_mode="content"))
    files = get_recorded_grep_files(session)
    assert len(files) >= 2
    assert any("sample.py" in f for f in files)
    assert any("other.py" in f for f in files)


async def test_recorder_content_mode(session: Session, grep_tool: Grep, sample_dir):
    await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "1-6"), output_mode="content")
    )
    files = get_recorded_grep_files(session)
    assert any("sample.py" in f for f in files)


async def test_recorder_disabled(session: Session, grep_tool: Grep, sample_dir):
    await grep_tool(
        Params(pattern="TODO", path=sample_dir, output_mode="content", record=False)
    )
    assert get_recorded_grep_files(session) == []


async def test_multi_entry_search(grep_tool: Grep, sample_dir):
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=f"{Path(sample_dir).as_posix()}/sample.py; {Path(sample_dir).as_posix()}/other.py",
            output_mode="content",
        )
    )
    assert not result.is_error
    assert "sample.py" in result.output
    assert "other.py" in result.output


async def test_backup_selector_parity(grep_tool: Grep, sample_dir):
    """Force the pure-Python fallback and repeat selector semantics."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None
    result = await grep_tool(
        Params(pattern="TODO", path=_sel(sample_dir, "sample.py", "1-6"), output_mode="content")
    )
    assert not result.is_error
    assert "line2 TODO" in result.output
    assert "line5 TODO" in result.output
    assert "line8 TODO" not in result.output


async def test_backup_files_mode_with_ranges(grep_tool: Grep, sample_dir):
    """Backup path can serve files_with_matches with ranges (scans the window)."""
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None
    result = await grep_tool(
        Params(
            pattern="TODO",
            path=_sel(sample_dir, "sample.py", "999-1000"),
            output_mode="files_with_matches",
        )
    )
    # No matches inside the window -> no file listed, but no error either.
    assert not result.is_error
    assert "sample.py" not in result.output


async def test_backup_archive_parity(grep_tool: Grep):
    grep_tool._rg_path = None
    grep_tool._rg_path_task = None
    with tempfile.TemporaryDirectory() as temp_dir:
        zpath = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("src/foo.ts", "needle here\n")
        rel = f"{zpath.as_posix()}:src/foo.ts"
        result = await grep_tool(
            Params(pattern="needle", path=rel, output_mode="content")
        )
        assert not result.is_error
        assert "bundle.zip:src/foo.ts" in result.output
        assert "needle" in result.output
