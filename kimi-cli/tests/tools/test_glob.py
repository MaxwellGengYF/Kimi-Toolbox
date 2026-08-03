"""Tests for the glob tool."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest
from kaos.path import KaosPath

from kimi_cli.tools.file.glob import MAX_MATCHES, Glob, Params


@pytest.fixture
async def test_files(temp_work_dir: KaosPath):
    """Create test files for glob testing."""
    # Create a directory structure
    await (temp_work_dir / "src" / "main").mkdir(parents=True)
    await (temp_work_dir / "src" / "test").mkdir(parents=True)
    await (temp_work_dir / "docs").mkdir()

    # Create test files
    await (temp_work_dir / "README.md").write_text("# README")
    await (temp_work_dir / "setup.py").write_text("setup")
    await (temp_work_dir / "src" / "main.py").write_text("main")
    await (temp_work_dir / "src" / "utils.py").write_text("utils")
    await (temp_work_dir / "src" / "main" / "app.py").write_text("app")
    await (temp_work_dir / "src" / "main" / "config.py").write_text("config")
    await (temp_work_dir / "src" / "test" / "test_app.py").write_text("test app")
    await (temp_work_dir / "src" / "test" / "test_config.py").write_text("test config")
    await (temp_work_dir / "docs" / "guide.md").write_text("guide")
    await (temp_work_dir / "docs" / "api.md").write_text("api")

    return temp_work_dir


async def test_glob_simple_pattern(glob_tool: Glob, test_files: KaosPath):
    """Test simple glob pattern matching."""
    result = await glob_tool(Params(pattern="*.py", directory=str(test_files)))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "setup.py" in result.output
    assert "Found 1 matches" in result.message


async def test_glob_multiple_matches(glob_tool: Glob, test_files: KaosPath):
    """Test glob pattern with multiple matches."""
    result = await glob_tool(Params(pattern="*.md", directory=str(test_files)))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "README.md" in result.output
    assert "Found 1 matches" in result.message


async def test_glob_recursive_pattern(glob_tool: Glob, test_files: KaosPath):
    """Test that recursive glob pattern starting with **/ works."""
    result = await glob_tool(Params(pattern="**/*.py", directory=str(test_files)))

    assert not result.is_error
    output = result.output.replace("\\", "/")
    assert "setup.py" in output
    assert "src/main.py" in output
    assert "src/main/app.py" in output
    assert "src/test/test_app.py" in output
    assert "Found 7 matches" in result.message


async def test_glob_safe_recursive_pattern(glob_tool: Glob, test_files: KaosPath):
    """Test safe recursive glob pattern that doesn't start with **/."""
    result = await glob_tool(Params(pattern="src/**/*.py", directory=str(test_files)))

    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")  # Normalize for Windows paths
    assert "src/main.py" in output
    assert "src/utils.py" in output
    assert "src/main/app.py" in output
    assert "src/main/config.py" in output
    assert "src/test/test_app.py" in output
    assert "src/test/test_config.py" in output
    assert "Found 6 matches" in result.message


async def test_glob_specific_directory(glob_tool: Glob, test_files: KaosPath):
    """Test glob pattern in specific directory."""
    src_dir = str(test_files / "src")
    result = await glob_tool(Params(pattern="*.py", directory=src_dir))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "main.py" in result.output
    assert "utils.py" in result.output
    assert "Found 2 matches" in result.message


async def test_glob_recursive_in_subdirectory(glob_tool: Glob, test_files: KaosPath):
    """Test recursive glob in subdirectory."""
    src_dir = str(test_files / "src")
    result = await glob_tool(Params(pattern="main/**/*.py", directory=src_dir))

    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")  # Normalize for Windows paths
    assert "main/app.py" in output
    assert "main/config.py" in output
    assert "Found 2 matches" in result.message


async def test_glob_test_files(glob_tool: Glob, test_files: KaosPath):
    """Test glob pattern for test files."""
    result = await glob_tool(Params(pattern="src/**/*test*.py", directory=str(test_files)))

    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")  # Normalize for Windows paths
    assert "src/test/test_app.py" in output
    assert "src/test/test_config.py" in output
    assert "Found 2 matches" in result.message


async def test_glob_no_matches(glob_tool: Glob, test_files: KaosPath):
    """Test glob pattern with no matches."""
    result = await glob_tool(Params(pattern="*.xyz", directory=str(test_files)))

    assert not result.is_error
    assert result.output == ""
    assert "No matches found" in result.message


async def test_glob_exclude_directories(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob with include_dirs=False."""
    # Create both files and directories
    await (temp_work_dir / "test_file.txt").write_text("content")
    await (temp_work_dir / "test_dir").mkdir()

    result = await glob_tool(
        Params(pattern="test_*", directory=str(temp_work_dir), include_dirs=False)
    )

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "test_file.txt" in result.output
    assert "test_dir" not in result.output
    assert "Found 1 matches" in result.message


async def test_glob_with_relative_path(glob_tool: Glob):
    """Test glob with relative path (should fail)."""
    result = await glob_tool(Params(pattern="*.py", directory="relative/path"))

    assert result.is_error
    assert "does not exist" in result.message


async def test_glob_tilde_path_expanded(glob_tool: Glob, temp_work_dir: KaosPath, monkeypatch: pytest.MonkeyPatch):
    """Test that ~ in directory path is expanded, not rejected as relative."""
    # Point the home dir at a tiny temp dir so `~` resolves there instead of
    # scanning the real (potentially huge) user profile, which can make this
    # test take tens of seconds (slow glob + slow Windows teardown).
    fake_home = Path(str(temp_work_dir)) / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    (fake_home / "tilde.txt").write_text("tilde")
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HOME", str(fake_home))

    # ~ expands to (the fake) home dir; glob searches it successfully
    result = await glob_tool(Params(pattern="*", directory="~/"))
    # Without expanduser() this would fail with "not an absolute path"
    assert "not an absolute path" not in result.message
    # Home directory exists and is searchable
    assert not result.is_error
    assert "tilde.txt" in result.output
    assert "Found 1 matches" in result.message


async def test_glob_outside_work_directory_nonexistent(glob_tool: Glob):
    """Test glob in nonexistent directory outside working directory."""
    dir = "/tmp/outside" if platform.system() != "Windows" else "C:/tmp/outside"
    result = await glob_tool(Params(pattern="*.py", directory=dir))

    assert result.is_error
    assert "does not exist" in result.message


async def test_glob_outside_work_directory_with_prefix(glob_tool: Glob, temp_work_dir: KaosPath):
    """Paths sharing the work dir prefix but outside are searchable if directory validation is not enforced."""
    base = Path(str(temp_work_dir))
    sneaky_dir = base.parent / f"{base.name}-sneaky"
    sneaky_dir.mkdir(parents=True, exist_ok=True)

    result = await glob_tool(Params(pattern="*.py", directory=str(sneaky_dir)))

    # Directory exists but is empty, so no matches
    assert not result.is_error
    assert "No matches found" in result.message


async def test_glob_nonexistent_directory(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob in nonexistent directory."""
    nonexistent_dir = str(temp_work_dir / "nonexistent")
    result = await glob_tool(Params(pattern="*.py", directory=nonexistent_dir))

    assert result.is_error
    assert "does not exist" in result.message


async def test_glob_not_a_directory(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob on a file instead of directory."""
    test_file = temp_work_dir / "test.txt"
    await test_file.write_text("content")

    result = await glob_tool(Params(pattern="*.py", directory=str(test_file)))

    assert result.is_error
    assert "is not a directory" in result.message


async def test_glob_single_character_wildcard(glob_tool: Glob, test_files: KaosPath):
    """Test single character wildcard."""
    result = await glob_tool(Params(pattern="?.md", directory=str(test_files)))

    assert not result.is_error
    assert result.output == ""
    # Should match single character .md files


async def test_glob_max_matches_limit(glob_tool: Glob, temp_work_dir: KaosPath):
    """Glob collects up to MAX_MATCHES and folds the display to max_results."""
    # Create more than MAX_MATCHES files
    for i in range(MAX_MATCHES + 50):
        await (temp_work_dir / f"file_{i}.txt").write_text(f"content {i}")
    result = await glob_tool(Params(pattern="*.txt", directory=str(temp_work_dir)))

    assert not result.is_error
    assert isinstance(result.output, str)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    # MAX_MATCHES are collected (count is exact), but the display folds to
    # max_results=500 (500 real lines + fold marker).
    real_lines = [ln for ln in output_lines if "lines omitted" not in ln]
    assert len(real_lines) == 500
    assert len(output_lines) == 501
    assert "… (500 lines omitted) …" in result.output
    assert f"Found {MAX_MATCHES} matches" in result.message
    assert "Showing 500 of 1000 (head+tail fold)" in result.message


async def test_glob_enhanced_double_star(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test enhanced ** pattern works recursively."""
    # Create some top-level files and directories for listing
    await (temp_work_dir / "file1.txt").write_text("content1")
    await (temp_work_dir / "file2.py").write_text("content2")
    await (temp_work_dir / "src").mkdir()
    await (temp_work_dir / "docs").mkdir()

    result = await glob_tool(Params(pattern="**/*.txt", directory=str(temp_work_dir)))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "file1.txt" in result.output
    assert "file2.py" not in result.output


async def test_glob_exactly_max_matches(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test behavior when exactly MAX_MATCHES files are found."""
    # Create exactly MAX_MATCHES files
    for i in range(MAX_MATCHES):
        await (temp_work_dir / f"test_{i}.py").write_text(f"code {i}")
    result = await glob_tool(Params(pattern="*.py", directory=str(temp_work_dir)))

    assert not result.is_error
    assert isinstance(result.output, str)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    # Count is exact (all MAX_MATCHES collected); display folds to 500.
    real_lines = [ln for ln in output_lines if "lines omitted" not in ln]
    assert len(real_lines) == 500
    assert len(output_lines) == 501
    assert f"Found {MAX_MATCHES} matches" in result.message
    assert "Showing 500 of 1000 (head+tail fold)" in result.message
    # Should NOT contain the capped-collection warning (exactly MAX_MATCHES).
    assert "Only the first" not in result.message
    assert "Search capped" not in result.message


async def test_glob_character_class(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test character class pattern."""
    await (temp_work_dir / "file1.py").write_text("content1")
    await (temp_work_dir / "file2.py").write_text("content2")
    await (temp_work_dir / "file3.txt").write_text("content3")
    result = await glob_tool(Params(pattern="file[1-2].py", directory=str(temp_work_dir)))

    assert not result.is_error
    assert isinstance(result.output, str)
    assert "file1.py" in result.output
    assert "file2.py" in result.output
    assert "file3.txt" not in result.output


async def test_glob_complex_pattern(glob_tool: Glob, test_files: KaosPath):
    """Test complex glob pattern combinations."""
    result = await glob_tool(Params(pattern="docs/**/main/*.py", directory=str(test_files)))

    assert not result.is_error
    assert result.output == ""
    # Should not match anything since there are no Python files in docs/main


async def test_glob_wildcard_with_double_star_patterns(glob_tool: Glob, test_files: KaosPath):
    """Test various patterns with ** that are allowed."""
    # Test pattern with ** at start works recursively
    result = await glob_tool(Params(pattern="**/main/*.py", directory=str(test_files)))

    assert not result.is_error
    output = result.output.replace("\\", "/")
    assert "src/main/app.py" in output
    assert "src/main/config.py" in output

    # Test pattern with ** not at the beginning
    result = await glob_tool(Params(pattern="src/**/test_*.py", directory=str(test_files)))

    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")  # Normalize for Windows paths
    assert "src/test/test_app.py" in output
    assert "src/test/test_config.py" in output


async def test_glob_pattern_edge_cases(glob_tool: Glob, test_files: KaosPath):
    """Test edge cases for pattern validation."""
    # Test pattern that has ** but not at the start
    result = await glob_tool(Params(pattern="src/**", directory=str(test_files)))
    assert not result.is_error

    # Test pattern that starts with * but not **
    result = await glob_tool(Params(pattern="*.py", directory=str(test_files)))
    assert not result.is_error

    # Test pattern that starts with **/ works recursively (no .txt files anywhere)
    result = await glob_tool(Params(pattern="**/*.txt", directory=str(test_files)))
    assert not result.is_error
    assert result.output == ""
    assert "No matches found" in result.message


async def test_glob_hidden_files(glob_tool: Glob, temp_work_dir: KaosPath):
    """Hidden files (dotfiles) should be matched by glob patterns."""
    # Create hidden files and visible files
    await (temp_work_dir / ".gitlab-ci.yml").write_text("stages: [build]")
    await (temp_work_dir / ".eslintrc.json").write_text("{}")
    await (temp_work_dir / "config.yml").write_text("key: value")

    result = await glob_tool(Params(pattern="*.yml", directory=str(temp_work_dir)))
    assert not result.is_error
    assert ".gitlab-ci.yml" in result.output
    assert "config.yml" in result.output


async def test_glob_hidden_files_in_recursive_pattern(glob_tool: Glob, temp_work_dir: KaosPath):
    """Hidden files inside hidden directories should be found by recursive glob patterns."""
    # Create hidden directory with files
    await (temp_work_dir / "src").mkdir()
    await (temp_work_dir / "src" / ".config").mkdir()
    await (temp_work_dir / "src" / ".config" / "settings.yml").write_text("debug: true")
    await (temp_work_dir / "src" / "main.py").write_text("pass")

    result = await glob_tool(Params(pattern="src/**/*.yml", directory=str(temp_work_dir)))
    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")
    assert "src/.config/settings.yml" in output


async def test_glob_hidden_directory_contents(glob_tool: Glob, temp_work_dir: KaosPath):
    """Files inside hidden directories should be discoverable."""
    await (temp_work_dir / ".github").mkdir()
    await (temp_work_dir / ".github" / "workflows").mkdir(parents=True)
    await (temp_work_dir / ".github" / "workflows" / "ci.yml").write_text("name: CI")
    await (temp_work_dir / "src").mkdir()
    await (temp_work_dir / "src" / "app.py").write_text("pass")

    result = await glob_tool(Params(pattern=".github/**/*.yml", directory=str(temp_work_dir)))
    assert not result.is_error
    assert isinstance(result.output, str)
    output = result.output.replace("\\", "/")
    assert ".github/workflows/ci.yml" in output


# --- Comprehensive edge-case tests ---


async def test_glob_empty_pattern(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob with empty pattern returns an error."""
    await (temp_work_dir / "file.txt").write_text("content")

    result = await glob_tool(Params(pattern="", directory=str(temp_work_dir)))
    assert result.is_error
    assert "Glob failed" in result.message


async def test_glob_include_dirs_true(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob with include_dirs=True includes directories."""
    await (temp_work_dir / "file.txt").write_text("content")
    await (temp_work_dir / "subdir").mkdir()

    result = await glob_tool(
        Params(pattern="*", directory=str(temp_work_dir), include_dirs=True)
    )
    assert not result.is_error
    assert "file.txt" in result.output
    assert "subdir" in result.output


async def test_glob_exception_handling(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test that exceptions during glob are handled gracefully."""
    from unittest.mock import patch

    with patch(
        "kaos.path.KaosPath.glob",
        side_effect=OSError("permission denied"),
    ):
        result = await glob_tool(Params(pattern="*.txt", directory=str(temp_work_dir)))

    assert result.is_error
    assert "Glob failed" in result.message
    assert "permission denied" in result.message


async def test_glob_default_directory(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob defaults to working directory when directory is None."""
    await (temp_work_dir / "default_test.txt").write_text("content")

    result = await glob_tool(Params(pattern="default_test.txt"))
    assert not result.is_error
    assert "default_test.txt" in result.output


async def test_glob_description_for_os():
    """Test _description_for_os includes Windows hint on Windows."""
    from kimi_cli.tools.file.glob import WINDOWS_PATH_HINT, _description_for_os

    desc = _description_for_os("Windows")
    assert WINDOWS_PATH_HINT in desc

    desc_unix = _description_for_os("Linux")
    assert WINDOWS_PATH_HINT not in desc_unix


async def test_glob_single_file_match(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob matching exactly one file."""
    await (temp_work_dir / "only.txt").write_text("content")

    result = await glob_tool(Params(pattern="only.txt", directory=str(temp_work_dir)))
    assert not result.is_error
    assert result.output == "only.txt"
    assert "Found 1 matches" in result.message


async def test_glob_deeply_nested_pattern(glob_tool: Glob, temp_work_dir: KaosPath):
    """Test glob with deeply nested directory structure."""
    deep = temp_work_dir / "a" / "b" / "c" / "d"
    await deep.mkdir(parents=True)
    await (deep / "deep.txt").write_text("content")

    result = await glob_tool(Params(pattern="a/**/deep.txt", directory=str(temp_work_dir)))
    assert not result.is_error
    assert "deep.txt" in result.output


async def test_glob_recursive_star(glob_tool: Glob, test_files: KaosPath):
    """Test recursive **/* pattern returns all files and directories."""
    result = await glob_tool(Params(pattern="src/**", directory=str(test_files)))

    assert not result.is_error
    output = result.output.replace("\\", "/")
    # Should include top-level files and dirs
    assert "src/main.py" in output
    assert "src/utils.py" in output
    # Should also include nested files
    assert "src/main/app.py" in output


async def test_glob_recursive_double_star(glob_tool: Glob, test_files: KaosPath):
    """Test recursive **/** pattern returns all files and directories."""
    result = await glob_tool(Params(pattern="src/**", directory=str(test_files)))

    assert not result.is_error
    output = result.output.replace("\\", "/")
    # Should include nested files
    assert "src/main.py" in output
    assert "src/main/app.py" in output
    assert "src/test/test_app.py" in output
    # Should also include nested files
    assert "src/main.py" in output
    assert "src/main/app.py" in output


async def test_glob_recursive_md(glob_tool: Glob, test_files: KaosPath):
    """Test recursive **/*.md returns all .md files."""
    result = await glob_tool(Params(pattern="**/*.md", directory=str(test_files)))

    assert not result.is_error
    output = result.output.replace("\\", "/")
    assert "README.md" in output
    assert "docs/guide.md" in output
    assert "docs/api.md" in output


# --- [out of work-dir] warning tests ---


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in Glob")
async def test_glob_outside_work_dir_has_warning(glob_tool: Glob):
    """Glob with directory outside work-dir should include [out of work-dir] in message."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file in the outside directory
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("content")
        result = await glob_tool(Params(pattern="*.txt", directory=tmpdir))
        assert not result.is_error
        assert "[out of work-dir]" in result.message


async def test_glob_inside_work_dir_no_warning(glob_tool: Glob, test_files: KaosPath):
    """Glob inside work-dir should NOT include [out of work-dir] in message."""
    result = await glob_tool(Params(pattern="*.txt", directory=str(test_files)))
    assert not result.is_error
    assert "[out of work-dir]" not in result.message


@pytest.mark.skipif(sys.platform == "win32", reason="[out of work-dir] warning not implemented in Glob")
async def test_glob_outside_work_dir_nonexistent_has_warning(glob_tool: Glob):
    """Glob with non-existent outside directory should include [out of work-dir]."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = str(Path(tmpdir) / "nonexistent")
        result = await glob_tool(Params(pattern="*", directory=nonexistent))
        assert result.is_error
        assert "[out of work-dir]" in result.message


# === Tests for output shaping (fold / max_results / gitignore fix) ===


async def test_glob_fold_output(glob_tool: Glob, temp_work_dir: KaosPath):
    """>max_results matches are head+tail folded with counts in message."""
    for i in range(500):
        await (temp_work_dir / f"f{i:03d}.txt").write_text(f"content {i}")
    # Explicit max_results=200: the default 500 would not fold 500 matches.
    result = await glob_tool(
        Params(pattern="*.txt", directory=str(temp_work_dir), max_results=200)
    )

    assert not result.is_error
    assert isinstance(result.output, str)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    real_lines = [ln for ln in output_lines if "lines omitted" not in ln]
    assert len(real_lines) == 200
    assert len(output_lines) == 201
    assert "… (300 lines omitted) …" in result.output
    assert "Found 500 matches" in result.message
    assert "Showing 200 of 500 (head+tail fold)" in result.message


async def test_glob_gitignore_dir_only_excludes_descendants(
    glob_tool: Glob, temp_work_dir: KaosPath
):
    """A dir-only rule (`.venv/`) excludes the dir AND its descendants."""
    await (temp_work_dir / ".gitignore").write_text(".venv/\n")
    await (temp_work_dir / "keep.py").write_text("x")
    await (temp_work_dir / ".venv").mkdir()
    await (temp_work_dir / ".venv" / "a.py").write_text("x")
    await (temp_work_dir / "sub").mkdir()
    await (temp_work_dir / "sub" / ".venv").mkdir()
    await (temp_work_dir / "sub" / ".venv" / "b.py").write_text("x")

    result = await glob_tool(Params(pattern="**/*.py", directory=str(temp_work_dir)))
    assert not result.is_error
    output = result.output.replace("\\", "/")
    assert "keep.py" in output
    assert ".venv/a.py" not in output
    assert "sub/.venv/b.py" not in output
    assert "Found 1 matches" in result.message


async def test_glob_gitignore_dir_only_negated_unignores(
    glob_tool: Glob, temp_work_dir: KaosPath
):
    """A negated dir-only rule (`!.venv/`) un-ignores the dir and its contents."""
    await (temp_work_dir / ".gitignore").write_text(".venv/\n!.venv/\n")
    await (temp_work_dir / ".venv").mkdir()
    await (temp_work_dir / ".venv" / "a.py").write_text("x")
    await (temp_work_dir / ".venv" / "sub").mkdir()
    await (temp_work_dir / ".venv" / "sub" / "b.py").write_text("x")

    result = await glob_tool(Params(pattern="**/*.py", directory=str(temp_work_dir)))
    assert not result.is_error
    output = result.output.replace("\\", "/")
    assert ".venv/a.py" in output
    assert ".venv/sub/b.py" in output
    assert "Found 2 matches" in result.message


async def test_glob_verbose_fold_and_truncate_line(
    glob_tool: Glob, temp_work_dir: KaosPath
):
    """Verbose lines are folded and per-line capped."""
    for i in range(250):
        await (temp_work_dir / f"f{i:03d}.txt").write_text("x")
    # A deeply nested long path forces per-line truncation in verbose mode.
    long_dir = "d" * 240
    long_name = "n" * 240
    await (temp_work_dir / long_dir).mkdir()
    await (temp_work_dir / long_dir / f"{long_name}.txt").write_text("deep")

    result = await glob_tool(
        Params(
            pattern="**/*.txt",
            directory=str(temp_work_dir),
            verbose=True,
            max_results=200,  # explicit: default 500 would not fold 251
        )
    )
    assert not result.is_error
    output_lines = result.output.split("\n")
    assert any("lines omitted" in ln for ln in output_lines)
    assert "Showing 200 of 251 (head+tail fold)" in result.message
    # Every verbose line is hard-capped (500 + marker slack).
    for ln in output_lines:
        assert len(ln) <= 520, f"line too long: {len(ln)} chars"
    assert any("… [+" in ln for ln in output_lines)


async def test_glob_fold_top_dirs_summary(glob_tool: Glob, temp_work_dir: KaosPath):
    """Folded results include a directory-grouped summary in the message."""
    for d in ("aaa", "bbb", "ccc"):
        await (temp_work_dir / d).mkdir()
    for i in range(150):
        await (temp_work_dir / "aaa" / f"f{i:03d}.txt").write_text("x")
    for i in range(100):
        await (temp_work_dir / "bbb" / f"f{i:03d}.txt").write_text("x")
    for i in range(50):
        await (temp_work_dir / "ccc" / f"f{i:03d}.txt").write_text("x")

    # Explicit max_results=200: the default 500 would not fold 300 matches.
    result = await glob_tool(
        Params(pattern="**/*.txt", directory=str(temp_work_dir), max_results=200)
    )
    assert not result.is_error
    assert "Showing 200 of 300 (head+tail fold)" in result.message
    assert "top dirs: aaa (150), bbb (100), ccc (50)" in result.message


async def test_glob_max_results_unlimited(glob_tool: Glob, temp_work_dir: KaosPath):
    """max_results=0 disables folding; the MAX_MATCHES cap still applies."""
    for i in range(MAX_MATCHES + 50):
        await (temp_work_dir / f"file_{i}.txt").write_text(f"content {i}")
    result = await glob_tool(
        Params(pattern="*.txt", directory=str(temp_work_dir), max_results=0)
    )

    assert not result.is_error
    assert isinstance(result.output, str)
    output_lines = [line for line in result.output.split("\n") if line.strip()]
    assert len(output_lines) == MAX_MATCHES
    assert "lines omitted" not in result.output
    assert f"Search capped at {MAX_MATCHES} matches." in result.message


def test_glob_params_max_results_default_500():
    assert Params(pattern="x").max_results == 500


def test_glob_params_max_results_alias_fold():
    params = Params.model_validate({"pattern": "x", "fold": 0})
    assert params.max_results == 0
