"""Tests for Glob's gitignore-aware "no matches" hint.

Regression: with ``respect_gitignore=True`` (default) a pattern whose only
matches are gitignored used to report a bare "No matches found" — exactly the
same wording as "the directory is empty", so the agent misread existing-but-
hidden paths as "nothing there" (e.g. globbing a gitignored ``models/`` dir).
Now the message states how many paths were excluded by .gitignore.
"""

from __future__ import annotations

import pytest
from kimi_cli.tools.file.glob import Glob


async def test_no_matches_explains_gitignore_exclusion(
    glob_tool: Glob, tmp_path: pytest.TempPathFactory
) -> None:
    """A pattern matching only gitignored files must say so, not stay silent."""
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "app.log").write_text("noise\n", encoding="utf-8")

    params = glob_tool.params(pattern="*.log", directory=str(tmp_path))
    result = await glob_tool(params)

    assert result.output == ""
    assert "No matches found" in result.message
    assert "excluded by .gitignore" in result.message
    assert "1 path(s)" in result.message
    assert "respect_gitignore=False" in result.message


async def test_include_ignored_lists_the_file(
    glob_tool: Glob, tmp_path: pytest.TempPathFactory
) -> None:
    """respect_gitignore=False must actually surface the gitignored file."""
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "app.log").write_text("noise\n", encoding="utf-8")

    params = glob_tool.params(
        pattern="*.log", directory=str(tmp_path), respect_gitignore=False
    )
    result = await glob_tool(params)

    assert "app.log" in result.output
    assert "Found 1 matches" in result.message


async def test_truly_empty_dir_has_no_hint(
    glob_tool: Glob, tmp_path: pytest.TempPathFactory
) -> None:
    """A genuinely empty directory keeps the plain wording (no false hint)."""
    params = glob_tool.params(pattern="*.log", directory=str(tmp_path))
    result = await glob_tool(params)

    assert result.output == ""
    assert result.message == "No matches found for pattern `*.log`."
    assert "excluded by .gitignore" not in result.message


async def test_ignored_directory_with_recursive_pattern(
    glob_tool: Glob, tmp_path: pytest.TempPathFactory
) -> None:
    """The conversation scenario: a whole directory is gitignored (e.g.
    ``models/``) — globbing into it must explain the empty result instead of
    reporting a bare "No matches found"."""
    (tmp_path / ".gitignore").write_text("models/\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    (models / "Qwen.gguf").write_text("binary-ish\n", encoding="utf-8")

    params = glob_tool.params(pattern="models/**/*", directory=str(tmp_path))
    result = await glob_tool(params)

    assert result.output == ""
    assert "No matches found" in result.message
    assert "excluded by .gitignore" in result.message
    assert "1 path(s)" in result.message
