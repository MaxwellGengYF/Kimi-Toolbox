from __future__ import annotations

# ruff: noqa

from dataclasses import replace
import platform
import pytest
from inline_snapshot import snapshot

from kimi_cli.tools.agent import Agent as AgentTool
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditFile
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.todo import TodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb


def test_glob_description(runtime):
    """Test the description of Glob tool."""
    runtime.environment = replace(runtime.environment, os_kind="Linux")
    glob_tool = Glob(runtime)
    windows_path_hint = "On Windows, the `directory` parameter accepts both Windows native paths"

    assert windows_path_hint not in glob_tool.base.description
    assert glob_tool.base.description == snapshot(
        "Find files by glob pattern. Use `ReadFile` to read the paths found.\n"
    )


def test_glob_description_on_windows(runtime):
    """Test the Windows-specific description of Glob tool."""
    runtime.environment = replace(runtime.environment, os_kind="Windows")
    glob_tool = Glob(runtime)
    windows_path_hint = "Windows: `directory` accepts native (`C:\\Users\\foo`) and POSIX-style (`/c/Users/foo`) paths. Results use backslashes — convert to forward slashes for shell commands."

    assert windows_path_hint in glob_tool.base.description
