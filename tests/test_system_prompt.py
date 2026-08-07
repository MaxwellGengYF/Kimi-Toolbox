"""Comprehensive tests for system_prompt ToolCallReason integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kimix.utils.system_prompt import SystemPromptType, get_system_prompt

_OUTPUT_MD = Path(__file__).with_name("output.md")


def _append_prompt(test_name: str, prompt: str) -> None:
    with _OUTPUT_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n## {test_name}\n\n{prompt}\n")


def _make_runtime(tmp_path: Path, custom_data: dict[str, Any] | None = None) -> SimpleNamespace:
    """Build a minimal runtime-like object for get_system_prompt."""
    builtin_args = SimpleNamespace(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=tmp_path,
        KIMI_WORK_DIR_LS="",
        KIMI_AGENTS_MD="",
        KIMI_SKILLS="",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS="Windows",
        KIMI_SHELL="bash",
    )
    session = SimpleNamespace(
        dir=tmp_path,
        id="test-session",
        custom_data=custom_data or {},
    )
    return SimpleNamespace(
        builtin_args=builtin_args,
        session=session,
    )


class TestSystemPromptAgentsMd:
    """Test AGENTS.md size limiting in system prompt."""

    def test_agents_md_too_long(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("x" * 5000, encoding="utf-8")

        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_agents_md_too_long", prompt)

        assert "read AGENTS.md before work" in prompt
        assert "x" * 100 not in prompt

    def test_agents_md_short(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        content = "Short agents file"
        agents_md.write_text(content, encoding="utf-8")

        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        prompt = prompt_func(runtime)
        _append_prompt("test_agents_md_short", prompt)

        assert content in prompt
        assert "read AGENTS.md before work" not in prompt


class TestSystemPromptShellTool:
    """The system prompt names the shell tool selected by agent config."""

    def _prompt(self, tmp_path: Path) -> str:
        runtime = _make_runtime(tmp_path)
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        return prompt_func(runtime)
