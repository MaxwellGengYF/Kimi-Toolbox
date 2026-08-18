"""Tests for system prompt bloat guard (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kimix.utils.system_prompt import get_system_prompt, SystemPromptType


def _make_runtime(tmp_path: Path, custom_data: dict[str, Any] | None = None) -> SimpleNamespace:
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


class TestSystemPromptBudget:
    """Test that system prompt respects max_system_prompt_tokens."""

    def test_prompt_under_budget(self, tmp_path: Path):
        prompt_func = get_system_prompt(
            work_dir=tmp_path,
            agent_role=SystemPromptType.Worker,
            max_system_prompt_tokens=10_000,
        )
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime)
        from kimi_cli.utils.tokens import count_tokens
        assert count_tokens(prompt) <= 10_000

    def test_prompt_over_budget_truncates_steps(self, tmp_path: Path):
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        steps = []
        for i in range(300):
            steps.append({
                "seq": i,
                "time": f"2024-01-01T00:00:00+00:00",
                "brief": f"brief {i}",
                "step": f"step text {i} " * 50,
                "result": f"result {i}",
                "files": [],
            })
        (steps_dir / "test-session.json").write_text(json.dumps(steps), encoding="utf-8")

        prompt_func = get_system_prompt(
            work_dir=tmp_path,
            agent_role=SystemPromptType.Worker,
            max_system_prompt_tokens=2_000,
        )
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime, is_compacting=True)
        from kimi_cli.utils.tokens import count_tokens
        assert count_tokens(prompt) <= 2_000

    def test_agents_md_dropped_when_over_budget(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        # Make AGENTS.md very large so it must be dropped
        agents_md.write_text("x" * 8000, encoding="utf-8")

        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        # Create many large steps to blow past the token budget
        steps = []
        for i in range(200):
            steps.append({
                "seq": i,
                "time": "t",
                "brief": "b" * 200,
                "step": "s" * 1000,
                "result": "r" * 1000,
                "files": [],
            })
        (steps_dir / "test-session.json").write_text(json.dumps(steps), encoding="utf-8")

        prompt_func = get_system_prompt(
            work_dir=tmp_path,
            agent_role=SystemPromptType.Worker,
            # The base Worker prompt includes the "# Tool Conventions" block
            # (≈150 tokens) plus the numbered behavior items, so the budget
            # must exceed that baseline for the AGENTS.md-drop guard to be the
            # thing that trips. The huge AGENTS.md below (≈2000 tokens) is
            # still far over this budget and must be dropped.
            max_system_prompt_tokens=1_000,
        )
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime, is_compacting=True)
        from kimi_cli.utils.tokens import count_tokens
        assert count_tokens(prompt) <= 1_000
        # AGENTS.md should be dropped to the short form
        assert "read AGENTS.md before work" in prompt

    def test_default_budget_is_4000(self, tmp_path: Path):
        prompt_func = get_system_prompt(work_dir=tmp_path, agent_role=SystemPromptType.Worker)
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime)
        from kimi_cli.utils.tokens import count_tokens
        assert count_tokens(prompt) <= 4_000


class TestSubAgentCoverageReport:
    """Sub-agents must report coverage so the parent can trust or re-check."""

    def test_trivial_subagent_prompt_requires_coverage_report(self, tmp_path: Path):
        prompt_func = get_system_prompt(
            work_dir=tmp_path,
            agent_role=SystemPromptType.TrivialSubAgent,
            max_system_prompt_tokens=10_000,
        )
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime)
        assert "Report coverage" in prompt
        assert "sampled/approximated" in prompt

    def test_worker_prompt_has_no_coverage_clause(self, tmp_path: Path):
        prompt_func = get_system_prompt(
            work_dir=tmp_path,
            agent_role=SystemPromptType.Worker,
            max_system_prompt_tokens=10_000,
        )
        runtime = _make_runtime(tmp_path)
        prompt = prompt_func(runtime)
        assert "Report coverage" not in prompt
