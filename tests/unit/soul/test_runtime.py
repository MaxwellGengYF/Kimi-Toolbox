"""Tests for Runtime.current_prompt field."""
from __future__ import annotations

from unittest.mock import MagicMock

from kimi_cli.soul.agent import Runtime


class TestRuntimeCurrentPrompt:
    """Verify the current_prompt field on Runtime."""

    def test_current_prompt_defaults_to_none(self) -> None:
        """When not provided, current_prompt is None."""
        runtime = Runtime(
            config=MagicMock(),
            oauth=MagicMock(),
            llm=None,
            session=MagicMock(),
            builtin_args=MagicMock(),
            denwa_renji=MagicMock(),
            approval=MagicMock(),
            labor_market=MagicMock(),
            environment=MagicMock(),
            notifications=MagicMock(),
            background_tasks=MagicMock(),
            skills={},
            additional_dirs=[],
            skills_dirs=[],
        )
        assert runtime.current_prompt is None

    def test_current_prompt_set_and_read(self) -> None:
        """current_prompt can be set after construction and read back."""
        runtime = Runtime(
            config=MagicMock(),
            oauth=MagicMock(),
            llm=None,
            session=MagicMock(),
            builtin_args=MagicMock(),
            denwa_renji=MagicMock(),
            approval=MagicMock(),
            labor_market=MagicMock(),
            environment=MagicMock(),
            notifications=MagicMock(),
            background_tasks=MagicMock(),
            skills={},
            additional_dirs=[],
            skills_dirs=[],
        )
        runtime.current_prompt = "hello world"
        assert runtime.current_prompt == "hello world"

    def test_current_prompt_accepts_empty_string(self) -> None:
        """Empty string is a valid value (falsy, but not None)."""
        runtime = Runtime(
            config=MagicMock(),
            oauth=MagicMock(),
            llm=None,
            session=MagicMock(),
            builtin_args=MagicMock(),
            denwa_renji=MagicMock(),
            approval=MagicMock(),
            labor_market=MagicMock(),
            environment=MagicMock(),
            notifications=MagicMock(),
            background_tasks=MagicMock(),
            skills={},
            additional_dirs=[],
            skills_dirs=[],
        )
        runtime.current_prompt = ""
        assert runtime.current_prompt == ""
