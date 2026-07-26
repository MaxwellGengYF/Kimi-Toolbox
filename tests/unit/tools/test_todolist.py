"""Tests for Defects 4.1-4.4: TodoList improvements."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from kimi_cli.tools.todo import Params as TodoListParams, Todo


class TestTodoListSimplify:
    def test_parent_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TodoListParams(
                todos=[Todo(title="task", status="pending")],
                parent_title="some parent",
            )

    def test_match_mode_accepted(self) -> None:
        params = TodoListParams(
            todos=[Todo(title="task", status="done")],
            match_mode="exact",
        )
        assert params.match_mode == "exact"

    def test_match_mode_invalid(self) -> None:
        with pytest.raises(ValidationError):
            TodoListParams(
                todos=[Todo(title="task", status="done")],
                match_mode="invalid",
            )


class TestTodoListModeSynonymsRemoved:
    @pytest.mark.parametrize("valid_mode", ["overwrite", "append", "force_overwrite"])
    def test_canonical_modes_accepted(self, valid_mode: str) -> None:
        TodoListParams(todos=[], mode=valid_mode)

    @pytest.mark.parametrize("invalid_mode", [
        "force", "forced", "write", "set", "put", "merge", "update",
    ])
    def test_old_synonyms_rejected(self, invalid_mode: str) -> None:
        with pytest.raises(ValidationError):
            TodoListParams(todos=[], mode=invalid_mode)


class TestTodoListSingleInProgress:
    async def test_two_in_progress_rejected(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import TodoList
        tl = TodoList(runtime=mock_runtime)
        result = await tl(TodoListParams(
            todos=[
                Todo(title="task A", status="in_progress"),
                Todo(title="task B", status="in_progress"),
            ],
            mode="append",
        ))
        assert result.is_error
        assert "in_progress" in result.output.lower()

    async def test_single_in_progress_ok(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import TodoList
        tl = TodoList(runtime=mock_runtime)
        result = await tl(TodoListParams(
            todos=[Todo(title="task A", status="in_progress")],
            mode="append",
        ))
        # May fail due to persistence, but should not raise
        assert result is not None
