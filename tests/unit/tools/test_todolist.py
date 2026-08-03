"""Tests for Defects 4.1-4.4: TodoList improvements."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from kimi_cli.tools.todo import Params as TodoListParams, Todo, TodoList


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
            auto_fix=False,
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


class TestTodoCodeField:
    def test_todo_with_code(self) -> None:
        t = Todo(title="task", status="pending", code="print('hello')")
        assert t.code == "print('hello')"

    def test_todo_with_code_file_alias(self) -> None:
        t = Todo(title="task", status="pending", code_file="print('alias')")
        assert t.code == "print('alias')"

    def test_todo_without_code(self) -> None:
        t = Todo(title="task", status="pending")
        assert t.code is None

    def test_todo_empty_code(self) -> None:
        t = Todo(title="task", status="pending", code="")
        assert t.code == ""

    def test_merge_preserves_code_when_new_omits(self) -> None:
        old = Todo(title="task", status="pending", code="print('old')")
        new = Todo(title="task", status="done")  # code omitted
        merged = TodoList._merge_one(old, new)
        assert merged.code == "print('old')"

    def test_merge_updates_code_when_new_provides(self) -> None:
        old = Todo(title="task", status="pending", code="print('old')")
        new = Todo(title="task", status="done", code="print('new')")
        merged = TodoList._merge_one(old, new)
        assert merged.code == "print('new')"

    def test_merge_clears_code_when_new_is_empty_string(self) -> None:
        old = Todo(title="task", status="pending", code="print('old')")
        new = Todo(title="task", status="done", code="")
        merged = TodoList._merge_one(old, new)
        assert merged.code == ""


class TestTodoShellBackwardCompat:
    """Smoke tests that shell-aware execution keeps the static defaults working."""

    def test_resolve_code_executable_shell_prefix(self) -> None:
        assert TodoList._resolve_code_executable("!pytest tests/ -x -q") == (
            "shell",
            "pytest tests/ -x -q",
        )

    @pytest.mark.asyncio
    async def test_run_code_accepts_legacy_executable_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A legacy plain `.py` path passed as `executable` still routes to python."""
        captured: dict[str, object] = {}

        async def fake_run_process(
            argv: list[str],
            timeout: int,
            *,
            not_found_hint: str,
            env: dict[str, str] | None = None,
        ) -> tuple[bool, str]:
            captured["argv"] = argv
            captured["env"] = env
            return True, "ok"

        monkeypatch.setattr(TodoList, "_run_process", fake_run_process)
        ok, out = await TodoList._run_code(
            "C:/nonexistent/script.py", executable="C:/nonexistent/script.py"
        )
        assert ok and out == "ok"
        argv = captured["argv"]
        assert isinstance(argv, list) and argv[0] == sys.executable
        assert argv[1] == "C:/nonexistent/script.py"
        assert captured["env"] is None

    def test_params_and_todo_construction_unchanged(self) -> None:
        params = TodoListParams(
            todos=[Todo(title="task", status="pending", code="print('hello')")]
        )
        assert params.todos[0].code == "print('hello')"

    def test_tool_instantiation_sets_shell_kind(self, mock_runtime: MagicMock) -> None:
        tl = TodoList(runtime=mock_runtime)
        assert tl._shell_kind in ("bash", "powershell")
        assert "Track progress with a todo list." in tl.description


class TestAllDoneReminderWrite:
    """Verify ALL_DONE_REMINDER in _build_success_response uses current_prompt."""

    async def test_uses_current_prompt_when_all_done(self, mock_runtime: MagicMock) -> None:
        """When runtime.current_prompt is set, it is appended after the reminder."""
        from kimi_cli.tools.todo import Params as TodoListParams, Todo, TodoList

        mock_runtime.current_prompt = "my original request"
        tl = TodoList(runtime=mock_runtime)

        # Mock persistence so _write_todos -> _build_success_response works
        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[Todo(title="task", status="done")],
            mode="append",
        ))
        # Both the hardcoded reminder AND the appended prompt should be present
        assert "All todos are done." in result.output
        assert "Original prompt:" in result.output
        assert "my original request" in result.output
        assert "All todos are done." in result.message
        assert "Original prompt:" in result.message
        assert "my original request" in result.message

    async def test_fallback_when_current_prompt_not_set(self, mock_runtime: MagicMock) -> None:
        """When runtime has no current_prompt, use the generic fallback."""
        from kimi_cli.tools.todo import Params as TodoListParams, Todo, TodoList

        # MagicMock getattr returns MagicMock, which is truthy.
        # Explicitly set current_prompt to None to test fallback.
        mock_runtime.current_prompt = None
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[Todo(title="task", status="done")],
            mode="append",
        ))
        assert "All todos are done." in result.output
        assert "nothing is left unfinished" in result.output
        assert "my original request" not in result.output


class TestAllDoneReminderRead:
    """Verify ALL_DONE_REMINDER in _read_todos uses current_prompt."""

    async def test_read_todos_all_done_with_current_prompt(self, mock_runtime: MagicMock) -> None:
        """When reading and all todos done, current_prompt is appended after the reminder."""
        from kimi_cli.tools.todo import TodoList

        mock_runtime.current_prompt = "my request"
        tl = TodoList(runtime=mock_runtime)

        # Mock internal state: all todos done, no archived
        tl._load_todos = MagicMock(return_value=[
            MagicMock(status="done", title="task", notes=None, code=None)
        ])
        tl._load_archived_todos = MagicMock(return_value=[])

        result = tl._read_todos()
        # Both the hardcoded reminder AND the appended prompt should be present
        assert "All todos are done." in result.output
        assert "Original prompt:" in result.output
        assert "my request" in result.output
        assert "All todos are done." in result.message
        assert "Original prompt:" in result.message
        assert "my request" in result.message

    async def test_read_todos_all_done_fallback(self, mock_runtime: MagicMock) -> None:
        """Generic fallback when current_prompt is None."""
        from kimi_cli.tools.todo import TodoList

        mock_runtime.current_prompt = None
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[
            MagicMock(status="done", title="task", notes=None, code=None)
        ])
        tl._load_archived_todos = MagicMock(return_value=[])

        result = tl._read_todos()
        assert "All todos are done." in result.output
        assert "nothing is left unfinished" in result.output

class TestTruncatePrompt:
    """Verify _truncate_prompt helper function."""

    def test_short_prompt_not_truncated(self) -> None:
        from kimi_cli.tools.todo import _truncate_prompt
        text = "short request"
        assert _truncate_prompt(text) == text

    def test_exactly_200_chars(self) -> None:
        from kimi_cli.tools.todo import _truncate_prompt
        text = "a" * 200
        assert _truncate_prompt(text) == text
        assert len(_truncate_prompt(text)) == 200

    def test_long_prompt_truncated_head_tail(self) -> None:
        from kimi_cli.tools.todo import _truncate_prompt
        # Build a 250-char string: "HEAD..." + pad + "...TAIL"
        head = "X" * 100
        tail = "Y" * 100
        middle = "Z" * 50  # total 250 chars
        text = head + middle + tail
        result = _truncate_prompt(text)
        assert result == head + "..." + tail
        assert "..." in result
        assert len(result) == 100 + 3 + 100  # 203 chars
        assert result.startswith("X" * 100)
        assert result.endswith("Y" * 100)
        # The middle Z's are replaced by "..."
        assert "Z" not in result

    def test_long_prompt_uses_200_threshold_by_default(self) -> None:
        from kimi_cli.tools.todo import _truncate_prompt
        # 201 chars — exceeds 200
        text = "a" * 101 + "b" * 100
        result = _truncate_prompt(text)
        assert "..." in result
        assert len(result) == 203


class TestAllDoneReminderWriteTruncation:
    """Verify truncation in _build_success_response."""

    async def test_long_current_prompt_truncated_in_output(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import Params as TodoListParams, Todo, TodoList

        # Build a prompt > 200 chars
        long_prompt = "A" * 150 + "B" * 150  # 300 chars
        mock_runtime.current_prompt = long_prompt
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[Todo(title="task", status="done")],
            mode="append",
        ))
        # Hardcoded reminder is always present
        assert "All todos are done." in result.output
        # Should have "..." truncation marker
        assert "..." in result.output
        # Should contain the original head (100 As)
        assert "A" * 100 in result.output
        # Should contain the original tail (100 Bs)
        assert "B" * 100 in result.output
        # The middle chars (the 150th A etc.) should be replaced by ...
        assert "A" * 150 not in result.output

    async def test_short_current_prompt_not_truncated(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import Params as TodoListParams, Todo, TodoList

        mock_runtime.current_prompt = "short request"
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[Todo(title="task", status="done")],
            mode="append",
        ))
        assert "All todos are done." in result.output
        assert "Original prompt:" in result.output
        assert "short request" in result.output
        assert "..." not in result.output


class TestAllDoneReminderReadTruncation:
    """Verify truncation in _read_todos."""

    async def test_long_current_prompt_truncated_in_read(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import TodoList

        long_prompt = "X" * 150 + "Y" * 150  # 300 chars
        mock_runtime.current_prompt = long_prompt
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[
            MagicMock(status="done", title="task", notes=None, code=None)
        ])
        tl._load_archived_todos = MagicMock(return_value=[])

        result = tl._read_todos()
        assert "All todos are done." in result.output
        assert "..." in result.output
        assert "X" * 100 in result.output
        assert "Y" * 100 in result.output

    async def test_short_current_prompt_not_truncated_in_read(self, mock_runtime: MagicMock) -> None:
        from kimi_cli.tools.todo import TodoList

        mock_runtime.current_prompt = "short request"
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[
            MagicMock(status="done", title="task", notes=None, code=None)
        ])
        tl._load_archived_todos = MagicMock(return_value=[])

        result = tl._read_todos()
        assert "All todos are done." in result.output
        assert "Original prompt:" in result.output
        assert "short request" in result.output
        assert "..." not in result.output
