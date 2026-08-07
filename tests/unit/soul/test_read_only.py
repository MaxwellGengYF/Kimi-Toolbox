"""Tests for read-only mode restrictions.

Covers:
- Runtime.read_only flag (default, propagation via copy_for_subagent)
- KimiToolset blocking blocked tools when read_only=True
- KimiToolset allowing unblocked tools when read_only=True
- TodoList stripping code fields when read_only=True
- Memory blocking write/append actions when read_only=True
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import _READ_ONLY_BLOCKED_TOOLS, KimiToolset
from kimi_cli.tools.memory import Memory
from kimi_cli.tools.todo import Params as TodoListParams
from kimi_cli.tools.todo import Todo, TodoList
from kimi_cli.wire.types import ToolCall, ToolResult
from kosong.tooling import ToolError

# =========================================================================
# Phase 6.1 — Runtime.read_only flag
# =========================================================================


class TestRuntimeReadOnlyFlag:
    """Verify Runtime.read_only defaults and propagation."""


    def test_copy_for_subagent_propagates_read_only(self) -> None:
        """Sub-agent inherits read_only from parent."""
        parent = Runtime(
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
            read_only=True,
        )
        child = parent.copy_for_subagent(agent_id="sub1", subagent_type="test")
        assert child.read_only is True

    def test_copy_for_subagent_defaults_false(self) -> None:
        """Sub-agent from a non-read-only parent also has read_only=False."""
        parent = Runtime(
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
            read_only=False,
        )
        child = parent.copy_for_subagent(agent_id="sub2", subagent_type="test")
        assert child.read_only is False


# =========================================================================
# Phase 6.2 — KimiToolset read_only blocking
# =========================================================================


def _make_tool_call(name: str, args: str = "{}") -> ToolCall:
    """Create a minimal ToolCall for testing."""
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function={"name": name, "arguments": args},
    )


def _result_is_blocked(result: ToolResult) -> bool:
    """Check if a ToolResult is a blocked read-only response."""
    if result is None:
        return False
    rv = result.return_value
    return (
        isinstance(rv, ToolError)
        and "forbidden in read-only mode" in str(rv.message)
    )


class TestKimiToolsetReadOnlyBlocking:
    """Verify that KimiToolset blocks restricted tools in read_only mode."""

    @pytest.fixture
    def toolset_factory(self):
        """Return a function that creates a KimiToolset with given read_only."""
        from unittest.mock import AsyncMock

        from kosong.tooling import ToolReturnValue

        def _make_dummy_tool(name: str):
            """Create a minimal tool mock suitable for KimiToolset."""
            dummy = MagicMock()
            dummy.name = name
            dummy.params = MagicMock()
            # call() must be async and return a proper ToolReturnValue
            dummy.call = AsyncMock(
                return_value=ToolReturnValue(
                    is_error=False,
                    output="ok",
                    message="ok",
                    display=[],
                )
            )
            return dummy

        def _create(read_only: bool = False, tools: list[str] | None = None):
            runtime = MagicMock()
            runtime.read_only = read_only
            ts = KimiToolset(runtime=runtime)

            # Register dummy tools for each blocked tool name
            if tools is None:
                tools = list(_READ_ONLY_BLOCKED_TOOLS)
            for name in tools:
                ts.add(_make_dummy_tool(name))

            # Also register some allowed tools
            for name in ("ReadFile", "Glob", "Grep", "FetchURL"):
                ts.add(_make_dummy_tool(name))

            return ts
        return _create

    @pytest.mark.parametrize("tool_name", sorted(_READ_ONLY_BLOCKED_TOOLS))
    async def test_blocked_tools_return_error_when_read_only(
        self, toolset_factory, tool_name: str
    ) -> None:
        """Each blocked tool returns ToolError when read_only=True."""
        ts = toolset_factory(read_only=True)
        # patch _call to avoid calling actual tool
        call = _make_tool_call(tool_name)
        result = ts.handle(call)
        if isinstance(result, ToolCall):
            result = await result
        assert _result_is_blocked(result), (
            f"Tool '{tool_name}' should be blocked in read_only mode"
        )

    async def test_allowed_tools_work_when_read_only(
        self, toolset_factory,
    ) -> None:
        """Allowed tools still execute when read_only=True."""
        ts = toolset_factory(read_only=True, tools=[])
        # ReadFile is registered by toolset_factory automatically
        call = _make_tool_call("ReadFile")
        # The handle returns a Task for known tools
        result = ts.handle(call)
        if hasattr(result, "__await__"):
            result = await result
        # Should NOT be a blocked error — allowed tools pass through.
        assert not _result_is_blocked(result)

    async def test_read_only_false_allows_blocked_tools(
        self, toolset_factory,
    ) -> None:
        """Blocked tools work normally when read_only=False."""
        ts = toolset_factory(read_only=False)
        for tool_name in list(_READ_ONLY_BLOCKED_TOOLS)[:3]:
            call = _make_tool_call(tool_name)
            result = ts.handle(call)
            if hasattr(result, "__await__"):
                result = await result
            assert not _result_is_blocked(result), (
                f"Tool '{tool_name}' should NOT be blocked when read_only=False"
            )


# =========================================================================
# Phase 6.3 — TodoList code stripping in read_only mode
# =========================================================================


class TestTodoListReadOnly:
    """Verify TodoList strips code fields when read_only=True."""

    @pytest.fixture
    def mock_runtime(self) -> MagicMock:
        return MagicMock()

    async def test_code_stripped_when_read_only(self, mock_runtime: MagicMock) -> None:
        """code field is set to None when todo has code and read_only=True."""
        mock_runtime.read_only = True
        tl = TodoList(runtime=mock_runtime)

        # Mock persistence to avoid side effects
        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[
                Todo(title="task A", status="pending", code="print('hello')"),
                Todo(title="task B", status="pending", code="!pytest"),
                Todo(title="task C", status="pending"),  # no code
            ],
            mode="append",
        ))

        # Should succeed with a warning
        assert not result.is_error
        # Warning should mention code stripping
        assert "<system-warning>" in result.output
        assert "code" in result.output.lower()
        assert "2 todo(s) affected" in result.output

    async def test_read_only_false_preserves_code(
        self, mock_runtime: MagicMock,
    ) -> None:
        """code field is preserved when read_only=False."""
        mock_runtime.read_only = False
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[
                Todo(title="task A", status="pending", code="print('hello')"),
            ],
            mode="append",
        ))

        # Should succeed without a warning about code stripping
        assert not result.is_error
        assert "<system-warning>" not in result.output

    async def test_no_warning_when_no_code_in_todos(
        self, mock_runtime: MagicMock,
    ) -> None:
        """No warning emitted when no todos have code fields."""
        mock_runtime.read_only = True
        tl = TodoList(runtime=mock_runtime)

        tl._load_todos = MagicMock(return_value=[])
        tl._load_archived_todos = MagicMock(return_value=[])
        tl._save_todos = MagicMock(return_value=None)

        result = await tl(TodoListParams(
            todos=[
                Todo(title="task A", status="pending"),
                Todo(title="task B", status="done"),
            ],
            mode="append",
        ))

        # Should succeed without warning
        assert not result.is_error
        assert "<system-warning>" not in result.output


# =========================================================================
# Phase 6.4 — Memory action blocking in read_only mode
# =========================================================================


class TestMemoryReadOnly:
    """Verify Memory write/append are allowed in read_only mode.

    The read-only guard lives in :data:`_READ_ONLY_BLOCKED_TOOLS` at the
    toolset level; the Memory tool itself does not block writes.
    """

    @pytest.fixture
    def session_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "session"

    @pytest.fixture
    def runtime(self, session_dir: Path) -> MagicMock:
        r = MagicMock()
        r.session.dir = session_dir
        return r

    @pytest.fixture
    def tool(self, runtime: MagicMock, session_dir: Path) -> Memory:
        t = Memory(runtime=runtime)  # type: ignore[arg-type]
        return t

    @pytest.mark.parametrize("action", ["write", "append"])
    async def test_write_append_allowed_when_read_only(
        self, tool: Memory, runtime: MagicMock, action: str,
    ) -> None:
        """Write/append actions still work when read_only=True (not a blocked tool)."""
        runtime.read_only = True
        from kimi_cli.tools.memory import Params as MemoryParams

        result = await tool(MemoryParams(
            action=action,  # type: ignore[arg-type]
            topic="test",
            content="some content",
        ))
        assert not (isinstance(result, ToolError) and "forbidden in read-only mode" in result.message)

    @pytest.mark.parametrize("action", ["read", "list", "search"])
    async def test_read_list_search_allowed_when_read_only(
        self, tool: Memory, runtime: MagicMock, action: str,
    ) -> None:
        """Read/list/search actions still work when read_only=True."""
        runtime.read_only = True
        from kimi_cli.tools.memory import Params as MemoryParams

        kwargs = {"action": action}  # type: ignore[arg-type]
        if action == "search":
            kwargs["query"] = "test"

        result = await tool(MemoryParams(**kwargs))

        # Should NOT be a blocked error (may be a different error like
        # "no memory topics", but not the read-only forbidden error)
        assert not (isinstance(result, ToolError) and "forbidden in read-only mode" in result.message)

    async def test_read_only_false_allows_write(
        self, tool: Memory, runtime: MagicMock,
    ) -> None:
        """Write action works normally when read_only=False."""
        runtime.read_only = False
        from kimi_cli.tools.memory import Params as MemoryParams

        result = await tool(MemoryParams(
            action="write",
            topic="test",
            content="some content",
        ))
        # Should NOT be a forbidden error
        assert not (isinstance(result, ToolError) and "forbidden in read-only mode" in result.message)
