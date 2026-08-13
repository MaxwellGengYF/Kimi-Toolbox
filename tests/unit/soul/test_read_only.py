"""Tests for read-only mode restrictions.

Covers:
- Runtime.read_only flag (default, propagation via copy_for_subagent)
- KimiToolset blocking blocked tools when read_only=True
- KimiToolset allowing unblocked tools when read_only=True
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import _READ_ONLY_BLOCKED_TOOLS, KimiToolset
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
            for name in ("read", "glob", "grep", "fetch_url"):
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
        # read is registered by toolset_factory automatically
        call = _make_tool_call("read")
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
