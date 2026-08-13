"""Tests for the context_prune tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kosong.message import Message

from kimi_cli.soul.context import Context
from kimi_cli.soul.context_pruning import ContextPruner
from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.tools.context_prune import context_prune, Params
from kimi_cli.wire.types import StatusUpdate, TextPart, ThinkPart


def _make_soul(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    """Build a minimal mock KimiSoul for context_prune tests."""
    history_index_path = tmp_path / "history_index.json"
    history_index = HistoryIndex(persist_path=history_index_path)

    context_path = tmp_path / "context.jsonl"
    context_path.touch()
    context = Context(file_backend=context_path)

    runtime = overrides.get("runtime")
    if runtime is None:
        llm = SimpleNamespace(
            max_context_size=128_000,
            chat_provider=SimpleNamespace(
                model_name="test-model",
                thinking_effort="off",
            ),
        )
        runtime = SimpleNamespace(llm=llm, role="root")

    loop_control = SimpleNamespace(prune_subagents=False)

    compact_called = False

    async def compact_context(*, manual: bool = False) -> None:
        nonlocal compact_called
        compact_called = True

    status = SimpleNamespace(
        context_usage=0.5,
        context_tokens=1000,
        max_context_tokens=128_000,
    )

    thinking = overrides.get("thinking", None)

    soul = SimpleNamespace(
        context=context,
        pruner=ContextPruner(
            trigger_ratio=0.0,
            target_ratio=0.0,
            stable_prefix_messages=0,
            recent_messages_protected=6,
            min_free_tokens=0,
            cooldown_steps=0,
        ),
        runtime=runtime,
        _history_index=history_index,
        _recently_restored_refs=set(),
        _loop_control=loop_control,
        current_step_no=1,
        status=status,
        compact_context=compact_context,
        thinking=thinking,
    )
    soul._compact_called = lambda: compact_called
    return soul


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant(text: str = "") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)] if text else [])


def _assistant_with_think(think: str) -> Message:
    return Message(role="assistant", content=[ThinkPart(think=think)])


def _tool(text: str, tool_call_id: str = "call_1") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id=tool_call_id)


def _system_reminder(text: str) -> Message:
    return Message(
        role="user",
        content=[TextPart(text=f"<system-reminder>\n{text}\n</system-reminder>")],
    )


@pytest.fixture
def tool(tmp_path: Path):
    return context_prune(_make_soul(tmp_path))


@pytest.fixture
def wire_capture(monkeypatch):
    """Capture wire events sent by the tool."""
    captured: list[Any] = []

    def _capture(event: Any) -> None:
        captured.append(event)

    monkeypatch.setattr("kimi_cli.tools.context_prune.wire_send", _capture)
    return captured


class TestContextPruneTool:
    @pytest.mark.asyncio
    async def test_dry_run_reports_no_mutation(self, tool: context_prune, wire_capture):
        # Long system reminder so current tokens exceed the target
        await tool._soul.context.append_message(_system_reminder("x" * 5000))
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("hi"))

        result = await tool(
            Params(
                mode="prune",
                dry_run=True,
                target_token_count=1000,
                keep_recent_turns=1,
            )
        )

        assert "Dry run" in result.output
        assert len(tool._soul.context.history) == 3  # unchanged
        assert not wire_capture  # no status update on dry run

    @pytest.mark.asyncio
    async def test_prune_mode_mutates_history(self, tool: context_prune, wire_capture):
        await tool._soul.context.append_message(_system_reminder("x" * 5000))
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("hi"))

        result = await tool(
            Params(
                mode="prune",
                keep_recent_turns=1,
                target_token_count=1000,
                dry_run=False,
            )
        )

        assert not result.is_error
        assert len(tool._soul.context.history) == 2
        assert tool._soul.context.history[0].role == "user"
        assert tool._soul.context.history[1].role == "assistant"
        assert len(wire_capture) == 0  # no wire in tests

    @pytest.mark.asyncio
    async def test_compact_mode_calls_compaction(self, tool: context_prune):
        result = await tool(Params(mode="compact", dry_run=False))
        assert not result.is_error
        assert result.message == "Context compacted"
        assert tool._soul._compact_called() is True

    @pytest.mark.asyncio
    async def test_compact_mode_dry_run_does_not_call_compaction(self, tool: context_prune):
        result = await tool(Params(mode="compact", dry_run=True))
        assert "Dry run" in result.output
        assert tool._soul._compact_called() is False

    @pytest.mark.asyncio
    async def test_strip_reasoning_mode_removes_think_parts(self, tool: context_prune):
        await tool._soul.context.append_message(_assistant_with_think("old reasoning"))
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("tail"))

        result = await tool(
            Params(
                mode="strip_reasoning",
                keep_recent_turns=1,
                dry_run=False,
            )
        )

        assert not result.is_error
        assistant_msgs = [m for m in tool._soul.context.history if m.role == "assistant"]
        # The tail assistant is protected; the earlier one is stripped
        old_assistant = assistant_msgs[0]
        assert not any(isinstance(p, ThinkPart) for p in old_assistant.content)

    @pytest.mark.asyncio
    async def test_strip_reasoning_preserves_empty_reasoning_when_thinking_active(
        self, tmp_path: Path, wire_capture
    ):
        soul = _make_soul(
            tmp_path,
            thinking=True,
            runtime=SimpleNamespace(
                llm=SimpleNamespace(
                    max_context_size=128_000,
                    chat_provider=SimpleNamespace(
                        model_name="test-model",
                        thinking_effort="medium",
                    ),
                )
            ),
        )
        tool = context_prune(soul)
        await tool._soul.context.append_message(_assistant_with_think("old reasoning"))
        await tool._soul.context.append_message(_user("hello"))

        result = await tool(
            Params(
                mode="strip_reasoning",
                keep_recent_turns=1,
                dry_run=False,
            )
        )

        assert not result.is_error
        old_assistant = [m for m in tool._soul.context.history if m.role == "assistant"][0]
        assert any(isinstance(p, ThinkPart) and p.think == "" for p in old_assistant.content)

    @pytest.mark.asyncio
    async def test_tool_refuses_to_remove_only_turn(self, tool: context_prune):
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("hi"))

        result = await tool(Params(mode="prune", keep_recent_turns=1))

        assert result.is_error
        assert "only one user/assistant pair" in result.message

    @pytest.mark.asyncio
    async def test_tool_preserves_tool_call_pairs(self, tool: context_prune):
        await tool._soul.context.append_message(_user("read file"))
        await tool._soul.context.append_message(
            Message(
                role="assistant",
                content=[TextPart(text="Reading")],
                tool_calls=[
                    {"id": "call_1", "function": {"name": "ReadFile", "arguments": "{}"}}
                ],
            )
        )
        await tool._soul.context.append_message(_tool("x" * 2000, tool_call_id="call_1"))
        await tool._soul.context.append_message(_user("next"))

        result = await tool(
            Params(
                mode="prune",
                keep_recent_turns=1,
                target_token_count=1000,
                dry_run=False,
            )
        )

        assert not result.is_error
        tool_ids = {m.tool_call_id for m in tool._soul.context.history if m.role == "tool"}
        assert "call_1" in tool_ids

    @pytest.mark.asyncio
    async def test_tool_indexes_elided_content(self, tool: context_prune):
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("call tool"))
        await tool._soul.context.append_message(_tool("x" * 4000))
        await tool._soul.context.append_message(_user("tail"))

        result = await tool(
            Params(
                mode="prune",
                keep_recent_turns=1,
                target_token_count=1000,
                dry_run=False,
            )
        )

        assert not result.is_error
        # The oversized tool result should be elided and indexed for retrieval
        tool_turns = [t for t in tool._soul._history_index._turns if t["role"] == "tool"]
        assert len(tool_turns) >= 1
        assert "x" * 10 in tool_turns[0]["text"]

    @pytest.mark.asyncio
    async def test_invalid_keep_recent_turns_rejected(self, tool: context_prune):
        with pytest.raises(Exception):
            await tool(Params(mode="prune", keep_recent_turns=25))

    @pytest.mark.asyncio
    async def test_subagent_pruning_blocked_by_default(self, tool: context_prune):
        tool._soul.runtime.role = "subagent"
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("hi"))

        result = await tool(Params(mode="prune", keep_recent_turns=1))

        assert result.is_error
        assert "subagents" in result.message.lower()

    @pytest.mark.asyncio
    async def test_subagent_pruning_allowed_when_enabled(self, tool: context_prune):
        tool._soul.runtime.role = "subagent"
        tool._soul._loop_control.prune_subagents = True
        await tool._soul.context.append_message(_system_reminder("x" * 5000))
        await tool._soul.context.append_message(_user("hello"))
        await tool._soul.context.append_message(_assistant("hi"))

        result = await tool(
            Params(mode="prune", keep_recent_turns=1, target_token_count=1000)
        )

        assert not result.is_error
