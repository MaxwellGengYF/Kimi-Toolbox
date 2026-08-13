"""End-to-end tests for context_prune through a KimiSoul-like setup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kosong.chat_provider.kimi import Kimi, _convert_message as _kimi_convert_message
from kosong.message import Message

from kimi_cli.soul.context import Context
from kimi_cli.soul.context_pruning import ContextPruner
from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.context_prune import context_prune, Params
from kimi_cli.wire.types import TextPart


def _user(text: str) -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _tool(text: str, tool_call_id: str = "call_1") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id=tool_call_id)


def _system_reminder(text: str) -> Message:
    return Message(
        role="user",
        content=[TextPart(text=f"<system-reminder>\n{text}\n</system-reminder>")],
    )


def _make_soul(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal soul wired to a real Context, HistoryIndex, and KimiToolset."""
    history_index_path = tmp_path / "history_index.json"
    history_index = HistoryIndex(persist_path=history_index_path)

    context_path = tmp_path / "context.jsonl"
    context_path.touch()
    context = Context(file_backend=context_path)

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
        thinking=False,
    )
    return soul


def _context_prune_toolset(soul: SimpleNamespace) -> KimiToolset:
    """Register context_prune on a KimiToolset and return it."""
    toolset = KimiToolset()
    toolset.add(context_prune(soul))
    return toolset


@pytest.mark.asyncio
async def test_context_prune_tool_reduces_tokens(tmp_path: Path, monkeypatch: Any) -> None:
    soul = _make_soul(tmp_path)
    toolset = _context_prune_toolset(soul)

    monkeypatch.setattr(
        "kimi_cli.tools.context_prune.get_wire_or_none", lambda: None
    )

    # Build a session whose oversized system reminder dominates the token count
    await soul.context.append_message(_system_reminder("x" * 8000))
    await soul.context.append_message(_user("hello"))
    await soul.context.append_message(_assistant("hi"))
    await soul.context.append_message(_user("tail"))

    before = soul.context.token_count_with_pending

    tool = toolset.find("context_prune")
    assert tool is not None
    result = await tool(
        Params(mode="prune", keep_recent_turns=1, target_token_count=1000)
    )

    assert not result.is_error
    after = soul.context.token_count_with_pending
    assert after < before


@pytest.mark.asyncio
async def test_context_prune_tool_elided_content_retrievable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    soul = _make_soul(tmp_path)
    toolset = _context_prune_toolset(soul)

    monkeypatch.setattr(
        "kimi_cli.tools.context_prune.get_wire_or_none", lambda: None
    )

    # Seed the history index with at least one other document so the BM25
    # searcher's candidate selection works during retrieval.
    soul._history_index.index_messages([_user("background context")])

    await soul.context.append_message(_user("hello"))
    await soul.context.append_message(_assistant("call tool"))
    await soul.context.append_message(_tool("elided_marker " + "x" * 4000))
    await soul.context.append_message(_user("tail"))

    tool = toolset.find("context_prune")
    assert tool is not None
    result = await tool(
        Params(mode="prune", keep_recent_turns=1, target_token_count=1000)
    )

    assert not result.is_error
    hits = soul._history_index.search("elided_marker", top_k=5)
    assert any("elided_marker" in hit["text"] for hit in hits)


@pytest.mark.asyncio
async def test_context_prune_tool_does_not_break_provider_conversion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    soul = _make_soul(tmp_path)
    toolset = _context_prune_toolset(soul)

    monkeypatch.setattr(
        "kimi_cli.tools.context_prune.get_wire_or_none", lambda: None
    )

    await soul.context.append_message(_user("read files"))
    await soul.context.append_message(
        Message(
            role="assistant",
            content=[TextPart(text="Reading")],
            tool_calls=[
                {"id": "call_1", "function": {"name": "ReadFile", "arguments": "{}"}}
            ],
        )
    )
    await soul.context.append_message(_tool("x" * 2000, tool_call_id="call_1"))
    await soul.context.append_message(_user("tail"))

    tool = toolset.find("context_prune")
    assert tool is not None
    result = await tool(
        Params(mode="prune", keep_recent_turns=1, target_token_count=1000)
    )

    assert not result.is_error

    provider = Kimi(
        model="kimi-k2-turbo-preview",
        api_key="dummy",
        base_url="http://localhost",
    )
    try:
        converted = [_kimi_convert_message(m) for m in soul.context.history]
        tool_call_ids = {
            tc["id"]
            for m in converted
            if m.get("role") == "assistant"
            for tc in m.get("tool_calls", [])
        }
        result_ids = {
            m["tool_call_id"]
            for m in converted
            if m.get("role") == "tool"
        }
        assert tool_call_ids == result_ids
    finally:
        await provider.aclose()
