"""Provider-specific invariant tests for context_prune output.

These tests exercise the low-level provider conversion code to confirm that
a pruned history still satisfies each backend's structural rules:

* Kimi thinking mode requires ``reasoning_content`` on assistant messages.
* OpenAI-compatible APIs require every assistant ``tool_calls`` entry to have
  a matching ``role="tool"`` message with the same ``tool_call_id``.
* Anthropic merges consecutive tool-result user messages into one user message.
"""

from __future__ import annotations

import pytest
from kosong.chat_provider.kimi import Kimi, _convert_message as _kimi_convert_message
from kosong.contrib.chat_provider.anthropic import (
    Anthropic,
    _is_tool_result_only,
)
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message, TextPart, ThinkPart


def _assistant_with_tool_calls(*call_ids: str) -> Message:
    return Message(
        role="assistant",
        content=[TextPart(text="Using tools")],
        tool_calls=[
            {"id": call_id, "function": {"name": "ReadFile", "arguments": "{}"}}
            for call_id in call_ids
        ],
    )


def _tool_result(call_id: str, text: str) -> Message:
    return Message(
        role="tool",
        content=[TextPart(text=text)],
        tool_call_id=call_id,
    )


@pytest.mark.asyncio
async def test_kimi_thinking_mode_requires_reasoning_backpass() -> None:
    provider = Kimi(
        model="kimi-k2-turbo-preview",
        api_key="dummy",
        base_url="http://localhost",
    )
    try:
        # Full reasoning content is forwarded back whenever the message
        # carries a ThinkPart, regardless of thinking configuration.
        msg = Message(role="assistant", content=[ThinkPart(think="deep reasoning")])
        converted = _kimi_convert_message(msg)
        assert converted.get("reasoning_content") == "deep reasoning"

        # After context_prune strips reasoning in thinking mode, an empty
        # reasoning_content placeholder is preserved so the API still receives
        # the required field.
        stripped = Message(role="assistant", content=[ThinkPart(think="")])
        converted_stripped = _kimi_convert_message(stripped)
        assert converted_stripped.get("reasoning_content") == ""

        # With preserved thinking active, assistant messages without reasoning
        # are backfilled with an empty reasoning_content field.
        plain = Message(role="assistant", content=[TextPart(text="done")])
        converted_plain = _kimi_convert_message(plain, preserved_thinking_enabled=True)
        assert converted_plain.get("reasoning_content") == ""

        # Without preserved thinking, plain assistant messages carry no field.
        converted_plain_off = _kimi_convert_message(plain)
        assert "reasoning_content" not in converted_plain_off
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_openai_tool_pair_invariant_preserved() -> None:
    provider = OpenAILegacy(
        model="gpt-4",
        api_key="dummy",
        base_url="http://localhost",
    )
    try:
        history = [
            _assistant_with_tool_calls("call_1"),
            _tool_result("call_1", "file contents"),
        ]
        converted = [provider._convert_message(m) for m in history]

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        tool_msg = next(m for m in converted if m["role"] == "tool")

        assert any(tc["id"] == "call_1" for tc in assistant_msg["tool_calls"])
        assert tool_msg["tool_call_id"] == "call_1"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_openai_missing_tool_result_becomes_error_message() -> None:
    provider = OpenAILegacy(
        model="gpt-4",
        api_key="dummy",
        base_url="http://localhost",
    )
    try:
        # A tool message without a tool_call_id would cause a provider 400.
        # The conversion layer turns it into a user-facing error instead.
        bad_tool = Message(role="tool", content=[TextPart(text="orphan result")])
        converted = provider._convert_message(bad_tool)

        assert converted["role"] == "user"
        assert "tool_call_id" in converted["content"].lower()
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_anthropic_tool_result_merge() -> None:
    provider = Anthropic(
        model="claude-3-5-sonnet",
        api_key="dummy",
        base_url="http://localhost",
        default_max_tokens=4096,
    )
    try:
        history = [
            _assistant_with_tool_calls("call_1", "call_2"),
            _tool_result("call_1", "result one"),
            _tool_result("call_2", "result two"),
        ]

        # Replicate the merge logic from Anthropic.generate()
        messages: list[dict] = []
        for message in history:
            converted = provider._convert_message(message)
            if (
                messages
                and converted["role"] == "user"
                and messages[-1]["role"] == "user"
                and _is_tool_result_only(messages[-1]["content"])
                and _is_tool_result_only(converted["content"])
            ):
                prev_content = messages[-1]["content"]
                new_content = converted["content"]
                messages[-1]["content"] = [*prev_content, *new_content]
            else:
                messages.append(converted)

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1

        tool_result_blocks = [
            b for b in user_messages[0]["content"] if b["type"] == "tool_result"
        ]
        assert len(tool_result_blocks) == 2
        assert {b["tool_use_id"] for b in tool_result_blocks} == {"call_1", "call_2"}
    finally:
        await provider.aclose()
