from __future__ import annotations

import pytest
from kosong.message import Message, TextPart, ThinkPart, ToolCall

from kimi_cli.soul.kimisoul import _is_final_text_block


def _tool_call(name: str = "grep") -> ToolCall:
    return ToolCall(id="call-1", function=ToolCall.FunctionBody(name=name, arguments="{}"))


# ── _is_final_text_block unit tests ────────────────────────────────────────


def test_final_text_block_true_for_plain_text() -> None:
    msg = Message(role="assistant", content=[TextPart(text="All done.")])
    assert _is_final_text_block(msg) is True


def test_final_text_block_true_for_multi_part_text() -> None:
    msg = Message(
        role="assistant",
        content=[TextPart(text="First. "), TextPart(text="Second.")],
    )
    assert _is_final_text_block(msg) is True


def test_final_text_block_false_for_empty_message() -> None:
    msg = Message(role="assistant", content=[])
    assert _is_final_text_block(msg) is False


def test_final_text_block_false_for_none() -> None:
    assert _is_final_text_block(None) is False


def test_final_text_block_false_for_tool_calls() -> None:
    msg = Message(role="assistant", content=[TextPart(text="I will search.")], tool_calls=[_tool_call()])
    assert _is_final_text_block(msg) is False


def test_final_text_block_false_for_tool_calls_only() -> None:
    msg = Message(role="assistant", content=[], tool_calls=[_tool_call()])
    assert _is_final_text_block(msg) is False


def test_final_text_block_false_for_think_only() -> None:
    msg = Message(role="assistant", content=[ThinkPart(think="reasoning...")])
    assert _is_final_text_block(msg) is False


def test_final_text_block_false_when_text_then_think() -> None:
    msg = Message(
        role="assistant",
        content=[TextPart(text="Almost done."), ThinkPart(think="reasoning...")],
    )
    assert _is_final_text_block(msg) is False


def test_final_text_block_false_for_whitespace_only_text() -> None:
    msg = Message(role="assistant", content=[TextPart(text="   \n  ")])
    assert _is_final_text_block(msg) is False
