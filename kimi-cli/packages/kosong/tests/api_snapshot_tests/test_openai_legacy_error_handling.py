"""Tests for OpenAILegacy provider format-error handling.

These tests guard against malformed messages crashing the client or causing
API 400 errors.  Invalid data is returned to the LLM as error text so the
model can recover.
"""

from __future__ import annotations

from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message, TextPart, ToolCall


class TestInvalidToolCallArguments:
    """When an assistant message contains a tool call with malformed JSON arguments,
    _convert_message must not crash.  It should emit an error text block and
    replace the broken arguments with '{}' so the LLM sees the problem."""

    def test_invalid_json_returns_error_to_llm(self) -> None:
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
        )
        message = Message(
            role="assistant",
            content=[TextPart(text="Let me call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_bad",
                    function=ToolCall.FunctionBody(
                        name="add", arguments='{"a": 1, "b": 2'  # missing closing }
                    ),
                )
            ],
        )
        result = provider._convert_message(message)
        assert result["role"] == "assistant"
        content = result["content"]
        # loads_relaxed repairs the broken JSON to a valid dict, so no error is emitted.
        # When there is exactly one TextPart, Message serializes it as a string.
        assert content == "Let me call a tool."
        # loads_relaxed validates successfully (repairs internally), but the original
        # argument string is preserved in the output.
        assert result["tool_calls"][0]["function"]["arguments"] == '{"a": 1, "b": 2'

    def test_non_dict_json_returns_error_to_llm(self) -> None:
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
        )
        message = Message(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCall(
                    id="call_bad2",
                    function=ToolCall.FunctionBody(
                        name="add", arguments="[1, 2]"  # array, not object
                    ),
                )
            ],
        )
        result = provider._convert_message(message)
        assert result["role"] == "assistant"
        content = result["content"]
        # When there is exactly one TextPart, Message serializes it as a string.
        assert isinstance(content, str)
        assert "must be a JSON object" in content
        assert result["tool_calls"][0]["function"]["arguments"] == "{}"


class TestGracefulDataErrors:
    """Data validation errors during message conversion must return text blocks
    to the LLM instead of crashing the client."""

    def test_missing_tool_call_id_returns_error_to_llm(self) -> None:
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
        )
        message = Message(
            role="tool",
            content=[TextPart(text="The result is 5")],
            tool_call_id=None,
        )
        result = provider._convert_message(message)
        assert result["role"] == "user"
        assert "missing `tool_call_id`" in result["content"]
        assert "The result is 5" in result["content"]
