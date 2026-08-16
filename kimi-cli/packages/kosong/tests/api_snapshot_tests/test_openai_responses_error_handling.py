"""Tests for OpenAIResponses provider format-error handling.

These tests guard against malformed messages crashing the client or causing
API 400 errors.  Invalid data is returned to the LLM as error text so the
model can recover.
"""

from __future__ import annotations

from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message, TextPart, ToolCall


class TestInvalidToolCallArguments:
    """When an assistant message contains a tool call with malformed JSON arguments,
    _convert_message must not crash.  It should emit an error text block and
    replace the broken arguments with '{}' so the LLM sees the problem."""

    def test_invalid_json_returns_error_to_llm(self) -> None:
        provider = OpenAIResponses(
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
        # loads_relaxed repairs the broken JSON to a valid dict, so no error is emitted.
        # Result is: message item with original text, then function_call with repaired args.
        assert len(result) == 2
        msg_item = result[0]
        assert msg_item["type"] == "message"
        assert msg_item["role"] == "assistant"
        content = msg_item["content"]
        assert len(content) == 1
        assert content[0]["type"] == "output_text"
        assert content[0]["text"] == "Let me call a tool."
        # loads_relaxed validates successfully (repairs internally), but the original
        # argument string is preserved in the output.
        assert result[1]["type"] == "function_call"
        assert result[1]["arguments"] == '{"a": 1, "b": 2'

    def test_non_dict_json_returns_error_to_llm(self) -> None:
        provider = OpenAIResponses(
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
        assert len(result) == 2
        msg_item = result[0]
        assert msg_item["type"] == "message"
        assert msg_item["role"] == "assistant"
        content = msg_item["content"]
        assert len(content) == 1
        assert content[0]["type"] == "output_text"
        assert "must be a JSON object" in content[0]["text"]
        assert result[1]["type"] == "function_call"
        assert result[1]["arguments"] == "{}"


class TestGracefulDataErrors:
    """Data validation errors during message conversion must return text blocks
    to the LLM instead of crashing the client."""

    def test_missing_tool_call_id_returns_error_to_llm(self) -> None:
        provider = OpenAIResponses(
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
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"
        assert "missing `tool_call_id`" in result[0]["content"]
        assert "The result is 5" in result[0]["content"]
