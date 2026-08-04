"""Snapshot tests for OpenAI Legacy (Chat Completions API) chat provider."""

import json

import respx
from common import COMMON_CASES, Case, make_chat_completion_response, run_test_cases
from httpx import Response
from inline_snapshot import snapshot

from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy
from kosong.message import Message, TextPart, ThinkPart, ToolCall

TEST_CASES: dict[str, Case] = {**COMMON_CASES}


async def test_openai_legacy_message_conversion():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("gpt-4.1"))
        )
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
        results = await run_test_cases(mock, provider, TEST_CASES, ("messages", "tools"))

        assert results == snapshot(
            {
                "simple_user_message": {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "tools": [],
                },
                "multi_turn_conversation": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                        {"role": "user", "content": "And 3+3?"},
                    ],
                    "tools": [],
                },
                "multi_turn_with_system": {
                    "messages": [
                        {"role": "system", "content": "You are a math tutor."},
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                        {"role": "user", "content": "And 3+3?"},
                    ],
                    "tools": [],
                },
                "image_url": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://example.com/image.png",
                                        "id": None,
                                    },
                                },
                            ],
                        }
                    ],
                    "tools": [],
                },
                "tool_definition": {
                    "messages": [{"role": "user", "content": "Add 2 and 3"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "add",
                                "description": "Add two integers.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {
                                            "type": "integer",
                                            "description": "First number",
                                        },
                                        "b": {
                                            "type": "integer",
                                            "description": "Second number",
                                        },
                                    },
                                    "required": ["a", "b"],
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "description": "Multiply two integers.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "integer", "description": "First number"},
                                        "b": {"type": "integer", "description": "Second number"},
                                    },
                                    "required": ["a", "b"],
                                },
                            },
                        },
                    ],
                },
                "tool_call_with_image": {
                    "messages": [
                        {"role": "user", "content": "Add 2 and 3"},
                        {
                            "role": "assistant",
                            "content": "I'll add those numbers for you.",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "call_abc123",
                                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": [
                                {"type": "text", "text": "5"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://example.com/image.png",
                                        "id": None,
                                    },
                                },
                            ],
                            "tool_call_id": "call_abc123",
                        },
                    ],
                    "tools": [],
                },
                "tool_call": {
                    "messages": [
                        {"role": "user", "content": "Add 2 and 3"},
                        {
                            "role": "assistant",
                            "content": "I'll add those numbers for you.",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "call_abc123",
                                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                                }
                            ],
                        },
                        {"role": "tool", "content": "5", "tool_call_id": "call_abc123"},
                    ],
                    "tools": [],
                },
                "parallel_tool_calls": {
                    "messages": [
                        {"role": "user", "content": "Calculate 2+3 and 4*5"},
                        {
                            "role": "assistant",
                            "content": "I'll calculate both.",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "call_add",
                                    "function": {
                                        "name": "add",
                                        "arguments": '{"a": 2, "b": 3}',
                                    },
                                },
                                {
                                    "type": "function",
                                    "id": "call_mul",
                                    "function": {
                                        "name": "multiply",
                                        "arguments": '{"a": 4, "b": 5}',
                                    },
                                },
                            ],
                        },
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<system-reminder>This is a system reminder"
                                    "</system-reminder>",
                                },
                                {"type": "text", "text": "5"},
                            ],
                            "tool_call_id": "call_add",
                        },
                        {
                            "role": "tool",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "<system-reminder>This is a system reminder"
                                    "</system-reminder>",
                                },
                                {"type": "text", "text": "20"},
                            ],
                            "tool_call_id": "call_mul",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "add",
                                "description": "Add two integers.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "integer", "description": "First number"},
                                        "b": {"type": "integer", "description": "Second number"},
                                    },
                                    "required": ["a", "b"],
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "description": "Multiply two integers.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "integer", "description": "First number"},
                                        "b": {"type": "integer", "description": "Second number"},
                                    },
                                    "required": ["a", "b"],
                                },
                            },
                        },
                    ],
                },
            }
        )


async def test_openai_legacy_reasoning_content():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Thinking..."), TextPart(text="4.")],
            ),
            Message(role="user", content="Thanks!"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == snapshot(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4.",
                    "reasoning_content": "Thinking...",
                },
                {"role": "user", "content": "Thanks!"},
            ]
        )


async def test_openai_legacy_empty_reasoning_content_is_round_tripped():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[ThinkPart(think=""), TextPart(text="4.")],
            ),
            Message(role="user", content="Thanks!"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4.",
                "reasoning_content": "",
            },
            {"role": "user", "content": "Thanks!"},
        ]


async def test_openai_legacy_reasoning_content_on_all_assistant_messages():
    """When reasoning_key is configured, every assistant message carries
    reasoning_content (empty string when the message has no reasoning) so that
    DeepSeek/Moonshot-compatible backends see a consistent field across
    assistant turns. User/tool messages must not carry the field."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4."),
            Message(role="user", content="Thanks!"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4.",
                "reasoning_content": "",
            },
            {"role": "user", "content": "Thanks!"},
        ]


async def test_openai_legacy_reasoning_content_on_all_assistant_messages_when_thinking_enabled():
    """When thinking mode is enabled (history contains ThinkPart), every assistant
    message carries reasoning_content so Moonshot-compatible backends see a
    consistent field across thinking-mode assistant turns.
    """
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Thinking..."), TextPart(text="4.")],
            ),
            Message(role="user", content="And 3+3?"),
            Message(role="assistant", content="6."),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "4.",
                "reasoning_content": "Thinking...",
            },
            {"role": "user", "content": "And 3+3?"},
            {
                "role": "assistant",
                "content": "6.",
                "reasoning_content": "",
            },
        ]


async def test_openai_legacy_moonshot_disables_auto_extra_body():
    """Moonshot models use the standard OpenAI reasoning wire format; the
    provider-specific extra_body keys are disabled by default.
    """
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="kimi-k2.5",
            api_key="test-key",
            base_url="https://api.moonshot.ai/v1",
            stream=False,
            reasoning_key="reasoning_content",
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "thinking" not in body
        assert "reasoning" not in body
        assert "chat_template_kwargs" not in body


async def test_openai_legacy_assistant_tool_call_omits_empty_content():
    """When an assistant message has tool calls and no visible text content,
    the `content` field must be omitted entirely to avoid backend validation
    errors such as Moonshot's "text content is empty".
    """
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="kimi-k2.5", api_key="test-key", stream=False
        )
        history = [
            Message(role="user", content="Call the add tool"),
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(
                            name="add", arguments='{"a": 2, "b": 3}'
                        ),
                    )
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"][1] == {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_abc123",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        }


async def test_openai_legacy_assistant_tool_call_with_reasoning_only():
    """When an assistant message has tool calls and reasoning but no visible
    text, both `reasoning_content` is preserved and `content` is omitted.
    """
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="kimi-k2.5",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="Think and call the add tool"),
            Message(
                role="assistant",
                content=[ThinkPart(think="I should call the add tool.")],
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(
                            name="add", arguments='{"a": 2, "b": 3}'
                        ),
                    )
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"][1] == {
            "role": "assistant",
            "reasoning_content": "I should call the add tool.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_abc123",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        }


async def test_openai_legacy_generation_kwargs():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="gpt-4.1", api_key="test-key", stream=False
        ).with_generation_kwargs(temperature=0.7, max_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert (body["temperature"], body["max_tokens"]) == snapshot((0.7, 2048))


async def test_openai_legacy_with_thinking():
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False).with_thinking(
            "high"
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["reasoning_effort"] == snapshot("high")


async def test_openai_legacy_supported_efforts_clamps_max():
    """A model that does not accept ``max`` must clamp it to ``high``."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="gpt-4.1",
            api_key="test-key",
            stream=False,
            supported_efforts={"low", "medium", "high"},
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["reasoning_effort"] == snapshot("high")


async def test_openai_legacy_supported_efforts_passes_xhigh():
    """A model configured with the full effort set must pass ``xhigh`` through."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="gpt-5.1-codex-max",
            api_key="test-key",
            stream=False,
            supported_efforts={"low", "medium", "high", "xhigh", "max"},
        ).with_thinking("xhigh")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["reasoning_effort"] == snapshot("xhigh")


async def test_openai_legacy_auto_reasoning_effort_when_history_has_think_part():
    """When reasoning_effort is not set but history contains ThinkPart and reasoning_key is
    configured, reasoning_effort should be auto-set to avoid server validation errors.

    Reproduces: https://github.com/MoonshotAI/kimi-cli/issues/1616
    """
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        # Provider with reasoning_key but NO explicit reasoning_effort
        provider = OpenAILegacy(
            model="kimi-k2.5",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Let me think..."), TextPart(text="Hi!")],
            ),
            Message(role="user", content="How are you?"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        # reasoning_effort should be auto-set because history contains ThinkPart
        assert body["reasoning_effort"] == "medium"
        # reasoning_content should still be present in the message
        assert body["messages"][1]["reasoning_content"] == "Let me think..."


async def test_openai_legacy_no_auto_reasoning_effort_without_think_part():
    """When history has no ThinkPart, reasoning_effort should remain unset."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="kimi-k2.5",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="How are you?"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "reasoning_effort" not in body


async def test_openai_legacy_with_parallel_tool_calls_disabled():
    """with_parallel_tool_calls(False) should send parallel_tool_calls=false."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="gpt-4.1", api_key="test-key", stream=False
        ).with_parallel_tool_calls(False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["parallel_tool_calls"] is False


async def test_openai_legacy_with_parallel_tool_calls_enabled():
    """with_parallel_tool_calls(True) should omit parallel_tool_calls."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
            .with_parallel_tool_calls(False)
            .with_parallel_tool_calls(True)
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "parallel_tool_calls" not in body


async def test_openai_legacy_no_auto_reasoning_effort_without_reasoning_key():
    """When reasoning_key is not configured, reasoning_effort should not be auto-set
    even if history has ThinkPart (ThinkPart would be silently dropped)."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        # No reasoning_key configured
        provider = OpenAILegacy(
            model="some-model",
            api_key="test-key",
            stream=False,
        )
        history = [
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Thinking..."), TextPart(text="Hi!")],
            ),
            Message(role="user", content="How are you?"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "reasoning_effort" not in body


async def test_openai_legacy_extra_body_defaults():
    """When reasoning_key is set, all auto extra_body keys are included by default."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == {"type": "disabled"}
        assert body["reasoning"] == {"effort": "no_think"}
        assert body["chat_template_kwargs"] == {"reasoning_effort": "no_think"}


async def test_openai_legacy_extra_body_disabled():
    """When all extra_body options are disabled, extra_body is not sent."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
            openai_settings={
                "thinking": False,
                "reasoning": False,
                "chat_template_kwargs": False,
            },
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "thinking" not in body
        assert "reasoning" not in body
        assert "chat_template_kwargs" not in body


async def test_openai_legacy_extra_body_partial():
    """Only enabled extra_body options are included."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
            openai_settings={"thinking": False},
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "thinking" not in body
        assert body["reasoning"] == {"effort": "no_think"}
        assert body["chat_template_kwargs"] == {"reasoning_effort": "no_think"}


async def test_openai_legacy_extra_body_user_supplied_preserved_when_auto_disabled():
    """User-supplied extra_body keys are preserved even when the auto-generated key
    is disabled via openai_settings."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = OpenAILegacy(
            model="deepseek-reasoner",
            api_key="test-key",
            stream=False,
            reasoning_key="reasoning_content",
            openai_settings={"thinking": False},
        ).with_generation_kwargs(extra_body={"thinking": {"keep": "all"}})
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == {"keep": "all"}
        assert "reasoning" in body
        assert "chat_template_kwargs" in body


async def test_openai_legacy_session_id_user_in_body():
    """The session id must reach the wire as ``user``."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("gpt-4.1"))
        )
        provider = OpenAILegacy(
            model="gpt-4.1", api_key="test-key", stream=False
        ).with_generation_kwargs(user="sess-abc-123")
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["user"] == "sess-abc-123"


async def test_openai_legacy_without_user_omits_field():
    """No session id → no ``user`` in the request body."""
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("gpt-4.1"))
        )
        provider = OpenAILegacy(model="gpt-4.1", api_key="test-key", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "user" not in body
