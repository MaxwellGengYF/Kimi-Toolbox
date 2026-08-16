"""Snapshot tests for Kimi chat provider."""

import json

import respx
from common import COMMON_CASES, Case, make_chat_completion_response, run_test_cases
from httpx import Response
from inline_snapshot import snapshot
from openai.types.chat import ChatCompletionChunk

from kosong.chat_provider.kimi import (  # pyright: ignore[reportPrivateUsage]
    Kimi,
    KimiStreamedMessage,
    _convert_tool,
)
from kosong.message import Message, TextPart, ThinkPart, ToolCall
from kosong.tooling import Tool

BUILTIN_TOOL = Tool(
    name="$web_search",
    description="Search the web",
    parameters={"type": "object", "properties": {}},
)

TEST_CASES: dict[str, Case] = {
    **COMMON_CASES,
    "builtin_tool": {
        "history": [Message(role="user", content="Search for something")],
        "tools": [BUILTIN_TOOL],
    },
    "assistant_with_reasoning": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(think="Let me think..."),
                    TextPart(text="The answer is 4."),
                ],
            ),
            Message(role="user", content="Thanks!"),
        ],
    },
    "assistant_with_empty_reasoning": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(think=""),
                    TextPart(text="The answer is 4."),
                ],
            ),
            Message(role="user", content="Thanks!"),
        ],
    },
    "assistant_tool_call_without_text": {
        "history": [
            Message(role="user", content="Call the add tool"),
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                    )
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ],
    },
    "assistant_tool_call_with_reasoning_only": {
        "history": [
            Message(role="user", content="Think and call the add tool"),
            Message(
                role="assistant",
                content=[ThinkPart(think="I should call the tool.")],
                tool_calls=[
                    ToolCall(
                        id="call_abc123",
                        function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                    )
                ],
            ),
            Message(role="tool", content="5", tool_call_id="call_abc123"),
        ],
    },
}


async def test_kimi_message_conversion():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("kimi-k2"))
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        results = await run_test_cases(mock, provider, TEST_CASES, ("messages", "tools"))

        assert results == snapshot(
            {
                "simple_user_message": {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ],
                },
                "multi_turn_conversation": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                        {"role": "user", "content": "And 3+3?"},
                    ],
                },
                "multi_turn_with_system": {
                    "messages": [
                        {"role": "system", "content": "You are a math tutor."},
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                        {"role": "user", "content": "And 3+3?"},
                    ],
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
                "builtin_tool": {
                    "messages": [{"role": "user", "content": "Search for something"}],
                    "tools": [
                        {
                            "type": "builtin_function",
                            "function": {"name": "$web_search"},
                        }
                    ],
                },
                "assistant_with_reasoning": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {
                            "role": "assistant",
                            "content": "The answer is 4.",
                            "reasoning_content": "Let me think...",
                        },
                        {"role": "user", "content": "Thanks!"},
                    ],
                },
                "assistant_with_empty_reasoning": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {
                            "role": "assistant",
                            "content": "The answer is 4.",
                            "reasoning_content": "",
                        },
                        {"role": "user", "content": "Thanks!"},
                    ],
                },
                "assistant_tool_call_without_text": {
                    "messages": [
                        {"role": "user", "content": "Call the add tool"},
                        {
                            "role": "assistant",
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
                },
                "assistant_tool_call_with_reasoning_only": {
                    "messages": [
                        {"role": "user", "content": "Think and call the add tool"},
                        {
                            "role": "assistant",
                            "reasoning_content": "I should call the tool.",
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
                },
            }
        )


async def test_kimi_generation_kwargs():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_generation_kwargs(temperature=0.7, max_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["temperature"] == snapshot(0.7)
        # `max_tokens` is the legacy alias; the provider rewrites it on the
        # wire so reasoning models do not interpret it as a thinking-budget cap.
        assert body["max_completion_tokens"] == snapshot(2048)
        assert "max_tokens" not in body


async def test_kimi_prefers_max_completion_tokens_over_max_tokens():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_generation_kwargs(max_completion_tokens=2048, max_tokens=4096)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["max_completion_tokens"] == snapshot(2048)
        assert "max_tokens" not in body


async def test_kimi_sends_no_completion_token_cap_by_default():
    """Without an explicit budget no cap goes on the wire — the upstream is
    responsible for clamping against the model context window."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body


async def test_kimi_omits_tools_when_empty():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "tools" not in body


async def test_kimi_with_thinking():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        # The effort rides inside the thinking object; no top-level
        # reasoning_effort is sent for this contract.
        assert "reasoning_effort" not in body
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "high"})


async def test_kimi_with_thinking_max():
    """Concrete effort strings pass through verbatim — ``max`` is not
    clamped to ``high``."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "reasoning_effort" not in body
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "max"})


async def test_kimi_with_thinking_xhigh_passes_through_verbatim():
    """The provider performs no effort clamping — model compatibility and
    fallback are resolved above the provider boundary, so even ``xhigh`` is
    forwarded verbatim."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_thinking("xhigh")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "xhigh"})


async def test_kimi_with_thinking_off_disables_without_stale_effort():
    """Switching efforts must replace the thinking object wholesale — a
    stale ``effort`` from a previous call must never linger on a disabled
    thinking config."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_thinking("high")
            .with_thinking("off")
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "reasoning_effort" not in body
        assert body["thinking"] == snapshot({"type": "disabled"})


async def test_kimi_with_thinking_carries_over_keep():
    """A ``thinking.keep`` installed via with_extra_body survives later
    with_thinking calls (including ``off``)."""
    provider = (
        Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        .with_extra_body({"thinking": {"keep": "all"}})
        .with_thinking("high")
    )
    assert provider.thinking_effort == "high"
    assert provider._generation_kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "effort": "high", "keep": "all"}
    }
    disabled = provider.with_thinking("off")
    assert disabled._generation_kwargs["extra_body"] == {
        "thinking": {"type": "disabled", "keep": "all"}
    }


def test_kimi_thinking_effort_property_round_trip():
    provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
    assert provider.thinking_effort is None
    assert provider.with_thinking("max").thinking_effort == "max"
    assert provider.with_thinking("xhigh").thinking_effort == "xhigh"
    assert provider.with_thinking("high").thinking_effort == "high"
    assert provider.with_thinking("medium").thinking_effort == "medium"
    assert provider.with_thinking("low").thinking_effort == "low"
    assert provider.with_thinking("off").thinking_effort == "off"

async def test_kimi_reasoning_content_passed_back_only_on_messages_with_reasoning():
    """Assistant messages that carried reasoning pass it back verbatim.
    Messages without reasoning do NOT get an empty backfill merely because
    thinking is enabled — that is reserved for preserved-thinking mode
    (``thinking.keep == "all"``)."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_thinking("high")
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
        assert body["messages"] == snapshot(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4.",
                    "reasoning_content": "Thinking...",
                },
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6."},
            ]
        )


async def test_kimi_reasoning_content_backfilled_when_preserved_thinking_active():
    """With ``thinking.keep == "all"`` and thinking not disabled, every
    assistant message carries ``reasoning_content`` (empty string when the
    message has no reasoning)."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_thinking("high")
            .with_extra_body({"thinking": {"keep": "all"}})
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
        assert body["messages"] == snapshot(
            [
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
        )


async def test_kimi_reasoning_content_backfilled_when_keep_all_without_thinking_type():
    """``keep == "all"`` alone (no explicit ``thinking.type``) still counts
    as preserved thinking — only an explicit ``"disabled"`` suppresses it."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_extra_body({"thinking": {"keep": "all"}})
        history = [Message(role="assistant", content="Done.")]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == snapshot(
            [{"role": "assistant", "content": "Done.", "reasoning_content": ""}]
        )


async def test_kimi_no_reasoning_content_backfill_for_other_keep_values():
    """keep values other than ``"all"`` must not trigger the backfill."""
    for keep in (None, False, "off"):
        with respx.mock(base_url="https://api.moonshot.ai") as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=Response(200, json=make_chat_completion_response())
            )
            provider = Kimi(
                model="kimi-k2-turbo-preview", api_key="test-key", stream=False
            ).with_extra_body({"thinking": {"type": "enabled", "keep": keep}})
            history = [
                Message(
                    role="assistant",
                    content=[],
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=ToolCall.FunctionBody(
                                name="lookup", arguments='{"q":"test"}'
                            ),
                        )
                    ],
                )
            ]
            stream = await provider.generate("", [], history)
            async for _ in stream:
                pass
            body = json.loads(mock.calls.last.request.content.decode())
            assert "reasoning_content" not in body["messages"][0]


async def test_kimi_no_reasoning_content_backfill_when_thinking_disabled():
    """``thinking.type == "disabled"`` suppresses the backfill even with
    ``keep == "all"``."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_extra_body({"thinking": {"type": "disabled", "keep": "all"}})
        history = [
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCall.FunctionBody(name="lookup", arguments='{"q":"test"}'),
                    )
                ],
            )
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "reasoning_content" not in body["messages"][0]


async def test_kimi_no_reasoning_content_on_non_assistant_messages():
    """The preserved-thinking backfill only applies to assistant messages."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_extra_body({"thinking": {"type": "enabled", "keep": "all"}})
        history = [
            Message(role="system", content="System."),
            Message(role="user", content="User."),
            Message(role="tool", content="Tool result.", tool_call_id="call_1"),
        ]
        stream = await provider.generate("You are helpful.", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        for message in body["messages"]:
            assert "reasoning_content" not in message

async def test_kimi_with_extra_body_thinking_deep_merge():
    """with_extra_body must deep-merge the ``thinking`` sub-dict so that
    a later call adding ``thinking.keep`` does not erase ``thinking.type``
    set by an earlier ``with_thinking`` call."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_thinking("high")
            .with_extra_body({"thinking": {"keep": "all"}})
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "high", "keep": "all"})


async def test_kimi_with_extra_body_thinking_empty_dict_is_noop():
    """Passing ``{"thinking": {}}`` must leave an earlier ``thinking.type``
    intact. An empty ``thinking`` patch is a no-op, not a clearing signal."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_thinking("high")
            .with_extra_body({"thinking": {}})
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "high"})


async def test_kimi_with_extra_body_thinking_starts_from_empty_dict():
    """Seeding ``thinking`` with ``{}`` first, then populating it via
    ``with_thinking`` must produce the populated config — the empty seed
    must not block subsequent field additions."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_extra_body({"thinking": {}})
            .with_thinking("high")
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "effort": "high"})


async def test_kimi_with_extra_body_non_thinking_key_shallow_merge():
    """Only the ``thinking`` key gets deep-merge special-casing; other
    top-level extra_body keys still follow the previous shallow-merge
    semantics (last writer wins)."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = (
            Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
            .with_extra_body({"custom": {"a": 1}})  # pyright: ignore[reportArgumentType]
            .with_extra_body({"custom": {"b": 2}})  # pyright: ignore[reportArgumentType]
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["custom"] == snapshot({"b": 2})


async def test_kimi_normalizes_invalid_tool_call_ids():
    """Histories persisted from other providers can carry tool-call ids that
    Moonshot rejects (e.g. ``Read:9``). They are sanitized to
    ``[a-zA-Z0-9_-]`` in both the assistant tool_calls and the matching tool
    messages."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        history = [
            Message(role="user", content="Read a file"),
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="Read:9",
                        function=ToolCall.FunctionBody(
                            name="Read", arguments='{"path":"/tmp/file"}'
                        ),
                    )
                ],
            ),
            Message(role="tool", content="content", tool_call_id="Read:9"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"] == snapshot(
            [
                {"role": "user", "content": "Read a file"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "Read_9",
                            "function": {"name": "Read", "arguments": '{"path":"/tmp/file"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "content", "tool_call_id": "Read_9"},
            ]
        )
        # The caller's history objects must not be mutated.
        assert history[1].tool_calls is not None
        assert history[1].tool_calls[0].id == "Read:9"
        assert history[2].tool_call_id == "Read:9"


async def test_kimi_tool_call_ids_truncated_and_deduped():
    """Ids longer than 64 chars are truncated; collisions after sanitization
    are made unique with ``_2``/``_3``... suffixes."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        first = "a" * 100
        second = "a" * 99 + ":x"  # sanitizes to the same 64-char id as `first`
        history = [
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(id=first, function=ToolCall.FunctionBody(name="f", arguments="{}")),
                    ToolCall(id=second, function=ToolCall.FunctionBody(name="g", arguments="{}")),
                ],
            ),
            Message(role="tool", content="1", tool_call_id=first),
            Message(role="tool", content="2", tool_call_id=second),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        messages = body["messages"]
        tool_calls = messages[0]["tool_calls"]
        assert tool_calls[0]["id"] == snapshot("a" * 64)
        assert tool_calls[1]["id"] == snapshot("a" * 62 + "_2")
        assert messages[1]["tool_call_id"] == "a" * 64
        assert messages[2]["tool_call_id"] == "a" * 62 + "_2"


async def test_kimi_valid_tool_call_ids_pass_through_unchanged():
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response())
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        history = [
            Message(
                role="assistant",
                content=[],
                tool_calls=[
                    ToolCall(
                        id="call_abc-123_XYZ",
                        function=ToolCall.FunctionBody(name="f", arguments="{}"),
                    )
                ],
            ),
            Message(role="tool", content="1", tool_call_id="call_abc-123_XYZ"),
        ]
        stream = await provider.generate("", [], history)
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["messages"][0]["tool_calls"][0]["id"] == "call_abc-123_XYZ"
        assert body["messages"][1]["tool_call_id"] == "call_abc-123_XYZ"


def test_kimi_convert_tool_dereferences_definitions_and_normalizes():
    """Draft-7 ``definitions``/``$ref`` indirections are inlined before
    type-filling, the resolved bucket is dropped, and enum-only schemas in
    exotic positions (``prefixItems``) still get a ``type``."""
    tool = Tool(
        name="choose_mode",
        description="Choose a mode.",
        parameters={
            "type": "object",
            "properties": {
                "mode": {"$ref": "#/definitions/Mode"},
                "tuple": {"prefixItems": [{"enum": ["left", "right"]}]},
            },
            "definitions": {
                "Mode": {"enum": ["fast", "safe"]},
            },
        },
    )
    converted = _convert_tool(tool)
    assert converted["function"].get("parameters") == snapshot(
        {
            "type": "object",
            "properties": {
                "mode": {"enum": ["fast", "safe"], "type": "string"},
                "tuple": {
                    "type": "array",
                    "prefixItems": [{"enum": ["left", "right"], "type": "string"}],
                },
            },
        }
    )
    # The source tool must not be mutated.
    assert "definitions" in tool.parameters


def _make_stream_chunk(tool_calls: list[dict], **choice_extra) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-kimi-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "kimi-k2-turbo-preview",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": tool_calls},
                    "finish_reason": choice_extra.get("finish_reason"),
                }
            ],
        }
    )


async def _collect_stream_parts(chunks: list[ChatCompletionChunk]):
    async def _aiter():
        for chunk in chunks:
            yield chunk

    streamed = KimiStreamedMessage(_aiter())
    return [part async for part in streamed]


async def test_kimi_stream_buffers_argument_deltas_until_name_arrives():
    """Some OpenAI-compatible servers emit argument chunks before the
    function name for a stream index. Those early chunks must be buffered
    and prepended to the header — never dropped."""
    chunks = [
        _make_stream_chunk(
            [{"index": 0, "id": "call_delayed", "function": {"name": "", "arguments": ""}}]
        ),
        _make_stream_chunk([{"index": 0, "function": {"arguments": '{"a'}}]),
        _make_stream_chunk([{"index": 0, "function": {"name": "foo"}}]),
        _make_stream_chunk([{"index": 0, "function": {"arguments": '":1}'}}]),
    ]
    parts = await _collect_stream_parts(chunks)
    assert [p.model_dump(exclude_none=True) for p in parts] == snapshot(
        [
            {
                "type": "function",
                "id": "call_delayed",
                "function": {"name": "foo", "arguments": '{"a'},
            },
            {"arguments_part": '":1}'},
        ]
    )


async def test_kimi_non_stream_yields_empty_think_part_for_empty_reasoning_content():
    """An explicitly empty ``reasoning_content`` must surface as an empty
    ThinkPart (not be dropped) so the next request passes the field back —
    Moonshot rejects thinking-mode histories whose assistant messages lost
    their reasoning field."""
    response_json = make_chat_completion_response()
    response_json["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"test"}'},
            }
        ],
    }
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(return_value=Response(200, json=response_json))
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="hi")])
        parts = [part async for part in stream]
        assert [p.model_dump(exclude_none=True) for p in parts] == snapshot(
            [
                {"type": "think", "think": ""},
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": '{"q":"test"}'},
                },
            ]
        )


async def test_kimi_stream_yields_empty_think_part_for_empty_reasoning_delta():
    async def _aiter():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-empty-reasoning",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "kimi-k2-turbo-preview",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": ""}, "finish_reason": None}
                ],
            }
        )

    streamed = KimiStreamedMessage(_aiter())
    parts = [part async for part in streamed]
    assert [p.model_dump(exclude_none=True) for p in parts] == snapshot(
        [{"type": "think", "think": ""}]
    )


async def test_kimi_stream_sequential_parallel_tool_calls():
    chunks = [
        _make_stream_chunk(
            [{"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": ""}}]
        ),
        _make_stream_chunk([{"index": 0, "function": {"arguments": '{"path":"a.txt"}'}}]),
        _make_stream_chunk(
            [{"index": 1, "id": "call_b", "function": {"name": "read_file", "arguments": ""}}]
        ),
        _make_stream_chunk([{"index": 1, "function": {"arguments": '{"path":"b.txt"}'}}]),
    ]
    parts = await _collect_stream_parts(chunks)
    assert [p.model_dump(exclude_none=True) for p in parts] == snapshot(
        [
            {
                "type": "function",
                "id": "call_a",
                "function": {"name": "read_file"},
            },
            {"arguments_part": '{"path":"a.txt"}'},
            {
                "type": "function",
                "id": "call_b",
                "function": {"name": "read_file"},
            },
            {"arguments_part": '{"path":"b.txt"}'},
        ]
    )


async def test_kimi_session_id_prompt_cache_key_in_body():
    """The session id must reach the wire as ``prompt_cache_key``.

    Doubles as the Phase-2 SDK-compat check: openai>=2.44 accepts
    ``prompt_cache_key`` as a typed top-level kwarg, so it must appear in the
    request body without any extra_body routing.
    """
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("kimi-k2"))
        )
        provider = Kimi(
            model="kimi-k2-turbo-preview", api_key="test-key", stream=False
        ).with_generation_kwargs(prompt_cache_key="sess-abc-123")
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["prompt_cache_key"] == "sess-abc-123"


async def test_kimi_without_prompt_cache_key_omits_field():
    """No session id → no ``prompt_cache_key`` in the request body."""
    with respx.mock(base_url="https://api.moonshot.ai") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("kimi-k2"))
        )
        provider = Kimi(model="kimi-k2-turbo-preview", api_key="test-key", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "prompt_cache_key" not in body
