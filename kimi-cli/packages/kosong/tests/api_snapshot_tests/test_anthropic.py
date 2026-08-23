"""Snapshot tests for Anthropic chat provider."""

import asyncio
import json

import pytest
import respx
from common import (
    B64_PNG,
    COMMON_CASES,
    Case,
    make_anthropic_response,
    make_httpx2_client,
    run_test_cases,
)
from httpx import Response
from inline_snapshot import snapshot

pytest.importorskip("anthropic", reason="Optional contrib dependency not installed")

from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.message import ImageURLPart, Message, TextPart, ThinkPart, ToolCall

TEST_CASES: dict[str, Case] = {
    **COMMON_CASES,
    "assistant_with_thinking": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(think="Let me think...", encrypted="sig_abc123"),
                    TextPart(text="The answer is 4."),
                ],
            ),
            Message(role="user", content="Thanks!"),
        ],
    },
    "thinking_without_signature_stripped": {
        "history": [
            Message(role="user", content="Hi"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Thinking..."), TextPart(text="Hello!")],
            ),
            Message(role="user", content="Bye"),
        ],
    },
    "base64_image": {
        "history": [
            Message(
                role="user",
                content=[
                    TextPart(text="Describe:"),
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(url=f"data:image/png;base64,{B64_PNG}")
                    ),
                ],
            )
        ],
    },
    "redacted_thinking": {
        "history": [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[
                    ThinkPart(think="", encrypted="enc_redacted_sig_xyz"),
                    TextPart(text="4."),
                ],
            ),
            Message(role="user", content="Thanks!"),
        ],
    },
}


async def test_anthropic_message_conversion():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        results = await run_test_cases(mock, provider, TEST_CASES, ("messages", "system", "tools"))

        assert results == snapshot(
            {
                "simple_user_message": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Hello!",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                    "system": [
                        {
                            "type": "text",
                            "text": "You are helpful.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "tools": [],
                },
                "multi_turn_conversation": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "2+2 equals 4."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "And 3+3?",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
                "multi_turn_with_system": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "2+2 equals 4."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "And 3+3?",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "system": [
                        {
                            "text": "You are a math tutor.",
                            "type": "text",
                            "cache_control": {"type": "ephemeral"},
                        }
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
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.com/image.png",
                                    },
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        }
                    ],
                    "tools": [],
                },
                "tool_definition": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Add 2 and 3",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two integers.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                        {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                "tool_call_with_image": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I'll add those numbers for you."},
                                {
                                    "type": "tool_use",
                                    "id": "call_abc123",
                                    "name": "add",
                                    "input": {"a": 2, "b": 3},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call_abc123",
                                    "content": [
                                        {"type": "text", "text": "5"},
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "url",
                                                "url": "https://example.com/image.png",
                                            },
                                        },
                                    ],
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
                "tool_call": {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Add 2 and 3"}]},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I'll add those numbers for you."},
                                {
                                    "type": "tool_use",
                                    "id": "call_abc123",
                                    "name": "add",
                                    "input": {"a": 2, "b": 3},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call_abc123",
                                    "content": [{"type": "text", "text": "5"}],
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
                "parallel_tool_calls": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Calculate 2+3 and 4*5"}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I'll calculate both."},
                                {
                                    "type": "tool_use",
                                    "id": "call_add",
                                    "name": "add",
                                    "input": {"a": 2, "b": 3},
                                },
                                {
                                    "type": "tool_use",
                                    "id": "call_mul",
                                    "name": "multiply",
                                    "input": {"a": 4, "b": 5},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call_add",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "<system-reminder>This is a system reminder"
                                            "</system-reminder>",
                                        },
                                        {"type": "text", "text": "5"},
                                    ],
                                },
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call_mul",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "<system-reminder>This is a system reminder"
                                            "</system-reminder>",
                                        },
                                        {"type": "text", "text": "20"},
                                    ],
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        },
                    ],
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two integers.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                        {
                            "name": "multiply",
                            "description": "Multiply two integers.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "integer", "description": "First number"},
                                    "b": {"type": "integer", "description": "Second number"},
                                },
                                "required": ["a", "b"],
                            },
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                "assistant_with_thinking": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "What is 2+2?"}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "Let me think...",
                                    "signature": "sig_abc123",
                                },
                                {"type": "text", "text": "The answer is 4."},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Thanks!",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
                "thinking_without_signature_stripped": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Hi"}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello!"}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Bye",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
                "base64_image": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe:"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "data": B64_PNG,
                                        "media_type": "image/png",
                                    },
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        }
                    ],
                    "tools": [],
                },
                "redacted_thinking": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "What is 2+2?"}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "enc_redacted_sig_xyz",
                                },
                                {"type": "text", "text": "4."},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Thanks!",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
            }
        )


async def test_anthropic_generation_kwargs():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_generation_kwargs(temperature=0.7, top_p=0.9, max_tokens=2048)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert (body["temperature"], body["top_p"], body["max_tokens"]) == snapshot(
            (0.7, 0.9, 2048)
        )


async def test_anthropic_generation_kwargs_none_extras():
    """None values for optional list/mapping extras must be treated as empty."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_generation_kwargs(beta_features=None, extra_headers=None)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["max_tokens"] == 1024


async def test_anthropic_max_tokens_clamped():
    """Over-large max_tokens values are clamped to the Anthropic upper bound."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_generation_kwargs(max_tokens=10_000_000)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        # Non-streaming requests are clamped to the SDK's 10-minute threshold.
        assert body["max_tokens"] == 21_333


async def test_anthropic_max_tokens_none_raises():
    """max_tokens is required; setting it to None must raise a clear error."""
    provider = Anthropic(
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        default_max_tokens=1024,
        stream=False,
    ).with_generation_kwargs(max_tokens=None)  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="max_tokens must be provided"):
        await provider.generate("", [], [Message(role="user", content="Hi")])


async def test_anthropic_with_thinking():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 32000})


async def test_anthropic_opus_46_adaptive_thinking():
    """Opus 4.6 adaptive thinking should opt-in to summarized display and pass effort."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-6-20260205",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "adaptive", "display": "summarized"})
        assert body["output_config"] == snapshot({"effort": "high"})
        # Adaptive thinking should not include interleaved-thinking beta header
        beta_header = mock.calls.last.request.headers.get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" not in beta_header


async def test_anthropic_opus_46_thinking_off():
    """Opus 4.6 with thinking off should send disabled and omit output_config."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-6-20260205",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("off")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "disabled"})
        assert "output_config" not in body


async def test_anthropic_opus_47_adaptive_thinking():
    """Opus 4.7 must use adaptive thinking (legacy rejected by API)."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "adaptive", "display": "summarized"})
        assert body["output_config"] == snapshot({"effort": "high"})


async def test_anthropic_opus_47_effort_low_passthrough():
    """Opus 4.7 with 'low' effort must not silently become 'high'."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("low")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "low"})


async def test_anthropic_opus_47_effort_medium_passthrough():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("medium")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "medium"})


async def test_anthropic_future_opus_48_uses_adaptive():
    """A future Opus 4.8 model should be recognized as adaptive-capable via regex extrapolation."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-8",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "adaptive", "display": "summarized"})


async def test_anthropic_sonnet_4_legacy_thinking_preserved():
    """Sonnet 4 (pre-4.6) uses budget_tokens and does NOT send output_config.

    Per Anthropic docs, only Mythos/Opus 4.7/4.6/Sonnet 4.6/Opus 4.5 are
    explicitly listed as supporting the effort parameter. Sonnet 4 is not
    on that list, so emitting `output_config.effort` would risk a 400.
    """
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("low")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 1024})
        assert "output_config" not in body


async def test_anthropic_claude_3_legacy_no_output_config():
    """Claude 3.x predates the effort parameter — output_config must be absent
    to avoid 400 validation errors on those models.
    """
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-3-5-sonnet-20240620",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("high")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 32000})
        assert "output_config" not in body


async def test_anthropic_haiku_45_legacy_no_output_config():
    """Haiku 4.5 is not in the explicit effort-supporting list — be conservative."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("medium")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 4096})
        assert "output_config" not in body


async def test_anthropic_opus_45_legacy_xhigh_clamps_to_high():
    """Opus 4.5 supports effort but not xhigh; explicit supported_efforts clamps
    xhigh down to high while still emitting output_config.
    """
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-5",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high"},
        ).with_thinking("xhigh")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "enabled", "budget_tokens": 32000})
        assert body["output_config"] == snapshot({"effort": "high"})


async def test_anthropic_opus_47_xhigh():
    """Opus 4.7 + xhigh should pass xhigh through when explicitly enabled."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high", "xhigh", "max"},
        ).with_thinking("xhigh")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "adaptive", "display": "summarized"})
        assert body["output_config"] == snapshot({"effort": "xhigh"})


async def test_anthropic_opus_47_max():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "max"})


async def test_anthropic_opus_46_max():
    """Opus 4.6 supports max effort."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-6",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "max"})


async def test_anthropic_opus_46_xhigh_clamps_to_high():
    """Opus 4.6 doesn't support xhigh (Opus 4.7-only) — supported_efforts clamps
    xhigh down to high.
    """
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-6",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high"},
        ).with_thinking("xhigh")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "high"})


async def test_anthropic_switching_from_adaptive_to_off_clears_output_config():
    """Switching from adaptive effort to off must not leave a stale output_config."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = (
            Anthropic(
                http_client=make_httpx2_client(mock),
                model="claude-opus-4-7",
                api_key="test-key",
                default_max_tokens=1024,
                stream=False,
            )
            .with_thinking("high")
            .with_thinking("off")
        )
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["thinking"] == snapshot({"type": "disabled"})
        assert "output_config" not in body


async def test_anthropic_metadata():
    """Metadata should be forwarded to the Anthropic API request."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            metadata={"user_id": "test-session-id"},
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["metadata"] == snapshot({"user_id": "test-session-id"})


async def test_anthropic_metadata_omitted_when_none():
    """Metadata should not be included in the request when not provided."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "metadata" not in body


async def test_anthropic_opus_46_thinking_effort_property():
    """thinking_effort roundtrips through output_config.effort in adaptive mode."""
    for effort in ("low", "medium", "high", "off"):
        provider = Anthropic(
            model="claude-opus-4-6-20260205",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_thinking(effort)  # type: ignore[arg-type]
        assert provider.thinking_effort == effort


async def test_anthropic_opus_47_thinking_effort_property():
    """Opus 4.7 roundtrips all effort levels through output_config."""
    for effort in ("low", "medium", "high", "xhigh", "max", "off"):
        provider = Anthropic(
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high", "xhigh", "max"},
        ).with_thinking(effort)  # type: ignore[arg-type]
        assert provider.thinking_effort == effort


async def test_anthropic_opus_46_xhigh_property_reflects_clamped_value():
    """After clamping, the getter reports the effective (clamped) effort."""
    provider = Anthropic(
        model="claude-opus-4-6",
        api_key="test-key",
        default_max_tokens=1024,
        stream=False,
        supported_efforts={"low", "medium", "high"},
    ).with_thinking("xhigh")
    # xhigh clamped to high when not in supported_efforts
    assert provider.thinking_effort == "high"


async def test_anthropic_supported_efforts_clamps_max_to_high():
    """A model configured without ``max`` must clamp it to the highest supported
    effort level.
    """
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high"},
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "high"})


async def test_anthropic_supported_efforts_passes_max():
    """A model configured with the full effort set must pass ``max`` through."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-opus-4-7",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            supported_efforts={"low", "medium", "high", "xhigh", "max"},
        ).with_thinking("max")
        stream = await provider.generate("", [], [Message(role="user", content="Think")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["output_config"] == snapshot({"effort": "max"})


async def test_anthropic_parallel_tool_results_merged_into_single_user_message():
    """Parallel tool results must be packed into a single user message.

    Per the Anthropic Messages API spec, every tool_use block in an assistant
    message must be answered by tool_result blocks inside the same (single)
    user message. Anthropic's official backend leniently merges consecutive
    same-role turns, but strict Anthropic-compatible backends (e.g. DeepSeek's
    /anthropic endpoint) reject the split form with 400 — and the docs warn
    that the split form also silently teaches Claude to avoid parallel calls.
    """
    from common import ADD_TOOL, MUL_TOOL, capture_request

    from kosong.message import ToolCall

    history = [
        Message(role="user", content="Calculate 2+3 and 4*5"),
        Message(
            role="assistant",
            content="I'll calculate both.",
            tool_calls=[
                ToolCall(
                    id="call_add",
                    function=ToolCall.FunctionBody(name="add", arguments='{"a": 2, "b": 3}'),
                ),
                ToolCall(
                    id="call_mul",
                    function=ToolCall.FunctionBody(name="multiply", arguments='{"a": 4, "b": 5}'),
                ),
            ],
        ),
        Message(role="tool", content="5", tool_call_id="call_add"),
        Message(role="tool", content="20", tool_call_id="call_mul"),
    ]

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        body = await capture_request(mock, provider, "", [ADD_TOOL, MUL_TOOL], history)

    messages = body["messages"]
    # Expected wire layout:
    #   [0] user      — original question
    #   [1] assistant — text + two tool_use blocks
    #   [2] user      — *both* tool_result blocks packed together
    assert [m["role"] for m in messages] == ["user", "assistant", "user"], (
        f"Parallel tool results should collapse into one trailing user message, "
        f"got roles: {[m['role'] for m in messages]}"
    )
    tool_results = [b for b in messages[-1]["content"] if b["type"] == "tool_result"]
    assert {b["tool_use_id"] for b in tool_results} == {"call_add", "call_mul"}


# -----------------------------------------------------------------------------
# Cross-provider tool-call id normalization
# -----------------------------------------------------------------------------


async def test_anthropic_normalizes_invalid_tool_call_ids():
    """Histories persisted from other providers can carry tool-call ids that
    Anthropic's id validation rejects (e.g. ``Read:9`` — only
    ``[a-zA-Z0-9_-]`` is accepted). They are sanitized in both the assistant
    tool_use block and the matching tool_result block, keeping them
    consistent."""
    from common import capture_request

    history = [
        Message(role="user", content="Read a file"),
        Message(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCall(
                    id="Read:9",
                    function=ToolCall.FunctionBody(name="Read", arguments='{"path":"/tmp/file"}'),
                )
            ],
        ),
        Message(role="tool", content="content", tool_call_id="Read:9"),
    ]

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        body = await capture_request(mock, provider, "", [], history)

    messages = body["messages"]
    tool_use = messages[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "Read_9"
    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "Read_9"
    # The caller's history objects must not be mutated.
    assert history[1].tool_calls is not None
    assert history[1].tool_calls[0].id == "Read:9"
    assert history[2].tool_call_id == "Read:9"


async def test_anthropic_valid_tool_call_ids_pass_through_unchanged():
    """Well-formed Anthropic ids (``toolu_...``) are sent verbatim."""
    from common import capture_request

    history = [
        Message(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCall(
                    id="toolu_abc-123_XYZ",
                    function=ToolCall.FunctionBody(name="f", arguments="{}"),
                )
            ],
        ),
        Message(role="tool", content="1", tool_call_id="toolu_abc-123_XYZ"),
    ]

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        body = await capture_request(mock, provider, "", [], history)

    assert body["messages"][0]["content"][0]["id"] == "toolu_abc-123_XYZ"
    assert body["messages"][1]["content"][0]["tool_use_id"] == "toolu_abc-123_XYZ"


# -----------------------------------------------------------------------------
# Malformed tool-call arguments (defensive parsing)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "expected_error_substring"),
    [
        ('{"a": 1, "b": 2', ""),  # broken JSON — loads_relaxed repairs it
        ("<args><a>1</a></args>", "must be a JSON object, got str."),  # XML
        ("a: 1\nb: 2", "must be a JSON object, got str."),  # YAML
        ("{{a=1, b=2}}", "must be a JSON object, got list."),  # DSML-like
        ("not json at all", "must be a JSON object, got str."),  # garbage
        ("[1, 2, 3]", "must be a JSON object"),  # valid JSON, but array
    ],
    ids=["broken_json", "xml", "yaml", "dsml", "garbage", "json_array"],
)
async def test_anthropic_malformed_tool_call_arguments_in_request(
    arguments: str, expected_error_substring: str
):
    """Malformed tool-call arguments in history must be surfaced to the LLM, not crash."""
    from common import capture_request

    history = [
        Message(
            role="assistant",
            content=[TextPart(text="I'll call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_bad",
                    function=ToolCall.FunctionBody(name="add", arguments=arguments),
                )
            ],
        )
    ]

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        body = await capture_request(mock, provider, "", [], history)

    messages = body["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "I'll call a tool."
    if not expected_error_substring:
        # loads_relaxed successfully repaired the JSON
        assert content[1]["type"] == "tool_use"
        assert content[1]["input"] == {"a": 1, "b": 2}
        assert content[1]["name"] == "add"
        assert content[1]["id"] == "call_bad"
    else:
        assert content[1]["type"] == "text"
        assert expected_error_substring in content[1]["text"]
        assert content[2]["type"] == "tool_use"
        assert content[2]["input"] == {}
        assert content[2]["name"] == "add"
        assert content[2]["id"] == "call_bad"


async def test_anthropic_empty_tool_call_arguments_in_request():
    """Empty string arguments should produce an empty tool_use input, not an error."""
    from common import capture_request

    history = [
        Message(
            role="assistant",
            content=[TextPart(text="I'll call a tool.")],
            tool_calls=[
                ToolCall(
                    id="call_empty",
                    function=ToolCall.FunctionBody(name="add", arguments=""),
                )
            ],
        )
    ]

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        body = await capture_request(mock, provider, "", [], history)

    messages = body["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "I'll call a tool."
    assert content[1]["type"] == "tool_use"
    assert content[1]["input"] == {}
    assert content[1]["name"] == "add"
    assert content[1]["id"] == "call_empty"


@pytest.mark.parametrize(
    ("input_value", "expected_type", "expected_content"),
    [
        ({"a": 1, "b": 2}, "tool_call", '{"a":1,"b":2}'),
        ('{"a": 1, "b": 2}', "tool_call", '{"a": 1, "b": 2}'),
        ("<args><a>1</a></args>", "text_error", "non-object input"),
        ("a: 1\nb: 2", "text_error", "non-object input"),
        ("{{a=1, b=2}}", "text_error", "non-object input"),
        ("not json at all", "text_error", "non-object input"),
        ("[1, 2, 3]", "text_error", "non-object input"),
    ],
    ids=["dict", "json_string", "xml", "yaml", "dsml", "garbage", "array_string"],
)
async def test_anthropic_malformed_tool_use_in_response(
    input_value: object, expected_type: str, expected_content: str
):
    """Backend returning tool_use with unexpected input types must not crash."""

    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "add",
                            "input": input_value,
                        }
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        )
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        parts = [p async for p in stream]

    assert len(parts) == 1
    if expected_type == "tool_call":
        from kosong.message import ToolCall as ToolCallClass

        assert isinstance(parts[0], ToolCallClass)
        assert parts[0].function.arguments == expected_content
    else:
        from kosong.message import TextPart as TextPartClass

        assert isinstance(parts[0], TextPartClass)
        assert expected_content in parts[0].text


async def test_anthropic_with_parallel_tool_calls_disabled():
    """with_parallel_tool_calls(False) should emit disable_parallel_tool_use."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ).with_parallel_tool_calls(False)
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["tool_choice"] == snapshot(
            {"type": "auto", "disable_parallel_tool_use": True}
        )


async def test_anthropic_session_id_metadata_user_id_in_body():
    """The session id must reach the wire as ``metadata.user_id``."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(
            return_value=Response(200, json=make_anthropic_response())
        )
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
            metadata={"user_id": "sess-abc-123"},
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["metadata"] == {"user_id": "sess-abc-123"}


async def test_anthropic_without_metadata_omits_field():
    """No session id → no ``metadata`` in the request body."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(
            return_value=Response(200, json=make_anthropic_response())
        )
        provider = Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "metadata" not in body


async def test_anthropic_aclose_is_idempotent():
    provider = Anthropic(
        model="claude-sonnet-4-20250514",
        api_key="test-key",
        default_max_tokens=1024,
        stream=False,
    )
    await provider.aclose()
    await provider.aclose()


async def test_anthropic_async_context_manager_closes_client():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        async with Anthropic(
            http_client=make_httpx2_client(mock),
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        ) as provider:
            stream = await provider.generate("", [], [Message(role="user", content="Hi")])
            async for _ in stream:
                pass
        assert provider._client.is_closed()


async def test_anthropic_with_parallel_tool_calls_enabled():
    """with_parallel_tool_calls(True) should omit disable_parallel_tool_use."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = (
            Anthropic(
                http_client=make_httpx2_client(mock),
                model="claude-sonnet-4-20250514",
                api_key="test-key",
                default_max_tokens=1024,
                stream=False,
            )
            .with_parallel_tool_calls(False)
            .with_parallel_tool_calls(True)
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert "tool_choice" not in body


def test_anthropic_across_event_loops_is_silent():
    """Closing a provider from a different event loop must not raise."""

    async def _make():
        return Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )

    provider = asyncio.run(_make())

    async def _aclose():
        await provider.aclose()

    asyncio.run(_aclose())


def test_anthropic_sync_close_across_event_loops_is_silent():
    """The sync close() helper must not emit unawaited-coroutine warnings."""

    async def _make():
        return Anthropic(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            default_max_tokens=1024,
            stream=False,
        )

    provider = asyncio.run(_make())
    provider.close()


async def test_anthropic_parallel_tool_calls_last_call_wins():
    """with_parallel_tool_calls overrides a prior user-supplied tool_choice."""
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = (
            Anthropic(
                http_client=make_httpx2_client(mock),
                model="claude-sonnet-4-20250514",
                api_key="test-key",
                default_max_tokens=1024,
                stream=False,
            )
            .with_generation_kwargs(tool_choice={"type": "any"})
            .with_parallel_tool_calls(False)
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hi")])
        async for _ in stream:
            pass
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["tool_choice"] == snapshot(
            {"type": "auto", "disable_parallel_tool_use": True}
        )
