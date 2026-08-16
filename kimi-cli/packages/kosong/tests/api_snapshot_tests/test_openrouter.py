"""Tests for the OpenRouter chat provider (reasoning + sticky routing)."""

import respx
from common import capture_request, make_chat_completion_response
from httpx import Response

from kosong.chat_provider.compat import OpenRouter
from kosong.message import Message


def make_provider(model: str = "anthropic/claude-sonnet-4.6", **kwargs):
    return OpenRouter(model=model, api_key="test", stream=False, **kwargs)


async def test_openrouter_thinking_high():
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("test-model"))
        )
        provider = make_provider().with_thinking("high")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["reasoning"] == {"enabled": True, "effort": "high"}


async def test_openrouter_thinking_off():
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("test-model"))
        )
        provider = make_provider().with_thinking("off")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["reasoning"] == {"enabled": False}


async def test_openrouter_session_id_sticky_routing():
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("test-model"))
        )
        provider = make_provider().with_generation_kwargs(session_id="session-123")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["session_id"] == "session-123"


async def test_openrouter_grok_conv_id_header():
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("x-ai/grok-4"))
        )
        provider = make_provider(model="x-ai/grok-4").with_generation_kwargs(
            session_id="session-abc"
        )
        await capture_request(mock, provider, "", [], [Message(role="user", content="Hello!")])
        headers = mock.calls.last.request.headers
        assert headers["x-grok-conv-id"] == "session-abc"


async def test_openrouter_no_grok_header_for_non_grok():
    with respx.mock(base_url="https://openrouter.ai") as mock:
        mock.post("/api/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("test-model"))
        )
        provider = make_provider(model="anthropic/claude-sonnet-4.6").with_generation_kwargs(
            session_id="session-abc"
        )
        await capture_request(mock, provider, "", [], [Message(role="user", content="Hello!")])
        headers = mock.calls.last.request.headers
        assert "x-grok-conv-id" not in headers


async def test_openrouter_thinking_effort_property():
    assert make_provider().thinking_effort is None
    assert make_provider().with_thinking("max").thinking_effort == "max"
    assert make_provider().with_thinking("off").thinking_effort == "off"
