"""Tests for the MiniMax chat providers (Anthropic + OpenAI routes)."""

import respx
from common import (
    capture_request,
    make_anthropic_response,
    make_chat_completion_response,
    make_httpx2_client,
)
from httpx import Response

from kosong.contrib.chat_provider.minimax import MiniMaxAnthropic, MiniMaxOpenAI
from kosong.message import Message


async def test_minimax_anthropic_identity():
    provider = MiniMaxAnthropic(model="MiniMax-M2.7", api_key="test")
    assert provider.name == "minimax"
    assert provider.model_name == "MiniMax-M2.7"
    assert str(provider._client.base_url).rstrip("/") == "https://api.minimax.io/anthropic"


async def test_minimax_anthropic_generate():
    with respx.mock(base_url="https://api.minimax.io/anthropic") as mock:
        mock.post("/v1/messages").mock(return_value=Response(200, json=make_anthropic_response()))
        provider = MiniMaxAnthropic(
            model="MiniMax-M2.7", api_key="test", stream=False, http_client=make_httpx2_client(mock)
        )
        stream = await provider.generate("", [], [Message(role="user", content="Hello!")])
        parts = [part async for part in stream]
        assert parts[0].text == "Hello"


async def test_minimax_openai_m3_thinking_adaptive():
    with respx.mock(base_url="https://api.minimax.io") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("MiniMax-M3"))
        )
        provider = MiniMaxOpenAI(model="MiniMax-M3", api_key="test", stream=False).with_thinking(
            "medium"
        )
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["reasoning_split"] is True
        assert body["thinking"] == {"type": "adaptive"}


async def test_minimax_openai_m3_thinking_off():
    with respx.mock(base_url="https://api.minimax.io") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("MiniMax-M3"))
        )
        provider = MiniMaxOpenAI(model="MiniMax-M3", api_key="test", stream=False).with_thinking(
            "off"
        )
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["reasoning_split"] is True
        assert body["thinking"] == {"type": "disabled"}


async def test_minimax_openai_non_m3_uses_reasoning_effort():
    with respx.mock(base_url="https://api.minimax.io") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("MiniMax-M2.7"))
        )
        provider = MiniMaxOpenAI(model="MiniMax-M2.7", api_key="test", stream=False).with_thinking(
            "high"
        )
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert "reasoning_split" not in body
        assert body["reasoning_effort"] == "high"


async def test_minimax_openai_thinking_effort_property():
    provider = MiniMaxOpenAI(model="MiniMax-M3", api_key="test", stream=False)
    assert provider.with_thinking("medium").thinking_effort == "medium"
    assert provider.with_thinking("off").thinking_effort == "off"
