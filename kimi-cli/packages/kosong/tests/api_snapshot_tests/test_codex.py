"""Tests for the OpenAI Codex and Actual Computer chat providers."""

from typing import Any

import httpx
import orjson
import respx
from httpx import Response

from kosong.chat_provider.codex import Actual, OpenAICodex
from kosong.message import Message, TextPart


def make_response() -> dict[str, Any]:
    return {
        "id": "resp_test123",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": "codex-mini-latest",
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


async def test_openai_codex_identity():
    provider = OpenAICodex(model="codex-mini-latest", api_key="test")
    assert provider.name == "openai-codex"
    assert provider.model_name == "codex-mini-latest"
    assert str(provider.client.base_url).rstrip("/") == "https://chatgpt.com/backend-api/codex"


async def test_openai_codex_generate():
    with respx.mock(base_url="https://chatgpt.com/backend-api/codex") as mock:
        mock.post("/responses").mock(return_value=Response(200, json=make_response()))
        provider = OpenAICodex(model="codex-mini-latest", api_key="test", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hello!")])
        parts = [part async for part in stream]
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "Hello"
        body = orjson.loads(mock.calls.last.request.content)
        assert body["store"] is False


async def test_openai_codex_uses_official_request_contract():
    with respx.mock(base_url="https://chatgpt.com/backend-api/codex") as mock:
        mock.post("/responses").mock(return_value=Response(200, json=make_response()))
        provider = OpenAICodex(
            session_id="gui_session",
            model="gpt-5.6-terra",
            api_key="test",
            stream=False,
        ).with_generation_kwargs(
            user="must-not-reach-codex",
            max_output_tokens=128_000,
            temperature=0.7,
            reasoning_effort="medium",
        )
        try:
            stream = await provider.generate(
                "Follow the project instructions.",
                [],
                [Message(role="user", content="Hello")],
            )
            parts = [part async for part in stream]
            request = mock.calls.last.request
        finally:
            await provider.shutdown()

    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "Hello"
    assert orjson.loads(request.content) == {
        "include": ["reasoning.encrypted_content"],
        "input": [
            {
                "content": [{"text": "Hello", "type": "input_text"}],
                "role": "user",
                "type": "message",
            }
        ],
        "instructions": "Follow the project instructions.",
        "model": "gpt-5.6-terra",
        "parallel_tool_calls": True,
        "prompt_cache_key": "gui_session",
        "reasoning": {"effort": "medium", "summary": "auto"},
        "store": False,
        "stream": False,
        "tool_choice": "auto",
        "tools": [],
    }
    assert request.headers["session-id"] == "gui_session"
    assert request.headers["thread-id"] == "gui_session"
    assert request.headers["session_id"] == "gui_session"
    assert request.headers["x-client-request-id"] == "gui_session"


async def test_openai_codex_maps_sequential_mode_to_official_parallel_flag():
    with respx.mock(base_url="https://chatgpt.com/backend-api/codex") as mock:
        mock.post("/responses").mock(return_value=Response(200, json=make_response()))
        provider = OpenAICodex(model="codex-mini-latest", api_key="test", stream=False)
        provider = provider.with_parallel_tool_calls(enabled=False)
        try:
            stream = await provider.generate(
                "Instructions", [], [Message(role="user", content="Hi")]
            )
            async for _ in stream:
                pass
            body = orjson.loads(mock.calls.last.request.content)
        finally:
            await provider.shutdown()

    assert body["parallel_tool_calls"] is False
    assert "max_tool_calls" not in body


async def test_openai_codex_non_owning_copy_defers_http_client_close():
    http_client = httpx.AsyncClient()
    provider = OpenAICodex(
        model="codex-mini-latest",
        api_key="test",
        http_client=http_client,
        own_http_client=False,
    ).with_generation_kwargs(reasoning_effort="medium")

    await provider.aclose()
    assert http_client.is_closed is False

    await provider.shutdown()
    assert http_client.is_closed is True


async def test_openai_codex_owns_http_client_by_default():
    http_client = httpx.AsyncClient()
    provider = OpenAICodex(
        model="codex-mini-latest",
        api_key="test",
        http_client=http_client,
    )

    await provider.aclose()

    assert http_client.is_closed is True


async def test_actual_identity_defaults():
    provider = Actual(model="actual-model", api_key="test")
    assert provider.name == "actual"
    assert str(provider.client.base_url).rstrip("/") == "https://api.actual.inc/v1"


async def test_actual_uses_actual_base_url_env(monkeypatch):
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080/v1")
    provider = Actual(model="actual-model", api_key="test")
    assert str(provider.client.base_url).rstrip("/") == "http://127.0.0.1:8080/v1"


async def test_actual_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ACTUAL_API_KEY", "env-key")
    provider = Actual(model="actual-model")
    assert provider.client.api_key == "env-key"


async def test_actual_generate():
    with respx.mock(base_url="https://api.actual.inc") as mock:
        # Actual's default base URL already includes the /v1 path segment.
        mock.post("/v1/responses").mock(return_value=Response(200, json=make_response()))
        provider = Actual(model="actual-model", api_key="test", stream=False)
        stream = await provider.generate("", [], [Message(role="user", content="Hello!")])
        parts = [part async for part in stream]
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "Hello"
