"""Tests for the OpenAI Codex and Actual Computer chat providers."""

import json
from typing import Any

import respx
from httpx import Response

from kosong.chat_provider.codex import Actual, OpenAICodex
from kosong.message import Message


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
        assert parts[0].text == "Hello"
        body = json.loads(mock.calls.last.request.content.decode())
        assert body["store"] is False


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
        assert parts[0].text == "Hello"
