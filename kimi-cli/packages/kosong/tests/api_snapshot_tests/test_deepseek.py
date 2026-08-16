"""Tests for the DeepSeek chat provider (thinking wire format)."""

import respx
from common import capture_request, make_chat_completion_response
from httpx import Response

from kosong.chat_provider.compat import DeepSeek
from kosong.message import Message, TextPart, ThinkPart


def make_provider(model: str = "deepseek-v4-pro", **kwargs):
    return DeepSeek(model=model, api_key="test", stream=False, **kwargs)


async def test_deepseek_thinking_high():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v4-pro"))
        )
        provider = make_provider().with_thinking("high")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "high"


async def test_deepseek_thinking_max_maps_to_max():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v4-pro"))
        )
        provider = make_provider().with_thinking("xhigh")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "max"


async def test_deepseek_thinking_off():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v4-pro"))
        )
        provider = make_provider().with_thinking("off")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in body


async def test_deepseek_thinking_enabled_by_default_for_v4():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v4-pro"))
        )
        provider = make_provider()
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in body


async def test_deepseek_v3_not_touched():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v3-0324"))
        )
        provider = make_provider(model="deepseek-v3-0324")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert "thinking" not in body
        assert "reasoning_effort" not in body


async def test_deepseek_reasoning_content_round_trip():
    with respx.mock(base_url="https://api.deepseek.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("deepseek-v4-pro"))
        )
        provider = make_provider()
        history = [
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                content=[ThinkPart(think="Thinking..."), TextPart(text="4.")],
            ),
            Message(role="user", content="Thanks!"),
        ]
        body = await capture_request(mock, provider, "", [], history)
        assert body["messages"][1]["reasoning_content"] == "Thinking..."
        assert body["messages"][1]["content"] == "4."


async def test_deepseek_thinking_effort_property():
    assert make_provider().thinking_effort is None
    assert make_provider().with_thinking("high").thinking_effort == "high"
    assert make_provider().with_thinking("off").thinking_effort == "off"
