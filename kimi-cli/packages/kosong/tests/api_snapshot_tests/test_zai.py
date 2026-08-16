"""Tests for the ZAI (GLM) chat provider (thinking wire format)."""

import respx
from common import capture_request, make_chat_completion_response
from httpx import Response

from kosong.chat_provider.compat import ZAI
from kosong.message import Message


def make_provider(model: str = "glm-5", **kwargs):
    return ZAI(model=model, api_key="test", stream=False, **kwargs)


async def test_zai_glm_5_thinking_enabled():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-5"))
        )
        provider = make_provider().with_thinking("high")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in body


async def test_zai_glm_5_2_reasoning_effort_high():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-5.2"))
        )
        provider = make_provider(model="glm-5.2").with_thinking("medium")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        # GLM-5.2 only supports high/max; medium clamps to high.
        assert body["reasoning_effort"] == "high"


async def test_zai_glm_5_2_reasoning_effort_max():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-5.2"))
        )
        provider = make_provider(model="glm-5.2").with_thinking("max")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "max"


async def test_zai_thinking_off():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-5.2"))
        )
        provider = make_provider(model="glm-5.2").with_thinking("off")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in body


async def test_zai_thinking_enabled_by_default_for_glm_45_plus():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-5"))
        )
        provider = make_provider()
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert body["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in body


async def test_zai_glm_4_9b_not_touched():
    with respx.mock(base_url="https://api.z.ai") as mock:
        mock.post("/api/paas/v4/chat/completions").mock(
            return_value=Response(200, json=make_chat_completion_response("glm-4-9b"))
        )
        provider = make_provider(model="glm-4-9b")
        body = await capture_request(
            mock, provider, "", [], [Message(role="user", content="Hello!")]
        )
        assert "thinking" not in body
        assert "reasoning_effort" not in body


async def test_zai_thinking_effort_property():
    assert make_provider().thinking_effort is None
    # GLM-5.2 carries a concrete top-level reasoning_effort; older GLM models
    # only toggle thinking on/off without an effort level.
    assert make_provider(model="glm-5.2").with_thinking("high").thinking_effort == "high"
    assert make_provider(model="glm-5.2").with_thinking("off").thinking_effort == "off"
    assert make_provider(model="glm-5").with_thinking("high").thinking_effort is None
