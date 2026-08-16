"""Tests for the AWS Bedrock chat provider."""

import pytest
import respx
from common import make_anthropic_response
from httpx import Response

from kosong.chat_provider import ChatProviderError
from kosong.contrib.chat_provider.bedrock import Bedrock
from kosong.message import Message

MODEL = "anthropic.claude-sonnet-4-20250514"


def make_provider(**kwargs):
    return Bedrock(
        model=MODEL,
        aws_access_key="test-key",
        aws_secret_key="test-secret",
        aws_region="us-east-1",
        default_max_tokens=1024,
        stream=False,
        **kwargs,
    )


async def test_bedrock_identity():
    provider = make_provider()
    assert provider.name == "bedrock"
    assert provider.model_name == MODEL
    assert str(provider._client.base_url) == "https://bedrock-runtime.us-east-1.amazonaws.com"


async def test_bedrock_generate():
    """Generate works with the AWS SDK; without it a clear error is raised."""
    import importlib.util

    if importlib.util.find_spec("boto3") is None:
        with pytest.raises(ChatProviderError, match="boto3"):
            await make_provider().generate("", [], [Message(role="user", content="Hello!")])
        return
    with respx.mock(base_url="https://bedrock-runtime.us-east-1.amazonaws.com") as mock:
        mock.post(f"/model/{MODEL}/invoke").mock(
            return_value=Response(200, json=make_anthropic_response(MODEL))
        )
        provider = make_provider()
        stream = await provider.generate("", [], [Message(role="user", content="Hello!")])
        parts = [part async for part in stream]
        assert parts[0].text == "Hello"


async def test_bedrock_thinking():
    provider = make_provider().with_thinking("high")
    assert provider.thinking_effort == "high"
    thinking = provider._generation_kwargs.get("thinking")
    assert thinking is not None
    assert thinking["type"] == "enabled"

    off_provider = make_provider().with_thinking("off")
    assert off_provider.thinking_effort == "off"
    assert off_provider._generation_kwargs["thinking"]["type"] == "disabled"


def test_bedrock_profile_requires_boto3_for_named_profile(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(ChatProviderError, match="boto3"):
        Bedrock(
            model=MODEL,
            aws_profile="some-profile",
            aws_region="us-east-1",
        )
