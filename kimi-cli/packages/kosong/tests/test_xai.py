"""Tests for the xAI chat provider."""

from __future__ import annotations

import pytest

from kosong.chat_provider import ChatProvider, RetryableChatProvider
from kosong.chat_provider.xai import XAI
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses


def test_xai_name_and_model_name():
    provider = XAI(model="grok-3", api_key="xai-test-key")
    assert provider.name == "xai"
    assert provider.model_name == "grok-3"


def test_xai_default_base_url():
    provider = XAI(model="grok-3", api_key="xai-test-key")
    assert str(provider.client.base_url) == "https://api.x.ai/v1/"


def test_xai_custom_base_url():
    provider = XAI(
        model="grok-3",
        api_key="xai-test-key",
        base_url="https://xai-proxy.test/v1",
    )
    assert str(provider.client.base_url) == "https://xai-proxy.test/v1/"


def test_xai_is_openai_responses_subclass():
    provider = XAI(model="grok-3", api_key="xai-test-key")
    assert isinstance(provider, OpenAIResponses)


def test_xai_implements_chat_provider_protocol():
    provider = XAI(model="grok-3", api_key="xai-test-key")
    assert isinstance(provider, ChatProvider)
    assert isinstance(provider, RetryableChatProvider)


def test_xai_api_key_passed_to_client():
    provider = XAI(model="grok-3", api_key="xai-secret-key")
    assert provider.client.api_key == "xai-secret-key"


def test_xai_token_auth_adds_xai_token_auth_header():
    provider = XAI(model="grok-3", api_key="xai-oauth-token", token_auth=True)
    assert provider.token_auth is True
    assert provider.client.default_headers.get("X-XAI-Token-Auth") == "xai-grok-cli"


def test_xai_token_auth_false_does_not_add_header():
    provider = XAI(model="grok-3", api_key="xai-api-key", token_auth=False)
    assert provider.token_auth is False
    assert "X-XAI-Token-Auth" not in provider.client.default_headers


def test_xai_token_auth_preserves_existing_default_headers():
    provider = XAI(
        model="grok-3",
        api_key="xai-oauth-token",
        token_auth=True,
        default_headers={"X-Custom": "value"},
    )
    assert provider.client.default_headers.get("X-XAI-Token-Auth") == "xai-grok-cli"
    assert provider.client.default_headers.get("X-Custom") == "value"


@pytest.mark.asyncio
async def test_xai_token_auth_setter_toggles_header():
    provider = XAI(model="grok-3", api_key="xai-token")
    assert provider.token_auth is False
    assert "X-XAI-Token-Auth" not in provider.client.default_headers

    provider.token_auth = True
    assert provider.token_auth is True
    assert provider.client.default_headers.get("X-XAI-Token-Auth") == "xai-grok-cli"

    provider.token_auth = False
    assert provider.token_auth is False
    assert "X-XAI-Token-Auth" not in provider.client.default_headers


@pytest.mark.asyncio
async def test_xai_token_auth_setter_preserves_api_key():
    provider = XAI(model="grok-3", api_key="xai-token")
    provider.token_auth = True
    assert provider.client.api_key == "xai-token"
