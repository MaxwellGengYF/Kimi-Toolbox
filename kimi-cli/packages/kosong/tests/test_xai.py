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

