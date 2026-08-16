"""Tests for the provider profile registry."""

import pytest

from kosong.chat_provider import ChatProviderError
from kosong.providers import (
    create_chat_provider,
    get_provider_profile,
    list_providers,
)


def test_registry_contains_all_hermes_providers():
    providers = list_providers()
    assert len(providers) == 35
    expected = {
        "ai-gateway",
        "alibaba",
        "alibaba-coding-plan",
        "arcee",
        "azure-foundry",
        "bedrock",
        "copilot",
        "custom",
        "deepinfra",
        "deepseek",
        "fireworks",
        "gmi",
        "huggingface",
        "kilocode",
        "kimi-coding",
        "minimax",
        "nous",
        "novita",
        "nvidia",
        "ollama-cloud",
        "openai-codex",
        "opencode-zen",
        "openrouter",
        "qwen-oauth",
        "stepfun",
        "upstage",
        "xiaomi",
        "zai",
        "anthropic",
        "gemini",
        "vertex",
        "xai",
        "actual",
        "copilot-acp",
        "kimi",
    }
    assert set(providers) == expected


def test_provider_profile_fields_ported_from_hermes():
    profile = get_provider_profile("deepseek")
    assert profile is not None
    assert profile.base_url == "https://api.deepseek.com/v1"
    assert profile.env_vars == ("DEEPSEEK_API_KEY",)
    assert "deepseek-chat" in profile.aliases
    assert profile.fallback_models == ("deepseek-v4-pro", "deepseek-v4-flash")

    zai = get_provider_profile("zai")
    assert zai is not None
    assert zai.base_url == "https://api.z.ai/api/paas/v4"
    assert zai.env_vars == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")

    xiaomi = get_provider_profile("xiaomi")
    assert xiaomi is not None
    assert xiaomi.supports_vision_tool_messages is False

    openrouter = get_provider_profile("openrouter")
    assert openrouter is not None
    assert openrouter.base_url == "https://openrouter.ai/api/v1"
    assert openrouter.env_vars == ("OPENROUTER_API_KEY",)

    minimax = get_provider_profile("minimax")
    assert minimax is not None
    assert minimax.base_url == "https://api.minimax.io/anthropic"
    assert minimax.env_vars == ("MINIMAX_API_KEY",)

    bedrock = get_provider_profile("bedrock")
    assert bedrock is not None
    assert bedrock.provider_class == "kosong.contrib.chat_provider.bedrock:Bedrock"

    openai_codex = get_provider_profile("openai-codex")
    assert openai_codex is not None
    assert openai_codex.base_url == "https://chatgpt.com/backend-api/codex"


def test_alias_resolution():
    assert get_provider_profile("glm") is get_provider_profile("zai")
    assert get_provider_profile("or") is get_provider_profile("openrouter")
    assert get_provider_profile("dashscope") is get_provider_profile("alibaba")
    assert get_provider_profile("unknown-provider") is None


def test_create_chat_provider_maps_to_correct_class():
    from kosong.chat_provider.codex import Actual, OpenAICodex
    from kosong.chat_provider.compat import ZAI, DeepSeek, OpenRouter, Xiaomi
    from kosong.chat_provider.kimi import Kimi
    from kosong.chat_provider.xai import XAI
    from kosong.contrib.chat_provider.anthropic import Anthropic
    from kosong.contrib.chat_provider.bedrock import Bedrock
    from kosong.contrib.chat_provider.google_genai import GoogleGenAI
    from kosong.contrib.chat_provider.minimax import MiniMaxAnthropic

    cases = {
        "deepseek": DeepSeek,
        "openrouter": OpenRouter,
        "zai": ZAI,
        "xiaomi": Xiaomi,
        "alibaba": type(create_chat_provider("alibaba", model="qwen3", api_key="k")),
        "minimax": MiniMaxAnthropic,
        "bedrock": Bedrock,
        "openai-codex": OpenAICodex,
        "actual": Actual,
        "anthropic": Anthropic,
        "gemini": GoogleGenAI,
        "kimi": Kimi,
        "xai": XAI,
        "alibaba-coding-plan": type(
            create_chat_provider("alibaba-coding-plan", model="qwen3", api_key="k")
        ),
    }
    for name, cls in cases.items():
        provider = create_chat_provider(name, model="test-model", api_key="test")
        assert isinstance(provider, cls), f"{name} -> {type(provider).__name__}"


def test_create_chat_provider_uses_profile_base_url():
    provider = create_chat_provider("deepseek", model="deepseek-v4-pro", api_key="test")
    assert str(provider.client.base_url).rstrip("/") == "https://api.deepseek.com/v1"

    provider = create_chat_provider("zai", model="glm-5", api_key="test")
    assert str(provider.client.base_url).rstrip("/") == "https://api.z.ai/api/paas/v4"


def test_create_chat_provider_api_key_env_fallback(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    provider = create_chat_provider("deepseek", model="deepseek-v4-pro")
    assert provider.client.api_key == "env-key"


def test_create_chat_provider_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    provider = create_chat_provider("deepseek", model="deepseek-v4-pro", api_key="explicit")
    assert provider.client.api_key == "explicit"


def test_create_chat_provider_unknown_raises():
    with pytest.raises(ChatProviderError, match="Unknown provider"):
        create_chat_provider("nope", model="m", api_key="k")


def test_create_chat_provider_copilot_acp_raises():
    # copilot-acp is an external ACP subprocess with no kosong implementation.
    with pytest.raises(ChatProviderError, match="no kosong chat provider"):
        create_chat_provider("copilot-acp", model="m", api_key="k")


def test_list_providers_is_sorted_by_registration():
    providers = list_providers()
    assert providers[0] == "ai-gateway"
    assert providers[-1] == "copilot-acp"
