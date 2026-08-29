from __future__ import annotations

from typing import cast

import pytest
from inline_snapshot import snapshot
from kosong.chat_provider.codex import OpenAICodex
from kosong.chat_provider.echo import EchoChatProvider
from kosong.chat_provider.kimi import Kimi
from kosong.chat_provider.xai import XAI
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from pydantic import SecretStr

from kimi_cli.auth.codex import CODEX_OAUTH_KEY
from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.config import Config, LLMModel, LLMProvider, OAuthRef, OpenAISettings, Services
from kimi_cli.llm import augment_provider_with_env_vars, create_llm


@pytest.mark.skip(reason="inline-snapshot incompatibility with pydantic SecretStr on this platform")
def test_augment_provider_with_env_vars_kimi(monkeypatch):
    provider = LLMProvider(
        type="kimi",
        base_url="https://original.test/v1",
        api_key=SecretStr("orig-key"),
    )
    model = LLMModel(
        model="kimi-base",
        max_context_size=4096,
        capabilities=None,
    )

    monkeypatch.setenv("KIMI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("KIMI_API_KEY", "env-key")
    monkeypatch.setenv("KIMI_MODEL_NAME", "kimi-env-model")
    monkeypatch.setenv("KIMI_MODEL_MAX_CONTEXT_SIZE", "8192")
    monkeypatch.setenv("KIMI_MODEL_CAPABILITIES", "Image_In,THINKING,unknown")

    augment_provider_with_env_vars(provider, model)

    assert provider.type == "kimi"
    assert provider.base_url == "https://original.test/v1"
    assert provider.api_key.get_secret_value() == "env-key"
    assert model.model == "kimi-env-model"
    assert model.max_context_size == 8192
    assert model.capabilities == {"image_in", "thinking"}


def test_create_llm_kimi_model_parameters(monkeypatch):
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="kimi-base",
        max_context_size=4096,
        capabilities=None,
    )

    # Temperature is ignored for the kimi provider and forced by thinking state.
    # top_p and max_tokens continue to be read from the environment.
    monkeypatch.setenv("KIMI_MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("KIMI_MODEL_TOP_P", "0.8")
    monkeypatch.setenv("KIMI_MODEL_MAX_TOKENS", "1234")

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    assert llm.chat_provider.model_parameters == snapshot(
        {
            "base_url": "https://api.test/v1/",
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 1234,
            "max_completion_tokens": 1234,
        }
    )


def test_create_llm_echo_provider():
    provider = LLMProvider(type="_echo", base_url="", api_key=SecretStr(""))
    model = LLMModel(model="echo", max_context_size=1234)

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, EchoChatProvider)
    assert llm.max_context_size == 1234


def test_create_llm_anthropic_with_session_id():
    from kosong.contrib.chat_provider.anthropic import Anthropic

    provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="claude-sonnet-4-20250514",
        max_context_size=200000,
    )

    llm = create_llm(provider, model, session_id="sess-abc-123")
    assert llm is not None
    assert isinstance(llm.chat_provider, Anthropic)
    assert llm.chat_provider._metadata == snapshot({"user_id": "sess-abc-123"})


def test_create_llm_anthropic_without_session_id():
    from kosong.contrib.chat_provider.anthropic import Anthropic

    provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="claude-sonnet-4-20250514",
        max_context_size=200000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Anthropic)
    assert llm.chat_provider._metadata is None


def test_create_llm_kimi_with_session_id(monkeypatch):
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="kimi-base",
        max_context_size=4096,
        capabilities=None,
    )
    monkeypatch.delenv("KIMI_MODEL_TOP_P", raising=False)
    monkeypatch.delenv("KIMI_MODEL_MAX_TOKENS", raising=False)

    llm = create_llm(provider, model, session_id="sess-abc-123")
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider._generation_kwargs["prompt_cache_key"] == "sess-abc-123"


def test_create_llm_openai_legacy_with_session_id(monkeypatch):
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )
    monkeypatch.delenv("KIMI_MODEL_TOP_P", raising=False)
    monkeypatch.delenv("KIMI_MODEL_MAX_TOKENS", raising=False)

    llm = create_llm(provider, model, session_id="sess-abc-123")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._generation_kwargs["user"] == "sess-abc-123"


def test_create_llm_openai_legacy_without_session_id(monkeypatch):
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )
    monkeypatch.delenv("KIMI_MODEL_TOP_P", raising=False)
    monkeypatch.delenv("KIMI_MODEL_MAX_TOKENS", raising=False)

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert "user" not in llm.chat_provider._generation_kwargs


def test_create_llm_openai_legacy_thinking_uses_max_completion_tokens(monkeypatch):
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="deepseek-v4-flash-0731",
        max_context_size=1000000,
        capabilities={"thinking"},
    )
    monkeypatch.delenv("KIMI_MODEL_TOP_P", raising=False)
    monkeypatch.delenv("KIMI_MODEL_MAX_TOKENS", raising=False)

    llm = create_llm(provider, model, max_tokens=384000, thinking_effort="max")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._generation_kwargs["max_completion_tokens"] == 384000
    assert "max_tokens" not in llm.chat_provider._generation_kwargs


def test_create_llm_openai_legacy_non_thinking_uses_max_tokens(monkeypatch):
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )
    monkeypatch.delenv("KIMI_MODEL_TOP_P", raising=False)
    monkeypatch.delenv("KIMI_MODEL_MAX_TOKENS", raising=False)

    llm = create_llm(provider, model, max_tokens=16384)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._generation_kwargs["max_tokens"] == 16384
    assert "max_completion_tokens" not in llm.chat_provider._generation_kwargs


def test_create_llm_openai_responses_with_session_id():
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model, session_id="sess-abc-123")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    assert llm.chat_provider._generation_kwargs["user"] == "sess-abc-123"


@pytest.mark.asyncio
async def test_create_llm_codex_oauth_uses_canonical_provider():
    provider = LLMProvider(
        type="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=SecretStr(""),
        oauth=OAuthRef(storage="file", key=CODEX_OAUTH_KEY),
    )
    model = LLMModel(
        model="gpt-5.4",
        max_context_size=272_000,
        capabilities={"thinking"},
    )
    config = Config(provider=provider, model=model, services=Services())
    oauth = OAuthManager(config)

    llm = create_llm(provider, model, session_id="session-1", oauth=oauth)

    assert llm is not None
    assert type(llm.chat_provider) is OpenAICodex
    assert llm.chat_provider.client.api_key == "oauth-managed"
    assert llm.chat_provider._session_id == "session-1"
    assert llm.chat_provider._own_http_client is True
    assert llm.chat_provider._client_kwargs["default_headers"] == {
        "User-Agent": "KimiCLI/kimix",
        "originator": "kimix",
    }
    await cast(OpenAICodex, llm.chat_provider).aclose()


def test_augment_provider_with_env_vars_xai(monkeypatch):
    provider = LLMProvider(
        type="xai",
        base_url="",
        api_key=SecretStr(""),
    )
    model = LLMModel(
        model="",
        max_context_size=0,
        capabilities=None,
    )

    monkeypatch.setenv("XAI_BASE_URL", "https://xai-env.test/v1")
    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    monkeypatch.setenv("XAI_MODEL_NAME", "grok-env")
    monkeypatch.setenv("XAI_MODEL_MAX_CONTEXT_SIZE", "131072")
    monkeypatch.setenv("XAI_MODEL_CAPABILITIES", "Image_In,THINKING,unknown")

    augment_provider_with_env_vars(provider, model)

    assert provider.type == "xai"
    assert provider.base_url == "https://xai-env.test/v1"
    assert provider.api_key.get_secret_value() == "xai-env-key"
    assert model.model == "grok-env"
    assert model.max_context_size == 131072
    assert model.capabilities == {"image_in", "thinking"}


def test_create_llm_xai_model_parameters():
    provider = LLMProvider(
        type="xai",
        base_url="https://api.x.ai/v1",
        api_key=SecretStr("xai-test-key"),
    )
    model = LLMModel(
        model="grok-3",
        max_context_size=131072,
    )

    llm = create_llm(provider, model, max_tokens=2048)
    assert llm is not None
    assert isinstance(llm.chat_provider, XAI)
    assert llm.chat_provider._generation_kwargs["max_output_tokens"] == 2048


def test_create_llm_xai_with_session_id():
    provider = LLMProvider(
        type="xai",
        base_url="https://api.x.ai/v1",
        api_key=SecretStr("xai-test-key"),
    )
    model = LLMModel(
        model="grok-3",
        max_context_size=131072,
    )

    llm = create_llm(provider, model, session_id="sess-abc-123")
    assert llm is not None
    assert isinstance(llm.chat_provider, XAI)
    assert llm.chat_provider._generation_kwargs["user"] == "sess-abc-123"


def test_create_llm_openai_responses_without_session_id():
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    assert "user" not in llm.chat_provider._generation_kwargs


def test_create_llm_openai_responses_maps_max_tokens_to_max_output_tokens():
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-5.6-sol",
        max_context_size=372000,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, max_tokens=128000, thinking=True)

    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    assert llm.chat_provider._generation_kwargs["max_output_tokens"] == 128000
    assert "max_tokens" not in llm.chat_provider._generation_kwargs
    assert "max_completion_tokens" not in llm.chat_provider._generation_kwargs


def test_create_llm_requires_base_url_for_kimi():
    provider = LLMProvider(type="kimi", base_url="", api_key=SecretStr("test-key"))
    model = LLMModel(model="kimi-base", max_context_size=4096)

    assert create_llm(provider, model) is None


def test_create_llm_openai_legacy_custom_headers():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "value", "X-Canary": "always"},
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._client_kwargs.get("default_headers") == {
        "X-Custom": "value",
        "X-Canary": "always",
    }


def test_create_llm_openai_legacy_default_reasoning_key():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="deepseek-reasoner",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._reasoning_key == "reasoning_content"


def test_create_llm_openai_legacy_custom_reasoning_key():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://example.test/v1",
        api_key=SecretStr("test-key"),
        reasoning_key="reasoning",
    )
    model = LLMModel(
        model="some-reasoner",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._reasoning_key == "reasoning"


def test_create_llm_openai_legacy_disabled_reasoning_key():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://example.test/v1",
        api_key=SecretStr("test-key"),
        reasoning_key="",
    )
    model = LLMModel(
        model="plain-model",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._reasoning_key == ""


def test_create_llm_openai_legacy_openai_settings():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
        openai_settings=OpenAISettings(thinking=False, chat_template_kwargs=False),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider._openai_settings == {
        "thinking": False,
        "reasoning": True,
        "chat_template_kwargs": False,
    }


def test_create_llm_openai_responses_custom_headers():
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "value"},
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    assert llm.chat_provider._client_kwargs.get("default_headers") == {
        "X-Custom": "value",
    }


def test_create_llm_anthropic_custom_headers():
    from kosong.contrib.chat_provider.anthropic import Anthropic

    provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "value"},
    )
    model = LLMModel(
        model="claude-sonnet-4-20250514",
        max_context_size=200000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Anthropic)
    # AsyncAnthropic stores custom headers in _custom_headers
    assert llm.chat_provider._client._custom_headers.get("X-Custom") == "value"


def test_create_llm_google_genai_custom_headers():
    from kosong.contrib.chat_provider.google_genai import GoogleGenAI

    provider = LLMProvider(
        type="google_genai",
        base_url="https://generativelanguage.googleapis.com",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "value"},
    )
    model = LLMModel(
        model="gemini-2.5-pro",
        max_context_size=1000000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, GoogleGenAI)
    # Google GenAI client stores http_options on _api_client
    http_options = llm.chat_provider._client._api_client._http_options
    assert http_options.headers is not None
    assert http_options.headers.get("X-Custom") == "value"


def test_create_llm_vertexai_custom_headers():
    from kosong.contrib.chat_provider.google_genai import GoogleGenAI

    provider = LLMProvider(
        type="vertexai",
        base_url="https://us-central1-aiplatform.googleapis.com",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "value"},
    )
    model = LLMModel(
        model="gemini-2.5-pro",
        max_context_size=1000000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, GoogleGenAI)
    http_options = llm.chat_provider._client._api_client._http_options
    assert http_options.headers is not None
    assert http_options.headers.get("X-Custom") == "value"


def test_create_llm_custom_headers_isolated_between_instances():
    """Mutating headers on one instance must not affect another created from the same provider."""
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
        custom_headers={"X-Custom": "original"},
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm1 = create_llm(provider, model)
    llm2 = create_llm(provider, model)
    assert llm1 is not None and llm2 is not None
    assert isinstance(llm1.chat_provider, OpenAILegacy)
    assert isinstance(llm2.chat_provider, OpenAILegacy)

    # Mutate headers on the first instance
    llm1.chat_provider._client_kwargs["default_headers"]["X-Custom"] = "mutated"

    # Second instance must be unaffected
    assert llm2.chat_provider._client_kwargs["default_headers"]["X-Custom"] == "original"
    # Original provider must also be unaffected
    assert provider.custom_headers is not None
    assert provider.custom_headers["X-Custom"] == "original"


def test_create_llm_no_custom_headers_has_empty_headers():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider.client._custom_headers == {}


def test_create_llm_openai_responses_thinking_false_no_reasoning_in_params():
    """thinking=False should call with_thinking("off"), which sets reasoning_effort=None.
    The OpenAIResponses provider handles this by omitting reasoning from the request."""
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="minimax/minimax-m2.5",
        max_context_size=128000,
        capabilities=None,
    )

    llm = create_llm(provider, model, thinking=False)

    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    # with_thinking("off") sets reasoning_effort=None in generation kwargs,
    # but generate() will omit reasoning from the actual API request when effort is None.
    assert llm.chat_provider.model_parameters == snapshot(
        {
            "base_url": "https://openrouter.ai/api/v1/",
            "reasoning_effort": None,
        }
    )


def _make_kimi_thinking_model() -> tuple[LLMProvider, LLMModel]:
    """Helper: build a kimi provider + always-thinking model pair."""
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="kimi-k2-thinking-turbo",
        max_context_size=4096,
        capabilities=None,
    )
    return provider, model


def test_create_llm_default_thinking_effort_is_max_anthropic():
    from kosong.contrib.chat_provider.anthropic import Anthropic

    provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="claude-opus-4-7",
        max_context_size=200000,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, thinking=True)
    assert llm is not None
    assert isinstance(llm.chat_provider, Anthropic)
    assert llm.chat_provider.thinking_effort == "max"


def test_create_llm_default_thinking_effort_is_max_openai_legacy():
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-5.1-codex-max",
        max_context_size=128000,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, thinking=True)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    # Default thinking effort should be 'max' (the highest level)
    assert llm.chat_provider.thinking_effort == "max"


def test_create_llm_thinking_effort_enables_thinking_when_thinking_unset():
    """Regression: a config like ``C:/dev/ds_cmdcode.json`` that sets
    ``thinking_effort: "high"`` and capabilities ``["thinking"]`` (but no
    explicit ``default_thinking``) must enable thinking even when
    ``create_llm`` is called without an explicit ``thinking`` argument.

    Before the fix the provider stayed in default mode (``thinking_effort``
    property ``None``), so the request went out with ``thinking.type:
    "disabled"`` and the backend returned no reasoning content.
    """
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.commandcode.ai/provider/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="deepseek/deepseek-v4-flash",
        max_context_size=1024000,
        max_tokens=131072,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, thinking_effort="high")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider.thinking_effort == "high"


def test_create_llm_thinking_effort_off_does_not_enable_thinking():
    """``thinking_effort: "off"`` must never flip thinking on when no explicit
    ``thinking`` argument is given."""
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.commandcode.ai/provider/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="deepseek/deepseek-v4-flash",
        max_context_size=1024000,
        max_tokens=131072,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, thinking_effort="off")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider.thinking_effort != "high"
    assert llm.chat_provider.thinking_effort is None


def test_create_llm_explicit_thinking_false_overrides_thinking_effort():
    """An explicit ``thinking=False`` must disable thinking even when the
    config carries a non-``off`` ``thinking_effort``."""
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.commandcode.ai/provider/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="deepseek/deepseek-v4-flash",
        max_context_size=1024000,
        max_tokens=131072,
        capabilities={"thinking"},
    )

    llm = create_llm(provider, model, thinking=False, thinking_effort="high")
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAILegacy)
    assert llm.chat_provider.thinking_effort == "off"


def test_create_llm_supported_efforts_clamps_xhigh():
    from kosong.contrib.chat_provider.anthropic import Anthropic

    provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="claude-opus-4-7",
        max_context_size=200000,
        capabilities={"thinking"},
        supported_efforts={"low", "medium", "high"},
    )

    llm = create_llm(provider, model, thinking=True, thinking_effort="max")
    assert llm is not None
    assert isinstance(llm.chat_provider, Anthropic)
    assert llm.chat_provider.thinking_effort == "high"


def test_create_llm_supported_efforts_passes_max():
    from kosong.contrib.chat_provider.anthropic import Anthropic
    from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

    anthropic_provider = LLMProvider(
        type="anthropic",
        base_url="https://api.anthropic.com",
        api_key=SecretStr("test-key"),
    )
    anthropic_model = LLMModel(
        model="claude-opus-4-7",
        max_context_size=200000,
        capabilities={"thinking"},
    )
    anthropic_llm = create_llm(
        anthropic_provider, anthropic_model, thinking=True, thinking_effort="max"
    )
    assert anthropic_llm is not None
    assert isinstance(anthropic_llm.chat_provider, Anthropic)
    assert anthropic_llm.chat_provider.thinking_effort == "max"

    openai_provider = LLMProvider(
        type="openai_legacy",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    openai_model = LLMModel(
        model="gpt-5.1-codex-max",
        max_context_size=128000,
        capabilities={"thinking"},
    )
    openai_llm = create_llm(openai_provider, openai_model, thinking=True, thinking_effort="max")
    assert openai_llm is not None
    assert isinstance(openai_llm.chat_provider, OpenAILegacy)
    assert openai_llm.chat_provider.thinking_effort == "max"


def _make_kimi_plain_model() -> tuple[LLMProvider, LLMModel]:
    """Helper: build a kimi provider + non-thinking model pair."""
    provider = LLMProvider(
        type="kimi",
        base_url="https://api.test/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="kimi-k2-turbo-preview",
        max_context_size=4096,
        capabilities=None,
    )
    return provider, model


def test_create_llm_kimi_thinking_keep_not_set_omits_field(monkeypatch):
    """When KIMI_MODEL_THINKING_KEEP is unset, extra_body.thinking must not
    contain a ``keep`` key, even for always-thinking models."""
    monkeypatch.delenv("KIMI_MODEL_THINKING_KEEP", raising=False)
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    extra_body = llm.chat_provider.model_parameters.get("extra_body") or {}
    thinking = extra_body.get("thinking") or {}
    assert "keep" not in thinking
    assert thinking.get("type") == "enabled"


def test_create_llm_kimi_thinking_keep_empty_string_omits_field(monkeypatch):
    """An empty-string env value must be treated as unset (consistent with
    other KIMI_MODEL_* envs that use walrus-truthy reads)."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "")
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    extra_body = llm.chat_provider.model_parameters.get("extra_body") or {}
    thinking = extra_body.get("thinking") or {}
    assert "keep" not in thinking


def test_create_llm_kimi_thinking_keep_all_injects_field(monkeypatch):
    """With a thinking-capable model and KIMI_MODEL_THINKING_KEEP=all, the
    provider's extra_body.thinking must carry both ``type`` (set by
    with_thinking) and ``keep`` (set by the env)."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "all")
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    assert llm.chat_provider.model_parameters.get("extra_body") == snapshot(
        {"thinking": {"type": "enabled", "effort": "max", "keep": "all"}}
    )


def test_create_llm_kimi_thinking_keep_arbitrary_value_passes_through(monkeypatch):
    """Non-'all' values must be forwarded unchanged — no casing normalization,
    no validation. The Moonshot API is the source of truth."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "xYz")
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    extra_body = llm.chat_provider.model_parameters.get("extra_body") or {}
    assert extra_body.get("thinking", {}).get("keep") == "xYz"


def test_create_llm_kimi_thinking_keep_skipped_when_thinking_off(monkeypatch):
    """When thinking=False (with_thinking("off")), keep must NOT be injected,
    even if the env is set. Avoids sending a `thinking.keep` without an
    accompanying `thinking.type` that the API actually honors."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "all")
    provider, model = _make_kimi_plain_model()
    # capabilities is None and model name has no "thinking"/"reason" marker, so
    # derive_model_capabilities returns an empty set. thinking=False then drives
    # with_thinking("off").
    llm = create_llm(provider, model, thinking=False)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    extra_body = llm.chat_provider.model_parameters.get("extra_body") or {}
    thinking = extra_body.get("thinking") or {}
    assert "keep" not in thinking


def test_create_llm_kimi_thinking_keep_skipped_when_no_thinking_branch(monkeypatch):
    """When the model has no thinking capability and thinking is None, neither
    with_thinking branch runs — keep must also NOT be injected."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "all")
    provider, model = _make_kimi_plain_model()

    llm = create_llm(provider, model, thinking=None)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    extra_body = llm.chat_provider.model_parameters.get("extra_body") or {}
    # extra_body might be missing entirely (no thinking branch ran), or present
    # with no thinking key. Both are acceptable; what must hold is "no keep".
    thinking = extra_body.get("thinking") or {}
    assert "keep" not in thinking


def test_create_llm_kimi_thinking_keep_injected_on_explicit_thinking_true(monkeypatch):
    """Covers the second half of the ``thinking_on`` condition: a
    thinking-capable (but not always_thinking) model with explicit
    ``thinking=True``. This exercises a different branch of
    ``"always_thinking" in capabilities or (thinking is True and "thinking" in capabilities)``
    than the always-thinking-name-based tests above."""
    monkeypatch.setenv("KIMI_MODEL_THINKING_KEEP", "all")
    provider, model = _make_kimi_plain_model()
    # Model name has no "thinking"/"reason" marker, so derive_model_capabilities
    # returns an empty set; manually granting only the "thinking" capability
    # means always_thinking is NOT in capabilities — thinking_on is driven
    # solely by the explicit thinking=True argument.
    model.capabilities = {"thinking"}

    llm = create_llm(provider, model, thinking=True)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)

    assert llm.chat_provider.model_parameters.get("extra_body") == snapshot(
        {"thinking": {"type": "enabled", "effort": "max", "keep": "all"}}
    )


def test_create_llm_kimi_thinking_on_forces_temperature_to_one(monkeypatch):
    """Always-thinking kimi models must use temperature=1.0."""
    monkeypatch.delenv("KIMI_MODEL_TEMPERATURE", raising=False)
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider.model_parameters["temperature"] == 1.0


def test_create_llm_kimi_thinking_off_forces_temperature_to_zero_six(monkeypatch):
    """Non-thinking kimi models must use temperature=0.6."""
    monkeypatch.delenv("KIMI_MODEL_TEMPERATURE", raising=False)
    provider, model = _make_kimi_plain_model()

    llm = create_llm(provider, model, thinking=False)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider.model_parameters["temperature"] == 0.6


def test_create_llm_kimi_explicit_temperature_ignored(monkeypatch):
    """Explicit temperature argument must not override kimi forced default."""
    monkeypatch.delenv("KIMI_MODEL_TEMPERATURE", raising=False)
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model, temperature=0.5)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider.model_parameters["temperature"] == 1.0


def test_create_llm_kimi_env_temperature_ignored(monkeypatch):
    """KIMI_MODEL_TEMPERATURE env var must not override kimi forced default."""
    monkeypatch.setenv("KIMI_MODEL_TEMPERATURE", "0.3")
    provider, model = _make_kimi_thinking_model()

    llm = create_llm(provider, model)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider.model_parameters["temperature"] == 1.0


def test_create_llm_kimi_config_temperature_ignored(monkeypatch):
    """Config-level temperature (passed as the temperature kwarg) must not
    override kimi forced default for thinking-off models."""
    monkeypatch.delenv("KIMI_MODEL_TEMPERATURE", raising=False)
    provider, model = _make_kimi_plain_model()

    llm = create_llm(provider, model, thinking=False, temperature=0.7)
    assert llm is not None
    assert isinstance(llm.chat_provider, Kimi)
    assert llm.chat_provider.model_parameters["temperature"] == 0.6


def test_create_llm_non_kimi_not_affected_by_kimi_temperature_forcing():
    """Non-kimi providers must not have their temperature forced to the kimi
    defaults. The explicit temperature argument is ignored for this provider
    path (pre-existing behavior); this test guards against accidentally
    applying the kimi logic to other providers."""
    provider = LLMProvider(
        type="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("test-key"),
    )
    model = LLMModel(
        model="gpt-4o",
        max_context_size=128000,
    )

    llm = create_llm(provider, model, temperature=0.5)
    assert llm is not None
    assert isinstance(llm.chat_provider, OpenAIResponses)
    assert "temperature" not in llm.chat_provider.model_parameters
