from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from kimi_cli.auth.codex import CodexModel

from kimix_gui.llm import (
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
    ChatGPTTarget,
    LLMSelectionError,
    RuntimeOverrides,
    SessionRuntime,
    apply_overrides,
)
from kimix_gui.llm.providers import chatgpt as chatgpt_provider
from kimix_gui.llm.providers.chatgpt import ChatGPTProviderKind


class FakeProvider:
    def __init__(self, generation_kwargs: dict[str, object] | None = None) -> None:
        self._generation_kwargs = dict(generation_kwargs or {})
        self.updated = False
        self.closed = False

    def with_thinking(self, effort: str) -> FakeProvider:
        self.updated = True
        return FakeProvider({**self._generation_kwargs, "reasoning_effort": effort})

    def with_generation_kwargs(self, **kwargs: object) -> FakeProvider:
        self.updated = True
        return FakeProvider({**self._generation_kwargs, **kwargs})

    async def aclose(self) -> None:
        self.closed = True


def _install_catalog_and_llm(
    monkeypatch: pytest.MonkeyPatch,
    model: CodexModel,
    provider: FakeProvider,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    async def fake_resolve(model_name: str, **_kwargs: object) -> CodexModel:
        assert model_name == model.slug
        return model

    def fake_create_llm(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(chat_provider=provider)

    monkeypatch.setattr(chatgpt_provider, "resolve_codex_model", fake_resolve)
    monkeypatch.setattr(chatgpt_provider, "create_llm", fake_create_llm)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ["medium", "max"])
async def test_chatgpt_plugin_applies_the_exact_catalog_parameter(
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    model = CodexModel(
        "gpt-test",
        reasoning_efforts=("low", "medium", "ultra", "none", "future-effort"),
        default_reasoning_effort="low",
    )
    provider = FakeProvider({"reasoning_effort": "low", "parallel_tool_calls": True})
    calls = _install_catalog_and_llm(monkeypatch, model, provider)

    result = await ChatGPTProviderKind().create_runtime(
        ChatGPTTarget("gpt-test"),
        session_id="session-1",
        overrides=RuntimeOverrides(thinking_effort=selected),
    )

    assert calls == [
        {
            "thinking": True,
            "session_id": "session-1",
            "thinking_effort": selected,
        }
    ]
    assert original_updated(provider, result)
    assert result.provider is not None
    assert cast(FakeProvider, result.provider)._generation_kwargs["reasoning_effort"] == selected
    assert cast(FakeProvider, result.provider)._generation_kwargs["parallel_tool_calls"] is True
    assert result.lease is None
    assert not provider.closed
    assert result.provider_dict["thinking_effort"] == selected
    assert result.provider_dict["supported_efforts"] == ["low", "medium", "max"]
    assert "loop_control" not in result.provider_dict
    assert result.provider_dict["api_key"] == "oauth-managed"


def original_updated(original: FakeProvider, result: SessionRuntime) -> bool:
    return original.updated and result.provider is not original


@pytest.mark.asyncio
async def test_chatgpt_plugin_uses_catalog_default_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CodexModel(
        "gpt-test",
        max_context_size=272_000,
        max_tokens=128_000,
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_reasoning_effort="medium",
    )
    provider = FakeProvider()
    calls = _install_catalog_and_llm(monkeypatch, model, provider)

    result = await ChatGPTProviderKind().create_runtime(
        ChatGPTTarget("gpt-test"),
        session_id="session-1",
        overrides=RuntimeOverrides(),
    )

    assert calls[0]["thinking_effort"] == "medium"
    assert result.provider_dict["thinking_effort"] == "medium"
    assert result.provider_dict["max_context_size"] == 272_000
    assert result.provider_dict["max_tokens"] == 128_000
    assert result.lease is None


@pytest.mark.asyncio
async def test_chatgpt_plugin_rejects_a_removed_parameter_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CodexModel(
        "gpt-test",
        reasoning_efforts=("low", "high"),
        default_reasoning_effort="low",
    )
    provider = FakeProvider()
    calls = _install_catalog_and_llm(monkeypatch, model, provider)

    with pytest.raises(LLMSelectionError) as caught:
        await ChatGPTProviderKind().create_runtime(
            ChatGPTTarget("gpt-test"),
            session_id="session-1",
            overrides=RuntimeOverrides(thinking_effort="retired-effort"),
        )

    assert caught.value.problem.kind == PROBLEM_PARAMETER_VALUE_UNAVAILABLE
    assert calls == []
    assert not provider.closed
    assert not provider.updated


@pytest.mark.asyncio
async def test_chatgpt_plugin_passes_future_generic_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CodexModel("plain-model")
    provider = FakeProvider()
    _install_catalog_and_llm(monkeypatch, model, provider)

    result = await ChatGPTProviderKind().create_runtime(
        ChatGPTTarget("plain-model"),
        session_id="session-1",
        overrides=RuntimeOverrides(
            max_context_size=500_000,
            max_tokens=12_000,
            beta_features=("long-context",),
            generation_kwargs=(("temperature", 0.2),),
        ),
    )

    assert provider.updated
    assert result.provider is not None
    generation = cast(FakeProvider, result.provider)._generation_kwargs
    assert generation["temperature"] == 0.2
    assert generation["beta_features"] == ["long-context"]
    assert result.provider_dict["max_context_size"] == 500_000
    assert result.provider_dict["max_tokens"] == 12_000
    assert result.provider_dict["beta_features"] == ["long-context"]
    assert result.lease is None
    assert not provider.closed


class SlottedLease:
    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_apply_overrides_supports_protocol_only_leases_and_merges_beta_features() -> None:
    provider = FakeProvider()
    lease = SlottedLease()
    runtime = SessionRuntime(
        {"beta_features": ["existing", "shared"]},
        "gpt-test",
        cast(Any, provider),
        lease,
    )

    result = apply_overrides(
        runtime,
        RuntimeOverrides(beta_features=("shared", "context-1m")),
    )

    assert result.provider_dict["beta_features"] == [
        "existing",
        "shared",
        "context-1m",
    ]
    assert result.provider is not None
    assert cast(FakeProvider, result.provider)._generation_kwargs["beta_features"] == [
        "existing",
        "shared",
        "context-1m",
    ]
    assert not lease.closed


@pytest.mark.asyncio
async def test_chatgpt_plugin_closes_provider_when_override_application_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProvider(FakeProvider):
        def with_thinking(self, effort: str) -> FakeProvider:
            del effort
            raise RuntimeError("override failed")

    model = CodexModel(
        "gpt-test",
        reasoning_efforts=("low", "high"),
        default_reasoning_effort="low",
    )
    provider = FailingProvider()
    _install_catalog_and_llm(monkeypatch, model, provider)

    with pytest.raises(RuntimeError, match="override failed"):
        await ChatGPTProviderKind().create_runtime(
            ChatGPTTarget("gpt-test"),
            session_id="session-1",
            overrides=RuntimeOverrides(thinking_effort="high"),
        )

    assert provider.closed
    assert not provider.updated
