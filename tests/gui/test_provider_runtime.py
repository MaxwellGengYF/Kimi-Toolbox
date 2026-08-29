from __future__ import annotations

from typing import Any, cast

import pytest
from kimi_cli.auth.codex import CodexModel
from kimi_cli.llm_codex import CodexProviderRuntime

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

    def with_thinking(self, effort: str) -> FakeProvider:
        self.updated = True
        return FakeProvider({**self._generation_kwargs, "reasoning_effort": effort})

    def with_generation_kwargs(self, **kwargs: object) -> FakeProvider:
        self.updated = True
        return FakeProvider({**self._generation_kwargs, **kwargs})


class FakeLease:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def runtime_for(model: CodexModel) -> tuple[CodexProviderRuntime, FakeProvider, FakeLease]:
    provider = FakeProvider({"reasoning_effort": "low", "parallel_tool_calls": True})
    lease = FakeLease(provider)
    runtime = CodexProviderRuntime(
        {"thinking_effort": "low"},
        cast(Any, provider),
        cast(Any, lease),
        model,
    )
    return runtime, provider, lease


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ["medium", "max"])
async def test_chatgpt_plugin_applies_the_exact_catalog_parameter(
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    runtime, original_provider, lease = runtime_for(
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "medium", "ultra", "none", "future-effort"),
            default_reasoning_effort="low",
        )
    )
    calls: list[dict[str, object]] = []

    async def fake_create(**kwargs: object) -> CodexProviderRuntime:
        calls.append(kwargs)
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)
    result = await ChatGPTProviderKind().create_runtime(
        ChatGPTTarget("gpt-test"),
        session_id="session-1",
        overrides=RuntimeOverrides(thinking_effort=selected),
    )

    assert calls == [
        {
            "model_name": "gpt-test",
            "session_id": "session-1",
            "thinking": False,
        }
    ]
    assert original_provider.updated
    assert result.provider is not None
    assert cast(FakeProvider, result.provider)._generation_kwargs["reasoning_effort"] == selected
    assert cast(FakeProvider, result.provider)._generation_kwargs["parallel_tool_calls"] is True
    assert result.lease is lease
    assert lease.provider is result.provider
    assert not lease.closed
    assert result.provider_dict["thinking_effort"] == selected


@pytest.mark.asyncio
async def test_chatgpt_plugin_rejects_a_removed_parameter_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, original_provider, lease = runtime_for(
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="low",
        )
    )

    async def fake_create(**_kwargs: object) -> CodexProviderRuntime:
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)
    with pytest.raises(LLMSelectionError) as caught:
        await ChatGPTProviderKind().create_runtime(
            ChatGPTTarget("gpt-test"),
            session_id="session-1",
            overrides=RuntimeOverrides(thinking_effort="retired-effort"),
        )

    assert caught.value.problem.kind == PROBLEM_PARAMETER_VALUE_UNAVAILABLE
    assert lease.closed
    assert not original_provider.updated


@pytest.mark.asyncio
async def test_chatgpt_plugin_passes_future_generic_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, original_provider, lease = runtime_for(CodexModel("plain-model"))

    async def fake_create(**_kwargs: object) -> CodexProviderRuntime:
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)
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

    assert original_provider.updated
    assert result.provider is not None
    generation = cast(FakeProvider, result.provider)._generation_kwargs
    assert generation["temperature"] == 0.2
    assert generation["beta_features"] == ["long-context"]
    assert result.provider_dict["max_context_size"] == 500_000
    assert result.provider_dict["max_tokens"] == 12_000
    assert result.provider_dict["beta_features"] == ["long-context"]
    assert not lease.closed


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
async def test_chatgpt_plugin_closes_lease_when_override_application_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProvider(FakeProvider):
        def with_thinking(self, effort: str) -> FakeProvider:
            del effort
            raise RuntimeError("override failed")

    provider = FailingProvider()
    lease = FakeLease(provider)
    runtime = CodexProviderRuntime(
        {"thinking_effort": "low"},
        cast(Any, provider),
        cast(Any, lease),
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="low",
        ),
    )

    async def fake_create(**_kwargs: object) -> CodexProviderRuntime:
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)

    with pytest.raises(RuntimeError, match="override failed"):
        await ChatGPTProviderKind().create_runtime(
            ChatGPTTarget("gpt-test"),
            session_id="session-1",
            overrides=RuntimeOverrides(thinking_effort="high"),
        )

    assert lease.closed
