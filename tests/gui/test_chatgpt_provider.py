from __future__ import annotations

from typing import Any, cast

import pytest
from kimi_cli.auth.codex import CodexModel
from kimi_cli.llm_codex import CodexProviderRuntime

from kimix_gui import chatgpt_provider
from kimix_gui.chatgpt_provider import create_selected_codex_provider
from kimix_gui.llm import PROBLEM_VARIANT_UNAVAILABLE, LLMSelectionError


class FakeProvider:
    def __init__(self, generation_kwargs: dict[str, object] | None = None) -> None:
        self._generation_kwargs = dict(generation_kwargs or {})
        self.updated = False

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
@pytest.mark.parametrize(
    ("selected", "metadata_effort"),
    [
        ("medium", "medium"),
        ("max", "max"),
        ("none", None),
        ("future-effort", None),
    ],
)
async def test_gui_adapter_pins_the_exact_catalog_effort_without_core_derivation(
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
    metadata_effort: str | None,
) -> None:
    runtime, original_provider, lease = runtime_for(
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "medium", "ultra", "none", "future-effort"),
            default_reasoning_effort="low",
        )
    )
    calls: list[dict[str, object]] = []

    async def fake_create(_service: object, **kwargs: object) -> CodexProviderRuntime:
        calls.append(kwargs)
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)

    result = await create_selected_codex_provider(
        cast(Any, object()),
        model_name="gpt-test",
        session_id="session-1",
        reasoning_effort=selected,
    )

    assert calls == [
        {
            "model_name": "gpt-test",
            "session_id": "session-1",
            "thinking": False,
        }
    ]
    assert original_provider.updated
    assert result.provider._generation_kwargs["reasoning_effort"] == selected
    assert result.provider._generation_kwargs["parallel_tool_calls"] is True
    assert result.lease.provider is result.provider
    assert not lease.closed
    if metadata_effort is None:
        assert "thinking_effort" not in result.provider_dict
    else:
        assert result.provider_dict["thinking_effort"] == metadata_effort


@pytest.mark.asyncio
async def test_gui_adapter_rejects_a_removed_effort_before_building_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, original_provider, lease = runtime_for(
        CodexModel(
            "gpt-test",
            reasoning_efforts=("low", "high"),
            default_reasoning_effort="low",
        )
    )

    async def fake_create(_service: object, **_kwargs: object) -> CodexProviderRuntime:
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)

    with pytest.raises(LLMSelectionError) as caught:
        await create_selected_codex_provider(
            cast(Any, object()),
            model_name="gpt-test",
            session_id="session-1",
            reasoning_effort="retired-effort",
        )

    assert caught.value.problem.kind == PROBLEM_VARIANT_UNAVAILABLE
    assert lease.closed
    assert not original_provider.updated


@pytest.mark.asyncio
async def test_gui_adapter_keeps_provider_default_when_catalog_has_no_efforts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _original_provider, lease = runtime_for(CodexModel("plain-model"))

    async def fake_create(_service: object, **_kwargs: object) -> CodexProviderRuntime:
        return runtime

    monkeypatch.setattr(chatgpt_provider, "create_codex_provider", fake_create)

    result = await create_selected_codex_provider(
        cast(Any, object()),
        model_name="plain-model",
        session_id="session-1",
        reasoning_effort=None,
    )

    assert result.provider._generation_kwargs["reasoning_effort"] is None
    assert "thinking_effort" not in result.provider_dict
    assert not lease.closed
