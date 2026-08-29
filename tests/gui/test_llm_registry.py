from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import orjson
import pytest

from kimix_gui.llm import (
    AXIS_CONTEXT_WINDOW,
    CatalogContext,
    ChatGPTTarget,
    KimixGuiConfigStore,
    LLMModelDescriptor,
    LLMSelection,
    ModelCatalogService,
    ParameterAssignment,
    ProviderFileTarget,
    ProviderRegistry,
    RuntimeOverrides,
    SessionRuntime,
    default_provider_registry,
    inspect_provider_file,
    resolve_selection,
)
from kimix_gui.llm.providers import provider_file as provider_file_module
from kimix_gui.llm.providers.provider_file import ProviderFileProviderKind


def test_legacy_llm_modules_are_removed_and_app_uses_catalog_services() -> None:
    repository = Path(__file__).resolve().parents[2]
    package = repository / "src" / "kimix_gui"
    removed = (
        package / "chatgpt_provider.py",
        package / "llm" / "catalog.py",
        package / "llm" / "provider_file.py",
        package / "qt" / "components" / "variant_picker.py",
    )

    assert all(not path.exists() for path in removed)
    app_source = (package / "app.py").read_text(encoding="utf-8")
    assert "chatgpt_models" not in app_source
    assert "CodexModelCatalog" not in app_source


def test_default_registry_dispatches_every_builtin_target(tmp_path: Path) -> None:
    registry = default_provider_registry()

    assert tuple(provider.id for provider in registry.providers) == (
        "chatgpt",
        "provider_file",
    )
    assert registry.for_target(ChatGPTTarget("gpt-test")).id == "chatgpt"
    assert registry.for_target(ProviderFileTarget(tmp_path / "provider.json")).id == "provider_file"


@dataclass(frozen=True, slots=True)
class FakeOverrideProvider:
    thinking_effort: str | None = None
    generation_kwargs: tuple[tuple[str, object], ...] = ()

    def with_thinking(self, effort: str) -> FakeOverrideProvider:
        return replace(self, thinking_effort=effort)

    def with_generation_kwargs(self, **kwargs: Any) -> FakeOverrideProvider:
        generation = dict(self.generation_kwargs)
        generation.update(kwargs)
        return replace(self, generation_kwargs=tuple(generation.items()))


@pytest.mark.asyncio
async def test_provider_file_context_axis_updates_request_and_compaction_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_file = tmp_path / "anthropic.json"
    provider_file.write_bytes(
        orjson.dumps(
            {
                "model": "bedrock-claude-sonnet-4-20250929",
                "max_context_size": 200_000,
                "max_tokens": 32_000,
                "capabilities": [],
                "url": "https://example.test/v1",
                "type": "anthropic",
                "api_key": "test-key",
            }
        )
    )
    fake_provider = FakeOverrideProvider()
    monkeypatch.setattr(
        provider_file_module,
        "create_llm",
        lambda *_args, **_kwargs: SimpleNamespace(chat_provider=fake_provider),
    )
    descriptor = inspect_provider_file(provider_file)
    selection = LLMSelection(
        descriptor.target,
        ParameterAssignment({AXIS_CONTEXT_WINDOW: "1m"}),
    )
    resolved = resolve_selection(selection, [descriptor])

    runtime = await ProviderFileProviderKind().create_runtime(
        descriptor.target,
        session_id="session-test",
        overrides=resolved.runtime,
    )

    assert resolved.available
    assert runtime.provider_dict["max_context_size"] == 1_000_000
    assert runtime.provider_dict["beta_features"] == ["context-1m-2025-08-07"]
    assert runtime.provider is not None
    assert dict(runtime.provider.generation_kwargs)["beta_features"] == [
        "context-1m-2025-08-07"
    ]


@dataclass(frozen=True, slots=True)
class ExampleTarget:
    name: str
    kind: str = "example"
    provider_id: str = "example"

    @property
    def key(self) -> str:
        return f"example:{self.name}"


class ExampleProvider:
    id = "example"
    def owns(self, target: object) -> bool:
        return isinstance(target, ExampleTarget)

    def describe(self, target: object, context: CatalogContext) -> LLMModelDescriptor:
        del context
        example = cast(ExampleTarget, target)
        return LLMModelDescriptor(
            target=example,
            model_id=example.name,
            provider_type=self.id,
            endpoint="memory://example",
            credential="None",
            file_format="Plugin",
        )

    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]:
        return (self.describe(ExampleTarget("one"), context),)

    async def create_runtime(
        self,
        target: object,
        *,
        session_id: str,
        overrides: RuntimeOverrides,
    ) -> SessionRuntime:
        del target, session_id, overrides
        return SessionRuntime({}, None, None)


def test_third_party_provider_registration_needs_no_app_or_backend_branch() -> None:
    provider = ExampleProvider()
    registry = ProviderRegistry((provider,))
    target = ExampleTarget("one")

    assert registry.for_target(target) is provider
    assert registry.describe(target, CatalogContext()).model_id == "one"
    assert registry.list_models(CatalogContext())[0].target == target
    assert LLMSelection(target).target.provider_id == "example"




def test_catalog_service_accepts_the_exact_provider_protocol_without_extra_flags(
    tmp_path: Path,
) -> None:
    provider = ExampleProvider()
    registry = ProviderRegistry((provider,))
    store = KimixGuiConfigStore(tmp_path / "kimix-gui.json")
    service = ModelCatalogService(registry, store, tmp_path)

    assert service.ready_for(LLMSelection(ExampleTarget("one")))
    assert service.models_for()[0].target == ExampleTarget("one")


class InvalidCatalogProvider(ExampleProvider):
    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]:
        del context
        return (
            LLMModelDescriptor(
                target=ChatGPTTarget("foreign"),
                model_id="foreign",
                provider_type=self.id,
                endpoint="memory://example",
                credential="None",
                file_format="Plugin",
            ),
        )


def test_registry_rejects_catalog_targets_not_owned_by_the_listing_provider() -> None:
    registry = ProviderRegistry((InvalidCatalogProvider(),))

    with pytest.raises(TypeError, match="Expected exactly one provider"):
        registry.list_models(CatalogContext())


class OverlappingExampleProvider(ExampleProvider):
    id = "overlap"

    def owns(self, target: object) -> bool:
        return isinstance(target, ExampleTarget)


def test_registry_rejects_overlapping_target_ownership() -> None:
    registry = ProviderRegistry((ExampleProvider(), OverlappingExampleProvider()))

    with pytest.raises(TypeError, match="Expected exactly one provider"):
        registry.list_models(CatalogContext())
