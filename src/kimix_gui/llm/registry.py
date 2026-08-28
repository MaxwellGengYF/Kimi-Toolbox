"""Provider registration and refreshable model-catalog ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from kimi_cli.auth.codex import CodexAuthService, CodexAuthSnapshot, CodexModelCatalog

from kimix_gui.llm.domain import (
    LLMModelDescriptor,
    LLMSelection,
    ProviderTarget,
    ResolvedLLMSelection,
    resolve_selection,
    target_key,
)
from kimix_gui.llm.parameters import RuntimeOverrides
from kimix_gui.llm.providers.base import (
    CatalogContext,
    ProviderKind,
    SessionRuntime,
)
from kimix_gui.llm.providers.chatgpt import ChatGPTProviderKind
from kimix_gui.llm.providers.provider_file import ProviderFileProviderKind
from kimix_gui.llm.store import KimixGuiConfigStore

CatalogListener = Callable[[], None]
SelectionWriter = Callable[[LLMSelection], None]


class ProviderRegistry:
    """Dispatch targets to provider implementations without concrete type branches."""

    def __init__(self, providers: Iterable[ProviderKind]) -> None:
        registered: dict[str, ProviderKind] = {}
        for provider in providers:
            if provider.id in registered:
                raise ValueError(f"Duplicate provider kind id: {provider.id}")
            registered[provider.id] = provider
        if not registered:
            raise ValueError("Provider registry must not be empty")
        self._providers = tuple(registered.values())

    @property
    def providers(self) -> tuple[ProviderKind, ...]:
        return self._providers

    def for_id(self, provider_id: str) -> ProviderKind:
        matches = tuple(provider for provider in self._providers if provider.id == provider_id)
        if len(matches) != 1:
            raise KeyError(f"Unknown provider kind id: {provider_id}")
        return matches[0]

    def for_target(self, target: ProviderTarget) -> ProviderKind:
        matches = tuple(provider for provider in self._providers if provider.owns(target))
        if len(matches) != 1:
            raise TypeError(
                f"Expected exactly one provider for target {target!r}, found {len(matches)}"
            )
        return matches[0]

    def describe(
        self,
        target: ProviderTarget,
        context: CatalogContext,
    ) -> LLMModelDescriptor:
        return self.for_target(target).describe(target, context)

    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]:
        models: list[LLMModelDescriptor] = []
        seen: set[str] = set()
        for provider in self._providers:
            provider_models = sorted(
                provider.list_models(context),
                key=lambda model: model.priority,
            )
            for model in provider_models:
                if self.for_target(model.target) is not provider:
                    raise TypeError(
                        f"Provider {provider.id!r} listed a target it does not uniquely own: "
                        f"{model.target!r}"
                    )
                key = target_key(model.target)
                if key in seen:
                    raise ValueError(f"Duplicate catalog target key: {key}")
                seen.add(key)
                models.append(model)
        return tuple(models)

    async def create_runtime(
        self,
        target: ProviderTarget,
        *,
        session_id: str,
        overrides: RuntimeOverrides,
    ) -> SessionRuntime:
        return await self.for_target(target).create_runtime(
            target,
            session_id=session_id,
            overrides=overrides,
        )


def default_provider_registry(codex_service: CodexAuthService) -> ProviderRegistry:
    """Build the built-in registry around the process's shared Codex service."""

    return ProviderRegistry(
        (
            ChatGPTProviderKind(codex_service),
            ProviderFileProviderKind(),
        )
    )


class ModelCatalogService:
    """Own catalog state, model aggregation, exact resolution, and writeback."""

    def __init__(
        self,
        registry: ProviderRegistry,
        store: KimixGuiConfigStore,
        work_dir: Path,
        *,
        context: CatalogContext | None = None,
    ) -> None:
        self.registry = registry
        self._store = store
        self._work_dir = work_dir
        self._context = context or CatalogContext()
        self._operation_id = max(
            self._context.codex_snapshot.operation_id,
            self._context.codex_catalog.operation_id,
        )
        self._listeners: list[CatalogListener] = []

    @property
    def context(self) -> CatalogContext:
        return self._context

    @property
    def ready(self) -> bool:
        return self._context.ready

    def ready_for(self, selection: LLMSelection) -> bool:
        """Return whether this target has enough initial catalog state to resolve."""

        provider = self.registry.for_target(selection.target)
        return self.ready if getattr(provider, "catalog_managed", False) else True

    def subscribe(self, listener: CatalogListener) -> Callable[[], None]:
        """Notify ``listener`` once for every accepted catalog-state change."""

        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def models_for(self, *selections: LLMSelection) -> tuple[LLMModelDescriptor, ...]:
        """Aggregate enumerable models, saved files, and exact referenced targets."""

        models: dict[str, LLMModelDescriptor] = {
            target_key(model.target): model
            for model in self.registry.list_models(self._context)
        }
        targets: list[ProviderTarget] = list(
            self._store.provider_targets_for(self._work_dir)
        )
        targets.extend(selection.target for selection in selections)
        for target in targets:
            key = target_key(target)
            if key not in models:
                models[key] = self.registry.describe(target, self._context)
        return tuple(models.values())

    def resolve(self, selection: LLMSelection) -> ResolvedLLMSelection:
        return resolve_selection(selection, self.models_for(selection))

    def materialize(
        self,
        selection: LLMSelection,
        *,
        writer: SelectionWriter | None = None,
    ) -> LLMSelection:
        """Pin catalog defaults and optionally persist only a valid fresh result."""

        resolved = self.resolve(selection)
        materialized = resolved.materialized_selection
        if not resolved.available or not resolved.needs_writeback:
            return selection
        if writer is not None:
            try:
                writer(materialized)
            except OSError:
                return selection
        return materialized

    def refresh_default(
        self,
        selection: LLMSelection,
        *,
        persisted: bool,
    ) -> tuple[LLMSelection, ResolvedLLMSelection]:
        """Materialize a project default and return its current resolved snapshot."""

        writer: SelectionWriter | None = None
        if persisted:

            def write_default(value: LLMSelection) -> None:
                self._store.set_default(self._work_dir, value)

            writer = write_default
        refreshed = self.materialize(selection, writer=writer)
        return refreshed, self.resolve(refreshed)

    def materialize_session(
        self,
        session_id: str,
        selection: LLMSelection,
    ) -> LLMSelection:
        return self.materialize(
            selection,
            writer=lambda value: self._store.set_session(
                self._work_dir,
                session_id,
                value,
            ),
        )

    def update_codex_auth(self, snapshot: CodexAuthSnapshot) -> bool:
        """Accept a non-stale auth snapshot and notify catalog consumers."""

        if snapshot.operation_id < self._operation_id:
            return False
        self._operation_id = snapshot.operation_id
        self._context = replace(
            self._context,
            codex_snapshot=snapshot,
            codex_initialized=True,
        )
        self._notify()
        return True

    def update_codex_catalog(self, catalog: CodexModelCatalog) -> bool:
        """Accept a non-stale model catalog and notify catalog consumers."""

        if catalog.operation_id < self._operation_id:
            return False
        self._operation_id = catalog.operation_id
        self._context = replace(self._context, codex_catalog=catalog)
        self._notify()
        return True

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            listener()
