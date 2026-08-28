"""Provider protocol and the single runtime-override application boundary."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from kimi_cli.auth.codex import (
    AUTH_CONNECTED,
    AUTH_DISCONNECTED,
    CodexAuthSnapshot,
    CodexModelCatalog,
    fallback_catalog,
)
from kosong.chat_provider import ChatProvider, ThinkingEffort

from kimix_gui.llm.domain import LLMModelDescriptor, ProviderTarget
from kimix_gui.llm.parameters import RuntimeOverrides


class RuntimeLease(Protocol):
    """Ownership handle for a provider whose normal ``aclose`` is non-owning."""

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CatalogContext:
    """Refreshable cross-provider catalog state, kept outside the Qt layer."""

    codex_snapshot: CodexAuthSnapshot = field(
        default_factory=lambda: CodexAuthSnapshot(
            operation_id=0,
            state=AUTH_DISCONNECTED,
        )
    )
    codex_catalog: CodexModelCatalog = field(default_factory=fallback_catalog)
    codex_initialized: bool = False

    @property
    def chatgpt_connected(self) -> bool:
        return self.codex_snapshot.state == AUTH_CONNECTED

    @property
    def ready(self) -> bool:
        """Whether startup may decide availability from the current state."""

        return self.codex_initialized and (
            not self.chatgpt_connected
            or self.codex_catalog.operation_id >= self.codex_snapshot.operation_id
        )


@dataclass(frozen=True, slots=True)
class SessionRuntime:
    """Provider-neutral inputs required to open one SDK session."""

    provider_dict: dict[str, Any]
    model: str | None
    provider: ChatProvider | None
    lease: RuntimeLease | None = None


class ProviderKind(Protocol):
    """One target family registered with the GUI's provider registry."""

    id: str
    def owns(self, target: ProviderTarget) -> bool: ...

    def describe(
        self,
        target: ProviderTarget,
        context: CatalogContext,
    ) -> LLMModelDescriptor: ...

    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]: ...

    async def create_runtime(
        self,
        target: ProviderTarget,
        *,
        session_id: str,
        overrides: RuntimeOverrides,
    ) -> SessionRuntime: ...


def apply_overrides(
    runtime: SessionRuntime,
    overrides: RuntimeOverrides,
) -> SessionRuntime:
    """Apply resolved parameters identically for every provider kind.

    The provider object and the Kimix ``provider_dict`` snapshot are changed
    together so request behavior, context accounting, and persisted session
    metadata cannot disagree.
    """

    provider_dict = dict(runtime.provider_dict)
    provider = runtime.provider

    if overrides.thinking_effort is not None:
        if provider is not None:
            provider = provider.with_thinking(
                cast(ThinkingEffort, overrides.thinking_effort)
            )
        provider_dict["thinking_effort"] = overrides.thinking_effort
    if overrides.max_context_size is not None:
        provider_dict["max_context_size"] = overrides.max_context_size
    if overrides.max_tokens is not None:
        provider_dict["max_tokens"] = overrides.max_tokens

    generation_kwargs = dict(overrides.generation_kwargs)
    if overrides.beta_features:
        existing_features = provider_dict.get("beta_features")
        configured_features = (
            tuple(item for item in existing_features if isinstance(item, str))
            if isinstance(existing_features, (list, tuple))
            else ()
        )
        beta_features = list(
            dict.fromkeys((*configured_features, *overrides.beta_features))
        )
        generation_kwargs["beta_features"] = beta_features
        provider_dict["beta_features"] = beta_features
    if generation_kwargs:
        provider_dict.update(generation_kwargs)
        if provider is not None:
            # Generation keyword shapes differ by concrete provider. Resolved
            # overrides are validated by the owning ProviderKind before this
            # single dynamic application boundary.
            provider = cast(
                ChatProvider,
                cast(Any, provider).with_generation_kwargs(**generation_kwargs),
            )

    return replace(runtime, provider_dict=provider_dict, provider=provider)
