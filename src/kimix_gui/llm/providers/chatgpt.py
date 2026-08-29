"""ChatGPT subscription catalog and runtime provider implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import SecretStr

from kimi_cli.auth.codex import (
    CODEX_BASE_URL,
    CODEX_OAUTH_KEY,
    CodexModel,
    kimix_reasoning_effort,
    kimix_reasoning_efforts,
    resolve_codex_model,
)
from kimi_cli.config import LLMModel, LLMProvider, OAuthRef
from kimi_cli.llm import create_llm

from kimix_gui.llm.axes import AXIS_THINKING_EFFORT, ORDER_THINKING_EFFORT
from kimix_gui.llm.domain import (
    PROBLEM_LOGIN_REQUIRED,
    PROBLEM_MODEL_UNAVAILABLE,
    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
    ChatGPTTarget,
    LLMModelDescriptor,
    LLMProblem,
    LLMSelectionError,
    ProviderTarget,
    unavailable_model,
)
from kimix_gui.llm.parameters import ParameterOption, ParameterSpec, RuntimeOverrides
from kimix_gui.llm.providers.base import (
    CatalogContext,
    ProviderKind,
    SessionRuntime,
    apply_overrides,
)

_KNOWN_REASONING_EFFORTS = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)


def normalize_reasoning_effort(value: object) -> str | None:
    """Normalize one catalog effort to the runtime's ThinkingEffort domain."""

    if not isinstance(value, str):
        return None
    effort = value.strip()
    if not effort or any(character.isspace() for character in effort):
        return None
    folded = effort.casefold()
    if folded == "off":
        return None
    if folded == "ultra":
        return "max"
    return folded if folded in _KNOWN_REASONING_EFFORTS else None


def reasoning_effort_parameter(
    efforts: Iterable[object],
    default_effort: object,
) -> ParameterSpec | None:
    """Build the ChatGPT thinking axis from authoritative catalog values."""

    normalized_default = normalize_reasoning_effort(default_effort)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in efforts:
        effort = normalize_reasoning_effort(item)
        if effort is None or effort in seen:
            continue
        seen.add(effort)
        normalized.append(effort)
    if not normalized:
        return None
    return ParameterSpec(
        axis=AXIS_THINKING_EFFORT,
        options=tuple(
            ParameterOption(
                effort,
                RuntimeOverrides(thinking_effort=effort),
                is_default=effort == normalized_default,
            )
            for effort in normalized
        ),
        order=ORDER_THINKING_EFFORT,
    )


def chatgpt_model_descriptor(
    model: CodexModel,
    *,
    connected: bool,
    stale: bool,
) -> LLMModelDescriptor:
    """Build the secret-free descriptor for one subscription model."""

    parameter = reasoning_effort_parameter(
        model.reasoning_efforts,
        model.default_reasoning_effort,
    )
    normalized_modalities = {item.casefold() for item in model.input_modalities}
    capabilities = ["thinking"]
    if normalized_modalities & {"image", "images", "vision"}:
        capabilities.append("image_in")
    if normalized_modalities & {"video", "videos"}:
        capabilities.append("video_in")
    return LLMModelDescriptor(
        target=ChatGPTTarget(model.slug),
        model_id=model.slug,
        display_name=model.display_name,
        provider_type="openai-codex",
        endpoint="ChatGPT subscription",
        credential="Connected account" if connected else "Not connected",
        file_format="Built-in",
        max_context_size=model.max_context_size,
        max_tokens=model.max_tokens,
        capabilities=tuple(capabilities),
        input_modalities=model.input_modalities,
        priority=model.priority,
        parameters=(parameter,) if parameter is not None else (),
        problem=None if connected else LLMProblem(PROBLEM_LOGIN_REQUIRED),
        catalog_stale=stale,
    )


def chatgpt_models(
    context: CatalogContext,
) -> tuple[LLMModelDescriptor, ...]:
    """Convert the complete current subscription catalog."""

    return tuple(
        chatgpt_model_descriptor(
            model,
            connected=context.chatgpt_connected,
            stale=context.codex_catalog.stale,
        )
        for model in context.codex_catalog.models
    )


def _chatgpt_capabilities(model: CodexModel) -> set[str]:
    capabilities: set[str] = {"thinking"}
    modalities = {value.lower() for value in model.input_modalities}
    if modalities & {"image", "images", "vision"}:
        capabilities.add("image_in")
    if modalities & {"video", "videos"}:
        capabilities.add("video_in")
    return capabilities


class ChatGPTProviderKind(ProviderKind):
    """Provider implementation for the managed ChatGPT subscription."""

    id = "chatgpt"
    catalog_managed = True

    def owns(self, target: ProviderTarget) -> bool:
        return isinstance(target, ChatGPTTarget)

    def describe(
        self,
        target: ProviderTarget,
        context: CatalogContext,
    ) -> LLMModelDescriptor:
        if not isinstance(target, ChatGPTTarget):
            raise TypeError(f"Unsupported ChatGPT target: {target!r}")
        model = next(
            (entry for entry in context.codex_catalog.models if entry.slug == target.model),
            None,
        )
        if model is not None:
            return chatgpt_model_descriptor(
                model,
                connected=context.chatgpt_connected,
                stale=context.codex_catalog.stale,
            )
        problem = (
            PROBLEM_MODEL_UNAVAILABLE
            if context.chatgpt_connected
            else PROBLEM_LOGIN_REQUIRED
        )
        return unavailable_model(target, problem)

    def list_models(self, context: CatalogContext) -> tuple[LLMModelDescriptor, ...]:
        return chatgpt_models(context)

    async def create_runtime(
        self,
        target: ProviderTarget,
        *,
        session_id: str,
        overrides: RuntimeOverrides,
    ) -> SessionRuntime:
        if not isinstance(target, ChatGPTTarget):
            raise TypeError(f"Unsupported ChatGPT target: {target!r}")
        catalog_model = await resolve_codex_model(target.model)
        available = kimix_reasoning_efforts(catalog_model.reasoning_efforts)
        available_set = set(available)
        effort = overrides.thinking_effort
        if effort is None:
            effort = kimix_reasoning_effort(catalog_model.default_reasoning_effort)
        elif effort not in available_set:
            raise LLMSelectionError(
                LLMProblem(
                    PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
                    reason=f"{AXIS_THINKING_EFFORT}={effort}",
                )
            )

        capabilities = _chatgpt_capabilities(catalog_model)
        provider_config = LLMProvider(
            type="openai-codex",
            base_url=CODEX_BASE_URL,
            api_key=SecretStr(""),
            oauth=OAuthRef(storage="file", key=CODEX_OAUTH_KEY),
        )
        model_data: dict[str, Any] = {
            "model": catalog_model.slug,
            "display_name": catalog_model.display_name,
            "max_context_size": catalog_model.max_context_size,
            "max_tokens": catalog_model.max_tokens,
            "capabilities": capabilities,
        }
        if available:
            model_data["supported_efforts"] = set(available)
        llm_model = LLMModel(**model_data)
        llm = create_llm(
            provider_config,
            llm_model,
            thinking=True,
            session_id=session_id,
            thinking_effort=effort,
        )
        chat_provider = None if llm is None else llm.chat_provider
        provider_dict: dict[str, Any] = {
            "name": catalog_model.display_name or catalog_model.slug,
            "model": catalog_model.slug,
            "max_context_size": catalog_model.max_context_size,
            "max_tokens": catalog_model.max_tokens,
            "capabilities": list(capabilities),
            "type": "openai-codex",
            "url": CODEX_BASE_URL,
            "api_key": "oauth-managed",
        }
        if available:
            provider_dict["supported_efforts"] = list(available)
        if effort in _KNOWN_REASONING_EFFORTS:
            provider_dict["thinking_effort"] = effort
        runtime = SessionRuntime(
            provider_dict=provider_dict,
            model=target.model,
            provider=chat_provider,
        )
        try:
            return apply_overrides(runtime, overrides)
        except BaseException:
            if chat_provider is not None:
                aclose = getattr(chat_provider, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except BaseException:
                        pass
            raise
