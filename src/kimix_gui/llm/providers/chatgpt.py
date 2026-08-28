"""ChatGPT subscription catalog and runtime provider implementation."""

from __future__ import annotations

from collections.abc import Iterable

from kimi_cli.auth.codex import CodexAuthService, CodexModel
from kimi_cli.llm_codex import create_codex_provider

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


class ChatGPTProviderKind(ProviderKind):
    """Provider implementation for the managed ChatGPT subscription."""

    id = "chatgpt"
    catalog_managed = True

    def __init__(self, service: CodexAuthService) -> None:
        self._service = service

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
        runtime = await create_codex_provider(
            self._service,
            model_name=target.model,
            session_id=session_id,
            thinking=False,
        )
        effort = overrides.thinking_effort
        if effort is not None:
            parameter = reasoning_effort_parameter(
                runtime.model.reasoning_efforts,
                runtime.model.default_reasoning_effort,
            )
            available = {option.value for option in parameter.options} if parameter else set()
            if effort not in available:
                await runtime.lease.close()
                raise LLMSelectionError(
                    LLMProblem(
                        PROBLEM_PARAMETER_VALUE_UNAVAILABLE,
                        reason=f"{AXIS_THINKING_EFFORT}={effort}",
                    )
                )
        generic = SessionRuntime(
            provider_dict=dict(runtime.provider_dict),
            model=target.model,
            provider=runtime.provider,
            lease=runtime.lease,
        )
        try:
            resolved = apply_overrides(generic, overrides)
            if resolved.provider is not None:
                runtime.lease.provider = resolved.provider
            return resolved
        except BaseException:
            try:
                await runtime.lease.close()
            except BaseException:
                pass
            raise
