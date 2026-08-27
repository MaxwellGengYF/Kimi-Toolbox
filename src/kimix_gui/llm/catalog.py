"""Translate the ChatGPT model catalog into LLM domain descriptors."""

from __future__ import annotations

from collections.abc import Iterable

from kimi_cli.auth.codex import CodexModel, CodexModelCatalog

from kimix_gui.llm.domain import (
    PROBLEM_LOGIN_REQUIRED,
    PROVIDER_DEFAULT_VARIANT,
    ChatGPTTarget,
    LLMModelDescriptor,
    LLMProblem,
    LLMRuntimeOptions,
    LLMVariantDescriptor,
    reasoning_effort_variant,
)

_KNOWN_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


def normalize_reasoning_effort(value: object) -> str | None:
    """Normalize one catalog effort without constraining future server values."""

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
    if folded in _KNOWN_REASONING_EFFORTS:
        return folded
    return effort


def reasoning_effort_variants(
    efforts: Iterable[object],
    default_effort: object,
) -> tuple[LLMVariantDescriptor, ...]:
    """Preserve catalog order while normalizing and de-duplicating efforts."""

    normalized_default = normalize_reasoning_effort(default_effort)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in efforts:
        effort = normalize_reasoning_effort(item)
        if effort is None or effort in seen:
            continue
        seen.add(effort)
        normalized.append(effort)
    return tuple(
        LLMVariantDescriptor(
            reasoning_effort_variant(effort),
            LLMRuntimeOptions(reasoning_effort=effort),
            is_default=effort == normalized_default,
        )
        for effort in normalized
    )


def chatgpt_model_descriptor(
    model: CodexModel,
    *,
    connected: bool,
    stale: bool,
) -> LLMModelDescriptor:
    """Build the secret-free descriptor for one ChatGPT subscription model."""

    variants = reasoning_effort_variants(
        model.reasoning_efforts,
        model.default_reasoning_effort,
    )
    if not variants and model.default_reasoning_effort is None:
        variants = (LLMVariantDescriptor(PROVIDER_DEFAULT_VARIANT, is_default=True),)
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
        variants=variants,
        problem=None if connected else LLMProblem(PROBLEM_LOGIN_REQUIRED),
        catalog_stale=stale,
    )


def chatgpt_models(
    catalog: CodexModelCatalog,
    *,
    connected: bool,
) -> tuple[LLMModelDescriptor, ...]:
    """Convert a complete catalog without inventing additional model rows."""

    return tuple(
        chatgpt_model_descriptor(model, connected=connected, stale=catalog.stale)
        for model in catalog.models
    )
