"""GUI-only adapter that pins a ChatGPT catalog variant at session creation."""

from __future__ import annotations

from dataclasses import replace

from kimi_cli.auth.codex import CodexAuthService
from kimi_cli.llm_codex import CodexProviderRuntime, create_codex_provider

from kimix_gui.llm import (
    PROBLEM_VARIANT_UNAVAILABLE,
    LLMProblem,
    LLMSelectionError,
    reasoning_effort_variants,
)

_KIMIX_METADATA_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


async def create_selected_codex_provider(
    service: CodexAuthService,
    *,
    model_name: str,
    session_id: str,
    reasoning_effort: str | None,
) -> CodexProviderRuntime:
    """Apply an exact GUI Variant without changing Kimix's own thinking semantics."""

    # Keep the Kimix API and its bool-to-effort policy intact. The GUI owns the
    # stronger invariant: after Kimix builds the provider, replace its generation
    # setting with the exact Variant selected in this client.
    runtime = await create_codex_provider(
        service,
        model_name=model_name,
        session_id=session_id,
        thinking=False,
    )
    available = {
        variant.options.reasoning_effort
        for variant in reasoning_effort_variants(
            runtime.model.reasoning_efforts,
            runtime.model.default_reasoning_effort,
        )
    }
    if reasoning_effort is None:
        valid = not available
    else:
        valid = reasoning_effort in available
    if not valid:
        await runtime.lease.close()
        raise LLMSelectionError(LLMProblem(PROBLEM_VARIANT_UNAVAILABLE))

    provider = runtime.provider.with_generation_kwargs(reasoning_effort=reasoning_effort)
    runtime.lease.provider = provider
    provider_dict = dict(runtime.provider_dict)
    if reasoning_effort in _KIMIX_METADATA_EFFORTS:
        provider_dict["thinking_effort"] = reasoning_effort
    else:
        provider_dict.pop("thinking_effort", None)
    return replace(runtime, provider_dict=provider_dict, provider=provider)
