"""Kimix Codex runtime construction with per-request OAuth credentials."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from kosong.chat_provider.codex import OpenAICodex

from kimi_cli.auth.codex import (
    CODEX_BASE_URL,
    CodexModel,
    CodexRequestAuth,
    resolve_codex_model,
)
from kimi_cli.codex_context import (
    CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS as CODEX_AUTO_COMPACT_FALLBACK_BUFFER_TOKENS,
)
from kimi_cli.codex_context import (
    CODEX_AUTO_COMPACT_PERCENT as CODEX_AUTO_COMPACT_PERCENT,
)
from kimi_cli.codex_context import (
    CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT as CODEX_EFFECTIVE_CONTEXT_WINDOW_PERCENT,
)
from kimi_cli.codex_context import (
    CODEX_TOKEN_BUDGET_REMINDER_TOKENS as CODEX_TOKEN_BUDGET_REMINDER_TOKENS,
)
from kimi_cli.codex_context import (
    codex_loop_control as codex_loop_control,
)
from kimi_cli.constant import get_user_agent

_KIMIX_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _wire_reasoning_effort(effort: str | None) -> str | None:
    # Official Codex exposes Ultra as a client mode: inference still receives
    # ``max`` while Codex separately enables automatic task delegation.
    return "max" if effort == "ultra" else effort


def _effective_reasoning_effort(model: CodexModel, *, thinking: bool) -> str | None:
    if not thinking:
        return _wire_reasoning_effort(model.default_reasoning_effort)
    for effort in reversed(model.reasoning_efforts):
        if effort not in {"none", "off"}:
            return _wire_reasoning_effort(effort)
    return "max"


def _kimix_reasoning_efforts(model: CodexModel) -> tuple[str, ...]:
    efforts: list[str] = []
    for effort in model.reasoning_efforts:
        wire_effort = _wire_reasoning_effort(effort)
        if wire_effort in _KIMIX_REASONING_EFFORTS and wire_effort not in efforts:
            efforts.append(wire_effort)
    return tuple(efforts)


class CodexProviderLease:
    """Close one shared provider exactly once after the top-level session cleanup."""

    def __init__(self, provider: OpenAICodex) -> None:
        self.provider = provider
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await self.provider.shutdown()


@dataclass(frozen=True, slots=True)
class CodexProviderRuntime:
    provider_dict: dict[str, Any]
    provider: OpenAICodex
    lease: CodexProviderLease
    model: CodexModel


async def create_codex_provider(
    *,
    model_name: str,
    session_id: str,
    thinking: bool,
) -> CodexProviderRuntime:
    """Build the canonical provider without placing credentials in metadata."""

    model = await resolve_codex_model(model_name)
    http_client = httpx.AsyncClient(
        auth=CodexRequestAuth(),
        headers={
            "User-Agent": get_user_agent(),
            "originator": "kimix",
        },
    )
    kimix_efforts = _kimix_reasoning_efforts(model)
    provider: OpenAICodex = OpenAICodex(
        session_id=session_id,
        own_http_client=False,
        model=model.slug,
        api_key="oauth-managed",
        base_url=CODEX_BASE_URL,
        http_client=http_client,
        default_headers={
            "User-Agent": get_user_agent(),
            "originator": "kimix",
        },
        max_retries=0,
        supported_efforts=set(kimix_efforts) or None,
    ).with_parallel_tool_calls(enabled=True)
    generation: dict[str, Any] = {}
    reasoning_effort = _effective_reasoning_effort(model, thinking=thinking)
    if reasoning_effort is not None:
        generation["reasoning_effort"] = reasoning_effort
    provider = provider.with_generation_kwargs(**generation)
    capabilities = ["thinking"]
    normalized_modalities = {modality.lower() for modality in model.input_modalities}
    if normalized_modalities & {"image", "images", "vision"}:
        capabilities.append("image_in")
    if normalized_modalities & {"video", "videos"}:
        capabilities.append("video_in")
    provider_dict: dict[str, Any] = {
        "name": model.display_name or model.slug,
        "model": model.slug,
        "max_context_size": model.max_context_size,
        "max_tokens": model.max_tokens,
        "capabilities": capabilities,
        "type": "openai-codex",
        "url": CODEX_BASE_URL,
        "api_key": "oauth-managed",
        "loop_control": codex_loop_control(model.max_context_size),
    }
    if kimix_efforts:
        provider_dict["supported_efforts"] = list(kimix_efforts)
    if reasoning_effort in _KIMIX_REASONING_EFFORTS:
        provider_dict["thinking_effort"] = reasoning_effort
    return CodexProviderRuntime(provider_dict, provider, CodexProviderLease(provider), model)
