"""Kimix Codex runtime construction with per-request OAuth credentials."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from kosong.chat_provider.codex import OpenAICodex

from kimi_cli.auth.codex import (
    CODEX_BASE_URL,
    PROBLEM_MODEL_UNAVAILABLE,
    CodexAuthError,
    CodexAuthService,
    CodexModel,
    CodexProblem,
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


class CodexRequestAuth(httpx.Auth):
    """Resolve the latest token for every request and replay one first 401."""

    requires_request_body = True

    def __init__(self, service: CodexAuthService) -> None:
        self._service = service

    async def async_auth_flow(self, request: httpx.Request):
        credentials = await self._service.ensure_credentials()
        self._apply(request, credentials.access_token, credentials.account_id)
        response = yield request
        if response.status_code != 401:
            return
        await response.aread()
        credentials = await self._service.ensure_credentials(
            force_refresh=True,
            failed_access_token=credentials.access_token,
        )
        self._apply(request, credentials.access_token, credentials.account_id)
        retry_response = yield request
        if retry_response.status_code == 401:
            await retry_response.aread()
            # The replay used a freshly refreshed token and was still rejected;
            # drop it so the next request re-authenticates instead of reusing a
            # credential the backend has already refused.
            await self._service.invalidate_credentials(credentials.access_token)

    @staticmethod
    def _apply(request: httpx.Request, access_token: str, account_id: str | None) -> None:
        request.headers["Authorization"] = f"Bearer {access_token}"
        if account_id:
            request.headers["ChatGPT-Account-ID"] = account_id
        else:
            request.headers.pop("ChatGPT-Account-ID", None)
        request.headers["User-Agent"] = get_user_agent()
        request.headers["originator"] = "kimix"


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
    service: CodexAuthService,
    *,
    model_name: str,
    session_id: str,
    thinking: bool,
) -> CodexProviderRuntime:
    """Build the canonical provider without placing credentials in metadata."""

    await service.ensure_credentials()
    catalog = await service.catalog()
    model = next((entry for entry in catalog.models if entry.slug == model_name), None)
    if model is None:
        raise CodexAuthError(CodexProblem(PROBLEM_MODEL_UNAVAILABLE))
    http_client = httpx.AsyncClient(
        auth=CodexRequestAuth(service),
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
