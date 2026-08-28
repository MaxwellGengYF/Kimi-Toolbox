"""Kimix's managed Kosong Codex provider with per-request OAuth credentials."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import openai
from kosong.chat_provider import StreamedMessage
from kosong.chat_provider.codex import OpenAICodex
from kosong.chat_provider.openai_common import convert_error
from kosong.contrib.chat_provider import openai_responses as responses_adapter
from kosong.contrib.chat_provider.common import normalize_tool_call_ids
from kosong.message import Message
from kosong.tooling import Tool
from openai.types.responses import ResponseInputParam

from kimi_cli.auth.codex import (
    CODEX_BASE_URL,
    PROBLEM_MODEL_UNAVAILABLE,
    CodexAuthError,
    CodexAuthService,
    CodexModel,
    CodexProblem,
)
from kimi_cli.constant import get_user_agent

_KIMIX_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_ENCRYPTED_REASONING_INCLUDE = ("reasoning.encrypted_content",)


def _codex_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


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

    @staticmethod
    def _apply(request: httpx.Request, access_token: str, account_id: str | None) -> None:
        request.headers["Authorization"] = f"Bearer {access_token}"
        if account_id:
            request.headers["ChatGPT-Account-ID"] = account_id
        else:
            request.headers.pop("ChatGPT-Account-ID", None)
        request.headers["User-Agent"] = get_user_agent()
        request.headers["originator"] = "kimix"


class ManagedOpenAICodex(OpenAICodex):
    """A shared provider that emits the official Codex Responses request shape."""

    def __init__(
        self,
        *,
        session_id: str | None,
        shared: bool = True,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(**client_kwargs)
        self._session_id = session_id
        self._shared = shared

    def _request_kwargs(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> tuple[dict[str, Any], ResponseInputParam]:
        inputs: ResponseInputParam = []
        for message in normalize_tool_call_ids(history):
            inputs.extend(self._convert_message(message))

        reasoning: dict[str, str] = {"summary": "auto"}
        reasoning_effort = self._generation_kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort

        # Mirror openai/codex's ResponsesApiRequest. In particular, the ChatGPT
        # Codex backend takes the stable session identity through
        # ``prompt_cache_key`` and request headers; its body rejects the legacy
        # public-Responses ``user`` field and explicit output-token limits.
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": system_prompt,
            "input": inputs,
            "tools": [_codex_tool(tool) for tool in tools],
            "tool_choice": "auto",
            "parallel_tool_calls": self._generation_kwargs.get("max_tool_calls") != 1,
            "reasoning": reasoning,
            "store": False,
            "stream": self._stream,
            "include": list(_ENCRYPTED_REASONING_INCLUDE),
        }
        if self._session_id:
            request["prompt_cache_key"] = self._session_id
            request["extra_headers"] = {
                "session-id": self._session_id,
                "thread-id": self._session_id,
                "session_id": self._session_id,
                "x-client-request-id": self._session_id,
            }
        return request, inputs

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> StreamedMessage:
        request, inputs = self._request_kwargs(system_prompt, tools, history)
        try:
            response = await self.client.responses.create(**request)
        except openai.APIStatusError as exc:
            if not (
                responses_adapter._is_invalid_encrypted_content_error(exc)
                and responses_adapter._strip_reasoning_encrypted_content(inputs)
            ):
                raise convert_error(exc) from exc
            try:
                response = await self.client.responses.create(**request)
            except (openai.OpenAIError, httpx.HTTPError) as retry_error:
                raise convert_error(retry_error) from retry_error
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise convert_error(exc) from exc
        return responses_adapter.OpenAIResponsesStreamedMessage(response)

    async def aclose(self) -> None:
        if not self._shared:
            await super().aclose()

    async def shutdown(self) -> None:
        await super().aclose()


class CodexProviderLease:
    """Close one shared provider exactly once after the top-level session cleanup."""

    def __init__(self, provider: ManagedOpenAICodex) -> None:
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
    provider: ManagedOpenAICodex
    lease: CodexProviderLease
    model: CodexModel


async def create_codex_provider(
    service: CodexAuthService,
    *,
    model_name: str,
    session_id: str,
    thinking: bool,
) -> CodexProviderRuntime:
    """Build the custom provider without placing credentials in provider metadata."""

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
    provider: ManagedOpenAICodex = ManagedOpenAICodex(
        session_id=session_id,
        shared=True,
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
    }
    if kimix_efforts:
        provider_dict["supported_efforts"] = list(kimix_efforts)
    if reasoning_effort in _KIMIX_REASONING_EFFORTS:
        provider_dict["thinking_effort"] = reasoning_effort
    return CodexProviderRuntime(provider_dict, provider, CodexProviderLease(provider), model)
