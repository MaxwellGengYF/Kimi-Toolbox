"""OpenAI Codex and Actual Computer chat providers.

Both are thin ``OpenAIResponses`` subclasses: Codex is OpenAI's
``/backend-api/codex`` Responses surface (OAuth-token auth, no env var) and
Actual Computer exposes an OpenAI-compatible Responses API with the base URL
taken from ``ACTUAL_BASE_URL`` (defaulting to the hosted endpoint).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseInputParam

from kosong.chat_provider import ChatProvider, RetryableChatProvider
from kosong.contrib.chat_provider.common import normalize_tool_call_ids
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(codex: OpenAICodex | Actual):
        _: ChatProvider = codex
        _: RetryableChatProvider = codex


def _convert_codex_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": False,
    }


class OpenAICodex(OpenAIResponses):
    """OpenAI Codex — Responses API via the ChatGPT backend.

    Auth is OAuth-external: callers pass the access token as ``api_key``.
    """

    name = "openai-codex"

    _DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
    _ENCRYPTED_REASONING_INCLUDE = ("reasoning.encrypted_content",)

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        session_id: str | None = None,
        own_http_client: bool = True,
        **client_kwargs: Any,
    ):
        """Initialize Codex request identity and HTTP-client ownership.

        Set ``own_http_client=False`` when a longer-lived runtime lease owns the
        injected client. In that mode normal provider copies may call
        :meth:`aclose` without closing their shared transport; the owner must
        eventually call :meth:`shutdown`.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **client_kwargs,
        )
        self._session_id = session_id
        self._own_http_client = own_http_client

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

        # Mirror openai/codex's ResponsesApiRequest. The ChatGPT Codex backend
        # takes stable session identity through ``prompt_cache_key`` and request
        # headers, and rejects public-Responses ``user`` and output-token fields.
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": system_prompt,
            "input": inputs,
            "tools": [_convert_codex_tool(tool) for tool in tools],
            "tool_choice": "auto",
            "parallel_tool_calls": self._generation_kwargs.get("max_tool_calls") != 1,
            "reasoning": reasoning,
            "store": False,
            "stream": self._stream,
            "include": list(self._ENCRYPTED_REASONING_INCLUDE),
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

    async def aclose(self) -> None:
        if self._own_http_client:
            await super().aclose()

    async def shutdown(self) -> None:
        """Close the client even when normal provider copies are non-owning."""
        await super().aclose()


class Actual(OpenAIResponses):
    """Actual Computer — Responses API.

    Hosted inference defaults to ``https://api.actual.inc/v1``; local offline
    inference is selected by setting ``ACTUAL_BASE_URL``.
    """

    name = "actual"

    _DEFAULT_BASE_URL = "https://api.actual.inc/v1"
    _ENV_VARS = ("ACTUAL_API_KEY", "ACTUAL_BASE_URL")

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = os.getenv("ACTUAL_API_KEY")
        resolved_base_url = base_url or os.getenv("ACTUAL_BASE_URL") or self._DEFAULT_BASE_URL
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=resolved_base_url,
            **client_kwargs,
        )


__all__ = ["Actual", "OpenAICodex"]
