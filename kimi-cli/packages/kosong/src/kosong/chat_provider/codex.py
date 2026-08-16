"""OpenAI Codex and Actual Computer chat providers.

Both are thin ``OpenAIResponses`` subclasses: Codex is OpenAI's
``/backend-api/codex`` Responses surface (OAuth-token auth, no env var) and
Actual Computer exposes an OpenAI-compatible Responses API with the base URL
taken from ``ACTUAL_BASE_URL`` (defaulting to the hosted endpoint).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from kosong.chat_provider import ChatProvider, RetryableChatProvider
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

if TYPE_CHECKING:

    def type_check(codex: OpenAICodex | Actual):
        _: ChatProvider = codex
        _: RetryableChatProvider = codex


class OpenAICodex(OpenAIResponses):
    """OpenAI Codex — Responses API via the ChatGPT backend.

    Auth is OAuth-external: callers pass the access token as ``api_key``.
    """

    name = "openai-codex"

    _DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **client_kwargs,
        )


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
