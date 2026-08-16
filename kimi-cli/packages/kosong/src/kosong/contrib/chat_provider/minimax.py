"""MiniMax chat providers.

MiniMax exposes two OpenAI-compatible surfaces:

- ``https://api.minimax.io/anthropic`` — an Anthropic Messages API compatible
  endpoint (the default for ``MiniMaxAnthropic``).
- ``https://api.minimax.io/v1`` — a plain OpenAI Chat Completions endpoint
  (``MiniMaxOpenAI``). MiniMax-M3 on this route requires
  ``extra_body.reasoning_split`` and uses ``extra_body.thinking`` to select
  the thinking mode.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any, Self, cast

from openai import omit

from kosong.chat_provider import ChatProvider, RetryableChatProvider, ThinkingEffort
from kosong.chat_provider.openai_common import clamp_thinking_effort
from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

if TYPE_CHECKING:

    def type_check(minimax: MiniMaxAnthropic, openai: MiniMaxOpenAI):
        _chat: ChatProvider = minimax
        _retry: RetryableChatProvider = openai


def _is_minimax_m3(model: str) -> bool:
    """Detect MiniMax-M3 across the alias spellings providers use."""
    normalized = model.strip().lower()
    return normalized in {"minimax-m3", "minimax/minimax-m3"}


class MiniMaxAnthropic(Anthropic):
    """MiniMax via its Anthropic Messages API compatible endpoint.

    >>> chat_provider = MiniMaxAnthropic(model="MiniMax-M2.7", api_key="test")
    >>> chat_provider.name
    'minimax'
    >>> chat_provider.model_name
    'MiniMax-M2.7'
    """

    name = "minimax"

    _DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
    _ENV_VAR = "MINIMAX_API_KEY"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_max_tokens: int = 50000,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = os.getenv(self._ENV_VAR)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            default_max_tokens=default_max_tokens,
            **client_kwargs,
        )


class MiniMaxOpenAI(OpenAILegacy):
    """MiniMax via its OpenAI Chat Completions endpoint.

    MiniMax-M3 keeps thinking inline unless ``extra_body.reasoning_split`` is
    sent, so that field is always added for M3 models. Hermes' effort levels
    are not a MiniMax depth knob on this route — they only select adaptive vs
    disabled thinking.
    """

    name = "minimax-openai"

    _DEFAULT_BASE_URL = "https://api.minimax.io/v1"
    _ENV_VAR = "MINIMAX_API_KEY"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = os.getenv(self._ENV_VAR)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **client_kwargs,
        )

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        """Translate *effort* into MiniMax's M3 thinking controls.

        Only applies to MiniMax-M3 models; everything else falls back to the
        base OpenAI reasoning_effort behavior.
        """
        if not _is_minimax_m3(self.model):
            return super().with_thinking(effort)

        new_self = copy.copy(self)
        clamped = clamp_thinking_effort(effort, set(self._supported_efforts))
        generation_kwargs = copy.deepcopy(new_self._generation_kwargs)
        extra_body: dict[str, Any] = dict(generation_kwargs.get("extra_body") or {})
        extra_body["reasoning_split"] = True
        if clamped == "off":
            extra_body["thinking"] = {"type": "disabled"}
            new_self._reasoning_effort = omit
        else:
            extra_body["thinking"] = {"type": "adaptive"}
            new_self._reasoning_effort = omit
        generation_kwargs["extra_body"] = extra_body
        new_self._generation_kwargs = generation_kwargs
        return new_self

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        extra_body: dict[str, Any] = self._generation_kwargs.get("extra_body") or {}
        thinking = extra_body.get("thinking")
        if isinstance(thinking, dict):
            thinking_dict = cast(dict[str, Any], thinking)
            if thinking_dict.get("type") == "disabled":
                return "off"
            if thinking_dict.get("type") == "adaptive":
                return "medium"
        return super().thinking_effort


__all__ = ["MiniMaxAnthropic", "MiniMaxOpenAI"]
