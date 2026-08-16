"""Thin OpenAI-compatible provider subclasses.

Each class in this module is a small ``OpenAILegacy`` subclass that pins the
provider's name, default base URL and environment variables, and — where the
provider's API requires it — a provider-specific thinking/reasoning wire
translation.

The defaults mirror the Hermes Agent ``plugins/model-providers`` profiles so
``kosong.providers.create_chat_provider`` can construct a fully-configured
provider from a profile alone.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, Unpack, cast

import httpx
from openai import OpenAIError, omit
from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionMessageParam

from kosong.chat_provider import (
    ChatProvider,
    RetryableChatProvider,
    ThinkingEffort,
)
from kosong.chat_provider.openai_common import (
    clamp_max_tokens,
    clamp_thinking_effort,
    convert_error,
    reasoning_effort_to_thinking_effort,
    tool_to_openai,
)
from kosong.contrib.chat_provider.common import normalize_tool_call_ids
from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy, OpenAILegacyStreamedMessage
from kosong.message import ImageURLPart, Message, TextPart
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(compat: CompatibleOpenAIProvider):
        _: ChatProvider = compat
        _: RetryableChatProvider = compat


def _first_env(vars: tuple[str, ...]) -> str | None:
    """Return the value of the first set environment variable in *vars*."""
    for name in vars:
        value = os.environ.get(name)
        if value:
            return value
    return None


class CompatibleOpenAIProvider(OpenAILegacy):
    """Base class for thin OpenAI-compatible provider wrappers.

    Subclasses only need to set ``name``, ``_DEFAULT_BASE_URL`` and
    ``_ENV_VARS``; the constructor resolves the API key from the environment
    and applies the default base URL.
    """

    _DEFAULT_BASE_URL: str = ""
    _ENV_VARS: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ):
        if api_key is None:
            api_key = _first_env(self._ENV_VARS)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL or None,
            **client_kwargs,
        )


class ThinkingControlledOpenAIProvider(CompatibleOpenAIProvider):
    """OpenAI-compatible provider with ``extra_body.thinking`` control.

    Used by providers (DeepSeek, ZAI) whose OpenAI-compatible endpoint expects
    a top-level ``thinking`` object plus an optional top-level
    ``reasoning_effort``. The base ``OpenAILegacy`` auto-extra-body machinery
    is bypassed in :meth:`generate` so the wire format stays exactly as the
    provider documents it.
    """

    def _thinking_enabled_by_default(self) -> bool:
        """Whether thinking should be enabled when the caller did not ask."""
        return False

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> OpenAILegacyStreamedMessage:
        generation_kwargs: dict[str, Any] = {}
        generation_kwargs.update(self._generation_kwargs)
        clamp_max_tokens(generation_kwargs)
        # ``max_output_tokens`` is a Responses-API parameter; never send it
        # to the Chat Completions endpoint.
        generation_kwargs.pop("max_output_tokens", None)

        reasoning_effort = self._reasoning_effort
        extra_body = dict(generation_kwargs.get("extra_body") or {})
        thinking = extra_body.get("thinking")
        if thinking is None and self._thinking_enabled_by_default():
            thinking = {"type": "enabled"}
            extra_body["thinking"] = thinking
        if (
            isinstance(thinking, dict)
            and cast("dict[str, Any]", thinking).get("type") == "disabled"
        ):
            # Disabled thinking never carries a top-level reasoning_effort.
            reasoning_effort = omit
        if reasoning_effort is not omit:
            extra_body["reasoning_effort"] = reasoning_effort
        if extra_body:
            generation_kwargs["extra_body"] = extra_body

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            self._convert_message(message) for message in normalize_tool_call_ids(history)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=(tool_to_openai(tool) for tool in tools),
                stream=self.stream,
                stream_options={"include_usage": True} if self.stream else omit,
                reasoning_effort=reasoning_effort,
                **generation_kwargs,
            )
            return OpenAILegacyStreamedMessage(response, self._reasoning_key)
        except (OpenAIError, httpx.HTTPError) as e:
            raise convert_error(e) from e


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


class OpenRouter(CompatibleOpenAIProvider):
    """OpenRouter aggregator — OpenAI-compatible Chat Completions endpoint.

    Thinking is translated to OpenRouter's ``extra_body.reasoning`` object
    (``{enabled, effort}``). A ``session_id`` generation kwarg is mapped to
    ``extra_body.session_id`` (OpenRouter's sticky-routing key), and for
    ``x-ai/grok-*`` / ``xai/grok-*`` models the same session id is attached as
    the ``x-grok-conv-id`` header so xAI's prompt cache stays pinned to one
    backend server.
    """

    name = "openrouter"

    _DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    _ENV_VARS = ("OPENROUTER_API_KEY",)

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        clamped = clamp_thinking_effort(effort, set(self._supported_efforts))
        reasoning: dict[str, Any] = {"enabled": clamped != "off"}
        if clamped != "off":
            reasoning["effort"] = clamped
        generation_kwargs = copy.deepcopy(new_self._generation_kwargs)
        extra_body = dict(generation_kwargs.get("extra_body") or {})
        extra_body["reasoning"] = reasoning
        generation_kwargs["extra_body"] = extra_body
        new_self._generation_kwargs = generation_kwargs
        return new_self

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        extra_body: dict[str, Any] = self._generation_kwargs.get("extra_body") or {}
        reasoning = extra_body.get("reasoning")
        if not isinstance(reasoning, dict):
            return None
        reasoning = cast(dict[str, Any], reasoning)
        if reasoning.get("enabled") is False:
            return "off"
        effort = reasoning.get("effort")
        if effort in ("low", "medium", "high", "xhigh", "max"):
            return cast(ThinkingEffort, effort)
        return None

    def with_generation_kwargs(self, **kwargs: Unpack[OpenAILegacy.GenerationKwargs]) -> Self:
        kwargs_dict: dict[str, Any] = dict(kwargs)
        session_id = kwargs_dict.pop("session_id", None)
        if session_id is not None:
            base: dict[str, Any] = dict(self._generation_kwargs.get("extra_body") or {})
            user_extra = kwargs_dict.pop("extra_body", None)
            if isinstance(user_extra, dict):
                base.update(cast("dict[str, Any]", user_extra))
            base["session_id"] = cast(str, session_id)
            kwargs_dict["extra_body"] = base
        return super().with_generation_kwargs(**kwargs_dict)

    def _grok_conv_id(self) -> str | None:
        """Return the sticky session id for xAI Grok models, if applicable."""
        extra_body: dict[str, Any] = self._generation_kwargs.get("extra_body") or {}
        session_id = extra_body.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return None
        model_lower = self.model.lower()
        if model_lower.startswith("x-ai/grok-") or model_lower.startswith("xai/grok-"):
            return session_id
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> OpenAILegacyStreamedMessage:
        new_self = copy.copy(self)
        if conv_id := self._grok_conv_id():
            generation_kwargs = copy.deepcopy(new_self._generation_kwargs)
            extra_headers = dict(generation_kwargs.get("extra_headers") or {})
            extra_headers["x-grok-conv-id"] = conv_id
            generation_kwargs["extra_headers"] = extra_headers
            new_self._generation_kwargs = generation_kwargs
        return await super(OpenRouter, new_self).generate(system_prompt, tools, history)


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


def _deepseek_supports_thinking(model: str) -> bool:
    """DeepSeek thinking-capable model families (V4 and newer).

    Mirrors the Hermes DeepSeek profile: ``deepseek-v4-*`` (and any future
    ``deepseek-vN``) support thinking; ``deepseek-v3-*`` and unknown models do
    not.
    """
    m = model.strip().lower()
    return m.startswith("deepseek-v") and not m.startswith("deepseek-v3")


class DeepSeek(ThinkingControlledOpenAIProvider):
    """DeepSeek — OpenAI-compatible endpoint with ``extra_body.thinking``.

    Wire shape (mirrors the Hermes DeepSeek profile):

        {"reasoning_effort": "<low|medium|high|max>",
         "thinking": {"type": "enabled" | "disabled"}}

    Thinking defaults to enabled for V4+ model families even when the caller
    did not configure it, so ``reasoning_content`` is always produced and the
    notorious "reasoning_content must be passed back" 400 is avoided.
    """

    name = "deepseek"

    _DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    _ENV_VARS = ("DEEPSEEK_API_KEY",)

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
            base_url=base_url,
            reasoning_key="reasoning_content",
            openai_settings={"thinking": True, "reasoning": False, "chat_template_kwargs": False},
            **client_kwargs,
        )

    def _thinking_enabled_by_default(self) -> bool:
        return _deepseek_supports_thinking(self.model)

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        clamped = clamp_thinking_effort(effort, set(self._supported_efforts))
        generation_kwargs = copy.deepcopy(new_self._generation_kwargs)
        extra_body = dict(generation_kwargs.get("extra_body") or {})
        if clamped == "off":
            new_self._reasoning_effort = omit
            extra_body["thinking"] = {"type": "disabled"}
        else:
            # xhigh/ultra collapse to max (DeepSeek's top tier); low/medium/
            # high pass through.
            mapped: str = "max" if clamped in ("xhigh", "max") else clamped
            new_self._reasoning_effort = cast(ReasoningEffort, mapped)
            extra_body["thinking"] = {"type": "enabled"}
        generation_kwargs["extra_body"] = extra_body
        new_self._generation_kwargs = generation_kwargs
        return new_self

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        extra_body: dict[str, Any] = self._generation_kwargs.get("extra_body") or {}
        thinking = extra_body.get("thinking")
        if (
            isinstance(thinking, dict)
            and cast("dict[str, Any]", thinking).get("type") == "disabled"
        ):
            return "off"
        if self._reasoning_effort is not omit:
            return reasoning_effort_to_thinking_effort(
                cast(ReasoningEffort, self._reasoning_effort)
            )
        return None


# ---------------------------------------------------------------------------
# ZAI / GLM
# ---------------------------------------------------------------------------


def _glm_supports_thinking(model: str) -> bool:
    """GLM thinking-capable families: ``glm-4.5`` and later."""
    m = model.strip().lower()
    if not m.startswith("glm-"):
        return False
    rest = m[len("glm-") :]
    parts = rest.split(".", 1)
    try:
        major = int(parts[0])
    except ValueError:
        return False
    minor = int(parts[1].split("-", 1)[0]) if len(parts) > 1 and parts[1][:1].isdigit() else 0
    return (major, minor) >= (4, 5)


def _is_glm_5_2(model: str) -> bool:
    """Detect GLM-5.2 across alias spellings (``glm-5.2``/``glm-5-2``/``glm-5p2``)."""
    m = model.strip().lower()
    return any(token in m for token in ("glm-5.2", "glm-5-2", "glm-5p2"))


class ZAI(ThinkingControlledOpenAIProvider):
    """Z.AI / GLM — OpenAI-compatible endpoint with ``extra_body.thinking``.

    GLM 4.5+ models accept ``thinking: {type: enabled|disabled}``; GLM-5.2
    additionally exposes a top-level ``reasoning_effort`` knob whose only two
    enabled levels are ``high`` and ``max``.
    """

    name = "zai"

    _DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
    _ENV_VARS = ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")

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
            base_url=base_url,
            reasoning_key="reasoning_content",
            openai_settings={"thinking": True, "reasoning": False, "chat_template_kwargs": False},
            **client_kwargs,
        )

    def _thinking_enabled_by_default(self) -> bool:
        # GLM 4.5+ defaults to thinking ON server-side; emit the explicit
        # enabled form so the reasoning_content round-trip contract holds.
        return _glm_supports_thinking(self.model) or _is_glm_5_2(self.model)

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        new_self = copy.copy(self)
        clamped = clamp_thinking_effort(effort, set(self._supported_efforts))
        generation_kwargs = copy.deepcopy(new_self._generation_kwargs)
        extra_body = dict(generation_kwargs.get("extra_body") or {})
        if clamped == "off":
            new_self._reasoning_effort = omit
            extra_body["thinking"] = {"type": "disabled"}
        else:
            extra_body["thinking"] = {"type": "enabled"}
            if _is_glm_5_2(self.model):
                # GLM-5.2 only supports high/max; xhigh/max/ultra → max,
                # everything else enabled → high.
                mapped = "max" if clamped in ("xhigh", "max") else "high"
                new_self._reasoning_effort = cast(ReasoningEffort, mapped)
            else:
                new_self._reasoning_effort = omit
        generation_kwargs["extra_body"] = extra_body
        new_self._generation_kwargs = generation_kwargs
        return new_self

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        extra_body: dict[str, Any] = self._generation_kwargs.get("extra_body") or {}
        thinking = extra_body.get("thinking")
        if (
            isinstance(thinking, dict)
            and cast("dict[str, Any]", thinking).get("type") == "disabled"
        ):
            return "off"
        if self._reasoning_effort is not omit:
            return reasoning_effort_to_thinking_effort(
                cast(ReasoningEffort, self._reasoning_effort)
            )
        return None


# ---------------------------------------------------------------------------
# Xiaomi
# ---------------------------------------------------------------------------


class Xiaomi(CompatibleOpenAIProvider):
    """Xiaomi MiMo — OpenAI-compatible endpoint.

    MiMo rejects list-type tool message content (HTTP 400 "text is not set"),
    so image parts inside tool-result messages are flattened to text before
    the request is built.
    """

    name = "xiaomi"

    _DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
    _ENV_VARS = ("XIAOMI_API_KEY",)

    supports_vision_tool_messages = False
    """MiMo accepts multimodal user messages but rejects list-type tool content."""

    def _convert_message(self, message: Message) -> ChatCompletionMessageParam:
        if message.role == "tool" and any(
            isinstance(part, ImageURLPart) for part in message.content
        ):
            parts: list[str] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append(part.text)
                elif isinstance(part, ImageURLPart):
                    parts.append(f"[image: {part.image_url.url}]")
            message = message.model_copy(update={"content": [TextPart(text="\n".join(parts))]})
        return super()._convert_message(message)


# ---------------------------------------------------------------------------
# Simple OpenAI-compatible providers
# ---------------------------------------------------------------------------


class Alibaba(CompatibleOpenAIProvider):
    """Alibaba Cloud DashScope — OpenAI-compatible endpoint."""

    name = "alibaba"

    _DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    _ENV_VARS = ("DASHSCOPE_API_KEY",)


class QwenOAuth(CompatibleOpenAIProvider):
    """Qwen Portal — OpenAI-compatible endpoint."""

    name = "qwen-oauth"

    _DEFAULT_BASE_URL = "https://portal.qwen.ai/v1"
    _ENV_VARS = ("QWEN_API_KEY",)


class StepFun(CompatibleOpenAIProvider):
    """StepFun — OpenAI-compatible endpoint."""

    name = "stepfun"

    _DEFAULT_BASE_URL = "https://api.stepfun.ai/step_plan/v1"
    _ENV_VARS = ("STEPFUN_API_KEY",)


class Upstage(CompatibleOpenAIProvider):
    """Upstage Solar — OpenAI-compatible endpoint."""

    name = "upstage"

    _DEFAULT_BASE_URL = "https://api.upstage.ai/v1"
    _ENV_VARS = ("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL")


class Nous(CompatibleOpenAIProvider):
    """Nous Portal — OpenAI-compatible endpoint."""

    name = "nous"

    _DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
    _ENV_VARS = ("NOUS_API_KEY",)


class Novita(CompatibleOpenAIProvider):
    """NovitaAI — OpenAI-compatible endpoint."""

    name = "novita"

    _DEFAULT_BASE_URL = "https://api.novita.ai/openai/v1"
    _ENV_VARS = ("NOVITA_API_KEY", "NOVITA_BASE_URL")


class NVIDIA(CompatibleOpenAIProvider):
    """NVIDIA NIM — OpenAI-compatible endpoint."""

    name = "nvidia"

    _DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    _ENV_VARS = ("NVIDIA_API_KEY",)


class Fireworks(CompatibleOpenAIProvider):
    """Fireworks AI — OpenAI-compatible endpoint."""

    name = "fireworks"

    _DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
    _ENV_VARS = ("FIREWORKS_API_KEY",)


class DeepInfra(CompatibleOpenAIProvider):
    """DeepInfra — OpenAI-compatible endpoint."""

    name = "deepinfra"

    _DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
    _ENV_VARS = ("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL")


class HuggingFace(CompatibleOpenAIProvider):
    """HuggingFace Inference API — OpenAI-compatible endpoint."""

    name = "huggingface"

    _DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
    _ENV_VARS = ("HF_TOKEN",)


class GMI(CompatibleOpenAIProvider):
    """GMI Cloud — OpenAI-compatible endpoint."""

    name = "gmi"

    _DEFAULT_BASE_URL = "https://api.gmi-serving.com/v1"
    _ENV_VARS = ("GMI_API_KEY", "GMI_BASE_URL")


class Kilocode(CompatibleOpenAIProvider):
    """Kilo Code — OpenAI-compatible endpoint."""

    name = "kilocode"

    _DEFAULT_BASE_URL = "https://api.kilo.ai/api/gateway"
    _ENV_VARS = ("KILOCODE_API_KEY",)


class Arcee(CompatibleOpenAIProvider):
    """Arcee AI — OpenAI-compatible endpoint."""

    name = "arcee"

    _DEFAULT_BASE_URL = "https://api.arcee.ai/api/v1"
    _ENV_VARS = ("ARCEEAI_API_KEY",)


class AIGateway(CompatibleOpenAIProvider):
    """Vercel AI Gateway — OpenAI-compatible endpoint."""

    name = "ai-gateway"

    _DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"
    _ENV_VARS = ("AI_GATEWAY_API_KEY",)


class OllamaCloud(CompatibleOpenAIProvider):
    """Ollama Cloud — OpenAI-compatible endpoint."""

    name = "ollama-cloud"

    _DEFAULT_BASE_URL = "https://ollama.com/v1"
    _ENV_VARS = ("OLLAMA_API_KEY",)


class Custom(CompatibleOpenAIProvider):
    """Custom / Ollama (local) — user-configured OpenAI-compatible endpoint."""

    name = "custom"

    _DEFAULT_BASE_URL = ""
    _ENV_VARS = ()


class OpenCodeZen(CompatibleOpenAIProvider):
    """OpenCode Zen — OpenAI-compatible endpoint."""

    name = "opencode-zen"

    _DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
    _ENV_VARS = ("OPENCODE_ZEN_API_KEY",)


class AzureFoundry(CompatibleOpenAIProvider):
    """Azure Foundry — per-resource OpenAI-compatible endpoint.

    ``base_url`` is user-supplied (the profile ships without a default).
    """

    name = "azure-foundry"

    _DEFAULT_BASE_URL = ""
    _ENV_VARS = ("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL")


class Copilot(CompatibleOpenAIProvider):
    """GitHub Copilot / GitHub Models — OpenAI-compatible endpoint."""

    name = "copilot"

    _DEFAULT_BASE_URL = "https://api.githubcopilot.com"
    _ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


class KimiCoding(CompatibleOpenAIProvider):
    """Kimi / Moonshot coding — OpenAI-compatible endpoint.

    Uses the Moonshot wire shape: ``extra_body.thinking`` plus a top-level
    ``reasoning_effort``, with ``reasoning_content`` echoed back on assistant
    messages.
    """

    name = "kimi-coding"

    _DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
    _ENV_VARS = ("KIMI_API_KEY", "KIMI_CODING_API_KEY")

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
            base_url=base_url,
            reasoning_key="reasoning_content",
            openai_settings={"thinking": True, "reasoning": False, "chat_template_kwargs": False},
            **client_kwargs,
        )


__all__ = [
    "AIGateway",
    "Alibaba",
    "Arcee",
    "AzureFoundry",
    "CompatibleOpenAIProvider",
    "Copilot",
    "Custom",
    "DeepInfra",
    "DeepSeek",
    "Fireworks",
    "GMI",
    "HuggingFace",
    "Kilocode",
    "KimiCoding",
    "NVIDIA",
    "Nous",
    "Novita",
    "OllamaCloud",
    "OpenCodeZen",
    "OpenRouter",
    "QwenOAuth",
    "StepFun",
    "ThinkingControlledOpenAIProvider",
    "Upstage",
    "Xiaomi",
    "ZAI",
]
