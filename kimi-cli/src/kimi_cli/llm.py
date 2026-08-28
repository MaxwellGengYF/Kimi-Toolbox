from __future__ import annotations

import os
from collections import Counter, deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

import orjson
from kosong.chat_provider import ChatProvider, ChatProviderError, StreamedMessagePart
from kosong.message import TextPart, ThinkPart
from pydantic import SecretStr

from kimi_cli.constant import get_user_agent
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kosong.chat_provider import StreamedMessage, TokenUsage
    from kosong.message import Message
    from kosong.tooling import Tool

    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.config import Config, LLMModel, LLMProvider

type ProviderType = Literal[
    "kimi",
    "xai",
    "openai_legacy",
    "openai_responses",
    "anthropic",
    "google_genai",  # for backward-compatibility, equals to `gemini`
    "gemini",
    "vertexai",
    # Hermes-ported OpenAI-compatible providers
    "ai-gateway",
    "alibaba",
    "alibaba-coding-plan",
    "arcee",
    "azure-foundry",
    "copilot",
    "custom",
    "deepinfra",
    "deepseek",
    "fireworks",
    "gmi",
    "huggingface",
    "kilocode",
    "kimi-coding",
    "nous",
    "novita",
    "nvidia",
    "ollama-cloud",
    "opencode-zen",
    "openrouter",
    "qwen-oauth",
    "stepfun",
    "upstage",
    "xiaomi",
    "zai",
    # Hermes-ported special-mode providers
    "actual",
    "bedrock",
    "minimax",
    "openai-codex",
    "vertex",
    "copilot-acp",
    "_echo",
    "_scripted_echo",
    "_chaos",
]

type ModelCapability = Literal["image_in", "video_in", "thinking", "always_thinking"]
ALL_MODEL_CAPABILITIES: set[ModelCapability] = set(get_args(ModelCapability.__value__))

# Recognized names for the Kimi coding model. Both `kimi-for-coding` and
# `kimi-for-coding-highspeed` are supported; `kimi-code` is kept as an alias.
_KIMI_FOR_CODING_NAMES = {
    "kimi-for-coding",
    "kimi-for-coding-highspeed",
    "kimi-code",
}


@dataclass(slots=True)
class LLM:
    chat_provider: ChatProvider
    max_context_size: int
    capabilities: set[ModelCapability]
    model_config: LLMModel | None = None
    provider_config: LLMProvider | None = None

    @property
    def model_name(self) -> str:
        return self.chat_provider.model_name


class LoopDetectedError(ChatProviderError):
    """Raised when the LLM stream repeats a single character or word too many times."""


class TextLoopDetector:
    """Detect single-character or single-word loops in streamed text.

    The detector keeps O(1) bounded state: it tracks a single-character run,
    a small pending word fragment, and a sliding word window of fixed size.
    It never stores the full response, so memory usage is independent of
    response length.
    """

    def __init__(
        self,
        char_threshold: int = 500,
        word_threshold: int = 500,
        word_window: int = 10,
    ) -> None:
        self.char_threshold = char_threshold
        self.word_threshold = word_threshold
        self.word_window = word_window
        self._char_run_char: str = ""
        self._char_run_length: int = 0
        self._pending: str = ""
        self._window: deque[str] = deque(maxlen=word_window)
        self._counts: Counter = Counter()

    @classmethod
    def from_env(cls) -> TextLoopDetector | None:
        """Create a detector from environment variables, or ``None`` if disabled."""
        enabled = os.getenv("KIMIX_LOOP_DETECTION_ENABLED", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        char_threshold = int(os.getenv("KIMIX_LOOP_CHAR_THRESHOLD", "500"))
        word_threshold = int(os.getenv("KIMIX_LOOP_WORD_THRESHOLD", "500"))
        word_window = int(os.getenv("KIMIX_LOOP_WORD_WINDOW", "10"))
        return cls(char_threshold, word_threshold, word_window)

    def feed(self, text: str) -> bool:
        """Return ``True`` as soon as a loop is detected in *text*."""
        if not text:
            return False

        # 1. Single-character loop: repeated identical non-whitespace characters.
        for char in text:
            if char.isspace():
                self._char_run_char = ""
                self._char_run_length = 0
            elif char == self._char_run_char:
                self._char_run_length += 1
                if self._char_run_length >= self.char_threshold:
                    return True
            else:
                self._char_run_char = char
                self._char_run_length = 1

        # 2. Single-word loop: repeated tokens inside a bounded sliding window.
        combined = self._pending + text
        pending_completed = False
        if text[0].isspace() and self._pending:
            if self._add_word(self._pending):
                return True
            pending_completed = True
            self._pending = ""

        tokens = combined.split()
        if text[-1].isspace():
            num_complete = len(tokens)
            new_pending = ""
        else:
            num_complete = len(tokens) - 1 if tokens else 0
            new_pending = tokens[-1] if tokens else ""

        # If the pending word was just completed, it is the first token and has
        # already been counted above.
        start_idx = 1 if pending_completed and tokens else 0
        for token in tokens[start_idx:num_complete]:
            if self._add_word(token):
                return True

        self._pending = new_pending
        return False

    def _add_word(self, word: str) -> bool:
        if len(self._window) == self.word_window:
            oldest = self._window.popleft()
            self._counts[oldest] -= 1
            if self._counts[oldest] <= 0:
                del self._counts[oldest]
        self._window.append(word)
        self._counts[word] += 1
        return self._counts[word] >= self.word_threshold


class _LoopDetectedStreamedMessage:
    """Wrap a provider stream and raise :class:`LoopDetectedError` on loops."""

    def __init__(self, original: StreamedMessage, detector: TextLoopDetector) -> None:
        self._original = original
        self._detector = detector

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[StreamedMessagePart]:
        async for part in self._original:
            if isinstance(part, TextPart):
                if self._detector.feed(part.text):
                    raise LoopDetectedError(
                        "Repeated single character or word detected in LLM stream."
                    )
            elif (
                isinstance(part, ThinkPart)
                and not part.encrypted
                and self._detector.feed(part.think)
            ):
                raise LoopDetectedError(
                    "Repeated single character or word detected in LLM stream."
                )
            yield part

    @property
    def id(self) -> str | None:
        return self._original.id

    @property
    def usage(self) -> TokenUsage | None:
        return self._original.usage


def _wrap_generate_with_loop_detection(chat_provider: ChatProvider) -> None:
    """Wrap a provider instance so its generated stream is monitored for loops."""
    detector = TextLoopDetector.from_env()
    if detector is None:
        return

    char_threshold = detector.char_threshold
    word_threshold = detector.word_threshold
    word_window = detector.word_window
    original_generate = chat_provider.generate

    async def generate_wrapper(
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> _LoopDetectedStreamedMessage:
        fresh_detector = TextLoopDetector(
            char_threshold=char_threshold,
            word_threshold=word_threshold,
            word_window=word_window,
        )
        stream = await original_generate(system_prompt, tools, history)
        return _LoopDetectedStreamedMessage(stream, fresh_detector)

    chat_provider.generate = generate_wrapper


def model_display_name(model_name: str | None, model: LLMModel | None = None) -> str:
    if model is not None and model.display_name:
        return model.display_name
    if not model_name:
        return ""
    if model_name in _KIMI_FOR_CODING_NAMES:
        # `kimi-code` is an alias; the other names are used verbatim.
        if model_name == "kimi-code":
            return "kimi-for-coding"
        return model_name
    return model_name


def augment_provider_with_env_vars(provider: LLMProvider, model: LLMModel) -> dict[str, str]:
    """Override provider/model settings from environment variables.

    Returns:
        Mapping of environment variables that were applied.
    """
    applied: dict[str, str] = {}

    match provider.type:
        case "kimi":
            if not provider.base_url and (base_url := os.getenv("KIMI_BASE_URL")):
                provider.base_url = base_url
                applied["KIMI_BASE_URL"] = base_url
            if not provider.api_key.get_secret_value() and (api_key := os.getenv("KIMI_API_KEY")):
                provider.api_key = SecretStr(api_key)
                applied["KIMI_API_KEY"] = "******"
            if not model.model and (model_name := os.getenv("KIMI_MODEL_NAME")):
                model.model = model_name
                applied["KIMI_MODEL_NAME"] = model_name
            if not model.max_context_size and (
                max_context_size := os.getenv("KIMI_MODEL_MAX_CONTEXT_SIZE")
            ):
                model.max_context_size = int(max_context_size)
                applied["KIMI_MODEL_MAX_CONTEXT_SIZE"] = max_context_size
            if not model.capabilities and (capabilities := os.getenv("KIMI_MODEL_CAPABILITIES")):
                caps_lower = (cap.strip().lower() for cap in capabilities.split(",") if cap.strip())
                model.capabilities = set(
                    cast(ModelCapability, cap)
                    for cap in caps_lower
                    if cap in get_args(ModelCapability.__value__)
                )
                applied["KIMI_MODEL_CAPABILITIES"] = capabilities
        case "xai":
            if not provider.base_url and (base_url := os.getenv("XAI_BASE_URL")):
                provider.base_url = base_url
                applied["XAI_BASE_URL"] = base_url
            if not provider.api_key.get_secret_value() and (api_key := os.getenv("XAI_API_KEY")):
                provider.api_key = SecretStr(api_key)
                applied["XAI_API_KEY"] = "******"
            if not model.model and (model_name := os.getenv("XAI_MODEL_NAME")):
                model.model = model_name
                applied["XAI_MODEL_NAME"] = model_name
            if not model.max_context_size and (
                max_context_size := os.getenv("XAI_MODEL_MAX_CONTEXT_SIZE")
            ):
                model.max_context_size = int(max_context_size)
                applied["XAI_MODEL_MAX_CONTEXT_SIZE"] = max_context_size
            if not model.capabilities and (capabilities := os.getenv("XAI_MODEL_CAPABILITIES")):
                caps_lower = (cap.strip().lower() for cap in capabilities.split(",") if cap.strip())
                model.capabilities = set(
                    cast(ModelCapability, cap)
                    for cap in caps_lower
                    if cap in get_args(ModelCapability.__value__)
                )
                applied["XAI_MODEL_CAPABILITIES"] = capabilities
        case "openai_legacy" | "openai_responses":
            if not provider.base_url and (base_url := os.getenv("OPENAI_BASE_URL")):
                provider.base_url = base_url
            if not provider.api_key.get_secret_value() and (api_key := os.getenv("OPENAI_API_KEY")):
                provider.api_key = SecretStr(api_key)
        case (
            "ai-gateway"
            | "alibaba"
            | "alibaba-coding-plan"
            | "arcee"
            | "azure-foundry"
            | "copilot"
            | "custom"
            | "deepinfra"
            | "deepseek"
            | "fireworks"
            | "gmi"
            | "huggingface"
            | "kilocode"
            | "kimi-coding"
            | "nous"
            | "novita"
            | "nvidia"
            | "ollama-cloud"
            | "opencode-zen"
            | "openrouter"
            | "qwen-oauth"
            | "stepfun"
            | "upstage"
            | "xiaomi"
            | "zai"
            | "actual"
            | "minimax"
            | "openai-codex"
        ):
            # Hermes-ported providers: default api_key / base_url from the
            # provider profile (env vars first, then the profile defaults).
            from kosong.providers import get_provider_profile

            profile = get_provider_profile(provider.type)
            if profile is not None:
                if not provider.api_key.get_secret_value():
                    for var in profile.env_vars:
                        if var.endswith("_BASE_URL"):
                            continue
                        if value := os.getenv(var):
                            provider.api_key = SecretStr(value)
                            applied[var] = "******"
                            break
                if not provider.base_url:
                    for var in profile.env_vars:
                        if var.endswith("_BASE_URL") and (value := os.getenv(var)):
                            provider.base_url = value
                            applied[var] = value
                            break
                    else:
                        if profile.base_url:
                            provider.base_url = profile.base_url
                            applied["base_url"] = profile.base_url
        case _:
            pass

    return applied


def _kimi_default_headers(provider: LLMProvider, oauth: OAuthManager | None) -> dict[str, str]:
    user_agent = get_user_agent() if provider.type in {"kimi", "_chaos"} else None
    headers = {"User-Agent": user_agent} if user_agent else dict()
    if oauth and provider.type != "openai-codex":
        headers.update(oauth.common_headers())
    if provider.custom_headers:
        headers.update(provider.custom_headers)
    return headers


LEGAL_THINKING_EFFORT = frozenset({"off", "low", "medium", "high", "xhigh", "max"})


# ---------------------------------------------------------------------------
# Cross-provider session contract
# ---------------------------------------------------------------------------
# A `Session` in kimi-cli is shared across providers via client-side history
# replay: the whole conversation is persisted per Session (context.db /
# context.jsonl) and replayed on every turn, so switching providers mid-session
# keeps conversation continuity. `session_id` is NOT what carries the
# conversation — it is only a per-request identity hint, mapped per provider:
#
#   kimi             -> `prompt_cache_key` (Moonshot server-side prompt cache
#                                           key; scoped to the Moonshot
#                                           backend only)
#   anthropic        -> `metadata.user_id` (telemetry / abuse-tracking only;
#                                           Anthropic prompt caching is
#                                           content-addressed, so caches are
#                                           never shared across providers)
#   openai_legacy    -> `user`             (standard stateless identity field)
#   openai_responses -> `user`             (identity field; server-side
#                                           sessions stay disabled)
#
# OpenAI Responses `store=False` / no `previous_response_id` is intentional:
# adopting server-side state would break cross-provider sharing (an
# Anthropic/Kimi history cannot continue from an OpenAI response id) and would
# leak provider-side state into client-persisted sessions.
#
# Identity fields are only set when `session_id` is truthy, so stateless
# requests (user=None / no metadata / no prompt_cache_key) stay byte-identical.
#
# Notes:
# - `user` / `metadata.user_id` are logged server-side; using the session id
#   (uuid hex, 32 chars) there is the intended identity signal. That format is
#   safe for all provider constraints (anthropic user_id <= 256 chars,
#   Moonshot prompt_cache_key rules).
# - Anthropic prompt caching is content-addressed: switching providers
#   mid-session never reuses caches, so the first request on a new provider
#   pays full input cost (cache creation) — expected, not a bug.
# ---------------------------------------------------------------------------
def create_llm(
    provider: LLMProvider,
    model: LLMModel,
    *,
    thinking: bool | None = None,
    session_id: str | None = None,
    oauth: OAuthManager | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    thinking_effort: str | None = None,
) -> LLM | None:
    if provider.type not in {"_echo", "_scripted_echo"} and (
        not provider.base_url or not model.model
    ):
        logger.warning(
            "Cannot create LLM: missing base_url or model (provider_type={provider_type})",
            provider_type=provider.type,
        )
        return None

    assert not thinking_effort or thinking_effort in LEGAL_THINKING_EFFORT, (
        "thinking_effort must be `off`, `low`, `medium`, `high`, `xhigh` and `max`"
    )
    resolved_api_key = (
        oauth.resolve_api_key(provider.api_key, provider.oauth)
        if oauth and provider.oauth
        else provider.api_key.get_secret_value()
    )

    # Resolve capabilities and final thinking state early so that the kimi
    # provider can force its temperature based on the same decision that later
    # drives with_thinking().
    capabilities = derive_model_capabilities(model)
    # A configured non-"off" thinking effort implies thinking should be on when
    # the caller did not make an explicit decision (thinking is None). This is
    # what makes configs like `C:/dev/ds_cmdcode.json` (`thinking_effort: "high"`
    # + `capabilities: ["thinking"]`, no `default_thinking`) emit reasoning
    # blocks even when create_llm is called without a `thinking` argument.
    if thinking is None and thinking_effort not in (None, "off") and "thinking" in capabilities:
        thinking = True
    thinking_on = "always_thinking" in capabilities or (
        thinking is True and "thinking" in capabilities
    )

    match provider.type:
        case "kimi":
            from kosong.chat_provider.kimi import Kimi

            chat_provider = Kimi(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
            )

            gen_kwargs: Kimi.GenerationKwargs = {}
            if session_id:
                gen_kwargs["prompt_cache_key"] = session_id
            # For the kimi provider, temperature is always forced by the final
            # thinking state. config.temperature, KIMI_MODEL_TEMPERATURE, and
            # any explicit temperature argument are intentionally ignored.
            temperature = 1.0 if thinking_on else 0.6
            gen_kwargs["temperature"] = temperature
            if top_p is None:
                top_p = os.getenv("KIMI_MODEL_TOP_P")
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            if max_tokens is None:
                max_tokens = os.getenv("KIMI_MODEL_MAX_TOKENS")
            if max_tokens is None:
                max_tokens = model.max_context_size
            if max_tokens is not None:
                max_tokens_int = int(max_tokens)
                gen_kwargs["max_tokens"] = max_tokens_int
                # ``max_completion_tokens`` is the modern replacement for
                # ``max_tokens`` recommended by OpenAI for reasoning models.
                # It accounts for both reasoning tokens and visible output
                # tokens.  Set both for broad compatibility — the provider's
                # ``clamp_max_tokens()`` ensures neither exceeds the safe cap.
                gen_kwargs["max_completion_tokens"] = max_tokens_int

            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case "openai_legacy":
            from kosong.contrib.chat_provider.openai_legacy import OpenAILegacy

            reasoning_key = (
                provider.reasoning_key
                if provider.reasoning_key is not None
                else "reasoning_content"
            )
            openai_settings = (
                provider.openai_settings.model_dump()
                if provider.openai_settings is not None
                else None
            )
            chat_provider = OpenAILegacy(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                reasoning_key=reasoning_key,
                openai_settings=openai_settings,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)

            gen_kwargs: OpenAILegacy.GenerationKwargs = {}
            if max_tokens is None:
                max_tokens = os.getenv("KIMI_MODEL_MAX_TOKENS")
            if max_tokens is None:
                max_tokens = model.max_context_size
            if max_tokens is not None:
                max_tokens_int = int(max_tokens)
                # OpenAI's Chat Completions API (and many compatible backends)
                # rejects requests that include both ``max_tokens`` and
                # ``max_completion_tokens``.  For reasoning models we need
                # ``max_completion_tokens`` (it counts reasoning tokens); for
                # non-reasoning models we keep the legacy ``max_tokens`` for the
                # broadest compatibility with older endpoints.
                if thinking_on:
                    gen_kwargs["max_completion_tokens"] = max_tokens_int
                else:
                    gen_kwargs["max_tokens"] = max_tokens_int
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is None:
                top_p = os.getenv("KIMI_MODEL_TOP_P")
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            # Per-request session identity: the Chat Completions API has no
            # server-side session; `user` is the standard identity field.
            if session_id:
                gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case "openai_responses":
            from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

            chat_provider = OpenAIResponses(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)

            # Per-request session identity. Server-side state (`store: true`,
            # `previous_response_id`) must stay disabled — see the session
            # contract comment above create_llm.
            gen_kwargs: OpenAIResponses.GenerationKwargs = {}
            if max_tokens is not None:
                # Responses API uses `max_output_tokens`; `max_tokens` is a
                # Chat Completions parameter and must not be sent here.
                gen_kwargs["max_output_tokens"] = int(max_tokens)
            if session_id:
                gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case "xai":
            from kosong.chat_provider.xai import XAI

            chat_provider = XAI(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)

            gen_kwargs: XAI.GenerationKwargs = {}
            if max_tokens is not None:
                gen_kwargs["max_output_tokens"] = int(max_tokens)
            if session_id:
                gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case (
            "ai-gateway"
            | "alibaba"
            | "alibaba-coding-plan"
            | "arcee"
            | "azure-foundry"
            | "copilot"
            | "custom"
            | "deepinfra"
            | "deepseek"
            | "fireworks"
            | "gmi"
            | "huggingface"
            | "kilocode"
            | "kimi-coding"
            | "nous"
            | "novita"
            | "nvidia"
            | "ollama-cloud"
            | "opencode-zen"
            | "openrouter"
            | "qwen-oauth"
            | "stepfun"
            | "upstage"
            | "xiaomi"
            | "zai"
        ):
            # Hermes-ported OpenAI-compatible providers. Each follows the
            # openai_legacy pattern: the Chat Completions API has no server-side
            # session, so `user` is the standard identity field. OpenRouter is
            # the exception: its `session_id` maps to extra_body.session_id for
            # sticky routing (see the OpenRouter provider).
            from kosong.chat_provider.compat import (
                GMI,
                NVIDIA,
                ZAI,
                AIGateway,
                Alibaba,
                Arcee,
                AzureFoundry,
                Copilot,
                Custom,
                DeepInfra,
                DeepSeek,
                Fireworks,
                HuggingFace,
                Kilocode,
                KimiCoding,
                Nous,
                Novita,
                OllamaCloud,
                OpenCodeZen,
                OpenRouter,
                QwenOAuth,
                StepFun,
                Upstage,
                Xiaomi,
            )

            compat_providers = {
                "ai-gateway": AIGateway,
                "alibaba": Alibaba,
                "alibaba-coding-plan": Alibaba,
                "arcee": Arcee,
                "azure-foundry": AzureFoundry,
                "copilot": Copilot,
                "custom": Custom,
                "deepinfra": DeepInfra,
                "deepseek": DeepSeek,
                "fireworks": Fireworks,
                "gmi": GMI,
                "huggingface": HuggingFace,
                "kilocode": Kilocode,
                "kimi-coding": KimiCoding,
                "nous": Nous,
                "novita": Novita,
                "nvidia": NVIDIA,
                "ollama-cloud": OllamaCloud,
                "opencode-zen": OpenCodeZen,
                "openrouter": OpenRouter,
                "qwen-oauth": QwenOAuth,
                "stepfun": StepFun,
                "upstage": Upstage,
                "xiaomi": Xiaomi,
                "zai": ZAI,
            }
            chat_provider = compat_providers[provider.type](
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)

            gen_kwargs: OpenAILegacy.GenerationKwargs = {}
            if max_tokens is None:
                max_tokens = os.getenv("KIMI_MODEL_MAX_TOKENS")
            if max_tokens is None:
                max_tokens = model.max_context_size
            if max_tokens is not None:
                max_tokens_int = int(max_tokens)
                gen_kwargs["max_tokens"] = max_tokens_int
                gen_kwargs["max_completion_tokens"] = max_tokens_int
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is None:
                top_p = os.getenv("KIMI_MODEL_TOP_P")
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            if session_id:
                if provider.type == "openrouter":
                    # OpenRouter sticky routing key lives in extra_body.session_id.
                    gen_kwargs["session_id"] = session_id
                else:
                    gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case "minimax":
            from kosong.contrib.chat_provider.minimax import MiniMaxAnthropic

            chat_provider = MiniMaxAnthropic(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_max_tokens=50000,
                metadata={"user_id": session_id} if session_id else None,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)
        case "bedrock":
            from kosong.contrib.chat_provider.bedrock import Bedrock

            chat_provider = Bedrock(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_max_tokens=50000,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)
        case "openai-codex":
            import httpx
            from kosong.chat_provider.codex import OpenAICodex

            from kimi_cli.auth.codex import CODEX_OAUTH_KEY, extract_chatgpt_account_id
            from kimi_cli.llm_codex import CodexRequestAuth

            codex_headers = {
                "User-Agent": get_user_agent(),
                "originator": "kimix",
            }
            if provider.custom_headers:
                codex_headers.update(provider.custom_headers)
            codex_client_kwargs: dict[str, object] = {}
            codex_api_key = resolved_api_key
            if provider.oauth is not None and provider.oauth.key == CODEX_OAUTH_KEY and oauth:
                codex_service = oauth.codex_service()
                credentials = codex_service.cached_credentials()
                if credentials is not None and credentials.account_id:
                    codex_headers["ChatGPT-Account-ID"] = credentials.account_id
                codex_client_kwargs["http_client"] = httpx.AsyncClient(
                    auth=CodexRequestAuth(codex_service),
                    headers=codex_headers,
                )
                codex_api_key = "oauth-managed"
            elif account_id := extract_chatgpt_account_id(resolved_api_key):
                codex_headers["ChatGPT-Account-ID"] = account_id

            chat_provider = OpenAICodex(
                session_id=session_id,
                own_http_client=True,
                model=model.model,
                base_url=provider.base_url,
                api_key=codex_api_key,
                default_headers=codex_headers,
                max_retries=0,
                supported_efforts=model.supported_efforts,
                **codex_client_kwargs,
            ).with_parallel_tool_calls(enabled=True)

        case "actual":
            from kosong.chat_provider.codex import Actual

            chat_provider = Actual(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)

            gen_kwargs: OpenAIResponses.GenerationKwargs = {}
            if max_tokens is not None:
                gen_kwargs["max_output_tokens"] = int(max_tokens)
            if session_id:
                gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
        case "vertex":
            from kosong.contrib.chat_provider.google_genai import GoogleGenAI

            os.environ.update(provider.env or {})
            chat_provider = GoogleGenAI(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                vertexai=True,
                default_headers=_kimi_default_headers(provider, oauth),
            )
        case "copilot-acp":
            # External ACP subprocess — no kosong ChatProvider implementation.
            logger.warning(
                "Provider type 'copilot-acp' is not supported by kosong; no LLM will be created."
            )
            return None
        case "anthropic":
            from kosong.contrib.chat_provider.anthropic import Anthropic

            chat_provider = Anthropic(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_max_tokens=50000,
                metadata={"user_id": session_id} if session_id else None,
                default_headers=_kimi_default_headers(provider, oauth),
                supported_efforts=model.supported_efforts,
            ).with_parallel_tool_calls(enabled=True)
        case "google_genai" | "gemini":
            from kosong.contrib.chat_provider.google_genai import GoogleGenAI

            chat_provider = GoogleGenAI(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                default_headers=_kimi_default_headers(provider, oauth),
            )
        case "vertexai":
            from kosong.contrib.chat_provider.google_genai import GoogleGenAI

            os.environ.update(provider.env or {})
            chat_provider = GoogleGenAI(
                model=model.model,
                base_url=provider.base_url,
                api_key=resolved_api_key,
                vertexai=True,
                default_headers=_kimi_default_headers(provider, oauth),
            )
        case "_echo":
            from kosong.chat_provider.echo import EchoChatProvider

            chat_provider = EchoChatProvider()
        case "_scripted_echo":
            from kosong.chat_provider.echo import ScriptedEchoChatProvider

            if provider.env:
                os.environ.update(provider.env)
            scripts = _load_scripted_echo_scripts()
            trace_value = os.getenv("KIMI_SCRIPTED_ECHO_TRACE", "")
            trace = trace_value.strip().lower() in {"1", "true", "yes", "on"}
            chat_provider = ScriptedEchoChatProvider(scripts, trace=trace)
        case "_chaos":
            from kosong.chat_provider.chaos import ChaosChatProvider, ChaosConfig
            from kosong.chat_provider.kimi import Kimi

            chat_provider = ChaosChatProvider(
                provider=Kimi(
                    model=model.model,
                    base_url=provider.base_url,
                    api_key=resolved_api_key,
                    default_headers=_kimi_default_headers(provider, oauth),
                ),
                chaos_config=ChaosConfig(
                    error_probability=0.8,
                    error_types=[429, 500, 503],
                ),
            )
    _generation_kwargs = None
    if chat_provider is not None:
        _generation_kwargs = getattr(chat_provider, "_generation_kwargs", None)
    if temperature is not None and _generation_kwargs and "temperature" in _generation_kwargs:
        _generation_kwargs["temperature"] = float(temperature)
    if top_p is not None and _generation_kwargs and "top_p" in _generation_kwargs:
        _generation_kwargs["top_p"] = float(top_p)
    if top_k is not None and _generation_kwargs and "top_k" in _generation_kwargs:
        _generation_kwargs["top_k"] = int(top_k)
    if max_tokens is not None and _generation_kwargs and "max_tokens" in _generation_kwargs:
        _generation_kwargs["max_tokens"] = int(max_tokens)

    # Apply thinking using the pre-computed capability/thinking decision so it
    # matches the temperature forced above for the kimi provider.
    if thinking_on:
        chat_provider = chat_provider.with_thinking(
            thinking_effort if thinking_effort is not None else "max"
        )
    elif thinking is False:
        chat_provider = chat_provider.with_thinking("off")
    # If thinking is None and model doesn't always think, leave as-is (default behavior)

    # Apply Moonshot-specific ``thinking.keep`` (preserved thinking) only when
    # the model is actually in thinking mode; otherwise the API would see a
    # ``thinking.keep`` without an accompanying ``thinking.type`` it honors.
    if thinking_on and provider.type == "kimi":
        from kosong.chat_provider.kimi import Kimi

        if isinstance(chat_provider, Kimi) and (
            thinking_keep := os.getenv("KIMI_MODEL_THINKING_KEEP")
        ):
            chat_provider = chat_provider.with_extra_body({"thinking": {"keep": thinking_keep}})

    # Wrap every real provider with loop detection. Skip the test echo providers
    # so scripted repetition used in tests does not trip the detector.
    if chat_provider is not None and provider.type not in {"_echo", "_scripted_echo"}:
        _wrap_generate_with_loop_detection(chat_provider)

    return LLM(
        chat_provider=chat_provider,
        max_context_size=model.max_context_size,
        capabilities=capabilities,
        model_config=model,
        provider_config=provider,
    )


def clone_llm_with_model_alias(
    llm: LLM | None,
    config: Config,
    model_alias: str | None,
    *,
    session_id: str,
    oauth: OAuthManager | None,
) -> LLM | None:
    if model_alias is None:
        return llm
    model = config.model
    provider = config.provider
    if model is not None:
        model = model.model_copy(update={"model": model_alias})
    else:
        model = LLMModel(model=model_alias, max_context_size=100_000)
    if provider is None:
        provider = LLMProvider(type="kimi", base_url="", api_key=SecretStr(""))
    thinking: bool | None = None
    if llm is not None:
        effort = getattr(llm.chat_provider, "thinking_effort", None)
        if effort is not None:
            thinking = effort != "off"
    return create_llm(
        provider,
        model,
        thinking=thinking,
        session_id=session_id,
        oauth=oauth,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        thinking_effort=config.thinking_effort,
    )


def derive_model_capabilities(model: LLMModel) -> set[ModelCapability]:
    capabilities = set(model.capabilities or ())
    # Models with "thinking" in their name are always-thinking models
    if "thinking" in model.model.lower() or "reason" in model.model.lower():
        capabilities.update(("thinking", "always_thinking"))
    # These models support thinking but can be toggled on/off
    elif model.model in _KIMI_FOR_CODING_NAMES:
        capabilities.update(("thinking", "image_in", "video_in"))
    return capabilities


def _load_scripted_echo_scripts() -> list[str]:
    script_path = os.getenv("KIMI_SCRIPTED_ECHO_SCRIPTS")
    if not script_path:
        raise ValueError("KIMI_SCRIPTED_ECHO_SCRIPTS is required for _scripted_echo.")
    path = Path(script_path).expanduser()
    if not path.exists():
        raise ValueError(f"Scripted echo file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data: object = orjson.loads(text)
    except orjson.JSONDecodeError:
        scripts = [chunk.strip() for chunk in text.split("\n---\n") if chunk.strip()]
        if scripts:
            return scripts
        raise ValueError(
            "Scripted echo file must be a JSON array of strings or a text file "
            "split by '\\n---\\n'."
        ) from None
    if isinstance(data, list):
        data_list = cast(list[object], data)
        if all(isinstance(item, str) for item in data_list):
            return cast(list[str], data_list)
    raise ValueError("Scripted echo JSON must be an array of strings.")
