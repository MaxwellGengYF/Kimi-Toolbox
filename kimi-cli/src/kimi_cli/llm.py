from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

import orjson
from kosong.chat_provider import ChatProvider
from pydantic import SecretStr

from kimi_cli.constant import get_user_agent
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.auth.oauth import OAuthManager
    from kimi_cli.config import Config, LLMModel, LLMProvider

type ProviderType = Literal[
    "kimi",
    "openai_legacy",
    "openai_responses",
    "anthropic",
    "google_genai",  # for backward-compatibility, equals to `gemini`
    "gemini",
    "vertexai",
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
            if not model.max_context_size and (max_context_size := os.getenv("KIMI_MODEL_MAX_CONTEXT_SIZE")):
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
        case "openai_legacy" | "openai_responses":
            if not provider.base_url and (base_url := os.getenv("OPENAI_BASE_URL")):
                provider.base_url = base_url
            if not provider.api_key.get_secret_value() and (api_key := os.getenv("OPENAI_API_KEY")):
                provider.api_key = SecretStr(api_key)
        case _:
            pass

    return applied


def _kimi_default_headers(provider: LLMProvider, oauth: OAuthManager | None) -> dict[str, str]:
    user_agent = get_user_agent() if provider.type in {"kimi", "_chaos"} else None
    headers = {"User-Agent": user_agent} if user_agent else dict()
    if oauth:
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
    
    assert not thinking_effort or thinking_effort in LEGAL_THINKING_EFFORT, 'thinking_effort must be `off`, `low`, `medium`, `high`, `xhigh` and `max`'
    resolved_api_key = (
        oauth.resolve_api_key(provider.api_key, provider.oauth)
        if oauth and provider.oauth
        else provider.api_key.get_secret_value()
    )

    # Resolve capabilities and final thinking state early so that the kimi
    # provider can force its temperature based on the same decision that later
    # drives with_thinking().
    capabilities = derive_model_capabilities(model)
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
                gen_kwargs["max_tokens"] = max_tokens_int
                gen_kwargs["max_completion_tokens"] = max_tokens_int
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
            if session_id:
                gen_kwargs["user"] = session_id
            if gen_kwargs:
                chat_provider = chat_provider.with_generation_kwargs(**gen_kwargs)
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
        _generation_kwargs = getattr(chat_provider, '_generation_kwargs', None)
    if temperature is not None and _generation_kwargs and 'temperature' in _generation_kwargs:
        _generation_kwargs['temperature'] = float(temperature)
    if top_p is not None and _generation_kwargs and 'top_p' in _generation_kwargs:
        _generation_kwargs['top_p'] = float(top_p)
    if top_k is not None and _generation_kwargs and 'top_k' in _generation_kwargs:
        _generation_kwargs['top_k'] = int(top_k)
    if max_tokens is not None and _generation_kwargs and 'max_tokens' in _generation_kwargs:
        _generation_kwargs['max_tokens'] = int(max_tokens)
    

    # Apply thinking using the pre-computed capability/thinking decision so it
    # matches the temperature forced above for the kimi provider.
    if thinking_on:
        chat_provider = chat_provider.with_thinking(thinking_effort if thinking_effort is not None else 'max')
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
