"""Provider profile registry.

Declarative profiles for every LLM provider that kosong knows about. The
profiles are ported from the Hermes Agent ``plugins/model-providers`` tree and
describe each provider's identity (``name`` / ``aliases``), auth (``env_vars``),
endpoint (``base_url``), model catalog fallback (``fallback_models``) and the
kosong ``ChatProvider`` class that implements it (``provider_class``).

The registry is intentionally decoupled from the provider implementations:
``create_chat_provider`` lazily imports the class so optional dependencies
(e.g. the AWS SDK for Bedrock) are never loaded unless actually used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

from kosong.chat_provider import ChatProvider, ChatProviderError

__all__ = [
    "ProviderProfile",
    "create_chat_provider",
    "get_provider_profile",
    "list_providers",
    "register_provider",
]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Declarative metadata for one LLM provider.

    Mirrors the declarative fields of Hermes' ``ProviderProfile`` base class
    plus ``provider_class``, the dotted ``module:ClassName`` path kosong uses
    to construct a :class:`~kosong.chat_provider.ChatProvider` for the
    profile.
    """

    # ── Identity ─────────────────────────────────────────────
    name: str
    provider_class: str | None = None
    """Dotted ``module.path:ClassName`` of the implementing ChatProvider.

    ``None`` means kosong has no chat-provider implementation for this
    profile yet (e.g. ``copilot-acp``, which is an external ACP subprocess)."""

    api_mode: str = "chat_completions"
    aliases: tuple[str, ...] = ()

    # ── Human-readable metadata ───────────────────────────────
    display_name: str = ""
    description: str = ""
    signup_url: str = ""

    # ── Auth & endpoints ─────────────────────────────────────
    env_vars: tuple[str, ...] = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    supports_health_check: bool = True

    # ── Vision support ────────────────────────────────────────
    supports_vision: bool = False
    supports_vision_tool_messages: bool = True
    supports_prompt_cache_key: bool = False

    # ── Model catalog ─────────────────────────────────────────
    fallback_models: tuple[str, ...] = ()
    hostname: str = ""

    # ── Client-level quirks ──────────────────────────────────
    default_headers: dict[str, str] = field(default_factory=dict[str, str])

    # ── Request-level quirks ─────────────────────────────────
    fixed_temperature: Any = None
    default_max_tokens: int | None = None
    default_aux_model: str = ""


_PROVIDERS: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}


def register_provider(profile: ProviderProfile) -> None:
    """Register *profile* (and its aliases) in the global registry.

    Registering a name that already exists replaces the previous profile.
    """
    _PROVIDERS[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Return the profile for *name* (or a registered alias), or ``None``."""
    profile = _PROVIDERS.get(name)
    if profile is not None:
        return profile
    canonical = _ALIASES.get(name)
    if canonical is not None:
        return _PROVIDERS.get(canonical)
    return None


def list_providers() -> list[str]:
    """Return the names of all registered providers, in registration order."""
    return list(_PROVIDERS)


def _first_env_var(vars: tuple[str, ...]) -> str | None:
    """Return the value of the first set environment variable in *vars*."""
    for name in vars:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _load_provider_class(profile: ProviderProfile) -> type[ChatProvider] | None:
    """Lazily import the ChatProvider class declared by *profile*."""
    path = profile.provider_class
    if not path:
        return None
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ChatProviderError(
            f"Provider {profile.name!r} has an invalid provider_class {path!r}"
        )
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise ChatProviderError(
            f"Provider {profile.name!r} provider_class {path!r} is not a class"
        )
    return cls


def create_chat_provider(
    name: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> ChatProvider:
    """Create a chat provider for the registered profile *name*.

    Args:
        name: Provider name or registered alias.
        model: The model to use.
        api_key: API key/token. Defaults to the first set ``env_vars`` of the
            profile when omitted.
        base_url: Base URL override. Defaults to the profile's ``base_url``.
        **kwargs: Extra constructor arguments forwarded to the provider class
            (e.g. ``stream``, ``default_max_tokens``).

    Raises:
        ChatProviderError: If the profile is unknown or has no kosong
            chat-provider implementation.
    """
    profile = get_provider_profile(name)
    if profile is None:
        raise ChatProviderError(f"Unknown provider: {name!r}")
    cls = _load_provider_class(profile)
    if cls is None:
        raise ChatProviderError(
            f"Provider {profile.name!r} has no kosong chat provider implementation"
        )
    resolved_api_key = api_key if api_key is not None else _first_env_var(profile.env_vars)
    resolved_base_url = base_url if base_url is not None else (profile.base_url or None)
    # Anthropic-style providers require ``default_max_tokens``; apply the
    # profile's default (or the kosong default) only to those classes.
    if (
        profile.api_mode in {"anthropic_messages", "bedrock_converse"}
        and "default_max_tokens" not in kwargs
    ):
        kwargs["default_max_tokens"] = profile.default_max_tokens or 50000
    return cast(Any, cls)(
        model=model, api_key=resolved_api_key, base_url=resolved_base_url, **kwargs
    )


def _register_all() -> None:
    """Register the built-in provider profiles (ported from Hermes)."""
    # OpenAI-compatible providers (kosong/chat_provider/compat.py)
    register_provider(
        ProviderProfile(
            name="ai-gateway",
            provider_class="kosong.chat_provider.compat:AIGateway",
            aliases=("vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"),
            display_name="Vercel AI Gateway",
            env_vars=("AI_GATEWAY_API_KEY",),
            base_url="https://ai-gateway.vercel.sh/v1",
            default_aux_model="google/gemini-3-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="alibaba",
            provider_class="kosong.chat_provider.compat:Alibaba",
            aliases=("dashscope", "alibaba-cloud", "qwen-dashscope"),
            env_vars=("DASHSCOPE_API_KEY",),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    )
    register_provider(
        ProviderProfile(
            name="alibaba-coding-plan",
            provider_class="kosong.chat_provider.compat:Alibaba",
            aliases=("alibaba_coding", "alibaba-coding", "dashscope-coding"),
            display_name="Alibaba Cloud (Coding Plan)",
            env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"),
            base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        )
    )
    register_provider(
        ProviderProfile(
            name="arcee",
            provider_class="kosong.chat_provider.compat:Arcee",
            aliases=("arcee-ai", "arceeai"),
            env_vars=("ARCEEAI_API_KEY",),
            base_url="https://api.arcee.ai/api/v1",
        )
    )
    register_provider(
        ProviderProfile(
            name="azure-foundry",
            provider_class="kosong.chat_provider.compat:AzureFoundry",
            aliases=("azure", "azure-ai-foundry", "azure-ai"),
            display_name="Azure Foundry",
            env_vars=("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"),
            base_url="",
            auth_type="api_key",
        )
    )
    register_provider(
        ProviderProfile(
            name="copilot",
            provider_class="kosong.chat_provider.compat:Copilot",
            aliases=("github-copilot", "github-models", "github-model", "github"),
            env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
            base_url="https://api.githubcopilot.com",
            auth_type="copilot",
        )
    )
    register_provider(
        ProviderProfile(
            name="custom",
            provider_class="kosong.chat_provider.compat:Custom",
            aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
            env_vars=(),
            base_url="",
            default_max_tokens=65536,
        )
    )
    register_provider(
        ProviderProfile(
            name="deepinfra",
            provider_class="kosong.chat_provider.compat:DeepInfra",
            aliases=("deep-infra", "deepinfra-ai"),
            display_name="DeepInfra",
            env_vars=("DEEPINFRA_API_KEY", "DEEPINFRA_BASE_URL"),
            base_url="https://api.deepinfra.com/v1/openai",
            default_aux_model="deepseek-ai/DeepSeek-V4-Flash",
            fallback_models=(),
        )
    )
    register_provider(
        ProviderProfile(
            name="deepseek",
            provider_class="kosong.chat_provider.compat:DeepSeek",
            aliases=("deepseek-chat",),
            display_name="DeepSeek",
            env_vars=("DEEPSEEK_API_KEY",),
            base_url="https://api.deepseek.com/v1",
            default_aux_model="deepseek-v4-flash",
            fallback_models=("deepseek-v4-pro", "deepseek-v4-flash"),
        )
    )
    register_provider(
        ProviderProfile(
            name="fireworks",
            provider_class="kosong.chat_provider.compat:Fireworks",
            aliases=("fireworks-ai", "fw"),
            display_name="Fireworks AI",
            env_vars=("FIREWORKS_API_KEY",),
            base_url="https://api.fireworks.ai/inference/v1",
            default_aux_model="accounts/fireworks/models/glm-5p2",
            fallback_models=(
                "accounts/fireworks/models/kimi-k2p6",
                "accounts/fireworks/models/glm-5p2",
                "accounts/fireworks/models/kimi-k2p7-code",
            ),
        )
    )
    register_provider(
        ProviderProfile(
            name="gmi",
            provider_class="kosong.chat_provider.compat:GMI",
            aliases=("gmi-cloud", "gmicloud"),
            display_name="GMI Cloud",
            env_vars=("GMI_API_KEY", "GMI_BASE_URL"),
            base_url="https://api.gmi-serving.com/v1",
            default_aux_model="google/gemini-3.1-flash-lite-preview",
            fallback_models=(
                "zai-org/GLM-5.1-FP8",
                "deepseek-ai/DeepSeek-V3.2",
                "moonshotai/Kimi-K2.5",
                "google/gemini-3.1-flash-lite-preview",
                "anthropic/claude-sonnet-5",
                "anthropic/claude-sonnet-4.6",
                "openai/gpt-5.4",
            ),
        )
    )
    register_provider(
        ProviderProfile(
            name="huggingface",
            provider_class="kosong.chat_provider.compat:HuggingFace",
            aliases=("hf", "hugging-face", "huggingface-hub"),
            display_name="HuggingFace",
            env_vars=("HF_TOKEN",),
            base_url="https://router.huggingface.co/v1",
            fallback_models=("Qwen/Qwen3.5-72B-Instruct", "deepseek-ai/DeepSeek-V3.2"),
        )
    )
    register_provider(
        ProviderProfile(
            name="kilocode",
            provider_class="kosong.chat_provider.compat:Kilocode",
            aliases=("kilo-code", "kilo", "kilo-gateway"),
            env_vars=("KILOCODE_API_KEY",),
            base_url="https://api.kilo.ai/api/gateway",
            default_aux_model="google/gemini-3.6-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="kimi-coding",
            provider_class="kosong.chat_provider.compat:KimiCoding",
            aliases=("kimi", "moonshot", "kimi-for-coding"),
            env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
            base_url="https://api.moonshot.ai/v1",
            default_max_tokens=32000,
            default_aux_model="kimi-k2-turbo-preview",
        )
    )
    register_provider(
        ProviderProfile(
            name="nous",
            provider_class="kosong.chat_provider.compat:Nous",
            aliases=("nous-portal", "nousresearch"),
            display_name="Nous Research",
            env_vars=("NOUS_API_KEY",),
            base_url="https://inference-api.nousresearch.com/v1",
            fallback_models=("hermes-3-405b", "hermes-3-70b"),
        )
    )
    register_provider(
        ProviderProfile(
            name="novita",
            provider_class="kosong.chat_provider.compat:Novita",
            aliases=("novita-ai", "novitaai"),
            display_name="NovitaAI",
            env_vars=("NOVITA_API_KEY", "NOVITA_BASE_URL"),
            base_url="https://api.novita.ai/openai/v1",
            default_aux_model="deepseek/deepseek-v3-0324",
            fallback_models=(
                "moonshotai/kimi-k2.5",
                "minimax/minimax-m2.7",
                "zai-org/glm-5",
                "deepseek/deepseek-v3-0324",
                "deepseek/deepseek-r1-0528",
                "qwen/qwen3-235b-a22b-fp8",
            ),
        )
    )
    register_provider(
        ProviderProfile(
            name="nvidia",
            provider_class="kosong.chat_provider.compat:NVIDIA",
            aliases=("nvidia-nim",),
            display_name="NVIDIA NIM",
            env_vars=("NVIDIA_API_KEY",),
            base_url="https://integrate.api.nvidia.com/v1",
            default_max_tokens=16384,
            fallback_models=(
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "nvidia/llama-3.3-70b-instruct",
            ),
        )
    )
    register_provider(
        ProviderProfile(
            name="ollama-cloud",
            provider_class="kosong.chat_provider.compat:OllamaCloud",
            aliases=("ollama_cloud",),
            env_vars=("OLLAMA_API_KEY",),
            base_url="https://ollama.com/v1",
            default_aux_model="nemotron-3-nano:30b",
        )
    )
    register_provider(
        ProviderProfile(
            name="opencode-zen",
            provider_class="kosong.chat_provider.compat:OpenCodeZen",
            aliases=("opencode", "opencode_zen", "zen"),
            env_vars=("OPENCODE_ZEN_API_KEY",),
            base_url="https://opencode.ai/zen/v1",
            default_aux_model="gemini-3-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="openrouter",
            provider_class="kosong.chat_provider.compat:OpenRouter",
            aliases=("or",),
            display_name="OpenRouter",
            env_vars=("OPENROUTER_API_KEY",),
            base_url="https://openrouter.ai/api/v1",
            models_url="https://openrouter.ai/api/v1/models",
            fallback_models=(
                "anthropic/claude-sonnet-4.6",
                "openai/gpt-5.4",
                "deepseek/deepseek-chat",
                "google/gemini-3.6-flash",
                "qwen/qwen3-plus",
            ),
        )
    )
    register_provider(
        ProviderProfile(
            name="qwen-oauth",
            provider_class="kosong.chat_provider.compat:QwenOAuth",
            aliases=("qwen", "qwen-portal", "qwen-cli"),
            env_vars=("QWEN_API_KEY",),
            base_url="https://portal.qwen.ai/v1",
            auth_type="oauth_external",
            default_max_tokens=65536,
        )
    )
    register_provider(
        ProviderProfile(
            name="stepfun",
            provider_class="kosong.chat_provider.compat:StepFun",
            aliases=("step", "stepfun-coding-plan"),
            env_vars=("STEPFUN_API_KEY",),
            base_url="https://api.stepfun.ai/step_plan/v1",
            default_aux_model="step-3.5-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="upstage",
            provider_class="kosong.chat_provider.compat:Upstage",
            aliases=("solar",),
            display_name="Upstage Solar",
            env_vars=("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL"),
            base_url="https://api.upstage.ai/v1",
            fallback_models=("solar-pro3",),
        )
    )
    register_provider(
        ProviderProfile(
            name="xiaomi",
            provider_class="kosong.chat_provider.compat:Xiaomi",
            aliases=("mimo", "xiaomi-mimo"),
            env_vars=("XIAOMI_API_KEY",),
            base_url="https://api.xiaomimimo.com/v1",
            supports_health_check=False,
            supports_vision=True,
            supports_vision_tool_messages=False,
        )
    )
    register_provider(
        ProviderProfile(
            name="zai",
            provider_class="kosong.chat_provider.compat:ZAI",
            aliases=("glm", "z-ai", "z.ai", "zhipu"),
            display_name="Z.AI (GLM)",
            env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
            base_url="https://api.z.ai/api/paas/v4",
            default_aux_model="glm-4.5-flash",
            fallback_models=("glm-5.2", "glm-5", "glm-4-9b"),
        )
    )
    # Anthropic-style / native providers
    register_provider(
        ProviderProfile(
            name="anthropic",
            provider_class="kosong.contrib.chat_provider.anthropic:Anthropic",
            aliases=("claude", "claude-oauth", "claude-code"),
            api_mode="anthropic_messages",
            env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
            base_url="https://api.anthropic.com",
            default_max_tokens=50000,
            default_aux_model="claude-haiku-4-5-20251001",
        )
    )
    register_provider(
        ProviderProfile(
            name="gemini",
            provider_class="kosong.contrib.chat_provider.google_genai:GoogleGenAI",
            aliases=("google", "google-gemini", "google-ai-studio"),
            env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta",
            default_aux_model="gemini-3.6-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="vertex",
            provider_class="kosong.contrib.chat_provider.google_genai:GoogleGenAI",
            aliases=("google-vertex", "vertex-ai", "gcp-vertex"),
            api_mode="chat_completions",
            env_vars=(),
            base_url="https://aiplatform.googleapis.com",
            auth_type="vertex",
            default_aux_model="google/gemini-3.6-flash",
        )
    )
    register_provider(
        ProviderProfile(
            name="xai",
            provider_class="kosong.chat_provider.xai:XAI",
            aliases=("grok", "x-ai", "x.ai"),
            api_mode="codex_responses",
            env_vars=("XAI_API_KEY",),
            base_url="https://api.x.ai/v1",
        )
    )
    register_provider(
        ProviderProfile(
            name="kimi",
            provider_class="kosong.chat_provider.kimi:Kimi",
            aliases=(),
            env_vars=("KIMI_API_KEY",),
            base_url="https://api.moonshot.ai/v1",
            default_max_tokens=32000,
            default_aux_model="kimi-k2-turbo-preview",
        )
    )
    # Special-mode providers
    register_provider(
        ProviderProfile(
            name="minimax",
            provider_class="kosong.contrib.chat_provider.minimax:MiniMaxAnthropic",
            aliases=("mini-max",),
            api_mode="anthropic_messages",
            env_vars=("MINIMAX_API_KEY",),
            base_url="https://api.minimax.io/anthropic",
            default_aux_model="MiniMax-M3",
        )
    )
    register_provider(
        ProviderProfile(
            name="bedrock",
            provider_class="kosong.contrib.chat_provider.bedrock:Bedrock",
            aliases=("aws", "aws-bedrock", "amazon-bedrock", "amazon"),
            api_mode="bedrock_converse",
            env_vars=(),
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            auth_type="aws_sdk",
        )
    )
    register_provider(
        ProviderProfile(
            name="openai-codex",
            provider_class="kosong.chat_provider.codex:OpenAICodex",
            aliases=("codex", "openai_codex"),
            api_mode="codex_responses",
            env_vars=(),
            base_url="https://chatgpt.com/backend-api/codex",
            auth_type="oauth_external",
        )
    )
    register_provider(
        ProviderProfile(
            name="actual",
            provider_class="kosong.chat_provider.codex:Actual",
            aliases=("actual-computer", "actualcomputer", "aci"),
            display_name="Actual Computer",
            env_vars=("ACTUAL_API_KEY", "ACTUAL_BASE_URL"),
            base_url="https://api.actual.inc/v1",
            api_mode="codex_responses",
        )
    )
    # External ACP subprocess — no kosong ChatProvider implementation.
    register_provider(
        ProviderProfile(
            name="copilot-acp",
            provider_class=None,
            aliases=("github-copilot-acp", "copilot-acp-agent"),
            api_mode="chat_completions",
            env_vars=(),
            base_url="acp://copilot",
            auth_type="external_process",
        )
    )


_register_all()
