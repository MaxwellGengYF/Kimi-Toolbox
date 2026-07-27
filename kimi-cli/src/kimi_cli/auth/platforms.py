from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from pydantic import BaseModel

from kimi_cli.auth import KIMI_CODE_PLATFORM_ID
from kimi_cli.config import Config, LLMModel, load_config, save_config
from kimi_cli.llm import ModelCapability
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    import aiohttp


class ModelInfo(BaseModel):
    """Model information returned from the API."""

    id: str
    context_length: int
    supports_reasoning: bool
    supports_image_in: bool
    supports_video_in: bool
    display_name: str | None = None

    @property
    def capabilities(self) -> set[ModelCapability]:
        """Derive capabilities from model info."""
        caps: set[ModelCapability] = set()
        if self.supports_reasoning:
            caps.add("thinking")
        # Models with "thinking" in name are always-thinking
        if "thinking" in self.id.lower():
            caps.update(("thinking", "always_thinking"))
        if self.supports_image_in:
            caps.add("image_in")
        if self.supports_video_in:
            caps.add("video_in")
        if self.id.lower().startswith("kimi-k2"):
            caps.update(("thinking", "image_in", "video_in"))
        return caps


class Platform(NamedTuple):
    id: str
    name: str
    base_url: str
    search_url: str | None = None
    fetch_url: str | None = None
    allowed_prefixes: list[str] | None = None


def _kimi_code_base_url() -> str:
    if base_url := os.getenv("KIMI_CODE_BASE_URL"):
        return base_url
    return "https://api.kimi.com/coding/v1"


PLATFORMS: list[Platform] = [
    Platform(
        id=KIMI_CODE_PLATFORM_ID,
        name="Kimi Code",
        base_url=_kimi_code_base_url(),
        search_url=f"{_kimi_code_base_url()}/search",
        fetch_url=f"{_kimi_code_base_url()}/fetch",
    ),
    Platform(
        id="moonshot-cn",
        name="Moonshot AI Open Platform (moonshot.cn)",
        base_url="https://api.moonshot.cn/v1",
        allowed_prefixes=["kimi-k"],
    ),
    Platform(
        id="moonshot-ai",
        name="Moonshot AI Open Platform (moonshot.ai)",
        base_url="https://api.moonshot.ai/v1",
        allowed_prefixes=["kimi-k"],
    ),
]

_PLATFORM_BY_ID = {platform.id: platform for platform in PLATFORMS}
_PLATFORM_BY_NAME = {platform.name: platform for platform in PLATFORMS}


def get_platform_by_id(platform_id: str) -> Platform | None:
    return _PLATFORM_BY_ID.get(platform_id)


def get_platform_by_name(name: str) -> Platform | None:
    return _PLATFORM_BY_NAME.get(name)


MANAGED_PROVIDER_PREFIX = "managed:"


def managed_provider_key(platform_id: str) -> str:
    return f"{MANAGED_PROVIDER_PREFIX}{platform_id}"


def managed_model_key(platform_id: str, model_id: str) -> str:
    return f"{platform_id}/{model_id}"


def parse_managed_provider_key(provider_key: str) -> str | None:
    if not provider_key.startswith(MANAGED_PROVIDER_PREFIX):
        return None
    return provider_key.removeprefix(MANAGED_PROVIDER_PREFIX)


def is_managed_provider_key(provider_key: str) -> bool:
    return provider_key.startswith(MANAGED_PROVIDER_PREFIX)


def get_platform_name_for_provider(provider_key: str) -> str | None:
    platform_id = parse_managed_provider_key(provider_key)
    if not platform_id:
        return None
    platform = get_platform_by_id(platform_id)
    return platform.name if platform else None


def _select_retry_api_keys(
    *,
    attempted_api_key: str,
    resolved_api_key: str,
    fallback_api_key: str,
) -> list[str]:
    result: list[str] = []
    for candidate in (resolved_api_key, fallback_api_key):
        if not candidate or candidate == attempted_api_key or candidate in result:
            continue
        result.append(candidate)
    return result


async def refresh_managed_models(config: Config) -> bool:
    if not config.is_from_default_location:
        return False

    provider = config.provider
    if provider is None or provider.oauth is None:
        return False

    platform = next(
        (p for p in PLATFORMS if p.base_url.rstrip("/") == provider.base_url.rstrip("/")),
        None,
    )
    if platform is None:
        return False

    fallback_api_key = provider.api_key.get_secret_value()
    api_key = fallback_api_key
    oauth_manager = None
    if provider.oauth:
        from kimi_cli.auth.oauth import OAuthManager

        oauth_manager = OAuthManager(config)
        try:
            await oauth_manager.ensure_fresh()
        except Exception as exc:
            logger.warning(
                "Failed to refresh OAuth token before model sync for {platform}: {error}",
                platform=platform.id,
                error=exc,
            )
        api_key = oauth_manager.resolve_api_key(provider.api_key, provider.oauth)
    if not api_key:
        logger.warning(
            "Missing API key for managed provider: {provider}",
            provider=platform.id,
        )
        return False
    import aiohttp

    try:
        models = await list_models(platform, api_key)
    except aiohttp.ClientResponseError as exc:
        if exc.status != 401 or provider.oauth is None or oauth_manager is None:
            logger.error(
                "Failed to refresh models for {platform}: {error}",
                platform=platform.id,
                error=exc,
            )
            return False
        logger.warning(
            "Received 401 while refreshing models for {platform}; attempting token refresh",
            platform=platform.id,
        )
        refresh_exc: Exception | None = None
        try:
            await oauth_manager.ensure_fresh(force=True)
        except Exception as exc2:
            refresh_exc = exc2
            logger.warning(
                "Failed to refresh OAuth token after 401 for {platform}: {error}",
                platform=platform.id,
                error=exc2,
            )

        retry_api_keys = _select_retry_api_keys(
            attempted_api_key=api_key,
            resolved_api_key=oauth_manager.resolve_api_key(provider.api_key, provider.oauth),
            fallback_api_key=fallback_api_key,
        )
        if not retry_api_keys:
            logger.error(
                "Failed to refresh models for {platform}: {error}",
                platform=platform.id,
                error=refresh_exc or exc,
            )
            return False
        retry_exc: Exception | None = None
        for retry_api_key in retry_api_keys:
            try:
                models = await list_models(platform, retry_api_key)
                break
            except Exception as exc3:
                retry_exc = exc3
        else:
            logger.error(
                "Failed to refresh models for {platform}: {error}",
                platform=platform.id,
                error=retry_exc or refresh_exc or exc,
            )
            return False
    except Exception as exc:
        logger.error(
            "Failed to refresh models for {platform}: {error}",
            platform=platform.id,
            error=exc,
        )
        return False

    changed = _apply_models(config, models)
    if changed:
        config_for_save = load_config()
        if _apply_models(config_for_save, models):
            save_config(config_for_save)
    return changed


async def list_models(platform: Platform, api_key: str) -> list[ModelInfo]:
    from kimi_cli.utils.aiohttp import new_client_session

    async with new_client_session() as session:
        models = await _list_models(
            session,
            base_url=platform.base_url,
            api_key=api_key,
        )
    if platform.allowed_prefixes is None:
        return models
    prefixes = tuple(platform.allowed_prefixes)
    return [model for model in models if model.id.startswith(prefixes)]


async def _list_models(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
) -> list[ModelInfo]:
    import aiohttp

    models_url = f"{base_url.rstrip('/')}/models"
    try:
        async with session.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            raise_for_status=True,
        ) as response:
            resp_json = await response.json()
    except aiohttp.ClientError:
        raise

    data = resp_json.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Unexpected models response for {base_url}")

    result: list[ModelInfo] = []
    for item in cast(list[dict[str, Any]], data):
        model_id = item.get("id")
        if not model_id:
            continue
        raw_display_name = item.get("display_name")
        display_name = str(raw_display_name) if raw_display_name else None
        result.append(
            ModelInfo(
                id=str(model_id),
                context_length=int(item.get("context_length") or 0),
                supports_reasoning=bool(item.get("supports_reasoning")),
                supports_image_in=bool(item.get("supports_image_in")),
                supports_video_in=bool(item.get("supports_video_in")),
                display_name=display_name,
            )
        )
    return result


def _select_default_model_and_thinking(
    models: list[ModelInfo],
) -> tuple[ModelInfo, bool] | None:
    """Pick a default model from a managed platform list and whether thinking is enabled."""
    if not models:
        return None
    selected = next(
        (m for m in models if "thinking" in m.capabilities or "always_thinking" in m.capabilities),
        models[0],
    )
    thinking = "thinking" in selected.capabilities or "always_thinking" in selected.capabilities
    return selected, thinking


def _apply_models(config: Config, models: list[ModelInfo]) -> bool:
    changed = False
    if not models:
        return changed

    current_id = config.model.model if config.model is not None else None
    model_by_id = {m.id: m for m in models}
    selected = model_by_id.get(current_id)
    if selected is None:
        selection = _select_default_model_and_thinking(models)
        if selection is None:
            return changed
        selected, thinking = selection
    else:
        thinking = "thinking" in selected.capabilities or "always_thinking" in selected.capabilities

    capabilities = selected.capabilities or None  # empty set -> None

    if config.model is None:
        config.model = LLMModel(
            model=selected.id,
            max_context_size=selected.context_length,
            capabilities=capabilities,
            display_name=selected.display_name,
        )
        changed = True
    else:
        if config.model.model != selected.id:
            config.model.model = selected.id
            changed = True
        if config.model.max_context_size != selected.context_length:
            config.model.max_context_size = selected.context_length
            changed = True
        if config.model.capabilities != capabilities:
            config.model.capabilities = capabilities
            changed = True
        if config.model.display_name != selected.display_name:
            config.model.display_name = selected.display_name
            changed = True

    if config.default_thinking != thinking:
        config.default_thinking = thinking
        changed = True

    return changed
