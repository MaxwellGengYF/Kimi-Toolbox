"""Web page content extraction tool (``web_extract``).

Ported from the Hermes project's ``tools/web_tools.web_extract_tool`` and
adapted to the kimi CallableTool2 / ToolResultBuilder conventions. Content is
extracted without LLM summarization: pages within the char budget are returned
whole, larger pages are head+tail truncated with the full text stored under
cache/web and a footer telling the model how to ``read_file`` the omitted
middle. URLs carrying embedded secrets or sensitive query parameters are
blocked, and SSRF checks filter private/internal network targets.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast, override

import orjson
from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.config import Config
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import ToolResultBuilder
from kimi_cli.tools.web.content import (
    convert_base64_images_to_links,
    get_extract_char_limit,
    truncate_with_footer,
)
from kimi_cli.tools.web.url_safety import (
    async_is_safe_url,
    normalize_url_for_request,
    sensitive_query_param_name,
    url_contains_secret,
)
from kimi_cli.utils.logging import logger


class Params(BaseModel):
    urls: list[Any] = Field(
        description=(
            "List of URLs (or search-result objects with a 'url'/'href' field) "
            "to extract content from (max 5)"
        ),
        max_length=5,
    )
    char_limit: int | None = Field(
        default=None,
        ge=2000,
        le=500_000,
        description=(
            "Per-page character budget (default 15000). Larger pages are "
            "head+tail truncated with the full text saved to disk."
        ),
    )


def _web_extract_url(value: Any) -> str | None:
    """Return a usable URL from a model-supplied extract item.

    Models sometimes forward a complete web-search result instead of its URL.
    Accept the two common URL keys, but reject missing/non-string values rather
    than stringifying arbitrary objects into misleading fetch targets.
    """
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        raw: Any = mapping.get("url") or mapping.get("href")
    else:
        raw = value
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    return raw or None


class WebExtract(CallableTool2[Params]):
    name: str = "web_extract"
    description: str = (
        "Extract page content from URLs as markdown/text (no LLM). Within char "
        "budget pages return whole; larger pages head+tail truncate with the "
        "full text saved to disk (read_file the omitted middle). On "
        "failure/timeout use fetch_url."
    )
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        import kimi_cli.tools.web.providers as providers  # noqa: F401  # pyright: ignore[reportUnusedImport]


        # Importing the providers module self-registers the local provider;
        # web_extract always has a usable local backend so it never skips load.
        self._config = config
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        builder = ToolResultBuilder(max_line_length=None)
        try:
            # ── Normalize inputs + block embedded secrets (exfiltration prevention) ──
            # ``url_contains_secret`` checks the raw URL, the unquoted URL, and
            # the normalized URL, so percent-encoded secrets are caught too.
            invalid_urls: dict[int, dict[str, Any]] = {}
            normalized_urls: list[str] = []
            normalized_indices: list[int] = []
            for index, item in enumerate(params.urls):
                raw_url = _web_extract_url(item)
                if raw_url is None:
                    invalid_urls[index] = {
                        "url": "",
                        "title": "",
                        "content": "",
                        "error": (
                            f"Invalid URL item at index {index}: expected a URL string "
                            "or an object with a string 'url' or 'href' field"
                        ),
                    }
                    continue
                normalized_url = normalize_url_for_request(raw_url)
                if url_contains_secret(raw_url):
                    return builder.error(
                        "Blocked: URL contains what appears to be an API key or "
                        "token. Secrets must not be sent in URLs.",
                        brief="Blocked: URL contains a secret",
                    )
                sensitive_key = sensitive_query_param_name(normalized_url)
                if sensitive_key:
                    return builder.error(
                        "Blocked: URL contains a credential-like query parameter "
                        f"({sensitive_key}). Web extract backends are third-party "
                        "readers; remove the sensitive query parameter or use a "
                        "local browser session when this access is explicitly "
                        "required.",
                        brief=f"Blocked: sensitive query parameter {sensitive_key}",
                    )
                normalized_urls.append(normalized_url)
                normalized_indices.append(index)

            # ── SSRF protection — filter private/internal URLs before dispatch ──
            safe_urls: list[str] = []
            safe_indices: list[int] = []
            ssrf_blocked: dict[int, dict[str, Any]] = {}
            for index, url in zip(normalized_indices, normalized_urls, strict=False):
                if not await async_is_safe_url(url):
                    ssrf_blocked[index] = {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": "Blocked: URL targets a private or internal network address",
                    }
                else:
                    safe_urls.append(url)
                    safe_indices.append(index)

            results: list[dict[str, Any]] = []
            if safe_urls:
                from kimi_cli.tools.web.providers import (
                    get_active_extract_provider,
                    get_provider,
                )

                provider = get_active_extract_provider(self._config)
                if provider is None:
                    provider = get_provider("local")
                if provider is None:
                    return builder.error(
                        "No web extract provider configured. Set "
                        "web.extract_backend to 'local'.",
                        brief="No extract provider configured",
                    )

                # Async providers are awaited; sync providers run in a thread
                # so the event loop is not blocked on network I/O.
                provider_any: Any = provider
                if inspect.iscoroutinefunction(provider_any.extract):
                    extracted: Any = await provider_any.extract(safe_urls)
                else:
                    extracted = await asyncio.to_thread(provider_any.extract, safe_urls)
                results = list(extracted) if extracted else []

            # Reconstruct the original input order across invalid, blocked, and
            # provider-processed entries. Providers preserve the order of the
            # safe URL list they receive.
            if invalid_urls or ssrf_blocked:
                safe_results = {
                    index: (
                        results[position]
                        if position < len(results)
                        else {
                            "url": safe_urls[position],
                            "title": "",
                            "content": "",
                            "error": "Extract backend returned no result for this URL",
                        }
                    )
                    for position, index in enumerate(safe_indices)
                }
                by_index = {**safe_results, **ssrf_blocked, **invalid_urls}
                results = [by_index[index] for index in range(len(params.urls))]

            # ── Truncate-and-store: no LLM ──
            effective_char_limit = (
                params.char_limit
                if params.char_limit is not None
                else get_extract_char_limit(self._config)
            )
            for result in results:
                if result.get("error"):
                    continue
                url = result.get("url", "")
                raw_content = result.get("raw_content", "") or result.get("content", "")
                if not raw_content:
                    continue
                clean = convert_base64_images_to_links(raw_content)
                model_text, _truncated = truncate_with_footer(
                    clean, url, effective_char_limit
                )
                result["content"] = model_text

            # ── Trim to minimal fields per entry ──
            trimmed_results = [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "error": r.get("error"),
                }
                for r in results
            ]

            if not trimmed_results:
                return builder.error(
                    "Content was inaccessible or not found",
                    brief="No content extracted",
                )

            result_json = orjson.dumps(
                {"results": trimmed_results}, option=orjson.OPT_INDENT_2
            ).decode()
            builder.write(result_json)
            return builder.ok()
        except Exception as exc:  # noqa: BLE001 — surface unexpected failures
            logger.warning("web_extract error: {error}", error=exc)
            return builder.error(
                f"Error extracting content: {exc}",
                brief="Extraction failed",
            )
