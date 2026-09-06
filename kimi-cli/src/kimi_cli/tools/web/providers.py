"""Pluggable web search/extract providers.

Ported from the Hermes project's ``agent/web_search_provider.py`` +
``agent/web_search_registry.py`` and its ``plugins/web/*`` providers
(``tavily``, ``exa``, ``brave_free``, ``searxng``, ``firecrawl``,
``parallel``, ``xai``). Providers self-register at import time; the ``kimi``
provider is constructed with a ``Config`` + ``Runtime`` and registered by
``SearchWeb.__init__``.

Response shapes (preserved from the Hermes contract):

Search results::

    {"success": True, "data": {"web": [{"title": str, "url": str,
     "description": str, "position": int, ...}]}}

Extract results::

    {"success": True, "data": [{"url": str, "title": str, "content": str,
     "raw_content": str, "error": str | None}, ...]}

On failure (either capability)::

    {"success": False, "error": str}
"""
from __future__ import annotations

import abc
import os
import threading
from typing import TYPE_CHECKING, Any

import orjson
import regex as re
from pydantic import ValidationError

from kimi_cli.constant import USER_AGENT
from kimi_cli.soul.toolset import get_current_tool_call_or_none
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.config import Config
    from kimi_cli.soul.agent import Runtime

__all__ = (
    "WebSearchProvider",
    "register_provider",
    "get_provider",
    "list_providers",
    "get_active_search_provider",
    "get_active_extract_provider",
    "KimiServiceProvider",
    "DDGSProvider",
    "LocalTrafilaturaProvider",
    "TavilyProvider",
    "ExaProvider",
    "BraveFreeProvider",
    "SearxngProvider",
    "FirecrawlProvider",
    "ParallelProvider",
    "XAIProvider",
    "_reset_for_tests",
)


def __getattr__(name: str) -> Any:
    """Lazily resolve ``aiohttp``-based helpers (kept monkeypatchable in tests)."""
    if name == "new_client_session":
        from kimi_cli.utils.aiohttp import new_client_session

        return new_client_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_new_client_session() -> Any:
    """Resolve ``new_client_session`` honoring test patches on either module.

    Legacy tests patch ``kimi_cli.tools.web.search.new_client_session``;
    future tests may patch ``kimi_cli.tools.web.providers.new_client_session``.
    Prefer whichever module attribute has been monkeypatched (i.e. differs from
    the real factory); otherwise return the real factory.
    """
    from kimi_cli.utils.aiohttp import new_client_session as real_factory

    own = globals().get("new_client_session")
    if own is not None and own is not real_factory:
        return own

    from kimi_cli.tools.web import search as _search_mod

    search_factory = getattr(_search_mod, "new_client_session", None)
    if search_factory is not None and search_factory is not real_factory:
        return search_factory
    return real_factory


def _resolve_logger() -> Any:
    """Resolve the search logger honoring the legacy test patch target.

    ``TestSearchWebLogging`` patches ``kimi_cli.tools.web.search.logger`` and
    expects the Kimi provider's HTTP warnings to go through it, so the Kimi
    provider resolves the logger lazily from the search module.
    """
    from kimi_cli.utils.logging import logger as real_logger

    own = globals().get("logger")
    if own is not None and own is not real_logger:
        return own

    from kimi_cli.tools.web import search as _search_mod

    search_logger = getattr(_search_mod, "logger", None)
    if search_logger is not None and search_logger is not real_logger:
        return search_logger
    return real_logger


def _env(name: str) -> str:
    """Return a stripped environment variable value ("" when unset).

    kimi-cli has no config-layer env fallback like Hermes'
    ``get_provider_env``; plain ``os.environ`` is the credential source.
    """
    return os.environ.get(name, "").strip()


async def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> tuple[int, Any]:
    """Execute one HTTP request via the (monkeypatchable) session factory.

    Shared transport for the ported Hermes providers. Returns
    ``(status, body)`` where ``body`` is the parsed JSON payload or ``None``
    when the response is not valid JSON. Network failures
    (``aiohttp.ClientError``, ``TimeoutError``) propagate to the caller.
    """
    import aiohttp

    new_client_session = _resolve_new_client_session()
    client_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None
    async with new_client_session(timeout=client_timeout) as session:
        if method == "GET":
            request = session.get(url, headers=headers, params=params)
        else:
            request = session.post(url, headers=headers, json=payload)
        async with request as response:
            try:
                body: Any = await response.json()
            except Exception:  # noqa: BLE001 — non-JSON body surfaces as None
                body = None
            return response.status, body


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class WebSearchProvider(abc.ABC):
    """Abstract base class for a web search/extract backend.

    Subclasses must implement :meth:`name` and :meth:`is_available`, plus at
    least one of :meth:`search` / :meth:`extract`. The capability flags let the
    registry route each tool call to the right backend.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``web.search_backend`` / ``web.backend``."""

    @property
    def display_name(self) -> str:
        """Human-readable label shown in tool descriptions. Defaults to ``name``."""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Cheap check only (env var present, optional dependency importable,
        instance URL set). Must NOT make network calls — this runs at
        registration/init time.
        """

    def supports_search(self) -> bool:
        """Return True if this provider implements :meth:`search`."""
        return True

    def supports_extract(self) -> bool:
        """Return True if this provider implements :meth:`extract`."""
        return False

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Execute a web search and return the Hermes response shape.

        Override when :meth:`supports_search` returns True. May be ``async def``
        — the dispatcher awaits coroutine results as needed.
        """
        raise NotImplementedError(
            f"{self.name} does not support search (override supports_search)"
        )

    def extract(self, urls: list[str], format: str | None = None) -> Any:
        """Extract content from one or more URLs.

        Override when :meth:`supports_extract` returns True. May be ``async def``.
        Returns a list of per-URL result dicts in input order.
        """
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, WebSearchProvider] = {}
_lock = threading.Lock()


def register_provider(provider: WebSearchProvider) -> None:
    """Register a web search/extract provider.

    Re-registration (same ``name``) overwrites the previous entry and logs a
    debug message so hot-reload scenarios (tests, dev loops) behave predictably.
    """
    if not isinstance(provider, WebSearchProvider):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"register_provider() expects a WebSearchProvider instance, "
            f"got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Web provider .name must be a non-empty string")
    name = raw_name.strip()
    with _lock:
        existing = _registry.get(name)
        _registry[name] = provider
    if existing is not None:
        logger.debug(
            "Web provider '%s' re-registered (was %r)",
            name,
            type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered web provider '%s' (%s)",
            name,
            type(provider).__name__,
        )


def get_provider(name: str) -> WebSearchProvider | None:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    with _lock:
        return _registry.get(name.strip())


def list_providers() -> list[WebSearchProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        items = list(_registry.values())
    return sorted(items, key=lambda p: p.name)


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _registry.clear()


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------

_SEARCH_LEGACY_PREFERENCE = (
    "kimi",
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "xai",
    "ddgs",
    "local",
)
_EXTRACT_LEGACY_PREFERENCE = ("local", "kimi", "firecrawl", "parallel", "tavily", "exa")


def _resolve(configured: str | None, *, capability: str) -> WebSearchProvider | None:
    """Resolve the active provider for a capability ("search" | "extract").

    Resolution rules (in order):

    1. **Explicit config wins, ignoring availability.** ``web.backend`` or
       ``web.{capability}_backend`` names a registered provider that supports
       the capability — return it even if ``is_available()`` is False so the
       caller can surface a precise error instead of silently rerouting.
    2. **Single-provider shortcut.** When exactly one registered provider
       supports the capability AND ``is_available()`` is True, return it.
    3. **Legacy preference walk, filtered by availability.** See
       ``_SEARCH_LEGACY_PREFERENCE`` / ``_EXTRACT_LEGACY_PREFERENCE``.
    """
    with _lock:
        snapshot = dict(_registry)

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    def _is_available_safe(p: WebSearchProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider can't kill resolution."""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
            return False

    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured,
                capability,
            )

    eligible = [
        p for p in snapshot.values() if _capable(p) and _is_available_safe(p)
    ]
    if len(eligible) == 1:
        return eligible[0]

    legacy = _SEARCH_LEGACY_PREFERENCE if capability == "search" else _EXTRACT_LEGACY_PREFERENCE
    for legacy_name in legacy:
        provider = snapshot.get(legacy_name)
        if (
            provider is not None
            and _capable(provider)
            and _is_available_safe(provider)
        ):
            return provider

    return None


def _explicit_config_name(config: Config | None, *, capability: str) -> str | None:
    """Read the configured backend name from a ``Config`` (best-effort)."""
    if config is None:
        try:
            from kimi_cli.config import load_config

            config = load_config()
        except Exception as exc:  # noqa: BLE001 — config is best-effort here
            logger.debug("Could not load config for web provider resolution: %s", exc)
    if config is None:
        return None
    web = config.web
    if capability == "search":
        return web.search_backend or web.backend
    if capability == "extract":
        return web.extract_backend or web.backend
    return None


def get_active_search_provider(config: Config | None = None) -> WebSearchProvider | None:
    """Resolve the currently-active web search provider."""
    explicit = _explicit_config_name(config, capability="search")
    return _resolve(explicit, capability="search")


def get_active_extract_provider(config: Config | None = None) -> WebSearchProvider | None:
    """Resolve the currently-active web extract provider."""
    explicit = _explicit_config_name(config, capability="extract")
    return _resolve(explicit, capability="extract")


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------


class KimiServiceProvider(WebSearchProvider):
    """Search via the Kimi HTTP search service (the legacy ``SearchWeb`` call)."""

    def __init__(self, config: Config, runtime: Runtime):
        self._config = config
        self._runtime = runtime
        self._service_config = config.services.search

    @property
    def name(self) -> str:
        return "kimi"

    @property
    def display_name(self) -> str:
        return "Kimi Search Service"

    def is_available(self) -> bool:
        svc = self._service_config
        return bool(
            svc is not None and svc.base_url and svc.api_key.get_secret_value()
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def _derived_service_url(self, suffix: str) -> str | None:
        """Derive ``<llm provider base_url>/<suffix>`` for the 404 self-heal retry.

        The search/fetch services are served from the same origin as the LLM
        API (``https://api.kimi.com/coding/v1`` -> ``.../search``); when the
        configured service URL is stale (e.g. legacy ``api.moonshot.cn/v1``
        paths, which return 404 ``url.not_found``) the derived URL is the
        current canonical endpoint. Returns None when no provider ``base_url``
        is configured.
        """
        provider = getattr(self._config, "provider", None)
        base_url = getattr(provider, "base_url", None) if provider is not None else None
        if not base_url:
            return None
        return f"{str(base_url).rstrip('/')}/{suffix}"

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        *,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute the legacy Kimi HTTP search call and return Hermes shape.

        ``new_client_session`` and the logger are resolved lazily from the
        ``kimi_cli.tools.web.search`` module so the existing test patches on
        ``search.new_client_session`` / ``search.logger`` keep intercepting.
        """
        import aiohttp

        from kimi_cli.tools.web.search import Response

        new_client_session = _resolve_new_client_session()
        log = _resolve_logger()

        svc = self._service_config
        if svc is None:
            return {
                "success": False,
                "error": (
                    "Search service is not configured. You may want to try other "
                    "methods to search."
                ),
            }

        api_key = self._runtime.oauth.resolve_api_key(svc.api_key, svc.oauth)
        if not svc.base_url or not api_key:
            return {
                "success": False,
                "error": (
                    "Search service is not configured. You may want to try other "
                    "methods to search."
                ),
            }

        tool_call = get_current_tool_call_or_none()
        assert tool_call is not None, "Tool call is expected to be set"

        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}",
            "X-Msh-Tool-Call-Id": tool_call.id,
            **self._runtime.oauth.common_headers(),
            **(svc.custom_headers or {}),
        }
        payload = {
            "text_query": query,
            "limit": limit,
            "enable_page_crawling": include_content,
            "timeout_seconds": 30,
        }

        async def _post_search(url: str) -> list[Any] | int | dict[str, Any]:
            """POST one search request to *url*.

            Returns the parsed ``SearchResult`` list on HTTP 200, the integer
            status code on other HTTP responses, or a failure dict on
            timeout/network/parse errors.
            """
            try:
                # Server-side timeout is 30s, but page crawling can take longer.
                search_timeout = aiohttp.ClientTimeout(
                    total=180, sock_read=90, sock_connect=15
                )
                async with (
                    new_client_session(timeout=search_timeout) as session,
                    session.post(url, headers=headers, json=payload) as response,
                ):
                    if response.status != 200:
                        return response.status
                    try:
                        return Response(**await response.json()).search_results
                    except ValidationError as e:
                        log.warning(
                            "web_search response parse error: {error}, query={query}",
                            error=e,
                            query=query,
                        )
                        return {
                            "success": False,
                            "error": (
                                f"Failed to parse search results. Error: {e}. "
                                "This may indicate that the search service is "
                                "currently unavailable."
                            ),
                        }
            except TimeoutError:
                log.warning("web_search request timed out: query={query}", query=query)
                return {
                    "success": False,
                    "error": (
                        "Search request timed out. The search service may be slow "
                        "or unavailable."
                    ),
                }
            except aiohttp.ClientError as e:
                log.warning(
                    "web_search network error: {error}, query={query}",
                    error=e,
                    query=query,
                )
                return {
                    "success": False,
                    "error": (
                        f"Search request failed: {e}. The search service may be "
                        "unavailable."
                    ),
                }

        outcome = await _post_search(svc.base_url)
        if isinstance(outcome, int) and outcome == 404:
            # Self-heal: a 404 means the configured service URL is stale; retry
            # once against the URL derived from the LLM provider base URL.
            fallback_url = self._derived_service_url("search")
            if fallback_url is not None and fallback_url != svc.base_url:
                log.warning(
                    "web_search service returned 404 at {url}; "
                    "retrying derived URL {fallback_url}",
                    url=svc.base_url,
                    fallback_url=fallback_url,
                )
                outcome = await _post_search(fallback_url)

        if isinstance(outcome, dict):
            return outcome
        if isinstance(outcome, int):
            log.warning(
                "web_search HTTP error: status={status}, query={query}",
                status=outcome,
                query=query,
            )
            return {
                "success": False,
                "error": (
                    f"Failed to search. Status: {outcome}. "
                    "This may indicate that the search service is "
                    "currently unavailable."
                ),
            }
        results = outcome

        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "description": r.snippet,
                        "position": i + 1,
                        "content": r.content,
                        "date": r.date,
                        "site_name": r.site_name,
                        "icon": r.icon,
                        "mime": r.mime,
                    }
                    for i, r in enumerate(results)
                ]
            },
        }


class DDGSProvider(WebSearchProvider):
    """DuckDuckGo search via the optional ``ddgs`` package (no API key)."""

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        """Return True when the ``ddgs`` package is importable."""
        try:
            import ddgs  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a DuckDuckGo search and return normalized Hermes results.

        ``include_content`` is accepted for dispatcher uniformity and ignored —
        ddgs returns snippets only.
        """
        try:
            import ddgs  # pyright: ignore[reportMissingImports, reportUnusedImport]  # noqa: F401
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
            }

        safe_limit = max(1, int(limit))
        try:
            from ddgs import (  # pyright: ignore[reportMissingImports]
                DDGS,  # pyright: ignore[reportUnknownVariableType]
            )

            with DDGS(timeout=10) as client:  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                hits: list[Any] = list(
                    client.text(query, max_results=safe_limit)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                )
        except Exception as exc:  # noqa: BLE001 — ddgs raises its own exceptions
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        web_results: list[dict[str, Any]] = []
        for i, hit in enumerate(hits):
            if i >= safe_limit:
                break
            web_results.append(
                {
                    "title": str(hit.get("title", "")),
                    "url": str(hit.get("href") or hit.get("url") or ""),
                    "description": str(hit.get("body", "")),
                    "position": i + 1,
                }
            )
        return {"success": True, "data": {"web": web_results}}


class LocalTrafilaturaProvider(WebSearchProvider):
    """Local content extraction via trafilatura (hard dependency of kimi-cli)."""

    _BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    @property
    def name(self) -> str:
        return "local"

    @property
    def display_name(self) -> str:
        return "Local Trafilatura"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(
        self,
        urls: list[str],
        format: str | None = None,  # noqa: A002 — kept for dispatcher parity
    ) -> list[dict[str, Any]]:
        """Fetch each URL and extract main text via trafilatura.

        Returns one dict per URL in input order; per-URL failures set ``error``
        instead of raising. ``format`` is accepted for dispatcher uniformity and
        ignored (txt output).
        """
        import aiohttp
        import trafilatura

        new_client_session = _resolve_new_client_session()
        timeout = aiohttp.ClientTimeout(total=30, sock_read=30, sock_connect=15)
        results: list[dict[str, Any]] = []

        async with new_client_session(timeout=timeout) as session:
            for url in urls:
                try:
                    async with session.get(
                        url, headers={"User-Agent": self._BROWSER_UA}
                    ) as response:
                        if response.status >= 400:
                            results.append(
                                {
                                    "url": url,
                                    "title": "",
                                    "content": "",
                                    "raw_content": "",
                                    "error": f"HTTP {response.status} error",
                                }
                            )
                            continue
                        resp_text = await response.text()

                    extracted = (
                        trafilatura.extract(
                            resp_text,
                            include_comments=True,
                            include_tables=True,
                            include_formatting=False,
                            output_format="txt",
                            with_metadata=True,
                        )
                        or ""
                    )
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": extracted,
                            "raw_content": extracted,
                            "error": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — per-URL failure
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": str(exc),
                        }
                    )

        return results


# ---------------------------------------------------------------------------
# Hermes-ported HTTP providers (``plugins/web/*``)
# ---------------------------------------------------------------------------
#
# All of these are pure aiohttp + stdlib (no vendor SDKs) and self-register at
# module import time; availability is env-var based and checked at resolution.


class TavilyProvider(WebSearchProvider):
    """Tavily search + extract (ported from Hermes ``plugins/web/tavily``).

    Env vars: ``TAVILY_API_KEY`` (required), ``TAVILY_BASE_URL`` (optional
    override of ``https://api.tavily.com``).
    """

    _DEFAULT_BASE_URL = "https://api.tavily.com"

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def display_name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """Return True when ``TAVILY_API_KEY`` is set to a non-empty value."""
        return bool(_env("TAVILY_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def _tavily_request(
        self, endpoint: str, payload: dict[str, Any]
    ) -> tuple[int, Any]:
        """POST to the Tavily API and return ``(status, parsed_json)``.

        Raises ``ValueError`` when ``TAVILY_API_KEY`` is unset; network
        failures propagate for the caller to surface.
        """
        api_key = _env("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable not set. "
                "Get your API key at https://app.tavily.com/home"
            )
        base_url = (_env("TAVILY_BASE_URL") or self._DEFAULT_BASE_URL).rstrip("/")
        body = dict(payload)  # don't mutate the caller's dict
        body["api_key"] = api_key
        return await _http_json(
            "POST", f"{base_url}/{endpoint.lstrip('/')}", payload=body, timeout=60
        )

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a Tavily ``/search`` and return normalized Hermes results.

        ``include_content`` is accepted for dispatcher uniformity and ignored —
        raw content comes from ``extract``.
        """
        del include_content
        log = _resolve_logger()
        try:
            status, data = await self._tavily_request(
                "search",
                {
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — aiohttp errors + timeouts
            log.warning("Tavily search error: {error}", error=exc)
            return {"success": False, "error": f"Tavily search failed: {exc}"}
        if status != 200:
            log.warning("Tavily search HTTP error: status={status}", status=status)
            return {"success": False, "error": f"Tavily search failed: HTTP {status}"}
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Tavily search failed: response was not valid JSON",
            }

        web_results: list[dict[str, Any]] = []
        for i, result in enumerate(data.get("results", []) or []):
            web_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("content", ""),
                    "position": i + 1,
                }
            )
        return {"success": True, "data": {"web": web_results}}

    async def extract(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        urls: list[str],
        format: str | None = None,  # noqa: A002 — kept for dispatcher parity
    ) -> list[dict[str, Any]]:
        """Extract content from one or more URLs via Tavily ``/extract``.

        Per-URL failures (``failed_results`` / ``failed_urls``) become result
        entries with an ``error`` field rather than raising.
        """
        del format  # Tavily extract has no format knob
        log = _resolve_logger()
        try:
            status, data = await self._tavily_request(
                "extract",
                {
                    "urls": urls,
                    "include_images": False,
                },
            )
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": str(exc)}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Tavily extract error: {error}", error=exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Tavily extract failed: {exc}",
                }
                for u in urls
            ]
        if status != 200 or not isinstance(data, dict):
            detail = (
                f"HTTP {status}"
                if status != 200
                else "response was not valid JSON"
            )
            log.warning("Tavily extract failed: {detail}", detail=detail)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Tavily extract failed: {detail}",
                }
                for u in urls
            ]

        fallback_url = urls[0] if urls else ""
        documents: list[dict[str, Any]] = []
        for result in data.get("results", []):
            url = result.get("url", fallback_url)
            raw = result.get("raw_content", "") or result.get("content", "")
            documents.append(
                {
                    "url": url,
                    "title": result.get("title", ""),
                    "content": raw,
                    "raw_content": raw,
                    "error": None,
                    "metadata": {"sourceURL": url, "title": result.get("title", "")},
                }
            )
        for fail in data.get("failed_results", []):
            documents.append(
                {
                    "url": fail.get("url", fallback_url),
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": fail.get("error", "extraction failed"),
                    "metadata": {"sourceURL": fail.get("url", fallback_url)},
                }
            )
        for fail_url in data.get("failed_urls", []):
            url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
            documents.append(
                {
                    "url": url_str,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": "extraction failed",
                    "metadata": {"sourceURL": url_str},
                }
            )
        return documents


_EXA_BASE_URL = "https://api.exa.ai"


class ExaProvider(WebSearchProvider):
    """Exa search + extract (ported from Hermes ``plugins/web/exa``).

    Hermes used the ``exa-py`` SDK; adapted here to the plain REST API
    (``POST /search`` / ``POST /contents`` with the ``x-api-key`` header) to
    avoid a new hard dependency.

    Env var: ``EXA_API_KEY`` (https://exa.ai).
    """

    @property
    def name(self) -> str:
        return "exa"

    @property
    def display_name(self) -> str:
        return "Exa"

    def is_available(self) -> bool:
        """Return True when ``EXA_API_KEY`` is set to a non-empty value."""
        return bool(_env("EXA_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    @staticmethod
    def _headers() -> dict[str, str]:
        api_key = _env("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "EXA_API_KEY environment variable not set. "
                "Get your API key at https://exa.ai"
            )
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute an Exa search (``contents.highlights`` -> description)."""
        del include_content
        log = _resolve_logger()
        try:
            status, data = await _http_json(
                "POST",
                f"{_EXA_BASE_URL}/search",
                headers=self._headers(),
                payload={
                    "query": query,
                    "numResults": limit,
                    "contents": {"highlights": True},
                },
                timeout=60,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.warning("Exa search error: {error}", error=exc)
            return {"success": False, "error": f"Exa search failed: {exc}"}
        if status != 200:
            log.warning("Exa search HTTP error: status={status}", status=status)
            return {"success": False, "error": f"Exa search failed: HTTP {status}"}
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Exa search failed: response was not valid JSON",
            }

        web_results: list[dict[str, Any]] = []
        for i, result in enumerate(data.get("results", []) or []):
            highlights = result.get("highlights") or []
            web_results.append(
                {
                    "url": result.get("url") or "",
                    "title": result.get("title") or "",
                    "description": " ".join(highlights) if highlights else "",
                    "position": i + 1,
                }
            )
        return {"success": True, "data": {"web": web_results}}

    async def extract(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        urls: list[str],
        format: str | None = None,  # noqa: A002 — kept for dispatcher parity
    ) -> list[dict[str, Any]]:
        """Fetch page contents via Exa ``/contents`` (``text: true``)."""
        del format  # Exa contents returns plain text
        log = _resolve_logger()
        try:
            status, data = await _http_json(
                "POST",
                f"{_EXA_BASE_URL}/contents",
                headers=self._headers(),
                payload={"urls": urls, "text": True},
                timeout=60,
            )
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": str(exc)}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Exa extract error: {error}", error=exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Exa extract failed: {exc}",
                }
                for u in urls
            ]
        if status != 200 or not isinstance(data, dict):
            detail = (
                f"HTTP {status}"
                if status != 200
                else "response was not valid JSON"
            )
            log.warning("Exa extract failed: {detail}", detail=detail)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Exa extract failed: {detail}",
                }
                for u in urls
            ]

        results: list[dict[str, Any]] = []
        for result in data.get("results", []) or []:
            content = result.get("text") or ""
            url = result.get("url") or ""
            title = result.get("title") or ""
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "error": None,
                    "metadata": {"sourceURL": url, "title": title},
                }
            )
        return results


_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveFreeProvider(WebSearchProvider):
    """Brave Search free tier (ported from Hermes ``plugins/web/brave_free``).

    Search-only (no extract capability). Free tier is 2,000 queries/month.

    Env var: ``BRAVE_SEARCH_API_KEY`` (https://brave.com/search/api/).
    """

    @property
    def name(self) -> str:
        # Hyphen form preserved for backward compat with Hermes config keys.
        return "brave-free"

    @property
    def display_name(self) -> str:
        return "Brave Search (Free)"

    def is_available(self) -> bool:
        """Return True when ``BRAVE_SEARCH_API_KEY`` is set to a non-empty value."""
        return bool(_env("BRAVE_SEARCH_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a search against the Brave Search API."""
        del include_content
        log = _resolve_logger()
        api_key = _env("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return {"success": False, "error": "BRAVE_SEARCH_API_KEY is not set"}

        # Brave's ``count`` is capped at 20.
        count = max(1, min(int(limit), 20))
        try:
            status, data = await _http_json(
                "GET",
                _BRAVE_ENDPOINT,
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                params={"q": query, "count": count},
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Brave Search request error: {error}", error=exc)
            return {"success": False, "error": f"Could not reach Brave Search: {exc}"}
        if status != 200:
            log.warning("Brave Search HTTP error: status={status}", status=status)
            return {
                "success": False,
                "error": f"Brave Search returned HTTP {status}",
            }
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Could not parse Brave Search response as JSON",
            }

        raw_results = (data.get("web") or {}).get("results", []) or []
        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("description", "")),
                "position": i + 1,
            }
            for i, r in enumerate(raw_results[:limit])
        ]
        return {"success": True, "data": {"web": web_results}}


class SearxngProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance (Hermes ``plugins/web/searxng``).

    Search-only — SearXNG aggregates upstream engines but does not fetch
    arbitrary URLs.

    Env var: ``SEARXNG_URL`` (e.g. ``http://localhost:8080``).
    """

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_env("SEARXNG_URL"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a search against the configured SearXNG instance."""
        del include_content
        log = _resolve_logger()
        base_url = _env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        try:
            status, data = await _http_json(
                "GET",
                f"{base_url}/search",
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                params={"q": query, "format": "json", "pageno": 1},
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("SearXNG request error: {error}", error=exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }
        if status != 200:
            log.warning("SearXNG HTTP error: status={status}", status=status)
            return {"success": False, "error": f"SearXNG returned HTTP {status}"}
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Could not parse SearXNG response as JSON",
            }

        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            raw_results,
            key=lambda r: float(r.get("score", 0)),
            reverse=True,
        )[:limit]
        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]
        return {"success": True, "data": {"web": web_results}}


_FIRECRAWL_DEFAULT_API_URL = "https://api.firecrawl.dev"


def _firecrawl_config() -> tuple[str, str] | None:
    """Return ``(api_key, api_url)`` for Firecrawl, or None when unconfigured.

    Mirrors Hermes' direct-mode config (``FIRECRAWL_API_KEY`` for cloud,
    ``FIRECRAWL_API_URL`` for self-hosted). The managed tool-gateway mode is
    Hermes-specific and intentionally not ported.
    """
    api_key = _env("FIRECRAWL_API_KEY")
    api_url = _env("FIRECRAWL_API_URL").rstrip("/")
    if not api_key and not api_url:
        return None
    return api_key, api_url or _FIRECRAWL_DEFAULT_API_URL


def _extract_web_search_results(data: Any) -> list[dict[str, Any]]:
    """Extract Firecrawl search results across direct/gateway response shapes."""
    if not isinstance(data, dict):
        return []
    inner = data.get("data")
    if isinstance(inner, list):
        return [item for item in inner if isinstance(item, dict)]
    if isinstance(inner, dict):
        for key in ("web", "results"):
            values = inner.get(key)
            if isinstance(values, list) and values:
                return [item for item in values if isinstance(item, dict)]
    for key in ("web", "results"):
        values = data.get(key)
        if isinstance(values, list) and values:
            return [item for item in values if isinstance(item, dict)]
    return []


def _extract_scrape_payload(data: Any) -> dict[str, Any]:
    """Normalize a Firecrawl scrape payload (unwraps the nested ``data`` key)."""
    if not isinstance(data, dict):
        return {}
    nested = data.get("data")
    if isinstance(nested, dict):
        return nested
    return data


class FirecrawlProvider(WebSearchProvider):
    """Firecrawl search + extract (ported from Hermes ``plugins/web/firecrawl``).

    Hermes used the ``firecrawl`` SDK plus a managed tool-gateway; adapted to
    the plain REST API (``POST /v2/search``, ``POST /v2/scrape``) with direct
    auth only. The redirect-aware SSRF re-check on the final URL is preserved
    via :mod:`kimi_cli.tools.web.url_safety`.

    Env vars: ``FIRECRAWL_API_KEY`` (cloud auth), ``FIRECRAWL_API_URL``
    (self-hosted instance).
    """

    @property
    def name(self) -> str:
        return "firecrawl"

    @property
    def display_name(self) -> str:
        return "Firecrawl"

    def is_available(self) -> bool:
        """Return True when Firecrawl direct config (key or URL) is set."""
        return _firecrawl_config() is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a Firecrawl ``/v2/search`` and normalize the result list."""
        del include_content
        log = _resolve_logger()
        config = _firecrawl_config()
        if config is None:
            return {
                "success": False,
                "error": (
                    "Web tools are not configured. Set FIRECRAWL_API_KEY for "
                    "cloud Firecrawl or set FIRECRAWL_API_URL for a self-hosted "
                    "Firecrawl instance."
                ),
            }
        api_key, api_url = config
        try:
            status, data = await _http_json(
                "POST",
                f"{api_url}/v2/search",
                headers=self._headers(api_key),
                payload={"query": query, "limit": limit},
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Firecrawl search error: {error}", error=exc)
            return {"success": False, "error": f"Firecrawl search failed: {exc}"}
        if status != 200:
            log.warning("Firecrawl search HTTP error: status={status}", status=status)
            return {
                "success": False,
                "error": f"Firecrawl search failed: HTTP {status}",
            }

        web_results: list[dict[str, Any]] = []
        for i, item in enumerate(_extract_web_search_results(data)):
            entry = dict(item)
            entry.setdefault("title", "")
            entry.setdefault("url", "")
            entry.setdefault("description", "")
            entry.setdefault("position", i + 1)
            web_results.append(entry)
        return {"success": True, "data": {"web": web_results}}

    async def extract(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        urls: list[str],
        format: str | None = None,  # noqa: A002 — kept for dispatcher parity
    ) -> list[dict[str, Any]]:
        """Scrape each URL via Firecrawl ``/v2/scrape`` (60s per-URL timeout).

        ``format`` selects ``"markdown"`` or ``"html"``; the default requests
        both and prefers markdown. After scraping, the final (post-redirect)
        URL is re-checked against SSRF rules. Per-URL failures become entries
        with an ``error`` field rather than raising.
        """
        from kimi_cli.tools.web.url_safety import async_is_safe_url

        log = _resolve_logger()
        config = _firecrawl_config()
        if config is None:
            message = (
                "Web tools are not configured. Set FIRECRAWL_API_KEY for cloud "
                "Firecrawl or set FIRECRAWL_API_URL for a self-hosted instance."
            )
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": message}
                for u in urls
            ]
        api_key, api_url = config

        formats: list[str]
        if format == "markdown":
            formats = ["markdown"]
        elif format == "html":
            formats = ["html"]
        else:
            formats = ["markdown", "html"]

        results: list[dict[str, Any]] = []
        for url in urls:
            try:
                status, data = await _http_json(
                    "POST",
                    f"{api_url}/v2/scrape",
                    headers=self._headers(api_key),
                    payload={"url": url, "formats": formats},
                    timeout=60,
                )
                if status != 200:
                    log.warning(
                        "Firecrawl scrape HTTP error for {url}: status={status}",
                        url=url,
                        status=status,
                    )
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": f"Firecrawl scrape failed: HTTP {status}",
                        }
                    )
                    continue

                scrape_payload = _extract_scrape_payload(data)
                metadata = scrape_payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                content_markdown = scrape_payload.get("markdown")
                content_html = scrape_payload.get("html")

                title = metadata.get("title", "")
                final_url = metadata.get("sourceURL", url)

                # Re-check SSRF safety after any redirect reported by Firecrawl.
                if not await async_is_safe_url(final_url):
                    log.warning(
                        "Blocked redirected web_extract for unsafe final URL: {url}",
                        url=final_url,
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": (
                                "Blocked: URL targets a private or internal "
                                "network address"
                            ),
                        }
                    )
                    continue

                if format == "markdown" or (format is None and content_markdown):
                    chosen_content = content_markdown
                else:
                    chosen_content = content_html or content_markdown or ""

                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": chosen_content,
                        "raw_content": chosen_content,
                        "error": None,
                        "metadata": metadata,
                    }
                )
            except Exception as scrape_err:  # noqa: BLE001 — per-URL failure
                log.warning(
                    "Firecrawl scrape failed for {url}: {error}",
                    url=url,
                    error=scrape_err,
                )
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(scrape_err),
                    }
                )

        return results


_PARALLEL_BASE_URL = "https://api.parallel.ai"


def _resolve_parallel_search_mode() -> str:
    """Return the validated ``PARALLEL_SEARCH_MODE`` value (default ``agentic``)."""
    mode = _env("PARALLEL_SEARCH_MODE").lower() or "agentic"
    if mode not in {"fast", "one-shot", "agentic"}:
        mode = "agentic"
    return mode


class ParallelProvider(WebSearchProvider):
    """Parallel.ai search + extract (ported from Hermes ``plugins/web/parallel``).

    Hermes used the ``parallel`` SDK; adapted to the plain REST API
    (``POST /v1beta/search`` / ``POST /v1beta/extract`` with the ``x-api-key``
    header) to avoid a new hard dependency.

    Env vars: ``PARALLEL_API_KEY`` (required), ``PARALLEL_SEARCH_MODE``
    (optional: ``agentic`` | ``fast`` | ``one-shot``).
    """

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set to a non-empty value."""
        return bool(_env("PARALLEL_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    @staticmethod
    def _headers() -> dict[str, str]:
        api_key = _env("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a Parallel beta search (``excerpts`` -> description)."""
        del include_content
        log = _resolve_logger()
        mode = _resolve_parallel_search_mode()
        try:
            status, data = await _http_json(
                "POST",
                f"{_PARALLEL_BASE_URL}/v1beta/search",
                headers=self._headers(),
                payload={
                    "search_queries": [query],
                    "objective": query,
                    "mode": mode,
                    "max_results": min(limit, 20),
                },
                timeout=60,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.warning("Parallel search error: {error}", error=exc)
            return {"success": False, "error": f"Parallel search failed: {exc}"}
        if status != 200:
            log.warning("Parallel search HTTP error: status={status}", status=status)
            return {"success": False, "error": f"Parallel search failed: HTTP {status}"}
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Parallel search failed: response was not valid JSON",
            }

        web_results: list[dict[str, Any]] = []
        for i, result in enumerate(data.get("results", []) or []):
            excerpts = result.get("excerpts") or []
            web_results.append(
                {
                    "url": result.get("url") or "",
                    "title": result.get("title") or "",
                    "description": " ".join(excerpts) if excerpts else "",
                    "position": i + 1,
                }
            )
        return {"success": True, "data": {"web": web_results}}

    async def extract(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        urls: list[str],
        format: str | None = None,  # noqa: A002 — kept for dispatcher parity
    ) -> list[dict[str, Any]]:
        """Extract content via Parallel ``/v1beta/extract`` (``full_content``).

        One entry per successful URL plus one per entry in the response
        ``errors`` list; errors are returned as items, never raised.
        """
        del format  # Parallel extract always returns full content
        log = _resolve_logger()
        try:
            status, data = await _http_json(
                "POST",
                f"{_PARALLEL_BASE_URL}/v1beta/extract",
                headers=self._headers(),
                payload={"urls": urls, "full_content": True},
                timeout=60,
            )
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": str(exc)}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Parallel extract error: {error}", error=exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Parallel extract failed: {exc}",
                }
                for u in urls
            ]
        if status != 200 or not isinstance(data, dict):
            detail = (
                f"HTTP {status}"
                if status != 200
                else "response was not valid JSON"
            )
            log.warning("Parallel extract failed: {detail}", detail=detail)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Parallel extract failed: {detail}",
                }
                for u in urls
            ]

        results: list[dict[str, Any]] = []
        for result in data.get("results", []) or []:
            content = result.get("full_content") or ""
            if not content:
                content = "\n\n".join(result.get("excerpts") or [])
            url = result.get("url") or ""
            title = result.get("title") or ""
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "error": None,
                    "metadata": {"sourceURL": url, "title": title},
                }
            )
        for error in data.get("errors", []) or []:
            results.append(
                {
                    "url": error.get("url") or "",
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": (
                        error.get("content")
                        or error.get("error_type")
                        or "extraction failed"
                    ),
                    "metadata": {"sourceURL": error.get("url") or ""},
                }
            )
        return results


_XAI_BASE_URL = "https://api.x.ai/v1"
_XAI_DEFAULT_MODEL = "grok-build-0.1"
_XAI_DEFAULT_TIMEOUT = 90

# Match the JSON object Grok is asked to emit. Tolerates leading/trailing
# prose since reasoning models occasionally narrate before the JSON block
# even when explicitly asked not to.
_XAI_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


class XAIProvider(WebSearchProvider):
    """xAI web search via Grok's server-side ``web_search`` tool.

    Ported from Hermes ``plugins/web/xai``. Sends a structured prompt to the
    xAI Responses API with ``tools=[{"type": "web_search"}]`` and parses the
    requested JSON result block, falling back to message annotations and then
    the raw ``citations`` list. Search-only.

    Dropped from the Hermes version (clearly separable, no kimi-cli
    equivalent): the Hermes-managed xAI OAuth path with refresh-on-401 retry,
    and the ``web.xai`` config knobs (model/timeout/domain filters) — kimi-cli
    has no such config section.

    Env var: ``XAI_API_KEY``.
    """

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI Web Search (Grok)"

    def is_available(self) -> bool:
        """Return True when ``XAI_API_KEY`` is set to a non-empty value."""
        return bool(_env("XAI_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    async def search(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        limit: int = 5,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Execute a Grok-backed web search via the xAI Responses API."""
        del include_content
        log = _resolve_logger()
        api_key = _env("XAI_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "No xAI credentials found. Set XAI_API_KEY.",
            }

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 100))

        payload: dict[str, Any] = {
            "model": _XAI_DEFAULT_MODEL,
            "input": [{"role": "user", "content": self._build_prompt(query, limit)}],
            "tools": [{"type": "web_search"}],
            # Drop inline citation markdown — we want the JSON block clean,
            # and we read URLs from annotations / citations separately.
            "include": ["no_inline_citations"],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        try:
            status, data = await _http_json(
                "POST",
                f"{_XAI_BASE_URL}/responses",
                headers=headers,
                payload=payload,
                timeout=_XAI_DEFAULT_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("xAI web search request error: {error}", error=exc)
            return {"success": False, "error": f"Could not reach xAI: {exc}"}
        if status != 200:
            log.warning("xAI web search HTTP error: status={status}", status=status)
            return {
                "success": False,
                "error": f"xAI web search returned HTTP {status}",
            }
        if not isinstance(data, dict):
            return {
                "success": False,
                "error": "Could not parse xAI Responses API reply as JSON",
            }

        # xAI's Responses surface sometimes returns HTTP 200 with an error
        # envelope (model overloaded, content-policy refusal, etc.).
        api_error = data.get("error")
        if isinstance(api_error, dict):
            err_msg = (
                api_error.get("message") or api_error.get("code") or "unknown error"
            )
            log.warning("xAI web search returned error envelope: {error}", error=err_msg)
            return {"success": False, "error": f"xAI returned an error: {err_msg}"}

        web_results = self._extract_results(data, limit=limit)
        # An empty list is a valid, successful response — the model decides
        # whether to retry (matches brave-free / exa behavior on 0 hits).
        return {"success": True, "data": {"web": web_results}}

    # -- Prompt + parsing (ported from Hermes, response contract identical) --

    @staticmethod
    def _build_prompt(query: str, limit: int) -> str:
        """Compose the prompt that asks Grok to act as a search engine."""
        return (
            "Use the web_search tool to find current information for the query below, "
            "then respond with ONLY a single JSON object — no prose, no markdown "
            "fences, no inline citation links — matching this exact schema:\n\n"
            '{"results": [{"title": "string", "url": "string", '
            '"description": "1-2 sentence summary"}]}\n\n'
            f"Return at most {limit} results, ordered by relevance, with absolute "
            "https:// URLs. If no usable results exist, return "
            '{"results": []}.\n\n'
            f"Query: {query}"
        )

    @classmethod
    def _extract_results(
        cls,
        response_data: dict[str, Any],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Pull ``[{title, url, description, position}, ...]`` from a reply.

        1. Walk ``output[*].content[*].text`` ``output_text`` blocks and parse
           the first JSON object that has a ``results`` list.
        2. Fall back to ``url_citation`` message annotations.
        3. Last-ditch: the raw ``citations`` list (no titles/descriptions).
        """
        text_blocks, annotations = cls._collect_output_text(response_data)

        for block in text_blocks:
            parsed = cls._try_parse_json_results(block, limit=limit)
            if parsed:
                return parsed

        if annotations:
            joined_text = "\n".join(text_blocks)
            annotation_results = cls._results_from_annotations(
                annotations, joined_text, limit=limit
            )
            if annotation_results:
                return annotation_results

        citations = response_data.get("citations") or []
        if isinstance(citations, list):
            return [
                {
                    "title": "",
                    "url": str(u),
                    "description": "",
                    "position": i + 1,
                }
                for i, u in enumerate(citations[:limit])
                if isinstance(u, str) and u.strip()
            ]
        return []

    @staticmethod
    def _collect_output_text(
        response_data: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Return ``(text_blocks, annotations)`` extracted from ``response.output``."""
        text_blocks: list[str] = []
        annotations: list[dict[str, Any]] = []
        output = response_data.get("output")
        if not isinstance(output, list):
            return text_blocks, annotations

        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict) or chunk.get("type") != "output_text":
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text)
                chunk_annotations = chunk.get("annotations")
                if isinstance(chunk_annotations, list):
                    for ann in chunk_annotations:
                        if isinstance(ann, dict):
                            annotations.append(ann)
        return text_blocks, annotations

    @staticmethod
    def _try_parse_json_results(
        text: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]] | None:
        """Parse a JSON object with a ``results`` array out of ``text``.

        Returns the normalized result list on success, ``None`` when the block
        has no valid JSON object or no ``results`` key. Tolerates leading and
        trailing prose around the JSON block.
        """
        candidates = [text]
        match = _XAI_JSON_BLOCK_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = orjson.loads(candidate)
            except ValueError:  # orjson.JSONDecodeError subclasses ValueError
                continue
            if not isinstance(parsed, dict):
                continue
            results = parsed.get("results")
            if not isinstance(results, list):
                continue
            normalized: list[dict[str, Any]] = []
            for row in results[:limit]:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "title": str(row.get("title", "")).strip(),
                        "url": url,
                        "description": str(row.get("description", "")).strip(),
                        # Renumber from the kept results so a dropped malformed
                        # row doesn't leave a position gap.
                        "position": len(normalized) + 1,
                    }
                )
            if normalized:
                return normalized
        return None

    @staticmethod
    def _results_from_annotations(
        annotations: list[dict[str, Any]],
        joined_text: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Best-effort fallback: derive results from ``url_citation`` annotations.

        Slices ~200 characters of text preceding each citation as the
        description.
        """
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for ann in annotations:
            if ann.get("type") != "url_citation":
                continue
            url = str(ann.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)

            description = ""
            start = ann.get("start_index")
            end = ann.get("end_index")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(joined_text)
            ):
                window_start = max(0, start - 200)
                description = joined_text[window_start:start].strip()
                if len(description) > 200:
                    description = description[-200:].strip()

            results.append(
                {
                    "title": "",
                    "url": url,
                    "description": description,
                    "position": len(results) + 1,
                }
            )
            if len(results) >= limit:
                break
        return results


register_provider(TavilyProvider())
register_provider(ExaProvider())
register_provider(BraveFreeProvider())
register_provider(SearxngProvider())
register_provider(FirecrawlProvider())
register_provider(ParallelProvider())
register_provider(XAIProvider())
register_provider(DDGSProvider())
register_provider(LocalTrafilaturaProvider())
