"""Pluggable web search/extract providers.

Ported from the Hermes project's ``agent/web_search_provider.py`` +
``agent/web_search_registry.py``. Providers self-register at import time
(``ddgs``, ``local``); the ``kimi`` provider is constructed with a
``Config`` + ``Runtime`` and registered by ``SearchWeb.__init__``.

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
import threading
from typing import TYPE_CHECKING, Any

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

_SEARCH_LEGACY_PREFERENCE = ("kimi", "ddgs", "local")
_EXTRACT_LEGACY_PREFERENCE = ("local", "kimi")


def _resolve(configured: str | None, *, capability: str) -> WebSearchProvider | None:
    """Resolve the active provider for a capability ("search" | "extract").

    Resolution rules (in order):

    1. **Explicit config wins, ignoring availability.** ``web.backend`` or
       ``web.{capability}_backend`` names a registered provider that supports
       the capability — return it even if ``is_available()`` is False so the
       caller can surface a precise error instead of silently rerouting.
    2. **Single-provider shortcut.** When exactly one registered provider
       supports the capability AND ``is_available()`` is True, return it.
    3. **Legacy preference walk, filtered by availability.** For search:
       ``kimi`` → ``ddgs`` → ``local``; for extract: ``local`` → ``kimi``.
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

        try:
            # Server-side timeout is 30s, but page crawling can take longer.
            search_timeout = aiohttp.ClientTimeout(
                total=180, sock_read=90, sock_connect=15
            )
            async with (
                new_client_session(timeout=search_timeout) as session,
                session.post(
                    svc.base_url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Authorization": f"Bearer {api_key}",
                        "X-Msh-Tool-Call-Id": tool_call.id,
                        **self._runtime.oauth.common_headers(),
                        **(svc.custom_headers or {}),
                    },
                    json={
                        "text_query": query,
                        "limit": limit,
                        "enable_page_crawling": include_content,
                        "timeout_seconds": 30,
                    },
                ) as response,
            ):
                if response.status != 200:
                    log.warning(
                        "web_search HTTP error: status={status}, query={query}",
                        status=response.status,
                        query=query,
                    )
                    return {
                        "success": False,
                        "error": (
                            f"Failed to search. Status: {response.status}. "
                            "This may indicate that the search service is "
                            "currently unavailable."
                        ),
                    }

                try:
                    results = Response(**await response.json()).search_results
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


register_provider(DDGSProvider())
register_provider(LocalTrafilaturaProvider())
