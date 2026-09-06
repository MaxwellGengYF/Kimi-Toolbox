"""Tests for the Hermes-ported web providers and the 404 self-heal fallback.

Covers:

- Import-time self-registration of the new built-in providers.
- Registry resolution with the extended legacy preference order, explicit
  ``web.search_backend`` config, and capability filtering.
- Each ported provider (tavily, exa, brave-free, searxng, firecrawl,
  parallel, xai): env-var availability plus one mocked-HTTP success and one
  failure test.
- ``KimiServiceProvider`` search 404 fallback to the URL derived from the
  LLM provider base URL, and the equivalent ``fetch_url`` service fallback.

All tests are offline: HTTP goes through fake ``new_client_session``
factories, same convention as ``test_web_providers.py``.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from pydantic import SecretStr

from kimi_cli.config import Config, FetchConfig, LLMProvider
from kimi_cli.tools.web.fetch import Params as FetchParams
from kimi_cli.tools.web.fetch import fetch_url
from kimi_cli.tools.web.providers import (
    BraveFreeProvider,
    DDGSProvider,
    ExaProvider,
    FirecrawlProvider,
    KimiServiceProvider,
    LocalTrafilaturaProvider,
    ParallelProvider,
    SearxngProvider,
    TavilyProvider,
    WebSearchProvider,
    XAIProvider,
    _reset_for_tests,
    get_active_extract_provider,
    get_active_search_provider,
    list_providers,
    register_provider,
)
from tests.conftest import tool_call_context

# Captured while the test session is still in the collection phase (no
# fixture has cleared the registry yet): proof that importing the providers
# module self-registers the built-ins.
_REGISTERED_AT_IMPORT = {p.name for p in list_providers()}


# ---------------------------------------------------------------------------
# Fake aiohttp primitives (same convention as test_web_providers.py)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: Any = None, text: str = "") -> None:
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload

    async def text(self) -> str:
        return self._text


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeSession:
    """Route ``(method, url)`` pairs to canned responses; record all calls."""

    def __init__(
        self,
        routes: dict[tuple[str, str], _FakeResponse] | None = None,
        *,
        fail: str | None = None,
    ) -> None:
        self._routes = routes or {}
        self._fail = fail
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeSession:
        if self._fail == "timeout":
            raise TimeoutError("timed out")
        if self._fail == "network":
            raise aiohttp.ClientError("connection refused")
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _FakeRequestContext:
        self.calls.append(("POST", url, kwargs))
        return _FakeRequestContext(self._routes[("POST", url)])

    def get(self, url: str, **kwargs: Any) -> _FakeRequestContext:
        self.calls.append(("GET", url, kwargs))
        return _FakeRequestContext(self._routes[("GET", url)])


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    """Route the providers' ``_resolve_new_client_session()`` to *session*."""
    monkeypatch.setattr(
        "kimi_cli.tools.web.providers.new_client_session", lambda **kw: session
    )


class _FakeProvider(WebSearchProvider):
    """Configurable fake provider for resolution tests."""

    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        search: bool = True,
        extract: bool = False,
    ) -> None:
        self._name = name
        self._available = available
        self._search = search
        self._extract = extract

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def supports_search(self) -> bool:
        return self._search

    def supports_extract(self) -> bool:
        return self._extract


@pytest.fixture(autouse=True)
def _registry_isolation() -> Any:
    """Save the built-in registry around each test."""
    before = list(list_providers())
    _reset_for_tests()
    yield
    _reset_for_tests()
    for provider in before:
        register_provider(provider)


# ---------------------------------------------------------------------------
# Registration / resolution
# ---------------------------------------------------------------------------


class TestImportTimeRegistration:
    def test_builtin_providers_self_register_on_import(self) -> None:
        for name in (
            "tavily",
            "exa",
            "brave-free",
            "searxng",
            "firecrawl",
            "parallel",
            "xai",
            "ddgs",
            "local",
        ):
            assert name in _REGISTERED_AT_IMPORT
        # ``kimi`` is registered by SearchWeb.__init__, not at import time.
        assert "kimi" not in _REGISTERED_AT_IMPORT


class TestResolutionWithNewProviders:
    def test_tavily_wins_when_kimi_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        register_provider(_FakeProvider("kimi", available=False))
        tavily = TavilyProvider()
        register_provider(tavily)
        register_provider(ExaProvider())
        # Two providers are available, so the legacy preference walk (tavily
        # before exa) decides.
        assert get_active_search_provider(Config()) is tavily

    def test_search_preference_firecrawl_then_parallel_then_tavily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        register_provider(_FakeProvider("kimi", available=False))
        firecrawl = FirecrawlProvider()
        parallel = ParallelProvider()
        tavily = TavilyProvider()
        register_provider(firecrawl)
        register_provider(parallel)
        register_provider(tavily)

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        assert get_active_search_provider(Config()) is firecrawl

        monkeypatch.delenv("FIRECRAWL_API_KEY")
        assert get_active_search_provider(Config()) is parallel

        monkeypatch.delenv("PARALLEL_API_KEY")
        assert get_active_search_provider(Config()) is tavily

    def test_explicit_config_wins_despite_unavailable(self) -> None:
        exa = ExaProvider()  # no EXA_API_KEY -> unavailable
        register_provider(exa)
        register_provider(_FakeProvider("other", available=True))
        config = Config()
        config.web.search_backend = "exa"
        assert get_active_search_provider(config) is exa

    def test_brave_free_not_chosen_for_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
        brave = BraveFreeProvider()
        register_provider(brave)
        # brave-free is search-only: extract resolution must skip it.
        assert get_active_extract_provider(Config()) is None
        assert get_active_search_provider(Config()) is brave

    def test_extract_preference_local_then_firecrawl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        local = LocalTrafilaturaProvider()
        firecrawl = FirecrawlProvider()
        register_provider(local)
        register_provider(firecrawl)
        assert get_active_extract_provider(Config()) is local

        # Swap the always-available local provider for an unavailable fake;
        # firecrawl is next in the extract preference list.
        _reset_for_tests()
        register_provider(_FakeProvider("local", available=False, extract=True, search=False))
        register_provider(firecrawl)
        assert get_active_extract_provider(Config()) is firecrawl


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


class TestTavilyProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        assert TavilyProvider().is_available() is True
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert TavilyProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = TavilyProvider()
        assert provider.name == "tavily"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    async def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.tavily.com/search"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "title": "T",
                                "url": "https://u.example",
                                "content": "snippet",
                            }
                        ]
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await TavilyProvider().search("hello", 10)

        assert result["success"] is True
        web = result["data"]["web"]
        assert web == [
            {
                "title": "T",
                "url": "https://u.example",
                "description": "snippet",
                "position": 1,
            }
        ]
        method, url, kwargs = session.calls[0]
        assert method == "POST"
        assert kwargs["json"]["api_key"] == "tvly-test-key"
        assert kwargs["json"]["query"] == "hello"
        assert kwargs["json"]["max_results"] == 10

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        session = _FakeSession(
            {("POST", "https://api.tavily.com/search"): _FakeResponse(500, {})}
        )
        _patch_session(monkeypatch, session)

        result = await TavilyProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    async def test_search_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = await TavilyProvider().search("hello")
        assert result["success"] is False
        assert "TAVILY_API_KEY" in result["error"]

    async def test_extract_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.tavily.com/extract"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "url": "https://a.example",
                                "title": "A",
                                "raw_content": "body A",
                            }
                        ],
                        "failed_results": [
                            {"url": "https://b.example", "error": "boom"}
                        ],
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        results = await TavilyProvider().extract(
            ["https://a.example", "https://b.example"]
        )

        assert len(results) == 2
        assert results[0]["url"] == "https://a.example"
        assert results[0]["content"] == "body A"
        assert results[0]["error"] is None
        assert results[1]["url"] == "https://b.example"
        assert results[1]["error"] == "boom"
        assert session.calls[0][2]["json"]["urls"] == [
            "https://a.example",
            "https://b.example",
        ]

    async def test_extract_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        session = _FakeSession(
            {("POST", "https://api.tavily.com/extract"): _FakeResponse(503, {})}
        )
        _patch_session(monkeypatch, session)

        results = await TavilyProvider().extract(["https://a.example"])
        assert len(results) == 1
        assert "HTTP 503" in results[0]["error"]


# ---------------------------------------------------------------------------
# Exa
# ---------------------------------------------------------------------------


class TestExaProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        assert ExaProvider().is_available() is True
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        assert ExaProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = ExaProvider()
        assert provider.name == "exa"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    async def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.exa.ai/search"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "url": "https://u.example",
                                "title": "T",
                                "highlights": ["h1", "h2"],
                            }
                        ]
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await ExaProvider().search("hello", 3)

        assert result["success"] is True
        web = result["data"]["web"]
        assert web == [
            {
                "url": "https://u.example",
                "title": "T",
                "description": "h1 h2",
                "position": 1,
            }
        ]
        kwargs = session.calls[0][2]
        assert kwargs["headers"]["x-api-key"] == "exa-test-key"
        assert kwargs["json"] == {
            "query": "hello",
            "numResults": 3,
            "contents": {"highlights": True},
        }

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        session = _FakeSession(
            {("POST", "https://api.exa.ai/search"): _FakeResponse(401, {})}
        )
        _patch_session(monkeypatch, session)

        result = await ExaProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 401" in result["error"]

    async def test_extract_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.exa.ai/contents"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {"url": "https://a.example", "title": "A", "text": "body"}
                        ]
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        results = await ExaProvider().extract(["https://a.example"])
        assert results[0]["content"] == "body"
        assert results[0]["error"] is None
        assert session.calls[0][2]["json"] == {"urls": ["https://a.example"], "text": True}

    async def test_extract_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        session = _FakeSession(
            {("POST", "https://api.exa.ai/contents"): _FakeResponse(500, {})}
        )
        _patch_session(monkeypatch, session)

        results = await ExaProvider().extract(["https://a.example"])
        assert "HTTP 500" in results[0]["error"]


# ---------------------------------------------------------------------------
# Brave (free tier)
# ---------------------------------------------------------------------------


class TestBraveFreeProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
        assert BraveFreeProvider().is_available() is True
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        assert BraveFreeProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = BraveFreeProvider()
        assert provider.name == "brave-free"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    async def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
        session = _FakeSession(
            {
                ("GET", "https://api.search.brave.com/res/v1/web/search"): _FakeResponse(
                    200,
                    {
                        "web": {
                            "results": [
                                {
                                    "title": "T",
                                    "url": "https://u.example",
                                    "description": "d",
                                }
                            ]
                        }
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await BraveFreeProvider().search("hello", 5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "T",
                "url": "https://u.example",
                "description": "d",
                "position": 1,
            }
        ]
        kwargs = session.calls[0][2]
        assert kwargs["params"] == {"q": "hello", "count": 5}
        assert kwargs["headers"]["X-Subscription-Token"] == "brave-test-key"

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
        session = _FakeSession(
            {
                ("GET", "https://api.search.brave.com/res/v1/web/search"): _FakeResponse(
                    429, {}
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await BraveFreeProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 429" in result["error"]

    async def test_search_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        result = await BraveFreeProvider().search("hello")
        assert result["success"] is False
        assert "BRAVE_SEARCH_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# SearXNG
# ---------------------------------------------------------------------------


class TestSearxngProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARXNG_URL", "http://searx.test:8080")
        assert SearxngProvider().is_available() is True
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        assert SearxngProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = SearxngProvider()
        assert provider.name == "searxng"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    async def test_search_success_sorts_by_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARXNG_URL", "http://searx.test:8080/")
        session = _FakeSession(
            {
                ("GET", "http://searx.test:8080/search"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "title": "low",
                                "url": "https://low.example",
                                "content": "c1",
                                "score": 0.1,
                            },
                            {
                                "title": "high",
                                "url": "https://high.example",
                                "content": "c2",
                                "score": 0.9,
                            },
                        ]
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await SearxngProvider().search("hello", 5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert [r["url"] for r in web] == ["https://high.example", "https://low.example"]
        assert web[0]["position"] == 1
        kwargs = session.calls[0][2]
        assert kwargs["params"] == {"q": "hello", "format": "json", "pageno": 1}

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEARXNG_URL", "http://searx.test:8080")
        session = _FakeSession(
            {("GET", "http://searx.test:8080/search"): _FakeResponse(404, {})}
        )
        _patch_session(monkeypatch, session)

        result = await SearxngProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 404" in result["error"]

    async def test_search_missing_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        result = await SearxngProvider().search("hello")
        assert result["success"] is False
        assert "SEARXNG_URL" in result["error"]


# ---------------------------------------------------------------------------
# Firecrawl
# ---------------------------------------------------------------------------


class TestFirecrawlProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
        assert FirecrawlProvider().is_available() is False

        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        assert FirecrawlProvider().is_available() is True

        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://firecrawl.test:3002")
        assert FirecrawlProvider().is_available() is True

    def test_name_and_capabilities(self) -> None:
        provider = FirecrawlProvider()
        assert provider.name == "firecrawl"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    async def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.firecrawl.dev/v2/search"): _FakeResponse(
                    200,
                    {
                        "success": True,
                        "data": {
                            "web": [
                                {
                                    "title": "T",
                                    "url": "https://u.example",
                                    "description": "d",
                                }
                            ]
                        },
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await FirecrawlProvider().search("hello", 5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert web[0]["title"] == "T"
        assert web[0]["url"] == "https://u.example"
        assert web[0]["description"] == "d"
        assert web[0]["position"] == 1
        kwargs = session.calls[0][2]
        assert kwargs["json"] == {"query": "hello", "limit": 5}
        assert kwargs["headers"]["Authorization"] == "Bearer fc-test-key"

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        session = _FakeSession(
            {("POST", "https://api.firecrawl.dev/v2/search"): _FakeResponse(500, {})}
        )
        _patch_session(monkeypatch, session)

        result = await FirecrawlProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    async def test_extract_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _safe)
        session = _FakeSession(
            {
                ("POST", "https://api.firecrawl.dev/v2/scrape"): _FakeResponse(
                    200,
                    {
                        "success": True,
                        "data": {
                            "markdown": "# md",
                            "html": "<h1>x</h1>",
                            "metadata": {
                                "title": "T",
                                "sourceURL": "https://a.example/",
                            },
                        },
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        results = await FirecrawlProvider().extract(["https://a.example/"])

        assert len(results) == 1
        assert results[0]["url"] == "https://a.example/"
        assert results[0]["title"] == "T"
        assert results[0]["content"] == "# md"
        assert results[0]["error"] is None
        assert session.calls[0][2]["json"] == {
            "url": "https://a.example/",
            "formats": ["markdown", "html"],
        }

    async def test_extract_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        session = _FakeSession(
            {("POST", "https://api.firecrawl.dev/v2/scrape"): _FakeResponse(500, {})}
        )
        _patch_session(monkeypatch, session)

        results = await FirecrawlProvider().extract(["https://a.example/"])
        assert "HTTP 500" in results[0]["error"]

    async def test_extract_blocks_unsafe_redirect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")

        async def _unsafe(url: str) -> bool:
            return False

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _unsafe)
        session = _FakeSession(
            {
                ("POST", "https://api.firecrawl.dev/v2/scrape"): _FakeResponse(
                    200,
                    {
                        "success": True,
                        "data": {
                            "markdown": "# md",
                            "metadata": {
                                "title": "T",
                                "sourceURL": "http://169.254.169.254/latest",
                            },
                        },
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        results = await FirecrawlProvider().extract(["https://a.example/"])
        assert results[0]["url"] == "http://169.254.169.254/latest"
        assert "private or internal" in results[0]["error"]


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


class TestParallelProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        assert ParallelProvider().is_available() is True
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        assert ParallelProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = ParallelProvider()
        assert provider.name == "parallel"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    async def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.parallel.ai/v1beta/search"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "url": "https://u.example",
                                "title": "T",
                                "excerpts": ["e1", "e2"],
                            }
                        ]
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await ParallelProvider().search("hello", 5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "url": "https://u.example",
                "title": "T",
                "description": "e1 e2",
                "position": 1,
            }
        ]
        kwargs = session.calls[0][2]
        assert kwargs["headers"]["x-api-key"] == "parallel-test-key"
        assert kwargs["json"] == {
            "search_queries": ["hello"],
            "objective": "hello",
            "mode": "agentic",
            "max_results": 5,
        }

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.parallel.ai/v1beta/search"): _FakeResponse(
                    500, {}
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await ParallelProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    async def test_extract_success_with_errors_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.parallel.ai/v1beta/extract"): _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "url": "https://a.example",
                                "title": "A",
                                "full_content": "full body",
                                "excerpts": [],
                            }
                        ],
                        "errors": [
                            {
                                "url": "https://b.example",
                                "content": "fetch exploded",
                                "error_type": "fetch_error",
                            }
                        ],
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        results = await ParallelProvider().extract(
            ["https://a.example", "https://b.example"]
        )

        assert len(results) == 2
        assert results[0]["content"] == "full body"
        assert results[0]["error"] is None
        assert results[1]["url"] == "https://b.example"
        assert results[1]["error"] == "fetch exploded"
        assert session.calls[0][2]["json"] == {
            "urls": ["https://a.example", "https://b.example"],
            "full_content": True,
        }

    async def test_extract_network_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARALLEL_API_KEY", "parallel-test-key")
        session = _FakeSession(fail="network")
        _patch_session(monkeypatch, session)

        results = await ParallelProvider().extract(["https://a.example"])
        assert len(results) == 1
        assert "Parallel extract failed" in results[0]["error"]
        assert "connection refused" in results[0]["error"]


# ---------------------------------------------------------------------------
# xAI
# ---------------------------------------------------------------------------


def _xai_payload_with_text(text: str) -> dict[str, Any]:
    return {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    }


class TestXAIProvider:
    def test_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        assert XAIProvider().is_available() is True
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert XAIProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = XAIProvider()
        assert provider.name == "xai"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    async def test_search_success_parses_json_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        payload = _xai_payload_with_text(
            '{"results": [{"title": "T", "url": "https://u.example", '
            '"description": "d"}]}'
        )
        session = _FakeSession(
            {("POST", "https://api.x.ai/v1/responses"): _FakeResponse(200, payload)}
        )
        _patch_session(monkeypatch, session)

        result = await XAIProvider().search("hello", 5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "T",
                "url": "https://u.example",
                "description": "d",
                "position": 1,
            }
        ]
        kwargs = session.calls[0][2]
        assert kwargs["headers"]["Authorization"] == "Bearer xai-test-key"
        assert kwargs["json"]["tools"] == [{"type": "web_search"}]
        assert kwargs["json"]["include"] == ["no_inline_citations"]

    async def test_search_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        session = _FakeSession(
            {("POST", "https://api.x.ai/v1/responses"): _FakeResponse(500, {})}
        )
        _patch_session(monkeypatch, session)

        result = await XAIProvider().search("hello")
        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    async def test_search_error_envelope_on_200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.x.ai/v1/responses"): _FakeResponse(
                    200, {"error": {"message": "model overloaded"}}
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await XAIProvider().search("hello")
        assert result["success"] is False
        assert "model overloaded" in result["error"]

    async def test_search_falls_back_to_citations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
        session = _FakeSession(
            {
                ("POST", "https://api.x.ai/v1/responses"): _FakeResponse(
                    200,
                    {
                        "output": [],
                        "citations": ["https://c1.example", "https://c2.example"],
                    },
                )
            }
        )
        _patch_session(monkeypatch, session)

        result = await XAIProvider().search("hello", 5)
        assert result["success"] is True
        web = result["data"]["web"]
        assert [r["url"] for r in web] == ["https://c1.example", "https://c2.example"]
        assert web[0]["position"] == 1

    async def test_search_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        result = await XAIProvider().search("hello")
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Kimi service 404 self-heal fallback
# ---------------------------------------------------------------------------

_KIMI_SEARCH_PAYLOAD: dict[str, Any] = {
    "search_results": [
        {
            "site_name": "s",
            "title": "t",
            "url": "u",
            "snippet": "sni",
            "content": "c",
            "date": "d",
            "icon": "i",
            "mime": "m",
        }
    ]
}


def _set_llm_provider(config: Config, base_url: str) -> None:
    config.provider = LLMProvider(
        type="kimi", base_url=base_url, api_key=SecretStr("test-api-key")
    )


class TestKimiServiceProvider404Fallback:
    async def test_404_retries_derived_url(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = config.services.search
        assert svc is not None
        svc.base_url = "https://api.moonshot.cn/v1/search"  # stale (404) URL
        _set_llm_provider(config, "https://api.kimi.com/coding/v1")

        session = _FakeSession(
            {
                ("POST", "https://api.moonshot.cn/v1/search"): _FakeResponse(404, {}),
                ("POST", "https://api.kimi.com/coding/v1/search"): _FakeResponse(
                    200, _KIMI_SEARCH_PAYLOAD
                ),
            }
        )
        monkeypatch.setattr(
            "kimi_cli.tools.web.search.new_client_session", lambda **kw: session
        )

        provider = KimiServiceProvider(config, runtime)
        with tool_call_context("SearchWeb"):
            result = await provider.search("hello", 5)

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "t"
        assert [url for _method, url, _kw in session.calls] == [
            "https://api.moonshot.cn/v1/search",
            "https://api.kimi.com/coding/v1/search",
        ]

    async def test_404_without_provider_base_url_does_not_retry(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = config.services.search
        assert svc is not None
        svc.base_url = "https://api.moonshot.cn/v1/search"
        assert config.provider is None

        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/search"): _FakeResponse(404, {})}
        )
        monkeypatch.setattr(
            "kimi_cli.tools.web.search.new_client_session", lambda **kw: session
        )

        provider = KimiServiceProvider(config, runtime)
        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "Status: 404" in result["error"]
        assert len(session.calls) == 1

    async def test_404_with_matching_derived_url_does_not_retry(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = config.services.search
        assert svc is not None
        svc.base_url = "https://api.moonshot.cn/v1/search"
        # Derived URL equals the configured one -> nothing new to try.
        _set_llm_provider(config, "https://api.moonshot.cn/v1/")

        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/search"): _FakeResponse(404, {})}
        )
        monkeypatch.setattr(
            "kimi_cli.tools.web.search.new_client_session", lambda **kw: session
        )

        provider = KimiServiceProvider(config, runtime)
        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "Status: 404" in result["error"]
        assert len(session.calls) == 1

    async def test_non_404_error_does_not_retry(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = config.services.search
        assert svc is not None
        svc.base_url = "https://api.moonshot.cn/v1/search"
        _set_llm_provider(config, "https://api.kimi.com/coding/v1")

        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/search"): _FakeResponse(500, {})}
        )
        monkeypatch.setattr(
            "kimi_cli.tools.web.search.new_client_session", lambda **kw: session
        )

        provider = KimiServiceProvider(config, runtime)
        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "Status: 500" in result["error"]
        assert len(session.calls) == 1


class TestFetchService404Fallback:
    def _make_tool(self, runtime: Any) -> fetch_url:
        runtime.config.services.fetch = FetchConfig(
            base_url="https://api.moonshot.cn/v1/fetch",
            api_key=SecretStr("test-key"),
        )
        return fetch_url(config=runtime.config, runtime=runtime)

    async def test_404_retries_derived_url(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_llm_provider(runtime.config, "https://api.kimi.com/coding/v1")
        tool = self._make_tool(runtime)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _safe)
        session = _FakeSession(
            {
                ("POST", "https://api.moonshot.cn/v1/fetch"): _FakeResponse(404),
                ("POST", "https://api.kimi.com/coding/v1/fetch"): _FakeResponse(
                    200, text="# Service Content"
                ),
            }
        )
        monkeypatch.setattr(
            "kimi_cli.utils.aiohttp.new_client_session", lambda **kw: session
        )

        with tool_call_context("fetch_url"):
            result = await tool._fetch_with_service(FetchParams(url="https://example.com"))

        assert not result.is_error
        assert result.output == "# Service Content"
        assert [url for _method, url, _kw in session.calls] == [
            "https://api.moonshot.cn/v1/fetch",
            "https://api.kimi.com/coding/v1/fetch",
        ]

    async def test_404_without_provider_base_url_does_not_retry(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert runtime.config.provider is None
        tool = self._make_tool(runtime)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _safe)
        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/fetch"): _FakeResponse(404)}
        )
        monkeypatch.setattr(
            "kimi_cli.utils.aiohttp.new_client_session", lambda **kw: session
        )

        with tool_call_context("fetch_url"):
            result = await tool._fetch_with_service(FetchParams(url="https://example.com"))

        assert result.is_error
        assert "Status: 404" in result.message
        assert len(session.calls) == 1

    async def test_404_with_matching_derived_url_does_not_retry(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Derived URL equals the configured one -> nothing new to try.
        _set_llm_provider(runtime.config, "https://api.moonshot.cn/v1/")
        tool = self._make_tool(runtime)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _safe)
        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/fetch"): _FakeResponse(404)}
        )
        monkeypatch.setattr(
            "kimi_cli.utils.aiohttp.new_client_session", lambda **kw: session
        )

        with tool_call_context("fetch_url"):
            result = await tool._fetch_with_service(FetchParams(url="https://example.com"))

        assert result.is_error
        assert "Status: 404" in result.message
        assert len(session.calls) == 1

    async def test_non_404_error_does_not_retry(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_llm_provider(runtime.config, "https://api.kimi.com/coding/v1")
        tool = self._make_tool(runtime)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.async_is_safe_url", _safe)
        session = _FakeSession(
            {("POST", "https://api.moonshot.cn/v1/fetch"): _FakeResponse(500)}
        )
        monkeypatch.setattr(
            "kimi_cli.utils.aiohttp.new_client_session", lambda **kw: session
        )

        with tool_call_context("fetch_url"):
            result = await tool._fetch_with_service(FetchParams(url="https://example.com"))

        assert result.is_error
        assert "Status: 500" in result.message
        assert len(session.calls) == 1
