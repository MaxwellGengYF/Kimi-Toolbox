"""Tests for the pluggable web search/extract provider architecture.

Covers the registry (register/get/list/reset), active-provider resolution
(explicit config wins, availability filtering, legacy preference order), and
the three built-in providers (KimiServiceProvider, DDGSProvider,
LocalTrafilaturaProvider).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import aiohttp
import pytest

from kimi_cli.config import Config
from kimi_cli.tools.web.providers import (
    DDGSProvider,
    KimiServiceProvider,
    LocalTrafilaturaProvider,
    WebSearchProvider,
    _reset_for_tests,
    get_active_extract_provider,
    get_active_search_provider,
    get_provider,
    list_providers,
    register_provider,
)
from tests.conftest import tool_call_context


class FakeProvider(WebSearchProvider):
    """Configurable fake provider for registry/resolution tests."""

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
        self.search_calls: list[tuple[Any, ...]] = []
        self.extract_calls: list[tuple[Any, ...]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def supports_search(self) -> bool:
        return self._search

    def supports_extract(self) -> bool:
        return self._extract

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append((query, limit, kwargs))
        return {"success": True, "data": {"web": []}}

    def extract(self, urls: list[str], format: str | None = None) -> list[dict[str, Any]]:  # noqa: A002
        self.extract_calls.append((urls, format))
        return [{"url": u, "title": "", "content": "", "raw_content": "", "error": None} for u in urls]


# ---------------------------------------------------------------------------
# Fake aiohttp primitives for KimiServiceProvider / LocalTrafilaturaProvider
# ---------------------------------------------------------------------------


class _KimiResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload


class _KimiPostContext:
    def __init__(self, response: _KimiResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _KimiResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _KimiSession:
    def __init__(self, response: _KimiResponse, *, fail: str | None = None) -> None:
        self._response = response
        self._fail = fail
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _KimiSession:
        if self._fail == "timeout":
            raise TimeoutError("timed out")
        if self._fail == "network":
            raise aiohttp.ClientError("connection refused")
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _KimiPostContext:
        self.post_calls.append((url, kwargs))
        return _KimiPostContext(self._response)


class _ExtractResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text


class _ExtractGetContext:
    def __init__(self, response: _ExtractResponse | None, *, fail: str | None = None) -> None:
        self._response = response
        self._fail = fail

    async def __aenter__(self) -> _ExtractResponse:
        if self._fail:
            raise aiohttp.ClientError(self._fail)
        assert self._response is not None
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _ExtractSession:
    def __init__(
        self,
        responses: dict[str, _ExtractResponse],
        fail_urls: set[str] | None = None,
    ) -> None:
        self._responses = responses
        self._fail_urls = fail_urls or set()

    async def __aenter__(self) -> _ExtractSession:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def get(self, url: str, **kwargs: Any) -> _ExtractGetContext:
        if url in self._fail_urls:
            return _ExtractGetContext(None, fail=f"boom for {url}")
        return _ExtractGetContext(self._responses[url])


@pytest.fixture(autouse=True)
def _registry_isolation() -> Any:
    """Save the built-in registry around each test."""
    before = list(list_providers())
    _reset_for_tests()
    yield
    _reset_for_tests()
    for provider in before:
        register_provider(provider)


class TestRegistry:
    def test_register_get_list(self) -> None:
        fake = FakeProvider("fake-a")
        register_provider(fake)
        assert get_provider("fake-a") is fake
        assert fake in list_providers()

    def test_duplicate_register_overwrites(self) -> None:
        first = FakeProvider("dup")
        second = FakeProvider("dup")
        register_provider(first)
        register_provider(second)
        assert get_provider("dup") is second
        assert len(list_providers()) == 1

    def test_register_non_provider_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="WebSearchProvider"):
            register_provider("not-a-provider")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_name", ["", "   "])
    def test_register_empty_name_raises_value_error(self, bad_name: str) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            register_provider(FakeProvider(bad_name))

    def test_get_provider_unknown_returns_none(self) -> None:
        assert get_provider("nope") is None

    def test_reset_for_tests_clears_registry(self) -> None:
        register_provider(FakeProvider("temp"))
        _reset_for_tests()
        assert get_provider("temp") is None
        assert list_providers() == []

    def test_get_provider_non_string_returns_none(self) -> None:
        assert get_provider(None) is None  # type: ignore[arg-type]


class TestActiveSelection:
    def test_explicit_config_wins_despite_unavailable(self) -> None:
        ddgs = FakeProvider("ddgs", available=False)
        register_provider(ddgs)
        register_provider(FakeProvider("other", available=True))
        config = Config()
        config.web.search_backend = "ddgs"
        assert get_active_search_provider(config) is ddgs

    def test_no_config_single_available_provider(self) -> None:
        only = FakeProvider("only", available=True)
        register_provider(only)
        assert get_active_search_provider(Config()) is only

    def test_search_legacy_preference_kimi_then_ddgs_then_local(self) -> None:
        kimi = FakeProvider("kimi", available=True)
        ddgs = FakeProvider("ddgs", available=True)
        local = FakeProvider("local", available=True, search=True)
        register_provider(kimi)
        register_provider(ddgs)
        register_provider(local)
        assert get_active_search_provider(Config()) is kimi

        kimi._available = False
        assert get_active_search_provider(Config()) is ddgs

        ddgs._available = False
        assert get_active_search_provider(Config()) is local

    def test_search_no_available_provider_returns_none(self) -> None:
        register_provider(FakeProvider("kimi", available=False))
        register_provider(FakeProvider("ddgs", available=False))
        assert get_active_search_provider(Config()) is None

    def test_extract_legacy_preference_local_then_kimi(self) -> None:
        local = FakeProvider("local", available=True, extract=True, search=False)
        kimi = FakeProvider("kimi", available=True, extract=True)
        register_provider(local)
        register_provider(kimi)
        assert get_active_extract_provider(Config()) is local

        local._available = False
        assert get_active_extract_provider(Config()) is kimi

    def test_extract_filters_by_capability(self) -> None:
        search_only = FakeProvider("kimi", available=True, search=True, extract=False)
        register_provider(search_only)
        assert get_active_extract_provider(Config()) is None

    def test_configured_name_not_registered_falls_back(self) -> None:
        local = FakeProvider("local", available=True, extract=True, search=False)
        register_provider(local)
        config = Config()
        config.web.extract_backend = "not-registered"
        assert get_active_extract_provider(config) is local


class TestKimiServiceProvider:
    def test_is_available_false_without_search_service(self) -> None:
        provider = KimiServiceProvider(Config(), runtime=None)  # type: ignore[arg-type]
        assert provider.is_available() is False

    def test_is_available_true_with_search_service(self, config: Config, runtime: Any) -> None:
        provider = KimiServiceProvider(config, runtime)
        assert provider.is_available() is True

    def test_name_and_capabilities(self, config: Config, runtime: Any) -> None:
        provider = KimiServiceProvider(config, runtime)
        assert provider.name == "kimi"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    @staticmethod
    def _payload() -> dict[str, Any]:
        return {
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

    async def test_search_success_default(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(200, self._payload()))
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello", 5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 1
        item = web[0]
        assert item["title"] == "t"
        assert item["url"] == "u"
        assert item["description"] == "sni"
        assert item["position"] == 1
        assert item["content"] == "c"
        assert item["date"] == "d"
        assert item["site_name"] == "s"
        assert item["icon"] == "i"
        assert item["mime"] == "m"

        assert len(session.post_calls) == 1
        _url, kwargs = session.post_calls[0]
        svc = config.services.search
        assert svc is not None
        assert _url == svc.base_url
        assert kwargs["json"]["text_query"] == "hello"
        assert kwargs["json"]["limit"] == 5
        assert kwargs["json"]["enable_page_crawling"] is False

    async def test_search_success_include_content(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(200, self._payload()))
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello", 3, include_content=True)

        assert result["success"] is True
        _url, kwargs = session.post_calls[0]
        assert kwargs["json"]["enable_page_crawling"] is True

    async def test_search_http_error(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(500, {}))
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "Status: 500" in result["error"]

    async def test_search_timeout(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(200, {}), fail="timeout")
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "timed out" in result["error"]

    async def test_search_network_error(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(200, {}), fail="network")
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "connection refused" in result["error"]

    async def test_search_response_parse_error(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(config, runtime)
        session = _KimiSession(_KimiResponse(200, {}))
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "Failed to parse search results" in result["error"]

    async def test_search_not_configured(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = KimiServiceProvider(Config(), runtime)
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: None)

        with tool_call_context("SearchWeb"):
            result = await provider.search("hello")

        assert result["success"] is False
        assert "not configured" in result["error"]


class TestDDGSProvider:
    def _inject_fake_ddgs(
        self, monkeypatch: pytest.MonkeyPatch, hits: list[dict[str, str]]
    ) -> None:
        class FakeDDGS:
            def __init__(self, timeout: int = 10) -> None:
                self.timeout = timeout

            def __enter__(self) -> FakeDDGS:
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
                return hits

        module = types.ModuleType("ddgs")
        module.DDGS = FakeDDGS  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ddgs", module)

    @staticmethod
    def _remove_fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)

    def test_is_available_true_when_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._inject_fake_ddgs(monkeypatch, [])
        assert DDGSProvider().is_available() is True

    def test_is_available_false_when_not_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._remove_fake_ddgs(monkeypatch)
        assert DDGSProvider().is_available() is False

    def test_name_and_capabilities(self) -> None:
        provider = DDGSProvider()
        assert provider.name == "ddgs"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    def test_search_normalizes_and_caps_at_safe_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [
            {"href": "https://a.example", "title": "A", "body": "body A"},
            {"url": "https://b.example", "title": "B", "body": "body B"},
            {"href": "https://c.example", "title": "C", "body": "body C"},
            {"href": "https://d.example", "title": "D", "body": "body D"},
        ]
        self._inject_fake_ddgs(monkeypatch, hits)

        result = DDGSProvider().search("hello", 3)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {
            "title": "A",
            "url": "https://a.example",
            "description": "body A",
            "position": 1,
        }
        # Fallback to the "url" key when "href" is absent.
        assert web[1]["url"] == "https://b.example"
        # Fourth hit is beyond the cap.
        assert all(item["url"] != "https://d.example" for item in web)

    def test_search_limit_zero_clamped_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._inject_fake_ddgs(
            monkeypatch,
            [{"href": "https://a.example", "title": "A", "body": "body A"}],
        )
        result = DDGSProvider().search("hello", 0)
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1

    def test_search_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._remove_fake_ddgs(monkeypatch)
        result = DDGSProvider().search("hello")
        assert result["success"] is False
        assert "pip install ddgs" in result["error"]


class TestLocalTrafilaturaProvider:
    _HTML = (
        "<html><head><title>Page</title></head><body><article>"
        "<h1>Hello</h1><p>World content here.</p></article></body></html>"
    )

    def test_name_capabilities_and_availability(self) -> None:
        provider = LocalTrafilaturaProvider()
        assert provider.name == "local"
        assert provider.supports_search() is False
        assert provider.supports_extract() is True
        assert provider.is_available() is True

    async def test_extract_returns_content_in_input_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = _ExtractSession(
            {
                "https://a.example": _ExtractResponse(self._HTML),
                "https://b.example": _ExtractResponse(self._HTML),
            }
        )
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        results = await LocalTrafilaturaProvider().extract(
            ["https://a.example", "https://b.example"]
        )

        assert [r["url"] for r in results] == ["https://a.example", "https://b.example"]
        assert all(r["error"] is None for r in results)
        assert "Hello" in results[0]["content"]
        assert "World content here." in results[0]["content"]

    async def test_extract_per_url_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _ExtractSession(
            {
                "https://a.example": _ExtractResponse(self._HTML),
                "https://b.example": _ExtractResponse(self._HTML),
            },
            fail_urls={"https://b.example"},
        )
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        results = await LocalTrafilaturaProvider().extract(
            ["https://a.example", "https://b.example"]
        )

        assert [r["url"] for r in results] == ["https://a.example", "https://b.example"]
        assert results[0]["error"] is None
        assert "Hello" in results[0]["content"]
        assert results[1]["error"] == "boom for https://b.example"
        assert results[1]["content"] == ""

    async def test_extract_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _ExtractSession(
            {
                "https://a.example": _ExtractResponse("", status=404),
            }
        )
        monkeypatch.setattr("kimi_cli.tools.web.search.new_client_session", lambda **kw: session)

        results = await LocalTrafilaturaProvider().extract(["https://a.example"])
        assert results[0]["error"] == "HTTP 404 error"


def test_module_exports_expected_providers() -> None:
    # The isolation fixture clears the registry; re-register the built-ins to
    # verify the public exports resolve to the expected classes.
    register_provider(DDGSProvider())
    register_provider(LocalTrafilaturaProvider())
    assert isinstance(get_provider("local"), LocalTrafilaturaProvider)
    assert isinstance(get_provider("ddgs"), DDGSProvider)
    assert {p.name for p in list_providers()} >= {"local", "ddgs"}
