"""Tests for the SearchWeb dispatcher and web_extract key-argument extraction.

Covers construction gating (SkipThisTool), provider dispatch through
``get_active_search_provider``, the configured-backend fallback path, and
``extract_key_argument`` URL extraction for ``web_extract``.
"""

from __future__ import annotations

from typing import Any

import pytest

from kimi_cli.config import Config
from kimi_cli.tools import SkipThisTool, extract_key_argument
from kimi_cli.tools.web.providers import (
    DDGSProvider,
    LocalTrafilaturaProvider,
    WebSearchProvider,
    _reset_for_tests,
    register_provider,
)
from kimi_cli.tools.web.search import Params, SearchWeb


class FakeSearchProvider(WebSearchProvider):
    def __init__(self, name: str = "fake", available: bool = True) -> None:
        self._name = name
        self._available = available
        self.search_calls: list[tuple[Any, ...]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append((query, limit, kwargs))
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Example Title",
                        "url": "https://example.com/",
                        "description": "Example summary",
                        "position": 1,
                    }
                ]
            },
        }


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Reset the provider registry and restore the built-ins around each test."""
    _reset_for_tests()
    register_provider(DDGSProvider())
    register_provider(LocalTrafilaturaProvider())
    yield
    _reset_for_tests()
    register_provider(DDGSProvider())
    register_provider(LocalTrafilaturaProvider())


class TestSearchWebDispatch:
    async def test_constructs_with_search_service_and_dispatches(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = SearchWeb(config, runtime)
        fake = FakeSearchProvider()
        monkeypatch.setattr(
            "kimi_cli.tools.web.providers.get_active_search_provider",
            lambda _config=None: fake,
        )

        result = await tool(Params(query="q"))

        assert not result.is_error
        assert "Title: Example Title" in result.output
        assert "URL: https://example.com/" in result.output
        assert "Summary: Example summary" in result.output
        assert fake.search_calls == [("q", 5, {})]

    def test_config_defaults_raise_skip_this_tool(self, runtime: Any) -> None:
        with pytest.raises(SkipThisTool):
            SearchWeb(Config(), runtime)

    async def test_configured_fake_backend_dispatches(
        self, runtime: Any
    ) -> None:
        fake = FakeSearchProvider(name="fake")
        register_provider(fake)

        config = Config()
        config.web.search_backend = "fake"
        tool = SearchWeb(config, runtime)

        result = await tool(Params(query="hello", limit=3))

        assert not result.is_error
        assert "Example Title" in result.output
        assert fake.search_calls == [("hello", 3, {})]

    async def test_include_content_passed_through(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = SearchWeb(config, runtime)
        fake = FakeSearchProvider()
        monkeypatch.setattr(
            "kimi_cli.tools.web.providers.get_active_search_provider",
            lambda _config=None: fake,
        )

        result = await tool(Params(query="q", include_content=True))

        assert not result.is_error
        assert fake.search_calls == [("q", 5, {"include_content": True})]

    async def test_provider_error_surfaces(
        self, config: Config, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FailingProvider:
            name = "failing"

            def search(self, query: str, limit: int = 5, **kwargs: Any) -> dict[str, Any]:
                return {"success": False, "error": "Search backend exploded"}

        tool = SearchWeb(config, runtime)
        monkeypatch.setattr(
            "kimi_cli.tools.web.providers.get_active_search_provider",
            lambda _config=None: FailingProvider(),
        )

        result = await tool(Params(query="q"))

        assert result.is_error
        assert "Search backend exploded" in result.message


class TestExtractKeyArgument:
    def test_returns_first_url_string(self) -> None:
        result = extract_key_argument(
            '{"urls": ["https://a.example/", "https://b.example/"]}',
            "web_extract",
        )
        assert result == "https://a.example/"

    def test_returns_first_dict_url(self) -> None:
        result = extract_key_argument(
            '{"urls": [{"url": "https://b.example/"}]}',
            "web_extract",
        )
        assert result == "https://b.example/"

    def test_returns_first_dict_href(self) -> None:
        result = extract_key_argument(
            '{"urls": [{"href": "https://c.example/"}]}',
            "web_extract",
        )
        assert result == "https://c.example/"

    def test_missing_urls_falls_back_to_raw_content(self) -> None:
        # No usable URL -> the dispatcher falls back to the raw argument text.
        assert extract_key_argument('{"query": "x"}', "web_extract") == '{"query": "x"}'

    def test_malformed_json_falls_back_to_raw_content(self) -> None:
        # Relaxed JSON parses the blob as a string; no URL found -> raw text.
        assert extract_key_argument("{not json", "web_extract") == "{not json"
