"""Tests for the ``web_extract`` tool and its content-processing helpers.

Covers the whole-call secret/sensitive-param blocks, per-index invalid/SSRF
inline errors, fake-provider multi-URL extraction, empty-result handling, and
the content helpers (base64 image conversion, truncation footer, full-text
storage, char-limit resolution).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import orjson
import pytest

from kimi_cli.config import Config, get_default_config
from kimi_cli.tools.web.content import (
    DEFAULT_EXTRACT_CHAR_LIMIT,
    convert_base64_images_to_links,
    get_extract_char_limit,
    store_full_text,
    truncate_with_footer,
)
from kimi_cli.tools.web.extract import Params, WebExtract
from kimi_cli.tools.web.providers import (
    DDGSProvider,
    LocalTrafilaturaProvider,
    WebSearchProvider,
    _reset_for_tests,
    register_provider,
)
from tests.conftest import tool_call_context


class FakeExtractProvider(WebSearchProvider):
    """Registered fake extract backend with deterministic results."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results
        self.called_with: list[list[str]] = []

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: list[str], format: str | None = None) -> list[dict[str, Any]]:  # noqa: A002
        self.called_with.append(list(urls))
        if self._results is not None:
            return list(self._results)
        return [
            {
                "url": url,
                "title": f"Title {i}",
                "content": f"Content {i}",
                "raw_content": f"Content {i}",
                "error": None,
            }
            for i, url in enumerate(urls)
        ]


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


@pytest.fixture
def extract_tool(runtime: Any) -> WebExtract:
    """Build a WebExtract tool against an empty default config."""
    return WebExtract(get_default_config(), runtime)


class TestWebExtractBlocks:
    async def test_secret_url_blocks_whole_call(self, extract_tool: WebExtract) -> None:
        with tool_call_context("WebExtract"):
            result = await extract_tool(
                Params(urls=["https://example.com/?q=sk-ABC1234567890"])
            )
        assert result.is_error
        assert "API key or token" in result.message

    async def test_sensitive_query_param_blocks_whole_call(
        self, extract_tool: WebExtract
    ) -> None:
        with tool_call_context("WebExtract"):
            result = await extract_tool(Params(urls=["https://example.com/?token=abc"]))
        assert result.is_error
        assert "token" in result.message

    async def test_ssrf_blocked_url_gets_inline_error(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_default_config()
        config.web.extract_backend = "fake"
        fake = FakeExtractProvider()
        register_provider(fake)

        async def _safe(url: str) -> bool:
            return "127.0.0.1" not in url

        monkeypatch.setattr("kimi_cli.tools.web.extract.async_is_safe_url", _safe)

        tool = WebExtract(config, runtime)
        with tool_call_context("WebExtract"):
            result = await tool(
                Params(urls=["http://127.0.0.1/", "https://example.com/"])
            )

        assert not result.is_error
        payload = orjson.loads(result.output)  # type: ignore[arg-type]
        assert payload["results"][0]["error"] == (
            "Blocked: URL targets a private or internal network address"
        )
        assert payload["results"][0]["content"] == ""
        # The safe URL still extracts through the fake provider.
        assert payload["results"][1]["error"] is None
        assert payload["results"][1]["content"] == "Content 0"
        assert fake.called_with == [["https://example.com/"]]

    async def test_invalid_input_item_gets_inline_error(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_default_config()
        config.web.extract_backend = "fake"
        fake = FakeExtractProvider()
        register_provider(fake)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.extract.async_is_safe_url", _safe)

        tool = WebExtract(config, runtime)
        with tool_call_context("WebExtract"):
            result = await tool(Params(urls=[123, {"href": "https://example.com/"}]))  # type: ignore[list-item]

        assert not result.is_error
        payload = orjson.loads(result.output)  # type: ignore[arg-type]
        assert "Invalid URL item at index 0" in payload["results"][0]["error"]
        assert payload["results"][1]["error"] is None
        assert payload["results"][1]["content"] == "Content 0"
        assert fake.called_with == [["https://example.com/"]]


class TestWebExtractMultiUrl:
    async def test_multi_url_extraction_trims_fields_and_preserves_order(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_default_config()
        config.web.extract_backend = "fake"
        fake = FakeExtractProvider()
        register_provider(fake)

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.extract.async_is_safe_url", _safe)

        tool = WebExtract(config, runtime)
        with tool_call_context("WebExtract"):
            result = await tool(
                Params(
                    urls=[
                        "https://a.example/",
                        {"url": "https://b.example/"},
                        {"href": "https://c.example/"},
                    ]
                )
            )

        assert not result.is_error
        payload = orjson.loads(result.output)  # type: ignore[arg-type]
        assert [item["url"] for item in payload["results"]] == [
            "https://a.example/",
            "https://b.example/",
            "https://c.example/",
        ]
        for item in payload["results"]:
            assert set(item.keys()) == {"url", "title", "content", "error"}
            assert "raw_content" not in item
        assert payload["results"][0]["title"] == "Title 0"
        assert payload["results"][1]["content"] == "Content 1"
        assert fake.called_with == [
            ["https://a.example/", "https://b.example/", "https://c.example/"]
        ]

    async def test_empty_results_error(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_default_config()
        config.web.extract_backend = "fake"
        register_provider(FakeExtractProvider(results=[]))

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.extract.async_is_safe_url", _safe)

        tool = WebExtract(config, runtime)
        with tool_call_context("WebExtract"):
            result = await tool(Params(urls=["https://example.com/"]))

        assert result.is_error
        assert result.message == "Content was inaccessible or not found"

    async def test_never_raises_skip_this_tool(self, runtime: Any) -> None:
        tool = WebExtract(Config(), runtime)
        assert tool.name == "web_extract"

    async def test_truncates_large_content_via_fake_provider(
        self, runtime: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        config = get_default_config()
        config.web.extract_backend = "fake"
        big = ("lorem ipsum dolor sit amet " * 1000)  # ~26k chars
        register_provider(
            FakeExtractProvider(
                results=[
                    {
                        "url": "https://example.com/",
                        "title": "Big",
                        "content": big,
                        "raw_content": big,
                        "error": None,
                    }
                ]
            )
        )

        async def _safe(url: str) -> bool:
            return True

        monkeypatch.setattr("kimi_cli.tools.web.extract.async_is_safe_url", _safe)
        with patch("kimi_cli.config.get_share_dir", return_value=tmp_path):
            tool = WebExtract(config, runtime)
            with tool_call_context("WebExtract"):
                result = await tool(Params(urls=["https://example.com/"], char_limit=2000))

        assert not result.is_error
        payload = orjson.loads(result.output)  # type: ignore[arg-type]
        content = payload["results"][0]["content"]
        assert "[TRUNCATED]" in content
        assert "read_file path=" in content


class TestConvertBase64ImagesToLinks:
    def test_markdown_base64_becomes_labeled_placeholder(self) -> None:
        text = "before ![alt text](data:image/png;base64,AAAA) after"
        assert convert_base64_images_to_links(text) == "before [IMAGE: alt text] after"

    def test_markdown_base64_without_alt(self) -> None:
        text = "![ ](data:image/png;base64,AAAA)"
        assert convert_base64_images_to_links(text) == "[IMAGE]"

    def test_bare_base64_becomes_placeholder(self) -> None:
        text = "prefix data:image/png;base64,AAAA suffix"
        assert convert_base64_images_to_links(text) == "prefix [IMAGE] suffix"

    def test_http_image_link_left_intact(self) -> None:
        text = "![alt](https://example.com/img.png)"
        assert convert_base64_images_to_links(text) == text


class TestTruncateWithFooter:
    def test_small_content_unchanged(self) -> None:
        text, truncated = truncate_with_footer("hello world", "https://example.com/", 1000)
        assert text == "hello world"
        assert truncated is False

    def test_large_content_gets_footer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        with patch("kimi_cli.config.get_share_dir", return_value=tmp_path):
            content = ("line of text\n" * 80)
            text, truncated = truncate_with_footer(content, "https://example.com/", 500)
        assert truncated is True
        assert "[TRUNCATED]" in text
        assert "read_file path=" in text
        assert "middle omitted" in text


class TestStoreFullText:
    def test_writes_file_under_cache_web(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        with patch("kimi_cli.config.get_share_dir", return_value=tmp_path):
            path_str = store_full_text("https://example.com/page", "full content here")
        assert path_str is not None
        path = Path(path_str)
        assert path.parent == tmp_path / "cache" / "web"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "full content here"

    def test_returns_absolute_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        with patch("kimi_cli.config.get_share_dir", return_value=tmp_path):
            path_str = store_full_text("https://example.com/", "content")
        assert path_str is not None
        assert str(tmp_path) in path_str


class TestGetExtractCharLimit:
    def test_default(self) -> None:
        assert get_extract_char_limit(Config()) == DEFAULT_EXTRACT_CHAR_LIMIT == 15000

    def test_clamped_low(self) -> None:
        config = Config()
        config.web.extract_char_limit = 100
        assert get_extract_char_limit(config) == 2000

    def test_clamped_high(self) -> None:
        config = Config()
        config.web.extract_char_limit = 999_999
        assert get_extract_char_limit(config) == 500_000
