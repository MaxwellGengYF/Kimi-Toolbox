"""Tests for the web URL safety hardening (SSRF + credential exfiltration).

Covers ``normalize_url_for_request``, ``sensitive_query_param_name``,
``url_contains_secret``, ``is_safe_url`` / ``async_is_safe_url`` and the
``KIMI_ALLOW_PRIVATE_URLS`` override (which never lifts the cloud-metadata
floor).
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from kimi_cli.tools.web.url_safety import (
    async_is_safe_url,
    is_safe_url,
    normalize_url_for_request,
    sensitive_query_param_name,
    url_contains_secret,
)


def _addr(ip: str) -> list[tuple[Any, ...]]:
    """Return a fake ``socket.getaddrinfo``-style result for *ip*."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


@pytest.fixture
def fake_resolver(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``url_safety.socket.getaddrinfo`` with a controllable stub."""
    state: dict[str, Any] = {}

    def getaddrinfo(host: str, _port: Any, *_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        if state.get("raise_gaierror"):
            raise socket.gaierror(f"Name or service not known: {host}")
        return state["result"]

    monkeypatch.setattr(
        "kimi_cli.tools.web.url_safety.socket.getaddrinfo", getaddrinfo
    )
    return state


class TestNormalizeUrlForRequest:
    def test_idna_host_converted_to_ascii(self) -> None:
        assert (
            normalize_url_for_request("https://münchen.de/path")
            == "https://xn--mnchen-3ya.de/path"
        )

    def test_non_ascii_path_percent_encoded(self) -> None:
        assert normalize_url_for_request("https://wttr.in/Köln") == (
            "https://wttr.in/K%C3%B6ln"
        )

    def test_whitespace_after_scheme_repaired(self) -> None:
        assert normalize_url_for_request("https:// docs.example") == "https://docs.example"

    def test_non_http_scheme_passthrough(self) -> None:
        assert normalize_url_for_request("ftp://example.com/file") == "ftp://example.com/file"

    def test_query_whitespace_percent_encoded(self) -> None:
        assert normalize_url_for_request("https://example.com/?q=a b&x=1") == (
            "https://example.com/?q=a%20b&x=1"
        )

    def test_empty_string_passthrough(self) -> None:
        assert normalize_url_for_request("") == ""


class TestSensitiveQueryParamName:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://x.com/?token=abc", "token"),
            ("https://x.com/?api_key=abc", "api_key"),
            ("https://x.com/?q=hello", None),
            ("ftp://x.com/?token=abc", None),
            ("https://x.com/noquery", None),
        ],
    )
    def test_sensitive_query_param_name(self, url: str, expected: str | None) -> None:
        assert sensitive_query_param_name(url) == expected

    def test_empty_value_is_not_flagged(self) -> None:
        # The check only fires on non-empty values to avoid blocking pages that
        # merely reference the parameter name (e.g. <input name="token">).
        assert sensitive_query_param_name("https://x.com/?token=") is None


class TestUrlContainsSecret:
    @pytest.mark.parametrize(
        "url",
        [
            "https://x.com/?k=sk-ABC1234567890",
            "https://x.com/?k=ghp_1234567890",
            "https://x.com/?k=AIzaSy0123456789_abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_known_credential_prefixes_flagged(self, url: str) -> None:
        assert url_contains_secret(url) is True

    def test_plain_url_not_flagged(self) -> None:
        assert url_contains_secret("https://x.com/") is False
        assert url_contains_secret("https://x.com/?q=hello") is False

    def test_percent_encoded_secret_flagged(self) -> None:
        # ``sk-%41BCDEFGHIJKL`` unquotes to ``sk-ABCDEFGHIJKL`` (12 token chars).
        assert url_contains_secret("https://x.com/?k=sk-%41BCDEFGHIJKL") is True

    def test_non_string_returns_false(self) -> None:
        assert url_contains_secret(None) is False  # type: ignore[arg-type]
        assert url_contains_secret(123) is False  # type: ignore[arg-type]


class TestIsSafeUrl:
    def test_public_ip_allowed(self, fake_resolver: dict[str, Any]) -> None:
        fake_resolver["result"] = _addr("93.184.216.34")
        assert is_safe_url("http://93.184.216.34/") is True

    def test_loopback_blocked(self, fake_resolver: dict[str, Any]) -> None:
        fake_resolver["result"] = _addr("127.0.0.1")
        assert is_safe_url("http://127.0.0.1/") is False

    def test_link_local_metadata_blocked_even_with_override(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["result"] = _addr("169.254.169.254")
        monkeypatch.setenv("KIMI_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://169.254.169.254/") is False

    def test_private_ip_blocked_by_default(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["result"] = _addr("10.0.0.1")
        monkeypatch.delenv("KIMI_ALLOW_PRIVATE_URLS", raising=False)
        assert is_safe_url("http://10.0.0.1/") is False

    def test_private_ip_allowed_with_override(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["result"] = _addr("10.0.0.1")
        monkeypatch.setenv("KIMI_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://10.0.0.1/") is True

    def test_dns_failure_fails_closed(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["raise_gaierror"] = True
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("ALL_PROXY", raising=False)
        assert is_safe_url("https://example.com/") is False

    def test_dns_failure_allowed_with_proxy_env(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["raise_gaierror"] = True
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
        # Non-literal-IP host → DNS delegated to the proxy.
        assert is_safe_url("https://example.com/") is True

    def test_dns_failure_with_proxy_still_blocks_literal_ip(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Literal IPs need no DNS; a proxy env must not let them through.
        fake_resolver["raise_gaierror"] = True
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
        assert is_safe_url("http://10.0.0.1/") is False

    def test_unsupported_scheme_blocked(self) -> None:
        assert is_safe_url("ftp://example.com/file") is False

    def test_empty_host_blocked(self) -> None:
        assert is_safe_url("https:///path") is False
        assert is_safe_url("") is False

    def test_blocked_internal_hostname_always_blocked(
        self, fake_resolver: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_resolver["result"] = _addr("93.184.216.34")
        monkeypatch.setenv("KIMI_ALLOW_PRIVATE_URLS", "true")
        assert is_safe_url("http://metadata.google.internal/") is False


class TestAsyncIsSafeUrl:
    @pytest.mark.parametrize("patched_result", [True, False])
    async def test_matches_is_safe_url(
        self, monkeypatch: pytest.MonkeyPatch, patched_result: bool
    ) -> None:
        def _stub(_url: str) -> bool:
            return patched_result

        monkeypatch.setattr("kimi_cli.tools.web.url_safety.is_safe_url", _stub)
        assert await async_is_safe_url("https://example.com/") is patched_result
