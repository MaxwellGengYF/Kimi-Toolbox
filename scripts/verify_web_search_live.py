"""Standalone LIVE verification for the web search/fetch service fix.

Runs real network calls (no pytest, no mocks) against the Kimi services to
verify, from a separate process (the already-running agent cannot pick up
code changes):

1. ``D:/k3-256.json`` loads via ``kimix.utils.config._create_config`` and
   carries the FIXED service URLs (``https://api.kimi.com/coding/v1/search``
   and ``.../fetch``).
2. ``KimiServiceProvider.search`` succeeds against the live search service.
3. The HTTP-404 self-heal retry works: with a deliberately stale
   ``services.search.base_url`` (``https://api.moonshot.cn/v1/search``, which
   returns 404) the provider falls back to ``<provider.base_url>/search`` and
   still succeeds.
4. The fetch service path (``fetch_url._fetch_with_service``) fetches
   ``https://example.com`` through the live fetch service; plus a bonus check
   that the same 404 self-heal works for the fetch service.
5. The provider registry self-registers the 7 newly ported providers
   (tavily, exa, brave-free, searxng, firecrawl, parallel, xai) plus the
   pre-existing ddgs/local ones at import time.

Usage (from the repo root, Windows paths with backslashes)::

    cd D:\\kimi-agent
    uv run python scripts\\verify_web_search_live.py

Exit code is 0 when every check passes, 1 otherwise. The real API key is
never printed (masked to its first 12 characters).
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orjson

CONFIG_PATH = Path(r"D:\k3-256.json")

EXPECTED_SEARCH_URL = "https://api.kimi.com/coding/v1/search"
EXPECTED_FETCH_URL = "https://api.kimi.com/coding/v1/fetch"
EXPECTED_PROVIDER_BASE_URL = "https://api.kimi.com/coding/v1"
BROKEN_SEARCH_URL = "https://api.moonshot.cn/v1/search"  # returns HTTP 404
BROKEN_FETCH_URL = "https://api.moonshot.cn/v1/fetch"  # returns HTTP 404

SEARCH_QUERY = "OpenAI GPT model latest news"
FETCH_URL = "https://example.com"
# The live fetch service returns example.com's main extracted text; the <h1>
# "Example Domain" heading is usually stripped by extraction, so accept the
# distinctive body sentence as evidence of a real fetch too.
EXAMPLE_COM_MARKERS = (
    "Example Domain",
    "This domain is for use in documentation examples",
)

EXPECTED_NEW_PROVIDERS = {
    "tavily",
    "exa",
    "brave-free",
    "searxng",
    "firecrawl",
    "parallel",
    "xai",
}
EXPECTED_PREEXISTING_PROVIDERS = {"ddgs", "local"}

NETWORK_RETRIES = 2  # retry a failing network check up to this many times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CheckFailed(AssertionError):
    """Raised when a verification check fails."""


def mask_secret(value: str) -> str:
    """Mask a secret: first 12 chars + ellipsis (never print the full key)."""
    if not value:
        return "<empty>"
    return value[:12] + "..."


@contextmanager
def tool_call_context(tool_name: str):
    """Minimal equivalent of tests/conftest.py's ``tool_call_context`` fixture.

    Sets the ``current_tool_call`` ContextVar that
    ``kimi_cli.soul.toolset.get_current_tool_call_or_none`` reads; both
    ``KimiServiceProvider.search`` and ``fetch_url._fetch_with_service``
    assert it is not None.
    """
    from kimi_cli.soul.toolset import current_tool_call
    from kimi_cli.wire.types import ToolCall

    token = current_tool_call.set(
        ToolCall(
            id="verify-live",
            function=ToolCall.FunctionBody(name=tool_name, arguments=None),
        )
    )
    try:
        yield
    finally:
        current_tool_call.reset(token)


def _note_fake_ip_sandbox(url: str) -> None:
    """Print DNS evidence for the KIMI_ALLOW_PRIVATE_URLS override (see main).

    This verification sandbox intercepts DNS with a fake-IP transparent proxy:
    every public hostname resolves into 198.18.0.0/15 (RFC 2544 benchmarking
    range), which ``ipaddress`` reports as ``is_private=True``. The SSRF guard
    in ``kimi_cli.tools.web.url_safety`` therefore blocks ANY public URL here.
    That is an environment artifact, not a defect of the code under test, so
    the documented ``KIMI_ALLOW_PRIVATE_URLS`` escape hatch (which keeps the
    cloud-metadata block intact) is enabled for the fetch checks.
    """
    import socket
    from urllib.parse import urlsplit

    hostname = urlsplit(url).hostname or ""
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        ips = sorted({str(info[-1][0]) for info in infos})
    except OSError as exc:
        ips = [f"<DNS error: {exc}>"]
    print(f"    DNS: {hostname} -> {ips} (fake-IP sandbox proxy)")
    print("    KIMI_ALLOW_PRIVATE_URLS=1 active (documented url_safety escape hatch)")


class RuntimeStub:
    """Minimal stand-in for ``kimi_cli.soul.agent.Runtime``.

    The code under test only touches ``runtime.oauth.resolve_api_key`` /
    ``runtime.oauth.common_headers`` (search + fetch) and
    ``runtime.config.provider.base_url`` (fetch 404 self-heal), so a tiny
    object holding the real ``Config`` and a real ``OAuthManager`` (safe to
    construct here: the config carries no OAuth refs, so its storage
    migration/loading is a no-op) is sufficient.
    """

    def __init__(self, config: Any) -> None:
        from kimi_cli.auth.oauth import OAuthManager

        self.config = config
        self.oauth = OAuthManager(config)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_config_load() -> tuple[Any, dict[str, Any]]:
    """Check 1: D:/k3-256.json loads with the FIXED service URLs."""
    from kimix.utils.config import _create_config

    if not CONFIG_PATH.exists():
        raise CheckFailed(f"Config file not found: {CONFIG_PATH}")
    raw = orjson.loads(CONFIG_PATH.read_bytes())
    cfg, _ = _create_config(raw)

    search_url = cfg.services.search.base_url
    fetch_url = cfg.services.fetch.base_url
    provider_url = cfg.provider.base_url
    key = cfg.services.search.api_key.get_secret_value()

    print(f"    services.search.base_url = {search_url}")
    print(f"    services.fetch.base_url  = {fetch_url}")
    print(f"    provider.base_url        = {provider_url}")
    print(f"    api_key (masked)         = {mask_secret(key)}")

    if search_url != EXPECTED_SEARCH_URL:
        raise CheckFailed(
            f"services.search.base_url is {search_url!r}, expected {EXPECTED_SEARCH_URL!r}"
        )
    if fetch_url != EXPECTED_FETCH_URL:
        raise CheckFailed(
            f"services.fetch.base_url is {fetch_url!r}, expected {EXPECTED_FETCH_URL!r}"
        )
    if provider_url != EXPECTED_PROVIDER_BASE_URL:
        raise CheckFailed(
            f"provider.base_url is {provider_url!r}, "
            f"expected {EXPECTED_PROVIDER_BASE_URL!r} (needed for the 404 fallback)"
        )
    return cfg, raw


async def check_live_search(cfg: Any) -> None:
    """Check 2: live search through KimiServiceProvider with the fixed URL."""
    from kimi_cli.tools.web.providers import KimiServiceProvider

    provider = KimiServiceProvider(cfg, RuntimeStub(cfg))
    with tool_call_context("web_search"):
        result = await provider.search(SEARCH_QUERY, 3)

    if not isinstance(result, dict) or result.get("success") is not True:
        raise CheckFailed(f"search did not succeed: {str(result)[:500]}")
    web = result["data"]["web"]
    if len(web) < 1:
        raise CheckFailed("search succeeded but returned 0 web results")
    first = web[0]
    print(f"    query   = {SEARCH_QUERY!r} (limit=3)")
    print(f"    results = {len(web)}")
    print(f"    first.title = {first.get('title', '')!r}")
    print(f"    first.url   = {first.get('url', '')!r}")


async def check_search_404_self_heal(raw: dict[str, Any]) -> None:
    """Check 3: stale service URL (404) self-heals via the derived URL."""
    from kimix.utils.config import _create_config
    from kimi_cli.tools.web.providers import KimiServiceProvider

    patched = orjson.loads(orjson.dumps(raw))  # deep copy of plain JSON data
    patched["services"]["search"]["base_url"] = BROKEN_SEARCH_URL
    broken_cfg, _ = _create_config(patched)

    svc_url = broken_cfg.services.search.base_url
    provider_url = broken_cfg.provider.base_url
    print(f"    stale services.search.base_url = {svc_url}")
    print(f"    provider.base_url (fallback)   = {provider_url}")
    if svc_url != BROKEN_SEARCH_URL:
        raise CheckFailed(f"test setup wrong: stale url is {svc_url!r}")
    if provider_url != EXPECTED_PROVIDER_BASE_URL:
        raise CheckFailed(f"test setup wrong: provider url is {provider_url!r}")

    provider = KimiServiceProvider(broken_cfg, RuntimeStub(broken_cfg))
    with tool_call_context("web_search"):
        result = await provider.search(SEARCH_QUERY, 3)

    # The stale URL returns HTTP 404; success here is only possible when the
    # code retried once against `<provider.base_url>/search`.
    if not isinstance(result, dict) or result.get("success") is not True:
        raise CheckFailed(
            "search with a stale (404) service URL did NOT self-heal: "
            f"{str(result)[:500]}"
        )
    web = result["data"]["web"]
    if len(web) < 1:
        raise CheckFailed("self-healed search succeeded but returned 0 web results")
    print(f"    self-heal OK: {len(web)} results after 404 retry")
    print(f"    first.title = {web[0].get('title', '')!r}")
    print(f"    first.url   = {web[0].get('url', '')!r}")


async def check_fetch_service(cfg: Any) -> None:
    """Check 4: live fetch of example.com through the fetch service path."""
    from kimi_cli.tools.web.fetch import Params, fetch_url

    _note_fake_ip_sandbox(FETCH_URL)
    tool = fetch_url(cfg, RuntimeStub(cfg))
    with tool_call_context("fetch_url"):
        ret = await tool._fetch_with_service(Params(url=FETCH_URL))

    body = f"{ret.output}\n{ret.message}"
    print(f"    is_error = {ret.is_error}")
    print(f"    message  = {ret.message!r}")
    if ret.is_error:
        raise CheckFailed(f"fetch service returned an error: {ret.message!r}")
    marker = next((m for m in EXAMPLE_COM_MARKERS if m in body), None)
    if marker is None:
        raise CheckFailed(
            f"fetched content matches no example.com marker "
            f"{EXAMPLE_COM_MARKERS}: {body[:300]!r}"
        )
    snippet = next(
        (line.strip() for line in ret.output.splitlines() if marker in line),
        "",
    )
    print(f"    evidence (matched {marker!r}): {snippet!r}")


async def check_fetch_404_self_heal(raw: dict[str, Any]) -> None:
    """Check 4b (bonus): stale fetch URL (404) self-heals via derived URL."""
    from kimix.utils.config import _create_config
    from kimi_cli.tools.web.fetch import Params, fetch_url

    patched = orjson.loads(orjson.dumps(raw))
    patched["services"]["fetch"]["base_url"] = BROKEN_FETCH_URL
    broken_cfg, _ = _create_config(patched)

    svc_url = broken_cfg.services.fetch.base_url
    provider_url = broken_cfg.provider.base_url
    print(f"    stale services.fetch.base_url = {svc_url}")
    print(f"    provider.base_url (fallback)  = {provider_url}")
    if svc_url != BROKEN_FETCH_URL:
        raise CheckFailed(f"test setup wrong: stale url is {svc_url!r}")

    tool = fetch_url(broken_cfg, RuntimeStub(broken_cfg))
    with tool_call_context("fetch_url"):
        ret = await tool._fetch_with_service(Params(url=FETCH_URL))

    if ret.is_error:
        raise CheckFailed(
            "fetch with a stale (404) service URL did NOT self-heal: "
            f"{ret.message!r}"
        )
    body = f"{ret.output}\n{ret.message}"
    marker = next((m for m in EXAMPLE_COM_MARKERS if m in body), None)
    if marker is None:
        raise CheckFailed(
            f"self-healed fetch content matches no example.com marker: {body[:300]!r}"
        )
    print(f"    self-heal OK: fetch succeeded after 404 retry (matched {marker!r})")


def check_provider_registry() -> None:
    """Check 5: the 7 new providers + ddgs/local self-register at import."""
    from kimi_cli.tools.web import providers  # noqa: F401 — import triggers registration
    from kimi_cli.tools.web.providers import list_providers

    names = sorted(p.name for p in list_providers())
    print(f"    registered providers: {names}")
    missing_new = EXPECTED_NEW_PROVIDERS - set(names)
    missing_old = EXPECTED_PREEXISTING_PROVIDERS - set(names)
    if missing_new:
        raise CheckFailed(f"new providers missing from registry: {sorted(missing_new)}")
    if missing_old:
        raise CheckFailed(f"pre-existing providers missing: {sorted(missing_old)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CheckFn = Callable[..., Any]


async def _run_one(
    name: str,
    fn: Callable[..., Coroutine[Any, Any, Any] | Any],
    *args: Any,
    retries: int = 0,
) -> bool:
    """Run one check, retrying network checks up to ``retries`` extra times."""
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            outcome = fn(*args)
            if isinstance(outcome, Coroutine):
                outcome = await outcome
            print(f"[PASS] {name}")
            return True
        except Exception as exc:  # noqa: BLE001 — report any failure as check failure
            if attempt == attempts:
                print(f"[FAIL] {name}")
                print(f"       {type(exc).__name__}: {exc}")
                if not isinstance(exc, CheckFailed):
                    traceback.print_exc(limit=5)
                return False
            # Network checks may flake (timeouts); retry before declaring failure.
            print(
                f"       attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}: {exc}); retrying..."
            )
    return False  # unreachable, keeps type checkers happy


async def amain() -> int:
    print("=" * 72)
    print("Standalone LIVE verification: web search/fetch service fix")
    print(f"config: {CONFIG_PATH}")
    print("=" * 72)

    results: list[tuple[str, bool]] = []

    # Check 1 (offline): config load. Hard prerequisite for the live checks.
    print("\n--- Check 1: config load (fixed service URLs) ---")
    try:
        cfg, raw = check_config_load()
        print("[PASS] Check 1: config load")
        results.append(("Check 1: config load", True))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Check 1: config load\n       {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=5)
        results.append(("Check 1: config load", False))
        return _summary(results)

    print("\n--- Check 2: live search via KimiServiceProvider ---")
    ok = await _run_one(
        "Check 2: live search", check_live_search, cfg, retries=NETWORK_RETRIES
    )
    results.append(("Check 2: live search", ok))

    print("\n--- Check 3: live 404 self-heal fallback (search) ---")
    ok = await _run_one(
        "Check 3: search 404 self-heal",
        check_search_404_self_heal,
        raw,
        retries=NETWORK_RETRIES,
    )
    results.append(("Check 3: search 404 self-heal", ok))

    print("\n--- Check 4: live fetch service (example.com) ---")
    ok = await _run_one(
        "Check 4: fetch service", check_fetch_service, cfg, retries=NETWORK_RETRIES
    )
    results.append(("Check 4: fetch service", ok))

    print("\n--- Check 4b: live 404 self-heal fallback (fetch, bonus) ---")
    ok = await _run_one(
        "Check 4b: fetch 404 self-heal",
        check_fetch_404_self_heal,
        raw,
        retries=NETWORK_RETRIES,
    )
    results.append(("Check 4b: fetch 404 self-heal", ok))

    print("\n--- Check 5: provider registry ---")
    ok = await _run_one("Check 5: provider registry", check_provider_registry)
    results.append(("Check 5: provider registry", ok))

    return _summary(results)


def _summary(results: list[tuple[str, bool]]) -> int:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS")
    return 0


def main() -> int:
    # This sandbox resolves every public hostname into the fake-IP range
    # 198.18.0.0/15 (transparent proxy), which the SSRF guard in
    # kimi_cli.tools.web.url_safety treats as private and blocks. Enable the
    # documented escape hatch so the live fetch path can run; the
    # cloud-metadata block (169.254.0.0/16 etc.) stays enforced regardless.
    os.environ.setdefault("KIMI_ALLOW_PRIVATE_URLS", "1")
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
