from __future__ import annotations

import ssl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

# NOTE: ``aiohttp`` (and the SSL context below) are created lazily on first
# use. Importing aiohttp pulls in a sizeable dependency tree (multidict, yarl,
# frozenlist, ...) that costs tens of milliseconds at startup, and it is only
# needed when an HTTP request is actually made.

_ssl_context: ssl.SSLContext | None = None

_DEFAULT_TIMEOUT: aiohttp.ClientTimeout | None = None


def _get_ssl_context() -> ssl.SSLContext:
    global _ssl_context
    if _ssl_context is None:
        import certifi

        _ssl_context = ssl.create_default_context(cafile=certifi.where())
    return _ssl_context


def _get_default_timeout() -> aiohttp.ClientTimeout:
    global _DEFAULT_TIMEOUT
    if _DEFAULT_TIMEOUT is None:
        import aiohttp

        _DEFAULT_TIMEOUT = aiohttp.ClientTimeout(
            total=120,
            sock_read=60,
            sock_connect=15,
        )
    return _DEFAULT_TIMEOUT


def new_client_session(
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> aiohttp.ClientSession:
    import aiohttp

    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=_get_ssl_context()),
        timeout=timeout or _get_default_timeout(),
    )
