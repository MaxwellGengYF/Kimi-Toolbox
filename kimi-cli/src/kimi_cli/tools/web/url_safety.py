"""URL safety checks — blocks requests to private/internal network addresses.

Ported from the Hermes project's ``tools/url_safety.py`` and
``agent/redact.py`` (credential-prefix patterns). Prevents SSRF (Server-Side
Request Forgery) where a malicious prompt or skill could trick the agent into
fetching internal resources like cloud metadata endpoints (169.254.169.254),
localhost services, or private network hosts.

Cloud metadata hostnames/IPs (metadata.google.internal, 169.254.169.254) are
**always** blocked — they are never legitimate agent targets. The env override
``KIMI_ALLOW_PRIVATE_URLS`` (values ``true``/``1``/``yes``) disables ordinary
private-IP blocking, but the metadata floor still applies.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

import regex as re

from kimi_cli.utils.logging import logger

# ── Proxy detection ──────────────────────────────────────────
# Proxy environment variables that indicate the runtime should delegate DNS to
# a proxy rather than attempting direct resolution.
_PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _proxy_is_configured() -> bool:
    """Return True when at least one HTTP proxy env var is set."""
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


def normalize_url_for_request(url: str) -> str:
    """Return an ASCII-safe HTTP URL for URL tools.

    Browsers and HTTP clients expect URIs, but users and models often provide
    IRIs such as ``https://wttr.in/Köln``. Preserve URL syntax and existing
    percent escapes while encoding non-ASCII host/path/query/fragment text.
    This is intentionally for URL tool inputs only; arbitrary shell commands
    must not be rewritten.
    """
    if not isinstance(url, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return url

    raw = url.strip()
    if not raw:
        return raw

    # Models sometimes emit otherwise valid URLs with whitespace between the
    # scheme separator and authority (``https:// docs.example``). Repairing
    # that position before parsing keeps web tools from failing on a formatting
    # artifact while leaving path/query whitespace to the percent-encoding path.
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    if parsed.scheme.lower() not in {"http", "https"}:
        return raw

    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)

    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    fragment = quote(parsed.fragment, safe="/%:@!$&'()*+,;=?")

    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


# Query parameter names that are unambiguously credential-bearing. Kept
# deliberately narrow: bare English words that double as normal page facets
# (``code``, ``key``, ``auth``, ``session``, ``sig``) are intentionally
# EXCLUDED to avoid blocking ordinary browsing.
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "awsaccesskeyid",
    "client_secret",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "secret",
    "session_id",
    "signature",
    "token",
    "x_amz_security_token",
    "x_amz_signature",
    "x-amz-security-token",
    "x-amz-signature",
})


def sensitive_query_param_name(url: str) -> str | None:
    """Return the first sensitive query parameter name in ``url``, if any."""
    if not isinstance(url, str) or "?" not in url:  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


# Hostnames that should always be blocked regardless of IP resolution or any
# config toggle. These are cloud metadata endpoints that an attacker could use
# to steal instance credentials.
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# IPs and networks that should always be blocked regardless of the
# allow-private toggle. These are cloud metadata / credential endpoints — the
# #1 SSRF target — and the link-local range where they all live.
#
# IPv4-mapped IPv6 variants are included because DNS resolvers may return
# ``::ffff:x.x.x.x`` for IPv4-only hosts, and Python's ipaddress module treats
# these as distinct from the plain IPv4 address.
_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle metadata
    ipaddress.ip_address("169.254.170.2"),  # AWS ECS task metadata (task IAM creds)
    ipaddress.ip_address("169.254.169.253"),  # Azure IMDS wire server
    ipaddress.ip_address("fd00:ec2::254"),  # AWS metadata (IPv6)
    ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud metadata
    # IPv4-mapped IPv6 variants — same endpoints reachable via ::ffff:x.x.x.x
    ipaddress.ip_address("::ffff:169.254.169.254"),
    ipaddress.ip_address("::ffff:169.254.170.2"),
    ipaddress.ip_address("::ffff:169.254.169.253"),
    ipaddress.ip_address("::ffff:100.100.100.200"),
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),  # Entire link-local range
    ipaddress.ip_network("::ffff:169.254.0.0/112"),  # IPv4-mapped link-local range
)

# 100.64.0.0/10 (CGNAT / Shared Address Space, RFC 6598) is NOT covered by
# ipaddress.is_private — it returns False for both is_private and is_global.
# Must be blocked explicitly. Used by carrier-grade NAT, Tailscale/WireGuard
# VPNs, and some cloud internal networks.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _global_allow_private_urls() -> bool:
    """Return True when ``KIMI_ALLOW_PRIVATE_URLS`` opts out of private-IP blocking."""
    return os.getenv("KIMI_ALLOW_PRIVATE_URLS", "").strip().lower() in {
        "true",
        "1",
        "yes",
    }


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    # IPv4-mapped IPv6 addresses (``::ffff:x.x.x.x``) should be checked by their
    # embedded IPv4 address, not as IPv6.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        embedded_ip = ip.ipv4_mapped
        return (
            embedded_ip.is_private
            or embedded_ip.is_loopback
            or embedded_ip.is_link_local
            or embedded_ip.is_reserved
            or embedded_ip.is_multicast
            or embedded_ip.is_unspecified
            or embedded_ip in _CGNAT_NETWORK
        )

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    return ip in _CGNAT_NETWORK


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname to an IP and checks against private ranges. Fails
    closed: DNS errors and unexpected exceptions block the request.

    When ``KIMI_ALLOW_PRIVATE_URLS`` is ``true``/``1``/``yes``, private-IP
    blocking is skipped. Cloud metadata endpoints (169.254.169.254,
    metadata.google.internal) remain blocked regardless.
    """
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in {"http", "https"}:
            logger.warning(
                "Blocked request — unsupported URL scheme: %s", scheme or "<empty>"
            )
            return False
        if not hostname:
            return False

        # Block known internal hostnames — ALWAYS, even with the toggle on.
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        allow_all_private = _global_allow_private_urls()

        try:
            addr_info = socket.getaddrinfo(
                hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except socket.gaierror:
            # DNS resolution failed. In sandbox / proxy environments the host
            # may block direct DNS — only HTTP(S) through the proxy is
            # permitted. When a proxy is configured, delegate DNS to the proxy
            # rather than blocking outright. Literal IPs never qualify — they
            # need no DNS, so keep them on the fail-closed path.
            is_literal_ip = True
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                is_literal_ip = False
            if not is_literal_ip and _proxy_is_configured():
                logger.debug(
                    "DNS resolution failed for %s — proxy configured, "
                    "allowing through for proxy-side resolution",
                    hostname,
                )
                return True
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = str(sockaddr[0])
            if "%" in ip_str:
                ip_str = ip_str.split("%")[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                # Still unparseable after scope ID strip — fail closed.
                logger.warning(
                    "Blocked request — unparseable IP address %r for hostname %s",
                    sockaddr[0],
                    hostname,
                )
                return False

            # Always block cloud metadata IPs and link-local, even with the toggle on.
            if ip in _ALWAYS_BLOCKED_IPS or any(
                ip in net for net in _ALWAYS_BLOCKED_NETWORKS
            ):
                logger.warning(
                    "Blocked request to cloud metadata address: %s -> %s",
                    hostname,
                    ip_str,
                )
                return False

            if not allow_all_private and _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname,
                    ip_str,
                )
                return False

        return True

    except Exception as exc:  # noqa: BLE001 — fail closed on unexpected errors
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    """Same rules as :func:`is_safe_url`, run off the event loop.

    ``socket.getaddrinfo`` can block; call this from async code paths instead
    of ``is_safe_url``.
    """
    return await asyncio.to_thread(is_safe_url, url)


# Known API key prefixes — match the prefix + contiguous token chars.
# Ported verbatim from Hermes ``agent/redact.py`` (lines 80-120).
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",  # OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",  # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",  # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",  # GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",  # GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",  # GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",  # GitHub refresh token
    r"xapp-\d+-[A-Za-z0-9-]{10,}",  # Slack app-Level token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack bot/app/user tokens
    r"AIza[A-Za-z0-9_-]{30,}",  # Google API keys
    r"pplx-[A-Za-z0-9]{10,}",  # Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",  # Fal.ai
    r"fc-[A-Za-z0-9]{10,}",  # Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",  # BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",  # Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",  # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",  # Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",  # Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",  # Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}",  # SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",  # HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",  # Replicate API token
    r"npm_[A-Za-z0-9]{10,}",  # npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",  # PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",  # DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",  # DigitalOcean OAuth
    r"am_[A-Za-z0-9_-]{10,}",  # AgentMail API key
    r"sk_[A-Za-z0-9_]{10,}",  # ElevenLabs TTS key (sk_ underscore, not sk- dash)
    r"tvly-[A-Za-z0-9]{10,}",  # Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",  # Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",  # Groq Cloud API key
    r"syt_[A-Za-z0-9]{10,}",  # Matrix access token
    r"retaindb_[A-Za-z0-9]{10,}",  # RetainDB API key
    r"hsk-[A-Za-z0-9]{10,}",  # Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",  # Mem0 Platform API key
    r"brv_[A-Za-z0-9]{10,}",  # ByteRover API key
    r"xai-[A-Za-z0-9]{30,}",  # xAI (Grok) API key
    r"ntn_[A-Za-z0-9]{10,}",  # Notion internal integration token
    r"fw-[A-Za-z0-9]{30,}",  # Fireworks AI API key
    r"fw_[A-Za-z0-9]{30,}",  # Fireworks AI API key
    r"fpk_[A-Za-z0-9]{30,}",  # Fireworks AI project key
    # GitLab token families (each pattern keeps a full literal prefix so the
    # prefix pre-screen stays false-negative-free).
    r"glpat-[A-Za-z0-9_\-]{10,}",  # GitLab personal access token
    r"gloas-[A-Za-z0-9_\-]{10,}",  # GitLab OAuth application secret
    r"gldt-[A-Za-z0-9_\-]{10,}",  # GitLab deploy token
    r"glrt-[A-Za-z0-9_.\-]{10,}",  # GitLab runner authentication token
    r"glrtr-[A-Za-z0-9_.\-]{10,}",  # GitLab runner registration token
    r"glcbt-[A-Za-z0-9_\-]{10,}",  # GitLab CI/CD job token
    r"glptt-[A-Za-z0-9_\-]{10,}",  # GitLab pipeline trigger token
    r"glft-[A-Za-z0-9_\-]{10,}",  # GitLab feed token
    r"glimt-[A-Za-z0-9_\-]{10,}",  # GitLab incoming mail token
    r"glagent-[A-Za-z0-9_\-]{10,}",  # GitLab agent (KAS) token
    r"glsoat-[A-Za-z0-9_\-]{10,}",  # GitLab service-account access token
    r"glffct-[A-Za-z0-9_\-]{10,}",  # GitLab feature-flags client token
    r"glwt-[A-Za-z0-9_\-]{10,}",  # GitLab workspace token
    r"GR1348941[A-Za-z0-9_\-]{10,}",  # GitLab legacy runner registration token
]

# Compile known prefix patterns into one alternation.
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def url_contains_secret(url: str) -> bool:
    """Return True when the URL carries a recognizable credential.

    Checks the raw URL, the ``unquote``d URL, and the normalized URL against
    the compiled credential-prefix regex (Hermes ``_PREFIX_RE``).
    """
    if not isinstance(url, str) or not url:  # pyright: ignore[reportUnnecessaryIsInstance]
        return False
    for candidate in (url, unquote(url), normalize_url_for_request(url)):
        if _PREFIX_RE.search(candidate):
            return True
    return False
