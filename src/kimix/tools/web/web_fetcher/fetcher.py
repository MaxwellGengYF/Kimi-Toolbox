import asyncio
import regex as re
import ssl

import httpx
from markdownify import markdownify as md

_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
)
_LOGIN_PATTERNS = re.compile(
    r"登录|密码登录|验证码登录|注册|Sign in|Log in|Login|Verification code|短信验证码",
    re.IGNORECASE,
)
_BROWSER_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


async def _fetch_html(url: str, user_agent: str, viewport: dict, wait_until: str) -> str:
    # Imported lazily: playwright is only needed when the HTTP fallback fails
    # and a real browser has to be launched.
    from playwright.async_api import TimeoutError as PWTimeoutError
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        html = ""
        try:
            context = await browser.new_context(
                user_agent=user_agent,
                viewport=viewport,
                locale="en-US",
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until=wait_until, timeout=60000)
            except PWTimeoutError:
                # If networkidle times out, the DOM is usually ready enough.
                # But if the page never loaded at all, bail out so the caller
                # can try the HTTP fallback instead of returning empty HTML.
                if page.url == "about:blank" and not url.lower().startswith("about:"):
                    raise RuntimeError(
                        f"Browser navigation timed out without loading {url}"
                    )
                pass
            await page.wait_for_timeout(2000)
            html = await page.content()
        finally:
            try:
                await browser.close()
            except Exception:
                pass
    return html


def _build_ssl_context(
    verify: bool = True, tls_version: str | None = None
) -> ssl.SSLContext:
    """Build an SSL context with optional verification and TLS version pinning."""
    if tls_version == "1.2":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    else:
        ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _fetch_html_http(
    url: str,
    user_agent: str,
    *,
    retries: int = 3,
    verify: bool = True,
    tls_version: str | None = None,
) -> str:
    """Fallback HTTP fetch using httpx with retries and relaxed SSL support."""
    headers = {"User-Agent": user_agent, **_BROWSER_HEADERS}
    ssl_context = _build_ssl_context(verify=verify, tls_version=tls_version)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=15.0),
        follow_redirects=True,
        verify=ssl_context,
    ) as client:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = await client.get(url, headers=headers)
                # Raise for 4xx/5xx so we can retry transient failures.
                if response.status_code >= 500:
                    response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # Retry only transient server errors and rate limits.
                if exc.response.status_code == 429 or exc.response.status_code >= 500:
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except (httpx.ConnectError, httpx.TimeoutException, ssl.SSLError) as exc:
                last_exc = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        # If we exhausted retries, surface the last error.
        raise last_exc or RuntimeError(f"Failed to fetch {url} after {retries} attempts")


async def fetch_html_http_with_fallback(url: str, user_agent: str) -> str:
    """Try HTTP fetch with cert verification, then without, then TLS 1.2 pinned."""
    attempts = [
        {"verify": True, "tls_version": None},
        {"verify": False, "tls_version": None},
        {"verify": False, "tls_version": "1.2"},
    ]
    last_exc: Exception | None = None
    for config in attempts:
        try:
            return await _fetch_html_http(url, user_agent, **config)
        except (httpx.ConnectError, ssl.SSLError) as exc:
            last_exc = exc
            continue
    raise last_exc or RuntimeError(f"Failed to fetch {url}")


def _html_to_markdown(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for tag_name in [
        "script",
        "style",
        "noscript",
        "img",
        "video",
        "audio",
        "source",
        "track",
        "iframe",
        "embed",
        "object",
        "canvas",
        "svg",
        "picture",
        "figure",
        "nav",
        "aside",
        "footer",
        "header",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    main = soup.find("main") or soup.find(role="main")
    body = soup.find("body")
    if main and len(main.get_text(strip=True)) >= 500:
        target = main
    elif body:
        target = body
    else:
        target = soup

    markdown = md(str(target), heading_style="ATX")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


async def fetch_to_markdown(url: str, wait_until: str = "networkidle") -> str:
    """Fetch a URL using a headless browser, execute JS, and return extracted text as Markdown."""
    try:
        html = await _fetch_html(url, _DESKTOP_UA, {"width": 1920, "height": 1080}, wait_until)
    except Exception:
        html = await fetch_html_http_with_fallback(url, _DESKTOP_UA)

    markdown = _html_to_markdown(html)

    # If the result looks like a login wall, retry with a mobile user agent.
    text_len = len(markdown.replace(" ", "").replace("\n", ""))
    if text_len < 300 or _LOGIN_PATTERNS.search(markdown):
        html_mobile = ""
        try:
            html_mobile = await _fetch_html(url, _MOBILE_UA, {"width": 390, "height": 844}, wait_until)
        except Exception:
            try:
                html_mobile = await fetch_html_http_with_fallback(url, _MOBILE_UA)
            except Exception:
                pass
        if html_mobile:
            markdown_mobile = _html_to_markdown(html_mobile)
            mobile_text_len = len(markdown_mobile.replace(" ", "").replace("\n", ""))
            if mobile_text_len > text_len:
                markdown = markdown_mobile

    return markdown
