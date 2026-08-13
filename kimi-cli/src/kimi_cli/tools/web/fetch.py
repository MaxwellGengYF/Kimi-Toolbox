from typing import Literal, override

import trafilatura
from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.config import Config
from kimi_cli.constant import USER_AGENT
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import get_current_tool_call_or_none
from kimi_cli.tools.utils import ToolResultBuilder
from kimi_cli.utils.logging import logger


class Params(BaseModel):
    url: str = Field(description="URL to fetch content from.")
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds (1-300).",
    )
    method: Literal["GET", "POST"] = Field(
        default="GET",
        description="HTTP method to use.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Custom HTTP headers (e.g., {'Authorization': 'Bearer token'}).",
    )
    body: str | None = Field(
        default=None,
        description="Request body for POST requests.",
    )
    follow_redirects: bool = Field(
        default=True,
        description="Automatically follow HTTP redirects.",
    )
    max_redirects: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum number of redirects to follow (0-20).",
    )


class fetch_url(CallableTool2[Params]):
    name: str = "fetch_url"
    description: str = "Fetch a URL and extract main text."
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        self._runtime = runtime
        self._service_config = config.services.fetch

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if self._service_config:
            ret = await self._fetch_with_service(params)
            if not ret.is_error:
                return ret
            logger.warning("Failed to fetch URL via service: {error}", error=ret.message)
            # fallback to local fetch if service fetch fails
        return await self.fetch_with_http_get(params)

    @staticmethod
    async def fetch_with_http_get(params: Params) -> ToolReturnValue:
        import aiohttp

        from kimi_cli.utils.aiohttp import new_client_session

        builder = ToolResultBuilder(max_line_length=None)
        try:
            # Build request headers
            req_headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                ),
            }
            if params.headers:
                req_headers.update(params.headers)

            # Configure timeout
            fetch_timeout = aiohttp.ClientTimeout(
                total=params.timeout,
                sock_read=min(60, params.timeout),
                sock_connect=min(15, params.timeout),
            )

            # Configure redirects
            max_redirects = params.max_redirects if params.follow_redirects else 0

            async with new_client_session(timeout=fetch_timeout) as session:
                if params.method == "POST":
                    req = session.post(
                        params.url,
                        headers=req_headers,
                        data=params.body,
                        allow_redirects=params.follow_redirects,
                        max_redirects=max_redirects,
                    )
                else:
                    req = session.get(
                        params.url,
                        headers=req_headers,
                        allow_redirects=params.follow_redirects,
                        max_redirects=max_redirects,
                    )

                async with req as response:
                    if response.status >= 400:
                        logger.warning(
                            "fetch_url HTTP error: status={status}, url={url}",
                            status=response.status,
                            url=params.url,
                        )
                        return builder.error(
                            (
                                f"Failed to fetch URL. Status: {response.status}. "
                                f"This may indicate the page is not accessible or the server is down."
                            ),
                            brief=f"HTTP {response.status} error",
                        )

                    resp_text = await response.text()

                    content_type = response.headers.get(aiohttp.hdrs.CONTENT_TYPE, "").lower()
                    if content_type.startswith(("text/plain", "text/markdown")):
                        builder.write(resp_text)
                        return builder.ok("The returned content is the full content of the page.")
        except TimeoutError:
            logger.warning("fetch_url timed out: url={url}", url=params.url)
            return builder.error(
                "Failed to fetch URL: request timed out. The server may be slow or unreachable.",
                brief="Request timed out",
            )
        except aiohttp.ClientError as e:
            logger.warning("fetch_url network error: {error}, url={url}", error=e, url=params.url)
            return builder.error(
                (
                    f"Failed to fetch URL due to network error: {e}. "
                    "This may indicate the URL is invalid or the server is unreachable."
                ),
                brief="Network error",
            )

        if not resp_text:
            return builder.ok(
                "The response body is empty.",
                brief="Empty response body",
            )

        extracted_text = trafilatura.extract(
            resp_text,
            include_comments=True,
            include_tables=True,
            include_formatting=False,
            output_format="txt",
            with_metadata=True,
        )

        if not extracted_text:
            return builder.error(
                (
                    "Failed to extract meaningful content from the page. "
                    "This may indicate the page content is not suitable for text extraction, "
                    "or the page requires JavaScript to render its content."
                ),
                brief="No content extracted",
            )

        builder.write(extracted_text)
        return builder.ok("The returned content is the main text content extracted from the page.")

    async def _fetch_with_service(self, params: Params) -> ToolReturnValue:
        assert self._service_config is not None

        tool_call = get_current_tool_call_or_none()
        assert tool_call is not None, "Tool call is expected to be set"

        builder = ToolResultBuilder(max_line_length=None)
        api_key = self._runtime.oauth.resolve_api_key(
            self._service_config.api_key, self._service_config.oauth
        )
        if not api_key:
            return builder.error(
                "Fetch service is not configured. You may want to try other methods to fetch.",
                brief="Fetch service not configured",
            )
        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/markdown",
            "X-Msh-Tool-Call-Id": tool_call.id,
            **self._runtime.oauth.common_headers(),
            **(self._service_config.custom_headers or {}),
        }

        import aiohttp

        from kimi_cli.utils.aiohttp import new_client_session

        try:
            async with (
                new_client_session() as session,
                session.post(
                    self._service_config.base_url,
                    headers=headers,
                    json={"url": params.url},
                ) as response,
            ):
                if response.status != 200:
                    logger.warning(
                        "fetch_url service HTTP error: status={status}, url={url}",
                        status=response.status,
                        url=params.url,
                    )
                    return builder.error(
                        f"Failed to fetch URL via service. Status: {response.status}.",
                        brief="Failed to fetch URL via fetch service",
                    )

                content = await response.text()
                builder.write(content)
                return builder.ok(
                    "The returned content is the main content extracted from the page."
                )
        except TimeoutError:
            logger.warning("fetch_url service timed out: url={url}", url=params.url)
            return builder.error(
                "Failed to fetch URL via service: request timed out.",
                brief="Service request timed out",
            )
        except aiohttp.ClientError as e:
            logger.warning(
                "fetch_url service network error: {error}, url={url}", error=e, url=params.url
            )
            return builder.error(
                (
                    f"Failed to fetch URL via service due to network error: {e}. "
                    "This may indicate the service is unreachable."
                ),
                brief="Network error when calling fetch service",
            )
