import sys
from typing import TYPE_CHECKING, Any, override

from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field, ValidationError

from kimi_cli.config import Config
from kimi_cli.constant import USER_AGENT
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.toolset import get_current_tool_call_or_none
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.utils import ToolResultBuilder
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.utils.aiohttp import new_client_session


def __getattr__(name: str) -> Any:
    """Lazily resolve ``aiohttp``-based helpers (kept monkeypatchable in tests)."""
    if name == "new_client_session":
        from kimi_cli.utils.aiohttp import new_client_session

        return new_client_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Params(BaseModel):
    query: str = Field(description="The search query.")
    limit: int = Field(
        description="Number of results. Prefer a specific query over a high limit.",
        default=5,
        ge=1,
        le=20,
    )
    include_content: bool = Field(
        description="Include full page content. Increases token usage.",
        default=False,
    )


class SearchWeb(CallableTool2[Params]):
    name: str = "web_search"
    description: str = (
        "Search the web for current information. Returns an optional summary "
        "answer and a list of source URLs."
    )
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        if config.services.search is None:
            raise SkipThisTool()
        self._runtime = runtime
        self._base_url = config.services.search.base_url
        self._api_key = config.services.search.api_key
        self._oauth_ref = config.services.search.oauth
        self._custom_headers = config.services.search.custom_headers or {}

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        import aiohttp

        # Resolved through the module attribute so tests can monkeypatch it.
        new_client_session = getattr(sys.modules[__name__], "new_client_session")

        builder = ToolResultBuilder(max_line_length=None)

        api_key = self._runtime.oauth.resolve_api_key(self._api_key, self._oauth_ref)
        if not self._base_url or not api_key:
            return builder.error(
                "Search service is not configured. You may want to try other methods to search.",
                brief="Search service not configured",
            )

        tool_call = get_current_tool_call_or_none()
        assert tool_call is not None, "Tool call is expected to be set"

        try:
            # Server-side timeout is 30s, but page crawling can take longer.
            search_timeout = aiohttp.ClientTimeout(total=180, sock_read=90, sock_connect=15)
            async with (
                new_client_session(timeout=search_timeout) as session,
                session.post(
                    self._base_url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Authorization": f"Bearer {api_key}",
                        "X-Msh-Tool-Call-Id": tool_call.id,
                        **self._runtime.oauth.common_headers(),
                        **self._custom_headers,
                    },
                    json={
                        "text_query": params.query,
                        "limit": params.limit,
                        "enable_page_crawling": params.include_content,
                        "timeout_seconds": 30,
                    },
                ) as response,
            ):
                if response.status != 200:
                    logger.warning(
                        "SearchWeb HTTP error: status={status}, query={query}",
                        status=response.status,
                        query=params.query,
                    )
                    return builder.error(
                        (
                            f"Failed to search. Status: {response.status}. "
                            "This may indicate that the search service is currently unavailable."
                        ),
                        brief="Failed to search",
                    )

                try:
                    results = Response(**await response.json()).search_results
                except ValidationError as e:
                    logger.warning(
                        "SearchWeb response parse error: {error}, query={query}",
                        error=e,
                        query=params.query,
                    )
                    return builder.error(
                        (
                            f"Failed to parse search results. Error: {e}. "
                            "This may indicate that the search service is currently unavailable."
                        ),
                        brief="Failed to parse search results",
                    )
        except TimeoutError:
            logger.warning("SearchWeb request timed out: query={query}", query=params.query)
            return builder.error(
                "Search request timed out. The search service may be slow or unavailable.",
                brief="Search request timed out",
            )
        except aiohttp.ClientError as e:
            logger.warning(
                "SearchWeb network error: {error}, query={query}",
                error=e,
                query=params.query,
            )
            return builder.error(
                f"Search request failed: {e}. The search service may be unavailable.",
                brief="Search request failed",
            )

        for i, result in enumerate(results):
            if i > 0:
                builder.write("---\n\n")
            builder.write(
                f"Title: {result.title}\nDate: {result.date}\n"
                f"URL: {result.url}\nSummary: {result.snippet}\n\n"
            )
            if result.content:
                builder.write(f"{result.content}\n\n")

        return builder.ok()


class SearchResult(BaseModel):
    site_name: str
    title: str
    url: str
    snippet: str
    content: str = ""
    date: str = ""
    icon: str = ""
    mime: str = ""


class Response(BaseModel):
    search_results: list[SearchResult]
