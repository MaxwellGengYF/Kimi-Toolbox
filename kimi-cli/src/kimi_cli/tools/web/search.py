import inspect
from typing import Any, override

from kosong.tooling import CallableTool2, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.config import Config
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools import SkipThisTool
from kimi_cli.tools.utils import ToolResultBuilder
from kimi_cli.utils.logging import logger as _logging_logger

# Module-level ``logger`` attribute kept so tests can patch
# ``kimi_cli.tools.web.search.logger``.
logger = _logging_logger


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


class SearchWeb(CallableTool2[Params]):
    name: str = "web_search"
    description: str = (
        "Search the web for current information. Returns an optional summary "
        "answer and a list of source URLs."
    )
    params: type[Params] = Params

    def __init__(self, config: Config, runtime: Runtime):
        super().__init__()
        import kimi_cli.tools.web.providers as providers

        # Importing the providers module self-registers the ddgs/local
        # providers; register the Kimi service provider only when a search
        # service is configured.
        if config.services.search is not None:
            providers.register_provider(
                providers.KimiServiceProvider(config, runtime)
            )
        if providers.get_active_search_provider(config) is None:
            raise SkipThisTool()
        self._config = config
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        from kimi_cli.tools.web.providers import get_active_search_provider

        builder = ToolResultBuilder(max_line_length=None)

        provider = get_active_search_provider(self._config)
        if provider is None:
            return builder.error(
                "Search service is not configured. You may want to try other methods to search.",
                brief="Search service not configured",
            )

        # ``provider_any`` keeps the call untyped so sync and async providers
        # with different optional kwargs (e.g. ``include_content``) both work.
        provider_any: Any = provider
        if params.include_content:
            result: Any = provider_any.search(
                params.query, params.limit, include_content=True
            )
        else:
            result = provider_any.search(params.query, params.limit)

        if inspect.iscoroutine(result):
            result = await result

        if not result.get("success"):
            return builder.error(
                result.get("error", "Search failed"),
                brief="Search failed",
            )

        data: Any = result.get("data") or {}
        web_items: Any = data.get("web", []) or []
        for i, item in enumerate(web_items):
            if i > 0:
                builder.write("---\n\n")
            builder.write(
                f"Title: {item.get('title', '')}\nDate: {item.get('date', '')}\n"
                f"URL: {item.get('url', '')}\nSummary: {item.get('description', '')}\n\n"
            )
            content = item.get("content") or ""
            if content:
                builder.write(f"{content}\n\n")

        return builder.ok()
