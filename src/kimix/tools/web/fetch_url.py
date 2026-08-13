"""Fetch web page content as Markdown."""
import asyncio
from pathlib import Path

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimix.tools.common import _maybe_export_output
from kimix.tools.web.web_fetcher import fetch_to_markdown


class Params(BaseModel):
    """Parameters for fetch_url tool."""
    url: str = Field(
        description="URL to fetch content from."
    )
    output_path: str | None = Field(
        default=None,
        description="Optional file path to save the fetched markdown content."
    )


class fetch_url(CallableTool2[Params]):
    """Fetch a web page and return its content as Markdown."""
    name: str = "fetch_url"
    description: str = "Fetch a web page as Markdown."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        """Fetch URL content asynchronously and return markdown."""
        try:
            markdown = await fetch_to_markdown(params.url)
        except Exception as exc:
            return ToolError(
                message=str(exc) or f"Failed to fetch {params.url}",
                output="",
                brief=f"Failed to fetch {params.url}"
            )

        if params.output_path:
            try:
                output_file = Path(params.output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(output_file.write_text, markdown, encoding="utf-8")
                display_path = params.output_path.replace("\\", "/")
                return ToolOk(
                    output=f"Content saved to {display_path} ({len(markdown)} characters).",
                    brief=f"Fetched {params.url} and saved to {display_path}"
                )
            except Exception as exc:
                display_path = params.output_path.replace("\\", "/")
                return ToolError(
                    message=str(exc),
                    output=markdown,
                    brief=f"Failed to write {display_path}"
                )

        # Micro-compress the fetched markdown (plan.md §8.1): prose-mode
        # compression — encoding normalisation, whitespace collapse, prefix
        # fold.  Idempotent and annotated (markers keep any fold visible).
        # Only the model-facing output is compacted; an explicit output_path
        # still saves the raw fetched content.
        from kimi_cli.tools.file.micro_compress import (
            MicroCompressConfig,
            compress as _mc_compress,
        )

        markdown = _mc_compress(
            markdown, kind="prose", config=MicroCompressConfig()
        )
        output = _maybe_export_output(markdown)
        return ToolOk(output=output, brief=f"Fetched {params.url}")
