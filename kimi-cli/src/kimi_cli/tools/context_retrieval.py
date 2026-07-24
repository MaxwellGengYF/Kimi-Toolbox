from __future__ import annotations

from typing import override

from kosong.tooling import CallableTool2, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.history_index import HistoryIndex


class Params(BaseModel):
    query: str = Field(default="", description="Search query (keywords or natural-language phrase) to search past conversation turns.")
    k: int = Field(default=3, ge=1, le=10, description="Number of top matching turns to return.")
    id: str | None = Field(default=None, description="Optional stable reference ID to retrieve a specific elided turn by ID.")


class ContextRetrieval(CallableTool2[Params]):
    name: str = "ContextRetrieval"
    description: str = (
        "Search past conversation turns (both current session and archived) matching a query. "
        "Returns verbatim excerpts from user, assistant, and tool-result messages. "
        "Archived turns (from before compaction/pruning) are annotated with ``[compacted]``, "
        "current-session turns with ``[current]``. "
        "Use to recall decisions, file paths, error messages, or tool outputs from anywhere "
        "in the conversation history. "
        "If an ``id`` is provided instead of a query, the exact turn with that reference ID is returned."
    )
    params: type[Params] = Params

    def __init__(self, history_index: HistoryIndex) -> None:
        super().__init__()
        self._history_index = history_index

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        # If an explicit id is given, retrieve by reference
        if params.id is not None:
            turn = self._history_index.get_by_id(params.id)
            if turn is None:
                return ToolOk(
                    output=f"No turn found with id={params.id!r}.",
                    message="No results",
                )
            role = turn["role"]
            text = turn["text"]
            if turn.get("is_compacted"):
                marker = " [compacted]"
            else:
                marker = " [current]"
            return ToolOk(
                output=(
                    f"Retrieved turn id={params.id!r}:\n"
                    f"> **{role}**{marker}\n"
                    f"> {text.replace(chr(10), chr(10) + '> ')}"
                ),
                message=f"Found turn id={params.id!r}",
            )

        # Otherwise search by query
        if not params.query.strip():
            return ToolOk(
                output="No query provided. Pass a ``query`` string or an ``id``.",
                message="No query",
            )

        results = self._history_index.search_with_recency(
            params.query,
            top_k=params.k,
            recency_weight=1.0,
        )
        if not results:
            return ToolOk(
                output="No matching past turns found.",
                message="No results",
            )

        lines: list[str] = [f"Retrieved {len(results)} past turn(s):"]
        for r in results:
            role = r["role"]
            text = r["text"]
            score = r.get("score", 0.0)
            if r.get("is_compacted"):
                marker = " [compacted]"
            else:
                marker = " [current]"
            lines.append(
                f"> **{role}**{marker} (relevance: {score:.2f})\n"
                f"> {text.replace(chr(10), chr(10) + '> ')}"
            )

        return ToolOk(
            output="\n\n".join(lines),
            message=f"Found {len(results)} turn(s)",
        )