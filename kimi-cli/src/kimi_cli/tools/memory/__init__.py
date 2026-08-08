"""Retrieve past conversation history (archived/compacted turns included) from the session's `HistoryIndex`.

The context window is volatile working memory. This tool recalls earlier
messages that may have been pruned or compacted, so the agent can recover
past decisions, file paths, invariants, and debugging findings on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from kosong.tooling import CallableTool2, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kimi_cli.soul.history_index import HistoryIndex


class Params(BaseModel):
    query: str = Field(
        default="",
        description="Search past conversation history (BM25 with recency boost) for this natural-language query.",
    )
    id: str | None = Field(
        default=None,
        description="Fetch a specific history turn by id (e.g. '0' or 'prune_0').",
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of history turns to return.",
    )


class Retrieve(CallableTool2[Params]):
    name: str = "Retrieve"
    description: str = (
        "Retrieve past conversation history, including compacted/archived turns. "
        "The context window is volatile; this tool recalls earlier messages that "
        "may have been pruned or compacted. Use `query` to search (natural language, "
        "relevance-ranked with a recency boost) or `id` to fetch a specific turn "
        "(e.g. a `prune_<n>` reference left by context pruning)."
    )
    params: type[Params] = Params

    def __init__(self) -> None:
        super().__init__()
        self._history_index: HistoryIndex | None = None

    def attach_history_index(self, history_index: HistoryIndex | None) -> None:
        """Attach the session HistoryIndex so this tool can search past turns."""
        self._history_index = history_index

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if self._history_index is None:
            return ToolOk(
                output=(
                    "No history index attached. The `Retrieve` tool needs the "
                    "session's `HistoryIndex` to search past conversation turns."
                ),
                message="No history index",
            )

        # Fetch by explicit reference first (e.g. a `prune_<n>` stub).
        if params.id is not None:
            return self._retrieve_by_id(params.id)

        # Otherwise search by query.
        if not params.query.strip():
            return ToolOk(
                output="No query provided. Pass a `query` string or an `id`.",
                message="No query",
            )

        results = self._history_index.search_with_recency(
            params.query,
            top_k=params.k,
            recency_weight=1.0,
        )
        if not results:
            return ToolOk(
                output="No matching results found in conversation history.",
                message="No results",
            )

        lines: list[str] = [f"Retrieved {len(results)} result(s):", "", "[Conversation history]"]
        for r in results:
            role = r["role"]
            text = r["text"]
            score = r.get("score", 0.0)
            marker = " [compacted]" if r.get("is_compacted") else " [current]"
            lines.append(
                f"> **{role}**{marker} (relevance: {score:.2f})\n"
                f"> {text.replace(chr(10), chr(10) + '> ')}"
            )

        return ToolOk(
            output="\n".join(lines),
            message=f"Found {len(results)} result(s)",
        )

    def _retrieve_by_id(self, ref_id: str) -> ToolReturnValue:
        """Retrieve a history turn by id (plain turn id or 'prune_<n>')."""
        assert self._history_index is not None
        turn = self._history_index.get_by_id(ref_id)
        if turn is None:
            return ToolOk(
                output=f"No turn found with id={ref_id!r}.",
                message="No results",
            )
        role = turn["role"]
        text = turn["text"]
        marker = " [compacted]" if turn.get("is_compacted") else " [current]"
        return ToolOk(
            output=(
                f"Retrieved turn id={ref_id!r}:\n"
                f"> **{role}**{marker}\n"
                f"> {text.replace(chr(10), chr(10) + '> ')}"
            ),
            message=f"Found turn id={ref_id!r}",
        )
