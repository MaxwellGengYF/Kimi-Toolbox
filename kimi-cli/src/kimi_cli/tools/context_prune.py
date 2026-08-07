"""ContextPrune tool — manual session-content removal.

Provides ``ContextPrune``, an agent tool that safely removes stale content
from the conversation history.  It supports three modes:

* ``prune`` — policy-driven ephemeral/substantive elision via
  :class:`kimi_cli.soul.context_pruning.ContextPruner`.
* ``compact`` — full compaction via :meth:`KimiSoul.compact_context`.
* ``strip_reasoning`` — remove old ``ThinkPart`` content while preserving
  provider back-pass invariants.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, override

from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.context_pruning import (
    ContextPruner,
    PruningResult,
    _compute_protected_indices,
    _protect_tool_pair_indices,
)
from kimi_cli.soul.message import is_system_reminder_message
from kimi_cli.soul import get_wire_or_none, wire_send
from kimi_cli.utils.logging import logger
from kimi_cli.utils.tokens import count_message_tokens
from kimi_cli.wire.types import StatusUpdate, TextPart, ThinkPart

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul


def _current_turn_start_index(history: Sequence[Message]) -> int | None:
    """Index of the current turn's first *real* user message.

    Scans backwards from the end and skips injected ``<system-reminder>``
    user messages (mirrors ``KimiSoul._current_turn_start_index``).
    """
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.role == "user" and not is_system_reminder_message(msg):
            return idx
    return None


class Params(BaseModel):
    mode: Literal["prune", "compact", "strip_reasoning"] = Field(
        default="prune",
        description="Strategy: prune stale content, compact old turns, or strip old reasoning.",
    )
    target_token_count: int | None = Field(
        default=None,
        ge=1000,
        description="Target max tokens after pruning.",
    )
    remove_reasoning: bool = Field(
        default=True,
        description="Remove old reasoning/thinking content.",
    )
    remove_tool_results: bool = Field(
        default=True,
        description="Remove old tool-result messages.",
    )
    keep_recent_turns: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Recent user/assistant turns to keep.",
    )
    dry_run: bool = Field(
        default=False,
        description="Report what would be removed without changing the session.",
    )


class ContextPrune(CallableTool2[Params]):
    name: str = "ContextPrune"
    description: str = (
        "Prune old session content (reasoning, tool results, stale messages) to save tokens. "
        "Recent turns and tool-call pairs are always preserved. "
        "Modes: 'prune' (smart elision), 'compact' (full compaction), "
        "'strip_reasoning' (remove old thinking content). "
        "Use dry_run=True to preview changes."
    )
    params: type[Params] = Params

    def __init__(self, soul: "KimiSoul") -> None:
        super().__init__()
        self._soul = soul

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        logger.info(
            "ContextPrune invoked: mode={mode}, keep_recent_turns={keep}, "
            "target_token_count={target}, dry_run={dry_run}",
            mode=params.mode,
            keep=params.keep_recent_turns,
            target=params.target_token_count,
            dry_run=params.dry_run,
        )

        soul = self._soul
        context = soul.context
        pruner = soul.pruner
        history = list(context.history)

        # Respect loop control: subagents are only allowed to prune when
        # explicitly enabled, matching the auto-pruner behavior in _step().
        is_subagent = getattr(soul.runtime, "role", None) == "subagent"
        if is_subagent:
            loop_control = getattr(soul, "_loop_control", None)
            if loop_control is not None and not loop_control.prune_subagents:
                return ToolError(
                    message=(
                        "ContextPrune is not enabled for subagents. "
                        "Enable loop_control.prune_subagents or run from the root session."
                    ),
                    brief="Subagent pruning disabled",
                )

        llm = soul.runtime.llm
        max_context_size = llm.max_context_size if llm is not None else 128_000
        model_name = llm.chat_provider.model_name if llm is not None else None

        # Structural validation
        validation_error = self._validate_params(
            params=params,
            history=history,
            pruner=pruner,
            max_context_size=max_context_size,
            model=model_name,
            current_turn_index=_current_turn_start_index(history),
        )
        if validation_error is not None:
            return validation_error

        if params.mode == "compact":
            return await self._run_compact(params=params)

        if params.mode == "strip_reasoning":
            pruned_messages, result = self._run_strip_reasoning(
                history=history,
                params=params,
                pruner=pruner,
                current_turn_index=_current_turn_start_index(history),
            )
        else:
            result = pruner.prune_with_policy(
                history,
                remove_reasoning=params.remove_reasoning,
                remove_tool_results=params.remove_tool_results,
                keep_recent_turns=params.keep_recent_turns,
                target_token_count=params.target_token_count,
                max_context_size=max_context_size,
                current_step=soul.current_step_no,
                current_turn_index=_current_turn_start_index(history),
                model=model_name,
            )
            pruned_messages = result.messages

        if result.earliest_removed_index is None:
            return ToolOk(
                output=f"ContextPrune ({params.mode}): no removable content found.",
                message="Nothing to prune",
            )

        summary = self._build_summary(
            mode=params.mode,
            history=history,
            pruned_messages=pruned_messages,
            result=result,
        )

        if params.dry_run:
            logger.info("ContextPrune dry-run: {summary}", summary=summary)
            return ToolOk(
                output=f"**Dry run** — session unchanged.\n\n{summary}",
                message="Dry run complete",
            )

        # Persist the pruned history
        await context.replace_history(pruned_messages)

        # Index Tier-B elided originals so they remain retrievable
        if result.elided:
            elided_messages = [
                Message(role=rec.role, content=[TextPart(text=rec.original_text)])
                for rec in result.elided
                if rec.original_text.strip()
            ]
            if elided_messages:
                soul._history_index.index_messages(elided_messages)
                soul._history_index.save()
                for rec in result.elided:
                    soul._recently_restored_refs.add(rec.ref)

        # Emit status update with new context usage (only if a wire is active)
        status = soul.status
        if get_wire_or_none() is not None:
            wire_send(
                StatusUpdate(
                    context_usage=status.context_usage,
                    context_tokens=status.context_tokens,
                    max_context_tokens=status.max_context_tokens,
                )
            )

        logger.info(
            "ContextPrune applied: mode={mode}, freed={freed}, earliest={idx}, "
            "elided={elided}",
            mode=params.mode,
            freed=result.freed_tokens,
            idx=result.earliest_removed_index,
            elided=len(result.elided),
        )

        return ToolOk(
            output=f"ContextPrune ({params.mode}) applied.\n\n{summary}",
            message="Context pruned",
        )

    # ------------------------------------------------------------------ #
    # Mode implementations
    # ------------------------------------------------------------------ #

    async def _run_compact(self, params: Params) -> ToolReturnValue:
        """Delegate to the compaction subsystem."""
        if params.dry_run:
            # Compaction is LLM-driven; a true dry-run estimate is expensive.
            # We report that compaction would run without mutating state.
            return ToolOk(
                output=(
                    "**Dry run** — session unchanged.\n\n"
                    "Mode 'compact' would invoke the compaction subsystem. "
                    "A precise preview requires an LLM call; run without dry_run to compact."
                ),
                message="Compact dry run",
            )

        await self._soul.compact_context(manual=True)
        return ToolOk(
            output="Context compacted successfully.",
            message="Context compacted",
        )

    def _run_strip_reasoning(
        self,
        *,
        history: Sequence[Message],
        params: Params,
        pruner: ContextPruner,
        current_turn_index: int | None = None,
    ) -> tuple[list[Message], PruningResult]:
        """Build a new history with ThinkPart removed outside the protected tail.

        When thinking mode is active, removed ``ThinkPart`` entries are replaced
        with empty ``ThinkPart(think="")`` so the provider back-pass invariant
        is preserved.
        """
        protected = _compute_protected_indices(
            history,
            stable_prefix_messages=pruner._stable_prefix_messages,
            recent_messages_protected=params.keep_recent_turns,
            current_turn_index=current_turn_index,
        )
        protected = _protect_tool_pair_indices(history, protected)

        thinking_active = self._is_thinking_active()
        pruned_messages: list[Message] = []
        changed_indices: set[int] = set()
        freed_tokens = 0

        for i, msg in enumerate(history):
            if msg.role != "assistant" or i in protected:
                pruned_messages.append(msg)
                continue

            new_content: list[TextPart | ThinkPart] = []
            removed_think = False
            for part in msg.content:
                if isinstance(part, ThinkPart):
                    removed_think = True
                    freed_tokens += max(len(part.think) // 4, 1)
                    if thinking_active:
                        # Preserve back-pass with empty reasoning
                        new_content.append(ThinkPart(think=""))
                else:
                    new_content.append(part)

            if removed_think:
                changed_indices.add(i)
                pruned_messages.append(
                    Message(
                        role=msg.role,
                        content=new_content,
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                pruned_messages.append(msg)

        earliest = min(changed_indices) if changed_indices else None
        result = PruningResult(
            messages=pruned_messages,
            elided=[],
            freed_tokens=freed_tokens,
            earliest_removed_index=earliest,
        )
        return pruned_messages, result

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_params(
        self,
        *,
        params: Params,
        history: Sequence[Message],
        pruner: ContextPruner,
        max_context_size: int,
        model: str | None,
        current_turn_index: int | None = None,
    ) -> ToolReturnValue | None:
        """Return a ToolError if the requested operation would violate invariants."""
        if params.mode == "compact":
            return None

        n = len(history)
        stable_prefix = pruner._stable_prefix_messages

        if params.keep_recent_turns > max(0, n - stable_prefix):
            return ToolError(
                message=(
                    f"keep_recent_turns={params.keep_recent_turns} is too large: "
                    f"history has {n} messages and {stable_prefix} are protected as a stable prefix."
                ),
                brief="Invalid keep_recent_turns",
            )

        if params.target_token_count is not None:
            protected = _compute_protected_indices(
                history,
                stable_prefix_messages=stable_prefix,
                recent_messages_protected=params.keep_recent_turns,
                current_turn_index=current_turn_index,
            )
            protected = _protect_tool_pair_indices(history, protected)
            protected_messages = [history[i] for i in sorted(protected)]
            protected_tokens = count_message_tokens(protected_messages, model=model)

            if params.target_token_count < protected_tokens:
                return ToolError(
                    message=(
                        f"target_token_count={params.target_token_count} is below the "
                        f"{protected_tokens} tokens required by the protected prefix/recent turns. "
                        "Increase the target or reduce keep_recent_turns."
                    ),
                    brief="Target too low",
                )

        # Refuse to prune the only user/assistant pair (would leave no conversation)
        non_system_roles = [msg.role for msg in history if msg.role in ("user", "assistant")]
        if params.mode == "prune" and len(non_system_roles) <= 2:
            return ToolError(
                message=(
                    "Refusing to prune: the history contains only one user/assistant pair. "
                    "Pruning would leave the conversation empty."
                ),
                brief="History too short",
            )

        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_thinking_active(self) -> bool:
        """Return True when the active provider requires reasoning back-pass."""
        thinking = self._soul.thinking
        return thinking if thinking is not None else False

    def _build_summary(
        self,
        *,
        mode: str,
        history: Sequence[Message],
        pruned_messages: Sequence[Message],
        result: PruningResult,
    ) -> str:
        """Build a markdown summary of the prune result."""
        changed = len(history) - len(pruned_messages) + len(result.elided)
        lines: list[str] = [
            f"- **Mode:** {mode}",
            f"- **Messages before:** {len(history)}",
            f"- **Messages after:** {len(pruned_messages)}",
            f"- **Messages changed/dropped:** {changed}",
            f"- **Estimated tokens freed:** {result.freed_tokens}",
        ]
        if result.earliest_removed_index is not None:
            lines.append(f"- **Earliest changed index:** {result.earliest_removed_index}")
        if result.elided:
            refs = ", ".join(f"`{rec.ref}`" for rec in result.elided)
            lines.append(f"- **Elided references:** {refs}")
        else:
            lines.append("- **Elided references:** none")

        blocked: list[str] = []
        if mode == "prune" and not result.elided:
            # Tool results were requested to be kept; note that Tier-B was disabled.
            pass

        if blocked:
            lines.append("- **Invariants preserved:**")
            for note in blocked:
                lines.append(f"  - {note}")

        return "\n".join(lines)
