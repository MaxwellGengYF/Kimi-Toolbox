from __future__ import annotations

import asyncio
import os
import random
import regex as re
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import kosong
import tenacity
from kosong import StepResult
from kosong.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
    RetryableChatProvider,
)
from kosong.message import Message
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base

from kimi_cli.approval_runtime import (
    ApprovalSource,
    get_current_approval_source_or_none,
    reset_current_approval_source,
    set_current_approval_source,
)
from kimi_cli.background import build_active_task_snapshot
from kimi_cli.hooks.engine import HookEngine
from kimi_cli.llm import ModelCapability
from kimi_cli.notifications import (
    NotificationView,
    build_notification_message,
    extract_notification_ids,
)
from kimi_cli.skill import Skill, read_skill_text
from kimi_cli.skill.flow import Flow, FlowEdge, FlowNode, parse_choice
from kimi_cli.soul import (
    LLMNotSet,
    LLMNotSupported,
    MaxStepsReached,
    SessionRestartRequired,
    Soul,
    StatusSnapshot,
    wire_send,
)
from kimi_cli.session_state import TodoItemState
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.compaction import (
    SAFETY_MARGIN_TOKENS,
    CompactionOptions,
    CompactionResult,
    CompactionShrinkError,
    CompactMode,
    ManualCompactionError,
    SimpleCompaction,
    SurfaceChangedError,
    adaptive_preserve_depth,
    estimate_text_tokens,
    should_auto_compact,
)
from kimi_cli.soul.compaction_ledger import CompactionLedger
from kimi_cli.soul.context_overflow import (
    OverflowRecoveryState,
    is_context_overflow_error,
)
from kimi_cli.soul.context_pruning import ContextPruner, is_pruned_stub
from kimi_cli.soul.context import Context
from kimi_cli.soul.dynamic_injection import (
    DynamicInjection,
    DynamicInjectionProvider,
    normalize_history,
)
from kimi_cli.soul.dynamic_injections.budget_reminder import BudgetReminderProvider
from kimi_cli.soul.dynamic_injections.compact_reminder import CompactReminderProvider
from kimi_cli.soul.dynamic_injections.context_meter import ContextMeterProvider
from kimi_cli.soul.dynamic_injections.target_churn import TargetChurnProvider
from kimi_cli.soul.verification_gate import VerificationGate
from kimi_cli.soul.dynamic_injections.todo_reminder import TodoReminderProvider
from kimi_cli.soul.llm_request_recorder import LLMRequestRecorder
from kimi_cli.soul.message import (
    check_message,
    is_system_reminder_message,
    strip_system_reminders,
    system,
    system_reminder,
    tool_result_to_message,
)
from kimi_cli.soul.slash import registry as soul_slash_registry
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.context_prune import context_prune
from kimi_cli.tools.todo import Todo, TodoList
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.export import perform_export
from kimi_cli.utils.logging import logger
from kimi_cli.utils.slashcmd import SlashCommand, parse_slash_command_call
from kimi_cli.utils.tokens import count_message_tokens, count_tokens
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.types import (
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    MCPLoadingBegin,
    MCPLoadingEnd,
    StatusUpdate,
    SteerInput,
    StepBegin,
    StepInterrupted,
    StepRetry,
    TextPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
)

if TYPE_CHECKING:

    def type_check(soul: KimiSoul):
        _: Soul = soul


SKILL_COMMAND_PREFIX = "skill:"
FLOW_COMMAND_PREFIX = "flow:"
DEFAULT_MAX_FLOW_MOVES = 1000


class _RateLimitAwareWait(wait_base):
    """Tenacity wait callable that honors ``Retry-After`` for 429 responses."""

    def __init__(
        self,
        *,
        default_initial: float = 0.3,
        default_max: float = 5.0,
        default_jitter: float = 0.5,
        rate_limit_initial: float = 1.0,
        rate_limit_max: float = 30.0,
        rate_limit_jitter: float = 1.0,
        max_retry_after: float = 60.0,
    ) -> None:
        self.default_initial = default_initial
        self.default_max = default_max
        self.default_jitter = default_jitter
        self.rate_limit_initial = rate_limit_initial
        self.rate_limit_max = rate_limit_max
        self.rate_limit_jitter = rate_limit_jitter
        self.max_retry_after = max_retry_after

    def __call__(self, retry_state: RetryCallState) -> float:
        attempt = retry_state.attempt_number
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, APIStatusError) and exception.status_code == 429:
            if exception.retry_after is not None:
                return float(min(exception.retry_after, self.max_retry_after))
            return min(
                self.rate_limit_initial * (2 ** (attempt - 1))
                + random.uniform(0, self.rate_limit_jitter),
                self.rate_limit_max,
            )
        return min(
            self.default_initial * (2 ** (attempt - 1))
            + random.uniform(0, self.default_jitter),
            self.default_max,
        )


_RETRY_WAIT = _RateLimitAwareWait()


def classify_api_error(e: Exception) -> tuple[str, int | None]:
    """Classify an LLM API exception into (error_type, status_code).

    Returns:
        (error_type, status_code) where status_code is None for non-HTTP errors.
    """
    status_code: int | None = None
    if isinstance(e, APIStatusError):
        status = getattr(e, "status_code", getattr(e, "status", 0))
        status_code = int(status) if status else None
        if status == 429:
            return "rate_limit", status_code
        if status in (401, 403):
            return "auth", status_code
        if status >= 500:
            return "5xx_server", status_code
        if 400 <= status < 500:
            # Delegate marker matching to context_overflow so the marker list
            # lives in exactly one place (Phase 4 §6.1).
            if is_context_overflow_error(e):
                return "context_overflow", status_code
            return "4xx_client", status_code
        return "api", status_code
    if isinstance(e, APIConnectionError):
        return "network", None
    if isinstance(e, (APITimeoutError, TimeoutError)):
        return "timeout", None
    if isinstance(e, APIEmptyResponseError):
        return "empty_response", None
    return "other", None


type StepStopReason = Literal["no_tool_calls", "tool_rejected", "tool_call_repeat"]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    stop_reason: StepStopReason
    assistant_message: Message


type TurnStopReason = StepStopReason


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    stop_reason: TurnStopReason
    final_message: Message | None
    step_count: int


def _current_turn_start_index(history: Sequence[Message]) -> int | None:
    """Return the index of the current turn's first *real* user message.

    Scans backwards from the end and stops at the most recent user message
    that is not an injected ``<system-reminder>`` (mirrors
    ``VerificationGate._current_turn_history``). Returns ``None`` when no
    such message exists (e.g. empty history).
    """
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.role == "user" and not is_system_reminder_message(msg):
            return idx
    return None


def _compaction_export_missing(prompt: str, work_dir: Any) -> bool:
    """Return True when *prompt* references a compaction export file that no
    longer exists (cache-05 §3.3).

    Compaction prompts embed the pre-compaction export path (relative to the
    work dir). Anonymous sessions delete those files on close, so a resumed
    session must not adopt a persisted prompt pointing at a deleted file.
    """
    if ".kimix_cache" not in prompt:
        return False
    for match in re.finditer(
        r"\.kimix_cache[/\\](?:context_compacted|context_[0-9a-f]+)\.md",
        prompt,
    ):
        candidate = Path(str(work_dir)) / match.group(0).replace("\\", "/")
        if not candidate.exists():
            return True
    return False


class KimiSoul:
    """The soul of Kimi Code CLI."""

    def __init__(
        self,
        agent: Agent,
        *,
        context: Context,
        anonymous: bool = False,
    ):
        """
        Initialize the soul.

        Args:
            agent (Agent): The agent to run.
            context (Context): The context of the agent.
            anonymous (bool): Whether the session is anonymous. When True,
                history files are deleted on close.
        """
        self._agent = agent
        self._runtime = agent.runtime
        self._anonymous = anonymous
        self._denwa_renji = agent.runtime.denwa_renji
        self._approval = agent.runtime.approval
        self._loop_control = agent.runtime.config.loop_control

        # Phase 3: History index for semantic retrieval over past turns
        # Lazy import to avoid circular dependency with kimix.retrieval
        from kimi_cli.soul.history_index import HistoryIndex
        from kimi_cli.tools.context_prune import context_prune

        history_db_path = agent.runtime.session.dir / "history.db"
        legacy_json_path = (
            agent.runtime.session.dir / "history_index" / f"{agent.runtime.session.id}.json"
        )
        self._history_index = HistoryIndex(
            db_path=history_db_path,
            legacy_json_path=legacy_json_path,
        )
        self._history_index.load()
        # Expose the index to the Session so delete/rename/copy can close the
        # apsw handle before moving/deleting the session directory on Windows.
        agent.runtime.session._history_index = self._history_index

        # Wire context appends into the history indexer
        def _on_append(msgs: Sequence[Message]) -> None:
            self._history_index.index_messages(msgs)

        context._on_append = _on_append
        self._context = context
        self._context.model_name = self.model_name

        # cache-04 §3.2 / cache-05 §3.3: on resume, adopt the exact system
        # prompt persisted with the context so the first request after resume
        # reuses the previous process's prompt string (provider prefix-cache
        # continuity). Skip adoption when the persisted prompt references a
        # compaction export file that no longer exists (e.g. anonymous session
        # cleanup) — re-render normally in that case.
        restored_prompt = self._context.system_prompt
        if restored_prompt and not _compaction_export_missing(
            restored_prompt, agent.runtime.session.work_dir
        ):
            agent.system_prompt_cached = restored_prompt

        # Phase 3/4: durable compaction transaction ledger. Failure-isolated —
        # a broken session dir or ledger path must never break soul init.
        # ``CompactionLedger.for_session`` already degrades to a no-op ledger on
        # disabled config or I/O errors; this extra guard treats any other
        # construction failure as ``None`` (compaction then simply skips the
        # ledger) with a warning.
        try:
            self._compaction_ledger = CompactionLedger.for_session(
                agent.runtime.session.dir,
                enabled=self._loop_control.compaction_ledger_enabled,
            )
        except Exception as exc:  # noqa: BLE001 — ledger must never break soul init
            logger.warning(
                "Failed to initialize compaction ledger for session {sid}: {err}; "
                "compaction transactions will not be persisted",
                sid=agent.runtime.session.id,
                err=exc,
            )
            self._compaction_ledger = None

        if self._loop_control.adaptive_preserve_enabled:
            self._compaction = SimpleCompaction(
                max_preserved_messages=self._loop_control.max_preserved_messages,
                preserve_depth=lambda msgs: adaptive_preserve_depth(
                    msgs,
                    min_preserved=self._loop_control.min_preserved_messages,
                    max_preserved=self._loop_control.max_preserved_messages,
                ),
                decision_section_enabled=self._loop_control.compaction_decision_section_enabled,
            )
        else:
            self._compaction = SimpleCompaction(
                max_preserved_messages=self._loop_control.max_preserved_messages,
                decision_section_enabled=self._loop_control.compaction_decision_section_enabled,
            )

        # Register context-management tools if the toolset supports it
        if isinstance(agent.toolset, KimiToolset):
            # Attach the history index to the retrieve tool so it can search
            # past conversation turns.
            from kimi_cli.tools.memory import retrieve
            retrieve_tool = agent.toolset.find("retrieve")
            if not isinstance(retrieve_tool, retrieve):
                retrieve_tool = retrieve()
                agent.toolset.add(retrieve_tool)
            retrieve_tool.attach_history_index(self._history_index)
            agent.toolset.add(context_prune(self))

        self._checkpoint_with_user_message = False

        self._llm_request_recorder = LLMRequestRecorder()
        self._recorder_restored = False
        self._steer_queue: asyncio.Queue[str | list[ContentPart]] = asyncio.Queue()
        # Wake event set by ``request_steer`` so a steer can interrupt the
        # currently streaming step (see ``_agent_loop`` step race).
        self._steer_wake_event = asyncio.Event()
        self._current_step_task: asyncio.Task[Any] | None = None
        self._run_active = False
        # The event loop the soul runs on (set by ``run``). Used by ``Steer``
        # to marshal thread-safe enqueues from other loops/threads.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_tool_calls: list[tuple[str, str]] = []
        self._current_turn_id: str = ""
        self._current_step_no: int = 0
        self._current_turn_user_text: str = ""
        self._last_auto_retrieved_turn_id: int | None = None
        self._recently_retrieved_turn_ids: set[int] = set()
        self._injection_providers: list[DynamicInjectionProvider] = [
            *(
                []
                if not self._loop_control.compact_reminder_enabled
                else [CompactReminderProvider(
                    threshold=self._loop_control.compact_reminder_threshold,
                )]
            ),
            *(
                []
                if not self._loop_control.todo_reminder_enabled
                else [TodoReminderProvider(
                    self._load_todo_states_for_reminder,
                    interval_steps=self._loop_control.todo_reminder_interval_steps,
                )]
            ),
            *(
                []
                if not self._loop_control.target_churn_enabled
                else [TargetChurnProvider(
                    file_warn=self._loop_control.target_churn_file_warn,
                    file_strong=self._loop_control.target_churn_file_strong,
                    error_warn=self._loop_control.target_churn_error_warn,
                    cooldown_steps=self._loop_control.target_churn_cooldown_steps,
                )]
            ),
            *(
                []
                if not self._loop_control.budget_reminder_enabled
                else [BudgetReminderProvider(
                    warn_ratios=tuple(self._loop_control.budget_warn_ratios),
                    wall_clock_seconds=self._loop_control.budget_wall_clock_seconds,
                )]
            ),
            *(
                []
                if not self._loop_control.context_meter_enabled
                else [ContextMeterProvider(
                    min_delta=self._loop_control.context_meter_min_delta,
                    cooldown_steps=self._loop_control.context_meter_cooldown_steps,
                    suppress_above=(
                        self._loop_control.compact_reminder_threshold
                        if self._loop_control.compact_reminder_enabled
                        else None
                    ),
                )]
            ),
        ]
        self._hook_engine: HookEngine = HookEngine()
        self._stop_hook_active: bool = False
        self._verification_gate = VerificationGate(
            max_nudges=self._loop_control.verification_gate_max_nudges,
        )
        if self.is_root:
            self._runtime.notifications.ack_ids("llm", extract_notification_ids(context.history))

        self._slash_commands = self._build_slash_commands()
        self._slash_command_map = self._index_slash_commands(self._slash_commands)

        # Track rotated compaction export files for cleanup on close.
        # We keep an explicit list of exported pre-compact markdown files so
        # anonymous sessions can remove each file individually instead of
        # wiping a whole directory (which could affect unrelated files).
        # Phase 2: Context pruner for smart history removal
        self._pruner = ContextPruner(
            enabled=self._loop_control.context_pruning_enabled,
            trigger_ratio=self._loop_control.prune_trigger_ratio,
            target_ratio=self._loop_control.prune_target_ratio,
            stable_prefix_messages=self._loop_control.prune_stable_prefix_messages,
            recent_messages_protected=self._loop_control.prune_recent_messages_protected,
            min_free_tokens=self._loop_control.prune_min_free_tokens,
            cooldown_steps=self._loop_control.prune_cooldown_steps,
            min_usage_growth=self._loop_control.prune_min_usage_growth,
            max_fraction_per_pass=self._loop_control.prune_max_fraction_per_pass,
            ephemeral_enabled=self._loop_control.prune_ephemeral_enabled,
            ephemeral_notifications=self._loop_control.prune_ephemeral_notifications,
            ephemeral_task_snapshots=self._loop_control.prune_ephemeral_task_snapshots,
            ephemeral_dmail_notices=self._loop_control.prune_ephemeral_dmail_notices,
            ephemeral_checkpoint_markers=self._loop_control.prune_ephemeral_checkpoint_markers,
            substantive_enabled=self._loop_control.prune_substantive_enabled,
            tool_output_min_tokens=self._loop_control.prune_tool_output_min_tokens,
            micro_compress_enabled=self._loop_control.prune_micro_compress_enabled,
            micro_compress_min_saved_chars=self._loop_control.prune_micro_compress_min_saved_chars,
            min_cache_prefix_depth=self._loop_control.prune_min_cache_prefix_depth,
            cache_loss_penalty=self._loop_control.prune_cache_loss_penalty,
        )
        self._recently_restored_refs: set[str] = set()

        self._compact_cache_dir: list[Path] = []
        self._finalizer = weakref.finalize(
            self,
            KimiSoul._sync_cleanup,
            self._compact_cache_dir,
            self._context.file_backend,
            self._anonymous,
        )

    def _tool_call_buffer_tokens(self) -> int:
        """Return the dynamic per-tool output budget in tokens.

        Used to reserve enough headroom for a tool result before the next
        compaction decision.
        """
        if not isinstance(self._agent.toolset, KimiToolset) or self._runtime.llm is None:
            return 0
        return self._agent.toolset.estimate_tool_output_token_budget(
            self._runtime.llm.max_context_size,
            self._context.token_count_with_pending,
        )

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def model_name(self) -> str:
        return self._runtime.llm.chat_provider.model_name if self._runtime.llm else ""

    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        if self._runtime.llm is None:
            return None
        return self._runtime.llm.capabilities

    @property
    def is_yolo(self) -> bool:
        """Whether explicit yolo mode is active."""
        return self._approval.is_yolo()

    @property
    def is_auto_approve(self) -> bool:
        """Whether tool approvals are bypassed (explicit yolo, or implied by afk)."""
        return self._approval.is_auto_approve()

    @property
    def is_afk(self) -> bool:
        """Whether no user is present (away-from-keyboard)."""
        return self._approval.is_afk()

    @property
    def is_afk_flag(self) -> bool:
        """Whether persisted afk mode is active."""
        return self._approval.is_afk_flag()

    @property
    def is_root(self) -> bool:
        """Whether this soul is the root session rather than a subagent."""
        return self._runtime.role == "root"

    @property
    def is_subagent(self) -> bool:
        """Whether this soul is running as a subagent rather than the root session."""
        return self._runtime.role == "subagent"

    @property
    def hook_engine(self) -> HookEngine:
        return self._hook_engine

    def set_hook_engine(self, engine: HookEngine) -> None:
        self._hook_engine = engine
        if isinstance(self._agent.toolset, KimiToolset):
            self._agent.toolset.set_hook_engine(engine)

    def _load_todo_states_for_reminder(self) -> list[TodoItemState]:
        """Load current todo states (root or subagent scope) for the todo reminder.

        Reuses the TodoList tool's loading logic so the reminder always agrees
        with what the tool would report. Never raises — a broken state file
        must not break the agent loop.
        """

        def convert(todo: Todo) -> TodoItemState:
            # Todo models store the title as ``content``; TodoItemState uses ``title``.
            return TodoItemState(
                title=todo.content,
                status=todo.status,
                notes=todo.notes,
                children=[convert(child) for child in todo.children],
            )

        try:
            todos = TodoList(self._runtime)._load_todos()
            return [convert(todo) for todo in todos]
        except Exception:
            logger.debug("Failed to load todos for reminder", exc_info=True)
            return []

    def add_injection_provider(self, provider: DynamicInjectionProvider) -> None:
        """Register an additional dynamic injection provider."""
        self._injection_providers.append(provider)

    async def _collect_injections(self) -> list[DynamicInjection]:
        """Collect dynamic injections from all registered providers."""
        injections: list[DynamicInjection] = []
        for provider in self._injection_providers:
            try:
                result = await provider.get_injections(self._context.history, self)
                injections.extend(result)
            except Exception:
                logger.warning(
                    "injection provider %s failed",
                    type(provider).__name__,
                    exc_info=True,
                )
        return injections

    async def _maybe_auto_retrieve_history(self) -> list[DynamicInjection]:
        """Auto-inject relevant turns from long-term, working, and recency memory.

        Returns up to three distinct injections per turn, deduplicated so the
        same turn_id is never injected twice in the same turn or in the
        immediately previous auto-retrieval.

        Only fires on the first step of a turn when at least one retrieval tier
        is enabled and the user query is non-trivial.
        """
        lc = self._loop_control
        if not (
            lc.auto_retrieve_history
            or lc.auto_retrieve_working_memory
            or lc.auto_retrieve_recency_memory
        ):
            return []
        if self._current_step_no != 1:
            return []
        query = self._current_turn_user_text
        if len(query) < 10:
            return []

        try:
            results = self._history_index.search_with_recency(
                query,
                top_k=10,
                recency_weight=lc.auto_retrieve_recency_weight,
            )
        except Exception:
            logger.debug("History index search failed during auto-retrieval", exc_info=True)
            return []

        # Deduplicate against recently retrieved turn IDs
        candidates = [
            r for r in results if r["turn_id"] not in self._recently_retrieved_turn_ids
        ]

        injections: list[DynamicInjection] = []
        used_turn_ids: set[int] = set()

        wrapper_overhead = 15  # tokens for <system-reminder> tags and newlines
        budget = lc.auto_retrieve_max_tokens_per_turn
        spent = 0

        def _can_afford(citation: str, turn_id: int) -> bool:
            nonlocal spent
            citation_tokens = count_tokens(citation, model=self.model_name)
            injection_cost = citation_tokens + wrapper_overhead
            if spent + injection_cost <= budget:
                spent += injection_cost
                return True
            logger.debug(
                "Skipping auto-retrieved turn {turn_id}: would exceed token budget "
                "({spent} + {cost} > {budget})",
                turn_id=turn_id,
                spent=spent,
                cost=injection_cost,
                budget=budget,
            )
            return False

        # ── A. Long-term memory (compacted turns) ────────────────────────────
        if lc.auto_retrieve_history:
            compacted = [
                r for r in candidates if r.get("is_compacted") and r["turn_id"] not in used_turn_ids
            ]
            if compacted:
                best = compacted[0]
                score = best.get("score", 0.0)
                if score >= lc.auto_retrieve_history_threshold:
                    turn_id = best["turn_id"]
                    role = best.get("role", "unknown")
                    text = best.get("text", "")
                    citation = (
                        f"[Auto-retrieved from past conversation — relevance: {score:.2f}]\n"
                        f"> **{role}**\n> {text.replace(chr(10), chr(10) + '> ')}"
                    )
                    if _can_afford(citation, turn_id):
                        used_turn_ids.add(turn_id)
                        logger.debug(
                            "Auto-retrieved history turn {turn_id} with score {score}",
                            turn_id=turn_id,
                            score=score,
                        )
                        injections.append(
                            DynamicInjection(type="auto_retrieved_history", content=citation)
                        )

        # ── B. Working memory (non-compacted turns, excluding recent context) ─
        if lc.auto_retrieve_working_memory:
            # Exclude the last 2 non-compacted turns (last user+assistant pair)
            # because they are already at the end of the LLM context.
            non_compacted = [
                r
                for r in candidates
                if not r.get("is_compacted") and r["turn_id"] not in used_turn_ids
            ]
            # Find the most recent non-compacted turn_ids to exclude
            all_non_compacted_turn_ids = [
                t["turn_id"] for t in self._history_index._turns if not t.get("is_compacted")
            ]
            recent_turn_ids = set(all_non_compacted_turn_ids[-2:]) if len(all_non_compacted_turn_ids) >= 2 else set(all_non_compacted_turn_ids)
            eligible = [
                r for r in non_compacted if r["turn_id"] not in recent_turn_ids
            ]
            if eligible:
                best = eligible[0]
                score = best.get("score", 0.0)
                if score >= lc.auto_retrieve_working_memory_threshold:
                    turn_id = best["turn_id"]
                    role = best.get("role", "unknown")
                    text = best.get("text", "")
                    citation = (
                        f"[Relevant context from our current conversation]\n"
                        f"> **{role}**\n> {text.replace(chr(10), chr(10) + '> ')}"
                    )
                    if _can_afford(citation, turn_id):
                        used_turn_ids.add(turn_id)
                        logger.debug(
                            "Auto-retrieved working memory turn {turn_id} with score {score}",
                            turn_id=turn_id,
                            score=score,
                        )
                        injections.append(
                            DynamicInjection(type="working_memory", content=citation)
                        )

        # ── C. Recency memory (best boosted score, compacted or non-compacted) ─
        if lc.auto_retrieve_recency_memory:
            eligible = [
                r
                for r in candidates
                if r["turn_id"] not in used_turn_ids
                and r.get("boosted_score", 0.0) >= lc.auto_retrieve_recency_memory_threshold
            ]
            if eligible:
                best = eligible[0]
                turn_id = best["turn_id"]
                role = best.get("role", "unknown")
                text = best.get("text", "")
                boosted_score = best.get("boosted_score", 0.0)
                citation = (
                    f"[Recently discussed — relevance: {boosted_score:.2f}]\n"
                    f"> **{role}**\n> {text.replace(chr(10), chr(10) + '> ')}"
                )
                if _can_afford(citation, turn_id):
                    used_turn_ids.add(turn_id)
                    logger.debug(
                        "Auto-retrieved recency memory turn {turn_id} with boosted_score {boosted_score}",
                        turn_id=turn_id,
                        boosted_score=boosted_score,
                    )
                    injections.append(
                        DynamicInjection(type="recency_memory", content=citation)
                    )

        # Respect the per-turn cap
        max_inj = lc.auto_retrieve_max_injections_per_turn
        if len(injections) > max_inj:
            injections = injections[:max_inj]
            used_turn_ids = set(list(used_turn_ids)[:max_inj])

        # Update dedup tracking
        for turn_id in used_turn_ids:
            self._recently_retrieved_turn_ids.add(turn_id)
        # Keep the set bounded to the last ~10 IDs (turn_id is monotonically
        # increasing, so the smallest id is the oldest).
        while len(self._recently_retrieved_turn_ids) > 10:
            self._recently_retrieved_turn_ids.discard(min(self._recently_retrieved_turn_ids))
        # Update backward-compat single-ID tracker with the most recent injection
        if used_turn_ids:
            self._last_auto_retrieved_turn_id = max(used_turn_ids)

        return injections

    async def _notify_injection_providers_compacted(self) -> None:
        """Notify all injection providers that the context has been compacted.

        Failures are isolated per-provider so a buggy third-party provider
        cannot abort compaction (which would skip the CompactionEnd wire event
        and PostCompact hook).
        """
        for provider in self._injection_providers:
            try:
                await provider.on_context_compacted()
            except Exception:
                logger.warning(
                    "injection provider %s on_context_compacted failed",
                    type(provider).__name__,
                    exc_info=True,
                )

    async def notify_afk_changed(self, enabled: bool) -> None:
        """Notify dynamic injection providers that afk mode changed."""
        for provider in self._injection_providers:
            try:
                await provider.on_afk_changed(enabled)
            except Exception:
                logger.warning(
                    "injection provider %s on_afk_changed failed",
                    type(provider).__name__,
                    exc_info=True,
                )

    @property
    def thinking(self) -> bool | None:
        """Whether thinking mode is enabled."""
        if self._runtime.llm is None:
            return None
        if thinking_effort := self._runtime.llm.chat_provider.thinking_effort:
            return thinking_effort != "off"
        return None

    @property
    def status(self) -> StatusSnapshot:
        token_count = self._context.token_count
        max_size = self._runtime.llm.max_context_size if self._runtime.llm is not None else 0
        return StatusSnapshot(
            context_usage=self._context_usage,
            yolo_enabled=self._approval.is_yolo_flag(),
            afk_enabled=self._approval.is_afk(),
            context_tokens=token_count,
            max_context_tokens=max_size,
            mcp_status=self._mcp_status_snapshot(),
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def context(self) -> Context:
        return self._context

    @property
    def pruner(self) -> ContextPruner:
        """The context pruner for smart history removal."""
        return self._pruner

    @property
    def current_step_no(self) -> int:
        """Current step number in the agent loop."""
        return self._current_step_no

    @property
    def _context_usage(self) -> float:
        if self._runtime.llm is not None:
            return self._context.token_count / self._runtime.llm.max_context_size
        return 0.0

    @property
    def wire_file(self) -> WireFile:
        return self._runtime.session.wire_file

    async def _ensure_recorder_restored(self) -> None:
        """One-shot scan of the existing wire.jsonl to seed request-trace dedup state."""
        if self._recorder_restored:
            return
        self._recorder_restored = True
        await self._llm_request_recorder.restore_from(self.wire_file)

    def _drain_mcp_discoveries(self) -> None:
        """Feed parked MCP tool discoveries into the request-trace recorder."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return
        for discovery in self._agent.toolset.drain_pending_mcp_discoveries():
            self._llm_request_recorder.record_mcp_discovery(
                discovery.server_name,
                discovery.tools,
                discovery.enabled_names,
                discovery.collisions,
            )

    def _mcp_status_snapshot(self):
        if not isinstance(self._agent.toolset, KimiToolset):
            return None
        return self._agent.toolset.mcp_status_snapshot()

    async def start_background_mcp_loading(self) -> bool:
        """Start deferred MCP loading, if any, without exposing toolset internals."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return False
        return await self._agent.toolset.start_deferred_mcp_tool_loading()

    async def wait_for_background_mcp_loading(self) -> None:
        """Wait for any in-flight MCP startup to finish."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return
        await self._agent.toolset.wait_for_mcp_tools()

    async def _checkpoint(self):
        await self._context.checkpoint(self._checkpoint_with_user_message)

    def steer(self, content: str | list[ContentPart]) -> None:
        """Queue a steer message for injection into the current turn.

        Step-boundary steering only: the message is consumed between steps (or
        before turn end), never mid-stream. Use :meth:`request_steer` for a
        mid-stream interrupt.
        """
        self._steer_queue.put_nowait(content)

    async def request_steer(self, content: str | list[ContentPart]) -> None:
        """Enqueue a steer and wake the agent loop so the currently streaming
        step is interrupted and the steer injected as a follow-up user
        message."""
        self._steer_queue.put_nowait(content)
        self._steer_wake_event.set()

    def is_running(self) -> bool:
        """Whether the soul is currently running a turn."""
        return self._run_active

    async def _consume_pending_steers(self) -> bool:
        """Drain the steer queue and inject as follow-up user messages.

        Returns True if any steers were consumed.

        Note: /btw is intercepted at the UI layer (``classify_input``) before
        reaching the steer queue, so it never appears here.
        """
        consumed = False
        while not self._steer_queue.empty():
            content = self._steer_queue.get_nowait()
            await self._inject_steer(content)
            wire_send(SteerInput(user_input=content))
            consumed = True
        return consumed

    async def _inject_steer(self, content: str | list[ContentPart]) -> None:
        """Inject a single steer as a regular follow-up user message."""
        parts = cast(
            list[ContentPart],
            [TextPart(text=content)] if isinstance(content, str) else list(content),
        )
        message = Message(role="user", content=parts)
        if self._runtime.llm is None:
            raise LLMNotSet()
        if missing_caps := check_message(message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))
        await self._context.append_message(message)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands

    async def run(
        self,
        user_input: str | list[ContentPart],
        *,
        skip_user_prompt_hook: bool = False,
    ):
        approval_source_token = None
        created_approval_source: ApprovalSource | None = None
        turn_started = False
        turn_finished = False
        if get_current_approval_source_or_none() is None:
            created_approval_source = ApprovalSource(kind="foreground_turn", id=uuid.uuid4().hex)
            approval_source_token = set_current_approval_source(created_approval_source)
        try:
            self._loop = asyncio.get_running_loop()
            # The soul persists across prompts, and each prompt may run in a
            # fresh event loop (e.g. ``kimix.utils.prompt.prompt`` wraps every
            # prompt in its own ``asyncio.run``). ``asyncio.Event``/``Queue``
            # bind to the loop on first use, so recreate them here — otherwise
            # the second prompt raises ``RuntimeError: <asyncio.locks.Event
            # ...> is bound to a different event loop`` on its first step.
            self._steer_wake_event = asyncio.Event()
            self._steer_queue = asyncio.Queue()
            self._run_active = True
            # Refresh OAuth tokens on each turn to avoid idle-time expirations.
            await self._runtime.oauth.ensure_fresh(self._runtime)

            # Set session_id ContextVar for toolset hooks
            from kimi_cli.soul.toolset import set_session_id

            set_session_id(self._runtime.session.id)

            from kimi_cli.hooks import events

            # --- UserPromptSubmit hook ---
            # Synthetic internal prompts (e.g. background-task notification
            # follow-ups injected by ``Print`` after a bg task finishes or
            # the wait ceiling is hit) must bypass ``UserPromptSubmit``:
            # they are not user input, and a user-configured prompt-blocking
            # hook would drop the notification and hang the wait loop.
            if not skip_user_prompt_hook:
                text_input_for_hook = user_input if isinstance(user_input, str) else ""

                hook_results = await self._hook_engine.trigger(
                    "UserPromptSubmit",
                    matcher_value=text_input_for_hook,
                    input_data=events.user_prompt_submit(
                        session_id=self._runtime.session.id,
                        cwd=str(self._runtime.session.work_dir),
                        prompt=text_input_for_hook,
                    ),
                )
                for result in hook_results:
                    if result.action == "block":
                        wire_send(TurnBegin(user_input=user_input))
                        turn_started = True
                        wire_send(TextPart(text=result.reason or "Prompt blocked by hook."))
                        wire_send(TurnEnd())
                        turn_finished = True
                        return

            wire_send(TurnBegin(user_input=user_input))
            turn_started = True


            user_message = Message(role="user", content=user_input)
            text_input = user_message.extract_text(" ").strip()

            if command_call := parse_slash_command_call(text_input):
                command = self._find_slash_command(command_call.name)
                if command is None:
                    # this should not happen actually, the shell should have filtered it out
                    wire_send(TextPart(text=f'Unknown slash command "/{command_call.name}".'))
                else:
                    ret = command.func(self, command_call.args)
                    if isinstance(ret, Awaitable):
                        await ret
            elif self._loop_control.max_ralph_iterations != 0:
                runner = FlowRunner.ralph_loop(
                    user_message,
                    self._loop_control.max_ralph_iterations,
                )
                await runner.run(self, "")
            else:
                await self._turn(user_message)

            # --- Stop hook (max 1 re-trigger to prevent infinite loop) ---
            if not self._stop_hook_active:
                stop_results = await self._hook_engine.trigger(
                    "Stop",
                    input_data=events.stop(
                        session_id=self._runtime.session.id,
                        cwd=str(self._runtime.session.work_dir),
                        stop_hook_active=False,
                    ),
                )
                for result in stop_results:
                    if result.action == "block" and result.reason:
                        self._stop_hook_active = True
                        try:
                            await self._turn(Message(role="user", content=result.reason))
                        finally:
                            self._stop_hook_active = False
                        break

            wire_send(TurnEnd())
            turn_finished = True

            # Auto-set title after first real turn (skip slash commands)
            if not command_call:
                session = self._runtime.session
                if session.state.custom_title is None:
                    from kimi_cli.utils.string import shorten

                    title = shorten(
                        Message(role="user", content=user_input).extract_text(" "),
                        width=50,
                    )
                    if title:
                        from kimi_cli.session_state import (
                            load_session_state,
                            save_session_state,
                        )

                        # Read-modify-write: load fresh state to avoid
                        # overwriting concurrent web changes
                        fresh = load_session_state(session.dir)
                        if fresh.custom_title is None:
                            fresh.custom_title = title
                            save_session_state(fresh, session.dir)
                        session.state.custom_title = fresh.custom_title
        finally:
            self._run_active = False
            if turn_started and not turn_finished:
                wire_send(TurnEnd())


            if created_approval_source is not None and self._runtime.approval_runtime is not None:
                self._runtime.approval_runtime.cancel_by_source(
                    created_approval_source.kind,
                    created_approval_source.id,
                )
            if approval_source_token is not None:
                reset_current_approval_source(approval_source_token)

    async def _turn(self, user_message: Message) -> TurnOutcome:
        if self._runtime.llm is None:
            raise LLMNotSet()

        if missing_caps := check_message(user_message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        self._current_turn_id = uuid.uuid4().hex
        self._current_turn_user_text = user_message.extract_text(" ").strip()
        self._last_tool_calls = []
        await self._checkpoint()  # this creates the checkpoint 0 on first run
        await self._context.append_message(user_message)
        logger.debug("Appended user message to context")
        return await self._agent_loop()

    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        commands: list[SlashCommand[Any]] = list(soul_slash_registry.list_commands())
        seen_names = {cmd.name for cmd in commands}

        for skill in self._runtime.skills.values():
            if skill.type not in ("standard", "flow"):
                continue
            name = f"{SKILL_COMMAND_PREFIX}{skill.name}"
            if name in seen_names:
                logger.warning(
                    "Skipping skill slash command /{name}: name already registered",
                    name=name,
                )
                continue
            commands.append(
                SlashCommand(
                    name=name,
                    func=self._make_skill_runner(skill),
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(name)

        for skill in self._runtime.skills.values():
            if skill.type != "flow":
                continue
            if skill.flow is None:
                logger.warning("Flow skill {name} has no flow; skipping", name=skill.name)
                continue
            command_name = f"{FLOW_COMMAND_PREFIX}{skill.name}"
            if command_name in seen_names:
                logger.warning(
                    "Skipping prompt flow slash command /{name}: name already registered",
                    name=command_name,
                )
                continue
            runner = FlowRunner(skill.flow, name=skill.name)
            commands.append(
                SlashCommand(
                    name=command_name,
                    func=runner.run,
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(command_name)

        return commands

    @staticmethod
    def _index_slash_commands(
        commands: list[SlashCommand[Any]],
    ) -> dict[str, SlashCommand[Any]]:
        indexed: dict[str, SlashCommand[Any]] = {}
        for command in commands:
            indexed[command.name] = command
            for alias in command.aliases:
                indexed[alias] = command
        return indexed

    def _find_slash_command(self, name: str) -> SlashCommand[Any] | None:
        return self._slash_command_map.get(name)

    def _make_skill_runner(self, skill: Skill) -> Callable[[KimiSoul, str], None | Awaitable[None]]:
        async def _run_skill(soul: KimiSoul, args: str, *, _skill: Skill = skill) -> None:

            skill_text = await read_skill_text(_skill)
            if skill_text is None:
                wire_send(
                    TextPart(text=f'Failed to load skill "/{SKILL_COMMAND_PREFIX}{_skill.name}".')
                )
                return
            extra = args.strip()
            if extra:
                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
            await soul._turn(Message(role="user", content=skill_text))

        _run_skill.__doc__ = skill.description
        return _run_skill

    async def _agent_loop(self) -> TurnOutcome:
        """The main agent loop for one run.

        Lifecycle:
            1. Turn Initialization   - clean up stale steers, load MCP tools.
            2. Step Loop             - iterate until the turn stops or fails.
               a. Step Guard         - enforce max-steps-per-turn limit.
               b. Step Begin         - emit StepBegin wire event.
               c. Context Compaction - auto-compact if context exceeds trigger ratio.
               d. Checkpoint         - persist current state before calling LLM.
               e. Step Execution     - run _step() (LLM call + tool execution).
               f. Error Handling     - BackToTheFuture (revert) or fatal exception.
               g. Outcome Resolution - steers / stop / continue.
            3. Turn Resolution       - return TurnOutcome to the caller.
        """
        assert self._runtime.llm is not None

        # ═══════════════════════════════════════════════════════════════════════
        # 1. TURN INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════════

        # Discard any stale steers from a previous turn.
        while not self._steer_queue.empty():
            self._steer_queue.get_nowait()
        # Reset the wake event so a steer arriving later in this turn can
        # interrupt a streaming step.
        self._steer_wake_event.clear()

        # ── 1a. MCP deferred loading ──────────────────────────────────────────
        if isinstance(self._agent.toolset, KimiToolset):
            await self.start_background_mcp_loading()
            loading = bool((snapshot := self._mcp_status_snapshot()) and snapshot.loading)
            if loading:
                wire_send(StatusUpdate(mcp_status=snapshot))
                wire_send(MCPLoadingBegin())
            try:
                await self.wait_for_background_mcp_loading()
            finally:
                if loading:
                    wire_send(StatusUpdate(mcp_status=self._mcp_status_snapshot()))
                    wire_send(MCPLoadingEnd())

        # ── 1b. Request-trace bookkeeping ─────────────────────────────────────
        # Restore dedup cursors from the existing wire.jsonl (resumed sessions),
        # then drain any MCP discoveries parked by background connect tasks.
        await self._ensure_recorder_restored()
        self._drain_mcp_discoveries()

        # ═══════════════════════════════════════════════════════════════════════
        # 2. STEP LOOP
        # ═══════════════════════════════════════════════════════════════════════
        step_no = 0
        self._current_step_no = 0
        while True:
            step_no += 1

            # ── 2a. Step Guard ──────────────────────────────────────────────────
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(self._loop_control.max_steps_per_turn)

            self._current_step_no = step_no

            # ── 2b. Step Begin ──────────────────────────────────────────────────
            wire_send(StepBegin(n=step_no))
            back_to_the_future: BackToTheFuture | None = None
            step_outcome: StepOutcome | None = None

            try:
                # ── 2c. Context Compaction ──────────────────────────────────────
                # Check if pruning can free enough space to avoid compaction
                _should_compact = should_auto_compact(
                    self._context.token_count_with_pending,
                    self._runtime.llm.max_context_size,
                    trigger_ratio=self._loop_control.compaction_trigger_ratio,
                    reserved_context_size=self._loop_control.reserved_context_size,
                    max_tokens=self._runtime.config.max_tokens,
                    tool_call_buffer_tokens=self._tool_call_buffer_tokens(),
                    safety_margin_tokens=SAFETY_MARGIN_TOKENS,
                )
                if _should_compact and self._loop_control.context_pruning_enabled:
                    # Estimate tokens after pruning — skip compaction only when
                    # pruning brings the *full* next-request input back below the
                    # auto-compaction thresholds (in particular below the
                    # reserved-output boundary ``max_context_size -
                    # max(tool_call_buffer_tokens, reserved_context_size,
                    # max_tokens + safety_margin_tokens)``: the point at
                    # which ``input_token_size >= context_token_size -
                    # max_output_token_size`` would hold and input + output would
                    # no longer fit in the context window). Comparing against the
                    # ratio threshold alone could skip compaction while the input
                    # is still at/over the boundary and overflow the window.
                    llm = self._runtime.llm
                    model_name = llm.chat_provider.model_name if llm else None
                    estimated = self._pruner.estimate_after_prune(
                        self._context.history,
                        context_usage=self.status.context_usage,
                        max_context_size=self._runtime.llm.max_context_size,
                        current_step=self._current_step_no,
                        current_turn_index=_current_turn_start_index(list(self._context.history)),
                        min_cache_prefix_depth=self._cache_depth_floor(len(self._context.history)),
                        model=model_name,
                    )
                    # ``estimate_after_prune`` counts message content only, while
                    # ``token_count_with_pending`` (used by the trigger above)
                    # also carries the system prompt and tool schemas. Add that
                    # non-history overhead back so the comparison uses the same
                    # basis as the trigger check.
                    current_history_tokens = count_message_tokens(
                        self._context.history, model=model_name
                    )
                    overhead = max(
                        0, self._context.token_count_with_pending - current_history_tokens
                    )
                    if not should_auto_compact(
                        estimated + overhead,
                        self._runtime.llm.max_context_size,
                        trigger_ratio=self._loop_control.compaction_trigger_ratio,
                        reserved_context_size=self._loop_control.reserved_context_size,
                        max_tokens=self._runtime.config.max_tokens,
                        tool_call_buffer_tokens=self._tool_call_buffer_tokens(),
                        safety_margin_tokens=SAFETY_MARGIN_TOKENS,
                    ):
                        # Pruning will free enough space — skip compaction
                        _should_compact = False
                        logger.debug(
                            "Pruning estimated to free enough space ({est} tokens, overhead={overhead}), skipping compaction",
                            est=estimated,
                            overhead=overhead,
                        )
                if _should_compact:
                    logger.info("Context too long, compacting...")
                    try:
                        await self.compact_context()
                    except Exception as compact_err:
                        logger.error(
                            "Context compaction failed at step {step_no}: {error_type}: {error}",
                            step_no=step_no,
                            error_type=type(compact_err).__name__,
                            error=compact_err,
                        )
                        raise

                # ── 2d. Checkpoint ──────────────────────────────────────────────
                logger.debug("Beginning step {step_no}", step_no=step_no)
                await self._checkpoint()
                self._denwa_renji.set_n_checkpoints(self._context.n_checkpoints)

                # ── 2e. Step Execution ──────────────────────────────────────────
                # Race the step against the steer wake event so a steer can
                # interrupt the step mid-stream (while reasoning/text parts are
                # still printing). The interrupted step's partial output is not
                # grown into the context; the steer is injected as a follow-up
                # user message and a fresh step starts with it in context.
                self._steer_wake_event.clear()
                step_task = asyncio.create_task(self._step())
                self._current_step_task = step_task
                steer_waiter = asyncio.create_task(self._steer_wake_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        {step_task, steer_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    steer_waiter.cancel()
                    with suppress(asyncio.CancelledError):
                        await steer_waiter
                    if not step_task.done():
                        step_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await step_task
                    self._current_step_task = None
                if step_task in done:
                    # Normal completion (or the step finished exactly as the
                    # steer arrived) — exceptions re-raise into the existing
                    # except clauses below.
                    step_outcome = step_task.result()
                else:
                    # Steer-interrupt path: the wake event fired first and the
                    # step was cancelled above. Inject the steers as user
                    # messages and end the interrupted step on the wire.
                    await self._consume_pending_steers()
                    wire_send(StepInterrupted())
                    continue

            except SessionRestartRequired:
                # ── 2f-i. Session restart signal ─────────────────────────────
                # Propagate to outer layers (Session.prompt) for auto-restart.
                # Do NOT send StepInterrupted here — the session will be cleared
                # and restarted, so reporting an error on the old session is
                # misleading.
                raise

            except BackToTheFuture as e:
                # ── 2f-ii. D-Mail revert signal ────────────────────────────────
                back_to_the_future = e

            except Exception as e:
                # ── 2f-iii. Fatal step error ──────────────────────────────────
                req_id = getattr(e, "request_id", None)
                logger.error(
                    "Agent step {step_no} failed: {error_type}: {error}"
                    + (" (request_id={request_id})" if req_id else ""),
                    step_no=step_no,
                    error_type=type(e).__name__,
                    error=e,
                    request_id=req_id,
                )
                wire_send(StepInterrupted())

                # --- StopFailure hook ---
                from kimi_cli.hooks import events as _hook_events

                _hook_task = asyncio.create_task(
                    self._hook_engine.trigger(
                        "StopFailure",
                        matcher_value=type(e).__name__,
                        input_data=_hook_events.stop_failure(
                            session_id=self._runtime.session.id,
                            cwd=str(self._runtime.session.work_dir),
                            error_type=type(e).__name__,
                            error_message=str(e),
                        ),
                    )
                )
                _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                # break the agent loop
                raise

            # ── 2g. Outcome Resolution ──────────────────────────────────────────
            if step_outcome is not None:
                # Step returned a stop reason -- check for steers before finishing.
                has_steers = await self._consume_pending_steers()
                if has_steers:
                    continue  # steers injected, force another LLM step

                # ── 2h. Verification Gate (P2, B-3) ────────────────────────────
                # Before ending the turn on no_tool_calls, check whether the
                # turn is actually finished (todos done, verifications run).
                if (
                    step_outcome.stop_reason == "no_tool_calls"
                    and self._loop_control.verification_gate_enabled
                ):
                    gate_msg = await self._verification_gate.check(self)
                    if gate_msg is not None:
                        await self._context.append_message(
                            Message(
                                role="user",
                                content=[TextPart(text=system_reminder(gate_msg).text)],
                            )
                        )
                        continue  # do not end the turn; force another step

                # ═══════════════════════════════════════════════════════════════
                # 3. TURN RESOLUTION
                # ═══════════════════════════════════════════════════════════════
                final_message = (
                    step_outcome.assistant_message
                    if step_outcome.stop_reason == "no_tool_calls"
                    else None
                )
                return TurnOutcome(
                    stop_reason=step_outcome.stop_reason,
                    final_message=final_message,
                    step_count=step_no,
                )

            if back_to_the_future is not None:
                # Revert context to the checkpoint and inject D-Mail message.
                await self._context.revert_to(back_to_the_future.checkpoint_id)
                self._last_tool_calls = []
                await self._checkpoint()
                await self._context.append_message(back_to_the_future.messages)

            # Consume any pending steers between steps before next iteration.
            await self._consume_pending_steers()

    def _cache_depth_floor(self, history_len: int) -> int | None:
        """Compute the pruner's cache-depth floor for a history of *history_len*.

        cache-03: protects the whole provider-cached head from a single prune
        pass. Uses ``loop_control.prune_min_cache_prefix_depth`` when set
        (``0`` disables the floor); otherwise derives a dynamic floor that
        protects everything except the recent tail band — the last
        ``prune_recent_messages_protected`` turns plus 8 messages, which the
        provider re-computes for the next request regardless.
        """
        config_floor = self._loop_control.prune_min_cache_prefix_depth
        if config_floor is not None:
            return config_floor if config_floor > 0 else None
        tail_band = self._loop_control.prune_recent_messages_protected + 8
        return max(0, history_len - tail_band)

    async def _step(
        self, _overflow_state: OverflowRecoveryState | None = None
    ) -> StepOutcome | None:
        """Run a single step and return a stop outcome, or None to continue.

        This is the implementation of ``2e. Step Execution`` in ``_agent_loop``.

        Args:
            _overflow_state: Per-step overflow retry budget shared across the
                re-entrant ``_step`` calls made by the context-overflow recovery
                loop (Phase 4 §6.2). ``None`` (the ``_agent_loop`` call site)
                creates a fresh budget for this top-level step; the overflow
                branch re-enters with the same instance so the budget is
                consumed across re-entries instead of being reset.

        Sub-lifecycle (2e.x):
            2e.1. Notification delivery  - push pending notifications (root only).
            2e.2. Dynamic injection      - collect and append provider injections.
            2e.3. History normalization  - merge adjacent user messages.
            2e.4. LLM call with retry    - kosong.step + tenacity retry + recovery.
               2e.4.1. Toolset begin_step- reset per-step dedup state.
               2e.4.2. kosong.step       - actual LLM call (may be interrupted).
            2e.5. Usage & status update  - track tokens, emit StatusUpdate.
            2e.6. Tool execution         - wait for all tool results.
            2e.7. Context growth         - append assistant + tool messages.
            2e.8. Outcome resolution     - rejection / D-Mail / stop / continue.
        """
        # already checked in `run`
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider

        # Phase 4: overflow retry budget for this top-level step. A fresh state
        # is created per top-level invocation (matching the plan's "reset on
        # each step begin"); re-entrant overflow calls share the same state via
        # ``_overflow_state`` so the budget is bounded across re-entries.
        overflow_state = _overflow_state or OverflowRecoveryState(
            self._loop_control.context_overflow_retries
        )

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.1. NOTIFICATION DELIVERY (root role only)
        # ═══════════════════════════════════════════════════════════════════════
        if self.is_root:

            async def _append_notification(view: NotificationView) -> None:
                await self._context.append_message(build_notification_message(view, self._runtime))
                # --- Notification hook ---
                from kimi_cli.hooks import events

                _hook_task = asyncio.create_task(
                    self._hook_engine.trigger(
                        "Notification",
                        matcher_value=view.event.type,
                        input_data=events.notification(
                            session_id=self._runtime.session.id,
                            cwd=str(self._runtime.session.work_dir),
                            sink="llm",
                            notification_type=view.event.type,
                            title=view.event.title,
                            body=view.event.body,
                            severity=view.event.severity,
                        ),
                    )
                )
                _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

            await self._runtime.notifications.deliver_pending(
                "llm",
                limit=4,
                before_claim=self._runtime.background_tasks.reconcile,
                on_notification=_append_notification,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.2. DYNAMIC INJECTION
        # ═══════════════════════════════════════════════════════════════════════

        # ── 2e.2a. Strip stale system reminders from previous steps/turns ─────
        # Providers re-inject fresh reminders below, so removing old ones is safe
        # (reminders are ephemeral: one fresh copy per step, never accumulated).
        # NOTE: this must NOT notify providers of "compaction" — doing so resets
        # their throttling state every step and makes throttled reminders (context
        # meter, todo, budget, compact) re-inject on every single step. Providers
        # keep their own cooldown state and re-decide below.
        strip_system_reminders(self._context._history)

        auto_retrieval_injections = await self._maybe_auto_retrieve_history()
        injections = await self._collect_injections()
        # Prepend auto-retrieved injections so they appear before provider injections
        for inj in reversed(auto_retrieval_injections):
            injections.insert(0, inj)
        if injections:
            combined_reminders = "\n".join(system_reminder(inj.content).text for inj in injections)
            await self._context.append_message(
                Message(
                    role="user",
                    content=[TextPart(text=combined_reminders)],
                )
            )

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.3. CONTEXT PRUNING (smart history removal, Layer 1)
        # ═══════════════════════════════════════════════════════════════════════
        # Run pruner on the LLM-visible history (non-destructive; storage intact).
        # Tier A drops consumed ephemera; Tier B (if enabled) elides stale content.
        history_for_llm = list(self._context.history)
        if self._loop_control.context_pruning_enabled and (
            self.is_root or self._loop_control.prune_subagents
        ):
            llm = self._runtime.llm
            max_context = llm.max_context_size if llm else 128_000
            model_name = llm.chat_provider.model_name if llm else None
            prune_result = self._pruner.prune(
                history_for_llm,
                current_step=self._current_step_no,
                context_usage=self.status.context_usage,
                max_context_size=max_context,
                model=model_name,
                current_turn_index=_current_turn_start_index(history_for_llm),
                min_cache_prefix_depth=self._cache_depth_floor(len(history_for_llm)),
            )
            if prune_result.earliest_removed_index is not None:
                cache_loss = count_message_tokens(
                    history_for_llm[prune_result.earliest_removed_index :],
                    model=model_name,
                )
                logger.info(
                    "Context pruner freed {freed} tokens, earliest_removed_index={idx}, "
                    "estimated_cache_loss={cache_loss} tokens, Tier B count={tier_b}",
                    freed=prune_result.freed_tokens,
                    idx=prune_result.earliest_removed_index,
                    cache_loss=cache_loss,
                    tier_b=len(prune_result.elided),
                )
                # Feed Tier B elided records into HistoryIndex for retrieval
                if prune_result.elided:
                    elided_messages = []
                    for rec in prune_result.elided:
                        # Reconstruct a Message from the elided record
                        msg = Message(
                            role=rec.role,
                            content=[TextPart(text=rec.original_text)],
                        )
                        elided_messages.append(msg)
                    self._history_index.index_messages(elided_messages)
                    self._history_index.save()
                # Track restored refs to avoid re-pruning restored content
                for rec in prune_result.elided:
                    self._recently_restored_refs.add(rec.ref)
                history_for_llm = prune_result.messages

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.4. HISTORY NORMALIZATION
        # ═══════════════════════════════════════════════════════════════════════
        effective_history = normalize_history(history_for_llm)

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.4. LLM CALL WITH RETRY
        # ═══════════════════════════════════════════════════════════════════════
        step_attempt = 0

        async def _run_step_once() -> StepResult:
            """Single LLM invocation (wrapped by retry + connection recovery)."""
            nonlocal step_attempt
            step_attempt += 1
            # ── 2e.4.1. Toolset begin_step ────────────────────────────────────
            if isinstance(self._agent.toolset, KimiToolset):
                self._agent.toolset.begin_step(self._last_tool_calls)
            # ── 2e.4.2. kosong.step ───────────────────────────────────────────
            system_prompt = self._agent.get_system_prompt()
            self._drain_mcp_discoveries()
            self._llm_request_recorder.record(
                chat_provider,
                system_prompt,
                self._agent.toolset.tools,
                effective_history,
                kind="loop",
                turn_step=self._current_step_no,
                attempt=step_attempt,
            )
            # run an LLM step (may be interrupted)
            return await kosong.step(
                chat_provider,
                system_prompt,
                self._agent.toolset,
                effective_history,
                on_message_part=wire_send,
                on_tool_result=wire_send,
            )

        max_attempts = self._loop_control.max_retries_per_step

        def _before_step_retry_sleep(retry_state: RetryCallState) -> None:
            self._retry_log("step", retry_state)
            self._emit_step_retry(retry_state, max_attempts=max_attempts)
            # When retrying a think-only response (token budget exhaustion during reasoning),
            # increase max_tokens progressively so the model has room to emit text after
            # completing its reasoning chain.
            if retry_state.outcome and retry_state.outcome.exception():
                exc = retry_state.outcome.exception()
                if isinstance(exc, APIEmptyResponseError):
                    # Only set the token-limit key the provider understands:
                    # - openai-responses / google_genai -> max_output_tokens
                    # - everything else                 -> max_tokens
                    _output_token_providers = frozenset({"openai-responses", "google_genai"})
                    _token_key = "max_output_tokens" if chat_provider.name in _output_token_providers else "max_tokens"
                    current = getattr(chat_provider, '_generation_kwargs', {}).get(_token_key) or 8192
                    new_max = int(current * 1.5)
                    logger.info(
                        "Think-only response (attempt {attempt}), increasing {key} "
                        "from {old} to {new} for retry",
                        attempt=retry_state.attempt_number,
                        key=_token_key,
                        old=current,
                        new=new_max,
                    )
                    chat_provider._generation_kwargs[_token_key] = new_max

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=_before_step_retry_sleep,
            wait=_RETRY_WAIT,
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )
        async def _kosong_step_with_retry() -> StepResult:
            return await self._run_with_connection_recovery(
                "step",
                _run_step_once,
                chat_provider=chat_provider,
            )

        t0 = time.monotonic()
        try:
            result = await _kosong_step_with_retry()
            # Phase 4: the step made progress — reset the overflow retry budget.
            # A fresh state is created per top-level ``_step`` anyway, so this is
            # belt-and-suspenders mirroring DSH resetting on agent/status idle
            # and assistant/message.
            overflow_state.reset()
        except APIEmptyResponseError:
            # All retries exhausted — the model keeps producing only thinking
            # content (output token budget exhausted during reasoning).
            # Stop the turn instead of looping infinitely.
            logger.warning(
                "All retries exhausted for think-only response at step {step_no}, "
                "stopping turn",
                step_no=self._current_step_no,
            )
            wire_send(TextPart(
                text="\n(The model produced only thinking content without a response. Stopping this turn.)\n"
            ))
            return StepOutcome(
                stop_reason="no_tool_calls",
                assistant_message=Message(role="assistant", content=[]),
            )
        except (APIStatusError, APIConnectionError, APITimeoutError) as e:
            # All retries exhausted for persistent API errors.
            # Phase 4: context-overflow recovery loop (DSH port). A
            # provider-confirmed context-window-exceeded error force-compacts the
            # context and retries the step instead of immediately interrupting
            # the session. ``overflow_state`` (shared across the re-entrant
            # ``_step`` calls below) bounds the re-entries to
            # ``context_overflow_retries`` per top-level step, so the loop
            # cannot recurse unboundedly (``max_steps_per_turn`` is an
            # additional outer guard).
            #
            # NOTE: the forced compaction bypasses ``should_auto_compact``
            # entirely — ``compact_context`` is called directly — so
            # ``context_overflow_force_threshold`` is effectively always True in
            # this branch (its plan default). The flag is kept in config as the
            # documented DSH "force one useful balanced reduction" switch; when
            # it is False this branch still force-compacts (otherwise the
            # recovery loop would be useless).
            if (
                is_context_overflow_error(e)
                and self._loop_control.context_overflow_retries > 0
                and overflow_state.can_retry()
            ):
                logger.warning(
                    "Context window exceeded at step {step}; force-compacting and retrying",
                    step=self._current_step_no,
                )
                try:
                    await self.compact_context(
                        manual=False,
                        mode=CompactMode.AGGRESSIVE,
                        options=CompactionOptions(
                            preserve_depth_override=(
                                self._loop_control.context_overflow_preserve_depth
                            ),
                        ),
                        trigger_override="overflow",
                    )
                except Exception as compact_err:
                    logger.error(
                        "Overflow compaction failed: {err}; preserving original error",
                        err=compact_err,
                    )
                    raise SessionRestartRequired(
                        f"Step {self._current_step_no}: {type(e).__name__}"
                        + (
                            f" (status={e.status_code})"
                            if isinstance(e, APIStatusError)
                            else ""
                        )
                        + " — context overflow recovery compaction failed, "
                        "restarting session",
                        original_error=e,
                    ) from e
                overflow_state.consumed()
                # Re-run the same step on the compacted context, sharing the
                # retry budget so the recovery loop stays bounded.
                return await self._step(_overflow_state=overflow_state)
            # Existing generic handling: interrupt the session and restart with
            # the same user input.
            recovery_exhausted = getattr(e, "_kimi_recovery_exhausted", False)
            raise SessionRestartRequired(
                f"Step {self._current_step_no}: {type(e).__name__}"
                + (f" (status={e.status_code})" if isinstance(e, APIStatusError) else "")
                + (" [connection recovery exhausted]" if recovery_exhausted else "")
                + " — retries exhausted, restarting session",
                original_error=e,
            ) from e

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.5. USAGE & STATUS UPDATE
        # ═══════════════════════════════════════════════════════════════════════
        llm_elapsed = time.monotonic() - t0
        usage = result.usage
        logger.info(
            "LLM step completed in {elapsed:.1f}s (input={input_tokens}, output={output_tokens})",
            elapsed=llm_elapsed,
            input_tokens=usage.input if usage else "?",
            output_tokens=usage.output if usage else "?",
        )
        status_update = StatusUpdate(
            token_usage=usage, message_id=result.id
        )
        if usage is not None:
            # mark the token count for the context before the step
            await self._context.update_token_count(usage.input)
            snap = self.status
            status_update.context_usage = snap.context_usage
            status_update.context_tokens = snap.context_tokens
            status_update.max_context_tokens = snap.max_context_tokens
        wire_send(status_update)

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.6. TOOL EXECUTION
        # ═══════════════════════════════════════════════════════════════════════
        # wait for all tool results (may be interrupted)
        results = await result.tool_results()
        logger.debug("Got tool results: {results}", results=results)
        # Update dedup tracking for the next step
        if isinstance(self._agent.toolset, KimiToolset):
            self._last_tool_calls = self._agent.toolset.end_step()

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.7. CONTEXT GROWTH
        # ═══════════════════════════════════════════════════════════════════════
        # shield the context manipulation from interruption
        await asyncio.shield(self._grow_context(result, results))

        # ═══════════════════════════════════════════════════════════════════════
        # 2e.8. OUTCOME RESOLUTION
        # ═══════════════════════════════════════════════════════════════════════
        rejected_errors = [
            result.return_value
            for result in results
            if isinstance(result.return_value, ToolRejectedError)
        ]
        if rejected_errors and not any(e.has_feedback for e in rejected_errors) and self.is_root:
            # Pure rejection (no user feedback) — stop the turn.
            # Subagents skip this so the LLM can see the rejection and try
            # an alternative approach instead of terminating immediately.
            _ = self._denwa_renji.fetch_pending_dmail()
            return StepOutcome(stop_reason="tool_rejected", assistant_message=result.message)

        # handle pending D-Mail
        if dmail := self._denwa_renji.fetch_pending_dmail():
            assert dmail.checkpoint_id >= 0, "DenwaRenji guarantees checkpoint_id >= 0"
            assert dmail.checkpoint_id < self._context.n_checkpoints, (
                "DenwaRenji guarantees checkpoint_id < n_checkpoints"
            )
            # raise to let the main loop take us back to the future
            raise BackToTheFuture(
                dmail.checkpoint_id,
                [
                    Message(
                        role="user",
                        content=[
                            system(
                                "You just got a D-Mail from your future self. "
                                "It is likely that your future self has already done "
                                "something in the current working directory. Please read "
                                "the D-Mail and decide what to do next. You MUST NEVER "
                                "mention to the user about this information. "
                                f"D-Mail content:\n\n{dmail.message.strip()}"
                            )
                        ],
                    )
                ],
            )

        if isinstance(self._agent.toolset, KimiToolset) and self._agent.toolset.force_stop_turn:
            return StepOutcome(stop_reason="tool_call_repeat", assistant_message=result.message)

        if result.tool_calls:
            return None
        return StepOutcome(stop_reason="no_tool_calls", assistant_message=result.message)

    async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
        logger.debug("Growing context with result: {result}", result=result)

        assert self._runtime.llm is not None
        tool_messages = [tool_result_to_message(tr) for tr in tool_results]
        for tm in tool_messages:
            if missing_caps := check_message(tm, self._runtime.llm.capabilities):
                logger.warning(
                    "Tool result message requires unsupported capabilities: {caps}",
                    caps=missing_caps,
                )
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._context.append_message(result.message)
        if result.usage is not None:
            await self._context.update_token_count(result.usage.total)

        logger.debug(
            "Appending tool messages to context: {tool_messages}", tool_messages=tool_messages
        )
        await self._context.append_message(tool_messages)
        # token count of tool results are not available yet

    async def compact_context(
        self,
        *,
        manual: bool = False,
        custom_instruction: str = "",
        avoid_cascade: bool = False,
        mode: CompactMode = CompactMode.BALANCED,
        options: CompactionOptions | None = None,
        trigger_override: Literal["auto", "manual", "overflow"] | None = None,
    ) -> None:
        """
        Compact the context.

        Args:
            manual: Whether the compaction was explicitly requested by the user
                (e.g. via the ``/compact`` slash command). When ``False``, the
                compaction is treated as auto-triggered by the system.
            custom_instruction: Optional user instruction to guide compaction focus.
            avoid_cascade: When ``True``, always use the structured ``COMPACT``
                prompt instead of the cascade prompt, regardless of compaction depth.
            mode: High-level compaction style / emphasis. Does not change preserve
                depth or adaptive preserve behavior.
            options: Base ``CompactionOptions`` for this compaction. When provided
                it is used as the base (e.g. the overflow recovery branch passes
                ``preserve_depth_override``); ``todos_max_items`` is filled from
                loop control when not set.
            trigger_override: Explicit wire/ledger trigger. When set, it wins over
                the derived ``manual``/``auto`` classification (the overflow
                recovery branch passes ``"overflow"``).

        Raises:
            LLMNotSet: When the LLM is not set.
            ChatProviderError: When the chat provider returns an error.
            ManualCompactionError: For manual compactions, classified failures
                (``changed``/``summary``) raised by the stability/shrink checks.
        """

        chat_provider = self._runtime.llm.chat_provider if self._runtime.llm is not None else None

        # Phase 4: the wire/ledger trigger. ``trigger_override`` (used by the
        # overflow recovery branch) wins; otherwise derive from ``manual``.
        trigger: Literal["auto", "manual", "overflow"] = (
            trigger_override if trigger_override is not None else ("manual" if manual else "auto")
        )
        # Merge the caller's options (Phase 4 overflow passes
        # ``preserve_depth_override``) with the session defaults.
        base_options = options if options is not None else CompactionOptions()
        compaction_options = CompactionOptions(
            avoid_cascade=base_options.avoid_cascade,
            mode=base_options.mode,
            todos_max_items=(
                base_options.todos_max_items
                if base_options.todos_max_items is not None
                else self._loop_control.todo_compact_injection_max_items
            ),
            preserve_depth_override=base_options.preserve_depth_override,
        )

        async def _run_compaction_once() -> CompactionResult:
            if self._runtime.llm is None:
                raise LLMNotSet()
            await self._ensure_recorder_restored()
            # Phase 2: KV-cache-aligned summarization input — replay the real
            # system prompt + tools so the provider's cacheable prefix stays
            # aligned. Only for KimiToolset souls; other toolsets (e.g.
            # ``EmptyToolset``/``SimpleToolset``) fall back to the legacy
            # flattened path.
            aligned_kwargs: dict[str, Any] = {}
            if isinstance(self._agent.toolset, KimiToolset):
                aligned_kwargs = {
                    "aligned_system_prompt": self._agent.get_system_prompt(),
                    "aligned_tools": list(self._agent.toolset.tools),
                }
            return await self._compaction.compact(
                self._context.history,
                self._runtime.llm,
                custom_instruction=custom_instruction,
                options=compaction_options,
                recorder=self._llm_request_recorder,
                todos_loader=(
                    self._load_todo_states_for_reminder
                    if self._loop_control.todo_compact_injection_enabled
                    else None
                ),
                ledger=self._compaction_ledger,
                trigger=trigger,
                **aligned_kwargs,
            )

        start_time = time.monotonic()
        retry_count = 0

        def _retry_log_compaction(retry_state: RetryCallState) -> None:
            nonlocal retry_count
            retry_count = retry_state.attempt_number
            self._retry_log("compaction", retry_state)

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=_retry_log_compaction,
            wait=_RETRY_WAIT,
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _compact_with_retry() -> CompactionResult:
            return await self._run_with_connection_recovery(
                "compaction",
                _run_compaction_once,
                chat_provider=chat_provider,
            )

        if not manual:
            trigger_reason = "auto"
        elif custom_instruction:
            trigger_reason = "manual-with-prompt"
        else:
            trigger_reason = "manual"
        before_tokens = self._context.token_count
        from kimi_cli.hooks import events

        await self._hook_engine.trigger(
            "PreCompact",
            matcher_value=trigger_reason,
            input_data=events.pre_compact(
                session_id=self._runtime.session.id,
                cwd=str(self._runtime.session.work_dir),
                trigger=trigger_reason,
                token_count=before_tokens,
            ),
        )

        # (Pre-compaction durable-state flush removed: Retrieve is history-only.)
        # Phase 3 transaction envelope: the Begin/End wire pair carries the
        # compaction_id (a provisional uuid generated before the LLM call;
        # ``CompactionEnd`` prefers the authoritative id returned by the
        # compaction, falling back to the provisional one for no-op compactions),
        # the trigger, and best-effort token accounting. ``CompactionEnd`` is
        # always emitted — on success with stats, on failure with the error — so
        # the wire always sees a balanced Begin/End pair.
        compaction_id = uuid.uuid4().hex
        wire_send(
            CompactionBegin(
                compaction_id=compaction_id,
                trigger=trigger,
                shadowed_tokens=None,
            )
        )

        async def _compact_with_stability_retry() -> CompactionResult:
            """Phase 3 stability check: if the conversation surface changed while
            the summary was generated, re-prepare and try once more before
            failing. A second ``SurfaceChangedError`` is a classified manual
            error (``changed``) for ``/compact``; auto compactions re-raise so
            the caller decides."""
            try:
                return await _compact_with_retry()
            except SurfaceChangedError:
                logger.warning(
                    "Conversation changed during compaction; re-preparing and retrying once"
                )
                try:
                    return await _compact_with_retry()
                except SurfaceChangedError as second_err:
                    if manual:
                        raise ManualCompactionError(
                            "changed", str(second_err)
                        ) from second_err
                    raise

        try:
            try:
                compaction_result = await _compact_with_stability_retry()
            except CompactionShrinkError as shrink_err:
                # Phase 3 shrink check: the summary must be smaller than the
                # region it replaces. Manual compactions surface a classified
                # error (``summary``); auto compactions re-raise as-is.
                if manual:
                    raise ManualCompactionError(
                        "summary", str(shrink_err)
                    ) from shrink_err
                raise

            # Mark all indexed turns as archived before clearing context
            self._history_index.mark_compacted()
            self._history_index.save()

            # --- Export pre-compaction context ---
            # cache-05: deterministic per-session export slot (one snapshot per
            # compaction, latest wins) instead of a random ``token_hex`` suffix —
            # the export path is embedded in the compacting system prompt, so a
            # random nonce would make every post-compaction prompt unique and
            # block prefix-cache continuity across runs/compactions.
            rotated_path = self._runtime.session.work_dir / ".kimix_cache" / "context_compacted.md"
            self._compact_cache_dir.append(rotated_path)
            if rotated_path is not None:
                export_result = await perform_export(
                    history=list(self._context.history),
                    session_id=self._runtime.session.id,
                    work_dir=str(self._runtime.session.work_dir),
                    token_count=self._context.token_count,
                    args=str(rotated_path),
                    default_dir=self._runtime.session.dir,
                )
                if isinstance(export_result, tuple):
                    compact_export_path = str(export_result[0])
                    logger.info("Pre-compaction context exported to: {path}", path=compact_export_path)
                else:
                    logger.warning("Failed to export pre-compaction context: {error}", error=export_result)
                    compact_export_path = None
            else:
                compact_export_path = None

            self._recently_retrieved_turn_ids.clear()
            self._pruner.reset_cooldown()
            await self._context.clear()
            # cache-05: render the compacting prompt (deterministic — the same
            # arguments always produce the same string) and promote it to the
            # persistent cache slot ONLY here, after the export attempt, so the
            # normal prompt slot is never silently overwritten by a compacting
            # render.
            system_prompt_text = self._agent.get_system_prompt(
                is_compacting=True, compact_export_path=compact_export_path
            )
            self._agent.system_prompt_cached = system_prompt_text
            await self._context.write_system_prompt(system_prompt_text)
            await self._checkpoint()
            await self._context.append_message(compaction_result.messages)

            if self.is_root:
                active_task_snapshot = build_active_task_snapshot(self._runtime.background_tasks)
                if active_task_snapshot is not None:
                    active_task_message = Message(
                        role="user",
                        content=[
                            system(
                                "The following background tasks are still active after compaction. "
                                "Use TaskList if you need to re-enumerate them later."
                            ),
                            TextPart(text=active_task_snapshot),
                        ],
                    )
                    await self._context.append_message(active_task_message)

            # (Post-compaction state-restore message removed: Retrieve is history-only.)

            # Recompute the token estimate from the rebuilt context so it reflects
            # the checkpoint marker, preserved messages, compaction summary, and any
            # active-task snapshot. Also account for the system prompt, because the
            # pre-compaction usage snapshots that callers compare against also
            # include the system prompt.
            estimated_token_count = estimate_text_tokens(
                self._context.history, model=self.model_name
            )
            if system_prompt_text:
                estimated_token_count += count_tokens(
                    system_prompt_text, model=self.model_name
                )

            # Estimate token count so context_usage is not reported as 0%
            await self._context.update_token_count(estimated_token_count)

            # Notify dynamic injection providers that history has been rebuilt so
            # they can reset any one-shot throttling state. Failures are isolated
            # per-provider so compaction completion (wire event) is not affected
            # by a buggy provider.
            await self._notify_injection_providers_compacted()

            wire_send(
                CompactionEnd(
                    compaction_id=compaction_result.compaction_id or compaction_id,
                    trigger=trigger,
                    shadowed_tokens=compaction_result.shadowed_tokens,
                    estimated_token_count=estimated_token_count,
                )
            )
        except Exception as exc:
            # End-of-transaction on failure: the ledger already records the error
            # inside ``SimpleCompaction.compact`` (Phase 3 §5.2); do not
            # double-record here — just make sure the wire sees a paired
            # ``CompactionEnd`` carrying the failure.
            wire_send(
                CompactionEnd(
                    compaction_id=compaction_id,
                    trigger=trigger,
                    error=str(exc),
                )
            )
            raise

        _hook_task = asyncio.create_task(
            self._hook_engine.trigger(
                "PostCompact",
                matcher_value=trigger_reason,
                input_data=events.post_compact(
                    session_id=self._runtime.session.id,
                    cwd=str(self._runtime.session.work_dir),
                    trigger=trigger_reason,
                    estimated_token_count=estimated_token_count,
                ),
            )
        )
        _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    @staticmethod
    def _sync_cleanup(
        compact_cache_paths: list[Path],
        file_backend: Path,
        anonymous: bool,
    ) -> None:
        """Best-effort synchronous cleanup of rotated compaction export files.

        This is used as a ``weakref.finalize`` callback so cleanup still runs
        even when ``KimiSoul`` is part of a reference cycle.

        Only deletes files when *anonymous* is True (the session is anonymous).
        """
        if not anonymous:
            return
        for compact_path in compact_cache_paths:
            with suppress(Exception):
                os.remove(str(compact_path))
        with suppress(Exception):
            os.remove(str(file_backend))
        # Clean up SQLite WAL/SHM companion files (for .db backends)
        if file_backend.suffix == ".db":
            for companion_suffix in (".db-wal", ".db-shm"):
                companion = file_backend.with_suffix(companion_suffix)
                with suppress(Exception):
                    os.remove(str(companion))

    async def __aenter__(self) -> KimiSoul:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Persist state and delete rotated compaction export files/context before process exit."""
        # Persist any turns indexed since the last explicit save.
        try:
            self._history_index.save()
        except Exception:
            logger.exception("Failed to save history index")

        # Close the FTS5-backed history index (apsw connection) so the
        # session directory can be moved/deleted on Windows without a lock.
        try:
            self._history_index.close()
        except Exception:
            logger.exception("Failed to close history index")

        # Break the soul <-> context reference cycle so the object can be
        # reclaimed promptly without waiting for cyclic GC.
        with suppress(Exception):
            self._context._on_append = None

        # Close the context storage backend (aiosqlite worker thread) so it
        # does not block Python's threading._shutdown() on interpreter exit.
        try:
            await self._context.close()
        except Exception:
            logger.exception("Failed to close context storage")

        # Close the chat provider's underlying HTTP client before the event loop
        # shuts down.  This avoids noisy "Event loop is closed" RuntimeErrors from
        # the httpx/anyio transport teardown on Windows/Python 3.14.
        if self._runtime.llm is not None:
            chat_provider = self._runtime.llm.chat_provider
            if hasattr(chat_provider, "aclose"):
                try:
                    await chat_provider.aclose()  # type: ignore[misc]
                except Exception:
                    logger.exception("Failed to close chat provider")

        if self._anonymous:
            for compact_path in self._compact_cache_dir:
                try:
                    await asyncio.to_thread(os.remove, str(compact_path))
                    logger.debug("Removed rotated compaction export: {path}", path=compact_path)
                except FileNotFoundError:
                    logger.debug("Rotated compaction export already gone: {path}", path=compact_path)
                except Exception:
                    logger.exception("Failed to remove rotated compaction export: {path}", path=compact_path)
            self._compact_cache_dir.clear()

            try:
                await asyncio.to_thread(os.remove, str(self._context.file_backend))
                logger.debug("Removed context file: {path}", path=self._context.file_backend)
            except FileNotFoundError:
                logger.debug("Context file already gone: {path}", path=self._context.file_backend)
            except Exception:
                logger.exception("Failed to remove context file: {path}", path=self._context.file_backend)

            # Clean up SQLite WAL/SHM companion files
            fb = self._context.file_backend
            if fb.suffix == ".db":
                for companion_suffix in (".db-wal", ".db-shm"):
                    companion = fb.with_suffix(companion_suffix)
                    try:
                        await asyncio.to_thread(os.remove, str(companion))
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logger.exception("Failed to remove companion file: {path}", path=companion)

        # Cleanup already happened; detach the finalizer so it does not run again.
        self._finalizer.detach()

    @staticmethod
    def _is_retryable_error(exception: BaseException) -> bool:
        if isinstance(exception, (APIConnectionError, APITimeoutError)):
            return not bool(getattr(exception, "_kimi_recovery_exhausted", False))
        if isinstance(exception, APIEmptyResponseError):
            return True
        return isinstance(exception, APIStatusError) and exception.status_code in (
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        )

    async def _run_with_connection_recovery(
        self,
        name: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        chat_provider: object | None = None,
        _auth_retried: bool = False,
        _connection_retried: bool = False,
    ) -> Any:
        try:
            return await operation()
        except APIStatusError as error:
            if error.status_code != 401 or _auth_retried:
                raise
            # Only attempt refresh+retry when the active model's provider
            # uses OAuth.  For plain API-key providers there is nothing
            # to refresh and retrying would just add latency.
            active_provider = (
                self._runtime.config.provider
                if self._runtime.llm and self._runtime.llm.model_config
                else None
            )
            if not (active_provider and active_provider.oauth):
                raise
            logger.warning(
                "Received 401 during {name}, attempting token refresh",
                name=name,
            )
            try:
                await self._runtime.oauth.ensure_fresh(self._runtime, force=True)
            except Exception as refresh_exc:
                logger.exception("Token refresh failed after 401.")
                raise error from refresh_exc
            # Re-enter full recovery so that transient connection errors
            # on the retry are still handled by on_retryable_error.
            return await self._run_with_connection_recovery(
                name,
                operation,
                chat_provider=chat_provider,
                _auth_retried=True,
                _connection_retried=_connection_retried,
            )
        except (APIConnectionError, APITimeoutError) as error:
            if _connection_retried:
                logger.warning(
                    "Chat provider recovery exhausted for {name}: {error_type}: {error}",
                    name=name,
                    error_type=type(error).__name__,
                    error=error,
                )
                error._kimi_recovery_exhausted = True  # type: ignore[attr-defined]
                raise
            if not isinstance(chat_provider, RetryableChatProvider):
                raise
            try:
                recovered = chat_provider.on_retryable_error(error)
            except Exception:
                logger.exception(
                    "Failed to recover chat provider during {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            if not recovered:
                logger.warning(
                    "Chat provider recovery not available for {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            logger.info(
                "Recovered chat provider during {name} after {error_type}; retrying once.",
                name=name,
                error_type=type(error).__name__,
            )
            # Re-enter the full recovery path so a 401 on the retry can still
            # trigger OAuth refresh instead of bubbling straight to the user.
            return await self._run_with_connection_recovery(
                name,
                operation,
                chat_provider=chat_provider,
                _auth_retried=_auth_retried,
                _connection_retried=True,
            )

    @staticmethod
    def _retry_log(name: str, retry_state: RetryCallState):
        error = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying {name} for the {n} time (last error: {error_type}: {error}). "
            "Waiting {sleep} seconds.",
            name=name,
            n=retry_state.attempt_number,
            error_type=type(error).__name__ if error else "unknown",
            error=error or "unknown",
            sleep=retry_state.next_action.sleep
            if retry_state.next_action is not None
            else "unknown",
        )

    def _emit_step_retry(self, retry_state: RetryCallState, *, max_attempts: int) -> None:
        error = retry_state.outcome.exception() if retry_state.outcome else None
        next_action = retry_state.next_action
        wait_s = next_action.sleep if next_action is not None else 0.0
        wire_send(
            StepRetry(
                n=self._current_step_no,
                next_attempt=retry_state.attempt_number + 1,
                max_attempts=max_attempts,
                wait_s=wait_s,
                error_type=type(error).__name__ if error else "unknown",
                status_code=error.status_code if isinstance(error, APIStatusError) else None,
            )
        )


class BackToTheFuture(Exception):
    """
    Raise when we need to revert the context to a previous checkpoint.
    The main agent loop should catch this exception and handle it.
    """

    def __init__(self, checkpoint_id: int, messages: Sequence[Message]):
        self.checkpoint_id = checkpoint_id
        self.messages = messages


class FlowRunner:
    def __init__(
        self,
        flow: Flow,
        *,
        name: str | None = None,
        max_moves: int = DEFAULT_MAX_FLOW_MOVES,
    ) -> None:
        self._flow = flow
        self._name = name
        self._max_moves = max_moves

    @staticmethod
    def ralph_loop(
        user_message: Message,
        max_ralph_iterations: int,
    ) -> FlowRunner:
        prompt_content = list(user_message.content)
        prompt_text = Message(role="user", content=prompt_content).extract_text(" ").strip()
        total_runs = max_ralph_iterations + 1
        if max_ralph_iterations < 0:
            total_runs = 1000000000000000  # effectively infinite

        nodes: dict[str, FlowNode] = {
            "BEGIN": FlowNode(id="BEGIN", label="BEGIN", kind="begin"),
            "END": FlowNode(id="END", label="END", kind="end"),
        }
        outgoing: dict[str, list[FlowEdge]] = {"BEGIN": [], "END": []}

        nodes["R1"] = FlowNode(id="R1", label=prompt_content, kind="task")
        nodes["R2"] = FlowNode(
            id="R2",
            label=(
                f"{prompt_text}. (Automated loop — choose STOP only when fully complete. "
                "If unsure, choose CONTINUE.)"
            ).strip(),
            kind="decision",
        )
        outgoing["R1"] = []
        outgoing["R2"] = []

        outgoing["BEGIN"].append(FlowEdge(src="BEGIN", dst="R1", label=None))
        outgoing["R1"].append(FlowEdge(src="R1", dst="R2", label=None))
        outgoing["R2"].append(FlowEdge(src="R2", dst="R2", label="CONTINUE"))
        outgoing["R2"].append(FlowEdge(src="R2", dst="END", label="STOP"))

        flow = Flow(nodes=nodes, outgoing=outgoing, begin_id="BEGIN", end_id="END")
        max_moves = total_runs
        return FlowRunner(flow, max_moves=max_moves)

    async def run(self, soul: KimiSoul, args: str) -> None:
        if args.strip():
            command = f"/{FLOW_COMMAND_PREFIX}{self._name}" if self._name else "/flow"
            logger.warning("Agent flow {command} ignores args: {args}", command=command, args=args)
            return
        if self._name:
            pass

        current_id = self._flow.begin_id
        moves = 0
        total_steps = 0
        while True:
            node = self._flow.nodes[current_id]
            edges = self._flow.outgoing.get(current_id, [])

            if node.kind == "end":
                logger.info("Agent flow reached END node {node_id}", node_id=current_id)
                return

            if node.kind == "begin":
                if not edges:
                    logger.error(
                        'Agent flow BEGIN node "{node_id}" has no outgoing edges; stopping.',
                        node_id=node.id,
                    )
                    return
                current_id = edges[0].dst
                continue

            if moves >= self._max_moves:
                raise MaxStepsReached(total_steps)
            next_id, steps_used = await self._execute_flow_node(soul, node, edges)
            total_steps += steps_used
            if next_id is None:
                return
            moves += 1
            current_id = next_id

    async def _execute_flow_node(
        self,
        soul: KimiSoul,
        node: FlowNode,
        edges: list[FlowEdge],
    ) -> tuple[str | None, int]:
        if not edges:
            logger.error(
                'Agent flow node "{node_id}" has no outgoing edges; stopping.',
                node_id=node.id,
            )
            return None, 0

        base_prompt = self._build_flow_prompt(node, edges)
        prompt = base_prompt
        steps_used = 0
        while True:
            result = await self._flow_turn(soul, prompt)
            steps_used += result.step_count
            if result.stop_reason == "tool_rejected":
                logger.error("Agent flow stopped after tool rejection.")
                return None, steps_used

            if node.kind != "decision":
                return edges[0].dst, steps_used

            choice = (
                parse_choice(result.final_message.extract_text(" "))
                if result.final_message
                else None
            )
            next_id = self._match_flow_edge(edges, choice)
            if next_id is not None:
                return next_id, steps_used

            options = ", ".join(edge.label or "" for edge in edges)
            logger.warning(
                "Agent flow invalid choice. Got: {choice}. Available: {options}.",
                choice=choice or "<missing>",
                options=options,
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Your last response did not include a valid choice. "
                "Reply with one of the choices using <choice>...</choice>."
            )

    @staticmethod
    def _build_flow_prompt(node: FlowNode, edges: list[FlowEdge]) -> str | list[ContentPart]:
        if node.kind != "decision":
            return node.label

        if not isinstance(node.label, str):
            label_text = Message(role="user", content=node.label).extract_text(" ")
        else:
            label_text = node.label
        choices = [edge.label for edge in edges if edge.label]
        lines = [
            label_text,
            "",
            "Available branches:",
            *(f"- {choice}" for choice in choices),
            "",
            "Reply with a choice using <choice>...</choice>.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _match_flow_edge(edges: list[FlowEdge], choice: str | None) -> str | None:
        if not choice:
            return None
        for edge in edges:
            if edge.label == choice:
                return edge.dst
        return None

    @staticmethod
    async def _flow_turn(
        soul: KimiSoul,
        prompt: str | list[ContentPart],
    ) -> TurnOutcome:
        wire_send(TurnBegin(user_input=prompt))
        res = await soul._turn(Message(role="user", content=prompt))  # type: ignore[reportPrivateUsage]
        wire_send(TurnEnd())
        return res
