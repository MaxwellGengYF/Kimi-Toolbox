"""Phase 4 tests: context-overflow detection + recovery loop (§6.3).

Covers:
- ``is_context_overflow_error`` marker matching and 429/500/network rejection.
- The ``_step`` overflow branch: force-compact with AGGRESSIVE mode + override
  depth, retry the step, and never raise ``SessionRestartRequired`` when the
  retry budget allows.
- Budget exhaustion (``context_overflow_retries + 1`` step attempts → restart).
- ``context_overflow_retries = 0`` disables recovery.
- ``classify_api_error`` still classifies via the shared markers (regression).
- ``OverflowRecoveryState`` unit semantics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import pytest
from kosong.chat_provider import (
    APIConnectionError,
    APIStatusError,
    StreamedMessagePart,
    ThinkingEffort,
    TokenUsage,
)
from kosong.message import Message, TextPart
from kosong.tooling import Tool
from kosong.tooling.simple import SimpleToolset

from kimi_cli.llm import LLM
from kimi_cli.soul import SessionRestartRequired, run_soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.compaction import CompactMode, CompactionOptions
from kimi_cli.soul.context import Context
from kimi_cli.soul.context_overflow import (
    CONTEXT_OVERFLOW_MARKERS,
    OverflowRecoveryState,
    is_context_overflow_error,
)
from kimi_cli.soul.kimisoul import KimiSoul, classify_api_error
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire

# The agent system prompt used by ``_make_soul`` below. The main loop calls
# ``generate`` with this prompt; the compaction LLM call (legacy path, non-Kimi
# toolset) uses its own fixed prompt — the fake providers use this to tell the
# two apart so compaction calls can succeed while main-loop calls overflow.
MAIN_LOOP_SYSTEM_PROMPT = "Retry test prompt."

OVERFLOW_MESSAGE = (
    "This model's maximum context length is 100000 tokens. However, your "
    "messages resulted in 120000 tokens."
)


class StaticStreamedMessage:
    """Minimal kosong ``StreamedMessage`` for fake providers."""

    def __init__(self, parts: Sequence[StreamedMessagePart]) -> None:
        self._iter = self._to_stream(parts)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> StreamedMessagePart:
        return await self._iter.__anext__()

    async def _to_stream(
        self, parts: Sequence[StreamedMessagePart]
    ) -> object:
        for part in parts:
            yield part

    @property
    def id(self) -> str | None:
        return "overflow-test"

    @property
    def usage(self) -> TokenUsage | None:
        return None


class OverflowOnceThenSuccessProvider:
    """Raises a context-overflow 400 on the first *main-loop* step call, then
    succeeds. Compaction calls (different system prompt) always succeed."""

    name = "overflow-then-success"

    def __init__(self) -> None:
        self.generate_attempts = 0
        self.compaction_calls = 0

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> StaticStreamedMessage:
        if system_prompt == MAIN_LOOP_SYSTEM_PROMPT:
            self.generate_attempts += 1
            if self.generate_attempts == 1:
                raise APIStatusError(400, OVERFLOW_MESSAGE)
        else:
            self.compaction_calls += 1
        return StaticStreamedMessage([TextPart(text="recovered")])

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        return self


class AlwaysMainLoopOverflowProvider:
    """Every *main-loop* step call overflows; compaction calls succeed (so the
    recovery loop exhausts its budget through real step retries)."""

    name = "always-overflow"

    def __init__(self) -> None:
        self.generate_attempts = 0
        self.compaction_calls = 0

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> StaticStreamedMessage:
        if system_prompt == MAIN_LOOP_SYSTEM_PROMPT:
            self.generate_attempts += 1
            raise APIStatusError(400, OVERFLOW_MESSAGE)
        self.compaction_calls += 1
        return StaticStreamedMessage([TextPart(text="compacted")])

    def with_thinking(self, effort: ThinkingEffort) -> Self:
        return self


def _runtime_with_llm(runtime: Runtime, llm: LLM) -> Runtime:
    return Runtime(
        config=runtime.config,
        llm=llm,
        session=runtime.session,
        builtin_args=runtime.builtin_args,
        denwa_renji=runtime.denwa_renji,
        approval=runtime.approval,
        labor_market=runtime.labor_market,
        environment=runtime.environment,
        notifications=runtime.notifications,
        background_tasks=runtime.background_tasks,
        skills=runtime.skills,
        oauth=runtime.oauth,
        additional_dirs=runtime.additional_dirs,
        skills_dirs=runtime.skills_dirs,
        role=runtime.role,
    )


def _make_soul(runtime: Runtime, llm: LLM, tmp_path: Path) -> tuple[KimiSoul, Context]:
    agent = Agent(
        name="Retry Test Agent",
        system_prompt=MAIN_LOOP_SYSTEM_PROMPT,
        toolset=SimpleToolset(),
        runtime=_runtime_with_llm(runtime, llm),
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    return KimiSoul(agent, context=context), context


async def _drain_ui_messages(wire: Wire) -> None:
    wire_ui = wire.ui_side(merge=True)
    while True:
        try:
            await wire_ui.receive()
        except QueueShutDown:
            return


# ---------------------------------------------------------------------------
# is_context_overflow_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", CONTEXT_OVERFLOW_MARKERS)
def test_is_context_overflow_error_matches_each_marker(marker: str) -> None:
    err = APIStatusError(400, f"This model's {marker} was exceeded.")
    assert is_context_overflow_error(err)


def test_is_context_overflow_error_matches_mixed_case_message() -> None:
    err = APIStatusError(400, "Maximum Context Length exceeded")
    assert is_context_overflow_error(err)


def test_is_context_overflow_error_rejects_429() -> None:
    # 429 is classified as rate_limit before the 4xx branch; overflow markers
    # must not win there.
    err = APIStatusError(429, OVERFLOW_MESSAGE)
    assert not is_context_overflow_error(err)


def test_is_context_overflow_error_rejects_auth_statuses() -> None:
    # 401/403 are classified as auth before the 4xx branch.
    assert not is_context_overflow_error(APIStatusError(401, OVERFLOW_MESSAGE))
    assert not is_context_overflow_error(APIStatusError(403, OVERFLOW_MESSAGE))


def test_is_context_overflow_error_rejects_500() -> None:
    err = APIStatusError(500, OVERFLOW_MESSAGE)
    assert not is_context_overflow_error(err)


def test_is_context_overflow_error_rejects_network_error() -> None:
    err = APIConnectionError(OVERFLOW_MESSAGE)
    assert not is_context_overflow_error(err)


def test_is_context_overflow_error_rejects_non_api_status_error() -> None:
    assert not is_context_overflow_error(ValueError(OVERFLOW_MESSAGE))


def test_is_context_overflow_error_rejects_non_marker_400() -> None:
    err = APIStatusError(400, "invalid request body")
    assert not is_context_overflow_error(err)


# ---------------------------------------------------------------------------
# OverflowRecoveryState
# ---------------------------------------------------------------------------


def test_overflow_recovery_state_can_retry_consumed_reset() -> None:
    state = OverflowRecoveryState(max_retries=2)
    assert state.can_retry() is True
    state.consumed()
    assert state.can_retry() is True
    state.consumed()
    assert state.can_retry() is False
    # consuming past zero stays at zero
    state.consumed()
    assert state.can_retry() is False
    state.reset()
    assert state.can_retry() is True


@pytest.mark.parametrize("max_retries", [0, -1, -5])
def test_overflow_recovery_state_zero_or_negative_never_retries(max_retries: int) -> None:
    state = OverflowRecoveryState(max_retries=max_retries)
    assert state.can_retry() is False
    state.consumed()
    assert state.can_retry() is False


# ---------------------------------------------------------------------------
# classify_api_error regression (delegates to is_context_overflow_error)
# ---------------------------------------------------------------------------


def test_classify_api_error_returns_context_overflow_for_marker_messages() -> None:
    err = APIStatusError(400, OVERFLOW_MESSAGE)
    assert classify_api_error(err) == ("context_overflow", 400)


def test_classify_api_error_returns_4xx_client_for_non_marker_400() -> None:
    err = APIStatusError(400, "unrelated client error")
    assert classify_api_error(err) == ("4xx_client", 400)


def test_classify_api_error_rate_limit_still_wins_over_markers() -> None:
    # A 429 whose body mentions the context window is still a rate limit.
    err = APIStatusError(429, OVERFLOW_MESSAGE)
    assert classify_api_error(err) == ("rate_limit", 429)


# ---------------------------------------------------------------------------
# _step overflow recovery loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_recovery_force_compacts_and_retries(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overflow once → the soul force-compacts (AGGRESSIVE + override depth,
    trigger 'overflow'), retries the step, and the final output is produced."""
    runtime.config.loop_control.context_overflow_retries = 1
    runtime.config.loop_control.context_overflow_preserve_depth = 2
    provider = OverflowOnceThenSuccessProvider()
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    soul, context = _make_soul(runtime, llm, tmp_path)

    calls: list[dict] = []

    async def fake_compact_context(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(soul, "compact_context", fake_compact_context)

    await run_soul(soul, "trigger overflow recovery", _drain_ui_messages, asyncio.Event())

    assert provider.generate_attempts == 2
    assert len(calls) == 1
    call = calls[0]
    assert call["manual"] is False
    assert call["mode"] is CompactMode.AGGRESSIVE
    assert call["trigger_override"] == "overflow"
    assert isinstance(call["options"], CompactionOptions)
    assert call["options"].preserve_depth_override == 2
    # the retried step produced the final assistant message
    assert context.history[-1].extract_text(" ").strip() == "recovered"


@pytest.mark.asyncio
async def test_overflow_recovery_end_to_end_with_real_compaction(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Overflow once → the real compact_context runs (export, ledger, wire
    events) and the retried step succeeds without SessionRestartRequired."""
    runtime.config.loop_control.context_overflow_retries = 1
    provider = OverflowOnceThenSuccessProvider()
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    soul, context = _make_soul(runtime, llm, tmp_path)

    await run_soul(soul, "trigger overflow recovery", _drain_ui_messages, asyncio.Event())

    assert provider.generate_attempts == 2
    assert context.history[-1].extract_text(" ").strip() == "recovered"


@pytest.mark.asyncio
async def test_overflow_recovery_budget_exhausted_raises_session_restart(
    runtime: Runtime, tmp_path: Path
) -> None:
    """Provider always overflows → after ``context_overflow_retries + 1`` step
    attempts the recovery budget is empty and SessionRestartRequired is raised."""
    runtime.config.loop_control.context_overflow_retries = 1
    provider = AlwaysMainLoopOverflowProvider()
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    soul, _ = _make_soul(runtime, llm, tmp_path)

    with pytest.raises(SessionRestartRequired):
        await run_soul(soul, "trigger always overflow", _drain_ui_messages, asyncio.Event())

    # attempt 1 → overflow+compact; attempt 2 → overflow, budget exhausted.
    assert provider.generate_attempts == 2


@pytest.mark.asyncio
async def test_overflow_recovery_disabled_when_retries_zero(
    runtime: Runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """context_overflow_retries=0 → the overflow branch is skipped and the
    generic SessionRestartRequired path runs immediately (no compaction)."""
    runtime.config.loop_control.context_overflow_retries = 0
    provider = AlwaysMainLoopOverflowProvider()
    llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    soul, _ = _make_soul(runtime, llm, tmp_path)

    calls: list[dict] = []

    async def fake_compact_context(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(soul, "compact_context", fake_compact_context)

    with pytest.raises(SessionRestartRequired):
        await run_soul(soul, "trigger disabled overflow", _drain_ui_messages, asyncio.Event())

    assert provider.generate_attempts == 1
    assert calls == []
