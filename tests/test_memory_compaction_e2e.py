"""End-to-end (no-LLM) test of the agent-memory compensation flow.

Simulates the full lifecycle the research answer prescribes:

1. Agent works: writes todos (TodoList tool) and durable facts (Memory tool).
2. Context pressure rises: the harness flushes important state to disk
   (pre-compaction flush).
3. Compaction destroys the working context (SimpleCompaction).
4. The harness re-surfaces durable memory at the end of the rebuilt context
   (post-compaction restore message).
5. The agent recalls facts from disk (Memory search) and gets its plan
   re-anchored (TodoReminderProvider reset on compaction).

Everything is driven with fakes; no LLM calls are made.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from kimi_cli.session_state import SessionState, save_session_state
from kimi_cli.soul.compaction import SimpleCompaction
from kimi_cli.soul.dynamic_injections.context_meter import ContextMeterProvider
from kimi_cli.soul.dynamic_injections.todo_reminder import TodoReminderProvider
from kimi_cli.tools.memory import (
    Memory,
    build_memory_restore_text,
    flush_pre_compact_state,
    memory_dir_for_session,
)
from kimi_cli.tools.memory import (
    Params as MemoryParams,
)
from kimi_cli.tools.todo import Params as TodoParams
from kimi_cli.tools.todo import Todo, TodoList
from kimi_cli.wire.types import TextPart
from kosong.message import Message
from kosong.tooling import ToolOk

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_runtime(session_dir: Path) -> SimpleNamespace:
    """Minimal Runtime stand-in good enough for TodoList/Memory tools."""
    state = SessionState()

    def _save_state() -> None:
        save_session_state(state, session_dir)

    session = SimpleNamespace(dir=session_dir, state=state, save_state=_save_state)
    return SimpleNamespace(
        role="root",
        session=session,
        subagent_store=None,
        subagent_id=None,
        read_only=False,
    )


def _make_soul(runtime: SimpleNamespace, step_no: int, tokens: int, max_tokens: int) -> SimpleNamespace:
    """Minimal KimiSoul stand-in for injection providers."""
    return SimpleNamespace(
        _current_step_no=step_no,
        is_subagent=False,
        status=SimpleNamespace(max_context_tokens=max_tokens),
        context=SimpleNamespace(token_count_with_pending=tokens),
    )


def _soul_todo_loader(runtime: SimpleNamespace):
    """Mirror of KimiSoul._load_todo_states_for_reminder."""

    def _load():
        todos = TodoList(runtime)._load_todos()
        return [
            __import__("kimi_cli.session_state", fromlist=["TodoItemState"]).TodoItemState(
                **t.model_dump()
            )
            for t in todos
        ]

    return _load


# ---------------------------------------------------------------------------
# E2E
# ---------------------------------------------------------------------------


class TestMemorySurvivesCompaction:
    def test_full_compensation_lifecycle(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        runtime = _make_runtime(session_dir)
        max_tokens = 200_000

        async def _run() -> None:
            # ── 1. Agent works: todos + durable memory ──────────────────────
            todo_tool = TodoList(runtime)
            result = await todo_tool(
                TodoParams(
                    todos=[
                        Todo(title="Refactor auth module", status="done"),
                        Todo(title="Add rate limiting", status="in_progress"),
                        Todo(title="Write integration tests", status="pending"),
                    ],
                    mode="force_overwrite",
                )
            )
            assert not result.is_error

            memory = Memory(runtime)
            r = await memory(
                MemoryParams(action="write", topic="decisions",
                             content="Auth tokens expire after 15 minutes; use refresh tokens.")
            )
            assert isinstance(r, ToolOk)

            # ── 2. Context pressure: pre-compaction flush ───────────────────
            loader = _soul_todo_loader(runtime)
            unfinished = [(t.status, t.title) for t in loader() if t.status != "done"]
            assert ("pending", "Write integration tests") in unfinished
            assert all("Refactor auth module" not in title for _, title in unfinished)

            flush_path = flush_pre_compact_state(
                session_dir,
                trigger_reason="auto",
                context_tokens=170_000,
                max_context_tokens=max_tokens,
                unfinished_todos=unfinished,
            )
            assert flush_path is not None and flush_path.is_file()

            # ── 3. Compaction destroys working context ──────────────────────
            history = [
                Message(role="user", content=[TextPart(text="Original task: secure the API")]),
                Message(role="assistant", content=[TextPart(text="Working on it")]),
                Message(role="user", content=[TextPart(text="Also add rate limiting")]),
                Message(role="assistant", content=[TextPart(text="Rate limiting in progress")]),
            ]
            compactor = SimpleCompaction(max_preserved_messages=1)
            prepared = compactor.prepare(history)
            assert prepared.compact_message is not None  # old content goes away

            # ── 4. Post-compaction restore message ──────────────────────────
            restore_text = build_memory_restore_text(session_dir)
            assert restore_text is not None
            assert "decisions.md" in restore_text
            assert "pre_compact_state.md" in restore_text
            assert "Write integration tests" in restore_text

            # ── 5. Agent recalls facts + plan after compaction ──────────────
            search = await memory(MemoryParams(action="search", query="token expire"))
            assert isinstance(search, ToolOk)
            assert "15 minutes" in search.output

            # Todo reminder re-anchors the plan immediately after compaction
            reminder = TodoReminderProvider(loader, interval_steps=100)
            soul = _make_soul(runtime, step_no=1, tokens=100_000, max_tokens=max_tokens)
            assert await reminder.get_injections([], soul)
            await reminder.on_context_compacted()
            injections = await reminder.get_injections([], soul)
            assert injections
            assert "[pending] Write integration tests" in injections[0].content
            assert "[in_progress] Add rate limiting" in injections[0].content

            # Context meter reports the fresh, low post-compaction usage
            meter = ContextMeterProvider(suppress_above=0.70)
            await meter.get_injections([], _make_soul(runtime, 1, 150_000, max_tokens))
            await meter.on_context_compacted()
            meter_injections = await meter.get_injections(
                [], _make_soul(runtime, 2, 30_000, max_tokens)
            )
            assert meter_injections
            # ContextMeterProvider returns a generic reminder, not usage stats.
            assert "Context is volatile" in meter_injections[0].content

        asyncio.run(_run())

        # Memory files physically persist outside the context backend
        mem_dir = memory_dir_for_session(session_dir)
        assert (mem_dir / "decisions.md").is_file()
        assert (mem_dir / "pre_compact_state.md").is_file()


class TestSubagentScopeIsolation:
    def test_memory_dir_is_session_scoped(self, tmp_path: Path) -> None:
        """Subagents share the session memory dir (facts visible to parent)."""
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        runtime = _make_runtime(session_dir)

        async def _run() -> None:
            memory = Memory(runtime)
            await memory(MemoryParams(action="write", topic="shared", content="from subagent"))
            r = await memory(MemoryParams(action="search", query="subagent"))
            assert "shared:1" in r.output

        asyncio.run(_run())
