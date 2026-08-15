"""Tests for Hermes-style todo-list re-injection into context compaction.

Covers :func:`kimi_cli.session_state.format_todo_injection` (the pure
renderer) and ``SimpleCompaction.compact(..., todos_loader=...)``
integration, plus the new ``LoopControl`` defaults.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from kosong.chat_provider.mock import MockChatProvider
from kosong.message import Message, TextPart

from kimi_cli.config import LoopControl
from kimi_cli.llm import LLM
from kimi_cli.session_state import (
    TODO_INJECTION_HEADER,
    TODO_INJECTION_MARKERS,
    TODO_INJECTION_TRUNCATION_MARKER,
    TodoItemState,
    TodoStatus,
    format_todo_injection,
)
from kimi_cli.soul.compaction import CompactionOptions, SimpleCompaction


def _todo(title: str, status: TodoStatus) -> TodoItemState:
    return TodoItemState(title=title, status=status)


# ---------------------------------------------------------------------------
# format_todo_injection — pure renderer
# ---------------------------------------------------------------------------


class TestFormatTodoInjection:
    def test_header_and_only_unfinished_items(self):
        todos = [
            _todo("Task pending", "pending"),
            _todo("Task in progress", "in_progress"),
            _todo("Task done", "done"),
        ]
        text = format_todo_injection(todos)
        assert text is not None
        assert text.startswith(TODO_INJECTION_HEADER)
        assert "- [ ] Task pending (pending)" in text
        assert "- [>] Task in progress (in_progress)" in text
        assert "Task done" not in text
        assert "(done)" not in text

    def test_markers_map_the_two_non_done_statuses(self):
        assert TODO_INJECTION_MARKERS == {"pending": "[ ]", "in_progress": "[>]"}

    def test_empty_list_returns_none(self):
        assert format_todo_injection([]) is None

    def test_all_done_list_returns_none(self):
        todos = [_todo("Done 1", "done"), _todo("Done 2", "done")]
        assert format_todo_injection(todos) is None

    def test_overflow_line_when_more_than_max_items(self):
        todos = [_todo(f"Task {i}", "pending") for i in range(25)]
        text = format_todo_injection(todos, max_items=20)
        assert text is not None
        assert "- … and 5 more (call todo_write to read all)" in text
        assert "- [ ] Task 0 (pending)" in text
        assert "- [ ] Task 19 (pending)" in text
        assert "Task 20" not in text

    def test_per_title_truncation_appends_marker(self):
        long_title = "t" * 300
        text = format_todo_injection([_todo(long_title, "pending")], per_title_chars=200)
        assert text is not None
        assert "t" * 200 + TODO_INJECTION_TRUNCATION_MARKER in text
        assert "t" * 201 not in text

    def test_total_char_cap_drops_tail_lines_with_marker(self):
        todos = [_todo(f"Task {i}", "pending") for i in range(50)]
        text = format_todo_injection(todos, max_chars=400)
        assert text is not None
        assert len(text) <= 400
        assert text.endswith(TODO_INJECTION_TRUNCATION_MARKER)
        assert text.startswith(TODO_INJECTION_HEADER)
        assert _all_lines_wellformed(text)

    def test_unknown_status_skipped(self):
        items = [
            _todo("Pending", "pending"),
            cast(TodoItemState, SimpleNamespace(status="completed", title="Completed")),
        ]
        text = format_todo_injection(items)
        assert text is not None
        assert "Completed" not in text
        assert "- [ ] Pending (pending)" in text

    def test_malformed_items_never_raise(self):
        malformed = [
            cast(TodoItemState, SimpleNamespace(status=None, title="No status")),
            cast(TodoItemState, SimpleNamespace(status="pending", title=None)),
            cast(TodoItemState, SimpleNamespace(status="done", title="Done task")),
            cast(TodoItemState, SimpleNamespace()),
        ]
        # All malformed / non-injectable -> None, no exception.
        assert format_todo_injection(malformed) is None
        # Mixed input: valid item survives, malformed items are skipped.
        mixed = [cast(TodoItemState, SimpleNamespace(status="pending", title="OK")), *malformed]
        text = format_todo_injection(mixed)
        assert text is not None
        assert "- [ ] OK (pending)" in text
        assert "No status" not in text
        assert "Done task" not in text


def _all_lines_wellformed(text: str) -> bool:
    """Every line (except the truncation marker) must be a complete item or
    overflow line — never a partially emitted line."""
    for line in text.splitlines()[1:]:
        if line == TODO_INJECTION_TRUNCATION_MARKER:
            continue
        if line.startswith("- … and "):
            continue
        if not (line.startswith("- [ ] ") or line.startswith("- [>] ")):
            return False
        if not line.endswith(")"):
            return False
    return True


# ---------------------------------------------------------------------------
# SimpleCompaction.compact + todos_loader integration
# ---------------------------------------------------------------------------


def _fake_llm() -> LLM:
    return LLM(
        chat_provider=MockChatProvider([TextPart(text="summary text")]),
        max_context_size=0,
        capabilities=set(),
    )


def _history() -> list[Message]:
    return [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]


class TestCompactTodoInjection:
    async def test_injects_todo_list_after_summary(self):
        history = _history()
        todos = [_todo("Ship the feature", "in_progress"), _todo("Write tests", "pending")]
        result = await SimpleCompaction(max_preserved_messages=1).compact(
            history,
            _fake_llm(),
            todos_loader=lambda: todos,
        )
        assert len(result.messages) == 3
        summary_text = result.messages[0].extract_text(" ")
        assert "summary text" in summary_text
        assert TODO_INJECTION_HEADER in summary_text
        assert "Ship the feature" in summary_text
        assert "[>]" in summary_text
        assert "Write tests" in summary_text
        # The injection must come right after the LLM summary parts.
        assert summary_text.index(TODO_INJECTION_HEADER) > summary_text.index("summary text")
        # Preserved turns are untouched (same objects, same order).
        assert result.messages[1] is history[0]
        assert result.messages[2] is history[3]

    async def test_loader_returning_empty_list_no_header(self):
        result = await SimpleCompaction(max_preserved_messages=1).compact(
            _history(),
            _fake_llm(),
            todos_loader=lambda: [],
        )
        assert TODO_INJECTION_HEADER not in result.messages[0].extract_text(" ")

    async def test_loader_raising_does_not_break_compaction(self):
        def boom() -> list[TodoItemState]:
            raise RuntimeError("loader broken")

        result = await SimpleCompaction(max_preserved_messages=1).compact(
            _history(),
            _fake_llm(),
            todos_loader=boom,
        )
        assert len(result.messages) == 3
        assert "summary text" in result.messages[0].extract_text(" ")
        assert TODO_INJECTION_HEADER not in result.messages[0].extract_text(" ")

    async def test_loader_none_matches_loader_empty(self):
        compactor = SimpleCompaction(max_preserved_messages=1)
        with_loader = await compactor.compact(
            _history(), _fake_llm(), todos_loader=lambda: []
        )
        without_loader = await compactor.compact(_history(), _fake_llm())
        assert with_loader.messages == without_loader.messages
        assert TODO_INJECTION_HEADER not in without_loader.messages[0].extract_text(" ")

    async def test_options_todos_max_items_flows_to_loader_output(self):
        todos = [_todo(f"Task {i}", "pending") for i in range(25)]
        result = await SimpleCompaction(max_preserved_messages=1).compact(
            _history(),
            _fake_llm(),
            options=CompactionOptions(todos_max_items=3),
            todos_loader=lambda: todos,
        )
        summary_text = result.messages[0].extract_text(" ")
        assert TODO_INJECTION_HEADER in summary_text
        assert "- … and 22 more (call todo_write to read all)" in summary_text


# ---------------------------------------------------------------------------
# LoopControl defaults
# ---------------------------------------------------------------------------


def test_loop_control_defaults():
    loop_control = LoopControl()
    assert loop_control.todo_compact_injection_enabled is True
    assert loop_control.todo_compact_injection_max_items == 20
