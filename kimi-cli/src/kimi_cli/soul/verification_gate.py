"""Soul-layer verification gate (P2, B-3).

When the agent tries to end a turn (``stop_reason == "no_tool_calls"``),
the gate checks whether the turn is *actually* finished:

1. unfinished todos remain (root or subagent scope);
2. todos with verification ``code`` exist whose verification fails
   (the verification is executed here — successful ones are auto-marked
   done, which may by itself clear the gate);
3. the turn modified files (edit-class tool calls) but ran no
   verification-class call at all.

If any condition hits, the gate returns a reminder text and the turn
continues. A hard ``max_nudges`` cap per turn prevents deadlocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson
from kosong.message import Message

from kimi_cli.soul.message import is_system_reminder_message
from kimi_cli.soul.tool_taxonomy import EDIT_TOOLS, VERIFICATION_TOOL_HINTS

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_MAX_UNFINISHED_LISTED = 10


class VerificationGate:
    """Decides whether a turn may end, and produces nudge text otherwise."""

    def __init__(self, *, max_nudges: int = 2) -> None:
        self._max_nudges = max(0, max_nudges)
        self._nudges = 0
        self._gate_turn_id = ""

    # ------------------------------------------------------------------
    # Turn-scope helpers
    # ------------------------------------------------------------------

    def _sync_turn(self, soul: KimiSoul) -> None:
        turn_id = soul._current_turn_id  # pyright: ignore[reportPrivateUsage]
        if turn_id != self._gate_turn_id:
            self._gate_turn_id = turn_id
            self._nudges = 0

    @staticmethod
    def _current_turn_history(soul: KimiSoul) -> list[Message]:
        """History suffix belonging to the current turn.

        Scans backwards from the end and stops before the most recent
        *real* user message (the one that started the turn). Injected
        system-reminder user messages are skipped.
        """
        history = list(soul.context.history)
        start = 0
        for idx in range(len(history) - 1, -1, -1):
            msg = history[idx]
            if msg.role == "user" and not is_system_reminder_message(msg):
                start = idx + 1
                break
        return history[start:]

    @staticmethod
    def _classify_turn_tool_calls(turn_history: list[Message]) -> tuple[bool, bool]:
        """Return ``(has_edits, has_verification)`` for the turn's tool calls."""
        has_edits = False
        has_verification = False
        for msg in turn_history:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                if name in EDIT_TOOLS:
                    has_edits = True
                if name in VERIFICATION_TOOL_HINTS:
                    # A TodoList call only counts as verification when it
                    # actually marks something done (which runs its code).
                    if name == "TodoList":
                        if VerificationGate._todolist_marks_done(tool_call.function.arguments):
                            has_verification = True
                    else:
                        has_verification = True
        return has_edits, has_verification

    @staticmethod
    def _todolist_marks_done(arguments: str | None) -> bool:
        if not arguments:
            return False
        try:
            args = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            return False
        if not isinstance(args, dict):
            return False
        todos = args.get("todos") or args.get("items")
        if isinstance(todos, dict):
            todos = [todos]
        if not isinstance(todos, list):
            return False
        return any(
            isinstance(t, dict) and t.get("status") == "done" for t in todos
        )

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    async def check(self, soul: KimiSoul) -> str | None:
        """Return a nudge text if the turn must continue, else ``None``."""
        self._sync_turn(soul)
        if self._nudges >= self._max_nudges:
            return None

        reasons: list[str] = []

        # Condition 1: unfinished todos (root or subagent scope).
        try:
            todos = soul._load_todo_states_for_reminder()  # pyright: ignore[reportPrivateUsage]
        except Exception:
            todos = []
        unfinished = [t for t in todos if t.status != "done"]
        if unfinished:
            lines = ["Unfinished TodoList tasks remain:"]
            for item in unfinished[:_MAX_UNFINISHED_LISTED]:
                lines.append(f"- [{item.status}] {item.title}")
            if len(unfinished) > _MAX_UNFINISHED_LISTED:
                lines.append(f"- … and {len(unfinished) - _MAX_UNFINISHED_LISTED} more")
            reasons.append("\n".join(lines))

        # Condition 2: todos with verification code that fails.
        code_todos = [t for t in unfinished if getattr(t, "code", None)]
        if code_todos:
            todo_tool = None
            try:
                todo_tool = soul.agent.toolset.find("TodoList")
            except Exception:
                todo_tool = None
            if todo_tool is not None:
                from kimi_cli.tools.todo.verify import verify_code_todos

                try:
                    code_reminder = await verify_code_todos(todo_tool, code_todos)
                except Exception:
                    code_reminder = None
                if code_reminder:
                    reasons.append(code_reminder)

        # Condition 3: edits this turn but no verification-class call.
        turn_history = self._current_turn_history(soul)
        has_edits, has_verification = self._classify_turn_tool_calls(turn_history)
        if has_edits and not has_verification:
            reasons.append(
                "You modified code this turn but ran no verification "
                "(no tests/check commands, no todo verification). "
                "Run the project's tests or a verification command before finishing."
            )

        if not reasons:
            return None

        self._nudges += 1
        return (
            "The turn cannot finish yet — verification gate findings:\n\n"
            + "\n\n".join(reasons)
            + "\n\nAddress the findings above, then finish. "
            f"(nudge {self._nudges}/{self._max_nudges} this turn)"
        )
