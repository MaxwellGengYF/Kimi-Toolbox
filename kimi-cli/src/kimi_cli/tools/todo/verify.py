"""Shared todo-code verification logic (P2, B-4).

Single implementation used by both the soul-layer verification gate
(condition 2) and the CLI-layer closing reminder: iterate todos that carry
verification ``code`` and are not done, execute the code, auto-mark
successful todos ``done``, and build a reminder text for the failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kimi_cli.tools.todo import TodoList

_MAX_FAILURE_OUTPUT_CHARS = 500


def _attr(todo: Any, name: str) -> Any:
    """Read an attribute from either a mapping or an object."""
    if isinstance(todo, dict):
        return todo.get(name)
    return getattr(todo, name, None)


async def verify_code_todos(
    todo_tool: TodoList,
    todos: list[Any],
    *,
    strong: bool = False,
) -> str | None:
    """Execute verification code for every non-done todo that has code.

    For each todo with ``code`` whose status is not ``done``:

    - the todo is marked ``done`` via ``_verify_and_set_todo_status``, which runs
      its verification code exactly once (the same single-execution contract used
      by TodoList/TodoSub/TodoPop);
    - on failure the todo is reverted to ``pending`` and the error tail is
      collected into the reminder.
    Returns a reminder string listing the failures, or ``None`` when no
    code todo failed (or none exist). Never raises.
    """
    failures: list[str] = []
    for todo in todos:
        code = _attr(todo, "code")
        status = _attr(todo, "status")
        title = _attr(todo, "title") or ""
        if not code or status == "done":
            continue

        try:
            err_msg = await todo_tool._verify_and_set_todo_status(title, "done")  # pyright: ignore[reportPrivateUsage]
        except Exception as exc:
            err_msg = str(exc)
        if err_msg:
            failures.append(f"  - [{status}] {title}: {err_msg[:_MAX_FAILURE_OUTPUT_CHARS]}")

    if not failures:
        return None

    lines: list[str] = []
    if strong:
        lines.append(
            "CRITICAL: The following todo items have code that failed verification:\n"
        )
    else:
        lines.append(
            "The following todo items have code that failed verification:\n"
        )
    lines.extend(failures)
    lines.append(
        "\nFix the errors and mark the todos `done` via `TodoList` "
        "to re-trigger automatic code verification."
    )
    return "\n".join(lines)
