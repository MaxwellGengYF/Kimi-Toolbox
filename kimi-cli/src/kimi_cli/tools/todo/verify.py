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

    - the code is executed via :meth:`TodoList._run_code`;
    - on success the todo is marked ``done`` (no re-run, since the status
      transition already happened inside ``_verify_and_set_todo_status``);
    - on failure the error tail is collected into the reminder.

    Returns a reminder string listing the failures, or ``None`` when no
    code todo failed (or none exist). Never raises.
    """
    from kimi_cli.tools.todo import TodoList as _TodoList

    failures: list[str] = []
    for todo in todos:
        code = _attr(todo, "code")
        status = _attr(todo, "status")
        title = _attr(todo, "title") or ""
        if not code or status == "done":
            continue

        try:
            executable = _TodoList._resolve_code_executable(code)  # pyright: ignore[reportPrivateUsage]
            if executable is None:
                failures.append(f"  - [{status}] {title}: code not runnable")
                continue
            try:
                success, output = await _TodoList._run_code(code, executable=executable)  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:
                success, output = False, str(exc)
            finally:
                _TodoList._cleanup_code_tempfile(executable)  # pyright: ignore[reportPrivateUsage]
        except Exception as exc:  # never break the caller on verification bugs
            success, output = False, str(exc)

        if success:
            try:
                await todo_tool._verify_and_set_todo_status(title, "done")  # pyright: ignore[reportPrivateUsage]
            except Exception:
                pass
        else:
            failures.append(f"  - [{status}] {title}: verification failed — {output[:_MAX_FAILURE_OUTPUT_CHARS]}")

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
