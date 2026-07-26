import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import orjson

import kimix.base as base
from kimi_agent_sdk import Session
from kosong.chat_provider import APIStatusError
from kimix.base import Color, MessageType, Style, print_agent_json, print_agent_json_flush_text
from kimix.tools.common import _export_to_temp_file
from kimix.utils.session import (
    _create_default_session,
    _create_default_session_async,
    _create_session_async,
    _print_usage,
    close_session_async,
)
from kimix.utils.system_prompt import SystemPromptType


def _read_subagent_state(path: Path) -> dict[str, Any]:
    """Read a subagent state file, returning an empty dict on missing/corrupt data."""
    import orjson

    if not path.exists():
        return {}
    try:
        data = orjson.loads(path.read_text(encoding="utf-8"))
    except (orjson.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


async def _maybe_build_todo_reminder(session: Session, *, strong: bool = False) -> str | None:
    cli = getattr(session, "_cli", None)
    if cli is None:
        return None

    toolset = getattr(getattr(getattr(cli, "soul", None), "agent", None), "toolset", None)
    if toolset is None:
        return None

    try:
        todo_tool = toolset.find("TodoList")
    except Exception:
        return None
    if todo_tool is None:
        return None

    runtime = getattr(cli, "_runtime", None)

    # Mirror the persistence split in ``kimi_cli.tools.todo.TodoList``:
    # root sessions store todos in ``SessionState`` and subagent sessions store
    # them in a separate ``state.json`` under the subagent instance directory.
    role = getattr(runtime, "role", "root") if runtime is not None else "root"
    if role == "root":
        state = getattr(getattr(cli, "session", None), "state", None)
        if state is None:
            return None
        todos = getattr(state, "todos", None) or []
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return None
        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        data = _read_subagent_state(state_file)
        raw_todos = data.get("todos", []) if isinstance(data.get("todos"), list) else []
        todos = []
        for t in raw_todos:
            sub_todos_raw = t.get("sub_todos", None) or []
            sub_todos_ns = [
                SimpleNamespace(title=st.get("title", ""), status=st.get("status", ""), notes=st.get("notes", None))
                for st in sub_todos_raw
            ]
            todos.append(
                SimpleNamespace(
                    title=t.get("title", ""),
                    status=t.get("status", ""),
                    notes=t.get("notes", None),
                    sub_todos=sub_todos_ns,
                )
            )

    if not todos:
        return None

    if all(getattr(todo, "status", None) == "done" for todo in todos):
        return None

    lines = []
    if strong:
        lines.append(
            "CRITICAL: Unfinished `TodoList` tasks remain. Mark every remaining item `done` with `TodoList` before ending this session. Do not declare completion or run final verification until the todo list is empty or all entries show `[done]`."
        )
    else:
        lines.append(
            "You have unfinished `TodoList` tasks. Update statuses below and complete all pending/in-progress items before finishing."
        )
    for todo in todos:
        title = getattr(todo, "title", "")
        status = getattr(todo, "status", "")
        # Skip todos already marked as done
        if status == "done":
            continue
        notes = getattr(todo, "notes", None)
        if notes:
            lines.append(f"- [{status}] {title}  Notes: {notes}")
        else:
            lines.append(f"- [{status}] {title}")
        # Render sub-todos if present (skip done sub-todos)
        sub_todos = getattr(todo, "sub_todos", None) or []
        for st in sub_todos:
            st_title = getattr(st, "title", "") if not isinstance(st, dict) else st.get("title", "")
            st_status = getattr(st, "status", "") if not isinstance(st, dict) else st.get("status", "")
            if st_status == "done":
                continue
            st_notes = getattr(st, "notes", None) if not isinstance(st, dict) else st.get("notes", None)
            if st_notes:
                lines.append(f"  - [{st_status}] {st_title}  Notes: {st_notes}")
            else:
                lines.append(f"  - [{st_status}] {st_title}")
    return "\n".join(lines)


async def _maybe_build_goal_reminder(session: Session, *, strong: bool = False) -> str | None:
    """Check if a goal is set and not yet done. Return a reminder prompt or None."""
    cli = getattr(session, "_cli", None)
    if cli is None:
        return None

    runtime = getattr(cli, "_runtime", None)
    role = getattr(runtime, "role", "root") if runtime is not None else "root"

    # Mirror the root/subagent persistence split from Goal tool (same as TodoList)
    if role == "root":
        custom_data = getattr(session, 'get_custom_data', lambda: None)()
        if custom_data is None:
            return None
        goal = custom_data.get("goal")
        if not isinstance(goal, dict):
            return None
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return None
        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        data = _read_subagent_state(state_file)
        goal = data.get("goal") if isinstance(data.get("goal"), dict) else None

    if not goal:
        return None
    if goal.get("status") == "done":
        return None

    goal_code = goal.get("code", "")
    if strong:
        return (
            "CRITICAL: The project `Goal` has NOT been successfully executed yet. "
            "The goal is defined as executable Python code. You MUST run it, verify it passes, "
            "and mark the goal as `done` before finishing the session.\n"
            f"Goal code:\n```python\n{goal_code}\n```"
        )
    else:
        return (
            "Reminder: The project `Goal` has not been successfully executed. "
            "Please run the goal code and mark it as done when it passes.\n"
            f"Goal code:\n```python\n{goal_code}\n```"
        )


async def _clear_session_goal(session: Session) -> None:
    """Clear both in-memory and persisted goal for the session."""
    cli = getattr(session, "_cli", None)
    if cli is None:
        return
    runtime = getattr(cli, "_runtime", None)
    if runtime is None:
        return

    if getattr(runtime, "role", "root") == "root":
        custom_data = getattr(session, 'get_custom_data', lambda: None)()
        if custom_data is not None:
            custom_data.pop("goal", None)
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is not None and subagent_id is not None:
            from kimi_cli.utils.io import atomic_json_write

            state_file = subagent_store.instance_dir(subagent_id) / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = _read_subagent_state(state_file)
            data.pop("goal", None)
            atomic_json_write(data, state_file)


async def _try_run_session_goal(session: Session) -> bool:
    """Try to run the current session goal code automatically.

    Loads the goal from the session (root or subagent), resolves the
    executable, runs it via ``Goal._run_goal_code``.  On success the goal
    is cleared and the function returns ``True``.  On failure the goal
    status is reset to ``pending`` and the function returns ``False``.

    Returns ``True`` (no-op) when there is no goal or it is already done.
    """
    from kimix.tools.goal import Goal

    cli = getattr(session, "_cli", None)
    if cli is None:
        return True
    runtime = getattr(cli, "_runtime", None)
    role = getattr(runtime, "role", "root") if runtime is not None else "root"

    # Load goal (same split as _maybe_build_goal_reminder)
    if role == "root":
        custom_data = getattr(session, "get_custom_data", lambda: None)()
        if custom_data is None:
            return True
        goal = custom_data.get("goal")
        if not isinstance(goal, dict):
            return True
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return True
        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        data = _read_subagent_state(state_file)
        goal = data.get("goal") if isinstance(data.get("goal"), dict) else None
        if goal is None:
            return True

    if goal.get("status") == "done":
        return True

    # Resolve the executable
    executable = Goal._resolve_goal_executable(goal)
    if executable is None:
        return False

    # Mark in_progress
    goal["status"] = "in_progress"
    await _save_session_goal(session, goal)

    try:
        success, output = await Goal._run_goal_code(goal)
    except Exception:
        goal["status"] = "pending"
        await _save_session_goal(session, goal)
        return False

    if success:
        # Clear the goal on success
        await _save_session_goal(session, None)
        return True
    else:
        goal["status"] = "pending"
        await _save_session_goal(session, goal)
        return False


async def _save_session_goal(session: Session, goal: dict[str, Any] | None) -> None:
    """Save (or clear) the goal for the given session.

    Mirrors the root/subagent split.
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return
    runtime = getattr(cli, "_runtime", None)
    if runtime is None:
        return

    if getattr(runtime, "role", "root") == "root":
        custom_data = getattr(session, "get_custom_data", lambda: None)()
        if custom_data is not None:
            if goal is None:
                custom_data.pop("goal", None)
            else:
                custom_data["goal"] = goal
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is not None and subagent_id is not None:
            from kimi_cli.utils.io import atomic_json_write

            state_file = subagent_store.instance_dir(subagent_id) / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = _read_subagent_state(state_file)
            if goal is None:
                data.pop("goal", None)
            else:
                data["goal"] = goal
            atomic_json_write(data, state_file)


async def _clear_session_todos(session: Session) -> None:
    """Clear both in-memory and persisted todo content for the session.

    Mirrors the persistence split in ``kimi_cli.tools.todo.TodoList``:
    root sessions store todos in ``SessionState`` and subagent sessions store
    them in a separate ``state.json`` under the subagent instance directory.
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return

    state = getattr(getattr(cli, "session", None), "state", None)
    if state is None:
        return

    if hasattr(state, "todos"):
        state.todos = []

    runtime = getattr(cli, "_runtime", None)
    if runtime is None:
        return

    if getattr(runtime, "role", "root") == "root":
        # Root session: persist the cleared list through SessionState.
        cli_session = cli.session
        if hasattr(cli_session, "save_state"):
            cli_session.save_state()
    else:
        # Subagent session: persist the cleared list to the subagent state file.
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is not None and subagent_id is not None:
            from kimi_cli.utils.io import atomic_json_write

            state_file = subagent_store.instance_dir(subagent_id) / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = _read_subagent_state(state_file)
            data["todos"] = []
            atomic_json_write(data, state_file)


async def _export_session_todos(session: Session, path: Path) -> None:
    """Export the session's current todo list to ``path`` as JSON.

    Mirrors the persistence split in ``kimi_cli.tools.todo.TodoList``:
    root sessions read todos from ``SessionState`` and subagent sessions read
    them from a separate ``state.json`` under the subagent instance directory.
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return

    toolset = getattr(getattr(getattr(cli, "soul", None), "agent", None), "toolset", None)
    if toolset is None:
        return

    try:
        todo_tool = toolset.find("TodoList")
    except Exception:
        return
    if todo_tool is None:
        return

    runtime = getattr(cli, "_runtime", None)
    role = getattr(runtime, "role", "root") if runtime is not None else "root"

    if role == "root":
        state = getattr(getattr(cli, "session", None), "state", None)
        if state is None:
            return
        todos = getattr(state, "todos", None) or []
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return
        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        data = _read_subagent_state(state_file)
        todos = data.get("todos", []) if isinstance(data.get("todos"), list) else []

    export_data: list[dict[str, Any]] = []
    for todo in todos:
        if isinstance(todo, dict):
            exp_entry: dict[str, Any] = {
                "title": todo.get("title", ""),
                "status": todo.get("status", ""),
            }
            sub_list_raw = todo.get("sub_todos", None)
            if sub_list_raw:
                exp_entry["sub_todos"] = [
                    {"title": st.get("title", ""), "status": st.get("status", "")}
                    for st in sub_list_raw
                ]
            export_data.append(exp_entry)
        else:
            exp_entry: dict[str, Any] = {
                "title": getattr(todo, "title", ""),
                "status": getattr(todo, "status", ""),
            }
            sub_list_raw = getattr(todo, "sub_todos", None)
            if sub_list_raw:
                exp_entry["sub_todos"] = [
                    {"title": getattr(st, "title", ""), "status": getattr(st, "status", "")}
                    for st in sub_list_raw
                ]
            export_data.append(exp_entry)
    if not export_data:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(orjson.dumps(export_data).decode("utf-8"), encoding="utf-8")
    except Exception as exc:
        base.print_error(f"Failed to export todo list to {path}: {exc}")


async def _run_single_prompt(
    session: Session,
    prompt_str: str,
    output_function: Callable[[str, MessageType], Any] | None,
    cancel_callable: Callable[[], bool] | None,
    merge_wire_messages: bool,
    info_print: bool,
    label: str = "Start...",
    format_output: bool = False,
) -> bool:
    """Send a single prompt to the session with retries and return True on success."""
    # Snapshot current goal state before running
    custom_data = getattr(session, 'get_custom_data', lambda: None)()
    if custom_data:
        goal = custom_data.get("goal")
        if isinstance(goal, dict) and goal.get("status") != "done":
            # Mark that we're about to attempt goal execution
            custom_data["_goal_attempt_pending"] = True

    if info_print:
        base._stream.colorful_print_word(f"{label}\n", fg=base.Color.BRIGHT_CYAN, require_new_line=True)

    max_retries = 3
    for attempt in range(max_retries):
        if session._cancel_event is not None and session._cancel_event.is_set():
            return False
        try:
            import time

            start_time = time.time()
            base._stream._last_char_was_newline = True
            async for message in session.prompt(prompt_str, merge_wire_messages=merge_wire_messages):
                if cancel_callable is not None and cancel_callable():
                    session.cancel()
                    break
                await print_agent_json(message, session, output_function, format_output=format_output)
            # After finishing, flush any remaining buffered text parts as formatted markdown.
            if format_output:
                print_agent_json_flush_text()
                if not base._stream._last_char_was_newline:
                    base._stream.print_word("\n", require_new_line=True)
            else:
                base._stream.print_word("\n", require_new_line=True)
            if info_print:
                end_time = time.time()
                _print_usage(session, end_time - start_time)
            return True
        except KeyboardInterrupt:
            if session:
                session.cancel()
            return False
        except Exception as e:
            base._stream.colorful_print_word(str(e), fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)
            if session:
                session.cancel()
            # HTTP API errors are already retried at the low-level chat-provider/soul
            # layer; do not add another retry loop here.
            if isinstance(e, APIStatusError):
                raise
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)
    return False


async def prompt_async(
    prompt_str: str,
    session: Session | None = None,
    # settings
    output_function: Callable[[str, MessageType], Any] | None = None,
    info_print: bool = True,
    cancel_callable: Callable[[], bool] | None = None,
    close_session_after_prompt: bool = False,
    merge_wire_messages: bool | None = None,
    ensure_todo_finished: bool = True,
    export_todo_list_path: Path | None = None,
    format_output: bool = False,
) -> None:
    from kimix.utils.prompt_str import escape_file_paths

    if export_todo_list_path is not None and export_todo_list_path.suffix.lower() != ".json":
        base.print_error(
            f"Invalid todo list export path: {export_todo_list_path}. "
            "Path must end with .json"
        )
        export_todo_list_path = None

    if session is None:
        session = await _create_default_session_async()
        close_session_after_prompt = False
    prompt_str = prompt_str.strip()
    prompt_str = escape_file_paths(prompt_str)
    if len(prompt_str) > 65536:  # too long, save to file
        name, new_id = _export_to_temp_file(content=prompt_str)
        prompt_str = f"read and execute: `{name}`"
    if merge_wire_messages is None:
        merge_wire_messages = output_function is not None
    try:
        prompt_success = await _run_single_prompt(
            session,
            prompt_str,
            output_function,
            cancel_callable,
            merge_wire_messages,
            info_print,
            label="Start...",
            format_output=format_output,
        )
        if prompt_success and ensure_todo_finished:
            max_todo_attempts = 2
            for attempt in range(max_todo_attempts):
                todo_reminder = await _maybe_build_todo_reminder(session, strong=(attempt > 0))
                if todo_reminder is None:
                    break
                if len(todo_reminder) > 65536:  # too long, save to file
                    name, new_id = _export_to_temp_file(content=todo_reminder)
                    todo_reminder = f"read and execute: `{name}`"
                label = "Todo review..." if attempt == 0 else "Final todo review..."
                try:
                    await _run_single_prompt(
                        session,
                        todo_reminder,
                        output_function,
                        cancel_callable,
                        merge_wire_messages,
                        info_print,
                        label=label,
                        format_output=True,
                    )
                except Exception as reminder_exc:
                    base._stream.colorful_print_word(
                        f"Todo reminder failed: {reminder_exc}",
                        fg=Color.BRIGHT_RED,
                        styles=[Style.BOLD],
                        require_new_line=True,
                    )
                    break

            # Goal enforcement loop
            max_goal_attempts = 2
            for attempt in range(max_goal_attempts):
                # Before reminding, try to run the goal code automatically
                auto_success = await _try_run_session_goal(session)
                if auto_success:
                    # Goal ran successfully (or no goal) — nothing to enforce
                    break

                goal_reminder = await _maybe_build_goal_reminder(session, strong=(attempt > 0))
                if goal_reminder is None:
                    break
                if len(goal_reminder) > 65536:  # too long, save to file
                    name, new_id = _export_to_temp_file(content=goal_reminder)
                    goal_reminder = f"read and execute: `{name}`"
                label = "Goal check..." if attempt == 0 else "Final goal check..."
                try:
                    await _run_single_prompt(
                        session,
                        goal_reminder,
                        output_function,
                        cancel_callable,
                        merge_wire_messages,
                        info_print,
                        label=label,
                        format_output=True,
                    )
                except Exception as reminder_exc:
                    base._stream.colorful_print_word(
                        f"Goal reminder failed: {reminder_exc}",
                        fg=Color.BRIGHT_RED,
                        styles=[Style.BOLD],
                        require_new_line=True,
                    )
                    break
        elif not prompt_success:
            base._stream.colorful_print_word("prompt failed.", fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)


    finally:
        if session:
            if export_todo_list_path is not None:
                await _export_session_todos(session, export_todo_list_path)
            await _clear_session_todos(session)
            await _clear_session_goal(session)
            if close_session_after_prompt:
                await close_session_async(session)
        base._stream.print_word("", True)


def prompt(
    prompt_str: str,
    session: Session | None = None,
    # settings
    output_function: Callable[[str, MessageType], Any] | None = None,
    info_print: bool = True,
    cancel_callable: Callable[[], bool] | None = None,
    close_session_after_prompt: bool = False,
    merge_wire_messages: bool | None = None,
    ensure_todo_finished: bool = True,
    export_todo_list_path: Path | None = None,
    format_output: bool = False,
) -> None:
    asyncio.run(
        prompt_async(
            prompt_str,
            session,
            # settings
            output_function,
            info_print,
            cancel_callable,
            close_session_after_prompt,
            merge_wire_messages=merge_wire_messages,
            ensure_todo_finished=ensure_todo_finished,
            export_todo_list_path=export_todo_list_path,
            format_output=format_output,
        ))


def prompt_path(path: Path, split_word: Optional[str] = None, session: Session | None = None, after_prompt_coro: Any = None) -> None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read()
    except Exception:
        base._stream.colorful_print_word(f'File {str(path)} not found.', fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)
    coro = None
    if after_prompt_coro is not None:
        coro = after_prompt_coro()
    if split_word:
        words = s.strip().split(split_word)
        for i in words:
            prompt(i, session=session)
            if coro is not None:
                try:
                    next(coro)
                except StopIteration:
                    coro = None
    else:
        prompt(s, session=session)
        if coro is not None:
            try:
                next(coro)
            except StopIteration:
                coro = None


async def prompt_plan_async(requirement: str, plan_file: str | Path = "plan.md") -> None:
    import kimix.tools.note as note

    plan_file = Path(plan_file)
    if plan_file.is_file():
        plan_file.unlink()

    note._enable_plan = True
    planner_session: Session | None = None

    try:
        planner_provider = base.get_default_sub_provider("planner")
        planner_session = await _create_session_async(
            agent_type=SystemPromptType.TodoMaker,
            agent_file='agent_planner.json',
            provider_dict=planner_provider,
        )
        planner_session.get_custom_data()["plan_writing_path"] = plan_file

        reminder = (
            "read the following requirement carefully and generate a comprehensive plan. "
            "save the complete plan to a file using the WritePlan tool. "
            f"Requirement:\n{requirement.strip()}"
        )

        max_plan_attempts = 3
        plan_generated = False
        for attempt in range(max_plan_attempts):
            if planner_session._cancel_event is not None and planner_session._cancel_event.is_set():
                break
            try:
                base._stream.colorful_print_word(
                    f"Generating plan (attempt {attempt + 1}/{max_plan_attempts})...\n",
                    fg=Color.BRIGHT_CYAN,
                    require_new_line=True,
                )
                async for message in planner_session.prompt(reminder):
                    await print_agent_json(message, planner_session, None, True)
                base._stream.print_word("\n", require_new_line=True)

                if plan_file.exists() and plan_file.stat().st_size > 0:
                    plan_generated = True
                    break

                if attempt < max_plan_attempts - 1:
                    reminder = (
                        "The plan file was not generated. "
                        "Please generate the plan and save it using the WritePlan tool.\n\n"
                        f"Requirement:\n{requirement.strip()}"
                    )
            except KeyboardInterrupt:
                if planner_session:
                    planner_session.cancel()
                break
            except Exception as exc:
                base._stream.colorful_print_word(
                    str(exc), fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True
                )
                if planner_session:
                    planner_session.cancel()
                if attempt == max_plan_attempts - 1:
                    raise
                await asyncio.sleep(1)
        if not plan_generated:
            base._stream.colorful_print_word(
                "Plan generation failed: plan file not found.",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            return

        def _open_plan_file(filepath: Path) -> None:
            """Open the plan file with the system default application."""
            try:
                if sys.platform == "win32":
                    os.startfile(str(filepath))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(filepath)])
                else:
                    subprocess.run(["xdg-open", str(filepath)])
            except Exception:
                pass

        base._stream.colorful_print_word(
            f"Plan generated: {plan_file.absolute()}\n",
            fg=Color.BRIGHT_GREEN,
            styles=[Style.BOLD],
            require_new_line=True,
        )
        _open_plan_file(plan_file)

        # Review loop: let the user approve or request revisions
        execute_plan = True
        while True:
            user_input = await asyncio.to_thread(
                input, "Do you want to implement the plan? (y/n): "
            )
            if user_input.strip().lower() == "y":
                break

            # User wants to revise the plan — get feedback and loop back to planner
            feedback = await asyncio.to_thread(
                input, "Please describe the changes you want (/quit to give up): "
            )
            feedback = feedback.strip()
            if not feedback:
                continue
            if feedback.lower() == '/quit':
                execute_plan = False
                break
            revision_reminder = (
                "The user reviewed the plan and wants the following changes:\n\n"
                f"{feedback.strip()}\n\n"
                "Please update the plan file accordingly using the WritePlan or EditPlan tools. "
            )
            try:
                base._stream.colorful_print_word(
                    "Revising plan...\n",
                    fg=Color.BRIGHT_CYAN,
                    require_new_line=True,
                )
                async for message in planner_session.prompt(revision_reminder):
                    await print_agent_json(message, planner_session, None, True)
                base._stream.print_word("\n", require_new_line=True)

                # Re-open the updated plan file for review
                if plan_file.exists():
                    _open_plan_file(plan_file)
            except KeyboardInterrupt:
                if planner_session:
                    planner_session.cancel()
                return
            except Exception as exc:
                base._stream.colorful_print_word(
                    f"Revision failed: {exc}",
                    fg=Color.BRIGHT_RED,
                    styles=[Style.BOLD],
                    require_new_line=True,
                )
                # Continue the loop so the user can try again

        note._enable_plan = False
        if not execute_plan:
            return

        if not plan_file.exists():
            base._stream.colorful_print_word(
                f"Plan file {plan_file} no longer exists. Aborting.",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            return

        plan_content = plan_file.read_text(encoding="utf-8", errors="replace")
        plan_size = len(plan_content.encode("utf-8"))
        regular_session = await _create_default_session_async()
        if plan_size > 100 * 1024:
            impl_prompt = f"Read this plan `{plan_file}`, carefully research, read all related files first, call TodoList to record, then implement the plan step-by-step."
            review_reminder = f"Review the plan in `{plan_file}` and ensure all tasks are completed."
        else:
            impl_prompt = f"Read this plan:\n\n{plan_content}\n\ncarefully research, read all related files first, call TodoList to record, then implement the plan step-by-step:"
            review_reminder = f"Review this plan and ensure all tasks are completed:\n\n{plan_content}"
        await prompt_async(
            impl_prompt,
            session=regular_session,
            ensure_todo_finished=False,
            format_output=True
        )

        await prompt_async(
            review_reminder,
            session=regular_session,
            ensure_todo_finished=False,
            format_output=True
        )
    except Exception as exc:
        base._stream.colorful_print_word(
            f"prompt_plan failed: {exc}",
            fg=Color.BRIGHT_RED,
            styles=[Style.BOLD],
            require_new_line=True,
        )
    finally:
        note._enable_plan = False
        if planner_session:
            await close_session_async(planner_session)


def prompt_plan(requirement: str, plan_file: str | Path = "plan.md") -> None:
    asyncio.run(prompt_plan_async(requirement, plan_file))
