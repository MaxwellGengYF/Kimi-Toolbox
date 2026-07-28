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
from kimix.ui.printing import Color, MessageType, Style
from kimix.ui.stream import print_agent_json, print_agent_json_flush_text
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


async def _maybe_build_code_todo_reminder(session: Session, *, strong: bool = False) -> str | None:
    """Check todos with code: run pending/in_progress code for verification.

    Delegates to the shared ``kimi_cli.tools.todo.verify.verify_code_todos``
    (B-4): code todos that pass are auto-marked done; failures are collected
    into a reminder string (or None if all code passed or no code todos).
    """
    from kimi_cli.tools.todo.verify import verify_code_todos

    cli = getattr(session, "_cli", None)
    if cli is None:
        return None

    runtime = getattr(cli, "_runtime", None)
    role = getattr(runtime, "role", "root") if runtime is not None else "root"

    # Get the TodoList tool instance from the runtime's toolset
    toolset = getattr(getattr(getattr(cli, "soul", None), "agent", None), "toolset", None)
    if toolset is None:
        return None
    todo_tool = toolset.find("TodoList")
    if todo_tool is None:
        return None

    if role == "root":
        state = getattr(getattr(cli, "session", None), "state", None)
        if state is None:
            return None
        todos_raw = getattr(state, "todos", None) or []
    else:
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return None
        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        data = _read_subagent_state(state_file)
        todos_raw = data.get("todos", []) if isinstance(data.get("todos"), list) else []

    try:
        return await verify_code_todos(todo_tool, list(todos_raw), strong=strong)
    except Exception:
        return None


def _get_cli_closing_reminder_rounds(session: Session) -> int:
    """Read ``cli_closing_reminder_rounds`` from the session's loop control.

    Defaults to 1 (soul gate is primary; CLI loop is the fallback).
    """
    cli = getattr(session, "_cli", None)
    soul = getattr(cli, "soul", None) if cli is not None else None
    loop_control = getattr(soul, "_loop_control", None)
    rounds = getattr(loop_control, "cli_closing_reminder_rounds", None)
    if not isinstance(rounds, int) or rounds < 0:
        return 1
    return rounds


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
    timeout: float | None = None,
) -> bool:
    """Send a single prompt to the session with retries and return True on success.

    Args:
        timeout: Maximum seconds for the entire prompt (including retries).
                 None means no timeout. When reached, raises TimeoutError.
    """
    if info_print:
        base._stream.colorful_print_word(f"{label}\n", fg=base.Color.BRIGHT_CYAN, require_new_line=True)

    max_retries = 3
    deadline = None
    if timeout is not None:
        deadline = time.monotonic() + timeout
    for attempt in range(max_retries):
        if session._cancel_event is not None and session._cancel_event.is_set():
            return False
        try:
            import time

            start_time = time.time()
            base._stream._last_char_was_newline = True

            # Wrap the prompt iteration with an optional timeout guard
            async def _run_prompt_iter():
                async for message in session.prompt(prompt_str, merge_wire_messages=merge_wire_messages):
                    if cancel_callable is not None and cancel_callable():
                        session.cancel()
                        break
                    await print_agent_json(message, session, output_function, format_output=format_output)

            if timeout is not None:
                # Compute remaining time for this attempt
                remaining = deadline - time.monotonic() if deadline else timeout
                if remaining <= 0:
                    raise TimeoutError(f"Prompt timed out after {timeout}s")
                await asyncio.wait_for(_run_prompt_iter(), timeout=remaining)
            else:
                await _run_prompt_iter()

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
        except (asyncio.TimeoutError, TimeoutError) as te:
            base._stream.colorful_print_word(
                f"Prompt timed out: {te}",
                fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True,
            )
            if session:
                session.cancel()
            # Timeout is not retried — propagate immediately
            raise
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
    timeout: float | None = None,
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
        if ensure_todo_finished:
            # When ensure_todo_finished=True, catch ALL exceptions from the
            # main prompt so the caller never sees an unhandled exception.
            # Errors are logged, and the todo-reminder loop still runs to
            # salvage any unfinished items before graceful return.
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
                    timeout=timeout,
                )
            except Exception as exc:
                base._stream.colorful_print_word(
                    f"Prompt error (gracefully handled): {exc}",
                    fg=Color.BRIGHT_RED,
                    styles=[Style.BOLD],
                    require_new_line=True,
                )
                prompt_success = False
        else:
            prompt_success = await _run_single_prompt(
                session,
                prompt_str,
                output_function,
                cancel_callable,
                merge_wire_messages,
                info_print,
                label="Start...",
                format_output=format_output,
                timeout=timeout,
            )
        if prompt_success and ensure_todo_finished:
            closing_rounds = _get_cli_closing_reminder_rounds(session)
            max_todo_attempts = closing_rounds
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

            # Code-todo reminder loop (replaces old goal enforcement loop)
            # Auto-verification happens inside _verify_and_set_todo_status
            # when a todo is marked `done`. This loop only provides reminders.
            max_code_attempts = closing_rounds
            for attempt in range(max_code_attempts):
                code_reminder = await _maybe_build_code_todo_reminder(session, strong=(attempt > 0))
                if code_reminder is None:
                    break
                if len(code_reminder) > 65536:
                    name, new_id = _export_to_temp_file(content=code_reminder)
                    code_reminder = f"read and execute: `{name}`"
                label = "Code check..." if attempt == 0 else "Final code check..."
                try:
                    await _run_single_prompt(
                        session,
                        code_reminder,
                        output_function,
                        cancel_callable,
                        merge_wire_messages,
                        info_print,
                        label=label,
                        format_output=True,
                    )
                except Exception as reminder_exc:
                    base._stream.colorful_print_word(
                        f"Code todo reminder failed: {reminder_exc}",
                        fg=Color.BRIGHT_RED,
                        styles=[Style.BOLD],
                        require_new_line=True,
                    )
                    break
        elif not prompt_success:
            base._stream.colorful_print_word("prompt failed.", fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)


    finally:
        try:
            if session:
                if export_todo_list_path is not None:
                    try:
                        await _export_session_todos(session, export_todo_list_path)
                    except Exception as exc:
                        base.print_error(f"Failed to export todos: {exc}")
                try:
                    await _clear_session_todos(session)
                except Exception as exc:
                    base.print_error(f"Failed to clear todos: {exc}")
                if close_session_after_prompt:
                    try:
                        await close_session_async(session)
                    except Exception as exc:
                        base.print_error(f"Failed to close session: {exc}")
        except Exception:
            # Absolute last resort: never let cleanup exceptions escape
            pass
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
    timeout: float | None = None,
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
            timeout=timeout,
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
