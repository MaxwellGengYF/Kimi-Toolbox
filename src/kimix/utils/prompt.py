import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import orjson
import kimix.base as base
from kaos.path import KaosPath
from kimi_agent_sdk import Session
from kosong.chat_provider import APIStatusError
from kimi_cli.llm import LoopDetectedError, TextLoopDetector
from kosong.message import ContentPart, TextPart, ThinkPart
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


_MAX_TEXT_BLOCK_RESUME_ROUNDS = 2
"""Maximum number of follow-up prompts used to force a session to end on a
plain text block.

When the agent's turn ends with a trailing tool call or reasoning-only block,
the task may be unfinished: the agent loop "quits" without a final answer. In
that case the session is resumed with a continuation prompt (up to this many
rounds) that asks it to finish the work and end with a plain text block.
"""


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


def _get_session_todos(session: Session) -> list[Any]:
    """Return the unified todo list for a session (root or subagent).

    Mirrors the persistence split in ``kimi_cli.tools.todo.TodoList``: root
    sessions store todos in ``SessionState`` and subagent sessions store them
    in a separate ``state.json`` under the subagent instance directory.
    Subagent items are returned as ``SimpleNamespace`` objects so callers can
    use ``getattr`` uniformly; root items keep their native shape (dicts or
    objects), which the existing render/export paths already handle.
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return []
    runtime = getattr(cli, "_runtime", None)
    role = getattr(runtime, "role", "root") if runtime is not None else "root"
    if role == "root":
        state = getattr(getattr(cli, "session", None), "state", None)
        if state is None:
            return []
        return getattr(state, "todos", None) or []
    subagent_store = getattr(runtime, "subagent_store", None)
    subagent_id = getattr(runtime, "subagent_id", None)
    if subagent_store is None or subagent_id is None:
        return []
    state_file = subagent_store.instance_dir(subagent_id) / "state.json"
    data = _read_subagent_state(state_file)
    raw_todos = data.get("todos", []) if isinstance(data.get("todos"), list) else []
    todos: list[Any] = []
    for t in raw_todos:
        children_raw = t.get("children", None) or []
        children_ns = [
            SimpleNamespace(title=st.get("title", ""), status=st.get("status", ""), notes=st.get("notes", None))
            for st in children_raw
        ]
        todos.append(
            SimpleNamespace(
                title=t.get("title", ""),
                status=t.get("status", ""),
                notes=t.get("notes", None),
                children=children_ns,
            )
        )
    return todos


def _set_session_todos(session: Session, todos: list[Any]) -> None:
    """Persist a todo list for the session (root or subagent).

    Mirrors the persistence split in ``kimi_cli.tools.todo.TodoList``: root
    sessions store todos in ``SessionState`` and subagent sessions store them
    in a separate ``state.json`` under the subagent instance directory. The
    in-memory ``SessionState.todos`` is updated for every role; persistence is
    skipped when the runtime is unavailable (same as the original logic).
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return
    state = getattr(getattr(cli, "session", None), "state", None)
    if state is not None and hasattr(state, "todos"):
        state.todos = todos
    runtime = getattr(cli, "_runtime", None)
    if runtime is None:
        return
    if getattr(runtime, "role", "root") == "root":
        # Root session: persist through SessionState.save_state.
        cli_session = getattr(cli, "session", None)
        if cli_session is not None and hasattr(cli_session, "save_state"):
            cli_session.save_state()
    else:
        # Subagent session: persist to the subagent state file.
        subagent_store = getattr(runtime, "subagent_store", None)
        subagent_id = getattr(runtime, "subagent_id", None)
        if subagent_store is None or subagent_id is None:
            return
        from kimi_cli.utils.io import atomic_json_write

        state_file = subagent_store.instance_dir(subagent_id) / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        data = _read_subagent_state(state_file)
        data["todos"] = todos
        atomic_json_write(data, state_file)


async def steer_session(session: Session, content: str | list[ContentPart]) -> bool:
    """Push a follow-up message into a running agent session.

    Resolves the session's ``KimiSoul`` (via ``session._cli.soul``) and injects
    *content* as a user message, interrupting the currently streaming step when
    the session is mid-turn (including while reasoning/text parts are still
    printing). The injection happens at the message/context layer, above the
    providers, so it works with every backend provider.

    Returns ``True`` when the content was delivered to a running soul; ``False``
    when no soul can be resolved from the session or the soul is idle.
    """
    from kimi_cli.soul.steer import Steer

    steer = Steer.from_session(session)
    if steer is None:
        return False
    return await steer.push(content)


def steer_session_sync(session: Session, content: str | list[ContentPart]) -> bool:
    """Synchronous wrapper around :func:`steer_session` for non-async callers.

    Only call this from a thread other than the one running the session's event
    loop — blocking the loop's own thread would deadlock.
    """
    from kimi_cli.soul.steer import Steer

    steer = Steer.from_session(session)
    if steer is None:
        return False
    return steer.push_sync(content)


async def _maybe_build_todo_reminder(session: Session, *, strong: bool = False) -> str | None:
    cli = getattr(session, "_cli", None)
    if cli is None:
        return None

    toolset = getattr(getattr(getattr(cli, "soul", None), "agent", None), "toolset", None)
    if toolset is None:
        return None

    try:
        todo_tool = toolset.find("todo_write")
    except Exception:
        return None
    if todo_tool is None:
        return None
    runtime = getattr(cli, "_runtime", None)
    todos = _get_session_todos(session)

    if not todos:
        return None

    if all(getattr(todo, "status", None) == "done" for todo in todos):
        return None

    lines = []
    # Inject the original user request as context so the agent remembers
    # what it was working on when it reviews unfinished todos.
    current_prompt = getattr(runtime, "current_prompt", None) if runtime is not None else None
    if current_prompt:
        # Truncate long prompts to avoid flooding the reminder with raw content.
        if len(current_prompt) > 200:
            current_prompt = current_prompt[:100] + "..." + current_prompt[-100:]
        lines.append(f"Original request: {current_prompt}")
        lines.append("")
    if strong:
        lines.append(
            "CRITICAL: Unfinished `todo_write` tasks remain. Mark every remaining item `completed` with `todo_update` (or `todo_write` for bulk edits) before ending this session. Do not declare completion or run final verification until the todo list is empty or all entries show `[completed]`."
        )
    else:
        lines.append(
            "You have unfinished `todo_write` tasks. Update statuses below with `todo_update` and mark every pending/in-progress item `completed` before finishing."
        )

    def _render_todo(item: Any, indent: int = 0) -> None:
        title = getattr(item, "title", "") if not isinstance(item, dict) else item.get("title", "")
        status = getattr(item, "status", "") if not isinstance(item, dict) else item.get("status", "")
        # Skip todos already marked as done
        if status == "done":
            return
        notes = getattr(item, "notes", None) if not isinstance(item, dict) else item.get("notes", None)
        prefix = "  " * indent
        if notes:
            lines.append(f"{prefix}- [{status}] {title}  Notes: {notes}")
        else:
            lines.append(f"{prefix}- [{status}] {title}")
        # Render children recursively (skip done children)
        children = getattr(item, "children", None) if not isinstance(item, dict) else item.get("children", None)
        for child in children or []:
            _render_todo(child, indent + 1)

    for todo in todos:
        _render_todo(todo)
    return "\n".join(lines)


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
    """Clear both in-memory and persisted todo content for the session."""
    cli = getattr(session, "_cli", None)
    if cli is None:
        return

    state = getattr(getattr(cli, "session", None), "state", None)
    if state is None:
        return

    _set_session_todos(session, [])


async def _export_session_todos(session: Session, path: Path) -> None:
    """Export the session's current todo list to ``path`` as JSON."""
    cli = getattr(session, "_cli", None)
    if cli is None:
        return

    toolset = getattr(getattr(getattr(cli, "soul", None), "agent", None), "toolset", None)
    if toolset is None:
        return

    try:
        todo_tool = toolset.find("todo_write")
    except Exception:
        return
    if todo_tool is None:
        return

    todos = _get_session_todos(session)

    export_data: list[dict[str, Any]] = []
    for todo in todos:
        if isinstance(todo, dict):
            exp_entry: dict[str, Any] = {
                "title": todo.get("title", ""),
                "status": todo.get("status", ""),
            }
            children_raw = todo.get("children", None)
            if children_raw:
                exp_entry["children"] = [
                    {"title": st.get("title", ""), "status": st.get("status", "")}
                    for st in children_raw
                ]
            export_data.append(exp_entry)
        else:
            exp_entry: dict[str, Any] = {
                "title": getattr(todo, "title", ""),
                "status": getattr(todo, "status", ""),
            }
            children_raw = getattr(todo, "children", None)
            if children_raw:
                exp_entry["children"] = [
                    {"title": getattr(st, "title", ""), "status": getattr(st, "status", "")}
                    for st in children_raw
                ]
            export_data.append(exp_entry)
    if not export_data:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(orjson.dumps(export_data).decode("utf-8"), encoding="utf-8")
    except Exception as exc:
        base.print_error(f"Failed to export todo list to {path}: {exc}")


def _provider_key(provider_dict: dict[str, Any]) -> tuple:
    """Build a hashable identity key for a provider_dict.

    Used to detect whether a backup provider is already the active one,
    preventing redundant provider switches.
    """
    return (
        provider_dict.get("type"),
        provider_dict.get("model"),
        provider_dict.get("url") or provider_dict.get("base_url"),
    )


def _get_active_provider_dict(session: Session) -> dict[str, Any] | None:
    """Return the provider_dict currently active on the session.

    Reads from session custom_data; falls back to base._default_provider.
    """
    data = session.get_custom_data() if hasattr(session, "get_custom_data") else None
    if data is not None:
        active = data.get("_active_provider_dict")
        if active is not None:
            return active
    return base._default_provider


def _set_active_provider_dict(session: Session, provider_dict: dict[str, Any]) -> None:
    """Record the active provider_dict on the session for failover tracking."""
    data = session.get_custom_data() if hasattr(session, "get_custom_data") else None
    if data is not None:
        data["_active_provider_dict"] = provider_dict


async def _switch_session_provider(
    session: Session,
    provider_dict: dict[str, Any],
) -> bool:
    """Swap the session's runtime LLM to a backup provider.

    Builds a new LLM from *provider_dict*, closes the previous chat provider's
    HTTP client (prevents resource leaks), and assigns the new LLM onto
    ``session._cli.soul.runtime.llm``. Also updates ``custom_config`` so
    sub-agents spawned later inherit the backup provider.

    Returns True if the swap succeeded, False if the LLM could not be built.
    """
    from kimix.utils.config import _create_config
    from kimi_cli.llm import create_llm

    cli = getattr(session, "_cli", None)
    if cli is None:
        return False
    soul = getattr(cli, "soul", None)
    if soul is None:
        return False
    runtime = getattr(soul, "runtime", None) or getattr(soul, "_runtime", None)
    if runtime is None:
        return False

    # Build Config + LLM from backup provider_dict
    cfg, _ = _create_config(provider_dict)
    if cfg.model is None or cfg.provider is None:
        return False

    new_llm = create_llm(
        cfg.provider,
        cfg.model,
        session_id=getattr(getattr(cli, "session", None), "id", None),
        thinking=base._default_thinking,
        oauth=getattr(runtime, "oauth", None),
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        thinking_effort=cfg.thinking_effort,
    )
    if new_llm is None:
        return False

    # Close the old chat provider's HTTP client
    old_llm = getattr(runtime, "llm", None)
    if old_llm is not None:
        old_provider = getattr(old_llm, "chat_provider", None)
        aclose = getattr(old_provider, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

    # Swap the LLM in place
    runtime.llm = new_llm

    # Update custom_config so sub-agent spawning picks up the new provider
    custom_config = session.get_custom_config() if hasattr(session, "get_custom_config") else None
    if custom_config is not None:
        custom_config["provider_dict"] = provider_dict

    # Track active provider for subsequent prompts
    _set_active_provider_dict(session, provider_dict)
    return True


async def _run_prompt_attempts(
    session: Session,
    prompt_str: str,
    output_function: Callable[[str, MessageType], Any] | None,
    cancel_callable: Callable[[], bool] | None,
    merge_wire_messages: bool,
    info_print: bool,
    format_output: bool,
    deadline: float | None,
    label: str = "Start...",
) -> None:
    """Run a single prompt with up to max_retries attempts on the CURRENT provider.

    Raises on failure (TimeoutError, APIStatusError, or after exhausting retries).
    Returns normally on success (after printing usage info).
    """
    max_retries = 3

    async def _run_prompt_iter():
        detector = TextLoopDetector.from_env()
        try:
            async for message in session.prompt(prompt_str, merge_wire_messages=merge_wire_messages):
                if cancel_callable is not None and cancel_callable():
                    session.cancel()
                    break
                if detector is not None:
                    loop_text = None
                    if isinstance(message, TextPart):
                        loop_text = message.text
                    elif isinstance(message, ThinkPart) and not message.encrypted:
                        loop_text = message.think
                    if loop_text is not None and detector.feed(loop_text):
                        base._stream.colorful_print_word(
                            "Loop detected; cancelling session.",
                            fg=Color.BRIGHT_RED,
                            styles=[Style.BOLD],
                            require_new_line=True,
                        )
                        session.cancel()
                        break
                await print_agent_json(message, session, output_function, format_output=format_output)
        except LoopDetectedError:
            base._stream.colorful_print_word(
                "Loop detected; cancelling session.",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            session.cancel()

    for attempt in range(max_retries):
        if session._cancel_event is not None and session._cancel_event.is_set():
            return  # caller treats normal return as success; cancel handled by caller
        try:
            start_time = time.time()
            base._stream._last_char_was_newline = True

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Prompt timed out")
                await asyncio.wait_for(_run_prompt_iter(), timeout=remaining)
            else:
                await _run_prompt_iter()

            # flush output
            if format_output:
                print_agent_json_flush_text()
                if not base._stream._last_char_was_newline:
                    base._stream.print_word("\n", require_new_line=True)
            else:
                base._stream.print_word("\n", require_new_line=True)
            if info_print:
                end_time = time.time()
                _print_usage(session, end_time - start_time)
            return  # success
        except KeyboardInterrupt:
            if session:
                session.cancel()
            raise
        except (asyncio.TimeoutError, TimeoutError) as te:
            base._stream.colorful_print_word(
                f"Prompt timed out: {te}",
                fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True,
            )
            if session:
                session.cancel()
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


def _trailing_content_kind(session: Session) -> str | None:
    """Return the content kind of the session's last user-visible block.

    Inspects the most recent meaningful block of the conversation history:
    the last assistant message that carries content parts. Returns ``"text"``,
    ``"think"`` or ``"tool"``; ``None`` when there is no inspectable history
    (e.g. fake sessions used in tests).
    """
    try:
        cli = getattr(session, "_cli", None)
        if cli is None:
            return None
        soul = getattr(cli, "soul", None)
        if soul is None:
            return None
        context = getattr(soul, "context", None)
        history = getattr(context, "history", None)
        if history is None:
            return None
        for message in reversed(history):
            if message.role != "assistant":
                continue
            parts = message.content or []
            if not parts and not message.tool_calls:
                continue
            # A text part always wins: the turn ended with a real answer.
            if any(isinstance(p, TextPart) for p in parts):
                return "text"
            if message.tool_calls:
                return "tool"
            if any(isinstance(p, ThinkPart) for p in parts):
                return "think"
            return None
    except Exception:
        return None
    return None


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
    """Send a single prompt with retries and backup-provider failover.

    Tries the current (active) provider first with up to ``max_retries``
    attempts. If all attempts fail, iterates through ``sub_providers`` with
    ``role == "backup"`` (in declaration order), switching the session's
    LLM to each backup provider and retrying. Once a backup succeeds, the
    session **stays** on that provider for subsequent prompts.

    Returns True on success, False on KeyboardInterrupt/cancel.
    Raises on unrecoverable failure (all providers exhausted).
    """
    if info_print:
        base._stream.colorful_print_word(f"{label}\n", fg=base.Color.BRIGHT_CYAN, require_new_line=True)

    # Compute the overall deadline (shared across primary + backups)
    deadline: float | None = None
    if timeout is not None:
        deadline = time.monotonic() + timeout

    # ── Phase A: Try the current (active) provider ──
    primary_exc: Exception | None = None
    try:
        await _run_prompt_attempts(
            session, prompt_str, output_function, cancel_callable,
            merge_wire_messages, info_print, format_output, deadline, label,
        )
        return True  # success on active provider
    except KeyboardInterrupt:
        return False
    except (asyncio.TimeoutError, TimeoutError) as te:
        # Timeout: if a deadline was set and we still have time, backups
        # may help; if the overall deadline passed, propagate.
        if deadline is not None and (deadline - time.monotonic()) <= 0:
            raise
        primary_exc = te
        # Fall through to backup providers
    except Exception as exc:
        # Active provider failed — fall through to backup providers
        primary_exc = exc

    # ── Phase B: Try backup providers ──
    backups = base.get_default_sub_providers_by_role("backup")
    if not backups:
        # No backups configured — propagate the primary failure.
        if primary_exc is not None:
            raise primary_exc
        return False

    active_key = _provider_key(_get_active_provider_dict(session) or {})
    last_error: Exception | None = None

    for backup in backups:
        # Skip the currently active provider (already tried)
        if _provider_key(backup) == active_key:
            continue
        if session._cancel_event is not None and session._cancel_event.is_set():
            return False

        # Switch the session's LLM to this backup provider
        switched = await _switch_session_provider(session, backup)
        if not switched:
            continue

        # Print a notice about the switch
        model_name = backup.get("model", "unknown")
        base._stream.colorful_print_word(
            f"Switching to backup provider: {model_name}\n",
            fg=Color.BRIGHT_YELLOW, styles=[Style.BOLD], require_new_line=True,
        )

        # Update the active key so we don't re-try this backup
        active_key = _provider_key(backup)

        try:
            await _run_prompt_attempts(
                session, prompt_str, output_function, cancel_callable,
                merge_wire_messages, info_print, format_output, deadline,
                label=f"Backup ({model_name})...",
            )
            return True  # backup succeeded — session stays on this provider
        except KeyboardInterrupt:
            return False
        except (asyncio.TimeoutError, TimeoutError):
            # This backup timed out — try the next backup if deadline allows
            if deadline is not None and (deadline - time.monotonic()) <= 0:
                break  # no time left
            continue
        except Exception as exc:
            last_error = exc
            continue  # try next backup

    # ── All providers exhausted ──
    if last_error is not None:
        raise last_error
    if primary_exc is not None:
        raise primary_exc
    return False


async def _resume_for_text_block(
    session: Session,
    output_function: Callable[[str, MessageType], Any] | None,
    cancel_callable: Callable[[], bool] | None,
    merge_wire_messages: bool,
    info_print: bool,
    format_output: bool,
    timeout: float | None,
) -> None:
    """Force the session to end on a plain text block.

    When the preceding agent turn finished with a trailing tool call or a
    reasoning-only block (no final text answer), the task may be unfinished:
    the loop "quit" without delivering a result. This resumes the exact same
    session with a short continuation prompt (up to
    ``_MAX_TEXT_BLOCK_RESUME_ROUNDS`` times) asking it to finish the remaining
    work and end with a plain text block.
    """
    cli = getattr(session, "_cli", None)
    if cli is None:
        return
    runtime = getattr(cli, "_runtime", None)
    work_dir = getattr(session, "work_dir", None)
    if work_dir is None:
        cli_session = getattr(cli, "session", None)
        work_dir = getattr(cli_session, "work_dir", None) if cli_session is not None else None
    work_dir_str = str(work_dir) if work_dir is not None else None
    current_prompt = getattr(runtime, "current_prompt", None) if runtime is not None else None

    for attempt in range(_MAX_TEXT_BLOCK_RESUME_ROUNDS):
        kind = _trailing_content_kind(session)
        if kind in (None, "text"):
            return

        lines = [
            "The previous response did not end with a plain text block "
            "(it ended with reasoning or a tool call, which means the work may be unfinished).",
            "",
            "Rules for this continuation:",
            "1. Do NOT call any tools unless truly necessary to finish the task.",
            "2. Finish any remaining work from the original request.",
            "3. End your response with a plain text block — a final text message that "
            "summarizes what was done and states the result. No trailing tool calls and "
            "no trailing reasoning.",
        ]
        if work_dir_str:
            lines.append(f"4. Working directory: {work_dir_str}")
        if current_prompt:
            truncated = current_prompt if len(current_prompt) <= 400 else current_prompt[:200] + "..." + current_prompt[-200:]
            lines.append("")
            lines.append(f"Original request: {truncated}")
        resume_prompt = "\n".join(lines)

        label = "Resume to finish (text block)..." if attempt == 0 else "Final text-block check..."
        try:
            await _run_single_prompt(
                session,
                resume_prompt,
                output_function,
                cancel_callable,
                merge_wire_messages,
                info_print,
                label=label,
                format_output=format_output,
                timeout=timeout,
            )
        except Exception as exc:
            base._stream.colorful_print_word(
                f"Text-block resume failed: {exc}",
                fg=Color.BRIGHT_RED,
                styles=[Style.BOLD],
                require_new_line=True,
            )
            return


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
        name, new_id = _export_to_temp_file(key=None, content=prompt_str)
        prompt_str = f"read and execute: `{name}`"
    if merge_wire_messages is None:
        merge_wire_messages = output_function is not None

    # Store the (possibly transformed) prompt_str on the runtime so that
    # todo_write and _maybe_build_todo_reminder can inject it into their
    # ALL_DONE_REMINDER / reminder messages.
    cli = getattr(session, "_cli", None)
    if cli is not None:
        runtime = getattr(cli, "_runtime", None)
        if runtime is not None:
            runtime.current_prompt = prompt_str

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
                    name, new_id = _export_to_temp_file(key=None, content=todo_reminder)
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
        elif not prompt_success:
            base._stream.colorful_print_word("prompt failed.", fg=Color.BRIGHT_RED, styles=[Style.BOLD], require_new_line=True)

        if prompt_success:
            # ── Text-block gate: never end the session on tool/reasoning ──────
            # The agent loop can quit even when the task is unfinished, leaving
            # a trailing tool call or reasoning-only block with no final text
            # answer. Resume the same session until the last assistant block is
            # a plain text message.
            await _resume_for_text_block(
                session,
                output_function,
                cancel_callable,
                merge_wire_messages,
                info_print,
                format_output,
                timeout,
            )


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
        import copy
        planner_provider = base.get_default_sub_provider("planner")
        if planner_provider is None or not planner_provider.get("type"):
            # No usable planner sub-provider configured (e.g. the config has
            # no `sub_providers` key): fall back to the main provider settings
            # so the planner session can still be created.
            planner_provider = base._default_provider
        # Deep-copy so we don't mutate the shared default provider dict
        planner_provider = copy.deepcopy(planner_provider) if planner_provider else {}
        planner_provider.setdefault("loop_control", {})
        planner_provider["loop_control"]["budget_reminder_enabled"] = False
        planner_provider["loop_control"]["context_meter_enabled"] = False
        planner_provider["loop_control"]["compact_reminder_enabled"] = False
        planner_provider["loop_control"]["todo_reminder_enabled"] = False
        planner_provider["loop_control"]["target_churn_enabled"] = False
        # Inherit the caller's working directory so relative plan paths and
        # AGENTS.md/skills resolution match the session that spawned the planner
        # (a planner session that defaults to the process CWD cannot find the
        # repo files the caller references).
        planner_work_dir = None
        try:
            from kimix.utils import _globals as _session_globals

            default = _session_globals._default_session
            if default is not None:
                raw = getattr(default, "work_dir", None)
                if raw is None:
                    cli = getattr(default, "_cli", None)
                    cli_session = getattr(cli, "session", None) if cli is not None else None
                    raw = getattr(cli_session, "work_dir", None)
                if raw is not None:
                    planner_work_dir = raw if isinstance(raw, KaosPath) else KaosPath(str(raw))
        except Exception:
            planner_work_dir = None
        planner_session = await _create_session_async(
            agent_type=SystemPromptType.TodoMaker,
            agent_file='agent_planner.json',
            provider_dict=planner_provider,
            work_dir=planner_work_dir,
        )
        planner_session.get_custom_data()["plan_writing_path"] = plan_file

        # Lock the planner session to read-only so it cannot write to the filesystem
        # or modify external state — its sole job is to generate a plan via WritePlan.
        if hasattr(planner_session, '_cli') and planner_session._cli is not None:
            _runtime = getattr(planner_session._cli, '_runtime', None)
            if _runtime is not None:
                _runtime.read_only = True

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
            impl_prompt = f"Read this plan `{plan_file}`, carefully research, read all related files first, call todo_write to record, then implement the plan step-by-step."
            review_reminder = f"Review the plan in `{plan_file}` and ensure all tasks are completed."
        else:
            impl_prompt = f"Read this plan:\n\n{plan_content}\n\ncarefully research, read all related files first, call todo_write to record, then implement the plan step-by-step:"
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
