# -*- coding: utf-8 -*-
"""SSE CLI debugger – connects to `kimix serve` and interactively tests SSE streams."""

from __future__ import annotations

import asyncio
import time
import pendulum
from pathlib import Path
from typing import Any, TextIO

from kimix.ui.printing import print_error, print  # noqa: F811 - use ui.printing.print for flush support
from kimix.server.client import KimixAsyncClient, MessagePart, parse_event, EventType, SSEEvent

def _fmt_arg(s: str, max_len: int = 120) -> str:
    """Truncate long arguments, keeping head and tail."""
    if len(s) <= max_len:
        return s
    head = max_len // 2
    tail = max_len - head - 3
    return s[:head] + "..." + s[-tail:]

def _fmt_ts(unix_t: float) -> str:
    """Format unix timestamp to HH:MM:SS."""
    if not unix_t:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(unix_t))


def _print_message_part(part: MessagePart) -> None:
    """Print a single MessagePart with formatting similar to print_agent_json."""
    from kimix.server.client import MessagePartType
    from kimix.ui.printing import colorful_text, Color

    if part.type == MessagePartType.TEXT:
        if part.text:
            print(part.text, end="", flush=True)

    elif part.type == MessagePartType.THINKING:
        if part.text:
            print(colorful_text(part.text, fg=Color.BRIGHT_CYAN), end="", flush=True)

    elif part.type == MessagePartType.TOOL_CALLING:
        tool_name = part.tool_name or "unknown"
        print(colorful_text(f"\n⚡ {tool_name}", fg=Color.BRIGHT_MAGENTA))
        status = part.tool_status or "unknown"
        extra: list[str] = []
        state = part.tool_state or {}
        if state.get("input"):
            extra.append(f"input: {_fmt_arg(str(state['input']))}")
        if state.get("output"):
            extra.append(f"output: {_fmt_arg(str(state['output']))}")
        if state.get("error"):
            extra.append(f"error: {_fmt_arg(str(state['error']))}")
        if status != "unknown":
            print(colorful_text(f"       status={status}", fg=Color.BRIGHT_BLACK))
        for line in extra:
            print(colorful_text(f"       {line}", fg=Color.BRIGHT_BLACK))

    elif part.type == MessagePartType.TOOL_CALLING_PART:
        if part.text:
            print(part.text, end="", flush=True)

    elif part.type == MessagePartType.TOOL_RESULT:
        result_text = part.tool_result or part.text or ""
        if result_text:
            prefix = "✓ "
            fg = Color.BRIGHT_GREEN
            state = part.tool_state or {}
            if state.get("error"):
                prefix = "✗ "
                fg = Color.BRIGHT_RED
            print(colorful_text(f"\n{prefix}{result_text}", fg=fg))

    elif part.type == MessagePartType.STEP_START:
        print(colorful_text("\n[STEP START]", fg=Color.BRIGHT_YELLOW))

    elif part.type == MessagePartType.STEP_FINISH:
        reason = part.reason or ""
        print(colorful_text(f"\n[STEP FINISH] reason={reason}", fg=Color.BRIGHT_YELLOW))


async def _sse_cli_main(host: str, port: int, debug: bool = False) -> None:
    client = KimixAsyncClient(host=host, port=port)
    log_file: TextIO | None = None
    if debug:
        log_name = f"sse_log_{pendulum.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = Path.cwd() / log_name
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"[SSE CLI] Debug mode ON, logging to {log_path}")
    print(f"[SSE CLI] Connecting to http://{host}:{port}")

    healthy = await client.health_check()
    if not healthy:
        print(f"[SSE CLI] Server not healthy at http://{host}:{port}")
        await client.close()
        return

    session = await client.create_session("SSE CLI debug session")
    print(f"[SSE CLI] Created session: {session.id}")
    print("[SSE CLI] Commands: /exit /new /abort /status /sessions /messages /clear /fix")

    tool_start_times: dict[str, float] = {}

    async def _cmd_help(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        print("[SSE CLI] Commands: /exit /new /abort /status /sessions /messages /clear /fix")
        return False

    async def _cmd_new(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        nonlocal session
        session = await client.create_session("SSE CLI debug session")
        print(f"[SSE CLI] New session: {session.id}")
        return False

    async def _cmd_abort(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        ok = await client.abort_session(session.id)
        print(f"[SSE CLI] Abort: {'ok' if ok else 'failed'}")
        return False

    async def _cmd_status(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        status = await client.get_session_status(session.id)
        print(f"[SSE CLI] Status: {status}")
        return False

    async def _cmd_sessions(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        sessions = await client.list_sessions()
        for s in sessions:
            print(f"  {s.id}: {s.title}")
        return False

    async def _cmd_messages(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        messages = await client.get_messages(session.id, limit=20)
        for m in messages:
            content = m.text_content[:100] if m.text_content else ""
            print(f"  [{m.role}] {content}...")
        return False

    async def _cmd_clear(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        ok = await client.clear_session(session.id)
        print(f"[SSE CLI] Clear: {'ok' if ok else 'failed'}")
        return False

    async def _cmd_compact(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        keep: int | None = None
        if len(task_split) > 1:
            try:
                keep = int(task_split[1])
            except ValueError:
                print("[SSE CLI] Usage: /compact[:N] (N = messages to keep)")
                return False
        ok = await client.compact_session(session.id, keep=keep)
        print(f"[SSE CLI] Compact: {'ok' if ok else 'failed'}")
        return False

    async def _cmd_export(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        output_path = task_split[1] if len(task_split) > 1 else None
        try:
            result = await client.export_session(session.id, output_path=output_path)
            print(f"[SSE CLI] Export: {result.get('count', 0)} messages -> {result.get('output', 'n/a')}")
        except Exception as exc:
            print(f"[SSE CLI] Export failed: {exc}")
        return False

    async def _cmd_unknown(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
        print(f"[SSE CLI] Unrecognized command: {task_split[0]}")
        return False

    _command_map = {
        "help": _cmd_help,
        "new": _cmd_new,
        "abort": _cmd_abort,
        "status": _cmd_status,
        "sessions": _cmd_sessions,
        "messages": _cmd_messages,
        "clear": _cmd_clear,
        "export": _cmd_export,
        "compact": _cmd_compact,
    }

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        cmd = text.strip()
        if not cmd:
            continue

        if cmd.startswith("/"): # command mode
            task = cmd[1:]
            split_idx = task.find(":")
            if split_idx >= 0:
                task_split = [task[:split_idx], task[split_idx + 1:]]
            else:
                task_split = [task]
            handler = _command_map.get(task_split[0], _cmd_unknown)
            should_break = await handler(task_split, [])
            if should_break:
                break
            continue

        ok = await client.send_prompt_async(session.id, text)
        if not ok:
            print("[SSE CLI] Failed to send prompt")
            continue

        print("[SSE CLI] Streaming events...")
        empty_polls = 0

        while True:
            await asyncio.sleep(0.5)

            try:
                messages = await client.get_messages(session.id, limit=50)
            except Exception as exc:
                print(f"[SSE CLI] get_messages error: {exc}")
                break

            new_count = 0
            for msg in messages:
                new_count += 1

                for part in msg.parts:
                    _print_message_part(part)

            if new_count == 0:
                empty_polls += 1
                if empty_polls >= 2:
                    try:
                        status = await client.get_session_status(session.id)
                        session_status = status.get("type", "idle")
                        if session_status in ("idle", "error"):
                            print(f"[SSE CLI] Session {session_status}, stream ended.")
                            messages = await client.get_messages(session.id)
                            for msg in messages:
                                for part in msg.parts:
                                    _print_message_part(part)
                                    
                            break
                    except Exception:
                        pass
                    empty_polls = 0
            else:
                empty_polls = 0

        #### The old legacy loop.
        # async for event in client.stream_events_robust(session.id):
        #     if debug:
        #         ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        #         debug_lines = [
        #             f"[{ts}] ===== SSE RAW EVENT =====",
        #             f"[{ts}] event: {event.event!r}",
        #             f"[{ts}] data: {event.data!r}",
        #             f"[{ts}] id: {event.id!r}",
        #             f"[{ts}] =========================",
        #         ]
        #         for line in debug_lines:
        #             print(line)
        #             if log_file:
        #                 log_file.write(line + "\n")
        #         if log_file:
        #             log_file.flush()
        #     parsed = parse_event(event, session.id)
        #     if parsed.type == EventType.SKIP:
        #         continue
        #     if parsed.type == EventType.TEXT_DELTA:
        #         print(parsed.delta, end="", flush=True)
        #     elif parsed.type == EventType.TEXT:
        #         print(parsed.delta, end="", flush=True)
        #     elif parsed.type == EventType.TOOL:
        #         extra: list[str] = []
        #         if parsed.tool_input:
        #             extra.append(f"input: {_fmt_arg(parsed.tool_input)}")
        #         if parsed.tool_output:
        #             extra.append(f"output: {_fmt_arg(parsed.tool_output)}")
        #         if parsed.tool_error:
        #             extra.append(f"error: {_fmt_arg(parsed.tool_error)}")
        #         if parsed.tool_call_id:
        #             extra.append(f"callId: {parsed.tool_call_id[:8]}")

        #         ts_info = ""
        #         if parsed.tool_status == "running" and parsed.tool_call_id:
        #             tool_start_times[parsed.tool_call_id] = parsed.created_at or time.time()
        #             ts_info = f"  start@{_fmt_ts(parsed.created_at or time.time())}"
        #         elif parsed.tool_status in ("completed", "error") and parsed.tool_call_id in tool_start_times:
        #             start_t = tool_start_times.pop(parsed.tool_call_id, 0)
        #             duration = (parsed.created_at or time.time()) - start_t
        #             ts_info = f"  took {duration:.1f}s  end@{_fmt_ts(parsed.created_at or time.time())}"
        #         elif parsed.created_at:
        #             ts_info = f"  {_fmt_ts(parsed.created_at)}"

        #         print(f"\n[TOOL] {parsed.tool_name} status={parsed.tool_status}{ts_info}")
        #         for line in extra:
        #             print(f"       {line}")
        #     elif parsed.type == EventType.REASONING:
        #         print(f"\n[REASONING] {parsed.text}")
        #     elif parsed.type == EventType.STEP_START:
        #         print("\n[STEP START]")
        #     elif parsed.type == EventType.STEP_FINISH:
        #         print(f"\n[STEP FINISH] reason={parsed.text}")
        #     elif parsed.type == EventType.SESSION_IDLE:
        #         print("\n[SESSION IDLE]")
        #     elif parsed.type == EventType.RECONNECTED:
        #         print(f"\n[RECONNECTED] {parsed.text}")
        #     elif parsed.type == EventType.UNKNOWN:
        #         print(f"\n[UNKNOWN] {parsed.raw}")
        #     if parsed.is_terminal():
        #         break
        print()  # newline after stream

    await client.close()
    if log_file:
        log_file.close()
        print(f"[SSE CLI] Debug log saved.")
    print("[SSE CLI] Bye.")


def run_sse_cli(host: str, port: int, debug: bool = False) -> None:
    try:
        asyncio.run(_sse_cli_main(host, port, debug))
    except KeyboardInterrupt:
        pass
