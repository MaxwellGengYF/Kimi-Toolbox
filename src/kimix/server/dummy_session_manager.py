# -*- coding: utf-8 -*-
"""Dummy session manager: stubs all SessionManager interfaces.

Prints each web request and its formatted arguments for debugging purposes.
Matches the interface consumed by `src/kimix/server/app.py`.

SessionStateClass holds a real SDK Session and runs prompts via a background
work thread that calls prompt_async from kimix.utils.prompt.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kimi_agent_sdk import Session
from kimix.base import _default_agent_file_dir
from kimix.ui.printing import MessageType
from kimix.ui.stream import format_tool_args
from kimix.server.bus import bus, BusEvent
from kimix.utils.session import _create_session_async
from kimix.utils.system_prompt import SystemPromptType


# ── Plan Command Marker ──────────────────────────────────────────


@dataclass
class PlanCommand:
    """Marker wrapper for plan prompts in the input queue."""
    text: str


# ── Minimal data models (replicated for standalone use) ──────────


@dataclass
class SessionStateClass:
    """Holds a real SDK Session plus queues and a background work thread.

    The work thread runs an asyncio event loop that:
      1. Pops prompt text from ``input_queue``.
      2. Calls ``prompt_async`` (from ``kimix.utils.prompt``) with an
         ``output_function`` that pushes captured chunks into ``output_queue``.
    """

    session: Optional[Session] = None
    planner_session: Optional[Session] = None
    input_queue: queue.Queue = field(default_factory=queue.Queue)
    output_queue: queue.Queue = field(default_factory=queue.Queue)
    work_thread: Optional[threading.Thread] = None
    running: bool = True
    is_working: bool = False
    _loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)
    part_id_map: Dict[str, str] = field(default_factory=dict)
    session_id: str = ""
    _part_counter: int = 0
    _plan_confirm_event: Optional[asyncio.Event] = field(default=None, repr=False)
    _plan_feedback: Optional[str] = field(default=None, repr=False)

    def stop(self) -> None:
        """Signal the work loop to stop and join the thread."""
        self.running = False
        self.input_queue.put(None)  # sentinel
        if self.work_thread is not None and self.work_thread.is_alive():
            self.work_thread.join(timeout=5.0)

    async def close_session(self) -> None:
        """Close the SDK session if active."""
        if self.session is not None:
            try:
                from kimix.utils import close_session_async
                await close_session_async(self.session)
            except Exception:
                pass
            self.session = None


@dataclass
class SessionInfo:
    id: str = ""
    title: Optional[str] = None
    createdAt: float = 0.0
    updatedAt: float = 0.0
    parentID: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
            "parentID": self.parentID,
        }


@dataclass
class SessionStatus:
    type: str = "idle"
    token_count: int = 0
    context_usage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "token_count": self.token_count,
            "context_usage": self.context_usage,
        }


# ── ID Helpers ──────────────────────────────────────────────────

_counter = 0


def _gen_id(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}_dummy_{_counter:04x}"


def _now_ts() -> float:
    return time.time()


# ── Work loop (runs in background thread) ────────────────────────


def _work_loop(state: SessionStateClass) -> None:
    """Background thread target: run an asyncio event loop that processes
    prompts and plan commands from ``input_queue`` and feeds output into ``output_queue``."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state._loop = loop

    try:
        while state.running:
            try:
                item = state.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # sentinel to stop
                break

            state.is_working = True
            try:
                if isinstance(item, PlanCommand):
                    loop.run_until_complete(
                        _process_plan(state, item.text)
                    )
                else:
                    # item is the prompt text
                    text = item
                    loop.run_until_complete(
                        _process_prompt(state, text)
                    )
            except Exception:
                pass
            finally:
                # Mark idle when prompt done and no more input queued
                if state.input_queue.qsize() == 0:
                    state.is_working = False
    finally:
        # Close the SDK session from within its own loop
        if state.session is not None:
            try:
                loop.run_until_complete(state.close_session())
            except Exception:
                pass
        loop.close()


async def _process_prompt(
    state: SessionStateClass, text: str
) -> None:
    """Call prompt_async with an output_function that enqueues messages."""
    from kimix.utils.prompt import prompt_async

    # Emit running status
    bus.emit(BusEvent(type="session.status", properties={
        "status": {"type": "running", "sessionID": state.session_id},
    }))

    part_text_buf: Dict[str, str] = {}

    def _get_or_create_part_id(msg_type: MessageType) -> str:
        key = msg_type.name if hasattr(msg_type, "name") else str(msg_type)
        if key in state.part_id_map:
            return state.part_id_map[key]
        state._part_counter += 1
        pid = f"prt_dummy_{state._part_counter:04x}"
        state.part_id_map[key] = pid
        return pid

    def _get_cumulative_text(msg_type: MessageType) -> str:
        key = msg_type.name if hasattr(msg_type, "name") else str(msg_type)
        return part_text_buf.setdefault(key, "")

    def _output_handler(chunk: str, msg_type: MessageType) -> None:
        item: Dict[str, Any] = {
            "type": msg_type.name if hasattr(msg_type, "name") else str(msg_type),
            "text": chunk,
            "time": _now_ts(),
        }
        if msg_type == MessageType.ToolCalling:
            name = chunk.split(" ", 1)[0] if chunk else ""
            if name:
                item["tool_name"] = name
            args = chunk.split(" ", 1)[1] if " " in chunk else ""
            fmt_args = format_tool_args(args)
            if fmt_args is not None:
                item["text"] = fmt_args
        elif msg_type == MessageType.ToolResult:
            prefix = "[ToolResult] "
            if chunk.startswith(prefix):
                item["tool_result"] = chunk[len(prefix):]
        state.output_queue.put(item)

        # Emit SSE event via bus — send cumulative text so frontend can display full content
        part_id = _get_or_create_part_id(msg_type)
        # Accumulate the chunk into the per-type buffer
        key = msg_type.name if hasattr(msg_type, "name") else str(msg_type)
        part_text_buf[key] = part_text_buf.get(key, "") + chunk
        cumulative_text = part_text_buf[key]

        sse_type = ""
        if msg_type == MessageType.Thinking:
            sse_type = "reasoning"
        elif msg_type == MessageType.Text:
            sse_type = "text"
        elif msg_type == MessageType.ToolCalling:
            sse_type = "tool"
        elif msg_type == MessageType.ToolResult:
            sse_type = "tool_result"
        else:
            sse_type = msg_type.name if hasattr(msg_type, "name") else str(msg_type)

        bus.emit(BusEvent(
            type="message.part.updated",
            properties={
                "part": {
                    "id": part_id,
                    "sessionID": state.session_id,
                    "type": sse_type,
                    "text": cumulative_text,
                }
            },
        ))

    await prompt_async(
        prompt_str=text,
        session=state.session,
        output_function=_output_handler,
        info_print=False,
    )

    # Emit idle status after completion
    bus.emit(BusEvent(type="session.status", properties={
        "status": {"type": "idle", "sessionID": state.session_id},
    }))


async def _process_plan(
    state: SessionStateClass, text: str
) -> None:
    """Process a plan command: create planner session, stream output via SSE.

    Follows the same pattern as _process_prompt but uses a dedicated planner
    agent and emits standard message.part.updated events.
    """
    from pathlib import Path
    from kimix.tools.note import _set_enable_plan
    from kimix.utils import close_session_async

    plan_file = Path("plan.md")
    if plan_file.is_file():
        plan_file.unlink()

    # Emit running status
    bus.emit(BusEvent(type="session.status", properties={
        "status": {"type": "running", "sessionID": state.session_id},
    }))

    step_snapshot = _gen_id("snap")
    part_text_buf: Dict[str, str] = {}

    def _get_or_create_plan_part_id(key: str) -> str:
        if key in state.part_id_map:
            return state.part_id_map[key]
        state._part_counter += 1
        pid = f"prt_dummy_{state._part_counter:04x}"
        state.part_id_map[key] = pid
        return pid

    # Emit step-start
    step_part_id = _get_or_create_plan_part_id("step-start")
    bus.emit(BusEvent(
        type="message.part.updated",
        properties={
            "part": {
                "id": step_part_id,
                "sessionID": state.session_id,
                "type": "step-start",
                "snapshot": step_snapshot,
            }
        },
    ))

    planner_session = None
    plan_generated = False
    text_part_id = _get_or_create_plan_part_id("plan-text")
    cumulative_text = ""

    try:
        # Create a planner SDK session
        planner_session = await _create_session_async(
            agent_type=SystemPromptType.TodoMaker,
            agent_file=_default_agent_file_dir / "agent_planner.json",
        )
        planner_session.get_custom_data()["plan_writing_path"] = plan_file
        state.planner_session = planner_session

        # Enable plan tools
        _set_enable_plan(True)

        prompt_text = (
            "read the following requirement carefully and generate a comprehensive plan. "
            "save the complete plan to a file using the WritePlan tool.\n\n"
            f"Requirement:\n{text.strip()}"
        )

        max_plan_attempts = 3

        for attempt in range(max_plan_attempts):
            try:
                from kimi_cli.wire.types import (
                    ContentPart,
                    StepBegin,
                    StepInterrupted,
                    ToolCall,
                    ToolCallPart,
                    ToolResult,
                )

                async for wire_msg in planner_session.prompt(prompt_text, merge_wire_messages=True):
                    # ── ToolCall: emit tool SSE event ──────────────────
                    if isinstance(wire_msg, ToolCall):
                        tool_name = wire_msg.function.name if wire_msg.function else "unknown"
                        tool_args_raw = wire_msg.function.arguments if wire_msg.function else ""
                        tool_part_id = _get_or_create_plan_part_id(f"tool-{wire_msg.id}")
                        bus.emit(BusEvent(
                            type="message.part.updated",
                            properties={
                                "part": {
                                    "id": tool_part_id,
                                    "sessionID": state.session_id,
                                    "type": "tool",
                                    "tool": tool_name,
                                    "callID": wire_msg.id,
                                    "state": {
                                        "status": "running",
                                        "input": tool_args_raw,
                                    },
                                },
                            },
                        ))
                        continue

                    # ── ToolCallPart: incremental argument chunks ───────
                    if isinstance(wire_msg, ToolCallPart):
                        continue

                    # ── ToolResult: emit tool_result SSE event ──────────
                    if isinstance(wire_msg, ToolResult):
                        rv = wire_msg.return_value
                        output = rv.output if isinstance(rv.output, str) else (rv.message or "")
                        is_error = rv.is_error
                        status = "error" if is_error else "completed"
                        tool_part_id = _get_or_create_plan_part_id(f"tool-result-{wire_msg.tool_call_id}")
                        bus.emit(BusEvent(
                            type="message.part.updated",
                            properties={
                                "part": {
                                    "id": tool_part_id,
                                    "sessionID": state.session_id,
                                    "type": "tool_result",
                                    "text": output,
                                    "tool_result": output,
                                    "state": {"status": status},
                                },
                            },
                        ))
                        continue

                    # ── ContentPart (e.g. ImageURLPart) — emit as text ──
                    if isinstance(wire_msg, ContentPart):
                        try:
                            import orjson
                            raw = wire_msg.model_dump()
                            part_text = orjson.dumps(raw).decode("utf-8")
                            cumulative_text += part_text
                            bus.emit(BusEvent(
                                type="message.part.updated",
                                properties={
                                    "part": {
                                        "id": text_part_id,
                                        "sessionID": state.session_id,
                                        "type": "text",
                                        "text": cumulative_text,
                                    },
                                    "delta": part_text,
                                },
                            ))
                        except Exception:
                            pass
                        continue

                    # ── StepBegin: new step boundary ────────────────────
                    if isinstance(wire_msg, StepBegin):
                        bus.emit(BusEvent(
                            type="message.part.updated",
                            properties={
                                "part": {
                                    "id": _get_or_create_plan_part_id("step-start"),
                                    "sessionID": state.session_id,
                                    "type": "step-start",
                                    "snapshot": step_snapshot,
                                },
                            },
                        ))
                        continue

                    # ── StepInterrupted ──────────────────────────────────
                    if isinstance(wire_msg, StepInterrupted):
                        continue

                    # ── ThinkPart (reasoning) — cumulative text ─────────
                    if hasattr(wire_msg, 'think') and wire_msg.think:
                        cumulative_text += wire_msg.think
                        bus.emit(BusEvent(
                            type="message.part.updated",
                            properties={
                                "part": {
                                    "id": text_part_id,
                                    "sessionID": state.session_id,
                                    "type": "text",
                                    "text": cumulative_text,
                                },
                                "delta": wire_msg.think,
                            },
                        ))

                    # ── TextPart — cumulative text ──────────────────────
                    if hasattr(wire_msg, 'text') and wire_msg.text:
                        cumulative_text += wire_msg.text
                        bus.emit(BusEvent(
                            type="message.part.updated",
                            properties={
                                "part": {
                                    "id": text_part_id,
                                    "sessionID": state.session_id,
                                    "type": "text",
                                    "text": cumulative_text,
                                },
                                "delta": wire_msg.text,
                            },
                        ))

                if plan_file.exists() and plan_file.stat().st_size > 0:
                    plan_generated = True
                    plan_content = plan_file.read_text(encoding="utf-8", errors="replace")
                    plan_display = f"\n\n--- Generated Plan ---\n{plan_content}"
                    cumulative_text += plan_display
                    bus.emit(BusEvent(
                        type="message.part.updated",
                        properties={
                            "part": {
                                "id": text_part_id,
                                "sessionID": state.session_id,
                                "type": "text",
                                "text": cumulative_text,
                            },
                            "delta": plan_display,
                        },
                    ))

                    # ── Confirmation flow ────────────────────────────────
                    state._plan_confirm_event = asyncio.Event()
                    state._plan_feedback = None

                    # Emit plan.confirm event so the frontend can show a dialog
                    bus.emit(BusEvent(
                        type="message.part.updated",
                        properties={
                            "part": {
                                "id": _get_or_create_plan_part_id("plan-confirm"),
                                "sessionID": state.session_id,
                                "type": "plan.confirm",
                                "text": plan_content,
                            },
                        },
                    ))

                    # Wait for confirmation (accept or revise) with timeout
                    try:
                        await asyncio.wait_for(state._plan_confirm_event.wait(), timeout=300.0)
                    except asyncio.TimeoutError:
                        state._plan_feedback = None
                        pass

                    confirm_event = state._plan_confirm_event
                    state._plan_confirm_event = None

                    if confirm_event is not None and confirm_event.is_set():
                        feedback = state._plan_feedback
                        state._plan_feedback = None
                        if feedback is not None and feedback != "accept":
                            # Revise: loop back with revision feedback
                            prompt_text = (
                                "The user reviewed the plan and requests the following changes:\n"
                                f"{feedback}\n\n"
                                "Please revise the plan accordingly and save the updated plan "
                                "using the WritePlan tool. Overwrite the existing plan file."
                            )
                            plan_generated = False
                            continue

                    # Accepted or timeout — proceed to finish
                    break

                if attempt < max_plan_attempts - 1:
                    prompt_text = (
                        "The plan file was not generated. "
                        "Please generate the plan and save it using the WritePlan tool.\n\n"
                        f"Requirement:\n{text.strip()}"
                    )
            except Exception as exc:
                if attempt == max_plan_attempts - 1:
                    raise
                import asyncio
                await asyncio.sleep(1)

    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("[DummySessionManager] Plan error: %s", exc, exc_info=True)
        plan_generated = False
    finally:
        # Emit step-finish
        reason = "stop" if plan_generated else (
            "cancelled" if hasattr(state, 'planner_session') and state.planner_session is None and not plan_generated
            else "Plan generation failed"
        )
        bus.emit(BusEvent(
            type="message.part.updated",
            properties={
                "part": {
                    "id": _get_or_create_plan_part_id("step-finish"),
                    "sessionID": state.session_id,
                    "type": "step-finish",
                    "reason": reason,
                    "snapshot": step_snapshot,
                }
            },
        ))

        # Emit idle status
        bus.emit(BusEvent(type="session.status", properties={
            "status": {"type": "idle", "sessionID": state.session_id},
        }))

        # Clean up
        _set_enable_plan(False)
        if planner_session is not None:
            try:
                await close_session_async(planner_session)
            except Exception:
                pass
        state.planner_session = None


# ── Dummy Session Manager ────────────────────────────────────────


class DummySessionManager:
    """Session manager backed by real SDK sessions.

    Creates actual ``kimi_agent_sdk.Session`` instances for each API session.
    Prompts are processed in background threads that call ``prompt_async``.
    """

    def __init__(self) -> None:
        self._session_states: Dict[str, SessionStateClass] = {}
        self._session_infos: Dict[str, SessionInfo] = {}
        self._lock = threading.Lock()

    # ── Session CRUD ─────────────────────────────────────────────

    async def create_session(
        self,
        title: Optional[str] = None,
        supervisor: bool = False,
        ralph_loop: int = 0,
    ) -> SessionInfo:
        """POST /session

        Creates a real SDK session and starts a background work thread.
        """
        session_id = _gen_id("ses")
        now = _now_ts()

        print(
            f"[DummySessionManager] create_session("
            f"title={title!r}, supervisor={supervisor!r}, ralph_loop={ralph_loop!r})"
        )

        # Create the appropriate SDK session type (async, no asyncio.run)
        if supervisor:
            sdk_session = await _create_session_async(
                session_id=session_id,
                agent_file=_default_agent_file_dir / 'agent_boss.json',
                agent_type=SystemPromptType.Supervisor,
                max_ralph_iterations=ralph_loop if ralph_loop > 0 else None,
            )
        else:
            sdk_session = await _create_session_async(
                session_id=session_id,
                max_ralph_iterations=ralph_loop if ralph_loop > 0 else None,
            )

        info = SessionInfo(
            id=session_id,
            title=title or "Dummy Session",
            createdAt=now,
            updatedAt=now,
        )

        state = SessionStateClass(session=sdk_session, session_id=session_id)
        state.work_thread = threading.Thread(
            target=_work_loop,
            args=(state,),
            name=f"session-{session_id}",
            daemon=True,
        )
        state.work_thread.start()

        with self._lock:
            self._session_states[session_id] = state
            self._session_infos[session_id] = info

        return info

    def get_session(self, session_id: str) -> SessionInfo:
        """GET /session/{sessionID}"""
        print(f"[DummySessionManager] get_session(session_id={session_id!r})")
        with self._lock:
            info = self._session_infos.get(session_id)
        if info is None:
            raise KeyError(f"Session not found: {session_id}")
        return info

    def list_sessions(self) -> List[SessionInfo]:
        """GET /session"""
        print("[DummySessionManager] list_sessions()")
        with self._lock:
            return sorted(
                self._session_infos.values(),
                key=lambda s: s.updatedAt,
                reverse=True,
            )

    async def delete_session(self, session_id: str) -> bool:
        """DELETE /session/{sessionID}"""
        print(f"[DummySessionManager] delete_session(session_id={session_id!r})")
        with self._lock:
            state = self._session_states.pop(session_id, None)
            self._session_infos.pop(session_id, None)
        if state is None:
            return False
        state.stop()
        return True

    def get_session_status(self, session_id: str) -> SessionStatus:
        """GET /session/{sessionID}/status

        Returns :class:`SessionStatus` with type ``running`` or ``idle``.
        When idle, also provides context usage percentage and token count.
        """
        print(
            f"[DummySessionManager] get_session_status("
            f"session_id={session_id!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")

        # Determine if the session is actively processing a prompt
        is_running = state.running and state.is_working

        token_count = 0
        context_usage = 0.0
        if state.session is not None:
            try:
                sdk_status = state.session.status
                token_count = getattr(sdk_status, "context_tokens", 0)
                context_usage = getattr(sdk_status, "context_usage", 0.0)
            except Exception:
                pass

        return SessionStatus(
            type="running" if is_running else "idle",
            token_count=token_count,
            context_usage=context_usage,
        )

    # ── Messages ─────────────────────────────────────────────────

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """GET /session/{sessionID}/message

        Drains the output_queue and returns collected messages.
        """
        print(
            f"[DummySessionManager] get_messages("
            f"session_id={session_id!r}, limit={limit!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")

        messages: List[Dict[str, Any]] = []
        while not state.output_queue.empty():
            try:
                messages.append(state.output_queue.get_nowait())
            except queue.Empty:
                break
        if limit is not None and limit > 0 and len(messages) > limit:
            messages = messages[-limit:]
        return messages

    # ── Prompt (fire-and-forget) ──────────────────────────────────

    async def prompt_async(
        self,
        session_id: str,
        text: str,
    ) -> None:
        """POST /session/{sessionID}/prompt_async

        Enqueues the prompt text into the session's input_queue.
        The background work thread will pick it up and call prompt_async.
        """
        print(
            f"[DummySessionManager] prompt_async("
            f"session_id={session_id!r}, text={text!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        state.input_queue.put(text)

    # ── Abort ────────────────────────────────────────────────────

    def abort_session(self, session_id: str) -> bool:
        """POST /session/{sessionID}/abort"""
        print(f"[DummySessionManager] abort_session(session_id={session_id!r})")
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        if state.session is not None:
            try:
                state.session.cancel()
            except Exception:
                pass
        if state.planner_session is not None:
            try:
                state.planner_session.cancel()
            except Exception:
                pass
        return True

    # ── Options ─────────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> bool:
        """GET /session/{sessionID}/clear"""
        print(f"[DummySessionManager] clear_session(session_id={session_id!r})")
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        if state.session is not None:
            try:
                await state.session.clear()
            except Exception:
                pass
        # Drain queues
        while not state.output_queue.empty():
            try:
                state.output_queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            info = self._session_infos.get(session_id)
            if info:
                info.updatedAt = _now_ts()
        return True

    async def compact_session(
        self, session_id: str, keep: Optional[int] = None
    ) -> bool:
        """GET /session/{sessionID}/compact"""
        print(
            f"[DummySessionManager] compact_session("
            f"session_id={session_id!r}, keep={keep!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        if state.session is not None:
            try:
                await state.session.compact()
            except Exception:
                pass
        with self._lock:
            info = self._session_infos.get(session_id)
            if info:
                info.updatedAt = _now_ts()
        return True

    async def get_session_context(
        self, session_id: str, keep: Optional[int] = None
    ) -> Dict[str, Any]:
        """GET /session/{sessionID}/context"""
        print(
            f"[DummySessionManager] get_session_context("
            f"session_id={session_id!r}, keep={keep!r})"
        )
        return {"sessionID": session_id, "context_usage": None}

    async def export_session(
        self, session_id: str, output_path: Optional[str] = None
    ) -> tuple[str, int]:
        """GET /session/{sessionID}/export"""
        print(
            f"[DummySessionManager] export_session("
            f"session_id={session_id!r}, output_path={output_path!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        if state.session is not None:
            try:
                return await state.session.export(output_path=output_path)
            except Exception:
                pass
        return (output_path or f"dummy_export_{session_id}.json", 0)

    # ── Plan ─────────────────────────────────────────────────

    async def plan_async(self, session_id: str, text: str) -> None:
        """POST /session/{sessionID}/plan

        Enqueues a plan command into the session's input_queue.
        The background work thread will pick it up and process the plan.
        """
        print(
            f"[DummySessionManager] plan_async("
            f"session_id={session_id!r}, text={text!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        state.input_queue.put(PlanCommand(text=text))

    async def confirm_plan_async(self, session_id: str, action: str, feedback: Optional[str] = None) -> bool:
        """POST /session/{sessionID}/plan/confirm

        Confirm or revise a plan after generation.
        "action" is "accept" or "revise".
        "feedback" is revision instructions when action="revise".
        """
        print(
            f"[DummySessionManager] confirm_plan_async("
            f"session_id={session_id!r}, action={action!r}, feedback={feedback!r})"
        )
        with self._lock:
            state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"Session not found: {session_id}")
        if state._plan_confirm_event is None:
            return False  # No confirmation pending
        state._plan_feedback = action if action == "accept" else feedback
        state._plan_confirm_event.set()
        return True


# Global singleton (drop-in for ``from kimix.server.session_manager import session_manager``)
session_manager = DummySessionManager()
