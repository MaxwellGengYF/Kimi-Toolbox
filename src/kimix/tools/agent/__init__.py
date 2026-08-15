from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import orjson
from kaos.path import KaosPath
from kimi_cli.session import Session
from pydantic import AliasChoices, BaseModel, Field

import kimix.base as base
import kimix.utils as utils
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_agent_sdk import Session as SdkSession
from kimix.tools.prompt_common import accepts_alias_text
from kimix.ui.printing import MessageType
from kimix.utils import _create_session_async, close_session_async
from kimix.utils import _globals as _session_globals
from kimix.utils.system_prompt import SystemPromptType

from .store import AgentSessionEntry, AgentSessionStore, ConversationTurn

# Module-level registry for cross-session lookup (AskParent tool → entry)
_agent_entries: dict[str, AgentSessionEntry] = {}

# Cross-agent messaging registry: agent id (session id) -> live SDK Session.
# Populated when an ``Agent`` tool spawns/resumes a sub-agent; used by the
# ``AskAgent`` tool to resolve the target and push a steer into its loop
# (``Steer.from_session`` needs ``session._cli.soul``).
_agent_sessions: dict[str, SdkSession] = {}


def _register_agent_session(session_id: str, session: SdkSession | None) -> None:
    if session_id and session is not None:
        _agent_sessions[session_id] = session


def _get_agent_session(session_id: str) -> SdkSession | None:
    if not session_id:
        return None
    return _agent_sessions.get(session_id)


def _unregister_agent_session(session_id: str) -> None:
    _agent_sessions.pop(session_id, None)


def _cli_session_id(session: Any) -> str:
    """Best-effort session id for a (possibly SDK-wrapped) session object.

    ``kimi_cli.session.Session`` exposes ``.id`` directly; the SDK wrapper
    (``kimi_agent_sdk.Session``) keeps it at ``session._cli.session.id``.
    """
    if session is None:
        return ""
    raw = getattr(session, "id", None)
    if raw:
        return str(raw)
    cli = getattr(session, "_cli", None)
    cli_session = getattr(cli, "session", None) if cli is not None else None
    return str(getattr(cli_session, "id", None) or "")


def _session_work_dir(session: Any) -> KaosPath | None:
    """Best-effort working directory of a (possibly SDK-wrapped) session.

    ``kimi_cli.session.Session`` exposes ``.work_dir`` directly; the SDK
    wrapper (``kimi_agent_sdk.Session``) keeps it at
    ``session._cli.session.work_dir``. Returns ``None`` when the session has
    no resolvable work dir (callers fall back to the process CWD).
    """
    if session is None:
        return None
    raw = getattr(session, "work_dir", None)
    if raw:
        return raw if isinstance(raw, KaosPath) else KaosPath(str(raw))
    cli = getattr(session, "_cli", None)
    cli_session = getattr(cli, "session", None) if cli is not None else None
    nested = getattr(cli_session, "work_dir", None)
    if nested:
        return nested if isinstance(nested, KaosPath) else KaosPath(str(nested))
    return None


def _sdk_session_by_id(session_id: str) -> SdkSession | None:
    """Fallback: find the live SDK session whose CLI session id matches.

    Sessions created through ``kimix.utils.session`` are tracked in
    ``_session_globals._live_sessions``; this rescues targets that were never
    explicitly registered via :func:`_register_agent_session`.
    """
    if not session_id:
        return None
    for sdk in list(_session_globals._live_sessions):
        if _cli_session_id(sdk) == session_id:
            return sdk
    return None


def _register_entry(session_id: str, entry: AgentSessionEntry) -> None:
    _agent_entries[session_id] = entry


def _get_entry(session_id: str) -> AgentSessionEntry | None:
    return _agent_entries.get(session_id)


def _unregister_entry(session_id: str) -> None:
    _agent_entries.pop(session_id, None)


# Pending messages for agents that are idle or closed: target session id ->
# list of formatted messages. ``AskAgent`` queues a message here when the
# target cannot be steered right now (soul not running) or its session is no
# longer live (closed / unregistered). ``Agent`` drains the queue into the
# target's prompt the next time the session is resumed with a new task, so
# the message is *listed at the next prompt* instead of being lost.
_pending_messages: dict[str, list[str]] = {}


# Max queued messages kept per target to bound memory when a session is
# never resumed again.
_MAX_PENDING_MESSAGES_PER_TARGET = 50


def _queue_pending_message(target_id: str, message: str) -> None:
    """Append *message* to the pending-message queue for *target_id*."""
    pending = _pending_messages.setdefault(target_id, [])
    if len(pending) >= _MAX_PENDING_MESSAGES_PER_TARGET:
        return
    pending.append(message)


def _drain_pending_messages(session_id: str) -> list[str]:
    """Pop and return queued messages for *session_id* (empty when none)."""
    return _pending_messages.pop(session_id, [])


def _pending_message_count(session_id: str) -> int:
    """Number of queued (not yet delivered) messages for *session_id*."""
    return len(_pending_messages.get(session_id, []))


def _format_pending_messages(messages: list[str]) -> str:
    """Render queued messages as a block to list at the next prompt."""
    if not messages:
        return ""
    lines = [
        "<pending-messages>",
        "You have the following queued message(s) from the parent agent "
        "(sent while you were idle or not running):",
    ]
    lines.extend(f"{i}. {m}" for i, m in enumerate(messages, 1))
    lines.append("</pending-messages>")
    return "\n".join(lines)


class SubAgentParams(BaseModel):
    model_config = {"populate_by_name": True}

    description: str | None = Field(
        default=None,
        description=(
            "A short (3-5 word) description of the delegated task, for display."
        ),
    )
    prompt: str = Field(
        validation_alias=AliasChoices("prompt", "task"),
        description=(
            "The complete, self-contained task for the subagent. It does not "
            "share this conversation's context, so include everything it needs. "
            + accepts_alias_text("prompt", "task", word=False)
        ),
    )
    run_in_background: bool = Field(
        default=True,
        description=(
            "Whether to run in the background and return a durable subagent id "
            "immediately. Defaults to true. Set false to wait for the result "
            "when your next action depends on it."
        ),
    )
    session_id: str | None = Field(
        default=None,
        alias="session",  # common LLM variant
        description=(
            "Optional session ID to resume an existing sub-agent session. "
            + accepts_alias_text("session_id", "session", word=False)
        ),
    )
    close_session: bool = Field(
        default=True,
        description="Close the subagent session after this prompt. Set to False to keep it open for future follow-up."
    )
    return_history: bool = Field(
        default=False,
        description="Return the full conversation history in extras."
    )
    history_format: Literal["json", "markdown", "summary"] = Field(
        default="json",
        description="'json': Raw conversation turns in JSON. "
        "'markdown': Formatted as Markdown with headings. "
        "'summary': Concise summary of what the sub-agent did.",
    )
    response: str | None = Field(
        default=None,
        description="[Deprecated] Response to the sub-agent's pending question. "
        "Use the send_message tool instead."
    )
    context_files: list[str] | None = Field(
        default=None,
        description="File paths to pre-read into the sub-agent's context before the prompt. "
        "Each file's content is included as context."
    )
    context_data: dict[str, Any] | None = Field(
        default=None,
        description="Structured JSON data to pass as context to the sub-agent."
    )
    inherit_context: bool = Field(
        default=False,
        description=(
            "When True, a NEW sub-agent session is initialized by copying the "
            "parent agent's current session context (its conversation history "
            "so far), mirroring the CLI `/store` + `/load` session-copy logic "
            "in `src/kimix/cli_impl/commands.py`: the parent session directory "
            "is copied to the new sub-agent session id via "
            "`kimi_cli.session.Session.copy` and the sub-agent resumes from "
            "that copy. Ignored when `session_id` resolves to an active "
            "sub-agent session (which is reused as-is)."
        ),
    )


def _get_store(session: Session) -> AgentSessionStore:
    store = session.custom_data.get("agent_conversation_store")
    if store is None:
        store = AgentSessionStore()
        session.custom_data["agent_conversation_store"] = store
    return store


class _AgentConversationCollector:
    def __init__(self) -> None:
        self.turns: list[ConversationTurn] = []
        self.text_buffer: list[str] = []
        self.think_buffer: list[str] = []
        self.tool_buffer: list[str] = []
        self.last_msg_type: MessageType | None = None

    def _finalize_previous(self) -> None:
        if self.text_buffer:
            text = "".join(self.text_buffer)
            self.text_buffer.clear()
            self.turns.append(ConversationTurn(
                role="assistant",
                content=text,
                timestamp=time.time(),
                metadata={"type": "text"},
            ))
        if self.think_buffer:
            text = "".join(self.think_buffer)
            self.think_buffer.clear()
            self.turns.append(ConversationTurn(
                role="assistant",
                content=text,
                timestamp=time.time(),
                metadata={"type": "thinking"},
            ))
        if self.tool_buffer:
            text = "".join(self.tool_buffer)
            self.tool_buffer.clear()
            self.turns.append(ConversationTurn(
                role="tool",
                content=text,
                timestamp=time.time(),
                metadata={"type": "tool_call"},
            ))

    def consume(self, text: str, msg_type: MessageType) -> None:
        if msg_type == MessageType.Text:
            if self.last_msg_type not in (None, MessageType.Text):
                self._finalize_previous()
            self.text_buffer.append(text)
        elif msg_type == MessageType.Thinking:
            if self.last_msg_type not in (None, MessageType.Thinking):
                self._finalize_previous()
            self.think_buffer.append(text)
        elif msg_type in (MessageType.ToolCalling, MessageType.ToolCallingPart):
            if self.last_msg_type not in (None, MessageType.ToolCalling, MessageType.ToolCallingPart):
                self._finalize_previous()
            if text:
                self.tool_buffer = [text]
        elif msg_type == MessageType.ToolResult:
            self._finalize_previous()
            self.turns.append(ConversationTurn(
                role="tool",
                content=text,
                timestamp=time.time(),
                metadata={"type": "tool_result"},
            ))
        self.last_msg_type = msg_type

    def finalize_user_turn(self, prompt: str) -> None:
        self._finalize_previous()
        self.turns.append(ConversationTurn(
            role="user",
            content=prompt,
            timestamp=time.time(),
        ))

    def finalize_assistant_turn(self) -> str:
        self._finalize_previous()
        output_parts: list[str] = []
        for turn in self.turns:
            if (
                turn.role == "assistant"
                and turn.metadata is not None
                and turn.metadata.get("type") == "text"
            ):
                if isinstance(turn.content, str):
                    output_parts.append(turn.content)
        return "".join(output_parts)


class AskAgentParams(BaseModel):
    message: str = Field(
        validation_alias=AliasChoices("message", "question"),
        description=(
            "The message to deliver to the subagent. "
            "Delivered immediately if the target is running; otherwise queued "
            "and listed at its next prompt."
        ),
    )
    subagent_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("subagent_id", "id"),
        description=(
            "The subagent id returned when the background subagent was started. "
            "Optional: omit to message the most recently active sub-agent. "
            "Ignored for sub-agents, which always message their parent."
        ),
    )


class AskAgent(CallableTool2):
    name: str = "send_message"
    description: str = (
        "Send a message to a background subagent by its subagent id, continuing "
        "the same conversation. It becomes the subagent's next turn: if it is "
        "still working, the message waits until its current turn finishes, so it "
        "cannot redirect work already underway. This call returns no answer from "
        "the subagent — only confirmation that the message was delivered — so "
        "use it to give it more work. A failure means the message was NOT "
        "delivered."
    )
    params: type[BaseModel] = AskAgentParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AskAgentParams) -> ToolReturnValue:
        caller_id = _cli_session_id(self._session)
        target_id, target_session, reason = self._resolve_target(params)
        if target_id and target_id == caller_id:
            return ToolError(
                output="",
                message="Cannot message yourself.",
                brief="Self message rejected",
            )

        message = params.message
        if caller_id:
            message = f"Message from agent '{caller_id}':\n{message}"

        # No live session (closed / never registered): persist the message so it
        # is listed in the target's next prompt instead of erroring out.
        if target_session is None:
            if target_id:
                _queue_pending_message(target_id, message)
                return ToolOk(
                    output=(
                        f"Agent '{target_id}' is not currently running (session closed "
                        "or idle). Message queued and will be listed in its next prompt."
                    ),
                    brief="Message queued",
                )
            return ToolError(
                output="",
                message=f"Cannot resolve target agent: {reason or 'target not found'}",
                brief="Target agent not found",
            )

        from kimi_cli.soul.steer import Steer

        steer = Steer.from_session(target_session)
        if steer is None:
            if target_id:
                _queue_pending_message(target_id, message)
                return ToolOk(
                    output=(
                        f"Agent '{target_id}' has no steerable session; message queued "
                        "and will be listed in its next prompt."
                    ),
                    brief="Message queued",
                )
            return ToolError(
                output="",
                message=f"Agent '{target_id or 'unknown'}' has no steerable session.",
                brief="Target not steerable",
            )

        delivered = await steer.push(message)
        if delivered:
            return ToolOk(
                output=f"Message delivered to agent '{target_id or 'unknown'}'.",
                brief="Message sent",
            )
        # The soul exists but is not running (idle): the steer queue would be
        # discarded as stale at the next turn init, so persist the message and
        # list it in the next prompt instead.
        if target_id:
            _queue_pending_message(target_id, message)
        return ToolOk(
            output=(
                f"Agent '{target_id or 'unknown'}' is not running; message queued "
                "and will be listed in its next prompt."
            ),
            brief="Message queued",
        )

    def _resolve_target(
        self, params: AskAgentParams
    ) -> tuple[str | None, SdkSession | None, str]:
        """Resolve ``(target_id, target_sdk_session, reason)`` for a message."""
        custom_config = getattr(self._session, "custom_config", None) or {}
        if custom_config.get("is_sub_agent"):
            # Sub-agents always message their parent; ``id`` is ignored.
            parent_id = str(custom_config.get("parent_session_id", "") or "")
            if not parent_id:
                return None, None, "sub-agent has no recorded parent_session_id"
            session = _get_agent_session(parent_id) or _sdk_session_by_id(parent_id)
            if session is None:
                return parent_id, None, f"parent agent '{parent_id}' is not registered"
            return parent_id, session, ""

        # Main agent: target by id, or default to the most recently active
        # sub-agent in this session's store.
        if params.subagent_id:
            target_id = str(params.subagent_id)
            session = _get_agent_session(target_id) or _sdk_session_by_id(target_id)
            if session is None:
                return target_id, None, f"agent '{target_id}' is not registered"
            return target_id, session, ""

        store = _get_store(self._session)
        active = [e for e in store.entries.values() if e.is_active]
        if not active:
            return None, None, "no active sub-agents to message"
        target = max(active, key=lambda e: e.last_accessed)
        return target.session_id, target.session, ""


class Agent(CallableTool2):
    name: str = "subagent"
    description: str = (
        "Delegate a self-contained task to a subagent (a separate agent that "
        "works in its own context) to offload focused, independent work — "
        "research, a scoped implementation, an analysis — so it does not "
        "consume this conversation's context. The subagent returns its result, "
        "not its intermediate steps. Give it a complete, standalone prompt: it "
        "does not see this conversation. This tool runs in the background by "
        "default, immediately returns a durable subagent id, and keeps the "
        "child conversation available for later turns. When that run settles, "
        "the runtime sends the parent a notice containing its outcome and any "
        "final assistant message; send_message starts a later turn in the same "
        "child conversation. Set run_in_background: false only when your next "
        "action depends on receiving the result. "
        "Use send_message to answer a sub-agent's pending question."
    )
    params: type[SubAgentParams] = SubAgentParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._semaphore = asyncio.Semaphore(8)

    def __del__(self):
        if sys.is_finalizing():
            return
        store = self._session.custom_data.pop("agent_conversation_store", None)
        if isinstance(store, AgentSessionStore):
            for entry in list(store.entries.values()):
                _unregister_entry(entry.session_id)
                _unregister_agent_session(entry.session_id)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(close_session_async(entry.session))
                except RuntimeError:
                    try:
                        asyncio.run(close_session_async(entry.session))
                    except Exception:
                        pass

    async def __call__(self, params: SubAgentParams) -> ToolReturnValue:
        if self._session is not None and self._session.custom_config.get("is_sub_agent"):
            return ToolError(
                output='',
                message='Recursive sub-agent call detected',
                brief='sub-agent recursively'
            )
        async with self._semaphore:
            try:
                session, session_id, is_reused = await self._resolve_session(params)
                store = _get_store(self._session)
                entry = store.get(session_id)

                # Handle very long prompts by offloading to a temp file
                prompt_bytes = params.prompt.encode('utf-8')
                if len(prompt_bytes) > 100 * 1024:
                    cache_dir = Path('.kimix_cache')
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = cache_dir / f'prompt_{uuid.uuid4().hex}.md'
                    temp_path.write_bytes(prompt_bytes)
                    task_prompt = f'Please read the task from `{temp_path}` and execute it.'
                else:
                    task_prompt = params.prompt

                # Build prompt with context files / context_data if provided
                prompt = task_prompt
                if params.context_files or params.context_data:
                    context_parts = ["<context>"]
                    if params.context_files:
                        work_dir = _session_work_dir(self._session)
                        base_dir = (
                            Path(str(work_dir)) if work_dir is not None else Path(".")
                        )
                        for fp in params.context_files:
                            try:
                                file_path = base_dir / fp
                                content = file_path.read_text(encoding="utf-8", errors="replace")
                                context_parts.append(f"<file path='{fp}'>\n{content}\n</file>")
                            except Exception as e:
                                context_parts.append(f"<file path='{fp}' error='{e}'/>")
                    if params.context_data:
                        import orjson as _orjson
                        context_parts.append(f"<data>\n{_orjson.dumps(params.context_data, option=_orjson.OPT_INDENT_2).decode()}\n</data>")
                    context_parts.append("</context>")
                    context_block = "\n".join(context_parts)
                    prompt = f"{context_block}\n\n{prompt}"

                # Inject response to pending question if provided
                if is_reused and entry and entry.pending_question and params.response:
                    prompt = (
                        f"The parent agent responded to your question "
                        f"({entry.pending_question}):\n\n{params.response}\n\n"
                        f"Now, regarding your original task: {prompt}"
                    )
                    entry.pending_question = None
                    entry.state = "running"

                # List any messages queued by ``AskAgent`` while this sub-agent
                # was idle or its session was closed at the resumed prompt.
                pending_messages = _drain_pending_messages(session_id)
                if pending_messages:
                    prompt = f"{prompt}\n\n{_format_pending_messages(pending_messages)}"

                collector = _AgentConversationCollector()
                collector.finalize_user_turn(prompt)

                def output_function(text: str, msg_type: MessageType) -> None:
                    if text:
                        collector.consume(text, msg_type)

                err_msg: str | None = None
                try:
                    await utils.prompt_async(
                        prompt_str=prompt,
                        session=session,
                        output_function=output_function,
                        info_print=False,
                        merge_wire_messages=True, format_output=True
                    )
                except Exception as e:
                    err_msg = str(e)
                    collector.turns.append(ConversationTurn(
                        role="error",
                        content=err_msg,
                        timestamp=time.time(),
                        metadata={"error_type": type(e).__name__},
                    ))

                output_text = collector.finalize_assistant_turn()
                if not output_text:
                    output_text = "(no text output)"

                output_prefix = f"Session ID: {session_id}\n\n"

                if err_msg:
                    # The prompt is intentionally not echoed in the brief: it is
                    # streamed live (formatted and colored) by the CLI printer
                    # while the tool call is generated (see kimix.base), so
                    # printing it here would show it twice.
                    result = ToolError(
                        output=output_prefix + output_text,
                        message=err_msg,
                        brief="sub-agent task failed",
                    )
                    result.extras = self._build_extras(
                        params, session_id, collector.turns, "closed"
                    )
                    await close_session_async(session)
                    store.close(session_id)
                    _unregister_entry(session_id)
                    _unregister_agent_session(session_id)
                    return result

                # Check if sub-agent asked parent for clarification
                current_entry = store.get(session_id)
                if current_entry and current_entry.state == "awaiting_response":
                    current_entry.conversation_history = collector.turns
                    current_entry.total_turns = len(collector.turns)
                    current_entry.last_accessed = time.time()
                    _register_entry(session_id, current_entry)
                    result = ToolOk(
                        output=output_prefix + output_text,
                        brief="Sub-agent is awaiting a response",
                    )
                    result.extras = self._build_extras(
                        params,
                        session_id,
                        collector.turns,
                        "awaiting_response",
                        question=current_entry.pending_question,
                    )
                    return result

                extras = self._build_extras(
                    params,
                    session_id,
                    collector.turns,
                    "closed" if params.close_session else "continued",
                )

                await self._update_store(params, session, session_id, is_reused, collector.turns)

                result = ToolOk(
                    output=output_prefix + output_text,
                    brief="Sub-agent task completed",
                )
                result.extras = extras
                return result

            except Exception as exc:
                return ToolError(
                    output="",
                    message=str(exc),
                    brief="Failed to create sub-agent session",
                )

    def _build_extras(
        self,
        params: SubAgentParams,
        session_id: str,
        turns: list[ConversationTurn],
        status: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the shared extras dict for ``Agent`` result branches.

        Every branch includes ``session_id``, ``status`` and ``turn_count``;
        ``conversation_history`` is added when ``params.return_history`` is
        set. Branch-specific keys are passed as keyword extras (e.g.
        ``question`` for the awaiting-response branch).
        """
        extras: dict[str, Any] = {
            "session_id": session_id,
            "status": status,
            "turn_count": len(turns),
            **extra,
        }
        if params.return_history:
            extras["conversation_history"] = self._format_history(
                turns, params.history_format
            )
        return extras

    def _format_history(self, turns: list[ConversationTurn], format: str) -> list[dict[str, Any]] | str:
        """Format conversation turns according to history_format."""
        if format == "json":
            return [turn.model_dump() for turn in turns]
        elif format == "markdown":
            lines: list[str] = []
            for i, turn in enumerate(turns):
                role_icon = {"user": "👤", "assistant": "🤖", "tool": "🔧", "error": "❌", "system": "⚙️"}
                icon = role_icon.get(turn.role, "?")
                label = turn.metadata.get("type", turn.role) if turn.metadata else turn.role
                lines.append(f"### Turn {i+1}: {icon} {label}")
                content = turn.content if isinstance(turn.content, str) else str(turn.content)
                lines.append(content)
                lines.append("")
            return "\n".join(lines)
        elif format == "summary":
            tool_calls = sum(1 for t in turns if t.metadata and t.metadata.get("type") == "tool_call")
            tool_results = sum(1 for t in turns if t.metadata and t.metadata.get("type") == "tool_result")
            text_turns = [t for t in turns if t.role == "assistant" and t.metadata and t.metadata.get("type") == "text"]
            total_chars = sum(len(str(t.content)) for t in text_turns)
            return (
                f"Sub-agent made {tool_calls} tool call(s) with {tool_results} "
                f"result(s), and produced {len(text_turns)} text response(s) "
                f"({total_chars} total characters)."
            )
        return []

    async def _resolve_session(self, params: SubAgentParams) -> tuple[Any, str, bool]:
        store = _get_store(self._session)

        if params.session_id:
            entry = store.get(params.session_id)
            if entry is not None and entry.is_active:
                entry.last_accessed = time.time()
                _register_entry(params.session_id, entry)
                self._register_agent_sessions(entry.session, params.session_id)
                return entry.session, params.session_id, True

        session_id = params.session_id or str(uuid.uuid4())

        # Inherit the parent agent's context: copy the parent session
        # directory into the new sub-agent session id (mirrors the CLI
        # `/store` + `/load` logic in src/kimix/cli_impl/commands.py) so the
        # resumed sub-agent starts from the parent's conversation so far.
        inherited = False
        if params.inherit_context:
            await self._inherit_parent_context(session_id)
            inherited = True

        custom_config = self._session.custom_config
        chat_provider = custom_config.get("chat_provider")
        default_sub_provider = (
            base.get_default_sub_provider("sub_agent")
            or custom_config.get("provider_dict", base._default_provider)
        )

        session = await _create_session_async(
            session_id=session_id,
            work_dir=_session_work_dir(self._session),
            agent_file=base._default_agent_file_dir / 'agent_subagent.json',
            agent_type=SystemPromptType.TrivialSubAgent,
            provider_dict=default_sub_provider,
            chat_provider=chat_provider,
            resume=True,
            anonymous=False,
            max_ralph_iterations=0,
        )

        sub_custom_config = session.get_custom_config()
        if sub_custom_config is not None:
            sub_custom_config['is_sub_agent'] = True
            # Record who spawned this sub-agent so its ``AskAgent`` tool can
            # resolve the parent and push steers into the parent's loop.
            parent_id = _cli_session_id(self._session)
            if parent_id:
                sub_custom_config['parent_session_id'] = parent_id

        if inherited:
            await self._reset_inherited_system_prompt(session)

        self._register_agent_sessions(session, session_id)

        return session, session_id, False

    async def _inherit_parent_context(self, target_session_id: str) -> None:
        """Copy the parent session's context into *target_session_id*.

        Mirrors the CLI ``/store`` / ``/load`` commands
        (``src/kimix/cli_impl/commands.py``): the parent session directory is
        copied to the new sub-agent session id with
        ``kimi_cli.session.Session.copy`` — the same primitive ``/store`` uses
        to save a snapshot under a name and ``/load`` uses to load a snapshot
        into a fresh session — and the sub-agent session is then resumed from
        that copy.

        Unlike the CLI commands, the parent session is *not* released before
        copying: the parent agent's loop is still running (this tool call is
        part of its current turn), so its resources must stay intact. The
        parent's context database is in WAL mode and commits after every
        append, and no writes happen while this tool call executes, so the
        directory copy captures a consistent snapshot (SQLite replays the
        copied WAL on open).

        Raises:
            ValueError: If the parent session cannot be copied (no session id
                / work dir, or a session with the target id already exists).
        """
        parent_id = _cli_session_id(self._session)
        work_dir = _session_work_dir(self._session)
        if not parent_id:
            raise ValueError(
                'Cannot inherit parent context: parent session has no id'
            )
        if work_dir is None:
            raise ValueError(
                'Cannot inherit parent context: parent session has no work dir'
            )
        await Session.copy(work_dir, parent_id, target_session_id)

    async def _reset_inherited_system_prompt(self, session: Any) -> None:
        """Replace the inherited parent system prompt with the sub-agent's.

        The copied session directory carries the parent's persisted system
        prompt, which the soul adopts on resume
        (``agent.system_prompt_cached``). Reset it so the sub-agent runs with
        its own (TrivialSubAgent) system prompt instead of the parent's, which
        would describe the parent's toolset and role.

        Best-effort: silently skips when the soul internals are unavailable or
        rendering the prompt fails.
        """
        cli = getattr(session, '_cli', None)
        soul = getattr(cli, 'soul', None)
        agent = getattr(soul, 'agent', None)
        context = getattr(soul, 'context', None)
        if agent is None or context is None:
            return
        try:
            agent.system_prompt_cached = None
            prompt = agent.get_system_prompt()
            if prompt:
                await context.write_system_prompt(prompt)
        except Exception:
            return

    def _register_agent_sessions(
        self, child_session: Any, child_session_id: str
    ) -> None:
        """Register the parent and child SDK sessions for cross-agent messaging.

        ``AskAgent`` resolves its target through ``_agent_sessions``; the parent
        is registered under its own id so any sub-agent can steer it back.
        """
        parent_id = _cli_session_id(self._session)
        parent_sdk = _sdk_session_by_id(parent_id)
        if parent_id and parent_sdk is not None:
            _register_agent_session(parent_id, parent_sdk)
        if child_session_id:
            _register_agent_session(child_session_id, child_session)

    async def _update_store(
        self,
        params: SubAgentParams,
        session: Any,
        session_id: str,
        is_reused: bool,
        turns: list[ConversationTurn],
    ) -> None:
        store = _get_store(self._session)
        if params.close_session:
            await close_session_async(session)
            store.close(session_id)
            _unregister_entry(session_id)
            _unregister_agent_session(session_id)
        else:
            existing = store.get(session_id)
            if existing is None:
                await store.evict_lru_if_needed()
            created_at = existing.created_at if existing else time.time()
            entry = AgentSessionEntry(
                session=session,
                session_id=session_id,
                created_at=created_at,
                last_accessed=time.time(),
                conversation_history=turns,
                total_turns=len(turns),
                is_active=True,
                pending_question=existing.pending_question if existing else None,
                state=existing.state if existing else "completed",
            )
            store.put(entry)
            _register_entry(session_id, entry)


class AgentRespondParams(BaseModel):
    session_id: str = Field(description="Sub-agent session ID that asked a question.")
    response: str = Field(description="Answer to the sub-agent's pending question.")
    close_session: bool = Field(
        default=True,
        description="Close the subagent session after this response? Set to False to keep it open for more follow-up."
    )


class AgentRespond(CallableTool2):
    name: str = "AgentRespond"
    description: str = (
        "Answer a pending question from a sub-agent. "
        "Use this when a sub-agent has asked the parent agent a question (status='awaiting_response'). "
        "Provide your response and the sub-agent will continue with the answer."
    )
    params: type[BaseModel] = AgentRespondParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AgentRespondParams) -> ToolReturnValue:
        store = _get_store(self._session)
        entry = store.get(params.session_id)
        if entry is None:
            return ToolError(
                output="",
                message=f"Session '{params.session_id}' not found.",
                brief="Session not found",
            )
        if entry.state != "awaiting_response":
            return ToolError(
                output="",
                message=f"Session '{params.session_id}' is not awaiting a response (state: {entry.state}).",
                brief="Not awaiting response",
            )
        # Send the response to the sub-agent via the Agent tool
        agent = Agent(self._session)
        sub_params = SubAgentParams(
            prompt="Continue with the parent's answer.",
            session_id=params.session_id,
            close_session=params.close_session,
            response=params.response,
        )
        return await agent(sub_params)


class AgentListParams(BaseModel):
    scope: str = Field(
        default="children",
        description=(
            "`children` (default) lists direct children only; `descendants` "
            "walks the complete tree below you."
        ),
    )


class AgentList(CallableTool2):
    name: str = "list_agents"
    description: str = (
        "List your continuable background subagents by durable id and label. "
        "Use it to recall which ones you started, not to poll for completion — "
        "you are told when one finishes. Status comes from the live registry: "
        "running means the agent is working right now, idle means it is loaded "
        "but between turns, and ready means it exists only in storage — "
        "resumable, not terminal, and not a result waiting to be collected; a "
        "send_message starts a new turn on the same conversation, and a direct "
        "child remains a send_message candidate in every status. Scope "
        "`descendants` walks the whole tree below you in stable pre-order, "
        "annotating each entry with its durable direct-parent session id and "
        "depth. You may use send_message only for depth-1 entries; deeper "
        "entries are candidates for interrupt_agent only."
    )
    params: type[BaseModel] = AgentListParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AgentListParams) -> ToolReturnValue:
        store = _get_store(self._session)
        sessions = store.list_active()
        output = orjson.dumps(sessions, option=orjson.OPT_INDENT_2)
        return ToolOk(output=output, brief="Listed active subagents")


class AgentCloseParams(BaseModel):
    model_config = {"populate_by_name": True}

    agent_id: str = Field(
        validation_alias=AliasChoices("agent_id", "session", "session_id"),
        description=(
            "The agent id of the running agent to interrupt. "
            + accepts_alias_text("agent_id", "session", "session_id", word=False)
        ),
    )


class AgentClose(CallableTool2):
    name: str = "interrupt_agent"
    description: str = (
        "Request cancellation of a background agent's current turn by its agent "
        "id. The target may be your direct child or a deeper agent created "
        "under you. Only the current turn stops: messages already queued for "
        "the agent stay parked until a later send_message, agents it started "
        "keep running, and the agent itself stays available for follow-ups. "
        "This call returns as soon as the stop request is accepted, so the "
        "target may keep running briefly; interrupting an agent that already "
        "finished is an accepted no-op."
    )
    params: type[BaseModel] = AgentCloseParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AgentCloseParams) -> ToolReturnValue:
        store = _get_store(self._session)
        entry = store.get(params.agent_id)
        if entry is None:
            return ToolError(
                output="",
                message="Session not found",
                brief="Session not found",
            )
        await close_session_async(entry.session)
        store.close(params.agent_id)
        _unregister_agent_session(params.agent_id)
        return ToolOk(
            output=f"Session {params.agent_id} closed.",
            brief="Session closed",
        )
