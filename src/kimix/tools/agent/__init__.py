from __future__ import annotations

import asyncio
import orjson
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field
from kimi_cli.session import Session
from kimix.ui.printing import MessageType
from kimix.utils import close_session_async, _create_session_async
from kimix.utils.system_prompt import SystemPromptType
import kimix.base as base
import kimix.utils as utils

from .store import AgentSessionStore, AgentSessionEntry, ConversationTurn

# Module-level registry for cross-session lookup (AskParent tool → entry)
_agent_entries: dict[str, AgentSessionEntry] = {}


def _register_entry(session_id: str, entry: AgentSessionEntry) -> None:
    _agent_entries[session_id] = entry


def _get_entry(session_id: str) -> AgentSessionEntry | None:
    return _agent_entries.get(session_id)


def _unregister_entry(session_id: str) -> None:
    _agent_entries.pop(session_id, None)


class SubAgentParams(BaseModel):
    model_config = {"populate_by_name": True}

    prompt: str = Field(
        alias="task",  # common LLM variant
        description="Task instructions for the sub-agent. Accepts `prompt` or `task`.",
    )
    session_id: str | None = Field(
        default=None,
        alias="session",  # common LLM variant
        description="Optional session ID to resume an existing sub-agent session. "
        "Accepts `session_id` or `session`.",
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
        "Use the AgentRespond tool instead."
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


class AskParentParams(BaseModel):
    question: str = Field(description="The specific question you need answered.")
    context: str | None = Field(
        default=None,
        description="Optional context about what you're trying to do."
    )


class AskParent(CallableTool2):
    name: str = "ask_parent"
    description: str = "Ask the parent agent a clarifying question when you need more information to proceed. The parent will see your question and respond in the next turn."
    params: type[BaseModel] = AskParentParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AskParentParams) -> ToolReturnValue:
        entry = _get_entry(getattr(self._session, "id", None))
        if entry is not None:
            entry.pending_question = params.question
            entry.state = "awaiting_response"
        return ToolOk(
            output="Question sent to parent agent. Waiting for response...",
            brief="Asked parent agent",
        )


class Agent(CallableTool2):
    name: str = "Agent"
    description: str = (
        "Launch a sub-agent for a task. "
        "Use AgentRespond to answer a sub-agent's pending question."
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
                        work_dir = Path(self._session.dir) if hasattr(self._session, "dir") else Path(".")
                        for fp in params.context_files:
                            try:
                                file_path = work_dir / fp
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

                if err_msg:
                    output_prefix = f"Session ID: {session_id}\n\n"
                    # The prompt is intentionally not echoed in the brief: it is
                    # streamed live (formatted and colored) by the CLI printer
                    # while the tool call is generated (see kimix.base), so
                    # printing it here would show it twice.
                    result = ToolError(
                        output=output_prefix + output_text,
                        message=err_msg,
                        brief="sub-agent task failed",
                    )
                    result.extras = {
                        "session_id": session_id,
                        "status": "closed",
                        "turn_count": len(collector.turns),
                    }
                    if params.return_history:
                        result.extras["conversation_history"] = self._format_history(
                            collector.turns, params.history_format
                        )
                    await close_session_async(session)
                    store.close(session_id)
                    _unregister_entry(session_id)
                    return result

                # Check if sub-agent asked parent for clarification
                current_entry = store.get(session_id)
                if current_entry and current_entry.state == "awaiting_response":
                    current_entry.conversation_history = collector.turns
                    current_entry.total_turns = len(collector.turns)
                    current_entry.last_accessed = time.time()
                    _register_entry(session_id, current_entry)
                    extras: dict[str, Any] = {
                        "session_id": session_id,
                        "status": "awaiting_response",
                        "turn_count": len(collector.turns),
                        "question": current_entry.pending_question,
                    }
                    if params.return_history:
                        extras["conversation_history"] = self._format_history(
                            collector.turns, params.history_format
                        )
                    output_prefix = f"Session ID: {session_id}\n\n"
                    result = ToolOk(
                        output=output_prefix + output_text,
                        brief="Sub-agent is awaiting a response",
                    )
                    result.extras = extras
                    return result

                extras: dict[str, Any] = {
                    "session_id": session_id,
                    "status": "closed" if params.close_session else "continued",
                    "turn_count": len(collector.turns),
                }
                if params.return_history:
                    extras["conversation_history"] = self._format_history(
                        collector.turns, params.history_format
                    )

                await self._update_store(params, session, session_id, is_reused, collector.turns)

                output_prefix = f"Session ID: {session_id}\n\n"
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
                f"Sub-agent made {tool_calls} tool call(s) and "
                f"produced {len(text_turns)} text response(s) "
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
                return entry.session, params.session_id, True

        session_id = params.session_id or str(uuid.uuid4())
        custom_config = self._session.custom_config
        chat_provider = custom_config.get("chat_provider")
        default_sub_provider = (
            base.get_default_sub_provider("sub_agent")
            or custom_config.get("provider_dict", base._default_provider)
        )

        session = await _create_session_async(
            session_id=session_id,
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

        return session, session_id, False

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
    pass


class AgentList(CallableTool2):
    name: str = "AgentList"
    description: str = "List all active subagent sessions."
    params: type[BaseModel] = AgentListParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AgentListParams) -> ToolReturnValue:
        store = _get_store(self._session)
        sessions = store.list_active()
        if not sessions:
            return ToolOk(output="No active sub-agents.", brief="No sub-agents")

        # Format as Markdown table
        lines = [
            "| Session ID | Turns | State | Last Accessed |",
            "|------------|-------|-------|---------------|",
        ]
        for s in sessions:
            sid = s.get("session_id", "-")
            turns = s.get("total_turns", 0)
            state = s.get("state", "unknown")
            last = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("last_accessed", 0)))
            lines.append(f"| `{sid}` | {turns} | {state} | {last} |")

        output = "\n".join(lines)
        result = ToolOk(output=output, brief=f"{len(sessions)} active sub-agent(s)")
        result.extras = {"sessions": sessions}
        return result


class AgentList(CallableTool2):
    name: str = "AgentList"
    description: str = "List all active subagent sessions."
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

    session_id: str = Field(
        alias="session",  # common LLM variant
        description="Subagent session ID to close. Accepts `session_id` or `session`.",
    )


class AgentClose(CallableTool2):
    name: str = "AgentClose"
    description: str = "Close an active subagent session and free its resources."
    params: type[BaseModel] = AgentCloseParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session

    async def __call__(self, params: AgentCloseParams) -> ToolReturnValue:
        store = _get_store(self._session)
        entry = store.get(params.session_id)
        if entry is None:
            return ToolError(
                output="",
                message="Session not found",
                brief="Session not found",
            )
        await close_session_async(entry.session)
        store.close(params.session_id)
        return ToolOk(
            output=f"Session {params.session_id} closed.",
            brief="Session closed",
        )
