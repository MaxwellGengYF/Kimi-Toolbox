"""Normalize public Kimi Agent SDK wire messages into transcript mutations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from types import SimpleNamespace
from typing import Any

import orjson

from kimi_agent_sdk import (
    ApprovalRequest,
    ApprovalResponse,
    AudioURLPart,
    BriefDisplayBlock,
    CompactionBegin,
    CompactionEnd,
    DiffDisplayBlock,
    DisplayBlock,
    ImageURLPart,
    ShellDisplayBlock,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    SubagentEvent,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    ToolCall,
    ToolCallPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
    UnknownDisplayBlock,
    VideoURLPart,
)
from kimix_gui.tool_display import (
    build_tool_call_content,
    build_tool_result_content,
    tool_family,
)
from kimix_gui.transcript_data import (
    ROOT_SOURCE,
    ActivityEntry,
    ActivityField,
    AppendText,
    ContentBlock,
    DiffBlock,
    EntrySource,
    FieldListBlock,
    FinishEntry,
    MediaBlock,
    NoticeEntry,
    QuestionBlock,
    QuestionItem,
    QuestionOption,
    RawBlock,
    ReplaceEntry,
    StartEntry,
    TextBlock,
    TextEntry,
    ToolActivity,
    ToolCallContent,
    ToolIdentity,
    ToolResultContent,
    TranscriptMutation,
    literal,
    localized,
)


@dataclass(frozen=True, slots=True)
class McpServer:
    """One MCP server's name and its raw wire state."""

    name: str
    state: str


@dataclass(frozen=True, slots=True)
class McpCounts:
    connected: int = 0
    total: int = 0
    tools: int = 0
    loading: bool = False
    servers: tuple[McpServer, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenCounts:
    total_input: int = 0
    new_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0


@dataclass(frozen=True, slots=True)
class StatusValues:
    """Status-line figures; the Qt status component owns their wording."""

    context_tokens: int | None = None
    max_context_tokens: int | None = None
    usage_percent: float | None = None
    tokens: TokenCounts | None = None
    mcp: McpCounts | None = None
    yolo_enabled: bool = False
    afk_enabled: bool = False


@dataclass(frozen=True, slots=True)
class NormalizedWireMessage:
    """Transcript mutations plus an optional independent status update."""

    mutations: tuple[TranscriptMutation, ...] = ()
    status: StatusValues | None = None


@dataclass(slots=True)
class PendingActivity:
    """All semantic state for one call id, including result-first calls."""

    call_id: str
    identity: ToolIdentity
    started: bool = False
    raw_arguments: str = ""
    extras: object = None
    call: ToolCallContent | None = None
    result: ToolResultContent | None = None

    def snapshot(self) -> ToolActivity:
        return ToolActivity(
            call_id=self.call_id,
            identity=self.identity,
            call=self.call,
            result=self.result,
        )


_OBSERVABILITY_EVENTS = {"LLMRequest", "LLMToolsSnapshot", "MCPToolsDiscovered"}
_MAX_PENDING_ACTIVITIES = 256


def _model_data(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    try:
        return model_dump(mode="json", exclude_none=True)
    except TypeError:
        return model_dump()


def _json_text(value: object) -> str:
    try:
        return orjson.dumps(
            _model_data(value),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=lambda item: str(item),
        ).decode("utf-8")
    except TypeError, ValueError:
        return str(value)


def _media_block(kind: str, media: object) -> MediaBlock:
    return MediaBlock(
        kind=kind,  # type: ignore[arg-type]
        url=str(getattr(media, "url", "")),
        media_id=str(media_id) if (media_id := getattr(media, "id", None)) else None,
    )


def content_blocks(value: object) -> tuple[ContentBlock, ...]:
    """Convert SDK content parts into blocks without flattening media into text."""

    if isinstance(value, str):
        return (TextBlock(literal(value)),) if value else ()
    if isinstance(value, TextPart):
        return (TextBlock(literal(value.text), format="markdown"),) if value.text else ()
    if isinstance(value, ThinkPart):
        return (TextBlock(literal(value.think), tone="muted"),) if value.think else ()
    if isinstance(value, ImageURLPart):
        return (_media_block("image", value.image_url),)
    if isinstance(value, AudioURLPart):
        return (_media_block("audio", value.audio_url),)
    if isinstance(value, VideoURLPart):
        return (_media_block("video", value.video_url),)
    if isinstance(value, list | tuple):
        return tuple(block for item in value for block in content_blocks(item))
    data = _model_data(value)
    if data is value:
        return (RawBlock(str(value)),)
    return (RawBlock(_json_text(data), label=literal(type(value).__name__)),)


def user_input_blocks(user_input: object) -> tuple[ContentBlock, ...]:
    """Normalize a TurnBegin/SteerInput payload without joining its media parts."""

    return content_blocks(user_input)


def user_input_text(user_input: object) -> str:
    """Diagnostic plain-text view of a normalized user input."""

    from kimix_gui.transcript_data import blocks_text

    return blocks_text(user_input_blocks(user_input)).strip()


def display_blocks(display: Sequence[object]) -> tuple[ContentBlock, ...]:
    """Normalize native display blocks into their semantic counterparts."""

    blocks: list[ContentBlock] = []
    for block in display:
        if isinstance(block, BriefDisplayBlock):
            if block.text:
                blocks.append(TextBlock(literal(block.text)))
            continue
        if isinstance(block, DiffDisplayBlock):
            blocks.append(
                DiffBlock(
                    path=block.path,
                    old_start=block.old_start,
                    new_start=block.new_start,
                    old_text=block.old_text,
                    new_text=block.new_text,
                    is_summary=block.is_summary,
                )
            )
            continue
        if isinstance(block, TodoDisplayBlock):
            from kimix_gui.transcript_data import TodoBlock, TodoItem

            blocks.append(
                TodoBlock(
                    tuple(
                        TodoItem(
                            title=item.title,
                            status=item.status,
                            depth=max(0, item.depth),
                            notes=item.notes or "",
                        )
                        for item in block.items
                    )
                )
            )
            continue
        if isinstance(block, ShellDisplayBlock):
            blocks.append(
                FieldListBlock(
                    (ActivityField("language", literal(block.language), role="secondary"),)
                )
            )
            continue
        if type(block).__name__ == "BackgroundTaskDisplayBlock":
            blocks.append(
                FieldListBlock(
                    tuple(
                        ActivityField(name, literal(value), role=role)
                        for name, value, role in (
                            ("status", getattr(block, "status", "?"), "primary"),
                            ("task_id", getattr(block, "task_id", "?"), "secondary"),
                            ("kind", getattr(block, "kind", "task"), "secondary"),
                            ("description", getattr(block, "description", ""), "detail"),
                        )
                        if value not in (None, "")
                    )
                )
            )
            continue
        if isinstance(block, UnknownDisplayBlock):
            blocks.append(RawBlock(_json_text(block.data), label=literal(block.type)))
            continue
        if isinstance(block, DisplayBlock):
            label = type(block).__name__.removesuffix("DisplayBlock") or block.type
            blocks.append(RawBlock(_json_text(block), label=literal(label)))
            continue
        blocks.append(RawBlock(_json_text(block), label=literal(type(block).__name__)))
    return tuple(blocks)


def _argument_object(arguments: str | None) -> tuple[Mapping[str, Any] | None, str]:
    """Decode tool JSON exactly once and distinguish incomplete from invalid input."""

    if not arguments or not arguments.strip():
        return None, "missing"
    stripped = arguments.rstrip()
    # Streaming fragments overwhelmingly end before the outer object can possibly
    # close. Avoid repeatedly parsing the whole growing buffer; the closing fragment
    # performs the one useful decode attempt and replaces the raw fallback.
    if stripped.startswith(("{", "[")) and not stripped.endswith(("}", "]")):
        return None, "partial"
    try:
        parsed = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        message = str(exc).lower()
        partial = (
            "unexpected end" in message
            or stripped.endswith(("{", "[", ":", ","))
            or (
                not stripped.endswith(("}", "]"))
                and (
                    stripped.count("{") > stripped.count("}")
                    or stripped.count("[") > stripped.count("]")
                )
            )
        )
        return None, "partial" if partial else "invalid"
    if not isinstance(parsed, Mapping):
        return None, "invalid"
    return parsed, "complete"


def _field(name: str, value: object, *, role: str = "detail", hint: str = "plain") -> ActivityField:
    return ActivityField(
        name=name,
        value=literal(_json_text(value) if isinstance(value, Mapping | list | tuple) else value),
        role=role,  # type: ignore[arg-type]
        hint=hint,  # type: ignore[arg-type]
    )


class WireNormalizer:
    """Stateful, single-pass SDK-message normalization boundary."""

    def __init__(self, *, scope: str = "root", source: EntrySource = ROOT_SOURCE) -> None:
        self._scope = scope
        self._source = source
        self._keys = count(1)
        self._stream_key: str | None = None
        self._stream_kind: str | None = None
        self._pending: dict[str, PendingActivity] = {}
        self._active_call_id: str | None = None
        self._subagents: dict[str, WireNormalizer] = {}
        self._status_values: dict[str, object] = {}

    def reset(self) -> None:
        self._stream_key = None
        self._stream_kind = None
        self._pending.clear()
        self._active_call_id = None
        self._subagents.clear()
        self._status_values.clear()

    def normalize(self, message: object) -> NormalizedWireMessage:
        if isinstance(message, TextPart):
            return self._stream("assistant", message.text, format="markdown")
        if isinstance(message, ThinkPart):
            return self._stream("thinking", message.think, tone="muted")

        mutations = list(self._finish_stream())
        if isinstance(message, TurnBegin):
            blocks = user_input_blocks(message.user_input)
            if blocks:
                mutations.append(
                    StartEntry(
                        TextEntry(
                            key=self._new_key("user"),
                            kind="user",
                            blocks=blocks,
                            source=self._source,
                        )
                    )
                )
            return NormalizedWireMessage(tuple(mutations))
        if type(message).__name__ == "SteerInput":
            blocks = user_input_blocks(getattr(message, "user_input", ""))
            if blocks:
                mutations.append(
                    StartEntry(
                        TextEntry(
                            key=self._new_key("user"),
                            kind="user",
                            blocks=blocks,
                            source=self._source,
                        )
                    )
                )
            return NormalizedWireMessage(tuple(mutations))
        if isinstance(message, ImageURLPart):
            return self._notice_with_prefix(
                mutations, "system", (_media_block("image", message.image_url),)
            )
        if isinstance(message, AudioURLPart):
            return self._notice_with_prefix(
                mutations, "system", (_media_block("audio", message.audio_url),)
            )
        if isinstance(message, VideoURLPart):
            return self._notice_with_prefix(
                mutations, "system", (_media_block("video", message.video_url),)
            )
        if isinstance(message, ToolCall):
            mutations.append(self._tool_call(message))
            return NormalizedWireMessage(tuple(mutations))
        if isinstance(message, ToolCallPart):
            mutations.append(self._tool_call_part(message))
            return NormalizedWireMessage(tuple(mutations))
        if isinstance(message, ToolResult):
            mutations.append(self._tool_result(message))
            return NormalizedWireMessage(tuple(mutations))
        if isinstance(message, ApprovalRequest):
            metadata = (
                ("Request ID", message.id, "secondary"),
                ("Tool call ID", message.tool_call_id, "secondary"),
                ("Source kind", message.source_kind, "secondary"),
                ("Source ID", message.source_id, "secondary"),
                ("Subagent type", message.subagent_type, "secondary"),
                ("Agent ID", message.agent_id, "secondary"),
                ("Source detail", message.source_description, "detail"),
            )
            blocks: list[ContentBlock] = [
                TextBlock(
                    localized(
                        "{sender} requests: {action}", sender=message.sender, action=message.action
                    )
                ),
                TextBlock(literal(message.description)),
                FieldListBlock(
                    tuple(
                        _field(name, value, role=role)
                        for name, value, role in metadata
                        if value not in (None, "")
                    )
                ),
                *display_blocks(message.display),
            ]
            return self._notice_with_prefix(mutations, "approval", tuple(blocks))
        if isinstance(message, ApprovalResponse):
            fields = [_field("Request ID", message.request_id, role="secondary")]
            if message.feedback:
                fields.append(_field("Feedback", message.feedback))
            return self._notice_with_prefix(
                mutations,
                "error" if message.response == "reject" else "system",
                (
                    TextBlock(localized("Approval {response}", response=message.response)),
                    FieldListBlock(tuple(fields)),
                ),
            )
        if isinstance(message, SubagentEvent):
            nested = self._subagent(message)
            return NormalizedWireMessage(tuple(mutations) + nested.mutations, nested.status)
        if isinstance(message, StepBegin):
            return self._notice_with_prefix(
                mutations, "system", (TextBlock(localized("Step {number}", number=message.n)),)
            )
        if isinstance(message, StepInterrupted):
            return self._notice_with_prefix(
                mutations, "error", (TextBlock(localized("Current step was interrupted")),)
            )
        if isinstance(message, CompactionBegin):
            values = (
                ("Trigger", message.trigger, "primary", "plain"),
                ("Shadowed tokens", message.shadowed_tokens, "secondary", "count"),
                ("Compaction ID", message.compaction_id, "secondary", "plain"),
            )
            return self._field_notice(
                mutations,
                "system",
                localized("Compacting context"),
                tuple(
                    _field(name, value, role=role, hint=hint)
                    for name, value, role, hint in values
                    if value not in (None, "")
                ),
            )
        if isinstance(message, CompactionEnd):
            values = (
                ("Trigger", message.trigger, "primary", "plain"),
                ("Shadowed tokens", message.shadowed_tokens, "secondary", "count"),
                ("Estimated tokens", message.estimated_token_count, "secondary", "count"),
                ("Compaction ID", message.compaction_id, "secondary", "plain"),
                ("Error", message.error, "detail", "plain"),
            )
            return self._field_notice(
                mutations,
                "error" if message.error else "system",
                localized("Context compaction failed" if message.error else "Context compacted"),
                tuple(
                    _field(name, value, role=role, hint=hint)
                    for name, value, role, hint in values
                    if value not in (None, "")
                ),
            )
        if isinstance(message, StatusUpdate):
            for name in (
                "context_usage",
                "context_tokens",
                "max_context_tokens",
                "token_usage",
                "message_id",
                "mcp_status",
            ):
                value = getattr(message, name, None)
                if value is not None:
                    self._status_values[name] = value
            return NormalizedWireMessage(
                tuple(mutations), status_values(SimpleNamespace(**self._status_values))
            )
        if isinstance(message, TurnEnd):
            for nested in self._subagents.values():
                mutations.extend(nested.finish())
            return NormalizedWireMessage(tuple(mutations))

        named = self._named_event(message, mutations)
        return named or NormalizedWireMessage(tuple(mutations))

    def finish(self) -> tuple[TranscriptMutation, ...]:
        mutations = list(self._finish_stream())
        for nested in self._subagents.values():
            mutations.extend(nested.finish())
        return tuple(mutations)

    def _stream(
        self,
        kind: str,
        fragment: str,
        *,
        format: str = "plain",
        tone: str = "primary",
    ) -> NormalizedWireMessage:
        if not fragment:
            return NormalizedWireMessage()
        mutations: list[TranscriptMutation] = []
        if self._stream_kind != kind or self._stream_key is None:
            mutations.extend(self._finish_stream())
            self._stream_kind = kind
            self._stream_key = self._new_key(kind)
            mutations.append(
                StartEntry(
                    TextEntry(
                        key=self._stream_key,
                        kind=kind,  # type: ignore[arg-type]
                        blocks=(
                            TextBlock(
                                literal(fragment),
                                format=format,  # type: ignore[arg-type]
                                tone=tone,  # type: ignore[arg-type]
                            ),
                        ),
                        source=self._source,
                        complete=False,
                    )
                )
            )
        else:
            mutations.append(
                AppendText(
                    key=self._stream_key,
                    kind=kind,  # type: ignore[arg-type]
                    source=self._source,
                    block=0,
                    fragment=fragment,
                    format=format,  # type: ignore[arg-type]
                    tone=tone,  # type: ignore[arg-type]
                )
            )
        return NormalizedWireMessage(tuple(mutations))

    def _finish_stream(self) -> tuple[TranscriptMutation, ...]:
        if self._stream_key is None:
            return ()
        mutation = FinishEntry(self._stream_key)
        self._stream_key = None
        self._stream_kind = None
        return (mutation,)

    def _new_key(self, prefix: str) -> str:
        return f"{self._scope}:{prefix}:{next(self._keys)}"

    def _activity_key(self, call_id: str) -> str:
        return f"{self._scope}:activity:{call_id}"

    def _pending_activity(self, call_id: str, name: str = "Tool") -> PendingActivity:
        pending = self._pending.get(call_id)
        if pending is None:
            pending = PendingActivity(
                call_id=call_id,
                identity=ToolIdentity(name, tool_family(name)),
            )
            self._pending[call_id] = pending
            if len(self._pending) > _MAX_PENDING_ACTIVITIES:
                oldest = next(iter(self._pending))
                if oldest != call_id:
                    self._pending.pop(oldest, None)
        return pending

    def _activity_mutation(self, pending: PendingActivity) -> ReplaceEntry:
        return ReplaceEntry(
            ActivityEntry(
                key=self._activity_key(pending.call_id),
                activity=pending.snapshot(),
                source=self._source,
                complete=pending.result is not None,
            )
        )

    def _tool_call(self, message: ToolCall) -> ReplaceEntry:
        pending = self._pending_activity(message.id, message.function.name)
        pending.started = True
        pending.identity = ToolIdentity(message.function.name, tool_family(message.function.name))
        pending.raw_arguments = message.function.arguments or ""
        pending.extras = message.extras
        parsed, parse_state = _argument_object(message.function.arguments)
        pending.call = build_tool_call_content(
            message.function.name,
            parsed,
            message.function.arguments,
            parse_state=parse_state,
            extras=message.extras,
        )
        self._active_call_id = message.id
        return self._activity_mutation(pending)

    def _tool_call_part(self, message: ToolCallPart) -> ReplaceEntry:
        call_id = next(
            (
                call_id
                for call_id, pending in self._pending.items()
                if pending.started
                and (pending.call is None or pending.call.parse_state in {"missing", "partial"})
            ),
            self._active_call_id,
        )
        if call_id is None:
            call_id = self._new_key("orphan-tool-part")
            self._active_call_id = call_id
        pending = self._pending_activity(call_id)
        pending.started = True
        pending.raw_arguments += message.arguments_part or ""
        parsed, parse_state = _argument_object(pending.raw_arguments)
        pending.call = build_tool_call_content(
            pending.identity.wire_name,
            parsed,
            pending.raw_arguments,
            parse_state=parse_state,
            extras=pending.extras,
        )
        return self._activity_mutation(pending)

    def _tool_result(self, message: ToolResult) -> ReplaceEntry:
        call_id = message.tool_call_id
        pending = self._pending_activity(call_id)
        result = message.return_value
        pending.result = build_tool_result_content(
            is_error=result.is_error,
            message=result.message or "",
            display_blocks=display_blocks(result.display),
            output_blocks=content_blocks(result.output),
            extras=result.extras,
        )
        if self._active_call_id == call_id:
            self._active_call_id = None
        return self._activity_mutation(pending)

    def _notice_with_prefix(
        self,
        prefix: Sequence[TranscriptMutation],
        kind: str,
        blocks: tuple[ContentBlock, ...],
    ) -> NormalizedWireMessage:
        mutation = StartEntry(
            NoticeEntry(
                key=self._new_key(kind),
                kind=kind,  # type: ignore[arg-type]
                blocks=blocks,
                source=self._source,
            )
        )
        return NormalizedWireMessage(tuple(prefix) + (mutation,))

    def _field_notice(
        self,
        prefix: Sequence[TranscriptMutation],
        kind: str,
        title: object,
        fields: tuple[ActivityField, ...],
    ) -> NormalizedWireMessage:
        blocks: list[ContentBlock] = [TextBlock(title)]  # type: ignore[arg-type]
        if fields:
            blocks.append(FieldListBlock(fields))
        return self._notice_with_prefix(prefix, kind, tuple(blocks))

    def _subagent(self, message: SubagentEvent) -> NormalizedWireMessage:
        identity = message.agent_id or message.parent_tool_call_id or "unknown"
        source = EntrySource(
            kind="subagent",
            identifier=identity,
            label=localized(
                "Subagent {kind} · {id}", kind=message.subagent_type or "agent", id=identity
            ),
        )
        nested = self._subagents.get(identity)
        if nested is None:
            nested = WireNormalizer(scope=f"{self._scope}/{identity}", source=source)
            self._subagents[identity] = nested
        return nested.normalize(message.event)

    def _named_event(
        self, message: object, prefix: Sequence[TranscriptMutation]
    ) -> NormalizedWireMessage | None:
        name = type(message).__name__
        if name in _OBSERVABILITY_EVENTS:
            return NormalizedWireMessage(tuple(prefix))
        if name == "StepRetry":
            return self._field_notice(
                prefix,
                "error",
                localized("Retrying step {number}", number=getattr(message, "n", "?")),
                tuple(
                    _field(label, value, role=role, hint=hint)
                    for label, value, role, hint in (
                        (
                            "Next attempt",
                            getattr(message, "next_attempt", None),
                            "secondary",
                            "count",
                        ),
                        (
                            "Maximum attempts",
                            getattr(message, "max_attempts", None),
                            "secondary",
                            "count",
                        ),
                        ("Wait", f"{getattr(message, 'wait_s', 0):g}s", "secondary", "plain"),
                        ("Error", getattr(message, "error_type", None), "primary", "plain"),
                        (
                            "HTTP status",
                            getattr(message, "status_code", None),
                            "secondary",
                            "count",
                        ),
                    )
                    if value not in (None, "")
                ),
            )
        if name in {"HookTriggered", "HookResolved"}:
            action = getattr(message, "action", "allow")
            fields = tuple(
                _field(label, value, role=role)
                for label, value, role in (
                    ("Event", getattr(message, "event", None), "primary"),
                    ("Target", getattr(message, "target", None), "secondary"),
                    ("Hooks", getattr(message, "hook_count", None), "secondary"),
                    ("Action", action if name == "HookResolved" else None, "primary"),
                    ("Reason", getattr(message, "reason", None), "detail"),
                    ("Duration", f"{getattr(message, 'duration_ms', 0)}ms", "secondary"),
                )
                if value not in (None, "")
            )
            return self._field_notice(
                prefix,
                "error" if name == "HookResolved" and action == "block" else "system",
                localized("Hook resolved" if name == "HookResolved" else "Hook triggered"),
                fields,
            )
        if name in {"MCPLoadingBegin", "MCPLoadingEnd"}:
            title = "Loading MCP servers" if name.endswith("Begin") else "MCP loading completed"
            return self._notice_with_prefix(prefix, "system", (TextBlock(localized(title)),))
        if name == "Notification":
            severity = str(getattr(message, "severity", "info"))
            return self._field_notice(
                prefix,
                "error" if severity == "error" else "system",
                localized("Notification · {title}", title=getattr(message, "title", "")),
                tuple(
                    _field(label, value, role=role, hint=hint)
                    for label, value, role, hint in (
                        ("Body", getattr(message, "body", None), "primary", "multiline"),
                        ("Severity", severity, "secondary", "plain"),
                        ("Category", getattr(message, "category", None), "secondary", "plain"),
                        ("Type", getattr(message, "type", None), "secondary", "plain"),
                        ("Source", getattr(message, "source_id", None), "secondary", "plain"),
                        ("Payload", getattr(message, "payload", {}), "detail", "json"),
                    )
                    if value not in (None, "", {})
                ),
            )
        if name == "BtwBegin":
            return self._notice_with_prefix(
                prefix,
                "system",
                (
                    TextBlock(localized("Side question")),
                    TextBlock(literal(getattr(message, "question", ""))),
                ),
            )
        if name == "BtwEnd":
            error = getattr(message, "error", None)
            text = error or getattr(message, "response", None) or "(no response)"
            return self._notice_with_prefix(
                prefix,
                "error" if error else "system",
                (TextBlock(localized("Side answer")), TextBlock(literal(text))),
            )
        if name == "QuestionRequest":
            questions = tuple(
                QuestionItem(
                    question=str(getattr(question, "question", "")),
                    header=str(getattr(question, "header", "")),
                    body=str(getattr(question, "body", "")),
                    options=tuple(
                        QuestionOption(
                            label=str(getattr(option, "label", option)),
                            description=str(getattr(option, "description", "")),
                        )
                        for option in getattr(question, "options", ())
                    ),
                    multi_select=bool(getattr(question, "multi_select", False)),
                )
                for question in getattr(message, "questions", ())
            )
            return self._notice_with_prefix(
                prefix,
                "approval",
                (
                    TextBlock(localized("Question request")),
                    FieldListBlock(
                        tuple(
                            _field(label, value, role="secondary")
                            for label, value in (
                                ("Request ID", getattr(message, "id", "")),
                                ("Tool call ID", getattr(message, "tool_call_id", "")),
                            )
                            if value
                        )
                    ),
                    QuestionBlock(questions),
                ),
            )
        if name in {"HookRequest", "ToolCallRequest"}:
            data = _model_data(message)
            label = localized("Hook request" if name == "HookRequest" else "Tool call request")
            return self._notice_with_prefix(
                prefix,
                "approval",
                (RawBlock(_json_text(data), label=label),),
            )
        data = _model_data(message)
        if data is message:
            return None
        return self._notice_with_prefix(
            prefix,
            "system",
            (RawBlock(_json_text(data), label=literal(name)),),
        )


def status_values(status: object) -> StatusValues:
    """Extract context/token/MCP figures without creating UI copy."""

    usage = getattr(status, "context_usage", None)
    percent: float | None = None
    if usage is not None:
        percent = usage * 100 if 0 <= usage <= 1 else usage

    token_usage = getattr(status, "token_usage", None)
    tokens: TokenCounts | None = None
    if token_usage is not None:
        new_input = getattr(token_usage, "input_other", 0)
        cache_read = getattr(token_usage, "input_cache_read", 0)
        cache_write = getattr(token_usage, "input_cache_creation", 0)
        tokens = TokenCounts(
            total_input=new_input + cache_read + cache_write,
            new_input=new_input,
            cache_read=cache_read,
            cache_write=cache_write,
            output=getattr(token_usage, "output", 0),
        )

    wire_mcp = getattr(status, "mcp_status", None)
    mcp: McpCounts | None = None
    if wire_mcp is not None:
        mcp = McpCounts(
            connected=getattr(wire_mcp, "connected", 0),
            total=getattr(wire_mcp, "total", 0),
            tools=getattr(wire_mcp, "tools", 0),
            loading=bool(getattr(wire_mcp, "loading", False)),
            servers=tuple(
                McpServer(
                    name=str(getattr(server, "name", "?")),
                    state=str(getattr(server, "status", "?")),
                )
                for server in getattr(wire_mcp, "servers", ()) or ()
            ),
        )

    return StatusValues(
        context_tokens=getattr(status, "context_tokens", None),
        max_context_tokens=getattr(status, "max_context_tokens", None),
        usage_percent=percent,
        tokens=tokens,
        mcp=mcp,
        yolo_enabled=bool(getattr(status, "yolo_enabled", False)),
        afk_enabled=bool(getattr(status, "afk_enabled", False)),
    )
