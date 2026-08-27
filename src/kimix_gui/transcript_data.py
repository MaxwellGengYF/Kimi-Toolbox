"""Immutable semantic data carried through the transcript pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass, replace
from typing import Literal

type TextFormat = Literal["plain", "markdown", "code"]
type TextTone = Literal["primary", "context", "muted", "error"]
type FieldRole = Literal["primary", "secondary", "detail"]
type FieldHint = Literal[
    "plain",
    "path",
    "command",
    "code",
    "multiline",
    "json",
    "count",
    "url",
]
type ParseState = Literal["complete", "partial", "invalid", "missing"]
type ToolStatus = Literal["pending", "ok", "error"]
type TextEntryKind = Literal["user", "assistant", "thinking"]
type NoticeKind = Literal["system", "error", "approval"]
type SourceKind = Literal["root", "subagent", "app", "external"]
type MediaKind = Literal["image", "audio", "video"]

type LocalizedArg = str | int | float | bool | None
type Translator = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class LiteralText:
    """Text supplied by a model, tool, SDK, or persisted wire record."""

    text: str


@dataclass(frozen=True, slots=True)
class LocalizedText:
    """Application copy stored as an English msgid plus named arguments."""

    msgid: str
    arguments: tuple[tuple[str, LocalizedArg], ...] = ()


type TextRef = LiteralText | LocalizedText


def literal(value: object) -> LiteralText:
    """Wrap external text without changing or translating it."""

    return LiteralText(str(value))


def localized(msgid: str, **arguments: LocalizedArg) -> LocalizedText:
    """Build application copy while retaining named formatting arguments."""

    return LocalizedText(msgid, tuple(arguments.items()))


def resolve_text(value: TextRef, translate: Translator | None = None) -> str:
    """Resolve a text reference at a presentation boundary."""

    if isinstance(value, LiteralText):
        return value.text
    template = translate(value.msgid) if translate is not None else value.msgid
    return template.format(**dict(value.arguments))


@dataclass(frozen=True, slots=True)
class EntrySource:
    """Where a transcript entry originated, without encoding it into its body."""

    kind: SourceKind = "root"
    identifier: str = ""
    label: TextRef | None = None


ROOT_SOURCE = EntrySource()
APP_SOURCE = EntrySource(kind="app")


@dataclass(frozen=True, slots=True)
class ActivityField:
    """One decoded tool field with an explicit presentation role and hint."""

    name: str
    value: TextRef
    role: FieldRole = "detail"
    hint: FieldHint = "plain"


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: TextRef
    format: TextFormat = "plain"
    tone: TextTone = "primary"


@dataclass(frozen=True, slots=True)
class MediaBlock:
    kind: MediaKind
    url: str
    media_id: str | None = None


@dataclass(frozen=True, slots=True)
class FieldListBlock:
    fields: tuple[ActivityField, ...]
    tone: TextTone = "context"


@dataclass(frozen=True, slots=True)
class DiffBlock:
    path: str
    old_start: int
    new_start: int
    old_text: str
    new_text: str
    is_summary: bool = False


@dataclass(frozen=True, slots=True)
class TodoItem:
    title: str
    status: str = "pending"
    depth: int = 0
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TodoBlock:
    items: tuple[TodoItem, ...]


@dataclass(frozen=True, slots=True)
class QuestionOption:
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class QuestionItem:
    question: str
    header: str = ""
    body: str = ""
    options: tuple[QuestionOption, ...] = ()
    multi_select: bool = False


@dataclass(frozen=True, slots=True)
class QuestionBlock:
    questions: tuple[QuestionItem, ...]


@dataclass(frozen=True, slots=True)
class RawBlock:
    """Lossless fallback for unknown or not-yet-decodable content."""

    payload: str
    label: TextRef | None = None
    parse_state: ParseState = "complete"


type ContentBlock = (
    TextBlock | MediaBlock | FieldListBlock | DiffBlock | TodoBlock | QuestionBlock | RawBlock
)


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    wire_name: str
    family: str


@dataclass(frozen=True, slots=True)
class ToolCallContent:
    summary_parts: tuple[TextRef, ...] = ()
    fields: tuple[ActivityField, ...] = ()
    details: tuple[ContentBlock, ...] = ()
    raw_arguments: str | None = None
    parse_state: ParseState = "complete"


@dataclass(frozen=True, slots=True)
class ToolResultContent:
    status: ToolStatus
    summary_parts: tuple[TextRef, ...] = ()
    blocks: tuple[ContentBlock, ...] = ()
    raw_fallback: str | None = None


@dataclass(frozen=True, slots=True)
class ToolActivity:
    call_id: str
    identity: ToolIdentity
    call: ToolCallContent | None = None
    result: ToolResultContent | None = None


@dataclass(frozen=True, slots=True)
class TextEntry:
    key: str
    kind: TextEntryKind
    blocks: tuple[ContentBlock, ...]
    source: EntrySource = ROOT_SOURCE
    complete: bool = True


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    key: str
    activity: ToolActivity
    source: EntrySource = ROOT_SOURCE
    complete: bool = False


@dataclass(frozen=True, slots=True)
class NoticeEntry:
    key: str
    kind: NoticeKind
    blocks: tuple[ContentBlock, ...]
    source: EntrySource = ROOT_SOURCE
    complete: bool = True


type TranscriptEntry = TextEntry | ActivityEntry | NoticeEntry


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    entry: TranscriptEntry
    turn: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class StartEntry:
    entry: TranscriptEntry


@dataclass(frozen=True, slots=True)
class AppendText:
    """Append one fragment, including enough information to recreate a trimmed row."""

    key: str
    kind: TextEntryKind
    source: EntrySource
    block: int
    fragment: str
    format: TextFormat = "plain"
    tone: TextTone = "primary"


@dataclass(frozen=True, slots=True)
class ReplaceEntry:
    """Replace an entry by key; a missing target is inserted at the tail."""

    entry: TranscriptEntry


@dataclass(frozen=True, slots=True)
class FinishEntry:
    key: str


@dataclass(frozen=True, slots=True)
class ClearTranscript:
    pass


type TranscriptMutation = StartEntry | AppendText | ReplaceEntry | FinishEntry | ClearTranscript


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    epoch: int
    mutation: TranscriptMutation


@dataclass(frozen=True, slots=True)
class MutationEffect:
    """The structural effect of applying one mutation to an ordered entry list."""

    action: Literal["insert", "replace", "finish", "clear", "unchanged"]
    index: int | None = None
    entry: TranscriptEntry | None = None
    previous: TranscriptEntry | None = None


def entry_key(entry: TranscriptEntry) -> str:
    return entry.key


def entry_kind(entry: TranscriptEntry) -> str:
    if isinstance(entry, TextEntry | NoticeEntry):
        return entry.kind
    return "tool"


def is_dialogue_entry(entry: TranscriptEntry) -> bool:
    return isinstance(entry, TextEntry) and entry.kind in {"user", "assistant"}


def _finished_entry(entry: TranscriptEntry) -> TranscriptEntry:
    if entry.complete:
        return entry
    return replace(entry, complete=True)


class TranscriptReducer:
    """Apply typed mutations once for both historical and live consumers."""

    def __init__(self, entries: Iterable[TranscriptEntry] = ()) -> None:
        self.entries: list[TranscriptEntry] = list(entries)
        self._positions: dict[str, int] = {
            entry.key: index for index, entry in enumerate(self.entries)
        }

    def apply(self, mutation: TranscriptMutation) -> MutationEffect:
        if isinstance(mutation, ClearTranscript):
            changed = bool(self.entries)
            self.entries.clear()
            self._positions.clear()
            return MutationEffect("clear" if changed else "unchanged")
        if isinstance(mutation, AppendText):
            return self._append_text(mutation)
        if isinstance(mutation, FinishEntry):
            position = self._positions.get(mutation.key)
            if position is None:
                return MutationEffect("unchanged")
            previous = self.entries[position]
            current = _finished_entry(previous)
            if current is previous:
                return MutationEffect("unchanged", position, previous, previous)
            self.entries[position] = current
            return MutationEffect("finish", position, current, previous)
        entry = mutation.entry
        position = self._positions.get(entry.key)
        if position is None:
            position = len(self.entries)
            self.entries.append(entry)
            self._positions[entry.key] = position
            return MutationEffect("insert", position, entry)
        previous = self.entries[position]
        if previous == entry:
            return MutationEffect("unchanged", position, entry, previous)
        self.entries[position] = entry
        return MutationEffect("replace", position, entry, previous)

    def replace_all(self, entries: Sequence[TranscriptEntry]) -> None:
        self.entries[:] = entries
        self._reindex()

    def remove_prefix(self, count: int) -> tuple[TranscriptEntry, ...]:
        count = max(0, min(count, len(self.entries)))
        removed = tuple(self.entries[:count])
        del self.entries[:count]
        self._reindex()
        return removed

    def take_all(self) -> tuple[TranscriptEntry, ...]:
        entries = tuple(self.entries)
        self.entries.clear()
        self._positions.clear()
        return entries

    def _append_text(self, mutation: AppendText) -> MutationEffect:
        position = self._positions.get(mutation.key)
        if position is None:
            blocks: list[ContentBlock] = [
                TextBlock(literal(""), format=mutation.format, tone=mutation.tone)
                for _ in range(mutation.block + 1)
            ]
            blocks[mutation.block] = TextBlock(
                literal(mutation.fragment),
                format=mutation.format,
                tone=mutation.tone,
            )
            entry = TextEntry(
                key=mutation.key,
                kind=mutation.kind,
                blocks=tuple(blocks),
                source=mutation.source,
                complete=False,
            )
            position = len(self.entries)
            self.entries.append(entry)
            self._positions[entry.key] = position
            return MutationEffect("insert", position, entry)

        previous = self.entries[position]
        if not isinstance(previous, TextEntry):
            replacement = TextEntry(
                key=mutation.key,
                kind=mutation.kind,
                blocks=(
                    TextBlock(
                        literal(mutation.fragment),
                        format=mutation.format,
                        tone=mutation.tone,
                    ),
                ),
                source=mutation.source,
                complete=False,
            )
            self.entries[position] = replacement
            return MutationEffect("replace", position, replacement, previous)

        blocks = list(previous.blocks)
        while len(blocks) <= mutation.block:
            blocks.append(TextBlock(literal(""), format=mutation.format, tone=mutation.tone))
        target = blocks[mutation.block]
        if isinstance(target, TextBlock):
            current_text = resolve_text(target.text)
            blocks[mutation.block] = TextBlock(
                literal(current_text + mutation.fragment),
                format=target.format,
                tone=target.tone,
            )
        else:
            blocks.insert(
                mutation.block,
                TextBlock(
                    literal(mutation.fragment),
                    format=mutation.format,
                    tone=mutation.tone,
                ),
            )
        current = replace(
            previous,
            kind=mutation.kind,
            blocks=tuple(blocks),
            source=mutation.source,
            complete=False,
        )
        self.entries[position] = current
        return MutationEffect("replace", position, current, previous)

    def _reindex(self) -> None:
        self._positions = {entry.key: index for index, entry in enumerate(self.entries)}


def _todo_marker(status: str) -> str:
    value = status.lower()
    if value in {"done", "completed"}:
        return "[x]"
    if value in {"in_progress", "in-progress", "doing"}:
        return "[>]"
    return "[ ]"


def block_text(block: ContentBlock, translate: Translator | None = None) -> str:
    """Lossless plain-text representation of one semantic content block."""

    if isinstance(block, TextBlock):
        return resolve_text(block.text, translate)
    if isinstance(block, MediaBlock):
        label = block.kind.title()
        suffix = f"\nID: {block.media_id}" if block.media_id else ""
        return f"[{label}]\nURL: {block.url}{suffix}"
    if isinstance(block, FieldListBlock):
        lines: list[str] = []
        for field in block.fields:
            value = resolve_text(field.value, translate)
            if field.hint in {"code", "multiline", "json"} or "\n" in value:
                lines.extend((f"{field.name}:", value))
            else:
                lines.append(f"{field.name}: {value}")
        return "\n".join(lines)
    if isinstance(block, DiffBlock):
        summary = " · summary" if block.is_summary else ""
        lines = [
            f"Diff: {block.path}{summary}",
            f"@@ -{block.old_start} +{block.new_start} @@",
        ]
        lines.extend(f"- {line}" for line in block.old_text.splitlines())
        lines.extend(f"+ {line}" for line in block.new_text.splitlines())
        return "\n".join(lines)
    if isinstance(block, TodoBlock):
        return "\n".join(
            f"{'  ' * max(0, item.depth)}{_todo_marker(item.status)} {item.title}"
            + (f" — {item.notes}" if item.notes else "")
            for item in block.items
        )
    if isinstance(block, QuestionBlock):
        lines: list[str] = []
        for index, question in enumerate(block.questions, start=1):
            header = f" [{question.header}]" if question.header else ""
            lines.append(f"{index}.{header} {question.question}")
            if question.body:
                lines.append(question.body)
            for option in question.options:
                detail = f" — {option.description}" if option.description else ""
                lines.append(f"  - {option.label}{detail}")
            if question.multi_select:
                lines.append("  Multiple selections allowed")
        return "\n".join(lines)
    label = resolve_text(block.label, translate) if block.label is not None else ""
    return f"{label}:\n{block.payload}" if label else block.payload


def blocks_text(
    blocks: Sequence[ContentBlock],
    translate: Translator | None = None,
    *,
    separator: str = "\n\n",
) -> str:
    return separator.join(text for block in blocks if (text := block_text(block, translate)))


def summary_text(parts: Sequence[TextRef], translate: Translator | None = None) -> str:
    return " · ".join(
        text for part in parts if (text := " ".join(resolve_text(part, translate).split()))
    )


def tool_call_text(call: ToolCallContent, translate: Translator | None = None) -> str:
    """Format a decoded invocation without reading meaning back from its text."""

    field_block = FieldListBlock(call.fields)
    parts = [block_text(field_block, translate)] if call.fields else []
    parts.extend(block_text(block, translate) for block in call.details)
    if not parts and call.raw_arguments:
        parts.append(call.raw_arguments)
    return "\n\n".join(part for part in parts if part)


def tool_result_text(result: ToolResultContent, translate: Translator | None = None) -> str:
    parts = [block_text(block, translate) for block in result.blocks]
    if result.raw_fallback and result.raw_fallback not in parts:
        parts.append(result.raw_fallback)
    return "\n\n".join(part for part in parts if part)


def entry_body_text(entry: TranscriptEntry, translate: Translator | None = None) -> str:
    if isinstance(entry, TextEntry | NoticeEntry):
        return blocks_text(entry.blocks, translate)
    call = tool_call_text(entry.activity.call, translate) if entry.activity.call else ""
    result = tool_result_text(entry.activity.result, translate) if entry.activity.result else ""
    return "\n\n".join(filter(None, (call, result)))


def entry_copy_text(entry: TranscriptEntry, translate: Translator | None = None) -> str:
    """Clipboard text built directly from the semantic entry."""

    source = resolve_text(entry.source.label, translate) if entry.source.label is not None else ""
    if not isinstance(entry, ActivityEntry):
        return "\n\n".join(filter(None, (source, entry_body_text(entry, translate))))
    activity = entry.activity
    name = activity.identity.wire_name or "Tool"
    call_text = ""
    if activity.call is not None:
        summary = summary_text(activity.call.summary_parts, translate)
        headline = f"{name}  {summary}" if summary else name
        details = tool_call_text(activity.call, translate)
        call_text = "\n".join(filter(None, (headline, details)))
    result_text = ""
    if activity.result is not None:
        summary = summary_text(activity.result.summary_parts, translate)
        headline = f"{name}  {summary}" if summary else name
        details = tool_result_text(activity.result, translate)
        result_text = "\n".join(filter(None, (headline, details)))
    return "\n\n".join(filter(None, (source, call_text, result_text)))


def _text_ref_cost(value: TextRef) -> int:
    if isinstance(value, LiteralText):
        return len(value.text)
    return len(value.msgid) + sum(
        len(name) + len(str(argument)) for name, argument in value.arguments
    )


def block_cost(block: ContentBlock) -> int:
    if isinstance(block, TextBlock):
        return _text_ref_cost(block.text)
    if isinstance(block, MediaBlock):
        return len(block.url) + len(block.media_id or "") + len(block.kind)
    if isinstance(block, FieldListBlock):
        return sum(len(field.name) + _text_ref_cost(field.value) for field in block.fields)
    if isinstance(block, DiffBlock):
        return len(block.path) + len(block.old_text) + len(block.new_text) + 32
    if isinstance(block, TodoBlock):
        return sum(len(item.title) + len(item.status) + len(item.notes) + 8 for item in block.items)
    if isinstance(block, QuestionBlock):
        return sum(
            len(question.question)
            + len(question.header)
            + len(question.body)
            + sum(len(option.label) + len(option.description) for option in question.options)
            for question in block.questions
        )
    return len(block.payload) + (_text_ref_cost(block.label) if block.label is not None else 0)


def entry_cost(entry: TranscriptEntry) -> int:
    """Recursive memory cost used by every bounded transcript cache."""

    source = len(entry.source.identifier)
    if entry.source.label is not None:
        source += _text_ref_cost(entry.source.label)
    if isinstance(entry, TextEntry | NoticeEntry):
        return source + len(entry.key) + sum(block_cost(block) for block in entry.blocks)
    activity = entry.activity
    total = source + len(entry.key) + len(activity.call_id)
    total += len(activity.identity.wire_name) + len(activity.identity.family)
    if activity.call is not None:
        total += sum(_text_ref_cost(part) for part in activity.call.summary_parts)
        total += sum(
            len(field.name) + _text_ref_cost(field.value) for field in activity.call.fields
        )
        total += sum(block_cost(block) for block in activity.call.details)
        total += len(activity.call.raw_arguments or "")
    if activity.result is not None:
        total += sum(_text_ref_cost(part) for part in activity.result.summary_parts)
        total += sum(block_cost(block) for block in activity.result.blocks)
        total += len(activity.result.raw_fallback or "")
    return total


def history_entries_cost(entries: Sequence[HistoryEntry]) -> int:
    return sum(entry_cost(item.entry) for item in entries)


def rebuild_positions(
    entries: Sequence[TranscriptEntry], positions: MutableMapping[str, int]
) -> None:
    """Small shared helper for consumers that retain their own row containers."""

    positions.clear()
    positions.update((entry.key, index) for index, entry in enumerate(entries))


def apply_mutation(
    entries: MutableSequence[TranscriptEntry], mutation: TranscriptMutation
) -> MutationEffect:
    """Apply the canonical reducer semantics to an arbitrary mutable sequence."""

    reducer = TranscriptReducer(entries)
    effect = reducer.apply(mutation)
    entries[:] = reducer.entries
    return effect
