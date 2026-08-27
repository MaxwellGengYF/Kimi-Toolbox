"""Framework-neutral transcript presentation composed directly from semantic entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from unicodedata import east_asian_width

from kimix_gui.tool_display import KNOWN_TOOL_FAMILIES, tool_label
from kimix_gui.transcript_data import (
    ActivityEntry,
    ContentBlock,
    DiffBlock,
    FieldListBlock,
    MediaBlock,
    NoticeEntry,
    QuestionBlock,
    RawBlock,
    TextBlock,
    TextEntry,
    TodoBlock,
    TranscriptEntry,
    Translator,
    block_text,
    entry_kind,
    is_dialogue_entry,
    resolve_text,
    summary_text,
)

STATUS_PENDING = "pending"
STATUS_OK = "ok"
STATUS_ERROR = "error"

LABELS: dict[str, str] = {
    "user": "You",
    "assistant": "AI",
    "thinking": "Think",
    "tool": "Tool",
    "approval": "Approval",
    "system": "System",
    "error": "Error",
}

BAR_COLOR_NAME: dict[str, str] = {
    "user": "cyan",
    "assistant": "green",
    "thinking": "muted",
    "tool": "muted",
    "approval": "muted",
    "system": "muted",
    "error": "red",
}

FAMILY_BAR_NAME: dict[str, str] = {
    "read": "cyan",
    "grep": "cyan",
    "glob": "cyan",
    "write": "yellow",
    "edit": "yellow",
    "shell": "green",
    "python": "green",
    "todo": "yellow",
    "search": "blue",
    "fetch": "blue",
    "agent": "magenta",
}

NO_DETAILS = "(no details)"

_OUTPUT_FIRST_FAMILIES = frozenset({"read", "grep", "glob", "shell", "search", "fetch", "agent"})


def cell_len(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if east_asian_width(char) in {"F", "W"} else 1
    return width


def set_cell_size(text: str, width: int) -> str:
    if width <= 0:
        return ""
    used = 0
    chars: list[str] = []
    for char in text:
        size = 2 if east_asian_width(char) in {"F", "W"} else 1
        if used + size > width:
            break
        chars.append(char)
        used += size
    return "".join(chars)


def fit_summary(summary: str, width: int) -> str:
    if cell_len(summary) <= width:
        return summary
    if width <= 3:
        return "." * max(0, width)
    return f"{set_cell_size(summary, width - 3)}..."


def default_expanded(entry: TranscriptEntry) -> bool:
    return isinstance(entry, TextEntry) and entry.kind == "thinking"


def is_compact_entry(entry: TranscriptEntry, *, expanded: bool = False) -> bool:
    return not is_dialogue_entry(entry) and not expanded


def record_label(entry: TranscriptEntry) -> str:
    if isinstance(entry, ActivityEntry):
        return tool_label(entry.activity.identity.wire_name)
    return LABELS.get(entry_kind(entry), entry_kind(entry).title())


def activity_status(entry: TranscriptEntry) -> str:
    if not isinstance(entry, ActivityEntry):
        return ""
    result = entry.activity.result
    return result.status if result is not None else STATUS_PENDING


def bar_color_name(entry: TranscriptEntry) -> str:
    if isinstance(entry, ActivityEntry):
        status = activity_status(entry)
        if status == STATUS_ERROR:
            return "red"
        return FAMILY_BAR_NAME.get(entry.activity.identity.family, "muted")
    return BAR_COLOR_NAME.get(entry_kind(entry), "muted")


@dataclass(frozen=True, slots=True)
class HeaderRun:
    text: str
    tone: Literal["label", "summary"]
    emphasis: bool = False


@dataclass(frozen=True, slots=True)
class BodySection:
    text: str
    format: Literal["plain", "markdown", "code"] = "plain"
    tone: Literal["primary", "context", "muted", "error"] = "primary"
    spacing: Literal["none", "paragraph"] = "none"


@dataclass(frozen=True, slots=True)
class RecordLayout:
    """Derived terminal strings are outputs; semantic input remains the entry AST."""

    header_runs: tuple[HeaderRun, ...]
    body_sections: tuple[BodySection, ...]
    compact: bool
    label: str
    bar_color: str
    italic_body: bool
    status: str = ""

    @property
    def header(self) -> str:
        return " ".join(run.text for run in self.header_runs if run.text)

    @property
    def summary(self) -> str:
        return " ".join(run.text for run in self.header_runs if run.tone == "summary")

    @property
    def body(self) -> str:
        parts: list[str] = []
        for section in self.body_sections:
            if not section.text:
                continue
            if parts and section.spacing == "paragraph":
                parts.append("")
            parts.append(section.text)
        return "\n".join(parts)


def _first_line(text: str) -> str:
    return next((" ".join(line.split()) for line in text.splitlines() if line.strip()), "")


def _block_summary(block: ContentBlock, translate: Translator | None) -> str:
    if isinstance(block, TextBlock):
        return _first_line(resolve_text(block.text, translate))
    if isinstance(block, MediaBlock):
        return block.url or block.kind.title()
    if isinstance(block, FieldListBlock):
        primary = next((field for field in block.fields if field.role == "primary"), None)
        field = primary or (block.fields[0] if block.fields else None)
        return resolve_text(field.value, translate) if field is not None else ""
    if isinstance(block, DiffBlock):
        return block.path
    if isinstance(block, TodoBlock):
        count = len(block.items)
        return f"{count} item" + ("s" if count != 1 else "")
    if isinstance(block, QuestionBlock):
        return block.questions[0].question if block.questions else ""
    if isinstance(block, RawBlock):
        if block.label is not None:
            return resolve_text(block.label, translate)
        return _first_line(block.payload)
    return ""


def entry_summary(entry: TranscriptEntry, translate: Translator | None = None) -> str:
    if isinstance(entry, ActivityEntry):
        call = entry.activity.call
        result = entry.activity.result
        call_summary = summary_text(call.summary_parts, translate) if call is not None else ""
        result_summary = summary_text(result.summary_parts, translate) if result is not None else ""
        if result_summary and result_summary != call_summary:
            return " · ".join(filter(None, (call_summary, result_summary)))
        return call_summary or result_summary
    blocks = entry.blocks
    return next(
        (summary for block in blocks if (summary := _block_summary(block, translate))),
        "",
    )


def _section_for_block(
    block: ContentBlock,
    translate: Translator | None,
    *,
    tone: str | None = None,
    spacing: str = "none",
) -> BodySection | None:
    text = block_text(block, translate)
    if not text:
        return None
    if isinstance(block, TextBlock):
        format_name = block.format
        section_tone = tone or block.tone
    elif isinstance(block, DiffBlock | RawBlock):
        format_name = "code"
        section_tone = tone or "primary"
    elif isinstance(block, FieldListBlock):
        format_name = "plain"
        section_tone = tone or block.tone
    else:
        format_name = "plain"
        section_tone = tone or "primary"
    return BodySection(
        text=text,
        format=format_name,  # type: ignore[arg-type]
        tone=section_tone,  # type: ignore[arg-type]
        spacing=spacing,  # type: ignore[arg-type]
    )


def _source_section(entry: TranscriptEntry, translate: Translator | None) -> BodySection | None:
    label = entry.source.label
    if label is None:
        return None
    return BodySection(resolve_text(label, translate), tone="context")


def _text_sections(
    entry: TextEntry | NoticeEntry, translate: Translator | None
) -> tuple[BodySection, ...]:
    sections: list[BodySection] = []
    source = _source_section(entry, translate)
    if source is not None:
        sections.append(source)
    for block in entry.blocks:
        section = _section_for_block(
            block,
            translate,
            tone="muted" if isinstance(entry, TextEntry) and entry.kind == "thinking" else None,
            spacing="paragraph" if sections else "none",
        )
        if section is not None:
            sections.append(section)
    return tuple(sections)


def _flat_context(fields: tuple, translate: Translator | None) -> str:
    parts: list[str] = []
    for field in fields:
        value = " ".join(resolve_text(field.value, translate).split())
        if not value or field.role == "detail" or field.hint in {"code", "multiline", "json"}:
            continue
        if field.role == "primary":
            parts.append(value)
        else:
            parts.append(f"{field.name.lower().replace('_', ' ')} {value}")
    return " · ".join(parts)


def _activity_sections(
    entry: ActivityEntry, translate: Translator | None
) -> tuple[BodySection, ...]:
    activity = entry.activity
    sections: list[BodySection] = []
    source = _source_section(entry, translate)
    if source is not None:
        sections.append(source)
    call = activity.call
    result = activity.result
    output_first = activity.identity.family in _OUTPUT_FIRST_FAMILIES
    if call is not None:
        context = _flat_context(call.fields, translate)
        if context:
            sections.append(
                BodySection(
                    context,
                    tone="context",
                    spacing="paragraph" if sections else "none",
                )
            )
        for field in call.fields:
            value = resolve_text(field.value, translate)
            rich = field.role == "detail" or field.hint in {"code", "multiline", "json"}
            if not rich:
                continue
            if call.details and activity.identity.family == "todo" and field.hint == "json":
                continue
            label = (
                f"{field.name}:\n"
                if field.hint in {"code", "multiline", "json"}
                else f"{field.name}: "
            )
            sections.append(
                BodySection(
                    label + value,
                    format="code" if field.hint in {"code", "json"} else "plain",
                    tone="primary" if not output_first else "context",
                    spacing="paragraph" if sections else "none",
                )
            )
        for block in call.details:
            section = _section_for_block(
                block,
                translate,
                tone="primary" if not output_first else "context",
                spacing="paragraph" if sections else "none",
            )
            if section is not None:
                sections.append(section)
        if not call.fields and not call.details and call.raw_arguments:
            sections.append(
                BodySection(
                    call.raw_arguments,
                    format="code",
                    tone="context",
                    spacing="paragraph" if sections else "none",
                )
            )
    if result is not None:
        for block in result.blocks:
            section = _section_for_block(
                block,
                translate,
                tone="primary",
                spacing="paragraph" if sections else "none",
            )
            if section is not None:
                sections.append(section)
        if result.raw_fallback:
            sections.append(
                BodySection(
                    result.raw_fallback,
                    format="code",
                    spacing="paragraph" if sections else "none",
                )
            )
    return tuple(sections)


def layout_entry(
    entry: TranscriptEntry,
    *,
    width: int,
    expanded: bool = False,
    translate: Translator | None = None,
    placeholder: str = NO_DETAILS,
) -> RecordLayout:
    """Compose one row; no field name, JSON, or formatted text is parsed here."""

    raw_label = record_label(entry)
    label = translate(raw_label) if translate is not None else raw_label
    compact = is_compact_entry(entry, expanded=expanded)
    status = activity_status(entry)
    chrome_cells = 5
    if not is_dialogue_entry(entry):
        chrome_cells += 3
    if status:
        chrome_cells += 3
    available = max(0, width - chrome_cells)
    summary = entry_summary(entry, translate) or placeholder
    summary_width = max(0, available - cell_len(label) - 1)
    fitted = fit_summary(summary, summary_width)
    header_runs = (HeaderRun(label, "label", True),)
    if compact or isinstance(entry, ActivityEntry):
        header_runs += (HeaderRun(fitted, "summary"),)
    if compact:
        sections: tuple[BodySection, ...] = ()
    elif isinstance(entry, ActivityEntry):
        sections = _activity_sections(entry, translate)
    else:
        sections = _text_sections(entry, translate)
    return RecordLayout(
        header_runs=header_runs,
        body_sections=sections,
        compact=compact,
        label=label,
        bar_color=bar_color_name(entry),
        italic_body=isinstance(entry, TextEntry) and entry.kind == "thinking",
        status=status,
    )


def visible_entry_text(
    entry: TranscriptEntry,
    *,
    expanded: bool,
    width: int,
    translate: Translator | None = None,
    placeholder: str = NO_DETAILS,
) -> str:
    layout = layout_entry(
        entry,
        width=width,
        expanded=expanded,
        translate=translate,
        placeholder=placeholder,
    )
    return "\n".join(filter(None, (layout.header, layout.body)))


def accessibility_text(entry: TranscriptEntry, translate: Translator | None = None) -> str:
    raw_label = record_label(entry)
    label = translate(raw_label) if translate is not None else raw_label
    from kimix_gui.transcript_data import entry_copy_text

    body = entry_copy_text(entry, translate)
    return "\n".join(filter(None, (label, body)))


def specialized_families() -> frozenset[str]:
    """Expose the single specialized-family membership contract to tests."""

    return KNOWN_TOOL_FAMILIES
