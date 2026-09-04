from __future__ import annotations

import regex as re
import textwrap
from typing import Any, Iterator

from kimix.ui.printing import (
    BgColor,
    BgColor256,
    Color,
    Color256,
    GRAY_LIGHT,
    Style,
    _ansi_prefix,
    _resolve_bg,
    _resolve_fg,
    _strip_ansi,
    colorful_text,
)


def _input(
    text: str,
    text_arr: list[str],
) -> str:
    if text_arr is None or len(text_arr) == 0:
        if text:
            print(text, end='', flush=True)
        return input()
    return text_arr.pop(0)


def _split_text(lines: list[str], command_map: set[str] | None = None) -> list[str]:
    text_arr: list[str] = []
    current_text: list[str] = []
    for line in lines:
        strip_line = line.strip()
        if len(strip_line) == 0:
            current_text.append('')
            continue
        if strip_line.startswith('/'):
            if len(strip_line) > 1:
                cmd = strip_line[1:].split()[0]
                if command_map is not None and cmd not in command_map:
                    current_text.append(line)
                    continue
            if current_text:
                text_arr.append('\n'.join(current_text))
                current_text = []
            if len(strip_line) > 1:
                text_arr.append(strip_line)
        else:
            current_text.append(line)
    if current_text:
        text_arr.append('\n'.join(current_text))
    return text_arr


# ---------------------------------------------------------------------------
# Default Markdown theme
# ---------------------------------------------------------------------------

DEFAULT_MD_THEME: dict[str, dict[str, Any]] = {
    "h1": {
        "fg": Color.BRIGHT_YELLOW,
        "bg": BgColor256(17),
        "styles": [Style.BOLD, Style.UNDERLINE],
        "bar_char": "═",
        "top_bar": True,
        "bottom_bar": True,
        "prefix": " ",
        "suffix": " ",
    },
    "h2": {
        "fg": Color.BRIGHT_YELLOW,
        "bg": BgColor256(17),
        "styles": [Style.BOLD],
        "bar_char": "─",
        "top_bar": False,
        "bottom_bar": True,
        "prefix": " ",
        "suffix": " ",
    },
    "h3": {
        "fg": Color.YELLOW,
        "bg": None,
        "styles": [Style.BOLD, Style.UNDERLINE],
    },
    "h4": {
        "fg": Color.YELLOW,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "h5": {
        "fg": Color.BRIGHT_WHITE,
        "bg": None,
        "styles": [Style.BOLD, Style.DIM],
    },
    "h6": {
        "fg": GRAY_LIGHT,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "bold": {
        "fg": Color.BRIGHT_WHITE,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "italic": {
        "fg": Color.BRIGHT_CYAN,
        "bg": None,
        "styles": [Style.ITALIC],
    },
    "bold_italic": {
        "fg": Color.BRIGHT_WHITE,
        "bg": None,
        "styles": [Style.BOLD, Style.ITALIC],
    },
    "inline_code": {
        "fg": Color.YELLOW,
        "bg": BgColor256(236),
        "styles": [],
    },
    "code_fence": {
        "fg": GRAY_LIGHT,
        "bg": None,
        "styles": [Style.BOLD, Style.DIM],
    },
    "code_block": {
        "fg": Color.GREEN,
        "bg": BgColor256(232),
        "styles": [],
    },
    "strikethrough": {
        "fg": Color.BRIGHT_BLACK,
        "bg": None,
        "styles": [Style.STRIKETHROUGH, Style.DIM],
    },
    "link_text": {
        "fg": Color.BRIGHT_BLUE,
        "bg": None,
        "styles": [Style.UNDERLINE],
    },
    "link_url": {
        "fg": Color.BLUE,
        "bg": BgColor256(235),
        "styles": [],
    },
    "image_alt": {
        "fg": Color.BRIGHT_MAGENTA,
        "bg": None,
        "styles": [Style.ITALIC],
    },
    "image_src": {
        "fg": Color.MAGENTA,
        "bg": BgColor256(235),
        "styles": [],
    },
    "blockquote_marker": {
        "fg": Color.BRIGHT_BLACK,
        "bg": None,
        "styles": [],
        "prefix": "▌ ",
    },
    "blockquote_text": {
        "fg": GRAY_LIGHT,
        "bg": BgColor256(235),
        "styles": [Style.ITALIC],
    },
    "list_marker": {
        "fg": Color.BRIGHT_YELLOW,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "task_checked": {
        "fg": Color.BRIGHT_GREEN,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "task_unchecked": {
        "fg": Color.BRIGHT_RED,
        "bg": None,
        "styles": [Style.BOLD],
    },
    "hr": {
        "fg": Color.BRIGHT_BLACK,
        "bg": None,
        "styles": [Style.DIM],
        "char": "─",
    },
    "table_border": {
        "fg": Color.CYAN,
        "bg": None,
        "styles": [Style.DIM],
    },
    "table_header": {
        "fg": Color.BRIGHT_CYAN,
        "bg": BgColor256(24),
        "styles": [Style.BOLD],
    },
    "table_cell": {
        "fg": None,
        "bg": None,
        "styles": [],
    },
}


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------

_ESCAPE_CHARS = r"\\*\\_\\`\\~\\[\\]\\(\\)\\!"


def _md_escape_inline(text: str) -> tuple[str, list[str]]:
    """Replace escaped markdown characters with unique placeholders."""
    placeholders: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        placeholder = f"\x00MDESC{len(placeholders)}\x00"
        placeholders.append(match.group(0))
        return placeholder

    escaped = re.sub(r"\\(.)", _repl, text)
    return escaped, placeholders


def _md_restore_inline(text: str, placeholders: list[str]) -> str:
    """Restore escaped markdown characters from placeholders."""
    for idx, original in enumerate(placeholders):
        text = text.replace(f"\x00MDESC{idx}\x00", original)
    return text


def _md_inline_code_split(text: str) -> Iterator[tuple[bool, str]]:
    """Yield (is_code, segment) pairs by splitting on inline code spans."""
    parts = re.split(r"(`[^`]+`)", text)
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            yield True, part[1:-1]
        else:
            yield False, part


# Image: ![alt](src) - one level of nested parens in src
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(((?:[^()]|\([^()]*\))*)\)")
# Link: [text](url) - one level of nested parens in url
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(((?:[^()]|\([^()]*\))*)\)")
# Strikethrough: ~~text~~
_STRIKE_RE = re.compile(r"~~([^~]+)~~")
# Bold+italic: ***text*** or ___text___
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___)(.+?)\1")
# Bold: **text** or __text__
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
# Italic: *text* only.  Underscore emphasis (_text_) is intentionally NOT
# transformed: agent output frequently contains literal tokens such as _TEXT_
# or snake_case identifiers whose underscores must be preserved verbatim
# instead of being converted into colored italic text.
_ITALIC_RE = re.compile(r"\*([^*]+)\*")


def _md_apply_inline_styles(segment: str, theme: dict[str, Any]) -> str:
    """Apply inline markdown highlighting to one non-code segment."""

    def _image_repl(m: re.Match[str]) -> str:
        alt = colorful_text(m.group(1), **theme["image_alt"])
        src = colorful_text(m.group(2), **theme["image_src"])
        return f"{alt}: {src}"

    def _link_repl(m: re.Match[str]) -> str:
        txt = colorful_text(m.group(1), **theme["link_text"])
        url = colorful_text(m.group(2), **theme["link_url"])
        return f"{txt} ({url})"

    def _strike_repl(m: re.Match[str]) -> str:
        return colorful_text(m.group(1), **theme["strikethrough"])

    def _bold_italic_repl(m: re.Match[str]) -> str:
        return colorful_text(m.group(2), **theme["bold_italic"])

    def _bold_repl(m: re.Match[str]) -> str:
        body = colorful_text(m.group(2), **theme["bold"])
        return body

    def _italic_repl(m: re.Match[str]) -> str:
        return colorful_text(m.group(1), **theme["italic"])

    segment = _IMAGE_RE.sub(_image_repl, segment)
    segment = _LINK_RE.sub(_link_repl, segment)
    segment = _STRIKE_RE.sub(_strike_repl, segment)
    segment = _BOLD_ITALIC_RE.sub(_bold_italic_repl, segment)
    segment = _BOLD_RE.sub(_bold_repl, segment)
    segment = _ITALIC_RE.sub(_italic_repl, segment)
    return segment


def _md_highlight_inline(text: str, theme: dict[str, Any]) -> str:
    """Highlight inline markdown in ``text`` while preserving inline code."""
    text, placeholders = _md_escape_inline(text)
    parts: list[str] = []
    for is_code, segment in _md_inline_code_split(text):
        if is_code:
            parts.append(colorful_text(segment, **theme["inline_code"]))
        else:
            parts.append(_md_apply_inline_styles(segment, theme))
    return _md_restore_inline("".join(parts), placeholders)


# ---------------------------------------------------------------------------
# Block renderers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UNORDERED_LIST_RE = re.compile(r"^([\s]*)([-*+])\s+(.*)$")
_ORDERED_LIST_RE = re.compile(r"^([\s]*)(\d+)\.\s+(.*)$")
_TASK_LIST_RE = re.compile(r"^([\s]*)([-*+])\s+\[([ xX])\]\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")
_HR_RE = re.compile(r"^([-*_])\s*\1\s*\1(?:\s*\1)*$")
_CODE_FENCE_RE = re.compile(r"^(```|~~~)(.*)$")
_TABLE_BORDER_RE = re.compile(r"^[\s|]*:?-+:?[\s|]*$")


def _md_terminal_width(width: int | None) -> int:
    if width is not None:
        return max(width, 20)
    try:
        import shutil
        return shutil.get_terminal_size().columns or 80
    except Exception:
        return 80


def _md_render_heading_bar(width: int, char: str, theme: dict[str, Any]) -> str:
    bar_text = char * max(width, 1)
    hr_spec = theme["hr"]
    return colorful_text(bar_text, fg=hr_spec.get("fg"), bg=hr_spec.get("bg"), styles=hr_spec.get("styles"))


def _md_render_heading(
    level: int,
    marker: str,
    content: str,
    theme: dict[str, Any],
    width: int | None,
    heading_bars: bool = True,
) -> str:
    key = f"h{level}"
    spec = theme.get(key, {})
    fg = spec.get("fg")
    bg = spec.get("bg")
    styles = spec.get("styles", [])
    bar_char = spec.get("bar_char", "")
    top_bar = spec.get("top_bar", False)
    bottom_bar = spec.get("bottom_bar", False)
    prefix = spec.get("prefix", "")
    suffix = spec.get("suffix", "")

    highlighted_content = _md_highlight_inline(content, theme)
    body = colorful_text(f"{prefix}{highlighted_content}{suffix}", fg=fg, bg=bg, styles=styles)
    lines: list[str] = []
    term_width = _md_terminal_width(width)

    if heading_bars and top_bar and bar_char:
        lines.append(_md_render_heading_bar(term_width, bar_char, theme))
    lines.append(body)
    if heading_bars and bottom_bar and bar_char:
        lines.append(_md_render_heading_bar(term_width, bar_char, theme))
    return "\n".join(lines)


def _md_render_list(line: str, theme: dict[str, Any]) -> str:
    task = _TASK_LIST_RE.match(line)
    if task:
        indent, bullet, checked, content = task.groups()
        checkbox_key = "task_checked" if checked.lower() == "x" else "task_unchecked"
        checkbox = "[x]" if checked.lower() == "x" else "[ ]"
        marker = colorful_text(f"{indent}{bullet} {checkbox} ", **theme[checkbox_key])
        return marker + _md_highlight_inline(content, theme)

    unordered = _UNORDERED_LIST_RE.match(line)
    if unordered:
        indent, bullet, content = unordered.groups()
        marker = colorful_text(f"{indent}{bullet} ", **theme["list_marker"])
        return marker + _md_highlight_inline(content, theme)

    ordered = _ORDERED_LIST_RE.match(line)
    if ordered:
        indent, number, content = ordered.groups()
        marker = colorful_text(f"{indent}{number}. ", **theme["list_marker"])
        return marker + _md_highlight_inline(content, theme)

    return _md_highlight_inline(line, theme)


def _md_render_blockquote(line: str, theme: dict[str, Any], width: int | None = None) -> str:
    m = _BLOCKQUOTE_RE.match(line)
    if not m:
        return _md_highlight_inline(line, theme)
    content = m.group(1)
    marker_spec = theme["blockquote_marker"]
    marker_prefix = marker_spec.get("prefix", "▌ ")
    marker_vis = len(marker_prefix)
    marker = colorful_text(marker_prefix, fg=marker_spec.get("fg"), bg=marker_spec.get("bg"), styles=marker_spec.get("styles"))
    # Apply inline highlighting first, then wrap in blockquote style
    highlighted = _md_highlight_inline(content, theme)
    blockquote_spec = theme["blockquote_text"]
    bq_fg = blockquote_spec.get("fg")
    bq_bg = blockquote_spec.get("bg")
    bq_styles = blockquote_spec.get("styles", [])
    bq_prefix = _ansi_prefix(_resolve_fg(bq_fg), _resolve_bg(bq_bg), tuple(s.value for s in bq_styles) if bq_styles else ())
    if bq_prefix:
        # After each ANSI reset, re-apply blockquote style so bg/fg/italic persists
        highlighted = highlighted.replace("\x1b[0m", f"\x1b[0m{bq_prefix}")
        text = f"{bq_prefix}{highlighted}\x1b[0m"
    else:
        text = highlighted

    # Width wrapping: wrap the text content, then prepend marker to first line
    if width is not None:
        term_width = _md_terminal_width(width)
        content_width = max(term_width - marker_vis, 10)
        plain = _strip_ansi(text)
        if len(plain) > content_width:
            wrapped_plain = textwrap.fill(
                plain,
                width=content_width,
                break_long_words=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            # Determine break positions from wrapped_plain
            break_indices: set[int] = set()
            vis_idx = 0
            for ch in wrapped_plain:
                if ch == "\n":
                    break_indices.add(vis_idx)
                else:
                    vis_idx += 1
            # Insert newlines in the ANSI text at break positions
            out_chars: list[str] = []
            vis_pos = 0
            i = 0
            while i < len(text):
                if text[i] == "\x1b":
                    end = text.find("m", i)
                    if end == -1:
                        end = len(text) - 1
                    out_chars.append(text[i : end + 1])
                    i = end + 1
                    continue
                if vis_pos in break_indices:
                    out_chars.append("\n")
                out_chars.append(text[i])
                vis_pos += 1
                i += 1
            text = "".join(out_chars)

    # Prepend marker to first line; indent continuation lines to align after marker
    if "\n" in text:
        indent = " " * marker_vis
        lines = text.split("\n")
        result = marker + lines[0]
        for line in lines[1:]:
            result += "\n" + indent + line
        return result
    return marker + text


def _md_render_hr(width: int, theme: dict[str, Any]) -> str:
    hr_spec = theme["hr"]
    char = hr_spec.get("char", "─")
    bar_text = char * max(width, 1)
    return colorful_text(bar_text, fg=hr_spec.get("fg"), bg=hr_spec.get("bg"), styles=hr_spec.get("styles"))


def _md_split_table_cells(row: str) -> list[str]:
    """Split a pipe table row into cells, ignoring outer pipes."""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _md_render_table(lines: list[str], theme: dict[str, Any]) -> str:
    rows = [_md_split_table_cells(line) for line in lines]
    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    col_widths = [0] * max_cols
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(_strip_ansi(cell)))

    def _pad(cell: str, width: int) -> str:
        visible = len(_strip_ansi(cell))
        return cell + " " * max(width - visible, 0)

    def _render_row(row: list[str], cell_spec: dict[str, Any]) -> str:
        cells = []
        for idx in range(max_cols):
            raw = row[idx] if idx < len(row) else ""
            styled = colorful_text(_pad(raw, col_widths[idx]), **cell_spec)
            cells.append(styled)
        border = colorful_text("|", **theme["table_border"])
        return border + border.join(cells) + border

    rendered_rows: list[str] = []
    for idx, row in enumerate(rows):
        if idx == 0:
            rendered_rows.append(_render_row(row, theme["table_header"]))
        elif _TABLE_BORDER_RE.match(lines[idx]):
            # separator line: render as a border line
            sep_parts = []
            for w in col_widths:
                sep_parts.append(colorful_text("-" * (w + 2), **theme["table_border"]))
            border = colorful_text("+", **theme["table_border"])
            rendered_rows.append(border + border.join(sep_parts) + border)
        else:
            rendered_rows.append(_render_row(row, theme["table_cell"]))

    return "\n".join(rendered_rows)


def _md_render_code_block(
    fence: str,
    info: str,
    body: list[str],
    theme: dict[str, Any],
) -> str:
    fence_spec = theme["code_fence"]
    body_spec = theme["code_block"]
    lines: list[str] = []
    info_text = f" {info}" if info else ""
    lines.append(colorful_text(f"{fence}{info_text}", **fence_spec))
    for line in body:
        lines.append(colorful_text(line, **body_spec))
    lines.append(colorful_text(fence, **fence_spec))
    return "\n".join(lines)


def _md_render_paragraph(text: str, theme: dict[str, Any], width: int | None) -> str:
    highlighted = _md_highlight_inline(text, theme)
    if width is None:
        return highlighted

    term_width = _md_terminal_width(width)
    # textwrap.fill doesn't understand ANSI, so work on plain text and rebuild
    plain = _strip_ansi(highlighted)
    if len(plain) <= term_width:
        return highlighted

    # Use textwrap to determine break positions in the plain text
    wrapped_plain = textwrap.fill(
        plain,
        width=term_width,
        break_long_words=False,
        replace_whitespace=False,
        drop_whitespace=False,
    )

    # Build a set of visible-char indices where newlines occur in wrapped_plain
    break_indices: set[int] = set()
    vis_idx = 0
    for ch in wrapped_plain:
        if ch == "\n":
            break_indices.add(vis_idx)
        else:
            vis_idx += 1

    # Walk highlighted and insert newlines at the recorded break positions
    out_chars: list[str] = []
    vis_pos = 0
    i = 0
    while i < len(highlighted):
        if highlighted[i] == "\x1b":
            end = highlighted.find("m", i)
            if end == -1:
                end = len(highlighted) - 1
            out_chars.append(highlighted[i : end + 1])
            i = end + 1
            continue
        # Regular character
        if vis_pos in break_indices:
            out_chars.append("\n")
        out_chars.append(highlighted[i])
        vis_pos += 1
        i += 1

    return "".join(out_chars)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def render_markdown(
    text: str,
    *,
    width: int | None = None,
    theme: dict[str, Any] | None = None,
    heading_bars: bool = True,
) -> str:
    """Render Markdown text to ANSI-styled terminal output."""

    merged_theme: dict[str, Any] = {}
    merged_theme.update(DEFAULT_MD_THEME)
    if theme:
        for key, value in theme.items():
            if isinstance(value, dict) and isinstance(merged_theme.get(key), dict):
                merged_theme[key] = {**merged_theme[key], **value}
            else:
                merged_theme[key] = value

    raw_lines = text.splitlines()
    lines = raw_lines if raw_lines else [""]

    output: list[str] = []
    paragraph_lines: list[str] = []
    idx = 0

    def _flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            output.append(_md_render_paragraph(" ".join(paragraph_lines), merged_theme, width))
            paragraph_lines = []
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        # Code fence
        fence_match = _CODE_FENCE_RE.match(stripped)
        if fence_match:
            _flush_paragraph()
            fence = fence_match.group(1)
            info = fence_match.group(2).strip()
            body: list[str] = []
            idx += 1
            while idx < len(lines):
                inner = lines[idx]
                if inner.strip().startswith(fence):
                    idx += 1
                    break
                body.append(inner)
                idx += 1
            output.append(_md_render_code_block(fence, info, body, merged_theme))
            continue

        # Heading
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            _flush_paragraph()
            level = len(heading_match.group(1))
            content = heading_match.group(2).strip()
            output.append(
                _md_render_heading(
                    level, heading_match.group(1), content, merged_theme, width, heading_bars
                )
            )
            if level <= 2:
                output.append("")
            idx += 1
            continue

        # Horizontal rule
        if _HR_RE.match(stripped):
            _flush_paragraph()
            output.append(_md_render_hr(_md_terminal_width(width), merged_theme))
            idx += 1
            continue

        # List item
        if _TASK_LIST_RE.match(stripped) or _UNORDERED_LIST_RE.match(stripped) or _ORDERED_LIST_RE.match(stripped):
            _flush_paragraph()
            output.append(_md_render_list(line, merged_theme))
            idx += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            _flush_paragraph()
            output.append(_md_render_blockquote(line, merged_theme, width))
            idx += 1
            continue

        # Table
        if stripped.startswith("|"):
            _flush_paragraph()
            table_lines: list[str] = []
            while idx < len(lines):
                candidate = lines[idx].strip()
                if not candidate.startswith("|"):
                    break
                table_lines.append(candidate)
                idx += 1
            output.append(_md_render_table(table_lines, merged_theme))
            continue

        # Blank line
        if not stripped:
            _flush_paragraph()
            output.append("")
            idx += 1
            continue

        # Paragraph continuation
        paragraph_lines.append(line.strip())
        idx += 1

    _flush_paragraph()

    result = "\n".join(output)
    # Ensure exactly one trailing newline
    if not result.endswith("\n"):
        result += "\n"
    return result


def print_markdown(text: str, *, end: str = "\n", **kwargs: Any) -> None:
    """Render and print Markdown text to the terminal."""
    rendered = render_markdown(text, **kwargs)
    print(rendered, end=end)
