"""Word the status line from the figures ``rendering.status_values()`` produced.

The pure layer computes; this module phrases. Splitting it that way is what makes the
status line translatable at all: ``rendering.py`` must not import PySide6, so it
cannot call ``tr()``, and the line is permanently visible under the composer rather
than SDK diagnostics.

Module functions with an explicit ``QCoreApplication.translate`` context instead of
``self.tr``: the caller is ``KimixBridge`` running on its worker thread, and
``translate`` is thread-safe and needs no ``QObject``. Every context and msgid is
spelled out literally because ``pyside6-lupdate`` only reads literal arguments.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication

from kimix_gui.rendering import McpCounts, StatusValues, TokenCounts

# Separator between status fields. Not translated: the glossary keeps ``·`` in every
# language, and it is punctuation rather than copy.
FIELD_SEPARATOR = " · "


def format_status_line(values: StatusValues | None) -> str:
    """Return the one-line status text, or the idle word when there is nothing yet."""

    if values is None:
        values = StatusValues()
    fields: list[str] = []
    if values.context_tokens is not None:
        fields.append(_context_field(values.context_tokens, values.max_context_tokens))
    if values.usage_percent is not None:
        fields.append(
            QCoreApplication.translate("StatusLine", "{percent}%").format(
                percent=f"{values.usage_percent:.1f}"
            )
        )
    if values.tokens is not None:
        fields.extend(_token_fields(values.tokens))
    if values.mcp is not None:
        fields.append(_mcp_field(values.mcp))
    if values.yolo_enabled:
        fields.append(QCoreApplication.translate("StatusLine", "YOLO enabled"))
    if values.afk_enabled:
        fields.append(QCoreApplication.translate("StatusLine", "AFK enabled"))
    return FIELD_SEPARATOR.join(fields) or QCoreApplication.translate("StatusLine", "ready")


def _context_field(tokens: int, maximum: int | None) -> str:
    # Two whole phrases rather than a translated stem plus an appended "/max": the
    # limit is not a suffix in every language.
    if maximum:
        return QCoreApplication.translate("StatusLine", "context {used}/{total}").format(
            used=f"{tokens:,}", total=f"{maximum:,}"
        )
    return QCoreApplication.translate("StatusLine", "context {used}").format(used=f"{tokens:,}")


def _token_fields(tokens: TokenCounts) -> list[str]:
    return [
        QCoreApplication.translate(
            "StatusLine", "tokens in {total} (new {new}, cache read {read}, cache write {write})"
        ).format(
            total=f"{tokens.total_input:,}",
            new=f"{tokens.new_input:,}",
            read=f"{tokens.cache_read:,}",
            write=f"{tokens.cache_write:,}",
        ),
        QCoreApplication.translate("StatusLine", "out {count}").format(count=f"{tokens.output:,}"),
    ]


def _mcp_field(mcp: McpCounts) -> str:
    # Loading and ready are two whole sentences, not a translated bare adjective
    # spliced into a stem: adjective placement is language specific.
    text = (
        QCoreApplication.translate("StatusLine", "MCP {connected}/{total} loading, {tools} tools")
        if mcp.loading
        else QCoreApplication.translate(
            "StatusLine", "MCP {connected}/{total} ready, {tools} tools"
        )
    ).format(connected=mcp.connected, total=mcp.total, tools=mcp.tools)
    if not mcp.servers:
        return text
    # ``name:state`` keeps the raw wire state: it is a protocol value, like the
    # ``_SDK_BORING_RESULT_MESSAGES`` set, and stays readable next to SDK logs.
    detail = ", ".join(f"{server.name}:{server.state}" for server in mcp.servers)
    return f"{text} [{detail}]"


__all__ = ["FIELD_SEPARATOR", "format_status_line"]
