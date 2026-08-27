"""Qt-side catalog for the pure layer's single-word display labels.

The pure layer must not import PySide6, so it cannot call ``tr()``. For labels that
are *single words* -- transcript row titles, tool family names, the placeholder
session title -- there is no word order to get wrong, so the pure layer keeps
returning the English word and that word **is** the msgid. Nothing in the pure layer
changes; the translation happens at the rendering boundary.

``pyside6-lupdate`` only extracts literal arguments, so every msgid is declared here
with ``QT_TRANSLATE_NOOP`` and the context is spelled out literally at each call.
The lookups below are what ``qt/paint.py``, ``qt/transcript.py`` and
``qt/home_view.py`` call.

Words outside the catalog are returned unchanged: the pure layer also derives labels
from wire tool names (``Bash``, ``Custom Mcp``) and from unknown record kinds, and
those must survive verbatim rather than raise or blank out.
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication

# Catalog context for the transcript vocabulary. Kept as one context because the
# words are shared between record kinds and tool families, and a single entry per
# word is what we want in the ``.ts``.
LABEL_CONTEXT = "TranscriptLabels"

# The values of ``transcript_layout.LABELS``: one title per transcript record kind.
RECORD_LABELS: tuple[str, ...] = (
    QT_TRANSLATE_NOOP("TranscriptLabels", "You"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "AI"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Think"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Tool"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Approval"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "System"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Error"),
)

# The values of ``tool_display._FAMILY_LABEL``: the specialized tool row titles.
# ``shell`` has no entry there on purpose -- it shows the wire name (``bash`` ->
# ``Bash``), which is not translatable copy.
TOOL_FAMILY_LABELS: tuple[str, ...] = (
    QT_TRANSLATE_NOOP("TranscriptLabels", "Read"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Grep"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Glob"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Write"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Edit"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Python"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Todo"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Search"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Fetch"),
    QT_TRANSLATE_NOOP("TranscriptLabels", "Agent"),
)

# ``transcript_layout.NO_DETAILS``: shown in place of a summary when a record carries
# nothing but bookkeeping. Filed with the transcript vocabulary because that is the
# only place it appears. The pure layer holds the same literal as its msgid and never
# compares against it, so translating this cannot change which lines are filtered.
NO_DETAILS_TEXT = QT_TRANSLATE_NOOP("TranscriptLabels", "(no details)")

# Application-authored transcript copy carried by ``LocalizedText``. The pure
# normalizer stores only these msgids; this Qt catalog is the single wording boundary.
TRANSCRIPT_MESSAGES: tuple[str, ...] = (
    QT_TRANSLATE_NOOP("TranscriptMessages", "succeeded"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "failed"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "{sender} requests: {action}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Approval {response}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Step {number}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Current step was interrupted"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Compacting context"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Context compacted"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Context compaction failed"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Subagent {kind} · {id}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Retrying step {number}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Hook triggered"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Hook resolved"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Loading MCP servers"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "MCP loading completed"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Notification · {title}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Side question"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Side answer"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Question request"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Hook request"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Tool call request"),
    QT_TRANSLATE_NOOP(
        "TranscriptMessages", "Failed to save session configuration metadata: {reason}"
    ),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Session: {id}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Failed to load history: {reason}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Showing the last turn ({omitted} earlier omitted)"),
    QT_TRANSLATE_NOOP(
        "TranscriptMessages", "Showing the last {shown} turns ({omitted} earlier omitted)"
    ),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Generation cancelled"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Approval decision: {decision}"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Question cancelled"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Question response"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Hook decision: allow"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Hook decision: block"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "External client-side tools are not supported yet"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Unsupported SDK request: {request}"),
    QT_TRANSLATE_NOOP(
        "TranscriptMessages",
        "/help  /status  /clear  /compact [instruction]  /quit (back to home)",
    ),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Session context cleared"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Compacting context…"),
    QT_TRANSLATE_NOOP("TranscriptMessages", "Unknown command: {command}"),
)

# ``session_index.UNTITLED_TITLE``: the stand-in for a session with no title. Filed
# under ``HomeView`` because that is the only view that shows it, which keeps it
# next to the rest of the home-screen copy in the ``.ts``.
UNTITLED_SESSION = QT_TRANSLATE_NOOP("HomeView", "Untitled")

_TRANSCRIPT_CATALOG = frozenset((*RECORD_LABELS, *TOOL_FAMILY_LABELS))
_TRANSCRIPT_MESSAGE_CATALOG = frozenset(TRANSCRIPT_MESSAGES)


def translate_label(label: str) -> str:
    """Translate one pure-layer transcript label, or return it unchanged.

    The membership check is deliberate rather than an unconditional ``translate``
    call: it documents exactly which words this module owns and makes the
    pass-through behaviour for wire-derived labels testable.
    """

    if label not in _TRANSCRIPT_CATALOG:
        return label
    return QCoreApplication.translate("TranscriptLabels", label)


def translate_no_details() -> str:
    """Translate the stand-in for a record with no informative line."""

    return QCoreApplication.translate("TranscriptLabels", NO_DETAILS_TEXT)


def translate_transcript_text(text: str) -> str:
    """Translate a semantic transcript msgid or label at the Qt boundary."""

    if text in _TRANSCRIPT_CATALOG:
        return QCoreApplication.translate("TranscriptLabels", text)
    if text in _TRANSCRIPT_MESSAGE_CATALOG:
        return QCoreApplication.translate("TranscriptMessages", text)
    return text


def translate_session_title(title: str) -> str:
    """Translate the placeholder session title, leaving real titles untouched."""

    if title != UNTITLED_SESSION:
        return title
    return QCoreApplication.translate("HomeView", "Untitled")


__all__ = [
    "LABEL_CONTEXT",
    "NO_DETAILS_TEXT",
    "RECORD_LABELS",
    "TOOL_FAMILY_LABELS",
    "TRANSCRIPT_MESSAGES",
    "UNTITLED_SESSION",
    "translate_label",
    "translate_no_details",
    "translate_session_title",
    "translate_transcript_text",
]
