"""Shared FTS5 infrastructure: sanitizer, CJK routing, and LIKE fallback.

This is a leaf module — it imports only stdlib and ``regex``.  It ports the
battle-tested patterns from the Hermes-CN-Core reference project
(``hermes_state_search.py`` / ``hermes_state_common.py``):

- ``sanitize_fts5_query`` protects quoted phrases, strips FTS5-special
  characters, and wraps hyphen/dot terms so arbitrary user input never raises
  ``OperationalError`` inside a MATCH query.
- CJK detection (``contains_cjk`` / ``count_cjk`` / ``has_lone_cjk_run`` /
  ``trigram_eligible_tokens``) drives query routing between the unicode61 FTS5
  index, the trigram index, and a LIKE substring fallback.
- ``escape_like`` escapes ``%`` / ``_`` / ``\\`` for the LIKE fallback.
"""

from __future__ import annotations

from typing import Any

import regex as re

# Cap user-controlled FTS input before any regex processing. Search queries do
# not need to be arbitrarily large; bounding them keeps sanitizer/runtime
# behavior predictable under adversarial input.
MAX_FTS5_QUERY_CHARS = 2_048

# Characters FTS5's query grammar rejects outside a quoted phrase. Anything
# missing from this set reaches MATCH raw and raises, which the execute site
# swallows into zero results — the failure this strip step exists to prevent.
# Assembled through re.escape so the backslash cannot be eaten as a regex
# escape inside the class.
#
# ``%`` is deliberately excluded: a CJK query falls back to a LIKE search that
# needs it preserved as a literal (that path escapes wildcards itself), so
# stripping it here widened those queries onto unrelated rows.
_FTS5_SPECIAL_CHARS = '+{}():"^@/#&|~[]<>,;!?$=\\\''
_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")


def _is_cjk_codepoint(cp: int) -> bool:
    """True for CJK Unified Ideographs, extensions, symbols, kana, hangul."""
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
    )


def contains_cjk(text: str) -> bool:
    """Check if *text* contains CJK (Chinese, Japanese, Korean) characters."""
    return any(_is_cjk_codepoint(ord(ch)) for ch in text)


def count_cjk(text: str) -> int:
    """Count CJK characters in *text*."""
    return sum(1 for ch in text if _is_cjk_codepoint(ord(ch)))


def has_lone_cjk_run(query: str) -> bool:
    """True when any maximal CJK run in *query* is a single char.

    The trigram index needs >=3 UTF-8 bytes (3 CJK chars) per token, so a
    1-char CJK term can't match inside longer runs there — those queries keep
    the LIKE substring route.
    """
    run = 0
    for ch in query:
        if _is_cjk_codepoint(ord(ch)):
            run += 1
        else:
            if run == 1:
                return True
            run = 0
    return run == 1


def trigram_eligible_tokens(query: str) -> bool:
    """True when every non-operator token is long enough for the trigram
    tokenizer to match (>=3 chars).

    The trigram tokenizer indexes overlapping 3-character sequences, so a
    token shorter than 3 chars produces no trigrams and can never match.
    With FTS5's implicit-AND between tokens, a single short token makes the
    whole MATCH return nothing, so the trigram path is only worth taking when
    every searchable token qualifies.
    """
    tokens = [
        t for t in query.strip('"').strip().split()
        if t.upper() not in {"AND", "OR", "NOT"}
    ]
    return bool(tokens) and all(len(t) >= 3 for t in tokens)


def escape_like(text: str) -> str:
    """Escape SQL LIKE wildcards so user text matches literally.

    Pair with ``ESCAPE '\\'`` in the clause.  ``%`` and ``_`` are wildcards to
    LIKE, and ``_`` in particular is common in paths/identifiers.
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user input for safe use in FTS5 MATCH queries.

    FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
    ``+``, ``*``, ``{``, ``}``, the column-filter operator ``:`` and bare
    boolean operators (``AND``, ``OR``, ``NOT``) have special meaning.
    Passing raw user input directly to MATCH can cause
    ``sqlite3.OperationalError``.

    Strategy:
    - Preserve properly paired quoted phrases (``"exact phrase"``)
    - Strip unmatched FTS5-special characters that would cause errors
    - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
      matches them as exact phrases instead of splitting on the
      hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
    """
    if not isinstance(query, str):
        return ""
    query = query[:MAX_FTS5_QUERY_CHARS]

    # Step 1: Extract balanced double-quoted phrases and protect them from
    # further processing via numbered placeholders. Single linear scan so
    # pathological quote runs cannot induce regex backtracking.
    quoted_parts: list[str] = []
    pieces: list[str] = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch != '"':
            pieces.append(ch)
            i += 1
            continue
        end = query.find('"', i + 1)
        if end == -1:
            # Unmatched quote: replace with whitespace.
            pieces.append(" ")
            i += 1
            continue
        quoted_parts.append(query[i : end + 1])
        pieces.append(f"\x00Q{len(quoted_parts) - 1}\x00")
        i = end + 1

    sanitized = "".join(pieces)

    # Step 2: Strip remaining (unmatched) FTS5-special characters. ``:`` is
    # FTS5's column-filter operator; an unquoted colon query like ``TODO: fix``
    # parses as ``column:term`` and raises "no such column" — swallowed at the
    # execute site into zero results. Strip it like the others.
    sanitized = _FTS5_SPECIAL_RE.sub(" ", sanitized)

    # Step 2b: ``%`` is excluded from the class above only to protect the CJK
    # LIKE-fallback path (LIKE treats % as a wildcard the fallback builds
    # itself). A non-CJK query never reaches that fallback, so strip it here.
    if "%" in sanitized and not contains_cjk(sanitized):
        sanitized = sanitized.replace("%", " ")

    # Step 3: Collapse repeated * into a single one, and remove leading *
    # (prefix-only needs at least one char before *).
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

    # Step 4: Remove dangling boolean operators at start/end that would cause
    # syntax errors (e.g. "hello AND" or "OR world").
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
    sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

    # Step 5: Wrap unquoted dotted and/or hyphenated terms in double quotes.
    # A single pass avoids the double-quoting bug that would occur if dotted,
    # hyphenated and underscored patterns were applied sequentially.
    sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

    # Step 6: Restore preserved quoted phrases.
    for i, quoted in enumerate(quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

    return sanitized.strip()


def quote_fts_tokens(raw_query: str) -> str:
    """Quote each non-operator token to neutralize FTS5 special characters
    while preserving boolean operators (AND/OR/NOT) for multi-term queries.

    Used by the trigram/CJK paths where tokens are matched as substrings.
    """
    parts: list[str] = []
    for tok in raw_query.split():
        if tok.upper() in {"AND", "OR", "NOT"}:
            parts.append(tok)
        else:
            parts.append('"' + tok.replace('"', '""') + '"')
    return " ".join(parts)


def extract_text_from_content(content: Any) -> str:
    """Extract plain searchable text from a message ``content`` value.

    Handles every shape stored by kimi-agent:
    - ``str`` content (legacy plain-string messages, checkpoint markers)
    - a single content part dict/object with a ``text`` attribute/key
    - a list of parts: ``TextPart``-like objects, plain strings, or dicts
      with ``{"type": "text", "text": ...}``.

    Returns the parts joined by newlines (matching ``HistoryIndex``'s
    ``_message_to_text`` behavior), or ``""`` when nothing is extractable.
    """
    parts: list[str] = []

    def _append(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts.append(value)
            return
        text = getattr(value, "text", None)
        if isinstance(text, str):
            parts.append(text)
            return
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                parts.append(text)
            return

    if isinstance(content, (list, tuple)):
        for item in content:
            _append(item)
    else:
        _append(content)
    return "\n".join(parts)


def message_text(message: Any) -> str:
    """Extract searchable text from a ``kosong.message.Message``-like object.

    Accepts anything with ``.role`` and ``.content`` so the shared helper can
    be used without importing ``kosong`` in this leaf module.
    """
    content = getattr(message, "content", None)
    if content is None:
        return ""
    return extract_text_from_content(content)
