"""Utilities for prompt string manipulation."""

import regex as re
import unicodedata
import uuid as _uuid
from pathlib import Path

from kimix.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TEXT = _native_get_module("text")


# ---------------------------------------------------------------------------
# Text safety: clean hidden/invisible characters and prevent tokenization failures
# ---------------------------------------------------------------------------

def clean_text(text: str, keep_newlines: bool = True) -> str:
    """Remove invisible/hidden characters from text.

    Targets:
    - Zero-width characters (\u200b, \u200c, \u200d, \ufeff, \u2060, etc.)
    - PDF/Word hidden format characters
    - Most C0/C1 control characters
    - Soft hyphens, directional marks, override chars
    """
    if not isinstance(text, str):
        text = str(text)

    # Native acceleration: kimix_native.text.clean_text (bit-identical ANSI
    # zero-width/control strip + NFC normalize + strip).
    if _native_use_native("TEXT") and _NATIVE_TEXT is not None:
        return _NATIVE_TEXT.clean_text(text, keep_newlines)

    # Step 1: Remove zero-width and format characters explicitly
    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\u00ad\ufeff"
        r"\u200e\u200f\u202a-\u202e\u2066-\u2069]",
        "",
        text,
    )

    # Step 2: Remove control characters (C0/C1), optionally keep \\n\\r\\t
    if keep_newlines:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    else:
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Step 3: Normalize Unicode (NFC) to collapse spoofed glyphs
    text = unicodedata.normalize("NFC", text)

    # Step 4: Strip leading/trailing whitespace artifacts
    return text.strip()


def _strip_invalid_unicode(text: str) -> str:
    """Remove surrogates, noncharacters, PUA, and replacement chars in one pass."""
    result: list[str] = []
    append = result.append
    for ch in text:
        cp = ord(ch)
        # Surrogates
        if 0xD800 <= cp <= 0xDFFF:
            continue
        # Replacement char
        if cp == 0xFFFD:
            continue
        # Noncharacters
        if 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFF) in (0xFFFE, 0xFFFF):
            continue
        # PUA
        if 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD:
            continue
        append(ch)
    return "".join(result)


_DEDUPE_CACHE: dict[int, re.Pattern[str]] = {}


def _dedupe_repeats(text: str, max_repeat: int = 100) -> str:
    """Collapse runs of a single character longer than *max_repeat*."""
    if max_repeat <= 0:
        return text
    pattern = _DEDUPE_CACHE.get(max_repeat)
    if pattern is None:
        pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
        _DEDUPE_CACHE[max_repeat] = pattern
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)


# Factored out the common suffix so the alternation is shorter and the regex
# engine only compiles/evaluates the character class once.
_PATH_RE = re.compile(
    r"""
    (?<![\w/\\:.])          # not preceded by word char, slash, colon, or dot
    (?P<path>
        (?: / | \.{1,2}/ | ~/ | [A-Za-z]:[\\/] | [\w.-]+[\\/] )
        (?:
            [^\s?#<>\"'`|{},;:!?)\]]+ [\\/]
            |
            [^\s?#<>\"'`|{},;:!?)\]]+ [^\S\r\n]+ [^\s?#<>\"'`|{},;:!?)\]]+ [\\/]
        )*
        (?:
            [^\s?#<>\"'`|{},;:!?)\]]++ (?![^\S\r\n]+(?![a-z]+\b)[^\s?#<>\"'`|{},;:!?)\]]+)
            |
            [^\s?#<>\"'`|{},;:!?)\]]+ [^\S\r\n]+ (?! [a-z]+ \b ) [^\s?#<>\"'`|{},;:!?)\]]+
        )?
    )
    """,
    re.VERBOSE,
)

_NON_PATH_RE = re.compile(
    r"^\d+/\d+$|"              # pure fraction
    r"^\d{4}/\d{1,2}/\d{1,2}$|"  # ISO date
    r"^\d{1,2}/\d{1,2}/\d{4}$"   # US date
)

_NON_PATH_RE_MATCH = _NON_PATH_RE.match
_TRAILING_PUNCTUATION = ". , ; : ! ? ) ] }".replace(" ", "")


class _Replacer:
    """Callable replacement helper to avoid re-creating a function on every call."""

    __slots__ = ("text", "text_len", "non_path_match", "trailing_punct", "_code_ranges")

    def __init__(self, text: str) -> None:
        self.text = text
        self.text_len = len(text)
        self.non_path_match = _NON_PATH_RE_MATCH
        self.trailing_punct = _TRAILING_PUNCTUATION
        # Pre-compute markdown fenced code-block ranges (``` … ```).
        self._code_ranges: list[tuple[int, int]] = []
        pos = 0
        while True:
            start = text.find("```", pos)
            if start == -1:
                break
            end = text.find("```", start + 3)
            if end == -1:
                self._code_ranges.append((start, len(text)))
                break
            self._code_ranges.append((start, end + 3))
            pos = end + 3

    def __call__(self, m: re.Match[str]) -> str:
        raw = m.group("path")
        raw_start, raw_end = m.span("path")
        text = self.text
        text_len = self.text_len

        # Inside a markdown fenced code block – leave as-is.
        for code_start, code_end in self._code_ranges:
            if code_start <= raw_start < code_end:
                return raw

        # Already inside quotes, backticks, or bracket pairs – leave as-is.
        if raw_start > 0 and raw_end < text_len:
            prev, nxt = text[raw_start - 1], text[raw_end]
            if prev == nxt and prev in "'\"`":
                return raw
            if (prev, nxt) in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")):
                return raw

        # Strip trailing punctuation – fast-path when unnecessary.
        trailing_punct = self.trailing_punct
        if raw and raw[-1] in trailing_punct:
            stripped = raw.rstrip(trailing_punct)
            trailing = raw[len(stripped) :]
            path = stripped
        else:
            path = raw
            trailing = ""

        # The regex guarantees a path separator in raw, and rstrip cannot
        # remove separators, so we only need the length check here.
        if len(path) < 2:
            return raw
        if "://" in path:
            return raw
        if self.non_path_match(path):
            return raw
        if not Path(path).exists():
            return raw

        return f"`{path.replace('\\', '/')}`{trailing}"


def _sanitize_text(text: str) -> str:
    """Apply normalize_encoding, remove_meaningless_symbols, and
    remove_redundant_whitespace with a single code-block extraction.
    """
    text, placeholders = _extract_code(text)

    # From normalize_encoding — NFKC converts full-width ASCII (U+FF01-U+FF5E)
    # to half-width (U+0021-U+007E) and full-width space (U+3000) to regular space.
    text = unicodedata.normalize("NFKC", text)

    # Traditional to Simplified (optional, lazy) — only imported when used.
    text = _to_simplified(text)

    # From remove_meaningless_symbols
    trans = str.maketrans("", "", _ZW_CHARS)
    text = text.translate(trans)
    text = _remove_emoji(text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    # From remove_redundant_whitespace – keep newlines, collapse horizontal
    # whitespace only.
    text = re.sub(r"[^\S\n]+", " ", text)
    text = text.strip()
    return _restore_code(text, placeholders)


def escape_file_paths(
    text: str,
    *,
    max_chars: int = 0,
    max_repeat: int = 100,
    truncate_msg: str = "",
    case_mode: str = "",
) -> str:
    """Detect legal file paths in *text* and wrap each one in backticks,
    then sanitize the result to prevent ``tokenization failed`` errors.

    Paths that are already wrapped in quotes or backticks are left untouched.
    URLs, pure fractions and bare dates are ignored.

    This function also merges the behavior of *remove_meaningless_symbols*,
    *normalize_encoding*, and *remove_redundant_whitespace*.
    *case_mode* can be set to ``'lower'`` or ``'title'`` to apply
    `normalize_case` as well.
    """
    if not isinstance(text, str):
        text = str(text)

    # Escape file paths
    if "/" in text or "\\" in text:
        text = _PATH_RE.sub(_Replacer(text), text)

    # Sanitize for tokenizer
    text = _strip_invalid_unicode(text)
    text = clean_text(text, keep_newlines=True)
    text = _dedupe_repeats(text, max_repeat=max_repeat)

    # Merge additional text normalizations with single code-block extraction
    text = _sanitize_text(text)

    if case_mode:
        text, placeholders = _extract_code(text)
        if case_mode == "lower":
            text = text.lower()
        elif case_mode == "title":
            text = text.title()
        text = _restore_code(text, placeholders)

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        if truncate_msg:
            if len(truncate_msg) < max_chars:
                text = text[: max_chars - len(truncate_msg)] + truncate_msg

    return text.strip()


# ---- helpers for text cleaning ----

def _remove_emoji(text: str) -> str:
    """Remove emoji characters using rich's emoji codepoint database.

    Uses a lazily-built ``str.translate()`` table from
    ``rich._emoji_codes.EMOJI``, covering all named emoji, regional
    indicators, skin-tone modifiers, VS-16, and ZWJ — so multi-codepoint
    sequences (flags, ZWJ families, skin-tone variants) are fully removed.

    More maintainable than a hardcoded regex — the codepoint set auto-updates
    when rich is upgraded.
    """
    if not text:
        return text
    if _EMOJI_TRANSLATE is None:
        _init_emoji_translate()
    return text.translate(_EMOJI_TRANSLATE)


_EMOJI_TRANSLATE: dict[int, None] | None = None


def _init_emoji_translate() -> None:
    """Build the emoji-strip translate table from rich's emoji database.

    Only codepoints >= 0x2000 are included to avoid stripping ASCII digits,
    ``#``, ``*``, ``©``, ``®`` (which appear as base chars for keycap/sequence
    emoji in rich's data but must not be removed from general text).
    """
    global _EMOJI_TRANSLATE
    from rich._emoji_codes import EMOJI
    codepoints: set[int] = set()
    for char in EMOJI.values():
        for c in char:
            cp = ord(c)
            if cp >= 0x2000:
                codepoints.add(cp)
    _EMOJI_TRANSLATE = {cp: None for cp in codepoints}

_PUNCT_CHARS = r"!?.。，,、;；:：…~～·\"\"''（）()【】\[\]{}《》<>「」『』〖〗｛｝［］\\|｜—–―"
_REPEAT_PUNCT_RE = re.compile(
    r"([" + _PUNCT_CHARS + r"])"
    r"[" + _PUNCT_CHARS + r"]{2,}"
)

_ZW_CHARS = "".join(
    chr(c)
    for c in (
        0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060, 0x00AD,
        0x034F, 0x180B, 0x180C, 0x180D,
        0xFE00, 0xFE01, 0xFE02, 0xFE03, 0xFE04, 0xFE05,
        0xFE06, 0xFE07, 0xFE08, 0xFE09, 0xFE0A, 0xFE0B,
        0xFE0C, 0xFE0D, 0xFE0E, 0xFE0F,
    )
)


def _extract_code(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract markdown fenced code blocks and inline code into placeholders.

    Returns ``(text, [(token, original), ...])`` where each ``token`` is a
    unique digit-only null-delimited string (UUID-derived) that survives the
    normalization passes (NFKC, case folds, whitespace collapse) without
    colliding with real content.
    """
    pairs: list[tuple[str, str]] = []

    def _repl(m: re.Match[str]) -> str:
        token = f"\x00{_uuid.uuid4().int:032d}\x00"
        pairs.append((token, m.group(0)))
        return token

    text = re.sub(r"```[\s\S]*?```", _repl, text)
    text = re.sub(r"`[^`]*`", _repl, text)
    return text, pairs


def _restore_code(text: str, placeholders: list[tuple[str, str]]) -> str:
    """Restore placeholders to original code blocks."""
    for token, original in placeholders:
        text = text.replace(token, original, 1)
    return text


_opencc_converter = None
_opencc_checked = False


def _to_simplified(text: str) -> str:
    """Convert Traditional → Simplified Chinese when ``opencc`` is installed.

    Lazily probes the optional dependency once per process (module-level flag),
    avoiding a try/except import on every sanitize call when it is not present.
    """
    global _opencc_converter, _opencc_checked
    if not _opencc_checked:
        _opencc_checked = True
        try:
            import opencc  # type: ignore[import-not-found]

            _opencc_converter = opencc.OpenCC("t2s")
        except ImportError:
            _opencc_converter = None
    if _opencc_converter is not None:
        return _opencc_converter.convert(text)
    return text


