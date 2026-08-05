import regex as re
import unicodedata

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

"""
Text safety utilities: clean hidden/invisible characters and prevent tokenization failures.
"""


def clean_text(text: str, keep_newlines: bool = True) -> str:
    """
    Remove invisible/hidden characters from text.

    Targets:
    - Zero-width characters (\u200b, \u200c, \u200d, \ufeff, \u2060, etc.)
    - PDF/Word hidden format characters
    - Most C0/C1 control characters
    - Soft hyphens, directional marks, override chars

    Args:
        text: Raw input string.
        keep_newlines: If True, preserves \\n, \\r, \\t.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        text = str(text)

    # Native acceleration: kimix_native.text.clean_text (bit-identical).
    if _native_use_native("TEXT"):
        _mod = _native_get_module("text")
        if _mod is not None:
            return _mod.clean_text(text, keep_newlines)

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


# ---------------------------------------------------------------------------
# Tokenization-safety helpers
# ---------------------------------------------------------------------------

def _strip_surrogates(text: str) -> str:
    """Remove lone surrogate code points."""
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def _strip_noncharacters(text: str) -> str:
    """Remove Unicode noncharacters."""

    def _keep(ch: str) -> bool:
        cp = ord(ch)
        if 0xFDD0 <= cp <= 0xFDEF:
            return False
        return (cp & 65535) not in (65534, 65535)

    return "".join(ch for ch in text if _keep(ch))


def _strip_pua(text: str) -> str:
    """Remove Private Use Area code points."""

    def _keep(ch: str) -> bool:
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            return False
        if 0xF0000 <= cp <= 0xFFFFD:
            return False
        return not 1048576 <= cp <= 1114109

    return "".join(ch for ch in text if _keep(ch))


def _strip_replacement_chars(text: str) -> str:
    """Remove Unicode replacement characters (sign of prior encoding corruption)."""
    return text.replace("\ufffd", "")


def _dedupe_repeats(text: str, max_repeat: int = 100) -> str:
    """
    Collapse runs of a single character longer than *max_repeat*.
    Prevents pathological inputs from exploding tokenizer buffers.
    """
    if max_repeat <= 0:
        return text
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", lambda m: m.group(1) * max_repeat, text)


def sanitize_for_tokenizer(
    text: str,
    *,
    max_chars: int = 0,
    max_repeat: int = 100,
    truncate_msg: str = "",
) -> str:
    """
    Aggressively sanitize text to prevent ``tokenization failed`` errors.

    Rules applied (in order):
    1. Coerce to ``str``.
    2. Remove surrogates (U+D800-U+DFFF) – invalid Unicode scalars.
    3. Remove noncharacters (U+FDD0-U+FDEF, U+FFFE, U+FFFF, …).
    4. Remove Private Use Area (PUA) glyphs – tokenizers have no vocab for them.
    5. Collapse consecutive replacement chars (U+FFFD).
    6. Run :func:`clean_text` (zero-width chars, controls, NFC).
    7. Collapse extreme character repetition (e.g. ``"A" * 10_000``).
    8. Truncate to *max_chars* if > 0.
    9. Strip leading/trailing whitespace.

    Args:
        text: Raw input.
        max_chars: Hard truncation limit (0 = disabled).
        max_repeat: Maximum allowed consecutive identical chars (0 = disabled).
        truncate_msg: Optional suffix appended when truncation occurs.

    Returns:
        Sanitized string safe for tokenizer ingestion.
    """
    if not isinstance(text, str):
        text = str(text)

    # Native acceleration: kimix_native.text.sanitize_for_tokenizer.
    if _native_use_native("TEXT"):
        _mod = _native_get_module("text")
        if _mod is not None:
            return _mod.sanitize_for_tokenizer(
                text,
                max_chars=max_chars,
                max_repeat=max_repeat,
                truncate_msg=truncate_msg,
            )

    # 2. Strip surrogates (invalid scalar values)
    text = _strip_surrogates(text)

    # 3. Strip noncharacters
    text = _strip_noncharacters(text)

    # 4. Strip PUA characters
    text = _strip_pua(text)

    # 5. Strip replacement chars
    text = _strip_replacement_chars(text)

    # 6. Standard clean (zero-width, controls, NFC)
    text = clean_text(text, keep_newlines=True)

    # 7. Deduplicate extreme repeats
    text = _dedupe_repeats(text, max_repeat=max_repeat)

    # 8. Truncate if requested
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        if truncate_msg and len(truncate_msg) < max_chars:
            text = text[: max_chars - len(truncate_msg)] + truncate_msg

    # 9. Final strip
    return text
