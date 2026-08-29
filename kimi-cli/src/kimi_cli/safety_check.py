from kimi_cli.native_loader import (
    get_compat as _native_get_compat,
    get_module as _native_get_module,
    use_native as _native_use_native,
)

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TEXT = _native_get_module("text")
# Pure-Python reference implementation (canonical copy lives in the shim);
# resolved lazily to avoid an import-time dependency on the shim package.
_COMPAT_TEXT = None


def _compat_text():
    global _COMPAT_TEXT
    if _COMPAT_TEXT is None:
        _COMPAT_TEXT = _native_get_compat("text")
    return _COMPAT_TEXT

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
    if _native_use_native("TEXT") and _NATIVE_TEXT is not None:
        return _NATIVE_TEXT.clean_text(text, keep_newlines)
    return _compat_text()._compat_clean_text(text, keep_newlines)


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
    if _native_use_native("TEXT") and _NATIVE_TEXT is not None:
        return _NATIVE_TEXT.sanitize_for_tokenizer(
            text,
            max_chars=max_chars,
            max_repeat=max_repeat,
            truncate_msg=truncate_msg,
        )
    return _compat_text()._compat_sanitize_for_tokenizer(
        text,
        max_chars=max_chars,
        max_repeat=max_repeat,
        truncate_msg=truncate_msg,
    )
