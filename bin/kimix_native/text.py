"""kimix_native.text — text kernels: heuristic token count + sanitizer.

Native implementations live in ``runtime_py.text`` (compiled kernels, GIL
released). The pure-Python ``_compat`` functions below mirror the reference
algorithms exactly:

- ``_estimate_chars_tokens`` / ``_is_cjk_text`` — kimi_cli/utils/tokens.py
- ``sanitize_for_tokenizer`` / ``clean_text`` — kimi_cli/safety_check.py

NFC hook (plan 002, v1): the C++ kernel implements steps 2-5 + 6a/6b
(``sanitize_pre_nfc``) and 6d+7+8 (``sanitize_post_nfc``) but NOT the NFC
normalization (step 6c). This shim runs ``unicodedata.normalize("NFC", …)``
between them, only when non-ASCII survived (ASCII fast path: NFC is identity,
so pure-ASCII input takes a single native call ``sanitize_for_tokenizer``).

Encoding contract: the native kernels are bytes-in/bytes-out. Strings are
encoded with ``errors="surrogatepass"`` so lone surrogates (invalid Unicode
scalars that Python ``str`` can hold) reach the kernel as CESU-8-like bytes;
the kernel's step-2 range check removes them — exactly like the Python
``_strip_surrogates`` pass. Strict encoding would raise on lone surrogates.
"""

from __future__ import annotations

import unicodedata

from . import _native, use_native

# ---------------------------------------------------------------------------
# _compat — exact mirrors of the Python reference implementations
# ---------------------------------------------------------------------------

# The 7 contiguous ranges of _CJK_RE (kimi_cli/utils/tokens.py).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),      # CJK Unified Ideographs
    (0x3400, 0x4DBF),      # CJK Extension A
    (0x20000, 0x2EBEF),    # CJK Extension B
    (0xAC00, 0xD7AF),      # Hangul Syllables
    (0x3040, 0x309F),      # Hiragana
    (0x30A0, 0x30FF),      # Katakana
    (0xFF00, 0xFFEF),      # Fullwidth Forms
)


def _compat_is_cjk_cp(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _compat_is_cjk_text(text: str, threshold: float = 0.15) -> bool:
    if not text:
        return False
    cjk_count = sum(1 for c in text if _compat_is_cjk_cp(ord(c)))
    return cjk_count / len(text) > threshold


def _compat_estimate(text: str) -> int:
    """Mirror of tokens.py::_estimate_chars_tokens."""
    if not text:
        return 0
    total = len(text)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / total
    if ascii_ratio > 0.95:
        return max(1, total // 4)
    if _compat_is_cjk_text(text):
        return max(1, total // 3)
    return max(1, int(total / 3.5))


# Zero-width / invisible format chars removed by clean_text step 1.
_ZERO_WIDTH = frozenset(
    [0x200B, 0x200C, 0x200D, 0x2060, 0x00AD, 0xFEFF, 0x200E, 0x200F]
    + list(range(0x202A, 0x202F))  # \u202a-\u202e
    + list(range(0x2066, 0x206A))  # \u2066-\u2069
)


def _compat_clean_text(text: str, keep_newlines: bool = True) -> str:
    """Mirror of safety_check.py::clean_text."""
    if not isinstance(text, str):
        text = str(text)
    text = "".join(ch for ch in text if ord(ch) not in _ZERO_WIDTH)
    if keep_newlines:
        text = "".join(
            ch
            for ch in text
            if not (
                ord(ch) <= 0x08
                or ord(ch) in (0x0B, 0x0C)
                or 0x0E <= ord(ch) <= 0x1F
                or 0x7F <= ord(ch) <= 0x9F
            )
        )
    else:
        text = "".join(
            ch for ch in text if not (ord(ch) <= 0x1F or 0x7F <= ord(ch) <= 0x9F)
        )
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def _compat_strip_surrogates(text: str) -> str:
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def _compat_strip_noncharacters(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if not (0xFDD0 <= ord(ch) <= 0xFDEF or (ord(ch) & 0xFFFF) in (0xFFFE, 0xFFFF))
    )


def _compat_strip_pua(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if not (
            0xE000 <= ord(ch) <= 0xF8FF
            or 0xF0000 <= ord(ch) <= 0xFFFFD
            or 0x100000 <= ord(ch) <= 0x10FFFD
        )
    )


def _compat_dedupe_repeats(text: str, max_repeat: int = 100) -> str:
    """Mirror of safety_check.py::_dedupe_repeats (run-collapse)."""
    if max_repeat <= 0:
        return text
    out = []
    run_ch = None
    run_len = 0
    for ch in text:
        if ch == run_ch:
            run_len += 1
        else:
            if run_ch is not None:
                out.append(run_ch * min(run_len, max_repeat))
            run_ch = ch
            run_len = 1
    if run_ch is not None:
        out.append(run_ch * min(run_len, max_repeat))
    return "".join(out)


def _compat_sanitize_for_tokenizer(
    text: str, *, max_chars: int = 0, max_repeat: int = 100, truncate_msg: str = ""
) -> str:
    """Mirror of safety_check.py::sanitize_for_tokenizer (source-faithful:
    no final strip after truncation)."""
    if not isinstance(text, str):
        text = str(text)
    text = _compat_strip_surrogates(text)
    text = _compat_strip_noncharacters(text)
    text = _compat_strip_pua(text)
    text = text.replace("\ufffd", "")
    text = _compat_clean_text(text, keep_newlines=True)
    text = _compat_dedupe_repeats(text, max_repeat=max_repeat)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        if truncate_msg and len(truncate_msg) < max_chars:
            text = text[: max_chars - len(truncate_msg)] + truncate_msg
    return text


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in *text*.

    Mirrors kimi_cli/utils/tokens.py::count_tokens: the tiktoken branch (when
    *model* is given) stays in Python; the heuristic fallback is native.
    """
    if model:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    if not use_native("TEXT") or _native is None:
        return _compat_estimate(text)
    return _native.text.count_tokens(text.encode("utf-8", "surrogatepass"))


def estimate_chars_tokens(text: str) -> int:
    if not use_native("TEXT") or _native is None:
        return _compat_estimate(text)
    return _native.text.estimate_chars_tokens(text.encode("utf-8", "surrogatepass"))


def is_cjk_text(text: str, threshold: float = 0.15) -> bool:
    if not use_native("TEXT") or _native is None:
        return _compat_is_cjk_text(text, threshold)
    return _native.text.is_cjk_text(text.encode("utf-8", "surrogatepass"), threshold)


def clean_text(text: str, keep_newlines: bool = True) -> str:
    """Mirror of safety_check.py::clean_text (zero-width, controls, NFC,
    strip)."""
    s = text if isinstance(text, str) else str(text)
    if not use_native("TEXT") or _native is None:
        return _compat_clean_text(s, keep_newlines)
    data = s.encode("utf-8", "surrogatepass")
    out = _native.text.clean_text(data, keep_newlines)
    if not out.isascii():
        out = unicodedata.normalize("NFC", out.decode("utf-8", "surrogatepass")).strip().encode(
            "utf-8", "surrogatepass"
        )
    return out.decode("utf-8", "surrogatepass")


def sanitize_for_tokenizer(
    text: str, *, max_chars: int = 0, max_repeat: int = 100, truncate_msg: str = ""
) -> str:
    """Mirror of safety_check.py::sanitize_for_tokenizer.

    Native path: pure-ASCII input takes one native call (NFC is identity);
    non-ASCII input goes through sanitize_pre_nfc -> NFC (only when non-ASCII
    survived) -> sanitize_post_nfc.
    """
    s = text if isinstance(text, str) else str(text)
    if not use_native("TEXT") or _native is None:
        return _compat_sanitize_for_tokenizer(
            s, max_chars=max_chars, max_repeat=max_repeat, truncate_msg=truncate_msg
        )
    data = s.encode("utf-8", "surrogatepass")
    msg = truncate_msg.encode("utf-8", "surrogatepass")
    if data.isascii():
        # NFC is the identity on pure ASCII -> single native call.
        out = _native.text.sanitize_for_tokenizer(data, max_chars, max_repeat, msg)
        return out.decode("utf-8", "surrogatepass")
    partial = _native.text.sanitize_pre_nfc(data)
    if not partial.isascii():
        partial = unicodedata.normalize("NFC", partial.decode("utf-8", "surrogatepass")).encode(
            "utf-8", "surrogatepass"
        )
    out = _native.text.sanitize_post_nfc(partial, max_chars, max_repeat, msg)
    return out.decode("utf-8", "surrogatepass")
