from __future__ import annotations

import regex as re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)

if TYPE_CHECKING:
    from kosong.message import Message


# Regex to detect CJK characters
_CJK_RE = re.compile(
    "[\u4e00-\u9fff"
    "\u3400-\u4dbf"
    "\U00020000-\U0002ebef"
    "\uac00-\ud7af"
    "\u3040-\u309f"
    "\u30a0-\u30ff"
    "\uff00-\uffef]"
)


def _is_cjk_text(text: str, threshold: float = 0.15) -> bool:
    """Return True if the fraction of CJK characters exceeds *threshold*."""
    # Native acceleration: kimix_native.text.is_cjk_text (bit-identical).
    if _native_use_native("TEXT"):
        _mod = _native_get_module("text")
        if _mod is not None:
            return _mod.is_cjk_text(text, threshold)
    if not text:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count / len(text) > threshold


def _estimate_chars_tokens(text: str) -> int:
    """Language-aware character heuristic.

    - English / mostly-ASCII  → ~4 chars per token
    - CJK-detected text       → ~3 chars per token (closer to reality for
      ideographic languages where each character is often its own token)
    - Mixed / code            → ~3.5 chars per token (split the difference)
    """
    if not text:
        return 0
    # Native acceleration: kimix_native.text.estimate_chars_tokens.
    if _native_use_native("TEXT"):
        _mod = _native_get_module("text")
        if _mod is not None:
            return _mod.estimate_chars_tokens(text)
    total = len(text)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / total

    if ascii_ratio > 0.95:
        return max(1, total // 4)
    if _is_cjk_text(text):
        return max(1, total // 3)
    # Mixed / code
    return max(1, int(total / 3.5))


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in *text* using the best available method.

    If ``tiktoken`` is installed and *model* is provided, the model-specific
    encoding is used.  Otherwise falls back to a language-aware character
    heuristic that is more accurate than a flat ``len(text) // 4``.
    """
    # Attempt tiktoken only when the package is present and a model is hinted.
    if model:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return _estimate_chars_tokens(text)


def count_message_tokens(messages: Sequence[Message], model: str | None = None) -> int:
    """Estimate tokens for a sequence of messages.

    Sums tokens from all :class:`TextPart` content in each message.
    """
    from kimi_cli.wire.types import TextPart

    total = 0
    for msg in messages:
        for part in msg.content:
            if isinstance(part, TextPart):
                total += count_tokens(part.text, model=model)
    return total
