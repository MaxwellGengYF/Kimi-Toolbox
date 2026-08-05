"""Shared input corpus for native vs pure-Python equivalence tests.

Every kernel equivalence case in ``test_behavior_equivalence.py`` draws its
inputs from here. The corpus is deliberately adversarial: empty input, pure
ASCII, CJK/mixed Unicode, emoji, lone surrogates, control/zero-width chars,
boundary sizes, malformed inputs and realistic workloads.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TEXT kernel corpus (tokens + sanitize)
# ---------------------------------------------------------------------------

TEXT_CORPUS = [
    "",
    "hello world",
    "The quick brown fox jumps over the lazy dog. 1234567890",
    "a" * 1,
    "b" * 200,
    "c" * 4096,
    "中文测试文本内容，用于验证启发式分词。",
    "mixed 中文 with ASCII 123 and punctuation!",
    "emoji 🎉🎊✨ mixed with text",
    "zero\u200bwidth\u200dchars\ufeffhere",
    "\ufffd\ufffd\ufffd replacement chars",
    "x" * 100 + "y" * 300 + "z" * 500,
    "tab\there\r\ncrlf\rlone-cr",
    "lone \ud800 surrogate \udfff pair",
    "ctrl\x00\x01\x08\x0b\x0c\x0e\x1f\x7f\x9f chars",
    "A" * 1000 + "B" * 2 + "C" * 1000,
    "nfc \u00e9 composed vs \u0065\u0301 decomposed",
    "pua \ue000\uf8ff\U000f0000 chars",
    "fullwidth\uff01\uff02\uff03 punctuation",
    "hangul \uac00\uac01\uac02 syllables",
    "hiragana \u3042\u3044\u3046 katakana \u30a2\u30a4\u30a6",
    "leading   multiple   spaces   inside",
    "caf\u00e9 r\u00e9sum\u00e9 na\u00efve \u2014 dash",
]

# max_chars / max_repeat boundary cases
SANITIZE_BOUNDARY_CASES = [
    ("", 0, 100, ""),
    ("abc", 3, 100, ""),
    ("abcde", 3, 100, ""),
    ("abcde", 5, 100, ""),
    ("abcde", 6, 100, ""),
    ("a" * 300, 0, 100, ""),
    ("a" * 300, 200, 100, ""),
    ("a" * 300, 200, 100, "..."),
    ("a" * 300, 200, 100, "TRUNCATED_SUFFIX"),
    ("b" * 5000, 0, 50, ""),
    ("b" * 5000, 0, 0, ""),
    ("mixed\u4e2d\u6587" * 50, 100, 10, "[cut]"),
]

# ---------------------------------------------------------------------------
# Kernel registration table (extended as kernels are integrated)
# ---------------------------------------------------------------------------

# Each entry: (kernel, [list of callables invoked with a single corpus item])
KERNEL_CASES: dict[str, list] = {}
