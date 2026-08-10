"""Behavior-equivalence gate for the STREAM kernel in src/kimix/ui/printing.py.

Runs the SAME corpus through ``_strip_ansi`` with the gate forced on (native
kimix_native.stream.strip_ansi) vs off (the pure-Python ``_ANSI_ESCAPE``
substitution) and asserts identical results. The corpus is deliberately
adversarial: ANSI colors, OSC/DCS/APC sequences, malformed escapes, plain
text, and text ending in a bare ESC byte.
"""

from __future__ import annotations

import pytest

from kimix.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)


def _force_gate(module, state: bool):
    attr = "_native_use_native"
    original = getattr(module, attr, None)
    setattr(module, attr, lambda kernel: state)
    return lambda: setattr(module, attr, original) if original is not None else delattr(
        module, attr
    )


STRIP_ANSI_CORPUS = [
    # plain text / empty
    "",
    "plain text no escapes",
    "a" * 4096,
    "line1\nline2\nline3",
    # ANSI colors (SGR)
    "\x1b[31mred\x1b[0m",
    "x\x1b[1;32mbold\x1b[0my",
    "\x1b[1m\x1b[31m\x1b[0m\x1b[0m",
    "\x1b[38;2;255;0;0mtrue color\x1b[0m",
    "\x1b[38;5;196m256 color\x1b[0m",
    "\x1b[K partial clear",
    "\x1b[?25lhide cursor",
    "\x1b[2J\x1b[Hclear screen",
    # OSC sequences (BEL and ST terminators)
    "\x1b]0;title\x07OSC title",
    "\x1b]0;abc\x07def",
    "\x1b]8;;http://example.com\x1b\\link\x1b]8;;\x1b\\",
    "\x1b]2;tab title\x07rest",
    "\x1b]133;A\x07osc marker",
    # DCS / PM / APC sequences
    "\x1bPabc\x1b\\dcs",
    "\x1b^apc\x1b\\",
    # single-character Fe sequences
    "\x1b_intermediate",
    "\x1b\x1b[31m",
    # malformed escapes / bare ESC
    "malformed \x1b[31",
    "trailing escape \x1b",
    "\x1b",
    "a\x1b",
    "end with esc \x1b",
    # unicode / emoji mixed with ANSI
    "中文\x1b[33m混合\x1b[0m color",
    "emoji 🎉 \x1b[34mblue\x1b[0m",
    "nfc \u00e9 vs \u0065\u0301 \x1b[35m\x1b[0m",
    # realistic multi-line output
    "line1\n\x1b[31mline2\x1b[0m\nline3",
    "x" * 5000 + "\x1b[31mred\x1b[0m",
    "\r\n\x1b[1mwin\r\ncrlf\x1b[0m\r\n",
    # ANSI with numbers/params edge cases
    "\x1b[31;1mcombined\x1b[0m",
    "\x1b[mbare reset",
    "\x1b[0;31;42mnested params\x1b[m",
]


@pytest.mark.parametrize("text", STRIP_ANSI_CORPUS)
def test_strip_ansi_equivalence(text):
    import kimix.ui.printing as mod

    restore = _force_gate(mod, True)
    try:
        native = mod._strip_ansi(text)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod._strip_ansi(text)
    finally:
        restore()
    assert native == python, (
        f"_strip_ansi native != python for {text!r}:\n"
        f"  native={native!r}\n  python={python!r}"
    )


def test_strip_ansi_no_esc_fast_path():
    """Text without ESC must be returned unchanged regardless of the gate."""
    import kimix.ui.printing as mod

    restore = _force_gate(mod, True)
    try:
        assert mod._strip_ansi("plain text") == "plain text"
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        assert mod._strip_ansi("plain text") == "plain text"
    finally:
        restore()


def test_ends_with_newline_equivalence():
    """The _strip_ansi consumer (_ends_with_newline) must not change behavior."""
    import kimix.ui.printing as mod

    corpus = [
        "word",
        "word\n",
        "word\x1b[31m\n\x1b[0m",
        "a\x1b[31m",
        "\x1b[0m\n",
        "plain\n\x1b[31mred\x1b[0m",
        "\x1b]0;title\x07\n",
        "",
    ]
    for word in corpus:
        restore = _force_gate(mod, True)
        try:
            native = mod._ends_with_newline(word)
        finally:
            restore()
        restore = _force_gate(mod, False)
        try:
            python = mod._ends_with_newline(word)
        finally:
            restore()
        assert native == python, f"_ends_with_newline mismatch for {word!r}"
