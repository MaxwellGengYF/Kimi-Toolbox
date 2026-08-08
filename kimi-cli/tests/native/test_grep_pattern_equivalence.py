"""Behavior-equivalence gate for the grep newline-pattern kernels (plan:
commit 0582e09 "Study from hermes").

Runs the SAME patterns through the native path (gate forced on) and the
pure-Python path (gate forced off) and asserts identical results for
kimi_cli.tools.file.grep_local._pattern_has_regex_newline and
_multiline_pattern.

Corpora cover odd/even backslash runs, real newlines, CRLF, empty patterns
and non-ASCII patterns (which route to the Python bodies by construction).
"""

from __future__ import annotations

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

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


def _assert_equivalent(native_result, python_result, case):
    assert native_result == python_result, (
        f"native != python for {case!r}:\n"
        f"  native={native_result!r}\n  python={python_result!r}"
    )


PATTERN_CORPUS = [
    "abc",
    "a\nb",
    "a\\nb",
    "a\\\\nb",
    "a\\\\\\nb",
    "a\nb\\nc",
    "\\n",
    "\\\\n",
    "\\\\\\n",
    "x\\ny",
    "a\r\nb",
    "a\r\nb\\nc",
    "\\\\\\n\\n",
    "a\\\\\\\\nb",
    "a\\\\\\\\\\nb",
    "\\n\\n\\n",
    "",
    "\u00e9\\n",  # non-ASCII routes to the Python body
    "\u00e9\nx",
]


@pytest.mark.parametrize("pattern", PATTERN_CORPUS)
def test_pattern_has_regex_newline_equivalence(pattern):
    from kimi_cli.tools.file import grep_local as mod

    restore = _force_gate(mod, True)
    try:
        native = mod._pattern_has_regex_newline(pattern)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod._pattern_has_regex_newline(pattern)
    finally:
        restore()
    _assert_equivalent(native, python, pattern)


@pytest.mark.parametrize("pattern", PATTERN_CORPUS)
def test_multiline_pattern_equivalence(pattern):
    from kimi_cli.tools.file import grep_local as mod

    restore = _force_gate(mod, True)
    try:
        native = mod._multiline_pattern(pattern)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod._multiline_pattern(pattern)
    finally:
        restore()
    _assert_equivalent(native, python, pattern)
