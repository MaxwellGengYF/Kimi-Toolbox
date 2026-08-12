"""Behavior-equivalence gate for the ADDITIONAL wired kernels.

Covers the kernels integrated from the "Additional shim modules" table of
docs/NATIVE_INTEGRATION.md:

- DIFF  -> kimi_cli.utils.diff.format_unified_diff / _build_diff_blocks_sync
- CODEC -> kimi_cli.wire.server._frame_jsonrpc / kimi_cli.wire.file._dump_line

Every case runs the SAME inputs through the native path (gate forced on) and
the pure-Python path (gate forced off) and asserts identical results. The
native extension is optional: when it is unavailable the suite skips.
"""

from __future__ import annotations

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)


def _native_on(module, state: bool):
    """Force the module's native gate to *state*; returns restore callable."""
    attr = "_native_use_native"
    original = getattr(module, attr, None)
    setattr(module, attr, lambda kernel: state)
    return lambda: setattr(module, attr, original) if original is not None else delattr(
        module, attr
    )


# ---------------------------------------------------------------------------
# DIFF kernel
# ---------------------------------------------------------------------------

DIFF_CASES = [
    (b"", b""),
    (b"a\nb\n", b"a\nb\n"),
    (b"a\nb\n", b"a\nb\nc\n"),
    (b"line1\nline2\nline3\n", b"line1\nline2-X\nline3\n"),
    (b"one\ntwo\nthree\nfour\nfive\n", b"one\ntwo\nthree\nTHREE\nfive\n"),
    (b"a\nb\nc\nd\ne\nf\ng\nh\n", b"a\nb\nX\nX\ne\nf\ng\nh\n"),
    (b"no trailing newline", b"no trailing newline"),
    (b"no trailing newline", b"no trailing newline\n"),
    (b"x", b"y"),
    ("caf\u00e9 \u4e16\u754c\n\u6d4b\u8bd5\n".encode("utf-8"),
     "caf\u00e9 \u4e16\u754c\n\u6d4b\u8bd5!\n".encode("utf-8")),
]


def test_format_unified_diff_equivalence():
    from kimi_cli.utils import diff as mod

    for old, new in DIFF_CASES:
        old_s = old.decode("utf-8", "surrogatepass")
        new_s = new.decode("utf-8", "surrogatepass")
        for path in ("", "file.txt"):
            for header in (True, False):
                restore = _native_on(mod, True)
                try:
                    native = mod.format_unified_diff(old_s, new_s, path, include_file_header=header)
                finally:
                    restore()
                restore = _native_on(mod, False)
                try:
                    python = mod.format_unified_diff(old_s, new_s, path, include_file_header=header)
                finally:
                    restore()
                assert native == python, (old, new, path, header)


def test_build_diff_blocks_equivalence():
    from kimi_cli.utils import diff as mod

    for old, new in DIFF_CASES:
        old_s = old.decode("utf-8", "surrogatepass")
        new_s = new.decode("utf-8", "surrogatepass")
        restore = _native_on(mod, True)
        try:
            native = mod._build_diff_blocks_sync("p.txt", old_s, new_s)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod._build_diff_blocks_sync("p.txt", old_s, new_s)
        finally:
            restore()
        assert native == python, (old, new)


# ---------------------------------------------------------------------------
# CODEC kernel
# ---------------------------------------------------------------------------


def test_wire_frame_jsonrpc_equivalence():
    from kimi_cli.wire import server as mod

    payloads = [
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
        b"",
        b"{}",
        '{"text":"caf\u00e9 \u4e16\u754c"}'.encode("utf-8"),
    ]
    for payload in payloads:
        restore = _native_on(mod, True)
        try:
            native = mod._frame_jsonrpc(payload)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod._frame_jsonrpc(payload)
        finally:
            restore()
        assert native == python == payload + b"\n", payload


def test_wire_file_dump_line_equivalence():
    from pydantic import BaseModel

    from kimi_cli.wire import file as mod

    class _M(BaseModel):
        x: int

    restore = _native_on(mod, True)
    try:
        native = mod._dump_line(_M(x=1))
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._dump_line(_M(x=1))
    finally:
        restore()
    assert native == python == '{"x":1}\n'
