"""Behavior-equivalence gate for the ADDITIONAL wired kernels.

Covers the kernels integrated from the "Additional shim modules" table of
docs/NATIVE_INTEGRATION.md:

- DIFF  -> kimi_cli.utils.diff.format_unified_diff / _build_diff_blocks_sync
- IMAGE -> kimi_cli.utils.image_compress.format_byte_size /
           sniff_image_dimensions / _is_animated_webp
- CODEC -> kimi_cli.wire.server._frame_jsonrpc / kimi_cli.wire.file._dump_line
- TODO  -> kimi_cli.tools.todo.TodoList._status_counts
- SOUL  -> kimi_cli.soul.context_pruning.ContextPruner.prune (native
           prune_history fast path vs pure-Python candidate selection)

Every case runs the SAME inputs through the native path (gate forced on) and
the pure-Python path (gate forced off) and asserts identical results. The
native extension is optional: when it is unavailable the suite skips.
"""

from __future__ import annotations

import random

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
# IMAGE kernel
# ---------------------------------------------------------------------------


def _png(w: int, h: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + w.to_bytes(4, "big") + h.to_bytes(4, "big")


def _gif(w: int, h: int) -> bytes:
    return b"GIF89a" + w.to_bytes(2, "little") + h.to_bytes(2, "little")


def _jpeg(width: int = 320, height: int = 240) -> bytes:
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + sof + b"\xff\xd9"


IMAGE_SNIFF_CASES = [
    _png(100, 50),
    _gif(16, 9),
    b"BM" + b"\x00" * 16 + (33).to_bytes(4, "little", signed=True) + (-22).to_bytes(4, "little", signed=True) + b"\x00" * 4,
    b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8 " + b"\x00" * 10 + (80).to_bytes(2, "little") + (60).to_bytes(2, "little"),
    _jpeg(),
    b"",
    b"garbage",
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (100).to_bytes(4, "big"),  # truncated
]


def test_image_format_byte_size_equivalence():
    from kimi_cli.utils import image_compress as mod

    for n in (0, 1, 1023, 1024, 1536, 2048, 5 * 1024, 1048576, 10 * 1024 * 1024):
        restore = _native_on(mod, True)
        try:
            native = mod.format_byte_size(n)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod.format_byte_size(n)
        finally:
            restore()
        assert native == python, n


def test_image_sniff_dimensions_equivalence():
    from kimi_cli.utils import image_compress as mod

    for data in IMAGE_SNIFF_CASES:
        restore = _native_on(mod, True)
        try:
            native = mod.sniff_image_dimensions(data)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod.sniff_image_dimensions(data)
        finally:
            restore()
        assert native == python, data[:20]


def test_image_is_animated_webp_equivalence():
    from kimi_cli.utils import image_compress as mod

    anim = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" + b"\x00" * 4 + b"\x02\x00\x00\x00" + b"\x00" * 4
    still = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" + b"\x00" * 4 + b"\x00\x00\x00\x00" + b"\x00" * 4
    for data in (anim, still, b"", _png(1, 1)):
        restore = _native_on(mod, True)
        try:
            native = mod._is_animated_webp(data)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod._is_animated_webp(data)
        finally:
            restore()
        assert native == python, data[:20]


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


# ---------------------------------------------------------------------------
# TODO kernel
# ---------------------------------------------------------------------------


def test_todo_status_counts_equivalence():
    from kimi_cli.tools.todo import Todo, TodoList

    rng = random.Random(7)
    statuses = ("pending", "in_progress", "done")
    todos = [
        Todo(title=f"T{i}", status=statuses[rng.randrange(3)])
        for i in range(rng.randrange(0, 20))
    ]
    if not todos:
        todos = [Todo(title="T0", status="pending")]
    restore = _native_on(TodoList, True)
    try:
        native = TodoList._status_counts(todos)
    finally:
        restore()
    restore = _native_on(TodoList, False)
    try:
        python = TodoList._status_counts(todos)
    finally:
        restore()
    assert native == python
    assert native["pending"] + native["in_progress"] + native["done"] == len(todos)


# ---------------------------------------------------------------------------
# SOUL kernel (context pruning native fast path)
# ---------------------------------------------------------------------------


def test_context_prune_native_vs_python():
    """ContextPruner.prune with the SOUL kernel_module available (native
    prune_history) vs unavailable (pure-Python candidate selection) yields the
    same drop decisions."""
    from kosong.message import Message
    from kimi_cli.wire.types import TextPart

    import kimi_cli.soul.context_pruning as mod
    from kimi_cli.soul.context_pruning import ContextPruner

    def msg(role: str, text: str, tool_call_id: str | None = None) -> Message:
        return Message(role=role, content=[TextPart(text=text)], tool_call_id=tool_call_id)

    history = [
        msg("system", "You are a helpful assistant."),
        msg("user", "<system-reminder>\ninternal note\n</system-reminder>"),
        msg("user", "What is the capital of France?"),
        msg("assistant", "The capital is Paris.", tool_call_id=None),
        msg("tool", "Paris", tool_call_id="call_1"),
        msg("user", "x" * 4000),  # oversized tool-output-like user text
        msg("user", "Thank you."),
    ]
    pruner = ContextPruner(
        stable_prefix_messages=1,
        recent_messages_protected=2,
        trigger_ratio=0.0,
        min_free_tokens=0,
        max_fraction_per_pass=1.0,
        enabled=True,
    )
    # NOTE: ContextPruner.token accounting uses count_message_tokens; force the
    # trigger/budget so pruning actually runs regardless of usage.
    pruner._trigger_ratio = 0.0
    pruner._target_ratio = 0.1

    original_kernel_module = mod.kernel_module
    try:
        # Native path: kernel_module("SOUL") returns the real shim submodule.
        mod.kernel_module = lambda kernel: original_kernel_module(kernel)
        pruner_native = ContextPruner(
            stable_prefix_messages=1,
            recent_messages_protected=2,
            trigger_ratio=0.0,
            min_free_tokens=0,
            max_fraction_per_pass=1.0,
            enabled=True,
        )
        pruner_native._trigger_ratio = 0.0
        pruner_native._target_ratio = 0.1
        native = pruner_native.prune(history, current_step=1, context_usage=0.99, max_context_size=8000)
        # Python path: kernel_module returns None (gate off); a FRESH pruner
        # avoids cross-run hysteresis/cooldown state.
        mod.kernel_module = lambda kernel: None
        pruner_python = ContextPruner(
            stable_prefix_messages=1,
            recent_messages_protected=2,
            trigger_ratio=0.0,
            min_free_tokens=0,
            max_fraction_per_pass=1.0,
            enabled=True,
        )
        pruner_python._trigger_ratio = 0.0
        pruner_python._target_ratio = 0.1
        python = pruner_python.prune(history, current_step=2, context_usage=0.99, max_context_size=8000)
    finally:
        mod.kernel_module = original_kernel_module

    assert [m.role for m in native.messages] == [m.role for m in python.messages]
    assert [len(m.content) for m in native.messages] == [len(m.content) for m in python.messages]
    assert native.freed_tokens == python.freed_tokens
    assert native.earliest_removed_index == python.earliest_removed_index
    assert [r.index for r in native.elided] == [r.index for r in python.elided]
