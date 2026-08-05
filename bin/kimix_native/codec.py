"""kimix_native.codec -- codec kernels: wire envelope, merge/args buffers,
JSON-RPC + jsonl frames, TCP recv buffer, SSE frames (plans 007/008/009).

Native implementations live in ``runtime_py.codec`` (compiled kernels, GIL
released, bytes in/out). The pure-Python ``_compat`` mirrors below implement
the same semantics (documented in the plan files + kernel headers):

- envelope: ``{"type": <str>, "payload": <json>}``, ONE JSON pass; payload
  key order = input insertion order (yyjson + json.dumps agree).
- canonicalize_payload: recursive object-key sort + compact re-encode
  (toolset._sort_json_value semantics).
- WireMergeBuffer: same-kind ``text``/``args`` parts merge; an empty buffer
  accepts any kind; anything else returns False (caller flushes).
- JSON-RPC frame = payload + b"\\n" (wire/server.py); jsonl record = frame
  + b"\\n" (wire/file.py _dump_line).
- RecvBuffer: BIG-ENDIAN length prefix (tcp_client.py ``int.from_bytes(...,
  "big")``), max_frame=0 -> 10 MiB default, auto-compact at >size/2.
- SSE frame = ``event: <name>\\nid: <id>\\ndata: <json>\\n\\n`` (bus.py
  to_sse): event line only when name non-empty, id line only when id != 0,
  multi-line data -> repeated ``data:`` lines.
"""

from __future__ import annotations

import json

import orjson

from . import _native, use_native

# ---------------------------------------------------------------------------
# Encoding helpers (bytes-in/bytes-out contract like the rest of the module)
# ---------------------------------------------------------------------------


def _enc(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def _dec(b: bytes) -> str:
    return b.decode("utf-8", "surrogatepass")


def _compact(obj) -> bytes:
    """orjson-fast compact JSON bytes (no spaces, raw UTF-8).

    Falls back to the stdlib serializer for values orjson rejects (lone
    surrogates, non-str keys, >64-bit ints) so the wire bytes are preserved.
    """
    try:
        return orjson.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8", "surrogatepass"
        )


# ---------------------------------------------------------------------------
# _compat -- pure-Python mirrors (same semantics as the kernels)
# ---------------------------------------------------------------------------


def _sort_json_value(value):
    if isinstance(value, dict):
        return {k: _sort_json_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_json_value(item) for item in value]
    return value


def _loads(data: bytes):
    """orjson-fast parse; stdlib fallback keeps lone-surrogate parity
    (orjson rejects CESU-8 lone surrogates with JSONDecodeError)."""
    try:
        return orjson.loads(data)
    except ValueError:
        return json.loads(data.decode("utf-8", "surrogatepass"))


def _compat_serialize_envelope(type_name: str, payload: bytes) -> bytes:
    try:
        parsed = _loads(payload)
    except ValueError:
        # Invalid payload JSON -> embed as an escaped string (kernel rule).
        text = payload.decode("utf-8", "surrogatepass")
        return _compact({"type": type_name, "payload": text})
    return _compact({"type": type_name, "payload": parsed})


def _compat_deserialize_envelope(frame: bytes):
    try:
        obj = _loads(frame)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    type_name = obj.get("type")
    payload = obj.get("payload")
    if not isinstance(type_name, str) or "payload" not in obj:
        return None
    return type_name, _compact(payload)


def _compat_canonicalize_payload(data: bytes):
    try:
        obj = _loads(data)
    except ValueError:
        return None
    return _compact(_sort_json_value(obj))


class _CompatWireMergeBuffer:
    __slots__ = ("_kind", "_parts")

    def __init__(self) -> None:
        self._kind: str | None = None
        self._parts: list[bytes] = []

    def append(self, kind: str, delta: bytes) -> bool:
        if self.empty():
            self._kind = kind
            self._parts = [bytes(delta)]
            return True
        if kind == self._kind and kind in ("text", "args"):
            self._parts.append(bytes(delta))
            return True
        return False

    def snapshot(self) -> bytes:
        return b"".join(self._parts)

    def reset(self) -> None:
        self._kind = None
        self._parts = []

    def empty(self) -> bool:
        return not self._parts


class _CompatArgsBuffer:
    __slots__ = ("_buf", "_watermark")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._watermark = 0

    def append(self, delta: bytes) -> None:
        self._buf += delta

    def snapshot(self) -> bytes:
        return bytes(self._buf)

    def delta_since(self) -> bytes:
        out = bytes(self._buf[self._watermark:])
        self._watermark = len(self._buf)
        return out

    def reset(self) -> None:
        self._buf.clear()
        self._watermark = 0


class _CompatRecvBuffer:
    __slots__ = ("_buf", "_consumed")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._consumed = 0

    def append(self, data: bytes) -> None:
        self._buf += data

    def size(self) -> int:
        return len(self._buf)

    def peek(self) -> bytes:
        return bytes(self._buf)

    def _maybe_compact(self) -> None:
        if self._consumed > 0 and self._consumed > len(self._buf) // 2:
            self._buf = self._buf[self._consumed:]
            self._consumed = 0

    def take_frame_length_prefixed(self, header_size: int = 4,
                                   max_frame: int = 0):
        if max_frame == 0:
            max_frame = 10 * 1024 * 1024
        avail = len(self._buf) - self._consumed
        if avail < header_size:
            return None
        length = int.from_bytes(bytes(self._buf[self._consumed:self._consumed + header_size]), "big")
        if length == 0 or length > max_frame:
            return None
        start = self._consumed + header_size
        if len(self._buf) < start + length:
            return None
        out = bytes(self._buf[start:start + length])
        self._consumed = start + length
        self._maybe_compact()
        return out

    def take_frame_delimiter(self, delim: bytes, max_frame: int = 0):
        if max_frame == 0:
            max_frame = 10 * 1024 * 1024
        idx = self._buf.find(delim, self._consumed)
        if idx < 0:
            return None
        frame = bytes(self._buf[self._consumed:idx])
        if len(frame) > max_frame:
            return None
        self._consumed = idx + len(delim)
        self._maybe_compact()
        return frame

    def compact(self) -> None:
        self._buf = self._buf[self._consumed:]
        self._consumed = 0

    def clear(self) -> None:
        self._buf.clear()
        self._consumed = 0


def _compat_build_sse_frame(event_name: str, data_json: bytes, id: int) -> bytes:
    out = bytearray()
    if event_name:
        out += b"event: " + _enc(event_name) + b"\n"
    if id:
        out += b"id: " + str(int(id)).encode("ascii") + b"\n"
    for line in bytes(data_json).split(b"\n"):
        out += b"data: " + line + b"\n"
    out += b"\n"
    return bytes(out)


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------

_USE = use_native("CODEC") and _native is not None


def serialize_envelope(type_name: str, payload: bytes) -> bytes:
    """Serialize {type, payload} into an envelope frame (ONE JSON pass)."""
    if _USE:
        return bytes(_native.codec.serialize_envelope(type_name, bytes(payload)))
    return _compat_serialize_envelope(type_name, payload)


def deserialize_envelope(frame: bytes):
    """Parse an envelope frame -> (type_name, payload_bytes), or None."""
    if _USE:
        return _native.codec.deserialize_envelope(bytes(frame))
    return _compat_deserialize_envelope(frame)


def canonicalize_payload(data: bytes):
    """Recursive object-key sort + compact re-encode; None on invalid JSON."""
    if _USE:
        return _native.codec.canonicalize_payload(bytes(data))
    return _compat_canonicalize_payload(data)


class WireMergeBuffer:
    """Incremental wire merge buffer (same-kind text/args parts merge)."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.codec.WireMergeBuffer()
        else:
            self._compat = _CompatWireMergeBuffer()

    def append(self, kind: str, delta: bytes) -> bool:
        if self._native is not None:
            return bool(self._native.append(kind, bytes(delta)))
        return self._compat.append(kind, delta)

    def snapshot(self) -> bytes:
        if self._native is not None:
            return bytes(self._native.snapshot())
        return self._compat.snapshot()

    def reset(self) -> None:
        if self._native is not None:
            self._native.reset()
        else:
            self._compat.reset()

    def empty(self) -> bool:
        if self._native is not None:
            return bool(self._native.empty())
        return self._compat.empty()


class ArgsBuffer:
    """Incremental args buffer: delta_since() returns only new bytes."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.codec.ArgsBuffer()
        else:
            self._compat = _CompatArgsBuffer()

    def append(self, delta: bytes) -> None:
        if self._native is not None:
            self._native.append(bytes(delta))
        else:
            self._compat.append(delta)

    def snapshot(self) -> bytes:
        if self._native is not None:
            return bytes(self._native.snapshot())
        return self._compat.snapshot()

    def delta_since(self) -> bytes:
        if self._native is not None:
            return bytes(self._native.delta_since())
        return self._compat.delta_since()

    def reset(self) -> None:
        if self._native is not None:
            self._native.reset()
        else:
            self._compat.reset()


class JsonRpcFrameWriter:
    """Newline-delimited JSON-RPC framing: write(payload) = payload + b'\\n'."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.codec.JsonRpcFrameWriter()

    def write(self, payload: bytes) -> bytes:
        if self._native is not None:
            return bytes(self._native.write(bytes(payload)))
        return bytes(payload) + b"\n"


class JsonlRecorder:
    """wire.jsonl line recorder: record(frame) = frame + b'\\n'."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.codec.JsonlRecorder()

    def record(self, frame: bytes) -> bytes:
        if self._native is not None:
            return bytes(self._native.record(bytes(frame)))
        return bytes(frame) + b"\n"


class RecvBuffer:
    """Growable TCP receive buffer (big-endian length-prefixed framing)."""

    def __init__(self) -> None:
        self._native = None
        if _USE:
            self._native = _native.codec.RecvBuffer()
        else:
            self._compat = _CompatRecvBuffer()

    def append(self, data: bytes) -> None:
        if self._native is not None:
            self._native.append(bytes(data))
        else:
            self._compat.append(data)

    def size(self) -> int:
        if self._native is not None:
            return int(self._native.size())
        return self._compat.size()

    def peek(self) -> bytes:
        if self._native is not None:
            return bytes(self._native.peek())
        return self._compat.peek()

    def take_frame_length_prefixed(self, header_size: int = 4,
                                   max_frame: int = 0):
        if self._native is not None:
            out = self._native.take_frame_length_prefixed(int(header_size),
                                                          int(max_frame))
            return None if out is None else bytes(out)
        return self._compat.take_frame_length_prefixed(header_size, max_frame)

    def take_frame_delimiter(self, delim: bytes, max_frame: int = 0):
        if self._native is not None:
            out = self._native.take_frame_delimiter(bytes(delim), int(max_frame))
            return None if out is None else bytes(out)
        return self._compat.take_frame_delimiter(delim, max_frame)

    def compact(self) -> None:
        if self._native is not None:
            self._native.compact()
        else:
            self._compat.compact()

    def clear(self) -> None:
        if self._native is not None:
            self._native.clear()
        else:
            self._compat.clear()


def build_sse_frame(event_name: str, data_json: bytes, id: int = 0) -> bytes:
    """One SSE frame: event:/id:/data: lines, ends with b'\\n\\n'."""
    if _USE:
        return bytes(_native.codec.build_sse_frame(event_name, bytes(data_json), int(id)))
    return _compat_build_sse_frame(event_name, data_json, id)
