"""Shared low-level helpers for the kimix_native kernel shim modules.

These tiny byte/JSON helpers were historically copy-pasted into several kernel
shim modules (codec, index, search, tools).  They are defined once here so the
shim keeps exactly one copy of each.

All helpers are pure-Python and dependency-light (stdlib + orjson), so this
module imports quickly and is safe to import from every kernel module.
"""

from __future__ import annotations

import json

import orjson


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
