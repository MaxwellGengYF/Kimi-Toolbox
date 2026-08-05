"""kimix_native.concurrency -- concurrency kernels: bounded MPSC event bus
and atomic ID generation (plan 008).

Native implementations live in ``runtime_py.concurrency`` (compiled
kernels, GIL released). The pure-Python ``_compat`` mirrors implement the
same contract:

- MpscEventBus: bounded ring with DROP_OLDEST policy; emit() is O(1)
  regardless of subscriber count; subscribers hold monotonic offsets and
  skip events dropped while they were slow; ids are never reused.
- IdGenerator: monotonic counter (no lock, no hash); reserve(n) returns n
  consecutive ids.
"""

from __future__ import annotations

import threading

from . import _native, use_native

_USE = use_native("CONCURRENCY") and _native is not None


# ---------------------------------------------------------------------------
# _compat -- pure-Python mirrors (same semantics as the kernels)
# ---------------------------------------------------------------------------


class _CompatMpscEventBus:
    __slots__ = ("_capacity", "_ring", "_head", "_count", "_seq",
                 "_offsets", "_next_sub_id", "_lock")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            capacity = 1
        self._capacity = int(capacity)
        self._ring = [b""] * self._capacity
        self._head = 0
        self._count = 0
        self._seq = 0
        self._offsets: dict[int, int] = {}
        self._next_sub_id = 1
        self._lock = threading.Lock()

    def emit(self, event_bytes: bytes) -> None:
        with self._lock:
            slot = self._seq % self._capacity
            self._ring[slot] = bytes(event_bytes)
            if self._count < self._capacity:
                self._count += 1
            self._seq += 1

    def subscribe(self) -> int:
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._offsets[sub_id] = self._seq
            return sub_id

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._offsets.pop(sub_id, None)

    def poll(self, sub_id: int):
        with self._lock:
            offset = self._offsets.get(sub_id)
            if offset is None or offset >= self._seq:
                return None
            oldest = self._seq - self._count
            if offset < oldest:
                offset = oldest
                self._offsets[sub_id] = offset
            if offset >= self._seq:
                return None
            slot = offset % self._capacity
            self._offsets[sub_id] = offset + 1
            return self._ring[slot]

    def seq(self) -> int:
        with self._lock:
            return self._seq

    def capacity(self) -> int:
        return self._capacity


class _CompatIdGenerator:
    __slots__ = ("_counter", "_lock")

    def __init__(self, seed: int = 0) -> None:
        self._counter = int(seed)
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            v = self._counter
            self._counter += 1
            return v

    def reserve(self, n: int) -> list[int]:
        with self._lock:
            start = self._counter
            self._counter += int(n)
            return list(range(start, start + int(n)))


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


class MpscEventBus:
    """Bounded single-producer/multi-consumer event bus (DROP_OLDEST)."""

    def __init__(self, capacity: int) -> None:
        self._native = None
        if _USE:
            self._native = _native.concurrency.MpscEventBus(int(capacity))
        else:
            self._compat = _CompatMpscEventBus(capacity)

    def emit(self, event_bytes: bytes) -> None:
        if self._native is not None:
            self._native.emit(bytes(event_bytes))
        else:
            self._compat.emit(event_bytes)

    def subscribe(self) -> int:
        if self._native is not None:
            return int(self._native.subscribe())
        return self._compat.subscribe()

    def unsubscribe(self, sub_id: int) -> None:
        if self._native is not None:
            self._native.unsubscribe(int(sub_id))
        else:
            self._compat.unsubscribe(sub_id)

    def poll(self, sub_id: int):
        if self._native is not None:
            out = self._native.poll(int(sub_id))
            return None if out is None else bytes(out)
        return self._compat.poll(sub_id)

    def seq(self) -> int:
        if self._native is not None:
            return int(self._native.seq())
        return self._compat.seq()

    def capacity(self) -> int:
        if self._native is not None:
            return int(self._native.capacity())
        return self._compat.capacity()


class IdGenerator:
    """Thread-safe monotonic ID generator (atomic fetch_add)."""

    def __init__(self, seed: int = 0) -> None:
        self._native = None
        if _USE:
            self._native = _native.concurrency.IdGenerator(int(seed))
        else:
            self._compat = _CompatIdGenerator(seed)

    def next(self) -> int:
        if self._native is not None:
            return int(self._native.next())
        return self._compat.next()

    def reserve(self, n: int) -> list[int]:
        if self._native is not None:
            return [int(v) for v in self._native.reserve(int(n))]
        return self._compat.reserve(n)
