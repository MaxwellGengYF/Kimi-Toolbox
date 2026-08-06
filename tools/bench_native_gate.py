"""Micro-benchmark: hot-path cost of the native gate + module accessors.

Compares three hot-path shapes on the SAME kernel ("SEARCH"):

* ``old``      — the pre-optimization implementation: per-call ``.upper()``
                 allocation + sentinel lookup + NATIVE_AVAILABLE/_shim checks
                 (reconstructed inline for comparison).
* ``new``      — the optimized ``kimix.native_loader``: precomputed import-time
                 tables, single dict hit per call.
* ``hoisted``  — what the hot call sites now do: the module is resolved once at
                 import time (``_NATIVE_SEARCH = get_module("search")``), so the
                 per-item cost is one gate lookup + one attribute check.

Run:  uv run python tools/bench_native_gate.py
"""

from __future__ import annotations

import importlib
import timeit

import kimix.native_loader as knl

_N = 300_000


# ---------------------------------------------------------------------------
# new / hoisted paths (real, current implementation)
# ---------------------------------------------------------------------------

_NATIVE_SEARCH = knl.get_module("search")


def new_two_calls() -> None:
    if knl.use_native("SEARCH") and knl.get_module("search") is not None:
        pass


def new_hoisted() -> None:
    if knl.use_native("SEARCH") and _NATIVE_SEARCH is not None:
        pass


def new_gate_only() -> None:
    knl.use_native("SEARCH")


def new_module_only() -> None:
    knl.get_module("search")


# ---------------------------------------------------------------------------
# old path (reconstructed from the pre-optimization implementation)
# ---------------------------------------------------------------------------

_MISSING = object()
_old_kernel_cache: dict[str, bool] = {}
_old_module_cache: dict[str, object] = {}
_NATIVE_AVAILABLE = knl.NATIVE_AVAILABLE
_shim = getattr(knl, "_shim", None)


def old_use_native(kernel: str) -> bool:
    if not _NATIVE_AVAILABLE or _shim is None:
        return False
    key = kernel.upper()
    cached = _old_kernel_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    result = bool(_shim.use_native(key))
    _old_kernel_cache[key] = result
    return result


def old_get_module(name: str):
    if not _NATIVE_AVAILABLE or _shim is None:
        return None
    if name in ("_native",) or name.startswith("_"):
        return None
    cached = _old_module_cache.get(name, _MISSING)
    if cached is not _MISSING:
        return cached
    try:
        mod = importlib.import_module(f"kimix_native.{name}")
    except ImportError:
        mod = None
    _old_module_cache[name] = mod
    return mod


def old_two_calls() -> None:
    if old_use_native("SEARCH"):
        if old_get_module("search") is not None:
            pass


# Warm every cache so we measure the steady-state hot path only.
knl.use_native("SEARCH")
knl.get_module("search")
old_use_native("SEARCH")
old_get_module("search")

CASES = [
    ("old  use_native+get_module (two dict hits)", old_two_calls),
    ("new  use_native+get_module (two dict hits)", new_two_calls),
    ("new  hoisted module (gate + attr check)     ", new_hoisted),
    ("new  use_native only                       ", new_gate_only),
    ("new  get_module only                       ", new_module_only),
]


def main() -> None:
    print(f"native available: {knl.NATIVE_AVAILABLE}; {_N} iterations each")
    print("-" * 64)
    results = []
    for label, fn in CASES:
        total = timeit.timeit(fn, number=_N)
        per = total / _N * 1e9
        results.append((label, per))
        print(f"{label}: {per:7.1f} ns/iter  ({total:.3f}s)")
    print("-" * 64)
    old = next(p for _l, p in results if _l.startswith("old"))
    new = next(p for _l, p in results if _l.startswith("new  use_native+get_module"))
    hoisted = next(p for _l, p in results if _l.startswith("new  hoisted"))
    print(f"new two-call vs old:      {old / new:.2f}x faster")
    print(f"hoisted call site vs old: {old / hoisted:.2f}x faster")


if __name__ == "__main__":
    main()
