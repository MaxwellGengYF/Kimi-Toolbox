"""kimi_cli.native_loader — lazy re-export of the shared kimix native loader.

kimi-cli-x and kimix are members of the same uv workspace; the loader logic
lives once in ``kimix.native_loader`` and is delegated to from here so
kimi_cli modules can use it without knowing which package owns it.

IMPORTANT — this module must never ``import kimix`` at module top: importing
the ``kimix`` package executes ``kimix/__init__.py`` which (via
``kimix.utils._globals -> kimi_agent_sdk -> kimi_cli.app -> kimi_cli.soul.agent``)
re-enters ``kimi_cli`` mid-import, hitting a pre-existing circular import.
Instead the loader is loaded STANDALONE by file path (it only uses the
standard library), so no package ``__init__`` runs. All resolution is deferred
to first use via module ``__getattr__`` (``from kimi_cli.native_loader import X``
falls back to ``__getattr__`` when ``X`` is not defined eagerly).

API (identical to kimix.native_loader): ``NATIVE_AVAILABLE``,
``NATIVE_PATH``, ``use_native(kernel)``, ``version()``, ``get_module(name)``,
plus attribute-style submodule access (``native_loader.text``).
"""

from __future__ import annotations

import importlib.util
import os
import sys

__all__ = [
    "NATIVE_AVAILABLE",
    "NATIVE_PATH",
    "use_native",
    "version",
    "get_module",
]

_impl = None
_FAILED = False

# Fast-path caches: per-kernel gate decisions and per-module resolutions are
# resolved once and cached. Env toggles are fixed at process start; runtime
# toggling is done via fresh subprocesses or by monkeypatching these
# functions in tests.
_MISSING = object()
_kernel_cache: dict[str, bool] = {}
_module_cache: dict[str, object] = {}
_kernel_module_cache: dict[str, object] = {}


def _candidate_loader_files() -> list[str]:
    """Filesystem candidates for kimix/native_loader.py (no package import)."""
    candidates: list[str] = []
    _here = os.path.dirname(os.path.abspath(__file__))
    # <repo>/src/kimix/native_loader.py
    _cli_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))  # <repo>
    candidates.append(os.path.join(_cli_root, "src", "kimix", "native_loader.py"))
    # <repo>/kimix/native_loader.py (alt layout)
    candidates.append(os.path.join(_cli_root, "kimix", "native_loader.py"))
    return candidates


def _load_standalone():
    """Load kimix/native_loader.py by file path (bypasses kimix/__init__)."""
    for path in _candidate_loader_files():
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("kimix_native_loader", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["kimix_native_loader"] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop("kimix_native_loader", None)
            continue
    return None


def _get_impl():
    """Load and cache the shared loader (lazy, cycle-safe, never raises)."""
    global _impl, _FAILED
    if _impl is not None or _FAILED:
        return _impl
    module = _load_standalone()
    if module is not None:
        _impl = module
        return _impl
    # Last resort: normal import (installed kimix, no workspace layout).
    try:
        from kimix import native_loader as _impl
        return _impl
    except Exception:
        _FAILED = True
        return None


def use_native(kernel: str) -> bool:
    """Per-kernel gate: True when native is active for *kernel*.

    Fast path: per-kernel decision cached after first resolution (env
    toggles are fixed at process start; tests toggle via subprocesses).
    """
    key = kernel.upper()
    cached = _kernel_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    impl = _get_impl()
    result = bool(impl.use_native(key)) if impl is not None else False
    _kernel_cache[key] = result
    return result


def _fallback_version() -> str:
    """Return the fallback version marker, synced from ``KIMIX_NATIVE_VERSION``."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        version_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(here))),
            "KIMIX_NATIVE_VERSION",
        )
        with open(version_file, "r", encoding="utf-8") as fh:
            version = fh.read().strip()
        if version:
            return f"kimix-native {version} (python fallback)"
    except Exception:
        pass
    return "kimix-native 0.1.0 (python fallback)"


_FALLBACK_VERSION = _fallback_version()


def version() -> str:
    """Native runtime version, or the fallback marker string."""
    impl = _get_impl()
    return impl.version() if impl is not None else _FALLBACK_VERSION


def get_module(name: str):
    """Return the ``kimix_native.<name>`` submodule, or None when unavailable.

    Fast path: resolved once and cached (single dict lookup afterwards).
    """
    cached = _module_cache.get(name, _MISSING)
    if cached is not _MISSING:
        return cached
    impl = _get_impl()
    mod = impl.get_module(name) if impl is not None else None
    _module_cache[name] = mod
    return mod


def kernel_module(kernel: str):
    """Cached combined gate+module accessor for hot loops (see
    kimix.native_loader.kernel_module): the native submodule for *kernel* or
    None when the gate is off / native unavailable."""
    key = kernel.upper()
    cached = _kernel_module_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    impl = _get_impl()
    mod = impl.kernel_module(key) if impl is not None else None
    _kernel_module_cache[key] = mod
    return mod


def __getattr__(name: str):
    """Deferred attribute access: NATIVE_AVAILABLE / NATIVE_PATH / submodules."""
    if name in ("NATIVE_AVAILABLE", "NATIVE_PATH"):
        impl = _get_impl()
        if impl is not None:
            return getattr(impl, name)
        return False if name == "NATIVE_AVAILABLE" else None
    impl = _get_impl()
    if impl is None:
        raise AttributeError(name)
    return getattr(impl, name)
