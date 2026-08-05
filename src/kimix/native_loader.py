"""kimix.native_loader — resolve and load the ``kimix_native`` shim.

The native acceleration path (``runtime_py.pyd`` + ``runtime.dll`` compiled
in the kimix-base repo, plus the pure-Python ``kimix_native`` shim that wraps
them) is OPTIONAL. This module locates the shim using an ordered list of
search paths, inserts the first usable directory on ``sys.path`` and imports
it once (cached). When the binaries are missing the loader degrades to the
pure-Python fallback — no import-time side effects, never raises (except for
the documented ``KIMIX_NATIVE=1`` contract).

Search order (first usable directory wins):

1. ``KIMIX_NATIVE_PATH`` env var — explicit override; when set, only that
   directory is tried.
2. **default**: ``<repo root>\\bin`` — the current project work-dir where
   ``tools\\sync_native.py`` stages ``runtime_py.pyd`` + ``runtime.dll`` (the
   ``kimix_native`` shim package is tracked by git and always present there).
   "Repo root" is the parent of
   ``src/kimix`` (falling back to ``os.getcwd()`` when the repo layout is not
   detectable, e.g. an installed wheel).
3. Already importable on ``sys.path`` (a ``runtime_py`` import that succeeds
   elsewhere, e.g. a pip-installed ``kimix-native`` wheel).
4. Dev-only last resort: ``<kimix-base>/bin/{release,releasedbg,debug}``
   (plus the sibling ``python`` dir for the shim) — convenience for local
   development, never required. ``<kimix-base>`` is the ``KIMIX_BASE`` env
   var when set, otherwise the ``kimix-base`` sibling of this repo root.

Env toggles (same contract as the shim):

* ``KIMIX_NATIVE=0`` — never use native (pure Python everywhere).
* ``KIMIX_NATIVE=1`` — require native; raise ImportError if unavailable.
* ``KIMIX_NATIVE=auto`` (default) — native when importable, fallback otherwise.
* ``KIMIX_NATIVE_<KERNEL>=0`` — disable one kernel (TEXT|INDEX|SEARCH|PARSE|
  SOUL|TOOLS|STREAM|CODEC|JSON|CONCURRENCY) while the rest stay native.
* ``KIMIX_BASE=<dir>`` — kimix-base repo root for the dev-only fallback
  (priority 4); defaults to the ``kimix-base`` sibling of this repo root.
"""

from __future__ import annotations

import importlib
import os
import sys

__all__ = [
    "NATIVE_AVAILABLE",
    "NATIVE_PATH",
    "use_native",
    "version",
    "get_module",
]

_KERNELS = (
    "TEXT",
    "INDEX",
    "SEARCH",
    "PARSE",
    "SOUL",
    "TOOLS",
    "STREAM",
    "CODEC",
    "JSON",
    "CONCURRENCY",
)

# kimix-base bin modes tried in dev fallback order.
_DEV_MODES = ("release", "releasedbg", "debug")

_shim = None  # the imported kimix_native module (or None)
NATIVE_AVAILABLE: bool = False
NATIVE_PATH: str | None = None  # resolved directory (diagnostics)


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    """Parent of the kimix package (``<root>/src/kimix/native_loader.py``)."""
    here = os.path.dirname(os.path.abspath(__file__))  # .../src/kimix
    parent = os.path.dirname(here)  # .../src
    root = os.path.dirname(parent)  # .../
    if os.path.basename(parent) == "src" and os.path.isdir(os.path.join(root, "src")):
        return root
    return os.getcwd()  # repo layout not detectable (installed wheel etc.)


def _dev_base() -> str:
    """kimix-base repo root for the dev-only fallback (priority 4).

    ``$KIMIX_BASE`` overrides the default, which is the ``kimix-base``
    sibling of this repo root — no absolute path is baked in, so the loader
    works on any platform/layout without editing code.
    """
    env = os.environ.get("KIMIX_BASE")
    if env:
        return env
    return os.path.join(os.path.dirname(_repo_root()), "kimix-base")


def _candidate_dirs() -> list[str]:
    """Ordered candidate directories (env override short-circuits)."""
    env = os.environ.get("KIMIX_NATIVE_PATH")
    if env:
        return [env]
    dirs = [os.path.join(_repo_root(), "bin")]
    base = os.path.join(_dev_base(), "bin")
    for mode in _DEV_MODES:
        dirs.append(os.path.join(base, mode))
    return dirs


def _shim_dirs_for(bin_dir: str) -> list[str]:
    """Directories that may hold the ``kimix_native`` shim for *bin_dir*."""
    staged = os.path.join(bin_dir, "kimix_native")
    if os.path.isdir(staged):
        return [bin_dir]
    # kimix-base staging: shim lives in the sibling python/ dir.
    sibling = os.path.join(os.path.dirname(bin_dir), "python")
    if os.path.isdir(os.path.join(sibling, "kimix_native")):
        return [bin_dir, sibling]
    return [bin_dir]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _try_import(require: bool) -> bool:
    """Import the shim; returns True when native is actually usable.

    Raises ImportError when *require* is True and the shim (or the native
    extension it needs) is unavailable — the documented ``KIMIX_NATIVE=1``
    contract.
    """
    global _shim
    try:
        import kimix_native as _shim  # noqa: F811
    except ImportError:
        _shim = None
        if require:
            raise
        return False
    # The shim may have imported fine while the .pyd failed (auto fallback):
    # native is usable only when the shim's own import succeeded.
    return getattr(_shim, "_native", None) is not None


def _setup() -> tuple[bool, str | None]:
    """Resolve paths, insert on sys.path, import; returns (available, dir)."""
    global NATIVE_AVAILABLE, NATIVE_PATH
    mode = os.environ.get("KIMIX_NATIVE", "auto")
    if mode == "0":
        NATIVE_AVAILABLE = False
        NATIVE_PATH = None
        return False, None
    require = mode == "1"
    inserted: list[str] = []
    for cand in _candidate_dirs():
        if not cand or not os.path.isdir(cand):
            continue
        shim_dirs = _shim_dirs_for(cand)
        for d in shim_dirs:
            if d not in sys.path:
                sys.path.insert(0, d)
                inserted.append(d)
        if _try_import(require):
            NATIVE_AVAILABLE = True
            NATIVE_PATH = cand
            return True, cand
        # Not usable here; try the next candidate (insertions stay; harmless
        # because the first successful import wins and is cached).
    # Nothing in the candidate list worked — the package may already be
    # importable via the ambient sys.path (pip-installed wheel).
    if _try_import(require):
        NATIVE_AVAILABLE = True
        NATIVE_PATH = None
        return True, None
    NATIVE_AVAILABLE = False
    NATIVE_PATH = None
    return False, None


_setup()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

# Fast-path caches: the per-module import and the per-kernel gate decision
# are resolved once and cached. Env toggles are fixed at process start;
# runtime toggling is done via fresh subprocesses or by monkeypatching these
# functions in tests (the loader is reloaded there when needed).
_MISSING = object()
_module_cache: dict[str, object] = {}
_kernel_module_cache: dict[str, object] = {}
_kernel_cache: dict[str, bool] = {}


def use_native(kernel: str) -> bool:
    """Per-kernel gate: True when native is active for *kernel*.

    Fast path: the per-kernel decision is resolved once (delegating to the
    shim) and cached — a single dict lookup afterwards. Env toggles are
    fixed at process start; tests toggle via fresh subprocesses or by
    monkeypatching this function.
    """
    if not NATIVE_AVAILABLE or _shim is None:
        return False
    key = kernel.upper()
    cached = _kernel_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    result = bool(_shim.use_native(key))
    _kernel_cache[key] = result
    return result


def version() -> str:
    """Native runtime version, or the fallback marker string."""
    if NATIVE_AVAILABLE and _shim is not None:
        try:
            return str(_shim.version())
        except Exception:
            return "kimix-native 0.1.0 (python fallback)"
    return "kimix-native 0.1.0 (python fallback)"


def get_module(name: str):
    """Return the ``kimix_native.<name>`` submodule, or None when unavailable.

    Submodules are not auto-imported by the shim package, so this uses an
    explicit ``importlib`` import — resolved once and cached (single dict
    lookup on the hot path afterwards).
    """
    if not NATIVE_AVAILABLE or _shim is None:
        return None
    if name in ("_native",) or name.startswith("_"):
        return None
    cached = _module_cache.get(name, _MISSING)
    if cached is not _MISSING:
        return cached
    try:
        mod = importlib.import_module(f"kimix_native.{name}")
    except ImportError:
        mod = None
    _module_cache[name] = mod
    return mod


def kernel_module(kernel: str):
    """Cached combined gate+module accessor for hot loops.

    Returns the native submodule for *kernel* (the shim module name is the
    lower-cased kernel: TEXT->text, TOOLS->tools, ...), or None when the
    kernel's gate is off or native is unavailable. After the first call this
    is a single dict lookup — prefer it over ``use_native`` + ``get_module``
    in per-item hot functions.
    """
    if not NATIVE_AVAILABLE or _shim is None:
        return None
    key = kernel.upper()
    cached = _kernel_module_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    mod = None
    if _shim.use_native(key):
        mod = get_module(key.lower())
    _kernel_module_cache[key] = mod
    return mod


def __getattr__(name: str):
    """Attribute-style submodule access: ``native_loader.text`` etc."""
    if name.startswith("_"):
        raise AttributeError(name)
    module = get_module(name)
    if module is None:
        raise AttributeError(
            f"kimix.native_loader has no attribute {name!r} "
            "(native submodule unavailable)"
        )
    return module
