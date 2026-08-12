"""kimix.native_loader — resolve and load the ``kimix_native`` shim.

The native acceleration path (``runtime_py.pyd`` on Windows / ``runtime_py.so``
on Linux & macOS, compiled in the kimix-base repo, plus the pure-Python
``kimix_native`` shim that wraps it) is OPTIONAL. This module locates the shim
using an ordered list of
search paths, inserts the first usable directory on ``sys.path`` and imports
it once (cached). When the binaries are missing the loader degrades to the
pure-Python fallback — no import-time side effects, never raises (except for
the documented ``KIMIX_NATIVE=1`` contract).

Search order (first usable directory wins):

1. ``KIMIX_NATIVE_PATH`` env var — explicit override; when set, only that
   directory is tried.
2. **default**: ``<repo root>\\bin`` — the current project work-dir where
   ``tools\\sync_native.py`` stages the compiled extension (``runtime_py.pyd``
   on Windows / ``runtime_py.so`` on Linux & macOS; the ``kimix_native``
   shim package is tracked by git and always present there).
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
  TOOLS|STREAM|CODEC|DIFF|GLOB) while the rest stay native.
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
    "get_compat",
]

_KERNELS = (
    "TEXT",
    "INDEX",
    "SEARCH",
    "PARSE",
    "TOOLS",
    "STREAM",
    "CODEC",
    "DIFF",
    "GLOB",
)

# The _KERNELS tuple above is informational; the actual per-kernel gate is
# delegated to the shim (kimix_native.use_native), so any kernel name the shim
# recognizes works even if it is not listed here.

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
    # The shim may have imported fine while the compiled extension failed
    # (auto fallback): native is usable only when the shim's own import
    # succeeded.
    return getattr(_shim, "_native", None) is not None


def _setup() -> tuple[bool, str | None]:
    """Resolve paths, insert on sys.path, import; returns (available, dir)."""
    global NATIVE_AVAILABLE, NATIVE_PATH
    mode = os.environ.get("KIMIX_NATIVE", "auto")
    if mode == "0":
        # Pure-Python mode: never import the compiled extension, but the shim
        # package (whose ``_compat`` modules are the canonical pure-Python
        # reference implementations that src/kimix re-exports) must remain
        # importable, so the candidate dirs are inserted exactly like the
        # auto path does below.  Nothing is imported and no native module is
        # used; the shim simply falls back to its pure-Python ``_compat``.
        for cand in _candidate_dirs():
            if not cand or not os.path.isdir(cand):
                continue
            for d in _shim_dirs_for(cand):
                if d not in sys.path:
                    sys.path.insert(0, d)
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

# Fast-path tables: the runtime environment is stable (env toggles are fixed
# at process start and the shim import is cached), so every per-kernel gate
# decision and per-module resolution is computed ONCE at import time and
# stored in precomputed dicts. The hot path is then a single dict lookup —
# no per-call ``.upper()`` allocation, no repeated NATIVE_AVAILABLE/_shim
# checks, no sentinel branch. Names outside the tables (e.g. kernels the
# shim knows but this loader does not) fall back to the dynamic path below
# and are memoized the same way. Runtime toggling is done via fresh
# subprocesses or by monkeypatching these functions in tests (the loader is
# reloaded there when needed).
_MISSING = object()
_kernel_cache: dict[str, bool] = {}  # dynamic fallback (unknown kernels)
_module_cache: dict[str, object] = {}  # dynamic fallback (unknown modules)
_kernel_module_cache: dict[str, object] = {}


def _build_kernel_table() -> dict[str, bool]:
    """Precompute per-kernel gate results for every known kernel name.

    Each known kernel is stored under its upper/lower/title spellings so
    callers never pay for a per-call ``.upper()`` on the hot path. Results
    come straight from the shim's ``use_native`` (env toggles included) and
    never change while the process lives. When native is unavailable
    (``KIMIX_NATIVE=0`` or missing extension) every known kernel is stored
    as ``False`` so the tables stay complete in every mode.
    """
    table: dict[str, bool] = {}
    for kernel in _KERNELS:
        result = (
            bool(_shim.use_native(kernel)) if NATIVE_AVAILABLE and _shim is not None else False
        )
        table[kernel] = result
        table[kernel.lower()] = result
        table[kernel.title()] = result
    return table


def _build_module_table() -> dict[str, object]:
    """Precompute the resolved submodule for every known kernel module name.

    All known submodules (``kimix_native.<kernel.lower()>``) are imported
    once here; each is a small shim module over the already-loaded
    ``runtime_py`` extension, so the one-time cost is negligible and the hot
    path becomes a single dict lookup (module object or None). When native
    is unavailable every known module is stored as ``None`` so the tables
    stay complete in every mode.
    """
    table: dict[str, object] = {}
    for kernel in _KERNELS:
        name = kernel.lower()
        mod: object = None
        if NATIVE_AVAILABLE and _shim is not None:
            try:
                mod = importlib.import_module(f"kimix_native.{name}")
            except ImportError:
                mod = None
        table[name] = mod
    return table


_KERNEL_TABLE = _build_kernel_table()
_MODULE_TABLE = _build_module_table()


def use_native(kernel: str) -> bool:
    """Per-kernel gate: True when native is active for *kernel*.

    Hot path: a single lookup into the precomputed kernel table (the runtime
    environment is stable, so per-kernel decisions never change). Kernels
    the shim recognizes but that are not in the table are resolved once via
    the shim and memoized.
    """
    cached = _KERNEL_TABLE.get(kernel, _MISSING)
    if cached is not _MISSING:
        return cached
    if not NATIVE_AVAILABLE or _shim is None:
        _KERNEL_TABLE[kernel] = False
        return False
    key = kernel.upper()
    cached = _kernel_cache.get(key, _MISSING)
    if cached is not _MISSING:
        return cached
    result = bool(_shim.use_native(key))
    _kernel_cache[key] = result
    _KERNEL_TABLE[kernel] = result
    return result


def _fallback_version() -> str:
    """Return the fallback version marker, synced from ``KIMIX_NATIVE_VERSION``."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        version_file = os.path.join(
            os.path.dirname(os.path.dirname(here)), "KIMIX_NATIVE_VERSION"
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
    if _shim is not None:
        try:
            return str(_shim.version())
        except Exception:
            pass
    return _FALLBACK_VERSION


def get_module(name: str):
    """Return the ``kimix_native.<name>`` submodule, or None when unavailable.

    Hot path: a single lookup into the precomputed module table (all known
    submodules resolved at import time). Unknown names are imported on
    demand and memoized.
    """
    cached = _MODULE_TABLE.get(name, _MISSING)
    if cached is not _MISSING:
        return cached
    if not NATIVE_AVAILABLE or _shim is None:
        return None
    if name in ("_native",) or name.startswith("_"):
        return None
    try:
        mod = importlib.import_module(f"kimix_native.{name}")
    except ImportError:
        mod = None
    _MODULE_TABLE[name] = mod
    return mod


def get_compat(name: str):
    """Return the ``kimix_native.<name>`` pure-Python compat submodule, or None.

    Unlike :func:`get_module` this works even when the compiled extension is
    unavailable (``KIMIX_NATIVE=0`` / missing binaries): the shim's ``_compat``
    modules are the canonical pure-Python reference implementations that
    ``src/kimix`` re-exports, so they are imported whenever the shim package
    itself is importable.  Returns None only when the shim cannot be imported
    at all (e.g. an installed wheel without the bundled shim).
    """
    try:
        return importlib.import_module(f"kimix_native.{name}")
    except ImportError:
        return None


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
    if use_native(key):
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
