"""kimix_native — Python shim for the compiled ``runtime_py`` extension module.

The compiled extension (``runtime_py.pyd``) is lazily imported as ``_native``.
The module-level environment toggle ``KIMIX_NATIVE`` (default ``auto``)
controls whether the native extension is used at all:

* ``KIMIX_NATIVE=0`` — never import the compiled module (pure-Python
  fallback; the framework's ``_compat`` implementations are used).
* ``KIMIX_NATIVE=1`` — require the compiled module (raise ImportError if the
  .pyd is unavailable).
* ``KIMIX_NATIVE=auto`` (default) — use the compiled module when it is
  importable, fall back to pure Python otherwise.

Per-kernel overrides: ``KIMIX_NATIVE_<KERNEL>`` (e.g. ``KIMIX_NATIVE_TEXT=0``)
disables the native implementation for one kernel while keeping the rest
native. This implements the report's ``--native`` / ``--python`` conformance
strategy: every kernel must have a bit-identical Python fallback (mirrored in
``_compat`` modules) so ``use_native(kernel) is False`` yields identical
behavior.

Version gate: the compiled module is only used when the version it reports
(``runtime_py.version()``, e.g. ``kimix-runtime 1.0.0``) matches the repo-root
``KIMIX_NATIVE_VERSION`` marker (falls back to kimix-base's ``version.txt``
for source checkouts). On mismatch — or when the marker cannot be read — the
native extension is disabled (``_native = None``, pure-Python fallback) and
the reason is recorded in ``DISABLE_REASON``. In ``KIMIX_NATIVE=1`` mode a
mismatch raises ImportError just like a missing binary.
"""

import os

USE_NATIVE = os.environ.get("KIMIX_NATIVE", "auto")
DISABLE_REASON: str | None = None


def _repo_root() -> str:
    """Repo root holding ``bin/kimix_native`` — the parent of ``bin``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _marker_version() -> str | None:
    """Read the version marker: repo-root ``KIMIX_NATIVE_VERSION`` first, then
    kimix-base's ``version.txt`` (source-checkout layout). Returns the stripped
    non-empty version string, or None when neither marker is readable."""
    root = _repo_root()
    for name in ("KIMIX_NATIVE_VERSION", "version.txt"):
        try:
            with open(os.path.join(root, name), "r", encoding="utf-8") as fh:
                version = fh.read().strip()
            if version:
                return version
        except Exception:
            continue
    return None


def _reported_version(mod) -> str | None:
    """Extract the version token ``runtime_py.version()`` reports
    (e.g. ``kimix-runtime 1.0.0`` -> ``1.0.0``); None when unreadable."""
    try:
        raw = str(mod.version()).strip()
    except Exception:
        return None
    if not raw:
        return None
    return raw.rsplit(" ", 1)[-1] if " " in raw else raw


_native = None
if USE_NATIVE != "0":
    try:
        import runtime_py as _native
    except ImportError:
        if USE_NATIVE == "1":
            raise
        _native = None
    if _native is not None:
        marker = _marker_version()
        reported = _reported_version(_native)
        if marker and reported and marker == reported:
            DISABLE_REASON = None
        else:
            if marker is None:
                why = "cannot read the KIMIX_NATIVE_VERSION marker"
            elif reported is None:
                why = f"cannot read runtime_py.version() (KIMIX_NATIVE_VERSION is {marker!r})"
            else:
                why = f"runtime_py reports {reported!r}, KIMIX_NATIVE_VERSION is {marker!r}"
            DISABLE_REASON = f"native runtime version mismatch: {why}"
            _native = None
            if USE_NATIVE == "1":
                raise ImportError(DISABLE_REASON)


def use_native(kernel: str) -> bool:
    """Per-kernel toggle: module flag, env var, then fallback."""
    if _native is None:
        return False
    flag = os.environ.get(f"KIMIX_NATIVE_{kernel.upper()}", str(USE_NATIVE).lower() != "0")
    return str(flag).lower() not in ("0", "false", "no", "")


def _fallback_version() -> str:
    """Return the fallback version marker, read from the repo-root
    ``KIMIX_NATIVE_VERSION`` config (falls back to kimix-base's
    ``version.txt``) when the native module is unavailable. The version
    literal lives only in those config files; this module never hard-codes
    it."""
    version = _marker_version()
    if version:
        return f"kimix-native {version} (python fallback)"
    return "kimix-native unknown (python fallback)"


_FALLBACK_VERSION = _fallback_version()


def version() -> str:
    return _native.version() if _native else _FALLBACK_VERSION
