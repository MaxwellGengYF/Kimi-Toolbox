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
"""

import os

USE_NATIVE = os.environ.get("KIMIX_NATIVE", "auto")
_native = None
if USE_NATIVE != "0":
    try:
        import runtime_py as _native
    except ImportError:
        if USE_NATIVE == "1":
            raise
        _native = None


def use_native(kernel: str) -> bool:
    """Per-kernel toggle: module flag, env var, then fallback."""
    if _native is None:
        return False
    flag = os.environ.get(f"KIMIX_NATIVE_{kernel.upper()}", str(USE_NATIVE).lower() != "0")
    return str(flag).lower() not in ("0", "false", "no", "")


def _fallback_version() -> str:
    """Return the fallback version marker, read from the single config file
    ``version.txt`` (in the repository root) when the native module is
    unavailable. The version literal lives only in ``version.txt``; this
    module never hard-codes it."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        version_file = os.path.join(
            os.path.dirname(os.path.dirname(here)), "version.txt"
        )
        with open(version_file, "r", encoding="utf-8") as fh:
            version = fh.read().strip()
        if version:
            return f"kimix-native {version} (python fallback)"
    except Exception:
        pass
    return "kimix-native unknown (python fallback)"


_FALLBACK_VERSION = _fallback_version()


def version() -> str:
    return _native.version() if _native else _FALLBACK_VERSION
