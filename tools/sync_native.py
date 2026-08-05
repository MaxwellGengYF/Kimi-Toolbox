"""sync_native — copy the kimix-base native runtime into this project's bin/.

Stages the compiled extension (``runtime_py.pyd``), its companion
``runtime.dll`` (plus any runtime DLL dependencies in the same directory) and
the pure-Python ``kimix_native`` shim package into ``<work-dir>\\bin`` so the
native acceleration path is importable from the running project without any
absolute cross-repo path baked in.

Usage::

    python tools\\sync_native.py [--mode release|debug|auto] [--dest <dir>]

* ``--mode release`` (default): copy from ``C:\\dev\\kimix-base\\bin\\release``,
  falling back to ``debug`` then ``releasedbg`` when the preferred dir has no
  valid build.
* ``--mode debug`` / ``--mode releasedbg``: force that build directory.
* ``--mode auto``: pick the newest valid build directory.
* ``--dest <dir>``: destination (default ``<repo root>\\bin``).

Idempotent: safe to run before every test/benchmark run. Exits non-zero when
no valid source build exists.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# The kimix-base repo (build output owner). Relative to this script:
#   tools/sync_native.py -> repo root -> kimix-base sibling.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_KIMIX_BASE = os.path.join(os.path.dirname(_REPO_ROOT), "kimix-base")
_KIMIX_BASE_BIN = os.path.join(_KIMIX_BASE, "bin")
_KIMIX_BASE_SHIM = os.path.join(_KIMIX_BASE, "python", "kimix_native")

_NATIVE_FILES = ("runtime_py.pyd", "runtime.dll")
_MODES = ("release", "debug", "releasedbg")
_SKIP_DIRS = {"__pycache__", ".pytest_cache"}
_SKIP_EXTS = {".pyc", ".pyo"}


def _valid_build(bin_dir: str) -> bool:
    """A build dir is valid when both native artifacts exist."""
    return all(os.path.isfile(os.path.join(bin_dir, f)) for f in _NATIVE_FILES)


def _source_dirs(mode: str) -> list[str]:
    """Ordered source candidates honoring *mode*."""
    if not os.path.isdir(_KIMIX_BASE_BIN):
        return []
    if mode == "auto":
        cands = [
            os.path.join(_KIMIX_BASE_BIN, m)
            for m in _MODES
            if _valid_build(os.path.join(_KIMIX_BASE_BIN, m))
        ]
        if not cands:
            return []
        cands.sort(key=lambda d: os.path.getmtime(os.path.join(d, "runtime_py.pyd")))
        return [cands[-1]]
    if mode == "release":
        # release first, then debug, then releasedbg (fallback order).
        order = ("release", "debug", "releasedbg")
        return [
            os.path.join(_KIMIX_BASE_BIN, m)
            for m in order
            if _valid_build(os.path.join(_KIMIX_BASE_BIN, m))
        ]
    cand = os.path.join(_KIMIX_BASE_BIN, mode)
    return [cand] if _valid_build(cand) else []


def _copytree_shim(src: str, dst: str) -> int:
    """Copy the kimix_native shim package, skipping caches; returns bytes."""
    total = 0
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            if name in _SKIP_DIRS:
                continue
            total += _copytree_shim(s, d)
        else:
            if name.endswith(tuple(_SKIP_EXTS)) or name in ("nul",):
                continue
            shutil.copy2(s, d)
            total += os.path.getsize(s)
    return total


def sync(mode: str = "release", dest: str | None = None) -> int:
    """Stage native artifacts into *dest*; returns total copied bytes.

    Raises FileNotFoundError when no valid source build exists.
    """
    if dest is None:
        dest = os.path.join(_REPO_ROOT, "bin")
    sources = _source_dirs(mode)
    if not sources:
        raise FileNotFoundError(
            f"no valid native build found under {_KIMIX_BASE_BIN!r} "
            f"(need runtime_py.pyd + runtime.dll; mode={mode!r})"
        )
    src = sources[0]
    os.makedirs(dest, exist_ok=True)

    total = 0
    copied = []
    # Native artifacts + any runtime DLL deps in the same dir.
    for name in sorted(os.listdir(src)):
        if name in _NATIVE_FILES:
            shutil.copy2(os.path.join(src, name), os.path.join(dest, name))
            copied.append(name)
            total += os.path.getsize(os.path.join(src, name))
        elif name.lower().endswith(".dll"):
            # Extra DLL deps shipped beside the runtime (e.g. ASAN debug
            # runtimes). Copy them so the extension loads from a bare staging.
            shutil.copy2(os.path.join(src, name), os.path.join(dest, name))
            copied.append(name)
            total += os.path.getsize(os.path.join(src, name))
    # The shim package (keeps the staged dir self-contained).
    shim_dest = os.path.join(dest, "kimix_native")
    if os.path.isdir(_KIMIX_BASE_SHIM):
        if os.path.isdir(shim_dest):
            shutil.rmtree(shim_dest)
        total += _copytree_shim(_KIMIX_BASE_SHIM, shim_dest)
        copied.append("kimix_native/ (shim)")
    print(f"source : {src}")
    print(f"dest   : {dest}")
    for name in copied:
        print(f"  -> {name}")
    print(f"copied : {total:,} bytes")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("release", "debug", "releasedbg", "auto"),
        default="release",
        help="build mode to copy from (default: release, fallback debug)",
    )
    parser.add_argument("--dest", default=None, help="destination dir (default: <repo>\\bin)")
    args = parser.parse_args(argv)
    try:
        sync(mode=args.mode, dest=args.dest)
    except FileNotFoundError as exc:
        print(f"sync_native: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
