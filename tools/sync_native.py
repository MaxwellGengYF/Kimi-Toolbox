"""sync_native — copy the kimix-base native runtime into this project's bin/.

Stages the compiled extension (``runtime_py.pyd``) and its companion
``runtime.dll`` (plus any runtime DLL dependencies in the same directory) into
``<work-dir>\\bin`` so the native acceleration path is importable from the
running project without any absolute cross-repo path baked in. The
pure-Python ``kimix_native`` shim package lives in ``bin\\kimix_native`` and is
tracked by git, so it is NOT copied here.

Usage::

    python tools\\sync_native.py [--mode release|debug|auto] [--dest <dir>]

* ``--mode release`` (default): copy from the sibling kimix-base repo's
  ``bin\\release``, falling back to ``debug`` then ``releasedbg`` when the
  preferred dir has no valid build.
* ``--mode debug`` / ``--mode releasedbg``: force that build directory.
* ``--mode auto``: pick the newest valid build directory.
* ``--dest <dir>``: destination (default ``<repo root>\\bin``).

Idempotent: safe to run before every test/benchmark run. Exits non-zero when
no valid source build exists. The kimix-base repo is located via the
``$KIMIX_BASE`` env var when set, otherwise as the ``kimix-base`` sibling of
this repo root — no absolute path is baked in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# The kimix-base repo (build output owner). Relative to this script:
#   tools/sync_native.py -> repo root -> kimix-base sibling (override with
#   the $KIMIX_BASE env var for other layouts/platforms).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def _kimix_base() -> str:
    """kimix-base repo root: $KIMIX_BASE env var, else the sibling of this repo."""
    env = os.environ.get("KIMIX_BASE")
    if env:
        return env
    return os.path.join(os.path.dirname(_REPO_ROOT), "kimix-base")


def _kimix_base_bin() -> str:
    """kimix-base bin dir (parent of the per-mode build dirs)."""
    return os.path.join(_kimix_base(), "bin")

_NATIVE_FILES = ("runtime_py.pyd", "runtime.dll")
_MODES = ("release", "debug", "releasedbg")


def _valid_build(bin_dir: str) -> bool:
    """A build dir is valid when both native artifacts exist."""
    return all(os.path.isfile(os.path.join(bin_dir, f)) for f in _NATIVE_FILES)


def _source_dirs(mode: str) -> list[str]:
    """Ordered source candidates honoring *mode*."""
    base_bin = _kimix_base_bin()
    if not os.path.isdir(base_bin):
        return []
    if mode == "auto":
        cands = [
            os.path.join(base_bin, m)
            for m in _MODES
            if _valid_build(os.path.join(base_bin, m))
        ]
        if not cands:
            return []
        cands.sort(key=lambda d: os.path.getmtime(os.path.join(d, "runtime_py.pyd")))
        return [cands[-1]]
    if mode == "release":
        # release first, then debug, then releasedbg (fallback order).
        order = ("release", "debug", "releasedbg")
        return [
            os.path.join(base_bin, m)
            for m in order
            if _valid_build(os.path.join(base_bin, m))
        ]
    cand = os.path.join(base_bin, mode)
    return [cand] if _valid_build(cand) else []


def sync(mode: str = "release", dest: str | None = None) -> int:
    """Stage native artifacts into *dest*; returns total copied bytes.

    Raises FileNotFoundError when no valid source build exists.
    """
    if dest is None:
        dest = os.path.join(_REPO_ROOT, "bin")
    sources = _source_dirs(mode)
    if not sources:
        raise FileNotFoundError(
            f"no valid native build found under {_kimix_base_bin()!r} "
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
