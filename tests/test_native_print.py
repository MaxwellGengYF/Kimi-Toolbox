"""Independent verification of the native PRINT kernel in kimix.ui.printing.

This test deliberately does NOT stage or replace ``<repo>/bin/runtime_py.pyd``
— that copy may be locked by another process (loaded into a running
interpreter) or predate the ``print`` submodule. Instead it points
``KIMIX_NATIVE_PATH`` at a kimix-base build that ships
``runtime_py.print.native_print`` (e.g. ``C:\\dev\\kimix-base\\bin\\debug``)
and asserts that ``kimix.ui.printing`` routes stdout through the native async
print stream. The ``kimix_native`` shim package is reused from
``<repo>/bin/kimix_native`` via ``PYTHONPATH`` — no files are copied or
modified.

The file lives at ``tests/`` (NOT ``tests/native/``) so the session-scoped
``sync_native.py`` fixture in ``tests/native/conftest.py`` never overwrites
``bin/``. All checks run in clean subprocesses because the loader resolves the
native environment once at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The staged artifact name is platform-dependent (runtime_py.pyd on Windows /
# runtime_py.so on Linux & macOS).
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"


def _candidate_native_bins() -> list[str]:
    """kimix-base bin dirs that actually contain a runtime_py artifact."""
    base = os.environ.get("KIMIX_BASE") or os.path.join(os.path.dirname(_REPO), "kimix-base")
    return [
        os.path.join(base, "bin", mode)
        for mode in ("debug", "releasedbg", "release")
        if os.path.isfile(os.path.join(base, "bin", mode, _NATIVE_FILE))
    ]


def _has_native_print(bin_dir: str) -> bool:
    """True when the runtime in *bin_dir* exposes print.native_print."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import runtime_py;"
        "print(hasattr(getattr(runtime_py, 'print', None), 'native_print'))" % bin_dir
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "True"


def _native_bin_dir() -> str | None:
    """First kimix-base bin whose runtime has the print kernel (or None)."""
    for bin_dir in _candidate_native_bins():
        if _has_native_print(bin_dir):
            return bin_dir
    return None


def _run_printing(
    code: str,
    *,
    native_bin: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter, optionally pointed at *native_bin*."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = "auto"
    if native_bin is not None:
        env["KIMIX_NATIVE_PATH"] = native_bin
    # kimix resolves via the repo src dir; the kimix_native shim resolves via
    # the repo bin dir (KIMIX_NATIVE_PATH only provides the compiled module).
    pythonpath = os.pathsep.join(
        p
        for p in (
            os.path.join(_REPO, "bin"),
            os.path.join(_REPO, "src"),
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    env["PYTHONPATH"] = pythonpath
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )


# Native path: every p.print(...) call below goes through the async stream.
_NATIVE_CODE = r"""
import sys
import time

import kimix.native_loader as n
import kimix.ui.printing as p

assert n.NATIVE_AVAILABLE, "native runtime should be available"
assert n.NATIVE_PATH, "NATIVE_PATH should be set (KIMIX_NATIVE_PATH)"
assert p._NATIVE_PRINT is not None, "runtime_py.print.native_print not resolved"
assert p._print_func.__name__ == "_native_print_func", p._print_func

# str() coercion + sep/end defaults
p.print("hello", "world")
# custom sep/end
p.print("a", "b", sep="-", end="!\n")
# non-str values
p.print(1, [2, 3])
# no values -> just end
p.print()
# explicit sys.stdout file goes native too
p.print("explicit-stdout", file=sys.stdout, flush=True)
# trailing async message; worker drains + fflushes before exit
p.print("tail", flush=True)

# Give the async worker a moment before interpreter teardown (the destructor
# also drains, so this is belt-and-suspenders).
time.sleep(0.3)
"""


def test_native_print_writes_async_stream() -> None:
    native_bin = _native_bin_dir()
    if native_bin is None:
        pytest.skip(
            "no kimix-base runtime with print.native_print found "
            "(build kimix-base debug: xmake f -m debug && xmake b runtime_py)"
        )
    proc = _run_printing(_NATIVE_CODE, native_bin=native_bin)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "hello world\n" in out
    assert "a-b!\n" in out
    assert "1 [2, 3]\n" in out
    assert "\n" in out  # p.print() with no values
    assert "explicit-stdout\n" in out
    assert "tail\n" in out


# A non-stdout file must fall back to the builtin print (native_print can only
# write to the process stdout), so nothing leaks to the native stream.
_FILE_FALLBACK_CODE = r"""
import io

import kimix.ui.printing as p

assert p._print_func.__name__ == "_native_print_func", p._print_func
buf = io.StringIO()
p.print("to-buffer", file=buf)
assert buf.getvalue() == "to-buffer\n", buf.getvalue()
p.print("to-stdout", flush=True)
"""


def test_native_print_non_stdout_file_falls_back() -> None:
    native_bin = _native_bin_dir()
    if native_bin is None:
        pytest.skip("no kimix-base runtime with print.native_print found")
    proc = _run_printing(_FILE_FALLBACK_CODE, native_bin=native_bin)
    assert proc.returncode == 0, proc.stderr
    assert "to-stdout\n" in proc.stdout
    assert "to-buffer" not in proc.stdout


# With KIMIX_NATIVE_PRINT=0 the callable is still resolved but the per-kernel
# gate is off, so printing stays on the builtin print.
_TOGGLE_OFF_CODE = r"""
import kimix.ui.printing as p

assert p._NATIVE_PRINT is not None, "native print should still resolve"
assert p._print_func.__name__ == "print", p._print_func
p.print("fallback-ok")
"""


def test_native_print_kernel_toggle_off() -> None:
    native_bin = _native_bin_dir()
    if native_bin is None:
        pytest.skip("no kimix-base runtime with print.native_print found")
    proc = _run_printing(
        _TOGGLE_OFF_CODE,
        native_bin=native_bin,
        extra_env={"KIMIX_NATIVE_PRINT": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "fallback-ok\n" in proc.stdout


# With KIMIX_NATIVE=0 the compiled module is never imported, so the native
# print callable cannot resolve and printing stays on the builtin print.
_NATIVE_OFF_CODE = r"""
import kimix.ui.printing as p

assert p._NATIVE_PRINT is None, p._NATIVE_PRINT
assert p._print_func.__name__ == "print", p._print_func
p.print("py-only")
"""


def test_native_print_falls_back_when_native_disabled() -> None:
    proc = _run_printing(
        _NATIVE_OFF_CODE,
        extra_env={"KIMIX_NATIVE": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "py-only\n" in proc.stdout
