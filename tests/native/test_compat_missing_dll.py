"""Compat test: src/kimix consumers work (pure Python) without the native
runtime — simulated by renaming the staged binaries away in a subprocess.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BIN = os.path.join(_REPO, "bin")


@pytest.fixture
def native_missing(tmp_path):
    moved = []
    if os.path.isdir(_BIN):
        for name in ("runtime_py.pyd", "runtime.dll"):
            src = os.path.join(_BIN, name)
            if os.path.isfile(src):
                dst = os.path.join(tmp_path, name)
                shutil.move(src, dst)
                moved.append((src, dst))
    yield
    for src, dst in moved:
        shutil.move(dst, src)


def test_consumers_work_without_native(native_missing):
    """filter_output/_dedup_output/parsers still produce correct Python results."""
    from kimix.tools.common import _dedup_output, filter_output

    assert filter_output("\x1b[31mred\x1b[0m\r\nline") == "red\nline"
    assert _dedup_output("x\nx\nx\nx\nx", 3) == "x  (5 repeats)"

    from kimix.parser.c_parser import CParser

    result = CParser().parse("// hi\nint x;\n")
    assert result.total_comments == 1
    assert result.comments[0].content == " hi"


def test_loader_fallback_subprocess(native_missing):
    code = (
        "import kimix.native_loader as n;"
        "print('AVAILABLE=%s' % n.NATIVE_AVAILABLE);"
        "print('VERSION=%s' % n.version())"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "AVAILABLE=False" in proc.stdout
    assert "fallback" in proc.stdout
