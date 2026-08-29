"""Compat test: src/kimix consumers work (pure Python) without the native
runtime — simulated by forcing ``KIMIX_NATIVE=0`` in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_with_native_disabled(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter with the native runtime disabled."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = "0"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )


def test_consumers_work_without_native():
    """filter_output/_dedup_output/parsers still produce correct Python results."""
    code = (
        "from kimix.tools.common import _dedup_output, filter_output; "
        "assert filter_output('\\x1b[31mred\\x1b[0m\\r\\nline') == 'red\\nline'; "
        "assert _dedup_output('x\\nx\\nx\\nx\\nx', 3) == 'x  (5 repeats)'; "
        "from kimix.parser.c_parser import CParser; "
        "result = CParser().parse('// hi\\nint x;\\n'); "
        "assert result.total_comments == 1; "
        "assert result.comments[0].content == ' hi'"
    )
    proc = _run_with_native_disabled(code)
    assert proc.returncode == 0, proc.stderr


def test_loader_fallback_subprocess():
    """A fresh interpreter with the native runtime disabled degrades gracefully."""
    code = (
        "import kimi_cli.native_loader as n; "
        "print('AVAILABLE=%s' % n.NATIVE_AVAILABLE); "
        "print('VERSION=%s' % n.version())"
    )
    proc = _run_with_native_disabled(code)
    assert proc.returncode == 0, proc.stderr
    assert "AVAILABLE=False" in proc.stdout
    assert "fallback" in proc.stdout
