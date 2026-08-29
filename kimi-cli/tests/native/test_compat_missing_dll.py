"""Compat test: consumers must work (pure Python) when the native runtime is
missing — simulated by forcing ``KIMIX_NATIVE=0`` in a subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_BIN = os.path.join(_REPO, "bin")

# The staged artifact name is platform-dependent (runtime_py.pyd on Windows /
# runtime_py.so on Linux & macOS).
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"


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
    """Token counting + sanitize still produce correct pure-Python results."""
    code = (
        "from kimi_cli.utils.tokens import _estimate_chars_tokens, _is_cjk_text; "
        "from kimi_cli.safety_check import sanitize_for_tokenizer, clean_text; "
        "assert _estimate_chars_tokens('hello world this is a test') > 0; "
        "assert _is_cjk_text('中文测试') is True; "
        "assert sanitize_for_tokenizer('a' * 500, max_chars=100) == 'a' * 100; "
        "assert clean_text('\\u200bhidden\\u200d') == 'hidden'"
    )
    proc = _run_with_native_disabled(code)
    assert proc.returncode == 0, proc.stderr


def test_loader_reports_fallback_without_native():
    """A fresh interpreter with the native runtime disabled degrades gracefully:
    no raise, fallback version marker, all gates False."""
    code = (
        "import kimi_cli.native_loader as n; "
        "print('AVAILABLE=%s' % n.NATIVE_AVAILABLE); "
        "print('VERSION=%s' % n.version()); "
        "print('TEXT=%s' % n.use_native('TEXT'))"
    )
    proc = _run_with_native_disabled(code)
    assert proc.returncode == 0, proc.stderr
    assert "AVAILABLE=False" in proc.stdout
    version_file = os.path.join(_REPO, "KIMIX_NATIVE_VERSION")
    with open(version_file, "r", encoding="utf-8") as fh:
        expected_version = fh.read().strip()
    assert f"VERSION=kimix-native {expected_version} (python fallback)" in proc.stdout
    assert "TEXT=False" in proc.stdout


def test_consumers_work_after_restore():
    """After the fixture restores the binaries, native works again (auto mode)."""
    if os.environ.get("KIMIX_NATIVE") == "0":
        pytest.skip("KIMIX_NATIVE=0 forces pure Python; nothing to restore-check")
    import importlib

    import kimi_cli.native_loader as knl

    if os.path.isfile(os.path.join(_BIN, _NATIVE_FILE)):
        # Reload the loader module so it re-resolves the staged binaries.
        import kimi_cli.native_loader as xn

        importlib.reload(xn)
        assert xn.NATIVE_AVAILABLE is True
        assert "fallback" not in knl.version()
