"""Compat test: consumers must work (pure Python) when the native runtime is
missing — simulated by temporarily renaming the staged ``runtime_py.pyd`` +
``runtime.dll`` out of the way, exercising the consumers, then restoring.
"""

from __future__ import annotations

import importlib
import os
import shutil

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_BIN = os.path.join(_REPO, "bin")


@pytest.fixture
def native_missing(tmp_path):
    """Rename staged native artifacts away; restore afterwards."""
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
    """Token counting + sanitize still produce correct pure-Python results."""
    from kimi_cli.utils.tokens import _estimate_chars_tokens, _is_cjk_text
    from kimi_cli.safety_check import sanitize_for_tokenizer, clean_text

    assert _estimate_chars_tokens("hello world this is a test") > 0
    assert _is_cjk_text("中文测试") is True
    assert sanitize_for_tokenizer("a" * 500, max_chars=100) == "a" * 100
    assert clean_text("\u200bhidden\u200d") == "hidden"


def test_loader_reports_fallback_without_native(native_missing):
    """A fresh interpreter with the binaries absent degrades gracefully:
    no raise, fallback version marker, all gates False."""
    import subprocess
    import sys

    code = (
        "import kimi_cli.native_loader as n;"
        "print('AVAILABLE=%s' % n.NATIVE_AVAILABLE);"
        "print('VERSION=%s' % n.version());"
        "print('TEXT=%s' % n.use_native('TEXT'))"
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
    assert "VERSION=kimix-native 0.1.0 (python fallback)" in proc.stdout
    assert "TEXT=False" in proc.stdout


def test_consumers_work_after_restore():
    """After the fixture restores the binaries, native works again (auto mode)."""
    if os.environ.get("KIMIX_NATIVE") == "0":
        pytest.skip("KIMIX_NATIVE=0 forces pure Python; nothing to restore-check")
    import kimi_cli.native_loader as knl

    if os.path.isfile(os.path.join(_BIN, "runtime_py.pyd")):
        # Reload the loader module so it re-resolves the staged binaries.
        import kimix.native_loader as xn

        importlib.reload(xn)
        assert xn.NATIVE_AVAILABLE is True
        assert "fallback" not in knl.version()
