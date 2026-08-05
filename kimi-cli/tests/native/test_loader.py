"""Loader behavior tests: NATIVE_AVAILABLE / use_native / version / path.

These run in-process against the currently staged binaries (see conftest.py
which runs tools\\sync_native.py as a session fixture).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import kimi_cli.native_loader as knl

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_WORKDIR_BIN = os.path.join(_REPO_ROOT, "bin")


def test_default_path_is_workdir_bin():
    """Without KIMIX_NATIVE_PATH the loader resolves <repo>\\bin."""
    if not knl.NATIVE_AVAILABLE:
        pytest.skip("native runtime not staged")
    assert knl.NATIVE_PATH == _WORKDIR_BIN


def test_version_shape():
    v = knl.version()
    if knl.NATIVE_AVAILABLE:
        assert "fallback" not in v and v.strip()
    else:
        assert "fallback" in v


def test_get_module_returns_submodule_or_none():
    if knl.NATIVE_AVAILABLE:
        mod = knl.get_module("text")
        assert mod is not None
        assert callable(getattr(mod, "estimate_chars_tokens", None))
        assert knl.get_module("does_not_exist") is None
    else:
        assert knl.get_module("text") is None


def test_use_native_consistent_with_availability():
    for kernel in ("TEXT", "INDEX", "SEARCH", "PARSE", "SOUL", "TOOLS", "STREAM",
                   "CODEC", "JSON", "CONCURRENCY"):
        assert knl.use_native(kernel) is (knl.NATIVE_AVAILABLE and True)


def test_attribute_submodule_access():
    if knl.NATIVE_AVAILABLE:
        assert knl.text is not None
    else:
        with pytest.raises(AttributeError):
            _ = knl.text


@pytest.mark.parametrize(
    "env",
    [
        {"KIMIX_NATIVE": "0"},
        {"KIMIX_NATIVE": "auto"},
        {"KIMIX_NATIVE": "1"},
    ],
)
def test_env_matrix_subprocess(env):
    """Full env contract exercised in a clean interpreter.

    Expectations are derived from the *subprocess* env plus the on-disk
    staged binaries (never from the parent process state, so the test is
    independent of the parent's KIMIX_NATIVE).
    """
    code = (
        "import kimi_cli.native_loader as n;"
        "print('AVAILABLE=%s' % n.NATIVE_AVAILABLE);"
        "print('VERSION=%s' % n.version())"
    )
    env_full = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env_full.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env_full,
        timeout=120,
    )
    staged = os.path.isfile(os.path.join(os.environ.get("KIMIX_NATIVE_PATH", _WORKDIR_BIN), "runtime_py.pyd"))
    mode = env["KIMIX_NATIVE"]
    if mode == "0":
        assert proc.returncode == 0
        assert "AVAILABLE=False" in proc.stdout
        assert "fallback" in proc.stdout
    elif mode == "1":
        if staged:
            assert proc.returncode == 0
            assert "AVAILABLE=True" in proc.stdout
        else:
            # Documented contract: KIMIX_NATIVE=1 + missing pyd raises.
            assert proc.returncode != 0
            assert "ImportError" in proc.stderr or "ModuleNotFoundError" in proc.stderr
    else:  # auto
        if staged:
            assert proc.returncode == 0
            assert "AVAILABLE=True" in proc.stdout
        else:
            assert proc.returncode == 0
            assert "AVAILABLE=False" in proc.stdout


def test_per_kernel_toggle_subprocess():
    """KIMIX_NATIVE_TEXT=0 disables only TEXT."""
    if not knl.NATIVE_AVAILABLE:
        pytest.skip("native runtime not staged")
    code = (
        "import kimi_cli.native_loader as n;"
        "print('TEXT=%s INDEX=%s' % (n.use_native('TEXT'), n.use_native('INDEX')))"
    )
    env_full = dict(os.environ)
    env_full["KIMIX_NATIVE_TEXT"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env_full, timeout=120
    )
    assert proc.returncode == 0
    assert "TEXT=False INDEX=True" in proc.stdout
