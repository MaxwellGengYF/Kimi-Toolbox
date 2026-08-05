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


def test_dev_fallback_defaults_to_sibling_kimix_base(monkeypatch):
    """Without $KIMIX_BASE the dev-only fallback is the sibling kimix-base repo
    (no absolute path baked in)."""
    monkeypatch.delenv("KIMIX_BASE", raising=False)
    from kimix.native_loader import _candidate_dirs, _dev_base

    assert _dev_base() == os.path.join(os.path.dirname(_REPO_ROOT), "kimix-base")
    dirs = _candidate_dirs()
    assert dirs[0] == _WORKDIR_BIN  # <repo>\bin stays the default first candidate
    for mode in ("release", "releasedbg", "debug"):
        expected = os.path.join(
            os.path.dirname(_REPO_ROOT), "kimix-base", "bin", mode
        )
        assert expected in dirs


def test_kimix_base_env_repoints_dev_fallback(monkeypatch):
    """$KIMIX_BASE overrides the sibling default for the dev-only fallback."""
    fake = os.path.join("some", "other", "kimix-base")
    monkeypatch.setenv("KIMIX_BASE", fake)
    from kimix.native_loader import _candidate_dirs, _dev_base

    assert _dev_base() == fake
    dirs = _candidate_dirs()
    assert dirs[0] == _WORKDIR_BIN  # default <repo>\bin remains first
    for mode in ("release", "releasedbg", "debug"):
        assert os.path.join(fake, "bin", mode) in dirs


def test_kimix_native_path_still_short_circuits(monkeypatch):
    """KIMIX_NATIVE_PATH is the explicit override and ignores KIMIX_BASE."""
    monkeypatch.setenv("KIMIX_NATIVE_PATH", os.path.join("explicit", "native"))
    monkeypatch.setenv("KIMIX_BASE", os.path.join("other", "kimix-base"))
    from kimix.native_loader import _candidate_dirs

    assert _candidate_dirs() == [os.path.join("explicit", "native")]


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
