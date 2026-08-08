"""Loader behavior tests for kimix.native_loader (root package).

Mirrors kimi-cli/tests/native/test_loader.py; exercises NATIVE_AVAILABLE /
use_native / version / NATIVE_PATH behavior under the env matrix.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import kimix.native_loader as knl

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKDIR_BIN = os.path.join(_REPO_ROOT, "bin")

# The staged artifact name is platform-dependent (runtime_py.pyd on Windows /
# runtime_py.so on Linux & macOS).
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"


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
        mod = knl.get_module("stream")
        assert mod is not None
        assert callable(getattr(mod, "filter_output", None))
        assert knl.get_module("does_not_exist") is None
    else:
        assert knl.get_module("stream") is None


def test_use_native_consistent_with_availability():
    for kernel in ("TEXT", "INDEX", "SEARCH", "PARSE", "SOUL", "TOOLS", "STREAM",
                   "CODEC", "JSON", "CONCURRENCY"):
        assert knl.use_native(kernel) is (knl.NATIVE_AVAILABLE and True)


def test_use_native_case_variants():
    """Upper/lower/title spellings of a known kernel resolve identically."""
    for kernel in knl._KERNELS:
        expected = knl.use_native(kernel)
        assert knl.use_native(kernel.lower()) is expected
        assert knl.use_native(kernel.title()) is expected


def test_kernel_table_precomputed():
    """The precomputed kernel table matches use_native for every spelling."""
    table = knl._KERNEL_TABLE
    assert table is not None
    for kernel in knl._KERNELS:
        expected = knl.use_native(kernel)
        assert table[kernel] is expected
        assert table[kernel.lower()] is expected
        assert table[kernel.title()] is expected


def test_module_table_precomputed():
    """The precomputed module table holds resolved submodules (or None)."""
    table = knl._MODULE_TABLE
    assert knl.get_module("index") is table["index"]
    assert knl.get_module("does_not_exist") is None
    if knl.NATIVE_AVAILABLE:
        assert table["index"] is not None
        assert callable(getattr(table["search"], "jaro_similarity", None))
    else:
        assert table["index"] is None


def test_use_native_unknown_kernel_memoized():
    """Unknown kernel names resolve via the shim and are memoized."""
    first = knl.use_native("FROBNICATE")
    assert isinstance(first, bool)
    assert knl._KERNEL_TABLE["FROBNICATE"] is first
    assert knl.use_native("FROBNICATE") is first


def test_hot_call_sites_hoist_native_modules():
    """Consuming modules resolve native submodules once at import time."""
    import kimix.retrieval as retrieval

    assert retrieval._NATIVE_INDEX is knl.get_module("index")
    assert retrieval._NATIVE_SEARCH is knl.get_module("search")


def test_attribute_submodule_access():
    if knl.NATIVE_AVAILABLE:
        assert knl.stream is not None
    else:
        with pytest.raises(AttributeError):
            _ = knl.stream


@pytest.mark.parametrize(
    "env",
    [
        {"KIMIX_NATIVE": "0"},
        {"KIMIX_NATIVE": "auto"},
        {"KIMIX_NATIVE": "1"},
    ],
)
def test_env_matrix_subprocess(env):
    """Full env contract exercised in a clean interpreter."""
    code = (
        "import kimix.native_loader as n;"
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
    staged = os.path.isfile(os.path.join(_WORKDIR_BIN, _NATIVE_FILE))
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
        "import kimix.native_loader as n;"
        "print('TEXT=%s INDEX=%s' % (n.use_native('TEXT'), n.use_native('INDEX')))"
    )
    env_full = dict(os.environ)
    env_full["KIMIX_NATIVE_TEXT"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env_full, timeout=120
    )
    assert proc.returncode == 0
    assert "TEXT=False INDEX=True" in proc.stdout
