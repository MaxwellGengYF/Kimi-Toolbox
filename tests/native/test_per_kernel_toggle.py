"""Per-kernel toggle test (plan 4.2): ``KIMIX_NATIVE_<KERNEL>=0`` flips ONLY
that kernel while the others stay native — verified in a clean subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The staged artifact name is platform-dependent (runtime_py.pyd on Windows /
# runtime_py.so on Linux & macOS).
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"

ALL_KERNELS = [
    "TEXT",
    "INDEX",
    "SEARCH",
    "PARSE",
    "TOOLS",
    "STREAM",
    "CODEC",
    "GLOB",
]


def _run(env_extra: dict[str, str]) -> dict[str, bool]:
    code = (
        "import kimi_cli.native_loader as n;"
        "print({%s})"
        % ", ".join(f"{k!r}: n.use_native({k!r})" for k in ALL_KERNELS)
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return eval(proc.stdout.strip())


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_kernel_toggle_flips_only_one(kernel):
    if not os.path.isfile(os.path.join(_REPO, "bin", _NATIVE_FILE)):
        pytest.skip("native runtime not staged")
    baseline = _run({"KIMIX_NATIVE": "auto"})
    if not all(baseline.values()):
        # The staged binaries can be momentarily unavailable (the missing-dll
        # compat fixture renames them out of the way); re-sync once and retry.
        sync = os.path.join(_REPO, "tools", "sync_native.py")
        if os.path.isfile(sync):
            subprocess.run([sys.executable, sync], capture_output=True, text=True, timeout=300)
            baseline = _run({"KIMIX_NATIVE": "auto"})
    assert all(baseline.values()), f"expected all kernels native, got {baseline}"
    toggled = _run({"KIMIX_NATIVE": "auto", f"KIMIX_NATIVE_{kernel}": "0"})
    assert toggled[kernel] is False
    for other in ALL_KERNELS:
        if other != kernel:
            assert toggled[other] is True, (
                f"KIMIX_NATIVE_{kernel}=0 unexpectedly disabled {other}"
            )


def test_all_kernels_off_equals_python():
    env = {"KIMIX_NATIVE": "auto"}
    env.update({f"KIMIX_NATIVE_{k}": "0" for k in ALL_KERNELS})
    all_off = _run(env)
    assert all(not v for v in all_off.values())
