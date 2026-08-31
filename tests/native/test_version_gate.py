"""Version-gate tests for the ``kimix_native`` shim.

The shim only activates the compiled ``runtime_py`` module when the version it
reports matches the repo-root ``KIMIX_NATIVE_VERSION`` marker.  Every scenario
is exercised in a FRESH subprocess with a **copied** staging tree under a
tempdir, so the real repo marker is never touched:

* matching marker -> native enabled (``_native`` set, ``DISABLE_REASON`` None)
* mismatched marker -> native disabled + reason recorded + fallback version
* missing marker -> native disabled
* ``KIMIX_NATIVE=1`` + mismatch -> ImportError (require-mode contract)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BIN = _REPO / "bin"
_NATIVE_FILE = "runtime_py.pyd" if sys.platform == "win32" else "runtime_py.so"

pytestmark = pytest.mark.skipif(
    not (_BIN / _NATIVE_FILE).is_file(),
    reason="native runtime not staged",
)

# Output probes run inside the temp staging tree; sys.path is pointed at the
# staged bin so ``runtime_py`` resolves next to the copied shim.
_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {bin_dir!r})
    import kimix_native as kn
    print("NATIVE=%s" % (kn._native is not None))
    print("REASON=%s" % (kn.DISABLE_REASON or ""))
    print("VERSION=%s" % kn.version())
    print("TEXT=%s" % kn.use_native("TEXT"))
    """
).strip()


def _stage(marker: str | None) -> Path:
    """Copy the shim + compiled extension into a temp repo-shaped tree.

    Returns the temp repo root; ``<root>/bin`` holds the artifacts and
    ``<root>/KIMIX_NATIVE_VERSION`` is written when *marker* is not None.
    """
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="kimix_native_gate_"))
    stage_bin = root / "bin"
    stage_bin.mkdir()
    shutil.copy2(_BIN / _NATIVE_FILE, stage_bin / _NATIVE_FILE)
    shutil.copytree(
        _BIN / "kimix_native",
        stage_bin / "kimix_native",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if marker is not None:
        (root / "KIMIX_NATIVE_VERSION").write_text(marker, encoding="utf-8")
    return root


def _run(marker: str | None, mode: str) -> subprocess.CompletedProcess[str]:
    root = _stage(marker)
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
        env["KIMIX_NATIVE"] = mode
        return subprocess.run(
            [sys.executable, "-c", _PROBE.format(bin_dir=str(root / "bin"))],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
            timeout=120,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _runtime_version() -> str:
    """The version token the staged runtime_py reports (e.g. ``1.0.0``)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(_BIN)!r}); import runtime_py; print(runtime_py.version())"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    raw = proc.stdout.strip()
    return raw.rsplit(" ", 1)[-1] if " " in raw else raw


def test_matching_version_enables_native():
    """runtime_py's reported version == marker -> native stays enabled."""
    version = _runtime_version()
    proc = _run(marker=version, mode="auto")
    assert proc.returncode == 0, proc.stderr
    assert "NATIVE=True" in proc.stdout
    assert "REASON=" in proc.stdout  # empty reason
    assert f"VERSION=kimix-runtime {version}" in proc.stdout
    assert "TEXT=True" in proc.stdout


def test_mismatched_version_disables_native():
    """runtime_py's reported version != marker -> native disabled."""
    runtime_version = _runtime_version()
    proc = _run(marker="9.9.9", mode="auto")
    assert proc.returncode == 0, proc.stderr
    assert "NATIVE=False" in proc.stdout
    assert "REASON=native runtime version mismatch" in proc.stdout
    assert f"{runtime_version!r}" in proc.stdout and "'9.9.9'" in proc.stdout
    # Fallback advertises the marker version, gates are all closed.
    assert "VERSION=kimix-native 9.9.9 (python fallback)" in proc.stdout
    assert "TEXT=False" in proc.stdout


def test_missing_marker_disables_native():
    """No KIMIX_NATIVE_VERSION marker -> native disabled, unknown fallback."""
    proc = _run(marker=None, mode="auto")
    assert proc.returncode == 0, proc.stderr
    assert "NATIVE=False" in proc.stdout
    assert "REASON=native runtime version mismatch" in proc.stdout
    assert "cannot read the KIMIX_NATIVE_VERSION marker" in proc.stdout
    assert "VERSION=kimix-native unknown (python fallback)" in proc.stdout
    assert "TEXT=False" in proc.stdout


def test_require_mode_raises_on_mismatch():
    """KIMIX_NATIVE=1 + version mismatch raises ImportError, like a missing binary."""
    proc = _run(marker="9.9.9", mode="1")
    assert proc.returncode != 0
    assert "ImportError" in proc.stderr
    assert "native runtime version mismatch" in proc.stderr
    assert "NATIVE=True" not in proc.stdout  # never usable