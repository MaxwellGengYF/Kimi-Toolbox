"""Session setup for the native test suites.

Runs ``tools\\sync_native.py`` once per session (idempotent; cheap when the
binaries are already staged). When no native build exists in kimix-base the
sync prints an error and native-dependent tests skip themselves with a clear
message — the pure-Python suites stay green.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="session", autouse=True)
def _sync_native_session():
    sync_script = os.path.join(_REPO_ROOT, "tools", "sync_native.py")
    if not os.path.isfile(sync_script):
        yield
        return
    proc = subprocess.run(
        [sys.executable, sync_script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Non-zero exit means no build found: native tests skip via
    # NATIVE_AVAILABLE=False. Never fail the session for that.
    yield
