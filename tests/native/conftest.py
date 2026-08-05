"""Session setup for the root native test suites (mirror kimi-cli's)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def _sync_native_session():
    sync_script = os.path.join(_REPO_ROOT, "tools", "sync_native.py")
    if not os.path.isfile(sync_script):
        yield
        return
    subprocess.run([sys.executable, sync_script], capture_output=True, text=True, timeout=300)
    yield
