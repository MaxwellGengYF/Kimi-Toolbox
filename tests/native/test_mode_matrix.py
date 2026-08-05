"""Mode-matrix test for src/kimix kernels: KIMIX_NATIVE=0 vs auto identical."""

from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPUS = [
    "plain text",
    "line1\nline2\nline3",
    "\x1b[31mred\x1b[0m text\r\ncrlf",
    "中文混合 content",
    "repeat\nrepeat\nrepeat\nrepeat\nunique",
    "x" * 2000,
]

SNIPPET = r"""
import json
from kimix.tools.common import filter_output, _dedup_output
from kimix.tools.file.bash.bash_fix import fix_bash_command
from kimix.tools.file.bash.bash_tool import _process_unquoted
out = {"filter": [], "dedup": [], "bashfix": [], "unquoted": []}
for t in CORPUS:
    out["filter"].append(filter_output(t))
    out["dedup"].append(_dedup_output(t, 3, max_block_lines=1))
    out["bashfix"].append(fix_bash_command(t + " && rev x").command)
    out["unquoted"].append(_process_unquoted(t))
print(json.dumps(out, sort_keys=True, ensure_ascii=False))
"""


def _run(mode: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = mode
    proc = subprocess.run(
        [sys.executable, "-c", SNIPPET.replace("CORPUS", repr(CORPUS))],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_mode_matrix_identical():
    outs = {"0": _run("0"), "auto": _run("auto")}
    assert outs["0"] == outs["auto"], (
        "KIMIX_NATIVE=0 and auto outputs differ:\n"
        f"  0   = {outs['0'][:400]}\n"
        f"  auto= {outs['auto'][:400]}"
    )


def test_mode_matrix_kernel_toggle_parity():
    """KIMIX_NATIVE_STREAM=0 (others native) still matches the 0-mode baseline."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = "auto"
    env["KIMIX_NATIVE_STREAM"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", SNIPPET.replace("CORPUS", repr(CORPUS))],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == json.loads(_run("0"))
