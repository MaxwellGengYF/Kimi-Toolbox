"""Mode-matrix test: KIMIX_NATIVE=0 vs auto must produce identical results.

Runs a representative corpus through the integrated app functions in a clean
subprocess under ``KIMIX_NATIVE=0`` and ``KIMIX_NATIVE=auto``, then compares
the serialized outputs byte-for-byte.
"""

from __future__ import annotations

import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CORPUS = [
    "hello world",
    "中文测试文本内容",
    "a" * 300,
    "mixed 中文 🎉 with \x1b[31mansi\x1b[0m",
    "\u200bzero\u200dwidth\ufeff",
    "x" * 100 + "\r\n" + "y" * 100,
]

SNIPPET = r"""
import json
from kimi_cli.utils.tokens import _estimate_chars_tokens, _is_cjk_text
from kimi_cli.safety_check import sanitize_for_tokenizer, clean_text
from kimi_cli.tools.file.hash_line import compute_line_hash
out = {"estimate": [], "cjk": [], "sanitize": [], "clean": [], "hash": []}
for t in CORPUS:
    out["estimate"].append(_estimate_chars_tokens(t))
    out["cjk"].append(_is_cjk_text(t))
    out["sanitize"].append(sanitize_for_tokenizer(t, max_chars=80))
    out["clean"].append(clean_text(t))
    out["hash"].append(compute_line_hash(7, t, "ZM"))
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
    """0-mode and auto-mode outputs must be byte-identical."""
    outs = {"0": _run("0"), "auto": _run("auto")}
    assert outs["0"] == outs["auto"], (
        "KIMIX_NATIVE=0 and auto outputs differ:\n"
        f"  0   = {outs['0'][:400]}\n"
        f"  auto= {outs['auto'][:400]}"
    )


def test_mode_matrix_kernel_toggle_parity():
    """Per-kernel off (TEXT=0) with the rest native still matches 0-mode."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = "auto"
    env["KIMIX_NATIVE_TEXT"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", SNIPPET.replace("CORPUS", repr(CORPUS))],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    # estimate/cjk/sanitize/clean are TEXT kernels -> python; hash is TOOLS ->
    # native. Both must still equal the all-0 baseline per kernel.
    baseline = __import__("json").loads(_run("0"))
    toggled = __import__("json").loads(proc.stdout)
    assert toggled == baseline
