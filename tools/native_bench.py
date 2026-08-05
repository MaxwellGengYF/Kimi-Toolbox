"""native_bench — compare native vs pure-Python kernel timings.

Run ``python tools\\sync_native.py`` first (or pass ``--mode`` to have it done
automatically), then::

    python tools\\native_bench.py                 # native (auto) vs python
    python tools\\native_bench.py --mode 0        # pure-Python baseline only

Each benchmark runs the SAME workload under ``KIMIX_NATIVE=0`` (pure Python)
and ``KIMIX_NATIVE=auto`` (native when staged) and prints a comparison table.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_snippet(mode: str, snippet: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KIMIX_NATIVE")}
    env["KIMIX_NATIVE"] = mode
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bench snippet failed in mode={mode}:\n{proc.stderr[-2000:]}")
    return proc.stdout


# Each benchmark: (name, snippet) where snippet times `fn` itself and prints
# "RESULT <name> <seconds>" plus an optional checksum line.
BENCHMARKS: dict[str, str] = {
    "sanitize_1mb": r"""
import time
from kimi_cli.safety_check import sanitize_for_tokenizer
text = ("mixed text with \x1b[31mansi\x1b[0m and 中文内容 " * 30000)[:1_000_000]
t0 = time.perf_counter()
for _ in range(3):
    out = sanitize_for_tokenizer(text, max_chars=0)
dt = (time.perf_counter() - t0) / 3
print("RESULT sanitize_1mb %.6f" % dt)
print("CHECKSUM %d" % len(out))
""",
    "hash_line_100k": r"""
import time
from kimi_cli.tools.file.hash_line import compute_line_hash
lines = [f"line {i} with some content and 中文 {i}" for i in range(100000)]
t0 = time.perf_counter()
prev = None
for i, ln in enumerate(lines, 1):
    prev = compute_line_hash(i, ln, prev)
dt = time.perf_counter() - t0
print("RESULT hash_line_100k %.6f" % dt)
print("CHECKSUM %s" % prev)
""",
    "bm25_1000": r"""
import time
from kimix import retrieval
tok = retrieval.NgramTokenizer(n=2)
t0 = time.perf_counter()
for i in range(1000):
    tok.tokenize("the quick brown fox jumps over the lazy dog 中文内容 123")
dt = time.perf_counter() - t0
print("RESULT ngram_1000 %.6f" % dt)
print("CHECKSUM %d" % len(tok.tokenize("abc")))
""",
    "filter_output_1mb": r"""
import time
from kimix.tools.common import filter_output
text = ("\x1b[31mred\x1b[0m line " * 100000)[:1_000_000]
t0 = time.perf_counter()
for _ in range(3):
    out = filter_output(text)
dt = (time.perf_counter() - t0) / 3
print("RESULT filter_output_1mb %.6f" % dt)
print("CHECKSUM %d" % len(out))
""",
    "parse_c_1mb": r"""
import time
from kimix.parser.c_parser import CParser
src = ("// comment here\nint x = 1; /* block */\n" * 30000)[:1_000_000]
t0 = time.perf_counter()
r = CParser().parse(src)
dt = time.perf_counter() - t0
print("RESULT parse_c_1mb %.6f" % dt)
print("CHECKSUM %d" % (r.total_comments + len(r.code_without_comments)))
""",
    "export_200_turns": r"""
import time
from kosong.message import Message
from kimi_cli.utils import export
from kimi_cli.wire.types import TextPart
import pendulum
history = []
for i in range(100):
    history.append(Message(role="user", content=[TextPart(text=f"question number {i} about 中文")]))
    history.append(Message(role="assistant", content=[TextPart(text=f"answer number {i} with some detail")]))
now = pendulum.now()
t0 = time.perf_counter()
for _ in range(3):
    out = export.build_export_markdown("s1", r"C:\work", history, 5000, now)
dt = (time.perf_counter() - t0) / 3
print("RESULT export_200_turns %.6f" % dt)
print("CHECKSUM %d" % len(out))
""",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("auto", "0", "1"), default="auto",
        help="native mode to benchmark (default: auto = native when staged)",
    )
    parser.add_argument("--bench", default=None, help="run a single benchmark by name")
    args = parser.parse_args(argv)

    names = [args.bench] if args.bench else list(BENCHMARKS)
    print(f"{'benchmark':<22} {'python (s)':>12} {'native (s)':>12} {'speedup':>9}  checksum_match")
    for name in names:
        snippet = BENCHMARKS[name]
        py_out = _run_snippet("0", snippet)
        nat_out = _run_snippet(args.mode, snippet)
        py_t = float(py_out.split("RESULT")[1].split()[1])
        nat_t = float(nat_out.split("RESULT")[1].split()[1])
        py_cs = [l for l in py_out.splitlines() if l.startswith("CHECKSUM")][0]
        nat_cs = [l for l in nat_out.splitlines() if l.startswith("CHECKSUM")][0]
        speedup = py_t / nat_t if nat_t > 0 else float("inf")
        print(f"{name:<22} {py_t:>12.4f} {nat_t:>12.4f} {speedup:>8.2f}x  {py_cs == nat_cs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
