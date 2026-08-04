"""Fuzz the _ToolCallStreamPrinter: randomly split realistic tool-call JSON
(especially Python-tool `code` values) into fragments and compare the
decoded printed output against orjson ground truth.

- Valid documents: exact-match the expected plain output
  (``\\ncode:\\n<code><compact args>`` + terminating newline).
- Invalid documents (malformed escapes etc.): assert no JSON scaffolding
  leaks into the streamed code and the printer recovers cleanly.

Usage: python scripts/fuzz_stream_printer.py [iterations] [seed]
"""

from __future__ import annotations

import random
import sys

import orjson

from kimix.ui import stream as stream_mod
from kimix.ui.printing import PrintStream, _strip_ansi


class FakeSession:
    def __init__(self) -> None:
        self._tmp_data = {}
        self.status = None


def _build_sampler() -> list[str]:
    """A pool of realistic string values that appear in Python-tool calls."""
    return [
        "print('hello')",
        "import os\nprint(os.getcwd())",
        'print("hello world")',
        'data = {"a": 1, "b": [1, 2, 3]}',
        'print("line1\\nline2")',
        "path = r\"C:\\dev\\kimi-agent\\src\"",
        "open(r'C:\\Users\\foo\\bar.txt')",
        'print("\\u4f60\\u597d")',
        'x = "\\ud83d\\ude00"',
        'print("\\u001b[31mred\\u001b[0m")',
        'print(r"\\u0041")',
        "def f():\n\treturn 42",
        "print(1)\r\nprint(2)",
        'print("a \\"quoted\\" string")',
        'print("C:\\\\temp\\\\x")',
        "print(" + ",".join(f"'{i}'" for i in range(200)) + ")",
        "text = '{\"a\": 1}'",
        "print('hello 😀 world')",
        "print('你好世界')",
        "line1\\nline2",
        "abc\\",
        "a\\ud83d",
        "b\\ude00",
        "tab\there",
        "a\\u12",
        "x = 0x1b + 'y'",
        "s = '\\''"
        "print('{}'.format(1))",
        "print(f\"{1 + 2}\")",
        "# comment with 'quote' and \"double\"",
        "s = '''triple\\nquoted'''",
        # Single-backslash Windows paths / raw-string slips (invalid JSON
        # but emitted by real LLMs):
        "p = r'C:\\dev\\src'",
        "open('C:\\temp\\x')",
        'print(r"\\u")',
        "q = r'C:\\users\\x'",
    ]


def _make_json_doc(rng: random.Random) -> str:
    """Build a realistic Python tool-call arguments document."""
    code = rng.choice(_build_sampler())
    args: dict[str, object] = {"code": code}
    if rng.random() < 0.5:
        args["timeout"] = rng.choice([30, 60, 120])
    if rng.random() < 0.5:
        args["mode"] = rng.choice(["execute", "run", "send", "interactive"])
    if rng.random() < 0.5:
        args["max_lines"] = rng.choice([10, 50, None])
    if rng.random() < 0.4:
        args["token_kill"] = rng.choice([True, False])
    if rng.random() < 0.3:
        args["output_path"] = rng.choice(["out.txt", None])
    return orjson.dumps(args).decode("utf-8")


def _split(doc: str, rng: random.Random, max_frags: int) -> list[str]:
    """Split doc into 1..max_frags fragments at arbitrary char boundaries,
    including adversarial cuts right after backslashes / after `\\u`."""
    if len(doc) <= 2 or rng.random() < 0.12:
        return [doc]
    n = min(rng.randint(2, max_frags), len(doc))
    cut_count = min(n - 1, len(doc) - 1)
    # Adversarial bias: force cuts right after backslashes and after `\\u`
    # so escape sequences are exercised at their most fragile boundaries.
    after_bs = [c for c in range(1, len(doc)) if doc[c - 1] == "\\"]
    after_u = [c for c in range(2, len(doc)) if doc[c - 2:c] == "\\u"]
    hot = sorted(set(after_bs) | set(after_u))
    cuts: set[int] = set()
    if hot and rng.random() < 0.6:
        for c in rng.sample(hot, min(len(hot), rng.randint(1, 2))):
            cuts.add(c)
    while len(cuts) < cut_count:
        c = rng.randint(1, len(doc) - 1)
        cuts.add(c)
    cuts = sorted(cuts)[:cut_count]
    prev = 0
    out = []
    for c in cuts:
        out.append(doc[prev:c])
        prev = c
    out.append(doc[prev:])
    return out


def _fmt_compact(key: str, value: object) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        text = "True" if value else "False"
    elif value is None:
        text = "None"
    else:
        text = str(value)
    return f" {stream_mod._canonical_key(key)}:{text}"


def _run_one(doc: str, frags: list[str]) -> tuple[str, bool, str, int]:
    captured: list[str] = []
    ps = PrintStream(print_func=lambda *a, **k: captured.append(
        "".join(str(v) for v in a) + k.get("end", "")))
    stream_mod._stream = ps
    session = FakeSession()
    printer = stream_mod._ToolCallStreamPrinter("Python", session)
    for f in frags:
        printer.feed(f)
    printer.finish()
    plain = _strip_ansi("".join(captured))
    return plain, printer._broken, "", printer._state


def _check(doc: str, plain: str) -> str | None:
    """Return an error message, or None when the output is correct."""
    try:
        parsed = orjson.loads(doc)
    except (orjson.JSONDecodeError, TypeError, ValueError):
        # Invalid document: the lexer must recover without leaking JSON
        # scaffolding into the streamed code and without raising.
        if '"}' in plain:
            return f"JSON scaffold leaked into output: {plain!r}"
        return None
    code = parsed.get("code", "")
    if not isinstance(code, str):
        code = ""
    compact = "".join(_fmt_compact(k, v) for k, v in parsed.items() if k != "code")
    expected = f"\ncode:\n{code}{compact}"
    if not expected.endswith("\n"):
        expected += "\n"
    if plain != expected:
        return f"expected={expected!r} got={plain!r}"
    return None


def main(iterations: int = 5000, seed: int = 0) -> int:
    rng = random.Random(seed)
    fails = 0
    for it in range(iterations):
        doc = _make_json_doc(rng)
        frags = _split(doc, rng, max_frags=20)
        plain, broken, _leak, _state = _run_one(doc, frags)
        err = _check(doc, plain)
        if err is not None:
            fails += 1
            print(f"[FAIL {it}] seed={seed} doc={doc!r}")
            print(f"    fragments={frags!r}")
            print(f"    {err}")
            if fails >= 5:
                break
    print(f"seed={seed}: {iterations} iterations, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    seeds = [int(s) for s in sys.argv[2:]] or [0]
    code = 0
    for seed in seeds:
        code |= main(iterations, seed)
    raise SystemExit(code)
