"""Behavior-equivalence tests for micro-compression TOOLS kernels (plan 016).

Compares the native C++ kernels (via kimix_native.tools.compress_*) against
the pure-Python reference implementations in
kimi-cli/src/kimi_cli/tools/file/micro_compress.py.
"""

from __future__ import annotations

import os
import random
import string
import sys
from typing import Any

import pytest

# Ensure we test the local kimi-cli source tree, not an installed wheel.
_KIMI_CLI_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "kimi-cli", "src")
if os.path.isdir(_KIMI_CLI_SRC) and _KIMI_CLI_SRC not in sys.path:
    sys.path.insert(0, _KIMI_CLI_SRC)

import importlib.util


_NATIVE_LOADER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "kimix", "native_loader.py"
)


def _load_native_loader():
    spec = importlib.util.spec_from_file_location("native_loader", _NATIVE_LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_native_loader = _load_native_loader()
NATIVE_AVAILABLE = _native_loader.NATIVE_AVAILABLE
use_native = _native_loader.use_native

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)

_MICRO_COMPRESS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "kimi-cli", "src", "kimi_cli", "tools", "file", "micro_compress.py"
)


def _load_micro_compress():
    name = "kimi_cli.tools.file.micro_compress"
    spec = importlib.util.spec_from_file_location(name, _MICRO_COMPRESS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_mc = _load_micro_compress()
MicroCompressConfig = _mc.MicroCompressConfig
collapse_whitespace = _mc.collapse_whitespace
intra_line_dedup = _mc.intra_line_dedup
renumber_lines = _mc.renumber_lines
strip_control_noise = _mc.strip_control_noise


def _rand_ascii(n: int) -> str:
    chars = string.ascii_letters + string.digits + string.punctuation + " \t\n\r"
    return "".join(random.choices(chars, k=n))


# ---------------------------------------------------------------------------
# strip_control_noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text no escapes\n",
        "\x1b[31mred\x1b[0m text\n",
        "\x1b]0;title\x07OSC title\x1b\\ end\n",
        "frame1\rframe2\nframe3\rframe4\n",
        "line1\r\nline2\r\n",
        "mixed \x1b[1mbold\x1b[0m and \rprogress\rdone\n",
        "a" * 4096 + "\rfinal\n",
    ],
)
def test_strip_control_noise_equivalence(text: str) -> None:
    from kimix_native import tools as native_tools

    native = native_tools.compress_strip_control_noise(text)
    python = strip_control_noise(text)
    assert native == python, f"strip_control_noise mismatch for {text!r}\n  native={native!r}\n  python={python!r}"


# ---------------------------------------------------------------------------
# renumber_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  1\ta\n  2\tb\n  3\tc\n",
        "  1\ta\nnot numbered\n  3\tc\n",
        "\n  1\ta\n[meta]\n  2\tb\n",
        "1\ta\n2\tb\n",
        "\n\n[meta]\n",
        _rand_ascii(200),
        "  1\t" + _rand_ascii(500) + "\n  2\t" + _rand_ascii(500) + "\n",
    ],
)
def test_renumber_lines_equivalence(text: str) -> None:
    from kimix_native import tools as native_tools

    native = native_tools.compress_renumber_lines(text)
    python = renumber_lines(text)
    assert native == python, f"renumber_lines mismatch for {text!r}\n  native={native!r}\n  python={python!r}"


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------


def _rand_config() -> MicroCompressConfig:
    return MicroCompressConfig(
        lossless_only=random.choice([True, False]),
        strip_trailing_ws=random.choice([True, False]),
        blank_line_collapse=random.randint(-1, 3),
        common_indent_factor=random.choice([True, False]),
        prefix_fold=random.choice([True, False]),
    )


@pytest.mark.parametrize(
    ("text", "kind", "config"),
    [
        ("", "log", MicroCompressConfig()),
        ("hello   \tworld \n", "log", MicroCompressConfig()),
        ("hello\t\nworld   \n", "code", MicroCompressConfig()),
        ("a\n\n\n\nb\n", "log", MicroCompressConfig(blank_line_collapse=1)),
        ("a\n\n\nb\n", "log", MicroCompressConfig(blank_line_collapse=0)),
        ("    line one\n    line two\n    line three\n", "log", MicroCompressConfig()),
        ("    line one\n    line two\n", "code", MicroCompressConfig()),
        ("    line one\n    line two\n", "log", MicroCompressConfig(lossless_only=True)),
        ("a   b     c", "log", MicroCompressConfig()),
        ("   a   b   ", "log", MicroCompressConfig()),
        ("a   b     c", "data", MicroCompressConfig()),
        ("already\nclean\n", "log", MicroCompressConfig()),
        ("$jWa$thSx4n3j\t@Wvn0qoc,>3)/e[~eUDy>S|xkgCx}~SfwbS5Ti&\r7AlC>F3+Wk\rrs>Ag$-|wc_} E~__{PCuB=\nK_,.RzD\tkx83mZ+!Q#W \nq\"[\" w`*|}\"uY#W7@(jh@Mb!nyk`\"M,;is+cfmR23kI^E.ByI#g=nVa&2{_t{krJEj.X\"1F-Jr*fCr/$$zYz~z'=*M", "log", MicroCompressConfig(lossless_only=True)),
    ]
    + [
        (_rand_ascii(random.randint(0, 500)), random.choice(["log", "prose", "code", "data"]), _rand_config())
        for _ in range(50)
    ],
)
def test_collapse_whitespace_equivalence(text: str, kind: str, config: MicroCompressConfig) -> None:
    from kimix_native import tools as native_tools

    native = native_tools.compress_collapse_whitespace(
        text,
        kind,
        config,
    )
    python = collapse_whitespace(text, kind=kind, config=config)
    assert native == python, (
        f"collapse_whitespace mismatch for kind={kind!r} config={config!r}\n"
        f"  text={text[:200]!r}\n  native={native[:200]!r}\n  python={python[:200]!r}"
    )


# ---------------------------------------------------------------------------
# intra_line_dedup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "threshold"),
    [
        ("", 10),
        ("abcabc", 10),
        ("abc" * 1000 + "\n", 2000),
        ("abc" * 1000 + "\n", 3000),
        ("abcd" * 500, 1000),
        ("ab" * 1000, 1000),
        ("short\n" + "xy" * 1500 + "\nalso short\n", 2000),
        ("\n\n", 10),
    ]
    + [
        ("\n".join(
            (_rand_ascii(random.randint(1, 10)).replace("\n", " ") * random.randint(1, 300))
            if random.random() < 0.3
            else _rand_ascii(random.randint(0, 100)).replace("\n", " ")
            for _ in range(random.randint(0, 8))
        ), random.randint(0, 3000))
        for _ in range(100)
    ],
)
def test_intra_line_dedup_equivalence(text: str, threshold: int) -> None:
    from kimix_native import tools as native_tools

    config = MicroCompressConfig(intra_line_dedup_len=threshold)
    # The public shim always applies the threshold/max-unit gate itself.
    native = native_tools.compress_intra_line_dedup(text, threshold, 2048)
    python = intra_line_dedup(text, kind="log", config=config)
    assert native == python, (
        f"intra_line_dedup mismatch for threshold={threshold}\n"
        f"  text={text[:200]!r}\n  native={native[:200]!r}\n  python={python[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Benchmark / performance gate
# ---------------------------------------------------------------------------


def _make_repeating_log(size: int, unit: str = "ERR: connection reset ") -> str:
    return (unit * (size // len(unit)))[:size]


def test_compress_native_is_faster() -> None:
    """Native compress kernels must be at least 2x faster on realistic input."""
    from kimix_native import tools as native_tools

    if not use_native("TOOLS"):
        pytest.skip("native TOOLS gate is off")

    text = _make_repeating_log(10 * 1024 * 1024)
    config = MicroCompressConfig()

    # Warm up and verify correctness.
    native_out = native_tools.compress_collapse_whitespace(text, "log", config)
    python_out = collapse_whitespace(text, kind="log", config=config)
    assert native_out == python_out

    import time

    def measure(fn: Any, *args: Any, rounds: int = 3) -> float:
        times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        return min(times)

    native_t = measure(native_tools.compress_collapse_whitespace, text, "log", config)
    python_t = measure(
        native_tools._compat_compress_collapse_whitespace, text, "log", config
    )

    # If the native kernel is not at least 2x faster, the failure criteria
    # require us to gate it off.  We surface that as a test failure so the
    # engineer can decide whether to revert the wire.
    assert native_t * 2 < python_t, (
        f"native collapse_whitespace is not >2x faster: native={native_t:.3f}s python={python_t:.3f}s"
    )

    # intra_line_dedup benchmark.
    long_line = "abc" * 2_000_000
    native_out = native_tools.compress_intra_line_dedup(long_line, 2000, 2048)
    python_out = intra_line_dedup(long_line, kind="log", config=config)
    assert native_out == python_out

    native_t = measure(native_tools.compress_intra_line_dedup, long_line, 2000, 2048)
    python_t = measure(
        native_tools._compat_compress_intra_line_dedup, long_line, 2000, 2048
    )

    assert native_t * 2 < python_t, (
        f"native intra_line_dedup is not >2x faster: native={native_t:.3f}s python={python_t:.3f}s"
    )
