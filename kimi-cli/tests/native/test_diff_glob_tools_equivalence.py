"""Behavior-equivalence gate for the newly wired app kernels:

- DIFF  -> kimi_cli.utils.rich.diff_render._build_offset_map
- GLOB  -> kimi_cli.utils.file_filter._parse_ls_files_output
- TOOLS -> kimi_cli.tools.file.hash_line._cumulative_hashes (bulk cumulative
           hashing) and its two call sites (HashlineMismatchError.__str__,
           validate_anchor_ref)

Every case runs the SAME inputs through the native path (gate forced on) and
the pure-Python path (gate forced off) and asserts identical results. The
corpus is deliberately adversarial and includes the divergence-edge inputs
(trailing path segments for GLOB, trailing empty lines for TOOLS) that the
native kernels do NOT replicate — those must fall back to Python and stay
bit-identical.
"""

from __future__ import annotations

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)


def _native_on(module, state: bool):
    """Force the module's native gate to *state*; returns restore callable."""
    attr = "_native_use_native"
    original = getattr(module, attr, None)
    setattr(module, attr, lambda kernel: state)
    return lambda: setattr(module, attr, original) if original is not None else delattr(
        module, attr
    )


# ---------------------------------------------------------------------------
# DIFF kernel: _build_offset_map (raw, rendered, tab_size)
# ---------------------------------------------------------------------------

BUILD_OFFSET_MAP_CASES = [
    # identical raw/rendered -> linear map
    ("", "", 4),
    ("abc", "abc", 4),
    ("a\nb", "a\nb", 4),
    # tab expansion
    ("a\tb", "a   b", 4),
    ("\t\tx", "        x", 4),
    ("a\tb\tc", "a   b   c", 8),
    ("\t", "    ", 4),
    ("tab\there\tnow", "tab here   now", 4),
    # unicode (code points, not bytes)
    ("héllo\twörld", "héllo   wörld", 4),
    ("中文\t字", "中文   字", 4),
    ("emoji 🎉\tend", "emoji 🎉    end", 4),
    ("nfc \u00e9 vs \u0065\u0301", "nfc \u00e9 vs \u0065\u0301", 4),
    # fallback linear-map branch (col != len(rendered))
    ("x" * 10, "y" * 10, 4),
    ("\t\tx", " x", 4),
    ("", "rendered-only", 4),
    ("abc", "zzz-longer", 4),
    # mixed content
    ("def\tg", "def  g!", 4),
    ("one two\tthree", "one two three", 4),
]


@pytest.mark.parametrize("raw,rendered,tab_size", BUILD_OFFSET_MAP_CASES)
def test_build_offset_map_equivalence(raw, rendered, tab_size):
    from kimi_cli.utils.rich import diff_render as mod

    restore = _native_on(mod, True)
    try:
        native = mod._build_offset_map(raw, rendered, tab_size)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._build_offset_map(raw, rendered, tab_size)
    finally:
        restore()
    assert native == python, (
        f"_build_offset_map native != python for {(raw, rendered, tab_size)!r}:\n"
        f"  native={native!r}\n  python={python!r}"
    )


def test_build_offset_map_shape():
    """Result is a list of len(raw) + 1 offsets, native path included."""
    from kimi_cli.utils.rich import diff_render as mod

    restore = _native_on(mod, True)
    try:
        offsets = mod._build_offset_map("a\tb", "a   b", 4)
    finally:
        restore()
    assert isinstance(offsets, list)
    assert all(isinstance(i, int) for i in offsets)
    assert len(offsets) == 3 + 1
    assert offsets == [0, 1, 4, 5]


# ---------------------------------------------------------------------------
# GLOB kernel: _parse_ls_files_output (NUL-delimited git ls-files output)
# ---------------------------------------------------------------------------

PARSE_LS_FILES_CASES = [
    # empty input
    "",
    # simple paths
    "src/main.py\x00src/lib/util.py\x00",
    "a/b\x00",
    "dir/subdir/file.txt\x00",
    "a\x00b\x00\x00c",
    # ignored dirs / files
    "node_modules/a/b.js\x00src/x.py\x00",
    ".venv/lib/x.py\x00main.py\x00",
    "vendor/pkg/init.py\x00app.py\x00",
    "__pycache__/x.pyc\x00run.py\x00",
    "build/out.o\x00src/a.c\x00",
    "foo.bar.pyc\x00backup.py~\x00keep.py\x00",
    "x.cache/y.txt\x00src/y.txt\x00",
    "dist/\x00build/\x00src/a.py\x00",
    # nested ignored prefixes stay filtered
    "node_modules/a/b/c.js\x00",
    # non-ascii names
    "ディレクトリ/ファイル.py\x00",
    "café/naïve.txt\x00",
    "中文目录/文件.txt\x00",
    # trailing slashes / empty segments (native does NOT replicate the
    # empty-segment-as-ignored behavior -> must fall back to Python)
    "trailing/\x00",
    "a/b/\x00c/d/e\x00",
    "/\x00",
    "a//b\x00",
    "src//main.py\x00",
    "one/two/three/\x00",
    "top/\x00mid/\x00",
    # filter_ignored=False keeps everything
    ("src/main.py\x00node_modules/x.js\x00", False),
    ("trailing/\x00a/b\x00", False),
]


@pytest.mark.parametrize(
    "stdout,filter_ignored",
    [
        (s, True) if isinstance(s, str) else s for s in PARSE_LS_FILES_CASES
    ],
)
def test_parse_ls_files_output_equivalence(stdout, filter_ignored):
    from kimi_cli.utils import file_filter as mod

    restore = _native_on(mod, True)
    try:
        native = mod._parse_ls_files_output(stdout, filter_ignored=filter_ignored)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._parse_ls_files_output(stdout, filter_ignored=filter_ignored)
    finally:
        restore()
    assert native == python, (
        f"_parse_ls_files_output native != python for {stdout[:40]!r} "
        f"(filter_ignored={filter_ignored}):\n"
        f"  native={native!r}\n  python={python!r}"
    )


# ---------------------------------------------------------------------------
# TOOLS kernel: _cumulative_hashes + wired call sites
# ---------------------------------------------------------------------------

CUMULATIVE_HASH_CASES = [
    [],
    [""],
    ["", ""],
    ["a"],
    ["a", "b"],
    ["a", ""],
    ["a", "b", ""],
    ["", "abc"],
    ["abc", "  ", "def"],
    ["   ", "\t", ""],
    ["a\r", "b"],
    ["line1", "line2", "line3"],
    ["héllo", "世界", "🎉"],
    ["x" * 1000, "y" * 1000, "z" * 1000],
    ["", "a", "", "b", ""],
    ["\t", "x"],
    ["one"],
    ["a", "b", "c", "d", "e"],
]


@pytest.mark.parametrize("file_lines", CUMULATIVE_HASH_CASES)
def test_cumulative_hashes_equivalence(file_lines):
    from kimi_cli.tools.file import hash_line as mod

    restore = _native_on(mod, True)
    try:
        native = mod._cumulative_hashes(file_lines)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._cumulative_hashes(file_lines)
    finally:
        restore()
    assert native == python, (
        f"_cumulative_hashes native != python for {file_lines!r}:\n"
        f"  native={native!r}\n  python={python!r}"
    )


def _mismatch_error_str(file_lines, mismatch_lines, gate):
    from kimi_cli.tools.file.hash_line import (
        HashlineMismatchError,
        HashMismatch,
    )

    import kimi_cli.tools.file.hash_line as mod

    mismatches = [HashMismatch(line, "XX", "YY") for line in mismatch_lines]
    restore = _native_on(mod, gate)
    try:
        return str(HashlineMismatchError(mismatches, file_lines))
    finally:
        restore()


def test_hashline_mismatch_error_str_equivalence():
    """HashlineMismatchError.__str__ (cumulative hashes of file_lines)."""
    cases = [
        (["a", "b", "c", "d"], [2]),
        (["line 1", "line 2", "line 3", "line 4", "line 5"], [1, 5]),
        (["x", "y", ""], [1]),
        ([""], [1]),
        (["only"], [1]),
        (["héllo", "世界", "🎉", "tail"], [2, 3]),
    ]
    for file_lines, mismatch_lines in cases:
        native = _mismatch_error_str(file_lines, mismatch_lines, True)
        python = _mismatch_error_str(file_lines, mismatch_lines, False)
        assert native == python, (
            f"HashlineMismatchError str mismatch for {file_lines!r}: "
            f"\n  native={native!r}\n  python={python!r}"
        )


def _validate(file_lines, line, hash_str, gate):
    from kimi_cli.tools.file.hash_line import AnchorRef, validate_anchor_ref

    import kimi_cli.tools.file.hash_line as mod

    mismatches = []
    errors = []
    restore = _native_on(mod, gate)
    try:
        validate_anchor_ref(
            AnchorRef(line=line, hash=hash_str),
            file_lines,
            mismatches,
            errors,
            cumulative_hashes=None,  # force the wired compute branch
        )
    finally:
        restore()
    return (len(mismatches), [m.line for m in mismatches], list(errors))


def test_validate_anchor_ref_none_branch_equivalence():
    """validate_anchor_ref with cumulative_hashes=None (wired compute branch)."""
    from kimi_cli.tools.file import hash_line as mod

    def hashes(file_lines):
        restore = _native_on(mod, False)
        try:
            return mod._cumulative_hashes(file_lines)
        finally:
            restore()

    cases = [
        (["a", "b", "c"], 2),
        (["a", "b", ""], 1),
        (["a", "b", ""], 3),
        ([""], 1),
        (["line 1", "line 2", "line 3", "line 4"], 4),
        (["héllo", "世界", "🎉"], 2),
    ]
    for file_lines, line in cases:
        h = hashes(file_lines)[line - 1]
        good = _validate(file_lines, line, h, True) == _validate(file_lines, line, h, False)
        bad_line = _validate(file_lines, line, "ZZ", True) == _validate(
            file_lines, line, "ZZ", False
        )
        out_of_range = _validate(file_lines, len(file_lines) + 5, "ZZ", True) == _validate(
            file_lines, len(file_lines) + 5, "ZZ", False
        )
        assert good and bad_line and out_of_range, (
            f"validate_anchor_ref mismatch for {file_lines!r} line {line}"
        )


def test_apply_hashline_edits_equivalence_gate_on_off():
    """End-to-end: apply_hashline_edits output identical with gate on vs off."""
    from kimi_cli.tools.file.hash_line import (
        AnchorRef,
        AppendEdit,
        ReplaceEdit,
        apply_hashline_edits,
    )

    import kimi_cli.tools.file.hash_line as mod

    def hashes(file_lines):
        restore = _native_on(mod, False)
        try:
            return mod._cumulative_hashes(file_lines)
        finally:
            restore()

    content = "line 1\nline 2\nline 3\nline 4\n"
    lines = content.splitlines()
    edits = [
        ReplaceEdit(
            op="replace",
            pos=AnchorRef(line=2, hash=hashes(lines)[1]),
            end=AnchorRef(line=3, hash=hashes(lines)[2]),
            lines=["REPLACED"],
        ),
        AppendEdit(op="append", pos=None, lines=["at eof"]),
    ]
    for gate in (True, False):
        restore = _native_on(mod, gate)
        try:
            result, first_changed = apply_hashline_edits(content, edits)
        finally:
            restore()
        if gate:
            native_result = result
            native_first = first_changed
        else:
            python_result = result
            python_first = first_changed
    assert native_result == python_result
    assert native_first == python_first
