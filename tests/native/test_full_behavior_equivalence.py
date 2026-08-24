"""Comprehensive behavior-equivalence gate for kimix_native kernels.

Every public function in every kernel submodule of ``bin/kimix_native`` is run
twice — once with the compiled extension forced on (``KIMIX_NATIVE_<KERNEL>=1``)
and once with the pure-Python fallback forced off (``...=0``) — and the results
(including exception type/args) must be identical.

Coverage:
  * stateless functions (all 9 kernel modules)
  * stateful classes exercised through full scenarios
  * edge cases: empty input, ASCII, CJK/mixed Unicode, control chars,
    surrogates, malformed input, boundary sizes
  * deterministic fuzz cases for the numerically-heavy kernels
  * exception parity (same exception type raised by both paths)
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BIN = _REPO_ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_BIN / "runtime_py.pyd"),
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)

@pytest.fixture(autouse=True)
def _restore_kimix_env():
    """Restore KIMIX_NATIVE* env vars after every test so subprocess-based
    tests elsewhere in the suite never observe our toggles."""
    saved = {k: os.environ[k] for k in os.environ if k.startswith("KIMIX_NATIVE")}
    yield
    for k in list(os.environ):
        if k.startswith("KIMIX_NATIVE") and k not in saved:
            del os.environ[k]
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# module/state plumbing
# ---------------------------------------------------------------------------

_KERNEL_FOR_MODULE = {
    "text": "TEXT",
    "stream": "STREAM",
    "codec": "CODEC",
    "diff": "DIFF",
    "glob": "GLOB",
    "index": "INDEX",
    "search": "SEARCH",
    "parse": "PARSE",
    "tools": "TOOLS",
}

# Modules whose dispatch decision is computed at import time (`_USE = ...`),
# so the module must be reloaded after toggling the env var.
_IMPORT_TIME_MODULES = {"codec"}


def _load_module(name: str, state: bool):
    """Return the kimix_native.<name> module with native toggled to *state*."""
    kernel = _KERNEL_FOR_MODULE[name]
    os.environ[f"KIMIX_NATIVE_{kernel}"] = "1" if state else "0"
    mod = importlib.import_module(f"kimix_native.{name}")
    if name in _IMPORT_TIME_MODULES:
        mod = importlib.reload(mod)
    return mod


def _norm_key(k):
    if isinstance(k, (str, int, float, bool)) or k is None:
        return k
    return repr(k)


def _norm(v):
    """Normalize a value (bytes/tuples/dataclasses/floats/sets) to JSON-safe
    structures so native and python results compare with plain ==."""
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, bytearray):
        return {"$bytes": bytes(v).hex()}
    if isinstance(v, float):
        if math.isnan(v):
            return {"$nan": True}
        if math.isinf(v):
            return {"$inf": 1 if v > 0 else -1}
        return v
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, set):
        return sorted((_norm(x) for x in v), key=repr)
    if isinstance(v, dict):
        return {_norm_key(k): _norm(val) for k, val in v.items()}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _norm(dataclasses.asdict(v))
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)


def _run_case(mod_name: str, func_name: str, state: bool, args, kwargs):
    """Run one stateless function case and capture result-or-exception."""
    mod = _load_module(mod_name, state)
    fn = getattr(mod, func_name)
    try:
        return {"ok": _norm(fn(*args, **kwargs))}
    except Exception as exc:  # noqa: BLE001 - parity capture
        return {"exc": type(exc).__name__, "args": _norm(list(exc.args))}


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------

_TEXT_CORPUS = [
    "",
    "plain ascii text",
    "a" * 500,
    "中文测试文本内容",
    "mixed 中文 with ascii words",
    "emoji 🎉🎉 and 中文",
    "é" * 200,
    " " * 100,
    "\x00\x01\x1b[31m control chars",
    "line1\nline2\tline3\r",
    "\ud800 lone surrogate",
    "x" * 10000,
]

BEHAVIOR_CASES: dict[str, list[tuple[str, tuple, dict]]] = {
    "text": [
        ("estimate_chars_tokens", (t,), {}) for t in _TEXT_CORPUS
    ]
    + [
        ("count_tokens", (t,), {}) for t in _TEXT_CORPUS
    ]
    + [
        ("count_tokens", ("hello model",), {"model": None}),
        ("is_cjk_text", ("中文文本",), {}),
        ("is_cjk_text", ("ascii only",), {}),
        ("is_cjk_text", ("",), {}),
        ("is_cjk_text", ("half中文",), {"threshold": 0.3}),
        ("is_cjk_text", ("many 中文 中文 chars",), {"threshold": 0.1}),
        ("clean_text", ("",), {}),
        ("clean_text", ("zero\u200bwidth\u200dchars",), {}),
        ("clean_text", ("\x00\x01\x02\x1f ctl",), {}),
        ("clean_text", ("line1\nline2\r\n",), {}),
        ("clean_text", ("line1\nline2\r\n",), {"keep_newlines": False}),
        ("clean_text", ("  padded  ",), {}),
        ("clean_text", ("emoji 🎉 中文",), {}),
        ("clean_text", (123,), {}),
        ("sanitize_for_tokenizer", ("",), {}),
        ("sanitize_for_tokenizer", ("hello world",), {}),
        ("sanitize_for_tokenizer", ("\ufffd replacement",), {}),
        ("sanitize_for_tokenizer", ("pua \ue000 private",), {}),
        ("sanitize_for_tokenizer", ("repeat " * 40,), {}),
        ("sanitize_for_tokenizer", ("abc",), {"max_chars": 2}),
        ("sanitize_for_tokenizer", ("abcdef",), {"max_chars": 4, "truncate_msg": "..."}),
        ("sanitize_for_tokenizer", ("ctrl \x07\x08\x0b chars",), {}),
        ("sanitize_for_tokenizer", ("中文文本",), {}),
        ("sanitize_for_tokenizer", ("emoji 🎉 text",), {}),
        ("sanitize_for_tokenizer", ("surrogate \ud800 here",), {}),
    ],
    "stream": [
        ("filter_output", (t,), {})
        for t in [
            "",
            "plain",
            "a" * 4096,
            "line1\nline2\r\nline3\r",
            "\x1b[31mred\x1b[0m text",
            "\x1b]0;title\x07OSC title\x1b\\ end",
            "repeat\nrepeat\nrepeat\nunique",
            "emoji 🎉\r\n中文",
            "tab\tand\x00nul",
            "mixed \x1b[1mbold\x1b[0m 中文",
        ]
    ]
    + [
        ("strip_ansi", (t,), {})
        for t in [
            "",
            "no escapes",
            "\x1b[31mred\x1b[0m",
            "\x1b]0;title\x07OSC",
            "a\x1b[1mb\x1b[0mc",
            "\x1b[38;2;255;0;0mtruecolor\x1b[0m",
            "中文\x1b[1m加粗\x1b[0m",
        ]
    ]
    + [
        ("strip_ansi", (b"\x1b[31mred\x1b[0m",), {}),
        ("strip_ansi", (b"plain bytes",), {}),
        ("strip_ansi", (bytearray(b"\x1b[0m x"),), {}),
    ],
    "codec": [
        ("serialize_envelope", ("text", b'{"a": 1}'), {}),
        ("serialize_envelope", ("args", b'[1,2,3]'), {}),
        ("serialize_envelope", ("text", b"not json at all"), {}),
        ("serialize_envelope", ("", b""), {}),
        ("serialize_envelope", ("text", b'{"k": "v"\n}'), {}),
        ("serialize_envelope", ("sse", b'{"event":"x","data":1}'), {}),
        ("serialize_envelope", ("中文", '{"中": 1}'.encode()), {}),
        ("deserialize_envelope", (b'{"type":"text","payload":{"a":1}}',), {}),
        ("deserialize_envelope", (b'{"type":"args","payload":[1,2]}',), {}),
        ("deserialize_envelope", (b"not json",), {}),
        ("deserialize_envelope", (b"{}",), {}),
        ("deserialize_envelope", (b'{"type":1,"payload":1}',), {}),
        ("deserialize_envelope", (b'{"payload":1}',), {}),
        ("deserialize_envelope", (b'{"type":"text"}',), {}),
        ("deserialize_envelope", ('{"type":"text","payload":"中文"}'.encode(),), {}),
        ("canonicalize_payload", (b'{"b":1,"a":2}',), {}),
        ("canonicalize_payload", (b"[3,1,2]",), {}),
        ("canonicalize_payload", (b'{"x":{"d":4,"c":3}}',), {}),
        ("canonicalize_payload", (b"bad json",), {}),
        ("canonicalize_payload", (b'{"a":"x","b":[{"y":1,"x":2}]}',), {}),
        ("canonicalize_payload", (b'{"z":null,"a":true,"m":[1,{"q":2,"p":1}]}',), {}),
        ("build_sse_frame", ("", b"{}", 0), {}),
        ("build_sse_frame", ("event", b'{"a":1}', 0), {}),
        ("build_sse_frame", ("ev", b"line1\nline2", 5), {}),
        ("build_sse_frame", ("名字", '{"中":1}'.encode(), 0), {}),
        ("build_sse_frame", ("e", b"", 7), {}),
    ],
    "diff": [
        ("unified_diff", (b"a\nb\nc\n", b"a\nB\nc\n"), {}),
        ("unified_diff", (b"", b"x\ny\n"), {}),
        ("unified_diff", (b"x\ny\n", b""), {}),
        ("unified_diff", (b"same\n", b"same\n"), {}),
        ("unified_diff", (b"l1\nl2\nl3\nl4\nl5\n", b"l1\nl2\nl3\nl4\nl5\n"), {}),
        ("unified_diff", (b"a\nb\nc\n", b"a\nb\nc\nd\n"), {"path": "f.txt"}),
        ("unified_diff", (b"a\nb\n", b"a\nb\n"), {"include_file_header": False}),
        ("unified_diff", (b"a\r\nb\r\n", b"a\r\nc\r\n"), {}),
        ("unified_diff", ("中文行1\n中文行2\n".encode(), "中文行1\n改行2\n".encode()), {}),
        ("diff_hunks", (b"a\nb\nc\n", b"a\nB\nc\n"), {}),
        ("diff_hunks", (b"1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n", b"1\n2\n3\n4\nX\n6\n7\n8\n9\n10\n"), {}),
        ("diff_hunks", (b"", b"new\n"), {}),
        ("diff_hunks", (b"x\n", b""), {}),
        ("diff_hunks", (b"a\nb\n", b"a\nb\n"), {"context_lines": 0}),
        ("diff_hunks", ("中文\n".encode(), "中文改\n".encode()), {}),
        ("inline_diff_ranges", ("hello world", "hello brave world"), {}),
        ("inline_diff_ranges", ("abc", "abc"), {}),
        ("inline_diff_ranges", ("completely", "different"), {}),
        ("inline_diff_ranges", ("", ""), {}),
        ("inline_diff_ranges", ("a\tb", "a\tx\tb"), {}),
        ("inline_diff_ranges", ("中文一", "中文二"), {}),
        ("inline_diff_ranges", ("abc", "axc"), {"min_ratio": 0.9}),
        ("build_offset_map", ("\tcol", "\tcol"), {}),
        ("build_offset_map", ("a\tb", "a\tb"), {}),
        ("build_offset_map", ("abc", "abc"), {}),
        ("build_offset_map", ("\t", "    "), {}),
        ("build_offset_map", ("中文", "中文"), {}),
        ("build_offset_map", ("a\tb", "a b"), {"tab_size": 2}),
    ],
    "glob": [
        ("parse_gitignore", (b"*.pyc\nbuild/\n# comment\n!keep.pyc\n/top.txt\n", "src"), {}),
        ("parse_gitignore", (b"", "src"), {}),
        ("parse_gitignore", (b"node_modules/\ndist*\n*.log\n", ""), {}),
        ("parse_gitignore", (b"\xef\xbb\xbf*.py\n", "src"), {}),
        ("is_ignored_name", ("node_modules",), {}),
        ("is_ignored_name", (".git",), {}),
        ("is_ignored_name", ("main.py",), {}),
        ("is_ignored_name", ("__pycache__",), {}),
        ("parse_ls_files_output", (b"a.py\nb/\nc.py\n",), {}),
        ("parse_ls_files_output", (b"",), {}),
        ("parse_ls_files_output", (b"x.py\n", False), {"filter_ignored": False}),
        ("parse_ls_files_output", ("中文文件.py\n".encode(),), {}),
    ],
    "search": [
        ("damerau_levenshtein", ("", ""), {}),
        ("damerau_levenshtein", ("abc", ""), {}),
        ("damerau_levenshtein", ("abc", "abc"), {}),
        ("damerau_levenshtein", ("abc", "abd"), {}),
        ("damerau_levenshtein", ("kitten", "sitting"), {}),
        ("damerau_levenshtein", ("flaw", "lawn"), {}),
        ("damerau_levenshtein", ("ab", "ba"), {}),
        ("damerau_levenshtein", ("中文", "中文"), {}),
        ("damerau_levenshtein", ("中文", "英文"), {}),
        ("damerau_levenshtein", ("x" * 40, "y" * 40), {}),
        ("damerau_levenshtein", ("abcde", "edcba"), {"max_dist": 4}),
        ("damerau_levenshtein", ("abcdef", "abcdef"), {"max_dist": 2}),
        ("jaro_similarity", ("", ""), {}),
        ("jaro_similarity", ("abc", "abc"), {}),
        ("jaro_similarity", ("abc", "abd"), {}),
        ("jaro_similarity", ("martha", "marhta"), {}),
        ("jaro_similarity", ("中文", "中文"), {}),
        ("jaro_similarity", ("abc", "xyz"), {}),
        ("jaro_winkler", ("martha", "marhta"), {}),
        ("jaro_winkler", ("abc", "abd"), {}),
        ("jaro_winkler", ("dwayne", "duane"), {}),
        ("jaro_winkler", ("abc", "abc"), {"prefix_scale": 0.2}),
        ("sorensen_dice", ("night", "nacht"), {}),
        ("sorensen_dice", ("", ""), {}),
        ("sorensen_dice", ("same", "same"), {}),
        ("sorensen_dice", ("中文一", "中文二"), {}),
        ("ngram_overlap", ("", ""), {}),
        ("ngram_overlap", ("abc", "abc"), {}),
        ("ngram_overlap", ("abcdef", "abcxyz"), {}),
        ("ngram_overlap", ("abc", "abc"), {"n": 1}),
        ("ngram_overlap", ("abc", "abc"), {"n": 3}),
        ("ngram_overlap", ("中文测试", "中文内容"), {}),
        ("freq_lower_bound", ("abc", "abcabc"), {}),
        ("freq_lower_bound", ("", "abc"), {}),
        ("freq_lower_bound", ("a", "a"), {}),
        ("freq_lower_bound", ("ab", "a"), {}),
        ("bm25_idf", (10, 2), {}),
        ("bm25_idf", (10, 2, 1.5, 0.5), {"k1": 1.5, "b": 0.5}),
        ("bm25_idf", (1, 1), {}),
        ("bm25_topk", ([0.5, 0.1, 0.9, 0.3], 2), {}),
        ("bm25_topk", ([], 3), {}),
        ("bm25_topk", ([1.0, 2.0], 0), {}),
        ("bm25_topk", ([1.0, 2.0], 5), {}),
        ("bm25_score", (
            [[(0, 1)], [(1, 1), (2, 1)]], [1.5, 1.2],
            [10, 20, 30], 20.0, 3,
        ), {}),
        ("bm25_score", (
            [], [], [10, 20], 15.0, 2,
        ), {}),
        ("simhash", ([],), {}),
        ("simhash", (["hello", "world", "hello"],), {}),
        ("simhash", (["中文", "内容"],), {}),
        ("simhash", (["a"] * 50,), {"seed": 42}),
        ("minhash", (["a", "b", "c"], 2), {}),
        ("minhash", ([], 3), {}),
        ("minhash", (["x"] * 100, 5, 7), {"seed": 7}),
        ("minhash", (["中文", "内容"], 4), {}),
        ("mmr_rerank", ([0.9, 0.8, 0.7], [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]), {}),
        ("mmr_rerank", ([0.9, 0.8, 0.7], [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]], 0.7, 2), {"lambda_param": 0.7, "k": 2}),
        ("mmr_rerank", ([], []), {}),
        ("xquad_rerank", ([0.5, 0.1, 0.9], 0), {}),
        ("xquad_rerank", ([0.5, 0.1, 0.9], 2), {"k": 2}),
        ("xquad_rerank", ([], 0), {}),
    ],
    "parse": [
        ("comment_spans", ("c", b"// hi\nint x;\n"), {}),
        ("comment_spans", ("c", b"/* block */ int x;\n"), {}),
        ("comment_spans", ("c", b'char *s = "not // c";\n// real\n'), {}),
        ("comment_spans", ("python", b"# comment\nx = 1\n"), {}),
        ("comment_spans", ("python", b'"""docstring"""\nx=1\n'), {}),
        ("comment_spans", ("shell", b"# comment\necho hi\n"), {}),
        ("comment_spans", ("sql", b"-- line\nSELECT 1;\n"), {}),
        ("comment_spans", ("html", b"<!-- c --><p>x</p>\n"), {}),
        ("comment_spans", ("lisp", b"; comment\n(defun f (x) x)\n"), {}),
        ("comment_spans", ("pascal", b"{ comment }\nprogram p;\n"), {}),
        ("comment_spans", ("c", "中文注释\n// hi\n".encode()), {}),
        ("comment_spans", ("python", "中文#注释\nx=1\n".encode()), {}),
        ("comment_spans", ("c", b""), {}),
        ("parse", ("c", "// hi\nint x;\n"), {}),
        ("parse", ("c", "/* A\nblock */\nint x;\n"), {}),
        ("parse", ("python", "# comment\nx = 1\n"), {}),
        ("parse", ("python", 'x = "not #"\n# yes\n'), {}),
        ("parse", ("shell", "echo '# not comment' # yes\n"), {}),
        ("parse", ("sql", "-- c\nSELECT 1;\n"), {}),
        ("parse", ("html", "<!-- c --><p>x</p>\n"), {}),
        ("parse", ("lisp", "; c\n(defun f (x) x)\n"), {}),
        ("parse", ("pascal", "{ c }\nprogram p;\n"), {}),
        ("parse", ("python", "中文注释\nx=1\n"), {}),
        ("parse", ("c", ""), {}),
        ("fix_bash_command", ("echo hello",), {}),
        ("fix_bash_command", ("rev file.txt",), {}),
        ("fix_bash_command", ("tree .",), {}),
        ("fix_bash_command", ("wget http://example.com/x",), {}),
        ("fix_bash_command", ('grep -r "rev" .',), {}),
        ("fix_bash_command", ("echo a; rev b; echo c",), {}),
        ("fix_bash_command", ("ls -la C:\\Users\\me\\file.txt",), {}),
        ("fix_bash_command", ("cd 'C:\\Program Files'",), {}),
        ("fix_bash_command", ("cat <<EOF\nrev\nEOF\nrev",), {}),
        ("fix_bash_command", ("echo 中文",), {}),
        ("fix_bash_command", ("cat /tmp/x.txt",), {}),
        ("fix_bash_command", ("echo /c/dev && echo /tmp/y",), {}),
        ("fix_bash_command", ("env --chdir=/tmp cmd",), {}),
        ("fix_bash_command", ("cd /d/foo",), {}),
        ("fix_bash_command", ("echo '/tmp/x'",), {}),
        ("fix_pwsh_command", ('Write-Output "hello"',), {}),
        ("fix_pwsh_command", ("Get-ChildItem C:\\Users",), {}),
        ("fix_pwsh_command", ('Write-Output \'He said "hi"\'',), {}),
        ("fix_pwsh_command", ('Write-Output "a`"b"',), {}),
        ("fix_pwsh_command", ("",), {}),
        ("fix_pwsh_command", ("   ",), {}),
        ("fix_pwsh_command", ('Write-Output "中文"',), {}),
        ("pwsh_transform", ('Write-Output "hello"',), {}),
        ("pwsh_transform", ("Get-Process | Where-Object { $_.CPU -gt 10 }",), {}),
        ("pwsh_transform", ("$x = @\"\nline\n\"@\n$x",), {}),
        ("pwsh_transform", ('Write-Output "中文"',), {}),
        ("pwsh_transform", ("",), {}),
    ],
    "tools": [
        ("line_hash", ("hello",), {}),
        ("line_hash", (b"hello\r\n",), {}),
        ("line_hash", ("",), {}),
        ("line_hash", ("  spaced  ",), {}),
        ("line_hash", ("中文行",), {}),
        ("line_hash", ("x" * 100, 42), {"seed": 42}),
        ("line_hashes", ("a\nb\nc\n",), {}),
        ("line_hashes", (b"x\r\ny\r\n",), {}),
        ("line_hashes", ("",), {}),
        ("line_hashes", ("line1\nline2\nline3", 7), {"seed": 7}),
        ("line_hashes", ("中文\n第二行\n",), {}),
        ("compute_line_hashes", ("a\nb\n",), {}),
        ("compute_line_hashes", ("",), {}),
        ("compute_line_hashes", ("中文\n内容\n",), {}),
        ("find_in_file", ("hello world\nhello again\n", "hello", False, "f.txt"), {"path": "f.txt"}),
        ("find_in_file", ("case\nSENSITIVE\ncase\n", "case"), {}),
        ("find_in_file", ("a\nab\nabc\n", "ab"), {}),
        ("find_in_file", ("中文\n中文内容\n", "中文"), {}),
        ("find_in_file", ("x" * 1000 + "\nneedle\n" + "y" * 1000, "needle"), {}),
        ("find_in_file", ("no match here", "zzz"), {}),
        ("scan_lines", ("line1\nline2\nline3\n", "line", True), {}),
        ("scan_lines", (b"a\nb\nA\n", b"a", True), {}),
        ("scan_lines", (b"a\nb\nA\n", b"a", False), {"case_insensitive": False}),
        ("scan_lines", ("中文\n内容\n", "中文"), {}),
        ("scan_lines", ("", ""), {}),
        ("redact_sensitive_output", ("password=secret123",), {}),
        ("redact_sensitive_output", ("token: abcdefghijklmnop",), {}),
        ("redact_sensitive_output", ("",), {}),
        ("redact_sensitive_output", ("key=12345 api_key=67890",), {}),
        ("redact_sensitive_output", ("中文密钥 abc123",), {}),
        ("scrub_child_env", ({"PATH": "/bin", "API_KEY": "sekret"},), {}),
        ("scrub_child_env", ({"SAFE": "1", "TOKEN": "x", "AUTH": "y"},), {}),
        ("scrub_child_env", ({},), {}),
        ("scrub_child_env", ({"中文键": "v"},), {}),
        ("validate_workdir", (None,), {}),
        ("validate_workdir", ("C:\\Users\\me",), {}),
        ("validate_workdir", ("/tmp",), {}),
        ("validate_workdir", ("",), {}),
        ("bounded_append", ("", "abc", 10), {}),
        ("bounded_append", ("aaaa", "bbbb", 6), {}),
        ("bounded_append", ("x" * 100, "y", 50), {}),
        ("bounded_append", ("中文", "内容", 3), {}),
        ("command_detection_variants", ("ls",), {}),
        ("command_detection_variants", ("",), {}),
        ("command_detection_variants", ("rm -rf /",), {}),
        ("detect_hardline_command", ("ls",), {}),
        ("detect_hardline_command", ("rm -rf /",), {}),
        ("check_hardline_blocked", ("ls",), {}),
        ("check_hardline_blocked", ("rm -rf /",), {}),
        ("foreground_background_guidance", ("ls",), {}),
        ("foreground_background_guidance", ("ping 8.8.8.8",), {}),
        ("base_command_name", ("ls -la",), {}),
        ("base_command_name", ("C:\\Program Files\\app.exe --x",), {}),
        ("base_command_name", ("",), {}),
        ("interpret_exit_code", ("ls", 0), {}),
        ("interpret_exit_code", ("ls", 2), {}),
        ("interpret_exit_code", ("git merge", 1), {}),
        ("annotate_failure", ("command not found: foo", "foo", 127), {}),
        ("annotate_failure", ("ok output", "ls", 0), {}),
        ("pattern_has_regex_newline", ("a",), {}),
        ("pattern_has_regex_newline", ("a\\n",), {}),
        ("multiline_pattern", ("a",), {}),
        ("multiline_pattern", ("a\\nb",), {}),
        ("multiline_pattern", ("a\nb",), {}),
    ],
}

# ---------------------------------------------------------------------------
# stateful scenarios
# ---------------------------------------------------------------------------
_HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}, {"type": "think", "think": "hmm"}],
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1",
     "content": [{"type": "tool_result", "content": "result", "tool_name": "read_file"}]},
    {"role": "user", "content": [{"type": "text", "text": "中文内容"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "回复"}]},
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
]

_TOOL_CALL_DUP = [
    {"role": "user", "content": [{"type": "text", "text": "go"}]},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "using"}],
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_1", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1",
     "content": [{"type": "tool_result", "content": "r1", "tool_name": "a"}]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": [{"type": "tool_result", "content": "r2", "tool_name": "b"}]},
]


def _scenario_line_processor(mod, params, chunks):
    lp = mod.LineProcessor(**params)
    out = []
    for c in chunks:
        out.append(lp.feed(c))
    out.append(lp.flush())
    return {
        "out": out,
        "bytes": lp.bytes_written(),
        "cps": lp.code_points_written(),
        "lines": lp.lines_written(),
    }


def _scenario_wire_merge(mod):
    b = mod.WireMergeBuffer()
    res = []
    res.append(b.append("text", b"a"))
    res.append(b.append("text", b"b"))
    res.append(b.snapshot())
    res.append(b.empty())
    res.append(b.append("args", b"c"))
    res.append(b.snapshot())
    b.reset()
    res.append(b.empty())
    res.append(b.append("args", b"x"))
    res.append(b.snapshot())
    return res


def _scenario_args_buffer(mod):
    b = mod.ArgsBuffer()
    res = []
    b.append(b"abc")
    res.append(b.snapshot())
    res.append(b.delta_since())
    b.append(b"def")
    res.append(b.delta_since())
    res.append(b.delta_since())
    b.reset()
    res.append(b.snapshot())
    return res


def _scenario_recv_buffer(mod):
    b = mod.RecvBuffer()
    res = []
    b.append((3).to_bytes(4, "big") + b"abc")
    res.append(b.take_frame_length_prefixed())
    res.append(b.size())
    b.append((2).to_bytes(4, "big") + b"xy" + (1).to_bytes(4, "big") + b"z")
    res.append(b.take_frame_length_prefixed())
    res.append(b.take_frame_length_prefixed())
    res.append(b.take_frame_length_prefixed())
    b.append(b"hello\nworld\n")
    res.append(b.take_frame_delimiter(b"\n"))
    res.append(b.take_frame_delimiter(b"\n"))
    res.append(b.take_frame_delimiter(b"\n"))
    b.append(b"part1")
    res.append(b.take_frame_delimiter(b"|", 100))
    b.append(b"|part2|")
    res.append(b.take_frame_delimiter(b"|", 100))
    res.append(b.take_frame_delimiter(b"|", 100))
    b.clear()
    res.append(b.size())
    return res


def _scenario_ngram(mod):
    t = mod.NgramTokenizer(default_n=2)
    res = []
    for text in ["hello world", "中文内容", "The quick brown fox", "a", "", "  spaced  "]:
        res.append(t.normalize(text))
        res.append(t.detect_n(text))
        res.append(t.tokenize(text))
    res.append(t.tokenize("hello world", 1))
    res.append(t.tokenize("hello world", 3))
    return res


def _scenario_inverted_index(mod):
    idx = mod.InvertedIndex()
    res = []
    idx.add_document(0, ["hello", "world"])
    idx.add_document(1, ["hello"])
    idx.add_document(2, ["foo", "bar", "baz"])
    res.append(idx.doc_count())
    res.append(idx.max_doc_id())
    res.append(idx.sum_doc_lengths())
    res.append(idx.avg_doc_len())
    res.append(idx.total_postings())
    res.append(idx.finalized())
    idx.finalize()
    res.append(idx.finalized())
    res.append(idx.get_postings("hello"))
    res.append(idx.get_postings("missing"))
    res.append(idx.has_term("world"))
    res.append(idx.has_term("nope"))
    res.append(idx.doc_length(0))
    res.append(idx.doc_length(9))
    res.append(idx.segment_count())
    blob = idx.save()
    res.append(blob[:6])
    idx2 = mod.InvertedIndex()
    res.append(idx2.load(blob))
    res.append(idx2.get_postings("hello"))
    res.append(idx2.has_term("foo"))
    res.append(idx.load(b"garbage"))
    idx.reset()
    res.append(idx.doc_count())
    return res


def _scenario_history_index(mod):
    hi = mod.HistoryIndex()
    res = []
    turns = [
        (1, 1000.0, "user", False, "hello world"),
        (2, 1001.0, "assistant", False, "hi there"),
        (3, 1002.0, "tool", True, "result data"),
        (4, 1003.0, "user", False, "中文检索测试"),
        (5, 1004.0, "user", False, "   "),
        (6, 1005.0, 3, False, "other role"),
    ]
    hi.append_turns(turns)
    res.append(hi.turn_count())
    res.append(hi.get_by_id(1))
    res.append(hi.get_by_id(2))
    res.append(hi.get_by_id("prune_3"))
    res.append(hi.get_by_id("bogus"))
    res.append(hi.search("hello", 2))
    res.append(hi.search("中文", 1))
    res.append(hi.search("zzz", 2))
    blob = hi.save()
    res.append(blob[:6])
    hi2 = mod.HistoryIndex()
    res.append(hi2.load(blob))
    res.append(hi2.turn_count())
    res.append(hi2.get_by_id(4))
    res.append(hi2.search("result", 1))
    hi.mark_compacted()
    res.append(hi.get_by_id(2))
    hi.pop_front()
    res.append(hi.turn_count())
    hi.reset()
    res.append(hi.turn_count())
    return res


def _scenario_symmetric_delete(mod):
    idx = mod.SymmetricDeleteIndex()
    res = []
    idx.add_term("hello", 2)
    idx.add_term("help", 2)
    idx.add_term("中文", 1)
    res.append(idx.term_count())
    res.append(idx.has_term("hello"))
    res.append(idx.has_term("nope"))
    res.append(idx.expand("helo", 2))
    res.append(idx.expand("hello", 1))
    res.append(idx.expand("hel", 1, 5))
    res.append(idx.expand("中文", 1))
    idx.reset()
    res.append(idx.term_count())
    return res


def _scenario_export_markdown(mod, history, opts):
    return mod.build_export_markdown(history, opts)


SCENARIOS: dict[str, dict[str, object]] = {
    "stream": {
        "line_processor_default": lambda m: _scenario_line_processor(
            m, {}, ["line1\n", "line2", "\n", "line3\n"]),
        "line_processor_bytes": lambda m: _scenario_line_processor(
            m, {}, [b"line1\n", b"line2\n"]),
        "line_processor_dedup_counter": lambda m: _scenario_line_processor(
            m, {"dedup_mode": 1, "threshold": 2}, ["x\nx\nx\ny\nx\nx\n"]),
        "line_processor_dedup_block": lambda m: _scenario_line_processor(
            m, {"dedup_mode": 2, "threshold": 2, "block_window": 2},
            ["b1\nb2\nb1\nb2\nb1\nb2\n"]),
        "line_processor_max_lines": lambda m: _scenario_line_processor(
            m, {"max_lines": 2}, ["a\nb\nc\nd\n"]),
        "line_processor_max_bytes": lambda m: _scenario_line_processor(
            m, {"max_bytes": 10}, ["0123456789\nmore\n"]),
        "line_processor_fold": lambda m: _scenario_line_processor(
            m, {"fold_col": 4}, ["abcdefgh\n"]),
        "line_processor_ansi": lambda m: _scenario_line_processor(
            m, {"strip_ansi": True}, ["\x1b[31mred\x1b[0m\n", "plain\n"]),
        "line_processor_mixed": lambda m: _scenario_line_processor(
            m, {}, ["中文\r\n", "emoji 🎉\n", b"bytes\n"]),
    },
    "codec": {
        "wire_merge_buffer": _scenario_wire_merge,
        "args_buffer": _scenario_args_buffer,
        "recv_buffer": _scenario_recv_buffer,
        "jsonrpc_frame_writer": lambda m: [
            m.JsonRpcFrameWriter().write(b'{"jsonrpc":"2.0","id":1}'),
            m.JsonRpcFrameWriter().write(b""),
        ],
        "jsonl_recorder": lambda m: [
            m.JsonlRecorder().record(b'{"type":"StepBegin"}'),
            m.JsonlRecorder().record(b""),
        ],
    },
    "index": {
        "ngram_tokenizer": _scenario_ngram,
        "inverted_index": _scenario_inverted_index,
        "history_index": _scenario_history_index,
    },
    "search": {
        "symmetric_delete_index": _scenario_symmetric_delete,
    },
    "tools": {
        "export_markdown": lambda m: _scenario_export_markdown(
            m, _HISTORY, {"max_chars": 80}),
        "export_markdown_dup": lambda m: _scenario_export_markdown(
            m, _TOOL_CALL_DUP, {"max_chars": 40}),
        "export_markdown_empty": lambda m: _scenario_export_markdown(m, [], {}),
    },
}

# ---------------------------------------------------------------------------
# deterministic fuzz cases for numerically-heavy kernels
# ---------------------------------------------------------------------------

_FUZZ_ALPHABETS = [
    "abc",
    "abcdefghijklmnopqrstuvwxyz",
    "中文ab",
    "aeiou",
    "abc \t\n",
]


def _fuzz_search_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    for _ in range(60):
        alpha = rng.choice(_FUZZ_ALPHABETS)
        a = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 20)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 20)))
        cases.append(("damerau_levenshtein", (a, b), {}))
        cases.append(("jaro_similarity", (a, b), {}))
        cases.append(("jaro_winkler", (a, b), {}))
        cases.append(("sorensen_dice", (a, b), {}))
        cases.append(("ngram_overlap", (a, b), {}))
        cases.append(("freq_lower_bound", (a[:5], b), {}))
    for _ in range(20):
        n = rng.randint(1, 50)
        tokens = [rng.choice(["alpha", "beta", "gamma", "中文", "x"]) for _ in range(n)]
        cases.append(("simhash", (tokens,), {}))
        cases.append(("minhash", (tokens, rng.randint(1, 8)), {}))
    return cases


def _fuzz_text_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    chunks = [
        "a" * rng.randint(0, 300),
        "中文" * rng.randint(0, 100),
        "\x00\x1b[31m\x07" * rng.randint(0, 10),
        "emoji 🎉 " * rng.randint(0, 30),
        "x\ny\n" * rng.randint(0, 50),
    ]
    for _ in range(40):
        text = "".join(rng.choice(chunks) for _ in range(rng.randint(1, 3)))
        cases.append(("estimate_chars_tokens", (text,), {}))
        cases.append(("is_cjk_text", (text,), {}))
        cases.append(("clean_text", (text,), {}))
        cases.append(("sanitize_for_tokenizer", (text,), {}))
    return cases


def _fuzz_stream_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    for _ in range(30):
        n = rng.randint(0, 200)
        text = "".join(
            rng.choice("abc \n\r\x1b[31m0m中文🎉\t") for _ in range(n)
        )
        cases.append(("filter_output", (text,), {}))
        cases.append(("strip_ansi", (text,), {}))
    return cases


def _fuzz_codec_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    for _ in range(30):
        payload = "".join(rng.choice("abc0123 {}\":,\n中") for _ in range(rng.randint(0, 60)))
        cases.append(("serialize_envelope", ("text", payload.encode()), {}))
        cases.append(("canonicalize_payload", (payload.encode(),), {}))
        cases.append(("deserialize_envelope", (payload.encode(),), {}))
    for _ in range(20):
        name = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        data = "".join(rng.choice("x\n🎉") for _ in range(rng.randint(0, 10))).encode()
        cases.append(("build_sse_frame", (name, data, rng.randint(0, 5)), {}))
    return cases


def _fuzz_diff_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    for _ in range(30):
        n = rng.randint(0, 30)
        old = "\n".join("".join(rng.choice("ab x中") for _ in range(rng.randint(0, 12))) for _ in range(n))
        new = "\n".join("".join(rng.choice("ab y中") for _ in range(rng.randint(0, 12))) for _ in range(n))
        cases.append(("unified_diff", (old.encode(), new.encode()), {}))
        cases.append(("diff_hunks", (old.encode(), new.encode()), {}))
    for _ in range(30):
        a = "".join(rng.choice("abc\t 中") for _ in range(rng.randint(0, 15)))
        b = "".join(rng.choice("abc\t 中") for _ in range(rng.randint(0, 15)))
        cases.append(("inline_diff_ranges", (a, b), {}))
        cases.append(("build_offset_map", (a, b), {}))
    return cases


def _fuzz_tools_cases(rng: random.Random) -> list[tuple[str, tuple, dict]]:
    cases = []
    for _ in range(40):
        line = "".join(rng.choice("ab 中文\t") for _ in range(rng.randint(0, 50)))
        cases.append(("line_hash", (line,), {}))
        cases.append(("line_hashes", (line + "\n" + line,), {}))
    for _ in range(20):
        content = "\n".join(
            "".join(rng.choice("abx中") for _ in range(rng.randint(0, 10)))
            for _ in range(rng.randint(0, 15))
        )
        needle = rng.choice(["a", "b", "x", "中", "zz"])
        cases.append(("find_in_file", (content, needle), {}))
        cases.append(("scan_lines", (content, needle), {}))
    for _ in range(20):
        cmd = rng.choice(["ls", "rm -rf /", "echo hi", "git push", "ping 1.1.1.1"])
        cases.append(("detect_hardline_command", (cmd,), {}))
        cases.append(("base_command_name", (cmd,), {}))
        cases.append(("command_detection_variants", (cmd,), {}))
    return cases


def _build_all_cases() -> dict[str, list[tuple[str, tuple, dict]]]:
    cases = {name: list(cases) for name, cases in BEHAVIOR_CASES.items()}
    rng = random.Random(20240811)
    cases["search"] += _fuzz_search_cases(rng)
    cases["text"] += _fuzz_text_cases(rng)
    cases["stream"] += _fuzz_stream_cases(rng)
    cases["codec"] += _fuzz_codec_cases(rng)
    cases["diff"] += _fuzz_diff_cases(rng)
    cases["tools"] += _fuzz_tools_cases(rng)
    return cases


ALL_CASES = _build_all_cases()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod_name", sorted(ALL_CASES))
def test_stateless_function_parity(mod_name):
    cases = ALL_CASES[mod_name]
    natives = [_run_case(mod_name, fn, True, args, kwargs) for fn, args, kwargs in cases]
    pythons = [_run_case(mod_name, fn, False, args, kwargs) for fn, args, kwargs in cases]
    for (fn, args, kwargs), n, p in zip(cases, natives, pythons):
        assert n == p, (
            f"{mod_name}.{fn}{args} kwargs={kwargs}\n"
            f"  native={n}\n  python={p}"
        )


def test_stateful_scenario_parity():
    """Run every stateful scenario in both modes and compare snapshots."""
    for mod_name in sorted(SCENARIOS):
        for sname, runner in sorted(SCENARIOS[mod_name].items()):
            mod_n = _load_module(mod_name, True)
            native = _norm(runner(mod_n))
            mod_p = _load_module(mod_name, False)
            python = _norm(runner(mod_p))
            assert native == python, (
                f"{mod_name} scenario {sname!r}:\n"
                f"  native={native}\n  python={python}"
            )


# ---------------------------------------------------------------------------
# exception parity (both paths must raise the same exception type)
# ---------------------------------------------------------------------------

EXCEPTION_CASES: dict[str, list[tuple[str, tuple, dict, str]]] = {
    "stream": [
        ("filter_output", (b"not a str",), {}, "TypeError"),
        ("filter_output", (123,), {}, "TypeError"),
    ],
    "parse": [
        ("comment_spans", ("nope", b"x"), {}, "ValueError"),
        ("comment_spans", ("c", "a string not bytes"), {}, "TypeError"),
        ("parse", ("nope", "x"), {}, "ValueError"),
        ("parse", ("c", 123), {}, "TypeError"),
        ("fix_bash_command", (123,), {}, "TypeError"),
        ("fix_pwsh_command", (None,), {}, "TypeError"),
        ("pwsh_transform", (None,), {}, "TypeError"),
    ],
    # NOTE: None is coerced by the shim (str(None)) on both paths, so no
    # exception is expected for text kernels — they are intentionally absent.
    "text": [],
    "tools": [
        # find_in_file(None, ...) raises AttributeError on both paths (the
        # attribute name differs: isascii vs splitlines — message-only diff).
        ("find_in_file", (None, "x"), {}, "AttributeError"),
    ],
    "search": [
        # NOTE: ngram_overlap(None, "x") is NOT in the corpus — native raises
        # AttributeError while the compat guard returns 0.0; invalid inputs of
        # the wrong type are outside the parity contract.
    ],
}


@pytest.mark.parametrize("mod_name", sorted(EXCEPTION_CASES))
def test_exception_parity(mod_name):
    """Invalid-input cases: both paths must raise the SAME exception type.

    (pybind11 and pure Python produce different message text for type errors,
    so only the exception type is part of the behavioral contract here.)
    """
    for fn, args, kwargs, expected_type in EXCEPTION_CASES[mod_name]:
        n = _run_case(mod_name, fn, True, args, kwargs)
        p = _run_case(mod_name, fn, False, args, kwargs)
        if n.get("exc") != p.get("exc"):
            raise AssertionError(
                f"{mod_name}.{fn}{args} kwargs={kwargs}: "
                f"native={n.get('exc')} python={p.get('exc')} "
                f"(expected {expected_type})"
            )
        if n.get("exc") != expected_type:
            raise AssertionError(
                f"{mod_name}.{fn}{args}: expected {expected_type}, got {n}"
            )
