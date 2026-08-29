"""Behavior-equivalence gate for the src/kimix native kernels.

Same principle as kimi-cli/tests/native/test_behavior_equivalence.py: run the
SAME inputs through the native path (gate forced on) and the pure-Python path
(gate forced off), assert identical results. A kernel whose equivalence test
is red or absent must default to pure Python.
"""

from __future__ import annotations

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)

from .corpus import DEDUP_CORPUS, FIND_STR_CORPUS, STREAM_CORPUS  # noqa: E402


def _force_gate(module, state: bool):
    attr = "_native_use_native"
    original = getattr(module, attr, None)
    setattr(module, attr, lambda kernel: state)
    return lambda: setattr(module, attr, original) if original is not None else delattr(
        module, attr
    )


def _assert_equivalent(native_result, python_result, case):
    assert native_result == python_result, (
        f"native != python for {case!r}:\n"
        f"  native={native_result!r}\n  python={python_result!r}"
    )


# ---------------------------------------------------------------------------
# STREAM kernels (src/kimix/tools/common.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", STREAM_CORPUS)
def test_filter_output_equivalence(text):
    import kimix.tools.common as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.filter_output(text)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.filter_output(text)
    finally:
        restore()
    _assert_equivalent(native, python, text)


def test_filter_output_typeerror_parity():
    import kimix.tools.common as mod

    restore = _force_gate(mod, True)
    try:
        with pytest.raises(TypeError):
            mod.filter_output(b"not a str")  # type: ignore[arg-type]
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        with pytest.raises(TypeError):
            mod.filter_output(b"not a str")  # type: ignore[arg-type]
    finally:
        restore()


@pytest.mark.parametrize(
    "output,threshold,max_block_lines", DEDUP_CORPUS
)
def test_dedup_output_equivalence(output, threshold, max_block_lines):
    import kimix.tools.common as mod

    restore = _force_gate(mod, True)
    try:
        native = mod._dedup_output(output, threshold, max_block_lines=max_block_lines)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod._dedup_output(output, threshold, max_block_lines=max_block_lines)
    finally:
        restore()
    _assert_equivalent(native, python, (output, threshold, max_block_lines))


# ---------------------------------------------------------------------------
# TOOLS kernels (src/kimix/tools/file/find_str.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content,needle", FIND_STR_CORPUS)
def test_find_in_file_equivalence(content, needle):
    """Run the real FindStr tool end-to-end in both modes over a temp file."""
    import asyncio
    import tempfile
    from pathlib import Path

    from kimix.tools.file import find_str as mod
    from kimix.tools.file.find_str import FindStr, FindStrParams

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text(content, encoding="utf-8")

        async def run(native: bool) -> str:
            restore = _force_gate(mod, native)
            try:
                tool = FindStr()
                res = await tool(
                    FindStrParams(content=needle, path=str(path), case_sensitive=False)
                )
                return res.output if hasattr(res, "output") else str(res)
            finally:
                restore()

        _assert_equivalent(asyncio.run(run(True)), asyncio.run(run(False)), (content, needle))


# ---------------------------------------------------------------------------
# PARSE kernels (comment parsers + bash/pwsh scanners)
# ---------------------------------------------------------------------------

PARSE_CORPUS = {
    "c": [
        "// hi\nint x;\n",
        "int x = 1;\n// A line comment\n",
        "/* A block\ncomment */\n",
        "/** A doc comment */\n",
        'char *s = "not // a comment";\n// real\n',
        "/* unclosed block\nstill going",
        "int a; /* c1 */ int b; /* c2 */\n",
        "// line1\n// line2\nint x;\n",
    ],
    "python": [
        "# comment\nx = 1\n",
        '"""docstring"""\nx = 1\n',
        "s = '#'\n# real comment\n",
        'x = "not # a comment"\n# yes\n',
        "'''unclosed doc",
        "def f():\n    # indent\n    pass\n",
    ],
    "shell": [
        "# comment\necho hi\n",
        'echo "# not comment" # yes\n',
        "echo 'single # quote'\n# real\n",
        "if true; then\n  # inside\n  echo ok\nfi\n",
        "echo unclosed \\\n  continuation # c\n",
    ],
    "sql": [
        "-- line comment\nSELECT 1;\n",
        "/* block */ SELECT 2;\n",
        "SELECT '-- not comment';\n-- real\n",
        "/* unclosed",
    ],
    "html": [
        "<!-- comment --><p>text</p>\n",
        "<?php echo 1; ?>\n",
        "<p><!-- inline --></p>\n<!-- unclosed",
    ],
    "lisp": [
        "; comment\n(defun f (x) x)\n",
        '"; not comment"\n; real\n',
        "#| block |# (f 1)\n",
    ],
    "pascal": [
        "{ comment }\nprogram p;\n",
        "(* block *) begin end.\n",
        "// line\nbegin\nend.\n",
        "{ unclosed",
    ],
}

PARSER_CLASSES = [
    ("c", "CParser", "C", "c_parser"),
    ("python", "PythonParser", "Python", "py_parser"),
    ("shell", "ShellParser", "Shell", "shell_parser"),
    ("sql", "SqlParser", "SQL", "sql_parser"),
    ("html", "HtmlParser", "HTML", "html_parser"),
    ("lisp", "LispParser", "Lisp", "lisp_parser"),
    ("pascal", "PascalParser", "Pascal", "pascal_parser"),
]


@pytest.mark.parametrize("lang,cls_name,_,module_name", PARSER_CLASSES)
def test_parser_equivalence(lang, cls_name, _, module_name):
    import kimi_cli.native_loader as nl

    parser_cls = __import__(
        f"kimix.parser.{module_name}", fromlist=[cls_name]
    ).__dict__[cls_name]
    for src in PARSE_CORPUS[lang]:
        orig = nl.use_native
        try:
            nl.use_native = lambda k: True
            native = parser_cls().parse(src)
            nl.use_native = lambda k: False
            python = parser_cls().parse(src)
        finally:
            nl.use_native = orig
        assert native == python, (
            f"{lang} parse mismatch for {src!r}:\n"
            f"  native={native!r}\n  python={python!r}"
        )


BASH_FIX_CASES = [
    "echo hello",
    "rev file.txt",
    "traceroute 8.8.8.8",
    "tree .",
    "nc -l 8080",
    "wget http://example.com/x",
    'grep -r "rev" .',
    "echo a; rev b; echo c",
    "cat <<EOF\nrev\nEOF\nrev",
    "cat <<$'\\U00110000'\nbody\n\\U00110000\nrev <<< after",
    "cat <<$'x'\nbody\nx\nrev <<< after",
    "ls -la C:\\Users\\me\\file.txt",
    "cd 'C:\\Program Files'",
    "python3 -m pip install x",
    "printf '%s\\n' 'a b'",
    "x=$(rev <<< 'hello')",
    "bash cd /c/dev/x && echo ok",
    "bash -c 'rev'",
    r"bash -c 'cd C:\x && rev'",
    "sh -c 'rev'",
    "'bash' cd /c/dev/x && echo ok",
    "bash script.sh",
    "cat /tmp/x.txt",
    "echo /c/dev && echo /tmp/y",
    "env --chdir=/tmp cmd",
    "bash -c 'cat /tmp/x'",
    "cd /d/foo",
    "echo '/tmp/x'",
]


@pytest.mark.parametrize("command", BASH_FIX_CASES)
def test_fix_bash_command_equivalence(command):
    from kimix.tools.file.bash import bash_fix as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.fix_bash_command(command)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.fix_bash_command(command)
    finally:
        restore()
    assert (
        native.command,
        native.replacements,
        native.path_changes,
    ) == (
        python.command,
        python.replacements,
        python.path_changes,
    ), f"bash_fix mismatch for {command!r}"


@pytest.mark.parametrize(
    "command",
    [
        "dir\\file",
        r"ls C:\Users\me",
        "'quoted\\path'",
        "echo a\\ b",
        "x=$(echo a\\b)",
        "rm -rf ./dist\\build",
        "cd sub\\dir && make",
    ],
)
def test_process_unquoted_equivalence(command):
    from kimix.tools.file.bash import bash_tool as mod

    restore = _force_gate(mod, True)
    try:
        native = mod._process_unquoted(command)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod._process_unquoted(command)
    finally:
        restore()
    _assert_equivalent(native, python, command)


PWSH_FIX_CASES = [
    'Write-Output "hello"',
    "Get-ChildItem C:\\Users",
    "Write-Output 'He said \"hi\"'",
    'Write-Output "a`"b"',
    '$x = @"\nline " with " quotes\n"@\nWrite-Output $x',
    "# comment \" quote\nWrite-Output ok",
    'cmd /c echo --% "hello world',
    'Write-Output "a$( "b" )c"',
    "Get-Process | Where-Object { $_.CPU -gt 10 }",
]


@pytest.mark.parametrize("command", PWSH_FIX_CASES)
def test_fix_pwsh_command_equivalence(command):
    from kimix.tools.file.bash import pwsh_fix as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.fix_pwsh_command(command)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.fix_pwsh_command(command)
    finally:
        restore()
    assert (native is None) == (python is None), f"pwsh None-ness mismatch for {command!r}"
    if native is not None:
        assert (native.command, native.warning) == (python.command, python.warning), (
            f"pwsh_fix mismatch for {command!r}"
        )


# ---------------------------------------------------------------------------
# INDEX + SEARCH kernels (src/kimix/retrieval.py)
# ---------------------------------------------------------------------------

RETRIEVAL_TEXT_CORPUS = [
    "",
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    "中文检索测试文本内容",
    "mixed 中文 with ascii words",
    "a" * 60,
    "word word repeat repeat word",
    "café résumé naïve — dash",
    "emoji 🎉 test",
    "   padded   ",
]

NGRAM_N_VALUES = [1, 2, 3]


@pytest.mark.parametrize("n", NGRAM_N_VALUES)
def test_ngram_tokenizer_equivalence(n):
    from kimix import retrieval as mod

    restore = _force_gate(mod, True)
    try:
        native_nt = mod.NgramTokenizer(n=n)
        native = [
            (native_nt.normalize(t), native_nt.tokenize(t), native_nt._detect_n(t))
            for t in RETRIEVAL_TEXT_CORPUS
        ]
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python_nt = mod.NgramTokenizer(n=n)
        python = [
            (python_nt.normalize(t), python_nt.tokenize(t), python_nt._detect_n(t))
            for t in RETRIEVAL_TEXT_CORPUS
        ]
    finally:
        restore()
    _assert_equivalent(native, python, ("ngram", n))


@pytest.mark.parametrize(
    "a,b",
    [
        ("", ""),
        ("abc", ""),
        ("abc", "abc"),
        ("abc", "abd"),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("abc", "cba"),
        ("中文", "中文"),
        ("中文", "英文"),
        ("book", "back"),
        ("a", "b"),
        ("ab", "ba"),
        ("x" * 50, "y" * 50),
    ],
)
def test_damerau_levenshtein_equivalence(a, b):
    from kimix import retrieval as mod

    restore = _force_gate(mod, True)
    try:
        native = mod.LevenshteinAutomaton._damerau_levenshtein(a, b)
    finally:
        restore()
    restore = _force_gate(mod, False)
    try:
        python = mod.LevenshteinAutomaton._damerau_levenshtein(a, b)
    finally:
        restore()
    _assert_equivalent(native, python, (a, b))


@pytest.mark.parametrize(
    "a,b",
    [
        ("", ""),
        ("abc", ""),
        ("abc", "abc"),
        ("abc", "abd"),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("jellyfish", "smellyfish"),
        ("中文", "中文"),
        ("abc", "cba"),
    ],
)
def test_fuzzy_functions_equivalence(a, b):
    from kimix import retrieval as mod

    for fn_name, args in [
        ("jaro_similarity", (a, b)),
        ("jaro_winkler_similarity", (a, b)),
        ("sorensen_dice_coefficient", (a, b)),
        ("ngram_overlap", (a, b, 2)),
    ]:
        fn = getattr(mod, fn_name)
        restore = _force_gate(mod, True)
        try:
            native = fn(*args)
        finally:
            restore()
        restore = _force_gate(mod, False)
        try:
            python = fn(*args)
        finally:
            restore()
        assert native == python or (
            isinstance(native, float)
            and isinstance(python, float)
            and abs(native - python) < 1e-15
        ), f"{fn_name} mismatch for {(a, b)}: native={native!r} python={python!r}"


def test_jaro_winkler_nondefault_prefix_keeps_python():
    """max_prefix != 4 must stay on the pure-Python path."""
    import kimi_cli.native_loader as nl
    from kimix import retrieval as mod

    restore = _force_gate(mod, True)
    try:
        v4 = mod.jaro_winkler_similarity("abcde", "abxde")
        assert nl.use_native("SEARCH")  # gate is on for other kernels
        r3 = mod.jaro_winkler_similarity("abcde", "abxde", max_prefix=3)
    finally:
        restore()
    # The native kernel fixes max_prefix at 4; calling with max_prefix=3 falls
    # back to Python, which must equal the pure-Python result.
    assert r3 == mod.jaro_winkler_similarity("abcde", "abxde", max_prefix=3)
    assert isinstance(v4, float)


# ---------------------------------------------------------------------------
# 4.4 extras: input-immutability + thread-safety smoke
# ---------------------------------------------------------------------------


def test_input_immutability_native_calls():
    """Native calls must not mutate caller-owned bytes/str/lists."""
    import kimi_cli.native_loader as nl
    from kimix.tools.common import _dedup_output, filter_output

    text = "line1\nline2\n\x1b[31mred\x1b[0m\r\nline4"
    text_copy = text
    orig = nl.use_native
    try:
        nl.use_native = lambda k: True
        _ = filter_output(text)
        _ = _dedup_output(text, 3, max_block_lines=1)
    finally:
        nl.use_native = orig
    assert text == text_copy, "filter/dedup mutated the caller's str"


def test_thread_safety_smoke():
    """Concurrent native calls from multiple threads: results must match the
    single-threaded result and must not deadlock (GIL-release smoke)."""
    import threading

    import kimi_cli.native_loader as nl
    from kimix.tools.common import filter_output

    corpus = ["\x1b[31mred\x1b[0m line " + str(i) for i in range(200)]
    orig = nl.use_native
    try:
        nl.use_native = lambda k: True
        expected = [filter_output(t) for t in corpus]
        results: list[list[str]] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append([filter_output(t) for t in corpus])
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
    finally:
        nl.use_native = orig

    assert not errors, f"thread errors: {errors!r}"
    assert all(r == expected for r in results), "threaded results differ from single-threaded"
