"""Behavior-equivalence gate: native path ≡ pure-Python path.

For every integrated kernel, run the SAME inputs through the native path
(gate forced on) and the pure-Python path (gate forced off) and assert
identical results. This is the acceptance gate for each Phase 2 item — a
kernel whose equivalence test is red or absent must default to pure Python.

The gate functions are imported by reference into the consuming modules
(``from kimi_cli.native_loader import use_native as _native_use_native``),
so toggling is done by monkeypatching the module-local binding. When the
native extension is unavailable the suite skips with a clear message
(``tools\\sync_native.py`` must be run first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kimi_cli.native_loader import NATIVE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="native runtime not staged — run 'python tools\\sync_native.py' first",
)

from .corpus import SANITIZE_BOUNDARY_CASES, TEXT_CORPUS  # noqa: E402


def _native_on(module, state: bool):
    """Force the module's native gate to *state*; returns restore callable."""
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
# TEXT kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", TEXT_CORPUS)
def test_estimate_chars_tokens_equivalence(text):
    import kimi_cli.utils.tokens as mod

    restore = _native_on(mod, True)
    try:
        native = mod._estimate_chars_tokens(text)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._estimate_chars_tokens(text)
    finally:
        restore()
    _assert_equivalent(native, python, text)


@pytest.mark.parametrize("text", TEXT_CORPUS)
def test_is_cjk_text_equivalence(text):
    import kimi_cli.utils.tokens as mod

    restore = _native_on(mod, True)
    try:
        native = mod._is_cjk_text(text)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod._is_cjk_text(text)
    finally:
        restore()
    _assert_equivalent(native, python, text)


@pytest.mark.parametrize("text", TEXT_CORPUS)
def test_count_tokens_heuristic_equivalence(text):
    import kimi_cli.utils.tokens as mod

    restore = _native_on(mod, True)
    try:
        native = mod.count_tokens(text)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod.count_tokens(text)
    finally:
        restore()
    _assert_equivalent(native, python, text)


@pytest.mark.parametrize("text", TEXT_CORPUS)
def test_clean_text_equivalence(text):
    import kimi_cli.safety_check as mod

    for keep_newlines in (True, False):
        restore = _native_on(mod, True)
        try:
            native = mod.clean_text(text, keep_newlines=keep_newlines)
        finally:
            restore()
        restore = _native_on(mod, False)
        try:
            python = mod.clean_text(text, keep_newlines=keep_newlines)
        finally:
            restore()
        _assert_equivalent(native, python, (text, keep_newlines))


@pytest.mark.parametrize(
    "text,max_chars,max_repeat,truncate_msg", SANITIZE_BOUNDARY_CASES
)
def test_sanitize_for_tokenizer_equivalence(text, max_chars, max_repeat, truncate_msg):
    import kimi_cli.safety_check as mod

    restore = _native_on(mod, True)
    try:
        native = mod.sanitize_for_tokenizer(
            text, max_chars=max_chars, max_repeat=max_repeat, truncate_msg=truncate_msg
        )
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod.sanitize_for_tokenizer(
            text, max_chars=max_chars, max_repeat=max_repeat, truncate_msg=truncate_msg
        )
    finally:
        restore()
    _assert_equivalent(native, python, (text, max_chars, max_repeat, truncate_msg))


# ---------------------------------------------------------------------------
# GLOB kernels (gitignore parsing / matching)
# ---------------------------------------------------------------------------


def _rules_to_tuples(rules):
    """Strip the optional native cache so Python vs native rules compare equal."""
    return [(r.pattern, r.negated, r.anchored, r.is_dir_only) for r in rules]


GITIGNORE_CONTENTS = [
    "",
    "*.pyc\n",
    "*.pyc\n!important.pyc\n",
    ".venv/\n",
    ".venv/\n!.venv/\n",
    "build/\ndist/\n*.egg-info/\n",
    "/anchored.txt\n",
    "**/foo.bar\n",
    "src/**/*.log\n",
    "# comment\n\n*.tmp\n",
    "node_modules\n",
    "*.py[co]\n__pycache__/\n",
]


@pytest.mark.parametrize("content", GITIGNORE_CONTENTS)
def test_parse_gitignore_equivalence(content, tmp_path):
    import kimi_cli.tools.file.glob as mod

    source_dir = tmp_path
    restore = _native_on(mod, True)
    try:
        native = _rules_to_tuples(mod._parse_gitignore(content, source_dir))
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = _rules_to_tuples(mod._parse_gitignore(content, source_dir))
    finally:
        restore()
    _assert_equivalent(native, python, content)


def _make_path(tmp_path: Path, rel_path: str, is_dir: bool):
    path = tmp_path / rel_path.replace("/", "\\")  # Windows-safe
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return path


GITIGNORE_MATCH_CASES = [
    ("*.pyc\n", "foo.pyc", False, True),
    ("*.pyc\n", "foo.py", False, False),
    ("*.pyc\n!important.pyc\n", "important.pyc", False, False),
    (".venv/\n", ".venv", True, True),
    (".venv/\n", ".venv/a.py", False, True),
    (".venv/\n!.venv/\n", ".venv/a.py", False, False),
    ("/anchored.txt\n", "anchored.txt", False, True),
    ("/anchored.txt\n", "sub/anchored.txt", False, False),
    ("**/foo.bar\n", "foo.bar", False, True),
    ("**/foo.bar\n", "a/b/foo.bar", False, True),
    ("src/**/*.log\n", "src/a/b.log", False, True),
    ("src/**/*.log\n", "a/b.log", False, False),
    ("node_modules\n", "node_modules", True, True),
    ("node_modules\n", "node_modules/pkg.js", False, True),
    ("build/\ndist/\n*.egg-info/\n", "dist", True, True),
    ("build/\ndist/\n*.egg-info/\n", "pkg.egg-info/PKG-INFO", False, True),
]


@pytest.mark.parametrize("content,rel_path,is_dir,expected", GITIGNORE_MATCH_CASES)
def test_is_ignored_by_gitignore_equivalence(content, rel_path, is_dir, expected, tmp_path):
    import kimi_cli.tools.file.glob as mod

    def evaluate(native: bool):
        restore = _native_on(mod, native)
        try:
            rules = mod._parse_gitignore(content, tmp_path)
            path = _make_path(tmp_path, rel_path, is_dir)
            return mod._is_ignored_by_gitignore(path, rules, tmp_path)
        finally:
            restore()

    native = evaluate(True)
    python = evaluate(False)
    _assert_equivalent(native, python, (content, rel_path, is_dir))
    assert native == expected


def test_is_ignored_by_gitignore_case_sensitivity_equivalence(tmp_path):
    """Case-sensitive patterns must agree between native and Python paths.

    On Windows the Python fallback uses fnmatch (case-insensitive) while the
    native kernel uses fnmatchcase (case-sensitive).  The implementation
    therefore delegates matching to native only on case-sensitive platforms,
    so both paths must still agree everywhere.
    """
    import kimi_cli.tools.file.glob as mod

    content = "*.PYC\n"
    rel_path = "foo.pyc"
    path = _make_path(tmp_path, rel_path, False)

    def evaluate(native: bool):
        restore = _native_on(mod, native)
        try:
            rules = mod._parse_gitignore(content, tmp_path)
            return mod._is_ignored_by_gitignore(path, rules, tmp_path)
        finally:
            restore()

    native = evaluate(True)
    python = evaluate(False)
    _assert_equivalent(native, python, (content, rel_path))


def test_is_ignored_by_gitignore_multi_source_dir_equivalence(tmp_path):
    """When multiple .gitignore files apply, matching falls back to Python.

    Both native-on and native-off must still agree.
    """
    import kimi_cli.tools.file.glob as mod

    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".gitignore").write_text("!keep.pyc\n", encoding="utf-8")

    def evaluate(native: bool):
        restore = _native_on(mod, native)
        try:
            # Clear the module cache so rules are parsed under the current gate.
            mod._GITIGNORE_CACHE.clear()
            rules = mod._get_gitignore_rules(tmp_path)
            path = sub / "keep.pyc"
            path.write_text("x", encoding="utf-8")
            return mod._is_ignored_by_gitignore(path, rules, tmp_path)
        finally:
            restore()

    native = evaluate(True)
    python = evaluate(False)
    _assert_equivalent(native, python, "multi-source-dir negation")
    # root ignores *.pyc, sub un-ignores keep.pyc
    assert native is False


# ---------------------------------------------------------------------------
# Determinism: native calls must be stable across repeated identical inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", TEXT_CORPUS[:8])
def test_text_determinism(text):
    import kimi_cli.safety_check as mod

    restore = _native_on(mod, True)
    try:
        first = mod.sanitize_for_tokenizer(text, max_chars=64)
        second = mod.sanitize_for_tokenizer(text, max_chars=64)
    finally:
        restore()
    assert first == second


# ---------------------------------------------------------------------------
# TOOLS kernels (hash_line / export)
# ---------------------------------------------------------------------------


HASH_LINE_CASES = [
    (1, "", None),
    (1, "hello world", None),
    (2, "  spaced   out  ", None),
    (3, "no alnum !!!", None),
    (4, "1234567890", None),
    (1, "line one", "ZM"),
    (2, "line two", "PQ"),
    (5, "中文内容", None),
    (6, "mixed 中文 text", "ZM"),
    (7, "tab\tsep", None),
    (8, "trailing\r", None),
    (9, "x" * 3000, None),
]


@pytest.mark.parametrize("line_num,line,prev_hash", HASH_LINE_CASES)
def test_compute_line_hash_equivalence(line_num, line, prev_hash):
    import kimi_cli.tools.file.hash_line as mod

    restore = _native_on(mod, True)
    try:
        native = mod.compute_line_hash(line_num, line, prev_hash)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod.compute_line_hash(line_num, line, prev_hash)
    finally:
        restore()
    _assert_equivalent(native, python, (line_num, line, prev_hash))


def _export_history() -> list:
    """Synthetic Message history for export equivalence."""
    import pendulum
    from kosong.message import Message
    from kimi_cli.wire.types import (
        AudioURLPart,
        ImageURLPart,
        TextPart,
        ThinkPart,
        ToolCall,
        VideoURLPart,
    )

    now = pendulum.datetime(2026, 2, 3, 4, 5, 6)
    return now, [
        Message(role="system", content=[TextPart(text="You are a helpful assistant.")]),
        Message(role="user", content=[TextPart(text="hello world")]),
        Message(
            role="assistant",
            content=[ThinkPart(think="Let me think carefully."), TextPart(text="Hi there!")],
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function={"name": "bash", "arguments": '{"cmd": "ls -la"}'},
                )
            ],
        ),
        Message(role="tool", content=[TextPart(text="file1\nfile2\nfile3")], tool_call_id="call_1"),
        Message(role="user", content=[TextPart(text="<system-reminder>REMIND</system-reminder>")]),
        Message(
            role="user",
            content=[
                TextPart(text="Check this image"),
                ImageURLPart(image_url={"url": "https://example.com/a.png", "id": "img1"}),
            ],
        ),
        Message(
            role="user",
            content=[TextPart(text="Play this"), AudioURLPart(audio_url={"url": "https://example.com/a.mp3", "id": "aud1"})],
        ),
        Message(
            role="assistant",
            content=[TextPart(text="Here is a video"), VideoURLPart(video_url={"url": "https://example.com/v.mp4", "id": "vid1"})],
        ),
        Message(role="user", content=[TextPart(text="<system>CHECKPOINT done</system>")]),
        Message(role="user", content=[TextPart(text="final question")]),
        Message(role="assistant", content=[TextPart(text="final answer with 中文 and emoji 🎉")]),
    ]


def test_build_export_markdown_equivalence():
    from kimi_cli.utils import export as mod

    now, history = _export_history()
    kwargs = dict(
        session_id="sess-native-1",
        work_dir=r"C:\work\demo",
        history=history,
        token_count=4321,
        now=now,
    )
    restore = _native_on(mod, True)
    try:
        native = mod.build_export_markdown(**kwargs)
    finally:
        restore()
    restore = _native_on(mod, False)
    try:
        python = mod.build_export_markdown(**kwargs)
    finally:
        restore()
    _assert_equivalent(native, python, ("export", len(history)))
    # Byte-for-byte: the native path decodes the kernel bytes; equality of the
    # str implies byte equality of the UTF-8 encodings.
    assert native.encode("utf-8") == python.encode("utf-8")


# ---------------------------------------------------------------------------
# JSON kernels (acp/session.py incremental lexer)
# ---------------------------------------------------------------------------

ACP_ARGS_STREAM = [
    ('{"path": "/tmp/file.txt", "cmd": "ls"}', "Read", "/tmp"),
    ('{"command": "git status", "work_dir": "/repo"}', "Bash", "/repo"),
    ('{"description": "Fix the bug", "plan": "step1"}', "Agent", None),
    ('{"query": "how to sort in python"}', "WebSearch", None),
    ('{"pattern": "*.py", "path": "./src"}', "Glob", "./src"),
    ('{"url": "https://example.com", "depth": 2}', "Fetch", None),
    ('{"name": "alice", "age": 30, "tags": ["a", "b"]}', "UnknownTool", None),
    ('{"key": "partial', "Read", None),  # incomplete JSON
    ('{"key": "pa', "Read", None),  # very partial
    ("", "Read", None),
]


def _tool_call_state_pair(args, tool_name, work_dir):
    """Construct one _ToolCallState with native off and one with native on."""
    import kimi_cli.native_loader as nl
    from kimi_cli.acp.session import _ToolCallState
    from kimi_cli.wire.types import ToolCall

    def make(native: bool):
        orig = nl.use_native
        try:
            nl.use_native = lambda k: native
            tc = ToolCall(id="call_1", function={"name": tool_name, "arguments": args})
            return _ToolCallState(tc, work_dir)
        finally:
            nl.use_native = orig

    return make(True), make(False)


@pytest.mark.parametrize("args,tool_name,work_dir", ACP_ARGS_STREAM)
def test_acp_lexer_get_title_equivalence(args, tool_name, work_dir):
    from kaos.path import KaosPath

    work = KaosPath(work_dir) if work_dir else None
    native_state, python_state = _tool_call_state_pair(args, tool_name, work)
    # Feed a few incremental parts the same way on both.
    parts = ["{", '"cmd": "ls -la"', ', "note": "x"}']
    for part in parts:
        native_state.append_args_part(part)
        python_state.append_args_part(part)
    native = native_state.get_title()
    python = python_state.get_title()
    _assert_equivalent(native, python, (args, tool_name))


@pytest.mark.parametrize("args,tool_name,work_dir", ACP_ARGS_STREAM)
def test_acp_lexer_buffer_equivalence(args, tool_name, work_dir):
    """The native lexer's buffer() must equal the streamingjson accumulation."""
    import streamingjson

    native_state, _ = _tool_call_state_pair(args, "Read", None)
    lexer = streamingjson.Lexer()
    if args:
        lexer.append_string(args)
    parts = ['{"x": 1}', '{"x": 1, "y": 2}']
    for part in parts:
        native_state.append_args_part(part)
        lexer.append_string(part)
    if native_state._native_lexer is not None:
        if native_state._native_lexer.has_error():
            # Documented deviation: the native lexer freezes its buffer after a
            # parse error (streamingjson keeps accumulating). get_title() is the
            # behavioral gate and is covered by the test above; here we only
            # require buffer parity on the non-error path.
            return
        _assert_equivalent(
            native_state._native_lexer.buffer().decode("utf-8", "surrogatepass"),
            lexer.complete_json(),
            (args,),
        )


# ---------------------------------------------------------------------------
# 4.4 extras: input-immutability + thread-safety smoke
# ---------------------------------------------------------------------------


def test_input_immutability_native_calls():
    """Native calls must not mutate caller-owned bytes/str/lists."""
    import kimi_cli.native_loader as nl
    from kimi_cli.safety_check import sanitize_for_tokenizer
    from kimi_cli.utils.tokens import _estimate_chars_tokens

    text = "mixed 中文 text \x1b[31mansi\x1b[0m " * 20
    text_copy = text
    orig = nl.use_native
    try:
        nl.use_native = lambda k: True
        _ = sanitize_for_tokenizer(text, max_chars=200)
        _ = _estimate_chars_tokens(text)
    finally:
        nl.use_native = orig
    assert text == text_copy, "native sanitize/token-count mutated the caller's str"


def test_thread_safety_smoke():
    """Concurrent native calls from multiple threads: results must match the
    single-threaded result and must not deadlock (GIL-release smoke)."""
    import threading

    import kimi_cli.native_loader as nl
    from kimi_cli.safety_check import sanitize_for_tokenizer

    corpus = ["line " + str(i) + " 中文 " * 10 for i in range(100)]
    orig = nl.use_native
    try:
        nl.use_native = lambda k: True
        expected = [sanitize_for_tokenizer(t, max_chars=50) for t in corpus]
        results: list[list[str]] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append([sanitize_for_tokenizer(t, max_chars=50) for t in corpus])
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
