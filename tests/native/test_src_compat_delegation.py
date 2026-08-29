"""Conformance gate: every refactored src/kimi-cli function delegates its
pure-Python fallback to the canonical ``kimix_native`` ``_compat_*`` copy.

After the dedup refactor, the pure-Python algorithm for each function below
lives ONCE, in ``bin/kimix_native``.  The src/kimi-cli wrappers keep their
native fast-path dispatch and delegate the fallback to the shim.  This test
forces the src gate OFF (pure-Python path) and asserts the result is identical
to the shim's ``_compat_*`` implementation on representative corpora — so a
regression that re-introduces a divergent body (or breaks the delegation) is
caught here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BIN = _REPO / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))


def _force_pure(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "_native_use_native", lambda kernel: False)


def _assert_eq(monkeypatch, src_module, src_name, shim_mod, shim_name, args):
    _force_pure(monkeypatch, src_module)
    src_fn = getattr(src_module, src_name)
    shim_fn = getattr(shim_mod, shim_name)
    try:
        a = src_fn(*args)
        ae = None
    except Exception as e:  # noqa: BLE001
        a, ae = None, (type(e).__name__, str(e))
    try:
        b = shim_fn(*args)
        be = None
    except Exception as e:  # noqa: BLE001
        b, be = None, (type(e).__name__, str(e))
    if ae is not None or be is not None:
        assert ae is not None and be is not None and ae[0] == be[0], (
            f"{src_name} vs {shim_name} exception mismatch: {ae} vs {be}"
        )
    else:
        assert a == b, f"{src_name} != {shim_name} for {args!r}: {a!r} vs {b!r}"


# ---------------------------------------------------------------------------
# src/kimix security + shell helpers
# ---------------------------------------------------------------------------

def test_security_delegation(monkeypatch):
    import kimix.tools.security as src
    import kimix_native.tools as sh

    envs = [
        {},
        {"PATH": "/usr/bin", "HOME": "/root"},
        {"PATH": "/usr/bin", "API_KEY": "x", "SSH_AUTH_SOCK": "/tmp/s"},
        {"AWS_SECRET_ACCESS_KEY": "x", "DATABASE_URL": "postgres://u:p@h/db"},
    ]
    for env in envs:
        _assert_eq(monkeypatch, src, "scrub_child_env", sh, "_compat_scrub_child_env", (env,))

    redact = ["", "plain", "password=x", "https://user:pass@example.com",
              "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345"]
    for out in redact:
        _assert_eq(monkeypatch, src, "redact_sensitive_output", sh, "_compat_redact_sensitive_output", (out,))

    for wd in [None, "", "/tmp", "a$b", "a;b", "C:\\x"]:
        _assert_eq(monkeypatch, src, "validate_workdir", sh, "_compat_validate_workdir", (wd,))


def test_output_enhance_delegation(monkeypatch):
    import kimix.tools.file.bash.output_enhance as src
    import kimix_native.tools as sh

    cmds = ["", "grep foo", "diff a b", "test -f x", "find / -name x", "curl https://x",
            "git diff", "echo hi | head -n 5", "ls", "FOO=1 git status"]
    for cmd in cmds:
        for code in [None, 0, 1, 2, 141]:
            _assert_eq(monkeypatch, src, "is_expected_exit", sh, "_compat_is_expected_exit", (cmd, code))
            _assert_eq(monkeypatch, src, "interpret_exit_code", sh, "_compat_interpret_exit_code", (cmd, code))
    for out in ["", "bash: ls: command not found", "cat: no such file or directory",
                "Permission denied", "some random output"]:
        _assert_eq(monkeypatch, src, "annotate_failure", sh, "_compat_annotate_failure", (out, "cmd", 1))


def test_safety_delegation(monkeypatch):
    import kimix.tools.file.bash.safety as src
    import kimix_native.tools as sh

    cmds = ["", "  ", "rm -rf /", "echo hi", "ls -la", "git status", "npm run dev",
            "shutdown now", "kill 1", "rm file.txt"]
    for cmd in cmds:
        _assert_eq(monkeypatch, src, "command_detection_variants", sh, "_compat_command_detection_variants", (cmd,))
        _assert_eq(monkeypatch, src, "detect_hardline_command", sh, "_compat_detect_hardline_command", (cmd,))
        _assert_eq(monkeypatch, src, "check_hardline_blocked", sh, "_compat_check_hardline_blocked", (cmd,))
        _assert_eq(monkeypatch, src, "foreground_background_guidance", sh, "_compat_foreground_background_guidance", (cmd,))
        _assert_eq(monkeypatch, src, "_segment_tokens", sh, "_compat_segment_tokens", (cmd, 0))
    _assert_eq(monkeypatch, src, "_looks_like_flag", sh, "_compat_looks_like_flag", ("-rf",))


def test_common_stream_delegation(monkeypatch):
    import kimix.tools.common as src_common
    import kimix.utils.prompt_str as src_prompt
    import kimix.ui.printing as src_printing
    import kimix_native.stream as sh_stream
    import kimix_native.text as sh_text
    import kimix_native._shell_compat as sh_shell

    outs = ["", "plain", "a\r\nb\r\nc", "a\rb", "\x1b[31mred\x1b[0m", "x\x1b]0;title\x07y"]
    for out in outs:
        _assert_eq(monkeypatch, src_common, "filter_output", sh_stream, "_compat_filter_output", (out,))
        _assert_eq(monkeypatch, src_printing, "_strip_ansi", sh_stream, "strip_ansi", (out,))
    for out in ["", "hello", "a\u200bb", "a\x00b", "café"]:
        _assert_eq(monkeypatch, src_prompt, "clean_text", sh_text, "_compat_clean_text", (out,))
    # shell scanning helpers (delegated to the shim's _shell_compat)
    for cmd in ["echo 'hi'", 'echo "a$(x)y"', 'cmd "a`x`b"', "echo $'a\\nb'"]:
        for s in range(len(cmd)):
            _assert_eq(monkeypatch, src_common, "_find_ansi_c_end", sh_shell, "_find_ansi_c_end", (cmd, s))
            _assert_eq(monkeypatch, src_common, "_find_backtick_end", sh_shell, "_find_backtick_end", (cmd, s))
            _assert_eq(monkeypatch, src_common, "_find_dq_end", sh_shell, "_find_dq_end", (cmd, s))
            if cmd[s] == "(":
                _assert_eq(monkeypatch, src_common, "_find_matching_paren", sh_shell, "_find_matching_paren", (cmd, s))


# ---------------------------------------------------------------------------
# kimi-cli modules
# ---------------------------------------------------------------------------

def test_kimi_tokens_delegation(monkeypatch):
    import kimi_cli.utils.tokens as src
    import kimix_native.text as sh

    texts = ["", "hello world", "你好世界", "mixed ascii 和中文", "a" * 1000]
    for t in texts:
        _assert_eq(monkeypatch, src, "_is_cjk_text", sh, "_compat_is_cjk_text", (t,))
        _assert_eq(monkeypatch, src, "_estimate_chars_tokens", sh, "_compat_estimate", (t,))
        _assert_eq(monkeypatch, src, "count_tokens", sh, "count_tokens", (t,))


def test_kimi_safety_check_delegation(monkeypatch):
    import kimi_cli.safety_check as src
    import kimix_native.text as sh

    for t in ["", "normal text", "a\u200b\u200cb", "\ud800bad", "\ufdd0x", "rep" * 500, "x" * 3000]:
        _assert_eq(monkeypatch, src, "clean_text", sh, "_compat_clean_text", (t,))
        _assert_eq(monkeypatch, src, "sanitize_for_tokenizer", sh, "_compat_sanitize_for_tokenizer", (t,))


def test_retrieval_delegation(monkeypatch):
    import kimix.retrieval as src
    import kimix_native.search as sh

    pairs = [("", ""), ("a", "a"), ("ab", "ab"), ("abc", "abd"), ("hello", "hallo"),
             ("saturday", "sunday"), ("kitten", "sitting"), ("марія", "мария")]
    for a, b in pairs:
        _assert_eq(monkeypatch, src, "jaro_similarity", sh, "_compat_jaro_similarity", (a, b))
        _assert_eq(monkeypatch, src, "ngram_overlap", sh, "_compat_ngram_overlap", (a, b))
        _assert_eq(monkeypatch, src, "sorensen_dice_coefficient", sh, "_compat_sorensen_dice", (a, b))


def test_kimi_toolset_grep_micro_filter_delegation(monkeypatch):
    import kimi_cli.soul.toolset as src_ts
    import kimi_cli.tools.file.grep_local as src_grep
    import kimi_cli.tools.file.micro_compress as src_mc
    import kimi_cli.utils.file_filter as src_ff
    import kimix_native.tools as sh_tools
    import kimix_native.codec as sh_codec
    import kimix_native.glob as sh_glob

    # toolset._sort_json_value has no native gate — it is a pure delegation,
    # so compare it directly without forcing the gate.
    assert src_ts._sort_json_value({"b": 1, "a": [3, 1]}) == \
        sh_codec._sort_json_value({"b": 1, "a": [3, 1]})
    for p in ["foo", "a\nb", "(?s)a.b", "^x$", "a|b"]:
        _assert_eq(monkeypatch, src_grep, "_pattern_has_regex_newline", sh_tools,
                   "_compat_pattern_has_regex_newline", (p,))
        _assert_eq(monkeypatch, src_grep, "_multiline_pattern", sh_tools,
                   "_compat_multiline_pattern", (p,))
    for t in ["", "   a   b   ", "\x1b[31mred\x1b[0m", "  42\tline\n 43\tnext"]:
        _assert_eq(monkeypatch, src_mc, "strip_control_noise", sh_tools,
                   "_compat_compress_strip_control_noise", (t,))
        _assert_eq(monkeypatch, src_mc, "renumber_lines", sh_tools,
                   "_compat_compress_renumber_lines", (t,))
    for name in ["node_modules", "main.py", "", "vendor", ".git"]:
        _assert_eq(monkeypatch, src_ff, "is_ignored", sh_glob, "_compat_is_ignored_name", (name,))
