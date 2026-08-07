"""Tests for token filter pipeline: _dedup_output, _truncate_lines, _token_filter_output."""

import os
from unittest.mock import patch

import pytest

from kimix.tools.common import (
    _dedup_output,
    _is_known_rtk_command,
    _maybe_rewrite_shell_command_with_rtk,
    _rtk_available,
    _rtk_binary_path,
    _token_filter_output,
    _truncate_lines,
)

# ── _dedup_output tests ──────────────────────────────────────────────

def test_dedup_empty():
    assert _dedup_output("") == ""


def test_dedup_single_line():
    assert _dedup_output("hello") == "hello"


def test_dedup_no_repeats():
    out = "a\nb\nc\nd"
    assert _dedup_output(out) == out


def test_dedup_below_threshold():
    # 3 copies, threshold=3 → all pass through
    out = "x\nx\nx"
    assert _dedup_output(out, threshold=3) == out


def test_dedup_above_threshold():
    # 4 copies, threshold=3 → collapsed to single annotated line
    out = "x\n" * 4
    result = _dedup_output(out.strip(), threshold=3)
    assert result == "x  (4 repeats)"


def test_dedup_preserves_order():
    out = "a\nb\na\nb\na\nc"
    # a appears 3× (≤3 threshold) → all pass; b appears 2× → all pass; c appears 1×
    result = _dedup_output(out, threshold=3)
    assert result == out


def test_dedup_interleaved_repeats():
    out = "ERROR\nINFO\nERROR\nINFO\nERROR\nINFO\nERROR"
    # ERROR 4× (>3) → collapsed; INFO 3× (≤3) → passes
    result = _dedup_output(out, threshold=3)
    assert "ERROR  (4 repeats)" in result
    assert result.count("INFO") == 3  # all 3 INFO lines preserved


def test_dedup_large_input():
    # 10,000 lines, 9,900 unique, 100 repeated 100×
    import random
    lines = [f"unique_{i}" for i in range(9900)] + ["repeat_me"] * 100
    random.shuffle(lines)
    out = "\n".join(lines)
    result = _dedup_output(out, threshold=3)
    assert "repeat_me  (100 repeats)" in result
    # Unique lines all present
    for i in range(9900):
        assert f"unique_{i}" in result


def test_dedup_multiline_two_line_block():
    out = "A\nB\n" * 4
    result = _dedup_output(out.strip(), threshold=3, max_block_lines=2)
    assert result == "A\nB  (4 repeats)"


def test_dedup_multiline_two_line_block_below_threshold():
    out = "A\nB\n" * 3
    result = _dedup_output(out.strip(), threshold=3, max_block_lines=2)
    assert result == out.strip()


def test_dedup_multiline_prefers_larger_block():
    # Should collapse as a 2-line block, not as individual A/B lines.
    out = "A\nB\n" * 5
    result = _dedup_output(out.strip(), threshold=3, max_block_lines=2)
    assert result == "A\nB  (5 repeats)"
    assert result.count("A") == 1


def test_dedup_multiline_mixed_repeats():
    out = "X\n" * 5 + "A\nB\n" * 4 + "Y\n" * 2
    result = _dedup_output(out.strip(), threshold=3, max_block_lines=2)
    assert "X  (5 repeats)" in result
    assert "A\nB  (4 repeats)" in result
    assert "Y" in result  # only 2 repeats, passes through


def test_dedup_multiline_block_larger_than_run():
    out = "A\nB\n" * 4
    result = _dedup_output(out.strip(), threshold=3, max_block_lines=5)
    assert result == "A\nB  (4 repeats)"


def test_dedup_multiline_non_contiguous_blocks_unchanged():
    out = "A\nB\nC\nA\nB\nD"
    result = _dedup_output(out, threshold=2, max_block_lines=2)
    assert result == out


# ── _truncate_lines tests ───────────────────────────────────────────

def test_truncate_short_unchanged():
    out = "\n".join(str(i) for i in range(50))
    assert _truncate_lines(out, 100) == out


def test_truncate_exact_boundary():
    out = "\n".join(str(i) for i in range(100))
    assert _truncate_lines(out, 100) == out


def test_truncate_folds_middle():
    lines = [f"line_{i}" for i in range(1000)]
    out = "\n".join(lines)
    result = _truncate_lines(out, 100)
    result_lines = result.splitlines()
    # fold marker present, head lines at start, tail lines at end
    assert "lines omitted" in result
    assert result_lines[0] == "line_0"
    assert result_lines[-1] == "line_999"
    assert "line_49" in result   # last head line (index 49, 0-based)
    assert "line_50" not in result  # first omitted line NOT present


def test_truncate_max_lines_3():
    lines = [f"line_{i}" for i in range(100)]
    out = "\n".join(lines)
    result = _truncate_lines(out, 3)
    # head_n = 1, tail_n = 1, fold
    assert result.startswith("line_0")
    assert result.endswith("line_99")
    assert "lines omitted" in result


def test_truncate_no_output():
    assert _truncate_lines("", 100) == ""


def test_truncate_max_lines_zero():
    out = "a\nb\nc"
    assert _truncate_lines(out, 0) == out  # max_lines <= 0 → no truncation


# ── _token_filter_output integration tests ──────────────────────────

@pytest.mark.asyncio
async def test_token_filter_no_params_passthrough():
    out = "line1\nline2\nline3"
    result, orig_path = await _token_filter_output(
        out, token_kill=False, max_lines=None
    )
    assert result == out
    assert orig_path is None  # no filter active → no original saved


@pytest.mark.asyncio
async def test_token_filter_dedup_only():
    out = "ERROR\n" * 10
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    assert "ERROR  (10 repeats)" in result
    assert orig_path is not None  # filter active → original saved


@pytest.mark.asyncio
async def test_token_filter_dedup_disabled():
    out = "ERROR\n" * 10
    result, orig_path = await _token_filter_output(
        out, token_kill=False, max_lines=None
    )
    assert result == out  # unchanged
    assert orig_path is None


@pytest.mark.asyncio
async def test_token_filter_truncate_only():
    lines = [f"L{i}" for i in range(500)]
    out = "\n".join(lines)
    result, orig_path = await _token_filter_output(
        out, token_kill=False, max_lines=50
    )
    assert "lines omitted" in result
    assert orig_path is not None
    assert "L0" in result
    assert "L499" in result


@pytest.mark.asyncio
async def test_token_filter_all_stages():
    # dedup → truncate
    lines = (
        ["ERROR: timeout"] * 100
        + ["INFO: ok"] * 50
        + ["WARN: check"] * 10
        + ["ERROR: retry"] * 5
    )
    out = "\n".join(lines)
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=3
    )
    assert "ERROR: timeout  (100 repeats)" in result  # first deduped line
    assert "ERROR: retry  (5 repeats)" in result  # last deduped line
    assert "lines omitted" in result  # 4 deduped lines → truncated
    assert orig_path is not None


@pytest.mark.asyncio
async def test_token_filter_saves_original_content():
    out = "original content here\nsecond line"
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    # Read the saved file
    import anyio
    async with await anyio.open_file(orig_path, 'r') as f:
        saved = await f.read()
    assert saved == out


@pytest.mark.asyncio
async def test_token_filter_empty_output():
    result, orig_path = await _token_filter_output(
        "", token_kill=True, max_lines=10
    )
    assert result == ""
    # When filter is active, original is saved even for empty output
    assert orig_path is not None


@pytest.mark.asyncio
async def test_token_filter_ansi_stripped_when_dedup_enabled():
    """ANSI escape codes are stripped via rich when dedup=True (merged behavior)."""
    out = "\x1B[31mHello\x1B[0m"
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    assert result == "Hello"
    assert orig_path is not None  # token_kill=True → filter active → original saved


@pytest.mark.asyncio
async def test_token_filter_ansi_left_intact_when_dedup_disabled():
    """ANSI codes are left intact when dedup=False (ANSI stripping is merged with dedup)."""
    out = "\x1B[31mHello\x1B[0m"
    result, orig_path = await _token_filter_output(
        out, token_kill=False, max_lines=None
    )
    assert result == out  # unchanged
    assert orig_path is None


@pytest.mark.asyncio
async def test_token_filter_ansi_no_ansi_unchanged():
    """token_kill=True with no ANSI codes leaves plain text unchanged."""
    out = "plain text without any escape codes\nsecond line"
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    assert result == out
    assert orig_path is not None


@pytest.mark.asyncio
async def test_token_filter_ansi_stripped_before_dedup():
    """ANSI stripping runs BEFORE dedup, so same text with different ANSI wrappers collapses."""
    out = "\x1B[31mERROR\x1B[0m\n\x1B[32mERROR\x1B[0m\n\x1B[31mERROR\x1B[0m\n\x1B[32mERROR\x1B[0m"
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    # After ANSI stripping, all 4 lines become "ERROR" -> dedup collapses to "ERROR  (4 repeats)"
    assert "ERROR  (4 repeats)" in result
    assert orig_path is not None


# ── Param validation tests ──────────────────────────────────────────


# ── RTK helper tests ─────────────────────────────────────────────────

def test_rtk_available_when_present(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bin_name = "rtk.exe" if os.name == "nt" else "rtk"
    (bin_dir / bin_name).touch()
    with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
        _rtk_binary_path.cache_clear()
        _rtk_available.cache_clear()
        assert _rtk_available() is True


def test_rtk_available_when_missing(tmp_path):
    with patch("kimi_cli.share.get_share_dir", return_value=tmp_path):
        _rtk_binary_path.cache_clear()
        _rtk_available.cache_clear()
        assert _rtk_available() is False


def test_is_known_rtk_command_known_names():
    assert _is_known_rtk_command("git") is True
    assert _is_known_rtk_command("cargo") is True
    assert _is_known_rtk_command("pytest") is True


def test_is_known_rtk_command_exe_extension():
    assert _is_known_rtk_command("git.exe") is True
    assert _is_known_rtk_command("GIT.EXE") is True


def test_is_known_rtk_command_unknown():
    assert _is_known_rtk_command("unknown-cmd") is False
    assert _is_known_rtk_command("echo") is False


@pytest.fixture
def rtk_available(tmp_path):
    """Pretend rtk exists: fake binary in a fake share/bin, available gate on.

    Yields the absolute fake rtk path for tests that still need it.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bin_name = "rtk.exe" if os.name == "nt" else "rtk"
    rtk_path = bin_dir / bin_name
    rtk_path.touch()
    _rtk_binary_path.cache_clear()
    with (
        patch("kimi_cli.share.get_share_dir", return_value=tmp_path),
        patch("kimix.tools.common._rtk_available", return_value=True),
    ):
        yield rtk_path
    _rtk_binary_path.cache_clear()


def test_rewrite_known_single_command(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk("git status", token_kill=True)
    assert changed is True
    assert rewritten == "rtk git status"


def test_rewrite_compound_command(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status && cargo test", token_kill=True
    )
    assert changed is True
    assert rewritten == "rtk git status && rtk cargo test"


def test_rewrite_already_prefixed(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "rtk git status", token_kill=True
    )
    assert changed is False
    assert rewritten == "rtk git status"


def test_rewrite_rtk_disabled(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "RTK_DISABLED=1 git status", token_kill=True
    )
    assert changed is False
    assert rewritten == "RTK_DISABLED=1 git status"


def test_rewrite_unknown_command(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "unknown-cmd arg", token_kill=True
    )
    assert changed is False
    assert rewritten == "unknown-cmd arg"


def test_rewrite_quoted_command(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        'echo "git status"', token_kill=True
    )
    assert changed is False
    assert rewritten == 'echo "git status"'


def test_rewrite_respects_token_kill_false(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status", token_kill=False
    )
    assert changed is False
    assert rewritten == "git status"


def test_rewrite_no_rtk():
    with patch("kimix.tools.common._rtk_available", return_value=False):
        rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
            "git status", token_kill=True
        )
    assert changed is False
    assert rewritten == "git status"


def test_rewrite_excludes_read_for_shell(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "read var", token_kill=True, exclude_read=True
    )
    assert changed is False
    assert rewritten == "read var"


def test_rewrite_leftmost_pipeline(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status | grep x", token_kill=True
    )
    assert changed is True
    assert rewritten == "rtk git status | grep x"


def test_rewrite_with_env_assignment(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "VAR=value git status", token_kill=True
    )
    assert changed is True
    assert rewritten == "VAR=value rtk git status"


def test_rewrite_command_substitution_unchanged(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        'echo "$(git status)"', token_kill=True
    )
    assert changed is False
    assert rewritten == 'echo "$(git status)"'


def test_rewrite_backtick_substitution_unchanged(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "echo `git status`", token_kill=True
    )
    assert changed is False
    assert rewritten == "echo `git status`"


@pytest.mark.asyncio
async def test_rewrite_rtk_rewritten_skips_dedup():
    """When rtk_rewritten=True, token_kill=True skips the local dedup pipeline."""
    out = "ERROR\n" * 10
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None, rtk_rewritten=True
    )
    assert result == out  # no dedup
    # No dedup means filter is not active unless max_lines is set
    assert orig_path is None


@pytest.mark.asyncio
async def test_token_filter_rtk_rewritten_with_max_lines_still_truncates():
    lines = [f"L{i}" for i in range(500)]
    out = "\n".join(lines)
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=50, rtk_rewritten=True
    )
    assert "lines omitted" in result
    # truncation active -> original saved
    assert orig_path is not None


@pytest.mark.asyncio
async def test_token_filter_multiline_dedup():
    out = "ERROR\n  details\n" * 5
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None, max_block_lines=2
    )
    assert "ERROR\n  details  (5 repeats)" in result
    assert orig_path is not None


@pytest.mark.asyncio
async def test_token_filter_default_still_single_line():
    out = "ERROR\n  details\n" * 5
    result, orig_path = await _token_filter_output(
        out, token_kill=True, max_lines=None
    )
    # With default max_block_lines=1, only individual lines collapse.
    assert "ERROR  (5 repeats)" in result
    assert "details  (5 repeats)" in result
    assert orig_path is not None


# ── Absolute-path rtk rewrite (no PATH reliance) ─────────────────────


def test_rewrite_semicolon_segments_absolute(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status; cargo test", token_kill=True
    )
    assert changed is True
    assert rewritten == "rtk git status; rtk cargo test"


def test_rewrite_pipe_segments_absolute(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status || cargo test", token_kill=True
    )
    assert changed is True
    assert rewritten == "rtk git status || rtk cargo test"


def test_rewrite_always_emits_bare_rtk(rtk_available):
    """The rewritten command must use the bare `rtk` prefix, never the absolute path."""
    for cmd in ("git status", "git status && cargo test", "git log | head"):
        rewritten, changed = _maybe_rewrite_shell_command_with_rtk(cmd, token_kill=True)
        if changed:
            assert rewritten.startswith("rtk ")
            assert str(rtk_available) not in rewritten


def test_rewrite_no_binary_no_rewrite(tmp_path):
    """When the available gate is True we trust it and rewrite to bare `rtk`."""
    with (
        patch("kimi_cli.share.get_share_dir", return_value=tmp_path),
        patch("kimix.tools.common._rtk_available", return_value=True),
    ):
        rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
            "git status", token_kill=True
        )
        assert changed is True
        assert rewritten == "rtk git status"


def test_rewrite_already_absolute_quoted_untouched(rtk_available):
    cmd = f'"{rtk_available}" git status'
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(cmd, token_kill=True)
    assert changed is False
    assert rewritten == cmd


def test_rewrite_already_absolute_unquoted_untouched(rtk_available):
    cmd = f"{rtk_available} git status"
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(cmd, token_kill=True)
    assert changed is False
    assert rewritten == cmd


# ── PowerShell mode (`&` call operator) ──────────────────────────────


def test_rewrite_pwsh_uses_call_operator(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status", token_kill=True, pwsh=True
    )
    assert changed is True
    assert rewritten == "& rtk git status"


def test_rewrite_pwsh_compound_command(rtk_available):
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        "git status; cargo test", token_kill=True, pwsh=True
    )
    assert changed is True
    assert rewritten == "& rtk git status; & rtk cargo test"


def test_rewrite_pwsh_already_rewritten_untouched(rtk_available):
    cmd = "& rtk git status"
    rewritten, changed = _maybe_rewrite_shell_command_with_rtk(
        cmd, token_kill=True, pwsh=True
    )
    assert changed is False
    assert rewritten == cmd
