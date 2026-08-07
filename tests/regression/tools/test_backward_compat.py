"""Backward compatibility regression tests.

Every deprecated parameter name / behavior must keep working.
This test suite gates removal of any backward-compat shim.
"""
from __future__ import annotations

# ── Parameter aliases ─────────────────────────────────────────────────────


# ── Pydantic aliases (pwsh-style) ──────────────────────────────────────────
# Every Params model that gained an explicit pydantic alias must accept BOTH
# spellings (canonical field name and alias) via populate_by_name=True.


# ── format_tool_args alias-aware display ────────────────────────────────────


class TestFormatToolArgsAliases:
    """Alias-spelled JSON args must produce the same one-line summary as the
    canonical spelling (the LLM's raw arguments reach the display pre-repair).

    The generic ``format_tool_args`` has no per-tool cases: every argument
    renders as ``key:value`` (canonical key, space-separated pairs), so new
    tools work unchanged."""

    def test_run_cmd_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"command": "ls"}')
        alias = format_tool_args('{"cmd": "ls"}')
        assert canonical == alias == "command:ls"

    def test_shell_token_kill_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"deduplicate_output": false}')
        alias = format_tool_args('{"token_kill": false}')
        assert canonical == alias == "deduplicate_output:False"

    def test_python_source_code_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"code": "print(1)"}')
        alias = format_tool_args('{"source_code": "print(1)"}')
        assert canonical == alias == "code:print(1)"

    def test_taskoutput_wait_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"wait": false}')
        alias = format_tool_args('{"block": false}')
        assert canonical == alias == "wait:False"

    def test_todolist_items_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args(
            '{"todos": [{"title": "a", "status": "pending"}]}'
        )
        alias = format_tool_args(
            '{"items": [{"title": "a", "status": "pending"}]}'
        )
        assert canonical == alias
        assert canonical.startswith("todos:")

    def test_readfile_file_path_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"path": "x.py"}')
        alias = format_tool_args('{"file_path": "x.py"}')
        assert canonical == alias == "path:x.py"

    def test_editfile_aliases_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args(
            '{"path": "x.py", "edit": [{"old": "a", "new": "b"}]}'
        )
        alias = format_tool_args(
            '{"file_path": "x.py", "edits": [{"old": "a", "new": "b"}]}'
        )
        assert canonical == alias
        assert canonical.startswith("path:x.py edit:")

    def test_writefile_aliases_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"path": "x.py", "content": "hi"}')
        alias = format_tool_args('{"file_path": "x.py", "text": "hi"}')
        assert canonical == alias == "path:x.py content:hi"

    def test_glob_path_alias_display(self) -> None:
        """Glob's ``path`` alias (for ``directory``) cannot be canonicalized
        tool-specifically anymore: the generic formatter is per-tool agnostic
        and ``path`` means ``path`` for file tools.  The value still displays
        — under the generic ``path:`` key — and execution-side repair maps
        it to ``directory`` as before."""
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"pattern": "*.py", "directory": "src"}')
        alias = format_tool_args('{"pattern": "*.py", "path": "src"}')
        assert canonical == "pattern:*.py directory:src"
        assert alias == "pattern:*.py path:src"

    def test_grep_context_aliases_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args(
            '{"pattern": "p", "before_context": 2, "after_context": 3, '
            '"context": 1, "line_number": false, "ignore_case": true}',
        )
        alias = format_tool_args(
            '{"pattern": "p", "-B": 2, "-A": 3, "-C": 1, "-n": false, "-i": true}',
        )
        assert canonical == alias
        assert "before_context:2" in canonical
        assert "after_context:3" in canonical
        assert "context:1" in canonical
        assert "line_number:False" in canonical
        assert "ignore_case:True" in canonical

    def test_agent_task_session_aliases_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"prompt": "do x", "session_id": "s1"}')
        alias = format_tool_args('{"task": "do x", "session": "s1"}')
        assert canonical == alias == "prompt:do x session_id:s1"

    def test_agentclose_session_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"session_id": "s1"}')
        alias = format_tool_args('{"session": "s1"}')
        assert canonical == alias == "session_id:s1"

    def test_writeplan_text_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"content": "plan"}')
        alias = format_tool_args('{"text": "plan"}')
        assert canonical == alias == "content:plan"

    def test_editplan_edits_alias_display(self) -> None:
        from kimix.ui.stream import format_tool_args
        canonical = format_tool_args('{"edit": [{"old": "a", "new": "b"}]}')
        alias = format_tool_args('{"edits": [{"old": "a", "new": "b"}]}')
        assert canonical == alias
        assert canonical.startswith("edit:")
