"""Backward compatibility regression tests.

Every deprecated parameter name / behavior must keep working.
This test suite gates removal of any backward-compat shim.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

# ── Parameter aliases ─────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Bash: both 'cmd' and 'command' work."""

    def test_bash_cmd_alias(self) -> None:
        from kimix.tools.file.bash.bash_tool import BashParams
        p1 = BashParams(cmd="echo a")
        p2 = BashParams(command="echo a")
        assert p1.cmd == p2.cmd

    def test_run_command_alias(self) -> None:
        from kimix.tools.file.run import RunParams
        p1 = RunParams(command="git status")
        p2 = RunParams(cmd="git status")
        assert p1.command == p2.command

    def test_taskoutput_block_alias(self) -> None:
        from kimix.tools.background import TaskOutputParams
        p1 = TaskOutputParams(task_id="t1", block=True)
        p2 = TaskOutputParams(task_id="t1", wait=True)
        assert p1.wait == p2.wait

    def test_bash_token_kill_alias(self) -> None:
        from kimix.tools.file.bash.bash_tool import BashParams
        p1 = BashParams(cmd="test", token_kill=False)
        p2 = BashParams(cmd="test", deduplicate_output=False)
        assert p1.deduplicate_output == p2.deduplicate_output

    def test_python_interactive_legacy_bool(self) -> None:
        from kimix.tools.py import Params as PythonParams
        params = PythonParams(interactive=True)
        assert params.mode == "interactive"

    def test_bash_interactive_legacy_bool(self) -> None:
        from kimix.tools.file.bash.bash_tool import BashParams
        params = BashParams(cmd="ls", interactive=True)
        assert params.mode == "interactive"

    def test_powershell_interactive_legacy_bool(self) -> None:
        from kimix.tools.file.bash.pwsh_tool import PowershellParams
        params = PowershellParams(cmd="Get-Date", interactive=True)
        assert params.mode == "interactive"

    def test_python_run_alias(self) -> None:
        from kimix.tools.py import Params as PythonParams
        params = PythonParams(code="print(1)", mode="run")
        assert params.mode == "execute"

    def test_python_background_alias(self) -> None:
        from kimix.tools.py import Params as PythonParams
        params = PythonParams(code="print(1)", mode="background")
        assert params.mode == "send"

    def test_bash_run_alias(self) -> None:
        from kimix.tools.file.bash.bash_tool import BashParams
        params = BashParams(cmd="echo hi", mode="run")
        assert params.mode == "execute"

    def test_bash_background_alias(self) -> None:
        from kimix.tools.file.bash.bash_tool import BashParams
        params = BashParams(cmd="echo hi", mode="background")
        assert params.mode == "send"

    def test_powershell_run_alias(self) -> None:
        from kimix.tools.file.bash.pwsh_tool import PowershellParams
        params = PowershellParams(cmd="Get-Date", mode="run")
        assert params.mode == "execute"

    def test_powershell_background_alias(self) -> None:
        from kimix.tools.file.bash.pwsh_tool import PowershellParams
        params = PowershellParams(cmd="Get-Date", mode="background")
        assert params.mode == "send"

    def test_editfile_single_edit_auto_wrap(self) -> None:
        from kimi_cli.tools.file.replace import Params as EditFileParams
        params = EditFileParams(path="f.txt", edit={"old": "a", "new": "b"})
        assert isinstance(params.edit, list)
        assert len(params.edit) == 1

    def test_todolist_append_mode_still_default(self) -> None:
        from kimi_cli.tools.todo import Params as TodoListParams, Todo
        params = TodoListParams(todos=[Todo(title="task", status="pending")])
        assert params.mode == "append"

    def test_readfile_path_alias_files(self) -> None:
        """'files' and 'paths' aliases should still work via field_aliases."""
        # These are handled by the tool's field_aliases, not Pydantic
        from kimi_cli.tools.file.read import ReadFile
        assert "files" in ReadFile.field_aliases
        assert ReadFile.field_aliases["files"] == "path"

    def test_grep_cli_aliases_in_field_aliases(self) -> None:
        """-A, -B, -C, -n, -i aliases must be registered."""
        from kimi_cli.tools.file.grep_local import Grep
        assert "-A" in Grep.field_aliases
        assert "-B" in Grep.field_aliases
        assert "-C" in Grep.field_aliases
        assert "-n" in Grep.field_aliases
        assert "-i" in Grep.field_aliases


# ── Pydantic aliases (pwsh-style) ──────────────────────────────────────────
# Every Params model that gained an explicit pydantic alias must accept BOTH
# spellings (canonical field name and alias) via populate_by_name=True.


class TestPydanticAliases:
    """Both canonical and alias spellings validate to the same canonical field."""

    def test_readfile_file_path_alias(self) -> None:
        from kimi_cli.tools.file.read import Params as ReadFileParams
        p1 = ReadFileParams.model_validate({"path": "a.py"})
        p2 = ReadFileParams.model_validate({"file_path": "a.py"})
        assert p1.path == p2.path == "a.py"

    def test_writefile_file_path_and_text_aliases(self) -> None:
        from kimi_cli.tools.file.write import Params as WriteFileParams
        p1 = WriteFileParams.model_validate({"path": "a", "content": "b"})
        p2 = WriteFileParams.model_validate({"file_path": "a", "text": "b"})
        assert p1.path == p2.path == "a"
        assert p1.content == p2.content == "b"

    def test_editfile_file_path_edits_old_new_string_aliases(self) -> None:
        from kimi_cli.tools.file.replace import Params as EditFileParams
        p1 = EditFileParams.model_validate(
            {"path": "f.txt", "edit": [{"old": "a", "new": "b"}]}
        )
        p2 = EditFileParams.model_validate(
            {"file_path": "f.txt", "edits": [{"old_string": "a", "new_string": "b"}]}
        )
        assert p1.path == p2.path == "f.txt"
        assert isinstance(p1.edit, list) and isinstance(p2.edit, list)
        assert p1.edit[0].old == p2.edit[0].old == "a"
        assert p1.edit[0].new == p2.edit[0].new == "b"

    def test_glob_path_alias(self) -> None:
        from kimi_cli.tools.file.glob import Params as GlobParams
        p1 = GlobParams.model_validate({"pattern": "*.py", "directory": "src"})
        p2 = GlobParams.model_validate({"pattern": "*.py", "path": "src"})
        assert p1.directory == p2.directory == "src"

    def test_python_source_code_alias(self) -> None:
        from kimix.tools.py import Params as PythonParams
        p1 = PythonParams.model_validate({"code": "print(1)"})
        p2 = PythonParams.model_validate({"source_code": "print(1)"})
        assert p1.code == p2.code == "print(1)"

    def test_python_deduplicate_output_canonical_accepted(self) -> None:
        """Regression: Params declared alias 'token_kill' without
        populate_by_name=True, so the canonical 'deduplicate_output' was
        silently ignored on direct validation."""
        from kimix.tools.py import Params as PythonParams
        p1 = PythonParams.model_validate({"code": "x", "token_kill": False})
        p2 = PythonParams.model_validate({"code": "x", "deduplicate_output": False})
        assert p1.deduplicate_output is False
        assert p2.deduplicate_output is False

    def test_taskoutput_wait_canonical_accepted(self) -> None:
        """Regression: TaskOutputParams declared alias 'block' without
        populate_by_name=True, so the canonical 'wait' was silently ignored."""
        from kimix.tools.background import TaskOutputParams
        p1 = TaskOutputParams.model_validate({"task_id": "t1", "block": False})
        p2 = TaskOutputParams.model_validate({"task_id": "t1", "wait": False})
        assert p1.wait is False
        assert p2.wait is False

    def test_agent_task_and_session_aliases(self) -> None:
        from kimix.tools.agent import SubAgentParams
        p1 = SubAgentParams.model_validate({"prompt": "do x", "session_id": "s1"})
        p2 = SubAgentParams.model_validate({"task": "do x", "session": "s1"})
        assert p1.prompt == p2.prompt == "do x"
        assert p1.session_id == p2.session_id == "s1"

    def test_agentclose_session_alias(self) -> None:
        from kimix.tools.agent import AgentCloseParams
        p1 = AgentCloseParams.model_validate({"session_id": "s1"})
        p2 = AgentCloseParams.model_validate({"session": "s1"})
        assert p1.session_id == p2.session_id == "s1"

    def test_writeplan_text_alias(self) -> None:
        from kimix.tools.note import WritePlanParams
        p1 = WritePlanParams.model_validate({"content": "plan"})
        p2 = WritePlanParams.model_validate({"text": "plan"})
        assert p1.content == p2.content == "plan"

    def test_editplan_edits_old_new_string_aliases(self) -> None:
        from kimix.tools.note import EditPlanParams
        p1 = EditPlanParams.model_validate({"edit": [{"old": "a", "new": "b"}]})
        p2 = EditPlanParams.model_validate(
            {"edits": [{"old_string": "a", "new_string": "b"}]}
        )
        assert isinstance(p1.edit, list) and isinstance(p2.edit, list)
        assert p1.edit[0].old == p2.edit[0].old == "a"
        assert p1.edit[0].new == p2.edit[0].new == "b"

    def test_todolist_items_alias(self) -> None:
        from kimi_cli.tools.todo import Params as TodoListParams
        p1 = TodoListParams.model_validate(
            {"todos": [{"title": "task", "status": "pending"}]}
        )
        p2 = TodoListParams.model_validate(
            {"items": [{"title": "task", "status": "pending"}]}
        )
        assert isinstance(p1.todos, list) and isinstance(p2.todos, list)
        assert p1.todos[0].title == p2.todos[0].title == "task"

    def test_todolist_extra_forbid_still_active(self) -> None:
        """populate_by_name must not weaken extra='forbid'."""
        from kimi_cli.tools.todo import Params as TodoListParams
        with pytest.raises(ValidationError):
            TodoListParams.model_validate({"unknown_key": 1})


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
