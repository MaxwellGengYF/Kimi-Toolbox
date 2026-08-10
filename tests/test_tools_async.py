"""Comprehensive async/await tests for tools using BackgroundStream and ProcessTask."""

import asyncio
import queue
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimi_agent_sdk import ToolError, ToolOk
from kimix.tools.background import TaskOutput, TaskOutputParams
from kimix.tools.background.utils import (
    BackgroundStream,
    add_task,
    discard_all_tasks,
    get_all_tasks,
    remove_task_id,
)
from kimix.tools.py import Params as PyParams
from kimix.tools.py import Python


@pytest.fixture
def mock_session(tmp_path: Path) -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    session.dir = tmp_path / ".kimi" / "sessions" / "test"
    session.dir.mkdir(parents=True, exist_ok=True)
    return session


@pytest.fixture(autouse=True)
async def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    await discard_all_tasks(mock_session)


@pytest.fixture(autouse=True)
def patch_find_bash() -> Any:
    with patch("kimix.tools.file.bash.bash_tool.find_bash", return_value=None):
        yield


# ---------------------------------------------------------------------------
# TaskList tool (via TaskOutput with task_id=None / action='list')
# ---------------------------------------------------------------------------
class TestTaskList:
    async def test_empty(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        result = await tool(TaskOutputParams(task_id=None))
        assert "No running tasks" in str(result.output)

    async def test_lists_tasks(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("hello")

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)
        result = await tool(TaskOutputParams(task_id=None))
        await stream.wait()
        assert "run_test" in str(result.output)

    async def test_action_list(self, mock_session: MagicMock) -> None:
        """action='list' should work like task_id=None."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("hello")

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)
        result = await tool(TaskOutputParams(action="list"))
        await stream.wait()
        assert "run_test" in str(result.output)
        # Should have structured extras
        assert hasattr(result, 'extras') and result.extras is not None


# ---------------------------------------------------------------------------
# TaskOutput tool
# ---------------------------------------------------------------------------
class TestTaskOutput:
    async def test_not_found(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        result = await tool(TaskOutputParams(task_id="missing"))
        assert "No running task" in str(result.message)

    async def test_wait_and_get_output(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("output_line")
            time.sleep(0.05)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        # Use new wait=True (was block=True)
        result = await tool(TaskOutputParams(task_id="run_test", wait=True, timeout=5))
        assert "output_line" in str(result.output)
        assert "run_test" not in get_all_tasks(mock_session)

    async def test_wait_alias_block_still_works(self, mock_session: MagicMock) -> None:
        """block=True (old name) should still work via alias."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("block_alias_works")
            time.sleep(0.05)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", block=True, timeout=5))
        assert "block_alias_works" in str(result.output)

    async def test_action_get(self, mock_session: MagicMock) -> None:
        """action='get' should work like default."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("action_get_works")
            time.sleep(0.05)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", action="get", wait=True, timeout=5))
        assert "action_get_works" in str(result.output)

    async def test_kill_running_task(self, mock_session: MagicMock) -> None:
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", block=False, kill=True))
        await stream.wait()
        assert "run_test" not in get_all_tasks(mock_session)

    async def test_action_kill(self, mock_session: MagicMock) -> None:
        """action='kill' should stop the task and return output."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("kill_output")
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", action="kill"))
        await stream.wait()
        assert "run_test" not in get_all_tasks(mock_session)
        assert "kill_output" in str(result.output)

    async def test_action_kill_missing_task_id(self, mock_session: MagicMock) -> None:
        """action='kill' without task_id should return error."""
        tool = TaskOutput(session=mock_session)
        result = await tool(TaskOutputParams(action="kill"))
        assert isinstance(result, ToolError)
        assert "task_id" in result.message.lower()

    async def test_export_to_file(self, mock_session: MagicMock, tmp_path: Path) -> None:
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("file_content")

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        out_path = tmp_path / "out.txt"
        result = await tool(TaskOutputParams(task_id="run_test", wait=True, timeout=5, output_path=str(out_path)))
        await stream.wait()
        assert out_path.exists()
        assert "file_content" in out_path.read_text(encoding="utf-8")
        assert "exported to file" in str(result.output)

    async def test_running_task_returns_inline_output(self, mock_session: MagicMock) -> None:
        """A running task should return its accumulated output inline,
        not unconditionally export it to a temporary file.
        """
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()

        def worker(q: queue.Queue[str]) -> None:
            q.put("hello_running")
            time.sleep(10)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", wait=False))
        assert "hello_running" in str(result.output)
        assert "exported to file" not in str(result.output)

        # cleanup
        await stream.stop()
        await stream.wait()

    async def test_rtk_truncated_output_is_exported(
        self, mock_session: MagicMock
    ) -> None:
        """TaskOutput exports the full stream when rtk left fold markers."""
        tool = TaskOutput(session=mock_session)
        stream = BackgroundStream()
        rtk_output = (
            "3 matches in 2 files:\n\n"
            "src/a.py:1:match\n"
            "  +5 more in src/b.py [see remaining: tail -n +2 /tmp/rtk.log]\n"
        )

        def worker(q: queue.Queue[str]) -> None:
            q.put(rtk_output)
            time.sleep(0.05)

        await stream.start(worker, stop_function=lambda: None)
        add_task(mock_session, "run_test", stream)

        result = await tool(TaskOutputParams(task_id="run_test", wait=True, timeout=5))
        await stream.wait()
        assert "rtk output exported to:" in str(result.output)
        # The full rtk stream should be saved in the referenced temp file.
        import re as _re
        m = _re.search(r"rtk output exported to: (.+?)\]", str(result.output))
        assert m is not None
        saved_path = m.group(1)
        assert Path(saved_path).read_text(encoding="utf-8") == rtk_output


# ---------------------------------------------------------------------------
# Python tool — Params validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Python tool — execution tests
# ---------------------------------------------------------------------------
class TestPython:
    async def test_foreground_success(self, mock_session: MagicMock) -> None:
        tool = Python(session=mock_session)
        params = PyParams(code="print('hello_py')", timeout=10)
        result = await tool(params)
        # Output is now a structured block containing the script output
        output_str = str(result.output)
        assert "hello_py" in output_str
        assert "status: completed" in output_str
        assert "task_id:" in output_str

    async def test_foreground_failure(self, mock_session: MagicMock) -> None:
        tool = Python(session=mock_session)
        params = PyParams(code="import sys; sys.exit(1)", timeout=10)
        result = await tool(params)
        output_str = str(result.output)
        assert "status: completed" in output_str
        assert "exit_code:" in output_str
        # Should be a ToolError
        assert isinstance(result, ToolError)

    async def test_foreground_timeout(self, mock_session: MagicMock) -> None:
        tool = Python(session=mock_session)
        params = PyParams(code="import time; time.sleep(100)", timeout=3)
        result = await tool(params)
        assert "timeout" in str(result.message).lower() or "background" in str(result.message).lower()
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_dest_export(self, mock_session: MagicMock, tmp_path: Path) -> None:
        tool = Python(session=mock_session)
        dest = tmp_path / "py_out.txt"
        params = PyParams(code="print('dest_out')", timeout=10, output_path=str(dest))
        result = await tool(params)
        assert dest.exists()
        assert "dest_out" in dest.read_text(encoding="utf-8")
        assert "exported to" in str(result.output)

    async def test_inactivity_timeout_returns_background_error(
        self, mock_session: MagicMock
    ) -> None:
        with patch(
            "kimix.tools.background.utils.DEFAULT_INACTIVITY_TIMEOUT", 2.0
        ):
            tool = Python(session=mock_session)
            params = PyParams(code="import time; time.sleep(120)", timeout=90)
            result = await tool(params)
            assert isinstance(result, ToolError)
            assert result.brief == "Timeout"
            assert "Running in background" in result.message
            assert "task_id" in result.message

    # File mode tests -------------------------------------------------------

    async def test_file_success(self, mock_session: MagicMock, tmp_path: Path) -> None:
        """Run an existing .py file directly."""
        py_file = tmp_path / "hello.py"
        py_file.write_text("print('hello_from_file')", encoding='utf-8')
        tool = Python(session=mock_session)
        params = PyParams(code=str(py_file), timeout=10)
        result = await tool(params)
        output_str = str(result.output)
        assert "hello_from_file" in output_str
        assert "task_id:" in output_str
        assert "status: completed" in output_str

    async def test_file_failure(self, mock_session: MagicMock, tmp_path: Path) -> None:
        """Run an existing .py file that exits with error."""
        py_file = tmp_path / "fail.py"
        py_file.write_text("import sys; sys.exit(42)", encoding='utf-8')
        tool = Python(session=mock_session)
        params = PyParams(code=str(py_file), timeout=10)
        result = await tool(params)
        # Should be a ToolError with structured output
        assert isinstance(result, ToolError)
        output_str = str(result.output)
        assert "status: completed" in output_str

    async def test_file_with_output_path(self, mock_session: MagicMock, tmp_path: Path) -> None:
        """Run a .py file and export output to a destination file."""
        py_file = tmp_path / "greet.py"
        py_file.write_text("print('file_dest_out')", encoding='utf-8')
        dest = tmp_path / "file_out.txt"
        tool = Python(session=mock_session)
        params = PyParams(code=str(py_file), timeout=10, output_path=str(dest))
        result = await tool(params)
        assert dest.exists()
        assert "file_dest_out" in dest.read_text(encoding="utf-8")
        assert "exported to" in str(result.output)

    async def test_file_not_found_treated_as_code(self, mock_session: MagicMock) -> None:
        """A .py path that does not exist should be treated as inline code."""
        tool = Python(session=mock_session)
        # "nonexistent.py" doesn't exist, so it's treated as inline code.
        # It compiles fine (attribute access ``nonexistent.py``) and then
        # fails at runtime with NameError.
        params = PyParams(code="nonexistent.py", timeout=10)
        result = await tool(params)
        assert isinstance(result, ToolError)
        output_str = str(result.output)
        # Should reference structured output block
        assert "task_id:" in output_str

    async def test_broken_inline_code_fails_fast_syntax_check(
        self, mock_session: MagicMock
    ) -> None:
        """Genuinely broken inline code is caught by the compile pre-check
        before any subprocess spawn, so the response has no task_id."""
        tool = Python(session=mock_session)
        params = PyParams(code="def broken(:", timeout=10)
        result = await tool(params)
        assert isinstance(result, ToolError)
        assert result.brief == "Syntax error"
        assert "Syntax error detected before execution" in str(result.message)
        assert "task_id:" not in str(result.output)

    # Interactive mode tests ------------------------------------------------

    async def test_interactive_start_no_code(self, mock_session: MagicMock) -> None:
        """interactive=True, code=""  -> returns ToolOk with task_id."""
        tool = Python(session=mock_session)
        params = PyParams(code="", interactive=True, timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "Interactive Python started" in result.message
        assert "task_id" in result.message
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_interactive_start_with_code(self, mock_session: MagicMock) -> None:
        """interactive=True, code="print('hi')" -> returns ToolOk with task_id."""
        tool = Python(session=mock_session)
        params = PyParams(code="print('hi')", interactive=True, timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "Interactive Python started" in result.message
        assert "task_id" in result.message
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_interactive_with_wait_pattern(self, mock_session: MagicMock) -> None:
        """interactive=True, wait_for_pattern waits and matches."""
        tool = Python(session=mock_session)
        # The Python interactive REPL prints '>>> ' as a prompt
        params = PyParams(
            code="print('ready')",
            interactive=True,
            wait_for_pattern=r">>>",
            timeout=10,
        )
        result = await tool(params)
        assert isinstance(result, ToolOk)
        output_str = str(result.output)
        assert "task_id:" in output_str
        assert "wait_matched:" in output_str
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_continue_session(self, mock_session: MagicMock) -> None:
        """Start interactive, then call again with task_id and new code."""
        tool = Python(session=mock_session)
        # Start interactive
        start_params = PyParams(code="", interactive=True, timeout=5)
        start_result = await tool(start_params)
        assert isinstance(start_result, ToolOk)
        assert "task_id" in start_result.message

        # Extract task_id from the message
        msg = start_result.message
        task_id = msg.split("task_id: `")[1].split("`")[0]

        # Continue session: send a simple print
        continue_params = PyParams(code="print('continue_test')", task_id=task_id, timeout=10, wait_for_pattern=r"continue_test")
        continue_result = await tool(continue_params)
        assert isinstance(continue_result, ToolOk)
        output_str = str(continue_result.output)
        assert "continue_test" in output_str or "task_id:" in output_str

        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_continue_session_not_found(self, mock_session: MagicMock) -> None:
        """task_id="nonexistent" -> ToolError."""
        tool = Python(session=mock_session)
        params = PyParams(code="print('test')", task_id="nonexistent", timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolError)
        assert "not found" in result.message.lower()

    async def test_max_lines_truncation(self, mock_session: MagicMock) -> None:
        """max_lines=10 on long output -> output is truncated."""
        tool = Python(session=mock_session)
        # Generate 50 lines of output
        code = "for i in range(50): print(f'line_{i}')"
        params = PyParams(code=code, timeout=10, max_lines=10)
        result = await tool(params)
        output_str = str(result.output)
        assert "output_truncated:" in output_str
        # Should have the fold marker (10 lines max -> fold)
        assert "omitted" in output_str or "lines" in output_str
        # Should have at most 15 lines (10 max + 5 metadata)
        lines = output_str.splitlines()
        # Count only output lines (after "output: |")
        output_section = False
        output_line_count = 0
        for line in lines:
            if "output: |" in line:
                output_section = True
                continue
            if output_section:
                if line.startswith("  "):
                    output_line_count += 1
                else:
                    break
        assert output_line_count <= 15, f"Expected <=15 output lines, got {output_line_count}"

    # Mode tests -----------------------------------------------------------

    async def test_mode_background(self, mock_session: MagicMock) -> None:
        """mode='background' (alias for 'send') starts a task and returns immediately."""
        tool = Python(session=mock_session)
        params = PyParams(code="import time; time.sleep(100)", mode="background", timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "Background task started" in result.brief
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_mode_send_new_name(self, mock_session: MagicMock) -> None:
        """mode='send' starts a task and returns immediately."""
        tool = Python(session=mock_session)
        params = PyParams(code="import time; time.sleep(100)", mode="send", timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "Background task started" in result.brief
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_mode_interactive(self, mock_session: MagicMock) -> None:
        """mode='interactive' starts a persistent REPL."""
        tool = Python(session=mock_session)
        params = PyParams(code="", mode="interactive", timeout=5)
        result = await tool(params)
        assert isinstance(result, ToolOk)
        assert "Interactive Python started" in result.message
        # cleanup
        for tid in list(get_all_tasks(mock_session).keys()):
            remove_task_id(mock_session, tid)

    async def test_mode_execute_default(self, mock_session: MagicMock) -> None:
        """mode='execute' (default) executes code and waits."""
        tool = Python(session=mock_session)
        params = PyParams(code="print('run_mode_test')", mode="execute", timeout=10)
        result = await tool(params)
        output_str = str(result.output)
        assert "run_mode_test" in output_str
        assert "status: completed" in output_str

    async def test_mode_run_alias_still_works(self, mock_session: MagicMock) -> None:
        """mode='run' (alias for 'execute') still works."""
        tool = Python(session=mock_session)
        params = PyParams(code="print('run_alias_test')", mode="run", timeout=10)
        result = await tool(params)
        output_str = str(result.output)
        assert "run_alias_test" in output_str
        assert "status: completed" in output_str

    async def test_process_output_exports_rtk_original_when_truncated(
        self, mock_session: MagicMock
    ) -> None:
        """If captured output contains rtk fold markers, the full stream is saved."""
        tool = Python(session=mock_session)
        output = (
            "2 matches in 2 files:\n\n"
            "src/a.py:1:match\n"
            "  +5 more in src/b.py [see remaining: tail -n +2 /tmp/rtk.log]\n"
        )
        params = PyParams(code="pass", deduplicate_output=False, max_lines=None)
        processed, output_path, output_truncated, original_path = await tool._process_output(
            params, output
        )
        assert original_path is not None
        saved = Path(original_path).read_text(encoding="utf-8")
        assert saved == output


class TestAsyncIntegration:
    async def test_background_stream_awaited_methods(self) -> None:
        stream = BackgroundStream()
        # All these are declared async; calling them with await should work even
        # if the underlying implementation is synchronous.
        assert await stream.is_started() is False
        assert await stream.is_stopped() is False
        assert await stream.thread_is_alive() is False
        assert await stream.success() is False
        assert await stream.get_output() == ""
        assert await stream.get_queue() is None

    async def test_process_task_all_async_methods_awaited(self, mock_session: MagicMock) -> None:
        from kimix.tools.common import ProcessTask

        task = ProcessTask(sys.executable, ["-c", "print('await_test')"])
        tid = await task.start(mock_session, kind="run", name="await")
        assert await task.thread_is_alive() is True
        await task.wait(timeout=5)
        assert await task.thread_is_alive() is False
        assert await task.stream.success() is True
        output = await task.stream.pop_output()
        assert "await_test" in output
        remove_task_id(mock_session, tid)

    async def test_concurrent_task_outputs(self, mock_session: MagicMock) -> None:
        stream1 = BackgroundStream()
        stream2 = BackgroundStream()

        def w1(q: queue.Queue[str]) -> None:
            q.put("a")

        def w2(q: queue.Queue[str]) -> None:
            q.put("b")

        await stream1.start(w1, stop_function=lambda: None)
        await stream2.start(w2, stop_function=lambda: None)
        add_task(mock_session, "t1", stream1)
        add_task(mock_session, "t2", stream2)

        out1, out2 = await asyncio.gather(
            stream1.get_output(),
            stream2.get_output(),
        )
        assert "a" in out1
        assert "b" in out2

        await stream1.wait()
        await stream2.wait()
        remove_task_id(mock_session, "t1")
        remove_task_id(mock_session, "t2")
