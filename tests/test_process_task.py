"""Comprehensive tests for ProcessTask."""

import asyncio
import queue
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimix.tools.background.utils import _pop_task_data
from kimix.tools.common import ProcessTask


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    return session


@pytest.fixture(autouse=True)
def cleanup_task_data(mock_session: MagicMock) -> Any:
    yield
    _pop_task_data(mock_session)


# ---------------------------------------------------------------------------
# Construction / __init__
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _run_process_bg
# ---------------------------------------------------------------------------
async def test_run_process_bg_success() -> None:
    task = ProcessTask(sys.executable, ["-c", "print('hello_world')"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    assert result[1] == 0

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "hello_world" in output


async def test_run_process_bg_stderr() -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; sys.stderr.write('err_msg\\n')"],
    )
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    assert result[1] == 0

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "[stderr] err_msg" in output


async def test_run_process_bg_nonzero_exit() -> None:
    task = ProcessTask(sys.executable, ["-c", "import sys; sys.exit(42)"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is False
    assert result[1] == 42

    messages = []
    while True:
        try:
            messages.append(q.get_nowait())
        except queue.Empty:
            break
    assert any("exited with code 42" in m for m in messages)


async def test_run_process_bg_stop_event_before_start() -> None:
    task = ProcessTask(sys.executable, ["-c", "print(1)"])
    task._stop_event.set()
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is False
    assert result[1] is None


async def test_run_process_bg_stop_event_during_run() -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.2)
    await task._stop_function()
    await task._stop_function()
    result = await asyncio.wait_for(bg, timeout=5)
    assert result[0] is False
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "stopped by user" in output


async def test_run_process_bg_exception_on_popen() -> None:
    task = ProcessTask("this_should_not_exist_command_12345")
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is False
    assert result[1] is None
    msg = q.get_nowait()
    assert msg.startswith("\n[Error:")


async def test_run_process_bg_popen_raises_oserror() -> None:
    task = ProcessTask(sys.executable)
    with patch("asyncio.create_subprocess_exec", side_effect=OSError("boom")):
        q: queue.Queue[str] = queue.Queue()
        result = await task._run_process_bg(q)
        assert result[0] is False
        assert result[1] is None
        msg = q.get_nowait()
        assert "boom" in msg


# ---------------------------------------------------------------------------
# _stop_function
# ---------------------------------------------------------------------------
async def test_stop_function_terminates_running_process() -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.2)
    assert task._process_ref is not None
    assert task._process_ref.returncode is None

    await task._stop_function()
    result = await asyncio.wait_for(bg, timeout=5)
    assert result[0] is False


# ---------------------------------------------------------------------------
# _input_function
# ---------------------------------------------------------------------------
async def test_input_function_writes_to_stdin() -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; line=sys.stdin.readline(); print('echo', line.strip())"],
    )
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.1)
    success = await task._input_function("hello\n")
    assert success is True

    result = await asyncio.wait_for(bg, timeout=5)
    assert result[0] is True
    assert result[1] == 0

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "echo hello" in output


async def test_input_function_returns_false_when_stopped() -> None:
    task = ProcessTask(sys.executable, ["-c", "print(1)"])
    task._stop_event.set()
    success = await task._input_function("data")
    assert success is False


async def test_input_function_returns_false_when_no_process() -> None:
    task = ProcessTask(sys.executable)
    task._stop_event.set()
    success = await task._input_function("data")
    assert success is False


# ---------------------------------------------------------------------------
# start / public API integration
# ---------------------------------------------------------------------------
async def test_start_returns_task_id(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('ok')"])
    tid = await task.start(mock_session, kind="run", name="test")
    assert tid is not None
    assert tid.startswith("run_test")
    assert task.task_id == tid


async def test_start_default_name(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable)
    tid = await task.start(mock_session, kind="cmd")
    assert tid == "cmd"
    assert task.task_id == tid


async def test_start_creates_stream(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('ok')"])
    await task.start(mock_session, kind="run")
    assert task.stream is not None
    assert await task.stream.is_started() is True


async def test_wait_completes(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('done')"])
    await task.start(mock_session, kind="run")
    await task.wait(timeout=5)
    assert await task.stream.thread_is_alive() is False
    output = await task.stream.get_output()
    assert "done" in output


async def test_wait_with_monitor_completes(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "print('done')"])
    await task.start(mock_session, kind="run")
    completed, elapsed, inactivity_timed_out = await task.wait_with_monitor(timeout=5.0)
    assert completed is True
    assert inactivity_timed_out is False
    assert elapsed < 5.0


async def test_wait_with_monitor_inactivity_timeout(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(120)"])
    await task.start(mock_session, kind="run")
    completed, elapsed, inactivity_timed_out = await task.wait_with_monitor(
        timeout=130.0, inactivity_timeout=2.0
    )
    assert completed is False
    assert inactivity_timed_out is True
    assert elapsed < 5.0
    assert await task.thread_is_alive() is True
    await task.stop()
    await task.wait(timeout=2)


async def test_wait_with_monitor_no_monitor_for_short_timeout(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(120)"])
    await task.start(mock_session, kind="run")
    start = asyncio.get_event_loop().time()
    completed, elapsed, inactivity_timed_out = await task.wait_with_monitor(
        timeout=1.0, inactivity_timeout=60.0
    )
    end = asyncio.get_event_loop().time()
    assert completed is False
    assert inactivity_timed_out is False
    assert 0.9 <= elapsed <= 2.0
    assert 0.9 <= (end - start) <= 2.0
    assert await task.thread_is_alive() is True
    await task.stop()
    await task.wait(timeout=2)


async def test_thread_is_alive_while_running(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(0.5)"])
    await task.start(mock_session, kind="run")
    assert await task.thread_is_alive() is True
    await task.wait(timeout=5)
    assert await task.thread_is_alive() is False


async def test_stop_via_public_api(mock_session: MagicMock) -> None:
    task = ProcessTask(sys.executable, ["-c", "import time; time.sleep(10)"])
    await task.start(mock_session, kind="run")
    await asyncio.sleep(0.1)
    assert await task.thread_is_alive() is True
    await task.stop()
    await task.wait(timeout=2)
    assert await task.stream.is_stopped() is True


async def test_input_via_public_api(mock_session: MagicMock) -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import sys; line=sys.stdin.readline(); print('got', line.strip())"],
    )
    await task.start(mock_session, kind="run")
    await asyncio.sleep(0.1)
    result = await task.input("hello\n")
    assert result is True
    await task.wait(timeout=5)
    output = await task.stream.get_output()
    assert "got hello" in output


async def test_input_returns_false_when_not_started() -> None:
    task = ProcessTask(sys.executable)
    result = await task.input("data")
    assert result is False


async def test_task_id_none_before_start() -> None:
    task = ProcessTask(sys.executable)
    assert task.task_id is None


async def test_stream_none_before_start() -> None:
    task = ProcessTask(sys.executable)
    assert task.stream is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
async def test_run_process_bg_empty_args() -> None:
    task = ProcessTask(sys.executable, ["-c", "print('empty_args_work')"])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    assert result[1] == 0

    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert "empty_args_work" in output

async def test_run_process_bg_with_cwd(tmp_path: Path) -> None:
    task = ProcessTask(
        sys.executable,
        ["-c", "import pathlib, sys; print(pathlib.Path.cwd())"],
        cwd=str(tmp_path),
    )
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    assert result[1] == 0
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    assert str(tmp_path) in output


async def test_run_process_bg_decoder_flush_on_stop() -> None:
    """Stopping a task must flush the incremental UTF-8 decoder so that
    trailing incomplete multi-byte sequences are not silently lost.
    """
    # Write exactly 4095 ASCII bytes + the first byte of a 3-byte UTF-8 char.
    # The reader will decode the ASCII and buffer the trailing byte.
    # When stopped, the finally block must flush it (as a replacement char).
    code = (
        "import sys, time\n"
        "sys.stdout.buffer.write(b'A' * 4095 + b'\\xe3')\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(10)\n"
    )
    task = ProcessTask(sys.executable, ["-c", code])
    q: queue.Queue[str] = queue.Queue()

    bg = asyncio.create_task(task._run_process_bg(q))
    await asyncio.sleep(0.3)
    await task._stop_function()
    result = await asyncio.wait_for(bg, timeout=5)

    assert result[0] is False
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break

    # The 4095 'A' characters must be present.
    assert output.count("A") == 4095
    # The buffered byte must have been flushed (as replacement char) rather
    # than silently discarded.
    assert "\ufffd" in output
