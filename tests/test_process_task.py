"""Comprehensive tests for ProcessTask."""

import asyncio
import os
import queue
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kimix.tools.background.utils import _pop_task_data
from kimix.tools.common import (
    ProcessTask,
    _kill_registered_process_trees,
    _process_registry,
    _register_child_process,
    _unregister_child_process,
    kill_child_tree,
)


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


# ---------------------------------------------------------------------------
# scrub_env / redact (WP1 security) + bounded output (WP4)
# ---------------------------------------------------------------------------

def _drain(q: queue.Queue[str]) -> str:
    output = ""
    while True:
        try:
            output += q.get_nowait()
        except queue.Empty:
            break
    return output


_ENV_PROBE = (
    "import os\n"
    "if os.environ.get('AWS_ACCESS_KEY_ID'):\n"
    "    print('SECRET_VISIBLE')\n"
    "else:\n"
    "    print('SECRET_SCRUBBED')\n"
)


async def test_scrub_env_applied_before_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With scrub_env=True the child must NOT see the parent's AWS key."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_LEAK_ME")
    task = ProcessTask(sys.executable, ["-c", _ENV_PROBE], scrub_env=True)
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    output = _drain(q)
    assert "SECRET_SCRUBBED" in output
    assert "SECRET_VISIBLE" not in output


async def test_no_scrub_leaks_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without scrubbing (default) the child inherits the full parent env."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_LEAK_ME")
    task = ProcessTask(sys.executable, ["-c", _ENV_PROBE], scrub_env=False)
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    output = _drain(q)
    assert "SECRET_VISIBLE" in output


async def test_scrub_env_merge_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scrubbing happens on the base env; an explicit env override restores a
    variable even when its name matches a secret substring."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_LEAK_ME")
    task = ProcessTask(
        sys.executable,
        ["-c", "import os; print(os.environ.get('AWS_ACCESS_KEY_ID', '<missing>'))"],
        scrub_env=True,
        env={"AWS_ACCESS_KEY_ID": "OVERRIDE_VALUE"},
    )
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is True
    output = _drain(q)
    assert "OVERRIDE_VALUE" in output
    assert "AKIA_LEAK_ME" not in output


_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


async def test_redact_masks_jwt_at_capture(mock_session: MagicMock) -> None:
    """redact=True masks a printed JWT before it reaches the stream buffer."""
    task = ProcessTask(
        sys.executable, ["-c", f"print('{_JWT}')"], redact=True
    )
    await task.start(mock_session, kind="run")
    await task.wait(timeout=5)
    output = await task.stream.pop_output()
    assert "[REDACTED]" in output
    assert "eyJ" not in output


async def test_redact_false_leaves_raw(mock_session: MagicMock) -> None:
    """redact=False keeps the captured output untouched."""
    task = ProcessTask(
        sys.executable, ["-c", f"print('{_JWT}')"], redact=False
    )
    await task.start(mock_session, kind="run")
    await task.wait(timeout=5)
    output = await task.stream.pop_output()
    assert _JWT in output


async def test_output_buffer_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long stream is truncated to head+tail in the internal output buffer;
    _find_error_line_index still locates an error line in the retained tail."""
    monkeypatch.setattr(
        "kimix.tools.background.utils.BACKGROUND_MAX_OUTPUT_CHARS", 5000
    )
    code = (
        "for i in range(2000):\n"
        "    print('filler', i)\n"
        "import sys\n"
        "print('ERROR: boom')\n"
        "sys.exit(3)\n"
    )
    task = ProcessTask(sys.executable, ["-c", code])
    q: queue.Queue[str] = queue.Queue()
    result = await task._run_process_bg(q)
    assert result[0] is False
    assert result[1] == 3
    messages = _drain(q)
    assert "Process exited with code 3" in messages
    assert "error at line" in messages
    line_num = int(messages.split("error at line ")[1].rstrip("]"))
    # The bounded buffer keeps only head (2000 chars) + tail (3000 chars) of a
    # ~2000-line stream, so the error line's position must be far below the
    # absolute line count (~2002) — proving the buffer was truncated.
    assert 0 < line_num < 1000


# ---------------------------------------------------------------------------
# kill_child_tree / process-tree registry / orphan prevention
# ---------------------------------------------------------------------------

def _pid_runs_script(pid: int, marker: str) -> bool:
    """Return True if process *pid* is running a command line containing
    *marker*.

    Aliveness-by-PID alone is unreliable on Windows: on a busy machine a PID
    can be recycled within milliseconds of the process dying, so a dead PID
    can appear "alive" again.  Matching the command line is immune to PID
    reuse — it only reports True when the exact script is still running.
    """
    if os.name == "nt":
        out = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        ).stdout
        return marker in out
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        try:
            cmd = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return False
        return marker in cmd
    # Non-Linux POSIX (e.g. macOS): PID reuse is rare, kill(0) is reliable.
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ── kill_child_tree unit tests ────────────────────────────────────────────

def test_kill_child_tree_posix_graceful() -> None:
    mock_os = MagicMock()
    mock_os.name = "posix"
    with patch("kimix.tools.common.os", mock_os):
        kill_child_tree(1234)
        mock_os.killpg.assert_called_once_with(1234, signal.SIGTERM)


def test_kill_child_tree_posix_force() -> None:
    mock_os = MagicMock()
    mock_os.name = "posix"
    with patch("kimix.tools.common.os", mock_os):
        kill_child_tree(1234, force=True)
        mock_os.killpg.assert_called_once_with(1234, getattr(signal, "SIGKILL", 9))


def test_kill_child_tree_posix_swallows_missing_group() -> None:
    mock_os = MagicMock()
    mock_os.name = "posix"
    mock_os.killpg.side_effect = ProcessLookupError
    with patch("kimix.tools.common.os", mock_os):
        # A group that already exited must not raise.
        kill_child_tree(1234)
        mock_os.killpg.assert_called_once_with(1234, signal.SIGTERM)


def test_kill_child_tree_windows_graceful_uses_taskkill_tree() -> None:
    with patch("kimix.tools.common.os.name", "nt"), \
            patch("kimix.tools.common.subprocess.run") as mock_run:
        kill_child_tree(1234)
        mock_run.assert_called_once()
        args = mock_run.call_args.args[0]
        assert args == ["taskkill", "/PID", "1234", "/T"]


def test_kill_child_tree_windows_force_appends_f() -> None:
    with patch("kimix.tools.common.os.name", "nt"), \
            patch("kimix.tools.common.subprocess.run") as mock_run:
        kill_child_tree(1234, force=True)
        args = mock_run.call_args.args[0]
        assert args == ["taskkill", "/PID", "1234", "/T", "/F"]


def test_kill_child_tree_windows_swallows_run_error() -> None:
    with patch("kimix.tools.common.os.name", "nt"), \
            patch("kimix.tools.common.subprocess.run", side_effect=OSError("no taskkill")):
        # Missing taskkill / permission errors must not raise.
        kill_child_tree(1234)


def test_kill_child_tree_windows_swallows_timeout() -> None:
    with patch("kimix.tools.common.os.name", "nt"), \
            patch(
                "kimix.tools.common.subprocess.run",
                side_effect=subprocess.TimeoutExpired("taskkill", 2.0),
            ):
        # A GUI child ignoring WM_CLOSE must not hang the caller forever.
        kill_child_tree(1234)


# ── process-tree registry ─────────────────────────────────────────────────

def test_registry_register_and_unregister() -> None:
    pid = 42_001
    try:
        _register_child_process(pid)
        assert pid in _process_registry
        _unregister_child_process(pid)
        assert pid not in _process_registry
    finally:
        _unregister_child_process(pid)


def test_kill_registered_process_trees_force_kills_all() -> None:
    pids = (42_002, 42_003)
    try:
        for pid in pids:
            _register_child_process(pid)
        with patch("kimix.tools.common.kill_child_tree") as mock_kill:
            _kill_registered_process_trees()
            for pid in pids:
                mock_kill.assert_any_call(pid, force=True)
    finally:
        for pid in pids:
            _unregister_child_process(pid)


def test_atexit_hook_registered() -> None:
    import atexit

    if not hasattr(atexit, "_ncallbacks"):
        pytest.skip("atexit internals not available on this interpreter")
    before = atexit._ncallbacks()
    # unregister() removes the hook only if it is registered (silent no-op
    # otherwise); re-register it afterwards to leave state unchanged.
    atexit.unregister(_kill_registered_process_trees)
    assert atexit._ncallbacks() == before - 1
    atexit.register(_kill_registered_process_trees)
    assert atexit._ncallbacks() == before


# ── grandchild orphan prevention (integration) ────────────────────────────

async def test_stop_kills_grandchild_process_tree(tmp_path: Path) -> None:
    """Stopping a task must terminate the whole process tree, not just the
    direct child.

    A surviving grandchild keeps the stdout/stderr pipe write ends open, so
    ``_run_process_bg`` would never see EOF and the task would be stuck in the
    "running" state forever.  This test spawns a real grandchild and verifies
    that (a) ``_run_process_bg`` returns promptly after a stop — the decisive
    proof that the pipes closed, i.e. the whole tree died — and (b) the
    grandchild's process is gone.
    """
    interp = getattr(sys, "_base_executable", None) or sys.executable
    pidfile = tmp_path / "grandchild.pid"

    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "import os, time\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import subprocess, sys\n"
        f"gc = subprocess.Popen([{interp!r}, {str(grandchild_script)!r}])\n"
        "try:\n"
        "    gc.wait()\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )

    task = ProcessTask(interp, [str(parent_script)])
    q: queue.Queue[str] = queue.Queue()
    bg = asyncio.create_task(task._run_process_bg(q))
    grandchild_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert pidfile.exists(), "grandchild never started"
        grandchild_pid = int(pidfile.read_text(encoding="utf-8").strip())
        # Sanity: the grandchild is really running our marker script.
        assert _pid_runs_script(grandchild_pid, "grandchild.py")

        await task._stop_function()
        # Decisive: _run_process_bg must return promptly once the tree is
        # killed.  It would hang until the 10s timeout if the grandchild kept
        # the pipe write ends open.
        result = await asyncio.wait_for(bg, timeout=10)
        assert result[0] is False

        # Secondary: the grandchild process is gone (command-line check is
        # immune to PID reuse).
        assert not _pid_runs_script(grandchild_pid, "grandchild.py")
    finally:
        # Defensive cleanup in case an assertion above failed.
        if grandchild_pid is not None and _pid_runs_script(grandchild_pid, "grandchild.py"):
            kill_child_tree(grandchild_pid, force=True)
