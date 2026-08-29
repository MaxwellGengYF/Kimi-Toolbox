import asyncio
import inspect
import io
import regex as re
import threading
import time
import queue
from typing import Any, Awaitable, Callable, cast

BackgroundOutputFormatter = Callable[
    [str, bool, int | None, float | None, bool | None],
    Awaitable[tuple[str, str, str | None, str | None, bool]],
]

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimi_cli.session import Session

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")


# Seconds of output inactivity that triggers an early return from a blocking
# wait (see ``wait_with_inactivity_timeout`` / ``wait_for_output``).  Callers
# that pass no explicit ``inactivity_timeout`` fall back to this value, so it
# is read at call time (never imported once at module load) to keep the value
# patchable in tests.
DEFAULT_INACTIVITY_TIMEOUT = 120.0

# Maximum characters retained in an output buffer before it is rewritten to
# head 40% + tail 60% with a marker line (see ``bounded_append``).  Read at
# call time so it stays patchable in tests.
BACKGROUND_MAX_OUTPUT_CHARS = 200_000

# Maximum queued output chunks before the oldest chunk is dropped (see
# ``bounded_put``).  Read at call time so it stays patchable in tests.
MAX_QUEUE_CHUNKS = 10_000


def bounded_append(buf: io.StringIO, text: str, cap: int) -> bool:
    """Write *text* to *buf*, bounding the retained output to *cap* chars.

    When the buffer size exceeds *cap* after the write, it is rewritten to the
    first 40% of *cap* plus the last 60% of *cap*, joined by a marker line
    ``[... (output truncated, keeping first N and last M chars)]``, and
    ``True`` is returned.  ``False`` is returned when no truncation happened.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None:
        # tell-guard: the native contract is (content, text, cap) ->
        # (new_content, truncated); only cross the boundary when a truncation
        # is actually needed (bit-identical to the Python body below).
        if buf.tell() + len(text) > cap:
            content, truncated = _NATIVE_TOOLS.bounded_append(
                buf.getvalue(), text, cap
            )
            buf.seek(0)
            buf.truncate(0)
            buf.write(content)
            return truncated
        buf.write(text)
        return False
    buf.write(text)
    if buf.tell() <= cap:
        return False
    full = buf.getvalue()
    head_len = int(cap * 0.4)
    tail_len = cap - head_len
    head = full[:head_len]
    tail = full[-tail_len:] if tail_len else ""
    marker = f"\n[... (output truncated, keeping first {head_len} and last {tail_len} chars)]\n"
    buf.seek(0)
    buf.truncate(0)
    buf.write(head + marker + tail)
    return True


def bounded_put(q: queue.Queue[str], text: str, max_chunks: int = MAX_QUEUE_CHUNKS) -> None:
    """Put *text* on the queue, dropping the oldest chunk when it is full.

    Keeps the queue size at most *max_chunks* so a long-running producer
    cannot grow the queue without bound.  The most recent output is always
    retained; only the oldest chunks are discarded.
    """
    while q.qsize() >= max_chunks:
        try:
            q.get_nowait()
        except queue.Empty:
            break
    q.put_nowait(text)


class TaskData:
    def __init__(self) -> None:
        self.task_names: dict[str, int] = {}
        self.tasks: dict[str, BackgroundStream] = {}


def _get_or_add_task_data(session: Session) -> TaskData:
    data = session.custom_data.get('background_task_data')
    if data is None:
        data = TaskData()
        session.custom_data['background_task_data'] = data
    return cast(TaskData, data)


def _get_task_data(session: Session) -> TaskData | None:
    return cast(TaskData | None, session.custom_data.get('background_task_data'))


def _pop_task_data(session: Session) -> TaskData | None:
    return cast(TaskData | None, session.custom_data.pop('background_task_data', None))


class BackgroundStream:
    """A wrapper for background thread execution with a thread-safe queue."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[str] | None = None
        self._started: bool = False
        self._stopped: bool = False
        self._stop_function: Callable[[], Any] | None = None
        self._input_function: Callable[[str], Any] | None = None
        self._lock = threading.Lock()
        self._success = False
        self._exit_code: int | None = None
        self._output = io.StringIO()
        self._output_truncated = False
        self._last_output_time = time.monotonic()
        self._completed_event = threading.Event()
        self._process_elapsed: float | None = None
        self.format_output: BackgroundOutputFormatter | None = None

    async def success(self) -> bool:
        return self._success

    @property
    def exit_code(self) -> int | None:
        """The exit code of the subprocess, or None if not yet completed."""
        return self._exit_code

    @property
    def output_truncated(self) -> bool:
        """True when the retained output buffer has been truncated (head+tail)."""
        return self._output_truncated

    @property
    def process_elapsed(self) -> float | None:
        """The total running time of the subprocess in seconds, or None."""
        return self._process_elapsed

    async def start(self, function: Callable[[queue.Queue[str]], Any] | Callable[[queue.Queue[str]], Awaitable[Any]], stop_function: Callable[[], Any] | Callable[[], Awaitable[Any]], input_function: Callable[[str], Any] | Callable[[str], Awaitable[Any]] | None = None) -> None:
        """Start the background thread with the given function.

        Args:
            function: A callable that accepts a queue.Queue[str] as its argument.
                     The function can put strings into the queue for retrieval by other threads.
        """
        with self._lock:
            if self._started:
                return

            q: queue.Queue[str] = queue.Queue()
            self._queue = q

            def func(v: BackgroundStream, function: Callable[[queue.Queue[str]], Any] | Callable[[queue.Queue[str]], Awaitable[Any]]) -> None:
                try:
                    if inspect.iscoroutinefunction(function):
                        result = asyncio.run(function(q))
                    else:
                        result = function(q)
                    if isinstance(result, tuple) and len(result) >= 2:
                        v._success = bool(result[0])
                        v._exit_code = result[1]
                    else:
                        v._success = False if result == False else True
                except Exception:
                    v._success = False
                finally:
                    v._completed_event.set()
            self._thread = threading.Thread(
                target=func, args=(self, function), daemon=True)
            self._stop_function = stop_function
            self._input_function = input_function
            self._started = True
        self._thread.start()

    async def input(self, data: str) -> bool:
        if self._input_function:
            if inspect.iscoroutinefunction(self._input_function):
                return bool(await self._input_function(data))
            return bool(self._input_function(data))
        return False

    async def thread_is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    async def wait(self, timeout: float | None = None) -> None:
        """Wait for the background thread to complete."""
        if not await self.thread_is_alive():
            return
        thread = self._thread
        if thread is None:
            return
        await asyncio.to_thread(thread.join, timeout=timeout)
        if not thread.is_alive():
            with self._lock:
                self._thread = None

    async def get_output(self) -> str:
        if self._queue is None:
            return self._output.getvalue()

        new_data = False
        while True:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break
            if bounded_append(self._output, chunk, BACKGROUND_MAX_OUTPUT_CHARS):
                self._output_truncated = True
            new_data = True
        if new_data:
            with self._lock:
                self._last_output_time = time.monotonic()
        return self._output.getvalue()

    async def pop_output(self) -> str:
        output = await self.get_output()
        self._output.truncate(0)
        self._output.seek(0)
        return output

    async def get_queue(self) -> queue.Queue[str] | None:
        """Get the thread-safe queue for retrieving messages.

        Returns:
            The queue if started, None otherwise.
        """
        return self._queue

    async def is_started(self) -> bool:
        """Check if the stream has been started."""
        return self._started

    async def is_stopped(self) -> bool:
        """Check if the stream has been stopped."""
        return self._stopped

    async def wait_with_inactivity_timeout(
        self,
        timeout: float,
        inactivity_timeout: float | None = None,
    ) -> tuple[bool, float, bool]:
        """Wait for the background thread, exiting early on output inactivity.

        If ``timeout > inactivity_timeout``, the wait is interrupted when no
        new output has been received for ``inactivity_timeout`` seconds.

        Args:
            timeout: Maximum total seconds to wait.
            inactivity_timeout: Seconds of output inactivity that triggers an
                early return. Only active when ``timeout`` exceeds it.
                Defaults to ``DEFAULT_INACTIVITY_TIMEOUT`` at call time so
                tests can patch the module constant.

        Returns:
            ``(completed, elapsed_seconds, inactivity_timed_out)``.
            ``completed`` is ``True`` when the thread finished on its own.
            ``inactivity_timed_out`` is ``True`` only when the early return was
            caused by output inactivity.
        """
        if inactivity_timeout is None:
            inactivity_timeout = DEFAULT_INACTIVITY_TIMEOUT

        start = time.monotonic()

        # Preserve exact original behavior for short timeouts.
        if timeout <= inactivity_timeout:
            await self.wait(timeout)
            elapsed = time.monotonic() - start
            return not await self.thread_is_alive(), elapsed, False

        # Long-timeout mode: monitor output activity.
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return False, elapsed, False
            if not await self.thread_is_alive():
                return True, elapsed, False

            # Drain any new output and refresh the activity timestamp.
            await self.get_output()
            # Check clean completion signal before inactivity timeout.
            if self._completed_event.is_set():
                return True, elapsed, False
            with self._lock:
                inactive_for = time.monotonic() - self._last_output_time
            if inactive_for >= inactivity_timeout:
                return False, elapsed, True

            await asyncio.sleep(0.5)

    async def wait_for_output(
        self,
        *,
        timeout: float,
        pattern: re.Pattern[str] | None = None,
        inactivity_timeout: float | None = None,
    ) -> tuple[str, bool, float]:
        """Wait for output, optionally until ``pattern`` matches.

        Args:
            timeout: Maximum seconds to wait. ``0`` returns immediately after
                checking the current accumulated output.
            pattern: Compiled regex pattern. When provided, the loop stops as
                soon as the pattern is found in the accumulated output.
            inactivity_timeout: Seconds of output inactivity that triggers an
                early return. When set, if no new output has been received for
                this many seconds, the wait ends early. Also checks the
                internal ``_completed_event`` to detect thread completion.

        Returns:
            ``(output, matched, elapsed_seconds)``. ``matched`` is ``True`` only
            when ``pattern`` was supplied and found before the timeout.
        """
        start = time.monotonic()
        matched = False
        output = ""
        elapsed = 0.0

        while True:
            output = await self.get_output()
            elapsed = time.monotonic() - start
            if pattern is not None and pattern.search(output):
                matched = True
                break
            if timeout <= 0 or elapsed >= timeout:
                break
            # Check if the background thread has finished (process completed).
            if self._completed_event.is_set():
                output = await self.get_output()
                break
            if not await self.thread_is_alive():
                # Process finished; grab any final data that arrived since the
                # last poll before leaving the loop.
                output = await self.get_output()
                break
            # Check inactivity timeout: if enabled and no output for too long, exit early.
            if inactivity_timeout is not None and inactivity_timeout > 0:
                with self._lock:
                    inactive_for = time.monotonic() - self._last_output_time
                if inactive_for >= inactivity_timeout:
                    # Drain any final output before returning.
                    output = await self.get_output()
                    break
            await asyncio.sleep(0.1)

        # Clear the internal buffer since the caller has consumed all
        # accumulated output via the returned `output` string. This ensures
        # subsequent get_output() / pop_output() calls do not repeat the
        # same data.
        self._output.truncate(0)
        self._output.seek(0)

        return output, matched, elapsed

    async def stop(self) -> bool:
        """Stop the background thread.

        Returns:
            True if the thread was stopped, False if it was not running.
        """
        with self._lock:
            if not self._started or self._stopped:
                return False
            self._stopped = True
            thread_alive = self._thread is not None and self._thread.is_alive()
            stop_func = self._stop_function if thread_alive else None

        if stop_func is not None:
            try:
                if inspect.iscoroutinefunction(stop_func):
                    await stop_func()
                else:
                    stop_func()
            except Exception:
                pass
        return thread_alive


def generate_task_id(session: Session, kind: str, name: str | None = None) -> str:
    if name:
        base_id = f"{kind}_{name}"
    else:
        base_id = kind
    data = _get_or_add_task_data(session)
    if base_id not in data.task_names:
        data.task_names[base_id] = 0
        return base_id

    data.task_names[base_id] += 1
    return f"{base_id}_{data.task_names[base_id]}".strip()


def remove_task_id(session: Session, task_id: str) -> BackgroundStream | None:
    task_id = task_id.strip()
    """Remove a task_id from the session task registry.

    Args:
        session: The session instance.
        task_id: The task identifier to remove.
    """
    try:
        data = _get_task_data(session)
        if data is not None:
            data.tasks.pop(task_id)
    except KeyError:
        pass
    return None


def add_task(session: Session, task_id: str, stream: BackgroundStream) -> None:
    task_id = task_id.strip()
    """Add a task to the session task registry.

    Args:
        session: The session instance.
        task_id: Unique identifier for the task.
        stream: The BackgroundStream instance to manage (should already be started).
    """
    _get_or_add_task_data(session).tasks[task_id] = stream


def get_all_tasks(session: Session) -> dict[str, BackgroundStream]:
    return _get_or_add_task_data(session).tasks


async def join_task(session: Session, task_id: str) -> bool:
    task_id = task_id.strip()
    """Join a task and clean up its resources.

    Args:
        session: The session instance.
        task_id: The task identifier to join.

    Returns:
        True if the task was found and joined, False otherwise.
    """
    data = _get_task_data(session)
    if (data is None) or (task_id not in data.tasks):
        return False

    stream = data.tasks.pop(task_id)
    await stream.wait()
    return True


async def discard_all_tasks(session: Session) -> None:
    """Join all tasks and clear the session registries."""
    data = _pop_task_data(session)
    if data is None:
        return
    for stream in list(data.tasks.values()):
        await stream.stop()
    del data
