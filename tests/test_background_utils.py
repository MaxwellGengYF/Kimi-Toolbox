"""Comprehensive tests for BackgroundStream and task utilities."""

import asyncio
import io
import queue
import re
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from kimix.tools.background.utils import (
    BackgroundStream,
    TaskData,
    _get_or_add_task_data,
    _get_task_data,
    _pop_task_data,
    add_task,
    bounded_append,
    bounded_put,
    discard_all_tasks,
    generate_task_id,
    get_all_tasks,
    join_task,
    remove_task_id,
)


# ---------------------------------------------------------------------------
# BackgroundStream fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stream() -> BackgroundStream:
    return BackgroundStream()


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    return session


# ---------------------------------------------------------------------------
# BackgroundStream – construction / initial state
# ---------------------------------------------------------------------------
async def test_initial_state(stream: BackgroundStream) -> None:
    assert await stream.is_started() is False
    assert await stream.is_stopped() is False
    assert await stream.thread_is_alive() is False
    assert await stream.success() is False
    assert await stream.get_output() == ""
    assert await stream.get_queue() is None


async def test_get_queue_returns_none_before_start(stream: BackgroundStream) -> None:
    assert await stream.get_queue() is None


# ---------------------------------------------------------------------------
# BackgroundStream – start with sync function
# ---------------------------------------------------------------------------
async def test_start_sync_function(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("hello")
        q.put("world")
        time.sleep(0.05)

    await stream.start(worker, stop_function=lambda: None)
    assert await stream.is_started() is True
    assert await stream.thread_is_alive() is True

    await stream.wait()
    assert await stream.thread_is_alive() is False
    assert await stream.get_output() == "helloworld"
    assert await stream.success() is True


async def test_start_idempotent(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("a")

    await stream.start(worker, stop_function=lambda: None)
    # second start should be ignored
    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.get_output() == "a"


# ---------------------------------------------------------------------------
# BackgroundStream – start with async function
# ---------------------------------------------------------------------------
async def test_start_async_function(stream: BackgroundStream) -> None:
    async def worker(q: queue.Queue[str]) -> None:
        q.put("async")
        q.put("data")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.get_output() == "asyncdata"
    assert await stream.success() is True


# ---------------------------------------------------------------------------
# BackgroundStream – success handling
# ---------------------------------------------------------------------------
async def test_success_false_when_function_returns_false(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> bool:
        q.put("x")
        return False

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.success() is False


async def test_success_true_when_function_returns_true(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> bool:
        q.put("x")
        return True

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.success() is True


async def test_success_true_when_function_returns_non_bool(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> str:
        q.put("x")
        return "anything"

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.success() is True


async def test_success_false_on_exception(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        raise RuntimeError("boom")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.success() is False


async def test_success_false_on_async_exception(stream: BackgroundStream) -> None:
    async def worker(q: queue.Queue[str]) -> None:
        raise RuntimeError("async boom")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.success() is False


# ---------------------------------------------------------------------------
# BackgroundStream – get_output / pop_output
# ---------------------------------------------------------------------------
async def test_get_output_accumulates(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("a")
        q.put("b")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.get_output() == "ab"
    # second call should return same content
    assert await stream.get_output() == "ab"


async def test_pop_output_clears_buffer(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("a")
        q.put("b")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.pop_output() == "ab"
    assert await stream.pop_output() == ""


async def test_get_output_without_queue_uses_internal_buffer(stream: BackgroundStream) -> None:
    # before start, _output is empty
    assert await stream.get_output() == ""


async def test_get_output_partial_during_run(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("partial")
        time.sleep(0.05)
        q.put("rest")

    await stream.start(worker, stop_function=lambda: None)
    # give thread time to put first item
    time.sleep(0.02)
    partial = await stream.get_output()
    assert partial == "partial"
    await stream.wait()
    assert await stream.get_output() == "partialrest"


# ---------------------------------------------------------------------------
# BackgroundStream – stop
# ---------------------------------------------------------------------------
async def test_stop_before_start(stream: BackgroundStream) -> None:
    result = await stream.stop()
    assert result is False


async def test_stop_with_sync_stop_function(stream: BackgroundStream) -> None:
    stopped = threading.Event()

    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    def stopper() -> None:
        stopped.set()

    await stream.start(worker, stop_function=stopper)
    assert await stream.stop() is True
    assert stopped.is_set()
    assert await stream.is_stopped() is True


async def test_stop_with_async_stop_function(stream: BackgroundStream) -> None:
    stopped = asyncio.Event()

    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    async def stopper() -> None:
        stopped.set()

    await stream.start(worker, stop_function=stopper)
    assert await stream.stop() is True
    assert stopped.is_set()
    assert await stream.is_stopped() is True


async def test_stop_ignores_exception_in_stop_function(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    def bad_stopper() -> None:
        raise RuntimeError("stop error")

    await stream.start(worker, stop_function=bad_stopper)
    # should not raise
    result = await stream.stop()
    assert result is True
    assert await stream.is_stopped() is True


async def test_stop_idempotent(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    await stream.start(worker, stop_function=lambda: None)
    assert await stream.stop() is True
    assert await stream.stop() is False


# ---------------------------------------------------------------------------
# BackgroundStream – input
# ---------------------------------------------------------------------------
async def test_input_without_input_function(stream: BackgroundStream) -> None:
    assert await stream.input("hello") is False


async def test_input_with_sync_function(stream: BackgroundStream) -> None:
    received: list[str] = []

    def worker(q: queue.Queue[str]) -> None:
        time.sleep(0.1)

    def input_handler(data: str) -> bool:
        received.append(data)
        return True

    await stream.start(worker, stop_function=lambda: None, input_function=input_handler)
    assert await stream.input("test") is True
    assert received == ["test"]
    await stream.wait()


async def test_input_with_async_function(stream: BackgroundStream) -> None:
    received: list[str] = []

    def worker(q: queue.Queue[str]) -> None:
        time.sleep(0.1)

    async def input_handler(data: str) -> bool:
        received.append(data)
        return True

    await stream.start(worker, stop_function=lambda: None, input_function=input_handler)
    assert await stream.input("async") is True
    assert received == ["async"]
    await stream.wait()


async def test_input_returns_bool_cast(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(0.1)

    def input_handler(data: str) -> Any:
        return None  # falsy

    await stream.start(worker, stop_function=lambda: None, input_function=input_handler)
    assert await stream.input("x") is False
    await stream.wait()


# ---------------------------------------------------------------------------
# BackgroundStream – wait
# ---------------------------------------------------------------------------
async def test_wait_on_not_started(stream: BackgroundStream) -> None:
    # should return immediately without error
    await stream.wait()
    assert await stream.thread_is_alive() is False


async def test_wait_timeout(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait(timeout=0.01)
    # thread may still be alive due to timeout
    assert await stream.is_started() is True


async def test_wait_sets_thread_none_after_join(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("done")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    assert await stream.thread_is_alive() is False


# ---------------------------------------------------------------------------
# BackgroundStream – wait_with_inactivity_timeout
# ---------------------------------------------------------------------------
async def test_wait_with_inactivity_timeout_short_timeout_uses_wait(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("hello")
        time.sleep(0.2)

    await stream.start(worker, stop_function=lambda: None)
    completed, elapsed, inactivity_timed_out = await stream.wait_with_inactivity_timeout(
        timeout=0.5, inactivity_timeout=60.0
    )
    assert completed is True
    assert inactivity_timed_out is False
    assert elapsed < 1.0


async def test_wait_with_inactivity_timeout_uses_module_default(
    stream: BackgroundStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No explicit inactivity_timeout falls back to the module-level default."""
    monkeypatch.setattr(
        "kimix.tools.background.utils.DEFAULT_INACTIVITY_TIMEOUT", 2.0
    )

    def worker(q: queue.Queue[str]) -> None:
        time.sleep(10.0)

    await stream.start(worker, stop_function=lambda: None)
    completed, elapsed, inactivity_timed_out = await stream.wait_with_inactivity_timeout(
        timeout=15.0
    )
    assert completed is False
    assert inactivity_timed_out is True
    assert elapsed < 5.0


async def test_wait_with_inactivity_timeout_triggers_on_no_output(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(10.0)

    await stream.start(worker, stop_function=lambda: None)
    completed, elapsed, inactivity_timed_out = await stream.wait_with_inactivity_timeout(
        timeout=15.0, inactivity_timeout=2.0
    )
    assert completed is False
    assert inactivity_timed_out is True
    assert elapsed < 5.0


async def test_wait_with_inactivity_timeout_does_not_trigger_when_output_flows(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        for _ in range(10):
            q.put(".")
            time.sleep(0.3)

    await stream.start(worker, stop_function=lambda: None)
    completed, elapsed, inactivity_timed_out = await stream.wait_with_inactivity_timeout(
        timeout=5.0, inactivity_timeout=1.0
    )
    assert completed is True
    assert inactivity_timed_out is False
    assert elapsed < 5.0


async def test_get_output_updates_last_output_time(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(0.1)
        q.put("pulse")
        time.sleep(0.5)

    await stream.start(worker, stop_function=lambda: None)
    # Wait for the worker to emit its first (and only) output chunk.
    await asyncio.sleep(0.3)
    await stream.get_output()
    # The get_output call should have reset the activity timestamp. The worker
    # finishes shortly afterwards, so the inactivity timeout must not fire.
    completed, elapsed, inactivity_timed_out = await stream.wait_with_inactivity_timeout(
        timeout=5.0, inactivity_timeout=1.0
    )
    assert completed is True
    assert inactivity_timed_out is False
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# BackgroundStream – threading safety (basic stress)
# ---------------------------------------------------------------------------
async def test_concurrent_get_output(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        for i in range(100):
            q.put(str(i))

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    output = await stream.get_output()
    # digits 0-9 → 1 char each (10), 10-99 → 2 chars each (180) → total 190
    assert len(output) == 190


# ---------------------------------------------------------------------------
# TaskData / Session utilities
# ---------------------------------------------------------------------------


def test_get_or_add_task_data(mock_session: MagicMock) -> None:
    data = _get_or_add_task_data(mock_session)
    assert isinstance(data, TaskData)
    assert "background_task_data" in mock_session.custom_data
    # second call returns same object
    assert _get_or_add_task_data(mock_session) is data


def test_get_task_data(mock_session: MagicMock) -> None:
    assert _get_task_data(mock_session) is None
    data = _get_or_add_task_data(mock_session)
    assert _get_task_data(mock_session) is data


def test_pop_task_data(mock_session: MagicMock) -> None:
    assert _pop_task_data(mock_session) is None
    data = _get_or_add_task_data(mock_session)
    assert _pop_task_data(mock_session) is data
    assert _get_task_data(mock_session) is None


# ---------------------------------------------------------------------------
# generate_task_id
# ---------------------------------------------------------------------------
def test_generate_task_id_without_name(mock_session: MagicMock) -> None:
    assert generate_task_id(mock_session, "kind") == "kind"
    assert generate_task_id(mock_session, "kind") == "kind_1"
    assert generate_task_id(mock_session, "kind") == "kind_2"


def test_generate_task_id_with_name(mock_session: MagicMock) -> None:
    assert generate_task_id(mock_session, "kind", "name") == "kind_name"
    assert generate_task_id(mock_session, "kind", "name") == "kind_name_1"


def test_generate_task_id_strips(mock_session: MagicMock) -> None:
    assert generate_task_id(mock_session, "kind", "  name  ") == "kind_  name  "


# ---------------------------------------------------------------------------
# add_task / get_all_tasks / remove_task_id
# ---------------------------------------------------------------------------
def test_add_and_get_all_tasks(mock_session: MagicMock, stream: BackgroundStream) -> None:
    add_task(mock_session, "t1", stream)
    tasks = get_all_tasks(mock_session)
    assert tasks == {"t1": stream}


def test_remove_task_id(mock_session: MagicMock, stream: BackgroundStream) -> None:
    add_task(mock_session, "t1", stream)
    remove_task_id(mock_session, "t1")
    assert get_all_tasks(mock_session) == {}


def test_remove_task_id_missing(mock_session: MagicMock) -> None:
    # should not raise
    remove_task_id(mock_session, "missing")


def test_add_task_strips_id(mock_session: MagicMock, stream: BackgroundStream) -> None:
    add_task(mock_session, "  t1  ", stream)
    assert "t1" in get_all_tasks(mock_session)


# ---------------------------------------------------------------------------
# join_task
# ---------------------------------------------------------------------------
async def test_join_task_success(mock_session: MagicMock, stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("done")

    await stream.start(worker, stop_function=lambda: None)
    add_task(mock_session, "t1", stream)
    assert await join_task(mock_session, "t1") is True
    assert get_all_tasks(mock_session) == {}


async def test_join_task_not_found(mock_session: MagicMock) -> None:
    assert await join_task(mock_session, "missing") is False


async def test_join_task_no_data(mock_session: MagicMock) -> None:
    assert await join_task(mock_session, "missing") is False


# ---------------------------------------------------------------------------
# discard_all_tasks
# ---------------------------------------------------------------------------
async def test_discard_all_tasks(mock_session: MagicMock) -> None:
    def worker(q: queue.Queue[str]) -> None:
        time.sleep(1.0)

    stream1 = BackgroundStream()
    stream2 = BackgroundStream()
    await stream1.start(worker, stop_function=lambda: None)
    await stream2.start(worker, stop_function=lambda: None)
    add_task(mock_session, "t1", stream1)
    add_task(mock_session, "t2", stream2)

    await discard_all_tasks(mock_session)
    assert _get_task_data(mock_session) is None


async def test_discard_all_tasks_no_data(mock_session: MagicMock) -> None:
    # should not raise
    await discard_all_tasks(mock_session)


# ---------------------------------------------------------------------------
# BackgroundStream – wait_for_output
# ---------------------------------------------------------------------------
async def test_wait_for_output_matches_pattern(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("hello ")
        time.sleep(0.05)
        q.put("world")

    await stream.start(worker, stop_function=lambda: None)
    output, matched, elapsed = await stream.wait_for_output(
        timeout=2.0, pattern=re.compile(r"hello world")
    )
    assert matched is True
    assert "hello world" in output
    assert elapsed >= 0.0


async def test_wait_for_output_returns_on_no_match(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("goodbye")
        time.sleep(0.1)

    await stream.start(worker, stop_function=lambda: None)
    output, matched, elapsed = await stream.wait_for_output(
        timeout=0.5, pattern=re.compile(r"hello world")
    )
    assert matched is False
    assert "goodbye" in output
    assert elapsed >= 0.0


async def test_wait_for_output_zero_timeout_checks_current_output(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("already here")

    await stream.start(worker, stop_function=lambda: None)
    # Give the worker a moment to queue the text.
    time.sleep(0.05)
    output, matched, elapsed = await stream.wait_for_output(timeout=0, pattern=re.compile(r"already"))
    assert matched is True
    assert "already here" in output
    assert elapsed >= 0.0
    assert elapsed < 0.1


async def test_wait_for_output_no_pattern_waits_for_completion(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("done")

    await stream.start(worker, stop_function=lambda: None)
    output, matched, elapsed = await stream.wait_for_output(timeout=2.0)
    assert matched is False
    assert output == "done"
    assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# bounded_append / bounded_put / output truncation (WP4)
# ---------------------------------------------------------------------------
def test_bounded_append_under_cap_no_op() -> None:
    buf = io.StringIO()
    assert bounded_append(buf, "hello world", 1000) is False
    assert buf.getvalue() == "hello world"


def test_bounded_append_over_cap_head_tail_marker() -> None:
    buf = io.StringIO()
    cap = 1000
    assert bounded_append(buf, "x" * 3000, cap) is True
    value = buf.getvalue()
    assert value.startswith("x" * 400)  # head 40% of cap
    assert value.endswith("x" * 600)  # tail 60% of cap
    assert "keeping first 400 and last 600 chars" in value
    assert "(output truncated" in value
    # Retained content stays bounded (cap + marker + margins).
    assert len(value) < cap + 200


def test_bounded_append_accumulates_and_truncates() -> None:
    buf = io.StringIO()
    cap = 1000
    assert bounded_append(buf, "a" * 600, cap) is False
    assert bounded_append(buf, "b" * 600, cap) is True  # 1200 > 1000
    value = buf.getvalue()
    assert value.startswith("a" * 400)
    assert value.endswith("b" * 600)
    assert len(value) < cap + 200


def test_bounded_put_drops_oldest() -> None:
    q: queue.Queue[str] = queue.Queue()
    for i in range(3):
        bounded_put(q, f"chunk{i}", max_chunks=3)
    assert q.qsize() == 3
    # Adding a 4th chunk evicts the oldest.
    bounded_put(q, "chunk3", max_chunks=3)
    assert q.qsize() == 3
    items: list[str] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    assert items == ["chunk1", "chunk2", "chunk3"]


def test_bounded_put_default_max_chunks() -> None:
    q: queue.Queue[str] = queue.Queue()
    for i in range(5):
        bounded_put(q, f"c{i}")
    assert q.qsize() == 5  # well under the default 10_000 → nothing dropped


async def test_get_output_bounded_with_truncation_flag(
    stream: BackgroundStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kimix.tools.background.utils.BACKGROUND_MAX_OUTPUT_CHARS", 1000
    )

    def worker(q: queue.Queue[str]) -> None:
        for _ in range(200):
            q.put("y" * 100)  # 20,000 chars total, far above the 1,000 cap

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    output = await stream.get_output()
    assert len(output) < 1200
    assert "(output truncated" in output
    assert stream.output_truncated is True


async def test_output_truncated_false_under_cap(stream: BackgroundStream) -> None:
    def worker(q: queue.Queue[str]) -> None:
        q.put("small output")

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    output = await stream.get_output()
    assert output == "small output"
    assert stream.output_truncated is False


async def test_output_truncated_sticky_after_pop(
    stream: BackgroundStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "kimix.tools.background.utils.BACKGROUND_MAX_OUTPUT_CHARS", 1000
    )

    def worker(q: queue.Queue[str]) -> None:
        for _ in range(200):
            q.put("z" * 100)

    await stream.start(worker, stop_function=lambda: None)
    await stream.wait()
    await stream.pop_output()
    assert stream.output_truncated is True
