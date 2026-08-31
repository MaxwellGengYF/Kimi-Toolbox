"""Tests for kimix.utils.session.close_session KeyboardInterrupt safety.

On Windows, Ctrl+C can land inside ``asyncio.run()`` while it is creating
the ProactorEventLoop (specifically ``socket.socketpair()`` used for the
self-pipe). This happens at session teardown when a background tool (e.g.
a running glob) delivered the interrupt. ``close_session`` must swallow it
so interpreter shutdown completes, and ``_shutdown_all_sessions`` must never
let it escape either.
"""

from __future__ import annotations

from typing import Any

import pytest

import kimix.utils._globals as _globals
import kimix.utils.session as session_mod


class _SessionThatRaisesKI:
    """Stands in for kimi_agent_sdk.Session; close() interrupts the loop."""

    def __init__(self, close_result: Any = None) -> None:
        self._close_result = close_result

    async def close(self) -> Any:
        # Simulate the OS delivering Ctrl+C during the creation of the
        # event loop inside asyncio.run() (before our coroutine body even
        # starts running). Raising here gives the same observable effect:
        # asyncio.run(session.close()) raises KeyboardInterrupt.
        raise KeyboardInterrupt


class _SessionThatRaisesKIFromLoopSetup:
    """A session whose close coroutine triggers a KeyboardInterrupt."""

    async def close(self) -> None:
        raise KeyboardInterrupt


def test_close_session_swallows_keyboard_interrupt() -> None:
    session = _SessionThatRaisesKI()
    # Must not propagate — the CLI calls this during teardown when
    # Ctrl+C was already handled by the caller.
    session_mod.close_session(session)  # type: ignore[arg-type]


def test_close_session_untracks_session_before_close() -> None:
    session = _SessionThatRaisesKI()
    _globals._track_session(session)  # type: ignore[arg-type]
    assert session in _globals._live_sessions
    session_mod.close_session(session)  # type: ignore[arg-type]
    assert session not in _globals._live_sessions


def test_close_session_still_raises_other_errors() -> None:
    class _Boom:
        async def close(self) -> None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        session_mod.close_session(_Boom())  # type: ignore[arg-type]


def test_close_session_still_swallows_event_loop_closed() -> None:
    class _LoopClosed:
        async def close(self) -> None:
            raise RuntimeError("Event loop is closed")

    # Existing resilience: transports bound to a closed ProactorEventLoop.
    session_mod.close_session(_LoopClosed())  # type: ignore[arg-type]


def test_shutdown_all_sessions_never_escapes_with_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    ki_session = _SessionThatRaisesKIFromLoopSetup()
    _globals._track_session(ki_session)  # type: ignore[arg-type]
    _globals._default_session = ki_session  # type: ignore[assignment]

    def fake_close(session: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(session_mod, "close_session", fake_close)

    # Must not raise — this runs inside threading._register_atexit during
    # interpreter shutdown, where an exception means the process dies with
    # a traceback instead of exiting.
    session_mod._shutdown_all_sessions()


def test_shutdown_all_sessions_swallows_arbitrary_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _globals._track_session(_SessionThatRaisesKI())  # type: ignore[arg-type]

    def fake_close(session: Any) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(session_mod, "close_session", fake_close)
    session_mod._shutdown_all_sessions()
    assert _globals._live_sessions == []
