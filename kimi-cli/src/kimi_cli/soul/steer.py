"""Mid-stream steering API for a running agent loop.

``Steer`` is the upper-level entry point (used by ``kimix.utils.prompt``) to
push a follow-up message into a **running** agent loop no matter what state the
session is in — including while the model is mid-stream printing reasoning
(``ThinkPart``) or text (``TextPart``).

The injection happens at the message/context layer, above the providers, so it
works with every backend provider (openai_legacy, openai_responses, anthropic,
gemini/google_genai, kimi, mock/echo).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from kosong.message import ContentPart

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul


class Steer:
    """Push a message into a running :class:`~kimi_cli.soul.kimisoul.KimiSoul`.

    Construct directly from a ``KimiSoul`` or resolve one from a session object
    with :meth:`from_session`.
    """

    def __init__(self, soul: KimiSoul) -> None:
        self._soul: KimiSoul | None = soul
        self._loop = self._capture_loop(soul)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_session(cls, session: object) -> Steer | None:
        """Resolve a ``KimiSoul`` from a session object.

        Resolution order:

        1. ``session._cli.soul`` (``kimi_agent_sdk.Session``)
        2. ``session.soul``
        3. ``session._soul``

        Returns ``None`` when no ``KimiSoul`` can be found.
        """
        soul = getattr(session, "_cli", None)
        if soul is not None:
            soul = getattr(soul, "soul", None)
        if soul is None:
            soul = getattr(session, "soul", None)
        if soul is None:
            soul = getattr(session, "_soul", None)
        if soul is None:
            return None
        return cls(soul)

    @staticmethod
    def _capture_loop(soul: KimiSoul) -> asyncio.AbstractEventLoop | None:
        """Capture the running loop when constructed inside one.

        Falls back to the soul's own captured loop when available (some soul
        wrappers expose one); ``None`` when constructed outside any loop.
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass
        return getattr(soul, "_loop", None)

    # ------------------------------------------------------------------ #
    # Push
    # ------------------------------------------------------------------ #

    async def push(self, content: str | list[ContentPart]) -> bool:
        """Push *content* into the running agent loop.

        Returns ``True`` when the content was delivered to a running soul.
        When the soul is not running, the content is still queued (it will be
        discarded as a stale steer at the next turn init, matching existing
        behavior) and ``False`` is returned.

        Thread safety: when ``push`` is called from a different event loop or
        thread, the enqueue is marshalled through the soul's loop via
        ``run_coroutine_threadsafe`` so the asyncio primitives are only touched
        on their owning loop.
        """
        soul = self._soul
        if soul is None:
            return False
        if not soul.is_running():
            # Queue anyway — it will be discarded as a stale steer at the next
            # turn init (matching existing behavior).
            soul.steer(content)
            return False
        loop = self._loop
        if loop is None or loop is asyncio.get_running_loop():
            await soul.request_steer(content)
        else:
            future = asyncio.run_coroutine_threadsafe(soul.request_steer(content), loop)
            await asyncio.wrap_future(future)
        return True

    def push_sync(self, content: str | list[ContentPart]) -> bool:
        """Synchronous wrapper around :meth:`push` for non-async callers.

        Only call this from a *different* thread than the one running the
        soul's event loop — blocking the loop's own thread would deadlock.
        When no usable loop is available, the content is queued best-effort and
        ``False`` is returned (not delivered).
        """
        soul = self._soul
        if soul is None:
            return False
        loop = self._loop
        if loop is None or loop.is_closed() or not loop.is_running():
            soul.steer(content)
            return False
        future = asyncio.run_coroutine_threadsafe(self.push(content), loop)
        return future.result()

    # ------------------------------------------------------------------ #
    # Queue introspection
    # ------------------------------------------------------------------ #

    def pending(self) -> int:
        """Number of steers currently queued but not yet consumed."""
        soul = self._soul
        if soul is None:
            return 0
        return soul._steer_queue.qsize()

    def clear(self) -> None:
        """Drain any pending steers without injecting them."""
        soul = self._soul
        if soul is None:
            return
        while not soul._steer_queue.empty():
            soul._steer_queue.get_nowait()

    def close(self) -> None:
        """Detach the reference to the soul."""
        self._soul = None
