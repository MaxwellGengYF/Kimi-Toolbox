"""A Qt object must never be destroyed by Python's cyclic collector.

Qt objects belong to one thread and may only be destroyed on it. PySide6 destroys
the C++ half of an *unparented* Qt object when its Python wrapper dies, and a wrapper
that is reachable only through a reference cycle dies inside ``gc.collect()`` -- which
CPython runs on whichever thread crossed an allocation threshold. This application
keeps a worker thread named ``kimix-bridge`` allocating continuously, so that thread
was regularly the one collecting, and the suite crashed roughly one run in eight with
access violations and heap corruption in exactly that thread.

So the invariant is not "do not leak" and not "collect on the right thread". It is:

    nothing reachable only through a cycle may still own a C++ Qt object.

Every wrapper whose C++ half is already gone, or whose C++ half is owned by a Qt
parent, is safe to free from any thread at any time. These tests assert that property
directly rather than trying to provoke the crash, which is probabilistic and only
reproduces under a real thread.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtCore import QEvent, QRect
from PySide6.QtGui import QTextCursor, QTextDocument
from PySide6.QtWidgets import QApplication

from kimix_gui.qt.transcript import Transcript
from kimix_gui.qt.transcript_cards import DOCUMENT_CACHE_SIZE

from .transcript_helpers import append_text

MARKDOWN = "a *body* with a [link](https://example.test) and `code`\n\n- one\n- two\n"


@pytest.fixture
def collected() -> Iterator[Callable[[], list[object]]]:
    """Return every object the cyclic collector reclaims, without reclaiming it.

    ``gc.DEBUG_SAVEALL`` diverts collected objects into ``gc.garbage`` instead of
    freeing them, which is the only way to look at them. It also means the cycles
    stay alive, so the teardown has to drop the list *and* collect again with the
    flag off, or everything the suite ever cycled would accumulate here.
    """

    def drain() -> None:
        gc.garbage.clear()
        gc.set_debug(0)
        gc.collect()

    def take() -> list[object]:
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        found = list(gc.garbage)
        drain()
        return found

    drain()
    try:
        yield take
    finally:
        drain()


def _owns_a_cpp_object(obj: object) -> bool:
    """Would freeing this wrapper destroy a Qt object?"""
    if isinstance(obj, type):
        return False
    module = type(obj).__module__ or ""
    if not module.startswith(("PySide6", "kimix_gui.qt")):
        return False
    try:
        return shiboken6.Shiboken.isValid(obj) and shiboken6.Shiboken.ownedByPython(obj)
    except TypeError, RuntimeError:
        return False  # not a wrapper at all, so pure Python and thread-agnostic


def _dangerous(objects: list[object]) -> list[str]:
    return sorted(type(obj).__name__ for obj in objects if _owns_a_cpp_object(obj))


def _fill(transcript: Transcript, rows: int) -> None:
    """Append rows and force a document for each, which is what painting does."""
    for index in range(rows):
        append_text(transcript, "assistant", f"row {index}: {MARKDOWN}", markdown=True)
    model = transcript.model()
    for row in range(model.rowCount()):
        transcript._delegate.document_for(QRect(0, 0, 600, 0), model.index(row, 0))


def _live(documents: list[QTextDocument]) -> int:
    return sum(1 for document in documents if shiboken6.Shiboken.isValid(document))


# --- The property that makes a cached document safe --------------------------------


def test_a_cached_document_is_owned_by_cpp(qtbot) -> None:
    """The delegate parents its documents, so Python is not the one who deletes them."""
    transcript = Transcript(max_chars=8192)
    qtbot.addWidget(transcript)
    _fill(transcript, 4)

    documents = list(transcript.bodies.documents.values())
    assert documents
    for document in documents:
        assert shiboken6.Shiboken.isValid(document)
        assert not shiboken6.Shiboken.ownedByPython(document)
        assert document.parent() is transcript.bodies


def test_an_evicted_document_is_destroyed_rather_than_leaked(qtbot) -> None:
    """C++ ownership would keep evicted documents alive, so eviction has to delete.

    ``deleteLater`` is what makes this the GUI thread's job. The observable effect is
    that the C++ object is gone once the deferred deletions are delivered, which also
    rules out the leak that plain parenting would have introduced.
    """
    transcript = Transcript(max_chars=1 << 20)
    qtbot.addWidget(transcript)
    _fill(transcript, 4)
    first = list(transcript.bodies.documents.values())
    assert _live(first) == len(first)

    _fill(transcript, DOCUMENT_CACHE_SIZE * 2)
    assert len(transcript.bodies.documents) <= DOCUMENT_CACHE_SIZE
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert _live(first) == 0, "evicted documents outlived their eviction"
    assert _live(list(transcript.bodies.documents.values())) == len(transcript.bodies.documents)


def test_a_reference_held_past_eviction_is_already_harmless(qtbot) -> None:
    """Eviction destroys the C++ object rather than handing ownership back to Python.

    ``document_for`` returns the document, so a caller can be holding a wrapper when
    the cache evicts it. Dropping the parent there would make that wrapper Python-owned
    again and put the whole problem back, one object at a time. ``deleteLater`` instead
    leaves the caller with a dangling wrapper, which is free to die anywhere.
    """
    transcript = Transcript(max_chars=1 << 20)
    qtbot.addWidget(transcript)
    _fill(transcript, 4)
    kept = next(iter(transcript.bodies.documents.values()))

    _fill(transcript, DOCUMENT_CACHE_SIZE * 2)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert not shiboken6.Shiboken.isValid(kept)


def test_invalidating_the_cache_destroys_the_documents(qtbot) -> None:
    """The font and theme handlers drop the whole cache; that must not leak either."""
    transcript = Transcript(max_chars=8192)
    qtbot.addWidget(transcript)
    _fill(transcript, 6)
    documents = list(transcript.bodies.documents.values())
    assert _live(documents) == len(documents)

    transcript.bodies.invalidate()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert transcript.bodies.documents == {}
    assert _live(documents) == 0


# --- The invariant itself, over the flows that used to break it --------------------


def test_dropping_a_transcript_leaves_nothing_dangerous_to_collect(qtbot, collected) -> None:
    """``MainWindow.show_chat`` rebuilds the whole chat view on every session open.

    That is the flow that made this a product problem rather than a test-only one: the
    delegate, the view and every ``Retranslator`` statement reference each other, so a
    dropped transcript reaches the collector as a cycle, carrying whatever its document
    cache still held. There were thirty-two of those, all owned by Python.
    """
    transcript = Transcript(max_chars=8192)
    transcript.resize(600, 400)
    _fill(transcript, DOCUMENT_CACHE_SIZE * 2)
    assert len(transcript.bodies.documents) == DOCUMENT_CACHE_SIZE

    transcript.deleteLater()
    del transcript
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    garbage = collected()
    assert any(type(obj).__name__ == "Transcript" for obj in garbage), (
        "the transcript did not reach the cyclic collector, so this test proves nothing"
    )
    assert _dangerous(garbage) == []


def test_dropping_a_home_view_leaves_nothing_dangerous_to_collect(
    qtbot, collected, tmp_path: Path
) -> None:
    """The session list rebuilds every row on every keystroke, and it is a lot of rows."""
    from kimix_gui.llm import resolved_provider_file
    from kimix_gui.qt.home_view import HomeView
    from kimix_gui.session_index import SessionSummary

    config = tmp_path / "provider.json"
    config.write_text(
        '{"model": "m", "name": "M", "max_context_size": 1000, '
        '"type": "openai_legacy", "url": "https://example.test/v1", "api_key": "k"}',
        encoding="utf-8",
    )
    home = HomeView(
        tmp_path,
        default_config=resolved_provider_file(config),
        session_config_loader=lambda _id: None,
    )
    home.resize(900, 600)
    summaries = [
        SessionSummary(id=f"s-{i}", title=f"session {i}", updated_at=1.0 + i) for i in range(20)
    ]
    for _ in range(5):
        home.show_sessions(summaries)

    home.deleteLater()
    del home
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    garbage = collected()
    assert any(type(obj).__name__ == "SessionRow" for obj in garbage), (
        "no session row reached the cyclic collector, so this test proves nothing"
    )
    assert _dangerous(garbage) == []


def test_a_transcript_in_use_leaves_nothing_dangerous_to_collect(qtbot, collected) -> None:
    """Eviction happens constantly while the view is alive, not only when it dies."""
    transcript = Transcript(max_chars=8192)
    qtbot.addWidget(transcript)
    _fill(transcript, DOCUMENT_CACHE_SIZE * 3)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert _dangerous(collected()) == []


# --- Why an unparented QTextDocument elsewhere is still fine -----------------------


def test_a_short_lived_document_never_reaches_the_collector(qtbot, collected) -> None:
    """``qt/paint.markdown_plain_text`` builds an unparented document and drops it.

    That is safe for a different reason: a bare ``QTextDocument`` does not take part
    in a cycle, so its wrapper dies by reference count, on the thread that dropped it.
    Measured against every operation the delegate performs on one -- ``setMarkdown``,
    ``setTextWidth``, ``documentLayout()``, a ``QTextCursor`` over it -- and none of
    them introduces one. The rule this protects is therefore about *storing* an
    unparented Qt object, not about creating one.
    """
    document = QTextDocument()
    document.setMarkdown(MARKDOWN)
    document.setTextWidth(300)
    document.documentLayout()
    QTextCursor(document).select(QTextCursor.SelectionType.Document)

    assert "link" in document.toPlainText()

    del document
    assert not any(type(obj) is QTextDocument for obj in collected())
