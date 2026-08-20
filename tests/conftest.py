"""Shared fixtures for tool unit/integration/regression tests."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest
from kaos.path import KaosPath


def pytest_configure(config: Any) -> None:
    """Register custom markers used across the test suite."""
    config.addinivalue_line(
        "markers",
        "benchmark: performance smoke tests for the FTS5 adoption plan "
        "(skip by default with '-m \"not benchmark\"')",
    )


# ── Temporary file fixtures ──────────────────────────────────────────────


@pytest.fixture
def work_dir() -> Any:
    """Temporary working directory with sample files."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print('hello')\n")
        (root / "src" / "utils.py").write_text("def add(a,b): return a+b\n")
        (root / "data.json").write_text('{"key": "value"}')
        (root / "README.md").write_text("# Test\n")
        yield root


@pytest.fixture
def sample_py_file(work_dir: Path) -> Path:
    """A .py file for Python tool tests."""
    p = work_dir / "test_script.py"
    p.write_text("import sys; print(sys.argv[1:])")
    return p


@pytest.fixture
def large_text_file(work_dir: Path) -> Path:
    """A 500-line text file for ReadFile truncation tests."""
    p = work_dir / "large.txt"
    p.write_text("\n".join(f"line {i:04d}" for i in range(500)))
    return p


@pytest.fixture
def sample_image(work_dir: Path) -> Path:
    """A small valid PNG image for ReadMediaFile tests."""
    from io import BytesIO
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(51, 102, 204))
    buf = BytesIO()
    img.save(buf, format="PNG")
    p = work_dir / "sample.png"
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def sample_avif(work_dir: Path) -> Path:
    """A minimal AVIF file for auto-convert tests."""
    # AVIF header bytes (not a valid image, but sufficient for format detection
    from kimi_cli.tools.file.utils import MEDIA_SNIFF_BYTES
    p = work_dir / "test.avif"
    # Write a minimal AVIF header + ftyp box
    p.write_bytes(b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00avif\x00\x00\x00\x00" + b"\x00" * MEDIA_SNIFF_BYTES)
    return p


# ── Mock session / runtime fixtures ──────────────────────────────────────


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    session.custom_config = {"provider_dict": {"name": "mock"}}
    session.id = "test-session-id"
    return session


@pytest.fixture
def mock_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = Path(tempfile.gettempdir())
    runtime.additional_dirs = []
    runtime.skills_dirs = []
    runtime.environment.os_kind = "Linux"
    return runtime


@pytest.fixture
def mock_approval() -> MagicMock:
    approval = AsyncMock()
    approval.request = AsyncMock(return_value=True)
    return approval


@pytest.fixture
def mock_soul() -> MagicMock:
    soul = MagicMock()
    soul.status.context_usage = 0.5
    soul.status.context_tokens = 50000
    soul.status.max_context_tokens = 100000
    soul.status.step_count = 100
    soul.custom_data = {}
    return soul


@pytest.fixture
def running_task(mock_session: MagicMock) -> str:
    """Register a mock running background task and return its ID."""
    from unittest.mock import AsyncMock
    from kimix.tools.background.utils import add_task, BackgroundStream

    stream = BackgroundStream()
    stream._started = True

    async def fake_worker(q: Any) -> tuple[bool, int | None]:
        import queue, time
        q.put("partial output")
        time.sleep(30)
        return True, 0

    import queue
    q = queue.Queue()
    q.put("partial output")
    stream._queue = q
    stream._worker = fake_worker
    stream._started = True
    stream._completed_event = type('Ev', (), {'is_set': lambda: False})()

    task_id = "run_test_42"
    add_task(mock_session, task_id, stream)
    return task_id


@pytest.fixture
def temp_work_dir() -> KaosPath:
    """Create a temp directory and return it as KaosPath."""
    td = tempfile.mkdtemp()
    yield KaosPath(td)
    import shutil
    shutil.rmtree(td, ignore_errors=True)


# ── Tool fixtures (when actual tool instances are needed) ────────────────


@pytest.fixture
def bash_tool(mock_session: MagicMock):
    from kimix.tools.file.bash.bash_tool import Bash
    return Bash(session=mock_session)


@pytest.fixture
def read_file_tool(mock_runtime: MagicMock, mock_session: MagicMock):
    from kimi_cli.tools.file.read import ReadFile
    return ReadFile(runtime=mock_runtime, session=mock_session)


@pytest.fixture
def read_media_file_tool(mock_runtime: MagicMock):
    from kimi_cli.tools.file.read_media import ReadMediaFile
    return ReadMediaFile(runtime=mock_runtime)


@pytest.fixture
def write_file_tool(mock_runtime: MagicMock, mock_approval: MagicMock, mock_session: MagicMock):
    from kimi_cli.tools.file.write import WriteFile
    return WriteFile(runtime=mock_runtime, approval=mock_approval, session=mock_session)


@pytest.fixture
def edit_file_tool(mock_runtime: MagicMock, mock_approval: MagicMock, mock_session: MagicMock):
    from kimi_cli.tools.file.replace import EditFile
    return EditFile(runtime=mock_runtime, approval=mock_approval, session=mock_session)


@pytest.fixture
def glob_tool(mock_runtime: MagicMock):
    from kimi_cli.tools.file.glob import Glob
    return Glob(runtime=mock_runtime)


@pytest.fixture
def grep_tool(mock_runtime: MagicMock):
    from kimi_cli.tools.file.grep_local import Grep
    return Grep(runtime=mock_runtime)


@pytest.fixture
def todo_list_tool(mock_runtime: MagicMock):
    from kimi_cli.tools.todo import TodoList
    return TodoList(runtime=mock_runtime)


@pytest.fixture
def fetch_url_tool():
    from kimi_cli.tools.web.fetch import fetch_url
    return fetch_url(config=MagicMock(), runtime=MagicMock())
