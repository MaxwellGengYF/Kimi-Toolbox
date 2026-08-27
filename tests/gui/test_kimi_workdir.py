from __future__ import annotations

import os
from pathlib import Path

import pytest
from kaos import get_current_kaos
from kimi_cli.metadata import Metadata, WorkDirMeta, load_metadata, save_metadata

from kimix_gui.kimi_workdir import WorkDirCaseConflict, resolve_kimi_work_dir

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows path casing only")


def _case_variants(path: Path) -> tuple[str, str]:
    canonical = str(path.resolve())
    alternate = canonical[0].swapcase() + canonical[1:]
    return canonical, alternate


def _metadata_with_path(path: str) -> Metadata:
    return Metadata(work_dirs=[WorkDirMeta(path=path, kaos=get_current_kaos().name)])


def test_repairs_metadata_when_only_canonical_hash_has_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    share_dir = tmp_path / "share"
    monkeypatch.setenv("KIMI_SHARE_DIR", str(share_dir))
    project = tmp_path / "Project"
    project.mkdir()
    canonical, alternate = _case_variants(project)
    metadata = _metadata_with_path(alternate)
    stored_meta = metadata.work_dirs[0]
    stored_dir = stored_meta.sessions_dir
    canonical_meta = WorkDirMeta(path=canonical, kaos=stored_meta.kaos)
    session_dir = canonical_meta.sessions_dir / "session-1"
    session_dir.mkdir()
    save_metadata(metadata)

    resolved = resolve_kimi_work_dir(project)

    assert str(resolved) == canonical
    assert load_metadata().work_dirs[0].path == canonical
    assert session_dir.is_dir()
    assert list(stored_dir.iterdir()) == []


def test_keeps_stored_case_when_only_stored_hash_has_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    project = tmp_path / "Project"
    project.mkdir()
    _canonical, alternate = _case_variants(project)
    metadata = _metadata_with_path(alternate)
    (metadata.work_dirs[0].sessions_dir / "session-1").mkdir()
    save_metadata(metadata)

    resolved = resolve_kimi_work_dir(project)

    assert str(resolved) == alternate
    assert load_metadata().work_dirs[0].path == alternate


def test_rejects_case_variants_that_both_contain_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    project = tmp_path / "Project"
    project.mkdir()
    canonical, alternate = _case_variants(project)
    metadata = _metadata_with_path(alternate)
    stored_meta = metadata.work_dirs[0]
    (stored_meta.sessions_dir / "stored-session").mkdir()
    canonical_meta = WorkDirMeta(path=canonical, kaos=stored_meta.kaos)
    (canonical_meta.sessions_dir / "canonical-session").mkdir()
    save_metadata(metadata)

    with pytest.raises(WorkDirCaseConflict):
        resolve_kimi_work_dir(project)
