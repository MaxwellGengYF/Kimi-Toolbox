"""Tests for the shared tool temp folder lifecycle and /exit cleanup.

The tools write scratch files to ``.kimix_cache/tmp_<pid>``.  That folder must
be removed when the owning process finishes — and leftovers from processes
that were killed (which never run atexit) must be swept by later starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kimix.tools.common as common
from kimix.cli_impl import commands


def _make_temp_dir(base: Path, pid: int, name: str = "0.txt") -> Path:
    folder = base / f"tmp_{pid}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("leftover", encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# _cleanup_temp_folder (own process folder)
# ---------------------------------------------------------------------------


def test_cleanup_temp_folder_removes_own_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _make_temp_dir(tmp_path, 12345)
    monkeypatch.setattr(common, "_temp_folder_abs", folder)
    common._cleanup_temp_folder()
    assert not folder.exists()


def test_cleanup_temp_folder_missing_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "tmp_missing"
    monkeypatch.setattr(common, "_temp_folder_abs", missing)
    common._cleanup_temp_folder()  # must not raise


def test_rmtree_retry_removes_folder(tmp_path: Path) -> None:
    folder = _make_temp_dir(tmp_path, 2222)
    assert common._rmtree_retry(folder) is True
    assert not folder.exists()


# ---------------------------------------------------------------------------
# _cleanup_stale_temp_folders (dead-process leftovers)
# ---------------------------------------------------------------------------


def test_stale_sweep_removes_dead_and_keeps_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dead = _make_temp_dir(tmp_path, 424242)
    live = _make_temp_dir(tmp_path, 424243)
    monkeypatch.setattr(common, "_temp_folder_abs", tmp_path / "tmp_own")
    monkeypatch.setattr(common, "_pid_alive", lambda pid: pid == 424243)
    common._cleanup_stale_temp_folders()
    assert not dead.exists()
    assert live.exists()


def test_stale_sweep_reused_pid_with_fresh_files_keeps_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # PID is alive again (reused by an unrelated process) and the folder is
    # fresh -> keep, so we never delete a live process's working data.
    reused = _make_temp_dir(tmp_path, 777001)
    monkeypatch.setattr(common, "_temp_folder_abs", tmp_path / "tmp_own")
    monkeypatch.setattr(common, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(common, "_folder_has_fresh_file", lambda path: True)
    common._cleanup_stale_temp_folders()
    assert reused.exists()


def test_stale_sweep_reused_pid_with_stale_folder_removes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # PID appears alive but the folder has been untouched for over a day:
    # Windows recycled the PID, the owning process is long dead -> remove.
    stale = _make_temp_dir(tmp_path, 777002)
    monkeypatch.setattr(common, "_temp_folder_abs", tmp_path / "tmp_own")
    monkeypatch.setattr(common, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(common, "_folder_has_fresh_file", lambda path: False)
    common._cleanup_stale_temp_folders()
    assert not stale.exists()


def test_folder_has_fresh_file(tmp_path: Path) -> None:
    folder = _make_temp_dir(tmp_path, 555001)
    assert common._folder_has_fresh_file(folder) is True

    import os
    import time

    old = tmp_path / "tmp_555002"
    old.mkdir()
    (old / "0.txt").write_text("x", encoding="utf-8")
    old_time = time.time() - 48 * 60 * 60
    os.utime(old / "0.txt", (old_time, old_time))
    assert common._folder_has_fresh_file(old) is False


def test_stale_sweep_never_touches_own_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    own = _make_temp_dir(tmp_path, 111)
    monkeypatch.setattr(common, "_temp_folder_abs", own)
    monkeypatch.setattr(common, "_pid_alive", lambda pid: False)
    common._cleanup_stale_temp_folders()
    assert own.exists()


def test_stale_sweep_ignores_non_temp_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keep = tmp_path / "not_a_temp_folder"
    keep.mkdir()
    monkeypatch.setattr(common, "_temp_folder_abs", tmp_path / "tmp_own")
    monkeypatch.setattr(common, "_pid_alive", lambda pid: False)
    common._cleanup_stale_temp_folders()
    assert keep.exists()


def test_stale_sweep_missing_base_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common, "_temp_folder_abs", tmp_path / "nope" / "tmp_own")
    common._cleanup_stale_temp_folders()  # must not raise


# ---------------------------------------------------------------------------
# cleanup_temp_folder (public entry point)
# ---------------------------------------------------------------------------


def test_cleanup_temp_folder_public_calls_own_then_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(common, "_cleanup_temp_folder", lambda: calls.append("own"))
    monkeypatch.setattr(
        common, "_cleanup_stale_temp_folders", lambda: calls.append("stale")
    )
    common.cleanup_temp_folder()
    assert calls == ["own", "stale"]


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------


def test_pid_alive_current_process_is_true() -> None:
    assert common._pid_alive(os.getpid()) is True


def test_pid_alive_invalid_pid_is_false() -> None:
    # A PID far outside any valid range can never be running.
    assert common._pid_alive(2**31 + 12345) is False
    assert common._pid_alive(0) is False
    assert common._pid_alive(-1) is False


# ---------------------------------------------------------------------------
# /exit command integration
# ---------------------------------------------------------------------------


def test_cmd_exit_cleans_temp_folder_before_bye(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    closed: list[str] = []

    monkeypatch.setattr(commands, "get_default_session", lambda: object())
    monkeypatch.setattr(commands, "close_session", lambda s: closed.append("closed"))
    monkeypatch.setattr(
        commands,
        "_globals",
        SimpleNamespace(_default_session=object(), _default_role=object()),
    )

    cleaned: list[str] = []
    fake_module = SimpleNamespace(cleanup_temp_folder=lambda: cleaned.append("cleaned"))
    monkeypatch.setitem(sys.modules, "kimix.tools.common", fake_module)

    result = commands._cmd_exit(["exit"], [])
    out = capsys.readouterr().out

    assert result == (None, True)
    assert "bye!" in out
    assert closed == ["closed"]
    assert cleaned == ["cleaned"]


def test_cmd_exit_swallows_cleanup_errors(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(commands, "get_default_session", lambda: None)
    monkeypatch.setattr(
        commands,
        "_globals",
        SimpleNamespace(_default_session=None, _default_role=None),
    )

    def boom() -> None:
        raise RuntimeError("cleanup exploded")

    fake_module = SimpleNamespace(cleanup_temp_folder=boom)
    monkeypatch.setitem(sys.modules, "kimix.tools.common", fake_module)

    result = commands._cmd_exit(["exit"], [])
    out = capsys.readouterr().out

    assert result == (None, True)
    assert "bye!" in out
