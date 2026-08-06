"""Tests for tools/sync_native.py path resolution.

Verifies that the kimix-base source repo is located without any baked-in
absolute path: the ``$KIMIX_BASE`` env var overrides the default sibling
layout, so the script works on any platform/checkout location.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")

sys.path.insert(0, _TOOLS_DIR)

import sync_native  # noqa: E402


def test_kimix_base_defaults_to_sibling(monkeypatch):
    """Without $KIMIX_BASE the source repo is the sibling kimix-base."""
    monkeypatch.delenv("KIMIX_BASE", raising=False)
    assert sync_native._kimix_base() == os.path.join(
        os.path.dirname(_REPO_ROOT), "kimix-base"
    )
    assert sync_native._kimix_base_bin() == os.path.join(
        os.path.dirname(_REPO_ROOT), "kimix-base", "bin"
    )


def test_kimix_base_env_override(monkeypatch):
    """$KIMIX_BASE re-points the source repo (no absolute path baked in)."""
    fake = os.path.join("other", "kimix-base")
    monkeypatch.setenv("KIMIX_BASE", fake)
    assert sync_native._kimix_base() == fake
    assert sync_native._kimix_base_bin() == os.path.join(fake, "bin")


def test_source_dirs_uses_override(tmp_path, monkeypatch):
    """_source_dirs() honors $KIMIX_BASE when a valid build exists there."""
    fake = tmp_path / "kimix-base"
    release = fake / "bin" / "release"
    release.mkdir(parents=True)
    (release / "runtime_py.pyd").write_bytes(b"x")
    monkeypatch.setenv("KIMIX_BASE", str(fake))
    dirs = sync_native._source_dirs("release")
    assert dirs == [str(release)]


def test_source_dirs_empty_when_override_missing(tmp_path, monkeypatch):
    """A $KIMIX_BASE pointing at a dir without builds yields no sources."""
    monkeypatch.setenv("KIMIX_BASE", str(tmp_path / "missing"))
    assert sync_native._source_dirs("release") == []
    with pytest.raises(FileNotFoundError):
        sync_native.sync(mode="release", dest=str(tmp_path / "dest"))
