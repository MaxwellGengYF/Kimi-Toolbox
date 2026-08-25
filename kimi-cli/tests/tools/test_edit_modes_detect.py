"""Tests for edit mode auto-detection."""

from __future__ import annotations

import pytest

from kimi_cli.tools.file.edit.params import EditParams, PatchEntry
from kimi_cli.tools.file.hash_line import compute_line_hash


def _line1_hash(content: str) -> str:
    line = content.splitlines()[0] if content.splitlines() else ""
    return compute_line_hash(1, line, None)


def test_detect_explicit_mode_wins():
    params = EditParams(mode="replace", input="[path#AB]\nPUT 1.=1:\n+x\n")
    assert params.resolved_mode == "replace"


def test_detect_apply_patch_from_input():
    params = EditParams(input="*** Begin Patch\n*** Add File: x.txt\n+hi\n*** End Patch\n")
    assert params.resolved_mode == "apply_patch"


def test_detect_hashline_from_input():
    params = EditParams(input="[path#AB]\nPUT 1.=1:\n+x\n")
    assert params.resolved_mode == "hashline"


def test_detect_sloppy_from_input():
    params = EditParams(input="§path\n⟪old│new⟫\n")
    assert params.resolved_mode == "sloppy"


def test_detect_patch_from_edits():
    params = EditParams(path="x.txt", edits=[{"op": "update", "diff": "@@ -1,1 +1,1 @@\n-old\n+new\n"}])
    assert params.resolved_mode == "patch"


def test_detect_replace_from_old_new():
    params = EditParams(path="x.txt", old_string="old", new_string="new")
    assert params.resolved_mode == "replace"


def test_detect_replace_from_edits():
    params = EditParams(
        path="x.txt",
        edits=[{"old_string": "old", "new_string": "new"}],
    )
    assert params.resolved_mode == "replace"


def test_detect_ambiguous_raises():
    with pytest.raises(ValueError):
        EditParams(path="x.txt")
