"""Tests for Defects 7.1-7.3: EditFile improvements."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kimi_cli.tools.file.replace import Edit, Params as EditFileParams


class TestEditFileParams:
    def test_match_mode_default_exact(self) -> None:
        edit = Edit(old="foo", new="bar")
        assert edit.match_mode == "exact"

    def test_match_mode_exact_accepted(self) -> None:
        edit = Edit(old="foo", new="bar", match_mode="exact")
        assert edit.match_mode == "exact"

    def test_max_replacements_accepted(self) -> None:
        edit = Edit(old="foo", new="bar", replace_all=True, max_replacements=3)
        assert edit.max_replacements == 3

    def test_max_replacements_min_enforced(self) -> None:
        with pytest.raises(ValidationError):
            Edit(old="foo", new="bar", max_replacements=0)

    def test_edit_always_list(self) -> None:
        params = EditFileParams(path="f.txt", edit=[Edit(old="a", new="b")])
        assert isinstance(params.edit, list)
        assert len(params.edit) == 1

    def test_single_dict_auto_wrapped(self) -> None:
        params = EditFileParams(path="f.txt", edit={"old": "a", "new": "b"})
        assert isinstance(params.edit, list)
        assert len(params.edit) == 1

    def test_replace_all_default_false(self) -> None:
        edit = Edit(old="foo", new="bar")
        assert edit.replace_all is False
