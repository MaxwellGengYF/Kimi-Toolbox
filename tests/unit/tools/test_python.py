"""Tests for Python tool: code/file split and unified mode."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kimix.tools.py import Params as PythonParams


# ── Defect 2.1: code vs file split ───────────────────────────────────────


class TestPythonCodeUnified:
    def test_code_only_accepted(self) -> None:
        params = PythonParams(code="print(1+1)")
        assert params.code == "print(1+1)"

    def test_neither_rejected_unless_interactive(self) -> None:
        with pytest.raises(ValidationError, match="code` must be provided"):
            PythonParams()

    def test_neither_ok_when_interactive(self) -> None:
        params = PythonParams(interactive=True)
        assert params.code == ""

    def test_file_alias_accepted(self) -> None:
        """file=... alias maps to the code field."""
        params = PythonParams(file="script.py")
        assert params.code == "script.py"

    def test_code_with_py_extension_accepted(self) -> None:
        """code ending with .py is accepted as a string."""
        params = PythonParams(code="my_script.py")
        assert params.code == "my_script.py"

    def test_code_with_inline_code_accepted(self) -> None:
        """Arbitrary inline code strings are accepted."""
        params = PythonParams(code="x = 1\nprint(x)")
        assert params.code == "x = 1\nprint(x)"


# ── Defect 2.2: Unified mode parameter ──────────────────────────────────


class TestPythonUnifiedMode:
    @pytest.mark.parametrize("mode", ["run", "background", "interactive"])
    def test_all_modes_accepted(self, mode: str) -> None:
        params = PythonParams(code="print(1)", mode=mode)
        assert params.mode == mode

    def test_legacy_interactive_bool_still_works(self) -> None:
        params = PythonParams(interactive=True)
        assert params.mode == "interactive"

    def test_legacy_run_in_background_bool_still_works(self) -> None:
        params = PythonParams(code="print(1)", run_in_background=True)
        assert params.mode == "background"

    def test_both_legacy_bools_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Cannot set both"):
            PythonParams(code="print(1)", interactive=True, run_in_background=True)


