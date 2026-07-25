"""Tests for Defects 1.1-1.6: Bash/Powershell/Run tool improvements."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kimix.tools.file.bash.bash_tool import BashParams
from kimix.tools.file.run import RunParams


# ── Defect 1.1: Parameter aliases ────────────────────────────────────────


class TestBashParamAliases:
    @pytest.mark.parametrize("kw,expected_cmd", [
        ({"cmd": "echo hello"}, "echo hello"),
        ({"command": "echo hello"}, "echo hello"),
    ])
    def test_both_names_accepted(self, kw: dict, expected_cmd: str) -> None:
        params = BashParams(**kw)
        assert params.cmd == expected_cmd

    def test_last_wins_when_both_given(self) -> None:
        params = BashParams(cmd="first", command="second")
        assert params.cmd == "second"

    def test_default_value_preserved(self) -> None:
        # Use mode='interactive' to allow empty cmd
        from kimix.tools.file.bash.bash_tool import BashParams
        params = BashParams(mode="interactive")
        assert params.cmd == ""
        assert params.timeout == 30

class TestRunParamAliases:
    @pytest.mark.parametrize("kw,expected", [
        ({"command": "git status"}, "git status"),
        ({"cmd": "git status"}, "git status"),
    ])
    def test_both_names_accepted(self, kw: dict, expected: str) -> None:
        params = RunParams(**kw)
        assert params.command == expected


# ── Defect 1.2: Mode parameter ──────────────────────────────────────────


class TestBashModeParameter:
    def test_mode_execute_is_default(self) -> None:
        params = BashParams(cmd="echo hello")
        assert params.mode == "execute"

    def test_mode_send_without_task_id_valid(self) -> None:
        params = BashParams(cmd="echo hi", mode="send")
        assert params.mode == "send"

    def test_mode_send_with_task_id(self) -> None:
        params = BashParams(cmd="input text", mode="send", task_id="bash_abc")
        assert params.mode == "send"
        assert params.task_id == "bash_abc"

    def test_run_alias_execute(self) -> None:
        params = BashParams(cmd="echo hi", mode="run")
        assert params.mode == "execute"

    def test_background_alias_send(self) -> None:
        params = BashParams(cmd="echo hi", mode="background")
        assert params.mode == "send"

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BashParams(cmd="echo hi", mode="exec")

    def test_mode_description_documents_aliases(self) -> None:
        desc = BashParams.model_fields["mode"].description or ""
        assert "run" in desc
        assert "background" in desc

    def test_task_id_continues_session_regardless_of_mode(self) -> None:
        params = BashParams(cmd="input", mode="execute", task_id="bash_abc")
        assert params.mode == "execute"
        assert params.task_id == "bash_abc"


# ── Defect 1.4: deduplicate_output ───────────────────────────────────────


class TestDeduplicateOutputRename:
    def test_new_name_works(self) -> None:
        params = BashParams(cmd="git log", deduplicate_output=True)
        assert params.deduplicate_output is True

    def test_old_name_still_works(self) -> None:
        params = BashParams(cmd="git log", token_kill=False)
        assert params.deduplicate_output is False

    def test_default_is_true(self) -> None:
        params = BashParams(cmd="echo test")
        assert params.deduplicate_output is True


# ── Defect 1.5: Timeout consistency ──────────────────────────────────────


class TestTimeoutConsistency:
    @pytest.mark.parametrize("params_cls,expected_default", [
        (BashParams, 30),
        (RunParams, 30),
    ])
    def test_consistent_default_timeout(self, params_cls, expected_default: int) -> None:
        # Construct with the appropriate field name
        if params_cls == RunParams:
            p = params_cls(command="test")
        else:
            p = params_cls(cmd="test")
        assert p.timeout == expected_default

    def test_timeout_min_is_1(self) -> None:
        p = BashParams(cmd="echo quick", timeout=1)
        assert p.timeout == 1

    def test_timeout_max_is_900(self) -> None:
        p = BashParams(cmd="long build", timeout=900)
        assert p.timeout == 900

    def test_timeout_below_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BashParams(cmd="test", timeout=0)

    def test_timeout_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BashParams(cmd="test", timeout=901)


# ── Defect 1.6: max_lines ────────────────────────────────────────────────


class TestMaxLinesTruncation:
    def test_max_lines_field(self) -> None:
        params = BashParams(cmd="test", max_lines=50)
        assert params.max_lines == 50

    def test_max_lines_none_means_unlimited(self) -> None:
        params = BashParams(cmd="test", max_lines=None)
        assert params.max_lines is None

    def test_max_lines_min_enforced(self) -> None:
        with pytest.raises(ValidationError):
            BashParams(cmd="test", max_lines=2)
