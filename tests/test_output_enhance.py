"""Tests for output_enhance.py: exit-code semantics, failure hints, redaction."""

import pytest

from kimix.tools.file.bash.output_enhance import (
    annotate_failure,
    interpret_exit_code,
    redact_sensitive_output,
)


# ============================================================================
# interpret_exit_code
# ============================================================================

class TestInterpretExitCode:
    @pytest.mark.parametrize(
        ("command", "code", "expected"),
        [
            ("grep foo file.txt", 1, "No matches found"),
            ("egrep foo file.txt", 1, "No matches found"),
            ("fgrep foo file.txt", 1, "No matches found"),
            ("rg foo", 1, "No matches found"),
            ("ag foo", 1, "No matches found"),
            ("ack foo", 1, "No matches found"),
            ("diff a b", 1, "Files differ"),
            ("colordiff a b", 1, "Files differ"),
            ("find / -name x", 1, "Some directories were inaccessible"),
            ("test -f x", 1, "Condition evaluated to false"),
            ("[ -f x ]", 1, "Condition evaluated to false"),
            ("curl https://example.com", 6, "Could not resolve host"),
            ("curl https://example.com", 7, "Failed to connect to host"),
            ("curl https://example.com", 22, "HTTP error"),
            ("curl https://example.com", 28, "timed out"),
            ("git diff", 1, "Non-zero exit"),
            ("/usr/bin/git status", 1, "Non-zero exit"),
            ("FOO=1 git diff", 1, "Non-zero exit"),
            ("echo a | grep foo", 1, "No matches found"),
            ("a && b; git diff", 1, "Non-zero exit"),
        ],
    )
    def test_known_meanings(self, command: str, code: int, expected: str) -> None:
        meaning = interpret_exit_code(command, code)
        assert meaning is not None, f"expected a meaning for {command!r} exit {code}"
        assert expected in meaning

    @pytest.mark.parametrize(
        ("command", "code"),
        [
            ("echo hi", 0),
            ("echo hi", None),
            ("nosuchcommand_xyz", 127),
            ("python script.py", 2),
            ("git diff", 2),
            ("grep foo file", 2),
        ],
    )
    def test_no_meaning(self, command: str, code: int | None) -> None:
        assert interpret_exit_code(command, code) is None

    def test_unknown_command_returns_none(self) -> None:
        assert interpret_exit_code("frobnicate --all", 42) is None


# ============================================================================
# annotate_failure
# ============================================================================

class TestAnnotateFailure:
    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("bash: foo: command not found", "command was not found"),
            (
                "'foo' is not recognized as an internal or external command",
                "command was not found",
            ),
            ("cat: nope: No such file or directory", "does not exist"),
            (
                "ls: cannot access 'x': No such file or directory",
                "does not exist",
            ),
            (
                "Traceback (most recent call last):\nModuleNotFoundError: No module named 'requests'",
                "Python module requests is missing",
            ),
            ("Permission denied", "Permission denied"),
            (
                "mkdir: cannot create directory 'x': Permission denied",
                "Permission denied",
            ),
        ],
    )
    def test_hints(self, output: str, expected: str) -> None:
        hint = annotate_failure(output, "some command", 1)
        assert hint is not None
        assert expected in hint

    def test_no_hint_on_clean_output(self) -> None:
        assert annotate_failure("everything looks fine", "ls", 0) is None

    def test_no_hint_on_empty_output(self) -> None:
        assert annotate_failure("", "ls", 1) is None


# ============================================================================
# redact_sensitive_output
# ============================================================================

_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


class TestRedactSensitiveOutput:
    def test_jwt_masked(self) -> None:
        out = redact_sensitive_output(_JWT)
        assert "[REDACTED]" in out
        assert "eyJ" not in out

    def test_pem_private_key_masked(self) -> None:
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA1234567890ABCDEF\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = redact_sensitive_output(pem)
        assert "[REDACTED]" in out
        assert "PRIVATE KEY" not in out

    def test_github_token_masked(self) -> None:
        out = redact_sensitive_output("token=ghp_1234567890123456789012")
        assert "ghp_" not in out
        assert "[REDACTED]" in out

    def test_github_pat_masked(self) -> None:
        out = redact_sensitive_output("github_pat_1234567890_ABCDEFGHIJKLMNOPQRST")
        assert "github_pat" not in out
        assert "[REDACTED]" in out

    def test_gitlab_token_masked(self) -> None:
        out = redact_sensitive_output("glpat-12345678901234567890")
        assert "[REDACTED]" in out
        assert "glpat" not in out

    def test_aws_key_masked(self) -> None:
        out = redact_sensitive_output("AKIAIOSFODNN7EXAMPLE")
        assert "[REDACTED]" in out
        assert "AKIA" not in out

    def test_auth_header_masked(self) -> None:
        out = redact_sensitive_output("Authorization: Bearer abcdefgh1234")
        assert "abcdefgh1234" not in out
        assert "authorization" not in out.lower()

    def test_api_key_header_masked(self) -> None:
        out = redact_sensitive_output("x-api-key: secret123")
        assert "secret123" not in out

    def test_url_userinfo_masked_keeps_scheme(self) -> None:
        out = redact_sensitive_output("https://user:pass123@example.com/path")
        assert "[REDACTED]" in out
        assert "pass123" not in out
        assert out.startswith("https://")

    def test_password_assignment_masked(self) -> None:
        out = redact_sensitive_output("password=hunter2secret")
        assert "hunter2secret" not in out
        assert "[REDACTED]" in out

    def test_short_password_not_masked(self) -> None:
        out = redact_sensitive_output("password=x")
        assert "password=x" in out

    def test_bearer_token_masked(self) -> None:
        out = redact_sensitive_output("token value: Bearer abcdefghijklmnopqrstuvwxyz012345")
        assert "[REDACTED]" in out

    def test_plain_text_unchanged(self) -> None:
        text = "build succeeded in 3.2s\nall tests passed"
        assert redact_sensitive_output(text) == text

    def test_empty_output(self) -> None:
        assert redact_sensitive_output("") == ""
