"""Tests for kimix.tools.security: env scrubbing, output redaction, workdir validation."""

import pytest

from kimix.tools.security import (
    redact_sensitive_output,
    scrub_child_env,
    validate_workdir,
)


# ============================================================================
# scrub_child_env
# ============================================================================

class TestScrubChildEnv:
    @pytest.mark.parametrize(
        "name",
        [
            "AWS_ACCESS_KEY_ID",
            "GITHUB_TOKEN",
            "DB_PASSWORD",
            "MY_SECRET_KEY",
            "API_KEY",
            "WEBHOOK_URL",
            "BEARER_AUTH",
            "DSN_VALUE",
            "CREDENTIALS",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "slack_token",  # case-insensitive match
            "aws_access_key_id",
            "db_password",
        ],
    )
    def test_secret_named_vars_dropped(self, name: str) -> None:
        env = {name: "secret-value"}
        result = scrub_child_env(env)
        assert result == {}

    @pytest.mark.parametrize(
        "name",
        [
            "PATH",
            "HOME",
            "USERPROFILE",
            "VIRTUAL_ENV",
            "KIMIX_PYTHON_EXECUTABLE",
            "SSH_AUTH_SOCK",
            "UV_INDEX_URL",
            "PIP_INDEX_URL",
            "LANG",
            "TERM",
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
            "TZ",
            "Path",  # case-insensitive prefix match
            "PROGRAMFILES(X86)",
            "USERNAME",
        ],
    )
    def test_safe_named_vars_kept(self, name: str) -> None:
        env = {name: "value"}
        result = scrub_child_env(env)
        assert result == {name: "value"}

    def test_database_url_kept_name_only_matching(self) -> None:
        """DATABASE_URL has no secret substring in its *name*, so it is kept
        (documented behavior: scrubbing is name-only, values are untouched)."""
        env = {"DATABASE_URL": "postgres://user:hunter2@host/db"}
        result = scrub_child_env(env)
        assert result == {"DATABASE_URL": "postgres://user:hunter2@host/db"}

    def test_mixed_env_keeps_only_safe_and_plain(self) -> None:
        env = {
            "PATH": "/usr/bin",
            "AWS_ACCESS_KEY_ID": "AKIA-LEAK",
            "HOME": "/home/me",
            "DB_PASSWORD": "hunter2",
            "GIT_AUTHOR_NAME": "Ada",
            "USERPROFILE": "C:\\Users\\me",
            "GITHUB_TOKEN": "ghp_abcdef",
            "TZ": "UTC",
        }
        result = scrub_child_env(env)
        assert result == {
            "PATH": "/usr/bin",
            "HOME": "/home/me",
            "GIT_AUTHOR_NAME": "Ada",
            "USERPROFILE": "C:\\Users\\me",
            "TZ": "UTC",
        }

    def test_input_not_mutated(self) -> None:
        env = {"PATH": "/usr/bin", "AWS_ACCESS_KEY_ID": "AKIA-LEAK", "LANG": "en"}
        before = dict(env)
        scrub_child_env(env)
        assert env == before

    def test_returns_new_dict(self) -> None:
        env = {"PATH": "/usr/bin"}
        result = scrub_child_env(env)
        assert result is not env

    def test_empty_input(self) -> None:
        assert scrub_child_env({}) == {}

    def test_never_returns_none(self) -> None:
        result = scrub_child_env({"AWS_ACCESS_KEY_ID": "x", "DB_PASSWORD": "y"})
        assert result is not None
        assert result == {}


# ============================================================================
# validate_workdir
# ============================================================================

class TestValidateWorkdir:
    @pytest.mark.parametrize(
        "workdir",
        [
            None,
            "",
            "C:/Users/me",
            r"C:\Users\me",
            "/home/user",
            "~/projects",
            "my project",
            ".",
            "./sub/dir",
            r"D:\work\a b\c",
        ],
    )
    def test_valid(self, workdir: str | None) -> None:
        assert validate_workdir(workdir) is None

    @pytest.mark.parametrize(
        "workdir",
        [
            "a;b",
            "$HOME",
            "a|b",
            "a&b",
            "a>b",
            "a<b",
            "a`b",
            "a(b)",
            'a"b',
            "a'b",
            "a*b",
            "a?b",
            "a!b",
            "a{b}",
            "x\x00y",
            "x\ty",
        ],
    )
    def test_invalid(self, workdir: str) -> None:
        err = validate_workdir(workdir)
        assert err is not None
        assert "Invalid workdir" in err
        assert "not allowed" in err


# ============================================================================
# redact_sensitive_output (smoke — full matrix lives in test_output_enhance.py)
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

    def test_plain_text_unchanged(self) -> None:
        text = "build succeeded in 3.2s\nall tests passed"
        assert redact_sensitive_output(text) == text

    def test_empty_output(self) -> None:
        assert redact_sensitive_output("") == ""

    def test_aws_key_masked(self) -> None:
        assert "AKIA" not in redact_sensitive_output("AKIAIOSFODNN7EXAMPLE")


# ============================================================================
# Re-export smoke: the bash modules must still provide the moved names
# ============================================================================

def test_output_enhance_reexports_redaction() -> None:
    from kimix.tools.file.bash.output_enhance import (
        redact_sensitive_output as bash_redact,
    )

    assert bash_redact is redact_sensitive_output


def test_safety_reexports_validate_workdir() -> None:
    from kimix.tools.file.bash.safety import validate_workdir as bash_validate

    assert bash_validate is validate_workdir
