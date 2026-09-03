"""Exercise the Codex OAuth commands without opening a browser or using credentials."""

from collections.abc import AsyncIterator
from unittest.mock import Mock

import orjson
import pytest
from typer.testing import CliRunner

from kimi_cli.auth import oauth
from kimi_cli.auth.oauth import OAuthEvent
from kimi_cli.cli import cli
from kimi_cli.config import Config


@pytest.fixture
def codex_cli_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    config = Config()
    monkeypatch.setattr("kimi_cli.config.load_config", lambda: config)
    return config


@pytest.mark.parametrize("command", ["login", "logout"])
@pytest.mark.parametrize("provider_name", ["codex", "CoDeX"])
@pytest.mark.parametrize("json_output", [False, True])
def test_codex_command_dispatches_and_emits_events(
    monkeypatch: pytest.MonkeyPatch,
    codex_cli_config: Config,
    command: str,
    provider_name: str,
    json_output: bool,
) -> None:
    received_configs: list[Config] = []
    event = OAuthEvent("success", f"Codex {command} completed.")

    async def operation(config: Config) -> AsyncIterator[OAuthEvent]:
        received_configs.append(config)
        yield event

    monkeypatch.setattr(oauth, f"{command}_codex", operation)
    args = [command, provider_name]
    if json_output:
        args.append("--json")

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.output
    assert len(received_configs) == 1
    assert received_configs[0] is codex_cli_config
    if json_output:
        assert [orjson.loads(line) for line in result.stdout.splitlines()] == [
            orjson.loads(event.json)
        ]
    else:
        assert event.message in result.stdout


@pytest.mark.parametrize("command", ["login", "logout"])
@pytest.mark.parametrize("json_output", [False, True])
def test_codex_command_reports_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
    codex_cli_config: Config,
    command: str,
    json_output: bool,
) -> None:
    event = OAuthEvent("error", f"Codex {command} failed: credential_store_unavailable")

    async def operation(config: Config) -> AsyncIterator[OAuthEvent]:
        assert config is codex_cli_config
        yield event

    monkeypatch.setattr(oauth, f"{command}_codex", operation)
    args = [command, "codex"]
    if json_output:
        args.append("--json")

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    if json_output:
        assert orjson.loads(result.stdout) == orjson.loads(event.json)
    else:
        assert event.message in result.stdout


def test_codex_login_rejects_xai_api_key_option(monkeypatch: pytest.MonkeyPatch) -> None:
    load_config = Mock()
    login_codex = Mock()
    monkeypatch.setattr("kimi_cli.config.load_config", load_config)
    monkeypatch.setattr(oauth, "login_codex", login_codex)

    result = CliRunner().invoke(cli, ["login", "codex", "--api-key", "test-key"])

    assert result.exit_code == 1
    assert "--api-key is only supported for xai." in result.stderr
    load_config.assert_not_called()
    login_codex.assert_not_called()
