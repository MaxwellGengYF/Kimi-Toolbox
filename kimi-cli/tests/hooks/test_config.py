import tomlkit

from kimi_cli.config import Config


def test_config_with_hooks():
    toml_str = """
[[hooks]]
event = "PreToolUse"
matcher = "Shell"
command = "echo ok"
timeout = 10

[[hooks]]
event = "PostToolUse"
matcher = "WriteFile"
command = "prettier --write"
"""
    data = tomlkit.parse(toml_str)
    config = Config.model_validate(data)
    assert len(config.hooks) == 2
    assert config.hooks[0].event == "PreToolUse"
    assert config.hooks[0].matcher == "Shell"
    assert config.hooks[1].timeout == 30


def test_config_without_hooks():
    config = Config.model_validate({})
    assert config.hooks == []
