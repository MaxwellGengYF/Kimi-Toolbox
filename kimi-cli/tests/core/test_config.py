from __future__ import annotations

import pytest

from kimi_cli.config import (
    get_default_config,
    load_config,
    load_config_from_string,
)
from kimi_cli.exception import ConfigError


def test_load_config_text_toml():
    config = load_config_from_string('theme = "dark"\n')
    assert config == get_default_config()


def test_load_config_text_json():
    config = load_config_from_string("{}")
    assert config == get_default_config()


def test_load_config_sets_source_file(tmp_path):
    config_file = tmp_path / "custom.toml"

    config = load_config(config_file)

    assert config.source_file == config_file.resolve()
    assert not config.is_from_default_location


def test_load_config_text_has_no_source_file():
    config = load_config_from_string("{}")

    assert config.source_file is None


def test_load_config_text_invalid():
    with pytest.raises(ConfigError, match="Invalid configuration text"):
        load_config_from_string("not valid {")


def test_load_config_invalid_ralph_iterations():
    with pytest.raises(ConfigError, match="max_ralph_iterations"):
        load_config_from_string('{"loop_control": {"max_ralph_iterations": -2}}')


def test_load_config_reserved_context_size():
    config = load_config_from_string('{"loop_control": {"reserved_context_size": 30000}}')
    assert config.loop_control.reserved_context_size == 30000


def test_load_config_max_steps_per_turn():
    config = load_config_from_string("[loop_control]\nmax_steps_per_turn = 42\n")
    assert config.loop_control.max_steps_per_turn == 42


def test_load_config_max_steps_per_run():
    config = load_config_from_string('{"loop_control": {"max_steps_per_run": 7}}')
    assert config.loop_control.max_steps_per_turn == 7


def test_load_config_reserved_context_size_too_low():
    with pytest.raises(ConfigError, match="reserved_context_size"):
        load_config_from_string('{"loop_control": {"reserved_context_size": 500}}')


def test_load_config_compaction_trigger_ratio():
    config = load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 0.8}}')
    assert config.loop_control.compaction_trigger_ratio == 0.8


def test_load_config_compaction_trigger_ratio_default():
    config = load_config_from_string("{}")
    assert config.loop_control.compaction_trigger_ratio == 0.8


def test_load_config_compaction_trigger_ratio_too_low():
    with pytest.raises(ConfigError, match="compaction_trigger_ratio"):
        load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 0.3}}')


def test_load_config_compaction_trigger_ratio_too_high():
    with pytest.raises(ConfigError, match="compaction_trigger_ratio"):
        load_config_from_string('{"loop_control": {"compaction_trigger_ratio": 1.0}}')


def test_load_config_supported_efforts():
    config = load_config_from_string(
        '{"model": {"model": "m", "max_context_size": 1000, "supported_efforts": ["low", "high"]}, "provider": {"type": "anthropic", "base_url": "https://example.com", "api_key": "k"}}'
    )
    assert config.model.supported_efforts == {"low", "high"}


def test_load_config_supported_efforts_defaults_to_full_set():
    config = load_config_from_string(
        '{"model": {"model": "m", "max_context_size": 1000}, "provider": {"type": "anthropic", "base_url": "https://example.com", "api_key": "k"}}'
    )
    assert config.model.supported_efforts == {"low", "medium", "high", "xhigh", "max"}


def test_load_config_invalid_supported_efforts():
    with pytest.raises(ConfigError, match="supported_efforts"):
        load_config_from_string(
            '{"model": {"model": "m", "max_context_size": 1000, "supported_efforts": ["low", "invalid"]}, "provider": {"type": "anthropic", "base_url": "https://example.com", "api_key": "k"}}'
        )


def test_load_config_supported_efforts_rejects_off():
    with pytest.raises(ConfigError, match="supported_efforts|off"):
        load_config_from_string(
            '{"model": {"model": "m", "max_context_size": 1000, "supported_efforts": ["low", "off"]}, "provider": {"type": "anthropic", "base_url": "https://example.com", "api_key": "k"}}'
        )


def test_load_config_prune_ratios_valid():
    """Default prune ratios satisfy: prune_target <= prune_trigger < compaction_trigger."""
    config = load_config_from_string("{}")
    assert config.loop_control.prune_target_ratio <= config.loop_control.prune_trigger_ratio < config.loop_control.compaction_trigger_ratio


def test_load_config_prune_ratios_invalid_target_gte_trigger():
    """Reject prune_target_ratio > prune_trigger_ratio."""
    with pytest.raises(ConfigError, match="Prune ratios must satisfy"):
        load_config_from_string(
            '{"loop_control": {"prune_target_ratio": 0.8, "prune_trigger_ratio": 0.7}}'
        )


def test_load_config_prune_ratios_equal_allowed():
    """Allow prune_target_ratio == prune_trigger_ratio (both zero by default)."""
    config = load_config_from_string(
        '{"loop_control": {"prune_target_ratio": 0.0, "prune_trigger_ratio": 0.0}}'
    )
    assert config.loop_control.prune_target_ratio == 0.0
    assert config.loop_control.prune_trigger_ratio == 0.0


def test_load_config_prune_ratios_invalid_trigger_gte_compaction():
    """Reject prune_trigger_ratio >= compaction_trigger_ratio."""
    with pytest.raises(ConfigError, match="Prune ratios must satisfy"):
        load_config_from_string(
            '{"loop_control": {"prune_trigger_ratio": 0.85, "compaction_trigger_ratio": 0.75}}'
        )
