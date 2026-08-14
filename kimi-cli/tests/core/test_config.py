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


def _model_config(model: str, **model_kwargs) -> str:
    """Build a minimal JSON config string with the given model settings."""
    import json

    model_section = {"model": model}
    model_section.update(model_kwargs)
    config = {
        "model": model_section,
        "provider": {"type": "openai_legacy", "base_url": "https://example.com", "api_key": "k"},
    }
    return json.dumps(config)


def test_load_config_model_defaults_derived_from_name():
    """Both max_context_size and max_tokens are derived from the model name."""
    config = load_config_from_string(_model_config("openai/gpt-5.4"))
    assert config.model.max_context_size == 1_000_000
    assert config.model.max_tokens == 128_000


def test_load_config_model_defaults_deepseek():
    """DeepSeek model defaults are resolved correctly."""
    config = load_config_from_string(_model_config("deepseek-v4-pro"))
    assert config.model.max_context_size == 1_000_000
    assert config.model.max_tokens == 384_000


def test_load_config_model_max_tokens_quarter_when_context_set():
    """If max_context_size is set but max_tokens is not, max_tokens = context // 4."""
    config = load_config_from_string(_model_config("unknown-model", max_context_size=200_000))
    assert config.model.max_context_size == 200_000
    assert config.model.max_tokens == 50_000


def test_load_config_model_explicit_max_tokens_preserved():
    """Explicit max_tokens is preserved while max_context_size is derived from name."""
    config = load_config_from_string(_model_config("claude-sonnet-5", max_tokens=10_000))
    assert config.model.max_context_size == 1_000_000
    assert config.model.max_tokens == 10_000


def test_load_config_model_unknown_without_context_exits():
    """An unknown model without max_context_size prints an error and exits."""
    with pytest.raises(SystemExit) as exc_info:
        load_config_from_string(_model_config("totally-unknown-model"))
    assert exc_info.value.code == 1


def test_load_config_model_grok_output_none():
    """Grok has no default max_output, so max_tokens stays None when both are unset."""
    config = load_config_from_string(_model_config("xai/grok"))
    assert config.model.max_context_size == 2_000_000
    assert config.model.max_tokens is None
