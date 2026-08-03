from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from kimi_cli.config import (
    Config,
    get_default_config,
    load_config,
    load_config_from_string,
)
from kimi_cli.exception import ConfigError


def test_default_config():
    config = get_default_config()
    assert config == snapshot(Config())


def test_default_config_dump():
    config = get_default_config()
    assert config.model_dump() == snapshot(
        {
            "default_thinking": False,
            "default_yolo": False,
            "default_editor": "",
            "theme": "dark",
            "show_thinking_stream": True,
            "model": None,
            "provider": None,
            "loop_control": {
                "max_steps_per_turn": 15000,
                "max_retries_per_step": 5,
                "max_session_restarts": 3,
                "max_ralph_iterations": 0,
                "reserved_context_size": 75000,
                "compaction_trigger_ratio": 0.8,
                "max_system_prompt_tokens": 4000,
                "max_preserved_messages": 2,
                "min_preserved_messages": 1,
                "adaptive_preserve_enabled": True,
                "compact_reminder_enabled": True,
                "compact_reminder_threshold": 0.7, 'todo_reminder_enabled': True, 'todo_reminder_interval_steps': 20, 'target_churn_enabled': False, 'target_churn_file_warn': 8, 'target_churn_file_strong': 15, 'target_churn_error_warn': 5, 'target_churn_cooldown_steps': 10, 'verification_gate_enabled': True, 'verification_gate_max_nudges': 2, 'cli_closing_reminder_rounds': 1, 'budget_reminder_enabled': False, 'budget_warn_ratios': (0.7, 0.9), 'budget_wall_clock_seconds': 0, 'compaction_decision_section_enabled': True, 'best_of_n_enabled': False, 'best_of_n': 4, 'best_of_n_selector': 'self_eval', 'context_meter_enabled': True, 'context_meter_min_delta': 0.15, 'context_meter_cooldown_steps': 30, 'pre_compact_flush_enabled': True, 'memory_restore_enabled': True, "auto_retrieve_history": True,
                "auto_retrieve_history_threshold": 5.0,
                "auto_retrieve_working_memory": True,
                "auto_retrieve_working_memory_threshold": 5.0,
                "auto_retrieve_recency_memory": True,
                "auto_retrieve_recency_memory_threshold": 4.0,
                "auto_retrieve_recency_weight": 1.0,
                "auto_retrieve_max_injections_per_turn": 3,
                "auto_retrieve_max_tokens_per_turn": 20000,
                "context_pruning_enabled": True,
                "prune_trigger_ratio": 0.0,
                "prune_target_ratio": 0.0,
                "prune_stable_prefix_messages": 4,
                "prune_recent_messages_protected": 6,
                "prune_min_free_tokens": 2000,
                "prune_cooldown_steps": 4,
                "prune_min_usage_growth": 0.05,
                "prune_max_fraction_per_pass": 0.5,
                "prune_ephemeral_enabled": True,
                "prune_ephemeral_notifications": True,
                "prune_ephemeral_task_snapshots": True,
                "prune_ephemeral_dmail_notices": True,
                "prune_ephemeral_checkpoint_markers": False,
                "prune_substantive_enabled": True,
                "prune_tool_output_min_tokens": 512,
                "prune_elide_thinking": True,
                "prune_dedupe_near_duplicates": True,
                "prune_persist": False,
                "prune_subagents": True,
            },
            "background": {
                "max_running_tasks": 4,
                "read_max_bytes": 30000,
                "notification_tail_lines": 20,
                "notification_tail_chars": 3000,
                "wait_poll_interval_ms": 500,
                "worker_heartbeat_interval_ms": 5000,
                "worker_stale_after_ms": 15000,
                "kill_grace_period_ms": 2000,
                "keep_alive_on_exit": False,
                "agent_task_timeout_s": 28800,
                "print_wait_ceiling_s": 3600,
            },
            "notifications": {
                "claim_stale_after_ms": 15000,
            },
            "services": {"search": None, "fetch": None},
            "mcp": {"client": {"tool_call_timeout_ms": 60000}},
            "hooks": [],
            "merge_all_available_skills": True,
            "extra_skill_dirs": [],
            "max_tokens": 384000,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "thinking_effort": None,
        }
    )


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


def test_loop_control_new_feature_defaults():
    """Default values for the P1-P7 feature keys (plan v2 §14)."""
    from kimi_cli.config import LoopControl

    lc = LoopControl()

    # P1: target churn
    assert lc.target_churn_enabled is False
    assert lc.target_churn_file_warn == 8
    assert lc.target_churn_file_strong == 15
    assert lc.target_churn_error_warn == 5
    assert lc.target_churn_cooldown_steps == 10

    # P2: verification gate + CLI closing loop
    assert lc.verification_gate_enabled is True
    assert lc.verification_gate_max_nudges == 2
    assert lc.cli_closing_reminder_rounds == 1

    # P3: budget reminder
    assert lc.budget_reminder_enabled is False
    assert lc.budget_warn_ratios == (0.7, 0.9)
    assert lc.budget_wall_clock_seconds == 0

    # P4: compaction decision sections
    assert lc.compaction_decision_section_enabled is True

    # P7: best-of-N (default off until A/B conclusions)
    assert lc.best_of_n_enabled is False
    assert lc.best_of_n == 4
    assert lc.best_of_n_selector == "self_eval"


def test_loop_control_new_feature_validation():
    """Field constraints on the new keys are enforced."""
    from pydantic import ValidationError

    from kimi_cli.config import LoopControl

    with pytest.raises(ValidationError):
        LoopControl(target_churn_file_warn=1)  # ge=2
    with pytest.raises(ValidationError):
        LoopControl(verification_gate_max_nudges=11)  # le=10
    with pytest.raises(ValidationError):
        LoopControl(best_of_n=0)  # ge=1
    with pytest.raises(ValidationError):
        LoopControl(best_of_n_selector="external_verifier")  # pattern
