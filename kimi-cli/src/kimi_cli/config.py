from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import orjson
import tomlkit
from kosong.chat_provider import ThinkingEffort
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_serializer,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from kimi_cli.exception import ConfigError
from kimi_cli.hooks.config import HookDef
from kimi_cli.llm import ModelCapability, ProviderType
from kimi_cli.share import get_share_dir
from kimi_cli.utils.logging import logger


class OAuthRef(BaseModel):
    """Reference to OAuth credentials stored outside the config file."""

    storage: Literal["keyring", "file"] = "file"
    """Credential storage backend."""
    key: str
    """Storage key to locate OAuth credentials."""


class OpenAISettings(BaseModel):
    """OpenAI Legacy provider-specific ``extra_body`` options."""

    thinking: bool = Field(
        default=True,
        description=(
            "If true, include the ``thinking`` key in the auto-generated "
            "``extra_body`` for ``openai_legacy`` providers."
        ),
    )
    reasoning: bool = Field(
        default=True,
        description=(
            "If true, include the ``reasoning`` key in the auto-generated "
            "``extra_body`` for ``openai_legacy`` providers."
        ),
    )
    chat_template_kwargs: bool = Field(
        default=True,
        description=(
            "If true, include the ``chat_template_kwargs`` key in the auto-generated "
            "``extra_body`` for ``openai_legacy`` providers."
        ),
    )


class LLMProvider(BaseModel):
    """LLM provider configuration."""

    type: ProviderType
    """Provider type"""
    base_url: str
    """API base URL"""
    api_key: SecretStr
    """API key"""
    env: dict[str, str] | None = None
    """Environment variables to set before creating the provider instance"""
    custom_headers: dict[str, str] | None = None
    """Custom headers to include in API requests"""
    reasoning_key: str | None = None
    """Message field name carrying reasoning content for OpenAI-compatible APIs.
    Applies to provider type ``openai_legacy``. Defaults to ``reasoning_content``
    when unset. Use an empty string to disable reasoning round-tripping."""
    openai_settings: OpenAISettings | None = None
    """OpenAI Legacy-specific ``extra_body`` options. Only used when the provider
    type is ``openai_legacy``."""
    oauth: OAuthRef | None = None
    """OAuth credential reference (do not store tokens here)."""

    @field_serializer("api_key", when_used="json")
    def dump_secret(self, v: SecretStr):
        return v.get_secret_value()


def _validate_supported_efforts(v: set[ThinkingEffort]) -> set[ThinkingEffort]:
    """Reject the special ``off`` value; it disables thinking rather than selecting an effort rank."""
    if "off" in v:
        raise ValueError("'off' is not a valid supported_efforts value")
    return v


class LLMModel(BaseModel):
    """LLM model configuration."""

    model: str
    """Model name"""
    max_context_size: int
    """Maximum context size (unit: tokens)"""
    capabilities: set[ModelCapability] | None = None
    """Model capabilities"""
    display_name: str | None = None
    """Human-readable model name (sourced from the provider's models API when available)"""
    supported_efforts: Annotated[
        set[ThinkingEffort],
        AfterValidator(_validate_supported_efforts),
    ] = Field(
        default_factory=lambda: {"low", "medium", "high", "xhigh", "max"},
        description=(
            "Thinking effort levels this model accepts. "
            "Defaults to the full set. "
            "The special value ``\"off\"`` is not an effort rank and must not be included."
        ),
    )

class LoopControl(BaseModel):
    """Agent loop control configuration."""

    max_steps_per_turn: int = Field(
        default=15000,
        ge=1,
        validation_alias=AliasChoices("max_steps_per_turn", "max_steps_per_run"),
    )
    """Maximum number of steps in one turn"""
    max_retries_per_step: int = Field(default=5, ge=1)
    """Maximum number of retries in one step"""
    max_session_restarts: int = Field(default=3, ge=0, le=10)
    """Maximum number of automatic session restarts when step retries are
expected. Set to 0 to disable auto-restart entirely.
Default is 3."""
    max_ralph_iterations: int = Field(default=0, ge=-1)
    """Extra iterations after the first turn in Ralph mode. Use -1 for unlimited."""
    reserved_context_size: int = Field(default=75_000, ge=1000)
    """Reserved token count for LLM response generation. Auto-compaction triggers when
    either context_tokens + reserved_context_size >= max_context_size or
    context_tokens >= max_context_size * compaction_trigger_ratio. Default is 50000."""
    compaction_trigger_ratio: float = Field(default=0.8, ge=0.5, le=0.99)
    """Context usage ratio threshold for auto-compaction. Default is 0.85 (85%).
    Auto-compaction triggers when context_tokens >= max_context_size * compaction_trigger_ratio
    or when context_tokens + reserved_context_size >= max_context_size."""
    max_system_prompt_tokens: int = Field(default=4_000, ge=1_000)
    """Maximum token count for the system prompt. If the constructed prompt exceeds
    this budget, step memory and changed-files lists are truncated progressively.
    Default is 4_000."""
    max_preserved_messages: int = Field(default=2, ge=1, le=10)
    """Maximum number of recent user/assistant message pairs to preserve verbatim
    during context compaction. Default is 2."""
    min_preserved_messages: int = Field(default=1, ge=1, le=10)
    """Minimum number of recent user/assistant message pairs to preserve verbatim
    during context compaction. Default is 1."""
    adaptive_preserve_enabled: bool = Field(default=True)
    """When true, dynamically adjust preserve depth based on session signals
    (errors, tool calls, reasoning). Default is true."""
    compact_reminder_enabled: bool = Field(default=True)
    """When true, inject a system-reminder to suggest compaction when context usage
    exceeds compact_reminder_threshold. Default is true."""
    compact_reminder_threshold: float = Field(default=0.70, ge=0.5, le=0.95)
    """Context usage ratio at which the compact reminder is injected.
    Should be lower than compaction_trigger_ratio to give the agent a heads-up.
    Default is 0.70 (70%)."""

    # ── Recency-edge re-injection & memory durability ────────────────────────

    todo_reminder_enabled: bool = Field(default=True)
    """When true, periodically re-inject unfinished TodoList items at the end
    of the context window, where model attention is strongest (recency edge).
    Default is true."""
    todo_reminder_interval_steps: int = Field(default=20, ge=1)
    """Minimum number of steps between repeated todo reminder injections when
    the todo list has not changed. Default is 20."""

    target_churn_enabled: bool = Field(default=True)
    """When true, inject a reminder when the agent repeatedly modifies the same
    file target (across different tools) or hits the same normalized error
    repeatedly — loop shapes that per-call streak detection cannot catch.
    Default is true."""
    target_churn_file_warn: int = Field(default=8, ge=2)
    """Number of edits to the same file target that triggers a churn reminder.
    Default is 8."""
    target_churn_file_strong: int = Field(default=15, ge=3)
    """Number of edits to the same file target that triggers a strong
    stop-patching reminder (once per file per turn). Default is 15."""
    target_churn_error_warn: int = Field(default=5, ge=2)
    """Consecutive identical (normalized) tool errors that trigger an
    error-signature reminder. Default is 5."""
    target_churn_cooldown_steps: int = Field(default=10, ge=0)
    """Minimum number of steps to stay silent after any target-churn
    injection. Default is 6."""

    verification_gate_enabled: bool = Field(default=True)
    """When true, a turn that ends with unfinished todos, failing todo
    verification, or unverified code edits is nudged to continue instead of
    finishing. Applies at the soul layer, so it covers CLI, server,
    subagent, and DAG/flow sessions. Default is true."""
    verification_gate_max_nudges: int = Field(default=2, ge=0, le=10)
    """Maximum number of verification-gate nudges per turn before the gate
    lets the turn finish anyway (deadlock prevention). Default is 2."""
    cli_closing_reminder_rounds: int = Field(default=1, ge=0, le=5)
    """Number of CLI-layer closing reminder rounds (todo + code-todo checks)
    after a successful prompt. The soul-layer verification gate is the
    primary mechanism; this is the CLI fallback. 0 disables the CLI
    fallback. Default is 1."""

    budget_reminder_enabled: bool = Field(default=False)
    """When true, inject budget-awareness reminders as step (or wall-clock)
    usage crosses ``budget_warn_ratios`` of the per-turn budget, so the
    agent can plan its wrap-up. Default is true."""
    budget_warn_ratios: tuple[float, ...] = Field(default=(0.7, 0.9))
    """Ascending usage ratios of the per-turn budget that each trigger one
    reminder per turn. The final ratio issues an urgent wrap-up reminder.
    Default is (0.7, 0.9)."""
    budget_wall_clock_seconds: int = Field(default=0, ge=0)
    """Optional per-turn wall-clock budget in seconds. 0 disables the
    wall-clock dimension (step budget only). Default is 0."""

    compaction_decision_section_enabled: bool = Field(default=True)
    """When true, compaction summaries must include `## Decisions &
    Conclusions` and `## Verification Status` sections, so early-task
    decisions and verification state survive compaction. Default is true."""

    best_of_n_enabled: bool = Field(default=False)
    """When true, enable best-of-N sampling (same task sampled N times in
    isolated workspaces, then selected and applied). Default is false —
    enable after single-sample A/B results converge (P5)."""
    best_of_n: int = Field(default=4, ge=1, le=16)
    """Default number of parallel samples for best-of-N mode. Default is 4."""
    best_of_n_selector: str = Field(default="self_eval", pattern="^(self_eval|majority)$")
    """Default candidate selection strategy for best-of-N mode:
    'self_eval' (one model review call) or 'majority' (pairwise votes).
    Default is 'self_eval'."""

    context_meter_enabled: bool = Field(default=True)
    """When true, inject a reminder to persist important facts with the `Memory`
    tool when usage materially changes, so the agent can self-regulate
    (checkpoint, flush memory) before the harness compacts. Default is true."""
    context_meter_min_delta: float = Field(default=0.05, ge=0.0, le=0.5)
    """Minimum usage-ratio change since the last context-meter injection
    required to inject again. Default is 0.05 (5%)."""
    context_meter_cooldown_steps: int = Field(default=10, ge=0)
    """Minimum number of steps between context-meter injections.
    Default is 10."""

    pre_compact_flush_enabled: bool = Field(default=True)
    """When true, flush important agent state (unfinished todos, notes) to the
    durable session memory directory before any compaction fires, so details
    destroyed by summarization are recoverable from disk. Default is true."""
    memory_restore_enabled: bool = Field(default=True)
    """When true, append a memory-pointer message after compaction that
    re-surfaces the durable memory directory (and the latest pre-compaction
    flush excerpt) at the end of the rebuilt context. Default is true."""

    auto_retrieve_history: bool = Field(default=True)
    """When true, automatically search archived conversation history before each
    turn and inject the most relevant past turn if it exceeds the similarity
    threshold. Default is true."""
    auto_retrieve_history_threshold: float = Field(default=5.0, ge=0.0)
    """Minimum BM25 relevance score for auto-injecting a matching archived turn.
    Higher values require stronger matches. Default is 5.0."""
    auto_retrieve_working_memory: bool = Field(default=True)
    """When true, search the current (non-compacted) conversation for relevant
    turns that may be buried deep in the context window. Default is true."""
    auto_retrieve_working_memory_threshold: float = Field(default=5.0, ge=0.0)
    """Minimum BM25 relevance score for auto-injecting a working-memory turn.
    Default is 5.0."""
    auto_retrieve_recency_memory: bool = Field(default=True)
    """When true, boost recent turns with a time-decay factor and inject the
    best boosted match if it exceeds the threshold. Default is true."""
    auto_retrieve_recency_memory_threshold: float = Field(default=4.0, ge=0.0)
    """Minimum boosted score for auto-injecting a recency-memory turn.
    Default is 4.0."""
    auto_retrieve_recency_weight: float = Field(default=1.0, ge=0.0)
    """Weight applied to the recency boost multiplier.
    Default is 1.0."""
    auto_retrieve_max_injections_per_turn: int = Field(default=3, ge=1, le=5)
    """Maximum number of auto-retrieved injections to inject per turn.
    Default is 3."""
    auto_retrieve_max_tokens_per_turn: int = Field(default=20_000, ge=500, le=100_000)
    """Maximum total token budget for all auto-retrieved history injections in one turn.
    If the cumulative token count of selected injections exceeds this budget,
    additional injections are skipped. Default is 2,000."""

    # ── Context pruning (smart history removal) ──────────────────────────────

    context_pruning_enabled: bool = Field(default=True)
    """When true, enable the context pruner to dynamically reclaim context
    space by removing historical information the LLM no longer needs,
    without harshly breaking the KV cache. Default is true."""

    prune_trigger_ratio: float = Field(default=0.0, ge=0.0, le=0.95)
    """Context usage ratio threshold for triggering a prune pass.
    Default is 0.0 — always prune regardless of context usage, so
    ephemeral content is cleaned up eagerly from the very first step.
    Must be lower than ``compaction_trigger_ratio``."""

    prune_target_ratio: float = Field(default=0.0, ge=0.0, le=0.9)
    """Target context usage ratio after a prune pass.
    Default is 0.0 — prune as aggressively as allowed by other limits
    (``prune_max_fraction_per_pass``, ``prune_min_free_tokens``, etc.).
    Must not exceed ``prune_trigger_ratio``. Default is 0.0 (0%)."""

    prune_stable_prefix_messages: int = Field(default=4, ge=1)
    """Number of initial messages to always keep as a stable cached prefix.
    Default is 4."""

    prune_recent_messages_protected: int = Field(default=6, ge=1)
    """Number of recent user/assistant turns (plus their tool messages)
    to protect from pruning. Default is 6."""

    prune_min_free_tokens: int = Field(default=2_000, ge=0)
    """Minimum token savings required to justify a prune pass.
    If the pass would free fewer tokens, it is skipped.
    Default is 2,000."""

    prune_cooldown_steps: int = Field(default=4, ge=1)
    """Minimum number of steps between consecutive prune passes.
    Default is 4."""

    prune_min_usage_growth: float = Field(default=0.05)
    """Minimum ratio of usage growth since the last prune to allow
    re-pruning. Default is 0.05 (5%)."""

    prune_max_fraction_per_pass: float = Field(default=0.5, ge=0.1, le=0.9)
    """Maximum fraction of effective tokens to prune in a single pass.
    Default is 0.5 (50%)."""

    # Tier A — ephemeral injected messages (primary, safest)
    prune_ephemeral_enabled: bool = Field(default=True)
    """When true, enable Tier A removal of consumed ephemeral messages.
    Default is true."""
    prune_ephemeral_notifications: bool = Field(default=True)
    """When true, drop consumed notification messages older than the
    recency window. Default is true."""
    prune_ephemeral_task_snapshots: bool = Field(default=True)
    """When true, keep only the most recent active-task snapshot and
    drop older ones. Default is true."""
    prune_ephemeral_dmail_notices: bool = Field(default=True)
    """When true, drop spent D-Mail notices once they are older than
    the turn they applied to. Default is true."""
    prune_ephemeral_checkpoint_markers: bool = Field(default=False)
    """When true, optionally drop CHECKPOINT markers. Default is false
    (keep them, since some flows correlate D-Mail by these)."""

    # Tier B — substantive content elision (escalation only)
    prune_substantive_enabled: bool = Field(default=True)
    """When true, enable Tier B elision of stale/oversized substantive
    content when Tier A is insufficient. Default is true."""
    prune_tool_output_min_tokens: int = Field(default=512, ge=64)
    """Minimum token count for a tool output to be considered oversized
    and eligible for elision. Default is 512."""
    prune_elide_thinking: bool = Field(default=True)
    """When true, elide old reasoning (ThinkPart) content outside the
    recency window. Default is true."""
    prune_dedupe_near_duplicates: bool = Field(default=True)
    """When true, detect and elide near-duplicate large blobs.
    Default is true."""

    prune_persist: bool = Field(default=False)
    """When true, persist prune operations to storage (Layer 2).
    Default is false — Layer 1 only (request-time pruning, history intact)."""
    prune_subagents: bool = Field(default=True)
    """When true, apply pruning to subagent sessions as well.
    Default is true."""

    @model_validator(mode="after")
    def validate_prune_ratios(self) -> Self:
        """Enforce: prune_target_ratio <= prune_trigger_ratio < compaction_trigger_ratio."""
        if not (self.prune_target_ratio <= self.prune_trigger_ratio < self.compaction_trigger_ratio):
            raise ValueError(
                f"Prune ratios must satisfy: prune_target_ratio ({self.prune_target_ratio}) <= "
                f"prune_trigger_ratio ({self.prune_trigger_ratio}) < "
                f"compaction_trigger_ratio ({self.compaction_trigger_ratio})"
            )
        return self


class BackgroundConfig(BaseModel):
    """Background task runtime configuration."""

    max_running_tasks: int = Field(default=4, ge=1)
    read_max_bytes: int = Field(default=30_000, ge=1024)
    notification_tail_lines: int = Field(default=20, ge=1)
    notification_tail_chars: int = Field(default=3_000, ge=256)
    wait_poll_interval_ms: int = Field(default=500, ge=50)
    worker_heartbeat_interval_ms: int = Field(default=5_000, ge=100)
    worker_stale_after_ms: int = Field(default=15_000, ge=1000)
    kill_grace_period_ms: int = Field(default=2_000, ge=100)
    keep_alive_on_exit: bool = Field(
        default=False,
        description="Keep background tasks alive when CLI exits. Default: kill on exit.",
    )
    agent_task_timeout_s: int = Field(default=28800, ge=60)
    """Maximum runtime in seconds for a background agent task. Default: 28800 (8 hours)."""
    print_wait_ceiling_s: int = Field(default=3600, ge=1)
    """Hard ceiling for how long ``--print`` mode waits for background tasks before
    killing them and exiting. The effective wait is
    ``min(max(active_task.timeout_s or agent_task_timeout_s), print_wait_ceiling_s)``.
    Default: 3600 (1 hour)."""


class NotificationConfig(BaseModel):
    """Notification runtime configuration."""

    claim_stale_after_ms: int = Field(default=15_000, ge=1000)


class MoonshotSearchConfig(BaseModel):
    """Moonshot Search configuration."""

    base_url: str
    """Base URL for Moonshot Search service."""
    api_key: SecretStr
    """API key for Moonshot Search service."""
    custom_headers: dict[str, str] | None = None
    """Custom headers to include in API requests."""
    oauth: OAuthRef | None = None
    """OAuth credential reference (do not store tokens here)."""

    @field_serializer("api_key", when_used="json")
    def dump_secret(self, v: SecretStr):
        return v.get_secret_value()


class MoonshotFetchConfig(BaseModel):
    """Moonshot Fetch configuration."""

    base_url: str
    """Base URL for Moonshot Fetch service."""
    api_key: SecretStr
    """API key for Moonshot Fetch service."""
    custom_headers: dict[str, str] | None = None
    """Custom headers to include in API requests."""
    oauth: OAuthRef | None = None
    """OAuth credential reference (do not store tokens here)."""

    @field_serializer("api_key", when_used="json")
    def dump_secret(self, v: SecretStr):
        return v.get_secret_value()


class Services(BaseModel):
    """Services configuration."""

    moonshot_search: MoonshotSearchConfig | None = None
    """Moonshot Search configuration."""
    moonshot_fetch: MoonshotFetchConfig | None = None
    """Moonshot Fetch configuration."""


class MCPClientConfig(BaseModel):
    """MCP client configuration."""

    tool_call_timeout_ms: int = 60000
    """Timeout for tool calls in milliseconds."""


class MCPConfig(BaseModel):
    """MCP configuration."""

    client: MCPClientConfig = Field(
        default_factory=MCPClientConfig, description="MCP client configuration"
    )


class Config(BaseModel):
    """Main configuration structure."""

    is_from_default_location: bool = Field(
        default=False,
        description="Whether the config was loaded from the default location",
        exclude=True,
    )
    source_file: Path | None = Field(
        default=None,
        description="Path to the loaded config file. None when loaded from --config text.",
        exclude=True,
    )
    default_thinking: bool = Field(default=False, description="Default thinking mode")
    default_yolo: bool = Field(default=False, description="Default yolo (auto-approve) mode")
    default_editor: str = Field(
        default="",
        description="Default external editor command (e.g. 'vim', 'code --wait')",
    )
    theme: Literal["dark", "light"] = Field(
        default="dark",
        description="Terminal color theme. Use 'light' for light terminal backgrounds.",
    )
    show_thinking_stream: bool = Field(
        default=True,
        description=(
            "If true, stream the raw reasoning text in the live area as a "
            "6-line scrolling preview and commit the full reasoning markdown "
            "to history when the block ends. Default true. Set to false to "
            "show only the compact 'Thinking ...' indicator and a one-line "
            "trace summary."
        ),
    )
    model: LLMModel | None = Field(
        default=None, description="Active LLM model configuration"
    )
    provider: LLMProvider | None = Field(
        default=None, description="Active LLM provider configuration"
    )
    loop_control: LoopControl = Field(default_factory=LoopControl, description="Agent loop control")
    background: BackgroundConfig = Field(
        default_factory=BackgroundConfig, description="Background task configuration"
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig, description="Notification configuration"
    )
    services: Services = Field(default_factory=Services, description="Services configuration")
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP configuration")
    hooks: list[HookDef] = Field(default_factory=list, description="Hook definitions")  # pyright: ignore[reportUnknownVariableType]
    merge_all_available_skills: bool = Field(
        default=True,
        description=(
            "Merge skills from all existing brand directories (kimi/claude/codex) "
            "instead of using only the first one found. Defaults to true so users "
            "who keep skills in multiple brand directories see everything out of "
            "the box; set to false to restore the first-match-only behaviour."
        ),
    )
    extra_skill_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "Extra directories to discover skills from, added on top of the "
            "built-in / user / project locations. Each entry may be an absolute "
            "path, ``~``-prefixed (expanded against $HOME), or relative to the "
            "project root (the nearest ``.git`` directory above the work dir). "
            "Missing paths are silently skipped."
        ),
    )
    # LLM override settings
    max_tokens: int | None = Field(default=384_000, description='LLM max output tokens')
    temperature: float | None = Field(default=None, description='LLM Temperature')
    top_p: float | None = Field(default=None, description='LLM top_p')
    top_k: int | None = Field(default=None, description='LLM top_k')
    thinking_effort: Literal["off", "low", "medium", "high", "xhigh", "max"] | None = Field(default=None, description='LLM thinking effort')

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.model is not None and self.provider is None:
            raise ValueError("Active model configured without a provider")
        return self


def get_config_file() -> Path:
    """Get the configuration file path."""
    return get_share_dir() / "config.toml"


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def load_config(config_file: Path | None = None) -> Config:
    """
    Load configuration from config file.
    If the config file does not exist, create it with default configuration.

    Args:
        config_file (Path | None): Path to the configuration file. If None, use default path.

    Returns:
        Validated Config object.

    Raises:
        ConfigError: If the configuration file is invalid.
    """
    default_config_file = get_config_file().expanduser().resolve(strict=False)
    if config_file is None:
        config_file = default_config_file
    config_file = config_file.expanduser().resolve(strict=False)
    is_default_config_file = config_file == default_config_file
    logger.debug("Loading config from file: {file}", file=config_file)

    # If the user hasn't provided an explicit config path, migrate legacy JSON config once.
    if is_default_config_file and not config_file.exists():
        _migrate_json_config_to_toml()

    if not config_file.exists():
        config = get_default_config()
        logger.debug("No config file found, creating default config: {config}", config=config)
        save_config(config, config_file)
        config.is_from_default_location = is_default_config_file
        config.source_file = config_file
        return config

    try:
        config_text = config_file.read_text(encoding="utf-8")
        if config_file.suffix.lower() == ".json":
            data = orjson.loads(config_text)
        else:
            data = tomlkit.loads(config_text)
        config = Config.model_validate(data)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in configuration file {config_file}: {e}") from e
    except TOMLKitError as e:
        raise ConfigError(f"Invalid TOML in configuration file {config_file}: {e}") from e
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration file {config_file}: {e}") from e
    config.is_from_default_location = is_default_config_file
    config.source_file = config_file
    return config


def load_config_from_string(config_string: str) -> Config:
    """
    Load configuration from a TOML or JSON string.

    Args:
        config_string (str): TOML or JSON configuration text.

    Returns:
        Validated Config object.

    Raises:
        ConfigError: If the configuration text is invalid.
    """
    if not config_string.strip():
        raise ConfigError("Configuration text cannot be empty")

    json_error: orjson.JSONDecodeError | None = None
    try:
        data = orjson.loads(config_string)
    except orjson.JSONDecodeError as exc:
        json_error = exc
        data = None

    if data is None:
        try:
            data = tomlkit.loads(config_string)
        except TOMLKitError as toml_error:
            raise ConfigError(
                f"Invalid configuration text: {json_error}; {toml_error}"
            ) from toml_error

    try:
        config = Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration text: {e}") from e
    config.is_from_default_location = False
    config.source_file = None
    return config


def save_config(config: Config, config_file: Path | None = None):
    """
    Save configuration to config file.

    Args:
        config (Config): Config object to save.
        config_file (Path | None): Path to the configuration file. If None, use default path.
    """
    config_file = config_file or get_config_file()
    logger.debug("Saving config to file: {file}", file=config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_data = config.model_dump(mode="json", exclude_none=True)
    with open(config_file, "w", encoding="utf-8") as f:
        if config_file.suffix.lower() == ".json":
            f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2).decode('utf-8'))
        else:
            f.write(tomlkit.dumps(config_data))  # type: ignore[reportUnknownMemberType]


def _migrate_json_config_to_toml() -> None:
    old_json_config_file = get_share_dir() / "config.json"
    new_toml_config_file = get_share_dir() / "config.toml"

    if not old_json_config_file.exists():
        return
    if new_toml_config_file.exists():
        return

    logger.info(
        "Migrating legacy config file from {old} to {new}",
        old=old_json_config_file,
        new=new_toml_config_file,
    )

    try:
        with open(old_json_config_file, encoding="utf-8") as f:
            data = orjson.loads(f.read())
        config = Config.model_validate(data)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in legacy configuration file: {e}") from e
    except ValidationError as e:
        raise ConfigError(f"Invalid legacy configuration file: {e}") from e

    # Write new TOML config, then keep a backup of the original JSON file.
    save_config(config, new_toml_config_file)
    backup_path = old_json_config_file.with_name("config.json.bak")
    old_json_config_file.replace(backup_path)
    logger.info("Legacy config backed up to {file}", file=backup_path)
