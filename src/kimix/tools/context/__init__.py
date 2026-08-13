"""Tools for introspecting and compacting conversation context."""
from __future__ import annotations

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.soul import get_current_soul_or_none
from kimi_cli.soul.compaction import CompactMode
from kimi_cli.soul.kimisoul import KimiSoul
from pydantic import BaseModel, Field, model_validator


class ContextUsageParams(BaseModel):
    """No parameters required."""

    pass


class context_usage(CallableTool2):
    name = "context_usage"
    description = (
        "Report the current conversation context usage: percentage, used tokens, "
        "and maximum context size. Use this to decide whether to call compact."
    )
    params = ContextUsageParams

    async def __call__(self, params: ContextUsageParams) -> ToolReturnValue:
        soul = get_current_soul_or_none()
        if soul is None:
            return ToolError(
                message="No active soul/session.",
                output="",
                brief="No active session",
            )
        status = soul.status
        used_pct = round(status.context_usage * 100, 1)
        output = (
            f"Context usage: {status.context_usage:.1%} "
            f"({status.context_tokens:,} / {status.max_context_tokens:,} tokens)"
        )
        result = ToolOk(output=output)
        result.extras = {
            "context_usage_pct": used_pct,
            "used_tokens": status.context_tokens,
            "max_tokens": status.max_context_tokens,
            "free_tokens": status.max_context_tokens - status.context_tokens,
        }
        return result


class CompactParams(BaseModel):
    instruction: str | None = Field(
        default=None,
        description="Optional instruction guiding what to preserve during compaction.",
    )
    mode: str = Field(
        default="auto",
        description=(
            "Compaction mode / style. "
            "'retentive' (default, keep more detail), 'balanced' (structured summary), "
            "'aggressive' (shorter), 'technical' (emphasize code/errors/design). "
            "'auto': automatically select based on current context usage. "
            "Does not affect preserve depth or cascade behavior.\n"
            "Mode selection guide:\n"
            "- retentive: Keeps more detail; use when context is moderately full.\n"
            "- balanced: Structured summary; use when context is very full.\n"
            "- aggressive: Shortest summary; use only when critically low on context.\n"
            "- technical: Emphasizes code, errors, and design decisions.\n"
            "- auto: Automatically picks based on context usage."
        ),
    )

    _COOLDOWN_STEPS = 5

    @model_validator(mode="after")
    def _resolve_mode(self) -> "CompactParams":
        """Resolve 'auto' mode based on current context usage."""
        if self.mode == "auto":
            soul = get_current_soul_or_none()
            if soul is not None:
                usage = soul.status.context_usage
                if usage > 0.9:
                    self.mode = "aggressive"
                elif usage > 0.75:
                    self.mode = "balanced"
                else:
                    self.mode = "retentive"
        return self


class compact(CallableTool2):
    name = "compact"
    description = (
        "Compact / summarize the conversation context to reduce token usage. "
        "Call this when context_usage shows usage is high and you want to free up context. "
        "Optionally pass an instruction and a compaction mode (balanced, aggressive, "
        "retentive, technical, auto) to control the summary style. "
        "[IMPORTANT] Do NOT call compact more than once every 5 steps, "
        "and only call it when context_usage exceeds 70%."
    )
    params = CompactParams

    async def __call__(self, params: CompactParams) -> ToolReturnValue:
        soul = get_current_soul_or_none()
        if not isinstance(soul, KimiSoul):
            return ToolError(
                message="No active KimiSoul to compact.",
                output="",
                brief="No active soul",
            )

        # Cooldown check
        current_step = soul.status.step_count if hasattr(soul.status, "step_count") else 0
        last_compact = soul.custom_data.get("last_compact_step", 0) if hasattr(soul, "custom_data") else 0
        if hasattr(soul, "custom_data") and current_step - last_compact < 5:
            return ToolError(
                message=(
                    f"Compact called too soon ({current_step - last_compact} steps ago). "
                    "Wait at least 5 steps between compactions."
                ),
                brief="Compaction too frequent",
            )

        # Resolve mode string to CompactMode enum
        mode_map = {
            "retentive": CompactMode.RETENTIVE,
            "balanced": CompactMode.BALANCED,
            "aggressive": CompactMode.AGGRESSIVE,
            "technical": CompactMode.TECHNICAL,
        }
        resolved_mode = mode_map.get(params.mode, CompactMode.RETENTIVE)

        try:
            await soul.compact_context(
                manual=True,
                custom_instruction=params.instruction or "",
                avoid_cascade=True,
                mode=resolved_mode,
            )
            # Update cooldown
            if hasattr(soul, "custom_data"):
                soul.custom_data["last_compact_step"] = current_step
        except Exception as exc:
            return ToolError(
                message=str(exc),
                output="",
                brief="Compaction failed",
            )

        return ToolOk(
            output="Compaction completed.",
            message=(
                "Compaction completed. "
                "[IMPORTANT] Do not call compact again until context usage exceeds 70%."
            ),
        )
