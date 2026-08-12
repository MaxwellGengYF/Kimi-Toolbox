"""AgentSwarm tool for dispatching parallel sub-agent tasks."""
from __future__ import annotations

import asyncio
import html
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kimi_cli.session import Session
from kimi_cli.tools import SkipThisTool
from pydantic import BaseModel, Field, model_validator

import kimix.base as base
import kimix.utils as utils
from kimix.ui.printing import MessageType
from kimix.tools.agent import _AgentConversationCollector
from kimix.utils.system_prompt import SystemPromptType

_MAX_SUB_AGENTS = 128
_DEFAULT_BURST = 5
_DEFAULT_INTERVAL_SECONDS = 0.7
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 1.0

_SUBAGENT_TYPE_MAP: dict[str, SystemPromptType] = {
    "coder": SystemPromptType.Worker,
    "explore": SystemPromptType.Reader,
    "plan": SystemPromptType.TodoMaker,
}


@dataclass
class SwarmTask:
    """A single sub-agent task."""

    prompt: str
    agent_id: str | None
    index: int


class SwarmSubagentResult:
    """Result of a single sub-agent task."""

    def __init__(
        self,
        index: int,
        agent_id: str,
        output: str,
        success: bool,
        error: str | None = None,
        elapsed: float | None = None,
    ):
        self.index = index
        self.agent_id = agent_id
        self.output = output
        self.success = success
        self.error = error
        self.elapsed = elapsed


class AgentSwarmParams(BaseModel):
    """Parameters for the AgentSwarm tool."""

    description: str = Field(description="Short description of the whole swarm.")
    mode: Literal["fanout", "parallel_sample"] = Field(
        default="fanout",
        description="'fanout': decompose into independent items (default). "
        "'parallel_sample': run the SAME task N times in isolated workspaces, "
        "then select and apply the best result (best-of-N).",
    )
    sample_n: int | None = Field(
        default=None,
        description="Number of parallel samples for mode='parallel_sample' (default 4).",
    )
    selector: Literal["self_eval", "majority"] | None = Field(
        default=None,
        description="Selection strategy for mode='parallel_sample' (default 'self_eval').",
    )
    subagent_type: str = Field(
        default="coder",
        description="Type of sub-agent. Built-in: 'coder', 'explore', 'plan'. "
        "Custom types can be registered in agent configuration.",
    )
    prompt_template: str | None = Field(
        default=None,
        description="Prompt template that contains the placeholder {{item}}. "
        "Mutually exclusive with prompt_prefix.",
    )
    prompt_prefix: str | None = Field(
        default=None,
        description="Text to prepend to each item. Alternative to prompt_template. "
        "Mutually exclusive with prompt_template.",
    )
    prompt_suffix: str | None = Field(
        default=None,
        description="Text to append after each item. Used with prompt_prefix.",
    )
    items: list[str] = Field(
        default_factory=list,
        description="List of items to expand the template with.",
    )
    resume_agent_ids: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping of existing agent ID to prompt for re-running failed sub-agents.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "AgentSwarmParams":
        if self.mode == "parallel_sample":
            if self.sample_n is not None and self.sample_n < 1:
                raise ValueError("sample_n must be >= 1.")
            uses_template = self.prompt_template is not None
            uses_prefix = self.prompt_prefix is not None
            if uses_template and uses_prefix:
                raise ValueError("Use either prompt_template or prompt_prefix+suffix, not both.")
            if not uses_template and not uses_prefix:
                raise ValueError(
                    "parallel_sample mode requires the task prompt via prompt_template "
                    "or prompt_prefix (+prompt_suffix)."
                )
            return self
        resume_count = len(self.resume_agent_ids) if self.resume_agent_ids else 0
        if len(self.items) < 2 and resume_count == 0:
            raise ValueError("Provide at least 2 items or resume_agent_ids.")
        total = len(self.items) + resume_count
        if total > _MAX_SUB_AGENTS:
            raise ValueError(f"Max {_MAX_SUB_AGENTS} sub-agents per swarm.")
        uses_template = self.prompt_template and "{{item}}" in self.prompt_template
        uses_prefix = self.prompt_prefix is not None
        if uses_template and uses_prefix:
            raise ValueError("Use either prompt_template or prompt_prefix+suffix, not both.")
        if not uses_template and not uses_prefix:
            raise ValueError(
                "prompt_template must contain '{{item}}', or set prompt_prefix. "
                "For example: prompt_template='Fix errors in {{item}}.'"
            )
        if uses_template and "{{item}}" not in self.prompt_template:
            raise ValueError(
                "prompt_template must contain '{{item}}'. "
                "For example: 'Fix all lint errors in {{item}}.'"
            )
        return self


class _RateLimiter:
    """Simple token-bucket rate limiter.

    Starts with ``burst`` tokens and refills 1 token every ``interval``
    seconds, capped at ``burst``.  The first ``burst`` acquisitions are
    therefore immediate.
    """

    def __init__(self, burst: int, interval: float) -> None:
        self.burst = burst
        self.interval = interval
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.burst, self._tokens + (now - self._last) / self.interval
            )
            if self._tokens < 1:
                wait_seconds = (1 - self._tokens) * self.interval
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
                self._tokens = min(
                    self.burst, self._tokens + (now - self._last) / self.interval
                )
            self._tokens -= 1
            self._last = time.monotonic()


class AgentSwarm(CallableTool2):
    name: str = "AgentSwarm"
    description: str = (
        "Dispatch a swarm of homogeneous sub-agents to execute tasks in parallel. "
        "Split a large request into small, independent items, provide a prompt "
        "template containing {{item}}, and receive an aggregated XML result."
    )
    params: type[BaseModel] = AgentSwarmParams

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        if not session.custom_data.get("is_swarm_session"):
            raise SkipThisTool()

    async def __call__(self, params: AgentSwarmParams) -> ToolReturnValue:
        # Recursive guard: sub-agents must not spawn further swarms.
        if self._session.custom_config.get("is_sub_agent"):
            return ToolError(
                output="",
                message="Recursive sub-agent swarm call detected.",
                brief="sub-agent recursively called AgentSwarm",
            )

        # Enforce a single AgentSwarm call per response.
        if self._session.custom_data.get("agent_swarm_in_flight"):
            return ToolError(
                output="",
                message="Another AgentSwarm call is already in progress.",
                brief="concurrent AgentSwarm call rejected",
            )

        self._session.custom_data["agent_swarm_in_flight"] = True
        try:
            return await self._execute(params)
        finally:
            self._session.custom_data.pop("agent_swarm_in_flight", None)

    async def _execute(self, params: AgentSwarmParams) -> ToolReturnValue:
        if params.mode == "parallel_sample":
            return await self._execute_parallel_sample(params)
        expanded_prompts = _expand_template(
            params.prompt_template, params.items,
            prefix=params.prompt_prefix, suffix=params.prompt_suffix,
        )
        _validate_uniqueness(expanded_prompts)

        tasks: list[SwarmTask] = [
            SwarmTask(prompt=prompt, agent_id=None, index=index)
            for index, prompt in enumerate(expanded_prompts)
        ]
        if params.resume_agent_ids:
            offset = len(tasks)
            for rel_index, (agent_id, prompt) in enumerate(
                params.resume_agent_ids.items()
            ):
                tasks.append(
                    SwarmTask(prompt=prompt, agent_id=agent_id, index=offset + rel_index)
                )

        results = await _run_swarm(tasks, params.subagent_type, self._session)
        results.sort(key=lambda r: r.index)
        return ToolOk(output=_render_results(results, params.description))

    async def _execute_parallel_sample(self, params: AgentSwarmParams) -> ToolReturnValue:
        """Best-of-N: sample the same task N times, select, apply, report."""
        from kaos.path import KaosPath

        from kimix.tools.swarm import best_of_n as bon

        if params.prompt_template is not None:
            task_prompt = params.prompt_template.replace("{{item}}", "").strip()
        else:
            task_prompt = f"{params.prompt_prefix or ''}{params.prompt_suffix or ''}".strip()

        n = params.sample_n if params.sample_n is not None else 4
        strategy = params.selector or "self_eval"
        parent = self._session
        subagent_type = params.subagent_type

        async def _sample_runner(prompt: str, worker_dir: Path) -> tuple[str, int, int]:
            """Run one sample as a sub-agent rooted in the worker workspace."""
            task = SwarmTask(prompt=prompt, agent_id=None, index=0)
            session, session_id, resolved_prompt = await _resolve_subagent_session(
                task, subagent_type, parent
            )
            _ = session_id
            collector = _AgentConversationCollector()
            collector.finalize_user_turn(resolved_prompt)
            try:
                await utils.prompt_async(
                    prompt_str=resolved_prompt,
                    session=session,
                    output_function=lambda text, msg_type: collector.consume(text, msg_type) if text else None,
                    info_print=False,
                    merge_wire_messages=True,
                    format_output=True,
                )
                report = collector.finalize_assistant_turn() or "(no text output)"
                return report, 0, 0
            finally:
                try:
                    await utils.close_session_async(session)
                except Exception:
                    pass

        async def _selector(prompt: str, review_text: str) -> int:
            """Model self-evaluation: one review sub-agent picks a candidate index."""
            review_prompt = (
                "You are reviewing multiple candidate solutions for the same task.\n\n"
                f"Task:\n{prompt}\n\n{review_text}\n\n"
                "Reply with ONLY the integer index of the best candidate "
                "(most correct, complete, and verified). No explanation."
            )
            report, _, _ = await _sample_runner(review_prompt, worker_dir=Path.cwd())
            import regex as _re

            match = _re.search(r"-?\d+", report)
            return int(match.group(0)) if match else 0

        work_dir = Path(str(KaosPath.cwd())) if hasattr(KaosPath, "cwd") else Path.cwd()
        try:
            result = await bon.best_of_n(
                task_prompt,
                work_dir,
                _sample_runner,
                _selector,
                n=n,
                strategy=strategy,
            )
        except bon.AllCandidatesFailedError as exc:
            return ToolError(output="", message=str(exc), brief="all samples failed")
        except bon.VerificationRejectedError as exc:
            return ToolError(output="", message=str(exc), brief="selected sample failed verification")

        lines = [
            "<best_of_n_result>",
            f"  <description>{_xml_escape(params.description)}</description>",
            f"  <samples>{len(result.candidates)}</samples>",
            f"  <winner>{result.winner_index}</winner>",
            f"  <selection>{_xml_escape(result.selection_reason)}</selection>",
        ]
        for candidate in result.candidates:
            status = "ok" if candidate.success else f"failed: {_xml_escape(candidate.error or '')}"
            lines.append(
                f'  <candidate index="{candidate.index}" status="{_xml_escape(status)}"/>'
            )
        lines.append("</best_of_n_result>")
        return ToolOk(output="\n".join(lines))


def _expand_template(template: str | None, items: list[str], prefix: str | None = None, suffix: str | None = None) -> list[str]:
    if template is not None and "{{item}}" in template:
        return [template.replace("{{item}}", item) for item in items]
    elif prefix is not None:
        suffix_str = suffix or ""
        return [f"{prefix}{item}{suffix_str}" for item in items]
    return list(items)


def _validate_uniqueness(prompts: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for prompt in prompts:
        if prompt in seen:
            duplicates.add(prompt)
        seen.add(prompt)
    if duplicates:
        raise ValueError(f"Expanded prompts must be unique; duplicates: {duplicates}")


def _xml_escape(text: str) -> str:
    return html.escape(text, quote=True)


def _render_results(results: list[SwarmSubagentResult], description: str) -> str:
    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count
    lines: list[str] = ["<agent_swarm_result>"]
    lines.append(f"  <description>{_xml_escape(description)}</description>")
    lines.append(f"  <total>{len(results)}</total>")
    lines.append(f"  <succeeded>{success_count}</succeeded>")
    lines.append(f"  <failed>{failed_count}</failed>")
    if failed_count > 0:
        lines.append(
            "  <resume_hint>Some sub-agents failed. Re-run with resume_agent_ids "
            "mapping the failed agent IDs to adjusted prompts.</resume_hint>"
        )
    lines.append("  <subagents>")
    for result in results:
        success_str = "true" if result.success else "false"
        elapsed_str = f'{result.elapsed:.1f}s' if result.elapsed else '-'
        lines.append(
            f'    <subagent id="{_xml_escape(result.agent_id)}" index="{result.index}" '
            f'success="{success_str}" elapsed="{elapsed_str}">'
        )
        lines.append(f"      <output>{_xml_escape(result.output)}</output>")
        if result.error:
            lines.append(f"      <error>{_xml_escape(result.error)}</error>")
        lines.append("    </subagent>")
    lines.append("  </subagents>")
    lines.append("</agent_swarm_result>")
    return "\n".join(lines)


async def _run_swarm(
    tasks: list[SwarmTask], subagent_type: str, parent_session: Session
) -> list[SwarmSubagentResult]:
    raw_max = os.environ.get("KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY")
    if raw_max is not None:
        try:
            max_concurrency = max(1, int(raw_max))
        except ValueError:
            max_concurrency = _DEFAULT_BURST
    else:
        max_concurrency = _DEFAULT_BURST

    semaphore = asyncio.Semaphore(max_concurrency)
    rate_limiter = _RateLimiter(
        burst=min(_DEFAULT_BURST, max_concurrency),
        interval=_DEFAULT_INTERVAL_SECONDS,
    )

    async def _run_one(task: SwarmTask) -> SwarmSubagentResult:
        await rate_limiter.acquire()
        async with semaphore:
            return await _run_subagent_task(task, subagent_type, parent_session)

    coroutines = [_run_one(task) for task in tasks]
    return await asyncio.gather(*coroutines)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of rate-limit / capacity errors."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "rate-limit",
            "too many requests",
            "429",
            "capacity",
            "throttled",
            "quota exceeded",
        )
    )


async def _run_subagent_task(
    task: SwarmTask, subagent_type: str, parent_session: Session
) -> SwarmSubagentResult:
    session: Session | None = None
    session_id: str | None = task.agent_id
    last_error: Exception | None = None
    start_time = time.monotonic()
    try:
        session, session_id, task_prompt = await _resolve_subagent_session(
            task, subagent_type, parent_session
        )

        for attempt in range(_MAX_RETRIES + 1):
            collector = _AgentConversationCollector()
            collector.finalize_user_turn(task_prompt)

            def output_function(text: str, msg_type: MessageType) -> None:
                if text:
                    collector.consume(text, msg_type)

            try:
                await utils.prompt_async(
                    prompt_str=task_prompt,
                    session=session,
                    output_function=output_function,
                    info_print=False,
                    merge_wire_messages=True,
                    format_output=True,
                )
                output_text = collector.finalize_assistant_turn()
                if not output_text:
                    output_text = "(no text output)"
                elapsed = time.monotonic() - start_time
                return SwarmSubagentResult(
                    index=task.index,
                    agent_id=session_id,
                    output=output_text,
                    success=True,
                    elapsed=elapsed,
                )
            except Exception as exc:
                last_error = exc
                if attempt < _MAX_RETRIES and _is_rate_limit_error(exc):
                    await asyncio.sleep(_RETRY_BASE_SECONDS * (2**attempt))
                    continue
                raise

        # Unreachable; satisfies type checker.
        raise last_error or RuntimeError("sub-agent task failed")
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        return SwarmSubagentResult(
            index=task.index,
            agent_id=session_id or task.agent_id or "unknown",
            output=str(exc),
            success=False,
            error=str(exc),
            elapsed=elapsed,
        )
    finally:
        if session is not None:
            try:
                await utils.close_session_async(session)
            except Exception:
                pass


async def _resolve_subagent_session(
    task: SwarmTask, subagent_type: str, parent_session: Session
) -> tuple[Session, str, str]:
    custom_config = parent_session.custom_config
    chat_provider = custom_config.get("chat_provider")
    default_sub_provider = (
        base.get_default_sub_provider("sub_agent")
        or custom_config.get("provider_dict", base._default_provider)
    )

    session_id = task.agent_id or str(uuid.uuid4())
    agent_type = _SUBAGENT_TYPE_MAP.get(subagent_type, SystemPromptType.Worker)

    # Offload very long prompts to a temp file, matching the Agent tool pattern.
    prompt_bytes = task.prompt.encode("utf-8")
    if len(prompt_bytes) > 100 * 1024:
        cache_dir = Path(".kimix_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_dir / f"prompt_{uuid.uuid4().hex}.md"
        temp_path.write_bytes(prompt_bytes)
        task_prompt = f"Please read the task from `{temp_path}` and execute it."
    else:
        task_prompt = task.prompt

    session = await utils._create_session_async(
        session_id=session_id,
        work_dir=parent_session.work_dir,
        agent_file=base._default_agent_file_dir / "agent_subagent.json",
        agent_type=agent_type,
        provider_dict=default_sub_provider,
        chat_provider=chat_provider,
        resume=task.agent_id is not None,
        anonymous=False,
        max_ralph_iterations=0,
    )

    sub_custom_config = session.get_custom_config()
    if sub_custom_config is not None:
        sub_custom_config["is_sub_agent"] = True

    return session, session_id, task_prompt
