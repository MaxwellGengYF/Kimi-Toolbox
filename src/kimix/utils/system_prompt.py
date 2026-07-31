from typing import Optional, Callable, Any
from pathlib import Path
import os
import orjson
from enum import Enum
from kaos.path import KaosPath
import kimix.base as base
from kimi_cli.soul.agent import BuiltinSystemPromptArgs
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.reason import ToolCallReason
from kimi_cli.utils.tokens import count_tokens

# Concise system prompt to reduce LLM overthinking and hallucination
_SYSTEM_PROMP = (
    '{AGENT_ROLE}:\n{NUMBERED}\n{AGENTS_MD}{SKILLS}'
)


class SystemPromptType(Enum):
    Worker = 0
    TodoMaker = 1
    Thinker = 2
    TrivialSubAgent = 3
    Supervisor = 4
    Reader = 5
    SwarmLeader = 6


class SystemPromptCallback:
    # called in role
    role_callback: Callable[[SystemPromptType, list[str]], None] | None = None


def _shell_tool_name() -> str:
    """Return the shell tool name active for this agent.

    The agent config's ``agent.shell`` key (e.g. ``"shell": "powershell"`` in
    ``agent_worker.json``) selects the shell tool: the enablement logic in
    ``kimix.tools.file.bash.bash_tool`` — ``_should_enable_bash`` and
    ``_should_enable_powershell`` — is config-aware, so the prompt always
    names the shell tool that is actually loaded.
    """
    from kimix.tools.file.bash.bash_tool import (
        _should_enable_bash,
        _should_enable_powershell,
    )

    if _should_enable_powershell():
        return 'Powershell'
    if _should_enable_bash():
        return 'Bash'
    return 'Run'


def get_system_prompt(
        yolo: bool | None = None,
        work_dir: Optional[KaosPath] = None,
        extra_system_prompt: SystemPromptCallback | None = None,
        agent_role: SystemPromptType = SystemPromptType.Worker,
        max_system_prompt_tokens: int = 4_000,
) -> Callable[[BuiltinSystemPromptArgs], str]:
    agent_md = (Path(str(work_dir)) if work_dir is not None else Path(
        os.curdir)) / 'AGENTS.md'
    yolo = yolo if yolo is not None else base._default_yolo


    def system_prompt_func(runtime: Runtime, is_compacting: bool = False, compact_export_path: str | None = None) -> str:
        args = runtime.builtin_args
        items: list[str] = []
        agent_md_doc = ''
        skill_doc = ''
        use_agent_md = False
        use_skills = False
        items.append('Call tools in parallel.')
        items.append(f'OS: {args.KIMI_OS}, Work-Dir: {args.KIMI_WORK_DIR}')
        if args.KIMI_OS == 'Windows':
            items.append(r'Windows paths use backslashes (`\`). Always use `\` instead of `/` for file paths.')
        def worker_logic(role: str, is_sub_agent: bool = False):
            nonlocal role_doc, use_agent_md, use_skills
            use_agent_md = True
            use_skills = True
            role_doc = f'You are a {role}'
            items.append('DO NOT use your own knowledge. Read the provided references, skills, and files first, then judge and act strictly from the evidence you read.')
            items.append('Persist: finish all requirements, keep trying until done.')
            items.append('One action per turn: each response = exactly one tool call, edit, or verification.')
            shell_tool = _shell_tool_name()
            items.append(f'For long commands, use `Python` instead of `{shell_tool}`.')
            items.append('Error recovery: retry, adjust approach, or break into sub-tasks. Never give up.')
            items.append('Verification gate: run all tests/checks and confirm they pass before finishing.')
            items.append('After completing an independent task, reach milestone, or finishing one part of the schedule, call `Compact` before the next step.')
            items.append('Multi-step: use `TodoList`. Finish all before ending. For any todo involving code changes, attach verification `code` — prefer the project\'s own tests in `!shell` form (e.g. `!pytest ...`); if none exist, write a minimal reproduction script. Never declare completion from reading code alone: all verification must actually run and pass before the todo is marked `done`.')
            if not is_sub_agent:
                if yolo:
                    items.append('Yolo: no asking. accept all. When choices or options appear, independently pick the best one and continue; do not ask the user which to choose.')

                items.append("Use `Memory` (action='retrieve') whenever not sure about the history and memory.")
            else:
                items.append('Sub-Agent: only report results. If any option, output the question and stop.')
        if extra_system_prompt and extra_system_prompt.role_callback:
            extra_system_prompt.role_callback(agent_role, items)

        match agent_role:
            case SystemPromptType.Worker:
                worker_logic('persistent autonomous agent')
            case SystemPromptType.TodoMaker:
                use_agent_md = True
                use_skills = True
                role_doc = 'You are a planner'
                items.append('Plan only. Do not implement.')
                items.append('Record comprehensive and detailed plan with `WritePlan` `EditPlan`.')
                items.append('For each phase of the plan, include file path reference.')
                items.append('You cannot write files or run commands. If any requirements need those abilities, reject them in `WritePlan` `EditPlan` and stop.')
            case SystemPromptType.Thinker:
                worker_logic('thinker')
                items.append(
                    'Think in <thinking>...</thinking>. End with <quit/>. Concise. No text outside tags.')
                items.append('Self-verify: catch errors and bad assumptions.')
            case SystemPromptType.TrivialSubAgent:
                worker_logic('persistent autonomous sub-agent', True)
                items.append('If you need clarification from the parent agent, call the `ask_parent` tool with your question, then stop.')
            case SystemPromptType.Reader:
                role_doc = 'You are a reader'
                items.append('Read the given content and report a concise summary: key results, errors, warnings, and next steps.')
                items.append('No commands, edits, or questions.')
                items.append('For large content, cover the most relevant parts and note omissions.')
            case SystemPromptType.Supervisor:
                use_agent_md = True
                use_skills = True
                role_doc = 'You are a supervisor'
                # Supervisor: outline → decompose → dispatch → track → accept/inquire/correct → verify.
                items.append('Outline goals, constraints, unknowns, acceptance criteria before delegating.')
                items.append('Decompose into non-overlapping tasks (Explorer/Worker/Reviewer/Verifier). Serial if same output.')
                items.append('Dispatch via `Agent` with role, goal, scope, non-goal, inputs, acceptance criteria.')
                items.append('Never do sub-agent work yourself. Route failures through inquiry, then narrow correction.')
                items.append('Track with `TodoList`. Accept or inquire/reject each result against criteria.')
                items.append('After all accepted and merged, run one overall verification suited to task type.')
                items.append('Final: report tasks, deliverables, verification result, unresolved work, merged conclusion.')
            case SystemPromptType.SwarmLeader:
                use_agent_md = True
                use_skills = True
                role_doc = 'You are a swarm orchestrator'
                items.append('The user wants to parallelize a request across multiple homogeneous sub-agents.')
                items.append('Analyze the request and decide how to split it into independent, homogeneous sub-tasks.')
                items.append('Call the `AgentSwarm` tool with: a clear description, subagent_type (coder/explore/plan), a prompt_template containing the placeholder {{item}}, and a list of items.')
                items.append('Do not implement the tasks yourself; only dispatch and summarize the aggregated result.')
                items.append('If the request cannot be parallelized, explain why and stop.')

        def _build_agent_md_doc() -> str:
            if use_agent_md and agent_md.is_file():
                agent_md_content = agent_md.read_text(
                    encoding='utf-8', errors='replace')
                if len(agent_md_content.encode('utf-8')) > 4096:
                    return 'read AGENTS.md before work\n'
                return f'AGENTS.md:\n```\n{agent_md_content}\n```\n'
            return ''

        if compact_export_path:
            items.append(f'Pre-compaction context exported to: {compact_export_path}')

        if use_skills and args.KIMI_SKILLS:
            skill_doc = f'Skills:\n{args.KIMI_SKILLS}\n'
        numbered_block = ''
        if items:
            numbered_block = ''.join(
                f'- {item}\n' for item in items
            )
        agent_md_doc = _build_agent_md_doc()
        if count_tokens(agent_md_doc) >= max_system_prompt_tokens:
            agent_md_doc = 'read AGENTS.md before work\n'
        prompt = _SYSTEM_PROMP.format(
            AGENT_ROLE=role_doc.strip(),
            NUMBERED=numbered_block,
            AGENTS_MD=agent_md_doc,
            SKILLS=skill_doc,
        ).strip()
        return prompt

    return system_prompt_func
