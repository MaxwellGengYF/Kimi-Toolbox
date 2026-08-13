# Kimi-CLI-X

> [中文文档](README_zh.md)

## Install from Source
```bash
python install.py
```

## pip Install
```bash
pip install kimix
python -m kimix.cli
# or
kimix
python -m kimix
```

> **Note:** This repo supports not only KIMI LLM but also various API keys! Like OpenAI, Anthropic, etc. Default config templates are in `docs/`; use `kimix --config=xx.json` after setup.

![teasor](kimix_en.png)

## Why Kimi-CLI-X?

Kimi-CLI-X is a deep optimization of the original Kimi-CLI, focusing on **prompt efficiency**, **tool reliability**, and **extensibility**, plus new tools for real-world development.

### Optimizations

1. **Lean system prompts** — Compressed initial prompts and tool descriptions down to ~2000 tokens while covering nearly all built-in tools.
2. **Hardened permissions & validation** — Properly handles Shell, Glob, and other tool validations to reduce retry loops from failures.
3. **Better subprocess output** — Redirects large outputs to temp files and filters redundant logs for easier backend retrieval.
4. **Simpler concurrency** — Streamlined design for subprocesses, sub-agents, and background tasks.
5. **Programmable prompts** — Allows custom system prompt injection at the upper layer for flexible scenarios.
6. **Explicit conversation management** — Clearer multi-task orchestration and state tracking.
7. **Write-and-validate** — Auto format checks and warnings on strict config files to prevent model-hallucinated errors.
8. **Multi-API support** — Import custom configs compatible with OpenAI, Anthropic, and more.
9. **Verified backends** — Tested against kimi, anthropic, openai_legacy, openai_responses, google_genai, vertexai, etc. See `kimi-cli/tests/core/test_create_llm.py`.

### New Capabilities

| Capability | Description |
|------------|-------------|
| **Interactive shell tools** | Start and continue `Bash`/`Powershell`/`Run` sessions via `task_id`, with optional `wait_for_pattern`. |
| **Docx / PDF conversion** | Built-in document conversion without external deps. |
| **Python script execution** | Agent can run Python scripts directly. |
| **Error logging** | Records tool-call errors for model backtracking and improvement. |
| **Script system** | Combines prompts with Python logic to orchestrate complex tasks. |
| **Enhanced web fetch (fetch_url)** | Headless-browser-based Markdown output (not plain text), supports `output_path` and auto-truncation for超长 content; zero external service dependency. |
| **Best-of-N sampling (`AgentSwarm` `parallel_sample`)** | Run the SAME task N times in isolated workspaces (git worktree / temp copy), pick the winner via `self_eval` or `majority` selection, then apply and verify the winning diff — never a silent accept. |

### Scriptable Workflows (Core Advantage)

Unlike traditional CLI interaction where you type commands one by one, **Kimi-CLI-X lets you write Python scripts to orchestrate entire workflows**. You can combine prompts, loops, conditionals, and tool calls into fully automated, reproducible task pipelines:

```python
from kimix import *
from pathlib import Path

clear_default_context()

for i in Path('docs').glob('*.md'):
    prompt(f'''According to the new git commits, update document `{i}`''')
```

Benefits:

- **Batch automation**: Use native Python syntax (`for` loops, file globbing) to fire tasks at multiple files at once.
- **Complex orchestration**: Freely compose mode switches, tool calls, and logic into multi-stage, multi-branch workflows.
- **Reproducible & maintainable**: Workflows live as version-controlled scripts, not ephemeral chat history.

---

### Context Memory Architecture

Kimi-CLI-X embeds an **automatic context memory system** inside the `KimiSoul` core loop, keeping long conversations coherent without manual intervention. Three layers work together:

#### 1. Conversation History Index (HistoryIndex)

Every user/assistant message is automatically indexed by **BM25 inverted index** (N-gram, n=2) on append, persisted to `<session>/history_index/<id>.json`, and survives process restarts. Cap at 500 rounds; oldest evicted automatically.

#### 2. Automatic Context Compaction (SimpleCompaction)

Triggered when context token ratio hits `compaction_trigger_ratio` or free space falls below `reserved_context_size`:

- **Retention policy**: Recent N rounds kept verbatim (depth adapted by `adaptive_preserve_depth` — deepened on errors, thinking, multi-file edits, etc.); first message always kept (primacy effect).
- **LLM summarization**: Old messages compressed into structured summaries via a lightweight LLM call; thinking blocks discarded.
- **Cascade handling**: When already-compacted content is compressed again (depth ≥3), switches to `COMPACT_CASCADE` prompt to prevent information degradation.
- Post-compaction, all rounds marked `is_compacted` in HistoryIndex for future retrieval.

#### 3. Auto History Retrieval + On-Demand Recall

- **Auto retrieval** (`_maybe_auto_retrieve_history`): Each round, if user input ≥10 chars, BM25-searches HistoryIndex for matching compacted rounds; injects matches above `auto_retrieve_history_threshold` as `[Auto-retrieved from past conversation]`.
- **`retrieve` tool**: the agent can actively search all archived history (including compacted rounds) by natural-language query, returning verbatim excerpts with relevance scores (or fetch a turn by `id`).

```
┌──────────────┐    append     ┌──────────────┐    overflow    ┌──────────────────┐
│   Context    │ ───────────► │ HistoryIndex │ ────────────► │ SimpleCompaction │
│ (live window)│              │ (BM25 index) │               │ (LLM summary)    │
└──────────────┘              └──────────────┘               └──────────────────┘
       ▲                            │                               │
       │       auto-retrieve        │                               │
       └────────────────────────────┘                               │
       │              Retrieve (agent主动recall)                       │
       └────────────────────────────────────────────────────────────┘
```

---

### Agent Harness: Self-Regulation & Reminders

The `KimiSoul` core loop actively keeps long runs on track — no manual babysitting. It works in CLI, server, and sub-agent sessions alike.

- **Verification Gate** — a turn can't end while todos are unfinished, or while files were edited without running any check. Failing checks are fed back to the agent to fix.
- **Anti-loop detection** — catches repeated edits to the same file across different tools, and the same error recurring without a root-cause fix; nudges the agent to change strategy.
- **Todo reminders** — unfinished todos are periodically re-surfaced at the end of the context, so goals never drift out of attention.
- **Compact reminders** — when context fills up (~70%), the agent is prompted to compact on its own terms before forced auto-compaction kicks in.
- **Budget reminders** (opt-in) — wrap-up warnings as the per-turn step/time budget runs out, so the agent finishes gracefully instead of being cut off.
- **Context meter** — when context usage shifts materially, the agent is reminded to recall past history with the `Retrieve` tool.
- **Decision-aware compaction** — compaction summaries preserve a `Decisions & Conclusions` and a `Verification Status` section, so early decisions and verified work survive.
- **Context pruning** — stale tool outputs, thinking blocks, and near-duplicate content are automatically elided to reclaim context space.

### TodoList
The `TodoList` tool tracks multi-step plans:
- Incremental updates with append/overwrite modes, fuzzy title matching, and per-todo notes.
- Nested sub-todos via `todo_push`/`todo_sub`/`todo_pop` with a `Stack:` breadcrumb.

### Best-of-N Sampling

`AgentSwarm`'s `parallel_sample` mode runs the **same task N times** in isolated workspaces (git worktree / temp copy), picks the winner by model self-evaluation or majority vote, then applies and verifies the winning diff. Failures are explicit errors — never silently accepted.

---

## Documentation Index

### Tutorials

| Document | Description |
|----------|-------------|
| [`docs/tutorials/1_quick_start_en.md`](docs/tutorials/1_quick_start_en.md) | Quick start guide: Git submodules, `uv` env setup, CLI args, and interactive commands. |
| [`docs/tutorials/2_long_task_en.md`](docs/tutorials/2_long_task_en.md) | Long task strategy in KimiX. |
| [`docs/tutorials/3_builtin_tools_en.md`](docs/tutorials/3_builtin_tools_en.md) | Complete built-in tool guide: file I/O, search, code execution, process management, doc conversion, plan mode, sub-agents, plus prompt strategies and best practices. |
| [`docs/tutorials/4_skills_en.md`](docs/tutorials/4_skills_en.md) | Custom skill authoring: design principles, directory structure, `SKILL.md` spec, resource organization, testing, packaging, and installation. |
| [`docs/tutorials/5_server_en.md`](docs/tutorials/5_server_en.md) | HTTP server tutorial: FastAPI + SSE, OpenCode-compatible REST API, session management, event streaming, SSE CLI debugger, dummy mode, and client implementation. |
| [`docs/tutorials/6_multi_provider_en.md`](docs/tutorials/6_multi_provider_en.md) | Multi-provider configuration: route sub-agents and planner to different LLM providers with role-tagged `sub_providers`. |

### Config Reference

| File | Description |
|------|-------------|
| [`docs/config.json`](docs/config.json) | Sample model config with `model`, `url`, `api_key`, `capabilities`, etc. |
| [`.kimix/config.json`](.kimix/config.json) | Workspace behavior config: `protected_write_paths`, `protected_read_paths`, `forbidden_commands`, etc. |
| [`.kimix/skill.json`](.kimix/skill.json) | Workspace skill directory config: `skill_dir` field (string or array) for extra skill directories, resolved relative to workspace. |
