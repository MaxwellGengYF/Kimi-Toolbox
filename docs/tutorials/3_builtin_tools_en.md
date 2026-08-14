# Built-in Tools Guide

A coding agent's power comes from efficient interaction with the environment. This guide covers all 26 built-in tools in `agent_worker.json` and how to prompt the agent to use them effectively.

> **Note:** `agent_worker.json` overrides the `tools` field via `extend: default`, so only the 26 tools listed there are available.

---

## Tool Overview

| Category | Tools | Typical Use |
|----------|-------|-------------|
| **File & I/O** | `write`, `read`, `edit`, `glob`, `grep` | Create, read, modify, search files |
| **Code Execution** | `Run`, `Python`, `Bash`, `pwsh` | Execute executables, bash / powershell commands, or Python code |
| **Process Management** | `job_output` | Read, list, export, or kill background tasks |
| **Search & Info** | `fetch_url` | Fetch web content |
| **State & Tracking** | `todo_write`, `todo_update` | Track progress |
| **Sub-agent & Session Management** | `subagent`, `list_agents`, `interrupt_agent` | Create, list, and close sub-agent sessions |

---

## File & I/O

#### `write`
Write to a file. Modes: `overwrite` (default), `append`. For content >100 lines, split into multiple calls (first `overwrite`, rest `append`).

#### `read`
Read text files by line. Options: `offset`, `limit`, negative offset for tail reading. Long lines are auto-truncated. Read large files in chunks.

#### `edit`
String-level replacement in text files. Supports single/multi-line edits and `replace_all`. **Preferred for minimal diffs** — preserves formatting, comments, and blank lines.

#### `glob`
Wildcard file search (`*`, `?`). Avoid `**` prefix or very large directories.

#### `grep`
Regex content search (ripgrep-powered). Options: `-i` (case-insensitive), `multiline`, `-B`/`-A`/`-C` (context), `type`/`include` filters.

---

## Code Execution & Process Management

#### `Run`
Execute programs or built-in mapped commands (100+ commands: `cat`, `ls`, `grep`, `find`, `curl`, `git`, etc.). Options: `args`, `cwd`, `output_path`, `timeout` (default 10s, range 3–180s). Exceeds timeout → background task with `task_id`.

> Not a shell interpreter — runs executables directly or calls internal mapped command implementations for safer, more predictable behavior.

#### `Bash`
Execute commands or script snippets in a bash shell, supporting shell features such as pipes, redirection, and variable expansion.
- **Platform**: mainly Linux / macOS.
- **Use cases**: running shell scripts, combining commands with pipes, complex commands requiring shell interpretation.
- **Interactive mode**: set `interactive=True` to start a persistent bash session. The tool returns a `task_id` immediately. To continue, call `Bash` again with `task_id=<id>` and `cmd` set to the input text; output is returned in the same call. Use `wait_for_pattern` to block until a prompt appears. Send `exit` to close the session.

#### `pwsh`
Execute commands or script snippets in PowerShell, supporting Windows-specific commands and pipelines.
- **Platform**: mainly Windows.
- **Use cases**: Windows management commands, calling .NET tools, handling cross-platform script compatibility on Windows.
- **Interactive mode**: set `interactive=True` to start a persistent PowerShell session. The tool returns a `task_id` immediately. To continue, call `pwsh` again with `task_id=<id>` and `cmd` set to the input text; output is returned in the same call. Use `wait_for_pattern` to block until a prompt appears. Send `exit` to close the session.

#### `Python`
Execute Python code in a subprocess. Params: `code` (required), `output_path`, `timeout` (default 10s, range 3–60s). Exceeds timeout → background task. Max 8 concurrent Python processes. Code >30000 chars auto-saved to temp `.py` file.

#### `job_output`
Get output from background tasks. Supports blocking wait, polling, `kill`, and `output_path` export.

#### Interactive sessions with `Bash`, `pwsh`, and `Run`
These tools can start a persistent session and continue it in later turns using the same tool:

1. Start: `Bash` / `pwsh` with `interactive=True`, or `Run` with `run_in_background=True`. The response includes a `task_id`.
2. Continue: call the same tool with `task_id=<id>` and `cmd`/`command` set to the input text. The input is sent to the process stdin and the accumulated output is returned in the same call.
3. Wait for a prompt: supply `wait_for_pattern` with a regex; the tool blocks up to `timeout` until the pattern appears.
4. Close: send the shell-specific exit command (e.g., `exit`) via `task_id` + `cmd`/`command`.

`job_output` remains available as a fallback to read, list, export, or kill background tasks without sending input.

---

## Search & Information

#### `fetch_url`
Fetch web content as Markdown via headless browser. Use for docs, API references, GitHub issues.

---

## State & Tracking

#### `todo_write`
Track multi-step task progress. States: `pending`, `in_progress`, `done`. Always pass the **complete list** on update. For lightweight single-item edits, prefer `todo_update`.

#### `todo_update`
Update a single existing todo by title without rewriting the whole list. Supports status changes, notes edits, and renaming. Fuzzy matching is enabled by default.


---

## Sub-agent & Session Management

#### `subagent`
Spawn an independent sub-agent for a specific subtask. Use for parallel work: code review, translation, module development.

#### `list_agents`
List all currently active sub-agent sessions. Use after spawning multiple sub-agents to see which are still running or waiting for input.

#### `interrupt_agent`
Close a specified sub-agent session and release its resources. Use when a sub-agent task completes or hangs abnormally.

---

## Prompting Strategies

### 1. Direct Instruction
Explicitly name tools and their purpose.

> "Use `glob` to find all `.cpp` files under `src/`, then `read` each to check for `deprecated` markers. Write results to `report.md` with `write`."

### 2. Goal-Oriented
Describe the goal, let the agent choose tools.

> "Find this project's entry point and its third-party dependencies. Search the codebase and report back."

### 3. Constrained Execution
Add explicit constraints.

> "Change `MAX_RETRIES` to `5` in `config.py`. Requirements:
> 1. Use `edit` for minimal changes
> 2. `read` first to confirm line numbers
> 3. `read` again after editing to verify"

### 4. Step-by-Step Workflow
Break complex tasks into tool-annotated steps.

> "1. **Research**: `glob` + `read` existing CLI commands
> 2. **Implement**: `write` or `edit` for new command
> 3. **Verify**: `Run` tests
> 4. **Track**: `todo_write` to mark complete"

### 5. Meta-Prompting
Embed tool guidelines in system prompts.

> "- **Observe before acting**: `read`/`grep` before modifying
> - **Minimal changes**: prefer `edit`, avoid full-file overwrites
> - **Async long tasks**: `Run` + `job_output` for >10s commands
> - **Delegate**: use `subagent` for independent subtasks"

---

## Best Practices

**Refactor a function safely:**
1. `grep` for all occurrences
2. `read` to verify context (avoid string false-positives)
3. `edit` for replacements
4. `grep` again to verify no leftovers
5. `Run` tests

**Interactive command:**
1. `Bash` / `pwsh` / `Run` to start a session (timeout → background task)
2. Reuse the same tool with `task_id` and `wait_for_pattern` to exchange input/output
3. `job_output` as a fallback to read or kill tasks
4. Loop until complete

**Multi-file feature:**
1. `glob` + `read` to research existing structure
2. Draft implementation plan
3. `edit`/`write` changes
4. `todo_write` track subtasks
5. `Run` tests

**External document analysis:**
1. `fetch_url` for web docs
2. Extract requirements
3. `grep` + `read` verify implementation
4. Output diff report

---

## Plan Mode

Plan mode (`/plan`) is a two-stage workflow that separates **planning** from **implementation**:

1. **Generate** — a specialized planner agent (loaded from `agent_planner.json`) analyzes your requirement and writes a comprehensive plan to a Markdown file.
2. **Implement** — after you review the plan, a regular worker agent executes it.

#### Usage

```
/plan              # writes to plan.md (default)
/plan:roadmap.md   # writes to a custom file
```

After invoking the command, type your requirement. End input with `/end`, or cancel with `/cancel`.

#### Workflow

1. **Input requirement** — describe what you want to build or refactor.
2. **Planner generates plan** — the planner uses the Note tool to save the plan and auto-opens the file for review.
3. **Confirm implementation** — answer `y` to hand the plan to a worker agent; answer `n` to enter the revision flow and describe changes for the Planner to update the plan; input `/quit` to abandon execution.

#### Why use Plan Mode?

- **Complex features** — multi-file refactors, architecture changes, or new modules benefit from upfront design.
- **Review before commit** — inspect the plan, adjust scope, or split work before any code is written.
- **Delegation** — the planner acts as a dedicated architect, while the worker focuses on execution.

---

## Summary

1. **Observe first**: `read` / `grep` / `glob` before modifying
2. **Minimize changes**: `edit` preferred; `write` only for new files or full rewrites
3. **Async long tasks**: `Run` timeout → background, manage via `job_output`
4. **Integrate external info**: `fetch_url` for docs and references
5. **Plan before build**: use `/plan` for complex tasks to separate design from implementation
