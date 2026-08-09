---
name: kimix_api
description: Guide for using KimiX API utilities covering kimix, kimix.utils, kimix.base, kimix.dag, kimix.network, kimix.server, kimix.parser, kimix.tools, kimix.cot, kimix.retrieval, kimix.summarize, and kimi_agent_sdk.
---

# Kimi API Utilities Guide

This guide explains how to use the utility functions from `kimix.utils`, `kimix.base`, and the full public API surface of `src/kimix` and `src/kimi_agent_sdk`.

## Initialization (kimix.utils.config)

### init

Initialize kimix global state. **Optional for scripts**: when ``prompt()``/``create_session()`` are called without ``init()`` first, the provider is auto-resolved from ``src/kimix/default_config.json``, then ``KIMI_API_KEY``/``KIMIX_API_KEY`` env vars, then ``~/.kimi`` OAuth credentials (kimi-code). Call ``init()`` when you need to override yolo/think/skill dirs or supply an explicit provider config.

```python
# File: src/kimix/utils/config.py
from kimix.utils import init

init(
    config_path="config.json",       # Optional: path to a JSON config file (provider settings)
    config_json=None,                # Optional: JSON string with provider settings (alternative to config_path)
    yolo=True,                       # Enable YOLO mode (auto-approve tool calls)
    think=True,                      # Enable thinking mode
    skill_dir=None,                  # Optional: additional skill directories to load
    ralph=None,                      # Optional: Ralph mode iteration limit
    manually_cot=False,              # Enable manually CoT mode
    colorful_print=True,             # Enable ANSI-colorful output
    clean=False,                     # Delete session cache after quit
)
```

**What `init` does (in order):**
1. Disposes the existing default session (if any) via `close_session`.
2. Resets base globals: `_colorful_print`, `set_default_thinking(think)`, `set_default_yolo(yolo)`, `set_default_manually_cot(manually_cot)`, `_default_ralph = ralph`.
3. Sets `CLEAN_MODE` when `clean=True` (deletes cache after quit).
4. Resets skill dirs to empty, auto-loads `.kimix/skill.json` (key `skill_dir`, relative paths resolved against CWD), then appends any explicit `skill_dir` entries.
5. Loads provider config from `config_path` / `config_json`, or falls back to `src/kimix/default_config.json`. `sub_provider`/`sub_providers` entries are normalized (missing required keys `type`/`max_context_size`/`model`/`url` are dropped, empty `role` defaults to `sub_agent`); if the root lacks `model`, a sub-provider is promoted to main provider (priority: no role → `sub_agent` → `planner`).
6. Applies the `ralph` override into `loop_control.max_ralph_iterations` of the provider dict.

### _create_config

Build a validated `kimi_agent_sdk.Config` from a provider dict (used internally by `create_session`).

```python
# File: src/kimix/utils/config.py
from kimix.utils import _create_config

cfg, provider_dict = _create_config(provider_dict)  # provider_dict=None → uses base._default_provider
```

- Asserts required provider keys: `type`, `max_context_size`, `model` (str), and `url` or `base_url`.
- Applies the provider's `env` dict entries into `os.environ`.
- Resolves `api_key` from the dict, else `KIMI_API_KEY`, else `KIMIX_API_KEY` env vars.
- Builds `LLMModel` / `LLMProvider` from declared fields (with special handling for `base_url`/`url`, `api_key` as `SecretStr`, `oauth` as `OAuthRef`, `openai_settings` as `OpenAISettings`).
- Warns about unrecognized provider keys.
- Applies `base._default_ralph` when `loop_control.max_ralph_iterations` is not set in the dict.

## Session Management (kimix.utils)

### create_session

Create a new or resume an existing Kimi session.

```python
# File: src/kimix/utils/session.py
from kimix.utils import create_session
from kaos.path import KaosPath
from kimix.utils.system_prompt import SystemPromptType

# Create a new session
session = create_session(
    session_id="my_session",           # Optional: unique session identifier
    work_dir=KaosPath("./workspace"),  # Optional: working directory
    skills_dir=None,                   # Optional: KaosPath to skills directory
    agent_file=None,                   # Optional: Path to custom agent_worker.json
    resume=False,                      # Optional: resume existing session
    provider_dict=None,                # Optional: custom LLM provider config dict
    chat_provider=None,                # Optional: custom ChatProvider instance
    agent_type=SystemPromptType.Worker, # Optional: Worker, TodoMaker, Thinker, etc.
    vfs_path=None,                     # Optional: Path for virtual file system
    extra_system_prompt=None,          # Optional: additional system prompt text
    max_ralph_iterations=None,         # Optional: max Ralph loop iterations
    anonymous=False,                   # Optional: anonymous session mode
    custom_data=None,                  # Optional: arbitrary dict stored on the session
)

# Close session when done
from kimix.utils import close_session, close_session_async
close_session(session)
# Or use async version
await close_session_async(session)
```

### create_session_async

Async version of `create_session` for use in async contexts.

```python
# File: src/kimix/utils/session.py
from kimix.utils.session import _create_session_async

session = await _create_session_async(
    session_id="my_session",
    resume=True,
    max_ralph_iterations=None,
    anonymous=False,
    custom_data={"foo": "bar"},  # Optional: arbitrary dict stored on the session
    # ... same parameters as create_session
)
```

### create_supervisor_session

Convenience wrapper to create a Supervisor session with `agent_boss.json`.

```python
# File: src/kimix/utils/session.py
from kimix.utils import create_supervisor_session

session = create_supervisor_session(
    session_id="supervisor_session",
    resume=True,
    # ... accepts most parameters of create_session, except:
    #     agent_file and agent_type are fixed internally to
    #     '<agent_dir>/agent_boss.json' and SystemPromptType.Supervisor
)
```

### Default Session

```python
# File: src/kimix/utils/session.py
from kimix.utils import get_default_session, _create_default_session

# Get or create the global default session (Worker role)
session = _create_default_session(resume=True)

# Get existing default session without creating
session = get_default_session()
```

### prompt / prompt_async

Send a prompt to the Kimi agent and get a response.

```python
# File: src/kimix/utils/prompt.py
from kimix.utils import prompt, prompt_async

# Simple prompt
prompt("What is the capital of France?")

# With options
prompt(
    "Analyze this code",
    session=session,                   # Optional: use specific session (None=default)
    output_function=custom_print,      # Optional: custom output handler for text chunks
    info_print=True,                   # Optional: print context usage after completion
    cancel_callable=None,              # Optional: callable that returns True to cancel
    close_session_after_prompt=False,  # Optional: close session after prompt completes
    merge_wire_messages=None,          # Optional: merge wire messages for output_function (defaults to True when output_function is set)
    ensure_todo_finished=True,         # Optional: run TodoList reminder rounds after the prompt
    export_todo_list_path=None,        # Optional: Path ending in .json — export the session's todos there after the prompt
    format_output=False,               # Optional: render text chunks as formatted markdown (used by prompt_plan)
    timeout=None,                      # Optional: max seconds for the whole prompt incl. retries; raises TimeoutError when reached
)

# Async version (coroutine) — same signature as prompt()
await prompt_async("Analyze this code", session=session)

# Get or create the global default session in async code
from kimix.utils import _create_default_session_async
session = await _create_default_session_async(resume=True)
```

**Automatic behaviors:**
- File paths in the prompt are automatically detected and wrapped in backticks via `escape_file_paths()`.
- Prompts longer than 65536 characters are automatically exported to a temp file and replaced with a `read and execute: <file>` instruction.
- When `output_function` is provided, `merge_wire_messages` defaults to `True`.
- **Retries:** non-API exceptions are retried up to 3 times with a 1s sleep (`APIStatusError` and timeouts are raised immediately). HTTP API errors are also retried with exponential backoff at the underlying chat-provider/soul layer.
- **Timeout:** when `timeout` is set, `asyncio.wait_for` guards each prompt attempt and `TimeoutError` propagates immediately (not retried); the session is cancelled.
- **Todo reminders (`ensure_todo_finished=True`):** after a successful prompt, unfinished `TodoList` items are collected and sent back as system reminders — up to `cli_closing_reminder_rounds` rounds (default 1; later rounds use a `strong=True` message). Todo items with attached `code` are auto-verified via `kimi_cli.tools.todo.verify.verify_code_todos`; failures are sent back as "Code check..." reminders.
- **Cleanup:** on finish the session's todos are cleared (`_clear_session_todos`, persisted to `SessionState` or the subagent `state.json`), optionally exported to `export_todo_list_path` (must end in `.json`), and the session is closed when `close_session_after_prompt=True`.

### Cancel Prompt

```python
# File: src/kimix/utils/session.py
from kimix.utils import cancel_prompt, get_cancel_event

# Cancel the current prompt on a session
cancel_prompt(session)  # session=None uses default session

# Get the cancel event for custom cancellation logic
event = get_cancel_event(session)
```

### Context Management

```python
# File: src/kimix/utils/session.py
from kimix.utils import (
    clear_default_context, compact_default_context,
    print_usage, delete_session_dir, context_path,
)
# NOTE: clear_context/compact_context and their async variants live in
# kimix.utils.session — they are NOT re-exported at the kimix.utils top level.
from kimix.utils.session import (
    clear_context, clear_context_async,
    compact_context, compact_context_async,
)

# Clear current context and start fresh
clear_default_context(force_create=True, resume=True, print_info=True)

# Compact context to reduce token usage
compact_default_context()

# Compact any session (sync/async) — only compacts when context usage > 1e-8
compact_context(session)
await compact_context_async(session)

# Clear any session (sync/async)
clear_context(session)
await clear_context_async(session)

# Print current context usage (session=None creates the default session)
print_usage(session)
print_usage()  # uses the default session

# Get the default session context storage path
path = context_path()  # Returns ~/.kimi/sessions

# Delete all session directories (~/.kimi/sessions)
delete_session_dir()
```

### Ralph Loop Control

```python
# File: src/kimix/utils/session.py
from kimix.utils import set_ralph_loop

# Set max Ralph iterations for a session (and default for future sessions)
set_ralph_loop(value=4, session=session)  # session=None uses default session
# Negative values are normalized to -1
```

### Tool Call Errors

```python
# File: src/kimix/utils/session.py
from kimix.utils import get_tool_call_errors

# Get failed tool calls for a session
errors = get_tool_call_errors(session)
# Returns list[dict[str, Any]] (currently returns empty list; stub for future use)
```

## System Prompt Types (kimix.utils.system_prompt)

```python
# File: src/kimix/utils/system_prompt.py
from kimix.utils.system_prompt import SystemPromptType, SystemPromptCallback, get_system_prompt

# Available agent types:
SystemPromptType.Worker           # Standard coding agent (terse, direct output)
SystemPromptType.TodoMaker        # Plan maker agent (creates implementation plans)
SystemPromptType.Thinker          # Thinker agent (thinks in <thinking> tags, self-verifies)
SystemPromptType.TrivialSubAgent  # Read-only sub-agent (rejects write/edit tasks)
SystemPromptType.Supervisor       # Supervisor agent (outlines, decomposes, dispatches, tracks, verifies)
SystemPromptType.Reader           # Read-only agent for retrieval/analysis tasks
SystemPromptType.SwarmLeader      # Swarm orchestrator (parallelizes a request via the AgentSwarm tool)

# Role-prompt rules per type (built into get_system_prompt):
# - TodoMaker: "Plan only. Do not implement." — records plans via WritePlan/EditPlan, cannot write files or run commands
# - Thinker:   thinks in <thinking>...</thinking>, ends with <quit/>, self-verifies
# - TrivialSubAgent: calls the `ask_parent` tool for clarification, then stops
# - Reader:    reports a concise summary only — no commands, edits, or questions
# - Supervisor: outline → decompose → dispatch → track → accept/inquire/correct → verify
# - SwarmLeader: splits a request into homogeneous sub-tasks and dispatches via `AgentSwarm` with
#                subagent_type (coder/explore/plan), a prompt_template containing {{item}}, and items

# Build a custom system prompt callback
class MyCallback(SystemPromptCallback):
    role_callback = lambda role, items: items.append("Custom rule here")

# Get the system prompt callable for Session.create
# Returns Callable[[BuiltinSystemPromptArgs], str]
system_prompts = get_system_prompt(
    yolo=True,
    work_dir=KaosPath("."),
    extra_system_prompt=MyCallback(),
    agent_role=SystemPromptType.Worker,
    max_system_prompt_tokens=4000,
)
```

## Colorful Printing (kimix.base)

### Basic Print Functions

```python
# File: src/kimix/base.py
from kimix.base import (
    print_success,    # Green bold - success messages
    print_error,      # Red bold - error messages
    print_warning,    # Yellow bold - warning messages
    print_info,       # Bright magenta - info messages
    print_debug,      # Bright cyan - debug messages (silent if _quiet=True)
    print_string,     # Plain text (respects _print_func)
)

# Usage
print_success("Operation completed successfully!")
print_error("File not found: config.yaml")
print_warning("This feature is deprecated.")
print_info("Processing step 3 of 5...")
print_debug("Variable x = 42")
print_string("Plain output")
```

### Advanced Color Printing

```python
# File: src/kimix/base.py
from kimix.base import colorful_print, colorful_text, Color, BgColor, Style

# Full control over colors and styles
colorful_print(
    "Important message!",
    fg=Color.BRIGHT_RED,
    bg=BgColor.YELLOW,
    styles=[Style.BOLD, Style.UNDERLINE]
)

# Get colored text without printing
colored = colorful_text("Warning", fg=Color.YELLOW, styles=[Style.BOLD])

# Available colors
# Foreground: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
#             BRIGHT_BLACK, BRIGHT_RED, BRIGHT_GREEN, BRIGHT_YELLOW
#             BRIGHT_BLUE, BRIGHT_MAGENTA, BRIGHT_CYAN, BRIGHT_WHITE
# Background: Same pattern with BgColor
# Styles: RESET, BOLD, DIM, ITALIC, UNDERLINE, BLINK, REVERSE, HIDDEN, STRIKETHROUGH
```

### Print Agent JSON

```python
# File: src/kimix/base.py
from kimix.base import print_agent_json

# Pretty-print streaming wire messages from the agent session (async — await it)
# Handles multiple message types intelligently:
# - ApprovalRequest: auto-resolves to "approve"
# - StepBegin, StepInterrupted, CompactionEnd: silently skipped
# - CompactionBegin: prints "Compacting..." info message
# - ThinkPart: prints thinking content in cyan (suppressed if _quiet)
# - TextPart: prints text chunks directly
# - ToolCall, ToolCallPart: prints "⚡ <name>" header (name resolved against the
#   session's live toolset — future-compatible, no hardcoded tool list), then
#   streams long argument values (keys in _STREAM_ARG_KEYS, e.g. WriteFile
#   `content`) decoded, token by token, as fragments arrive, each on its own
#   `key:\n` line; short arguments print inline on the header line as
#   ` key:value` segments. Every tool streams — there is no whitelist and no
#   legacy compact path; new tools need no per-tool display code.
# - ToolResult: prints ✓/✗ result line; calls output_function with MessageType.ToolResult
# - Type transitions: prints black context usage/token count using the provided session
await print_agent_json(
    wire_msg=message,
    session=session,
    output_function=custom_handler,  # Optional: callback(text, MessageType) for text/think/tool content
)

# Note: with merge_wire_messages=True a single complete ToolCall arrives, so the
# full decoded content prints in one go via the same stream printer.
# For a compact one-line `key:value` summary of raw arguments JSON (e.g. for
# the web frontend), use kimix.ui.stream.format_tool_args(args).
```

## Threading (kimix.base)

### Running Functions in Background

```python
# File: src/kimix/base.py
from kimix.base import run_thread, sync_all

# Run function in background thread (max 8 concurrent)
def my_task(data):
    # Long running operation
    process(data)

thread = run_thread(my_task, (data,))

# Wait for all threads to complete
sync_all()
```

### Async Prompt Helpers

```python
# File: src/kimix/utils/fix_error.py
from kimix.utils import async_prompt, async_fix_error

# Run prompt in background thread (creates new session if None, closes after)
thread = async_prompt("Analyze this file", session=None)

# Run fix_error in background thread
# merge_wire_messages is hardcoded to True internally
thread = async_fix_error(
    command="python main.py",
    extra_prompt="Handle edge cases",
    skip_success=True,
    keycode=("error",),
    max_loop=4,
    session=None,
    # NOTE: no merge_wire_messages parameter - always True internally
)
```

### Process Execution

```python
# File: src/kimix/base.py
from kimix.base import run_process_with_error, run_process_with_error_async

# Run command and capture output with error detection
error_output = run_process_with_error(
    command="npm run build",
    keycode=("error", "failed"),     # Required: keywords to look for in output (or None)
    skip_success=True                # Return None if no error keywords found and code==0
)

# Async version
error_output = await run_process_with_error_async(
    command="npm run build",
    keycode=("error", "failed"),
    skip_success=True
)
```

### Run Script in New Console

```python
# File: src/kimix/base.py
from kimix.base import run_script

# Launch a Python script in a new console window
proc = run_script("./my_script.py")
```

## File Operations (kimix.utils)

```python
# File: src/kimix/utils/prompt.py
from kimix.utils import prompt_path
from pathlib import Path

# Prompt with file content
prompt_path(Path("instructions.txt"))

# Prompt with split content and optional iterator callback
prompt_path(
    Path("tasks.txt"),
    split_word="---",
    session=session,
    after_prompt_coro=generator_func  # Optional: callable returning a generator/iterator; next() is called after each chunk
)
```

## Error Fix Loop (kimix.utils)

```python
# File: src/kimix/utils/fix_error.py
from kimix.utils import fix_error
from kimix.utils.fix_error import fix_error_async

# Automatically detect and fix errors from a command (sync)
success = fix_error(
    command="python main.py",
    extra_prompt="Make sure to handle edge cases",  # Optional: extra instructions
    keycode=("error", "exception"),                  # Optional: keywords to detect
    skip_success=True,                               # Optional: skip if return code is 0
    session=None,                                    # Optional: session to use
    max_loop=4,                                      # Optional: max fix attempts
    merge_wire_messages=False                        # Optional: merge wire messages
)
# Returns True if no error or fixed, False if max_loop reached

# Async version
success = await fix_error_async(
    command="python main.py",
    extra_prompt="Handle edge cases",
    keycode=("error",),
    session=None,
    max_loop=4,
    merge_wire_messages=False
)
```

## Plan Execution (kimix.utils)

### prompt_plan / prompt_plan_async

Generate an implementation plan with a dedicated `TodoMaker` planner session, review it interactively, then implement it with the default Worker session.

```python
# File: src/kimix/utils/prompt.py
from kimix.utils import prompt_plan, prompt_plan_async
from pathlib import Path

# Synchronous plan-and-implement flow
prompt_plan("Build a web application", plan_file="plan.md")

# Async version
await prompt_plan_async("Build a web application", plan_file=Path("plan.md"))
```

**Flow:**
1. Deletes any existing `plan_file` and enables plan mode (`note._enable_plan = True`).
2. Creates a planner session with `agent_type=SystemPromptType.TodoMaker` and `agent_file='agent_planner.json'`; it uses the `planner` sub-provider when configured (falls back to the main provider), disables budget/context-meter/compact/todo/target-churn reminders in `loop_control`, and is locked **read-only** so it cannot touch the filesystem.
3. Asks the planner to generate a comprehensive plan and save it via the `WritePlan` tool — up to 3 attempts, retrying when the plan file is not produced.
4. Opens the generated plan with the system default application.
5. Interactively asks whether to implement the plan; `y` proceeds, otherwise the user gives revision feedback (fed back to the planner via `WritePlan`/`EditPlan` and the plan is re-opened), or `/quit` aborts.
6. Closes the planner session and runs the implementation in the default Worker session (`ensure_todo_finished=False`, `format_output=True`). Large plans (>100 KiB) are referenced by file path instead of inlined.
7. Sends a follow-up review prompt to verify all tasks are completed.

**Parameters:**
- `requirement` — the task description to plan and implement.
- `plan_file` — path to the plan file (default `"plan.md"`).

## Configuration Variables (kimix.base)

Default configuration values you can import and modify:

```python
# File: src/kimix/base.py
from kimix.base import (
    _default_thinking,       # Deep thinking mode (default: True)
    _default_yolo,           # Yolo mode (default: True)
    _default_agent_file,     # Path to agent_worker.json
    _default_agent_file_dir, # Directory containing agent_worker.json
    _default_skill_dirs,     # List of skill directories
    _default_provider,       # Custom provider dict or None
    _default_sub_providers,  # List of role-tagged auxiliary provider dicts
    _default_ralph,          # Max Ralph iterations override or None
    _default_manually_cot,   # Manual chain-of-thought mode (default: False)
    _quiet,                  # If True, suppresses print_debug
    _colorful_print,         # If False, disables ANSI colors
    _print_func,             # Optional custom print handler (text, end) -> None
    COMMON_SKILL_DIRS,       # Default skill directory paths
)
```

### Configuration Setters

```python
# File: src/kimix/base.py
from kimix.base import (
    set_default_thinking,
    set_default_yolo,
    set_default_agent_file_dir,
    set_default_agent_file,
    set_default_skill_dirs,
    set_default_provider,
    set_default_sub_providers,
    get_default_sub_provider,
    set_default_manually_cot,
)

# Set default configuration values
set_default_thinking(True)
set_default_yolo(True)
set_default_agent_file_dir(Path("./custom_agents"))
set_default_agent_file(Path("./custom_agent.yaml"))
set_default_skill_dirs(["./skills", "./more_skills"])
set_default_provider({"name": "custom", "api_key": "..."})
set_default_sub_providers([{"name": "sub-custom", "api_key": "...", "role": "sub_agent"}])
planner_provider = get_default_sub_provider("planner")
set_default_manually_cot(False)
```

### Skill Directories

```python
# File: src/kimix/base.py
from kimix.base import get_skill_dirs

# Auto-discover skill directories (checked paths: .agents/skills, .config/.agents/skills, .opencode/skills, skills)
dirs = get_skill_dirs(use_kaos_path=True)
```

### Utility Functions

```python
# File: src/kimix/base.py
from kimix.base import percentage_str, percentage_and_token, generate_memory

# Format number as percentage string
s = percentage_str(0.7533)  # Returns "75.3%"

# Format context usage with percentage and token count
usage = percentage_and_token(session)  # Returns e.g. "42.5% (128000 tokens)"

# Standard prompt for generating session memory summaries
generate_memory  # String constant with structured memory generation instructions
```

### Path Utilities

```python
# File: src/kimix/utils/session.py
from kimix.utils import make_kaos_dir
from kaos.path import KaosPath

# Convert any path to KaosPath
kaos_path = make_kaos_dir("./my_folder")
```

## Prompt String Utilities (kimix.utils.prompt_str)

```python
# File: src/kimix/utils/prompt_str.py
from kimix.utils.prompt_str import escape_file_paths, clean_text

# Sanitize prompt text: detect file paths and wrap in backticks,
# strip invalid unicode surrogates/noncharacters/PUA, remove invisible chars,
# normalize Unicode (NFKC), convert full-width to half-width, remove emojis,
# collapse repeated punctuation, dedupe long character runs, and normalize whitespace.
# Paths already inside quotes/backticks/brackets, URLs, pure fractions and bare dates
# are left untouched; markdown fenced/inline code is preserved verbatim.
safe_text = escape_file_paths(
    raw_text,
    max_chars=0,           # Optional (keyword-only): truncate after N chars (0 = no limit)
    max_repeat=100,        # Optional (keyword-only): collapse repeated char runs longer than N
    truncate_msg="",       # Optional (keyword-only): suffix when truncating
    case_mode="",          # Optional (keyword-only): "lower" or "title" — applies to non-code text
)
# case_mode="lower"/"title" runs str.lower()/str.title() outside code blocks.
# When the optional `opencc` package is installed, Traditional → Simplified
# conversion is also applied during sanitization.

# Clean invisible/hidden characters from text
# Targets zero-width chars, control chars, soft hyphens, directional marks.
cleaned = clean_text(text, keep_newlines=True)  # keep_newlines=False strips \n\r\t too
```

## Windows Environment (kimix.utils.windows_env)

Refresh the current process environment from the Windows registry (useful after installing software or modifying system PATH).

```python
# File: src/kimix/utils/windows_env.py
from kimix.utils import refresh_env_from_registry

# Reload HKLM/HKCU environment variables into os.environ
refresh_env_from_registry()
```

- `refresh_env_from_registry()` — re-reads Windows registry environment values and updates `os.environ` in the current process.
- `_expand_registry_string(value)` and `_read_registry_value(...)` are internal helpers re-exported from `kimi_cli.utils.environment`.

## Complete Example

```python
# File: src/kimix/utils/session.py
"""Example script using Kimi API utilities."""
from pathlib import Path
from kimix.utils import create_session, prompt, close_session, clear_default_context
from kimix.base import print_success, print_error, print_info

# Create session
session = create_session(
    session_id="example",
)

try:
    # Prompt with custom message
    prompt(
        "Review this authentication code",
        session=session
    )
    
    print_success("Analysis complete!")
    
except Exception as e:
    print_error(f"Error: {e}")
    
finally:
    close_session(session)
```

## Best Practices

1. **Always close sessions** - Use `close_session()` when done to free resources
2. **Use colorful prints** - Makes output more readable and organized
3. **Handle errors** - Wrap prompts in try/except blocks
4. **Background tasks** - Use `run_thread()` for long-running operations
5. **Session reuse** - Reuse sessions for related prompts to save context
6. **Clear context** - Call `clear_default_context()` when switching topics
7. **Compact context** - Use `compact_default_context()` to reduce token usage
8. **Fix errors automatically** - Use `fix_error()` for iterative debugging
9. **Skill directories** - Place skills in `.agents/skills/` for auto-discovery
10. **Cancel long prompts** - Use `cancel_prompt()` to stop running prompts
11. **Ralph loop** - Use `set_ralph_loop()` to control max agent iterations
12. **Sanitize prompts** - Use `escape_file_paths()` before sending untrusted text

## Common Imports

```python
# File: src/kimix/utils/__init__.py
# Core utilities (kimix.utils.__all__)
from kimix.utils import (
    init,
    create_session, close_session, close_session_async,
    create_supervisor_session,
    prompt, prompt_async, clear_default_context, compact_default_context, print_usage,
    get_default_session, _create_default_session, _create_default_session_async, _create_session_async,
    get_tool_call_errors,
    cancel_prompt, get_cancel_event,
    prompt_path, prompt_plan, prompt_plan_async,
    fix_error, async_prompt, async_fix_error,
    context_path, delete_session_dir, make_kaos_dir,
    set_ralph_loop,
    refresh_env_from_registry,
    # Internal/advanced
    _create_config, _ensure_skill_dirs, _print_usage,
    _default_session, _should_print_usage,
    _cli_sessions, _add_cli_session, _remove_cli_session, _refresh_cli_sessions,
    _SYSTEM_PROMP, get_system_prompt,
)
# NOTE: clear_context / compact_context / clear_context_async / compact_context_async
# are defined in kimix.utils.session but NOT re-exported from kimix.utils — import
# them from kimix.utils.session directly.
# _cli_sessions / _add_cli_session / _remove_cli_session / _refresh_cli_sessions are
# session-state internals originating in kimix.utils._globals (re-exported via __all__).
from kimix.utils.system_prompt import SystemPromptType, SystemPromptCallback
from kimix.utils.fix_error import fix_error_async  # Not in top-level __all__
from kimix.utils.prompt_str import escape_file_paths, clean_text  # Not in top-level __all__

from kimix.base import (
    print_success, print_error, print_warning,
    print_info, print_debug, print_string, colorful_print, colorful_text,
    Color, BgColor, Style, run_thread, sync_all,
    run_process_with_error, run_process_with_error_async,
    run_script, print_agent_json,
    get_skill_dirs, percentage_str, percentage_and_token,
    COMMON_SKILL_DIRS,
    set_default_thinking, set_default_yolo,
    set_default_agent_file_dir, set_default_agent_file,
    set_default_skill_dirs, set_default_provider,
    set_default_sub_providers, get_default_sub_provider,
    set_default_manually_cot,
)

# Standard library
from pathlib import Path
import asyncio
```

## `kimi_agent_sdk` — Kimi Agent SDK

This is a Python SDK for building AI agents powered by Kimi.

### High-level API

```python
# File: src/kimi_agent_sdk/_prompt.py
import asyncio
import kimi_agent_sdk

async def main():
    async for msg in kimi_agent_sdk.prompt(
        "Write a Python script that greets the user",
        work_dir="./workspace",
        thinking=True,
        yolo=True,
    ):
        print(msg)

asyncio.run(main())
```

- `kimi_agent_sdk.prompt(user_input, *, work_dir=None, config=None, model=None, thinking=False, yolo=False, approval_handler_fn=None, agent_file=None, mcp_configs=None, skills_dir=None, skills_dirs=None, max_steps_per_turn=None, max_retries_per_step=None, max_ralph_iterations=None, final_message_only=False)` — async generator yielding `Message` objects.
- `ApprovalHandlerFn` — type alias for sync/async callback `(ApprovalRequest) -> None`.

### Low-level API

```python
# File: src/kimi_agent_sdk/_session.py
import asyncio
import kimi_agent_sdk

async def main():
    async with kimi_agent_sdk.Session.create(
        work_dir="./workspace",
        thinking=True,
        yolo=False,
    ) as session:
        async for wire in session.prompt("Hello, agent!"):
            print(wire)
        
        print("Session ID:", session.id)
        print("Model:", session.model_name)
        print("Status:", session.status)
        
        # Export context to file
        path, token_count = session.export()
        
        # Compact to reduce token usage
        await session.compact("Keep the key decisions")
        
        # Rename session
        await session.rename("my-renamed-session")
        
        # Cancel ongoing prompt
        await session.cancel()
        
        # Clear context and start fresh
        await session.clear()

asyncio.run(main())
```

- `kimi_agent_sdk.Session` — async context manager with methods:
  - `Session.create(work_dir=None, *, session_id=None, config=None, model=None, thinking=False, yolo=False, plan_mode=False, agent_file=None, mcp_configs=None, skills_dir=None, skills_dirs=None, anonymous=False, max_steps_per_turn=None, max_retries_per_step=None, max_ralph_iterations=None, **custom_arguments)`
  - `Session.resume(work_dir, session_id=None, *, ...)`
  - `session.prompt(user_input, *, merge_wire_messages=False)` — async generator yielding `WireMessage`
  - `session.cancel()`
  - `session.close()`
  - `session.clear(**custom_arguments)`
  - `session.rename(new_session_id)`
  - `session.compact(custom_instruction="")`
  - `session.export(output_path=None) -> tuple[Path, int]`
  - `session.id` (property)
  - `session.model_name` (property)
  - `session.status` (property, returns `StatusSnapshot`)
  - `session.get_custom_data()`
  - `session.get_custom_config()`

### Re-exported types from `kimi_agent_sdk.__all__`

- **Core:** `prompt`, `Session`, `ExportedContext`
- **Approval:** `ApprovalHandlerFn`, `ApprovalRequest`
- **Message types:** `Message`, `ContentPart`, `TextPart`, `ThinkPart`, `ImageURLPart`, `AudioURLPart`, `VideoURLPart`, `ToolCall`
- **Wire types:** `WireMessage`, `Event`, `Request`, `TurnBegin`, `TurnEnd`, `StepBegin`, `StepInterrupted`, `CompactionBegin`, `CompactionEnd`, `StatusUpdate`, `ToolCallPart`, `ToolResult`, `ToolReturnValue`, `ApprovalResponse`, `SubagentEvent`, `DisplayBlock`, `BriefDisplayBlock`, `DiffDisplayBlock`, `ShellDisplayBlock`, `TodoDisplayBlock`, `TodoDisplayItem`, `TokenUsage`, `is_event`, `is_request`
- **Tooling:** `CallableTool2`, `ToolOk`, `ToolError`
- **Exceptions:** `KimiAgentException` (alias for `KimiCLIException`), `ConfigError`, `AgentSpecError`, `InvalidToolError`, `MCPConfigError`, `MCPRuntimeError`, `SystemPromptTemplateError`, `LLMNotSet`, `LLMNotSupported`, `ChatProviderError`, `APIConnectionError`, `APITimeoutError`, `APIStatusError`, `APIEmptyResponseError`, `MaxStepsReached`, `RunCancelled`, `PromptValidationError`, `SessionStateError`
- **Config:** `Config`, `MCPConfig`

### Internal Aggregator

- `MessageAggregator(final_message_only=False)` — aggregates `WireMessage` stream into `Message` stream.
  - `feed(msg) -> list[Message]`
  - `flush() -> list[Message]`

## `kimix.dag` — DAG Task Dependency Execution

Package exports: `Context`, `DAG`, `TaskNode`, `Executor`, `TopologicalSorter`, `CycleError`, `DAGValidationError`, `DependencyError`, `ExecutionError`, `detect_cycle`, `validate_dag`

```python
# File: src/kimix/dag/dag.py
from kimix.dag import DAG, TaskNode, Executor, Context

# Build a DAG
dag = DAG()
dag.add_node(TaskNode("A", lambda ctx: "result A"))
dag.add_node(TaskNode("B", lambda ctx: f"result B using {ctx.get('A')}", dependencies=["A"]))
dag.add_node(TaskNode("C", lambda ctx: "result C", dependencies=["A"]))
dag.add_node(TaskNode("D", lambda ctx: "result D", dependencies=["B", "C"]))

# Execute with a context
ctx = Context()
executor = Executor(max_workers=4)
results = executor.execute(dag, ctx=ctx)
print(results)  # {"A": "result A", "B": "result B using result A", ...}
```

- `DAG` — `add_node(node)`, `add_edge(upstream, downstream)`, `get_node(name)`, `validate()`, `nodes`, `edges`
- `TaskNode(name, func, params=None, dependencies=None, retries=0)` — `execute(ctx)`, `mark_done(result, error)`, `done`, `name`, `func`, `params`, `dependencies`, `retries`, `result`, `error`
- `Context` — `cancel()`, `cancelled`, `check_cancelled()`, `get(key, default)`, `set(key, value)`, `update(other_dict)`
- `Executor(max_workers=None)` — `execute(dag, ctx=None) -> dict[str, Any]`
- `TopologicalSorter(edges)` — `sort() -> list[str]`
- Exceptions: `DAGValidationError(ValueError)`, `CycleError(DAGValidationError)`, `DependencyError(RuntimeError)`, `ExecutionError(RuntimeError)` with `.errors: dict[str, Exception]`
- `detect_cycle(graph) -> list[str] | None`
- `validate_dag(nodes, edges)`

## `kimix.network` — TCP / JSON-RPC Networking

### TCP Client / Server

```python
# File: src/kimix/network/tcp_client.py, src/kimix/network/tcp_server.py, src/kimix/network/tcp_group_server.py
from kimix.network.tcp_client import TCPClient
from kimix.network.tcp_server import TCPServer
from kimix.network.tcp_group_server import TcpGroupServer

client = TCPClient(host="127.0.0.1", port=8888)
await client.connect()
await client.send({"type": "hello"})
await client.disconnect()

server = TCPServer(host="127.0.0.1", port=8888)
server.on_message(lambda msg: print("Received:", msg))
await server.start(blocking=False)
# ...
await server.stop()
```

- `TCPClient(host="127.0.0.1", port=8888)` — `connect(blocking=False)`, `disconnect()`, `send(message)`, `send_bytes(data)`, `is_connected()`, `on_connect(callback)`, `on_disconnect(callback)`, `on_message(callback)`
- `TCPServer(host="127.0.0.1", port=8888)` — `start(blocking=False)`, `stop()`, `send(message)`, `send_bytes(data)`, `disconnect_client()`, `is_client_connected()`, `on_connect(callback)`, `on_disconnect(callback)`, `on_message(callback)`, `on_raw_data(callback)`
- `TcpGroupServer(host="127.0.0.1", port=8888, max_workers=10)` — `start(blocking=False)`, `stop()`, `send(client_id, message)`, `send_bytes(client_id, data)`, `broadcast(message)`, `disconnect_client(client_id)`, `get_client_ids()`, `get_client_count()`, `wait_for_clients(count, timeout=5.0)`, `on_client_connect(callback)`, `on_client_disconnect(callback)`, `on_message(callback)`, `on_raw_data(callback)`

### JSON-RPC

- `JSONRPCClient(host=DEFAULT_HOST, port=DEFAULT_PORT)` — `connect()`, `disconnect()`, `is_connected()`, `call(method, *args, timeout=5.0)` (from `kimix.network.rpc_client`)
- `JSONRPCServer(host=DEFAULT_HOST, port=DEFAULT_PORT, max_workers=10, ...)` — `register(name, func)`, `register_function(func)`, `start(blocking=True)`, `stop()`, `get_client_count()`, `wait_for_connection(timeout=5.0)`, `disconnect_client(client_id)`, `start_websocket_server(ws_port, blocking=False)`, `stop_websocket_server()` (from `kimix.network.rpc_server`)

## `kimix.server` — Opencode-Style HTTP Server (FastAPI + SSE)

```python
# File: src/kimix/server/app.py
from kimix.server.app import create_app
from kimix.server.client import KimixAsyncClient

# Create and run the FastAPI app
app = create_app()

# Or use the async client to interact with a running server
client = KimixAsyncClient(host="127.0.0.1", port=4096)

async def interact():
    session = await client.create_session(title="My Session")
    sid = session["id"]
    
    async for event in client.stream_events_robust(sid):
        print(event)
    
    await client.send_prompt_async(sid, "Hello, Kimi!")
    await client.abort_session(sid)
    await client.delete_session(sid)
```

These classes are defined in submodules under `kimix.server` (the package `__init__.py` does not re-export them):

- `create_app()` — returns FastAPI with routes: health, SSE event stream, session CRUD, prompt, abort, permissions, clear, context, compact, export (from `kimix.server.app`)
- `BusEvent(type, properties)` — `to_dict()`, `to_json()` (from `kimix.server.bus`)
- `EventBus` — `subscribe(callback)`, `create_async_queue()`, `remove_async_queue(q)`, `emit(event)`, `emit_type(event_type, **properties)`; global `bus = EventBus()` (from `kimix.server.bus`)
- `KimixAsyncClient(host="127.0.0.1", port=4096, timeout=30.0)` — `health_check()`, `create_session(title=None)`, `get_session(id)`, `list_sessions()`, `delete_session(id)`, `get_messages(id, limit=10)`, `send_prompt_async(id, text, agent=None)`, `abort_session(id)`, `clear_session(id)`, `compact_session(id, keep=None)`, `export_session(id, output_path=None)`, `stream_events(session_id, timeout=14400.0)`, `stream_events_robust(session_id, timeout=14400.0, max_reconnects=5, reconnect_delay=2.0, on_reconnect=None)` (from `kimix.server.client`)
- `SessionManager` — `create_session(title=None)`, `get_session(id)`, `get_sdk_session(id)`, `list_sessions()`, `delete_session(id)`, `get_session_status()`, `get_messages(id, limit=None)`, `prompt(id, text, agent=None)`, `prompt_async(...)`, `abort_session(id)`, `clear_session(id)`, `compact_session(id, keep=None)`, `export_session(id, output_path=None)`, `get_session_context(id, keep=None)`; global `session_manager = SessionManager()` (from `kimix.server.session_manager`)
- `serve_cli(args)` — CLI entry point for `kimix serve` (from `kimix.server.serve`)

## `kimix.parser` — Source Code Comment Parsers

Package exports: `Comment`, `ParseResult`, `BaseParser`, `PythonParser`, `CParser`, `ShellParser`, `HtmlParser`, `PascalParser`, `LispParser`, `SqlParser`

```python
# File: src/kimix/parser/__init__.py
from kimix.parser import PythonParser, ParseResult

parser = PythonParser()
result: ParseResult = parser.parse(source_code)
print(f"Found {result.total_comments} comments")
for c in result.comments:
    print(f"Line {c.line}: {c.content}")
```

- `Comment(content, line, column, kind)`
- `ParseResult(language, comments, code_without_comments)` — `total_comments`, `comment_lines`, `get_comments_by_kind(kind)`
- `BaseParser` — abstract `parse(source_code) -> ParseResult`; concrete `parse_file(file_path, encoding="utf-8")`
- Language parsers: `PythonParser`, `CParser`, `ShellParser`, `HtmlParser`, `PascalParser`, `LispParser`, `SqlParser`

## `kimix.tools` — Built-in Agent Tools

All tools are `CallableTool2` subclasses. They are organized in subpackages under `kimix.tools` (the package `__init__.py` does not re-export them); import from the relevant submodule, e.g. `from kimix.tools.agent import Agent, AgentList, AgentClose, AskAgent`. Key ones:

- `Agent` — launch sub-agent; params: `prompt`, `session_id`, `close_session=True`, `return_history=False`, `response` (from `kimix.tools.agent`)
- `AskAgent` — send a message to another agent via the steer mechanism; params: `question`, `id` (optional for the main agent, ignored for sub-agents which always message their parent) (from `kimix.tools.agent`)
- `AgentList` — list active sub-agent sessions (from `kimix.tools.agent`)
- `AgentClose` — close sub-agent session; params: `session_id` (from `kimix.tools.agent`)
- `TaskOutput` — get background task output; params: `task_id`, `block=True`, `timeout=60`, `output_path`, `kill=False`
- `BackgroundStream` — `start(function, stop_function, input_function=None)`, `wait(timeout=None)`, `stop()`, `get_output()`, `pop_output()`, `input(data)`, `success()`
- `Bash` / `Powershell` — shell execution; params: `cmd`, `timeout=10` (from `kimix.tools.file.bash.bash_tool`)
- `Run` — run external executable; params: `command`, `timeout=10`, `output_path`, `cwd`, `env`, `run_in_background=False`
- `FindStr` — search text in files; params: `content`, `path`, `case_sensitive=False`
- `Mkdir` / `Rm` — create/remove directories
- `Python` — execute Python code; params: `code`, `output_path`, `timeout=10`, `run_in_background=False` (from `kimix.tools.py`)
- `PySyntaxCheck` — check Python syntax with ruff; params: `file_path`
- `SyntaxLint` — unified syntax lint dispatcher; params: `file_path`, `project_root=".", clangd_path="clangd", verbose=False`
- `MypyCheck` — Python type check; params: `file_path`, `project_root=".", verbose=False`
- `Cpplint` — C++ lint via clangd; params: `file_path`, `project_root=".", clangd_path="clangd", verbose=False`
- `JsTsSyntaxCheck` — JS/TS syntax check via tree-sitter; params: `file_path`, `verbose=False`
- `Ocr` — OCR from image; params: `image_path`, `output_path`, `language="eng"`, `preprocess=False`
- `Docx2md` — convert DOCX to Markdown; params: `docx_path`, `output_path`
- `Pdf2md` — convert PDF to Markdown; params: `pdf_path`, `output_path`, `extract_images=False`, `ocr=False`, `extract_tables=True`, `page_range`
- `ParserTool` — parse/extract/strip comments; params: `language`, `source_code|file_path`, `mode="extract"`, `encoding="utf-8"`
- `WritePlan` / `ReadPlan` / `EditPlan` — plan file tools
- `StoreSession` / `LoadSession` / `LsSession` — key-value session persistence
- `FetchURL` — fetch web page as Markdown; params: `url`, `output_path`
- `fetch_to_markdown(url, wait_until="networkidle")` — Playwright-based fetcher
- `Zip` / `Unzip` — 7z archive tools; params: `source`, `destination`, `password`

## `kimix.cot` — Chain-of-Thought

```python
# File: src/kimix/cot.py
from kimix.cot import cot_prompt, cot_prompt_async, CoTResult

# Synchronous CoT with self-verification
result: CoTResult = cot_prompt(
    "Explain quantum computing in simple terms",
    self_verify=True,
    max_iterations=10,
)
print(result.thinking)
print("Quit:", result.quit)

# Async version
result = await cot_prompt_async("Explain quantum computing", self_verify=True)

# Two-pass verification (sync)
result = cot_prompt_with_verification("Complex reasoning task")

# Two-pass verification (async)
result = await cot_prompt_with_verification_async("Complex reasoning task")
```

- `CoTResult(thinking: str, quit: bool = False)`
- `cot_prompt(prompt_str, self_verify=True, existing_thinking=None, max_iterations=10)` — sync
- `cot_prompt_async(prompt_str, self_verify=True, existing_thinking=None, max_iterations=10)` — async
- `cot_prompt_with_verification(prompt_str, existing_thinking=None)` — sync two-pass
- `cot_prompt_with_verification_async(prompt_str, existing_thinking=None)` — async two-pass

## `kimix.retrieval` — BM25 Retrieval Engine

- `NgramTokenizer(n=2)` — `normalize(text)`, `tokenize(text, n=None)`
- `InvertedIndex` — `add_document(doc_id, tokens)`, `finalize(stop_threshold=0.5, prune_df=None)`, `get_postings(term)`, `doc_freq(term)`, `has_term(term)`, `terms()`, `save(path)`, `load(path)`, `N`, `avgdl`, `doc_lengths`, `doc_lengths_arr`
- `BM25Scorer(index, k1=1.2, b=0.75)` — `score(query_tokens, candidate_docs=None)`, `score_topk(query_tokens, top_k, candidate_docs=None)`
- `LevenshteinAutomaton(pattern, max_edits, prefix_length=1)` — `auto_fuzziness(term)`, `match(dictionary, max_expansions=50)`
- `Searcher(index, tokenizer=None, scorer=None, k1=1.2, b=0.75, min_should_match=0.5, fuzziness="AUTO", max_expansions=50, prefix_length=1)` — `search(query, top_k=10)`
- `SimHash(text="", hashbits=64)` — `distance(other)`, `is_near_duplicate(other, threshold=3)`
- `SimHashLSH(hashbits=64, band_bits=4)` — `add(doc_id, simhash)`, `candidates(simhash)`, `remove(doc_id)`
- `RM3Expander(index, scorer, fb_docs=3, fb_terms=10, alpha=0.5)` — `expand(query_tokens, top_k=10)`
- `RocchioExpander(index, scorer, alpha=1.0, beta=0.75, gamma=0.15, fb_docs=3, fb_terms=10)` — `expand(query_tokens, non_rel_docs=None)`
- `LambdaMART(n_iterations=50, learning_rate=0.05)` / `CoordinateAscent` — `fit(X, y)`, `predict(X)`, `rank(doc_features)`
- `QueryPerformancePredictor(index, scorer)` — `avg_idf(query_tokens)`, `max_idf(query_tokens)`, `query_scope(query_tokens)`, `is_hard_query(query_tokens, avg_idf_threshold=2.0)`
- `RankSVM(learning_rate=0.01, n_iterations=1000, margin=1.0)` — `fit(X, y)`, `predict(X)`, `rank(doc_features)`
- `RankBoost(n_iterations=100)` — `fit(X, y)`, `predict(X)`, `rank(doc_features)`
- `MinHash(text="", num_perm=128, k=3)` — `jaccard(other)`
- `NoisyChannelSpeller(dictionary, max_edits=2)` — `correct(word)`
- Utility functions: `jaro_similarity`, `jaro_winkler_similarity`, `sorensen_dice_coefficient`, `ngram_overlap`, `jaccard_similarity_tokens`, `hamming_distance`, `cosine_similarity_tfidf`, `soundex`, `metaphone`, `porter_stem`, `clarity_score`, `scq`, `mmr_rerank`, `xquad_rerank`, `i_match_fingerprint`

## `kimix.summarize` — Context Compaction

- `summarize(temp_file=None, session=None, only_return_remember_str=False) -> str | None`
- `summarize_mistake(result_file, session=None)`

## `kimix.cli` — CLI Entry Point

- `cli()` — launches the interactive Kimix CLI

## Dynamic System Reminders

The agent runtime injects short system reminders into the LLM context before a step. Reminders are produced by pluggable `DynamicInjectionProvider`s registered on `KimiSoul`.

### Core data structures

```python
# File: kimi_cli/soul/dynamic_injection.py
from dataclasses import dataclass
from kosong.message import Message

@dataclass(frozen=True, slots=True)
class DynamicInjection:
    type: str      # identifier, e.g. "compact_reminder"
    content: str   # plain text; will be wrapped in <system-reminder> tags

class DynamicInjectionProvider(ABC):
    @abstractmethod
    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]: ...

    async def on_context_compacted(self) -> None:
        """Override to reset throttling after context compaction."""

    async def on_afk_changed(self, enabled: bool) -> None:
        """Override to reset throttling when afk mode toggles."""
```

Providers decide for themselves whether to inject, usually throttling by step number, message index, or internal state.

### Wrapping and delivery

```python
# File: kimi_cli/soul/message.py
def system(message: str) -> ContentPart:
    return TextPart(text=f"<system>{message}</system>")

def system_reminder(message: str) -> TextPart:
    return TextPart(text=f"<system-reminder>\n{message}\n</system-reminder>")
```

Before each LLM step, `KimiSoul._collect_injections()` awaits every provider, concatenates the resulting `content`s, wraps each with `system_reminder()`, and appends the combined text as a user-role message. Adjacent user messages (including these reminders) are later merged by `normalize_history()` for the API call.

### Registration in `KimiSoul`

```python
# File: kimi_cli/src/kimi_cli/soul/kimisoul.py
class KimiSoul:
    def __init__(self, agent: Agent, *, context: Context, anonymous: bool = False):
        ...
        self._injection_providers: list[DynamicInjectionProvider] = [
            CompactReminderProvider(threshold=loop_control.compact_reminder_threshold)
                if loop_control.compact_reminder_enabled else [],
        ]

    def add_injection_provider(self, provider: DynamicInjectionProvider) -> None:
        self._injection_providers.append(provider)

    async def _collect_injections(self) -> list[DynamicInjection]:
        ...

    async def _notify_injection_providers_compacted(self) -> None:
        ...

    async def notify_afk_changed(self, enabled: bool) -> None:
        ...
```

Built-in providers are registered in `__init__` according to `Config` / `LoopControl` flags. External code can add more via `add_injection_provider()`.

### Built-in providers

| Provider | File | Type | Purpose | Config knobs |
|----------|------|------|---------|--------------|
| `CompactReminderProvider` | `dynamic_injections/compact_reminder.py` | `compact_reminder` | Suggests calling `Compact` when context usage exceeds a threshold. | `LoopControl.compact_reminder_enabled`, `compact_reminder_threshold` |

Both providers (and any custom providers) skip subagent sessions and reset their throttling state in `on_context_compacted()` and/or `on_afk_changed()`.

### Implementing a custom provider

```python
from kosong.message import Message
from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider

class MyReminderProvider(DynamicInjectionProvider):
    def __init__(self) -> None:
        self._fired = False

    async def get_injections(self, history: Sequence[Message], soul: KimiSoul) -> list[DynamicInjection]:
        if self._fired or soul.is_subagent:
            return []
        if not history or history[-1].role != "assistant":
            return []
        text = history[-1].extract_text(" ")
        if "my trigger" not in text.lower():
            return []
        self._fired = True
        return [DynamicInjection(type="my_reminder", content="Remember to do X.")]

    async def on_context_compacted(self) -> None:
        self._fired = False
```

Register it on an existing soul:

```python
soul.add_injection_provider(MyReminderProvider())
```

## Context Pruning (Smart History Removal)

The agent runtime includes a **context pruner** that dynamically frees context space
by removing historical information the LLM no longer needs, without harshly breaking
the LLM backend's prefix KV cache.

### Architecture

- **Module:** `kimi_cli/soul/context_pruning.py` — `ContextPruner`, `PruningResult`, `ElidedRecord`
- **Integration point:** `_step()` in `kimisoul.py`, between dynamic injection and history normalization
- **Layer 1 (default, request-time only):** prunes a *copy* of history for the LLM call;
  storage, checkpoints, and notification ack are unaffected
- **Layer 2 (opt-in, `prune_persist=True`):** persists removals to storage

### Two Tiers

**Tier A — Ephemeral injected messages (primary, safest)**
Drops consumed/superseded accumulating ephemera outright (no stub, no retrieval ref):
- Superseded active-task snapshots (`<active-background-tasks>`) — only the most recent kept
- Consumed notifications (`<notification ...>`) — older than the recency window
- Spent D-Mail notices — older than the turn they applied to
- CHECKPOINT markers — config-gated (default off)
- System reminders — already handled by `strip_system_reminders`

**Tier B — Stale/oversized substantive content (escalation only)**
Elides content (keeps `role`/`tool_call_id`, replaces body with a compact stub):
- Superseded tool reads (e.g. read followed by write/edit)
- Oversized tool outputs (> `prune_tool_output_min_tokens`)
- Resolved errors (error followed by same-tool success)
- Old reasoning (`ThinkPart`) — gated by `prune_elide_thinking`
- Near-duplicate large blobs — gated by `prune_dedupe_near_duplicates`

### Elision Stub Format

```
<system>[context-elided: {kind} — {short_summary}. ~{tokens} tokens freed.
Retrieve full content with Retrieve id={ref})]</system>
```

### Cache-Conservative Policy

1. **Protect the recent tail** — never prune the last `K` turns and their tool messages
2. **Protect a stable head** — first `P` messages never removed
3. **Prune only the middle band**, tail-inward (largest index first → shallowest recompute)
4. **Rare + batched** — cooldown (steps + usage growth) between passes
5. **Min-payoff gate** — skip if savings < `prune_min_free_tokens`
6. **Deterministic + idempotent** — same input → same output; already-pruned regions untouched
7. **Prefer Tier A over Tier B** — cheapest, safest space first
8. **Piggyback the existing break** — runs at `strip_system_reminders` point (one cache-break event)

### Configuration (LoopControl)

| Field | Default | Description |
|-------|---------|-------------|
| `context_pruning_enabled` | `True` | Master switch |
| `prune_trigger_ratio` | `0.70` | Usage ratio to trigger a prune pass |
| `prune_target_ratio` | `0.55` | Target usage after pruning |
| `prune_stable_prefix_messages` | `4` | Head messages kept as cached prefix |
| `prune_recent_messages_protected` | `6` | Recent turns protected from pruning |
| `prune_min_free_tokens` | `2000` | Minimum payoff for a prune pass |
| `prune_cooldown_steps` | `4` | Hysteresis between passes |
| `prune_min_usage_growth` | `0.05` | Growth required before re-pruning |
| `prune_max_fraction_per_pass` | `0.5` | Max fraction of tokens pruned per pass |
| `prune_ephemeral_enabled` | `True` | Enable Tier A ephemeral removal |
| `prune_ephemeral_notifications` | `True` | Drop consumed notifications |
| `prune_ephemeral_task_snapshots` | `True` | Keep only latest task snapshot |
| `prune_ephemeral_dmail_notices` | `True` | Drop spent D-Mail notices |
| `prune_ephemeral_checkpoint_markers` | `False` | Drop CHECKPOINT markers |
| `prune_substantive_enabled` | `True` | Enable Tier B substantive elision |
| `prune_tool_output_min_tokens` | `512` | Min token count for oversized output |
| `prune_elide_thinking` | `True` | Elide old ThinkPart content |
| `prune_dedupe_near_duplicates` | `True` | Elide near-duplicate large blobs |
| `prune_persist` | `False` | Persist removals to storage (Layer 2) |
| `prune_subagents` | `True` | Apply pruning to subagent sessions |

Defaults enforce: `prune_target_ratio < prune_trigger_ratio < compaction_trigger_ratio`.

### Retrieval of Elided Content

Tier B elided content stays reachable via `Retrieve`:
- Tool results are now indexed in `HistoryIndex` (previously only user/assistant turns)
- `HistoryIndex.get_by_id(ref)` resolves the stub's reference deterministically
- `Retrieve` accepts an optional `id` parameter for direct retrieval
- Auto-retrieval (`_maybe_auto_retrieve_history`) resurfaces relevant elided turns automatically

### Slash Command

- `/prune` — manually trigger a prune pass (analogous to `/compact`)

### Observability

Each prune pass logs:
- `freed_tokens` — estimated tokens reclaimed
- `earliest_removed_index` — cache-break depth (index of earliest change)
- `tier_b` count — number of elided records indexed

## Complete Package Index

| Package / Module | Description |
|------------------|-------------|
| `kimi_agent_sdk` | Python SDK for building AI agents powered by Kimi (Session, prompt, wire/message types, config, exceptions) |
| `kimix.base` | Core base utilities: colorful printing, threading helpers, process execution, configuration variables |
| `kimix.cli` | Interactive CLI entry point (`cli()`) |
| `kimix.cot` | Chain-of-thought reasoning utilities (`cot_prompt`, `CoTResult`) |
| `kimix.dag` | DAG task dependency execution engine (`DAG`, `TaskNode`, `Executor`, `Context`) |
| `kimix.network` | TCP / JSON-RPC networking layer (`TCPClient`, `TCPServer`, `JSONRPCClient`, `JSONRPCServer`) |
| `kimix.parser` | Source code comment parsers for multiple languages (`PythonParser`, `CParser`, etc.) |
| `kimix.retrieval` | BM25 retrieval engine, fuzzy search, ranking, and query performance prediction |
| `kimix.server` | Opencode-style HTTP server with FastAPI + SSE (`create_app`, `KimixAsyncClient`, `SessionManager`) |
| `kimix.summarize` | Context compaction / summarization helpers |
| `kimix.tools` | Built-in agent tools: shell, Python, file ops, OCR, PDF/DOCX conversion, linting, planning |
| `kimix.utils` | High-level session management, prompting, plan execution, error fixing, initialization, and prompt string utilities |
| `kimix.utils._globals` | Module-level session state: `_default_session`, `_should_print_usage`, and the `_cli_sessions` cache with `_add/_remove/_refresh` accessors |
| `kimix.utils.config` | Kimix global initialization (`init`) and `Config` construction from provider dicts (`_create_config`) |
| `kimix.utils.fix_error` | Iterative error detection and auto-fix loop (`fix_error`, `fix_error_async`, `async_prompt`, `async_fix_error`) |
| `kimix.utils.prompt` | Prompt helpers, plan generation/implementation (`prompt_plan`, `prompt_path`, `prompt`, `prompt_async`) |
| `kimix.utils.prompt_str` | Prompt sanitization: escape file paths, clean invisible characters |
| `kimix.utils.session` | Session creation, resumption, context management, and lifecycle (incl. `clear_context`/`compact_context` — not top-level re-exports) |
| `kimix.utils.system_prompt` | System prompt types and builders for different agent roles |
| `kimix.utils.windows_env` | Windows registry environment refresh helpers |
