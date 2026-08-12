# Kimix Quick Start

This guide covers environment setup, installation, and basic CLI usage.

---

## Quick Install

```bash
pip install kimix
python -m kimix.cli
# or
python -m kimix
```

For source-based development, see the detailed steps below.

---

## Git Submodules

Kimix uses Git submodules for some dependencies.

**Clone with submodules:**
```bash
git clone --recursive <repo-url>
```

**Update existing clone:**
```bash
uv run clone_submodule.py
# or manually:
git submodule update --init --recursive
```

---

## Install with uv

Recommended: use [uv](https://docs.astral.sh/uv/) for Python environment management.

```bash
cd /path/to/kimix
uv tool install -e .
uv run kimix
```

- `-e .` installs in editable mode (changes reflect without reinstall)
- `uv run kimix` uses uv's managed environment automatically

---

## Environment Variables

Configure API keys before running. Priority: JSON config `api_key` field > `KIMI_API_KEY` > `KIMIX_API_KEY`.

| Variable | Description |
|----------|-------------|
| `KIMI_API_KEY` | Kimi API key |
| `KIMIX_API_KEY` | Fallback API key |

**Linux / macOS:**
```bash
export KIMI_API_KEY=your-api-key
```

**Windows PowerShell:**
```powershell
$env:KIMI_API_KEY="your-api-key"
```

---

## CLI Usage

### Subcommands

| Subcommand | Description | Common Options |
|------------|-------------|----------------|
| `serve` | Start HTTP server (OpenCode style) | `--host` (default `127.0.0.1`), `--port` (default `4096`) |
| `ssecli` | SSE CLI debugger for `kimix serve`. Supports `/new`, `/abort`, `/status`, `/sessions`, `/messages`, `/clear`, `/compact[:N]`, `/export[:path]`, `/help`; press `Ctrl+C` or send EOF (`Ctrl+D` / `Ctrl+Z`) to exit | `--host`, `--port`, `--debug` (saves raw event log as `sse_log_<YYYYMMDD_HHMMSS>.txt`) |

**Examples:**
```bash
uv run kimix serve --port 4096
uv run kimix ssecli --host 127.0.0.1 --port 4096 --debug
```

### LLM Config Initialization

If no `--config` is provided, the built-in default (`src/kimix/default_config.json`) is used.

Run `/init` in the interactive terminal to create the default config interactively:
```
/init
```

**Config fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Provider: `kimi`, `openai_legacy`, `openai_responses`, `anthropic`, `google_genai`, `gemini`, `vertexai` |
| `model` | Yes | Model name for API requests |
| `url` | Yes | API base URL |
| `max_context_size` | Yes | Max context length (`128k`, `200k`, `256k`, `512k`, `1M`) |
| `capabilities` | No | Model capabilities: `thinking`, `always_thinking`, `image_in`, `video_in` |
| `api_key` | No | API key (falls back to env vars) |
| `custom_headers` | No | Custom HTTP headers |
| `oauth` | No | OAuth config, e.g. `{"storage": "file", "key": "my-key"}` |
| `loop_control` | No | Loop params: `max_steps_per_turn`, `max_retries_per_step`, `max_ralph_iterations`, `reserved_context_size`, `compaction_trigger_ratio` |
| `max_tokens` | No | Max tokens per request |
| `show_thinking_stream` | No | Stream thinking process |
| `thinking_effort` | No | `off`, `low`, `medium`, `high`, `xhigh`, `max` |
| `temperature` | No | Sampling temperature `[0.0, 2.0]` |
| `background` | No | Background task settings |
| `notifications` | No | Notification settings |
| `mcp` | No | MCP (Model Context Protocol) config |
| `env` | No | Extra env vars (dict) |

Load custom config:
```bash
uv run kimix --config=<path>
```

### Launch Options

| Flag | Description |
|------|-------------|
| `-c`, `--clean` | Auto-delete cache on exit |
| `--no_think` | Disable thinking mode |
| `--no_yolo` | Disable YOLO mode |
| `--no_color` | Disable colored output |
| `--manually-cot` | Enable manual CoT (may use multiple sessions) |
| `--ralph` | Enable Ralph mode (optional iteration count) |
| `-s`, `--skill-dir` | Custom skill directory (repeatable) |
| `--config` | JSON config path. Searches: cwd parents, package parents, `PATH` |

> **Auto-loading skill directories**: On startup, Kimix also reads `.kimix/skill.json` in the current directory. If it contains a `skill_dir` field (string or array of strings) and the directories exist, they are automatically appended to the default skill search paths.

### Interactive Commands

| Command | Description |
|---------|-------------|
| `<path>` | Load file. `.py` files are executed directly (`__file__` points to the file); other files are read entirely as a single prompt |
| `/file:<path>` | Read entire file as a single prompt |
| `/todo:<path>` | Scan code files for TODO comments and prompt the agent to implement them. Supports `.py`, C/C++ family (`.c/.cpp/.h/.java/.js/.ts/.go/.rs`, etc.), Shell (`.sh/.bash/.zsh`), HTML/XML, Pascal, Lisp, SQL, and more |
| `/clear` | Clear current context |
| `/summarize` | Summarize context to memory |
| `/exit` | Exit |
| `/help` | Show help |
| `/context` | Print context usage |
| `/fix:<command>` | Run command, auto-retry on error |
| `/txt` | Multi-line text mode (end with `/end`, cancel with `/cancel`) |
| `/init` | Interactive LLM config initialization (resets session) |
| `/compact` | Compact the current session's conversation context |
| `/export:<path>` | Export the current session's messages to the specified file |
| `/resume:<id>` | Close current session and resume a session by ID |
| `/store:<id>` | Copy the current session to a new named session |
| `/load:<id>` | Copy a named session into a new anonymous session |
| `/ralph:on/off/<num>` | Set Ralph mode |
| `/reflection` | Reflect on the current conversation context, find misunderstandings caused by the current agent design, and change the source code to improve the project (requires a non-empty context; writes a change report to `docs/reflection_report_*.md`) |
| `/supervisor` | Enter multi-line input mode to create a session with the Supervisor role and execute one task (end with `/end`, cancel with `/cancel`) |
| `/plan` / `/plan:<file>` | Use the TodoMaker Agent to generate a task plan. Task requirements are provided via multi-line input (end with `/end`); `<file>` specifies the plan output file path, and will be overwritten if it already exists. After generation, you can review and modify the plan, then confirm to execute; a review prompt is appended after execution |
| `/cmd:<command>` | Execute system command |
| `/code:<path> [args...]` | Run a script file (`.py` or other executables) with optional arguments |


---

## MCP (Model Context Protocol)

Kimix can act as both an MCP client and an MCP server.

### Using MCP Servers

Add an MCP server to Kimix so that its tools, resources, and prompts are available to the agent:

```bash
# stdio server
kimix mcp add --transport stdio my-server -- npx -y @example/mcp-server

# streamable HTTP server
kimix mcp add --transport http my-server https://api.example.com/mcp

# list configured servers
kimix mcp list

# test a connection
kimix mcp test my-server
```

Project-level servers can also be committed to version control in `.kimix/mcp.json`. Kimix automatically merges global (`~/.kimi/mcp.json`), project (`.kimix/mcp.json`), and explicitly supplied configs, with explicit configs taking highest priority.

### Serving Kimix as an MCP Server

Expose the current Kimix runtime to external MCP clients such as Claude Desktop or Cursor:

```bash
# stdio (for clients that spawn a subprocess)
kimix mcp serve --transport stdio

# streamable HTTP
kimix mcp serve --transport http --host 127.0.0.1 --port 4097
```

By default the server exposes:

- **tools**: every tool in the active agent toolset
- **resources**: `AGENTS.md`, `README.md`, and project files under the work directory
- **prompts**: the agent's system prompt

Use `--no-resource` or `--no-prompt` to disable resources or prompts. Use `--agent-file` to load a specific agent specification.


