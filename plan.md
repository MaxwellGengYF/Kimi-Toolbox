# Plan: De-duplicate shared tool prompts (Bash / Python / Powershell / Run)

## 1. Problem

The Bash tool (`src/kimix/tools/file/bash/bash_tool.py`) and the Python tool
(`src/kimix/tools/py/__init__.py`) — plus the Powershell tool
(`src/kimix/tools/file/bash/pwsh_tool.py`) and the Run tool
(`src/kimix/tools/file/run.py`) — each hand-write large, largely-identical
prompt fragments:

- tool `description` prose (head+tail fold rules, `cmd`/`command` aliases,
  `cwd`/`workdir` aliases, dedup/`token_kill`, `rtk` usage, env scrubbing)
- param field descriptions inside the pydantic `Params` classes (`timeout`,
  `task_id`, `wait_for_pattern`, `max_lines`, `deduplicate_output`, `cwd`,
  `mode`)
- identical model validators (`_normalize_mode`, input-required checks)

Every tool in the tool list is serialized to the LLM as a self-contained JSON
object (`tool.description` + the params JSON schema, see
`kimi-cli/packages/kosong/src/kosong/chat_provider/openai_common.py:345`,
`anthropic.py:829`, `google_genai.py:374`). There is **no cross-tool
reference** in the wire format, so the same text is re-sent N times on every
API request.

## 2. Current state (runtime census, measured 2025-07)

| Fragment | Where | Copies | Duplicated chars (excess) |
|---|---|---|---|
| "Output longer than `max_lines` is collapsed via head+tail fold … Set `max_lines=None`…" | Bash, Python, Run + Powershell `__init__` | 4 | ~528 |
| `timeout` desc: "Timeout in seconds (1-900)." | all 4 `Params` | 4 | ~81 |
| `max_lines` desc (113 ch) | all 4 `Params` | 4 | ~339 |
| `wait_for_pattern` desc (133 ch) | all 4 `Params` | 4 | ~399 |
| `task_id` desc (~114 ch, 'cmd'/'code'/'command' variant) | Bash, Pwsh, Python, Run | 4 | ~228 |
| `deduplicate_output` desc (~115 ch) | Bash/Run identical, Python variant | 2+1 | ~115 |
| `cwd` desc (~62 ch, 'command'/'script' variant) | Bash/Pwsh identical, Python variant | 2+1 | ~62 |
| "Accepts `cmd` or `command` parameter." + "`cwd`/`workdir` sets the working directory…" | Bash + Powershell `__init__` | 2 | ~97 |
| `_normalize_mode` model_validator | BashParams, PyParams — byte-identical (~20 lines) | 2 | (source) |
| input-required validators | BashParams, PyParams, PwshParams, RunParams | 4 | (source) |

**Total duplicated wire-visible text ≈ ~1850 chars ≈ ~450 tokens per request**
(4 chars/token, conservative), **recurring on every LLM call** while these
tools are enabled. Current description sizes: Bash 711 ch, Python 1886 ch,
Run 507 ch (Powershell appends ~400 ch at init).

Note: `_interactive_scope_text(is_shell=…)` in `src/kimix/tools/common.py`
already proves the shared-helper pattern; the rest of the prose was never
consolidated.

## 3. Root cause

Each tool was written independently; only the interactive/background-scope
sentence was extracted into `kimix/tools/common._interactive_scope_text`.
Also an inconsistency: the Bash tool applies `rtk` rewriting and
`deduplicate_output` at runtime but never documents them, while Python
documents a longer rtk paragraph — the duplicated prose is also *divergent*
(example command lists differ: "git, npm, etc." vs "pytest, ruff, etc.").

## 4. Solution — two layers

### Layer 1 (source DRY, zero wire change, do first)
Extract every shared fragment into one module and have all four tools compose
from it. Text stays byte-identical on the wire → no behavior change, safe.

### Layer 2 (real token savings, optional follow-up)
Trim the wire format itself:
- **(a) Internal redundancy**: drop the fold paragraph from each tool
  `description` — it already lives in the `max_lines` param description, which
  is serialized in the same tool object (Bash, Run keep it in `description`
  *and* in `max_lines` today).
- **(b) Global conventions block (biggest win)**: hoist the *generic*
  conventions (head+tail fold, `deduplicate_output`/`token_kill`, `rtk` usage,
  `cwd`/`workdir` aliases) into ONE place in the system prompt —
  `kimi-cli/src/kimi_cli/agents/default/system.md` (a new
  `${KIMI_TOOL_CONVENTIONS}` section or static paragraph) — and delete the
  copies from all four tool descriptions/params, keeping only tool-specific
  text. N copies → 1 copy: ~1850 chars/tool-list per request.

Recommend doing Layer 1 now (low risk, unlocks Layer 2) and Layer 2(b) only
after A/B-measuring model behavior — terse descriptions can affect tool-call
quality, so verify before shipping.

## 5. Implementation steps

### Step 1 — new module `src/kimix/tools/prompt_common.py`

Imports only from `kimix.tools.common` (no cycle; `common` never imports it).
Contents (sketch):

```python
"""Shared tool-description / param-description text for shell+python tools."""
from pydantic import Field
from kimix.tools.common import _interactive_scope_text  # already shared

# ── description fragments ────────────────────────────────────────────────
MAX_LINES_FOLD_TEXT = (
    "Output longer than `max_lines` is collapsed via head+tail fold "
    "(first N + last N lines, with middle replaced by a truncation marker). "
    "Set `max_lines=None` for unlimited output."
)
ACCEPTS_CMD_OR_COMMAND_TEXT = "Accepts `cmd` or `command` parameter."
CWD_WORKDIR_TEXT = "`cwd`/`workdir` sets the working directory for this command."

def dedup_output_text(example_commands: str) -> str:
    return (
        f"Deduplicate repeated output lines from known commands "
        f"({example_commands}, etc.). Set to False to see raw, unfiltered output."
    )

RTK_TEXT = (
    "When invoking known CLI tools (pytest, ruff, mypy, pip, uv, git, npm, …) "
    "via subprocess, you can use the \"rtk\" executable to reduce token usage: "
    "`rtk <process> <arguments...>`. rtk automatically deduplicates and "
    "truncates the output of the wrapped command."
)

# ── param field factories (return pydantic Field) ────────────────────────
def timeout_field() -> Field:
    return Field(default=30, ge=1, le=900, description="Timeout in seconds (1-900).")

def max_lines_field() -> Field:
    return Field(default=None, ge=3, description=(
        "Max lines to return via head+tail fold. <N> head lines + <N> tail "
        "lines kept; middle collapsed. None = unlimited."
    ))

def wait_for_pattern_field() -> Field:
    return Field(default=None, description=(
        "Optional regex pattern. After starting or sending input, the tool "
        "blocks up to 'timeout' seconds until the pattern appears in output."
    ))

def task_id_field(payload: str = "cmd") -> Field:
    return Field(default=None, description=(
        "Existing session/task ID to continue. When provided, "
        f"'{payload}' is sent to the process stdin instead of being executed."
    ))

def deduplicate_output_field(example_commands: str = "git, npm") -> Field:
    return Field(default=True, alias="token_kill",  # backward compat
                 description=dedup_output_text(example_commands))

def cwd_field(subject: str = "command") -> Field:
    return Field(default=None, alias="workdir",
                 description=f"Working directory for the {subject} (absolute or relative path).")

def mode_field(shell_name: str | None = None) -> Field:
    # 'execute'/'send'/'interactive' with tool-specific wording
    ...

# ── shared validators ────────────────────────────────────────────────────
def normalize_mode_validator(data: dict) -> dict:
    """Convert deprecated boolean flags and mode aliases to canonical names."""
    # body == current BashParams._normalize_mode / Params._normalize_mode

def require_input_validator(*, field: str, task_field: str = "task_id") -> "classmethod":
    # shared 'cannot be empty' checks (keep per-tool quirks as thin wrappers)
```

### Step 2 — refactor the four `Params` classes

- `BashParams`, `PowershellParams` → `timeout_field()`, `task_id_field("cmd")`,
  `wait_for_pattern_field()`, `max_lines_field()`,
  `deduplicate_output_field("git, npm")`, `cwd_field("command")`,
  `mode_field("bash"/"powershell")`, plus `@model_validator` wrapping
  `normalize_mode_validator`.
- `Params` (Python) → same factories with `task_id_field("code")`,
  `deduplicate_output_field("pytest, ruff")`, `cwd_field("script")`; keep
  `output_path` and the `code`/`file` alias fields tool-specific.
- `RunParams` → same factories (`task_id_field("command")`); keep
  `command`/`env`/`run_in_background`/`shell` tool-specific.

### Step 3 — refactor the four `description` strings

- Bash: `"Execute a bash command. Supports Unix-style / POSIX bash syntax. " + ACCEPTS_CMD_OR_COMMAND_TEXT + " " + CWD_WORKDIR_TEXT + " " + "Prefer `Glob`/`Grep` tools over `find`/`ls`/`grep`/`rg`…" + MAX_LINES_FOLD_TEXT + _interactive_scope_text(is_shell=True)` (plus the Windows slash sentence appended in `__init__`).
- Python: tool-specific opener + `MAX_LINES_FOLD_TEXT` + `dedup_output_text("pytest, ruff")` + `RTK_TEXT` + Python-interpreter notes + `"Set `cwd` (or `workdir`)…"` + env-scrub notes + `_interactive_scope_text(is_shell=False)`.
- Powershell: keep `load_desc(pwsh_tool.md)` + shared fragments via constants (replaces the hand-typed `__init__` strings at lines 281–286).
- Run: tool-specific opener + `ACCEPTS_CMD_OR_COMMAND_TEXT` + `MAX_LINES_FOLD_TEXT` + `_interactive_scope_text(is_shell=False)`.

Constraint: the *composed* strings must stay byte-identical to today (Layer 1),
so add a snapshot test (Step 5).

### Step 4 — Layer 2 (token savings), only after Layer 1 lands

- **(a)** Drop `MAX_LINES_FOLD_TEXT` from `description` (it remains in the
  `max_lines` param schema) — removes ~176 ch × 4 tools.
- **(b)** Add a "Tool conventions" block once in
  `kimi-cli/src/kimi_cli/agents/default/system.md` (plain static text or a new
  `${KIMI_TOOL_CONVENTIONS}` arg; check `system_prompt_args` wiring in
  `kimi-cli/src/kimi_cli/agentspec.py`), then delete
  `MAX_LINES_FOLD_TEXT`, `dedup_output_text`, `RTK_TEXT`, and the
  `cwd`/`workdir` alias sentences from all four tools, leaving only
  tool-specific sentences. Measure before/after with the script below.

### Step 5 — tests & measurement

- New test `tests/unit/tools/test_prompt_common.py`:
  - `test_shared_fragments_identical`: for each shared fragment,
    `BashParams.model_fields["timeout"].description ==
    Params.model_fields["timeout"].description` (etc.).
  - `test_descriptions_unchanged` (Layer-1 guard): compare the four composed
    `description` strings against expected snapshots; then update snapshots
    when Layer 2 lands.
  - `test_validators_shared`: `_normalize_mode` behaves identically for
    `{"interactive": True}`, `{"mode": "run"}`, `{"mode": "background"}` on
    all four Params classes.
- Measurement script (documents the saving):

```python
import asyncio, json
from kimix.tools.common import make_tool_error  # noqa
from kimix.tools.file.bash.bash_tool import Bash
from kimix.tools.py import Python
# serialize the real tool list the way the chat provider does:
# json.dumps({t.name: {"description": t.description,
#                      "schema": t.params.model_json_schema()}})
# and diff total char count before vs after Layer 2.
```

## 6. Verification

Run after every step:

```
uv run tools/syntax_check.py src/kimix/tools/prompt_common.py \
    src/kimix/tools/file/bash/bash_tool.py \
    src/kimix/tools/file/bash/pwsh_tool.py \
    src/kimix/tools/py/__init__.py \
    src/kimix/tools/file/run.py \
    tests/unit/tools/test_prompt_common.py

uv run pytest tests/test_bash.py tests/test_powershell.py tests/test_run.py \
    tests/unit/tools/test_python.py tests/unit/tools/test_py_syntax_check.py \
    tests/unit/tools/test_prompt_common.py -q
```

Relevant existing suites (no exact-description assertions were found, so the
refactor is low-risk): `tests/test_bash.py`, `tests/test_powershell.py`,
`tests/test_run.py`, `tests/unit/tools/test_python.py`,
`tests/unit/tools/test_py_syntax_check.py`, `tests/test_shell_common.py`.

## 7. Risks / tradeoffs

- **Layer 1**: none functional (text byte-identical); only import/refactor risk.
- **Layer 2(a)**: description loses the fold reminder — the `max_lines` param
  still carries it in the same tool object.
- **Layer 2(b)**: system.md is shared by every agent; conventions must be
  phrased generically. Terse tool descriptions can change tool-selection
  behavior — A/B measure (or at least sanity-check real sessions) before
  shipping.

## 8. Expected savings

- **Layer 1**: ~130 lines of duplicated source removed; single source of truth
  (`prompt_common.py`); fixes the Bash-vs-Python rtk/dedup wording divergence.
- **Layer 2**: ~1850 chars (~450 tokens) per tool-list serialization per LLM
  request while Bash+Python+Powershell+Run are enabled; more with subagents
  that carry the same tool list.

---

# Part 2 — Full builtin tool list audit (beyond Bash/Python)

Part 1 covered the Bash/Python/Powershell/Run cluster. Auditing the **whole
builtin tool list** (`kimi-cli/src/kimi_cli/tools/` + `src/kimix/tools/`)
reveals more reusable token savings. Rough total wire-visible tool-list budget:
**~15,000 chars ≈ ~3,800 tokens per request** (top consumers: ReadMediaFile
2.7 KB, Python 1.9 KB, Grep 1.4 KB, ReadFile 1.3 KB).

## 2.1 Cross-tool duplicated fragments (wire-visible)

| Pattern | Occurrences | Excess |
|---|---|---|
| `Accepts \`X\` or \`Y\`.` alias prose, hand-written per param | **20×** in 11 files (read, write, replace, glob, todo, bash, pwsh, py, run, background, hash) | ~700 chars |
| "head+tail fold" concept | 5 tools (Glob `max_results`, Bash, Pwsh, Python, Run) | ~530 chars |
| `wait_for_pattern` desc (~133 ch) | 4 identical (Bash, Pwsh, Python, Run) + 1 variant (TaskOutput) | ~400 chars |
| `timeout` desc — 3 phrasings | (1-900)×4; FetchURL "(1-300)"; Glob+Grep "Maximum time… search" | ~150 chars |
| "Output file path." | 3 (Python `output_path`, Run, TaskOutput) | ~40 chars |
| "Absolute … outside the working directory" | 4 (ReadFile, HashRead path descs; WriteFile, EditFile "Absolute paths required…") | ~180 chars |
| `_normalize_mode` / input-required validators | BashParams/PyParams identical; Run/Pwsh near-identical | (source) |

Note: `kosong.tooling` already centralizes the alias *mechanism*
(`FIELD_ALIASES_GENERAL/FILE/WEB`, `_COMMON_FIELD_ALIASES`), but the
"Accepts X or Y" **prose** is still hand-written next to every alias.

## 2.2 Intra-tool redundancy (same text twice inside one tool)

- **ReadFile**: the glob-pattern rule (`*`, `?`, `[...]`, recursive `**`,
  unsafe-pattern rejection) is written in full in `file/read.md` **and**
  again in the `path` param description (`read.py:70-74`). ~230 chars × 2.
- **Retrieve**: module docstring ≈ tool description ≈ `query` param
  description — the same idea 3× (`memory/__init__.py:1-6, 38-44, 22`).
- **TodoList**: `_TODO_LIST_DESCRIPTION_NEUTRAL` (line 181) is **dead code** —
  `__init__` always overrides it with `_tool_description(kind)`, so the
  ~330-char neutral variant is never serialized; "verification" appears 14×
  and the example `!pytest tests/ -x -q` 5× across the neutral desc, instance
  desc, `code` field desc, and the schema patch.
- **ReadMediaFile** (`read_media.md`, 2.7 KB): boilerplate tips ("Make sure
  you follow the description of each tool parameter", "This tool is a tool
  that you typically want to use in parallel") plus the downsample guidance
  duplicated across two bullets.

## 2.3 Conciseness candidates (keep meaning, cut text)

- **Python description (1886 chars)** — the longest inline description. Cut:
  the rtk paragraph (~450 ch) → 1 line "use `rtk <cmd>` to reduce token
  usage" (full behavior already auto-applied); the env-scrub sentence
  (~200 ch) → "child env is scrubbed of secret-looking vars". Keep the
  interpreter-resolution facts.
- **Grep description + params (~1.4 KB)** — trim "Example:
  pattern='def foo\(.*?\):'…" (the `multiline` param already says newline
  patterns auto-enable multiline).
- **ReadMediaFile**: drop the two boilerplate bullets, keep the actionable
  ones (downsample → `region`/`full_resolution`, error → resize first).
- **agent/description.md (788 ch)** — "**When Not To Use**" duplicates
  guidance already in `system.md` ("Prefer Glob/Shell for small searches").

## 2.4 Tool-merge candidates (fewer tool entries = fixed overhead saved)

- **FetchURL + SearchWeb → one `Web` tool** with `action: fetch|search`:
  they share service-config plumbing, OAuth headers, `X-Msh-Tool-Call-Id`,
  timeout/network error handling, and "service not configured" errors
  (`fetch.py:52-65` vs `search.py:57-71`). Saves one tool entry + ~150 lines
  of near-identical error handling. Descriptions are already short, so this
  is mostly a code/context win.
- **ReadFile + ReadMediaFile**: rejected — different payloads/params; only
  merge the shared "path safety" validation (`_validate_path` is duplicated
  in read.py, hash_line.py, read_media.py).
- **TaskOutput** already merged TaskList/TaskStop/TaskOutput into one tool
  via `action` (good precedent for the merges above).
- **Run** is auto-disabled whenever bash exists (`run.py` `SkipThisTool`),
  so it rarely appears in the wire list — keep as-is.
- **HashRead/HashEdit**: niche hash-anchored editing; keep separate.

## 2.5 Additional token-saving levers

1. **Generate alias prose from metadata**: replace hand-written
   "Accepts `X` or `Y`." (20×) with a helper that reads `field_aliases` /
   pydantic `alias` — e.g. `alias_note("path", "file_path")` in
   `prompt_common.py` — or drop it entirely in favor of one global note
   "all parameters accept their documented aliases" (Part 1, Layer 2b).
2. **Dead code removal**: delete `_TODO_LIST_DESCRIPTION_NEUTRAL` and other
   class-level descriptions never serialized (keep one source of truth).
3. **Extend the global conventions block** (Part 1 §4 Layer 2b) to also cover:
   `wait_for_pattern` semantics, `timeout` ranges, head+tail fold, and the
   "absolute path outside workdir" rule — then delete those fragments from
   all 5-7 tools.
4. **Shorten param descriptions whose constraints are already in the JSON
   schema** — `ge`/`le`/`default` are serialized, so e.g. "Timeout in seconds
   (1-900)." → "Timeout (s)." without losing information.

## 2.6 Consolidated expected savings (Parts 1 + 2)

| Lever | Savings per request |
|---|---|
| Part 1 Layer 2(b): global conventions block | ~1,850 chars (~450 tokens) |
| §2.1 alias prose generated once (20× → 1×) | ~700 chars (~175 tokens) |
| §2.2 intra-tool dedup (ReadFile glob, Retrieve 3×, TodoList dead code) | ~900 chars (~225 tokens) |
| §2.3 concise rewrites (Python desc, Grep, ReadMediaFile, agent.md) | ~1,200 chars (~300 tokens) |
| §2.4 FetchURL+SearchWeb merge | ~1 tool entry + ~200 chars |
| **Total** | **~4,850+ chars ≈ ~1,200+ tokens per LLM request** |

Apply in this order: Part 1 Layer 1 (source DRY, zero wire change) → Part 1
Layer 2(a) internal redundancy → §2.2 dead code → §2.1 alias generation →
§2.3 conciseness → Part 1 Layer 2(b) global block → §2.4 merge (last, needs
A/B check). Re-run the measurement script (§5) after each step.

---

# Part 3 — Execution status (2025-07, applied)

Everything below was implemented and verified in this repo (see
`tools/measure_tool_prompt.py` and `tests/unit/tools/test_prompt_common.py`):

- **Part 1 Layer 1**: `src/kimix/tools/prompt_common.py` added; the four
  `Params` classes and tool `description`s compose from it. Composed text was
  verified byte-identical to the pre-refactor wire output (baseline captured
  in `tools/baseline_tool_prompts.json`).
- **Part 1 Layer 2(a)**: fold paragraph dropped from the four descriptions
  (−708 chars).
- **Part 1 Layer 2(b)**: "# Tool Conventions" block added to
  `kimi-cli/src/kimi_cli/agents/default/system.md`; generic fragments deleted
  from the four tools; `timeout`/`max_lines`/`wait_for_pattern`/
  `deduplicate_output` param descriptions shortened; TaskOutput's
  `wait_for_pattern` now uses the shared factory.
- **Measured 4-tool budget**: 12,594 → 9,857 chars (−2,737 ≈ −684 tokens).
- **§2.1**: `kosong.tooling.alias_note()` added (single implementation);
  `prompt_common.accepts_alias_text()` delegates to it; 22 hand-written
  "Accepts `X` or `Y`" fragments replaced across bash/pwsh/py/run/background/
  agent/note/todo/glob/read/replace/write.
- **§2.2**: `_TODO_LIST_DESCRIPTION_NEUTRAL` deleted (dead code); `read.md`
  glob-rule paragraph deduped (kept in the `path` param); Retrieve description
  trimmed (was 3× the same idea).
- **§2.3**: Python description (rtk → 1 line, env-scrub → short, then both
  hoisted into conventions); Grep description (dropped the `def foo` example);
  ReadMediaFile (dropped 4 boilerplate bullets; inline snapshots updated);
  `agent/description.md` (dropped "When Not To Use", duplicated in system.md).
- **§2.5.3**: "absolute path outside working directory" fragments removed from
  read/write/replace/hash_line path descriptions (rule now lives once in the
  conventions block).

**§2.4 FetchURL+SearchWeb merge — NOT applied (measured decision).** The plan
marks this "needs A/B check". Serializing the real wire objects:

- Default agent today (kimi-cli `FetchURL` + `SearchWeb`): 1,836 chars.
- Capability-preserving merged `Web` (action + union of both param sets):
  1,936 chars → **+100 chars, no token saving**; only benefit is one fewer
  tool entry.
- Lean merged `Web` (action + `url`/`query`/`limit`/`include_content`):
  904 chars (−932) but silently drops `method`/`headers`/`body`/
  `follow_redirects`/`max_redirects` — a fetch capability regression.
- kimix agents (kimix `FetchURL` + `SearchWeb`) would *grow* by +319 chars.

Both designs change the model's tool-selection surface and require the A/B
check the plan calls for; the lean design also regresses capabilities. The
merge is therefore deferred pending a live-model A/B evaluation.

