# Long-Running Tasks

Kimix provides the **`/plan` command** for complex or time-consuming tasks: it lets a dedicated Planner Agent generate a reviewable task plan, then a Worker Agent serially implements and reviews it after confirmation.

## `/plan`

Flow: **plan generation → user review → execution → review**. Best for tasks that need explicit steps and user confirmation before execution.

### Basic Usage

```
/plan
>>>> Start input requirement for plan, end with /end, cancel with /cancel
Add complete error handling to this project:
1. Add parameter validation to all functions
2. Unify exception types and error codes
3. Add logging
/end
```

### Execution Flow (`prompt_plan_async`)

See `src/kimix/utils/prompt.py` and `src/kimix/cli_impl/commands.py`.

#### Phase 1: Plan Generation

1. **Create Planner session**: Using `agent_planner.json` config and the `TodoMaker` system prompt.
2. **Specify plan file**: If no file path is provided, a `plan_<uuid>.md` is auto-generated under `.kimix_cache/` in the current directory; if the specified file already exists, it is deleted first.
3. **Generate plan**: The Planner reads the requirement, breaks the task into steps, and writes the plan file via the `WritePlan` tool. Generation retries up to **3 times** to ensure the plan is written correctly.
4. **Open for review**: After generation, the file is opened with the system default application for review.

#### Phase 2: Review & Revision

After the plan is generated, a review loop begins:

- Prompt: `Do you want to implement the plan? (y/n)`
  - Input `y`: enter the execution phase.
  - Input `n` or anything else: enter the revision flow.
- Revision prompt: `Please describe the changes you want (/quit to give up):`
  - Input `/quit`: abandon execution.
  - Input specific feedback: the Planner updates the plan file using the `WritePlan` or `EditPlan` tools based on the feedback, then reopens it for review. The loop repeats until confirmed or abandoned.

#### Phase 3: Execution & Review

1. **Close Planner session**: After user confirmation, the Planner session is closed.
2. **Create Worker session**: A regular session is created with the default Worker Agent.
3. **Send implementation prompt**:
   - If the plan file is smaller than **100 KB**, the plan content is embedded directly in the prompt;
   - If the plan file is larger than **100 KB**, the Agent is prompted to read and execute the plan file.
4. **Append review prompt**: After implementation, an additional review prompt is sent asking the Agent to check whether the plan was fully completed.

### Specifying the Plan Output File

Use `/plan:<file>` to specify the plan output path. Note: this is the **plan output file**; the task requirement is still provided via multi-line input:

```
/plan:docs/plan.md
>>>> Start input requirement for plan, end with /end, cancel with /cancel
Add complete error handling to this project:
1. Add parameter validation to all functions
2. Unify exception types and error codes
3. Add logging
/end
```

If the specified plan file already exists, it will be overwritten.

---

## Long Prompts & Error Handling

### Auto-Truncate

Prompts exceeding **65536 chars** are exported to a temp file and replaced with `read and execute: <temp_file>`.

### Auto-Retry

`prompt_async` retries up to **5 times** with exponential backoff:

| Status | Behavior |
|--------|----------|
| `429` | Wait `min(2^attempt, 60)`s, retry |
| `400`, `500`, `502`, `503` | Exponential backoff, retry |
| Other | Wait 1s, retry; throw on last attempt |

---

### TODO Reminder

After each main prompt execution, the system checks whether there are unfinished `todos` in the current session. If so, a `<system-reminder>` is automatically appended reminding the Agent to complete all pending / in_progress todos before finishing. This works with the `todo_write` tool to ensure intermediate steps are not missed in long tasks.

---

## `todo_write`

Progress tracking tool, auto-invoked during execution.

- **Read mode** (`todos` = `null`): returns current todo list
- **Write mode** (list provided): updates and persists todos

```python
class Todo:
    title: str
    status: str  # "pending" | "in_progress" | "done"
```

**Persistence:**
- Root Agent: `state.todos`
- Sub Agent: `state.json` in agent directory

---

## When to Use `/plan`

| Feature | `/plan` |
|---------|---------|
| Execution | Serial |
| Task split | Linear steps |
| Resume | No (current implementation completes generation, review, execution, and review in one flow) |
| Progress | Step visualization |
| Use case | Ordered, dependent tasks |
| Overhead | Lower (single session) |
| Examples | Feature implementation, code refactoring |

**Choose `/plan`** when the task needs explicit steps, user review, and confirmation before execution, suitable for one-shot long tasks.

