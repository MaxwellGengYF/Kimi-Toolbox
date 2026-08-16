---
name: tool
description: Guide for creating tools using CallableTool2 and Params pattern, plus YAML agent registration
---

# Tool Development Guide

Create custom tools with the `CallableTool2` + `Params` pattern, then register them in a YAML agent file.

## Quick Template

```python
"""Brief description of what this tool does."""
from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue

class Params(BaseModel):
    """Define tool parameters here."""
    required_param: str = Field(
        description="Description of this parameter for the LLM."
    )
    optional_param: str | None = Field(
        default=None,
        description="Optional parameter with default value."
    )

class MyTool(CallableTool2):
    name: str = "MyTool" # Tool identifier
    description: str = "What this tool does."  # For LLM to understand usage
    params: type[Params] = Params # Link to Params class

    async def __call__(self, params: Params) -> ToolReturnValue:
        """Execute the tool logic."""
        try:
            # Your tool logic here
            result = f"Processed: {params.required_param}"
            return ToolOk(output=result)
        except Exception as e:
            return ToolError(
                message=str(e),
                output="Partial output if available",
                brief="Short error summary"
            )
```

## Key Patterns

- **Return values**: `ToolOk(output=...)` on success; `ToolError(message=..., output=..., brief=...)` on failure.
- **Validators**: `ge`/`le`/`gt`/`lt` (numeric), `min_length`/`max_length`/`pattern` (strings/collections), `default_factory` for mutable defaults.
- **Best practices**: type hints everywhere; clear `Field(description=...)`; always async `__call__`; handle exceptions → ToolError; one tool = one job.

## Registration

Every new tool must be registered in a YAML agent file as `"module.path:ClassName"` under `tools:`. `KimiToolset.load_tools()` splits on the last `:`, imports the module, gets the class, instantiates, and adds it. Agent files: `agent_worker.json` (default), `agent_boss.json`, `agent_subagent.json`; base agents in `kimi-cli/src/kimi_cli/agents/default/` (`agent.yaml`, `coder.yaml`, `explore.yaml`, `plan.yaml`). Child YAML `tools:` replaces the parent's list; use `allowed_tools:`/`exclude_tools:` to restrict. Use `kimix.tools.*` prefix for new tools under `src/kimix/tools/`.

Full complete example, Params reference, background-task tools, interactive-session reuse, and YAML details: read `references/guide.md`.
