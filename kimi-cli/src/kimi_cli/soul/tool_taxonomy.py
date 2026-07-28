"""Shared tool classification constants (P1/P2).

Single source of truth for "which tools edit files" and "which tools can
run verification", so the target-churn provider (P1) and the verification
gate (P2) never drift apart.
"""

from __future__ import annotations

# Tools whose primary effect is modifying a file's content.
EDIT_TOOLS: frozenset[str] = frozenset(
    {
        "WriteFile",
        "EditFile",
        "HashEdit",
        "WritePlan",
        "EditPlan",
        # Aliases used by other tool naming schemes.
        "Write",
        "Edit",
        "Replace",
        "StrReplace",
    }
)

# Tools that execute arbitrary shell commands.
SHELL_TOOLS: frozenset[str] = frozenset({"Bash", "Powershell", "Run"})

# Tools that can act as verification signals for the P2 gate:
# marking a todo `done` triggers its attached verification code; shell
# tools can run the project's own test/check commands.
VERIFICATION_TOOL_HINTS: frozenset[str] = frozenset({"TodoList"}) | SHELL_TOOLS

# Argument keys that carry the target file path for edit tools.
PATH_PARAM_KEYS: tuple[str, ...] = ("path", "file_path", "filename")

# Argument keys that carry the command string for shell tools.
COMMAND_PARAM_KEYS: tuple[str, ...] = ("command", "cmd", "code")
