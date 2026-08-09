"""Shared tool-description / param-description helpers for shell+python tools.

Single source of truth for the param ``Field`` factories and the input
validators previously hand-written in the Bash, Powershell, Python and Run
tools, plus the alias-prose helper (``accepts_alias_text``) used across the
whole builtin tool list.

Generic conventions (head+tail fold, output dedup, ``rtk`` usage, parameter
aliases, ``wait_for_pattern`` semantics, ``timeout`` ranges, working-directory
rules) live once in ``kimi-cli/src/kimi_cli/agents/default/system.md`` under
"# Tool Conventions"; per-tool param descriptions stay minimal because the
conventions block travels with every request.  See ``plan.md`` Part 1.
"""

from typing import Any

from pydantic import AliasChoices, Field, model_validator

__all__ = [
    "accepts_alias_text",
    "timeout_field",
    "max_lines_field",
    "wait_for_pattern_field",
    "task_id_field",
    "deduplicate_output_field",
    "cwd_field",
    "mode_field",
    "normalize_mode_validator",
    "shell_cmd_required_validator",
]


def accepts_alias_text(*names: str, word: bool = True) -> str:
    """Return the "Accepts `a` or `b`." prose for a parameter with aliases.

    ``names`` are the accepted spellings in canonical-first order, e.g.
    ``accepts_alias_text("cmd", "command")``.  With three or more names the
    last two are joined with "or" (e.g. ``code | code_file``).  ``word``
    controls the " parameter" suffix (tool descriptions vs param fields).
    Delegates to ``kosong.tooling.alias_note`` so every tool (kimix *and*
    kimi-cli) shares one implementation.
    """
    from kosong.tooling import alias_note

    return alias_note(*names, word=word)


# ── param field factories (return pydantic Field) ────────────────────────

def timeout_field() -> Field:
    """``timeout``: seconds, default 30 (range is serialized in the schema)."""
    return Field(default=30, ge=1, le=900, description="Timeout in seconds.")


def max_lines_field() -> Field:
    """``max_lines``: head+tail fold cap, default unlimited (fold in conventions)."""
    return Field(default=None, ge=3, description="Max lines to return. None = unlimited.")


def wait_for_pattern_field() -> Field:
    """``wait_for_pattern``: regex to wait on (semantics in tool conventions)."""
    return Field(default=None, description="Pattern to wait for in the tool output.")


def task_id_field(payload: str = "cmd", tail: str = "being executed.") -> Field:
    """``task_id``: resume-session param; ``payload`` is the stdin carrier name.

    Bash/Powershell pass ``"cmd"``, Python passes ``"code"`` (with the extra
    "as a new script" tail), Run passes ``"command"``.
    """
    return Field(
        default=None,
        description=(
            "Existing session/task ID to continue. When provided, "
            f"'{payload}' is sent to the process stdin instead of {tail}"
        ),
    )


def deduplicate_output_field(*, accepts_alias: bool = False) -> Field:
    """``deduplicate_output`` (alias ``token_kill``): repeated-output dedup.

    ``accepts_alias=True`` (Python) adds the "Accepts `deduplicate_output` or
    `token_kill`." sentence; the dedup behavior itself is a tool convention.
    """
    text = "Deduplicate repeated output lines. Set to False to see raw, unfiltered output."
    if accepts_alias:
        text = (
            "Deduplicate repeated output lines. "
            "Accepts `deduplicate_output` or `token_kill`. "
            "Set to False to see raw, unfiltered output."
        )
    return Field(default=True, alias="token_kill", description=text)  # backward compat


def cwd_field(subject: str = "command", *, via_alias: bool = True) -> Field:
    """``cwd`` (alias ``workdir``): working directory for the tool.

    ``via_alias=True`` (Bash/Powershell) uses ``alias="workdir"`` so the JSON
    schema property is named ``workdir``; ``via_alias=False`` (Python) uses
    ``validation_alias=AliasChoices("cwd", "workdir")`` so the property stays
    named ``cwd``.  Both spellings are accepted on input either way.
    """
    description = (
        f"Working directory for the {subject} (absolute or relative path)."
    )
    if via_alias:
        return Field(
            default=None,
            alias="workdir",  # LLM can use "workdir" instead of "cwd"
            description=description,
        )
    return Field(
        default=None,
        validation_alias=AliasChoices("cwd", "workdir"),
        description=description,
    )


def mode_field(
    *,
    aliases: bool = True,
    execute_desc: str,
    send_desc: str,
    interactive_desc: str | None = None,
) -> Field:
    """``mode`` field with per-tool wording.

    Shell/Python tools document the deprecated ``run``/``background`` aliases
    and the ``interactive`` mode; Run (``aliases=False``, no interactive mode)
    uses its own literal wording.
    """
    if aliases:
        parts = [
            f"'execute' (alias: 'run'): {execute_desc}",
            f"'send' (alias: 'background'): {send_desc}",
        ]
    else:
        parts = [f"'execute': {execute_desc}", f"'send': {send_desc}"]
    if interactive_desc:
        parts.append(f"'interactive': {interactive_desc}")
    return Field(default="execute", description=" ".join(parts))


# ── shared validators ────────────────────────────────────────────────────

def normalize_mode_validator(data: dict) -> dict:
    """Convert deprecated boolean flags and mode aliases to canonical names.

    Body is identical to the pre-refactor ``BashParams._normalize_mode`` /
    ``PowershellParams._normalize_mode`` / ``Params._normalize_mode``.
    """
    if isinstance(data, dict):
        if data.get("interactive", False):
            data["mode"] = "interactive"
        if "mode" in data:
            if data["mode"] == "run":
                data["mode"] = "execute"
            elif data["mode"] == "background":
                data["mode"] = "send"
    return data


def shell_cmd_required_validator(field: str = "cmd"):
    """Return a ``model_validator(mode="after")`` enforcing shell input rules.

    Byte-identical behavior to the pre-refactor ``BashParams._validate_cmd`` /
    ``PowershellParams._validate_cmd``: the input (``cmd``) must be non-empty
    when executing, and must be present whenever a session is continued via
    ``task_id``.
    """

    def _validate(self: Any) -> Any:
        value = getattr(self, field)
        task_id = getattr(self, "task_id", None)
        if self.mode == "execute" and not value and task_id is None:
            raise ValueError(
                f"{field} cannot be empty when mode='execute' and no task_id"
            )
        if task_id is not None and not value:
            raise ValueError(
                f"{field} cannot be empty when continuing a session via task_id"
            )
        return self

    return model_validator(mode="after")(_validate)
