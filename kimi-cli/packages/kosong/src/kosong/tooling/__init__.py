from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Iterable
from dataclasses import dataclass, field
from rapidfuzz import fuzz
from functools import lru_cache
import orjson
import typing
from typing import Any, ClassVar, Protocol, Self, cast, override, runtime_checkable

import jsonschema
import pydantic
from pydantic import BaseModel, GetCoreSchemaHandler, model_validator
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

from kosong.message import ContentPart, ToolCall
from kosong.utils.jsonschema import deref_json_schema
from kosong.utils.jsonx import loads_relaxed
from kosong.utils.typing import JsonType

type ParametersType = dict[str, Any]


# Precompiled meta-schema validator. ``jsonschema.validate()`` re-creates a
# validator bound to the meta-schema on every call (several ms each); reusing
# a single instance makes per-tool parameter schema validation ~6x faster,
# which matters at startup when many tools are instantiated at once.
_META_SCHEMA_VALIDATOR = jsonschema.Draft202012Validator(
    jsonschema.Draft202012Validator.META_SCHEMA
)


class Tool(BaseModel):
    """The definition of a tool that can be recognized by the model."""

    name: str
    """The name of the tool."""

    description: str
    """The description of the tool."""

    parameters: ParametersType
    """The parameters of the tool, in JSON Schema format."""

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        _META_SCHEMA_VALIDATOR.validate(self.parameters)
        return self


class DisplayBlock(BaseModel, ABC):
    """
    A block of content to be displayed to the user.

    Similar to `ContentPart`, but scoped to user-facing UI.
    `ContentPart` is for model-facing message content; `DisplayBlock` is for tool/UI extensions.

    Unlike `ContentPart`, Kosong users may directly subclass `DisplayBlock` to define custom
    display blocks for their applications.
    """

    __display_block_registry: ClassVar[dict[str, type["DisplayBlock"]]] = {}

    type: str
    ...  # to be added by subclasses

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        invalid_subclass_error_msg = (
            f"DisplayBlock subclass {cls.__name__} must have a `type` field of type `str`"
        )

        type_value = getattr(cls, "type", None)
        if type_value is None or not isinstance(type_value, str):
            raise ValueError(invalid_subclass_error_msg)

        cls.__display_block_registry[type_value] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # If we're dealing with the base DisplayBlock class, use custom validation
        if cls.__name__ == "DisplayBlock":

            def validate_display_block(value: Any) -> Any:
                # if it's already an instance of a DisplayBlock subclass, return it
                if hasattr(value, "__class__") and issubclass(value.__class__, cls):
                    return value

                # if it's a dict with a type field, dispatch to the appropriate subclass
                if isinstance(value, dict) and "type" in value:
                    type_value: Any | None = cast(dict[str, Any], value).get("type")
                    if not isinstance(type_value, str):
                        raise ValueError(f"Cannot validate {value} as DisplayBlock")
                    target_class = cls.__display_block_registry.get(type_value)
                    if target_class is None:
                        data = {k: v for k, v in cast(dict[str, Any], value).items() if k != "type"}
                        return UnknownDisplayBlock.model_validate(
                            {"type": type_value, "data": data}
                        )
                    return target_class.model_validate(value)

                raise ValueError(f"Cannot validate {value} as DisplayBlock")

            return core_schema.no_info_plain_validator_function(validate_display_block)

        # for subclasses, use the default schema
        return handler(source_type)


class UnknownDisplayBlock(DisplayBlock):
    """Fallback display block for unknown types."""

    type: str = "unknown"
    data: JsonType


class BriefDisplayBlock(DisplayBlock):
    """A brief display block with plain string content."""

    type: str = "brief"
    text: str


class ToolReturnValue(BaseModel):
    """The return type of a callable tool."""

    is_error: bool
    """Whether the tool call resulted in an error."""

    # For model
    output: str | list[ContentPart]
    """The output content returned by the tool."""
    message: str
    """An explanatory message to be given to the model."""

    # For user
    display: list[DisplayBlock]
    """The content blocks to be displayed to the user."""

    # For debugging/testing
    extras: dict[str, JsonType] | None = None

    @property
    def brief(self) -> str:
        """Get the brief display block data, if any."""
        for block in self.display:
            if isinstance(block, BriefDisplayBlock):
                return block.text
        return ""


class ToolOk(ToolReturnValue):
    """Subclass of `ToolReturnValue` representing a successful tool call."""

    def __init__(
        self,
        *,
        output: str | ContentPart | list[ContentPart],
        message: str = "",
        brief: str = "",
        display_block: DisplayBlock | None = None
    ) -> None:
        super().__init__(
            is_error=False,
            output=([output] if isinstance(output, ContentPart) else output),
            message=message,
            display=[display_block] if display_block is not None else ([BriefDisplayBlock(text=brief)] if brief else []),
        )


class ToolError(ToolReturnValue):
    """Subclass of `ToolReturnValue` representing a failed tool call."""

    def __init__(
        self, *, message: str, brief: str, output: str | ContentPart | list[ContentPart] = ""
    ):
        super().__init__(
            is_error=True,
            output=([output] if isinstance(output, ContentPart) else output),
            message=message,
            display=[BriefDisplayBlock(text=brief)] if brief else [],
        )


class CallableTool(Tool, ABC):
    """
    The abstract base class of tools that can be called as callables.

    The tool will be called with the arguments provided in the `ToolCall`.
    If the arguments are given as a JSON array, it will be unpacked into positional arguments.
    If the arguments are given as a JSON object, it will be unpacked into keyword arguments.
    Otherwise, the arguments will be passed as a single argument.
    """

    @property
    def base(self) -> Tool:
        """The base tool definition."""
        return self

    async def call(self, arguments: JsonType) -> ToolReturnValue:
        from kosong.tooling.error import ToolValidateError

        try:
            jsonschema.validate(arguments, self.parameters)
        except jsonschema.ValidationError as e:
            return ToolValidateError(str(e))

        if isinstance(arguments, list):
            ret = await self.__call__(*arguments)
        elif isinstance(arguments, dict):
            ret = await self.__call__(**arguments)
        else:
            ret = await self.__call__(arguments)
        if not isinstance(ret, ToolReturnValue):  # type: ignore[reportUnnecessaryIsInstance]
            # let's do not trust the return type of the tool
            ret = ToolError(
                message=f"Invalid return type: {type(ret)}",
                brief="Invalid return type",
            )
        return ret

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolReturnValue:
        """
        @public

        The implementation of the callable tool.
        """
        ...


# Common LLM field-name substitutions that differ from the canonical schema name.
# Maps the *wrong* key the LLM often sends → the *correct* field name.
#
# These are split into category-specific dicts so tools can opt into only the
# aliases relevant to their parameter schema, reducing false-positive repairs.

FIELD_ALIASES_GENERAL: dict[str, str] = {
    # title / description / message
    "content": "title",
    "text": "title",
    "name": "title",
    "desc": "description",
    "detail": "description",
    "label": "description",
    "summary": "description",
    "info": "description",
    "msg": "message",
    "note": "message",
    # reason / cause
    "cause": "reason",
    "explanation": "reason",
    "rationale": "reason",
    "justification": "reason",
    "purpose": "reason",
    # prompt / instruction
    "instruction": "prompt",
    "task": "prompt",
    "request": "prompt",
    # step / stage
    "stage": "step",
    "phase": "step",
    "entry": "step",
    # result / outcome
    "outcome": "result",
    "return": "result",
    "status": "result",
    # action / operation
    "operation": "action",
    "op": "action",
    "verb": "action",
    # brief
    "short": "brief",
    # status / state
    "state": "status",
    # question / query
    "query": "question",
    # options / choices
    "choices": "options",
    "answers": "options",
    # multi_select
    "multiple": "multi_select",
    "allow_multiple": "multi_select",
    # header
    "category": "header",
    "tag": "header",
}

FIELD_ALIASES_FILE: dict[str, str] = {
    # path / file / directory
    "file": "path",
    "filepath": "path",
    "file_path": "path",
    "filename": "path",
    "file_name": "path",
    "dir": "path",
    "directory": "path",
    "folder": "path",
    "location": "path",
    # content / data
    "data": "content",
    "body": "content",
    "source": "content",
    "value": "content",
    # mode / method
    "method": "mode",
    "write_mode": "mode",
    # line_offset
    "offset": "line_offset",
    "start": "line_offset",
    "start_line": "line_offset",
    # n_lines
    "lines": "n_lines",
    "count": "n_lines",
    "num_lines": "n_lines",
    # max_char
    "chars": "max_char",
    "max_chars": "max_char",
    "char_limit": "max_char",
    # char_offset
    "byte_offset": "char_offset",
    "position": "char_offset",
    # include_dirs
    "dirs": "include_dirs",
    "directories": "include_dirs",
    # include_ignored
    "ignored": "include_ignored",
    "gitignore": "include_ignored",
    "hidden": "include_ignored",
    # case_insensitive
    "ignore_case": "case_insensitive",
    "insensitive": "case_insensitive",
    # head_limit
    "max": "head_limit",
    "max_results": "head_limit",
    "limit": "head_limit",
    # multiline
    "multi_line": "multiline",
    # output_mode
    "format": "output_mode",
    # type (file type)
    "file_type": "type",
    "kind": "type",
    # pattern / regex
    "regex": "pattern",
    "expr": "pattern",
    "expression": "pattern",
    "match": "pattern",
    # edit / changes
    "changes": "edit",
    "modifications": "edit",
    "patch": "edit",
    # edit nested fields
    "original": "old",
    "find": "old",
    "target": "old",
    "old_str": "old",
    "old_string": "old",
    "old_content": "old",
    "replace_with": "new",
    "to": "new",
    "new_str": "new",
    "new_string": "new",
    "new_content": "new",
    "all": "replace_all",
    # output_path / destination
    "out": "output_path",
    "output": "output_path",
    "destination": "output_path",
    "dest": "output_path",
    # files / paths
    "paths": "files",
    "file_list": "files",
    # glob / filter
    "filter": "glob",
    "file_pattern": "glob",
    # directory (Glob)
    "path": "directory",
    # pages (read)
    "page": "pages",
    "page_range": "pages",
    # edit (edit)
    "edits": "edit",
    # line_number (Grep)
    "show_line_numbers": "line_number",
    "line_numbers": "line_number",
    "show_numbers": "line_number",
    "number": "line_number",
    # ignore_case (Grep) - reciprocal
    "case_insensitive": "ignore_case",
    # before_context / after_context / context (Grep -B/-A/-C)
    "before": "before_context",
    "after": "after_context",
    "around": "context",
    # include_content (Search web)
    "include": "include_content",
    "show_content": "include_content",
    "with_content": "include_content",
    # dest_path (Search/Skill)
    "destination_path": "dest_path",
}

FIELD_ALIASES_SHELL: dict[str, str] = {
    # command / code / script
    "cmd": "command",
    "script": "command",
    "shell_command": "command",
    "program": "code",
    "snippet": "code",
    # timeout / wait
    "wait": "timeout",
    "delay": "timeout",
    "time_limit": "timeout",
    "duration": "timeout",
    # run_in_background
    "background": "run_in_background",
    "async": "run_in_background",
    "detach": "run_in_background",
    "bg": "run_in_background",
    # args / arguments
    "arguments": "args",
    "params": "args",
    "arg": "args",
    "parameters": "args",
    # cwd
    "working_dir": "cwd",
    "work_dir": "cwd",
    # env / variables
    "environment": "env",
    "vars": "env",
    "variables": "env",
    # command
    "code": "command",
}

FIELD_ALIASES_WEB: dict[str, str] = {
    # url / link
    "link": "url",
    "href": "url",
    "address": "url",
    "uri": "url",
    "site": "url",
    # query / search
    "q": "query",
    "search": "query",
    "keyword": "query",
    "keywords": "query",
    "term": "query",
    "question": "query",
}

FIELD_ALIASES_TASK: dict[str, str] = {
    # task_id
    "id": "task_id",
    "job_id": "task_id",
    # block / sync
    "blocking": "block",
    "sync": "block",
    # kill / stop
    "force": "kill",
    "terminate": "kill",
    "stop": "kill",
    # limit (TaskList)
    "head_limit": "limit",
    # reason (TaskStop)
    "message": "reason",
}

FIELD_ALIASES_INPUT: dict[str, str] = {
    # text / input / stdin
    "input": "text",
    "stdin": "text",
}

FIELD_ALIASES_SEARCH: dict[str, str] = {
    # k / n / top_k
    "n": "k",
    "top_k": "k",
    "num": "k",
    # questions
    "queries": "questions",
    "msgs": "questions",
}

FIELD_ALIASES_MODEL: dict[str, str] = {
    # model / llm
    "llm": "model",
    "model_name": "model",
    # resume / session
    "continue": "resume",
    "agent_id": "resume",
}

FIELD_ALIASES_TODO: dict[str, str] = {
    # todos
    "items": "todos",
    "list": "todos",
    "tasks": "todos",
    "entries": "todos",
    "replace": "mode",
    "override": "mode",
    "overwrite": "mode",
    "append": "mode",
    "merge": "mode",
    "update": "mode",
}

FIELD_ALIASES_ACTIVE: dict[str, str] = {
    # active_only
    "active": "active_only",
    "running": "active_only",
    "current": "active_only",
}

FIELD_ALIASES_SUBAGENT: dict[str, str] = {
    # subagent_type
    "agent_type": "subagent_type",
    # session_id
    "session": "session_id",
    "sid": "session_id",
    # close_session
    "close": "close_session",
    "end": "close_session",
    "end_session": "close_session",
    "finish": "close_session",
    "done": "close_session",
    # return_history
    "history": "return_history",
    "full_history": "return_history",
    # response
    "reply": "response",
    "answer": "response",
    # context (AskParentParams)
    "ctx": "context",
}

# ── Common LLM value substitutions for enum/literal fields ──────────
# Maps the *wrong* value the LLM often sends → the *correct* canonical value.
# Applied during _repair_dict_for_model for string fields whose annotation
# is a Literal type, as a first-pass before fuzzy matching.


# --- todo_write status (Literal["pending", "in_progress", "done"]) ---
# source: kimi_cli/tools/todo/__init__.py:46
VALUE_ALIASES_STATUS: dict[str, str] = {
    # "done" synonyms
    "completed": "done",
    "complete": "done",
    "finished": "done",
    "finished.": "done",
    "resolved": "done",
    "closed": "done",
    "success": "done",
    # "in_progress" synonyms
    "started": "in_progress",
    "in-progress": "in_progress",
    "in progress": "in_progress",
    "active": "in_progress",
    "working": "in_progress",
    # "pending" synonyms
    "todo": "pending",
    "to-do": "pending",
    "not_started": "pending",
    "not-started": "pending",
    "waiting": "pending",
    "queued": "pending",
    "open": "pending",
}

# --- Agent store state (Literal["running", "awaiting_response", "completed"]) ---
# source: src/kimix/tools/agent/store.py:31
VALUE_ALIASES_AGENT_STATE: dict[str, str] = {
    # "running" synonyms
    "active": "running",
    "started": "running",
    "in_progress": "running",
    "in-progress": "running",
    # "awaiting_response" synonyms
    "waiting": "awaiting_response",
    "waiting_response": "awaiting_response",
    "awaiting": "awaiting_response",
    "awaiting-reply": "awaiting_response",
    "pending_response": "awaiting_response",
    # "completed" synonyms
    "done": "completed",
    "finished": "completed",
    "complete": "completed",
    "closed": "completed",
}

# --- Chat role (Literal["user", "assistant", "system", "tool", "error"]) ---
# source: src/kimix/tools/agent/store.py:14
VALUE_ALIASES_CHAT_ROLE: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "bot": "assistant",
    "agent": "assistant",
    "model": "assistant",
    "llm": "assistant",
    "sys": "system",
    "tools": "tool",
    "tool_call": "tool",
    "function": "tool",
    "err": "error",
    "exception": "error",
    "fail": "error",
}

# --- Mode: overwrite / append / replace(+force) ---
# sources:
#   kimi_cli/tools/todo/__init__.py (mode Literal["append", "replace", "clear"] + force flag;
#     legacy "overwrite" → "replace", "force_overwrite" → mode='replace' force=True)
#   src/kimix/tools/note/__init__.py:36 (Literal["overwrite", "append"])
#   kimi_cli/tools/file/write.py:73 (Literal["overwrite", "append"])
VALUE_ALIASES_OVERWRITE_MODE: dict[str, str] = {
    # overwrite synonyms
    "replace": "overwrite",
    "override": "overwrite",
    "write": "overwrite",
    "create": "overwrite",
    "new": "overwrite",
    # append synonyms
    "merge": "append",
    "add": "append",
    "update": "append",
    "extend": "append",
    # force_overwrite synonyms
    "force_overwrite": "force_overwrite",
    "force": "force_overwrite",
    "force-overwrite": "force_overwrite",
    "force_override": "force_overwrite",
    "forceoverride": "force_overwrite",
    "forcewrite": "force_overwrite",
}

# --- Match mode (Literal["exact", "fuzzy"]) ---
# sources: kimi_cli/tools/todo/__init__.py:155, kimi_cli/tools/file/replace.py:60
VALUE_ALIASES_MATCH_MODE: dict[str, str] = {
    "exact": "exact",
    "strict": "exact",
    "precise": "exact",
    "literal": "exact",
    "fuzzy": "fuzzy",
    "approximate": "fuzzy",
    "similar": "fuzzy",
    "loose": "fuzzy",
}

# --- Process execution mode (Literal["execute", "send", "interactive"]) ---
# sources:
#   src/kimix/tools/py/__init__.py:51
#   src/kimix/tools/file/bash/bash_tool.py:566
#   src/kimix/tools/file/bash/pwsh_tool.py:193
VALUE_ALIASES_EXEC_MODE: dict[str, str] = {
    # execute synonyms
    "run": "execute",
    "exec": "execute",
    "start": "execute",
    "launch": "execute",
    # send synonyms
    "background": "send",
    "bg": "send",
    "async": "send",
    "detach": "send",
    # interactive synonyms
    "interact": "interactive",
    "repl": "interactive",
    "shell": "interactive",
}

# --- Run mode (Literal["execute", "send"]) ---
# source: src/kimix/tools/file/run.py:89
VALUE_ALIASES_RUN_MODE: dict[str, str] = {
    # execute synonyms
    "run": "execute",
    "exec": "execute",
    "start": "execute",
    # send synonyms
    "background": "send",
    "bg": "send",
    "async": "send",
    "detach": "send",
}

# --- Compact mode (str: auto/retentive/balanced/aggressive/technical) ---
# source: src/kimix/tools/context/__init__.py:54
VALUE_ALIASES_COMPACT_MODE: dict[str, str] = {
    "auto": "auto",
    "automatic": "auto",
    "default": "auto",
    # retentive synonyms
    "retentive": "retentive",
    "retain": "retentive",
    "keep": "retentive",
    "detailed": "retentive",
    "full": "retentive",
    # balanced synonyms
    "balanced": "balanced",
    "balance": "balanced",
    "medium": "balanced",
    "normal": "balanced",
    "moderate": "balanced",
    # aggressive synonyms
    "aggressive": "aggressive",
    "short": "aggressive",
    "minimal": "aggressive",
    "compact": "aggressive",
    "tiny": "aggressive",
    # technical synonyms
    "technical": "technical",
    "tech": "technical",
    "code": "technical",
    "detail": "technical",
}

# --- context_prune mode (Literal["prune", "compact", "strip_reasoning"]) ---
# source: kimi_cli/tools/context_prune.py:38
VALUE_ALIASES_PRUNE_MODE: dict[str, str] = {
    "prune": "prune",
    "trim": "prune",
    "clean": "prune",
    "shrink": "prune",
    "remove": "prune",
    # compact synonyms
    "compact": "compact",
    "compress": "compact",
    "summarize": "compact",
    "reduce": "compact",
    # strip_reasoning synonyms
    "strip_reasoning": "strip_reasoning",
    "strip": "strip_reasoning",
    "reasoning": "strip_reasoning",
    "thinking": "strip_reasoning",
    "strip-thinking": "strip_reasoning",
}

# --- Swarm mode (Literal["fanout", "parallel_sample"]) ---
# source: src/kimix/tools/swarm/__init__.py:70
VALUE_ALIASES_SWARM_MODE: dict[str, str] = {
    "fanout": "fanout",
    "fan": "fanout",
    "fan_out": "fanout",
    "fan-out": "fanout",
    "decompose": "fanout",
    "split": "fanout",
    # parallel_sample synonyms
    "parallel_sample": "parallel_sample",
    "parallel": "parallel_sample",
    "sample": "parallel_sample",
    "best_of_n": "parallel_sample",
    "best-of-n": "parallel_sample",
    "bestofn": "parallel_sample",
    "bonsai": "parallel_sample",
}

# --- Swarm selector (Literal["self_eval", "majority"]) ---
# source: src/kimix/tools/swarm/__init__.py:80
VALUE_ALIASES_SWARM_SELECTOR: dict[str, str] = {
    "self_eval": "self_eval",
    "self": "self_eval",
    "self-eval": "self_eval",
    "selfeval": "self_eval",
    "auto": "self_eval",
    "model": "self_eval",
    # majority synonyms
    "majority": "majority",
    "major": "majority",
    "vote": "majority",
    "voting": "majority",
    "consensus": "majority",
}

# --- history_format (Literal["json", "markdown", "summary"]) ---
# source: src/kimix/tools/agent/__init__.py:59
VALUE_ALIASES_HISTORY_FORMAT: dict[str, str] = {
    "json": "json",
    "raw": "json",
    "dict": "json",
    # markdown synonyms
    "markdown": "markdown",
    "md": "markdown",
    "markup": "markdown",
    "rich": "markdown",
    "formatted": "markdown",
    # summary synonyms
    "summary": "summary",
    "summarize": "summary",
    "summ": "summary",
    "brief": "summary",
    "concise": "summary",
    "short": "summary",
}

# --- job_output action (Literal["get", "list", "kill"]) ---
# source: src/kimix/tools/background/__init__.py:23
VALUE_ALIASES_TASK_ACTION: dict[str, str] = {
    "get": "get",
    "fetch": "get",
    "retrieve": "get",
    "show": "get",
    "output": "get",
    "read": "get",
    # list synonyms
    "list": "list",
    "ls": "list",
    "all": "list",
    "tasks": "list",
    "status": "list",
    # kill synonyms
    "kill": "kill",
    "stop": "kill",
    "terminate": "kill",
    "force": "kill",
    "cancel": "kill",
    "end": "kill",
}

# --- HTTP method (Literal["GET", "POST"]) ---
# source: kimi_cli/tools/web/fetch.py:23
VALUE_ALIASES_HTTP_METHOD: dict[str, str] = {
    "GET": "GET",
    "get": "GET",
    "fetch": "GET",
    "download": "GET",
    # POST synonyms
    "POST": "POST",
    "post": "POST",
    "send": "POST",
    "submit": "POST",
    "upload": "POST",
    "put": "POST",
    "patch": "POST",
}

# --- Grep output_mode (Literal["files_with_matches", "count_matches", "content"]) ---
# source: kimi_cli/tools/file/grep_local.py:76
VALUE_ALIASES_GREP_OUTPUT: dict[str, str] = {
    "files_with_matches": "files_with_matches",
    "files": "files_with_matches",
    "file": "files_with_matches",
    "filenames": "files_with_matches",
    # count_matches synonyms
    "count_matches": "count_matches",
    "count": "count_matches",
    "matches": "count_matches",
    "stats": "count_matches",
    # content synonyms
    "content": "content",
    "body": "content",
    "full": "content",
    "text": "content",
    "lines": "content",
    "detail": "content",
}

# --- HashLine op (Literal["replace", "append", "prepend", "delete"]) ---
# source: kimi_cli/tools/file/hash_line.py:126-145
VALUE_ALIASES_HASHLINE_OP: dict[str, str] = {
    "replace": "replace",
    "change": "replace",
    "edit": "replace",
    "set": "replace",
    "update": "replace",
    "modify": "replace",
    # append synonyms
    "append": "append",
    "add": "append",
    "insert_after": "append",
    # prepend synonyms
    "prepend": "prepend",
    "insert": "prepend",
    "before": "prepend",
    "insert_before": "prepend",
    # delete synonyms
    "delete": "delete",
    "remove": "delete",
    "del": "delete",
    "erase": "delete",
    "drop": "delete",
}

# --- read_media kind (Literal["untouched", "downsampled", "crop", "full"]) ---
# source: kimi_cli/tools/file/read_media.py:232
VALUE_ALIASES_MEDIA_KIND: dict[str, str] = {
    "untouched": "untouched",
    "original": "untouched",
    "raw": "untouched",
    "passthrough": "untouched",
    "none": "untouched",
    # downsampled synonyms
    "downsampled": "downsampled",
    "downsample": "downsampled",
    "down": "downsampled",
    "small": "downsampled",
    "reduced": "downsampled",
    "lowres": "downsampled",
    # crop synonyms
    "crop": "crop",
    "cropped": "crop",
    "trim": "crop",
    "region": "crop",
    # full synonyms
    "full": "full",
    "complete": "full",
    "entire": "full",
    "highres": "full",
    "maximum": "full",
}

# --- read_media file type (Literal["image", "video"]) ---
# source: kimi_cli/tools/file/read_media.py:250
VALUE_ALIASES_MEDIA_TYPE: dict[str, str] = {
    "image": "image",
    "img": "image",
    "picture": "image",
    "photo": "image",
    # video synonyms
    "video": "video",
    "vid": "video",
    "movie": "video",
    "clip": "video",
}

# --- file utils kind (Literal["text", "image", "video", "unknown"]) ---
# source: kimi_cli/tools/file/utils.py:179
VALUE_ALIASES_FILE_KIND: dict[str, str] = {
    "text": "text",
    "txt": "text",
    "string": "text",
    # image synonyms
    "image": "image",
    "img": "image",
    "picture": "image",
    # video synonyms
    "video": "video",
    "vid": "video",
    "movie": "video",
    # unknown synonyms
    "unknown": "unknown",
    "other": "unknown",
    "binary": "unknown",
    "data": "unknown",
}

# --- boolean value aliases (general-purpose) ---
VALUE_ALIASES_BOOL_STR: dict[str, str] = {
    "true": "true",
    "True": "true",
    "yes": "true",
    "on": "true",
    "1": "true",
    "enable": "true",
    "enabled": "true",
    "false": "false",
    "False": "false",
    "no": "false",
    "off": "false",
    "0": "false",
    "disable": "false",
    "disabled": "false",
}

# Merged common value aliases for general use
# NOTE: Put STATUS last to override overlapping keys from AGENT_STATE
# (e.g. "complete" → "done" for status, not "completed" for agent state).
# The _fuzzy_match_literal_value function validates candidates against the
# actual Literal options, so the correct option is selected per context.
_COMMON_VALUE_ALIASES: dict[str, str] = {
    **VALUE_ALIASES_AGENT_STATE,
    **VALUE_ALIASES_BOOL_STR,
    **VALUE_ALIASES_CHAT_ROLE,
    **VALUE_ALIASES_COMPACT_MODE,
    **VALUE_ALIASES_EXEC_MODE,
    **VALUE_ALIASES_FILE_KIND,
    **VALUE_ALIASES_GREP_OUTPUT,
    **VALUE_ALIASES_HASHLINE_OP,
    **VALUE_ALIASES_HISTORY_FORMAT,
    **VALUE_ALIASES_HTTP_METHOD,
    **VALUE_ALIASES_MATCH_MODE,
    **VALUE_ALIASES_MEDIA_KIND,
    **VALUE_ALIASES_MEDIA_TYPE,
    **VALUE_ALIASES_OVERWRITE_MODE,
    **VALUE_ALIASES_PRUNE_MODE,
    **VALUE_ALIASES_RUN_MODE,
    **VALUE_ALIASES_SWARM_MODE,
    **VALUE_ALIASES_SWARM_SELECTOR,
    **VALUE_ALIASES_TASK_ACTION,
    **VALUE_ALIASES_STATUS,  # last so it overrides overlapping generic keys
}


# Merge all categories into the common set for backward compatibility.
_COMMON_FIELD_ALIASES: dict[str, str] = {
    **FIELD_ALIASES_GENERAL,
    **FIELD_ALIASES_FILE,
    **FIELD_ALIASES_SHELL,
    **FIELD_ALIASES_WEB,
    **FIELD_ALIASES_TASK,
    **FIELD_ALIASES_INPUT,
    **FIELD_ALIASES_SEARCH,
    **FIELD_ALIASES_MODEL,
    **FIELD_ALIASES_TODO,
    **FIELD_ALIASES_ACTIVE,
    **FIELD_ALIASES_SUBAGENT,
}

# Precomputed reverse lookup for the common aliases to avoid rebuilding it
# on every call to ``_repair_dict_for_model``.
_REVERSE_ALIASES: dict[str, list[str]] = {}
for _bad_key, _good_key in _COMMON_FIELD_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_good_key, []).append(_bad_key)


def alias_note(*names: str, word: bool = True) -> str:
    """Return the \"Accepts `a` or `b`\" prose for a parameter with aliases.

    ``names`` are the accepted spellings in canonical-first order, e.g.
    ``alias_note("cmd", "command")``.  With three or more names the last two
    are joined with \"or\" (e.g. ``code | source_code | file``).  ``word=True``
    appends \" parameter\" (used in tool descriptions); field-level param
    descriptions historically omit it (``word=False``).  This is the single
    implementation shared by every tool description (see
    ``kimix.tools.prompt_common.accepts_alias_text``, which delegates here).
    """
    if len(names) == 2:
        suffix = " parameter." if word else "."
        return f"Accepts `{names[0]}` or `{names[1]}`{suffix}"
    if len(names) > 2:
        head = ", ".join(f"`{n}`" for n in names[:-1])
        suffix = " parameter." if word else "."
        return f"Accepts {head} or `{names[-1]}`{suffix}"
    raise ValueError("alias_note requires at least two names")


def _get_base_model_type(annotation: Any) -> type[BaseModel] | None:
    """Extract a BaseModel subclass from a type annotation, unwrapping generics."""
    origin = typing.get_origin(annotation)
    if origin is not None:
        for arg in typing.get_args(annotation):
            result = _get_base_model_type(arg)
            if result is not None:
                return result
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _maybe_parse_json_string(value: Any) -> Any:
    """If *value* is a string containing a JSON object/array, return it parsed.

    Returns the original value when it is not a string, when parsing fails,
    or when the parsed value is not a structural JSON object/array, so that
    normal Pydantic validation can report the error.
    """
    if not isinstance(value, str):
        return value
    try:
        parsed = loads_relaxed(value)
    except Exception:
        return value
    if isinstance(parsed, (dict, list)):
        return parsed
    return value


def _clamp_numeric_value(value: Any, finfo: Any) -> Any | None:
    """Clamp a numeric value to the field's min/max constraints.

    Returns the clamped value if it was out of bounds, otherwise None.
    """
    constraints = _extract_constraints(finfo)
    if constraints is None:
        return None
    return _apply_constraints(value, constraints)


def _extract_constraints(finfo: Any) -> tuple[float | int | None, float | int | None, bool, bool] | None:
    """Extract numeric constraints from a FieldInfo's metadata.

    Returns (min_val, max_val, min_inclusive, max_inclusive) if any constraints exist,
    otherwise None.
    """
    min_val: float | int | None = None
    max_val: float | int | None = None
    min_inclusive = True
    max_inclusive = True
    has_any = False

    for m in getattr(finfo, "metadata", ()):
        if hasattr(m, "ge"):
            min_val = m.ge
            min_inclusive = True
            has_any = True
        elif hasattr(m, "gt"):
            min_val = m.gt
            min_inclusive = False
            has_any = True
        elif hasattr(m, "le"):
            max_val = m.le
            max_inclusive = True
            has_any = True
        elif hasattr(m, "lt"):
            max_val = m.lt
            max_inclusive = False
            has_any = True

    return (min_val, max_val, min_inclusive, max_inclusive) if has_any else None


def _apply_constraints(value: Any, constraints: tuple[float | int | None, float | int | None, bool, bool]) -> Any | None:
    """Clamp a numeric value to pre-extracted constraints.

    Returns the clamped value if it was out of bounds, otherwise None.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None

    min_val, max_val, min_inclusive, max_inclusive = constraints
    clamped = value

    if min_val is not None:
        if min_inclusive and clamped < min_val:
            clamped = min_val
        elif not min_inclusive and clamped <= min_val:
            if isinstance(clamped, int) and isinstance(min_val, int):
                clamped = min_val + 1
            else:
                clamped = min_val

    if max_val is not None:
        if max_inclusive and clamped > max_val:
            clamped = max_val
        elif not max_inclusive and clamped >= max_val:
            if isinstance(clamped, int) and isinstance(max_val, int):
                clamped = max_val - 1
            else:
                clamped = max_val

    return clamped if clamped != value else None


def _resolve_literal_options(annotation: Any) -> tuple[str, ...] | None:
    """Extract string literal values from a Literal type annotation.

    Returns a tuple of allowed string values if the annotation is a
    ``Literal["a", "b", ...]``, otherwise returns ``None``.
    Supports generic originals like ``typing.Literal``, plain
    ``Literal`` (Python 3.8+), and nested Optional/Union wrappers.
    """
    origin = typing.get_origin(annotation)
    if origin is None:
        return None
    origin_name = getattr(origin, "__name__", getattr(origin, "_name", ""))

    # Handle Literal directly
    if origin_name == "Literal":
        args = typing.get_args(annotation)
        if not args:
            return None
        # Only return string literals
        options = tuple(a for a in args if isinstance(a, str))
        return options if options else None

    # Handle Union / Optional wrappers: unwrap and check inner type
    if origin_name in ("Union", "Optional"):
        args = typing.get_args(annotation)
        for arg in args:
            if arg is not type(None):
                result = _resolve_literal_options(arg)
                if result is not None:
                    return result

    return None


@lru_cache(maxsize=512)
def _cached_model_field_info(model: type[BaseModel]) -> tuple[
    dict[str, str],
    list[tuple[
        str,
        type[BaseModel] | None,
        tuple[float | int | None, float | int | None, bool, bool] | None,
        bool,
        type | None,
        tuple[str, ...] | None,  # literal_options (added for value-level fuzzy matching)
    ]],
    bool,
    frozenset[str],
]:
    """Precompute known key mapping and field metadata for a model.

    Returns ``(known, field_meta, extra_forbid, valid_names)`` where
    ``field_meta`` tuples contain:
    ``(field_name, nested_model, constraints, is_list, scalar_type, literal_options)``.
    """
    known: dict[str, str] = {}
    field_meta: list[tuple[
        str,
        type[BaseModel] | None,
        tuple[float | int | None, float | int | None, bool, bool] | None,
        bool,
        type | None,
        tuple[str, ...] | None,
    ]] = []
    extra_forbid = model.model_config.get("extra") == "forbid"

    for fname, finfo in model.model_fields.items():
        known[fname] = fname
        if finfo.alias and finfo.alias != fname:
            known[finfo.alias] = fname
        val_alias = getattr(finfo, "validation_alias", None)
        if isinstance(val_alias, str) and val_alias != fname:
            known[val_alias] = fname

        nested = _get_base_model_type(finfo.annotation)
        constraints = _extract_constraints(finfo)
        origin = typing.get_origin(finfo.annotation)
        origin_name = getattr(origin, "__name__", "") if origin is not None else ""
        is_list = origin_name == "list"
        scalar_type = _resolve_scalar_type(finfo.annotation)
        literal_options = _resolve_literal_options(finfo.annotation)
        field_meta.append((fname, nested, constraints, is_list, scalar_type, literal_options))

    return known, field_meta, extra_forbid, frozenset(model.model_fields.keys())


@lru_cache(maxsize=65536)
def _sequence_ratio(a: str, b: str) -> float:
    """Cached SequenceMatcher ratio for string pair (a, b)."""
    return fuzz.ratio(a, b) / 100.0


def _fuzzy_match_keys(
    missing: set[str],
    available: set[str],
    cutoff_short: float = 0.80,
    cutoff_long: float = 0.75,
    min_length: int = 4,
) -> dict[str, str]:
    """Fuzzy-map missing field names to still-available unmapped keys.

    The matching is case-insensitive so that plausible LLM casing differences
    (e.g. ``base_url`` vs ``base_URL``) do not suppress an otherwise strong
    match. Each available key is consumed at most once, and the strongest
    matches are applied first.

    Returns a mapping ``missing_field -> original_available_key``.
    """
    if not missing or not available:
        return {}

    # Build lowercase lookup so we can match case-insensitively while still
    # returning the original key.
    available_lower: dict[str, str] = {}
    for key in available:
        lower = key.lower()
        available_lower.setdefault(lower, key)
    available_list = list(available_lower.keys())

    candidates: list[tuple[float, str, str]] = []
    for missing_field in missing:
        if len(missing_field) < min_length:
            continue
        m_lower = missing_field.lower()
        cutoff = cutoff_long if len(missing_field) >= 8 else cutoff_short
        # Pre-filter candidates: skip items whose first character doesn't match,
        # which eliminates most candidates early without expensive SequenceMatcher.
        best_score = cutoff - 1e-9
        best_match: str | None = None
        if m_lower:
            first_char = m_lower[0]
            for candidate in available_list:
                if not candidate or candidate[0] != first_char:
                    continue
                # Use cached ratio — repeated comparisons across iterations
                # hit the LRU cache instead of recomputing.
                ratio = _sequence_ratio(m_lower, candidate)
                if ratio > best_score:
                    best_score = ratio
                    best_match = candidate

        if best_match is None or best_score < cutoff:
            continue
        matched_key = available_lower[best_match]
        if len(matched_key) < min_length:
            continue
        candidates.append((best_score, missing_field, matched_key))

    # Apply strongest matches first; each unmapped key can only be used once.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used: set[str] = set()
    result: dict[str, str] = {}
    for _ratio, missing_field, matched_key in candidates:
        if matched_key in used:
            continue
        used.add(matched_key)
        result[missing_field] = matched_key
    return result


@lru_cache(maxsize=2048)
def _fuzzy_match_literal_value(
    value: str,
    options: tuple[str, ...],
) -> str | None:
    """Attempt to fuzzy-match a string value to a set of Literal options.

    1. Try exact match (case-insensitive, separator-normalized).
    2. Try the ``_COMMON_VALUE_ALIASES`` map.
    3. Try rapidfuzz fuzzy matching against each option.

    Returns the matched canonical option, or ``None`` if no match found.
    """
    from kosong.tooling import _COMMON_VALUE_ALIASES

    # Normalize: strip, lowercase, normalize separators.
    norm_val = value.strip().lower().replace("-", "_").replace(" ", "_")

    # 1. Direct match (case-insensitive, separator-normalized) against options.
    for opt in options:
        norm_opt = opt.lower().replace("-", "_").replace(" ", "_")
        if norm_val == norm_opt:
            return opt

    # 2. Try the value aliases map.
    if norm_val in _COMMON_VALUE_ALIASES:
        candidate = _COMMON_VALUE_ALIASES[norm_val]
        # Verify the alias target is actually one of the valid options.
        if candidate in options or candidate.lower() in (o.lower() for o in options):
            return candidate

    # Also try the raw (un-normalized) value in the alias map.
    raw_val = value.strip()
    if raw_val in _COMMON_VALUE_ALIASES:
        candidate = _COMMON_VALUE_ALIASES[raw_val]
        if candidate in options or candidate.lower() in (o.lower() for o in options):
            return candidate

    # 3. Fuzzy match via rapidfuzz.
    best_score = 0.60  # cutoff
    best_match: str | None = None
    for opt in options:
        # Use cached ratio for repeated comparisons.
        ratio = _sequence_ratio(norm_val, opt.lower().replace("-", "_").replace(" ", "_"))
        if ratio > best_score:
            best_score = ratio
            best_match = opt

    return best_match


@lru_cache(maxsize=2048)
def normalize_tool_name(name: str) -> str:
    """Canonical comparison key for a tool name.

    Removes ``-`` and ``_`` separators and casefolds, so CamelCase, snake_case,
    kebab-case and SCREAMING_SNAKE_CASE spellings of the same logical name
    collapse to a single string (e.g. ``write_file`` / ``write-file`` /
    ``WRITE_FILE`` / ``WriteFile`` all -> ``writefile``).
    """
    return name.replace("-", "").replace("_", "").casefold()


@lru_cache(maxsize=256)
def _prepared_candidates(
    valid_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Precompute ``(lowercase, normalized)`` forms for a fixed candidate set.

    The candidate tool set is normally stable across calls, so caching keyed on
    the candidate tuple lets the per-candidate ``str.lower()`` /
    ``normalize_tool_name`` work run once per tool set instead of on every call.
    """
    lowers = tuple(c.lower() for c in valid_names)
    norms = tuple(normalize_tool_name(c) for c in valid_names)
    return lowers, norms


def fuzzy_match_tool_name(
    tool_name: str,
    valid_names: Iterable[str],
    n: int = 3,
    cutoff: float = 0.5,
    min_length: int = 3,
) -> list[str]:
    """Return up to *n* likely-intended real tool names for *tool_name*.

    Resolution order (highest confidence first):
      1. Case-insensitive exact match (e.g. ``writefile`` -> ``WriteFile``).
      2. Separator-insensitive normalized exact match, covering snake_case,
         kebab-case and SCREAMING_SNAKE_CASE spellings of a CamelCase tool.
      3. Fuzzy similarity on the normalized forms (handles separators *and*
         typos), keeping candidates with ratio >= *cutoff*.

    Names shorter than *min_length* are skipped to avoid false positives.
    Steps 1-2 are high confidence and returned regardless of *cutoff*.
    Results are deterministic: ties are broken by the raw (case-insensitive)
    similarity ratio and then by candidate name, and the original candidate
    strings are returned.
    """
    if len(tool_name) < min_length:
        return []

    valid_tuple = tuple(valid_names)
    lower = tool_name.lower()

    # Lowercase and normalized forms of the (stable) candidate set are cached
    # per tool set, so the per-candidate work is not repeated on every call.
    lowers, norms = _prepared_candidates(valid_tuple)

    # 1. Case-insensitive exact match (preserves prior behavior).
    ci_exact = [c for c, cl in zip(valid_tuple, lowers) if cl == lower]
    if ci_exact:
        return ci_exact

    # 2. Separator-insensitive normalized exact match
    #    (snake/kebab/SCREAMING/mixed -> CamelCase).
    norm = normalize_tool_name(tool_name)
    norm_exact = [(c, cl) for c, cl, cn in zip(valid_tuple, lowers, norms) if cn == norm]
    if norm_exact:
        # Deterministic order for the rare multi-candidate (collision) case.
        norm_exact.sort(key=lambda t: (-_sequence_ratio(lower, t[1]), t[0]))
        return [c for c, _ in norm_exact[:n]]

    # 3. Normalized fuzzy match (separator + typo tolerant). ``_sequence_ratio``
    #    is memoized, so repeated (query, candidate) pairs collapse to a cache
    #    hit across calls.
    scored: list[tuple[float, float, str]] = []
    for c, cl, cn in zip(valid_tuple, lowers, norms):
        ratio = _sequence_ratio(norm, cn)
        if ratio >= cutoff:
            scored.append((ratio, _sequence_ratio(lower, cl), c))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [c for _, _, c in scored[:n]]


@dataclass(frozen=True, slots=True)
class ToolNameResolution:
    """The result of resolving a (possibly mis-formatted) tool name.

    ``name`` is the resolved real tool name, or ``None`` when no tool matched.
    ``corrected`` is ``True`` when the resolved name differs from the requested
    ``original`` (i.e. an auto-correction happened). ``suggestions`` is only
    populated when ``name`` is ``None``.
    """

    name: str | None
    original: str
    corrected: bool = False
    suggestions: list[str] = field(default_factory=list)


def resolve_tool_name(
    name: str,
    valid_names: Iterable[str],
    *,
    auto_correct_cutoff: float = 0.75,
    suggest_cutoff: float = 0.5,
    n_suggestions: int = 3,
    min_length: int = 3,
    redirects: dict[str, str] | None = None,
) -> ToolNameResolution:
    """Resolve a possibly mis-formatted tool name to a real tool.

    Pure and side-effect free. Resolution order:
      1. Exact match -> returned unchanged (``corrected=False``).
      2. Redirect map lookup via normalized key (if *redirects* provided).
      3. High-confidence fuzzy match at or above *auto_correct_cutoff* ->
         auto-corrected (``corrected=True``).
      4. Otherwise ``name`` is ``None`` and *suggestions* (matches at or above
         *suggest_cutoff*) are populated for an error message.

    Names shorter than *min_length* are never matched.
    """
    # Materialize once as a tuple: accepts dict.keys()/generators and lets the
    # inner fuzzy_match_tool_name call reuse the cached per-candidate prep.
    valid_tuple = tuple(valid_names)
    if name in valid_tuple:
        return ToolNameResolution(name=name, original=name)

    # Check the redirect map (pre-normalized keys) before fuzzy matching.
    if redirects:
        norm_name = normalize_tool_name(name)
        redirected = redirects.get(norm_name)
        if redirected is not None and redirected in valid_tuple:
            return ToolNameResolution(name=redirected, original=name, corrected=True)

    if auto_correct_cutoff >= suggest_cutoff:
        # Fast path (the normal regime): a single scan at the lower suggestion
        # cutoff yields the ranked candidates, and the auto-correct decision is
        # derived from the top candidate instead of a second full scan. This is
        # behaviour-identical to the two-call form because the fuzzy ranking is
        # monotonic in *cutoff* and the exact-match steps ignore *cutoff*.
        matches = fuzzy_match_tool_name(
            name, valid_tuple, n=n_suggestions, cutoff=suggest_cutoff, min_length=min_length
        )
        if not matches:
            return ToolNameResolution(name=None, original=name)
        top = matches[0]
        if (
            top.lower() == name.lower()
            or normalize_tool_name(top) == normalize_tool_name(name)
            or _sequence_ratio(normalize_tool_name(name), normalize_tool_name(top))
            >= auto_correct_cutoff
        ):
            return ToolNameResolution(name=top, original=name, corrected=True)
        return ToolNameResolution(name=None, original=name, suggestions=matches)

    # General fallback for the degenerate case where the auto-correct cutoff is
    # below the suggestion cutoff: keep the exact two-scan semantics.
    auto = fuzzy_match_tool_name(
        name, valid_tuple, n=1, cutoff=auto_correct_cutoff, min_length=min_length
    )
    if auto:
        return ToolNameResolution(name=auto[0], original=name, corrected=True)
    suggestions = fuzzy_match_tool_name(
        name, valid_tuple, n=n_suggestions, cutoff=suggest_cutoff, min_length=min_length
    )
    return ToolNameResolution(name=None, original=name, suggestions=suggestions)


# ──────────────────────────────────────────────────────────────────────
# Tool Name Hallucination Redirect Map
# ──────────────────────────────────────────────────────────────────────
#
# When an LLM backend hallucinates a tool name (e.g. ``AppendFile`` instead
# of ``WriteFile``), this map provides a hardcoded redirect to the real name.
# The lookup uses ``normalize_tool_name()`` so case, hyphen and underscore
# variations all resolve correctly.
#
# Platform-specific overrides (e.g. ``Bash`` -> ``Powershell`` on Windows)
# are applied in ``KimiToolset.handle()``.

TOOL_NAME_REDIRECTS: dict[str, str] = {
    # ── WriteFile hallucinations ──
    "AppendFile": "write",
    "CreateFile": "write",
    "SaveFile": "write",
    "NewFile": "write",
    "PutFile": "write",
    "TouchFile": "write",
    "OverwriteFile": "write",
    "UpdateFile": "write",
    "Write": "write",
    "WriteText": "write",
    "WriteContent": "write",
    "FileWrite": "write",
    "OutputFile": "write",
    "DumpFile": "write",
    "StoreFile": "write",
    "PersistFile": "write",
    "Create": "write",
    "Save": "write",
    # ── EditFile hallucinations ──
    "ReplaceFile": "edit",
    "ModifyFile": "edit",
    "PatchFile": "edit",
    "ChangeFile": "edit",
    "SubFile": "edit",
    "SedFile": "edit",
    "Edit": "edit",
    "FileEdit": "edit",
    "ReplaceInFile": "edit",
    "FindReplace": "edit",
    "SearchReplace": "edit",
    "ApplyPatch": "edit",
    "TextReplace": "edit",
    "StringReplace": "edit",
    "ReplaceText": "edit",
    "UpdateFileContent": "edit",
    "AlterFile": "edit",
    "AdjustFile": "edit",
    "RewriteFile": "edit",
    # ── ReadFile hallucinations ──
    "Read": "read",
    # Legacy class-name redirects (old tool names -> report canonical names).
    "ReadFile": "read",
    "WriteFile": "write",
    "EditFile": "edit",
    "Glob": "glob",
    "Grep": "grep",
    "ReadMediaFile": "read_image",
    "SearchWeb": "web_search",
    "Agent": "subagent",
    "AskAgent": "send_message",
    "AgentList": "list_agents",
    "AgentClose": "interrupt_agent",
    "TaskOutput": "job_output",
    "TodoList": "todo_write",
    "AgentSwarm": "workflow",
    "Cat": "read",
    "ViewFile": "read",
    "ShowFile": "read",
    "DisplayFile": "read",
    "OpenFile": "read",
    "GetFile": "read",
    "LoadFile": "read",
    "ReadText": "read",
    "FileRead": "read",
    "PrintFile": "read",
    "Dump": "read",
    "Head": "read",
    "Tail": "read",
    "Preview": "read",
    "Inspect": "read",
    "Peek": "read",
    "ReadLines": "read",
    # ── Glob hallucinations ──
    "ListFiles": "glob",
    "FindFiles": "glob",
    "SearchFiles": "glob",
    "FileGlob": "glob",
    "Ls": "glob",
    "Dir": "glob",
    "Walk": "glob",
    "FileSearch": "glob",
    "FileList": "glob",
    "ListDir": "glob",
    "ListDirectory": "glob",
    "DirectoryList": "glob",
    "GlobFiles": "glob",
    "MatchFiles": "glob",
    "Wildcard": "glob",
    # ── Grep hallucinations ──
    "Search": "grep",
    "Find": "grep",
    "RipGrep": "grep",
    "Rg": "grep",
    "TextSearch": "grep",
    "ContentSearch": "grep",
    "GrepText": "grep",
    "SearchInFiles": "grep",
    "FindInFiles": "grep",
    "FileGrep": "grep",
    "PatternSearch": "grep",
    "RegexSearch": "grep",
    "MatchPattern": "grep",
    "Lookup": "grep",
    # ── TodoList hallucinations ──
    "TaskList": "todo_write",
    "Todo": "todo_write",
    "Todos": "todo_write",
    "Tasks": "todo_write",
    "TaskManager": "todo_write",
    "TaskPlan": "todo_write",
    "Plan": "todo_write",
    "Checklist": "todo_write",
    # "TodoList" redirects above (legacy class name)
    "TaskTracker": "todo_write",
    "Progress": "todo_write",
    # ── fetch_url hallucinations ──
    "Fetch": "fetch_url",
    "HttpGet": "fetch_url",
    "GetUrl": "fetch_url",
    "Download": "fetch_url",
    "WebFetch": "fetch_url",
    "FetchPage": "fetch_url",
    "OpenUrl": "fetch_url",
    "Browse": "fetch_url",
    "Visit": "fetch_url",
    "ReadUrl": "fetch_url",
    "UrlFetch": "fetch_url",
    "HttpRequest": "fetch_url",
    "Curl": "fetch_url",
    "Wget": "fetch_url",
    # ── SearchWeb hallucinations ──
    # "SearchWeb" redirects above (legacy class name)
    "WebSearch": "web_search",
    "GoogleSearch": "web_search",
    "SearchInternet": "web_search",
    "InternetSearch": "web_search",
    "SearchOnline": "web_search",
    "OnlineSearch": "web_search",
    "WebQuery": "web_search",
    "QueryWeb": "web_search",
    "SearchEngine": "web_search",
    "Search": "web_search",
    "LookupWeb": "web_search",
    "FindOnline": "web_search",
    # ── Context management hallucinations ──
    # compact, context_prune, retrieve are canonical
    "PruneContext": "context_prune",
    "CompactContext": "compact",
    "Summarize": "compact",
    "SummarizeContext": "compact",
    "ContextSummary": "compact",
    "ContextTrim": "context_prune",
    "TrimContext": "context_prune",
    "Recall": "retrieve",
    "RetrieveContext": "retrieve",
    "SearchHistory": "retrieve",
    "HistorySearch": "retrieve",
    # ── Agent/Subagent hallucinations ──
    "SubAgent": "subagent",
    "Subagent": "subagent",
    "SpawnAgent": "subagent",
    "StartAgent": "subagent",
    "RunAgent": "subagent",
    "LaunchAgent": "subagent",
    "AgentCreate": "subagent",
    "AgentSpawn": "subagent",
    "AgentRun": "subagent",
    "CreateAgent": "subagent",
    "NewAgent": "subagent",
    "Agents": "list_agents",
    # "AgentList" redirects above (legacy class name)
    "ListAgents": "list_agents",
    "ActiveAgents": "list_agents",
    "AgentStatus": "list_agents",
    "CloseAgent": "interrupt_agent",
    # "AgentClose" redirects above (legacy class name)
    "StopAgent": "interrupt_agent",
    "KillAgent": "interrupt_agent",
    "EndAgent": "interrupt_agent",
    "TerminateAgent": "interrupt_agent",
    "AgentStop": "interrupt_agent",
    "AgentKill": "interrupt_agent",
    # "AgentSwarm" is a canonical name — no redirect needed
    "Swarm": "workflow",
    "AgentTeam": "workflow",
    "MultiAgent": "workflow",
    # ── Background task hallucinations ──
    "ListTasks": "TaskList",
    "Tasks": "TaskList",
    "TaskStatus": "TaskList",
    "GetTaskOutput": "job_output",
    # "TaskOutput" redirects above (legacy class name)
    "TaskResult": "job_output",
    "GetOutput": "job_output",
    "ReadOutput": "job_output",
    "StopTask": "TaskStop",
    # "TaskStop" is a canonical name — no redirect needed
    "KillTask": "TaskStop",
    "CancelTask": "TaskStop",
    "AbortTask": "TaskStop",
    "TerminateTask": "TaskStop",
    "TaskKill": "TaskStop",
    "TaskCancel": "TaskStop",
    # ── Python/Run hallucinations ──
    # "Python" is a canonical name — no redirect needed
    "RunPython": "Python",
    "ExecutePython": "Python",
    "Py": "Python",
    "PythonExec": "Python",
    "PythonRun": "Python",
    "PythonCode": "Python",
    "Eval": "Python",
    "ExecPython": "Python",
    "Run": "Python",
    "Execute": "Python",
    "Exec": "Python",
    # ── Miscellaneous hallucinations ──
    # "ReadMediaFile" redirects above (legacy class name)
    "MediaRead": "read_image",
    "ReadMedia": "read_image",
    "ViewMedia": "read_image",
    "AskUser": "AskUserQuestion",
    # "AskUserQuestion" is a canonical name — no redirect needed
    "UserQuestion": "AskUserQuestion",
    "Ask": "AskUserQuestion",
    "QueryUser": "AskUserQuestion",
    "PromptUser": "AskUserQuestion",
    # "Think" is a canonical name — no redirect needed
    "Reason": "Think",
    "Reflect": "Think",
    "Consider": "Think",
    "Analyze": "Think",
    # "Mkdir" is a canonical name — no redirect needed
    "MakeDir": "Mkdir",
    "CreateDir": "Mkdir",
    "CreateDirectory": "Mkdir",
    # "Rm" is a canonical name — no redirect needed
    "Remove": "Rm",
    "Delete": "Rm",
    "RemoveFile": "Rm",
    "DeleteFile": "Rm",
    "RmDir": "Rm",
    "RemoveDir": "Rm",
    "DeleteDir": "Rm",
    "Unlink": "Rm",
    # "ParserTool" is a canonical name — no redirect needed
    "Parse": "ParserTool",
    "Parser": "ParserTool",
    # "FindStr" is a canonical name — no redirect needed
    "FindString": "FindStr",
    "SearchString": "FindStr",
    # "AskParent" is a canonical name — no redirect needed
    "AskParentAgent": "AskParent",
    "QueryParent": "AskParent",
    "ParentQuery": "AskParent",
    # "HashRead" is a canonical name — no redirect needed
    "HashReadFile": "HashRead",
    "ReadHash": "HashRead",
    "FileHash": "HashRead",
    # "HashEdit" is a canonical name — no redirect needed
    "HashEditFile": "HashEdit",
    "EditHash": "HashEdit",
    # ── Shell command hallucinations (platform-neutral baseline) ──
    # On POSIX, these all redirect to ``Bash`` (the real tool).
    # On Windows, ``Bash`` and all other shell names redirect to ``Powershell``.
    # Platform overrides are applied in ``KimiToolset.handle()``.
    "Shell": "Bash",
    "Terminal": "Bash",
    "Cmd": "Bash",
    "Command": "Bash",
    "Sh": "Bash",
    "Zsh": "Bash",
    "ShellCommand": "Bash",
    "BashCommand": "Bash",
    "RunCommand": "Bash",
    "Powershell": "Bash",
    "PowerShell": "Bash",
    "Pwsh": "Bash",
    "PS": "Bash",
    # "Bash" is a canonical name (real tool on POSIX) — no redirect needed
}

# Pre-normalized version for O(1) lookup after normalize_tool_name().
# Self-mapping entries (raw or normalized) are filtered out — they add no value.
_TOOL_NAME_REDIRECTS_NORMALIZED: dict[str, str] = {
    normalize_tool_name(k): v
    for k, v in TOOL_NAME_REDIRECTS.items()
    if k != v and normalize_tool_name(k) != normalize_tool_name(v)
}


def _score_argument_fit(
    arguments: dict[str, Any],
    params_model: type[BaseModel] | None,
) -> tuple[float, dict[str, Any] | None]:
    """
    Score how well *arguments* fit a tool's parameter schema.

    Returns ``(score 0.0–1.0, repaired_args_or_None)`` where a higher score
    means a better fit.  ``repaired_args`` is only set when the arguments
    could be meaningfully repaired against the model (i.e. at least one field
    matched).

    Score components:
    - Each required field present in arguments (after repair): +0.3 / n_required
    - Each optional field present: +0.1 / n_optional
    - Each field matched via alias/fuzzy: +0.05 bonus
    - Each field matched by exact name: +0.1 bonus
    - Each unmapped key (LLM sent a field no candidate has): -0.15
    - Each type coercion applied: +0.02 bonus (weak signal)

    Clamped to [0.0, 1.0].

    Returns ``(0.0, None)`` when the model is not a BaseModel subclass
    (e.g. legacy CallableTool without typed params).
    """
    if params_model is None or not isinstance(params_model, type) or not issubclass(params_model, BaseModel):
        return 0.0, None
    if not isinstance(arguments, dict) or not arguments:
        return 0.0, None

    # Attempt to repair the arguments against this model.
    repaired = _repair_dict_for_model(arguments, params_model)

    model_fields = params_model.model_fields
    n_total = len(model_fields)
    if n_total == 0:
        return 1.0, {} if not arguments else None

    # Count required and optional fields.
    n_required = 0
    for finfo in model_fields.values():
        if finfo.is_required():
            n_required += 1
    n_optional = n_total - n_required
    if n_required == 0:
        n_required = 1  # avoid division by zero; treat as optional-only

    score = 0.0
    exact_matches = 0
    alias_matches = 0
    coercion_count = 0

    # Track which original argument keys were successfully mapped.
    original_keys = set(arguments.keys())
    mapped_keys: set[str] = set()

    for fname, finfo in model_fields.items():
        if fname not in repaired:
            # Field not provided — no score contribution
            continue
        mapped_keys.add(fname)

        # Check if the key was originally provided (exact match) or came via alias/fuzzy.
        was_exact = fname in arguments
        if was_exact:
            exact_matches += 1
            if finfo.is_required():
                score += 0.3 / max(n_required, 1)
            else:
                score += 0.1 / max(n_optional, 1)
            score += 0.1  # exact match bonus
        else:
            alias_matches += 1
            if finfo.is_required():
                score += 0.3 / max(n_required, 1)
            else:
                score += 0.1 / max(n_optional, 1)
            score += 0.05  # alias/fuzzy match bonus

        # Detect type coercion: compare original value vs repaired value type.
        original_val = arguments.get(fname)
        if original_val is not None and fname in repaired:
            repaired_val = repaired[fname]
            if type(original_val) is not type(repaired_val):
                coercion_count += 1

    # Penalty for unmapped keys (LLM sent fields this model does not have).
    for key in arguments:
        if key not in model_fields and key not in repaired:
            score -= 0.15
        elif key in repaired and repaired[key] is not arguments.get(key):
            pass  # was successfully repaired

    # Bonus for type coercions (weak signal).
    score += coercion_count * 0.02

    # Clamp to [0.0, 1.0].
    score = max(0.0, min(1.0, score))

    # Only return repaired args if at least one field was matched.
    if mapped_keys:
        return score, repaired
    return 0.0, None


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    """A candidate tool name with its argument-match score."""

    name: str
    score: float  # 0.0–1.0, higher = better argument match
    args_repaired: dict[str, Any] | None = None


def resolve_tool_by_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    candidates: list[str],
    tool_dict: dict[str, Any],
    *,
    min_argument_score: float = 0.3,
    score_delta: float = 0.15,
) -> ToolNameResolution | None:
    """
    Try to resolve a hallucinated tool name by testing arguments against
    each candidate tool's parameter schema.

    Returns a ``ToolNameResolution`` if the arguments clearly point to one
    candidate (score >= *min_argument_score* and beat second place by
    *score_delta*).  Otherwise returns ``None`` to fall through to name-only
    matching.

    Tools that do not have a typed ``params`` (legacy ``CallableTool``,
    not ``CallableTool2``) are silently skipped.
    """
    if not isinstance(arguments, dict) or not arguments:
        return None
    if not candidates:
        return None

    scored: list[ToolCandidate] = []

    for name in candidates:
        tool = tool_dict.get(name)
        if tool is None:
            continue

        # Get the params model — only CallableTool2 has typed params.
        params_model: type[BaseModel] | None = getattr(tool, "params", None)
        if params_model is None:
            # Legacy CallableTool — cannot score arguments.
            continue

        score, repaired = _score_argument_fit(arguments, params_model)
        scored.append(ToolCandidate(name=name, score=score, args_repaired=repaired))

    if not scored:
        return None

    # Sort by score descending.
    scored.sort(key=lambda c: c.score, reverse=True)

    best = scored[0]
    second_best = scored[1] if len(scored) > 1 else None

    if best.score >= min_argument_score and (
        second_best is None or (best.score - second_best.score) >= score_delta
    ):
        return ToolNameResolution(
            name=best.name,
            original=tool_name,
            corrected=True,
        )

    return None


def _repair_dict_for_model(
    data: dict[str, Any],
    model: type[BaseModel],
    common_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Recursively repair dict keys to match a Pydantic model's expected fields.

    1. Map exact field names and declared aliases.
    2. For remaining missing fields, try common LLM aliases.
    3. Fuzzy-match unmapped keys to missing fields.
    4. Recurse into nested BaseModel fields and list items.
    5. Clamp numeric values that exceed field constraints.
    6. Wrap/unwrap list ↔ scalar mismatches (LLMs often send a single value
       for a list field, or a single-element list for a scalar field).
    7. Apply type coercion for common LLM type mistakes (string→int/float/bool,
       int↔float).
    8. Strip unmapped keys when the model has ``extra="forbid"``.
    """
    if not isinstance(data, dict):
        return {} if data is None else data  # type: ignore[return-value]
    if not data:
        return {}

    if common_aliases is None:
        common_aliases = _COMMON_FIELD_ALIASES

    known, field_meta, extra_forbid, valid_names = _cached_model_field_info(model)

    # First pass: map known keys.
    mapped: dict[str, Any] = {}
    unmapped_keys: set[str] = set()
    for key, value in data.items():
        canonical = known.get(key)
        if canonical is not None:
            mapped[canonical] = value
        else:
            unmapped_keys.add(key)
            mapped[key] = value

    # Second pass: try common aliases for fields that are still missing.
    # Use the precomputed reverse map when the default aliases are in play.
    if common_aliases is _COMMON_FIELD_ALIASES:
        reverse_aliases = _REVERSE_ALIASES
    else:
        reverse_aliases: dict[str, list[str]] = {}
        for bad_key, good_key in common_aliases.items():
            reverse_aliases.setdefault(good_key, []).append(bad_key)

    missing = set(model.model_fields.keys()) - set(mapped.keys())
    for missing_field in list(missing):
        for bad_key in reverse_aliases.get(missing_field, ()):
            if bad_key in unmapped_keys:
                mapped[missing_field] = mapped.pop(bad_key)
                unmapped_keys.discard(bad_key)
                missing.discard(missing_field)
                break

    # Third pass: fuzzy match remaining missing fields against still-available unmapped keys.
    fuzzy_matches = _fuzzy_match_keys(missing, unmapped_keys)
    for missing_field, matched_key in fuzzy_matches.items():
        mapped[missing_field] = mapped.pop(matched_key)

    # Fourth pass: recurse, clamp, fix list/scalar, coerce, all in one loop.
    for fname, nested, constraints, is_list, scalar_type, literal_options in field_meta:
        if fname not in mapped:
            continue
        val = mapped[fname]
        if nested is not None:
            # LLMs sometimes serialize nested objects as JSON strings; parse them
            # before recursing so the nested model repair rules apply.
            if isinstance(val, str):
                val = _maybe_parse_json_string(val)
                mapped[fname] = val

            if isinstance(val, dict):
                mapped[fname] = _repair_dict_for_model(val, nested, common_aliases)
            elif isinstance(val, list):
                repaired_items: list[Any] = []
                for item in val:
                    parsed_item = _maybe_parse_json_string(item)
                    if isinstance(parsed_item, dict):
                        repaired_items.append(
                            _repair_dict_for_model(parsed_item, nested, common_aliases)
                        )
                    else:
                        repaired_items.append(item)
                mapped[fname] = repaired_items
        elif constraints is not None:
            clamped = _apply_constraints(val, constraints)
            if clamped is not None:
                val = clamped
                mapped[fname] = val

        # Fix list ↔ scalar mismatches.
        if is_list and not isinstance(val, list):
            val = [val]
            mapped[fname] = val
        elif not is_list and isinstance(val, list) and len(val) == 1:
            val = val[0]
            mapped[fname] = val

        # Type coercion for common LLM type mistakes.
        if scalar_type is not None and val is not None:
            coerced = _coerce_value(val, scalar_type)
            if coerced is not None:
                mapped[fname] = coerced

        # Fuzzy-match Literal string values to valid options.
        # Always attempt fuzzy matching so that separator-normalized values
        # (e.g. "in-progress" → "in_progress") and aliases (e.g. "completed" →
        # "done") are corrected before Pydantic validation.
        if literal_options is not None and isinstance(val, str):
            matched = _fuzzy_match_literal_value(val, literal_options)
            if matched is not None and matched != val:
                mapped[fname] = matched

    # Fifth pass: strip unmapped keys for "extra=forbid" models.
    if extra_forbid:
        mapped = {k: v for k, v in mapped.items() if k in valid_names}

    return mapped


@lru_cache(maxsize=512)
def _resolve_scalar_type(annotation: Any) -> type | None:
    """Resolve a type annotation to a simple scalar type (str, int, float, bool).

    Handles Optional[X], Union[X, None], X | None, and plain X.
    Returns None if the annotation is not a simple scalar.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is None:
        # Plain type like str, int, float, bool
        if annotation in (str, int, float, bool):
            return annotation
        return None

    # Handle Union / Optional types
    origin_name = getattr(origin, "__name__", "")
    if origin_name in ("Union", "Optional"):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and non_none[0] in (str, int, float, bool):
            return non_none[0]

    return None


def _coerce_value(value: Any, target_type: type) -> Any | None:
    """Attempt to coerce a value to *target_type*.  Returns None if coercion
    is not possible or would be lossy."""
    if isinstance(value, bool):
        # bool is a subclass of int — don't coerce it unless target is int/float
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        if target_type is str:
            return str(value).lower()
        return None

    if target_type is str and not isinstance(value, str):
        if isinstance(value, (int, float)):
            return str(value)
        return None

    if target_type is int and not isinstance(value, int):
        if isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
        if isinstance(value, float) and value == int(value):
            # Only coerce if no precision loss
            return int(value)
        return None

    if target_type is float and not isinstance(value, float):
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        if isinstance(value, int):
            return float(value)
        return None

    if target_type is bool and not isinstance(value, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off", ""):
                return False
        if isinstance(value, int):
            return bool(value)
        return None

    return None


# Known lowercase Python/JSONSchema builtin type names that Pydantic may
# include in error locations as union-branch discriminators.  These are NOT
# real field names and must be filtered out of error paths.
_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset({
    "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "frozenset", "bytes", "None", "none", "null", "object", "array",
    "number", "string", "boolean", "integer",
})


def _clean_error_loc(loc: tuple[str | int, ...]) -> str:
    """Remove union-discriminator noise from a Pydantic error location.

    Pydantic v2 includes union-branch names (e.g. ``Edit``, ``list[Edit]``,
    ``str``, ``int``) in ``loc`` tuples.  This makes the path hard for an
    LLM to read and often points to a branch that was *not* intended.  We
    keep only the real field / index segments.
    """
    cleaned: list[str] = []
    for part in loc:
        s = str(part)
        # Union branch names are capitalised model names or type expressions
        # like ``list[Edit]``.  They may also start with an underscore in
        # private helper classes (e.g. ``_Edit``).  Real field names in this
        # codebase are snake_case; integers are list indices.
        stripped = s.lstrip("_")
        if (stripped and stripped[:1].isupper()) or "[" in s:
            continue
        # Filter lowercase builtin type names that Pydantic uses as
        # union-branch discriminators (e.g. ``"value", "str"``).
        if stripped in _BUILTIN_TYPE_NAMES:
            continue
        cleaned.append(s)
    return ".".join(cleaned) if cleaned else ".".join(str(p) for p in loc)


def _format_pydantic_validation_error(
    error: pydantic.ValidationError,
    tool_name: str,
    params_schema: dict[str, Any] | None = None,
) -> str:
    """Format a pydantic ``ValidationError`` into a concise, LLM-friendly message."""
    errors_list = error.errors()
    lines: list[str] = []
    lines.append(
        f"Invalid arguments for tool `{tool_name}` — "
        f"{len(errors_list)} validation error(s):"
    )
    lines.append("")

    for i, err in enumerate(errors_list, 1):
        loc = _clean_error_loc(err["loc"])
        msg = err["msg"]
        err_type = err.get("type", "")

        lines.append(f"{i}. `{loc}` — {msg}")

        inp = err.get("input")
        if inp is not None:
            inp_str = orjson.dumps(inp).decode()
            if len(inp_str) > 300:
                inp_str = inp_str[:300] + " ..."
            lines.append(f"  Received: {inp_str}")

        # Actionable hints for common error types
        ctx = err.get("ctx", {})
        match err_type:
            case "missing":
                lines.append("  Hint: this field is required but was not provided.")
            case "list_type":
                lines.append("  Hint: this field should be an array (list).")
            case "dict_type":
                lines.append("  Hint: this field should be an object (dict).")
            case "string_type":
                lines.append("  Hint: this field should be a string.")
            case "int_type":
                lines.append("  Hint: this field should be an integer.")
            case "bool_type":
                lines.append("  Hint: this field should be a boolean.")
            case "float_type":
                lines.append("  Hint: this field should be a number.")
            case "literal_error":
                expected = ctx.get("expected", "")
                if expected:
                    lines.append(f"  Hint: the value must be one of: {expected}")
                else:
                    lines.append("  Hint: the value must be one of the allowed options.")
            case "extra_forbidden":
                lines.append(
                    "  Hint: this field is not recognized — remove it or check the schema."
                )
            case "url_parsing":
                lines.append("  Hint: the value must be a valid URL.")
            case "too_short":
                lines.append("  Hint: the value is too short.")
            case "too_long":
                lines.append("  Hint: the value is too long.")
            case "enum":
                expected = ctx.get("expected", "")
                if expected:
                    lines.append(f"  Hint: this value is not one of the allowed options: {expected}")
                else:
                    lines.append("  Hint: this value is not one of the allowed options.")
            case "value_error":
                lines.append("  Hint: this value is invalid — check field constraints.")
            case "union_tag_not_found":
                lines.append("  Hint: a required discriminator field is missing.")
            case "union_tag_invalid":
                tag = ctx.get("discriminator", "")
                if tag:
                    lines.append(f"  Hint: the discriminator value for '{tag}' is not recognized.")
                else:
                    lines.append("  Hint: the discriminator value is not recognized.")
            case "model_type":
                lines.append("  Hint: this field should be a JSON object (dict).")
            case "pattern_mismatch":
                pat = ctx.get("pattern", "")
                if pat:
                    lines.append(f"  Hint: the string does not match the required pattern '{pat}'.")
                else:
                    lines.append("  Hint: the string does not match the required pattern.")
            case "multiple_of":
                mult = ctx.get("multiple_of", "?")
                lines.append(f"  Hint: the number must be a multiple of {mult}.")
            case "finite_number":
                lines.append("  Hint: the value must be a finite number (not NaN or Infinity).")
            case "set_type":
                lines.append("  Hint: this field should be a set/array.")
            case "tuple_type":
                lines.append("  Hint: this field should be a tuple/array.")
            case "date_type":
                lines.append("  Hint: this field should be a valid date.")
            case "datetime_type":
                lines.append("  Hint: this field should be a valid datetime.")
            case "time_type":
                lines.append("  Hint: this field should be a valid time.")
            case "json_type":
                lines.append("  Hint: this field should be a valid JSON string.")
            case "json_invalid":
                lines.append("  Hint: the JSON string is invalid.")
            case "url_scheme":
                allowed = ctx.get("allowed_schemes", "")
                if allowed:
                    lines.append(f"  Hint: the URL scheme must be one of: {allowed}")
                else:
                    lines.append("  Hint: the URL scheme is not allowed.")
            case "none_required":
                lines.append("  Hint: this field must be null/None.")
            case "invalid_key":
                lines.append("  Hint: one or more keys in the object are invalid.")
            case "get_attribute_error":
                lines.append("  Hint: an attribute access error occurred.")
            case "bytes_too_long":
                lines.append("  Hint: the byte string is too long.")
            case "bytes_too_short":
                lines.append("  Hint: the byte string is too short.")
            case "decimal_max_digits":
                lines.append("  Hint: the number has too many digits.")
            case "decimal_max_places":
                lines.append("  Hint: the number has too many decimal places.")
            case "recursion_loop":
                lines.append("  Hint: the data structure contains a recursion loop.")
            case _ if err_type.startswith("greater_than"):
                lines.append("  Hint: the value is too small.")
            case _ if err_type.startswith("less_than"):
                lines.append("  Hint: the value is too large.")
            case _:
                # Fallback: produce a generic hint for unknown error types
                lines.append(f"  Hint: check the field constraints (error type: {err_type}).")

    # Only include the full JSON schema when the errors suggest structural
    # confusion (extra fields, discriminator issues) or when there are many
    # errors (>2), to keep simple error messages concise.
    if params_schema is not None:
        structural_error_types = frozenset({
            "extra_forbidden", "union_tag_not_found", "union_tag_invalid",
        })
        has_structural_error = any(
            e.get("type", "") in structural_error_types
            for e in errors_list
        )
        if has_structural_error or len(errors_list) > 2:
            lines.append("")
            lines.append("Expected JSON schema:")
            schema_str = orjson.dumps(params_schema, option=orjson.OPT_INDENT_2).decode()
            lines.append(schema_str)

    return "\n".join(lines)


class _GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    """Custom JSON schema generator that omits titles."""

    @override
    def field_title_should_be_set(self, schema) -> bool:  # type: ignore[reportMissingParameterType]
        return False

    @override
    def _update_class_schema(self, json_schema, cls, config) -> None:  # type: ignore[reportMissingParameterType]
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)


class CallableTool2[Params: BaseModel](ABC):
    """
    The abstract base class of tools that can be called as callables, with typed parameters.

    The tool will be called with the arguments provided in the `ToolCall`.
    The arguments must be a JSON object, and will be validated by Pydantic to the `Params` type.
    """

    name: str
    """The name of the tool."""
    description: str
    """The description of the tool."""
    params: type[Params]
    """The Pydantic model type of the tool parameters."""
    field_aliases: ClassVar[dict[str, str]] = _COMMON_FIELD_ALIASES
    """Aliases used when repairing dict keys to match this tool's parameter schema."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        params: type[Params] | None = None,
    ) -> None:
        cls = self.__class__

        self.name = name or getattr(cls, "name", "")
        if not self.name:
            raise ValueError(
                "Tool name must be provided either as class variable or constructor argument"
            )
        if not isinstance(self.name, str):  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Tool name must be a string")

        self.description = description or getattr(cls, "description", "")
        if not self.description:
            raise ValueError(
                "Tool description must be provided either as class variable or constructor argument"
            )
        if not isinstance(self.description, str):  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Tool description must be a string")

        self.params = params or getattr(cls, "params", None)  # type: ignore
        if not self.params:
            raise ValueError(
                "Tool param must be provided either as class variable or constructor argument"
            )
        if not isinstance(self.params, type) or not issubclass(self.params, BaseModel):  # type: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Tool params must be a subclass of pydantic.BaseModel")

        self._base = Tool(
            name=self.name,
            description=self.description,
            parameters=deref_json_schema(
                self.params.model_json_schema(schema_generator=_GenerateJsonSchemaNoTitles)
            ),
        )

    @property
    def base(self) -> Tool:
        """The base tool definition."""
        return self._base

    async def call(self, arguments: JsonType) -> ToolReturnValue:
        from kosong.tooling.error import ToolValidateError

        if arguments is None or arguments == "":
            return ToolValidateError(
                f"Invalid arguments for tool `{self.name}`: "
                "expected a JSON object, got an empty value."
            )
        if not isinstance(arguments, dict):
            return ToolValidateError(
                f"Invalid arguments for tool `{self.name}`: "
                f"expected a JSON object, got `{type(arguments).__name__}`."
            )

        try:
            params = self.params.model_validate(arguments)
        except pydantic.ValidationError as e:
            # Attempt to repair common LLM field-name mismatches and re-validate.
            repair_attempted = False
            if isinstance(arguments, dict):
                repaired = _repair_dict_for_model(arguments, self.params, self.field_aliases)
                if repaired != arguments:
                    repair_attempted = True
                    try:
                        params = self.params.model_validate(repaired)
                    except pydantic.ValidationError:
                        pass  # fall through to return the original error
                    else:
                        return await self.__call__(params)
            message = _format_pydantic_validation_error(e, self.name, self.base.parameters)
            if repair_attempted:
                message += (
                    "\n\nNote: automatic argument repair was attempted "
                    "but could not resolve all issues. Please review the "
                    "schema and correct the arguments manually."
                )
            return ToolValidateError(message)

        ret = await self.__call__(params)
        if not isinstance(ret, ToolReturnValue):  # type: ignore[reportUnnecessaryIsInstance]
            # let's do not trust the return type of the tool
            ret = ToolError(
                message=f"Invalid return type: {type(ret)}",
                brief="Invalid return type",
            )
        return ret

    @abstractmethod
    async def __call__(self, params: Params) -> ToolReturnValue:
        """
        @public

        The implementation of the callable tool.
        """
        ...


class ToolResult(BaseModel):
    """The result of a tool call."""

    tool_call_id: str
    """The ID of the tool call."""
    return_value: ToolReturnValue
    """The actual return value of the tool call."""


ToolResultFuture = Future[ToolResult]
type HandleResult = ToolResultFuture | ToolResult


@runtime_checkable
class Toolset(Protocol):
    """
    The interface of toolsets that can register tools and handle tool calls.
    """

    @property
    def tools(self) -> list[Tool]:
        """The list of tool definitions registered in this toolset."""
        ...

    def handle(self, tool_call: ToolCall) -> HandleResult:
        """
        Handle a tool call.
        The result of the tool call, or the async future of the result, should be returned.
        The result should be a `ToolReturnValue`.

        This method MUST NOT do any blocking operations because it will be called during
        consuming the chat response stream.
        This method MUST NOT raise any exception except for `asyncio.CancelledError`. Any other
        error should be returned as a `ToolReturnValue` with `is_error=True`.
        """
        ...
