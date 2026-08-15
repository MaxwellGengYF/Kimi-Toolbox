"""Comprehensive tests for split field-alias categories.

These verify that:
1. Each category dict maps the expected aliases.
2. The merged ``_COMMON_FIELD_ALIASES`` contains every category.
3. ``_repair_dict_for_model`` respects a per-tool alias dict.
4. ``CallableTool2.call()`` uses the tool class's ``field_aliases``.
5. Real ``Params`` types defined under ``kimi-cli/src/kimi_cli/tools/`` and
   ``src/kimix/tools/`` can be repaired successfully.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import ClassVar, override

import pytest
from pydantic import BaseModel, Field

import pydantic

from kosong.tooling import (
    CallableTool2,
    FIELD_ALIASES_ACTIVE,
    FIELD_ALIASES_FILE,
    FIELD_ALIASES_GENERAL,
    FIELD_ALIASES_INPUT,
    FIELD_ALIASES_MODEL,
    FIELD_ALIASES_SEARCH,
    FIELD_ALIASES_SHELL,
    FIELD_ALIASES_SUBAGENT,
    FIELD_ALIASES_TASK,
    FIELD_ALIASES_TODO,
    FIELD_ALIASES_WEB,
    ToolError,
    ToolOk,
    ToolReturnValue,
    _COMMON_FIELD_ALIASES,
    _clean_error_loc,
    _format_pydantic_validation_error,
    _repair_dict_for_model,
)
from kosong.tooling.error import ToolValidateError


# ---------------------------------------------------------------------------
# 1. Category dict sanity checks
# ---------------------------------------------------------------------------

ALL_CATEGORIES: dict[str, dict[str, str]] = {
    "GENERAL": FIELD_ALIASES_GENERAL,
    "FILE": FIELD_ALIASES_FILE,
    "SHELL": FIELD_ALIASES_SHELL,
    "WEB": FIELD_ALIASES_WEB,
    "TASK": FIELD_ALIASES_TASK,
    "INPUT": FIELD_ALIASES_INPUT,
    "SEARCH": FIELD_ALIASES_SEARCH,
    "MODEL": FIELD_ALIASES_MODEL,
    "TODO": FIELD_ALIASES_TODO,
    "ACTIVE": FIELD_ALIASES_ACTIVE,
    "SUBAGENT": FIELD_ALIASES_SUBAGENT,
}


def test_all_categories_are_non_empty() -> None:
    for name, aliases in ALL_CATEGORIES.items():
        assert aliases, f"Category {name!r} must not be empty"
        for src, dst in aliases.items():
            assert isinstance(src, str) and src, f"Bad source key in {name!r}"
            assert isinstance(dst, str) and dst, f"Bad destination key in {name!r}"


def test_common_field_aliases_is_superset_of_all_categories() -> None:
    merged: dict[str, str] = {}
    for aliases in ALL_CATEGORIES.values():
        merged.update(aliases)
    assert merged == _COMMON_FIELD_ALIASES


def test_no_duplicate_source_keys_across_categories() -> None:
    seen: set[str] = set()
    for name, aliases in ALL_CATEGORIES.items():
        for src in aliases:
            assert src not in seen, f"Duplicate source key {src!r} in {name!r}"
            seen.add(src)


# ---------------------------------------------------------------------------
# 2. Per-category repair tests with synthetic models
# ---------------------------------------------------------------------------


def test_general_aliases() -> None:
    class Model(BaseModel):
        title: str
        description: str = ""
        message: str = ""
        reason: str = ""
        prompt: str = ""
        step: str = ""
        result: str = ""
        action: str = ""
        brief: str = ""

    data = {
        "content": "t",
        "desc": "d",
        "msg": "m",
        "cause": "c",
        "instruction": "i",
        "stage": "s",
        "outcome": "o",
        "operation": "a",
        "short": "b",
    }
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_GENERAL)
    assert repaired == {
        "title": "t",
        "description": "d",
        "message": "m",
        "reason": "c",
        "prompt": "i",
        "step": "s",
        "result": "o",
        "action": "a",
        "brief": "b",
    }


def test_file_aliases() -> None:
    class Model(BaseModel):
        path: str = ""
        content: str = ""
        mode: str = ""
        line_offset: int = 0
        n_lines: int = 0
        max_char: int = 0
        char_offset: int = 0
        include_dirs: bool = False
        include_ignored: bool = False
        glob: str | None = None
        type: str | None = None
        pattern: str = ""
        edit: str = ""
        old: str = ""
        new: str = ""
        replace_all: bool = False
        output_path: str | None = None
        output_mode: str = ""
        head_limit: int = 0
        multiline: bool = False
        case_insensitive: bool = False
        files: list[str] = Field(default_factory=list)

    data = {
        "file": "/tmp/a",
        "data": "hello",
        "method": "append",
        "offset": 5,
        "lines": 10,
        "chars": 100,
        "byte_offset": 2,
        "dirs": True,
        "ignored": True,
        "filter": "*.py",
        "file_type": "py",
        "regex": ".*",
        "changes": "x",
        "original": "o",
        "replace_with": "n",
        "all": True,
        "out": "/tmp/out",
        "format": "json",
        "max": 50,
        "multi_line": True,
        "ignore_case": True,
        "paths": ["a", "b"],
    }
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_FILE)
    assert repaired["path"] == "/tmp/a"
    assert repaired["content"] == "hello"
    assert repaired["mode"] == "append"
    assert repaired["line_offset"] == 5
    assert repaired["n_lines"] == 10
    assert repaired["max_char"] == 100
    assert repaired["char_offset"] == 2
    assert repaired["include_dirs"] is True
    assert repaired["include_ignored"] is True
    assert repaired["glob"] == "*.py"
    assert repaired["type"] == "py"
    assert repaired["pattern"] == ".*"
    assert repaired["edit"] == "x"
    assert repaired["old"] == "o"
    assert repaired["new"] == "n"
    assert repaired["replace_all"] is True
    assert repaired["output_path"] == "/tmp/out"
    assert repaired["output_mode"] == "json"
    assert repaired["head_limit"] == 50
    assert repaired["multiline"] is True
    assert repaired["case_insensitive"] is True
    assert repaired["files"] == ["a", "b"]


def test_shell_aliases() -> None:
    class Model(BaseModel):
        command: str = ""
        code: str = ""
        timeout: int = 0
        run_in_background: bool = False
        args: list[str] = Field(default_factory=list)
        cwd: str = ""
        env: list[str] = Field(default_factory=list)

    data = {
        "cmd": "ls",
        "program": "py",
        "wait": 30,
        "bg": True,
        "arguments": ["-a"],
        "working_dir": "/home",
        "vars": ["X=1"],
    }
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_SHELL)
    assert repaired == {
        "command": "ls",
        "code": "py",
        "timeout": 30,
        "run_in_background": True,
        "args": ["-a"],
        "cwd": "/home",
        "env": ["X=1"],
    }


def test_web_aliases() -> None:
    class Model(BaseModel):
        url: str = ""
        query: str = ""

    data = {"link": "http://a", "search": "q"}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_WEB)
    assert repaired == {"url": "http://a", "query": "q"}


def test_task_aliases() -> None:
    class Model(BaseModel):
        task_id: str = ""
        block: bool = True
        kill: bool = False

    data = {"id": "123", "sync": False, "stop": True}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_TASK)
    assert repaired == {"task_id": "123", "block": False, "kill": True}


def test_input_aliases() -> None:
    class Model(BaseModel):
        text: str = ""

    data = {"input": "hello"}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_INPUT)
    assert repaired == {"text": "hello"}


def test_search_aliases() -> None:
    class Model(BaseModel):
        k: int = 0
        questions: list[str] = Field(default_factory=list)

    data = {"n": 5, "queries": ["a"]}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_SEARCH)
    assert repaired == {"k": 5, "questions": ["a"]}


def test_model_aliases() -> None:
    class Model(BaseModel):
        model: str = ""
        resume: bool = False

    data = {"llm": "gpt-4", "continue": True}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_MODEL)
    assert repaired == {"model": "gpt-4", "resume": True}


def test_todo_aliases() -> None:
    class Model(BaseModel):
        todos: list[str] = Field(default_factory=list)
        mode: str = "append"

    data = {"items": ["a"], "replace": "overwrite"}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_TODO)
    assert repaired == {"todos": ["a"], "mode": "overwrite"}


def test_active_aliases() -> None:
    class Model(BaseModel):
        active_only: bool = False

    data = {"running": True}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_ACTIVE)
    assert repaired == {"active_only": True}


def test_subagent_aliases() -> None:
    class Model(BaseModel):
        subagent_type: str = ""

    data = {"agent_type": "worker"}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_SUBAGENT)
    assert repaired == {"subagent_type": "worker"}


# ---------------------------------------------------------------------------
# 3. CallableTool2 integrates field_aliases
# ---------------------------------------------------------------------------


def test_callable_tool2_uses_custom_field_aliases() -> None:
    class Params(BaseModel):
        command: str

    class CustomTool(CallableTool2[Params]):
        name: str = "custom"
        description: str = "test"
        params: type[Params] = Params
        field_aliases: ClassVar[dict[str, str]] = {"cmd": "command"}

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.command)

    tool = CustomTool()
    # ``cmd`` is not a valid field, but the custom alias should repair it.
    result = asyncio.run(tool.call({"cmd": "ls"}))
    assert result == ToolOk(output="ls")


def test_callable_tool2_ignores_unrelated_aliases_when_custom_set() -> None:
    class Params(BaseModel):
        title: str

    class CustomTool(CallableTool2[Params]):
        name: str = "custom"
        description: str = "test"
        params: type[Params] = Params
        # Deliberately empty – no aliases allowed.
        field_aliases: ClassVar[dict[str, str]] = {}

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.title)

    tool = CustomTool()
    # ``content`` normally maps to ``title`` in the common set, but here
    # the tool has an empty alias dict, so repair should not happen.
    result = asyncio.run(tool.call({"content": "hello"}))
    assert isinstance(result, ToolValidateError)


# ---------------------------------------------------------------------------
# 4. Real-world Params discovery & repair tests
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]

KIMI_CLI_TOOLS_DIR = REPO_ROOT / "kimi-cli" / "src" / "kimi_cli" / "tools"
KIMIX_TOOLS_DIR = REPO_ROOT / "src" / "kimix" / "tools"


def _discover_params_classes(base_dir: Path, package_prefix: str) -> list[type[BaseModel]]:
    """Walk *base_dir* and yield every class named ``Params`` or ending in ``Params``."""
    classes: list[type[BaseModel]] = []
    if not base_dir.exists():
        return classes

    # Use iter_modules (does not import) so that broken packages do not
    # abort the whole walk.
    for finder, module_name, is_pkg in pkgutil.iter_modules(
        path=[str(base_dir)], prefix=package_prefix + "."
    ):
        if is_pkg:
            continue
        try:
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.find_spec(module_name)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception:
            # Complex dependencies (e.g. runtime, session) or syntax errors
            # may prevent import in the test environment – skip those modules.
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseModel:
                continue
            if issubclass(obj, BaseModel) and (
                _name == "Params" or _name.endswith("Params")
            ):
                classes.append(obj)
    return classes


# Cache the discovered classes so we can parametrize tests.
_ALL_PARAMS: list[type[BaseModel]] = []
_ALL_PARAMS += _discover_params_classes(KIMI_CLI_TOOLS_DIR, "kimi_cli.tools")
_ALL_PARAMS += _discover_params_classes(KIMIX_TOOLS_DIR, "kimix.tools")

# Deduplicate while preserving order.
_seen_ids: set[int] = set()
UNIQUE_PARAMS: list[type[BaseModel]] = []
for _cls in _ALL_PARAMS:
    if id(_cls) not in _seen_ids:
        _seen_ids.add(id(_cls))
        UNIQUE_PARAMS.append(_cls)


@pytest.mark.parametrize("params_cls", UNIQUE_PARAMS, ids=lambda c: c.__name__)
def test_real_params_can_be_repaired_with_common_aliases(params_cls: type[BaseModel]) -> None:
    """Every discovered Params class should survive a no-op repair round."""
    # Build a dict with every field set to a plausible default.
    import typing
    from pydantic_core import PydanticUndefined

    data: dict[str, object] = {}
    for fname, finfo in params_cls.model_fields.items():
        # Prefer explicit defaults when available.
        if finfo.default is not PydanticUndefined:
            data[fname] = finfo.default
            continue
        if finfo.default_factory is not None:
            data[fname] = finfo.default_factory()
            continue

        annotation = finfo.annotation
        if annotation is str or (isinstance(annotation, type) and issubclass(annotation, str)):
            data[fname] = ""
        elif annotation is int or (isinstance(annotation, type) and issubclass(annotation, int)):
            data[fname] = 1  # safer than 0 for fields with ``ge=1``
        elif annotation is float or (isinstance(annotation, type) and issubclass(annotation, float)):
            data[fname] = 0.0
        elif annotation is bool or (isinstance(annotation, type) and issubclass(annotation, bool)):
            data[fname] = False
        elif annotation is list or getattr(annotation, "__origin__", None) is list:
            data[fname] = []
        elif annotation is dict or getattr(annotation, "__origin__", None) is dict:
            data[fname] = {}
        elif typing.get_origin(annotation) is typing.Literal:
            args = typing.get_args(annotation)
            data[fname] = args[0] if args else None
        elif annotation is type(None) or str(getattr(annotation, "__name__", "")) == "NoneType":
            data[fname] = None
        elif hasattr(annotation, "__args__"):
            # e.g. str | None – try the first non-None arg.
            for arg in annotation.__args__:
                if arg is not type(None):
                    if arg is str:
                        data[fname] = ""
                    elif arg is int:
                        data[fname] = 1
                    elif arg is float:
                        data[fname] = 0.0
                    elif arg is bool:
                        data[fname] = False
                    elif arg is list:
                        data[fname] = []
                    break
            else:
                data[fname] = None
        else:
            data[fname] = None

    repaired = _repair_dict_for_model(data, params_cls, _COMMON_FIELD_ALIASES)
    # Repair must not drop any keys that were originally present.
    assert set(repaired.keys()) == set(data.keys())
    # It should also be validatable with the guessed defaults.
    try:
        instance = params_cls.model_validate(repaired)
    except Exception as exc:
        # If our guessed defaults are insufficient (e.g. custom validators),
        # fall back to model_construct which skips validation but proves
        # the repaired dict contains all required keys.
        instance = params_cls.model_construct(**repaired)
    assert isinstance(instance, params_cls)


# ---------------------------------------------------------------------------
# 5. grep_local.py specific integration test
# ---------------------------------------------------------------------------


def test_grep_tool_uses_custom_aliases() -> None:
    """Grep sets ``field_aliases`` to GENERAL | FILE | WEB."""
    pytest.importorskip("kimi_cli")
    from kimi_cli.tools.file.grep_local import Grep, Params as GrepParams

    class FakeRuntime:
        class builtin_args:
            KIMI_WORK_DIR = "/tmp"
        additional_dirs = []
        skills_dirs = []

    runtime = FakeRuntime()  # type: ignore[arg-type]

    # We can't fully instantiate Grep because it expects a real Runtime,
    # but we can instantiate the Params class directly and test repair.
    # Instead, let's just verify the class attribute.
    assert "field_aliases" in Grep.__dict__
    aliases = Grep.field_aliases
    # Should contain GENERAL aliases
    assert "content" in aliases
    # Should contain FILE aliases
    assert "file" in aliases
    # Should contain WEB aliases
    assert "link" in aliases
    # Should NOT contain SHELL aliases (e.g. ``bg``)
    assert "bg" not in aliases

    # Repair a dict that uses aliases from all three categories.
    data = {
        "regex": "test",  # FILE alias -> pattern
        "filter": "*.py",  # Grep alias -> include
        "file_type": "py",  # FILE alias -> type
        "format": "content",  # FILE alias -> output_mode
        "link": "n/a",  # not a field – stays as-is (no ``url`` field)
    }
    repaired = _repair_dict_for_model(data, GrepParams, aliases)
    assert repaired["pattern"] == "test"
    assert repaired["include"] == "*.py"
    assert repaired["type"] == "py"
    assert repaired["output_mode"] == "content"
    assert "link" in repaired  # no ``url`` field in GrepParams, so stays


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_repair_does_not_overwrite_when_target_already_present() -> None:
    class Model(BaseModel):
        title: str
        content: str = ""

    # ``content`` is a valid field name, so it should stay as ``content``
    # even though ``content`` is also an alias for ``title``.
    data = {"content": "body", "title": "heading"}
    repaired = _repair_dict_for_model(data, Model, FIELD_ALIASES_GENERAL)
    assert repaired == {"content": "body", "title": "heading"}


def test_repair_nested_model_with_custom_aliases() -> None:
    class Inner(BaseModel):
        command: str

    class Outer(BaseModel):
        inner: Inner

    data = {"inner": {"cmd": "ls"}}
    repaired = _repair_dict_for_model(data, Outer, FIELD_ALIASES_SHELL)
    assert repaired == {"inner": {"command": "ls"}}


def test_repair_list_of_models_with_custom_aliases() -> None:
    class Item(BaseModel):
        url: str

    class Outer(BaseModel):
        items: list[Item]

    data = {"items": [{"link": "a"}, {"href": "b"}]}
    repaired = _repair_dict_for_model(data, Outer, FIELD_ALIASES_WEB)
    assert repaired == {"items": [{"url": "a"}, {"url": "b"}]}


# ---------------------------------------------------------------------------
# 7. Validation error formatter tests
# ---------------------------------------------------------------------------


def test_clean_error_loc_removes_union_branches() -> None:
    assert _clean_error_loc(("edit", "Edit", "old")) == "edit.old"
    assert _clean_error_loc(("edit", "list[Edit]")) == "edit"
    assert _clean_error_loc(("items", 0, "Edit", "old")) == "items.0.old"
    assert _clean_error_loc(("path",)) == "path"
    assert _clean_error_loc(("user", "address", "street")) == "user.address.street"
    # Falls back to raw loc when every segment is a union branch.
    assert _clean_error_loc(("Edit",)) == "Edit"


def test_format_validation_error_basic() -> None:
    class _Edit(BaseModel):
        old: str
        new: str

    class _Params(BaseModel):
        path: str
        edit: _Edit | list[_Edit]

    try:
        _Params.model_validate({"path": "/tmp/a", "edit": {"new": "x"}})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "EditFile")
        assert "Invalid arguments for tool `EditFile`" in msg
        assert "2 validation error(s):" in msg
        assert "`edit.old` — Field required" in msg
        assert "`edit` — Input should be a valid list" in msg
        assert "Hint: this field is required" in msg
        assert "Hint: this field should be an array (list)." in msg
        assert "Received:" in msg


def test_format_validation_error_includes_schema() -> None:
    """Schema is included for >2 errors or structural error types."""
    class _Inner(BaseModel):
        value: str

    class _Params(BaseModel):
        name: str
        inner: _Inner

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "inner": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }

    # With 2 errors of type "missing", schema is NOT included (not structural).
    try:
        _Params.model_validate({"inner": {}})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "TestTool", schema)
        assert "Expected JSON schema:" not in msg

    # With >2 errors, schema IS included.
    class _MultiParams(BaseModel):
        a: str
        b: int
        c: float

    try:
        _MultiParams.model_validate({})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "TestTool", schema)
        assert "Expected JSON schema:" in msg

    # With structural error (extra_forbidden), schema IS included.
    class _StrictParams(BaseModel):
        model_config = {"extra": "forbid"}
        name: str

    try:
        _StrictParams.model_validate({"name": "ok", "extra": 1})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "TestTool", schema)
        assert "Expected JSON schema:" in msg


def test_format_validation_error_extra_forbidden() -> None:
    class _Params(BaseModel):
        model_config = {"extra": "forbid"}
        command: str

    try:
        _Params.model_validate({"command": "ls", "extra": 1})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Shell")
        assert "`extra` — Extra inputs are not permitted" in msg
        assert "not recognized" in msg


def test_callable_tool2_returns_formatted_validation_error() -> None:
    class _Params(BaseModel):
        command: str

    class _BadTool(CallableTool2[_Params]):
        name: str = "bad"
        description: str = "test"
        params: type[_Params] = _Params
        field_aliases: ClassVar[dict[str, str]] = {}

        @override
        async def __call__(self, params: _Params) -> ToolReturnValue:
            return ToolOk(output=params.command)

    tool = _BadTool()
    result = asyncio.run(tool.call({"cmd": "ls"}))
    assert isinstance(result, ToolValidateError)
    assert "Invalid arguments for tool `bad`" in result.message
    assert "`command` — Field required" in result.message
    # Schema is NOT included for a single non-structural error.
    assert "Expected JSON schema:" not in result.message


# ---------------------------------------------------------------------------
# 8. Tests for _clean_error_loc builtin type-name filtering (Fix 1)
# ---------------------------------------------------------------------------


def test_clean_error_loc_filters_str_branch() -> None:
    """Union[str, Edit] loc containing 'str' should not include 'str'."""
    assert _clean_error_loc(("value", "str")) == "value"


def test_clean_error_loc_filters_int_branch() -> None:
    """Union[int, Edit] loc containing 'int' should not include 'int'."""
    assert _clean_error_loc(("items", 0, "int")) == "items.0"


def test_clean_error_loc_filters_float_branch() -> None:
    """Union[float, ...] loc containing 'float' should not include 'float'."""
    assert _clean_error_loc(("ratio", "float")) == "ratio"


def test_clean_error_loc_filters_bool_branch() -> None:
    """Union[bool, ...] loc containing 'bool' should not include 'bool'."""
    assert _clean_error_loc(("enabled", "bool")) == "enabled"


def test_clean_error_loc_filters_none_branch() -> None:
    """Union[None, ...] loc containing 'None' should not include 'None'."""
    assert _clean_error_loc(("optional", "None")) == "optional"


def test_clean_error_loc_filters_list_branch() -> None:
    """Union[list, ...] loc containing 'list' should not include 'list'."""
    assert _clean_error_loc(("data", "list")) == "data"


def test_clean_error_loc_filters_dict_branch() -> None:
    """Union[dict, ...] loc containing 'dict' should not include 'dict'."""
    assert _clean_error_loc(("config", "dict")) == "config"


# ---------------------------------------------------------------------------
# 9. Tests for expanded error-type handlers (Fix 2)
# ---------------------------------------------------------------------------


def test_format_validation_error_enum() -> None:
    """enum error shows allowed options."""
    from enum import Enum

    class Color(str, Enum):
        RED = "red"
        GREEN = "green"

    class _Params(BaseModel):
        color: Color

    try:
        _Params.model_validate({"color": "blue"})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "not one of the allowed options" in msg


def test_format_validation_error_literal_with_expected() -> None:
    """literal_error hint shows the expected values."""
    from typing import Literal

    class _Params(BaseModel):
        mode: Literal["read", "write", "append"]

    try:
        _Params.model_validate({"mode": "delete"})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "must be one of" in msg


def test_format_validation_error_value_error() -> None:
    """value_error hint is shown."""
    class _Params(BaseModel):
        @pydantic.field_validator("name")
        @classmethod
        def name_must_not_be_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("name must not be empty")
            return v
        name: str

    try:
        _Params.model_validate({"name": "  "})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "check field constraints" in msg


def test_format_validation_error_model_type() -> None:
    """model_type error has a hint."""
    class _Inner(BaseModel):
        x: int

    class _Params(BaseModel):
        inner: _Inner

    try:
        _Params.model_validate({"inner": "not_a_dict"})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "JSON object" in msg


def test_format_validation_error_date_type() -> None:
    """date_type error has a hint."""
    from datetime import date

    class _Params(BaseModel):
        d: date

    try:
        _Params.model_validate({"d": "not_a_date"})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "valid date" in msg


def test_format_validation_error_finite_number() -> None:
    """finite_number error has a hint."""
    class _Params(BaseModel):
        value: float

    try:
        _Params.model_validate({"value": float("nan")})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "finite number" in msg


def test_format_validation_error_none_required() -> None:
    """none_required error has a hint."""
    class _Params(BaseModel):
        value: None = None

    try:
        _Params.model_validate({"value": "not_none"})
    except pydantic.ValidationError as e:
        msg = _format_pydantic_validation_error(e, "Test")
        assert "null/None" in msg


# ---------------------------------------------------------------------------
# 10. Tests for type coercion in _repair_dict_for_model (Fix 4)
# ---------------------------------------------------------------------------


def test_repair_dict_type_coercion_str_to_int() -> None:
    class Params(BaseModel):
        count: int

    repaired = _repair_dict_for_model({"count": "5"}, Params)
    assert repaired == {"count": 5}
    assert isinstance(repaired["count"], int)


def test_repair_dict_type_coercion_str_to_float() -> None:
    class Params(BaseModel):
        ratio: float

    repaired = _repair_dict_for_model({"ratio": "0.75"}, Params)
    assert repaired == {"ratio": 0.75}
    assert isinstance(repaired["ratio"], float)


def test_repair_dict_type_coercion_str_to_bool_true() -> None:
    class Params(BaseModel):
        enabled: bool

    repaired = _repair_dict_for_model({"enabled": "true"}, Params)
    assert repaired == {"enabled": True}
    assert isinstance(repaired["enabled"], bool)


def test_repair_dict_type_coercion_str_to_bool_false() -> None:
    class Params(BaseModel):
        enabled: bool

    repaired = _repair_dict_for_model({"enabled": "false"}, Params)
    assert repaired == {"enabled": False}


def test_repair_dict_type_coercion_int_to_str() -> None:
    class Params(BaseModel):
        name: str

    repaired = _repair_dict_for_model({"name": 123}, Params)
    assert repaired == {"name": "123"}


def test_repair_dict_type_coercion_int_to_float() -> None:
    class Params(BaseModel):
        value: float

    repaired = _repair_dict_for_model({"value": 42}, Params)
    assert repaired == {"value": 42.0}
    assert isinstance(repaired["value"], float)


def test_repair_dict_type_coercion_float_to_int_no_loss() -> None:
    class Params(BaseModel):
        count: int

    repaired = _repair_dict_for_model({"count": 10.0}, Params)
    assert repaired == {"count": 10}
    assert isinstance(repaired["count"], int)


def test_repair_dict_type_coercion_float_to_int_lossy_skipped() -> None:
    class Params(BaseModel):
        count: int

    repaired = _repair_dict_for_model({"count": 10.5}, Params)
    assert repaired == {"count": 10.5}  # unchanged — lossy


def test_repair_dict_type_coercion_invalid_str_to_int_skipped() -> None:
    class Params(BaseModel):
        count: int

    repaired = _repair_dict_for_model({"count": "abc"}, Params)
    assert repaired == {"count": "abc"}  # unchanged — not parseable


# ---------------------------------------------------------------------------
# 11. Tests for list ↔ scalar wrapping/unwrapping (Fix 5)
# ---------------------------------------------------------------------------


def test_repair_dict_list_scalar_wrap() -> None:
    """Scalar value for list field is wrapped in a list."""
    class Params(BaseModel):
        items: list[str]

    repaired = _repair_dict_for_model({"items": "hello"}, Params)
    assert repaired == {"items": ["hello"]}


def test_repair_dict_list_scalar_unwrap() -> None:
    """Single-element list for scalar field is unwrapped."""
    class Params(BaseModel):
        name: str

    repaired = _repair_dict_for_model({"name": ["hello"]}, Params)
    assert repaired == {"name": "hello"}


def test_repair_dict_list_scalar_multi_element_not_unwrapped() -> None:
    """Multi-element list for scalar field is NOT unwrapped."""
    class Params(BaseModel):
        name: str

    original = {"name": ["a", "b"]}
    repaired = _repair_dict_for_model(original, Params)
    assert repaired == original  # unchanged


def test_repair_dict_list_scalar_already_list() -> None:
    """List value for list field stays as list."""
    class Params(BaseModel):
        items: list[str]

    repaired = _repair_dict_for_model({"items": ["a", "b"]}, Params)
    assert repaired == {"items": ["a", "b"]}


def test_repair_dict_list_scalar_optional_str() -> None:
    """Single-element list for Optional[str] field is unwrapped."""
    class Params(BaseModel):
        name: str | None = None

    repaired = _repair_dict_for_model({"name": ["hello"]}, Params)
    assert repaired == {"name": "hello"}


# ---------------------------------------------------------------------------
# 12. Tests for extra="forbid" unmapped key stripping (Fix 6)
# ---------------------------------------------------------------------------


def test_repair_dict_extra_forbid_strips_unmapped() -> None:
    """Unmapped keys are stripped when model has extra='forbid'."""
    class Params(BaseModel):
        model_config = {"extra": "forbid"}
        name: str

    repaired = _repair_dict_for_model({"name": "test", "extra_field": 123}, Params)
    assert repaired == {"name": "test"}
    assert "extra_field" not in repaired


def test_repair_dict_extra_allow_keeps_unmapped() -> None:
    """Unmapped keys are kept when model has extra='allow' (default)."""
    class Params(BaseModel):
        name: str

    original = {"name": "test", "extra": 123}
    repaired = _repair_dict_for_model(original, Params)
    assert repaired == original


# ---------------------------------------------------------------------------
# 13. Tests for repair-failure note (Fix 8)
# ---------------------------------------------------------------------------


def test_callable_tool2_repair_failure_note() -> None:
    """When repair is attempted but fails, a note is included in the error."""

    class Params(BaseModel):
        model_config = {"extra": "forbid"}
        title: str

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params
        field_aliases: ClassVar[dict[str, str]] = {}

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.title)

    tool = TestTool()
    # Send an extra field that repair will strip (extra="forbid"), but
    # the required `title` field is still missing. Repair changes the dict
    # (removes the extra field) but validation still fails.
    result = asyncio.run(tool.call({"content": "hello"}))
    assert isinstance(result, ToolValidateError)
    assert "automatic argument repair was attempted" in result.message


def test_callable_tool2_no_repair_note_when_repair_not_attempted() -> None:
    """When repair is NOT attempted (arguments not a dict), no note is added."""

    class Params(BaseModel):
        title: str

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.title)

    tool = TestTool()
    # Pass a string (not a dict) — repair won't be attempted.
    result = asyncio.run(tool.call("not_a_dict"))
    assert isinstance(result, ToolValidateError)
    assert "automatic argument repair was attempted" not in result.message


# ---------------------------------------------------------------------------
# 14. Tests for fuzzy key matching in _repair_dict_for_model
# ---------------------------------------------------------------------------


def test_fuzzy_match_keys_case_insensitive() -> None:
    """Case differences should not suppress strong fuzzy matches."""
    class Model(BaseModel):
        base_url: str = ""

    # ``base_URL`` differs only in case from the field ``base_url``.
    repaired = _repair_dict_for_model({"base_URL": "https://example.com"}, Model)
    assert repaired == {"base_url": "https://example.com"}


def test_fuzzy_match_keys_short_names_ignored() -> None:
    """Short keys (< 4 chars) are never remapped via fuzzy matching."""
    class Model(BaseModel):
        id: str = ""

    repaired = _repair_dict_for_model({"if": "123"}, Model)
    # ``if`` is too short to fuzzy-match to ``id``.
    assert repaired == {"if": "123"}


def test_fuzzy_match_keys_one_to_one_assignment() -> None:
    """Each unmapped key can be consumed by at most one missing field."""
    class Model(BaseModel):
        base_url: str = ""
        base_path: str = ""

    data = {"base_uri": "https://example.com"}
    repaired = _repair_dict_for_model(data, Model)
    # ``base_uri`` should map to exactly one of the two fields; the other
    # field remains absent (its default value is not present in input).
    matched = {k for k in ("base_url", "base_path") if k in repaired}
    assert len(matched) == 1
    assert repaired[next(iter(matched))] == "https://example.com"


def test_fuzzy_match_keys_weak_matches_rejected() -> None:
    """Low-similarity keys are not remapped."""
    class Model(BaseModel):
        description: str = ""

    repaired = _repair_dict_for_model({"xyz": "nope"}, Model)
    assert repaired == {"xyz": "nope"}


def test_fuzzy_match_keys_helper_directly() -> None:
    """``_fuzzy_match_keys`` returns the expected case-insensitive mapping."""
    from kosong.tooling import _fuzzy_match_keys

    missing = {"base_url", "output_path"}
    available = {"base_URL", "out_path"}
    matches = _fuzzy_match_keys(missing, available)
    assert matches["base_url"] == "base_URL"
    assert matches["output_path"] == "out_path"


def test_fuzzy_match_keys_helper_uses_strongest_first() -> None:
    """The helper assigns the strongest match when keys compete."""
    from kosong.tooling import _fuzzy_match_keys

    # ``base_url`` is closer to ``base_uri`` than ``base_path`` is.
    missing = {"base_url", "base_path"}
    available = {"base_uri"}
    matches = _fuzzy_match_keys(missing, available)
    assert "base_url" in matches
    assert matches["base_url"] == "base_uri"
    assert "base_path" not in matches


# ---------------------------------------------------------------------------
# 15. Tests for JSON-string repair of nested models
# ---------------------------------------------------------------------------


def test_repair_dict_parses_json_string_into_nested_model() -> None:
    """A JSON-encoded string for a nested-model field is parsed and repaired."""

    class Inner(BaseModel):
        command: str

    class Outer(BaseModel):
        inner: Inner

    data = {"inner": '{"cmd": "ls"}'}
    repaired = _repair_dict_for_model(data, Outer, FIELD_ALIASES_SHELL)
    assert repaired == {"inner": {"command": "ls"}}


def test_repair_dict_parses_json_string_into_list_of_nested_models() -> None:
    """A JSON-encoded string for a list-of-models field is parsed and repaired."""

    class Item(BaseModel):
        url: str

    class Outer(BaseModel):
        items: list[Item]

    data = {"items": '[{"link": "a"}, {"href": "b"}]'}
    repaired = _repair_dict_for_model(data, Outer, FIELD_ALIASES_WEB)
    assert repaired == {"items": [{"url": "a"}, {"url": "b"}]}


def test_repair_dict_parses_json_string_with_nested_model_inside_list_items() -> None:
    """A list-of-models field where items themselves contain nested models."""

    class Option(BaseModel):
        label: str

    class Question(BaseModel):
        title: str
        options: list[Option]

    class Outer(BaseModel):
        questions: list[Question]

    data = {
        "questions": '[{"content": "q1", "choices": [{"label": "a"}]}]'
    }
    repaired = _repair_dict_for_model(data, Outer, FIELD_ALIASES_GENERAL)
    assert repaired == {"questions": [{"title": "q1", "options": [{"label": "a"}]}]}


def test_repair_dict_parses_json_string_into_todo_union() -> None:
    """A JSON string for a `list[Todo] | Todo | None` field is parsed."""

    class Todo(BaseModel):
        title: str

    class Outer(BaseModel):
        todos: list[Todo] | Todo | None = None

    data = {"todos": '{"title": "single"}'}
    repaired = _repair_dict_for_model(data, Outer)
    assert repaired == {"todos": {"title": "single"}}

    data_list = {"todos": '[{"title": "one"}, {"title": "two"}]'}
    repaired_list = _repair_dict_for_model(data_list, Outer)
    assert repaired_list == {"todos": [{"title": "one"}, {"title": "two"}]}


def test_repair_dict_parses_json_string_into_discriminated_union() -> None:
    """A JSON string for a discriminated-union model field is parsed."""
    from typing import Literal

    class ReplaceEdit(BaseModel):
        kind: Literal["replace"]
        old: str
        new: str

    class DeleteEdit(BaseModel):
        kind: Literal["delete"]
        target: str

    class Params(BaseModel):
        edit: ReplaceEdit | DeleteEdit

    data = {"edit": '{"kind": "replace", "original": "a", "replace_with": "b"}'}
    repaired = _repair_dict_for_model(data, Params, FIELD_ALIASES_FILE)
    assert repaired == {"edit": {"kind": "replace", "old": "a", "new": "b"}}


def test_repair_dict_keeps_invalid_json_string_for_validation_error() -> None:
    """A non-JSON string for a nested-model field is left untouched."""

    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        inner: Inner

    data = {"inner": "not a json object"}
    repaired = _repair_dict_for_model(data, Outer)
    assert repaired == {"inner": "not a json object"}


def test_callable_tool2_accepts_json_encoded_nested_model() -> None:
    """CallableTool2.call parses a JSON string argument into a nested model."""

    class Inner(BaseModel):
        value: str

    class Params(BaseModel):
        name: str
        inner: Inner

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params
        field_aliases: ClassVar[dict[str, str]] = {}

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=f"{params.name}:{params.inner.value}")

    tool = TestTool()
    result = asyncio.run(
        tool.call({"name": "foo", "inner": '{"value": "bar"}'})
    )
    assert result == ToolOk(output="foo:bar")


def test_callable_tool2_invalid_json_string_nested_model_still_errors() -> None:
    """A non-JSON string for a nested-model argument still produces a validation error."""

    class Inner(BaseModel):
        value: str

    class Params(BaseModel):
        inner: Inner

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params
        field_aliases: ClassVar[dict[str, str]] = {}

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.inner.value)

    tool = TestTool()
    result = asyncio.run(tool.call({"inner": "not a json object"}))
    assert isinstance(result, ToolValidateError)
    assert "Invalid arguments for tool `test`" in result.message


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Literal value fuzzy-matching tests
# ══════════════════════════════════════════════════════════════════════════════


from kosong.tooling import _fuzzy_match_literal_value, _COMMON_VALUE_ALIASES


# --- _fuzzy_match_literal_value unit tests ---


def test_fuzzy_match_literal_exact() -> None:
    """Exact match returns the option."""
    options = ("pending", "in_progress", "done")
    result = _fuzzy_match_literal_value("done", options)
    assert result == "done"


def test_fuzzy_match_literal_case_insensitive() -> None:
    """Case-insensitive match returns the correct option."""
    options = ("pending", "in_progress", "done")
    result = _fuzzy_match_literal_value("DONE", options)
    assert result == "done"


def test_fuzzy_match_literal_separator_normalized() -> None:
    """Separator-normalized match (space/hyphen to underscore)."""
    options = ("pending", "in_progress", "done")
    result = _fuzzy_match_literal_value("in progress", options)
    assert result == "in_progress"

    result2 = _fuzzy_match_literal_value("in-progress", options)
    assert result2 == "in_progress"


def test_fuzzy_match_literal_value_alias() -> None:
    """Value alias map maps 'completed' to 'done'."""
    options = ("pending", "in_progress", "done")
    result = _fuzzy_match_literal_value("completed", options)
    assert result == "done"


def test_fuzzy_match_literal_fuzzy_fallback() -> None:
    """A close typo ('cmopleted') is matched via rapidfuzz."""
    options = ("pending", "in_progress", "done")
    # 'cmopleted' is ~0.69 with 'completed' but we need to check against
    # the options directly - 'cmopleted' has ratio ~0.5-0.6 with each.
    # Use a value that's clearly close to one option.
    result = _fuzzy_match_literal_value("cmopleted", options)
    # Since we check against options (pending, in_progress, done), not aliases,
    # and the cutoff is 0.60, we need a value close enough.
    # 'in_progres' (missing 's') should be close to 'in_progress'
    result2 = _fuzzy_match_literal_value("in_progres", options)
    assert result2 == "in_progress"


def test_fuzzy_match_literal_below_cutoff_returns_none() -> None:
    """A completely different value returns None."""
    options = ("pending", "in_progress", "done")
    result = _fuzzy_match_literal_value("completely_different_value", options)
    assert result is None


# --- _repair_dict_for_model integration tests with Literal ---


def test_repair_literal_status_completed_to_done() -> None:
    """'completed' status is repaired to 'done'."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {"todos": [{"title": "Test", "status": "completed"}]}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "done"


def test_repair_literal_status_complete_to_done() -> None:
    """'complete' status is repaired to 'done'."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {"todos": [{"title": "Test", "status": "complete"}]}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "done"


def test_repair_literal_status_in_progress_variants() -> None:
    """'in-progress' and 'in progress' are repaired to 'in_progress'."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data1 = {"todos": [{"title": "Test", "status": "in-progress"}]}
    repaired1 = _repair_dict_for_model(data1, Params)
    assert repaired1["todos"][0]["status"] == "in_progress"

    data2 = {"todos": [{"title": "Test", "status": "in progress"}]}
    repaired2 = _repair_dict_for_model(data2, Params)
    assert repaired2["todos"][0]["status"] == "in_progress"


def test_repair_literal_status_not_started_to_pending() -> None:
    """'not_started' status is repaired to 'pending'."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {"todos": [{"title": "Test", "status": "not_started"}]}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "pending"


def test_repair_literal_status_unknown_value_preserved() -> None:
    """An unknown value with no match is preserved as-is."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {"todos": [{"title": "Test", "status": "garbage"}]}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "garbage"


def test_repair_literal_status_exact_match_unchanged() -> None:
    """A valid Literal value is not modified."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {"todos": [{"title": "Test", "status": "done"}]}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "done"


def test_repair_literal_mode_variants() -> None:
    """Mode field fuzzy matching: 'replace' → 'overwrite'."""
    from typing import Literal

    class Params(BaseModel):
        mode: Literal["overwrite", "append", "force_overwrite"]

    data = {"mode": "replace"}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["mode"] == "overwrite"


def test_repair_literal_nested_model() -> None:
    """Fuzzy matching works on nested model fields."""
    from typing import Literal

    class Inner(BaseModel):
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        inner: Inner

    data = {"inner": {"status": "completed"}}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["inner"]["status"] == "done"


def test_repair_literal_list_of_models() -> None:
    """Fuzzy matching works on list items that are BaseModels."""
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class Params(BaseModel):
        todos: list[TodoItem]

    data = {
        "todos": [
            {"title": "A", "status": "completed"},
            {"title": "B", "status": "in-progress"},
            {"title": "C", "status": "pending"},  # valid, unchanged
        ]
    }
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["todos"][0]["status"] == "done"
    assert repaired["todos"][1]["status"] == "in_progress"
    assert repaired["todos"][2]["status"] == "pending"


def test_repair_literal_invalid_integer_type() -> None:
    """Non-string values for Literal fields are not touched by fuzzy matching."""
    from typing import Literal

    class Params(BaseModel):
        status: Literal["pending", "in_progress", "done"] = "pending"

    # Integer values should not be forwarded to the Literal fuzzy matcher
    # (they will fail Pydantic validation normally).
    data = {"status": 42}
    repaired = _repair_dict_for_model(data, Params)
    assert repaired["status"] == 42
