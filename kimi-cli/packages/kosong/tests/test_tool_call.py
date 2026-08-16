import asyncio
import inspect
import json
from typing import override

from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from kosong.message import ToolCall
from kosong.tooling import (
    BriefDisplayBlock,
    CallableTool,
    CallableTool2,
    ParametersType,
    ToolError,
    ToolOk,
    ToolResult,
    ToolResultFuture,
    ToolReturnValue,
    _repair_dict_for_model,
)
from kosong.tooling.error import (
    ToolNotFoundError,
    ToolRuntimeError,
    ToolValidateError,
)
from kosong.tooling.simple import SimpleToolset


def test_callable_tool_int_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "integer",
        }

        @override
        async def __call__(self, test: int) -> ToolReturnValue:
            return ToolOk(output=f"Test tool called with {test}")

    tool = TestTool()
    assert asyncio.run(tool.call(1)) == ToolOk(output="Test tool called with 1")


def test_callable_tool_list_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "array",
            "items": {
                "type": "string",
            },
        }

        @override
        async def __call__(self, a: str, b: str) -> ToolReturnValue:
            return ToolOk(output="Test tool called with a and b")

    tool = TestTool()
    assert asyncio.run(tool.call(["a", "b"])) == ToolOk(output="Test tool called with a and b")


def test_callable_tool_dict_argument():
    class TestTool(CallableTool):
        name: str = "test"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }

        @override
        async def __call__(self, a: str, b: int) -> ToolReturnValue:
            return ToolOk(output=f"Test tool called with {a} and {b}")

    tool = TestTool()
    assert asyncio.run(tool.call({"a": "a", "b": 1})) == ToolOk(
        output="Test tool called with a and 1"
    )


def test_simple_toolset():
    class PlusTool(CallableTool):
        name: str = "plus"
        description: str = "This is a plus tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        @override
        async def __call__(self, a: int, b: int) -> "ToolReturnValue":  # noqa: UP037
            return ToolOk(output=str(a + b))

    class CompareTool(CallableTool):
        name: str = "compare"
        description: str = "This is a compare tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        @override
        async def __call__(self, a: int, b: int) -> "ToolReturnValue":  # noqa: UP037
            return ToolOk(output="greater" if a > b else "less" if a < b else "equal")

    class RaiseTool(CallableTool):
        name: str = "raise"
        description: str = "This is a raise tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> ToolReturnValue:
            raise Exception("test exception")

    class ErrorTool(CallableTool):
        name: str = "error"
        description: str = "This is a error tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> ToolReturnValue:
            return ToolError(message="test error", brief="Error")

    class InvalidReturnTypeTool(CallableTool):
        name: str = "invalid_return_type"
        description: str = "This is a invalid return type tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> str:  # type: ignore[reportIncompatibleMethodOverride]
            return "invalid return type"

    toolset = SimpleToolset([PlusTool()])
    toolset += CompareTool()
    toolset += RaiseTool()
    toolset.add(ErrorTool())
    assert toolset.tools[0].name == "plus"
    assert toolset.tools[1].name == "compare"
    assert toolset.tools[2].name == "raise"
    assert toolset.tools[3].name == "error"

    try:
        toolset += InvalidReturnTypeTool()
    except TypeError as e:
        assert str(e) == (
            "Expected tool `invalid_return_type` to return `ToolReturnValue`, "
            "but got `<class 'str'>`"
        )
    else:
        raise AssertionError("Expected TypeError")

    tool_calls = [
        ToolCall(
            id="1",
            function=ToolCall.FunctionBody(
                name="plus",
                arguments=json.dumps({"a": 1, "b": 2}),
            ),
        ),
        ToolCall(
            id="2",
            function=ToolCall.FunctionBody(
                name="compare",
                arguments='{"a": 1, "b": 2}',
            ),
        ),
        ToolCall(
            id="3",
            function=ToolCall.FunctionBody(
                name="plus",
                arguments='{"a": 1}',
            ),
        ),
        ToolCall(
            id="4",
            function=ToolCall.FunctionBody(
                name="raise",
                arguments=None,
            ),
        ),
        ToolCall(
            id="5",
            function=ToolCall.FunctionBody(
                name="not_found",
                arguments=None,
            ),
        ),
        ToolCall(
            id="6",
            function=ToolCall.FunctionBody(
                name="error",
                arguments=None,
            ),
        ),
    ]

    async def run() -> list[ToolResult]:
        futures: list[ToolResultFuture] = []
        for tool_call in tool_calls:
            result = toolset.handle(tool_call)
            if isinstance(result, ToolResult):
                future = ToolResultFuture()
                future.set_result(result)
                futures.append(future)
            else:
                futures.append(result)
        return await asyncio.gather(*futures)

    results = asyncio.run(run())
    assert results[0].tool_call_id == "1"
    assert results[0].return_value == ToolOk(output="3")
    assert results[1].return_value == ToolOk(output="less")
    assert isinstance(results[2].return_value, ToolValidateError)
    assert isinstance(results[3].return_value, ToolRuntimeError)
    assert isinstance(results[4].return_value, ToolNotFoundError)
    assert isinstance(results[5].return_value, ToolError)
    assert results[5].return_value.message == "test error"
    assert results[5].return_value.display == snapshot([BriefDisplayBlock(text="Error")])


def test_callable_tool_2():
    class TestParams(BaseModel):
        a: int = Field(description="The first argument")
        b: int = Field(default=0, description="The second argument")
        c: str = Field(default="", alias="-c", description="The third argument")

    class TestTool(CallableTool2[TestParams]):
        name: str = "test"
        description: str = "This is a test tool"
        params: type[TestParams] = TestParams

        @override
        async def __call__(self, params: TestParams) -> ToolReturnValue:
            return ToolOk(output=f"Test tool called with {params.a} and {params.b}")

    tool = TestTool()
    assert tool.base.name == "test"
    assert tool.base.description == "This is a test tool"
    assert tool.base.parameters == {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "The first argument"},
            "b": {"type": "integer", "description": "The second argument", "default": 0},
            "-c": {"type": "string", "description": "The third argument", "default": ""},
        },
        "required": ["a"],
    }

    assert asyncio.run(tool.call({"a": 1, "b": 2})) == ToolOk(
        output="Test tool called with 1 and 2"
    )
    assert asyncio.run(tool.call({"a": 1})) == ToolOk(output="Test tool called with 1 and 0")
    assert isinstance(asyncio.run(tool.call({"b": 2})), ToolValidateError)


def test_simple_toolset_sub():
    class TestParams(BaseModel):
        pass

    class TestTool(CallableTool2[TestParams]):
        name: str = "test"
        description: str = "This is a test tool"
        params: type[TestParams] = TestParams

        @override
        async def __call__(self, params: TestParams) -> ToolReturnValue:
            return ToolOk(output="Test tool called")

    toolset = SimpleToolset([TestTool()])
    assert len(toolset.tools) == 1
    toolset.remove(TestTool.name)
    assert len(toolset.tools) == 0


# Tests for both real type and string annotations support
# These tests verify that SimpleToolset works correctly in both scenarios:
# 1. When type annotations are actual type objects (normal case)
# 2. When type annotations are strings (with `from __future__ import annotations`)


def test_simple_toolset_with_real_type_annotation_callable_tool():
    """Test that SimpleToolset works with CallableTool when using real type annotation."""

    class TestTool(CallableTool):
        name: str = "test_real"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> ToolReturnValue:
            return ToolOk(output="test")

    # Verify the annotation is actually a type (not string)
    assert inspect.signature(TestTool().__call__).return_annotation is ToolReturnValue

    toolset = SimpleToolset()
    toolset += TestTool()
    assert len(toolset.tools) == 1
    assert toolset.tools[0].name == "test_real"


def test_simple_toolset_with_string_annotation_callable_tool():
    """Test that SimpleToolset works with CallableTool when using string annotation."""

    class TestTool(CallableTool):
        name: str = "test_str"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> "ToolReturnValue":  # noqa: UP037  # type: ignore[reportIncompatibleMethodOverride]
            return ToolOk(output="test")

    # Verify the annotation is actually a string
    assert isinstance(inspect.signature(TestTool().__call__).return_annotation, str)

    toolset = SimpleToolset()
    toolset += TestTool()
    assert len(toolset.tools) == 1
    assert toolset.tools[0].name == "test_str"


def test_simple_toolset_with_invalid_string_annotation_rejected():
    """Test that SimpleToolset rejects invalid string annotations."""

    class TestTool(CallableTool):
        name: str = "test_invalid"
        description: str = "This is a test tool"
        parameters: ParametersType = {
            "type": "object",
            "properties": {},
        }

        @override
        async def __call__(self) -> "InvalidType":  # noqa: F821, UP037  # type: ignore[reportUnknownParameterType]
            return ToolOk(output="test")  # type: ignore[return-value]

    tool_instance = TestTool()
    sig = inspect.signature(tool_instance.__call__)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    # Verify the annotation is actually a string
    assert isinstance(sig.return_annotation, str)

    toolset = SimpleToolset()
    try:
        toolset += TestTool()
        raise AssertionError("Expected TypeError for invalid string annotation")
    except TypeError as e:
        assert "InvalidType" in str(e)


def test_simple_toolset_with_real_type_annotation_callable_tool2():
    """Test that SimpleToolset works with CallableTool2 when using real type annotation."""

    class TestParams(BaseModel):
        value: int = Field(description="A test value")

    class TestTool(CallableTool2[TestParams]):
        name: str = "test2_real"
        description: str = "This is a test tool 2"
        params: type[TestParams] = TestParams

        @override
        async def __call__(self, params: TestParams) -> ToolReturnValue:
            return ToolOk(output=f"value: {params.value}")

    # Verify the annotation is actually a type (not string)
    assert inspect.signature(TestTool().__call__).return_annotation is ToolReturnValue

    toolset = SimpleToolset()
    toolset += TestTool()
    assert len(toolset.tools) == 1
    assert toolset.tools[0].name == "test2_real"


def test_simple_toolset_with_string_annotation_callable_tool2():
    """Test that SimpleToolset works with CallableTool2 when using string annotation."""

    class TestParams(BaseModel):
        value: int = Field(description="A test value")

    class TestTool(CallableTool2[TestParams]):
        name: str = "test2_str"
        description: str = "This is a test tool 2"
        params: type[TestParams] = TestParams

        @override
        async def __call__(self, params: TestParams) -> "ToolReturnValue":  # noqa: UP037
            return ToolOk(output=f"value: {params.value}")

    # Verify the annotation is actually a string
    assert isinstance(inspect.signature(TestTool().__call__).return_annotation, str)

    toolset = SimpleToolset()
    toolset += TestTool()
    assert len(toolset.tools) == 1
    assert toolset.tools[0].name == "test2_str"


async def _test_handle_async_with_string_annotation():
    """Helper async function to test tool handling with string annotation."""

    class TestTool(CallableTool):
        name: str = "add_str"
        description: str = "Add two numbers"
        parameters: ParametersType = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        @override
        async def __call__(self, a: int, b: int) -> "ToolReturnValue":  # noqa: UP037
            return ToolOk(output=str(a + b))

    # Verify the annotation is actually a string
    assert isinstance(inspect.signature(TestTool().__call__).return_annotation, str)

    toolset = SimpleToolset([TestTool()])
    tool_call = ToolCall(
        id="1",
        function=ToolCall.FunctionBody(
            name="add_str",
            arguments='{"a": 2, "b": 3}',
        ),
    )

    result = toolset.handle(tool_call)
    if asyncio.isfuture(result):
        result = await result
    return result


def test_simple_toolset_with_string_annotation_handle():
    """Test that tools with string annotations can be called correctly."""
    result = asyncio.run(_test_handle_async_with_string_annotation())
    assert result.return_value == ToolOk(output="5")


# ---------------------------------------------------------------------------
# Tests for generic argument repair (Option 3)
# ---------------------------------------------------------------------------


def test_repair_dict_common_alias():
    """content -> title mapping works via common aliases."""

    class Todo(BaseModel):
        title: str = Field(description="Title")
        status: str = Field(description="Status")

    repaired = _repair_dict_for_model({"content": "Buy milk", "status": "pending"}, Todo)
    assert repaired == {"title": "Buy milk", "status": "pending"}


def test_repair_dict_exact_alias():
    """Declared pydantic aliases are respected."""

    class Params(BaseModel):
        query: str = Field(alias="q")

    repaired = _repair_dict_for_model({"q": "hello"}, Params)
    assert repaired == {"query": "hello"}


def test_repair_dict_no_unnecessary_changes():
    """Valid dicts are returned unchanged."""

    class Params(BaseModel):
        title: str
        status: str

    original = {"title": "Buy milk", "status": "pending"}
    repaired = _repair_dict_for_model(original, Params)
    assert repaired == original


def test_repair_dict_nested_model():
    """Repair recurses into nested BaseModel fields."""

    class Inner(BaseModel):
        title: str

    class Outer(BaseModel):
        inner: Inner

    repaired = _repair_dict_for_model({"inner": {"content": "nested"}}, Outer)
    assert repaired == {"inner": {"title": "nested"}}


def test_repair_dict_list_of_models():
    """Repair recurses into list items that are BaseModels."""

    class Item(BaseModel):
        title: str

    class Params(BaseModel):
        items: list[Item]

    repaired = _repair_dict_for_model(
        {"items": [{"content": "a"}, {"content": "b"}]},
        Params,
    )
    assert repaired == {"items": [{"title": "a"}, {"title": "b"}]}


def test_repair_dict_union_list_single():
    """Repair handles list[T] | T | None annotations."""

    class Item(BaseModel):
        title: str

    class Params(BaseModel):
        items: list[Item] | Item | None = None

    repaired = _repair_dict_for_model({"items": {"content": "single"}}, Params)
    assert repaired == {"items": {"title": "single"}}


def test_callable_tool_2_repair_on_call():
    """CallableTool2.call() auto-repairs and succeeds when common alias is used."""

    class Todo(BaseModel):
        title: str = Field(description="Title")
        status: str = Field(default="pending")

    class Params(BaseModel):
        todos: list[Todo]

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=params.todos[0].title)

    tool = TestTool()
    result = asyncio.run(
        tool.call({"todos": [{"content": "Buy milk", "status": "done"}]})
    )
    assert result == ToolOk(output="Buy milk")


def test_callable_tool_2_repair_falls_back_to_error():
    """When repair cannot fix the arguments, original ValidationError is returned."""

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
    result = asyncio.run(tool.call({"completely_wrong": "Buy milk"}))
    assert isinstance(result, ToolValidateError)


# ---------------------------------------------------------------------------
# Tests for fuzzy field-name matching
# ---------------------------------------------------------------------------


def test_repair_dict_fuzzy_long_field():
    """Long field names (>=8 chars) are fuzzy-matched with cutoff 0.75."""

    class Params(BaseModel):
        line_offset: int = Field(default=1)
        output_path: str | None = Field(default=None)

    repaired = _repair_dict_for_model({"lineoff": 5, "outputh": "/tmp"}, Params)
    assert repaired == {"line_offset": 5, "output_path": "/tmp"}


def test_repair_dict_fuzzy_medium_field():
    """Medium field names (4-7 chars) are fuzzy-matched with cutoff 0.80."""

    class Params(BaseModel):
        model: str = Field(default="")

    repaired = _repair_dict_for_model({"modl": "gpt-4"}, Params)
    assert repaired == {"model": "gpt-4"}


def test_repair_dict_fuzzy_short_field_skipped():
    """Short field names (<4 chars) are not fuzzy-matched."""

    class Params(BaseModel):
        mod: str = Field(default="")

    repaired = _repair_dict_for_model({"md": "x"}, Params)
    assert repaired == {"md": "x"}


def test_repair_dict_fuzzy_best_match_wins():
    """When two missing fields match the same unmapped key, the strongest match wins."""

    class Params(BaseModel):
        case_insensitive: bool = Field(default=False)
        case_sensitive: bool = Field(default=False)

    # "case_insenstive" is a typo for case_insensitive (ratio 0.968)
    # It should go to case_insensitive, not case_sensitive (ratio 0.897)
    repaired = _repair_dict_for_model({"case_insenstive": True}, Params)
    assert repaired == {"case_insensitive": True}


def test_repair_dict_fuzzy_unknown_key_stays():
    """Keys that do not fuzzy-match any missing field are left untouched."""

    class Params(BaseModel):
        title: str

    original = {"title": "ok", "timeout": 30}
    repaired = _repair_dict_for_model(original, Params)
    assert repaired == original


def test_repair_dict_fuzzy_exact_match_untouched():
    """Exact matches are not affected by fuzzy logic."""

    class Params(BaseModel):
        line_offset: int
        char_offset: int

    original = {"line_offset": 1, "char_offset": 2}
    repaired = _repair_dict_for_model(original, Params)
    assert repaired == original


def test_callable_tool_2_fuzzy_on_call():
    """CallableTool2.call() auto-repairs via fuzzy matching and succeeds."""

    class Params(BaseModel):
        background: bool
        command: str

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=f"bg={params.background} cmd={params.command}")

    tool = TestTool()
    result = asyncio.run(tool.call({"backgroud": True, "commnd": "ls"}))
    assert result == ToolOk(output="bg=True cmd=ls")


# ---------------------------------------------------------------------------
# Tests for numeric value clamping
# ---------------------------------------------------------------------------


def test_repair_dict_clamp_ge_le():
    """Values exceeding ge/le bounds are clamped to the boundary."""

    class Params(BaseModel):
        timeout: int = Field(default=10, ge=3, le=300)

    # Above max → clamp to max
    assert _repair_dict_for_model({"timeout": 600}, Params) == {"timeout": 300}
    # Below min → clamp to min
    assert _repair_dict_for_model({"timeout": 1}, Params) == {"timeout": 3}
    # In range → unchanged
    assert _repair_dict_for_model({"timeout": 150}, Params) == {"timeout": 150}


def test_repair_dict_clamp_gt_lt_int():
    """Integer values exceeding gt/lt bounds are clamped just inside the boundary."""

    class Params(BaseModel):
        count: int = Field(default=50, gt=0, lt=100)

    # Above max (lt) → clamp to max - 1
    assert _repair_dict_for_model({"count": 100}, Params) == {"count": 99}
    # At max boundary (lt) → clamp to max - 1
    assert _repair_dict_for_model({"count": 100}, Params) == {"count": 99}
    # Below min (gt) → clamp to min + 1
    assert _repair_dict_for_model({"count": 0}, Params) == {"count": 1}
    # In range → unchanged
    assert _repair_dict_for_model({"count": 50}, Params) == {"count": 50}


def test_repair_dict_clamp_float():
    """Float values are clamped to inclusive boundaries."""

    class Params(BaseModel):
        ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    assert _repair_dict_for_model({"ratio": 2.5}, Params) == {"ratio": 1.0}
    assert _repair_dict_for_model({"ratio": -0.5}, Params) == {"ratio": 0.0}
    assert _repair_dict_for_model({"ratio": 0.75}, Params) == {"ratio": 0.75}


def test_repair_dict_clamp_skips_bool():
    """Boolean values are not treated as numbers for clamping."""

    class Params(BaseModel):
        enabled: bool
        timeout: int = Field(default=10, ge=3, le=300)

    repaired = _repair_dict_for_model({"enabled": True, "timeout": 600}, Params)
    assert repaired == {"enabled": True, "timeout": 300}


def test_repair_dict_clamp_no_constraints():
    """Values on fields without numeric constraints pass through unchanged."""

    class Params(BaseModel):
        name: str

    assert _repair_dict_for_model({"name": "hello"}, Params) == {"name": "hello"}


def test_repair_dict_clamp_nested_model():
    """Clamping recurses into nested BaseModel fields."""

    class Inner(BaseModel):
        timeout: int = Field(default=10, ge=3, le=300)

    class Outer(BaseModel):
        inner: Inner

    repaired = _repair_dict_for_model({"inner": {"timeout": 600}}, Outer)
    assert repaired == {"inner": {"timeout": 300}}


def test_callable_tool_2_clamp_on_call():
    """CallableTool2.call() auto-clamps out-of-range numeric values."""

    class Params(BaseModel):
        timeout: int = Field(ge=3, le=300)

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output=f"timeout={params.timeout}")

    tool = TestTool()
    result = asyncio.run(tool.call({"timeout": 600}))
    assert result == ToolOk(output="timeout=300")


def test_callable_tool2_empty_dict():
    """CallableTool2.call({}) should validate and call the tool successfully."""

    class Params(BaseModel):
        pass

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output="called")

    tool = TestTool()
    assert asyncio.run(tool.call({})) == ToolOk(output="called")


def test_callable_tool2_non_dict_arguments():
    """CallableTool2.call() should return ToolValidateError for non-object inputs."""

    class Params(BaseModel):
        pass

    class TestTool(CallableTool2[Params]):
        name: str = "test"
        description: str = "test"
        params: type[Params] = Params

        @override
        async def __call__(self, params: Params) -> ToolReturnValue:
            return ToolOk(output="called")

    tool = TestTool()
    for bad in (None, [], [("a", "b", "c")], "not an object"):
        result = asyncio.run(tool.call(bad))
        assert isinstance(result, ToolValidateError), f"unexpected result for {bad!r}: {result}"
        assert "JSON object" in result.message


def test_repair_dict_empty_input():
    """_repair_dict_for_model short-circuits on empty or None input."""

    class Params(BaseModel):
        pass

    assert _repair_dict_for_model({}, Params) == {}
    assert _repair_dict_for_model(None, Params) == {}


# ---------------------------------------------------------------------------
# Tests for Phase 1: Literal value validation (currently expected to fail)
# ---------------------------------------------------------------------------


def test_callable_tool2_rejects_invalid_literal_value():
    """When a Literal field receives an invalid value, CallableTool2.call()
    should fuzzy-match it to the closest valid option.

    "completed" should be normalized to "done" via value-level fuzzy matching.
    """
    from typing import Literal

    class TodoItem(BaseModel):
        title: str
        status: Literal["pending", "in_progress", "done"]

    class TodoParams(BaseModel):
        todos: list[TodoItem]

    class TestTodoTool(CallableTool2[TodoParams]):
        name: str = "TestTodo"
        description: str = "Test todo tool"
        params: type[TodoParams] = TodoParams

        @override
        async def __call__(self, params: TodoParams) -> ToolReturnValue:
            return ToolOk(output="ok")

    tool = TestTodoTool()
    result = asyncio.run(tool.call({"todos": [{"title": "Test", "status": "completed"}]}))

    # After Phase 2's value-level fuzzy matching, "completed" should be
    # normalized to "done" and the call should succeed.
    assert result == ToolOk(output="ok"), (
        f"Expected ToolOk for 'completed' status, got {result}"
    )
