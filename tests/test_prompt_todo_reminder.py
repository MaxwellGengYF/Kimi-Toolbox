from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kimi_cli.wire.types import TextPart

prompt_mod = importlib.import_module("kimix.utils.prompt")


@dataclass
class FakeStatus:
    context_usage: float
    context_tokens: int


class FakeToolset:
    def __init__(self, has_set_todo: bool = True) -> None:
        self.has_set_todo = has_set_todo

    def find(self, name: str) -> object | None:
        if name == "todo_write" and self.has_set_todo:
            return object()
        return None


class FakeAgent:
    def __init__(self, has_set_todo: bool = True) -> None:
        self.toolset = FakeToolset(has_set_todo=has_set_todo)


class FakeSoul:
    def __init__(self, has_set_todo: bool = True, closing_rounds: int | None = None) -> None:
        self.agent = FakeAgent(has_set_todo=has_set_todo)
        if closing_rounds is not None:
            from types import SimpleNamespace

            self._loop_control = SimpleNamespace(cli_closing_reminder_rounds=closing_rounds)


@dataclass
class TodoItemState:
    title: str
    status: str


class FakeState:
    def __init__(self, todos: list[TodoItemState] | None = None) -> None:
        self.todos = todos or []


class FakeCLISession:
    def __init__(self, todos: list[TodoItemState] | None = None) -> None:
        self.state = FakeState(todos=todos)


class FakeCLI:
    def __init__(self, has_set_todo: bool = True, todos: list[TodoItemState] | None = None, closing_rounds: int | None = None, current_prompt: str | None = None) -> None:
        self.soul = FakeSoul(has_set_todo=has_set_todo, closing_rounds=closing_rounds)
        self.session = FakeCLISession(todos=todos)
        self._runtime = type("FakeRuntime", (), {"role": "root", "current_prompt": current_prompt})()


class FakeSubTodoItemState:
    title: str
    status: str
    notes: str

    def __init__(self, title: str, status: str, notes: str = "") -> None:
        self.title = title
        self.status = status
        self.notes = notes


class FakeSessionWithCLI:
    def __init__(
        self,
        has_set_todo: bool = True,
        todos: list[TodoItemState] | None = None,
        context_usage: float = 0.125,
        context_tokens: int = 1024,
        closing_rounds: int | None = None,
        current_prompt: str | None = None,
    ) -> None:
        self._cli = FakeCLI(has_set_todo=has_set_todo, todos=todos, closing_rounds=closing_rounds, current_prompt=current_prompt)
        self.status = FakeStatus(context_usage=context_usage, context_tokens=context_tokens)
        self.cancelled = False
        self._cancel_event = None
        self._tmp_data = {}
        self.prompts: list[str] = []

    async def prompt(self, prompt: str, *, merge_wire_messages: bool = False) -> Any:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        yield TextPart(text="prompt output")

    def cancel(self) -> None:
        self.cancelled = True


class FakeSessionWithoutCLI:
    def __init__(self, context_usage: float = 0.125, context_tokens: int = 1024) -> None:
        self.status = FakeStatus(context_usage=context_usage, context_tokens=context_tokens)
        self.cancelled = False
        self._cancel_event = None
        self._tmp_data = {}
        self.prompts: list[str] = []

    async def prompt(self, prompt: str, *, merge_wire_messages: bool = False) -> Any:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        yield TextPart(text="prompt output")

    def cancel(self) -> None:
        self.cancelled = True


def _suppress_stream(monkeypatch: Any) -> None:
    monkeypatch.setattr(prompt_mod.base._stream, "colorful_print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod.base._stream, "print_word", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_mod, "_print_usage", lambda *args, **kwargs: None)


def test_reminder_injected_when_todos_unfinished(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithCLI(
        has_set_todo=True,
        todos=[
            TodoItemState(title="Analyze requirement", status="pending"),
            TodoItemState(title="Implement helper", status="in_progress"),
            TodoItemState(title="Run tests", status="done"),
        ],
        closing_rounds=2,
    )

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    # The fake session never updates todo statuses, so both the regular and the
    # strong follow-up reminder are injected before cleanup (2 rounds configured).
    assert len(session.prompts) == 3
    assert session.prompts[0] == "hello"
    reminder = session.prompts[1]
    assert "You have unfinished `todo_write` tasks" in reminder
    assert "- [pending] Analyze requirement" in reminder
    assert "- [in_progress] Implement helper" in reminder
    # Done todos are excluded from the reminder
    assert "- [done] Run tests" not in reminder

    strong_reminder = session.prompts[2]
    assert "CRITICAL" in strong_reminder
    assert (
        "Mark every remaining item `completed` with `todo_update` (or `todo_write` for bulk edits)"
        in strong_reminder
    )
    assert "- [pending] Analyze requirement" in strong_reminder
    assert "- [in_progress] Implement helper" in strong_reminder
    # Done todos are excluded from strong reminder too
    assert "- [done] Run tests" not in strong_reminder


def test_no_reminder_when_all_todos_done(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithCLI(
        has_set_todo=True,
        todos=[
            TodoItemState(title="Analyze requirement", status="done"),
            TodoItemState(title="Implement helper", status="done"),
        ],
    )

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session.prompts == ["hello"]


def test_no_reminder_when_todo_list_empty(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithCLI(has_set_todo=True, todos=[])

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session.prompts == ["hello"]


def test_no_reminder_when_set_todo_tool_absent(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithCLI(
        has_set_todo=False,
        todos=[TodoItemState(title="Analyze requirement", status="pending")],
    )

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session.prompts == ["hello"]


def test_prompt_async_works_without_cli_attribute(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithoutCLI()

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session.prompts == ["hello"]


def test_reminder_stops_when_todos_marked_done(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    todos = [
        TodoItemState(title="Analyze requirement", status="pending"),
    ]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)

    async def mark_done_prompt(self: Any, prompt: str, *, merge_wire_messages: bool = False) -> Any:
        if "unfinished" in prompt and "todo_write" in prompt:
            session._cli.session.state.todos[0].status = "done"
        self.last_prompt = prompt
        self.prompts.append(prompt)
        yield TextPart(text="prompt output")

    monkeypatch.setattr(FakeSessionWithCLI, "prompt", mark_done_prompt)

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert len(session.prompts) == 2
    assert session.prompts[0] == "hello"
    assert "You have unfinished `todo_write` tasks" in session.prompts[1]


def test_no_reminder_when_ensure_todo_finished_false(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    session = FakeSessionWithCLI(
        has_set_todo=True,
        todos=[
            TodoItemState(title="Analyze requirement", status="pending"),
            TodoItemState(title="Implement helper", status="in_progress"),
        ],
    )

    asyncio.run(
        prompt_mod.prompt_async(
            "hello", session=session, info_print=False, ensure_todo_finished=False
        )
    )

    assert session.prompts == ["hello"]


def test_reminder_includes_children(monkeypatch: Any) -> None:
    """Child todos appear indented in the reminder output."""
    _suppress_stream(monkeypatch)
    from types import SimpleNamespace

    todo = SimpleNamespace(
        title="Parent task",
        status="in_progress",
        notes="",
        children=[
            SimpleNamespace(title="Sub task A", status="pending", notes=""),
            SimpleNamespace(title="Sub task B", status="done", notes=""),
        ],
    )
    session = FakeSessionWithCLI(has_set_todo=True, todos=[todo])

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert len(session.prompts) >= 2
    reminder = session.prompts[1]
    assert "- [in_progress] Parent task" in reminder
    assert "  - [pending] Sub task A" in reminder
    # Done children are excluded from the reminder
    assert "Sub task B" not in reminder


def test_todos_are_cleared_after_prompt_async(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    todos = [
        TodoItemState(title="Analyze requirement", status="pending"),
        TodoItemState(title="Implement helper", status="in_progress"),
        TodoItemState(title="Run tests", status="done"),
    ]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session._cli.session.state.todos == []


class FakeCLISessionWithSave(FakeCLISession):
    def __init__(self, todos: list[TodoItemState] | None = None) -> None:
        super().__init__(todos=todos)
        self.saved = False

    def save_state(self) -> None:
        self.saved = True


class FakeRuntimeRoot:
    role: str = "root"


class FakeCLIRoot:
    def __init__(self, todos: list[TodoItemState] | None = None) -> None:
        self.session = FakeCLISessionWithSave(todos=todos)
        self._runtime = FakeRuntimeRoot()


class FakeSessionRoot:
    def __init__(self, todos: list[TodoItemState] | None = None) -> None:
        self._cli = FakeCLIRoot(todos=todos)


def test_root_todos_cleared_from_disk(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    todos = [TodoItemState(title="task", status="pending")]
    session = FakeSessionRoot(todos=todos)

    asyncio.run(prompt_mod._clear_session_todos(session))

    assert session._cli.session.state.todos == []
    assert session._cli.session.saved is True


class FakeSubagentStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def instance_dir(self, agent_id: str) -> Path:
        path = self._base_dir / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakeRuntimeSubagent:
    role: str = "subagent"

    def __init__(self, store: FakeSubagentStore, agent_id: str) -> None:
        self.subagent_store = store
        self.subagent_id = agent_id


class FakeCLISubagent:
    def __init__(self, state: FakeState, runtime: FakeRuntimeSubagent) -> None:
        self.session = FakeCLISession()
        self.session.state = state
        self._runtime = runtime


class FakeSessionSubagent:
    def __init__(self, state: FakeState, runtime: FakeRuntimeSubagent) -> None:
        self._cli = FakeCLISubagent(state, runtime)


def test_subagent_todos_cleared_from_disk(tmp_path: Path) -> None:
    store = FakeSubagentStore(tmp_path / "subagents")
    runtime = FakeRuntimeSubagent(store, "agent1")
    state = FakeState(todos=[TodoItemState(title="task", status="pending")])
    session = FakeSessionSubagent(state, runtime)

    state_file = store.instance_dir("agent1") / "state.json"
    state_file.write_text(json.dumps({"todos": [{"title": "old", "status": "pending"}], "other": "data"}))

    asyncio.run(prompt_mod._clear_session_todos(session))

    assert session._cli.session.state.todos == []
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data.get("todos") == []
    assert data.get("other") == "data"


def test_todos_cleared_even_when_reminder_fails(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)

    async def failing_prompt(self: Any, prompt: str, *, merge_wire_messages: bool = False) -> Any:
        if "unfinished" in prompt and "todo_write" in prompt:
            raise RuntimeError("reminder failed")
        self.last_prompt = prompt
        self.prompts.append(prompt)
        yield TextPart(text="prompt output")

    monkeypatch.setattr(FakeSessionWithCLI, "prompt", failing_prompt)

    todos = [
        TodoItemState(title="Analyze requirement", status="pending"),
    ]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    assert session._cli.session.state.todos == []


class FakeCLISubagentWithToolset(FakeCLISubagent):
    def __init__(self, state: FakeState, runtime: FakeRuntimeSubagent) -> None:
        super().__init__(state, runtime)
        self.soul = FakeSoul(has_set_todo=True)


class FakeSessionSubagentWithToolset:
    def __init__(self, state: FakeState, runtime: FakeRuntimeSubagent) -> None:
        self._cli = FakeCLISubagentWithToolset(state, runtime)


def test_subagent_reminder_reads_from_state_file(tmp_path: Path) -> None:
    store = FakeSubagentStore(tmp_path / "subagents")
    runtime = FakeRuntimeSubagent(store, "agent1")
    state = FakeState(todos=[])
    session = FakeSessionSubagentWithToolset(state, runtime)

    state_file = store.instance_dir("agent1") / "state.json"
    state_file.write_text(
        json.dumps({"todos": [{"title": "Subagent task", "status": "pending"}]})
    )

    reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))

    assert reminder is not None
    assert "Subagent task" in reminder
    assert "- [pending] Subagent task" in reminder


def test_subagent_reminder_reads_children_from_state_file(tmp_path: Path) -> None:
    """Nested children in subagent state files are rendered indented."""
    store = FakeSubagentStore(tmp_path / "subagents")
    runtime = FakeRuntimeSubagent(store, "agent1")
    state = FakeState(todos=[])
    session = FakeSessionSubagentWithToolset(state, runtime)

    state_file = store.instance_dir("agent1") / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "todos": [
                    {
                        "title": "Parent",
                        "status": "in_progress",
                        "children": [
                            {"title": "Child", "status": "pending"},
                            {"title": "Done child", "status": "done"},
                        ],
                    }
                ]
            }
        )
    )

    reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))

    assert reminder is not None
    assert "- [in_progress] Parent" in reminder
    assert "  - [pending] Child" in reminder
    assert "Done child" not in reminder


def test_export_todo_list_to_json(tmp_path: Path, monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    todos = [
        TodoItemState(title="Analyze requirement", status="pending"),
        TodoItemState(title="Implement helper", status="in_progress"),
        TodoItemState(title="Run tests", status="done"),
    ]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)
    export_path = tmp_path / "todos.json"

    asyncio.run(
        prompt_mod.prompt_async(
            "hello",
            session=session,
            info_print=False,
            export_todo_list_path=export_path,
        )
    )

    assert export_path.exists()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported == [
        {"title": "Analyze requirement", "status": "pending"},
        {"title": "Implement helper", "status": "in_progress"},
        {"title": "Run tests", "status": "done"},
    ]
    # Todos are cleared after exporting.
    assert session._cli.session.state.todos == []


def test_invalid_export_path_prints_error_and_clears(monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    errors: list[str] = []
    monkeypatch.setattr(prompt_mod.base, "print_error", errors.append)

    todos = [TodoItemState(title="task", status="pending")]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)

    asyncio.run(
        prompt_mod.prompt_async(
            "hello",
            session=session,
            info_print=False,
            export_todo_list_path=Path("todos.txt"),
        )
    )

    assert any("Invalid todo list export path" in e for e in errors)
    # Falls back to default clear behavior.
    assert session._cli.session.state.todos == []


def test_export_session_todos_for_subagent(tmp_path: Path) -> None:
    store = FakeSubagentStore(tmp_path / "subagents")
    runtime = FakeRuntimeSubagent(store, "agent1")
    state = FakeState(todos=[])
    session = FakeSessionSubagentWithToolset(state, runtime)

    state_file = store.instance_dir("agent1") / "state.json"
    state_file.write_text(
        json.dumps({"todos": [{"title": "Subagent task", "status": "pending"}]})
    )

    export_path = tmp_path / "exported.json"
    asyncio.run(prompt_mod._export_session_todos(session, export_path))

    assert export_path.exists()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported == [{"title": "Subagent task", "status": "pending"}]
    # Source todos must remain untouched.
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["todos"] == [{"title": "Subagent task", "status": "pending"}]


def test_export_todo_list_includes_children(tmp_path: Path, monkeypatch: Any) -> None:
    """Exported JSON preserves the `children` nesting field."""
    _suppress_stream(monkeypatch)
    from types import SimpleNamespace

    todos = [
        SimpleNamespace(
            title="Parent",
            status="in_progress",
            children=[
                SimpleNamespace(title="Child", status="pending"),
            ],
        )
    ]
    session = FakeSessionWithCLI(has_set_todo=True, todos=todos)
    export_path = tmp_path / "todos.json"

    asyncio.run(
        prompt_mod.prompt_async(
            "hello",
            session=session,
            info_print=False,
            export_todo_list_path=export_path,
        )
    )

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported == [
        {
            "title": "Parent",
            "status": "in_progress",
            "children": [{"title": "Child", "status": "pending"}],
        }
    ]


def test_reminder_renders_nested_children(monkeypatch: Any) -> None:
    """Multi-level children are rendered with increasing indentation."""
    _suppress_stream(monkeypatch)
    from types import SimpleNamespace

    todo = SimpleNamespace(
        title="Level 0",
        status="in_progress",
        notes="",
        children=[
            SimpleNamespace(
                title="Level 1",
                status="pending",
                notes="",
                children=[
                    SimpleNamespace(title="Level 2", status="pending", notes=""),
                ],
            ),
        ],
    )
    session = FakeSessionWithCLI(has_set_todo=True, todos=[todo])

    asyncio.run(prompt_mod.prompt_async("hello", session=session, info_print=False))

    reminder = session.prompts[1]
    assert "- [in_progress] Level 0" in reminder
    assert "  - [pending] Level 1" in reminder
    assert "    - [pending] Level 2" in reminder


class FakePlannerSessionForPlan:
    def __init__(self, plan_file: Path) -> None:
        self._cancel_event = None
        self.prompts: list[str] = []
        self._plan_file = plan_file
        self._custom_data: dict[str, Any] = {}

    def get_custom_data(self) -> dict[str, Any]:
        return self._custom_data

    async def prompt(self, prompt_str: str, *, merge_wire_messages: bool = False) -> Any:
        self.prompts.append(prompt_str)
        if not self._plan_file.exists():
            self._plan_file.write_text("# Plan\n\n1. Do thing\n", encoding="utf-8")
        yield TextPart(text="plan output")

    def cancel(self) -> None:
        pass


class FakeExecutionSessionForPlan:
    def __init__(self) -> None:
        self.prompts: list[str] = []


def test_prompt_plan_async_prompts_execution_agent(tmp_path: Path, monkeypatch: Any) -> None:
    _suppress_stream(monkeypatch)
    plan_file = tmp_path / "plan.md"
    planner_session = FakePlannerSessionForPlan(plan_file)
    execution_session = FakeExecutionSessionForPlan()

    prompt_async_calls: list[tuple[str, Any, dict[str, Any]]] = []

    async def fake_create_session_async(*args: Any, **kwargs: Any) -> Any:
        return planner_session

    def fake_create_default_session() -> Any:
        return execution_session

    async def fake_create_default_session_async(*args: Any, **kwargs: Any) -> Any:
        return execution_session

    async def fake_close_session_async(session: Any) -> None:
        pass

    async def fake_prompt_async(prompt_str: str, session: Any, **kwargs: Any) -> None:
        prompt_async_calls.append((prompt_str, session, kwargs))
        session.prompts.append(prompt_str)

    monkeypatch.setattr(prompt_mod, "_create_session_async", fake_create_session_async)
    monkeypatch.setattr(prompt_mod, "_create_default_session", fake_create_default_session)
    monkeypatch.setattr(prompt_mod, "_create_default_session_async", fake_create_default_session_async)
    monkeypatch.setattr(prompt_mod, "close_session_async", fake_close_session_async)
    monkeypatch.setattr(prompt_mod, "prompt_async", fake_prompt_async)
    # ``os.startfile`` only exists on Windows; create it so the code under
    # test can be exercised on any platform.
    monkeypatch.setattr(prompt_mod.os, "startfile", lambda _path: None, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    asyncio.run(prompt_mod.prompt_plan_async("test requirement", plan_file))

    # The planner was used to generate the plan.
    assert len(planner_session.prompts) >= 1

    # No planner export or todo import should happen
    planner_export_calls = [
        (p, s, k) for p, s, k in prompt_async_calls
        if s is planner_session and k.get("export_todo_list_path") is not None
    ]
    assert len(planner_export_calls) == 0

    # Execution prompts should implement and review the plan.
    assert any("implement the plan" in p for p in execution_session.prompts)
    assert any("Review this plan" in p for p in execution_session.prompts)


def test_prompt_plan_async_falls_back_to_main_provider_without_sub_providers(tmp_path: Path, monkeypatch: Any) -> None:
    """When no `sub_providers` are configured, the planner session falls back
    to the main provider settings instead of failing with an empty provider dict.
    """
    _suppress_stream(monkeypatch)
    plan_file = tmp_path / "plan.md"
    planner_session = FakePlannerSessionForPlan(plan_file)
    execution_session = FakeExecutionSessionForPlan()

    captured_kwargs: dict[str, Any] = {}

    async def fake_create_session_async(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return planner_session

    async def fake_create_default_session_async(*args: Any, **kwargs: Any) -> Any:
        return execution_session

    async def fake_close_session_async(session: Any) -> None:
        pass

    async def fake_prompt_async(prompt_str: str, session: Any, **kwargs: Any) -> None:
        session.prompts.append(prompt_str)

    monkeypatch.setattr(prompt_mod, "_create_session_async", fake_create_session_async)
    monkeypatch.setattr(prompt_mod, "_create_default_session_async", fake_create_default_session_async)
    monkeypatch.setattr(prompt_mod, "close_session_async", fake_close_session_async)
    monkeypatch.setattr(prompt_mod, "prompt_async", fake_prompt_async)
    monkeypatch.setattr(prompt_mod.os, "startfile", lambda _path: None, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    # No sub-providers configured at all — only a main provider.
    main_provider = {
        "model": "deepseek-v4-flash-official",
        "max_context_size": 1048576,
        "url": "http://v2.open.venus.oa.com/llmproxy",
        "type": "openai_legacy",
        "api_key": "key",
    }
    monkeypatch.setattr(prompt_mod.base, "_default_sub_providers", [])
    monkeypatch.setattr(prompt_mod.base, "_default_provider", dict(main_provider))

    asyncio.run(prompt_mod.prompt_plan_async("test requirement", plan_file))

    provider_dict = captured_kwargs.get("provider_dict", {})
    # The planner session must inherit the main provider settings.
    assert provider_dict.get("type") == "openai_legacy"
    assert provider_dict.get("model") == "deepseek-v4-flash-official"
    assert provider_dict.get("url") == "http://v2.open.venus.oa.com/llmproxy"
    # Loop-control overrides are still applied on top of the fallback.
    assert provider_dict.get("loop_control", {}).get("budget_reminder_enabled") is False


def test_prompt_plan_async_uses_planner_sub_provider_when_present(tmp_path: Path, monkeypatch: Any) -> None:
    """A configured planner sub-provider still wins over the main provider."""
    _suppress_stream(monkeypatch)
    plan_file = tmp_path / "plan.md"
    planner_session = FakePlannerSessionForPlan(plan_file)
    execution_session = FakeExecutionSessionForPlan()

    captured_kwargs: dict[str, Any] = {}

    async def fake_create_session_async(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return planner_session

    async def fake_create_default_session_async(*args: Any, **kwargs: Any) -> Any:
        return execution_session

    async def fake_close_session_async(session: Any) -> None:
        pass

    async def fake_prompt_async(prompt_str: str, session: Any, **kwargs: Any) -> None:
        session.prompts.append(prompt_str)

    monkeypatch.setattr(prompt_mod, "_create_session_async", fake_create_session_async)
    monkeypatch.setattr(prompt_mod, "_create_default_session_async", fake_create_default_session_async)
    monkeypatch.setattr(prompt_mod, "close_session_async", fake_close_session_async)
    monkeypatch.setattr(prompt_mod, "prompt_async", fake_prompt_async)
    monkeypatch.setattr(prompt_mod.os, "startfile", lambda _path: None, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    planner_provider = {
        "model": "planner-model",
        "max_context_size": 65536,
        "url": "http://planner.example.com",
        "type": "openai_legacy",
        "role": "planner",
    }
    monkeypatch.setattr(prompt_mod.base, "_default_sub_providers", [dict(planner_provider)])
    monkeypatch.setattr(
        prompt_mod.base, "_default_provider",
        {"model": "main-model", "max_context_size": 1048576, "url": "http://main.example.com", "type": "openai_legacy"},
    )

    asyncio.run(prompt_mod.prompt_plan_async("test requirement", plan_file))

    provider_dict = captured_kwargs.get("provider_dict", {})
    assert provider_dict.get("model") == "planner-model"
    assert provider_dict.get("url") == "http://planner.example.com"


class TestCurrentPromptInReminder:
    """Verify _maybe_build_todo_reminder prepends current_prompt."""

    def test_reminder_includes_current_prompt(self) -> None:
        """When runtime.current_prompt is set, it appears in the reminder."""
        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos, current_prompt="user request")

        reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))
        assert reminder is not None
        assert "Original request: user request" in reminder
        assert "You have unfinished" in reminder

    def test_reminder_no_current_prompt_no_prefix(self) -> None:
        """When runtime has no current_prompt, no prefix is injected."""
        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos)

        reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))
        assert reminder is not None
        assert "Original request:" not in reminder
        assert "You have unfinished" in reminder

    def test_reminder_no_runtime_no_prefix(self) -> None:
        """When runtime has no current_prompt attribute, no prefix."""
        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos)
        # Remove the _runtime attribute to test defensive code path
        del session._cli._runtime

        reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))
        assert reminder is not None
        assert "Original request:" not in reminder
        assert "You have unfinished" in reminder


class TestPromptAsyncSetsCurrentPrompt:
    """Verify prompt_async stores current_prompt on the runtime."""

    def test_sets_current_prompt_on_runtime(self, monkeypatch: Any) -> None:
        """After prompt_async, runtime.current_prompt equals the prompt_str."""
        _suppress_stream(monkeypatch)
        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos)

        asyncio.run(prompt_mod.prompt_async("hello world", session=session, info_print=False))

        assert getattr(session._cli._runtime, "current_prompt", None) == "hello world"

    def test_current_prompt_persists_through_prompt_flow(self, monkeypatch: Any) -> None:
        """current_prompt is set before the main prompt runs."""
        _suppress_stream(monkeypatch)

        captured_prompt_during_run: list[str] = []
        original_prompt = FakeSessionWithCLI.prompt

        async def intercept_prompt(self: Any, prompt: str, **kwargs: Any) -> Any:
            rt = getattr(self._cli, "_runtime", None)
            cp = getattr(rt, "current_prompt", None) if rt is not None else None
            captured_prompt_during_run.append(cp or "")
            async for msg in original_prompt(self, prompt, **kwargs):
                yield msg

        monkeypatch.setattr(FakeSessionWithCLI, "prompt", intercept_prompt)

        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos)

        asyncio.run(prompt_mod.prompt_async("my prompt", session=session, info_print=False))

        assert any("my prompt" == cp for cp in captured_prompt_during_run), \
            f"current_prompt should be set before prompt runs, got: {captured_prompt_during_run}"


class TestCurrentPromptTruncation:
    """Verify current_prompt is truncated when too long."""

    def test_short_prompt_not_truncated(self) -> None:
        """Short prompts are not truncated."""
        todos = [TodoItemState(title="task", status="pending")]
        session = FakeSessionWithCLI(todos=todos, current_prompt="short request")

        reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))
        assert reminder is not None
        assert "Original request: short request" in reminder
        assert "..." not in reminder

    def test_long_prompt_truncated(self) -> None:
        """Long prompts (>200 chars) are truncated with head+tail."""
        todos = [TodoItemState(title="task", status="pending")]
        long_prompt = "A" * 150 + "B" * 150  # 300 chars
        session = FakeSessionWithCLI(todos=todos, current_prompt=long_prompt)

        reminder = asyncio.run(prompt_mod._maybe_build_todo_reminder(session))
        assert reminder is not None
        assert "Original request: " in reminder
        # Should have truncation marker
        assert "..." in reminder
        # Should contain head
        assert "A" * 100 in reminder
        # Should contain tail
        assert "B" * 100 in reminder
        # Should NOT contain the full original
        assert "A" * 150 not in reminder
