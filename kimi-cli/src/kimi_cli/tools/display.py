from kosong.tooling import DisplayBlock
from pydantic import BaseModel

from kimi_cli.session_state import TodoStatus


class DiffDisplayBlock(DisplayBlock):
    """Display block for a file diff."""

    type: str = "diff"
    path: str
    old_text: str
    new_text: str
    old_start: int = 1
    new_start: int = 1
    is_summary: bool = False


class TodoDisplayItem(BaseModel):
    title: str
    status: TodoStatus
    notes: str | None = None
    code: str | None = None


class TodoDisplayBlock(DisplayBlock):
    """Display block for a todo list update."""

    type: str = "todo"
    items: list[TodoDisplayItem]


class ShellDisplayBlock(DisplayBlock):
    """Display block for a shell command."""

    type: str = "shell"
    language: str
    command: str


class BackgroundTaskDisplayBlock(DisplayBlock):
    """Display block for a background task."""

    type: str = "background_task"
    task_id: str
    kind: str
    status: str
    description: str
