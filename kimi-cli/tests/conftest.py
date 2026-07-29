"""Test configuration and fixtures."""

from __future__ import annotations

import os
import platform
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
import pytest_asyncio
from kaos import get_current_kaos, reset_current_kaos, set_current_kaos
from kaos.local import LocalKaos
from kaos.path import KaosPath
from kosong.chat_provider.mock import MockChatProvider
from pydantic import SecretStr

from kimi_cli.auth.oauth import OAuthManager
from kimi_cli.background import BackgroundTaskManager
from kimi_cli.config import Config, SearchConfig, get_default_config
from kimi_cli.llm import ALL_MODEL_CAPABILITIES, LLM
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.notifications import NotificationManager
from kimi_cli.session import Session
from kimi_cli.session_state import SessionState
from kimi_cli.soul.agent import BuiltinSystemPromptArgs, LaborMarket, Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.soul.denwarenji import DenwaRenji
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.subagents import AgentTypeDefinition, ToolPolicy
from kimi_cli.tools.agent import Agent as AgentTool
from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.hash_line import HashEdit, HashLine, HashRead
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import EditFile
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.todo import TodoList
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb
from kimi_cli.utils.environment import Environment
from kimi_cli.wire.file import WireFile


@pytest.fixture
def config() -> Config:
    """Create a Config instance."""
    conf = get_default_config()
    conf.services.search = SearchConfig(
        base_url="https://api.kimi.com/coding/v1/search",
        api_key=SecretStr("test-api-key"),
    )
    return conf


@pytest.fixture
def llm() -> LLM:
    """Create a LLM instance."""
    return LLM(
        chat_provider=MockChatProvider([]),
        max_context_size=100_000,
        capabilities=ALL_MODEL_CAPABILITIES,
    )


@pytest.fixture
def temp_work_dir() -> Generator[KaosPath]:
    """Create a temporary working directory for tests."""
    import platform
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=platform.system() == "Windows") as tmpdir:
        original_cwd = Path.cwd()
        p = Path(tmpdir).resolve()
        os.chdir(p)
        token = set_current_kaos(LocalKaos())
        try:
            yield KaosPath.unsafe_from_local_path(p)
        finally:
            reset_current_kaos(token)
            os.chdir(original_cwd)


@pytest.fixture
def temp_share_dir() -> Generator[Path]:
    """Create a temporary shared directory for tests."""
    import platform
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=platform.system() == "Windows") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def builtin_args(temp_work_dir: KaosPath) -> BuiltinSystemPromptArgs:
    """Create builtin arguments with temporary work directory."""
    return BuiltinSystemPromptArgs(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=temp_work_dir,
        KIMI_WORK_DIR_LS="Test ls content",
        KIMI_AGENTS_MD="Test agents content",
        KIMI_SKILLS="No skills found.",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS="macOS",
        KIMI_SHELL="bash (`/bin/bash`)",
    )


@pytest.fixture
def denwa_renji() -> DenwaRenji:
    """Create a DenwaRenji instance."""
    return DenwaRenji()


@pytest.fixture
def session(temp_work_dir: KaosPath, temp_share_dir: Path) -> Session:
    """Create a Session instance."""
    return Session(
        id="test",
        work_dir=temp_work_dir,
        work_dir_meta=WorkDirMeta(path=str(temp_work_dir), kaos=get_current_kaos().name),
        context_file=temp_share_dir / "context.jsonl",
        wire_file=WireFile(path=temp_share_dir / "wire.jsonl"),
        state=SessionState(),
        title="Test Session",
        updated_at=0.0,
        custom_data={},
        custom_config={},
    )


@pytest.fixture
def approval() -> Approval:
    """Create a Approval instance."""
    return Approval(yolo=True)


@pytest.fixture
def labor_market() -> LaborMarket:
    """Create a LaborMarket instance."""
    return LaborMarket()


@pytest.fixture
def environment() -> Environment:
    """Create an Environment instance."""
    if platform.system() == "Windows":
        return Environment(
            os_kind="Windows",
            os_arch="x86_64",
            os_version="1.0",
            shell_name="pwsh",
            shell_path=KaosPath(r"C:\Program Files\Git\bin\bash.exe"),
        )
    else:
        return Environment(
            os_kind="Unix",
            os_arch="aarch64",
            os_version="1.0",
            shell_name="bash",
            shell_path=KaosPath("/bin/bash"),
        )


@pytest.fixture
def runtime(
    config: Config,
    llm: LLM,
    builtin_args: BuiltinSystemPromptArgs,
    denwa_renji: DenwaRenji,
    session: Session,
    approval: Approval,
    labor_market: LaborMarket,
    environment: Environment,
) -> Runtime:
    """Create a Runtime instance."""
    notifications = NotificationManager(
        session.context_file.parent / "notifications", config.notifications
    )
    rt = Runtime(
        config=config,
        llm=llm,
        builtin_args=builtin_args,
        denwa_renji=denwa_renji,
        session=session,
        approval=approval,
        labor_market=labor_market,
        environment=environment,
        notifications=notifications,
        background_tasks=BackgroundTaskManager(
            session,
            config.background,
            notifications=notifications,
        ),
        skills={},
        oauth=OAuthManager(config),
        additional_dirs=[],
        skills_dirs=[],
        role="root",
    )
    rt.labor_market.add_builtin_type(
        AgentTypeDefinition(
            name="mocker",
            description="The mock agent for testing purposes.",
            agent_file=Path("/tmp/mocker-agent.yaml"),
            tool_policy=ToolPolicy(mode="inherit"),
        )
    )
    return rt


@pytest.fixture
def toolset() -> KimiToolset:
    return KimiToolset()


@contextmanager
def tool_call_context(tool_name: str) -> Generator[None]:
    """Create a tool call context."""
    from kimi_cli.soul.toolset import current_tool_call
    from kimi_cli.wire.types import ToolCall

    token = current_tool_call.set(
        ToolCall(id="test", function=ToolCall.FunctionBody(name=tool_name, arguments=None))
    )
    try:
        yield
    finally:
        current_tool_call.reset(token)


@pytest.fixture
def agent_tool(runtime: Runtime) -> AgentTool:
    """Create an Agent tool instance."""
    return AgentTool(runtime)


@pytest.fixture
def todo_list_tool(runtime: Runtime) -> TodoList:
    """Create a TodoList tool instance."""
    return TodoList(runtime)


@pytest.fixture
def read_file_tool(runtime: Runtime, session: Session) -> ReadFile:
    """Create a ReadFile tool instance."""
    return ReadFile(runtime, session)


@pytest.fixture
def read_media_file_tool(runtime: Runtime) -> ReadMediaFile:
    """Create a ReadMediaFile tool instance."""
    return ReadMediaFile(runtime)


@pytest.fixture
def glob_tool(runtime: Runtime) -> Glob:
    """Create a Glob tool instance."""
    return Glob(runtime)


@pytest_asyncio.fixture
async def grep_tool(runtime: Runtime) -> Grep:
    """Create a Grep tool instance."""
    return Grep(runtime)


@pytest.fixture
def write_file_tool(runtime: Runtime, approval: Approval, session: Session) -> Generator[WriteFile]:
    """Create a WriteFile tool instance."""
    with tool_call_context("WriteFile"):
        yield WriteFile(runtime, approval, session)


@pytest.fixture
def edit_file_tool(runtime: Runtime, approval: Approval, session: Session) -> Generator[EditFile]:
    """Create a EditFile tool instance."""
    with tool_call_context("EditFile"):
        yield EditFile(runtime, approval, session)


@pytest.fixture
def hash_line_tool(runtime: Runtime, session: Session) -> HashLine:
    """Create a HashLine tool instance (backward-compat alias for HashRead)."""
    return HashLine(runtime, session)


@pytest.fixture
def hash_read_tool(runtime: Runtime, session: Session) -> HashRead:
    """Create a HashRead tool instance."""
    return HashRead(runtime, session)


@pytest.fixture
def hash_edit_tool(runtime: Runtime, session: Session) -> HashEdit:
    """Create a HashEdit tool instance."""
    return HashEdit(runtime, session)


@pytest.fixture
def search_web_tool(config: Config, runtime: Runtime) -> SearchWeb:
    """Create a SearchWeb tool instance."""
    return SearchWeb(config, runtime)


@pytest.fixture
def fetch_url_tool(config: Config, runtime: Runtime) -> FetchURL:
    """Create a FetchURL tool instance."""
    return FetchURL(config, runtime)


# misc fixtures


@pytest.fixture
def outside_file() -> Generator[Path]:
    """Return a path to a file outside the working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "outside_file.txt"
