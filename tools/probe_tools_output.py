"""Probe current Grep/Glob tool output to measure token usage."""
from __future__ import annotations

import asyncio
import platform
import tempfile
from pathlib import Path

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
from kimi_cli.subagents import AgentTypeDefinition, ToolPolicy
from kimi_cli.tools.file.glob import Glob, Params as GlobParams
from kimi_cli.tools.file.grep_local import Grep, Params as GrepParams
from kimi_cli.utils.environment import Environment
from kimi_cli.wire.file import WireFile


def make_runtime(work_dir: Path) -> Runtime:
    conf = get_default_config()
    conf.services.search = SearchConfig(
        base_url="https://api.kimi.com/coding/v1/search",
        api_key=SecretStr("test-api-key"),
    )
    llm = LLM(
        chat_provider=MockChatProvider([]),
        max_context_size=100_000,
        capabilities=ALL_MODEL_CAPABILITIES,
    )
    tmp = Path(tempfile.mkdtemp())
    env = Environment(
        os_kind=platform.system(),
        os_arch="x86_64" if platform.system() == "Windows" else "aarch64",
        os_version="1.0",
        shell_name="pwsh" if platform.system() == "Windows" else "bash",
        shell_path=KaosPath("/bin/bash"),
    )
    builtin = BuiltinSystemPromptArgs(
        KIMI_NOW="1970-01-01T00:00:00+00:00",
        KIMI_WORK_DIR=KaosPath(str(work_dir)),
        KIMI_WORK_DIR_LS="",
        KIMI_AGENTS_MD="",
        KIMI_SKILLS="",
        KIMI_ADDITIONAL_DIRS_INFO="",
        KIMI_OS=platform.system(),
        KIMI_SHELL="pwsh",
    )
    sess = Session(
        id="probe",
        work_dir=KaosPath(str(work_dir)),
        work_dir_meta=WorkDirMeta(path=str(work_dir), kaos=get_current_kaos().name),
        context_file=tmp / "context.jsonl",
        wire_file=WireFile(path=tmp / "wire.jsonl"),
        state=SessionState(),
        title="Probe",
        updated_at=0.0,
        custom_data={},
        custom_config={},
    )
    notifications = NotificationManager(tmp / "notifications", conf.notifications)
    rt = Runtime(
        config=conf,
        llm=llm,
        builtin_args=builtin,
        denwa_renji=DenwaRenji(),
        session=sess,
        approval=Approval(yolo=True),
        labor_market=LaborMarket(),
        environment=env,
        notifications=notifications,
        background_tasks=BackgroundTaskManager(sess, conf.background, notifications=notifications),
        skills={},
        oauth=OAuthManager(conf),
        additional_dirs=[],
        skills_dirs=[],
        role="root",
    )
    return rt


async def main() -> None:
    work_dir = Path(r"C:\dev\kimi-agent")
    kaos = LocalKaos()
    token = set_current_kaos(kaos)
    try:
        rt = make_runtime(work_dir)

        glob = Glob(rt)
        grep = Grep(rt)

        print("=" * 60)
        print("GLOB: pattern='*.py' directory=root, verbose=False")
        r = await glob(GlobParams(pattern="*.py", directory=str(work_dir)))
        print("message:", r.message[:200])
        print("output chars:", len(r.output), "| lines:", len(r.output.splitlines()))
        print("output head:", r.output[:400])
        print()

        print("=" * 60)
        print("GLOB: pattern='**/*.py' verbose=True")
        r = await glob(GlobParams(pattern="**/*.py", directory=str(work_dir), verbose=True))
        print("message:", r.message[:200])
        print("output chars:", len(r.output), "| lines:", len(r.output.splitlines()))
        print("output head:", r.output[:400])
        print()

        print("=" * 60)
        print("GREP files_with_matches: pattern='def ' path=src/kimix/tools head_limit=0")
        r = await grep(GrepParams(pattern="def ", path=r"src\kimix\tools", output_mode="files_with_matches", head_limit=0, deduplicate_output=True))
        print("message:", r.message[:300])
        print("output chars:", len(r.output), "| lines:", len(r.output.splitlines()))
        print("output head:", r.output[:600])
        print()

        print("=" * 60)
        print("GREP content: pattern='def ' path=src/kimix/tools/common.py line_number=True")
        r = await grep(GrepParams(pattern="def ", path=r"src\kimix\tools\common.py", output_mode="content", line_number=True, deduplicate_output=True))
        print("message:", r.message[:300])
        print("output chars:", len(r.output), "| lines:", len(r.output.splitlines()))
        print("output head:", r.output[:800])
        print()

        print("=" * 60)
        print("GREP content repeated: pattern='args:' path=src/kimix/tools/common.py")
        r = await grep(GrepParams(pattern="Args:", path=r"src\kimix\tools\common.py", output_mode="content", line_number=True, deduplicate_output=True))
        print("message:", r.message[:300])
        print("output chars:", len(r.output), "| lines:", len(r.output.splitlines()))
        print("output head:", r.output[:800])
    finally:
        reset_current_kaos(token)


if __name__ == "__main__":
    asyncio.run(main())
