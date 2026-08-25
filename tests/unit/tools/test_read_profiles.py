"""Tests for CPU/sample profile rendering."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from kaos.path import KaosPath
from kosong.tooling import ToolOk

from kimi_cli.tools.file.read import Params as ReadFileParams, ReadFile
from kimi_cli.tools.file.read_profiles import (
    is_cpuprofile_path,
    is_sample_profile_path,
    render_cpu_profile,
    render_sample_profile,
)

import pytest


@pytest.fixture
def read_tool(tmp_path: Path) -> ReadFile:
    runtime = MagicMock()
    runtime.builtin_args.KIMI_WORK_DIR = KaosPath(str(tmp_path))
    runtime.additional_dirs = []
    runtime.llm.capabilities = set()
    session = MagicMock()
    session.id = "test"
    session.custom_data = {}
    session.custom_config = {"config_json": {}}
    return ReadFile(runtime, session)


class TestProfileHelpers:
    def test_is_cpuprofile_path(self) -> None:
        assert is_cpuprofile_path("/tmp/x.cpuprofile")
        assert not is_cpuprofile_path("/tmp/x.txt")

    def test_is_sample_profile_path(self) -> None:
        assert is_sample_profile_path("/tmp/crash.sample.txt")
        assert not is_sample_profile_path("/tmp/crash.txt")


class TestProfileReadFile:
    async def test_cpuprofile_summary(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "x.cpuprofile"
        profile = {
            "startTime": 0,
            "endTime": 1000,
            "nodes": [
                {"id": 1, "callFrame": {"functionName": "(root)"}, "children": [2, 3]},
                {"id": 2, "callFrame": {"functionName": "work"}, "children": []},
                {"id": 3, "callFrame": {"functionName": "(idle)"}, "children": []},
            ],
            "samples": [2, 2, 2, 3],
            "timeDeltas": [250, 250, 250, 250],
        }
        f.write_text(json.dumps(profile))
        result = await read_tool(ReadFileParams(path=str(f)))
        assert isinstance(result, ToolOk)
        assert "V8 CPU profile" in result.output
        assert "work" in result.output
        assert "Summarized view" in result.output

    async def test_cpuprofile_raw(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "x.cpuprofile"
        f.write_text(json.dumps({"nodes": []}))
        result = await read_tool(ReadFileParams(path=str(f), profile_raw=True))
        assert isinstance(result, ToolOk)
        assert '"nodes"' in result.output

    async def test_sample_profile_summary(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "x.sample.txt"
        f.write_text(
            "Sampling process 123 every 1 millisecond\n"
            "Call graph:\n"
            "    1234 Thread_0\n"
            "    +   1235 _main\n"
            "    + ! 1236 _work\n"
            "    +   1235 _main\n"
            "    + ! 1236 _work\n"
        )
        result = await read_tool(ReadFileParams(path=str(f)))
        assert isinstance(result, ToolOk)
        assert "macOS sample profile" in result.output
        assert "_work" in result.output

    async def test_malformed_profile_falls_through(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "x.cpuprofile"
        f.write_text("not json")
        # Malformed profile falls back to plain text.
        result = await read_tool(ReadFileParams(path=str(f)))
        assert isinstance(result, ToolOk)
        assert "not json" in result.output
