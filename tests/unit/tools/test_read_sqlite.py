"""Tests for SQLite browsing via ReadFile."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import apsw
import pytest

from kaos.path import KaosPath
from kosong.tooling import ToolOk, ToolError

from kimi_cli.tools.file.read import Params as ReadFileParams, ReadFile
from kimi_cli.tools.file.read_sqlite import validate_where, WhereValidationError


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


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db = tmp_path / "app.db"
    conn = apsw.Connection(str(db))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users (name) VALUES ('alice'), ('bob'), ('carol')")
    conn.close()
    return db


class TestSqliteHelpers:
    def test_validate_where_rejects_injection(self) -> None:
        validate_where("id > 10")
        with pytest.raises(WhereValidationError):
            validate_where("id > 10; DROP TABLE users")
        with pytest.raises(WhereValidationError):
            validate_where("id > 10 -- comment")
        with pytest.raises(WhereValidationError):
            validate_where("id > 10 LIMIT 1")


class TestSqliteReadFile:
    async def test_list_tables(self, read_tool: ReadFile, sample_db: Path) -> None:
        result = await read_tool(ReadFileParams(path=str(sample_db)))
        assert isinstance(result, ToolOk)
        assert "users" in result.output
        assert "3 rows" in result.output

    async def test_query_pagination(self, read_tool: ReadFile, sample_db: Path) -> None:
        result = await read_tool(
            ReadFileParams(path=str(sample_db), sql_table="users", sql_limit=2, sql_offset=0)
        )
        assert isinstance(result, ToolOk)
        assert "alice" in result.output
        assert "bob" in result.output
        assert "carol" not in result.output
        assert "1 more row" in result.message

    async def test_where_validation(self, read_tool: ReadFile, sample_db: Path) -> None:
        result = await read_tool(
            ReadFileParams(path=str(sample_db), sql_table="users", sql_where="id > 10; DROP")
        )
        assert result.is_error
        assert "disallowed" in result.message.lower()

    async def test_raw_query_cap(self, read_tool: ReadFile, sample_db: Path) -> None:
        result = await read_tool(
            ReadFileParams(path=str(sample_db), sql_query="SELECT * FROM users")
        )
        assert isinstance(result, ToolOk)
        assert "alice" in result.output

    async def test_raw_mutation_rejected(self, read_tool: ReadFile, sample_db: Path) -> None:
        result = await read_tool(
            ReadFileParams(path=str(sample_db), sql_query="DELETE FROM users")
        )
        assert result.is_error
        assert "Only SELECT" in result.message

    async def test_non_sqlite_falls_through(self, read_tool: ReadFile, tmp_path: Path) -> None:
        f = tmp_path / "not.db"
        f.write_text("hello world\n")
        result = await read_tool(ReadFileParams(path=str(f)))
        assert isinstance(result, ToolOk)
        assert "hello world" in result.output

    async def test_wal_database_readable(self, read_tool: ReadFile, tmp_path: Path) -> None:
        # A database in WAL mode (with -wal/-shm sidecars present) is readable.
        db = tmp_path / "wal.db"
        conn = apsw.Connection(str(db))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.close()
        result = await read_tool(ReadFileParams(path=str(db), sql_query="SELECT * FROM t"))
        assert isinstance(result, ToolOk)
        assert "1" in result.output
