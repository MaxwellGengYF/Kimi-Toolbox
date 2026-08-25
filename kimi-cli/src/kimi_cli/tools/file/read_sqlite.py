"""SQLite browsing helpers for the ``read`` tool.

Uses ``apsw`` (per project performance rules) in read-only mode with WAL
sidecar initialization when needed.
"""

from __future__ import annotations

import asyncio
import regex as re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import apsw

__all__ = [
    "is_sqlite_path",
    "sniff_sqlite",
    "open_read_connection",
    "validate_where",
    "resolve_sqlite_selector",
    "render_table",
    "SqliteMode",
]

SQLITE_EXTENSIONS: tuple[str, ...] = (".sqlite", ".sqlite3", ".db", ".db3")

ROW_COUNT_PROBE_CAP = 50_000
MAX_RAW_QUERY_ROWS = 1000
DEFAULT_QUERY_LIMIT = 20
MAX_QUERY_LIMIT = 500

# Width constraints for ASCII table rendering.
MIN_COL_WIDTH = 3
MAX_COL_WIDTH = 40
MAX_TABLE_WIDTH = 120
VERTICAL_FALLBACK_COLUMNS = 19


class WhereValidationError(ValueError):
    """Raised when a WHERE fragment contains suspicious content."""


class ReadOnlyViolationError(ValueError):
    """Raised when raw SQL is not a SELECT."""


@dataclass(frozen=True, slots=True)
class SqliteMode:
    """Resolved mode for a SQLite read request."""

    kind: str  # "list", "schema", "query", "raw"
    table: str | None = None
    where: str | None = None
    order: str | None = None
    limit: int = DEFAULT_QUERY_LIMIT
    offset: int = 0
    query: str | None = None


def is_sqlite_path(path: str) -> bool:
    return Path(path).suffix.lower() in SQLITE_EXTENSIONS


def sniff_sqlite(header: bytes) -> bool:
    return header.startswith(b"SQLite format 3\x00")


def _wal_sidecar_missing(path: Path) -> bool:
    """Return True when the DB is in WAL mode but sidecars are absent."""
    try:
        with open(path, "rb") as f:
            f.seek(18)
            version_bytes = f.read(2)
        if len(version_bytes) < 2:
            return False
        # SQLite write version at offset 18; value 2 means WAL.
        return version_bytes[0] == 2 and not (path.parent / (path.name + "-wal")).exists()
    except OSError:
        return False


def open_read_connection(path: str | Path) -> apsw.Connection:
    """Open a read-only, query-only apsw connection.

    If the database is in WAL mode and the -wal/-shm sidecars are missing,
    first open once with READWRITE (never CREATE) to let SQLite create them,
    then reopen read-only.
    """
    p = Path(path)
    flags = apsw.SQLITE_OPEN_READONLY
    if _wal_sidecar_missing(p):
        rw_flags = apsw.SQLITE_OPEN_READWRITE
        try:
            conn = apsw.Connection(str(p), flags=rw_flags)
            conn.close()
        except apsw.CantOpenError:
            pass

    conn = apsw.Connection(str(p), flags=flags)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 3000")
    return conn


def _split_order(order: str | None, columns: list[str]) -> tuple[str, str] | None:
    """Parse ``sql_order`` as 'col' or 'col:asc|desc'."""
    if not order:
        return None
    order = order.strip()
    if ":" in order:
        col, direction = order.split(":", 1)
    else:
        col, direction = order, "asc"
    col = col.strip().strip('"')
    direction = direction.strip().lower()
    if direction not in {"asc", "desc"}:
        direction = "asc"
    if col not in columns:
        return None
    return col, direction


_WHERE_VIOLATION_PATTERN = re.compile(
    r";|--|/\*|\*/|\b(limit|offset|union|intersect|except|attach|detach|pragma)\b",
    re.IGNORECASE,
)


def validate_where(fragment: str) -> None:
    """Reject WHERE fragments containing statement terminators, comments, or
    pagination/control keywords outside quoted identifiers."""
    if not fragment:
        return
    if _WHERE_VIOLATION_PATTERN.search(fragment):
        raise WhereValidationError(
            "WHERE clause contains disallowed characters or keywords (; -- /* */ "
            "LIMIT OFFSET UNION INTERSECT EXCEPT ATTACH DETACH PRAGMA)."
        )


def _quote_identifier(ident: str) -> str:
    """Quote an identifier with double quotes, escaping internal quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _table_columns(conn: apsw.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def _table_exists(conn: apsw.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _estimate_row_count(conn: apsw.Connection, table: str) -> tuple[str, int]:
    """Return (exact|estimate|atLeast, count) for a table."""
    try:
        cur = conn.execute(
            "SELECT n FROM sqlite_stat1 WHERE tbl=? AND idx IS NULL LIMIT 1",
            (table,),
        )
        row = cur.fetchone()
        if row:
            text = str(row[0])
            # sqlite_stat1 stores estimates like "1000000" or "100000 50000 1".
            first = text.split()[0]
            try:
                return ("estimate", int(first))
            except ValueError:
                pass
    except apsw.Error:
        pass

    # Bounded COUNT(*) probe.
    try:
        cur = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM {_quote_identifier(table)} LIMIT ?)",
            (ROW_COUNT_PROBE_CAP + 1,),
        )
        count = cur.fetchone()[0]
        if count > ROW_COUNT_PROBE_CAP:
            return ("atLeast", ROW_COUNT_PROBE_CAP + 1)
        return ("exact", count)
    except apsw.Error as exc:
        return ("exact", 0)


def _list_tables(conn: apsw.Connection) -> list[tuple[str, str, int]]:
    """Return [(name, count_kind, count), ...] sorted case-insensitively."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name COLLATE NOCASE"
    ).fetchall()
    tables: list[tuple[str, str, int]] = []
    for (name,) in rows:
        kind, count = _estimate_row_count(conn, name)
        tables.append((name, kind, count))
    return tables


def _execute_read_query(
    conn: apsw.Connection,
    sql: str,
    bindings: tuple[Any, ...] | None = None,
    *,
    max_rows: int,
) -> tuple[list[str], list[list[Any]], bool]:
    """Execute a read-only query and return (columns, rows, truncated)."""
    if bindings is None:
        bindings = ()
    rows: list[list[Any]] = []
    truncated = False
    columns: list[str] = []
    cursor = conn.execute(sql, bindings)
    if cursor.description:
        columns = [str(d[0]) for d in cursor.description]
    for row in cursor:
        rows.append(list(row))
        if len(rows) >= max_rows:
            truncated = True
            break
    return columns, rows, truncated


def _render_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    text = str(value)
    # Sanitize tabs/newlines so table rendering isn't broken.
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return text


def _column_widths(columns: list[str], rows: list[list[Any]]) -> list[int]:
    widths = [max(MIN_COL_WIDTH, len(c)) for c in columns]
    for row in rows:
        for i, value in enumerate(row):
            if i >= len(widths):
                widths.append(MIN_COL_WIDTH)
            text = _render_value(value)
            widths[i] = max(widths[i], min(MAX_COL_WIDTH, len(text)))
    return widths


def render_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Render rows as an ASCII pipe table, or vertical blocks when too many columns."""
    if not columns:
        return "(no columns)"
    if len(columns) > VERTICAL_FALLBACK_COLUMNS:
        return _render_vertical_blocks(columns, rows)

    widths = _column_widths(columns, rows)
    lines: list[str] = []
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header = "|" + "|".join(f" {c[:w].ljust(w)} " for c, w in zip(columns, widths)) + "|"
    lines.extend([sep, header, sep])
    for row in rows:
        cells: list[str] = []
        for i, value in enumerate(row):
            if i >= len(widths):
                break
            text = _render_value(value)[: widths[i]]
            cells.append(f" {text.ljust(widths[i])} ")
        # Pad missing cells.
        for i in range(len(cells), len(widths)):
            cells.append(f" {' '.ljust(widths[i])} ")
        lines.append("|" + "|".join(cells) + "|")
    lines.append(sep)
    return "\n".join(lines)


def _render_vertical_blocks(columns: list[str], rows: list[list[Any]]) -> str:
    lines: list[str] = []
    for row_idx, row in enumerate(rows, 1):
        lines.append(f"--- row {row_idx} ---")
        for col, value in zip(columns, row):
            lines.append(f"{col}: {_render_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def render_table_list(tables: list[tuple[str, str, int]]) -> str:
    lines: list[str] = []
    for name, kind, count in tables:
        if kind == "exact":
            lines.append(f"{name} ({count} rows)")
        elif kind == "estimate":
            lines.append(f"{name} (~{count} rows estimated)")
        else:
            lines.append(f"{name} (>{count - 1} rows)")
    return "\n".join(lines)


def resolve_sqlite_selector(
    conn: apsw.Connection,
    *,
    sql_query: str | None,
    sql_table: str | None,
    sql_where: str | None,
    sql_order: str | None,
    sql_limit: int | None,
    sql_offset: int | None,
) -> SqliteMode:
    """Resolve rich-SQL params into a concrete mode."""
    if sql_query is not None:
        sql_query = sql_query.strip()
        if not sql_query:
            raise ValueError("sql_query cannot be empty")
        return SqliteMode(kind="raw", query=sql_query)

    if sql_table is None:
        return SqliteMode(kind="list")

    table = sql_table.strip()
    if not _table_exists(conn, table):
        raise ValueError(f"Table not found: {table!r}")

    # Schema mode: table requested with no pagination/filter/order.
    if sql_where is None and sql_order is None and sql_limit is None and sql_offset is None:
        return SqliteMode(kind="schema", table=table)

    columns = _table_columns(conn, table)
    where = ""
    if sql_where:
        validate_where(sql_where)
        where = f"WHERE {sql_where}"

    order_clause = ""
    parsed_order = _split_order(sql_order, columns)
    if parsed_order:
        col, direction = parsed_order
        order_clause = f"ORDER BY {_quote_identifier(col)} {direction.upper()}"

    limit = max(1, min(sql_limit or DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT))
    offset = max(0, sql_offset or 0)

    return SqliteMode(
        kind="query",
        table=table,
        where=where,
        order=order_clause,
        limit=limit,
        offset=offset,
    )


def _first_keyword(sql: str) -> str:
    # Strip leading comments and whitespace; return first identifier-ish token.
    cleaned = re.sub(r"--[^\n]*", "", sql)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    match = re.match(r"([A-Za-z]+)", cleaned)
    return match.group(1).lower() if match else ""


def _validate_raw_select(sql: str) -> None:
    keyword = _first_keyword(sql)
    if keyword != "select":
        raise ReadOnlyViolationError(
            "Only SELECT queries are allowed for sql_query."
        )


def execute_sqlite_read(
    path: str | Path,
    *,
    sql_query: str | None = None,
    sql_table: str | None = None,
    sql_where: str | None = None,
    sql_order: str | None = None,
    sql_limit: int | None = None,
    sql_offset: int | None = None,
) -> tuple[str, str, bool]:
    """Execute a SQLite read request and return (output, message, is_error).

    This is a synchronous helper; read.py wraps it with ``asyncio.to_thread``.
    """
    conn = open_read_connection(path)
    try:
        mode = resolve_sqlite_selector(
            conn,
            sql_query=sql_query,
            sql_table=sql_table,
            sql_where=sql_where,
            sql_order=sql_order,
            sql_limit=sql_limit,
            sql_offset=sql_offset,
        )

        if mode.kind == "list":
            tables = _list_tables(conn)
            output = render_table_list(tables)
            return (
                output,
                f"{len(tables)} table(s) in database. Path: {path}",
                False,
            )

        if mode.kind == "schema":
            schema_sql = ""
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (mode.table,),
            ):
                schema_sql = row[0] or ""
            sample_cols, sample_rows, _ = _execute_read_query(
                conn,
                f"SELECT * FROM {_quote_identifier(mode.table)} LIMIT 5",
                max_rows=5,
            )
            parts = [f"CREATE TABLE SQL for {mode.table}:", schema_sql or "(not available)", ""]
            if sample_rows:
                parts.append(f"Sample rows from {mode.table}:")
                parts.append(render_table(sample_cols, sample_rows))
            else:
                parts.append("(no sample rows)")
            return (
                "\n".join(parts),
                f"Schema for table {mode.table}. Path: {path}",
                False,
            )

        if mode.kind == "query":
            table = _quote_identifier(mode.table)
            # Total count for pagination footer.
            count_sql = f"SELECT COUNT(*) FROM {table}"
            bindings: tuple[Any, ...] = ()
            if mode.where:
                count_sql += f" {mode.where}"
                bindings = ()
            try:
                total = conn.execute(count_sql).fetchone()[0]
            except apsw.Error as exc:
                total = None

            sql = f"SELECT * FROM {table}"
            if mode.where:
                sql += f" {mode.where}"
            if mode.order:
                sql += f" {mode.order}"
            sql += " LIMIT ? OFFSET ?"
            columns, rows, truncated = _execute_read_query(
                conn, sql, (*bindings, mode.limit, mode.offset), max_rows=mode.limit
            )
            output = render_table(columns, rows)
            remaining = max(0, (total or 0) - mode.offset - len(rows))
            message = f"{len(rows)} row(s) read from table {mode.table}."
            if total is not None and remaining > 0:
                message += f" [{remaining} more row(s); use sql_offset={mode.offset + len(rows)} to continue]"
            message += f" Path: {path}"
            return (output, message, False)

        if mode.kind == "raw":
            _validate_raw_select(mode.query)
            columns, rows, truncated = _execute_read_query(
                conn, mode.query, max_rows=MAX_RAW_QUERY_ROWS
            )
            output = render_table(columns, rows)
            message = f"{len(rows)} row(s) returned by raw query."
            if truncated:
                message += f" (capped at {MAX_RAW_QUERY_ROWS} rows)"
            message += f" Path: {path}"
            return (output, message, False)

        raise ValueError(f"Unknown SQLite mode: {mode.kind}")
    finally:
        conn.close()


async def read_sqlite(
    path: str | Path,
    *,
    sql_query: str | None = None,
    sql_table: str | None = None,
    sql_where: str | None = None,
    sql_order: str | None = None,
    sql_limit: int | None = None,
    sql_offset: int | None = None,
) -> tuple[str, str, bool]:
    """Async wrapper around :func:`execute_sqlite_read`."""
    return await asyncio.to_thread(
        execute_sqlite_read,
        path,
        sql_query=sql_query,
        sql_table=sql_table,
        sql_where=sql_where,
        sql_order=sql_order,
        sql_limit=sql_limit,
        sql_offset=sql_offset,
    )
