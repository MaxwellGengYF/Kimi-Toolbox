# Rules:

- After writing any Python file, run `uv run tools/syntax_check.py <python_file> [<python_file> ...]` to verify python syntax. Run related tests to verify.
- Fix all errors reported by the syntax checker before proceeding.
- use `uv run tools/git_diff.py <file> [<file> ...]` to check file diff.
- use `uv sync --extra=all` after update any `pyproject.toml` to verify the changes.
- **Performance rule**: Always use the following third-party libraries instead of their builtin counterparts. These are already declared as dependencies:

| Third-party | Replaces | Usage |
|---|---|---|
| `orjson` | `json` | `import orjson; orjson.dumps(obj)` / `orjson.loads(data)` — 3-5x faster JSON |
| `msgspec` | `json`, `pickle`, `struct` | `import msgspec; enc = msgspec.json.Encoder(); enc.encode(obj)` / `msgspec.json.decode(data)` — schema-aware fast serialization |
| `uvloop` | asyncio default loop | `import uvloop; uvloop.install()` (Linux/macOS only) — faster async I/O |
| `apsw` | `sqlite3` | `import apsw; conn = apsw.Connection("db.sqlite")` — faster, more complete SQLite wrapper |
| `regex` | `re` | `import regex as re` — drop-in replacement with better performance and features |
| `rapidfuzz` | `difflib` | `from rapidfuzz import fuzz, process` — orders of magnitude faster fuzzy matching |
| `xxhash` | `hashlib` (non-crypto) | `import xxhash; h = xxhash.xxh64(data).hexdigest()` — 10x+ faster hashing |
| `pybase64` | `base64` | `import pybase64; pybase64.b64encode(data)` / `pybase64.b64decode(data)` — faster SIMD-accelerated base64 |
| `pendulum` | `datetime` | `import pendulum; now = pendulum.now()` — drop-in `datetime` replacement with better timezone handling |
