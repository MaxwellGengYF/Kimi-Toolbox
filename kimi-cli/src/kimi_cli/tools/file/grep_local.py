"""
The local version of the Grep tool using ripgrep.
Be cautious that `KaosPath` is not used in this implementation.
"""

import asyncio
import concurrent.futures
import contextlib
import fnmatch
import heapq
import os
import platform
import regex as re
import shlex
import stat
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, override

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kaos.path import KaosPath
from kosong.tooling import (
    FIELD_ALIASES_FILE,
    FIELD_ALIASES_GENERAL,
    FIELD_ALIASES_WEB,
    CallableTool2,
    ToolError,
    ToolReturnValue,
    alias_note,
)
from pydantic import AliasChoices, BaseModel, Field, field_validator

from kimi_cli._ripgrep_common import (
    RG_VERSION,
    RG_BASE_URL,
    _detect_rg_target,
    _extract_rg_archive,
    _rg_archive_name,
    _rg_binary_name,
    _rg_download_url,
)
from kimi_cli._rtk_common import _rtk_binary_name
from kimi_cli.install import _RTK_DOWNLOAD_LOCK, _download_and_install_rtk
from kimi_cli.share import get_share_dir
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.file.grep_archive import (
    materialize_archive_members,
    parse_archive_path_candidates,
)
from kimi_cli.tools.file.grep_output import (
    format_grouped_output,
    group_lines_by_file,
    should_group,
)
from kimi_cli.tools.file.grep_recorder import record_grep_files
from kimi_cli.tools.file.grep_selectors import (
    GrepPathSpec,
    LineRange,
    expand_path_entries,
    is_line_in_ranges,
    selector_line_ranges,
    split_path_and_sel,
)
from kimi_cli.tools.file.micro_compress import (
    MicroCompressConfig,
    compress_lines as _mc_compress_lines,
)
from kimi_cli.tools.file.output_utils import (
    dedup_lines,
    fold_lines,
    parse_rtk_rg_output,
    truncate_line,
)
from kimi_cli.tools.utils import ToolResultBuilder
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import (
    is_within_workspace,
    kaos_path_from_user_input,
    local_path_for_cwd,
    normalize_user_path,
)
from kimi_cli.utils.sensitive import is_sensitive_file, sensitive_file_warning
from kimi_cli.vfs import VFS

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_TOOLS = _native_get_module("tools")

# Output mode map — only canonical values accepted
_OUTPUT_MODE_MAP: dict[str, Literal["files_with_matches", "count_matches", "content"]] = {
    "files_with_matches": "files_with_matches",
    "count_matches": "count_matches",
    "content": "content",
}


# Matches a regex ``\n`` escape (odd number of backslashes before ``n``).
# Even backslashes, e.g. ``\\n``, mean a literal backslash+n search.
_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _pattern_has_regex_newline(pattern: str) -> bool:
    """Return True when a search regex tries to match a newline.

    The tool normally runs in line-oriented mode, so newline regexes cannot
    match across lines.  Detect both a literal newline already decoded into
    the pattern and a regex ``\n`` escape (odd number of backslashes before
    ``n``).  Even backslashes, e.g. ``\\n``, mean a literal backslash+n
    search and should stay line-oriented.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and pattern.isascii():
        return _NATIVE_TOOLS.pattern_has_regex_newline(pattern)
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def _multiline_pattern(pattern: str) -> str:
    """Rewrite newline constructs in *pattern* so they also match CRLF.

    ripgrep's ``--multiline`` mode matches raw file bytes, so on CRLF files
    (the Windows default) a multi-line pattern written with ``\n`` silently
    finds nothing.  Rewriting newlines to ``\\r?\\n`` (optional carriage
    return) makes the same pattern match both LF and CRLF files.  Only
    called when multiline mode is active.
    """
    if _native_use_native("TOOLS") and _NATIVE_TOOLS is not None and pattern.isascii():
        return _NATIVE_TOOLS.multiline_pattern(pattern)
    if "\n" not in pattern and not _REGEX_NEWLINE_ESCAPE_RE.search(pattern):
        return pattern
    # Normalize explicit CRLF in the pattern, then rewrite real newlines and
    # regex ``\n`` escapes to ``\r?\n``.  Lambdas keep the replacement text
    # literal (re.sub would otherwise interpret ``\r``/``\n`` escapes).
    p = pattern.replace("\r\n", "\n")
    p = _REGEX_NEWLINE_ESCAPE_RE.sub(lambda _m: r"\r?\n", p)
    return p.replace("\n", r"\r?\n")


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    pattern: str = Field(
        description="Regular expression to search for (ripgrep syntax)."
    )
    path: str | list[str] = Field(
        description=(
            "File or directory to search. Defaults to the session workspace; "
            "a relative path resolves against it. Also accepts embedded "
            "line-range selectors (`file.py:50-100`, `file.py:50+10`, "
            "`file.py:301-`, `file.py:5-16,960-973`, `..` alias), archive "
            "members (`bundle.zip:src/foo.ts`, combined "
            "`bundle.zip:src/foo.ts:50-100`), and multi-entry strings "
            "(`\"src; tests\"`) or lists."
        ),
        default=".",
    )
    grouped: bool | None = Field(
        description=(
            "Group content-mode results by file with `# path` headers and "
            "`*N|`/` N|` match/context markers. None = auto: grouped only "
            "when a line-range selector or archive member is used; True "
            "force grouped; False force legacy `path:line:text` output."
        ),
        default=None,
    )
    record: bool = Field(
        description=(
            "Persist the deduplicated matched-file list (relative paths) in "
            "the session so a follow-up read/edit pass can operate on "
            "exactly the files this grep surfaced."
        ),
        default=True,
    )
    include: str | None = Field(
        validation_alias=AliasChoices("include", "glob"),
        description=(
            "One glob filter for which files to search (e.g. `*.ts`, "
            "`*.{js,jsx}`). Not a list; negation is not supported. "
            + alias_note("include", "glob", word=False)
        ),
        default=None,
    )
    output_mode: Literal["files_with_matches", "count_matches", "content"] = Field(
        description="Output format: 'files_with_matches', 'count_matches', or 'content'.",
        default="files_with_matches",
    )

    @field_validator("output_mode", mode="before")
    @classmethod
    def _validate_output_mode(cls, v: str) -> str:
        normalized = v.strip().lower().replace("-", "_")
        canonical = _OUTPUT_MODE_MAP.get(normalized)
        if canonical is None:
            raise ValueError(
                f"Invalid output_mode '{v}'. Must be 'files_with_matches', "
                "'count_matches', or 'content'."
            )
        return canonical

    before_context: int | None = Field(
        default=None,
        alias="-B",
        description="Lines before match (content mode only)."
    )
    after_context: int | None = Field(
        default=None,
        alias="-A",
        description="Lines after match (content mode only)."
    )
    context: int | None = Field(
        default=None,
        alias="-C",
        description="Lines around match (content mode only)."
    )
    line_number: bool = Field(
        default=True,
        alias="-n",
        description="Show line numbers (content mode only)."
    )
    ignore_case: bool = Field(
        default=False,
        alias="-i",
        description="Case-insensitive search."
    )
    type: str | None = Field(
        description="File type filter.",
        default=None,
    )
    head_limit: int | None = Field(
        description="Max results (0 = unlimited).",
        default=500,
        ge=0,
    )
    offset: int = Field(
        description="Skip first N results.",
        default=0,
        ge=0,
    )
    multiline: bool = Field(
        description="Multiline regex mode. Patterns containing a newline or a "
        "`\\n` regex escape automatically enable multiline mode.",
        default=False,
    )
    include_ignored: bool = Field(
        description="Include .gitignore files.",
        default=False,
    )
    timeout: int = Field(
        description="Maximum time in seconds to wait for the search to complete.",
        default=60,
        ge=1,
    )
    deduplicate_output: bool = Field(
        default=True,
        alias="token_kill",  # backward compat
        description="Deduplicate repeated output lines via rtk (token killer). "
        "Set to False to see raw, unfiltered output.",
    )
    max_output_lines: int = Field(
        default=500,
        alias="fold",
        ge=0,
        description=(
            "Maximum number of lines in the final tool output. Longer results "
            "are head+tail folded with an omitted-count marker and a summary "
            "in `message`. 0 = unlimited (the byte cap still applies). "
            "Applied after offset/head_limit pagination."
        ),
    )


RG_MAX_BUFFER = 20_000_000  # 20MB stdout/stderr buffer limit
RG_KILL_GRACE = 5  # seconds: SIGTERM -> SIGKILL
MAX_BYTES = 100 << 10  # 100KB
_RG_HEAD_LIMIT_MARGIN = 1000  # extra matches for content-mode --max-count
RG_RANGE_FETCH_CAP = 200_000  # per-file fetch cap when line ranges are used
_RG_DOWNLOAD_LOCK = asyncio.Lock()
_RG_CMD = "rg"

# rg content-line grammar with explicit match/context delimiter semantics:
#   match:   "path:LN:text"   context: "path-LN-text"   separator: "--"
_RG_CONTENT_LINE_RE = re.compile(r"^(.*?)([:\-])(\d+)\2(.*)$", re.DOTALL)


def parse_content_line(line: str) -> tuple[str, int, str, bool] | None:
    """Parse one rg content-mode line.

    Returns ``(path, line_no, text, is_match)`` or ``None`` for separators
    (``--``) and anything that does not parse. Match lines use ``:`` as the
    delimiter (``path:LN:text``); context lines use ``-`` (``path-LN-text``).
    """
    if line == "--":
        return None
    m = _RG_CONTENT_LINE_RE.match(line)
    if m is None:
        return None
    path, sep, line_no, text = m.group(1), m.group(2), m.group(3), m.group(4)
    if not path:
        return None
    return path, int(line_no), text, sep == ":"


def _env_with_shared_bin_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of environment with the shared ``bin`` directory first on PATH.

    The shared ``bin`` directory contains ``rg`` and ``rtk``.  Prepending it
    (and removing any duplicate entry elsewhere in PATH) guarantees that our
    binaries win over any globally-installed copies (``rg`` / ``rg.exe``).

    Args:
        env: Optional base environment dict. If None, ``os.environ`` is used.

    Returns:
        A new dict with ``PATH`` updated so the shared ``bin`` directory is first.
    """
    bin_dir = str(get_share_dir() / "bin")
    result = os.environ.copy() if env is None else env.copy()

    current_path = result.get("PATH", "")
    path_sep = os.pathsep
    entries = [e for e in current_path.split(path_sep) if e and e != bin_dir]
    result["PATH"] = path_sep.join([bin_dir] + entries)

    return result


def _find_existing_rtk(bin_name: str) -> Path | None:
    """Find rtk binary in the share bin directory only.

    Unlike ``kimi_cli.install._find_existing_rtk`` this intentionally ignores
    the bundled deps directory and the global PATH so that subprocess calls
    never rely on global resolution; the spawned argv always uses the
    absolute path of the binary Kimi manages in ``share/bin``.
    """
    share_bin = get_share_dir() / "bin" / bin_name
    if share_bin.is_file():
        return share_bin

    return None


async def _ensure_rtk_path() -> str:
    bin_name = _rtk_binary_name()
    existing = _find_existing_rtk(bin_name)
    if existing:
        return str(existing)

    async with _RTK_DOWNLOAD_LOCK:
        existing = _find_existing_rtk(bin_name)
        if existing:
            return str(existing)

        downloaded = await _download_and_install_rtk(bin_name)
        return str(downloaded)


def _find_existing_rg(bin_name: str) -> Path | None:
    """Find rg binary in the share bin directory only.

    The global PATH rg is intentionally ignored because it can be broken or
    incompatible; Kimi always uses the binary it manages in ``share/bin``.
    """
    share_bin = get_share_dir() / "bin" / bin_name
    if share_bin.is_file():
        return share_bin

    return None


async def _download_and_install_rg(bin_name: str) -> Path:
    import aiohttp

    from kimi_cli.utils.aiohttp import new_client_session

    target = _detect_rg_target()
    filename = _rg_archive_name(RG_VERSION, target)
    url = _rg_download_url(RG_VERSION, target)
    logger.info("Downloading ripgrep from {url}", url=url)

    share_bin_dir = get_share_dir() / "bin"
    share_bin_dir.mkdir(parents=True, exist_ok=True)
    destination = share_bin_dir / bin_name

    download_timeout = aiohttp.ClientTimeout(total=600, sock_read=60, sock_connect=15)
    async with new_client_session(timeout=download_timeout) as session:
        with tempfile.TemporaryDirectory(prefix="kimi-rg-") as tmpdir:
            archive_path = Path(tmpdir) / filename

            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    with open(archive_path, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1024 * 64):
                            if chunk:
                                fh.write(chunk)
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise RuntimeError("Failed to download ripgrep binary") from exc

            _extract_rg_archive(archive_path, destination, target, bin_name)

    logger.info("Installed ripgrep to {destination}", destination=destination)
    return destination


async def _ensure_rg_path() -> str:
    bin_name = _rg_binary_name()
    existing = _find_existing_rg(bin_name)
    if existing:
        return str(existing)

    async with _RG_DOWNLOAD_LOCK:
        existing = _find_existing_rg(bin_name)
        if existing:
            return str(existing)

        downloaded = await _download_and_install_rg(bin_name)
        return str(downloaded)


def _build_rg_args(
    rg_cmd: str,
    params: Params,
    *,
    single_threaded: bool = False,
    resolved_path: str | None = None,
    rtk_path: str | None = None,
    max_count_override: int | None = None,
    path_input: str | None = None,
) -> list[str]:
    """Build ripgrep command-line arguments from Params.

    ``rg_cmd`` is the bare executable name (``rg``).  Callers must ensure the
    shared ``bin`` directory is first on ``PATH`` (see
    ``_env_with_shared_bin_path``) so this resolves to the binary Kimi manages
    even if another ``rg`` / ``rg.exe`` exists globally.

    When ``rtk_path`` is set, the returned argv is prefixed with the absolute
    rtk path so rtk wraps rg (``[<abs rtk>, rg, ...]``) and deduplicates its
    output. rtk dispatches on the wrapped executable's stem, matching the
    proven pattern in ``kimix.tools.file.run``.
    """
    args: list[str] = [rg_cmd]

    # Fixed args
    args.append("--no-config")  # avoid user config adding slow options
    if params.output_mode != "content":
        args.extend(["--max-columns", "500"])
    args.append("--hidden")
    if params.include_ignored:
        args.append("--no-ignore")
    for vcs_dir in (".git", ".svn", ".hg", ".bzr", ".jj", ".sl"):
        args.extend(["--glob", f"!{vcs_dir}"])

    if single_threaded:
        args.extend(["-j", "1"])

    # Search options
    if params.ignore_case:
        args.append("--ignore-case")
    use_multiline = params.multiline or _pattern_has_regex_newline(params.pattern)
    if use_multiline:
        args.extend(["--multiline", "--multiline-dotall"])

    # Content display options (only for content mode)
    if params.output_mode == "content":
        if params.before_context is not None:
            args.extend(["--before-context", str(params.before_context)])
        if params.after_context is not None:
            args.extend(["--after-context", str(params.after_context)])
        if params.context is not None:
            args.extend(["--context", str(params.context)])
        # Always be explicit about line numbers: raw rg defaults to off, but
        # rtk's ``rg`` subcommand defaults to on.  Passing the flag explicitly
        # keeps behavior consistent whether rg is invoked bare or via rtk.
        if params.line_number:
            args.append("--line-number")
        else:
            args.append("--no-line-number")
        # Stop ripgrep early once we have enough matches for the requested
        # page. A generous margin is included so that sensitive-file
        # filtering still leaves enough results in the common case.
        # NB: when rtk wraps rg (content mode, deduplicate_output=True) it
        # caps per-file output at 25 lines anyway, so this margin is
        # pointless there — kept as-is for the plain-rg path (harmless).
        if max_count_override:
            # Ranged selectors widen the fetch budget so in-range hits are
            # not starved by out-of-range matches preceding them.
            args.extend(["--max-count", str(max_count_override)])
        elif params.head_limit:
            max_count = (params.offset or 0) + params.head_limit + _RG_HEAD_LIMIT_MARGIN
            args.extend(["--max-count", str(max_count)])

    # File filtering options
    if params.include:
        args.extend(["--glob", params.include])
    if params.type:
        args.extend(["--type", params.type])

    # Output mode
    if params.output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif params.output_mode == "count_matches":
        args.append("--count-matches")

    # Separate pattern from flags to avoid ambiguity (e.g. pattern starting with -)
    args.append("--")
    args.append(_multiline_pattern(params.pattern) if use_multiline else params.pattern)
    # Use the resolved path when available so ripgrep's output matches the
    # search_base used for prefix stripping (fixes Windows short/long path
    # mismatches with tempfile directories).
    if resolved_path:
        args.append(resolved_path)
    else:
        fallback = path_input if path_input is not None else (
            params.path if isinstance(params.path, str) else (
                params.path[0] if params.path else "."
            )
        )
        args.append(os.path.expanduser(normalize_user_path(fallback)))

    if rtk_path is not None:
        args = [rtk_path, *args]

    return args


def _format_cmd(
    params: Params,
    *,
    rg_cmd: str = _RG_CMD,
    rtk_path: str | None = None,
    path_input: str | None = None,
) -> str:
    """Format the equivalent ripgrep command string for display."""
    args = _build_rg_args(rg_cmd, params, rtk_path=rtk_path, path_input=path_input)
    if rtk_path is not None and args and args[0] == rtk_path:
        args[0] = "rtk"
    return shlex.join(args)


async def _read_stream(
    stream: asyncio.StreamReader,
    buffer: bytearray,
    limit: int,
    truncated_flag: list[bool] | None = None,
) -> bool:
    """Incrementally read from stream into buffer, up to limit bytes.

    After hitting the limit, continues draining the pipe (discarding data)
    so the child process doesn't block on a full pipe buffer.

    Returns True if output was truncated (exceeded limit).
    """
    truncated = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if len(buffer) < limit:
            needed = limit - len(buffer)
            buffer.extend(chunk[:needed])
            if len(chunk) > needed:
                truncated = True
                if truncated_flag is not None:
                    truncated_flag[0] = True
        else:
            truncated = True
            if truncated_flag is not None:
                truncated_flag[0] = True
    return truncated


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Two-phase kill: SIGTERM -> grace period -> SIGKILL."""
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=RG_KILL_GRACE)
    except TimeoutError:
        process.kill()
        await process.wait()


def _is_eagain(stderr: str) -> bool:
    return "os error 11" in stderr or "Resource temporarily unavailable" in stderr


# Windows reserved DOS device names (case-insensitive, with or without extension).
_WINDOWS_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"com{i}" for i in range(1, 10)
} | {
    f"lpt{i}" for i in range(1, 10)
}


def _is_windows_reserved_name(path: str) -> bool:
    """Check if any path component is a Windows reserved DOS device name."""
    if platform.system() != "Windows":
        return False
    normalized = os.path.normpath(path)
    for part in normalized.split(os.sep):
        if not part:
            continue
        stem = part.split(".")[0].lower()
        if stem in _WINDOWS_RESERVED_NAMES:
            return True
    return False


_RG_LINE_RE = re.compile(r"^(.*?)([:\-])(\d+)\2")


@lru_cache(maxsize=1024)
def _is_sensitive_cached(path: str) -> bool:
    """Cached wrapper for is_sensitive_file to avoid redundant checks."""
    return is_sensitive_file(path)


def _join_with_byte_limit(lines: list[str], max_bytes: int = MAX_BYTES) -> tuple[str, bool]:
    """Join lines with newlines, stopping when byte limit is reached.

    Returns (output, was_truncated).
    """
    result_lines: list[str] = []
    n_bytes = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        separator_bytes = 1 if result_lines else 0
        result_lines.append(line)
        n_bytes += separator_bytes + line_bytes
        if n_bytes >= max_bytes:
            return "\n".join(result_lines), True
    return "\n".join(result_lines), False


def _strip_path_prefix(lines: list[str], search_base: str) -> list[str]:
    """Strip search_base prefix from each line to produce relative paths."""
    # Normalize to forward slashes so Windows paths from ripgrep (which may
    # preserve the drive-letter slash from the input path while using
    # backslashes elsewhere) still match search_base from os.path.abspath.
    prefix = search_base.replace("\\", "/").rstrip("/")
    prefix_slash = prefix + "/"
    result: list[str] = []
    for line in lines:
        if line.replace("\\", "/").startswith(prefix_slash):
            result.append(line[len(prefix_slash):])
        else:
            result.append(line)
    return result


def _normalize_output_lines(lines: list[str], output_mode: str) -> list[str]:
    """No-op passthrough (paths kept in native OS format)."""
    return lines


def _rtk_fold_note(
    meta: dict[str, Any], *, original_path: str | None = None
) -> str | None:
    """Build a human-readable summary of rtk's fold markers.

    rtk folds long content-mode output and records what it hid in protocol
    lines; this turns that metadata into a message fragment the model can
    act on (e.g. ``tail -n +26 <log>`` to page through the full output).

    When ``original_path`` is provided, it is surfaced as a fallback the model
    can read to see the full, unfiltered rtk output.

    Returns ``None`` when no fold markers were present.
    """
    parts: list[str] = []
    for entry in meta.get("folded_files") or []:
        parts.append(f"{entry['count']} more lines in {entry['path']}")
    skipped = meta.get("skipped_files")
    if skipped:
        parts.append(f"{skipped} more files")
    if not parts:
        return None

    note = "rtk folded output: " + "; ".join(parts) + "."
    log: str | None = None
    folded = meta.get("folded_files") or []
    if folded and folded[-1].get("log"):
        last = folded[-1]
        log = last["log"]
        if last.get("start_line") is not None:
            log = f"tail -n +{last['start_line']} {last['log']}"
    elif meta.get("skipped_log"):
        log = meta["skipped_log"]
    if log:
        note += f" Full log: {log}"
    if original_path:
        note += f" Original output: {original_path.replace(chr(92), '/')}"
    return note


# Minimal type-to-extension mapping for common file types.
_TYPE_MAP: dict[str, list[str]] = {
    "py": [".py"],
    "js": [".js", ".jsx", ".mjs", ".cjs"],
    "ts": [".ts", ".tsx", ".mts", ".cts"],
    "rs": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h", ".hh", ".hxx"],
    "c": [".c", ".h"],
    "md": [".md", ".markdown"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "xml": [".xml"],
    "html": [".html", ".htm", ".xhtml"],
    "css": [".css", ".scss", ".sass", ".less"],
    "sh": [".sh", ".bash", ".zsh", ".fish"],
    "sql": [".sql"],
    "lua": [".lua"],
    "vim": [".vim"],
    "docker": ["Dockerfile"],
    "make": ["Makefile", ".mk"],
    "ruby": [".rb"],
    "php": [".php"],
    "cs": [".cs"],
}

# Directories skipped unconditionally (VCS) or when include_ignored=False.
_VCS_DIRS = {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"}

_IGNORED_DIRS = {
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".egg-info",
    ".idea",
    ".vscode",
    "target",
    "out",
    ".next",
    ".nuxt",
}

_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _should_skip_dir(dirname: str, include_ignored: bool) -> bool:
    if dirname in _VCS_DIRS:
        return True
    return not include_ignored and dirname in _IGNORED_DIRS


def _matches_type(file_path: Path, type_name: str | None) -> bool:
    if type_name is None:
        return True
    extensions = _TYPE_MAP.get(type_name)
    if extensions is None:
        return False
    name = file_path.name
    return any(name.endswith(ext) for ext in extensions)


def _matches_glob(file_path: Path, pattern: str | None) -> bool:
    if pattern is None:
        return True
    return fnmatch.fnmatch(file_path.name, pattern)


def _safe_getmtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except (OSError, ValueError):
        return 0.0


async def _safe_getmtime_async(path: str) -> float:
    try:
        return await asyncio.to_thread(os.path.getmtime, path)
    except (OSError, ValueError):
        return 0.0


@lru_cache(maxsize=128)
def _compile_regex_cached(pattern: str, flags: int) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _read_file_text(file_path: Path, vfs: VFS | None = None) -> str | None:
    """Read a file in a single pass: binary read, null-byte check, then decode."""
    if vfs is not None:
        with contextlib.suppress(ValueError):
            file_path = vfs.translate_path(file_path)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if _is_binary(data):
            return None
        return data.decode("utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[int]] = [list(sorted_intervals[0])]
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(m[0], m[1]) for m in merged]


@dataclass
class _GrepCtx:
    """Per-call selector/archive context threaded through post-processing."""

    prefix_base: str
    prefix_base_is_file: bool
    grouped: bool
    ranges: dict[str, list[LineRange]] = field(default_factory=dict)
    display_map: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_BARE_CONTENT_RE = re.compile(r"^(\d+)([:\-])")


def _reattach_single_file_prefix(lines: list[str], prefix: str) -> list[str]:
    """Re-attach the file path when rg searches a single file.

    Bare rg omits the path prefix for single-file targets (``2:text`` instead
    of ``path:2:text``); the line-stream pipeline needs full ``path:LN:text``
    lines for range filtering, grouping, and the recorder.
    """
    if not prefix:
        return lines
    out: list[str] = []
    for line in lines:
        m = _BARE_CONTENT_RE.match(line)
        if m is not None:
            sep = m.group(2)
            out.append(f"{prefix}{sep}{line}")
        else:
            out.append(line)
    return out


def _plain_ctx(search_path: Path) -> _GrepCtx:
    """Context for legacy (selector-free) searches: byte-identical pipeline."""
    return _GrepCtx(
        prefix_base=str(search_path),
        prefix_base_is_file=search_path.is_file(),
        grouped=False,
    )


def _glob_chars(s: str) -> bool:
    return any(ch in s for ch in "*?[")


def _display_path_for(abs_path: Path, cwd: Path) -> str:
    """Workspace-relative forward-slash display path (name when at root)."""
    try:
        rel = abs_path.relative_to(cwd)
        rel_str = str(rel).replace("\\", "/")
        return rel_str if rel_str != "." else abs_path.name
    except ValueError:
        return abs_path.name


async def _resolve_selector_specs(
    entries: list[str],
    work_dir,
    additional_dirs: list,
    scratch_dir: Path | None,
) -> tuple[_GrepCtx | None, list[str], list[GrepPathSpec], ToolError | None]:
    """Resolve path entries (possibly with selectors/archives) to search targets.

    Returns ``(ctx, resolved_paths, specs, error)``; ``ctx`` is None on error.
    ``resolved_paths[i]`` corresponds to ``specs[i]`` (scratch path for
    materialized archive members, absolute resolved path otherwise).
    """
    cwd = local_path_for_cwd(work_dir)
    specs: list[GrepPathSpec] = []
    for entry in entries:
        path_part, sel = split_path_and_sel(entry)
        try:
            ranges = selector_line_ranges(sel)
        except ValueError as exc:
            return None, [], [], ToolError(
                message=f"Invalid line-range selector `{entry}`: {exc}",
                brief="Invalid selector",
            )
        has_ranges = ranges is not None
        has_archive = bool(parse_archive_path_candidates(path_part))
        if has_ranges or has_archive:
            if _glob_chars(path_part):
                return None, [], [], ToolError(
                    message=(
                        "Line-range selector/archive member requires a single "
                        f"file, not a glob: `{entry}`."
                    ),
                    brief="Selector requires a file",
                )
            if not has_archive:
                probe = Path(
                    cwd / Path(normalize_user_path(path_part)).expanduser()
                )
                if probe.is_dir():
                    return None, [], [], ToolError(
                        message=(
                            "Line-range selector requires a single file, not a "
                            f"directory: `{entry}`."
                        ),
                        brief="Selector requires a file",
                    )
        specs.append(GrepPathSpec(original=entry, clean=path_part, ranges=ranges))

    notes: list[str] = []
    display_map: dict[str, str] = {}
    if any(parse_archive_path_candidates(s.clean) for s in specs):
        if scratch_dir is None:
            return None, [], [], ToolError(
                message="Internal error: archive search without scratch dir.",
                brief="Archive search error",
            )
        specs, display_map, unreadable = await materialize_archive_members(
            specs, cwd, scratch_dir
        )
        notes.extend(unreadable)
        if not specs:
            entries_str = ", ".join(f"`{e}`" for e in entries)
            detail = "; ".join(unreadable) if unreadable else "no readable members"
            return None, [], [], ToolError(
                message=(
                    f"Cannot search archive member(s): {entries_str} \u2014 read the "
                    f"member with `read <archive>:<member>`. Details: {detail}"
                ),
                brief="Unreadable archive members",
            )

    resolved: list[str] = []
    for spec in specs:
        raw = spec.clean
        if raw in display_map:  # already a scratch path
            resolved.append(raw)
            continue
        p = (cwd / Path(normalize_user_path(raw)).expanduser()).resolve()
        logical = KaosPath(str(p)).canonical()
        original_is_absolute = kaos_path_from_user_input(raw).is_absolute()
        if (
            not is_within_workspace(logical, work_dir, additional_dirs)
            and not original_is_absolute
        ):
            return None, [], [], ToolError(
                message=f"`{raw.replace(chr(92), '/')}` is outside the workspace.",
                brief="Path outside workspace",
            )
        resolved.append(str(p))
    ctx = _GrepCtx(
        prefix_base=str(cwd),
        prefix_base_is_file=False,
        grouped=False,
        display_map=display_map,
        notes=notes,
    )
    return ctx, resolved, specs, None


def _strip_key_for(path_arg: str, prefix_base: str) -> str:
    """Exactly mirror ``_strip_path_prefix`` for one path → display key."""
    norm = path_arg.replace("\\", "/")
    pb = prefix_base.replace("\\", "/").rstrip("/")
    if pb and norm.startswith(pb + "/"):
        return norm[len(pb) + 1 :]
    return norm


def _build_ranges_map(
    ctx: _GrepCtx,
    resolved: list[str],
    specs: list[GrepPathSpec],
    *,
    absolute_keys: bool,
) -> None:
    """Populate ``ctx.ranges`` keyed to how paths appear in the output stream.

    ``absolute_keys=False`` keys on the post-strip display path (native
    pipeline); ``absolute_keys=True`` keys on the absolute path (backup
    pipeline, before prefix stripping). Archive scratch entries are keyed on
    their ``archive:member`` display form in both cases.
    """
    for spec, path_arg in zip(specs, resolved, strict=True):
        if not spec.ranges:
            continue
        if path_arg in ctx.display_map:
            ctx.ranges[ctx.display_map[path_arg]] = list(spec.ranges)
        elif absolute_keys:
            ctx.ranges[path_arg] = list(spec.ranges)
        else:
            ctx.ranges[_strip_key_for(path_arg, ctx.prefix_base)] = list(spec.ranges)


def _remap_display(lines: list[str], display_map: dict[str, str]) -> list[str]:
    """Rewrite scratch-path prefixes to their original archive:member form."""
    if not display_map:
        return lines
    pairs = [
        (scratch.replace("\\", "/"), display)
        for scratch, display in display_map.items()
    ]
    out: list[str] = []
    for line in lines:
        norm = line.replace("\\", "/")
        for scratch_fwd, display in pairs:
            if norm.startswith(scratch_fwd):
                out.append(display + norm[len(scratch_fwd):])
                break
        else:
            out.append(line)
    return out


def _normalize_slashes_content(lines: list[str], output_mode: str) -> list[str]:
    """Normalize backslashes to forward slashes in the path prefix of rich
    output lines so range keys / display keys match on Windows."""
    if os.sep != "\\":
        return lines
    out: list[str] = []
    for line in lines:
        parsed = parse_content_line(line)
        if parsed is not None:
            path, line_no, text, is_match = parsed
            sep = ":" if is_match else "-"
            out.append(f"{path.replace(chr(92), '/')}{sep}{line_no}{sep}{text}")
        elif output_mode != "content":
            out.append(line.replace("\\", "/"))
        else:
            out.append(line)
    return out


def _range_filter_lines(
    lines: list[str], ranges_map: dict[str, list[LineRange]]
) -> list[str]:
    """Drop content matches/context outside the per-file ranges; prune orphan `--`."""
    if not ranges_map:
        return lines
    kept: list[str] = []
    for line in lines:
        parsed = parse_content_line(line)
        if parsed is None:
            kept.append(line)
            continue
        path, line_no, _text, _is_match = parsed
        spec_ranges = ranges_map.get(path)
        if spec_ranges is not None and not is_line_in_ranges(line_no, spec_ranges):
            continue
        kept.append(line)
    swept: list[str] = []
    for line in kept:
        if line == "--":
            if not swept or swept[-1] == "--":
                continue
        swept.append(line)
    while swept and swept[-1] == "--":
        swept.pop()
    return swept


def _collect_record_files(lines: list[str], output_mode: str) -> list[str]:
    """Distinct file paths (in stream order) from a post-strip output stream."""
    seen: dict[str, None] = {}
    if output_mode == "content":
        for line in lines:
            parsed = parse_content_line(line)
            if parsed is not None:
                seen[parsed[0]] = None
    else:
        for line in lines:
            idx = line.rfind(":") if output_mode == "count_matches" else -1
            seen[line[:idx] if idx > 0 else line] = None
    return list(seen)


def _text_in_ranges(text: str, ranges: list[LineRange]) -> str:
    """Extract the text of the in-range lines only (backup range windows)."""
    lines = text.split("\n")
    kept = [ln for i, ln in enumerate(lines, 1) if is_line_in_ranges(i, ranges)]
    return "\n".join(kept)


def _backup_extract_path(line: str, output_mode: str) -> str | None:
    """Extract the file path from a rich backup output line (display-aware).

    Unlike ``_extract_path`` this understands archive ``archive:member``
    display paths in content lines: it splits on the first ``:N:`` /
    ``-N-`` delimiter via ``parse_content_line``.
    """
    if output_mode == "content":
        if line == "--":
            return None
        parsed = parse_content_line(line)
        if parsed is not None:
            return parsed[0]
        # line_number=False emits "path:text" / "path-text" without numbers.
        for i, ch in enumerate(line):
            if ch in (":", "-"):
                return line[:i]
        return line
    return None


def _entries_are_rich(entries: list[str]) -> bool:
    """True when any entry needs the rich pipeline (selector/archive/multi)."""
    if len(entries) != 1:
        return True
    path_part, sel = split_path_and_sel(entries[0])
    if sel is not None:
        return True
    return bool(parse_archive_path_candidates(path_part))


class Grep(CallableTool2[Params]):
    name: str = "grep"
    description: str = (
        "Search file contents with a ripgrep regular expression. "
        "Returns matching lines with line numbers, grouped by file. "
        "Returns the first 250 matches inline; a capped result reports where "
        "the complete match list was saved. "
        "Use read on a matched file for surrounding context. "
        "Multiline patterns match across line boundaries."
    )
    params: type[Params] = Params
    field_aliases = {
        **FIELD_ALIASES_GENERAL,
        **FIELD_ALIASES_FILE,
        **FIELD_ALIASES_WEB,
        "paths": "path",
        "glob": "include",
        "filter": "include",
        "file_pattern": "include",
        "-B": "before_context",
        "-A": "after_context",
        "-C": "context",
        "-n": "line_number",
        "-i": "ignore_case",
    }

    def __init__(self, runtime: Runtime, vfs: VFS | None = None) -> None:
        super().__init__(self.name, self.description, self.params)
        self._runtime = runtime
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._vfs = vfs

        bin_name = _rg_binary_name()
        existing = _find_existing_rg(bin_name)
        self._rg_path: str | None = str(existing) if existing else None
        self._rg_path_task: asyncio.Task[str] | None = None
        if self._rg_path is None:
            with contextlib.suppress(RuntimeError):
                self._rg_path_task = asyncio.create_task(_ensure_rg_path())

        rtk_bin_name = _rtk_binary_name()
        existing_rtk = _find_existing_rtk(rtk_bin_name)
        self._rtk_path: str | None = str(existing_rtk) if existing_rtk else None
        self._rtk_path_task: asyncio.Task[str] | None = None
        if self._rtk_path is None:
            with contextlib.suppress(RuntimeError):
                self._rtk_path_task = asyncio.create_task(_ensure_rtk_path())

    async def _resolve_rtk_path(self) -> str | None:
        """Resolve the absolute rtk binary path, or None on any failure.

        rtk is an optional output-dedup wrapper: when it cannot be resolved
        (missing binary, failed download) the caller silently falls back to
        plain rg, which is fully functional alone.
        """
        if self._rtk_path is not None:
            return self._rtk_path
        if self._rtk_path_task is None:
            return None
        try:
            rtk_path = await self._rtk_path_task
        except Exception as e:
            logger.warning(
                "Failed to ensure rtk binary, falling back to plain rg: {error}", error=e
            )
            return None
        self._rtk_path = rtk_path
        return rtk_path

    @override
    async def __call__(self, params: Params, *, _retry: bool = False) -> ToolReturnValue:
        has_dirty = (
            self._vfs is not None
            and self._vfs.virtual_root.exists()
            and any(p.is_file() for p in self._vfs.virtual_root.rglob("*"))
        )
        if has_dirty:
            return await self.backup_grep(params)

        rg_path = self._rg_path
        if rg_path is None:
            if self._rg_path_task is not None:
                try:
                    rg_path = await self._rg_path_task
                    self._rg_path = rg_path
                except Exception as e:
                    logger.warning("Failed to ensure ripgrep binary: {error}", error=e)
                    return await self.backup_grep(params)
            else:
                return await self.backup_grep(params)

        # Resolve rtk (output dedup wrapper) before building argv. Any
        # failure degrades silently to plain rg — never backup_grep, never an
        # error. When deduplicate_output is False the argv is byte-for-byte
        # identical to the plain-rg invocation.
        rtk_path: str | None = None
        if params.deduplicate_output:
            rtk_path = await self._resolve_rtk_path()

        # Selector/archive/multi-entry searches route through the rich
        # pipeline; a single plain entry keeps the legacy byte-identical path.
        entries = expand_path_entries(params.path) or ["."]
        if _entries_are_rich(entries):
            return await self._rich_call(params, entries, rtk_path, _retry=_retry)
        path_input = entries[0]

        try:
            message = ""

            # Resolve search path against the session work directory.
            search_path = (
                local_path_for_cwd(self._work_dir)
                / Path(normalize_user_path(path_input)).expanduser()
            ).resolve()

            # Windows reserved device names (NUL, CON, etc.) cause os error 1.
            if _is_windows_reserved_name(str(search_path)):
                return ToolError(
                    message=(
                        f"`{path_input}` is a reserved device name on Windows "
                        f"and cannot be searched."
                    ),
                    brief=f"Reserved device name | {_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}",
                )

            # Validate workspace using the work-dir-resolved path.
            logical_search_path = KaosPath(str(search_path)).canonical()
            original_is_absolute = kaos_path_from_user_input(path_input).is_absolute()
            if (
                not is_within_workspace(
                    logical_search_path, self._work_dir, self._additional_dirs
                )
                and not original_is_absolute
            ):
                display_path = path_input.replace("\\", "/")
                return ToolError(
                    message=f"`{display_path}` is outside the workspace.",
                    brief=f"Path outside workspace | {_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}",
                )

            logger.debug("Using ripgrep binary: {rg_bin}", rg_bin=rg_path)
            if rtk_path is not None:
                logger.debug("Wrapping rg with rtk binary: {rtk_bin}", rtk_bin=rtk_path)
            args = _build_rg_args(
                _RG_CMD,
                params,
                single_threaded=_retry,
                resolved_path=str(search_path),
                rtk_path=rtk_path,
            )

            output, stderr_str, returncode, timed_out, buffer_truncated = (
                await self._run_rg_subprocess(args, params.timeout)
            )

            # Drop last incomplete line if buffer was truncated
            if buffer_truncated:
                last_nl = output.rfind("\n")
                output = output[:last_nl] if last_nl >= 0 else ""
                message = "Output exceeded buffer limit. Some results omitted."

            # Timeout: return partial results if available, otherwise error
            if timed_out:
                if not output.strip():
                    return ToolError(
                        message=(
                            f"Grep timed out after {params.timeout}s. "
                            "Try a more specific path or pattern."
                        ),
                        brief=f"Grep timed out | {_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}",
                    )
                message = (
                    f"{message} Grep timed out after {params.timeout}s. "
                    "Partial results returned."
                    if message
                    else f"Grep timed out after {params.timeout}s. Partial results returned."
                )

            # rg exit codes: 0=matches found, 1=no matches, 2+=error
            if not timed_out and returncode not in (0, 1):
                # EAGAIN: retry once with single-threaded mode
                if not _retry and _is_eagain(stderr_str):
                    logger.warning("rg EAGAIN error, retrying with -j 1")
                    return await self.__call__(params, _retry=True)
                return ToolError(
                    message=f"Failed to grep. Error: {stderr_str}",
                    brief=f"Failed to grep | {_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}",
                )

            ctx = _plain_ctx(search_path)
            ctx.grouped = bool(params.grouped) if params.grouped is not None else False

            # --- Post-processing pipeline ---
            # The rg subprocess is bounded by params.timeout above; the
            # post-processing (rtk parsing, sensitive filtering, pagination and
            # micro-compression) can also be expensive on pathological inputs,
            # so it gets its own timeout instead of hanging the tool call.
            try:
                return await asyncio.wait_for(
                    self._postprocess(
                        params=params,
                        output=output,
                        timed_out=timed_out,
                        buffer_truncated=buffer_truncated,
                        rtk_path=rtk_path,
                        message=message,
                        ctx=ctx,
                    ),
                    timeout=params.timeout,
                )
            except TimeoutError:
                return ToolError(
                    message=(
                        f"Grep post-processing timed out after {params.timeout}s. "
                        "Try a more specific path or pattern."
                    ),
                    brief=(
                        f"Grep post-processing timed out | "
                        f"{_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}"
                    ),
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Grep failed: pattern={pattern}, path={path}: {error}",
                pattern=params.pattern,
                path=path_input,
                error=e,
            )
            return ToolError(
                message=f"Failed to grep. Error: {str(e)}",
                brief=f"Failed to grep | {_format_cmd(params, rtk_path=rtk_path, path_input=path_input)}",
            )

    async def _run_rg_subprocess(
        self, args: list[str], timeout: int
    ) -> tuple[str, str, int | None, bool, bool]:
        """Run rg (optionally rtk-wrapped) and collect bounded output.

        Returns ``(output, stderr_str, returncode, timed_out, buffer_truncated)``.
        """
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._work_dir),
            env=_env_with_shared_bin_path(),
        )

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        timed_out = False
        stdout_truncated_flag: list[bool] = [False]

        try:
            assert process.stdout is not None
            assert process.stderr is not None
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream(
                        process.stdout, stdout_buf, RG_MAX_BUFFER, stdout_truncated_flag
                    ),
                    _read_stream(process.stderr, stderr_buf, RG_MAX_BUFFER),
                ),
                timeout=timeout,
            )
            await process.wait()
        except asyncio.CancelledError:
            await _kill_process(process)
            raise
        except TimeoutError:
            await _kill_process(process)
            timed_out = True

        output = stdout_buf.decode("utf-8", errors="replace")
        stderr_str = stderr_buf.decode("utf-8", errors="replace")
        return (
            output,
            stderr_str,
            process.returncode,
            timed_out,
            stdout_truncated_flag[0],
        )

    async def _rich_call(
        self,
        params: Params,
        entries: list[str],
        rtk_path: str | None,
        *,
        _retry: bool = False,
    ) -> ToolReturnValue:
        """Native rg pipeline for selector/archive/multi-entry searches."""
        brief_cmd = _format_cmd(params, rtk_path=rtk_path, path_input=entries[0])
        has_archive = any(
            parse_archive_path_candidates(split_path_and_sel(e)[0]) for e in entries
        )
        tmp_dir: tempfile.TemporaryDirectory | None = None
        try:
            scratch_dir: Path | None = None
            if has_archive:
                tmp_dir = tempfile.TemporaryDirectory(prefix="kimi-grep-")
                scratch_dir = Path(tmp_dir.name)

            ctx, resolved, specs, err = await _resolve_selector_specs(
                entries, self._work_dir, self._additional_dirs, scratch_dir
            )
            if err is not None or ctx is None:
                return err if err is not None else ToolError(
                    message="Failed to resolve search paths.",
                    brief="Failed to resolve paths",
                )

            ranged = [s for s in specs if s.ranges]
            if ranged and params.output_mode != "content":
                return ToolError(
                    message=(
                        "Line-range selector requires output_mode='content' "
                        "(files_with_matches/count_matches have no per-line stream)."
                    ),
                    brief="Selector requires content mode",
                )

            max_count_override: int | None = None
            if ranged:
                max_end = max(
                    (r.end_line or 1) for s in ranged for r in (s.ranges or [])
                )
                base = (params.offset or 0) + (params.head_limit or 0) + _RG_HEAD_LIMIT_MARGIN
                max_count_override = min(RG_RANGE_FETCH_CAP, max(base, max_end))

            _build_ranges_map(ctx, resolved, specs, absolute_keys=False)
            # rtk's near-duplicate folding would silently drop in-range lines
            # of repetitive files BEFORE range filtering can keep them (plan
            # 23 §9.3) — run ranged searches through plain rg.
            effective_rtk = None if ranged else rtk_path
            ctx.grouped = (
                params.output_mode == "content"
                and should_group(params, has_rich_entries=True)
            )

            # Fan out one rg invocation per resolved target and concatenate.
            outputs: list[str] = []
            message = ""
            timed_out_any = False
            buffer_truncated_any = False
            for path_arg in resolved:
                args = _build_rg_args(
                    _RG_CMD,
                    params,
                    single_threaded=_retry,
                    resolved_path=path_arg,
                    rtk_path=effective_rtk,
                    max_count_override=max_count_override,
                )
                output, stderr_str, returncode, timed_out, buffer_truncated = (
                    await self._run_rg_subprocess(args, params.timeout)
                )
                if buffer_truncated:
                    last_nl = output.rfind("\n")
                    output = output[:last_nl] if last_nl >= 0 else ""
                    buffer_truncated_any = True
                    message = (
                        f"{message} Output exceeded buffer limit. Some results omitted."
                        if message
                        else "Output exceeded buffer limit. Some results omitted."
                    )
                if timed_out:
                    timed_out_any = True
                    if not output.strip():
                        return ToolError(
                            message=(
                                f"Grep timed out after {params.timeout}s. "
                                "Try a more specific path or pattern."
                            ),
                            brief=f"Grep timed out | {brief_cmd}",
                        )
                if not timed_out and returncode not in (0, 1):
                    if not _retry and _is_eagain(stderr_str):
                        logger.warning("rg EAGAIN error, retrying with -j 1")
                        return await self._rich_call(
                            params, entries, rtk_path, _retry=True
                        )
                    return ToolError(
                        message=f"Failed to grep. Error: {stderr_str}",
                        brief=f"Failed to grep | {brief_cmd}",
                    )
                if output.strip():
                    # Bare rg omits the path prefix for single-file targets;
                    # re-attach this target's display key so the shared
                    # line-stream pipeline sees full `path:LN:text` lines.
                    display_key = ctx.display_map.get(
                        path_arg, _strip_key_for(path_arg, ctx.prefix_base)
                    )
                    reattached = _reattach_single_file_prefix(
                        output.splitlines(), display_key
                    )
                    outputs.append("\n".join(reattached))

            if timed_out_any:
                timeout_msg = (
                    f"Grep timed out after {params.timeout}s. Partial results returned."
                )
                message = f"{message} {timeout_msg}" if message else timeout_msg

            combined = "\n".join(outputs)
            try:
                return await asyncio.wait_for(
                    self._postprocess(
                        params=params,
                        output=combined,
                        timed_out=timed_out_any,
                        buffer_truncated=buffer_truncated_any,
                        rtk_path=rtk_path,
                        message=message,
                        ctx=ctx,
                    ),
                    timeout=params.timeout,
                )
            except TimeoutError:
                return ToolError(
                    message=(
                        f"Grep post-processing timed out after {params.timeout}s. "
                        "Try a more specific path or pattern."
                    ),
                    brief=f"Grep post-processing timed out | {brief_cmd}",
                )
        finally:
            if tmp_dir is not None:
                with contextlib.suppress(OSError):
                    tmp_dir.cleanup()

    async def _postprocess(
        self,
        *,
        params: Params,
        output: str,
        timed_out: bool,
        buffer_truncated: bool,
        rtk_path: str | None,
        message: str,
        ctx: _GrepCtx | None = None,
        search_path: Path | None = None,
    ) -> ToolReturnValue:
        """Run the post-subprocess output pipeline and build the final result.

        Extracted from ``__call__`` so it can be bounded by ``params.timeout``:
        the micro-compression stage is CPU-heavy and previously ran with no
        timeout, so a pathological single-line match could hang the tool for
        minutes.
        """
        if ctx is None:
            # Back-compat: callers passing only ``search_path`` get the
            # legacy (selector-free) context.
            ctx = _plain_ctx(search_path if search_path is not None else Path("."))
        builder = ToolResultBuilder()
        # --- Post-processing pipeline ---
        lines = output.splitlines()
        if lines and lines[-1] == "":
            lines.pop()

        files_truncated_early = False
        total_raw_files = 0

        # Step 0: strip rtk protocol lines (content mode only) and keep
        # their metadata for the summary message. Other modes pass through
        # untouched (rtk does not emit protocol lines for them).
        rtk_meta: dict[str, Any] = {}
        rtk_original_path: str | None = None
        if params.output_mode == "content":
            lines, rtk_meta = parse_rtk_rg_output(lines)
            # When rtk truly truncated output (per-file folds or skipped
            # files), preserve the original stream so the model can page
            # through the full results.
            if rtk_meta.get("folded_files") or rtk_meta.get("skipped_files"):
                from kimix.tools.common import _export_to_temp_file_async

                rtk_original_path, _ = await _export_to_temp_file_async(
                    key=None, content=output, ext=".txt"
                )

        # Step 1: mtime sorting (files_with_matches only, skip on timeout)
        if not timed_out and params.output_mode == "files_with_matches":
            lines = [ln for ln in lines if ln.strip()]
            total_raw_files = len(lines)
            mtimes = await asyncio.gather(*[_safe_getmtime_async(p) for p in lines])

            k = params.offset + (params.head_limit or 0)
            if k and len(lines) > k:
                lines = [
                    p for _, p in heapq.nlargest(
                        k, zip(mtimes, lines, strict=True), key=lambda x: x[0]
                    )
                ]
                files_truncated_early = True
            else:
                lines = [
                    p for _, p in sorted(
                        zip(mtimes, lines, strict=True), key=lambda x: x[0], reverse=True
                    )
                ]

        # Step 2: shorten paths to relative (prefix stripping)
        search_base = ctx.prefix_base
        if ctx.prefix_base_is_file:
            search_base = str(Path(ctx.prefix_base).parent)
        lines = _strip_path_prefix(lines, search_base)

        # Rich searches (selectors/archives): normalize separators, remap
        # archive scratch paths to their ``archive:member`` display form.
        # Both happen BEFORE sensitive filtering and range filtering so the
        # keys match (plan 23 §9.9 ordering).
        is_rich = bool(ctx.display_map) or bool(ctx.ranges)
        if is_rich:
            lines = _normalize_slashes_content(lines, params.output_mode)
            lines = _remap_display(lines, ctx.display_map)

        # Step 3: filter sensitive files from output (now on a clean
        # stream: rtk fold markers can no longer be mistaken for paths)
        filtered_paths: list[str] = []
        kept_lines: list[str] = []
        sensitive_path_set: set[str] = set()
        for line in lines:
            if params.output_mode == "content":
                # Match lines: "file.py:10:matched text"
                # Context lines: "file.py-10-context text"
                # Separator: "--"
                if line == "--":
                    kept_lines.append(line)
                    continue
                m = _RG_LINE_RE.match(line)
                file_path = m.group(1) if m else line
            elif params.output_mode == "count_matches":
                # Count lines: "file.py:42"
                idx = line.rfind(":")
                file_path = line[:idx] if idx > 0 else line
            else:
                # files_with_matches: pure path per line
                file_path = line

            if file_path and _is_sensitive_cached(file_path):
                if file_path not in sensitive_path_set:
                    sensitive_path_set.add(file_path)
                    filtered_paths.append(file_path)
            else:
                kept_lines.append(line)

        if filtered_paths:
            # Remove trailing "--" separators left after filtering
            while kept_lines and kept_lines[-1] == "--":
                kept_lines.pop()
            warning = sensitive_file_warning(filtered_paths)
            message = f"{message} {warning}" if message else warning

        lines = kept_lines

        # Line-range post-filter (rich searches): drop out-of-range matches
        # and context, then prune orphan separators.
        if ctx.ranges and params.output_mode == "content":
            lines = _range_filter_lines(lines, ctx.ranges)

        # File recorder: persist deduplicated matched files on the session.
        # The note is appended EARLY (before summaries/fold notes) so
        # existing tests that parse the message tail (e.g. "Original
        # output: <path>") still find their anchor as the last token.
        record_note = ""
        if params.record and lines:
            record_files = _collect_record_files(lines, params.output_mode)
            if record_files:
                record_grep_files(
                    self._runtime.session,
                    record_files,
                    cwd=str(local_path_for_cwd(self._work_dir)),
                )
                record_note = (
                    f"Recorded {len(record_files)} matched file(s) in session "
                    "(use `read`/`edit` on them)."
                )
                message = (
                    f"{message} {record_note}" if message else record_note
                )

        # Step 4: summaries (before pagination, on full results)
        if params.output_mode == "count_matches":
            total_matches = 0
            total_files = 0
            for line in lines:
                idx = line.rfind(":")
                if idx > 0:
                    try:
                        total_matches += int(line[idx + 1:])
                        total_files += 1
                    except ValueError:
                        pass
            count_summary = (
                f"Found {total_matches} total occurrences across {total_files} files."
            )
            message = f"{message} {count_summary}" if message else count_summary

        if (
            params.output_mode == "content"
            and rtk_meta.get("total_matches") is not None
        ):
            # rtk header reported totals for the whole search.
            rtk_summary = (
                f"Found {rtk_meta['total_matches']} matches in "
                f"{rtk_meta['total_files']} files."
            )
            message = f"{message} {rtk_summary}" if message else rtk_summary
            fold_note = _rtk_fold_note(rtk_meta, original_path=rtk_original_path)
            if fold_note:
                message = f"{message} {fold_note}" if message else fold_note

        if params.output_mode == "files_with_matches":
            files_summary = f"Found {len(lines)} files matching {params.pattern!r}."
            message = f"{message} {files_summary}" if message else files_summary

        if ctx.notes:
            skip_note = (
                "Skipped archive entries (text members only): "
                + "; ".join(ctx.notes) + "."
            )
            message = f"{message} {skip_note}" if message else skip_note

        # Step 5: local dedup fallback — only when rtk did NOT run, so
        # repeated lines are never collapsed twice. Skipped for grouped
        # output (headers make cross-file dedup undesirable).
        dedup_saved = 0
        if (
            params.output_mode == "content"
            and params.deduplicate_output
            and rtk_path is None
            and not ctx.grouped
        ):
            lines, dedup_saved = dedup_lines(lines)
            if dedup_saved:
                dedup_msg = f"Removed {dedup_saved} repeated line(s) via dedup."
                message = f"{message} {dedup_msg}" if message else dedup_msg

        # Step 6: offset + head_limit pagination
        if params.offset > 0:
            lines = lines[params.offset:]

        effective_limit = params.head_limit
        if effective_limit and len(lines) > effective_limit:
            total = len(lines) + params.offset
            lines = lines[:effective_limit]
            truncation_msg = (
                f"Results truncated to {effective_limit} lines (total: {total}). "
                f"Use offset={params.offset + effective_limit} to see more."
            )
            message = f"{message} {truncation_msg}" if message else truncation_msg
        elif (
            effective_limit
            and params.output_mode == "files_with_matches"
            and files_truncated_early
            and len(lines) == effective_limit
        ):
            truncation_msg = (
                f"Results truncated to {effective_limit} lines (total: {total_raw_files}). "
                f"Use offset={params.offset + effective_limit} to see more."
            )
            message = f"{message} {truncation_msg}" if message else truncation_msg

        # Grouped rendering (rich searches or explicit grouped=True), after
        # pagination: `# path` headers with `*N|` / ` N|` body markers.
        if (
            ctx.grouped
            and params.output_mode == "content"
            and lines
        ):
            groups = group_lines_by_file(lines, parse_content_line)
            lines = format_grouped_output(groups)

        # Step 7: final display fold budget (head+tail fold with marker).
        # 0 = unlimited → the byte cap below is the only remaining limit.
        omitted_by_fold = 0
        if params.max_output_lines:
            lines, omitted_by_fold = fold_lines(lines, params.max_output_lines)
            if omitted_by_fold:
                fold_msg = (
                    f"Results folded to {len(lines) - 1} lines "
                    f"({omitted_by_fold} omitted). "
                    "Use max_output_lines=0 or offset to see more."
                )
                message = f"{message} {fold_msg}" if message else fold_msg

        # Step 7.5: micro-compress — lossless stages (1-3, 5) plus the
        # annotated prefix fold (Stage 4) which collapses the repeated
        # absolute-path prefix on every match.  Near-duplicate collapse
        # (Stage 8) is disabled: every distinct match must stay visible.
        if lines:
            # Truncate each line before micro-compression so a single
            # gigantic line (e.g. a 3MB minified/data file match) can never
            # reach the compressor's prefix-folding stages (O(n^2) on the
            # first line). The final per-line hygiene pass below is then a
            # cheap no-op.
            lines = [truncate_line(ln) for ln in lines]
            lines, _mc_saved = _mc_compress_lines(
                lines,
                kind="log",
                config=MicroCompressConfig(
                    lossless_only=False,
                    near_dup_collapse=False,
                    # Stage 4 prefix fold would distort `#` headers in
                    # grouped output — disable it there (plan 23 §9.8).
                    prefix_fold=not ctx.grouped,
                ),
            )

        lines = _normalize_output_lines(lines, params.output_mode)
        # Per-line hygiene before the byte cap: no single line can hog the
        # whole budget (mirror of the display builder's own truncation).
        lines = [truncate_line(ln) for ln in lines]
        output, truncated_by_bytes = _join_with_byte_limit(lines)

        if not output and not buffer_truncated:
            no_match_msg = "No matches found"
            if message:
                no_match_msg = f"{no_match_msg}. {message}"
            return builder.ok(
                message=no_match_msg, brief=_format_cmd(params, rtk_path=rtk_path)
            )

        if truncated_by_bytes:
            byte_msg = f"Output truncated to {MAX_BYTES} bytes."
            message = f"{message} {byte_msg}" if message else byte_msg

        builder.write(output)
        return builder.ok(message=message, brief=_format_cmd(params, rtk_path=rtk_path))


    async def backup_grep(self, params: Params) -> ToolReturnValue:
        """Pure-Python fallback (no ripgrep/rtk, or dirty VFS).

        Bounded by ``params.timeout`` like the native path so a huge tree can't
        hang the tool call indefinitely.
        """
        try:
            return await asyncio.wait_for(
                self._backup_grep_impl(params), timeout=params.timeout
            )
        except TimeoutError:
            return ToolError(
                message=(
                    f"Grep (fallback) timed out after {params.timeout}s. "
                    "Try a more specific path or pattern."
                ),
                brief=f"Grep fallback timed out | {_format_cmd(params)}",
            )

    async def _backup_grep_impl(self, params: Params) -> ToolReturnValue:
        try:
            if not params.pattern:
                return ToolError(
                    message="Pattern cannot be empty.",
                    brief=f"Empty pattern | {_format_cmd(params)}",
                )

            flags = 0
            if params.ignore_case:
                flags |= re.IGNORECASE
            use_multiline = params.multiline or _pattern_has_regex_newline(params.pattern)
            if use_multiline:
                flags |= re.DOTALL
            pattern = _multiline_pattern(params.pattern) if use_multiline else params.pattern

            try:
                regex = _compile_regex_cached(pattern, flags)
            except re.error as e:
                return ToolError(
                    message=f"Invalid regex pattern: {e}",
                    brief=f"Invalid pattern | {_format_cmd(params)}",
                )

            search_path: Path | None = None
            entries = expand_path_entries(params.path) or ["."]
            rich = _entries_are_rich(entries)
            entry_display = entries[0] if entries else "."

            ranges_display: dict[str, list[LineRange]] = {}
            display_map: dict[str, str] = {}
            rich_notes: list[str] = []
            grouped = False
            file_ranges: dict[Path, list[LineRange] | None] = {}
            prefix_base: str | None = None
            tmp_dir: tempfile.TemporaryDirectory | None = None

            if not rich:
                search_path = (
                    local_path_for_cwd(self._work_dir)
                    / Path(normalize_user_path(entry_display)).expanduser()
                ).resolve()

                # Windows reserved device names (NUL, CON, etc.) cause os error 1.
                if _is_windows_reserved_name(str(search_path)):
                    display_path = entry_display.replace("\\", "/")
                    return ToolError(
                        message=(
                            f"`{display_path}` is a reserved device name on Windows "
                            f"and cannot be searched."
                        ),
                        brief=f"Reserved device name | {_format_cmd(params)}",
                    )

                # Validate workspace
                logical_search_path = KaosPath(str(search_path)).canonical()
                original_is_absolute = kaos_path_from_user_input(entry_display).is_absolute()
                if (
                    not is_within_workspace(logical_search_path, self._work_dir, self._additional_dirs)
                    and not original_is_absolute
                ):
                    display_path = entry_display.replace("\\", "/")
                    return ToolError(
                        message=f"`{display_path}` is outside the workspace.",
                        brief=f"Path outside workspace | {_format_cmd(params)}",
                    )

                # Translate search path through VFS for I/O
                if self._vfs is not None:
                    with contextlib.suppress(ValueError):
                        search_path = self._vfs.translate_path(search_path)

                if not search_path.exists():
                    display_path = entry_display.replace("\\", "/")
                    return ToolError(
                        message=f"`{display_path}` does not exist.",
                        brief=f"Path not found | {_format_cmd(params)}",
                    )

                files = self._collect_files(search_path, params)
                prefix_base = str(search_path)
                grouped = (
                    params.output_mode == "content" and params.grouped is True
                )
            else:
                has_archive = any(
                    parse_archive_path_candidates(split_path_and_sel(e)[0])
                    for e in entries
                )
                if has_archive:
                    tmp_dir = tempfile.TemporaryDirectory(prefix="kimi-grep-")
                ctx, resolved, specs, err = await _resolve_selector_specs(
                    entries,
                    self._work_dir,
                    self._additional_dirs,
                    Path(tmp_dir.name) if tmp_dir is not None else None,
                )
                if err is not None or ctx is None:
                    if tmp_dir is not None:
                        tmp_dir.cleanup()
                    return err if err is not None else ToolError(
                        message="Failed to resolve search paths.",
                        brief="Failed to resolve paths",
                    )
                # The backup scanner implements ranges for ALL output modes
                # (files/count scan the in-range window directly).
                display_map = ctx.display_map
                rich_notes = list(ctx.notes)
                _build_ranges_map(ctx, resolved, specs, absolute_keys=True)
                cwd = local_path_for_cwd(self._work_dir)
                files = []
                for path_arg in resolved:
                    p = Path(path_arg)
                    if self._vfs is not None:
                        with contextlib.suppress(ValueError):
                            p = self._vfs.translate_path(p)
                    if not p.exists():
                        if tmp_dir is not None:
                            tmp_dir.cleanup()
                        return ToolError(
                            message=f"`{p}` does not exist.",
                            brief=f"Path not found | {_format_cmd(params)}",
                        )
                    for f in self._collect_files(p, params):
                        files.append(f)
                        file_ranges[f] = ctx.ranges.get(path_arg)
                prefix_base = str(cwd)
                grouped = (
                    params.output_mode == "content"
                    and should_group(params, has_rich_entries=True)
                )

            try:
                return await self._backup_grep_search(
                    params,
                    regex=regex,
                    files=files,
                    file_ranges=file_ranges,
                    ranges_display=ranges_display,
                    display_map=display_map,
                    rich_notes=rich_notes,
                    grouped=grouped,
                    prefix_base=prefix_base or str(local_path_for_cwd(self._work_dir)),
                    legacy_search_path=search_path,
                )
            finally:
                if tmp_dir is not None:
                    with contextlib.suppress(OSError):
                        tmp_dir.cleanup()
        except Exception as e:
            logger.warning(
                "Grep backup failed: pattern={pattern}, path={path}: {error}",
                pattern=params.pattern,
                path=params.path,
                error=e,
            )
            return ToolError(
                message=f"Failed to grep. Error: {str(e)}",
                brief=f"Failed to grep | {_format_cmd(params)}",
            )

    async def _backup_grep_search(
        self,
        params: Params,
        *,
        regex,
        files: list[Path],
        file_ranges: dict[Path, list[LineRange] | None],
        ranges_display: dict[str, list[LineRange]],
        display_map: dict[str, str],
        rich_notes: list[str],
        grouped: bool,
        prefix_base: str,
        legacy_search_path: Path | None,
    ) -> ToolReturnValue:
        try:
            output_mode = params.output_mode

            # Execute search in parallel across files.
            loop = asyncio.get_running_loop()
            max_workers = min(32, (os.cpu_count() or 1) + 4)

            def _process_one(file_path: Path) -> list[str]:
                text = _read_file_text(file_path, self._vfs)
                if text is None:
                    return []

                ranges = file_ranges.get(file_path)

                if output_mode == "files_with_matches":
                    window = _text_in_ranges(text, ranges) if ranges else text
                    if regex.search(window):
                        return [str(file_path)]
                    return []

                if output_mode == "count_matches":
                    window = _text_in_ranges(text, ranges) if ranges else text
                    count = len(list(regex.finditer(window)))
                    if count > 0:
                        return [f"{file_path}:{count}"]
                    return []

                # content mode
                return self._search_content_single(
                    file_path, text, regex, params, ranges=ranges
                )

            # Explicit shutdown(wait=False, cancel_futures=True): the context
            # manager form waits for every worker on exit, which would defeat
            # cancellation from asyncio.wait_for(timeout) — a slow tree must be
            # able to abort promptly. Running workers finish in the background.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = [
                    loop.run_in_executor(executor, _process_one, f) for f in files
                ]
                results = await asyncio.gather(*futures)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            raw_lines = [line for r in results for line in r]

            # Filter sensitive files from output.
            filtered_paths: list[str] = []
            sensitive_path_set: set[str] = set()
            kept_lines: list[str] = []
            for line in raw_lines:
                file_path = self._extract_path(line, output_mode)
                if file_path and _is_sensitive_cached(file_path):
                    if file_path not in sensitive_path_set:
                        sensitive_path_set.add(file_path)
                        filtered_paths.append(file_path)
                else:
                    kept_lines.append(line)

            message = ""
            if filtered_paths:
                warning = sensitive_file_warning(filtered_paths)
                message = warning

            lines = kept_lines
            total_raw = 0
            files_truncated_early = False

            # Post-processing specific to output mode.
            if output_mode == "files_with_matches":
                total_raw = len(lines)
                lines_with_mtime = [(p, _safe_getmtime(p)) for p in lines]

                k = params.offset + (params.head_limit or 0)
                if k and len(lines) > k:
                    lines = [
                        p
                        for p, _ in heapq.nlargest(
                            k, lines_with_mtime, key=lambda x: x[1]
                        )
                    ]
                    files_truncated_early = True
                else:
                    lines_with_mtime.sort(key=lambda x: x[1], reverse=True)
                    lines = [p for p, _ in lines_with_mtime]

            elif output_mode == "count_matches":
                total_matches = 0
                total_files = 0
                for line in lines:
                    idx = line.rfind(":")
                    if idx > 0:
                        try:
                            total_matches += int(line[idx + 1:])
                            total_files += 1
                        except ValueError:
                            pass
                count_summary = (
                    f"Found {total_matches} total occurrences across {total_files} files."
                )
                message = f"{message} {count_summary}" if message else count_summary

            # files_with_matches summary (after filtering, before pagination).
            if output_mode == "files_with_matches":
                files_summary = f"Found {len(lines)} files matching {params.pattern!r}."
                message = f"{message} {files_summary}" if message else files_summary

            # Local dedup fallback (backup_grep never runs rtk). Skipped for
            # grouped output (headers make cross-file dedup undesirable).
            dedup_saved = 0
            if output_mode == "content" and params.deduplicate_output and not grouped:
                lines, dedup_saved = dedup_lines(lines)
                if dedup_saved:
                    dedup_msg = f"Removed {dedup_saved} repeated line(s) via dedup."
                    message = f"{message} {dedup_msg}" if message else dedup_msg

            # Strip search-base prefix for relative paths.
            if legacy_search_path is not None:
                search_base = str(legacy_search_path)
                if legacy_search_path.is_file():
                    search_base = str(legacy_search_path.parent)
            else:
                search_base = prefix_base
            lines = _strip_path_prefix(lines, search_base)

            # Rich searches: remap archive scratch paths to archive:member
            # display form (after prefix strip, before recording/grouping).
            if display_map:
                lines = _normalize_slashes_content(lines, output_mode)
                lines = _remap_display(lines, display_map)

            # Offset + head_limit pagination.
            if output_mode == "files_with_matches":
                if params.offset > 0:
                    lines = lines[params.offset:]

                effective_limit = params.head_limit
                if effective_limit and len(lines) > effective_limit:
                    total = len(lines) + params.offset
                    lines = lines[:effective_limit]
                    truncation_msg = (
                        f"Results truncated to {effective_limit} lines (total: {total}). "
                        f"Use offset={params.offset + effective_limit} to see more."
                    )
                    message = f"{message} {truncation_msg}" if message else truncation_msg
                elif (
                    effective_limit
                    and files_truncated_early
                    and len(lines) == effective_limit
                ):
                    truncation_msg = (
                        f"Results truncated to {effective_limit} lines (total: {total_raw}). "
                        f"Use offset={params.offset + effective_limit} to see more."
                    )
                    message = f"{message} {truncation_msg}" if message else truncation_msg
            else:
                if params.offset > 0:
                    lines = lines[params.offset:]

                effective_limit = params.head_limit
                if effective_limit and len(lines) > effective_limit:
                    total = len(lines) + params.offset
                    lines = lines[:effective_limit]
                    truncation_msg = (
                        f"Results truncated to {effective_limit} lines (total: {total}). "
                        f"Use offset={params.offset + effective_limit} to see more."
                    )
                    message = f"{message} {truncation_msg}" if message else truncation_msg

            # File recorder: persist deduplicated matched files on the session.
            if params.record and lines:
                record_files = _collect_record_files(lines, output_mode)
                if record_files:
                    record_grep_files(
                        self._runtime.session,
                        record_files,
                        cwd=str(local_path_for_cwd(self._work_dir)),
                    )

            if rich_notes:
                skip_note = (
                    "Skipped archive entries (text members only): "
                    + "; ".join(rich_notes) + "."
                )
                message = f"{message} {skip_note}" if message else skip_note

            # Grouped rendering (rich searches or explicit grouped=True).
            if grouped and output_mode == "content" and lines:
                groups = group_lines_by_file(lines, parse_content_line)
                lines = format_grouped_output(groups)

            # Final display fold budget (head+tail fold with marker).
            omitted_by_fold = 0
            if params.max_output_lines:
                lines, omitted_by_fold = fold_lines(lines, params.max_output_lines)
                if omitted_by_fold:
                    fold_msg = (
                        f"Results folded to {len(lines) - 1} lines "
                        f"({omitted_by_fold} omitted). "
                        "Use max_output_lines=0 or offset to see more."
                    )
                    message = f"{message} {fold_msg}" if message else fold_msg

            lines = _normalize_output_lines(lines, output_mode)
            # Per-line hygiene before the byte cap.
            lines = [truncate_line(ln) for ln in lines]
            builder = ToolResultBuilder()
            output, truncated_by_bytes = _join_with_byte_limit(lines)

            if not output:
                no_match_msg = "No matches found"
                if message:
                    no_match_msg = f"{no_match_msg}. {message}"
                return builder.ok(message=no_match_msg, brief=_format_cmd(params))

            if truncated_by_bytes:
                byte_msg = f"Output truncated to {MAX_BYTES} bytes."
                message = f"{message} {byte_msg}" if message else byte_msg

            builder.write(output)
            return builder.ok(message=message, brief=_format_cmd(params))

        except Exception as e:
            logger.warning(
                "Grep backup failed: pattern={pattern}, path={path}: {error}",
                pattern=params.pattern,
                path=params.path,
                error=e,
            )
            return ToolError(
                message=f"Failed to grep. Error: {str(e)}",
                brief=f"Failed to grep | {_format_cmd(params)}",
            )

    def _collect_files(self, search_path: Path, params: Params) -> list[Path]:
        files: list[Path] = []
        if search_path.is_file():
            if self._is_valid_file(search_path, params):
                files.append(search_path)
        else:
            for root, dirs, filenames in os.walk(search_path):
                dirs[:] = [
                    d for d in dirs
                    if not _should_skip_dir(d, params.include_ignored)
                ]
                for filename in filenames:
                    file_path = Path(root) / filename
                    if self._is_valid_file(file_path, params):
                        files.append(file_path)
        return files

    def _is_valid_file(self, file_path: Path, params: Params) -> bool:
        if not file_path.is_file():
            return False
        try:
            if file_path.stat().st_size > _MAX_FILE_SIZE:
                return False
        except OSError:
            return False
        if not _matches_glob(file_path, params.include):
            return False
        return _matches_type(file_path, params.type)

    def _search_content_single(
        self,
        file_path: Path,
        content: str,
        regex: re.Pattern[str],
        params: Params,
        *,
        ranges: list[LineRange] | None = None,
    ) -> list[str]:
        before = params.before_context or 0
        after = params.after_context or 0
        if params.context is not None:
            before = after = params.context

        if not content:
            return []

        lines = content.splitlines()
        match_line_nums: set[int] = set()

        use_multiline = params.multiline or _pattern_has_regex_newline(params.pattern)
        if use_multiline:
            for m in regex.finditer(content):
                start_line = content.count("\n", 0, m.start()) + 1
                end_line = content.count("\n", 0, m.end()) + 1
                for ln in range(start_line, end_line + 1):
                    match_line_nums.add(ln)
        elif self._use_native_line_scan(content):
            # Native line scan (kimix_native.tools.scan_lines_cb): offsets/
            # line-splitting stay native; the regex matcher stays in Python.
            # Only used when splitlines() == \n-splitting (guard in
            # _use_native_line_scan), so line numbers match exactly.
            if _NATIVE_TOOLS is not None:
                def _matcher(line_bytes: bytes, line_index: int) -> bool:
                    line_text = line_bytes.decode("utf-8", "surrogatepass")
                    return bool(regex.search(line_text))

                hits = _NATIVE_TOOLS.scan_lines_cb(content, _matcher)
                match_line_nums = {int(ln) + 1 for ln, _off, _len in hits}
        else:
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    match_line_nums.add(i)

        # Line-range selector: only in-range lines are matches.
        if ranges is not None:
            match_line_nums = {
                ln for ln in match_line_nums if is_line_in_ranges(ln, ranges)
            }

        if not match_line_nums:
            return []

        intervals = [(ln - before, ln + after) for ln in match_line_nums]
        merged = _merge_intervals(intervals)

        results: list[str] = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                results.append("--")
            for ln in range(max(1, start), min(len(lines), end) + 1):
                # Context clamping: a ranged search never leaks out-of-range
                # content — context outside the windows is dropped.
                if ranges is not None and not is_line_in_ranges(ln, ranges):
                    continue
                text = lines[ln - 1]
                if ln in match_line_nums:
                    if params.line_number:
                        results.append(f"{file_path}:{ln}:{text}")
                    else:
                        results.append(f"{file_path}:{text}")
                else:
                    if params.line_number:
                        results.append(f"{file_path}-{ln}-{text}")
                    else:
                        results.append(f"{file_path}-{text}")

        return results

    @staticmethod
    def _use_native_line_scan(content: str) -> bool:
        """True when native line scanning is safe for *content*.

        The native kernel splits on \n only; ``str.splitlines`` also splits on
        \r (alone or in \r\n) and other Unicode line separators. When any of
        those appear, the Python path runs so line numbers stay identical.
        """
        if not _native_use_native("TOOLS") or _NATIVE_TOOLS is None:
            return False
        for ch in content:
            if ch in "\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029":
                return False
        return True

    def _extract_path(self, line: str, output_mode: str) -> str | None:
        if output_mode == "files_with_matches":
            return line
        if output_mode == "count_matches":
            idx = line.rfind(":")
            return line[:idx] if idx > 0 else line
        # content mode
        if line == "--":
            return None
        for i, ch in enumerate(line):
            if ch in (":", "-"):
                return line[:i]
        return line
