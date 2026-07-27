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
from functools import lru_cache
from pathlib import Path
from typing import Literal, override

from kaos.path import KaosPath
from kosong.tooling import (
    FIELD_ALIASES_FILE,
    FIELD_ALIASES_GENERAL,
    FIELD_ALIASES_WEB,
    CallableTool2,
    ToolError,
    ToolReturnValue,
)
from pydantic import BaseModel, Field, field_validator

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

# Output mode map — only canonical values accepted
_OUTPUT_MODE_MAP: dict[str, Literal["files_with_matches", "count_matches", "content"]] = {
    "files_with_matches": "files_with_matches",
    "count_matches": "count_matches",
    "content": "content",
}


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    pattern: str = Field(description="Regex pattern.")
    path: str = Field(
        description="Search target directory or file.",
        default=".",
    )
    glob: str | None = Field(
        description="Glob filter.",
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
        description="Multiline regex mode.",
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


RG_MAX_BUFFER = 20_000_000  # 20MB stdout/stderr buffer limit
RG_KILL_GRACE = 5  # seconds: SIGTERM -> SIGKILL
MAX_BYTES = 100 << 10  # 100KB
_RG_HEAD_LIMIT_MARGIN = 1000  # extra matches for content-mode --max-count
_RG_DOWNLOAD_LOCK = asyncio.Lock()
_RG_CMD = "rg"


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
    if params.multiline:
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
        if params.head_limit:
            max_count = (params.offset or 0) + params.head_limit + _RG_HEAD_LIMIT_MARGIN
            args.extend(["--max-count", str(max_count)])

    # File filtering options
    if params.glob:
        args.extend(["--glob", params.glob])
    if params.type:
        args.extend(["--type", params.type])

    # Output mode
    if params.output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif params.output_mode == "count_matches":
        args.append("--count-matches")

    # Separate pattern from flags to avoid ambiguity (e.g. pattern starting with -)
    args.append("--")
    args.append(params.pattern)
    # Use the resolved path when available so ripgrep's output matches the
    # search_base used for prefix stripping (fixes Windows short/long path
    # mismatches with tempfile directories).
    args.append(resolved_path or os.path.expanduser(normalize_user_path(params.path)))

    if rtk_path is not None:
        args = [rtk_path, *args]

    return args


def _format_cmd(params: Params, *, rg_cmd: str = _RG_CMD, rtk_path: str | None = None) -> str:
    """Format the equivalent ripgrep command string for display."""
    args = _build_rg_args(rg_cmd, params, rtk_path=rtk_path)
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


class Grep(CallableTool2[Params]):
    name: str = "Grep"
    description: str = (
        "Search files using ripgrep. "
        "Multiline patterns match across line boundaries. "
        "Example: pattern='def foo\\(.*?\\):' with multiline=True matches function signatures "
        "spanning multiple lines."
    )
    params: type[Params] = Params
    field_aliases = {
        **FIELD_ALIASES_GENERAL,
        **FIELD_ALIASES_FILE,
        **FIELD_ALIASES_WEB,
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

        try:
            builder = ToolResultBuilder()
            message = ""

            # Resolve search path against the session work directory.
            search_path = (
                local_path_for_cwd(self._work_dir)
                / Path(normalize_user_path(params.path)).expanduser()
            ).resolve()

            # Windows reserved device names (NUL, CON, etc.) cause os error 1.
            if _is_windows_reserved_name(str(search_path)):
                return ToolError(
                    message=(
                        f"`{params.path}` is a reserved device name on Windows "
                        f"and cannot be searched."
                    ),
                    brief=f"Reserved device name | {_format_cmd(params, rtk_path=rtk_path)}",
                )

            # Validate workspace using the work-dir-resolved path.
            logical_search_path = KaosPath(str(search_path)).canonical()
            original_is_absolute = kaos_path_from_user_input(params.path).is_absolute()
            if (
                not is_within_workspace(
                    logical_search_path, self._work_dir, self._additional_dirs
                )
                and not original_is_absolute
            ):
                display_path = params.path.replace("\\", "/")
                return ToolError(
                    message=f"`{display_path}` is outside the workspace.",
                    brief=f"Path outside workspace | {_format_cmd(params, rtk_path=rtk_path)}",
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

            # Execute search as async subprocess (non-blocking, cancellable)
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._work_dir),
                env=_env_with_shared_bin_path(),
            )

            # Stream stdout/stderr incrementally with buffer limit
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
                    timeout=params.timeout,
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

            # truncated_flag is set synchronously inside _read_stream at the
            # moment of truncation, so it's available even after timeout.
            buffer_truncated = stdout_truncated_flag[0]

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
                        brief=f"Grep timed out | {_format_cmd(params, rtk_path=rtk_path)}",
                    )
                timeout_msg = f"Grep timed out after {params.timeout}s. Partial results returned."
                message = f"{message} {timeout_msg}" if message else timeout_msg

            # rg exit codes: 0=matches found, 1=no matches, 2+=error
            if not timed_out and process.returncode not in (0, 1):
                # EAGAIN: retry once with single-threaded mode
                if not _retry and _is_eagain(stderr_str):
                    logger.warning("rg EAGAIN error, retrying with -j 1")
                    return await self.__call__(params, _retry=True)
                return ToolError(
                    message=f"Failed to grep. Error: {stderr_str}",
                    brief=f"Failed to grep | {_format_cmd(params, rtk_path=rtk_path)}",
                )

            # --- Post-processing pipeline ---
            lines = output.splitlines()
            if lines and lines[-1] == "":
                lines.pop()

            files_truncated_early = False
            total_raw_files = 0

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
            search_base = str(search_path)
            if search_path.is_file():
                search_base = str(search_path.parent)
            lines = _strip_path_prefix(lines, search_base)

            # Step 3: filter sensitive files from output
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

            # Step 4: count_matches summary (before pagination, on full results)
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

            # Step 5: offset + head_limit pagination
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

            lines = _normalize_output_lines(lines, params.output_mode)
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

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Grep failed: pattern={pattern}, path={path}: {error}",
                pattern=params.pattern,
                path=params.path,
                error=e,
            )
            return ToolError(
                message=f"Failed to grep. Error: {str(e)}",
                brief=f"Failed to grep | {_format_cmd(params, rtk_path=rtk_path)}",
            )

    async def backup_grep(self, params: Params) -> ToolReturnValue:
        try:
            if not params.pattern:
                return ToolError(
                    message="Pattern cannot be empty.",
                    brief=f"Empty pattern | {_format_cmd(params)}",
                )

            flags = 0
            if params.ignore_case:
                flags |= re.IGNORECASE
            if params.multiline:
                flags |= re.DOTALL

            try:
                regex = _compile_regex_cached(params.pattern, flags)
            except re.error as e:
                return ToolError(
                    message=f"Invalid regex pattern: {e}",
                    brief=f"Invalid pattern | {_format_cmd(params)}",
                )

            search_path = (
                local_path_for_cwd(self._work_dir)
                / Path(normalize_user_path(params.path)).expanduser()
            ).resolve()

            # Windows reserved device names (NUL, CON, etc.) cause os error 1.
            if _is_windows_reserved_name(str(search_path)):
                display_path = params.path.replace("\\", "/")
                return ToolError(
                    message=(
                        f"`{display_path}` is a reserved device name on Windows "
                        f"and cannot be searched."
                    ),
                    brief=f"Reserved device name | {_format_cmd(params)}",
                )

            # Validate workspace
            logical_search_path = KaosPath(str(search_path)).canonical()
            original_is_absolute = kaos_path_from_user_input(params.path).is_absolute()
            if (
                not is_within_workspace(logical_search_path, self._work_dir, self._additional_dirs)
                and not original_is_absolute
            ):
                display_path = params.path.replace("\\", "/")
                return ToolError(
                    message=f"`{display_path}` is outside the workspace.",
                    brief=f"Path outside workspace | {_format_cmd(params)}",
                )

            # Translate search path through VFS for I/O
            if self._vfs is not None:
                with contextlib.suppress(ValueError):
                    search_path = self._vfs.translate_path(search_path)

            if not search_path.exists():
                display_path = params.path.replace("\\", "/")
                return ToolError(
                    message=f"`{display_path}` does not exist.",
                    brief=f"Path not found | {_format_cmd(params)}",
                )

            output_mode = params.output_mode

            # Collect candidate files.
            files = self._collect_files(search_path, params)

            # Execute search in parallel across files.
            loop = asyncio.get_running_loop()
            max_workers = min(32, (os.cpu_count() or 1) + 4)

            def _process_one(file_path: Path) -> list[str]:
                text = _read_file_text(file_path, self._vfs)
                if text is None:
                    return []

                if output_mode == "files_with_matches":
                    if regex.search(text):
                        return [str(file_path)]
                    return []

                if output_mode == "count_matches":
                    count = len(list(regex.finditer(text)))
                    if count > 0:
                        return [f"{file_path}:{count}"]
                    return []

                # content mode
                return self._search_content_single(file_path, text, regex, params)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    loop.run_in_executor(executor, _process_one, f) for f in files
                ]
                results = await asyncio.gather(*futures)
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

            # Strip search-base prefix for relative paths.
            search_base = str(search_path)
            if search_path.is_file():
                search_base = str(search_path.parent)
            lines = _strip_path_prefix(lines, search_base)

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

            lines = _normalize_output_lines(lines, output_mode)
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
        if not _matches_glob(file_path, params.glob):
            return False
        return _matches_type(file_path, params.type)

    def _search_content_single(
        self, file_path: Path, content: str, regex: re.Pattern[str], params: Params
    ) -> list[str]:
        before = params.before_context or 0
        after = params.after_context or 0
        if params.context is not None:
            before = after = params.context

        if not content:
            return []

        lines = content.splitlines()
        match_line_nums: set[int] = set()

        if params.multiline:
            for m in regex.finditer(content):
                start_line = content.count("\n", 0, m.start()) + 1
                end_line = content.count("\n", 0, m.end()) + 1
                for ln in range(start_line, end_line + 1):
                    match_line_nums.add(ln)
        else:
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    match_line_nums.add(i)

        if not match_line_nums:
            return []

        intervals = [(ln - before, ln + after) for ln in match_line_nums]
        merged = _merge_intervals(intervals)

        results: list[str] = []
        for i, (start, end) in enumerate(merged):
            if i > 0:
                results.append("--")
            for ln in range(max(1, start), min(len(lines), end) + 1):
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
