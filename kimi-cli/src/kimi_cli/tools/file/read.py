import asyncio
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import (
    CallableTool2,
    ToolError,
    ToolOk,
    ToolReturnValue,
    _COMMON_FIELD_ALIASES,
)
from pydantic import BaseModel, Field, model_validator

from kimi_cli.session import Session
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.file.utils import MEDIA_SNIFF_BYTES, detect_file_type
from kimi_cli.tools.utils import load_desc, truncate_line
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import (
    is_within_workspace,
    kaos_path_from_tool_input,
    kaos_path_from_user_input,
)
from kimi_cli.utils.sensitive import is_sensitive_file
from kimi_cli.vfs import VFS

from .glob import _get_gitignore_rules, _is_ignored_by_gitignore
from .utils import resolve_vfs

MAX_LINES = 5000
MAX_LINE_LENGTH = 4000
MAX_FILES = 32

_DEFAULT_READ_MAX_BYTES = 100 << 10  # 100 KiB fallback

MAX_BYTES = _DEFAULT_READ_MAX_BYTES  # kept for backward compatibility


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    path: str | list[str] = Field(
        alias="file_path",  # common LLM variant
        description=(
            "File path, or a list of file paths. Accepts `path` or `file_path`. "
            "When `glob=True`, the final path component may contain wildcards "
            "(`*`, `?`, `[...]`), and recursive patterns starting with `**` are not allowed. "
            "Absolute for files outside working directory."
        )
    )
    line_offset: int = Field(
        default=1,
        description=(
            "Start line, 1-based. Negative reads from end. "
            f"Max abs {MAX_LINES}. Applies to all files in multi-file reads."
        ),
    )
    n_lines: int = Field(
        default=MAX_LINES,
        description=(
            f"Lines to read, max {MAX_LINES}. "
            "Applies to all files in multi-file reads."
        ),
    )
    max_char: int = Field(
        default=16000,
        description=(
            "Maximum number of characters to return. "
            "Applies to all files in multi-file reads. "
            "Default 16K balances completeness with context efficiency."
        ),
    )
    char_offset: int = Field(
        default=0,
        description=(
            "Character offset to start returning from. "
            "Applies to all files in multi-file reads."
        ),
    )
    glob: bool = Field(
        default=False,
        description=(
            "When True, treat `path` as a glob pattern (e.g., '*.py', 'src/**/*.ts'). "
            "When False (default), treat `path` as a literal file path."
        ),
    )
    show_line_numbers: bool = Field(
        default=True,
        description=(
            "When True (default), prefix each line with its line number "
            "(e.g., '    42\\tcontent'). "
            "When False, return raw content without line numbers."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "Params":
        n = len(self.path) if isinstance(self.path, list) else 1

        if n > MAX_FILES:
            raise ValueError(f"Cannot read more than {MAX_FILES} files in one call.")

        fields: list[tuple[str, int, int]] = [
            ("line_offset", self.line_offset, -MAX_LINES),
            ("n_lines", self.n_lines, 1),
            ("max_char", self.max_char, 0),
            ("char_offset", self.char_offset, 0),
        ]
        for name, value, min_value in fields:
            if name == "line_offset":
                if value == 0:
                    raise ValueError(
                        f"{name} cannot be 0; use 1 for the first line "
                        "or -1 for the last line"
                    )
                if value < -MAX_LINES:
                    raise ValueError(
                        f"{name} cannot be less than -{MAX_LINES}. "
                        "Use a positive line_offset with the total line count "
                        "to read from a specific position."
                    )
            elif value < min_value:
                raise ValueError(f"{name} must be >= {min_value}.")
        return self


_GLOB_META = frozenset("*?[")


def _is_glob_pattern(raw: str) -> bool:
    """Return True if the raw path contains glob metacharacters."""
    return any(ch in raw for ch in _GLOB_META)


def _split_glob_path(raw: str) -> tuple[str, str]:
    """Return (base_dir, pattern) for a glob path.

    Only the final path component may contain wildcards. If no separator
    exists before the first metacharacter, the base directory defaults to
    the current working directory (`.`).
    """
    norm = raw.replace("\\", "/")
    meta_indices = [norm.find(ch) for ch in _GLOB_META]
    meta_idx = min((idx for idx in meta_indices if idx != -1), default=-1)
    if meta_idx == -1:
        raise ValueError("not a glob pattern")
    sep_idx = norm.rfind("/", 0, meta_idx)
    if sep_idx == -1:
        return ".", raw
    base = raw[:sep_idx]
    if not base:
        base = "."
    pattern = raw[sep_idx + 1 :]
    return base, pattern


class ReadFile(CallableTool2[Params]):
    name: str = "ReadFile"
    params: type[Params] = Params
    field_aliases = {
        **_COMMON_FIELD_ALIASES,
        "files": "path",
        "paths": "path",
    }

    def __init__(
        self,
        runtime: Runtime,
        session: Session,
        vfs: VFS | None = None,
    ) -> None:
        self.session_id = session.id
        self._session = session
        description = load_desc(
            Path(__file__).parent / "read.md",
            {
                "MAX_LINES": MAX_LINES,
                "MAX_LINE_LENGTH": MAX_LINE_LENGTH,
                "MAX_BYTES": MAX_BYTES,
                "MAX_FILES": MAX_FILES,
            },
        )
        super().__init__(description=description)
        self._runtime = runtime
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._vfs = vfs

    async def _validate_path(
        self, path: KaosPath, raw_path: str
    ) -> ToolError | None:
        """Validate that the path is safe to read."""
        resolved_path = path.canonical()
        original_is_absolute = kaos_path_from_user_input(raw_path).is_absolute()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not original_is_absolute
        ):
            # Outside files can only be read with absolute paths
            return ToolError(
                message=(
                    f"`{raw_path}` is not an absolute path. "
                    "You must provide an absolute path to read a file "
                    "outside the working directory."
                ),
                brief="Invalid path",
            )

        protected_paths = self._session.custom_config.get("config_json", {}).get("protected_read_paths")
        if protected_paths:
            from .utils import check_path_protected
            if matched := check_path_protected(resolved_path, protected_paths, self._work_dir):
                return ToolError(
                    message=f"Reading `{path}` is blocked by protected path rule: `{matched}`.",
                    brief="Protected path",
                )
        return None

    async def _validate_glob_directory(
        self,
        dir_path: KaosPath,
        raw_path: str,
        base_str: str,
    ) -> ToolError | None:
        """Validate that the directory is safe to search for glob expansion."""
        resolved_path = dir_path.canonical()
        original_is_absolute = kaos_path_from_user_input(base_str).is_absolute()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not original_is_absolute
        ):
            return ToolError(
                message=(
                    f"`{raw_path}` is not an absolute path. "
                    "You must provide an absolute path to read outside the working directory."
                ),
                brief="Invalid path",
            )

        protected_paths = self._session.custom_config.get("config_json", {}).get("protected_read_paths")
        if protected_paths:
            from .utils import check_path_protected
            if matched := check_path_protected(resolved_path, protected_paths, self._work_dir):
                return ToolError(
                    message=f"Reading `{raw_path}` is blocked by protected path rule: `{matched}`.",
                    brief="Protected path",
                )
        return None

    async def _expand_glob_path(
        self,
        raw_path: str,
        options: tuple[int, int, int, int],
    ) -> tuple[list[tuple[str, tuple[int, int, int, int]]], ToolError | None]:
        """Expand a single glob path into concrete (path_string, options) entries."""
        base_str, pattern = _split_glob_path(raw_path)

        # Reject recursive patterns that start with **, matching Glob's safety rule.
        if pattern.replace("\\", "/").startswith("**"):
            return [], ToolError(
                message=f"Pattern `{raw_path}` starts with `**`, which is disallowed.",
                brief="Unsafe glob pattern",
            )

        try:
            if base_str == ".":
                base_str = str(self._work_dir)
            base = kaos_path_from_tool_input(base_str, self._work_dir)
            if err := await self._validate_glob_directory(base, raw_path, base_str):
                return [], err

            base = await resolve_vfs(str(base), self._vfs, for_write=False, work_dir=self._work_dir)
            if not await base.exists():
                return [], ToolError(
                    message=f"Directory for `{raw_path}` does not exist.",
                    brief="Directory not found",
                )
            if not await base.is_dir():
                return [], ToolError(
                    message=f"`{raw_path}` is not a directory.",
                    brief="Invalid path",
                )

            # Load gitignore rules for the search root.
            gitignore_rules: list = []
            try:
                resolved_base = Path(str(base)).resolve()
                gitignore_rules = await asyncio.to_thread(
                    _get_gitignore_rules, resolved_base
                )
            except Exception:
                pass

            matches: list[KaosPath] = []
            async for match in base.glob(pattern):
                if not await match.is_file():
                    continue
                if gitignore_rules:
                    try:
                        match_resolved = Path(str(match)).resolve()
                        if _is_ignored_by_gitignore(
                            match_resolved, gitignore_rules, resolved_base
                        ):
                            continue
                    except Exception:
                        pass
                matches.append(match)

            matches.sort()

            if not matches:
                return [], ToolError(
                    message=f"No files matched pattern `{raw_path}`.",
                    brief="No matches",
                )

            # Prefer a path relative to the work dir for display; fall back to absolute.
            entries: list[tuple[str, tuple[int, int, int, int]]] = []
            for match in matches:
                display = str(match)
                try:
                    display = str(match.relative_to(self._work_dir))
                except Exception:
                    pass
                entries.append((display, options))
            return entries, None

        except Exception as e:
            logger.warning("ReadFile glob expansion failed: {path}: {error}", path=raw_path, error=e)
            return [], ToolError(
                message=f"Failed to expand glob `{raw_path}`: {e}",
                brief="Glob expansion failed",
            )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        raw_paths: list[str] = [params.path] if isinstance(params.path, str) else params.path

        if not raw_paths:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )
        if len(raw_paths) > MAX_FILES:
            return ToolError(
                message=f"Cannot read more than {MAX_FILES} files in one call.",
                brief="Too many files",
            )

        # Common options for all files (scalar params broadcast)
        options = (
            params.line_offset,
            params.n_lines,
            params.max_char,
            params.char_offset,
        )

        # Expand any glob paths into concrete file entries while preserving order.
        entries: list[tuple[str, tuple[int, int, int, int], str | ToolError]] = []
        for i, raw_path in enumerate(raw_paths):
            if params.glob:
                # Explicit glob mode: always expand
                if _is_glob_pattern(raw_path):
                    concrete, err = await self._expand_glob_path(raw_path, options)
                else:
                    # No glob metacharacters — treat as a literal single-file path
                    concrete = [(raw_path, options)]
                    err = None
                if err is not None:
                    entries.append((raw_path, options, err))
                else:
                    for path_str, opts in concrete:
                        try:
                            canonical = str(kaos_path_from_tool_input(path_str, self._work_dir).canonical())
                            entries.append((path_str, opts, canonical))
                        except Exception as e:
                            logger.warning(
                                "ReadFile path resolution failed: {path}: {error}",
                                path=path_str, error=e,
                            )
                            err = ToolError(
                                message=f"Invalid path `{path_str}`: {e}",
                                brief="Invalid path",
                            )
                            entries.append((path_str, opts, err))
            else:
                # glob=False (default): treat as literal path, no auto-detection
                try:
                    canonical = str(kaos_path_from_tool_input(raw_path, self._work_dir).canonical())
                    entries.append((raw_path, options, canonical))
                except Exception as e:
                    logger.warning(
                        "ReadFile path resolution failed: {path}: {error}",
                        path=raw_path, error=e,
                    )
                    err = ToolError(
                        message=f"Invalid path `{raw_path}`: {e}",
                        brief="Invalid path",
                    )
                    entries.append((raw_path, options, err))

        # Deduplicate concrete files by canonical path, preserving order and the
        # first options tuple. Error entries are kept as-is.
        seen_canonical: set[str] = set()
        deduped_entries: list[tuple[str, tuple[int, int, int, int], str | ToolError]] = []
        for path_str, options, marker in entries:
            if isinstance(marker, ToolError):
                deduped_entries.append((path_str, options, marker))
            elif marker not in seen_canonical:
                seen_canonical.add(marker)
                deduped_entries.append((path_str, options, marker))

        file_count = sum(
            1 for _, _, marker in deduped_entries if not isinstance(marker, ToolError)
        )
        if file_count > MAX_FILES:
            return ToolError(
                message=f"Cannot read more than {MAX_FILES} files in one call.",
                brief="Too many files",
            )

        results: list[ToolReturnValue] = []
        display_paths: list[str] = []
        success_count = 0
        error_count = 0
        for path_str, opts, marker in deduped_entries:
            if isinstance(marker, ToolError):
                result = marker
                error_count += 1
            else:
                line_offset, n_lines, max_char, char_offset = opts
                result = await self._read_single_file(
                    path_str, line_offset, n_lines, char_offset, max_char,
                    show_line_numbers=params.show_line_numbers,
                )
                if result.is_error:
                    error_count += 1
                else:
                    success_count += 1
            display_paths.append(path_str.replace("\\", "/"))
            results.append(result)

        # Single-file reads keep the original output format for backward compatibility.
        if len(deduped_entries) == 1:
            return results[0]

        if success_count == 0:
            messages = [r.message for r in results]
            return ToolError(
                message=f"Failed to read {error_count} file(s). " + " ".join(messages),
                brief="Failed to read files",
            )

        parts: list[str] = []
        for idx, (display_path, result) in enumerate(zip(display_paths, results)):
            parts.append(f"======== {display_path} ========")
            if result.is_error:
                parts.append(result.message)
            else:
                parts.append(result.output)
            if idx < len(deduped_entries) - 1:
                parts.append("")
        final_output = "\n".join(parts)

        messages = [r.message for r in results]
        final_message = f"Read {success_count} file(s), {error_count} error(s). " + " ".join(messages)
        return ToolOk(
            output=final_output,
            message=final_message,
            brief=f"Read {success_count} files",
        )

    async def _read_single_file(
        self,
        raw_path: str,
        line_offset: int,
        n_lines: int,
        char_offset: int,
        max_char: int,
        *,  # keyword-only from here
        show_line_numbers: bool = True,
    ) -> ToolReturnValue:
        display_path = raw_path.replace("\\", "/")
        if not raw_path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )

        try:
            p = kaos_path_from_tool_input(raw_path, self._work_dir)
            logical_path = p
            if err := await self._validate_path(p, raw_path):
                return err

            p = await resolve_vfs(raw_path, self._vfs, for_write=False, work_dir=self._work_dir)

            if is_sensitive_file(str(logical_path)):
                return ToolError(
                    message=(
                        f"`{display_path}` appears to contain secrets "
                        "(matched sensitive file pattern). "
                        "Reading this file is blocked to protect credentials."
                    ),
                    brief="Sensitive file",
                )

            if not await p.exists():
                return ToolError(
                    message=f"`{display_path}` does not exist.",
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{display_path}` is not a file.",
                    brief="Invalid path"
                )

            header = await p.read_bytes(MEDIA_SNIFF_BYTES)
            file_type = detect_file_type(str(logical_path), header=header)
            if file_type.kind in ("image", "video"):
                return ToolError(
                    message=(
                        f"`{display_path}` is a {file_type.kind} file. "
                        "Use other appropriate tools to read image or video files."
                    ),
                    brief="Unsupported file type",
                )

            if file_type.kind == "unknown":
                return ToolError(
                    message=(
                        f"`{display_path}` seems not readable. "
                        "You may need to read it with proper shell commands, Python tools "
                        "or MCP tools if available. "
                        "If you read/operate it with Python, you MUST ensure that any "
                        "third-party packages are installed in a virtual environment (venv)."
                    ),
                    brief="File not readable",
                )

            assert n_lines >= 1
            assert line_offset != 0

            if line_offset < 0:
                result = await self._read_tail(p, display_path, line_offset, n_lines, show_line_numbers=show_line_numbers)
            else:
                result = await self._read_forward(p, display_path, line_offset, n_lines, show_line_numbers=show_line_numbers)

            if isinstance(result, ToolOk):
                if isinstance(result.output, str):
                    result.output = result.output[char_offset:max_char]
                self._session.file_mtime.clean_file(raw_path)
            return result
        except Exception as e:
            logger.warning("ReadFile failed: {path}: {error}", path=raw_path, error=e)
            return ToolError(
                message=f"Failed to read {display_path}. Error: {e}",
                brief="Failed to read file",
            )

    async def _read_forward(
        self,
        p: KaosPath,
        display_path: str,
        line_offset: int,
        n_lines: int,
        *,  # keyword-only
        show_line_numbers: bool = True,
    ) -> ToolReturnValue:
        """Read file from a positive line_offset."""
        lines_with_no: list[str] = []
        n_bytes = 0
        truncated_line_numbers: list[int] = []
        max_lines_reached = False
        max_bytes_reached = False
        current_line_no = 0
        target_lines = min(n_lines, MAX_LINES)
        eof_reached = True

        async for line in p.read_lines(errors="replace"):
            current_line_no += 1
            if current_line_no < line_offset:
                continue
            truncated = truncate_line(line, MAX_LINE_LENGTH)
            if truncated != line:
                truncated_line_numbers.append(current_line_no)
            b_len = len(truncated.encode("utf-8"))
            if show_line_numbers:
                lines_with_no.append(f"{current_line_no:6d}\t{truncated}")
            else:
                lines_with_no.append(truncated)
            n_bytes += b_len
            if len(lines_with_no) >= target_lines:
                max_lines_reached = target_lines >= MAX_LINES
                eof_reached = False
                break
            if n_bytes >= MAX_BYTES:
                max_bytes_reached = True
                eof_reached = False
                break

        start_line = line_offset

        message = (
            f"{len(lines_with_no)} lines read from file starting from line {start_line}."
            if len(lines_with_no) > 0
            else "No lines read from file."
        )
        if eof_reached:
            message += f" Total lines in file: {current_line_no}."
        if max_lines_reached:
            message += f" Max {MAX_LINES} lines reached."
        elif max_bytes_reached:
            message += f" Max {MAX_BYTES} bytes reached."
        elif len(lines_with_no) < n_lines:
            message += " End of file reached."
        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."
        message += f" Path: {display_path}"
        return ToolOk(
            output="".join(lines_with_no),
            message=message,
            brief="Read file",
        )

    async def _read_tail(
        self,
        p: KaosPath,
        display_path: str,
        line_offset: int,
        n_lines: int,
        *,  # keyword-only
        show_line_numbers: bool = True,
    ) -> ToolReturnValue:
        """Read file from a negative line_offset (tail mode)."""
        tail_count = abs(line_offset)
        line_limit = min(n_lines, MAX_LINES)

        # Bounded list keeping the last `tail_count` lines.
        tail_buf: list[tuple[int, str, bool, int]] = []
        current_line_no = 0
        async for line in p.read_lines(errors="replace"):
            current_line_no += 1
            truncated = truncate_line(line, MAX_LINE_LENGTH)
            b_len = len(truncated.encode("utf-8"))
            tail_buf.append((current_line_no, truncated, truncated != line, b_len))
            if len(tail_buf) > tail_count:
                tail_buf.pop(0)

        total_lines = current_line_no

        # Apply n_lines / MAX_LINES from head of tail_buf.
        candidates = tail_buf[:line_limit]
        max_lines_reached = len(tail_buf) > MAX_LINES and len(candidates) == MAX_LINES

        # Apply max_bytes — reverse-scan to keep the newest lines that fit.
        max_bytes = MAX_BYTES
        if candidates:
            total_candidate_bytes = sum(entry[3] for entry in candidates)
            if total_candidate_bytes > max_bytes:
                max_bytes_reached = True
                kept = 0
                n_bytes = 0
                for entry in reversed(candidates):
                    n_bytes += entry[3]
                    if n_bytes > max_bytes:
                        break
                    kept += 1
                candidates = candidates[len(candidates) - kept :]
            else:
                max_bytes_reached = False
        else:
            max_bytes_reached = False

        # Build output directly.
        lines_with_no: list[str] = []
        truncated_line_numbers: list[int] = []
        for line_no, truncated, was_truncated, _ in candidates:
            if was_truncated:
                truncated_line_numbers.append(line_no)
            if show_line_numbers:
                lines_with_no.append(f"{line_no:6d}\t{truncated}")
            else:
                lines_with_no.append(truncated)

        start_line = candidates[0][0] if candidates else total_lines + 1
        message = (
            f"{len(lines_with_no)} lines read from file starting from line {start_line}."
            if len(lines_with_no) > 0
            else "No lines read from file."
        )
        message += f" Total lines in file: {total_lines}."
        if max_lines_reached:
            message += f" Max {MAX_LINES} lines reached."
        elif max_bytes_reached:
            message += f" Max {max_bytes} bytes reached."
        elif len(lines_with_no) < n_lines:
            message += " End of file reached."
        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."
        message += f" Path: {display_path}"
        return ToolOk(
            output="".join(lines_with_no),
            message=message,
            brief="Read file",
        )
