from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable
from contextlib import suppress
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import (
    _COMMON_FIELD_ALIASES,
    CallableTool2,
    ToolError,
    ToolOk,
    ToolReturnValue,
    alias_note,
)
from pydantic import AliasChoices, BaseModel, Field, model_validator
from rapidfuzz import fuzz

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

from .glob import (
    _get_gitignore_rules,
    _is_ignored_by_gitignore,
    _is_unsafe_recursive_pattern,
)
from .read_extract import (
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)
from .utils import resolve_vfs

MAX_LINES = 5000
MAX_LINE_LENGTH = 4000
MAX_FILES = 32

_DEFAULT_READ_MAX_BYTES = 100 << 10  # 100 KiB fallback

MAX_BYTES = _DEFAULT_READ_MAX_BYTES  # kept for backward compatibility

# Documents larger than this are never extracted into text.
MAX_EXTRACT_BYTES = 50 * 1024 * 1024

# Similar-file suggestion tuning for the "does not exist" branch.
_PARENT_LISTING_CAP = 200
_MAX_SUGGESTIONS = 3
_SIMILARITY_CUTOFF = 65


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    file_path: str | list[str] = Field(
        validation_alias=AliasChoices("file_path", "path"),
        description=(
            "Path to read, resolved by the filesystem backend. "
            + alias_note("file_path", "path", word=False)
            + " May be a single file path or a list of file paths. "
            "When `glob=True`, the final path component may contain wildcards "
            "(`*`, `?`, `[...]`); recursive patterns like `src/**/*.ts` are "
            "supported, only unsafe all-wildcard patterns (e.g. `**`, `**/*`) "
            "are rejected."
        ),
    )
    offset: int | list[int] = Field(
        default=1,
        validation_alias=AliasChoices("offset", "line_offset"),
        description=(
            "1-based first line to return. Defaults to 1. "
            + alias_note("offset", "line_offset", word=False)
            + " Negative reads from end. "
            f"Max abs {MAX_LINES}. May be a scalar applied to all files, "
            "or a list with one value per file path."
        ),
    )
    limit: int | list[int] = Field(
        default=2000,
        validation_alias=AliasChoices("limit", "n_lines"),
        description=(
            "Maximum number of lines to return. Defaults to 2000. "
            + alias_note("limit", "n_lines", word=False)
            + f" Max {MAX_LINES}. May be a scalar applied to all files, "
            "or a list with one value per file path."
        ),
    )
    max_char: int | list[int] = Field(
        default=16000,
        description=(
            "Maximum number of characters to return (starting from char_offset). "
            "May be a scalar applied to all files, "
            "or a list with one value per file path. "
            "Default 16K balances completeness with context efficiency."
        ),
    )
    char_offset: int | list[int] = Field(
        default=0,
        description=(
            "Character offset to start returning from. "
            "May be a scalar applied to all files, "
            "or a list with one value per file path."
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
    def _validate(self) -> Params:
        n = len(self.file_path) if isinstance(self.file_path, list) else 1

        if n > MAX_FILES:
            raise ValueError(f"Cannot read more than {MAX_FILES} files in one call.")

        for name in ("offset", "limit", "max_char", "char_offset"):
            value = getattr(self, name)
            if isinstance(value, list):
                if len(value) != n:
                    raise ValueError(
                        f"{name} must be a scalar or a list with one value per "
                        f"file path ({len(value)} values given for {n} file(s))."
                    )
                values = value
            else:
                values = [value]
            for v in values:
                self._validate_value(name, v)
        return self

    @staticmethod
    def _validate_value(name: str, value: int) -> None:
        if name == "offset":
            if value == 0:
                raise ValueError(
                    f"{name} cannot be 0; use 1 for the first line or -1 for the last line"
                )
            if value < -MAX_LINES:
                raise ValueError(
                    f"{name} cannot be less than -{MAX_LINES}. "
                    "Use a positive offset with the total line count "
                    "to read from a specific position."
                )
            return
        min_value = {"limit": 1, "max_char": 0, "char_offset": 0}[name]
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value}.")


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


def _broadcast_option(value: int | list[int], n: int) -> list[int]:
    """Broadcast a scalar option to *n* entries, or return the per-file list."""
    return value if isinstance(value, list) else [value] * n


def _apply_char_window(
    result: ToolReturnValue,
    char_offset: int,
    max_char: int,
) -> ToolReturnValue:
    """Apply the ``char_offset``/``max_char`` window to a read result.

    The line/byte budgets inside ``_render_result`` already surface in
    ``message`` ("Max N bytes reached", "End of file reached"), but the char
    window is applied afterwards and would otherwise hide content *silently*
    while the message claims the whole file was shown. When the window hides
    any rendered content, append an explicit notice so the agent knows the
    read was partial and how to continue it.
    """
    if not isinstance(result, ToolOk) or not isinstance(result.output, str):
        return result
    original = result.output
    result.output = original[char_offset : char_offset + max_char]
    total = len(original)
    end = char_offset + max_char
    if end < total or char_offset > 0:
        if char_offset > 0 and end < total:
            where = f"middle chars {char_offset}..{end} of {total}"
            hidden = "content before and after is hidden"
        elif char_offset > 0:
            where = f"tail chars {char_offset}..{total} of {total}"
            hidden = "content before is hidden"
        else:
            where = f"head chars 0..{end} of {total}"
            hidden = "content after is hidden"
        result.message = (result.message or "") + (
            f" NOTE: output window shows {where} ({hidden}); max_char={max_char}. "
            "Raise max_char / adjust char_offset to read the rest."
        )
    return result


def _similar_names(
    parent: Path,
    requested: str,
    cap: int,
    top_n: int,
    cutoff: int,
) -> list[str]:
    """Rank sibling file names by fuzzy similarity to *requested*.

    Only regular files are considered (the requested name itself is skipped)
    and the listing is capped at *cap* entries. Scoring is extension-aware:
    both the full basename and the basename-without-suffix are compared with
    rapidfuzz. Returns at most *top_n* names with score >= *cutoff*.
    """
    candidates: list[str] = []
    try:
        with os.scandir(parent) as it:
            for i, entry in enumerate(it):
                if i >= cap:
                    break
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                name = entry.name
                if name == requested:
                    continue
                candidates.append(name)
    except OSError:
        return []
    if not candidates:
        return []

    requested_stem = Path(requested).stem

    def _score(choice: str) -> float:
        # Extension-aware scoring: full basename and basename-without-suffix.
        return max(
            fuzz.ratio(requested, choice),
            fuzz.ratio(requested_stem, Path(choice).stem),
        )

    # NOTE: rapidfuzz's ``process.extract`` treats a Python callable scorer's
    # return value as a 0..1 normalized similarity and rescales it, so a
    # 0..100 scorer always comes back as 100.0 and the cutoff is defeated.
    # Score manually instead so ``cutoff`` is honored.
    scored = ((_score(choice), choice) for choice in candidates)
    best = sorted((s for s in scored if s[0] >= cutoff), key=lambda t: t[0], reverse=True)
    return [choice for _score, choice in best[:top_n]]


class ReadFile(CallableTool2[Params]):
    name: str = "read"
    params: type[Params] = Params
    field_aliases = {
        **_COMMON_FIELD_ALIASES,
        "files": "file_path",
        "paths": "file_path",
        "path": "file_path",
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

    async def _validate_path(self, path: KaosPath, raw_path: str) -> ToolError | None:
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

        protected_paths = self._session.custom_config.get("config_json", {}).get(
            "protected_read_paths"
        )
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

        protected_paths = self._session.custom_config.get("config_json", {}).get(
            "protected_read_paths"
        )
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

        # Reject unsafe all-wildcard recursive patterns (e.g. `**`, `**/*`),
        # matching Glob's safety rule. `src/**/*.ts` etc. are allowed.
        if _is_unsafe_recursive_pattern(raw_path):
            return [], ToolError(
                message=(
                    f"Pattern `{raw_path}` is an unsafe recursive pattern, which is disallowed."
                ),
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
                gitignore_rules = await asyncio.to_thread(_get_gitignore_rules, resolved_base)
            except Exception:
                pass

            matches: list[KaosPath] = []
            async for match in base.glob(pattern):
                if not await match.is_file():
                    continue
                if gitignore_rules:
                    try:
                        match_resolved = Path(str(match)).resolve()
                        if _is_ignored_by_gitignore(match_resolved, gitignore_rules, resolved_base):
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
                with suppress(Exception):
                    display = str(match.relative_to(self._work_dir))
                entries.append((display, options))
            return entries, None

        except Exception as e:
            logger.warning(
                "read glob expansion failed: {path}: {error}",
                path=raw_path,
                error=e,
            )
            return [], ToolError(
                message=f"Failed to expand glob `{raw_path}`: {e}",
                brief="Glob expansion failed",
            )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        raw_paths: list[str] = (
            [params.file_path] if isinstance(params.file_path, str) else params.file_path
        )

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

        # Per-entry options: scalars broadcast, lists apply per file path.
        n_raw = len(raw_paths)
        line_offsets = _broadcast_option(params.offset, n_raw)
        n_lines_values = _broadcast_option(params.limit, n_raw)
        max_char_values = _broadcast_option(params.max_char, n_raw)
        char_offset_values = _broadcast_option(params.char_offset, n_raw)

        # Expand any glob paths into concrete file entries while preserving order.
        entries: list[tuple[str, tuple[int, int, int, int], str | ToolError]] = []
        for i, raw_path in enumerate(raw_paths):
            options = (
                line_offsets[i],
                n_lines_values[i],
                max_char_values[i],
                char_offset_values[i],
            )
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
                            canonical = str(
                                kaos_path_from_tool_input(path_str, self._work_dir).canonical()
                            )
                            entries.append((path_str, opts, canonical))
                        except Exception as e:
                            logger.warning(
                                "read path resolution failed: {path}: {error}",
                                path=path_str,
                                error=e,
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
                        "read path resolution failed: {path}: {error}",
                        path=raw_path,
                        error=e,
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

        file_count = sum(1 for _, _, marker in deduped_entries if not isinstance(marker, ToolError))
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
                    path_str,
                    line_offset,
                    n_lines,
                    char_offset,
                    max_char,
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
        for idx, (display_path, result) in enumerate(zip(display_paths, results, strict=False)):
            parts.append(f"======== {display_path} ========")
            if result.is_error:
                parts.append(result.message)
            else:
                parts.append(result.output)
            if idx < len(deduped_entries) - 1:
                parts.append("")
        final_output = "\n".join(parts)

        messages = [r.message for r in results]
        final_message = f"Read {success_count} file(s), {error_count} error(s). " + " ".join(
            messages
        )
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
                suggestions = await self._similar_file_suggestions(p)
                message = f"`{display_path}` does not exist."
                if suggestions:
                    message += "\n\nDid you mean:\n  " + "\n  ".join(suggestions)
                return ToolError(
                    message=message,
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(message=f"`{display_path}` is not a file.", brief="Invalid path")

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

            if is_extractable_document(str(logical_path)):
                suffix = Path(str(logical_path)).suffix.lower()
                try:
                    stat_result = await p.stat()
                except Exception:
                    stat_result = None
                if stat_result is not None and stat_result.st_size > MAX_EXTRACT_BYTES:
                    return ToolError(
                        message=(
                            f"`{display_path}` is a {suffix} document larger than "
                            f"{MAX_EXTRACT_BYTES} bytes and cannot be extracted as text."
                        ),
                        brief="File not readable",
                    )
                try:
                    extracted = await asyncio.to_thread(extract_document_text, str(p))
                except ExtractionError as e:
                    return ToolError(
                        message=(
                            f"`{display_path}` is a {suffix} document but could not "
                            f"be extracted: {e}"
                        ),
                        brief="Document extraction failed",
                    )
                result = await self._read_content(
                    extracted,
                    display_path,
                    line_offset,
                    n_lines,
                    char_offset,
                    max_char,
                    show_line_numbers=show_line_numbers,
                    suffix=suffix,
                )
                if isinstance(result, ToolOk):
                    self._session.file_mtime.clean_file(raw_path)
                return result

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
                result = await self._read_tail(
                    p,
                    display_path,
                    line_offset,
                    n_lines,
                    show_line_numbers=show_line_numbers,
                )
            else:
                result = await self._read_forward(
                    p,
                    display_path,
                    line_offset,
                    n_lines,
                    show_line_numbers=show_line_numbers,
                )

            if isinstance(result, ToolOk):
                result = _apply_char_window(result, char_offset, max_char)
                self._session.file_mtime.clean_file(raw_path)
            return result
        except Exception as e:
            logger.warning("read failed: {path}: {error}", path=raw_path, error=e)
            return ToolError(
                message=f"Failed to read {display_path}. Error: {e}",
                brief="Failed to read file",
            )

    async def _similar_file_suggestions(
        self,
        p: KaosPath,
        *,
        cap: int = _PARENT_LISTING_CAP,
        top_n: int = _MAX_SUGGESTIONS,
        cutoff: int = _SIMILARITY_CUTOFF,
    ) -> list[str]:
        """Suggest similarly-named sibling files for a missing path.

        Never raises: returns an empty list on any error. The requested name
        itself is skipped and only regular files are considered.
        """
        try:
            parent = Path(str(p)).parent
            requested = Path(str(p)).name
            if not parent.is_dir():
                return []
            return await asyncio.to_thread(_similar_names, parent, requested, cap, top_n, cutoff)
        except Exception:
            return []

    async def _read_content(
        self,
        text: str,
        display_path: str,
        line_offset: int,
        n_lines: int,
        char_offset: int,
        max_char: int,
        *,
        show_line_numbers: bool = True,
        suffix: str = "",
    ) -> ToolReturnValue:
        """Render in-memory text (e.g. extracted documents) like a normal read."""

        async def _text_lines() -> AsyncIterable[str]:
            for line in text.splitlines(keepends=True):
                yield line

        note = f" (extracted from {suffix} document)" if suffix else ""
        result = await self._render_lines(
            _text_lines(),
            display_path,
            line_offset,
            n_lines,
            show_line_numbers=show_line_numbers,
            note=note,
        )
        return _apply_char_window(result, char_offset, max_char)

    async def _render_lines(
        self,
        lines: AsyncIterable[str],
        display_path: str,
        line_offset: int,
        n_lines: int,
        *,
        show_line_numbers: bool = True,
        note: str = "",
    ) -> ToolReturnValue:
        """Render an async iterable of lines (line endings included).

        Positive ``line_offset`` reads forward; negative reads the tail window.
        ``note`` is appended to the success message (e.g. the
        document-extraction notice). Shared by file reads and extracted text.
        """
        assert n_lines >= 1
        assert line_offset != 0

        if line_offset < 0:
            return await self._render_tail(
                lines,
                display_path,
                line_offset,
                n_lines,
                show_line_numbers=show_line_numbers,
                note=note,
            )
        return await self._render_forward(
            lines,
            display_path,
            line_offset,
            n_lines,
            show_line_numbers=show_line_numbers,
            note=note,
        )

    async def _render_forward(
        self,
        lines: AsyncIterable[str],
        display_path: str,
        line_offset: int,
        n_lines: int,
        *,
        show_line_numbers: bool = True,
        note: str = "",
    ) -> ToolReturnValue:
        """Render lines forward from a positive line_offset with line/byte budgets."""
        entries: list[tuple[int, str, bool, int]] = []
        n_bytes = 0
        max_lines_reached = False
        max_bytes_reached = False
        current_line_no = 0
        target_lines = min(n_lines, MAX_LINES)
        eof_reached = True

        async for line in lines:
            current_line_no += 1
            if current_line_no < line_offset:
                continue
            truncated = truncate_line(line, MAX_LINE_LENGTH)
            b_len = len(truncated.encode("utf-8"))
            entries.append((current_line_no, truncated, truncated != line, b_len))
            n_bytes += b_len
            if len(entries) >= target_lines:
                max_lines_reached = target_lines >= MAX_LINES
                eof_reached = False
                break
            if n_bytes >= MAX_BYTES:
                max_bytes_reached = True
                eof_reached = False
                break

        return self._render_result(
            entries,
            display_path,
            n_lines,
            line_offset,
            show_line_numbers=show_line_numbers,
            total_lines=current_line_no if eof_reached else None,
            max_lines_reached=max_lines_reached,
            max_bytes_reached=max_bytes_reached,
            end_of_file=len(entries) < n_lines,
            note=note,
        )

    async def _render_tail(
        self,
        lines: AsyncIterable[str],
        display_path: str,
        line_offset: int,
        n_lines: int,
        *,
        show_line_numbers: bool = True,
        note: str = "",
    ) -> ToolReturnValue:
        """Render the tail window (negative line_offset) with line/byte budgets."""
        tail_count = abs(line_offset)
        line_limit = min(n_lines, MAX_LINES)

        # Bounded list keeping the last `tail_count` lines.
        tail_buf: list[tuple[int, str, bool, int]] = []
        current_line_no = 0
        async for line in lines:
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

        start_line = candidates[0][0] if candidates else total_lines + 1
        return self._render_result(
            candidates,
            display_path,
            n_lines,
            start_line,
            show_line_numbers=show_line_numbers,
            total_lines=total_lines,
            max_lines_reached=max_lines_reached,
            max_bytes_reached=max_bytes_reached,
            end_of_file=len(candidates) < n_lines,
            note=note,
        )

    def _render_result(
        self,
        candidates: list[tuple[int, str, bool, int]],
        display_path: str,
        n_lines: int,
        start_line: int,
        *,
        show_line_numbers: bool,
        total_lines: int | None,
        max_lines_reached: bool,
        max_bytes_reached: bool,
        end_of_file: bool,
        note: str = "",
    ) -> ToolOk:
        """Build the final ToolOk (output + message) from budgeted candidates."""
        lines_with_no: list[str] = []
        truncated_line_numbers: list[int] = []
        for line_no, truncated, was_truncated, _ in candidates:
            if was_truncated:
                truncated_line_numbers.append(line_no)
            if show_line_numbers:
                lines_with_no.append(f"{line_no:6d}\t{truncated}")
            else:
                lines_with_no.append(truncated)

        message = (
            f"{len(lines_with_no)} lines read from file starting from line {start_line}."
            if len(lines_with_no) > 0
            else "No lines read from file."
        )
        if total_lines is not None:
            message += f" Total lines in file: {total_lines}."
        if max_lines_reached:
            message += f" Max {MAX_LINES} lines reached."
        elif max_bytes_reached:
            message += f" Max {MAX_BYTES} bytes reached."
        elif end_of_file:
            message += " End of file reached."
        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."
        message += f" Path: {display_path}"
        if note:
            message += note
        return ToolOk(
            output="".join(lines_with_no),
            message=message,
            brief="Read file",
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
        return await self._render_lines(
            p.read_lines(errors="replace"),
            display_path,
            line_offset,
            n_lines,
            show_line_numbers=show_line_numbers,
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
        return await self._render_lines(
            p.read_lines(errors="replace"),
            display_path,
            line_offset,
            n_lines,
            show_line_numbers=show_line_numbers,
        )
