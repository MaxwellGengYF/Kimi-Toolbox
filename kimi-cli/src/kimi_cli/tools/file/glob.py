"""Glob tool implementation."""

import asyncio
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import kaos_path_from_tool_input
from kimi_cli.vfs import VFS

from .utils import resolve_vfs

MAX_MATCHES = 1000
MAX_BYTES = 100 << 10  # 100KB
GLOB_DESC_PATH = Path(__file__).parent / "glob.md"


def _is_unsafe_recursive_pattern(pattern: str) -> bool:
    """Check if a glob pattern would recursively match all files/dirs.

    Blocks patterns like ``**``, ``**/*``, ``**/**``, ``**\\*`` (Windows),
    or any pattern consisting only of wildcard segments (``*``, ``**``)
    that contains at least one ``**`` segment — these are meaningless
    and can be extremely slow.
    """
    # Normalize Windows backslashes to forward slashes
    p = pattern.replace("\\", "/")
    # Strip leading ./
    p = p.lstrip("./")

    # Split into segments
    parts = p.split("/")

    # Must have at least one ** segment to be recursive-all
    if "**" not in parts:
        return False

    # If every segment is a bare wildcard (* or **), it recurses everything
    return all(part in ("*", "**") for part in parts)


WINDOWS_PATH_HINT = (
    "Windows: `directory` accepts native (`C:\\Users\\foo`) and POSIX-style "
    "(`/c/Users/foo`) paths. Results use backslashes — convert to forward "
    "slashes for shell commands."
)

# Global cache for .gitignore files under a root directory.
# Key: root directory path (str)
# Value: _GitignoreCacheEntry
_GITIGNORE_CACHE: dict[str, _GitignoreCacheEntry] = {}


@dataclass
class _GitignoreRule:
    """A single parsed gitignore rule."""

    pattern: str
    negated: bool
    anchored: bool  # True if pattern contains '/' (not just trailing)
    is_dir_only: bool  # True if pattern ends with '/'
    source_dir: Path  # Directory containing the .gitignore


@dataclass
class _GitignoreCacheEntry:
    """Cached gitignore state for a root directory."""

    gitignore_paths: list[Path] = field(default_factory=list)
    rules: list[_GitignoreRule] = field(default_factory=list)
    mtimes: dict[str, float] = field(default_factory=dict)


def _description_for_os(os_kind: str) -> str:
    return load_desc(
        GLOB_DESC_PATH,
        {
            "MAX_MATCHES": str(MAX_MATCHES),
            "WINDOWS_PATH_HINT": WINDOWS_PATH_HINT if os_kind == "Windows" else "",
        },
    )


def _parse_gitignore(content: str, source_dir: Path) -> list[_GitignoreRule]:
    """Parse a .gitignore file into a list of rules."""
    rules: list[_GitignoreRule] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        if not line:
            continue
        is_dir_only = line.endswith("/")
        if is_dir_only:
            line = line[:-1]
        # Anchored if it contains a slash anywhere (not just trailing)
        anchored = "/" in line
        # Remove leading slash for anchored patterns
        if line.startswith("/"):
            line = line[1:]
            anchored = True
        rules.append(
            _GitignoreRule(
                pattern=line,
                negated=negated,
                anchored=anchored,
                is_dir_only=is_dir_only,
                source_dir=source_dir,
            )
        )
    return rules


def _gitignore_match(path: Path, rel_path: str, is_dir: bool, rule: _GitignoreRule) -> bool:
    """Check if a path matches a single gitignore rule."""
    if rule.is_dir_only and not is_dir:
        return False

    pattern = rule.pattern

    # Handle ** patterns
    if "**" in pattern:
        rel_parts = rel_path.split("/")

        if pattern == "**":
            return True
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            # Match suffix against any suffix of rel_parts
            for i in range(len(rel_parts)):
                sub = "/".join(rel_parts[i:])
                if fnmatch.fnmatch(sub, suffix) or fnmatch.fnmatch(rel_parts[-1], suffix):
                    return True
            return False
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            return rel_path.startswith(prefix + "/") or rel_path == prefix
        if "/**/" in pattern:
            prefix, suffix = pattern.split("/**/", 1)
            if rel_path.startswith(prefix + "/") or rel_path == prefix:
                rest = rel_path[len(prefix) + 1 :] if rel_path.startswith(prefix + "/") else ""
                if not suffix:
                    return True
                # suffix must match somewhere in the rest
                rest_parts = rest.split("/")
                for i in range(len(rest_parts)):
                    sub = "/".join(rest_parts[i:])
                    if fnmatch.fnmatch(sub, suffix) or fnmatch.fnmatch(rest_parts[-1], suffix):
                        return True
            return False

        # Generic ** fallback: replace ** with * and match
        simple_pattern = pattern.replace("**", "*")
        return fnmatch.fnmatch(rel_path, simple_pattern) or fnmatch.fnmatch(rel_path.split("/")[-1], simple_pattern)

    if rule.anchored:
        # Match against the relative path from the gitignore directory
        return fnmatch.fnmatch(rel_path, pattern)
    else:
        # Match against basename or any path component
        basename = rel_path.split("/")[-1]
        if fnmatch.fnmatch(basename, pattern):
            return True
        # Also match if any directory component matches
        return any(fnmatch.fnmatch(part, pattern) for part in rel_path.split("/")[:-1])


def _is_ignored_by_gitignore(
    path: Path, rules: list[_GitignoreRule], root_dir: Path
) -> bool:
    """Check if a path is ignored by any gitignore rule (with negation support)."""
    # Find the deepest .gitignore that applies (i.e., closest ancestor)
    # Standard gitignore behavior: later rules override earlier ones
    # We process all rules in order of discovery (root-first, then deeper)
    ignored = False
    for rule in rules:
        try:
            rel_path = str(path.relative_to(rule.source_dir))
        except ValueError:
            continue
        is_dir = path.is_dir()
        if _gitignore_match(path, rel_path, is_dir, rule):
            ignored = not rule.negated
    return ignored


def _safe_getmtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except (OSError, ValueError):
        return 0.0


def _find_gitignore_files(root: Path) -> list[Path]:
    """Find all .gitignore files under root."""
    gitignores: list[Path] = []
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            if ".gitignore" in filenames:
                gitignores.append(Path(dirpath) / ".gitignore")
    except OSError:
        pass
    return gitignores


def _load_gitignore_rules(root: Path) -> tuple[list[Path], list[_GitignoreRule], dict[str, float]]:
    """Load all gitignore rules under root and their mtimes."""
    gitignore_paths = _find_gitignore_files(root)
    rules: list[_GitignoreRule] = []
    mtimes: dict[str, float] = {}
    for gi_path in gitignore_paths:
        mtime = _safe_getmtime(str(gi_path))
        mtimes[str(gi_path)] = mtime
        try:
            with open(gi_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        rules.extend(_parse_gitignore(content, gi_path.parent))
    return gitignore_paths, rules, mtimes


def _get_gitignore_rules(root: Path) -> list[_GitignoreRule]:
    """Get cached gitignore rules for a root directory, refreshing if needed."""
    global _GITIGNORE_CACHE
    root_str = str(root.resolve())

    cache = _GITIGNORE_CACHE.get(root_str)
    needs_refresh = True

    if cache is not None:
        # Check if any file was modified or deleted
        needs_refresh = False
        # Check if new files appeared or old ones changed
        current_paths = set(str(p) for p in cache.gitignore_paths)
        for path_str, old_mtime in cache.mtimes.items():
            if not os.path.exists(path_str):
                needs_refresh = True
                break
            new_mtime = _safe_getmtime(path_str)
            if new_mtime != old_mtime:
                needs_refresh = True
                break
        if not needs_refresh:
            # Also check if any new .gitignore files were added
            gitignores = _find_gitignore_files(root)
            new_paths = set(str(p) for p in gitignores)
            if new_paths != current_paths:
                needs_refresh = True

    if needs_refresh:
        gitignore_paths, rules, mtimes = _load_gitignore_rules(root)
        cache = _GitignoreCacheEntry(
            gitignore_paths=gitignore_paths,
            rules=rules,
            mtimes=mtimes,
        )
        _GITIGNORE_CACHE[root_str] = cache

    return cache.rules


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    pattern: str = Field(description="Glob pattern. Unsafe recursive patterns (``**``, ``**/*``, ``**/**``, etc.) are forbidden.")
    directory: str | None = Field(
        alias="path",  # common LLM variant
        description="Absolute search path. Defaults to working directory. "
        "Accepts `directory` or `path`.",
        default=None,
    )
    include_dirs: bool = Field(
        description="Include directories in results.",
        default=False,
    )
    respect_gitignore: bool = Field(
        default=True,
        description="When True (default), skip files matched by .gitignore rules. "
        "When False, include all files regardless of .gitignore settings.",
    )
    include_ignored: bool = Field(
        default=False,
        description="[Deprecated] Use respect_gitignore=False instead.",
    )
    verbose: bool = Field(
        default=False,
        description="When True, include file size, modification time, and type for each match.",
    )
    timeout: int = Field(
        description="Maximum time in seconds to wait for the search to complete.",
        default=10,
        ge=1,
    )


class Glob(CallableTool2[Params]):
    name: str = "Glob"
    description: str = _description_for_os("")
    params: type[Params] = Params
    def __init__(self, runtime: Runtime, vfs: VFS | None = None) -> None:
        super().__init__(description=_description_for_os(runtime.environment.os_kind))
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._skills_dirs = runtime.skills_dirs
        self._vfs = vfs

    # async def _validate_directory(self, directory: KaosPath) -> ToolError | None:
    #     """Validate that the directory is safe to search."""
    #     resolved_dir = directory.canonical()

    #     # Allow directories within the workspace (work_dir or additional dirs)
    #     if is_within_workspace(resolved_dir, self._work_dir, self._additional_dirs):
    #         return None

    #     # Allow directories within any discovered skills root
    #     if any(is_within_directory(resolved_dir, d) for d in self._skills_dirs):
    #         return None

    #     return ToolError(
    #         message=(
    #             f"`{directory}` is outside the workspace. "
    #             "You can only search within the working directory, "
    #             "additional directories, and skills directories."
    #         ),
    #         brief="Directory outside workspace",
    #     )

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            pattern = params.pattern

            # Reject patterns that would recursively match everything under root
            if _is_unsafe_recursive_pattern(pattern):
                return ToolError(
                    message=(
                        f"Unsafe pattern `{pattern}` — this would recursively "
                        "match all files/dirs under the search root, which is "
                        "meaningless and can be extremely slow. "
                        "Use a more specific pattern (e.g. `src/**/*.py`)."
                    ),
                    brief=f"Unsafe pattern: {pattern}",
                )

            if params.directory:
                dir_path = kaos_path_from_tool_input(params.directory, self._work_dir)
            else:
                dir_path = self._work_dir
            dir_path = await resolve_vfs(str(dir_path), self._vfs, for_write=False, work_dir=self._work_dir)
            if not await dir_path.exists():
                display_dir = str(dir_path)
                return ToolError(
                    message=f"`{display_dir}` does not exist.",
                    brief=f"Directory not found: {display_dir}",
                )
            if not await dir_path.is_dir():
                display_dir = str(dir_path)
                return ToolError(
                    message=f"`{display_dir}` is not a directory.",
                    brief=f"Invalid directory: {display_dir}",
                )

            # Load gitignore rules if needed (sync I/O in executor) and perform
            # the glob search under a single timeout budget.
            gitignore_rules: list[_GitignoreRule] = []
            matches: list[KaosPath] = []
            truncated = False
            timed_out = False

            # Handle deprecated include_ignored -> respect_gitignore inversion
            respect_gitignore = params.respect_gitignore
            if params.include_ignored:
                respect_gitignore = False

            try:
                async with asyncio.timeout(params.timeout):
                    if respect_gitignore:
                        try:
                            resolved_dir = Path(str(dir_path)).resolve()
                            gitignore_rules = await asyncio.to_thread(
                                _get_gitignore_rules, resolved_dir
                            )
                        except Exception:
                            pass

                    async for match in dir_path.glob(pattern):
                        if not params.include_dirs and not await match.is_file():
                            continue
                        # Apply gitignore filtering
                        if gitignore_rules:
                            try:
                                match_resolved = Path(str(match)).resolve()
                                if _is_ignored_by_gitignore(
                                    match_resolved, gitignore_rules, resolved_dir
                                ):
                                    continue
                            except Exception:
                                pass
                        matches.append(match)
                        if len(matches) > MAX_MATCHES:
                            truncated = True
                            matches.pop()
                            break
            except TimeoutError:
                timed_out = True

            # Sort for consistent output
            matches.sort()

            # Build output with byte limit
            output_lines: list[str] = []
            n_bytes = 0
            truncated_by_bytes = False
            for p in matches:
                relative_path = str(p.relative_to(dir_path))
                if params.verbose:
                    # Include metadata
                    try:
                        stat_result = await p.stat()
                        size = stat_result.st_size
                        from pendulum import from_timestamp
                        mtime = from_timestamp(stat_result.st_mtime).to_datetime_string()
                        kind = "dir" if await p.is_dir() else "file"
                        line = f"{relative_path}  ({size} bytes, {kind}, {mtime})"
                    except Exception:
                        line = relative_path
                else:
                    line = relative_path
                line_bytes = len(line.encode("utf-8"))
                separator_bytes = 1 if output_lines else 0
                output_lines.append(line)
                n_bytes += separator_bytes + line_bytes
                if n_bytes >= MAX_BYTES:
                    truncated_by_bytes = True
                    break

            output = "\n".join(output_lines)

            # Build message
            shown_count = len(output_lines)
            if shown_count > 0:
                message = f"Found {shown_count} matches for pattern `{pattern}`."
            else:
                message = f"No matches found for pattern `{pattern}`."

            if truncated:
                message += (
                    f" Showing first {MAX_MATCHES} matches. "
                    "Use a more specific pattern."
                )

            if timed_out:
                message += (
                    f" Search timed out after {params.timeout}s; "
                    "showing matches collected so far."
                )

            if truncated_by_bytes:
                message += f" Output truncated to {MAX_BYTES} bytes."

            display_dir = str(dir_path)
            return ToolOk(
                output=output,
                message=message,
                brief=f"Glob {display_dir}",
            )

        except Exception as e:
            logger.warning(
                "Glob failed: pattern={pattern}: {error}", pattern=params.pattern, error=e
            )
            return ToolError(
                message=f"Glob failed for `{params.pattern}`: {e}",
                brief=f"Glob failed: {params.pattern}",
            )
