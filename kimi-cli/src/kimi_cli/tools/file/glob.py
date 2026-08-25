"""Glob tool implementation."""

import asyncio
import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue, alias_note
from pydantic import AliasChoices, BaseModel, Field

from kimi_cli.native_loader import (
    get_module as _native_get_module,
    use_native as _native_use_native,
)
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.file.micro_compress import (
    MicroCompressConfig,
    compress_lines as _mc_compress_lines,
)
from kimi_cli.tools.file.output_utils import fold_lines, truncate_line
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.logging import logger
from kimi_cli.utils.path import kaos_path_from_tool_input
from kimi_cli.vfs import VFS

from .utils import resolve_vfs

# Resolved once at import time (stable runtime: result never changes).
_NATIVE_GLOB = _native_get_module("glob")

MAX_MATCHES = 1000
MAX_BYTES = 100 << 10  # 100KB
GLOB_DESC_PATH = Path(__file__).parent / "glob.md"

# The native glob matcher uses fnmatchcase (case-sensitive).  The Python
# fallback uses fnmatch, which is case-insensitive on Windows.  To keep
# bit-identical outputs we only delegate matching to native on platforms where
# the two agree; native parsing is still used everywhere.
_NATIVE_GLOB_MATCH_CASE_SENSITIVE = not sys.platform.startswith("win")


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
    "Windows: `path` accepts native (`C:\\Users\\foo`) and POSIX-style "
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
    _native: tuple[str, bool, bool, bool] | None = field(
        default=None, repr=False, compare=False
    )


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
    # Native fast path: kimix_native.glob.parse_gitignore (bit-identical).
    if _native_use_native("GLOB") and _NATIVE_GLOB is not None:
        try:
            native_rules = _NATIVE_GLOB.parse_gitignore(
                content.encode("utf-8"), str(source_dir)
            )
            return [
                _GitignoreRule(
                    pattern=pattern,
                    negated=negated,
                    anchored=anchored,
                    is_dir_only=is_dir_only,
                    source_dir=source_dir,
                    _native=(pattern, negated, anchored, is_dir_only),
                )
                for pattern, negated, anchored, is_dir_only in native_rules
            ]
        except Exception:
                pass

    # Pure-Python fallback.
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
    """Check if a path matches a single gitignore rule.

    A dir-only rule (trailing ``/``, e.g. ``.venv/``) matches the directory
    itself **and every descendant**: when *is_dir* is False the rule pattern
    is matched against every ancestor prefix of *rel_path* (``.venv/``
    excludes ``.venv/a.py`` as well as ``a/b/.venv/c/d.py``).  The same
    applies to ``!``-negated dir-only rules, which un-ignore the directory
    and its contents.
    """
    # Normalize to forward slashes so prefix matching works on Windows too.
    rel_path = rel_path.replace("\\", "/")

    if rule.is_dir_only and not is_dir:
        # Check every ancestor directory prefix of the file: a dir-only rule
        # excludes descendants, not just the directory entry itself.
        parts = rel_path.split("/")
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if _gitignore_match(path, prefix, True, rule):
                return True
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
    # Native fast path: when all rules come from a single .gitignore file we
    # can evaluate the whole list in one native call.  Multi-source-dir cases
    # keep the Python loop because the native is_ignored() API consumes one
    # rel_path and one flat rules list.
    if (
        rules
        and _native_use_native("GLOB")
        and _NATIVE_GLOB is not None
        and _NATIVE_GLOB_MATCH_CASE_SENSITIVE
    ):
        source_dir = rules[0].source_dir
        if all(rule.source_dir == source_dir for rule in rules):
            try:
                rel_path = str(path.relative_to(source_dir)).replace("\\", "/")
                is_dir = path.is_dir()
                native_rules = [
                    rule._native
                    if rule._native is not None
                    else (
                        rule.pattern,
                        rule.negated,
                        rule.anchored,
                        rule.is_dir_only,
                    )
                    for rule in rules
                ]
                return _NATIVE_GLOB.is_ignored(rel_path, is_dir, native_rules)
            except Exception:
                pass

    # Standard gitignore behavior: later rules override earlier ones.
    # We process all rules in order of discovery (root-first, then deeper).
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


def _top_dirs_summary(
    matches: list[KaosPath], dir_path: KaosPath, top: int = 3
) -> str:
    """Summarize the match set by top-level directory (no extra I/O).

    Helps the agent avoid re-globbing junk directories when results were
    folded, e.g. ``top dirs: .venv (900), src (40), tests (27)``.  Files at
    the search root itself are not counted (they have no top-level dir).
    """
    counts: dict[str, int] = {}
    for p in matches:
        try:
            rel = str(p.relative_to(dir_path)).replace("\\", "/")
        except ValueError:
            continue
        parts = rel.split("/")
        if len(parts) > 1 and parts[0]:
            counts[parts[0]] = counts.get(parts[0], 0) + 1
    top_entries = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    if not top_entries:
        return ""
    return "top dirs: " + ", ".join(f"{name} ({n})" for name, n in top_entries)


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


def invalidate_gitignore_cache(path: str) -> None:
    """Drop cached gitignore rule sets affected by a change at *path*.

    Called by ``fs_cache`` after a ``.gitignore`` is written/created/deleted.
    Every cache entry whose search root contains *path* (i.e. the root is an
    ancestor of the file's directory, so its walk would have discovered the
    changed ``.gitignore``) is dropped; the mtime self-refresh inside
    ``_get_gitignore_rules`` remains the backstop.
    """
    try:
        target = Path(path)
    except (TypeError, ValueError):
        return
    try:
        resolved = target.resolve(strict=False)
        candidates = {str(resolved.parent), str(resolved)}
    except (OSError, ValueError):
        candidates = {str(target)}
    stale_roots = []
    for root_str in _GITIGNORE_CACHE:
        for cand in candidates:
            try:
                if Path(cand).is_relative_to(root_str) or root_str == cand:
                    stale_roots.append(root_str)
                    break
            except (TypeError, ValueError):
                continue
    for root_str in stale_roots:
        _GITIGNORE_CACHE.pop(root_str, None)


class Params(BaseModel):
    model_config = {"populate_by_name": True}

    pattern: str = Field(
        description=(
            "Glob pattern to match file paths against (e.g. `**/*.ts`, "
            "`src/**/*.test.js`). A pattern with no \"/\" matches the basename "
            "at any depth, so `*` and `*.ts` both search the whole tree; include "
            "a separator to anchor the depth. Unsafe recursive patterns "
            "(``**``, ``**/*``, ``**/**``, etc.) are forbidden."
        )
    )
    path: str | None = Field(
        validation_alias=AliasChoices("path", "directory"),
        description=(
            "Directory to search in. Defaults to the session workspace; a "
            "relative path resolves against it. "
            + alias_note("path", "directory", word=False)
        ),
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
    max_results: int = Field(
        default=500,
        alias="fold",
        ge=0,
        description=(
            "Maximum number of result lines in the output. Longer results "
            "are head+tail folded with an omitted-count marker and the total "
            "is reported in `message`. 0 = unlimited (the MAX_MATCHES "
            "collection cap still applies)."
        ),
    )


class Glob(CallableTool2[Params]):
    name: str = "glob"
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
            if params.path:
                dir_path = kaos_path_from_tool_input(params.path, self._work_dir)
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
            # Matches suppressed by .gitignore — lets the tool explain WHY a
            # search came back empty instead of silently reporting "no matches"
            # when the paths exist but are hidden by an ignore rule.
            ignored_count = 0

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
                                match_path = Path(str(match))
                                match_resolved = match_path.resolve()
                                if _is_ignored_by_gitignore(
                                    match_resolved, gitignore_rules, resolved_dir
                                ):
                                    ignored_count += 1
                                    continue
                                # resolve() dereferences junctions/symlinks
                                # (e.g. uv's .venv pointing into a cache):
                                # the resolved target may escape the search
                                # root so relative_to() fails and the rule
                                # silently doesn't apply. Retry against the
                                # walked path, which is the one the user
                                # sees under the search root.
                                if match_resolved != match_path and _is_ignored_by_gitignore(
                                    match_path, gitignore_rules, resolved_dir
                                ):
                                    ignored_count += 1
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
            total = len(matches)

            # Build output lines (relative paths, optional verbose metadata).
            # Per-line cap keeps a single huge verbose line from hogging the
            # byte budget.
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
                        line = truncate_line(
                            f"{relative_path}  ({size} bytes, {kind}, {mtime})"
                        )
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

            # Apply the final display fold budget (head+tail with marker).
            # 0 = unlimited → MAX_MATCHES collection cap remains the limit.
            omitted_by_fold = 0
            if params.max_results:
                output_lines, omitted_by_fold = fold_lines(
                    output_lines, params.max_results
                )

            # Micro-compress — lossless stages (1-3, 5) plus the annotated
            # prefix fold (Stage 4) which factors out a long shared directory
            # prefix on long result lists.  Near-duplicate collapse (Stage 8)
            # is disabled: every distinct path must stay visible.
            if output_lines:
                output_lines, _ = _mc_compress_lines(
                    output_lines,
                    kind="log",
                    config=MicroCompressConfig(near_dup_collapse=False),
                )

            output = "\n".join(output_lines)

            # Build message
            shown_count = len(output_lines) - (1 if omitted_by_fold else 0)
            if total > 0:
                message = f"Found {total} matches for pattern `{pattern}`."
            else:
                message = f"No matches found for pattern `{pattern}`."
                if respect_gitignore and ignored_count > 0 and not timed_out:
                    message += (
                        f" {ignored_count} path(s) matched but were excluded by"
                        " .gitignore — pass respect_gitignore=False to include them."
                    )

            if omitted_by_fold:
                message += (
                    f" Showing {shown_count} of {total} (head+tail fold). "
                    "Use max_results=0 or a more specific pattern to see more."
                )
                top_dirs = _top_dirs_summary(matches, dir_path)
                if top_dirs:
                    message += f" {top_dirs}"

            if truncated:
                message += f" Search capped at {MAX_MATCHES} matches."

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
