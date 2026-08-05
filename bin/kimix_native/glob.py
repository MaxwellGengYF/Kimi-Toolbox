"""kimix_native.glob -- glob kernels: gitignore parsing/matching, path filtering.

Native implementations live in ``runtime_py.glob`` (compiled kernels, GIL
released). The pure-Python ``_compat`` functions below mirror the reference
algorithms exactly:

- ``_parse_gitignore`` / ``_gitignore_match`` -- kimi_cli/tools/file/glob.py
- ``is_ignored`` / ``_parse_ls_files_output`` -- kimi_cli/utils/file_filter.py
"""

from __future__ import annotations

import fnmatch
import re

from . import _native, use_native

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#_compat-- exact mirrors of the Python reference implementations
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

_IGNORED_NAMES: frozenset[str] = frozenset(
    (
#vcs metadata
        ".DS_Store",
        ".bzr",
        ".git",
        ".hg",
        ".svn",
#tooling caches
        ".build",
        ".cache",
        ".coverage",
        ".fleet",
        ".gradle",
        ".idea",
        ".ipynb_checkpoints",
        ".pnpm-store",
        ".pytest_cache",
        ".pub-cache",
        ".ruff_cache",
        ".swiftpm",
        ".tox",
        ".venv",
        ".vs",
        ".vscode",
        ".yarn",
        ".yarn-cache",
#js / frontend
        ".next",
        ".nuxt",
        ".parcel-cache",
        ".svelte-kit",
        ".turbo",
        ".vercel",
        "node_modules",
#python packaging
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "htmlcov",
        "pip-wheel-metadata",
        "venv",
#java / jvm
        ".mvn",
        "out",
        "target",
#dotnet / native
        "bin",
        "cmake-build-debug",
        "cmake-build-release",
        "obj",
#bazel / buck
        "bazel-bin",
        "bazel-out",
        "bazel-testlogs",
        "buck-out",
#misc artifacts
        ".dart_tool",
        ".serverless",
        ".stack-work",
        ".terraform",
        ".terragrunt-cache",
        "DerivedData",
        "Pods",
        "deps",
        "tmp",
        "vendor",
    )
)

_IGNORED_PATTERNS: re.Pattern[str] = re.compile(
    r"|".join(
        (
            r".*_cache$",
            r".*-cache$",
            r".*\.egg-info$",
            r".*\.dist-info$",
            r".*\.py[co]$",
            r".*\.class$",
            r".*\.sw[po]$",
            r".*~$",
            r".*\.(?:tmp|bak)$",
        )
    ),
    re.IGNORECASE,
)


def _compat_parse_gitignore(content: bytes, source_dir: str) -> list[tuple[str, bool, bool, bool]]:
    """Parse a .gitignore file into a list of rules."""
    text = content.decode("utf-8", errors="replace")
    rules: list[tuple[str, bool, bool, bool]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        if not line:
            continue
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        anchored = "/" in line
        if line.startswith("/"):
            line = line[1:]
            anchored = True
        rules.append((line, negated, anchored, dir_only))
    return rules


def _compat_gitignore_match(rel_path: str, is_dir: bool, rule: tuple[str, bool, bool, bool]) -> bool:
    """Check if a relative path matches a single gitignore rule."""
    rel_path = rel_path.replace("\\", "/")
    pattern, negated, anchored, dir_only = rule

    if dir_only and not is_dir:
        parts = rel_path.split("/")
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if _compat_gitignore_match(prefix, True, rule):
                return True
        return False

    if "**" in pattern:
        parts = rel_path.split("/")

        if pattern == "**":
            return True
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if fnmatch.fnmatchcase(sub, suffix) or fnmatch.fnmatchcase(parts[-1], suffix):
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
                rest_parts = rest.split("/")
                for i in range(len(rest_parts)):
                    sub = "/".join(rest_parts[i:])
                    if fnmatch.fnmatchcase(sub, suffix) or fnmatch.fnmatchcase(rest_parts[-1], suffix):
                        return True
            return False

        simple_pattern = pattern.replace("**", "*")
        return fnmatch.fnmatchcase(rel_path, simple_pattern) or fnmatch.fnmatchcase(
            rel_path.split("/")[-1], simple_pattern
        )

    if anchored:
        return fnmatch.fnmatchcase(rel_path, pattern)
    else:
        basename = rel_path.split("/")[-1]
        if fnmatch.fnmatchcase(basename, pattern):
            return True
        return any(fnmatch.fnmatchcase(part, pattern) for part in rel_path.split("/")[:-1])


def _compat_is_ignored(rel_path: str, is_dir: bool, rules: list[tuple[str, bool, bool, bool]]) -> bool:
    """Check if a path is ignored by any gitignore rule (with negation support)."""
    ignored = False
    for rule in rules:
        if _compat_gitignore_match(rel_path, is_dir, rule):
            ignored = not rule[1]
    return ignored


def _compat_filter_paths(
    paths: list[str],
    is_dir_mask: list[bool],
    rules: list[tuple[str, bool, bool, bool]],
) -> list[bool]:
    return [_compat_is_ignored(p, d, rules) for p, d in zip(paths, is_dir_mask)]


def _compat_is_ignored_name(name: str) -> bool:
    if not name:
        return True
    if name in _IGNORED_NAMES:
        return True
    return bool(_IGNORED_PATTERNS.fullmatch(name))


def _compat_parse_ls_files_output(stdout: bytes, filter_ignored: bool = True) -> list[str]:
    """Parse NUL-delimited git ls-files -z output into paths with synthesised dirs."""
    text = stdout.decode("utf-8", errors="replace")
    paths: list[str] = []
    seen_dirs: set[str] = set()
    ignored_prefixes: set[str] = set()
    for entry in text.split("\0"):
        if not entry:
            continue

        parts = entry.split("/")

        if filter_ignored:
            skip = False
            for i, part in enumerate(parts):
                prefix = "/".join(parts[: i + 1]) + "/"
                if prefix in ignored_prefixes:
                    skip = True
                    break
                if _compat_is_ignored_name(part):
                    ignored_prefixes.add(prefix)
                    skip = True
                    break
            if skip:
                continue

        for i in range(1, len(parts)):
            dir_path = "/".join(parts[:i]) + "/"
            if dir_path not in seen_dirs:
                seen_dirs.add(dir_path)
                paths.append(dir_path)
        paths.append(entry)
    return paths

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Public API(native with _compat fallback)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -


def parse_gitignore(content: bytes, source_dir: str) -> list[tuple[str, bool, bool, bool]]:
    if not use_native("GLOB") or _native is None:
        return _compat_parse_gitignore(content, source_dir)
    return _native.glob.parse_gitignore(content, source_dir)


def is_ignored(rel_path: str, is_dir: bool, rules: list[tuple[str, bool, bool, bool]]) -> bool:
    if not use_native("GLOB") or _native is None:
        return _compat_is_ignored(rel_path, is_dir, rules)
    return _native.glob.is_ignored(rel_path, is_dir, rules)


def filter_paths(
    paths: list[str],
    is_dir_mask: list[bool],
    rules: list[tuple[str, bool, bool, bool]],
) -> list[bool]:
    if not use_native("GLOB") or _native is None:
        return _compat_filter_paths(paths, is_dir_mask, rules)
    return _native.glob.filter_paths(paths, is_dir_mask, rules)


def is_ignored_name(name: str) -> bool:
    if not use_native("GLOB") or _native is None:
        return _compat_is_ignored_name(name)
    return _native.glob.is_ignored_name(name)


def parse_ls_files_output(stdout: bytes, filter_ignored: bool = True) -> list[str]:
    if not use_native("GLOB") or _native is None:
        return _compat_parse_ls_files_output(stdout, filter_ignored)
    return _native.glob.parse_ls_files_output(stdout, filter_ignored)
