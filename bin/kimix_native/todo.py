"""kimix_native.todo — todo kernels: merge, status counts, plain-text summary.

Native implementations live in ``runtime_py.todo`` (compiled kernels, GIL
released). The pure-Python functions below mirror the reference algorithms
exactly so ``KIMIX_NATIVE_TODO=0`` yields identical output.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from . import _native, use_native


def _canonical_status(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"Invalid status '{v}'. Must be one of: pending, in_progress, done.")
    normalized = v.strip().lower().replace("-", "_")
    if normalized not in {"pending", "in_progress", "done"}:
        raise ValueError(f"Invalid status '{v}'. Must be one of: pending, in_progress, done.")
    return normalized


def _normalize_title(v: Any) -> str:
    if not isinstance(v, str):
        v = str(v)
    stripped = v.strip()
    if not stripped:
        raise ValueError("Title cannot be empty or contain only whitespace.")
    return stripped


def _normalize_notes(v: Any) -> str | None:
    if v is None:
        return None
    stripped = str(v).strip()
    return stripped if stripped else None


def _normalize_code(v: Any) -> str | None:
    return None if v is None else str(v)


def _todo_item(v: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a raw todo dict."""
    return {
        "title": _normalize_title(v.get("title")),
        "status": _canonical_status(v.get("status")),
        "notes": _normalize_notes(v.get("notes")),
        "code": _normalize_code(v.get("code")),
    }


def _is_unfinished(item: dict[str, Any]) -> bool:
    return item["status"] != "done"


def _format_title_list(titles: list[str]) -> str:
    return "[" + ", ".join(f'"{t}"' for t in titles) + "]"


def _merge_one(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Produce an updated todo preserving old notes/code when new omits them."""
    return {
        "title": old["title"],
        "status": new["status"],
        "notes": new["notes"] if new["notes"] is not None else old["notes"],
        "code": new["code"] if new["code"] is not None else old["code"],
    }


def _merge_by_title_update(
    old_items: list[dict[str, Any]], new_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    new_by_title = {t["title"]: t for t in new_items}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for old in old_items:
        new = new_by_title.get(old["title"])
        if new is not None:
            merged.append(_merge_one(old, new))
        else:
            merged.append(old)
        seen.add(old["title"])

    for new in new_items:
        if new["title"] not in seen:
            merged.append(new)
            seen.add(new["title"])

    return merged


def _shell_kind() -> str:
    """Return the label used for !-prefixed code (matches the native default)."""
    return "pwsh" if sys.platform == "win32" else "bash"


def _format_kind(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("!"):
        return _shell_kind()
    lowered = stripped.lower()
    if lowered.endswith((".py", ".sh", ".ps1")) and os.path.isfile(stripped):
        return "file"
    return "inline"


def merge(
    old_items: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    mode: str,
    fuzzy_warnings: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Merge/update todo lists.

    Returns a dict with ``items``, ``warnings``, ``error``, ``regressed``,
    and ``archived`` keys. This mirrors the runtime_py.todo kernel exactly.
    """
    if fuzzy_warnings is None:
        fuzzy_warnings = {}

    if use_native("TODO") and _native is not None:
        return _native.todo.merge(old_items, new_items, mode, fuzzy_warnings)

    result: dict[str, Any] = {
        "items": [],
        "warnings": [],
        "error": None,
        "regressed": [],
        "archived": [],
    }

    # Canonicalize inputs (mirror the native binding's indexed error messages).
    old: list[dict[str, Any]] = []
    for i, x in enumerate(old_items):
        try:
            old.append(_todo_item(x))
        except ValueError as exc:
            result["error"] = f"Invalid todo at index {i}: {exc}"
            return result
    new = []
    for i, x in enumerate(new_items):
        try:
            new.append(_todo_item(x))
        except ValueError as exc:
            result["error"] = f"Invalid todo at index {i}: {exc}"
            return result

    # Mode validation.
    mode_norm = mode.strip().lower().replace("-", "_")
    is_append = mode_norm == "append"
    is_overwrite = mode_norm == "overwrite"
    is_force = mode_norm == "force_overwrite"
    if not (is_append or is_overwrite or is_force):
        result["error"] = (
            f"Invalid mode '{mode}'. Must be one of: append, overwrite, force_overwrite."
        )
        return result

    # Duplicate titles in incoming list.
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for item in new:
        seen[item["title"]] = seen.get(item["title"], 0) + 1
        if seen[item["title"]] == 2:
            duplicates.append(item["title"])
    if duplicates:
        duplicates.sort()
        result["error"] = f"Duplicate todo titles found: {_format_title_list(duplicates)}"
        return result

    # Branch on mode.
    if is_force:
        final = list(new)
    elif is_overwrite:
        if any(_is_unfinished(x) for x in old):
            unfinished = "\n".join(x["title"] for x in old if _is_unfinished(x))
            result["error"] = (
                "Error: Cannot overwrite todos while old todos are not all done. "
                "Use mode='force_overwrite' if you really want to discard unfinished work.\n"
                f"Unfinished:\n{unfinished}"
            )
            return result
        final = list(new)
    else:  # append
        if not new:
            if any(_is_unfinished(x) for x in old):
                unfinished = ", ".join(x["title"] for x in old if _is_unfinished(x))
                result["error"] = (
                    "Error: Cannot clear todos while old todos are not all done. "
                    f"Unfinished: {unfinished}\n"
                    "Next step: mark them done first, "
                    "or use mode='force_overwrite' to discard them intentionally."
                )
                return result
            final = []
        else:
            final = _merge_by_title_update(old, new)

    # Fuzzy warnings in new-item order.
    for item in new:
        result["warnings"].extend(fuzzy_warnings.get(item["title"], []))

    # Regression detection (unless force_overwrite).
    if not is_force and old:
        old_status_map = {x["title"]: x["status"] for x in old}
        clamped: list[dict[str, Any]] = []
        for item in final:
            if old_status_map.get(item["title"]) == "done" and item["status"] != "done":
                item = dict(item)
                item["status"] = "done"
                result["regressed"].append(item["title"])
            clamped.append(item)
        final = clamped

    # Archive completed todos that are no longer kept.
    kept_titles = {x["title"] for x in final}
    result["archived"] = [
        dict(x) for x in old if x["status"] == "done" and x["title"] not in kept_titles
    ]

    # Enforce single in_progress (auto-fix).
    if not is_force:
        fixed: list[dict[str, Any]] = []
        seen_in_progress = False
        for item in final:
            if item["status"] == "in_progress":
                if seen_in_progress:
                    item = dict(item)
                    item["status"] = "done"
                    result["warnings"].append(
                        f'Auto-fixed "{item["title"]}": set to done (only one item may be in_progress)'
                    )
                else:
                    seen_in_progress = True
            fixed.append(item)
        final = fixed

    result["items"] = final
    return result


def status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count todos by canonical status."""
    if use_native("TODO") and _native is not None:
        return _native.todo.status_counts(items)

    counts: dict[str, int] = {"pending": 0, "in_progress": 0, "done": 0}
    for item in items:
        try:
            status = _canonical_status(item.get("status"))
        except ValueError:
            continue
        if status in counts:
            counts[status] += 1
    return counts


def format_summary(items: list[dict[str, Any]], max_items: int = 50) -> str:
    """Return the plain-text todo summary rendered by the reference tool."""
    if use_native("TODO") and _native is not None:
        return _native.todo.format_summary(items, max_items)

    try:
        todos = [_todo_item(x) for x in items]
    except ValueError:
        return ""

    selected = [t for t in todos if t["status"] in ("pending", "in_progress")]
    if not selected:
        return ""
    selected = selected[:max_items]

    display_status = {
        "pending": "pending",
        "in_progress": "in progress",
        "done": "done",
    }
    lines: list[str] = []
    for t in selected:
        line = f"- [{display_status[t['status']]}] {t['title']}"
        if t["code"]:
            line += f"  `[code: {_format_kind(t['code'])}]`"
        if t["status"] == "in_progress" and t["notes"]:
            line += f"  Notes: {t['notes']}"
        lines.append(line)
    return "\n".join(lines)
