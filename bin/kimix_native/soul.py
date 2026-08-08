"""kimix_native.soul -- soul-domain kernels (plans 014/015/016).

Native implementations live in ``runtime_py.soul`` (compiled kernels, GIL
released). The pure-Python ``_compat`` mirrors below replicate the reference
algorithms from the kimi-agent repo (chat_provider/kimi.py _convert_message +
_normalize_tool_call_ids, soul/context_pruning.py candidate selection,
soul/message.py is_system_reminder_message, soul/dynamic_injection.py
normalize_history, soul/compaction.py SimpleCompaction.prepare), so
``use_native("SOUL") is False`` yields bit-identical behavior.

MESSAGE-VIEW BRIDGE CONTRACT (shared with the C++ bindings):

    history : list[dict]  -- pydantic-like messages
        {"role": "user"|"system"|"assistant"|"tool",
         "content": [{"type": "text"|"think"|"image_url"|..., ...}, ...],
         "tool_calls": [{"type":"function","id":...,"function":{...}}] | None,
         "tool_call_id": str | None}

    The shim encodes every string into ONE UTF-8 bytes buffer and passes
    parallel offset arrays to the native kernel (no pydantic crossing, no
    deepcopy). Content-part kinds: 0=text 1=think 2=tool_call 3=image
    4=audio 5=file 6=other. Non-text/think parts carry their serialized JSON
    object in the buffer. tool_calls tuples are
    (id_s, id_e, name_s, name_e, args_s, args_e) with args_s == -1 meaning
    function.arguments is None.

Public API:
    - build_payload(history, preserved_thinking=False) -> bytes
    - normalize_tool_call_ids(history) -> list[(msg_index, call_index, new_id)]
    - prune_scan(history, policy=None) -> list[(index, reason)]
    - count_leading_reminders(history) -> int
    - build_normalize_plan(history) -> list[(index, op, target_index)]
    - build_compaction_prompt(history, system_prompt="") -> bytes
    - apply_normalize_plan(history, plan) -> list[dict]   (test helper)
    - apply_id_fixes(history, fixes) -> list[dict]        (test helper)
"""

from __future__ import annotations

import json

import orjson

from . import _native, use_native
from ._prompt_texts import (
    BALANCED_MODE_GUIDANCE,
    COMPACT_CASCADE_PROMPT_TEXT,
    COMPACT_PROMPT_TEXT,
    DECISION_SECTION_GUIDANCE,
)

_USE = use_native("SOUL") and _native is not None

# ---------------------------------------------------------------------------
# Encoding helpers (bytes-in/bytes-out contract like the rest of the module)
# ---------------------------------------------------------------------------


def _enc(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def _dec(b: bytes) -> str:
    return b.decode("utf-8", "surrogatepass")


def _compact(obj) -> bytes:
    """orjson-fast compact JSON bytes (no spaces, raw UTF-8).

    Falls back to the stdlib serializer for values orjson rejects (lone
    surrogates, non-str keys, >64-bit ints) so the wire bytes are preserved.
    """
    try:
        return orjson.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8", "surrogatepass"
        )


def _loads(data: bytes):
    """orjson-fast parse; stdlib fallback keeps lone-surrogate parity."""
    try:
        return orjson.loads(data)
    except ValueError:
        return json.loads(_dec(data))


_ROLE_TO_INT = {"system": 0, "user": 1, "assistant": 2, "tool": 3}
_INT_TO_ROLE = {0: "system", 1: "user", 2: "assistant", 3: "tool"}

# content-part "type" -> part_kind (0 text 1 think 2 tool_call 3 image
# 4 audio 5 file 6 other)
_PART_TYPE_KIND = {
    "text": 0,
    "think": 1,
    "tool_call": 2,
    "image_url": 3,
    "audio_url": 4,
    "video_url": 5,
    "file": 5,
}


def _part_kind(part: dict) -> int:
    return _PART_TYPE_KIND.get(part.get("type", ""), 6)


def _part_dump(part: dict) -> bytes:
    """exclude_none part dump (what the payload builder embeds verbatim)."""
    return _compact({k: v for k, v in part.items() if v is not None})


def _encode_message(msg: dict, buf: bytearray, spans: dict) -> None:
    """Append the message's strings to the shared buffer, recording spans."""
    role = _ROLE_TO_INT.get(msg.get("role"), 1)
    spans["roles"].append(role)

    # tool_call_id
    tcid = msg.get("tool_call_id")
    if tcid is None:
        spans["tool_call_ids"].append(None)
    else:
        data = _enc(tcid)
        start = len(buf)
        buf += data
        spans["tool_call_ids"].append((start, start + len(data)))

    # tool_calls
    tcs = msg.get("tool_calls")
    if tcs is None:
        spans["tool_calls"].append(None)
    else:
        call_spans = []
        for tc in tcs:
            fn = tc.get("function", {})
            tid = _enc(tc.get("id", ""))
            name = _enc(fn.get("name", ""))
            args = fn.get("arguments")
            id_s = len(buf)
            buf += tid
            id_e = len(buf)
            name_s = len(buf)
            buf += name
            name_e = len(buf)
            if args is None:
                args_s = -1
                args_e = 0
            else:
                adata = _enc(args)
                args_s = len(buf)
                buf += adata
                args_e = len(buf)
            call_spans.append((id_s, id_e, name_s, name_e, args_s, args_e))
        spans["tool_calls"].append(call_spans)

    # content parts
    part_spans = []
    for part in msg.get("content") or []:
        kind = _part_kind(part)
        if kind in (0, 1):
            text = _enc(part.get("text") or part.get("think") or "")
        else:
            text = _part_dump(part)
        start = len(buf)
        buf += text
        part_spans.append((start, start + len(text), kind))
    spans["parts"].append(part_spans)


def _build_structure(history: list[dict]) -> tuple[bytes, dict]:
    buf = bytearray()
    spans = {"roles": [], "parts": [], "tool_calls": [], "tool_call_ids": []}
    for msg in history:
        _encode_message(msg, buf, spans)
    return bytes(buf), spans


# ---------------------------------------------------------------------------
# _compat -- pure-Python mirrors of the reference algorithms
# ---------------------------------------------------------------------------


def _strip(s: str) -> str:
    return s.strip()


def _compat_extract_reasoning(content: list[dict]):
    reasoning = ""
    visible: list[dict] = []
    for part in content or []:
        if part.get("type") == "think":
            reasoning += part.get("think") or ""
        else:
            visible.append(part)
    return reasoning, visible


def _compat_effectively_empty(content: list[dict]) -> bool:
    for part in content:
        if part.get("type") != "text":
            return False
        if (part.get("text") or "").strip():
            return False
    return True


def _compat_convert_message(msg: dict, preserved_thinking: bool = False) -> dict:
    """Exact port of kosong chat_provider/kimi.py::_convert_message."""
    content = msg.get("content") or []
    has_reasoning_part = any(p.get("type") == "think" for p in content)
    reasoning, visible = _compat_extract_reasoning(content)

    dumped: dict = {}
    dumped["role"] = msg.get("role")
    if msg.get("name") is not None:
        dumped["name"] = msg["name"]
    if len(visible) == 1 and visible[0].get("type") == "text":
        dumped["content"] = visible[0].get("text", "")
    else:
        dumped["content"] = [
            {k: v for k, v in p.items() if v is not None} for p in visible
        ]
    if msg.get("tool_calls"):
        dumped["tool_calls"] = [
            {
                "type": "function",
                "id": tc.get("id"),
                "function": {
                    k: v
                    for k, v in (tc.get("function") or {}).items()
                    if v is not None
                },
            }
            for tc in msg["tool_calls"]
        ]
    if msg.get("tool_call_id") is not None:
        dumped["tool_call_id"] = msg["tool_call_id"]
    if msg.get("partial") is not None:
        dumped["partial"] = msg["partial"]

    if (
        dumped["role"] == "assistant"
        and dumped.get("tool_calls")
        and _compat_effectively_empty(visible)
    ):
        dumped.pop("content", None)

    if has_reasoning_part or (
        preserved_thinking and dumped["role"] == "assistant"
    ):
        dumped["reasoning_content"] = reasoning
    return dumped


_ID_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_ID_MAX = 64
_EMPTY_ID = "tool_call"


def _compat_sanitize_id(tool_call_id: str) -> str:
    return "".join(c if c in _ID_SAFE_CHARS else "_" for c in tool_call_id)[:_ID_MAX]


def _compat_make_unique(normalized: str, used: set[str]) -> str:
    base = normalized if normalized else _EMPTY_ID
    candidate = base[:_ID_MAX]
    if candidate not in used:
        return candidate
    index = 2
    while True:
        suffix = f"_{index}"
        candidate = base[: _ID_MAX - len(suffix)] + suffix
        if candidate not in used:
            return candidate
        index += 1


def _compat_normalize_tool_call_ids(history: list[dict]) -> list[tuple]:
    """Returns [(msg_index, call_index, new_id)]; call_index -1 = tool_call_id."""
    raw_ids: list[str] = []
    seen: set[str] = set()
    for message in history:
        for tc in message.get("tool_calls") or []:
            tid = tc.get("id", "")
            if tid not in seen:
                seen.add(tid)
                raw_ids.append(tid)
        tcid = message.get("tool_call_id")
        if tcid is not None and tcid not in seen:
            seen.add(tcid)
            raw_ids.append(tcid)
    if not raw_ids:
        return []

    mapped: dict[str, str] = {}
    used: set[str] = set()
    for raw_id in raw_ids:
        normalized = _compat_sanitize_id(raw_id)
        if normalized == raw_id and normalized:
            mapped[raw_id] = normalized
            used.add(normalized)
    for raw_id in raw_ids:
        if raw_id in mapped:
            continue
        unique = _compat_make_unique(_compat_sanitize_id(raw_id), used)
        mapped[raw_id] = unique
        used.add(unique)

    if all(mapped[raw_id] == raw_id for raw_id in raw_ids):
        return []

    fixes: list[tuple] = []
    for i, message in enumerate(history):
        for j, tc in enumerate(message.get("tool_calls") or []):
            new_id = mapped[tc.get("id", "")]
            if new_id != tc.get("id", ""):
                fixes.append((i, j, new_id))
        tcid = message.get("tool_call_id")
        if tcid is not None:
            new_id = mapped[tcid]
            if new_id != tcid:
                fixes.append((i, -1, new_id))
    return fixes


def _compat_is_system_reminder(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip().startswith("<system-reminder>")
    if not content or len(content) != 1:
        return False
    part = content[0]
    if not isinstance(part, dict):
        return False
    return part.get("type") == "text" and (
        (part.get("text") or "").strip().startswith("<system-reminder>")
    )


def _compat_is_notification(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.lstrip().startswith("<notification ")
    if not content or len(content) != 1:
        return False
    part = content[0]
    if not isinstance(part, dict):
        return False
    return part.get("type") == "text" and (
        (part.get("text") or "").lstrip().startswith("<notification ")
    )


def _compat_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in (content or [])
                   if isinstance(p, dict) and p.get("type") == "text")


def _compat_is_task_snapshot(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    text = _compat_text(msg)
    return "<active-background-tasks>" in text or "active background tasks" in text.lower()


def _compat_is_dmail(msg: dict) -> bool:
    return msg.get("role") == "user" and "D-Mail from your future self" in _compat_text(msg)


def _compat_is_checkpoint(msg: dict) -> bool:
    if msg.get("role") not in ("user", "system"):
        return False
    text = _compat_text(msg)
    if "CHECKPOINT" not in text:
        return False
    stripped = text.strip()
    return stripped.startswith("<system>CHECKPOINT") or "<system>CHECKPOINT" in stripped


def _compat_is_ephemeral(msg: dict, policy: dict) -> bool:
    if _compat_is_system_reminder(msg):
        return True
    if policy.get("drop_notifications", True) and _compat_is_notification(msg):
        return True
    if policy.get("drop_task_snapshots", True) and _compat_is_task_snapshot(msg):
        return True
    if policy.get("drop_dmail", True) and _compat_is_dmail(msg):
        return True
    if policy.get("drop_checkpoints", False) and _compat_is_checkpoint(msg):
        return True
    return False


def _compat_prune_scan(history: list[dict], policy: dict | None = None) -> list[tuple]:
    """context_pruning.py candidate selection: (index, reason) per message.

    reasons: 0=superseded_read 1=resolved_error 2=compact(Tier A drop)
             3=protect 4=oversized_output
    """
    policy = dict(policy or {})
    stable_prefix = policy.get("stable_prefix_messages", 4)
    recent_protected = policy.get("recent_messages_protected", 6)
    min_tokens = policy.get("tool_output_min_tokens", 512)
    current_turn = policy.get("current_turn_index")
    n = len(history)

    protected: set[int] = set()
    for i in range(min(stable_prefix, n)):
        protected.add(i)
    tail: list[int] = []
    for i in range(n - 1, -1, -1):
        if len(tail) >= recent_protected:
            break
        if history[i].get("role") in ("user", "assistant"):
            tail.append(i)
    for i in tail:
        protected.add(i)
    # tool-pair protection
    pair_ids: set[str] = set()
    for i in protected:
        msg = history[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pair_ids.update(tc.get("id", "") for tc in msg["tool_calls"])
    for j in range(n):
        if j in protected:
            continue
        if history[j].get("role") == "tool" and history[j].get("tool_call_id") in pair_ids:
            protected.add(j)
    if current_turn is not None:
        for i in range(current_turn, n):
            protected.add(i)

    texts = [_compat_text(m) for m in history]

    def is_superseded(i: int) -> bool:
        text = texts[i]
        if not text.strip():
            return False
        for j in range(i + 1, n):
            if history[j].get("role") == "tool":
                later = texts[j]
                if later and ("Tool output is empty" in later or len(later) < len(text) // 2):
                    return True
        return False

    def is_resolved(i: int) -> bool:
        text = texts[i]
        if "<system>ERROR:" not in text:
            return False
        for j in range(i + 1, n):
            if history[j].get("role") == "tool":
                later = texts[j]
                if later and "<system>ERROR:" not in later:
                    return True
        return False

    def is_oversized(i: int) -> bool:
        return max(len(texts[i]) // 4, 1) >= min_tokens

    snapshots = [
        i
        for i in range(n)
        if i not in protected
        and policy.get("drop_task_snapshots", True)
        and _compat_is_task_snapshot(history[i])
    ]
    latest_snapshot = max(snapshots) if snapshots else -1

    actions: list[tuple] = []
    for i in range(n):
        if i in protected:
            actions.append((i, 3))
            continue
        if _compat_is_ephemeral(history[i], policy):
            if _compat_is_task_snapshot(history[i]):
                if (
                    policy.get("drop_task_snapshots", True)
                    and len(snapshots) >= 2
                    and i != latest_snapshot
                ):
                    actions.append((i, 2))
            else:
                actions.append((i, 2))
            continue
        if history[i].get("role") != "tool":
            continue
        if is_superseded(i):
            actions.append((i, 0))
            continue
        if is_oversized(i):
            actions.append((i, 4))
            continue
        if is_resolved(i):
            actions.append((i, 1))
    return actions


def _compat_prune_history(messages: list[dict], policy: dict) -> dict:
    """Full prune_history pass (Plan 017): Tier A drops + Tier B elision stubs.

    Mirrors ContextPruner.prune but uses the explicit ``max_elision_tokens``
    budget from the policy instead of computing one from token counts.
    """
    policy = dict(policy or {})
    stable_prefix = policy.get("stable_prefix_messages", 4)
    recent_protected = policy.get("recent_messages_protected", 6)
    current_turn = policy.get("current_turn_index")
    max_tokens = policy.get("max_elision_tokens", 0)
    min_tokens = policy.get("tool_output_min_tokens", 512)
    superseded_enabled = policy.get("superseded_read_enabled", True)
    oversized_enabled = policy.get("oversized_output_enabled", True)
    stale_enabled = policy.get("stale_tool_result_enabled", True)
    n = len(messages)

    # ---- protected set -----------------------------------------------------
    protected: set[int] = set()
    for i in range(min(stable_prefix, n)):
        protected.add(i)
    tail: list[int] = []
    for i in range(n - 1, -1, -1):
        if len(tail) >= recent_protected:
            break
        if messages[i].get("role") in ("user", "assistant"):
            tail.append(i)
    for i in tail:
        protected.add(i)
    pair_ids: set[str] = set()
    for i in protected:
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pair_ids.update(tc.get("id", "") for tc in msg["tool_calls"])
    for j in range(n):
        if j in protected:
            continue
        if messages[j].get("role") == "tool" and messages[j].get("tool_call_id") in pair_ids:
            protected.add(j)
    if current_turn is not None:
        for i in range(current_turn, n):
            protected.add(i)

    # ---- per-message text --------------------------------------------------
    texts = [_compat_text(m) for m in messages]

    def _first_text(msg: dict) -> str:
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if not content:
            return ""
        first = content[0]
        if not isinstance(first, dict):
            return ""
        if first.get("type") == "text":
            return first.get("text") or ""
        if first.get("type") == "think":
            return first.get("think") or ""
        return ""

    first_texts = [_first_text(m) for m in messages]

    # ---- Tier B helpers ----------------------------------------------------
    def is_superseded(i: int) -> bool:
        text = texts[i]
        if not text.strip():
            return False
        for j in range(i + 1, n):
            if messages[j].get("role") == "tool":
                later = texts[j]
                if later and ("Tool output is empty" in later or len(later) < len(text) // 2):
                    return True
        return False

    def is_resolved(i: int) -> bool:
        text = texts[i]
        if "<system>ERROR:" not in text:
            return False
        for j in range(i + 1, n):
            if messages[j].get("role") == "tool":
                later = texts[j]
                if later and "<system>ERROR:" not in later:
                    return True
        return False

    def is_oversized(i: int) -> bool:
        return max(len(texts[i]) // 4, 1) >= min_tokens

    # ---- collect candidates ------------------------------------------------
    candidates: list[tuple[int, int, str, str]] = []  # (index, savings, tier, kind)

    # Tier A: ephemeral drops (task snapshots keep only the latest).
    snapshots = [
        i
        for i in range(n)
        if i not in protected
        and policy.get("drop_task_snapshots", True)
        and _compat_is_task_snapshot(messages[i])
    ]
    latest_snapshot = max(snapshots) if snapshots else -1

    for i in range(n):
        if i in protected:
            continue
        if _compat_is_ephemeral(messages[i], policy):
            savings = max(len(first_texts[i]) // 4, 1)
            if _compat_is_task_snapshot(messages[i]):
                if len(snapshots) >= 2 and i != latest_snapshot:
                    candidates.append((i, savings, "A", "ephemeral"))
            else:
                candidates.append((i, savings, "A", "ephemeral"))

    # Tier B: substantive elision (superseded -> oversized -> resolved).
    for i in range(n):
        if i in protected:
            continue
        if messages[i].get("role") != "tool":
            continue
        savings = max(len(texts[i]) // 4, 1)
        if superseded_enabled and is_superseded(i):
            candidates.append((i, savings, "B", "superseded_read"))
            continue
        if oversized_enabled and is_oversized(i):
            candidates.append((i, savings, "B", "oversized_output"))
            continue
        if stale_enabled and is_resolved(i):
            candidates.append((i, savings, "B", "resolved_error"))

    if not candidates:
        return {
            "messages": list(messages),
            "elided": [],
            "freed_tokens": 0,
            "earliest_removed_index": None,
        }

    # ---- greedy tail-inward selection --------------------------------------
    candidates.sort(key=lambda x: (-x[0], 0 if x[2] == "A" else 1, -x[1]))
    selected_indices: set[int] = set()
    total_freed = 0
    selected: list[tuple[int, int, str, str]] = []
    for idx, savings, tier, kind in candidates:
        if total_freed >= max_tokens:
            break
        if idx in selected_indices:
            continue
        selected_indices.add(idx)
        total_freed += savings
        selected.append((idx, savings, tier, kind))

    if not selected:
        return {
            "messages": list(messages),
            "elided": [],
            "freed_tokens": 0,
            "earliest_removed_index": None,
        }

    # ---- build result ------------------------------------------------------
    # Sort selected by index ascending so refs are assigned in index order.
    selected.sort(key=lambda x: x[0])
    selected_by_index = {idx: (savings, tier, kind) for idx, savings, tier, kind in selected}

    result_messages: list[dict] = []
    elided: list[dict] = []
    ref_counter = 0
    changes: set[int] = set()

    for i, msg in enumerate(messages):
        if i not in selected_indices:
            result_messages.append(msg)
            continue

        changes.add(i)
        _, tier, kind = selected_by_index[i]
        if tier == "A":
            continue

        ref = f"prune_{ref_counter}"
        ref_counter += 1
        savings, _, _ = selected_by_index[i]
        stub_text = (
            f"<system>[context-elided: {kind} — content elided. "
            f"~{savings} tokens freed. "
            f"Retrieve full content with Retrieve id={ref}]</system>"
        )
        elided.append({
            "index": i,
            "role": msg.get("role"),
            "kind": kind,
            "summary": f"{kind} at index {i}",
            "original_text": texts[i],
            "ref": ref,
        })
        new_msg = dict(msg)
        new_msg["content"] = [{"type": "text", "text": stub_text}]
        result_messages.append(new_msg)

    return {
        "messages": result_messages,
        "elided": elided,
        "freed_tokens": total_freed,
        "earliest_removed_index": min(changes) if changes else None,
    }


def _compat_count_leading_reminders(history: list[dict]) -> int:
    count = 0
    for msg in history:
        if not _compat_is_system_reminder(msg):
            break
        count += 1
    return count


def _compat_build_normalize_plan(history: list[dict]) -> list[tuple]:
    steps: list[tuple] = []
    last_role = None
    last_notification = False
    last_origin = 0
    for i, msg in enumerate(history):
        role = msg.get("role")
        is_notif = _compat_is_notification(msg)
        if (
            i > 0
            and last_role == role
            and role == "user"
            and not last_notification
            and not is_notif
        ):
            steps.append((i, 1, last_origin))
        else:
            steps.append((i, 0, i))
            last_role = role
            last_notification = is_notif
            last_origin = i
    return steps


def _compat_build_compaction_prompt(history: list[dict], system_prompt: str = "") -> bytes:
    out = ""
    if system_prompt:
        out += system_prompt + "\n\n"
    for i, msg in enumerate(history):
        out += f"## Message {i + 1}\nRole: {msg.get('role')}\nContent:\n"
        out += "".join(p.get("text", "") for p in (msg.get("content") or [])
                       if p.get("type") == "text")
    depth = sum(
        1
        for msg in history
        if any(
            p.get("type") == "text" and "Previous context has been compacted" in p.get("text", "")
            for p in (msg.get("content") or [])
        )
    )
    out += "\n"
    out += COMPACT_CASCADE_PROMPT_TEXT if depth >= 3 else COMPACT_PROMPT_TEXT
    out += "\n\n" + BALANCED_MODE_GUIDANCE
    return _enc(out)


# ---------------------------------------------------------------------------
# Public API (native with _compat fallback)
# ---------------------------------------------------------------------------


def build_payload(history: list[dict], preserved_thinking: bool = False) -> bytes:
    """One-pass OpenAI ChatCompletionMessageParam JSON for the history."""
    buf, structure = _build_structure(history)
    if _USE:
        return bytes(_native.soul.build_payload(buf, structure, bool(preserved_thinking)))
    return _compact(
        [
            _compat_convert_message(m, preserved_thinking=bool(preserved_thinking))
            for m in history
        ]
    )


def normalize_tool_call_ids(history: list[dict]) -> list[tuple]:
    """Plan of id rewrites: [(msg_index, call_index, new_id)] (call_index -1
    means the tool_call_id field)."""
    if _USE:
        buf, structure = _build_structure(history)
        fixes = _native.soul.normalize_tool_call_ids(buf, structure)
        return [
            (int(i), int(c), bytes(nid)) for i, c, nid in fixes
        ]
    return _compat_normalize_tool_call_ids(history)


def prune_scan(history: list[dict], policy: dict | None = None) -> list[tuple]:
    """(index, reason) per message; reasons 0=superseded_read 1=resolved_error
    2=compact 3=protect 4=oversized_output."""
    if _USE:
        buf, structure = _build_structure(history)
        return [
            (int(i), int(r)) for i, r in _native.soul.prune_scan(buf, structure, policy)
        ]
    return _compat_prune_scan(history, policy)


def prune_history(messages: list[dict], policy: dict) -> dict:
    """Full prune pass: Tier A drops + Tier B elision stubs.

    Returns ``{messages, elided, freed_tokens, earliest_removed_index}``.
    """
    if _USE:
        return _native.soul.prune_history(messages, policy)
    return _compat_prune_history(messages, policy)


def count_leading_reminders(history: list[dict]) -> int:
    if _USE:
        buf, structure = _build_structure(history)
        return int(_native.soul.count_leading_reminders(buf, structure))
    return _compat_count_leading_reminders(history)


def build_normalize_plan(history: list[dict]) -> list[tuple]:
    """normalize_history merge plan: [(index, op, target_index)]."""
    if _USE:
        buf, structure = _build_structure(history)
        return [
            (int(i), int(op), int(t))
            for i, op, t in _native.soul.build_normalize_plan(buf, structure)
        ]
    return _compat_build_normalize_plan(history)


def build_compaction_prompt(history: list[dict], system_prompt: str = "") -> bytes:
    """SimpleCompaction.prepare compact-message text (balanced mode defaults)."""
    if _USE:
        buf, structure = _build_structure(history)
        sp = _enc(system_prompt) if system_prompt else b""
        return bytes(_native.soul.build_compaction_prompt(buf, structure, sp))
    return _compat_build_compaction_prompt(history, system_prompt)


# ---------------------------------------------------------------------------
# Plan application helpers (used by the parity tests)
# ---------------------------------------------------------------------------


def apply_normalize_plan(history: list[dict], plan: list[tuple]) -> list[dict]:
    """Apply a normalize plan: op 1 merges the message's parts into the output
    message that originated at target_index."""
    result: list[dict] = []
    origin: list[int] = []  # origin index of each result entry
    by_origin: dict[int, int] = {}  # origin -> result position
    for i, op, target in plan:
        if op == 0:
            result.append(dict(history[i]))
            origin.append(i)
            by_origin[i] = len(result) - 1
        elif op == 1:
            pos = by_origin[target]
            result[pos] = {
                **result[pos],
                "content": list(result[pos].get("content") or [])
                + list(history[i].get("content") or []),
            }
        elif op == 3:
            continue
    return result


def apply_id_fixes(history: list[dict], fixes: list[tuple]) -> list[dict]:
    """Apply normalize_tool_call_ids fixes (deep-copied messages)."""
    out = [_loads(_compact(m)) for m in history]
    for msg_index, call_index, new_id in fixes:
        if isinstance(new_id, bytes):
            new_id = _dec(new_id)
        if call_index == -1:
            out[msg_index] = {**out[msg_index], "tool_call_id": new_id}
        else:
            tcs = list(out[msg_index].get("tool_calls") or [])
            tc = dict(tcs[call_index])
            tc = {**tc, "id": new_id}
            tcs[call_index] = tc
            out[msg_index] = {**out[msg_index], "tool_calls": tcs}
    return out
