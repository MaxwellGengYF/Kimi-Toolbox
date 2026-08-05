"""kimix_native.tools - line hashing, string find, grep line scan.

Native implementations live in ``runtime_py.tools`` (compiled kernels, GIL
released). The ``_compat`` mirrors below replicate the reference algorithms
from C:/dev/kimi-agent (hash_line.py::compute_line_hash, find_str.py::
find_in_file, grep_local.py backup_grep content-mode line scanning), so
``use_native("TOOLS") is False`` yields bit-identical behavior.

Public API:
  - line_hash(line, seed=0) -> int            (xxh32 & 0xFF of the filtered line)
  - line_hashes(content, seed=0) -> list[int] (chained per-line hashes, ints)
  - compute_line_hashes(content) -> list[str] (reference-shaped: 2-char nibble
      strings, HASH_SEED chain; exactly the reference's cumulative hashes for
      LF/CRLF files)
  - find_in_file(content, needle, case_sensitive=False, path="") -> list[dict]
      reference-shaped {file, line, column, content} records (1-based)
  - scan_lines(content, pattern, case_insensitive=True) -> list[(line_index,
      byte_offset, line_len)]   (0-based line index; literal-pattern matcher)
  - scan_lines_cb(content, callback) -> list[(line_index, byte_offset,
      line_len)]   (per-line Python matcher; offsets stay native)

Native-path notes (documented deviations):
  - find_in_file and scan_lines fold ASCII A-Z only (the reference uses full
    Unicode str.lower()); content or patterns with any non-ASCII byte are
    routed to _compat.
  - line_hash/line_hashes are Unicode-exact (the kernel embeds the Unicode
    whitespace + alnum tables) and run natively on any UTF-8 input.
"""

from __future__ import annotations

import json

from . import _native, use_native
from .json import _compat_indent2
from .soul import _build_structure

NIBBLE_STR = "ZPMQVRWSNKTXJBYH"
HASH_SEED = 0


# ---------------------------------------------------------------------------
# _compat: pure-Python mirrors of the reference algorithms
# ---------------------------------------------------------------------------

def _compat_xxh32(data: bytes, seed: int = 0) -> int:
    """Canonical XXH32 (public domain, Yann Collet) - mirror of xxhash.xxh32."""
    PRIME1 = 0x9E3779B1
    PRIME2 = 0x85EBCA77
    PRIME3 = 0xC2B2AE3D
    PRIME4 = 0x27D4EB2F
    PRIME5 = 0x165667B1
    MASK = 0xFFFFFFFF

    def rotl(x, r):
        return ((x << r) | (x >> (32 - r))) & MASK

    def xround(acc, inp):
        acc = (acc + inp * PRIME2) & MASK
        acc = rotl(acc, 13)
        acc = (acc * PRIME1) & MASK
        return acc

    n = len(data)
    p = 0
    if n >= 16:
        limit = n - 16
        v1 = (seed + PRIME1 + PRIME2) & MASK
        v2 = (seed + PRIME2) & MASK
        v3 = seed & MASK
        v4 = (seed - PRIME1) & MASK
        while p <= limit:
            v1 = xround(v1, int.from_bytes(data[p : p + 4], "little"))
            v2 = xround(v2, int.from_bytes(data[p + 4 : p + 8], "little"))
            v3 = xround(v3, int.from_bytes(data[p + 8 : p + 12], "little"))
            v4 = xround(v4, int.from_bytes(data[p + 12 : p + 16], "little"))
            p += 16
        h = (rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18)) & MASK
    else:
        h = (seed + PRIME5) & MASK
    h = (h + n) & MASK
    while p + 4 <= n:
        h = (h + int.from_bytes(data[p : p + 4], "little") * PRIME3) & MASK
        h = (rotl(h, 17) * PRIME4) & MASK
        p += 4
    while p < n:
        h = (h + data[p] * PRIME5) & MASK
        h = (rotl(h, 11) * PRIME1) & MASK
        p += 1
    h ^= h >> 15
    h = (h * PRIME2) & MASK
    h ^= h >> 13
    h = (h * PRIME3) & MASK
    h ^= h >> 16
    return h & MASK


def _compat_compute_line_hash(line_num: int, line: str, prev_hash: str | None) -> str:
    """Exact mirror of hash_line.py::compute_line_hash (41-69)."""
    if line.endswith("\r"):
        line = line[:-1]
    chars: list[str] = []
    has_significant = False
    for c in line:
        if not c.isspace():
            chars.append(c)
            if not has_significant and c.isalnum():
                has_significant = True
    if prev_hash is not None:
        seed = 0
        for c in prev_hash:
            seed = ((seed * 256) + ord(c)) & 0xFFFFFFFF
    elif has_significant:
        seed = HASH_SEED
    else:
        seed = line_num
    data = "".join(chars).encode("utf-8")
    h = _compat_xxh32(data, seed) & 0xFF
    return NIBBLE_STR[h >> 4] + NIBBLE_STR[h & 0x0F]


def _compat_compute_line_hashes(content: str) -> list[str]:
    hashes: list[str] = []
    prev: str | None = None
    for i, line in enumerate(content.splitlines(), 1):
        h = _compat_compute_line_hash(i, line, prev)
        hashes.append(h)
        prev = h
    return hashes


def _compat_find_in_file(content: str, needle: str, case_sensitive: bool) -> list[dict]:
    """Exact mirror of find_str.py::find_in_file (per-line, overlapping)."""
    results = []
    lines = content.splitlines(keepends=True)
    if not case_sensitive:
        search_lower = needle.lower()
    else:
        search_lower = needle
    for line_num, line in enumerate(lines, start=1):
        line_to_search = line if case_sensitive else line.lower()
        start = 0
        while True:
            idx = line_to_search.find(search_lower, start)
            if idx == -1:
                break
            results.append({
                "file": "",
                "line": line_num,
                "column": idx + 1,
                "content": line.rstrip("\n\r"),
            })
            start = idx + 1
    return results


def _compat_scan_lines(content: str, pattern: str, case_insensitive: bool) -> list[tuple[int, int, int]]:
    """backup_grep content-mode line scan with a literal-pattern matcher."""
    if not pattern:
        return []
    needle = pattern.lower() if case_insensitive else pattern
    hits: list[tuple[int, int, int]] = []
    start = 0
    line_index = 0
    n = len(content)
    while start < n:
        nl = content.find("\n", start)
        line_end = n if nl < 0 else nl
        line = content[start:line_end]
        test = line.lower() if case_insensitive else line
        if needle in test:
            hits.append((line_index, start, line_end - start))
        start = line_end + 1 if nl >= 0 else n
        line_index += 1
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def line_hash(line: str | bytes, seed: int = 0) -> int:
    """xxh32(non-whitespace(line without trailing CR), seed) & 0xFF."""
    if isinstance(line, (bytes, bytearray)):
        data = bytes(line)
    else:
        data = str(line).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None:
        return _native.tools.line_hash(data, seed)
    # _compat fallback replicates the kernel contract (CR strip + Unicode
    # whitespace filter + xxh32).
    text = data.decode("utf-8", "surrogatepass")
    if text.endswith("\r"):
        text = text[:-1]
    filtered = "".join(c for c in text if not c.isspace())
    return _compat_xxh32(filtered.encode("utf-8", "surrogatepass"), seed) & 0xFF


def line_hashes(content: str | bytes, seed: int = 0) -> list[int]:
    """Chained per-line hashes (ints 0..255) with reference seed semantics."""
    if isinstance(content, (bytes, bytearray)):
        data = bytes(content)
    else:
        data = str(content).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None:
        return _native.tools.line_hashes(data, seed)
    text = data.decode("utf-8", "surrogatepass")
    return [NIBBLE_STR.index(h[0]) * 16 + NIBBLE_STR.index(h[1])
            for h in _compat_compute_line_hashes(text)]


def compute_line_hashes(content: str) -> list[str]:
    """Reference-shaped cumulative hashes (2-char nibble strings)."""
    if use_native("TOOLS") and _native is not None:
        data = content.encode("utf-8", "surrogatepass")
        hs = _native.tools.line_hashes(data, HASH_SEED)
        return [NIBBLE_STR[h >> 4] + NIBBLE_STR[h & 0x0F] for h in hs]
    return _compat_compute_line_hashes(content)


def find_in_file(
    content: str, needle: str, case_sensitive: bool = False, path: str = ""
) -> list[dict]:
    """Reference-shaped records {file, line, column, content} (1-based)."""
    if use_native("TOOLS") and _native is not None and content.isascii() and needle.isascii():
        data = content.encode("utf-8", "surrogatepass")
        ndata = needle.encode("utf-8", "surrogatepass")
        matches = _native.tools.find_in_file(data, ndata, not case_sensitive)
        lines = data.split(b"\n")
        if data.endswith(b"\n"):
            lines = lines[:-1]
        results = []
        for line_index, col, length in matches:
            line_bytes = lines[line_index] if line_index < len(lines) else b""
            results.append({
                "file": path,
                "line": line_index + 1,
                "column": col + 1,
                "content": line_bytes.rstrip(b"\n\r").decode("utf-8", "surrogatepass"),
            })
        return results
    compat = _compat_find_in_file(content, needle, case_sensitive)
    for rec in compat:
        rec["file"] = path
    return compat


def scan_lines(
    content: str | bytes, pattern: str | bytes, case_insensitive: bool = True
) -> list[tuple[int, int, int]]:
    """(line_index 0-based, byte_offset, line_len) for matching lines."""
    if isinstance(content, (bytes, bytearray)):
        cdata = bytes(content)
    else:
        cdata = str(content).encode("utf-8", "surrogatepass")
    if isinstance(pattern, (bytes, bytearray)):
        pdata = bytes(pattern)
    else:
        pdata = str(pattern).encode("utf-8", "surrogatepass")
    if use_native("TOOLS") and _native is not None and cdata.isascii() and pdata.isascii():
        return _native.tools.scan_lines(cdata, pdata, case_insensitive)
    text = cdata.decode("utf-8", "surrogatepass")
    pat = pdata.decode("utf-8", "surrogatepass")
    return _compat_scan_lines(text, pat, case_insensitive)


def scan_lines_cb(
    content: str | bytes, callback
) -> list[tuple[int, int, int]]:
    """Per-line matcher callback; offsets computed natively (regex stays in
    Python). callback(line_bytes, line_index) -> bool."""
    if _native is None:
        raise RuntimeError("scan_lines_cb requires the native extension")
    if isinstance(content, (bytes, bytearray)):
        data = bytes(content)
    else:
        data = str(content).encode("utf-8", "surrogatepass")
    return _native.tools.scan_lines_cb(data, callback)

# ---------------------------------------------------------------------------
# Plan 016: session export markdown builder.
# ---------------------------------------------------------------------------

NL = chr(10)


def _exp_shorten(text: str, width: int) -> str:
    """kimi_cli/utils/string.py::shorten (whitespace-normalize + truncate)."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    cut = width - 1  # placeholder "..." (U+2026)
    if cut <= 0:
        return text[:width]
    space = text.rfind(" ", 0, cut + 1)
    if space > 0:
        cut = space
    return text[:cut].rstrip() + chr(0x2026)


def _exp_part_md(part: dict, stringify: bool = False) -> str:
    ptype = part.get("type")
    if ptype == "text":
        return part.get("text", "")
    if ptype == "think":
        think = part.get("think", "")
        if stringify:
            return "[think]"
        if not think.strip():
            return ""
        return "<details><summary>Thinking</summary>" + NL + NL + think + NL + NL + "</details>"
    if ptype == "image_url":
        return "[image]"
    if ptype == "audio_url":
        if stringify:
            aid = (part.get("audio_url") or {}).get("id")
            return "[audio:" + str(aid) + "]" if aid else "[audio]"
        return "[audio]"
    if ptype == "video_url":
        return "[video]"
    return "[" + ptype + "]"


def _exp_message_stringify(msg: dict) -> str:
    return "".join(_exp_part_md(p, stringify=True) for p in (msg.get("content") or []))


def _exp_internal_user(msg: dict) -> bool:
    content = msg.get("content") or []
    if msg.get("role") != "user" or len(content) != 1:
        return False
    part = content[0]
    text = part.get("text", "") if part.get("type") == "text" else ""
    return (
        text.strip().startswith("<system>CHECKPOINT")
        or text.strip().startswith("<system-reminder>")
        or text.lstrip().startswith("<notification ")
    )


_EXP_HINT_KEYS = ("path", "file_path", "command", "query", "url", "name", "pattern")


def _exp_extract_hint(args_json: str) -> str:
    try:
        parsed = json.loads(args_json)
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    for key in _EXP_HINT_KEYS:
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            return _exp_shorten(val, 60)
    for val in parsed.values():
        if isinstance(val, str) and 0 < len(val) <= 80:
            return _exp_shorten(val, 60)
    return ""


def _exp_format_tool_call_md(tc: dict) -> str:
    args_raw = (tc.get("function") or {}).get("arguments") or "{}"
    hint = _exp_extract_hint(args_raw)
    title = "#### Tool Call: " + str((tc.get("function") or {}).get("name", ""))
    if hint:
        title += " (`" + hint + "`)"
    try:
        parsed = json.loads(args_raw)
        args_formatted = _compat_indent2(parsed).decode("utf-8", "surrogatepass")
    except ValueError:
        args_formatted = args_raw
    return (title + NL + "<!-- call_id: " + str(tc.get("id", "")) + " -->"
            + NL + "```json" + NL + args_formatted + NL + "```")


def _exp_format_tool_result_md(msg: dict, tool_name: str, hint: str) -> str:
    call_id = msg.get("tool_call_id") or "unknown"
    result_parts = []
    for part in msg.get("content") or []:
        text = _exp_part_md(part)
        if text.strip():
            result_parts.append(text)
    result_text = NL.join(result_parts)
    summary = "Tool Result: " + str(tool_name)
    if hint:
        summary += " (`" + str(hint) + "`)"
    return ("<details><summary>" + summary + "</summary>" + NL + NL
            + "<!-- call_id: " + call_id + " -->" + NL
            + result_text + NL + NL + "</details>")


def _exp_group_turns(history: list) -> list:
    turns = []
    current = []
    for msg in history:
        if _exp_internal_user(msg):
            continue
        if msg.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def _exp_format_turn_md(messages: list, turn_number: int) -> str:
    lines = ["## Turn " + str(turn_number), ""]
    tool_call_info = {}
    assistant_header_written = False
    for msg in messages:
        if _exp_internal_user(msg):
            continue
        if msg.get("role") == "user":
            lines.append("### User")
            lines.append("")
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
        elif msg.get("role") == "assistant":
            if not assistant_header_written:
                lines.append("### Assistant")
                lines.append("")
                assistant_header_written = True
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
            for tc in msg.get("tool_calls") or []:
                args_raw = (tc.get("function") or {}).get("arguments") or "{}"
                hint = _exp_extract_hint(args_raw)
                tool_call_info[tc.get("id", "")] = (
                    (tc.get("function") or {}).get("name", ""),
                    hint,
                )
                lines.append(_exp_format_tool_call_md(tc))
                lines.append("")
        elif msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id") or ""
            name, hint = tool_call_info.get(tc_id, ("unknown", ""))
            lines.append(_exp_format_tool_result_md(msg, name, hint))
            lines.append("")
        elif msg.get("role") in ("system", "developer"):
            lines.append("### " + str(msg.get("role")).capitalize())
            lines.append("")
            for part in msg.get("content") or []:
                text = _exp_part_md(part)
                if text.strip():
                    lines.append(text)
                    lines.append("")
    return NL.join(lines)


def _compat_build_export_markdown(history: list, opts: dict) -> bytes:
    token_count = int(opts.get("token_count", 0))
    lines = ["---",
             "session_id: " + str(opts.get("session_id", "")),
             "exported_at: " + str(opts.get("exported_at", "")),
             "work_dir: " + str(opts.get("work_dir", "")),
             "message_count: " + str(len(history)),
             "token_count: " + str(token_count),
             "---", "", "# Kimi Session Export", ""]

    turns = _exp_group_turns(history)

    topic = ""
    for msg in history:
        if msg.get("role") == "user" and not _exp_internal_user(msg):
            topic = _exp_shorten(_exp_message_stringify(msg), 80)
            break
    n_tool_calls = sum(len(m.get("tool_calls") or []) for m in history)

    comma_tokens = format(token_count, ",")
    overview = ["## Overview", "",
                ("- **Topic**: " + topic) if topic else "- **Topic**: (empty)",
                "- **Conversation**: " + str(len(turns)) + " turns | "
                + str(n_tool_calls) + " tool calls | " + comma_tokens + " tokens",
                "", "---"]
    lines.append(NL.join(overview))
    lines.append("")
    for idx, turn_messages in enumerate(turns):
        lines.append(_exp_format_turn_md(turn_messages, idx + 1))
    return NL.join(lines).encode("utf-8", "surrogatepass")


def build_export_markdown(history: list, opts: dict) -> bytes:
    """Full session export markdown (export.py build_export_markdown)."""
    if use_native("TOOLS") and _native is not None:
        buf, structure = _build_structure(history)
        return bytes(_native.tools.build_export_markdown(buf, structure, dict(opts)))
    return _compat_build_export_markdown(history, opts)
