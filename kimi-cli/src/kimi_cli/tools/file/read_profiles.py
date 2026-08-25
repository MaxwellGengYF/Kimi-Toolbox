"""CPU/sample profile rendering helpers for the ``read`` tool.

Ports the V8 ``.cpuprofile`` and macOS ``sample`` summary renderers.
Malformed inputs return ``None`` so the caller can fall back to plain text.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import regex

__all__ = [
    "is_cpuprofile_path",
    "is_sample_profile_path",
    "render_cpu_profile",
    "render_sample_profile",
    "MAX_PROFILE_SUMMARY_BYTES",
]

MAX_PROFILE_SUMMARY_BYTES = 32 * 1024 * 1024


# ── CPU profile (.cpuprofile) ───────────────────────────────────────────────


def is_cpuprofile_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".cpuprofile"


@dataclass
class _CpuNode:
    node_id: int
    call_frame: dict[str, Any]
    self_micros: int = 0
    total_micros: int = 0
    children: list["_CpuNode"] = field(default_factory=list)

    def label(self) -> str:
        cf = self.call_frame
        name = cf.get("functionName") or "(anonymous)"
        url = cf.get("url") or ""
        line = cf.get("lineNumber", -1)
        if url and line >= 0:
            return f"{name} ({url}:{line})"
        return name


def _parse_cpu_profile(text: str) -> dict[str, Any] | None:
    try:
        data = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # Chrome DevTools wraps the profile in {"profile": {...}}.
    profile = data.get("profile", data)
    if not isinstance(profile, dict):
        return None
    nodes = profile.get("nodes")
    samples = profile.get("samples")
    time_deltas = profile.get("timeDeltas")
    start_time = profile.get("startTime")
    end_time = profile.get("endTime")
    if not isinstance(nodes, list) or not isinstance(samples, list):
        return None
    return {
        "profile": profile,
        "nodes": nodes,
        "samples": samples,
        "time_deltas": time_deltas,
        "start_time": start_time,
        "end_time": end_time,
    }


def _build_cpu_tree(nodes: list[dict[str, Any]]) -> dict[int, _CpuNode]:
    node_map: dict[int, _CpuNode] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_id = n.get("id")
        if not isinstance(node_id, int):
            continue
        cf = n.get("callFrame") or {}
        if not isinstance(cf, dict):
            cf = {}
        node_map[node_id] = _CpuNode(node_id=node_id, call_frame=cf)
    for n in nodes:
        if not isinstance(n, dict):
            continue
        node_id = n.get("id")
        parent_id = n.get("parent")
        if isinstance(node_id, int) and isinstance(parent_id, int) and parent_id in node_map:
            node_map[parent_id].children.append(node_map[node_id])
    return node_map


def _compute_self_times(
    parsed: dict[str, Any],
    node_map: dict[int, _CpuNode],
) -> tuple[dict[int, _CpuNode], int]:
    samples = parsed["samples"]
    time_deltas = parsed["time_deltas"]
    nodes = parsed["nodes"]

    # Determine sample interval from timeDeltas when available.
    total_micros = 0
    positive_deltas: list[int] = []
    if isinstance(time_deltas, list) and len(time_deltas) == len(samples):
        for delta in time_deltas:
            if isinstance(delta, (int, float)) and delta > 0:
                positive_deltas.append(int(delta))
                total_micros += int(delta)
    else:
        # Fall back to hitCount * interval guess from start/end.
        start = parsed.get("start_time") or 0
        end = parsed.get("end_time") or 0
        duration = max(0, end - start)
        total_micros = int(duration)
        positive_deltas = [int(duration / max(1, len(samples)))] * len(samples)

    avg_interval = total_micros // max(1, len(positive_deltas) or len(samples))

    # hitCount fallback: when nodes carry hitCount but no samples/timeDeltas.
    if not samples and nodes and all(isinstance(n, dict) and n.get("hitCount") for n in nodes):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            node_id = n.get("id")
            hit = n.get("hitCount", 0)
            if isinstance(node_id, int) and isinstance(hit, int) and node_id in node_map:
                node_map[node_id].self_micros = hit * avg_interval
                total_micros += hit * avg_interval
        return node_map, total_micros

    for sample_id in samples:
        if isinstance(sample_id, int) and sample_id in node_map:
            node_map[sample_id].self_micros += avg_interval
            total_micros += avg_interval

    return node_map, total_micros


def _promote_root(node_map: dict[int, _CpuNode], root_id: int | None) -> _CpuNode | None:
    if root_id is None:
        # Pick the node with no parent as root.
        for node in node_map.values():
            # Heuristic: root has functionName "(root)" or no parent.
            if node.call_frame.get("functionName") == "(root)":
                return node
        return next(iter(node_map.values())) if node_map else None
    return node_map.get(root_id)


def _prune_hot_tree(node: _CpuNode, threshold: int, depth: int = 0, max_depth: int = 8) -> list[str]:
    if depth > max_depth:
        return []
    if node.total_micros < threshold and depth > 0:
        return []
    indent = "  " * depth
    pct = (node.total_micros / max(1, node.total_micros)) * 100 if depth == 0 else 0.0
    line = f"{indent}{node.label()}"
    lines = [line]
    for child in sorted(node.children, key=lambda c: c.total_micros, reverse=True):
        if child.total_micros >= threshold:
            lines.extend(_prune_hot_tree(child, threshold, depth + 1, max_depth))
    return lines


def _aggregate_totals(node_map: dict[int, _CpuNode]) -> None:
    # Simple post-order accumulation of total times.
    def walk(node: _CpuNode) -> int:
        total = node.self_micros
        for child in node.children:
            total += walk(child)
        node.total_micros = total
        return total

    # Find roots (nodes not referenced as children) and walk from each.
    child_ids = set()
    for node in node_map.values():
        for child in node.children:
            child_ids.add(child.node_id)
    roots = [n for nid, n in node_map.items() if nid not in child_ids]
    for root in roots:
        walk(root)


def render_cpu_profile(text: str) -> str | None:
    """Render a V8 CPU profile summary, or ``None`` if not a valid profile."""
    parsed = _parse_cpu_profile(text)
    if parsed is None:
        return None

    profile = parsed["profile"]
    nodes = parsed["nodes"]
    node_map = _build_cpu_tree(nodes)
    if not node_map:
        return None

    node_map, total_micros = _compute_self_times(parsed, node_map)
    _aggregate_totals(node_map)

    start = profile.get("startTime", 0)
    end = profile.get("endTime", 0)
    wall_micros = max(0, end - start) or total_micros
    sample_count = len(parsed["samples"]) or sum(
        n.hitCount for n in nodes if isinstance(n, dict) and isinstance(n.get("hitCount"), int)
    )
    avg_interval = total_micros // max(1, sample_count)

    # Prune threshold: max(3 * avgInterval, 2% total).
    threshold = max(3 * avg_interval, int(total_micros * 0.02))

    # Find root and top self-time functions.
    root_id = profile.get("root")
    root = _promote_root(node_map, root_id)

    # Top functions by self time, excluding idle.
    self_nodes = [
        n
        for n in node_map.values()
        if n.self_micros > 0 and n.call_frame.get("functionName") != "(idle)"
    ]
    self_nodes.sort(key=lambda n: n.self_micros, reverse=True)
    top_functions = self_nodes[:20]

    lines: list[str] = []
    lines.append(
        f"V8 CPU profile: {wall_micros}μs wall clock, "
        f"{sample_count} samples (avg interval {avg_interval}μs)"
    )
    lines.append("")

    lines.append("## Hot paths")
    if root is not None:
        # Treat root as 100% and prune children.
        for child in sorted(root.children, key=lambda c: c.total_micros, reverse=True):
            if child.total_micros >= threshold:
                lines.extend(_prune_hot_tree(child, threshold))
    else:
        for node in sorted(self_nodes, key=lambda n: n.total_micros, reverse=True)[:20]:
            lines.append(node.label())
    lines.append("")

    lines.append("## Top functions by self time")
    for i, node in enumerate(top_functions, 1):
        pct = node.self_micros / max(1, total_micros) * 100
        lines.append(f"{i}. {node.label()} — {node.self_micros}μs ({pct:.2f}%)")
    lines.append("")

    lines.append(
        "[Summarized view of CPU profile. Use profile_raw=True to read the original JSON.]"
    )
    return "\n".join(lines)


# ── Sample profile (macOS sample .sample.txt) ─────────────────────────────────


def is_sample_profile_path(path: str) -> bool:
    return Path(path).name.endswith(".sample.txt") or Path(path).suffix.lower() == ".sample"


_WAIT_SYMBOLS = frozenset({
    "_pthread_cond_wait", "__psynch_cvwait", "__semwait_signal", "mach_msg_trap",
    "__workq_kernreturn", "__ulock_wait", "__recvfrom", "__select", "__poll",
    "kevent", "epoll_wait", "poll", "select", "nanosleep", "usleep", "sleep",
})


@dataclass
class _SampleFrame:
    raw: str
    symbol: str = ""
    module: str = ""
    self_count: int = 0

    def __post_init__(self) -> None:
        self.symbol, self.module = _parse_frame_text(self.raw)


def _parse_frame_text(text: str) -> tuple[str, str]:
    """Parse a sample frame line like 'symbol (in module)' or 'symbol + 123'."""
    text = text.strip()
    m = regex.match(r"^(.*?)\s+\(in\s+(.+)\)\s*(?:\+.*)?$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = regex.match(r"^(.*?)\s*\+.*$", text)
    if m:
        return m.group(1).strip(), ""
    return text, ""


def _demangle_symbol(symbol: str) -> str:
    """Best-effort Rust v0 / legacy C++ demangling."""
    # Rust v0 symbols start with _R and use a specific alphabet.
    if symbol.startswith("_R") and len(symbol) > 2:
        # Very rough decoding: collapse _R... to a readable-ish placeholder.
        return f"{symbol[:10]}..."
    # Legacy _ZN...E Itanium mangling.
    if symbol.startswith("_ZN") and symbol.endswith("E"):
        inner = symbol[3:-1]
        parts: list[str] = []
        i = 0
        while i < len(inner):
            # Read length-prefixed identifier.
            m = regex.match(r"(\d+)", inner[i:])
            if not m:
                break
            length = int(m.group(1))
            i += len(m.group(1))
            parts.append(inner[i : i + length])
            i += length
        if parts:
            return "::".join(parts)
    return symbol


def _is_wait_frame(symbol: str) -> bool:
    return any(sym in symbol for sym in _WAIT_SYMBOLS)


def _parse_sample_profile(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    if not lines:
        return None
    # Recognize macOS sample preamble.
    preamble_match = False
    call_graph_started = False
    for line in lines[:20]:
        if "Sampling process" in line or "Analysis of sampling" in line:
            preamble_match = True
        if "Call graph:" in line:
            call_graph_started = True
    if not (preamble_match or call_graph_started):
        return None

    threads: dict[str, list[list[_SampleFrame]]] = {}
    current_thread: str | None = None
    current_stack: list[_SampleFrame] = []

    # Decorators used by sample(1): + siblings, ! running, : suspended, | subtree.
    decorator_regex = regex.compile(r"^(\s*)[+|!:](\s*)(.*)$")


    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Thread_"):
            if current_thread and current_stack:
                threads.setdefault(current_thread, []).append(current_stack)
            current_thread = stripped.split()[0]
            current_stack = []
            continue
        if stripped == "Call graph:":
            continue
        m = decorator_regex.match(line)
        if not m:
            continue
        indent = len(m.group(1)) + len(m.group(2))
        frame_text = m.group(3).strip()
        if not frame_text:
            continue
        frame = _SampleFrame(raw=frame_text)
        # Deeper indent means deeper stack; shallower means new branch.
        depth = indent // 2
        if depth >= len(current_stack):
            current_stack.append(frame)
        else:
            current_stack = current_stack[:depth] + [frame]
        # Each leaf frame occurrence is a self sample; record at the leaf.
        if current_thread is None:
            current_thread = "Thread_0"
        threads.setdefault(current_thread, []).append(current_stack.copy())

    if current_thread and current_stack:
        threads.setdefault(current_thread, []).append(current_stack)

    return {"threads": threads}


def render_sample_profile(text: str) -> str | None:
    """Render a macOS sample profile summary, or ``None`` if not recognized."""
    parsed = _parse_sample_profile(text)
    if parsed is None:
        return None

    threads = parsed["threads"]
    self_counts: dict[str, int] = {}
    for thread, stacks in threads.items():
        for stack in stacks:
            if not stack:
                continue
            leaf = stack[-1]
            symbol = _demangle_symbol(leaf.symbol) or leaf.raw
            self_counts[symbol] = self_counts.get(symbol, 0) + 1

    total_samples = sum(self_counts.values())
    if total_samples == 0:
        return None

    # Exclude idle/wait frames from top ranking.
    active_counts = {s: c for s, c in self_counts.items() if not _is_wait_frame(s)}
    if not active_counts:
        active_counts = self_counts

    top = sorted(active_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    idle_total = sum(c for s, c in self_counts.items() if _is_wait_frame(s))
    idle_pct = idle_total / total_samples * 100

    lines: list[str] = []
    lines.append(
        f"macOS sample profile: {total_samples} samples across {len(threads)} thread(s), "
        f"{idle_pct:.1f}% in wait/idle frames"
    )
    lines.append("")
    lines.append("## Top functions by self samples")
    for i, (symbol, count) in enumerate(top, 1):
        pct = count / total_samples * 100
        lines.append(f"{i}. {symbol} — {count} self samples ({pct:.2f}%)")
    lines.append("")
    lines.append(
        "[Summarized view of sample profile. Use profile_raw=True to read the original text.]"
    )
    return "\n".join(lines)
