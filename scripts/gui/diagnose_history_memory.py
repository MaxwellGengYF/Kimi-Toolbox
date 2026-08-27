"""Profile random history seeks, rapid scrollbar drags, and retained memory.

The profiler separates wire hydration, timeline flattening, Qt model replacement,
layout, and painting. It can also generate a deterministic large wire log:

    uv run python scripts/gui/diagnose_history_memory.py .kimix_cache/profiles/huge.jsonl \
        --target-gib 2 --body-bytes 4096 --jumps 80 --drag-events 600
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import ctypes
import gc
import os
import pstats
import random
import statistics
import sys
import time
import tracemalloc
import weakref
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from kimix_gui.history import (
    Timeline,
    WireHistoryIndex,
    _read_wire_turns,
    _scan_wire_history_index,
)
from kimix_gui.qt.transcript import Transcript

MIB = 1024**2
_TRACKED_TYPES = (
    "ActivityEntry",
    "HistoryEntry",
    "QTextDocument",
    "TextEntry",
    "TranscriptRow",
)


@dataclass(slots=True)
class SeekSample:
    target: int
    planned_turns: int
    hydrated_turns: int
    entries: int
    plan_ms: float
    drop_ms: float
    hydrate_ms: float
    trim_ms: float
    flatten_ms: float
    model_ms: float
    layout_ms: float
    paint_ms: float

    @property
    def total_ms(self) -> float:
        return sum(
            (
                self.plan_ms,
                self.drop_ms,
                self.hydrate_ms,
                self.trim_ms,
                self.flatten_ms,
                self.model_ms,
                self.layout_ms,
                self.paint_ms,
            )
        )


@dataclass(slots=True)
class MemorySample:
    phase: str
    cycle: int
    python_mib: float
    python_peak_mib: float
    rss_mib: float
    peak_rss_mib: float
    materialized_turns: int
    materialized_entries: int
    presentation_rows: int
    history_source_rows: int
    documents: int
    height_entries: int
    tracked_objects: dict[str, int]


def process_memory_mib() -> tuple[float, float]:
    """Return current and peak resident memory without an optional dependency."""

    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = peak / (MIB if sys.platform == "darwin" else 1024)
        return 0.0, peak_mib

    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_process = kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(get_process(), ctypes.byref(counters), counters.cb):
        return 0.0, 0.0
    return counters.WorkingSetSize / MIB, counters.PeakWorkingSetSize / MIB


def _wire_line(message_type: str, payload: dict[str, Any], timestamp: float) -> bytes:
    return (
        orjson.dumps(
            {
                "timestamp": timestamp,
                "message": {"type": message_type, "payload": payload},
            }
        )
        + b"\n"
    )


def _tool_messages(turn: int, filler: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    families = ("read", "shell", "python", "edit", "todo_write", "search")
    family = families[turn % len(families)]
    call_id = f"{family}-{turn}"
    arguments: dict[str, Any]
    display: list[dict[str, Any]]
    if family == "read":
        arguments = {"path": f"C:/src/{turn}/key: value.py", "line_start": turn % 200}
        display = [{"type": "brief", "text": f"Read synthetic file {turn}"}]
    elif family == "shell":
        arguments = {"command": f"rg --line-number profile-{turn} src"}
        display = [{"type": "shell", "language": "powershell"}]
    elif family == "python":
        arguments = {"code": f"print('profile:{turn}')", "timeout": 30}
        display = [{"type": "shell", "language": "python"}]
    elif family == "edit":
        arguments = {
            "path": f"C:/src/{turn}/module.py",
            "old_text": "value: old",
            "new_text": "value: new",
        }
        display = [
            {
                "type": "diff",
                "path": f"C:/src/{turn}/module.py",
                "old_text": "value: old\n",
                "new_text": "value: new\n",
                "old_start": 1,
                "new_start": 1,
                "is_summary": False,
            }
        ]
    elif family == "todo_write":
        arguments = {
            "todos": [
                {"title": f"Profile turn {turn}", "status": "in_progress"},
                {"title": "Verify memory plateau", "status": "pending"},
            ]
        }
        display = [
            {
                "type": "todo",
                "items": [
                    {
                        "title": f"Profile turn {turn}",
                        "status": "in_progress",
                        "notes": "mixed message fixture",
                        "depth": 0,
                    },
                    {
                        "title": "Verify memory plateau",
                        "status": "pending",
                        "notes": None,
                        "depth": 1,
                    },
                ],
            }
        ]
    else:
        arguments = {"query": f"structured transcript profiler {turn}", "limit": 20}
        display = [
            {
                "type": "unknown",
                "data": {"query": arguments["query"], "matches": turn % 31},
            }
        ]
    arguments_text = orjson.dumps(arguments).decode()
    return (
        (
            "ToolCall",
            {
                "type": "function",
                "id": call_id,
                "function": {"name": family, "arguments": arguments_text},
                "extras": {"fixture": "mixed", "turn": turn},
            },
        ),
        (
            "ToolResult",
            {
                "tool_call_id": call_id,
                "return_value": {
                    "is_error": turn % 47 == 0,
                    "output": f"{family} output for turn {turn}\n" + filler[:1024],
                    "message": "synthetic failure" if turn % 47 == 0 else "",
                    "display": display,
                    "extras": {"duration_ms": turn % 1000},
                },
            },
        ),
    )


def _mixed_messages(turn: int, filler: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return one deterministic mixture slice; every 12 turns covers all families."""

    variant = turn % 12
    if variant < 5:
        return _tool_messages(turn, filler)
    if variant == 5:
        request_id = f"approval-{turn}"
        return (
            (
                "ApprovalRequest",
                {
                    "id": request_id,
                    "tool_call_id": f"shell-{turn}",
                    "sender": "Shell",
                    "action": "run command",
                    "description": f"Approve profiler command for turn {turn}",
                    "source_kind": "foreground_turn",
                    "source_id": f"turn-{turn}",
                    "agent_id": None,
                    "subagent_type": None,
                    "source_description": "2 GiB mixed fixture",
                    "display": [{"type": "brief", "text": "No external side effect"}],
                },
            ),
            (
                "ApprovalResponse",
                {
                    "request_id": request_id,
                    "response": "reject" if turn % 24 == 5 else "approve",
                    "feedback": "synthetic decision",
                },
            ),
        )
    if variant == 6:
        severity = "error" if turn % 24 == 6 else "info"
        return (
            (
                "Notification",
                {
                    "id": f"notice-{turn}",
                    "category": "profiler",
                    "type": "fixture",
                    "source_kind": "session",
                    "source_id": f"turn-{turn}",
                    "title": "Synthetic history event",
                    "body": f"Notification body for turn {turn}: key: value",
                    "severity": severity,
                    "created_at": float(turn),
                    "payload": {"turn": turn, "severity": severity},
                },
            ),
            (
                "StatusUpdate",
                {
                    "context_usage": (turn % 100) / 100,
                    "context_tokens": turn % 120000,
                    "max_context_tokens": 128000,
                    "token_usage": {
                        "input_other": 100 + turn % 1000,
                        "output": 50 + turn % 500,
                        "input_cache_read": turn % 200,
                        "input_cache_creation": turn % 100,
                    },
                    "message_id": f"message-{turn}",
                    "mcp_status": None,
                },
            ),
        )
    if variant == 7:
        return (
            ("StepBegin", {"n": turn}),
            (
                "StepRetry",
                {
                    "n": turn,
                    "next_attempt": 2,
                    "max_attempts": 3,
                    "wait_s": 0.25,
                    "error_type": "SyntheticTransientError",
                    "status_code": 429,
                },
            ),
            ("StepInterrupted", {}),
        )
    if variant == 8:
        compaction_id = f"compact-{turn}"
        return (
            (
                "CompactionBegin",
                {
                    "compaction_id": compaction_id,
                    "trigger": "overflow",
                    "shadowed_tokens": 64000,
                },
            ),
            (
                "CompactionEnd",
                {
                    "compaction_id": compaction_id,
                    "trigger": "overflow",
                    "shadowed_tokens": 64000,
                    "estimated_token_count": 32000,
                    "error": None,
                },
            ),
            ("MCPLoadingBegin", {}),
            ("MCPLoadingEnd", {}),
        )
    if variant == 9:
        return (
            (
                "ContentPart",
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"https://example.invalid/image/{turn}.png",
                        "id": f"image-{turn}",
                    },
                },
            ),
            (
                "ContentPart",
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"https://example.invalid/audio/{turn}.mp3",
                        "id": f"audio-{turn}",
                    },
                },
            ),
            (
                "ContentPart",
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"https://example.invalid/video/{turn}.mp4",
                        "id": f"video-{turn}",
                    },
                },
            ),
        )
    if variant == 10:
        return (
            (
                "SubagentEvent",
                {
                    "parent_tool_call_id": f"task-{turn}",
                    "agent_id": f"agent-{turn % 32}",
                    "subagent_type": "researcher",
                    "event": {
                        "type": "ContentPart",
                        "payload": {"type": "think", "think": f"Subagent analysis {turn}"},
                    },
                },
            ),
            (
                "SubagentEvent",
                {
                    "parent_tool_call_id": f"task-{turn}",
                    "agent_id": f"agent-{turn % 32}",
                    "subagent_type": "researcher",
                    "event": {
                        "type": "ContentPart",
                        "payload": {"type": "text", "text": f"Subagent result {turn}"},
                    },
                },
            ),
        )
    side_question_id = f"btw-{turn}"
    return (
        (
            "BtwBegin",
            {
                "id": side_question_id,
                "question": f"Side question for profiler turn {turn}?",
            },
        ),
        (
            "BtwEnd",
            {
                "id": side_question_id,
                "response": f"Side answer {turn}: key: value",
                "error": None,
            },
        ),
        (
            "HookTriggered",
            {"event": "PreToolUse", "target": f"read-{turn}", "hook_count": 2},
        ),
        (
            "HookResolved",
            {
                "event": "PreToolUse",
                "target": f"read-{turn}",
                "action": "allow",
                "reason": "synthetic hook",
                "duration_ms": turn % 50,
            },
        ),
    )


def generate_wire(
    path: Path,
    *,
    turns: int,
    target_bytes: int,
    body_bytes: int,
) -> tuple[int, int]:
    """Create an actual-size, deterministic wire log with mixed message families."""

    path.parent.mkdir(parents=True, exist_ok=True)
    paragraph = (
        "- structured history item with **bold text**, `inline_code()`, a colon: value, "
        "and [a link](https://example.invalid/profile)\n"
    )
    filler = (paragraph * (max(256, body_bytes) // len(paragraph) + 2))[:body_bytes]
    buffered = bytearray()
    header = orjson.dumps({"type": "metadata", "protocol_version": "3"}) + b"\n"
    written = len(header)
    turn = 0
    with path.open("wb") as wire_file:
        wire_file.write(header)
        while (written + len(buffered) < target_bytes) if target_bytes > 0 else (turn < turns):
            timestamp = float(turn)
            user = f"Inspect synthetic turn {turn}: key: value and C:/src/{turn}/file.py"
            answer = f"## Result for turn {turn}\n\n{filler}"
            split = len(answer) // 2
            buffered.extend(_wire_line("TurnBegin", {"user_input": user}, timestamp))
            for fragment in (answer[:split], answer[split:]):
                buffered.extend(
                    _wire_line(
                        "ContentPart",
                        {"type": "text", "text": fragment},
                        timestamp,
                    )
                )
            buffered.extend(
                _wire_line(
                    "ContentPart",
                    {"type": "think", "think": f"Reasoning for turn {turn}: {filler[:256]}"},
                    timestamp,
                )
            )
            for message_type, payload in _mixed_messages(turn, filler):
                buffered.extend(_wire_line(message_type, payload, timestamp))
            buffered.extend(_wire_line("TurnEnd", {}, timestamp))
            if len(buffered) >= 8 * MIB:
                wire_file.write(buffered)
                written += len(buffered)
                buffered.clear()
            turn += 1
        if buffered:
            wire_file.write(buffered)
            written += len(buffered)
    return turn, written


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _stage_summary(samples: list[SeekSample]) -> dict[str, dict[str, float]]:
    names = (
        "plan_ms",
        "drop_ms",
        "hydrate_ms",
        "trim_ms",
        "flatten_ms",
        "model_ms",
        "layout_ms",
        "paint_ms",
        "total_ms",
    )
    return {
        name: {
            "mean": statistics.fmean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "max": max(values),
        }
        for name in names
        if (values := [float(getattr(sample, name)) for sample in samples])
    }


def _print_stage_summary(summary: dict[str, dict[str, float]]) -> None:
    print("random_seek_latency_ms")
    for name, values in summary.items():
        print(
            f"  {name.removesuffix('_ms'):>8} mean={values['mean']:8.2f} "
            f"p50={values['p50']:8.2f} p95={values['p95']:8.2f} "
            f"max={values['max']:8.2f}"
        )


def _flush_qt_deletes(qt_app: QApplication) -> None:
    qt_app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


def _tracked_object_counts() -> dict[str, int]:
    counts = Counter(type(item).__name__ for item in gc.get_objects())
    return {name: counts[name] for name in _TRACKED_TYPES}


def _take_memory_sample(
    phase: str,
    cycle: int,
    timeline: Timeline,
    transcript: Transcript,
) -> MemorySample:
    current, peak = tracemalloc.get_traced_memory()
    rss, peak_rss = process_memory_mib()
    history_source = getattr(transcript.model(), "_history_source", ())
    return MemorySample(
        phase=phase,
        cycle=cycle,
        python_mib=current / MIB,
        python_peak_mib=peak / MIB,
        rss_mib=rss,
        peak_rss_mib=peak_rss,
        materialized_turns=timeline.materialized_turn_count,
        materialized_entries=timeline.materialized_record_count,
        presentation_rows=len(transcript.records),
        history_source_rows=len(history_source),
        documents=len(transcript.bodies.documents),
        height_entries=len(transcript.bodies.heights),
        tracked_objects=_tracked_object_counts(),
    )


async def _seek_and_display(
    timeline: Timeline,
    transcript: Transcript,
    qt_app: QApplication,
    target: int,
    *,
    paint: bool,
    ui_profile: cProfile.Profile | None = None,
) -> SeekSample:
    started = time.perf_counter()
    start, end = timeline.window_bounds(target)
    after_plan = time.perf_counter()
    timeline._drop_outside_bounds(start, end)
    after_drop = time.perf_counter()
    hydrated = await timeline.materialize_turns(start, end)
    after_hydrate = time.perf_counter()
    timeline._trim_to_budget(target)
    after_trim = time.perf_counter()

    if ui_profile is not None:
        ui_profile.enable()
    entries = timeline.history_entries()
    after_flatten = time.perf_counter()
    transcript.replace_history(entries, target_turn=target)
    after_model = time.perf_counter()
    transcript.jump_to_turn(target)
    after_layout = time.perf_counter()
    qt_app.processEvents()
    if paint:
        frame = transcript.viewport().grab()
        del frame
        qt_app.processEvents()
    after_paint = time.perf_counter()
    if ui_profile is not None:
        ui_profile.disable()

    return SeekSample(
        target=target,
        planned_turns=end - start,
        hydrated_turns=hydrated,
        entries=len(entries),
        plan_ms=(after_plan - started) * 1000,
        drop_ms=(after_drop - after_plan) * 1000,
        hydrate_ms=(after_hydrate - after_drop) * 1000,
        trim_ms=(after_trim - after_hydrate) * 1000,
        flatten_ms=(after_flatten - after_trim) * 1000,
        model_ms=(after_model - after_flatten) * 1000,
        layout_ms=(after_layout - after_model) * 1000,
        paint_ms=(after_paint - after_layout) * 1000,
    )


def _profile_parser(
    index: WireHistoryIndex,
    targets: list[int],
    output: Path,
    *,
    samples: int,
) -> None:
    timeline = Timeline(index=index)
    timeline._ensure_turn_slots(index.total_turns)
    profiler = cProfile.Profile()
    profiler.enable()
    for target in targets[:samples]:
        start, end = timeline.window_bounds(target)
        start_offset, _ = timeline.turn_range(start)
        _, end_offset = timeline.turn_range(end - 1)
        _read_wire_turns(index.path, start_offset, end_offset)
    profiler.disable()
    profiler.dump_stats(output)


def _print_profile(path: Path, *, title: str, limit: int = 20) -> None:
    print(f"{title} profile={path}")
    pstats.Stats(str(path)).strip_dirs().sort_stats("cumulative").print_stats(limit)


def _rapid_drag(
    transcript: Transcript,
    qt_app: QApplication,
    *,
    events: int,
    paint_every: int,
    profiler: cProfile.Profile | None = None,
) -> dict[str, float | int]:
    counts = {
        "top": 0,
        "bottom": 0,
        "model_reset": 0,
        "rows_inserted": 0,
        "rows_removed": 0,
    }

    def top() -> None:
        counts["top"] += 1

    def bottom() -> None:
        counts["bottom"] += 1

    def model_reset() -> None:
        counts["model_reset"] += 1

    def rows_inserted(_parent: object, first: int, last: int) -> None:
        counts["rows_inserted"] += last - first + 1

    def rows_removed(_parent: object, first: int, last: int) -> None:
        counts["rows_removed"] += last - first + 1

    transcript.reached_top.connect(top)
    transcript.reached_bottom.connect(bottom)
    transcript.model().modelReset.connect(model_reset)
    transcript.model().rowsInserted.connect(rows_inserted)
    transcript.model().rowsRemoved.connect(rows_removed)
    fractions = (0.05, 0.18, 0.55, 0.0, 0.88, 0.82, 0.42, 1.0)
    set_seconds = 0.0
    event_seconds = 0.0
    paints = 0
    if profiler is not None:
        profiler.enable()
    started = time.perf_counter()
    bar = transcript.verticalScrollBar()
    try:
        for step in range(events):
            maximum = bar.maximum()
            before_set = time.perf_counter()
            bar.setValue(round(maximum * fractions[step % len(fractions)]))
            after_set = time.perf_counter()
            set_seconds += after_set - before_set
            if (step + 1) % paint_every == 0:
                before_events = time.perf_counter()
                transcript._flush_scroll()
                qt_app.processEvents()
                frame = transcript.viewport().grab()
                del frame
                qt_app.processEvents()
                event_seconds += time.perf_counter() - before_events
                paints += 1
    finally:
        transcript.reached_top.disconnect(top)
        transcript.reached_bottom.disconnect(bottom)
        transcript.model().modelReset.disconnect(model_reset)
        transcript.model().rowsInserted.disconnect(rows_inserted)
        transcript.model().rowsRemoved.disconnect(rows_removed)
    elapsed = time.perf_counter() - started
    if profiler is not None:
        profiler.disable()
    return {
        "events": events,
        "paints": paints,
        "elapsed_ms": elapsed * 1000,
        "events_per_second": events / elapsed if elapsed else 0.0,
        "set_value_ms": set_seconds * 1000,
        "event_and_paint_ms": event_seconds * 1000,
        **counts,
    }


def _print_memory_sample(sample: MemorySample) -> None:
    print(
        f"memory phase={sample.phase} cycle={sample.cycle} "
        f"python={sample.python_mib:.1f}MiB rss={sample.rss_mib:.1f}MiB "
        f"turns={sample.materialized_turns} entries={sample.materialized_entries} "
        f"rows={sample.presentation_rows}/{sample.history_source_rows} "
        f"docs={sample.documents} heights={sample.height_entries} "
        f"objects={sample.tracked_objects}"
    )


async def _memory_cycles(
    timeline: Timeline,
    transcript: Transcript,
    qt_app: QApplication,
    args: argparse.Namespace,
    randomizer: random.Random,
) -> tuple[list[MemorySample], list[tracemalloc.StatisticDiff]]:
    _flush_qt_deletes(qt_app)
    tracemalloc.start()
    gc.collect()
    before = tracemalloc.take_snapshot()
    samples = [_take_memory_sample("baseline", 0, timeline, transcript)]
    _print_memory_sample(samples[-1])

    for cycle in range(1, args.memory_cycles + 1):
        for _ in range(args.memory_jumps):
            target = randomizer.randrange(timeline.total_turns)
            await _seek_and_display(
                timeline,
                transcript,
                qt_app,
                target,
                paint=not args.skip_paint,
            )
        _flush_qt_deletes(qt_app)
        gc.collect()
        samples.append(_take_memory_sample("random_seek", cycle, timeline, transcript))
        _print_memory_sample(samples[-1])

    for cycle in range(1, args.memory_cycles + 1):
        _rapid_drag(
            transcript,
            qt_app,
            events=args.memory_drag_events,
            paint_every=max(1, args.drag_paint_every),
        )
        _flush_qt_deletes(qt_app)
        gc.collect()
        samples.append(_take_memory_sample("rapid_drag", cycle, timeline, transcript))
        _print_memory_sample(samples[-1])

    after = tracemalloc.take_snapshot()
    differences = after.compare_to(before, "lineno")[:15]
    tracemalloc.stop()
    return samples, differences


def _phase_delta(samples: list[MemorySample], phase: str) -> dict[str, float]:
    phase_samples = [sample for sample in samples if sample.phase == phase]
    if not phase_samples:
        return {}
    first_index = samples.index(phase_samples[0])
    baseline = samples[max(0, first_index - 1)]
    final = phase_samples[-1]
    tail = phase_samples[len(phase_samples) // 2 :]
    return {
        "python_delta_mib": final.python_mib - baseline.python_mib,
        "rss_delta_mib": final.rss_mib - baseline.rss_mib,
        "tail_python_range_mib": max(item.python_mib for item in tail)
        - min(item.python_mib for item in tail),
        "tail_rss_range_mib": max(item.rss_mib for item in tail)
        - min(item.rss_mib for item in tail),
    }


async def main(args: argparse.Namespace) -> None:
    path = args.path.resolve()
    target_bytes = round(max(0.0, args.target_gib) * 1024**3)
    should_generate = args.generate_turns > 0 or target_bytes > 0
    if should_generate and (args.regenerate or not path.exists() or path.stat().st_size == 0):
        started = time.perf_counter()
        generated_turns, generated_bytes = generate_wire(
            path,
            turns=args.generate_turns,
            target_bytes=target_bytes,
            body_bytes=args.body_bytes,
        )
        print(
            f"generated path={path} turns={generated_turns} "
            f"size={generated_bytes / MIB:.1f}MiB "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )
    if not path.exists():
        raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    randomizer = random.Random(args.seed)

    started = time.perf_counter()
    index = await asyncio.to_thread(_scan_wire_history_index, path)
    index_seconds = time.perf_counter() - started
    print(
        f"index turns={index.total_turns} records={sum(index.turn_record_counts)} "
        f"size={index.file_size / MIB:.1f}MiB elapsed={index_seconds:.3f}s"
    )
    if index.total_turns <= 0:
        raise RuntimeError("The wire log contains no TurnBegin records")

    targets = [randomizer.randrange(index.total_turns) for _ in range(args.jumps)]
    parser_profile = output_dir / "history-parser.pstats"
    await asyncio.to_thread(
        _profile_parser,
        index,
        targets,
        parser_profile,
        samples=min(args.parser_profile_jumps, len(targets)),
    )

    timeline = Timeline(index=index)
    started = time.perf_counter()
    await timeline.open()
    open_seconds = time.perf_counter() - started
    print(
        f"open elapsed={open_seconds:.3f}s turns={timeline.materialized_turn_count} "
        f"entries={timeline.materialized_record_count} "
        f"ast_cost={timeline.hydrated_chars() / MIB:.1f}MiB"
    )

    transcript = Transcript()
    transcript.resize(args.width, args.height)
    transcript.show()
    qt_app.processEvents()
    seek_samples = [
        await _seek_and_display(
            timeline,
            transcript,
            qt_app,
            target,
            paint=not args.skip_paint,
        )
        for target in targets
    ]
    seek_summary = _stage_summary(seek_samples)
    _print_stage_summary(seek_summary)
    print(
        f"random_window planned_mean={statistics.fmean(s.planned_turns for s in seek_samples):.1f} "
        f"hydrated_mean={statistics.fmean(s.hydrated_turns for s in seek_samples):.1f} "
        f"entries_mean={statistics.fmean(s.entries for s in seek_samples):.1f}"
    )

    ui_profile = cProfile.Profile()
    for target in targets[: args.ui_profile_jumps]:
        await _seek_and_display(
            timeline,
            transcript,
            qt_app,
            target,
            paint=not args.skip_paint,
            ui_profile=ui_profile,
        )
    ui_profile_path = output_dir / "history-ui.pstats"
    ui_profile.dump_stats(ui_profile_path)

    drag_summary = _rapid_drag(
        transcript,
        qt_app,
        events=args.drag_events,
        paint_every=max(1, args.drag_paint_every),
    )
    print(
        "rapid_drag "
        + " ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in drag_summary.items()
        )
    )
    drag_profile = cProfile.Profile()
    _rapid_drag(
        transcript,
        qt_app,
        events=min(args.drag_events, args.drag_profile_events),
        paint_every=max(1, args.drag_paint_every),
        profiler=drag_profile,
    )
    drag_profile_path = output_dir / "history-scrollbar.pstats"
    drag_profile.dump_stats(drag_profile_path)

    memory_samples, allocation_differences = await _memory_cycles(
        timeline,
        transcript,
        qt_app,
        args,
        randomizer,
    )
    memory_summary = {
        phase: _phase_delta(memory_samples, phase) for phase in ("random_seek", "rapid_drag")
    }
    print(f"memory_summary {memory_summary}")
    print("memory_top_allocations")
    for difference in allocation_differences:
        print(f"  {difference}")

    transcript_ref = weakref.ref(transcript)
    transcript.bodies.invalidate()
    transcript.close()
    transcript.deleteLater()
    del transcript
    _flush_qt_deletes(qt_app)
    gc.collect()
    print(f"after_close transcript_alive={transcript_ref() is not None}")

    results = {
        "input": {
            "path": str(path),
            "turns": index.total_turns,
            "records": sum(index.turn_record_counts),
            "bytes": index.file_size,
            "index_seconds": index_seconds,
            "open_seconds": open_seconds,
        },
        "random_seek": {
            "samples": [asdict(sample) | {"total_ms": sample.total_ms} for sample in seek_samples],
            "summary": seek_summary,
        },
        "rapid_drag": drag_summary,
        "memory": {
            "samples": [asdict(sample) for sample in memory_samples],
            "summary": memory_summary,
            "top_allocations": [str(item) for item in allocation_differences],
        },
    }
    result_path = output_dir / "history-profile.json"
    result_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    print(f"results={result_path}")
    _print_profile(parser_profile, title="wire_to_ast")
    _print_profile(ui_profile_path, title="ast_to_qt")
    _print_profile(drag_profile_path, title="rapid_scrollbar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--generate-turns", type=int, default=0)
    parser.add_argument("--target-gib", type=float, default=0.0)
    parser.add_argument("--body-bytes", type=int, default=4096)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--jumps", type=int, default=80)
    parser.add_argument("--parser-profile-jumps", type=int, default=16)
    parser.add_argument("--ui-profile-jumps", type=int, default=16)
    parser.add_argument("--drag-events", type=int, default=600)
    parser.add_argument("--drag-profile-events", type=int, default=200)
    parser.add_argument("--drag-paint-every", type=int, default=4)
    parser.add_argument("--memory-cycles", type=int, default=6)
    parser.add_argument("--memory-jumps", type=int, default=20)
    parser.add_argument("--memory-drag-events", type=int, default=160)
    parser.add_argument("--seed", type=int, default=246822)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--skip-paint", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".kimix_cache/profiles/large-history"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
