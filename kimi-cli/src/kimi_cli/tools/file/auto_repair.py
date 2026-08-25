"""Edit auto-repair for parse regressions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from rapidfuzz.distance import Levenshtein

from kimi_cli.session import Session
from kimi_cli.tools.display import DisplayBlock
from kimi_cli.tools.file.parse_check import source_parses
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.logging import logger

CONTEXT_LINES = 6
MAX_REGION_LINES = 150
MAX_PAIR_SEARCH_HUNKS = 24
MAX_ATTEMPTS = 2
COMPLETION_MAX_TOKENS = 4096
REPAIR_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class EditHunk:
    """One changed line run in pre-image (a) / post-image (b) coordinates."""

    a_start: int
    a_end: int
    b_start: int
    b_end: int


@dataclass(frozen=True)
class RepairRegion:
    """Localized broken region plus a parseable pre-image reference."""

    b_start: int
    b_end: int
    broken_text: str
    reference_text: str
    language: str


@dataclass(frozen=True)
class RegionRepair:
    """A successful region repair: the full repaired file content."""

    content: str
    region: RepairRegion
    attempts: int


@dataclass(frozen=True)
class EditAutoRepairOutcome:
    """A committed auto-repair result surfaced to the user."""

    diff: str
    diff_blocks: list[DisplayBlock]
    model: str
    attempts: int


CompleteFn = Callable[[str], Awaitable[str]]


def _content_lines(text: str) -> list[str]:
    """Split text into lines, preserving trailing empty line semantics."""
    return text.split("\n")


def build_hunks(prev: str, next: str) -> tuple[list[EditHunk], list[str], list[str]]:
    """Build replace hunks from two texts using rapidfuzz opcodes."""
    a = _content_lines(prev)
    b = _content_lines(next)
    opcodes = Levenshtein.opcodes(a, b)
    hunks: list[EditHunk] = []

    for op in opcodes:
        tag = op.tag
        i1, i2 = op.src_start, op.src_end
        j1, j2 = op.dest_start, op.dest_end
        if tag == "equal":
            continue
        if tag in ("replace", "delete", "insert"):
            del_lines = i2 - i1 if tag in ("replace", "delete") else 0
            add_lines = j2 - j1 if tag in ("replace", "insert") else 0
            last = hunks[-1] if hunks else None
            if last is not None and last.a_end == i1 and last.b_end == j1:
                last = EditHunk(
                    last.a_start,
                    last.a_end + del_lines,
                    last.b_start,
                    last.b_end + add_lines,
                )
                hunks[-1] = last
            else:
                hunks.append(EditHunk(i1, i1 + del_lines, j1, j1 + add_lines))

    return hunks, a, b


def revert_hunks(a: list[str], b: list[str], hunks: list[EditHunk], indices: list[int]) -> str:
    """Return the post-image with the selected hunks swapped to pre-image lines."""
    sorted_idx = sorted(indices)
    out: list[str] = []
    bi = 0
    for i in sorted_idx:
        h = hunks[i]
        out.extend(b[bi : h.b_start])
        out.extend(a[h.a_start : h.a_end])
        bi = h.b_end
    out.extend(b[bi:])
    return "\n".join(out)


def isolate_culprit_hunks(path: str, a: list[str], b: list[str], hunks: list[EditHunk]) -> list[int] | None:
    """Find the smallest hunk set whose reversion restores the parse."""
    n = len(hunks)
    if n == 0:
        return None

    for i in range(n):
        if source_parses(revert_hunks(a, b, hunks, [i]), path):
            return [i]

    if n <= MAX_PAIR_SEARCH_HUNKS:
        for i in range(n):
            for j in range(i + 1, n):
                if source_parses(revert_hunks(a, b, hunks, [i, j]), path):
                    return [i, j]

    keep = set(range(n))
    for i in range(n):
        trial = keep.copy()
        trial.discard(i)
        if source_parses(revert_hunks(a, b, hunks, list(trial)), path):
            keep.discard(i)

    if not keep:
        return None
    if not source_parses(revert_hunks(a, b, hunks, list(keep)), path):
        return None
    return list(keep)


def compute_repair_region(path: str, prev: str, next: str) -> RepairRegion | None:
    """Localize the parse breakage to a bounded region."""
    hunks, a, b = build_hunks(prev, next)
    culprits = isolate_culprit_hunks(path, a, b, hunks)
    if culprits is None:
        return None

    hs = sorted([hunks[i] for i in culprits], key=lambda h: h.b_start)
    b_start = max(0, hs[0].b_start - CONTEXT_LINES)
    b_end = min(len(b), hs[-1].b_end + CONTEXT_LINES)
    if b_end - b_start > MAX_REGION_LINES:
        return None

    ref: list[str] = []
    bi = b_start
    for h in hs:
        ref.extend(b[bi : h.b_start])
        ref.extend(a[h.a_start : h.a_end])
        bi = h.b_end
    ref.extend(b[bi : b_end])

    language = _language_from_path(path)
    return RepairRegion(
        b_start=b_start,
        b_end=b_end,
        broken_text="\n".join(b[b_start:b_end]),
        reference_text="\n".join(ref),
        language=language,
    )


def _language_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mapping = {
        ".py": "python",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }
    return mapping.get(suffix, "source")


def splice_region(b: list[str], region: RepairRegion, text: str) -> str:
    """Splice `text` into the post-image at the repair region."""
    lines = text.split("\n")
    return "\n".join(b[: region.b_start] + lines + b[region.b_end :])


def realign_to_source(src_lines: list[str], candidate: str) -> str:
    """Re-indent candidate by trimmed-line alignment against source lines."""
    out = candidate.split("\n")
    opcodes = Levenshtein.opcodes(
        [l.strip() for l in src_lines],
        [l.strip() for l in out],
    )
    merged: list[str] = []
    si = 0
    oi = 0
    for op in opcodes:
        tag = op.tag
        i1, i2 = op.src_start, op.src_end
        j1, j2 = op.dest_start, op.dest_end
        if tag == "equal":
            for k in range(i2 - i1):
                merged.append(src_lines[si + k])
            si += i2 - i1
            oi += j2 - j1
        elif tag == "delete":
            si += i2 - i1
        elif tag in ("insert", "replace"):
            for k in range(j2 - j1):
                merged.append(out[oi + k])
            oi += j2 - j1
    return "\n".join(merged)


def strip_code_fence(text: str) -> str:
    """Strip one wrapping markdown code fence, if present."""
    import regex as re

    trimmed = text.strip()
    match = re.match(r"^```[^\n]*\n([\s\S]*?)\n?```$", trimmed)
    if match:
        return match.group(1)
    return trimmed


def normalize_for_revert_check(text: str) -> str:
    import regex as re

    return re.sub(r"\s+", " ", text).strip()


def _build_prompt(region: RepairRegion, previous_attempt: str | None) -> str:
    before = region.reference_text
    after = region.broken_text
    prompt = (
        f"An automated edit just modified a region of a {region.language} file and the file no longer parses. "
        "The BEFORE region parsed; the AFTER region contains the syntax error.\n\n"
        f"BEFORE (valid {region.language}):\n```\n{before}\n```\n\n"
        f"AFTER (broken):\n```\n{after}\n```\n\n"
        "Task: output the corrected AFTER region. Keep the intended change from BEFORE to AFTER; "
        "fix ONLY the syntax error (e.g. stray/missing braces, duplicated or truncated lines). "
        "Do not revert the intended change. Output only the corrected code, no commentary, no code fence."
    )
    if previous_attempt:
        prompt += (
            "\n\nA previous attempt produced the following, and the file STILL did not parse after splicing it "
            "in place of the AFTER region. Produce a better correction. Reproduce the surrounding context lines "
            "of AFTER exactly (including leading whitespace); the output replaces the AFTER region line-for-line.\n\n"
            f"PREVIOUS ATTEMPT (rejected):\n```\n{previous_attempt}\n```"
        )
    return prompt


async def repair_parse_regression(
    path: str,
    prev: str,
    next: str,
    complete: CompleteFn | None = None,
) -> RegionRepair | None:
    """Repair a parse regression deterministically and optionally via a model."""
    region = compute_repair_region(path, prev, next)
    if region is None:
        return None

    b = _content_lines(next)
    normalized_reference = normalize_for_revert_check(region.reference_text)

    # Deterministic candidates first.
    candidates = _deterministic_candidates(b, region)
    for text in candidates:
        repair = _try_candidate(path, b, region, text, normalized_reference, allow_reference=True)
        if repair is not None:
            return repair

    if complete is None:
        return None

    previous_attempt: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prompt = _build_prompt(region, previous_attempt)
            candidate = strip_code_fence(await asyncio.wait_for(complete(prompt), timeout=REPAIR_TIMEOUT_SECONDS))
            previous_attempt = candidate
        except Exception:
            return None

        for text in _model_variants(b, region, candidate):
            repair = _try_candidate(path, b, region, text, normalized_reference, allow_reference=False)
            if repair is not None:
                return repair

    return None


def _deterministic_candidates(b: list[str], region: RepairRegion) -> list[str]:
    """Generate deterministic repair candidates."""
    broken_region = "\n".join(b[region.b_start : region.b_end])
    candidates = [
        realign_to_source(b[region.b_start : region.b_end], region.reference_text),
        realign_to_source(region.reference_text.split("\n"), broken_region),
        region.reference_text,
        broken_region.strip(),
    ]
    return list(dict.fromkeys(candidates))


def _model_variants(b: list[str], region: RepairRegion, candidate: str) -> list[str]:
    """Generate realigned variants of a model candidate."""
    return [
        realign_to_source(b[region.b_start : region.b_end], candidate),
        realign_to_source(region.reference_text.split("\n"), candidate),
        candidate,
    ]


def _try_candidate(
    path: str,
    b: list[str],
    region: RepairRegion,
    text: str,
    normalized_reference: str,
    allow_reference: bool = False,
) -> RegionRepair | None:
    normalized_candidate = normalize_for_revert_check(text)
    if normalized_candidate == normalized_reference:
        # If the edit broke only whitespace/indentation, the reference is a
        # legitimate repair, not a semantic revert.
        broken_normalized = normalize_for_revert_check("\n".join(b[region.b_start : region.b_end]))
        if not (allow_reference and broken_normalized == normalized_reference):
            return None
    content = splice_region(b, region, text)
    if source_parses(content, path):
        return RegionRepair(content=content, region=region, attempts=0)
    return None


async def attempt_edit_auto_repair(
    session: Session,
    path: str,
    prev: str,
    next: str,
    *,
    complete: CompleteFn | None = None,
    enabled_for_write: bool = False,
) -> EditAutoRepairOutcome | None:
    """Attempt to auto-repair a committed parse regression.

    Returns an outcome if a repair was applied, otherwise None.
    """
    config = session.custom_config.get("config_json", {}).get("edit", {})
    repair_cfg = config.get("autoRepair", {})
    if not repair_cfg.get("enabled", True) and not enabled_for_write:
        return None
    if enabled_for_write and not repair_cfg.get("enabledForWrite", False):
        # Still allow recording; but do not attempt repair.
        return None

    # Re-read from disk; if it parses now, nothing to do.
    try:
        from kaos.path import KaosPath

        current = await KaosPath(path).read_text(encoding="utf-8")
    except Exception:
        return None
    if source_parses(current, path):
        return None

    # If no completer was injected, try to build one from the session provider.
    if complete is None:
        complete = await _resolve_completer(session)

    repair = await repair_parse_regression(path, prev, current, complete)
    if repair is None:
        return None

    try:
        await KaosPath(path).write_text(repair.content, encoding="utf-8")
    except Exception:
        return None

    model = "deterministic"
    if complete is not None:
        # Best-effort label; actual completer may be model-based.
        model = repair_cfg.get("model", "model")

    diff_blocks = await build_diff_blocks(path, current, repair.content)
    diff_text = "\n".join(str(b) for b in diff_blocks)
    return EditAutoRepairOutcome(
        diff=diff_text,
        diff_blocks=diff_blocks,
        model=model,
        attempts=repair.attempts,
    )


async def _resolve_completer(session: Session) -> CompleteFn | None:
    """Try to resolve a small-model completion function from session config."""
    try:
        from kosong.generate import generate

        provider = session.custom_config.get("chat_provider")
        if not provider:
            return None

        async def _complete(prompt: str) -> str:
            response = await generate(
                chat_provider=provider,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=COMPLETION_MAX_TOKENS,
                disable_reasoning=True,
            )
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            return content

        return _complete
    except Exception as e:
        logger.debug("failed to resolve auto-repair completer", error=str(e))
        return None
