#!/usr/bin/env python3
"""Prompt-quality lint (plan.md Part 4 §4.3, P4).

Checks the wire-visible builtin tool descriptions against the Q1-Q5 rules
from the tool-prompt refactor plan.  A non-zero exit code means at least one
description violates a rule; ``tools/syntax_check_all.py`` fails the build on
that condition.

Rules (see plan.md "make it concise and short but keep all important
information"):

- Q1 — No hoisted-convention duplication: generic conventions (head+tail
  fold, ``rtk``, output dedup, ``cwd``/``workdir`` aliases, timeout ranges)
  live once in the ``# Tool Conventions`` block of the default system prompt
  and must not be re-sent inside every tool description.
- Q2 — No schema-redundant constraint text: ranges/defaults already
  serialized in the params JSON schema (e.g. ``(1-900)``) must not be
  repeated in prose.
- Q3 — No boilerplate/filler: filler phrases ("This tool is a tool ...",
  "you typically want to use ...", "Make sure you follow ...") add tokens
  without information.
- Q4 — No duplicate sentences inside one description.
- Q5 — No empty or truncated descriptions: every tool keeps a non-empty
  description that does not end mid-sentence.

Usage:
    uv run python tools/check_prompt_quality.py
    uv run python tools/check_prompt_quality.py --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.measure_tool_prompt import full_sources  # noqa: E402


# ── Q1: generic conventions that belong once in system.md ────────────────
CONVENTION_FRAGMENTS = (
    "head+tail fold",
    "Set `max_lines=None`",
    "`rtk <process>",
    "rtk <process> <arguments",
    "deduplicated automatically",
    "`cwd`/`workdir` sets",
    "Accepts `cmd` or `command` parameter",
    "Accepts `command` or `cmd` parameter",
    "Timeout in seconds (1-900)",
    "Timeout in seconds (1-300)",
)

# ── Q2: constraint text already present in the JSON schema ───────────────
SCHEMA_REDUNDANT_PATTERNS = (
    re.compile(r"\(\s*1\s*-\s*900\s*\)"),
    re.compile(r"\(\s*1\s*-\s*300\s*\)"),
    re.compile(r"\(max(?:imum)?\s+500\s*\)"),
    re.compile(r"\(1-500\s*\)"),
)

# ── Q3: boilerplate / filler ─────────────────────────────────────────────
BOILERPLATE_FRAGMENTS = (
    "This tool is a tool",
    "you typically want to use",
    "Make sure you follow",
    "Please note",
    "Note that ",
    "In order to ",
    "It is important to note",
)


def _normalize_sentences(text: str) -> list[str]:
    """Split prose into normalized sentences for duplicate detection."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for sentence in sentences:
        norm = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if len(norm) >= 20:
            out.append(norm)
    return out


def _check_description(name: str, description: str) -> list[str]:
    violations: list[str] = []
    text = description or ""

    # Q1
    for fragment in CONVENTION_FRAGMENTS:
        if fragment in text:
            violations.append(
                f"Q1: duplicated generic convention {fragment!r} "
                "(lives once in system.md '# Tool Conventions')"
            )

    # Q2
    for pattern in SCHEMA_REDUNDANT_PATTERNS:
        if pattern.search(text):
            violations.append(
                f"Q2: constraint already serialized in the JSON schema: "
                f"{pattern.pattern!r}"
            )

    # Q3
    for fragment in BOILERPLATE_FRAGMENTS:
        if fragment.lower() in text.lower():
            violations.append(f"Q3: boilerplate/filler {fragment!r}")

    # Q4
    sentences = _normalize_sentences(text)
    seen: set[str] = set()
    for sentence in sentences:
        if sentence in seen:
            violations.append(
                f"Q4: duplicate sentence {sentence[:60]!r} appears twice"
            )
        seen.add(sentence)

    # Q5 (description only; template conditionals in .md sources are rendered
    # by their owning tools before reaching the model, so ignore those lines)
    stripped = text.strip()
    if not stripped:
        violations.append("Q5: empty description")
    else:
        last_line = stripped.splitlines()[-1].strip()
        if last_line.startswith("{%"):
            body = stripped.splitlines()
            last_line = next(
                (line.strip() for line in reversed(body) if not line.strip().startswith("{%")),
                "",
            )
        if re.search(r"(\.\.\.|…|\s+[-—]$|\s+(and|or|the|to)$)$", last_line):
            violations.append(f"Q5: description looks truncated: {last_line!r}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check builtin tool descriptions against Q1-Q5 prompt-quality rules."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print violations (no table header)",
    )
    args = parser.parse_args()

    sources = full_sources()
    failed = False
    if not args.quiet:
        print(f"{'source':<28}{'desc':>8}  status")
    for name in sorted(sources):
        item = sources[name]
        violations = _check_description(name, item["description"])
        desc_len = len(item["description"])
        if violations:
            failed = True
            if not args.quiet:
                print(f"{name:<28}{desc_len:>8}  FAIL")
            for violation in violations:
                print(f"  [{name}] {violation}")
        elif not args.quiet:
            print(f"{name:<28}{desc_len:>8}  ok")

    if failed:
        print("\nPrompt-quality check FAILED: tool descriptions violate Q1-Q5 rules.")
        return 1
    if not args.quiet:
        print("\nPrompt-quality check OK: all tool descriptions are concise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
