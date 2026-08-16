---
name: skill-creator
description: Guide for creating effective skills. Use when users want to create or update a skill that extends agent's capabilities.
---

# Skill Creator

Guidance for creating modular skill packages that extend Kimi's capabilities.

## Core Principles

- **Concise is Key** — context is limited; only add info Kimi doesn't already know; prefer concise examples over verbose explanations.
- **Degrees of Freedom** — match specificity to task fragility: high freedom (text), medium (pseudocode/scripts with params), low (specific scripts needing consistency).
- **Anatomy** — `SKILL.md` (required: YAML frontmatter `name`+`description`, optional `type`; body under 500 lines) + optional resources: `scripts/` (executable code), `references/` (large docs, loaded on demand — keep >10k-word files here), `assets/` (templates/images). Do NOT include README.md/CHANGELOG.md.

## Progressive Disclosure

1. **Metadata** (name + description) — always in context
2. **SKILL.md body** — when skill triggers
3. **Bundled resources** — as needed

Keep SKILL.md lean; link reference files with clear guidance on when to read them; avoid deeply nested references.

## Locations

Priority (most specific wins): `--skills-dir` → project (`.kimi`/`.claude`/`.codex`/`.agents` skills) → user (`~/.config/agents/skills`, `~/.agents/skills`, `~/.kimi/skills`, ...) → built-in. Brand-specific dirs (`.kimi`, `.claude`, `.codex`) beat generic (`.agents`, `.config/agents`).

## Supported Forms

- **Subdirectory (canonical)**: `<skills-root>/<skill-name>/SKILL.md` — use for bundled resources.
- **Flat**: `<skills-root>/<skill-name>.md` — single-file skills; stem = name when `name` omitted.

## Creation Process

1. Understand → 2. Plan (identify reusable resources) → 3. Initialize (dir + SKILL.md + resource folders) → 4. Edit → 5. Validate (frontmatter, naming, structure, discovery) → 6. Iterate.

**Naming**: lowercase letters/digits/hyphens, <64 chars, verb-led (e.g. `gh-address-comments`); folder name matches skill name.

**Frontmatter**: `name`, `description` (what + when to use); `type: flow` adds a `mermaid`/`d2` block.

**Body**: imperative form; multi-step workflows with decision points; output formats/quality standards; links to reference files.

## Testing & Validation

Test scripts (representative sample for many similar ones); verify discovery with `kimi --skills-dir <parent-root>`. Before complete: frontmatter starts/ends with `---`; `name` matches dir/stem; `description` present (what + when); `type` is `standard` or `flow` (omit for standard); flow skills have a valid `mermaid`/`d2` block; no README.md/CHANGELOG.md; resources referenced clearly from SKILL.md.
