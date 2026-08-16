---
name: doc
description: Guide for navigating and maintaining Kimix project documentation. Use when: (1) user asks about the docs structure, (2) user wants to add, update, or append content to project documents.
---

# Project Documentation Guide

Reference for the Kimix documentation layout and how to modify or extend it safely.

## Document Structure

```
<repo-root>/
├── AGENTS.md                     # Agent rules (always loaded)
├── README.md                     # English project overview and doc index
├── README_zh.md                  # Chinese project overview and doc index
├── ChangeLog.md                  # Release changelog (kept minimal)
└── docs/
    ├── tutorials/
    │   ├── 1_quick_start.md      # Chinese quick start
    │   ├── 1_quick_start_en.md   # English quick start
    │   ├── 2_long_task.md        # Chinese /plan workflow
    │   ├── 2_long_task_en.md     # English /plan workflow
    │   ├── 3_builtin_tools.md    # Chinese built-in tools guide
    │   ├── 3_builtin_tools_en.md # English built-in tools guide
    │   ├── 4_skills.md           # Chinese skill authoring
    │   ├── 4_skills_en.md        # English skill authoring
    │   ├── 5_server.md           # Chinese HTTP server tutorial
    │   └── 5_server_en.md        # English HTTP server tutorial
    ├── server/
    │   └── opencode_style_sse.md # OpenCode SSE protocol details
    └── *.json                    # Provider config samples
        ├── kimi.json
        ├── anthropic.json
        ├── openai_legacy.json
        ├── openai_responses.json
        ├── google_genai.json
        ├── gemini.json
        ├── vertexai.json
        ├── deepseek.json
        ├── deepseek_responses.json
        └── multiple_models.json
```

> The `*.json` files are copy-paste samples for a subset of providers. The authoritative list of supported `type` values lives in `kimi-cli/src/kimi_cli/llm.py` (`ProviderType`) and is summarized in `docs/tutorials/1_quick_start.md` / `1_quick_start_en.md` (Supported Providers).

| Area | What lives there |
|------|------------------|
| Root markdown | Project rules, high-level overview, bilingual indexes |
| `docs/tutorials/` | User-facing how-to guides in Chinese + English pairs |
| `docs/server/` | Protocol and SSE reference docs |
| `docs/*.json` | Copy-paste model/provider config templates |

## Appending New Content

### Append to an existing document

Use `write` with `mode: append`:

```python
write(
    path="ChangeLog.md",
    content="\n## 0.2.0\n\n- Added feature X.\n",
    mode="append"
)
```

Guidelines:
- Append to the **end** of changelog/release notes.
- For append-only sections, keep a leading newline so the new block starts cleanly.
- If content exceeds 100 lines, split into multiple `append` calls.

### Insert at a specific location

Use `edit` when the new content belongs in the middle of a file (e.g., adding a row to an index table):

```python
edit(
    path="README.md",
    edit={
        "old": "| [`docs/tutorials/5_server_en.md`](docs/tutorials/5_server_en.md) | HTTP server tutorial. |\n",
        "new": "| [`docs/tutorials/5_server_en.md`](docs/tutorials/5_server_en.md) | HTTP server tutorial. |\n| [`docs/tutorials/6_new_topic_en.md`](docs/tutorials/6_new_topic_en.md) | New topic guide. |\n"
    }
)
```

### Add a new tutorial document

1. Pick the next number and create both language files:
   - `docs/tutorials/<N>_<topic>.md`
   - `docs/tutorials/<N>_<topic>_en.md`
2. Add a matching row to both:
   - `README.md` → English tutorials table
   - `README_zh.md` → Chinese tutorials table
3. Keep the doc under 500 lines; move large schemas to `references/` if needed.

### Add a new config sample

1. Create `docs/<provider>.json` following the fields in `docs/kimi.json`.
2. Add a short description row to the config reference table in `README.md` and `README_zh.md`.

## Editing Rules

1. **Read first**: use `read` or `grep` to confirm the target location before editing.
2. **Prefer `edit`**: use it for index tables, single sections, and small insertions.
3. **Use `write` append mode** only for end-of-file additions (changelogs, new sections).
4. **Keep both languages in sync**: when one language file changes, update its counterpart if it exists.
5. **Do not create README/CHANGELOG inside skills**: applies only to skill folders, not project docs.
