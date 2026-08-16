Start a subagent for focused tasks. Create new or resume by `agent_id`.

**Usage**
- Keep `description` short (3-5 words).
- `subagent_type` (default: `coder`), `model` to override; `resume` continues existing instances.
- Foreground by default; `run_in_background=true` only for independent tasks.
- Be explicit: code or research only.

**Explore Agent** — Preferred for codebase research (read-only). Use when you need >3 searches, module understanding, or concurrent investigations. Specify thoroughness: "quick" (find file), "medium" (understand module), "thorough" (architecture analysis).
