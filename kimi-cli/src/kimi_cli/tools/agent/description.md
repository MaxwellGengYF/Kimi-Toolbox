Start a subagent for focused tasks. Create new or resume by `agent_id`.

**Usage**
- Keep `description` short (3-5 words).
- Use `subagent_type` (default: `coder`), `model` to override.
- Use `resume` to continue existing instances with context.
- Run in foreground by default; `run_in_background=true` only for independent tasks.
- Be explicit: code or research only.
- Subagent results are private—summarize for user if needed.

**Explore Agent** — Preferred for codebase research (read-only). Use when you need >3 searches, module understanding, or concurrent investigations. Specify thoroughness: "quick" (find file), "medium" (understand module), "thorough" (architecture analysis).
