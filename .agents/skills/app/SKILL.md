---
name: app
description: Guide for the Kimix web frontend (src/app/) — vanilla TypeScript + Vite, SSE-based chat UI mirroring sse_cli.py. Use when modifying the frontend, adding new UI features, changing the API client, or understanding the message rendering pipeline.
---

# Kimix Web Frontend (`src/app/`)

Vanilla TypeScript + Vite chat UI mirroring `sse_cli.py`. Zero runtime deps; connects to the backend on port 4096 with poll-based SSE message streaming.

**Key files**: `src/main.ts` (glue/DOM/poll loop), `src/client.ts` (HTTP + SSE), `src/types.ts` (wire-format parsing), `src/renderer.ts` (message → HTML), `src/styles.css` (Catppuccin Mocha).

**Commands**:
```bash
cd src/app
npm run dev        # Vite dev server on port 5173
npm run build      # TypeScript check + production build → dist/
npm run preview    # Preview production build
```
Monorepo runner: `uv run kimix gui` (backend + Vite dev), `--build`, `--fe-port 3000`, `--no-fe` (backend only); `uv run scripts/run_app.py` delegates to `kimix gui`.

**Adding a UI feature**: backend endpoint (serve skill) → `KimixClient` method → `index.html` elements → wire in `main.ts` `handleCommand()` → optional new MessagePartType across `types.ts`/`renderer.ts`/`styles.css`.

**Tests**: `cd src/app && npx tsx src/parse_test.ts` (11 parse tests).

Full architecture, data flow, API client table, and styling details: read `references/architecture.md`.
