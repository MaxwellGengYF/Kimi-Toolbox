---
name: serve
description: Guide for the Kimix HTTP serve system — FastAPI backend + TypeScript/Vite frontend. Use when adding new backend endpoints, modifying the session manager, changing the frontend API client, or understanding how data flows from backend to UI. Covers dummy_app.py, dummy_session_manager.py, sse_cli.py, and src/app/.
---

# Kimix Serve — Backend + Frontend

FastAPI backend (port 4096, REST + SSE) + vanilla TypeScript/Vite frontend (port 5173). `dummy_app.py` + `DummySessionManager` for stub prompts; `app.py` + `SessionManager` for live SDK sessions.

**Start**:
```bash
uv run kimix gui                         # dummy backend + Vite dev
uv run kimix gui --build                 # build frontend first
uv run kimix gui --port 8080 --fe-port 3000
uv run kimix gui --no-fe                 # backend only
uv run scripts/run_app.py                # wrapper → `kimix gui`
uv run kimix ssecli                      # terminal client for SSE debugging
```

**Backend entry points**: `dummy_app.py` (default), `app.py` (real), `dummy_session_manager.py` (stub), `session_manager.py` (real), `serve.py` (CLI), `client.py` (Python async client).

**Session manager interface** (both managers): `create_session`, `list_sessions`, `get_session`, `delete_session`, `get_messages`, `prompt_async` (204), `abort_session`, `clear_session`, `compact_session`, `get_session_status`, `export_session`, `get_session_context`.

**Wire formats**: Dummy = `{"type": "Text", "text": ..., "time": ...}` (Enum.name, no wrapper); Opencode = `{"info": {...}, "parts": [{"type": "text"|"tool"|"reasoning"|"step-start"|"step-finish", ...}]}`.

**Adding a feature**: session manager method → HTTP route in `dummy_app.py`/`app.py` → `KimixClient` method → UI wiring in `main.ts` → optional new `MessageType` (base.py, types.ts, renderer.ts, styles.css).

**Debugging**: backend logs to stdout; frontend DevTools Console; API docs at `http://127.0.0.1:4096/docs`; `uv run kimix ssecli` for terminal SSE testing.

Full architecture, route patterns, code examples: read `references/architecture.md`.
