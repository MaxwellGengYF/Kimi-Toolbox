---
name: opcode_api
description: Guide for Kimix opencode-style HTTP server (FastAPI + SSE), including route definitions, builtin endpoints, and app.post usage.
---

# Kimix Opencode API Server

FastAPI HTTP server in `kimix.server.app` exposing an opencode-compatible REST API with SSE event streaming. Create via `create_app()`; run with `kimix serve --host 127.0.0.1 --port 4096`.

**Handler style**: async handlers; path params `{name}` as function args; query params `Optional[int] = Query(...)`; Pydantic request bodies; errors via `HTTPException`.

**Builtin endpoints**:

| Method | Path | Summary |
|--------|------|---------|
| GET | `/global/health` | Health check |
| GET | `/event` | SSE event stream (global) |
| POST | `/session` | Create session |
| GET | `/session` | List sessions |
| GET | `/session/status` | Get all session statuses |
| GET | `/session/{sessionID}` | Get session info |
| DELETE | `/session/{sessionID}` | Delete session |
| GET | `/session/{sessionID}/message` | Get messages |
| POST | `/session/{sessionID}/prompt_async` | Send message (fire-and-forget, 204) |
| POST | `/session/{sessionID}/abort` | Abort session |
| POST | `/session/{sessionID}/permissions/{permissionID}` | Grant permission |

**SSE `/event`**: OpenCode protocol — no `event:` field, plain `data: {json}\n\n` lines; heartbeat is `: heartbeat\n\n`.

**app.post patterns**: create resource (200 + body), fire-and-forget (204, no body), action (200 + empty body), nested path params.

Full code examples and models: read `references/guide.md`.
