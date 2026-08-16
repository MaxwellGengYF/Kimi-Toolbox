# Kimix Serve — Architecture Reference

Full architecture and extension guide for the FastAPI backend + TypeScript/Vite frontend.

## Overview

```
                         HTTP (REST + SSE)
┌─────────────────────┐ ◄═══════════════► ┌──────────────────────┐
│  Backend (FastAPI)  │                   │  Frontend (Vite+TS)  │
│                     │                   │                      │
│  dummy_app.py       │  JSON wire format │  src/client.ts       │
│  ↓ delegates to     │  ───────────────► │  src/types.ts        │
│  DummySessionMgr    │                   │  src/renderer.ts     │
│                     │                   │  src/main.ts         │
│  port 4096          │                   │  port 5173           │
└─────────────────────┘                   └──────────────────────┘
```

**Backend**: `FastAPI` app factory (`dummy_app.py`) with `DummySessionManager` for stub prompts, or `app.py` with `SessionManager` for live SDK sessions.

**Frontend**: Vanilla `TypeScript` + `Vite`. Mirrors `sse_cli.py` logic — connect, create session, send prompts, poll messages, render parts.

**Start**:
```bash
uv run kimix gui                           # dummy backend + Vite dev
uv run kimix gui --build                   # build frontend before starting
uv run kimix gui --port 8080 --fe-port 3000  # custom ports
uv run kimix gui --no-fe                   # backend only (no Node.js/Vite required)
uv run scripts/run_app.py                  # backward-compatible wrapper → `kimix gui`
```

**Debug CLI**: `uv run kimix ssecli` — terminal client (same logic as the web UI). Connects to the running backend and interactively tests SSE streams.

## Backend Architecture

### Entry points

| File | Role |
|------|------|
| `src/kimix/server/dummy_app.py` | FastAPI app factory (default, for `kimix serve`). Uses `DummySessionManager`. |
| `src/kimix/server/app.py` | FastAPI app factory (real). Uses `SessionManager` with live SDK. |
| `src/kimix/server/dummy_session_manager.py` | Stub session manager. Background thread runs real `prompt_async`, pushes output to `queue.Queue`. |
| `src/kimix/server/session_manager.py` | Real session manager. Builds opencode-format messages, emits SSE events via bus. |
| `src/kimix/server/serve.py` | CLI entry for `kimix serve`. Imports `dummy_app.create_app`. |
| `src/kimix/server/client.py` | Python async client (`KimixAsyncClient`). Used by `sse_cli.py`. |

### Route pattern (FastAPI decorators inside `create_app()`)

```python
@app.get("/path", response_model=Model, tags=["Tag"], summary="...")
async def handler(param: str) -> Dict[str, Any]:
    return session_manager.some_method(param)
```

All handlers are async. Path params use `{name}` in the route and become function args. Query params use `Query(default=...)`.

### Session manager interface

Both `DummySessionManager` and `SessionManager` implement the same methods:

| Method | HTTP route | Purpose |
|--------|-----------|---------|
| `create_session(title)` | `POST /session` | Create session, returns `SessionInfo` |
| `list_sessions()` | `GET /session` | List all sessions |
| `get_session(id)` | `GET /session/{id}` | Get session metadata |
| `delete_session(id)` | `DELETE /session/{id}` | Delete session |
| `get_messages(id, limit)` | `GET /session/{id}/message` | Get messages (drains output queue in dummy) |
| `prompt_async(id, text)` | `POST /session/{id}/prompt_async` | Fire-and-forget prompt (returns 204) |
| `abort_session(id)` | `POST /session/{id}/abort` | Cancel running prompt |
| `clear_session(id)` | `GET /session/{id}/clear` | Clear session history |
| `compact_session(id, keep)` | `GET /session/{id}/compact` | Compact/trim history |
| `get_session_status(id)` | `GET /session/{id}/status` | Running/idle status |
| `export_session(id, path)` | `GET /session/{id}/export` | Export to file |
| `get_session_context(id)` | `GET /session/{id}/context` | Get session context usage |

### DummySessionManager internals

```
prompt_async() → input_queue.put(text)
        ↓ (background thread)
_process_prompt() → prompt_async() from kimix.utils.prompt
        ↓
output_function(chunk, MessageType) → output_queue.put(item)
        ↓
get_messages() → drain output_queue → return List[Dict]
```

The `_output_handler` (line 163-177) builds items:
```python
item = {
    "type": msg_type.name,  # "Text" | "Thinking" | "ToolCalling" | "ToolCallingPart" | "ToolResult"
    "text": chunk,
    "time": _now_ts(),
}
# Extra fields: tool_name (for ToolCalling), tool_result (for ToolResult)
```

### Wire formats (two paths)

**Dummy format** (from `DummySessionManager.get_messages`):
```json
{"type": "Text", "text": "Hello", "time": 1234567890.123}
```
`type` uses Python `Enum.name` → capital first letter. No `info` or `parts` wrapper.

**Opencode format** (from `SessionManager.get_messages`):
```json
{
  "info": {"id": "msg_xxx", "role": "assistant", "time": {"created": 123}},
  "parts": [
    {"id": "prt_xxx", "type": "text", "text": "Hello", "time": {"start": 1, "end": 2}},
    {"id": "prt_xxx", "type": "tool", "tool": "read", "callID": "toolu_xxx",
     "state": {"status": "running", "input": {...}}}
  ]
}
```
Part types: `"text"`, `"tool"`, `"reasoning"`, `"step-start"`, `"step-finish"`.

## Frontend Architecture

### Files

| File | Role |
|------|------|
| `src/types.ts` | `MessagePartType` enum, `MessagePart`/`Message`/`Session` interfaces, plus `parseMessagePart()`, `parseMessage()`, `parseSession()` |
| `src/client.ts` | `KimixClient` class — `fetch()`-based HTTP client for all REST endpoints + `EventSource` for SSE |
| `src/renderer.ts` | `renderMessagePart()` — renders a `MessagePart` to an HTML `<span>` element with CSS classes |
| `src/main.ts` | Application glue — DOM wiring, connect/disconnect, poll loop, slash commands, `appendPart()` |
| `src/styles.css` | Catppuccin Mocha theme with `.part-text`, `.part-thinking`, `.part-tool-calling`, etc. |
| `index.html` | Entry HTML — connection panel, command bar, output area, input bar, script import |
| `package.json` | npm config — `npm run dev` (Vite), `npm run build` (tsc + vite) |
| `vite.config.ts` | Vite bundler config (dev port 5173, sourcemaps) |

### Data flow

```
User types "hello" → sendPrompt()
  → client.sendPromptAsync(sessionId, "hello")  // POST /session/{id}/prompt_async → 204
  → startPolling()                               // setInterval(pollMessages, 500)
       ↓
  pollMessages()
    → client.getMessages(sessionId, 50)          // GET /session/{id}/message?limit=50
    → resp.json().map(parseMessage)              // parses both wire formats
    → for each new message part:
        renderMessagePart(part) → HTML element
        appendPart(el)          → append to output area
    → idle detection: after 2s of no new messages, check status → stop polling if idle
```

### Message parsing (types.ts)

`parseMessage()` detects the format:
- `if (!data.info)` → dummy format → uses `DUMMY_TYPE_MAP`
- else → opencode format → iterates `data.parts`, calls `parseMessagePart()` which uses `OPENCODE_TYPE_MAP`

Type maps:
```typescript
DUMMY_TYPE_MAP = { Text→TEXT, Thinking→THINKING, ToolCalling→TOOL_CALLING, ... }
OPENCODE_TYPE_MAP = { tool→TOOL_CALLING, reasoning→THINKING }
```

### Message rendering (renderer.ts)

Each `MessagePartType` maps to a CSS class and rendering behavior:
- `TEXT` / `THINKING` → inline `<span>` with text content
- `TOOL_CALLING` → block with header (`⚡ tool_name`) + detail lines (status, input, output, error)
- `TOOL_CALLING_PART` → inline text
- `TOOL_RESULT` → block with `✓`/`✗` prefix
- `STEP_START` / `STEP_FINISH` → block markers

### Inline text chaining (main.ts, appendPart)

`appendPart()` uses `.has-inline` class on wrapper divs to chain consecutive TEXT parts inline, matching the Python CLI's `print(..., end="")` behavior.

## How to Add a New Backend Feature

### Step 1: Add method to session manager

In `src/kimix/server/dummy_session_manager.py`, add to `DummySessionManager`:

```python
def my_new_action(self, session_id: str, param: str) -> Dict[str, Any]:
    """Perform a new action on a session."""
    with self._lock:
        state = self._session_states.get(session_id)
    if state is None:
        raise KeyError(f"Session not found: {session_id}")
    # ... implement logic ...
    return {"result": "ok", "sessionID": session_id}
```

Also add the same method to `SessionManager` in `session_manager.py` if needed.

### Step 2: Expose HTTP route

In `src/kimix/server/dummy_app.py` (and `app.py` if applicable), inside `create_app()`:

```python
@app.get(
    "/session/{sessionID}/my-action",
    tags=["Options"],
    summary="My new action",
    description="Description of what this action does.",
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
)
async def my_action(
    sessionID: str,
    param: str = Query(default=..., description="Action parameter"),
) -> Dict[str, Any]:
    try:
        return session_manager.my_new_action(sessionID, param)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {sessionID}")
```

**Rules**:
- Use `@app.get` for reads, `@app.post` for mutations
- `response_model=` for OpenAPI docs; `tags=` for grouping
- Path params: `{sessionID}` in route, `sessionID: str` in signature
- Query params: `Query(default=..., description="...")`
- Errors: `raise HTTPException(status_code=..., detail="...")`

### Step 3: Add frontend client method

In `src/app/src/client.ts`, add to `KimixClient`:

```typescript
async myAction(sessionId: string, param: string): Promise<Record<string, unknown>> {
    const params = new URLSearchParams({ param });
    const resp = await fetch(
        `${this.baseUrl}/session/${sessionId}/my-action?${params}`
    );
    if (!resp.ok) throw new Error(`myAction failed: ${resp.status}`);
    return await resp.json();
}
```

**Pattern**: `baseUrl` + path + query params → fetch → check `resp.ok` → return JSON.

### Step 4: Add UI integration

In `src/app/src/main.ts`:

1. Add a command button in `index.html` (e.g. `<button id="btn-myaction" disabled>/myaction</button>`)
2. Wire it in `main.ts`:
```typescript
const btnMyAction = document.getElementById("btn-myaction") as HTMLButtonElement;
btnMyAction.addEventListener("click", () => handleCommand("/myaction"));
```
3. Add to the `switch` in `handleCommand()`:
```typescript
case "myaction": {
    const result = await client.myAction(session.id, "param-value");
    log(`[SSE CLI] MyAction: ${JSON.stringify(result)}`, "info");
    break;
}
```
4. Add to the `setConnected()` disabled-buttons loop.

### Step 5: Add output type (if new message format)

If the backend produces a new `MessageType`, add it to:
1. `kimix/base.py` → `MessageType` enum
2. `src/app/src/types.ts` → `MessagePartType` enum and `DUMMY_TYPE_MAP`
3. `src/app/src/renderer.ts` → `partTypeClass()` and `renderMessagePart()` switch
4. `src/app/src/styles.css` → `.part-<newtype>` styling

## Debugging

- **Backend logs**: `DummySessionManager` prints each request to stdout
- **Frontend console**: Open browser DevTools → Console tab; `fetch` errors and `catch` blocks log there
- **API docs**: `http://127.0.0.1:4096/docs` (Swagger UI) for interactive testing
- **SSE CLI**: `uv run kimix ssecli` connects to the running server for terminal debugging
