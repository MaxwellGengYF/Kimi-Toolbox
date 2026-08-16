# Kimix Opencode API Server — Reference

Full reference for the FastAPI-based HTTP server in `kimix.server.app` exposing an opencode-compatible REST API with SSE event streaming.

## Application Factory

The server is created via `create_app()` in `kimix/server/app.py`:

```python
from kimix.server.app import create_app
app = create_app()
```

`create_app()` configures:
- `FastAPI(title="Kimix API", version="0.1.0", docs_url="/docs", openapi_url="/openapi.json", redoc_url="/redoc")`
- `CORSMiddleware` with `allow_origins=["*"]`
- Shutdown handler that wakes all SSE queues

## Interface Function Format

Route handlers are defined **inside** `create_app()` using FastAPI decorator methods:

```python
@app.<method>(
    "/path",
    response_model=ResponseModel, # Optional: Pydantic response schema
    tags=["Tag"], # Optional: OpenAPI tag grouping
    summary="Short title", # Optional: endpoint title
    description="Longer explanation",  # Optional: endpoint docs
    responses={404: {"model": ErrorResponse, "description": "Not found"}},  # Optional: extra response docs
    status_code=200, # Optional: explicit success code
)
async def handler_name(param: Type) -> ReturnType:
    ...
```

### Example from app.py

```python
@app.post(
    "/session",
    response_model=SessionResponse,
    tags=["Session"],
    summary="Create session",
    description="Create a new chat session. Returns the session metadata.",
    status_code=200,
)
async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
    info = await session_manager.create_session(title=body.title)
    return info.to_dict()
```

### Handler style rules

- All route handlers are **async**.
- Path parameters use `{name}` in the route string and are declared as function arguments (`sessionID: str`).
- Query parameters use `Optional[int] = Query(default=None, description="...")`.
- Request bodies use Pydantic models (`body: CreateSessionRequest`).
- Errors raise `HTTPException(status_code=..., detail="...")`.

## Builtin Endpoints (routes starting with `/`)

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

### SSE `/event` format

OpenCode protocol: **no** SSE `event:` field. All events are plain `data: {json}\n\n` lines.

Example initial event:
```
data: {"type": "server.connected", "properties": {}}\n\n
```
Heartbeat comment (no `event:` field):
```
: heartbeat\n\n
```

## app.post Usage

`app.post` is used for state-changing operations. Common patterns:

### 1. Create resource (returns 200 + body)

```python
@app.post(
    "/session",
    response_model=SessionResponse,
    summary="Create session",
)
async def create_session(body: CreateSessionRequest) -> Dict[str, Any]:
    info = await session_manager.create_session(title=body.title)
    return info.to_dict()
```

### 2. Fire-and-forget (returns 204, no body)

```python
@app.post(
    "/session/{sessionID}/prompt_async",
    status_code=204,
    tags=["Message"],
    summary="Send message (async)",
    description="Send a prompt fire-and-forget style. Returns 204 immediately. Response events are streamed via SSE /event.",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
    },
)
async def send_prompt_async(sessionID: str, body: PromptInput) -> Response:
    text_parts = [p.text for p in body.parts if p.type == "text" and p.text]
    text = "\n".join(text_parts)
    if not text:
        raise HTTPException(status_code=400, detail="No text content in parts")
    try:
        await session_manager.prompt_async(sessionID, text, agent=body.agent)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {sessionID}")
    return Response(status_code=204)
```

### 3. Action endpoint (returns 200 + empty body)

```python
@app.post(
    "/session/{sessionID}/abort",
    summary="Abort session",
    description="Abort the current running prompt in a session.",
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
)
async def abort_session(sessionID: str) -> Response:
    try:
        session_manager.abort_session(sessionID)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {sessionID}")
    return Response(status_code=200)
```

### 4. Nested path parameter

```python
@app.post(
    "/session/{sessionID}/permissions/{permissionID}",
    summary="Grant permission",
    description="Grant a pending permission request.",
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
)
async def grant_permission(sessionID: str, permissionID: str) -> Response:
    logger.info("Permission granted: session=%s, permission=%s", sessionID, permissionID)
    return Response(status_code=200)
```

## Request/Response Models

```python
from pydantic import BaseModel, Field

class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="Session title")

class PromptPart(BaseModel):
    type: str = Field("text", description="Part type: text")
    text: str = Field("", description="Text content")

class PromptInput(BaseModel):
    parts: List[PromptPart] = Field(default_factory=list, description="Message parts")
    agent: Optional[str] = Field(None, description="Agent name to use")
    model: Optional[str] = Field(None, description="Model name to use")
```

## Common Imports

```python
from kimix.server.app import create_app
from kimix.server.bus import bus, BusEvent
from kimix.server.session_manager import session_manager
```

## Starting the Server

```python
from kimix.server.app import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="127.0.0.1", port=4096, log_level="info")
```

Or via CLI:
```bash
kimix serve --host 127.0.0.1 --port 4096
```
