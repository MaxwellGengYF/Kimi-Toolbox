# Kimix Web Frontend (`src/app/`) — Architecture Reference

Detailed setup, architecture, and extension guide for the vanilla TypeScript + Vite chat UI.

## Overview

```
┌────────────────────────────────────────────────────────────────────┐
│  src/app/  (Vanilla TypeScript + Vite)                             │
│                                                                    │
│  index.html   ──►  src/main.ts  (app glue, DOM, poll loop)         │
│                    src/client.ts   (HTTP + SSE client)             │
│                    src/types.ts     (wire format parsing)           │
│                    src/renderer.ts  (message → HTML)               │
│                    src/styles.css   (Catppuccin Mocha theme)        │
│                    src/parse_test.ts (unit tests for parsing)       │
└────────────────────────────────────────────────────────────────────┘
```

Zero-dependency (except Vite + TypeScript dev tooling) web UI mirroring the Python `sse_cli.py` terminal client. Connects to the backend on port 4096 with poll-based SSE message streaming.

## Environment Setup

First-time setup from a clean clone:

```bash
cd src/app
npm install     # typescript ^5.7.0 + vite ^6.0.0 (dev only), ~2s
npm run dev     # Vite dev server on port 5173, opens browser
```

Key observations:
- Zero runtime dependencies — only `typescript` and `vite` as dev tooling.
- `npm install` completes in ~2 seconds.
- Vite serves TypeScript directly (no build step needed for development).
- Frontend connects to backend on port 4096 — backend must run separately (see serve skill).

Troubleshooting:
- `npm install` fails → check Node.js >= 18.
- Port 5173 in use → modify `vite.config.ts` or use `--port`.
- Frontend can't connect → ensure backend runs on port 4096, check the connection panel defaults.

## File Map

| File | Role | Lines |
|------|------|-------|
| `index.html` | Entry HTML — connection panel, command bar, output area, input bar | 66 |
| `package.json` | npm config — `dev` / `build` / `preview` scripts, TS+Vite deps | 16 |
| `tsconfig.json` | TS config — ES2022, strict, bundler resolution | 17 |
| `vite.config.ts` | Vite config — dev port 5173, sourcemaps, dist/ output | 14 |
| `src/types.ts` | Types, enums, and wire-format parsers (`parseMessage`, `parseMessagePart`, `parseSession`) | 175 |
| `src/client.ts` | `KimixClient` — `fetch()`-based HTTP client + `EventSource` SSE | 169 |
| `src/renderer.ts` | `renderMessagePart()` — `MessagePart` → HTML `<span>` element | 134 |
| `src/main.ts` | Application glue — DOM wiring, connect/disconnect, poll loop, slash commands | 421 |
| `src/styles.css` | Catppuccin Mocha theme — CSS custom properties, part-type styling | 335 |
| `src/parse_test.ts` | Standalone parse tests (11 tests, both wire formats) | 343 |

## Architecture

### Data Flow

```
User types "hello" → sendPrompt()
  → client.sendPromptAsync(sessionId, "hello")   // POST /session/{id}/prompt_async → 204
  → startPolling()                                // setInterval(pollMessages, 500ms)
       ↓
  pollMessages()
    → client.getMessages(sessionId, 50)           // GET /session/{id}/message?limit=50
    → resp.json().map(parseMessage)               // detects & parses both wire formats
    → for each new message part:
        renderMessagePart(part) → HTML element
        appendPart(el)          → add to output area
    → idle detection: 4 empty polls (2s) → check status → stop polling if idle
```

### State Machine

```
Disconnected ──[Connect]──► Connected ──[Send]──► Streaming ──[Idle]──► Connected
     ▲                         │                      ▲
     └──────[Disconnect]───────┘                      │
                                                      │
                              Error ──────────────────┘
```

Key state variables in `main.ts`:
- `client: KimixClient | null` — HTTP client instance
- `session: Session | null` — current session
- `pollTimer` — `setInterval` handle for message polling
- `seenMessageCount` — cursor for incremental message display
- `emptyPolls` — consecutive polls with no new messages (idle detection)
- `connected` / `debugMode` — UI flags

## Type System (`src/types.ts`)

### MessagePartType Enum

```typescript
enum MessagePartType {
  TEXT = "text",
  THINKING = "thinking",
  TOOL_CALLING = "tool_calling",
  TOOL_CALLING_PART = "tool_calling_part",
  TOOL_RESULT = "tool_result",
  STEP_START = "step-start",
  STEP_FINISH = "step-finish",
  UNKNOWN = "unknown",
}
```

### Core Interfaces

```typescript
interface MessagePart {
  type: MessagePartType;
  text: string | null;
  tool_name: string | null;
  tool_status: string | null;
  tool_state: Record<string, unknown> | null;
  tool_result: string | null;
  call_id: string | null;
  reason: string | null;
  cost: number | null;
  tokens: Record<string, unknown> | null;
  raw_data: Record<string, unknown> | null;
}

interface Message {
  id: string;
  role: string;
  parts: MessagePart[];
  created_at: number | null;
  text_content: string;  // concatenated TEXT parts
}

interface Session {
  id: string;
  title: string | null;
  created_at: number | null;
  updated_at: number | null;
  parent_id: string | null;
}
```

### Wire Format Detection

`parseMessage()` detects the backend format:

**Dummy format** (from `DummySessionManager`): `data.info` absent → `DUMMY_TYPE_MAP`:
```typescript
DUMMY_TYPE_MAP = {
  Text → TEXT, Thinking → THINKING, ToolCalling → TOOL_CALLING,
  ToolCallingPart → TOOL_CALLING_PART, ToolResult → TOOL_RESULT
}
```
```json
{"type": "Text", "text": "Hello", "time": 1234567890.123}
```

**Opencode format** (from `SessionManager`): `data.info` present → iterates `data.parts`, uses `OPENCODE_TYPE_MAP`:
```typescript
OPENCODE_TYPE_MAP = { tool → TOOL_CALLING, reasoning → THINKING }
```
```json
{
  "info": {"id": "msg_001", "role": "assistant", "time": {"created": 123}},
  "parts": [
    {"id": "prt_001", "type": "text", "text": "Hello"},
    {"id": "prt_002", "type": "tool", "tool": "read", "callID": "toolu_001",
     "state": {"status": "running", "input": {"path": "/foo"}}}
  ]
}
```

## API Client (`src/client.ts`)

`KimixClient` wraps all backend REST endpoints. Base URL: `http://{host}:{port}`.

| Method | HTTP | Endpoint |
|--------|------|----------|
| `healthCheck()` | GET | `/global/health` |
| `createSession(title?)` | POST | `/session` |
| `getSession(id)` | GET | `/session/{id}` |
| `deleteSession(id)` | DELETE | `/session/{id}` |
| `listSessions()` | GET | `/session` |
| `getMessages(id, limit?)` | GET | `/session/{id}/message?limit=N` |
| `getSessionStatus(id)` | GET | `/session/{id}/status` |
| `sendPromptAsync(id, text, agent?)` | POST | `/session/{id}/prompt_async` |
| `abortSession(id)` | POST | `/session/{id}/abort` |
| `clearSession(id)` | GET | `/session/{id}/clear` |
| `compactSession(id, keep?)` | GET | `/session/{id}/compact?keep=N` |
| `exportSession(id, outputPath?)` | GET | `/session/{id}/export?output_path=P` |
| `streamEvents(onEvent, onError?)` | SSE | `/event` (EventSource) |

**Pattern**: `fetch(baseUrl + path)` → check `resp.ok` → return JSON or boolean.

## Message Rendering (`src/renderer.ts`)

`renderMessagePart(part: MessagePart): HTMLElement` — each `MessagePartType` → CSS class + rendering:

| Type | CSS class | Rendering |
|------|-----------|-----------|
| `TEXT` | `.part-text` | Inline `<span>` with text content |
| `THINKING` | `.part-thinking` | Inline `<span>` with text (cyan color) |
| `TOOL_CALLING` | `.part-tool-calling` | Block: header (`⚡ tool_name`) + detail lines (status, input, output, error) |
| `TOOL_CALLING_PART` | `.part-tool-calling-part` | Inline text (magenta) |
| `TOOL_RESULT` | `.part-tool-result` | Block: `✓ result` or `✗ error` prefix |
| `STEP_START` | `.part-step-start` | Block: `[STEP START]` (yellow) |
| `STEP_FINISH` | `.part-step-finish` | Block: `[STEP FINISH] reason=...` (yellow) |
| `UNKNOWN` | `.part-unknown` | JSON dump of `raw_data` (gray) |

Utilities: `fmtArg(s, maxLen=120)` — truncate long strings keeping head+tail; `fmtTs(unixT)` — format unix timestamp → `HH:MM:SS`; `partTypeClass(type)` — map enum → CSS class name.

## Application Logic (`src/main.ts`)

### DOM Wiring & Output Helpers

All elements obtained once via `getElementById()` and stored in module-level variables. `setConnected()` enables/disables button/input groups.

- `log(text, cls)` — append `<div>` with optional CSS class (`.info`, `.error`, `.debug`, `.user-input`)
- `logHtml(html, cls)` — same but with innerHTML
- `appendPart(el)` — smart append: consecutive TEXT parts chain inline via `.has-inline` wrapper divs
- `scrollToBottom()` — auto-scroll output area
- `setStatus(text, ok)` — update status indicator with green/red class

### Slash Commands

`handleCommand()` switch. Commands with optional arguments use `:` separator (e.g., `/compact:5`).

| Command | Action |
|---------|--------|
| `/help` | Print command list |
| `/new` | Create new session, clear output |
| `/abort` | Cancel running prompt |
| `/status` | Show session status JSON |
| `/sessions` | List all sessions |
| `/messages` | Show recent messages |
| `/clear` | Clear session history + output |
| `/compact[:N]` | Compact context, optional keep count |
| `/export[:path]` | Export session to file |
| `/exit` | Disconnect |

### Poll Loop

`pollMessages()` runs every 500ms:
1. Fetches last 50 messages
2. Renders new messages from `seenMessageCount` cursor
3. After 4 empty polls (2s), checks session status → if idle/error, stops polling

## Styling (`src/styles.css`)

Catppuccin Mocha theme via CSS custom properties:

```css
:root {
  --bg: #1e1e2e;        /* base */
  --fg: #cdd6f4;        /* text */
  --surface: #181825;   /* mantle */
  --surface2: #313244;  /* surface0 */
  --border: #45475a;    /* surface1 */
  --accent: #89b4fa;    /* blue */
  --green: #a6e3a1;
  --red: #f38ba8;
  --yellow: #f9e2af;
  --cyan: #89dceb;
  --magenta: #cba6f7;
  --gray: #6c7086;
  --font: "Cascadia Code", "Fira Code", "JetBrains Mono", "Consolas", monospace;
}
```

Layout: `#app` flex column max-width 1200px height 100vh; `#connection-panel` host/port inputs + connect/disconnect + debug toggle; `#command-bar` slash-command shortcut buttons; `#output-panel` flex:1 scrollable; `#input-panel` text input + Send.

Part styling: `.part-text` `var(--fg)`; `.part-thinking` `var(--cyan)`; `.part-tool-calling` block `var(--magenta)`; `.part-tool-calling-part` inline magenta; `.part-tool-result` block green (`.tool-error` red); `.part-step-start`/`.part-step-finish` block yellow; `.part-unknown` gray.

## How to Add a New UI Feature

1. **Add backend endpoint (if needed)**: follow the `serve` skill — add method to `DummySessionManager` + `SessionManager`, expose route in `dummy_app.py`/`app.py`.
2. **Add client method** in `src/client.ts`:
   ```typescript
   async myAction(sessionId: string, param: string): Promise<SomeType> {
       const resp = await fetch(`${this.baseUrl}/session/${sessionId}/my-action?param=${param}`);
       if (!resp.ok) throw new Error(`myAction failed: ${resp.status}`);
       return await resp.json();
   }
   ```
3. **Add UI elements** in `index.html`: button in `#command-bar` (`<button id="btn-myaction" disabled>/myaction</button>`) or inputs as needed.
4. **Wire in main.ts**:
   ```typescript
   const btnMyAction = document.getElementById("btn-myaction") as HTMLButtonElement;
   btnMyAction.addEventListener("click", () => handleCommand("/myaction"));
   // Add case to handleCommand() switch
   case "myaction": {
       const result = await client.myAction(session.id, "value");
       log(`[SSE CLI] MyAction: ${JSON.stringify(result)}`, "info");
       break;
   }
   // Add to setConnected() disabled-buttons array
   for (const btn of [..., btnMyAction]) { ... }
   ```
5. **Add new message type (if needed)**: update `src/types.ts` (MessagePartType enum + `DUMMY_TYPE_MAP`/`OPENCODE_TYPE_MAP`), `src/renderer.ts` (`partTypeClass()` + `renderMessagePart()`), `src/styles.css` (`.part-<newtype>`).

## Testing

### Parse Tests (`src/parse_test.ts`)

```bash
cd src/app && npx tsx src/parse_test.ts
```

11 tests covering Dummy format (Text, Thinking, ToolCalling, ToolResult), Opencode format (Text, Tool→tool_calling, Reasoning→thinking, Step-Start/Finish), and edge cases (empty data, unknown types, multi-part messages).

### Browser Debugging

- DevTools → Console for `fetch` errors and `catch` block logs
- Toggle Debug checkbox to enable verbose logging (`debugMode`)
- Network tab for REST calls to port 4096

## Dependencies

- **Runtime**: none (vanilla TypeScript, no framework).
- **Dev only**: `typescript ^5.7.0`, `vite ^6.0.0`.
- `package.json` is `"type": "module"` for ESM imports.
