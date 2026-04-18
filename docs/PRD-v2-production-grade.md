# UMCP v2 — Production-Grade Upgrade PRD

**Author:** Engineering Review  
**Date:** 2026-04-18  
**Status:** Awaiting Approval  

---

## Executive Summary

UMCP is a local-first AI agent that routes natural-language queries to MCP tool servers via Ollama. The core architecture is sound. This PRD identifies every gap between the current state and a production-grade application and proposes **five work phases**, ordered by risk and ROI. Each phase is independently shippable.

---

## Audit Findings

### 🔴 Critical — Security

| # | Finding | Location | Risk |
|---|---------|----------|------|
| S1 | CORS is `allow_origins=["*"]` | `web.py:36` | Any website can make credentialed requests to the dashboard |
| S2 | Zero authentication on web dashboard | `web.py` — all routes | Anyone on the LAN can submit prompts, delete sessions, reload config |
| S3 | No rate limiting on `/api/run` | `web.py:119` | A single client can flood Ollama with concurrent requests |
| S4 | No input size validation | `web.py:124` | Unlimited prompt length → OOM / context overflow |
| S5 | `/api/config/reload` unauthenticated | `web.py:220` | Attacker can hot-swap server config to point at malicious MCP server |
| S6 | `/api/servers` leaks commands + env | `web.py:54` | Exposes server binary paths, URLs, and transport details |
| S7 | Tool result size unbounded | `loop.py:446` | A malicious MCP server can inject arbitrarily large content into LLM context |
| S8 | Session IDs come from untrusted HTTP body | `web.py:125` | `_safe()` sanitizes but no length cap — path-manipulation potential |
| S9 | `ollama_base_url` not validated as localhost | `config.py:167` | SSRF: config could point to internal network hosts |
| S10 | Secrets potentially logged | `config.py` env vars | If trace is written to disk with full args, env vars may appear in logs |

### 🟠 High — Reliability

| # | Finding | Location | Risk |
|---|---------|----------|------|
| R1 | `collect_tools()` called on **every run** with no cache | `loop.py:104` | Hammers MCP subprocess stdin on every request; slow and fragile |
| R2 | Module-level mutable globals in `web.py` | `web.py:19-21` | Config reload mid-request is not thread-safe; can corrupt in-flight runs |
| R3 | `except: pass` swallows all I/O errors in session.py | `session.py:85-89` | Session history silently lost on disk full / permission errors |
| R4 | No concurrent request limit | `web.py` | 10 simultaneous `/api/run` calls will queue 10 Ollama requests |
| R5 | Web session message history has no cap | `client.py:180` | Web sessions grow indefinitely; CLI caps at 30 messages |
| R6 | `asyncio.create_task` in SSE endpoint is untracked | `web.py:195` | Orphaned task on server shutdown; cannot be cancelled |
| R7 | `assert self._connected` uses Python assert | `client.py:145` | Disabled with `python -O`; silent failure in optimized mode |
| R8 | New `OllamaAdapter` created per model override | `client.py:149` | Defeats in-memory capability cache; re-probes model on every override run |
| R9 | No connection health monitoring | `aggregator.py` | Disconnected MCP server not detected until a run fails |
| R10 | Tool call result not truncated before adding to context | `loop.py:446` | 1 MB tool response → context overflow → LLM failure |

### 🟡 Medium — Performance

| # | Finding | Location | Risk |
|---|---------|----------|------|
| P1 | TF-IDF vectorizer rebuilt from scratch every filter call | `filter.py:166` | Wasted CPU on every request with `strategy=hybrid` |
| P2 | `index.html` read from disk on every GET `/` | `web.py:40` | No caching; unnecessary disk I/O |
| P3 | Session files read/written per turn with no in-memory LRU | `session.py` | Hot sessions hit disk on every message |
| P4 | No gzip/brotli compression on API responses | `web.py` | Large tool lists and session histories sent uncompressed |
| P5 | `check_capability()` called inside `chat_once` loop | `ollama.py:163` | Extra await on every LLM call (mitigated by instance cache, but still async overhead) |

### 🔵 Production Grade — Observability & Operations

| # | Finding | Location | Risk |
|---|---------|----------|------|
| O1 | `print()` used for all logging instead of `structlog` | Throughout | No log levels, no JSON output, no filtering in production |
| O2 | No `/health` or `/ready` endpoint | `web.py` | Cannot use with any reverse proxy, load balancer, or K8s probe |
| O3 | No request correlation ID | `web.py` | Cannot trace a specific request through logs |
| O4 | No graceful shutdown (SIGTERM) | `web.py:265` | Active SSE streams dropped immediately on restart |
| O5 | `autoSessionId` not persisted across page refreshes | `index.html` JS | Web chat loses context on every F5 |
| O6 | No session rename or export | UI + API | Users cannot manage their chat history |
| O7 | Error responses lack error codes | `web.py` | Frontend cannot distinguish "prompt required" from "client not connected" |

---

## Proposed Work Phases

---

### Phase 1 — Security Hardening *(~2 days)*

**Goal:** Make the dashboard safe to run on a shared machine or behind a reverse proxy.

#### 1.1 API Key Authentication
- Add `dashboard_api_key` field to `AppConfig` (optional; if set, enforce on all routes)
- Middleware reads `Authorization: Bearer <key>` or `X-API-Key: <key>` header
- Configurable via `.env`: `UMCP_API_KEY=...`
- `GET /` (dashboard HTML) is exempt; all `/api/*` routes require the key
- Return `401 Unauthorized` with JSON `{"error": "Unauthorized", "code": "AUTH_REQUIRED"}` if missing/wrong

#### 1.2 Rate Limiting
- Add per-IP rate limiting to `/api/run` and `/api/run/stream`: default 10 req/min, 2 concurrent
- Use an in-process `asyncio.Semaphore` for concurrency; sliding window counter for rate
- Return `429 Too Many Requests` with `Retry-After` header

#### 1.3 Input Validation
- Enforce `prompt` max length: 4096 characters
- Enforce `session_id` max length: 128 characters; reject if contains `..`, `/`, `\`
- Return `422 Unprocessable Entity` with field-level error on violation

#### 1.4 CORS Hardening
- Replace `allow_origins=["*"]` with configurable `cors_origins` list in `AppConfig`
- Default: `["http://localhost:8765", "http://127.0.0.1:8765"]`
- Only allow `GET`, `POST`, `DELETE` methods

#### 1.5 Response Sanitization
- `/api/servers`: strip `command`, `args`, `env` fields before returning (only expose `name`, `transport`, `status`, `tool_count`)
- `/api/config`: already sanitized; add explicit allowlist of safe keys

#### 1.6 Tool Result Truncation
- Truncate any single tool result to 32 KB before adding to session context
- Append `[... truncated at 32 KB — ask for a specific subset]` when truncated
- Make limit configurable: `execution.max_tool_result_bytes` in `mcp.json`

#### 1.7 `ollama_base_url` Validation
- Reject configs where `ollama_base_url` resolves to a non-loopback address unless `security.allow_remote_ollama: true` is explicitly set

---

### Phase 2 — Reliability & Consistency *(~2 days)*

**Goal:** Eliminate silent failures and make every run deterministic.

#### 2.1 Tool List Cache with TTL
- Cache `collect_tools()` result in `ToolAggregator` with a 30-second TTL
- Invalidate on `config_reload` and on any server reconnect
- Reduces MCP subprocess IPC from N-per-run to once-per-30s

#### 2.2 Session Message Cap for Web
- Web sessions: cap history at 50 messages (same discipline as CLI's 30)
- When cap is reached, drop oldest assistant+tool pairs (keep system prompt + recent turns)
- Add `session.max_messages: int = 50` to `AppConfig`

#### 2.3 Structured Error Handling in Session Store
- Replace all `except: pass` with `except Exception as exc: log.warning(...)` using `structlog`
- Surface disk write errors as `HTTPException(503)` in web endpoints

#### 2.4 Fix Concurrent Request Safety
- Replace module-level globals in `web.py` with a `AppState` dataclass held on `app.state`
- Use `asyncio.Lock` around config reload to prevent race with in-flight requests

#### 2.5 Replace `assert` with Explicit Exceptions
- Replace `assert self._connected` with `if not self._connected: raise RuntimeError("...")`
- Works correctly under `python -O`

#### 2.6 Track `asyncio.create_task` in SSE
- Store the task reference; cancel it on client disconnect detection
- Use `request.is_disconnected()` in the generate loop to abort early

#### 2.7 Model Override Adapter Reuse
- Cache `OllamaAdapter` instances by model name in `MCPClient`
- Prevents re-probing capability on every `--model` override run

#### 2.8 Fix `autoSessionId` Page Refresh
- Persist `autoSessionId` to `sessionStorage` (survives page refresh, cleared on tab close)
- On page load: resume existing `autoSessionId` if present and not older than 1 hour

---

### Phase 3 — Performance *(~1 day)*

**Goal:** Sub-100ms overhead for the plumbing layer on warm requests.

#### 3.1 TF-IDF Vectorizer Caching
- Cache the fitted `TfidfVectorizer` keyed on sorted tool full-names
- Rebuild only when tool list changes (version hash comparison)
- Reduces filter latency from ~50ms to ~1ms on warm path

#### 3.2 Static File Serving with Cache Headers
- Serve `index.html` with `Cache-Control: no-store` (it changes during dev)
- Serve any future static assets (CSS/JS files if extracted) with `Cache-Control: max-age=3600`
- Read `index.html` once at startup into memory; re-read on file change via mtime check

#### 3.3 Session In-Memory LRU Cache
- Add `maxsize=20` LRU cache over `SessionStore.load()` using `functools.lru_cache`-equivalent
- Invalidate on every `save()` for that session_id
- Eliminates repeated disk reads for active sessions

#### 3.4 Gzip Compression
- Enable `GZipMiddleware` from `starlette.middleware.gzip` for responses > 1 KB
- Compresses tool list JSON (~60% reduction) and session message lists

---

### Phase 4 — Observability & Operations *(~1 day)*

**Goal:** Make the app operable in production: logs, health, graceful shutdown.

#### 4.1 Structured Logging Throughout
- Replace all `print(..., file=sys.stderr)` with `structlog.get_logger().info/warning/error(...)`
- Configure JSON output when `logging.output = "file"` or `LOG_FORMAT=json` env var
- Include `request_id`, `session_id`, `tool`, `server` as bound context fields

#### 4.2 Request Correlation IDs
- Generate `X-Request-ID` UUID for every incoming request
- Propagate to all log lines within that request via `structlog.contextvars.bind_contextvars`
- Echo back in response headers

#### 4.3 Health + Ready Endpoints
```
GET /health  → 200 {"status": "ok", "uptime_s": 123}  (always)
GET /ready   → 200 {"ollama": true, "servers": {"sqlite": true, ...}}
              or 503 if any critical dependency is down
```

#### 4.4 Graceful Shutdown
- On SIGTERM: stop accepting new `/api/run` requests (return 503), drain in-flight runs up to 30s, then close
- Track in-flight run count with an `asyncio.Semaphore`

#### 4.5 Structured Error Codes
- All error responses: `{"error": "...", "code": "PROMPT_REQUIRED" | "AUTH_REQUIRED" | "RATE_LIMITED" | "SERVER_ERROR" | ...}`
- Frontend uses `code` field for user-facing messages, not raw `error` string

#### 4.6 `.env` Loading in `web.py` Startup
- Currently only `cli.py` loads `.env`; add `load_dotenv()` at the top of `serve()` so `umcp web` also picks up secrets

---

### Phase 5 — UX Polish *(~1 day)*

**Goal:** Make the chat UI feel complete and trustworthy.

#### 5.1 Session Rename
- `PATCH /api/sessions/{session_id}` — update title only
- Double-click on session title in sidebar to rename inline

#### 5.2 Session Export
- `GET /api/sessions/{session_id}/export?format=markdown` — returns chat as markdown
- Download button in session header

#### 5.3 Delete Confirmation
- Show `Are you sure?` micro-confirm before deleting a session (shift+click to skip)

#### 5.4 Error State in Chat UI
- When backend returns `success: false`, show a distinct error bubble with `exit_reason` and retry button
- Currently errors show as plain text indistinguishable from answers

#### 5.5 Typing Indicator
- Show animated dots while a run is in progress (currently the send button just disables)

#### 5.6 Session Search
- Client-side filter input above session list to search by title
- Matches on first keypress, no debounce needed (all sessions are in memory after initial load)

#### 5.7 Tool Call Trace in UI
- Expandable "Tool calls" section per message showing tool name, args, result summary, latency
- Data already available in `/api/trace`; wire it per-message

---

## Implementation Priority

| Phase | Effort | Risk Reduction | Recommended Order |
|-------|--------|---------------|-------------------|
| Phase 1 — Security | ~2 days | 🔴 Critical | **First** |
| Phase 2 — Reliability | ~2 days | 🟠 High | **Second** |
| Phase 4 — Observability | ~1 day | 🔵 Medium | Third |
| Phase 3 — Performance | ~1 day | 🟡 Medium | Fourth |
| Phase 5 — UX | ~1 day | 🔵 Low | Fifth |

**Total estimate: ~7 engineering days for full production-grade quality.**

Each phase is independently mergeable and testable. Phases 1 and 2 are non-negotiable before any external exposure.

---

## What This Does NOT Change

- The agent loop architecture (it's solid)
- The MCP transport layer (stdio/SSE/HTTP work correctly)
- The tool filter pipeline (affinity threshold + keyword scoring works)
- The session storage format (backward compatible)
- The CLI interface (no breaking changes to `umcp run` / `umcp chat`)

---

*Awaiting approval to begin Phase 1 implementation.*
