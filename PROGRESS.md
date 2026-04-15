# umcp — Implementation Progress

**Last updated:** 2026-04-15  
**Test count:** 126 passing / 126 total  
**Python:** 3.10+  
**MCP SDK:** 1.13.1

---

## Quick Status

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Core MVP — stdio + agent loop | DONE |
| Phase 2 | Multi-transport + multi-server + safety | DONE |
| Phase 3 | Chat mode + caching + SDK polish | DONE |
| Phase 3.5 | Parallel tool execution | DONE |
| Phase 4 | Hardening + extensibility + plugin system | DONE |

---

## Phase 1 — DONE

**Goal:** Working single-shot CLI against a single MCP server via stdio.

### What was built

| File | Description |
|------|-------------|
| `umcp/config.py` | Pydantic v2 models for all of `mcp.json`. `AppConfig.load()` searches `./mcp.json` then `~/.config/umcp/mcp.json`. Includes `AuthConfig`, `ServerConfig`, `ExecutionConfig`, `CacheConfig`, `SecurityConfig`, `ToolFilterConfig`. |
| `umcp/transports/base.py` | Abstract `BaseTransport` + `ToolInfo` + `ToolResult` dataclasses. All transports implement this interface — the rest of the system is transport-agnostic. |
| `umcp/transports/stdio.py` | Full stdio transport using `mcp.client.stdio.stdio_client`. Manages `AsyncExitStack` lifetime. Handles `asyncio.wait_for` timeout per call. Extracts text/data from MCP content blocks. |
| `umcp/adapters/ollama.py` | Ollama `/api/chat` adapter. Capability detection (known-good list + live probe). Caches per-model capability in `~/.config/umcp/model_capabilities.json`. Formats tools for Ollama's function-call schema. Supports both blocking `chat_once` and async-generator `chat_stream`. |
| `umcp/adapters/fallback.py` | Parser for `<tool_call>...</tool_call>` blocks in plain model output. Includes JSON auto-repair (trailing commas, single quotes). Case-insensitive tag matching. |
| `umcp/prompts/base.txt` | Versioned base system prompt. Six rules: prefer tools, no hallucination, exact names, handle errors, stop when done, admit failure. |
| `umcp/prompts/fallback.txt` | Addendum injected when model has no native tool calling. Defines `<tool_call>` JSON block format. |
| `umcp/validator.py` | JSON Schema validation via `jsonschema`. Auto-coerces: `"42"→42`, `"true"→True`, `1.0→1`, `"[...]"→list`. Returns `(valid, errors, coerced_args)`. |
| `umcp/retry.py` | Five typed retry reasons: `TOOL_ERROR`, `HALLUCINATION`, `INVALID_JSON`, `SCHEMA_FAILURE`, `TRANSPORT_ERROR`. `decide_retry()` maps each to `RETRY_TOOL`, `REPROMPT`, or `ABORT`. `RetryState` tracks per-tool and total budget. Exponential backoff (100ms→200ms→400ms). |
| `umcp/security.py` | Sanitizes tool descriptions — strips HTML tags, Llama tokens (`[INST]`), and injection phrases ("ignore previous instructions"). Truncates to 512 chars. `check_server_trusted()` enforces trusted-server-only mode. |
| `umcp/filter.py` | Tool filter pipeline. Applies exclusion globs first, then explicit whitelist (if provided), then keyword relevance scoring. Tokenizes with stop-word removal. `strategy="all"` bypasses relevance, returns everything. `top_n` caps keyword/hybrid strategies. |
| `umcp/cache.py` | TTL in-memory cache. Key = `sha256(full_name + schema_fingerprint + normalized_args)`. Normalization: sorted keys, type coercion (`1.0→1`), stable JSON. LRU eviction at `max_size`. Glob-based write-tool exclusion. |
| `umcp/trace.py` | `TraceEntry` dataclass with all fields (latency, cache_hit, retry_count, status). `Tracer` emits JSON to stderr on each call. `save_last()` persists to `~/.config/umcp/last_trace.json`. `load_last()` used by `umcp trace last`. |
| `umcp/aggregator.py` | `ToolAggregator` — connects to N transports, collects and merges tool manifests. Tracks degraded servers. `mark_degraded()` removes server's tools from active pool. `degraded_notice()` returns LLM-injectable message. |
| `umcp/session.py` | `RunSession` — holds message history for one agent loop run. Methods: `add_system`, `add_user`, `add_assistant`, `add_tool_result`, `inject_notice`. `add_tool_result(native_mode=True/False)` switches between `role=tool` and `role=user`. |
| `umcp/loop.py` | `AgentLoop` — full 11-step cycle: connect → sanitize → filter → build prompt → chat → parse → validate name → validate schema → cache check → execute with retry → record trace. Handles native + fallback tool call formats. Server degradation mid-loop. |
| `umcp/client.py` | `MCPClient` — async context manager SDK entry point. `run()`, `list_tools()`, `call_tool()`, `add_server()`. Event hook decorators: `on_tool_call`, `on_error`, `on_cache_hit`. |
| `umcp/cli.py` | Typer CLI. Commands: `run`, `chat` (basic), `tools`, `servers`, `models`, `trace last`, `cache clear`, `cache stats`, `config validate`, `config init`. All flags from PRD implemented. |

### CLI commands working (Phase 1)
```bash
umcp run "prompt"
umcp run --model qwen2.5:7b --dry-run "prompt"
umcp run --tools "weather.*" --tool-selection hybrid "prompt"
umcp run --json --no-trace "prompt"
umcp tools
umcp tools --server myserver --filter "db.*"
umcp models
umcp trace last
umcp cache stats
umcp config validate
umcp config init
```

### Tests (Phase 1)
- `tests/test_config.py` — 9 tests: loading, validation, auth resolution, server filtering
- `tests/test_validator.py` — 9 tests: schema validation, all coercion types
- `tests/test_filter.py` — 9 tests: whitelist, exclude, keyword scoring, top_n, stop words
- `tests/test_retry.py` — 8 tests: all 5 retry reasons, budget tracking, correction messages
- `tests/test_security.py` — 6 tests: HTML, Llama tokens, injection phrases, truncation
- `tests/test_fallback_parser.py` — 7 tests: single/multi call parsing, strip, JSON repair

---

## Phase 2 — DONE

**Goal:** Full transport coverage, multi-server tool aggregation, auth wiring.

### What was built

| File | Description |
|------|-------------|
| `umcp/transports/sse.py` | Full SSE transport using `mcp.client.sse.sse_client(url, headers)`. Auth headers forwarded on both the SSE GET stream and all message POSTs (handled automatically by SDK). Identical call interface to stdio transport. |
| `umcp/transports/http.py` | Full Streamable HTTP transport using `mcp.client.streamable_http.streamablehttp_client(url, headers)`. Handles the SDK's 3-tuple return `(read, write, get_session_id)` — `get_session_id` discarded client-side. Auth headers forwarded on all HTTP requests. |
| `umcp/transports/__init__.py` | `make_transport(server)` factory — routes to `StdioTransport`, `HttpTransport`, or `SseTransport` by `server.transport` field. |

### Key implementation detail
The MCP SDK's `streamablehttp_client` yields a **3-tuple** `(read, write, get_session_id)` — unlike stdio/SSE which yield a 2-tuple. The HTTP transport handles this correctly by unpacking and discarding `get_session_id`.

### Tests (Phase 2)
- `tests/test_transports.py` — 16 tests: factory routing for all 3 transports, mock transport lifecycle, aggregator with multiple servers, server failure handling, auth header verification for SSE and HTTP, timeout behaviour
- `tests/test_aggregator.py` — 6 tests: unique name aggregation from 2 servers, same-bare-name disambiguation (server prefix prevents collision), degradation exclusion, transport routing, `active_server_names`, `close_all`

### All 70 tests passing

---

## Phase 3 — DONE

**Goal:** Interactive chat mode, response caching end-to-end, SDK polish, pip packaging.

### What was built

| File | Description |
|------|-------------|
| `umcp/session.py` | Added `SessionStore` with `load(id)`, `save(id)`, `list_sessions()`, `delete(id)`, `save_trace()`, `load_trace()`. `RunSession` gets `session_id` field. |
| `umcp/cli.py` | Full rewrite: `chat` is now a true streaming REPL with session history persistence; `--cache/--no-cache/--cache-ttl/--session/--system-prompt` flags on `run`; `trace session <id>` and `trace tail` subcommands; `sessions list/delete` commands; `cache stats` shows live stats. |
| `umcp/client.py` | `run()` accepts `session_id`, `cache`, `cache_ttl`, `system_prompt` overrides; wires `SessionStore` load/save; fires `on_tool_call`/`on_error` hooks post-run; plugin system wired. |
| `umcp/loop.py` | `run()` accepts `prior_messages` (chat continuity), `cache_enabled`, `cache_ttl`, `system_prompt_override`; invalid JSON auto-repair attempt before re-prompt; parallel execution path; `LoopResult` carries `messages` list. |
| `umcp/cache.py` | Added `FileCache` backend using `~/.config/umcp/cache/<key>.json`; `ResponseCache` dispatches by `config.storage`; unified `stats()` and `clear()`. |
| `umcp/trace.py` | Added `group_id` field to `TraceEntry`; `save_session`/`load_session` static methods for named session traces. |
| `umcp/plugins/__init__.py` | Full `PluginRegistry` with pipeline composition for `system_prompt`, `tool_filter`; side-effect dispatch for `logging`. |
| `LICENSE` | MIT License. |
| `README.md` | Full install, quickstart, CLI reference, SDK usage, plugin hooks, mcp.json reference, architecture diagram. |

### Tests (Phase 3)
- `tests/test_cache.py` — 19 tests: key normalization, TTL, LRU eviction, write-tool exclusion, stats, file cache backend
- `tests/test_session.py` — 14 tests: RunSession messages, SessionStore load/save/list/delete/trace, multi-turn accumulation

---

## Phase 3.5 — DONE

**Goal:** Execute multiple tool calls from one LLM response concurrently.

### What was built

| File | Description |
|------|-------------|
| `umcp/config.py` | Added `parallel_tools: bool = False` to `ExecutionConfig`. Also added `OAuth2Config` fields to `AuthConfig`, `SchemaValidationConfig` with `auto_coerce`. |
| `umcp/trace.py` | Added `group_id: str | None` to `TraceEntry` for parallel call grouping. |
| `umcp/loop.py` | `_execute_parallel()` uses `asyncio.gather` for independent concurrent calls; trace annotates group_id. |

---

## Phase 4 — DONE

**Goal:** Production-ready for team/open-source use. Plugin system, OAuth2, relevance filtering, persistent cache.

### What was built

| File | Description |
|------|-------------|
| `umcp/plugins/__init__.py` | `PluginRegistry` with `register()`, `get()`, `has()`, `clear()`. Supports `system_prompt`, `tool_filter`, `logging` hooks. |
| `umcp/client.py` | `plugin(hook_name)` decorator wires into `PluginRegistry`; passed into `AgentLoop`. |
| `umcp/transports/sse.py` | OAuth2 client credentials flow via `_resolve_oauth2_token()` — fetches bearer token from `token_url` before connecting. |
| `umcp/transports/http.py` | Same OAuth2 flow as SSE. |
| `umcp/filter.py` | `_tfidf_filter()` using `sklearn.TfidfVectorizer` (optional dep, falls back gracefully); `_embedding_filter_async()` via Ollama `/api/embed` + cosine similarity; `apply_filter_async()` exposes full pipeline. |
| `umcp/adapters/ollama.py` | `embed(texts, model)` method calling `/api/embed`; `chat_stream` now yields full chunk dicts (not just token strings). |
| `umcp/cache.py` | `FileCache` backend with per-key JSON files + absolute TTL. |
| `umcp/cli.py` | `--system-prompt` flag on `run` and `chat`. |
| `.github/workflows/ci.yml` | pytest on push/PR, Python 3.10 + 3.11 + 3.12 matrix. |

### Tests (Phase 4)
- `tests/test_loop.py` — 12 tests: completed run, tool call flow, hallucinated tool, max iterations, dry run, cache hit, helpers, messages continuity
- `tests/test_client.py` — 11 tests: run delegation, list_tools filter, call_tool, add_server, plugin hooks, event hooks, session management

### All 126 tests passing

### What to build

#### 3.1 `umcp chat` — real streaming REPL
The `chat` command exists in `cli.py` but uses `run()` which is blocking. Phase 3 upgrades it to:
- True token streaming via `ollama.chat_stream()` (already implemented in adapter)
- Persistent conversation history across turns within a session (currently each `run()` is stateless)
- `--session <id>` flag saves/loads history from `~/.config/umcp/sessions/<id>.json`
- `SessionStore` class in `session.py` to handle persistence

**Files to create/modify:**
- `umcp/session.py` — add `SessionStore` with `load(id)`, `save(id)`, `list_sessions()`
- `umcp/cli.py` — rewrite `chat` command to maintain running history + stream tokens

#### 3.2 Response cache — wire end-to-end
`cache.py` is fully implemented but the `AgentLoop` only calls `cache.get/set` when `cache.config.enabled=True`. Phase 3 adds:
- `--cache` / `--no-cache` CLI flags properly plumbed through `client.run()` → `AgentLoop`
- `--cache-ttl <seconds>` override per run
- Session-scoped cache mode (`cache.storage = "session"`) — cache lives for one `umcp chat` session

**Files to modify:**
- `umcp/client.py` — pass cache overrides into loop
- `umcp/cli.py` — wire `--cache`, `--cache-ttl` flags

#### 3.3 `umcp trace` — session trace command
`umcp trace last` works. Phase 3 adds:
- `umcp trace session <id>` — loads trace for a named session
- `umcp trace --tail` — live tail mode (polls `last_trace.json` while a run is in progress)
- `--filter tool=db.*` syntax for trace filtering

**Files to modify:**
- `umcp/cli.py` — expand `trace_app` with `session` and `--tail` subcommands

#### 3.4 Invalid JSON auto-repair retry
Currently in `retry.py` the `INVALID_JSON` reason triggers `REPROMPT` on second attempt. Phase 3 wires the auto-repair attempt *before* the reprompt:
1. Parse fails → call `auto_repair_json()` from `adapters/fallback.py`
2. If repair succeeds → use repaired JSON, no reprompt needed
3. If repair fails → reprompt with format instruction

**Files to modify:**
- `umcp/loop.py` — add repair attempt in the fallback parsing path

#### 3.5 pip package + PyPI publish
- Add `LICENSE` file (MIT)
- Flesh out `README.md` with install, quickstart, config reference
- `python -m build` → `dist/umcp-0.1.0-py3-none-any.whl`
- `twine upload dist/*` → PyPI (manual step, not automated)

**Files to create:**
- `LICENSE`
- Update `README.md`

#### 3.6 Tests to add
- `tests/test_cache.py` — cache key normalization, TTL expiry, LRU eviction, write-tool exclusion, stats
- `tests/test_session.py` — session persistence, load/save roundtrip, message history across turns

---

## Phase 3.5 — NOT STARTED

**Goal:** Execute multiple tool calls from one LLM response concurrently.

### What to build

When an LLM response contains multiple tool calls (Ollama supports this), Phase 1-2 execute them **sequentially**. Phase 3.5 switches to **parallel execution** via `asyncio.gather` for independent calls.

#### Implementation plan
1. In `umcp/loop.py`, after parsing `calls_to_process`:
   - Group calls by dependency (calls that don't reference prior results can run in parallel)
   - For Phase 3.5: treat all calls in one response as independent (safe assumption for most tool sets)
   - Execute with `asyncio.gather(*[execute(tc) for tc in calls_to_process])`
2. Trace output shows concurrent call groups: `[iter=2, parallel=3]`
3. Config opt-in: `execution.parallel_tools: true` (default false until stable)

**Files to modify:**
- `umcp/loop.py` — refactor `_execute_with_retry` to be parallelisable, add `asyncio.gather` path
- `umcp/config.py` — add `parallel_tools: bool = False` to `ExecutionConfig`
- `umcp/trace.py` — add `group_id` field to `TraceEntry` for parallel call grouping

---

## Phase 4 — NOT STARTED

**Goal:** Production-ready for team/open-source use. Plugin system, OAuth, relevance filtering, persistent cache.

### What to build

#### 4.1 Plugin hook system
Allow users to override system prompt, tool filter, and logging behaviour without forking.

```python
# User code
@client.plugin("system_prompt")
def my_prompt(base: str) -> str:
    return base + "\nAlways respond in Japanese."

@client.plugin("tool_filter")
def my_filter(tools, prompt):
    return [t for t in tools if "dangerous" not in t.name]
```

**Files to create/modify:**
- `umcp/plugins/__init__.py` — `PluginRegistry`, `@plugin(hook_name)` decorator
- `umcp/client.py` — call registered hooks at right points in `run()`
- `umcp/loop.py` — accept injected system prompt + tool filter from plugin registry

#### 4.2 OAuth2 hook point
Provide a hook for OAuth2 token acquisition so enterprise MCP servers can be reached:

```json
{
  "auth": {
    "type": "oauth2",
    "token_url": "https://auth.example.com/token",
    "client_id": "env:OAUTH_CLIENT_ID",
    "client_secret": "env:OAUTH_CLIENT_SECRET"
  }
}
```

**Files to modify:**
- `umcp/config.py` — add `OAuth2Config` fields to `AuthConfig`
- `umcp/transports/sse.py` + `http.py` — use `httpx.Auth` subclass for OAuth2 token refresh

#### 4.3 Relevance-based tool filtering (TF-IDF → embeddings)
Phase 1-2 implements keyword overlap scoring. Phase 4 completes the 3-stage pipeline:

- **Stage 2 upgrade:** True TF-IDF scoring using `sklearn.feature_extraction.text.TfidfVectorizer` (optional dependency)
- **Stage 3:** Embedding similarity via Ollama's `/api/embed` endpoint using `nomic-embed-text`
  - Encode prompt + all tool descriptions at session start
  - Cosine similarity ranking
  - Enabled via `--tool-selection embedding`

**Files to modify:**
- `umcp/filter.py` — add `_tfidf_filter()` and `_embedding_filter()` functions
- `umcp/adapters/ollama.py` — add `embed(texts)` method calling `/api/embed`
- `umcp/config.py` — no change needed (embedding_model already in `ToolFilterConfig`)

#### 4.4 Persistent file-based cache
`cache.py` has `storage: "file"` in the config but only in-memory is implemented.

**Files to modify:**
- `umcp/cache.py` — add `FileCache` backend using `~/.config/umcp/cache/` with one JSON file per cache key, TTL checked on read

#### 4.5 `--system-prompt` CLI override (via plugin)
```bash
umcp run --system-prompt "Always be concise." "summarize the logs"
```

**Files to modify:**
- `umcp/cli.py` — add `--system-prompt` flag, register as inline plugin

#### 4.6 Comprehensive test suite + CI
- Integration tests using `tests/fixtures/mock_server.py` as a real stdio subprocess
- `tests/test_loop.py` — agent loop with mock Ollama responses + mock transports
- `tests/test_client.py` — SDK end-to-end with mock components
- `.github/workflows/ci.yml` — pytest on push, Python 3.10 + 3.11 + 3.12

---

## File Map (current state)

```
D:/AllBoutMcp/
├── PRD.md                        Product requirements (v1.2 — final)
├── PROGRESS.md                   This file
├── pyproject.toml                Package metadata + dependencies
├── mcp.json.example              Annotated example config
├── README.md                     (minimal — Phase 3 will flesh out)
├── umcp/
│   ├── __init__.py               Exports: MCPClient, AppConfig, LoopResult
│   ├── client.py                 MCPClient SDK entry point
│   ├── cli.py                    Typer CLI (all commands)
│   ├── config.py                 Pydantic config models
│   ├── aggregator.py             Multi-server transport pool
│   ├── session.py                Per-run message history
│   ├── loop.py                   Agent loop (11-step cycle)
│   ├── filter.py                 Tool filter (whitelist + keyword)
│   ├── validator.py              JSON Schema validation + auto-coerce
│   ├── retry.py                  Typed retry strategies
│   ├── cache.py                  TTL in-memory response cache
│   ├── security.py               Description sanitization
│   ├── trace.py                  Structured observability
│   ├── prompts/
│   │   ├── base.txt              Base system prompt (versioned)
│   │   └── fallback.txt          Fallback mode addendum
│   ├── adapters/
│   │   ├── ollama.py             Ollama API adapter + capability detection
│   │   └── fallback.py           <tool_call> block parser
│   ├── transports/
│   │   ├── base.py               Abstract transport interface
│   │   ├── stdio.py              stdio transport (DONE)
│   │   ├── sse.py                SSE transport (DONE)
│   │   └── http.py               Streamable HTTP transport (DONE)
│   └── plugins/
│       └── __init__.py           Hook registry placeholder (Phase 4)
└── tests/
    ├── test_config.py            9 tests
    ├── test_validator.py         9 tests
    ├── test_filter.py            9 tests
    ├── test_retry.py             8 tests
    ├── test_security.py          6 tests
    ├── test_fallback_parser.py   7 tests
    ├── test_transports.py        16 tests
    ├── test_aggregator.py        6 tests
    └── fixtures/
        └── mock_server.py        FastMCP test server (4 tools)
```

---

## Dependencies (installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `mcp` | 1.13.1 | Official MCP Python SDK — all transports |
| `httpx` | ≥0.27 | Async HTTP for Ollama API + SSE/HTTP transports |
| `typer` | ≥0.12 | CLI framework |
| `pydantic` | v2 | Config validation |
| `jsonschema` | ≥4.22 | Tool input schema validation |
| `structlog` | ≥24 | Structured logging (wired in Phase 3) |
| `rich` | ≥13 | CLI tables and formatting |
| `anyio` | ≥4 | Async primitives (MCP SDK dependency) |

---

## Known Gaps / Decisions to Revisit

| # | Gap | When to fix |
|---|-----|-------------|
| G1 | `structlog` is installed but not yet used — trace uses plain `json.dumps` to stderr | Phase 3 — replace with structlog processor chain |
| G2 | `umcp chat` does not persist history across turns yet — each prompt is a fresh `run()` | Phase 3 — `SessionStore` + running message list |
| G3 | Cache is implemented but `--cache` CLI flag doesn't yet override `config.cache.enabled` end-to-end | Phase 3 — wire through `client.run()` |
| G4 | `umcp trace session <id>` subcommand not yet implemented | Phase 3 |
| G5 | `execution.parallel_tools` config key doesn't exist yet | Phase 3.5 |
| G6 | No integration test that spawns a real stdio subprocess and runs a full loop | Phase 4 |
| G7 | `cache.storage = "file"` config accepted but silently falls back to memory | Phase 4 |
