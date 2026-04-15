# PRD: Universal MCP Client (umcp)
**Version:** 1.2  
**Date:** 2026-04-15  
**Status:** Final Draft — all open questions resolved

---

## 1. Overview

`umcp` (Universal MCP Client) is a cross-platform, library-first Python tool that connects to any MCP (Model Context Protocol) server and drives it using a locally-running LLM via Ollama. It exposes a CLI for direct usage and an SDK for programmatic integration. The core value proposition is zero-cloud-dependency AI tool execution — the LLM runs locally, MCP servers can run anywhere, and the client glues them together with production-grade reliability.

---

## 2. Problem Statement

MCP servers expose tools and resources over a standard protocol, but consuming them requires:
- Writing custom LLM integration code per project
- Relying on cloud LLMs (OpenAI, Anthropic) which adds cost and privacy concerns
- No single client that handles all transport modes (stdio, SSE, HTTP)
- No unified config format for multi-server tool aggregation
- No safety layer between the LLM and tool execution (schema validation, timeouts, retries)

Developers either end up with one-off scripts or no client at all.

---

## 3. Goals

| # | Goal |
|---|------|
| G1 | Connect to any MCP server regardless of transport (stdio, SSE, HTTP) |
| G2 | Drive tool selection and execution via a local Ollama LLM |
| G3 | Support multi-server tool aggregation in a single session |
| G4 | Expose a clean Python SDK so apps can embed the client |
| G5 | Provide a reliable CLI for both chat and single-shot automation |
| G6 | Full observability: traces, logs, errors, timing per tool call |
| G7 | Be distributable as a pip package, usable as a cloned repo |
| G8 | Give users precise control over which tools the LLM can access per run |
| G9 | Protect against unreliable LLM output and malicious MCP server behavior |

## 4. Non-Goals (v1)

- No built-in OAuth flow (extensibility hooks provided, not implementation)
- No web UI or desktop GUI (future milestone)
- No cloud LLM support (Ollama-only in v1)
- No Docker distribution
- No MCP server hosting — this is a client only
- No parallel tool execution (sequential in v1, explicitly planned for v2)

---

## 5. Target Users

### Primary: Developer / Power User
- Runs MCP servers locally or on internal infra
- Wants CLI to script tool calls into automation pipelines
- Values privacy (no data leaving their machine)
- Comfortable with `mcp.json` config files

### Secondary: Team / Platform Engineer
- Wants to embed `umcp` as an SDK dependency in a larger app
- Needs structured logging and trace output for observability pipelines
- May run multiple MCP servers for different domains (DB, weather, internal tools)

---

## 6. Core Features

### 6.1 Transport Support
The client must support all three MCP transport modes:

| Transport | Protocol | When Used |
|-----------|----------|-----------|
| `stdio` | subprocess stdin/stdout | Local MCP servers run as child processes |
| `sse` | HTTP + Server-Sent Events | Remote or local servers with streaming |
| `http` | Plain HTTP (streamable) | REST-style MCP servers |

Transport is auto-detected from config or explicitly set per-server. Each transport implements a common abstract interface so the rest of the system treats them identically.

### 6.2 Multi-Server Tool Aggregation
- Multiple MCP servers can be declared in `mcp.json`
- At session start, the client connects to all configured servers and fetches their tool manifests
- All tools are merged into a single namespace exposed to the LLM
- Conflicts (duplicate tool names) are resolved by server name prefix: `weather.get_forecast` vs `db.get_forecast`
- **Tool name prefixing is always ON internally** — the internal system, logs, and traces always use `server.tool` format for consistency and debuggability
- **UX display is context-aware**: single-server runs may show clean names in output; multi-server runs always show prefixed names
- `--no-prefix-display` flag suppresses the prefix in CLI output only (internal naming unchanged)
- **The LLM always receives only the post-filter tool list** — it never sees tools that were filtered out (reliability boundary, not a security boundary)

### 6.3 LLM Integration (Ollama)

- Communicates with Ollama via its local REST API (`http://localhost:11434`)
- Supports tool-calling via Ollama's native tool use format
- Model is selectable at runtime; defaults to config value
- Primary target: `qwen2.5:7b` and `qwen2.5:14b` (best tool-calling reliability)
- Secondary: `llama3`, `deepseek-coder`

**Model capability detection** (automatic, no user action required):

On first use of a model, `umcp` probes Ollama to determine whether the model supports native structured tool calling. The result is cached locally.

| Detection outcome | Behavior |
|------------------|----------|
| Native tool calling supported | Uses Ollama's tool call format directly |
| No native tool calling | Falls back to prompt-based JSON extraction (see 6.3a) |
| Unknown model | Warns user, attempts native first, auto-falls back on failure |

**6.3a — Prompt-Based Fallback**

When a model does not support native tool calling, the system prompt instructs the model to output tool calls as a fenced JSON block. The fallback parser extracts and validates this JSON before executing.

```
<tool_call>
{"name": "weather.get_forecast", "arguments": {"city": "Tokyo"}}
</tool_call>
```

### 6.4 Interaction Modes

**Single-shot mode** (primary, for automation)
```bash
umcp run "check the weather in Tokyo and save it to the database"
umcp run --model qwen2.5:7b --server weather,db "..."
umcp run --dry-run "deploy my service"     # preview tool calls without executing
umcp run --stream "summarize the logs"     # stream tokens as they arrive
```
- Runs the task, executes all tool calls the LLM decides to make, returns final answer
- **Default: waits for full response** — clean, pipeable output
- **`--stream` flag**: streams LLM tokens to stdout as they arrive; tool call traces still go to stderr
- Exits with code 0 on success, non-zero on failure
- Output is clean, machine-readable (JSON with `--json` flag)

**Chat mode** (secondary, for exploration/debugging)
```bash
umcp chat
umcp chat --model llama3 --session my-debug-session
```
- REPL-style loop
- **Always streams** — token-by-token output, no waiting for full response
- Optional session memory (off by default)
- Full tool call trace visible inline with `--verbose`

### 6.5 MCP Server Configuration (`mcp.json`)

Default location: `./mcp.json`, then `~/.config/umcp/mcp.json`

```json
{
  "version": "1",
  "default_model": "qwen2.5:7b",
  "ollama_base_url": "http://localhost:11434",
  "servers": [
    {
      "name": "weather",
      "transport": "http",
      "url": "http://localhost:8000",
      "auth": {
        "type": "bearer",
        "token": "env:WEATHER_API_TOKEN"
      }
    },
    {
      "name": "database",
      "transport": "sse",
      "url": "http://localhost:9000/sse",
      "auth": {
        "type": "api_key",
        "header": "X-API-Key",
        "value": "env:DB_API_KEY"
      }
    },
    {
      "name": "local_tools",
      "transport": "stdio",
      "command": "python",
      "args": ["./tools_server.py"],
      "env": {
        "TOOLS_ENV": "dev"
      }
    }
  ],
  "execution": {
    "max_iterations": 10,
    "tool_timeout_ms": 5000,
    "total_timeout_ms": 30000,
    "max_retries_per_tool": 2
  },
  "cache": {
    "enabled": false,
    "ttl_seconds": 60,
    "storage": "memory"
  },
  "security": {
    "trusted_servers_only": false,
    "sanitize_tool_descriptions": true
  },
  "session": {
    "persist": false,
    "storage_path": "~/.config/umcp/sessions"
  },
  "logging": {
    "level": "info",
    "trace": true,
    "output": "stderr"
  }
}
```

CLI flags can override any config value on a per-run basis.

### 6.6 Authentication Model

| Auth Type | Config `type` | How It Works |
|-----------|--------------|--------------|
| Bearer token | `bearer` | Adds `Authorization: Bearer <token>` header |
| API key (header) | `api_key` | Adds custom header with key value |
| No auth | `none` / omitted | No auth headers sent |
| OAuth (future) | `oauth2` | Hook point defined, not implemented in v1 |

All secret values are resolved from environment variables using the `env:VAR_NAME` syntax. Inline secret strings are supported but warned against at startup.

### 6.7 Observability

Every tool call is traced with a structured log entry:

```json
{
  "trace_id": "abc123",
  "session_id": "sess_xyz",
  "iteration": 2,
  "timestamp": "2026-04-15T10:00:00Z",
  "server": "weather",
  "tool": "get_forecast",
  "input": { "city": "Tokyo" },
  "input_valid": true,
  "output": { "temp": "22C", "condition": "Sunny" },
  "cache_hit": false,
  "retry_count": 0,
  "latency_ms": 312,
  "status": "success"
}
```

**CLI trace commands:**
```bash
umcp trace last               # show last session's tool call trace
umcp trace session <id>       # show trace for a specific session
umcp trace --tail             # live trace output during a run
umcp trace --filter tool=db.* # filter trace by server/tool glob
```

Logs go to stderr by default (clean stdout for piping). Structured JSON logs available with `--log-format json`.

---

### 6.8 Tool Selection & Filtering

With many MCP servers connected, the LLM is exposed to potentially dozens of tools. Tool explosion degrades LLM reliability. The tool filter layer sits between the aggregator and the LLM — it controls exactly which tools the model sees on each run.

**OQ6 resolved: The LLM always sees only the post-filter tool list.** Sending all tools degrades accuracy and increases hallucination. Filtering is not optional — it is how the system maintains reliability at scale.

**Filter modes:**

| Mode | How to use | When to use |
|------|-----------|-------------|
| Whitelist (glob) | `--tools weather.*,db.insert` | Targeted automation runs |
| Server scope | `--server weather` | Limit to one server's tools |
| Keyword (default auto-filter) | `--tool-selection keyword` | Fast, lightweight relevance pruning |
| TF-IDF scoring | `--tool-selection hybrid` | Better relevance, still no extra model |
| Embedding similarity | `--tool-selection embedding` | Highest accuracy, requires embed model |
| All tools (explicit opt-in) | `--tool-selection all` | Exploration / debugging only |

**3-Stage Relevance Pipeline** (progressive enhancement):

```
User prompt
    │
    ▼
Stage 1 — Keyword filter (v1, always runs)
    Removes tools with zero keyword overlap with prompt.
    Fast, zero cost. Significant noise reduction.
    │
    ▼
Stage 2 — TF-IDF scoring (v1, runs after keyword pass)
    Scores remaining tools by term frequency against prompt.
    Top-N tools passed to LLM (configurable, default N=20).
    No external model required.
    │
    ▼
Stage 3 — Embedding similarity (v2, opt-in)
    Encodes prompt + tool descriptions via embedding model.
    Cosine similarity ranking. Highest accuracy.
    Requires: local embed model (e.g. nomic-embed-text via Ollama)
```

**CLI usage:**
```bash
# Default: keyword + TF-IDF filtering (auto)
umcp run "log the Tokyo forecast"

# Explicit tool whitelist (overrides relevance pipeline)
umcp run --tools "weather.*,db.insert_record" "log the Tokyo forecast"

# Select relevance strategy
umcp run --tool-selection hybrid "deploy and notify"
umcp run --tool-selection embedding "find tools related to payments"

# Opt out of filtering entirely (debug only)
umcp run --tool-selection all "what tools exist?"

# Single server scope
umcp run --server database "count all users"
```

**SDK usage:**
```python
result = await client.run(
    "log the Tokyo forecast",
    tools=["weather.*", "db.insert_record"],          # explicit whitelist
    tool_selection="hybrid"                            # or relevance strategy
)
```

**Config-level defaults:**
```json
{
  "tool_filter": {
    "strategy": "hybrid",
    "top_n": 20,
    "default_whitelist": ["*"],
    "exclude": ["dangerous_tool.*"],
    "embedding_model": "nomic-embed-text"
  }
}

---

### 6.9 Execution Guards (Timeouts & Limits)

Prevents hanging sessions and runaway agent loops.

| Guard | Default | Config key | CLI override |
|-------|---------|------------|-------------|
| Max agent iterations | 10 | `execution.max_iterations` | `--max-iterations` |
| Per-tool call timeout | 5000ms | `execution.tool_timeout_ms` | `--tool-timeout` |
| Total run timeout | 30000ms | `execution.total_timeout_ms` | `--timeout` |
| Max retries per tool | 2 | `execution.max_retries_per_tool` | — |

When a guard triggers:
- `tool_timeout_ms` exceeded → that tool call fails with `TIMEOUT` status → error fed back to LLM
- `total_timeout_ms` exceeded → run aborted, partial answer returned, exit code 2
- `max_iterations` exceeded → run aborted with warning, partial answer returned, exit code 2

---

### 6.10 Retry Strategy

Retries are typed. Different failure modes require different recovery approaches.

| Failure Type | Detection | Strategy |
|-------------|-----------|----------|
| Tool execution error | Tool returns error response | Retry same tool up to `max_retries_per_tool` times; pass error to LLM if all retries fail |
| LLM hallucination | Tool name not in tool list, or arguments don't match schema | Re-prompt with explicit correction: `"Tool X does not exist. Available tools: [...]"` |
| Invalid tool call JSON | JSON parse error on LLM output | Attempt auto-repair (common fixes: trailing commas, unquoted keys); if repair fails, re-prompt |
| Schema validation failure | Tool input fails JSON Schema check | Auto-coerce minor type mismatches (string→int); re-prompt with schema error if coercion fails |
| Network / transport error | httpx timeout, subprocess crash | Retry with exponential backoff (100ms, 200ms, 400ms); mark server degraded after 3 consecutive failures |

**Retry budget is shared** — total retries across all tools in one run is capped at `max_iterations * max_retries_per_tool` to prevent explosion.

The trace log includes `retry_count` and `failure_reason` per tool call for post-run diagnosis.

---

### 6.11 Tool Schema Validation

Before any tool call is dispatched to an MCP server, the input is validated against the tool's JSON Schema (from its manifest). This prevents server-side errors caused by LLM hallucinating wrong argument types or missing required fields.

**Validation pipeline:**

```
LLM outputs tool call
        │
        ▼
1. Tool name exists in filtered tool list?
   → No: hallucination retry (see 6.10)
        │
        ▼
2. Arguments pass JSON Schema validation?
   → Minor type mismatch: auto-coerce (e.g. "42" → 42)
   → Missing required field: re-prompt with schema error
   → Wrong type (not coercible): re-prompt with schema error
        │
        ▼
3. Dispatch to MCP server
```

Schema validation is always on. It cannot be disabled (it protects server stability).
Auto-coercion can be disabled via config: `"schema_validation": {"auto_coerce": false}`.

---

### 6.12 Response Caching

Some tool calls are deterministic for a given input (e.g., `get_forecast(city="Tokyo")` called twice in one session). Caching avoids redundant server round-trips.

**Cache key (normalized):**
```python
cache_key = hash(
    tool_name +                   # always prefixed: "weather.get_forecast"
    tool_schema_version +         # invalidates cache when tool signature changes
    stable_json(normalize(args))  # normalized argument fingerprint
)
```

**Normalization rules applied before hashing:**
1. Sort all object keys alphabetically
2. Coerce types to canonical form (`"42"` → `42`, `"true"` → `true`)
3. Remove fields explicitly marked `cache_ignore` in tool schema (e.g. request timestamps)
4. Stable JSON stringify (no whitespace, deterministic key order)

This maximizes cache hit rate across equivalent calls that differ only in formatting or type representation.

**Cache modes:**

| Mode | Config `cache.storage` | Scope |
|------|----------------------|-------|
| In-memory (default) | `"memory"` | Current run only |
| Session-scoped | `"session"` | Entire chat session |
| Persistent (file) | `"file"` | Survives across runs (TTL-gated) |

**Config:**
```json
{
  "cache": {
    "enabled": false,
    "ttl_seconds": 300,
    "max_size": 1000,
    "storage": "memory",
    "exclude_tools": ["db.insert_*", "*.write_*", "*.delete_*"]
  }
}
```

Cache is **off by default**. When enabled:
- `exclude_tools` glob patterns skip caching for write/mutating tools — these are never cached
- `max_size` caps number of entries (LRU eviction when full)
- `ttl_seconds` default raised to 300 (5 min) — more useful for real workflows
- A `cache_hit: true` field appears in the trace for cached calls
- `umcp cache clear` command flushes all stored cache entries

**CLI:**
```bash
umcp run --cache "what's the weather in Tokyo?"
umcp run --no-cache "..."          # force-disable even if config enables it
umcp run --cache-ttl 600 "..."    # override TTL for this run
umcp cache clear
umcp cache stats                   # hit rate, size, oldest entry, top cached tools
```

---

### 6.13 System Prompt Contract

The system prompt is the primary reliability lever for the agent loop. It is standardized, versioned, and not user-modifiable in v1 (a plugin hook is planned for v2).

**Base system prompt (condensed):**

```
You are a tool-calling agent. Follow these rules strictly:

1. ALWAYS prefer using tools over generating answers from memory.
2. NEVER fabricate tool names or outputs — only call tools from the provided list.
3. ALWAYS use the exact tool name and argument schema shown — no variations.
4. If a tool fails, reason about the failure and decide whether to retry, use an
   alternative tool, or report the failure to the user.
5. When all required tool calls are complete, produce a final answer and STOP.
   Do not continue calling tools after the task is resolved.
6. If the task cannot be completed with available tools, say so explicitly.
   Do not hallucinate a completion.

Available tools will be provided in each message. Respond only with valid tool
calls or a final plain-text answer. Never mix tool calls and final answers in
the same response.
```

**Fallback mode addendum** (injected when model has no native tool calling):

```
To call a tool, output ONLY the following block and nothing else:
<tool_call>
{"name": "<tool_name>", "arguments": {<args as JSON>}}
</tool_call>

To give a final answer, output plain text with no tool_call block.
```

The system prompt is stored in `umcp/prompts/base.txt` (versioned with the package). A `--system-prompt` override flag is available in v2 via the plugin system.

---

### 6.14 Security

Security threats are non-trivial once `umcp` is used against external or untrusted MCP servers.

**Threat model:**

| Threat | Vector | Mitigation |
|--------|--------|-----------|
| Prompt injection | Malicious tool descriptions containing instructions to the LLM | Sanitize tool descriptions before including in prompt (strip `<`, `>`, instruction-like patterns) |
| Malicious tool execution | Untrusted MCP server returning a tool that deletes files | Trusted server list (`security.trusted_servers_only: true`) + `--dry-run` before executing |
| Secret leakage via args | LLM passes env secrets as tool arguments | Env var values are never logged; masked in traces as `***` |
| Runaway execution | LLM loops infinitely calling tools | Max iteration + total timeout guards (see 6.9) |
| Schema confusion | Server returns malformed tool manifest | Strict Pydantic validation of tool manifests at connection time |

**Config:**
```json
{
  "security": {
    "trusted_servers_only": false,
    "trusted_servers": ["local_tools", "database"],
    "sanitize_tool_descriptions": true,
    "mask_env_values_in_logs": true
  }
}
```

**`--dry-run` as a safety primitive:**
```bash
# See every tool call the LLM would make — without executing any of them
umcp run --dry-run "delete all staging records and redeploy"
```
Output:
```
[DRY RUN] Would call: db.delete_records({"env": "staging"})
[DRY RUN] Would call: deploy.trigger({"env": "staging"})
No tools were executed.
```
`--dry-run` is available from Phase 2 (not deferred to Phase 4).

---

## 7. Python SDK Interface

The SDK is the core layer — the CLI is a thin wrapper over it.

### 7.1 Basic Usage

```python
from umcp import MCPClient

client = MCPClient(config="./mcp.json")

# Single task execution
result = await client.run("get the weather in Tokyo")
print(result.answer)
print(result.tool_calls)      # list of all tool calls made
print(result.cache_hits)      # number of cache hits
print(result.total_latency_ms)

# Single task with tool filter
result = await client.run(
    "log the Tokyo forecast",
    tools=["weather.*", "db.insert_record"],
    dry_run=True              # preview without executing
)

# Chat session
session = client.session(model="qwen2.5:7b")
async for chunk in session.chat("What tools do you have?"):
    print(chunk, end="")
```

### 7.2 Server Management

```python
# Connect to specific servers only
client = MCPClient(servers=["weather", "db"])

# Add a server at runtime
client.add_server({
    "name": "my_tools",
    "transport": "stdio",
    "command": "python my_server.py"
})

# List all aggregated tools (post-filter)
tools = await client.list_tools(filter="weather.*")
for tool in tools:
    print(f"{tool.server}.{tool.name}: {tool.description}")

# Call a tool directly (bypass LLM — no schema validation skipped)
result = await client.call_tool("weather.get_forecast", {"city": "Tokyo"})
```

### 7.3 Observability Hooks

```python
@client.on_tool_call
def log_call(trace):
    print(f"[{trace.server}] {trace.tool} took {trace.latency_ms}ms (retry={trace.retry_count})")

@client.on_error
def handle_error(err):
    sentry.capture(err)

@client.on_cache_hit
def log_cache(trace):
    print(f"Cache hit: {trace.tool}")
```

### 7.4 Plugin Hooks (v2 preview)

```python
# Override system prompt
@client.plugin("system_prompt")
def custom_prompt(base_prompt: str) -> str:
    return base_prompt + "\nAdditional rule: always respond in Japanese."

# Custom tool filter
@client.plugin("tool_filter")
def my_filter(tools: list[Tool], prompt: str) -> list[Tool]:
    return [t for t in tools if "dangerous" not in t.name]
```

---

## 8. CLI Reference

### Commands

```
umcp run <prompt>              Run a single task
umcp chat                      Start interactive chat session
umcp tools                     List all tools from all connected servers
umcp tools --server <name>     List tools from a specific server
umcp tools --filter <glob>     Filter tool list by name glob
umcp servers                   List configured servers and their connection status
umcp trace last                Show trace from last run
umcp trace session <id>        Show trace for session ID
umcp trace --tail              Live trace output while a run is in progress
umcp trace --filter <expr>     Filter trace output (e.g. tool=db.*)
umcp cache clear               Flush all cache entries
umcp cache stats               Show cache hit rate and size
umcp config init               Generate a starter mcp.json
umcp config validate           Validate mcp.json and report errors
umcp models                    List Ollama models and their tool-call capability
```

### Global Flags

```
--config <path>              Path to mcp.json (default: ./mcp.json)
--model <name>               Ollama model to use
--server <name,...>          Comma-separated server names to use (default: all)
--tools <glob,...>           Comma-separated tool name globs to whitelist
--tool-selection <mode>      keyword | hybrid | embedding | all  (default: hybrid)
--verbose, -v                Show tool call details inline
--json                       Output result as JSON
--stream                     Stream LLM tokens to stdout as they arrive (run mode)
--dry-run                    Show planned tool calls without executing
--no-prefix-display          Hide server prefix in CLI output (internal naming unchanged)
--no-trace                   Disable trace logging for this run
--session <id>               Use a named session (enables history)
--cache / --no-cache         Enable or force-disable caching for this run
--cache-ttl <seconds>        Override cache TTL for this run
--max-iterations <n>         Override max agent loop iterations
--timeout <ms>               Override total run timeout
--tool-timeout <ms>          Override per-tool call timeout
--log-level <level>          debug | info | warn | error
--log-format <fmt>           text | json
```

### Examples

```bash
# Basic single-shot
umcp run "what's the weather in Tokyo?"

# Use specific model and servers
umcp run --model qwen2.5:14b --server weather,db "check weather and log it"

# Limit which tools the LLM can use
umcp run --tools "weather.*,db.insert_record" "log Tokyo forecast to db"

# Preview tool calls without executing (safety check before destructive ops)
umcp run --dry-run "delete all staging data and redeploy"

# Output as JSON for scripting
umcp run --json "get all users from db" | jq '.answer'

# Verbose run showing all tool calls and retries
umcp run -v "deploy the staging service"

# Interactive chat with history
umcp chat --session dev-debug

# Inspect available tools, filtered
umcp tools
umcp tools --server database
umcp tools --filter "db.*"

# Check model capabilities
umcp models

# Validate config
umcp config validate --config ./my-mcp.json

# View cache stats
umcp cache stats
```

---

## 9. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         CLI Layer                             │
│         (umcp run / umcp chat / umcp tools / umcp trace)      │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                    MCPClient (SDK Core)                        │
│  - Config loader & Pydantic validator                         │
│  - Session manager                                            │
│  - LLM orchestration loop (plan → filter → validate → call)   │
│  - Trace & observability engine                               │
└──────┬──────────────────┬─────────────────────────┬──────────┘
       │                  │                         │
┌──────▼──────┐  ┌────────▼──────────┐  ┌──────────▼──────────┐
│Ollama       │  │Tool Filter Layer  │  │MCP Server Pool       │
│Adapter      │  │                   │  │                       │
│- /api/chat  │  │- Whitelist filter │  │Server 1 (http)        │
│- Tool call  │  │- Glob matching    │  │Server 2 (sse)         │
│  format     │  │- Namespace prefix │  │Server 3 (stdio)       │
│- Capability │  │- Future: relevance│  │                       │
│  detection  │  │  scoring          │  │Each server:           │
│- Fallback   │  └────────┬──────────┘  │- Connect on init      │
│  parser     │           │             │- Fetch tool manifest  │
└──────┬──────┘  ┌────────▼──────────┐  │- Execute tool calls   │
       │         │Schema Validation  │  │- Handle auth          │
       │         │Layer              │  │- Timeout + retry      │
       │         │- JSON Schema check│  └──────────────────────┘
       │         │- Auto-coerce      │
       │         │- Re-prompt logic  │  ┌──────────────────────┐
       │         └───────────────────┘  │Response Cache         │
       │                                │- hash(server+tool+    │
       └────────────────────────────────│  args) key            │
                                        │- TTL-based expiry     │
                                        │- Excludes write tools │
                                        └──────────────────────┘
```

### Agent Loop (single task, detailed)

```
1.  Connect to configured servers → fetch + validate tool manifests
2.  Apply security sanitization to tool descriptions
3.  Run relevance filter pipeline (keyword → TF-IDF → embedding if enabled)
    → Produces filtered tool list (top-N most relevant to user prompt)
    → Explicit --tools whitelist overrides relevance pipeline entirely
4.  Build final prefixed tool list for LLM (always server.tool format internally)
5.  Send (system prompt + filtered tool list + user task) to Ollama
    → stream tokens if --stream or chat mode; buffer if single-shot default
6.  Receive response:
    a. Plain text → final answer → go to 11
    b. Tool calls present → for each tool call:
       i.   Validate tool name exists in filtered list
            → if not: hallucination re-prompt → go to 5
       ii.  Validate arguments against JSON Schema
            → auto-coerce if possible; else schema re-prompt → go to 5
       iii. Check response cache (if enabled)
            → cache hit: use cached result, skip iv-v
       iv.  Check per-tool timeout guard → dispatch to MCP server
       v.   On transport/network error:
            → Retry with exponential backoff (100ms → 200ms → 400ms)
            → After 3 consecutive failures on same server:
               · Mark server as DEGRADED
               · Remove its tools from active pool
               · Inject into next LLM message:
                 "Server '<name>' is unavailable. Do not call its tools."
               · Continue execution with remaining servers
       vi.  On tool execution error: typed retry (see 6.10)
       vii. Append tool result (or error) to conversation
    c. Go to 5 (next iteration)
7.  Check max_iterations guard → abort if exceeded
8.  Check total_timeout guard → abort if exceeded
9.  (aborted) Return partial answer + warning; exit code 2
10. Clean up: terminate stdio subprocesses, close SSE connections
11. Emit full trace, return result
```

---

## 10. Technical Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Language | Python 3.11+ | MCP SDK is Python-first |
| MCP SDK | `mcp` (official Python SDK) | Protocol compliance, maintained |
| Ollama | `ollama` Python client or raw `httpx` | Lightweight, direct control |
| HTTP client | `httpx` (async) | SSE + HTTP in one library |
| Async runtime | `asyncio` | Required by MCP SDK |
| CLI framework | `typer` | Ergonomic CLI, auto-generates help |
| Config | `pydantic` v2 + JSON | Validation, type safety, good error messages |
| Schema validation | `jsonschema` | Tool input validation against MCP manifests |
| Logging | `structlog` | Structured JSON logging |
| Caching | `cachetools` | TTL-aware in-memory cache |
| Packaging | `pyproject.toml` + `hatchling` | Modern Python packaging |
| Testing | `pytest` + `pytest-asyncio` | Async test support |

---

## 11. Project Structure

```
umcp/
├── pyproject.toml
├── mcp.json.example
├── README.md
├── umcp/
│   ├── __init__.py
│   ├── client.py              # MCPClient — core SDK entry point
│   ├── cli.py                 # Typer CLI commands
│   ├── config.py              # mcp.json loading + Pydantic models
│   ├── session.py             # Session state + history management
│   ├── aggregator.py          # Tool namespace merging + conflict resolution
│   ├── filter.py              # Tool filter layer (whitelist, glob, future: relevance)
│   ├── validator.py           # Tool input JSON Schema validation + auto-coerce
│   ├── loop.py                # LLM agent loop (plan → filter → validate → call)
│   ├── retry.py               # Typed retry strategies
│   ├── cache.py               # Response cache (memory / session / file)
│   ├── security.py            # Tool description sanitization, trusted server list
│   ├── trace.py               # Observability engine + trace storage
│   ├── prompts/
│   │   ├── base.txt           # Base system prompt (versioned)
│   │   └── fallback.txt       # Prompt-mode addendum for non-tool-call models
│   ├── adapters/
│   │   ├── ollama.py          # Ollama API adapter + capability detection
│   │   └── fallback.py        # Prompt-based tool call extraction
│   ├── transports/
│   │   ├── base.py            # Abstract transport interface
│   │   ├── http.py            # HTTP transport
│   │   ├── sse.py             # SSE transport
│   │   └── stdio.py           # stdio subprocess transport
│   └── plugins/               # v2: plugin hook registry
│       └── __init__.py
└── tests/
    ├── test_client.py
    ├── test_aggregator.py
    ├── test_filter.py
    ├── test_validator.py
    ├── test_retry.py
    ├── test_cache.py
    ├── test_security.py
    ├── test_transports.py
    └── fixtures/
        └── mock_server.py
```

---

## 12. Phased Roadmap

### Phase 1 — Core (MVP)
**Goal:** Working single-shot CLI against a single MCP server via stdio

- [ ] Config loader + Pydantic validation
- [ ] stdio transport
- [ ] Ollama adapter with tool-calling support + capability detection
- [ ] Prompt-based fallback for non-native-tool-call models
- [ ] Base system prompt (`prompts/base.txt` + `fallback.txt`)
- [ ] Basic agent loop (max iterations guard)
- [ ] Tool schema validation + auto-coerce
- [ ] Execution guards: per-tool timeout + total timeout
- [ ] Basic typed retry strategy (tool failure + hallucination re-prompt)
- [ ] `umcp run` CLI command
- [ ] Basic trace logging to stderr
- [ ] `umcp tools` command
- [ ] `umcp models` command (capability detection output)

### Phase 2 — Multi-Transport + Multi-Server + Safety
**Goal:** Full transport coverage, tool aggregation, and baseline safety features

- [ ] HTTP transport
- [ ] SSE transport
- [ ] Multi-server connection pool
- [ ] Tool namespace aggregation + conflict resolution
- [ ] Tool filter layer (whitelist + glob matching)
- [ ] `--tools` CLI flag
- [ ] `--dry-run` flag (safe preview of tool calls)
- [ ] Auth: bearer token + API key (env-based)
- [ ] Security: tool description sanitization + trusted server list
- [ ] `umcp servers` command
- [ ] Config validation command

### Phase 3 — Chat Mode + Caching + SDK Polish
**Goal:** Interactive chat, response cache, and embeddable SDK

- [ ] `umcp chat` REPL mode
- [ ] Optional session persistence
- [ ] Response cache (memory + session modes)
- [ ] `umcp cache` commands
- [ ] `client.on_tool_call` / `client.on_error` / `client.on_cache_hit` hooks
- [ ] `umcp trace` commands + filtering
- [ ] Invalid JSON auto-repair retry
- [ ] pip package + PyPI publish

### Phase 3.5 — Parallel Tool Execution
**Goal:** Execute multiple tool calls from one LLM response concurrently

- [ ] Detect when LLM requests multiple independent tool calls
- [ ] Execute independent calls in parallel via `asyncio.gather`
- [ ] Dependency-aware execution (calls that depend on prior results remain sequential)
- [ ] Parallel execution trace (shows concurrent call groups)
- [ ] Config: `execution.parallel_tools: true` (opt-in)

### Phase 4 — Hardening + Extensibility
**Goal:** Production-ready for team/open-source use + plugin system

- [ ] Plugin hook system (`system_prompt`, `tool_filter`, `logging`)
- [ ] OAuth2 hook point
- [ ] Relevance-based tool filtering (score tools against prompt before sending to LLM)
- [ ] Persistent file-based response cache with TTL
- [ ] `--system-prompt` CLI override (via plugin)
- [ ] Comprehensive test suite (unit + integration with mock MCP servers)
- [ ] GitHub Actions CI
- [ ] Schema validation: `"auto_coerce": false` opt-out

---

## 13. Error Handling Strategy

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Clear error: `"Ollama not reachable at <url>. Is it running?"` |
| Model not found | Error with list of available models from Ollama |
| Model has no tool calling | Auto-detected; silently switches to fallback mode; logged at debug level |
| MCP server unreachable at startup | Log warning, skip server entirely, continue with remaining servers |
| MCP server goes offline mid-session | Retry once with backoff; on second failure mark DEGRADED, remove tools from pool, inject unavailability notice into LLM context, continue session |
| Tool call returns error | Typed retry: feed error back to LLM; retry up to `max_retries_per_tool` |
| LLM hallucinates tool name | Re-prompt: `"Tool X does not exist. Available tools: [...]"` |
| Invalid tool call JSON | Auto-repair attempt; if failed, re-prompt requesting valid JSON |
| Schema validation failure | Auto-coerce if possible; else re-prompt with schema detail |
| Transport timeout | Retry with backoff; mark server degraded after 3 consecutive timeouts |
| Max iterations reached | Return partial answer + warning; exit code 2 |
| Total timeout exceeded | Abort run; return partial answer + warning; exit code 2 |
| Auth failure (401/403) | Surface immediately with server name + auth config hint |
| Invalid mcp.json | Pydantic validation error with field-level detail |
| Malformed tool manifest | Skip that server's tools; log warning with server name |
| Prompt injection in tool desc | Sanitized silently at connection time; logged at debug level |

---

## 14. Success Metrics (v1)

| Metric | Target |
|--------|--------|
| Connect to any MCP server (all 3 transports) | 100% protocol compliance |
| Tool call round-trip latency (local, no cache) | < 500ms excluding LLM time |
| Multi-server tool aggregation | Tested with 3+ servers simultaneously |
| Tool filter reduces LLM confusion | Measurably fewer hallucinated tool names vs unfiltered |
| Schema validation prevents server errors | 0 invalid-input errors reaching MCP servers |
| CLI usable without reading docs | `umcp --help` covers 80% of use cases |
| SDK embeddable with < 10 lines of code | See 7.1 basic usage example |
| Retry strategy recovers from transient failures | > 80% recovery rate on retryable errors |

---

## 15. Resolved Design Decisions

All open questions from prior drafts are closed. Decisions are incorporated into the relevant sections above.

| # | Question | Decision | Where Applied |
|---|----------|----------|---------------|
| OQ1 | Tool name prefixing — opt-in or always on? | Always ON internally; `--no-prefix-display` hides it in CLI output only. Internal naming, logs, and traces always use `server.tool`. | §6.2, §8 CLI flags |
| OQ2 | Relevance filtering — keyword or embeddings? | Hybrid 3-stage pipeline: keyword → TF-IDF (both v1, no extra model) → embedding similarity (v2, opt-in via `--tool-selection embedding`). | §6.8 (full pipeline spec) |
| OQ3 | MCP server goes offline mid-session? | Retry once with backoff. On second failure: mark DEGRADED, remove tools from active pool, inject LLM notice (`"Server X unavailable — do not call its tools"`), continue session. | §9 agent loop step 6v, §13 error table |
| OQ4 | Stream output or wait for full response? | Default: full response (clean, pipeable). Opt-in streaming via `--stream` flag. Chat mode always streams. | §6.4, §8 CLI flags |
| OQ5 | Cache key — exact args or normalized? | Normalized: sort keys + type coercion + `tool_schema_version` + `cache_ignore` field stripping. Maximizes hit rate across equivalent calls. | §6.12 (full normalization spec) |
| OQ6 | Should LLM see all tools or filtered tools? | Filtered only — always. LLM never sees tools outside the post-filter list. Sending all tools degrades accuracy and increases hallucination. | §6.2, §6.8 |
