# umcp — Universal MCP Client

[![CI](https://github.com/your-org/umcp/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/umcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/umcp)](https://pypi.org/project/umcp/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**`umcp` is a cross-platform, library-first Python tool that connects to any [MCP](https://modelcontextprotocol.io) server and drives it using a locally-running LLM via [Ollama](https://ollama.com).**

Zero cloud dependency. Your data stays on your machine.

---

## Features

- 🔌 **All MCP transports**: stdio, SSE, Streamable HTTP
- 🤖 **Local-first LLM**: Ollama with native tool calling + fallback for models without it
- 🗄️ **Multi-server aggregation**: connect to N servers, merged tool namespace
- 🔍 **Smart tool filtering**: keyword → TF-IDF → embedding relevance pipeline
- 🛡️ **Safety built-in**: schema validation, retry strategies, execution guards, dry-run
- 📦 **Clean SDK**: embed in any Python app with `MCPClient`
- 🔌 **Plugin hooks**: customize system prompt and tool filter without forking
- 💾 **Session persistence**: named chat sessions with history across runs
- 📊 **Full observability**: structured trace per tool call

---

## Installation

```bash
pip install umcp
```

Or from source:

```bash
git clone https://github.com/your-org/umcp
cd umcp
pip install -e ".[dev]"
```

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one MCP server

---

## Quickstart

### 1. Pull a model
```bash
ollama pull qwen2.5:7b
```

### 2. Create `mcp.json`
```bash
umcp config init
```

Or manually:
```json
{
  "version": "1",
  "default_model": "qwen2.5:7b",
  "ollama_base_url": "http://localhost:11434",
  "servers": [
    {
      "name": "mytools",
      "transport": "stdio",
      "command": "python",
      "args": ["./my_server.py"]
    }
  ]
}
```

### 3. Run a task
```bash
umcp run "get the weather in Tokyo"
```

---

## CLI Reference

### `umcp run` — Single-shot task
```bash
umcp run "what's the weather in Tokyo?"
umcp run --model qwen2.5:14b --server weather,db "check weather and log it"
umcp run --tools "weather.*,db.insert_record" "log Tokyo forecast to db"
umcp run --dry-run "delete all staging data and redeploy"       # preview tool calls
umcp run --json "get all users from db" | jq '.answer'          # machine-readable output
umcp run --stream "summarize the logs"                          # stream tokens
umcp run --cache "what's the weather?"                          # enable response cache
umcp run --no-cache "..."                                       # force-disable cache
umcp run --cache-ttl 600 "..."                                  # custom TTL
umcp run --session my-session "continue our work"               # named session
umcp run --system-prompt "Always respond in Japanese."  "..."   # custom prompt addendum
umcp run -v "deploy staging"                                    # verbose (show tool stats)
```

### `umcp chat` — Interactive REPL
```bash
umcp chat
umcp chat --model llama3 --session my-debug-session    # with history
umcp chat --system-prompt "Be concise."
```
Always streams. Ctrl+C to exit. With `--session`, history is preserved across runs.

### `umcp tools` — Tool inspection
```bash
umcp tools
umcp tools --server database
umcp tools --filter "db.*"
umcp tools --json
```

### `umcp servers` — Server status
```bash
umcp servers
```

### `umcp models` — Ollama model capabilities
```bash
umcp models
umcp models --json
```

### `umcp trace` — Observability
```bash
umcp trace last                     # show last run's trace
umcp trace last --filter "db.*"     # filter by tool glob
umcp trace session my-session       # show named session trace
umcp trace tail                     # live-tail while a run is active
umcp trace last --json              # machine-readable JSON
```

### `umcp cache` — Cache management
```bash
umcp cache clear
umcp cache stats
```

### `umcp sessions` — Session management
```bash
umcp sessions list
umcp sessions delete my-session
```

### `umcp config` — Configuration
```bash
umcp config validate
umcp config validate --config ./my-mcp.json
umcp config init
```

---

## Global Flags

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to mcp.json (default: ./mcp.json) |
| `--model <name>` | Ollama model to use |
| `--server <name,...>` | Comma-separated server names |
| `--tools <glob,...>` | Tool name globs to whitelist |
| `--tool-selection <mode>` | `keyword` \| `hybrid` \| `embedding` \| `all` |
| `--verbose, -v` | Show tool call details inline |
| `--json` | Output result as JSON |
| `--stream` | Stream LLM tokens (run mode) |
| `--dry-run` | Preview tool calls without executing |
| `--no-prefix-display` | Hide server prefix in CLI output |
| `--no-trace` | Disable trace logging |
| `--session <id>` | Named session (enables history) |
| `--cache / --no-cache` | Enable/disable caching for this run |
| `--cache-ttl <seconds>` | Override cache TTL |
| `--system-prompt <text>` | Append to base system prompt |
| `--max-iterations <n>` | Override max agent loop iterations |
| `--timeout <ms>` | Override total run timeout |
| `--tool-timeout <ms>` | Override per-tool timeout |
| `--log-level <level>` | `debug` \| `info` \| `warn` \| `error` |
| `--log-format <fmt>` | `text` \| `json` |

---

## Python SDK

```python
from umcp import MCPClient

async with MCPClient(config="./mcp.json") as client:
    # Single task
    result = await client.run("get the weather in Tokyo")
    print(result.answer)
    print(result.tool_calls_made, result.cache_hits, result.total_latency_ms)

    # With options
    result = await client.run(
        "log the Tokyo forecast",
        tools=["weather.*", "db.insert_record"],   # tool whitelist
        dry_run=True,                              # preview only
        session_id="my-session",                   # persist history
        cache=True,                                # enable cache
        system_prompt="Always be concise.",        # addendum
    )

    # Chat session with history
    result1 = await client.run("What's the weather?", session_id="demo")
    result2 = await client.run("And tomorrow?", session_id="demo")  # remembers context

    # List tools
    tools = await client.list_tools(filter="weather.*")

    # Direct tool call (bypass LLM)
    result = await client.call_tool("weather.get_forecast", {"city": "Tokyo"})

    # Manage sessions
    sessions = client.list_sessions()
    client.delete_session("old-session")
```

### Plugin Hooks

```python
@client.plugin("system_prompt")
def my_prompt(base: str) -> str:
    return base + "\nAlways respond in Japanese."

@client.plugin("tool_filter")
def my_filter(tools, prompt):
    return [t for t in tools if "dangerous" not in t.name]
```

### Event Hooks

```python
@client.on_tool_call
def log_call(entry):
    print(f"[{entry.server}] {entry.tool} took {entry.latency_ms}ms")

@client.on_error
def handle_error(result):
    sentry.capture_message(result.exit_reason)
```

---

## `mcp.json` Configuration

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
      "auth": { "type": "bearer", "token": "env:WEATHER_API_TOKEN" }
    },
    {
      "name": "database",
      "transport": "sse",
      "url": "http://localhost:9000/sse",
      "auth": { "type": "api_key", "header": "X-API-Key", "value": "env:DB_API_KEY" }
    },
    {
      "name": "local_tools",
      "transport": "stdio",
      "command": "python",
      "args": ["./tools_server.py"],
      "env": { "TOOLS_ENV": "dev" }
    }
  ],
  "execution": {
    "max_iterations": 10,
    "tool_timeout_ms": 5000,
    "total_timeout_ms": 30000,
    "max_retries_per_tool": 2,
    "parallel_tools": false
  },
  "cache": {
    "enabled": false,
    "ttl_seconds": 300,
    "max_size": 1000,
    "storage": "memory",
    "exclude_tools": ["*.insert_*", "*.write_*", "*.delete_*"]
  },
  "tool_filter": {
    "strategy": "hybrid",
    "top_n": 20,
    "exclude": ["dangerous_tool.*"],
    "embedding_model": "nomic-embed-text"
  },
  "security": {
    "trusted_servers_only": false,
    "sanitize_tool_descriptions": true,
    "mask_env_values_in_logs": true
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

---

## Tool Filtering

umcp uses a 3-stage pipeline:

| Stage | Strategy | Requires |
|-------|----------|----------|
| 1 — Keyword | Always runs | Nothing |
| 2 — TF-IDF | `--tool-selection hybrid` | `scikit-learn` (optional) |
| 3 — Embedding | `--tool-selection embedding` | Ollama + `nomic-embed-text` |

```bash
umcp run "log the Tokyo forecast"                          # default: hybrid
umcp run --tool-selection keyword "..."                    # keyword only
umcp run --tool-selection embedding "..."                  # embedding similarity
umcp run --tool-selection all "what tools exist?"          # no filtering (debug)
umcp run --tools "weather.*,db.insert_record" "..."        # explicit whitelist
```

For embedding support install: `pip install umcp[embed]` (adds `nomic-embed-text` pull instructions).

---

## Architecture

```
CLI Layer (umcp run / chat / tools / trace)
         │
MCPClient (SDK Core)
  - Config loader & Pydantic validator
  - Session manager (history persistence)
  - Plugin registry
       │
AgentLoop (plan → filter → validate → call)
  ├── OllamaAdapter (tool calling + streaming + capability detection + embeddings)
  ├── Tool Filter Layer (keyword → TF-IDF → embedding)
  ├── Schema Validation (auto-coerce + re-prompt)
  ├── Response Cache (memory / file, TTL-based)
  └── MCP Server Pool (stdio / SSE / HTTP transports)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Build distribution
python -m build

# Publish to PyPI
twine upload dist/*
```

---

## Recommended Models

| Model | Tool Calling | Notes |
|-------|-------------|-------|
| `qwen2.5:7b` | ✅ Native | Best balance of speed + accuracy |
| `qwen2.5:14b` | ✅ Native | Better for complex multi-tool tasks |
| `llama3.1:8b` | ✅ Native | Good alternative |
| `deepseek-r1:7b` | ✅ Native | Strong reasoning |
| `gemma2:9b` | ❌ Fallback | Uses `<tool_call>` prompt mode |

---

## License

MIT — see [LICENSE](LICENSE).
