"""Web dashboard server for umcp — FastAPI backend."""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .client import MCPClient
from .config import AppConfig
from .log import get_logger
from .trace import Tracer

_log = get_logger()

_STATIC = Path(__file__).parent / "static"

# Static file in-memory cache (re-read on mtime change for dev hot-reload)
_html_path = _STATIC / "index.html"
_html_content: str = _html_path.read_text(encoding="utf-8")
_html_mtime: float = _html_path.stat().st_mtime

# ------------------------------------------------------------------ #
# Singleton app state (set once in create_app, never mutated after)
# ------------------------------------------------------------------ #
_client: MCPClient | None = None
_config: AppConfig | None = None
_config_path: Path | None = None
_config_reload_lock: asyncio.Lock | None = None  # protects hot-reload

# ------------------------------------------------------------------ #
# Input validation constants
# ------------------------------------------------------------------ #
_MAX_PROMPT_LEN = 4096
_MAX_SESSION_ID_LEN = 128
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def _validate_session_id(sid: str | None) -> str | None:
    """Return an error string if session_id is invalid, else None."""
    if not sid:
        return None
    if len(sid) > _MAX_SESSION_ID_LEN:
        return f"session_id exceeds {_MAX_SESSION_ID_LEN} characters"
    if ".." in sid or "/" in sid or "\\" in sid:
        return "session_id contains invalid characters"
    return None


def _validate_run_body(prompt: str, session_id: str | None) -> str | None:
    """Return an error string if request body is invalid, else None."""
    if len(prompt) > _MAX_PROMPT_LEN:
        return f"prompt exceeds {_MAX_PROMPT_LEN} characters"
    return _validate_session_id(session_id)


# ------------------------------------------------------------------ #
# Rate limiting (per-IP sliding window + concurrent semaphore)
# ------------------------------------------------------------------ #
_RATE_WINDOW_S = 60
_RATE_MAX_REQUESTS = 10   # per IP per window
_rate_store: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    now = time.monotonic()
    window = _rate_store[ip]
    # Evict timestamps outside the window
    while window and window[0] < now - _RATE_WINDOW_S:
        window.popleft()
    if len(window) >= _RATE_MAX_REQUESTS:
        return False
    window.append(now)
    return True


def create_app(
    client: MCPClient,
    config: AppConfig,
    config_path: Path | None = None,
) -> FastAPI:
    global _client, _config, _config_path, _config_reload_lock
    _client = client
    _config = config
    _config_path = config_path
    _config_reload_lock = asyncio.Lock()

    # Resolve API key once at startup
    _api_key: str | None = config.resolve_dashboard_api_key()

    # Concurrent run semaphore — at most 2 LLM runs at the same time
    _run_semaphore = asyncio.Semaphore(2)

    app = FastAPI(title="umcp dashboard", docs_url=None, redoc_url=None)

    # ── Middleware: CORS (hardened) ──────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["GET", "POST", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # ── Middleware: gzip ────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # ── Middleware: Request correlation ID ─────────────────────────
    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        request_id = uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Middleware: API key auth ────────────────────────────────────
    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        # Dashboard HTML and /health are always public
        if request.url.path in ("/", "/health") or _api_key is None:
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or _parse_bearer(
            request.headers.get("Authorization", "")
        )
        if not provided or not secrets.compare_digest(provided, _api_key):
            return JSONResponse(
                {"error": "Unauthorized", "code": "AUTH_REQUIRED"},
                status_code=401,
            )
        return await call_next(request)

    # ── Routes ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        global _html_content, _html_mtime
        try:
            mtime = _html_path.stat().st_mtime
            if mtime != _html_mtime:
                _html_content = _html_path.read_text(encoding="utf-8")
                _html_mtime = mtime
        except OSError:
            pass
        return HTMLResponse(_html_content, headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/auth/status")
    async def api_auth_status():
        """Let the frontend know whether an API key is required."""
        return {"required": _api_key is not None}

    @app.get("/api/config")
    async def api_config():
        if not _config:
            return {}
        return {
            "model": _config.default_model,
            "ollama_url": _config.ollama_base_url,
            "server_count": len(_config.servers),
            "parallel_tools": _config.execution.parallel_tools,
            "cache_enabled": _config.cache.enabled,
        }

    @app.get("/api/servers")
    async def api_servers():
        if not _client or not _config:
            return []
        agg = _client._aggregator
        try:
            all_tools = await agg.collect_tools()
        except Exception:
            all_tools = []
        result = []
        for s in _config.servers:
            tool_count = sum(1 for t in all_tools if t.server == s.name)
            # Strip sensitive fields: command, args, env, auth
            result.append({
                "name": s.name,
                "transport": s.transport,
                "degraded": agg.is_degraded(s.name),
                "tool_count": tool_count,
            })
        return result

    @app.get("/api/tools")
    async def api_tools():
        if not _client:
            return []
        try:
            tools = await _client.list_tools()
            return [
                {
                    "name": t.name,
                    "full_name": t.full_name,
                    "server": t.server,
                    "description": t.description or "",
                    "schema": t.input_schema,
                }
                for t in tools
            ]
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc), "code": "SERVER_ERROR"}, status_code=500
            )

    @app.get("/api/models")
    async def api_models():
        if not _config:
            return {"models": [], "ollama_url": "", "reachable": False}
        from .adapters.ollama import OllamaAdapter
        ollama = OllamaAdapter(_config.ollama_base_url, _config.default_model)
        try:
            models = await ollama.model_capability_summary()
            return {"models": models, "ollama_url": _config.ollama_base_url, "reachable": True}
        except Exception:
            return {"models": [], "ollama_url": _config.ollama_base_url, "reachable": False}
        finally:
            await ollama.close()

    @app.get("/api/trace")
    async def api_trace():
        return Tracer.load_last()

    @app.get("/api/cache/stats")
    async def api_cache_stats():
        if not _client or not _client._cache:
            return {"hits": 0, "misses": 0, "entries": 0, "hit_rate": 0.0}
        return _client._cache.stats()

    @app.post("/api/run")
    async def api_run(request: Request):
        if not _client:
            return JSONResponse(
                {"error": "Client not connected", "code": "SERVICE_UNAVAILABLE"},
                status_code=503,
            )

        ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(ip):
            return JSONResponse(
                {"error": "Too many requests", "code": "RATE_LIMITED"},
                status_code=429,
                headers={"Retry-After": str(_RATE_WINDOW_S)},
            )

        body = await request.json()
        prompt = body.get("prompt", "").strip()
        session_id = body.get("session_id") or None
        dry_run = body.get("dry_run", False)
        model = body.get("model") or None

        if not prompt:
            return JSONResponse(
                {"error": "prompt is required", "code": "PROMPT_REQUIRED"},
                status_code=400,
            )

        err = _validate_run_body(prompt, session_id)
        if err:
            return JSONResponse(
                {"error": err, "code": "INVALID_INPUT"}, status_code=422
            )

        try:
            async with _run_semaphore:
                result = await _client.run(
                    prompt,
                    session_id=session_id,
                    dry_run=dry_run,
                    model=model,
                )
            trace = Tracer.load_last()
            return {
                "answer": result.answer,
                "tool_calls_made": result.tool_calls_made,
                "cache_hits": result.cache_hits,
                "iterations": result.iterations,
                "total_latency_ms": round(result.total_latency_ms, 1),
                "exit_reason": result.exit_reason,
                "success": result.success,
                "trace": trace,
            }
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc), "code": "SERVER_ERROR"}, status_code=500
            )

    @app.post("/api/run/stream")
    async def api_run_stream(request: Request):
        """SSE endpoint — streams token and tool events as they happen."""
        if not _client:
            return JSONResponse(
                {"error": "Client not connected", "code": "SERVICE_UNAVAILABLE"},
                status_code=503,
            )

        ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(ip):
            return JSONResponse(
                {"error": "Too many requests", "code": "RATE_LIMITED"},
                status_code=429,
                headers={"Retry-After": str(_RATE_WINDOW_S)},
            )

        body = await request.json()
        prompt = body.get("prompt", "").strip()
        session_id = body.get("session_id") or None
        dry_run = body.get("dry_run", False)
        model = body.get("model") or None

        if not prompt:
            return JSONResponse(
                {"error": "prompt is required", "code": "PROMPT_REQUIRED"},
                status_code=400,
            )

        err = _validate_run_body(prompt, session_id)
        if err:
            return JSONResponse(
                {"error": err, "code": "INVALID_INPUT"}, status_code=422
            )

        queue: asyncio.Queue = asyncio.Queue()

        async def run_task() -> None:
            async with _run_semaphore:
                try:
                    result = await _client.run(
                        prompt,
                        session_id=session_id,
                        dry_run=dry_run,
                        model=model,
                        event_queue=queue,
                    )
                    trace = Tracer.load_last()
                    await queue.put({
                        "type": "done",
                        "answer": result.answer,
                        "tool_calls_made": result.tool_calls_made,
                        "cache_hits": result.cache_hits,
                        "iterations": result.iterations,
                        "total_latency_ms": round(result.total_latency_ms, 1),
                        "exit_reason": result.exit_reason,
                        "success": result.success,
                        "trace": trace,
                    })
                except Exception as exc:
                    await queue.put({"type": "error", "message": str(exc), "code": "SERVER_ERROR"})
                finally:
                    await queue.put(None)  # sentinel

        task = asyncio.create_task(run_task())

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=180.0)
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Run timed out', 'code': 'TIMEOUT'})}\n\n"
                        task.cancel()
                        break
                    if item is None:
                        break
                    yield f"data: {json.dumps(item, default=str)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {exc}', 'code': 'SERVER_ERROR'})}\n\n"
                task.cancel()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/config/reload")
    async def api_config_reload():
        """Re-read mcp.json from disk and reconnect all servers."""
        global _client, _config
        if not _client:
            return JSONResponse(
                {"error": "Client not initialized", "code": "SERVICE_UNAVAILABLE"},
                status_code=503,
            )
        async with _config_reload_lock:
            try:
                new_config = AppConfig.load(_config_path)
                await _client.close()
                new_client = MCPClient(config=new_config)
                await new_client.connect()
                new_client._aggregator.invalidate_cache()
                _client = new_client
                _config = new_config
                return {
                    "status": "ok",
                    "servers": len(new_config.servers),
                    "model": new_config.default_model,
                }
            except Exception as exc:
                return JSONResponse(
                    {"error": str(exc), "code": "SERVER_ERROR"}, status_code=500
                )

    @app.get("/api/sessions")
    async def api_sessions():
        if not _client:
            return []
        return _client._session_store.list_sessions_with_meta()

    @app.get("/api/sessions/{session_id}/messages")
    async def api_session_messages(session_id: str):
        err = _validate_session_id(session_id)
        if err:
            return JSONResponse({"error": err, "code": "INVALID_INPUT"}, status_code=422)
        if not _client:
            return {"messages": []}
        messages = _client._session_store.load(session_id)
        display = [m for m in messages if m.get("role") != "system"]
        return {"messages": display}

    @app.delete("/api/sessions/{session_id}")
    async def api_session_delete(session_id: str):
        err = _validate_session_id(session_id)
        if err:
            return JSONResponse({"error": err, "code": "INVALID_INPUT"}, status_code=422)
        if not _client:
            return JSONResponse(
                {"error": "Not connected", "code": "SERVICE_UNAVAILABLE"}, status_code=503
            )
        deleted = _client._session_store.delete(session_id)
        return {"deleted": deleted}

    return app


async def serve(
    config: AppConfig,
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: Path | None = None,
) -> None:
    """Connect client and start uvicorn server."""
    # Load .env so MCP server subprocesses inherit secrets
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(override=False)
    except ImportError:
        pass

    import uvicorn

    client = MCPClient(config=config)
    await client.connect()

    app = create_app(client, config, config_path=config_path)
    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    api_key = config.resolve_dashboard_api_key()
    key_notice = " (API key required)" if api_key else " (no auth — set UMCP_API_KEY to secure)"
    print(f"\n  UMCP Dashboard  ->  http://{host}:{port}{key_notice}\n")

    try:
        await server.serve()
    finally:
        await client.close()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _parse_bearer(auth_header: str) -> str | None:
    """Extract token from 'Bearer <token>' header value."""
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return None
