"""Web dashboard server for umcp — FastAPI backend."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .client import MCPClient
from .config import AppConfig
from .trace import Tracer

_STATIC = Path(__file__).parent / "static"

_client: MCPClient | None = None
_config: AppConfig | None = None


def create_app(client: MCPClient, config: AppConfig) -> FastAPI:
    global _client, _config
    _client = client
    _config = config

    app = FastAPI(title="umcp dashboard", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

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
        result = []
        for s in _config.servers:
            tools = []
            try:
                all_tools = await agg.collect_tools()
                tools = [t.name for t in all_tools if t.server == s.name]
            except Exception:
                pass
            result.append({
                "name": s.name,
                "transport": s.transport,
                "location": s.url or s.command or "",
                "degraded": agg.is_degraded(s.name),
                "tool_count": len(tools),
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
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/models")
    async def api_models():
        if not _config:
            return []
        from .adapters.ollama import OllamaAdapter
        ollama = OllamaAdapter(_config.ollama_base_url, _config.default_model)
        try:
            return await ollama.model_capability_summary()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
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
            return JSONResponse({"error": "Client not connected"}, status_code=503)
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        session_id = body.get("session_id") or None
        dry_run = body.get("dry_run", False)
        model = body.get("model") or None

        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        try:
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
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/sessions")
    async def api_sessions():
        if not _client:
            return []
        return _client.list_sessions()

    return app


async def serve(config: AppConfig, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Connect client and start uvicorn server."""
    import uvicorn

    client = MCPClient(config=config)
    await client.connect()

    app = create_app(client, config)
    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)
    print(f"\n  UMCP Dashboard  ->  http://{host}:{port}\n")
    try:
        await server.serve()
    finally:
        await client.close()
