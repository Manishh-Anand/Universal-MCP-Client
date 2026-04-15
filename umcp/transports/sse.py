"""SSE transport — MCP over HTTP + Server-Sent Events."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from .base import BaseTransport, ToolInfo, ToolResult
from ..config import ServerConfig
from .stdio import _extract_content, _extract_text


async def _resolve_oauth2_token(server: ServerConfig) -> str | None:
    """Fetch an OAuth2 client-credentials token if configured.

    Returns the bearer token string, or None if auth type is not oauth2.
    """
    if server.auth.type != "oauth2":
        return None
    import httpx
    auth = server.auth
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                auth.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": auth._resolve(auth.client_id),
                    "client_secret": auth._resolve(auth.client_secret),
                },
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
    except Exception as exc:
        raise RuntimeError(f"OAuth2 token fetch failed for server '{server.name}': {exc}") from exc


class SseTransport(BaseTransport):
    """MCP transport that connects to a remote server via Server-Sent Events.

    The server must expose an SSE endpoint (e.g. http://host:port/sse).
    Auth headers (bearer token, API key, OAuth2) are forwarded on both the SSE
    stream connection and all message POSTs.
    """

    def __init__(self, server: ServerConfig) -> None:
        super().__init__(server)
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

    async def connect(self) -> None:
        # Resolve OAuth2 token if needed
        if self.server.auth.type == "oauth2":
            token = await _resolve_oauth2_token(self.server)
            auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
        else:
            auth_headers = self.server.auth.get_headers()

        read, write = await self._exit_stack.enter_async_context(
            sse_client(
                url=self.server.url,
                headers=auth_headers or None,
                timeout=5.0,
                sse_read_timeout=300.0,
            )
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        self._connected = True

    async def list_tools(self) -> list[ToolInfo]:
        assert self._session is not None, "SseTransport not connected"
        response = await self._session.list_tools()
        return [
            ToolInfo(
                server=self.name,
                name=t.name,
                full_name=f"{self.name}.{t.name}",
                description=t.description or "",
                input_schema=t.inputSchema if t.inputSchema else {
                    "type": "object", "properties": {}
                },
            )
            for t in response.tools
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_ms: int = 5000,
    ) -> ToolResult:
        assert self._session is not None, "SseTransport not connected"
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout_ms / 1000.0,
            )
            if result.isError:
                return ToolResult(
                    success=False, content=None,
                    error=_extract_text(result.content),
                )
            return ToolResult(success=True, content=_extract_content(result.content))
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, content=None,
                error=f"Tool '{tool_name}' timed out after {timeout_ms}ms",
            )
        except Exception as exc:
            return ToolResult(success=False, content=None, error=str(exc))

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
