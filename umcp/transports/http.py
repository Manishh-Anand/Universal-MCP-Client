"""HTTP transport — MCP over Streamable HTTP."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .base import BaseTransport, ToolInfo, ToolResult
from ..config import ServerConfig
from .stdio import _extract_content, _extract_text
from .sse import _resolve_oauth2_token


class HttpTransport(BaseTransport):
    """MCP transport that connects to a remote server via Streamable HTTP.

    The server must expose a streamable HTTP endpoint (e.g. http://host:port/mcp).
    Auth headers (bearer token, API key, OAuth2) are forwarded on all requests.

    Note: streamablehttp_client yields (read, write, get_session_id) — a 3-tuple.
    We discard get_session_id as it's not needed by the client side.
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

        # streamablehttp_client yields (read, write, get_session_id)
        read, write, _get_session_id = await self._exit_stack.enter_async_context(
            streamablehttp_client(
                url=self.server.url,
                headers=auth_headers or None,
                timeout=30.0,
                sse_read_timeout=300.0,
            )
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        self._connected = True

    async def list_tools(self) -> list[ToolInfo]:
        assert self._session is not None, "HttpTransport not connected"
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
        assert self._session is not None, "HttpTransport not connected"
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
