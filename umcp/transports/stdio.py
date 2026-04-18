"""stdio transport — spawns a subprocess and communicates over stdin/stdout."""
from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from .base import BaseTransport, ToolInfo, ToolResult
from ..config import ServerConfig


class StdioTransport(BaseTransport):
    """MCP transport that communicates with a locally-spawned subprocess."""

    def __init__(self, server: ServerConfig) -> None:
        super().__init__(server)
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

    async def connect(self) -> None:
        merged_env = {**os.environ, **self.server.env} if self.server.env else None
        params = StdioServerParameters(
            command=self.server.command,
            args=self.server.args,
            env=merged_env,
        )
        try:
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            self._connected = True
        except Exception:
            # Clean up the exit stack in THIS task so anyio cancel scopes are
            # exited in the same task they were entered — avoids the cross-task
            # RuntimeError when the async generator is GC'd during shutdown.
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            raise

    async def list_tools(self) -> list[ToolInfo]:
        assert self._session is not None, "StdioTransport not connected"
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
        assert self._session is not None, "StdioTransport not connected"
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout_ms / 1000.0,
            )
            if result.isError:
                error_text = _extract_text(result.content)
                return ToolResult(success=False, content=None, error=error_text)
            return ToolResult(success=True, content=_extract_content(result.content))
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=None,
                error=f"Tool '{tool_name}' timed out after {timeout_ms}ms",
            )
        except Exception as exc:
            return ToolResult(success=False, content=None, error=str(exc))

    async def close(self) -> None:
        # On Windows, anyio subprocess pipe teardown can raise CancelledError or
        # ClosedResourceError during event-loop shutdown — suppress both safely.
        try:
            await self._exit_stack.aclose()
        except BaseException:
            pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


def _extract_text(content_list: list) -> str:
    """Extract text from content blocks as a single string."""
    parts = []
    for block in content_list:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _extract_content(content_list: list) -> Any:
    """Extract content from MCP result blocks.

    Single text block → plain string.
    Multiple blocks → list of strings / raw values.
    """
    if not content_list:
        return None
    if len(content_list) == 1:
        block = content_list[0]
        if hasattr(block, "text"):
            return block.text
        if hasattr(block, "data"):
            return block.data
        return str(block)
    parts = []
    for block in content_list:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(block.data)
        else:
            parts.append(str(block))
    return parts
