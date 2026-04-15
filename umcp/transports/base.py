"""Abstract transport interface — all transports implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import ServerConfig


@dataclass
class ToolInfo:
    """Normalized tool descriptor from any MCP server."""
    server: str           # server name (e.g. "weather")
    name: str             # bare tool name from server (e.g. "get_forecast")
    full_name: str        # prefixed name (e.g. "weather.get_forecast")
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    content: Any
    error: str | None = None


class BaseTransport(ABC):
    """Common interface every transport must implement.

    Lifecycle:
        transport = SomeTransport(server_config)
        await transport.connect()
        tools = await transport.list_tools()
        result = await transport.call_tool("tool_name", {"arg": "val"})
        await transport.close()
    """

    def __init__(self, server: "ServerConfig") -> None:
        self.server = server
        self.name: str = server.name

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and initialize the MCP session."""

    @abstractmethod
    async def list_tools(self) -> list[ToolInfo]:
        """Return all tools this server exposes (not yet prefixed)."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_ms: int = 5000,
    ) -> ToolResult:
        """Execute a tool by bare name (no server prefix)."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up all connections and subprocesses."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport has an active connection."""
