from .base import BaseTransport, ToolInfo, ToolResult
from .stdio import StdioTransport
from .http import HttpTransport
from .sse import SseTransport

__all__ = [
    "BaseTransport", "ToolInfo", "ToolResult",
    "StdioTransport", "HttpTransport", "SseTransport",
]


def make_transport(server) -> BaseTransport:
    """Factory: create the right transport for a ServerConfig."""
    if server.transport == "stdio":
        return StdioTransport(server)
    if server.transport == "http":
        return HttpTransport(server)
    if server.transport == "sse":
        return SseTransport(server)
    raise ValueError(f"Unknown transport: {server.transport!r}")
