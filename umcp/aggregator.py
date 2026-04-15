"""Tool aggregator — connects to all transports and merges their tool manifests."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .transports.base import BaseTransport, ToolInfo
from .transports import make_transport

if TYPE_CHECKING:
    from .config import AppConfig, ServerConfig


class ToolAggregator:
    """Manages a pool of transports and provides a unified tool namespace."""

    def __init__(self) -> None:
        self._transports: dict[str, BaseTransport] = {}
        self._tools: dict[str, ToolInfo] = {}       # full_name → ToolInfo
        self._degraded: set[str] = set()            # server names marked degraded

    async def connect_all(self, servers: list["ServerConfig"]) -> list[str]:
        """Connect to all servers. Returns list of server names that failed."""
        failed = []
        for server in servers:
            try:
                transport = make_transport(server)
                await transport.connect()
                self._transports[server.name] = transport
            except Exception as exc:
                failed.append(server.name)
                print(
                    f"[umcp] WARNING: Could not connect to server "
                    f"{server.name!r}: {exc}",
                    flush=True,
                )
        return failed

    async def collect_tools(self) -> list[ToolInfo]:
        """Fetch tool manifests from all connected, non-degraded servers."""
        all_tools: list[ToolInfo] = []
        for name, transport in self._transports.items():
            if name in self._degraded:
                continue
            try:
                tools = await transport.list_tools()
                all_tools.extend(tools)
            except Exception as exc:
                print(
                    f"[umcp] WARNING: Could not fetch tools from {name!r}: {exc}",
                    flush=True,
                )
        self._tools = {t.full_name: t for t in all_tools}
        return all_tools

    def get_tool(self, full_name: str) -> ToolInfo | None:
        return self._tools.get(full_name)

    def get_transport(self, server_name: str) -> BaseTransport | None:
        return self._transports.get(server_name)

    def mark_degraded(self, server_name: str) -> None:
        """Mark a server as degraded — removes its tools from the active pool."""
        self._degraded.add(server_name)
        # Remove its tools from the active tool map
        self._tools = {
            k: v for k, v in self._tools.items()
            if v.server != server_name
        }

    def is_degraded(self, server_name: str) -> bool:
        return server_name in self._degraded

    def active_server_names(self) -> list[str]:
        return [n for n in self._transports if n not in self._degraded]

    def degraded_notice(self) -> str | None:
        """Return an LLM-injectable notice about degraded servers, or None."""
        if not self._degraded:
            return None
        names = ", ".join(f"'{n}'" for n in self._degraded)
        return (
            f"NOTE: The following servers are currently unavailable: {names}. "
            f"Do NOT attempt to call their tools."
        )

    async def close_all(self) -> None:
        for transport in self._transports.values():
            try:
                await transport.close()
            except Exception:
                pass
        self._transports.clear()
