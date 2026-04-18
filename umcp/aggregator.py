"""Tool aggregator — connects to all transports and merges their tool manifests."""
from __future__ import annotations

import asyncio
import time as _time
from typing import TYPE_CHECKING

from .log import get_logger
from .transports.base import BaseTransport, ToolInfo
from .transports import make_transport

_log = get_logger()

if TYPE_CHECKING:
    from .config import AppConfig, ServerConfig


class ToolAggregator:
    """Manages a pool of transports and provides a unified tool namespace."""

    _CACHE_TTL = 30.0  # seconds

    def __init__(self) -> None:
        self._transports: dict[str, BaseTransport] = {}
        self._tools: dict[str, ToolInfo] = {}       # full_name → ToolInfo
        self._degraded: set[str] = set()            # server names marked degraded
        self._cached_tools: list[ToolInfo] = []
        self._cache_time: float = 0.0

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
                _log.warning("server_connect_failed", server=server.name, error=str(exc))
        return failed

    async def collect_tools(self, force: bool = False) -> list[ToolInfo]:
        """Fetch tool manifests from all connected, non-degraded servers.

        Results are cached for _CACHE_TTL seconds. Pass force=True to bypass.
        """
        now = _time.monotonic()
        if not force and self._cached_tools and (now - self._cache_time) < self._CACHE_TTL:
            return list(self._cached_tools)

        all_tools: list[ToolInfo] = []
        for name, transport in self._transports.items():
            if name in self._degraded:
                continue
            try:
                tools = await transport.list_tools()
                all_tools.extend(tools)
            except Exception as exc:
                _log.warning("tool_fetch_failed", server=name, error=str(exc))
        self._tools = {t.full_name: t for t in all_tools}
        self._cached_tools = all_tools
        self._cache_time = now
        return all_tools

    def invalidate_cache(self) -> None:
        """Force next collect_tools() call to re-fetch from all servers."""
        self._cache_time = 0.0
        self._cached_tools = []

    def get_tool(self, full_name: str) -> ToolInfo | None:
        return self._tools.get(full_name)

    def get_transport(self, server_name: str) -> BaseTransport | None:
        return self._transports.get(server_name)

    def mark_degraded(self, server_name: str) -> None:
        """Mark a server as degraded — removes its tools from the active pool."""
        self._degraded.add(server_name)
        self._tools = {
            k: v for k, v in self._tools.items()
            if v.server != server_name
        }
        self.invalidate_cache()

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
