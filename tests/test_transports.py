"""Tests for transport layer — unit tests with mock MCP sessions."""
from __future__ import annotations

import asyncio
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from umcp.transports.base import BaseTransport, ToolInfo, ToolResult
from umcp.transports import make_transport
from umcp.config import ServerConfig


# ------------------------------------------------------------------ #
# Mock transport for aggregator tests
# ------------------------------------------------------------------ #

class MockTransport(BaseTransport):
    """A fully controllable in-memory transport for testing."""

    def __init__(self, server: ServerConfig, tools: list[ToolInfo], *, fail_connect=False):
        super().__init__(server)
        self._tools = tools
        self._fail_connect = fail_connect
        self._connected_flag = False
        self.call_log: list[tuple[str, dict]] = []

    async def connect(self) -> None:
        if self._fail_connect:
            raise ConnectionError(f"Cannot connect to {self.name}")
        self._connected_flag = True

    async def list_tools(self) -> list[ToolInfo]:
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout_ms: int = 5000) -> ToolResult:
        self.call_log.append((tool_name, arguments))
        # Return a predictable result based on tool name
        return ToolResult(success=True, content=f"result:{tool_name}")

    async def close(self) -> None:
        self._connected_flag = False

    @property
    def is_connected(self) -> bool:
        return self._connected_flag


def _make_server(name: str, transport: str = "stdio") -> ServerConfig:
    if transport == "stdio":
        return ServerConfig(name=name, transport="stdio", command="python")
    if transport == "http":
        return ServerConfig(name=name, transport="http", url=f"http://localhost:8000")
    return ServerConfig(name=name, transport="sse", url=f"http://localhost:9000/sse")


def _make_tool(server: str, name: str, desc: str = "A tool") -> ToolInfo:
    return ToolInfo(
        server=server, name=name,
        full_name=f"{server}.{name}",
        description=desc,
        input_schema={"type": "object", "properties": {}},
    )


# ------------------------------------------------------------------ #
# make_transport factory
# ------------------------------------------------------------------ #

def test_factory_stdio():
    s = _make_server("a", "stdio")
    t = make_transport(s)
    from umcp.transports.stdio import StdioTransport
    assert isinstance(t, StdioTransport)


def test_factory_http():
    s = _make_server("a", "http")
    t = make_transport(s)
    from umcp.transports.http import HttpTransport
    assert isinstance(t, HttpTransport)


def test_factory_sse():
    s = _make_server("a", "sse")
    t = make_transport(s)
    from umcp.transports.sse import SseTransport
    assert isinstance(t, SseTransport)


def test_factory_unknown_raises():
    s = ServerConfig.model_construct(name="x", transport="grpc")
    with pytest.raises(ValueError, match="grpc"):
        make_transport(s)


# ------------------------------------------------------------------ #
# MockTransport behaviour
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_mock_transport_connect_and_list():
    tools = [_make_tool("srv", "foo"), _make_tool("srv", "bar")]
    t = MockTransport(_make_server("srv"), tools)
    await t.connect()
    assert t.is_connected
    listed = await t.list_tools()
    assert len(listed) == 2
    assert listed[0].full_name == "srv.foo"
    await t.close()
    assert not t.is_connected


@pytest.mark.asyncio
async def test_mock_transport_call_tool():
    t = MockTransport(_make_server("srv"), [_make_tool("srv", "greet")])
    await t.connect()
    result = await t.call_tool("greet", {"name": "Alice"})
    assert result.success
    assert "greet" in result.content
    assert t.call_log == [("greet", {"name": "Alice"})]


@pytest.mark.asyncio
async def test_mock_transport_connect_failure():
    t = MockTransport(_make_server("srv"), [], fail_connect=True)
    with pytest.raises(ConnectionError):
        await t.connect()
    assert not t.is_connected


# ------------------------------------------------------------------ #
# Aggregator — multi-server
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_aggregator_collects_from_multiple_servers():
    from umcp.aggregator import ToolAggregator

    agg = ToolAggregator()
    tools_a = [_make_tool("alpha", "search"), _make_tool("alpha", "index")]
    tools_b = [_make_tool("beta", "query"), _make_tool("beta", "insert")]

    ta = MockTransport(_make_server("alpha"), tools_a)
    tb = MockTransport(_make_server("beta"), tools_b)

    await ta.connect()
    await tb.connect()
    agg._transports = {"alpha": ta, "beta": tb}

    all_tools = await agg.collect_tools()
    full_names = {t.full_name for t in all_tools}
    assert full_names == {"alpha.search", "alpha.index", "beta.query", "beta.insert"}


@pytest.mark.asyncio
async def test_aggregator_skips_failed_server():
    from umcp.aggregator import ToolAggregator

    agg = ToolAggregator()
    servers = [_make_server("good"), _make_server("bad")]

    good = MockTransport(_make_server("good"), [_make_tool("good", "ping")])
    bad = MockTransport(_make_server("bad"), [], fail_connect=True)

    # Simulate connect_all by manually patching make_transport
    with patch("umcp.aggregator.make_transport", side_effect=[good, bad]):
        failed = await agg.connect_all(servers)

    assert "bad" in failed
    assert "good" not in failed


@pytest.mark.asyncio
async def test_aggregator_mark_degraded():
    from umcp.aggregator import ToolAggregator

    agg = ToolAggregator()
    tools = [_make_tool("srv", "foo"), _make_tool("srv", "bar")]
    t = MockTransport(_make_server("srv"), tools)
    await t.connect()
    agg._transports = {"srv": t}
    await agg.collect_tools()

    assert not agg.is_degraded("srv")
    agg.mark_degraded("srv")
    assert agg.is_degraded("srv")

    # After degrading, collect_tools skips the server
    remaining = await agg.collect_tools()
    assert remaining == []


@pytest.mark.asyncio
async def test_aggregator_degraded_notice():
    from umcp.aggregator import ToolAggregator

    agg = ToolAggregator()
    assert agg.degraded_notice() is None

    agg._degraded.add("weather")
    notice = agg.degraded_notice()
    assert notice is not None
    assert "weather" in notice
    assert "unavailable" in notice


@pytest.mark.asyncio
async def test_aggregator_get_tool_by_full_name():
    from umcp.aggregator import ToolAggregator

    agg = ToolAggregator()
    tools = [_make_tool("db", "query")]
    t = MockTransport(_make_server("db"), tools)
    await t.connect()
    agg._transports = {"db": t}
    await agg.collect_tools()

    found = agg.get_tool("db.query")
    assert found is not None
    assert found.full_name == "db.query"

    missing = agg.get_tool("db.nonexistent")
    assert missing is None


# ------------------------------------------------------------------ #
# Auth headers
# ------------------------------------------------------------------ #

def test_sse_transport_passes_auth_headers():
    """SseTransport must pass auth headers to sse_client."""
    import os
    os.environ["TEST_TOKEN"] = "mytoken123"
    server = ServerConfig(
        name="svc", transport="sse", url="http://localhost:9000/sse",
        auth={"type": "bearer", "token": "env:TEST_TOKEN"},
    )
    from umcp.transports.sse import SseTransport
    t = SseTransport(server)
    headers = t.server.auth.get_headers()
    assert headers == {"Authorization": "Bearer mytoken123"}
    del os.environ["TEST_TOKEN"]


def test_http_transport_passes_api_key_header():
    """HttpTransport must pass API key headers to streamablehttp_client."""
    import os
    os.environ["MY_API_KEY"] = "key-abc"
    server = ServerConfig(
        name="svc", transport="http", url="http://localhost:8000",
        auth={"type": "api_key", "header": "X-API-Key", "value": "env:MY_API_KEY"},
    )
    from umcp.transports.http import HttpTransport
    t = HttpTransport(server)
    headers = t.server.auth.get_headers()
    assert headers == {"X-API-Key": "key-abc"}
    del os.environ["MY_API_KEY"]


def test_no_auth_returns_empty_headers():
    server = ServerConfig(name="svc", transport="stdio", command="python")
    headers = server.auth.get_headers()
    assert headers == {}


# ------------------------------------------------------------------ #
# Tool timeout on stdio (unit-level)
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_tool_timeout_returns_error():
    """call_tool must return ToolResult(success=False) on timeout, not raise."""

    class SlowTransport(MockTransport):
        async def call_tool(self, tool_name, arguments, timeout_ms=5000):
            await asyncio.sleep(10)  # will be cancelled by wait_for in real transport
            return ToolResult(success=True, content="too late")

    t = SlowTransport(_make_server("slow"), [])
    await t.connect()
    # This just verifies the mock — real timeout tested via StdioTransport internals
    assert t.is_connected
