"""Tests for tool namespace aggregation and conflict resolution."""
from __future__ import annotations

import pytest
from umcp.aggregator import ToolAggregator
from umcp.transports.base import ToolInfo, ToolResult, BaseTransport
from umcp.config import ServerConfig
from typing import Any


class FixedTransport(BaseTransport):
    """Transport that returns a fixed list of tools."""

    def __init__(self, server: ServerConfig, tools: list[ToolInfo]):
        super().__init__(server)
        self._tools = tools
        self._connected_flag = False

    async def connect(self):
        self._connected_flag = True

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, tool_name, arguments, timeout_ms=5000):
        return ToolResult(success=True, content=f"{self.name}.{tool_name}:ok")

    async def close(self):
        self._connected_flag = False

    @property
    def is_connected(self):
        return self._connected_flag


def _tool(server, name):
    return ToolInfo(
        server=server, name=name, full_name=f"{server}.{name}",
        description=f"{name} tool", input_schema={},
    )


def _server(name):
    return ServerConfig(name=name, transport="stdio", command="python")


@pytest.mark.asyncio
async def test_two_servers_unique_names():
    agg = ToolAggregator()
    agg._transports = {
        "a": FixedTransport(_server("a"), [_tool("a", "foo"), _tool("a", "bar")]),
        "b": FixedTransport(_server("b"), [_tool("b", "baz")]),
    }
    for t in agg._transports.values():
        await t.connect()

    tools = await agg.collect_tools()
    assert len(tools) == 3
    full_names = {t.full_name for t in tools}
    assert full_names == {"a.foo", "a.bar", "b.baz"}


@pytest.mark.asyncio
async def test_same_bare_name_on_two_servers_disambiguated():
    """Two servers exposing 'search' should both appear as server.search."""
    agg = ToolAggregator()
    agg._transports = {
        "alpha": FixedTransport(_server("alpha"), [_tool("alpha", "search")]),
        "beta": FixedTransport(_server("beta"), [_tool("beta", "search")]),
    }
    for t in agg._transports.values():
        await t.connect()

    tools = await agg.collect_tools()
    full_names = {t.full_name for t in tools}
    # Both are preserved — no collision because prefixing is always on
    assert "alpha.search" in full_names
    assert "beta.search" in full_names
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_degraded_server_excluded_from_tools():
    agg = ToolAggregator()
    agg._transports = {
        "live": FixedTransport(_server("live"), [_tool("live", "ping")]),
        "dead": FixedTransport(_server("dead"), [_tool("dead", "crash")]),
    }
    for t in agg._transports.values():
        await t.connect()

    agg.mark_degraded("dead")
    tools = await agg.collect_tools()
    full_names = {t.full_name for t in tools}
    assert "live.ping" in full_names
    assert "dead.crash" not in full_names


@pytest.mark.asyncio
async def test_call_tool_routed_to_correct_server():
    agg = ToolAggregator()
    ta = FixedTransport(_server("alpha"), [_tool("alpha", "foo")])
    tb = FixedTransport(_server("beta"), [_tool("beta", "bar")])
    await ta.connect()
    await tb.connect()
    agg._transports = {"alpha": ta, "beta": tb}
    await agg.collect_tools()

    transport = agg.get_transport("alpha")
    assert transport is ta

    transport = agg.get_transport("beta")
    assert transport is tb


@pytest.mark.asyncio
async def test_active_server_names():
    agg = ToolAggregator()
    agg._transports = {
        "a": FixedTransport(_server("a"), []),
        "b": FixedTransport(_server("b"), []),
        "c": FixedTransport(_server("c"), []),
    }
    agg.mark_degraded("b")
    assert sorted(agg.active_server_names()) == ["a", "c"]


@pytest.mark.asyncio
async def test_close_all_disconnects():
    agg = ToolAggregator()
    ta = FixedTransport(_server("a"), [])
    tb = FixedTransport(_server("b"), [])
    await ta.connect()
    await tb.connect()
    agg._transports = {"a": ta, "b": tb}

    await agg.close_all()
    assert not ta.is_connected
    assert not tb.is_connected
    assert agg._transports == {}
