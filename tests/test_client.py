"""Tests for MCPClient SDK — end-to-end with mock components."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umcp.client import MCPClient
from umcp.config import AppConfig
from umcp.loop import LoopResult
from umcp.transports.base import ToolInfo, ToolResult


def _make_tool(server: str = "mock", name: str = "ping") -> ToolInfo:
    return ToolInfo(
        server=server,
        name=name,
        full_name=f"{server}.{name}",
        description="Ping tool",
        input_schema={"type": "object", "properties": {}},
    )


def _make_loop_result(answer: str = "Done", success: bool = True) -> LoopResult:
    return LoopResult(
        answer=answer,
        tool_calls_made=1,
        cache_hits=0,
        iterations=2,
        total_latency_ms=123.0,
        exit_reason="completed",
        success=success,
        messages=[{"role": "user", "content": "test"}, {"role": "assistant", "content": answer}],
    )


# ------------------------------------------------------------------ #
# Test: client run() delegates to AgentLoop
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_client_run_delegates_to_loop(tmp_path):
    cfg = AppConfig()
    cfg.session.storage_path = str(tmp_path)
    client = MCPClient(config=cfg)

    # Mock all internals
    client._connected = True
    client._ollama = MagicMock()
    client._aggregator = MagicMock()
    client._aggregator.collect_tools = AsyncMock(return_value=[])
    client._aggregator.close_all = AsyncMock()
    client._aggregator.degraded_notice = MagicMock(return_value=None)

    # Provide a mock tracer so save_last() doesn't fail
    from umcp.trace import Tracer, new_trace_id
    client._tracer = Tracer(session_id=new_trace_id(), enabled=False)

    expected = _make_loop_result("The answer is 42.")

    with patch("umcp.client.AgentLoop") as MockLoop:
        loop_instance = MagicMock()
        loop_instance.run = AsyncMock(return_value=expected)
        MockLoop.return_value = loop_instance

        result = await client.run("What is the answer?")

    assert result.answer == "The answer is 42."
    assert result.success


# ------------------------------------------------------------------ #
# Test: client list_tools filters by glob
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_client_list_tools_filter():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    client._connected = True

    tools = [
        _make_tool("weather", "forecast"),
        _make_tool("db", "query"),
        _make_tool("weather", "alerts"),
    ]
    client._aggregator = MagicMock()
    client._aggregator.collect_tools = AsyncMock(return_value=tools)

    result = await client.list_tools(filter="weather.*")
    assert len(result) == 2
    assert all(t.server == "weather" for t in result)


# ------------------------------------------------------------------ #
# Test: client list_tools no filter returns all
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_client_list_tools_no_filter():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    client._connected = True

    tools = [_make_tool("a", "x"), _make_tool("b", "y")]
    client._aggregator = MagicMock()
    client._aggregator.collect_tools = AsyncMock(return_value=tools)

    result = await client.list_tools()
    assert len(result) == 2


# ------------------------------------------------------------------ #
# Test: call_tool validates args and calls transport
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_client_call_tool_success():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    client._connected = True

    tool = _make_tool("mock", "ping")
    transport = MagicMock()
    transport.call_tool = AsyncMock(return_value=ToolResult(success=True, content="pong"))

    client._aggregator = MagicMock()
    client._aggregator.collect_tools = AsyncMock(return_value=[tool])
    client._aggregator.get_transport = MagicMock(return_value=transport)

    result = await client.call_tool("mock.ping", {})
    assert result == "pong"


@pytest.mark.asyncio
async def test_client_call_tool_unknown_raises():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    client._connected = True

    client._aggregator = MagicMock()
    client._aggregator.collect_tools = AsyncMock(return_value=[])

    with pytest.raises(ValueError, match="not found"):
        await client.call_tool("ghost.tool", {})


# ------------------------------------------------------------------ #
# Test: add_server adds to config
# ------------------------------------------------------------------ #

def test_client_add_server():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    client.add_server({
        "name": "new_server",
        "transport": "http",
        "url": "http://localhost:9000",
    })
    assert any(s.name == "new_server" for s in client._config.servers)


# ------------------------------------------------------------------ #
# Test: plugin hook registration and dispatch
# ------------------------------------------------------------------ #

def test_client_plugin_system_prompt():
    cfg = AppConfig()
    client = MCPClient(config=cfg)

    @client.plugin("system_prompt")
    def add_rule(base: str) -> str:
        return base + "\nAlways respond in Japanese."

    hook = client._plugin_registry.get("system_prompt")
    assert hook is not None
    result = hook("Base prompt.")
    assert "Japanese" in result


def test_client_plugin_tool_filter():
    cfg = AppConfig()
    client = MCPClient(config=cfg)

    @client.plugin("tool_filter")
    def remove_dangerous(tools, prompt):
        return [t for t in tools if "dangerous" not in t.name]

    tools = [_make_tool("a", "safe"), _make_tool("a", "dangerous_op")]
    hook = client._plugin_registry.get("tool_filter")
    filtered = hook(tools, "do stuff")
    assert len(filtered) == 1
    assert filtered[0].name == "safe"


# ------------------------------------------------------------------ #
# Test: on_tool_call hook registration
# ------------------------------------------------------------------ #

def test_client_on_tool_call_registers():
    cfg = AppConfig()
    client = MCPClient(config=cfg)
    called = []

    @client.on_tool_call
    def my_hook(entry):
        called.append(entry)

    assert my_hook in client._on_tool_call


# ------------------------------------------------------------------ #
# Test: session storage path passed through
# ------------------------------------------------------------------ #

def test_client_session_store_uses_config_path(tmp_path):
    cfg = AppConfig()
    cfg.session.storage_path = str(tmp_path)
    client = MCPClient(config=cfg)
    # SessionStore should have been initialized with the config path
    assert str(tmp_path) in str(client._session_store._dir)


# ------------------------------------------------------------------ #
# Test: list_sessions and delete_session
# ------------------------------------------------------------------ #

def test_client_list_and_delete_sessions(tmp_path):
    cfg = AppConfig()
    cfg.session.storage_path = str(tmp_path)
    client = MCPClient(config=cfg)

    client._session_store.save("session-alpha", [{"role": "user", "content": "hi"}])
    client._session_store.save("session-beta", [{"role": "user", "content": "bye"}])

    sessions = client.list_sessions()
    assert "session-alpha" in sessions
    assert "session-beta" in sessions

    deleted = client.delete_session("session-alpha")
    assert deleted
    assert "session-alpha" not in client.list_sessions()
