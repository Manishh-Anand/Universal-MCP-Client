"""Tests for AgentLoop — mock Ollama + mock transports.

These tests verify the full agent loop logic without real MCP servers or Ollama.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umcp.cache import ResponseCache
from umcp.config import AppConfig, CacheConfig, ExecutionConfig
from umcp.loop import AgentLoop, LoopResult, _find_tool, _normalize_tool_calls
from umcp.trace import Tracer, new_trace_id
from umcp.transports.base import ToolInfo, ToolResult


# ------------------------------------------------------------------ #
# Helpers / Factories
# ------------------------------------------------------------------ #

def _make_tool(server: str = "weather", name: str = "get_forecast") -> ToolInfo:
    return ToolInfo(
        server=server,
        name=name,
        full_name=f"{server}.{name}",
        description="Get weather forecast",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": []},
    )


def _make_config(**kwargs) -> AppConfig:
    cfg = AppConfig()
    if "max_iterations" in kwargs:
        cfg.execution.max_iterations = kwargs["max_iterations"]
    if "parallel_tools" in kwargs:
        cfg.execution.parallel_tools = kwargs["parallel_tools"]
    cfg.cache = CacheConfig(enabled=False)
    return cfg


def _make_tracer() -> Tracer:
    return Tracer(session_id="test-sess", enabled=False)


def _make_aggregator(tools: list[ToolInfo], tool_result: ToolResult) -> MagicMock:
    agg = MagicMock()
    agg.collect_tools = AsyncMock(return_value=tools)
    agg.degraded_notice = MagicMock(return_value=None)

    transport = MagicMock()
    transport.call_tool = AsyncMock(return_value=tool_result)
    agg.get_transport = MagicMock(return_value=transport)
    agg.is_degraded = MagicMock(return_value=False)
    return agg


def _make_ollama(responses: list[dict]) -> MagicMock:
    """Create mock Ollama that returns responses in sequence."""
    ollama = MagicMock()
    ollama.check_capability = AsyncMock(return_value=True)
    ollama.chat_once = AsyncMock(side_effect=responses)
    return ollama


def _tool_call_response(tool_name: str, args: dict) -> dict:
    """Build an Ollama response dict simulating a tool call."""
    return {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": tool_name, "arguments": args}}],
        }
    }


def _text_response(text: str) -> dict:
    """Build an Ollama response dict with a plain text answer."""
    return {"message": {"content": text, "tool_calls": []}}


# ------------------------------------------------------------------ #
# Test: simple completed run
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_completes_with_final_answer():
    tool = _make_tool()
    agg = _make_aggregator([tool], ToolResult(success=True, content="22C sunny"))
    ollama = _make_ollama([_text_response("The weather is sunny!")])

    loop = AgentLoop(
        config=_make_config(),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    result = await loop.run("What's the weather in Tokyo?")
    assert result.success
    assert result.exit_reason == "completed"
    assert "sunny" in result.answer


# ------------------------------------------------------------------ #
# Test: one tool call then final answer
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_calls_tool_then_answers():
    tool = _make_tool()
    agg = _make_aggregator([tool], ToolResult(success=True, content="22C sunny"))
    ollama = _make_ollama([
        _tool_call_response("weather.get_forecast", {"city": "Tokyo"}),
        _text_response("The weather in Tokyo is 22C and sunny."),
    ])

    loop = AgentLoop(
        config=_make_config(),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    result = await loop.run("What's the weather in Tokyo?")
    assert result.success
    assert result.tool_calls_made == 1
    assert result.iterations == 2


# ------------------------------------------------------------------ #
# Test: hallucinated tool name causes re-prompt
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_handles_hallucinated_tool():
    tool = _make_tool()
    agg = _make_aggregator([tool], ToolResult(success=True, content="ok"))
    ollama = _make_ollama([
        _tool_call_response("weather.fake_tool", {}),   # hallucinated
        _text_response("I cannot complete the task."),
    ])

    loop = AgentLoop(
        config=_make_config(max_iterations=5),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    result = await loop.run("Do something")
    # Should still succeed (got a final answer after hallucination reprompt)
    assert result.success or result.exit_reason in ("completed", "max_iterations")


# ------------------------------------------------------------------ #
# Test: max iterations guard
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_max_iterations():
    tool = _make_tool()
    agg = _make_aggregator([tool], ToolResult(success=True, content="ok"))

    # Always returns tool call — never final answer
    tool_responses = [
        _tool_call_response("weather.get_forecast", {"city": "Tokyo"})
        for _ in range(10)
    ]
    ollama = _make_ollama(tool_responses)

    loop = AgentLoop(
        config=_make_config(max_iterations=3),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    result = await loop.run("Do many things")
    assert not result.success
    assert result.exit_reason == "max_iterations"


# ------------------------------------------------------------------ #
# Test: dry run
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_dry_run_does_not_execute_tool(capsys):
    tool = _make_tool()
    transport = MagicMock()
    transport.call_tool = AsyncMock(return_value=ToolResult(success=True, content="real result"))

    agg = MagicMock()
    agg.collect_tools = AsyncMock(return_value=[tool])
    agg.degraded_notice = MagicMock(return_value=None)
    agg.get_transport = MagicMock(return_value=transport)
    agg.is_degraded = MagicMock(return_value=False)

    ollama = _make_ollama([
        _tool_call_response("weather.get_forecast", {"city": "Tokyo"}),
        _text_response("Done."),
    ])

    loop = AgentLoop(
        config=_make_config(),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    loop._dry_run = True
    result = await loop.run("Get weather", dry_run=True)

    # Tool should NOT have been called
    transport.call_tool.assert_not_called()
    captured = capsys.readouterr()
    assert "[DRY RUN]" in captured.out


# ------------------------------------------------------------------ #
# Test: cache hit skips tool execution
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_cache_hit_skips_transport():
    tool = _make_tool()
    transport = MagicMock()
    transport.call_tool = AsyncMock(return_value=ToolResult(success=True, content="fresh"))

    agg = MagicMock()
    agg.collect_tools = AsyncMock(return_value=[tool])
    agg.degraded_notice = MagicMock(return_value=None)
    agg.get_transport = MagicMock(return_value=transport)
    agg.is_degraded = MagicMock(return_value=False)

    ollama = _make_ollama([
        _tool_call_response("weather.get_forecast", {"city": "Tokyo"}),
        _text_response("Cached weather!"),
    ])

    cache_config = CacheConfig(enabled=True, ttl_seconds=60)
    cache = ResponseCache(cache_config)
    # Pre-populate cache
    key = cache.make_key("weather.get_forecast", {"city": "Tokyo"}, tool.input_schema)
    cache._mem_set(key, "22C and cached")

    loop = AgentLoop(
        config=_make_config(),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=cache,
    )
    result = await loop.run("Get weather", cache_enabled=True)
    transport.call_tool.assert_not_called()
    assert result.cache_hits == 1


# ------------------------------------------------------------------ #
# Test: _find_tool helper
# ------------------------------------------------------------------ #

def test_find_tool_found():
    tools = [_make_tool("weather", "forecast"), _make_tool("db", "query")]
    found = _find_tool(tools, "weather.forecast")
    assert found is not None
    assert found.full_name == "weather.forecast"


def test_find_tool_not_found():
    tools = [_make_tool("weather", "forecast")]
    assert _find_tool(tools, "db.query") is None


# ------------------------------------------------------------------ #
# Test: _normalize_tool_calls helper
# ------------------------------------------------------------------ #

def test_normalize_native_tool_calls():
    native = [{"function": {"name": "weather.forecast", "arguments": {"city": "Tokyo"}}}]
    result = _normalize_tool_calls(native, [])
    assert result == [("weather.forecast", {"city": "Tokyo"})]


def test_normalize_string_arguments():
    """String-encoded JSON arguments should be parsed."""
    import json
    native = [{"function": {"name": "a.b", "arguments": json.dumps({"x": 1})}}]
    result = _normalize_tool_calls(native, [])
    assert result[0][1] == {"x": 1}


# ------------------------------------------------------------------ #
# Test: messages returned in LoopResult (chat continuity)
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_loop_returns_messages_for_continuity():
    tool = _make_tool()
    agg = _make_aggregator([tool], ToolResult(success=True, content="22C"))
    ollama = _make_ollama([_text_response("Sunny!")])

    loop = AgentLoop(
        config=_make_config(),
        aggregator=agg,
        ollama=ollama,
        tracer=_make_tracer(),
        cache=ResponseCache(CacheConfig(enabled=False)),
    )
    result = await loop.run("Weather?")
    assert isinstance(result.messages, list)
    assert len(result.messages) >= 2  # system + user + at least reply
