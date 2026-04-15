"""Tests for fallback tool call parser."""
import pytest
from umcp.adapters.fallback import parse_tool_calls, strip_tool_call_blocks, auto_repair_json


def test_parses_single_tool_call():
    text = '<tool_call>\n{"name": "weather.get_forecast", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "weather.get_forecast"
    assert calls[0].arguments == {"city": "Tokyo"}


def test_parses_multiple_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        " some text "
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "a"
    assert calls[1].name == "b"


def test_no_calls_returns_empty():
    calls = parse_tool_calls("This is just a plain answer with no tool calls.")
    assert calls == []


def test_strips_tool_call_blocks():
    text = 'Before <tool_call>{"name": "x", "arguments": {}}</tool_call> After'
    stripped = strip_tool_call_blocks(text)
    assert "<tool_call>" not in stripped
    assert "Before" in stripped
    assert "After" in stripped


def test_auto_repair_trailing_comma():
    raw = '{"name": "x", "arguments": {"a": 1,}}'
    repaired = auto_repair_json(raw)
    import json
    parsed = json.loads(repaired)
    assert parsed["arguments"]["a"] == 1


def test_malformed_json_returns_empty():
    text = "<tool_call>this is not json at all!!</tool_call>"
    calls = parse_tool_calls(text)
    assert calls == []


def test_case_insensitive_tags():
    text = '<TOOL_CALL>{"name": "foo", "arguments": {}}</TOOL_CALL>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "foo"
