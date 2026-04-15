"""Tests for tool filtering."""
import pytest
from umcp.filter import apply_filter, _keyword_filter, _tokenize
from umcp.transports.base import ToolInfo
from umcp.config import ToolFilterConfig


def _make_tool(server: str, name: str, desc: str = "") -> ToolInfo:
    return ToolInfo(
        server=server,
        name=name,
        full_name=f"{server}.{name}",
        description=desc,
        input_schema={},
    )


def _default_config(**kwargs) -> ToolFilterConfig:
    return ToolFilterConfig(**kwargs)


ALL_TOOLS = [
    _make_tool("weather", "get_forecast", "Get weather forecast for a city"),
    _make_tool("weather", "get_alerts", "Get weather alerts"),
    _make_tool("db", "insert_record", "Insert a record into the database"),
    _make_tool("db", "query", "Query the database"),
    _make_tool("deploy", "trigger", "Trigger a deployment"),
]


def test_whitelist_glob():
    cfg = _default_config(strategy="all")
    result = apply_filter(ALL_TOOLS, "anything", cfg, whitelist=["weather.*"])
    names = [t.full_name for t in result]
    assert "weather.get_forecast" in names
    assert "weather.get_alerts" in names
    assert "db.insert_record" not in names


def test_whitelist_specific_tool():
    cfg = _default_config(strategy="all")
    result = apply_filter(ALL_TOOLS, "anything", cfg, whitelist=["db.query"])
    assert len(result) == 1
    assert result[0].full_name == "db.query"


def test_strategy_all_returns_everything():
    cfg = _default_config(strategy="all")
    result = apply_filter(ALL_TOOLS, "whatever", cfg)
    assert len(result) == len(ALL_TOOLS)


def test_exclude_pattern():
    cfg = _default_config(strategy="all", exclude=["db.*"])
    result = apply_filter(ALL_TOOLS, "query data", cfg)
    names = [t.full_name for t in result]
    assert "db.insert_record" not in names
    assert "db.query" not in names


def test_keyword_filter_ranks_relevant_tools():
    tools = [
        _make_tool("weather", "get_forecast", "Get weather forecast for a city"),
        _make_tool("db", "query", "Query the database"),
        _make_tool("deploy", "trigger", "Trigger a deployment pipeline"),
    ]
    result = _keyword_filter(tools, "what is the weather forecast today", top_n=10)
    # weather tool should score highest
    assert result[0].full_name == "weather.get_forecast"


def test_keyword_no_overlap_returns_all_tools():
    """If prompt has no keyword match, all tools are returned (LLM decides)."""
    tools = [_make_tool("a", "foo", "bar"), _make_tool("b", "baz", "qux")]
    result = _keyword_filter(tools, "xyzzy completely unrelated", top_n=10)
    assert len(result) == len(tools)


def test_top_n_cap():
    # strategy="all" bypasses relevance filtering AND the top_n cap
    # (it means "give me everything" — for debugging)
    cfg = _default_config(strategy="all", top_n=2)
    result = apply_filter(ALL_TOOLS, "anything", cfg)
    assert len(result) == len(ALL_TOOLS)


def test_top_n_cap_with_keyword_strategy():
    # keyword strategy DOES respect top_n
    cfg = _default_config(strategy="keyword", top_n=2)
    result = apply_filter(ALL_TOOLS, "weather forecast database query deploy", cfg)
    assert len(result) <= 2


def test_tokenize_removes_stopwords():
    tokens = _tokenize("what is the weather in tokyo today")
    assert "what" not in tokens
    assert "the" not in tokens
    assert "in" not in tokens
    assert "weather" in tokens
    assert "tokyo" in tokens
