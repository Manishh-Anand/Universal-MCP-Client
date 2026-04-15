"""Tests for config loading and validation."""
import json
import pytest
from pathlib import Path
from umcp.config import AppConfig, ServerConfig, ExecutionConfig


def test_default_config():
    cfg = AppConfig()
    assert cfg.default_model == "qwen2.5:7b"
    assert cfg.execution.max_iterations == 10
    assert cfg.execution.tool_timeout_ms == 5000


def test_load_from_dict():
    data = {
        "version": "1",
        "default_model": "llama3",
        "servers": [
            {
                "name": "test",
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
            }
        ],
    }
    cfg = AppConfig.model_validate(data)
    assert cfg.default_model == "llama3"
    assert len(cfg.servers) == 1
    assert cfg.servers[0].name == "test"


def test_stdio_requires_command():
    with pytest.raises(Exception, match="command"):
        ServerConfig(name="bad", transport="stdio")


def test_http_requires_url():
    with pytest.raises(Exception, match="url"):
        ServerConfig(name="bad", transport="http")


def test_sse_requires_url():
    with pytest.raises(Exception, match="url"):
        ServerConfig(name="bad", transport="sse")


def test_load_from_file(tmp_path: Path):
    data = {
        "version": "1",
        "default_model": "qwen2.5:7b",
        "servers": [
            {
                "name": "local",
                "transport": "stdio",
                "command": "python",
                "args": ["srv.py"],
            }
        ],
    }
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps(data))
    cfg = AppConfig.load(cfg_file)
    assert cfg.servers[0].name == "local"


def test_filter_servers():
    cfg = AppConfig.model_validate({
        "servers": [
            {"name": "a", "transport": "stdio", "command": "python"},
            {"name": "b", "transport": "stdio", "command": "python"},
            {"name": "c", "transport": "stdio", "command": "python"},
        ]
    })
    result = cfg.filter_servers(["a", "c"])
    assert [s.name for s in result] == ["a", "c"]


def test_auth_bearer_resolution(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    from umcp.config import AuthConfig
    auth = AuthConfig(type="bearer", token="env:MY_TOKEN")
    headers = auth.get_headers()
    assert headers == {"Authorization": "Bearer secret123"}


def test_auth_missing_env_raises(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    from umcp.config import AuthConfig
    auth = AuthConfig(type="bearer", token="env:MISSING_VAR")
    with pytest.raises(ValueError, match="MISSING_VAR"):
        auth.get_headers()
