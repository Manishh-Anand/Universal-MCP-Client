"""Configuration models for umcp — loaded from mcp.json."""
from __future__ import annotations

import json
import os
import time as _time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Module-level OAuth2 token cache: key -> (access_token, expires_at_unix)
_oauth2_token_cache: dict[str, tuple[str, float]] = {}


class AuthConfig(BaseModel):
    type: Literal["bearer", "api_key", "none", "oauth2"] = "none"
    token: str | None = None       # bearer: "env:VAR" or literal
    header: str | None = None      # api_key: header name
    value: str | None = None       # api_key: "env:VAR" or literal
    # OAuth2 fields (Phase 4)
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    def _resolve(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        if raw.startswith("env:"):
            var = raw[4:]
            val = os.environ.get(var)
            if val is None:
                raise ValueError(f"Environment variable {var!r} is not set")
            return val
        return raw

    def get_headers(self) -> dict[str, str]:
        if self.type == "bearer":
            token = self._resolve(self.token)
            return {"Authorization": f"Bearer {token}"} if token else {}
        if self.type == "api_key":
            val = self._resolve(self.value)
            return {self.header: val} if self.header and val else {}
        if self.type == "oauth2":
            return self._get_oauth2_headers()
        return {}

    def _get_oauth2_headers(self) -> dict[str, str]:
        """Fetch an OAuth2 client_credentials token, caching until near-expiry."""
        if not self.token_url:
            return {}

        cache_key = f"{self.token_url}:{self.client_id}"
        cached = _oauth2_token_cache.get(cache_key)
        if cached:
            token, expires_at = cached
            if _time.time() < expires_at - 30:   # 30-second buffer
                return {"Authorization": f"Bearer {token}"}

        client_id = self._resolve(self.client_id)
        client_secret = self._resolve(self.client_secret)

        try:
            import httpx
            with httpx.Client(timeout=10.0) as http:
                resp = http.post(
                    self.token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": client_id or "",
                        "client_secret": client_secret or "",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise ValueError(
                f"OAuth2 token fetch failed for {self.token_url!r}: {exc}"
            ) from exc

        token = data.get("access_token")
        if not token:
            raise ValueError(f"OAuth2 response missing 'access_token': {data}")

        expires_in = int(data.get("expires_in", 3600))
        _oauth2_token_cache[cache_key] = (token, _time.time() + expires_in)
        return {"Authorization": f"Bearer {token}"}


class ServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "sse", "http"]
    # stdio-only
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # network transports
    url: str | None = None
    # shared
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "ServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"Server {self.name!r}: stdio transport requires 'command'")
        if self.transport in ("sse", "http") and not self.url:
            raise ValueError(
                f"Server {self.name!r}: {self.transport} transport requires 'url'"
            )
        return self


class ExecutionConfig(BaseModel):
    max_iterations: int = 10
    tool_timeout_ms: int = 5000
    total_timeout_ms: int = 30000
    max_retries_per_tool: int = 2
    parallel_tools: bool = False   # Phase 3.5: execute concurrent tool calls with asyncio.gather


class CacheConfig(BaseModel):
    enabled: bool = False
    ttl_seconds: int = 300
    max_size: int = 1000
    storage: Literal["memory", "session", "file"] = "memory"
    exclude_tools: list[str] = Field(
        default_factory=lambda: [
            "*.insert_*", "*.write_*", "*.delete_*",
            "*.update_*", "*.create_*", "*.remove_*",
        ]
    )


class SecurityConfig(BaseModel):
    trusted_servers_only: bool = False
    trusted_servers: list[str] = Field(default_factory=list)
    sanitize_tool_descriptions: bool = True
    mask_env_values_in_logs: bool = True


class SessionConfig(BaseModel):
    persist: bool = False
    storage_path: str = "~/.config/umcp/sessions"


class LoggingConfig(BaseModel):
    level: Literal["debug", "info", "warn", "error"] = "info"
    trace: bool = True
    output: Literal["stderr", "stdout", "file"] = "stderr"


class ToolFilterConfig(BaseModel):
    strategy: Literal["keyword", "hybrid", "embedding", "all"] = "hybrid"
    top_n: int = 20
    default_whitelist: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)
    embedding_model: str = "nomic-embed-text"


class SchemaValidationConfig(BaseModel):
    auto_coerce: bool = True   # Phase 4: can disable auto type coercion


class AppConfig(BaseModel):
    version: str = "1"
    default_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    servers: list[ServerConfig] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tool_filter: ToolFilterConfig = Field(default_factory=ToolFilterConfig)
    schema_validation: SchemaValidationConfig = Field(default_factory=SchemaValidationConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        """Load config from file. Searches path → ./mcp.json → ~/.config/umcp/mcp.json."""
        candidates: list[Path] = []
        if path:
            candidates.append(Path(path))
        candidates.extend([
            Path("mcp.json"),
            Path.home() / ".config" / "umcp" / "mcp.json",
        ])
        for candidate in candidates:
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return cls.model_validate(data)
        return cls()  # all defaults — no servers configured

    def get_server(self, name: str) -> ServerConfig | None:
        return next((s for s in self.servers if s.name == name), None)

    def filter_servers(self, names: list[str] | None) -> list[ServerConfig]:
        """Return subset of servers by name list, or all if names is None."""
        if not names:
            return list(self.servers)
        return [s for s in self.servers if s.name in names]
