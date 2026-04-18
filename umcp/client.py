"""MCPClient — the primary SDK entry point."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .adapters.ollama import OllamaAdapter
from .aggregator import ToolAggregator
from .cache import ResponseCache
from .config import AppConfig, ServerConfig
from .log import configure as _configure_log, get_logger
from .loop import AgentLoop, LoopResult
from .plugins import PluginRegistry
from .session import SessionStore
from .trace import Tracer, new_trace_id
from .transports.base import ToolInfo

_log = get_logger()


class MCPClient:
    """Universal MCP client — connect to any MCP servers, driven by a local Ollama LLM.

    Usage::

        async with MCPClient(config="./mcp.json") as client:
            result = await client.run("get the weather in Tokyo")
            print(result.answer)
    """

    def __init__(
        self,
        config: str | Path | AppConfig | None = None,
        model: str | None = None,
        servers: list[str] | None = None,
    ) -> None:
        if isinstance(config, AppConfig):
            self._config = config
        else:
            self._config = AppConfig.load(config)

        if model:
            self._config = self._config.model_copy(update={"default_model": model})

        self._server_filter = servers
        self._aggregator = ToolAggregator()
        self._ollama: OllamaAdapter | None = None
        self._tracer: Tracer | None = None
        self._cache: ResponseCache | None = None
        self._connected = False
        self._plugin_registry = PluginRegistry()
        self._session_store = SessionStore(self._config.session.storage_path)
        self._adapter_cache: dict[str, OllamaAdapter] = {}

        # Event hooks
        self._on_tool_call: list[Callable] = []
        self._on_error: list[Callable] = []
        self._on_cache_hit: list[Callable] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> "MCPClient":
        """Connect to all configured MCP servers."""
        _configure_log(self._config.logging.level, self._config.logging.output)
        session_id = new_trace_id()
        self._tracer = Tracer(
            session_id=session_id,
            enabled=self._config.logging.trace,
        )
        self._ollama = OllamaAdapter(
            base_url=self._config.ollama_base_url,
            model=self._config.default_model,
        )
        await self._ollama.ensure_running()
        self._cache = ResponseCache(self._config.cache)

        servers = self._config.filter_servers(self._server_filter)
        if not servers:
            _log.warning("no_servers_configured", hint="Add servers to mcp.json or pass --server")

        failed = await self._aggregator.connect_all(servers)
        if failed:
            _log.warning("servers_unavailable", servers=failed, hint="Continuing with available servers")

        self._connected = True
        return self

    async def close(self) -> None:
        """Close all server connections."""
        await self._aggregator.close_all()
        if self._ollama:
            await self._ollama.close()
        for adapter in self._adapter_cache.values():
            await adapter.close()
        self._adapter_cache.clear()
        if self._tracer:
            self._tracer.save_last()
        if self._cache:
            self._cache.save_stats()
        self._connected = False

    async def __aenter__(self) -> "MCPClient":
        return await self.connect()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Core run
    # ------------------------------------------------------------------ #

    async def run(
        self,
        prompt: str,
        model: str | None = None,
        tools: list[str] | None = None,
        dry_run: bool = False,
        stream: bool = False,
        session_id: str | None = None,
        prior_messages: list[dict] | None = None,
        cache: bool | None = None,
        cache_ttl: int | None = None,
        system_prompt: str | None = None,
        event_queue: asyncio.Queue | None = None,
    ) -> LoopResult:
        """Execute a single task and return the result.

        Args:
            prompt: The task or question to execute.
            model: Override the model for this run.
            tools: Glob patterns to whitelist specific tools (e.g. ["weather.*"]).
            dry_run: Print planned tool calls without executing them.
            stream: Stream LLM tokens to stdout (final answer only).
            session_id: Named session — loads history before run, saves after.
            cache: Override cache enabled state for this run.
            cache_ttl: Override cache TTL in seconds for this run.
            system_prompt: Additional text appended to the base system prompt.
        """
        if not self._connected:
            raise RuntimeError("Call await client.connect() before run()")

        ollama = self._ollama
        if model and model != self._config.default_model:
            if model not in self._adapter_cache:
                self._adapter_cache[model] = OllamaAdapter(
                    self._config.ollama_base_url, model
                )
            ollama = self._adapter_cache[model]

        # Load prior conversation history — session store takes priority over direct arg
        effective_prior: list[dict] = []
        if session_id:
            effective_prior = self._session_store.load(session_id)
        elif prior_messages:
            effective_prior = prior_messages

        loop = AgentLoop(
            config=self._config,
            aggregator=self._aggregator,
            ollama=ollama,
            tracer=self._tracer,
            cache=self._cache,
            plugin_registry=self._plugin_registry,
        )

        result = await loop.run(
            prompt,
            tools_whitelist=tools,
            dry_run=dry_run,
            stream=stream,
            prior_messages=effective_prior,
            cache_enabled=cache,
            cache_ttl=cache_ttl,
            system_prompt_override=system_prompt,
            event_queue=event_queue,
        )

        # Persist updated history for named sessions
        if session_id and result.messages:
            messages = result.messages
            max_msgs = self._config.session.max_messages
            if max_msgs > 0 and len(messages) > max_msgs:
                system_msgs = [m for m in messages if m.get("role") == "system"]
                non_system = [m for m in messages if m.get("role") != "system"]
                trimmed = non_system[-(max_msgs - len(system_msgs)):]
                messages = system_msgs + trimmed
            self._session_store.save(session_id, messages)
            # Auto-generate title from first user message and save metadata
            first_user = next(
                (m.get("content", "") for m in result.messages if m.get("role") == "user"),
                session_id,
            )
            title = (first_user[:60] + "…") if len(first_user) > 60 else first_user
            turn_count = sum(1 for m in result.messages if m.get("role") == "user")
            self._session_store.save_meta(session_id, title.strip() or session_id, turn_count)
            # Save trace for named sessions too
            if self._tracer:
                from dataclasses import asdict as _asdict
                self._session_store.save_trace(
                    session_id,
                    [_asdict(e) for e in self._tracer.all_entries()],
                )

        self._tracer.save_last()

        # Fire event hooks
        for hook in self._on_tool_call:
            for entry in self._tracer.all_entries():
                try:
                    hook(entry)
                except Exception:
                    pass
        if not result.success:
            for hook in self._on_error:
                try:
                    hook(result)
                except Exception:
                    pass

        return result

    # ------------------------------------------------------------------ #
    # Tool inspection
    # ------------------------------------------------------------------ #

    async def list_tools(self, filter: str | None = None) -> list[ToolInfo]:
        """Return all aggregated tools, optionally filtered by glob pattern."""
        if not self._connected:
            raise RuntimeError("Call connect() first")
        tools = await self._aggregator.collect_tools()
        if filter:
            import fnmatch
            tools = [t for t in tools if fnmatch.fnmatch(t.full_name, filter)]
        return tools

    async def call_tool(self, full_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool directly, bypassing the LLM.

        The tool name must be prefixed: e.g. 'weather.get_forecast'.
        Schema validation still runs.
        """
        if not self._connected:
            raise RuntimeError("Call connect() first")
        from .validator import validate_and_coerce
        from .loop import _find_tool

        tools = await self._aggregator.collect_tools()
        tool_info = _find_tool(tools, full_name)
        if tool_info is None:
            raise ValueError(f"Tool '{full_name}' not found in connected servers")

        valid, errors, coerced = validate_and_coerce(tool_info.input_schema, arguments)
        if not valid:
            raise ValueError(f"Invalid arguments for '{full_name}': {errors}")

        server_name = tool_info.server
        transport = self._aggregator.get_transport(server_name)
        if transport is None:
            raise RuntimeError(f"No transport for server '{server_name}'")

        result = await transport.call_tool(
            tool_info.name,
            coerced,
            self._config.execution.tool_timeout_ms,
        )
        if not result.success:
            raise RuntimeError(f"Tool '{full_name}' failed: {result.error}")
        return result.content

    # ------------------------------------------------------------------ #
    # Server management
    # ------------------------------------------------------------------ #

    def add_server(self, server_dict: dict[str, Any]) -> None:
        """Add a server at runtime (before or after connect)."""
        server = ServerConfig.model_validate(server_dict)
        self._config.servers.append(server)

    # ------------------------------------------------------------------ #
    # Event hooks
    # ------------------------------------------------------------------ #

    def on_tool_call(self, fn: Callable) -> Callable:
        self._on_tool_call.append(fn)
        return fn

    def on_error(self, fn: Callable) -> Callable:
        self._on_error.append(fn)
        return fn

    def on_cache_hit(self, fn: Callable) -> Callable:
        self._on_cache_hit.append(fn)
        return fn

    # ------------------------------------------------------------------ #
    # Plugin system (Phase 4)
    # ------------------------------------------------------------------ #

    def plugin(self, hook_name: str) -> Callable:
        """Decorator to register a plugin hook.

        Example::

            @client.plugin("system_prompt")
            def my_prompt(base: str) -> str:
                return base + "\\nAlways respond in Japanese."

            @client.plugin("tool_filter")
            def my_filter(tools, prompt):
                return [t for t in tools if "dangerous" not in t.name]
        """
        def decorator(fn: Callable) -> Callable:
            self._plugin_registry.register(hook_name, fn)
            return fn
        return decorator

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def list_sessions(self) -> list[str]:
        """Return all known chat session IDs."""
        return self._session_store.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a named session. Returns True if it existed."""
        return self._session_store.delete(session_id)
