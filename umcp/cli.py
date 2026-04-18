"""umcp CLI — Typer-based command interface."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .client import MCPClient
from .config import AppConfig
from .trace import Tracer

# Load .env from project root before anything else touches os.environ
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)
except ImportError:
    pass

app = typer.Typer(
    name="umcp",
    help="Universal MCP Client — local LLM (Ollama) + any MCP server.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ------------------------------------------------------------------ #
# Shared options (reused across commands)
# ------------------------------------------------------------------ #

def _load_config(config_path: str | None, model: str | None, server: str | None) -> AppConfig:
    cfg = AppConfig.load(config_path)
    if model:
        cfg = cfg.model_copy(update={"default_model": model})
    return cfg


def _server_list(server: str | None) -> list[str] | None:
    if not server:
        return None
    return [s.strip() for s in server.split(",") if s.strip()]


def _tool_list(tools: str | None) -> list[str] | None:
    if not tools:
        return None
    return [t.strip() for t in tools.split(",") if t.strip()]


# ------------------------------------------------------------------ #
# umcp run
# ------------------------------------------------------------------ #

@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task or question to execute"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to mcp.json"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model to use"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Comma-separated server names"),
    tools: Optional[str] = typer.Option(None, "--tools", help="Comma-separated tool glob whitelist"),
    tool_selection: Optional[str] = typer.Option(None, "--tool-selection", help="keyword|hybrid|embedding|all"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned tool calls without executing"),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM tokens to stdout"),
    no_prefix_display: bool = typer.Option(False, "--no-prefix-display", help="Hide server prefix in output"),
    output_json: bool = typer.Option(False, "--json", help="Output result as JSON"),
    no_trace: bool = typer.Option(False, "--no-trace", help="Disable trace logging"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool call details"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Override max iterations"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Override total timeout (ms)"),
    tool_timeout: Optional[int] = typer.Option(None, "--tool-timeout", help="Override per-tool timeout (ms)"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="debug|info|warn|error"),
    log_format: Optional[str] = typer.Option(None, "--log-format", help="text|json"),
    cache: Optional[bool] = typer.Option(None, "--cache/--no-cache", help="Enable/disable cache for this run"),
    cache_ttl: Optional[int] = typer.Option(None, "--cache-ttl", help="Override cache TTL (seconds)"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Named session for persistent history"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Append text to base system prompt"),
) -> None:
    """Run a single task against configured MCP servers."""
    cfg = _load_config(config, model, server)

    # Apply CLI overrides
    if max_iterations is not None:
        cfg.execution.max_iterations = max_iterations
    if timeout is not None:
        cfg.execution.total_timeout_ms = timeout
    if tool_timeout is not None:
        cfg.execution.tool_timeout_ms = tool_timeout
    if no_trace:
        cfg.logging.trace = False
    if tool_selection:
        cfg.tool_filter.strategy = tool_selection  # type: ignore[assignment]

    async def _run() -> None:
        client = MCPClient(
            config=cfg,
            servers=_server_list(server),
        )
        async with client:
            result = await client.run(
                prompt,
                tools=_tool_list(tools),
                dry_run=dry_run,
                stream=stream,
                session_id=session_id,
                cache=cache,
                cache_ttl=cache_ttl,
                system_prompt=system_prompt,
            )

        if output_json:
            out = {
                "answer": result.answer,
                "tool_calls_made": result.tool_calls_made,
                "cache_hits": result.cache_hits,
                "iterations": result.iterations,
                "total_latency_ms": result.total_latency_ms,
                "exit_reason": result.exit_reason,
                "success": result.success,
            }
            typer.echo(json.dumps(out, indent=2))
        else:
            answer = result.answer
            if no_prefix_display:
                import re
                for srv in cfg.servers:
                    answer = re.sub(rf'\b{re.escape(srv.name)}\.', '', answer)
            typer.echo(answer)

            if verbose or not result.success:
                err_console.print(
                    f"\n[dim]iterations={result.iterations} "
                    f"tools={result.tool_calls_made} "
                    f"cache_hits={result.cache_hits} "
                    f"latency={result.total_latency_ms:.0f}ms "
                    f"exit={result.exit_reason}[/dim]",
                    highlight=False,
                )

        if not result.success:
            raise typer.Exit(code=2)

    asyncio.run(_run())


# ------------------------------------------------------------------ #
# umcp chat
# ------------------------------------------------------------------ #

@app.command()
def chat(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    tools: Optional[str] = typer.Option(None, "--tools"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID for persistent history"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt"),
    no_trace: bool = typer.Option(False, "--no-trace"),
) -> None:
    """Start an interactive chat session. Always streams. Ctrl+C to exit."""
    cfg = _load_config(config, model, server)
    if no_trace:
        cfg.logging.trace = False

    async def _chat() -> None:
        client = MCPClient(config=cfg, servers=_server_list(server))
        async with client:
            if session_id:
                # Load existing history
                history = client._session_store.load(session_id)
                turn_count = sum(1 for m in history if m.get("role") == "user")
                console.print(
                    f"[bold green]umcp chat[/bold green] — session: [cyan]{session_id}[/cyan] "
                    f"({turn_count} prior turns) | Ctrl+C to exit\n"
                )
            else:
                console.print(
                    "[bold green]umcp chat[/bold green] — type your task, Ctrl+C to exit\n"
                )

            # In-memory history for sessionless chat — carries context between turns
            chat_history: list[dict] = []

            while True:
                try:
                    user_prompt = typer.prompt("you")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[dim]Goodbye.[/dim]")
                    if session_id:
                        console.print(
                            f"[dim]Session saved as '{session_id}'. "
                            f"Resume with: umcp chat --session {session_id}[/dim]"
                        )
                    break

                console.print("\n[bold cyan]umcp:[/bold cyan] ", end="")

                result = await client.run(
                    user_prompt,
                    tools=_tool_list(tools),
                    stream=True,
                    session_id=session_id,
                    prior_messages=chat_history if not session_id else None,
                    system_prompt=system_prompt,
                )

                # Update in-memory history for next turn (no-session mode only)
                if not session_id and result.messages:
                    chat_history = [m for m in result.messages if m.get("role") != "system"]
                    # Cap at 30 messages to avoid filling the context window
                    if len(chat_history) > 30:
                        chat_history = chat_history[-30:]

                # In stream mode loop.py already printed tokens; print trailing newline
                console.print()

                if verbose:
                    err_console.print(
                        f"[dim]tools={result.tool_calls_made} "
                        f"iterations={result.iterations} "
                        f"cache_hits={result.cache_hits}[/dim]"
                    )

    asyncio.run(_chat())


# ------------------------------------------------------------------ #
# umcp tools
# ------------------------------------------------------------------ #

@app.command()
def tools(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    filter: Optional[str] = typer.Option(None, "--filter", help="Glob pattern to filter tools"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """List all tools from connected MCP servers."""
    cfg = _load_config(config, None, server)

    async def _tools() -> None:
        client = MCPClient(config=cfg, servers=_server_list(server))
        async with client:
            tool_list = await client.list_tools(filter=filter)

        if not tool_list:
            console.print("[yellow]No tools found.[/yellow]")
            return

        if output_json:
            out = [
                {
                    "name": t.full_name,
                    "server": t.server,
                    "description": t.description,
                }
                for t in tool_list
            ]
            typer.echo(json.dumps(out, indent=2))
            return

        table = Table(title="Available Tools", show_lines=True)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Server", style="green")
        table.add_column("Description")

        for t in tool_list:
            table.add_row(t.full_name, t.server, t.description or "—")

        console.print(table)

    asyncio.run(_tools())


# ------------------------------------------------------------------ #
# umcp servers
# ------------------------------------------------------------------ #

@app.command()
def servers(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """List configured servers and their connection status."""
    cfg = _load_config(config, None, None)

    if not cfg.servers:
        console.print("[yellow]No servers configured in mcp.json.[/yellow]")
        return

    async def _servers() -> None:
        table = Table(title="Configured Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Transport", style="green")
        table.add_column("URL / Command")
        table.add_column("Auth")
        table.add_column("Status")

        from .transports import make_transport

        for s in cfg.servers:
            location = s.url or f"{s.command} {' '.join(s.args)}"
            auth_type = s.auth.type if s.auth else "none"

            transport = make_transport(s)
            try:
                await transport.connect()
                status = "[green]connected[/green]"
                await transport.close()
            except NotImplementedError:
                status = "[yellow]phase 2[/yellow]"
            except BaseException as exc:
                status = f"[red]error: {type(exc).__name__}[/red]"

            table.add_row(s.name, s.transport, location, auth_type, status)

        console.print(table)

    asyncio.run(_servers())


# ------------------------------------------------------------------ #
# umcp models
# ------------------------------------------------------------------ #

@app.command()
def models(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """List available Ollama models and their tool-calling capability."""
    cfg = _load_config(config, None, None)

    async def _models() -> None:
        from .adapters.ollama import OllamaAdapter

        ollama = OllamaAdapter(cfg.ollama_base_url, cfg.default_model)
        try:
            rows = await ollama.model_capability_summary()
        except Exception as exc:
            console.print(f"[red]Could not reach Ollama at {cfg.ollama_base_url}: {exc}[/red]")
            console.print("Is Ollama running? Try: [cyan]ollama serve[/cyan]")
            raise typer.Exit(1)
        finally:
            await ollama.close()

        if not rows:
            console.print("[yellow]No models found. Pull a model with: ollama pull qwen2.5:7b[/yellow]")
            return

        if output_json:
            typer.echo(json.dumps(rows, indent=2))
            return

        table = Table(title="Ollama Models")
        table.add_column("Model", style="cyan")
        table.add_column("Tool Calling")
        table.add_column("Size (MB)")

        for row in rows:
            tc = row["tool_calling"]
            if tc == "yes":
                tc_display = "[green]yes[/green]"
            elif tc == "no":
                tc_display = "[red]no (fallback mode)[/red]"
            else:
                tc_display = "[yellow]unknown (auto-detect on first use)[/yellow]"

            size_mb = round(row.get("size", 0) / 1_000_000)
            table.add_row(row["name"], tc_display, str(size_mb))

        console.print(table)
        console.print(
            "\n[dim]Recommended: qwen2.5:7b or qwen2.5:14b for best tool-calling reliability.[/dim]"
        )

    asyncio.run(_models())


# ------------------------------------------------------------------ #
# umcp trace
# ------------------------------------------------------------------ #

trace_app = typer.Typer(help="View tool call traces.")
app.add_typer(trace_app, name="trace")


@trace_app.command("last")
def trace_last(
    filter: Optional[str] = typer.Option(None, "--filter", help="Filter by tool glob (e.g. db.*)"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the trace from the last run."""
    entries = Tracer.load_last()
    if not entries:
        console.print("[yellow]No trace found. Run a task first.[/yellow]")
        return

    _display_trace(entries, filter, output_json, title="Last Run Trace")


@trace_app.command("session")
def trace_session(
    session_id: str = typer.Argument(..., help="Session ID to show trace for"),
    filter: Optional[str] = typer.Option(None, "--filter", help="Filter by tool glob"),
    output_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show the trace for a named session."""
    entries = Tracer.load_session(session_id)
    if not entries:
        console.print(f"[yellow]No trace found for session '{session_id}'.[/yellow]")
        return

    _display_trace(entries, filter, output_json, title=f"Session Trace: {session_id}")


@trace_app.command("tail")
def trace_tail(
    interval: float = typer.Option(0.5, "--interval", help="Poll interval in seconds"),
    filter: Optional[str] = typer.Option(None, "--filter", help="Filter by tool glob"),
) -> None:
    """Live-tail trace output while a run is in progress.

    Reads from the live JSONL file written by the agent loop in real time.
    Shows entries from the current (or most recent) run only.
    """
    from .trace import _LIVE_TRACE_PATH
    import fnmatch as _fnmatch

    console.print("[dim]Tailing live trace... start a run in another terminal. Ctrl+C to stop.[/dim]")

    try:
        with _LIVE_TRACE_PATH.open("a+", encoding="utf-8") as f:
            # Seek to start so we catch any already-written entries from current run
            f.seek(0)
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tool = entry.get("tool", "?")
                    if filter and not _fnmatch.fnmatch(tool, filter):
                        continue
                    status = entry.get("status", "?")
                    latency = entry.get("latency_ms", 0)
                    color = "green" if status == "success" else "red" if status == "error" else "yellow"
                    console.print(
                        f"[{color}]{status:12}[/{color}] [cyan]{tool}[/cyan] "
                        f"[dim]{latency:.0f}ms[/dim]"
                    )
                else:
                    time.sleep(interval)
    except FileNotFoundError:
        console.print("[yellow]No live trace file found. Start a run first.[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[dim]Tail stopped.[/dim]")


def _display_trace(
    entries: list[dict],
    filter: str | None,
    output_json: bool,
    title: str,
) -> None:
    """Render trace entries as a table or JSON."""
    if filter:
        import fnmatch
        entries = [e for e in entries if fnmatch.fnmatch(e.get("tool", ""), filter)]

    if output_json:
        typer.echo(json.dumps(entries, indent=2))
        return

    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim")
    table.add_column("Tool", style="cyan")
    table.add_column("Status")
    table.add_column("Latency")
    table.add_column("Cache")
    table.add_column("Retries")
    table.add_column("Group", style="dim")

    for i, e in enumerate(entries, 1):
        status = e.get("status", "?")
        if status == "success":
            status_display = "[green]success[/green]"
        elif status == "cache_hit":
            status_display = "[blue]cache_hit[/blue]"
        elif status == "dry_run":
            status_display = "[yellow]dry_run[/yellow]"
        else:
            status_display = f"[red]{status}[/red]"

        table.add_row(
            str(i),
            e.get("tool", "?"),
            status_display,
            f"{e.get('latency_ms', 0):.1f}ms",
            "yes" if e.get("cache_hit") else "no",
            str(e.get("retry_count", 0)),
            e.get("group_id") or "—",
        )

    console.print(table)


# ------------------------------------------------------------------ #
# umcp cache
# ------------------------------------------------------------------ #

cache_app = typer.Typer(help="Manage response cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def cache_clear(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Clear all cached tool responses."""
    cfg = _load_config(config, None, None)
    from .cache import ResponseCache
    cache = ResponseCache(cfg.cache)
    count = cache.clear()
    console.print(f"[green]Cleared {count} cache entries.[/green]")


@cache_app.command("stats")
def cache_stats(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show cache statistics."""
    cfg = _load_config(config, None, None)
    from .cache import ResponseCache
    cache = ResponseCache(cfg.cache)
    stats = cache.stats()
    status = "enabled" if cfg.cache.enabled else "disabled"
    console.print(f"Cache: [bold]{status}[/bold]")
    console.print(f"  TTL: {cfg.cache.ttl_seconds}s")
    console.print(f"  Max size: {cfg.cache.max_size} entries")
    console.print(f"  Storage: {cfg.cache.storage}")
    console.print(f"  Excluded tools: {', '.join(cfg.cache.exclude_tools) or 'none'}")
    console.print(f"  Current entries: {stats.get('entries', 0)}")
    console.print(f"  Hit rate: {stats.get('hit_rate', 0.0):.1%}")


# ------------------------------------------------------------------ #
# umcp config
# ------------------------------------------------------------------ #

config_app = typer.Typer(help="Manage mcp.json configuration.")
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate(
    config: Optional[str] = typer.Option("mcp.json", "--config", "-c"),
) -> None:
    """Validate mcp.json and report any errors."""
    path = Path(config)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    try:
        cfg = AppConfig.load(path)
        console.print(f"[green]Valid[/green] - {len(cfg.servers)} server(s) configured")
        for s in cfg.servers:
            console.print(f"  [cyan]{s.name}[/cyan] ({s.transport})")
    except Exception as exc:
        console.print(f"[red]✗ Invalid:[/red] {exc}")
        raise typer.Exit(1)


@config_app.command("init")
def config_init(
    output: str = typer.Option("mcp.json", "--output", "-o"),
) -> None:
    """Generate a starter mcp.json in the current directory."""
    dest = Path(output)
    if dest.exists():
        overwrite = typer.confirm(f"{dest} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit()

    example = Path(__file__).parent.parent / "mcp.json.example"
    if example.exists():
        dest.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # Fallback: write a minimal config
        dest.write_text(
            '{\n  "version": "1",\n  "default_model": "qwen2.5:7b",\n'
            '  "ollama_base_url": "http://localhost:11434",\n  "servers": []\n}\n',
            encoding="utf-8",
        )
    console.print(f"[green]Created {dest}[/green]")


# ------------------------------------------------------------------ #
# umcp sessions
# ------------------------------------------------------------------ #

sessions_app = typer.Typer(help="Manage named chat sessions.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """List all saved chat sessions."""
    from .session import SessionStore
    cfg = _load_config(config, None, None)
    store = SessionStore(cfg.session.storage_path)
    session_ids = store.list_sessions()
    if not session_ids:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Messages")
    for sid in session_ids:
        messages = store.load(sid)
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        table.add_row(sid, str(user_turns) + " user turns")
    console.print(table)


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Delete a named chat session."""
    from .session import SessionStore
    cfg = _load_config(config, None, None)
    store = SessionStore(cfg.session.storage_path)
    deleted = store.delete(session_id)
    if deleted:
        console.print(f"[green]Deleted session '{session_id}'.[/green]")
    else:
        console.print(f"[yellow]Session '{session_id}' not found.[/yellow]")


# ------------------------------------------------------------------ #
# umcp web
# ------------------------------------------------------------------ #

@app.command()
def web(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8765, "--port", "-p", help="Port to listen on"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
) -> None:
    """Start the web dashboard at http://localhost:8765"""
    cfg = _load_config(config, None, server)

    async def _serve() -> None:
        from .web import serve
        cfg_path = Path(config) if config else None
        await serve(cfg, host=host, port=port, config_path=cfg_path)

    asyncio.run(_serve())


if __name__ == "__main__":
    app()
