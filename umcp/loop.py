"""Agent loop — the core plan → filter → validate → call → observe cycle."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.fallback import auto_repair_json, parse_tool_calls, strip_tool_call_blocks
from .adapters.ollama import OllamaAdapter
from .aggregator import ToolAggregator
from .cache import ResponseCache
from .config import AppConfig
from .filter import apply_filter
from .log import get_logger
from .retry import (
    RetryDecision,
    RetryReason,
    RetryState,
    backoff_sleep,
    build_correction_message,
    decide_retry,
)
from .security import sanitize_tool
from .session import RunSession
from .trace import Tracer, new_trace_id
from .transports.base import ToolInfo
from .validator import validate_and_coerce


@dataclass
class LoopResult:
    answer: str
    tool_calls_made: int
    cache_hits: int
    iterations: int
    total_latency_ms: float
    exit_reason: str   # "completed" | "max_iterations" | "timeout" | "error"
    success: bool
    messages: list[dict[str, Any]] = field(default_factory=list)
    """Updated message history (including this run's turns) — used by chat mode."""


def _load_prompt(filename: str) -> str:
    prompt_dir = Path(__file__).parent / "prompts"
    return (prompt_dir / filename).read_text(encoding="utf-8").strip()


class AgentLoop:
    """Drives the LLM agent loop for a single task or chat turn."""

    def __init__(
        self,
        config: AppConfig,
        aggregator: ToolAggregator,
        ollama: OllamaAdapter,
        tracer: Tracer,
        cache: ResponseCache,
        plugin_registry: Any = None,
    ) -> None:
        self.config = config
        self.aggregator = aggregator
        self.ollama = ollama
        self.tracer = tracer
        self.cache = cache
        self.plugin_registry = plugin_registry

    async def run(
        self,
        prompt: str,
        tools_whitelist: list[str] | None = None,
        dry_run: bool = False,
        stream: bool = False,
        prior_messages: list[dict[str, Any]] | None = None,
        cache_enabled: bool | None = None,
        cache_ttl: int | None = None,
        system_prompt_override: str | None = None,
        event_queue: asyncio.Queue | None = None,
    ) -> LoopResult:
        """Run a single agent loop turn.

        Args:
            prompt: The task or question.
            tools_whitelist: Glob patterns to whitelist tools.
            dry_run: Preview tool calls without executing.
            stream: Stream LLM tokens to stdout (chat/run --stream mode).
            prior_messages: Existing conversation history to seed this turn.
            cache_enabled: Override config.cache.enabled for this run.
            cache_ttl: Override config.cache.ttl_seconds for this run.
            system_prompt_override: Additional text appended to base system prompt.
            event_queue: If set, stream tokens and tool events into this queue for SSE delivery.
        """
        start_time = time.monotonic()

        # --- Resolve cache settings ---
        effective_cache_enabled = (
            cache_enabled if cache_enabled is not None else self.cache.config.enabled
        )
        if cache_ttl is not None:
            self.cache.config.ttl_seconds = cache_ttl

        # --- Collect + sanitize + filter tools ---
        all_tools = await self.aggregator.collect_tools()

        if self.config.security.sanitize_tool_descriptions:
            all_tools = [sanitize_tool(t) for t in all_tools]

        if self.plugin_registry:
            custom_filter = self.plugin_registry.get("tool_filter")
            if custom_filter:
                all_tools = custom_filter(all_tools, prompt)

        # Build filter prompt: include last 2 user turns from history so follow-up
        # queries ("now sort by date") still match the same server as the prior turn.
        filter_prompt = prompt
        if prior_messages:
            recent_user = [
                m["content"] for m in prior_messages[-8:]
                if m.get("role") == "user" and m.get("content")
            ]
            if recent_user:
                filter_prompt = " ".join(recent_user[-2:]) + " " + prompt

        filtered_tools = apply_filter(
            all_tools,
            filter_prompt,
            self.config.tool_filter,
            whitelist=tools_whitelist,
        )

        # --- Determine LLM mode ---
        supports_native = await self.ollama.check_capability()

        # --- Build system prompt ---
        base_prompt = _load_prompt("base.txt")

        if self.plugin_registry:
            prompt_hook = self.plugin_registry.get("system_prompt")
            if prompt_hook:
                base_prompt = prompt_hook(base_prompt)

        if system_prompt_override:
            base_prompt = base_prompt + "\n\n" + system_prompt_override

        if not supports_native:
            fallback_addendum = _load_prompt("fallback.txt")
            system_content = f"{base_prompt}\n\n{fallback_addendum}"
        else:
            system_content = base_prompt

        degraded_notice = self.aggregator.degraded_notice()
        if degraded_notice:
            system_content += f"\n\n{degraded_notice}"

        # --- Initialize session ---
        session = RunSession()
        session.add_system(system_content)

        if prior_messages:
            for msg in prior_messages:
                if msg.get("role") != "system":
                    session.messages.append(msg)

        session.add_user(prompt)

        retry_state = RetryState(
            max_per_tool=self.config.execution.max_retries_per_tool,
            max_total=self.config.execution.max_iterations * self.config.execution.max_retries_per_tool,
        )

        tool_calls_made = 0
        cache_hits = 0
        _call_counts: dict[str, int] = {}  # (tool_name:args_repr) → count for loop detection

        # --- Main agent loop ---
        for iteration in range(self.config.execution.max_iterations):
            session.iteration = iteration

            elapsed_ms = (time.monotonic() - start_time) * 1000
            if elapsed_ms > self.config.execution.total_timeout_ms:
                return LoopResult(
                    answer="[Run aborted: total timeout exceeded]",
                    tool_calls_made=tool_calls_made,
                    cache_hits=cache_hits,
                    iterations=iteration,
                    total_latency_ms=elapsed_ms,
                    exit_reason="timeout",
                    success=False,
                    messages=session.snapshot(),
                )

            mid_notice = self.aggregator.degraded_notice()
            if mid_notice and iteration > 0:
                session.inject_notice(mid_notice)

            try:
                if event_queue is not None:
                    # Use chat_once for reliability — Ollama streaming is inconsistent
                    # about including tool_calls in the done chunk across versions.
                    response = await self.ollama.chat_once(session.messages, filtered_tools)
                    # Emit any text content as a token event for the frontend.
                    _content_preview = response.get("message", {}).get("content", "") or ""
                    if _content_preview:
                        await event_queue.put({"type": "token", "content": _content_preview})
                elif stream:
                    response = await self._stream_and_collect(session.messages, filtered_tools)
                else:
                    response = await self.ollama.chat_once(session.messages, filtered_tools)
            except Exception as exc:
                return LoopResult(
                    answer=f"[LLM error: {exc}]",
                    tool_calls_made=tool_calls_made,
                    cache_hits=cache_hits,
                    iterations=iteration,
                    total_latency_ms=(time.monotonic() - start_time) * 1000,
                    exit_reason="error",
                    success=False,
                    messages=session.snapshot(),
                )

            msg = response.get("message", {})
            content: str = msg.get("content", "") or ""
            native_tool_calls: list[dict] = msg.get("tool_calls", [])

            fallback_calls = []
            if not supports_native and not native_tool_calls:
                fallback_calls = parse_tool_calls(content)
                if not fallback_calls:
                    repaired = auto_repair_json(content)
                    if repaired != content:
                        fallback_calls = parse_tool_calls(repaired)
                if fallback_calls:
                    content = strip_tool_call_blocks(content)

            if not native_tool_calls and not fallback_calls:
                return LoopResult(
                    answer=content.strip(),
                    tool_calls_made=tool_calls_made,
                    cache_hits=cache_hits,
                    iterations=iteration + 1,
                    total_latency_ms=(time.monotonic() - start_time) * 1000,
                    exit_reason="completed",
                    success=True,
                    messages=session.snapshot(),
                )

            if supports_native:
                session.add_assistant(content, native_tool_calls)
            else:
                raw_content = msg.get("content", "")
                session.add_assistant(raw_content)

            calls_to_process = _normalize_tool_calls(native_tool_calls, fallback_calls)

            # Loop-break: detect repeated identical tool calls (e.g. echo-loop)
            for _tc_name, _tc_args in calls_to_process:
                _call_key = f"{_tc_name}:{repr(sorted(_tc_args.items()) if isinstance(_tc_args, dict) else _tc_args)}"
                _call_counts[_call_key] = _call_counts.get(_call_key, 0) + 1
                if _call_counts[_call_key] >= 3:
                    return LoopResult(
                        answer=f"[Run stopped: tool '{_tc_name}' called with identical arguments 3 times — possible loop. Please rephrase your request.]",
                        tool_calls_made=tool_calls_made,
                        cache_hits=cache_hits,
                        iterations=iteration + 1,
                        total_latency_ms=(time.monotonic() - start_time) * 1000,
                        exit_reason="loop_detected",
                        success=False,
                        messages=session.snapshot(),
                    )

            if self.config.execution.parallel_tools and len(calls_to_process) > 1:
                added_calls, added_hits = await self._execute_parallel(
                    calls_to_process, filtered_tools, session, supports_native,
                    retry_state, iteration, effective_cache_enabled,
                    dry_run=dry_run, event_queue=event_queue,
                )
                tool_calls_made += added_calls
                cache_hits += added_hits
            else:
                for tc_name, tc_args in calls_to_process:
                    tool_calls_made, cache_hits = await self._process_single_call(
                        tc_name, tc_args, filtered_tools, session, supports_native,
                        retry_state, iteration, effective_cache_enabled,
                        tool_calls_made, cache_hits, start_time,
                        dry_run=dry_run, event_queue=event_queue,
                    )

        return LoopResult(
            answer="[Run stopped: maximum iterations reached]",
            tool_calls_made=tool_calls_made,
            cache_hits=cache_hits,
            iterations=self.config.execution.max_iterations,
            total_latency_ms=(time.monotonic() - start_time) * 1000,
            exit_reason="max_iterations",
            success=False,
            messages=session.snapshot(),
        )

    async def _stream_and_collect(
        self,
        messages: list[dict[str, Any]],
        filtered_tools: list[ToolInfo],
    ) -> dict[str, Any]:
        """Stream tokens to stdout and return the assembled response dict."""
        import sys
        full_content = ""
        final_response: dict[str, Any] = {}
        async for chunk in self.ollama.chat_stream(messages, filtered_tools):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                print(delta, end="", flush=True)
                full_content += delta
            if chunk.get("done"):
                final_response = chunk
        print()
        if "message" not in final_response:
            final_response["message"] = {}
        final_response["message"]["content"] = full_content
        return final_response

    async def _stream_to_queue(
        self,
        messages: list[dict[str, Any]],
        filtered_tools: list[ToolInfo],
        queue: asyncio.Queue,
    ) -> dict[str, Any]:
        """Stream tokens into an asyncio.Queue for SSE delivery."""
        full_content = ""
        final_response: dict[str, Any] = {}
        async for chunk in self.ollama.chat_stream(messages, filtered_tools):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                await queue.put({"type": "token", "content": delta})
                full_content += delta
            if chunk.get("done"):
                final_response = chunk
        if "message" not in final_response:
            final_response["message"] = {}
        final_response["message"]["content"] = full_content
        return final_response

    async def _process_single_call(
        self,
        tc_name: str,
        tc_args: dict,
        filtered_tools: list[ToolInfo],
        session: RunSession,
        supports_native: bool,
        retry_state: RetryState,
        iteration: int,
        effective_cache_enabled: bool,
        tool_calls_made: int,
        cache_hits: int,
        start_time: float,
        dry_run: bool = False,
        event_queue: asyncio.Queue | None = None,
    ) -> tuple[int, int]:
        """Process a single tool call and update session. Returns (tool_calls_made, cache_hits)."""
        tool_calls_made += 1
        retry_key = f"{tc_name}:{iteration}"
        trace_id = new_trace_id()
        call_start = time.monotonic()

        tool_info = _find_tool(filtered_tools, tc_name)
        if tool_info is None:
            available = [t.full_name for t in filtered_tools]
            correction = build_correction_message(
                RetryReason.HALLUCINATION, tc_name, available
            )
            session.add_tool_result(correction, native_mode=supports_native)
            self.tracer.finish_tool_call(
                trace_id, iteration, "?", tc_name, tc_args,
                input_valid=False, output=None, cache_hit=False,
                retry_count=0, start_time=call_start,
                status="error", error="hallucination",
                failure_reason="hallucination",
            )
            return tool_calls_made, cache_hits

        valid, errors, coerced_args = validate_and_coerce(
            tool_info.input_schema, tc_args
        )
        if not valid:
            correction = build_correction_message(
                RetryReason.SCHEMA_FAILURE, tc_name, [], "; ".join(errors)
            )
            session.add_tool_result(correction, native_mode=supports_native)
            self.tracer.finish_tool_call(
                trace_id, iteration, tool_info.server, tool_info.name,
                tc_args, input_valid=False, output=None, cache_hit=False,
                retry_count=0, start_time=call_start,
                status="error", error="; ".join(errors),
                failure_reason="schema_failure",
            )
            return tool_calls_made, cache_hits

        # Emit tool_start for SSE streaming
        if event_queue is not None:
            await event_queue.put({
                "type": "tool_start",
                "tool": tool_info.name,
                "server": tool_info.server,
                "args": coerced_args,
            })

        if dry_run:
            return await self._handle_dry_run(
                tc_name, coerced_args, tool_info, session, supports_native,
                trace_id, iteration, call_start, tool_calls_made, cache_hits,
            )

        cache_key = None
        if effective_cache_enabled and not self.cache.is_excluded(tool_info.full_name):
            cache_key = self.cache.make_key(tc_name, coerced_args, tool_info.input_schema)
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                cache_hits += 1
                result_str = _truncate_tool_result(
                    str(cached_value), self.config.execution.max_tool_result_bytes
                )
                session.add_tool_result(result_str, native_mode=supports_native)
                self.tracer.finish_tool_call(
                    trace_id, iteration, tool_info.server, tool_info.name,
                    coerced_args, input_valid=True, output=cached_value,
                    cache_hit=True, retry_count=0, start_time=call_start,
                    status="cache_hit",
                )
                if event_queue is not None:
                    await event_queue.put({
                        "type": "tool_done",
                        "tool": tool_info.name,
                        "server": tool_info.server,
                        "status": "cache_hit",
                        "latency_ms": round((time.monotonic() - call_start) * 1000, 1),
                        "output": str(cached_value)[:400],
                        "cache_hit": True,
                    })
                return tool_calls_made, cache_hits

        result_str, exec_status, exec_error = await self._execute_with_retry(
            tool_info, coerced_args, retry_state, retry_key, iteration
        )

        if exec_status == "success" and cache_key is not None and effective_cache_enabled:
            self.cache.set(cache_key, result_str, tool_info)

        result_str = _truncate_tool_result(result_str, self.config.execution.max_tool_result_bytes)
        session.add_tool_result(result_str, native_mode=supports_native)
        self.tracer.finish_tool_call(
            trace_id, iteration, tool_info.server, tool_info.name,
            coerced_args, input_valid=True, output=result_str,
            cache_hit=False,
            retry_count=retry_state.count_for(retry_key),
            start_time=call_start,
            status=exec_status,
            error=exec_error,
        )
        if event_queue is not None:
            await event_queue.put({
                "type": "tool_done",
                "tool": tool_info.name,
                "server": tool_info.server,
                "status": exec_status,
                "latency_ms": round((time.monotonic() - call_start) * 1000, 1),
                "output": str(result_str)[:400] if result_str else None,
                "cache_hit": False,
                "error": exec_error,
            })
        return tool_calls_made, cache_hits

    async def _handle_dry_run(
        self,
        tc_name: str,
        coerced_args: dict,
        tool_info: ToolInfo,
        session: RunSession,
        supports_native: bool,
        trace_id: str,
        iteration: int,
        call_start: float,
        tool_calls_made: int,
        cache_hits: int,
    ) -> tuple[int, int]:
        import json as _json
        dry_msg = f"[DRY RUN] Would call: {tc_name}({_json.dumps(coerced_args)})"
        session.add_tool_result(
            f"Dry run: '{tc_name}' not executed.", native_mode=supports_native
        )
        self.tracer.finish_tool_call(
            trace_id, iteration, tool_info.server, tool_info.name,
            coerced_args, input_valid=True, output=None, cache_hit=False,
            retry_count=0, start_time=call_start, status="dry_run",
        )
        print(dry_msg, flush=True)
        return tool_calls_made, cache_hits

    async def _execute_parallel(
        self,
        calls_to_process: list[tuple[str, dict]],
        filtered_tools: list[ToolInfo],
        session: RunSession,
        supports_native: bool,
        retry_state: RetryState,
        iteration: int,
        effective_cache_enabled: bool,
        dry_run: bool = False,
        event_queue: asyncio.Queue | None = None,
    ) -> tuple[int, int]:
        """Execute multiple independent tool calls concurrently via asyncio.gather."""
        import sys

        group_id = new_trace_id()
        print(
            f"\n[umcp] Parallel execution: {len(calls_to_process)} tool calls (group={group_id})",
            file=sys.stderr, flush=True,
        )

        async def _run_one(tc_name: str, tc_args: dict) -> tuple[str, str, bool]:
            """Returns (result_str, status, cache_hit)."""
            _call_start = time.monotonic()
            tool_info = _find_tool(filtered_tools, tc_name)
            if tool_info is None:
                return f"Tool '{tc_name}' not found.", "error", False
            valid, errors, coerced = validate_and_coerce(tool_info.input_schema, tc_args)
            if not valid:
                return f"Schema error: {'; '.join(errors)}", "error", False
            if event_queue is not None:
                await event_queue.put({
                    "type": "tool_start",
                    "tool": tool_info.name,
                    "server": tool_info.server,
                    "args": coerced,
                })
            if dry_run:
                import json as _json
                print(f"[DRY RUN] Would call: {tc_name}({_json.dumps(coerced)})", flush=True)
                if event_queue is not None:
                    await event_queue.put({
                        "type": "tool_done", "tool": tool_info.name,
                        "server": tool_info.server, "status": "dry_run",
                        "latency_ms": 0, "output": None, "cache_hit": False,
                    })
                return f"Dry run: '{tc_name}' not executed.", "dry_run", False
            if effective_cache_enabled and not self.cache.is_excluded(tool_info.full_name):
                cache_key = self.cache.make_key(tc_name, coerced, tool_info.input_schema)
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    if event_queue is not None:
                        latency = round((time.monotonic() - _call_start) * 1000, 1)
                        await event_queue.put({
                            "type": "tool_done", "tool": tool_info.name,
                            "server": tool_info.server, "status": "cache_hit",
                            "latency_ms": latency, "output": str(cached_value)[:400],
                            "cache_hit": True,
                        })
                    return str(cached_value), "cache_hit", True
            result_str, status, _ = await self._execute_with_retry(
                tool_info, coerced, retry_state, f"{tc_name}:{iteration}", iteration
            )
            if event_queue is not None:
                latency = round((time.monotonic() - _call_start) * 1000, 1)
                await event_queue.put({
                    "type": "tool_done", "tool": tool_info.name,
                    "server": tool_info.server, "status": status,
                    "latency_ms": latency, "output": str(result_str)[:400] if result_str else None,
                    "cache_hit": False,
                })
            return result_str, status, False

        results = await asyncio.gather(*[_run_one(n, a) for n, a in calls_to_process])

        total_calls = 0
        total_hits = 0
        for (tc_name, _), (result_str, _status, was_cache_hit) in zip(calls_to_process, results):
            result_str = _truncate_tool_result(result_str, self.config.execution.max_tool_result_bytes)
            session.add_tool_result(result_str, native_mode=supports_native)
            total_calls += 1
            if was_cache_hit:
                total_hits += 1

        return total_calls, total_hits

    async def _execute_with_retry(
        self,
        tool_info: ToolInfo,
        arguments: dict,
        retry_state: RetryState,
        retry_key: str,
        iteration: int,
    ) -> tuple[str, str, str | None]:
        """Execute a tool call with transport-error retry + degradation logic."""
        consecutive_failures = 0

        for attempt in range(self.config.execution.max_retries_per_tool + 1):
            transport = self.aggregator.get_transport(tool_info.server)
            if transport is None or self.aggregator.is_degraded(tool_info.server):
                return (
                    f"Server '{tool_info.server}' is unavailable.",
                    "error",
                    "server_degraded",
                )

            result = await transport.call_tool(
                tool_info.name,
                arguments,
                self.config.execution.tool_timeout_ms,
            )

            if result.success:
                return str(result.content), "success", None

            error_msg = result.error or "unknown error"
            consecutive_failures += 1

            if consecutive_failures >= 3:
                self.aggregator.mark_degraded(tool_info.server)
                return (
                    f"Server '{tool_info.server}' has been marked unavailable after repeated failures.",
                    "error",
                    "server_degraded",
                )

            decision = decide_retry(
                RetryReason.TOOL_ERROR,
                retry_state.count_for(retry_key),
                self.config.execution.max_retries_per_tool,
            )

            if decision == RetryDecision.ABORT:
                return f"Tool error: {error_msg}", "error", error_msg

            retry_state.increment(retry_key)
            await backoff_sleep(attempt)

        return f"Tool failed after {self.config.execution.max_retries_per_tool} retries.", "error", "max_retries"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _truncate_tool_result(result: str, max_bytes: int) -> str:
    """Truncate a tool result string to max_bytes UTF-8 bytes."""
    if not result:
        return result
    encoded = result.encode("utf-8")
    if len(encoded) <= max_bytes:
        return result
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + f"\n[... truncated at {max_bytes // 1024} KB — ask for a specific subset]"


def _find_tool(tools: list[ToolInfo], full_name: str) -> ToolInfo | None:
    # Exact match first
    result = next((t for t in tools if t.full_name == full_name), None)
    if result:
        return result
    # Fuzzy match — handles LLM mis-naming real tools (e.g. "sqlite.query" → "sqlite.read_query")
    import difflib
    names = [t.full_name for t in tools]
    matches = difflib.get_close_matches(full_name, names, n=1, cutoff=0.9)
    if matches:
        get_logger().debug("fuzzy_tool_match", requested=full_name, matched=matches[0])
        return next((t for t in tools if t.full_name == matches[0]), None)
    return None


def _normalize_tool_calls(
    native: list[dict],
    fallback: list,
) -> list[tuple[str, dict]]:
    """Produce a uniform list of (tool_full_name, arguments) pairs."""
    result = []
    for tc in native:
        fn = tc.get("function", tc)
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        result.append((name, args))
    for ftc in fallback:
        result.append((ftc.name, ftc.arguments))
    return result
