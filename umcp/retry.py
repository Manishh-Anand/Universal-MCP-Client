"""Typed retry strategies for the agent loop."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetryReason(Enum):
    TOOL_ERROR = "tool_error"          # tool returned an error response
    HALLUCINATION = "hallucination"    # LLM called a non-existent tool
    INVALID_JSON = "invalid_json"      # LLM output could not be parsed
    SCHEMA_FAILURE = "schema_failure"  # tool args failed schema validation
    TRANSPORT_ERROR = "transport_error"  # network / subprocess error


class RetryDecision(Enum):
    RETRY_TOOL = "retry_tool"    # re-execute the same tool call
    REPROMPT = "reprompt"        # inject correction message, let LLM try again
    ABORT = "abort"              # give up on this tool call


@dataclass
class RetryState:
    """Tracks retry counts per tool call key within one agent run."""
    _counts: dict[str, int] = field(default_factory=dict)
    _total: int = 0
    max_per_tool: int = 2
    max_total: int = 20

    def increment(self, key: str) -> int:
        """Increment retry count for key. Returns new count."""
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1
        return self._counts[key]

    def count_for(self, key: str) -> int:
        return self._counts.get(key, 0)

    def budget_exceeded(self, key: str) -> bool:
        return (
            self._counts.get(key, 0) >= self.max_per_tool
            or self._total >= self.max_total
        )


def decide_retry(reason: RetryReason, retry_count: int, max_retries: int) -> RetryDecision:
    """Given a failure reason and retry history, decide what to do next."""
    if retry_count >= max_retries:
        return RetryDecision.ABORT

    if reason == RetryReason.TOOL_ERROR:
        return RetryDecision.RETRY_TOOL

    if reason in (RetryReason.HALLUCINATION, RetryReason.SCHEMA_FAILURE):
        return RetryDecision.REPROMPT

    if reason == RetryReason.INVALID_JSON:
        # First attempt: retry (may have been a one-off parse failure)
        # Second attempt: reprompt with explicit format instruction
        return RetryDecision.RETRY_TOOL if retry_count == 0 else RetryDecision.REPROMPT

    if reason == RetryReason.TRANSPORT_ERROR:
        return RetryDecision.RETRY_TOOL

    return RetryDecision.ABORT


def build_correction_message(
    reason: RetryReason,
    tool_name: str,
    available_tools: list[str],
    error_detail: str = "",
) -> str:
    """Build the correction message to inject into the conversation."""
    if reason == RetryReason.HALLUCINATION:
        tool_list = ", ".join(available_tools[:20])
        return (
            f"Error: Tool '{tool_name}' does not exist. "
            f"You MUST only call tools from this list: [{tool_list}]. "
            f"Try again using an available tool."
        )
    if reason == RetryReason.SCHEMA_FAILURE:
        return (
            f"Error: Invalid arguments for tool '{tool_name}': {error_detail}. "
            f"Check the tool's parameter schema and retry with correct arguments."
        )
    if reason == RetryReason.INVALID_JSON:
        return (
            f"Error: Your tool call response could not be parsed as valid JSON. "
            f"Output ONLY a valid JSON object inside <tool_call>...</tool_call>. "
            f"No extra text."
        )
    if reason == RetryReason.TOOL_ERROR:
        return (
            f"Tool '{tool_name}' returned an error: {error_detail}. "
            f"You may retry or use an alternative approach."
        )
    return f"Error calling '{tool_name}': {error_detail}. Please try again."


async def backoff_sleep(attempt: int) -> None:
    """Exponential backoff: 100ms, 200ms, 400ms."""
    delay = 0.1 * (2 ** min(attempt, 2))
    await asyncio.sleep(delay)
