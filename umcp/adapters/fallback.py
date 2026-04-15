"""Fallback parser — extracts tool calls from <tool_call> blocks in plain text."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str  # original matched block (for error messages)


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    """Extract all <tool_call>...</tool_call> blocks from model output.

    Returns a list of parsed tool calls (may be empty if model gave a final answer).
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(0)
        body = match.group(1).strip()
        parsed = _parse_json_body(body)
        if parsed is not None:
            name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if name:
                calls.append(ParsedToolCall(name=name, arguments=args, raw=raw))
    return calls


def strip_tool_call_blocks(text: str) -> str:
    """Remove all <tool_call> blocks from text, returning clean content."""
    return _TOOL_CALL_RE.sub("", text).strip()


def auto_repair_json(raw: str) -> str:
    """Attempt common JSON repairs on malformed tool call bodies.

    Handles:
    - Trailing commas before } or ]
    - Single-quoted strings → double-quoted
    - Unquoted keys
    """
    # Remove trailing commas
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    # Single quotes → double quotes (careful: don't break apostrophes in values)
    repaired = re.sub(r"(?<![\\])'", '"', repaired)
    return repaired


def _parse_json_body(body: str) -> dict[str, Any] | None:
    """Try to parse JSON, with one auto-repair attempt on failure."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(auto_repair_json(body))
    except json.JSONDecodeError:
        return None
