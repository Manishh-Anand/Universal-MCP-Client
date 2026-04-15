"""Security — sanitizes tool descriptions and enforces trusted-server policy."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transports.base import ToolInfo
    from .config import SecurityConfig

# Patterns that suggest prompt injection attempts in tool descriptions
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<[^>]{1,100}>", re.IGNORECASE),           # HTML/XML tags
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),      # Llama instruction tokens
    re.compile(r"<<SYS>>|<</SYS>>", re.IGNORECASE),        # Llama system tokens
    re.compile(r"ignore (all )?(previous|prior) instructions?", re.IGNORECASE),
    re.compile(r"you (must|should|are now) (be |act |pretend)", re.IGNORECASE),
    re.compile(r"disregard (all |your )?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"new (role|persona|instruction|directive)", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
]

_MAX_DESCRIPTION_LENGTH = 512


def sanitize_description(text: str) -> str:
    """Clean a tool description to remove prompt injection attempts."""
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[removed]", text)
    # Truncate
    if len(text) > _MAX_DESCRIPTION_LENGTH:
        text = text[:_MAX_DESCRIPTION_LENGTH] + "…"
    return text.strip()


def sanitize_tool(tool: "ToolInfo") -> "ToolInfo":
    """Return a copy of the tool with its description sanitized."""
    from dataclasses import replace
    return replace(tool, description=sanitize_description(tool.description))


def check_server_trusted(server_name: str, config: "SecurityConfig") -> bool:
    """Return True if the server is allowed under the current security config."""
    if not config.trusted_servers_only:
        return True
    return server_name in config.trusted_servers


def mask_secrets(value: str) -> str:
    """Replace the actual value of an env var with ***."""
    return "***"
