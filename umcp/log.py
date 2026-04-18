"""Application-wide structured logging via structlog."""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_configured = False


def configure(level: str = "info", output: str = "stderr") -> None:
    """Configure structlog processors. Call once at startup with AppConfig.logging settings."""
    global _configured

    log_level = getattr(logging, level.upper(), logging.INFO)
    use_json = os.environ.get("LOG_FORMAT", "").lower() == "json" or output == "file"

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    processors = shared + (
        [structlog.processors.JSONRenderer()]
        if use_json
        else [structlog.dev.ConsoleRenderer(colors=False)]
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "umcp") -> Any:
    if not _configured:
        _default()
    return structlog.get_logger(name)


def _default() -> None:
    global _configured
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True
