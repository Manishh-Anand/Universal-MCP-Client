"""Observability — structured trace entries for every tool call."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_LAST_TRACE_PATH = Path.home() / ".config" / "umcp" / "last_trace.json"
_LIVE_TRACE_PATH = Path.home() / ".config" / "umcp" / "trace_live.jsonl"


@dataclass
class TraceEntry:
    trace_id: str
    session_id: str
    iteration: int
    timestamp: str
    server: str
    tool: str
    input: dict[str, Any]
    input_valid: bool
    output: Any
    cache_hit: bool
    retry_count: int
    latency_ms: float
    status: str          # "success" | "error" | "timeout" | "dry_run" | "cache_hit"
    error: str | None = None
    failure_reason: str | None = None
    group_id: str | None = None   # Phase 3.5: parallel execution group identifier


class Tracer:
    """Collects trace entries and emits them to stderr as structured JSON."""

    def __init__(self, session_id: str, enabled: bool = True) -> None:
        self.session_id = session_id
        self.enabled = enabled
        self._entries: list[TraceEntry] = []
        # Clear the live file so `trace tail` always shows the current run only
        if enabled:
            try:
                _LIVE_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _LIVE_TRACE_PATH.write_text("", encoding="utf-8")
            except Exception:
                pass

    def record(self, entry: TraceEntry) -> None:
        self._entries.append(entry)
        if self.enabled:
            _emit(entry)
            _append_live(entry)

    def start_tool_call(
        self,
        trace_id: str,
        iteration: int,
        server: str,
        tool: str,
        input_args: dict[str, Any],
        input_valid: bool,
    ) -> float:
        """Record the start of a tool call. Returns the start timestamp."""
        return time.monotonic()

    def finish_tool_call(
        self,
        trace_id: str,
        iteration: int,
        server: str,
        tool: str,
        input_args: dict[str, Any],
        input_valid: bool,
        output: Any,
        cache_hit: bool,
        retry_count: int,
        start_time: float,
        status: str,
        error: str | None = None,
        failure_reason: str | None = None,
        group_id: str | None = None,
    ) -> TraceEntry:
        """Record completion of a tool call and return the entry."""
        entry = TraceEntry(
            trace_id=trace_id,
            session_id=self.session_id,
            iteration=iteration,
            timestamp=_iso_now(),
            server=server,
            tool=tool,
            input=input_args,
            input_valid=input_valid,
            output=output,
            cache_hit=cache_hit,
            retry_count=retry_count,
            latency_ms=round((time.monotonic() - start_time) * 1000, 2),
            status=status,
            error=error,
            failure_reason=failure_reason,
            group_id=group_id,
        )
        self.record(entry)
        return entry

    def all_entries(self) -> list[TraceEntry]:
        return list(self._entries)

    def save_last(self) -> None:
        """Persist current trace to disk for `umcp trace last`."""
        try:
            _LAST_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(e) for e in self._entries]
            _LAST_TRACE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def save_session(self, session_id: str, entries: list["TraceEntry"] | None = None) -> None:
        """Persist trace to a named session file.

        If entries is None, saves current entries.
        """
        sessions_dir = Path.home() / ".config" / "umcp" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        path = sessions_dir / f"{safe}_trace.json"
        data_entries = entries if entries is not None else self._entries
        try:
            path.write_text(
                json.dumps([asdict(e) for e in data_entries], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @staticmethod
    def load_last() -> list[dict[str, Any]]:
        """Load the last saved trace from disk."""
        if not _LAST_TRACE_PATH.exists():
            return []
        try:
            return json.loads(_LAST_TRACE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []

    @staticmethod
    def load_session(session_id: str) -> list[dict[str, Any]]:
        """Load trace for a named session."""
        sessions_dir = Path.home() / ".config" / "umcp" / "sessions"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        path = sessions_dir / f"{safe}_trace.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []


def _emit(entry: TraceEntry) -> None:
    """Write a single trace entry as JSON to stderr."""
    try:
        line = json.dumps(asdict(entry), default=str)
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


def _append_live(entry: TraceEntry) -> None:
    """Append a single entry to the live JSONL file for `trace tail`."""
    try:
        line = json.dumps(asdict(entry), default=str) + "\n"
        with _LIVE_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
