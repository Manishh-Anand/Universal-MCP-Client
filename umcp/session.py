"""Run session — holds conversation message history for one agent loop run.

Phase 3 adds SessionStore for persistent chat history across turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SESSIONS_DIR = Path.home() / ".config" / "umcp" / "sessions"


@dataclass
class RunSession:
    """Conversation state for a single agent loop execution."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    session_id: str | None = None

    def add_system(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, content: str, native_mode: bool = True) -> None:
        """Append a tool result.

        Native mode: role="tool" (Ollama understands this).
        Fallback mode: role="user" (simulate tool result as user message).
        """
        if native_mode:
            self.messages.append({"role": "tool", "content": content})
        else:
            self.messages.append({"role": "user", "content": content})

    def inject_notice(self, notice: str) -> None:
        """Inject an informational notice as a user message (e.g., server degraded)."""
        self.messages.append({"role": "user", "content": notice})

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.messages)


class SessionStore:
    """Manages persistent chat sessions on disk.

    Each session is stored as a JSON file at:
        ~/.config/umcp/sessions/<session_id>.json

    Format: list of message dicts (role/content/tool_calls).
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._dir = Path(storage_path) if storage_path else _SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        # Sanitize session_id to safe filename
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        return self._dir / f"{safe}.json"

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load message history for a session. Returns [] if not found."""
        path = self._path(session_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist message history for a session."""
        try:
            self._path(session_id).write_text(
                json.dumps(messages, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def list_sessions(self) -> list[str]:
        """Return all known session IDs (stem of .json files)."""
        try:
            return [
                p.stem
                for p in sorted(self._dir.glob("*.json"))
                if not p.stem.endswith("_trace")
            ]
        except Exception:
            return []

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def save_trace(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        """Persist trace entries for a named session."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        trace_path = self._dir / f"{safe}_trace.json"
        try:
            trace_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Load trace entries for a named session."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        trace_path = self._dir / f"{safe}_trace.json"
        if not trace_path.exists():
            return []
        try:
            return json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            return []
