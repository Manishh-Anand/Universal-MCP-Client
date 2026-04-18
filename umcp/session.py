"""Run session — holds conversation message history for one agent loop run."""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from .log import get_logger

_log = get_logger()
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _warn(msg: str) -> None:
    _log.warning(msg)

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
        if native_mode:
            self.messages.append({"role": "tool", "content": content})
        else:
            self.messages.append({"role": "user", "content": content})

    def inject_notice(self, notice: str) -> None:
        self.messages.append({"role": "user", "content": notice})

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.messages)


class SessionStore:
    """Manages persistent chat sessions on disk.

    Each session is two files:
        ~/.config/umcp/sessions/<id>.json       — message history
        ~/.config/umcp/sessions/<id>_meta.json  — title, timestamps, turn count
    """

    _LRU_SIZE = 20

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._dir = Path(storage_path).expanduser() if storage_path else _SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lru: OrderedDict[str, list[dict]] = OrderedDict()

    def _safe(self, session_id: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{self._safe(session_id)}.json"

    def _meta_path(self, session_id: str) -> Path:
        return self._dir / f"{self._safe(session_id)}_meta.json"

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #

    def load(self, session_id: str) -> list[dict[str, Any]]:
        if session_id in self._lru:
            self._lru.move_to_end(session_id)
            return list(self._lru[session_id])
        path = self._path(session_id)
        if not path.exists():
            return []
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        self._lru[session_id] = messages
        self._lru.move_to_end(session_id)
        if len(self._lru) > self._LRU_SIZE:
            self._lru.popitem(last=False)
        return list(messages)

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self._lru.pop(session_id, None)  # invalidate before write
        try:
            self._path(session_id).write_text(
                json.dumps(messages, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            _warn(f"session save failed for {session_id!r}: {exc}")

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def save_meta(self, session_id: str, title: str, turn_count: int) -> None:
        """Create or update session metadata."""
        meta_path = self._meta_path(session_id)
        now = datetime.now().isoformat(timespec="seconds")
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                existing["updated_at"] = now
                existing["turn_count"] = turn_count
                meta_path.write_text(json.dumps(existing), encoding="utf-8")
                return
            except Exception as exc:
                _warn(f"meta update failed for {session_id!r}: {exc}")
        meta = {
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "turn_count": turn_count,
        }
        try:
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as exc:
            _warn(f"meta create failed for {session_id!r}: {exc}")

    def load_meta(self, session_id: str) -> dict[str, Any] | None:
        meta_path = self._meta_path(session_id)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_sessions_with_meta(self) -> list[dict[str, Any]]:
        """Return all sessions sorted by last update, with metadata."""
        sessions = []
        try:
            for p in sorted(
                self._dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            ):
                stem = p.stem
                if stem.endswith("_trace") or stem.endswith("_meta"):
                    continue
                meta = self.load_meta(stem)
                if meta:
                    sessions.append(meta)
                else:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
                    sessions.append({
                        "session_id": stem,
                        "title": stem,
                        "created_at": mtime,
                        "updated_at": mtime,
                        "turn_count": 0,
                    })
        except Exception:
            pass
        return sessions

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def list_sessions(self) -> list[str]:
        try:
            return [
                p.stem
                for p in sorted(self._dir.glob("*.json"))
                if not p.stem.endswith("_trace") and not p.stem.endswith("_meta")
            ]
        except Exception:
            return []

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        meta_path = self._meta_path(session_id)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
        return deleted

    # ------------------------------------------------------------------ #
    # Trace
    # ------------------------------------------------------------------ #

    def save_trace(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        safe = self._safe(session_id)
        trace_path = self._dir / f"{safe}_trace.json"
        try:
            trace_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_trace(self, session_id: str) -> list[dict[str, Any]]:
        safe = self._safe(session_id)
        trace_path = self._dir / f"{safe}_trace.json"
        if not trace_path.exists():
            return []
        try:
            return json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            return []
