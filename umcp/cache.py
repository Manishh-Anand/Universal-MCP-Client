"""Response cache — TTL-aware cache with in-memory and file-based backends.

Phase 4 adds FileCache backed by ~/.config/umcp/cache/<key>.json.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .transports.base import ToolInfo
    from .config import CacheConfig

_FILE_CACHE_DIR = Path.home() / ".config" / "umcp" / "cache"


@dataclass
class CacheEntry:
    value: Any
    expires_at: float  # monotonic timestamp


class ResponseCache:
    """Cache for tool call responses.

    Dispatches to in-memory or file backend based on config.storage.
    Key = hash(tool_full_name + schema_version + normalized_args)
    """

    def __init__(self, config: "CacheConfig") -> None:
        self.config = config
        self._store: dict[str, CacheEntry] = {}   # in-memory store
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        full_name: str,
        arguments: dict[str, Any],
        schema: dict[str, Any],
    ) -> str:
        """Build a normalized, deterministic cache key."""
        schema_version = _schema_fingerprint(schema)
        normalized_args = _normalize_args(arguments)
        raw = f"{full_name}|{schema_version}|{json.dumps(normalized_args, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        if self.config.storage == "file":
            return self._file_get(key)
        return self._mem_get(key)

    def set(self, key: str, value: Any, tool: "ToolInfo") -> None:
        """Store a value. Skips if tool matches an exclude pattern or cache is disabled."""
        if not self.config.enabled:
            return
        if self._is_excluded(tool.full_name):
            return
        if self.config.storage == "file":
            self._file_set(key, value)
        else:
            self._mem_set(key, value)

    def is_excluded(self, full_name: str) -> bool:
        return self._is_excluded(full_name)

    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        if self.config.storage == "file":
            return self._file_clear()
        return self._mem_clear()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        base = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "max_size": self.config.max_size,
            "storage": self.config.storage,
        }
        if self.config.storage == "file":
            base["entries"] = len(list(_FILE_CACHE_DIR.glob("*.json"))) if _FILE_CACHE_DIR.exists() else 0
        else:
            base["entries"] = len(self._store)
        return base

    # ------------------------------------------------------------------ #
    # In-memory backend
    # ------------------------------------------------------------------ #

    def _mem_get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.value

    def _mem_set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.config.max_size:
            self._evict_oldest()
        self._store[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.config.ttl_seconds,
        )

    def _mem_clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        self._hits = 0
        self._misses = 0
        return count

    def _evict_oldest(self) -> None:
        """Remove the entry with the earliest expiry (LRU approximation)."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
        del self._store[oldest_key]

    # ------------------------------------------------------------------ #
    # File-based backend (Phase 4)
    # ------------------------------------------------------------------ #

    def _file_get(self, key: str) -> Any | None:
        path = _FILE_CACHE_DIR / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Check TTL: stored as absolute unix timestamp
            if time.time() > data.get("expires_at", 0):
                path.unlink(missing_ok=True)
                self._misses += 1
                return None
            self._hits += 1
            return data.get("value")
        except Exception:
            self._misses += 1
            return None

    def _file_set(self, key: str, value: Any) -> None:
        _FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _FILE_CACHE_DIR / f"{key}.json"
        try:
            path.write_text(
                json.dumps(
                    {"value": value, "expires_at": time.time() + self.config.ttl_seconds},
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _file_clear(self) -> int:
        if not _FILE_CACHE_DIR.exists():
            return 0
        count = 0
        for p in _FILE_CACHE_DIR.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        self._hits = 0
        self._misses = 0
        return count

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _is_excluded(self, full_name: str) -> bool:
        return any(fnmatch.fnmatch(full_name, p) for p in self.config.exclude_tools)


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Sort keys and canonicalize types for deterministic cache key hashing."""
    result = {}
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, bool):
            result[k] = v
        elif isinstance(v, float) and v == int(v):
            result[k] = int(v)  # 1.0 → 1
        elif isinstance(v, dict):
            result[k] = _normalize_args(v)
        else:
            result[k] = v
    return result


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    """Short fingerprint of a tool's input schema — changes when schema changes."""
    raw = json.dumps(schema, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
