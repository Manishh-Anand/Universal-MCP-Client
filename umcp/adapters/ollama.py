"""Ollama API adapter — drives the local LLM for tool-calling."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from ..transports.base import ToolInfo

# Models known to support native tool calling in Ollama
_KNOWN_TOOL_CAPABLE: set[str] = {
    "qwen2.5", "qwen2", "qwen3",
    "llama3.1", "llama3.2", "llama3.3",
    "mistral-nemo", "mistral-large", "mistral-small",
    "command-r", "command-r-plus",
    "deepseek-r1", "deepseek-v3",
    "phi4", "phi3.5",
    "granite3", "granite3.1",
    "firefunction-v2", "nemotron-mini",
    "hermes3",
}

# Models known NOT to support native tool calling
_KNOWN_NO_TOOLS: set[str] = {
    "codellama", "llama2", "gemma", "gemma2",
    "phi", "phi3", "stablelm", "orca-mini",
    "neural-chat", "vicuna", "wizardlm",
}

_CAPABILITY_CACHE_PATH = Path.home() / ".config" / "umcp" / "model_capabilities.json"


class OllamaAdapter:
    """Wraps Ollama's /api/chat endpoint with tool-calling + fallback support."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)
        self._supports_tools: bool | None = None
        self._capability_cache: dict[str, bool] = _load_capability_cache()

    # ------------------------------------------------------------------ #
    # Capability detection
    # ------------------------------------------------------------------ #

    def _model_family(self) -> str:
        """'qwen2.5:7b' → 'qwen2.5'"""
        return self.model.split(":")[0].lower()

    async def check_capability(self) -> bool:
        """Return True if model supports native structured tool calling."""
        if self._supports_tools is not None:
            return self._supports_tools

        if self.model in self._capability_cache:
            self._supports_tools = self._capability_cache[self.model]
            return self._supports_tools

        family = self._model_family()
        if family in _KNOWN_TOOL_CAPABLE:
            result = True
        elif family in _KNOWN_NO_TOOLS:
            result = False
        else:
            result = await self._probe()

        self._supports_tools = result
        self._capability_cache[self.model] = result
        _save_capability_cache(self._capability_cache)
        return result

    async def _probe(self) -> bool:
        """Send a lightweight probe to detect tool-call support."""
        probe_tools = [{
            "type": "function",
            "function": {
                "name": "probe",
                "description": "Probe function — call it",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }]
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Call the probe function now."}],
                    "tools": probe_tools,
                    "stream": False,
                },
                timeout=30.0,
            )
            data = resp.json()
            return bool(data.get("message", {}).get("tool_calls"))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Chat (non-streaming)
    # ------------------------------------------------------------------ #

    async def chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolInfo],
    ) -> dict[str, Any]:
        """Send a single chat request and return the full Ollama response dict."""
        supports = await self.check_capability()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if supports and tools:
            payload["tools"] = _format_tools(tools)

        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Chat (streaming)
    # ------------------------------------------------------------------ #

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolInfo],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw Ollama chunk dicts as they stream.

        Each chunk has shape: {"message": {"content": "..."}, "done": bool, ...}
        The caller is responsible for printing content tokens and detecting done/tool_calls.
        """
        supports = await self.check_capability()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if supports and tools:
            payload["tools"] = _format_tools(tools)

        async with self._client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    yield chunk
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------ #
    # Embeddings (Phase 4 — embedding-based tool filtering)
    # ------------------------------------------------------------------ #

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts via Ollama /api/embed.

        Args:
            texts: List of strings to embed.
            model: Embedding model to use (defaults to nomic-embed-text).

        Returns:
            List of embedding vectors (one per input text).
        """
        embed_model = model or "nomic-embed-text"
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": embed_model, "input": texts},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Model listing
    # ------------------------------------------------------------------ #

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the list of locally available Ollama models."""
        resp = await self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def model_capability_summary(self) -> list[dict[str, Any]]:
        """Return models with their tool-calling capability flag."""
        models = await self.list_models()
        rows = []
        for m in models:
            name = m.get("name", "")
            family = name.split(":")[0].lower()
            if name in self._capability_cache:
                capable: str = "yes" if self._capability_cache[name] else "no"
            elif family in _KNOWN_TOOL_CAPABLE:
                capable = "yes"
            elif family in _KNOWN_NO_TOOLS:
                capable = "no"
            else:
                capable = "unknown"
            rows.append({"name": name, "tool_calling": capable, "size": m.get("size", 0)})
        return rows

    async def close(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _format_tools(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    """Convert ToolInfo list → Ollama tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.full_name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _load_capability_cache() -> dict[str, bool]:
    if _CAPABILITY_CACHE_PATH.exists():
        try:
            return json.loads(_CAPABILITY_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_capability_cache(cache: dict[str, bool]) -> None:
    try:
        _CAPABILITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CAPABILITY_CACHE_PATH.write_text(
            json.dumps(cache, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
