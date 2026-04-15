"""Tool filter layer — controls which tools the LLM sees per run.

Implements a 3-stage relevance pipeline:
    Stage 1: Keyword overlap (always runs — fast, zero dependencies)
    Stage 2: TF-IDF scoring (strategy="hybrid", requires scikit-learn — optional)
    Stage 3: Embedding similarity (strategy="embedding", requires Ollama embed model)
"""
from __future__ import annotations

import fnmatch
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .transports.base import ToolInfo
    from .config import ToolFilterConfig


def apply_filter(
    tools: list["ToolInfo"],
    prompt: str,
    config: "ToolFilterConfig",
    whitelist: list[str] | None = None,
    ollama: Any = None,   # OllamaAdapter — only needed for embedding strategy
) -> list["ToolInfo"]:
    """Apply the tool filter pipeline and return tools the LLM should see.

    Pipeline:
        1. Apply exclusion patterns (always)
        2. Apply explicit whitelist (if provided — overrides relevance pipeline)
        3. Apply relevance scoring (keyword / hybrid / embedding) based on config.strategy
        4. Cap at config.top_n
    """
    # Step 1: exclusion
    tools = _apply_exclude(tools, config.exclude)

    # Step 2: explicit whitelist overrides everything else
    if whitelist:
        return _apply_whitelist(tools, whitelist)

    # Step 3: strategy
    strategy = config.strategy
    if strategy == "all":
        return tools

    if strategy == "keyword":
        return _keyword_filter(tools, prompt, config.top_n)

    if strategy == "hybrid":
        # Keyword first, then TF-IDF re-ranking
        candidates = _keyword_filter(tools, prompt, config.top_n * 2)
        return _tfidf_filter(candidates, prompt, config.top_n)

    if strategy == "embedding":
        # Embedding similarity — requires ollama adapter (async, so this is a sync stub)
        # In practice the loop calls _embedding_filter_sync which blocks or uses cached vectors
        candidates = _keyword_filter(tools, prompt, config.top_n * 3)
        return _embedding_filter_sync(candidates, prompt, config.top_n)

    return tools[:config.top_n]


# ------------------------------------------------------------------ #
# Exclusion
# ------------------------------------------------------------------ #

def _apply_exclude(tools: list["ToolInfo"], patterns: list[str]) -> list["ToolInfo"]:
    if not patterns:
        return tools
    return [t for t in tools if not _matches_any(t.full_name, patterns)]


# ------------------------------------------------------------------ #
# Explicit whitelist (glob patterns)
# ------------------------------------------------------------------ #

def _apply_whitelist(tools: list["ToolInfo"], patterns: list[str]) -> list["ToolInfo"]:
    """Keep only tools matching at least one whitelist glob pattern."""
    return [t for t in tools if _matches_any(t.full_name, patterns)]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


# ------------------------------------------------------------------ #
# Stage 1: Keyword relevance scoring
# ------------------------------------------------------------------ #

def _keyword_filter(
    tools: list["ToolInfo"],
    prompt: str,
    top_n: int,
) -> list["ToolInfo"]:
    """Score tools by keyword overlap with the prompt. Return top_n."""
    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens:
        return tools[:top_n]

    scored: list[tuple[float, "ToolInfo"]] = []
    for tool in tools:
        score = _score_tool_keyword(tool, prompt_tokens)
        scored.append((score, tool))

    # Sort descending by score; preserve original order for ties
    scored.sort(key=lambda x: x[0], reverse=True)

    # If any tool has a non-zero score, take top_n of those
    # If all are zero (no overlap at all), fall back to all tools (LLM decides)
    positive = [t for s, t in scored if s > 0]
    if positive:
        return positive[:top_n]
    return [t for _, t in scored][:top_n]


def _score_tool_keyword(tool: "ToolInfo", prompt_tokens: set[str]) -> float:
    """Simple keyword overlap score between tool text and prompt tokens."""
    tool_text = f"{tool.full_name} {tool.description}".lower()
    name_parts = set(re.split(r"[._\-]", tool.full_name.lower()))
    tool_tokens = _tokenize(tool_text) | name_parts
    overlap = len(prompt_tokens & tool_tokens)
    return overlap / max(len(prompt_tokens), 1)


# ------------------------------------------------------------------ #
# Stage 2: TF-IDF re-ranking (Phase 4)
# ------------------------------------------------------------------ #

def _tfidf_filter(
    tools: list["ToolInfo"],
    prompt: str,
    top_n: int,
) -> list["ToolInfo"]:
    """Re-rank tools by TF-IDF similarity to prompt.

    Uses scikit-learn if available, falls back to keyword scoring.
    """
    if not tools:
        return tools

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        # scikit-learn not installed — fall back to raw keyword results
        return tools[:top_n]

    # Build corpus: [prompt] + [tool text for each tool]
    tool_texts = [
        f"{t.full_name} {t.description or ''}".lower()
        for t in tools
    ]
    corpus = [prompt.lower()] + tool_texts

    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        prompt_vec = tfidf_matrix[0]
        tool_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(prompt_vec, tool_vecs).flatten()

        # Sort by similarity descending
        ranked = sorted(zip(sims, tools), key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked][:top_n]
    except Exception:
        # Any sklearn error → return as-is
        return tools[:top_n]


# ------------------------------------------------------------------ #
# Stage 3: Embedding similarity (sync stub — async version in loop.py)
# ------------------------------------------------------------------ #

def _embedding_filter_sync(
    tools: list["ToolInfo"],
    prompt: str,
    top_n: int,
) -> list["ToolInfo"]:
    """Embedding-based filtering — sync stub using keyword fallback.

    Full async embedding is handled in the loop via ollama.embed().
    This sync fallback is used when no ollama adapter is passed.
    """
    return _tfidf_filter(tools, prompt, top_n)


async def apply_filter_async(
    tools: list["ToolInfo"],
    prompt: str,
    config: "ToolFilterConfig",
    whitelist: list[str] | None = None,
    ollama: Any = None,
) -> list["ToolInfo"]:
    """Async version of apply_filter — supports embedding strategy with real Ollama calls."""
    # Step 1: exclusion
    tools = _apply_exclude(tools, config.exclude)

    # Step 2: explicit whitelist
    if whitelist:
        return _apply_whitelist(tools, whitelist)

    strategy = config.strategy
    if strategy == "all":
        return tools

    if strategy == "keyword":
        return _keyword_filter(tools, prompt, config.top_n)

    if strategy == "hybrid":
        candidates = _keyword_filter(tools, prompt, config.top_n * 2)
        return _tfidf_filter(candidates, prompt, config.top_n)

    if strategy == "embedding" and ollama is not None:
        candidates = _keyword_filter(tools, prompt, config.top_n * 3)
        return await _embedding_filter_async(candidates, prompt, config, ollama)

    # Default: hybrid
    candidates = _keyword_filter(tools, prompt, config.top_n * 2)
    return _tfidf_filter(candidates, prompt, config.top_n)


async def _embedding_filter_async(
    tools: list["ToolInfo"],
    prompt: str,
    config: "ToolFilterConfig",
    ollama: Any,
) -> list["ToolInfo"]:
    """Compute cosine similarity between prompt and tool descriptions using Ollama embeddings."""
    if not tools:
        return tools

    try:
        tool_texts = [
            f"{t.full_name} {t.description or ''}".strip()
            for t in tools
        ]
        all_texts = [prompt] + tool_texts
        embeddings = await ollama.embed(all_texts, model=config.embedding_model)
        if len(embeddings) < len(all_texts):
            return tools[:config.top_n]

        prompt_vec = embeddings[0]
        tool_vecs = embeddings[1:]
        sims = [_cosine(prompt_vec, tv) for tv in tool_vecs]

        ranked = sorted(zip(sims, tools), key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked][:config.top_n]
    except Exception:
        # Embedding failed — fall back to TF-IDF
        return _tfidf_filter(tools, prompt, config.top_n)


def _cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ------------------------------------------------------------------ #
# Tokenization
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, filtering stop words and very short tokens."""
    _STOP = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "not", "with", "my", "me", "i", "do", "can",
        "please", "use", "get", "from", "into", "this", "that", "what",
        "how", "why", "all", "any", "be", "by", "as", "so", "if",
    }
    tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower()))
    return tokens - _STOP
