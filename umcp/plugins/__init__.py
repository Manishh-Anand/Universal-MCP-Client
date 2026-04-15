"""Plugin hook registry — allows users to customize umcp behaviour without forking.

Phase 4 plugin hooks:

    @client.plugin("system_prompt")
    def my_prompt(base: str) -> str:
        return base + "\\nAlways respond in Japanese."

    @client.plugin("tool_filter")
    def my_filter(tools, prompt):
        return [t for t in tools if "dangerous" not in t.name]

Available hooks:
    system_prompt(base: str) -> str
        Called with the base system prompt. Return the modified prompt.

    tool_filter(tools: list[ToolInfo], prompt: str) -> list[ToolInfo]
        Called after built-in filtering. Return the filtered tool list.

    logging(entry: TraceEntry) -> None
        Called after each tool call. Use for custom observability sinks.
"""
from __future__ import annotations

from typing import Any, Callable


class PluginRegistry:
    """Stores and dispatches plugin hooks by name."""

    SUPPORTED_HOOKS = {"system_prompt", "tool_filter", "logging"}

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable]] = {}

    def register(self, hook_name: str, fn: Callable) -> None:
        """Register a function for a named hook.

        Multiple functions can be registered for the same hook.
        They are called in registration order.
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(fn)

    def get(self, hook_name: str) -> Callable | None:
        """Return a composed callable for the hook, or None if no hooks registered.

        For hooks that return a value (system_prompt, tool_filter), the output of
        each function is passed as input to the next (pipeline pattern).
        For side-effect hooks (logging), all functions are called in order.
        """
        fns = self._hooks.get(hook_name, [])
        if not fns:
            return None

        if hook_name == "system_prompt":
            def _composed_prompt(base: str) -> str:
                result = base
                for fn in fns:
                    result = fn(result)
                return result
            return _composed_prompt

        if hook_name == "tool_filter":
            def _composed_filter(tools: list, prompt: str) -> list:
                result = tools
                for fn in fns:
                    result = fn(result, prompt)
                return result
            return _composed_filter

        if hook_name == "logging":
            def _composed_logger(entry: Any) -> None:
                for fn in fns:
                    fn(entry)
            return _composed_logger

        # Generic: call all, return last result
        def _generic(*args: Any, **kwargs: Any) -> Any:
            result = None
            for fn in fns:
                result = fn(*args, **kwargs)
            return result
        return _generic

    def has(self, hook_name: str) -> bool:
        """Return True if any handlers are registered for this hook."""
        return bool(self._hooks.get(hook_name))

    def clear(self) -> None:
        """Remove all registered hooks (useful for testing)."""
        self._hooks.clear()
