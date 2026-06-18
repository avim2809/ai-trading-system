"""Unified LLM service wrapping LiteLLM for multi-provider chat completions."""

from __future__ import annotations

import json
from typing import Any

from firm.llm.cache import ResponseCache


class LLMService:
    """Thin wrapper around LiteLLM providing caching, usage tracking, and JSON mode."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.default_model: str = config.get("default_model", "gpt-4o")
        self.temperature: float = config.get("temperature", 0.3)
        self.max_tokens: int = config.get("max_tokens", 2000)

        cache_enabled = config.get("cache_enabled", True)
        cache_db = config.get("cache_db", "data/llm_cache.db")
        self._cache: ResponseCache | None = ResponseCache(cache_db) if cache_enabled else None

        self._total_tokens = 0
        self._total_cost = 0.0
        self._cache_hits = 0
        self._cache_misses = 0

        try:
            import litellm
            litellm.set_verbose = False
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm is required for LLMService. "
                "Install with: pip install 'firm[llm]'"
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request, using cache if available."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature

        if self._cache is not None:
            cache_key = ResponseCache._hash(
                model, messages,
                temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        cost = response._hidden_params.get("response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0

        self._total_tokens += tokens_in + tokens_out
        self._total_cost += cost

        if self._cache is not None:
            self._cache.put(cache_key, content, model, tokens_in, tokens_out, cost)

        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat with json_mode enabled, returning parsed JSON dict."""
        raw = self.chat(messages, model=model, json_mode=True, **kwargs)

        def _as_dict(parsed: Any) -> dict[str, Any]:
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("expected a JSON object", raw, 0)
            return parsed

        try:
            return _as_dict(json.loads(raw))
        except json.JSONDecodeError:
            # Best-effort recovery of an embedded object; never return a
            # partial/non-object value silently – raise so callers fall back.
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return _as_dict(json.loads(raw[start:end]))
            raise

    @property
    def usage_stats(self) -> dict[str, Any]:
        return {
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
