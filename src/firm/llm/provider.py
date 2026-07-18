"""Unified LLM service wrapping LiteLLM for multi-provider chat completions."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from firm.llm.cache import ResponseCache


class LLMService:
    """Thin wrapper around LiteLLM providing caching, usage tracking, and JSON mode."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # Free-by-default: Groq's free tier. Override in config/llm.yaml.
        self.default_model: str = config.get(
            "default_model", "groq/llama-3.3-70b-versatile"
        )
        self.temperature: float = config.get("temperature", 0.3)
        self.max_tokens: int = config.get("max_tokens", 2000)

        # Model pool for load-balancing + fallback: default_model first, then
        # any configured fallback_models, de-duplicated. A single-entry pool
        # (the common case) makes _select_model()/chat() behave exactly as
        # before this was added.
        self.fallback_models: list[str] = list(config.get("fallback_models", []))
        self.load_balance: bool = bool(config.get("load_balance", False))
        self._model_pool: list[str] = []
        for m in (self.default_model, *self.fallback_models):
            if m and m not in self._model_pool:
                self._model_pool.append(m)
        self._rr_index = 0
        self._rr_lock = threading.Lock()

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
        prompt_cache: bool = False,
    ) -> str:
        """Send a chat completion request, using cache if available.

        When *prompt_cache* is set and the active model is an Anthropic Claude
        model, the first system message is marked with ``cache_control`` so its
        (large, static) content is served from Anthropic's prompt cache on
        repeat calls — ~90% cheaper on the cached prefix. No-op for providers
        that don't honour explicit cache breakpoints.
        """
        model = model or self._select_model(messages)
        temperature = temperature if temperature is not None else self.temperature

        # Response-cache key is computed from the original messages so it stays
        # stable regardless of any prompt-cache wrapping applied below.
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

        send_messages = self._apply_prompt_cache(messages, model) if prompt_cache else messages

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": send_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Any other pool member can stand in if *model* errors (rate limit,
        # timeout, provider outage, ...) — LiteLLM tries them in order and
        # only raises once every one of them has failed.
        fallback_chain = [m for m in self._model_pool if m != model]
        if fallback_chain:
            kwargs["fallbacks"] = fallback_chain

        response = self._litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""

        # The model that actually served the response may differ from
        # *model* if a fallback kicked in — record that one for cost/cache
        # bookkeeping so usage stats reflect what really ran.
        served_model = getattr(response, "model", None) or model

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        cost = response._hidden_params.get("response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0

        self._total_tokens += tokens_in + tokens_out
        self._total_cost += cost

        if self._cache is not None:
            self._cache.put(cache_key, content, served_model, tokens_in, tokens_out, cost)

        return content

    def _select_model(self, messages: list[dict[str, Any]] | None = None) -> str:
        """Pick the model for a call with no explicit ``model=`` override.

        Spreads requests across ``[default_model, *fallback_models]`` when
        ``load_balance`` is on, so no single free-tier cap gets hammered.
        Routing is a deterministic hash of *messages*, not a rotating
        counter: the same prompt always lands on the same model, so
        identical repeated calls still hit the response cache (whose key
        includes the model) — a round-robin counter would bounce identical
        prompts across different models on every call and defeat caching
        entirely. Distinct prompts still hash to different models across
        the pool, so load still spreads across the wider corpus of calls.
        """
        if not self.load_balance or len(self._model_pool) <= 1:
            return self.default_model
        if not messages:
            # Nothing to hash against (e.g. called speculatively) — fall
            # back to a rotating counter so load still spreads.
            with self._rr_lock:
                picked = self._model_pool[self._rr_index % len(self._model_pool)]
                self._rr_index += 1
            return picked
        digest = hashlib.sha256(
            json.dumps(messages, sort_keys=True, default=str).encode()
        ).hexdigest()
        return self._model_pool[int(digest, 16) % len(self._model_pool)]

    @staticmethod
    def _apply_prompt_cache(
        messages: list[dict[str, Any]], model: str
    ) -> list[dict[str, Any]]:
        """Mark the first system message with Anthropic ``cache_control``.

        Only applies to Anthropic Claude models (the providers that honour
        explicit cache breakpoints); returns *messages* unchanged otherwise.
        """
        from firm.llm.config import is_anthropic

        if not is_anthropic(model):
            return messages

        out: list[dict[str, Any]] = []
        cached = False
        for m in messages:
            if (
                not cached
                and m.get("role") == "system"
                and isinstance(m.get("content"), str)
            ):
                out.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": m["content"],
                        "cache_control": {"type": "ephemeral"},
                    }],
                })
                cached = True
            else:
                out.append(m)
        return out

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
