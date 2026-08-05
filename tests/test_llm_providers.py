"""Provider registry/factory + opt-in Anthropic prompt-cache wrapping."""

from __future__ import annotations

import os

import pytest

from firm.llm.provider import LLMService
from firm.llm.providers import (
    PROVIDERS,
    build_llm_service,
    get_provider,
    list_providers,
    resolve_provider,
)


class TestRegistry:
    def test_common_providers_present(self):
        keys = {p.key for p in list_providers()}
        assert {"groq", "anthropic", "openai", "gemini", "mistral", "ollama"} <= keys

    def test_get_provider_google_alias(self):
        assert get_provider("google").key == "gemini"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError):
            get_provider("not-a-provider")

    def test_resolve_provider_by_prefix(self):
        assert resolve_provider("groq/llama-3.3-70b-versatile").key == "groq"
        assert resolve_provider("gpt-4o").key == "openai"
        assert resolve_provider("anthropic/claude-opus-4-8").key == "anthropic"
        assert resolve_provider("gemini/gemini-2.5-flash").key == "gemini"

    def test_resolve_bare_claude_id(self):
        # No "anthropic/" prefix, still routes to Anthropic.
        assert resolve_provider("claude-opus-4-8").key == "anthropic"

    def test_resolve_bare_gemini_id(self):
        assert resolve_provider("gemini-2.5-flash").key == "gemini"

    def test_resolve_unknown_is_none(self):
        assert resolve_provider("totally-made-up-model") is None

    def test_ollama_needs_no_key(self):
        assert PROVIDERS["ollama"].is_configured() is True


class TestBuildLLMService:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            build_llm_service(provider="anthropic")

    def test_key_present_builds_with_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        svc = build_llm_service(provider="anthropic")
        assert isinstance(svc, LLMService)
        assert svc.default_model == "anthropic/claude-opus-4-8"

    def test_model_override_infers_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        svc = build_llm_service(model="gpt-4o-mini")
        assert svc.default_model == "gpt-4o-mini"

    def test_require_key_false_skips_validation(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        svc = build_llm_service(provider="mistral", require_key=False)
        assert svc.default_model == "mistral/mistral-large-latest"

    def test_gemini_google_api_key_alias(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        svc = build_llm_service(provider="gemini")
        assert svc.default_model == "gemini/gemini-2.5-pro"
        assert os.environ.get("GEMINI_API_KEY") == "google-key"

    def test_request_timeout_passed_to_litellm(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        captured: dict = {}

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                captured.update(kwargs)
                resp = type("R", (), {})()
                resp.choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]
                resp.usage = None
                resp._hidden_params = {}
                return resp

        monkeypatch.setattr("litellm.completion", FakeLitellm.completion)
        svc = LLMService({"default_model": "groq/llama-3.3-70b-versatile", "request_timeout": 42, "cache_enabled": False})
        assert svc.chat([{"role": "user", "content": "hi"}]) == "ok"
        assert captured.get("timeout") == 42


class TestPromptCache:
    def test_anthropic_system_gets_cache_control(self):
        messages = [
            {"role": "system", "content": "big static context"},
            {"role": "user", "content": "hi"},
        ]
        out = LLMService._apply_prompt_cache(messages, "anthropic/claude-opus-4-8")
        sys_block = out[0]["content"][0]
        assert sys_block["cache_control"] == {"type": "ephemeral"}
        assert sys_block["text"] == "big static context"
        assert out[1] == {"role": "user", "content": "hi"}  # untouched

    def test_non_anthropic_unchanged(self):
        messages = [{"role": "system", "content": "ctx"}]
        out = LLMService._apply_prompt_cache(messages, "groq/llama-3.3-70b-versatile")
        assert out == messages

    def test_only_first_system_wrapped(self):
        messages = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
        ]
        out = LLMService._apply_prompt_cache(messages, "claude-opus-4-8")
        assert isinstance(out[0]["content"], list)   # wrapped
        assert out[1]["content"] == "b"              # plain string, untouched


class TestModelRouting:
    """_select_model must route by prompt content, not a rotating counter —
    otherwise identical repeated prompts bounce across different models and
    never hit the response cache (whose key includes the served model)."""

    def _service(self, **overrides):
        config = {
            "default_model": "groq/llama-3.3-70b-versatile",
            "fallback_models": ["openrouter/openai/gpt-oss-20b:free", "groq/llama-3.1-8b-instant"],
            "load_balance": True,
            "cache_enabled": False,
            **overrides,
        }
        return LLMService(config)

    def test_same_prompt_always_routes_to_same_model(self):
        svc = self._service()
        messages = [{"role": "user", "content": "What's the AAPL sentiment?"}]
        picks = {svc._select_model(messages) for _ in range(10)}
        assert len(picks) == 1

    def test_different_prompts_spread_across_the_pool(self):
        svc = self._service()
        picks = {
            svc._select_model([{"role": "user", "content": f"query {i}"}])
            for i in range(30)
        }
        # With 30 distinct prompts hashed over a 3-model pool, expect more
        # than one model to get used (astronomically unlikely otherwise).
        assert len(picks) > 1
        assert picks <= set(svc._model_pool)

    def test_load_balance_off_always_uses_default(self):
        svc = self._service(load_balance=False)
        messages = [{"role": "user", "content": "anything"}]
        assert svc._select_model(messages) == svc.default_model

    def test_single_model_pool_always_uses_default(self):
        svc = self._service(fallback_models=[])
        messages = [{"role": "user", "content": "anything"}]
        assert svc._select_model(messages) == svc.default_model

    def test_no_messages_falls_back_to_rotation(self):
        svc = self._service()
        picks = [svc._select_model(None) for _ in range(len(svc._model_pool) * 2)]
        # Rotates rather than always returning the same model.
        assert len(set(picks)) > 1
