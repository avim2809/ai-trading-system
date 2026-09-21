"""Unit tests for firm.llm.config.llm_service_config.

LLMService reads cache_enabled/cache_db from the "optimization" section of
the LLM config YAML, but every LLMService construction site on the live
path (firm.runtime.build_orchestrator, firm.live.engine.LiveTradingEngine.
_get_llm_service, firm.api.routers.agents) was passing bare
provider_config() as the whole config dict, which never includes
"optimization" at all -- so cache_enabled/cache_db were silently absent
rather than merely defaulted. llm_service_config() merges the two sections
so the values LLMService actually reads (default_model, fallback_models,
load_balance, temperature, max_tokens, request_timeout, cache_enabled,
cache_db) all survive.
"""

from __future__ import annotations

from firm.llm.config import llm_service_config


class TestLLMServiceConfig:
    def test_merges_optimization_cache_keys_into_provider_config(self):
        cfg = {
            "provider": {"default_model": "groq/openai/gpt-oss-120b"},
            "optimization": {"cache_enabled": True, "cache_db": "data/llm_cache_ab_llm.db"},
        }
        merged = llm_service_config(cfg)
        assert merged["default_model"] == "groq/openai/gpt-oss-120b"
        assert merged["cache_enabled"] is True
        assert merged["cache_db"] == "data/llm_cache_ab_llm.db"

    def test_missing_optimization_section_falls_back_to_defaults(self):
        cfg = {"provider": {"default_model": "groq/openai/gpt-oss-120b"}}
        merged = llm_service_config(cfg)
        # optimization_config()'s own defaults still apply -- not "missing".
        assert merged["cache_enabled"] is True
        assert merged["cache_db"] == "data/llm_cache.db"

    def test_provider_fields_not_clobbered_by_optimization_section(self):
        cfg = {
            "provider": {
                "default_model": "groq/openai/gpt-oss-120b",
                "fallback_models": ["gemini/gemini-3.7-flash"],
                "load_balance": True,
            },
            "optimization": {"cache_enabled": False, "cache_db": "data/other_cache.db"},
        }
        merged = llm_service_config(cfg)
        assert merged["fallback_models"] == ["gemini/gemini-3.7-flash"]
        assert merged["load_balance"] is True
        assert merged["cache_enabled"] is False
        assert merged["cache_db"] == "data/other_cache.db"

    def test_does_not_leak_unrelated_optimization_keys(self):
        cfg = {
            "provider": {},
            "optimization": {"compression_enabled": False, "compression_ratio": 0.9},
        }
        merged = llm_service_config(cfg)
        # Only cache_enabled/cache_db are LLMService's concern -- compression
        # knobs belong to LLMAgentMixin._compress via optimization_config()
        # directly and must not end up passed into LLMService's own kwargs.
        assert "compression_enabled" not in merged
        assert "compression_ratio" not in merged

    def test_none_cfg_loads_from_disk_without_raising(self):
        # Smoke test only: no assertions on content since this reads
        # whatever config/llm.yaml (or FIRM_LLM_CONFIG) resolves to in the
        # test environment -- just confirms the no-arg path is wired.
        merged = llm_service_config()
        assert "cache_enabled" in merged
        assert "cache_db" in merged
