"""Cost-gating for LLM-enhanced agents."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from firm.agents.base import AgentContext
from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst
from firm.contracts.models import Signal

NOW = datetime(2024, 1, 15, 16, 0)


def _sig(symbol: str, strategy: str, score: float) -> Signal:
    return Signal(
        symbol=symbol, strategy=strategy, score=score,
        confidence=0.8, horizon="5d", asof=NOW,
    )


class MockLLMService:
    def __init__(self, config=None):
        self.usage_stats = {}
        self.calls = 0

    def chat_json(self, messages, **kw):
        self.calls += 1
        return {"score": 0.9, "confidence": 0.9, "rationale": "mock"}

    def get_cached(self, messages, **kw):
        return None


class TestEnhancementGating:
    def test_weak_signals_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls",
                "min_abs_score": 0.5,
                "max_signals_per_agent": 8,
                "rag_n_results": 2,
            },
        )
        agent = LLMSentimentAnalyst(
            strategies=[MagicMock(name="news", generate=MagicMock(return_value=[_sig("AAPL", "news", 0.1)]))],
            llm_config={},
        )
        agent.strategies[0].name = "news"
        agent._llm = MockLLMService()
        agent._retrieve_context = lambda *a, **k: "news context"

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        assert agent._llm.calls == 0
        assert result.signals[0].meta.get("llm_enhanced") is None

    def test_max_signals_cap(self, monkeypatch):
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls",
                "min_abs_score": 0.0,
                "max_signals_per_agent": 2,
                "rag_n_results": 2,
            },
        )
        signals = [_sig("A", "news", 0.9), _sig("B", "news", 0.8), _sig("C", "news", 0.7)]
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = signals
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockLLMService()
        agent._retrieve_context = lambda *a, **k: "ctx"

        agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        assert agent._llm.calls == 2

    def test_cache_only_skips_miss(self, monkeypatch):
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "cache_only",
                "min_abs_score": 0.0,
                "max_signals_per_agent": 8,
                "rag_n_results": 2,
            },
        )
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = [_sig("AAPL", "news", 0.8)]
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockLLMService()
        agent._retrieve_context = lambda *a, **k: "ctx"

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        assert agent._llm.calls == 0
        assert result.signals[0].score == pytest.approx(0.8)
