"""Self-consistency sampling in LLMAgentMixin._call_llm (see
firm.live.planning_cycle and Orchestrator._apply_cycle_llm_mode).

Ships off by default (enhancement.self_consistency_samples == 1, a no-op)
-- these tests exercise the >1 path directly, since no production config
enables it yet.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import _aggregate_llm_samples
from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst
from firm.contracts.models import Signal

NOW = datetime(2024, 1, 15, 16, 0)


def _sig(symbol: str, strategy: str, score: float) -> Signal:
    return Signal(symbol=symbol, strategy=strategy, score=score, confidence=0.8, horizon="5d", asof=NOW)


class TestAggregateLlmSamples:
    def test_single_sample_passthrough(self):
        assert _aggregate_llm_samples([{"score": 0.7}]) == {"score": 0.7}

    def test_numeric_fields_average(self):
        samples = [{"score": 0.6}, {"score": 0.8}, {"score": 1.0}]
        assert _aggregate_llm_samples(samples)["score"] == pytest.approx(0.8)

    def test_string_fields_majority_vote(self):
        samples = [
            {"stance": "bullish"}, {"stance": "bullish"}, {"stance": "bearish"},
        ]
        assert _aggregate_llm_samples(samples) == {"stance": "bullish"}

    def test_mixed_numeric_and_string_fields(self):
        samples = [
            {"score": 0.4, "stance": "bullish"},
            {"score": 0.6, "stance": "bullish"},
            {"score": 0.8, "stance": "neutral"},
        ]
        result = _aggregate_llm_samples(samples)
        assert result["score"] == 0.6
        assert result["stance"] == "bullish"

    def test_plain_text_samples_return_first_not_fabricated_aggregate(self):
        samples = ["first thesis text", "second thesis text"]
        assert _aggregate_llm_samples(samples) == "first thesis text"

    def test_missing_key_in_some_samples_only_averages_present_values(self):
        samples = [{"score": 0.5, "extra": 1.0}, {"score": 0.7}]
        result = _aggregate_llm_samples(samples)
        assert result["score"] == 0.6
        assert result["extra"] == 1.0


class MockVaryingLLMService:
    """Returns a different score each call, in sequence -- proves
    aggregation actually combines multiple distinct samples rather than
    just re-returning one cached value N times."""

    def __init__(self, scores: list[float]):
        self.usage_stats = {}
        self.calls = 0
        self._scores = scores

    def chat_json(self, messages, **kw):
        score = self._scores[self.calls % len(self._scores)]
        self.calls += 1
        return {"score": score, "confidence": 0.9, "rationale": "mock"}

    def get_cached(self, messages, **kw):
        return None


class TestCallLlmSelfConsistencyIntegration:
    def test_samples_n_times_and_averages(self, monkeypatch):
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls",
                "min_abs_score": 0.0,
                "max_signals_per_agent": 8,
                "rag_n_results": 2,
                "self_consistency_samples": 3,
            },
        )
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = [_sig("AAPL", "news", 0.5)]
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockVaryingLLMService(scores=[0.6, 0.8, 1.0])
        agent._retrieve_context = lambda *a, **k: "ctx"

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))

        assert agent._llm.calls == 3
        assert result.signals[0].meta["llm_enhanced"] is True
        # Aggregated score is the average of the 3 sampled scores (0.6/0.8/1.0),
        # not just whichever call happened to run last.
        assert result.signals[0].score == pytest.approx(0.8)

    def test_default_of_one_makes_exactly_one_call(self, monkeypatch):
        """self_consistency_samples defaults to 1 -- must behave exactly
        like today, one call, no aggregation path taken at all."""
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls",
                "min_abs_score": 0.0,
                "max_signals_per_agent": 8,
                "rag_n_results": 2,
            },
        )
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = [_sig("AAPL", "news", 0.5)]
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockVaryingLLMService(scores=[0.6, 0.8, 1.0])
        agent._retrieve_context = lambda *a, **k: "ctx"

        agent.run(AgentContext(now=NOW, pit_view=MagicMock()))

        assert agent._llm.calls == 1
