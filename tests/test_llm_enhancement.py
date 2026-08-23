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
        self.last_kwargs: dict = {}

    def chat_json(self, messages, **kw):
        self.calls += 1
        self.last_kwargs = kw
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

    def test_configured_temperature_reaches_the_llm_call(self, monkeypatch):
        """Regression: enhancement.temperature must reach chat_json/get_cached
        as an explicit per-call override -- these scoring calls feed straight
        into the z-scored analyst signal, so sampling noise from the global
        provider.temperature (0.3) is a real, avoidable source of day-to-day
        signal wobble independent of any genuine information change."""
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls",
                "min_abs_score": 0.0,
                "max_signals_per_agent": 8,
                "rag_n_results": 2,
                "temperature": 0.1,
            },
        )
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = [_sig("AAPL", "news", 0.8)]
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockLLMService()
        agent._retrieve_context = lambda *a, **k: "ctx"

        agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        assert agent._llm.calls == 1
        assert agent._llm.last_kwargs.get("temperature") == 0.1

    def test_unset_temperature_does_not_override_llm_service_default(self, monkeypatch):
        """Backward compatibility: temperature absent/None (the default) must
        not pass an explicit override at all, preserving byte-identical
        behavior for every existing caller."""
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
        strat.generate.return_value = [_sig("AAPL", "news", 0.8)]
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = MockLLMService()
        agent._retrieve_context = lambda *a, **k: "ctx"

        agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        assert "temperature" not in agent._llm.last_kwargs


class _OutOfBoundsLLMService:
    """Mock LLM that ignores the documented [-1,1]/[0,1] prompt ranges."""

    def __init__(self, score=5.0, confidence=-0.3, score_sequence=None):
        self.usage_stats = {}
        self._score = score
        self._confidence = confidence
        self._score_sequence = list(score_sequence) if score_sequence else None
        self._calls = 0

    def chat_json(self, messages, **kw):
        score = self._score
        if self._score_sequence:
            score = self._score_sequence[self._calls % len(self._score_sequence)]
            self._calls += 1
        return {"score": score, "confidence": self._confidence, "rationale": "mock"}

    def get_cached(self, messages, **kw):
        return None


class TestBoundsClampingAndReZScore:
    def _agent(self, monkeypatch, signals, llm):
        monkeypatch.setattr(
            "firm.llm.config.enhancement_config",
            lambda overrides=None: {
                "policy": "live_calls", "min_abs_score": 0.0,
                "max_signals_per_agent": 0, "rag_n_results": 2,
            },
        )
        strat = MagicMock()
        strat.name = "news"
        strat.generate.return_value = signals
        agent = LLMSentimentAnalyst(strategies=[strat], llm_config={})
        agent._llm = llm
        agent._retrieve_context = lambda *a, **k: "ctx"
        return agent

    def test_out_of_bounds_llm_score_and_confidence_are_clamped(self, monkeypatch):
        signals = [_sig("AAPL", "news", 0.9), _sig("MSFT", "news", -0.9)]
        agent = self._agent(monkeypatch, signals, _OutOfBoundsLLMService(score=5.0, confidence=-0.3))

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        # After clamping (score -> 1.0, confidence -> 0.0) and re-z-scoring,
        # no signal should retain the raw hallucinated magnitude/sign.
        for sig in result.signals:
            assert -1e6 < sig.score < 1e6  # finite, not blown up
            assert sig.meta.get("llm_enhanced") is True

    def test_nan_and_non_numeric_llm_values_fall_back_to_quant(self, monkeypatch):
        signals = [_sig("AAPL", "news", 0.4), _sig("MSFT", "news", -0.4)]
        agent = self._agent(monkeypatch, signals, _OutOfBoundsLLMService(score=float("nan"), confidence="not-a-number"))

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        for sig in result.signals:
            assert sig.score == sig.score  # not NaN
            assert 0.0 <= sig.confidence <= 1.0

    def test_re_zscored_group_has_zero_mean_after_llm_override(self, monkeypatch):
        signals = [_sig("AAPL", "news", 0.9), _sig("MSFT", "news", -0.9), _sig("GOOG", "news", 0.1)]
        llm = _OutOfBoundsLLMService(confidence=1.5, score_sequence=[5.0, -5.0, 0.3])
        agent = self._agent(monkeypatch, signals, llm)

        result = agent.run(AgentContext(now=NOW, pit_view=MagicMock()))
        scores = [s.score for s in result.signals]
        # Distinct clamped scores (1.0, -1.0, 0.3) give a non-degenerate std,
        # so zscore_signals actually re-normalises rather than passing through.
        assert sum(scores) / len(scores) == pytest.approx(0.0, abs=1e-9)


class TestLLMOverrideBoundsHelper:
    """Direct unit coverage of the shared clamp helper on LLMAgentMixin."""

    def _mixin(self):
        agent = LLMSentimentAnalyst(strategies=[], llm_config={})
        return agent

    def test_score_and_confidence_clamped_to_documented_range(self):
        agent = self._mixin()
        score, confidence = agent._bounded_override("AAPL", "news", 5.0, -0.3, 0.1, 0.5)
        assert score == 1.0
        assert confidence == 0.0

        score, confidence = agent._bounded_override("AAPL", "news", -5.0, 2.0, 0.1, 0.5)
        assert score == -1.0
        assert confidence == 1.0

    def test_in_range_values_pass_through_unchanged(self):
        agent = self._mixin()
        score, confidence = agent._bounded_override("AAPL", "news", 0.42, 0.73, 0.1, 0.5)
        assert score == pytest.approx(0.42)
        assert confidence == pytest.approx(0.73)

    def test_non_numeric_and_nan_fall_back_to_quant_values(self):
        agent = self._mixin()
        score, confidence = agent._bounded_override("AAPL", "news", "oops", None, 0.1, 0.5)
        assert score == 0.1
        assert confidence == 0.5

        score, confidence = agent._bounded_override("AAPL", "news", float("nan"), float("nan"), 0.1, 0.5)
        assert score == 0.1
        assert confidence == 0.5

    def test_conviction_clamped(self):
        agent = self._mixin()
        assert agent._bounded_conviction("AAPL", "bull", 1.5, 0.5) == 1.0
        assert agent._bounded_conviction("AAPL", "bull", -0.5, 0.5) == 0.0
        assert agent._bounded_conviction("AAPL", "bull", 0.65, 0.5) == pytest.approx(0.65)
        assert agent._bounded_conviction("AAPL", "bull", "bad", 0.5) == 0.5

    def test_net_conviction_clamped(self):
        agent = self._mixin()
        assert agent._bounded_net_conviction("AAPL", 3.0, 0.0) == 1.0
        assert agent._bounded_net_conviction("AAPL", -3.0, 0.0) == -1.0
        assert agent._bounded_net_conviction("AAPL", 0.42, 0.0) == pytest.approx(0.42)
        assert agent._bounded_net_conviction("AAPL", float("nan"), 0.25) == 0.25
