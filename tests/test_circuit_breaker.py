"""Tests for the generic per-strategy rolling-Sharpe circuit breaker.

Covers the standalone damping-factor computation
(:func:`firm.agents.research._circuit_breaker.compute_strategy_damping`), the
signal-damping application, and the end-to-end wiring through
``net_scores_for_blackboard`` (the bull/bear researchers' shared combination
entry point).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.base import AgentContext
from firm.agents.blackboard import Blackboard
from firm.agents.research._circuit_breaker import (
    DEFAULT_CONFIG,
    apply_circuit_breaker,
    compute_strategy_damping,
)
from firm.agents.research._combine import net_scores_for_blackboard
from firm.contracts.models import Signal, SignalSet

NOW = datetime(2026, 1, 2)


def _returns(n: int, mean: float, std: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


class TestComputeStrategyDamping:
    def test_disabled_by_default_is_noop(self):
        history = {"regime_hmm": _returns(100, -0.01, 0.01)}
        assert compute_strategy_damping(history, None) == {}
        assert compute_strategy_damping(history, {"enabled": False}) == {}

    def test_no_history_is_noop(self):
        assert compute_strategy_damping(None, {"enabled": True}) == {}
        assert compute_strategy_damping({}, {"enabled": True}) == {}

    def test_healthy_strategy_undamped(self):
        # Strong positive mean relative to std -> healthy Sharpe, no gate.
        history = {"momentum": _returns(100, 0.01, 0.01)}
        damping = compute_strategy_damping(history, {"enabled": True})
        assert "momentum" not in damping

    def test_persistently_negative_strategy_damped(self):
        # Strong negative mean relative to std -> deeply negative Sharpe.
        history = {"regime_hmm": _returns(100, -0.01, 0.01)}
        damping = compute_strategy_damping(history, {"enabled": True})
        assert "regime_hmm" in damping
        assert DEFAULT_CONFIG["damping_floor"] <= damping["regime_hmm"] < 1.0

    def test_thin_track_record_is_not_judged(self):
        history = {"new_strategy": _returns(5, -0.05, 0.01)}
        damping = compute_strategy_damping(
            history, {"enabled": True, "min_track_record_days": 20}
        )
        assert "new_strategy" not in damping

    def test_only_uses_trailing_lookback_window(self):
        # First half of history is disastrous, second half (trailing window)
        # is healthy -> should NOT be damped once it recovers.
        bad = _returns(100, -0.02, 0.005, seed=1)
        good = _returns(100, 0.02, 0.005, seed=2)
        history = {"recovering": pd.concat([bad, good], ignore_index=True)}
        damping = compute_strategy_damping(
            history, {"enabled": True, "lookback_days": 60, "min_track_record_days": 20}
        )
        assert "recovering" not in damping

    def test_damping_never_below_floor(self):
        history = {"regime_hmm": _returns(200, -0.05, 0.01)}
        damping = compute_strategy_damping(
            history, {"enabled": True, "damping_floor": 0.3}
        )
        assert damping["regime_hmm"] == pytest.approx(0.3)

    def test_invalid_thresholds_disable_gate_safely(self):
        history = {"regime_hmm": _returns(100, -0.05, 0.01)}
        damping = compute_strategy_damping(
            history,
            {"enabled": True, "trigger_sharpe": -1.0, "full_cutoff_sharpe": -0.5},
        )
        assert damping == {}

    def test_zero_variance_series_not_judged(self):
        history = {"flat": pd.Series([0.0] * 50)}
        damping = compute_strategy_damping(history, {"enabled": True})
        assert "flat" not in damping


class TestApplyCircuitBreaker:
    def _signals(self) -> list[Signal]:
        return [
            Signal("AAPL", "regime_hmm", 1.0, 0.8, "5d", NOW),
            Signal("AAPL", "momentum", 1.0, 0.8, "5d", NOW),
        ]

    def test_disabled_returns_same_list(self):
        out = apply_circuit_breaker(self._signals(), None, {"enabled": False})
        assert out == self._signals()

    def test_damps_only_the_gated_strategy(self):
        history = {"regime_hmm": _returns(100, -0.01, 0.01)}
        out = apply_circuit_breaker(self._signals(), history, {"enabled": True})
        by_strat = {s.strategy: s for s in out}
        assert by_strat["regime_hmm"].score < 1.0
        assert by_strat["momentum"].score == 1.0

    def test_does_not_mutate_input(self):
        signals = self._signals()
        history = {"regime_hmm": _returns(100, -0.01, 0.01)}
        apply_circuit_breaker(signals, history, {"enabled": True})
        assert signals[0].score == 1.0


class TestWiringThroughNetScores:
    def _bb(self) -> Blackboard:
        signals = [
            Signal("AAPL", "regime_hmm", 2.0, 1.0, "5d", NOW),
            Signal("AAPL", "momentum", 2.0, 1.0, "5d", NOW),
        ]
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(SignalSet(domain="technical", asof=NOW, signals=signals))
        return bb

    def test_default_config_leaves_scores_unchanged(self):
        history = {"regime_hmm": _returns(100, -0.05, 0.01), "momentum": _returns(100, 0.01, 0.01)}
        ctx = AgentContext(now=NOW, strategy_returns=history)
        scores = net_scores_for_blackboard(self._bb(), ctx, {})
        # Circuit breaker off by default -> plain confidence-weighted mean.
        assert scores["AAPL"] == pytest.approx(2.0)

    def test_enabled_config_damps_gated_strategy_contribution(self):
        history = {"regime_hmm": _returns(100, -0.05, 0.01), "momentum": _returns(100, 0.01, 0.01)}
        ctx = AgentContext(now=NOW, strategy_returns=history)
        config = {"strategy_circuit_breaker": {"enabled": True}}
        scores = net_scores_for_blackboard(self._bb(), ctx, config)
        # regime_hmm's contribution is damped below momentum's undamped 2.0,
        # so the confidence-weighted mean must now be < 2.0 (previously 2.0).
        assert scores["AAPL"] < 2.0

    def test_works_alongside_optimal_method(self):
        history = {"regime_hmm": _returns(100, -0.05, 0.01), "momentum": _returns(100, 0.01, 0.01)}
        ctx = AgentContext(now=NOW, strategy_returns=history)
        config = {
            "signal_combination": {"method": "optimal"},
            "strategy_circuit_breaker": {"enabled": True},
        }
        # Should not raise, and should produce a finite score.
        scores = net_scores_for_blackboard(self._bb(), ctx, config)
        assert np.isfinite(scores["AAPL"])
