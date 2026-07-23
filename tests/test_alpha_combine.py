"""Tests for optimal signal combination in firm.agents.analysts."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.analysts import (
    combine_signals_optimal,
    optimal_signal_weights,
)
from firm.contracts.models import Signal

NOW = datetime(2026, 1, 2)


def _signal(strategy: str, score: float, symbol: str = "AAPL") -> Signal:
    return Signal(
        symbol=symbol, strategy=strategy, score=score,
        confidence=1.0, horizon="21d", asof=NOW,
    )


def _series(seed: int, n: int = 250, scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, scale, size=n)


class TestOptimalWeights:
    def test_duplicates_share_weight(self):
        """A duplicated pair should each get ~half of an independent signal."""
        a = _series(1)
        dup = _series(2)
        frame = pd.DataFrame({"A": a, "B": dup, "C": dup})  # B and C identical
        weights, _ = optimal_signal_weights(frame)
        # B and C are redundant → together ~ one independent signal (like A).
        assert weights["A"] > weights["B"]
        assert weights["B"] == pytest.approx(weights["C"], abs=1e-6)
        assert weights["B"] + weights["C"] == pytest.approx(weights["A"], rel=0.25)

    def test_effective_n_drops_with_correlation(self):
        indep = pd.DataFrame({c: _series(i) for i, c in enumerate("ABC")})
        _, eff_indep = optimal_signal_weights(indep)

        base = _series(10)
        corr = pd.DataFrame({"A": base, "B": base, "C": _series(11)})
        _, eff_corr = optimal_signal_weights(corr)
        assert eff_indep > eff_corr
        assert eff_indep == pytest.approx(3.0, abs=0.5)

    def test_single_column(self):
        w, eff = optimal_signal_weights(pd.DataFrame({"A": _series(0)}))
        assert w["A"] == 1.0
        assert eff == 1.0

    def test_higher_variance_downweighted(self):
        frame = pd.DataFrame({"LO": _series(1, scale=0.005), "HI": _series(2, scale=0.05)})
        w, _ = optimal_signal_weights(frame)
        assert w["LO"] > w["HI"]


class TestCombineSignalsOptimal:
    def test_thin_history_falls_back(self):
        signals = [
            _signal("momentum", 1.0),
            _signal("trend", 1.0),
        ]
        # No history → confidence-weighted mean fallback.
        out = combine_signals_optimal(signals, strategy_returns={})
        assert out["AAPL"] == pytest.approx(1.0)

    def test_uses_history_when_available(self):
        rng = np.random.default_rng(0)
        hist = {
            "momentum": pd.Series(rng.normal(0, 0.01, 100)),
            "trend": pd.Series(rng.normal(0, 0.01, 100)),
        }
        signals = [
            _signal("momentum", 2.0),
            _signal("trend", 0.0),
        ]
        out = combine_signals_optimal(signals, strategy_returns=hist)
        assert 0.0 < out["AAPL"] <= 2.0
