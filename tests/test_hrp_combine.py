"""Tests for Hierarchical Risk Parity (HRP) signal combination.

Mirrors tests/test_alpha_combine.py's structure and fixtures for the
inverse-covariance ``optimal`` method, applied to the HRP alternative in
firm.agents.analysts (hrp_signal_weights / combine_signals_hrp).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.analysts import (
    combine_signals_hrp,
    hrp_signal_weights,
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


class TestHRPWeights:
    def test_weights_sum_to_one_and_nonnegative(self):
        frame = pd.DataFrame({c: _series(i) for i, c in enumerate("ABCD")})
        weights, _ = hrp_signal_weights(frame)
        assert (weights.to_numpy() >= 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)

    def test_equal_uncorrelated_strategies_get_near_equal_weight(self):
        """With no correlation structure to exploit, HRP should land close
        to equal weight across equally-volatile, independent strategies."""
        frame = pd.DataFrame({c: _series(i) for i, c in enumerate("ABC")})
        weights, _ = hrp_signal_weights(frame)
        assert weights["A"] == pytest.approx(1 / 3, abs=0.1)
        assert weights["B"] == pytest.approx(1 / 3, abs=0.1)
        assert weights["C"] == pytest.approx(1 / 3, abs=0.1)

    def test_correlated_pair_shares_weight_like_one_independent_signal(self):
        """The defining HRP property `optimal` lacks a clean version of:
        two highly-correlated strategies should be clustered together and,
        combined, contribute about as much weight as one independent
        strategy — not double-counted as two separate independent edges."""
        indep = _series(1)
        base = _series(2)
        noise = np.random.default_rng(3).normal(0, 0.001, size=len(base))
        frame = pd.DataFrame({
            "A": indep,
            "B": base,
            "C": base + noise,  # near-duplicate of B
        })
        weights, _ = hrp_signal_weights(frame)
        assert weights["B"] == pytest.approx(weights["C"], abs=1e-2)
        # B+C together should be in the same ballpark as A, not ~2x it.
        assert (weights["B"] + weights["C"]) == pytest.approx(weights["A"], rel=0.5)

    def test_higher_variance_downweighted(self):
        frame = pd.DataFrame({"LO": _series(1, scale=0.005), "HI": _series(2, scale=0.05)})
        w, _ = hrp_signal_weights(frame)
        assert w["LO"] > w["HI"]

    def test_single_column(self):
        w, eff = hrp_signal_weights(pd.DataFrame({"A": _series(0)}))
        assert w["A"] == 1.0
        assert eff == 1.0

    def test_thin_history_falls_back_to_equal_weight(self):
        frame = pd.DataFrame({"A": [0.01], "B": [0.02]})  # < 2 usable rows each
        w, eff = hrp_signal_weights(frame)
        assert w["A"] == pytest.approx(0.5)
        assert w["B"] == pytest.approx(0.5)
        assert eff == pytest.approx(2.0)


class TestCombineSignalsHRP:
    def test_thin_history_falls_back(self):
        signals = [
            _signal("momentum", 1.0),
            _signal("trend", 1.0),
        ]
        out = combine_signals_hrp(signals, strategy_returns={})
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
        out = combine_signals_hrp(signals, strategy_returns=hist)
        assert 0.0 < out["AAPL"] <= 2.0

    def test_fewer_than_two_strategies_falls_back(self):
        signals = [_signal("momentum", 1.5)]
        out = combine_signals_hrp(
            signals, strategy_returns={"momentum": pd.Series([0.01, 0.02, 0.03])}
        )
        assert out["AAPL"] == pytest.approx(1.5)
