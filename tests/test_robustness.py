"""Tests for firm.eval.robustness (Monte Carlo bootstrap analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.eval.robustness import MonteCarloAnalyzer


def _returns(seed: int = 0, n: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, size=n))


class TestMonteCarlo:
    def test_drawdowns_ordered(self):
        mc = MonteCarloAnalyzer(n_simulations=500, seed=42)
        dd = mc.analyze_drawdowns(_returns())
        # Drawdowns are negative; worst_case is the most negative.
        assert dd["worst_case"] <= dd["expected_max_dd"] <= 0.0

    def test_confidence_interval_ordered(self):
        mc = MonteCarloAnalyzer(n_simulations=500, seed=42)
        ci = mc.confidence_interval(_returns(), periods=252)
        assert ci["lower_bound"] <= ci["expected"] <= ci["upper_bound"]

    def test_probability_of_loss_is_fraction(self):
        mc = MonteCarloAnalyzer(n_simulations=500, seed=42)
        pol = mc.probability_of_loss(_returns(), holding_periods=[21, 63])
        assert set(pol) == {21, 63}
        assert all(0.0 <= v <= 1.0 for v in pol.values())

    def test_stable_under_fixed_seed(self):
        a = MonteCarloAnalyzer(n_simulations=300, seed=7).confidence_interval(_returns())
        b = MonteCarloAnalyzer(n_simulations=300, seed=7).confidence_interval(_returns())
        assert a == b

    def test_summary_keys(self):
        mc = MonteCarloAnalyzer(n_simulations=200, seed=1)
        summary = mc.summary(_returns())
        assert {"drawdowns", "probability_of_loss", "confidence_interval"} <= set(summary)

    def test_empty_returns_degrade(self):
        mc = MonteCarloAnalyzer()
        assert mc.summary(pd.Series([], dtype=float)) == {}


class TestSharpeConfidenceInterval:
    def test_ordered(self):
        mc = MonteCarloAnalyzer(n_simulations=500, confidence=0.90, seed=42)
        ci = mc.sharpe_confidence_interval(_returns())
        assert ci["lower_bound"] <= ci["expected"] <= ci["upper_bound"]

    def test_positive_drift_lower_bound_positive(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.003, 0.005, size=500))
        mc = MonteCarloAnalyzer(n_simulations=1000, confidence=0.90, seed=42)
        ci = mc.sharpe_confidence_interval(returns)
        assert ci["lower_bound"] > 0

    def test_negative_drift_lower_bound_not_positive(self):
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(-0.003, 0.01, size=500))
        mc = MonteCarloAnalyzer(n_simulations=1000, confidence=0.90, seed=42)
        ci = mc.sharpe_confidence_interval(returns)
        assert ci["lower_bound"] <= 0

    def test_stable_under_fixed_seed(self):
        a = MonteCarloAnalyzer(n_simulations=300, seed=7).sharpe_confidence_interval(_returns())
        b = MonteCarloAnalyzer(n_simulations=300, seed=7).sharpe_confidence_interval(_returns())
        assert a == b

    def test_too_few_observations_returns_empty(self):
        mc = MonteCarloAnalyzer()
        assert mc.sharpe_confidence_interval(pd.Series([0.01])) == {}
        assert mc.sharpe_confidence_interval(pd.Series([], dtype=float)) == {}
