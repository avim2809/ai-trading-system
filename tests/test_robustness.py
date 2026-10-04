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


# ---------------------------------------------------------------------------
# P1-10: opt-in stationary bootstrap (default iid path must stay bit-identical)
# ---------------------------------------------------------------------------

import json
import subprocess
import sys
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


def _golden_input() -> np.ndarray:
    return np.random.default_rng(0).normal(0.0005, 0.01, 100)


def _ar1(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.01, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def test_default_iid_bit_identical():
    golden = np.load(_FIXTURES / "robustness_iid_golden.npy")
    out = MonteCarloAnalyzer(n_simulations=50, seed=42).bootstrap_returns(_golden_input())
    assert np.array_equal(out, golden)
    # explicit method="iid" is the same stream
    out2 = MonteCarloAnalyzer(n_simulations=50, seed=42, method="iid").bootstrap_returns(_golden_input())
    assert np.array_equal(out2, golden)
    expected = json.loads((_FIXTURES / "robustness_summary_golden.json").read_text())
    got = json.loads(json.dumps(MonteCarloAnalyzer(n_simulations=50, seed=42).summary(_golden_input())))
    assert got == expected


def test_stationary_changes_drawdown_tail_on_autocorrelated():
    x = _ar1(0.4, 1000, seed=3)
    dd_iid = MonteCarloAnalyzer(n_simulations=1000, seed=7).analyze_drawdowns(x)
    dd_sb = MonteCarloAnalyzer(n_simulations=1000, seed=7, method="stationary").analyze_drawdowns(x)
    # drawdowns are negative fractions: a fatter tail is a MORE negative worst_95pct
    assert dd_sb["worst_95pct"] < dd_iid["worst_95pct"]


def test_stationary_shape_and_determinism():
    x = _ar1(0.3, 300, seed=1)
    a = MonteCarloAnalyzer(n_simulations=20, seed=5, method="stationary", block_len=5.0)
    b = MonteCarloAnalyzer(n_simulations=20, seed=5, method="stationary", block_len=5.0)
    sa = a.bootstrap_returns(x, 120)
    assert sa.shape == (20, 120)
    assert np.array_equal(sa, b.bootstrap_returns(x, 120))
    # per-call override
    over = MonteCarloAnalyzer(n_simulations=20, seed=5).bootstrap_returns(
        x, 120, method="stationary", block_len=5.0
    )
    assert np.array_equal(over, sa)


def test_stationary_empty_input_returns_zeros():
    out = MonteCarloAnalyzer(n_simulations=4, method="stationary").bootstrap_returns(
        np.array([np.nan])
    )
    assert out.shape == (4, 0) and not out.any()


def test_stationary_not_imported_by_default():
    code = (
        "import sys, numpy as np\n"
        "from firm.eval.robustness import MonteCarloAnalyzer\n"
        "mc = MonteCarloAnalyzer(n_simulations=10)\n"
        "mc.summary(np.random.default_rng(0).normal(0, 0.01, 200))\n"
        "assert 'firm.validation.bootstrap' not in sys.modules\n"
        "assert 'firm.validation' not in sys.modules\n"
        "MonteCarloAnalyzer(n_simulations=10, method='stationary').bootstrap_returns(np.arange(50.0))\n"
        "assert 'firm.validation.bootstrap' in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        MonteCarloAnalyzer(method="nope").bootstrap_returns(_golden_input())
    with pytest.raises(ValueError):
        MonteCarloAnalyzer().bootstrap_returns(_golden_input(), method="nope")
