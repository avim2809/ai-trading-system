"""Tests for firm.eval.overfitting (PBO / Deflated & Probabilistic Sharpe)."""

from __future__ import annotations

import numpy as np

from firm.eval.overfitting import (
    cscv_pbo,
    deflated_sharpe,
    probabilistic_sharpe,
    verdict,
    walk_forward_overfitting,
)


class TestPBO:
    def test_genuine_signal_low_pbo(self):
        """One column with a real, consistent edge → low PBO (not overfit)."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, size=(240, 9))
        genuine = rng.normal(0.002, 0.01, size=(240, 1))  # persistent drift
        matrix = np.hstack([genuine, noise])
        pbo = cscv_pbo(matrix, n_partitions=8)
        assert 0.0 <= pbo <= 1.0
        assert pbo < 0.5

    def test_pure_noise_higher_pbo_than_genuine(self):
        """Pure-noise trials should look more overfit than a genuine signal."""
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.01, size=(240, 10))
        pbo_noise = cscv_pbo(noise, n_partitions=8)

        genuine = np.hstack(
            [rng.normal(0.003, 0.01, size=(240, 1)), rng.normal(0, 0.01, size=(240, 9))]
        )
        pbo_genuine = cscv_pbo(genuine, n_partitions=8)
        assert pbo_noise > pbo_genuine

    def test_too_small_returns_uninformative(self):
        assert cscv_pbo(np.zeros((3, 1))) == 0.5


class TestSharpeStats:
    def test_genuine_drift_high_psr(self):
        rng = np.random.default_rng(2)
        good = rng.normal(0.001, 0.008, size=500)
        assert probabilistic_sharpe(good, 0.0) > 0.9

    def test_dsr_never_exceeds_psr(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0008, 0.01, size=400)
        trials = rng.normal(0.05, 0.05, size=50)
        psr = probabilistic_sharpe(returns, 0.0)
        dsr = deflated_sharpe(returns, trials)
        assert dsr <= psr + 1e-9

    def test_more_trials_lower_dsr(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(0.0008, 0.01, size=400)
        few = rng.normal(0.05, 0.05, size=5)
        many = rng.normal(0.05, 0.05, size=200)
        assert deflated_sharpe(returns, many) <= deflated_sharpe(returns, few) + 1e-9

    def test_short_series_returns_zero(self):
        assert probabilistic_sharpe(np.array([0.1, 0.2])) == 0.0


class TestVerdict:
    def test_pass_when_both_good(self):
        v = verdict(pbo=0.1, dsr=0.99)
        assert v["verdict"] == "pass"
        assert len(v["lines"]) == 2

    def test_fail_when_overfit(self):
        assert verdict(pbo=0.8, dsr=0.99)["verdict"] == "fail"


class TestWalkForward:
    def test_walk_forward_overfitting_shape(self):
        rng = np.random.default_rng(5)
        folds = [rng.normal(0.001, 0.01, size=60) for _ in range(6)]
        result = walk_forward_overfitting(folds)
        assert result["n_folds"] == 6
        assert "deflated_sharpe" in result
        assert "probabilistic_sharpe" in result
        assert result["deflated_sharpe"] <= result["probabilistic_sharpe"] + 1e-9
        assert result["verdict"] in ("pass", "fail")

    def test_too_few_folds_empty(self):
        assert walk_forward_overfitting([np.array([0.1, 0.2, 0.3])]) == {}
