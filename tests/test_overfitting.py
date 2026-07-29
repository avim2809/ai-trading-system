"""Tests for firm.eval.overfitting (PBO / Deflated & Probabilistic Sharpe)."""

from __future__ import annotations

import numpy as np
import pytest

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

    def test_embargo_zero_matches_default(self):
        """``embargo_pct=0.0`` must reproduce the un-embargoed split exactly
        — every existing caller/test relies on this being a strict no-op."""
        rng = np.random.default_rng(0)
        matrix = np.hstack(
            [rng.normal(0.002, 0.01, size=(240, 1)), rng.normal(0, 0.01, size=(240, 9))]
        )
        assert cscv_pbo(matrix, n_partitions=8, embargo_pct=0.0) == cscv_pbo(
            matrix, n_partitions=8
        )

    def test_embargo_purges_boundary_leakage(self):
        """A trial column with a leak spike concentrated at every block's
        edge (simulating serial-correlation bleed from the adjacent
        in-sample block) should be scored differently once the embargo
        purges those boundary rows from the out-of-sample side."""
        rng = np.random.default_rng(42)
        rows_per_block, n_blocks = 20, 6
        T = rows_per_block * n_blocks
        genuine = rng.normal(0.0015, 0.01, size=T)
        noise = rng.normal(0.0, 0.01, size=(T, 5))
        leak_rows = 4
        for b in range(1, n_blocks):
            start = b * rows_per_block
            noise[start : start + leak_rows, 0] += 0.15
        matrix = np.column_stack([genuine, noise])

        pbo_no_embargo = cscv_pbo(matrix, n_partitions=n_blocks, embargo_pct=0.0)
        pbo_embargo = cscv_pbo(matrix, n_partitions=n_blocks, embargo_pct=0.2)
        assert pbo_no_embargo != pbo_embargo

    def test_embargo_full_purge_returns_uninformative(self):
        """An embargo large enough to purge every split's entire OOS block
        (S=2: the two blocks are always mutually adjacent) leaves nothing to
        rank against — must degrade to the uninformative 0.5, not error."""
        rng = np.random.default_rng(1)
        matrix = rng.normal(0, 0.01, size=(20, 3))
        assert cscv_pbo(matrix, n_partitions=2, embargo_pct=1.0) == 0.5


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

    def test_no_trial_data_omits_pbo_and_dsr_equals_psr(self):
        """Without genuine competing-candidate data, PBO can't be computed
        (there's nothing to run CSCV over) and DSR must degrade to plain PSR
        — treating sequential OOS folds of one fixed config as if they were
        independent trials would misrepresent how many configs were tried."""
        rng = np.random.default_rng(6)
        folds = [rng.normal(0.001, 0.01, size=60) for _ in range(5)]
        result = walk_forward_overfitting(folds)
        assert "pbo" not in result
        assert result["deflated_sharpe"] == pytest.approx(
            result["probabilistic_sharpe"]
        )

    def test_genuine_trial_data_produces_pbo(self):
        """With a real per-fold parameter grid, PBO is computed from each
        fold's own (candidates x train-periods) matrix, not from the OOS
        fold-to-fold spread."""
        rng = np.random.default_rng(7)
        oos_folds = [rng.normal(0.001, 0.01, size=40) for _ in range(4)]
        # Each fold "trained" 3 candidates over a 60-period train window.
        trial_returns = [
            [rng.normal(0.0005, 0.01, size=60) for _ in range(3)] for _ in range(4)
        ]
        result = walk_forward_overfitting(oos_folds, fold_trial_returns=trial_returns)
        assert "pbo" in result
        assert 0.0 <= result["pbo"] <= 1.0
        assert result["pbo_n_folds"] == 4

    def test_genuine_trial_data_feeds_dsr_trial_count(self):
        """DSR's trial penalty should reflect the real number of candidates
        tried (n_folds * candidates_per_fold), not the fold count alone."""
        rng = np.random.default_rng(8)
        oos_folds = [rng.normal(0.001, 0.01, size=40) for _ in range(4)]
        many_trials = [
            [rng.normal(0.0005, 0.01, size=60) for _ in range(10)] for _ in range(4)
        ]
        few_trials = [
            [rng.normal(0.0005, 0.01, size=60) for _ in range(1)] for _ in range(4)
        ]
        many_result = walk_forward_overfitting(oos_folds, fold_trial_returns=many_trials)
        few_result = walk_forward_overfitting(oos_folds, fold_trial_returns=few_trials)
        assert many_result["deflated_sharpe"] <= few_result["deflated_sharpe"] + 1e-9
