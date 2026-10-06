"""Tests for scripts/run_standalone_strategy_evaluation.py mechanics (synthetic data only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_standalone_strategy_evaluation as ev
import standalone_strategy_preregistered_bars as prereg


def test_fingerprint_stable_and_eleven_strategies():
    assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
    assert len(prereg.STRATEGIES) == 11
    assert prereg.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.05 / 11)


class TestBookReturns:
    W = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    R = np.array([[0.0, 0.0], [0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])

    def test_lag0_earns_next_day(self):
        net, gross, turn = ev.book_returns(self.W, self.R, lag=0, cost_bps=0.0)
        assert gross.tolist() == pytest.approx([0.0, 0.01, 0.03, 0.06])
        assert turn.tolist() == pytest.approx([1.0, 0.0, 2.0, 0.0])

    def test_lag1_delays_one_more_day_and_charges_on_adoption(self):
        net, gross, turn = ev.book_returns(self.W, self.R, lag=1, cost_bps=10.0)
        assert gross.tolist() == pytest.approx([0.0, 0.0, 0.03, 0.05])
        assert turn.tolist() == pytest.approx([0.0, 1.0, 0.0, 2.0])
        assert net[3] == pytest.approx(0.05 - 2.0 * 10 / 1e4)


def test_hedge_uses_train_beta_only():
    rng = np.random.default_rng(0)
    spy = rng.normal(0, 0.01, 200)
    net = 0.5 * spy + rng.normal(0, 0.001, 200)
    net[150:] += 5.0 * spy[150:]          # a test-window beta change must not leak into beta
    tr = np.zeros(200, bool)
    tr[:100] = True
    te = np.zeros(200, bool)
    te[150:] = True
    hedged, betas = ev.hedge(net, spy, [tr], [te])
    assert betas[0] == pytest.approx(0.5, abs=0.05)
    assert np.isnan(hedged[:150]).all()


def test_permute_months_is_one_column_permutation_per_month():
    dates = pd.bdate_range("2024-01-01", periods=60)
    W = np.random.default_rng(1).normal(size=(60, 5))
    Wp = ev.permute_months(W, dates, np.random.default_rng(2))
    assert np.abs(Wp).sum(1) == pytest.approx(np.abs(W).sum(1))
    months = dates.to_period("M")
    for m in months.unique():
        rows = np.flatnonzero(months == m)
        perm = [int(np.flatnonzero(np.isclose(W[rows[0]], v))[0]) for v in Wp[rows[0]]]
        for r in rows:
            assert Wp[r] == pytest.approx(W[r][perm])
