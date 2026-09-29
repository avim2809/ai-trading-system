"""Tests for the alternative-premia edge-search pre-registration and harness
(scripts/alt_premia_preregistered_bars.py, scripts/run_alt_premia_evaluation.py).

Pure mechanics on synthetic data only -- no network, no real series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import alt_premia_preregistered_bars as prereg  # noqa: E402
import run_alt_premia_evaluation as ev  # noqa: E402


class TestPrereg:
    def test_fingerprint_is_deterministic(self):
        assert prereg.bars_fingerprint() == prereg.bars_fingerprint()
        assert len(prereg.bars_fingerprint()) == 64

    def test_every_candidate_has_windows_and_benchmarks(self):
        for name, spec in prereg.CANDIDATES.items():
            assert pd.Timestamp(spec["eval_start"]) < pd.Timestamp(spec["post_publication_start"]), name
            assert spec["benchmarks"], name

    def test_bonferroni_divisor_is_candidate_count(self):
        assert prereg.BOOTSTRAP["alpha_one_sided"] == pytest.approx(0.05 / len(prereg.CANDIDATES))

    @pytest.mark.parametrize("a_ok,d,b_ok,want", [
        (True, False, True, "A"),
        (True, True, True, "A"),     # A takes precedence over everything
        (False, True, True, "D"),    # measurably worse beats "safer"
        (False, False, True, "B"),
        (False, False, False, "C"),
    ])
    def test_classify_precedence(self, a_ok, d, b_ok, want):
        bars = {"A1": a_ok, "A2": True}
        tier_b = {"B_a": b_ok, "B_b": True}
        assert prereg.classify(bars, d, tier_b) == want


class TestSimulator:
    def test_drift_and_cost_accounting(self):
        rets = np.array([[0.0], [0.10], [0.0]])
        rf = np.zeros(3)
        # Buy 50% at close of day 0; hold.
        net, held = ev.simulate(rets, rf, lambda i, w: np.array([0.5]) if i == 0 else None,
                                np.array([10.0]))
        assert net[0] == pytest.approx(-0.5 * 10 / 1e4)
        assert net[1] == pytest.approx(0.05)
        assert held[1, 0] == pytest.approx(0.55 / 1.05)

    def test_rejects_leverage(self):
        with pytest.raises(ValueError):
            ev.simulate(np.zeros((2, 1)), np.zeros(2), lambda i, w: np.array([1.5]), np.zeros(1))

    def test_cash_earns_rf(self):
        net, _ = ev.simulate(np.zeros((3, 1)), np.full(3, 0.001), lambda i, w: None, np.zeros(1))
        assert np.allclose(net, 0.001)


class TestStats:
    def test_stationary_indices_valid_and_blocky(self):
        rng = np.random.default_rng(0)
        idx = ev.stationary_indices(500, 50, 20, rng)
        assert idx.shape == (50, 500)
        assert idx.min() >= 0 and idx.max() < 500
        consecutive = (np.diff(idx, axis=1) == 1).mean()
        assert 0.9 < consecutive < 0.99   # ~1 - 1/20

    def test_paired_gap_boot_is_zero_for_identical_series(self):
        x = np.random.default_rng(1).normal(0.0005, 0.01, 400)
        assert np.allclose(ev.paired_sharpe_gap_boot(x, x, seed=3), 0.0)

    def test_worst_episodes_finds_deepest_non_overlapping(self):
        r = np.array([0.1, -0.2, 0.5, -0.1, 0.3, -0.05])
        eps = ev.worst_episodes(r, 3)
        depths = [d for _, _, d in eps]
        assert depths == sorted(depths)
        assert depths[0] == pytest.approx(-0.2)

    def test_max_drawdown(self):
        assert ev.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)


class TestCalendar:
    def test_btc_bar_maps_to_next_spy_date_strictly_after(self):
        spy = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])      # Fri, Mon
        btc = pd.Series([0.01, 0.02, 0.03, 0.04],
                        index=pd.date_range("2024-01-05", periods=4, freq="D"))  # Fri..Mon
        out = ev._btc_to_spy(btc, spy)
        assert out.loc["2024-01-05"] == 0.0            # Friday's own bar isn't known at Friday's US close
        assert out.loc["2024-01-08"] == pytest.approx(1.01 * 1.02 * 1.03 - 1)  # Fri, Sat, Sun bars
