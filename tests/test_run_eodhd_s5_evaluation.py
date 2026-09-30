"""Tests for the S5 PHASE 2 evaluation harness
(scripts/run_eodhd_s5_evaluation.py). Synthetic data only -- no network, no
real crypto parquet files are read here (those are exercised by the actual
`build`/`evaluate` CLI runs against data/research/eodhd, separately).
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

import run_eodhd_s5_evaluation as ev  # noqa: E402
import eodhd_s5_crypto_momentum_preregistered_bars as prereg  # noqa: E402


def _dates(start: str, n: int) -> np.ndarray:
    return pd.date_range(start, periods=n, freq="D").values


class TestCoinWeekTable:
    def test_eligible_when_liquid_and_long_enough(self):
        dates = _dates("2020-01-01", 100)
        aclose = np.full(100, 10.0)
        vol = np.full(100, 200_000.0)  # dvol = 10 * 200,000 = $2,000,000/day > $1M floor
        seg = np.zeros(100, dtype=np.int32)
        sundays = np.array([pd.Timestamp("2020-03-01")]).astype("datetime64[ns]")  # day 60
        elig, dvol, ret28 = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert elig[0]
        assert dvol[0] == pytest.approx(2_000_000.0)
        assert ret28[0] == pytest.approx(0.0)  # flat price -> 0 return

    def test_ineligible_below_liquidity_floor(self):
        dates = _dates("2020-01-01", 100)
        aclose = np.full(100, 10.0)
        vol = np.full(100, 1_000.0)  # dvol = $10,000/day, well under $1M
        seg = np.zeros(100, dtype=np.int32)
        sundays = np.array([pd.Timestamp("2020-03-01")]).astype("datetime64[ns]")
        elig, _, _ = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert not elig[0]

    def test_ineligible_when_too_short_history(self):
        dates = _dates("2020-01-01", 10)  # only 10 days -- can't cover 28d lookback or 30d window
        aclose = np.full(10, 10.0)
        vol = np.full(10, 500_000.0)
        seg = np.zeros(10, dtype=np.int32)
        sundays = np.array([pd.Timestamp("2020-01-10")]).astype("datetime64[ns]")
        elig, _, _ = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert not elig[0]

    def test_ineligible_when_stale(self):
        # Last real bar is 10 days before the review -- exceeds max_stale_days=3.
        dates = _dates("2020-01-01", 100)
        aclose = np.full(100, 10.0)
        vol = np.full(100, 500_000.0)
        seg = np.zeros(100, dtype=np.int32)
        stale_dates = dates[:90]  # series "ends" 10 days before the review below
        sundays = np.array([pd.Timestamp(dates[99])]).astype("datetime64[ns]")
        elig, _, _ = ev.coin_week_table(stale_dates, aclose[:90], vol[:90], seg[:90], sundays, floor_usd=1_000_000.0)
        assert not elig[0]

    def test_ineligible_when_segment_break_within_30_days(self):
        # Segment 0 for the first 90 days, a break (new segment 1) at day 85
        # -- only 15 days into the new segment by the review (day 100) --
        # ineligible per DATA['segment_rule'] (segment must cover 30 days back).
        dates = _dates("2020-01-01", 100)
        aclose = np.full(100, 10.0)
        vol = np.full(100, 500_000.0)
        seg = np.zeros(100, dtype=np.int32)
        seg[85:] = 1
        sundays = np.array([pd.Timestamp(dates[99])]).astype("datetime64[ns]")
        elig, _, _ = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert not elig[0]

    def test_eligible_once_new_segment_covers_30_days(self):
        dates = _dates("2020-01-01", 140)
        aclose = np.full(140, 10.0)
        vol = np.full(140, 500_000.0)
        seg = np.zeros(140, dtype=np.int32)
        seg[40:] = 1  # new segment starts day 40
        sundays = np.array([pd.Timestamp(dates[139])]).astype("datetime64[ns]")  # 99 days into new segment
        elig, _, _ = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert elig[0]

    def test_ret28_computed_correctly_within_same_segment(self):
        dates = _dates("2020-01-01", 100)
        aclose = np.linspace(10.0, 20.0, 100)  # rising price
        vol = np.full(100, 500_000.0)
        seg = np.zeros(100, dtype=np.int32)
        sundays = np.array([pd.Timestamp(dates[99])]).astype("datetime64[ns]")
        elig, _, ret28 = ev.coin_week_table(dates, aclose, vol, seg, sundays, floor_usd=1_000_000.0)
        assert elig[0]
        expected = aclose[99] / aclose[99 - 28] - 1.0
        assert ret28[0] == pytest.approx(expected)


class TestTickerCollisions:
    def test_keeps_higher_dvol_member(self):
        eligible = np.array([True, True, False])
        dvol = np.array([5.0, 9.0, 1.0])
        base_syms = np.array(["TAO", "TAO", "SOL"])
        out = ev.resolve_ticker_collisions(eligible, dvol, base_syms)
        assert list(out) == [False, True, False]

    def test_no_collision_leaves_untouched(self):
        eligible = np.array([True, True])
        dvol = np.array([5.0, 9.0])
        base_syms = np.array(["TAO", "SOL"])
        out = ev.resolve_ticker_collisions(eligible, dvol, base_syms)
        assert list(out) == [True, True]


class TestSelectHoldings:
    def test_top_tercile_equal_weight_no_filter(self):
        n = 20
        eligible = np.ones(n, dtype=bool)
        dvol = np.arange(n, dtype=float) + 1  # coin n-1 has highest dvol -> in top-20 pool (all are)
        ret28 = np.array([1.0 if i < 7 else -1.0 for i in range(n)])  # first 7 (by ret) score highest
        # Rank by ret28 descending: indices 0..6 have the highest return (1.0)
        holdings = ev.select_holdings(eligible, dvol, ret28, n_universe=20, absolute_filter=False)
        assert len(holdings) == prereg.tercile_count(20) == 7
        assert all(abs(w - 1 / 7) < 1e-9 for w in holdings.values())
        assert set(holdings) == set(range(7))

    def test_absolute_filter_drops_negative_return_slots(self):
        n = 20
        eligible = np.ones(n, dtype=bool)
        dvol = np.ones(n)
        ret28 = np.full(n, -0.01)  # every candidate has a negative 28d return
        ret28[0] = 0.5
        holdings = ev.select_holdings(eligible, dvol, ret28, n_universe=20, absolute_filter=True)
        # Only index 0 clears the absolute filter; everything else in the
        # tercile is cash (dropped from the dict, not renormalised).
        assert holdings == {0: pytest.approx(1 / 7)}

    def test_smaller_pool_uses_its_own_tercile_count(self):
        n = 3
        eligible = np.ones(n, dtype=bool)
        dvol = np.ones(n)
        ret28 = np.array([1.0, 0.5, -0.5])
        holdings = ev.select_holdings(eligible, dvol, ret28, n_universe=20, absolute_filter=False)
        assert len(holdings) == prereg.tercile_count(3) == 1
        assert 0 in holdings

    def test_empty_pool_returns_empty(self):
        eligible = np.zeros(5, dtype=bool)
        holdings = ev.select_holdings(eligible, np.zeros(5), np.zeros(5), 20, False)
        assert holdings == {}


class TestPlaceboHoldings:
    def test_draws_from_same_pool_size(self):
        n = 20
        eligible = np.ones(n, dtype=bool)
        dvol = np.arange(n, dtype=float) + 1
        ret28 = np.zeros(n)
        rng = np.random.default_rng(0)
        holdings = ev.placebo_holdings(eligible, dvol, ret28, n_universe=20, absolute_filter=False, rng=rng)
        assert len(holdings) == prereg.tercile_count(20)
        assert all(0 <= i < n for i in holdings)

    def test_seeded_reproducibility(self):
        n = 30
        eligible = np.ones(n, dtype=bool)
        dvol = np.arange(n, dtype=float) + 1
        ret28 = np.linspace(-1, 1, n)
        h1 = ev.placebo_holdings(eligible, dvol, ret28, 20, True, np.random.default_rng(42))
        h2 = ev.placebo_holdings(eligible, dvol, ret28, 20, True, np.random.default_rng(42))
        assert h1 == h2


class TestAopenReturns:
    def test_flat_price_zero_return(self):
        dates = _dates("2020-01-01", 10)
        open_ = np.full(10, 10.0)
        close = np.full(10, 10.0)
        aclose = np.full(10, 10.0)
        cal = pd.date_range("2020-01-01", periods=10, freq="D")
        r = ev.aopen_returns(dates, open_, close, aclose, cal)
        assert np.allclose(r[:-1], 0.0)

    def test_matches_hand_computed_gap_up(self):
        dates = _dates("2020-01-01", 3)
        open_ = np.array([10.0, 12.0, 12.0])
        close = np.array([10.0, 12.0, 12.0])
        aclose = np.array([10.0, 12.0, 12.0])  # no adjustment (adjusted_close == close)
        cal = pd.date_range("2020-01-01", periods=3, freq="D")
        r = ev.aopen_returns(dates, open_, close, aclose, cal)
        assert r[0] == pytest.approx(0.2)  # aopen day0=10 -> day1=12

    def test_gap_day_forward_filled_to_zero_return(self):
        dates = np.array(_dates("2020-01-01", 1).tolist() + _dates("2020-01-03", 1).tolist())
        open_ = np.array([10.0, 11.0])
        close = np.array([10.0, 11.0])
        aclose = np.array([10.0, 11.0])
        cal = pd.date_range("2020-01-01", periods=3, freq="D")  # day 2020-01-02 has no real bar
        r = ev.aopen_returns(dates, open_, close, aclose, cal)
        assert r[0] == pytest.approx(0.0)  # aopen ffilled through the gap day


class TestSimulateWithDiagnostics:
    def test_decide_at_i_takes_effect_at_i_plus_1(self):
        # 3 assets, 4 days. Asset 0 doubles on day 2 only. Rebalance into
        # 100% asset 0 decided at day i=0 -> should hold from day i=1 onward
        # (captures day-1's move, matching the aopen-to-aopen "next bar" timing).
        rets = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],   # asset 0 +100% on day 1
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        rf = np.zeros(4)

        def decide(i, w):
            return {0: 1.0} if i == 0 else None

        out, held, cost, turnover = ev.simulate_with_diagnostics(rets, rf, decide, cost_bps=0.0)
        assert out[1] == pytest.approx(1.0)  # day 1 captured the +100% move
        assert held[0, 0] == pytest.approx(1.0)  # target already set after day-0 processing
        assert turnover[0] == pytest.approx(1.0)

    def test_cost_charged_on_turnover(self):
        rets = np.zeros((2, 1))
        rf = np.zeros(2)

        def decide(i, w):
            return {0: 0.5} if i == 0 else None

        out, held, cost, turnover = ev.simulate_with_diagnostics(rets, rf, decide, cost_bps=35.0)
        assert cost[0] == pytest.approx(0.5 * 35.0 / 1e4)
        assert out[0] == pytest.approx(-0.5 * 35.0 / 1e4)

    def test_rejects_leverage(self):
        rets = np.zeros((1, 1))
        rf = np.zeros(1)
        with pytest.raises(ValueError):
            ev.simulate_with_diagnostics(rets, rf, lambda i, w: {0: 1.5}, cost_bps=0.0)

    def test_cash_earns_rf(self):
        out, _, _, _ = ev.simulate_with_diagnostics(np.zeros((3, 1)), np.full(3, 0.001), lambda i, w: None, 0.0)
        assert np.allclose(out, 0.001)


class TestCalendarMapping:
    def test_weekend_return_compounds_onto_next_monday(self):
        # Fri, Sat, Sun bars; SPY only trades Fri and the following Mon.
        crypto = pd.Series([0.01, 0.02, 0.03],
                            index=pd.date_range("2024-01-05", periods=3, freq="D"))  # Fri, Sat, Sun
        spy_dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])  # Fri, Mon
        out = ev.crypto_to_spy_calendar(crypto, spy_dates)
        assert out.loc["2024-01-05"] == pytest.approx(0.0)  # Friday's own bar isn't known at Friday's US close
        assert out.loc["2024-01-08"] == pytest.approx(1.01 * 1.02 * 1.03 - 1)


class TestStatsHelpers:
    def test_max_drawdown(self):
        assert ev.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)

    def test_stationary_indices_valid_and_blocky(self):
        rng = np.random.default_rng(0)
        idx = ev.stationary_indices(500, 50, 91, rng)
        assert idx.shape == (50, 500)
        assert idx.min() >= 0 and idx.max() < 500

    def test_paired_gap_boot_zero_for_identical_series(self):
        x = np.random.default_rng(1).normal(0.0005, 0.01, 400)
        out = ev.paired_sharpe_gap_boot(x, x, mean_block=91, ann=ev.ANN_CRYPTO, n_boot=200, seed=3)
        assert np.allclose(out, 0.0)

    def test_cagr_matches_hand_solved(self):
        r = np.full(365, (1.10) ** (1 / 365) - 1)  # exactly 10% over 1 year of daily compounding
        assert ev.cagr(r, 365) == pytest.approx(0.10, abs=1e-6)

    def test_classify_matches_prereg(self):
        bars = {"A1": True}
        assert ev.classify(bars, False, {}) == "A"
        assert ev.classify({"A1": False}, True, {}) == "D"


class TestPreregIssuesDeclared:
    def test_four_issues_recorded(self):
        assert len(ev.PREREG_ISSUES) == 4
        assert all(isinstance(s, str) and len(s) > 20 for s in ev.PREREG_ISSUES)
