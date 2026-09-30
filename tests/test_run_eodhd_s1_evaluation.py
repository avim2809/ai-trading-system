"""Synthetic-data tests for scripts/run_eodhd_s1_evaluation.py -- the S1 Phase-2
harness. No real EODHD/FRED files are touched: every test builds its own small,
deterministic price/volume arrays. Covers: return math (adjusted-open/overnight/
intraday decomposition), next-open execution + costs, cash accrual (BIL/DTB3
splice), the VFITX/IEF splice, the placebo mechanics, and the halves split.
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

import run_eodhd_s1_evaluation as ev  # noqa: E402
import eodhd_s1_industry_momentum_preregistered_bars as prereg  # noqa: E402


# ---------------------------------------------------------------------------
# Return math
# ---------------------------------------------------------------------------

class TestAdjustedOpenAndSplitReturn:
    def test_adjusted_open_formula(self):
        open_ = np.array([100.0, 50.0])
        close_ = np.array([102.0, 51.0])
        adj_close_ = np.array([51.0, 51.0])  # a 2-for-1 split on the first bar
        out = ev.adjusted_open(open_, close_, adj_close_)
        assert out[0] == pytest.approx(100.0 * 51.0 / 102.0)
        assert out[1] == pytest.approx(50.0)

    def test_overnight_intraday_recompose_to_full_return(self):
        # Random-ish synthetic path with a stock split between two bars.
        rng = np.random.default_rng(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
        adj_close = close.copy()
        adj_close[20:] /= 2.0  # 2-for-1 split effective bar 20
        open_ = close * (1 + rng.normal(0, 0.003, 50))
        adj_open = ev.adjusted_open(open_, close, adj_close)
        overnight, intraday = ev.split_day_return(np.r_[np.nan, adj_close[:-1]], adj_open, adj_close)
        full = adj_close[1:] / adj_close[:-1] - 1.0
        recomposed = (1 + overnight[1:]) * (1 + intraday[1:]) - 1.0
        np.testing.assert_allclose(recomposed, full, rtol=1e-10)

    def test_intraday_is_raw_same_day_open_to_close(self):
        # Same-day adjustment factor cancels: intraday = close/open - 1, unadjusted.
        adj_close_prev = np.array([100.0])
        open_ = np.array([50.0])
        close_ = np.array([52.0])
        adj_close_ = np.array([26.0])  # split factor 2 applied same day
        adj_open = ev.adjusted_open(open_, close_, adj_close_)
        _, intraday = ev.split_day_return(adj_close_prev, adj_open, adj_close_)
        assert intraday[0] == pytest.approx(close_[0] / open_[0] - 1.0)


# ---------------------------------------------------------------------------
# Cash accrual (BIL / DTB3) and the VFITX/IEF splice
# ---------------------------------------------------------------------------

class TestCashAndSplice:
    def test_cash_uses_dtb3_before_bil_and_bil_after(self):
        idx = pd.date_range("2007-05-28", periods=6, freq="D")
        bil = pd.Series([np.nan, np.nan, np.nan, 0.0001, 0.0002, 0.0003], index=idx)
        dtb3 = pd.Series([0.0002] * 6, index=idx)
        out = ev.cash_return_series(bil, dtb3)
        assert out.iloc[:3].tolist() == pytest.approx([0.0002] * 3)
        assert out.iloc[3:].tolist() == pytest.approx([0.0001, 0.0002, 0.0003])

    def test_bond_splice_uses_vfitx_before_ief_inception(self):
        idx = pd.date_range("2002-07-20", periods=8, freq="D")
        ief = pd.Series([np.nan] * 5 + [0.001, 0.0015, -0.0005], index=idx)
        vfitx = pd.Series([0.0003] * 8, index=idx)
        out = ev.splice_bond_return(ief, vfitx)
        assert out.iloc[:5].tolist() == pytest.approx([0.0003] * 5)
        assert out.iloc[5:].tolist() == pytest.approx([0.001, 0.0015, -0.0005])

    def test_splice_falls_back_on_iefs_own_undefined_first_day(self):
        # IEF's pct_change is NaN on its very own inception day (no prior IEF
        # close) -- that day must still come from the VFITX proxy.
        idx = pd.date_range("2002-07-25", periods=3, freq="D")
        ief = pd.Series([np.nan, np.nan, 0.002], index=idx)  # inception at idx[1]
        vfitx = pd.Series([0.0001, 0.0002, 0.0003], index=idx)
        out = ev.splice_bond_return(ief, vfitx)
        assert out.tolist() == pytest.approx([0.0001, 0.0002, 0.002])


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

class TestCosts:
    def test_cost_tier_by_adv_threshold(self):
        assert ev.cost_bps_for_adv(60_000_000) == 3.0
        assert ev.cost_bps_for_adv(50_000_000) == 3.0  # >= boundary
        assert ev.cost_bps_for_adv(49_999_999) == 10.0
        assert ev.cost_bps_for_adv(1_000_000) == 10.0

    def test_nan_adv_defaults_to_the_higher_tier(self):
        assert ev.cost_bps_for_adv(float("nan")) == 10.0
        assert ev.cost_bps_for_adv(None) == 10.0


# ---------------------------------------------------------------------------
# Selection / ranking
# ---------------------------------------------------------------------------

class TestSelectHoldings:
    def test_top_tercile_rounds_and_selects_highest_signal(self):
        signal = {"A": 0.10, "B": 0.05, "C": 0.20, "D": -0.05, "E": 0.01, "F": 0.15}
        held = ev.select_holdings(signal, list(signal), "top_tercile", n_min_for_trading=3)
        assert len(held) == round(6 / 3)  # 2
        assert held == ["C", "F"]

    def test_top_n_caps_at_available_pool(self):
        signal = {"A": 0.1, "B": 0.2}
        held = ev.select_holdings(signal, list(signal), "top_n", n_min_for_trading=1, top_n=3)
        assert held == ["B", "A"]

    def test_tie_break_is_ascending_ticker_symbol(self):
        signal = {"ZZZ": 0.10, "AAA": 0.10, "MMM": 0.10}
        held = ev.select_holdings(signal, list(signal), "top_n", n_min_for_trading=1, top_n=1)
        assert held == ["AAA"]

    def test_below_n_min_for_trading_returns_cash(self):
        signal = {"A": 0.1, "B": 0.2}
        held = ev.select_holdings(signal, list(signal), "top_tercile", n_min_for_trading=9)
        assert held == []

    def test_nan_signal_excludes_from_pool(self):
        signal = {"A": 0.1, "B": float("nan"), "C": 0.2}
        held = ev.select_holdings(signal, list(signal), "top_n", n_min_for_trading=1, top_n=3)
        assert set(held) == {"A", "C"}


# ---------------------------------------------------------------------------
# Halves
# ---------------------------------------------------------------------------

class TestHalfWindows:
    def test_halves_partition_the_window_with_no_gap_or_overlap(self):
        h1s, h1e, h2s, h2e = ev.half_windows("2002-01-31", "2026-09-29", "2014-05-31")
        assert h1s == "2002-01-31" and h1e == "2014-05-31"
        assert h2s == "2014-06-01" and h2e == "2026-09-29"
        dates = pd.date_range("2002-01-31", "2026-09-29", freq="D")
        h1 = (dates >= pd.Timestamp(h1s)) & (dates <= pd.Timestamp(h1e))
        h2 = (dates >= pd.Timestamp(h2s)) & (dates <= pd.Timestamp(h2e))
        assert (h1 | h2).all()
        assert not (h1 & h2).any()


# ---------------------------------------------------------------------------
# Next-open execution + costs (the core simulator)
# ---------------------------------------------------------------------------

class TestSimulateOpenExec:
    def test_no_trade_days_just_drift(self):
        T, N = 5, 1
        full_ret = np.full((T, N), 0.01)
        overnight = np.zeros((T, N))
        intraday = np.zeros((T, N))
        rf = np.zeros(T)
        out, held, turnover = ev.simulate_open_exec(full_ret, overnight, intraday, rf, {}, {})
        assert np.allclose(out, 0.0)  # never invested (no targets set): all cash, rf=0
        assert np.allclose(turnover, 0.0)

    def test_execution_day_splits_overnight_old_and_intraday_new(self):
        T, N = 3, 2
        full_ret = np.zeros((T, N))
        overnight = np.array([[0.0, 0.0], [0.02, 0.0], [0.0, 0.0]])
        intraday = np.array([[0.0, 0.0], [0.0, 0.03], [0.0, 0.0]])
        rf = np.zeros(T)
        # Start fully in asset 0; at day 1, rotate fully into asset 1.
        targets = {1: np.array([0.0, 1.0])}
        cost_bps = {1: np.array([0.0, 0.0])}
        out, held, turnover = ev.simulate_open_exec(full_ret, overnight, intraday, rf,
                                                     {**{0: np.array([1.0, 0.0])}, **targets}, {
                                                         0: np.array([0.0, 0.0]), **cost_bps})
        # Day 1: overnight leg earns asset-0's 2% (old weights), intraday leg earns
        # asset-1's 3% (new weights) -- zero cost.
        assert out[1] == pytest.approx(1.02 * 1.03 - 1.0)
        assert turnover[1] == pytest.approx(2.0)  # |1-0| + |0-1| after the overnight drift (fully invested)

    def test_cost_is_charged_on_turnover(self):
        T, N = 2, 2
        full_ret = np.zeros((T, N))
        overnight = np.zeros((T, N))
        intraday = np.zeros((T, N))
        rf = np.zeros(T)
        targets = {0: np.array([0.0, 1.0])}
        cost_bps = {0: np.array([0.0, 20.0])}  # 20bps on asset 1's leg
        out, held, turnover = ev.simulate_open_exec(full_ret, overnight, intraday, rf, targets, cost_bps)
        # Starting from all-cash, moving to 100% asset 1: turnover = 1.0, cost = 1.0*20/1e4.
        assert out[0] == pytest.approx(-20.0 / 1e4)
        assert turnover[0] == pytest.approx(1.0)


class TestSimulateCloseExec:
    def test_rebalances_only_when_decide_returns_target(self):
        T, N = 4, 1
        rets = np.array([[0.0], [0.05], [0.0], [0.0]])
        rf = np.zeros(T)

        def decide(i, w):
            return np.array([1.0]) if i == 1 else None
        out, held = ev.simulate_close_exec(rets, rf, decide, lambda i: np.array([0.0]))
        assert held[0, 0] == 0.0  # not yet invested
        assert held[1, 0] == pytest.approx(1.0)  # invested at close of day 1
        assert out[2] == pytest.approx(0.0)  # day 2's own return (0%) with the new weight


# ---------------------------------------------------------------------------
# Placebo mechanics (via a small monkeypatched synthetic universe)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_universe(monkeypatch):
    """Patch the module's 44-ticker UNIVERSE down to 6 synthetic tickers with
    fully controlled inception dates / prices / volumes, so build_targets,
    eligibility_at and placebo_targets can be exercised end to end without any
    real EODHD file."""
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    cal = pd.date_range("2000-01-03", periods=500, freq="B")
    assets = {}
    rng = np.random.default_rng(42)
    for k, t in enumerate(tickers):
        first = cal[10 + 20 * k]
        prices = pd.Series(np.nan, index=cal)
        dom = cal[cal >= first]
        prices.loc[dom] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dom)))
        vol = pd.Series(np.nan, index=cal)
        vol.loc[dom] = 2_000_000.0  # adjusted_close*volume well above ADV floor once 20 bars exist
        adv20 = vol.rolling(20, min_periods=20).median()
        c2c = prices.pct_change()
        assets[t] = ev.Asset(c2c, prices, prices, adv20, first, 0)
    monkeypatch.setattr(ev, "UNIVERSE", tickers)
    monkeypatch.setattr(ev, "IDX", {t: i for i, t in enumerate(tickers)})
    monkeypatch.setattr(ev, "N_UNIV", len(tickers))
    monkeypatch.setattr(prereg, "ELIGIBILITY", {**prereg.ELIGIBILITY, "min_history_calendar_days": 25,
                                               "adv20_floor_usd": 1_000_000, "n_min_for_trading": 2})
    return cal, assets, tickers


class TestEligibilityAndBuildTargets:
    def test_eligibility_grows_as_tickers_season(self, small_universe):
        cal, assets, tickers = small_universe
        early = ev.eligibility_at(assets, cal[15])
        late = ev.eligibility_at(assets, cal[400])
        assert len(early) < len(late)
        assert len(late) == len(tickers)

    def test_build_targets_top_tercile_holds_expected_count(self, small_universe):
        cal, assets, tickers = small_universe
        month_ends = [d for d in cal[ev.month_end_mask(cal)] if d >= cal[100]][:3]
        built = ev.build_targets(assets, cal, month_ends, all_holdings=False,
                                 formation_months=3, skip_months=1, selection="top_tercile")
        for row in built["diag"]:
            # n_hold tracks the POOL with a computable signal (n_signal_ok), which
            # can be smaller than n_eligible if a ticker hasn't seasoned long enough
            # for this variant's own lookback (see prereg_issues issue_1).
            assert row["n_signal_ok"] <= row["n_eligible"]
            assert row["n_hold"] == max(1, round(row["n_signal_ok"] / 3)) or row["n_hold"] == 0

    def test_build_targets_all_holdings_equal_weights_every_eligible_name(self, small_universe):
        cal, assets, tickers = small_universe
        month_ends = [d for d in cal[ev.month_end_mask(cal)] if d >= cal[300]][:2]
        built = ev.build_targets(assets, cal, month_ends, all_holdings=True)
        for pos, w in built["targets"].items():
            n_nonzero = int((w > 0).sum())
            if n_nonzero:
                assert w[w > 0] == pytest.approx(1.0 / n_nonzero)


class TestPlaceboTargets:
    def test_placebo_respects_declared_n_hold_and_eligible_pool(self, small_universe):
        cal, assets, tickers = small_universe
        month_ends = [d for d in cal[ev.month_end_mask(cal)] if d >= cal[300]][:4]
        n_hold_by_month = {str(d.date()): 2 for d in month_ends}
        rng = np.random.default_rng(1)
        pt = ev.placebo_targets(assets, cal, month_ends, n_hold_by_month, rng)
        for pos, w in pt["targets"].items():
            n_nonzero = int((w > 1e-12).sum())
            assert n_nonzero <= 2

    def test_placebo_draws_vary_with_seed(self, small_universe):
        cal, assets, tickers = small_universe
        month_ends = [d for d in cal[ev.month_end_mask(cal)] if d >= cal[300]][:4]
        n_hold_by_month = {str(d.date()): 2 for d in month_ends}
        pt_a = ev.placebo_targets(assets, cal, month_ends, n_hold_by_month, np.random.default_rng(1))
        pt_b = ev.placebo_targets(assets, cal, month_ends, n_hold_by_month, np.random.default_rng(2))
        any_diff = any(not np.allclose(pt_a["targets"][k], pt_b["targets"][k]) for k in pt_a["targets"])
        assert any_diff


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class TestStatsHelpers:
    def test_sharpe_of_constant_positive_return_is_large_positive(self):
        assert ev.sharpe(np.full(300, 0.0005)) > 0

    def test_max_drawdown_of_monotone_series_is_zero(self):
        assert ev.max_drawdown(np.full(100, 0.001)) == pytest.approx(0.0, abs=1e-9)

    def test_cagr_of_doubling_series(self):
        # 252 daily returns compounding to exactly 2x should annualise to ~100%.
        r = np.full(252, 2.0 ** (1 / 252) - 1)
        assert ev.cagr(r) == pytest.approx(1.0, rel=1e-6)

    def test_paired_sharpe_gap_boot_shape_and_determinism(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.001, 0.01, 500)
        b = rng.normal(0.0005, 0.01, 500)
        boot1 = ev.paired_sharpe_gap_boot(a, b, seed=7)
        boot2 = ev.paired_sharpe_gap_boot(a, b, seed=7)
        assert boot1.shape == (prereg.BOOTSTRAP["n_boot"],)
        np.testing.assert_array_equal(boot1, boot2)
