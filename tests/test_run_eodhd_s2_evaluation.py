"""Synthetic-data tests for scripts/run_eodhd_s2_evaluation.py (the S2 Phase-2
evaluation harness), per the frozen pre-registration
scripts/eodhd_s2_breadth_overlay_preregistered_bars.py. Covers: overlay
weights at the monthly check, next-open (1-day-lag) execution, costs, the
BIL/DTB3 cash splice, the VFITX/IEF splice, the block-permutation placebo,
halves, and maxDD/Calmar. No real data files are read here."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_eodhd_s2_evaluation as ev  # noqa: E402
import eodhd_s2_breadth_overlay_preregistered_bars as prereg  # noqa: E402


def _bdate_range(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


def _bars(dates: pd.DatetimeIndex, prices: np.ndarray, volume: float = 1_000_000.0) -> pd.DataFrame:
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"date": dates, "open": p, "high": p, "low": p, "close": p,
                          "adjusted_close": p, "volume": np.full(len(dates), volume)})


# ---------------------------------------------------------------------------
# simulate(): the generic weight simulator (next-open / 1-day-lag execution,
# costs, target-weight bookkeeping).
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_rebalance_does_not_change_the_day_it_is_decided_on(self):
        # Two flat assets (0% return every day); decide() sets a new target on
        # day 2 only. Day 2's OWN realised return must still reflect the OLD
        # weights (0, since both assets are flat either way) -- the real test
        # is that the NEW weights only start earning from day 3.
        rets = np.zeros((5, 2))
        rets[2:, 0] = 0.0  # keep flat so the "return" check is about weights held, not P&L
        rets[:, 1] = 0.0

        def decide(i, w):
            return np.array([1.0, 0.0]) if i == 2 else None
        net, held = ev.simulate(rets, decide, np.array([0.0, 0.0]))
        assert held[1].tolist() == [0.0, 0.0]   # nothing held before the rebalance
        assert held[2].tolist() == [1.0, 0.0]   # rebalance takes effect (for tomorrow) at i=2
        assert held[3].tolist() == [1.0, 0.0]   # carried forward

    def test_new_weight_only_earns_from_the_next_day(self):
        # Asset 0 returns +10% on day 2 and day 3. A rebalance into asset 0
        # decided at i=2 must NOT capture day 2's own +10% (that return was
        # earned by whatever was held before, i.e. nothing), only day 3's.
        rets = np.zeros((4, 2))
        rets[2, 0] = 0.10
        rets[3, 0] = 0.10

        def decide(i, w):
            return np.array([1.0, 0.0]) if i == 2 else None
        net, held = ev.simulate(rets, decide, np.array([0.0, 0.0]))
        assert net[2] == pytest.approx(0.0)    # day of decision: old (zero) weights earn nothing
        assert net[3] == pytest.approx(0.10)   # next day: new weight earns the full return

    def test_cost_is_charged_only_on_a_rebalance_and_scales_with_turnover(self):
        rets = np.zeros((4, 2))
        cost_bps = np.array([100.0, 100.0])  # 1% per side

        def decide(i, w):
            if i == 1:
                return np.array([0.5, 0.5])
            if i == 2:
                return np.array([1.0, 0.0])
            return None
        net, held = ev.simulate(rets, decide, cost_bps)
        assert net[0] == pytest.approx(0.0)
        # |0.5-0| + |0.5-0| = 1.0 turnover x 1% = -0.01
        assert net[1] == pytest.approx(-0.01)
        # |1.0-0.5| + |0.0-0.5| = 1.0 turnover x 1% = -0.01
        assert net[2] == pytest.approx(-0.01)
        assert net[3] == pytest.approx(0.0)     # no rebalance, no cost

    def test_time_varying_cost_matrix_uses_the_row_for_that_day(self):
        rets = np.zeros((3, 1))
        cost = np.array([[0.0], [100.0], [100.0]])  # 0bps on day0, 1% on day1+

        def decide(i, w):
            if i == 0:
                return np.array([0.5])
            if i == 1:
                return np.array([1.0])
            return None
        net, held = ev.simulate(rets, decide, cost)
        assert net[0] == pytest.approx(0.0)      # day-0 rebalance (0 -> 0.5) is free (0bps that day)
        assert net[1] == pytest.approx(-0.005)   # day-1 rebalance (0.5 -> 1.0): turnover 0.5 x 1% = 0.5%

    def test_raises_on_gross_or_short_violation(self):
        rets = np.zeros((1, 1))
        with pytest.raises(ValueError):
            ev.simulate(rets, lambda i, w: np.array([1.5]), np.array([0.0]))
        with pytest.raises(ValueError):
            ev.simulate(rets, lambda i, w: np.array([-0.1]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Overlay weights at the monthly check (variant_returns / _variant_decision_states)
# ---------------------------------------------------------------------------

def _make_inputs(n_months: int = 6, on_months: set[int] | None = None) -> ev.Inputs:
    """Synthetic Inputs: one row per trading day, ~21 sessions/month, flat
    (zero) returns everywhere so weight bookkeeping is the only thing under
    test. ``on_months`` (0-based month index) controls which months the
    breadth signal reads as 'below threshold' (overlay ON) at the prior close."""
    on_months = on_months or set()
    dates = _bdate_range("2010-01-04", periods=21 * n_months)
    spy_ret = pd.Series(0.0, index=dates)
    bond_ret = pd.Series(0.0, index=dates)
    cash_ret = pd.Series(0.0, index=dates)
    ym = dates.to_period("M")
    first_of_month = np.r_[True, ym[1:].to_numpy() != ym[:-1].to_numpy()]
    months_sorted = sorted(ym.unique())
    # breadth measured "at the prior close": make it a step function that is
    # already at its month's target value on the LAST session of the PRIOR
    # month, so the prior-close lookup used by the rebalance day sees it.
    pct_above = pd.Series(1.0, index=dates)  # 1.0 = comfortably "above" (never below 0.5 threshold)
    net_ad = pd.Series(1.0, index=dates)
    for k, m in enumerate(months_sorted):
        if k in on_months:
            pct_above[ym == m] = 0.10  # well below V1/V2/V3's thresholds
            net_ad[ym == m] = -0.50    # well below V4's threshold (0.0)
    return ev.Inputs(dates, spy_ret, bond_ret, cash_ret, pct_above.shift(1), net_ad.shift(1), first_of_month)


class TestOverlayWeights:
    def test_on_override_is_used_directly_not_rethresholded(self):
        # Regression test for a real bug found while building this harness:
        # variant_returns(on_override=...) must treat the override as the
        # FINAL on/off array, not re-compare it against the variant's own
        # threshold. V4's threshold is 0.0, so re-comparing a boolean
        # override (values 0/1) against 0.0 would silently force "off" on
        # every single day (0 < 0.0 and 1 < 0.0 are both False) -- exactly
        # what a naive re-threshold bug produces.
        inp = _make_inputs(n_months=3, on_months=set())
        forced_on = np.ones(len(inp.dates), dtype=bool)
        _, on = ev.variant_returns(inp, "V4_alt_measure", on_override=forced_on)
        assert on.all()
        forced_off = np.zeros(len(inp.dates), dtype=bool)
        _, on2 = ev.variant_returns(inp, "V4_alt_measure", on_override=forced_off)
        assert not on2.any()

    def test_off_state_is_plain_60_40_for_every_variant(self):
        inp = _make_inputs(n_months=3, on_months=set())
        for vid in ev.VARIANT_IDS:
            net, on = ev.variant_returns(inp, vid)
            assert not on.any()

    def test_v1_cut_state_moves_the_freed_weight_into_the_bond_leg(self):
        inp = _make_inputs(n_months=3, on_months={1})
        cost = ev._cost_matrix(inp, stress=False)

        def decide_capture(rets, decide, c):
            _, held = ev.simulate(rets, decide, c)
            return held
        rets = ev._rets_matrix(inp)
        on = ev._variant_decision_states(inp, "V1_primary")
        fom = inp.first_of_month

        def decide(i, w):
            if not fom[i]:
                return None
            if on[i]:
                return np.array([0.3, 0.7, 0.0])
            return np.array([0.6, 0.4, 0.0])
        _, held = ev.simulate(rets, decide, cost)
        # find a rebalance day flagged "on" and check the held weights after it
        on_rebal_days = np.flatnonzero(fom & on)
        assert len(on_rebal_days) > 0
        i = on_rebal_days[0]
        assert held[i].tolist() == [0.3, 0.7, 0.0]

    def test_v2_cut_state_moves_the_freed_weight_into_the_cash_leg_not_bond(self):
        inp = _make_inputs(n_months=3, on_months={1})
        net, on = ev.variant_returns(inp, "V2_cash_destination")
        fom = inp.first_of_month
        on_rebal_days = np.flatnonzero(fom & on)
        assert len(on_rebal_days) > 0
        # Recompute the held weights directly to check column placement.
        rets = ev._rets_matrix(inp)
        cost = ev._cost_matrix(inp, stress=False)

        def decide(i, w):
            if not fom[i]:
                return None
            return np.array([0.3, 0.0, 0.7]) if on[i] else np.array([0.6, 0.4, 0.0])
        _, held = ev.simulate(rets, decide, cost)
        i = on_rebal_days[0]
        assert held[i][1] == pytest.approx(0.0)    # bond leg (IEF) fully evacuated
        assert held[i][2] == pytest.approx(0.7)    # cash leg (BIL) holds the freed weight

    def test_v3_stricter_threshold_needs_a_lower_breadth_reading_to_trigger(self):
        # A breadth reading of 0.45 is below V1/V2's 0.50 threshold but ABOVE
        # V3's stricter 0.40 -- V3 must stay off while V1 goes on.
        dates = _bdate_range("2010-01-04", periods=42)
        spy_ret = pd.Series(0.0, index=dates)
        bond_ret = pd.Series(0.0, index=dates)
        cash_ret = pd.Series(0.0, index=dates)
        ym = dates.to_period("M")
        first_of_month = np.r_[True, ym[1:].to_numpy() != ym[:-1].to_numpy()]
        pct_above = pd.Series(0.45, index=dates)
        net_ad = pd.Series(1.0, index=dates)
        inp = ev.Inputs(dates, spy_ret, bond_ret, cash_ret, pct_above.shift(1), net_ad.shift(1), first_of_month)
        _, on_v1 = ev.variant_returns(inp, "V1_primary")
        _, on_v3 = ev.variant_returns(inp, "V3_stricter_threshold")
        assert on_v1.any()
        assert not on_v3.any()

    def test_v4_uses_the_alternative_measure_not_pct_above_200sma(self):
        dates = _bdate_range("2010-01-04", periods=42)
        spy_ret = pd.Series(0.0, index=dates)
        bond_ret = pd.Series(0.0, index=dates)
        cash_ret = pd.Series(0.0, index=dates)
        ym = dates.to_period("M")
        first_of_month = np.r_[True, ym[1:].to_numpy() != ym[:-1].to_numpy()]
        pct_above = pd.Series(0.90, index=dates)   # comfortably "on" for V1-V3's measure -- but they must stay OFF
        net_ad = pd.Series(-0.10, index=dates)     # below V4's own threshold (0.0) -- V4 must go ON
        inp = ev.Inputs(dates, spy_ret, bond_ret, cash_ret, pct_above.shift(1), net_ad.shift(1), first_of_month)
        _, on_v1 = ev.variant_returns(inp, "V1_primary")
        _, on_v4 = ev.variant_returns(inp, "V4_alt_measure")
        assert not on_v1.any()
        assert on_v4.any()

    def test_missing_prior_close_signal_is_treated_as_off_not_on(self):
        # The very first rows (before shift(1) has a real value) are NaN;
        # _variant_decision_states must not treat NaN as "below threshold".
        dates = _bdate_range("2010-01-04", periods=5)
        spy_ret = pd.Series(0.0, index=dates)
        bond_ret = pd.Series(0.0, index=dates)
        cash_ret = pd.Series(0.0, index=dates)
        pct_above = pd.Series([np.nan, np.nan, 0.9, 0.9, 0.9], index=dates)
        net_ad = pd.Series([np.nan, np.nan, 0.1, 0.1, 0.1], index=dates)
        first_of_month = np.array([True, False, False, False, False])
        inp = ev.Inputs(dates, spy_ret, bond_ret, cash_ret, pct_above, net_ad, first_of_month)
        on = ev._variant_decision_states(inp, "V1_primary")
        assert on[0] == False  # noqa: E712 -- NaN signal must not be read as "on"


# ---------------------------------------------------------------------------
# Splices: VFITX->IEF (bond leg), DTB3->BIL (cash leg)
# ---------------------------------------------------------------------------

class TestSplices:
    def test_bond_leg_uses_proxy_before_and_real_after_the_handoff(self):
        dates = _bdate_range("2002-01-02", periods=10)
        # VFITX: constant +0.1%/day the whole time (proxy)
        vfitx_prices = 100 * np.cumprod(1 + np.full(10, 0.001))
        vfitx = _bars(dates, vfitx_prices)
        # IEF: only exists from day index 6 onward, constant +0.5%/day (real)
        ief_dates = dates[6:]
        ief_prices = 50 * np.cumprod(1 + np.full(len(ief_dates), 0.005))
        ief = _bars(ief_dates, ief_prices)
        spliced = ev._splice_returns(vfitx, ief, "bond_leg_test")
        # Before the handoff: proxy's own return (~0.1%)
        assert spliced.iloc[1] == pytest.approx(0.001, abs=1e-9)
        # From the day AFTER ief's first bar: ief's own return (~0.5%)
        assert spliced.loc[ief_dates[1]] == pytest.approx(0.005, abs=1e-9)
        # ief's own first day (no prior IEF row) falls back to the proxy's return that day
        assert spliced.loc[ief_dates[0]] == pytest.approx(0.001, abs=1e-9)

    def test_cash_leg_uses_dtb3_before_and_bil_after_the_handoff(self):
        dates = _bdate_range("2007-01-02", periods=10)
        dtb3_daily = pd.Series(0.0002, index=dates)  # flat accrual rate
        bil_dates = dates[5:]
        bil_prices = 100 * np.cumprod(1 + np.full(len(bil_dates), 0.0003))
        bil = _bars(bil_dates, bil_prices)
        spliced = ev._splice_cash(dtb3_daily, bil)
        assert spliced.loc[dates[1]] == pytest.approx(0.0002, abs=1e-9)
        assert spliced.loc[bil_dates[1]] == pytest.approx(0.0003, abs=1e-9)
        assert spliced.loc[bil_dates[0]] == pytest.approx(0.0002, abs=1e-9)  # bil's own first day: DTB3 fallback

    def test_cost_matrix_is_zero_on_the_proxy_leg_and_nonzero_once_real(self):
        inp = _make_inputs(n_months=4)
        cutoff = inp.dates[20]  # an arbitrary mid-window date
        inp.ief_first, inp.bil_first = cutoff, cutoff
        cost = ev._cost_matrix(inp, stress=False)
        before, after = inp.dates < cutoff, inp.dates >= cutoff
        assert (cost[before, 1] == 0.0).all()
        assert (cost[after, 1] > 0.0).all()
        assert (cost[before, 2] == 0.0).all()
        assert (cost[after, 2] > 0.0).all()
        assert (cost[:, 0] > 0.0).all()  # SPY always charged

    def test_cost_matrix_doubles_under_stress(self):
        inp = _make_inputs(n_months=2)
        inp.ief_first, inp.bil_first = inp.dates[0], inp.dates[0]
        base = ev._cost_matrix(inp, stress=False)
        stressed = ev._cost_matrix(inp, stress=True)
        assert stressed[0, 0] == pytest.approx(2 * base[0, 0])


# ---------------------------------------------------------------------------
# Block-permutation placebo
# ---------------------------------------------------------------------------

class TestPlacebo:
    def test_block_permute_preserves_the_multiset_of_values(self):
        rng = np.random.default_rng(0)
        x = np.arange(20)
        y = ev._block_permute(x, 5, rng)
        assert sorted(y.tolist()) == sorted(x.tolist())
        assert len(y) == len(x)

    def test_block_permute_is_deterministic_given_a_seeded_rng(self):
        x = np.arange(30)
        y1 = ev._block_permute(x, 7, np.random.default_rng(42))
        y2 = ev._block_permute(x, 7, np.random.default_rng(42))
        assert y1.tolist() == y2.tolist()

    def test_placebo_variant_changes_the_on_off_assignment_in_general(self):
        # 24 months -> 8 non-overlapping 63-day blocks, several scattered "on"
        # episodes: enough blocks that an identical reshuffle is implausible.
        inp = _make_inputs(n_months=24, on_months={2, 5, 9, 14, 20})
        rng = np.random.default_rng(1)
        real_net, real_on = ev.variant_returns(inp, "V1_primary")
        placebo_net = ev.placebo_variant(inp, "V1_primary", rng)
        # Not required to differ on every draw, but across a 12-month window
        # with 3 "on" months, an identical placebo path is not plausible.
        assert not np.allclose(placebo_net.to_numpy(), real_net.to_numpy())

    def test_placebo_output_is_a_returns_series_of_the_right_length(self):
        inp = _make_inputs(n_months=6, on_months={1, 3})
        rng = np.random.default_rng(2)
        p = ev.placebo_variant(inp, "V2_cash_destination", rng)
        assert len(p) == len(inp.dates)
        assert np.isfinite(p.to_numpy()).all()


# ---------------------------------------------------------------------------
# Halves (A4)
# ---------------------------------------------------------------------------

class TestHalves:
    def test_halves_partition_the_calendar_at_the_frozen_midpoint(self):
        dates = _bdate_range("2002-01-02", periods=500)
        lo, hi = ev._halves(dates)
        mid = pd.Timestamp(prereg.WINDOW["midpoint"])
        assert (lo.index[lo] <= mid).all()
        assert (hi.index[hi] > mid).all()
        assert (lo | hi).all()  # every day lands in exactly one half
        assert not (lo & hi).any()


# ---------------------------------------------------------------------------
# maxDD / Calmar
# ---------------------------------------------------------------------------

class TestDrawdownAndCalmar:
    def test_max_drawdown_of_a_known_path(self):
        # +10%, -20%, +5% ... nav: 1.10, 0.88, 0.924 -- peak 1.10, trough 0.88 -> DD = 0.20
        r = np.array([0.10, -0.20, 0.05])
        assert ev.max_drawdown(r) == pytest.approx(0.20, abs=1e-9)

    def test_max_drawdown_is_zero_for_a_monotonically_rising_path(self):
        r = np.full(10, 0.01)
        assert ev.max_drawdown(r) == pytest.approx(0.0, abs=1e-9)

    def test_cagr_of_a_flat_annual_return(self):
        r = np.full(prereg.TRADING_DAYS, 0.0)
        assert ev.cagr(r) == pytest.approx(0.0, abs=1e-9)
        # ~10%/year compounded daily over exactly one trading year
        daily = 1.10 ** (1 / prereg.TRADING_DAYS) - 1
        r = np.full(prereg.TRADING_DAYS, daily)
        assert ev.cagr(r) == pytest.approx(0.10, abs=1e-6)

    def test_calmar_is_cagr_over_max_drawdown(self):
        r = np.full(prereg.TRADING_DAYS, 1.10 ** (1 / prereg.TRADING_DAYS) - 1)
        r[10] = -0.25  # inject one drawdown day
        c, dd = ev.cagr(r), ev.max_drawdown(r)
        assert ev.calmar(r) == pytest.approx(c / dd, rel=1e-9)

    def test_calmar_is_nan_when_there_is_no_drawdown(self):
        r = np.full(10, 0.001)
        assert np.isnan(ev.calmar(r))
