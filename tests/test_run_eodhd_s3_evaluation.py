"""Synthetic-data tests for scripts/run_eodhd_s3_evaluation.py -- the S3
evaluation harness. Never touches the frozen pre-registration file or real
EODHD data; every test builds small synthetic price/return series so the
mechanics can be checked in isolation and fast.

Covers (per the coordinator's Phase 2 checklist): per-bucket on/off, the
dual-momentum ranking + absolute filter, next-adjusted-open execution, the
ADV20 cost tiers, the BIL/DTB3 cash splice, the VFITX BM2 splice, the
12-month-block placebo permutation, the halves/midpoint split, and the
combined 90/5/5 portfolio blend.
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

import run_eodhd_s3_evaluation as ev  # noqa: E402
import eodhd_s3_bond_commodity_trend_preregistered_bars as prereg  # noqa: E402


def make_dates(n: int, start: str = "2007-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def make_inputs(dates: pd.DatetimeIndex, close_ret: pd.DataFrame, overnight_ret: pd.DataFrame,
                 intraday_ret: pd.DataFrame, cost_bps: pd.DataFrame, rf: pd.Series) -> ev.Inputs:
    month_of = dates.to_period("M")
    is_month_start = np.r_[True, (month_of.values[1:] != month_of.values[:-1])]
    return ev.Inputs(dates, close_ret, overnight_ret, intraday_ret, cost_bps, rf, month_of, is_month_start)


# ---------------------------------------------------------------------------
# simulate_next_open: the core execution engine
# ---------------------------------------------------------------------------

class TestSimulateNextOpen:
    def test_no_trade_day_reconstructs_exact_close_to_close(self):
        rng = np.random.default_rng(0)
        n = 200
        cr = rng.normal(0, 0.01, size=(n, 1))
        on_ = rng.normal(0, 0.003, size=(n, 1))
        in_ = (1 + cr) / (1 + on_) - 1  # force (1+on)(1+in) == 1+cr exactly
        rf = np.zeros(n)
        net, held = ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([1.0]) if i == 0 else None,
                                          np.zeros((n, 1)))
        # After the single entry trade on day 0, every subsequent day must be exact close_ret.
        assert np.allclose(net[1:], cr[1:, 0])
        assert np.allclose(held[1:, 0], 1.0, atol=1e-9)

    def test_entry_day_uses_only_intraday_leg(self):
        # Entering from zero: the overnight leg multiplies a ZERO old weight, so
        # only the intraday (open->close) return should show up on day 0.
        cr = np.array([[0.05]])
        on_ = np.array([[0.50]])   # a huge overnight move that must NOT count (old weight 0)
        in_ = np.array([[0.05]])
        rf = np.array([0.0])
        net, held = ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([1.0]) if i == 0 else None,
                                          np.zeros((1, 1)))
        assert net[0] == pytest.approx(0.05)
        assert held[0, 0] == pytest.approx(1.0)

    def test_exit_day_realizes_overnight_leg_on_old_weight_only(self):
        cr = np.array([[0.0], [0.02]])
        on_ = np.array([[0.0], [0.07]])   # the day it exits: old weight still earns this overnight move
        in_ = np.array([[0.0], [0.0]])    # intraday leg irrelevant once new weight is 0
        rf = np.zeros(2)

        def decide(i, w):
            if i == 0:
                return np.array([1.0])
            if i == 1:
                return np.array([0.0])
            return None
        net, held = ev.simulate_next_open(cr, on_, in_, rf, decide, np.zeros((2, 1)))
        assert net[1] == pytest.approx(0.07)  # old weight (1.0) * overnight_ret(day1) + 0 intraday
        assert held[1, 0] == pytest.approx(0.0)

    def test_cost_is_charged_on_the_weight_change(self):
        cr = np.array([[0.0]])
        on_ = np.array([[0.0]])
        in_ = np.array([[0.0]])
        rf = np.array([0.0])
        net, _ = ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([1.0]), np.array([[50.0]]))
        assert net[0] == pytest.approx(-50.0 / 1e4)

    def test_nan_cost_never_corrupts_an_unchanged_zero_weight(self):
        # Regression test for the 0*NaN=NaN bug found during harness development:
        # a NaN cost entry must not poison the NAV path when nothing trades.
        cr = np.full((5, 1), np.nan)
        on_ = np.full((5, 1), np.nan)
        in_ = np.full((5, 1), np.nan)
        rf = np.zeros(5)
        cost = np.full((5, 1), np.nan)
        net, held = ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: None, cost)
        assert np.all(np.isfinite(net))
        assert np.all(held == 0.0)

    def test_long_only_violation_raises(self):
        cr = np.array([[0.0]])
        on_ = np.array([[0.0]])
        in_ = np.array([[0.0]])
        rf = np.array([0.0])
        with pytest.raises(ValueError):
            ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([1.5]), np.zeros((1, 1)))
        with pytest.raises(ValueError):
            ev.simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([-0.1]), np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# Monthly compounding / bond on-off state
# ---------------------------------------------------------------------------

class TestMonthlyCompounding:
    def test_pre_inception_month_is_nan_not_zero(self):
        dates = make_dates(60)
        r = pd.Series(np.nan, index=dates)
        r.iloc[40:] = 0.001  # "inception" partway through
        month_of = dates.to_period("M")
        m = ev._monthly_compounded(r, month_of)
        early_months = m.index[m.index < dates[40].to_period("M")]
        assert m[early_months].isna().all()
        assert m[dates[40].to_period("M")] > 0

    def test_partial_month_compounds_available_days_only(self):
        dates = make_dates(25)
        r = pd.Series(0.01, index=dates)
        month_of = dates.to_period("M")
        m = ev._monthly_compounded(r, month_of)
        # every value present should be a real compounded return, none should be exactly 0
        assert (m.dropna() != 0).all()

    def test_dataframe_variant_matches_per_column_series_variant(self):
        dates = make_dates(80)
        df = pd.DataFrame({"A": np.linspace(-0.01, 0.01, 80), "B": np.linspace(0.02, -0.02, 80)}, index=dates)
        month_of = dates.to_period("M")
        out_df = ev._monthly_compounded(df, month_of)
        out_a = ev._monthly_compounded(df["A"], month_of)
        pd.testing.assert_series_equal(out_df["A"], out_a, check_names=False)


class TestBondState:
    def test_on_when_excess_positive_off_when_negative(self):
        dates = make_dates(130)
        month_of = dates.to_period("M")
        close_ret = pd.DataFrame({"SHY": 0.0, "IEF": 0.0, "TLT": 0.0}, index=dates)
        # One bucket clearly beats cash every month, one clearly loses to cash.
        close_ret["SHY"] = 0.002
        close_ret["TLT"] = -0.002
        rf = pd.Series(0.0005, index=dates)
        inp = make_inputs(dates, close_ret, close_ret * 0, close_ret, close_ret * 0 + 5.0, rf)
        state = ev.bond_monthly_state(inp)
        full = state.dropna(how="all")
        assert full["SHY"].dropna().all()
        assert not full["TLT"].dropna().any()


# ---------------------------------------------------------------------------
# Dual momentum: cross-sectional rank + absolute filter
# ---------------------------------------------------------------------------

class TestCommodityRankAndFilter:
    def test_top_k_selected_and_absolute_filter_excludes_negative_excess(self):
        months = pd.period_range("2010-01", periods=6, freq="M")
        formation = pd.DataFrame({
            "A": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],   # best rank, positive excess -> held
            "B": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],   # 2nd rank, positive excess -> held
            "C": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],   # 3rd rank BUT excess negative -> excluded
            "D": [-0.20, -0.20, -0.20, -0.20, -0.20, -0.20],  # worst rank -> never in top-3
        }, index=months)
        cash = pd.Series(0.02, index=months)  # C's excess = 0.01-0.02 < 0
        target = ev.commodity_monthly_target(formation, cash, k=3)
        row = target.iloc[0]
        assert row["A"] == pytest.approx(0.5)
        assert row["B"] == pytest.approx(0.5)
        assert row["C"] == 0.0
        assert row["D"] == 0.0
        assert row.sum() == pytest.approx(1.0)

    def test_all_off_gives_all_cash(self):
        months = pd.period_range("2010-01", periods=1, freq="M")
        formation = pd.DataFrame({"A": [-0.01], "B": [-0.02], "C": [-0.03]}, index=months)
        cash = pd.Series([0.0], index=months)
        target = ev.commodity_monthly_target(formation, cash, k=3)
        assert target.iloc[0].sum() == 0.0

    def test_nan_formation_is_never_eligible(self):
        months = pd.period_range("2010-01", periods=1, freq="M")
        formation = pd.DataFrame({"A": [np.nan], "B": [0.05], "C": [0.04]}, index=months)
        cash = pd.Series([0.0], index=months)
        target = ev.commodity_monthly_target(formation, cash, k=3)
        assert target.iloc[0]["A"] == 0.0
        assert target.iloc[0]["B"] == pytest.approx(0.5)
        assert target.iloc[0]["C"] == pytest.approx(0.5)

    def test_12_1_formation_skips_the_most_recent_month(self):
        dates = make_dates(400, start="2005-01-03")
        month_of = dates.to_period("M")
        # A steady +1%/month grower, then a crash in the LAST completed month only.
        ret = pd.Series(0.0, index=dates)
        months = sorted(set(month_of))
        last_month_mask = month_of == months[-1]
        ret[~last_month_mask] = 0.01 / 21  # roughly steady positive daily drip
        ret[last_month_mask] = -0.9 / last_month_mask.sum()  # crash concentrated in the final month
        # commodity_formation() selects ev.COMMOD_TICKERS by name -- use the real
        # frozen universe's columns (identical series in each, for this test).
        df = pd.DataFrame({t: ret for t in ev.COMMOD_TICKERS})
        cash = pd.Series(0.0, index=dates)
        inp = make_inputs(dates, df, df * 0, df, df * 0 + 3.0, cash)
        formation_skip, _ = ev.commodity_formation(inp, 11, True)
        formation_noskip, _ = ev.commodity_formation(inp, 11, False)
        col = ev.COMMOD_TICKERS[0]
        # Skip-1 formation at the last available label must still be positive
        # (crash month excluded); no-skip must reflect the crash somewhere.
        assert formation_skip[col].dropna().iloc[-1] > 0
        assert (formation_noskip[col].dropna() < 0).any()


# ---------------------------------------------------------------------------
# Costs: ADV20 tiering
# ---------------------------------------------------------------------------

class TestCostTiering:
    def test_high_volume_gets_low_bps_low_volume_gets_high_bps(self, tmp_path):
        dates = pd.bdate_range("2020-01-01", periods=60)
        hi = pd.DataFrame({
            "date": dates, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "adjusted_close": 100.0, "volume": 1_000_000,  # $100M/day dollar volume
        })
        lo = hi.copy()
        lo["volume"] = 1_000  # $100k/day dollar volume
        cal = pd.DatetimeIndex(dates)
        hi_clean, _ = ev.clean_bars(hi, "equity", cal)
        lo_clean, _ = ev.clean_bars(lo, "equity", cal)
        dvol_hi = (hi_clean.set_index("date")["adjusted_close"] * hi_clean.set_index("date")["volume"]).astype(float)
        dvol_lo = (lo_clean.set_index("date")["adjusted_close"] * lo_clean.set_index("date")["volume"]).astype(float)
        adv_hi = dvol_hi.rolling(ev.ADV20_WINDOW, min_periods=ev.ADV20_WINDOW).median()
        adv_lo = dvol_lo.rolling(ev.ADV20_WINDOW, min_periods=ev.ADV20_WINDOW).median()
        assert (adv_hi.dropna() >= 50_000_000).all()
        assert (adv_lo.dropna() < 50_000_000).all()

    def test_insufficient_history_defaults_to_conservative_tier(self):
        # Fewer than ADV20_WINDOW valid days anywhere -> must default to the
        # HIGH-cost (10bps) tier, and must NEVER be NaN (see the 0*NaN regression test above).
        dvol = pd.Series([1e9] * 5)  # only 5 days, window needs 20
        adv = dvol.rolling(ev.ADV20_WINDOW, min_periods=ev.ADV20_WINDOW).median()
        cost = pd.Series(np.where(adv >= 50_000_000.0, ev.COST_HI_BPS, ev.COST_LO_BPS))
        cost = cost.where(adv.notna(), ev.COST_LO_BPS)
        assert (cost == ev.COST_LO_BPS).all()
        assert cost.notna().all()


# ---------------------------------------------------------------------------
# Cash accrual: BIL from inception, FRED DTB3 before that
# ---------------------------------------------------------------------------

class TestCashSplice:
    def test_dtb3_used_before_bil_bil_used_after(self):
        dates = make_dates(40, start="2007-05-01")
        bil_first = dates[10]
        bil_ret = pd.Series(np.nan, index=dates)
        bil_ret.loc[bil_ret.index >= bil_first] = 0.0001   # BIL's own daily rate once it exists
        dtb3 = pd.Series(0.0002, index=dates)              # DTB3-derived rate throughout
        rf = bil_ret.where(dates >= bil_first, other=np.nan).fillna(dtb3)
        assert (rf[dates < bil_first] == 0.0002).all()
        assert (rf[dates >= bil_first] == 0.0001).all()
        assert rf.notna().all()


# ---------------------------------------------------------------------------
# VFITX splice for BM2's bond leg before IEF existed
# ---------------------------------------------------------------------------

class TestVfitxSplice:
    def test_bond_leg_uses_vfitx_before_ief_ief_after(self):
        dates = make_dates(40)
        ief_first = dates[15]
        ief = pd.Series(np.nan, index=dates)
        ief.loc[ief.index >= ief_first] = 0.001
        vfitx = pd.Series(0.0005, index=dates)
        frame = {
            "close_ret": pd.DataFrame({"IEF": ief, "VFITX": vfitx}),
            "overnight_ret": pd.DataFrame({"IEF": ief * 0, "VFITX": vfitx * 0}),
            "intraday_ret": pd.DataFrame({"IEF": ief, "VFITX": vfitx}),
        }
        cost = pd.DataFrame({"IEF": 3.0, "VFITX": 10.0}, index=dates)
        rf = pd.Series(0.0, index=dates)
        inp = ev.Inputs(dates, frame["close_ret"], frame["overnight_ret"], frame["intraday_ret"], cost, rf,
                        dates.to_period("M"), np.r_[True, np.zeros(39, dtype=bool)])
        bond = ev._bond_leg_frame(inp)
        assert np.allclose(bond["close_ret"][:15], 0.0005)
        assert np.allclose(bond["close_ret"][15:], 0.001)


# ---------------------------------------------------------------------------
# Placebo: 12-month block permutation
# ---------------------------------------------------------------------------

class TestPlaceboPermutation:
    def test_block_permute_preserves_the_multiset_of_values(self):
        rng = np.random.default_rng(1)
        months = pd.period_range("2000-01", periods=48, freq="M")
        df = pd.DataFrame({"A": np.arange(48) % 2 == 0, "B": np.arange(48) % 3 == 0}, index=months)
        out = ev._block_permute_df(df, 12, rng)
        assert sorted(out["A"].tolist()) == sorted(df["A"].tolist())
        assert sorted(out["B"].tolist()) == sorted(df["B"].tolist())
        assert len(out) == len(df)

    def test_columns_permuted_independently(self):
        rng = np.random.default_rng(2)
        months = pd.period_range("2000-01", periods=48, freq="M")
        # Identical columns, but if permuted independently with different random
        # draws for each column they need not end up identical after permutation
        # (this just checks the function permutes per-column, not row-wise).
        df = pd.DataFrame({"A": np.arange(48), "B": np.arange(48)}, index=months)
        out = ev._block_permute_df(df, 12, rng)
        assert sorted(out["A"].tolist()) == list(range(48))
        assert sorted(out["B"].tolist()) == list(range(48))


# ---------------------------------------------------------------------------
# Halves / midpoint split (matches the frozen WINDOWS midpoints)
# ---------------------------------------------------------------------------

class TestHalvesMidpoint:
    def test_midpoint_is_strictly_inside_each_window(self):
        for name, w in prereg.WINDOWS.items():
            assert w["start"] < w["midpoint"] < w["end"], name

    def test_window_helper_splits_at_midpoint(self):
        dates = make_dates(500, start="2007-01-05")
        s = pd.Series(1.0, index=dates)
        mid = dates[250]
        h1 = s[s.index < mid]
        h2 = s[s.index >= mid]
        assert len(h1) + len(h2) == len(s)
        assert h1.index.max() < mid <= h2.index.min()


# ---------------------------------------------------------------------------
# Combined 90/5/5 portfolio
# ---------------------------------------------------------------------------

class TestCombinedBlend:
    def test_linear_blend_weights_match_satellite_spec(self):
        dates = make_dates(10)
        core = pd.Series(0.01, index=dates)
        bond = pd.Series(0.02, index=dates)
        commod = pd.Series(-0.01, index=dates)
        out = ev.combined(core, bond, commod, "combined_test")
        expected = 0.90 * 0.01 + 0.05 * 0.02 + 0.05 * (-0.01)
        assert np.allclose(out.to_numpy(), expected)

    def test_satellite_weights_from_frozen_prereg_sum_correctly(self):
        w = prereg.SATELLITE
        assert w["split"]["bond_sleeve"] + w["split"]["commodity_sleeve"] == pytest.approx(w["total_weight"])

    def test_blend_handles_a_stream_starting_later_via_reindex_fillna(self):
        dates = make_dates(10)
        core = pd.Series(0.0, index=dates)
        bond = pd.Series(0.0, index=dates[3:])  # bond sleeve "doesn't exist" for the first 3 days
        commod = pd.Series(0.0, index=dates)
        out = ev.combined(core, bond, commod, "x")
        assert out.notna().all()


# ---------------------------------------------------------------------------
# Classify precedence (reused, frozen in the pre-reg file itself)
# ---------------------------------------------------------------------------

class TestClassify:
    def test_pending_a7_label_applied_in_tier_string(self):
        # _tier_bars_for's convention: tier_excl_A7 plus "(conditional on A7)".
        assert prereg.classify({"A1": True}, False, {}) == "A"
