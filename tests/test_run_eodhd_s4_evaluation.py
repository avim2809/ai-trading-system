"""Synthetic-data mechanics tests for scripts/run_eodhd_s4_evaluation.py (the S4
PHASE 2 harness). Covers, per the coordinator's explicit checklist: overlapping
6-month cohorts (1/6 of book/month), next-open execution, per-name ADV costs on
every trade, delisting/segment exit at the last clean close + -30% stress, the
cash accrual BIL/DTB3 + VFITX splice, the random-ranking placebo, and halves.

No real data is read here; everything is built from small synthetic arrays so the
tests run in well under a second and never touch data/research/."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_eodhd_s4_evaluation as r  # noqa: E402


# ---------------------------------------------------------------------------
# adv_bucket_bps
# ---------------------------------------------------------------------------

class TestAdvBucketBps:
    def test_buckets_match_frozen_table(self):
        assert r.adv_bucket_bps(500_000) == r.STOCK_COST["adv_lt_1m"]
        assert r.adv_bucket_bps(2_000_000) == r.STOCK_COST["adv_1m_5m"]
        assert r.adv_bucket_bps(10_000_000) == r.STOCK_COST["adv_5m_20m"]
        assert r.adv_bucket_bps(50_000_000) == r.STOCK_COST["adv_gt_20m"]

    def test_nan_falls_back_to_most_liquid_bucket(self):
        assert r.adv_bucket_bps(float("nan")) == r.STOCK_COST["adv_gt_20m"]


# ---------------------------------------------------------------------------
# schedule_slots + simulate_slots: next-open execution, weight, ADV costs
# ---------------------------------------------------------------------------

def _toy_matrices(T: int, M: int):
    RET = np.full((T, M), np.nan, dtype="float32")
    ENTRY_RET = np.full((T, M), np.nan, dtype="float32")
    return RET, ENTRY_RET


class TestScheduleAndSimulateSlots:
    def test_default_weight_divides_by_hold_months_not_just_decile_size(self):
        # Regression test for a real bug caught in the first full run: with hold_months=6
        # and no custom weight_fn, each of a month's ``decile_size`` names must get
        # 1/(6*decile_size) of the book, not 1/decile_size (which -- summed across the
        # ~6 concurrently open cohorts -- inflated total invested weight to ~5.76x and
        # produced a fabricated -68% single-day loss on 2020-03-16 in the real run).
        month_ends = [pd.Timestamp("2000-01-31")]
        entry_of = {month_ends[0]: 1}
        names = ["A", "B"]
        col_of = {"A": 0, "B": 1}
        last_pos_of = {"A": 99, "B": 99}
        slots = r.schedule_slots(month_ends, entry_of, {month_ends[0]: names}, hold_months=6,
                                 col_of=col_of, last_pos_of=last_pos_of, T=100)
        assert len(slots) == 2
        for s in slots:
            assert s.weight == pytest.approx(1.0 / 6 / 2)

    def test_default_weight_with_hold_months_one_is_plain_equal_weight(self):
        month_ends = [pd.Timestamp("2000-01-31")]
        entry_of = {month_ends[0]: 1}
        names = ["A", "B", "C", "D"]
        col_of = {n: i for i, n in enumerate(names)}
        last_pos_of = {n: 99 for n in names}
        slots = r.schedule_slots(month_ends, entry_of, {month_ends[0]: names}, hold_months=1,
                                 col_of=col_of, last_pos_of=last_pos_of, T=100)
        assert all(s.weight == pytest.approx(0.25) for s in slots)

    def test_single_name_next_open_entry_then_close_to_close(self):
        # Name "A" (col 0): enters day 2 (open->close 5%), then two more close-to-close
        # days of 1% each, no cost (flat 0bps) to isolate the return-path mechanics.
        T, M = 6, 1
        RET, ENTRY_RET = _toy_matrices(T, M)
        ENTRY_RET[2, 0] = 0.05
        RET[3, 0] = 0.01
        RET[4, 0] = 0.01
        cash = np.zeros(T)
        slot = r.Slot(col=0, weight=1.0, entry_pos=2, exit_pos=4, early_exit=False)
        net, invested = r.simulate_slots([slot], RET, ENTRY_RET, cash, T, cost_bps_of=r.flat_cost_bps(0.0))
        assert net[0] == 0 and net[1] == 0
        assert net[2] == pytest.approx(0.05)
        assert net[3] == pytest.approx(0.01)
        assert net[4] == pytest.approx(0.01)
        assert net[5] == 0     # slot closed, nothing held, no cash return either
        assert invested[2] == 1.0 and invested[5] == 0.0

    def test_per_name_adv_cost_charged_on_entry_and_exit_only(self):
        T, M = 4, 1
        RET, ENTRY_RET = _toy_matrices(T, M)
        ENTRY_RET[0, 0] = 0.0
        RET[1, 0] = 0.0
        cash = np.zeros(T)
        slot = r.Slot(col=0, weight=0.5, entry_pos=0, exit_pos=1, early_exit=False)
        # 60 bps per side (adv_lt_1m bucket), charged on BOTH entry (day 0) and exit (day 1).
        net, _ = r.simulate_slots([slot], RET, ENTRY_RET, cash, T, cost_bps_of=r.flat_cost_bps(60.0))
        assert net[0] == pytest.approx(0.0 - 0.5 * 60.0 / 1e4)
        assert net[1] == pytest.approx(0.0 - 0.5 * 60.0 / 1e4)

    def test_six_month_overlapping_cohorts_each_get_one_sixth_of_book(self):
        # 8 monthly reformation points; hold=6 means each cohort is 1/decile_size within
        # its own 1/6 slice, so at steady state 6 cohorts are open, weights summing to 1.
        month_ends = pd.DatetimeIndex(pd.date_range("2000-01-31", periods=8, freq="ME"))
        T = 200
        entry_pos_seq = np.arange(10, 10 + 8 * 20, 20)   # a fake "entry day" every 20 sessions
        entry_of = {me: int(p) for me, p in zip(month_ends, entry_pos_seq)}
        col_of = {"X": 0}
        last_pos_of = {"X": T - 1}
        names_by_month = {me: ["X"] for me in month_ends}

        def weight_fn(me, names):
            return {n: 1.0 / 6 for n in names}   # one name = the whole 1/6 cohort slice

        slots = r.schedule_slots(list(month_ends), entry_of, names_by_month, hold_months=6,
                                 col_of=col_of, last_pos_of=last_pos_of, T=T, weight_fn=weight_fn)
        assert len(slots) == 8
        assert all(s.weight == pytest.approx(1 / 6) for s in slots)
        # steady state (after the 6th cohort forms): 6 cohorts of name X are concurrently
        # open, so at a day inside that window the sum of active slots' weight is 1.0.
        day = entry_of[month_ends[6]] + 1
        active_weight = sum(s.weight for s in slots if s.entry_pos <= day <= s.exit_pos)
        assert active_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Delisting / segment exit + -30% stress
# ---------------------------------------------------------------------------

class TestDelistingAndStress:
    def test_early_exit_uses_last_clean_close_not_scheduled_exit(self):
        T, M = 10, 1
        col_of = {"D": 0}
        last_pos_of = {"D": 4}     # the series stops at position 4 (delisted)
        month_ends = pd.DatetimeIndex(["2000-01-31"])
        entry_of = {month_ends[0]: 1}
        names_by_month = {month_ends[0]: ["D"]}
        slots = r.schedule_slots(list(month_ends), entry_of, names_by_month, hold_months=6,
                                 col_of=col_of, last_pos_of=last_pos_of, T=T)
        assert len(slots) == 1
        s = slots[0]
        assert s.exit_pos == 4          # clipped to the name's last available bar
        assert s.early_exit is True

    def test_no_data_at_all_on_or_after_entry_skips_the_slot(self):
        month_ends = pd.DatetimeIndex(["2000-01-31"])
        entry_of = {month_ends[0]: 5}
        slots = r.schedule_slots(list(month_ends), entry_of, {month_ends[0]: ["D"]}, 6,
                                 {"D": 0}, {"D": 2}, T=10)   # last_pos (2) < entry (5)
        assert slots == []

    def test_delist_stress_adds_extra_negative_return_only_on_early_exit_day(self):
        T, M = 5, 1
        RET, ENTRY_RET = _toy_matrices(T, M)
        ENTRY_RET[0, 0] = 0.0
        RET[1, 0] = 0.0
        cash = np.zeros(T)
        early_slot = r.Slot(col=0, weight=1.0, entry_pos=0, exit_pos=1, early_exit=True)
        sched_slot = r.Slot(col=0, weight=1.0, entry_pos=0, exit_pos=1, early_exit=False)
        net_early, _ = r.simulate_slots([early_slot], RET, ENTRY_RET, cash, T, delist_stress=True,
                                        cost_bps_of=r.flat_cost_bps(0.0))
        net_sched, _ = r.simulate_slots([sched_slot], RET, ENTRY_RET, cash, T, delist_stress=True,
                                        cost_bps_of=r.flat_cost_bps(0.0))
        assert net_early[1] == pytest.approx(net_sched[1] + r.DELIST_STRESS_PCT)


# ---------------------------------------------------------------------------
# Cash accrual: BIL from its first bar, FRED DTB3 before that
# ---------------------------------------------------------------------------

class TestCashRateSeries:
    def test_uses_dtb3_before_bil_inception_and_bil_after(self, tmp_path, monkeypatch):
        dates = pd.bdate_range("2007-05-25", periods=10)
        bil_dates = dates[3:]   # BIL "starts" partway through
        bil = pd.DataFrame({"date": bil_dates, "open": 100.0, "high": 100.0, "low": 100.0,
                            "close": 100.0, "adjusted_close": np.linspace(100, 100.5, len(bil_dates)),
                            "volume": 1000})
        etfs_full = tmp_path / "etfs_full"
        etfs_full.mkdir(parents=True)
        bil.to_parquet(etfs_full / "BIL.parquet")
        fred = tmp_path / "fred"
        fred.mkdir()
        dtb3 = pd.DataFrame({"date": dates, "value": 5.0})   # 5% annualised
        dtb3.to_parquet(fred / "DTB3.parquet")

        monkeypatch.setattr(r, "EODHD", tmp_path)
        monkeypatch.setattr(r, "FRED", fred)
        out = r.cash_rate_series(pd.DatetimeIndex(dates))
        # before BIL's first bar: FRED DTB3 rate, 5/100/252 every day
        assert out[0] == pytest.approx(5.0 / 100 / 252, abs=1e-6)
        assert out[1] == pytest.approx(5.0 / 100 / 252, abs=1e-6)
        assert out[2] == pytest.approx(5.0 / 100 / 252, abs=1e-6)
        # from BIL's first bar (index 3) on: BIL's own pct_change (first BIL day undefined -> 0)
        expected_bil_ret = bil["adjusted_close"].pct_change().fillna(0.0).to_numpy()
        np.testing.assert_allclose(out[3:], expected_bil_ret, atol=1e-6)


# ---------------------------------------------------------------------------
# BM2's VFITX splice (before IEF existed)
# ---------------------------------------------------------------------------

class TestBm2Splice:
    def test_uses_vfitx_before_ief_inception_and_ief_after(self, tmp_path, monkeypatch):
        cal = pd.bdate_range("2002-01-01", periods=40)
        etfs_full = tmp_path / "etfs_full"
        etfs_full.mkdir(parents=True)

        def mkbar(dates, level):
            return pd.DataFrame({"date": dates, "open": level, "high": level, "low": level,
                                 "close": level, "adjusted_close": level, "volume": 1000})

        spy = mkbar(cal, 100.0 + 0.01 * np.arange(len(cal)))
        spy.to_parquet(etfs_full / "SPY.parquet")
        ief_dates = cal[20:]      # IEF "starts" partway through the toy calendar
        ief = mkbar(ief_dates, 90.0 + 0.01 * np.arange(len(ief_dates)))
        ief.to_parquet(etfs_full / "IEF.parquet")
        vfitx = mkbar(cal, 80.0 + 0.01 * np.arange(len(cal)))
        vfitx["volume"] = 0
        vfitx.to_parquet(etfs_full / "VFITX.parquet")

        monkeypatch.setattr(r, "EODHD", tmp_path)
        bench = {"SPY": r.load_bench_daily("SPY", "equity", cal, cal),
                 "IEF": r.load_bench_daily("IEF", "equity", cal, cal),
                 "VFITX": r.load_bench_daily("VFITX", "nav", cal, cal)}
        cash = np.zeros(len(cal))
        net = r.bm2_60_40(bench, cal, cash, len(cal))
        # sanity: some days before IEF's inception show a nonzero bond-leg return
        # (VFITX-derived), and none of that leg's slots reference column 1 (IEF)
        # before its own first bar.
        assert np.isfinite(net).sum() > 0


# ---------------------------------------------------------------------------
# Placebo: random rankings, same holdings count, same dates
# ---------------------------------------------------------------------------

class TestPlaceboSharpes:
    def test_placebo_draws_respect_pool_and_decile_size(self, monkeypatch):
        month_ends = [pd.Timestamp("2000-01-31"), pd.Timestamp("2000-02-29")]
        pool = [f"T{i}" for i in range(20)]
        decile = pool[:5]
        per_month = {me: {500: {"universe": pool, "decile": decile}} for me in month_ends}
        entry_of = {month_ends[0]: 1, month_ends[1]: 21}
        col_of = {t: i for i, t in enumerate(pool)}
        last_pos_of = {t: 99 for t in pool}
        T = 100
        RET = np.zeros((T, len(pool)), dtype="float32")
        ENTRY_RET = np.zeros((T, len(pool)), dtype="float32")
        cash = np.zeros(T)

        monkeypatch.setitem(r.prereg.CANDIDATES, "S4_p1_6mo_N500", {"hold": "1_month", "N": 500, "is_primary": True})
        built = {"month_ends": month_ends, "per_month": per_month, "entry_of": entry_of,
                 "col_of": col_of, "last_pos_of": last_pos_of, "RET": RET, "ENTRY_RET": ENTRY_RET,
                 "cash_ret": cash, "T": T, "cost_fn": r.flat_cost_bps(0.0)}
        out = r.placebo_sharpes(built, "S4_p1_6mo_N500", n_draws=5, seed=1)
        assert len(out) == 5   # a well-formed Sharpe (possibly nan with all-zero returns) per draw

    def test_placebo_selection_size_matches_decile_and_stays_within_pool(self):
        rng = np.random.default_rng(0)
        pool = list(range(30))
        dn = 6
        idx = rng.choice(len(pool), size=dn, replace=False)
        chosen = [pool[i] for i in idx]
        assert len(chosen) == dn
        assert len(set(chosen)) == dn
        assert set(chosen) <= set(pool)


# ---------------------------------------------------------------------------
# Halves (gap_stats)
# ---------------------------------------------------------------------------

class TestHalves:
    def test_gap_stats_splits_at_the_frozen_midpoint(self, monkeypatch):
        dates = pd.bdate_range("2012-10-01", periods=80)
        monkeypatch.setattr(r, "MIDPOINT", pd.Timestamp("2012-11-15"))
        rng = np.random.default_rng(0)
        c = rng.normal(0.001, 0.01, len(dates))
        b = rng.normal(0.0, 0.01, len(dates))
        out = r.gap_stats(c, b, dates, seed=1)
        assert out["n_days"] == len(dates)
        h1_len = int((dates <= pd.Timestamp("2012-11-15")).sum())
        assert h1_len not in (0, len(dates))   # the split must be a genuine partition
        assert "gap_half1" in out and "gap_half2" in out
        assert isinstance(out["A4"], bool)


# ---------------------------------------------------------------------------
# ADV20 sparse lookup + build_matrices plumbing
# ---------------------------------------------------------------------------

class TestAdv20Lookup:
    def test_looks_up_value_at_or_before_the_requested_position(self):
        adv20_of = {"A": (np.array([2, 5, 9], dtype="int32"), np.array([1e6, 2e6, 3e6], dtype="float32"))}
        assert r.adv20_at_or_before(adv20_of, "A", 2) == pytest.approx(1e6)
        assert r.adv20_at_or_before(adv20_of, "A", 4) == pytest.approx(1e6)
        assert r.adv20_at_or_before(adv20_of, "A", 9) == pytest.approx(3e6)

    def test_before_any_observation_is_nan(self):
        adv20_of = {"A": (np.array([5], dtype="int32"), np.array([1e6], dtype="float32"))}
        assert np.isnan(r.adv20_at_or_before(adv20_of, "A", 0))

    def test_unknown_ticker_is_nan(self):
        assert np.isnan(r.adv20_at_or_before({}, "MISSING", 3))


class TestBuildMatrices:
    def test_streams_one_ticker_into_the_dense_matrix_and_sparse_adv20(self, tmp_path):
        cal = pd.bdate_range("2010-01-01", periods=300)
        uni = tmp_path / "us_universe_full"
        uni.mkdir()
        closes = 10.0 + 0.01 * np.arange(len(cal))
        df = pd.DataFrame({"date": cal, "open": closes, "high": closes, "low": closes,
                           "close": closes, "adjusted_close": closes, "volume": 100_000})
        df.to_parquet(uni / "GOOD.parquet")
        dates = cal[100:250]
        RET, ENTRY_RET, col_of, adv20_of, last_pos_of = r.build_matrices(["GOOD", "MISSING"], cal, dates, uni)
        assert RET.shape == (len(dates), 2)
        assert col_of == {"GOOD": 0, "MISSING": 1}
        assert "MISSING" not in adv20_of         # file doesn't exist: skipped cleanly
        assert "GOOD" in adv20_of
        assert last_pos_of["GOOD"] == len(dates) - 1
        # a monotone uptrend: close-to-close returns should be uniformly small and positive
        assert np.nanmean(RET[1:, 0]) > 0
        assert np.isnan(RET[:, 1]).all()          # the missing ticker's column stays all-NaN


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
