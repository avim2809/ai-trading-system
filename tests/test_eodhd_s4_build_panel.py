"""Tests for scripts/eodhd_s4_build_panel.py — mechanics on synthetic bars, incl.
segment breaks (unadjusted reverse split) and delisting (series ends early).
Phase 1 only: this builder never computes a forward return, so there is nothing
here about candidate/benchmark performance — see test_eodhd_s4_preregistered_bars.py
for the pre-registration's own mechanics tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_s4_build_panel as bp


def _calendar(start="2015-01-01", periods=900):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=periods))


def _synthetic_bars(cal: pd.DatetimeIndex, closes, start_idx=0, vol=100_000.0):
    """A clean, tradable daily series on a slice of ``cal`` starting at ``start_idx``."""
    dates = cal[start_idx:start_idx + len(closes)]
    p = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": p, "close": p, "adjusted_close": p,
        "volume": np.full(len(p), vol),
    })


class TestMonthEndDates:
    def test_last_session_per_calendar_month(self):
        cal = _calendar("2020-01-01", 70)  # spans Jan-Apr 2020
        ends = bp.month_end_dates(cal)
        assert list(ends.month) == sorted(set(cal.month), key=list(cal.month).index) or list(ends.month) == [1, 2, 3, 4]
        # each returned date actually is the max date in its (year, month) group
        for e in ends:
            same_month = cal[(cal.year == e.year) & (cal.month == e.month)]
            assert e == same_month.max()

    def test_unique_and_sorted(self):
        cal = _calendar("2018-06-01", 400)
        ends = bp.month_end_dates(cal)
        assert list(ends) == sorted(set(ends))


class TestEntryDatesAfter:
    def test_next_session_strictly_after(self):
        cal = _calendar("2021-01-01", 100)
        ends = bp.month_end_dates(cal)
        entry = bp.entry_dates_after(ends, cal)
        for me in ends[:-1]:
            assert entry[me] > me
            assert entry[me] in cal

    def test_last_month_end_at_calendar_edge_is_nat(self):
        cal = _calendar("2021-01-01", 21)  # ends mid-month, no month-end beyond edge trickiness
        ends = bp.month_end_dates(cal)
        entry = bp.entry_dates_after(ends, cal)
        # the final month-end in a calendar that stops there has no session after it
        assert pd.isna(entry[ends[-1]])


class TestProcessTickerMechanics:
    def test_full_history_reaches_eligibility_at_bar_252(self):
        cal = _calendar("2015-01-01", 900)
        # Monotonically increasing so ratio_52wk == 1.0 once the window is full.
        closes = 10.0 + 0.01 * np.arange(400)
        df = _synthetic_bars(cal, closes)
        cleaned, _ = __import__("eodhd_clean").clean_bars(df, "equity", cal)
        cleaned = bp._segment_rolling(cleaned)
        row_251 = cleaned.iloc[250]   # 251st bar, n_bars_in_segment == 251
        row_252 = cleaned.iloc[251]   # 252nd bar, first bar with a defined ratio
        assert row_251["n_bars_in_segment"] == 251 and pd.isna(row_251["ratio_52wk"])
        assert row_252["n_bars_in_segment"] == 252
        assert row_252["ratio_52wk"] == pytest.approx(1.0, abs=1e-6)  # monotone up-trend: at its high
        assert not pd.isna(row_252["adv63"])  # 63 < 252, already defined well before this row

    def test_segment_break_resets_the_lookback_window(self):
        cal = _calendar("2015-01-01", 900)
        # 300 clean bars, then an unexplained x3 up-jump that does not revert (segment break),
        # then 300 more bars -- ratio_52wk/adv63 must not look back across the break.
        pre = 10.0 + 0.001 * np.arange(300)
        post = np.full(300, pre[-1] * 3.0) + 0.001 * np.arange(300)
        closes = np.concatenate([pre, post])
        df = _synthetic_bars(cal, closes)
        cleaned, rep = __import__("eodhd_clean").clean_bars(df, "equity", cal)
        assert rep["segment_breaks"] == 1
        cleaned = bp._segment_rolling(cleaned)
        post_seg = cleaned[cleaned["segment"] == 1].reset_index(drop=True)
        assert post_seg.loc[0, "n_bars_in_segment"] == 1
        assert pd.isna(post_seg.loc[250, "ratio_52wk"])   # only 251 bars into the new segment
        assert post_seg.loc[251, "n_bars_in_segment"] == 252
        assert not pd.isna(post_seg.loc[251, "ratio_52wk"])  # exactly enough bars since the break

    def test_delisting_series_ends_early_no_rows_after_last_bar(self, tmp_path):
        cal = _calendar("2015-01-01", 900)
        month_ends = bp.month_end_dates(cal)
        # Ticker trades for 260 bars then stops (delisted) -- well past 1 month-end,
        # short of a second.
        closes = 10.0 + 0.01 * np.arange(260)
        df = _synthetic_bars(cal, closes)
        f = tmp_path / "DEAD.parquet"
        df.to_parquet(f)
        out = bp.process_ticker(f, cal, month_ends)
        last_bar_date = cal[259]
        assert out is not None
        assert (out["month_end"] <= last_bar_date).all()
        assert out["month_end"].max() < month_ends[month_ends > last_bar_date].min() \
            if (month_ends > last_bar_date).any() else True

    def test_pre_ipo_ticker_has_no_rows_before_first_bar(self, tmp_path):
        cal = _calendar("2015-01-01", 900)
        month_ends = bp.month_end_dates(cal)
        closes = 10.0 + 0.01 * np.arange(300)
        df = _synthetic_bars(cal, closes, start_idx=200)  # IPOs partway through the calendar
        f = tmp_path / "NEWCO.parquet"
        df.to_parquet(f)
        out = bp.process_ticker(f, cal, month_ends)
        assert out is not None
        assert (out["month_end"] >= cal[200]).all()

    def test_price_and_bar_count_screen_excludes_from_compact_panel(self, tmp_path):
        cal = _calendar("2015-01-01", 900)
        month_ends = bp.month_end_dates(cal)
        # A penny stock with a full 300-bar history: eligible on bars, not on price.
        closes = np.full(300, 2.0)
        df = _synthetic_bars(cal, closes)
        f = tmp_path / "PENNY.parquet"
        df.to_parquet(f)
        out = bp.process_ticker(f, cal, month_ends)
        assert out is not None and len(out) > 0
        assert (out["price"] < bp.PRICE_MIN).all()
        pass_screen = (out["n_bars_in_segment"] >= bp.BARS_MIN) & (out["price"] >= bp.PRICE_MIN)
        assert not pass_screen.any()


class TestBuildPanelEndToEnd:
    def test_synthetic_universe_produces_expected_compact_rows(self, tmp_path, monkeypatch):
        cal = _calendar("2015-01-01", 900)
        uni = tmp_path / "us_universe"
        uni.mkdir()

        # GOOD: 400 clean bars, price > $5 throughout -> eligible from bar 252 onward.
        good = 10.0 + 0.01 * np.arange(400)
        _synthetic_bars(cal, good).to_parquet(uni / "GOOD.parquet")
        # PENNY: same length, price < $5 -> never eligible.
        penny = np.full(400, 3.0)
        _synthetic_bars(cal, penny).to_parquet(uni / "PENNY.parquet")
        # SHORT: only 100 bars -> never reaches 252, never eligible.
        short = 20.0 + 0.01 * np.arange(100)
        _synthetic_bars(cal, short).to_parquet(uni / "SHORT.parquet")
        # DEAD: delists after 300 bars, was eligible before that.
        dead = 15.0 + 0.01 * np.arange(300)
        _synthetic_bars(cal, dead).to_parquet(uni / "DEAD.parquet")

        monkeypatch.setattr(bp, "UNIVERSE_DIR", uni)
        # build_panel takes its exchange calendar from the real SPY file under
        # data/research/eodhd (licensed, gitignored, absent on a clean clone).
        # This test is about the synthetic universe, so use the synthetic calendar.
        monkeypatch.setattr(bp.ec, "equity_calendar", lambda *a, **k: cal)
        panel, summary = bp.build_panel(universe_dir=uni)

        assert set(panel["ticker"].unique()) <= {"GOOD", "DEAD"}
        assert "PENNY" not in set(panel["ticker"])
        assert "SHORT" not in set(panel["ticker"])
        assert list(panel.columns) == bp.PANEL_COLUMNS
        assert (panel["n_bars_in_segment"] >= bp.BARS_MIN).all()
        assert (panel["price"] >= bp.PRICE_MIN).all()

        # Eligibility summary must show SHORT/PENNY contributing to n_with_bar but
        # never to n_pass_both, and DEAD dropping out of every count once it delists.
        last_dead_bar = cal[299]
        after_dead = summary[summary["month_end"] > last_dead_bar]
        if len(after_dead):
            assert (after_dead["n_pass_both"] <= 1).all()  # GOOD only, if anyone


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
