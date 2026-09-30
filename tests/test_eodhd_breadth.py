"""Synthetic-data tests for scripts/eodhd_breadth.py (the S2 breadth-signal
builder). Every price path below is hand-constructed so eligible_count,
pct_above_200sma and net_ad_ratio can be checked by arithmetic, not just
shape -- including the segment-reset rule (a ticker must re-accumulate
SMA_WINDOW days of history AFTER a cleaning-rule segment break before it
counts as eligible again). No return series is built or touched anywhere
in this file, consistent with the module under test.
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

import eodhd_breadth as eb  # noqa: E402

SMA_WINDOW = eb.UNIVERSE_SCREEN["sma_window"]          # 200
PRICE_MIN = eb.UNIVERSE_SCREEN["price_min_usd"]        # 5.0
ADV_MIN = eb.UNIVERSE_SCREEN["adv20_min_usd"]          # 1_000_000.0


def _frame(calendar: pd.DatetimeIndex, prices: np.ndarray, volume: float | np.ndarray) -> pd.DataFrame:
    n = len(calendar)
    vol = np.full(n, volume, dtype=float) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    return pd.DataFrame({
        "date": calendar, "open": prices, "close": prices, "adjusted_close": prices, "volume": vol,
    })


@pytest.fixture()
def calendar() -> pd.DatetimeIndex:
    # 320 business days: enough for a 200-day SMA to form, plus room for a
    # post-break segment that deliberately does NOT reach 200 days.
    return pd.bdate_range("2015-01-02", periods=320)


@pytest.fixture()
def universe_dir(tmp_path: Path, calendar: pd.DatetimeIndex) -> Path:
    d = tmp_path / "us_universe"
    d.mkdir()
    n = len(calendar)

    # A: flat $50 for 200 sessions, then a real (non-reverting, <150%) rise to
    # $60 that stays -- ample volume. Becomes eligible exactly at session 200
    # (0-based index 199) and is ABOVE its 200sma from session 201 (index 200)
    # once the new $60 print pulls the average up past itself... actually the
    # average is still ~$50.05 at that point, so $60 > sma: "above".
    a = np.concatenate([np.full(200, 50.0), np.full(n - 200, 60.0)])
    _frame(calendar, a, 1_000_000.0).to_parquet(d / "AAAA.parquet")

    # B: flat $50 for 200 sessions, then a real fall to $40 that stays.
    # Eligible at the same session as A, but BELOW its 200sma once the drop
    # happens (the average is still ~$49.95, current price $40 is below it).
    b = np.concatenate([np.full(200, 50.0), np.full(n - 200, 40.0)])
    _frame(calendar, b, 1_000_000.0).to_parquet(d / "BBBB.parquet")

    # C: fails the price floor forever ($2, well under PRICE_MIN) despite
    # ample volume and 320 days of flat history.
    c = np.full(n, 2.0)
    _frame(calendar, c, 1_000_000.0).to_parquet(d / "CCCC.parquet")

    # D: fails the ADV20 floor forever (dollar volume ~$5,000/day, well under
    # ADV_MIN) despite a normal $50 price and 320 days of flat history.
    dd = np.full(n, 50.0)
    _frame(calendar, dd, 100.0).to_parquet(d / "DDDD.parquet")

    # F: flat $10 for 250 sessions (would be eligible from session 200 on),
    # then a one-day +200% jump to $30 that does NOT revert -- a segment
    # break per eodhd_clean's own rule (> +150%, no reversion within 5 bars).
    # Only n - 250 sessions follow the break, deliberately < SMA_WINDOW, so F
    # must NOT be counted as eligible again post-break in this fixture.
    f = np.concatenate([np.full(250, 10.0), np.full(n - 250, 30.0)])
    _frame(calendar, f, 1_000_000.0).to_parquet(d / "FFFF.parquet")

    assert n - 250 < SMA_WINDOW, "fixture must keep F's post-break segment short of a full SMA window"
    return d


def test_ineligible_before_sma_window_fills(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame.set_index("date")
    day_198 = calendar[198]  # n_in_seg = 199 for A/B -- one short of SMA_WINDOW
    assert frame.loc[day_198, "eligible_count"] == 0


def test_eligible_count_and_breadth_at_the_known_day(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame.set_index("date")

    # A, B and F are all still flat ($50, $50, $10) and all reach n_in_seg=200
    # at the same session (F's own break is 50 sessions further out) -- so
    # all three are eligible here, all flat vs. their own SMA (none "above").
    day_199 = calendar[199]
    row = frame.loc[day_199]
    assert row["eligible_count"] == 3
    assert row["pct_above_200sma"] == 0.0

    day_200 = calendar[200]  # the day after A rose to $60 and B fell to $40; F is still flat at $10
    row = frame.loc[day_200]
    assert row["eligible_count"] == 3
    assert row["pct_above_200sma"] == pytest.approx(1 / 3)  # only A is above its sma
    assert row["net_ad_ratio"] == pytest.approx(0.0)        # 1 advance (A) - 1 decline (B), F flat (neither)


def test_price_floor_excludes_cheap_stock_forever(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame
    # A, B and F (while still in their original segment) can all be eligible
    # at once -- 3 is the true ceiling given this fixture. If C's $2 price
    # were NOT being excluded by the floor, the ceiling would be 4.
    assert frame["eligible_count"].max() == 3


def test_adv_floor_excludes_illiquid_stock_forever(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame.set_index("date")
    last = calendar[-1]
    # By the last day F has dropped back out post-break (see the segment-reset
    # test below), leaving only A and B. If D's ~$5,000/day dollar volume were
    # NOT being excluded by the ADV floor, this would be 3, not 2.
    assert frame.loc[last, "eligible_count"] == 2


def test_segment_break_resets_the_sma_window(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame.set_index("date")

    day_249 = calendar[249]  # F's last day in its original ($10) segment, n_in_seg = 250: eligible
    # F alone would add a 3rd eligible name here if segment resets didn't apply;
    # confirm A and B are also eligible by now so this isn't a screen-wide zero.
    assert frame.loc[day_249, "eligible_count"] == 3

    last = calendar[-1]  # F's post-break segment is (320 - 250) = 70 days: short of SMA_WINDOW (200)
    assert frame.loc[last, "eligible_count"] == 2  # F has dropped back out; only A and B remain


def test_net_ad_ratio_and_21d_average_are_bounded(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    frame = res.frame
    valid = frame["net_ad_ratio"].dropna()
    assert ((valid >= -1.0) & (valid <= 1.0)).all()
    valid21 = frame["net_ad_21d"].dropna()
    assert ((valid21 >= -1.0) & (valid21 <= 1.0)).all()


def test_skipped_and_used_counts_are_consistent(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar)
    assert res.n_tickers_scanned == 5  # AAAA, BBBB, CCCC, DDDD, FFFF
    assert res.n_tickers_used == 5     # all 5 produce >=1 kept bar after cleaning; none are unreadable
    assert res.skipped == []


def test_limit_caps_the_number_of_files_scanned(universe_dir, calendar):
    res = eb.build_breadth(universe_dir=universe_dir, calendar=calendar, limit=2)
    assert res.n_tickers_scanned == 2
