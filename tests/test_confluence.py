"""Tests for firm.patterns.confluence.

Covers weekly resampling correctness (including the look-ahead guard that
drops the most recent, potentially still-forming week), the trend-direction
read on synthetic weekly series, and the confluence-modifier mapping for
every direction/trend combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm.patterns.confluence import (
    confluence_modifier,
    resample_to_weekly,
    weekly_trend_direction,
)

# ---------------------------------------------------------------------------
# resample_to_weekly
# ---------------------------------------------------------------------------


def _business_day_fixture(n: int, start: str = "2024-01-01"):
    """`n` business days starting on a Monday, with strictly increasing
    close/high/low so each week's aggregate values are trivially derivable
    (week close = last day's close, week high/low = last day's high/low
    since the series is monotonic).
    """
    dates = pd.bdate_range(start=start, periods=n)
    close = np.arange(1, n + 1, dtype=float)
    high = close + 1.0
    low = close - 1.0
    volume = np.full(n, 1_000.0)
    return dates, high, low, close, volume


def test_resample_to_weekly_drops_partial_trailing_week():
    # 43 business days starting Monday 2024-01-01 -> exactly 8 complete
    # Mon-Fri weeks (40 days) plus Mon/Tue/Wed (3 days) of a 9th, partial
    # week that must NOT appear in the output.
    dates, high, low, close, volume = _business_day_fixture(43)
    weekly = resample_to_weekly(dates, high, low, close, volume)

    assert len(weekly) == 8
    # Week 1's Friday is day index 4 (0-based) -> close 5.0.
    assert weekly.iloc[0]["close"] == pytest.approx(5.0)
    # Week 8's Friday is day index 39 (0-based) -> close 40.0. The partial
    # week's days (41, 42, 43) must never leak into the output.
    assert weekly.iloc[-1]["close"] == pytest.approx(40.0)
    assert 41.0 not in weekly["close"].to_numpy()
    assert 42.0 not in weekly["close"].to_numpy()
    assert 43.0 not in weekly["close"].to_numpy()


def test_resample_to_weekly_ohlc_aggregation_correctness():
    # Single complete week (Mon-Fri) plus one partial day dropped by the
    # look-ahead guard.
    dates, high, low, close, volume = _business_day_fixture(6)  # 5 complete + 1 partial
    weekly = resample_to_weekly(dates, high, low, close, volume)

    assert len(weekly) == 1
    row = weekly.iloc[0]
    assert row["high"] == pytest.approx(6.0)  # day5 high = close(5.0)+1
    assert row["low"] == pytest.approx(0.0)  # day1 low = close(1.0)-1
    assert row["close"] == pytest.approx(5.0)  # last day of the week
    assert row["volume"] == pytest.approx(5 * 1_000.0)  # summed


def test_resample_to_weekly_no_complete_weeks_returns_empty():
    # Only 3 business days -- not even one full Mon-Fri week yet.
    dates, high, low, close, volume = _business_day_fixture(3)
    weekly = resample_to_weekly(dates, high, low, close, volume)
    assert weekly.empty


def test_resample_to_weekly_empty_input_returns_empty():
    empty = np.array([])
    weekly = resample_to_weekly(pd.Series([], dtype="datetime64[ns]"), empty, empty, empty, empty)
    assert weekly.empty


def test_resample_to_weekly_accepts_plain_numpy_datetime_array():
    dates, high, low, close, volume = _business_day_fixture(11)  # 2 complete weeks + 1 partial day
    weekly = resample_to_weekly(np.asarray(dates.to_numpy()), high, low, close, volume)
    assert len(weekly) == 2


# ---------------------------------------------------------------------------
# weekly_trend_direction
# ---------------------------------------------------------------------------


def _weekly_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-07", periods=len(closes), freq="W-SUN")
    return pd.DataFrame(
        {
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000.0] * len(closes),
        },
        index=idx,
    )


def test_weekly_trend_direction_up():
    closes = [100.0 + i * 3.0 for i in range(10)]  # clearly, steadily rising
    weekly = _weekly_df(closes)
    assert weekly_trend_direction(weekly, lookback_weeks=8) == "up"


def test_weekly_trend_direction_down():
    closes = [200.0 - i * 3.0 for i in range(10)]  # clearly, steadily falling
    weekly = _weekly_df(closes)
    assert weekly_trend_direction(weekly, lookback_weeks=8) == "down"


def test_weekly_trend_direction_flat():
    rng = np.random.default_rng(0)
    closes = [100.0 + rng.uniform(-0.05, 0.05) for _ in range(10)]  # sub-threshold noise only
    weekly = _weekly_df(closes)
    assert weekly_trend_direction(weekly, lookback_weeks=8) == "flat"


def test_weekly_trend_direction_insufficient_history_defaults_flat():
    closes = [100.0, 200.0, 50.0]  # only 3 rows, lookback_weeks default is 8
    weekly = _weekly_df(closes)
    assert weekly_trend_direction(weekly) == "flat"


def test_weekly_trend_direction_empty_defaults_flat():
    empty = pd.DataFrame(columns=["high", "low", "close", "volume"])
    assert weekly_trend_direction(empty) == "flat"


def test_weekly_trend_direction_exactly_at_threshold_boundary():
    # lookback_weeks=1: exactly +1% change -> "up" ("up" is >= threshold).
    weekly = _weekly_df([100.0, 101.0])
    assert weekly_trend_direction(weekly, lookback_weeks=1) == "up"
    weekly_down = _weekly_df([100.0, 99.0])
    assert weekly_trend_direction(weekly_down, lookback_weeks=1) == "down"


# ---------------------------------------------------------------------------
# confluence_modifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern_direction,weekly_trend,expected",
    [
        ("long", "up", 1.0),
        ("long", "down", -1.0),
        ("long", "flat", 0.0),
        ("short", "down", 1.0),
        ("short", "up", -1.0),
        ("short", "flat", 0.0),
    ],
)
def test_confluence_modifier_all_combinations(pattern_direction, weekly_trend, expected):
    assert confluence_modifier(pattern_direction, weekly_trend) == expected


def test_confluence_modifier_unrecognized_pattern_direction_defaults_neutral():
    assert confluence_modifier("sideways", "up") == 0.0
    assert confluence_modifier("sideways", "down") == 0.0
    assert confluence_modifier("sideways", "flat") == 0.0
