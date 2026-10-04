"""Missing-day QA against the NYSE calendar plus special closures (P2-03). Synthetic fixtures only."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from firm.data.qa import SPECIAL_CLOSURES, check_missing_days, expected_trading_days


def _full(start: str, end: str, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    idx = expected_trading_days(dt.date.fromisoformat(start), dt.date.fromisoformat(end))
    idx = idx.drop(pd.to_datetime(list(drop)), errors="ignore")
    return pd.DataFrame({"close": 1.0}, index=idx.rename("date"))


def test_special_closure_table_is_complete_and_cited():
    expected = {dt.date(1994, 4, 27), *(dt.date(2001, 9, d) for d in (11, 12, 13, 14)), dt.date(2004, 6, 11),
                dt.date(2007, 1, 2), dt.date(2012, 10, 29), dt.date(2012, 10, 30), dt.date(2018, 12, 5),
                dt.date(2025, 1, 9)}
    assert set(SPECIAL_CLOSURES) == expected
    assert all(isinstance(v, str) and len(v) > 10 for v in SPECIAL_CLOSURES.values())  # citation text per date


def test_expected_days_skip_closures_and_holidays():
    d = expected_trading_days(dt.date(2001, 9, 10), dt.date(2001, 9, 18))
    assert [x.day for x in d] == [10, 17, 18]
    assert pd.Timestamp("1994-04-27") not in expected_trading_days(dt.date(1994, 4, 1), dt.date(1994, 4, 30))
    assert pd.Timestamp("1997-01-20") in expected_trading_days(dt.date(1997, 1, 1), dt.date(1997, 1, 31))  # MLK not a holiday pre-1998


@pytest.mark.parametrize("a,b", [("2001-09-04", "2001-09-28"), ("2012-10-22", "2012-11-05"), ("1994-04-20", "1994-05-05")])
def test_special_closures_accepted(a, b):
    assert check_missing_days(_full(a, b), calendar="nyse") == []


def test_missing_day_after_closure_is_flagged():
    f = check_missing_days(_full("2001-09-04", "2001-09-28", drop=("2001-09-17",)))
    assert [x.severity for x in f] == ["flag"]
    assert f[0].date == dt.date(2001, 9, 17)


def test_random_missing_wednesday_flagged():
    f = check_missing_days(_full("2015-03-02", "2015-03-31", drop=("2015-03-18",)))
    assert len(f) == 1 and f[0].severity == "flag" and "1" in f[0].detail


def test_mlk_1997_absence_flagged():
    f = check_missing_days(_full("1997-01-06", "1997-02-07", drop=("1997-01-20",)))
    assert len(f) == 1 and f[0].date == dt.date(1997, 1, 20)


def test_detail_lists_first_ten_and_count():
    f = check_missing_days(_full("2015-03-02", "2015-05-29", drop=tuple(str(x.date()) for x in pd.bdate_range("2015-04-06", "2015-04-24"))))
    assert "15" in f[0].detail and f[0].detail.count("2015-04") <= 10


def test_dates_outside_series_range_are_not_missing():
    # inception/delisting: only the span between first and last bar is compared
    assert check_missing_days(_full("2015-03-02", "2015-03-10")) == []


def test_accepts_date_column():
    df = _full("2015-03-02", "2015-03-13").reset_index()
    assert check_missing_days(df) == []
