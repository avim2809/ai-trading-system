from __future__ import annotations

import datetime as dt

import pandas as pd

from firm.data.qa import check_negative_prices


def _crude():
    idx = pd.to_datetime(["2020-04-17", "2020-04-20", "2020-04-21"])
    return pd.DataFrame({"close": [18.27, -37.63, 10.01]}, index=idx.rename("date"))


def test_crude_negative_is_info_for_futures_when_allow_listed():
    f = check_negative_prices(_crude(), asset="futures", symbol="CL")
    assert len(f) == 1 and f[0].severity == "info" and f[0].date == dt.date(2020, 4, 20) and "-37.63" in f[0].detail


def test_same_fixture_is_flag_for_equity_and_etf_and_crypto():
    for a in ("equity", "etf", "crypto"):
        f = check_negative_prices(_crude(), asset=a, symbol="CL")
        assert [x.severity for x in f] == ["flag"]


def test_unlisted_negative_futures_price_is_warn():
    f = check_negative_prices(_crude(), asset="futures", symbol="ZZ")
    assert [x.severity for x in f] == ["warn"]


def test_zero_price_is_flagged_and_positive_series_clean():
    b = pd.DataFrame({"close": [1.0, 0.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=3))
    assert [x.severity for x in check_negative_prices(b, asset="etf")] == ["flag"]
    assert check_negative_prices(b.assign(close=[1.0, 1.5, 2.0]), asset="etf") == []
