from __future__ import annotations

import pandas as pd

from firm.data.qa import check_fx_conversion


def _fx(n=20):
    idx = pd.bdate_range("2024-01-01", periods=n)
    fx = pd.Series(3.6 + 0.01 * pd.Series(range(n), index=idx), index=idx)
    local = pd.Series(100.0 + pd.Series(range(n), index=idx), index=idx)
    return local, fx, local * fx


def test_correct_conversion_passes():
    local, fx, usd = _fx()
    assert check_fx_conversion(local, fx, usd) == []


def test_mismatched_conversion_flagged():
    local, fx, usd = _fx()
    usd.iloc[7] *= 1.01
    f = check_fx_conversion(local, fx, usd)
    assert [x.severity for x in f] == ["flag"] and f[0].date == usd.index[7].date()


def test_nonpositive_fx_flagged():
    local, fx, _ = _fx()
    fx.iloc[3] = 0.0
    f = check_fx_conversion(local, fx, local * fx)
    assert any(x.severity == "flag" and "positive" in x.detail for x in f)


def test_fx_daily_jump_over_10pct_flagged():
    local, fx, _ = _fx()
    fx.iloc[10:] *= 1.12
    f = check_fx_conversion(local, fx, local * fx)
    assert [x.date for x in f] == [fx.index[10].date()]
