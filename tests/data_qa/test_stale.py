from __future__ import annotations

import pandas as pd

from firm.data.qa import check_stale


def _bars(run: int, vol_const=True):
    n = 30
    close = [100.0 + i for i in range(n)]
    for k in range(10, 10 + run):
        close[k] = 150.0
    vol = [1e6] * n if vol_const else [1e6 + i for i in range(n)]
    return pd.DataFrame({"close": close, "volume": vol}, index=pd.bdate_range("2020-01-01", periods=n).rename("date"))


def test_six_identical_closes_flagged():
    f = check_stale(_bars(6))
    assert len(f) == 1 and f[0].severity == "flag" and f[0].check == "stale"


def test_five_identical_closes_not_flagged():
    assert check_stale(_bars(5)) == []


def test_identical_close_with_moving_volume_is_warn():
    f = check_stale(_bars(6, vol_const=False))
    assert [x.severity for x in f] == ["warn"]


def test_max_repeat_days_param():
    assert len(check_stale(_bars(4), max_repeat_days=3)) == 1


def test_input_not_mutated():
    b = _bars(6)
    before = b.copy()
    check_stale(b)
    pd.testing.assert_frame_equal(b, before)
