from __future__ import annotations

import numpy as np
import pandas as pd

from firm.data.qa import check_spikes


def _rets(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2015-01-01", periods=n))


def test_12_sigma_spike_flagged_5_sigma_not():
    r = _rets()
    sd = r.iloc[300:552].std()
    r.iloc[552] = 12 * sd
    r.iloc[580] = 5 * sd
    f = [x for x in check_spikes(r) if x.severity == "flag"]
    assert [x.date for x in f] == [r.index[552].date()]


def test_negative_spike_flagged():
    r = _rets()
    r.iloc[400] = -12 * r.iloc[148:400].std()
    assert any(x.severity == "flag" and x.date == r.index[400].date() for x in check_spikes(r))


def test_first_252_days_emit_no_flag_only_info():
    r = _rets(n=252)
    r.iloc[100] = 0.5
    f = check_spikes(r)
    assert f and all(x.severity == "info" for x in f)


def test_uses_only_past_returns():
    r = _rets()
    base = [x.date for x in check_spikes(r) if x.severity == "flag"]
    r2 = r.copy()
    r2.iloc[-1] = 1.0  # a future shock must not change an earlier verdict
    assert [x.date for x in check_spikes(r2) if x.severity == "flag" and x.date != r.index[-1].date()] == base


def test_input_not_mutated():
    r = _rets()
    r.iloc[500] = 0.3
    before = r.copy()
    check_spikes(r)
    pd.testing.assert_series_equal(r, before)
