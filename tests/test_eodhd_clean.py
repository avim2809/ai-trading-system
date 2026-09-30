"""Frozen EODHD bar-cleaning rule (scripts/eodhd_clean.py) on synthetic bars."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_clean as ec  # noqa: E402

FROZEN_FP = "f62cb2e4a139ea1d3cf240ce938f1d6f573d6208a9e2c004cedee66a47d07ccd"  # v2; a shortlist prereg cites this value


def _bars(prices, start="2015-12-21", vol=1000.0, freq="B"):
    dates = pd.date_range(start, periods=len(prices), freq=freq)
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"date": dates, "open": p, "close": p, "adjusted_close": p, "volume": vol})


def test_fingerprint_is_frozen():
    # Changing any rule changes this; a shortlist prereg cites the value.
    assert ec.CLEANING_RULES["version"] == 2
    assert ec.cleaning_fingerprint() == FROZEN_FP


def test_phantom_holiday_bar_is_dropped():
    # SMLP-style: a zero-volume $0.0002 bar on a market holiday between ~$18 bars.
    d = _bars([18.0, 18.1, 18.3, 18.35, 0.0002, 18.26, 17.8], freq="D")
    d.loc[4, "volume"] = 0
    cal = pd.DatetimeIndex([x for x in d["date"] if x != d["date"].iloc[4]])
    out, rep = ec.clean_bars(d, "equity", cal)
    assert rep["zero_volume"] == 1 and len(out) == 6
    assert (out["adjusted_close"] > 1).all()


def test_off_calendar_bar_is_dropped_for_equities_not_crypto():
    d = _bars([10.0] * 5, freq="D")
    cal = pd.DatetimeIndex(d["date"].iloc[[0, 1, 3, 4]])
    _, rep = ec.clean_bars(d, "equity", cal)
    assert rep["off_calendar"] == 1
    _, rep_c = ec.clean_bars(d, "crypto")
    assert "off_calendar" not in rep_c and rep_c["n_out"] == 5


def test_spike_that_reverts_is_dropped():
    # XBKS-style scale error for two bars, then back.
    d = _bars([4.5, 4.6, 1150.0, 1149.0, 4.55, 4.5])
    out, rep = ec.clean_bars(d, "equity", pd.DatetimeIndex(d["date"]))
    assert rep["spike_reversal"] == 2
    a = out["adjusted_close"].to_numpy()
    assert np.max(np.abs(a[1:] / a[:-1] - 1)) < 0.05


def test_unadjusted_reverse_split_starts_a_new_segment():
    # CERN-style x3.93 jump that stays: a segment break, bars kept.
    d = _bars([9.6, 9.8, 9.7, 38.2, 38.0, 37.6, 37.8, 38.1])
    out, rep = ec.clean_bars(d, "equity", pd.DatetimeIndex(d["date"]))
    assert rep["segment_breaks"] == 1 and rep["n_out"] == 8
    assert list(out["segment"]) == [0, 0, 0, 1, 1, 1, 1, 1]


def test_real_crash_is_kept_and_normal_moves_untouched():
    d = _bars([50.0, 51.0, 12.0, 11.5, 11.8, 12.2, 12.0, 11.9])   # -76% and no recovery
    out, rep = ec.clean_bars(d, "equity", pd.DatetimeIndex(d["date"]))
    assert rep["n_out"] == 8 and rep["segment_breaks"] == 0 and rep["spike_reversal"] == 0
    rng = np.random.default_rng(0)
    calm = _bars(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)))
    _, rep = ec.clean_bars(calm, "equity", pd.DatetimeIndex(calm["date"]))
    assert rep["n_out"] == 300


def test_nav_funds_keep_zero_volume_bars_but_obey_the_calendar():
    d = _bars([10.0, 10.02, 10.01, 10.05, 10.04], vol=0.0, freq="D")
    cal = pd.DatetimeIndex(d["date"].iloc[[0, 1, 2, 4]])
    out, rep = ec.clean_bars(d, "nav", cal)
    assert rep["zero_volume"] == 0 and rep["off_calendar"] == 1 and len(out) == 4
    _, rep_eq = ec.clean_bars(d, "equity", cal)
    assert rep_eq["n_out"] == 0


def test_bad_prices_and_inputs():
    d = _bars([10.0, -1.0, 10.1, np.nan, 10.2])
    _, rep = ec.clean_bars(d, "equity", pd.DatetimeIndex(d["date"]))
    assert rep["price"] == 2
    with pytest.raises(ValueError):
        ec.clean_bars(d, "equity")
    with pytest.raises(ValueError):
        ec.clean_bars(d, "fx")
