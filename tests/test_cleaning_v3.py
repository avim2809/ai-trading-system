"""EODHD cleaning v3 (src/firm/data/cleaning.py) on synthetic bars only."""

from __future__ import annotations

import importlib.util
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.data.cleaning import (
    CLEANING_RULES_V3,
    SEAL_DATE,
    clean_bars_v3,
    cleaning_fingerprint_v3,
    equity_calendar_v3,
)
from tests.test_eodhd_clean import _bars

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FP_V3 = "2deb690edb2db0bd4eb64c06e61bc25a37d6534da718dcc07472083e6cb0c0af"  # written at the OD-14 freeze; changing any v3 rule changes it
V2_KEYS = ("n_in", "price", "zero_volume", "off_calendar", "spike_reversal", "segment_breaks", "n_out")


def _v2():
    spec = importlib.util.spec_from_file_location("eodhd_clean_v2", ROOT / "scripts" / "eodhd_clean.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cal(d):
    return pd.DatetimeIndex(d["date"])


def _wiggle(n, base=50.0, seed=0):
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(0, 0.005, n)))


def test_fingerprint_v3_is_frozen():
    assert CLEANING_RULES_V3["version"] == 3
    assert cleaning_fingerprint_v3() == FROZEN_FP_V3
    assert cleaning_fingerprint_v3() != _v2().cleaning_fingerprint()


def test_sentinel_bars_dropped():
    p = _wiggle(200)
    p[100:130] = 999999.9999
    d = _bars(p)
    out, rep = clean_bars_v3(d, "equity", _cal(d))
    assert rep["sentinel"] == 30 and len(out) == 170
    assert rep["segment_breaks"] == 0 and rep["spike_reversal"] == 0 and rep["constant_run"] == 0
    assert (out["adjusted_close"] < 1e3).all()


def test_sentinel_in_single_column_dropped_with_tolerance():
    p = _wiggle(50)
    d = _bars(p)
    d.loc[10, "open"] = 999999.9999005
    d["high"] = d["close"]
    d.loc[20, "high"] = 999999.9999
    out, rep = clean_bars_v3(d, "equity", _cal(d))
    assert rep["sentinel"] == 2 and len(out) == 48


def test_constant_sentinel_series_not_a_return():
    p = _wiggle(300)
    p[150:200] = 999999.9999
    d = _bars(p)
    out, _ = clean_bars_v3(d, "equity", _cal(d))
    r = out["adjusted_close"].pct_change().dropna()
    assert r.abs().max() < 1.0


def test_constant_run_rule():
    p = _wiggle(200)
    p[50:55] = 123456.0  # run of 5 identical >= 1e5 on a ~50 asset
    p[100:104] = 123456.0  # run of 4: kept
    d = _bars(p)
    _, rep = clean_bars_v3(d, "equity", _cal(d))
    assert rep["constant_run"] == 5 and rep["sentinel"] == 0
    # median >= 1e4: rule not applicable (e.g. a genuinely high-priced asset)
    d2 = _bars(np.full(100, 200000.0))
    _, rep2 = clean_bars_v3(d2, "equity", _cal(d2))
    assert rep2["constant_run"] == 0


def test_v3_equals_v2_without_sentinels():
    ec = _v2()
    cases = [
        (_bars([18.0, 18.1, 18.3, 18.35, 0.0002, 18.26, 17.8], freq="D"), "equity"),
        (_bars([10, 10.2, 25.0, 10.1, 10.3, 10.4]), "equity"),  # spike reversal
        (_bars([10, 10.1, 39.3, 39.0, 38.5, 39.9, 40.0]), "equity"),  # segment break
        (_bars([10, 10.1, 4.0, 4.1, 4.2]), "equity"),  # down-jump kept
        (_bars(_wiggle(120, seed=3)), "equity"),
        (_bars(_wiggle(120, 3.0, seed=4), freq="D"), "crypto"),
        (_bars(_wiggle(120, 90.0, seed=5)).assign(volume=np.nan), "nav"),
    ]
    cases[0][0].loc[4, "volume"] = 0
    for d, asset in cases:
        cal = _cal(d) if asset != "crypto" else None
        if asset != "crypto" and len(d) == 7:
            cal = pd.DatetimeIndex([x for x in d["date"] if x != d["date"].iloc[4]])
        o2, r2 = ec.clean_bars(d, asset, cal)
        o3, r3 = clean_bars_v3(d, asset, cal)
        pd.testing.assert_frame_equal(o3, o2)
        # v2 omits ``off_calendar`` for crypto; v3 always reports it (0 there)
        assert {k: r3[k] for k in r2} == r2 and set(r2) <= set(V2_KEYS)
        assert set(V2_KEYS) - set(r2) <= {"off_calendar"}
        assert r3["sentinel"] == 0 and r3["constant_run"] == 0


def test_equity_calendar_v3_never_returns_sealed_dates():
    dates = pd.bdate_range("2026-09-20", "2026-10-09")
    cal = equity_calendar_v3(pd.DataFrame({"date": dates}))
    assert len(cal) > 0 and (cal < pd.Timestamp(SEAL_DATE)).all()
    assert SEAL_DATE == date(2026, 10, 1)
    cal2 = equity_calendar_v3(pd.Series(dates[::-1]), asof=date(2026, 9, 25))
    assert cal2.is_monotonic_increasing and cal2.max() == pd.Timestamp("2026-09-25")
    cal3 = equity_calendar_v3(pd.DatetimeIndex(list(dates) * 2))
    assert cal3.is_unique


def test_futures_keep_negative_prices():
    # crude, April 2020: collapse to -37.63 and a regime that stays far from the prior level
    p = list(_wiggle(60, 50.0, seed=1))
    d = _bars(p + [18.27, -37.63, -30.0, -25.0, -20.0, -22.0, -21.0])
    d["volume"] = 1000.0
    out, rep = clean_bars_v3(d, "futures")
    assert (out["adjusted_close"] == -37.63).sum() == 1 and rep["price"] == 0
    _, rep_eq = clean_bars_v3(d, "equity", _cal(d))
    assert rep_eq["price"] == 6  # every negative bar is dropped for equity


def test_futures_zero_price_kept_nan_dropped():
    d = _bars(_wiggle(40, 50.0, seed=2))
    d.loc[5, "adjusted_close"] = 0.0
    d.loc[5, ["open", "close"]] = 0.0
    d.loc[9, "close"] = np.nan
    d.loc[12, "volume"] = 0.0
    d.loc[13, "volume"] = np.nan
    out, rep = clean_bars_v3(d, "futures")
    assert rep["price"] == 1 and rep["zero_volume"] == 1  # NaN close, NaN volume
    assert (out["volume"] == 0).sum() == 1
    # the zero-price bar passes the price rule; the (pre-registered) difference spike rule then
    # removes it because the series reverts at once
    assert rep["spike_reversal"] == 1 and (out["adjusted_close"] == 0).sum() == 0


def test_futures_negative_series_spike_uses_differences():
    # full reversion within 5 bars of a one-day plunge: removed as a bad print by the
    # pre-registered difference rule (documented caveat); a lone bad up-spike too
    base = [20.0] * 30
    d = _bars(base + [-37.63, 10.0, 14.0, 17.0, 19.0, 19.5] + base[:10] + [999.0, 20.0, 20.5] + [-1.0])
    d["volume"] = 10.0
    out, rep = clean_bars_v3(d, "futures")
    assert rep["spike_reversal"] >= 2 and (out["adjusted_close"] > 100).sum() == 0
    # positive futures series use the v2 level rule (identical to equity without calendar)
    pos = _bars([10, 10.2, 25.0, 10.1, 10.3, 10.4])
    o_f, r_f = clean_bars_v3(pos, "futures")
    o_e, _ = clean_bars_v3(pos, "equity", _cal(pos))
    pd.testing.assert_frame_equal(o_f, o_e)
    assert r_f["spike_reversal"] == 1


def test_futures_needs_no_calendar():
    d = _bars(_wiggle(30, 80.0, seed=7))
    out, rep = clean_bars_v3(d, "futures")
    assert len(out) == 30 and rep["off_calendar"] == 0
    with pytest.raises(ValueError):
        clean_bars_v3(d, "equity")


def test_bad_asset_raises():
    with pytest.raises(ValueError):
        clean_bars_v3(_bars([1.0, 2.0]), "bond")  # type: ignore[arg-type]


def test_missing_column_raises():
    with pytest.raises(ValueError):
        clean_bars_v3(_bars([1.0, 2.0]).drop(columns=["adjusted_close"]), "crypto")


def test_sentinel_hit_logged_with_symbol(caplog):
    p = _wiggle(30)
    p[5] = 999999.9999
    d = _bars(p)
    with caplog.at_level(logging.WARNING, logger="firm.data.cleaning"):
        clean_bars_v3(d, "equity", _cal(d), symbol="XYZ")
    assert any("XYZ" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)

