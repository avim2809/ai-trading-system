"""Mechanics of scripts/run_insider_cluster_evaluation.py on synthetic data (no real data, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_insider_cluster_evaluation as ev  # noqa: E402


def _series(n=300, start="2010-01-04", px=10.0, drift=0.0, vol=200_000.0, adj_ratio=1.0):
    dates = pd.bdate_range(start, periods=n)
    close = px * np.cumprod(np.full(n, 1 + drift))
    d = pd.DataFrame({"date": dates, "open": close, "close": close,
                      "adjusted_close": close * adj_ratio, "volume": vol})
    d["adj_open"] = d["open"] * adj_ratio
    d["dollar_vol"] = d["adjusted_close"] * d["volume"]
    return d


def _events(ticker, entry_idxs, d, name_ok=True):
    return pd.DataFrame([{"issuer_cik": "1", "ticker": ticker, "known_date": d["date"].iloc[i - 1],
                          "status": "covered", "entry_idx": i, "entry_date": d["date"].iloc[i],
                          "name_ok": name_ok} for i in entry_idxs])


def _benches(n=300):
    b = _series(n, px=100.0, vol=1e7)
    return {s: b for s in ("IWC", "IWM", "IJH")}


def test_buckets_and_benchmarks():
    assert ev.adv_bucket(800_000) == "adv_lt_1m" and ev.bench_symbol(800_000) == "IWC"
    assert ev.adv_bucket(3e6) == "adv_1m_5m" and ev.bench_symbol(3e6) == "IWC"
    assert ev.adv_bucket(1e7) == "adv_5m_20m" and ev.bench_symbol(1e7) == "IWM"
    assert ev.adv_bucket(3e7) == "adv_gt_20m" and ev.bench_symbol(3e7) == "IJH"


def test_return_cost_and_excess_math():
    d = _series(drift=0.001)                 # dollar vol 10 x 200k = $2M -> 60 bps, IWC
    out = ev.event_returns(_events("X", [50], d), {"X": d}, _benches(), 63, d["date"].iloc[-1])
    r = out.iloc[0]
    expected_gross = d["adjusted_close"].iloc[50 + 62] / d["adj_open"].iloc[50] - 1
    assert r["gross"] == pytest.approx(expected_gross)
    assert r["cost"] == pytest.approx(0.006)
    assert r["xs_net"] == pytest.approx(r["net"] - r["bench_primary"])
    assert r["bench_primary"] == pytest.approx(0.0)   # flat benchmark
    assert not r["early_exit"]


def test_universe_filters_drop_illiquid_and_penny_names():
    illiquid = _series(vol=10_000)           # $100k/day < $0.5M
    penny = _series(px=1.5, vol=2_000_000)   # price < $2
    for d in (illiquid, penny):
        out = ev.event_returns(_events("X", [50], d), {"X": d}, _benches(), 63, d["date"].iloc[-1])
        assert out.empty


def test_overlap_rule_skips_events_while_a_position_is_open():
    d = _series()
    out = ev.event_returns(_events("X", [50, 80, 200], d), {"X": d}, _benches(), 63, d["date"].iloc[-1])
    assert list(out["entry_date"]) == [d["date"].iloc[50], d["date"].iloc[200]]


def test_early_exit_and_delisting_stress():
    d = _series(n=150)                       # series ends 30 days after entry at 100
    bench = _benches(400)
    last = bench["IWM"]["date"].iloc[-1]     # market continues long after -> a delisting
    out = ev.event_returns(_events("X", [120], d), {"X": d}, bench, 63, last)
    r = out.iloc[0]
    assert r["early_exit"]
    assert r["net_delist_stress"] == pytest.approx((1 + r["gross"]) * 0.7 - 1 - r["cost"])


def test_hold_past_data_end_is_not_evaluated():
    d = _series(n=150)
    out = ev.event_returns(_events("X", [120], d), {"X": d}, _benches(150), 63, d["date"].iloc[-1])
    assert out.empty


def test_month_cluster_bootstrap_centres_on_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.01, 0.1, 2000)
    months = np.repeat(np.arange(100), 20).astype(str)
    boot = ev.month_cluster_boot(v, months, 2000, seed=1)
    assert boot.mean() == pytest.approx(v.mean(), abs=0.002)
    assert np.quantile(boot, 0.01) < v.mean() < np.quantile(boot, 0.99)
