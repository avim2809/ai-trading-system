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

import run_insider_cluster_evaluation as ev


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


def _calendar_time_reference(df, cache, benches, bench_col, dates):
    """The original per-event pandas implementation, kept to pin the fast one."""
    num = pd.Series(0.0, index=dates)
    cnt = pd.Series(0.0, index=dates)
    for r in df.itertuples(index=False):
        d = cache[r.ticker].set_index("date")
        sym = r.bench_primary_sym if bench_col == "bench_primary" else "IWM"
        b = benches[sym].set_index("date")
        span = d.loc[r.entry_date:r.exit_date]
        if len(span) < 2:
            continue
        s = span["adjusted_close"].pct_change()
        s.iloc[0] = span["adjusted_close"].iloc[0] / span["adj_open"].iloc[0] - 1
        bs = b.loc[r.entry_date:r.exit_date, "adjusted_close"].pct_change()
        if len(bs):
            bs.iloc[0] = b.loc[r.entry_date, "adjusted_close"] / b.loc[r.entry_date, "adj_open"] - 1
        xs = (s - bs.reindex(s.index)).fillna(0.0)
        xs.iloc[0] -= r.cost / 2
        xs.iloc[-1] -= r.cost / 2
        idx = xs.index.intersection(dates)
        num.loc[idx] += xs.loc[idx].to_numpy()
        cnt.loc[idx] += 1
    return (num / cnt.replace(0, np.nan)).fillna(0.0)


def test_fast_calendar_time_matches_reference_implementation():
    rng = np.random.default_rng(3)
    cache = {}
    for i, t in enumerate(("A", "B", "C")):
        d = _series(n=400, px=10.0 + i, vol=300_000.0)
        d["adjusted_close"] = d["adjusted_close"] * np.cumprod(1 + rng.normal(0, 0.02, len(d)))
        d["adj_open"] = d["adjusted_close"] * (1 + rng.normal(0, 0.005, len(d)))
        cache[t] = d
    cache["B"].loc[150, "adjusted_close"] = np.nan          # a missing price inside a hold
    benches = {}
    for s in ("IWC", "IWM", "IJH"):
        b = _series(n=400, px=100.0, vol=1e7)
        b["adjusted_close"] = b["adjusted_close"] * np.cumprod(1 + rng.normal(0, 0.01, len(b)))
        b["adj_open"] = b["adjusted_close"] * (1 + rng.normal(0, 0.003, len(b)))
        benches[s] = b
    benches["IWC"] = benches["IWC"].drop(index=[130, 131, 260]).reset_index(drop=True)  # bench gaps
    rows = []
    for t, e, x, sym in (("A", 40, 102, "IWC"), ("A", 120, 182, "IWC"), ("B", 100, 225, "IWM"),
                         ("C", 250, 312, "IJH"), ("C", 300, 362, "IWC"), ("B", 380, 399, "IJH")):
        d = cache[t]
        rows.append({"ticker": t, "entry_date": d["date"].iloc[e], "exit_date": d["date"].iloc[x],
                     "cost": 0.006, "bench_primary_sym": sym})
    df = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(benches["IWM"]["date"].iloc[60:])   # positions open before the calendar starts
    prep: dict = {}
    for bcol in ("bench_primary", "bench_iwm"):
        fast = ev.calendar_time(df, cache, benches, bcol, dates, prep)
        ref = _calendar_time_reference(df, cache, benches, bcol, dates)
        np.testing.assert_allclose(fast.to_numpy(), ref.to_numpy(), rtol=0, atol=1e-15)
        assert (fast.index == ref.index).all()
