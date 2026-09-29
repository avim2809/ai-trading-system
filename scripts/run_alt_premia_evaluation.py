"""Pre-registered evaluation of the alternative-premia edge search (Step 2, 2026-09-29).

Frozen design: ``scripts/alt_premia_preregistered_bars.py`` (fingerprinted).
Plan: ``docs/edge_search_plan_2026_09.md``. Raw data: ``scripts/alt_premia_data.py``.

Subcommands::

    python scripts/run_alt_premia_evaluation.py selfcheck --data <dir>
    python scripts/run_alt_premia_evaluation.py power     --data <dir> --out <dir>
    python scripts/run_alt_premia_evaluation.py evaluate  --data <dir> --out <dir> \
        [--report docs/alt_premia_evaluation_2026_09.json] [--append-ledger]

``selfcheck`` is the mechanical look-ahead test: every input after a cutoff is
randomly perturbed, and no candidate's net return on or before the cutoff may
change. ``power`` uses benchmark returns only, never a candidate's.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import alt_premia_preregistered_bars as prereg  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402

log = logging.getLogger(__name__)

LEDGER = _ROOT / "docs" / "alt_premia_trial_history.json"
ETFS = ["SPY", "IEF", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ", "SVXY"]
T2_ASSETS = prereg.CANDIDATES["T2_cross_asset_trend"]["instrument"]
SVXY_LEVERAGE_CHANGE = pd.Timestamp("2018-02-28")  # first -0.5x return day
ANN = math.sqrt(prereg.TRADING_DAYS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    dates: pd.DatetimeIndex        # SPY trading calendar
    px: pd.DataFrame               # adjClose, SPY calendar
    vix: pd.Series
    vix3m: pd.Series
    put: pd.Series
    rf: pd.Series                  # per-SPY-day cash return
    btc: pd.Series                 # BTC close, own UTC daily calendar
    rf_btc: pd.Series              # per-calendar-day cash return on the BTC calendar
    fomc: pd.DatetimeIndex


def load_inputs(data_dir: Path) -> Inputs:
    end = pd.Timestamp(prereg.DATA_END)
    px = {}
    for t in ETFS:
        s = pd.read_parquet(data_dir / f"tiingo_{t}.parquet").set_index("date")["adjClose"]
        px[t] = s[~s.index.duplicated()]
    px = pd.DataFrame(px)
    dates = px["SPY"].dropna().index
    dates = dates[dates <= end]
    px = px.reindex(dates)
    for col in px.columns:
        first = px[col].first_valid_index()
        gaps = int(px.loc[first:, col].isna().sum()) if first is not None else 0
        if gaps:
            log.warning("%s: %d missing SPY trading days after inception (returns spanning them are dropped)", col, gaps)

    def cboe(name: str, col: str) -> pd.Series:
        s = pd.read_parquet(data_dir / f"cboe_{name}.parquet").set_index("date")[col]
        return s[~s.index.duplicated()].reindex(dates)

    vix, vix3m = cboe("VIX", "close").ffill(), cboe("VIX3M", "close").ffill()
    # PUT returns on its own calendar, then onto SPY dates (a missing PUT day must
    # not swallow the return spanning it).
    put_raw = pd.read_parquet(data_dir / "cboe_PUT.parquet").set_index("date")["put"]
    put_raw = put_raw[~put_raw.index.duplicated()].sort_index()
    put_raw = put_raw[put_raw.index >= pd.Timestamp("2007-01-01")]
    missing_put = dates[(dates >= put_raw.index.min()) & ~dates.isin(put_raw.index)]
    if len(missing_put):
        log.warning("PUT index missing on %d SPY trading days since 2007 (e.g. %s); their return is folded into the next PUT day",
                    len(missing_put), list(missing_put[:5].date))
    put = put_raw.reindex(put_raw.index.union(dates)).ffill().reindex(dates)
    tb = pd.read_parquet(data_dir / "fred_DTB3.parquet").set_index("date")["value"]
    tb = tb[~tb.index.duplicated()]
    rf = (tb.reindex(tb.index.union(dates)).ffill().reindex(dates) / 100.0 / prereg.TRADING_DAYS).fillna(0.0)
    btc = pd.read_parquet(data_dir / "tiingo_BTCUSD.parquet").set_index("date")["close"]
    btc = btc[~btc.index.duplicated()].sort_index()
    btc = btc[btc.index <= end]
    btc = btc.reindex(pd.date_range(btc.index.min(), btc.index.max(), freq="D")).ffill()
    rf_btc = (tb.reindex(tb.index.union(btc.index)).ffill().reindex(btc.index) / 100.0 / 365.0).fillna(0.0)
    fomc = pd.DatetimeIndex(pd.read_parquet(data_dir / "fomc_scheduled.parquet")["announcement"])
    missing = fomc[(fomc >= dates.min()) & (fomc <= end) & ~fomc.isin(dates)]
    if len(missing):
        log.warning("FOMC dates not in SPY calendar (dropped): %s", list(missing.date))
    fomc = fomc[fomc.isin(dates)]
    return Inputs(dates, px, vix, vix3m, put, rf, btc, rf_btc, fomc)


def perturbed(inp: Inputs, cutoff: pd.Timestamp, rng: np.random.Generator) -> Inputs:
    """Copy of ``inp`` with every input strictly after ``cutoff`` randomly perturbed."""
    def noise(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        s = s.copy()
        mask = s.index > cutoff
        shape = s.loc[mask].shape
        s.loc[mask] = s.loc[mask] * np.exp(rng.normal(0, 0.05, size=shape))
        return s
    return Inputs(inp.dates, noise(inp.px), noise(inp.vix), noise(inp.vix3m), noise(inp.put),
                  noise(inp.rf), noise(inp.btc), noise(inp.rf_btc), inp.fomc)


# ---------------------------------------------------------------------------
# Generic drifting-weight simulator
# ---------------------------------------------------------------------------

Decide = Callable[[int, np.ndarray], "np.ndarray | None"]


def simulate(rets: np.ndarray, rf: np.ndarray, decide: Decide, cost_bps: np.ndarray,
             cash: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Net daily returns of a long-only book.

    ``rets[i]`` is each asset's return on day i. Weights held from close i-1
    earn day i. After day i's return and drift, ``decide(i, w)`` may return new
    targets, which are set at close i (costs charged on day i). Anything not in
    assets sits in cash earning ``rf[i]`` (when ``cash``). Returns (net returns,
    weights held after each day's rebalance).
    """
    T, N = rets.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N))
    c = cost_bps / 1e4
    for i in range(T):
        r = np.nan_to_num(rets[i])
        cash_w = (1.0 - w.sum()) if cash else 0.0
        rp = float(w @ r + cash_w * rf[i])
        if 1.0 + rp > 0:
            w = w * (1.0 + r) / (1.0 + rp)
        tgt = decide(i, w)
        cost = 0.0
        if tgt is not None:
            tgt = np.asarray(tgt, dtype=float)
            if tgt.sum() > prereg.EXECUTION["max_gross"] + 1e-9 or (tgt < -1e-12).any():
                raise ValueError(f"target violates long-only/max_gross at {i}: {tgt}")
            cost = float(np.abs(tgt - w) @ c)
            w = tgt
        out[i] = rp - cost
        held[i] = w
    return out, held


def month_end_flags(dates: pd.DatetimeIndex) -> np.ndarray:
    m = dates.to_period("M")
    return np.r_[m[1:] != m[:-1], True]


def _cost_mult(stress: bool) -> float:
    return prereg.COSTS["stress_multiplier"] if stress else 1.0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bm1(inp: Inputs, stress: bool = False) -> pd.Series:
    r = inp.px[["SPY"]].pct_change().to_numpy()
    first = int(np.argmax(~np.isnan(r[:, 0])))
    net, _ = simulate(r, inp.rf.to_numpy(), lambda i, w: np.array([1.0]) if i == first else None,
                      np.array([prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)]))
    return pd.Series(net, inp.dates, name="BM1_SPY")


def bm2(inp: Inputs, stress: bool = False) -> pd.Series:
    r = inp.px[["SPY", "IEF"]].pct_change().to_numpy()
    me = month_end_flags(inp.dates)
    ok = inp.px[["SPY", "IEF"]].notna().all(axis=1).to_numpy()

    def decide(i: int, w: np.ndarray):
        if i >= 1 and me[i - 1] and ok[i - 1]:
            return np.array([0.6, 0.4])
        return None
    net, _ = simulate(r, inp.rf.to_numpy(), decide,
                      np.full(2, prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)))
    return pd.Series(net, inp.dates, name="BM2_60_40")


def bm3(inp: Inputs, stress: bool = False) -> pd.Series:
    spec = prereg.BENCHMARKS["BM3_SPY_VT"]
    rs = inp.px["SPY"].pct_change()
    sig = (rs.rolling(spec["vol_window"]).std() * ANN).to_numpy()
    r = rs.to_numpy()[:, None]

    def decide(i: int, w: np.ndarray):
        if i < 1 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        tgt = min(1.0, spec["target_vol"] / sig[i - 1])
        return np.array([tgt]) if abs(tgt - w[0]) > spec["band_abs"] else None
    net, _ = simulate(r, inp.rf.to_numpy(), decide,
                      np.array([prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)]))
    return pd.Series(net, inp.dates, name="BM3_SPY_VT")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _svxy_leverage(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.where(dates >= SVXY_LEVERAGE_CHANGE, 0.5, 1.0)


def v1_signal(inp: Inputs) -> np.ndarray:
    on = (inp.vix < inp.vix3m).to_numpy()
    valid = (inp.vix.notna() & inp.vix3m.notna()).to_numpy()
    return np.where(valid, on, False)


def v1(inp: Inputs, stress: bool = False, on: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    spec = prereg.CANDIDATES["V1_short_vol_timed"]
    lev = _svxy_leverage(inp.dates)
    r_svxy = inp.px["SVXY"].pct_change()
    r_idx = r_svxy / lev
    sig = (r_idx.rolling(spec["vol_window"]).std() * ANN).to_numpy()
    on = v1_signal(inp) if on is None else on
    r = r_svxy.to_numpy()[:, None]
    T = len(inp.dates)

    def decide(i: int, w: np.ndarray):
        if i < 2 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        lev_next = lev[min(i + 1, T - 1)]
        held_e = w[0] * lev_next
        tgt_e = (min(spec["exposure_cap"], spec["target_vol"] / sig[i - 1]) if on[i - 1] else 0.0)
        flip = bool(on[i - 1]) != bool(on[i - 2])
        drift = held_e > 0 and abs(tgt_e - held_e) > spec["band_rel"] * held_e
        over_cap = held_e > spec["exposure_cap"]
        if flip or drift or over_cap or (held_e == 0 and tgt_e > 0):
            return np.array([tgt_e / lev_next])
        return None
    net, held = simulate(r, inp.rf.to_numpy(), decide,
                         np.array([prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)]))
    exposure = held[:, 0] * np.r_[lev[1:], lev[-1]]
    return pd.Series(net, inp.dates, name="V1_short_vol_timed"), exposure


def v2(inp: Inputs, stress: bool = False) -> pd.Series:
    drag = prereg.COSTS["put_write_drag_bps_per_year"] * _cost_mult(stress) / 1e4 / prereg.TRADING_DAYS
    r = inp.put.pct_change(fill_method=None) - drag
    return r.rename("V2_put_write")


def k1_held_days(inp: Inputs) -> np.ndarray:
    d = inp.dates
    me = month_end_flags(d)
    held = me.copy()                               # last trading day of each month
    first = np.r_[True, me[:-1]]                   # first trading day of each month
    pos = np.flatnonzero(first)
    for k in range(3):
        idx = pos + k
        held[idx[idx < len(d)]] = True             # trading days 1..3 of the month
    held |= d.isin(inp.fomc)
    return held


def k1(inp: Inputs, stress: bool = False, held_days: np.ndarray | None = None) -> pd.Series:
    held_days = k1_held_days(inp) if held_days is None else held_days
    r = inp.px[["SPY"]].pct_change().to_numpy()
    T = len(inp.dates)

    def decide(i: int, w: np.ndarray):
        # At close i, position for day i+1 (schedule known in advance).
        want = 1.0 if (i + 1 < T and held_days[i + 1]) else 0.0
        return np.array([want]) if abs(want - w[0]) > 1e-12 else None
    net, _ = simulate(r, inp.rf.to_numpy(), decide,
                      np.array([prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)]))
    return pd.Series(net, inp.dates, name="K1_calendar")


def c1_signal(inp: Inputs) -> tuple[np.ndarray, np.ndarray]:
    spec = prereg.CANDIDATES["C1_btc_trend"]
    px = inp.btc
    on = (px / px.shift(spec["lookback_days"]) - 1.0 > 0).to_numpy()
    sig = (px.pct_change().rolling(spec["vol_window"]).std() * math.sqrt(365)).to_numpy()
    return on, sig


def _btc_to_spy(net: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Compound BTC UTC-day returns onto SPY dates: bar d goes to the first SPY date strictly after d."""
    pos = dates.searchsorted(net.index, side="right")
    keep = pos < len(dates)
    g = pd.Series(np.log1p(net.to_numpy()[keep]), index=dates[pos[keep]])
    out = np.expm1(g.groupby(level=0).sum())
    return out.reindex(dates).fillna(0.0)


def c1_btc(inp: Inputs, stress: bool = False, on: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    spec = prereg.CANDIDATES["C1_btc_trend"]
    on_s, sig = c1_signal(inp)
    on = on_s if on is None else on
    sunday = (inp.btc.index.dayofweek == 6)
    r = inp.btc.pct_change().to_numpy()[:, None]

    def decide(i: int, w: np.ndarray):
        # Review on Sunday d = i-1, effective at close of d+1 = i.
        if i < 1 or not sunday[i - 1] or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        tgt = min(1.0, spec["target_vol"] / sig[i - 1]) if on[i - 1] else 0.0
        flip = (tgt > 0) != (w[0] > 1e-12)
        if flip or abs(tgt - w[0]) > spec["band_abs"]:
            return np.array([tgt])
        return None
    net, held = simulate(r, inp.rf_btc.to_numpy(), decide,
                         np.array([prereg.COSTS["btc_bps_per_side"] * _cost_mult(stress)]))
    return pd.Series(net, inp.btc.index), held[:, 0]


def c1(inp: Inputs, stress: bool = False, on: np.ndarray | None = None) -> tuple[pd.Series, pd.Series]:
    net_btc, held = c1_btc(inp, stress, on)
    return _btc_to_spy(net_btc, inp.dates).rename("C1_btc_trend"), pd.Series(held, inp.btc.index)


def btc_bh_matched(inp: Inputs, weight: float, stress: bool = False) -> pd.Series:
    first_of_month = np.r_[True, inp.btc.index.month[1:] != inp.btc.index.month[:-1]]
    r = inp.btc.pct_change().to_numpy()[:, None]
    net, _ = simulate(r, inp.rf_btc.to_numpy(), lambda i, w: np.array([weight]) if first_of_month[i] else None,
                      np.array([prereg.COSTS["btc_bps_per_side"] * _cost_mult(stress)]))
    return _btc_to_spy(pd.Series(net, inp.btc.index), inp.dates).rename("BTC_BH_MATCHED")


def t2_states(inp: Inputs) -> pd.DataFrame:
    """Month-end on/off state per asset (index = month-end dates)."""
    spec = prereg.CANDIDATES["T2_cross_asset_trend"]
    px = inp.px[T2_ASSETS]
    L = spec["lookback_days"]
    cash_growth = (1 + inp.rf).cumprod()
    cash_ret = cash_growth / cash_growth.shift(L) - 1
    on = (px / px.shift(L) - 1).gt(cash_ret, axis=0) & px.shift(L).notna()
    me = month_end_flags(inp.dates)
    return on[me]


def t2(inp: Inputs, stress: bool = False, states: pd.DataFrame | None = None) -> pd.Series:
    spec = prereg.CANDIDATES["T2_cross_asset_trend"]
    states = t2_states(inp) if states is None else states
    rets = inp.px[T2_ASSETS].pct_change()
    r = rets.to_numpy()
    vw = spec["vol_window"]
    idx_of = {d: k for k, d in enumerate(inp.dates)}
    state_at = {idx_of[d]: states.loc[d].to_numpy(dtype=bool) for d in states.index}

    def decide(i: int, w: np.ndarray):
        t = i - 1
        if t not in state_at:
            return None
        on = state_at[t]
        if t + 1 < vw + 1 or not on.any():
            return np.zeros(len(T2_ASSETS))
        hist = r[t - vw + 1:t + 1]
        sig = np.nanstd(hist, axis=0, ddof=1) * ANN
        ok = on & np.isfinite(sig) & (sig > 0) & ~np.isnan(hist).any(axis=0)
        if not ok.any():
            return np.zeros(len(T2_ASSETS))
        raw = np.where(ok, 1.0 / np.where(ok, sig, 1.0), 0.0)
        wt = raw / raw.sum()
        basket = np.nan_to_num(hist) @ wt
        sp = basket.std(ddof=1) * ANN
        s = min(1.0, spec["target_vol"] / sp) if sp > 0 else 0.0
        return s * wt
    net, _ = simulate(r, inp.rf.to_numpy(), decide,
                      np.full(len(T2_ASSETS), prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)))
    return pd.Series(net, inp.dates, name="T2_cross_asset_trend")


def combo(inp: Inputs, streams: dict[str, pd.Series], name: str, vol_window: int,
          stress: bool = False, v1_exposure: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    """Inverse-vol mix of stream net returns, monthly; streams already hold their own cash."""
    keys = list(streams)
    mat = pd.DataFrame(streams).reindex(inp.dates).to_numpy()
    me = month_end_flags(inp.dates)
    avail = ~np.isnan(mat)

    def decide(i: int, w: np.ndarray):
        t = i - 1
        if t < vol_window or not me[t]:
            return None
        hist = mat[t - vol_window + 1:t + 1]
        if not avail[t - vol_window + 1:t + 1].all():
            return None
        sig = hist.std(axis=0, ddof=1)
        if (sig <= 0).any():
            return None
        raw = 1.0 / sig
        return raw / raw.sum()
    net, held = simulate(np.nan_to_num(mat), inp.rf.to_numpy(), decide,
                         np.full(len(keys), prereg.COSTS["etf_bps_per_side"] * _cost_mult(stress)), cash=False)
    # Periods before the first rebalance hold nothing and earn 0 (they're excluded by eval_start).
    tail = None
    if v1_exposure is not None and "V1_short_vol_timed" in keys:
        tail = held[:, keys.index("V1_short_vol_timed")] * v1_exposure
    return pd.Series(net, inp.dates, name=name), tail


def build_all(inp: Inputs, stress: bool = False) -> dict:
    out: dict = {}
    out["BM1_SPY"], out["BM2_60_40"], out["BM3_SPY_VT"] = bm1(inp, stress), bm2(inp, stress), bm3(inp, stress)
    out["V1_short_vol_timed"], v1_exp = v1(inp, stress)
    out["V2_put_write"] = v2(inp, stress)
    out["K1_calendar"] = k1(inp, stress)
    out["C1_btc_trend"], c1_held = c1(inp, stress)
    out["T2_cross_asset_trend"] = t2(inp, stress)
    c1_start = pd.Timestamp(prereg.CANDIDATES["C1_btc_trend"]["eval_start"])
    avg_w = float(c1_held[c1_held.index >= c1_start].mean())
    out["BTC_BH_MATCHED"] = btc_bh_matched(inp, avg_w, stress)
    m1s = {k: out[k] for k in prereg.CANDIDATES["M1_combo"]["streams"]}
    out["M1_combo"], m1_tail = combo(inp, m1s, "M1_combo", prereg.CANDIDATES["M1_combo"]["vol_window"], stress, v1_exp)
    m2s = {k: out[k] for k in prereg.CANDIDATES["M2_combo_crypto"]["streams"]}
    out["M2_combo_crypto"], m2_tail = combo(inp, m2s, "M2_combo_crypto",
                                            prereg.CANDIDATES["M2_combo_crypto"]["vol_window"], stress, v1_exp)
    out["_tail"] = {"V1_short_vol_timed": v1_exp, "M1_combo": m1_tail, "M2_combo_crypto": m2_tail}
    out["_c1_avg_weight"] = avg_w
    return out


# ---------------------------------------------------------------------------
# Placebos
# ---------------------------------------------------------------------------

def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    blocks = [x[k:k + block] for k in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[k] for k in order])[:n]


def placebo_k1_days(inp: Inputs, rng: np.random.Generator) -> np.ndarray:
    d = inp.dates
    held = np.zeros(len(d), dtype=bool)
    months = d.to_period("M")
    fomc = d.isin(inp.fomc)
    for m in months.unique():
        pos = np.flatnonzero(months == m)
        if len(pos) >= 4:
            s = rng.integers(0, len(pos) - 3)
            held[pos[s:s + 4]] = True
        for _ in range(int(fomc[pos].sum())):
            free = pos[~held[pos]]
            if len(free):
                held[rng.choice(free)] = True
    return held


def placebo_streams(inp: Inputs, base: dict, rng: np.random.Generator) -> dict:
    spec_v1 = prereg.CANDIDATES["V1_short_vol_timed"]
    on_v1 = v1_signal(inp)
    first = int(np.argmax(inp.px["SVXY"].notna().to_numpy()))
    on_p = on_v1.copy()
    on_p[first:] = _block_permute(on_v1[first:], 63, rng)
    v1p, v1p_exp = v1(inp, on=on_p)
    k1p = k1(inp, held_days=placebo_k1_days(inp, rng))
    on_c1, _ = c1_signal(inp)
    sundays = np.flatnonzero(inp.btc.index.dayofweek == 6)
    wk = on_c1[sundays]
    on_c1p = on_c1.copy()
    on_c1p[sundays] = _block_permute(wk, 13, rng)
    c1p, _ = c1(inp, on=on_c1p)
    st = t2_states(inp)
    stp = st.copy()
    for col in st.columns:
        stp[col] = _block_permute(st[col].to_numpy(), 12, rng)
    t2p = t2(inp, states=stp)
    m1s = {"BM1_SPY": base["BM1_SPY"], "T2_cross_asset_trend": t2p, "V1_short_vol_timed": v1p, "K1_calendar": k1p}
    m1p, _ = combo(inp, m1s, "M1_combo", prereg.CANDIDATES["M1_combo"]["vol_window"])
    m2p, _ = combo(inp, {**m1s, "C1_btc_trend": c1p}, "M2_combo_crypto", prereg.CANDIDATES["M2_combo_crypto"]["vol_window"])
    _ = spec_v1
    return {"V1_short_vol_timed": v1p, "K1_calendar": k1p, "C1_btc_trend": c1p,
            "T2_cross_asset_trend": t2p, "M1_combo": m1p, "M2_combo_crypto": m2p}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def sharpe(ex: np.ndarray) -> float:
    ex = ex[np.isfinite(ex)]
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * ANN) if sd > 0 else float("nan")


def stationary_indices(n: int, n_boot: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.broadcast_to(np.arange(n), (n_boot, n))
    block_start_pos = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    start_val = np.take_along_axis(starts, block_start_pos, axis=1)
    return ((start_val + pos - block_start_pos) % n).astype(np.int32)


def paired_sharpe_gap_boot(a: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    cfg = prereg.BOOTSTRAP
    rng = np.random.default_rng(seed)
    n = len(a)
    out = np.empty(cfg["n_boot"])
    chunk = 250
    for k in range(0, cfg["n_boot"], chunk):
        m = min(chunk, cfg["n_boot"] - k)
        idx = stationary_indices(n, m, cfg["mean_block_days"], rng)
        A, B = a[idx], b[idx]
        sa = A.mean(1) / A.std(1, ddof=1)
        sb = B.mean(1) / B.std(1, ddof=1)
        out[k:k + m] = (sa - sb) * ANN
    return out


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def vol_matched(rb: np.ndarray, rf: np.ndarray, target_sd: float) -> np.ndarray:
    ex = rb - rf
    lam = target_sd / ex.std(ddof=1)
    return rf + lam * ex


def worst_episodes(r: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """(peak_idx, trough_idx, depth) of the k deepest non-overlapping peak-to-trough episodes."""
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    new_high = np.flatnonzero(nav >= peak)
    bounds = list(new_high) + [len(nav)]
    eps = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s <= 1:
            continue
        seg = nav[s:e]
        j = int(np.argmin(seg))
        depth = seg[j] / nav[s] - 1
        if depth < 0:
            eps.append((s, s + j, float(depth)))
    return sorted(eps, key=lambda x: x[2])[:k]


def window(s: pd.Series, start: str, end: str | None = None) -> pd.Series:
    end = end or prereg.DATA_END
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# selfcheck / power / evaluate
# ---------------------------------------------------------------------------

SERIES = ["V1_short_vol_timed", "V2_put_write", "K1_calendar", "C1_btc_trend",
          "T2_cross_asset_trend", "M1_combo", "M2_combo_crypto"]


def cmd_selfcheck(args: argparse.Namespace) -> int:
    inp = load_inputs(Path(args.data))
    base = build_all(inp)
    rng = np.random.default_rng(7)
    cutoffs = [pd.Timestamp(x) for x in ("2012-06-15", "2016-03-31", "2018-02-02", "2020-03-11", "2024-08-02")]
    bad = 0
    for cut in cutoffs:
        alt = build_all(perturbed(inp, cut, rng))
        for name in SERIES + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]:
            a, b = base[name], alt[name]
            m = a.index <= cut
            diff = np.nanmax(np.abs(a[m].to_numpy() - b[m].to_numpy())) if m.any() else 0.0
            ok = diff == 0.0 or not np.isfinite(diff)
            log.info("look-ahead check cutoff=%s %-22s max|diff| on/before cutoff = %.3g %s",
                     cut.date(), name, diff, "OK" if ok else "LEAK")
            bad += 0 if ok else 1
    # Lag check: a VIX-only change on day t must not move V1's return on t+1.
    for cut in cutoffs:
        k = inp.dates.get_loc(inp.dates[inp.dates <= cut][-1])
        vix = inp.vix.copy()
        vix.iloc[k] = vix.iloc[k] * 3.0
        alt_inp = Inputs(inp.dates, inp.px, vix, inp.vix3m, inp.put, inp.rf, inp.btc, inp.rf_btc, inp.fomc)
        a, ea = v1(inp)
        b, eb = v1(alt_inp)
        # Returns through t and the position held at close t must be identical;
        # day t+1 may differ only by the cost of the trade placed at close t+1
        # (its new position first earns on t+2).
        d_ret = float(np.abs(a.iloc[:k + 1].to_numpy() - b.iloc[:k + 1].to_numpy()).max())
        d_pos = float(np.abs(ea[:k + 1] - eb[:k + 1]).max())
        d_next = abs(float(a.iloc[k + 1] - b.iloc[k + 1]))
        cost_bound = prereg.COSTS["etf_bps_per_side"] / 1e4 * 2.0 + 1e-12  # |dw| <= 2 SVXY weight units
        ok = d_ret == 0 and d_pos == 0 and d_next <= cost_bound
        log.info("lag check V1 VIX spike on %s: returns<=t %.3g, position@t %.3g, day t+1 %.3g (cost bound %.3g) %s",
                 inp.dates[k].date(), d_ret, d_pos, d_next, cost_bound, "OK" if ok else "LAG VIOLATION")
        bad += 0 if ok else 1
    log.info("selfcheck: %s", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


def _benchmarks_for(cand: str) -> list[str]:
    return prereg.CANDIDATES[cand]["benchmarks"]


def _bm_window_start(cand: str, bm: str) -> str:
    spec = prereg.CANDIDATES[cand]
    if cand == "K1_calendar" and bm == "BM2_60_40":
        return spec["bm2_eval_start"]
    return spec["eval_start"]


def cmd_power(args: argparse.Namespace) -> int:
    inp = load_inputs(Path(args.data))
    bms = {"BM1_SPY": bm1(inp), "BM2_60_40": bm2(inp), "BM3_SPY_VT": bm3(inp)}
    z_a = _norm_ppf(1 - prereg.BOOTSTRAP["alpha_one_sided"])
    z_b = _norm_ppf(0.80)
    rows = []
    for cand, (rho, claimed) in prereg.POWER_PRIORS.items():
        for bm in _benchmarks_for(cand):
            if bm == "BTC_BH_MATCHED":
                btc = _btc_to_spy(inp.btc.pct_change().fillna(0.0), inp.dates)
                ex = window(btc - inp.rf, prereg.CANDIDATES[cand]["eval_start"]).to_numpy()
            else:
                ex = window(bms[bm] - inp.rf, _bm_window_start(cand, bm)).to_numpy()
            T_years = len(ex) / prereg.TRADING_DAYS
            sb = sharpe(ex)
            # Jobson-Korkie/Memmel: Var(SR_c - SR_b), per-period SRs, n periods:
            # [2(1-rho) + 0.5 (sb^2 + sc^2 - 2 sb sc rho^2)] / n. Annualised gap -> x 252.
            # The SR^2 terms must be per-period (daily) Sharpes, not annual ones.
            n = len(ex)
            sbd = sb / ANN
            g = 0.5
            for _ in range(50):  # fixed point: g = (z_a + z_b) * SE(g)
                scd = (sb + g) / ANN
                var_d = (2 * (1 - rho) + 0.5 * (sbd**2 + scd**2 - 2 * sbd * scd * rho**2)) / n
                g = (z_a + z_b) * math.sqrt(var_d) * ANN
            rows.append({"candidate": cand, "benchmark": bm, "years": round(T_years, 2),
                         "benchmark_sharpe": round(sb, 3), "rho_prior": rho,
                         "mde_sharpe_gap_80pct_power": round(g, 3), "literature_gap": claimed,
                         "underpowered": bool(g > claimed)})
            log.info("power %-22s vs %-15s years=%.1f SR_b=%.2f rho=%.2f MDE=%.2f claimed=%.2f %s",
                     cand, bm, T_years, sb, rho, g, claimed, "UNDERPOWERED" if g > claimed else "")
    # Simulation check of the analytic MDE on one pair (T2 vs BM1): power at the MDE should be ~0.8.
    row = next(r for r in rows if r["candidate"] == "T2_cross_asset_trend" and r["benchmark"] == "BM1_SPY")
    ex_b = window(bms["BM1_SPY"] - inp.rf, prereg.CANDIDATES["T2_cross_asset_trend"]["eval_start"]).to_numpy()
    rng = np.random.default_rng(prereg.SEED + 2)
    # Both series must be random: resample the benchmark path (stationary blocks),
    # and build the candidate from it plus fresh noise, without re-standardising,
    # so the realised Sharpe gap varies around the true gap g.
    sd = ex_b.std(ddof=1)
    mu_b = ex_b.mean()
    rho, g = row["rho_prior"], row["mde_sharpe_gap_80pct_power"]
    sb = sharpe(ex_b)
    hits, n_sim = 0, 100
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]
    for s in range(n_sim):
        b_path = ex_b[stationary_indices(len(ex_b), 1, prereg.BOOTSTRAP["mean_block_days"], rng)[0]]
        eps = rng.standard_normal(len(b_path))
        c = (sb + g) / ANN * sd + sd * (rho * (b_path - mu_b) / sd + math.sqrt(1 - rho**2) * eps)
        boot = paired_sharpe_gap_boot(c, b_path, seed=1000 + s)
        hits += int(np.quantile(boot, alpha) > 0)
    sim_power = hits / n_sim
    log.info("power sim check T2 vs BM1 at MDE=%.2f: empirical power %.2f (n_sim=%d)", g, sim_power, n_sim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": prereg.bars_fingerprint(),
           "method": "analytic Jobson-Korkie/Memmel SE of the Sharpe gap, one-sided alpha_one_sided, 80% power; "
                     "priors from POWER_PRIORS; uses benchmark returns only",
           "rows": rows, "sim_check": {"pair": "T2_cross_asset_trend vs BM1_SPY", "mde": g,
                                       "empirical_power": sim_power, "n_sim": n_sim}}
    (out / "power_analysis.json").write_text(json.dumps(rep, indent=2))
    log.info("wrote %s", out / "power_analysis.json")
    return 0


def _norm_ppf(p: float) -> float:
    from firm.eval.overfitting import _norm_ppf as f
    return f(p)


def _seed_for(*parts: str) -> int:
    import hashlib
    return prereg.SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) % 10_000_000


def cmd_evaluate(args: argparse.Namespace) -> int:
    fp = prereg.bars_fingerprint()
    inp = load_inputs(Path(args.data))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("fingerprint %s", fp)
    base = build_all(inp)
    stress = build_all(inp, stress=True)
    rf = inp.rf
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]

    frame = pd.DataFrame({k: v for k, v in base.items() if not k.startswith("_")})
    frame["rf"] = rf
    frame.to_parquet(out_dir / "returns_base.parquet")
    pd.DataFrame({k: v for k, v in stress.items() if not k.startswith("_")}).assign(rf=rf).to_parquet(
        out_dir / "returns_stress.parquet")
    pd.DataFrame({k: pd.Series(v, inp.dates) for k, v in base["_tail"].items() if v is not None}).to_parquet(
        out_dir / "tail_exposure.parquet")

    # Placebos
    log.info("placebos: %d draws", prereg.PLACEBO["n_draws"])
    prng = np.random.default_rng(prereg.PLACEBO["seed"])
    plac: dict[str, list[float]] = {k: [] for k in SERIES if prereg.PLACEBO["rules"].get(k)}
    for draw in range(prereg.PLACEBO["n_draws"]):
        ps = placebo_streams(inp, base, prng)
        for k in plac:
            ex = window(ps[k] - rf, prereg.CANDIDATES[k]["eval_start"]).to_numpy()
            plac[k].append(sharpe(ex))
        if draw % 50 == 0:
            log.info("placebo draw %d", draw)
    pd.DataFrame(plac).to_parquet(out_dir / "placebo_sharpes.parquet")

    # PBO / DSR
    pbo_cols = SERIES + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]
    pbo_mat = window(frame[pbo_cols].sub(rf, axis=0), prereg.PBO["window_start"]).dropna()
    pbo = float(cscv_pbo(pbo_mat.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    trial_daily_sr = (pbo_mat.mean() / pbo_mat.std(ddof=1)).to_numpy()
    log.info("PBO %.3f over %d series x %d days", pbo, pbo_mat.shape[1], pbo_mat.shape[0])

    results = {}
    for cand in SERIES:
        spec = prereg.CANDIDATES[cand]
        c_ex_full = window(base[cand] - rf, spec["eval_start"])
        res = {"eval_start": spec["eval_start"], "n_days": int(len(c_ex_full)),
               "sharpe": sharpe(c_ex_full.to_numpy()),
               "cagr": float(np.prod(1 + window(base[cand], spec["eval_start"]).to_numpy()) ** (
                   prereg.TRADING_DAYS / len(c_ex_full)) - 1),
               "vol": float(c_ex_full.std(ddof=1) * ANN),
               "max_dd": max_drawdown(window(base[cand], spec["eval_start"]).to_numpy()),
               "vs": {}}
        res["dsr"] = float(deflated_sharpe(c_ex_full.to_numpy(), trial_daily_sr, prior_trials=prereg.DSR["prior_trials"]))
        if cand in plac:
            arr = np.asarray(plac[cand])
            res["placebo_p95"] = float(np.nanpercentile(arr, prereg.PLACEBO["pass_percentile"]))
            res["placebo_median"] = float(np.nanmedian(arr))
            res["A3"] = bool(res["sharpe"] > res["placebo_p95"])
        else:
            res["A3"] = True
            res["A3_note"] = "n/a (untimed)"
        if cand in prereg.TAIL["applies_to"]:
            tail = base["_tail"][cand]
            ts = pd.Series(tail, inp.dates)
            res["max_short_vol_exposure"] = float(window(ts, spec["eval_start"]).max())
            res["TAIL"] = bool(res["max_short_vol_exposure"] <= 0.25 + 1e-9)
        bm_list = spec["benchmarks"] + spec.get("reported_only_vs", [])
        for bm in bm_list:
            start = _bm_window_start(cand, bm)
            c = window(base[cand], start)
            b = window(base[bm], start)
            r_f = window(rf, start)
            j = pd.concat([c, b, r_f], axis=1, keys=["c", "b", "rf"]).dropna()
            ce, be = (j.c - j.rf).to_numpy(), (j.b - j.rf).to_numpy()
            gap = sharpe(ce) - sharpe(be)
            boot = paired_sharpe_gap_boot(ce, be, _seed_for(cand, bm))
            lb, ub = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
            pp = window(j.c, spec["post_publication_start"]), window(j.b, spec["post_publication_start"])
            pr = window(j.rf, spec["post_publication_start"])
            gap_post = sharpe((pp[0] - pr).to_numpy()) - sharpe((pp[1] - pr).to_numpy())
            cs, bs = window(stress[cand], start), window(stress[bm], start)
            js = pd.concat([cs, bs, r_f], axis=1, keys=["c", "b", "rf"]).dropna()
            gap_stress = sharpe((js.c - js.rf).to_numpy()) - sharpe((js.b - js.rf).to_numpy())
            vm = vol_matched(j.b.to_numpy(), j.rf.to_numpy(), ce.std(ddof=1))
            dd_c, dd_vm = max_drawdown(j.c.to_numpy()), max_drawdown(vm)
            eps = worst_episodes(vm, 3)
            ep_c = [float(np.prod(1 + j.c.to_numpy()[p + 1:t + 1]) - 1) for p, t, _ in eps]
            ep_b = [d for _, _, d in eps]
            gated = bm in spec["benchmarks"]
            res["vs"][bm] = {
                "gated": gated, "window_start": start, "n_days": int(len(j)),
                "sharpe_c": sharpe(ce), "sharpe_b": sharpe(be), "gap": gap, "lb": lb, "ub": ub,
                "gap_post_publication": gap_post, "gap_stress_costs": gap_stress,
                "max_dd_c": dd_c, "max_dd_vol_matched_b": dd_vm, "max_dd_b_raw": max_drawdown(j.b.to_numpy()),
                "worst_episodes_b": [{"peak": str(j.index[p].date()), "trough": str(j.index[t].date()), "depth": d}
                                     for p, t, d in eps],
                "episode_returns_c": ep_c,
                "A1": bool(gap > 0 and lb > 0), "A4": bool(gap_post > 0), "A5": bool(gap_stress > 0),
                "A6": bool(dd_c <= dd_vm), "D": bool(ub < 0),
                "B_a": bool(lb > -0.10), "B_b": bool(dd_c <= 0.75 * dd_vm),
                "B_c": bool(len(eps) > 0 and np.mean(ep_c) >= 0.5 * np.mean(ep_b)),
            }
            log.info("%-22s vs %-15s SR %.2f vs %.2f gap %+.2f [LB %+.2f, UB %+.2f] post %+.2f stress %+.2f DD %.1f%% vs %.1f%%%s",
                     cand, bm, sharpe(ce), sharpe(be), gap, lb, ub, gap_post, gap_stress, 100 * dd_c, 100 * dd_vm,
                     "" if gated else " (reported only)")
        gated = [v for v in res["vs"].values() if v["gated"]]
        bars = {
            "A1": all(v["A1"] for v in gated), "A2": bool(res["dsr"] > 0.95), "A3": res["A3"],
            "A4": all(v["A4"] for v in gated), "A5": all(v["A5"] for v in gated),
            "A6": all(v["A6"] for v in gated), "A7": bool(pbo < 0.50),
        }
        if "TAIL" in res:
            bars["TAIL"] = res["TAIL"]
        tier_b = {"B_a": all(v["B_a"] for v in gated), "B_b": all(v["B_b"] for v in gated),
                  "B_c": all(v["B_c"] for v in gated), "B_d": bars["A3"] and bars["A4"] and bars["A5"]}
        if "TAIL" in res:
            tier_b["TAIL"] = res["TAIL"]
        tier_d = any(v["D"] for v in gated)
        res["bars_A"], res["bars_B"], res["tier_D_condition"] = bars, tier_b, tier_d
        res["tier"] = prereg.classify(bars, tier_d, tier_b)
        results[cand] = res
        log.info("==> %s tier %s  A=%s  B=%s  D=%s", cand, res["tier"], bars, tier_b, tier_d)

    bm_summary = {}
    for bm in ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]:
        for start in sorted({prereg.CANDIDATES[c]["eval_start"] for c in SERIES}):
            s = window(base[bm], start)
            bm_summary.setdefault(bm, {})[start] = {
                "sharpe": sharpe((s - window(rf, start)).to_numpy()), "max_dd": max_drawdown(s.to_numpy())}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
        "prereg_at": prereg.PREREGISTERED_AT, "data_end": prereg.DATA_END,
        "alpha_one_sided": alpha, "pbo": pbo, "dsr_trials": int(len(trial_daily_sr)),
        "c1_avg_weight": base["_c1_avg_weight"], "candidates": results, "benchmarks": bm_summary,
        "fallback_benchmark": prereg.FALLBACK_BENCHMARK,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "alt_premia", "entries": []}
        ledger["entries"].append({
            "date": datetime.now(timezone.utc).date().isoformat(), "fingerprint": fp,
            "n_trials": int(len(trial_daily_sr)),
            "trials": pbo_cols, "trial_daily_sharpes": [float(x) for x in trial_daily_sr],
            "tiers": {k: v["tier"] for k, v in results.items()},
        })
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
        log.info("ledger updated: %s (cumulative %d)", LEDGER, ledger["cumulative_trials"])
    log.info("tiers: %s", {k: v["tier"] for k, v in results.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selfcheck", "power", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        if name != "selfcheck":
            p.add_argument("--out", required=True)
        if name == "evaluate":
            p.add_argument("--report")
            p.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"selfcheck": cmd_selfcheck, "power": cmd_power, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
