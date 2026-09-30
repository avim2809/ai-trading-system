"""Evaluation harness for S1 (industry/sector ETF momentum), EODHD shortlist.

Frozen design: ``scripts/eodhd_s1_industry_momentum_preregistered_bars.py``
(``bars_fingerprint()`` = 287294d0552396299b2e4fbe916a92a9adef564d53dbb6458c4a591941c8b359,
PREREGISTERED_AT 2026-09-30T19:18:54Z). The pre-registration file is FROZEN and
is never edited by this harness; any bug or ambiguity found while implementing
it is resolved by the most literal reading and recorded in ``PREREG_ISSUES``
below, not patched into the frozen file.

Subcommand::

    python scripts/run_eodhd_s1_evaluation.py evaluate --out <dir> \
        [--report docs/eodhd_s1_evaluation_2026_10.json] [--append-ledger]

A7 (independent recompute) is PENDING by instruction: this run does not attempt
an independent recompute of itself, and any resulting Tier A is reported as
"conditional on A7", never as a final A.

House style / reuse: ``stationary_indices`` is imported verbatim from
``scripts/run_alt_premia_evaluation.py`` per instruction (a pure, prereg-
agnostic block-bootstrap index generator). Everything that depends on THIS
prereg's own config (bootstrap alpha, seed, mean block length) is written
fresh here rather than reusing alt_premia's own wrapper functions, which are
hard-wired to alt_premia's own (different) prereg config.
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

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eodhd_s1_industry_momentum_preregistered_bars as prereg  # noqa: E402
from eodhd_clean import clean_bars, equity_calendar  # noqa: E402
from run_alt_premia_evaluation import stationary_indices  # noqa: E402  (reused per instruction)
from firm.eval.overfitting import cscv_pbo, deflated_sharpe, _norm_ppf  # noqa: E402

log = logging.getLogger(__name__)

TRADING_DAYS = prereg.TRADING_DAYS
ANN = math.sqrt(TRADING_DAYS)
EODHD = _ROOT / "data" / "research" / "eodhd" / "etfs_full"
FRED_DTB3 = _ROOT / "data" / "research" / "fred" / "DTB3.parquet"
LEDGER = _ROOT / "docs" / "S1_trial_history.json"

UNIVERSE = prereg.UNIVERSE  # 44, fixed declared order
IDX = {t: i for i, t in enumerate(UNIVERSE)}
N_UNIV = len(UNIVERSE)
BIL_START = pd.Timestamp("2007-05-30")
IEF_START = pd.Timestamp("2002-07-26")

# ---------------------------------------------------------------------------
# Issues found implementing the FROZEN pre-registration. Per instruction: never
# patch the frozen file; implement the most literal reading and record here.
# ---------------------------------------------------------------------------
PREREG_ISSUES = [
    {
        "id": "issue_1_history_floor_undersizes_12_1",
        "text": "ELIGIBILITY['min_history_calendar_days'] = 210 (~7 months) is sized only for "
               "the 6-1 variant's own lookback (6+1 months), but VARIANTS also declares 12-1 "
               "variants needing ~13 months of history; the ELIGIBILITY text itself claims 210 "
               "days covers '6-1 (or 12-1)', which is not arithmetically true for 12-1. Most "
               "literal fix implemented: use the single global 210-day/$1M ELIGIBILITY floor "
               "for universe membership (unchanged, exactly as frozen), AND separately require "
               "a computable (non-NaN, asof-based) signal for whichever variant is being ranked "
               "that month -- a ticker that is nominally 'eligible' but lacks the extra ~6 months "
               "a 12-1 signal needs (only possible for a ticker added to the universe well after "
               "window start, e.g. XLRE 2015, XLC 2018, or any later-seasoning industry ETF) is "
               "silently dropped from THAT variant's ranking pool for that month only, logged and "
               "counted (see 'n_12_1_signal_gaps' in the report).",
    },
    {
        "id": "issue_2_open_vs_close_execution_conflict",
        "text": "Protocol §2 states execution generally as 'trades fill at the next bar's "
               "adjusted open' for every candidate, but §3's own BM2/BM3 text says the target is "
               "set 'at close t+1' (a full close-to-close rebalance, not an open/close split) -- "
               "the protocol's general rule and its own benchmark-specific text disagree. Most "
               "literal reading implemented: BM1/BM2/BM3 follow their OWN literal §3 text (close "
               "t+1, no open/close split -- reusing the alt_premia house-style close-based "
               "simulator); the candidate (all 4 variants) and PRIMARY_BENCHMARK use §2's "
               "adjusted-open execution, since PRIMARY_BENCHMARK's own text says 'same cadence "
               "as the candidate,' which I read as also meaning the same fill convention.",
    },
    {
        "id": "issue_3_vfitx_cost_unspecified",
        "text": "COSTS only defines ETF ADV-tier bps; it says nothing about a mutual-fund NAV "
               "proxy (VFITX). Implemented: VFITX leg costs 0 bps (it is a return proxy standing "
               "in for a benchmark's pre-2002-07-26 bond leg, never an actual traded position, "
               "per the cleaning doc's own 'never as a traded holding' language) -- a documented "
               "assumption, not a computed figure.",
    },
    {
        "id": "issue_4_which_bars_apply_per_variant",
        "text": "TIER_A_BARS/TIER_D_RULE/TIER_B_BARS are written in the singular ('the "
               "candidate'), while DSR/PBO explicitly span all 4 declared variants. Implemented: "
               "A1/A3/A4/A5/D/B are computed ONLY for PRIMARY_VARIANT (that is 'the candidate' "
               "being tiered); A2 (DSR) and A6 (PBO) are panel-wide statistics computed over all "
               "4 variants (+ BM1-3 for PBO), exactly as their own dict text specifies. The other "
               "3 variants are never separately tiered.",
    },
    {
        "id": "issue_5_pbo_reference_is_primary_benchmark_not_cash",
        "text": "PBO['series'] literally specifies 'daily excess returns over the primary "
               "benchmark' (not over cash/rf) -- implemented literally: the PBO panel's 7 columns "
               "(4 variants + BM1-3) are each (series - PRIMARY_BENCHMARK), not (series - rf). "
               "A1/A4/A5/D/bootstrap, by contrast, use the more conventional excess-of-cash Sharpe "
               "(subtracting rf) before taking a Sharpe gap -- this is the alt_premia house-style "
               "convention, since the protocol itself does not say whether 'Sharpe' means Sharpe "
               "of raw or cash-excess returns; flagged here since it is the one place this "
               "harness makes that convention choice rather than reading it directly off frozen "
               "text.",
    },
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested on synthetic data in
# tests/test_run_eodhd_s1_evaluation.py)
# ---------------------------------------------------------------------------

def adjusted_open(open_: np.ndarray, close_: np.ndarray, adj_close_: np.ndarray) -> np.ndarray:
    """``open * adjusted_close / close`` (protocol §2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return open_ * adj_close_ / close_


def split_day_return(adj_close_prev: np.ndarray, adj_open: np.ndarray,
                      adj_close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(overnight, intraday) legs of a next-open execution day.

    overnight = adj_open / adj_close_prev - 1 (close t-1 -> open t, OLD weights)
    intraday  = adj_close / adj_open - 1      (open t -> close t, NEW weights)
    (1+overnight)*(1+intraday) - 1 == the plain close-to-close return.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = adj_open / adj_close_prev - 1.0
        intraday = adj_close / adj_open - 1.0
    return overnight, intraday


def cash_return_series(bil_c2c: pd.Series, dtb3_rate: pd.Series) -> pd.Series:
    """BIL total return where available, else FRED DTB3 accrued (rate/100/252)."""
    return bil_c2c.where(bil_c2c.notna(), dtb3_rate)


def splice_bond_return(primary_c2c: pd.Series, proxy_c2c: pd.Series) -> pd.Series:
    """IEF's own return where computable, else the VFITX proxy's (protocol §3)."""
    return primary_c2c.where(primary_c2c.notna(), proxy_c2c)


def cost_bps_for_adv(adv20: float, adv_threshold: float = 50_000_000.0,
                      bps_ge: float = 3.0, bps_lt: float = 10.0) -> float:
    """ETF cost tier by ADV20 (protocol §2). NaN ADV defaults to the higher (10bps) tier."""
    if adv20 is None or not np.isfinite(adv20):
        return bps_lt
    return bps_ge if adv20 >= adv_threshold else bps_lt


def select_holdings(signal: dict[str, float], eligible: list[str], selection: str,
                     n_min_for_trading: int, top_n: int | None = None) -> list[str]:
    """Rank ``eligible`` tickers with a computable (non-NaN) signal, tie-break by
    ascending ticker symbol (a lower symbol wins a tie), and select the top
    tercile or top-N. Returns [] (cash) if fewer than ``n_min_for_trading``
    tickers have a computable signal."""
    pool = [t for t in eligible if t in signal and np.isfinite(signal[t])]
    if len(pool) < n_min_for_trading:
        return []
    order = sorted(pool, key=lambda t: (-signal[t], t))
    if selection == "top_tercile":
        n_hold = max(1, round(len(pool) / 3))
    else:
        n_hold = min(top_n, len(pool))
    return order[:n_hold]


def month_end_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    m = dates.to_period("M")
    return np.r_[m[1:] != m[:-1], True]


def half_windows(window_start: str, window_end: str, midpoint: str) -> tuple[str, str, str, str]:
    """(h1_start, h1_end, h2_start, h2_end): WINDOW split at ``midpoint`` (protocol
    §4 'Halves'), first half inclusive of the midpoint date itself."""
    mid = pd.Timestamp(midpoint)
    h2_start = (mid + pd.Timedelta(days=1)).date().isoformat()
    return window_start, midpoint, h2_start, window_end


def sharpe(ex: np.ndarray) -> float:
    ex = np.asarray(ex, dtype=float)
    ex = ex[np.isfinite(ex)]
    if len(ex) < 2:
        return float("nan")
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * ANN) if sd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + np.nan_to_num(r))
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def cagr(r: np.ndarray, periods_per_year: float = TRADING_DAYS) -> float:
    r = np.nan_to_num(r)
    n = len(r)
    if n == 0:
        return float("nan")
    total = float(np.prod(1 + r))
    return float(total ** (periods_per_year / n) - 1)


def paired_sharpe_gap_boot(a: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """Paired stationary block bootstrap of the annualised Sharpe gap (a - b),
    using THIS prereg's own BOOTSTRAP config (mean_block_days, n_boot, seed)."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Asset:
    c2c: pd.Series       # close-to-close return, cal-aligned, NaN outside domain / across gaps
    adj_open: pd.Series  # cal-aligned adjusted open
    adj_close: pd.Series  # cal-aligned adjusted close (for asof signal lookups)
    adv20: pd.Series     # cal-aligned 20-session median dollar volume
    first_date: pd.Timestamp
    n_gap_days: int


def _load_clean(ticker: str, asset: str, cal: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pd.read_parquet(EODHD / f"{ticker}.parquet")
    d, rep = clean_bars(raw, asset=asset, calendar=cal)
    if d["segment"].nunique() != 1:
        log.warning("%s: %d segments at this cleaning fingerprint (unexpected; AVAILABILITY "
                    "claims 1 for the S1 universe) -- cross-segment returns are NaN'd", ticker,
                    d["segment"].nunique())
    return d


def _to_asset(d: pd.DataFrame, cal: pd.DatetimeIndex) -> Asset:
    d = d.set_index("date")
    idx = cal[(cal >= d.index.min()) & (cal <= d.index.max())]
    n_gap = int((~idx.isin(d.index)).sum())
    close = d["close"].reindex(cal)
    adj_close = d["adjusted_close"].reindex(cal)
    open_ = d["open"].reindex(cal)
    seg = d["segment"].reindex(cal)
    adj_open = pd.Series(adjusted_open(open_.to_numpy(), close.to_numpy(), adj_close.to_numpy()), cal)
    dollar_vol = (d["adjusted_close"].astype(float) * d["volume"].astype(float))
    adv20 = dollar_vol.rolling(20, min_periods=20).median().reindex(cal)
    c2c = adj_close.pct_change()
    cross_seg = seg.ne(seg.shift(1)) & seg.notna() & seg.shift(1).notna()
    c2c = c2c.where(~cross_seg, np.nan)
    return Asset(c2c, adj_open, adj_close, adv20, d.index.min(), n_gap)


def load_universe(cal: pd.DatetimeIndex) -> dict[str, Asset]:
    out: dict[str, Asset] = {}
    for t in UNIVERSE + ["SPY", "IEF", "BIL"]:
        out[t] = _to_asset(_load_clean(t, "equity", cal), cal)
    out["VFITX"] = _to_asset(_load_clean("VFITX", "nav", cal), cal)
    for t, a in out.items():
        if a.n_gap_days:
            log.warning("%s: %d missing sessions within its own clean domain (returns spanning "
                        "them are dropped)", t, a.n_gap_days)
    return out


def load_dtb3(cal: pd.DatetimeIndex) -> pd.Series:
    tb = pd.read_parquet(FRED_DTB3).set_index("date")["value"]
    tb = tb[~tb.index.duplicated()].sort_index()
    return (tb.reindex(tb.index.union(cal)).ffill().reindex(cal) / 100.0 / TRADING_DAYS).fillna(0.0)


# ---------------------------------------------------------------------------
# Eligibility / signal
# ---------------------------------------------------------------------------

def eligibility_at(assets: dict[str, Asset], t: pd.Timestamp) -> list[str]:
    out = []
    for tk in UNIVERSE:
        a = assets[tk]
        if pd.isna(a.first_date) or t < a.first_date:
            continue
        days = (t - a.first_date).days
        adv = a.adv20.asof(t)
        if days >= prereg.ELIGIBILITY["min_history_calendar_days"] and pd.notna(adv) and \
                adv >= prereg.ELIGIBILITY["adv20_floor_usd"]:
            out.append(tk)
    return out


def signal_at(adj_close: pd.Series, t: pd.Timestamp, formation_months: int, skip_months: int) -> float:
    skip_end = t - pd.DateOffset(months=skip_months)
    formation_start = skip_end - pd.DateOffset(months=formation_months)
    p_end = adj_close.asof(skip_end)
    p_start = adj_close.asof(formation_start)
    if pd.isna(p_end) or pd.isna(p_start) or p_start <= 0:
        return float("nan")
    return float(p_end / p_start - 1.0)


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def simulate_open_exec(full_ret: np.ndarray, overnight: np.ndarray, intraday: np.ndarray,
                        rf: np.ndarray, targets: dict[int, np.ndarray],
                        cost_bps: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Next-bar-adjusted-open execution (protocol §2). ``targets``/``cost_bps`` are
    keyed by the EXECUTION day's integer position; weights drift between trades."""
    T, N = full_ret.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N))
    turnover = np.zeros(T)
    for i in range(T):
        if i in targets:
            w_old = w.copy()
            cash_w_old = 1.0 - w_old.sum()
            ov = np.nan_to_num(overnight[i])
            rp1 = float(w_old @ ov + cash_w_old * rf[i])
            w_mid = w_old * (1.0 + ov) / (1.0 + rp1) if 1.0 + rp1 > 0 else w_old
            tgt = targets[i]
            c = cost_bps[i] / 1e4
            cost = float(np.abs(tgt - w_mid) @ c)
            turnover[i] = float(np.abs(tgt - w_mid).sum())
            it = np.nan_to_num(intraday[i])
            rp2 = float(tgt @ it)
            w = tgt * (1.0 + it) / (1.0 + rp2) if 1.0 + rp2 > 0 else tgt
            out[i] = (1.0 + rp1) * (1.0 + rp2) - 1.0 - cost
        else:
            r = np.nan_to_num(full_ret[i])
            cash_w = 1.0 - w.sum()
            rp = float(w @ r + cash_w * rf[i])
            if 1.0 + rp > 0:
                w = w * (1.0 + r) / (1.0 + rp)
            out[i] = rp
        held[i] = w
    return out, held, turnover


def simulate_close_exec(rets: np.ndarray, rf: np.ndarray, decide, cost_bps_fn) -> tuple[np.ndarray, np.ndarray]:
    """Close-based execution (protocol §3's own literal BM2/BM3 text): target set
    'at close t+1', taking effect the following day. Adapted from
    run_alt_premia_evaluation.py's ``simulate`` with a per-day cost callback."""
    T, N = rets.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N))
    for i in range(T):
        r = np.nan_to_num(rets[i])
        cash_w = 1.0 - w.sum()
        rp = float(w @ r + cash_w * rf[i])
        if 1.0 + rp > 0:
            w = w * (1.0 + r) / (1.0 + rp)
        tgt = decide(i, w)
        cost = 0.0
        if tgt is not None:
            tgt = np.asarray(tgt, dtype=float)
            cost = float(np.abs(tgt - w) @ (cost_bps_fn(i) / 1e4))
            w = tgt
        out[i] = rp - cost
        held[i] = w
    return out, held


def window(s: pd.Series, start: str, end: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# Return matrices (full calendar 1993-2026; targets are only ever populated at
# execution dates inside WINDOW, so the pre-WINDOW portion is simply 100% cash
# and is dropped by ``window()`` at report time -- matches alt_premia's own
# build-full-then-window convention).
# ---------------------------------------------------------------------------

def build_return_matrices(assets: dict[str, Asset], cal: pd.DatetimeIndex):
    full_ret = pd.DataFrame({t: assets[t].c2c for t in UNIVERSE}, index=cal).to_numpy()
    adj_close_prev = pd.DataFrame({t: assets[t].adj_close.shift(1) for t in UNIVERSE}, index=cal).to_numpy()
    adj_open = pd.DataFrame({t: assets[t].adj_open for t in UNIVERSE}, index=cal).to_numpy()
    adj_close = pd.DataFrame({t: assets[t].adj_close for t in UNIVERSE}, index=cal).to_numpy()
    overnight, intraday = split_day_return(adj_close_prev, adj_open, adj_close)
    return full_ret, overnight, intraday


def month_end_sessions_in_window(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    me = month_end_mask(cal)
    dates = cal[me]
    start, end = pd.Timestamp(prereg.WINDOW["start"]), pd.Timestamp(prereg.WINDOW["end"])
    return [d for d in dates if start <= d <= end]


def _next_pos_after(cal: pd.DatetimeIndex, t: pd.Timestamp, pos_of: dict) -> int | None:
    p = pos_of[t] + 1
    return p if p < len(cal) else None


def build_targets(assets: dict[str, Asset], cal: pd.DatetimeIndex, month_ends: list[pd.Timestamp],
                   *, all_holdings: bool, formation_months: int = 0, skip_months: int = 0,
                   selection: str = "top_tercile", top_n: int | None = None) -> dict:
    pos_of = {d: i for i, d in enumerate(cal)}
    targets: dict[int, np.ndarray] = {}
    cost_bps: dict[int, np.ndarray] = {}
    diag = []
    n_signal_gaps = 0
    for t in month_ends:
        elig = eligibility_at(assets, t)
        if all_holdings:
            held = list(elig) if len(elig) >= prereg.ELIGIBILITY["n_min_for_trading"] else []
            n_sig_ok = len(elig)
        else:
            sig = {}
            for tk in elig:
                v = signal_at(assets[tk].adj_close, t, formation_months, skip_months)
                if np.isfinite(v):
                    sig[tk] = v
                else:
                    n_signal_gaps += 1
            held = select_holdings(sig, elig, selection, prereg.ELIGIBILITY["n_min_for_trading"], top_n)
            n_sig_ok = len(sig)
        exec_pos = _next_pos_after(cal, t, pos_of)
        if exec_pos is None:
            continue
        w = np.zeros(N_UNIV)
        cb = np.full(N_UNIV, prereg.COSTS["etf_bps_per_side_adv_lt_50m"])
        if held:
            wt = 1.0 / len(held)
            for tk in held:
                w[IDX[tk]] = wt
        for tk in elig:
            adv = assets[tk].adv20.asof(t)
            cb[IDX[tk]] = cost_bps_for_adv(adv, prereg.COSTS["adv_threshold_usd"],
                                           prereg.COSTS["etf_bps_per_side_adv_ge_50m"],
                                           prereg.COSTS["etf_bps_per_side_adv_lt_50m"])
        targets[exec_pos] = w
        cost_bps[exec_pos] = cb
        diag.append({"t": str(t.date()), "n_eligible": len(elig), "n_signal_ok": n_sig_ok,
                     "n_hold": len(held), "held": held})
    return {"targets": targets, "cost_bps": cost_bps, "diag": diag, "n_signal_gaps": n_signal_gaps}


# ---------------------------------------------------------------------------
# BM1 / BM2 / BM3 -- close-based execution, protocol §3's OWN literal text
# (issue_2 above). Cost: SPY/IEF always 3bps (large, liquid); VFITX leg 0bps
# (issue_3). Adapted closely from run_alt_premia_evaluation.py's bm1/bm2/bm3.
# ---------------------------------------------------------------------------
ETF_BPS = prereg.COSTS["etf_bps_per_side_adv_ge_50m"]


def bm1(cal: pd.DatetimeIndex, spy: Asset, rf: np.ndarray) -> pd.Series:
    r = spy.c2c.to_numpy()[:, None]
    first = int(np.argmax(np.isfinite(r[:, 0])))

    def decide(i, w):
        return np.array([1.0]) if i == first else None
    net, _ = simulate_close_exec(np.nan_to_num(r), rf, decide, lambda i: np.array([ETF_BPS]))
    return pd.Series(net, cal, name="BM1_SPY")


def bm2(cal: pd.DatetimeIndex, spy: Asset, ief: Asset, vfitx: Asset, rf: np.ndarray) -> pd.Series:
    bond = splice_bond_return(ief.c2c, vfitx.c2c)
    r = pd.DataFrame({"SPY": spy.c2c, "BOND": bond}, index=cal)
    me = month_end_mask(cal)
    ok = r.notna().all(axis=1).to_numpy()
    is_ief = cal >= IEF_START
    r_arr = np.nan_to_num(r.to_numpy())

    def decide(i, w):
        if i >= 1 and me[i - 1] and ok[i - 1]:
            return np.array([0.6, 0.4])
        return None

    def cost_fn(i):
        return np.array([ETF_BPS, ETF_BPS if is_ief[i] else 0.0])
    net, _ = simulate_close_exec(r_arr, rf, decide, cost_fn)
    return pd.Series(net, cal, name="BM2_60_40")


def bm3(cal: pd.DatetimeIndex, spy: Asset, rf: np.ndarray) -> pd.Series:
    spec = prereg.BENCHMARKS["BM3_SPY_VT"]
    rs = spy.c2c
    sig = (rs.rolling(spec["vol_window"]).std() * ANN).to_numpy()
    r = np.nan_to_num(rs.to_numpy())[:, None]

    def decide(i, w):
        if i < 1 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        tgt = min(1.0, spec["target_vol"] / sig[i - 1])
        return np.array([tgt]) if abs(tgt - w[0]) > spec["band_abs"] else None
    net, _ = simulate_close_exec(r, rf, decide, lambda i: np.array([ETF_BPS]))
    return pd.Series(net, cal, name="BM3_SPY_VT")


# ---------------------------------------------------------------------------
# Placebo (protocol §4, S1's own rule): random rankings with the same n_hold
# per month on the same rebalance dates, drawn from that month's eligible set.
# Run only for PRIMARY_VARIANT (issue_4: A3 is a primary-variant-only bar).
# ---------------------------------------------------------------------------

def placebo_targets(assets: dict[str, Asset], cal: pd.DatetimeIndex, month_ends: list[pd.Timestamp],
                     n_hold_by_month: dict[str, int], rng: np.random.Generator) -> dict:
    pos_of = {d: i for i, d in enumerate(cal)}
    targets: dict[int, np.ndarray] = {}
    cost_bps: dict[int, np.ndarray] = {}
    for t in month_ends:
        elig = eligibility_at(assets, t)
        n_hold = n_hold_by_month.get(str(t.date()), 0)
        n_hold = min(n_hold, len(elig))
        held = list(rng.choice(elig, size=n_hold, replace=False)) if n_hold > 0 and elig else []
        exec_pos = _next_pos_after(cal, t, pos_of)
        if exec_pos is None:
            continue
        w = np.zeros(N_UNIV)
        cb = np.full(N_UNIV, prereg.COSTS["etf_bps_per_side_adv_lt_50m"])
        if held:
            wt = 1.0 / len(held)
            for tk in held:
                w[IDX[tk]] = wt
        for tk in elig:
            adv = assets[tk].adv20.asof(t)
            cb[IDX[tk]] = cost_bps_for_adv(adv, prereg.COSTS["adv_threshold_usd"],
                                           prereg.COSTS["etf_bps_per_side_adv_ge_50m"],
                                           prereg.COSTS["etf_bps_per_side_adv_lt_50m"])
        targets[exec_pos] = w
        cost_bps[exec_pos] = cb
    return {"targets": targets, "cost_bps": cost_bps}


def _mult_cost(cost_bps: dict[int, np.ndarray], mult: float) -> dict[int, np.ndarray]:
    return {k: v * mult for k, v in cost_bps.items()}


def _run_open_exec(full_ret, overnight, intraday, rf, built: dict, stress: bool = False):
    cb = _mult_cost(built["cost_bps"], prereg.COSTS["stress_multiplier"]) if stress else built["cost_bps"]
    return simulate_open_exec(full_ret, overnight, intraday, rf, built["targets"], cb)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def _gap_bars(c: pd.Series, b: pd.Series, rf: pd.Series, start: str, end: str,
              alpha: float, seed: int) -> dict:
    j = pd.concat([c, b, rf], axis=1, keys=["c", "b", "rf"])
    j = window(j, start, end).dropna()
    ce, be = (j.c - j.rf).to_numpy(), (j.b - j.rf).to_numpy()
    sc, sb = sharpe(ce), sharpe(be)
    gap = sc - sb
    boot = paired_sharpe_gap_boot(ce, be, seed)
    lb, ub = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    return {"sharpe_c": sc, "sharpe_b": sb, "gap": gap, "lb": lb, "ub": ub, "n_days": int(len(j))}


def _gap_point(c: pd.Series, b: pd.Series, rf: pd.Series, start: str, end: str) -> float:
    j = pd.concat([c, b, rf], axis=1, keys=["c", "b", "rf"])
    j = window(j, start, end).dropna()
    if len(j) < 2:
        return float("nan")
    return sharpe((j.c - j.rf).to_numpy()) - sharpe((j.b - j.rf).to_numpy())


def cmd_evaluate(args: argparse.Namespace) -> int:
    fp = prereg.bars_fingerprint()
    assert fp == "287294d0552396299b2e4fbe916a92a9adef564d53dbb6458c4a591941c8b359", \
        f"fingerprint mismatch: prereg file changed since freeze ({fp})"
    log.info("fingerprint %s (frozen %s)", fp, prereg.PREREGISTERED_AT)
    cal = equity_calendar()
    assets = load_universe(cal)
    dtb3 = load_dtb3(cal)
    rf_s = cash_return_series(assets["BIL"].c2c, dtb3)
    rf = rf_s.to_numpy()
    full_ret, overnight, intraday = build_return_matrices(assets, cal)
    month_ends = month_end_sessions_in_window(cal)
    w_start, w_end = prereg.WINDOW["start"], prereg.WINDOW["end"]
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]

    # Candidate variants
    variant_series: dict[str, pd.Series] = {}
    variant_series_stress: dict[str, pd.Series] = {}
    variant_built: dict[str, dict] = {}
    for name, spec in prereg.VARIANTS.items():
        built = build_targets(assets, cal, month_ends, all_holdings=False,
                              formation_months=spec["formation_months"], skip_months=spec["skip_months"],
                              selection=spec["selection"], top_n=spec.get("top_n"))
        variant_built[name] = built
        net, held, turnover = _run_open_exec(full_ret, overnight, intraday, rf, built)
        variant_series[name] = pd.Series(net, cal, name=name)
        net_s, *_ = _run_open_exec(full_ret, overnight, intraday, rf, built, stress=True)
        variant_series_stress[name] = pd.Series(net_s, cal, name=name)
        if name == prereg.PRIMARY_VARIANT:
            primary_held, primary_turnover = held, turnover

    # Primary benchmark (equal-weight of the same point-in-time eligible universe)
    bm_primary_built = build_targets(assets, cal, month_ends, all_holdings=True)
    net, _, _ = _run_open_exec(full_ret, overnight, intraday, rf, bm_primary_built)
    primary_bm = pd.Series(net, cal, name="EW_universe_bh")
    net_s, *_ = _run_open_exec(full_ret, overnight, intraday, rf, bm_primary_built, stress=True)
    primary_bm_stress = pd.Series(net_s, cal, name="EW_universe_bh")

    # BM1-3 (close-based execution, protocol's own literal §3 text)
    base_bms = {"BM1_SPY": bm1(cal, assets["SPY"], rf), "BM2_60_40": bm2(cal, assets["SPY"], assets["IEF"],
               assets["VFITX"], rf), "BM3_SPY_VT": bm3(cal, assets["SPY"], rf)}

    def _bm_with_stress_mult(mult: float) -> dict[str, pd.Series]:
        spy, ief, vfitx = assets["SPY"], assets["IEF"], assets["VFITX"]
        b1 = bm1(cal, spy, rf)
        if mult != 1.0:
            # Re-run with doubled cost by scaling the cost closures directly.
            r = spy.c2c.to_numpy()[:, None]
            first = int(np.argmax(np.isfinite(r[:, 0])))
            net1, _ = simulate_close_exec(np.nan_to_num(r), rf, lambda i, w: (np.array([1.0]) if i == first else None),
                                          lambda i: np.array([ETF_BPS * mult]))
            b1 = pd.Series(net1, cal, name="BM1_SPY")
            bond = splice_bond_return(ief.c2c, vfitx.c2c)
            rdf = pd.DataFrame({"SPY": spy.c2c, "BOND": bond}, index=cal)
            me = month_end_mask(cal)
            ok = rdf.notna().all(axis=1).to_numpy()
            is_ief = cal >= IEF_START
            r_arr = np.nan_to_num(rdf.to_numpy())

            def decide2(i, w):
                if i >= 1 and me[i - 1] and ok[i - 1]:
                    return np.array([0.6, 0.4])
                return None
            net2, _ = simulate_close_exec(r_arr, rf, decide2,
                                          lambda i: np.array([ETF_BPS * mult, (ETF_BPS if is_ief[i] else 0.0) * mult]))
            b2 = pd.Series(net2, cal, name="BM2_60_40")
            spec = prereg.BENCHMARKS["BM3_SPY_VT"]
            sig = (spy.c2c.rolling(spec["vol_window"]).std() * ANN).to_numpy()
            r3 = np.nan_to_num(spy.c2c.to_numpy())[:, None]

            def decide3(i, w):
                if i < 1 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
                    return None
                tgt = min(1.0, spec["target_vol"] / sig[i - 1])
                return np.array([tgt]) if abs(tgt - w[0]) > spec["band_abs"] else None
            net3, _ = simulate_close_exec(r3, rf, decide3, lambda i: np.array([ETF_BPS * mult]))
            b3 = pd.Series(net3, cal, name="BM3_SPY_VT")
            return {"BM1_SPY": b1, "BM2_60_40": b2, "BM3_SPY_VT": b3}
        return {"BM1_SPY": b1, "BM2_60_40": bm2(cal, spy, ief, vfitx, rf), "BM3_SPY_VT": bm3(cal, spy, rf)}

    stress_bms = _bm_with_stress_mult(prereg.COSTS["stress_multiplier"])

    primary = variant_series[prereg.PRIMARY_VARIANT]
    primary_stress = variant_series_stress[prereg.PRIMARY_VARIANT]

    # ---- A1 / D (vs primary benchmark only) ----
    a1d = _gap_bars(primary, primary_bm, rf_s, w_start, w_end, alpha, seed=prereg.SEED + 101)
    A1 = bool(a1d["gap"] > 0 and a1d["lb"] > 0)
    D = bool(a1d["ub"] < 0)

    # ---- A4 (both halves, vs primary benchmark + BM1-3) ----
    h1_start, h1_end, h2_start, h2_end = half_windows(w_start, w_end, prereg.WINDOW["midpoint_date"])
    a4_rows = {}
    for bname, bser in {**{"EW_universe_bh": primary_bm}, **base_bms}.items():
        g1 = _gap_point(primary, bser, rf_s, h1_start, h1_end)
        g2 = _gap_point(primary, bser, rf_s, h2_start, h2_end)
        a4_rows[bname] = {"gap_half1": g1, "gap_half2": g2, "pass": bool(g1 > 0 and g2 > 0)}
    A4 = all(v["pass"] for v in a4_rows.values())

    # ---- A5 (2x costs, vs primary benchmark + BM1-3) ----
    a5_rows = {}
    for bname, bser in {**{"EW_universe_bh": primary_bm_stress}, **stress_bms}.items():
        g = _gap_point(primary_stress, bser, rf_s, w_start, w_end)
        a5_rows[bname] = {"gap_stress": g, "pass": bool(g > 0)}
    A5 = all(v["pass"] for v in a5_rows.values())

    # ---- A2 / DSR ----
    trial_sharpes = []
    for name in prereg.VARIANTS:
        j = pd.concat([variant_series[name], rf_s], axis=1, keys=["c", "rf"])
        j = window(j, w_start, w_end).dropna()
        ex = (j.c - j.rf).to_numpy()
        ex = ex[np.isfinite(ex)]
        trial_sharpes.append(float(ex.mean() / ex.std(ddof=1)) if ex.std(ddof=1) > 0 else 0.0)
    primary_ex_full = window(pd.concat([primary, rf_s], axis=1, keys=["c", "rf"]), w_start, w_end).dropna()
    primary_ex = (primary_ex_full.c - primary_ex_full.rf).to_numpy()
    dsr = float(deflated_sharpe(primary_ex, np.array(trial_sharpes), prior_trials=prereg.DSR["prior_trials"]))
    A2 = bool(dsr > 0.95)

    # ---- A6 / PBO (7 cols: 4 variants + BM1-3, excess over the PRIMARY BENCHMARK; issue_5) ----
    pbo_frame = pd.DataFrame({**{k: v for k, v in variant_series.items()}, **base_bms}, index=cal)
    pbo_frame = window(pbo_frame, w_start, w_end)
    pbo_ex = pbo_frame.sub(window(primary_bm, w_start, w_end), axis=0).dropna()
    pbo = float(cscv_pbo(pbo_ex.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    A6 = bool(pbo < 0.50)

    # ---- A3 / placebo (primary variant only, issue_4) ----
    n_hold_by_month = {d["t"]: d["n_hold"] for d in variant_built[prereg.PRIMARY_VARIANT]["diag"]}
    prng = np.random.default_rng(prereg.PLACEBO["seed"])
    placebo_sharpes = []
    for draw in range(prereg.PLACEBO["n_draws"]):
        pt = placebo_targets(assets, cal, month_ends, n_hold_by_month, prng)
        net, _, _ = simulate_open_exec(full_ret, overnight, intraday, rf, pt["targets"], pt["cost_bps"])
        s = pd.Series(net, cal)
        j = window(pd.concat([s, rf_s], axis=1, keys=["c", "rf"]), w_start, w_end).dropna()
        ex = (j.c - j.rf).to_numpy()
        placebo_sharpes.append(sharpe(ex))
        if draw % 100 == 0:
            log.info("placebo draw %d/%d", draw, prereg.PLACEBO["n_draws"])
    placebo_sharpes = np.array(placebo_sharpes)
    placebo_p95 = float(np.nanpercentile(placebo_sharpes, prereg.PLACEBO["pass_percentile"]))
    primary_sharpe_full = sharpe(primary_ex)
    A3 = bool(primary_sharpe_full > placebo_p95)

    # ---- A7: PENDING ----
    bars = {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6}
    bars_without_a7 = dict(bars)
    tier_b = {"B": bool(a1d["gap"] > 0 and A3 and A4 and A5)}
    provisional_tier = prereg.classify(bars_without_a7, D, tier_b)
    tier = "A (conditional on A7)" if provisional_tier == "A" else provisional_tier

    # ---- Sanity checks ----
    spy_w = window(base_bms["BM1_SPY"], w_start, w_end).to_numpy()
    bm2_w = window(base_bms["BM2_60_40"], w_start, w_end).to_numpy()
    sanity = {
        "spy_bh": {"cagr": cagr(spy_w), "vol": float(np.nanstd(spy_w, ddof=1) * ANN), "max_dd": max_drawdown(spy_w)},
        "sixty_forty": {"cagr": cagr(bm2_w), "vol": float(np.nanstd(bm2_w, ddof=1) * ANN), "max_dd": max_drawdown(bm2_w)},
    }
    all_series_for_sanity = {"primary_variant": primary, "EW_universe_bh": primary_bm, **base_bms}
    big_move_rows = []
    for name, s in all_series_for_sanity.items():
        sw = window(s, w_start, w_end)
        big = sw[sw.abs() > 0.15]
        for d, v in big.items():
            known = (pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-30")) or \
                   (pd.Timestamp("2020-02-15") <= d <= pd.Timestamp("2020-04-30"))
            big_move_rows.append({"series": name, "date": str(d.date()), "return": float(v),
                                  "explained": "2008-09 GFC window" if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-30")
                                  else ("2020 COVID crash window" if pd.Timestamp("2020-02-15") <= d <= pd.Timestamp("2020-04-30")
                                       else "UNEXPLAINED -- needs manual review")})
    nan_days = {name: int(window(s, w_start, w_end).isna().sum()) for name, s in all_series_for_sanity.items()}
    ann_turnover = float(window(pd.Series(primary_turnover, cal), w_start, w_end).sum() / (
        (pd.Timestamp(w_end) - pd.Timestamp(w_start)).days / 365.25))
    sanity["n_big_moves_gt_15pct"] = len(big_move_rows)
    sanity["big_moves"] = big_move_rows
    sanity["nan_days_by_series"] = nan_days
    sanity["candidate_annualised_one_way_turnover"] = ann_turnover
    sanity["n_12_1_signal_gaps_by_variant"] = {k: v["n_signal_gaps"] for k, v in variant_built.items()}

    # ---- Report table (primary variant, primary benchmark, BM1-3) ----
    def row(s: pd.Series, rf_series: pd.Series) -> dict:
        j = window(pd.concat([s, rf_series], axis=1, keys=["c", "rf"]), w_start, w_end).dropna()
        raw = j.c.to_numpy()
        ex = (j.c - j.rf).to_numpy()
        return {"sharpe_excess_of_cash": sharpe(ex), "cagr": cagr(raw), "vol": float(np.std(raw, ddof=1) * ANN),
               "max_dd": max_drawdown(raw), "n_days": int(len(j))}

    table = {
        "primary_variant": row(primary, rf_s), "EW_universe_bh": row(primary_bm, rf_s),
        "BM1_SPY": row(base_bms["BM1_SPY"], rf_s), "BM2_60_40": row(base_bms["BM2_60_40"], rf_s),
        "BM3_SPY_VT": row(base_bms["BM3_SPY_VT"], rf_s),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
        "prereg_at": prereg.PREREGISTERED_AT, "data_end": prereg.DATA_END, "window": prereg.WINDOW,
        "primary_variant": prereg.PRIMARY_VARIANT, "alpha_one_sided": alpha,
        "table": table,
        "bars": bars, "tier_D_condition": D, "tier_B": tier_b, "tier": tier,
        "A7_status": "PENDING (independent recompute not yet run)",
        "a1_detail": a1d, "a4_detail": a4_rows, "a5_detail": a5_rows,
        "dsr": dsr, "dsr_trials": prereg.N_VARIANTS, "dsr_prior_trials": prereg.DSR["prior_trials"],
        "trial_daily_sharpes": trial_sharpes, "pbo": pbo,
        "placebo_p95": placebo_p95, "placebo_median": float(np.nanmedian(placebo_sharpes)),
        "primary_sharpe_full_window": primary_sharpe_full,
        "sanity": sanity, "prereg_issues": PREREG_ISSUES,
    }

    if tier.startswith("A") or tier == "B":
        log.info("tier %s: running post-pass audits (2-day lag, per-year Sharpe gap table)", tier)
        lagged_built = build_targets(assets, cal, [t for t in month_ends], all_holdings=False,
                                     formation_months=prereg.VARIANTS[prereg.PRIMARY_VARIANT]["formation_months"],
                                     skip_months=prereg.VARIANTS[prereg.PRIMARY_VARIANT]["skip_months"],
                                     selection=prereg.VARIANTS[prereg.PRIMARY_VARIANT]["selection"],
                                     top_n=prereg.VARIANTS[prereg.PRIMARY_VARIANT].get("top_n"))
        # 2-day lag: shift the execution position by one extra trading day.
        lagged_targets = {k + 1: v for k, v in lagged_built["targets"].items() if k + 1 < len(cal)}
        lagged_cost = {k + 1: v for k, v in lagged_built["cost_bps"].items() if k + 1 < len(cal)}
        net_lag, _, _ = simulate_open_exec(full_ret, overnight, intraday, rf, lagged_targets, lagged_cost)
        lag_series = pd.Series(net_lag, cal, name="primary_2day_lag")
        report["post_pass_audit"] = {
            "signal_lag_2day_sharpe_gap_vs_primary_bm": _gap_point(lag_series, primary_bm, rf_s, w_start, w_end),
            "per_year_sharpe_gap": {
                str(y): _gap_point(primary, primary_bm, rf_s, f"{y}-01-01", f"{y}-12-31")
                for y in range(pd.Timestamp(w_start).year, pd.Timestamp(w_end).year + 1)
            },
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    returns_frame = pd.DataFrame({**{k: v for k, v in variant_series.items()}, "EW_universe_bh": primary_bm,
                                  **base_bms, "rf": rf_s}, index=cal)
    window(returns_frame, w_start, w_end).astype("float32").to_parquet(out_dir / "returns.parquet")
    pd.DataFrame({"placebo_sharpe": placebo_sharpes}).to_parquet(out_dir / "placebo_sharpes.parquet")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "S1", "entries": []}
        ledger["entries"].append({
            "date": datetime.now(timezone.utc).date().isoformat(), "fingerprint": fp,
            "n_trials": prereg.N_VARIANTS, "trials": list(prereg.VARIANTS),
            "trial_daily_sharpes": trial_sharpes, "tier": tier,
        })
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(ledger, indent=2))
        log.info("ledger updated: %s (cumulative %d)", LEDGER, ledger["cumulative_trials"])
    log.info("==> S1 tier: %s  bars=%s  D=%s  B=%s", tier, bars, D, tier_b)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--out", required=True)
    p.add_argument("--report")
    p.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
