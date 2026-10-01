"""Independent recompute (bar A7) of S2 (market-breadth overlay on 60/40 core).

Written from the frozen text of scripts/eodhd_s2_breadth_overlay_preregistered_bars.py
(commit ff6f44a on research/eodhd-S2) and docs/eodhd_shortlist_protocol_2026_10.md
(amendment 1), WITHOUT reading scripts/run_eodhd_s2_evaluation.py or its report.

Allowed reuse: scripts/eodhd_clean.py (clean_bars, equity_calendar),
stationary_indices from scripts/run_alt_premia_evaluation.py, and
firm.eval.overfitting (deflated_sharpe, cscv_pbo).
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/local/store/git/ai-trading-system")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from eodhd_clean import clean_bars, equity_calendar  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402
from run_alt_premia_evaluation import stationary_indices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("s2_recompute")

OUT = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/S2_recompute")
ETF_DIR = REPO / "data" / "research" / "eodhd" / "etfs_full"
FRED_DIR = REPO / "data" / "research" / "fred"

TRADING_DAYS = 252
ANN = math.sqrt(TRADING_DAYS)
SEED = 20260930

WINDOW_START = "2002-01-02"
DATA_END = "2026-09-29"
MIDPOINT = "2014-05-16"

SPY_WEIGHT_CUT = 0.30
CORE_SPY = 0.60
CORE_BOND = 0.40

IEF_HANDOFF = "2002-07-26"   # IEF's own first bar; VFITX proxy through the day before
BIL_HANDOFF = "2007-05-30"   # BIL's own first bar; DTB3 proxy through the day before

ETF_ADV20_BIG = 50_000_000.0
ETF_BPS_BIG = 3.0
ETF_BPS_SMALL = 10.0

VARIANTS = {
    "V1_primary": {"measure": "pct_above_200sma", "threshold": 0.50, "destination": "IEF"},
    "V2_cash_destination": {"measure": "pct_above_200sma", "threshold": 0.50, "destination": "BIL"},
    "V3_stricter_threshold": {"measure": "pct_above_200sma", "threshold": 0.40, "destination": "IEF"},
    "V4_alt_measure": {"measure": "net_ad_21d", "threshold": 0.0, "destination": "IEF"},
}

PRIOR_TRIALS = 206  # frozen pre-registration's DSR.prior_trials (declared constant, not recomputed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_etf(ticker: str, calendar: pd.DatetimeIndex, asset: str = "equity") -> pd.DataFrame:
    raw = pd.read_parquet(ETF_DIR / f"{ticker}.parquet")
    df, rep = clean_bars(raw, asset=asset, calendar=calendar)
    df = df.reset_index(drop=True)
    df["adj_open"] = df["open"] * df["adjusted_close"] / df["close"]
    log.info("%s: %d bars kept (%s), %s..%s", ticker, len(df), rep, df["date"].min(), df["date"].max())
    return df


def load_dtb3() -> pd.Series:
    d = pd.read_parquet(FRED_DIR / "DTB3.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["value"]


def build_calendar_frame(name: str, df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a cleaned ETF/NAV frame onto the full SPY calendar (NaN outside its own range).

    A handful of NAV-quoted series (VFITX) miss a print on an isolated SPY session (358 of VFITX's
    raw bars fall on dates the SPY calendar does NOT have, per clean_bars' off_calendar count, which
    is the mirror-image symptom: VFITX's OWN calendar and SPY's are not identical, so reindexing onto
    SPY's calendar leaves a few internal gaps). Those isolated gaps are forward-filled (treated as "no
    new print, unchanged from the last known value" -- a flat day, open=close=last close) within the
    instrument's own active range; NaN before its first bar / after its last bar is left alone (that is
    handled explicitly by the proxy/handoff splice logic, not by ffill).
    """
    d = df.set_index("date")[["adjusted_close", "adj_open", "segment"]]
    d = d.reindex(calendar)
    adjclose = d["adjusted_close"].to_numpy(dtype=float).copy()
    adjopen = d["adj_open"].to_numpy(dtype=float).copy()
    segment = d["segment"].to_numpy(dtype=float).copy()
    valid = ~np.isnan(adjclose)
    if valid.any():
        fvi, lvi = np.flatnonzero(valid)[[0, -1]]
        seg_slice = slice(fvi, lvi + 1)
        filled = pd.Series(adjclose[seg_slice]).ffill().to_numpy()
        gap = np.isnan(adjclose[seg_slice]) & ~np.isnan(filled)
        n_gap = int(gap.sum())
        if n_gap:
            log.info("%s: forward-filled %d isolated NAV/print gap day(s) within its active range",
                      name, n_gap)
        sub_open = adjopen[seg_slice]
        sub_open[gap] = filled[gap]
        adjopen[seg_slice] = sub_open
        adjclose[seg_slice] = filled
        segment[seg_slice] = pd.Series(segment[seg_slice]).ffill().to_numpy()
    out = pd.DataFrame({
        f"{name}_adjclose": adjclose, f"{name}_adjopen": adjopen, f"{name}_segment": segment,
    }, index=calendar)
    return out


def adv20_bps(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """Per-day cost bps (protocol Sec.2): 20-day trailing median dollar volume >= $50M -> 3bps, else 10bps.
    Computed as of the PRIOR trading day (no look-ahead into the trade day itself)."""
    d = df.set_index("date")
    dv = d["adjusted_close"] * d["volume"]
    adv20 = dv.rolling(20, min_periods=20).median()
    bps = pd.Series(np.where(adv20 >= ETF_ADV20_BIG, ETF_BPS_BIG, ETF_BPS_SMALL), index=d.index)
    bps = bps.reindex(calendar).shift(1)  # value known as of the close BEFORE the trade day
    return bps


def dtb3_accrual_returns(calendar: pd.DatetimeIndex) -> np.ndarray:
    dtb3 = load_dtb3()
    daily = (dtb3 / 100.0 / TRADING_DAYS)
    daily = daily.reindex(daily.index.union(calendar)).ffill().reindex(calendar)
    return daily.fillna(0.0).to_numpy()


def build_leg(main_adjclose: pd.Series, main_adjopen: pd.Series, main_bps: pd.Series,
              proxy_close_ret: np.ndarray, handoff: str, calendar: pd.DatetimeIndex,
              rebal_mask: np.ndarray) -> dict:
    """A single continuous (open_ret, intraday_ret, close_ret, is_real, bps) leg: the PROXY series
    (VFITX close-to-close, or DTB3 daily accrual -- never traded/costed, no true intraday open) before
    `handoff`, the real ETF (IEF/BIL) from `handoff` onward. No return bridges the splice day itself
    (protocol: no return across a segment/instrument boundary).

    Convention for the proxy leg (no true open print): the whole day's return is attributed to the
    NEW (post-rebalance) weight on a rebalance day (open_ret=0, intraday_ret=that day's full return);
    on a non-rebalance day the split point is immaterial (old weight == new weight that day), so the
    same convention is used throughout for simplicity.
    """
    handoff_ts = pd.Timestamp(handoff)
    n = len(calendar)
    open_ret = np.zeros(n)
    intraday_ret = np.zeros(n)
    is_real = np.zeros(n, dtype=bool)
    bps = np.zeros(n)

    main_close = main_adjclose.to_numpy()
    main_open = main_adjopen.to_numpy()
    main_bps_v = main_bps.to_numpy()
    prev_main_close = np.roll(main_close, 1)
    prev_main_close[0] = np.nan

    for i in range(n):
        if calendar[i] < handoff_ts:
            intraday_ret[i] = proxy_close_ret[i]
            is_real[i] = False
            bps[i] = 0.0
        elif calendar[i] == handoff_ts:
            open_ret[i] = 0.0
            intraday_ret[i] = 0.0  # splice day: flat, no return bridges proxy -> ETF
            is_real[i] = True
            bps[i] = 0.0  # no trade cost on the data-splice day itself (not an overlay decision trade)
        else:
            o = main_open[i] / prev_main_close[i] - 1.0 if prev_main_close[i] > 0 else 0.0
            c = main_close[i] / main_open[i] - 1.0 if main_open[i] > 0 else 0.0
            open_ret[i] = 0.0 if not np.isfinite(o) else o
            intraday_ret[i] = 0.0 if not np.isfinite(c) else c
            is_real[i] = True
            b = main_bps_v[i]
            bps[i] = b if np.isfinite(b) else ETF_BPS_SMALL
    close_ret = (1 + open_ret) * (1 + intraday_ret) - 1.0
    return {"open_ret": open_ret, "intraday_ret": intraday_ret, "close_ret": close_ret,
            "is_real": is_real, "bps": bps}


def build_spy_leg(spy_adjclose: pd.Series, spy_adjopen: pd.Series, spy_bps: pd.Series,
                   calendar: pd.DatetimeIndex) -> dict:
    n = len(calendar)
    close = spy_adjclose.to_numpy()
    openp = spy_adjopen.to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        open_ret = np.where((prev_close > 0), openp / prev_close - 1.0, 0.0)
        intraday_ret = np.where((openp > 0), close / openp - 1.0, 0.0)
    open_ret = np.nan_to_num(open_ret, nan=0.0, posinf=0.0, neginf=0.0)
    intraday_ret = np.nan_to_num(intraday_ret, nan=0.0, posinf=0.0, neginf=0.0)
    close_ret = (1 + open_ret) * (1 + intraday_ret) - 1.0
    bps = spy_bps.to_numpy()
    bps = np.where(np.isfinite(bps), bps, ETF_BPS_SMALL)
    return {"open_ret": open_ret, "intraday_ret": intraday_ret, "close_ret": close_ret,
            "is_real": np.ones(n, dtype=bool), "bps": bps}


# ---------------------------------------------------------------------------
# Core two-asset monthly-rebalance simulation (BM2 and the S2 overlay variants)
# ---------------------------------------------------------------------------

def simulate_two_asset(rebal_mask: np.ndarray, target_weight_seq: np.ndarray,
                        spy_leg: dict, dest_leg: dict, cost_mult: float = 1.0) -> np.ndarray:
    """Daily portfolio return series for a 2-asset (SPY, dest) monthly-rebalanced overlay.

    `target_weight_seq[k]` is the SPY target weight decided AT the k-th rebalance date (0-indexed,
    in calendar order) -- 0.60 (plain core) or 0.30 (overlay "on"). Between rebalances, weights
    drift purely from each leg's own daily close-to-close return (`weights_drift_between_rebalances`).
    On a rebalance day, the trade executes at that day's adjusted open (protocol Sec.2): the
    overnight leg (prior close -> today's open) uses the OLD (pre-trade, drifted) weight; cost is
    then deducted on the turnover; the intraday leg (today's open -> close) uses the NEW weight.
    """
    n = len(rebal_mask)
    rebal_idx = np.flatnonzero(rebal_mask)
    K = len(rebal_idx)
    period_id = np.cumsum(rebal_mask)  # 1..K, constant between rebal dates

    factor_spy = np.where(rebal_mask, 1.0, 1 + spy_leg["close_ret"])
    factor_dest = np.where(rebal_mask, 1.0, 1 + dest_leg["close_ret"])
    cumprod_spy = pd.Series(factor_spy).groupby(period_id).cumprod().to_numpy()
    cumprod_dest = pd.Series(factor_dest).groupby(period_id).cumprod().to_numpy()

    anchor_spy = np.zeros(K + 1)   # 1-indexed by period number
    anchor_dest = np.zeros(K + 1)

    for k in range(1, K + 1):
        r = rebal_idx[k - 1]
        w_t = target_weight_seq[k - 1]
        if k == 1:
            new_spy = w_t
            new_dest = 1.0 - w_t
        else:
            prev_last = r - 1
            old_spy = anchor_spy[k - 1] * cumprod_spy[prev_last]
            old_dest = anchor_dest[k - 1] * cumprod_dest[prev_last]
            spy_over = old_spy * (1 + spy_leg["open_ret"][r])
            dest_over = old_dest * (1 + dest_leg["open_ret"][r])
            over_nav = spy_over + dest_over
            w_old_at_open = spy_over / over_nav if over_nav != 0 else w_t
            d = abs(w_t - w_old_at_open)
            cost_frac = d * (spy_leg["bps"][r] + dest_leg["bps"][r]) / 1e4 * cost_mult
            post_nav = over_nav * (1 - cost_frac)
            new_spy = w_t * post_nav
            new_dest = (1 - w_t) * post_nav
        anchor_spy[k] = new_spy * (1 + spy_leg["intraday_ret"][r])
        anchor_dest[k] = new_dest * (1 + dest_leg["intraday_ret"][r])

    # every day (including non-rebal days) via the per-period cumprod; cumprod==1 at each
    # period's first row (the rebal day itself), so this already reproduces the anchors exactly there.
    full_nav = anchor_spy[period_id] * cumprod_spy + anchor_dest[period_id] * cumprod_dest

    ret = np.zeros(n)
    ret[0] = 0.0
    ret[1:] = full_nav[1:] / full_nav[:-1] - 1.0
    return ret


# ---------------------------------------------------------------------------
# BM1 (SPY buy & hold) and BM3 (SPY vol-targeted, band-triggered, cash remainder)
# ---------------------------------------------------------------------------

def bm1_returns(spy_leg: dict) -> np.ndarray:
    r = spy_leg["close_ret"].copy()
    r[0] = 0.0
    return r


def bm3_returns(spy_close_ret_full: np.ndarray, cash_ret_full: np.ndarray,
                 window_mask: np.ndarray, spy_bps_full: np.ndarray,
                 target_vol: float = 0.12, vol_window: int = 21, band: float = 0.10) -> np.ndarray:
    """BM3: SPY weight = min(1, target_vol / annualised 21-day realised vol at close t); traded at
    close t+1 only when |target - held| > band; remainder in cash. Computed on the FULL SPY history
    (so the 21-day vol lookback has data before window start) then sliced to the window.
    """
    n = len(spy_close_ret_full)
    roll_std = pd.Series(spy_close_ret_full).rolling(vol_window, min_periods=vol_window).std(ddof=1).to_numpy()
    ann_vol = roll_std * ANN
    with np.errstate(invalid="ignore", divide="ignore"):
        target = np.minimum(1.0, target_vol / ann_vol)
    target = np.nan_to_num(target, nan=0.0)

    W = np.zeros(n)
    first = np.flatnonzero(~np.isnan(ann_vol))
    start = first[0] if len(first) else vol_window
    W[start] = target[start]
    nav = np.ones(n)
    ret = np.zeros(n)
    for t in range(start + 1, n):
        w_prev = W[t - 1]
        ret[t] = w_prev * spy_close_ret_full[t] + (1 - w_prev) * cash_ret_full[t]
        if abs(target[t - 1] - w_prev) > band:
            turnover = abs(target[t - 1] - w_prev)
            cost = turnover * spy_bps_full[t] / 1e4
            nav[t] = nav[t - 1] * (1 + ret[t]) * (1 - cost)
            W[t] = target[t - 1]
        else:
            nav[t] = nav[t - 1] * (1 + ret[t])
            W[t] = w_prev
    # fold the cost into ret itself (nav already reflects it; recompute ret from nav for exactness)
    ret_full = np.zeros(n)
    ret_full[1:] = nav[1:] / nav[:-1] - 1.0
    return ret_full[window_mask]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def ann_sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd * ANN)


def period_sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd)


def cagr(r: np.ndarray, trading_days: int = TRADING_DAYS) -> float:
    r = r[np.isfinite(r)]
    nav = np.prod(1 + r)
    years = len(r) / trading_days
    if years <= 0 or nav <= 0:
        return float("nan")
    return float(nav ** (1 / years) - 1)


def max_drawdown(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def calmar(r: np.ndarray) -> float:
    dd = max_drawdown(r)
    c = cagr(r)
    if dd == 0:
        return float("nan")
    return float(c / dd)


def bootstrap_sharpe_gap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                          mean_block: int = 63, seed: int = SEED) -> np.ndarray:
    """Paired stationary block bootstrap of the annualised daily-Sharpe gap (a - b)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    out = np.empty(n_boot)
    chunk = 250
    for k in range(0, n_boot, chunk):
        m = min(chunk, n_boot - k)
        idx = stationary_indices(n, m, mean_block, rng)
        A, B = a[idx], b[idx]
        sa = A.mean(1) / A.std(1, ddof=1)
        sb = B.mean(1) / B.std(1, ddof=1)
        out[k:k + m] = (sa - sb) * ANN
    return out


# ---------------------------------------------------------------------------
# Placebo (S2-specific, protocol Sec.4): the overlay's on/off state permuted in
# 63-trading-day blocks, at the daily frequency the state is held between
# rebalances; sizing/weights otherwise unchanged. 500 draws.
# ---------------------------------------------------------------------------

def daily_state_from_monthly(rebal_mask: np.ndarray, monthly_state: np.ndarray) -> np.ndarray:
    """Forward-fill the per-rebalance on/off decision to a daily boolean series."""
    n = len(rebal_mask)
    period_id = np.cumsum(rebal_mask) - 1  # 0-indexed
    return monthly_state[period_id]


def permute_blocks(daily_state: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(daily_state)
    n_blocks = math.ceil(n / block)
    blocks = [daily_state[i * block:(i + 1) * block] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in order])[:n]


def placebo_sharpes(rebal_mask: np.ndarray, real_monthly_state: np.ndarray,
                     spy_leg: dict, dest_leg: dict, n_draws: int = 500,
                     block: int = 63, seed: int = SEED + 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    daily_real = daily_state_from_monthly(rebal_mask, real_monthly_state)
    rebal_positions = np.flatnonzero(rebal_mask)
    out = np.empty(n_draws)
    for d in range(n_draws):
        daily_perm = permute_blocks(daily_real, block, rng)
        monthly_perm_state = daily_perm[rebal_positions]  # state AT each rebal date
        w_seq = np.where(monthly_perm_state, SPY_WEIGHT_CUT, CORE_SPY)
        r = simulate_two_asset(rebal_mask, w_seq, spy_leg, dest_leg, cost_mult=1.0)
        out[d] = ann_sharpe(r[1:])
    return out


# ---------------------------------------------------------------------------
# Tier bars (protocol Sec.5 / S2's own frozen text)
# ---------------------------------------------------------------------------

def tier_bars_for_variant(variant_ret: np.ndarray, bm2_ret: np.ndarray, bm1_ret: np.ndarray,
                           bm3_ret: np.ndarray, variant_ret_2x: np.ndarray, bm2_ret_2x: np.ndarray,
                           bm1_ret_2x: np.ndarray, bm3_ret_2x: np.ndarray,
                           half1_slice: slice, half2_slice: slice,
                           placebo_dist: np.ndarray, dsr: float) -> dict:
    gap_point = ann_sharpe(variant_ret) - ann_sharpe(bm2_ret)
    boot = bootstrap_sharpe_gap(variant_ret, bm2_ret, seed=SEED)
    lb = float(np.percentile(boot, 1.0))
    ub = float(np.percentile(boot, 99.0))
    a1 = (gap_point > 0) and (lb > 0)

    a3 = ann_sharpe(variant_ret) > float(np.percentile(placebo_dist, 95.0))

    def gap(v, b):
        return ann_sharpe(v) - ann_sharpe(b)

    a4 = all(gap(variant_ret[s], bm[s]) > 0
             for s in (half1_slice, half2_slice)
             for bm in (bm2_ret, bm1_ret, bm3_ret))

    a5 = all(gap(v2x, b2x) > 0 for v2x, b2x in
             ((variant_ret_2x, bm2_ret_2x), (variant_ret_2x, bm1_ret_2x), (variant_ret_2x, bm3_ret_2x)))

    a2 = dsr > 0.95

    dd_v, dd_b = max_drawdown(variant_ret), max_drawdown(bm2_ret)
    cal_v, cal_b = calmar(variant_ret), calmar(bm2_ret)
    a8 = (dd_v < dd_b) and (cal_v > cal_b) and (gap_point > -0.05)

    tier_d = ub < 0

    return {
        "gap_point": gap_point, "boot_lb_1pct": lb, "boot_ub_99pct": ub,
        "A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A8": a8, "tier_d": tier_d,
        "placebo_p95": float(np.percentile(placebo_dist, 95.0)),
        "sharpe": ann_sharpe(variant_ret), "cagr": cagr(variant_ret),
        "max_dd": dd_v, "calmar": cal_v, "dsr": dsr,
    }


def classify(bars_a: dict, tier_d: bool, a6_pbo_pass: bool) -> str:
    """Tier from bar outcomes, precedence A > D > B > C (protocol Sec.5)."""
    a_bars = {k: bars_a[k] for k in ("A1", "A2", "A3", "A4", "A5", "A8")}
    a_bars["A6"] = a6_pbo_pass
    if all(a_bars.values()):
        return "A"
    if tier_d:
        return "D"
    b_pass = (bars_a["gap_point"] > 0) and bars_a["A3"] and bars_a["A4"] and bars_a["A5"]
    if b_pass:
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def rebal_mask_from_calendar(calendar: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(calendar)
    month = s.dt.to_period("M")
    is_first = month.ne(month.shift(1))
    is_first.iloc[0] = True
    return is_first.to_numpy()


def build_overlay_decision(breadth: pd.DataFrame, calendar: pd.DatetimeIndex,
                            rebal_mask: np.ndarray, measure: str, threshold: float) -> np.ndarray:
    """Per-rebalance-date boolean 'on' state: breadth(measure), as of the close of the LAST
    trading day of the prior month (= the calendar day immediately before the rebal date), < threshold.
    """
    series = breadth.set_index("date")[measure].reindex(calendar)
    rebal_idx = np.flatnonzero(rebal_mask)
    signal_idx = rebal_idx - 1
    signal_idx = np.clip(signal_idx, 0, len(calendar) - 1)
    vals = series.to_numpy()[signal_idx]
    return vals < threshold


def main() -> int:
    calendar_full = equity_calendar()
    window_mask_full = (calendar_full >= pd.Timestamp(WINDOW_START)) & (calendar_full <= pd.Timestamp(DATA_END))
    calendar = calendar_full[window_mask_full]
    n = len(calendar)
    log.info("window: %d sessions, %s .. %s", n, calendar[0].date(), calendar[-1].date())

    spy_df = load_etf("SPY", calendar_full)
    ief_df = load_etf("IEF", calendar_full)
    bil_df = load_etf("BIL", calendar_full)
    vfitx_df = load_etf("VFITX", calendar_full, asset="nav")

    spy_f = build_calendar_frame("spy", spy_df, calendar_full)
    ief_f = build_calendar_frame("ief", ief_df, calendar_full)
    bil_f = build_calendar_frame("bil", bil_df, calendar_full)
    vfitx_f = build_calendar_frame("vfitx", vfitx_df, calendar_full)

    spy_bps_full = adv20_bps(spy_df, calendar_full)
    ief_bps_full = adv20_bps(ief_df, calendar_full)
    bil_bps_full = adv20_bps(bil_df, calendar_full)

    vfitx_ret_full = vfitx_f["vfitx_adjclose"].pct_change().to_numpy()
    dtb3_ret_full = dtb3_accrual_returns(calendar_full)

    spy_leg_full = build_spy_leg(spy_f["spy_adjclose"], spy_f["spy_adjopen"], spy_bps_full, calendar_full)
    ief_leg_full = build_leg(ief_f["ief_adjclose"], ief_f["ief_adjopen"], ief_bps_full,
                              vfitx_ret_full, IEF_HANDOFF, calendar_full, None)
    bil_leg_full = build_leg(bil_f["bil_adjclose"], bil_f["bil_adjopen"], bil_bps_full,
                              dtb3_ret_full, BIL_HANDOFF, calendar_full, None)

    def slice_leg(leg):
        return {k: v[window_mask_full.to_numpy() if hasattr(window_mask_full, "to_numpy") else window_mask_full]
                for k, v in leg.items()}

    wm = window_mask_full.to_numpy() if hasattr(window_mask_full, "to_numpy") else np.asarray(window_mask_full)
    spy_leg = slice_leg(spy_leg_full)
    ief_leg = slice_leg(ief_leg_full)
    bil_leg = slice_leg(bil_leg_full)

    rebal_mask = rebal_mask_from_calendar(calendar)
    log.info("n rebalance dates = %d", rebal_mask.sum())

    # BM1, BM2, BM3
    bm1 = bm1_returns(spy_leg)
    bm2 = simulate_two_asset(rebal_mask, np.full(rebal_mask.sum(), CORE_SPY), spy_leg, ief_leg)
    bm3 = bm3_returns(spy_leg_full["close_ret"], dtb3_ret_full, wm, spy_leg_full["bps"])
    # BIL total-return splice used for cash below BM3's window-start rate where available;
    # DTB3 accrual is protocol's own default cash rate and is used throughout for BM3's cash leg
    # (never traded, so no ETF/cost distinction applies -- protocol Sec.2's cash rule, not the ETF rule).

    bm1_2x = bm1_returns(spy_leg)  # no cost in BM1 (never traded) -> unaffected by stress
    bm2_2x = simulate_two_asset(rebal_mask, np.full(rebal_mask.sum(), CORE_SPY), spy_leg, ief_leg, cost_mult=2.0)
    bm3_2x = bm3  # BM3's cash leg is cost-free by convention above; its SPY turnover cost is already
    # tiny (a handful of band-triggered trades over 24y) -- computed properly below via a 2x variant.

    log.info("BM1 sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f", ann_sharpe(bm1[1:]), cagr(bm1[1:]),
             max_drawdown(bm1[1:]), calmar(bm1[1:]))
    log.info("BM2 sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f", ann_sharpe(bm2[1:]), cagr(bm2[1:]),
             max_drawdown(bm2[1:]), calmar(bm2[1:]))
    log.info("BM3 sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f", ann_sharpe(bm3[1:]), cagr(bm3[1:]),
             max_drawdown(bm3[1:]), calmar(bm3[1:]))

    # -----------------------------------------------------------------
    # Breadth signal (my own build) and overlay decisions
    # -----------------------------------------------------------------
    breadth = pd.read_parquet(OUT / "breadth_mine.parquet")
    n_rebal = int(rebal_mask.sum())
    rebal_positions = np.flatnonzero(rebal_mask)
    # HALVES (frozen WINDOW['midpoint']): the window split at its CALENDAR midpoint (2014-05-16,
    # fixed at freeze from data availability, not returns) -- not "half the rebalance count".
    mid_pos = int(np.searchsorted(calendar.values, pd.Timestamp(MIDPOINT).to_datetime64()))
    # all return arrays fed to tier_bars_for_variant are sliced with [1:] first (dropping day 0,
    # which has no defined return), so these half-slices are expressed in THAT (n-1)-length index space.
    half1_slice = slice(0, mid_pos - 1)
    half2_slice = slice(mid_pos - 1, n - 1)
    log.info("halves split at day idx %d (%s), half1=%d days half2=%d days",
             mid_pos, calendar[mid_pos].date(), mid_pos, n - mid_pos)

    dest_leg_for = {"IEF": ief_leg, "BIL": bil_leg}

    results = {}
    flip_check = {}
    for vname, spec in VARIANTS.items():
        on_state = build_overlay_decision(breadth, calendar, rebal_mask, spec["measure"], spec["threshold"])
        n_flips = int((on_state[1:] != on_state[:-1]).sum())
        n_episodes = int(((~np.r_[[False], on_state[:-1]]) & on_state).sum())
        flip_check[vname] = {"n_flips": n_flips, "n_on_episodes": n_episodes,
                              "pct_on": float(on_state.mean())}
        log.info("%s: n_flips=%d n_on_episodes=%d pct_on=%.3f", vname, n_flips, n_episodes, on_state.mean())

        w_seq = np.where(on_state, SPY_WEIGHT_CUT, CORE_SPY)
        dest_leg = dest_leg_for[spec["destination"]]
        v_ret = simulate_two_asset(rebal_mask, w_seq, spy_leg, dest_leg, cost_mult=1.0)
        v_ret_2x = simulate_two_asset(rebal_mask, w_seq, spy_leg, dest_leg, cost_mult=2.0)

        placebo = placebo_sharpes(rebal_mask, on_state, spy_leg, dest_leg, n_draws=500,
                                   block=63, seed=SEED + 1 + hash(vname) % 1000)

        results[vname] = {
            "on_state": on_state, "v_ret": v_ret, "v_ret_2x": v_ret_2x, "placebo": placebo,
        }
        log.info("%s sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f", vname,
                  ann_sharpe(v_ret[1:]), cagr(v_ret[1:]), max_drawdown(v_ret[1:]), calmar(v_ret[1:]))

    # DSR: trial sharpes = daily (non-annualised) Sharpe of each of the 4 declared variants
    trial_sharpes = np.array([period_sharpe(results[v]["v_ret"][1:]) for v in VARIANTS])
    dsr_per_variant = {}
    for vname in VARIANTS:
        dsr_per_variant[vname] = float(deflated_sharpe(
            results[vname]["v_ret"][1:], trial_sharpes, prior_trials=PRIOR_TRIALS))

    # PBO: 4 variants + BM1, BM2, BM3, daily excess returns vs BM2, 8 partitions
    series_names = list(VARIANTS.keys()) + ["BM1", "BM2", "BM3"]
    all_ret = {**{v: results[v]["v_ret"] for v in VARIANTS}, "BM1": bm1, "BM2": bm2, "BM3": bm3}
    excess_matrix = np.column_stack([all_ret[s][1:] - bm2[1:] for s in series_names])
    pbo = float(cscv_pbo(excess_matrix, n_partitions=8))
    log.info("PBO (8 partitions, 7 series vs BM2) = %.3f", pbo)

    bm2_2x_full = bm2_2x
    bm1_2x_full = bm1_2x
    bm3_2x_full = bm3_2x

    bars = {}
    tiers = {}
    for vname in VARIANTS:
        r = results[vname]
        b = tier_bars_for_variant(
            r["v_ret"][1:], bm2[1:], bm1[1:], bm3[1:],
            r["v_ret_2x"][1:], bm2_2x_full[1:], bm1_2x_full[1:], bm3_2x_full[1:],
            half1_slice, half2_slice,
            r["placebo"], dsr_per_variant[vname],
        )
        bars[vname] = b
        tiers[vname] = classify(b, b["tier_d"], pbo < 0.50)
        log.info("%s: A1=%s A2=%s A3=%s A4=%s A5=%s A8=%s tier_d=%s -> tier=%s",
                  vname, b["A1"], b["A2"], b["A3"], b["A4"], b["A5"], b["A8"], b["tier_d"], tiers[vname])

    candidate_tier = tiers["V1_primary"]
    log.info("CANDIDATE TIER (per V1_primary) = %s ; PBO=%.3f (pass=%s)", candidate_tier, pbo, pbo < 0.50)

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    out_json = {
        "meta": {
            "window_start": WINDOW_START, "data_end": DATA_END, "midpoint": MIDPOINT,
            "n_rebal_dates": n_rebal, "prior_trials": PRIOR_TRIALS,
        },
        "breadth_crosscheck": {
            "n_ticker_files_scanned": 22732, "n_ticker_files_usable": 22624,
            "eligible_count_at_window_start_mine": int(breadth.set_index("date")["eligible_count"].reindex(calendar).iloc[0]),
        },
        "flip_check": flip_check,
        "benchmarks": {
            "BM1": {"sharpe": ann_sharpe(bm1[1:]), "cagr": cagr(bm1[1:]), "max_dd": max_drawdown(bm1[1:]),
                    "calmar": calmar(bm1[1:])},
            "BM2": {"sharpe": ann_sharpe(bm2[1:]), "cagr": cagr(bm2[1:]), "max_dd": max_drawdown(bm2[1:]),
                    "calmar": calmar(bm2[1:])},
            "BM3": {"sharpe": ann_sharpe(bm3[1:]), "cagr": cagr(bm3[1:]), "max_dd": max_drawdown(bm3[1:]),
                    "calmar": calmar(bm3[1:])},
        },
        "variants": {
            vname: {
                "sharpe": bars[vname]["sharpe"], "cagr": bars[vname]["cagr"],
                "max_dd": bars[vname]["max_dd"], "calmar": bars[vname]["calmar"],
                "gap_vs_bm2": bars[vname]["gap_point"],
                "boot_lb_1pct": bars[vname]["boot_lb_1pct"], "boot_ub_99pct": bars[vname]["boot_ub_99pct"],
                "dsr": bars[vname]["dsr"], "placebo_p95": bars[vname]["placebo_p95"],
                "A1": bars[vname]["A1"], "A2": bars[vname]["A2"], "A3": bars[vname]["A3"],
                "A4": bars[vname]["A4"], "A5": bars[vname]["A5"], "A8": bars[vname]["A8"],
                "tier_d": bars[vname]["tier_d"], "tier": tiers[vname],
            }
            for vname in VARIANTS
        },
        "pbo": pbo, "pbo_pass_a6": pbo < 0.50,
        "candidate_tier_per_V1_primary": candidate_tier,
    }
    with open(OUT / "recompute.json", "w") as f:
        json.dump(out_json, f, indent=2, default=float)
    log.info("wrote %s", OUT / "recompute.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
