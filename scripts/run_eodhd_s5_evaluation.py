"""PHASE 2 evaluation harness for the FROZEN S5 pre-registration.

Frozen design: scripts/eodhd_s5_crypto_momentum_preregistered_bars.py
(bars_fingerprint 30decf93..., PREREGISTERED_AT 2026-09-30T19:18:54Z,
commit f0ec293). THIS FILE MUST NEVER EDIT THAT ONE. Any bug or ambiguity
found in it while building this harness is implemented per the MOST LITERAL
reading and recorded in ``PREREG_ISSUES`` below plus the run report's
``prereg_issues`` key -- never silently patched.

Subcommands::

    python scripts/run_eodhd_s5_evaluation.py build    --out <dir>
    python scripts/run_eodhd_s5_evaluation.py evaluate --out <dir> \
        --report docs/eodhd_s5_evaluation_2026_10.json [--append-ledger]

``build`` scans data/research/eodhd/crypto/*.parquet once (streamed, one file
at a time) and caches: (1) a per-ISO-week eligibility/dollar-volume/28d-return
table for every USD-quoted, non-excluded coin, and (2) cleaned daily OHLC
arrays for every coin that is eligible in at least one week (the only coins
that can ever enter a real or placebo holding). Cached to ``--out`` as .npz/
.parquet so ``evaluate`` can be re-run without rescanning 7,000+ files.

``evaluate`` runs the 3 frozen variants + placebo + BM1-3 + BTC_BH + C1_live,
the paired stationary block bootstrap, DSR, CSCV PBO, the frozen tiers (A7
PENDING), sanity checks, and writes the report + ledger entry.

PREREG_ISSUES (ambiguities in the FROZEN file, resolved by literal reading,
not by patching it -- see also the run report's own "prereg_issues" list,
which restates these with resolution + reasoning for the record):

1. DATA['segment_rule'] states a coin needs ">=29 calendar days" of
   same-segment history (28-day lookback + 1), while the same paragraph also
   requires the segment to cover "the trailing 30 calendar days needed for
   the dollar-volume ranking". These are two different numbers (29 vs 30)
   for two different purposes. Literal reading implemented: BOTH conditions
   are checked (AND), which in practice makes the 30-day condition binding
   (30 > 29) -- a coin passing the 30-day coverage check always also passes
   the 29-day one. No behavioural effect, just an internal inconsistency in
   the frozen wording, flagged for the record.
2. BOOTSTRAP['mean_block_calendar_days'] = 91 is stated once, with no
   separate figure for the BM1-BM3 sub-comparisons (which run on the
   SPY-trading-day calendar after the crypto-to-SPY mapping, not the native
   365-day crypto calendar the primary pair uses). Literal reading
   implemented: 91 is applied as 91 ROWS of whichever series is being paired
   (crypto-calendar rows for the BTC_BH/C1_live comparisons, SPY-calendar
   rows for the BM1-3 comparisons) -- i.e. the single declared number is used
   uniformly, not converted between calendars. This is NOT the same amount
   of real time in both cases (91 SPY trading-day rows ~= 129 calendar days,
   longer than 91 crypto calendar days) -- flagged, not silently corrected.
3. C1_live's own instrument field (inherited from
   scripts/alt_premia_preregistered_bars.py CANDIDATES['C1_btc_trend']) cites
   "BTC/USD spot (Tiingo crypto daily, UTC bars)" as its data source, but that
   Tiingo file is not present in this research data folder, and S5's own
   DATA['source'] is exclusively data/research/eodhd/crypto/*.parquet.
   Literal reading implemented: C1_live's RULE (28d on/off, 63d-vol sizing,
   Sunday review, 1-day lag -- verbatim from btc_trend.py/C1_btc_trend) is
   applied to data/research/eodhd/crypto/BTC-USD.parquet (this evaluation's
   own, already gate-validated BTC series), for data-source consistency with
   BTC_BH and every other series in this report, not to a Tiingo series that
   isn't available here.
4. WINDOWS['note_variant_windows'] fixes S5_top30_abs's OWN trading start at
   2017-05-28 (2 weeks after the other two variants) but does not restate
   whether its bootstrap/DSR/PBO windows should also start later. Literal
   reading implemented: all inference (bootstrap, DSR trial Sharpes, PBO) for
   S5_top30_abs uses ITS OWN 2017-05-28 start, consistent with "it trades
   from 2017-05-28"; A4's halves split still uses the shared
   WINDOWS['midpoint'] (2022-01-23), per the frozen note's own text ("not a
   separately recomputed midpoint").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eodhd_s5_crypto_momentum_preregistered_bars as prereg  # noqa: E402
import eodhd_clean  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402

log = logging.getLogger(__name__)

EODHD = _ROOT / "data" / "research" / "eodhd"
CRYPTO_DIR = EODHD / "crypto"
FRED_DIR = _ROOT / "data" / "research" / "fred"
ETFS_FULL = EODHD / "etfs_full"

ANN_CRYPTO = math.sqrt(365.0)   # native crypto calendar: trades every day
ANN_SPY = math.sqrt(252.0)      # SPY trading-day calendar

DAY = np.timedelta64(1, "D")

PREREG_ISSUES = [
    "segment_rule states both '>=29 calendar days' and 'the trailing 30 calendar days' as the "
    "same-segment coverage requirement; both are checked (AND), 30 is the binding one -- no "
    "behavioural effect, an internal wording inconsistency only.",
    "BOOTSTRAP mean_block_calendar_days=91 has no separately-stated figure for the BM1-3 "
    "sub-comparisons (SPY-calendar rows); applied literally as 91 ROWS on whichever series is "
    "paired, not converted between the crypto (365/yr) and SPY (252/yr) calendars.",
    "C1_live's inherited instrument field cites Tiingo BTC data, unavailable in this research "
    "folder; C1's RULE is applied verbatim to data/research/eodhd/crypto/BTC-USD.parquet instead, "
    "for data-source consistency with the rest of this report.",
    "WINDOWS['note_variant_windows'] fixes S5_top30_abs's trading start 2 weeks later "
    "(2017-05-28) but does not say whether its OWN inference windows (bootstrap/DSR/PBO) should "
    "also start later; implemented so they do, while A4's halves split keeps the shared midpoint.",
]


# ---------------------------------------------------------------------------
# Stage 0: universe file listing
# ---------------------------------------------------------------------------

def crypto_files() -> list[Path]:
    return sorted(f for f in CRYPTO_DIR.glob("*-USD.parquet") if not prereg.is_excluded(f.stem))


def sunday_grid(start: str, end_exclusive_review: str) -> pd.DatetimeIndex:
    """Sunday-UTC review dates from ``start`` (inclusive) through the last
    Sunday strictly before ``end_exclusive_review`` (DATA_END is itself a
    Monday-execution date, never a review date)."""
    g = pd.date_range(start, end_exclusive_review, freq="W-SUN")
    return g[g < pd.Timestamp(end_exclusive_review)] if g[-1] == pd.Timestamp(end_exclusive_review) else g


# ---------------------------------------------------------------------------
# Stage 1: per-coin, per-review-week table (PURE, testable on synthetic arrays)
# ---------------------------------------------------------------------------

def segment_start_dates(dates: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """``out[k]`` = the date of the first bar in segment ``k`` (segments are
    0, 1, 2, ... in order of appearance, per eodhd_clean.clean_bars)."""
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    return dates[starts]


def coin_week_table(dates: np.ndarray, aclose: np.ndarray, vol: np.ndarray, seg: np.ndarray,
                     sundays: np.ndarray, floor_usd: float,
                     lookback_days: int = 28, dvol_window_days: int = 30,
                     max_stale_days: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eligible, dvol30, ret28) for ONE coin's cleaned, date-sorted REAL bars
    (no reindex/ffill -- see module docstring on why dollar volume must use
    only real bars), aligned to ``sundays``.

    Literal implementation of UNIVERSE['eligibility'] + DATA['segment_rule']:
    a bar within max_stale_days of the review; the coin's CURRENT segment
    (the one containing the review bar) must itself have started at or
    before review-30d (PREREG_ISSUES #1: binding condition, also covers the
    28d-lookback same-segment requirement since 30 > 28); the 28d-lookback
    bar independently checked same-segment as an explicit belt-and-suspenders
    (frozen text states it separately); trailing-30d median dollar volume
    (adjusted_close x volume, real bars only in [review-30, review)) >= floor.
    """
    n = len(sundays)
    eligible = np.zeros(n, dtype=bool)
    dvol = np.full(n, np.nan, dtype=np.float64)
    ret28 = np.full(n, np.nan, dtype=np.float64)
    if len(dates) == 0:
        return eligible, dvol, ret28

    seg_start = segment_start_dates(dates, seg)
    lb_dates = sundays - lookback_days * DAY
    dv_start_dates = sundays - dvol_window_days * DAY

    idx_review = np.searchsorted(dates, sundays, side="right") - 1
    idx_lb = np.searchsorted(dates, lb_dates, side="right") - 1
    idx_dv_start = np.searchsorted(dates, dv_start_dates, side="left")
    idx_dv_end = np.searchsorted(dates, sundays, side="left")  # exclusive of the review date itself

    for i in range(n):
        ir = idx_review[i]
        if ir < 0:
            continue
        stale_days = (sundays[i] - dates[ir]) / DAY
        if stale_days > max_stale_days:
            continue
        il = idx_lb[i]
        if il < 0 or seg[il] != seg[ir]:
            continue
        this_seg_start = seg_start[seg[ir]]
        if this_seg_start > sundays[i] - dvol_window_days * DAY:
            continue  # current segment doesn't cover the full 30-day window
        s0, s1 = idx_dv_start[i], idx_dv_end[i]
        if s1 <= s0:
            continue
        dv = float(np.median(aclose[s0:s1] * vol[s0:s1]))
        if not np.isfinite(dv) or dv < floor_usd:
            continue
        eligible[i] = True
        dvol[i] = dv
        ret28[i] = float(aclose[ir] / aclose[il] - 1.0)
    return eligible, dvol, ret28


def base_symbol_array(codes: list[str]) -> np.ndarray:
    return np.array([prereg.base_symbol(c) for c in codes])


def resolve_ticker_collisions(eligible_row: np.ndarray, dvol_row: np.ndarray,
                               base_syms: np.ndarray) -> np.ndarray:
    """Return a COPY of eligible_row with collisions resolved: within each
    base-symbol group with >1 eligible member this week, keep only the
    highest-dvol one (UNIVERSE['ticker_collision_rule'])."""
    out = eligible_row.copy()
    elig_idx = np.flatnonzero(out)
    if len(elig_idx) < 2:
        return out
    groups: dict[str, list[int]] = {}
    for i in elig_idx:
        groups.setdefault(base_syms[i], []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda i: dvol_row[i])
        for i in members:
            if i != keep:
                out[i] = False
    return out


def select_holdings(eligible_row: np.ndarray, dvol_row: np.ndarray, ret28_row: np.ndarray,
                     n_universe: int, absolute_filter: bool) -> dict[int, float]:
    """Real (ranked) holdings for one review week: top-n_universe-by-dvol
    pool, ranked by ret28 descending, top tercile, equal weight
    1/tercile_count, absolute filter zeroes a slot to cash (not renormalised)."""
    pool = np.flatnonzero(eligible_row)
    if len(pool) == 0:
        return {}
    pool = pool[np.argsort(-dvol_row[pool])][:n_universe]
    k = prereg.tercile_count(len(pool)) if len(pool) < n_universe else prereg.tercile_count(n_universe)
    ranked = pool[np.argsort(-ret28_row[pool])][:k]
    w = 1.0 / k
    holdings = {}
    for i in ranked:
        if absolute_filter and not (ret28_row[i] > 0):
            continue
        holdings[int(i)] = w
    return holdings


def placebo_holdings(eligible_row: np.ndarray, dvol_row: np.ndarray, ret28_row: np.ndarray,
                      n_universe: int, absolute_filter: bool, rng: np.random.Generator) -> dict[int, float]:
    """Placebo holdings: SAME top-n_universe-by-dvol pool as select_holdings,
    but a uniform random draw (no replacement) of tercile_count names instead
    of ranking by ret28 (PLACEBO['rule'])."""
    pool = np.flatnonzero(eligible_row)
    if len(pool) == 0:
        return {}
    pool = pool[np.argsort(-dvol_row[pool])][:n_universe]
    k = prereg.tercile_count(len(pool)) if len(pool) < n_universe else prereg.tercile_count(n_universe)
    if k == 0:
        return {}
    k = min(k, len(pool))
    drawn = rng.choice(pool, size=k, replace=False)
    w = 1.0 / k
    holdings = {}
    for i in drawn:
        if absolute_filter and not (ret28_row[i] > 0):
            continue
        holdings[int(i)] = w
    return holdings


# ---------------------------------------------------------------------------
# Stage 2: daily adjusted-open-to-adjusted-open return series per coin
# ---------------------------------------------------------------------------

def aopen_returns(dates: np.ndarray, open_: np.ndarray, close: np.ndarray, aclose: np.ndarray,
                   full_calendar: pd.DatetimeIndex) -> np.ndarray:
    """Daily r[i] = aopen_{i+1}/aopen_i - 1 on ``full_calendar`` (one row per
    calendar day), aopen = open * adjusted_close/close. Gaps are forward-
    filled on aopen itself (a missing day repeats the last known aopen,
    i.e. a 0 return through the gap -- the same "missing data -> 0 return"
    convention as scripts/run_alt_premia_evaluation.py's simulate()).
    EXECUTION['timing']: "next bar's adjusted open" -- see this module's
    docstring for why an aopen-to-aopen daily grid implements that rule
    exactly via the same decide()-at-i-takes-effect-at-i+1 mechanism as
    alt_premia's simulate()."""
    safe_close = np.where(close == 0, np.nan, close)
    aopen = open_ * aclose / safe_close
    aopen = np.where(np.isfinite(aopen), aopen, open_)
    s = pd.Series(aopen, index=pd.DatetimeIndex(dates))
    s = s[~s.index.duplicated(keep="last")].reindex(full_calendar).ffill()
    r = (s.shift(-1) / s - 1.0).to_numpy(dtype=np.float64)
    return r


# ---------------------------------------------------------------------------
# Stage 3: drifting-weight simulator with diagnostics (turnover, cost, held)
# ---------------------------------------------------------------------------

Decide = Callable[[int, np.ndarray], "dict[int, float] | None"]


def simulate_with_diagnostics(rets: np.ndarray, rf: np.ndarray, decide: Decide,
                               cost_bps: float, max_gross: float = 1.0
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same drifting-weight mechanics as run_alt_premia_evaluation.simulate,
    generalised to a sparse dict-valued decide() over a (possibly very wide)
    fixed column space, and additionally returning per-day cost and turnover
    (sum |target - held-before-rebalance|) for the sanity-check report.
    """
    T, N = rets.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N), dtype=np.float32)
    cost_out = np.zeros(T)
    turnover_out = np.zeros(T)
    c = cost_bps / 1e4
    for i in range(T):
        r = np.nan_to_num(rets[i])
        cash_w = 1.0 - w.sum()
        rp = float(w @ r + cash_w * rf[i])
        if 1.0 + rp > 0:
            w = w * (1.0 + r) / (1.0 + rp)
        tgt_dict = decide(i, w)
        cost = 0.0
        if tgt_dict is not None:
            tgt = np.zeros(N)
            for j, wj in tgt_dict.items():
                tgt[j] = wj
            if tgt.sum() > max_gross + 1e-9 or (tgt < -1e-12).any():
                raise ValueError(f"target violates long-only/max_gross at {i}: sum={tgt.sum()}")
            turnover = float(np.abs(tgt - w).sum())
            cost = turnover * c
            turnover_out[i] = turnover
            w = tgt
        out[i] = rp - cost
        cost_out[i] = cost
        held[i] = w
    return out, held, cost_out, turnover_out


# ---------------------------------------------------------------------------
# Stage 4: benchmarks -- BM1/BM2/BM3 (amendment-1 etfs_full/ + FRED DTB3),
# BTC_BH and C1_live (EODHD BTC-USD.parquet, PREREG_ISSUES #3)
# ---------------------------------------------------------------------------

def _load_etf_full(ticker: str) -> pd.Series:
    d = pd.read_parquet(ETFS_FULL / f"{ticker}.parquet", columns=["date", "adjusted_close"])
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["adjusted_close"].sort_index()
    return s[~s.index.duplicated(keep="last")]


def spy_calendar(end: str) -> pd.DatetimeIndex:
    spy = _load_etf_full("SPY")
    return spy.index[spy.index <= pd.Timestamp(end)]


def fred_dtb3_daily(dates: pd.DatetimeIndex) -> pd.Series:
    d = pd.read_parquet(FRED_DIR / "DTB3.parquet")
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["value"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return (s.reindex(s.index.union(dates)).ffill().reindex(dates) / 100.0 / 252.0).fillna(0.0)


def cash_rf_series(dates: pd.DatetimeIndex) -> pd.Series:
    """BIL total return from 2007-05-30; FRED DTB3 before that (COSTS['cash_rate'])."""
    bil = _load_etf_full("BIL")
    bil_ret = bil.pct_change().reindex(dates)
    dtb3 = fred_dtb3_daily(dates)
    out = bil_ret.where(dates >= bil.index.min(), dtb3)
    return out.fillna(dtb3)


def bm1_spy(dates: pd.DatetimeIndex) -> pd.Series:
    spy = _load_etf_full("SPY").reindex(dates).ffill()
    return spy.pct_change().fillna(0.0)


def bm2_60_40(dates: pd.DatetimeIndex) -> pd.Series:
    """60% SPY / 40% IEF, weights DRIFTING with realised returns between
    rebalances and reset to target at each month-end close; VFITX proxies
    the bond leg before IEF's first bar (2002-07-26) --
    BENCHMARKS['BM2_60_40']. (Earlier draft of this function re-blended a
    constant 60/40 every day, which is a continuously-rebalanced-daily
    benchmark, not a monthly-rebalanced one -- fixed before any real S5
    variant result was computed, per run log.)"""
    spy = _load_etf_full("SPY").reindex(dates).ffill()
    ief = _load_etf_full("IEF")
    vfitx = _load_etf_full("VFITX")
    bond = ief.reindex(dates).ffill()
    pre_ief = dates < ief.index.min()
    spy_ret = spy.pct_change().fillna(0.0).to_numpy()
    bond_ret = bond.pct_change().fillna(0.0).to_numpy()
    if pre_ief.any():
        vfitx_ret = vfitx.pct_change().reindex(dates).fillna(0.0).to_numpy()
        bond_ret = np.where(pre_ief, vfitx_ret, bond_ret)
    month_end = np.r_[dates.to_period("M")[1:].to_numpy() != dates.to_period("M")[:-1].to_numpy(), True]
    rets = np.column_stack([spy_ret, bond_ret])
    rf = np.zeros(len(dates))

    def decide(i: int, w: np.ndarray) -> dict[int, float] | None:
        # decide(i, ...) takes effect at i+1 (simulate_with_diagnostics semantics),
        # so firing on month_end[i] itself rebalances effective the first trading
        # day of the NEXT month -- the standard "rebalanced monthly" convention.
        return {0: 0.6, 1: 0.4} if (i == 0 or month_end[i]) else None

    out, _, _, _ = simulate_with_diagnostics(rets, rf, decide, cost_bps=0.0)
    return pd.Series(out, index=dates)


def bm3_spy_vt(dates: pd.DatetimeIndex, target_vol: float = 0.12, vol_window: int = 21,
               band_abs: float = 0.10) -> pd.Series:
    spy = _load_etf_full("SPY").reindex(dates).ffill()
    ret = spy.pct_change().fillna(0.0)
    rf = fred_dtb3_daily(dates)
    sigma = ret.rolling(vol_window).std() * math.sqrt(252)
    target = (target_vol / sigma).clip(upper=1.0).fillna(0.0)
    w = 0.0
    out = np.zeros(len(dates))
    for i in range(len(dates)):
        if abs(target.iloc[i] - w) > band_abs:
            w = float(target.iloc[i])
        out[i] = w * ret.iloc[i] + (1 - w) * rf.iloc[i]
    return pd.Series(out, index=dates)


def crypto_to_spy_calendar(daily_ret: pd.Series, spy_dates: pd.DatetimeIndex) -> pd.Series:
    """Compound each native-calendar day's return onto the first SPY trading
    date STRICTLY after that day (BENCHMARKS['calendar_alignment'], the same
    convention already frozen for C1 in alt_premia_preregistered_bars.py /
    run_alt_premia_evaluation._btc_to_spy)."""
    bucket = np.searchsorted(spy_dates.values, daily_ret.index.values, side="right")
    n = len(spy_dates)
    out = np.zeros(n)
    valid = bucket < n
    log_ret = np.log1p(daily_ret.to_numpy(dtype=np.float64))
    np.add.at(out, bucket[valid], log_ret[valid])
    return pd.Series(np.expm1(out), index=spy_dates)


def load_btc_bars() -> pd.DataFrame:
    raw = pd.read_parquet(CRYPTO_DIR / "BTC-USD.parquet")
    cleaned, _ = eodhd_clean.clean_bars(raw, asset="crypto")
    return cleaned


def btc_buy_and_hold(btc: pd.DataFrame, start: str, end: str) -> pd.Series:
    s = btc.set_index("date")["adjusted_close"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).ffill()
    ret = s.pct_change().fillna(0.0)
    return ret[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]


def c1_live_returns(btc: pd.DataFrame, start: str, end: str,
                     full_calendar: pd.DatetimeIndex, cost_bps: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C1_live: the live BtcTrendSleeve/C1_btc_trend rule verbatim (on iff
    28d return > 0; weight = min(1, 0.40/sigma_63d_annualised); reviewed
    each Sunday UTC; trade if on/off flips or |target-held| > 0.10; position
    effective the next day), applied to EODHD BTC-USD.parquet
    (PREREG_ISSUES #3), on ``full_calendar`` via the SAME
    simulate_with_diagnostics engine as the S5 variants (so its turnover pays
    the SAME crypto_bps_per_side cost -- a live strategy that trades does
    incur real costs; BENCHMARKS['primary'] calls this "the honest
    comparison", which requires it to pay costs too, not run cost-free).
    Returns (net, held, cost)."""
    s = btc.set_index("date")["adjusted_close"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    px = s.reindex(full).ffill()
    ret = px.pct_change()
    on = (px / px.shift(28) - 1.0) > 0
    sigma = ret.rolling(63).std() * math.sqrt(365)
    target_vol = 0.40
    target = (target_vol / sigma).clip(upper=1.0).where(np.isfinite(sigma), 0.0)
    weight = target.where(on, 0.0).fillna(0.0)
    sundays = px.index[(px.index.dayofweek == 6) & (px.index >= pd.Timestamp(start))]

    idx_of = {d: i for i, d in enumerate(full_calendar)}
    band = 0.10
    decisions: dict[int, float] = {}
    for sunday in sundays:
        i = idx_of.get(sunday)
        if i is None:
            continue
        decisions[i] = float(weight.loc[sunday])

    rets = aopen_returns(btc["date"].to_numpy(dtype="datetime64[ns]"),
                          btc["open"].to_numpy(dtype=np.float64),
                          btc["close"].to_numpy(dtype=np.float64),
                          btc["adjusted_close"].to_numpy(dtype=np.float64),
                          full_calendar).reshape(-1, 1)
    rf = np.zeros(len(full_calendar))

    def decide(i: int, w: np.ndarray) -> dict[int, float] | None:
        tgt = decisions.get(i)
        if tgt is None:
            return None
        currently_on = w[0] > 1e-9
        target_on = tgt > 1e-9
        if target_on != currently_on or abs(tgt - w[0]) > band:
            return {0: tgt}
        return None

    out, held, cost, _ = simulate_with_diagnostics(rets, rf, decide, cost_bps)
    mask = (full_calendar >= pd.Timestamp(start)) & (full_calendar <= pd.Timestamp(end))
    return out[mask], held[mask, 0], cost[mask]


# ---------------------------------------------------------------------------
# Stage 5: stats -- bootstrap, DSR, PBO, tiers
# ---------------------------------------------------------------------------

def sharpe(ex: np.ndarray, ann: float = 1.0) -> float:
    ex = ex[np.isfinite(ex)]
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * ann) if sd > 0 else float("nan")


def stationary_indices(n: int, n_boot: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Identical algorithm to run_alt_premia_evaluation.stationary_indices
    (generic, no prereg dependency) -- duplicated here to keep this module
    import-independent of the alt-premia harness."""
    p = 1.0 / mean_block
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.broadcast_to(np.arange(n), (n_boot, n))
    block_start_pos = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    start_val = np.take_along_axis(starts, block_start_pos, axis=1)
    return ((start_val + pos - block_start_pos) % n).astype(np.int32)


def paired_sharpe_gap_boot(a: np.ndarray, b: np.ndarray, mean_block: int, ann: float,
                            n_boot: int, seed: int) -> np.ndarray:
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
        out[k:k + m] = (sa - sb) * ann
    return out


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def cagr(r: np.ndarray, periods_per_year: float) -> float:
    nav = np.cumprod(1 + r)
    years = len(r) / periods_per_year
    return float(nav[-1] ** (1 / years) - 1) if years > 0 and nav[-1] > 0 else float("nan")


def annual_vol(r: np.ndarray, periods_per_year: float) -> float:
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    return prereg.classify(bars, tier_d, tier_b)


# ---------------------------------------------------------------------------
# Full-scan pipeline (build)
# ---------------------------------------------------------------------------

def build_tables(files: list[Path], sundays: pd.DatetimeIndex, floor_usd: float,
                  log_every: int = 1000) -> dict:
    """Stream every file once: clean it, compute its (eligible, dvol, ret28)
    row, and -- only if it is eligible at least one week -- cache its cleaned
    daily OHLC arrays (the only coins that can ever enter a real or placebo
    holding). Returns a dict of numpy arrays + the ohlc cache."""
    n_files = len(files)
    n_weeks = len(sundays)
    sundays_arr = sundays.values.astype("datetime64[ns]")
    eligible = np.zeros((n_weeks, n_files), dtype=bool)
    dvol = np.full((n_weeks, n_files), np.nan, dtype=np.float32)
    ret28 = np.full((n_weeks, n_files), np.nan, dtype=np.float32)
    codes = [f.stem for f in files]
    base_syms = base_symbol_array(codes)
    ohlc: dict[str, dict[str, np.ndarray]] = {}
    n_empty = n_cached = 0
    for j, f in enumerate(files):
        if log_every and j % log_every == 0:
            log.info("build_tables: %d/%d files (%d cached so far)", j, n_files, n_cached)
        try:
            raw = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
        except Exception as exc:  # pragma: no cover -- defensive, logged not silenced
            log.warning("%s: unreadable (%s), treated as empty", f.stem, exc)
            n_empty += 1
            continue
        if raw.empty:
            n_empty += 1
            continue
        cleaned, _ = eodhd_clean.clean_bars(raw, asset="crypto")
        if cleaned.empty:
            n_empty += 1
            continue
        dates = cleaned["date"].to_numpy(dtype="datetime64[ns]")
        close = cleaned["close"].to_numpy(dtype=np.float64)
        aclose = cleaned["adjusted_close"].to_numpy(dtype=np.float64)
        vol = cleaned["volume"].to_numpy(dtype=np.float64)
        seg = cleaned["segment"].to_numpy(dtype=np.int32)
        elig_row, dvol_row, ret28_row = coin_week_table(dates, aclose, vol, seg, sundays_arr, floor_usd)
        eligible[:, j] = elig_row
        dvol[:, j] = dvol_row
        ret28[:, j] = ret28_row
        if elig_row.any():
            n_cached += 1
            ohlc[codes[j]] = {
                "dates": dates,
                "open": cleaned["open"].to_numpy(dtype=np.float32),
                "close": close.astype(np.float32),
                "adjusted_close": aclose.astype(np.float32),
            }
        del raw, cleaned
    log.info("build_tables done: %d files, %d empty/unreadable, %d cached (ever eligible)",
              n_files, n_empty, n_cached)
    return {"eligible": eligible, "dvol": dvol, "ret28": ret28, "codes": codes,
            "base_syms": base_syms, "sundays": sundays, "ohlc": ohlc,
            "n_empty": n_empty, "n_cached": n_cached}


def resolve_collisions_all_weeks(tables: dict) -> np.ndarray:
    """eligible[nw, nf], with ticker collisions resolved independently per week."""
    eligible = tables["eligible"]
    dvol = tables["dvol"]
    base_syms = tables["base_syms"]
    out = eligible.copy()
    # Only base symbols with >1 file anywhere can ever collide; restrict the
    # per-week work to those groups only (cheap: a handful of symbols).
    uniq, counts = np.unique(base_syms, return_counts=True)
    dup_syms = uniq[counts > 1]
    if len(dup_syms) == 0:
        return out
    dup_groups = [np.flatnonzero(base_syms == s) for s in dup_syms]
    for w in range(eligible.shape[0]):
        for members in dup_groups:
            elig_members = members[out[w, members]]
            if len(elig_members) > 1:
                keep = elig_members[np.argmax(dvol[w, elig_members])]
                for m in elig_members:
                    if m != keep:
                        out[w, m] = False
    return out


def pool_indices_per_week(eligible_resolved: np.ndarray, dvol: np.ndarray, n_universe: int) -> list[np.ndarray]:
    """Per week, the top-n_universe-by-dvol eligible column indices (post
    collision resolution), descending dvol order."""
    out = []
    for w in range(eligible_resolved.shape[0]):
        pool = np.flatnonzero(eligible_resolved[w])
        pool = pool[np.argsort(-dvol[w, pool])][:n_universe]
        out.append(pool)
    return out


def union_pool_codes(tables: dict, eligible_resolved: np.ndarray, n_max: int) -> list[str]:
    pools = pool_indices_per_week(eligible_resolved, tables["dvol"], n_max)
    union_idx = sorted(set(int(i) for p in pools for i in p))
    return [tables["codes"][i] for i in union_idx]


def build_aopen_matrix(ohlc: dict[str, dict[str, np.ndarray]], pool_codes: list[str],
                        full_calendar: pd.DatetimeIndex) -> tuple[np.ndarray, dict[str, int]]:
    """(rets[T, len(pool_codes)], code->column index), aopen-to-aopen daily
    returns for every coin that is ever in the top-n_universe pool of any
    variant (real or placebo can only ever hold names from this set)."""
    col_of = {c: k for k, c in enumerate(pool_codes)}
    T = len(full_calendar)
    rets = np.zeros((T, len(pool_codes)), dtype=np.float32)
    for c in pool_codes:
        d = ohlc[c]
        rets[:, col_of[c]] = aopen_returns(d["dates"], d["open"], d["close"], d["adjusted_close"], full_calendar)
    return rets, col_of


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------

def run_variant(variant_name: str, tables: dict, eligible_resolved: np.ndarray,
                 col_of: dict[str, int], rets: np.ndarray, rf: np.ndarray,
                 full_calendar: pd.DatetimeIndex, sundays: pd.DatetimeIndex,
                 monday_after: dict[pd.Timestamp, int], trade_start: str,
                 placebo_rng: np.random.Generator | None = None,
                 cost_bps: float | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Net daily returns (on full_calendar) for one variant. If
    ``placebo_rng`` is given, uses placebo_holdings instead of select_holdings
    (PLACEBO['rule'])."""
    spec = prereg.VARIANTS[variant_name]
    n_universe = spec["n_universe"]
    absolute_filter = spec["absolute_filter"]
    codes = tables["codes"]
    dvol, ret28 = tables["dvol"], tables["ret28"]
    trade_start_ts = pd.Timestamp(trade_start)

    decisions: dict[int, dict[int, float]] = {}
    for w, sunday in enumerate(sundays):
        if sunday < trade_start_ts:
            continue
        i = monday_after.get(sunday)
        if i is None:
            continue
        if placebo_rng is not None:
            holdings = placebo_holdings(eligible_resolved[w], dvol[w], ret28[w], n_universe,
                                         absolute_filter, placebo_rng)
        else:
            holdings = select_holdings(eligible_resolved[w], dvol[w], ret28[w], n_universe, absolute_filter)
        mapped = {}
        for file_idx, wt in holdings.items():
            col = col_of.get(codes[file_idx])
            if col is not None:
                mapped[col] = wt
        decisions[i - 1] = mapped  # decide() at i-1 (Sunday's row) takes effect at i (Monday)

    def decide(i: int, w: np.ndarray) -> dict[int, float] | None:
        return decisions.get(i)

    bps = prereg.COSTS["crypto_bps_per_side"] if cost_bps is None else cost_bps
    out, held, cost, turnover = simulate_with_diagnostics(rets, rf, decide, bps)
    return out, held, cost, turnover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _seed_for(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sundays = sunday_grid(prereg.WINDOWS["start"], prereg.DATA_END)
    log.info("sunday grid: %d reviews, %s .. %s", len(sundays), sundays[0].date(), sundays[-1].date())
    files = crypto_files()
    log.info("%d USD-quoted, non-excluded candidate files", len(files))
    tables = build_tables(files, sundays, prereg.UNIVERSE["liquidity_floor_usd"])
    eligible_resolved = resolve_collisions_all_weeks(tables)
    union_codes = union_pool_codes(tables, eligible_resolved, max(v["n_universe"] for v in prereg.VARIANTS.values()))
    log.info("union pool across all weeks/variants: %d distinct coins", len(union_codes))
    np.savez_compressed(
        out_dir / "s5_tables.npz",
        eligible=tables["eligible"], eligible_resolved=eligible_resolved,
        dvol=tables["dvol"], ret28=tables["ret28"],
        codes=np.array(tables["codes"]), base_syms=tables["base_syms"],
        sundays=sundays.values.astype("datetime64[ns]"),
    )
    ohlc_dir = out_dir / "ohlc"
    ohlc_dir.mkdir(exist_ok=True)
    for c in union_codes:
        d = tables["ohlc"][c]
        np.savez(ohlc_dir / f"{c.replace('/', '_')}.npz", **d)
    (out_dir / "union_pool_codes.json").write_text(json.dumps(union_codes))
    (out_dir / "build_meta.json").write_text(json.dumps({
        "n_files": len(files), "n_empty": tables["n_empty"], "n_cached_ever_eligible": tables["n_cached"],
        "n_union_pool": len(union_codes), "n_weeks": len(sundays),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    log.info("wrote build artifacts to %s", out_dir)
    return 0


def _load_build(out_dir: Path) -> dict:
    z = np.load(out_dir / "s5_tables.npz", allow_pickle=False)
    sundays = pd.DatetimeIndex(z["sundays"])
    codes = list(z["codes"])
    union_codes = json.loads((out_dir / "union_pool_codes.json").read_text())
    ohlc = {}
    for c in union_codes:
        d = np.load(out_dir / "ohlc" / f"{c.replace('/', '_')}.npz")
        ohlc[c] = {k: d[k] for k in d.files}
    return {"eligible": z["eligible"], "eligible_resolved": z["eligible_resolved"],
            "dvol": z["dvol"], "ret28": z["ret28"], "codes": codes,
            "base_syms": z["base_syms"], "sundays": sundays, "ohlc": ohlc,
            "union_codes": union_codes}


def _top_contributors(held: np.ndarray, rets: np.ndarray, col_of: dict[str, int], k: int = 10) -> list[dict]:
    code_of = {v: c for c, v in col_of.items()}
    # contribution of column j on day i uses the weight ACTIVE during day i's
    # own return, i.e. held[i-1] (held[i] is the weight set AFTER day i, per
    # simulate_with_diagnostics) -- held_prev[0] = 0 (nothing held before day 0).
    held_prev = np.vstack([np.zeros((1, held.shape[1]), dtype=held.dtype), held[:-1]])
    contrib = (held_prev * np.nan_to_num(rets)).sum(axis=0)
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"code": code_of.get(j, f"col{j}"), "total_contribution": float(contrib[j])} for j in order]


def cmd_evaluate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    tables = _load_build(out_dir)
    sundays = tables["sundays"]
    eligible_resolved = tables["eligible_resolved"]
    full_calendar = pd.date_range(prereg.WINDOWS["start"], prereg.DATA_END, freq="D")
    idx_of = {d: i for i, d in enumerate(full_calendar)}
    monday_after: dict[pd.Timestamp, int] = {}
    for s in sundays:
        m = s + pd.Timedelta(days=1)
        if m in idx_of:
            monday_after[s] = idx_of[m]

    rets, col_of = build_aopen_matrix(tables["ohlc"], tables["union_codes"], full_calendar)
    rf = cash_rf_series(full_calendar).to_numpy()

    trade_start = {"S5_top20_abs": prereg.WINDOWS["start"], "S5_top20_rel": prereg.WINDOWS["start"],
                   "S5_top30_abs": "2017-05-28"}

    results = {}
    for name in prereg.VARIANTS:
        out, held, cost, turnover = run_variant(name, tables, eligible_resolved, col_of, rets, rf,
                                                 full_calendar, sundays, monday_after, trade_start[name])
        results[name] = {"net": out, "held": held, "cost": cost, "turnover": turnover}

    primary_name = next(n for n, v in prereg.VARIANTS.items() if v["primary"])
    primary = results[primary_name]["net"]

    btc = load_btc_bars()
    bh = btc_buy_and_hold(btc, prereg.WINDOWS["start"], prereg.DATA_END).reindex(full_calendar).fillna(0.0).to_numpy()
    c1_net, c1_held, c1_cost = c1_live_returns(btc, prereg.WINDOWS["start"], prereg.DATA_END, full_calendar,
                                                prereg.COSTS["crypto_bps_per_side"])

    spy_dates = spy_calendar(prereg.DATA_END)
    spy_dates = spy_dates[spy_dates >= full_calendar[0]]
    bm1 = bm1_spy(spy_dates).to_numpy()
    bm2 = bm2_60_40(spy_dates).to_numpy()
    bm3 = bm3_spy_vt(spy_dates).to_numpy()
    rf_spy = cash_rf_series(spy_dates).to_numpy()

    # ---- no-NaN check (raw, before any excess/mapping arithmetic) ----
    nan_report = {name: int(np.isnan(r["net"]).sum()) for name, r in results.items()}
    nan_report.update({"BTC_BH": int(np.isnan(bh).sum()), "C1_live": int(np.isnan(c1_net).sum()),
                       "BM1_SPY": int(np.isnan(bm1).sum()), "BM2_60_40": int(np.isnan(bm2).sum()),
                       "BM3_SPY_VT": int(np.isnan(bm3).sum())})

    # ---- excess returns ----
    ex_primary = primary - rf
    ex_bh = bh - rf
    ex_c1 = c1_net - rf
    ex_bm1, ex_bm2, ex_bm3 = bm1 - rf_spy, bm2 - rf_spy, bm3 - rf_spy
    ex_primary_spy = crypto_to_spy_calendar(pd.Series(ex_primary, index=full_calendar), spy_dates).to_numpy()

    SEED = prereg.BOOTSTRAP["seed"]
    ALPHA = prereg.BOOTSTRAP["alpha_one_sided"]
    MEAN_BLOCK = prereg.BOOTSTRAP["mean_block_calendar_days"]  # PREREG_ISSUES #2
    N_BOOT = prereg.BOOTSTRAP["n_boot"]

    def gap_and_ci(a: np.ndarray, b: np.ndarray, ann: float, tag: str) -> dict:
        sa, sb = sharpe(a, ann), sharpe(b, ann)
        boot = paired_sharpe_gap_boot(a, b, MEAN_BLOCK, ann, N_BOOT, _seed_for("boot", tag, str(SEED)))
        return {"sharpe_candidate": sa, "sharpe_benchmark": sb, "gap": sa - sb,
                "boot_lower_alpha01": float(np.quantile(boot, ALPHA)),
                "boot_upper_alpha01": float(np.quantile(boot, 1 - ALPHA))}

    gaps_primary = {
        "BTC_BH": gap_and_ci(ex_primary, ex_bh, ANN_CRYPTO, "primary_vs_bh"),
        "C1_live": gap_and_ci(ex_primary, ex_c1, ANN_CRYPTO, "primary_vs_c1"),
        "BM1_SPY": gap_and_ci(ex_primary_spy, ex_bm1, ANN_SPY, "primary_vs_bm1"),
        "BM2_60_40": gap_and_ci(ex_primary_spy, ex_bm2, ANN_SPY, "primary_vs_bm2"),
        "BM3_SPY_VT": gap_and_ci(ex_primary_spy, ex_bm3, ANN_SPY, "primary_vs_bm3"),
    }

    a1 = all(gaps_primary[k]["gap"] > 0 and gaps_primary[k]["boot_lower_alpha01"] > 0
             for k in ("BTC_BH", "C1_live"))
    tier_d = any(gaps_primary[k]["boot_upper_alpha01"] < 0 for k in ("BTC_BH", "C1_live"))

    # ---- A4: halves ----
    mid = pd.Timestamp(prereg.WINDOWS["midpoint"])
    h1, h2 = full_calendar <= mid, full_calendar > mid
    h1s, h2s = spy_dates <= mid, spy_dates > mid
    a4_checks = {}
    for name, (a, b, ann) in {"BTC_BH": (ex_primary, ex_bh, ANN_CRYPTO),
                              "C1_live": (ex_primary, ex_c1, ANN_CRYPTO)}.items():
        g1, g2 = sharpe(a[h1], ann) - sharpe(b[h1], ann), sharpe(a[h2], ann) - sharpe(b[h2], ann)
        a4_checks[name] = {"half1_gap": g1, "half2_gap": g2, "pass": bool(g1 > 0 and g2 > 0)}
    for name, b in {"BM1_SPY": ex_bm1, "BM2_60_40": ex_bm2, "BM3_SPY_VT": ex_bm3}.items():
        g1 = sharpe(ex_primary_spy[h1s], ANN_SPY) - sharpe(b[h1s], ANN_SPY)
        g2 = sharpe(ex_primary_spy[h2s], ANN_SPY) - sharpe(b[h2s], ANN_SPY)
        a4_checks[name] = {"half1_gap": g1, "half2_gap": g2, "pass": bool(g1 > 0 and g2 > 0)}
    a4 = all(v["pass"] for v in a4_checks.values())

    # ---- A5: 2x costs (candidate + C1_live pay double; BM1-3 have no cost
    # model declared in this candidate's own COSTS dict, so are unchanged) ----
    stress = prereg.COSTS["stress_multiplier"]
    out2x, _, _, _ = run_variant(primary_name, tables, eligible_resolved, col_of, rets, rf,
                                  full_calendar, sundays, monday_after, trade_start[primary_name],
                                  cost_bps=prereg.COSTS["crypto_bps_per_side"] * stress)
    c1_net_2x, _, _ = c1_live_returns(btc, prereg.WINDOWS["start"], prereg.DATA_END, full_calendar,
                                      prereg.COSTS["crypto_bps_per_side"] * stress)
    ex_primary_2x, ex_c1_2x = out2x - rf, c1_net_2x - rf
    ex_primary_2x_spy = crypto_to_spy_calendar(pd.Series(ex_primary_2x, index=full_calendar), spy_dates).to_numpy()
    a5_checks = {
        "BTC_BH": bool(sharpe(ex_primary_2x, ANN_CRYPTO) - sharpe(ex_bh, ANN_CRYPTO) > 0),
        "C1_live": bool(sharpe(ex_primary_2x, ANN_CRYPTO) - sharpe(ex_c1_2x, ANN_CRYPTO) > 0),
        "BM1_SPY": bool(sharpe(ex_primary_2x_spy, ANN_SPY) - sharpe(ex_bm1, ANN_SPY) > 0),
        "BM2_60_40": bool(sharpe(ex_primary_2x_spy, ANN_SPY) - sharpe(ex_bm2, ANN_SPY) > 0),
        "BM3_SPY_VT": bool(sharpe(ex_primary_2x_spy, ANN_SPY) - sharpe(ex_bm3, ANN_SPY) > 0),
    }
    a5 = all(a5_checks.values())

    # ---- A2: DSR, using every variant's daily (non-annualised) Sharpe on
    # its own SPY-mapped window (PREREG_ISSUES #4) ----
    variant_ex_spy = {}
    for name in prereg.VARIANTS:
        ex_v = results[name]["net"] - rf
        ts = pd.Timestamp(trade_start[name])
        v_spy_dates = spy_dates[spy_dates >= ts]
        mapped = crypto_to_spy_calendar(pd.Series(ex_v, index=full_calendar)[full_calendar >= ts], v_spy_dates)
        variant_ex_spy[name] = mapped
    trial_sharpes = np.array([sharpe(variant_ex_spy[name].to_numpy(), 1.0) for name in prereg.VARIANTS])
    dsr = float(deflated_sharpe(variant_ex_spy[primary_name].to_numpy(), trial_sharpes,
                                 prior_trials=prereg.DSR["prior_trials"]))
    a2 = bool(dsr > 0.95)

    # ---- A6: CSCV PBO over the 3 variants + BM1-3, aligned to the shortest
    # available common range (S5_top30_abs's own 2017-05-28 start) ----
    common_start = max(pd.Timestamp(trade_start[n]) for n in prereg.VARIANTS)
    common_spy = spy_dates[spy_dates >= common_start]
    pbo_cols = []
    for name in prereg.VARIANTS:
        pbo_cols.append(variant_ex_spy[name].reindex(common_spy).to_numpy())
    pbo_cols += [pd.Series(ex_bm1, index=spy_dates).reindex(common_spy).to_numpy(),
                 pd.Series(ex_bm2, index=spy_dates).reindex(common_spy).to_numpy(),
                 pd.Series(ex_bm3, index=spy_dates).reindex(common_spy).to_numpy()]
    pbo_matrix = np.nan_to_num(np.column_stack(pbo_cols))
    pbo = float(cscv_pbo(pbo_matrix, n_partitions=prereg.PBO["n_partitions"]))
    a6 = bool(pbo < 0.50)

    # ---- A3: placebo, primary variant only (the one being tiered) ----
    rng = np.random.default_rng(prereg.PLACEBO["seed"])
    placebo_sharpes = np.empty(prereg.PLACEBO["n_draws"])
    for d in range(prereg.PLACEBO["n_draws"]):
        pout, _, _, _ = run_variant(primary_name, tables, eligible_resolved, col_of, rets, rf,
                                     full_calendar, sundays, monday_after, trade_start[primary_name],
                                     placebo_rng=rng)
        placebo_sharpes[d] = sharpe(pout - rf, ANN_CRYPTO)
    placebo_pctl = float((placebo_sharpes < sharpe(ex_primary, ANN_CRYPTO)).mean() * 100)
    a3 = bool(placebo_pctl > prereg.PLACEBO["pass_percentile"])

    bars = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5, "A6": a6}
    tier_b_bars = {"B_a": bool(gaps_primary["BTC_BH"]["gap"] > 0 and gaps_primary["C1_live"]["gap"] > 0),
                   "B_b": bool(a3 and a4 and a5)}
    tier = prereg.classify(bars, tier_d, tier_b_bars)
    tier_conditional = f"{tier} (conditional on A7 -- PENDING, per instructions)"

    # ---- sanity checks ----
    def perf(r: np.ndarray, ann_periods: float, ann_sqrt: float) -> dict:
        return {"cagr": cagr(r, ann_periods), "annual_vol": annual_vol(r, ann_periods),
                "sharpe_excess": float("nan"), "max_drawdown": max_drawdown(r)}

    sanity_perf = {
        primary_name: perf(primary, 365, ANN_CRYPTO),
        "BTC_BH": perf(bh, 365, ANN_CRYPTO),
        "C1_live": perf(c1_net, 365, ANN_CRYPTO),
        "BM1_SPY": perf(bm1, 252, ANN_SPY),
        "BM2_60_40": perf(bm2, 252, ANN_SPY),
        "BM3_SPY_VT": perf(bm3, 252, ANN_SPY),
    }
    big_move_mask = np.abs(primary) > 0.40
    big_moves = []
    for i in np.flatnonzero(big_move_mask):
        day = full_calendar[i]
        holders = [col for col, wt in [(c, results[primary_name]["held"][i, col_of[c]])
                                        for c in tables["union_codes"] if c in col_of] if abs(wt) > 1e-6]
        big_moves.append({"date": str(day.date()), "return": float(primary[i]),
                          "held_names": holders[:10]})
    turnover_total = float(results[primary_name]["turnover"].sum())
    cost_total = float(results[primary_name]["cost"].sum())
    top10 = _top_contributors(results[primary_name]["held"], rets, col_of)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint_at_freeze": "30decf9398917c2e5229cfee3b186f9f1f9d623a814273ba5ac629a194dbd584",
        "fingerprint_recomputed": prereg.bars_fingerprint(),
        "preregistered_at": prereg.PREREGISTERED_AT,
        "primary_variant": primary_name,
        "window": {"start": prereg.WINDOWS["start"], "end": prereg.DATA_END, "midpoint": prereg.WINDOWS["midpoint"]},
        "prereg_issues": PREREG_ISSUES,
        "bars": bars,
        "tier_d_triggered": tier_d,
        "tier_b_bars": tier_b_bars,
        "tier": tier,
        "tier_conditional_on_A7": tier_conditional,
        "gaps_vs_benchmarks": gaps_primary,
        "a4_halves": a4_checks,
        "a5_stress_pass": a5_checks,
        "dsr": dsr, "dsr_prior_trials": prereg.DSR["prior_trials"], "trial_sharpes": trial_sharpes.tolist(),
        "pbo": pbo,
        "placebo": {"n_draws": prereg.PLACEBO["n_draws"], "candidate_percentile": placebo_pctl,
                   "candidate_sharpe": sharpe(ex_primary, ANN_CRYPTO),
                   "placebo_sharpe_mean": float(np.mean(placebo_sharpes)),
                   "placebo_sharpe_p95": float(np.quantile(placebo_sharpes, 0.95))},
        "sanity_checks": {
            "performance": sanity_perf,
            "n_nan_days": nan_report,
            "n_days_abs_return_gt_40pct": int(big_move_mask.sum()),
            "big_moves": big_moves,
            "turnover_total_primary": turnover_total,
            "cost_drag_total_primary": cost_total,
            "top10_contributors_primary": top10,
        },
        "union_pool_size": len(tables["union_codes"]),
    }

    if tier in ("A", "B"):
        # post-pass audits: 2-day signal lag (Tuesday execution instead of
        # Monday) + per-year table -- only run when they're actually needed.
        def run_lagged(variant_name: str) -> np.ndarray:
            spec = prereg.VARIANTS[variant_name]
            codes = tables["codes"]
            decisions = {}
            ts = pd.Timestamp(trade_start[variant_name])
            for w, sunday in enumerate(sundays):
                if sunday < ts:
                    continue
                tue = sunday + pd.Timedelta(days=2)
                i = idx_of.get(tue)
                if i is None:
                    continue
                holdings = select_holdings(eligible_resolved[w], tables["dvol"][w], tables["ret28"][w],
                                            spec["n_universe"], spec["absolute_filter"])
                mapped = {col_of[codes[fi]]: wt for fi, wt in holdings.items() if codes[fi] in col_of}
                decisions[i - 1] = mapped

            def decide(i, w):
                return decisions.get(i)
            out, _, _, _ = simulate_with_diagnostics(rets, rf, decide, prereg.COSTS["crypto_bps_per_side"])
            return out
        primary_2daylag = run_lagged(primary_name)
        ex_primary_2daylag = primary_2daylag - rf
        report["post_pass_audit"] = {
            "2day_lag_sharpe_gap_vs_BTC_BH": sharpe(ex_primary_2daylag, ANN_CRYPTO) - sharpe(ex_bh, ANN_CRYPTO),
            "2day_lag_sharpe_gap_vs_C1_live": sharpe(ex_primary_2daylag, ANN_CRYPTO) - sharpe(ex_c1, ANN_CRYPTO),
        }
        years = full_calendar.year
        per_year = {}
        for y in sorted(set(years)):
            m = years == y
            if m.sum() < 30:
                continue
            per_year[int(y)] = {"primary_sharpe": sharpe(ex_primary[m], ANN_CRYPTO),
                                "bh_sharpe": sharpe(ex_bh[m], ANN_CRYPTO),
                                "c1_sharpe": sharpe(ex_c1[m], ANN_CRYPTO)}
        report["per_year_table"] = per_year

    report_path = Path(args.report) if args.report else (_ROOT / "docs" / "eodhd_s5_evaluation_2026_10.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote report to %s", report_path)

    if args.append_ledger:
        ledger_path = _ROOT / "docs" / "s5_trial_history.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"entries": []}
        ledger["entries"].append({
            "appended_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": prereg.bars_fingerprint(),
            "trial_daily_sharpes": trial_sharpes.tolist(),
            "variant_names": list(prereg.VARIANTS),
            "prior_trials_used": prereg.DSR["prior_trials"],
            "tier": tier,
        })
        ledger_path.write_text(json.dumps(ledger, indent=2))
        log.info("appended ledger entry to %s", ledger_path)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--out", required=True)
    e.add_argument("--report")
    e.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"build": cmd_build, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
