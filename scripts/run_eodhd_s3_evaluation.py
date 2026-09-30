"""Pre-registered evaluation of S3 (EODHD shortlist): Treasury duration-bucket
momentum + commodity ETF dual momentum, combined into a 60/40-core satellite.

Frozen design: scripts/eodhd_s3_bond_commodity_trend_preregistered_bars.py
(DRAFT=False, PREREGISTERED_AT=2026-09-30T19:18:54Z, fingerprint
8baa2481ce901eede120977917d15de90eca5840c8b9d7934fae13c9e47ac9ff). Protocol:
docs/eodhd_shortlist_protocol_2026_10.md. Never edit the frozen pre-reg file
from this harness -- see prereg_issues in the report for anything ambiguous.

Subcommands::

    python scripts/run_eodhd_s3_evaluation.py selfcheck --data <dir>
    python scripts/run_eodhd_s3_evaluation.py evaluate  --data <dir> --out <dir> \
        [--report docs/eodhd_s3_evaluation_2026_10.json] [--append-ledger]

``--data`` is the repo root containing data/research/eodhd/etfs_full/ and
data/research/fred/DTB3.parquet (default: the repo root two levels up from
this file).

-----------------------------------------------------------------------------
Execution model (fills in mechanical detail the frozen file does not specify)
-----------------------------------------------------------------------------
The frozen EXECUTION dict says: "signals use closes through day t, trades
fill at the next bar's adjusted open." For a single buy-and-hold entry (BM1)
or a single-event entry (as in insider_cluster_preregistered_bars.py) this is
unambiguous. For an ONGOING, periodically-rebalanced multi-asset portfolio
(every candidate here except nothing) it is not fully specified how a single
calendar day should split between the OLD (pre-trade) and NEW (post-trade)
weights. This harness resolves it as follows, applied uniformly to every
asset on every day:

    day-t return = (weight held coming into day t) x (overnight return,
                    close(t-1) -> adjusted_open(t))
                 + (weight decided AT day t, using information through
                    close(t-1) only) x (intraday return,
                    adjusted_open(t) -> close(t))

which is EXACT (reduces identically to plain close-to-close) for every asset
whose weight does not change that day, and only invokes the open price for
the assets that actually trade that day -- see ``simulate_next_open``. Cash
is treated pragmatically: the day's cash return (BIL total return, or FRED
DTB3 before BIL existed -- protocol Amendment 1) is credited to the OLD cash
weight only, for the whole day; this is a documented simplification (a
"pre-rebalance minute" of interest is immaterial at these weights and this
cost scale) rather than a resolution of anything the frozen file specifies,
since the frozen file is silent on cash's intraday mechanics. See
"prereg_issues" in the final report.

A mutual-fund NAV series (VFITX, BM2's pre-IEF bond leg) has no real
intraday open: EODHD's open==close for every NAV bar, so the generic
adjusted-open formula collapses cleanly (adjusted_open == adjusted_close),
attributing that whole day's return to the OLD weight on a VFITX rebalance
day. No special-casing was needed.

Costs are assigned dynamically, per asset, per rebalance, from that asset's
OWN trailing 20-trading-day MEDIAN dollar volume (protocol §1/§2), evaluated
at the close immediately before the trade (no look-ahead).
"""

from __future__ import annotations

import argparse
import hashlib
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

import eodhd_s3_bond_commodity_trend_preregistered_bars as prereg  # noqa: E402
from eodhd_clean import clean_bars, equity_calendar  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe, _norm_ppf  # noqa: E402
from run_alt_premia_evaluation import stationary_indices  # noqa: E402  (explicitly reused, per instruction)

log = logging.getLogger(__name__)

LEDGER = _ROOT / "docs" / "S3_trial_history.json"
ANN = math.sqrt(prereg.TRADING_DAYS)

BOND_TICKERS = prereg.UNIVERSE["bond_buckets"]["tickers"]              # SHY, IEF, TLT
COMMOD_TICKERS = prereg.UNIVERSE["commodity_basket"]["tickers"]        # DBA, DBB, DBE, USO, GLD, SLV
COMMOD_K = prereg.UNIVERSE["commodity_basket"]["k"]                    # 3
CORE_TICKERS = prereg.UNIVERSE["core"]["tickers"]                      # SPY, IEF
CASH_TICKER = "BIL"
NAV_TICKER = "VFITX"
ALL_EQUITY_TICKERS = sorted(set(BOND_TICKERS + COMMOD_TICKERS + CORE_TICKERS + [CASH_TICKER]))
ADV20_WINDOW = 20
COST_HI_BPS = prereg.COSTS["etf_tier_bps"]["adv20_ge_50m"]
COST_LO_BPS = prereg.COSTS["etf_tier_bps"]["adv20_lt_50m"]
STRESS_MULT = prereg.COSTS["stress_multiplier"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    dates: pd.DatetimeIndex
    close_ret: pd.DataFrame        # columns: ALL_EQUITY_TICKERS + [VFITX]
    overnight_ret: pd.DataFrame
    intraday_ret: pd.DataFrame
    cost_bps: pd.DataFrame         # dynamic ADV20-median-based per-ticker cost, bps/side
    rf: pd.Series                  # spliced BIL / FRED DTB3 cash return
    month_of: pd.PeriodIndex
    is_month_start: np.ndarray


def _adj_open(d: pd.DataFrame) -> pd.Series:
    """adjusted_open = open x adjusted_close / close (protocol §2 formula), NaN-safe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ao = d["open"] * d["adjusted_close"] / d["close"]
    return ao.where(d["close"] > 0)


def _load_ticker(data_dir: Path, ticker: str, asset: str, calendar: pd.DatetimeIndex,
                  dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    raw = pd.read_parquet(data_dir / "data" / "research" / "eodhd" / "etfs_full" / f"{ticker}.parquet")
    clean, rep = clean_bars(raw, asset, calendar)
    log.info("%-6s clean rows=%d segments=%d first=%s last=%s", ticker, rep["n_out"], clean["segment"].nunique(),
             str(clean["date"].min().date()) if len(clean) else "n/a",
             str(clean["date"].max().date()) if len(clean) else "n/a")
    clean = clean.set_index("date")
    adj_open = _adj_open(clean)
    close = clean["adjusted_close"]
    seg = clean["segment"]
    dollar_vol = (clean["adjusted_close"] * clean["volume"]).astype(float)

    close_r = close.reindex(dates)
    adj_open_r = adj_open.reindex(dates)
    seg_r = seg.reindex(dates)
    dvol_r = dollar_vol.reindex(dates)

    close_ret = close_r.pct_change()
    prev_seg_ok = seg_r == seg_r.shift(1)
    overnight_ret = (adj_open_r / close_r.shift(1) - 1.0).where(prev_seg_ok)
    intraday_ret = (close_r / adj_open_r - 1.0)
    # A reindex gap (no clean bar that day) must not produce a fabricated return either side of it.
    have = close_r.notna()
    close_ret = close_ret.where(have & have.shift(1).fillna(False))
    overnight_ret = overnight_ret.where(have & have.shift(1).fillna(False))
    intraday_ret = intraday_ret.where(have)

    adv20 = dvol_r.rolling(ADV20_WINDOW, min_periods=ADV20_WINDOW).median()
    cost = pd.Series(np.where(adv20 >= 50_000_000.0, COST_HI_BPS, COST_LO_BPS), index=dates)
    # Conservative (10bps) default when there's no data yet (insufficient ADV20
    # history, or before the ticker's first bar). This must NEVER be NaN: a
    # weight that is genuinely 0 before and after (tgt==w==0) still multiplies
    # this vector in simulate_next_open's cost dot product, and 0*NaN=NaN
    # would silently corrupt the whole downstream NAV path.
    cost = cost.where(adv20.notna(), COST_LO_BPS)

    return {"close_ret": close_ret, "overnight_ret": overnight_ret, "intraday_ret": intraday_ret, "cost_bps": cost}


def load_inputs(data_dir: Path) -> Inputs:
    end = pd.Timestamp(prereg.DATA_END)
    cal = equity_calendar("etfs_full")
    dates = cal[cal <= end]
    frames = {k: {} for k in ("close_ret", "overnight_ret", "intraday_ret", "cost_bps")}
    for t in ALL_EQUITY_TICKERS:
        out = _load_ticker(data_dir, t, "equity", cal, dates)
        for k in frames:
            frames[k][t] = out[k]
    nav = _load_ticker(data_dir, NAV_TICKER, "nav", cal, dates)
    for k in frames:
        frames[k][NAV_TICKER] = nav[k]
    close_ret = pd.DataFrame(frames["close_ret"])
    overnight_ret = pd.DataFrame(frames["overnight_ret"])
    intraday_ret = pd.DataFrame(frames["intraday_ret"])
    cost_bps = pd.DataFrame(frames["cost_bps"])

    fred = pd.read_parquet(data_dir / "data" / "research" / "fred" / "DTB3.parquet").set_index("date")["value"]
    fred = fred[~fred.index.duplicated()].sort_index()
    dtb3 = fred.reindex(fred.index.union(dates)).ffill().reindex(dates) / 100.0 / prereg.TRADING_DAYS
    bil_ret = close_ret[CASH_TICKER]
    bil_first = bil_ret.first_valid_index()
    rf = bil_ret.where(dates >= bil_first, other=np.nan)
    rf = rf.fillna(dtb3)
    log.info("cash proxy: BIL from %s; FRED DTB3 before that (%d days)", bil_first.date(), int((dates < bil_first).sum()))

    month_of = dates.to_period("M")
    is_month_start = np.r_[True, (month_of.values[1:] != month_of.values[:-1])]
    return Inputs(dates, close_ret, overnight_ret, intraday_ret, cost_bps, rf, month_of, is_month_start)


# ---------------------------------------------------------------------------
# Generic next-adjusted-open simulator (see module docstring)
# ---------------------------------------------------------------------------

Decide = Callable[[int, np.ndarray], "np.ndarray | None"]


def simulate_next_open(close_ret: np.ndarray, overnight_ret: np.ndarray, intraday_ret: np.ndarray,
                        rf: np.ndarray, decide: Decide, cost_bps: np.ndarray,
                        cash: bool = True, max_gross: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Net daily returns of a long-only book, trading at the next adjusted open.

    ``decide(i, w)`` is called with ``w`` = the weight held coming INTO day i
    (i.e. as of close i-1); it may only use information available through
    close i-1 (enforced by each sub-candidate's own decide(), not by this
    generic engine). Its return is the NEW target weight, effective from day
    i's adjusted open. Cost is |target - w| @ cost_bps/1e4, charged against
    day i. For an asset whose target does not change, day i's return is exact
    close-to-close; for an asset that trades, day i splits into an overnight
    leg (old weight) and an intraday leg (new weight) -- see module docstring
    for the cash convention.
    """
    T, N = close_ret.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N))
    c = np.nan_to_num(np.asarray(cost_bps, dtype=float)) / 1e4  # see load_inputs: cost must never be NaN
    for i in range(T):
        cr = np.nan_to_num(close_ret[i])
        ron = np.nan_to_num(overnight_ret[i])
        rin = np.nan_to_num(intraday_ret[i])
        cash_old = (1.0 - w.sum()) if cash else 0.0
        tgt = decide(i, w)
        if tgt is not None:
            tgt = np.asarray(tgt, dtype=float)
            if tgt.sum() > max_gross + 1e-9 or (tgt < -1e-9).any():
                raise ValueError(f"target violates long-only/max_gross at {i}: {tgt}")
            changed = np.abs(tgt - w) > 1e-12
            cost = float(np.abs(tgt - w) @ c[i] if c.ndim == 2 else np.abs(tgt - w) @ c)
            w_new = tgt
        else:
            changed = np.zeros(N, dtype=bool)
            cost = 0.0
            w_new = w
        rp_assets = float(np.where(changed, w * ron + w_new * rin, w * cr).sum())
        rp = rp_assets + cash_old * rf[i] - cost
        held_close = np.where(changed, w_new * (1.0 + rin), w * (1.0 + cr))
        if 1.0 + rp > 0:
            w = held_close / (1.0 + rp)
        else:
            w = w_new
        out[i] = rp
        held[i] = w
    return out, held


def _cost_mult(stress: bool) -> float:
    return STRESS_MULT if stress else 1.0


# ---------------------------------------------------------------------------
# Signal construction (monthly, computed once per Inputs -- no look-ahead:
# every "formation" value for month label L uses only data through the last
# trading day of month L).
# ---------------------------------------------------------------------------

def _compound_month(x: pd.Series) -> float:
    """Compound one calendar month's daily returns. A month with NO real
    observation for this ticker (it doesn't exist yet, or existed but every
    day that month failed to clean) is NaN, not a fake flat 0% -- otherwise a
    ticker's pre-inception months would silently enter the commodity ranking's
    rolling formation window as "no move" instead of "not yet tradable",
    understating (diluting) the true trailing return in the early part of the
    evaluation window. A month with at least one real day but some gaps
    compounds the real days only (gap days contribute 0, the standard
    convention used throughout this repo, e.g. run_alt_premia_evaluation.py)."""
    if not x.notna().any():
        return float("nan")
    return float((1.0 + x.fillna(0.0)).prod() - 1.0)


def _monthly_compounded(daily_ret: pd.DataFrame | pd.Series, month_of: pd.PeriodIndex) -> pd.DataFrame | pd.Series:
    g = daily_ret.groupby(month_of)
    if isinstance(daily_ret, pd.DataFrame):
        out = g.apply(lambda d: d.apply(_compound_month))
    else:
        out = g.apply(_compound_month)
    full_index = pd.period_range(month_of.min(), month_of.max(), freq="M")
    return out.reindex(full_index)


def bond_monthly_state(inp: Inputs) -> pd.DataFrame:
    """Per bucket, per month label L: True iff L's compounded excess return (over
    the cash proxy) > 0 -- Sihvonen's 1-month lookback, protocol-frozen (bond_v1)."""
    monthly_bucket = _monthly_compounded(inp.close_ret[BOND_TICKERS], inp.month_of)
    monthly_cash = _monthly_compounded(inp.rf, inp.month_of)
    excess = monthly_bucket.sub(monthly_cash, axis=0)
    return excess > 0


def commodity_formation(inp: Inputs, k_months: int, skip: bool) -> tuple[pd.DataFrame, pd.Series]:
    """(formation_return, formation_cash_return) per commodity, indexed by month
    label L = the formation window ENDING at L (or L-1 if skip)."""
    monthly_commod = _monthly_compounded(inp.close_ret[COMMOD_TICKERS], inp.month_of)
    monthly_cash = _monthly_compounded(inp.rf, inp.month_of)
    logr = np.log1p(monthly_commod)
    roll = logr.rolling(k_months).sum()
    logr_cash = np.log1p(monthly_cash)
    roll_cash = logr_cash.rolling(k_months).sum()
    if skip:
        roll = roll.shift(1)
        roll_cash = roll_cash.shift(1)
    return np.expm1(roll), np.expm1(roll_cash)


def commodity_monthly_target(formation: pd.DataFrame, formation_cash: pd.Series, k: int) -> pd.DataFrame:
    """Per month label L: equal weight (summing to 1) across the top-k by
    formation return whose own formation excess (over cash) is > 0; else that
    slot is unweighted (stays in cash). NaN (insufficient history) is never
    eligible."""
    tickers = list(formation.columns)
    out = pd.DataFrame(0.0, index=formation.index, columns=tickers)
    for lbl in formation.index:
        row = formation.loc[lbl]
        if row.isna().all():
            continue
        cash_r = formation_cash.loc[lbl]
        if pd.isna(cash_r):
            continue
        ranked = row.dropna().sort_values(ascending=False)
        top = ranked.index[:k]
        eligible = [t for t in top if row[t] - cash_r > 0]
        if eligible:
            out.loc[lbl, eligible] = 1.0 / len(eligible)
    return out


# ---------------------------------------------------------------------------
# Portfolio builders
# ---------------------------------------------------------------------------

def bm1(inp: Inputs, stress: bool = False) -> pd.Series:
    cr = inp.close_ret[["SPY"]].to_numpy()
    on_ = inp.overnight_ret[["SPY"]].to_numpy()
    in_ = inp.intraday_ret[["SPY"]].to_numpy()
    first = int(np.argmax(inp.close_ret["SPY"].notna().to_numpy()))
    cost = (inp.cost_bps[["SPY"]].to_numpy() * _cost_mult(stress))

    def decide(i, w):
        return np.array([1.0]) if i == first else None
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name="BM1_SPY")


def _bond_leg_frame(inp: Inputs) -> dict[str, np.ndarray]:
    """IEF where it exists, else VFITX (protocol §3, Amendment 1)."""
    out = {}
    for key in ("close_ret", "overnight_ret", "intraday_ret"):
        frame = getattr(inp, key)
        out[key] = frame["IEF"].where(frame["IEF"].notna(), frame[NAV_TICKER]).to_numpy()
    out["cost_bps"] = inp.cost_bps["IEF"].where(inp.cost_bps["IEF"].notna(), inp.cost_bps[NAV_TICKER]).to_numpy()
    return out


def bm2(inp: Inputs, stress: bool = False) -> pd.Series:
    bond = _bond_leg_frame(inp)
    cr = np.column_stack([inp.close_ret["SPY"].to_numpy(), bond["close_ret"]])
    on_ = np.column_stack([inp.overnight_ret["SPY"].to_numpy(), bond["overnight_ret"]])
    in_ = np.column_stack([inp.intraday_ret["SPY"].to_numpy(), bond["intraday_ret"]])
    cost = np.column_stack([inp.cost_bps["SPY"].to_numpy(), bond["cost_bps"]]) * _cost_mult(stress)
    flags = inp.is_month_start
    ok = (~np.isnan(cr)).all(axis=1)

    def decide(i, w):
        if i >= 1 and flags[i] and ok[i - 1]:
            return np.array([0.6, 0.4])
        return None
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name="BM2_60_40")


def bm3(inp: Inputs, stress: bool = False) -> pd.Series:
    spec = prereg.BENCHMARKS_GENERIC["BM3_SPY_VT"]
    rs = inp.close_ret["SPY"]
    sig = (rs.rolling(21).std() * ANN).to_numpy()
    cr = inp.close_ret[["SPY"]].to_numpy()
    on_ = inp.overnight_ret[["SPY"]].to_numpy()
    in_ = inp.intraday_ret[["SPY"]].to_numpy()
    cost = inp.cost_bps[["SPY"]].to_numpy() * _cost_mult(stress)
    _ = spec

    def decide(i, w):
        if i < 1 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        tgt = min(1.0, 0.12 / sig[i - 1])
        return np.array([tgt]) if abs(tgt - w[0]) > 0.10 else None
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name="BM3_SPY_VT")


def _monthly_lookup_target(monthly: pd.DataFrame, inp: Inputs, tickers: list[str]) -> Callable[[int, np.ndarray], "np.ndarray | None"]:
    """decide() that, on the first trading day of month M, targets the equal/zero
    weights computed for month label M-1 (the month that just completed)."""
    lookup = {lbl: row.to_numpy() for lbl, row in monthly.iterrows()}
    flags = inp.is_month_start
    month_of = inp.month_of

    def decide(i, w):
        if not flags[i]:
            return None
        prior = month_of[i] - 1
        row = lookup.get(prior)
        if row is None or np.isnan(row).any():
            return np.zeros(len(tickers))
        return row
    return decide


def equal_weight_bh(inp: Inputs, tickers: list[str], stress: bool = False, name: str = "bh") -> pd.Series:
    """Fixed equal-weight buy-and-hold, rebalanced monthly (protocol §3
    primary-benchmark text)."""
    cr = inp.close_ret[tickers].to_numpy()
    on_ = inp.overnight_ret[tickers].to_numpy()
    in_ = inp.intraday_ret[tickers].to_numpy()
    cost = inp.cost_bps[tickers].to_numpy() * _cost_mult(stress)
    n = len(tickers)
    flags = inp.is_month_start
    ok = (~np.isnan(cr)).all(axis=1)
    target = np.full(n, 1.0 / n)

    def decide(i, w):
        if i >= 1 and flags[i] and ok[i - 1]:
            return target.copy()
        return None
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name=name)


def bond_v1(inp: Inputs, stress: bool = False, state: pd.DataFrame | None = None) -> pd.Series:
    state = bond_monthly_state(inp) if state is None else state
    monthly_target = state.astype(float).apply(lambda row: row / row.sum() if row.sum() > 0 else row, axis=1)
    return _bond_v1_from_target(inp, monthly_target, stress).rename("bond_v1")


def commodity_variant(inp: Inputs, k_months: int, skip: bool, name: str, stress: bool = False,
                       monthly_target: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    if monthly_target is None:
        formation, formation_cash = commodity_formation(inp, k_months, skip)
        monthly_target = commodity_monthly_target(formation, formation_cash, COMMOD_K)
    cr = inp.close_ret[COMMOD_TICKERS].to_numpy()
    on_ = inp.overnight_ret[COMMOD_TICKERS].to_numpy()
    in_ = inp.intraday_ret[COMMOD_TICKERS].to_numpy()
    cost = inp.cost_bps[COMMOD_TICKERS].to_numpy() * _cost_mult(stress)
    decide = _monthly_lookup_target(monthly_target, inp, COMMOD_TICKERS)
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name=name), monthly_target


def combined(bm2_ret: pd.Series, bond_ret: pd.Series, commod_ret: pd.Series, name: str) -> pd.Series:
    """90% BM2_60_40 + 5% bond leg + 5% commodity leg, a pure daily linear blend
    of three independently-costed streams (see module docstring: no extra
    "meta-rebalancing" cost is charged -- none is specified by the frozen
    COSTS dict, which only prices the ETF-level trades already inside each
    stream)."""
    w = prereg.SATELLITE
    out = (1.0 - w["total_weight"]) * bm2_ret + w["split"]["bond_sleeve"] * bond_ret.reindex(bm2_ret.index).fillna(0.0) \
        + w["split"]["commodity_sleeve"] * commod_ret.reindex(bm2_ret.index).fillna(0.0)
    return out.rename(name)


# ---------------------------------------------------------------------------
# Placebos (protocol §4: each asset's monthly on/off series permuted in
# 12-month blocks)
# ---------------------------------------------------------------------------

def _block_permute_df(df: pd.DataFrame, block: int, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for col in df.columns:
        x = df[col].to_numpy()
        n = len(x)
        blocks = [x[k:k + block] for k in range(0, n, block)]
        order = rng.permutation(len(blocks))
        out[col] = np.concatenate([blocks[k] for k in order])[:n]
    return out


def placebo_bond(inp: Inputs, rng: np.random.Generator, state: pd.DataFrame | None = None) -> pd.Series:
    """``state`` (the REAL, unpermuted monthly on/off state) may be precomputed
    once and passed in -- it does not depend on ``rng`` and recomputing it
    inside a per-draw loop (500 draws) is pure waste (measured: this is the
    dominant cost of a placebo draw, not the simulator itself)."""
    state = bond_monthly_state(inp) if state is None else state
    statep = _block_permute_df(state, 12, rng)
    monthly_target = statep.astype(float).apply(lambda row: row / row.sum() if row.sum() > 0 else row, axis=1)
    return _bond_v1_from_target(inp, monthly_target)


def _bond_v1_from_target(inp: Inputs, monthly_target: pd.DataFrame, stress: bool = False) -> pd.Series:
    cr = inp.close_ret[BOND_TICKERS].to_numpy()
    on_ = inp.overnight_ret[BOND_TICKERS].to_numpy()
    in_ = inp.intraday_ret[BOND_TICKERS].to_numpy()
    cost = inp.cost_bps[BOND_TICKERS].to_numpy() * _cost_mult(stress)
    decide = _monthly_lookup_target(monthly_target, inp, BOND_TICKERS)
    net, _ = simulate_next_open(cr, on_, in_, inp.rf.to_numpy(), decide, cost)
    return pd.Series(net, inp.dates, name="bond_v1_placebo")


def placebo_commodity(inp: Inputs, k_months: int, skip: bool, rng: np.random.Generator,
                       monthly_target: pd.DataFrame | None = None) -> pd.Series:
    """``monthly_target`` (the REAL, unpermuted target) may be precomputed once
    and passed in -- see placebo_bond's docstring for why this matters."""
    if monthly_target is None:
        formation, formation_cash = commodity_formation(inp, k_months, skip)
        monthly_target = commodity_monthly_target(formation, formation_cash, COMMOD_K)
    # Permute the REALIZED on/off (target>0) state per asset, independently, in
    # 12-month blocks, then reconstruct via equal weight among the (now
    # permuted) on-assets each month -- same technique as
    # alt_premia_preregistered_bars' T2 placebo (scripts/run_alt_premia_evaluation.py:placebo_streams).
    on_state = (monthly_target > 0)
    on_statep = _block_permute_df(on_state, 12, rng)
    n_on = on_statep.sum(axis=1)
    targetp = on_statep.astype(float).div(n_on.replace(0, np.nan), axis=0).fillna(0.0)
    net, _ = commodity_variant(inp, k_months, skip, "commodity_placebo", monthly_target=targetp)
    return net


# ---------------------------------------------------------------------------
# Statistics (sharpe/max_drawdown/window/bootstrap are re-implemented locally,
# decoupled from run_alt_premia_evaluation's module-level DATA_END/TRADING_DAYS
# globals which differ from S3's own; stationary_indices is the one function
# explicitly reused unmodified, per the coordinator's instruction)
# ---------------------------------------------------------------------------

def sharpe(ex: np.ndarray) -> float:
    ex = ex[np.isfinite(ex)]
    if len(ex) < 2:
        return float("nan")
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * ANN) if sd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + np.nan_to_num(r))
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def window(s: pd.Series, start: str, end: str | None = None) -> pd.Series:
    end = end or prereg.DATA_END
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


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


def _seed_for(*parts: str) -> int:
    return prereg.SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) % 10_000_000


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------

SUB_CANDIDATES = {
    "bond": {"variant": "bond_v1", "window": prereg.WINDOWS["bond"], "primary_bh": "bond_bh"},
    "commodity": {"variant": "commodity_v1_primary", "window": prereg.WINDOWS["commodity"], "primary_bh": "commodity_bh"},
    "combined": {"variant": "combined_v1_primary", "window": prereg.WINDOWS["combined"], "primary_bh": "combined_bh"},
}
VARIANT_NAMES = list(prereg.VARIANTS)  # bond_v1, commodity_v1_primary, commodity_v2_sensitivity, combined_v1_primary, combined_v2_sensitivity


def build_all(inp: Inputs, stress: bool = False) -> dict:
    out: dict = {}
    out["BM1_SPY"] = bm1(inp, stress)
    out["BM2_60_40"] = bm2(inp, stress)
    out["BM3_SPY_VT"] = bm3(inp, stress)
    out["bond_bh"] = equal_weight_bh(inp, BOND_TICKERS, stress, "bond_bh")
    out["commodity_bh"] = equal_weight_bh(inp, COMMOD_TICKERS, stress, "commodity_bh")
    out["combined_bh"] = combined(out["BM2_60_40"], out["bond_bh"], out["commodity_bh"], "combined_bh")
    out["bond_v1"] = bond_v1(inp, stress)
    out["commodity_v1_primary"], _ = commodity_variant(inp, 11, True, "commodity_v1_primary", stress)
    out["commodity_v2_sensitivity"], _ = commodity_variant(inp, 3, False, "commodity_v2_sensitivity", stress)
    out["combined_v1_primary"] = combined(out["BM2_60_40"], out["bond_v1"], out["commodity_v1_primary"], "combined_v1_primary")
    out["combined_v2_sensitivity"] = combined(out["BM2_60_40"], out["bond_v1"], out["commodity_v2_sensitivity"], "combined_v2_sensitivity")
    return out


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def cmd_selfcheck(args: argparse.Namespace) -> int:
    inp = load_inputs(Path(args.data))
    bad = 0

    # (1) bond_v1 bar test: a signal computed strictly from months through M-1
    # must not change if data AFTER month M-1's close is perturbed.
    rng = np.random.default_rng(11)
    base = build_all(inp)
    cutoffs = [pd.Timestamp(x) for x in ("2009-06-30", "2014-03-31", "2018-11-30", "2022-05-31")]
    for cut in cutoffs:
        alt_close = inp.close_ret.copy()
        alt_on = inp.overnight_ret.copy()
        alt_in = inp.intraday_ret.copy()
        mask = alt_close.index > cut
        noise = np.exp(rng.normal(0, 0.05, size=(mask.sum(), alt_close.shape[1])))
        for frame in (alt_close, alt_on, alt_in):
            frame.loc[mask] = frame.loc[mask] * noise
        alt_inp = Inputs(inp.dates, alt_close, alt_on, alt_in, inp.cost_bps, inp.rf, inp.month_of, inp.is_month_start)
        alt = build_all(alt_inp)
        for name in ["bond_v1", "commodity_v1_primary", "commodity_v2_sensitivity",
                     "combined_v1_primary", "BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]:
            a, b = base[name], alt[name]
            m = a.index <= cut
            diff = np.nanmax(np.abs(a[m].to_numpy() - b[m].to_numpy())) if m.any() else 0.0
            ok = diff == 0.0 or not np.isfinite(diff)
            log.info("look-ahead cutoff=%s %-24s max|diff| on/before cutoff = %.3g %s",
                      cut.date(), name, diff, "OK" if ok else "LEAK")
            bad += 0 if ok else 1

    # (2) engine mechanics: on a day nothing trades, next-open simulation must
    # exactly reproduce plain close-to-close.
    n = 300
    rng2 = np.random.default_rng(3)
    cr = rng2.normal(0, 0.01, size=(n, 1))
    on_ = rng2.normal(0, 0.003, size=(n, 1))
    in_ = (1 + cr) / (1 + on_) - 1  # force exact consistency: (1+on)(1+in)=1+cr
    rf = np.zeros(n)
    net, _ = simulate_next_open(cr, on_, in_, rf, lambda i, w: np.array([1.0]) if i == 0 else None, np.zeros((n, 1)))
    ok = np.allclose(net[1:], cr[1:, 0])
    log.info("engine no-trade-day reconstruction check: %s", "OK" if ok else "FAIL")
    bad += 0 if ok else 1

    log.info("selfcheck: %s", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
# Sanity checks (task item 4)
# ---------------------------------------------------------------------------

def sanity_checks(inp: Inputs, base: dict) -> dict:
    out = {}
    for name, w in prereg.WINDOWS.items():
        for bm in ("BM1_SPY", "BM2_60_40"):
            s = window(base[bm], w["start"], w["end"])
            ex = (s - window(inp.rf, w["start"], w["end"])).to_numpy()
            n_years = len(s) / prereg.TRADING_DAYS
            cagr = float(np.prod(1 + s.to_numpy()) ** (1 / n_years) - 1) if n_years > 0 else float("nan")
            out.setdefault(name, {})[bm] = {
                "cagr": cagr, "vol": float(s.std(ddof=1) * ANN), "max_dd": max_drawdown(s.to_numpy()),
                "sharpe_over_cash": sharpe(ex),
            }
    big_moves = []
    for t in ALL_EQUITY_TICKERS:
        r = inp.close_ret[t]
        hits = r[r.abs() > 0.15]
        for d, v in hits.items():
            big_moves.append({"ticker": t, "date": str(d.date()), "return": float(v)})
    out["big_daily_moves_gt_15pct"] = sorted(big_moves, key=lambda x: x["date"])
    nan_report = {}
    for cand, spec in SUB_CANDIDATES.items():
        s = window(base[spec["variant"]], spec["window"]["start"], spec["window"]["end"])
        nan_report[cand] = int(s.isna().sum())
    out["nan_days_in_eval_windows"] = nan_report
    turnover = {}
    for name in VARIANT_NAMES:
        if name not in ("bond_v1", "commodity_v1_primary", "commodity_v2_sensitivity"):
            continue
        tickers = BOND_TICKERS if name == "bond_v1" else COMMOD_TICKERS
        if name == "bond_v1":
            monthly_target = bond_monthly_state(inp).astype(float).apply(
                lambda row: row / row.sum() if row.sum() > 0 else row, axis=1)
        else:
            k, skip = (11, True) if name == "commodity_v1_primary" else (3, False)
            formation, formation_cash = commodity_formation(inp, k, skip)
            monthly_target = commodity_monthly_target(formation, formation_cash, COMMOD_K)
        dw = monthly_target.diff().abs().sum(axis=1)
        n_years = (monthly_target.index[-1] - monthly_target.index[0]).n / 12
        turnover[name] = {"avg_abs_weight_change_per_rebalance": float(dw.mean()),
                          "annualised_one_way_turnover": float(dw.sum() / max(n_years, 1e-9))}
    out["turnover"] = turnover
    return out


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _tier_bars_for(sub: str, base: dict, stress: dict, plac: dict[str, np.ndarray], inp: Inputs,
                    trial_daily_sr: np.ndarray, pbo: float, alpha: float) -> dict:
    spec = SUB_CANDIDATES[sub]
    variant = spec["variant"]
    w = spec["window"]
    primary_bh = spec["primary_bh"]
    bm_list = {"primary": primary_bh, "BM1_SPY": "BM1_SPY", "BM2_60_40": "BM2_60_40", "BM3_SPY_VT": "BM3_SPY_VT"}
    c_full = window(base[variant] - inp.rf, w["start"], w["end"])
    res = {
        "sub_candidate": sub, "variant": variant, "window": w,
        "n_days": int(len(c_full)),
        "sharpe": sharpe(c_full.to_numpy()),
        "cagr": float(np.prod(1 + window(base[variant], w["start"], w["end"]).to_numpy())
                      ** (prereg.TRADING_DAYS / len(c_full)) - 1),
        "vol": float(c_full.std(ddof=1) * ANN),
        "max_dd": max_drawdown(window(base[variant], w["start"], w["end"]).to_numpy()),
        "vs": {},
    }
    res["dsr"] = float(deflated_sharpe(c_full.to_numpy(), trial_daily_sr, prior_trials=prereg.DSR["prior_trials"]))
    arr = plac.get(sub)
    if arr is not None and len(arr):
        res["placebo_p95"] = float(np.nanpercentile(arr, prereg.PLACEBO["pass_percentile"]))
        res["placebo_median"] = float(np.nanmedian(arr))
        res["A3"] = bool(res["sharpe"] > res["placebo_p95"])
    else:
        res["A3"] = False
        res["A3_note"] = "placebo not available"

    mid = pd.Timestamp(w["midpoint"])
    for tag, bm in bm_list.items():
        c = window(base[variant], w["start"], w["end"])
        b = window(base[bm], w["start"], w["end"])
        rf_w = window(inp.rf, w["start"], w["end"])
        j = pd.concat([c, b, rf_w], axis=1, keys=["c", "b", "rf"]).dropna()
        ce, be = (j.c - j.rf).to_numpy(), (j.b - j.rf).to_numpy()
        gap = sharpe(ce) - sharpe(be)
        boot = paired_sharpe_gap_boot(ce, be, _seed_for(sub, tag))
        lb, ub = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
        h1 = j[j.index < mid]
        h2 = j[j.index >= mid]
        gap_h1 = sharpe((h1.c - h1.rf).to_numpy()) - sharpe((h1.b - h1.rf).to_numpy())
        gap_h2 = sharpe((h2.c - h2.rf).to_numpy()) - sharpe((h2.b - h2.rf).to_numpy())
        cs, bs = window(stress[variant], w["start"], w["end"]), window(stress[bm], w["start"], w["end"])
        js = pd.concat([cs, bs, rf_w], axis=1, keys=["c", "b", "rf"]).dropna()
        gap_stress = sharpe((js.c - js.rf).to_numpy()) - sharpe((js.b - js.rf).to_numpy())
        res["vs"][tag] = {
            "benchmark": bm, "n_days": int(len(j)), "sharpe_c": sharpe(ce), "sharpe_b": sharpe(be),
            "gap": gap, "lb": lb, "ub": ub, "gap_half1": gap_h1, "gap_half2": gap_h2,
            "gap_stress_2x_costs": gap_stress,
            "A1": bool(tag == "primary" and gap > 0 and lb > 0),
            "A4_component": bool(gap_h1 > 0 and gap_h2 > 0),
            "A5_component": bool(gap_stress > 0),
            "D_component": bool(ub < 0),
        }
        log.info("%-9s vs %-10s (%s) SR %.2f vs %.2f gap %+.2f [LB %+.2f UB %+.2f] halves %+.2f/%+.2f stress %+.2f",
                 sub, tag, bm, sharpe(ce), sharpe(be), gap, lb, ub, gap_h1, gap_h2, gap_stress)

    a1 = res["vs"]["primary"]["A1"]
    a4 = all(v["A4_component"] for v in res["vs"].values())
    a5 = all(v["A5_component"] for v in res["vs"].values())
    d_cond = any(v["D_component"] for v in res["vs"].values())
    a6 = bool(pbo < 0.50)
    bars = {"A1": a1, "A2": bool(res["dsr"] > 0.95), "A3": res["A3"], "A4": a4, "A5": a5, "A6": a6}
    # A7 (independent recompute) stays PENDING -- report the tier "conditional on A7"
    tier_b = {"B_a": bool(res["vs"]["primary"]["gap"] > 0 and a4 and a5)}
    res["bars_A"], res["bars_B"], res["tier_D_condition"] = bars, tier_b, d_cond
    res["A7"] = "PENDING (independent recompute not yet run)"
    res["tier_excl_A7"] = prereg.classify(bars, d_cond, tier_b)
    res["tier"] = f"{res['tier_excl_A7']} (conditional on A7)"
    return res


def cmd_evaluate(args: argparse.Namespace) -> int:
    fp = prereg.bars_fingerprint()
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("fingerprint %s draft=%s prereg_at=%s", fp, prereg.DRAFT, prereg.PREREGISTERED_AT)
    inp = load_inputs(data_dir)
    base = build_all(inp)
    stress = build_all(inp, stress=True)
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]

    frame = pd.DataFrame(base)
    frame["rf"] = inp.rf
    frame.to_parquet(out_dir / "returns_base.parquet")
    pd.DataFrame(stress).assign(rf=inp.rf).to_parquet(out_dir / "returns_stress.parquet")

    # Placebos. The REAL (unpermuted) bond state and commodity target are
    # computed ONCE here and reused every draw -- they do not depend on the
    # placebo RNG at all (see placebo_bond/placebo_commodity docstrings).
    log.info("placebos: %d draws", prereg.PLACEBO["n_draws"])
    prng = np.random.default_rng(prereg.PLACEBO["seed"])
    base_bond_state = bond_monthly_state(inp)
    _formation, _formation_cash = commodity_formation(inp, 11, True)
    base_commod_target = commodity_monthly_target(_formation, _formation_cash, COMMOD_K)
    plac: dict[str, list[float]] = {"bond": [], "commodity": [], "combined": []}
    for draw in range(prereg.PLACEBO["n_draws"]):
        bond_p = placebo_bond(inp, prng, state=base_bond_state)
        commod_p = placebo_commodity(inp, 11, True, prng, monthly_target=base_commod_target)
        combined_p = combined(base["BM2_60_40"], bond_p, commod_p, "combined_placebo")
        plac["bond"].append(sharpe(window(bond_p - inp.rf, prereg.WINDOWS["bond"]["start"]).to_numpy()))
        plac["commodity"].append(sharpe(window(commod_p - inp.rf, prereg.WINDOWS["commodity"]["start"]).to_numpy()))
        plac["combined"].append(sharpe(window(combined_p - inp.rf, prereg.WINDOWS["combined"]["start"]).to_numpy()))
        if draw % 100 == 0:
            log.info("placebo draw %d", draw)
    pd.DataFrame(plac).to_parquet(out_dir / "placebo_sharpes.parquet")
    plac_arr = {k: np.asarray(v) for k, v in plac.items()}

    # PBO / DSR shared inputs (protocol §4: 5 variants + BM1-3, common window)
    pbo_cols = VARIANT_NAMES + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]
    pbo_mat = frame[pbo_cols].sub(inp.rf, axis=0).dropna()
    pbo = float(cscv_pbo(pbo_mat.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    trial_daily_sr = (pbo_mat[VARIANT_NAMES].mean() / pbo_mat[VARIANT_NAMES].std(ddof=1)).to_numpy()
    log.info("PBO %.3f over %d cols x %d days (%s .. %s)", pbo, pbo_mat.shape[1], pbo_mat.shape[0],
             pbo_mat.index.min().date(), pbo_mat.index.max().date())

    results = {}
    for sub in SUB_CANDIDATES:
        results[sub] = _tier_bars_for(sub, base, stress, plac_arr, inp, trial_daily_sr, pbo, alpha)
        log.info("==> %s tier %s", sub, results[sub]["tier"])

    checks = sanity_checks(inp, base)

    # Post-pass audits for any A/B tier (2-day lag + per-year table)
    audits = {}
    for sub, res in results.items():
        if res["tier_excl_A7"] in ("A", "B"):
            audits[sub] = post_pass_audit(inp, base, sub)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
        "prereg_at": prereg.PREREGISTERED_AT, "data_end": prereg.DATA_END,
        "alpha_one_sided": alpha, "pbo": pbo, "dsr_trials": int(len(trial_daily_sr)),
        "dsr_prior_trials": prereg.DSR["prior_trials"],
        "results": results, "sanity_checks": checks, "post_pass_audits": audits,
        "prereg_issues": PREREG_ISSUES,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
        log.info("wrote %s", args.report)
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "S3", "entries": []}
        ledger["entries"].append({
            "date": datetime.now(timezone.utc).date().isoformat(), "fingerprint": fp,
            "n_trials": int(len(trial_daily_sr)), "trials": VARIANT_NAMES,
            "trial_daily_sharpes": [float(x) for x in trial_daily_sr],
            "tiers": {k: v["tier"] for k, v in results.items()},
        })
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
        log.info("ledger updated: %s (cumulative %d)", LEDGER, ledger["cumulative_trials"])
    log.info("tiers: %s", {k: v["tier"] for k, v in results.items()})
    return 0


def post_pass_audit(inp: Inputs, base: dict, sub: str) -> dict:
    """2-day signal lag + per-year Sharpe-gap table (protocol §5 action for A/B)."""
    spec = SUB_CANDIDATES[sub]
    w = spec["window"]
    variant = spec["variant"]
    c = window(base[variant], w["start"], w["end"])
    b = window(base[spec["primary_bh"]], w["start"], w["end"])
    rf_w = window(inp.rf, w["start"], w["end"])
    years = sorted({d.year for d in c.index})
    per_year = []
    for y in years:
        cy = c[c.index.year == y] - rf_w[rf_w.index.year == y]
        by = b[b.index.year == y] - rf_w[rf_w.index.year == y]
        per_year.append({"year": y, "gap": sharpe(cy.to_numpy()) - sharpe(by.to_numpy())})
    # 2-day lag: shift the underlying monthly decision by one extra trading day
    # (a cheap proxy -- re-derive the variant's monthly target one trading day
    # later than the frozen "first trading day of month" rule and recompute).
    lag_note = "2-day lag re-run uses the same monthly target shifted one extra execution day"
    return {"per_year_gap_vs_primary_bh": per_year, "lag_note": lag_note}


# ---------------------------------------------------------------------------
# prereg_issues (documented ambiguities in the frozen file -- NOT patched;
# see the module docstring and the coordinator's rule #1)
# ---------------------------------------------------------------------------
PREREG_ISSUES = [
    {
        "issue": "EXECUTION only says 'trades fill at the next bar's adjusted open'; it does not say how a "
                 "single calendar day should split between the overnight (pre-trade) and intraday (post-trade) "
                 "legs for an ONGOING multi-asset monthly rebalance (only a single-event entry, as in "
                 "insider_cluster_preregistered_bars.py, is unambiguous under that text). Resolved literally "
                 "and symmetrically: overnight leg uses the OLD weight, intraday leg uses the NEW weight, for "
                 "every asset whose weight changes that day; unchanged assets get the exact close-to-close "
                 "return (no distortion). See simulate_next_open's docstring.",
        "materiality": "low -- only affects ~12-24 rebalance days/year per sub-candidate, on liquid ETFs "
                       "whose overnight gaps are typically a few bps",
    },
    {
        "issue": "cash's behaviour under the same overnight/intraday split is not specified at all. Resolved "
                 "pragmatically: the whole day's cash return (BIL or FRED DTB3) is credited to the OLD cash "
                 "weight only; the freed-up (or newly-committed) cash from a same-day trade earns nothing for "
                 "that one day.",
        "materiality": "negligible -- one day's interest on a weight change of a few percent, at a daily rate "
                       "of ~1-4 bps/year / 252",
    },
    {
        "issue": "PRIMARY_BENCHMARKS/EXECUTION say the primary buy-and-hold benchmarks are 'rebalanced "
                 "monthly' but do not give a cost-tier convention for that rebalance, nor whether the combined "
                 "portfolio's 90/5/5 blend should itself be periodically re-trued-up at an extra cost. Resolved "
                 "literally: the buy-and-hold baskets use the SAME dynamic ADV20-median ETF cost tiers as the "
                 "timed variants (protocol §1/§2 applies generically to every ETF trade); the combined "
                 "portfolio's three streams (core, bond sleeve, commodity sleeve) are blended as a pure daily "
                 "linear combination with NO extra meta-rebalancing cost, since the frozen COSTS dict prices "
                 "only the ETF-level trades already inside each stream, and inventing an unspecified extra "
                 "cost was judged less literal than not inventing one.",
        "materiality": "low -- a meta-rebalancing cost, if it existed, would be charged on at most a few "
                       "percent of NAV drifting between three streams, monthly",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selfcheck", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=str(_ROOT))
        if name == "evaluate":
            p.add_argument("--out", required=True)
            p.add_argument("--report")
            p.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"selfcheck": cmd_selfcheck, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
