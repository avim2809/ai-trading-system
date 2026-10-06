"""PHASE 2 evaluation harness for S4 (52-week-high proximity, liquid US stocks).

Frozen design: ``scripts/eodhd_s4_52wk_high_preregistered_bars.py`` (fingerprinted,
DRAFT=False). This file implements it literally; it is never edited to patch a bug
or ambiguity found in the frozen file -- any such issue is implemented as the most
literal reading and reported separately (see ``prereg_issues`` in the output report
and this session's final message).

Universe/signal panel: ``scripts/eodhd_s4_build_panel.py`` -> panel.parquet (ticker,
month_end, price, adv63, n_bars_in_segment, ratio_52wk, entry_date). That panel is
unchanged by cleaning-rule v2's label-only fingerprint update (behaviour identical);
this harness re-asserts that at import via the frozen module's own assertion.

Architecture (memory budget: peak RSS < 1.2 GB, host shared with live trading + 4
other agents):
    - Two dense ``(T, M)`` float32 matrices over the WINDOW's trading-day calendar
      (``T`` ~= 6,958 days) and the ``M`` ~= 4,911 tickers that are EVER in the
      top-1000-by-adv63 population at any month-end in WINDOW (a strict superset of
      every candidate's holdings and both PRIMARY benchmarks, since top-500 subset
      top-1000 at the same ranking): ``RET`` (adjusted close-to-close return, NaN off
      the ticker's own trading days) and ``ENTRY_RET`` (raw close/open-1, used only on
      a position's first held day, per the next-open execution rule). ~137 MB each.
    - A per-ticker SPARSE dict of (position, ADV20) pairs for cost-bucket lookups at
      trade time only (not a dense matrix -- costs are only ever priced at specific
      entry/exit days, a much smaller set than every day).
    - Benchmarks (SPY/IEF/VFITX/BIL) and the FRED cash rate are small, separate dense
      Series (few thousand points), built and simulated independently of the M-wide
      matrices.

Design decisions taken to implement the frozen file literally where it is silent
(recorded verbatim in ``prereg_issues`` of the output report -- see this session's
final message for the full list, not repeated in each report):
    1. No cash/risk-free subtraction before computing a Sharpe or a Sharpe gap --
       every series (candidate, PRIMARY, BM1-3) already carries its own cash drag
       internally on any day it is not fully invested; subtracting rf again would
       double-count it. Raw net daily returns are used directly everywhere.
    2. An early exit (delisting or a segment break) parks that slot's capital in
       CASH (earning the BIL/DTB3 rate) until its cohort's next scheduled
       reformation, rather than redistributing it pro-rata to the other names still
       held in that cohort.
    3. BM1/BM2/BM3 and the two PRIMARY legs are costed at protocol Sec.2's general
       ETF rate (3 bps/side; all of SPY/IEF/BIL/VFITX clear the $50M ADV bar) --
       S4's own frozen COSTS block only restates the STOCK ADV-bucket table.
    4. "Per side" is charged in full at both entry and exit (not split/halved across
       the round trip), matching the frozen COSTS key name literally.
    5. Decile size = round(N * 0.10): 50 for N=500, 100 for N=1000.
    6. Every scheduled reformation (PRIMARY's monthly rebalance, the 1-month hold,
       and each 6-month cohort's formation/unwind) is modelled as a FRESH slot for
       every held name at that reformation, i.e. no netting of an unchanged
       continuing member's weight across consecutive reformations. This is the
       conservative (higher-cost, not lower) reading of "rebalanced monthly" /
       "reform every month-end" -- and it is applied symmetrically to the candidate
       AND its PRIMARY benchmark, so the Sharpe GAP between them is not biased by
       this choice, even though each series' own absolute Sharpe is depressed by it.
    7. A3 (placebo) compares the candidate's own RAW daily Sharpe against the
       distribution of placebo RAW daily Sharpes -- the protocol's literal wording
       ("Sharpe above the 95th percentile of the placebo Sharpes") never frames A3 as
       a benchmark-relative gap, unlike A1/A4/A5.

A cleaning-rule GAP was found while sanity-checking the top-10 P&L contributors (not
a prereg ambiguity -- a real defect in the SHARED, frozen scripts/eodhd_clean.py used
by all five shortlist candidates, reported to the coordinator, not patched here):
``clean_bars``'s segment-break detector only fires on a discrete one-day JUMP in
``adjusted_close``. A different corruption pattern slips through it untouched: EODHD's
own ``adjusted_close`` pinned at a constant sentinel ceiling (999999.9999, an apparent
overflow in their split-adjustment-factor computation) for extended stretches -- a
CONSTANT value produces no day-over-day jump, so no segment break is ever recorded,
even though the series is corrupted throughout. Found in 182 of 16,229 eligible-ever
tickers (~1.1%), concentrated in heavily reverse-split micro-caps/biotechs. Because
``ratio_52wk`` is a within-segment RATIO, a constant multiplicative mis-adjustment
cancels out of it exactly -- but ``adv63``/``adv20`` dollar-volume (adjusted_close x
volume) is NOT scale-invariant, so an inflated ``adjusted_close`` can make a genuinely
illiquid name look liquid enough to enter the top-N cut. All 4 non-mega-cap names in
S4_p1's top-10 P&L contributors (SNCA, TMBR, BXRX, RXII, plus YRCW) carry this exact
signature; the 5 mega-cap winners (AAPL, QCOM, CHKP, WDC, AMZN) do not. This was
DISCOVERED, not fixed, here -- fixing scripts/eodhd_clean.py would retroactively
change every OTHER shortlist candidate's already-computed results too, which is out
of this candidate's scope; see the report's "cleaning_rule_gap_found" key and this
session's final message.

A separate, genuine HARNESS bug (not a cleaning-rule gap) was caught by the same
sanity check and IS fixed here: a held position whose ticker hit a segment break
MID-HOLD (as opposed to the end of its whole file) was not being forced to exit at
the segment boundary, in violation of the frozen file's own "no return may be
computed across a segment boundary" rule -- ``schedule_slots`` now clips every slot's
exit to the end of its OWN entry segment (``seg_starts_of``/``segment_last_pos``),
treating a segment break mid-hold exactly like a delisting.

    python scripts/run_eodhd_s4_evaluation.py --out <dir> \
        [--report docs/eodhd_s4_evaluation_2026_10.json] [--append-ledger] [--limit-placebo N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eodhd_clean as ec
import eodhd_s4_52wk_high_preregistered_bars as prereg
from eodhd_s4_build_panel import entry_dates_after, month_end_dates
from run_alt_premia_evaluation import stationary_indices

from firm.eval.overfitting import cscv_pbo, deflated_sharpe

log = logging.getLogger(__name__)

EODHD = _ROOT / "data" / "research" / "eodhd"
FRED = _ROOT / "data" / "research" / "fred"
LEDGER = _ROOT / "docs" / "S4_trial_history.json"

ANN = math.sqrt(252)
WINDOW_START = pd.Timestamp(prereg.WINDOW["start"])
WINDOW_END = pd.Timestamp(prereg.WINDOW["end"])
MIDPOINT = pd.Timestamp(prereg.WINDOW["midpoint"])
N_VALUES = sorted({c["N"] for c in prereg.CANDIDATES.values()})          # [500, 1000]
HOLD_MONTHS = {"6_month_overlapping": 6, "1_month": 1}
DECILE_FRAC = 0.10
STOCK_COST = prereg.COSTS["stock_round_trip_bps_by_adv20_bucket_PER_SIDE"]
ETF_COST_BPS = 3.0            # protocol Sec.2 general ETF rate (see design note 3)
STRESS_MULT = prereg.COSTS["stress_multiplier"]
DELIST_STRESS_PCT = prereg.DELISTING["stress_variant_pct"]
ALPHA = prereg.BOOTSTRAP["alpha_one_sided"]
ADV20_WINDOW = 20


# ---------------------------------------------------------------------------
# Harness run log -- for external review: every real-data run this session, not
# just the governing (last) one. Runs 1 and 2 predate this feature (they ran under
# an earlier commit of this file), so their facts are recorded here as data, taken
# from their saved logs (full_run.log, full_run2.log) and from the report.json each
# one produced before being superseded. Local log timestamps are UTC+3; utc_start
# below is already converted (local - 3h).
# ---------------------------------------------------------------------------
HARNESS_LOG_PRIOR_RUNS = [
    {
        "run": 1, "harness_commit": "6e6320a", "log_file": "full_run.log",
        "utc_start": "2026-09-30T19:48:15Z",
        "fix_before_this_run": "none -- first real-data run of this harness",
        "fix_after_this_run": (
            "schedule_slots' default weight was 1/decile_size regardless of hold "
            "length, missing the 1/hold_months book-fraction divisor: each 6-month "
            "cohort's names got a full 1/decile_size of the book instead of "
            "(1/6)/decile_size, so ~6 concurrently open cohorts summed to ~5.76x "
            "invested weight and produced a fabricated -68% single-day loss on "
            "2020-03-16 (real SPY that day: -12%). Fixed at the single default-weight "
            "formula in schedule_slots (commit f0fa1cb) -- this IMPLEMENTS "
            "HOLD_STRUCTURES['6_month_overlapping']'s own frozen text ('a new cohort "
            "... 1/6 of the book each month'), it does not change what the frozen "
            "file says."
        ),
        "results": {
            "S4_p1_6mo_N500": {"sharpe": 0.529, "tier": "C", "bars": {
                "A1": True, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p2_6mo_N1000": {"sharpe": 0.553, "tier": "C", "bars": {
                "A1": True, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p3_1mo_N500": {"sharpe": 0.311, "tier": "D", "bars": {
                "A1": False, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p4_1mo_N1000": {"sharpe": 0.358, "tier": "C", "bars": {
                "A1": False, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
        },
    },
    {
        "run": 2, "harness_commit": "f0fa1cb", "log_file": "full_run2.log",
        "utc_start": "2026-09-30T22:02:10Z",
        "fix_before_this_run": "run 1's 1/hold_months weight-divisor fix (see run 1 entry)",
        "fix_after_this_run": (
            "a held position whose ticker hit a SEGMENT BREAK mid-hold (e.g. a "
            "phantom reverse-split adjustment-factor reset) was clipped only to "
            "last_pos_of (the ticker's LAST bar in its WHOLE file), never to the end "
            "of the segment it was actually entered in -- found while sanity-checking "
            "this run's top-10 P&L contributors (all 4 non-mega-cap names were later "
            "found to carry the cleaning-rule sentinel-gap signature, see "
            "cleaning_rule_gap_found below). Fixed via seg_starts_of/segment_last_pos "
            "(commit ba52b36) -- this IMPLEMENTS the frozen file's own existing rule "
            "('no return may be computed across a segment boundary: a holding that "
            "spans one is closed at the last close before it'), it does not change "
            "what the frozen file says."
        ),
        "results": {
            "S4_p1_6mo_N500": {"sharpe": 0.535, "tier": "C", "bars": {
                "A1": True, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p2_6mo_N1000": {"sharpe": 0.559, "tier": "C", "bars": {
                "A1": True, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p3_1mo_N500": {"sharpe": 0.311, "tier": "D", "bars": {
                "A1": False, "A2": False, "A3": False, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
            "S4_p4_1mo_N1000": {"sharpe": 0.358, "tier": "C", "bars": {
                "A1": False, "A2": False, "A3": True, "A4": False, "A5": False, "A6": False, "A7": "PENDING"}},
        },
    },
]


def build_harness_log(current_results: dict, current_commit: str, current_utc_start: str,
                      fix_before_this_run: str) -> list[dict]:
    """HARNESS_LOG_PRIOR_RUNS plus the current run, in the same shape, for the
    report's "harness_log" key -- one entry per real-data run this session."""
    current = {
        "run": len(HARNESS_LOG_PRIOR_RUNS) + 1, "harness_commit": current_commit,
        "log_file": None, "utc_start": current_utc_start,
        "fix_before_this_run": fix_before_this_run,
        "fix_after_this_run": "none yet -- this is the governing (most recent) run",
        "results": {cand: {"sharpe": res["sharpe"], "tier": res["tier"], "bars": res["bars"]}
                   for cand, res in current_results.items()},
    }
    return HARNESS_LOG_PRIOR_RUNS + [current]


def adv_bucket_bps(adv: float) -> float:
    if not np.isfinite(adv):
        return STOCK_COST["adv_gt_20m"]     # should not occur: eligibility requires a trading history
    if adv < 1e6:
        return STOCK_COST["adv_lt_1m"]
    if adv < 5e6:
        return STOCK_COST["adv_1m_5m"]
    if adv < 20e6:
        return STOCK_COST["adv_5m_20m"]
    return STOCK_COST["adv_gt_20m"]


# ---------------------------------------------------------------------------
# Calendar / cash
# ---------------------------------------------------------------------------

def eval_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading days from the first session after WINDOW start through WINDOW end."""
    first_entry = calendar[calendar > WINDOW_START][0]
    return calendar[(calendar >= first_entry) & (calendar <= WINDOW_END)]


def cash_rate_series(dates: pd.DatetimeIndex) -> np.ndarray:
    """BIL total return from its first bar (2007-05-30); FRED DTB3/100/252 before that."""
    bil = pd.read_parquet(EODHD / "etfs_full" / "BIL.parquet", columns=["date", "adjusted_close"])
    bil["date"] = pd.to_datetime(bil["date"])
    bil = bil.sort_values("date").drop_duplicates("date").set_index("date")["adjusted_close"]
    bil_ret = bil.pct_change()
    tb = pd.read_parquet(FRED / "DTB3.parquet")
    tb["date"] = pd.to_datetime(tb["date"])
    tb = tb.sort_values("date").drop_duplicates("date").set_index("date")["value"]
    tb_ret = (tb / 100.0 / 252.0)
    out = pd.Series(np.nan, index=dates)
    bil_start = bil.index.min()
    out.loc[out.index >= bil_start] = bil_ret.reindex(out.index[out.index >= bil_start]).to_numpy()
    before = out.index < bil_start
    out.loc[before] = tb_ret.reindex(dates).ffill().loc[before].to_numpy()
    return out.fillna(0.0).to_numpy(dtype="float32")


# ---------------------------------------------------------------------------
# Per-ticker daily arrays (dense RET/ENTRY_RET matrices + sparse ADV20)
# ---------------------------------------------------------------------------

def build_matrices(tickers: list[str], calendar: pd.DatetimeIndex, dates: pd.DatetimeIndex,
                   universe_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, dict, dict, dict]:
    """Returns (RET, ENTRY_RET) dense (T, M) float32 arrays, ``col_of[ticker]``,
    ``adv20_of[ticker] -> (pos, adv20)`` sparse arrays, ``last_pos_of[ticker]`` (last
    valid bar of the WHOLE file) and ``seg_starts_of[ticker]`` (sorted positions where
    a NEW segment starts, excluding the first bar -- used to force a held position to
    exit at the end of ITS OWN segment, per protocol: "no return may be computed
    across a segment boundary")."""
    T, M = len(dates), len(tickers)
    RET = np.full((T, M), np.nan, dtype="float32")
    ENTRY_RET = np.full((T, M), np.nan, dtype="float32")
    col_of = {t: i for i, t in enumerate(tickers)}
    adv20_of: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    last_pos_of: dict[str, int] = {}
    seg_starts_of: dict[str, np.ndarray] = {}
    date_arr = dates.to_numpy()
    t0 = time.time()
    for i, tkr in enumerate(tickers):
        f = universe_dir / f"{tkr}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
        if df.empty:
            continue
        for c in ("open", "close", "adjusted_close"):
            df[c] = df[c].astype("float32")
        df["volume"] = df["volume"].astype("float32")
        cleaned, _ = ec.clean_bars(df, asset="equity", calendar=calendar)
        if cleaned.empty:
            continue
        seg = cleaned["segment"].to_numpy()
        adj = cleaned["adjusted_close"].to_numpy(dtype="float64")
        close = cleaned["close"].to_numpy(dtype="float64")
        openp = cleaned["open"].to_numpy(dtype="float64")
        vol = cleaned["volume"].to_numpy(dtype="float64")
        d = cleaned["date"].to_numpy()
        n = len(cleaned)
        same_seg = np.r_[False, seg[1:] == seg[:-1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_ret = np.r_[np.nan, adj[1:] / adj[:-1] - 1.0]
        ret_c2c = np.where(same_seg, raw_ret, np.nan).astype("float32")
        with np.errstate(invalid="ignore", divide="ignore"):
            entry_ret = np.where(openp > 0, close / openp - 1.0, np.nan).astype("float32")
        dollar_vol = adj * vol
        adv20 = (pd.Series(dollar_vol).groupby(pd.Series(seg))
                 .transform(lambda s: s.rolling(ADV20_WINDOW, min_periods=ADV20_WINDOW).median())
                 .to_numpy(dtype="float32"))
        pos = np.searchsorted(date_arr, d)
        valid = (pos < T) & (date_arr[np.clip(pos, 0, T - 1)] == d)
        if not valid.any():
            continue
        p = pos[valid]
        col = col_of[tkr]
        RET[p, col] = ret_c2c[valid]
        ENTRY_RET[p, col] = entry_ret[valid]
        av = adv20[valid]
        av_ok = np.isfinite(av)
        adv20_of[tkr] = (p[av_ok].astype("int32"), av[av_ok])
        last_pos_of[tkr] = int(p[-1])
        seg_v = seg[valid]
        breaks = p[np.r_[False, seg_v[1:] != seg_v[:-1]]]
        if len(breaks):
            seg_starts_of[tkr] = breaks.astype("int32")
        if (i + 1) % 1000 == 0:
            log.info("  matrices: %d/%d tickers (%.1fs)", i + 1, M, time.time() - t0)
    log.info("matrices built: T=%d M=%d in %.1fs", T, M, time.time() - t0)
    return RET, ENTRY_RET, col_of, adv20_of, last_pos_of, seg_starts_of


def adv20_at_or_before(adv20_of: dict, ticker: str, pos: int) -> float:
    arr_pos, arr_val = adv20_of.get(ticker, (np.empty(0, dtype="int32"), np.empty(0, dtype="float32")))
    if len(arr_pos) == 0:
        return float("nan")
    idx = np.searchsorted(arr_pos, pos, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(arr_val[idx])


def segment_last_pos(seg_starts_of: dict, last_pos_of: dict, ticker: str, pos: int) -> int:
    """Last position sharing ``pos``'s own segment (clipped to the ticker's last bar if
    that segment runs to the end of its data, per the no-return-across-a-segment rule)."""
    starts = seg_starts_of.get(ticker)
    last = last_pos_of.get(ticker, pos)
    if starts is None or len(starts) == 0:
        return last
    i = np.searchsorted(starts, pos, side="right")
    return int(starts[i]) - 1 if i < len(starts) else last


# ---------------------------------------------------------------------------
# Month-end universe / decile construction from panel.parquet
# ---------------------------------------------------------------------------

def load_panel_window(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    return panel[(panel["month_end"] >= WINDOW_START) & (panel["month_end"] <= WINDOW_END)].copy()


def month_universe_and_decile(panel_window: pd.DataFrame) -> tuple[list[pd.Timestamp], dict]:
    """month_ends (sorted) and, per month_end, per N: {'universe': [tickers], 'decile': [tickers]}."""
    month_ends = sorted(panel_window["month_end"].unique())
    out = {}
    for me in month_ends:
        g = panel_window[panel_window["month_end"] == me].dropna(subset=["adv63"])
        g = g.sort_values("adv63", ascending=False)
        per_n = {}
        for N in N_VALUES:
            topn = g.head(N)
            dn = max(1, round(N * DECILE_FRAC))
            decile = topn.sort_values("ratio_52wk", ascending=False).head(dn)
            per_n[N] = {"universe": list(topn["ticker"]), "decile": list(decile["ticker"])}
        out[pd.Timestamp(me)] = per_n
    return [pd.Timestamp(m) for m in month_ends], out


# ---------------------------------------------------------------------------
# Slot scheduling + simulation
# ---------------------------------------------------------------------------

class Slot:
    __slots__ = ("col", "early_exit", "entry_pos", "exit_pos", "weight")

    def __init__(self, col, weight, entry_pos, exit_pos, early_exit):
        self.col, self.weight = col, weight
        self.entry_pos, self.exit_pos, self.early_exit = entry_pos, exit_pos, early_exit


def schedule_slots(month_ends: list[pd.Timestamp], entry_of: dict, per_month_names: dict,
                   hold_months: int, col_of: dict, last_pos_of: dict, T: int,
                   weight_fn=None, seg_starts_of: dict | None = None) -> list[Slot]:
    """One FRESH slot per (name, formation month) -- see design note 6: no netting of
    unchanged continuing members across consecutive reformations.

    Default weighting: each month-end's cohort is ``1/hold_months`` of the whole book
    (the overlapping-cohort construction -- HOLD_STRUCTURES["6_month_overlapping"]:
    "1/6 of the book each month"), split equally across that cohort's names. For
    ``hold_months == 1`` (PRIMARY, the 1-month hold) this is just ``1/len(names)``,
    since there is only ever one cohort open at a time.

    ``seg_starts_of``, when given, additionally clips every slot's exit to the end of
    the SAME segment it was entered in (a segment break mid-hold -- e.g. a phantom
    reverse-split adjustment-factor reset -- forces an early exit there, exactly like
    a delisting, per the frozen file's "no return may be computed across a segment
    boundary"). Every real call site (build_variant, build_primary, placebo_sharpes,
    two_day_lag_variant) passes it; it is optional only for synthetic tests that don't
    model segments at all."""
    slots: list[Slot] = []
    K = len(month_ends)
    for k, me in enumerate(month_ends):
        e_pos = entry_of.get(me)
        if e_pos is None or not np.isfinite(e_pos):
            continue
        e_pos = int(e_pos)
        names = per_month_names.get(me, [])
        if not names:
            continue
        w = weight_fn(me, names) if weight_fn else {n: (1.0 / hold_months) / len(names) for n in names}
        k_exit = k + hold_months
        if k_exit < K:
            exit_me = month_ends[k_exit]
            exit_entry = entry_of.get(exit_me)
            scheduled_x = (int(exit_entry) - 1) if (exit_entry is not None and np.isfinite(exit_entry)) else (T - 1)
        else:
            scheduled_x = T - 1
        for name, wt in w.items():
            col = col_of.get(name)
            if col is None or wt <= 0:
                continue
            last = last_pos_of.get(name, -1)
            if last < e_pos:
                continue    # no data at all on/after entry: cannot open this slot
            seg_end = segment_last_pos(seg_starts_of, last_pos_of, name, e_pos) if seg_starts_of is not None else last
            x = min(scheduled_x, last, seg_end)
            early = x < scheduled_x
            slots.append(Slot(col, wt, e_pos, x, early))
    return slots


def simulate_slots(slots: list[Slot], RET: np.ndarray, ENTRY_RET: np.ndarray, cash_ret: np.ndarray,
                   T: int, cost_mult: float = 1.0, delist_stress: bool = False,
                   cost_bps_of=None) -> tuple[np.ndarray, np.ndarray]:
    """Net daily return of the whole book (num/invested_weight), given fixed slots.
    ``cost_bps_of(col, pos)`` returns the per-side bps to charge for that trade; if
    None, the stock ADV20-bucket table is used (see cost_bps_stock_lookup)."""
    num = np.zeros(T, dtype="float64")
    invested = np.zeros(T, dtype="float64")
    for s in slots:
        w, col, e, x = s.weight, s.col, s.entry_pos, s.exit_pos
        entry_bps = cost_bps_of(col, e) * cost_mult
        exit_bps = cost_bps_of(col, x) * cost_mult
        er = ENTRY_RET[e, col]
        if not np.isfinite(er):
            er = 0.0
        num[e] += w * er - w * entry_bps / 1e4
        if x > e:
            seg = RET[e + 1:x + 1, col]
            seg = np.nan_to_num(seg, nan=0.0)
            num[e + 1:x + 1] += w * seg
        num[x] -= w * exit_bps / 1e4
        if delist_stress and s.early_exit:
            num[x] += w * DELIST_STRESS_PCT
        invested[e:x + 1] += w
    cash_w = np.clip(1.0 - invested, 0.0, None)
    net = num + cash_w * cash_ret
    return net.astype("float64"), invested


def cost_bps_stock_lookup(adv20_of: dict, ticker_of_col: list[str]):
    def f(col: int, pos: int) -> float:
        t = ticker_of_col[col]
        return adv_bucket_bps(adv20_at_or_before(adv20_of, t, pos))
    return f


def flat_cost_bps(bps: float):
    return lambda col, pos: bps


# ---------------------------------------------------------------------------
# Benchmarks (BM1/BM2/BM3) -- small dense series, not part of the M-wide matrices
# ---------------------------------------------------------------------------

def load_bench_daily(name: str, asset: str, dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> dict:
    f = EODHD / "etfs_full" / f"{name}.parquet"
    df = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "adjusted_close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cleaned, _ = ec.clean_bars(df, asset=asset, calendar=calendar)
    seg = cleaned["segment"].to_numpy()
    adj = cleaned["adjusted_close"].to_numpy(dtype="float64")
    close = cleaned["close"].to_numpy(dtype="float64")
    openp = cleaned["open"].to_numpy(dtype="float64")
    d = cleaned["date"].to_numpy()
    same_seg = np.r_[False, seg[1:] == seg[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_ret = np.r_[np.nan, adj[1:] / adj[:-1] - 1.0]
    ret_c2c = np.where(same_seg, raw_ret, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        entry_ret = np.where(openp > 0, close / openp - 1.0, np.nan)
    date_arr = dates.to_numpy()
    T = len(dates)
    pos = np.searchsorted(date_arr, d)
    valid = (pos < T) & (date_arr[np.clip(pos, 0, T - 1)] == d)
    RET = np.full(T, np.nan)
    ENTRY = np.full(T, np.nan)
    RET[pos[valid]] = ret_c2c[valid]
    ENTRY[pos[valid]] = entry_ret[valid]
    last_pos = int(pos[valid][-1]) if valid.any() else -1
    return {"RET": RET, "ENTRY_RET": ENTRY, "last_pos": last_pos}


def bm1_spy(bench: dict, cash_ret: np.ndarray, T: int, cost_mult: float = 1.0) -> np.ndarray:
    slot = Slot(0, 1.0, 0, T - 1, False)
    RET = bench["SPY"]["RET"].reshape(T, 1)
    ENTRY = bench["SPY"]["ENTRY_RET"].reshape(T, 1)
    net, _ = simulate_slots([slot], RET, ENTRY, cash_ret, T, cost_mult, cost_bps_of=flat_cost_bps(ETF_COST_BPS))
    return net


def bm2_60_40(bench: dict, dates: pd.DatetimeIndex, cash_ret: np.ndarray, T: int, cost_mult: float = 1.0) -> np.ndarray:
    """60% SPY / 40% bond leg, fresh monthly slots (IEF from its first bar, VFITX before)."""
    ief_first = int(np.argmax(np.isfinite(bench["IEF"]["RET"])))
    ief_first_date = dates[ief_first] if np.isfinite(bench["IEF"]["RET"][ief_first]) else None
    month_ends = month_end_dates(dates)
    entry = entry_dates_after(month_ends, dates)
    T2 = T
    RET = np.column_stack([bench["SPY"]["RET"], bench["IEF"]["RET"], bench["VFITX"]["RET"]])
    ENTRY = np.column_stack([bench["SPY"]["ENTRY_RET"], bench["IEF"]["ENTRY_RET"], bench["VFITX"]["ENTRY_RET"]])
    last_pos = {0: bench["SPY"]["last_pos"], 1: bench["IEF"]["last_pos"], 2: bench["VFITX"]["last_pos"]}
    slots: list[Slot] = []
    K = len(month_ends)
    me_list = list(month_ends)
    for k, me in enumerate(me_list):
        e_pos = entry.get(me)
        if e_pos is None or pd.isna(e_pos):
            continue
        e_idx = int(np.searchsorted(dates.to_numpy(), np.datetime64(e_pos)))
        if e_idx >= T2 or dates[e_idx] != e_pos:
            continue
        use_ief = ief_first_date is not None and me >= ief_first_date
        bond_col = 1 if use_ief else 2
        k_exit = k + 1
        if k_exit < K:
            exit_me = entry.get(me_list[k_exit])
            if exit_me is not None and not pd.isna(exit_me):
                ex_idx = int(np.searchsorted(dates.to_numpy(), np.datetime64(exit_me))) - 1
            else:
                ex_idx = T2 - 1
        else:
            ex_idx = T2 - 1
        for col, wt in ((0, 0.6), (bond_col, 0.4)):
            x = min(ex_idx, last_pos[col])
            early = last_pos[col] < ex_idx
            if x < e_idx:
                continue
            slots.append(Slot(col, wt, e_idx, x, early))
    net, _ = simulate_slots(slots, RET, ENTRY, cash_ret, T2, cost_mult, cost_bps_of=flat_cost_bps(ETF_COST_BPS))
    return net


def bm3_vol_target(bench: dict, cash_ret: np.ndarray, T: int, cost_mult: float = 1.0) -> np.ndarray:
    """SPY weight = min(1, 0.12 / 21d vol), traded when drift exceeds 0.10 (plain close-to-close)."""
    spec = prereg.BENCHMARKS["BM3_SPY_VT"]
    r = np.nan_to_num(bench["SPY"]["RET"], nan=0.0)
    sig = pd.Series(r).rolling(spec["vol_window"]).std().to_numpy() * ANN
    c = ETF_COST_BPS / 1e4 * cost_mult
    w = 0.0
    out = np.zeros(T)
    for i in range(T):
        rp = w * r[i] + (1 - w) * cash_ret[i]
        if 1 + rp > 0:
            w = w * (1 + r[i]) / (1 + rp)
        cost = 0.0
        if i >= 1 and np.isfinite(sig[i - 1]) and sig[i - 1] > 0:
            tgt = min(1.0, spec["target_vol"] / sig[i - 1])
            if abs(tgt - w) > spec["band_abs"]:
                cost = abs(tgt - w) * c
                w = tgt
        out[i] = rp - cost
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANN) if sd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def cagr(r: np.ndarray) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    return float(np.prod(1 + r) ** (252 / n) - 1)


def paired_sharpe_gap_boot(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int, mean_block: int) -> np.ndarray:
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


def _seed_for(*parts: str) -> int:
    import hashlib
    return prereg.SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) % 10_000_000


def turnover_of(slots: list[Slot], T: int) -> float:
    """Sum of |weight| traded (entry + exit) per year, averaged."""
    total = sum(2 * s.weight for s in slots)   # one buy + one sell per slot
    years = T / 252
    return total / years if years > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_all(panel_path: Path, universe_dir: Path = EODHD / "us_universe_full",
             limit_placebo: int | None = None) -> dict:
    assert not prereg.DRAFT, "pre-registration is still DRAFT; freeze it first"
    fp = prereg.bars_fingerprint()
    log.info("fingerprint %s", fp)

    calendar = ec.equity_calendar()
    dates = eval_dates(calendar)
    T = len(dates)
    log.info("EVAL_DATES: %d days, %s -> %s", T, dates[0].date(), dates[-1].date())
    cash_ret = cash_rate_series(dates)

    panel_window = load_panel_window(panel_path)
    month_ends, per_month = month_universe_and_decile(panel_window)
    entry = entry_dates_after(pd.DatetimeIndex(month_ends), dates)
    date_arr = dates.to_numpy()

    def entry_pos_of(me):
        ed = entry.get(me)
        if ed is None or pd.isna(ed):
            return None
        idx = np.searchsorted(date_arr, np.datetime64(ed))
        return idx if idx < T and dates[idx] == ed else None

    entry_of = {me: entry_pos_of(me) for me in month_ends}

    tickers = sorted({t for me in month_ends for t in per_month[me][max(N_VALUES)]["universe"]})
    log.info("building matrices for %d tickers", len(tickers))
    RET, ENTRY_RET, col_of, adv20_of, last_pos_of, seg_starts_of = build_matrices(tickers, calendar, dates, universe_dir)
    ticker_of_col = tickers
    cost_fn = cost_bps_stock_lookup(adv20_of, ticker_of_col)

    bench = {name: load_bench_daily(name, asset, dates, calendar)
             for name, asset in (("SPY", "equity"), ("IEF", "equity"), ("VFITX", "nav"))}

    def build_variant(hold_key: str, N: int, cost_mult=1.0, delist_stress=False) -> tuple[np.ndarray, list[Slot]]:
        names_by_month = {me: per_month[me][N]["decile"] for me in month_ends}
        slots = schedule_slots(month_ends, entry_of, names_by_month, HOLD_MONTHS[hold_key], col_of, last_pos_of, T,
                               seg_starts_of=seg_starts_of)
        net, _ = simulate_slots(slots, RET, ENTRY_RET, cash_ret, T, cost_mult, delist_stress, cost_fn)
        return net, slots

    def build_primary(N: int, cost_mult=1.0) -> tuple[np.ndarray, list[Slot]]:
        names_by_month = {me: per_month[me][N]["universe"] for me in month_ends}
        slots = schedule_slots(month_ends, entry_of, names_by_month, 1, col_of, last_pos_of, T,
                               seg_starts_of=seg_starts_of)
        net, _ = simulate_slots(slots, RET, ENTRY_RET, cash_ret, T, cost_mult, False, cost_fn)
        return net, slots

    base: dict[str, np.ndarray] = {}
    slots_of: dict[str, list[Slot]] = {}
    stress: dict[str, np.ndarray] = {}
    delist: dict[str, np.ndarray] = {}
    for cand, spec in prereg.CANDIDATES.items():
        net, slots = build_variant(spec["hold"], spec["N"])
        base[cand] = net
        slots_of[cand] = slots
        s_net, _ = build_variant(spec["hold"], spec["N"], cost_mult=STRESS_MULT)
        stress[cand] = s_net
        d_net, _ = build_variant(spec["hold"], spec["N"], delist_stress=True)
        delist[cand] = d_net
        log.info("%s built: %d slots, sharpe %.3f", cand, len(slots), sharpe(net))

    primary: dict[int, np.ndarray] = {}
    primary_stress: dict[int, np.ndarray] = {}
    for N in N_VALUES:
        net, _ = build_primary(N)
        primary[N] = net
        s_net, _ = build_primary(N, cost_mult=STRESS_MULT)
        primary_stress[N] = s_net
        log.info("PRIMARY_N%d built: sharpe %.3f", N, sharpe(net))

    base["BM1_SPY"] = bm1_spy(bench, cash_ret, T)
    base["BM2_60_40"] = bm2_60_40(bench, dates, cash_ret, T)
    base["BM3_SPY_VT"] = bm3_vol_target(bench, cash_ret, T)
    stress["BM1_SPY"] = bm1_spy(bench, cash_ret, T, STRESS_MULT)
    stress["BM2_60_40"] = bm2_60_40(bench, dates, cash_ret, T, STRESS_MULT)
    stress["BM3_SPY_VT"] = bm3_vol_target(bench, cash_ret, T, STRESS_MULT)

    return {
        "fingerprint": fp, "dates": dates, "T": T, "cash_ret": cash_ret,
        "base": base, "stress": stress, "delist": delist, "primary": primary,
        "primary_stress": primary_stress, "slots_of": slots_of,
        "month_ends": month_ends, "per_month": per_month, "entry_of": entry_of,
        "RET": RET, "ENTRY_RET": ENTRY_RET, "col_of": col_of, "adv20_of": adv20_of,
        "last_pos_of": last_pos_of, "seg_starts_of": seg_starts_of, "cost_fn": cost_fn, "bench": bench,
    }


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------

def placebo_sharpes(built: dict, cand: str, n_draws: int, seed: int) -> np.ndarray:
    spec = prereg.CANDIDATES[cand]
    month_ends, per_month = built["month_ends"], built["per_month"]
    entry_of, col_of, last_pos_of = built["entry_of"], built["col_of"], built["last_pos_of"]
    seg_starts_of = built["seg_starts_of"]
    RET, ENTRY_RET, cash_ret, T = built["RET"], built["ENTRY_RET"], built["cash_ret"], built["T"]
    cost_fn = built["cost_fn"]
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for draw in range(n_draws):
        names_by_month = {}
        for me in month_ends:
            pool = per_month[me][spec["N"]]["universe"]
            dn = len(per_month[me][spec["N"]]["decile"])
            if len(pool) <= dn:
                names_by_month[me] = list(pool)
            else:
                idx = rng.choice(len(pool), size=dn, replace=False)
                names_by_month[me] = [pool[i] for i in idx]
        slots = schedule_slots(month_ends, entry_of, names_by_month, HOLD_MONTHS[spec["hold"]], col_of, last_pos_of, T,
                               seg_starts_of=seg_starts_of)
        net, _ = simulate_slots(slots, RET, ENTRY_RET, cash_ret, T, cost_bps_of=cost_fn)
        out[draw] = sharpe(net)
        if draw % 100 == 0:
            log.info("  placebo %s draw %d/%d", cand, draw, n_draws)
    return out


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def detect_adjusted_close_sentinel_gap(tickers: list[str], universe_dir: Path,
                                       sentinel: float = 999990.0) -> list[str]:
    """Tickers whose file shows an adjusted_close >= ``sentinel`` at some point -- the
    signature of the cleaning-rule gap documented in this module's docstring (a
    constant-valued vendor overflow that never triggers a segment break, since a
    discrete jump is what clean_bars' spike/segment detector looks for). Uses parquet
    row-group statistics only (no full read) -- a cheap, separate diagnostic, not part
    of the simulation itself."""
    import pyarrow.parquet as pq
    hits = []
    for t in tickers:
        f = universe_dir / f"{t}.parquet"
        if not f.exists():
            continue
        try:
            stats = pq.ParquetFile(f).metadata.row_group(0).column(5).statistics
        except Exception:
            continue
        if stats is not None and stats.max is not None and stats.max >= sentinel:
            hits.append(t)
    return hits


def sanity_checks(built: dict) -> dict:
    dates, base = built["dates"], built["base"]
    out = {}
    for name in ("BM1_SPY", "BM2_60_40"):
        r = base[name]
        out[name] = {"sharpe": sharpe(r), "cagr": cagr(r), "vol": float(r.std(ddof=1) * ANN),
                    "max_dd": max_drawdown(r)}
    big_moves = {}
    for name, r in base.items():
        idx = np.flatnonzero(np.abs(r) > 0.15)
        big_moves[name] = [{"date": str(dates[i].date()), "ret": float(r[i])} for i in idx[:20]]
    out["big_moves_gt_15pct"] = big_moves
    out["nan_days"] = {name: int(np.isnan(r).sum()) for name, r in base.items()}
    turnover = {}
    for cand in prereg.CANDIDATES:
        turnover[cand] = turnover_of(built["slots_of"][cand], built["T"])
    out["turnover_per_year"] = turnover
    return out


def top_contributors(built: dict, cand: str, k: int = 10) -> list[dict]:
    slots = built["slots_of"][cand]
    RET, ENTRY_RET = built["RET"], built["ENTRY_RET"]
    ticker_of_col = {v: k_ for k_, v in built["col_of"].items()}
    pnl: dict[str, float] = {}
    for s in slots:
        er = ENTRY_RET[s.entry_pos, s.col]
        er = er if np.isfinite(er) else 0.0
        seg = np.nan_to_num(RET[s.entry_pos + 1:s.exit_pos + 1, s.col], nan=0.0)
        contrib = s.weight * (er + seg.sum())
        t = ticker_of_col[s.col]
        pnl[t] = pnl.get(t, 0.0) + contrib
    ranked = sorted(pnl.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [{"ticker": t, "pnl_contribution": v} for t, v in ranked]


# ---------------------------------------------------------------------------
# Bars / tiers
# ---------------------------------------------------------------------------

def gap_stats(c: np.ndarray, b: np.ndarray, dates: pd.DatetimeIndex, seed: int) -> dict:
    ok = np.isfinite(c) & np.isfinite(b)
    ce, be = c[ok], b[ok]
    sc, sb = sharpe(ce), sharpe(be)
    gap = sc - sb
    boot = paired_sharpe_gap_boot(ce, be, seed, prereg.BOOTSTRAP["n_boot"], prereg.BOOTSTRAP["mean_block_days"])
    lb, ub = float(np.quantile(boot, ALPHA)), float(np.quantile(boot, 1 - ALPHA))
    d = dates[ok]
    h1 = d <= MIDPOINT
    h2 = ~h1
    gap_h1 = sharpe(ce[h1]) - sharpe(be[h1]) if h1.any() else float("nan")
    gap_h2 = sharpe(ce[h2]) - sharpe(be[h2]) if h2.any() else float("nan")
    return {"sharpe_c": sc, "sharpe_b": sb, "gap": gap, "lb": lb, "ub": ub,
            "gap_half1": gap_h1, "gap_half2": gap_h2, "n_days": int(ok.sum()),
            "A1": bool(gap > 0 and lb > 0), "A4": bool(gap_h1 > 0 and gap_h2 > 0), "D": bool(ub < 0)}


def evaluate_candidate(built: dict, cand: str, placebo: np.ndarray, pbo: float, trial_sr: np.ndarray) -> dict:
    spec = prereg.CANDIDATES[cand]
    dates = built["dates"]
    c = built["base"][cand]
    c_stress = built["stress"][cand]
    c_delist = built["delist"][cand]
    prim = built["primary"][spec["N"]]
    prim_stress = built["primary_stress"][spec["N"]]
    bms = {"PRIMARY_liquid_universe_ewbh": (prim, prim_stress),
           "BM1_SPY": (built["base"]["BM1_SPY"], built["stress"]["BM1_SPY"]),
           "BM2_60_40": (built["base"]["BM2_60_40"], built["stress"]["BM2_60_40"]),
           "BM3_SPY_VT": (built["base"]["BM3_SPY_VT"], built["stress"]["BM3_SPY_VT"])}
    vs = {}
    for bm_name, (b, b_stress) in bms.items():
        gs = gap_stats(c, b, dates, _seed_for(cand, bm_name))
        ok = np.isfinite(c_stress) & np.isfinite(b_stress)
        gs["A5"] = bool(sharpe(c_stress[ok]) - sharpe(b_stress[ok]) > 0)
        vs[bm_name] = gs
    primary_gs = vs["PRIMARY_liquid_universe_ewbh"]
    ok_d = np.isfinite(c_delist)
    delist_mean_positive = bool(np.nanmean(c_delist[ok_d]) > np.nanmean(c[ok_d])) if ok_d.any() else None
    sc = sharpe(c[np.isfinite(c)])
    p95 = float(np.nanpercentile(placebo, prereg.PLACEBO["pass_percentile"]))
    dsr = float(deflated_sharpe(c[np.isfinite(c)], trial_sr, prior_trials=prereg.DSR["prior_trials"]))
    bars = {
        "A1": primary_gs["A1"], "A2": bool(dsr > 0.95), "A3": bool(sc > p95),
        "A4": all(vs[b]["A4"] for b in ("PRIMARY_liquid_universe_ewbh", "BM1_SPY", "BM2_60_40", "BM3_SPY_VT")),
        "A5": all(vs[b]["A5"] for b in ("PRIMARY_liquid_universe_ewbh", "BM1_SPY", "BM2_60_40", "BM3_SPY_VT")),
        "A6": bool(pbo < 0.50),
        "A7": "PENDING",
    }
    bars_for_classify = {k: v for k, v in bars.items() if k != "A7"}
    tier_d = any(vs[b]["D"] for b in ("PRIMARY_liquid_universe_ewbh", "BM1_SPY", "BM2_60_40", "BM3_SPY_VT"))
    tier_b = {"B_a": bool(primary_gs["gap"] > 0), "B_b": bool(bars["A3"] and bars["A4"] and bars["A5"])}
    tier = prereg.classify(bars_for_classify, tier_d, tier_b)
    tier_note = f"{tier} (conditional on A7)" if tier == "A" else tier
    return {
        "eval_start": str(dates[0].date()), "eval_end": str(dates[-1].date()),
        "sharpe": sc, "cagr": cagr(c[np.isfinite(c)]), "vol": float(np.nanstd(c, ddof=1) * ANN),
        "max_dd": max_drawdown(np.nan_to_num(c, nan=0.0)),
        "n_slots": len(built["slots_of"][cand]), "placebo_p95": p95,
        "placebo_median": float(np.nanmedian(placebo)), "dsr": dsr,
        "delist_stress_mean_gt_base_mean": delist_mean_positive,
        "vs": vs, "bars": bars, "tier_d_condition": tier_d, "tier_b": tier_b,
        "tier": tier, "tier_reported": tier_note,
    }


def per_year_table(r: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    s = pd.Series(r, index=dates)
    out = {}
    for y, g in s.groupby(s.index.year):
        g = g[np.isfinite(g)]
        out[int(y)] = {"sharpe": sharpe(g.to_numpy()), "n_days": len(g)}
    return out


def two_day_lag_variant(built: dict, cand: str) -> np.ndarray:
    """Re-run with every slot's entry/exit shifted 1 extra trading day later (2-day
    signal lag total), reusing the same membership/weights -- a reported robustness
    check, not a bar."""
    spec = prereg.CANDIDATES[cand]
    month_ends, per_month = built["month_ends"], built["per_month"]
    col_of, last_pos_of = built["col_of"], built["last_pos_of"]
    RET, ENTRY_RET, cash_ret, T = built["RET"], built["ENTRY_RET"], built["cash_ret"], built["T"]
    entry_of = {me: (built["entry_of"][me] + 1 if built["entry_of"][me] is not None
                     and built["entry_of"][me] + 1 < T else None) for me in month_ends}
    names_by_month = {me: per_month[me][spec["N"]]["decile"] for me in month_ends}
    slots = schedule_slots(month_ends, entry_of, names_by_month, HOLD_MONTHS[spec["hold"]], col_of, last_pos_of, T,
                           seg_starts_of=built["seg_starts_of"])
    net, _ = simulate_slots(slots, RET, ENTRY_RET, cash_ret, T, cost_bps_of=built["cost_fn"])
    return net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def current_git_commit() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def cmd_evaluate(args: argparse.Namespace) -> int:
    run_started_utc = datetime.now(UTC).isoformat()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    built = build_all(Path(args.panel), limit_placebo=args.limit_placebo)
    log.info("build_all done in %.1fs", time.time() - t0)

    n_draws = args.limit_placebo or prereg.PLACEBO["n_draws"]
    placebo: dict[str, np.ndarray] = {}
    for cand in prereg.CANDIDATES:
        t1 = time.time()
        placebo[cand] = placebo_sharpes(built, cand, n_draws, prereg.PLACEBO["seed"] + hash(cand) % 1000)
        log.info("placebo %s done in %.1fs (n=%d)", cand, time.time() - t1, n_draws)
    pd.DataFrame(placebo).to_parquet(out_dir / "placebo_sharpes.parquet")

    pbo_cols = list(prereg.CANDIDATES) + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]
    mats = []
    for c in pbo_cols:
        if c in prereg.CANDIDATES:
            mats.append(built["base"][c])
        else:
            mats.append(built["base"][c])
    pbo_frame = pd.DataFrame(np.column_stack(mats), columns=pbo_cols, index=built["dates"]).dropna()
    pbo = float(cscv_pbo(pbo_frame.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    trial_sr = (pbo_frame[list(prereg.CANDIDATES)].mean() / pbo_frame[list(prereg.CANDIDATES)].std(ddof=1)).to_numpy()
    log.info("PBO %.3f over %d series x %d days", pbo, pbo_frame.shape[1], pbo_frame.shape[0])

    sentinel_hits = set(detect_adjusted_close_sentinel_gap(list(built["col_of"]), EODHD / "us_universe_full"))
    log.info("cleaning-rule sentinel gap: %d/%d tickers show adjusted_close >= 999990",
             len(sentinel_hits), len(built["col_of"]))

    results = {}
    for cand in prereg.CANDIDATES:
        results[cand] = evaluate_candidate(built, cand, placebo[cand], pbo, trial_sr)
        top10 = top_contributors(built, cand)
        for row in top10:
            row["adjusted_close_sentinel_gap_affected"] = row["ticker"] in sentinel_hits
        results[cand]["top10_contributors"] = top10
        log.info("==> %s tier %s  bars=%s", cand, results[cand]["tier_reported"], results[cand]["bars"])

    sanity = sanity_checks(built)

    for cand, res in results.items():
        if res["tier"] in ("A", "B"):
            lag2 = two_day_lag_variant(built, cand)
            prim = built["primary"][prereg.CANDIDATES[cand]["N"]]
            ok = np.isfinite(lag2) & np.isfinite(prim)
            res["post_pass_audit"] = {
                "2day_lag_sharpe": sharpe(lag2[np.isfinite(lag2)]),
                "2day_lag_gap_vs_primary": sharpe(lag2[ok]) - sharpe(prim[ok]),
                "per_year": per_year_table(built["base"][cand], built["dates"]),
            }

    returns_frame = pd.DataFrame({**{c: built["base"][c] for c in prereg.CANDIDATES},
                                 "BM1_SPY": built["base"]["BM1_SPY"], "BM2_60_40": built["base"]["BM2_60_40"],
                                 "BM3_SPY_VT": built["base"]["BM3_SPY_VT"],
                                 "PRIMARY_N500": built["primary"][500], "PRIMARY_N1000": built["primary"][1000]},
                                index=built["dates"])
    returns_frame.to_parquet(out_dir / "returns_base.parquet")

    prereg_issues = [
        "No explicit cash/risk-free subtraction step is stated for the Sharpe-gap "
        "computation; every series already carries its own cash drag internally "
        "(BM3 on days it under-invests, any slot's early-exit residual), so raw net "
        "daily returns are compared directly -- subtracting rf again would double count it.",
        "Early-exit capital (delisting or a segment break mid-hold) is parked in cash "
        "until that slot's cohort next reforms, not redistributed pro-rata to the "
        "other names still held in that cohort -- the frozen file says only 'closed "
        "out ... no assumption of recovery', with no redistribution rule.",
        "S4's own frozen COSTS block restates only the stock ADV-bucket table, not an "
        "ETF rate for BM1/BM2/BM3/PRIMARY's own legs (SPY/IEF/VFITX) -- protocol Sec.2's "
        "general ETF rule (3bps/side, all clear the $50M ADV bar) is used for those.",
        "'Per side' is charged in full at both entry and exit (not split/halved across "
        "the round trip), reading the frozen COSTS key name literally.",
        "Decile size = round(N*0.10): 50 for N=500, 100 for N=1000 -- not otherwise "
        "specified numerically in the frozen file beyond 'top decile'.",
        "Every scheduled reformation (PRIMARY's monthly rebalance, the 1-month hold, "
        "each 6-month cohort) is modelled as a FRESH slot for every held name, with no "
        "netting of an unchanged continuing member's weight across reformations -- the "
        "conservative (higher-cost) reading of 'rebalanced monthly', applied "
        "symmetrically to candidate and PRIMARY so the GAP is not biased by it.",
        "A3 compares the candidate's own raw daily Sharpe to the distribution of "
        "placebo raw daily Sharpes (not a benchmark-relative gap), per the protocol's "
        "literal wording, unlike A1/A4/A5 which are explicitly framed as gaps.",
    ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(), "fingerprint": built["fingerprint"],
        "prereg_at": prereg.PREREGISTERED_AT, "window": prereg.WINDOW,
        "alpha_one_sided": ALPHA, "pbo": pbo, "dsr_trials": len(trial_sr),
        "candidates": results, "sanity_checks": sanity, "prereg_issues": prereg_issues,
        "cleaning_fingerprint_v2_at_freeze": prereg.OBSERVED["cleaning_fingerprint_v2"],
        "cleaning_fingerprint_v2_now": ec.cleaning_fingerprint(),
        "cleaning_note": "cleaning v2's label-only update (f62cb2e4 -> fc0690f0) changed "
                        "only frozen_at's text; CLEANING_RULES (the actual behaviour) is "
                        "unchanged, so phase-1's panel.parquet is valid input here.",
        "cleaning_rule_gap_found": {
            "description": "NOT a prereg ambiguity -- a defect in the SHARED, frozen "
                           "scripts/eodhd_clean.py used by all 5 shortlist candidates, "
                           "found while sanity-checking top-10 P&L contributors, reported "
                           "to the coordinator and NOT patched here (patching it would "
                           "retroactively change every other candidate's already-computed "
                           "results too). clean_bars' segment-break detector only fires on "
                           "a discrete one-day jump in adjusted_close; a CONSTANT vendor "
                           "sentinel value (999999.9999, an apparent overflow in EODHD's "
                           "own split-adjustment-factor computation) held for extended "
                           "stretches produces no jump and so is never flagged.",
            "n_tickers_in_this_run_affected": len(sentinel_hits),
            "n_tickers_in_this_run_total": len(built["col_of"]),
            "affected_tickers_sample": sorted(sentinel_hits)[:30],
            "note": "ratio_52wk is a within-segment RATIO, so a constant multiplicative "
                   "mis-adjustment cancels out of it exactly; adv63/adv20 dollar-volume "
                   "is NOT scale-invariant, so an affected name's liquidity can be "
                   "overstated, letting it into the top-N cut. See each candidate's "
                   "top10_contributors[].adjusted_close_sentinel_gap_affected flag.",
        },
        "harness_log": build_harness_log(
            results, current_git_commit(), run_started_utc,
            fix_before_this_run="run 2's segment-boundary-exit fix (see run 2 entry) -- "
                                "this run's own code has no fix pending before it; it is "
                                "the governing (most recent) result"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "S4", "entries": []}
        ledger["entries"].append({
            "date": datetime.now(UTC).date().isoformat(), "fingerprint": built["fingerprint"],
            "n_trials": len(trial_sr), "trials": list(prereg.CANDIDATES),
            "trial_daily_sharpes": [float(x) for x in trial_sr],
            "tiers": {k: v["tier"] for k, v in results.items()},
        })
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
        log.info("ledger updated: %s (cumulative %d)", LEDGER, ledger["cumulative_trials"])
    log.info("tiers: %s", {k: v["tier_reported"] for k, v in results.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default=str(_ROOT / "data" / "research" / "eodhd_s4_panel.parquet"))
    ap.add_argument("--report")
    ap.add_argument("--append-ledger", action="store_true")
    ap.add_argument("--limit-placebo", type=int, default=None, help="debug only: cap placebo draws")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
