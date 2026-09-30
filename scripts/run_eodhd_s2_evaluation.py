"""Pre-registered evaluation of S2 (market-breadth overlay on the 60/40 core).

Frozen design: ``scripts/eodhd_s2_breadth_overlay_preregistered_bars.py``
(DRAFT=False, PREREGISTERED_AT="2026-09-30T19:18:54Z", commit ff6f44a on
research/eodhd-S2). This harness implements the frozen bars exactly; it must
NOT be used to justify editing that file. Any bug or ambiguity found while
building this harness is recorded in the report under ``prereg_issues``, not
patched into the frozen module.

Inputs (local only, no network):
    data/research/eodhd/etfs_full/{SPY,IEF,BIL,VFITX}.parquet
    data/research/fred/DTB3.parquet
    $S/runs/S2/breadth.parquet   (built by scripts/eodhd_breadth.py from
        us_universe_full; REUSED AS-IS here -- see the module docstring's
        note on the cleaning-fingerprint label fix: behaviour is unchanged,
        only ``CLEANING_RULES['frozen_at']``'s label string changed between
        f62cb2e4... (when breadth.parquet was built) and fc0690f0... (the
        currently-frozen value cited in the pre-registration), so the file
        is valid to reuse without a rebuild).

Subcommands::

    python scripts/run_eodhd_s2_evaluation.py selfcheck --breadth <path>
    python scripts/run_eodhd_s2_evaluation.py evaluate  --breadth <path> --out <dir> \
        [--report docs/eodhd_s2_evaluation_2026_10.json] [--append-ledger]
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

import eodhd_clean as ec  # noqa: E402
import eodhd_s2_breadth_overlay_preregistered_bars as prereg  # noqa: E402
from firm.eval.overfitting import cscv_pbo, deflated_sharpe  # noqa: E402
from run_alt_premia_evaluation import stationary_indices  # noqa: E402  (reused verbatim, per protocol Sec.4)

log = logging.getLogger(__name__)

EODHD = _ROOT / "data" / "research" / "eodhd"
FRED = _ROOT / "data" / "research" / "fred"
LEDGER = _ROOT / "docs" / "S2_trial_history.json"
ANN = math.sqrt(prereg.TRADING_DAYS)
VARIANT_IDS = list(prereg.VARIANTS)  # ["V1_primary", "V2_cash_destination", "V3_stricter_threshold", "V4_alt_measure"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    dates: pd.DatetimeIndex            # full etfs_full/SPY calendar (from 1993-01-29), NOT window-sliced
    spy_ret: pd.Series                 # SPY total-return daily pct change
    bond_ret: pd.Series                # VFITX (nav) spliced onto IEF -- see PROXIES
    cash_ret: pd.Series                # FRED DTB3 accrual spliced onto BIL -- see PROXIES
    breadth_pct_above_prior: pd.Series  # pct_above_200sma measured at the PRIOR trading day's close
    breadth_net_ad_prior: pd.Series     # net_ad_21d measured at the PRIOR trading day's close
    first_of_month: np.ndarray          # bool, aligned to ``dates``
    ief_first: pd.Timestamp = pd.Timestamp("2002-07-26")  # real IEF inception (default: etfs_full's actual value)
    bil_first: pd.Timestamp = pd.Timestamp("2007-05-30")  # real BIL inception (default: etfs_full's actual value)


def _clean_equity(path: Path, cal: pd.DatetimeIndex) -> pd.DataFrame:
    return ec.clean_bars(pd.read_parquet(path), asset="equity", calendar=cal)[0]


def _clean_nav(path: Path, cal: pd.DatetimeIndex) -> pd.DataFrame:
    return ec.clean_bars(pd.read_parquet(path), asset="nav", calendar=cal)[0]


def _splice_returns(proxy: pd.DataFrame, real: pd.DataFrame, handoff_col_name: str) -> pd.Series:
    """Daily return series: ``proxy`` before ``real`` starts, ``real`` from its own first bar on.

    ``real``'s own first bar has no prior row to compute a return from (it is
    a new listing), so that single day's return is taken from ``proxy``
    instead -- the position only becomes literally the real ETF from the next
    trading day. This is a mechanical choice about how to define "day one" of
    a new listing's return, not a deviation from the frozen PROXIES design
    (which only fixes the handoff DATE, not this one-day edge case); recorded
    under prereg_issues in the report.
    """
    proxy_ret = proxy.set_index("date")["adjusted_close"].pct_change().sort_index()
    real_ret = real.set_index("date")["adjusted_close"].pct_change().sort_index()
    real_start = real_ret.index.min()
    before = proxy_ret[proxy_ret.index < real_start]
    after = real_ret[real_ret.index > real_start]
    first_day = proxy_ret.get(real_start, np.nan)
    if pd.isna(first_day):
        first_day = 0.0
        log.warning("%s: no proxy return available on the real asset's first day %s; set to 0.0",
                    handoff_col_name, real_start.date())
    first_day_series = pd.Series([first_day], index=[real_start])
    return pd.concat([before, first_day_series, after]).sort_index()


def _splice_cash(dtb3_daily: pd.Series, bil: pd.DataFrame) -> pd.Series:
    """Daily cash-leg return: FRED DTB3 accrual before BIL's own first bar, BIL's own
    total return from its first bar on (same one-day edge case as _splice_returns)."""
    bil_ret = bil.set_index("date")["adjusted_close"].pct_change().sort_index()
    bil_start = bil_ret.index.min()
    before = dtb3_daily[dtb3_daily.index < bil_start]
    after = bil_ret[bil_ret.index > bil_start]
    first_day_series = pd.Series([float(dtb3_daily.get(bil_start, 0.0))], index=[bil_start])
    return pd.concat([before, first_day_series, after]).sort_index()


def load_inputs(breadth_path: Path) -> Inputs:
    cal = ec.equity_calendar()  # etfs_full/SPY, from 1993-01-29
    spy = _clean_equity(EODHD / "etfs_full" / "SPY.parquet", cal)
    ief = _clean_equity(EODHD / "etfs_full" / "IEF.parquet", cal)
    bil = _clean_equity(EODHD / "etfs_full" / "BIL.parquet", cal)
    vfitx = _clean_nav(EODHD / "etfs_full" / "VFITX.parquet", cal)

    dates = pd.DatetimeIndex(spy["date"])
    spy_ret = spy.set_index("date")["adjusted_close"].pct_change().reindex(dates)
    bond_ret = _splice_returns(vfitx, ief, "bond_leg").reindex(dates).fillna(0.0)

    dtb3 = pd.read_parquet(FRED / "DTB3.parquet").set_index("date")["value"]
    dtb3 = dtb3[~dtb3.index.duplicated()]
    dtb3_daily = (dtb3.reindex(dtb3.index.union(dates)).ffill().reindex(dates) / 100.0 / prereg.TRADING_DAYS).fillna(0.0)
    cash_ret = _splice_cash(dtb3_daily, bil).reindex(dates).fillna(0.0)

    breadth = pd.read_parquet(breadth_path).set_index("date")
    breadth = breadth.reindex(dates)  # same calendar by construction (eodhd_breadth.py uses equity_calendar())
    breadth_pct_above_prior = breadth["pct_above_200sma"].shift(1)
    breadth_net_ad_prior = breadth["net_ad_21d"].shift(1)

    ym = dates.to_period("M")
    first_of_month = np.r_[True, ym[1:].to_numpy() != ym[:-1].to_numpy()]

    ief_first, bil_first = pd.Timestamp(ief["date"].min()), pd.Timestamp(bil["date"].min())
    return Inputs(dates, spy_ret, bond_ret, cash_ret, breadth_pct_above_prior, breadth_net_ad_prior,
                  first_of_month, ief_first, bil_first)


def window_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    start, end = pd.Timestamp(prereg.WINDOW["start"]), pd.Timestamp(prereg.WINDOW["end"])
    return (dates >= start) & (dates <= end)


# ---------------------------------------------------------------------------
# Simulator (drifting weights, 3 assets: SPY, bond_leg, cash_leg). Supports a
# per-day-per-asset cost array (T, N) so the proxy legs (VFITX, DTB3) can be
# charged 0 while the real ETF (IEF, BIL) is charged the protocol's ETF rate
# -- PROXIES: "never traded... carry no transaction cost by construction".
# ---------------------------------------------------------------------------

Decide = Callable[[int, np.ndarray], "np.ndarray | None"]


def simulate(rets: np.ndarray, decide: Decide, cost_bps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Net daily returns of a fully-invested (cash=False), long-only book.

    ``cost_bps`` is (N,) [constant per asset] or (T, N) [time-varying, e.g. a
    proxy-vs-real-ETF cost schedule]. Mirrors run_alt_premia_evaluation.py's
    ``simulate`` (cash=False path), generalised for a time-varying cost.
    """
    T, N = rets.shape
    w = np.zeros(N)
    out = np.zeros(T)
    held = np.zeros((T, N))
    time_varying = cost_bps.ndim == 2
    for i in range(T):
        r = np.nan_to_num(rets[i])
        rp = float(w @ r)
        if 1.0 + rp > 0:
            w = w * (1.0 + r) / (1.0 + rp)
        tgt = decide(i, w)
        cost = 0.0
        if tgt is not None:
            tgt = np.asarray(tgt, dtype=float)
            if tgt.sum() > prereg.EXECUTION["max_gross"] + 1e-9 or (tgt < -1e-12).any():
                raise ValueError(f"target violates long-only/max_gross at {i}: {tgt}")
            c = (cost_bps[i] if time_varying else cost_bps) / 1e4
            cost = float(np.abs(tgt - w) @ c)
            w = tgt
        out[i] = rp - cost
        held[i] = w
    return out, held


def sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * ANN) if sd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    nav = np.cumprod(1 + np.nan_to_num(r))
    peak = np.maximum.accumulate(nav)
    return float((1 - nav / peak).max())


def cagr(r: np.ndarray) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    total = float(np.prod(1 + np.nan_to_num(r)))
    return float(total ** (prereg.TRADING_DAYS / n) - 1)


def calmar(r: np.ndarray) -> float:
    dd = max_drawdown(r)
    return float(cagr(r) / dd) if dd > 0 else float("nan")


def paired_sharpe_gap_boot(a: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """Paired stationary block bootstrap of the annualised Sharpe gap a-b.

    Uses THIS module's own frozen BOOTSTRAP config (protocol Sec.4: mean
    block 63 trading days, 5000 draws, alpha 0.01 one-sided) -- NOT
    run_alt_premia_evaluation.py's own BOOTSTRAP (different seed/alpha), so
    only ``stationary_indices`` is reused from that module, per the
    coordinator's instruction, not its whole bootstrap wrapper.
    """
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


def _block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    blocks = [x[k:k + block] for k in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[k] for k in order])[:n]


def _seed_for(*parts: str) -> int:
    import hashlib
    return prereg.SEED + int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) % 10_000_000


# ---------------------------------------------------------------------------
# Overlay variants and benchmarks
# ---------------------------------------------------------------------------

def _rets_matrix(inp: Inputs) -> np.ndarray:
    return np.column_stack([inp.spy_ret.to_numpy(), inp.bond_ret.to_numpy(), inp.cash_ret.to_numpy()])


def _cost_matrix(inp: Inputs, stress: bool) -> np.ndarray:
    """(T, 3) cost array: SPY always charged; bond/cash legs charged only once
    the real ETF (IEF/BIL) has started -- 0 while a proxy (VFITX/DTB3)."""
    mult = prereg.COSTS["stress_multiplier"] if stress else 1.0
    etf_bps = prereg.COSTS["etf_bps_per_side_ge_50m_adv20"] * mult  # SPY/IEF/BIL always clear $50M ADV20 (design note)
    T = len(inp.dates)
    cost = np.zeros((T, 3))
    cost[:, 0] = etf_bps
    cost[:, 1] = np.where(inp.dates >= inp.ief_first, etf_bps, 0.0)
    cost[:, 2] = np.where(inp.dates >= inp.bil_first, etf_bps, 0.0)
    return cost


def _variant_decision_states(inp: Inputs, variant_id: str) -> np.ndarray:
    """Bool array (len(inp.dates)): True on a day where the variant's OWN measure/threshold
    reads "below" (overlay ON, SPY cut). NaN signal (not enough history yet) reads as OFF."""
    v = prereg.VARIANTS[variant_id]
    measure = inp.breadth_pct_above_prior if v["measure"] == "pct_above_200sma" else inp.breadth_net_ad_prior
    sig = measure.to_numpy()
    on = sig < v["threshold"]
    return np.where(np.isfinite(sig), on, False)


def _variant_simulate(inp: Inputs, variant_id: str, stress: bool = False,
                       on_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared simulation core for a variant: (net returns, on/off state, held weights (T,3))."""
    rets = _rets_matrix(inp)
    cost = _cost_matrix(inp, stress)
    on = np.asarray(on_override, dtype=bool) if on_override is not None else _variant_decision_states(inp, variant_id)
    destination_is_cash_leg = prereg.VARIANTS[variant_id]["destination"] == "BIL"
    fom = inp.first_of_month

    def decide(i: int, w: np.ndarray):
        if not fom[i]:
            return None
        if on[i]:
            cut = prereg.OVERLAY["spy_weight_cut"]
            dest = 1.0 - cut
            return np.array([cut, dest, 0.0]) if not destination_is_cash_leg else np.array([cut, 0.0, dest])
        return np.array([0.6, 0.4, 0.0])

    net, held = simulate(rets, decide, cost)
    return net, on, held


def variant_returns(inp: Inputs, variant_id: str, stress: bool = False,
                     on_override: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    """Net daily returns for one overlay variant, and its daily on/off state array.

    ``on_override``, when given, is used DIRECTLY as the final bool on/off
    array (e.g. a placebo's already-permuted daily state) -- it is NOT
    re-compared against the variant's threshold (that comparison only
    happens once, inside ``_variant_decision_states``, for the REAL signal).
    """
    net, on, _ = _variant_simulate(inp, variant_id, stress, on_override)
    return pd.Series(net, inp.dates, name=variant_id), on


def variant_held_weights(inp: Inputs, variant_id: str, stress: bool = False) -> np.ndarray:
    """(T, 3) realised weight path [SPY, bond_leg, cash_leg] for one overlay variant."""
    _, _, held = _variant_simulate(inp, variant_id, stress)
    return held


def bm1_spy(inp: Inputs, stress: bool = False) -> pd.Series:
    cost = _cost_matrix(inp, stress)[:, :1]
    r = inp.spy_ret.to_numpy()[:, None]
    first = int(np.argmax(~np.isnan(r[:, 0])))
    net, _ = simulate(r, lambda i, w: np.array([1.0]) if i == first else None, cost)
    return pd.Series(net, inp.dates, name="BM1_SPY")


def bm2_60_40(inp: Inputs, stress: bool = False) -> pd.Series:
    rets = _rets_matrix(inp)[:, :2]
    cost = _cost_matrix(inp, stress)[:, :2]
    fom = inp.first_of_month

    def decide(i: int, w: np.ndarray):
        return np.array([0.6, 0.4]) if fom[i] else None
    net, _ = simulate(rets, decide, cost)
    return pd.Series(net, inp.dates, name="BM2_60_40")


def bm3_spy_vt(inp: Inputs, stress: bool = False) -> pd.Series:
    spec = prereg.BENCHMARKS["BM3_SPY_VT"]
    rs = inp.spy_ret
    sig = (rs.rolling(spec["vol_window"]).std() * ANN).to_numpy()
    r = np.column_stack([rs.to_numpy(), inp.cash_ret.to_numpy()])
    cost = _cost_matrix(inp, stress)[:, [0, 2]]

    def decide(i: int, w: np.ndarray):
        if i < 1 or not np.isfinite(sig[i - 1]) or sig[i - 1] <= 0:
            return None
        tgt = min(1.0, spec["target_vol"] / sig[i - 1])
        if abs(tgt - w[0]) > spec["band_abs"]:
            return np.array([tgt, 1.0 - tgt])
        return None
    net, _ = simulate(r, decide, cost)
    return pd.Series(net, inp.dates, name="BM3_SPY_VT")


# ---------------------------------------------------------------------------
# POST-HOC diagnostic (does not change the frozen tier): is a variant's edge
# over BM2 timing, or just a lower average equity weight? A STATIC mix at the
# variant's own realised time-average SPY weight, same monthly cadence / 2%
# no-trade band / costs / window as BM2, isolates the answer.
# ---------------------------------------------------------------------------

def static_mix_returns(inp: Inputs, spy_weight: float, stress: bool = False) -> pd.Series:
    rets = np.column_stack([inp.spy_ret.to_numpy(), inp.bond_ret.to_numpy()])
    cost = _cost_matrix(inp, stress)[:, :2]
    fom = inp.first_of_month
    target = np.array([spy_weight, 1.0 - spy_weight])
    band = 0.02

    def decide(i: int, w: np.ndarray):
        if not fom[i]:
            return None
        if abs(w[0] - spy_weight) > band:
            return target
        return None
    net, _ = simulate(rets, decide, cost)
    return pd.Series(net, inp.dates, name=f"static_mix_{spy_weight:.4f}")


def posthoc_static_mix_diagnostic(inp: Inputs, wmask: np.ndarray, variant_id: str) -> dict:
    """Sharpe/CAGR/maxDD/Calmar for ``variant_id`` and a static SPY/bond-leg mix
    at its realised time-average SPY weight, plus the paired bootstrap Sharpe
    gap (variant - static mix). Labelled 'post-hoc, not a bar' by the caller."""
    held = variant_held_weights(inp, variant_id)
    avg_spy_w = float(pd.Series(held[:, 0], inp.dates).loc[wmask].mean())
    static = static_mix_returns(inp, avg_spy_w)

    v_net, _ = variant_returns(inp, variant_id)
    v = v_net.loc[wmask].dropna()
    s = static.loc[wmask].dropna()
    j = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()
    ve, se = j.v.to_numpy(), j.s.to_numpy()
    gap = sharpe(ve) - sharpe(se)
    boot = paired_sharpe_gap_boot(ve, se, _seed_for("posthoc_static_mix", variant_id))
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]
    lb, ub = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    return {
        "note": "post-hoc, not a bar -- does not change the frozen tier",
        "variant": variant_id, "realised_avg_spy_weight": avg_spy_w,
        variant_id: {"sharpe": sharpe(ve), "cagr": cagr(v.to_numpy()), "max_dd": max_drawdown(v.to_numpy()),
                     "calmar": calmar(v.to_numpy())},
        "static_mix": {"sharpe": sharpe(se), "cagr": cagr(s.to_numpy()), "max_dd": max_drawdown(s.to_numpy()),
                       "calmar": calmar(s.to_numpy())},
        "sharpe_gap_variant_minus_static": gap,
        "bootstrap_99pct_one_sided": {"lb": lb, "ub": ub, "alpha_one_sided": alpha},
    }


# ---------------------------------------------------------------------------
# Placebo: overlay on/off permuted in 63-trading-day blocks, at the daily
# frequency the state is held between rebalances (PLACEBO, verbatim).
# ---------------------------------------------------------------------------

def placebo_variant(inp: Inputs, variant_id: str, rng: np.random.Generator) -> pd.Series:
    real_on_monthly = _variant_decision_states(inp, variant_id)  # True only meaningful on rebalance days, but
    # "the daily frequency the state is held": expand to every day by holding each rebalance's decision
    # until the next one, THEN block-permute that fully-expanded daily series.
    fom = inp.first_of_month
    daily = np.zeros(len(inp.dates), dtype=bool)
    last = False
    for i in range(len(inp.dates)):
        if fom[i]:
            last = bool(real_on_monthly[i])
        daily[i] = last
    permuted_daily = _block_permute(daily, 63, rng)
    net, _ = variant_returns(inp, variant_id, on_override=permuted_daily)
    return net


# ---------------------------------------------------------------------------
# selfcheck: mechanical look-ahead test (a change after a cutoff must not move
# any series' return on or before that cutoff).
# ---------------------------------------------------------------------------

def cmd_selfcheck(args: argparse.Namespace) -> int:
    inp = load_inputs(Path(args.breadth))
    base = {vid: variant_returns(inp, vid)[0] for vid in VARIANT_IDS}
    base["BM1_SPY"], base["BM2_60_40"], base["BM3_SPY_VT"] = bm1_spy(inp), bm2_60_40(inp), bm3_spy_vt(inp)

    rng = np.random.default_rng(7)
    cutoffs = [pd.Timestamp(x) for x in ("2005-06-15", "2010-03-31", "2014-05-16", "2020-03-11", "2024-08-02")]
    bad = 0
    for cut in cutoffs:
        alt_inp = Inputs(
            inp.dates,
            _noise_after(inp.spy_ret, cut, rng), _noise_after(inp.bond_ret, cut, rng),
            _noise_after(inp.cash_ret, cut, rng),
            _noise_after(inp.breadth_pct_above_prior, cut, rng), _noise_after(inp.breadth_net_ad_prior, cut, rng),
            inp.first_of_month,
        )
        alt = {vid: variant_returns(alt_inp, vid)[0] for vid in VARIANT_IDS}
        alt["BM1_SPY"], alt["BM2_60_40"], alt["BM3_SPY_VT"] = bm1_spy(alt_inp), bm2_60_40(alt_inp), bm3_spy_vt(alt_inp)
        for name in list(base):
            a, b = base[name], alt[name]
            m = a.index <= cut
            diff = np.nanmax(np.abs(a[m].to_numpy() - b[m].to_numpy())) if m.any() else 0.0
            ok = diff == 0.0 or not np.isfinite(diff)
            log.info("look-ahead cutoff=%s %-24s max|diff| on/before cutoff = %.3g %s",
                     cut.date(), name, diff, "OK" if ok else "LEAK")
            bad += 0 if ok else 1
    log.info("selfcheck: %s", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


def _noise_after(s: pd.Series, cutoff: pd.Timestamp, rng: np.random.Generator) -> pd.Series:
    s = s.copy()
    mask = s.index > cutoff
    s.loc[mask] = s.loc[mask] + rng.normal(0, 0.01, size=int(mask.sum()))
    return s


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _halves(dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    mid = pd.Timestamp(prereg.WINDOW["midpoint"])
    lo = pd.Series(dates <= mid, index=dates)
    hi = pd.Series(dates > mid, index=dates)
    return lo, hi


def cmd_evaluate(args: argparse.Namespace) -> int:
    if prereg.DRAFT:
        raise RuntimeError("refusing to evaluate: pre-registration is still DRAFT")
    fp = prereg.bars_fingerprint()
    log.info("fingerprint %s (PREREGISTERED_AT %s)", fp, prereg.PREREGISTERED_AT)
    inp = load_inputs(Path(args.breadth))
    wmask = window_mask(inp.dates)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = prereg.BOOTSTRAP["alpha_one_sided"]

    base = {vid: variant_returns(inp, vid) for vid in VARIANT_IDS}      # {vid: (returns, on_state)}
    stress = {vid: variant_returns(inp, vid, stress=True)[0] for vid in VARIANT_IDS}
    bms = {"BM1_SPY": bm1_spy(inp), "BM2_60_40": bm2_60_40(inp), "BM3_SPY_VT": bm3_spy_vt(inp)}
    bms_stress = {"BM1_SPY": bm1_spy(inp, True), "BM2_60_40": bm2_60_40(inp, True), "BM3_SPY_VT": bm3_spy_vt(inp, True)}

    all_series = {vid: base[vid][0] for vid in VARIANT_IDS}
    all_series.update(bms)
    frame = pd.DataFrame(all_series)
    frame.loc[wmask].to_parquet(out_dir / "returns_base.parquet")
    pd.DataFrame({vid: stress[vid] for vid in VARIANT_IDS} | bms_stress).loc[wmask].to_parquet(
        out_dir / "returns_stress.parquet")
    pd.DataFrame({vid: base[vid][1] for vid in VARIANT_IDS}, index=inp.dates).loc[wmask].to_parquet(
        out_dir / "overlay_state.parquet")

    # sanity checks -------------------------------------------------------
    spy_w = bms["BM1_SPY"].loc[wmask]
    bm2_w = bms["BM2_60_40"].loc[wmask]
    big_moves = {}
    for name in frame.columns:
        s = frame.loc[wmask, name]
        flagged = s[s.abs() > 0.15]
        big_moves[name] = {"n": int(len(flagged)), "dates": [str(d.date()) for d in flagged.index],
                            "values": [float(v) for v in flagged.to_numpy()]}
    nan_days = {name: int(frame.loc[wmask, name].isna().sum()) for name in frame.columns}
    sanity = {
        "spy_cagr": cagr(spy_w.to_numpy()), "spy_vol": float(spy_w.std(ddof=1) * ANN), "spy_max_dd": max_drawdown(spy_w.to_numpy()),
        "bm2_cagr": cagr(bm2_w.to_numpy()), "bm2_vol": float(bm2_w.std(ddof=1) * ANN), "bm2_max_dd": max_drawdown(bm2_w.to_numpy()),
        "n_days_abs_ret_gt_15pct": big_moves, "n_nan_days": nan_days,
        "pct_time_defensive_by_variant": {
            vid: float(pd.Series(base[vid][1], inp.dates).loc[wmask].mean()) for vid in VARIANT_IDS
        },
    }
    log.info("sanity: %s", sanity)

    # placebo ---------------------------------------------------------------
    log.info("placebos: %d draws x %d variants", prereg.PLACEBO["n_draws"], len(VARIANT_IDS))
    prng = np.random.default_rng(prereg.PLACEBO["seed"])
    placebo_sharpes: dict[str, list[float]] = {vid: [] for vid in VARIANT_IDS}
    for draw in range(prereg.PLACEBO["n_draws"]):
        for vid in VARIANT_IDS:
            p = placebo_variant(inp, vid, prng)
            ex = (p.loc[wmask] - bms["BM2_60_40"].loc[wmask]).to_numpy()
            placebo_sharpes[vid].append(sharpe(ex))
        if draw % 100 == 0:
            log.info("placebo draw %d/%d", draw, prereg.PLACEBO["n_draws"])
    pd.DataFrame(placebo_sharpes).to_parquet(out_dir / "placebo_sharpes.parquet")

    # PBO / DSR ---------------------------------------------------------------
    pbo_cols = VARIANT_IDS + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]
    excess_vs_bm2 = frame.loc[wmask, pbo_cols].sub(frame.loc[wmask, "BM2_60_40"], axis=0).dropna()
    pbo_value = float(cscv_pbo(excess_vs_bm2.to_numpy(), n_partitions=prereg.PBO["n_partitions"]))
    trial_daily_sr = (excess_vs_bm2[VARIANT_IDS].mean() / excess_vs_bm2[VARIANT_IDS].std(ddof=1)).to_numpy()
    log.info("PBO %.3f over %d series x %d days; DSR trial daily Sharpes %s", pbo_value, len(pbo_cols),
              len(excess_vs_bm2), trial_daily_sr)

    lo_mask, hi_mask = _halves(inp.dates)
    results = {}
    for vid in VARIANT_IDS:
        c = frame.loc[wmask, vid]
        c_full = c.dropna()
        res = {"n_days": int(len(c_full)),
               "sharpe": sharpe(c_full.to_numpy()), "cagr": cagr(c_full.to_numpy()),
               "vol": float(c_full.std(ddof=1) * ANN), "max_dd": max_drawdown(c_full.to_numpy()),
               "calmar": calmar(c_full.to_numpy()), "vs": {}}
        ex_vs_bm2 = (c - bms["BM2_60_40"].loc[wmask]).dropna().to_numpy()
        res["dsr"] = float(deflated_sharpe(ex_vs_bm2, trial_daily_sr, prior_trials=prereg.DSR["prior_trials"]))
        arr = np.asarray(placebo_sharpes[vid])
        res["placebo_p95"] = float(np.nanpercentile(arr, prereg.PLACEBO["pass_percentile"]))
        res["A3"] = bool(sharpe(ex_vs_bm2) > res["placebo_p95"])

        for bm in ("BM1_SPY", "BM2_60_40", "BM3_SPY_VT"):
            b = bms[bm].loc[wmask]
            j = pd.concat([c, b], axis=1, keys=["c", "b"]).dropna()
            ce, be = j.c.to_numpy(), j.b.to_numpy()
            gap = sharpe(ce) - sharpe(be)
            boot = paired_sharpe_gap_boot(ce, be, _seed_for(vid, bm))
            lb, ub = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
            gap_lo = sharpe(j.c[lo_mask.reindex(j.index, fill_value=False)].to_numpy()) - \
                     sharpe(j.b[lo_mask.reindex(j.index, fill_value=False)].to_numpy())
            gap_hi = sharpe(j.c[hi_mask.reindex(j.index, fill_value=False)].to_numpy()) - \
                     sharpe(j.b[hi_mask.reindex(j.index, fill_value=False)].to_numpy())
            cs, bs = pd.Series(stress[vid]).loc[wmask], bms_stress[bm].loc[wmask]
            js = pd.concat([cs, bs], axis=1, keys=["c", "b"]).dropna()
            gap_stress = sharpe(js.c.to_numpy()) - sharpe(js.b.to_numpy())
            res["vs"][bm] = {
                "n_days": int(len(j)), "sharpe_c": sharpe(ce), "sharpe_b": sharpe(be), "gap": gap,
                "lb": lb, "ub": ub, "gap_half_1": gap_lo, "gap_half_2": gap_hi, "gap_stress": gap_stress,
                "max_dd_c": max_drawdown(ce), "max_dd_b": max_drawdown(be),
                "calmar_c": calmar(ce), "calmar_b": calmar(be),
                "A1": bool(gap > 0 and lb > 0) if bm == "BM2_60_40" else None,
                "A4": bool(gap_lo > 0 and gap_hi > 0), "A5": bool(gap_stress > 0),
                "D": bool(ub < 0) if bm == "BM2_60_40" else None,
            }
            log.info("%-24s vs %-12s SR %.2f vs %.2f gap %+.2f [LB %+.2f, UB %+.2f] halves %+.2f/%+.2f stress %+.2f",
                      vid, bm, sharpe(ce), sharpe(be), gap, lb, ub, gap_lo, gap_hi, gap_stress)

        v_bm2 = res["vs"]["BM2_60_40"]
        a8 = bool(v_bm2["max_dd_c"] < v_bm2["max_dd_b"] and v_bm2["calmar_c"] > v_bm2["calmar_b"]
                  and v_bm2["gap"] > -0.05)
        bars = {
            "A1": v_bm2["A1"], "A2": bool(res["dsr"] > 0.95), "A3": res["A3"],
            "A4": all(res["vs"][b]["A4"] for b in ("BM1_SPY", "BM2_60_40", "BM3_SPY_VT")),
            "A5": all(res["vs"][b]["A5"] for b in ("BM1_SPY", "BM2_60_40", "BM3_SPY_VT")),
            "A6": bool(pbo_value < 0.50), "A8": a8,
        }
        tier_d = any(res["vs"][b]["D"] for b in ("BM2_60_40",) if res["vs"][b]["D"] is not None)
        tier_b = {"B_a": bool(v_bm2["gap"] > 0), "B_b": bool(res["A3"] and bars["A4"] and bars["A5"])}
        res["bars"] = bars
        res["A7"] = "PENDING (independent recompute not yet performed)"
        res["tier_conditional_on_A7"] = prereg.classify(bars, tier_d, tier_b)
        results[vid] = res
        log.info("==> %s tier(conditional on A7)=%s bars=%s", vid, res["tier_conditional_on_A7"], bars)

    prereg_issues = [
        "VARIANTS['V2_cash_destination']'s OVERLAY rule_template (\"destination weight += "
        "(0.60 - spy_weight_cut)\") is ambiguous about what happens to the OTHER (non-destination) "
        "leg during the cut state when destination != IEF: does IEF keep its normal 0.40 (leaving "
        "0.30 in un-remunerated cash, total 0.30+0.40+0.30=1.0 with 0.40 idle) or does the destination "
        "fully replace the non-SPY sleeve (0.30 SPY / 0.70 BIL, IEF=0)? Implemented the latter (full "
        "replacement) as the most literal, fully-invested reading -- matches 'SPY 0.60/IEF 0.40 (plain "
        "core)' being the explicit ELSE branch for every variant, and avoids idle, non-cash-earning NAV.",
        "DSR['trial_sharpes'] text ('daily Sharpes of every variant the candidate declares') and "
        "PBO['series'] text ('daily excess returns vs BM2') don't explicitly say DSR's trial Sharpes "
        "are ALSO computed vs BM2 rather than as each variant's raw return Sharpe. Implemented "
        "consistently as excess-vs-BM2 throughout (DSR and PBO both), since A1/A4/A5/A8 are all framed "
        "'vs BM2' and PBO is explicit about the vs-BM2 convention.",
        "PBO['series'] includes BM2_60_40 itself as one of the 7 series being excess-differenced "
        "against BM2 -- that column is identically 0 by construction (a series can't be excess vs "
        "itself). Implemented literally (BM2's column is all zeros); does not appear to break cscv_pbo "
        "mechanically (a constant column never wins an in-sample ranking) but is a literal degenerate "
        "case worth flagging.",
        "The bond/cash-leg proxy splice (VFITX->IEF, DTB3->BIL) leaves the real ETF's OWN first "
        "trading day return undefined (no prior close in ITS OWN series); implemented as: that one day "
        "uses the proxy's return for the same date (see _splice_returns docstring) -- a mechanical "
        "choice, not a deviation from PROXIES' handoff DATE.",
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
        "prereg_at": prereg.PREREGISTERED_AT, "cleaning_fingerprint_used_for_breadth": "f62cb2e4... (label-only "
        "predecessor of the frozen fc0690f0...; cleaning BEHAVIOUR identical -- see module docstring)",
        "window": prereg.WINDOW, "alpha_one_sided": alpha, "pbo": pbo_value,
        "dsr_trials": len(trial_daily_sr), "dsr_prior_trials": prereg.DSR["prior_trials"],
        "sanity": sanity, "variants": results, "prereg_issues": prereg_issues,
        "A7_status": "PENDING -- every reported tier is conditional on A7 (independent recompute)",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "S2", "entries": []}
        ledger["entries"].append({
            "date": datetime.now(timezone.utc).date().isoformat(), "fingerprint": fp,
            "n_trials": len(trial_daily_sr), "trials": VARIANT_IDS,
            "trial_daily_sharpes": [float(x) for x in trial_daily_sr],
            "tiers_conditional_on_A7": {k: v["tier_conditional_on_A7"] for k, v in results.items()},
        })
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
        log.info("ledger updated: %s (cumulative %d)", LEDGER, ledger["cumulative_trials"])
    log.info("tiers (conditional on A7): %s", {k: v["tier_conditional_on_A7"] for k, v in results.items()})
    return 0


def cmd_posthoc_static_mix(args: argparse.Namespace) -> int:
    """POST-HOC diagnostic (does not change the frozen tier): merge
    'posthoc_static_mix_diagnostic' into an existing report.json for one or
    more variants, keyed by variant id."""
    inp = load_inputs(Path(args.breadth))
    wmask = window_mask(inp.dates)
    out = {}
    for vid in args.variants:
        res = posthoc_static_mix_diagnostic(inp, wmask, vid)
        out[vid] = res
        log.info("posthoc static-mix %-24s avg_spy_w=%.4f  %s sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f | "
                  "static sharpe=%.3f cagr=%.3f maxdd=%.3f calmar=%.3f | gap=%+.3f [LB %+.3f, UB %+.3f]",
                  vid, res["realised_avg_spy_weight"], vid, res[vid]["sharpe"], res[vid]["cagr"],
                  res[vid]["max_dd"], res[vid]["calmar"], res["static_mix"]["sharpe"], res["static_mix"]["cagr"],
                  res["static_mix"]["max_dd"], res["static_mix"]["calmar"], res["sharpe_gap_variant_minus_static"],
                  res["bootstrap_99pct_one_sided"]["lb"], res["bootstrap_99pct_one_sided"]["ub"])
    for report_path in args.report:
        p = Path(report_path)
        report = json.loads(p.read_text())
        report.setdefault("posthoc_static_mix_diagnostic", {}).update(out)
        p.write_text(json.dumps(report, indent=2, default=float))
        log.info("merged posthoc_static_mix_diagnostic into %s", p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_sc = sub.add_parser("selfcheck")
    p_sc.add_argument("--breadth", required=True)
    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--breadth", required=True)
    p_ev.add_argument("--out", required=True)
    p_ev.add_argument("--report")
    p_ev.add_argument("--append-ledger", action="store_true")
    p_ph = sub.add_parser("posthoc-static-mix")
    p_ph.add_argument("--breadth", required=True)
    p_ph.add_argument("--variants", nargs="+", required=True)
    p_ph.add_argument("--report", nargs="+", required=True, help="report.json path(s) to update in place")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"selfcheck": cmd_selfcheck, "evaluate": cmd_evaluate,
             "posthoc-static-mix": cmd_posthoc_static_mix}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
