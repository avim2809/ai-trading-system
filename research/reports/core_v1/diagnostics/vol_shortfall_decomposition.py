"""Read-only decomposition of the core_v1 volatility shortfall (w25a). NO engine run, NO backtest, NO ledger row, NO new trial.

The frozen target volatility is tau = 9%; the selected configuration's STORED engine returns have realised vol of about 2.2-2.4%. This
script rebuilds the sizing stages of the selected configuration (grid:4: forecast_cap 15, buffer 0.2, all speeds) from the real panel
(``firm.data.etf_loader`` through ``firm.research.data_access``, asof 2026-09-30) and the committed ``constants.json``, then measures the
realised vol of each stage as a frictionless fixed-capital replay (``units_{t-2} * P_{t-1} / capital * r_t``: no costs, no equity
compounding, no engine). The stages are cumulative, so the ratios multiply to the final figure:

  tau -> S0 unit forecast (f = 10 everywhere, all long) -> S1 actual combined forecast (FDM, floor 0, cap) -> S2 x vol scale
      -> S3 buffer -> S4 instrument risk cap -> S5 gross cap 1.0 (= the evaluation's sizing path)

S5 is checked against the stored engine series (read through ``data_access``, not re-run). Writes ``vol_shortfall_decomposition.json``
next to this file. The host ledger size and mtime are asserted unchanged and ``FIRM_RESEARCH_LEDGER_ROOT`` points at an empty tmp dir.

    PYTHONPATH=$PWD/src python research/reports/core_v1/diagnostics/vol_shortfall_decomposition.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
_TMP_LEDGER = tempfile.mkdtemp(prefix="w25a_ledger_")
os.environ["FIRM_RESEARCH_LEDGER_ROOT"] = _TMP_LEDGER

import core_v1_preregistered as pre

from firm.backtest import vector_engine as VE
from firm.data.etf_loader import load_etf_universe
from firm.portfolio import forecast_combine as FC
from firm.research import core_v1_pipeline as P
from firm.research import data_access, seal

HERE = Path(__file__).resolve().parent
HOST_LEDGER = Path("/local/store/research-ledger/trials.jsonl")
SELECTED_TRIAL = "f9f74dc1f4fc48898f3a3d7ae64c249d"   # results.json trial_ids['grid:grid:4']; committed ledger returns, READ ONLY
WINDOWS = {"2005+": ("2005-01-03", "2026-09-29"), "full": (None, None)}
SELECTED = {"forecast_cap": 15.0, "buffer_fraction": 0.2, "speed_subset": "all"}   # results.json selected_config
PPY = 252


def load_constants(consts: dict) -> P.ConstantsBundle:
    """Same construction as scripts/run_core_v1_evaluation.load_constants (not importable without its startup checks)."""
    rho = pd.DataFrame(np.array(consts["rho"]["matrix"], dtype=float), index=consts["rho"]["rules"], columns=consts["rho"]["rules"])
    surv = {s: {k: [int(x) for x in v] for k, v in d.items()} for s, d in consts["speed_filter"]["survivors"].items()}
    return P.ConstantsBundle(
        scalars={k: float(v) for k, v in consts["scalars"].items()}, survivors=surv, rho=rho, idm=float(consts["idm"]),
        instrument_weights={k: float(v) for k, v in consts["instrument_weights"].items()},
        group_weights={k: float(v) for k, v in consts["group_weights"].items()}, window=(consts["window"][0], consts["window"][1]),
        asset_class={m: c for c, members in pre.ASSET_CLASSES.items() for m in members})


def vol_of(x: pd.Series, lo, hi) -> float:
    return float(x.loc[lo:hi].std() * np.sqrt(PPY))


def replay(units: pd.DataFrame, panel: P.Panel, cap) -> pd.Series:
    """Frictionless return of decided unit targets: held_{t-1} = target_{t-2} (one bar decide->fill lag), divided by ``cap`` (a number, or
    a per-date series of the equity at the START of the day)."""
    held_prev = units.shift(2).fillna(0.0)
    w = held_prev * panel.close.shift(1)
    w = w / cap if np.isscalar(cap) else w.div(cap, axis=0)
    return (w * panel.ret.reindex_like(panel.close).fillna(0.0)).sum(axis=1)


def stored_engine_returns() -> pd.Series:
    path = Path(os.environ.get("RESEARCH_LEDGER_RETURNS", "/local/store/research-ledger/returns")) / f"{SELECTED_TRIAL}.parquet"
    df = data_access.read_parquet(path, asof=seal.max_research_date())
    r = df["returns"].astype(float)
    r.index = pd.DatetimeIndex(pd.to_datetime(df.index if "date" not in df.columns else df["date"]))
    return r


def main() -> int:
    ledger_before = HOST_LEDGER.stat()
    asof = seal.max_research_date()
    seal.check_asof(asof, what="vol_shortfall_decomposition")
    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    consts = json.loads((ROOT / "research/reports/core_v1/constants.json").read_text())
    constants = load_constants(consts)
    tau = float(consts["tau"])
    base = P.params_from_gates(gates, tau, frozen=pre.FROZEN_LIMITS)
    params, ids = P.with_config(base, SELECTED, pre.SPEED_SUBSETS)
    series = load_etf_universe(ROOT / "config/universe_etf.yaml", asof=asof, include_delisted=False)
    panel = P.panel_from_series({s: series[s] for s in pre.UNIVERSE}, snapshot_id="diag", dividends=None, fx=None, source="diag")
    syms, idx = panel.symbols, panel.close.index

    # ---- forecast side (no sizing): vol, signed rule forecasts, combined forecast (FDM, floor 0, cap) and its un-floored twin
    vol = P.vol_annual(panel, params)
    raw = P.raw_forecasts(panel, params, P.rule_defs(base, params), vol)
    fcs = P.signed_forecasts(raw, constants.scalars, params)
    comb_floor = pd.DataFrame(np.nan, index=idx, columns=syms)
    comb_signed = comb_floor.copy()
    fdm_used = {}
    for s in syms:
        w, fdm = P.fdm_for(constants, s, ids, params)
        fdm_used[s] = fdm
        comb_floor[s] = FC.combine_forecasts(fcs[s][list(w)], w, fdm, cap=params.forecast_cap, floor=0.0)
        comb_signed[s] = FC.combine_forecasts(fcs[s][list(w)], w, fdm, cap=params.forecast_cap, floor=-params.forecast_cap)
    active = comb_floor.notna()
    ev_start = vol.dropna(how="all").index[0]

    w_inst = constants.instrument_weights
    mult = dict.fromkeys(syms, 1.0)
    cap_cash = P.INITIAL_CAPITAL
    raw_w = P.raw_portfolio_weights(comb_floor, vol, constants, tau)
    lim, scale, cov = P._limits_inputs(panel, constants, params, comb_floor, vol)

    def size(comb, *, scale_s=None, buffer, limits=False, gross=None, diag=None):
        kw = {}
        if limits:
            kw = {"risk_limits": lim, "cov_annual": cov, "diagnostics": diag if diag is not None else {}, "risk_cap_kwargs": P.CAP_OPTIONS}
        t, flags = VE.targets_from_forecasts(
            comb, panel.close, vol, weights=w_inst, idm=constants.idm, tau=tau, capital=cap_cash, multipliers=mult,
            buffer_fraction=buffer, long_only=True, gross_cap=gross, rounding="toward_zero", vol_scale=scale_s, **kw)
        return t, flags

    ten = pd.DataFrame(np.where(active, 10.0, np.nan), index=idx, columns=syms)
    big = 1.0e3   # gross cap disabled in the risk-cap path (risk shares are scale invariant, see targets_from_forecasts)
    diag_a: dict = {}
    stages = {}
    stages["S0_unit_forecast_10"] = size(ten, buffer=0.0, gross=None)
    stages["S1_combined_forecast"] = size(comb_floor, buffer=0.0, gross=None)
    stages["S2_vol_scale"] = size(comb_floor, scale_s=scale, buffer=0.0, gross=None)
    stages["S3_buffer"] = size(comb_floor, scale_s=scale, buffer=params.buffer_fraction, gross=None)
    stages["S4_instrument_cap"] = size(comb_floor, scale_s=scale, buffer=params.buffer_fraction, limits=True, gross=big)
    stages["S5_gross_cap_final"] = size(comb_floor, scale_s=scale, buffer=params.buffer_fraction, limits=True, gross=params.gross_cap, diag=diag_a)
    stages["X_gross_cap_without_instrument_cap"] = size(comb_floor, scale_s=scale, buffer=params.buffer_fraction, gross=params.gross_cap)

    stored = stored_engine_returns()
    out: dict = {"asof": asof.isoformat(), "tau": tau, "selected_config": SELECTED, "eval_start": ev_start.date().isoformat(), "windows": {}}
    reps = {k: replay(v[0], panel, cap_cash) for k, v in stages.items()}
    # S6: the stored engine equity (100k * cumprod(1 + stored returns), start-of-day) instead of the fixed sizing capital as the denominator
    eq_start = (P.INITIAL_CAPITAL * (1.0 + stored.reindex(idx).fillna(0.0)).cumprod()).shift(1).fillna(P.INITIAL_CAPITAL)
    reps["S6_equity_denominator"] = replay(stages["S5_gross_cap_final"][0], panel, eq_start)
    gross = {k: (v[0] * panel.close / cap_cash).sum(axis=1) for k, v in stages.items()}
    held_n = {k: (v[0] > 0).sum(axis=1) for k, v in stages.items()}
    for wname, (lo, hi) in WINDOWS.items():
        lo = lo or ev_start.date().isoformat()
        hi = hi or idx[-1].date().isoformat()
        names = list(stages)
        reps_names = [*names, "S6_equity_denominator"]
        vols = {k: vol_of(reps[k], lo, hi) for k in reps_names}
        chain = ["S0_unit_forecast_10", "S1_combined_forecast", "S2_vol_scale", "S3_buffer", "S4_instrument_cap", "S5_gross_cap_final", "S6_equity_denominator"]
        ratios = {"S0_unit_forecast_10": vols["S0_unit_forecast_10"] / tau}
        for a, b in itertools.pairwise(chain):
            ratios[b] = vols[b] / vols[a]
        mask = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
        st = stored.loc[lo:hi]
        st_vol = float(st.std() * np.sqrt(PPY))
        out["windows"][wname] = {
            "lo": lo, "hi": hi, "days": int(mask.sum()),
            "stage_realised_vol": vols, "stage_ratio_to_previous": ratios, "product_of_ratios_times_tau": float(tau * np.prod(list(ratios.values()))),
            "stage_mean_gross": {k: float(gross[k][mask].mean()) for k in names}, "stage_mean_n_held": {k: float(held_n[k][mask].mean()) for k in names},
            "stored_over_S6_vol": float(st_vol / vols["S6_equity_denominator"]), "mean_equity_over_capital": float((eq_start[mask] / P.INITIAL_CAPITAL).mean()),
            "stored_engine_vol": float(st.std() * np.sqrt(PPY)), "stored_engine_days": len(st),
            "corr_replay_S5_vs_stored": float(reps["S5_gross_cap_final"].loc[lo:hi].corr(st)),
            "ratio_stored_over_tau": float(st.std() * np.sqrt(PPY) / tau),
        }
        # forecast statistics on active instrument-days of the window
        a = active.loc[lo:hi]
        cf, cs = comb_floor.loc[lo:hi].where(a), comb_signed.loc[lo:hi].where(a)
        n = int(a.to_numpy().sum())
        out["windows"][wname]["forecast"] = {
            "active_instrument_days": n, "frac_combined_zero": float((cf == 0).to_numpy().sum() / n),
            "mean_combined_floored": float(np.nanmean(cf.to_numpy())), "mean_abs_combined_unfloored": float(np.nanmean(cs.abs().to_numpy())),
            "mean_combined_unfloored_signed": float(np.nanmean(cs.to_numpy())), "frac_at_cap": float((cf >= params.forecast_cap - 1e-12).to_numpy().sum() / n),
            "frac_days_all_instruments_flat": float(((cf.fillna(0) == 0).all(axis=1)).mean()),
        }
        sc = scale.loc[lo:hi].dropna()
        rw = raw_w.loc[lo:hi]
        out["windows"][wname]["vol_scale"] = {
            "mean": float(sc.mean()), "min": float(sc.min()), "max": float(sc.max()), "frac_at_cap": float((sc >= lim.max_vol_scale - 1e-12).mean()),
            "implied_sigma_hat_median": float(np.median(tau / sc)), "raw_weight_gross_mean": float(rw.sum(axis=1).mean()),
        }
    out["selected_sizing_diagnostics_replay"] = diag_a
    # S0 gross by instrument: weight_i * IDM * tau / sigma_i (the fraction of capital needed at forecast 10), 2005+ mean
    s0w = (stages["S0_unit_forecast_10"][0] * panel.close / cap_cash).loc["2005-01-03":"2026-09-29"]
    out["s0_mean_weight_by_symbol_2005plus"] = {s: float(s0w[s].mean()) for s in syms}
    out["fdm_used"] = fdm_used
    out["idm"] = constants.idm
    out["instrument_weights"] = w_inst
    out["instrument_weight_sum"] = float(sum(w_inst.values()))
    # structural: instrument risk cap = 2 x weight; is the cap set feasible (sum of caps of HELD instruments >= 1)?
    held = stages["S3_buffer"][0] > 0
    ws = pd.Series(w_inst)
    held_share = held.mul(ws, axis=1).sum(axis=1)
    m = held.any(axis=1) & (idx >= ev_start)
    out["held_weight_share"] = {"mean_when_any_held": float(held_share[m].mean()),
                                "frac_days_cap_set_infeasible_among_held_days": float((held_share[m] * lim.max_instrument_risk_mult < 1.0 - 1e-12).mean()),
                                "frac_days_nothing_held": float((~held.any(axis=1))[idx >= ev_start].mean())}
    # vol normalisation: average 1/sigma_i weighted by instrument weight, and the floor on vol
    v_ev = vol.loc[ev_start:]
    out["vol_annual_mean_by_symbol"] = {s: float(v_ev[s].mean()) for s in syms}
    # ledger guard
    ledger_after = HOST_LEDGER.stat()
    assert (ledger_before.st_size, ledger_before.st_mtime_ns) == (ledger_after.st_size, ledger_after.st_mtime_ns), "host ledger changed"
    assert not any(Path(_TMP_LEDGER).iterdir()), "tmp ledger root not empty"
    out["host_ledger_unchanged"] = {"size": ledger_after.st_size, "mtime": int(ledger_after.st_mtime)}
    (HERE / "vol_shortfall_decomposition.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for wname, r in out["windows"].items():
        print(f"== window {wname} {r['lo']}..{r['hi']} ({r['days']} days)  stored engine vol {r['stored_engine_vol']*100:.3f}%  "
              f"(x{r['ratio_stored_over_tau']:.3f} of tau)  corr(replay S5, stored) {r['corr_replay_S5_vs_stored']:.3f}")
        for k, v in r["stage_realised_vol"].items():
            print(f"  {k:38s} vol {v*100:7.3f}%  ratio-to-previous {r['stage_ratio_to_previous'].get(k, float('nan')):6.3f}  "
                  f"mean gross {r['stage_mean_gross'].get(k, float('nan')):6.3f}  "
                  f"mean n held {r['stage_mean_n_held'].get(k, float('nan')):5.2f}")
        print("  forecast", json.dumps(r["forecast"], sort_keys=True))
        print("  vol_scale", json.dumps(r["vol_scale"], sort_keys=True))
    print("replay sizing diagnostics", json.dumps(diag_a, sort_keys=True))
    print("held weight share", json.dumps(out["held_weight_share"], sort_keys=True))
    print("host ledger unchanged: size", out["host_ledger_unchanged"]["size"], "mtime", out["host_ledger_unchanged"]["mtime"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
