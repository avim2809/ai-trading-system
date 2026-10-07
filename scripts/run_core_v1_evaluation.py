"""P3-08 driver: the core_v1 research run through G-RESEARCH 1-8 (ETF path). ONE run, honestly reported as pass, fail or insufficient.

    cd <worktree> && nice -n 10 ionice -c3 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$PWD/src \\
      prlimit --as=2500000000 <research-venv>/bin/python scripts/run_core_v1_evaluation.py \\
      --manifest research/data_manifests/<etf manifest>.json

Same guards as ``scripts/run_core_v1_constants.py`` (non-root, sealed paths unreadable, ETF store readable, red-team pass recorded, run window
from ``date -u``, address-space cap; no override flag exists) plus step 1 below. It never edits ``config/gates.yaml`` or the frozen module:
the thresholds come from the gates file through ``firm.reporting.gate_report``, the tier is decided by ``gate_report`` and never by prose.

Step 1 verifies (any failure is a STOP): frozen inputs and the charter (``pre.verify_before_run``: gates sha256, stress-periods sha256, universe
and tax hashes, charter approved and committed before the first core_v1 ledger row); the pre-registration is APPROVED in the index and covers the
default grid point; the owner-copied constants addendum ``research/preregistration/core_v1_constants_addendum.yaml`` exists and its recorded
sha256 equals the sha256 of ``research/reports/core_v1/constants.json``; the instrument-weight vector in ``constants.json`` equals the frozen hash.

Then: grid (at most 12) at 1x cost, each a registered ledger trial; CPCV (10 groups, 2 test groups, 9 paths, 7-of-9 rule, with the selection rule
re-run on the purged and embargoed TRAIN groups of EVERY split); PBO (S=16); DSR with the FAMILY N (gate) and the raw ledger count reported
side by side, N=463 and N=210 sensitivities, length-adjusted var_sr floored at the grid variance (unadjusted and all-family sensitivities);
cost stress 1x/2x/3x plus a comparison-only flat-bps row; the stress suite against the charter's bootstrap reference; robustness +/-25% one
sided with the pooled scalars re-estimated; the after-tax benchmark (BM2 ANNUAL via ``gate_benchmark_bm2``, monthly as sensitivity,
``core_only_100`` re-run through the engine) with a power analysis; the H4 ENB check; the Kelly bound ``tau <= 0.5 x Kelly vol`` once, on the
deflated Sharpe with a 50% haircut (a breach is Tier D). Outputs: ``research/reports/core_v1/{results.json,REPORT.md}`` and an append-only
trial-history DRAFT under ``research/reports/core_v1/drafts/`` (the owner commits it to ``docs/core_v1_trial_history.json``).

Known leakage listed before any result (gates ``cpcv.in_sample_procedure``): the pooled scalars, speed filter, FDM and IDM come from the
full-window P3-11 run and are NOT re-estimated on the CPCV training groups (that would need 45 x 12 further ledgered engine runs); the
instrument weights are fixed ex ante. Every engine run also writes one exploratory ledger row under the family ``core_v1_engine`` (the engine
refuses an unlogged run), so the raw ledger count exceeds the enumerated registered count by the number of engine runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import core_v1_preregistered as pre
import numpy as np
import pandas as pd
import run_core_v1_constants as C1
import yaml

import firm.research  # noqa: F401  (ledger / seal entry-point check)
from firm.reporting import after_tax as AT
from firm.reporting import diversification_report as DR
from firm.reporting import gate_report as GR
from firm.research import core_v1_pipeline as P
from firm.research import ledger as L
from firm.research import run_guards as G
from firm.research import seal
from firm.validation import cv as CV
from firm.validation import effective_trials as ET
from firm.validation import pbo as PBO
from firm.validation import sharpe_stats as SS
from firm.validation import stress_suite as SU

log = logging.getLogger("run_core_v1_evaluation")

OUT_SUBDIR = Path("research") / "reports" / "core_v1"
ADDENDUM_REL = "research/preregistration/core_v1_constants_addendum.yaml"
CONSTANTS_REL = "research/reports/core_v1/constants.json"
SQRT_PPY = math.sqrt(252)
LEGACY_VAR_SR_SOURCES = {          # var_sr_family name -> (legacy source file, trial name or index)
    "legacy_trend_standalone": ("docs/standalone_strategy_trial_history.json", "trend"),
    "alt_premia_T2": ("docs/alt_premia_trial_history.json", "T2_cross_asset_trend"),
    "alt_premia_C1": ("docs/alt_premia_trial_history.json", "C1_btc_trend"),
    **{f"eodhd_s3_trial_{k}": ("docs/S3_trial_history.json", k - 1) for k in range(1, 6)},
}
N_RAW_SENSITIVITIES = {"raw_N_463": 463, "ledgered_N_210": 210}
KNOWN_LEAKAGE = [
    "pooled scalars, speed filter, FDM and IDM were estimated on the full window (P3-11) and are not re-estimated on the CPCV training groups",
    "the instrument weights are fixed ex ante (handcrafted, one group per asset class); the universe was chosen without performance input",
    "all data to 2026-09-30 is in-sample (OD-04): CPCV and PBO address selection among the grid only",
]


# ---------------------------------------------------------------------------------------------------------------------
# step 1: verification
# ---------------------------------------------------------------------------------------------------------------------
class StartupError(RuntimeError):
    """A step-1 check failed: STOP (nothing has run)."""


def file_sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_start(repo: Path, *, ledger=None, check_index: bool = True) -> dict:
    """Step 1. Returns the verified facts; raises ``StartupError`` / ``PreregError`` on any failure."""
    facts = pre.verify_before_run(repo, ledger=ledger)
    if check_index:
        from firm.research import prereg as PR

        if not PR.is_approved(pre.FAMILY, dict(pre.DEFAULT_CONFIG), repo_dir=repo):
            raise StartupError("the core_v1 pre-registration is not APPROVED (or does not cover the default grid point) in the index")
        problems = PR.verify_index(PR.load_index(), repo)
        if problems:
            raise StartupError(f"prereg index does not verify: {problems[:3]}")
    cj, add = repo / CONSTANTS_REL, repo / ADDENDUM_REL
    if not cj.is_file():
        raise StartupError(f"{CONSTANTS_REL} is missing: run scripts/run_core_v1_constants.py first")
    if not add.is_file():
        raise StartupError(f"{ADDENDUM_REL} is missing: the owner has not yet copied constants_addendum.draft.yaml (refusing to run)")
    rec = yaml.safe_load(add.read_text(encoding="utf-8"))
    if rec.get("constants_json_sha256") != file_sha256(cj):
        raise StartupError("constants.json sha256 differs from the one recorded in the owner-committed addendum")
    consts = json.loads(cj.read_text(encoding="utf-8"))
    if consts["gates_sha256"] != pre.GATES_SHA256 or consts["prereg_fingerprint"] != pre.bars_fingerprint():
        raise StartupError("constants.json was produced under a different gates file or frozen module")
    if consts["instrument_weights_sha256"] != pre.INSTRUMENT_WEIGHTS_SHA256:
        raise StartupError("constants.json instrument-weight hash differs from the frozen module's")
    if abs(float(consts["tau"]) - float(facts["tau"])) > 0:
        raise StartupError("constants.json tau differs from the charter's tau")
    return {"charter": facts, "constants": consts, "addendum": rec, "constants_sha256": file_sha256(cj)}


def load_constants(consts: dict) -> P.ConstantsBundle:
    rho = pd.DataFrame(np.array(consts["rho"]["matrix"], dtype=float), index=consts["rho"]["rules"], columns=consts["rho"]["rules"])
    surv = {s: {k: [int(x) for x in v] for k, v in d.items()} for s, d in consts["speed_filter"]["survivors"].items()}
    return P.ConstantsBundle(scalars={k: float(v) for k, v in consts["scalars"].items()}, survivors=surv, rho=rho, idm=float(consts["idm"]),
                             instrument_weights={k: float(v) for k, v in consts["instrument_weights"].items()},
                             group_weights={k: float(v) for k, v in consts["group_weights"].items()},
                             window=(consts["window"][0], consts["window"][1]),
                             asset_class={m: c for c, members in pre.ASSET_CLASSES.items() for m in members})


# ---------------------------------------------------------------------------------------------------------------------
# grid, selection, CPCV, PBO
# ---------------------------------------------------------------------------------------------------------------------
def build_grid(prereg=pre) -> list[dict]:
    """The pre-registered grid (at most 12 configs), taken from the frozen module; nothing is added or reordered."""
    grid = [dict(g) for g in prereg.GRID]
    if len(grid) > prereg.MAX_GRID_SIZE:
        raise StartupError(f"grid has {len(grid)} configs > {prereg.MAX_GRID_SIZE}")
    return grid


def sharpe_pp(x: np.ndarray) -> np.ndarray:
    return PBO.sharpe_cols(x)


def select_argmax(train: np.ndarray) -> int:
    """Pre-registered selection rule: highest training-group per-period net Sharpe (1x cost); ties to the lowest grid index."""
    return int(np.argmax(sharpe_pp(train)))


def run_grid(grid: list[dict], panel: P.Panel, constants: P.ConstantsBundle, base: P.CoreParams, ledger_ctx: P.LedgerCtx,
             speed_subsets: dict | None = None) -> tuple[pd.DataFrame, list[P.RunResult]]:
    """Every grid config through the engine at 1x cost; each one is a registered ledger trial (``kind='grid'``). Returns the T x N matrix."""
    subsets = speed_subsets or pre.SPEED_SUBSETS
    cols, runs = {}, []
    for i, cfg in enumerate(grid):
        params, ids = P.with_config(base, cfg, subsets)
        res = P.run_config(panel, constants, params, subset_ids=ids, ledger_mode=ledger_ctx.mode, label=f"grid:{i}", base=base,
                           prereg=ledger_ctx.prereg, snapshot_id=ledger_ctx.snapshot_id, seed=ledger_ctx.seed)
        with P.logged_trial(ledger_ctx, "grid", "grid", str(i), params=cfg, detail={"config": cfg, "engine_trial": res.trial_id},
                            returns=res.engine.excess_returns):
            pass
        cols[i] = res.engine.excess_returns
        runs.append(res)
    return pd.DataFrame(cols), runs


def run_cpcv(matrix: pd.DataFrame, selection_rule=select_argmax, n_groups: int = 10, k: int = 2, *, embargo_pct: float) -> dict:
    """CPCV where the selection rule is re-run on the TRAIN groups of EVERY split; the chosen config's returns on the test groups fill the paths."""
    M = matrix.to_numpy(float)
    idx = matrix.index
    t1 = pd.Series(idx[1:].append(idx[-1:] + pd.Timedelta(days=1)), index=idx)
    cv = CV.CombinatorialPurgedCV(n_groups, k, label_end=t1, embargo_pct=embargo_pct)
    preds, chosen = {}, []
    for sid, (train, test) in enumerate(cv.split()):
        sel = int(selection_rule(M[train]))
        chosen.append(sel)
        preds[sid] = M[test, sel]
    paths = cv.backtest_paths(preds)
    sr = [float(sharpe_pp(p[:, None])[0]) for p in paths]
    counts = {int(c): int(n) for c, n in zip(*np.unique(chosen, return_counts=True), strict=True)}
    return {"path_sharpes": sr, "n_paths": cv.n_paths, "n_splits": cv.get_n_splits(), "median": float(np.median(sr)),
            "n_positive": int(sum(x > 0 for x in sr)), "selection_counts": counts, "embargo_pct": embargo_pct}


def run_pbo(matrix: pd.DataFrame, S: int, family_variant_count: int) -> dict:
    """PBO by CSCV (S blocks) plus the selected config's median OOS Sharpe and the informativeness inputs (effective grid N, column correlation)."""
    M = matrix.to_numpy(float)
    calls: list[np.ndarray] = []

    def metric(x):
        s = sharpe_pp(x)
        calls.append(s)
        return s

    res = PBO.pbo(M, S=S, metric=metric)
    is_m, oos_m = np.vstack(calls[0::2]), np.vstack(calls[1::2])
    star = np.argmax(is_m, axis=1)
    oos_star = oos_m[np.arange(len(star)), star]
    corr = ET.pairwise_corr(matrix, min_overlap=min(250, len(matrix) // 2)).to_numpy()
    off = corr[np.triu_indices_from(corr, k=1)]
    off = off[np.isfinite(off)]
    try:
        eff = float(ET.effective_n(matrix, "enb", min_overlap=min(250, len(matrix) // 2)))
    except (ValueError, np.linalg.LinAlgError):
        eff = 1.0
    return {"pbo": float(res.pbo), "prob_oos_loss": float(res.prob_oos_loss), "selected_median_oos_sharpe": float(np.median(oos_star)),
            "n_splits": int(res.n_splits), "n_blocks": S, "grid_n": int(M.shape[1]), "effective_grid_n": eff,
            "median_pairwise_corr": float(np.median(off)) if len(off) else 1.0, "family_variant_count": int(family_variant_count),
            "perf_degradation_slope": float(res.perf_degradation_slope), "rows_used": int(res.rows_used)}


# ---------------------------------------------------------------------------------------------------------------------
# DSR and variance
# ---------------------------------------------------------------------------------------------------------------------
def legacy_family_sharpes(ledger_trials: pd.DataFrame, names: list[str]) -> tuple[dict[str, float], list[float]]:
    """Per-period Sharpes of the legacy members of ``var_sr_family`` (from the legacy ledger rows), and every legacy per-period Sharpe in
    the ledger (for the all-family sensitivity). Raises if a named member cannot be resolved or the periods_per_year differ."""
    leg = ledger_trials[ledger_trials["mode"] == "legacy"]
    out: dict[str, float] = {}
    for name in names:
        if name == "core_v1_grid":
            continue
        src, key = LEGACY_VAR_SR_SOURCES[name]
        rows = leg[leg["source_file"] == src]
        if rows.empty:
            raise StartupError(f"legacy ledger rows for {src} are missing (var_sr_family member {name})")
        cfg = rows.iloc[0]["config"]
        trials = cfg.get("trials") or cfg.get("variant_names")
        sharpes = cfg["trial_daily_sharpes"]
        j = trials.index(key) if isinstance(key, str) else int(key)
        out[name] = float(sharpes[j])
        if rows.iloc[0]["periods_per_year"] not in (252, None) and not pd.isna(rows.iloc[0]["periods_per_year"]):
            raise StartupError("refusing to pool trials with different periods_per_year")
    allv: list[float] = []
    for _, r in leg.iterrows():
        v = (r["config"] or {}).get("trial_daily_sharpes")
        if isinstance(v, list):
            allv += [float(x) for x in v if x is not None and math.isfinite(float(x))]
    return out, allv


def dsr_or_none(sr, n_obs, skew, kurt, n_trials, var_sr):
    try:
        return float(SS.dsr(sr, n_obs, skew, kurt, int(n_trials), float(var_sr)))
    except (SS.DegenerateInputError, ValueError):
        return None


def build_dsr(sel_ret: pd.Series, grid_sharpes: np.ndarray, legacy: dict[str, float], all_legacy: list[float], gates: dict, n_family: int,
              n_raw: int) -> dict:
    """Gate DSR at the FAMILY N with the length-adjusted var_sr floored at the grid variance, the raw-count DSR beside it, and the sensitivities."""
    sr, skew, kurt, n_obs = SS.moments(sel_ret.to_numpy(float))
    fam = list(legacy.values())
    srs = np.array([*fam, *grid_sharpes])
    n_each = [n_obs] * len(srs)                       # legacy trial lengths are not recorded: assume the candidate's (the conservative choice)
    var_adj = GR.gate_var_sr(srs, n_each, n_obs, grid_sharpes=grid_sharpes)
    var_unadj = float(np.var(srs, ddof=1))
    var_all = float(np.var(np.array([*all_legacy, *grid_sharpes]), ddof=1)) if len(all_legacy) + len(grid_sharpes) > 1 else var_unadj
    block = {"sharpe": sr, "sharpe_annual": sr * SQRT_PPY, "skew": skew, "kurt": kurt, "n_obs": n_obs, "n_gate": int(n_family), "n_raw": int(n_raw),
             "var_sr": var_adj, "var_sr_unadjusted": var_unadj, "var_sr_all_family": var_all, "var_sr_members": len(srs),
             "var_sr_source": ("length-adjusted over " + ", ".join(gates["var_sr_family"]) + "; legacy trial lengths assumed equal to the candidate's; "
                               "floored at the grid variance"),
             "dsr": dsr_or_none(sr, n_obs, skew, kurt, n_family, var_adj), "dsr_raw_n": dsr_or_none(sr, n_obs, skew, kurt, n_raw, var_adj),
             "dsr_unadjusted_var": dsr_or_none(sr, n_obs, skew, kurt, n_family, var_unadj),
             "dsr_all_family_var": dsr_or_none(sr, n_obs, skew, kurt, n_family, var_all)}
    for name, n in N_RAW_SENSITIVITIES.items():
        block[f"dsr_{name}"] = dsr_or_none(sr, n_obs, skew, kurt, n, var_adj)
    return block


def deflated_sharpe_annual(sr_pp: float, n_trials: int, var_sr: float) -> float:
    """(per-period SR - E[max SR over N null trials]) x sqrt(252), floored at 0: the 'deflated Sharpe' fed to the Kelly bound."""
    try:
        return max(0.0, (sr_pp - SS.expected_max_sr(int(n_trials), float(var_sr))) * SQRT_PPY)
    except SS.DegenerateInputError:
        return 0.0


# ---------------------------------------------------------------------------------------------------------------------
# cost stress, robustness
# ---------------------------------------------------------------------------------------------------------------------
def run_cost_stress(sel_cfg: dict, panel, constants, base, ledger_ctx, one_x: P.RunResult, dsr_block: dict, mult=(1, 2, 3),
                    with_flat_bps: bool = True) -> tuple[dict, dict[str, P.RunResult]]:
    """Net Sharpe and DSR at 1x/2x/3x (full CostBreakdown scaling by the engine's stress_multiplier); 1x is the grid run. A flat-bps row is
    logged as ``kind='comparison'`` and never enters a gate."""
    params, ids = P.with_config(base, sel_cfg, pre.SPEED_SUBSETS)
    runs = {"1": one_x}
    out: dict = {"net_sharpe": {}, "dsr": {}, "n_obs": dsr_block["n_obs"]}
    for m in mult:
        key = str(int(m))
        if m != 1:
            res = P.run_config(panel, constants, params, subset_ids=ids, ledger_mode=ledger_ctx.mode, label=f"cost_stress:{key}x",
                               stress=float(m), base=base, prereg=ledger_ctx.prereg, snapshot_id=ledger_ctx.snapshot_id, seed=ledger_ctx.seed)
            with P.logged_trial(ledger_ctx, "cost_stress", "stress", f"{key}x", params=sel_cfg, detail={"multiplier": m},
                                returns=res.engine.excess_returns):
                pass
            runs[key] = res
        r = runs[key].engine.excess_returns.loc[one_x.engine.excess_returns.index]
        sr, skew, kurt, n_obs = SS.moments(r.loc[dsr_block["_index"]].to_numpy(float))
        out["net_sharpe"][key] = sr
        out["dsr"][key] = dsr_or_none(sr, n_obs, skew, kurt, dsr_block["n_gate"], dsr_block["var_sr"])
    if with_flat_bps:
        flat = P.run_config(panel, constants, params, subset_ids=ids, ledger_mode=ledger_ctx.mode, label="comparison:flat_bps", base=base,
                            flat_bps=P.FLAT_BPS_PER_SIDE, prereg=ledger_ctx.prereg, snapshot_id=ledger_ctx.snapshot_id, seed=ledger_ctx.seed)
        with P.logged_trial(ledger_ctx, "comparison", "flat_bps", "legacy", params=sel_cfg, detail={"bps_per_side": P.FLAT_BPS_PER_SIDE},
                            returns=flat.engine.excess_returns):
            pass
        out["flat_bps_net_sharpe_comparison_only"] = float(SS.moments(flat.engine.excess_returns.loc[dsr_block["_index"]].to_numpy(float))[0])
    return out, runs


def run_robustness(sel_cfg: dict, panel, constants, consts_raw: dict, base, ledger_ctx, gates: dict, chosen_sr: float, one_x: P.RunResult,
                   eval_index) -> dict:
    """+/-25% on every numeric core parameter, one at a time and one-sided. Every perturbation re-estimates ALL pooled scalars on the same window
    with the perturbed definitions (so mean |f| = 10 holds before the Sharpe is computed); the speed filter, FDM and IDM stay fixed unless the
    perturbed parameter is one of their own inputs (``speed_cost_max_fraction``, ``fdm_cap``, ``idm_cap``)."""
    sel_params, ids = P.with_config(base, sel_cfg, pre.SPEED_SUBSETS)
    names = P.perturbable_names(gates, frozen=pre.FROZEN_LIMITS)
    win = constants.window
    held_fdm = {s: P.fdm_for(constants, s, ids, sel_params, capped=False)[1] for s in panel.symbols}
    out = []
    for name in names["assessed"]:
        for factor, label in ((0.75, "-25%"), (1.25, "+25%")):
            pert = P.perturb(sel_params, name, factor)
            defs = P.rule_defs(base, pert)
            raw = P.raw_forecasts(panel, pert, defs)
            sc = P.estimate_pooled_scalars({r: {s: raw[r][s] for s in panel.symbols} for r in raw}, win)
            cb, fixed = constants, held_fdm
            if name == "speed_cost_max_fraction":
                sf = consts_raw["speed_filter"]
                surv = P.apply_cost_speed_filter(sf["turnover"], sf["cost_per_trade"], sf["sigma_pct"], consts_raw["expected_rule_sharpe"],
                                                 pert.speed_cost_max_fraction)
                cb, fixed = dataclasses.replace(constants, survivors=surv), None     # survivors changed: FDM recomputed from the stored rho
            if name == "idm_cap":
                cb = dataclasses.replace(constants, idm=min(float(consts_raw["idm_uncapped"]), pert.idm_cap))
            res = P.run_config(panel, cb, pert, subset_ids=ids, ledger_mode=ledger_ctx.mode, label=f"robustness:{name}:{label}", base=base,
                               scalars=sc, fdm_fixed=fixed, prereg=ledger_ctx.prereg, snapshot_id=ledger_ctx.snapshot_id, seed=ledger_ctx.seed)
            r = res.engine.excess_returns.loc[eval_index]
            with P.logged_trial(ledger_ctx, "robustness", "perturb", f"{name}|{label}", params=sel_cfg,
                                detail={"parameter": name, "factor": factor}, returns=res.engine.excess_returns):
                pass
            out.append({"param": name, "direction": label, "net_sharpe": float(SS.moments(r.to_numpy(float))[0]),
                        "mean_abs_forecast_after_reestimation": float(np.mean([
                            np.mean(np.abs(raw[q].loc[win[0]:win[1]].stack().dropna().to_numpy()) * sc[q]) for q in sc])),
                        "gross_cap_bound_share": res.gross_cap_bound_share})
    return {"chosen_net_sharpe": chosen_sr, "perturbations": out, "unassessed": names["unassessed"],
            "gross_cap_bound_share": one_x.gross_cap_bound_share, "n_perturbation_rows": len(out)}


# ---------------------------------------------------------------------------------------------------------------------
# stress suite and reference drawdown
# ---------------------------------------------------------------------------------------------------------------------
def run_stress(sel: P.RunResult, panel, vol, gates: dict, charter_proc: dict, tau: float, ledger_ctx, periods_path: Path, eval_index,
               draws: int, seed: int) -> dict:
    ret = sel.engine.excess_returns.loc[eval_index]
    mult = float(gates["g_research"]["stress"]["max_loss_multiple_of_charter_max_dd"])
    min_frac = getattr(pre, "MIN_ACTIVE_FRACTION", None)    # frozen in the re-frozen prereg module; None only if an old module is used (gate 5 insufficient)
    cache: dict[int, float] = {}

    def ref(n: int) -> float:
        if n not in cache:
            cache[n] = P.bootstrap_max_dd_p95(ret.to_numpy(float), path_len=int(n), draws=draws, seed=seed)
        return cache[n]

    pos = sel.engine.positions.where(vol.notna())
    periods = SU.load_stress_periods(str(periods_path))
    res = SU.run_stress_suite(ret, pos, periods, tau, ref, breach_multiple=mult, min_active_fraction=0.0 if min_frac is None else float(min_frac))
    eps = []
    for e in res:
        rdd = ref(e.n_days) if e.n_days >= 5 else None
        d = {"name": e.name, "status": e.status, "max_drawdown": None if math.isnan(e.max_drawdown) else e.max_drawdown, "reference_max_dd": rdd,
             "n_days": e.n_days, "n_active_instruments": e.n_active_instruments, "breach": e.breach, "total_return": e.total_return}
        eps.append(d)
        with P.logged_trial(ledger_ctx, "stress_suite", "episode", e.name, params=ledger_ctx.default_params, detail=d):
            pass
    survival = P.bootstrap_max_dd_p95(ret.to_numpy(float), path_len=int(charter_proc["path_length_days"]), draws=draws, seed=seed)
    return {"min_active_fraction": min_frac, "episodes": eps, "summary": SU.suite_summary(res), "survival_reference_p95_2520d": survival,
            "survival_ref_max_with_2p5_tau": max(survival, 2.5 * tau), "bootstrap_draws": draws, "bootstrap_seed": seed}


# ---------------------------------------------------------------------------------------------------------------------
# benchmark (after tax), power analysis, H4
# ---------------------------------------------------------------------------------------------------------------------
def no_cpi_config(cfg: AT.TaxConfig) -> AT.TaxConfig:
    """Israeli CPI is unavailable in the research allow-list: the config's documented sensitivity ``inflation_adjust: false`` (overstates tax)."""
    a = {**cfg.assumptions, "inflation_adjust": "False (FORCED: Israeli CPI unavailable; documented sensitivity; overstates tax)"}
    return dataclasses.replace(cfg, inflation_adjust=False, assumptions=a)


def candidate_navs(sel: P.RunResult, panel, cfg: AT.TaxConfig, eval_index) -> dict[str, pd.Series]:
    """Candidate NAV (ILS) in the three states (gross, after cost, after tax) through the same tax model as the benchmark."""
    syms = panel.symbols
    costs = None
    tr = sel.engine.trades
    if len(tr):
        costs = tr.pivot_table(index="date", columns="symbol", values="total", aggfunc="sum").reindex(index=sel.engine.positions.index, columns=syms).fillna(0.0)
    trades = P.tax_model_trades(sel.engine.positions, panel.close, panel.raw_close, panel.dividends, syms, cost_usd=costs)
    trades = trades[trades["date"].isin(eval_index) | (trades["date"] < eval_index[0])]
    cpi = pd.Series(1.0, index=panel.close.index)
    dom = dict.fromkeys(syms, "US")
    div = panel.dividends if panel.dividends is not None else pd.DataFrame({"date": [], "symbol": [], "amount": []})
    common = {"prices_usd": panel.raw_close, "fx_usdils": panel.fx, "cpi": cpi, "dividends": div, "domicile": dom, "cfg": cfg,
              "initial_cash_usd": P.INITIAL_CAPITAL}
    gross = AT.apply_tax(trades.assign(cost_usd=0.0), taxes=False, **common)
    after_cost = AT.apply_tax(trades, taxes=False, **common)
    after_tax = AT.apply_tax(trades, taxes=True, **common)
    return {"gross": gross["nav_after_tax_ils"], "after_cost": after_cost["nav_after_tax_ils"], "after_tax": after_tax["nav_after_tax_ils"]}


def run_benchmark(sel: P.RunResult, panel, constants, base, gates: dict, ledger_ctx, tax_cfg: AT.TaxConfig, cost_cfg: dict, eval_index,
                  n_boot: int, seed: int, rationale_claimed: bool) -> dict:
    cfg = no_cpi_config(tax_cfg)
    start = eval_index[0]
    px = panel.raw_close.loc[start:, ["SPY", "IEF"]]
    div = panel.dividends if panel.dividends is not None else pd.DataFrame({"date": [], "symbol": [], "amount": []})
    div = div[pd.to_datetime(div["date"]) >= start]
    cpi = pd.Series(1.0, index=px.index)
    variant, gate_frame, both = AT.gate_benchmark_bm2(px, panel.fx, cpi, div, cfg, cost_cfg, gates=gates, cost_spec=P.COST_SPEC,
                                                      initial_usd=P.INITIAL_CAPITAL)
    sens_name = pre.BENCHMARK_SENSITIVITY_VARIANT
    nav_c = {k: v.loc[start:] for k, v in candidate_navs(sel, panel, cfg, eval_index).items()}
    bench = {s: gate_frame[s] for s in AT.STATES}
    cmp_gate = AT.compare(nav_c, bench, n_boot=n_boot, seed=seed)
    cmp_sens = AT.compare(nav_c, {s: both[sens_name][s] for s in AT.STATES}, n_boot=n_boot, seed=seed)
    for name, frame, kind_key in ((variant, both[variant], "bm2_gate"), (sens_name, both[sens_name], "bm2_sensitivity")):
        r = frame["after_tax"].pct_change().dropna()
        with P.logged_trial(ledger_ctx, "benchmark", "bm2", f"{name}", params=ledger_ctx.default_params,
                            detail={"variant": name, "role": kind_key, "after_tax_sharpe": AT.summarise(frame["after_tax"]).sharpe}, returns=r):
            pass
    # core_only_100 re-run through the engine (parity): 60/40 SPY/IEF at 100% of capital, monthly plus 2-point drift band
    ptx = panel.close.loc[start:, ["SPY", "IEF"]]
    tg = P.core_only_100_targets(ptx, P.INITIAL_CAPITAL)
    eng = P.VE.run_vector_backtest(
        ptx, tg, {"SPY": 1.0, "IEF": 1.0}, P.make_cost_fn(panel, None), P.VE.EngineConfig(initial_capital=P.INITIAL_CAPITAL, allow_short=False),
        adv=panel.adv.loc[start:, ["SPY", "IEF"]], vol_pct=(panel.ret[["SPY", "IEF"]].rolling(60, min_periods=20).std().bfill().loc[start:].fillna(0.01)),
        ledger_ctx=P.VE.LedgerContext(family="core_v1_engine", mode="exploratory", data_snapshot_id=ledger_ctx.snapshot_id, seed=ledger_ctx.seed,
                                      config={"label": "core_only_100", "outer_mode": ledger_ctx.mode}))
    with P.logged_trial(ledger_ctx, "benchmark", "core_only_100", "engine", params=ledger_ctx.default_params, detail={"band_abs": P.CORE_ONLY_BAND_ABS},
                        returns=eng.excess_returns):
        pass
    cand_ret = nav_c["after_tax"].pct_change().dropna()
    bm_ret = bench["after_tax"].pct_change().dropna()
    j = pd.concat([cand_ret, bm_ret], axis=1, join="inner").dropna()
    corr = float(j.corr().iloc[0, 1])
    d = cmp_gate["diff"]["after_tax"]["sharpe"]
    se = (d["ci_high"] - d["ci_low"]) / (2 * 1.959964)
    z = SS._norm_ppf(1 - 0.05 / 2) + SS._norm_ppf(0.80)
    mde = float(z * se)
    sc, sb = cmp_gate["system"]["after_tax"].sharpe, cmp_gate["bench"]["after_tax"].sharpe
    return {"sharpe_candidate": float(sc), "sharpe_bm2": float(sb), "correlation": corr, "margin": mde, "rationale_claimed": bool(rationale_claimed),
            "gate_variant": variant, "sensitivity_variant": sens_name, "gap_ci": d, "gap_standard_error": se, "alpha": 0.05, "power": 0.80,
            "minimum_detectable_sharpe_gap": mde, "underpowered": bool(mde > abs(sc - sb)),
            "sensitivity_monthly_bm2": {"sharpe_bm2": float(cmp_sens["bench"]["after_tax"].sharpe), "gap": cmp_sens["diff"]["after_tax"]["sharpe"]},
            "all_states_gate_variant": {s: {"candidate": dataclasses.asdict(cmp_gate["system"][s]), "bm2": dataclasses.asdict(cmp_gate["bench"][s])}
                                        for s in AT.STATES},
            "core_only_100": {"net_sharpe_annual_pre_tax": float(SS.moments(eng.excess_returns.to_numpy(float))[0] * SQRT_PPY),
                              "n_trades": int(eng.meta["n_trades"])},
            "tax_assumptions": AT.describe_assumptions(cfg).splitlines(), "block_length": cmp_gate["block_length"], "n_boot": n_boot, "seed": seed}


def run_h4(sel: P.RunResult, panel, asset_classes: dict[str, list[str]], bm2_ret: pd.Series, stress_periods, gates: dict, trial_ids: list[str],
           eval_index) -> dict:
    """H4: asset-class P&L contributions of the selected config -> weight-dependent ENB (Meucci) vs the ETF minimum in gates."""
    pos, px = sel.engine.positions, panel.close
    eq = P.INITIAL_CAPITAL * (1.0 + sel.engine.returns).cumprod().shift(1).fillna(P.INITIAL_CAPITAL)
    contrib = (pos.shift(1) * px.diff()).div(eq, axis=0)
    ac = pd.DataFrame({c: contrib[[s for s in m if s in contrib.columns]].sum(axis=1) for c, m in asset_classes.items()
                       if any(s in contrib.columns for s in m)}).loc[eval_index]
    fam = pd.concat({"core_v1": sel.engine.excess_returns, "bm2_annual_after_cost_passive": bm2_ret}, axis=1).reindex(eval_index)
    mask = pd.Series(False, index=eval_index)          # one mask over the evaluation index, shared by the family and asset-class frames
    for p in stress_periods:
        mask |= (eval_index >= pd.Timestamp(p.start)) & (eval_index <= pd.Timestamp(p.end))
    rep = DR.build_report(fam, ac, trial_ids, mask, gates, instrument_type="etf")
    return {"enb_asset_class": float(rep.enb_asset_class), "enb_threshold": rep.enb_threshold, "enb_ok": bool(rep.enb_pass),
            "enb_family_diagnostic": float(rep.enb_family), "participation_ratio": float(rep.participation_ratio),
            "diversification_ratio": float(rep.diversification_ratio), "stress_corr_flags_asset_class": [list(f) for f in rep.stress_corr_flags_asset_class],
            "n_asset_classes": int(ac.shape[1]), "rolling_window": rep.rolling_window,
            "note": "family streams are core_v1 versus the passive BM2 (one active family only); the gate quantity is the asset-class ENB"}


# ---------------------------------------------------------------------------------------------------------------------
# trial history draft (append-only) and the orchestration
# ---------------------------------------------------------------------------------------------------------------------
def append_trial_history(path: Path, entry: dict, family: str = pre.FAMILY) -> None:
    """Append-only: an existing file's entries are re-read and must be unchanged (compared before and after); a new entry is added at the end."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"family": family, "entries": []}
    old = json.dumps(doc["entries"], sort_keys=True)
    doc["entries"].append(entry)
    assert json.dumps(doc["entries"][:-1], sort_keys=True) == old
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(P.jsonable(doc), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run_evaluation(panel: P.Panel, *, consts: dict, gates: dict, ctx: P.LedgerCtx, out_dir: Path, charter: dict, charter_proc: dict, code_commit: str,
                   periods_path: Path, tax_cfg: AT.TaxConfig, cost_cfg: dict, asset_classes: dict[str, list[str]], grid: list[dict] | None = None,
                   draws: int = 10_000, n_boot: int = 10_000, trial_history_path: Path | None = None, ledger_reader=None,
                   prereg_hash: str | None = None, pbo_blocks: int = 16, speed_subsets: dict | None = None) -> dict:
    """Steps 2-10. ``panel`` real (main) or synthetic (tests). Returns the results dict (also written to results.json / REPORT.md)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    constants = load_constants(consts)
    tau = float(consts["tau"])
    base = P.params_from_gates(gates, tau, frozen=pre.FROZEN_LIMITS)     # P4-03 layer on: all 27 gate-6 parameters are perturbable
    grid = grid or build_grid()
    seed = int(ctx.seed or pre.SEED)
    ledger_reader = ledger_reader or L.trials

    # grid, matrix, selection
    M_full, runs = run_grid(grid, panel, constants, base, ctx, speed_subsets)
    vol = P.vol_annual(panel, base)
    eval_start = vol.dropna(how="all").index[0]
    M = M_full.loc[eval_start:]
    eval_index = M.index
    sel_i = select_argmax(M.to_numpy(float))
    sel_cfg, sel_run = grid[sel_i], runs[sel_i]
    sel_ret = sel_run.engine.excess_returns.loc[eval_index]
    grid_sr = sharpe_pp(M.to_numpy(float))

    # gates 1 (DSR) -- family N now, the raw count after every row is logged
    led = ledger_reader()
    legacy, all_legacy = legacy_family_sharpes(led, [n for n in gates["var_sr_family"]])
    n_family = int(pre.FAMILY_N)
    dsr_block = build_dsr(sel_ret, grid_sr, legacy, all_legacy, gates, n_family, n_raw=ET.gate_n(led, pd.DataFrame()).gate_n)
    dsr_block["_index"] = eval_index
    # gate 3 (CPCV), gate 2 (PBO)
    cpcv = run_cpcv(M, select_argmax, pre.CPCV["n_groups"], pre.CPCV["k_test_groups"], embargo_pct=pre.EMBARGO_PCT)
    pbo = run_pbo(M, pbo_blocks, n_family)
    # gate 4 (cost stress; flat bps comparison)
    cost, _ = run_cost_stress(sel_cfg, panel, constants, base, ctx, sel_run, dsr_block)
    # gate 5 (stress suite)
    stress = run_stress(sel_run, panel, vol, gates, charter_proc, tau, ctx, periods_path, eval_index, draws, int(charter_proc.get("seed", seed)))
    # gate 6 (robustness)
    rob = run_robustness(sel_cfg, panel, constants, consts, base, ctx, gates, float(dsr_block["sharpe"]), sel_run, eval_index)
    # gate 7 (benchmark) and H4
    bench = run_benchmark(sel_run, panel, constants, base, gates, ctx, tax_cfg, cost_cfg, eval_index, n_boot, seed, rationale_claimed=False)
    bm2_ret = _bm2_returns(panel, gates, tax_cfg, cost_cfg, eval_index)
    h4 = run_h4(sel_run, panel, asset_classes, bm2_ret, SU.load_stress_periods(str(periods_path)), gates,
                [t for t in ctx.trial_ids.values() if t][:1] or ["n/a"], eval_index)

    # raw N at the end of the run (every row so far), sensitivities
    led_end = ledger_reader()
    n_raw_end = ET.gate_n(led_end, pd.DataFrame()).gate_n
    dsr_block["n_raw"] = int(n_raw_end)
    dsr_block["dsr_raw_n"] = dsr_or_none(dsr_block["sharpe"], dsr_block["n_obs"], dsr_block["skew"], dsr_block["kurt"], n_raw_end, dsr_block["var_sr"])
    dsr_block.pop("_index")
    mech = {"ok": True, "detail": f"charter {charter.get('path')} approved at commit {charter.get('approved_commit')}; verified before the first core_v1 row"}
    results = {"dsr": dsr_block, "pbo": pbo, "cpcv": cpcv, "cost_stress": cost, "stress": stress, "robustness": rob, "benchmark": bench, "mechanism": mech}
    sizing_layer = {"frozen": {"vol_ewma_span": base.vol_ewma_span, "max_vol_scale": base.max_vol_scale,
                                          "instrument_risk_cap_multiple": base.instrument_risk_cap_multiple,
                                          "class_risk_cap_applied": P.CLASS_RISK_CAP_APPLIED}, "selected_config_diagnostics": sel_run.limits}
    outcomes = GR.evaluate_g_research(results, gates)
    kelly = GR.evaluate_kelly_bound({"tau": tau, "deflated_sharpe_annual": deflated_sharpe_annual(dsr_block["sharpe"], n_family, dsr_block["var_sr"])}, gates)
    outcomes.append(kelly)
    tier = GR.combine_tier(outcomes, h4["enb_ok"])
    reason = GR.tier_reason(outcomes, h4["enb_ok"])
    all_ids = sorted({t for t in ctx.trial_ids.values() if t})
    GR.verify_provenance({"trial_ids": all_ids}, ledger_reader(family=ctx.family) if _accepts_family(ledger_reader) else ledger_reader())
    meta = {
        "data_snapshot_id": panel.snapshot_id, "code_commit": code_commit, "prereg_hash": prereg_hash or pre.bars_fingerprint(), "gates_sha256": pre.GATES_SHA256,
        "n_gate": n_family, "n_raw": int(n_raw_end), "var_sr": dsr_block["var_sr"], "var_sr_source": dsr_block["var_sr_source"], "enb": h4["enb_asset_class"],
        "enb_threshold": h4["enb_threshold"], "tier_label": "family-N" if tier == "A" else None, "tier_reason": reason,
        "sensitivities": [f"DSR at family N {n_family}: {dsr_block['dsr']}", f"DSR at raw N {n_raw_end}: {dsr_block['dsr_raw_n']}",
                          f"DSR at N=463: {dsr_block.get('dsr_raw_N_463')}", f"DSR at N=210: {dsr_block.get('dsr_ledgered_N_210')}",
                          f"DSR with unadjusted var_sr {dsr_block['var_sr_unadjusted']:.3g}: {dsr_block['dsr_unadjusted_var']}",
                          f"DSR with all-family var_sr {dsr_block['var_sr_all_family']:.3g}: {dsr_block['dsr_all_family_var']}",
                          f"monthly BM2 sensitivity: Sharpe gap {bench['sensitivity_monthly_bm2']['gap']}",
                          f"flat-bps comparison net Sharpe (never gates): {cost.get('flat_bps_net_sharpe_comparison_only')}",
                          f"Kelly bound on the raw (undeflated) annual net Sharpe, informational: tau <= {0.5 * 0.5 * max(dsr_block['sharpe_annual'], 0):.4f}"],
        "deviations": ["family membership of N=31 is provisional (gates family_membership_confirmed_by_owner: false)",
                       "legacy trial lengths are unknown: var_sr assumes they equal the candidate's (conservative)",
                       "H4 family streams: one active family, so the family-level ENB is diagnostic only",
                       "robustness: all 27 parameters are perturbed (P4-03 layer wired in; a parameter that cannot be perturbed would be listed unassessed, gate 6 insufficient)",
                       "P4-03 sizing layer: vol scale, instrument risk cap and gross cap; the 0.40 class cap is not applied (not a gate-6 parameter)",
                       "rf is None throughout (Sharpe versus zero), as in P2-05"],
        "known_leakage": KNOWN_LEAKAGE,
    }
    comparators = {"BM2 annual (gate, after tax, annualised)": {"sharpe": bench["sharpe_bm2"]},
                   f"BM2 {bench['sensitivity_variant']} (sensitivity)": {"sharpe": bench["sensitivity_monthly_bm2"]["sharpe_bm2"]},
                   "core_only_100 (engine, pre-tax, annualised)": {"sharpe": bench["core_only_100"]["net_sharpe_annual_pre_tax"]}}
    md = GR.render_report(outcomes, tier, meta, comparators)
    out = {"tier": tier, "tier_reason": reason, "outcomes": [dataclasses.asdict(o) for o in outcomes], "results": results, "h4": h4, "meta": meta,
           "selected_config": sel_cfg, "selected_index": sel_i, "sizing_layer": sizing_layer, "grid_sharpes_per_period": [float(x) for x in grid_sr],
           "trial_ids": ctx.trial_ids, "ledger_rows_registered": len(ctx.trial_ids),
           "expected_ledger_rows": {"grid": pre.EXPECTED_GRID_ROWS, **pre.EXPECTED_LEDGER_ROWS}}
    P.dump_json(out, out_dir / "results.json")
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    if trial_history_path is not None:
        append_trial_history(trial_history_path, {
            "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"), "fingerprint": pre.bars_fingerprint(), "n_trials": len(grid),
            "trials": [json.dumps(g, sort_keys=True) for g in grid], "trial_daily_sharpes": [float(x) for x in grid_sr], "tier": tier,
            "selected_index": sel_i, "ledger_trial_ids": {k: v for k, v in ctx.trial_ids.items() if k.startswith("grid:")}})
    return out


def _accepts_family(fn) -> bool:
    import inspect

    return "family" in inspect.signature(fn).parameters


def _bm2_returns(panel, gates, tax_cfg, cost_cfg, eval_index) -> pd.Series:
    cfg = no_cpi_config(tax_cfg)
    start = eval_index[0]
    px = panel.raw_close.loc[start:, ["SPY", "IEF"]]
    div = panel.dividends if panel.dividends is not None else pd.DataFrame({"date": [], "symbol": [], "amount": []})
    div = div[pd.to_datetime(div["date"]) >= start]
    _, frame, _ = AT.gate_benchmark_bm2(px, panel.fx, pd.Series(1.0, index=px.index), div, cfg, cost_cfg, gates=gates, cost_spec=P.COST_SPEC,
                                        initial_usd=P.INITIAL_CAPITAL)
    return frame["after_cost"].pct_change().dropna()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / OUT_SUBDIR)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    G.seal_preflight(ROOT, euid=lambda: os.geteuid(), access=lambda p, m: os.access(p, m))
    G.assert_in_window(clock=lambda: G.current_utc())
    G.require_memory_cap()
    if "preregistration" in a.out.resolve().parts or "integrity" in a.out.resolve().parts:
        raise G.PreflightError("--out must not be under research/preregistration/ or tests/integrity/")
    ver = verify_start(ROOT)
    from firm.data.etf_loader import load_dividends, load_etf_universe, usd_ils
    from firm.research import charter as CH

    asof = seal.max_research_date()
    seal.check_asof(asof, what="run_core_v1_evaluation")
    series = load_etf_universe(ROOT / "config" / "universe_etf.yaml", asof=asof, include_delisted=False)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    if manifest["snapshot_id"] != ver["constants"]["data_snapshot_id"]:
        raise StartupError("the manifest snapshot id differs from the one the constants were estimated on")
    panel = P.panel_from_series({s: series[s] for s in pre.UNIVERSE}, snapshot_id=manifest["snapshot_id"],
                                dividends=load_dividends(pre.UNIVERSE, asof=asof), fx=usd_ils(asof, source="eodhd"),
                                source="eodhd etfs_full via firm.data.etf_loader (cleaning v3)")
    fm, _ = CH.parse_charter((ROOT / pre.CHARTER_PATH).read_text(encoding="utf-8"))
    ctx = C1.build_ledger_ctx(panel.snapshot_id, pre.bars_fingerprint(), pre.SEED)
    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    out_dir = a.out
    run_evaluation(panel, consts=ver["constants"], gates=gates, ctx=ctx, out_dir=out_dir, charter=ver["charter"],
                   charter_proc=fm["max_dd_procedure"] | {"seed": fm["max_dd_procedure"].get("seed", pre.SEED),
                                                           "path_length_days": fm["max_dd_procedure"]["path_length_days"]},
                   code_commit=C1.git_head(ROOT), periods_path=ROOT / "config" / "stress_periods.yaml",
                   tax_cfg=AT.load_tax_config(ROOT / "config" / "tax_il.yaml"), cost_cfg=P.CM.load_cost_config(), asset_classes=pre.ASSET_CLASSES,
                   draws=int(fm["max_dd_procedure"]["draws"]), trial_history_path=out_dir / "drafts" / "core_v1_trial_history.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
