"""Pre-registered evaluation of `signal_combination` candidates (2026-09-28).

Frozen design: ``scripts/combination_preregistered_bars.py`` (fingerprinted).
Write-up: ``docs/optimal_combination_fix_2026_09.md``.

Two subcommands:

    # one continuous live-config backtest per candidate (run them in parallel)
    FIRM_DATA_DIR=<scratch> python scripts/run_combination_evaluation.py run \
        --candidate C1_robust_attribution --out-dir <dir>

    # once every candidate's returns exist: slice folds, compute every bar
    python scripts/run_combination_evaluation.py evaluate --out-dir <dir> \
        --report docs/combination_evaluation_2026_09.json [--append-ledger]

Set FIRM_DATA_DIR to a scratch dir for ``run``: ExecutionAgent's audit log
defaults to ``data/execution_audit.jsonl`` -- the live IBKR instance's own
file, which earlier backtests have already polluted.
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
import yaml

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import combination_preregistered_bars as prereg

log = logging.getLogger(__name__)

LEDGER = _ROOT / "docs" / "combination_trial_history.json"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def build_config(candidate: str) -> dict:
    """Flat execute_backtest config mirroring config/live.yaml, overriding only
    ``signal_combination``."""
    from firm.config import get_settings

    live = yaml.safe_load((_ROOT / "config" / "live.yaml").read_text())
    bt = get_settings().backtest.model_dump()
    costs = live.get("costs") or {}
    return {
        **bt,
        **{k: costs[k] for k in (
            "commission_pct", "slippage_pct", "spread_pct", "market_impact_coefficient",
            "rebalance_band_pct", "rebalance_fraction",
        ) if k in costs},
        "start_date": prereg.START_DATE,
        "end_date": prereg.END_DATE,
        "initial_capital": float(live.get("initial_capital", bt["initial_capital"])),
        "rebalance_frequency": live.get("rebalance_frequency", "daily"),
        "strategies": list(live["strategies"]["enabled"]),
        "strategy_params": dict(live.get("strategy_params") or {}),
        **dict(live["risk"]),
        "allocation_method": live.get("allocation_method", "conviction_weighted"),
        "kelly_fraction": live.get("kelly_fraction", 0.5),
        "conviction_smoothing_enabled": bool(live.get("conviction_smoothing_enabled", False)),
        "conviction_smoothing_halflife_days": float(live.get("conviction_smoothing_halflife_days", 3.0)),
        "strategy_circuit_breaker": dict(live.get("strategy_circuit_breaker") or {}),
        "strategy_regime_weights": dict(live.get("strategy_regime_weights") or {}),
        "signal_combination": dict(prereg.CANDIDATES[candidate]),
        "data_source": "cache",
        "universe_symbols": list(live["universe"]["symbols"]),
        "seed": prereg.SEED,
    }


def cmd_run(args: argparse.Namespace) -> int:
    from firm.backtest.run import execute_backtest

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args.candidate)
    (out / f"{args.candidate}.config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log.info("run %s: %s -> %s, fingerprint=%s", args.candidate, cfg["start_date"],
             cfg["end_date"], prereg.bars_fingerprint())
    t0 = time.time()
    report = execute_backtest(cfg)
    secs = time.time() - t0
    rets = report.returns.astype(float)
    rets.index = pd.DatetimeIndex(rets.index)
    rets.to_frame("ret").to_parquet(out / f"{args.candidate}.returns.parquet")
    d = report.to_dict()
    meta = {
        "candidate": args.candidate,
        "bars_fingerprint": prereg.bars_fingerprint(),
        "seconds": round(secs),
        "n_days": len(rets),
        "portfolio": d.get("portfolio", {}),
        "turnover": d.get("turnover", {}),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (out / f"{args.candidate}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("run %s done in %.0fs (%d days)", args.candidate, secs, len(rets))
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(252))


def _max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float(np.max(1.0 - eq / np.maximum.accumulate(eq))) if len(eq) else 0.0


def _stationary_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence of length n."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def paired_sharpe_diff_ci(a: np.ndarray, b: np.ndarray, level: float) -> dict:
    """CI for Sharpe(a) - Sharpe(b) under a paired stationary block bootstrap."""
    cfg = prereg.BOOTSTRAP
    rng = np.random.default_rng(cfg["seed"])
    n = len(a)
    diffs = np.empty(cfg["n_boot"])
    for i in range(cfg["n_boot"]):
        ix = _stationary_indices(n, cfg["mean_block_days"], rng)
        diffs[i] = _sharpe(a[ix]) - _sharpe(b[ix])
    alpha = 1.0 - level
    return {
        "point": _sharpe(a) - _sharpe(b),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_le_0": float((diffs <= 0).mean()),
    }


def _prior_trials() -> int:
    if not LEDGER.exists():
        return 0
    data = json.loads(LEDGER.read_text())
    return int(sum(int(e.get("n_trials", 0)) for e in data.get("entries", [])))


def cmd_evaluate(args: argparse.Namespace) -> int:
    from firm.eval.overfitting import cscv_pbo, deflated_sharpe
    from firm.experiments.runner import ExperimentRunner

    out = Path(args.out_dir)
    fp = prereg.bars_fingerprint()
    series: dict[str, pd.Series] = {}
    for c in prereg.CANDIDATES:
        meta = json.loads((out / f"{c}.meta.json").read_text())
        if meta["bars_fingerprint"] != fp:
            raise SystemExit(f"{c} was run against a different pre-registration ({meta['bars_fingerprint']})")
        series[c] = pd.read_parquet(out / f"{c}.returns.parquet")["ret"]
    full = pd.DataFrame(series).dropna()

    splits = ExperimentRunner._compute_walk_forward_splits(
        prereg.START_DATE, prereg.END_DATE, prereg.N_SPLITS, prereg.TRAIN_PCT, prereg.EMBARGO_DAYS,
    )
    fold_masks = [
        (full.index >= pd.Timestamp(ts)) & (full.index <= pd.Timestamp(te))
        for _, _, ts, te in splits
    ]
    oos_mask = np.logical_or.reduce(fold_masks)
    oos = full[oos_mask]

    per_fold = []
    for (trs, tre, ts, te), m in zip(splits, fold_masks):
        f = full[m]
        per_fold.append({
            "train": [trs, tre], "test": [ts, te], "n_days": len(f),
            "sharpe": {c: _sharpe(f[c].to_numpy()) for c in full},
            "total_return": {c: float(np.prod(1 + f[c].to_numpy()) - 1) for c in full},
            "max_dd": {c: _max_dd(f[c].to_numpy()) for c in full},
        })

    trial_sharpes = np.array([_sharpe(oos[c].to_numpy()) / math.sqrt(252) for c in full])
    prior = _prior_trials()
    pbo = float(cscv_pbo(full.to_numpy(), n_partitions=8))
    base = oos[prereg.BASELINE].to_numpy()
    plc = oos[prereg.PLACEBO].to_numpy()

    results = {}
    for c in prereg.PRIMARY:
        r = oos[c].to_numpy()
        vs_base = paired_sharpe_diff_ci(r, base, prereg.CI_LEVEL)
        vs_plc = paired_sharpe_diff_ci(r, plc, prereg.CI_LEVEL)
        dsr = float(deflated_sharpe(r, trial_sharpes, prior_trials=prior))
        folds_won = sum(pf["sharpe"][c] > pf["sharpe"][prereg.BASELINE] for pf in per_fold)
        dd_c, dd_b = _max_dd(r), _max_dd(base)
        bars = {
            "B1_beats_baseline": vs_base["point"] > 0 and vs_base["lo"] > 0,
            "B2_beats_placebo": vs_plc["point"] > 0 and vs_plc["lo"] > 0,
            "B3_deflated_sharpe": dsr > 0.95,
            "B4_fold_consistency": folds_won >= 3,
            "B5_pbo": pbo < 0.50,
            "B6_drawdown": dd_c <= 1.25 * dd_b,
        }
        results[c] = {
            "oos_sharpe": _sharpe(r), "vs_baseline": vs_base, "vs_placebo": vs_plc,
            "dsr": dsr, "folds_beating_baseline": int(folds_won),
            "oos_max_dd": dd_c, "baseline_oos_max_dd": dd_b,
            "bars": bars, "n_bars_passed": int(sum(bars.values())),
            "verdict": "PASS" if all(bars.values()) else "FAIL",
        }

    secondary = {
        "C0_vs_C3_confidence": paired_sharpe_diff_ci(base, oos["C3_confidence"].to_numpy(), prereg.CI_LEVEL),
        "C0_vs_placebo": paired_sharpe_diff_ci(base, plc, prereg.CI_LEVEL),
        "C3_vs_placebo": paired_sharpe_diff_ci(oos["C3_confidence"].to_numpy(), plc, prereg.CI_LEVEL),
        "oos_sharpe": {c: _sharpe(oos[c].to_numpy()) for c in full},
        "full_period_sharpe": {c: _sharpe(full[c].to_numpy()) for c in full},
        "oos_total_return": {c: float(np.prod(1 + oos[c].to_numpy()) - 1) for c in full},
        "turnover": {
            c: json.loads((out / f"{c}.meta.json").read_text()).get("turnover", {}) for c in full
        },
    }
    report = {
        "bars_fingerprint": fp,
        "preregistered_at": prereg.PREREGISTERED_AT,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "n_full_days": len(full), "n_oos_days": len(oos),
        "prior_trials_from_ledger": prior, "this_run_trials": len(full.columns),
        "pbo": pbo, "per_fold": per_fold, "primary": results, "secondary": secondary,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({c: {k: results[c][k] for k in ("verdict", "n_bars_passed", "oos_sharpe", "dsr")}
                      for c in results}, indent=2, default=float))

    if args.append_ledger:
        data = json.loads(LEDGER.read_text())
        data["entries"].append({
            "date": datetime.now(UTC).date().isoformat(),
            "source": "scripts/run_combination_evaluation.py (" + fp[:12] + ")",
            "n_trials": len(full.columns),
            "candidates": list(full.columns),
            "oos_sharpes": secondary["oos_sharpe"],
            "verdicts": {c: results[c]["verdict"] for c in results},
        })
        LEDGER.write_text(json.dumps(data, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.setLevel(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidate", required=True, choices=list(prereg.CANDIDATES))
    r.add_argument("--out-dir", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--report", required=True)
    e.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
