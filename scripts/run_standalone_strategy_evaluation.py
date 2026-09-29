"""Step 1 of the 2026-09-29 edge search: standalone OOS edge of each current strategy.

Frozen design: ``scripts/standalone_strategy_preregistered_bars.py`` (fingerprinted).

    # the live-config backtest (C0 of the 9/28 design), capturing every strategy's signal book
    FIRM_DATA_DIR=<scratch> python scripts/run_standalone_strategy_evaluation.py run --out-dir <dir>

    # books -> lagged, costed, fold-wise beta-hedged returns -> bars
    python scripts/run_standalone_strategy_evaluation.py evaluate --out-dir <dir> \
        [--report docs/standalone_strategy_evaluation_2026_09.json] [--append-ledger]

``run`` writes ``C0_legacy_optimal.{returns.parquet,meta.json,config.json}`` in the
format ``run_combination_evaluation.py evaluate`` expects, so the same run also
serves the corrected-engine replication of the 9/28 evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import combination_preregistered_bars as comb_prereg  # noqa: E402
import standalone_strategy_preregistered_bars as prereg  # noqa: E402

log = logging.getLogger(__name__)

LEDGER = _ROOT / "docs" / "standalone_strategy_trial_history.json"
CANDIDATE = "C0_legacy_optimal"
ANN = math.sqrt(252)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    import run_combination_evaluation as comb_run
    from firm.backtest.run import execute_backtest
    from firm.portfolio.attribution import PerformanceAttribution as PA

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = comb_run.build_config(CANDIDATE)
    if args.start:
        cfg["start_date"] = args.start
    if args.end:
        cfg["end_date"] = args.end
    (out / f"{CANDIDATE}.config.json").write_text(json.dumps(cfg, indent=2, default=str))

    state: dict = {"date": None, "attr": None}
    rows: list[tuple] = []
    orig_update, orig_record = PA.update_daily, PA.record_signals

    def update_daily(self, date, prices, nav, strategy_holdings=None):
        state["date"], state["attr"] = date, self
        return orig_update(self, date, prices, nav, strategy_holdings)

    def record_signals(self, signals):
        orig_record(self, signals)
        d = state["date"]
        if d is None:
            log.warning("record_signals before any update_daily; book not captured")
            return
        for strat, book in self._signal_books.items():
            for sym, w in book.items():
                rows.append((pd.Timestamp(d), strat, sym, float(w)))
            if not book:
                rows.append((pd.Timestamp(d), strat, "", 0.0))

    PA.update_daily, PA.record_signals = update_daily, record_signals
    log.info("run %s %s -> %s, standalone fp=%s, combination fp=%s", CANDIDATE, cfg["start_date"], cfg["end_date"],
             prereg.bars_fingerprint(), comb_prereg.bars_fingerprint())
    t0 = time.time()
    try:
        report = execute_backtest(cfg)
    finally:
        PA.update_daily, PA.record_signals = orig_update, orig_record
    secs = time.time() - t0

    rets = report.returns.astype(float)
    rets.index = pd.DatetimeIndex(rets.index)
    rets.to_frame("ret").to_parquet(out / f"{CANDIDATE}.returns.parquet")
    books = pd.DataFrame(rows, columns=["date", "strategy", "symbol", "weight"])
    books = books.drop_duplicates(["date", "strategy", "symbol"], keep="last")
    books.to_parquet(out / "signal_books.parquet")
    attr = state["attr"]
    if attr is not None:
        sr = attr.get_all_signal_returns()
        pd.DataFrame(sr).to_parquet(out / "attribution_signal_returns.parquet")
    d = report.to_dict()
    meta = {
        "candidate": CANDIDATE,
        "bars_fingerprint": comb_prereg.bars_fingerprint(),        # for run_combination_evaluation evaluate
        "standalone_fingerprint": prereg.bars_fingerprint(),
        "seconds": round(secs), "n_days": int(len(rets)),
        "n_book_rows": int(len(books)), "strategies_captured": sorted(books["strategy"].unique().tolist()),
        "portfolio": d.get("portfolio", {}), "turnover": d.get("turnover", {}),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / f"{CANDIDATE}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("run done in %.0fs (%d days, %d book rows, strategies %s)", secs, len(rets), len(books),
             meta["strategies_captured"])
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * ANN) if sd > 0 else float("nan")


def _stationary_indices(n: int, n_boot: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.broadcast_to(np.arange(n), (n_boot, n))
    bsp = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    return ((np.take_along_axis(starts, bsp, axis=1) + pos - bsp) % n).astype(np.int32)


def boot_sharpe_lb(x: np.ndarray, seed: int) -> float:
    cfg = prereg.BOOTSTRAP
    rng = np.random.default_rng(seed)
    out = []
    for k in range(0, cfg["n_boot"], 500):
        idx = _stationary_indices(len(x), min(500, cfg["n_boot"] - k), cfg["mean_block_days"], rng)
        X = x[idx]
        out.append(X.mean(1) / X.std(1, ddof=1) * ANN)
    return float(np.quantile(np.concatenate(out), cfg["alpha_one_sided"]))


def load_prices(dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    from firm.data.cache import ParquetCache
    from firm.config import get_settings
    p = ParquetCache(get_settings().data.cache_dir).get("combined/prices")
    p = p[p.symbol.isin(symbols)]
    px = p.pivot_table(index="date", columns="symbol", values="adj_close")
    px.index = pd.DatetimeIndex(px.index)
    return px.reindex(dates)


def book_matrix(books: pd.DataFrame, strategy: str, dates: pd.DatetimeIndex, symbols: list[str]) -> np.ndarray:
    b = books[(books.strategy == strategy) & (books.symbol != "")]
    W = b.pivot_table(index="date", columns="symbol", values="weight", aggfunc="last")
    recorded = pd.DatetimeIndex(books.loc[books.strategy == strategy, "date"].unique())
    W = W.reindex(recorded).fillna(0.0).reindex(columns=symbols).fillna(0.0)
    # Days with no record keep the last recorded book (the attribution semantics).
    return W.reindex(dates).ffill().fillna(0.0).to_numpy()


def book_returns(W: np.ndarray, R: np.ndarray, lag: int, cost_bps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(net, gross, turnover) daily series. Book W[d] adopted at close d+lag, earns R from d+lag+1."""
    T = len(R)
    Wa = np.zeros_like(W)                      # book adopted at close t
    Wa[lag:] = W[:T - lag] if lag else W
    held = np.vstack([np.zeros((1, W.shape[1])), Wa[:-1]])  # held during day t
    gross = np.nansum(held * np.nan_to_num(R), axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), Wa]), axis=0)).sum(axis=1)
    net = gross - turn * cost_bps / 1e4
    return net, gross, turn


def hedge(net: np.ndarray, spy: np.ndarray, masks_train: list, masks_test: list) -> tuple[np.ndarray, list[float]]:
    out = np.full(len(net), np.nan)
    betas = []
    for tr, te in zip(masks_train, masks_test):
        x, y = spy[tr], net[tr]
        ok = np.isfinite(x) & np.isfinite(y)
        xv = x[ok] - x[ok].mean()
        beta = float((xv * (y[ok] - y[ok].mean())).sum() / (xv**2).sum()) if (xv**2).sum() > 0 else 0.0
        betas.append(beta)
        out[te] = net[te] - beta * np.nan_to_num(spy[te])
    return out, betas


def permute_months(W: np.ndarray, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(W)
    months = dates.to_period("M")
    for m in months.unique():
        rows = months == m
        perm = rng.permutation(W.shape[1])
        out[rows] = W[rows][:, perm]
    return out


def cmd_evaluate(args: argparse.Namespace) -> int:
    from firm.eval.overfitting import deflated_sharpe
    from firm.experiments.runner import ExperimentRunner

    out = Path(args.out_dir)
    meta = json.loads((out / f"{CANDIDATE}.meta.json").read_text())
    fp = prereg.bars_fingerprint()
    if meta.get("standalone_fingerprint") != fp:
        raise SystemExit(f"run was made against a different pre-registration: {meta.get('standalone_fingerprint')}")
    books = pd.read_parquet(out / "signal_books.parquet")
    books["date"] = pd.DatetimeIndex(books["date"])
    port = pd.read_parquet(out / f"{CANDIDATE}.returns.parquet")["ret"]
    dates = pd.DatetimeIndex(port.index)
    symbols = sorted(s for s in books.symbol.unique() if s)
    px = load_prices(dates, sorted(set(symbols) | {"SPY"}))
    R = px[symbols].pct_change(fill_method=None).to_numpy()
    spy = px["SPY"].pct_change(fill_method=None).to_numpy()

    splits = ExperimentRunner._compute_walk_forward_splits(
        prereg.SOURCE_RUN["start"], prereg.SOURCE_RUN["end"], prereg.FOLDS["n_splits"],
        prereg.FOLDS["train_pct"], prereg.FOLDS["embargo_days"])
    m_tr = [np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))) for a, b, _, _ in splits]
    m_te = [np.asarray((dates >= pd.Timestamp(c)) & (dates <= pd.Timestamp(d))) for _, _, c, d in splits]
    oos = np.logical_or.reduce(m_te)
    cost = prereg.COSTS["bps_per_side"]

    # Fidelity: lag-0 gross recompute vs the engine's own attribution series.
    fidelity = {}
    attr_path = out / "attribution_signal_returns.parquet"
    attr = pd.read_parquet(attr_path) if attr_path.exists() else pd.DataFrame()
    if not attr.empty:
        attr.index = pd.DatetimeIndex(attr.index)

    strategies = [s for s in prereg.STRATEGIES if s in set(books.strategy)]
    missing = sorted(set(prereg.STRATEGIES) - set(strategies))
    if missing:
        log.warning("strategies with no captured book (treated as FAIL): %s", missing)
    series, results = {}, {}
    for s in strategies:
        W = book_matrix(books, s, dates, symbols)
        net, gross, turn = book_returns(W, R, prereg.BOOK_RETURNS["primary_lag_days"], cost)
        net0, gross0, _ = book_returns(W, R, prereg.BOOK_RETURNS["secondary_lag_days"], cost)
        hedged, betas = hedge(net, spy, m_tr, m_te)
        hedged0, _ = hedge(net0, spy, m_tr, m_te)
        x = hedged[oos]
        series[s] = x
        folds = [_sharpe(hedged[m]) for m in m_te]
        if s in attr.columns:
            a = attr[s].reindex(dates)
            ok = a.notna().to_numpy() & np.isfinite(gross0)
            fidelity[s] = {"max_abs_diff": float(np.abs(a.to_numpy()[ok] - gross0[ok]).max()) if ok.any() else None,
                           "corr": float(np.corrcoef(a.to_numpy()[ok], gross0[ok])[0, 1]) if ok.sum() > 2 else None}
        results[s] = {
            "oos_days": int(len(x)), "sharpe_oos_hedged_net": _sharpe(x),
            "lb_one_sided": boot_sharpe_lb(x, prereg.BOOTSTRAP["seed"] + hash(s) % 1000),
            "fold_sharpes": folds, "betas": betas,
            "sharpe_oos_hedged_gross": _sharpe(hedge(gross, spy, m_tr, m_te)[0][oos]),
            "sharpe_oos_hedged_net_lag0": _sharpe(hedged0[oos]),
            "sharpe_full_net_unhedged": _sharpe(net), "mean_daily_turnover": float(np.nanmean(turn)),
            "days_with_book": int((np.abs(W).sum(1) > 0).sum()),
        }
        log.info("%-20s OOS hedged net SR %+.2f (LB %+.2f) folds %s gross %+.2f lag0 %+.2f turnover %.3f",
                 s, results[s]["sharpe_oos_hedged_net"], results[s]["lb_one_sided"],
                 [round(f, 2) for f in folds], results[s]["sharpe_oos_hedged_gross"],
                 results[s]["sharpe_oos_hedged_net_lag0"], results[s]["mean_daily_turnover"])

    # Placebos
    rng = np.random.default_rng(prereg.PLACEBO["seed"])
    for s in strategies:
        W = book_matrix(books, s, dates, symbols)
        sh = []
        for _ in range(prereg.PLACEBO["n_draws"]):
            Wp = permute_months(W, dates, rng)
            net, _, _ = book_returns(Wp, R, prereg.BOOK_RETURNS["primary_lag_days"], cost)
            h, _ = hedge(net, spy, m_tr, m_te)
            sh.append(_sharpe(h[oos]))
        results[s]["placebo_p95"] = float(np.nanpercentile(sh, prereg.PLACEBO["pass_percentile"]))
        results[s]["placebo_median"] = float(np.nanmedian(sh))

    trial_daily = np.array([np.nanmean(v) / np.nanstd(v, ddof=1) for v in series.values()])
    for s in strategies:
        r = results[s]
        r["dsr"] = float(deflated_sharpe(series[s][np.isfinite(series[s])], trial_daily))
        r["bars"] = {
            "S1": bool(r["sharpe_oos_hedged_net"] > 0 and r["lb_one_sided"] > 0),
            "S2": bool(r["sharpe_oos_hedged_net"] > r["placebo_p95"]),
            "S3": bool(r["dsr"] > 0.95),
            "S4": bool(sum(f > 0 for f in r["fold_sharpes"]) >= 3),
        }
        r["survives"] = all(r["bars"].values())
        if s in fidelity:
            r["fidelity_vs_attribution_lag0_gross"] = fidelity[s]
        log.info("==> %-20s %s survives=%s (placebo p95 %+.2f, DSR %.3f)", s, r["bars"], r["survives"],
                 r["placebo_p95"], r["dsr"])
    for s in missing:
        results[s] = {"survives": False, "note": "no signal book captured"}

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
              "folds": [list(x) for x in splits], "cost_bps_per_side": cost,
              "alpha_one_sided": prereg.BOOTSTRAP["alpha_one_sided"], "strategies": results,
              "survivors": [s for s, r in results.items() if r.get("survives")]}
    (out / "standalone_report.json").write_text(json.dumps(report, indent=2, default=float))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float))
    if args.append_ledger:
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "standalone_strategy", "entries": []}
        ledger["entries"].append({"date": datetime.now(timezone.utc).date().isoformat(), "fingerprint": fp,
                                  "n_trials": len(prereg.STRATEGIES), "trials": prereg.STRATEGIES,
                                  "trial_daily_sharpes": [float(v) for v in trial_daily],
                                  "survivors": report["survivors"]})
        ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
        LEDGER.write_text(json.dumps(ledger, indent=2))
    log.info("survivors: %s", report["survivors"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--start", help="smoke test only; the real run uses the frozen window")
    r.add_argument("--end")
    e = sub.add_parser("evaluate")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--report")
    e.add_argument("--append-ledger", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"run": cmd_run, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
