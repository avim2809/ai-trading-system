"""FROZEN pre-registration for Step 1 of the 2026-09-29 edge search: standalone
out-of-sample edge of each current strategy, plus a corrected-engine
replication of the 2026-09-28 signal_combination evaluation.

Plan: docs/edge_search_plan_2026_09.md §4. Written and fingerprinted BEFORE the
backtest that produces the signal books ran. Do not edit after results exist.

Question 1 (primary): does any of the 11 enabled strategies' own standalone
signal book earn a positive, beta-hedged, net-of-cost Sharpe out of sample,
beating a symbol-permutation placebo, after deflating for 11 trials?

Question 2 (replication, found 2026-09-29): the backtest broker marked
positions at raw, split-unadjusted closes, so stock splits booked fake P&L
(e.g. -6.96% for C3 on 2024-06-10, the NVDA 10:1 split), and several split days
fall inside the 9/28 OOS windows. Does the frozen 9/28 design
(scripts/combination_preregistered_bars.py, fingerprint 96bf26c8...) reach the
same verdict once the feed is total-return adjusted? Nothing about that design
changes except the engine fix (src/firm/backtest/datafeeds.py
_total_return_adjust). Its own bars and evaluator are reused as-is.

Tuning disclosure: config/live.yaml's strategy_params override only stat_arb
(coint_pvalue 0.10, entry_z 2.0, exit_z 0.5, lookback 60, six predefined
pairs) and pattern_recognition.cnn_scoring_enabled=false. The block was
introduced 2026-07-23 (6fea380) and has not been re-tuned since. Every other
strategy uses its code defaults, whose selection history is undocumented, so
"out of sample" here means with respect to this evaluation's choices, not
provably with respect to the defaults' authors. pattern_recognition was
grid-searched over 2018-2025 in earlier sessions (docs/pattern_ml_trial_history.json,
104 trials), so a pattern_recognition pass would be treated as in-sample.
"""

from __future__ import annotations

import hashlib
import json

PREREGISTERED_AT = "2026-09-29T11:35:00Z"

# Same backtest and fold design as the 9/28 evaluation, so windows are comparable.
SOURCE_RUN = {
    "config": "run_combination_evaluation.build_config('C0_legacy_optimal') == config/live.yaml (IBKR, blended) incl. "
              "signal_combination {method: optimal}; data_source cache; seed 42",
    "start": "2020-01-01", "end": "2026-06-30",
    "engine": "worktree with the total-return-adjusted backtest feed",
    "capture": "PerformanceAttribution._signal_books after every record_signals call, keyed by the date of the "
               "preceding update_daily (the backtest bar date); last capture per date wins",
}
FOLDS = {"n_splits": 4, "train_pct": 0.7, "embargo_days": 1,
         "source": "ExperimentRunner._compute_walk_forward_splits(start, end, 4, 0.7, 1)"}
STRATEGIES = ["momentum", "trend", "mean_reversion", "volatility_breakout", "stat_arb", "seasonality",
              "multi_factor", "event_driven", "sentiment", "regime_hmm", "pattern_recognition"]

BOOK_RETURNS = {
    "prices": "cache combined/prices adj_close (total return), close-to-close",
    "primary_lag_days": 1,   # book formed at close d is adopted at close d+1, first earns close(d+1)->close(d+2)
    "secondary_lag_days": 0, # attribution's own convention: earns close(d)->close(d+1)
    "missing_price": "a symbol with no return that day contributes 0",
}
COSTS = {
    "bps_per_side": 12.0,  # config/live.yaml costs: commission 5 + slippage 5 + spread 2 bps; no impact (unit book)
    "turnover": "sum |w_adopted(t) - w_adopted(t-1)| per adoption day, charged that day",
}
HEDGE = {
    "rule": "per fold: OLS slope beta of the net book return on SPY adj-close return over the fold's TRAIN window; "
            "test-window hedged return = net - beta * r_SPY (beta never uses test data); no hedge cost",
}
SHARPE = "annualised mean/std(ddof=1) x sqrt(252) of the hedged net daily return (a self-financing book, no rf)"
BOOTSTRAP = {"method": "stationary_block", "mean_block_days": 21, "n_boot": 5000, "seed": 20260929,
             "alpha_one_sided": 0.05 / len(STRATEGIES)}
PLACEBO = {"n_draws": 500, "seed": 20260930, "pass_percentile": 95.0,
           "rule": "each calendar month, one random permutation of the universe's symbol columns is applied to the "
                   "strategy's book weights for that month; lag, costs and fold-wise beta hedge recomputed identically"}
BARS = [
    {"id": "S1", "rule": "concatenated-OOS hedged net Sharpe > 0 AND bootstrap one-sided lower bound at alpha > 0"},
    {"id": "S2", "rule": "concatenated-OOS hedged net Sharpe > 95th percentile of its placebo Sharpes"},
    {"id": "S3", "rule": "DSR > 0.95; trial_sharpes = daily OOS Sharpes of all 11 strategies; new ledger "
                         "docs/standalone_strategy_trial_history.json"},
    {"id": "S4", "rule": "hedged net Sharpe > 0 in >= 3 of 4 OOS test windows"},
]
REPORTED = ["full-period hedged net Sharpe", "lag-0 variants", "gross (pre-cost) Sharpe", "mean daily turnover",
            "fidelity: lag-0 gross recompute vs attribution.get_all_signal_returns (max abs diff, corr)"]
SURVIVOR_ACTION = ("provisional only (hindsight-picked 25-name universe): reported, fed to Step 2 T4 as a "
                   "reported-only sleeve, flagged for a survivorship-free retest; never a live change on this evidence")

REPLICATION = {
    "design": "scripts/combination_preregistered_bars.py unchanged (fingerprint 96bf26c8aa81b57db21adb8eba6dc3f7afd7dece24857d0195bb25aa8da5e9a2)",
    "runs": ["C0_legacy_optimal (the Step 1 source run above)", "C1_robust_attribution", "C2_robust_standalone",
             "C3_confidence", "P_random_weights"],
    "evaluator": "scripts/run_combination_evaluation.py evaluate (as committed), on the new run outputs",
    "ledger": "docs/combination_trial_history.json +5 trials (a re-measurement is counted as new trials, conservatively)",
    "reading": "report old vs corrected numbers side by side; the 9/28 verdict stands unless corrected bars differ",
}


def bars_fingerprint() -> str:
    payload = json.dumps(
        {"PREREGISTERED_AT": PREREGISTERED_AT, "SOURCE_RUN": SOURCE_RUN, "FOLDS": FOLDS, "STRATEGIES": STRATEGIES,
         "BOOK_RETURNS": BOOK_RETURNS, "COSTS": COSTS, "HEDGE": HEDGE, "SHARPE": SHARPE, "BOOTSTRAP": BOOTSTRAP,
         "PLACEBO": PLACEBO, "BARS": BARS, "REPORTED": REPORTED, "SURVIVOR_ACTION": SURVIVOR_ACTION,
         "REPLICATION": REPLICATION},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(bars_fingerprint())
