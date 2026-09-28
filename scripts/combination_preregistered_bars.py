"""FROZEN pre-registration for the 2026-09-28 `signal_combination` evaluation.

Written, fingerprinted and committed BEFORE any candidate backtest was run
(see docs/optimal_combination_fix_2026_09.md). Do not edit after results
exist: a changed fingerprint means the run no longer tests the frozen design.
A genuinely new hypothesis needs a new file and a new ledger entry in
docs/combination_trial_history.json, not an edit to this one.

Question: does fixing `optimal`'s input/estimator defects (ragged-start
lock-out, per-cycle cadence, unbounded history, unshrunk pinv) produce a
combination that genuinely beats (a) the current live `optimal` and (b) an
uninformed random-weights placebo, out of sample, after honest deflation for
this layer's cumulative test history?
"""

from __future__ import annotations

import hashlib
import json

PREREGISTERED_AT = "2026-09-28T04:40:00Z"

START_DATE = "2020-01-01"
END_DATE = "2026-06-30"
N_SPLITS = 4
TRAIN_PCT = 0.7
EMBARGO_DAYS = 1
SEED = 42

# Everything else comes from config/live.yaml (IBKR, blended): the 11-strategy
# roster, strategy_params, the full risk block incl. regime_overlay, costs,
# allocation_method, conviction smoothing, universe, initial_capital. Each
# candidate below overrides ONLY `signal_combination`.
CANDIDATES: dict[str, dict] = {
    "C0_legacy_optimal": {"method": "optimal"},  # live today
    "C1_robust_attribution": {
        "method": "optimal", "estimator": "robust", "returns_source": "attribution",
        "lookback_days": 126, "min_obs": 20,
    },
    "C2_robust_standalone": {
        "method": "optimal", "estimator": "robust", "returns_source": "standalone",
        "lookback_days": 126, "min_obs": 20,
    },
    "C3_confidence": {"method": "confidence"},  # pre-2026-07-26 default
    "P_random_weights": {"method": "random_weights", "seed": 0},  # placebo
}
BASELINE = "C0_legacy_optimal"
PLACEBO = "P_random_weights"
# Two primary hypotheses -> two-sided 97.5% CIs (Bonferroni over 2).
PRIMARY = ["C1_robust_attribution", "C2_robust_standalone"]
CI_LEVEL = 0.975

BOOTSTRAP = {
    # Portfolio-level daily returns: there is no per-symbol trade event to
    # block on (unlike the pattern-ML test), so the dependence structure to
    # respect is serial. Paired stationary block bootstrap: the SAME resampled
    # days for both series of a comparison.
    "method": "paired_stationary_block",
    "mean_block_days": 21,
    "n_boot": 5000,
    "seed": 20260928,
}

# A primary candidate PASSES only if every bar passes. Evaluated on the
# concatenated out-of-sample test windows of the N_SPLITS folds (from
# ExperimentRunner._compute_walk_forward_splits), sliced from one continuous
# run per candidate so attribution/signal history carries across folds as it
# does live.
BARS = [
    {"id": "B1_beats_baseline",
     "rule": "OOS annualised Sharpe(cand) - Sharpe(C0) > 0 AND paired block-bootstrap CI_LEVEL lower bound > 0"},
    {"id": "B2_beats_placebo",
     "rule": "OOS Sharpe(cand) - Sharpe(P) > 0 AND paired block-bootstrap CI_LEVEL lower bound > 0"},
    {"id": "B3_deflated_sharpe",
     "rule": "DSR(cand OOS daily returns) > 0.95 with trials = this run's 5 candidates "
             "+ prior_trials from docs/combination_trial_history.json"},
    {"id": "B4_fold_consistency",
     "rule": "OOS Sharpe(cand) > OOS Sharpe(C0) in >= 3 of 4 folds"},
    {"id": "B5_pbo",
     "rule": "CSCV PBO < 0.50 over the full-period daily return matrix of all 5 candidates (n_partitions=8)"},
    {"id": "B6_drawdown",
     "rule": "OOS max drawdown(cand) <= 1.25 x OOS max drawdown(C0)"},
]

# Reported, never used to pass/fail anything:
SECONDARY = [
    "C0 vs C3: does legacy optimal beat confidence OOS (re-test of the 2026-07-26 3/3 claim under live config)",
    "C0 vs P: does legacy optimal beat an uninformed placebo",
    "per-fold OOS Sharpe/return/maxDD/turnover for all candidates",
]


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "PREREGISTERED_AT": PREREGISTERED_AT, "START_DATE": START_DATE,
            "END_DATE": END_DATE, "N_SPLITS": N_SPLITS, "TRAIN_PCT": TRAIN_PCT,
            "EMBARGO_DAYS": EMBARGO_DAYS, "SEED": SEED, "CANDIDATES": CANDIDATES,
            "BASELINE": BASELINE, "PLACEBO": PLACEBO, "PRIMARY": PRIMARY,
            "CI_LEVEL": CI_LEVEL, "BOOTSTRAP": BOOTSTRAP, "BARS": BARS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(bars_fingerprint())
