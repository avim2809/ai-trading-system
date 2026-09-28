---
name: project-optimal-combination-lockout-sep28
description: "9/28 DONE (commits 8688512..d44e726): eval FAIL, all combination methods negative OOS, placebo best, PBO 0.971; optimal combination zeroes late-start/no-history strategies (NaN->corrcoef->0) in the real code path; D1's sentiment-54% was a proxy artifact. Robust estimator built OFF by default; pre-registered 5-candidate eval run; WS1 cadence sweep fixed regime refit + test audit-log pollution"
metadata:
  node_type: memory
  type: project
  originSessionId: 7e68870f-025c-467b-8177-1f4914055172
  modified: 2026-09-28T05:21:19.706Z
---

Session 2026-09-28. Full write-up is in `docs/optimal_combination_fix_2026_09.md`.

**Lock-out (confirmed):**
- `combine_signals_optimal` builds its frame with `dropna(how="all")`, no fillna.
- Attribution series start at a strategy's first fill and are never truncated.
- So a late starter gets NaN → `corrcoef` row → `nan_to_num` 0 → exactly 0 weight, permanently.
- Live IBKR at 9/25: momentum, pattern_recognition and volatility_breakout = 0; seasonality ~47%.
- Alpaca (sleeved) is effectively unaffected: one strategy per bucket, so it falls back to confidence.
- In backtests, negative-weight clipping also zeroes mean_reversion on 89% of calls.
- D1's "54% sentiment" came from D1's own `fillna(0)` proxy. A correction banner has been added to `docs/ensemble_redundancy_audit_2026_09.md`.
- The feedback-loop hypothesis is partly refuted: attribution splits by |score|, not by `optimal` weight. But the input series are shared-position contributions, so their correlations are inflated.

**Fix, off by default:**
- `signal_combination.estimator: robust` (+ `returns_source: attribution|standalone`, `lookback_days` 126, `min_obs` 20).
- Robust estimator: daily compounding, trailing window, Ledoit-Wolf shrinkage; an immature strategy gets 1/n, never 0.
- Standalone returns: `PerformanceAttribution.record_signals` / `get_all_signal_returns` (unit-gross score×conf book), persisted.
- Research placebo: `method: random_weights`.
- Frozen prereg `scripts/combination_preregistered_bars.py`, fingerprint 96bf26c8…
- Harness `scripts/run_combination_evaluation.py`; ledger `docs/combination_trial_history.json` (47 prior trials, counted conservatively).
- Runs were launched 04:46 UTC, before the commit (the full suite was too slow under contention). Provenance hashes are in the scratchpad `runs/launch_provenance.txt`.

**WS1 cadence sweep (verified against live logs):**
- Fixed: `regime_overlay.retrain_frequency` counted calls, not bars. Alpaca called it 5–9×/cycle. Tests prove backtest cadence unchanged. Takes effect at the next restart; no restart done.
- Fixed: tests wrote into the live `data/execution_audit.jsonl` (~9k fixture records). Autouse conftest fixture added.
- Not changed (user decision): `rebalance_fraction` 0.7 is applied per cycle live vs per day in backtest.
- Low impact: IBKR's afterhours cycle trades on a stale bar.
- Dormant per-cycle feeds: circuit breaker, hrp, joint_optimizer.

**Own bug caught:** my helper insertion stripped `@staticmethod` from `_check_sleeved_llm_cost_safety`. That would have crashed Alpaca at its next restart. The sleeved API tests caught it and it's fixed, never committed. Lesson: the full suite must pass before any commit, even for "additive" edits.

Related: [[project_pattern_fpr_and_metalabeling_sep27]], [[feedback_production_incident_priority]], [[feedback_verify_before_trusting_a_heuristic]].

**Final outcome (2026-09-28, evening):**
- Pre-registered evaluation: **FAIL** (both primaries pass only 1/6 bars).
- OOS Sharpe: C0 legacy -0.69, C1 -0.54, C2 -0.52, confidence -0.32, **random placebo -0.23 (best)**.
- PBO 0.971; every paired CI is about ±1.1 Sharpe.
- Independently reproduced to floating-point precision.
- Conclusion: the combination method is not where the edge is. No live config change. `estimator: robust` is shipped but off.
- Next lever: per-strategy standalone OOS edge, using `get_all_signal_returns`.
- Ledger `docs/combination_trial_history.json` now holds 52 trials.
- Test fixes committed:
  - audit-log isolation;
  - TestRuns job-lock leak (pre-existing);
  - bs4 importorskip (pre-existing since the 9/22 venv rebuild; beautifulsoup4 installed into the venv).
- Final gates: backend 2817 passed / 0 failed; frontend 141/141; dist rebuilt.
- Both services restarted 16:56 UTC. Healthy, brokers connected, running `signal_combination {method: optimal}` (legacy).
- Not pushed; the user never asked.
