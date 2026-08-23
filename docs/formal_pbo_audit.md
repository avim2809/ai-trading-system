# Formal PBO walk-forward audit

First end-to-end run of `scripts/run_walk_forward_pbo_audit.py` on the cache
panel (2020-01-01 → 2026-06-30), using a **genuine** `param_grid` that mirrors
real firm tuning choices:

| Candidate | `signal_combination` | `allocation_method` |
|-----------|---------------------|---------------------|
| 0 | `confidence` | `conviction_weighted` |
| 1 | `optimal` | `conviction_weighted` |
| 2 | `optimal` | `equal_weight` |

Setup: 10-strategy roster, 25-symbol universe, `data_source: cache`, 3 folds,
70% train / 30% test per fold, `selection_metric: sharpe_ratio`.

Reproduce:

```bash
python scripts/run_walk_forward_pbo_audit.py --n-splits 3
python scripts/run_pbo_trial_audit.py --fold-ids <ids from output>
```

## Results (2026-07-27)

| Metric | Value |
|--------|-------|
| **PBO** | 0.686 |
| **Deflated Sharpe (DSR)** | 0.285 |
| **Probabilistic Sharpe (PSR)** | 0.631 |
| **Verdict** | `fail` |
| **PBO folds** | 3 |

Fold run ids:

- `20260727_174614_024361_74dd75ad`
- `20260727_175123_164255_09ac5df9`
- `20260727_175622_175468_db8d4e77`

OOS Sharpe (per fold): -0.24, 0.58, 0.41 (mean ≈ 0.25).

## Interpretation

High PBO (~0.69) indicates that **in-sample winner selection among these three
candidates is not reliable** — the train-window best config often underperforms
on the subsequent test window. This is expected on a short cache panel with only
three competing trials and does **not** by itself invalidate `optimal` combination
(which won the separate 3-window A/B in `docs/portfolio_construction_diagnosis.md`);
it flags that **grid search over these knobs without a longer panel / hold-out
folds risks overfitting**.

**Next steps:**

1. Re-run after `longer-dataset` (10y+ with delistings) lands.
2. Expand `param_grid` only with candidates you would genuinely deploy (avoid
   fishing on many variants).
3. Treat PBO `fail` as a gate before promoting any new hyperparameter set to
   `config/live.yaml`.

Raw JSON: `/tmp/walk_forward_pbo_audit.json` (local run artifact).

## Results (2026-08-23): turnover-fix candidates

Re-run of the same harness, this time gating a real production change: a
`rebalance_band_pct`/`rebalance_fraction` no-trade-band + turnover-aware-sizing
fix in `ExecutionAgent` (see `docs/remediation_progress.md` #57-58 for the full
writeup). Setup: same 25-symbol universe, 10-strategy roster, `data_source:
cache`, 2020-01-01→2026-06-30, **4 folds** (not 3), 70/30 train/test,
`selection_metric: sharpe_ratio`.

| Candidate | `rebalance_band_pct` | `rebalance_fraction` |
|-----------|----------------------|-----------------------|
| 0 | 0.0 (pre-fix baseline) | 1.0 (pre-fix baseline) |
| 1 | 0.05 (shipped) | 0.7 (shipped) |

**Process note — gate order was violated here, flagging honestly:** items #3
above says PBO `fail` should gate promotion to `config/live.yaml` *before*
shipping. What actually happened: a faster, ad hoc 3-quarter comparison
(2024-Q1/2022-Q1/2023-Q3 — see #57) showed a clean, consistent win and was
shipped first; this formal walk-forward ran afterward as the "final
sign-off" and came back `fail`. Turnover reduction itself is not in doubt
(confirmed again below), but the ad hoc check was not a substitute for this
gate on the *profitability* question, and shouldn't have been treated as one.

| Metric | Value |
|--------|-------|
| **PBO** | 0.464 |
| **Deflated Sharpe (DSR)** | 0.0031 |
| **Probabilistic Sharpe (PSR)** | 0.197 |
| **Verdict** | `fail` |
| **PBO folds** | 4 |

Fold run ids: `20260823_012615_982647_b5018a9c`, `20260823_014312_778804_13e8502e`,
`20260823_020017_787239_5560fdbf`, `20260823_021648_525201_b609ea50`.

| Fold | Train period | Winner (in-sample Sharpe) | OOS test Sharpe | OOS total_turnover |
|---|---|---|---|---|
| 1 | 2020-01→2021-02 | baseline (1.28 vs 1.19) | −1.28 | 10.28 |
| 2 | 2021-08→2022-10 | baseline (0.72 vs −0.42) | −2.56 | 21.96 |
| 3 | 2023-04→2024-05 | shipped (−0.15 vs 0.69) | −1.32 | 1.46 |
| 4 | 2024-11→2026-01 | shipped (0.01 vs 2.35) | **+3.87** | 1.83 |

OOS Sharpe mean ≈ **−0.32** (worse than the 2026-07-27 audit's +0.25 mean —
this is a harder test: 4 non-overlapping folds across signal-quality regimes,
not 3 combination-method variants).

## Interpretation (2026-08-23)

Two separable findings, easy to conflate:

1. **Turnover reduction is real and confirmed again**: whenever the shipped
   candidate wins (folds 3-4), OOS turnover drops ~85-93% vs. the baseline
   folds (1.46-1.83 vs. 10.28-21.96). This is a mechanical, cost-efficiency
   result independent of alpha quality and isn't what's failing here.
2. **The underlying combined-signal edge is weak/unstable across most of
   2020-2026, regardless of which turnover treatment is used**: 3 of 4 OOS
   folds are meaningfully negative — including fold 3, where the *shipped*
   candidate still lost out-of-sample. DSR≈0 means there is essentially no
   statistical confidence that whichever candidate looked best in-sample
   reflects genuine skill rather than luck, once corrected for having tried
   2 candidates. This reinforces (does not newly discover) the standing,
   unresolved finding in `docs/portfolio_construction_diagnosis.md`: the
   blended portfolio has never beaten its own best single strategy.

**Practical takeaway**: keep the turnover fix (it's a clear cost-efficiency
win with no observed downside), but don't read this walk-forward as
validating live-trading profitability — that question is still open and is
what the in-progress Phase 4 concentration work
(`docs/remediation_progress.md` → "In progress") is trying to close.

Raw JSON: `/tmp/walk_forward_pbo_audit.json` (turnover-fix grid; local run
artifact, not committed — re-run to reproduce).

## Results (2026-08-23): strategy-concentration candidates — also `fail`, worse

Same harness/setup, candidate 0 = current 10-strategy roster (empty override),
candidate 1 = drop `momentum`+`seasonality` (8 strategies). Motivation: #58's
per-fold strategy attribution (above) showed `momentum` negative in 3/4 folds
and `seasonality` strongly negative in the 2 folds it had any trades in — but
seemingly note **this evidence was thinner than first stated**: cross-checking
against `trades.parquet` per fold showed folds 3-4 (where the shipped
turnover fix won) had almost no trades at all for most strategies to evaluate
(the turnover fix is working; it just leaves little to attribute), and even
fold 1's samples were as small as 1-4 trades per strategy. Only fold 2's
12-62-trade samples are robust; `seasonality`'s case (30 trades, Sharpe -2.89)
held up there, `momentum`'s (12 trades, -0.78) was real but milder than
initially characterized.

| Metric | Value | vs. turnover-fix audit above |
|--------|-------|-------------------------------|
| **PBO** | 0.571 | worse (was 0.464) |
| **Deflated Sharpe (DSR)** | 0.0003 | worse (was 0.0031) |
| **Probabilistic Sharpe (PSR)** | 0.132 | worse (was 0.197) |
| **Verdict** | `fail` | still fail |
| Mean OOS Sharpe | −0.63 | worse (was −0.32) |

Fold run ids: `20260823_120519_494940_34e04242`, `20260823_122636_330550_2bd633d2`,
`20260823_124440_589312_b85dc808`, `20260823_130117_189514_a34e7a80`.

| Fold | Winner (in-sample) | OOS test Sharpe |
|---|---|---|
| 1 | concentrated (1.35 vs 1.19) | −0.82 |
| 2 | full 10-strategy (−0.42 vs −1.07) | −2.79 |
| 3 | concentrated (0.85 vs 0.69) | −2.79 |
| 4 | full 10-strategy (2.35 vs 1.51) | +3.87 |

**Interpretation**: concentration doesn't help — the two folds that selected
the concentrated set (1, 3) didn't do any better than the two that kept the
full set (2, 4); fold 4 (the one genuinely good OOS period) actually preferred
the *full* set in-sample, undercutting the "momentum is a universal drag"
hypothesis. This is the **third** consecutive negative result on the core
profitability question this session (alongside `zscore_demean` in
`docs/remediation_progress.md` #58 and this doc's turnover-fix section above),
against a run of clean, validated wins on every execution/cost-related fix.
That pattern is itself the finding: **the problem is not turnover, not this
particular strategy mix, and not the z-scoring detail tested — it looks
structural, most likely in the analyst/combination mechanism itself (forced
mean-0 z-scoring → L1-normalized full-investment sizing → sequential
risk-clipping), the same layer `portfolio_construction_diagnosis.md`
originally flagged.** Mechanical, one-knob-at-a-time A/B testing of this
architecture has now been tried three ways without moving the needle;
further progress on profitability likely needs a design-level rethink of
that layer rather than another quick candidate grid. See
`docs/remediation_progress.md` #59 for the full session conclusion.

Raw JSON: `/tmp/concentration_audit.json` (local run artifact, not committed).

## Results (2026-08-23/24): `joint_optimizer` redesign candidate — `fail`

Same harness/setup as above, now with a 4th candidate added:
`allocation_method: joint_optimizer` (the PART 2 combination-layer redesign —
see `docs/remediation_progress.md` #61 for the full design/implementation
writeup), with the #58 bolt-ons (`rebalance_band_pct`, `rebalance_fraction`,
`conviction_smoothing_enabled`) explicitly disabled for this candidate only,
so its own native transaction-cost-driven no-trade region is tested on its
own terms rather than double-damped on top of the existing bolt-ons.

**Two genuine implementation bugs were found and fixed mid-validation**, both
via re-running this exact gate and noticing the result didn't move when it
should have — same "don't trust a green light without checking the mechanism
actually fired" discipline as the rest of this session:

1. **IC daily/annualized-IR units mismatch** (`estimate_ic` in
   `src/firm/portfolio/optimizer.py`): the realized IR (mean/std of *daily*
   returns) was compared directly against `ir_ref`/`ir_cap`, which are
   annualized-IR-scale thresholds — a genuinely strong annualized Sharpe of
   ~1.5 has a daily ratio of only ~0.09 (dividing by `sqrt(252)` is exactly
   what annualizing does), chronically far below `ir_ref=1.0`. This starved
   `alpha` of real magnitude for any book with a real, decent track record.
   Fixed by annualizing the daily ratio before the comparison.
2. **`ctx.portfolio.history` is never populated during a backtest** —
   confirmed by `firm.backtest.engine`'s own existing comment:
   `PortfolioState.record_snapshot()` is only ever called from the live path
   (`firm.live.portfolio_sync`), never from the backtest/`FirmStrategy` loop.
   `_trailing_book_returns` (the Path-B IC proxy's data source) read exactly
   that field, so the entire trust-building mechanism was **permanently
   inert** (pinned at `ic_prior=0.03`) in every backtest — including the
   *first* full run of this gate, which came back bit-for-bit identical
   after bug #1's fix alone, which is what surfaced this second bug. Fixed
   by having `TraderAgent` maintain its own rolling NAV history as instance
   state (`_book_nav_history`, fed from `ctx.portfolio.nav` — available in
   both backtest and live), with `get_state`/`load_state` persistence
   matching the existing `conviction_ema` pattern, and a no-look-ahead
   guarantee (this cycle's own nav is appended only after estimating
   `ic_eff`, never before). 40 new tests added across
   `tests/test_portfolio_optimizer.py` (30) and a new
   `TestTraderJointOptimizer` class in `tests/test_agents.py` (10), including
   direct regression coverage for both bugs.

**Three full 4-fold gate runs were needed to reach an honest result**:

| Run | Fix state | PBO | DSR | Verdict | joint_optimizer per-fold train Sharpe |
|---|---|---|---|---|---|
| v1 | neither fix | 0.4429 | 0.00526 | fail | [1.871, -0.169, 0.067, -0.949] |
| v2 | IC units fix only | 0.4429 | 0.00526 | fail | **bit-identical to v1** — confirmed bug #2 meant bug #1's fix changed nothing yet |
| v3 | both fixes | 0.4214 | 0.00061 | fail | **[1.786, -2.437, -0.437, -0.588]** |

v3's per-fold in-sample (train-window) Sharpes for `joint_optimizer` are the
real, honest read once both bugs were fixed: **it never wins the in-sample
candidate selection in any of the 4 folds** (winners were `conviction_
weighted`/`confidence` ×2, `equal_weight` ×1, `conviction_weighted`/`optimal`
×1 — all pre-existing methods), and is meaningfully negative in 3 of 4
(-2.44, -0.44, -0.59), only positive-but-non-winning in the 4th (1.79 vs. the
winner's 2.74).

**vs. the #59 turnover-fix baseline** (PBO=0.464, DSR=0.0031, mean OOS
Sharpe≈−0.32): PBO nudges marginally better (0.421 vs 0.464) but **DSR is
worse** (0.00061 vs 0.0031) — essentially zero statistical confidence either
way, and no material, honest improvement on the metric that actually matters
here. Both required success-criteria thresholds (PBO<0.5 *and* DSR>0.95)
remain unmet by a wide margin.

**A plausible mechanism for fold 2's especially bad result** (train window
2021-08-16→2022-10-05, spanning the 2022 bear market): Path-B's IC-trust
mechanism sizes the book up as its *own* trailing realized Sharpe improves —
a textbook performance-chasing dynamic that can lever into a position right
before a regime reversal, with no regime-awareness or mean-reversion
adjustment to guard against it. This is a genuine property of the current
Path-B design, not an implementation bug — a real limitation worth flagging
for anyone extending this in a future increment, not something fixed here (a
third fix in the same session would risk the exact "keep tuning until it
passes" pattern this redesign was scoped specifically to avoid).

**Decision: `joint_optimizer` does not clear the gate.** Not promoted to
`config/settings.yaml` or `config/live.yaml`; `allocation_method`'s shipped
default is unchanged everywhere. Both paper-trading engines remain stopped.
The module (`src/firm/portfolio/optimizer.py`, the `TraderAgent` wiring, all
40 tests) ships as validated, tested, **off-by-default** infrastructure —
same "built, honestly negative, left off" pattern already established this
session for `zscore_demean`, `strategy_circuit_breaker`, and the regime
ensemble. Full detail and the fourth-consecutive-negative-result session
conclusion in `docs/remediation_progress.md` #61/#62.

Raw JSON: local run artifacts (not committed) —
`joint_optimizer_walk_forward_pbo_audit_v3.json` (final/authoritative),
`_v2.json`/(v1, overwritten) kept only as the bug-fix trail described above.
