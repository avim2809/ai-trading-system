# Portfolio Construction Diagnosis: Confidence vs. Optimal Signal Combination

**Date:** 2026-07-26
**Status:** Complete — recommendation applied (see "Action taken" below)
**Script:** `/tmp/diagnose_portfolio_construction.py` (temp, not checked in; raw
output archived at `/tmp/portfolio_construction_diagnosis.json`)

## Question

The original system audit flagged a structural risk: a 12-strategy → 3-analyst
→ PM → risk → execution pipeline has many places where genuine alpha visible
in individual strategies can be diluted, cancelled out, or destroyed by naive
combination and risk-capping — before it ever reaches the market. Two
sub-questions needed a direct, data-driven (not inferred) answer:

1. Does the live default signal-combination method (`confidence` — a
   confidence-weighted mean across strategies) leave a lot of value on the
   table relative to the `optimal` alternative (inverse-covariance weighting,
   which down-weights correlated/noisy strategies)?
2. Does the fully-executed, risk-capped portfolio ever beat the best single
   strategy it's built from, running in isolation? If not, is the multi-agent
   pipeline actually adding value over "just run the best strategy"?

## Method

Re-ran the full production agent pipeline (strategies → analysts → PM → risk
→ execution) via `execute_backtest()` — the same code path used for real
backtests — across 3 independent historical windows already used elsewhere in
this repo (`run_18mo_2025_2026`, `wf_fold0_2020_2021`, `wf_fold1`), each once
with `signal_combination.method = confidence` and once with `= optimal`.
Same universe (25 symbols), same 12 strategies, same risk caps, same seed
(42), same cached PIT data — combination method is the only thing that
changes between each pair.

**Caveat:** the diagnostic's risk block omits `regime_overlay` (the live HMM
exposure-scaling overlay in `config/live.yaml`), so absolute Sharpe/return
numbers here are *not* directly comparable to previously archived `runs/`
reports for the same date windows. This does not affect the confidence-vs-
-optimal comparison itself, since both legs of each pair share an identical
config apart from the combination method.

## Results

| window | method | portfolio Sharpe | best component strategy | best component Sharpe | portfolio beats best? |
|---|---|---|---|---|---|
| run_18mo_2025_2026 | confidence | 0.398 | event_driven | 2.372 | No |
| run_18mo_2025_2026 | **optimal** | **0.478** | ml_prediction | 1.877 | No |
| wf_fold0_2020_2021 | confidence | 1.580 | seasonality | 2.486 | No |
| wf_fold0_2020_2021 | **optimal** | **1.596** | multi_factor | 2.284 | No |
| wf_fold1 | confidence | 0.160 | gann | 4.089 | No |
| wf_fold1 | **optimal** | **0.999** | stat_arb | 3.173 | No |

Full per-strategy Sharpe breakdown and raw metrics: `/tmp/portfolio_construction_diagnosis.json`.

### Finding 1 — `optimal` beats `confidence` in every tested window, never worse

`optimal` improved portfolio Sharpe in all 3 windows (+20%, +1%, and
+524% respectively), including one dramatic case (wf_fold1: 0.160 → 0.999).
Inverse-covariance weighting mechanically down-weights strategies that are
either poor performers or highly correlated with the rest of the blend —
exactly the mechanism needed to stop a single bad or redundant strategy from
diluting the blend the way a flat confidence-weighted mean does. `optimal`
was never worse than `confidence` in any of the 3 windows tested.

This is consistent with, and reinforces, an earlier fix in this remediation
pass: live attribution (`PerformanceAttribution`) is now tracked
unconditionally regardless of combination method (previously it only ran
when `optimal` was already selected — a bootstrap problem where `optimal`
could never accumulate the return history it needs to activate). `optimal`
now has real per-strategy return history available from the first day of a
live run.

### Finding 2 — the portfolio never beat its own best single strategy (6/6)

In every one of the 6 runs, the fully executed, risk-capped, multi-strategy
portfolio underperformed the Sharpe of its single best-performing component
strategy run in isolation. This is expected *to some degree* — a diversified
blend should have a smoother, lower-variance return stream than any single
concentrated bet, so matching the single best strategy's Sharpe is a high
bar. But the gap is large enough in several cases (e.g. wf_fold1 confidence:
portfolio 0.160 vs. gann 4.089 in isolation) to indicate real value is being
left on the table, not just "diversification costs some Sharpe."

A contributing cause, visible directly in the per-strategy breakdown: **`regime_hmm`
had a negative Sharpe ratio in all 6 runs** (range: -0.32 to -2.85). A
strategy that is reliably negative-Sharpe across independent historical
windows is not adding a diversifying, uncorrelated return stream — it is a
persistent drag that a flat/confidence-weighted blend cannot fully filter
out. `optimal`'s inverse-covariance weighting mitigates this somewhat (it
naturally shrinks bad strategies' weight) but does not remove them.
`volatility_breakout` and `trend` also showed negative Sharpe in a majority
of runs.

## Action taken

1. **Switched the live and backtest default `signal_combination.method` from
   `confidence` to `optimal`** (`config/live.yaml`, `config/settings.yaml`).
   The evidence is consistent (3/3 windows, never worse, one large
   improvement) and the mechanism is sound (`optimal` already existed,
   already has a documented fallback to `confidence`-equivalent behaviour
   when return history is thin — see `src/firm/agents/research/_combine.py`).
   No new risk-path is introduced; this only changes how already-computed
   analyst scores are blended.
2. Confirmed the earlier attribution fix (`dominant_strategy_by_symbol`
   fallback in `ExecutionAgent`) is working as intended: `composite_present_in_attribution`
   is `false` in all 6 runs — no trade is silently dumped into an
   unattributed "composite" bucket anymore.

## Recommended follow-up (not yet implemented — tracked as a new backlog item)

- Re-run this same diagnostic with `regime_overlay` enabled (matching
  `config/live.yaml` exactly) to confirm the `optimal` vs `confidence` gap
  holds under the full production risk config, not just the simplified
  diagnostic config. **Not yet done.**
- Because this diagnostic only covers 3 windows with material overlap in
  regime/date range, treat the magnitude (not just the direction) of the
  `optimal` improvement as provisional; the direction is more actionable
  than the exact 3rd number (+524%) which came from a short, low-signal
  window (`wf_fold1`, 5 months).

## Follow-up (2026-07-27): `regime_hmm` fix + generic circuit breaker

The `regime_hmm` investigation above was picked back up and closed out with two
independent mechanisms, both A/B-tested on the same 3 windows/config as this
diagnostic (temp scripts: `/tmp/verify_separation_fix.py`,
`/tmp/diagnose_circuit_breaker.py` — not checked in). Full detail in
`docs/PROJECT_CONTEXT.md` → "Portfolio-construction diagnosis follow-up". Summary:

1. **`regime_hmm` signal-logic fix (shipped, on by default)**: states were
   being labelled Bull/Bear by mean return alone, with no regard for whether
   the gap to the next state was statistically meaningful — a classic
   label-switching failure mode on a short, noisy per-symbol fit. Added a
   `separation` effect-size metric (`GaussianRegimeModel._build_separation`)
   and damp any Bull/Bear signal below a minimum separation
   (`min_state_separation`, default `0.5`) toward neutral instead of trading
   it at full confidence. **This fixed the originally-reported problem**: a
   controlled A/B (fix on vs. off, otherwise identical config) flipped
   `regime_hmm`'s own attributed Sharpe from negative to positive in all 3
   windows (-0.20→+0.66, -0.78→+1.64, -2.73→+1.37). Portfolio-level Sharpe
   improved substantially in 2/3 windows but got worse in the third
   (`wf_fold1` — already flagged above as a short, low-signal window,
   plausibly sensitive to how `optimal`'s inverse-covariance weights shift
   once `regime_hmm`'s correlation with the other 11 strategies changes).
2. **Generic per-strategy rolling-Sharpe circuit breaker (shipped, off by
   default)**: built as the more general fallback mechanism proposed above
   (`firm.agents.research._circuit_breaker`), damping any strategy whose
   trailing realized Sharpe is persistently very negative, independent of
   `optimal`'s variance-only weighting. **A/B result: this measurably *hurt*
   portfolio Sharpe in all 3 windows** with the natural-seeming default
   thresholds (`trigger_sharpe=-0.5`, 60-day lookback) — most of the 12
   strategies show a negative trailing 60-day Sharpe at some point in a short
   backtest window purely from variance, so the gate over-triggered and
   suppressed genuine (if volatile) edge more often than genuine drag. Kept
   in the codebase, fully wired end-to-end, but **disabled by default** —
   this needs either longer/more-stable trailing windows, a stricter trigger,
   or a different statistic (e.g. a t-stat with a track-record-length
   penalty) before it's net-positive. Treat as a validated-but-not-yet-tuned
   research tool, not a ready-to-enable safety feature.

Net effect on the original "does the portfolio beat its own best strategy"
question: unchanged in direction (still no in this diagnostic's 6 legs), but
the specific `regime_hmm` drag identified as "a contributing cause" is now
resolved at the strategy level; the circuit breaker did not turn out to be a
free general-purpose fix for the remaining gap.
