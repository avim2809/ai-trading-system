# Regime ensemble — implemented and A/B'd (shipped disabled by default)

Follow-up item from the 2026-07-29 P/L-improvement research pass (Phase 7).
Originally scoped-only (see git history for the pre-implementation version of
this doc); implemented end-to-end on 2026-07-29 following the "ensemble-HMM
voting" design below, then A/B'd against the same 3 diagnostic windows used
throughout `docs/portfolio_construction_diagnosis.md`. **Result: shipped, but
left disabled by default** — the ensemble measurably calms the underlying
HMM's own noise, but does not rescue `strategy_regime_weights` at the
portfolio level (see Results).

## Why

`GaussianRegimeModel` (`src/firm/regime/model.py`) already had one real bug
fixed this session: label instability (Bull/Bear labels swapping between
refits on thin separation) was addressed via a separation-based damping
factor in `HMMRegimeStrategy` (`src/firm/strategies/regime_hmm.py`). That
fix flipped `regime_hmm`'s own attributed Sharpe from negative to positive
in all 3 diagnostic windows. But the *complementary* feature —
`strategy_regime_weights` (per-strategy multipliers conditioned on the
detected market regime, `src/firm/agents/research/_regime_weights.py`) — was
A/B'd with real thresholds and **hurt** portfolio Sharpe, and remains
disabled by default in both `config/settings.yaml` and `config/live.yaml`.
The hypothesis: a single Gaussian HMM's regime *label* is still noisier than
the strategy-weighting feature assumes, even with separation damping, and
ensemble-HMM voting (2025-2026 regime-detection literature) might calm that
noise enough to make the feature net-positive.

## What shipped

- `src/firm/regime/ensemble.py` — `EnsembleRegimeModel`: fits 5
  independently-seeded `GaussianRegimeModel` members (`DEFAULT_SEEDS`) and
  combines their classifications by majority vote. `confidence` blends vote
  agreement with the winning members' average posterior confidence
  (`vote_share * mean(member_confidence)`), so *either* a noisy single-model
  posterior *or* ensemble disagreement reduces trust in the label.
  `posterior` is repurposed as the vote-share vector `[P(Bull), P(Chop),
  P(Bear)]` (no current consumer reads it, only `label`/`confidence`).
  Tolerates individual member fit failures (majority still meaningful with
  3-4/5 survivors); re-raises immediately if `hmmlearn` itself is missing.
- `src/firm/regime/detector.py` — `MarketRegimeDetector(ensemble=True,
  ensemble_seeds=...)` swaps in `EnsembleRegimeModel` behind the exact same
  `fit`/`classify` interface. **Off by default** — `ensemble=False`
  reproduces prior behaviour exactly.
- Wired through both existing consumers' freeform config dicts: `RiskAgent`'s
  `regime_overlay.ensemble` (`_detect_regime`) and the orchestrator's
  `strategy_regime_weights.ensemble` (`_resolve_market_regime`) — both
  default `False`.
- `scripts/calibrate_regime_ensemble.py` — the A/B harness (mirrors
  `calibrate_strategy_regime_weights.py`'s structure), comparing the single
  vs ensemble detector with `strategy_regime_weights.enabled=True` held
  constant, same 10-strategy roster / `optimal` combination / 3 windows.
- Tests: `tests/test_regime_ensemble.py` — deterministic vote/confidence
  aggregation against hand-crafted fake members (no dependency on
  `hmmlearn`'s stochastic convergence), a real-fit smoke test, partial/total
  member-failure handling, and end-to-end wiring through
  `MarketRegimeDetector`/`RiskAgent`/the orchestrator.

## Results (2026-07-29, real cache data, `data_source: cache`)

`python scripts/calibrate_regime_ensemble.py` — full 10-strategy roster,
`optimal` combination, `strategy_regime_weights.enabled=True` for both arms
(only the detector differs):

| Window | Detector | Portfolio Sharpe | `regime_hmm` own Sharpe |
|---|---|---:|---:|
| `run_18mo_2025_2026` | single | **0.498** | -1.455 |
| `run_18mo_2025_2026` | ensemble | 0.350 | **-0.575** |
| `wf_fold0_2020_2021` | single | **0.362** | -1.516 |
| `wf_fold0_2020_2021` | ensemble | 0.064 | **-0.599** |
| `wf_fold1` | single | -1.021 | **2.046** |
| `wf_fold1` | ensemble | -1.019 | 2.016 |

Consistent, real pattern across all 3 windows — not a coincidence of one run:

1. **The ensemble does calm the underlying HMM's noise, exactly as
   hypothesized.** `regime_hmm`'s own attributed Sharpe moves toward zero
   (less extreme in either direction) under the ensemble in every window:
   -1.455→-0.575, -1.516→-0.599, 2.046→2.016. In the two windows where the
   single model's regime label was actively hurting `regime_hmm`, the
   ensemble roughly halves the damage.
2. **But portfolio Sharpe gets *worse* with the ensemble in 2/3 windows**
   (0.498→0.350, and sharply 0.362→0.064), and is flat/negligible in the
   third (-1.021→-1.019). Calming the label's noise does not translate into
   a better-performing `strategy_regime_weights` feature once `optimal`'s
   inverse-covariance combination reacts to the changed regime-conditional
   weighting — the same portfolio-construction interaction effect already
   documented for the separation-damping fix (`docs/PROJECT_CONTEXT.md`
   "Portfolio-construction diagnosis follow-up").

**Conclusion: the ensemble detector does not rescue `strategy_regime_weights`.**
Since that feature already tested net-negative with the *single* detector,
finding it's still net-negative (worse, in fact) with a demonstrably
less-noisy detector underneath it is informative — it points at the
inverse-covariance/regime-weight *interaction* as the real problem, not
detector noise. Raw JSON: `/tmp/regime_ensemble_calibration_1window.json`,
`/tmp/regime_ensemble_calibration_2windows.json` (local run artifacts).

## Recommendation

Ship the `ensemble` flag as a tested, available opt-in knob on both
`regime_overlay` and `strategy_regime_weights` (same "built, A/B'd, shipped
disabled" pattern as `strategy_circuit_breaker`) — but **leave
`strategy_regime_weights` disabled by default regardless of detector**, and
do not enable `ensemble=True` for `regime_overlay` without a separate A/B of
that specific consumer (this A/B only tested the `strategy_regime_weights`
path). The interesting open question this surfaces —why does calming
`regime_hmm`'s own noise make the portfolio-level interaction with
`optimal` *worse*, not better — is a genuine follow-up for whoever next
picks up `regime-conditional-weighting`, not something to chase further
here.

## Explicit non-goals

- Not replacing `HMMRegimeStrategy`'s per-symbol alpha signal (already fixed
  and net-positive this session, unaffected by this work) — this item is
  scoped to the *market-level* regime detector feeding
  `strategy_regime_weights`/`RiskAgent.regime_overlay` only.
- HMM + boosted/bagged classifier and statistical jump model (the other two
  candidate designs from the original scoping) were not attempted — the
  cheapest design (ensemble-HMM voting) already gave a clean, real A/B
  result answering the question this spike was scoped to answer.
