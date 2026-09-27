# Pattern-recognition ML layer: final verdict (2026-09-27/28)

**Status: settled. Recommendation: keep `xgb_confirmation_enabled` and
`xgb_meta_confirmation_enabled` off in both `config/live.yaml` and
`config/live_alpaca.yaml`. No config file has been changed.**

This closes out the investigation that began from the user's question "what is
required to make the AI layer worth enabling" (2026-09-27). See
`/root/.claude/plans/typed-splashing-moonbeam.md` for the full plan and
`docs/pattern_recognition_plan.md` §10-11 for the earlier Part A/B/D1 history this
built on.

## What changed since the last isolated evaluation

`docs/pattern_ml_isolated_evaluation_2026_09.md` (earlier the same day) reported a
FAIL on the *pre-fix* wiring — but investigating *why* a 73%-precise classifier
produced a negative Sharpe found the real cause: `find_confirmation`
(`src/firm/patterns/confirmation.py`) tested "is price still beyond the level" instead
of "did price cross the level," so `entry` was a price already run through by a
median 1.7R, not a real entry. Every prior test of this ML layer — including that
same-day isolated evaluation — was unknowingly measuring an unexecutable thesis.

Four workstreams ran in parallel to settle this properly:

- **Workstream A** — cross-cutting fixes unrelated to the edge question (a live bug
  in position-sizing smoothing that decayed ~7x too fast in production vs. backtest;
  the backtest cost model overstating round-trip costs ~40%; an AI-enhancement
  opt-out silently not applying on the cycles it mattered most). Committed
  independently (`c967ba4`).
- **Workstream B** — independently re-measured whether the entry-price diagnosis was
  real, with a symbol-block bootstrap and a direction-flipped placebo control on the
  real 25-symbol universe. **Confirmed**: executable expectancy is genuinely negative
  (−0.21R unfixed, CI excludes zero) and loses to its own inverse. One overstatement
  corrected: the inverse doesn't win outright, it's roughly break-even — the honest
  claim is "loses money AND loses to its own placebo," not "the placebo wins."
- **Workstream C** — fixed `find_confirmation` to require a genuine crossing
  (commit `275ab4c`), rebaselined the golden benchmark (detector false-positive rate
  on synthetic noise actually *improved*, 19.0%→14.3%, as a side effect of the
  correctness fix), and retrained both models on honest labels. Signal volume
  collapsed 1,849→970 confirmed patterns — independently reproduced twice (by C, and
  again by the coordinating session after a timestamp-ordering concern), and matches
  Workstream B's own independent re-scan of the fixed code exactly.
- **Workstream D** — built a pre-registered, one-shot evaluation harness (seven bars
  fixed before any result was seen, hashed via `bars_fingerprint()` so they can't be
  quietly adjusted afterward) and, once C landed, ran the real, final evaluation.

## The real, final result

`scripts/validate_pattern_ml_workstream_d.py`, real run (not `--dry-run`),
2026-09-27T20:06:55–21:16:26 UTC, 8-fold walk-forward × 6 genuinely distinct
candidates, pattern_recognition running alone with full capital/risk budget,
2018-2025. Report: `docs/pattern_ml_trial_history.json`,
`docs/pattern_ml_workstream_d_run_log.json` (both committed, `d3e2194`).

| Bar | Result | Pass? |
|---|---|---|
| Executable expectancy > 0 | −0.050R (n=63) | FAIL |
| CI excludes zero | [−0.213, +0.147] | FAIL |
| Beats direction-flipped placebo | −0.050R vs. placebo −0.076R | PASS |
| Deflated Sharpe > 0.95 | **0.00029** | FAIL |
| PBO < 0.50 (6-candidate grid) | 0.345 | PASS |
| Grid ≥ 5 genuinely distinct candidates | 6 | PASS |
| Fold consistency ≥ 75% | 0.50 (4/8 folds) | FAIL |

**4 of 7 bars fail. Verdict: FAIL.**

The DSR deflation is honest in a way no prior test of this layer was: it accounts for
**104 cumulative trials** across this layer's real test history (56 from three prior
sessions + 48 from this run) — not just this run's own 6-candidate grid. Today's raw,
undeflated Sharpe (0.963) would have looked strong without that correction. That gap
is exactly the trap `docs/pattern_recognition_plan.md`'s own "honest framing" section
warned about, and it is why this result should be trusted over any earlier,
less-deflated one.

Confidence-threshold diagnostics (not used for pass/fail, reported for transparency):
sweeping the meta-model's confidence gate at 0.60/0.70/0.75 all showed similarly
negative-to-near-zero expectancy with wide, zero-straddling confidence intervals. No
threshold produces a clean profitable subset. Workstream B's earlier in-sample hint
(`p_act≥0.6` → +0.098R) did **not** replicate once restricted to genuine out-of-sample
walk-forward test windows — itself a useful, honest negative result: it confirms that
hint was exactly the in-sample noise it was flagged as being at the time, not a real
signal that a sloppier test would have missed.

## Reading it honestly

The classifier is not worthless — it modestly beats its own inverse (one real pass),
and a genuine 6-candidate grid gives a technically-passing PBO. But expectancy is
negative with a confidence interval that includes zero, deflated Sharpe is
indistinguishable from zero once this layer's real cumulative test history is
accounted for, and there is no stable train-window preference for meta-gating over
the plain rule-based baseline (50/50 fold split). This is not "inconclusive, needs
another look" — it is a clear, honest FAIL against bars that were fixed before any
result existed.

## What is and isn't settled by this

**Settled**: the specific ML-confirmation/meta-gating axes tested here (3 boolean/
threshold knobs × `zscore_demean`) are not worth enabling on this evidence. The
entry-price correctness bug is fixed regardless of this verdict, and the detector
itself is measurably better (lower false-positive rate) as a result — that part of
today's work stands on its own.

**Not settled, flagged honestly rather than silently**: `pattern_recognition`'s
`default_params` gained 15 new keys during today's earlier Part A work (regime
discount, statistical-significance test, sample-size discount — all default off) that
predate `DEFAULT_PARAM_GRID`'s design and were not varied in this grid. Every
candidate here ran with those 15 knobs at their off defaults, uniformly — so the PBO/
DSR comparison is not invalidated, but this run does not test whether any of those
other knobs interact with ML-confirmation in a way that changes the picture. A
dedicated follow-up sweep would be needed to rule that out, not assumed.

## What would be required to revisit this

Per the plan's own pre-registration discipline: **not** a re-cut window or a wider
grid on the same question — that would be exactly the goalpost-moving this process
was built to avoid. A legitimate re-test would need a **new, independently-motivated
hypothesis** (the same bar this session's own re-investigation was held to), for
example: a materially larger/less survivorship-biased universe (Part D5 of the
original plan, gated on a paid-data decision); a genuinely different execution design
for the corrected signal (Workstream D of the *original* draft plan — persistence,
risk-defined sizing via `RiskAgent` — deliberately not built here since there was no
edge to convert); or new live-accumulated data once enough real trading history
exists to test against real fills rather than a backtest cost model.

## No config changed

`config/live.yaml` and `config/live_alpaca.yaml` are untouched throughout this entire
investigation (verified via `git diff` against the state before this program began).
Every new knob introduced today (`xgb_meta_confirmation_enabled`,
`xgb_meta_min_confidence`, the regime/significance/sample-size knobs from Part A)
defaults off. This is a recommendation with the full measurement attached, not a
unilateral decision — enabling anything remains a human call, and on this evidence the
honest recommendation is not to.
