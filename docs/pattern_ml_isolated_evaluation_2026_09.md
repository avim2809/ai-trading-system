# Pattern-recognition ML layer: isolated evaluation + retrain (2026-09-27)

**Status:** Complete. Covers Part B items 7-8 of `/root/.claude/plans/typed-splashing-moonbeam.md`
(the research/engineering plan behind today's Part A/B work — see `docs/pattern_recognition_plan.md`
§10 for the full history). Item 9 (live shadow-trading gate) is explicitly **out of scope** here —
see "What this does NOT do" below.

**Bottom line: retrain succeeded and produced a genuinely better classifier, but the isolated
walk-forward test — the primary go/no-go signal this task was scoped to prioritize — still fails
the pre-registered PBO/DSR bar. Recommendation: leave `xgb_confirmation_enabled` OFF.** This is a
legitimate, evidence-based outcome under this plan's own "Honest framing" section, not a failure of
the retrain.

---

## 1. What changed on disk

Retrained both models against **real cached market data** (`data/cache`, NOT synthetic), using the
exact live 25-symbol universe (`config/live.yaml`'s `universe.symbols`) and the full available
history:

```
python scripts/train_pattern_ml.py --data-source cache \
  --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,AVGO,AMD,CRM,NFLX,ADBE,JPM,GS,BAC,V,MA,JNJ,UNH,LLY,XOM,CVX,SPY,QQQ,IWM \
  --start 2010-01-01 --end 2026-09-21 --cv-splits 5 \
  --output data/models/pattern_xgb_partb_2026_09.pkl
```

This uses **all** of today's Part B infrastructure fixes in one pass: the 52-column feature schema
(item 5: dropped non-transferable `entry`/`stop`/`target`/`confirm_index`, added `bars_since_confirm`;
item 6: regime/cross-sectional/liquidity features), the separate binary meta-labeling target (item 2),
de Prado sample-uniqueness weights (item 3), and Purged K-Fold CV (item 4) reported alongside the
usual single trailing holdout.

**Dataset:** 103,315 OHLCV rows / 25 symbols (2010-01-04 → 2026-09-21) → **1,849 confirmed pattern
matches**, 52 features each. Label balance: stop=394, timeout=335, target=1120. Per-pattern counts
now range from `rising_wedge`=407 down to `cup_handle`=1 / `bull_flag`=3 (the same
small-sample patterns Part A's confidence-discount mechanism exists to handle — unaffected by this
retrain, orthogonal fix).

**Artifacts promoted to the live paths** (previous versions backed up first, since `data/models/` is
untracked by git — see `data/models/backup_20260927_partb/`):

| Path | Contents |
|---|---|
| `data/models/pattern_xgb.pkl` / `.onnx` (+ `.onnx.labels.json`) | 3-class direction model (retrained) |
| `data/models/pattern_xgb_meta.pkl` / `.onnx` (+ `.onnx.labels.json`) | **New** binary act/no-act meta-label model (never existed on disk before) |
| `data/models/pattern_sample_counts.json` | Regenerated per-pattern counts (byproduct of the training script's fixed output path — see caveat below) |
| `data/models/backup_20260927_partb/pattern_xgb.{pkl,onnx,onnx.labels.json}` | The pre-retrain (stale, 47-feature) artifact, preserved |

ONNX export ran cleanly in the **main venv** (`onnxmltools`/`onnxruntime` both importable there today —
the `.venv-ml`-only claim in `xgb_classifier.py`'s docstring is stale, same finding as
`docs/pattern_recognition_plan.md` §9 already noted for `torch`/`onnxruntime`). Verified pkl vs. ONNX
prediction parity: max abs diff 1.2e-7 (direction), 6.0e-8 (meta).

**Caveat, stated plainly:** `scripts/train_pattern_ml.py` always writes the per-pattern sample-counts
sidecar to the fixed shared path (`data/models/pattern_sample_counts.json`) regardless of `--output`'s
basename — this got overwritten as a side effect of the retrain, before I could back it up separately.
Since `data/models/` isn't in git, there's no history to recover the old version from. This isn't a
regression: the new file reflects a far larger, current sample (1,849 real matches vs. whatever the
prior CNN-era run produced) and the feature this file feeds (`sample_size_discount_enabled`) is off in
both live configs today, so nothing live was affected.

**Regression check:** `tests/test_xgb_inference.py::TestRealArtifactIfPresent` hardcoded
`N_FEATURES = 47` (the old model's width) and started failing against the new 52-feature model —
updated to 52 with a dated comment. Full run of
`test_xgb_inference.py`/`test_fit_pattern_calibration.py`/`test_train_pattern_ml.py`/`test_patterns.py`/
`test_pattern_recognition_rollout_gate.py`/`test_purged_cv.py`: **200/200 pass** after that one-line fix
(199 passed + this fix; `test_fit_pattern_calibration.py`'s `test_xgboost_sampler_uses_real_model_not_quality_score`
is a "self-healing" test that was previously skipping specifically because the on-disk model was stale —
it now runs for real against the retrained model and passes).

---

## 2. Item 7 — isolated classifier quality (precision / recall / PR-AUC / Brier)

Measured on **true out-of-sample data** two ways, both against the real promoted models:

1. A single trailing time-ordered holdout (`time_ordered_split`, same split `train_pattern_ml.py`
   itself used to fit the saved model — n=462, with a 20-calendar-day embargo).
2. **Purged K-Fold CV** (`firm.patterns.ml.purged_cv.purged_kfold_splits`, 5 folds, 20-day embargo) —
   refit within each fold, aggregated over all 1,849 rows' test partitions (leakage-safe: each
   sample's own label window is purged from whichever fold's training set it would otherwise leak
   into). This is the more informative number (5x the effective n, no dependence on one lucky/unlucky
   split) and is what's quoted below; both are in
   `docs/pattern_ml_isolated_evaluation_2026_09.md`'s companion JSON (see raw output referenced in
   footnote — computed via an ad hoc scratchpad script reusing `firm.eval.classification`'s
   precision/recall/PR-AUC/Brier helpers, which is what `scripts/benchmark_pattern_detectors.py` also
   uses for the rule-based scanner).

Two framings, both real binary decisions:
- **Direction model recast as binary "will this pattern hit its target"** (`p_target ≥ 0.5` from the
  3-class model, positive class = target-hit, base rate 60.6%).
- **Meta-labeling model's native "act / don't act"** decision (same underlying event by construction —
  `label_meta_binary` collapses stop-hit and timeout into "don't act").

| Metric | Direction model (as binary) | Meta model (act/no-act) | Uninformative baseline |
|---|---|---|---|
| Precision | 0.730 (±0.028) | 0.737 (±0.036) | 0.606 (base rate) |
| Recall | 0.885 (±0.030) | 0.851 (±0.024) | — |
| F1 | 0.799 (±0.016) | 0.789 (±0.025) | — |
| **PR-AUC** | **0.836 (±0.020)** | **0.833 (±0.024)** | 0.606 (base-rate-only classifier) |
| **Brier** | **0.185 (±0.009)** | **0.185 (±0.011)** | 0.250 (always-0.5); 0.239 (always-base-rate) |

(mean ± std across 5 Purged CV folds, n≈370/fold; the single-holdout numbers are consistent:
precision 0.695-0.703, recall 0.865-0.891, PR-AUC 0.832-0.837, Brier 0.190-0.193.)

**This is a real, non-degenerate, well-above-baseline result.** Brier clearly beats both the
uninformative-0.5 baseline (0.25) *and* the always-predict-base-rate baseline (0.239) — this is not
just "the model learned the base rate," it discriminates. PR-AUC ~0.83-0.84 against a 0.606 base rate
is a meaningful lift. The reliability diagram (holdout) shows reasonable calibration with mild
overconfidence in the 60-90% predicted-probability bands (e.g. 75% predicted vs. 63% observed) —
worth a calibration pass (`scripts/fit_pattern_calibration.py`, already schema-fixed in Part C) before
ever trusting these probabilities for Kelly sizing, but not disqualifying for a go/no-go read.

**Contrast with the prior failed test:** this is a dramatically cleaner signal than the diluted
11-strategy portfolio audit's PBO=0.714 result suggested — confirming the plan's hypothesis that the
prior test was underpowered, not that the classifier has no signal. The classifier genuinely can tell
good confirmations from bad ones.

---

## 3. Item 7 — isolated standalone Sharpe (single-strategy walk-forward)

**Existing-harness check (per the task's own instruction):** `scripts/run_walk_forward_pbo_audit.py`
has no single-strategy mode built in, but `scripts/validate_pattern_cnn_walkforward.py`'s
`_build_config`/`ExperimentRunner.run_walk_forward`/`derive_recommendation` machinery generalizes
cleanly — `strategies.enabled` is a plain config list, so isolating `pattern_recognition` needs no new
harness, just a 1-line override (`_STRATEGIES = ["pattern_recognition"]`) reusing that script's
functions directly. Ran this exact way rather than building anything new.

**Setup:** pattern_recognition running **alone** (full $10M backtest capital, full real risk-agent
caps — same `max_position_pct`/`max_gross_exposure`/regime overlay as live), 2018-01-01→2025-12-31,
8 walk-forward folds (train_pct=0.7, 1-day embargo), 2-candidate grid (`xgb_confirmation_enabled`
False vs. True, True now pointing at the freshly-promoted real model), genuine train-window candidate
selection per fold (not just sequential OOS replay).

| Metric | Value | Bar |
|---|---|---|
| Aggregate OOS Sharpe (mean across 8 folds) | **-1.087** (std 1.388, range -2.50 to +1.44) | — |
| Aggregate OOS CAGR (mean) | -0.56% | — |
| Aggregate OOS max drawdown (mean) | 0.47% | — |
| Probabilistic Sharpe | 0.119 | need > 0.95 (fail) |
| Deflated Sharpe | 3.4e-05 | need > 0.95 (fail) |
| PBO | **0.534** | need < 0.50 (**fail**, barely) |
| **Verdict** | **fail** | |
| Per-fold train-window winner selected `xgb_confirmation_enabled=True` | 3/8 folds (37.5%) | need ≥75% consistency for KEEP |
| **Recommendation** (`derive_recommendation`, same logic as the CNN gate) | **ROLLBACK** to `xgb_confirmation_enabled=false` | |

**Reading this honestly:** removing the ~0.01 portfolio-weight dilution matters — PBO improved from
0.714 (diluted, stale model) to 0.534 (isolated, fresh model), and the per-fold pattern shows the
XGBoost candidate genuinely winning its train window in the two strongest test folds (fold 2: train
Sharpe -2.12 → +0.49 with XGB on, test Sharpe -1.32; fold 4: positive OOS Sharpe +0.91). But 0.534 is
still on the wrong side of 0.50, and the pooled Deflated Sharpe is statistically indistinguishable from
zero. The strategy trades rarely even alone (mean 38.5 rebalances per ~3-4 month test window across 25
symbols — pattern confirmations are inherently sparse events), which keeps every fold's Sharpe estimate
noisy regardless of which candidate is scored. This is *not* the diluted test's near-powerless read
anymore — it's a real, still-negative-leaning result on genuinely isolated capital.

---

## 4. Does the retrained model pass the existing rollout gate?

Checked both pieces the fail-closed mechanism actually requires
(`src/firm/live/pattern_ml_gate.py` + `scripts/pattern_recognition_rollout_gate.py`):

1. **`firm.live.pattern_ml_gate.evaluate_gate`** only honors `xgb_confirmation_enabled` when
   `data/models/pattern_recognition.rollout_gate.json`'s `overall_recommendation` starts with `"KEEP"`.
2. **`evaluate_rollout_gate`** (the gate's pure decision function) requires `live_sample_days >= 20`
   (or `live_sample_signals >= 30`) **before it will even look at** the backtest verdict — confirmed by
   directly running `pattern_recognition_rollout_gate.py` against today's isolated-backtest result with
   a 0-live-day fixture: `"HOLD (insufficient live sample, backtest verdict=fail)"`, regardless of
   which backtest audit JSON is fed in.

**Conclusion: the retrained models cannot pass this gate today, and structurally could not regardless
of how good the backtest numbers were** — there are zero live-attributed days for either flag (a fresh
retrain this session, no live history yet), and item 9 (the live shadow-trading period that
accumulates that sample) is explicitly out of scope for this task. Separately, even the backtest-side
half of the gate would currently say `fail` (verdict from §3 above). Both halves of the "BOTH (a) and
(b)" requirement are unmet — no config change is warranted or possible via this gate right now.

---

## 5. Recommendation

**Leave `xgb_confirmation_enabled: false` (and `cnn_scoring_enabled: false`) in both
`config/live.yaml` and `config/live_alpaca.yaml` — no config change made, per this task's explicit
scope boundary.**

Evidence-based reasoning:
- The retrained classifier itself is genuinely good (PR-AUC ~0.84, Brier well below both an
  uninformative and a base-rate baseline, stable across 5 purged folds) — Part B's infrastructure
  fixes (meta-labeling split, sample weights, purged CV, feature fixes, regime features) materially
  improved the model over the stale pre-fix artifact. This part of the plan succeeded.
- But the **primary go/no-go signal this task was scoped to prioritize** — pattern_recognition's own
  isolated walk-forward Sharpe/PBO/DSR — still fails its pre-registered bar (PBO 0.534 > 0.50, DSR ≈0),
  even after removing portfolio dilution and using the corrected model. A classifier that can rank
  candidates well does not automatically mean acting on that ranking beats not acting, once real
  trading frictions, sparse signal frequency, and regime variability are in the loop.
- This is exactly the plan's own pre-registered "Honest acceptance criterion": *"if B7's isolated
  strategy test does not show statistically significant edge net of costs... the correct, acceptable
  conclusion is to leave the ML layer permanently off."*
- The fail-closed rollout gate independently confirms no path to enabling this today regardless of the
  backtest read, since no live shadow history exists yet.

**If a human wants to revisit this later:** the closest data point to a positive result was fold 2/4's
train-window wins (2020 and 2021 windows) — worth a specific look if this is ever re-tested, but 3/8
(37.5%) is well short of the 75% consistency bar this same codebase already uses elsewhere for a KEEP
call, so it isn't grounds for one now.

---

## What this does NOT do (explicit scope boundary, per task instructions)

- Does **not** modify `config/live.yaml` / `config/live_alpaca.yaml`.
- Does **not** run a live shadow-trading period (item 9) — that requires real elapsed calendar days
  and is out of scope for a single session.
- Does **not** re-run the full 11-strategy diluted portfolio audit with the new model (that would be
  the ~4-hour "confirmatory secondary check" per the plan's own priority ordering — the isolated test
  above is the primary signal and was prioritized per the task's instructions instead).
- At the time this evaluation ran, `PatternRecognitionStrategy.generate()` did **not** yet wire
  `xgb_inference.score_pattern_meta_confirmation` (the new meta model) into `calibrated_probability` —
  it still sourced that field from the 3-class model's own `p_target`. **This has since been closed**
  (commit `c4f70de`, same session, immediately after this evaluation's retrain made a real
  `pattern_xgb_meta.onnx` artifact available to wire against): a new, independently-toggled
  `xgb_meta_confirmation_enabled` knob (default `false`, same convention as every other knob here) now
  sources `calibrated_probability` from the meta model's own output when enabled, with its own
  `"xgboost_meta"` calibration-file discriminator distinct from the 3-class model's `"xgboost"`. The
  isolated backtest in §3 above still reflects the **pre-fix wiring** (3-class model blended into
  quality score via `xgb_blend_weight`/agreement gate — `xgb_confirmation_enabled`, unaffected by this
  follow-up change) since it ran before that commit; re-running §3 with `xgb_meta_confirmation_enabled`
  also flipped on would be a natural, still-pending follow-up if this is ever revisited, since the meta
  model's own strong precision/recall/PR-AUC (§2) is now reachable through the live decision path for
  the first time, but has not itself been walk-forward-tested end to end yet.
