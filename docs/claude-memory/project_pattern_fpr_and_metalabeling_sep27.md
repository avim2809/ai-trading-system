---
name: project_pattern_fpr_and_metalabeling_sep27
description: "9/27 full-day program fixing pattern-detector false-positive rate + building real meta-labeling infra; Part A done, Part B items 1-6 done, 7-9 in progress"
metadata:
  node_type: memory
  type: project
  originSessionId: 7487b3a2-b036-4c00-9ffc-c6ecf4af8cb2
  modified: 2026-09-27T12:55:52.315Z
---

9/27: user asked for a "very deep research" pass (finance/quant-specialist framing) on two findings — "detector flags ~2/3 of random noise as a real pattern" and "make the ML layer worth turning on" — plus any other P&L/ROI lever. Produced and got explicit approval on a research+engineering plan (`/root/.claude/plans/typed-splashing-moonbeam.md`), executed with full autonomy through Parts C/A/D1/B (user's own words: "Full autonmy untill B which is a paid step" — meaning autonomy holds through everything except new paid market-data spend, Part D5). User also asked for a Discord ping after each step with clear mobile-friendly formatting — did this via a scratchpad script reading `ALERT_WEBHOOK_URL` from `.env`, one ping per completed item.

**Part A (false-positive-rate fix), all 7 items shipped, commits `e688f90`..`a6e1bda`:**
- Removed redundant scorer components (follow_through/breakout_distance measured the same thing) + fixed dead `duration` component.
- Real statistical significance test (bootstrap null vs. symbol's own realized vol) + Benjamini-Hochberg FDR control across each scan cycle's candidates.
- Per-pattern minimum-sample confidence discount (rare patterns like cup_handle=1 example no longer trusted like rising_wedge=1351).
- Regime-aware confidence discount (own MarketRegimeDetector instance, direction-vs-regime-conflict discount, damped by label separation/thinness).
- Golden benchmark promoted to a HARD CI release gate (pre-registered floors, verified it actually trips).
- **Net measured result**: synthetic-noise FPR 70%→19.0% (live default, scorer fix alone) →4.8% (opt-in significance+FDR layer). Recall stayed 100% throughout. Brier 0.269→0.073, PR-AUC 0.481→0.942.

**Part B (real meta-labeling infra per de Prado's AFML), items 1-6 of 9 shipped, commits `4e2d899`..`1d44093`:**
1. (done via Part C, calibration model-mismatch fix, earlier commit `522c44a`)
2. Separate binary act/no-act model (label_meta_binary + score_pattern_meta_confirmation against a NEW pattern_xgb_meta.onnx), distinct from the 3-class direction model's own p_target — closes the exact conflation de Prado warns against. NOT yet wired into pattern_recognition.py's calibrated_probability (no real trained meta artifact existed yet at that point) — documented as a deliberate, known gap.
3. Sample-uniqueness weighting (de Prado ch.4, per-symbol average uniqueness from real event spans) feeding xgb_classifier.train's new sample_weight param.
4. Purged + embargoed K-Fold CV (de Prado ch.7.4, calendar-date-based). Found+fixed a real latent bug along the way: sklearn's roc_auc_score silently returns NaN (not an exception) when a CV fold's y_test is missing one of 3 possible labels — was silently corrupting any average.
5. Dropped non-transferable raw features (entry/stop/target price levels, and confirm_index which was a raw-time leak in the rolling-cutoff training construction) — replaced confirm_index with bars_since_confirm (stationary, matches live meta field).
6. Regime/cross-sectional/liquidity ML features added (44→52 feature columns) — dollar_volume_adv_log, relative_strength vs market proxy, 4 raw regime features via compute_regime_features (NOT a full HMM refit per match — too expensive, lets XGBoost learn its own regime structure). Wired into BOTH real callers (live strategy + training script), verified actually populated, not dead code.

**Items 7-8 DONE** (commits `eaf211a`/`c4f70de`, full detail `docs/pattern_ml_isolated_evaluation_2026_09.md`): retrained both models on real cached data (25-symbol universe, 2010-2026, 1849 matches). Classifier itself is genuinely good in isolation (PR-AUC ~0.83-0.84 vs 0.606 base rate, Brier beats both baselines, Purged 5-fold CV). But the PRIMARY go/no-go signal — standalone isolated walk-forward Sharpe — still fails: PBO 0.534 (need <0.50), only 3/8 fold wins (need 75%+). Improved from diluted test's 0.714 but not over the line. **Recommendation: keep xgb_confirmation_enabled/cnn_scoring_enabled OFF** — honest, plan-sanctioned outcome, not a failure. Also closed the item-2 wiring gap (calibrated_probability now sources from the new meta model via an independently-toggled xgb_meta_confirmation_enabled knob, still off by default — zero live impact). Item 9 (live shadow period) explicitly out of scope, needs real elapsed calendar time; gate confirmed structurally can't pass today (zero live-attributed days).

**D1 (parallel, separate task) DONE** (`docs/ensemble_redundancy_audit_2026_09.md`): real correlation audit across the 11-strategy live roster. Prior "momentum/event_driven=0.74" claim did NOT replicate (-0.44, opposite sign, different roster/window — flagged honestly as disagreement, not covered up). Real redundant cluster found: multi_factor↔seasonality=0.72 (multi_factor is the standout, Sharpe 0.80; seasonality flat 0.06, regime_hmm negative -0.55). Bigger finding: signal_combination:optimal empirically degenerate — parks 54.3% weight on `sentiment` (likely just a data-coverage artifact, its cache only starts 2025-11-17) while starving multi_factor (best Sharpe) down to 1.4% weight, smallest of all 11; effective_n=3.07 (11 strategies behaving like ~3). Leaner-8-strategy-subset walk-forward vs full-11: genuinely mixed (leaner won 2/3 folds, both strongly positive OOS; full roster's one win was strongly negative), but formal PBO gate still fails (0.705, n=3 folds, no cross-candidate counterfactual). **Recommendation: inconclusive, no config change** — two concrete follow-ups instead: (a) surgical "drop regime_hmm+seasonality only" re-test, (b) fix optimal's history-length blind spot (arguably more urgent than the redundancy question itself).

No config files touched anywhere in this entire day's work (Parts A, B, D1) — every new knob defaults off, matching this codebase's own established convention.

Full narrative + exact numbers logged in `docs/pattern_recognition_plan.md` §10 (repo doc, kept in sync same session per [[feedback_repo_doc_and_memory_upkeep]]).

**Why:** real production-adjacent ML/statistics work with measurable before/after numbers at every step; user wants this level of rigor and traceability, not just "I fixed it."
**How to apply:** when resuming, check `docs/pattern_recognition_plan.md` §10 first for the up-to-date state (items 7-9 outcome), then `git log --oneline` for exact commits. Don't re-litigate Part A/B items 1-6 — they're done, tested, committed. If items 7-9 recommend "leave ML off," that's a valid, expected outcome per the plan's own honesty bar, not something to keep re-testing.
