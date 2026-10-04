# Legacy trial census, 2026-10

**Ticket:** P0-05 (`plan/tickets/P0-05.md`). **Decision:** OD-09 (signed 2026-10-03: count ALL trials, Gann about 145, not 8).
**Machine-readable counterpart:** `research/ledger/legacy_backfill.csv` (37 rows; regenerate with `scripts/seed_legacy_census.py`; consistency test `tests/test_legacy_backfill_csv.py`).
**Status: DRAFT, awaiting owner sign-off (see the end of this file). Nothing here is signed.**

Method and limits. Sources: git history on all refs (351+ commits, first 2026-06-09), tracked `docs/`, tracked `scripts/`, and the Gann archive branch read with `git show` only. `runs/`, `data/`, logs and untracked files were deliberately NOT read (they are not durable records). Unknown counts use conservative upper estimates (`count_is_estimate=true`); when unsure a trial is counted. Overlaps (pre_ledger_ab vs danelfin) are not de-duplicated, deliberately.

## 1. Census table

| Family | n_variants (DSR-facing) | alt convention | estimate? | Source |
|---|---|---|---|---|
| combination | 57 | 57 | false | `docs/combination_trial_history.json` |
| pattern_ml | 104 | 13 | false | `docs/pattern_ml_trial_history.json` |
| standalone_strategy | 11 | 11 | false | `docs/standalone_strategy_trial_history.json` |
| alt_premia | 10 | 10 | false | `docs/alt_premia_trial_history.json` |
| insider_cluster | 8 | 8 | false | `docs/insider_cluster_trial_history.json` |
| eodhd_s1 | 4 | 4 | false | `docs/S1_trial_history.json` |
| eodhd_s2 | 4 | 4 | false | `docs/S2_trial_history.json` |
| eodhd_s3 | 5 | 5 | false | `docs/S3_trial_history.json` |
| eodhd_s4 | 4 | 4 | false | `docs/S4_trial_history.json` |
| eodhd_s5 | 3 | 3 | false | `docs/s5_trial_history.json` |
| allocation_forward_test | 0 |  | false | `docs/allocation_forward_test_trial_history.json` |
| gann | 145 | 8 | true | `origin/research/gann-archive` |
| pre_ledger_ab | 20 |  | true | `docs/remediation_progress.md; docs/archive/strategy_regime_weights_calibration.md` |
| strategy_construction | 39 |  | true | `docs/remediation_progress.md; docs/edge_search_verdict_2026_09.md` |
| pattern_pre_9_25 | 20 |  | true | `docs/pattern_recognition_plan.md; docs/pattern_ml_final_verdict_2026_09.md; docs/pattern_ml_isolated_evaluation_2026_09.md` |
| danelfin | 8 |  | true | `docs/remediation_progress.md; docs/danelfin_best_stocks_arm.md` |
| llm_configs | 6 |  | true | `docs/llm_ab_experiment_log.md` |
| sleeves_allocation | 10 |  | true | `docs/capital_sleeves_plan.md; docs/allocation_portfolio_backtest_2026_09.json` |
| reruns_after_results | 5 |  | true | `git log` |
Totals: ledger-backed **210** (57+104+11+10+8+4+4+5+4+3+0); unledgered estimates 145+20+39+20+8+6+10+5 = **253**; gross **463**. The carried figure is exactly 463 (plus later additions), never rounded to "about 460". Defensible range: 210 (ledger only) to 463; the gate N is the raw 463 per OD-09, with sensitivity at 210.
Excluded: planned `futures_trend` (+2 if ever run; the DRAFT preregistered bars script counts 0 today) and any forward test (`allocation_forward_test`, S2 shadow forward test).

### Evidence per row
- **Ledger-backed rows (210).** One CSV row per JSON entry (combination 16, pattern_ml 4, others 1), so the P1-01 loader can be idempotent on `source_file + source_entry_index`. The per-entry rows sum to each file's counter (checked by the test): `cumulative_trials` where present, `cumulative_trials_through_last_entry` for pattern_ml; `s5_trial_history.json` has no counter and no family key, its count is `len(variant_names)` = 3; `allocation_forward_test_trial_history.json` has no `n_trials` (0, a forward test).
- **Gann, 145 (estimate).** Reconstructed from `origin/research/gann-archive` via `git show` (no checkout). Counted from the grids in the archive scripts: IC ablation `VARIANTS` 10 (`gann_ic_study.py`), follow-up `GANN_VARIANTS` 3 (`gann_followup_study.py`), swing event study `GANN_CALENDAR_CYCLES` 8, correct-cycles 24 natural cycles + 20 price-derived (5 multipliers x 3 scales = 15, plus 5 direct scales), multi-asset 10 natural weekly cycles + 42 (6 multipliers x 7 scales) + 1 Bitcoin halving cycle. That is 118 counted. The anniversary and squaring event-study grids were not enumerated in detail; 27 is an upper estimate for them (CLI defaults: anniversary 3 cycles x 2 pivot orders, squaring thresholds 0.05..0.20 x 2 pivot orders, plus range projections). Total 145. The source plan's "8 experiments" (alt convention) is a lower bound only. **The owner must sign 145 rather than 8.** Note that `docs/gann_research_closeout.md` on the archive branch says experiment scripts and run artefacts "were removed after closeout" while the archive branch keeps them; the archive branch is the evidence.
- **pre_ledger_ab 20, strategy_construction 39, pattern_pre_9_25 20, danelfin 8, llm_configs 6, sleeves_allocation 10, reruns_after_results 5.** Estimates fixed by the ticket's sketch, each with the doc references in the CSV `evidence_ref`. I confirmed the referenced files and the four re-run commits (f0fa1cb, ba52b36 on 2026-10-01; e95649a, 32eca01 on 2026-09-30) exist; I did NOT re-derive the individual counts line by line, so these seven are unverified upper estimates the owner should review.

## 2. Counting conventions and the gate rule
- pattern_ml: **104** = folds x candidates (entries 16, 24, 16, 48) versus **13** underlying configs (2+3+2+6). Combination: every audit contributes its FULL candidate grid, baseline included (rule fixed 2026-09-28), 57.
- Rule: **gate on the max** (104 for pattern_ml). The CSV carries both numbers (`n_variants`, `n_variants_alt_convention`). Under the alt convention the gross would be 463 - 91 - 137 = 235 (pattern_ml -91, gann -137); shown for information only.

## 3. Holdout consumption
There was never a sealed holdout (no `data/holdout`, no freeze config). The only repo-defined "holdout" is the Gann window 2024-07-01..2026-06-30 (`scripts/gann_followup_study.py:62-63` on the archive branch). Per the ticket's reconciliation (counts taken from the ticket, not independently re-counted by this agent), at least **28 evaluations** touched 2024-07..2026-06: Gann follow-up 1; portfolio-construction diagnosis plus 8-9 follow-up A/Bs on `run_18mo_2025_2026` about 10; walk-forward PBO audits 7/27..8/25, 9; D1 ensemble audit 1; combination eval 9/28 and 9/30, 2; standalone eval 1; pattern CNN/XGB/D walk-forwards 4. Every post-7/27 walk-forward reused the fold-4 test window 2026-01-04..06-30. Conclusion: the whole 2020-01-01..2026-06-30 panel is burned for confirmatory purposes.

## 4. Stale numbers (frozen files not edited)
- `docs/alt_premia_trial_history.json` note says combination = 52; the file is 57.
- `docs/eodhd_shortlist_protocol_2026_10.md:74` states `prior_trials` = 190 (57+104+11+10+8); the S-family preregs hard-code placeholders 205 (S3, `scripts/eodhd_s3_..._preregistered_bars.py:409`), 206 (S1, S2) and 207 (s5 ledger `prior_trials_used`). The running ledger total 210 appears in no file. The signed census total supersedes all of these.

## 5. The 0.277 figure
It appears only at `research/gann-archive/results/gann_correct_cycles/CLOSEOUT.txt:19` and is the baseline 10-strategy pipeline Sharpe on the 2024-07..2026-06 window. It is not a best candidate and not out of sample; the source plan's wording is corrected accordingly.

## 6. Data-coverage caveat
Per the ticket: sentiment data starts 2025-11-17 and fundamentals 2020-07-30, so some audits ran 9 of 10 strategies in 3 of 4 folds. This weakens those trials but does not remove them from the count.

## 7. Sharpe capture for var_sr (P0-08)
`sharpe_values_per_period` holds per-period (daily) Sharpes. Trend-family files (`trial_daily_sharpes`) are already daily. S2 uses `trial_daily_sharpes_cash_excess_governing` (the bm2-excess series is descriptive). Combination `oos_sharpes` (entries 14, 15) are annualised in the source (x sqrt(252)); the CSV stores them divided by sqrt(252) with `sharpe_frequency=annualised`, `conversion=SR_d = SR_a/sqrt(252)`. All stored values satisfy |SR| < 0.5 (tested). Legacy estimates carry no Sharpes.

## 8. Items the owner must sign off (OD-09)
1. Gann **145** (not 8): 118 counted from archive grids, 27 upper estimate for two unread scripts' grids.
2. The seven other estimate rows (20, 39, 20, 8, 6, 10, 5 = 108): unverified upper estimates, overlaps kept.
3. Gross **463** as the raw gate N; **210** as the reported sensitivity. The owner may raise a row but not lower it without a written reason.
4. Treatment of `strategy_construction` holdout status ("unknown", assumed touched).
5. Whether forward tests (llm_configs 6) stay counted (they are, conservatively).

## Sign-off
Owner signs here once the rows are reviewed (UTC timestamp from `date -u`): `signed: ______ at ______Z`. Until a signature exists this census is a draft and P1-01 / P0-08 must not treat 463 as signed.

_Draft generated 2026-10-03T22:16Z by agent job B._
