# core_v1 research run: verdict (P3-08)

Run: 2026-10-13 00:03 to 01:55 UTC, evaluation driver at commit e6b7dd0, result commit 959ed49. Data to 2026-09-30 (all in-sample). Tier is computed by `gate_report` from the frozen `config/gates.yaml` (v2), not by prose.

## Verdict: TIER D. Not paper-eligible. The plan's outcome is the passive book.

Four independent point failures each force Tier D on their own: the Kelly bound, gate 5 (stress), gate 6 (gross-cap days), gate 7 (benchmark), plus the H4 diversification miss (ENB 2.448 < 2.5), which the report names as the stop reason.

## Every gate

| Gate | Status | Evidence |
|---|---|---|
| G1 DSR | insufficient | DSR 0.628 < 0.95 at the signed family N=31 |
| G2 PBO | insufficient | PBO uninformative: effective grid N 1.12 < 4, median column correlation 0.983 (PBO value 0.709, never a pass) |
| G3 CPCV | **pass** | median path Sharpe 0.0589 (per period), 9 of 9 paths positive |
| G4 cost 2x | insufficient | DSR 0.535 < 0.90 at 2x cost (net Sharpe stays positive at 1x/2x/3x) |
| G5 stress | **fail** | Volmageddon 0.030 > 1.5 x 0.020 |
| G6 robustness | **fail** | gross-cap days 42.3% > 20% |
| G7 after-tax vs BM2 | **fail** (see anomaly) | gap -0.562 vs BM2, correlation 0.55 so the correlation branch does not apply; judged on the FX-covered sub-window 2005-01-03..2026-09-29 (5,469 days) |
| G8 charter order | **pass** | charter approved at 76303104 before the first core_v1 row |
| Kelly bound | **fail** | tau 0.09 > max tau 0.0144 (50% haircut on the deflated Sharpe). The undeflated variant (informational only) would allow tau up to 0.2421. |
| H4 ENB | **miss** | asset-class ENB 2.448 < 2.5 |

Note on the Kelly line: only the gate reading (deflated Sharpe, 50% haircut) fails; the undeflated figure is shown for information.

## Sensitivities (reported, never gate)

DSR at family N 31: 0.628. At N plus the 24 duplicate grid rows (55): 0.409. At raw N 843: 0.0075. At N=463: 0.023. At N=210: 0.083. With all-family var_sr: ~1e-9. Monthly BM2 sensitivity: gap -0.539 (95% CI -0.93 to -0.15). Selected config: forecast_cap 15, buffer 0.2, speed subset all.

## Sizing layer (selected config, 8,474 days)

Vol scale at its 1.5 cap on 3,997 days (mean 1.33). Instrument risk cap bound on 5,018 days and infeasible on 2,355 days. Non-converged cap days: 24. Class cap not applied (frozen decision). The 42% gross-cap figure is the mechanical consequence the w20a report warned about.

## Ledger accounting

Registered core_v1 rows: 82 in this evaluation (grid 12, cost stress 2, stress suite 10, robustness 54, benchmark 3, comparison 1) plus 40 constant-estimation rows from the constants step = **122, matching the expected 122**. Extra: 24 registered grid rows from the failed first attempt (kept, never edited; recorded as `prior_attempts`) and the exploratory `core_v1_engine` rows (raw count only). Raw N at run end: 843. Ledger total 416 rows.

## Gate 7 anomaly: RESOLVED, it was my misreading (w23a, w24a)

The first version of this write-up said the candidate's ILS NAV lost about 7 points a year the engine series earned. That was wrong. I multiplied the annual Sharpe (0.968) by the ILS NAV's volatility (8.2%) and called it a 7.9% return, but that volatility is mostly USD/ILS FX volatility, not strategy volatility. Reading the stored engine return series of the selected trial directly (read-only, no run, no ledger row; `reports/w24a_report.md`): on the gate-7 window (2005-01-03..2026-09-29) the USD CAGR is **2.25%** at **2.35%** volatility (Sharpe 0.96). Converted through USD/ILS (-1.6% a year) the stored path is worth about 0.60% a year in shekels, against the NAV's 0.47% gross. The NAV code is correct (w23a: gross ILS NAV equals engine equity times FX to 1e-6 on synthetic data).

What is real:
- Gate 7's failure reflects a real property of the candidate: about 2.2% a year in USD, swamped by FX drag, against a 60/40 benchmark earning far more. The -0.56 Sharpe gap stands as measured.
- One small real-data fault: raw `close` is not split-adjusted (EEM, EFA, IWM, IYR in 2005, EEM again in 2008, QQQ in 2000). Estimated effect on the shekel NAV: about 0.1 to 0.4 points a year (1.4 at the very outside). It cannot change the verdict.
- Worth its own look: realised volatility was 2.3% against the frozen tau of 9%. The instrument risk cap was bound on 5,018 days and infeasible on 2,355, so the volatility target was not delivered. That is a property of the frozen sizing, not a bug fix target for this run.

The tier is unchanged: gates 5 and 6, the Kelly bound and the ENB miss do not depend on any of this.

## Deviations (all recorded in results.json)

Gate 7 judged on the FX-covered sub-window (USD/ILS begins 2005-01-03); the other gates keep the full 1994 window. Family N=31 membership is provisional. Legacy trial lengths assumed equal to the candidate's. One active family, so the family ENB is diagnostic only. rf = None throughout. The failed first attempt's 24 grid rows are in the ledger.

## Owner follow-ups

1. Optional: decide whether split-unadjusted `close` should be fixed in cleaning (new snapshot, new pre-registration if re-run), and whether the 2.3% realised volatility versus tau 9% deserves its own audit.
2. Commit the drafts in `research/reports/core_v1/drafts/`: the trial-history entry (add its baseline to `tests/integrity/test_trial_history_append_only.py` first), the two integrity-test drafts, and register `core_v1` in `firm.research.legacy_adapters.ADAPTERS` as a no-row adapter.
3. Set `core_v1` to status RUN in `research/preregistration/INDEX.yaml`.
4. Per the plan, Tier D means the passive outcome: keep the 60/40 core. No new variant may be run on the same data without a new pre-registration.
