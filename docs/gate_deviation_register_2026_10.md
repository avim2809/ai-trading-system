# Gate deviation register (OD-16): for owner signature

**Status: DRAFT, unsigned.** Prepared 2026-10-04 for the owner. Nothing in `config/gates.yaml` (ticket P0-08) can be written
until this register is signed, and nothing in P1-08, P3-08 or P6-02 can start before that.

## What you are signing

The source plan's gates (G-RESEARCH 1-8, G-PAPER 1-5, G-LIVE-STEP, G-DECOMMISSION) could not be applied to this repo as
written. Some were ambiguous (75% of 9 paths), some could not be computed (a variance across trials that only kept a
Sharpe number), and some conflicted with code already frozen in the repo. Each row below states what the source said, what
the plan now says, and why. Signing means: **these are the thresholds, they are fixed before any result is seen, and a later
change is a new signed version plus a new pre-registration, never an edit.**

Source of every row: `plan/tickets/P0-08.md` (rows 1-19) and `PLAN.md` section 5. Numbers are unchanged from those files.

## Three things to understand before you sign

1. **The top tier is practically out of reach, by design of honest counting.** With all legacy trials counted (N = 463; 210 if
   only ledgered trials count), a deflated Sharpe of 0.95 needs an annualised Sharpe of about **1.65** (about 1.05 even if the
   null were perfectly length-matched). The plan's own expectation for a diversified trend portfolio is **0.3 to 0.6**. So a
   Tier A verdict (the only one that allows paper trading) needs the real Sharpe to be roughly three times the plan's own
   expectation. This was accepted in advance: the likely honest outcome is the passive 60/40 portfolio (already live on Alpaca).
2. **Two gates pull in opposite directions.** Gate 6 wants Sharpe to stay flat when parameters move; the PBO test (gate 2) on
   near-identical variants tends to read 0.5. The register therefore never lets PBO alone produce a hard failure (row 10).
3. **Your capital is under $25k (OD-01).** The plan says a bespoke system is not run live at that size, so these gates mainly
   decide whether the research result is credible, not whether money goes in.

How to answer: mark one box per row. **Approve** = sign as drafted. **Amend** = write the change in the box (I will redraft the
row). **Revert** = use the source plan's wording; the ambiguity named in the row then has to be settled some other way before
`gates.yaml` is written.

## Owner responses recorded so far (2026-10-04, in chat; the signature block below is still unsigned)

- Approved as drafted: rows 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19.
- Amended: row 15 (forward-data check is reported, not required before the first live step).
- Open: row 2 (N rule). The owner asked whether the resulting top-tier bar is realistic; the answer is in the hand-off message and in the note under row 2 once decided.

## Register

### A. G-RESEARCH (candidate to paper-eligible)

| # | Item | Source plan said | This register says | Why | Your call |
|---|---|---|---|---|---|
| 1 | **DSR variance (`var_sr`)**, gate 1 | "SR variance across ledger trials of the same family" | Length-adjusted variance: `max(0, var(SR_i) - mean(sampling_var_i)) + sampling_var(T_cand)`, using per-period Sharpes of the named `var_sr_family` (legacy `trend`, alt_premia T2 and C1, S3 x5, futures_trend if ever run, all core_v1 grid trials), floored at the candidate grid's own variance. The unadjusted and all-family variances are reported as sensitivities. A second named list, `prereg_family` (`core_v1`), is used only by gate 8. Pooling Sharpes with different periods per year is refused. | Grid-only variance can be near zero, which gives no deflation. Short legacy windows (`trend` has 491 OOS days) would inflate a raw variance. | ☑ Approve ☐ Amend ☐ Revert |
| 2 | **N rule** (number of trials), gate 1 | N = effective trials including legacy, gates use the max of the estimates plus raw legacy count | `N_gate` = raw count of all trials (returns-bearing + estimate-only legacy rows + counted unregistered and API-capture rows). **463 to start**; sensitivity at **210**. Effective-N estimates (ONC, ENB, participation ratio, MP eigenvalues) are reported, not gated. `dsr()` raises instead of silently falling back to PSR(0). | Every effective-N estimate is at most the raw count, so a max over them would always equal the raw count; stating the raw count says what is really used. *Alternative not adopted:* N = max(onc, enb, mp) over returns-bearing trials + raw estimate-only rows. | ☐ Approve ☐ Amend ☐ Revert |
| 3 | **PBO**, gate 2 | PBO < 0.30; prefer < 0.20 above 20 variants | CSCV with S = 16. **< 0.30**, and **< 0.20** because the family has more than 20 variants (all trend-family trials count). PBO is "uninformative" (cannot pass, never fails hard) when the grid's effective N is 3 or less, or median column correlation exceeds 0.95. The grid must vary things that change returns (cap, buffer, speed subset); tau is fixed ex ante (OD-17). Legacy `cscv_pbo` keeps S = 8 for frozen code. | "Prefer" is not a threshold. A near-identical grid makes PBO meaningless. | ☑ Approve ☐ Amend ☐ Revert |
| 4 | **CPCV**, gate 3 | Median OOS path Sharpe > 0 and at least 75% of paths positive | 10 groups, 2 test groups = 9 paths. Median > 0 and **at least 7 of 9** positive (75% of 9 is 6.75, not a whole number). The in-sample procedure is frozen: purge + 1% embargo; re-estimate the P3-11 constants on the training groups only; pick the grid config by training Sharpe (at most 12 configs); score the test groups with it. Constants that cannot be re-estimated are listed as known leakage before any result. | Without re-estimation every path uses the same config and "7 of 9" collapses to one full-sample Sharpe test. | ☑ Approve ☐ Amend ☐ Revert |
| 5 | **Cost stress**, gate 4 | Net Sharpe > 0 and DSR >= 0.90 at 2x costs | Same thresholds. The multiplier scales every component of the cost breakdown (commission, fees, half-spread, impact, roll). 3x is reported only. Legacy flat-bps shown for comparison only. | Makes "2x costs" unambiguous. | ☑ Approve ☐ Amend ☐ Revert |
| 6 | **Stress episodes**, gate 5 | No named episode loses more than 1.5x the charter max-DD | Same 1.5x, but the reference is the pre-committed bootstrap p95 (row 11), never a live peak that can be reset. Each episode reports how many instruments were active; below `min_active_fraction` the episode is "insufficient", not a pass. | Stops a short or half-empty episode passing by default. | ☑ Approve ☐ Amend ☐ Revert |
| 7 | **Parameter robustness**, gate 6 | "Every free parameter" +/-25%, Sharpe within 30% | One parameter at a time, +/-25% each direction. Pass if perturbed net Sharpe is **at least 0.7x** the chosen one. One-sided (a perturbation that improves Sharpe is not a breach). Integer parameters round half up, minimum 2. Applies to **every** numeric core parameter, no exemptions; the full list is frozen in `gates.yaml`. A parameter whose scalar depends on a constant re-estimates it with the P3-11 procedure. | "Within 30%" did not say which direction or what happens to integers. | ☑ Approve ☐ Amend ☐ Revert |
| 8 | **Benchmark**, gate 7 | After-tax, after-cost Sharpe at least the passive benchmark, or correlation <= 0.3 with a rationale | Both branches kept, plus a mandatory power analysis (minimum detectable Sharpe gap; paired bootstrap, alpha 0.05, power 0.80). Decision table: beats BM2 by the pre-set margin = pass; beats it by less = pass only with correlation <= 0.3 plus charter rationale, else Tier C; fails = pass only with correlation <= 0.3 plus rationale, else Tier D. BM2 = 60/40 SPY/IEF, **annual** rebalancing for the after-tax test (monthly shown as a sensitivity; the gate uses whichever has the higher after-tax Sharpe), after cost and tax (`config/tax_il.yaml`), pre-seal data only. `core_only_100` is a second reference. The live Alpaca book is excluded (post-seal). | Monthly rebalancing realises gains more often and would flatter the candidate. | ☑ Approve ☐ Amend ☐ Revert |
| 9 | **Mechanism**, gate 8 | Economic mechanism in the charter, reviewed before any backtest | The mechanism section is written by you and approved in `research/charters/` with a git timestamp earlier than the first ledger row carrying the `core_v1` prereg id. CI checks the order. A failure is Tier D. | Makes "before any backtest" checkable. | ☑ Approve ☐ Amend ☐ Revert |
| 10 | **Verdict tiers** | Pass or fail | **Tier A** = all 8 gates explicitly pass (the only paper-eligible tier). **Tier D** = a point-estimate failure: Sharpe <= 0, PBO >= 0.5 *and* probability of OOS loss >= 0.5, median CPCV path <= 0, net Sharpe <= 0 at 2x, stress breach, robustness breach, gate 7 failing with correlation > 0.3, gate 8 failure, or an ENB miss at H4. **Tier C** = no point failure but a confidence bar missed (DSR < 0.95, PBO over its limit without the Tier D pair, PBO uninformative, fewer than 7 of 9 paths, DSR at 2x < 0.90, an "insufficient" stress episode). Any missing or unmapped gate result counts as D. C and D both lead to the passive outcome. | The repo already uses tiers A to D; a "can't tell" result must not look like a pass. | ☑ Approve ☐ Amend ☐ Revert |
| 11 | **Charter max drawdown** | One "pre-committed max-DD" used for two purposes | One fixed procedure: stationary block bootstrap, Politis-White block length, 10,000 draws, recorded seed, **2,520-day paths**, 95th percentile, on the chosen config's 1x-cost backtest. It gives two references: gate 5 uses 1.5x the p95 for paths of each episode's own length; the kill switch and decommission use `survival_ref = max(p95 at 2,520 days, 2.5 x tau)`. The 2.5 x tau figure is a rule of thumb derived from the plan's own 20-35% at 12-15% vol; it is not a figure from the source's evidence section. | p95 drawdown grows with horizon, so the path length must be frozen. A lucky backtest must not set a hair-trigger kill switch. | ☑ Approve ☐ Amend ☐ Revert |

### B. G-PAPER, unseal and phase exits

| # | Item | Source plan said | This register says | Why | Your call |
|---|---|---|---|---|---|
| 12 | **G-PAPER 1** | 6 months and 26 rebalances | At least **26 scheduled weekly review events** (traded or not) and at least 6 calendar months, starting after the 2-week embargo, on the dedicated candidate account only. | A "rebalance" can be skipped by a buffer; a review event cannot. | ☑ Approve ☐ Amend ☐ Revert |
| 13 | **G-PAPER 2 and 3** | Correlation >= 0.95 and tracking error <= 25% of tau against a shadow replay; costs <= 1.5x modelled | Same numbers, over a trailing 63-trading-day window (row 19). Cost test (median per trade and aggregate, commission included) is evaluated only after **at least 30 fills across at least 10 instruments**; paper runs longer until reached. | A cost ratio on a handful of fills is noise. | ☑ Approve ☐ Amend ☐ Revert |
| 14 | **G-PAPER 4 and 5** | Zero position breaks over 1 day; kill switches fault-tested | Same, with durable records of break duration and missed reviews. A stuck working order that blocks a review counts as a missed review unless explained. Every kill-switch tier is fault-injected. | Needs evidence that survives a restart. | ☑ Approve ☐ Amend ☐ Revert |
| 15 | **Unseal of forward data** (OD-18) | Not specified | At least **12 months** of accrued post-seal data before the one-time forward evaluation. **AMENDED by owner 2026-10-04: the check is reported but NOT required before the first live step** (was: required). Pass = forward net Sharpe not below the 5th percentile of block-bootstrapped 12-month backtest Sharpes, and forward max-DD within the charter envelope (`survival_ref`, 12-month-horizon bootstrap p95). Labelled a no-contradiction check, not proof of edge. When `gates.yaml` is written, G-LIVE-STEP in PLAN.md (which still says the P3-10 pass is needed before step 1) and P7-02 must be updated to match. | One look at the forward data per family. | ☐ Approve ☑ Amend ☐ Revert |
| 16 | **H4 and H5 exits** | Checkpoints described in words | H4: ENB across asset classes at least 2.5 (ETF) or 3 (futures) is a **hard exit**; a miss stops the candidate with no after-the-fact changes to the universe. H5: met by a dry-run `Allocator.plan` enforcement test, not by a document. | Makes the checkpoints testable. | ☑ Approve ☐ Amend ☐ Revert |
| 17 | **P1-08 acceptance rates** (does the statistics library work?) | Size and power on synthetic data at stated rates | Frozen before the harness is written. Size <= 0.07 at alpha 0.05 (K = 50, 500 sims). Power >= 0.80 for one test at annualised SR 1.0 over 10 years; for K = 50, power >= 0.80 **with the true strategy at SR 1.3**, or within 0.05 of a simulated best possible test. SR 1.0 among 50 is reported only (about 0.53-0.55 for any correct implementation). PBO averages 0.4-0.6 on noise and below 0.1 in the pinned strong-drift case (N = 30, SR 3.0, T = 1,600, S = 16, 20+ seeds). DSR false-pass at most 5%, plus a two-sided calibration. All scenario parameters frozen in `gates.yaml`. | The source's rates cannot be met by a correct implementation (recomputed in the ticket). | ☑ Approve ☐ Amend ☐ Revert |

### C. G-LIVE-STEP, G-DECOMMISSION and other frozen keys

| # | Item | Source plan said | This register says | Why | Your call |
|---|---|---|---|---|---|
| 18 | **Live steps and decommission** | 25% to 50% to 100%; any one trigger = review; two = de-risk to 0 | Same. One trigger = probation and review; two = target weight 0 until you approve. Drawdown trigger is 1.5x `survival_ref`. For a live step the drawdown reference is `survival_ref` x 1.0 (1.5x is decommission, not pass); at least 3 months per step; the cost test on live fills only after 30 fills across 10 instruments. `g_paper` may carry at most one pre-declared re-evaluation date. | Closes ambiguity between "pass" and "decommission". | ☑ Approve ☐ Amend ☐ Revert |
| 19 | **Other frozen keys** | Not in source | `tau <= 0.5 x Kelly vol` enforced as a charter-integrity check; H4 measured by the weight-dependent ENB on asset-class P&L with a 0.7 stress-correlation flag; P4-03 vol-targeting settings (`max_vol_scale` 1.5, 2x instrument risk cap); P4-04 kill-switch behaviour (block new orders, never flatten automatically; settings in `config/kill_switch.yaml`); stress-period list hash; fidelity window of 63 trading days with a zero-exposure-day rule; `n_rule` text equal to row 2. | Everything a later ticket would otherwise choose after seeing results. | ☑ Approve ☐ Amend ☐ Revert |

## Known limits you are accepting

- Constants that cannot be re-estimated inside each CPCV split (row 4) carry leakage from having seen all pre-seal data; they are listed before results.
- All data up to 2026-09-30 is treated as already seen (OD-04). Only forward data from 2026-10-01 is clean, so a strong pre-seal result is not out-of-sample evidence.
- Passing every gate shows **no contradiction with an edge**, never proof (a Sharpe of 0.5 needs about 11 years to confirm).
- The tax model and tax figures are information only; confirm with an Israeli tax professional before any live step.

## Signature

Nothing below is valid until you fill it in yourself. An agent must not sign.

```
Rows approved as drafted:  ____ of 19      Rows amended: ____      Rows reverted: ____
Signed by: ________________    Date/time (UTC, from `date -u`): ________________
sha256 of config/gates.yaml with the meta block removed (filled in at freeze): ________________
```

After you sign: ticket P0-08 writes `config/gates.yaml` from the signed rows, the integrity test pins its hash to the line above,
and every later pre-registration stores the same hash.
