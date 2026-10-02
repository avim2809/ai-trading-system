# PLAN: Making ai-trading-system Credible (repo-reconciled)

**Purpose.** This is the master plan for turning the owner's "credibility plan" into work that fits this repository as it exists on 2026-10-02. The goal is to build a research and promotion pipeline in which a candidate strategy can only reach real money after passing pre-committed statistical, cost, stress, paper-trading and live-step gates, and in which failure (falling back to a passive 60/40 benchmark) is an accepted, expected outcome. The source plan was written by an author who could not see this repo, so every path was proposed. This document replaces those paths with real ones, marks what is new, and adds the safeguards needed because two live paper-trading services and two forward tests run from this same checkout. Ticket detail lives in `plan/tickets/<ID>.md` (B3 template); owner decisions live in [plan/OWNER_DECISIONS.md](plan/OWNER_DECISIONS.md).

Audience: the repo owner (decisions, sign-offs) and coding agents (executing tickets). Today is 2026-10-02. All UTC timestamps in artefacts come from `date -u` or git, never from log lines (the host clock is Asia/Jerusalem).

---

## 1. Realistic expectations (source B0, in substance)

- **Target.** A diversified trend (and, only if futures are pursued, carry) system at roughly 12-15% annual volatility. On the ETF path (long/flat, gross at most 1.0x) the target vol is set ex ante from asset vols, expected about 8-10% (OD-17). A plausible long-run net Sharpe is 0.3-0.6. This is an estimate, not a forecast.
- **Drawdowns.** Expect a maximum drawdown of 20-35% over a decade and multi-year flat periods. These are normal for this style.
- **Detectability.** At Sharpe 0.5, the minimum track record for 95% confidence is about (1.645/0.5)^2, roughly 11 years. The gates therefore test fidelity, robustness and absence of overfitting, not live P&L. A pass is "no contradiction", never proof of edge.
- **Benchmark.** Passive after-tax, after-cost: BM2, a 60/40 SPY/IEF portfolio, on pre-seal data. **[ADJ]** For the after-tax gate 7 the benchmark uses annual rebalancing costs (as source P2-05 says; monthly rebalancing realises gains more often and would bias gate 7(a) toward the candidate). The monthly-rebalanced BM2 (repo precedent) is a reported sensitivity, and gate 7 uses whichever variant has the higher after-tax Sharpe. If the candidate cannot beat it risk-adjusted after tax with a credible DSR, the rational outcome is to stop. That is acceptable and is accepted in advance.
- **What the repo already says.** 0 of 11 live pipeline strategies have standalone edge (`docs/standalone_strategy_evaluation_2026_09.json`); all alt-premia results and every S1-S5 family verdict are Tier C or worse (S4 1-month N500 variant Tier D; S2/S3 conditional on A7); only the BTC 4-week trend is a lead and is already an 8% sleeve on Alpaca. Tier C means the tests lacked power to confirm a modest real edge. The prior for this plan is that Tier C or D, and therefore the passive outcome, is the most likely honest result.
- **Correction carried from the source.** The "0.277 best holdout Sharpe" is the baseline 10-strategy pipeline Sharpe on the Gann follow-up window (2024-07..2026-06), not a best candidate and not out of sample.

## 2. How this plan was reconciled with the repo

Nine read-only subsystem surveys (stats, ledger, census, data, costs, strategies/live, risk/lifecycle, guardrails, research) compared every source path with the code. Main findings that shaped the plan:

1. **There is no historical holdout.** The 2024-07..2026-06 "holdout" was a Gann backtest window; the 2020-01..2026-06-30 panel was reused by about 28 evaluations; EODHD studies read through 2026-09-30. Everything up to 2026-09-30 is declared burned. Only forward data from 2026-10-01 is sealed (OD-04).
2. **Post-seal data already sits on this host** (`data/cache`, `data/`, `data_alpaca/`, `data/research/s2_forward/`, `docs/s2_forward_snapshot.json`, logs). The seal is enforced by ACL, deny rules and a fail-closed loader, not by physical absence.
3. **Much of the numerics already exists.** `src/firm/eval/overfitting.py` has PSR, DSR and CSCV PBO matching the source formulas; MinTRL, CPCV, RC/SPA/Romano-Wolf, ENB, a unified ledger and CUSUM do not exist, and there is no unified cost model (cost logic is split across `firm.agents._liquidity`, backtest commissions and other places). `strategy_correlation.py` does not exist on any branch.
4. **The package is `src/firm/`.** Top-level `src/validation` etc. would need a pyproject change and a reinstall of the live `.venv`, which is forbidden.
5. **Frozen paths.** `scripts/*_preregistered*.py`, `docs/*evaluation*.json`, `docs/allocation_portfolio_backtest_2026_09.json`, `docs/allocation_replay_2026_09.json`, `scripts/eodhd_clean.py` are never moved, renamed or edited. `docs/*_trial_history.json` are never moved and their old entries are immutable (append only).
6. **The live services import this checkout.** Any merge that touches a live-imported module is a deploy (the 9/28 version-skew incident). Hence the worktree-only rule and the LIVE-IMPORT-PATH protocol (section 8).
7. **Several source rules conflict with ops practice** (autonomous restarts, LLM arm B, `CLAUDE.md = @AGENTS.md` dropping safety rules of thumb). These are resolved by scoping research rules to research sessions and strategies managed by `firm.lifecycle`.

Status vocabulary used in the ticket index: NEW, EXTEND_EXISTING (builds on or wraps existing code, without editing frozen code), MODIFIED_SCOPE (source ticket changed), ADDED (not in the source), DEFERRED (waiting on an owner decision or on the calendar).

## 3. Package and file layout decision

All new code goes under the existing package `src/firm/`. New subpackages are additive; `firm.live`, `firm.api` and `firm.runtime` do not import them until the owner-approved integration ticket P6-01. P0-06 enforces this with a fresh-interpreter subprocess test that names every new module.

| Area | Location | Notes |
|---|---|---|
| Validation stats | `src/firm/validation/` (new): `sharpe_stats`, `pbo`, `cv`, `bootstrap`, `nulls`, `multiple_testing`, `effective_trials`, `diversification`, `synthetic`, `stress_suite` | Not `src/firm/eval/`: eval is on the live import path (capital_gate, eval/reports, patterns/significance -> robustness) and has about 10 frozen consumers. New code reuses `overfitting._norm_cdf/_norm_ppf`; equivalence tests pin legacy outputs. `overfitting.py` and `patterns/ml/purged_cv.py` are not edited. |
| Research infrastructure | `src/firm/research/` (new): `ledger`, `legacy_adapters`, `prereg`, `seal`, `data_access`, `capture` | `data_access` is the fail-closed loader all new harnesses must use. |
| Costs | `src/firm/costs/model.py` (new) | Reuses `firm.agents._liquidity` without editing it. |
| Signals | `src/firm/signals/{vol,ewmac,breakout,carry}.py` (new) | Not `strategies/`, whose ABC is `BaseStrategy.generate(pit_view)`. |
| Portfolio | `src/firm/portfolio/{forecast_combine,sizing,weights}.py` (new modules in an existing package) | Listed by name in the isolation test. |
| Engine | `src/firm/backtest/vector_engine.py` (new) | Signed, multiplier-aware, cost-model driven. backtrader untouched. |
| Risk / monitoring / lifecycle / reporting | `src/firm/risk/{limits,kill_switch,kelly}.py`, `src/firm/monitoring/{allocation_forward,decay,fidelity,shadow_loader}.py`, `src/firm/lifecycle/{state_machine,gates,decommission,gated_sleeve}.py`, `src/firm/reporting/{after_tax,diversification_report,gate_report}.py` | All new. |
| Data | `src/firm/data/{etf_loader,manifest,qa,cleaning}.py` | Cleaning v3 is additive; `scripts/eodhd_clean.py` v2 stays. `futures_loader`/`futures_roll` deferred (P2-07). |
| Allocation | `src/firm/allocation/carver_sleeve.py` (P6) | One SLEEVE_REGISTRY entry; `Allocator.plan` unchanged. |
| Ledger (canonical) | `/local/store/research-ledger/trials.jsonl` + `returns/<trial_id>.parquet` + `inbox/` (P1-12 capture lines) | Host-level, outside every worktree, owner-provisioned (P1-01), flock-serialised, optional `chattr +a` (OD-10). Returns parquet never enter git (licensed-data derivatives); git holds `research/ledger/returns_manifest.jsonl` only. |
| Tracked artefacts | `research/` : `ledger/` (mirror, `legacy_backfill.csv`), `preregistration/` (`INDEX.yaml`, new-family YAML), `charters/`, `lifecycle/registry.yaml`, `approvals/`, `reports/`, `validation/`, `data_manifests/`, `data_qa/`, `monitoring/` (non-data artefacts only). `research/monitoring_sealed/` exists on the host but is NOT tracked: gitignored (P0-06) and ACL-restricted (OD-05), because a tracked post-seal file would be checked out into every research clone | `runs/` is gitignored and holds only 9/25-9/28 runs, so the source's `runs/approvals|research|monitoring` become `research/...`. Daily monitor state: `data/forward_monitors/<name>/`. |
| Plan files | `PLAN.md` (repo root, this file), `plan/OWNER_DECISIONS.md`, tickets in `plan/tickets/<ID>.md`, agent drafts of owner-committed files in `plan/drafts/<ID>/` | Decision: PLAN.md stays at the repo root; every plan file now cites `PLAN.md` (reconciled 2026-10-02). The `docs/README.md` index line is added by P0-01 with the root path. |
| Config (flat) | `config/{research_freeze,gates,costs,risk,kill_switch,tax_il,stress_periods,universe_etf,universe_futures}.yaml` | New candidate configs never touch `live.yaml`, `live_alpaca.yaml` or `llm_ab_*.yaml`. |
| Docs (top level) | `docs/{REPO_MAP,HOLDOUT_POLICY,DEPRECATIONS,LLM_POLICY,JURISDICTION_NOTES,GUARDRAIL_REDTEAM,legacy_trial_census_2026_10,gate_deviation_register_2026_10}.md`, each indexed in `docs/README.md` | |
| Tests | Flat `tests/test_*.py`; `tests/integrity/` (CODEOWNERS-protected) and `tests/data_qa/` (synthetic fixtures) get `__init__.py` | The suite has 3402 tests; none are moved. |
| New-family harnesses | `scripts/<family>_preregistered.py` + `docs/<family>_trial_history.json`, also registered in the ledger and `INDEX.yaml` | Repo convention. |

Ledger mirroring: a serial owner/ops sync step (`scripts/sync_ledger_mirror.py`) copies the host `trials.jsonl` into tracked `research/ledger/`, writes the returns manifest (sha256, n_obs per parquet) and ingests the P1-12 `inbox/` lines idempotently. CI checks that each commit's mirror is a strict prefix-extension of the previous one. Parallel or abandoned worktrees therefore cannot fork the hash chain.

## 4. Phase overview

| Phase | Goal | Depends | Exit gate | Human checkpoint |
|---|---|---|---|---|
| P0 Freeze and cleanup | Declare history burned, seal forward data (ACL + guard), registry statuses and DEPRECATIONS, guardrails, census, frozen `gates.yaml` | none | H0: integrity tests green; seal in place and red-team passed (incl. `settings.local.json` bypass and a check that a root ops session can still read live state), census signed, statuses and DEPRECATIONS merged (live removal deferred to P0-07), gates frozen | Yes: OD-04..OD-09, OD-16, census sign-off |
| P1 Validation infra | Host ledger, PSR/DSR/MinTRL, PBO S=16, purged/CPCV, RC/SPA/RW, nulls, synthetic harness | P0 (P1-02/05/07 start at P0-06) | H1: legacy backfill loaded; P1-08 meets pre-set size/power rates (size <= 0.07 at alpha 0.05, K=50, 500 sims; power >= 0.80 at SR 1.0/yr over 10y daily for K=1, and >= 0.80 at K=50 with the true strategy at SR >= 1.3/yr, or within 0.05 of a simulated oracle max-t test; PBO 0.5 +/- 0.1 under noise and < 0.1 under a strong drift (pinned scenario frozen in `gates.yaml` `p1_08_acceptance.scenarios`: N=30, SR 3.0/yr, T=1600, S=16, mean over >= 20 seeds) with the oracle expectation recorded; DSR false-pass <= 5% plus the two-sided calibration in G-RESEARCH 1). **[ADJ]** The power and PBO-under-drift rates were recomputed: SR 1.0/yr, K=50, 10y daily gives max-t power of only about 0.53-0.55 (analytic Bonferroni vs simulated oracle) and mean CSCV PBO of about 0.25-0.30 (P1-08 scratch runs), so source-style rates cannot be met by a correct implementation (deviation register row) | Yes |
| P2 Data and costs | ETF universe, loader, cleaning v3, QA, cost model, tax model, jurisdiction memo | P0 (P2-08, P2-06 start early) | H2: universe chosen without performance input; data QA green and manifests hashed; cost model reconciled to fee schedules; after-tax report runs on the benchmark; cost and tax models sensible | Yes (OD-01, OD-12, OD-14) |
| P3 Core strategies | Vol, EWMAC, breakout, (carry code), combine, sizing, vector engine, stress suite, constants, core_v1 run | P1, P2, charter | H3: honest G-RESEARCH verdict (Tier A/C/D) | Yes (P3-08 review 150 min) |
| P4 Portfolio and risk | Weights, diversification report, vol targeting/limits, tiered kill switch, Kelly bound | P3-09 and inputs | H4: risk limits fault-tested; ENB across asset classes >= 2.5 (ETF) or >= 3 (futures); a miss stops the candidate | Yes |
| P5 Lifecycle and monitoring | Charters, state machine, decommission combiner, CUSUM, fidelity monitor, LLM policy, allocation forward-test monitor | P1, P4 | H5: charters approved; lifecycle enforcement shown by a dry-run Allocator test; monitors alert on injected faults | Yes (charters 120 min) |
| P6 Paper | Dedicated candidate instance, opt-in engine additions, 6+ month paper period | P5, Tier A only | H6: G-PAPER met | Yes |
| P7 Small live | Capital selector, phased 25/50/100%, annual review | P6, P3-10 | H7.x: G-LIVE-STEP at each step | Yes, every step |

If the P3-08 verdict is Tier C or D, or the H4 ENB bar is missed, P5-01..P7 are not run for that candidate and the outcome is passive.

## 5. Global gates

The numbers below are the source B2 numbers. Where this plan changes them, the change is marked **[ADJ]** and is recorded in the gate deviation register for signature (OD-16) before `config/gates.yaml` is frozen (P0-08). Gates are fixed before results are seen.

### G-RESEARCH (candidate -> paper-eligible): all 8 must pass

1. **DSR >= 0.95.** Source: N = effective trials from the ledger including all legacy; SR variance across ledger trials of the same family. **[ADJ]** N = the raw count of ALL trials (returns-bearing, Sharpe-only and estimate-only legacy rows, plus counted unregistered and API-capture rows): about 460 to start, with sensitivity reported at 210. `effective_n` (ONC, participation ratio, Meucci ENB, MP edge; P1-03) is reported only, because every such estimate is <= the trial count it is computed on, so a `max(effective_n, raw)` rule would always equal the raw count and only look like it uses effective N. The source-faithful alternative (N = max over methods of effective_n on returns-bearing trials + raw count of Sharpe-only/estimate-only/unregistered/API rows) is offered to the owner in OD-09; P0-08 adds an integrity assertion that the gate's N formula matches the signed text. `var_sr` is the length-adjusted cross-trial variance: `var_sr_adj = max(0, var(SR_i) - mean(sampling_var_i)) + sampling_var(T_cand)`, with per-period Sharpes converted from the variance of the candidate grid and the `var_sr_family` (see below), sampling variance from the Lo/Mertens SE formula, T_i recorded for every family trial in the ledger, floored at the candidate grid's variance; the unadjusted value is a reported sensitivity (short-window legacy trials such as `trend` with 491 OOS days would otherwise inflate var_sr and over-deflate E[max SR]). Two named concepts are frozen in `gates.yaml` before results: `var_sr_family` (legacy `trend`, alt_premia T2 and C1, S3 (5 trials), futures_trend if ever run, and all core_v1 grid trials, used for variance) and `prereg_family` (`core_v1`, used by gate 8). DSR with all-family variance is a reported sensitivity. **Implied minimum Sharpe (owner must see before signing OD-16):** at N=460 with var_sr near 7e-4 (SD about 0.027 per day) and T about 5000, DSR >= 0.95 needs about 0.104/day, roughly 1.65/yr annualised (about 1.05/yr even for a perfectly length-matched null); at N=210 slightly lower. The plan's own expectation is 0.3-0.6 (section 1), so Tier A is effectively out of reach unless the real Sharpe is far above expectation; this is accepted in advance. **Calibration (two-sided):** with N = K and var_sr = 1/T, PSR(0) of a single null series is roughly Uniform(0,1) (KS p > 0.01), and the DSR pass rate in a strong-drift scenario is within Monte Carlo error of a simulated oracle. The new `dsr()` raises rather than silently falling back to PSR(0).
2. **PBO < 0.30 via CSCV, S = 16;** source "prefer < 0.20 if selected from > 20 variants". **[ADJ]** The 0.20 threshold is mandatory when the family ledger count exceeds 20 variants (it does, counting all trend-family trials). With effective grid N <= 3 (P1-03 clustering or participation ratio on the grid return matrix, not the nominal config count), PBO is reported as uninformative (Tier C) and cannot support a pass. The pre-registered grid must vary dimensions that change the return stream (cap, buffer, speed subset); tau has almost no effect on net Sharpe when the gross cap does not bind. Known tension: gate 6 wants flat Sharpe across perturbations while CSCV on near-equal variants tends to PBO 0.5 (the P1-04 noise test); this is why PBO alone cannot force Tier D (see verdict vocabulary) and is recorded in the register for the owner to sign knowingly. Legacy `cscv_pbo` keeps S = 8 for frozen consumers.
3. **Purged/embargoed CPCV:** median OOS path Sharpe > 0 and at least 75% of paths positive. **[ADJ]** With N = 10 groups and k = 2 there are 9 paths; the requirement is at least 7 of 9 (78%), since 75% of 9 is not an integer. Applied to the pre-registered grid only: within each of the 45 splits the pre-registered selection rule (argmax in-sample Sharpe over the grid) is applied on the purged training groups, and each path is built from the selected config's OOS returns on the test groups. The P3-11 constants (scalars, FDM, IDM, speed filter) are re-estimated per training split where cheap; otherwise the leakage (they saw all pre-seal data) is listed as a known limitation. A test asserts the 9 paths are not all identical when the grid has more than one distinct config.
4. **Cost stress:** net Sharpe > 0 and DSR >= 0.90 at 2x costs; 3x reported, not required. **[ADJ]** Multipliers scale the full `firm.costs` CostBreakdown (commission, fees, half-spread, impact, roll). Flat-bps results are shown for comparison only.
5. **Stress suite:** no named episode loses more than 1.5x the charter's pre-committed max-DD; every episode reported; never used to tune. **[ADJ]** The charter max-DD is a procedure fixed before any backtest: stationary block bootstrap with Politis-White block length, recorded seed, 10,000 draws, 95th percentile, on the selected config at 1x cost, with a FIXED path length of 10 years (matching source B0's 'over a decade'; recorded in `max_dd_procedure`), because expected max-DD grows with horizon and a full 16-30 year history would loosen both the stress gate and the kill switch. The 2.5 tau figure is a rule of thumb (plan choice), not an analytic bound from source A9 (A9 implies about 1-1.3 tau); it is reported alongside. For this gate each episode is compared against the bootstrapped p95 max-DD over windows of the same length as the episode (or, if 'stricter' is intended, min(p95_10y, 2.5 tau)); the P5-02 procedure states which. Each episode reports how many instruments were active (many ETFs only become usable after the 2520-day vol warm-up). All ten periods are pre-seal.
6. **Parameter robustness:** net Sharpe within 30% of the chosen value under +/-25% perturbation of every free parameter. **[ADJ]** Every numeric core parameter, no exemptions (listed in `gates.yaml`: vol span 35, 70/30 blend, long window 2520, vol-floor percentile, EWMAC speeds and 4x ratio, forecast cap, breakout N set and N/4 smoothing, FDM/IDM caps, buffer 0.1, tau, gross cap, speed-cost cutoff, plus the P4-03 portfolio parameters `vol_ewma_span`, `max_vol_scale` 1.5 and the 2x instrument risk-cap multiple). The fraction of days at the gross cap must stay <= 20% (OD-17).
7. **Benchmark:** after-tax, after-cost Sharpe >= passive benchmark, or correlation <= 0.3 with a documented diversification rationale. **[ADJ]** Benchmark is BM2 on pre-seal data (annual-rebalance after-tax variant, see section 1). Both branches are kept; a power analysis (minimum detectable Sharpe gap) is mandatory and reported. Decision table, tested row by row in `gate_report`: branch (a) passes on the POINT ESTIMATE (source wording), and is marked Tier C only if it passes by less than a pre-set margin (with about 20 years the minimum detectable gap at 80% power is about 0.68 at rho = 0.3, so branch (a) is nearly always underpowered); (a) failing on the point estimate with correlation > 0.3 is Tier D; correlation <= 0.3 with a documented charter rationale passes. The `core_only_100` replay is a second reference. The live Alpaca book is excluded (it is post-seal).
8. **Mechanism:** economic mechanism in the charter, human-written and approved before any backtest. **[ADJ]** CI checks that the charter's git timestamp precedes the first ledger row carrying the `core_v1` prereg_id (`prereg_family`, which includes the P3-11 estimation trials), not the first trial of the wider trend variance family. A gate 8 failure is Tier D (or blocked/not evaluable).

**Verdict vocabulary [ADJ].**
- Tier A, "pass": all 8 tests pass as specified. Only Tier A is paper-eligible.
- Tier D, "fail": any point-estimate failure: SR <= 0, PBO >= 0.5 AND prob_oos_loss >= 0.5 (or the selected config's median OOS Sharpe <= 0; PBO >= 0.30 or >= 0.20 on its own is Tier C, because PBOResult is read together with degradation and prob_oos_loss per Bailey et al. 2017), median CPCV path <= 0, net Sharpe <= 0 at 2x costs, a stress or robustness breach, test 7 failing the point-estimate branch with correlation > 0.3, or gate 8 failing. An uninformative PBO is Tier C.
- Tier C, "insufficient evidence": no point-estimate failure, but a confidence or power threshold is missed: DSR < 0.95, PBO between 0.30 and 0.5 (0.20 and 0.5 above 20 variants), fewer than 7 of 9 CPCV paths positive, or DSR at 2x below 0.90.
- Tier C and Tier D both lead to the passive outcome (BM2, or keeping the current Alpaca 92/8 book while noting its 8% BTC satellite is active, not passive).

### G-PAPER (paper -> live-eligible): dedicated candidate account only

1. At least 6 calendar months and at least 26 rebalances, whichever is later. **[ADJ]** "Rebalance" means a scheduled weekly review event (traded or not): 26 review events and at least 6 months, starting after a 2-week EMBARGO.
2. Daily paper vs shadow-replay correlation >= 0.95; annualised tracking error <= 25% of target vol.
3. Realised costs <= 1.5x modelled, median per trade and aggregate (commission included). **[ADJ]** Evaluated only after at least 30 fills across at least 10 instruments; paper is extended until reached.
4. Zero unreconciled position breaks longer than 1 trading day; zero unexplained missed rebalances. **[ADJ]** A stuck working order that blocks a review counts as missed unless explained; durable records are kept.
5. All kill switches fault-injection tested during paper. **[ADJ]** Includes the decommission combiner; drawdown is measured from an immutable HWM series; precedence against `RiskAgent._drawdown_breaker` and the engine kill switch is defined.

The allocation forward test on Alpaca is governed only by its own frozen I1-I5 and BAR_SLEEVE_DRIFT bars, reported side by side with G-PAPER and never merged.

### G-LIVE-STEP (25% -> 50% -> 100%)

At least 3 months per step; G-PAPER 2-4 hold on live fills; drawdown within the envelope (CUSUM not alarmed); human sign-off committed in `research/approvals/` (source said `runs/approvals/`). **[ADJ]** The P3-10 forward-holdout check must have passed before step 1 (OD-18). It is a no-contradiction check against pre-set criteria and is labelled that way.

### G-DECOMMISSION (implemented in P5-01)

Triggers: (1) drawdown > 1.5x the charter max-DD; (2) CUSUM alarm; (3) a G-PAPER 2 or 3 fidelity breach in 2 consecutive months; (4) mechanism invalidated (owner decision). One trigger moves the strategy to PROBATION and opens a review. Two set the target weight to 0 automatically and require approval to resume. **[ADJ]** For survival thresholds (kill switch, decommission) the reference is max(bootstrap p95, 2.5 tau), so a lucky backtest cannot set a hair-trigger. The two directions (stricter for the stress gate, looser for survival) are deliberate and recorded in OD-16.

---

## 6. Critical path

**Phase 0 (H0).** P0-01 -> P0-06. After P0-06, in parallel: P0-02 (LIVE-IMPORT-PATH), P0-03 (LIVE-IMPORT-PATH), P0-05 (census). P0-02 -> P0-04 -> [owner signs OD-16] -> P0-08 (also needs P0-05).

**Phase 1 (H1).** Starts at P0-06, overlapping P0: P1-02, P1-05, P1-07 in parallel. After P1-02: P1-04. After P1-07: P1-06 and P1-10 (LIVE-IMPORT-PATH). P0-05 and P0-04 -> P1-01 -> P1-12 (LIVE-IMPORT-PATH; also needs P0-04, your acknowledgement that `api/app.py` joins the LIVE-IMPORT-PATH set, and your provisioning of the ledger root and `inbox/`), with P1-03 (also needs P1-02) and P1-11. P1-09 needs P1-01, P0-04, P0-08. P1-08 needs P0-08 and all stats modules.

**Phase 2 (H2).** In parallel with P1: P2-08 from P0-06; P2-06 from P0-01; P2-01 needs P0-02 and P0-04; then P2-01 -> P2-02 (also needs P2-08 and P0-02's `data_access.read_parquet`) -> P2-03 and P2-01 -> P2-04. P2-05 needs P2-02, P2-04, P2-06, P1-07.

**Early code (synthetic fixtures only).** P3-01 from P0-06 -> P3-02 (also needs P2-04), P3-03, P3-04 -> P3-05 -> P3-06 -> P3-09 (needs P1-01, P0-02, P2-02). P4-01, P4-03, P4-05 in parallel once inputs land; P4-02 needs P1-03, P1-07, P2-05, P3-09; P4-04 needs P4-03, P5-02, P1-10, P2-03 (special-closures table). P5-02 charters are drafted during P1/P2 and approved by the owner; mechanism sections are human-written.

**Real-data gate (H3, H4).** P3-11 (constants) needs approved P5-02, P1-09, P0-08, H1, P2-03, P3-09. P3-07 needs P3-09. P3-08 needs P3-11, P3-07, P4-01..P4-05, P2-05, P1-12 and delivers H3 and H4 together. Tier C/D or ENB miss: stop, passive outcome.

**If Tier A (H5, H6, H7.x).** P5-03 -> P5-01 and P5-04 (needs P5-06). 2-week EMBARGO -> P6-03 -> P6-01 (LIVE-IMPORT-PATH; preferably after the allocation forward test's 6-month checkpoint, about 2027-03-30) -> P6-02 (6+ months) -> H6. Then P3-10 (no earlier than 2027-10-01), OD-20, P7-01 decision, P7-02 steps.

**Off the critical path.** P5-06 (after P0-06; agent builds with synthetic tests, you run the real-data steps; gated by OD-15, OD-19 and OD-05; worth starting soon, since the frozen I1-I5 bars are computed by nothing), P5-05 (after P0-03), P1-10, P1-11, P2-06, P7-01 tool (after P2-01, P2-02 and P1-03; it builds its own frozen input file through `data_access` as an added mode of `scripts/select_instruments_for_capital.py`). Deferred: P0-07 (OD-03), P2-07 (OD-01/OD-02), P3-10 (calendar), P7-02, P7-03.

## 7. Human checkpoints H0..H7.x

| Checkpoint | What you confirm |
|---|---|
| H0 | Forward data sealed by ACL plus guard (historical holdout declared burned); guardrail red-team passed, explicitly including the `settings.local.json` / `disableAllHooks` bypass and shell-redirect writes; legacy trial count not understated (census signed, Gann at about 145, not 8); registry statuses and `DEPRECATIONS.md` merged; `gates.yaml` frozen (deviation register signed). |
| H1 | Stats modules meet the pre-set synthetic size/power rates (P1-08). |
| H2 | Universe chosen without performance input; cost and tax models sensible. |
| H3 | G-RESEARCH report; honest verdict accepted, including "use passive". |
| H4 | Risk limits fault-tested; diversification report with ENB >= 2.5 (ETF) or >= 3 (futures). |
| H5 | Charters approved (mechanism human-written); lifecycle enforcement shown by the dry-run Allocator test. |
| H6 | G-PAPER evidence on the dedicated candidate account. |
| H7.x | Each live capital step (25%, 50%, 100%), each with explicit sign-off in `research/approvals/`. |

Review burden is about 2,860 minutes (about 48 hours) across all tickets. The largest single reviews: P3-08 (150), P0-05 (120), P5-02 (120), P6-01 (120), P7-03 (120), P6-02 (240), P7-02 (180).

## 8. Coexistence with live services and running forward tests

**What runs from this checkout (Restart=always).** IBKR `:8000`: blended 11-strategy pipeline, LLM arm B (`FIRM_LLM_CONFIG=config/llm_ab_llm.yaml`). Alpaca `:8001`: allocation mode, 60/40 core plus BTC 4-week trend (92/8 book) since 2026-09-30, `FIRM_ENABLE_PATTERN_SCAN=1`. The S2 shadow forward test (run manually so far; its timer `deploy/s2-forward-shadow.timer` is not installed) and the allocation forward test are active. `firm.api.app` imports `firm.eval` (robustness, overfitting) and `firm.patterns.significance` at startup, and `live/engine.py` imports `runtime` and `pit_store` at module top, so all of these are loaded in both running services. A merge to main that touches them takes effect at the next process start (crash, Restart=always or reboot). Request handlers may still lazily import other modules, so a running process can hold old code for some modules and new code for others (the 9/28 skew class). `firm.eval.*` is therefore already in the live graph and is not in the P0-06 forbidden list.

**Rules for every ticket.**
- Research agents (separate unix user, OD-07) work in a separate clone OUTSIDE the live checkout (`/local/store/research/ai-trading-system`, owned by the research user; one worktree per ticket at `<clone>/.claude/worktrees/<ID>`, created by `scripts/new_research_worktree.sh`, P0-06), pushing branches to the remote under the OD-06 machine identity, with no write permission on the live checkout or its `.git` (git worktrees of the live checkout share its refs, config and hooks, so a worktree there could move `refs/heads/main` of the tree the services run from). The owner merges and pulls into the live checkout in acknowledged windows. Research venvs live at `/local/store/research-venvs/<depset>`, are NON-editable installs, and are used with `PYTHONPATH=<worktree>/src`; the Stop hook fails closed (non-zero exit) in research sessions when no safe interpreter is found. Share one research venv per dependency set; nothing is ever pip-installed into the live `.venv` (its editable install points at the live `src`).
- Never edit the live checkout in place. Never read `.env` or live state files. No systemctl, no restarts, no edits to `config/live*.yaml`, `llm_ab_*.yaml` or deploy units except in touches-live tickets you sign off.
- At most 3 concurrent agents. `pytest` runs under `nice -n 10` and `ionice -c3`; xdist `-n 2` at most, and only in the shared research venv, where P0-06's worktree script installs `pytest-xdist` (it is not in `pyproject.toml` and never goes into the live `.venv`). Check at least 20 GB free disk before creating a worktree. Heavy runs (P1-08, P3-08, P3-11) stay outside 09:15 ET and the allocation rebalance window.

**LIVE-IMPORT-PATH protocol** for P0-02, P0-03, P1-10, P1-12 (behaviour-neutral by design, deploy-class by risk); P6-01 and P7-02 follow it too, on top of their own touches-live sign-off. Each must: (1) be self-contained, with no new cross-module symbols that already-loaded modules would need; (2) pass the full suite in the worktree venv; (3) pass `scripts/live_import_smoke.py` in a fresh interpreter (`FIRM_DATA_DIR` = tmp, `FIRM_AUTO_START_LIVE` unset, no broker connect, no network; imports `firm.api.app` and `firm.live.engine`, runs `build_orchestrator` for `config/live.yaml` and `config/live_alpaca.yaml` via `resolve_live_startup`); (4) merge only in a window you acknowledge, outside the 09:15 ET planning cycle and both rebalance windows; (5) in the same acknowledged window, the owner (or an ops session with an explicit go-ahead) runs a controlled restart of both units, then checks `GET /api/live/status` on :8000 and :8001, confirms `kill_switch_state` is unchanged and the engines are running, and confirms the next cycle completes without errors in the logs (otherwise Restart=always would first load the new code at an uncontrolled time, possibly mid-cycle); (6) a pre-written rollback (git revert of the merge commit plus a controlled restart) with a trigger condition (engine not running, kill-switch state changed, errors in the next cycle).

**Isolation of new code.** Everything else in P0-P5 is new modules no live code imports. P0-06's subprocess test lists them by name, is CODEOWNERS-protected, and changes only in P6-01 under your review.

**Tickets that change running services:** P0-07 (deferred, OD-03), P6-03 (new `:8002` instance and nginx), P6-01 (new registered sleeve and opt-in engine additions, controlled restart), P6-02 (paper period), P7-02 (real capital behind `FIRM_ALLOW_TRADING`, OD-20). Unchanged by this plan: the IBKR 8% and Alpaca 25% kill-switch regimes, and `capital_gate.py`.

**IBKR `:8000`** configuration and strategy set stay untouched through P0-P5 as the blended control; four LIVE-IMPORT-PATH tickets change modules it imports, behaviour-neutral by design: P0-02 (`data/pit_store.py`, `runtime.py`), P0-03 (`strategies/registry.py`, `runtime.py`, `live/engine.py`, `live/pipeline_warmup.py`, `api/routers/live.py`, `api/routers/meta.py`), P1-10 (`eval/robustness.py`) and P1-12 (`backtest/run.py`, `runtime.py`, `experiments/runner.py`, `api/app.py`, new stdlib-only `backtest/_capture_state.py`). `runtime.py` is edited by three of them, so their merges are serialised. Registry statuses are hard-enforced only in new research and backtest entry points. In live and API fallback paths archived names are warn-and-skip (WARNING log), which changes behaviour only if a config omits its strategy list. The 11 IBKR-enabled strategies carry a warn-only LEGACY_LIVE status. `list_strategies()` is unchanged; `list_allocatable()` is added.

**Alpaca `:8001` is not a candidate account.** It is the first real dataset for the fidelity tooling: P5-06 implements the frozen I1-I5 bars by calling `scripts/allocation_replay.replay()` directly. P6-01 must not change the behaviour of `Allocator.plan`, existing sleeves or anything else `allocation_replay` imports while the forward test runs. The only permitted edit to an imported module is one additive `SLEEVE_REGISTRY` entry in `sleeves.py`, and the golden replay test proves it has no effect; a golden test must show identical plans and orders for `live_alpaca.yaml` over the 2015-2026 replay. Candidates paper-trade on a separate account and instance (P6-03).

**Post-seal data on this host** exists today in `data/cache`, `data/`, `data_alpaca/`, `data/research/s2_forward/`, `docs/s2_forward_snapshot.json` (tracked) and logs; it will also exist in `data/forward_monitors/` and `research/monitoring_sealed/` once P5-06/P5-04 land, and API backtest captures (P1-12) go to the owner-provisioned `/local/store/research-ledger/inbox/` (one stdlib JSON line per backtest, outside both `FIRM_DATA_DIR`s, ingested by the owner/ops sync step), so neither the capture nor the sync needs read access to broker-state directories. `docs/HOLDOUT_POLICY.md` lists each. The seal rests on a non-root research user with no read access to those paths (OD-07), deny rules and the `deny_holdout` hook that live in that user's own root-owned, read-only `~/.claude/settings.json` (NOT in the committed `.claude/settings.json` or `/etc/claude-code/managed-settings.json`, which apply to root ops sessions too and would lock them out of `data_alpaca/kill_switch_state.json`, `data_alpaca/logs/` and live logs), the fail-closed `firm.research.data_access`, and default-None access-guard hooks in `pit_store.py` and `runtime.py` that only `firm.research.seal` installs. There is no env-var detection and live modules never import `firm.research`. **Residual risk:** the guard is fail-open for code that never imports `firm.research`. This is stated, recorded under OD-04, and mitigated by the ACL and by P1-12 marking such runs non-promotable. Human knowledge of post-seal market moves cannot be avoided, so every core_v1 choice must be frozen and dated in its prereg before P3-08 with a declaration that no post-seal data was examined.

**P5-06 is split by session type** (reconciled with the ticket): the agent builds the code and SYNTHETIC tests in a research session and never contacts :8001 or reads real outputs; the real dry run, the backfill from 2026-09-30, the output review, the weekly `export` of sealed snapshots and the timer install are done by you (owner identity, monitor exemption). The timer is installed only after code review and that dry run. **Timers.** The P5-06 monitor and P5-04 reconcile timers are installed only with your go-ahead (OD-15). The S2 timer remains your separate pending decision; S2 files and state are untouched and deny-listed for research.

## 9. Owner decisions

Full text, options, trade-offs and a blank for your decision are in [plan/OWNER_DECISIONS.md](plan/OWNER_DECISIONS.md).

| ID | One-line summary |
|---|---|
| OD-01 | Real investable capital: decides ETF vs futures path (recommend ETF path first regardless). |
| OD-02 | Buy futures history (Norgate/CSI/Databento)? Recommend defer. |
| OD-03 | Retire IBKR 11-strategy pipeline and LLM layer, and when? Recommend keep as control until P6. |
| OD-04 | Meaning of seal_date 2026-10-01 given post-seal data on the host (ACL-based, history declared burned). |
| OD-05 | Where post-seal data lives and who holds the unseal token. |
| OD-06 | CODEOWNERS and branch protection for a solo owner (second identity for agents). |
| OD-07 | Sandbox, deny settings and managed settings vs ops workflows; research user. |
| OD-08 | AGENTS.md/CLAUDE.md structure and scope of rules 8 and 10. |
| OD-09 | Canonical trial count N for DSR (about 460, sensitivity at 210). |
| OD-10 | Ledger format and location (host JSONL mirrored to git). |
| OD-11 | Candidate paper account and fidelity thresholds (separate `:8002` instance). |
| OD-12 | UCITS vs US-listed ETFs and Israeli tax treatment. |
| OD-13 | Is the frozen DRAFT futures_trend prereg superseded? |
| OD-14 | Freeze EODHD cleaning v3 now (additively). |
| OD-15 | Approve new systemd timers (P5-06 now; others per ticket). |
| OD-16 | Sign the gate deviation register (all numeric deviations from the source gates). |
| OD-17 | ETF-path target vol tau and the gross-cap ceiling. |
| OD-18 | Is a forward-holdout pass required before the first live step? Recommend yes. |
| OD-19 | Source of live NAV for the allocation monitor; monitor alerts only. |
| OD-20 | Real-money broker, account and host security posture. |

Decisions needed first, to unblock the early waves: OD-04, OD-05, OD-06, OD-07, OD-08 (P0-02, P0-04), OD-09 (census, ledger), OD-10 (ledger root and `inbox/` provisioning, P1-01, P1-12), OD-14 (cleaning v3), OD-15 and OD-19 (P5-06), then OD-16 (P0-08). OWNER_DECISIONS.md separates each decision's blocking tickets from the tickets that only reference it.

## 10. Ticket index

Links point to `plan/tickets/<ID>.md` (already drafted from the B3 template; P0-01 reviews them and reconciles them against REPO_MAP and the source tickets' interfaces, tests, DoD and DO NOT lines). "Touches live" means the ticket changes behaviour of, or is imported by, a running service.

| ID | Title | Phase | Status | Depends on | Touches live | Agent-h |
|---|---|---|---|---|---|---|
| [P0-01](plan/tickets/P0-01.md) | Repository inventory, path reconciliation, plan files | P0 | MODIFIED_SCOPE | - | no | 3.5 |
| [P0-06](plan/tickets/P0-06.md) | Worktree discipline, live-import isolation test, smoke, host resource policy | P0 | ADDED | P0-01 | no | 3 |
| [P0-02](plan/tickets/P0-02.md) | Discovery freeze and forward-data seal | P0 | MODIFIED_SCOPE | P0-01, P0-06 | YES (LIVE-IMPORT-PATH) | 6 |
| [P0-03](plan/tickets/P0-03.md) | Strategy lifecycle statuses and deprecation register | P0 | MODIFIED_SCOPE | P0-01, P0-06 | YES (LIVE-IMPORT-PATH) | 5 |
| [P0-04](plan/tickets/P0-04.md) | Agent guardrails: AGENTS/CLAUDE.md, settings, hooks, CODEOWNERS, integrity CI | P0 | MODIFIED_SCOPE | P0-02 | no | 6 |
| [P0-05](plan/tickets/P0-05.md) | Legacy trial census (CSV for sign-off) | P0 | EXTEND_EXISTING | P0-01 | no | 5 |
| [P0-08](plan/tickets/P0-08.md) | Gate deviation register and frozen config/gates.yaml | P0 | ADDED | P0-04, P0-05 | no | 3 |
| [P0-07](plan/tickets/P0-07.md) | Live disposition of IBKR strategies, LLM layer, pattern scan | P0 | DEFERRED | P0-03, P5-05 | YES | 2 |
| [P1-01](plan/tickets/P1-01.md) | Host-level hash-chained trial ledger, git mirror, legacy backfill | P1 | NEW | P0-05, P0-06, P0-04 | no | 10 |
| [P1-02](plan/tickets/P1-02.md) | PSR, DSR, MinTRL summary-stat API | P1 | EXTEND_EXISTING | P0-06 | no | 4 |
| [P1-03](plan/tickets/P1-03.md) | Effective number of trials and diversification primitives | P1 | NEW | P1-01, P1-02 | no | 5 |
| [P1-04](plan/tickets/P1-04.md) | PBO via CSCV with PBOResult | P1 | EXTEND_EXISTING | P1-02 | no | 5 |
| [P1-05](plan/tickets/P1-05.md) | PurgedKFold and CombinatorialPurgedCV | P1 | EXTEND_EXISTING | P0-06 | no | 6 |
| [P1-06](plan/tickets/P1-06.md) | White RC, Hansen SPA, Romano-Wolf | P1 | NEW | P1-07 | no | 9 |
| [P1-07](plan/tickets/P1-07.md) | Shared null library and stationary bootstrap | P1 | EXTEND_EXISTING | P0-06 | no | 7 |
| [P1-08](plan/tickets/P1-08.md) | Synthetic GARCH-t validation harness (size/power) | P1 | NEW | P0-08, P1-01..P1-07 | no | 7 |
| [P1-09](plan/tickets/P1-09.md) | Pre-registration index plus new-family YAML preregs | P1 | EXTEND_EXISTING | P1-01, P0-04, P0-08 | no | 6 |
| [P1-10](plan/tickets/P1-10.md) | Opt-in block bootstrap in MonteCarloAnalyzer | P1 | ADDED | P1-07 | YES (LIVE-IMPORT-PATH) | 1.5 |
| [P1-11](plan/tickets/P1-11.md) | Review-package integration of ledger, preregs, reports | P1 | ADDED | P1-01, P1-09 | no | 2 |
| [P1-12](plan/tickets/P1-12.md) | Entry-point trial capture for unregistered and API backtests | P1 | ADDED | P1-01, P0-04 | YES (LIVE-IMPORT-PATH) | 5 |
| [P2-01](plan/tickets/P2-01.md) | Instrument universes (ETF now, futures draft) | P2 | MODIFIED_SCOPE | P0-02, P0-04 | no | 5 |
| [P2-02](plan/tickets/P2-02.md) | ETF point-in-time loader, FX/CPI series, data manifests | P2 | MODIFIED_SCOPE | P2-01, P2-08, P0-02 | no | 8 |
| [P2-03](plan/tickets/P2-03.md) | Data QA suite | P2 | EXTEND_EXISTING | P2-02 | no | 6 |
| [P2-04](plan/tickets/P2-04.md) | Unified cost model with impact, roll, stress multiplier | P2 | NEW | P2-01 | no | 7 |
| [P2-05](plan/tickets/P2-05.md) | After-tax, after-cost benchmark reporting | P2 | NEW | P2-02, P2-04, P2-06, P1-07 | no | 7 |
| [P2-06](plan/tickets/P2-06.md) | Jurisdiction memo (Israel resident) | P2 | NEW | P0-01 | no | 3 |
| [P2-07](plan/tickets/P2-07.md) | Futures contract loader, back-adjust, roll calendar | P2 | DEFERRED | P2-01 | no | 12 |
| [P2-08](plan/tickets/P2-08.md) | EODHD cleaning v3, frozen additively | P2 | ADDED | P0-06 | no | 3 |
| [P3-01](plan/tickets/P3-01.md) | Blended EWMA vol estimator | P3 | NEW | P0-06 | no | 3 |
| [P3-02](plan/tickets/P3-02.md) | EWMAC trend forecasts | P3 | NEW | P3-01, P2-04 | no | 4 |
| [P3-03](plan/tickets/P3-03.md) | Breakout forecasts | P3 | NEW | P3-01 | no | 3 |
| [P3-04](plan/tickets/P3-04.md) | Carry forecast (pure function now) | P3 | MODIFIED_SCOPE | P3-01 | no | 3 |
| [P3-05](plan/tickets/P3-05.md) | Forecast combination and FDM | P3 | NEW | P3-02, P3-03 | no | 4 |
| [P3-06](plan/tickets/P3-06.md) | Position sizing, buffering, IDM | P3 | NEW | P3-05, P2-04 | no | 5 |
| [P3-07](plan/tickets/P3-07.md) | Named stress suite | P3 | NEW | P3-09, P0-02 | no | 4 |
| [P3-08](plan/tickets/P3-08.md) | Core v1 research run through G-RESEARCH (ETF path) | P3 | MODIFIED_SCOPE | P3-11, P3-07, P4-01..P4-05, P2-05, P1-12 | no | 10 |
| [P3-09](plan/tickets/P3-09.md) | Signed, multiplier-aware, cost-model vector engine | P3 | ADDED | P3-06, P2-04, P1-01, P0-02, P2-02 | no | 10 |
| [P3-10](plan/tickets/P3-10.md) | One-time forward-holdout evaluation of frozen core_v1 | P3 | DEFERRED | P3-08, P0-02 | no | 4 |
| [P3-11](plan/tickets/P3-11.md) | Pre-registered constant estimation on real data | P3 | ADDED | P5-02, P1-09, P0-08, P1-08, P2-03, P3-09 | no | 4 |
| [P4-01](plan/tickets/P4-01.md) | Handcrafted instrument weights with HRP diagnostic | P4 | NEW | P1-03, P2-01 | no | 4 |
| [P4-02](plan/tickets/P4-02.md) | Per-family return streams and diversification report (ENB exit) | P4 | MODIFIED_SCOPE | P1-03, P1-07, P2-05, P3-09 | no | 5 |
| [P4-03](plan/tickets/P4-03.md) | Portfolio vol targeting, leverage, exposure limits | P4 | NEW | P3-09 | no | 5 |
| [P4-04](plan/tickets/P4-04.md) | Tiered drawdown and operational kill switch (pure module) | P4 | EXTEND_EXISTING | P4-03, P5-02, P1-10, P2-03 | no | 5 |
| [P4-05](plan/tickets/P4-05.md) | Fractional Kelly sanity bound | P4 | NEW | P1-02 | no | 2 |
| [P5-01](plan/tickets/P5-01.md) | Lifecycle state machine, G-PAPER evaluator, decommission combiner | P5 | NEW | P0-08, P1-09, P0-03, P4-04, P5-03 | no | 9 |
| [P5-02](plan/tickets/P5-02.md) | Charter template and core charters | P5 | NEW | P0-04, P1-10 | no | 3.5 |
| [P5-03](plan/tickets/P5-03.md) | CUSUM decay monitor | P5 | NEW | P1-02 | no | 5 |
| [P5-04](plan/tickets/P5-04.md) | Fidelity monitor and daily reconcile for candidates | P5 | EXTEND_EXISTING | P5-03, P5-06, P3-09, P2-04 | no | 7 |
| [P5-05](plan/tickets/P5-05.md) | LLM policy document and registry statuses | P5 | MODIFIED_SCOPE | P0-03 | no | 2 |
| [P5-06](plan/tickets/P5-06.md) | Allocation forward-test I1-I5 monitor (synthetic build; owner runs real-data steps and installs the timer) | P5 | ADDED | P0-06 | no (reads live data under owner identity) | 6 |
| [P6-01](plan/tickets/P6-01.md) | Candidate order management via new registered Sleeve | P6 | EXTEND_EXISTING | P6-03, P5-01, P4-04, P3-06, P5-04 | YES (LIVE-IMPORT-PATH) | 18 |
| [P6-02](plan/tickets/P6-02.md) | Paper period: 6+ months, 26 review events, G-PAPER | P6 | MODIFIED_SCOPE | P6-01, P5-04, P5-03 | YES | 6 |
| [P6-03](plan/tickets/P6-03.md) | Provision candidate paper account and instance (:8002) | P6 | ADDED | P3-08, P5-01, P4-04 | YES | 4 |
| [P7-01](plan/tickets/P7-01.md) | Capital-path instrument selector | P7 | MODIFIED_SCOPE | P2-01, P2-02, P1-03 | no | 4 |
| [P7-02](plan/tickets/P7-02.md) | Phased real capital: 25%, 50%, 100% | P7 | DEFERRED | P6-02, P7-01, P3-10 | YES | 3 |
| [P7-03](plan/tickets/P7-03.md) | Annual review | P7 | DEFERRED | P7-02 | no | 4 |

## 11. Concurrency waves

After P0-06 start: P0-02 [after OD-04, OD-05], P0-03, P0-05 [after OD-09], P1-02, P1-05, P1-07, P2-08 [after OD-14], P2-06, P5-06 [synthetic build; real-data steps owner-run; after OD-15, OD-19, OD-05] and P3-01 (at most 3 agents at once, so these are queued, not simultaneous). After P1-02: P1-04. After P1-07: P1-06, P1-10. After P3-01: P3-02, P3-03, P3-04 (P3-02 also needs P2-04). P4-01, P4-03, P4-05 run in parallel once their inputs land.

## 12. Total effort

- To the H3/H4 verdict: **220.5 agent-hours**. P0 without P0-07: 31.5h; P1 including P1-12: 67.5h; P2 without P2-07: 39h; P3 including P3-11 but not P3-10: 50h; P4: 21h; P5-02, P5-05, P5-06: 11.5h.
- Full plan: **298.5 agent-hours**, adding the deferred P2-07 (12h), P3-10, P0-07, the Tier-A-only P5-01/03/04, P6 and P7.
- Human review: about **2,860 minutes (about 48 hours)**.
- Then 6+ months of paper and 6+ months of stepped live trading.
- With at most 3 concurrent agents, P0-P4 wall-clock is bounded by your sign-offs (OD-16, the census, P5-02 charters), not by agent time.

## 13. Caveats

From the source plan:
- Several academic references were cited from bibliographic knowledge and not re-fetched (Romano-Wolf, Harvey and Liu, Arnott/Harvey/Markowitz, Levine and Pedersen, Baltas and Kosowski, HRP).
- SG Trend figures come from manager-index and practitioner reports and carry index-construction biases.
- Carver thresholds are not peer-reviewed; the $72k minimum-capital figure is weak.
- Carry Sharpe is mostly pre-2012 in-sample; expect lower.
- Claude Code details change. Red-team permissions and sandboxing. GitHub issues report enforcement gaps (#24846: Read deny not enforced for `.env`; #61208: sandbox `denyRead` not working). Keeping sealed data physically away from the agent machine is the only fully reliable control.
- Tax figures are information only; verify with an Israeli tax professional.
- Failure is an acceptable outcome. The most likely honest result may be that the core fails G-RESEARCH or only matches passive after tax. That is a legitimate, money-saving outcome, not to be engineered around.

Repo-specific:
- There is no sealed historical holdout and none can be created; the seal covers forward data only and is ACL-based on this host (fail-open for code that never imports `firm.research`).
- CODEOWNERS and CI integrity checks are decorative unless a second identity exists for agents and branch protection plus a push-trigger CI are enabled (OD-06). CI is PR-only today and the owner commits to main directly.
- Agent guardrails in `.claude/settings.json` do not bind Cursor or an agent running as root with `git *` and `python -c` allowed; only root-owned managed settings or a separate unix user provide real separation (OD-07). Memory entries that grant "run on your own" autonomy and service restarts must be scoped so research sessions exclude restarts and live-config edits.
- Legacy trial counts are partly estimates (Gann about 145 vs the source's 8; total about 460 vs 210 ledgered). Per-trial return series for legacy trials were mostly not retained, so legacy rows carry counts and Sharpe only. The pattern_ml ledger counts folds x candidates (104), not configs (13); both are carried.
- Stale numbers inside frozen files (alt_premia note says combination = 52, actual 57; S-family prereg `prior_trials` placeholders 205/206/207; the repo's running total of 210 appears in no file) are noted in the census and never edited.
- `eodhd_clean.py` v2 misses the 999999.9999 sentinel; v3 must be frozen before any further EODHD-based test (OD-14).
- Futures plumbing is estimated at 34-56 person-days and futures data has not been bought; the plan runs the ETF core first. Futures margins in the integration sketch are unverified. Carry has never been tested in this repo, and the brief says FX and commodity carry decayed.
- The 12-month minimum before forward-holdout unseal means the earliest P3-10 date is 2027-10-01; with 12 months of data it can only reject a contradiction, not prove edge.
- Tax and jurisdiction material in the repo comes from one external brief that was not independently checked.
- Effort figures are agent-hour estimates, not measurements, and the futures estimates look low relative to the repo's own sketch.
