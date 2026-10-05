---
# DRAFT charter template (P5-02). The owner copies it into the charters directory as TEMPLATE.md and commits it under CODEOWNERS.
# Machine-readable: firm.research.charter parses and validates this block. Values in <angle brackets> are the owner's to fill.
# Do not type any backtest-derived number here; the max-DD result is a derived ledger artefact (trial_id-referenced).
family: "<must match INDEX.yaml family, e.g. etf_trend>"
charter_version: 1
evidence_base: "<citations for the premium, human-written>"
rebalance_frequency: "<e.g. weekly review with a no-trade buffer>"
expected_turnover: "<annual one-way, ex-ante estimate>"
expected_annual_cost_bps: "<ex-ante estimate from the P2-04 cost model, no backtest performance input>"
expected_worst_year: "<ex-ante statement>"
expected_longest_flat_months: "<ex-ante statement; multi-year flat periods are normal for this style>"
correlation_expectations: "<vs the other core families and vs SPY/IEF; gate-7b rationale is conditional on this>"
falsification: "<one line: what observation would kill the mechanism (full text in section 7b)>"
gates_yaml_sha256: "<sha256 of config/gates.yaml at approval>"
mechanism_committed_at_utc: "<from git at approval, never from a log line>"
tau: "<annualised target vol as a fraction, set ex ante (OD-17 working value 0.09)>"
tau_derivation: "<path to the ex-ante vol note, no performance input>"
gross_cap: 1.0
gross_cap_bound_ceiling: 0.20
max_dd_procedure:
  method: stationary_bootstrap
  block_length: politis_white
  draws: 10000
  quantile: 0.95
  seed: "<int, recorded in the prereg>"
  path_length_days: 2520
  path_length_rationale: "<fixed by config/gates.yaml charter.max_dd_procedure (register row 11); one procedure serves stress and survival>"
  applied_to: selected_config_1x_cost
max_dd_analytic_2p5_tau: true
survival_reference: max(bootstrap_p95, 2.5*tau)
stress_reference: bootstrap_p95_at_episode_length
expected_sharpe_haircut_min: 0.5
combined_forecast_avg_abs_expected: 5
speed_weights_rule: equal_within_group_over_surviving_speeds
decommission:
  soft_dd_mult: 1.0
  hard_dd_mult: 1.5
  triggers: [dd, cusum, fidelity_2_months, mechanism]
approved_by: "<owner>"
approved_commit: "<sha>"
---

# Charter: <family>  (version 1)

Status: DRAFT (unsigned). Template sections in order; the machine-readable block above must agree with the prose.

## 1. Mechanism  **OWNER-WRITTEN**

<Economic reason the premium should exist, who is on the other side, why it should persist. No agent writes this section; its git
timestamp must precede the first ledger trial of the family.>

## 1b. Evidence base

<Citations, each with a finding and a grade. Published figures are fine here; backtest results of this family are not.>

## 2. Universe and instruments

<Link to config/universe_etf.yaml and docs/universe_rationale_2026_10.md. Rebalance frequency, expected turnover, annual cost in bps
(ex-ante, from the cost model, no performance input).>

## 3. Expected Sharpe, worst year, flat period

<Published or pre-seal-independent prior, minus a haircut of at least 50%. Expected worst calendar year and longest flat period.>

## 4. Target volatility tau

<Set ex ante from long-run asset volatilities, no performance input (OD-17). Fraction of days at the gross cap must stay at or below
the front-matter ceiling; exceeding it is a G-RESEARCH 6 failure.>

## 5. Drawdown procedure and references

- Stationary block bootstrap, Politis-White block length, recorded seed, `max_dd_procedure.draws` draws, `path_length_days`-day paths,
  95th percentile, applied to the selected config's 1x-cost backtest. The result is stored in the ledger, never typed here.
- 2.5 tau is a rule of thumb (source B0 range), NOT an analytic bound; it is reported alongside as a companion.
- `survival_dd = max(bootstrap_p95_dd, 2.5 * tau)`; `stress_dd = bootstrap_p95_dd`; G-RESEARCH 5 limit = 1.5 x `stress_dd`;
  hard decommission = 1.5 x `survival_dd`.

## 6. Decommission rules

One trigger (dd above 1.5x `survival_dd`, CUSUM alarm, fidelity breach in 2 consecutive months, mechanism invalidated) puts the
strategy on PROBATION and opens a review. Two triggers set the target weight to 0 automatically; resuming needs owner approval.
Matches G-DECOMMISSION in `config/gates.yaml`.

## 7. Correlation expectations and diversification rationale

<Expected correlation vs the other core families and vs SPY/IEF. If branch (b) of G-RESEARCH 7 (correlation <= 0.3) is to be used,
write the rationale here.>

## 7b. Falsification

<What observation would kill the mechanism. Its own section; the integrity test requires it.>

## 8. Declarations

- No post-seal data has been examined.
- Every choice of this family is frozen before the P3-08 run.

## 9. Approval

Mechanism written by: ____________  Approved by: ____________  Date (UTC, `date -u`): ____________
