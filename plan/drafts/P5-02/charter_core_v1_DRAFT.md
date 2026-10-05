---
# DRAFT charter for core_v1 in the validator's format (firm.research.charter). Copy to research/charters/core_v1.md, rewrite the parts marked
# OWNER, delete every "AGENT" label, fill the approval keys and commit. Estimates marked AGENT ESTIMATE come from the plan's cited sources,
# not from any backtest.
family: "core_v1"
charter_version: 1
evidence_base: "Hurst, Ooi and Pedersen (JPM 2017); Moskowitz, Ooi and Pedersen (JFE 2012); Huang et al. (JFE 2020); Kim, Tse and Wald (JFM 2016); McLean and Pontiff (JF 2016); Harvey et al. (JPM 2018); Cederburg et al. (JFE 2020); SG Trend Index practitioner data (section 1b)"
rebalance_frequency: "weekly review with a no-trade buffer"
expected_turnover: "about 200-400% one-way a year (AGENT ESTIMATE, edit)"
expected_annual_cost_bps: "10-20 on the Alpaca zero-commission basis (AGENT ESTIMATE, edit)"
expected_worst_year: "about -10% to -15% (AGENT ESTIMATE, edit)"
expected_longest_flat_months: "36 to 60 (AGENT ESTIMATE, edit)"
correlation_expectations: "0.5 to 0.7 versus the 60/40 benchmark; the gate-7 correlation branch is not claimed (AGENT ESTIMATE, edit)"
falsification: "No net-of-cost, net-of-tax advantage over 60/40 across non-overlapping decades, returns explained by static beta, drawdown beyond the survival reference, or a premium visible in only one market or sub-period (section 7b)"
gates_yaml_sha256: "a2ad524590376bb43fa0fd13ecc030181015a98575aeef8c7677fdaf623ed5be"
mechanism_committed_at_utc: "<from git at approval, never from a log line>"
tau: 0.09
tau_derivation: "plan/OWNER_DECISIONS.md (OD-17: 8-10% set ex ante from long-run asset volatilities, working value 9%; addendum 2026-10-05)"
gross_cap: 1.0
gross_cap_bound_ceiling: 0.20
max_dd_procedure:
  method: stationary_bootstrap
  block_length: politis_white
  draws: 10000
  quantile: 0.95
  seed: 20261005
  path_length_days: 2520
  path_length_rationale: "fixed by config/gates.yaml charter.max_dd_procedure (register row 11); one procedure serves stress and survival"
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

# Charter: core_v1  (version 1)

Status: DRAFT (unsigned).

## 1. Mechanism  **OWNER-WRITTEN**

> **AGENT DRAFT (2026-10-05). Rewrite it in your own words, then delete this label; you are the author of record.**
> Trends in asset prices persist at horizons of about one to twelve months, and a portfolio that goes long what has been rising and steps aside from what has been falling captures that persistence while avoiding some of the deepest falls. Two explanations are usually given. Behavioural: investors under-react to news and then herd, so prices drift in the direction of the news before they fully adjust (Moskowitz, Ooi and Pedersen 2012). Risk transfer: hedgers and other holders who care about something other than return make trends worth earning for those who accept the risk of being wrong. The evidence is strongest for a diversified, volatility-scaled portfolio across many markets (Hurst, Ooi and Pedersen 2017: positive in every decade since 1880, and in 8 of the 10 largest 60/40 drawdowns). It is weak market by market (Huang et al. 2020), and much of the benefit comes from volatility scaling (Kim, Tse and Wald 2016). So this charter claims a modest, diversifying premium, not a per-asset forecast.

> **AGENT DRAFT. Rewrite in your own words.**
> The other side is made of participants who trade for reasons unrelated to expected return: hedgers and insurers who must hold or sell risk, index funds and rebalancers who buy after falls and sell after rises by mandate, and forced sellers (margin calls, redemptions, regulatory limits). They keep taking that side because their mandate or constraint compels them, not because they think it pays. In this long/flat ETF version there is no short leg, so the compensation is mostly the avoided drawdown and the diversification, not a short-side premium.

> **AGENT DRAFT. Rewrite in your own words.**
> It should persist as long as those constraints and the behavioural slow-reaction exist, and that is structural rather than a one-off pattern. It can fade: published premia lose about half their return after publication (McLean and Pontiff 2016), crowding can compress it, and trend followers have had long bad stretches (the SG Trend Index's worst rolling 12 months was -18.6% to spring 2025, and multi-year flat periods are normal). It would stop working if markets stopped trending after costs for a decade, if the avoided-drawdown benefit disappeared because falls became fast reversals (March 2023, August 2024, April 2025 were hostile episodes), or if costs and taxes consume the small edge at this account size.

## 1b. Evidence base

| Source | Finding | Grade |
|---|---|---|
| Hurst, Ooi and Pedersen (JPM 2017) | Diversified time-series momentum positive in every decade 1880-2016 across 67 markets; positive in 8 of the 10 largest 60/40 drawdowns | Strong |
| Moskowitz, Ooi and Pedersen (JFE 2012) | Time-series momentum across about 58 liquid futures | Strong |
| Huang, Li, Wang and Zhou (JFE 2020) | Little asset-by-asset evidence; pooled t-statistic below bootstrap critical values | Strong, contested |
| Kim, Tse and Wald (JFM 2016) | Much of the alpha is volatility scaling | Moderate |
| McLean and Pontiff (JF 2016) | Published predictors lose about 58% of returns after publication | Strong |
| Harvey et al. (JPM 2018); Cederburg et al. (JFE 2020) | Vol targeting cuts tails everywhere but raises Sharpe only for risk assets; implementable vol-managed alpha is weak | Strong |
| SG Trend Index (practitioner data) | About 5% a year since 2000; worst rolling 12 months -18.6% (to April/May 2025) | Moderate |

## 2. Universe and instruments

ETF path, long/flat, no leverage. The frozen 14-ETF universe in 10 cells is in `config/universe_etf.yaml` with the rationale in
`docs/universe_rationale_2026_10.md` (chosen without performance input; VGK dropped at owner review). Carry is futures-only and is not part
of the ETF core. Rebalance: weekly review with a no-trade buffer. Expected turnover and annual cost: see the front matter (agent estimates;
the real figures come from the cost model after the run).

## 3. Expected Sharpe, worst year, flat period

The plan's own range is 0.3 to 0.6 net, a haircut of at least 50% from published figures (front matter keys carry the worst-year and
flat-period expectations). Passing every gate shows no contradiction with an edge, not proof.

## 4. Target volatility tau

Working value 9% from the owner's 8-10% decision, set ex ante from long-run asset volatilities with no performance input (OD-17). The
fraction of days at the 1.0x gross cap must stay at or below the front-matter ceiling (20%); exceeding it is a G-RESEARCH 6 failure.

## 5. Drawdown procedure and references

- Stationary block bootstrap, Politis-White block length, recorded seed, 10,000 draws, 2,520-day paths, 95th percentile, applied to the
  selected config's 1x-cost backtest. The result is stored in the ledger, never typed here.
- 2.5 tau is a rule of thumb (source B0 range), NOT an analytic bound; it is reported alongside as a companion.
- `survival_dd = max(bootstrap_p95_dd, 2.5 * tau)`; `stress_dd = bootstrap_p95_dd`; G-RESEARCH 5 limit = 1.5 x `stress_dd`;
  hard decommission = 1.5 x `survival_dd`.

## 6. Decommission rules

One trigger (drawdown above 1.5x `survival_dd`, CUSUM alarm, fidelity breach in 2 consecutive months, mechanism invalidated) puts the
strategy on PROBATION and opens a review. Two triggers set the target weight to 0 automatically; resuming needs owner approval. Matches
G-DECOMMISSION in `config/gates.yaml`.

## 7. Correlation expectations and diversification rationale

Expected correlation with the 60/40 benchmark (BM2) is 0.5 to 0.7 (AGENT ESTIMATE): a long/flat ETF trend book holds the same assets as
60/40. At that level gate 7's correlation branch (at most 0.3) is unlikely to apply; the test rests on beating BM2 after tax, and no
diversification rationale is claimed.

## 7b. Falsification

> **AGENT DRAFT. Rewrite in your own words.**
> The mechanism is falsified if, over the pre-seal research window and then forward, (a) the long/flat portfolio's net Sharpe after cost and tax is not above the 60/40 benchmark's in a majority of non-overlapping decades; (b) its returns are explained by static equity-and-bond beta with no timing contribution (so the trend rule adds nothing beyond buy-and-hold); (c) drawdowns exceed the survival reference; or (d) the premium is only visible in one market or one sub-period rather than across the diversified universe.

## 8. Declarations

- No post-seal data has been examined.
- Every choice of this family is frozen before the P3-08 run.

## 9. Approval

Mechanism written by: ____________  Approved by: ____________  Date (UTC, `date -u`): ____________
