# Charter: core_v1 (ETF path, diversified time-series trend)  DRAFT FOR OWNER

**How to use this file.** Copy it to `research/charters/core_v1.md`. Sections marked **YOUR WORDS** must be written by you: gate 8 requires
a human-written mechanism, and its git timestamp must precede the first ledger row for `core_v1`. Everything else is pre-filled from
signed decisions and `config/gates.yaml`; check it and change anything you disagree with before you commit. Do not run any backtest
before this is committed.

Status: DRAFT (unsigned). Approved by: ____________  Date (UTC): ____________

---

## 1. Mechanism  **YOUR WORDS**

Write two or three sentences for each. No agent may write this section.

**1a. Why should a diversified, volatility-scaled trend portfolio earn a premium?** (who is willing to pay, what risk or behaviour creates it)

> ______________________________________________________________________________

**1b. Who is on the other side of the trade, and why would they keep taking it?**

> ______________________________________________________________________________

**1c. Why should it persist, and what would make it stop?**

> ______________________________________________________________________________

*Prompts, not answers.* Think about: investors who hold risk positions for reasons other than expected return (hedgers, index funds,
forced sellers); slow-moving information and behavioural under-reaction versus risk-premium explanations; why diversification
across many markets and risk scaling matter (the literature says per-market evidence is weak, the pooled portfolio is strong);
what the post-publication decay evidence implies.

## 1b. Evidence base (pre-filled; edit freely)

| Source | Finding | Grade |
|---|---|---|
| Hurst, Ooi & Pedersen (JPM 2017) | Diversified time-series momentum positive in every decade 1880-2016 across 67 markets; positive in 8 of the 10 largest 60/40 drawdowns | Strong |
| Moskowitz, Ooi & Pedersen (JFE 2012) | Time-series momentum across about 58 liquid futures | Strong |
| Huang, Li, Wang & Zhou (JFE 2020) | Little asset-by-asset evidence; pooled t-statistic below bootstrap critical values | Strong, contested |
| Kim, Tse & Wald (JFM 2016) | Much of the alpha is volatility scaling | Moderate |
| McLean & Pontiff (JF 2016) | Published predictors lose about 58% of returns after publication | Strong |
| Harvey et al. (JPM 2018); Cederburg et al. (JFE 2020) | Vol targeting cuts tails everywhere but raises Sharpe only for risk assets; implementable vol-managed alpha is weak | Strong |
| SG Trend Index (practitioner data) | About 5% a year since 2000; max drawdown 20.61%; worst rolling 12 months -18.6% (to April/May 2025) | Moderate |

## 2. Universe and instruments (pre-filled)

ETF path, long/flat, no leverage (gross at most 1.0x). Instruments and the written inclusion rationale for each:
`config/universe_etf.yaml` (15 ETFs across 10 cells). Chosen without any performance input (P2-01). Carry is futures-only and is not
part of the ETF core. Rebalance frequency: weekly review with a no-trade buffer. Expected turnover and annual cost: **YOUR NUMBERS**
(fill from the cost model `config/costs.yaml` once A4 is done): ______ % turnover, ______ bps a year.

## 3. Expected Sharpe, worst year, flat period  **YOUR CALL**

The plan's own range is 0.3 to 0.6 net, a haircut of at least 50% from published figures. Pre-filled from the plan; change if you disagree:
- Expected net Sharpe: **0.3 to 0.6** (estimate, not a forecast).
- Expected worst calendar year: ______ (guide: the SG Trend Index's worst rolling 12 months was -18.6%).
- Expected longest flat period: ______ years (guide: multi-year flat periods are normal for this style).

## 4. Target volatility tau (OD-17, decided)

**Working value 9%**, from your 8-10% decision: set ex ante from long-run asset volatilities with no performance input. The fraction of
days at the 1.0x gross cap must stay at or below 20% (gate 6). Confirm or change: tau = ______ %

## 5. Drawdown procedure and references (pre-filled from `config/gates.yaml`)

- Stationary block bootstrap, Politis-White block length, 10,000 draws, recorded seed, 2,520-day paths, 95th percentile, on the chosen
  config at 1x cost.
- Stress-gate limit: 1.5 x the episode-length bootstrap p95. Survival reference: `survival_dd = max(p95 at 2520 days, 2.5 x tau)` (2.5 tau is a rule of
  thumb, not an analytic bound). Hard decommission drawdown: 1.5 x `survival_dd`.

## 6. Decommission rules (pre-filled, matches G-DECOMMISSION)

One trigger (drawdown above 1.5x `survival_dd`, CUSUM alarm, a fidelity breach in 2 consecutive months, mechanism invalidated) puts the
strategy on PROBATION and opens a review. Two triggers set the target weight to 0 automatically; resuming needs your approval.

## 7. Correlation expectation and diversification rationale  **YOUR CALL**

Expected correlation with the 60/40 benchmark (BM2): ______ . If you expect it at or below 0.3, write the diversification rationale
here (it is what gate 7's correlation branch needs): ______________________________________________

## 7b. What would falsify the mechanism  **YOUR WORDS**

> ______________________________________________________________________________

## 8. Declarations (confirm by keeping them)

- All data up to 2026-09-30 is in-sample; no post-seal data has been examined.
- Every `core_v1` choice (grid of at most 12 configs, constants, procedures) is frozen before the P3-08 run.
- Passing every gate shows no contradiction with an edge, not proof; the expected honest outcome at this capital is the passive portfolio.

## 9. Approval

Mechanism written by: ____________  Approved by: ____________  Date (UTC, `date -u`): ____________
