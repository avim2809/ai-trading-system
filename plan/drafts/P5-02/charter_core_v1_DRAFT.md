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

> **AGENT DRAFT (2026-10-04). Rewrite it in your own words, then delete this label; you are the author of record.**
> Trends in asset prices persist at horizons of about one to twelve months, and a portfolio that goes long what has been rising and steps aside from what has been falling captures that persistence while avoiding some of the deepest falls. Two explanations are usually given. Behavioural: investors under-react to news and then herd, so prices drift in the direction of the news before they fully adjust (Moskowitz, Ooi and Pedersen 2012). Risk transfer: hedgers and other holders who care about something other than return make trends worth earning for those who accept the risk of being wrong. The evidence is strongest for a *diversified, volatility-scaled portfolio* across many markets (Hurst, Ooi and Pedersen 2017: positive in every decade since 1880, and in 8 of the 10 largest 60/40 drawdowns). It is weak market by market (Huang et al. 2020), and much of the benefit comes from volatility scaling (Kim, Tse and Wald 2016). So this charter claims a modest, diversifying premium, not a per-asset forecast.

**1b. Who is on the other side of the trade, and why would they keep taking it?**

> **AGENT DRAFT. Rewrite in your own words.**
> The other side is made of participants who trade for reasons unrelated to expected return: hedgers and insurers who must hold or sell risk, index funds and rebalancers who buy after falls and sell after rises by mandate, and forced sellers (margin calls, redemptions, regulatory limits). They keep taking that side because their mandate or constraint compels them, not because they think it pays. In this long/flat ETF version there is no short leg, so the compensation is mostly the avoided drawdown and the diversification, not a short-side premium.

**1c. Why should it persist, and what would make it stop?**

> **AGENT DRAFT. Rewrite in your own words.**
> It should persist as long as those constraints and the behavioural slow-reaction exist, and that is structural rather than a one-off pattern. It can fade: published premia lose about half their return after publication (McLean and Pontiff 2016), crowding can compress it, and trend followers have had long bad stretches (the SG Trend Index's worst rolling 12 months was -18.6% to spring 2025, and multi-year flat periods are normal). It would stop working if markets stopped trending after costs for a decade, if the avoided-drawdown benefit disappeared because falls became fast reversals (March 2023, August 2024, April 2025 were hostile episodes), or if costs and taxes consume the small edge at this account size.

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
part of the ETF core. Rebalance frequency: weekly review with a no-trade buffer. Expected turnover and annual cost: **AGENT ESTIMATE, edit freely**: about 200-400% one-way turnover a year and about 10-20 bps a year
of cost on the Alpaca zero-commission basis (spread and regulatory fees only). A rough figure, not derived from any backtest; the real number comes from the cost model after the run.

## 3. Expected Sharpe, worst year, flat period  **YOUR CALL**

The plan's own range is 0.3 to 0.6 net, a haircut of at least 50% from published figures. Pre-filled from the plan; change if you disagree:
- Expected net Sharpe: **0.3 to 0.6** (estimate, not a forecast).
- Expected worst calendar year: **about -10% to -15%** (AGENT ESTIMATE: the plan's 20-35% max drawdown at 12-15% vol, scaled to tau 9%; guide: the SG Trend Index's worst rolling 12 months was -18.6%).
- Expected longest flat period: **3 to 5 years** (AGENT ESTIMATE; multi-year flat periods are normal for this style).

## 4. Target volatility tau (OD-17, decided)

**Working value 9%**, from your 8-10% decision: set ex ante from long-run asset volatilities with no performance input. The fraction of
days at the 1.0x gross cap must stay at or below 20% (gate 6). Confirm or change: tau = **9 %** (working value from your 8-10% decision; AGENT-filled, confirm)

## 5. Drawdown procedure and references (pre-filled from `config/gates.yaml`)

- Stationary block bootstrap, Politis-White block length, 10,000 draws, recorded seed, 2,520-day paths, 95th percentile, on the chosen
  config at 1x cost.
- Stress-gate limit: 1.5 x the episode-length bootstrap p95. Survival reference: `survival_dd = max(p95 at 2520 days, 2.5 x tau)` (2.5 tau is a rule of
  thumb, not an analytic bound). Hard decommission drawdown: 1.5 x `survival_dd`.

## 6. Decommission rules (pre-filled, matches G-DECOMMISSION)

One trigger (drawdown above 1.5x `survival_dd`, CUSUM alarm, a fidelity breach in 2 consecutive months, mechanism invalidated) puts the
strategy on PROBATION and opens a review. Two triggers set the target weight to 0 automatically; resuming needs your approval.

## 7. Correlation expectation and diversification rationale  **YOUR CALL**

Expected correlation with the 60/40 benchmark (BM2): **0.5 to 0.7** (AGENT ESTIMATE: a long/flat ETF trend book holds the same assets as 60/40, so
the correlation stays high). At that level gate 7's correlation branch (at most 0.3) is unlikely to apply, and the test rests on beating BM2 after
tax; no diversification rationale is claimed.

## 7b. What would falsify the mechanism  **YOUR WORDS**

> **AGENT DRAFT. Rewrite in your own words.**
> The mechanism is falsified if, over the pre-seal research window and then forward, (a) the long/flat portfolio's net Sharpe after cost and tax is not above the 60/40 benchmark's in a majority of non-overlapping decades; (b) its returns are explained by static equity-and-bond beta with no timing contribution (so the "trend" adds nothing beyond buy-and-hold); (c) drawdowns exceed the survival reference `survival_dd`; or (d) the premium is only visible in one market or one sub-period rather than across the diversified universe.

## 8. Declarations (confirm by keeping them)

- All data up to 2026-09-30 is in-sample; no post-seal data has been examined.
- Every `core_v1` choice (grid of at most 12 configs, constants, procedures) is frozen before the P3-08 run.
- Passing every gate shows no contradiction with an edge, not proof; the expected honest outcome at this capital is the passive portfolio.

## 9. Approval

Mechanism written by: ____________  Approved by: ____________  Date (UTC, `date -u`): ____________
