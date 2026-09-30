# Research brief: tradeable return sources beyond US stocks

You are researching for the owner of a small, self-built systematic trading
system. Your job is to find out which **instruments and strategies beyond US
stocks** are worth adding. Back each recommendation with evidence from
trustworthy sources, a realistic cost picture, the service providers that would
make it work, and a concrete plan for plugging it into the existing code. **Say
plainly** when the honest answer is "not worth it".

## 0. Repository access

You have **read-only** access to the code through GitHub's web interface:
`https://github.com/avim2809/ai-trading-system` (branch `main`). You can't run
anything. Read the code to ground your answers, and **cite file paths** when you
make a claim about how the system works.

Read these first, in this order:

1. `CLAUDE.md`, then `docs/PROJECT_CONTEXT.md`: architecture, deployment, live configuration.
2. `config/live.yaml` and `config/live_alpaca.yaml`: universe, risk limits, costs, strategies.
3. The broker interface: `src/firm/brokers/base.py`. Every broker adapter implements it; `ibkr.py` and `alpaca.py` are the two existing examples.
4. The data layer: `src/firm/data/providers/` (one module per vendor, plus the fallback chain).
5. The instrument and position model: `src/firm/strategies/base.py` (`Signal`, `PitView`), `src/firm/portfolio/state.py`, `src/firm/agents/execution.py` and `src/firm/agents/risk.py` (sizing, costs, risk caps).
6. The backtest: `src/firm/backtest/engine.py` and `src/firm/backtest/run.py`.
7. The evaluation standard and past results:
   - `src/firm/eval/overfitting.py`, `scripts/combination_preregistered_bars.py`
   - `docs/optimal_combination_fix_2026_09.md`, `docs/formal_pbo_audit.md`, `docs/pattern_ml_final_verdict_2026_09.md`

**Not on `main` yet:** the most recent evaluation (§2, "alternative premia") and a
backtest price-adjustment fix live on an unpublished branch. Rely on the numbers
in this brief for them.

## 1. The system you're extending

- **What it is.** A Python daily-bar trading system on a Linux VPS (6 cores, 11 GB RAM).
- **Cadence.** Decisions come from completed daily bars, and orders go out through broker APIs. New designs are expected to decide **at most once per trading day**.
- **Instruments today:** long/short **US stocks and ETFs only**, traded on two paper accounts (Interactive Brokers and Alpaca). The code has no options, forex, futures or CFD order types. Adding one is real engineering work, so identify from the code what would break. Likely candidates: integer share quantities, no contract multipliers, expiries or rolls, and no swap or financing accrual.
- **Free data it can use:**
  - Tiingo: daily total-return prices and BTC.
  - CBOE: index history (VIX, VIX3M, PUT/BXM).
  - FRED: macro and rates.
  - SEC EDGAR filings.

  Paid data is fine if it's worth it, with the cost stated for the owner to approve.
- **Owner: resident of Israel, open to any provider.** For these new instruments the
  owner will use **whichever brokers, data vendors or bridge services work best**.
  They're not tied to the current accounts. Don't spend effort assessing the
  existing brokers' capabilities. Find the best provider for each instrument.

  A provider qualifies if it:
  - accepts Israeli residents;
  - is properly regulated, with the regulator and entity named;
  - offers an **API this system can automate from Python on Linux**, with automated trading allowed by its terms;
  - offers a demo or paper account with the same API.

  If an instrument has no provider that meets this bar, say so. Flag anything Israeli residents can't access.

## 2. What has already been tested (don't re-propose these as new)

The owner's standard:
- pre-registered, walk-forward out-of-sample tests;
- paired block-bootstrap confidence intervals;
- placebo controls;
- Deflated Sharpe counted over the cumulative number of trials;
- PBO/CSCV.

On that standard, these have **failed**:

**The current stock strategy set.** 11 strategies:
- momentum, trend, mean reversion;
- volatility breakout, stat-arb pairs;
- seasonality, multi-factor, event-driven;
- news sentiment;
- an HMM regime model;
- chart patterns with an ML filter.

Every way of combining them lost money out of sample, 2020–2026, and a random-weights placebo did best. Note: a split-accounting bug in the backtest was found on 2026-09-29, and this result is being re-measured.

**A 2007–2026 test of alternative premia** against buy-and-hold SPY, 60/40 SPY/IEF and volatility-targeted SPY, net of costs, independently recomputed:

| Candidate | Sharpe gap vs benchmarks |
|---|---|
| VIX-term-structure-timed short volatility (via SVXY, including the Feb-2018 blow-up) | about −0.4 |
| Index put-writing (CBOE PUT) | −0.1 to −0.25 |
| Turn-of-month plus pre-FOMC SPY | −0.1 to −0.3 |
| Cross-asset ETF time-series momentum, long/flat | ≈ 0 to −0.1 |
| A risk-parity mix of the above | −0.24 |

None of these gaps was statistically significant.

**The only lead.** A 4-week Bitcoin trend rule beat buy-and-hold BTC (Sharpe 1.35 vs 1.07). It is not statistically proven, and about a third of the edge comes from 2022.

**Power problem.** With about 15–20 years of daily data, these tests can only detect a Sharpe advantage of roughly **0.4–0.9**. **Prefer ideas with many independent bets per year, or very large documented effects.**

## 3. What to research

These are new paths for the owner, so **everything must be researched online
and backed by trustworthy sources**. Nothing may rest on general knowledge or
opinion. Work in two stages.

### Source rules (apply to both stages)

Use sources in this order of trust, and say which tier each citation is:

1. **Peer-reviewed research** (e.g. Journal of Finance, Review of Financial Studies, Journal of Financial Economics, Journal of Portfolio Management) and **independent replications**. Examples:
   - Hou, Xue & Zhang, "Replicating Anomalies";
   - Jensen, Kelly & Pedersen, "Is There a Replication Crisis in Finance?", and their global factor data;
   - McLean & Pontiff on post-publication decay;
   - Open Source Asset Pricing (Chen & Zimmermann).
2. **Working papers** from established authors (NBER, SSRN), and **practitioner research that publishes its data and method**, such as AQR, Man Group, Research Affiliates, or the Kenneth French data library.
3. **Primary documents** for every fact about costs, rules, taxes and APIs:
   - the provider's own fee, margin, contract-spec and API documentation;
   - exchange documentation (e.g. CME, Cboe, TASE);
   - regulators, e.g. the Israel Securities Authority and the provider's home regulator;
   - the Israel Tax Authority;
   - data vendors' own pricing pages.

**Not acceptable as evidence:**
- provider marketing or blogs;
- affiliate "best broker" review sites;
- forums, social media and video;
- AI-generated summaries;
- backtests with no stated method or costs.

You may use these only to find a primary source, then cite the primary source.

**For every claim:**
- give a link, the date accessed, and for papers the sample period and whether results are gross or net of costs;
- when sources disagree, report both and explain the gap, rather than picking one;
- if a fact can't be verified from a primary or tier-1/2 source, mark it **UNVERIFIED** and don't use it in a recommendation;
- never estimate a number without showing how.

### Stage A: evidence survey of every item

Research **each** item below properly. For each, give a sourced write-up of about half a page to a page, covering:

- **The best available evidence of a durable edge, or of a real diversification or risk benefit.** Give the strongest paper or replication, its sample, its net-of-cost result, what happened after publication, and any credible evidence that it has decayed.
- **Why the return should persist:** who pays it and why (risk premium, behavioural, structural flow), with a source.
- **All-in costs**, from primary sources.
- **Historical data:** free or paid, with vendor, price and fitness for an honest backtest.
- **The best provider for an Israeli resident** that meets the §1 bar, and the runner-up.
- **Israeli tax and regulatory points**, from ISA or Israel Tax Authority sources, or marked for a tax adviser.
- **Tail risk**, with the historical episodes that show it.
- **Detectability:** estimated independent bets per year, compared with the minimum detectable Sharpe gap of about 0.4–0.9 (§2).
- **A confidence rating** (high, medium or low) based on the quality of the sources found.

Add anything important that's missing. Drop anything unavailable to an Israeli resident, citing why.

1. **Listed options** on US indices, ETFs and single stocks: premium selling, spreads, collars, and volatility-surface or skew trades.
2. **Spot forex:** carry, momentum and value, as major-pair and G10 baskets.
3. **CFDs** on indices, FX, commodities and stocks.
4. **Futures:** equity-index, Treasury, commodity, FX and VIX futures. Cover trend, carry and roll-yield, and calendar spreads.
5. **Bonds and cash:** US Treasuries and T-bills, bond ETFs, duration and credit timing, and yield-curve carry or roll-down.
6. **Commodities** via ETFs or ETCs, for example gold, broad commodity baskets, and roll-optimised funds.
7. **Crypto beyond spot BTC:**
   - ETH and a small crypto basket;
   - spot crypto ETFs;
   - CME Bitcoin futures and their basis;
   - perpetual-futures funding-rate carry.

   Only regulated venues an Israeli resident can reach through an API count.
8. **Volatility products:** VIX futures and VIX ETPs, beyond the SVXY test in §2.
9. **Israeli market (TASE):**
   - TA-35/TA-125 stocks and index funds;
   - Israeli government bonds (Shachar, Galil) and Makam T-bills;
   - Maof TA-35 index options;
   - shekel (ILS) carry and hedging.

   Include which providers offer API access to TASE.
10. **ETF and fund structure effects:**
    - closed-end fund discounts;
    - leveraged-ETF rebalancing flows and volatility decay, whether harvestable and at what borrow cost;
    - index reconstitution and additions;
    - ETF creation/redemption premium or discount.
11. **Event-driven equities that need no new instrument type:** merger arbitrage, spin-offs, post-earnings drift, and insider-purchase clusters. SEC filings are already ingested, so note what data each needs.
12. **Passive and structural allocations:** risk parity across stocks, bonds and commodities; dividend and quality tilts; international diversification. Judge these as benchmark-improving allocations, not alpha.

### Stage B: deep dive on the top candidates

Pick **up to 5** items or strategies from Stage A on the strength of their evidence, and justify the choice. For each, answer the questions below. The same source rules apply.

1. **The specific strategy and its evidence.** Give a rule precise enough to pre-register (universe, signal, rebalance frequency, sizing), plus its source, sample period, post-publication performance and Sharpe *net of realistic costs*. Candidates to evaluate, not assume:
   - FX carry, FX momentum and FX value;
   - volatility risk premium via options, including short strangles, covered calls, and dispersion or skew;
   - futures trend-following, cross-asset carry, and term-structure/roll yield.

   Say what edge is left after 2010, and why someone keeps paying it.
2. **Why it wouldn't repeat the failures in §2.** Is it structurally different from SVXY short-vol, put-writing or ETF trend? Does it add breadth?
3. **Real costs through the recommended provider.** Current figures with sources:
   - spreads and commissions;
   - options bid-ask on the relevant strikes and tenors;
   - swap or financing;
   - margin;
   - futures roll costs;
   - contract minimums against a US$100k–1M account.
4. **Provider and API.** Cover:
   - the recommended broker and any bridge or data service, and why;
   - regulator and client-money protection;
   - the API type and the exact calls or contract types needed;
   - rate limits and automation terms;
   - whether the paper account realistically simulates fills, assignment and exercise, swaps and margin calls;
   - known gaps where paper differs from live;
   - monthly cost of every service involved.
5. **Israel-specific rules and tax.** Israeli tax treatment of gains, losses and financing compared with stocks: whether losses offset, how foreign-provider income is reported, and any withholding. Also ISA leverage limits and retail restrictions. Tax is for orientation only; say where a local tax adviser is needed.
6. **Historical data for an honest backtest.**
   - The cheapest adequate source, with price, coverage, survivorship handling and point-in-time correctness.
   - Specifically, as relevant: options chains with bid/ask history, forex rates *plus* historical interest-rate differentials for carry, continuous futures with roll schedules, and CFD price and financing history.
   - Flag data that looks free but isn't fit for purpose.
7. **Risk and failure modes.**
   - tail events such as the 2015 CHF de-peg, Feb-2018 Volmageddon and March 2020;
   - margin-call dynamics;
   - gap risk;
   - early assignment;
   - counterparty risk.
8. **Testability here.** Estimate the number of independent bets per year and whether the documented effect is detectable with the tests in §2. Name the right benchmark, for example a passive carry index or buy-and-write.
9. **Integration plan against the real code.**
   - Which classes and methods would need to change or be added? For example a new broker adapter implementing `src/firm/brokers/base.py`, a data provider under `src/firm/data/providers/`, and the instrument and quantity model in `src/firm/portfolio/state.py` and `src/firm/agents/execution.py`.
   - Which assumptions in the current code break, with file paths?
   - What does the backtest need, for example multipliers, expiries, swap accrual and margin?
   - Give a rough effort estimate for each.

## 4. Deliverable

A report with no fixed page limit (depth matters more than brevity), containing:

1. **A ranked shortlist of up to 5 candidates.** For each give:
   - the pre-registrable rule;
   - evidence, with net Sharpe and post-publication behaviour;
   - estimated all-in cost per year;
   - data source and cost;
   - recommended provider(s) and their monthly cost;
   - main tail risk;
   - a testability verdict: detectable, borderline, or not detectable;
   - the integration effort, pointing to the code.
2. **The full Stage A evidence survey**, one sourced section per item, plus a summary table with: evidence strength, net edge after publication, cost, data, best provider for an Israeli resident, tax and regulatory fit, detectability, implementation effort, and confidence.
3. **A provider table** for the shortlist: provider, regulator and entity serving Israel, API type, automation terms, paper account, costs and data. Give a runner-up for each.
4. **"Don't bother" list:** popular ideas that fail on costs, capacity, data or post-publication decay, with the reason for each.
5. **Open questions for the owner**, such as margin appetite and paid-data budget.
6. **Sources:** links to papers, provider fee, margin and API documentation, regulator pages and data-vendor pricing, each with the date accessed. Mark anything unverified as **UNVERIFIED**, and never invent a number. If a figure is an estimate, say how you got it.

**Standards:**
- Follow the source rules in §3.
- Be more sceptical of positive results than negative ones.
- Prefer independent replications, and net-of-cost post-2010 evidence.
- If the evidence says an idea doesn't work for a small, automated, Israeli-resident account, say so. A well-sourced "no" is a useful result.
