# Research brief (standalone): tradeable return sources beyond US stocks

You are researching for the owner of a small, self-built systematic trading
system. Your job is to find out which **instruments and strategies beyond US
stocks** are worth adding. Back each recommendation with evidence from
trustworthy sources, a realistic cost picture, the service providers that would
make it work, and a concrete plan for plugging it into the existing code. **Say
plainly** when the honest answer is "not worth it".

## 0. How to use this brief

You have **no access to the owner's code**. §1 and the appendix at the end
describe the system accurately enough to plan an integration. Base every
integration statement on that description, and don't guess at code you can't see.
You **do** need web access: everything in §3 must be researched online.

## 1. The system you're extending

- **What it is.** A Python daily-bar trading system on a Linux VPS (6 cores, 11 GB RAM).
- **Cadence.** Decisions come from completed daily bars, and orders go out through broker APIs. New designs are expected to decide **at most once per trading day**.
- **Instruments today:** long/short **US stocks and ETFs only**, traded on two paper accounts (Interactive Brokers and Alpaca). The code has no options, forex, futures or CFD order types. Adding one is real engineering work; the appendix lists the assumptions that would break.
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
9. **Integration plan against the system described in the appendix.**
   - What would a new broker adapter need, mapped onto the `Broker` interface (appendix A.2)?
   - What would a new data provider need (appendix A.3)?
   - Which assumptions in appendix A.4 break for this instrument, and what the minimal extension of the instrument and position model looks like, e.g. an instrument type, contract multiplier, expiry and roll, fractional lots, swap or financing accrual, margin.
   - What does the backtest need (appendix A.5)?
   - Give a rough effort estimate for each piece.

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
   - the integration effort, against the appendix.
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

---

## Appendix: the system, as far as integration is concerned

This describes the owner's code as of 2026-09-30. Names are the real ones.

### A.1 Pipeline

Every cycle runs these steps in order:
1. Strategies emit per-symbol signals.
2. Three analysts z-score them per strategy.
3. A bull/bear debate step.
4. A portfolio manager turns scores into **target weights per symbol** (fractions of NAV, long or short).
5. A risk agent clips them:
   - per-name cap;
   - sector cap;
   - net and gross exposure caps;
   - liquidity cap at a fraction of 20-day average dollar volume;
   - drawdown kill switch.
6. An execution agent converts weight changes into orders:
   - a rebalance band, so small changes aren't traded;
   - a partial-fill fraction per day;
   - share rounding.

Capital can be one shared book, or split into independent per-strategy "sleeves".

### A.2 Broker interface (abstract class `Broker`)

Every broker adapter implements these methods:
- `connect`, `disconnect`, `is_connected`, `reconnect`, `health_check`
- `get_account() -> dict`, for cash, equity and buying power
- `get_positions() -> list[BrokerPosition]`, `get_position(symbol)`
- `submit_order(OrderRequest) -> OrderStatus`, `cancel_order(id)`, `get_order_status(id)`, `get_open_orders()`
- `get_current_price(symbol)`, `get_current_prices(symbols)`
- `is_market_open()`, `market_hours()`

`OrderRequest` has these fields:
- `symbol: str`
- `side: "buy"|"sell"`
- `quantity: float`
- `order_type: market|limit|stop|stop_limit|trailing_stop`
- `limit_price`, `stop_price`, `trail_percent`, `trail_amount`
- `time_in_force` (default `"day"`)
- `extended_hours`
- `strategy`
- `client_order_id`

`BrokerPosition` has `symbol`, `quantity`, `avg_cost`, `market_value` and `unrealized_pnl`.

`OrderStatus` has `order_id`, `symbol`, `side`, `quantity`, `filled_quantity`, `avg_fill_price`, `status` (`pending`/`filled`/`partial`/`cancelled`/`rejected`), `timestamp` and `order_type`.

**No field carries an instrument type, currency, exchange, multiplier, expiry, strike or right.**

The two existing adapters wrap the IBKR TWS API (synchronous, run on its own thread) and `alpaca-py`.

### A.3 Data layer

A `DataProvider` base class returns pandas DataFrames:
- `get_prices(symbols, start, end)`, with columns `date, symbol, open, high, low, close, volume, adj_close`, daily bars;
- `get_fundamentals`, `get_news_sentiment`, `get_corporate_actions`, `get_universe_constituents`, `get_analyst_ratings`.

A fallback chain tries providers in order. Existing providers:
- Tiingo, Alpha Vantage, FMP, Massive, Finnhub, Twelve Data;
- FRED, SEC EDGAR;
- IBKR, Alpaca.

Strategies read data only through a point-in-time view (`PitView.prices(symbols, lookback_days)`, `.fundamentals(...)`, `.sentiment(...)`, ...), which filters everything to `date <= asof`.

A strategy implements `generate(pit_view) -> list[Signal]`. A `Signal` has:
- `symbol: str`
- `strategy: str`
- `score: float`
- `confidence: float` in [0, 1]
- `horizon: str` (e.g. `"21d"`)
- `asof: datetime`
- `meta: dict`

### A.4 Position and instrument model: assumptions that would break

- **One string symbol is the whole identity of an instrument.** Holdings are `dict[symbol -> shares: float]`, with cash in one currency (USD). There is no instrument type, contract multiplier, expiry or roll, strike or right, or quote currency.
- **Exposure is `shares × price / NAV`,** so there is no concept of margin, notional vs premium, or delta.
- **Protective-stop quantities are rounded to whole shares,** and orders are sized from target weights. Fractional FX lots or contract counts need a different sizing rule.
- **Costs are percentage-of-notional:**
  - commission 5 bps;
  - slippage 5 bps;
  - spread 2 bps;
  - plus square-root market impact vs 20-day dollar volume.

  There is no per-contract fee, no swap or financing accrual (short-borrow cost accrues only on short stocks), and no option assignment or exercise handling.
- **The calendar is the US equity trading day,** with cycles in US/Eastern market hours. 24-hour or 24/5 markets would need a session model.
- **Risk caps assume equity-like instruments:** per-name weight, GICS-style sector map, net and gross exposure.

### A.5 Backtest

- It is built on **backtrader**. There is one data feed per symbol from daily OHLCV. The broker fills at the feed's close, and backtrader's percentage commission and slippage apply.
- Positions are marked at total-return-adjusted closes.
- There is no margin model, contract multiplier, expiry or roll, or financing accrual.
- The same agent pipeline runs in the backtest and live.
- Evaluations use walk-forward folds, paired block-bootstrap CIs, placebo controls, Deflated Sharpe with cumulative trial counts, and CSCV PBO.
