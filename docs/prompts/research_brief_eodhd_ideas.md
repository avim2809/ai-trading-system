# Research brief: new strategy ideas that use an EODHD "Historian" subscription

You are researching for the owner of a small, self-built systematic trading
system. The owner has just paid for **one month** of EODHD's "Historian" data
plan. Your job: propose **new, testable strategy ideas that this data (plus the
free data below) makes possible**, rank them by evidence and detectability, and
make each one concrete enough to pre-register and test within that month.

**Say plainly** when an idea isn't worth testing. A short list of strong ideas
beats a long list of weak ones.

## 0. How to use this brief

- **You can't see the code.** §1 and §4 describe the system well enough to plan
  a test and an integration. If you also have read access to
  `https://github.com/avim2809/ai-trading-system` (branch `main`), read
  `docs/edge_search_verdict_2026_09.md`, `docs/allocation_deploy_runbook.md` and
  `src/firm/allocation/` for detail. Cite file paths if you do.
- **You must research online.** Every claim about a strategy's evidence, and
  every fact about EODHD's data, must come from a source you cite (rules in §5).

## 1. The system today

- **Stack:** Python, a Linux VPS (6 cores, 11 GB RAM), daily bars. New designs
  decide **at most once per trading day**, from completed bars.
- **Two paper accounts:**
  - **Alpaca, ~US$98k.** Since 2026-09-30 it runs an **allocation portfolio**:
    - 92% passive core (60% SPY / 40% IEF, monthly rebalance, 2% drift band);
    - 8% satellite (a BTC 4-week trend rule, weekly);
    - 1% cash.

    New ideas are meant to become **additional small satellite "sleeves"**
    (typically 5–15% of NAV each) inside this structure.
  - **IBKR, US$1M.** Still runs the old 11-strategy stock-picking system, as a
    control.
- **Owner:** an Israeli resident, open to any regulated broker with a
  Python/Linux-automatable API and a paper account. Alpaca supports US
  stocks/ETFs and crypto; IBKR supports stocks, ETFs, options, futures and forex.

## 2. What's already been tested (don't re-propose these as new)

The owner's standard:
- pre-registered, walk-forward out-of-sample tests;
- paired block-bootstrap confidence intervals;
- placebo controls;
- Deflated Sharpe with cumulative trial counts;
- PBO/CSCV.

Results so far:

**Failed or inconclusive:**
- **The 11 current stock strategies**, each tested alone: momentum, trend, mean
  reversion, volatility breakout, stat-arb pairs, seasonality, multi-factor,
  event-driven (a crude PEAD proxy), news sentiment (only ~1 year of data), an
  HMM regime model, and chart patterns with an ML filter. **0 of 11 had an
  edge**, and their books turned over 60–100% a day.
- **Every way of combining those 11** lost out of sample, no better than a
  random-weights placebo.
- **Alternative premia, 2007–2026,** against SPY / 60-40 / vol-targeted SPY: all
  inconclusive, mostly with negative point estimates:
  - VIX-contango-timed short vol (SVXY);
  - CBOE put-write;
  - turn-of-month plus pre-FOMC SPY;
  - cross-asset ETF trend (long/flat, 8 ETFs);
  - an inverse-vol mix of these.

**Only lead:** BTC 4-week trend vs holding BTC (Sharpe 1.35 vs 1.07). Unproven, but
now running live as the satellite.

**Already scoped elsewhere (don't duplicate):**
- **Insider-purchase clusters** from SEC Form 4. Its test is being built now,
  using this subscription's delisted-stock prices.
- **Futures trend-following.** Needs separate futures data.

**Ruled out by external research** (post-2010 decay, cost, or the same premium in
a different wrapper): FX carry, PEAD, CFDs, VIX-futures basis, covered calls,
individual-commodity roll yield, spin-offs, index reconstitution, ETF
creation/redemption arbitrage.

**The power problem.** With about 15–20 years of daily data, these tests can only
detect a Sharpe advantage of roughly **0.4–0.9**. **Prefer ideas with many
independent bets per year** (cross-sectional, many assets, event-driven), or very
large, well-documented effects.

## 3. The data available

### EODHD "Historian" (paid, one month)

Verify each item against EODHD's own documentation; flag anything you can't
confirm.

**Included:**
- **End-of-day prices, 30+ years:** OHLCV plus adjusted close, for stocks, ETFs
  and indices worldwide, **including delisted tickers**. The US delisted list has
  about 60,000 symbols, of which about 33,000 are common stocks.
- **Forex end-of-day** (major and minor pairs).
- **Crypto end-of-day** (many coins, not only BTC).
- **Commodities end-of-day.**
- **Splits and dividends,** full history.
- **Financial news, with sentiment scores** (verify how far back the history goes).
- **US Treasury yields and bills.**
- **Bulk per-day downloads** (a whole exchange's end-of-day data for one date).
- **Symbol lists** per exchange: active and delisted.

**Not included:** fundamentals (so no market cap or accounting data), intraday
bars, technical-indicator API, insider transactions, earnings calendar, options.

**Limits:** 100,000 API calls a day, ≤1,000 requests a minute. Most requests cost
1 call; news costs 5. It's a personal-use licence.

**Everything needed must be downloaded within the month.** Check whether the
licence lets downloaded data be kept after cancelling, and flag it if not.

### Free data already available

- SEC EDGAR (Form 3/4/5 insider data sets, 2006 onward, already parsed; filings);
- FRED (macro, rates incl. OECD short rates);
- CBOE index history (VIX, VIX3M, PUT, BXM);
- Tiingo (US end-of-day prices);
- Alpaca and IBKR broker data.

## 4. How a new idea would plug in

A new idea should be a **Sleeve**:
- `target_weights(asof, history) → {symbol: weight}` returns long-only weights
  within the sleeve, summing to at most 1, with the remainder in cash.
- `history` is completed daily closes per symbol, up to `asof`.
- `is_rebalance_due(asof, last_rebalance)` sets its cadence: daily, weekly or
  monthly.

The allocator combines sleeves into target weights of current NAV. It trades only
on a sleeve's schedule or when drift exceeds a band, and never shorts or uses
leverage. It routes orders through existing safety gates. **Crypto and fractional
quantities work on Alpaca.**

Anything outside this shape must be called out explicitly with a rough effort
estimate. That includes shorting, leverage, futures, options, forex execution,
intraday data, and cross-sectional books of hundreds of stocks (the allocator
today handles a handful of symbols per sleeve).

## 5. Source rules

**Tiers of evidence:**
1. **Peer-reviewed research** and **independent replications**, e.g. Hou/Xue/Zhang
   "Replicating Anomalies", Jensen/Kelly/Pedersen, McLean/Pontiff on
   post-publication decay, Open Source Asset Pricing.
2. **Working papers** from established authors (NBER/SSRN), and **practitioner
   research with published data and method** (AQR, Man, Research Affiliates, the
   Ken French library).
3. **Primary documents** for every fact about data, costs and APIs: EODHD's own
   docs and pricing, broker fee and API pages, exchange docs.

**Not evidence:** vendor marketing, affiliate or review sites, forums and social
media, AI summaries, and backtests with no stated method or costs.

**For every claim:** give a link and the date you accessed it, the sample period,
and whether results are gross or net of costs. Mark anything you can't verify as
**UNVERIFIED**, and don't base a recommendation on it.

## 6. What to research

**Stage A: idea generation, 10–20 candidates.** Every candidate must *need* data
from §3; say which fields. Directions to consider (not assume), each with its
evidence:
- **Cross-sectional stock anomalies that need only prices and volumes**, run on a
  broad, survivorship-free US universe (now possible with delisted prices). Examples:
  - short-term reversal;
  - industry momentum;
  - 52-week-high proximity;
  - low volatility / low beta;
  - idiosyncratic volatility;
  - max-effect / lottery stocks;
  - turnover and liquidity effects;
  - seasonality in the cross-section (Heston–Sadka).

  Say which survive replication after 2000 and after costs, for a small account
  holding a limited number of names.
- **Dividend-based effects** using the dividends data: ex-dividend price
  behaviour, dividend initiations and changes as events, dividend-month
  seasonality.
- **Split events** as signals, if the post-split drift literature survives
  replication.
- **Forex:** momentum and value (carry is ruled out), using EODHD forex plus FRED
  rates. Also a G10 FX trend sleeve implemented through currency ETFs, if FX
  execution isn't available.
- **Crypto:** cross-sectional momentum or trend across a basket of large coins,
  using survivorship-aware coin lists. This extends the live BTC sleeve.
- **Global and international:** country-index momentum and value, and the
  international ETF lead-lag effect.
- **Commodity and bond ETF signals** that weren't already tested (see §2).
- **News-sentiment signals,** only if EODHD's news history is long enough and
  point-in-time. **Check how far back it goes and whether scores could have been
  revised later.** If it's short, say that it supports only a forward test.
- **Breadth and market-internal timing signals** (advance/decline, new
  highs/lows) built from the full-universe end-of-day data, as a risk overlay on
  the core rather than an alpha sleeve.

**Stage B: deep dive on the top 3–5.** For each:
1. A **pre-registrable rule**: universe, signal, rebalance frequency, sizing,
   fixed parameters taken from the source paper.
2. **Evidence:** the strongest paper or replication, its sample, net-of-cost
   result, post-publication behaviour and any decay.
3. **Why it isn't a repeat** of a failure in §2, and what it adds (breadth,
   diversification against the live 60/40 + BTC book).
4. **Detectability:** independent bets per year against the 0.4–0.9 Sharpe
   floor, and the right benchmark.
5. **Exact data needed:** EODHD endpoints, symbol counts, date range, estimated
   API calls (must fit well inside 100k a day and the one-month window), plus any
   free data.
6. **Costs:** realistic trading costs for a US$100k account at the broker that
   would run it.
7. **Integration:** does it fit the Sleeve shape in §4? If not, what's missing
   and roughly how much work it is.
8. **Tail risks.**

## 7. Deliverable

1. **A ranked shortlist (3–5)** with everything in Stage B.
2. **A download plan for the month:** the exact EODHD requests to make, in
   priority order, with a call budget, so nothing is missing after cancellation.
3. **The Stage A long list,** one short sourced paragraph each, with a
   keep-or-drop reason.
4. **A "don't bother" list:** popular ideas that fail on evidence, costs, data or
   detectability.
5. **Open questions for the owner.**
6. **Sources**, with access dates.

**Standards:**
- Be more sceptical of positive results than negative ones.
- Prefer independent replications and post-2010 net-of-cost evidence.
- A well-sourced "no" is a useful result.
