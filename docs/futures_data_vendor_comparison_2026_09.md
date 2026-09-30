# Futures data vendor comparison (2026-09-30)

**Status: research only, no purchase made.** Written for the futures
trend-following candidate scoped in `docs/research_findings_beyond_equities_2026_09_30.md`
(Part 1, item 1) and `docs/edge_search_verdict_2026_09.md`. No vendor account
was created or paid for — this gathers public pricing-page quotes only, per
the owner's explicit no-signup/no-spend rule. Anything not confirmed from a
vendor's own page is marked **UNVERIFIED**.

**Target:** an honest ≥20-year daily-bar backtest across the ~19-market
universe named in the brief — ES, NQ, RTY, FDAX (equity index); ZF, ZN, ZB,
FGBL, FGBS (rates); 6E, 6J, 6B, 6A, 6C (FX); CL, GC, HG, ZC, ZS (commodities)
— using either (a) continuous contracts with a documented roll schedule and
back-adjustment methodology, or (b) raw individual expired contracts to build
that series in-house.

## Comparison table

| Vendor | Price | Coverage (years / markets) | Continuous/back-adjusted contracts | Linux / API | Format | Licence | Fitness |
|---|---|---|---|---|---|---|---|
| **Norgate Data (Futures Package)** | **USD 270/yr** (~$22.50/mo); confirmed on norgatedata.com/futurespackage.php | ~100 futures markets, 11 exchange groups; history "back to around 1980 or the first day of trading" — comfortably ≥20yr for all 19 target markets except the newest micros | **Yes** — both unadjusted and back-adjusted spot-continuous series included, built-in | Python package (`norgatedata` on PyPI) reads from a **locally-updated database**; the updater itself, **NDU, is Windows-only** (confirmed: forum/PyPI/own FAQ). Workarounds: run NDU once in a Windows VM or under WSL2 to seed/refresh the local DB, then read it from Linux via the mounted files, or keep a small always-off VM that's powered on only to refresh | Local proprietary DB + Python accessor | Personal/single-seat; redistribution restricted (standard EOD-data terms) | **Best price/coverage/adjustment-built-in combination.** The only real friction is the Windows-only updater — a one-time or periodic acquisition step, not a live-service dependency, so it doesn't conflict with the "never restart production services" rule. Eurex (FDAX/FGBL/FGBS) coverage specifically **UNVERIFIED** from public pages — the "11 worldwide exchange groups" figure implies it, but wasn't confirmed line-item. |
| **CSI Data / Unfair Advantage — World Futures (Gold)** | **USD 200/yr** (confirmed: apps.csidata.com/OrderUA pricing tiers); North American-only Gold tier is $120/yr | 30 years of history in the Gold-tier database (comfortably ≥20yr); "World Futures" tier explicitly adds European/Asian/Australian exchanges — Eurex almost certainly included but not line-item confirmed (**UNVERIFIED**) | **Yes**, and more flexible than Norgate's: UA lets you choose your own roll rule when building the continuous/back-adjusted series, rather than receiving one fixed vendor methodology | **UNVERIFIED** — no Linux/API mention found on public pages; CSI's Unfair Advantage has historically been a Windows desktop app. Its own docs mention flat-file/CSV export, which would be Linux-usable once exported, but this needs a direct (non-purchase) inquiry to CSI support to confirm before buying | Windows desktop app; CSV export | UNVERIFIED (not on public pages) | **Cheapest of the confirmed options, if Linux/API access holds up.** Recommend a pre-purchase support email to confirm CSV export / Linux workflow before spending — that email is not a purchase and doesn't violate the no-spend rule. |
| **Databento** | Pay-as-you-go ($/GB, no subscription) **or** Standard $199/mo, Plus/Unlimited $1,750+/yr; $125 free credit for new accounts (expires 6mo) | CME/CBOT/NYMEX/COMEX confirmed (covers ES, NQ, RTY, ZF, ZN, ZB, CL, GC, HG, ZC, ZS). Eurex ("Eurex now available" — added recently, per their own blog) covers FDAX/FGBL, but **history for Eurex products is almost certainly much shorter than the ~16 years quoted for the US groups** (recent addition; exact Eurex start date UNVERIFIED) | Raw/native symbology with instrument definitions; a continuous ("parent") symbology exists but back-adjustment is **not obviously a pre-built product** — likely still needs in-house construction (**UNVERIFIED** whether a back-adjusted series ships out of the box) | Fully cloud/REST/Python, genuinely Linux-native, no desktop dependency | Tick/MBO/MBP/OHLCV via API | Standard data-licence terms; no redistribution | 16+ years (~2010-) is **short of the ≥20-year ask**, but happens to line up well with this system's own primary OOS window (post-2010, since the trend rule's parameters come from pre-2012 papers) — so it's adequate for the OOS test even though not for the full-period one. Best engineering fit (Linux, API-first) of any option here; worse cost/history fit than Norgate/CSI. |
| **Barchart OnDemand** | API access reported starting around **$500/month**; a separate Excel add-in exists at $69.95/mo but doesn't give programmatic access | REST API covers futures among other asset classes; specific expired-contract/continuous-contract depth **UNVERIFIED** — public pricing/coverage pages returned blocked/empty on repeated fetches | UNVERIFIED | REST API | UNVERIFIED | UNVERIFIED | **Not adequate on cost alone** — roughly 2x Databento's subscription tier and ~20x Norgate's annual cost for what is very likely a comparable (or narrower) historical daily-bar dataset; this system only needs daily bars on a monthly rebalance, so Barchart's real-time/tick-oriented pricing tier buys capability that isn't needed here. Not recommended. |
| **Portara / CQG (Portara CQG, "CQG Data Factory")** | No published flat rate. One-off purchase: **$220–330 per commodity** for full historical daily data (roll service + delivery included); portfolio orders get bulk discounts up to ~80% off, unquantified without a quote | 123 years of history from 1899 for some markets, 45+ years of intraday from 1987 — very deep, plausibly covers all 19 target markets, not line-item confirmed | **Yes** — roll service and back-adjustment are part of the one-off product | Delivered as flat files via FTP + a Windows desktop tool (Portara) for extraction/manipulation. The **static delivery (CSV/flat files) is more Linux-friendly than Norgate's live local-DB dependency** for a one-time backtest build, since no ongoing updater daemon is required once the files are in hand | Flat files (CSV) + proprietary format via Portara software | Team licence allows up to 5 seats; standard terms otherwise | Deep history and flexible delivery, but list-price math (19 markets × $220–330 each ≈ **$4,200–6,300** before any portfolio discount) is far above Norgate/CSI unless the "up to 80%" discount applies at this order size — needs an actual quote to compare. Worth a non-binding quote request before ruling out. |
| **Nasdaq Data Link (formerly Quandl)** | N/A — no current self-serve low-cost futures-history product found | The free `CHRIS` continuous-futures dataset that used to serve this exact need is **confirmed deprecated/no longer updated** | N/A | N/A | N/A | N/A | **Not adequate.** The free/cheap era of Quandl futures data has ended; current Nasdaq Data Link futures offerings are enterprise-oriented with no comparable self-serve low-cost tier found. Not recommended for further evaluation. |
| **Tiingo** | N/A | Equities/ETFs/mutual funds/crypto/FX are Tiingo's actual coverage; one stray reference to a "new futures exchange" support item marked "TBA" suggests futures coverage is nonexistent or nascent and unpriced today | N/A | N/A (would be REST/Python if it existed, matching Tiingo's existing API) | N/A | N/A | **Not adequate today.** This contradicts the research brief's own item-12 note that "Tiingo (already free) is fully adequate" for the *passive bond/commodity ETF sleeve* — that claim is about ETFs (IEF/GLD/DBC/PDBC etc.), which Tiingo does cover, not about futures contracts, which it does not appear to. Worth re-checking if Tiingo ships a real futures product later. |
| **IBKR's own historical futures data (TWS API)** | $0 incremental for historical-bar requests (no separate historical-data subscription line item); live/delayed streaming market data for CME/CBOT/NYMEX/COMEX would add roughly $10-15/mo per exchange group if later needed for live trading (not needed for an offline backtest) | **Structurally inadequate for this task.** IBKR's own TWS API docs state expired-futures historical data is unavailable **older than two years past the contract's expiration** (`includeExpired=true` extends to that window, no further) — confirmed from `interactivebrokers.github.io/tws-api/historical_limitations.html`. Since relevant contracts (ES, NQ, ZN, etc.) expire quarterly, this rules out building a 15-20 year rolled history from IBKR alone | **No.** IBKR streams individual contract-months only; no continuous/back-adjusted product exists on this API | Already-integrated `IBKRBroker`/TWS socket API, Linux-native (this system already runs against it) | Bar data via `reqHistoricalData` | Existing account terms, no incremental data licence | **Not usable as the backtest data source** — fails the ≥20-year/continuous-series requirement by design, not by cost. Remains useful as (a) the live execution venue once a signal is deployed, and (b) a free cross-check of the last ~2 years of the front contract against whichever vendor's history is purchased. |

## Recommendation for owner approval

**Primary: Norgate Data, Futures Package, USD 270/year.** Cheapest confirmed
option with the deepest confirmed history, built-in back-adjusted continuous
contracts, and no per-symbol pricing surprises. The one real cost is
operational, not financial: NDU (the updater) needs a Windows environment
(a WSL2 session or a small VM, powered on only to pull/refresh data) — a
one-time or periodic acquisition step that does not touch any live service or
violate the "never restart production services" rule, since this is pure
backtest-data preparation, disconnected from `config/live.yaml` / the
running IBKR and Alpaca instances entirely.

**Possible cheaper alternative, unconfirmed: CSI Data World Futures (Gold),
USD 200/year.** Slightly cheaper than Norgate and offers a more flexible
(self-chosen) roll methodology, but its Linux/API story is genuinely
**UNVERIFIED** from public pages. Before spending on either vendor, a
non-binding pre-purchase email to CSI support (no signup, no payment) to
confirm CSV/flat-file export and a Linux-usable workflow would resolve the
open question cheaply. If that comes back favorable and Eurex coverage is
confirmed, CSI would edge out Norgate on both price and Linux friction.

**Not recommended:** Barchart OnDemand (too expensive for a daily-bar-only
need), Nasdaq Data Link (its relevant free product is deprecated, no
adequate paid self-serve alternative found), Tiingo (no real futures product
today), Portara/CQG (likely adequate but meaningfully pricier at list rates
— worth an informal quote only if Norgate/CSI's Eurex coverage turns out to
be inadequate), IBKR's own API (structurally capped at ~2 years of expired-
contract history, unusable as the primary source regardless of cost).

**Open items needing the owner's sign-off before any spend:**
1. Confirm Eurex (FDAX/FGBL/FGBS) coverage with Norgate or CSI directly
   (their sales lines / support emails) before purchasing — neither vendor's
   public page line-items this for the "World"/full package tier.
2. Decide whether the Windows-VM/WSL2 step for Norgate's updater is
   acceptable, or whether it's worth the (likely higher) list price of
   Portara's flat-file delivery to avoid it entirely.
3. If CSI's Linux/API answer comes back unfavorable, Norgate at $270/yr is
   the fallback recommendation regardless.

All prices/coverage claims above are from each vendor's own public pages, as
fetched 2026-09-30, except where marked UNVERIFIED. No account was created,
no trial started, no payment made.
