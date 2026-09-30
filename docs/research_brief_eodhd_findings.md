# EODHD Historian research brief: findings (2026-09-30)

Answers `docs/research_brief_eodhd_ideas.md`. Access date for every citation below is
**2026-09-30** unless stated otherwise. Research was split across three
web-research passes (EODHD/broker primary docs; the 8 classic cross-sectional
equity anomalies; dividends/splits/FX/crypto/international/commodity/breadth) —
all citations are from live web search/fetch this session, not from memory.
Anything not independently confirmed is marked **UNVERIFIED** and is not the
basis for a recommendation.

**Headline answer:** of ~20 candidates screened, only **4** are worth a real
pre-registered test this month, and none is a strong lead — they're the least-bad
options against the owner's own 0.4–0.9 Sharpe detection floor and cost
standard. **1 more (crypto cross-sectional momentum) is gated on a data question
that must be resolved before it's worth scoping at all.** Two of the four best
ideas don't cleanly fit the `Sleeve` shape in brief §4 as-is — flagged explicitly
below, per §4's own instruction to call that out.

---

## 1. Ranked shortlist (Stage B)

### 1. Industry/sector momentum via liquid ETFs

**Pre-registrable rule:** Universe = ~20–30 liquid US sector/industry ETFs
(11 SPDR Select Sector funds + a set of narrower industry ETFs, e.g. semis,
biotech, banks, homebuilders, retail). Signal = trailing 6-month total return
per ETF (skip most recent month, the standard Jegadeesh-Titman convention).
Rebalance monthly: go long the top-3 (or top-tercile) ranked ETFs, equal
weight, rest in cash. Fixed parameters (6-1 month formation/skip, monthly
rebalance, top-tercile) taken directly from the source paper, not tuned.

**Evidence:** Moskowitz & Grinblatt (1999), *"Do Industries Explain
Momentum?"*, *J. Finance* 54:1249–1290, July 1963–July 1995, NYSE/AMEX/Nasdaq
2-digit-SIC industry portfolios
([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00146)) —
industry momentum is *more* profitable than individual-stock momentum; once
industry-adjusted, individual-stock momentum becomes mostly insignificant.
Gross-of-cost only in the original paper. Text-based industry construction
(Hoberg & Phillips, *JFQA*,
[PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/6E04891A7AD30C24C21CBDE37EC06BF4/S0022109018000479a.pdf/text-based-industry-momentum.pdf))
finds the effect *stronger* and robust to a lag/bid-ask-bounce critique
(Grundy & Martin 2001) that has been raised against the classic SIC version —
**that specific rebuttal was not verified against Grundy & Martin's primary
text this session; UNVERIFIED at first-hand level.** HXZ (2020) reportedly
still lists industry momentum among the anomalies that survive their stricter
NYSE-breakpoint/value-weighted replication — **also UNVERIFIED at the exact
t-stat level** (only seen via secondary summary); read the HXZ appendix table
directly before sizing capital.

**Why it isn't a repeat / what it adds:** The 11 failed strategies traded
individual stocks with 60–100%/day turnover; this trades ~20–30 liquid ETFs
monthly. It's a different signal (relative sector strength, not
stock-picking), a different turnover profile, and diversifies against the live
60/40 + BTC book by adding *equity-sector* rotation risk instead of duration or
crypto-trend risk.

**Detectability:** ~20–30 "assets," 12 rebalances/year → modest bet count
(nowhere near a full cross-section), but each rebalance is itself a
cross-sectional rank across ~20–30 names, which is more independent
information than a single time-series trend rule. Benchmark: equal-weight
buy-and-hold of the same ETF universe, and SPY. Given the owner's own
0.4–0.9 Sharpe floor, this is underpowered for a clean statistical verdict at
20 years of monthly data — treat any single-history backtest result as
indicative only, and lean on the placebo/PBO pipeline already built for the
alt-premia study.

**Exact data needed:** EODHD EOD history for ~30 sector/industry ETF tickers,
full available range (most 1998–2005+ for SPDR sectors, shorter for narrower
funds) — **~30 API calls total** (1 call/ticker, full-range pull). No
delisted-ticker or bulk-endpoint need. Free: none required beyond EODHD.

**Costs:** Alpaca — commission-free equities/ETFs, pass-through SEC ($0.0000206/$
of sell value)/FINRA TAF ($0.000195/share, sell-side, capped $9.79/trade)/CAT
($0.000003/executed-equivalent-share) only
([Alpaca fee schedule PDF](https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf),
revised 2026-09-17). IBKR Tiered — $0.0035/share, $0.35 min/order, capped at 1%
of trade value
([IBKR pricing](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)).
At ~$100k and monthly rebalancing of 3–8 ETF positions, either broker's cost is
immaterial (well under 10bp/year).

**Integration:** Fits the `Sleeve` shape in brief §4 directly — it's a
`StaticSleeve`-like rotation among a small, fixed symbol set
(`src/firm/allocation/sleeves.py:106`), just with monthly-recomputed weights
instead of fixed ones (subclass `Sleeve`, override `target_weights` to rank
trailing 6-1-month returns and `is_rebalance_due` via
`is_calendar_rebalance_due("monthly", ...)`, mirroring
`BtcTrendSleeve`'s pattern at `src/firm/allocation/btc_trend.py:49`). Small
effort — this is the best architecture fit of anything in this brief.

**Tail risks:** Sector-momentum crashes (2009 reversal-style events) can hit
hard and fast; a rank-based long-only construction has no explicit stop, so a
regime change could produce a larger-than-expected drawdown inside a single
month between rebalances. Small (~20–30 name) universe means idiosyncratic
ETF-specific liquidity/tracking issues in any one sector fund matter more than
in a broad-market book.

---

### 2. Market breadth as a regime/risk overlay (not a stand-alone sleeve)

**Pre-registrable rule:** Build a breadth signal from the full US common-stock
universe's daily closes — e.g., % of stocks above their 200-day moving
average, or net advance/decline over a trailing window. Use it to scale the
**core 60/40 sleeve's effective equity exposure** (e.g., reduce SPY weight and
raise IEF/cash when breadth falls below a pre-registered threshold), not as an
independent long-only sleeve of its own.

**Evidence:** *"Herding for profits: Market breadth and the cross-section of
global equity returns,"* *International Review of Financial Analysis*
([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0264999319312982),
[ResearchGate](https://www.researchgate.net/publication/340699814)) — 64
countries, 1973–2018: market breadth (avg. #rising minus #falling within a
portfolio) robustly predicts future market and industry returns, controlling
for size/style/volatility/skewness/momentum/trend-following. This is the
best-sourced single citation in the entire brief (large sample, long history,
peer-reviewed, multi-country, survives many controls). **Caveat, important:**
this paper tests breadth as a **direct return predictor**, not specifically as
a **drawdown/regime overlay on a separate core portfolio** — that exact framing
is untested in anything found this session; it is the open empirical question
to pre-register, not an established result. Do not conflate with Chen, Hong &
Stein (2001) *"Breadth of Ownership and Stock Returns"*
([SSRN](https://www.ssrn.com/abstract=262106)) — that is *breadth of ownership*
(mutual-fund-holdings dispersion per stock), a different construct. Separately,
classic advance/decline-line *market-timing* claims are **not** well-supported:
a 93-indicator study found "little evidence they predict stock market
returns," and one source states the A/D line "has not been subjected to
rigorous statistical analysis" as a timing signal — don't conflate the modern
"market breadth" factor-style result with the old technical-analysis A/D-line
claim.

**Why it isn't a repeat / what it adds:** Nothing in the 11-strategy or
alt-premia edge search tested a breadth-based *overlay*; the HMM regime model
that failed was a different mechanism (return/vol regimes from price series
alone, not cross-sectional breadth). This is explicitly a risk-management
layer on the existing 60/40 core, not a new alpha book — it's meant to reduce
tail risk, not add return.

**Detectability:** Full-universe daily signal, but the *decision* it drives
(scale the core sleeve) only changes state a handful of times per cycle — bet
count for the *overlay's effect on drawdown* is much lower than the breadth
factor's own literature (monthly cross-sectional bets). Benchmark: the core
60/40 itself, un-overlaid, on max drawdown and Calmar ratio, not raw Sharpe —
the claimed benefit (if real) is defensive, not return-additive.

**Exact data needed:** Daily OHLCV for the full active + recently-delisted US
common-stock universe, to compute %-above-200dma/advance-decline breadth.
EODHD's bulk whole-exchange-per-day endpoint
(`GET /api/eod-bulk-last-day/US`, **100 calls per date requested**
— [confirmed](https://eodhd.com/financial-apis/bulk-api-eod-splits-dividends))
is the wrong tool for a 20-year daily backfill (100 calls × ~5,000 trading
days ≈ 500,000 calls just for this signal — feasible in raw budget terms
across the month, but wasteful and slow at the 1,000-req/min cap). Cheaper:
per-ticker EOD pulls (1 call/ticker, full range in one call) across a
capped universe of ~3,000–5,000 large/mid-cap active + delisted-since-2010
names is enough to build a stable breadth measure without needing every
micro-cap ever delisted.

**Costs:** N/A directly (an overlay changes the core sleeve's own rebalance
weights, not a new position set) — its cost is whatever incremental turnover
it adds to the existing 60/40 rebalance, likely small.

**Integration — does NOT fit the `Sleeve` shape as-is.** The `Allocator`
(`src/firm/allocation/allocator.py`) combines fixed-weight sleeves; there is no
existing mechanism for one signal to *scale another sleeve's weight*
dynamically. Building this properly needs new allocator-level machinery (a
"regime multiplier" applied to the core sleeve's target weight, or a
breadth-derived `weight` override at rebalance time) — this is genuinely
**outside brief §4's Sleeve shape**, and is a real, if modest, engineering
task (roughly: extend `Allocator` to accept an optional per-sleeve scaling
input, plus the breadth-computation module itself). Flagging per the brief's
own instruction to call this out explicitly.

**Tail risks:** Breadth deteriorating *without* index price deterioration
(narrow-leadership rallies) is exactly the scenario this is meant to catch —
but it's also the scenario most vulnerable to whipsaw (de-risking into a
narrow-but-still-rising market, then re-risking late). Needs its own
pre-registered walk-forward test against the plain 60/40 before being trusted
with a single dollar of the live book's risk budget.

---

### 3. Diversified trend across bond and commodity ETFs

**Pre-registrable rule:** Two related, separable sleeves, both time-series
(not cross-sectional) trend/momentum, both distinct from the already-rejected
single-commodity roll-yield and the already-rejected 8-ETF cross-asset trend
alt-premia test:
- **Bond:** apply a 1-month-lookback time-series momentum rule (long if prior
  month's return > 0, else cash) separately to 3–4 duration buckets (e.g.
  SHY/IEF/TLT or similar), rebalanced monthly. Based on Sihvonen,
  *"Yield Curve Momentum,"* *Review of Finance* (forthcoming/2022+;
  [publisher](https://revfin.org/yield-curve-momentum/),
  [CEPR VoxEU summary](https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice))
  — past-month Treasury returns predict next month's, driven by autocorrelated
  yield changes, not carry. **Important limitation the source paper itself
  states: the effect is "short-lived, insignificant after the next month"** —
  this is a narrow-horizon effect; don't extend the lookback beyond what the
  source paper tests.
- **Commodity:** dual-momentum (relative-strength ranking + absolute trend
  filter) across ~8–10 commodity-sector ETFs, monthly. Practitioner-grade
  evidence only (Quantpedia backtests, 2007–2026, not peer-reviewed — one tier
  below the bond citation, treat as illustrative) plus one real live
  comparison point: Direxion's Auspice Broad Commodity Strategy ETF (COM), a
  real traded multi-commodity trend product, +7.70% in 2025 — evidence the
  strategy *exists and trades*, not evidence of edge.

**Why it isn't a repeat / what it adds:** §2 already ruled out
VIX-contango/put-write/turn-of-month/8-ETF-cross-asset-trend/inverse-vol-mix —
none of those is yield-curve-change momentum on Treasuries by duration bucket,
or a genuinely multi-commodity (not single-commodity roll-yield) trend book.
Adds duration-curve and commodity-basket diversification against the 60/40 +
BTC book, which currently has zero explicit commodity exposure and only one
crude duration split (60/40 fixed).

**Detectability:** Bond side: only 3–4 duration buckets × 12 months = ~36–48
bets/year — thin, matching the source paper's own point that the effect is
narrow and single-month. Commodity side: ~8–10 ETFs × 12 = up to ~120 bets/year
cross-sectionally, better breadth. Benchmark: an unlevered constant-duration
bond index and an equal-weight static commodity-ETF basket, respectively — not
just SPY/60-40.

**Exact data needed:** EODHD EOD for ~10–15 bond ETF tickers + ~8–10 commodity
ETF tickers (full range, 1 call/ticker ≈ 20–25 calls), plus EODHD's own
Commodities API for cross-checking spot series (23 series, "beta," 1 call each
— **note: precious metals (gold/silver/platinum/palladium) were discontinued
by FRED and are NOT available**, only energy/2 metals(aluminum, copper)/6
agricultural/5 index series —
[confirmed](https://eodhd.com/financial-apis/commodities-api-historical-prices-for-oil-gas-metals-agriculture-beta)).
Free: FRED constant-maturity Treasury yield series (1977+) for the
yield-change robustness check.

**Costs:** Same ETF-cost structure as candidate #1 (Alpaca commission-free +
pass-through fees, or IBKR ~$0.0035/share) — trivial at $100k scale and
monthly rebalancing.

**Integration:** Fits the `Sleeve` shape well — two small, fixed-symbol-set
sleeves (`symbols()` returns 10–15 and 8–10 tickers respectively), each a
straightforward `target_weights`/`is_rebalance_due("monthly", ...)`
implementation in the same shape as `BtcTrendSleeve`
(`src/firm/allocation/btc_trend.py:49`) but with a cross-sectional rank step
instead of a single-symbol on/off rule.

**Tail risks:** Commodity trend books are vulnerable to sharp mean-reversions
(e.g., 2020 oil-price collapse/negative-price event); bond momentum can
whipsaw hard around rate-regime turning points (2022-style), exactly when
duration diversification is supposed to help.

---

### 4. 52-week-high proximity (equity, liquid-name-only)

**Pre-registrable rule:** Rank a liquid large/mid-cap US universe by
(price / trailing 52-week high), monthly, long the top decile/quintile
(nearest to their 52-week high), equal weight, monthly rebalance. Universe
restricted to liquid names only (see cost caveat below) — fixed parameters
from George & Hwang (2004).

**Evidence:** George & Hwang (2004), *"The 52-Week High and Momentum
Investing,"* *J. Finance* 59:2145–2176, July 1963–Dec 2001, NYSE/AMEX/Nasdaq
([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00695.x))
— nearness-to-52-week-high dominates classic past-return momentum as a
predictor; ~0.45%/month gross decile spread; unlike JT momentum, does **not**
mean-revert long-run. Net-of-cost: Bettman, Sault & von Reibnitz (2010,
Australia,
[SAGE](https://journals.sagepub.com/doi/10.1177/0312896210385282)) find the
strategy "fails to produce significant dollar profits" once short-sale
restrictions/transaction costs/liquidity constraints are applied broadly, but
— importantly — restricting to **liquid names only** still shows significant
positive raw returns, while illiquid names show negative returns: the edge, if
real, survives specifically in the segment this system could actually trade.
**Conflicting signal on the HXZ (2020) stricter replication:** one secondary
summary says 52-week-high variants are "hard-pressed to perform" under HXZ's
NYSE-breakpoint/value-weighted methodology; another summary of the same paper
lists 52-week-high variants among anomalies that remain significant (best
performer among momentum-family variants, ~1.9%/month gross). **This
discrepancy is unresolved — mark the exact HXZ t-stat UNVERIFIED and read the
HXZ appendix table directly before committing capital.**

**Why it isn't a repeat / what it adds:** The failed 11-strategy "momentum"
strategy used classic past-return momentum, not 52-week-high proximity —
George & Hwang's own point is that these are empirically distinct signals.
Also distinct from the failed "chart patterns with ML filter" strategy (a
different, technical-pattern-based signal).

**Detectability:** Full liquid cross-section (hundreds of large/mid-caps),
monthly — good nominal bet count, best of anything equity-related in this
brief. But note the system currently sizes ~5–15% satellite sleeves of a
handful of symbols each (brief §4); a genuine full-cross-section
implementation is a bigger book than anything live today.

**Exact data needed:** EODHD EOD for a liquid large/mid-cap US universe
(e.g., current + historical S&P 500/S&P 400 constituents, several hundred
tickers including delisted/removed members for survivorship-freedom) — several
hundred calls (1/ticker, full range).

**Costs:** Same as above; turnover is higher than #1/#3 (potentially dozens of
names changing rank monthly across a few-hundred-name universe) — model
realistic per-name commission + spread costs explicitly in the pre-registered
test, not just headline commission-free framing, since spread cost (not
commission) is what the net-of-cost literature above is actually worried
about.

**Integration — does NOT fit the current handful-of-symbols Sleeve pattern
without real work.** A `Sleeve` that ranks hundreds of names and holds a
rotating decile is a bigger `symbols()` list and a heavier `target_weights`
computation than anything currently built (`BtcTrendSleeve` holds 1 symbol;
`StaticSleeve` holds a fixed handful) — feasible within the `Sleeve` ABC's
contract (still a pure function of `history`), but a materially larger
engineering and data-pull effort than #1 or #3. Flag as **moderate effort**,
not small.

**Tail risks:** A liquid-only universe restriction may itself be
survivorship/self-selection biased if not constructed carefully (using
current index membership introduces look-ahead — must use historical,
point-in-time index constituents, which needs care with EODHD's delisted
symbol list).

---

### 5 (conditional). Crypto cross-sectional momentum — gated, do not scope yet

**Pre-registrable rule (if the gate clears):** Monthly cross-sectional
momentum/trend across the ~20–30 largest coins by market cap (excluding
stablecoins), long top-tercile, equal weight, extends the live BTC 4-week
trend sleeve (`src/firm/allocation/btc_trend.py`) to a basket.

**Evidence:** Two strong anchors — Liu & Tsyvinski (2021), *"Risks and Returns
of Cryptocurrency,"* *Review of Financial Studies* 34(6):2689–2727
([Oxford](https://academic.oup.com/rfs/article-abstract/34/6/2689/5912024)):
strong time-series crypto momentum, driven by network/adoption and
attention proxies. Liu, Tsyvinski & Wu (2022), *"Common Risk Factors in
Cryptocurrency,"* *J. Finance* 77(2):1133–1177
([Wiley](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13119),
[NBER](https://www.nber.org/papers/w25882)): 3-factor model (market, size,
momentum) captures 10 characteristic-based long-short crypto strategies.
**But:** decay evidence is real and post-2020 (post-July-2020 sample momentum
reportedly negative/insignificant per search summary — **UNVERIFIED**, exact
paper not pinned down this session), and realistic transaction-cost modeling
(taker fees, slippage as a function of trade size, funding costs) materially
erodes gross momentum profit in recent literature.

**THE GATE — must be resolved before this is scoped further:** it is
**UNVERIFIED whether EODHD's crypto EOD product includes historical prices for
delisted/dead coins** the same way its equity product explicitly does. EODHD's
own materials confirm equity delisted-data coverage in detail, and confirm
"6,403 delisted tickers" are queryable via `?delisted=1` on the `CC` exchange
code in the Exchange Symbol List
([confirmed](https://eodhd.com/financial-apis/list-supported-crypto-currencies))
— so the delisted list *exists* — but no source found this session confirms
that **EOD price history** is actually populated for those delisted-coin
symbols (as opposed to just their metadata being listed). Given crypto's very
high token failure/delisting rate, a basket-momentum backtest run on
live-coins-only would materially overstate returns via survivorship bias. **Do
not begin building this until a direct check** (pull EOD history for 5–10
known-dead/delisted coins from the `CC` delisted list and confirm non-empty,
plausible price series) resolves this — this is a ~10-call, five-minute check,
not a research question, and should be the very first thing done with the
subscription if this idea is pursued at all.

**Costs, integration:** Alpaca supports crypto with fractional quantities
(`fractional=True`, `time_in_force="gtc"`, matching `BtcTrendSleeve`'s
pattern) — natural extension of the live sleeve if the gate clears. Alpaca
crypto costs are tiered maker/taker (0.15%/0.25% at the lowest volume tier,
improving with 30-day volume —
[Alpaca](https://alpaca.markets/support/crypto-maker-taker-gmt-faq), page
states "effective March 13, 2023," not independently cross-checked against a
fresher primary page this session — re-verify before sizing).

---

## 2. Download plan for the month

EODHD Historian: **100,000 calls/day, 1,000 requests/min**
([confirmed](https://eodhd.com/financial-apis/api-limits)). Total available
over a 30-day month ≈ 3,000,000 calls — call *budget* is not the binding
constraint for anything in this brief; getting the **ticker universe and the
crypto delisted-data gate right, early**, is. Priority order:

| # | Request | Endpoint | Est. calls | Why first |
|---|---|---|---|---|
| 1 | US active + delisted symbol lists | `exchange-symbol-list/US`, `?delisted=1` | ~2 | Needed to scope every other pull; do this on day 1. |
| 2 | Crypto active + delisted symbol lists | `list-supported-crypto-currencies`, `CC?delisted=1` | ~2 | Needed for the candidate-#5 gate check. |
| 3 | **Crypto delisted-coin EOD gate check** (5–10 dead coins) | per-ticker EOD | ~10 | Resolves whether #5 is buildable at all — do this before anything else crypto-related. |
| 4 | Sector/industry ETF EOD, full range (#1 above) | per-ticker EOD | ~30 | Cheapest, cleanest candidate; get it running first. |
| 5 | Bond + commodity ETF EOD, full range (#3 above) | per-ticker EOD | ~25 | Same tier of effort as #4. |
| 6 | EODHD Commodities API, all 23 series | `commodities-api` | ~23 | Cross-check vs. bond/commodity ETF prices. |
| 7 | Liquid large/mid-cap universe EOD, incl. historical index members (#4/52-week-high) | per-ticker EOD | ~500–1,000 | Bigger pull; needs the point-in-time constituent list built first (from free sources + EODHD delisted list). |
| 8 | Breadth-universe EOD, capped ~3,000–5,000 large/mid-cap active+delisted-since-2010 names (#2 overlay) | per-ticker EOD | ~3,000–5,000 | Largest pull; schedule mid-month once #1–7 are validated. |
| 9 | News/sentiment depth check, 5–10 tickers, oldest available date | `news`, `sentiments` (5 calls each) | ~50–100 | Resolves the undocumented history-depth and point-in-time-revision questions (§3 below) — cheap, do it, but don't build a strategy on the result without the PIT question separately resolved. |
| 10 | Splits/dividends, full history, all tickers already pulled above | `api-splits-dividends` or bulk | ~included in per-ticker pulls | For candidate C (dividend-month premium) if pursued from the Stage A list. |

Everything above totals well under 10,000 calls against a 3,000,000-call
budget — there is no reason to ration; the reason to sequence it this way is
so the crypto gate (#3) and universe-scoping (#1–2) happen before any larger
commitment of research time, not API budget.

---

## 3. Stage A long list (all candidates screened)

**Cross-sectional equity anomalies** (Hou-Xue-Zhang 2020, McLean-Pontiff 2016,
Jensen-Kelly-Pedersen 2023, Novy-Marx-Velikov 2016, Open Source Asset Pricing
used as meta-evidence throughout):

- **Short-term reversal** — Jegadeesh (1990)/Lehmann (1990), gross-only
  anchors; edge concentrated in illiquid names, decimalization/HFT decay since
  ~2001. **DROP.**
- **Industry momentum** — see Stage B #1. **KEEP (pilot).**
- **52-week-high proximity** — see Stage B #4. **KEEP (conditional, moderate
  effort).**
- **Low volatility / low beta (BAB)** — Frazzini & Pedersen (2014); survives a
  serious 2022 methodological attack (Novy-Marx & Velikov) at ~half the
  headline Sharpe, but only evidenced at ~100+-name book scale; no
  concentrated (20–50 name) version found in the literature. **DROP for
  near-term integration** (real effect, wrong scale for this system today).
- **Idiosyncratic volatility puzzle** — Ang, Hodrick, Xing & Zhang (2006);
  fails HXZ's stricter replication outright (t < 2 at every horizon) and the
  leading explanation is a reversal/microstructure artifact, not a distinct
  premium. **DROP.**
- **MAX effect / lottery demand** — Bali, Cakici & Whitelaw (2011); explicitly
  named as insignificant in the HXZ replication. **DROP.**
- **Turnover / liquidity (Amihud; Datar-Naik-Radcliffe)** — fails HXZ
  replication under value-weighting, and even the original effect needs an
  out-of-scope illiquid-microcap book. **DROP.**
- **Cross-sectional seasonality (Heston-Sadka 2008)** — real original result,
  but Keloharju, Linnainmaa & Nyberg (2016) show seasonalities and their
  reversals sum to ~zero over a full year; no post-2010 replication found;
  same data-mining profile as the already-rejected turn-of-month/pre-FOMC
  candidate. **DROP** without a fresh pre-registered test.

**Dividend-based:**

- **Ex-dividend price behavior** — real, well-replicated tax-clientele effect
  (Elton-Gruber lineage), but retail-scale profit is arbitraged away by
  spread/adjustment costs; institutional/dealer-scale only. **DROP.**
- **Dividend initiations/increases (drift)** — Michaely, Thaler & Womack
  (1995) is a real, large original effect, but later work shows reduced
  magnitude and possible confound-driven fragility (**UNVERIFIED** specific
  citation); also needs an announcement-date feed EODHD may not cleanly
  provide (dividend history gives ex-div/pay dates, not necessarily the
  original announcement date). **DROP / weak lead at best.**
- **Dividend-month premium** — see Stage B discussion; independently
  replicated in Germany and the Nordics with an explicit net-of-cost check in
  one, but **no US-specific post-2013 replication found**. **Tentative KEEP
  for further scoping**, not included in the top-5 Stage B slots this round
  given the shortlist is already full of comparably-or-better-evidenced ideas;
  revisit if one of the top 5 fails its pre-registered test.

**Split events:**

- **Post-split drift** — Ikenberry, Rankine & Stice (1996) original effect;
  multiple lines of evidence (a 2022 decimalization/HFT-driven-disappearance
  study, **partially verified** — secondary summary only; a pre-2000
  methodology critique) converge on the drift being extinct post-decimalization.
  **DROP.**

**Forex:**

- **FX momentum** — Menkhoff, Sarno, Schmeling & Schrimpf (2012), real
  pre-2012 effect, credible post-2010/post-publication decay evidence
  (McLean-Pontiff-style dissemination effect, BIS corroboration). **DROP.**
- **FX value (PPP-based)** — the anchor authors' own follow-up
  (Menkhoff et al., *"Currency Value,"*
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2492082)) argue
  the predictability is better explained as a risk premium for cross-country
  fundamental differences than a tradeable mispricing — the paper you'd cite
  to justify this undercuts it. **DROP.** (Also: IBKR-only for spot execution;
  Alpaca has none.)

**Crypto:**

- **Cross-sectional momentum/trend, basket of large coins** — see Stage B #5
  (conditional/gated). **KEEP, gated on the delisted-coin-history check.**

**Global/international:**

- **Country-index momentum + value combined** — Asness, Moskowitz & Pedersen
  (2013); Balvers & Wu (2006, sample ends 1999, one of the few anchors with an
  affirmative original net-of-cost claim). A broader 42-country, 1996–2017
  replication finds costs "largely lethal" to pure single-signal versions.
  **Weak/marginal KEEP**, only as the combined value+momentum design, unproven
  post-2010 — not included in top 5 given weaker sourcing than the shortlist
  items.
- **International ETF lead-lag / stale-NAV arbitrage** — Zitzewitz/Petajisto
  literature is rigorous but describes a 1990s–2000s inefficiency specifically
  targeted and largely remediated by fair-value-pricing reforms; no post-2010
  net-of-cost replication found for a direct-trading (non-mutual-fund)
  implementation. One potentially decisive recent paper on overnight/morning
  gaps could not be accessed (paywalled/403). **DROP**, pending one more
  access attempt if the owner has a library proxy.

**Commodity/bond ETF:**

- **Bond yield-change momentum + multi-commodity ETF trend** — see Stage B #3.
  **KEEP.**
- **Gold seasonality ("autumn effect")** — real academic citation (Baur,
  ScienceDirect), but explicitly not tested net-of-cost in what was found, and
  precious metals aren't even in EODHD's commodities coverage (discontinued by
  FRED) — would need gold ETF (GLD) prices instead, which is fine, but the
  net-of-cost gap remains. **DROP as a standalone; minor overlay idea at best.**

**Breadth:**

- **Market breadth as risk/regime overlay** — see Stage B #2. **KEEP.**

**News-sentiment:**

- **UNVERIFIED on two separate axes: history depth and point-in-time
  status.** EODHD's own docs state only "Paid plans: full history" with no
  specific start date, and say nothing about whether sentiment scores can be
  revised after initial publication. Both are genuine documentation gaps, not
  research failures — resolve with the cheap depth-check in the download plan
  (item 9) before deciding anything. **Do not build a backtest on this without
  first confirming both; a forward-test-only framing may be all that's
  supportable even after checking.**

---

## 4. Don't-bother list

Popular ideas that fail on evidence, cost, data, or detectability, with the
single strongest reason each:

- **Ex-dividend "dividend capture"** — retail-scale profit is arbitraged away
  by bid-ask spread and the price adjustment itself; institutional/dealer
  scale only.
- **Post-split drift** — multiple convergent sources place its extinction at
  ~2001–2006, driven by decimalization and HFT-driven price discovery.
- **Idiosyncratic volatility, MAX effect, turnover/liquidity anomalies** — all
  three explicitly and directly fail Hou-Xue-Zhang's (2020) stricter,
  standard replication (t < 1.96).
- **Short-term reversal** — real but mechanically a small/illiquid-stock
  microstructure return, compressed to near-zero for tradable liquid names
  since decimalization.
- **FX carry, FX momentum, FX "value"** — carry already ruled out per the
  brief; momentum shows credible post-publication decay; "value" is
  undercut by its own anchor authors' later work (a risk-premium story, not
  mispricing).
- **International ETF stale-NAV lead-lag** — a real, 1990s–2000s effect
  specifically targeted and largely closed by fair-value-pricing reforms; no
  post-2010 net-of-cost replication found.
- **Cross-sectional seasonality (Heston-Sadka style)** — Keloharju et al.
  (2016) show the seasonal pattern and its reversal roughly cancel over a full
  year; shares the exact data-mining profile of the already-rejected
  turn-of-month/pre-FOMC candidate.
- **Gold/precious-metals seasonality via EODHD Commodities** — not
  buildable from the Commodities API at all; gold/silver/platinum/palladium
  were discontinued (LBMA licensing changes at FRED, EODHD's upstream source).
- **Low-vol/BAB at this system's current scale** — the real, well-evidenced
  version of this effect needs a book of ~100+ names for stable beta
  estimation; nothing found supports a 20–50-name concentrated version, which
  is what this system's architecture currently supports.

---

## 5. Open questions for the owner

1. **Crypto delisted-coin EOD coverage** (candidate #5's gate) — worth
   spending ~10 API calls verifying before deciding whether to scope this at
   all. Want it done as part of this write-up's follow-up, or held for a
   separate session?
2. **Breadth-as-overlay is explicitly outside the `Sleeve` shape** — it needs
   new `Allocator`-level machinery (a way for one signal to scale another
   sleeve's weight). Worth the engineering effort for an unproven-but-
   well-sourced defensive overlay, or should it wait until industry momentum
   (#1) and bond/commodity trend (#3) — both of which fit the existing
   architecture with no allocator changes — have results?
3. **52-week-high (#4) needs a point-in-time historical index-constituent
   list** (not current S&P 500/400 membership, which would be look-ahead
   biased) — is there an existing source for this in the repo/free data
   already, or does it need to be built from EODHD's delisted-symbol data plus
   external historical-constituent research?
4. **News-sentiment depth/PIT status** — worth spending ~50–100 calls to
   empirically check before writing this idea off entirely, or is a forward-
   test-only framing (regardless of what the depth check shows) acceptable
   given the PIT-revision question may be unanswerable either way?
5. **EODHD data-retention-after-cancellation is UNVERIFIED** — the licence
   terms don't address whether downloaded data may be kept and used after the
   one-month subscription lapses. Worth a direct support-ticket question to
   EODHD before the month is up, given every candidate above depends on
   keeping this data usable afterward?
6. Given only 4 candidates (5 gated) survived screening and none is a strong
   lead, does the owner want all 4 pre-registered and tested in parallel this
   month, or ranked into a single one-at-a-time sequence (as the existing
   alt-premia work was run)?

---

## 6. Sources (access date 2026-09-30 unless noted)

**Meta-evidence / methodology:**
- McLean & Pontiff (2016), *J. Finance* 71:5–32. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365
- Hou, Xue & Zhang (2020), *Rev. Fin. Studies* 33:2019–2133. https://academic.oup.com/rfs/article-abstract/33/5/2019/5236964 (NBER WP23394: https://www.nber.org/papers/w23394)
- Jensen, Kelly & Pedersen (2023), *J. Finance* 78:2465–2518. https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249
- Chen & Zimmermann, Open Source Asset Pricing. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3604626, https://www.openassetpricing.com/
- Novy-Marx & Velikov (2016), *Rev. Fin. Studies* 29:104–147. https://academic.oup.com/rfs/article-abstract/29/1/104/1844518

**Equity anomalies:**
- Jegadeesh (1990), *J. Finance* 45:881–898. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1990.tb05110.x
- Lehmann (1990), *QJE* 105:1–28. https://academic.oup.com/qje/article-abstract/105/1/1/1928416
- Moskowitz & Grinblatt (1999), *J. Finance* 54:1249–1290. https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00146
- Hoberg & Phillips, *JFQA*. https://www.cambridge.org/core/services/aop-cambridge-core/content/view/6E04891A7AD30C24C21CBDE37EC06BF4/S0022109018000479a.pdf/text-based-industry-momentum.pdf
- George & Hwang (2004), *J. Finance* 59:2145–2176. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x
- Bettman, Sault & von Reibnitz (2010). https://journals.sagepub.com/doi/10.1177/0312896210385282
- Frazzini & Pedersen (2014), *J. Fin. Econ.* 111:1–25. https://www.nber.org/system/files/working_papers/w16601/w16601.pdf
- Novy-Marx & Velikov (2022). https://www.ssrn.com/abstract=3300965
- Ang, Hodrick, Xing & Zhang (2006), *J. Finance* 61:259–299. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00836.x
- Fu (2009), *J. Fin. Econ.* 91:24–37. https://www.sciencedirect.com/science/article/abs/pii/S0304405X08001694
- Bali, Cakici & Whitelaw (2011), *J. Fin. Econ.* 99:427–446. https://www.sciencedirect.com/science/article/abs/pii/S0304405X1000190X
- Amihud (2002), *J. Fin. Markets* 5:31–56. https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf
- Datar, Naik & Radcliffe (1998), *J. Fin. Markets* 1:203–219. https://www.sciencedirect.com/science/article/abs/pii/S1386418197000049
- Heston & Sadka (2008), *J. Fin. Econ.* 87:418–445. https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/063f-sadka.pdf
- Keloharju, Linnainmaa & Nyberg (2016), *J. Finance* 71:1557–1590. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12398

**Dividends/splits:**
- Elton, Gruber & Blake, SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=363620
- Graham, Michaely & Roberts (2003), *J. Finance* 58:2611–2636. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=318483
- Whitworth (2010), *Financial Management*. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1755-053X.2010.01078.x
- Michaely, Thaler & Womack (1995), *J. Finance* 50:573–608. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1995.tb04796.x (NBER: https://www.nber.org/papers/w4778)
- Hartzmark & Solomon (2013), *J. Fin. Econ.* 109:640–660. https://doi.org/10.2139/ssrn.1930620
- Germany replication (2021), *J. Asset Management*. https://link.springer.com/article/10.1057/s41260-021-00215-3
- Nordics replication (Aalto). https://aaltodoc.aalto.fi/handle/123456789/41312
- Ikenberry, Rankine & Stice (1996), *JFQA*. https://www.ssrn.com/abstract=7929

**FX:**
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), *J. Fin. Econ.* 106:660–684. https://faculty.washington.edu/ss1110/IF/Sarno%20Currency%20Momentum%20JFE%20(1).pdf
- BIS Quarterly Review (2011). https://www.bis.org/publ/qtrpdf/r_qt1112x.htm
- Asness, Moskowitz & Pedersen (2013), *J. Finance* 68:929–985. https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf
- Menkhoff, Sarno, Schmeling & Schrimpf, "Currency Value." https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2492082

**Crypto:**
- Liu & Tsyvinski (2021), *Rev. Fin. Studies* 34:2689–2727. https://academic.oup.com/rfs/article-abstract/34/6/2689/5912024
- Liu, Tsyvinski & Wu (2022), *J. Finance* 77:1133–1177. https://onlinelibrary.wiley.com/doi/10.1111/jofi.13119 (NBER: https://www.nber.org/papers/w25882)
- EODHD crypto coverage. https://eodhd.com/financial-apis/list-supported-crypto-currencies

**International/breadth:**
- Balvers & Wu (2006), *J. Empirical Finance*. https://static1.squarespace.com/static/58ab3694e4fcb58d2b18adb5/t/5ad0c76d562fa7b8c913f720/1523631982119/Momentum+%26+Mean+Reversion+Across+Equity+Markets_Balvers+%26+Wu_2005.pdf
- Zitzewitz, "Who Cares About Shareholders?" https://papers.ssrn.com/sol3/papers.cfm?abstract_id=832314
- Petajisto, "Inefficiencies in the Pricing of ETFs." http://www.petajisto.net/papers/etf26.pdf
- "Herding for profits," *Int. Rev. Fin. Analysis*. https://www.sciencedirect.com/science/article/pii/S0264999319312982
- Chen, Hong & Stein (2001), SSRN. https://www.ssrn.com/abstract=262106

**Bond/commodity:**
- Sihvonen, "Yield Curve Momentum," *Review of Finance*. https://revfin.org/yield-curve-momentum/ / https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice
- Baur, "The autumn effect of gold." https://www.sciencedirect.com/science/article/abs/pii/S0275531912000323

**EODHD primary docs:**
- Pricing/plans: https://eodhd.com/pricing
- EOD historical depth: https://eodhd.com/financial-apis/api-for-historical-data-and-volumes
- Delisted stocks: https://eodhd.com/financial-apis/delisted-stock-companies-data-2
- Bulk EOD/splits/dividends: https://eodhd.com/financial-apis/bulk-api-eod-splits-dividends
- Splits/dividends per-ticker: https://eodhd.com/financial-apis/api-splits-dividends
- News/sentiment: https://eodhd.com/financial-apis/stock-market-financial-news-api
- Forex: https://eodhd.com/lp/forex-api
- Crypto: https://eodhd.com/financial-apis/list-supported-crypto-currencies
- Commodities (beta): https://eodhd.com/financial-apis/commodities-api-historical-prices-for-oil-gas-metals-agriculture-beta
- US Treasury rates (beta): https://eodhd.com/financial-apis/us-treasury-ust-interest-rates-api-beta
- API limits/costs: https://eodhd.com/financial-apis/api-limits
- Terms & conditions: https://eodhd.com/financial-apis/terms-conditions
- Commercial vs. personal licence: https://eodhd.com/financial-apis/commercial-vs-personal-license-use

**Broker costs:**
- Alpaca equities fee schedule (rev. 2026-09-17): https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf
- Alpaca crypto maker/taker (states "effective March 13, 2023"): https://alpaca.markets/support/crypto-maker-taker-gmt-faq
- IBKR stock commissions: https://www.interactivebrokers.com/en/pricing/commissions-stocks.php
- IBKR spot FX commissions: https://www.interactivebrokers.com/en/pricing/commissions-spot-currencies.php

**Repo files referenced:**
- `docs/edge_search_verdict_2026_09.md`
- `docs/allocation_deploy_runbook.md`
- `src/firm/allocation/sleeves.py`
- `src/firm/allocation/btc_trend.py`
- `src/firm/allocation/allocator.py`
