# External research: return sources beyond US stocks (2026-09-30)

**Status: primary research from an external agent, pasted in by the owner.
Read the corrections in §0 first — they change the report's own top
recommendation.** Nothing here has been pre-registered or tested against this
system's evaluation standard. It is an input to that process, not a result of
it.

Brief used: `docs/prompts/research_brief_new_instruments.md` (the repo-access version;
the report cites `src/firm/...` paths directly).

## 0. Corrections made after a direct repo read

The report flagged two things it couldn't verify itself and asked the owner to
resolve by reading the code (its own "Repo-read follow-ups", Part 5). Both are
now resolved:

1. **The report's #2 pick — a "passive bond/commodity diversification sleeve,
   2–4 person-days, mostly `config/live.yaml` changes" — is not buildable that
   way.** It assumed `allocation_method: risk_parity` could host a fixed,
   always-on allocation. It can't: `TraderAgent._risk_parity`
   (`src/firm/agents/trader.py:256`) only inverse-vol-weights symbols the 11
   strategies *already* produced a nonzero conviction on that cycle — it never
   adds an always-on position in an instrument no strategy is signalling on.
   A static buy-and-hold sleeve needs the same kind of standalone allocator
   proposed for the 60/40 fallback in `docs/edge_search_verdict_2026_09.md`
   §4, not a config change. That raises its integration effort from "2–4
   days" to roughly the same order as the 60/40 proposal.

2. **A quick empirical check of the report's own suggested test** ("does it
   help when it's needed most" — compare book vs. book+sleeve across crisis
   windows), run just now on data already cached from Step 2 (a 50/30/20
   IEF/GLD/DBC sleeve at 30% of NAV, blended with the real split-fixed
   2020–2026 live-config book):

   | Window | Book max drawdown | 70/30 blend max drawdown | Book total | Blend total |
   |---|---|---|---|---|
   | 2020 COVID crash | 5.5% | 5.0% | −3.7% | −2.0% |
   | 2022 selloff | 3.6% | 3.2% | **+6.0%** | **+1.8%** |
   | Full 2020–2026 | 11.2% | 6.9% | +24.6% | +34.6% |

   **This complicates the report's own rationale.** The classic 60/40
   diversification argument assumes an equity-like book that draws down hard
   in a crisis. This system's live book has a beta near 0 and is already
   low-volatility (§2 of the verdict), so it barely draws down on its own —
   in the 2022 selloff the book was *positive* and the bond/commodity sleeve
   dragged it down, the well-documented 2022 60/40 break showing up here too.
   The better full-period number is mostly the sleeve's own return, not a
   diversification effect on this particular book. This is a single scenario,
   not a pre-registered test — it's a reason for caution before treating this
   as the "safe, cheap win," not a rejection.

Not independently checked: IBKR TASE-in-paper-account support, the Israeli tax
questions, and every cost/regulatory figure in the report. Treat those as the
report states them — several are itself marked UNVERIFIED.

## 1. The report

The rest of this file is the agent's report, unedited.

---

# Tradeable Return Sources Beyond US Stocks: A Research Brief for a Small, Israeli-Resident Systematic Trading System

## Executive Summary

After a two-stage evidence survey covering twelve instrument/strategy classes and a deep dive on five shortlisted candidates, the honest conclusion is that **most popular "alternative" ideas do not clear this system's own evaluation bar**, and the ones that might are expensive to build. Key findings:

* **Multi-asset futures trend-following is the strongest candidate**, resting on a century of replicated, largely undecayed evidence, but its live, post-2010 Sharpe (≈0.15–0.5, based on the SG Trend Index and AQR's own managed-futures analysis) is far below the 1.0+ figures in the original academic papers, and it requires 36–54 person-days of engineering because the system has no concept of contract multipliers, margin, or rolls anywhere in `src/firm/brokers/base.py`, `src/firm/portfolio/state.py`, `src/firm/agents/risk.py`, or `src/firm/backtest/engine.py`.

* **A passive, unlevered bond/commodity diversification sleeve is the highest value-for-effort item found in this entire brief** — 2–4 person-days of integration (mostly `config/live.yaml` changes) because it reuses the existing equity/ETF execution path entirely, but it must be sold honestly: it is a diversification argument, not an alpha claim, and the 2022 stock-bond correlation break (60/40 down ≈17%, its worst year since 1937) shows plainly that the bond leg fails exactly when inflation-driven equity selloffs occur, leaving the commodity leg to do the diversifying work in that regime.

* **SEC-filings-driven insider-purchase clustering is worth a pre-registered test but PEAD should be dropped outright** — Tier-1 evidence (Kettell, McInnis & Zhao) shows post-earnings-drift is "indistinguishable from zero" after 2017, and a proper point-in-time SUE signal needs paid analyst-consensus data (I/B/E/S/WRDS) that has no free retail-accessible equivalent, which is the single biggest blocker found anywhere in this research.

* **FX carry is not worth building; FX momentum is borderline.** Multiple independent, peer-reviewed sources confirm the "carry died after the 2008 financial crisis" narrative is real, not a myth — post-2010 carry Sharpe estimates run 0.04–0.16, an order of magnitude below this system's 0.4–0.9 detection floor.

* **Crypto carry beyond spot BTC (CME basis, perpetual funding) is the clearest "not worth it" of the five deep dives.** The BIS's own research shows the arbitrage would have faced forced liquidation in over half of all sample months at just 10x leverage, and the engineering cost (53–83 person-days, the largest in this brief) because the system's `Broker` abstraction assumes one venue per engine instance and cannot natively hold a two-legged basis position.

* **CFDs, options-based short-volatility variants, individual commodities, VIX-futures-term-structure trades, and standalone PEAD** all fail on evidence, cost, or duplication-of-an-already-tested premium, and are detailed in the "Don't Bother" list below.

---

## Detailed Analysis

### Part 1 — Ranked Shortlist (Stage B Deep-Dive Candidates)

#### 1. Multi-asset futures trend-following

**Pre-registrable rule.** A 19-contract universe spanning equity index (ES, NQ, RTY, FDAX), rates (ZF, ZN, ZB, FGBL, FGBS), FX (6E, 6J, 6B, 6A, 6C) and commodities (CL, GC, HG, ZC, ZS); signal = equal blend of 1-, 3- and 12-month time-series momentum, sign(return) × volatility scaling; sizing = inverse-volatility to a 12% annualized portfolio target, matching the system's existing `vol_targeting_enabled` convention in `src/firm/agents/risk.py`; rebalance monthly, fully compatible with "decide once per completed daily bar."

**Evidence.** Moskowitz, Ooi & Pedersen's "Time Series Momentum" (*JFE* 2012, 58 markets, 1985–2009) found diversified Sharpe of 1.1–1.8; Hurst, Ooi & Pedersen's "A Century of Evidence" (AQR/JOIM, 1880–2016, 67 markets) found the strategy positive in nearly every decade including the Depression and 2008, and Babu et al.'s "Trends Everywhere" (JOIM 2020) is a genuine out-of-sample test on 82 securities not in the original study, still finding a gross Sharpe of 1.17. But **post-2010 live performance has visibly decayed**: the SG Trend Index shows a full-period Sharpe of only ~0.42–0.47, and a conservative net-of-cost estimate for this system's specific 19-market universe, after modeling roll/slippage drag of 0.3–0.6%/year, implies a realistic live Sharpe in the **0.15–0.5 range**, not the 1.0+ headline figures. The honest conclusion: **bonds failed as a diversifier in 2022; commodities (PDBC/GLD, up materially that year) did the diversifying work instead.** This supports a modest (20–35% of NAV), unlevered, multi-sleeve allocation — not a bond-only or leveraged risk-parity replacement of the core book.

**All-in cost.** Weighted expense ratio across a typical 60/30/10 (bonds/commodities/gold) split within the diversifying sleeve ≈0.32%/year on that sleeve, or ~0.10–0.13%/year on total NAV at a 30–40% allocation; near-zero incremental commission or spread drag given monthly rebalancing of highly liquid, large-AUM ETFs.

**Data source and cost.** Tiingo (already free and in the fallback chain) is fully adequate — all five ETFs are long-lived, high-AUM, non-survivorship-prone.

**Recommended provider and monthly cost.** Interactive Brokers, already integrated — $0 incremental.

**Main tail risk.** The 2022 stock-bond correlation break (above); March 2020 fixed-income ETF NAV discounts (TLT's average discount widened from 0.13% to 0.86%, with intraday discounts reported as large as ~5%); negative WTI futures affecting PDBC/DBC's underlying legs.

**Testability verdict: not a statistical-significance question — reframed as "does it help when it's needed most."** Test by comparing [current book] vs. [current book + sleeve] across 2008, 2013 taper tantrum, March 2020, and 2022 specifically, reporting Sharpe/max-drawdown per window rather than one blended number.

**Integration effort: ~2–4 person-days** — largely `config/live.yaml` changes (`universe.symbols` additions, new `sector_map` entries for "fixed_income"/"commodity"), contingent on confirming what the existing `allocation_method: risk_parity` config option actually implements (flagged for a direct repo read of `src/firm/agents/trader.py`).

#### 3. SEC-filings-driven insider-purchase clustering (drop the PEAD leg)

**Pre-registrable rule.** Universe: all US-listed common stocks, $50M–$2B market cap, ≥$500K 20-day average dollar volume (~2,000–3,000 tickers, versus today's fixed ~25-name book). Signal: a "cluster" = 3+ distinct insiders filing Form 4 code-"P" open-market purchases within a 30-day window, at least one classified "opportunistic" per Cohen, Malloy & Pomorski's definition (not a routine, pre-scheduled trade). Hold 3–6 months, rebalance monthly.

**Evidence and its decay.** Cohen, Malloy & Pomorski (*Journal of Finance*, 2012, 1986–2007) found opportunistic-buy abnormal returns of 82bp/month value-weighted (t=2.15), gross of costs. A 2024 replication on 2008–2024 data found only 0.2–0.3%/month — roughly one-third of the original estimate — attributed to the SEC's 2-business-day filing deadline enabling faster algorithmic monitoring since 2008 (flagged UNVERIFIED pending direct retrieval, but consistent across two independent sources found). Small-cap round-trip transaction costs (median effective spreads of 0.92–1.235% for the smallest cap decile, per the SEC's own market-quality research) plausibly exceed the decayed gross alpha for the most micro-cap, most cluster-signal-dense names, meaning **this is likely negative net-of-cost unless position sizing is restricted to the more liquid end of the qualifying universe**, where the alpha itself shrinks toward 1–2%/year.

**Why it wouldn't repeat the failure of the existing `event_driven` strategy.** The current strategy is documented as a "simplified PEAD proxy; needs fundamentals" — a crude implementation with no true analyst-consensus-based SUE and no insider-transaction signal at all, running over a fixed 25-name mega-cap universe where insider clustering and PEAD are both structurally weakest. Insider-cluster detection via Form 4 parsing is an entirely new data source, not a re-run.

**Drop PEAD.** Kettell, McInnis & Zhao (Columbia CEASA) find SUE-decile hedge returns are "indistinguishable from zero" after 2017, gross of costs — a Tier-1 finding that closes the door on standalone PEAD regardless of engineering effort. Compounding this, a proper SUE computation needs historical analyst-consensus estimates, which exist only as a paid, WRDS/I-B-E-S-licensed product with no free retail-accessible point-in-time equivalent — the single biggest data blocker identified anywhere in this brief.

**All-in cost.** IBKR/Alpaca commissions are immaterial (a few basis points); the binding cost is small-cap bid-ask spread (0.9–1.2%+ one-way) and market impact, which the existing `RiskAgent.max_participation_pct` will bind much harder against on a $500K–$5M-ADV universe than on today's mega-cap book.

**Data source and cost.** SEC EDGAR's own APIs (10 requests/second, IP-wide fair-access limit, mandatory descriptive User-Agent header) are free and adequate for Form 4 self-parsing; a paid convenience vendor (e.g., Quiver Quantitative, ~$75/month) buys convenience, not a capability gap, so self-building is the right call.

**Recommended provider and monthly cost.** No new broker needed — trades existing US equities via the current `Broker` interface. Data cost: $0/month for the insider-cluster leg (EDGAR only).

**Main tail risk.** A structural, unavoidable gap-risk mismatch: earnings releases and Form 4 filings typically post after-hours, so a daily-bar-deciding system's entry price and the signal-generating event sit on opposite sides of an overnight gap by construction; small-cap liquidity risk in stressed markets; EDGAR rate-limit outage risk that could starve the whole system's fundamentals ingestion on the same IP.

**Testability verdict: borderline-detectable for insider clusters; not detectable for PEAD (drop it).** ~300–1,500 quasi-independent bets/year for insider clusters after correlation haircuts — nominally well-powered, but bet-count arithmetic cannot rescue a per-bet edge that has decayed toward the cost floor.

**Integration effort: ~24–36 person-days** (with PEAD dropped) — new `insider_transactions()` accessor on `PitView` in `src/firm/strategies/base.py`, a new `insider_cluster.py` strategy module, a new EDGAR Form-4 ingestion pipeline, extension of the existing `sp500_dynamic_universe`/`danelfin_dynamic_universe` scaffolding in `src/firm/live/dynamic_universe_state.py` to full small/micro-cap breadth, and new small-cap-specific `RiskAgent` parameters.

#### 4. FX momentum via IBKR IdealPro (carry: do not pursue as capital allocation)

**Pre-registrable rule.** Nine G10 pairs vs. USD (AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK); monthly formation on 12-month total spot return; long top-3/short bottom-3 tercile, equal-weighted; daily monitoring only for risk-cap triggers.

**Evidence.** Menkhoff, Sarno, Schmeling & Schrimpf (*JFE* 2012) find up to ~10%/year cross-sectional momentum spread, 1976–2010, robust to transaction-cost adjustment, with costs concentrated in minor/high-spread currencies not in this G10-only scope. **Carry is the item this brief must state plainly does not work**: three independent peer-reviewed/practitioner sources converge on the "carry died post-GFC" finding — Dupuy finds full-sample carry Sharpe of 0.76 falling to 0.06 since 2008; a second peer-reviewed paper finds post-2010 dollar-based carry Sharpe of 0.04; Macrosynergy's fully-disclosed-methodology research finds "almost no positive PnL contribution since 2010." This is a *verified*, not assumed, decay.

**Why this differs from the failed ETF momentum test.** It is a cross-sectional (relative-value), not time-series (self-referential), signal on a different asset class (currencies) with a distinct risk mechanism (crash/skewness compensation per Brunnermeier, Nagel & Pedersen, rather than equity-ETF behavioral under/over-reaction).

**All-in cost.** IBKR IdealPro charges 0.2bp of trade value with a $2.50/leg minimum — genuinely institutional pricing — but the $25,000 IdealPro minimum-lot threshold means a $100K–250K account sizing individual legs at 5–10% of NAV will frequently fall below the threshold and route as costlier odd lots, a real, previously unquantified constraint; only accounts of roughly $500K–$1M clear true IdealPro pricing on all legs.

**Data source and cost.** Spot prices via IBKR's own free `reqHistoricalData`/`Forex()` API (Tiingo does not cover FX); interest-rate differentials (needed for carry, not momentum) via FRED's free OECD short-term-rate series per country.

**Recommended provider and monthly cost.** Interactive Brokers, already integrated — $0 incremental.

**Main tail risk.** The January 15, 2015 CHF de-peg: CHF appreciated up to 30–39% intraday; a rule ranking currencies by rate differential would mechanically have been short CHF whenever its near-zero/negative policy rate sat in the bottom tercile, which was frequently the case through 2011–2022 — a textbook realization of carry's crash-risk premium, and direct evidence for why carry is being deprioritized here.

**Testability verdict: momentum borderline-detectable; carry not detectable — deprioritize carry.** Momentum's effective independent bets (~100–150 of 180–240 nominal monthly observations, since momentum P&L is closer to i.i.d.) combined with a plausible net Sharpe of 0.4–0.6 sits right at this system's detection edge; carry's already-weak post-2010 gross Sharpe (0.04–0.16) sits well below the 0.4–0.9 floor regardless of bet count.

**Integration effort: ~20–28 person-days** — a new `Forex` contract-building branch in `src/firm/brokers/ibkr.py`, a decision to keep USD-notional-only portfolio accounting rather than a native multi-currency ledger in `src/firm/portfolio/state.py`, a new FX/rate-differential data provider, and a new financing-accrual model in the backtest engine distinct from the existing equity-short-borrow mechanism.

#### 5. Crypto carry beyond spot BTC — recommend NOT building this now

**Pre-registrable rule (for completeness).** CME cash-and-carry: short MBT/MET micro futures hedged with spot BTC/ETH held at Kraken (IBKR's Paxos spot crypto is explicitly US-residents-only), entering only when annualized basis exceeds ~15–20% (roughly 2 standard deviations above the BIS paper's ~6.4–8% documented mean); perpetual funding carry: short Kraken Derivatives perpetuals hedged with Kraken spot when trailing funding exceeds 2–3x the ~11%/year "normal regime" median.

**Evidence, and why the honest answer is "not now."** The BIS's own paper on this exact trade (Schmeling, Schrimpf & Todorov, April 2019–July 2024) finds that because CME does not permit cross-margining with spot BTC, a textbook 10x-leveraged cash-and-carry trade would have faced forced liquidation in **over half the sample months**, and the January 2024 spot BTC ETF launch has already compressed CME carry by ~5 percentage points (97% of the mean) as institutional capital arbitrages it away. A separate paper on perpetual funding finds price convergence, not the funding payments themselves, drives two-thirds to three-quarters of total realized return — the "steady carry" story is weaker than marketed — and the FTX collapse drove negative funding simultaneously across every solvent exchange, direct evidence of correlated counterparty contagion.

**Why it's structurally new but not worth building yet.** It genuinely diversifies the failure mode of the only validated lead in this whole research program (spot-BTC trend, itself only weakly significant) since carry is a market-neutral relative-value bet rather than a directional one — but a cadence mismatch is unavoidable: funding settles every 8 hours and liquidation risk is continuous/intraday, while this system decides once per completed daily bar.

**All-in cost.** Kraken spot fees 0.06–0.80% (tiered by 30-day volume); Kraken Derivatives 0.02–0.05%; CME MBT via IBKR $0.85 down to $0.43/contract (tiered) plus ~$1.15 exchange-fee offset; CME margin on a full BTC contract ≈$20,000+ (indicative, third-party sourced, UNVERIFIED against CME's own table).

**Data source and cost.** No free source for CME futures curves or Kraken funding-rate history in point-in-time-correct form exists in the current stack; would need CME/Barchart-sourced data plus Kraken's own API, with real survivorship-bias risk given multiple historical exchange failures (FTX and others).

**Recommended provider and monthly cost.** Kraken for spot/perpetuals — but flagged plainly: the entity actually contracting with an Israeli client is **Payward Trading Ltd (British Virgin Islands)** for spot/margin and **Payward Digital Solutions Ltd (Bermuda, Bermuda Monetary Authority Class F license)** for derivatives, materially weaker consumer protection than Kraken's FCA/MiCA-regulated EEA/UK book; IBKR (already integrated) covers only the CME-futures leg. Kraken has no public spot demo/testnet at all (only a request-based UAT for Business Pro accounts) — the spot leg of any basis trade cannot be realistically paper-tested.

**Main tail risk.** The BIS liquidation-cascade finding itself (above); FTX-contagion negative funding; offshore-entity counterparty/custody risk; 24/7 gap risk against a once-daily decision cadence.

**Testability verdict: borderline-to-not-detectable.** Because basis/funding regimes persist for weeks (highly autocorrelated at short lags per the BIS paper), the truly independent bet count across BTC+ETH is only ~12–26/year — comparable to, not better than, the already-weak spot-BTC-trend result's effective sample size.

**Integration effort: ~53–83 person-days — the largest in this brief.** `src/firm/brokers/base.py`'s `Broker` abstraction assumes exactly one execution venue per engine instance and has no concept of a two-legged, cross-venue basis position; a new supervisory orchestration layer above two mirrored engine instances (or a genuinely new abstraction) would be needed, plus a new `KrakenBroker` adapter (spot and derivatives use separate authentication systems), a new funding-rate/futures-curve data provider, sub-daily cash-flow accrual tracking in `src/firm/portfolio/state.py`, and cross-venue margin/liquidation-buffer logic in `src/firm/agents/risk.py`. **The honest conclusion: this candidate's engineering cost plausibly exceeds its risk-adjusted expected edge for this system today.**

---

### Part 2 — Full Stage A Evidence Survey

#### 1. Listed options (premium selling, spreads, collars, dispersion/skew)

The CBOE BuyWrite Index (BXM) has a well-replicated history (Ibbotson 2004, Callan 2006, Hewitt EnnisKnupp 2012, Whaley 2002) showing annualized return ≈8.4–8.6% at ~two-thirds S&P 500 volatility, Sharpe ≈0.53–0.62, but the *absolute Sharpe gap* over SPY (≈0.06–0.4) sits inside or below this system's 0.4–0.9 detection floor. Dispersion trading's correlation-risk-premium explanation is contested by its own literature ("returns depend mainly on mispricing and market inefficiency," which does not survive costs reliably). CBOE's own post-mortem on Volmageddon (Feb 2018) states listed BXM/PUT strategies "performed largely as expected" while only *leveraged* VIX ETPs (XIV, SVXY) were destroyed — meaning listed options premium-selling survived the exact tail event that killed the already-failed SVXY test, but the 2008 crisis remains a serious absolute drawdown (BXM −29% to −32%). Best provider: Interactive Brokers, already integrated, options-capable, Israeli residents accepted on IBKR Pro; ESMA's own aggregated National Competent Authority disclosures show 74–89% of retail CFD accounts lose money. Of named providers, **only IG and Saxo Bank clear the automation bar** (REST/FIX API, ToS-permitted algo trading, same-API demo); Plus500 and eToro offer no usable automation channel for a self-built system. **Confidence: High (for the negative conclusion).** Any strategy expressed as "trade X via CFD" should be re-expressed as trading the listed future/ETF/option on X via IBKR instead — this instrument type is a **don't-bother** regardless of underlying strategy.

#### 4. Futures (equity-index, Treasury, commodity, FX, VIX)

Covered in depth in the Stage B deep dive above (trend-following selected as top candidate). Commodity carry/roll-yield specifically has decayed: an NBER follow-up re-testing 2005–2014 found the roll-yield premium fell to 3.67%/year with a t-stat of only 0.76 — no longer significant. VIX futures term-structure/basis trading (Simon & Campasano, *Journal of Derivatives* 2014) is the *identical underlying risk premium* already tested and rejected via the SVXY term-structure test — implementing it via VIX futures instead of SVXY does not create a new hypothesis. **Confidence: High** for trend-following, **Low** for commodity carry and VIX-futures-basis specifically.

#### 5. Bonds and cash

Koijen, Moskowitz, Pedersen & Vrugt's "Carry" (*JFE* 2018) and a 2021 *Financial Analysts Journal* 70-year (1950–2020) international replication show Treasury/bond carry with **no observed out-of-sample decay** across a genuine 30-year holdout — unusually strong for this literature, corroborated by a Reserve Bank of Australia working paper. No rigorous, disclosed-method study isolating a pure VIX-futures *trend* signal (distinct from term-structure timing) was found — this would need to be developed from scratch, not sourced from existing literature. **Confidence: High (for the negative conclusion).**

#### 9. Israeli market (TASE)

TASE itself states it offers no execution API — only a paid Data Hub subscription for reference/historical data, no live trading connectivity. Interactive Brokers is the only verified broker connecting retail clients to TASE, via TWS/IB Gateway, but **whether TASE-listed instruments are even tradable in IBKR's paper account is unverified** — a real risk that no qualifying provider exists once the "same-API paper account" requirement is applied strictly. Critically, and contrary to a common assumption, **Israeli residents get the standard 25% domestic capital-gains rate on TASE securities, not a tax-favored rate** — the exemption applies only to non-residents. Maof (TA-35) options are a thin, structurally illiquid market. Detectability is poor: a 35–125-name universe gives fewer, not more, independent bets than the US book. **Confidence: Low.** Worth revisiting only for ILS-hedging/diversification purposes, not alpha, and only after directly confirming IBKR's TASE-in-paper-account capability with IBKR support.

#### 10. ETF and fund structure effects

Closed-end fund discounts are large (~8.4% historical average) and highly persistent (Pontiff, *JFE*/*QJE*, 1995/1996; still confirmed in a 2023 replication), but Pontiff's own finding is that the discount survives *because* arbitrage is costly — no Tier-1/2 source found shows a net-of-cost, tradeable-today edge for a small account. Leveraged-ETF decay is real but the harvesting evidence is contradictory and market-specific: one 2025 SSRN paper finds Sharpe 2.12 shorting bull LETFs in the US while shorting bear LETFs (the theoretically favored side) is unprofitable, and other 2025/2026 work shows LETFs actually *outperform* in trending markets and only decay in choppy ones — the opposite of the naive "always short" heuristic. Index-reconstitution arbitrage has **decayed from a 5–9% announcement pop pre-2000 to roughly 1% today**, and one 25-year event study claims the effect has flipped sign entirely — a textbook case of an anomaly arbitraged away. ETF creation/redemption premiums for liquid products run 0.01–0.03%, below what a retail-size trade can clear net of costs. **Confidence: Medium** on mechanism existence, **Low** on net-tradeable edge for a small account.

#### 11. Event-driven equities (merger arb, spin-offs, PEAD, insider clusters)

Covered in depth above (insider clusters selected; PEAD dropped). Merger arbitrage (Mitchell & Pulvino, *Journal of Finance*, 2001) shows a genuine ~3.5–4%/year net-of-cost excess return with a documented crowding effect (arbitrage AUM grew from $233M to $28B, 1996–2007, shrinking the spread) — but with only 200–400 liquid US deals/year, this is a data-labeling problem (deal-contingency monitoring from 8-K/proxy text) as much as a signal problem, and was not selected for the top-5 given the lower bet count relative to insider clusters. Spin-offs (Cusatis, Miles & Woolridge, *JFE*, 1993) **failed an explicit out-of-sample replication** (McConnell, Ozbilgin & Wahal, 2001) — the original result did not survive ex-ante testing, and a Swedish thesis found post-spin-off entities underperform the S&P 500 on a risk-adjusted basis. **Confidence: Medium-High** (insider clusters, PEAD dropped), **Medium** (merger arb, not selected), **Low** (spin-offs, do not pursue).

#### 12. Passive and structural allocations

Covered in depth above (selected as the risk-parity/diversification candidate). Quality-Minus-Junk (Asness, Frazzini & Pedersen, *Review of Accounting Studies*, 2019) is genuinely low/negatively correlated to market and size factors and shows positive convexity during crises across 24 countries, but overlaps substantially with the already-failed `multi_factor` strategy in the current stock stack — its incremental diversification value is likely small and it should be treated as an allocation tilt, not a new signal. International diversification evidence is genuinely mixed across Tier 1/2 sources (a 118-year, 21-market study finds a 28% average Sharpe improvement, concentrated in low-domestic-Sharpe countries — a poor fit for a US-based book that already has high domestic Sharpe; rising US/international correlations over recent decades mechanically shrink the benefit) — marked UNVERIFIED as a strong recommendation, directionally positive but weaker and more time-varying than the bond/commodity sleeve. **Confidence: Medium-High** (risk parity mechanism, quality's correlation profile), **Low-Medium** (international diversification).

#### Stage A Summary Table

| Item | Evidence strength | Net edge post-2010 | Annual cost | Data source/cost | Best Israel-eligible provider | Tax/reg fit | Detectability | Integration effort | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| 1. Listed options | Tier 1 (VRP, skew) but same risk as failed SVXY/PUT | Inside/below detection floor | Spreads 1.7-20%+ of premium | OptionMetrics (WRDS, institutional) | IBKR (existing) | UNVERIFIED | ~12 bets/yr single-index | Low (existing broker) | Low-Medium |
| 2. Spot forex | Tier 1 (momentum); carry verified decayed | Momentum borderline; carry ~0.04-0.16 Sharpe | ~0.2bp+$2.50/leg (IdealPro) | IBKR API (free) + FRED | IBKR (existing) | Asymmetric loss-offset trap (Sec 9(13)/29) | Momentum borderline; carry not detectable | 20-28 p-days | Medium (mom.), Low (carry) |
| 3. CFDs | None found for wrapper itself | N/A — don't bother | ESMA caps don't extend to Israel | N/A | IG (FCA) / Saxo (Danish FSA) | Not banned but unprotected | N/A | N/A | High (negative) |
| 4. Futures | Tier 1, century-long, some decay | ~0.15-0.5 Sharpe (from ~1.1-1.8) | $0.10-0.85/contract + fees | Norgate ($270/yr) | IBKR (existing) | No 1256-style MTM; needs adviser | 26-61 effective bets/yr | 36-54 p-days | Medium-High |
| 5. Bonds/cash | Tier 1, no decay found (70yr) | Undecayed but low power alone | 0.04-0.15% ETF expense | Tiingo (free) | IBKR (existing) | Standard 25% CGT | ~12 bets/yr, underpowered alone | Folded into #12 | Medium |
| 6. Commodities | Tier 1, premium ≈0 individually | ≈0% | 0.40-0.87% ETF expense | Tiingo (free) | IBKR (existing) | Standard 25% CGT | N/A (no real edge) | Folded into #12 | Low |
| 7. Crypto beyond BTC | Tier 2 (BIS), real but liquidation-fragile | Shrinking (ETF-launch effect) | 0.02-0.80% + funding | None free; CME/Kraken paid | Kraken (BVI/Bermuda entities) | Funding-vs-capital-gain UNVERIFIED | ~12-26 bets/yr | 53-83 p-days | Medium |
| 8. Vol products (VIX futures) | Same mechanism as failed SVXY test | Same as failed test | 0.85-0.95% ETP expense | CBOE (free, index only) | IBKR (existing) | UNVERIFIED | ~12 bets/yr | Low | High (negative) |
| 9. TASE | Low; no edge evidence found | N/A | Bank markup 0.1-0.8% | TASE Data Hub (paid, UNVERIFIED price) | IBKR (TASE-in-paper unverified) | Standard 25%, NOT tax-favored for residents | 20-50 bets/yr, underpowered | High (API access unresolved) | Low |
| 10. ETF structure effects | Tier 1 mechanism; no net-cost edge shown | Decayed/reversed (reconstitution) | Varies | Tiingo adequate for ETFs | IBKR (existing) | Standard 25% CGT | 10-40 bets/yr | Low-Medium | Medium (mechanism), Low (tradeable) |
| 11. Event-driven equities | Tier 1 (insider, PEAD); PEAD ≈0 post-2017 | Insider ~1/3 of 2012 estimate; PEAD ≈0 | Small-cap spreads 0.9-2.5% | EDGAR (free) + paid consensus data (PEAD blocker) | IBKR/Alpaca (existing) | Standard, minor turnover-reclass risk | 300-1500 bets/yr (insider) | 24-36 p-days | Medium-High |
| 12. Passive allocations | Tier 1/2, structural not arbitraged | 2022 break for bonds; commodities held up | 0.10-0.32%/yr | Tiingo (free) | IBKR (existing) | Standard 25% CGT | N/A (diversification test) | 2-4 p-days | Medium-High |

---

### Part 3 — Provider Table for the Shortlist

| Candidate | Recommended provider | Regulator/entity serving Israel | API type | Automation terms | Paper account | Monthly cost | Runner-up |
|---|---|---|---|---|---|---|---|
| Futures trend-following | Interactive Brokers | SEC/FINRA (IBKR LLC) or FCA/Central Bank of Ireland for non-US entities; Israel on IBKR's accepted-country list | TWS/IB Gateway socket API | Explicit "Retail, Algorithmic and Proprietary Traders" use case; non-commercial license permits automated order entry | Yes, same TWS API (port 7497) | ~$0 (existing) + $15-45 live futures data + $270/yr Norgate | None needed (only realistic route) |
| Diversification sleeve | Interactive Brokers | Same as above | Existing equity/ETF order path | Already covered by existing usage | Yes | $0 | None needed |
| Insider-cluster equities | IBKR / Alpaca (existing) | Same as above / Alpaca | Existing equity order path + SEC EDGAR REST | EDGAR: free, 10 req/s fair-access, User-Agent required | Yes (same brokers) | $0 (EDGAR) | Quiver Quantitative (~$75/mo, convenience only) |
| FX momentum | Interactive Brokers (IdealPro) | Same as above | TWS `Forex` contract type | Same TWS terms | Yes, same IdealPro venue/API | $0 | None credibly better for automation |
| Crypto carry beyond BTC | Kraken (spot+derivatives) + IBKR (CME leg) | Payward Trading Ltd (BVI) for spot/margin; Payward Digital Solutions Ltd (Bermuda, Bermuda Monetary Authority Class F) for derivatives; IBKR as above for CME | Kraken REST/WebSocket (spot); separate Futures REST/WebSocket (derivatives); IBKR TWS for CME | Kraken markets explicitly for "systematic and algorithmic traders"; IBKR as above | Kraken: derivatives demo yes (`demo-futures.kraken.com`); **spot has no public demo, request-only UAT** | Kraken: $0 subscription, transactional fees only; IBKR as above | No credibly better alternative found for the perpetuals leg from Israel |

---

### Part 4 — "Don't Bother" List

* **CFDs (any underlying).** No evidence the wrapper adds return; ESMA's leverage caps and negative-balance protections do not extend to Israeli clients; 74–89% of EU retail CFD accounts lose money per ESMA's own aggregated disclosures; most retail providers (Plus500, eToro) don't even offer a usable automation API. Trade the listed underlying via IBKR instead.

* **VIX-futures term-structure/basis trading.** Identical underlying risk premium to the already-tested SVXY strategy (Sharpe gap ≈−0.4); implementing via futures instead of an ETP does not create a new hypothesis, and CBOE's own data confirms the same >80%-of-the-time contango driving the decay.

* **Index put-writing variants and covered-call/buy-write programs.** The Sharpe gap versus a risk-matched benchmark (≈0.06–0.4 per multiple independent replications of BXM) sits inside or below this system's 0.4–0.9 detection floor, and it is economically the same short-volatility exposure as the already-tested CBOE PUT strategy.

* **Standalone PEAD.** Tier-1 evidence (Kettell, McInnis & Zhao) shows the signal is statistically indistinguishable from zero after 2017, before any cost adjustment; the required consensus-estimate data (I/B/E/S/WRDS) has no free retail-accessible point-in-time equivalent, an unbounded cost for a signal that's already dead.

* **Individual-commodity risk premia (roll yield, storage-cost timing).** Multiple independent studies by the same research group that originally found a positive commodity premium now find it averages approximately zero at the individual-commodity level, both before and after financialization; the popular "roll yield" story sold by PDBC/DBC's marketing is explicitly debunked as having no mechanical relationship to returns.

* **TASE alpha strategies (stock-picking, Maof options).** No credible edge evidence found; a 35–125-name universe is a *lower*-power environment than the US book, not higher; execution-API access for TASE is unresolved even via IBKR; and — contrary to a common assumption — Israeli residents get the standard, non-favored 25% domestic capital-gains rate on TASE securities.

* **ETF creation/redemption arbitrage and S&P 500 index-reconstitution trading.** Both have decayed to the point of near-irrelevance for a retail-size account: creation/redemption premiums (0.01–0.03% for liquid products) are smaller than round-trip costs, and the classic reconstitution "pop" has compressed from 5–9% pre-2000 to roughly 1% today, with some post-2000 event studies claiming the effect has reversed sign entirely.

* **Merger arbitrage and spin-offs.** Both carry genuine, Tier-1-documented mechanisms, but merger arb's spread has visibly shrunk as arbitrage AUM grew more than 100x (1996–2007), and the foundational spin-off result (Cusatis, Miles & Woolridge) explicitly **failed an out-of-sample replication** by McConnell, Ozbilgin & Wahal — not selected for the shortlist given lower bet counts and, for spin-offs, a failed replication.

---

### Part 5 — Open Questions for the Owner

* **Margin appetite.** The futures trend-following candidate ties up roughly $25,000–$45,000 of a $250,000 account in SPAN margin at any time; is running any margin-based, mark-to-market futures book (versus the current fully cash-equity book) something you're comfortable operating unattended?

* **Paid-data budget.** Norgate Data (~$270/year) for futures is cheap and clearly worth it if futures trend-following is pursued; a paid insider-transaction convenience vendor (~$75/month) is optional; the PEAD-blocking consensus-estimate data (I/B/E/S/WRDS-class, likely $1,000s–$10,000s/year) is very unlikely to be worth it given the underlying signal is already dead — confirm you agree PEAD should simply be dropped rather than funded.

* **Tax adviser engagement.** At least four genuinely ambiguous Israeli tax questions surfaced that this brief could not resolve from primary ITA sources: futures P&L characterization (no US-style Section 1256 mark-to-market exists in Israel), the FX currency-linkage loss-offset asymmetry (Section 9(13)/29), whether perpetual-futures funding payments are capital gains or ordinary income, and whether higher-turnover strategies risk ITA "trader status" reclassification from capital-gains to labor-income rates. Do you want a single consultation covering all four before building anything, or resolve them one at a time as each candidate is greenlit?

* **Cadence tolerance.** Two of the five candidates (crypto carry, and to a lesser extent the CME-basis leg of it) have a documented mismatch between this system's once-per-day decision cadence and the strategy's actual risk clock (8-hour funding settlements, continuous liquidation risk). Is a cadence change (e.g., intraday risk monitoring layered on top of daily-bar decisioning) something you'd consider, or should such candidates simply be excluded on architectural grounds?

* **Repo-read follow-ups.** Two implementation details were flagged as needing a direct repo read rather than being assumed from documentation: (a) exactly what `allocation_method: risk_parity` in `config/live.yaml` currently does (in `src/firm/agents/trader.py`), which determines whether the diversification sleeve needs any new code at all; and (b) whether IBKR's paper account actually supports TASE-listed instrument fills, which determines whether TASE is worth any further investigation at all.

---

## Conclusion

The pattern across this entire survey is consistent with what the system's own prior testing already found: **most well-known "alternative" return sources are either the same risk premium already tested and rejected wearing a different instrument wrapper (options premium-selling, VIX futures, individual commodities), or a genuinely distinct premium that has demonstrably decayed since 2010 (FX carry, commodity roll yield, PEAD, S&P reconstitution)**. Only a small number of ideas combine a still-credible, undecayed mechanism with enough independent bets per year to be statistically detectable at this system's scale: multi-asset futures trend-following is the strongest of these, a passive bond/commodity diversification sleeve is the cheapest and most defensible near-term addition (explicitly as risk management, not alpha), and insider-purchase clustering is worth a disciplined test once PEAD is dropped from scope. FX momentum is a reasonable secondary test; FX carry and crypto carry beyond spot BTC should not receive capital under current evidence and, for crypto carry specifically, the engineering cost of even trying it plausibly exceeds its risk-adjusted edge. Given that the system's own evaluation standard has so far rejected everything it has tried, the recommended sequence is to spend the cheapest integration (the diversification sleeve, 2–4 person-days) first, validate the evaluation pipeline on it, and only then commit the 20–54 person-day budgets that futures trend-following, insider clustering, or FX momentum would require — while treating crypto carry as a "watch, don't build" item pending either a lower-friction Israeli-accessible venue or a materially larger documented edge.
