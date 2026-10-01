# Insider-purchase clusters: verdict (2026-09-30)

**Tier C: no demonstrated edge. Do not trade it.**

The test asked whether buying a stock after a cluster of SEC Form 4 insider purchases beats a size-matched ETF over the next 3 or 6 months, after costs. It does not. The typical event loses money. The positive average comes from a few extreme winners, part of them bad price data. The effect is negative after 2013.

- Frozen design: `scripts/insider_cluster_preregistered_bars.py`. It was frozen 2026-09-30 17:15Z, before any event return was computed; fingerprint `f355b075…`.
- Harness: `scripts/run_insider_cluster_evaluation.py`.
- Results: `insider_cluster_evaluation_2026_09.json` (primary) and `insider_cluster_recompute_2026_09.json` (independent recompute and post-hoc diagnostic).
- Trial ledger: `insider_cluster_trial_history.json` (new family, 8 trials).

## Data

- **Events:** SEC Form 4 insider-purchase clusters, 33,258 events in total.
- **Prices:** EODHD end-of-day data, including delisted stocks.
- **Coverage:** 28,477 events are covered. The rest are:
  - 1,727 with no price data;
  - 1,632 whose price series doesn't span the event;
  - 1,422 whose ticker can't be mapped.
- **Primary set:** covered events whose EODHD name matches the issuer, that pass the liquidity filter (ADV20 $0.5M–$50M, price ≥ $2), and that don't overlap an open position on the same ticker. That leaves 8,373 events at 3 months and 7,222 at 6 months.
- **Window:** the primary post-sample window, 2008 onward.

## Result (frozen rules)

Excess return is after costs, per event, against the ADV-bucket ETF (IWC, IWM or IJH). Bounds come from a month-cluster bootstrap at α = 0.05/8.

| | 3 months | 6 months |
|---|---|---|
| Mean excess | +0.87% | +1.20% |
| Lower / upper bound | −0.64% / +2.66% | −1.46% / +4.68% |
| **Median excess** | **−1.6%** | **−3.3%** |
| Mean since 2013 | −0.20% | −1.10% |
| Mean at 2× cost | +0.31% | +0.64% |
| Mean with −30% delisting stress | +0.65% | +0.75% |

| Bar | Result | Robust to the data problems below? |
|---|---|---|
| A1: mean > 0 and lower bound > 0 | FAIL | yes, still fails after cleaning |
| A2: calendar-time DSR > 0.95 | pass | **no**: an artifact; 0.92 / 0.83 after cleaning |
| A3: mean > placebo p95 | FAIL | yes |
| A4: positive before and after 2013 | FAIL | yes |
| A5: positive at 2× cost | pass | weak: driven by outliers |
| A6: PBO < 0.50 (0.071) | pass | **no**: 0.80 after cleaning |
| A7: positive under delisting stress | pass | weak: driven by outliers |

Tier D (upper bound < 0) doesn't apply, so under the frozen rules this is **Tier C**. An independent recompute reproduced the tier and every bar outcome exactly. It was written from the pre-registration text without using the harness code. Means match to 5 decimals and PBO matches exactly. The only differences are Monte Carlo noise in the bootstrap and placebo bounds, and 2 events per hold that fall on the other side of the $1M liquidity-bucket boundary.

## Data-quality problems in EODHD (important for any future EODHD test)

The recompute traced absurd numbers to bad bars. For example, the calendar-time series showed +3,389%/yr "excess" at 13,919% volatility.

- **SMLP:** a phantom 2015-12-25 bar (a market holiday) with open = close = $0.0002 and volume 0, between real ~$18 bars. That one bar produces a fake +8.4-million-percent day, and it accounts for ~95% of the calendar-time mean.
- **QPAC:** zero volume on nearly every day; quotes flip between $0.01 and ~$9.50.
- **XBKS:** a recurring ~250× scale error on 99 days, 2005–2017.
- **ACRX, CERN:** reverse splits that aren't adjusted (1:20, ~4×). They create +25× / +52× / +4.9× "returns" on real events.
- Moves the recompute checked and found plausibly real include FPRX (the Amgen buyout), OSTK, TNDM and a dozen small-cap and biotech rallies.

Bad data remains beyond these five tickers:
- The 6-month placebo p95 is still +75% after excluding them.
- Some events show excess returns below −200%, which a long stock position can't produce.

**Any future pre-registration on EODHD prices must freeze a cleaning rule before the run.** For example:
- drop zero-volume bars and bars off the exchange calendar;
- drop single-day moves beyond ±X% that reverse within N days;
- check the adjusted_close / close ratio for unabsorbed split jumps.

## Post-hoc diagnostic (not a verdict change)

This excludes the 5 tickers above, which removes 10 events at 3 months and 9 at 6 months.

| | 3 months | 6 months |
|---|---|---|
| Mean excess | +0.56% | +0.42% |
| Mean, winsorized at 1%/99% | +0.11% | −0.46% |
| Median | −1.6% | −3.3% |
| Calendar-time: annual mean excess / vol | 4.0% / 10.2% | 2.7% / 9.1% |
| Calendar-time DSR | 0.918 (fails 0.95) | 0.835 (fails) |

PBO rises to 0.80. Once the tail is tamed there is nothing left: the median is negative and the winsorized mean is about zero or negative.

## Consequences

- Insider clusters join the standalone strategies and the alternative premia as tested and rejected. Nothing changes live.
- The allocation portfolio on Alpaca and the IBKR control carry on as before.
- The EODHD data stays useful for the owner's shortlist in `research_brief_eodhd_findings.md`: sector/industry ETF momentum, a breadth overlay, bond/commodity trend, 52-week-high proximity, and crypto cross-sectional momentum. Each needs its own pre-registration, including the cleaning rule above.

## Erratum: the freeze timestamp label

`PREREGISTERED_AT = "2026-09-30T17:15:00Z"` in the frozen pre-registration is wrong. The host clock runs on Israel time (UTC+3), and a local time was written as UTC.

The authoritative record is git:
- the freeze commit `d6013c5` is 2026-09-30T16:57:30Z;
- the evaluation's log opens at 16:57:44Z, 14 seconds later.

So the order, freeze before run, holds. The label is left as it is, because editing it would change the frozen fingerprint `f355b075`.
