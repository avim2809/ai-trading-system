# Independent recompute: SEC insider-purchase-cluster evaluation

Frozen design: `scripts/insider_cluster_preregistered_bars.py` (fingerprint
`f355b0756683cf6fb1a3ac5b2db47cc891a0d9e0a1a2076dd575142278693a74`, matches
the primary run -- both read the same immutable module).

Independent implementation: `recompute.py` (event coverage, event-level
returns, bootstrap, placebo, calendar-time, DSR/PBO, tier classification) and
`diagnose2.py` / `posthoc_full.py` (data-quality diagnosis + post-hoc
exclusion pass), written from `FREEZE_NOTES`/`TIER_A_BARS`/`COSTS`/`WINDOWS`/
`BOOTSTRAP`/`PLACEBO`/`DSR`/`PBO`/`classify()` without importing or copying
`scripts/run_insider_cluster_evaluation.py`.

## 1. Side-by-side: primary vs. independent recompute

Coverage (identical):

| status | primary | recompute |
|---|---|---|
| covered | 28,477 | 28,477 |
| no_price_data | 1,727 | 1,727 |
| series_not_spanning_event | 1,632 | 1,632 |
| unmappable_ticker | 1,422 | 1,422 |

3-month hold:

| stat | primary | recompute | diff |
|---|---|---|---|
| n_evaluated_covered | 8,866 | 8,866 | 0 |
| n_primary_strict | 8,373 | 8,373 | 0 |
| early_exit_rate | 0.007405 | 0.007405 | 0 |
| mean xs_net | 0.008711 | 0.008712 | +0.000001 |
| median xs_net | -0.015555 | -0.015555 | 0 |
| hit_rate | 0.462080 | 0.462080 | 0 |
| bootstrap LB | -0.006386 | -0.006384 | +0.000002 |
| bootstrap UB | 0.026563 | 0.026565 | +0.000002 |
| post-2013 mean | -0.002016 | -0.002016 | 0 |
| gross_raw_mean | 0.050414 | 0.050414 | 0 |
| placebo p95 | 0.066629 | 0.065568 | -0.001061 |
| placebo median | -0.014232 | -0.014192 | +0.00004 |

6-month hold:

| stat | primary | recompute | diff |
|---|---|---|---|
| n_evaluated_covered | 7,662 | 7,662 | 0 |
| n_primary_strict | 7,222 | 7,222 | 0 |
| mean xs_net | 0.011975 | 0.011977 | +0.000002 |
| bootstrap LB | -0.014638 | -0.014636 | +0.000002 |
| post-2013 mean | -0.011032 | -0.011032 | 0 |
| placebo p95 | 0.954511 | 0.954468 | -0.00004 |
| placebo median | -0.024645 | -0.024624 | +0.00002 |

Calendar-time / DSR / PBO (both holds essentially identical):

| stat | primary 3m | recompute 3m | primary 6m | recompute 6m |
|---|---|---|---|---|
| ann_mean_excess | 33.8940 | 33.8940 | 18.0508 | 18.0508 |
| ann_vol | 139.188 | 139.188 | 73.533 | 73.533 |
| sharpe | 0.24351 | 0.24351 | 0.24548 | 0.24548 |
| dsr | 0.98630 | 0.98630 | 0.98757 | 0.98757 |
| days_with_positions | 4,713 | 4,713 | 4,715 | 4,715 |

PBO: primary 0.071429, recompute **0.071429** (identical).

Bars: primary `{A1 F, A2 T, A3 F, A4 F, A5 T, A6 T, A7 T}`, tier_D F, bars_B
`{B_a T, B_b F}`, **tier C** -- recompute reproduces **every bar and the
tier exactly**.

### Mismatches found, and their exact cause

1. **Bootstrap/placebo bounds differ in the 4th-6th decimal** (LB/UB, placebo
   p95/median). Cause: both implementations do a genuine Monte Carlo
   (month-cluster bootstrap, 5,000 draws; placebo re-draw, 500 draws) with
   *different* RNG streams -- the recompute follows the frozen seed formula
   (`SEED`, `SEED+hold`, `SEED+1+hold`) but the primary harness's internal
   draw order differs from a from-scratch reimplementation's, so the exact
   draws differ even with nominally the same seed. This is expected
   Monte-Carlo noise, not a disagreement about the design; every point
   estimate (mean, median, hit_rate, n) matches exactly.
2. **`by_adv_bucket` n splits 2 events differently at the $1M ADV boundary**
   (3-month: recompute's `adv_1m_5m`=3,369 vs primary's 3,367; `adv_lt_1m`
   1,535 vs 1,537; 6-month: `adv_1m_5m` 2,852 vs 2,850; `adv_lt_1m` 1,355 vs
   1,357 -- always +2/-2, never a net gain or loss of events). Cause: to fit
   the ~8,300-ticker price universe in this host's available RAM (a first
   attempt at float64/6-column caching was OOM-killed at ~3GB RSS with the
   live trading services also running), the recompute stores `adjusted_close`
   /`adj_open`/`dollar_vol` as **float32**. Two events have a 20-day median
   dollar volume landing within noise of exactly $1,000,000, and the
   float32-vs-float64 rounding at the 15th-16th bit flips which side of the
   bucket boundary they fall on. It does not change which events are
   *evaluated* (both sides sum to the same total n), only which ADV-bucket
   cost/benchmark they're reported under, and it has no effect on any bar or
   the tier.
3. No other numeric, coverage, or bar mismatch was found. Two example
   matched events (ticker, entry_date, hold) reproduced identically byte-for-
   byte in the point estimate: `ACRX, 2012-12-12, 3_month` (both runs:
   gross 25.11, xs_net +24.9566) and `SMLP` calendar-time contribution on
   `2015-12-28` (both runs' calendar-time series show this as the single
   largest day, 3_month: 601.94 primary-equivalent / 601.94 recompute).

## 2. Data-quality diagnosis

The primary's calendar-time series (`3_month|strict|bench_primary`) has
`ann_mean_excess` ~3,389%, `ann_vol` ~13,919% -- reproduced exactly in the
recompute. Tracing this to individual calendar days and tickers:

**Root cause, 3-month hold: a single calendar day, 2015-12-28, contributes
601.94 out of the whole series' ~4,713 non-zero days** (mean 0.1345/day is
~95% explained by this one day: 601.94 / 4,713 = 0.1277). The 6-month
series shows the identical date as its dominant outlier (317.998).

That day's positions were traced (entry_date <= 2015-12-28 <= exit_date,
3-month strict, 145 tickers open that day) and each ticker's
`adjusted_close.pct_change()` on that exact date was computed. One ticker
explains it completely:

- **SMLP** (Summit Midstream Partners): `2015-12-25` (a market holiday --
  Christmas Day, when the market is closed) has a phantom row: open=high=low
  =close=`0.0002`, volume=0, sandwiched between real rows of
  `~$18.35` (2015-12-24) and `~$17.78` (2015-12-28). This single fabricated
  near-zero tick produces a `+84,272x` (8,427,250%) same-day return on
  2015-12-28 when the series reverts to its real price level.
  **BAD DATA** (unambiguous -- a holiday with zero volume and a garbage
  price should never have a bar at all).

**Root cause, second-largest anomaly (2015-03-27, 2015-03-16, both holds):**

- **QPAC**: a near-zero-volume ticker whose entire series alternates between
  `$0.01` and `$9.40-9.85` on successive trading days with **volume = 0 on
  all but one of the ~15 days inspected** (the one exception, 2015-03-27,
  has volume 498,900 -- everything else is untraded). This is an
  untradeable/garbage quote series, not a real stock with real price
  discovery. **BAD DATA.**

**Root cause, third cluster (~20 days spread May-Oct 2010, magnitude 0.3-0.9
per day):**

- **XBKS** (Xenith Bankshares, penny community-bank stock): a recurring,
  systemic ~250x scale-factor error. On dozens of specific days (99 of 3,272
  trading days 2005-2017, concentrated 2009-2015: 51 in 2010 alone, 24 in
  2011, 17 in 2012) the reported close is exactly ~250x the surrounding
  days' level (e.g. 2010-04-28: `$725.00` vs `$8.81` the day before and
  `$8.40` the day after; 2010-02-01: `$507.50` vs a ~$8 range) before
  reverting the next session. Volume on the spike days is unremarkable
  (not a liquidity-driven print). **BAD DATA** -- looks like a vendor feed
  occasionally reporting a quote from a different venue/scale for this
  thinly-traded name.

**Root cause, event-level extreme excess returns (top of
`events_evaluated.parquet` by `abs(xs_net)`):** a per-event scan (window =
entry_date-3d to exit_date+3d) comparing same-day raw-close vs.
adjusted-close returns found two more tickers where a real-looking raw price
jump is **not offset by the split ratio** (ratio stays constant at a
non-1.0 value straddling the jump, instead of stepping to absorb it --
diagnostic of an un-adjusted corporate action, distinct from a *correctly*
handled split where the ratio changes in sync with the raw jump and
`adjusted_close` stays continuous):

- **ACRX** (AcelRx Pharmaceuticals): ratio is a constant 20.0x for the
  *entire* 2011-02-11 to 2022-10-25 span, including straight through an
  obvious raw-close jump on 2012-12-17 (`$4.08` -> `$78.20`, ~19.2x, itself
  consistent with a real reverse split) that should have reset the ratio to
  ~1 at that date but didn't. Contaminates 2 real events: 3-month entry
  2012-12-12 (gross +25.11, xs_net +24.96) and 6-month same entry (gross
  +53.32, xs_net +52.32). Its *other* 2 real events (2017, 2018 entries,
  after the ratio correctly settles) are unaffected and show normal-sized
  returns (-0.40, +0.30). **BAD DATA**, localized to the one event pair.
- **CERN** (Cerner Corp -- a mega-cap; a genuine same-day +293% move is
  implausible on its face): ratio constant at 0.242 straddling a raw jump
  on 2008-12-18 (`$9.71`->`$38.20`). Contaminates its single real 6-month
  event (xs_net +4.93). **BAD DATA.**

**Checked and cleared (plausibly real, not data quality issues) -- verified
by confirming `adjusted_close/close` is either a single constant value
throughout the whole event window (no split boundary crosses it at all:
OSTK, TNDM, CWEI, CLW, AVEO, CNST, BKD, ALPN, TRXC, ESPR, FTSV, VAPO, FPRX,
APPS, CDTX, PEIX, AGO), or the ratio changes exactly in step with a matching
raw-close jump so `adjusted_close` stays continuous across it (AGL: raw
close jumps ~24.6x on 2026-03-31 while adjusted_close moves only -1.6% that
day; PTN: raw jumps ~42.6x on 2025-08-12 while adjusted_close moves only
~-15%) -- both correctly-handled splits, not artifacts. FPRX's +237% single-
day jump (2020-11-11, on 79.7M shares vs. a typical <1M) matches the real,
publicly known Amgen acquisition of Five Prime Therapeutics announced that
day (deal price ~$38, and the stock does trade dead-flat near $37-38 for the
next several months awaiting close -- the signature of a real merger-arb
spread, not a data glitch, which my first-pass heuristic mis-flagged before
requiring the ratio to be non-1.0 elsewhere in the ticker's history).

### Offender summary (top of both lenses: event-level cumulative AND
calendar-time daily -- the two catch different failure modes, see below)

| ticker | issue | evidence | verdict |
|---|---|---|---|
| SMLP | phantom zero-price holiday row (2015-12-25) | close=0.0002, vol=0, between $18.35 and $17.78 real bars | BAD DATA |
| QPAC | untradeable garbage-quote shell | volume=0 on ~14/15 days inspected, price randomly $0.01 or ~$9.50 | BAD DATA |
| XBKS | recurring ~250x scale-factor error | 99/3,272 days (2005-2017) with |daily return| >200%, concentrated 2009-2015 | BAD DATA |
| ACRX | un-adjusted 1:20 reverse split (2012-12-17) | ratio constant 20.0 straddling a matching raw jump | BAD DATA (2 of 6 events) |
| CERN | un-adjusted split-like jump (2008-12-18) | ratio constant 0.242 straddling a matching raw jump; implausible for a mega-cap | BAD DATA (1 event) |
| OSTK, TNDM, CWEI, CLW | large real rally | ratio=1 (or a single constant) throughout window, no discontinuity | plausibly real |
| AGL, PTN | properly-adjusted reverse split | ratio steps in sync with raw jump, adjusted_close continuous | plausibly real |
| FPRX | +237% same-day (M&A) | huge volume spike + subsequent flat merger-arb spread near deal price | plausibly real (Amgen/Five Prime) |
| AVEO, AGO, CNST, BKD, ALPN, TRXC, ESPR, FTSV, VAPO, APPS, CDTX, PEIX | large sustained biotech/small-cap move | no single-day discontinuity or ratio artifact found | plausibly real |

**Key structural finding**: event-level cumulative return (`xs_net` over the
whole 63/126-day hold) and the daily calendar-time series are corrupted by
*different* subsets of bad data. SMLP/QPAC/XBKS's damage is nearly invisible
in the event-level table (their real strict events have unremarkable
xs_net: -0.02 to -0.98) because a one-day bad tick mostly cancels out over a
63+ day hold -- but it is catastrophic in the calendar-time *daily* series,
which uses each day's return directly with no such cancellation. ACRX/CERN's
damage (a genuinely un-corrected, *permanent* scale shift, not a one-day
tick) shows up in both.

## POST-HOC DIAGNOSTIC (excludes ACRX, CERN, SMLP, QPAC, XBKS -- 10 of 8,373
3-month / 9 of 7,222 6-month strict events; NOT a change to the frozen
verdict, which stands as computed on the full, uncorrected data)

| stat | 3-month, full | 3-month, excl. bad tickers | 6-month, full | 6-month, excl. bad tickers |
|---|---|---|---|---|
| mean xs_net | 0.008711 | **0.005575** | 0.011975 | **0.004245** |
| median xs_net | -0.015555 | -0.015504 | -0.032618 | -0.032555 |
| bootstrap LB | -0.006386 | -0.007534 | -0.014638 | -0.017761 |
| bootstrap UB | 0.026563 | 0.022372 | 0.046815 | 0.029723 |
| **1%/99%-winsorized mean** | n/a | **0.001062** | n/a | **-0.004559** |
| max xs_net (single event) | 52.32 (ACRX) | 5.59 (AGL) | 52.32 (ACRX) | 11.82 (TNDM) |
| placebo p95 | 0.066629 | 0.064510 | 0.954511 | **0.748338** |
| placebo frac(draws>+50%) | n/a | 0.042 | ~0.10 (primary) | **0.092** |
| calendar-time ann_mean_excess | 33.894 (3,389%) | **0.0403 (4.0%)** | 18.051 (1,805%) | **0.0275 (2.7%)** |
| calendar-time ann_vol | 139.19 (13,919%) | **0.1019 (10.2%)** | 73.53 (7,353%) | **0.0911 (9.1%)** |
| calendar-time Sharpe | 0.2435 | **0.3953** | 0.2455 | **0.3015** |
| calendar-time DSR | 0.9863 (pass) | **0.9183 (fail)** | 0.9876 (pass) | **0.8347 (fail)** |
| PBO (8-variant CSCV) | 0.0714 (pass) | **0.80 (fail)** | (same 8-variant grid, both holds) | |

Even after removing the 5 confirmed-bad tickers, the **6-month placebo p95
is still 74.8%** and 9.2% of draws still land above +50% excess -- both
still absurd for a real return distribution. This means the universe likely
still contains *other*, unidentified bad-data tickers beyond the 5 traced
here; a full cleanup would require scanning the whole ~8,300-ticker universe
for the same ratio-discontinuity / zero-volume-spike / recurring-scale-error
signatures, which this pass did not have budget to do exhaustively (it
started from the primary's own top-25-by-`|xs_net|` event list plus the
calendar-time series' own top single-day outliers, which is a strong but not
exhaustive net).

## 3. Verdict

**The Tier C classification under the frozen rules IS reproduced** -- every
one of the 7 A-bars, the D-condition, both B-bars, and PBO match the primary
run exactly (recompute PBO 0.071429 == primary 0.071429; tier C == tier C).
Since Tier C only requires *not* being Tier A/B/D, this conclusion is robust
regardless of the data-quality issue: the post-hoc numbers, if anything, make
the case for deploying this candidate *weaker*, not stronger.

**Is any bar outcome an artifact of bad data? Yes -- A2 clearly, and it
drags A6 down with it:**

- **A2 (DSR > 0.95, the pass)**: **artifact of bad data.** Excluding just 5
  tickers (10-19 events out of ~8,300-15,600) drops the 3-month DSR from
  0.986 to 0.918 and the 6-month DSR from 0.988 to 0.835 -- both now *below*
  the 0.95 bar, flipping A2 from PASS to FAIL. The calendar-time
  `ann_mean_excess`/`ann_vol` collapse by more than two orders of magnitude
  (3,389% -> 4.0%; 13,919% -> 10.2%) once SMLP's phantom Christmas Day row
  and QPAC/XBKS's recurring scale errors are removed, which is exactly what
  you'd expect if a handful of garbage single-day returns -- not a real,
  tradable daily excess -- were inflating both the numerator and (to a
  lesser extent) the denominator of the Sharpe that DSR is evaluated on.
- **A6 (PBO < 0.50, the pass)**: **also looks like an artifact**, in the
  opposite direction from what might be guessed -- PBO on the cleaned
  8-variant grid is *worse* (0.80, fail) than on the contaminated grid
  (0.071, pass). CSCV ranks trial Sharpes across time partitions; the
  original PBO's apparent "not overfit" reading depended on how the same
  few extreme days happened to move all 8 trial variants (strict/covered x
  2 benchmarks x 2 holds) together in a way that preserved their relative
  ranking in-sample vs. out-of-sample. Removing those days changes which
  configuration ranks best in which partition, and the post-hoc PBO says
  the cleaned signal is *more* overfit-looking, not less. Either way, PBO's
  original pass is not a reliable, data-quality-independent result.
- **A1, A3, A4 (fails in both the primary and the recompute)**: still fail,
  and for the right reasons independent of the bad data -- the post-hoc
  mean nearly halves for 3-month (0.0087->0.0056) and drops to near-zero for
  6-month (0.0120->0.0042), and the 1%/99%-winsorized means are barely
  positive (+0.0011, 3-month) or **negative** (-0.0046, 6-month). The
  bootstrap LB stays negative for both holds even after cleaning. So A1's
  fail was never driven by the bad data (it fails harder after cleaning),
  and A4 (post-2013 positive) was already failing on the uncontaminated
  post-2013 subset in both the primary and the recompute.

**Bottom line**: Tier C is the right call either way, but for a cleaner
reason than the primary report shows -- the frozen run's two "passing" bars
(A2 DSR, A6 PBO) both evaporate/reverse under even a partial data-quality
cleanup, while its "failing" bars are, if anything, confirmed and
strengthened once the handful of bad-data spikes are removed. The genuinely
concerning number for anyone considering this candidate is the **1%/99%-
winsorized 6-month mean excess of -0.46%** -- a return-distribution-outlier-
robust estimate that this candidate has no real edge net of costs over
2008-2026, consistent with the frozen verdict's own post-2013 and cost-2x
splits.
