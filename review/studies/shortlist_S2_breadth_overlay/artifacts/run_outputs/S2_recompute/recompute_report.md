# S2 independent recompute (bar A7) -- report

Written from the frozen text of `scripts/eodhd_s2_breadth_overlay_preregistered_bars.py`
(commit `ff6f44a`, branch `research/eodhd-S2`) and `docs/eodhd_shortlist_protocol_2026_10.md`
(amendment 1). The breadth SIGNAL was built independently from
`data/research/eodhd/us_universe_full/*.parquet` (22,732 files, streamed per ticker, float64,
~12 min) **before** reading the primary's `breadth.parquet` or its harness/report. All code is
new, in `s2_recompute.py` / `build_breadth.py` / `s2_excess_check.py`, using only the allowed
reuse: `eodhd_clean.clean_bars`/`equity_calendar`, `stationary_indices` (from
`run_alt_premia_evaluation.py`), and `firm.eval.overfitting` (`deflated_sharpe`, `cscv_pbo`).

## Breadth signal: reproduces

| Check | Frozen prereg (OBSERVED) | My build | Primary's `breadth.parquet` |
|---|---|---|---|
| files scanned / usable | 22,732 / 22,624 | 22,732 / 22,624 (exact) | 22,732 / 22,624 (exact) |
| first nonzero eligible date | 1993-11-11 | 1993-11-11 (exact) | 1993-11-11 |
| eligible_count min/max (full hist.) | 352 / 3,726 | 352 / 3,726 (exact) | matches |
| eligible_count median (full hist.) | 2,880 | 2,880.0 (exact) | matches |
| eligible_count at window start (2002-01-02) | 2,155 | 2,154 | 2,155 |
| eligible_count at data end | 3,182 | 3,182 (exact) | 3,182 |
| n rebalance dates in window | 297 | 297 (exact) | 297 |

Row-by-row vs. the primary's `breadth.parquet`: `eligible_count` differs on 2,349/8,474 days,
always by 1-3 (primary usually 1-3 *higher*), never more -- a small, one-directional, boundary-
level discrepancy (most plausibly a `>=`/`>` or rounding difference in the ADV20/price screen on
names that sit exactly at a threshold). Effect on the derived series is negligible:
`pct_above_200sma` differs by a median 0.10pp (max 0.57pp); `net_ad_21d` differs by a median
6e-7 (max 4.8e-4). **V1/V2's flip count (38 flips / 19 "on" episodes) and V3's (30/15) match the
frozen prereg's OBSERVED numbers exactly**; V4's (146/73) also matches exactly. `pct_on` differs
by <1pp for every variant. This discrepancy has no material effect on anything downstream.

## Benchmarks and backtest engine sanity

Own two-asset (SPY + destination leg) monthly-rebalance engine: signal through prior month's
last close, trade at the rebalance day's adjusted open (overnight leg at old weight, cost on
turnover, intraday leg at new weight), ADV20-conditioned per-side costs (3bps/10bps, computed
day-by-day, not assumed flat -- **my own ADV20 check found IEF and BIL both spend large stretches
of their early post-launch history below the $50M/3bps threshold: IEF 2002-08 to 2014-05 (2,183
days), BIL 2007-06 to 2018-10 (2,703 days) -- contradicting the frozen prereg's COSTS note that
"3bps/side applies essentially always"; that note's own "trailing-5y minimum" language only
checked the last 5 years as of 2026, not each ETF's low-liquidity launch years**). This raises
both ETFs' realised trading costs above what a flat-3bps assumption would give (a small effect
given the sub-1bps average cost gap and ~1.6 flips/year turnover) and I flag it as a genuine,
independently-found data point for the reviewer.

| | Sharpe | CAGR | maxDD | Calmar |
|---|---|---|---|---|
| BM1 (SPY) | 0.595 | 9.9% | 55.2% | 0.18 |
| BM2 (60/40, primary) | 0.751 | 7.7% | 32.5% | 0.237 |
| BM3 (vol-target) | 0.755 | 8.4% | 30.5% | 0.277 |

These match known SPY/60-40 stylised facts for 2002-2026 closely (SPY ~55% GFC drawdown, ~10%
CAGR; 60/40 Sharpe ~0.75, ~33% drawdown), a strong sanity check on the engine itself.

## Per-variant results (my recompute, raw-variant-Sharpe convention for A2/A3)

| Variant | Sharpe | CAGR | maxDD | Calmar | gap vs BM2 | boot [1%,99%] | DSR | placebo p95 | A3 | A4 | A5 | A8 | Tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1_primary | 0.898 | 7.17% | 20.4% | 0.352 | +0.146 | [-0.056, 0.350] | 0.9996 | 0.874 | T | T | T | T | B |
| V2_cash_dest | 0.752 | 6.05% | 23.0% | 0.264 | +0.001 | [-0.244, 0.219] | 0.9958 | 0.738 | T | F | F | T | C |
| V3_stricter | 0.923 | 7.74% | 20.1% | 0.386 | +0.172 | [0.004, 0.341] | 0.9997 | 0.846 | T | T | T | T | **A** |
| V4_alt_measure | 0.895 | 7.06% | 21.7% | 0.326 | +0.144 | [-0.091, 0.368] | 0.9996 | 0.875 | T | F | T | T | C |

PBO (7 series: V1-V4 + BM1-BM3, excess vs BM2, 8 partitions) = **0.000** (A6 pass).

**V3's A1 bootstrap lower bound (0.0036) is razor-thin** -- essentially a coin-flip away from
failing; it should be read as fragile, not a robust Tier-A pass on its own.

## Reconciliation with the primary's report

`docs/eodhd_s2_evaluation_2026_10.json` (read only after the above was final) computes **A2/A3
on EXCESS-vs-BM2 daily returns**, not the raw variant series (`run_eodhd_s2_evaluation.py`
L555-574, self-documented in its own `prereg_issues`). My first pass used the raw variant Sharpe,
which explains essentially all of the large A2/A3 gap. Redoing A2/A3 the same way
(`s2_excess_check.py`) reproduces the primary's result **exactly**: A3 = False for all 4
variants, and DSR lands in the same near-zero band:

| Variant | my DSR (excess) | primary DSR | my A3 (excess) | primary A3 |
|---|---|---|---|---|
| V1 | 0.0030 | 0.0058 | False | False |
| V2 | 0.00004 | 0.0002 | False | False |
| V3 | 0.0139 | 0.0253 | False | False |
| V4 | 0.0023 | 0.0032 | False | False |

**The task's flagged bug** ("the on/off override was fed back through the threshold, inverting
the placebo") is confirmed already fixed in the final harness: `placebo_variant()`'s
`on_override` path bypasses `_variant_decision_states` entirely (its own docstring says so), so
the permuted daily state is used directly, not re-compared against the threshold. My
independently-built placebo (same 63-day block permutation, 500 draws, built from scratch without
reading their code first) is the actual A7 check here, and it **confirms A3=False for every
variant** once the same excess-vs-BM2 convention is used -- i.e. bar A3 reproduces.

Once A2/A3 use this (more internally-consistent, since A1/A4/A5/A8 are all already framed "vs
BM2") convention, **no variant clears Tier B** (needs A3+A4+A5 all true) **or Tier A** in my
recompute either -- matching the primary's `tier_conditional_on_A7: "C"` for all four variants.
**Candidate tier: C, reproduces.**

Two smaller, fully explained mismatches:
- **V1 vs BM3, half 2** (one of six A4 sub-checks): primary's gap is barely negative (-0.0036);
  mine was uniformly positive. Traced to my own BM3 shortcut -- I hold the cash leg on the DTB3
  accrual proxy for BM3's entire history rather than switching to BIL's real total return from
  2007-05-30 (the general cash rule), a modeling gap in my own code, not the primary's.
- **V2's maxDD/Calmar/A5**: primary's decide() keeps the off-state leg on IEF for every variant,
  including V2, and only routes the incremental weight to BIL in on-state months. My 2-asset
  engine used BIL as V2's sole non-SPY leg throughout (on AND off), which is wrong for the
  off-state months. This is a genuine bug in my V2 implementation (V1/V3/V4 are unaffected, since
  their destination *is* IEF, matching the off-state leg exactly). Does not change V2's tier
  (Tier C either way).

## Bottom line

Breadth signal, PBO, A6, A8, and -- once the same excess-vs-BM2 convention is used for A2/A3 as
the primary's (documented) implementation -- A2 and A3 (including the task's flagged placebo
bug-fix) all reproduce. **Candidate tier reproduces: C.** The only unreconciled items are a
razor-thin single half/benchmark A4 sub-check for V1 (traced to my own BM3 cash-leg shortcut) and
a known bug in my own V2 implementation (BIL used for V2's off-state leg instead of IEF) -- both
explained, neither changes any tier outcome, and a new, independent finding (IEF/BIL both spend
long early stretches below the $50M ADV20 3bps threshold) worth the reviewer's attention.
