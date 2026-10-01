# EODHD shortlist: verdict (2026-09-30)

**Result: none of the five candidates shows an edge. Nothing changes live.**

**What was tested:** five ideas from the owner's external research shortlist (`research_brief_eodhd_findings.md`). Each was tested under its own pre-registration, frozen 2026-09-30T19:18:54Z before any return on its window was computed.

**Shared rules:** `eodhd_shortlist_protocol_2026_10.md`, fixing:
- α = 0.01, from 0.05/5;
- DSR prior trials: 190 from earlier ledgers plus the other shortlist variants;
- costs and next-open execution;
- the benchmarks SPY, 60/40 and vol-targeted SPY, plus a primary benchmark per candidate.

**Data:** full-history EODHD data from 1985, cleaned by the frozen rule in `scripts/eodhd_clean.py` (v2, `fc0690f0`).

**For a reviewer:** start with `review/README.md`.

| | Candidate | Window | Primary benchmark | Result (Sharpe, candidate vs primary) | Tier |
|---|---|---|---|---|---|
| S1 | Industry/sector ETF momentum (6-1, top tercile) | 2002-01 → 2026-09 | equal-weight same ETFs | 0.431 vs 0.436. It trails SPY, 60/40 and vol-targeted SPY, sits below the random-ranking placebo, and has PBO 0.67 | **C** |
| S2 | Breadth overlay on 60/40 (% of stocks above their 200-day average < 50% → SPY 0.6 → 0.3) | 2002-01 → 2026-09 | plain 60/40 | 0.91 vs 0.76 (total return), maxDD 21% vs 32%. Passes A2, A3, A5, A6 and A8. Fails A1 (gap lower bound −0.062) and A4 (half-2 gap vs vol-targeted SPY −0.0036) | **C** (near miss) |
| S3 | Bond duration momentum / commodity dual momentum / 60/40 + 10% of both | 2002-07 / 2007-01 → 2026-09 | the same assets held | bond 0.18 vs 0.21; commodity 0.26 vs 0.27; combined 0.65 vs 0.64 | **C, C, C** |
| S4 | 52-week-high proximity, top decile of the 500 most liquid US stocks, 6-month overlapping cohorts | 1999-02 → 2026-09 | equal-weight same universe | 0.54 vs 0.25. Passes A1 (lower bound +0.07) and A3. Fails A2 (DSR 0.84), A4, A5 and A6 (PBO 0.54). About the same as SPY (0.52), below 60/40 and vol-targeted SPY. The 1-month variant is Tier D | **C** |
| S5 | Crypto cross-sectional momentum (top 20 coins by volume, ≥ $1M/day) | 2017-05 → 2026-09 | BTC held; live C1 BTC-trend rule | 0.45 vs 0.89 (BTC) and 1.13 (C1). Ranking beats only 59% of random picks | **C** |

## What the results say

- **The ranking signals (S1, S4, S5) add nothing over random or equal-weight selection** from the same universe, once point-in-time universes, costs and next-open execution are applied.
- **S2 is the only near miss.** It is judged on its declared primary variant V1.
  - It fails only the Sharpe significance bar (A1) and one half-window comparison against vol-targeted SPY (A4, −0.0036).
  - The non-primary variant V3 (threshold 40%) would individually reach Tier B, with an A1 lower bound of −0.003. Choosing it after the fact would be multiple testing, so this is descriptive only.
  - The effect is defensive. A labelled post-hoc check (it doesn't change the tier) compared the overlay with a static mix at the overlay's own average SPY weight:

  | Variant | Sharpe | CAGR | maxDD | Gap vs static mix, 99% bounds |
  |---|---|---|---|---|
  | V1, overlay | 0.91 | 7.25% | 21% | +0.10, [−0.10, +0.29] |
  | V1, static 52/48 | 0.81 | 7.31% | 27% | |
  | V3, overlay | 0.93 | 7.83% | 19% | +0.14, [−0.03, +0.30] |
  | V3, static 55/45 | 0.79 | 7.49% | 29% | |

  Most of its edge over 60/40 comes from holding less equity on average. What's left is a lower drawdown at about the same return, and it isn't significant. It is the one lead from the shortlist that could justify a forward paper test, and the owner decides.
- **S3's trend rules match buy-and-hold** of the same bonds or commodities. Timing adds nothing, consistent with the earlier cross-asset trend result (alt-premia T2, Tier C).
- **S5's benchmark confirms an earlier lead.** The live C1 BTC 4-week trend rule again beat BTC buy-and-hold over 2017-05 → 2026-09 (Sharpe 1.13 vs 0.89). This supports the live BTC sleeve, but it wasn't the tested hypothesis.

- **S4 beats a weak benchmark, not the market.** It beats its own equal-weight universe (3.1% CAGR, 77% maxDD), but only matches SPY. Part of that universe's weakness is bad vendor data (below), which flatters S4's A1 pass.

## Data-quality gap found during S4 (cleaning rule v3 needed)

`scripts/eodhd_clean.py` v2 misses a **constant vendor placeholder price**: 999999.9999 held across many bars. It neither reverts within 5 bars nor jumps once.
- It affects 69 of 4,911 tickers in S4's run and 182 of the 16,229 ever eligible.
- Five of S4's top-10 losing contributors carry it.
- It mainly distorts dollar-volume eligibility and the equal-weight benchmark, so S4's verdict is conservative.

Any future EODHD test must first freeze a v3 rule that drops placeholder values (e.g. ≥ 999999) before it runs.

## Process notes (full list in `review/README.md` §6)

- **Freeze order:** every pre-registration was frozen in its own commit before its harness ran. The harness was committed before the real-data run. The UTC order is shown in `review/TIMELINE.md`.
- **S2 placebo bug:** the harness's first real run inverted the placebo, which made A3 falsely pass. It was fixed in a separate commit.
- **S2 convention correction:** the harness then computed A2 and A3 on returns in excess of 60/40, which was an inference. The frozen text says "Sharpe of each variant" (for PBO it explicitly says "excess vs BM2"), so A2/A3 were recomputed on the literal reading (Sharpe in excess of cash) in a disclosed commit, `21e5dad`, with both conventions reported. V1's tier is C under either.
- **S2 recompute:** an independent recompute reproduces the breadth signal (episode counts match exactly), PBO 0.000, and every bar under the matched convention. Its only disagreements come from its own shortcuts (vol-targeted SPY's cash leg; a V2 destination bug) and a total-return rather than cash-excess Sharpe scale. None moves V1's tier.
- **S5 smoke test:** a build-only smoke test ran before the harness commit; no bar was viewed. This is disclosed.
- **S5 bad coin bar:** one coin with implausible prices (MAPR) got past the cleaning rule and *raised* S5's return. The verdict is conservative.
- **S4's three runs:** S4 ran three times, because two genuine harness bugs were found by its sanity checks: a missing 1/6 cohort weight, and a holding not exiting at a series break. All three runs are disclosed in the report's `harness_log`; no tier or primary bar moved across them.
- **Pre-registration issues:** each report lists its `prereg_issues`, ambiguities resolved by the most literal reading. None could change a tier.
- **Power:** every pre-registration's power analysis said the test was underpowered against the decayed, post-publication effect sizes. Tier C was the expected outcome unless an effect was large. A Tier C result means "not shown", not "shown to be zero".

## Records

| | Evaluation report | Ledger |
|---|---|---|
| S1 | `docs/eodhd_s1_evaluation_2026_10.json` | `docs/S1_trial_history.json` |
| S2 | `docs/eodhd_s2_evaluation_2026_10.json` | the file named in its pre-registration |
| S3 | `docs/eodhd_s3_evaluation_2026_10.json` | `docs/S3_trial_history.json` |
| S4 | `docs/eodhd_s4_evaluation_2026_10.json` | the file named in its pre-registration |
| S5 | `docs/eodhd_s5_evaluation_2026_10.json` | the file named in its pre-registration |

Run artifacts are in `review/studies/shortlist_S*/`.
