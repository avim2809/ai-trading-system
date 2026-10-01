---
name: project-eodhd-data-and-insider-verdict-sep30
description: 9/30 EODHD Historian download (local, owner may cancel) + insider-cluster prereg result Tier C + EODHD bad-bar data-quality issues
metadata:
  type: project
---
**EODHD Historian bought for 1 month** (owner, 2026-09-30; key `EODHD_API_KEY` in .env).
- **Downloaded** to `data/research/eodhd/` (gitignored, licensed, 2.8 GB), so the owner can cancel:
  - insider-universe EOD prices incl. delisted;
  - all 997 forex pairs;
  - 7,085 crypto series, active and delisted (the survivorship gate for crypto cross-sectional momentum);
  - splits and dividends for ~7.4k tickers;
  - 127 ETFs (industry, bond, commodity, country);
  - 18,025 US exchange-listed common stocks, active and delisted.
- **Scripts:** `scripts/fetch_eodhd_prices.py` and `scripts/fetch_eodhd_extras.py`. Both are resumable and take `--max-rps`; concurrent runs must sum below ~16/s (the limit is 1,000 req/min). The manifest merge is locked.
- **Not in the plan:** fundamentals, intraday, insider data.
- **Quirks:** volume is split-adjusted but open/close are raw, so dollar volume = adjusted_close × volume. News is sparse before 2019, so use it for forward tests only.

**Insider-purchase clusters (pre-registered, fp f355b075): Tier C, no edge.**
- The median event loses (−1.6% at 3m, −3.3% at 6m, excess vs the ADV-bucket ETF). The mean is +0.9%/+1.2% only from outliers, and negative after 2013.
- A Sonnet agent's independent recompute reproduced every bar exactly.
- Docs: `docs/insider_cluster_verdict_2026_09.md` and the `_recompute_2026_09.json` results; ledger `docs/insider_cluster_trial_history.json` (8 trials).
- Harness `calendar_time` was sped up (10+ min → 21 s), bit-identical, commit 7a41a58.

**EODHD bad bars faked two bar passes (DSR A2, PBO A6):**
- SMLP has a phantom 2015-12-25 bar at $0.0002 (~95% of the calendar-time mean);
- XBKS has a 250× scale error;
- ACRX and CERN have unadjusted reverse splits;
- QPAC has garbage quotes.

More remain: the 6m placebo p95 is still +75% after exclusions, and there are excess returns below −200%.

**Why:** these faked results and are easy to miss.
**How to apply:** every new EODHD pre-registration (owner's shortlist in `docs/research_brief_eodhd_findings.md`) must freeze a cleaning rule before the run: drop zero-volume and off-calendar bars, drop reverting single-day spikes, and check split jumps in the adj/close ratio. Related: [[project-edge-search-sep29]], [[project-allocation-portfolio-build-sep30]], [[feedback-verify-before-trusting-a-heuristic]].

**Update 2026-10-01: EODHD shortlist S1-S5, all Tier C** (`docs/eodhd_shortlist_verdict_2026_10.md`).
- Frozen 19:18:54Z on 9/30 under a shared protocol.
- **S1 (industry ETF momentum):** equals the equal-weight ETFs.
- **S2 (breadth overlay on 60/40):** a near miss. It fails only A1 and one A4 half, and the effect is defensive. The non-primary V3 would be Tier B, but choosing it after the fact is cherry-picking. It's a candidate for a forward paper test if the owner wants one.
- **S3 (bond/commodity trend):** equals buy-and-hold.
- **S4 (52-week high):** matches SPY.
- **S5 (crypto momentum):** no better than random. C1 BTC-trend beat BTC held again (Sharpe 1.13 vs 0.89, 2017-26).
- **Full history:** data re-downloaded from 1985 into `*_full/` (the first pulls were capped at 2005).
- **Cleaning v2 gap:** v2 (fc0690f0) misses the constant 999999.9999 sentinel. **Freeze a v3 before any new EODHD test.**
- **Agent lesson:** agents' "inferred" conventions can deviate from the literal frozen text. S2's harness computed A2/A3 vs BM2 although the text says the variant's Sharpe. Always check the bar definitions against the frozen text when a recompute disagrees.
- **Review package:** `review/README.md` + `scripts/build_review_package.py`; the private data bundle is under /local/store/review_bundles/. See [[feedback-external-review-package]].
