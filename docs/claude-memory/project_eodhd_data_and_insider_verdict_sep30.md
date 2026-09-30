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
