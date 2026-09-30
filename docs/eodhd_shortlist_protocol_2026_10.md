# EODHD shortlist: shared evaluation protocol (frozen 2026-09-30, amended once)

**Amendment 1 (2026-09-30 22:40Z).** Made before any candidate's pre-registration froze or computed a return. The first downloads all started on 2005-01-01 because of a fixed setting in the download script (`START`), so a full-history set (from 1985) was fetched into `*_full/` folders. Amendment 1 does four things:
- switches every candidate to the `*_full/` folders;
- adds cash and 60/40 proxies for the years before BIL and IEF existed;
- adds a `nav` asset type for no-volume mutual-fund series;
- moves the cleaning rule to v2 (fingerprint `f62cb2e4…`).

This covers the five candidates in `research_brief_eodhd_findings.md` §1, tested in parallel. Each candidate gets its own pre-registration, `scripts/<id>_preregistered_bars.py`, structured like `alt_premia_preregistered_bars.py`. Each is frozen and committed before any of its returns on the test window is computed, and each must follow this protocol. Where a pre-registration differs from this protocol, the protocol wins.

| id | Candidate | Implementation shape |
|---|---|---|
| S1 | Industry/sector ETF momentum | long-only rotation sleeve |
| S2 | Market-breadth overlay on the 60/40 core | overlay: scales the core's SPY weight |
| S3 | Bond duration momentum + commodity ETF dual momentum | two sub-sleeves, tested separately and combined |
| S4 | 52-week-high proximity, liquid US stocks | long-only cross-sectional sleeve |
| S5 | Crypto cross-sectional momentum | gated: the gate is checked first, and a failed gate ends S5 |

## 1. Data

- **Source:** only local data (no network):
  - EODHD full-history folders `data/research/eodhd/{etfs_full,us_universe_full,forex_full}/` (from 1985 or inception), plus `crypto/` (crypto history starts ~2010 in any case);
  - FRED `data/research/fred/DTB3.parquet` (3-month T-bill, from 1954).
  - The 2005-start folders `etfs/` and `us_universe/` are superseded.
- **Cleaning:** every bar series goes through `scripts/eodhd_clean.py:clean_bars` first. Each pre-registration records the cleaning fingerprint (v2, `f62cb2e4…`, from `cleaning_fingerprint()`).
  - The exchange calendar is SPY's dates, so equity bars before 1993-01-29 are dropped.
  - Mutual-fund NAV series (VFITX, VUSTX, VFISX in `etfs_full/`) are cleaned with `asset="nav"`. They are used only as the proxies named in §2–3, never as a traded holding. No return may be computed across a `segment` boundary: a holding that spans one is closed at the last close before it.
- **Survivorship:**
  - S4 builds its universe at each rebalance from **all** `us_universe/` stocks, active and delisted, using only information known at that date, i.e. trailing dollar volume and price.
  - Current index membership must not be used.
  - S5 uses active and delisted coins.
- **Dollar volume:** `adjusted_close × volume`, because EODHD volume is split-adjusted while open/close are raw.

## 2. Execution and costs

- **Timing:** signals use closes through day *t*, and trades fill at the **next bar's adjusted open** (`open × adjusted_close / close`). Crypto uses the next daily bar, stamped 00:00 UTC.
- **Robustness check (not a bar):** each pre-registration also reports results with a 2-day signal lag.
- **Costs per side:**
  - ETFs with 20-day median dollar volume ≥ $50M: 3 bps. Other ETFs: 10 bps.
  - Stocks, by ADV20: 60 bps below $1M, 30 bps for $1–5M, 15 bps for $5–20M, 5 bps above $20M. That's half the insider pre-registration's round-trip table.
  - Crypto: 35 bps (Alpaca taker fee of 25 bps plus 10 bps spread).
- **Cash:** cash earns BIL's total return from BIL's first bar (2007-05-30). Before that it earns FRED DTB3 accrued per trading day (rate / 100 / 252).
- **2× cost:** A5 doubles every cost.

## 3. Benchmarks

Every candidate is compared with:
- **BM1:** SPY buy-and-hold;
- **BM2:** 60/40 SPY/IEF, rebalanced monthly. Before IEF's first bar (2002-07-26), the bond leg is VFITX (Vanguard Intermediate-Term Treasury, NAV total return, from 1991-12);
- **BM3:** SPY vol-targeted to 12%, with weight = min(1, 0.12 / 21-day realised vol), traded when the target weight moves by more than 0.10;

on the candidate's own window. BM1 and BM3 start 1993-01-29 (SPY). No candidate window may start before 1993-02-01.

The candidate-specific *primary* benchmark is the honest comparison for its claim:

| id | Primary benchmark | Why |
|---|---|---|
| S1 | equal-weight buy-and-hold of the same ETF universe, rebalanced monthly | isolates the momentum ranking from sector beta |
| S2 | the plain 60/40 core (BM2) | the claim is that the overlay improves the core |
| S3 | bond: equal-weight buy-and-hold of the same duration ETFs; commodity: equal-weight basket of the same commodity ETFs; combined: 60/40 + the same-weight sleeve held as buy-and-hold | isolates the timing |
| S4 | equal-weight buy-and-hold of the same point-in-time liquid universe | isolates the 52-week-high ranking |
| S5 | BTC buy-and-hold, and the live C1 BTC 4-week trend rule | the claim is that a basket beats what already runs live |

A1 (below) applies to the primary benchmark. A4, A5 and D apply to the primary benchmark and BM1–BM3.

## 4. Inference

- **Multiple testing:** 5 candidates are tested at once, so the one-sided alpha is **0.05 / 5 = 0.01** per candidate.
- **Bootstrap:** paired stationary block bootstrap of the daily Sharpe gap (candidate minus benchmark), resampling the same days for both.
  - Mean block length: 63 trading days; for S5, 91 calendar days.
  - 5,000 draws; reuse `stationary_indices` from `scripts/run_alt_premia_evaluation.py`.
- **DSR:** `firm.eval.overfitting.deflated_sharpe` with:
  - trial Sharpes = the daily Sharpes of every variant the candidate declares;
  - `prior_trials` = **190** (the cumulative trials in every existing ledger up to 2026-09-30: combination 57, pattern ML 104, standalone 11, alt premia 10, insider 8) **plus** the variant counts of the other four shortlist candidates, fixed at freeze from each pre-registration's declared grid.
  - The variant grid is small, declared up front, and never extended after a run.
- **PBO:** `cscv_pbo` over the candidate's variants plus BM1–BM3, 8 partitions.
- **Placebo:** 500 draws; the rule is chosen per candidate and frozen:
  - rank-based candidates (S1, S4, S5): random rankings with the same number of holdings and the same rebalance dates;
  - S2: the breadth signal's on/off states permuted in 63-day blocks;
  - S3: each asset's monthly on/off series permuted in 12-month blocks.
- **Halves:** the candidate's window is split at its midpoint date, fixed at freeze from data availability, not returns. A4 requires a positive gap in both halves.

## 5. Tiers (precedence A > D > B > C, as in `alt_premia_preregistered_bars.classify`)

- **A1:** Sharpe gap vs the primary benchmark > 0, and its bootstrap lower bound at alpha 0.01 > 0.
- **A2:** DSR > 0.95.
- **A3:** Sharpe above the 95th percentile of the placebo Sharpes.
- **A4:** Sharpe gap > 0 in both halves, vs the primary benchmark and BM1–BM3.
- **A5:** Sharpe gap > 0 at 2× costs, vs the primary benchmark and BM1–BM3.
- **A6:** PBO < 0.50.
- **A7:** an independent recompute reproduces every bar outcome.
- **S2 only:** it is a defensive claim, so it adds **A8**: max drawdown and Calmar ratio both better than plain 60/40, with the Sharpe gap vs 60/40 > −0.05. It is judged on max drawdown and Calmar, not on raw Sharpe alone.
- **D:** the bootstrap upper bound at alpha 0.01 of the Sharpe gap is < 0 vs the primary benchmark.
- **B:** point estimate > 0 vs the primary benchmark, and A3, A4 and A5 pass. Deploy only as "not proven"; the owner decides.
- **C:** everything else.

**Actions:**
- **A:** a live config diff for the owner's sign-off. The 2-day-lag and per-year audits come first.
- **B:** the owner decides.
- **C:** not deployed.
- **D:** rejected.

## 6. Process rules

- **Freeze first:** design work may look at data *availability* (coverage, start dates, counts, liquidity), never at candidate or placebo returns on the window. The frozen pre-registration is committed before the first return is computed.
- **Ledgers and outputs:** each candidate gets its own ledger, `docs/<id>_trial_history.json` (new family, never reset), and results in `docs/<id>_evaluation_2026_10.json`.
- **Resources:** the host is shared with two live trading services, and about 3 GB of RAM is free.
  - Stream per ticker; use float32; read only the columns needed.
  - Keep peak memory per process under 1.5 GB, and run with `nice -n 10`.
  - Run no full pytest suite, only the candidate's own tests.
- **Boundaries:** no edits to `src/`, `config/` or services; no network calls.
