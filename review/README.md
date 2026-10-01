# Review package: 2026-09 research programme

**Audience:** a third-party reviewer with no prior access to this system.

**What is being reviewed:** the owner asked for an approach that makes money out of sample, even at the cost of abandoning the existing architecture.

**The programme:**
1. The existing 11-strategy system was tested and rejected.
2. A simple allocation portfolio went live on Alpaca paper trading.
3. Nine new ideas were each pre-registered and tested. None beat honest benchmarks.

**Conventions:**
- Every claim below points to a file in this repo, a commit, or an artifact under `review/`.
- All times are **UTC**. The host clock is Asia/Jerusalem (UTC+3), and raw log lines are local time. `TIMELINE.md` converts them.

## 1. How to check the work

1. **Freeze came before the run.**
   - Each study has a pre-registration module (`scripts/*_preregistered*.py`) that fixes its data, rules, costs, benchmarks, bars (pass/fail tests) and tiers.
   - Each module has a fingerprint: a SHA-256 over its frozen dicts, from `bars_fingerprint()`.
   - Compare the freeze commit's UTC time with the study's first run log in `TIMELINE.md`.
   - Recompute a fingerprint with `python -c "import sys; sys.path[:0]=['scripts','src']; import <module> as m; print(m.bars_fingerprint())"` and check it against the value cited in the study's report.
2. **The harness implements the frozen rules.**
   - Read `scripts/run_*_evaluation.py` against its pre-registration.
   - Run its synthetic-data tests (`tests/test_*`).
   - Each report lists `prereg_issues`: ambiguities in the frozen text, resolved literally, never patched.
3. **The numbers reproduce.**
   - Rerun the harness on the inputs, using the data bundle or your own vendor pull (§3), and compare with `review/studies/<study>/artifacts/`.
   - Independent recomputes, written from the pre-registration text without the harness code, exist for the insider test and S2.
4. **The data are what they claim.**
   - `review/data/MANIFEST.csv.gz` lists every input file with its size, SHA-256, row count and first/last date.
   - `review/data/MANIFEST_SUMMARY.csv` summarises it by dataset and licence.

## 2. Studies

Tiers use the precedence A > D > B > C:
- **A:** passes every bar; proposed for live use.
- **D:** significantly worse than its benchmark.
- **B:** positive point estimate and robust, but not proven.
- **C:** inconclusive; not deployed.

| # | Study | Pre-registration (freeze commit) | Harness / tests | Result | Verdict doc | Artifacts |
|---|---|---|---|---|---|---|
| 1 | Edge search step 1: 11 existing strategies standalone | `standalone_strategy_preregistered_bars.py` (`bd11bd9`) | `run_standalone_strategy_evaluation.py` | 0 of 11 have an edge | `docs/edge_search_verdict_2026_09.md` | `studies/edge_search_step1_standalone/` |
| 2 | Edge search step 2: 7 alternative premia | `alt_premia_preregistered_bars.py` (`bd11bd9`) | `run_alt_premia_evaluation.py`, `alt_premia_data.py` | all Tier C | same | `studies/edge_search_step2_alt_premia/` |
| 3 | Signal-combination methods, re-measured on the fixed engine | `combination_preregistered_bars.py` (`655e5e5`) | `run_combination_evaluation.py` | FAIL | `docs/optimal_combination_fix_2026_09.md` | (step-1 runs) |
| 4 | Allocation portfolio (60/40 core + BTC-trend sleeve), forward test | `allocation_forward_test_preregistered.py`, `docs/allocation_forward_test_trial_history.json` | `allocation_replay.py`, `src/firm/allocation/`, `tests/test_allocation_*` | live on Alpaca paper since 2026-09-30; judged forward | `docs/allocation_forward_test_plan.md`, `docs/allocation_deploy_runbook.md` | replay: `docs/allocation_replay_2026_09.json` |
| 5 | SEC Form 4 insider-purchase clusters | `insider_cluster_preregistered_bars.py` (`d6013c5`) | `run_insider_cluster_evaluation.py` | Tier C; bad vendor bars documented | `docs/insider_cluster_verdict_2026_09.md` | `studies/insider_clusters/` |
| 6 | S1 industry/sector ETF momentum | branch `research/eodhd-S1` (`d7224a5`) | `run_eodhd_s1_evaluation.py` | Tier C | `docs/eodhd_shortlist_verdict_2026_10.md` | `studies/shortlist_S1_industry_momentum/` |
| 7 | S2 market-breadth overlay on 60/40 | `research/eodhd-S2` (`ff6f44a`) | `run_eodhd_s2_evaluation.py`, `eodhd_breadth.py` | Tier C; defensive effect only (post-hoc) | same | `studies/shortlist_S2_breadth_overlay/` |
| 8 | S3 bond duration + commodity ETF trend | `research/eodhd-S3` (`7c5ff4e`) | `run_eodhd_s3_evaluation.py` | Tier C (all three parts) | same | `studies/shortlist_S3_bond_commodity_trend/` |
| 9 | S4 52-week-high proximity, liquid US stocks | `research/eodhd-S4` (`f379eca`) | `run_eodhd_s4_evaluation.py`, `eodhd_s4_build_panel.py` | Tier C (matches SPY; beats only a weak equal-weight universe) | same | `studies/shortlist_S4_52wk_high/` |
| 10 | S5 crypto cross-sectional momentum | `research/eodhd-S5` (`f0ec293`) | `run_eodhd_s5_evaluation.py` | Tier C | same | `studies/shortlist_S5_crypto_momentum/` |

**Shared rules for S1–S5:** `docs/eodhd_shortlist_protocol_2026_10.md`, frozen before any S design, with one amendment made before any freeze. It fixes:
- α = 0.05/5 = 0.01;
- DSR prior trials: 190 from earlier ledgers plus the other shortlist variants;
- costs and next-open execution;
- benchmarks: SPY, 60/40, vol-targeted SPY, plus a primary benchmark per candidate;
- tiers.

**Cumulative trial ledgers (never reset):** `docs/*_trial_history.json`.

**Bar-cleaning rule for EODHD data:** `scripts/eodhd_clean.py` (v2, fingerprint `fc0690f0…`), tests in `tests/test_eodhd_clean.py`. It was frozen before the S designs, after the insider test showed that bad vendor bars can fake a pass.

## 3. Data

| Dataset | Source | Licence | In git? | How to obtain |
|---|---|---|---|---|
| EODHD EOD prices (stocks active and delisted, ETFs, crypto, forex, splits/dividends) | EODHD Historian | licensed | no, manifest only | `scripts/fetch_eodhd_prices.py` and `scripts/fetch_eodhd_extras.py` (`--start 1985-01-01 --suffix _full`) with your own `EODHD_API_KEY` |
| SEC Form 4 insider purchases and cluster events | SEC EDGAR | public | bundle | `scripts/fetch_insider_data.py` |
| 3-month T-bill (DTB3) | FRED | public | bundle | `alt_premia_data.fetch_fred("DTB3", key)` |
| Alt-premia inputs (ETF/BTC prices, VIX, PUT index, FOMC calendar) | Tiingo, CBOE, FRED, Fed | Tiingo licensed; rest public | bundle | `scripts/alt_premia_data.py` |
| System price cache (edge-search step 1) | the system's vendor feeds | licensed | bundle | the system's own fetch path; see `docs/PROJECT_CONTEXT.md` |

- **Exact bytes used:** in the private data bundle (`DATA_BUNDLE.md`). The owner decides who receives it.
- **Without the bundle:** re-download and compare hashes and row counts. Vendors revise history, so some files will differ.
- **Known vendor-data issues:** documented in `docs/insider_cluster_verdict_2026_09.md`:
  - phantom holiday bars;
  - scale errors;
  - unadjusted reverse splits;
  - zero-volume quotes;
  - EODHD volume split-adjusted while open/close are raw.

## 4. Environment

- Python version and platform: `ENVIRONMENT.json`.
- Exact packages: `requirements-lock.txt` (`pip install -r review/requirements-lock.txt`).
- Scripts expect `PYTHONPATH=src` and run from the repo root.
- Research runs used a scratch `FIRM_DATA_DIR` so they never touched live state.

## 5. Ad-hoc scripts

`session_scratch_scripts/` holds every script that ran from the session scratchpad and is not in `scripts/`. These include:
- availability checks behind the design choices;
- the independent recomputes (`run_outputs/insider_recompute/recompute.py`, `posthoc_full.py`, and the S2 recompute);
- the equivalence check for the faster insider harness (`verify_fast_cal.py`);
- the freeze script for S1–S5 (`freeze_shortlist.py`);
- the Discord notifier.

None of them holds credentials; keys are read from `.env`, which is not included.

## 6. Errata and disclosures

- **Insider pre-registration timestamp:** its label reads 17:15Z, but the freeze commit `d6013c5` is 16:57:30Z and the run began at 16:57:44Z. It was local time written as UTC. The order holds, and the label was left in place so the fingerprint stays unchanged. See the verdict doc.
- **Cleaning rule freeze time:** v2's "frozen at" label was corrected (fingerprint `f62cb2e4` → `fc0690f0`) before any shortlist freeze.
- **S2 placebo bug:** the harness's first real run fed the placebo's on/off overrides back through the threshold. That inverted the placebo and made A3 falsely pass. It was fixed in a separate commit and rerun, which turned A3 into a fail.
- **S5 smoke test:** a build-only smoke test (202 files) and the full data build ran before the harness commit, but no bar was viewed. This is disclosed in the report's `harness_log`.
- **Pre-registration issues:** each shortlist report lists its `prereg_issues`, ambiguities resolved literally.
- **Post-hoc analyses:** all are labelled as such: the insider bad-ticker exclusion and the S2 static-mix comparison. They never change a tier.
