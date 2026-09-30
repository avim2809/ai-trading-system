# Edge search verdict (2026-09-29/30)

**Question:** does any approach make money out of sample, net of costs, and beat
simple benchmarks? Changing the architecture was allowed.

**Answer: no, not on the evidence available.** Everything tested was pre-registered and fingerprinted before it ran:
- the 11 current strategies, one at a time;
- every way of combining them, re-measured on a corrected engine;
- seven alternative return sources with long public track records.

None of them demonstrably beats buy-and-hold SPY, 60/40, or vol-targeted SPY.
Over 2020–2026 the live configuration behaved like **cash plus a little**:
- Sharpe above T-bills 0.14;
- +25% total return;
- beta about 0.

Buy-and-hold SPY returned +152% over the same days.

**The honest recommendation is the one the brief anticipated: trade the benchmark.**
The pre-registered fallback is **60/40 SPY/IEF, rebalanced monthly** (decision D1,
default). The one lead worth a forward paper test is a small **Bitcoin trend
sleeve**. It is not proven and it's labelled as such.

Plan: `docs/edge_search_plan_2026_09.md`.

Pre-registrations:
- `scripts/alt_premia_preregistered_bars.py`, fingerprint `76598975…`;
- `scripts/standalone_strategy_preregistered_bars.py`, fingerprint `1d54517a…`;
- the 9/28 design, reused unchanged, fingerprint `96bf26c8…`.

Every headline number below was recomputed by an independent agent with its own code (§6).

---

## 1. A real engine bug, found along the way, and what it changed

The backtest broker filled and marked positions at the cache's **raw,
split-unadjusted close**. Every stock split in the universe therefore booked a
fake one-day P&L jump on any position held through it:
- a long lost 75–95%;
- a short gained the same.

The live-config C3 run's worst day, **−6.96% on a 0.4%-daily-vol book, was the
NVDA 10:1 split** (2024-06-10). There are 8 such days in 2020–2026, several inside
the 9/28 test windows. Raw closes also ignored dividends.

**The fix is total-return-adjusted backtest prices** (commit `7e9fedb`, on branch
`research/edge-search`, not yet on `main`). It's covered by regression tests that
fail on the old code.

Re-running the frozen 9/28 design on the fixed engine:

| | C0 legacy (live) | C1 robust/attr | C2 robust/standalone | C3 confidence | P random placebo |
|---|---|---|---|---|---|
| OOS Sharpe, 9/28 (bug) | −0.69 | −0.54 | −0.52 | −0.32 | −0.23 |
| **OOS Sharpe, fixed** | **−0.22** | −0.73 | +0.31 | −0.07 | −0.24 |
| Full-period Sharpe, 9/28 | 0.07 | 0.42 | 0.60 | 0.57 | 0.49 |
| **Full-period Sharpe, fixed** | **0.67** | 0.51 | 0.35 | 0.94 | 0.67 |

These are raw-return Sharpes, the 9/28 evaluator's frozen convention.

**This is a re-measurement on a corrected price feed, not a split-day patch.**
The fix rescales every bar of history, adding dividends and giving adjusted OHLC
and volume for the strategies that read them. And once a portfolio stops taking
fake split losses, its NAV and positions follow a different path. Old and
corrected daily returns correlate only 0.56–0.80 even on non-split days. Same
design and code path; different, correct inputs.

PBO fell from 0.97 to 0.46. **The verdict is unchanged: FAIL.**
- Both primary candidates fail B1–B4.
- C2's +0.53 edge over C0 has a CI of [−1.31, +2.44] and a DSR of 0.22.
- No method is distinguishable from the random-weights placebo.

The bug inflated how bad the 9/28 system looked, but not enough to create an edge.

**Against cash and SPY**, same days, 2020-01 → 2026-06:

| | Sharpe above T-bills (full) | Total return (full) | OOS Sharpe above T-bills | Beta |
|---|---|---|---|---|
| SPY buy-and-hold | 0.67 | +152% | +1.35 | 1 |
| C0 live config | 0.14 | +25% | −0.82 | 0.02 |
| C3 confidence | 0.47 | +42% | −0.65 | 0.08 |
| Random placebo | 0.03 | +20% | −1.04 | −0.02 |

## 2. Step 1: do any of the current strategies have standalone edge? **No.**

Each strategy's own signal book was evaluated on the same 4 OOS windows as 9/28, 491 days in total:
- the unit-gross `score × confidence` book the engine records;
- adopted with a 1-day lag;
- charged 12 bps per side;
- beta-hedged to SPY using the train window only.

Bars: bootstrap lower bound > 0 at α = 0.05/11; beat a monthly symbol-permutation placebo; DSR > 0.95; ≥3/4 folds positive. **Survivors: none.**

| Strategy | OOS hedged net Sharpe | Gross (pre-cost) | Daily turnover | Folds > 0 |
|---|---|---|---|---|
| multi_factor | +0.10 | +0.28 | 0.07 | 2/4 |
| momentum | −0.20 | −0.06 | 0.11 | 2/4 |
| trend | −0.56 | −0.51 | 0.02 | 1/4 |
| event_driven | −1.13 | −0.66 | 0.25 | 0/4 |
| regime_hmm | −1.41 | **+0.88** | 0.68 | 0/4 |
| mean_reversion | −1.56 | −0.73 | 0.78 | 0/4 |
| pattern_recognition | −1.67 | −0.77 | 0.56 | 0/4 |
| sentiment | −2.59 | −0.85 | 0.13 | 0/1 (data only from 2025-11) |
| seasonality | −2.68 | −1.02 | 1.00 | 0/4 |
| volatility_breakout | −3.11 | −1.34 | 0.65 | 0/4 |
| stat_arb | −3.41 | −0.89 | 0.61 | 0/4 |

**How to read it:**
- Most books are negative even **before costs**.
- Several books turn over 60–100% of their gross exposure every day, which turns weak gross results into strongly negative net ones.
- `regime_hmm` has a positive gross signal that its turnover destroys. That's a post-hoc observation from 1 of 11, not a result. A slower variant would be a new hypothesis with a new ledger entry.
- `pattern_recognition` was already grid-searched on this period, so it is in-sample regardless.
- My recomputed book returns match the engine's own attribution series to about 1e-16 for all 11 strategies.
- Ledger: `docs/standalone_strategy_trial_history.json` (11 trials).

**Universe caveat:** the 25 names were picked in 2026, with winners in hindsight. That biases *towards* finding edge, and none was found.

## 3. Step 2: a genuinely different design. **Nothing beats the benchmarks.**

Seven candidates were chosen from families with long public records, with parameters taken from the source papers and never tuned:
- 2007–2026 data, free from Tiingo, CBOE and FRED;
- signal at close t, trade at close t+1;
- costs charged;
- no leverage.

Every candidate lands in **Tier C (inconclusive)**. None is Tier A (proven better), B (not worse and measurably safer) or D (measurably worse).

| Candidate | Window | Sharpe | Gap vs SPY / 60-40 / VT-SPY | Worst lower bound | Tier |
|---|---|---|---|---|---|
| V1: VIX-contango-timed short vol (SVXY, includes Feb-2018) | 2011-12→ | 0.42 | −0.42 / −0.42 / −0.45 | −1.06 | C |
| V2: CBOE put-write index | 2007-02→ | 0.42 | −0.13 / −0.22 / −0.25 | −0.57 | C |
| K1: turn-of-month plus pre-FOMC SPY | 1994-02→ | 0.37 | −0.14 / −0.30 / −0.20 | −0.84 | C |
| T2: cross-asset trend, 8 ETFs, long/flat | 2007-03→ | 0.55 | −0.01 / −0.09 / −0.13 | −0.67 | C |
| M1: inverse-vol mix of SPY, T2, V1, K1 | 2012-04→ | 0.55 | −0.24 / −0.24 / −0.26 | −0.61 | C |
| M2: M1 plus BTC trend | 2015-05→ | 0.82 | +0.12 / +0.16 / +0.08 | −0.39 | C |
| **C1: BTC 4-week trend vs BTC held at matched exposure** | 2015-02→ | **1.35 vs 1.07** | **+0.28** | −0.21 | C |

Sharpes here are above T-bills.

**How to read it:**
- The point estimates are mostly **negative**. Well-known premia didn't beat the benchmarks in this period, net of costs.
- The pre-run power analysis (`docs/alt_premia_power_analysis_2026_09.json`) showed the tests could only detect gaps of 0.4–0.9 Sharpe. So "inconclusive" is the expected outcome for a real but modest edge. The negative point estimates are why it's not just a power problem.
- **C1 is the only lead.** It passes 6 of 7 Tier A bars:
  - DSR 0.997;
  - beats its placebo;
  - positive after publication (+0.33);
  - survives double costs;
  - smaller drawdown;
  - PBO.

  It fails only significance.
- **C1 audits:**
  - positive in 8 of 12 years;
  - leaving out any single year keeps the gap between +0.18 and +0.43;
  - 2022 contributes about a third of it;
  - it degrades smoothly with extra execution delay: +0.24 at 1 day, +0.23 at 2 days, +0.15 at 4 days.
- **C1's caveats:**
  - BTC is crypto's survivor;
  - the drawdown is still 49%;
  - volatility is 32%.
- Report: `docs/alt_premia_evaluation_2026_09.json`. Ledger: `docs/alt_premia_trial_history.json` (10 trials).

## 4. What it would take to run the recommendation live (proposal, not applied)

**No `config/live*.yaml` change has been made.** Everything below needs your sign-off on the exact diff.

### A. 60/40 SPY/IEF, monthly (the pre-registered fallback)

**It can't be expressed through the current pipeline.** The risk agent's caps would block it:
- `max_position_pct` is 5% per name;
- the net-exposure cap is 0.5.

A 60% SPY weight is impossible without switching them off.

Options, smallest first:
1. **A standalone allocator job**, about 100–150 lines.
   - A scheduled script: on the first US trading day of each month, after the open, it reads positions through the existing `Broker` adapter and sends market orders to reach 60/40 by NAV.
   - It skips anything within a 2% band and logs through the standard audit path.
   - The 11-strategy engine must be **stopped on that instance**, because both would trade the same account. Setting `FIRM_AUTO_START_LIVE=0` for that unit keeps it stopped across restarts.
   - Cheapest, and easy to reason about.
2. **An `allocation` strategy mode in the engine.**
   - Targets computed from completed daily bars once per ET trading day, bypassing analysts/PM.
   - A minimal risk check: long-only, gross ≤ 1.
   - The existing `ExecutionAgent` band and `_day_anchor`.
   - More work, but it keeps one process, the dashboard and the kill switch.

**Suggested rollout:**
- Switch **one** instance to 60/40 and keep the other on the current system as the control for 3 months.
- Alpaca (sleeved, $100k) is the natural candidate to switch: it would reuse this session's fixes and none of the sleeve machinery would be needed.

### B. BTC trend sleeve, forward test only (Tier C, not a validated edge)

- Alpaca supports BTC/USD spot. Size it at **5–10% of NAV**, with the rule exactly as pre-registered:
  - weekly review;
  - long if the 4-week return > 0, else cash;
  - 40% volatility target;
  - 25 bps per-side cost assumption.
- Treat it as a forward paper experiment with its bars fixed in advance, not as a deployment.
- It needs crypto order support in the allocator, and a 24/7 calendar for the weekly review.

## 5. What was tried and failed, in one list

- **Signal combination** (5 methods × 2 engines). The placebo is indistinguishable.
- Each current strategy standalone: **0/11**.
- Timed short vol, put-writing, turn-of-month/FOMC, cross-asset trend, inverse-vol premia mix: **all Tier C with negative point estimates**.
- M2 (premia plus BTC) and C1 (BTC trend): Tier C, slightly positive, not proven.
- Earlier failures, still standing: pattern-ML layer, zscore_demean, concentration, joint_optimizer, stat_arb pairs, seasonality and macro overlays (`docs/formal_pbo_audit.md`).

## 6. Verification and rigor notes

**Step 2** was recomputed from the spec text by an independent agent with its own engine. It matched to 3–4 decimals on every point estimate; bootstrap bounds were within about 0.05 (resampling noise). Also confirmed:
- no look-ahead: shifting every signal one day only degrades results;
- the Feb-2018 SVXY −83% is in the data and V1's exposure cap held at exactly 0.25;
- FOMC dates are correct;
- BTC data gaps are immaterial.

A mechanical look-ahead self-check also passed: every input after a cutoff was perturbed, and no earlier return changed.

**Step 1 and the replication** were recomputed by a second independent agent from the raw signal books, prices and return files:
- **Step 1:** all 11 Sharpes, fold Sharpes, gross Sharpes and betas match to floating-point precision. Bootstrap bounds differ only by seed noise and never change sign.
- **Turnover is real:** the actual weight vectors show `seasonality` flipping its whole 25-name book long↔short on many days, and `mean_reversion` reweighting continuously.
- **`regime_hmm`'s gross edge is sign-stable but noisy:** +1.45, +0.88 and +1.19 at lags 0, 1 and 2.
- **Replication:** every OOS and full-period Sharpe and the PBO (0.4571) match exactly, and bootstrap point estimates match.
- Not independently recomputed: the placebo percentiles and DSRs. They aren't load-bearing, because the bars the verifier did check already fail for every candidate.

**Not ruled out by any check here:** look-ahead *inside* the strategies' own signal code. The books were shown to match the engine exactly, but the strategy code wasn't re-audited. That can only make the negative results less negative, so it doesn't threaten the verdict.

**Provenance:** the overnight runs used tree `adb8919` (06e5d54 + fix + preregs), recorded in the session's `runs/step1/launch_provenance.txt`. The branch was then rebased onto `main`'s `3be238f`, a live-only order-routing change.

**Disclosures:**
- The Step 2 harness had two bugs, found by its own simulation check before any candidate ran:
  - the analytic power formula used annual instead of daily Sharpes;
  - the power simulation fixed the realised gap.

  Both were fixed. They affected reporting only, not any bar.
- The Step 1 prereg was committed before its run.
- The Step 2 prereg was frozen (hashes in the session scratchpad) about 1h before its commit. It was byte-identical at commit.

**Also found, not fixed:**
- **IBKR live, latent.** When IBKR's data farm is down, the REST fallback serves raw OHLC. `LiveTradingEngine._resolve_cycle_prices` and `_closing_price` prefer raw `close`. Around a split, per-strategy attribution, and so the live `optimal` weights, would take a fake jump. Current-price order sizing is unaffected. Alpaca is clean.
- **Test hygiene.** Some tests write into `data/llm_cache.db` and `data/vectordb` relative to the working directory. From the live checkout that's the production instance's data dir.
