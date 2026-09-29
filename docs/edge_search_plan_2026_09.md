# Edge search: plan for review (DRAFT v2, 2026-09-29)

**Status: DRAFT for owner review. Nothing has been run, and nothing is frozen.**
Once you approve it, the frozen version becomes fingerprinted pre-registration
files (§8), committed before any evaluation runs. This doc will then point to
them and will not be edited to fit results.

**v2 changes the approach itself.** Owner feedback on v1 was that testing
textbook ETF trend-following, where the likely result is "trade the benchmark",
isn't looking hard enough. §2 explains what went wrong and what v2 does instead.

---

## 1. Starting point

- **Leading hypothesis:** the current 11-strategy / 25-symbol signal set has no edge.
- **The evidence:**
  - the 9/28 combination evaluation (`docs/optimal_combination_fix_2026_09.md`): every combination method lost out of sample, a random placebo did best, PBO 0.971;
  - `docs/formal_pbo_audit.md`: 6 architecture changes all failed;
  - `docs/pattern_ml_final_verdict_2026_09.md`;
  - `docs/portfolio_construction_diagnosis.md`.
- **What's excluded:** more tuning of combination, pattern-ML or the strategy roster.

**Facts found while planning:**
- **No per-strategy series on disk.** The 9/28 runs saved portfolio returns only, so Step 1 needs a new run.
- **The signal books carry no costs.** `get_all_signal_returns` is built from analyst output (per-strategy cross-sectional z-score, demeaned, × confidence, unit gross).
- **Strategies can run in isolation.** They are pure `generate(pit_view)` with no portfolio state, so they can run in parallel.
- **The 25 stocks were picked in 2026.** Any stock-level result on them is survivorship-biased and provisional.
- **Free data is enough for every Step 2 family.** No paid-data decision is needed:
  - Tiingo: ETF total returns, VXX from 2009, SVXY from 2011, BTC daily from 2015;
  - CBOE: VIX from 1990, VIX3M from 2009, put-write index PUT from 1991;
  - FRED: T-bill rates;
  - SEC EDGAR: every Form 4 insider filing, delisted companies included.

## 2. Why v1 was weak, and the idea behind v2

v1 tested **small, heavily-traded effects with few independent bets**: monthly
switches on a handful of ETFs. Published trend-following edges over 60/40 are
around 0–0.2 Sharpe. Twenty years of data can reliably detect a gap of about
0.45 or more. So v1's most likely verdict was "can't tell", whatever the truth.

v2 changes **where we look**. Every candidate family has to meet three
conditions:

1. **A structural reason for the return to persist.** Someone is paying for it
   and keeps paying, for example hedgers buying insurance or forced calendar flows.
   Behavioural effects count only if the literature shows them surviving after
   publication.
2. **An effect large enough to detect** with the history available, checked by
   a power calculation before any run (§5.3).
3. **Different from what we have.** None of the families is a variant of the
   current 11 strategies.

It also adds a **combination of uncorrelated return sources** as the headline
candidate. Several weak but independent edges add up to a larger, more
detectable one. This is the best-documented way to beat a single-asset benchmark
on risk-adjusted terms.

## 3. Order of work

1. You review v2 and answer §9.
2. Write the pre-registration and power analysis, fingerprint them, run the full suites green, commit.
3. **Step 2, the families in §5.** Seconds to minutes of CPU, fine to run today.
4. **Step 1, the current strategies' standalone edge (§4).** Overnight, after 23:00 IDT.
5. An independent agent recomputes every headline number with its own code.
6. Verdict doc, plus a live diff proposal for anything that passes.
7. **Proposals needing your go-ahead:** Step 3 paid data for the event-driven families (§6), and a live-only LLM forward test (§7).

## 4. Step 1: does any current strategy have standalone OOS edge?

This is kept from v1, and it's cheap to settle. A pass is still provisional
because of the universe bias.

**Series:** the signal-book daily return of each of the 11 enabled strategies,
2020-01-01 → 2026-06-30, live config.

**Harness:**
- Analysts and strategies run on their own, in a git worktree with a scratch `FIRM_DATA_DIR`.
- They must match the full pipeline's `get_all_signal_returns` to 1e-12 on a 1-month window.
- If they don't, fall back to one full-pipeline run (~10.4 h, overnight).

**Adjustments:**
- net of turnover × the corrected cost model from `config/live.yaml` (`c967ba4`);
- beta to SPY estimated on each fold's **train** window only, then applied unchanged to the test window.

**Windows:** the same 4 OOS windows as 9/28 (491 days) are primary. The full period is secondary (decision D4).

**Bars. A strategy "survives" only if it passes every bar:**

| Bar | Rule |
|---|---|
| S1 | Hedged net Sharpe > 0. The one-sided lower bound of a paired stationary block bootstrap (21-day blocks, 5000 draws) must be > 0 at α = 0.05/11. |
| S2 | Real Sharpe above the 95th percentile of 500 placebos. Each placebo reassigns the weights to random symbols, redrawn monthly, which keeps exposure and turnover and breaks the stock selection. |
| S3 | DSR > 0.95 on a new ledger, `docs/standalone_strategy_trial_history.json`, with 11 trials. The existing ledgers keep their counts: `combination_trial_history.json` (52) and `pattern_ml_trial_history.json` (104). |
| S4 | Hedged net Sharpe > 0 in at least 3 of 4 windows. |

**Checked before the run:** the git history of `live.yaml` `strategy_params`.
If they were tuned on 2020–2026, the windows are not really out of sample, and
the prereg will say so.

**A survivor** goes into the combination (§5, M-candidates) as a sleeve for
reporting only. It is also flagged for a retest on survivorship-free data (§6).

## 5. Step 2: candidate families (free data)

**Rules shared by every candidate:**
- The signal uses the close on day t. Trading happens at the close on day t+1.
- Cash earns the T-bill rate. Gross exposure is at most 1.0 (no leverage).
- Trading costs are 5 bps per side. Fund expense ratios are charged where the live implementation would pay them.
- **Parameters come from the source papers and are never tuned here.** Each candidate's post-publication period gets its own bar (A4).

### 5.1 Candidates

| ID | Family | Why it should persist | Rule (fixed) | Test window | Live instrument |
|---|---|---|---|---|---|
| **V1** | Volatility risk premium, timed | Hedgers overpay for crash insurance. Implied vol has exceeded realised vol about 85% of months since 1990. | Short VIX-futures exposure only in **contango** (VIX < VIX3M at close t), flat otherwise. Size = 10% vol target on trailing 21-day vol, capped at 0.3 of NAV. Return = −k × VXX daily return, minus SVXY's 0.95%/yr fee. (Simon & Campasano 2014; term-structure filter) | 2009-10 → 2026-09 (includes Feb-2018 and Mar-2020) | Long **SVXY** (−0.5×) at 2k, so no shorting is needed |
| **V2** | Volatility risk premium, untimed, long history | Same premium, but 35 years of data and the slowest, safest form | CBOE **PUT** index (fully collateralised monthly ATM SPX put-writing), held throughout | 1991-03 → 2026-09 | Sell 1-month ATM SPY puts, cash-secured (needs options permission) |
| **K1** | Calendar / forced flows | Month-end pension and payroll inflows (Etula et al. 2020); pre-FOMC drift (Lucca & Moench 2015) | Hold SPY from the last trading day of the month through the 3rd trading day of the next, plus the day before each scheduled FOMC announcement. Cash otherwise. | 1994-01 → 2026-09 (FOMC leg: post-publication 2015+ reported separately) | SPY, about 30 trades a year |
| **C1** | Crypto trend | Trends in crypto are strong and mostly retail-driven (Liu & Tsyvinski 2021, time-series momentum at 1–4 week horizons) | Long BTC if its trailing 4-week return > 0, else cash. Vol-targeted, reviewed weekly. Measured **against BTC buy-and-hold at equal average weight**, since the question is whether trend timing adds value. | 2015-02 → 2026-09 (post-publication 2021+) | BTC/USD spot on Alpaca |
| **T2** | Cross-asset trend (from v1) | Slow-moving capital; delivered crisis gains in 2008 and 2022 | SPY/EFA/EEM/IEF/TLT/GLD/DBC/VNQ, each held when its 12-month return beats T-bills. Inverse 63-day-vol weights, 10% portfolio vol target, long/flat, monthly. (Moskowitz, Ooi & Pedersen 2012) | 2007-02 → 2026-09 | ETFs |
| **M1** | **Combination (headline)** | Diversifying low-correlated positive streams raises Sharpe more than any one stream. Short vol and trend have opposite crisis behaviour. | Equal risk contribution across SPY, T2, V1 and K1, using trailing 63-day vol only, rebalanced monthly. Weights are not optimised on returns. | 2010-10 → 2026-09 | Combination of the above |
| **M2** | Combination + crypto | As M1 | M1 plus C1 as a 5th stream | 2016-03 → 2026-09 | As above |

**Dropped from v1:** GTAA-5, which duplicates T2, and dual momentum, which has a
low prior because its post-publication decay is documented. Both are dropped to
keep the trial count honest.

**Excluded:** cross-sectional stock strategies on the 25 names, because of the
hindsight-picked universe. They move to §6.

### 5.2 Benchmarks (reported on each candidate's own window)

- **BM1:** SPY buy-and-hold.
- **BM2:** 60/40 SPY/IEF, rebalanced monthly.
- **BM3:** SPY vol-targeted: weight = min(1, target / 21-day realised vol), traded when it moves more than 0.10. Target 12% (decision D3).

C1 is also compared with BTC buy-and-hold at equal average exposure.

### 5.3 Power analysis (run and frozen before any candidate)

For each candidate's window, a simulation answers one question: **what Sharpe
gap over each benchmark would A1 detect 80% of the time?**

- Method: bootstrap the benchmark's real returns, inject known gaps, and use the correlation to the benchmark each family's literature implies.
- The result goes into the prereg next to the bars, so every verdict can be read in context.
- **If a family can't possibly detect the edge its own literature claims**, it's reported as "not testable here". That's more honest than running it and reporting a meaningless FAIL.

### 5.4 Outcome tiers (pre-registered; every candidate lands in exactly one)

| Tier | Meaning | Rule | Action |
|---|---|---|---|
| **A: Proven better** | Beats every benchmark with statistical confidence | All of A1–A7 below pass | Live proposal (§10) |
| **B: Not worse, measurably safer** | Same return per unit of risk, much smaller crashes | (a) Non-inferiority: the Sharpe gap's one-sided 95% lower bound > −0.10 against each benchmark. (b) Max drawdown ≤ 0.75× the benchmark's. (c) Average loss over the 3 worst benchmark drawdowns ≤ 0.5× the benchmark's. (d) A3, A4 and A5 pass. | Live proposal, clearly labelled "risk improvement, not proven alpha". **You decide.** |
| **C: Inconclusive** | Neither | — | Not deployed. The benchmark (D1) is the recommendation for that slot. |
| **D: Worse** | Measurably worse | The upper bound of the Sharpe gap against any benchmark is < 0 | Rejected, reported |

**Tier A bars. Every one must pass against every benchmark:**

| Bar | Rule |
|---|---|
| A1 | Net Sharpe gap > 0. The one-sided lower bound of the paired stationary block bootstrap (63-day blocks, 5000 draws) must be > 0 at α = 0.05 / number of candidates. Within a candidate this is an intersection-union test over the 3 benchmarks. |
| A2 | DSR > 0.95. Trials are counted cumulatively on **one** new ledger for this search, `docs/alt_premia_trial_history.json`. Initial count = 7 candidates + 3 benchmarks. Using one ledger for the whole search is conservative. |
| A3 | Real Sharpe above the 95th percentile of 500 **random-timing placebos**: on/off states shuffled in blocks, so average exposure and trade count stay the same. Applies to the timed candidates V1, K1, C1 and T2. For V2, M1 and M2 the comparison is against SPY at matched vol. |
| A4 | A1's point estimate > 0 in the post-publication period alone. |
| A5 | A1's point estimate still > 0 at double costs (10 bps). |
| A6 | Max drawdown ≤ the benchmark's. |
| A7 | CSCV PBO < 0.50 across all candidates and benchmarks, 8 partitions. |

**Tail-risk bar for the short-vol candidates V1 and M1-with-V1, added to both
tiers A and B:** the position must lose less than 25% of NAV in a single day
under a **2× repeat of 5 Feb 2018**, when VXX rose about 100% in one day.

## 6. Step 3 proposal: event-driven stocks (needs your spend decision)

**Why this is the strongest "different" idea:**
- It gives thousands of **independent bets per year**, where the ETF families give tens.
- That much more data lets it detect edges of a size ETF strategies never could, within a few years.
- The event data is **free and includes delisted companies**, so it has no survivorship bias.

**Candidate hypotheses:**
- **Insider purchase clusters** from SEC Form 4: several executives buying the same stock on the open market (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012, "opportunistic" insiders).
- **Post-earnings drift** (Bernard & Thomas 1989), where earnings-surprise data from our FMP tier allows it.

**What's missing:** daily prices for delisted stocks, since returns can't be
computed without them. That is the paid part.

**Before any spend** I'll bring you:
- vendor quotes, e.g. Norgate, Sharadar via Nasdaq Data Link, EODHD, Tiingo paid tier;
- which ones cover delisted names;
- the exact hypothesis and bars each would test.

Nothing is bought without your yes.

The FactSet connector isn't authorized in this session. You can authorize it in
claude.ai connector settings if you want it considered as a source.

## 7. Parallel track proposal: live-only forward test of the LLM layer (needs your go-ahead)

**Why it can't be backtested:** the models were trained on data that includes
the "future". Any backtest of an LLM news or filing signal leaks, so historical
tests of this system's LLM machinery can't be trusted.

**The honest test is forward-only**, and the paper accounts already exist for exactly that:
- Pre-register one simple LLM signal: next-day direction from overnight 8-K filings and headlines on the liquid universe.
- Fix its bar and duration, e.g. 6 months with a pre-computed minimum detectable effect.
- Run it in a **separate** paper account, or as a zero-capital shadow book that logs what it would have traded, if a third account isn't available.

**Caveat:** it is slow and statistically weak. But it's the only route by which
the system's most distinctive component could show real edge.

## 8. Threats and exactly how each is handled

| Threat | What goes wrong | Handling | Type |
|---|---|---|---|
| Look-ahead | A day-t weight uses a later price | Weights come from closes up to t and returns start at t+1. **Mechanical check:** randomly change every price after day t and assert that the day-t weights don't move, for every candidate. | Prevent + detect |
| Warmup leak | Metrics include days before a signal had enough history | Warmup days are dropped from every metric. The recompute agent checks each series' first valid date. | Prevent + detect |
| Costs understated | It only works if trading is nearly free | 5 bps per side, far above SPY's ~0.3 bp spread, and it must survive 10 bps (A5). Fund fees and a crypto spread of about 10 bps per side are charged. | Prevent |
| **Instrument-history traps** | VXX was relaunched in Jan 2019. SVXY went from −1× to −0.5× on 2018-02-27. | VXX daily returns are checked for continuity around 2019-01, and V1's return is cross-checked against SVXY's own daily returns (×1 before 2018-02-27, ×2 after). A day that differs by more than 50 bps is investigated before any run. | Detect |
| Short-vol tail under-sampled | 17 years may not contain the worst possible day | The 2× Feb-2018 stress bar (§5.4), plus a hard size cap of 0.3 NAV | Prevent |
| Suspicious positive | A pass comes from a bug, not an edge | Every pass is re-run with a 2-day lag, broken down year by year, and recomputed independently | Detect |
| Hindsight in instrument choice | Picking assets already known to have won, like BTC (crypto's survivor) | Instruments are fixed from the papers. C1 is judged against *BTC buy-and-hold*, so what's tested is the timing, not having picked BTC. Post-publication windows get their own bar. | Reduce, and disclose what remains |
| Survivorship (Step 1) | The 25 stocks were chosen because they won | Can't be fixed with current data. Results are labelled provisional, and the Step 3 data is what fixes it. | Disclose |
| Many families searched | Testing enough ideas makes one look good by chance | Bonferroni within A1, DSR on one cumulative ledger (A2), PBO across everything (A7) | Prevent |
| Backtest ≠ live cadence | Live runs about 7 cycles a day; the backtest runs 1 | Every candidate trades at most daily from completed bars, identically in backtest and live (§10) | Prevent |

## 9. Artifacts to be created

**Pre-registration:**
- `scripts/standalone_strategy_preregistered_bars.py`
- `scripts/alt_premia_preregistered_bars.py`

Both follow the `combination_preregistered_bars.py` pattern: fingerprint
embedded in every run's meta, and the power-analysis output frozen alongside.

**Harnesses:**
- `scripts/run_standalone_strategy_evaluation.py`
- `scripts/run_alt_premia_evaluation.py`

The data loaders for CBOE, Tiingo and FRED cache everything to the scratchpad,
so runs are reproducible offline.

**Ledgers:** the two new ones from §4 and §5.4. The old ledgers are untouched.

**Verdict:** `docs/edge_search_verdict_2026_09.md`.

**Upkeep:** `PROJECT_CONTEXT.md`, `.cursor/rules/`, memory and the
`docs/claude-memory/` mirror, plus Discord updates at each milestone.

**Operating rules:** worktree only, services untouched, `config/live*.yaml`
untouched, suites green before every commit.

## 10. Decisions needed from you before the freeze

- **D1: the benchmark to recommend for "inconclusive" slots.** My default: BM3 (vol-targeted SPY) if you want drawdowns capped near 20%, otherwise BM2 (60/40).
- **D2: T2 long/flat, no leverage.** This is my default.
- **D3: BM3 target 12%.** This is my default. A 15–16% target would make BM3 a purer vol-timing test.
- **D4: Step 1 primary window.** My default is the 4 windows, as your brief asked, with the full period reported beside them.
- **D5: accept Tier B as deployable?** The alternative is to treat it as inconclusive.
- **D6: options and crypto.** Are options trading (for V2) and crypto (for C1 and M2) acceptable on the paper accounts? If not, I drop those candidates before the freeze, which also lowers the trial count.
- **D7: approve §6 (vendor quotes for Step 3) and/or §7 (LLM forward test) as follow-ups?**

## 11. If something reaches Tier A or B: shape of the live proposal (not applied)

- A new `strategy_mode: allocation` computes targets from completed daily bars **once per ET trading day** (first cycle). It trades only on each candidate's own schedule, reusing `rebalance_band_pct` and `_day_anchor`.
- It runs first on one instance, with the other kept as the control.
- It needs broker support for the instrument: SVXY and ETFs work on both brokers; BTC is Alpaca only; SPY options need broker permission and new order types.
- Delivered as an exact `config/live_*.yaml` diff with the measurement attached, for your sign-off. **No config change without it.**
