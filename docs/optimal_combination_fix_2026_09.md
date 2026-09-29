# `signal_combination: optimal` — real defects, fix, and pre-registered evaluation (2026-09-28)

**Status:** defects confirmed; fix shipped **off by default**
(`signal_combination.estimator: legacy` is unchanged and remains the default);
pre-registered evaluation **pending** (results section below is filled in only
after the run). `config/live.yaml` / `config/live_alpaca.yaml` untouched.

## 1. What `docs/ensemble_redundancy_audit_2026_09.md` §2 got wrong

D1 reported `optimal` parking 54.3% of weight on `sentiment` and attributed it to
`sentiment`'s short history. That number came from D1's own proxy:
`pd.DataFrame(...).fillna(0.0)` over the full history. **The production path does
not fill.** `combine_signals_optimal` builds its per-symbol frame with
`pd.DataFrame(hist).dropna(how="all")`, and `PerformanceAttribution` series begin
only at a strategy's first attributed fill and are never truncated. The real
behaviour for a late-starting strategy is the opposite of D1's:

1. Its column has leading NaNs.
2. `np.corrcoef` gives it an all-NaN row.
3. `nan_to_num` zeroes that row, diagonal included.
4. `pinv` assigns it **exactly 0 weight, permanently**, since history is never truncated.

Reproduced on synthetic data: the late starter gets 0.0 on the legacy path,
0.964 under D1's `fillna(0)` proxy, and 0.907 on a common window. D1's proposed
fix (a minimum-history floor) therefore targets a failure mode that does not
occur in production.

## 2. Confirmed defects

**Live IBKR (blended).** Read-only probe of `data/live_state.db`, last
attribution 2026-09-25, weights recomputed with the production function for a
bucket containing all 11 enabled strategies:

| strategy | production weight | note |
|---|---|---|
| seasonality | 0.469 | |
| stat_arb | 0.225 | |
| event_driven | 0.105 | |
| multi_factor | 0.102 | |
| regime_hmm / mean_reversion / trend / sentiment | 0.022–0.031 | |
| **momentum** | **0.000** | first attributed fill 8/11 vs peers 7/30, so NaN lock-out |
| **pattern_recognition**, **volatility_breakout** | **0.000** | no attribution series at all on IBKR |

`effective_n` = 3.4, estimated from 66 points of which 41 are nonzero.

**Live Alpaca (sleeved)** is effectively unaffected: `_step_sleeved` partitions the
blackboard per strategy, so each bucket holds one strategy and `optimal` falls
back to confidence.

**Backtest.** Instrumented `optimal_signal_weights` over a live-config 2024-Q1
backtest (2,450 calls):

- The NaN lock-out hits only `pattern_recognition` (zero on 100% of calls).
- Negative-weight clipping after an unshrunk `pinv` zeroes others far more often:
  `mean_reversion` 89% of calls, `regime_hmm` 50%, `momentum` 40%.

**Structural issues:**

- **No lookback.** History is unbounded.
- **No cadence normalisation.** Live appends one point per cycle (~7/day), the
  backtest one per day.
- **Inputs are shares of the blended book's P&L, not standalone strategy returns.**
  `TraderAgent._attribute_to_strategies` splits each traded position across its
  signalling strategies by |score|. Strategies co-holding the same symbols
  therefore look correlated because they share positions, not because their
  signals agree. That is exactly what an inverse covariance then punishes.
- **Earlier feedback-loop hypothesis: partly refuted.** Attribution shares follow
  |score|, not the `optimal` weight, so a strategy's weight does not directly feed
  its own series. A locked-out strategy is still *credited* P&L for positions it
  had no say in.

**Consequence for past results.** Every prior backtest of `optimal`, including
`docs/portfolio_construction_diagnosis.md`'s "optimal beats confidence 3/3" (the
basis for making it the live default), ran with these defects. Those results are
real measurements of that code, not of inverse-covariance combination in general.

## 3. The fix (opt-in)

- `signal_combination.estimator: robust` → `combine_signals_optimal_robust` /
  `robust_optimal_signal_weights` (`src/firm/agents/analysts/__init__.py`):
  - compound to daily first;
  - trailing `lookback_days` (126);
  - a strategy with fewer than `min_obs` (20) observations, or zero variance, is
    *immature* and gets the neutral equal share `1/n`, never 0;
  - mature strategies share the rest via a Ledoit-Wolf-shrunk `Σ⁻¹·1` over their
    common rows;
  - negatives are still clipped, so there are no sign flips.
- `returns_source: attribution | standalone`. `standalone` uses the new
  `PerformanceAttribution.record_signals` / `get_all_signal_returns`: each
  strategy's own unit-gross `score×confidence` book, marked to market,
  independent of what the blended book traded. Every strategy gets a stream from
  its first signal. It is persisted in `export_state`, with backward-compatible
  restore.
- Research control `method: random_weights`: per-strategy Exp(1) weights,
  redrawn monthly and seeded, so it is reproducible. It is never a live setting.
- Tests: `tests/test_optimal_combination_robust.py`, 22 tests. They include a
  regression test pinning the legacy defect.

Live behaviour is unchanged until `estimator: robust` is set. Signal books are
recorded on every live cycle regardless, which has no trading effect.

## 4. Pre-registration (frozen before any candidate ran)

`scripts/combination_preregistered_bars.py`, fingerprint
`96bf26c8aa81b57db21adb8eba6dc3f7afd7dece24857d0195bb25aa8da5e9a2`, written
(`PREREGISTERED_AT` 04:40 UTC) before the runs launched at 04:46 UTC.

**Disclosure:** it was *committed* only after the runs, because the full test
suite (a hard pre-commit gate) could not finish under run contention. Evidence
that nothing moved:
- Every run's meta embeds the fingerprint.
- Launch-time copies of the prereg, harness and ledger, plus a SHA-256 of the
  source diff, were saved beside the run outputs (`runs/launch_provenance.txt`
  in the session scratchpad).
- The committed files were checked `cmp`-identical to those copies before the
  evaluation ran.

**Design:**
- Candidates:
  - C0: legacy `optimal` (live today)
  - C1: robust estimator on attribution returns
  - C2: robust estimator on standalone returns
  - C3: `confidence`
  - P: `random_weights` placebo
- Primary hypotheses: C1 and C2 against C0.
- One continuous live-config backtest per candidate, 2020-01-01 → 2026-06-30.
- 4 walk-forward folds with OOS test windows sliced from the continuous runs, so
  attribution and signal history carry across folds as they do live.

**Bars (all must pass):**

| Bar | Rule |
|---|---|
| B1 | OOS Sharpe beats C0, and the 97.5% paired stationary-block-bootstrap CI (21-day mean block) excludes zero |
| B2 | Beats the placebo P, same CI rule |
| B3 | DSR > 0.95, with trials = 5 + **47 prior combination-layer trials** (`docs/combination_trial_history.json`, full grid sizes counted conservatively) |
| B4 | Beats C0 in ≥3 of 4 folds |
| B5 | CSCV PBO < 0.50 across all 5 candidates |
| B6 | OOS max drawdown ≤ 1.25× C0's |

**Choices, and the case against each:**
- **Why a block bootstrap instead of the symbol-block bootstrap used for pattern-ML:**
  that test had per-symbol trade events to block on. This one compares
  portfolio-level daily returns, where the dependence is serial.
- **Why a monthly, per-strategy placebo:** a per-day, per-symbol random draw would
  lose to anything on turnover alone. That would make B2 easy to pass for the
  wrong reason.
- **Continuous runs instead of per-fold resets:** closer to live and 4× cheaper.
  The cost is that folds share state, so they are not independent samples. The
  bootstrap over the concatenated OOS days is the primary inference; B4 is only a
  consistency check.
- **Known weakness:** `sentiment` has cache data only from 2025-11-17 and
  `pattern_recognition` signals sparsely. Both are immature for much of the
  window, which C1/C2 handle by design, while C0 locks them out.
- **Power (noted before results, from a synthetic dry run of the evaluator):**
  there are ~508 OOS days, about 2 years across the 4 test windows. Between
  *independent* series, an annualised Sharpe gap of +0.7 does not clear B1's
  97.5% CI. Real candidates share most positions, so the paired SE is smaller,
  but a B1 FAIL still reads as "improvement not demonstrated", not "shown to be
  no better".
- **What a PASS would not prove:** it would not justify any live change by
  itself. That still needs the user's sign-off on the exact config diff.

## 5. Results — **FAIL** (both primary hypotheses)

Real run: 5 continuous live-config backtests, 2020-01-01 → 2026-06-30 (1,631
trading days each, ~10.4 h of CPU each, completed 2026-09-28 ~15:16 UTC). Every
run's meta carries fingerprint `96bf26c8…`. The prereg, harness and ledger were
byte-identical to their launch-time copies (checked with `cmp`) when the
evaluation ran. Report: `docs/combination_evaluation_2026_09.json`. The ledger
entry is appended to `docs/combination_trial_history.json` (now 52 trials).
**Independent re-verification: reproduced exactly.** A separate agent wrote its own numpy/pandas implementation (reusing only the library `cscv_pbo`/`deflated_sharpe`) and recomputed from the raw return parquets:
- fold windows, all 5 full-period and OOS Sharpes, all 20 per-fold Sharpes, PBO and both DSRs: all match to floating-point precision;
- its own circular block bootstrap: all primary CIs still straddle zero;
- sanity checks: configs match the frozen design, no NaNs, no warmup leak, pairwise return correlation 0.43–0.62 (so the candidates genuinely differ), and C1 = C2 only on the first 20 days, when every strategy is immature;
- Spearman ρ of first-half vs second-half candidate Sharpe = 0.00, consistent with PBO 0.97.

| | C0 legacy (live) | C1 robust/attribution | C2 robust/standalone | C3 confidence | P random placebo |
|---|---|---|---|---|---|
| **OOS Sharpe** (491 days, 4 test windows) | **−0.69** | −0.54 | −0.52 | −0.32 | **−0.23** |
| OOS total return | −5.9% | −5.3% | −4.9% | −4.9% | −2.2% |
| Full-period Sharpe (not a bar) | 0.07 | 0.42 | 0.60 | 0.57 | 0.49 |
| Full-period max DD | 12.5% | 12.2% | 9.8% | 9.1% | 10.4% |

Per-fold OOS Sharpe:

| test window | C0 | C1 | C2 | C3 | P |
|---|---|---|---|---|---|
| 2021-02-20 → 2021-08-16 | −0.54 | −1.58 | −0.49 | −0.17 | −1.71 |
| 2022-10-06 → 2023-04-01 | −2.73 | −2.29 | −2.77 | −1.66 | −2.91 |
| 2024-05-21 → 2024-11-14 | 0.26 | −0.68 | −0.74 | −0.74 | 1.44 |
| 2026-01-04 → 2026-06-30 | 1.93 | 2.69 | 2.40 | 1.55 | 2.81 |

Bars:

| Bar | C1 | C2 |
|---|---|---|
| B1 beats C0 (97.5% CI) | FAIL: +0.15, CI [−1.02, +1.22] | FAIL: +0.17, CI [−0.98, +1.20] |
| B2 beats placebo | FAIL: −0.30, CI [−1.52, +0.93] | FAIL: −0.29, CI [−1.46, +0.80] |
| B3 DSR > 0.95 (5 + 47 prior trials) | FAIL: 0.089 | FAIL: 0.094 |
| B4 ≥3/4 folds beat C0 | FAIL: 2/4 | FAIL: 2/4 |
| B5 PBO < 0.50 | FAIL: **0.971** | FAIL: 0.971 |
| B6 max DD ≤ 1.25× C0 | pass | pass |

Secondary comparisons, all with CIs straddling zero: C0 − C3 = −0.37, C0 − P = −0.46, C3 − P = −0.09.

### Reading it honestly

- **Every combination method loses out of sample, and uninformed random weights did
  best.** No candidate is statistically distinguishable from any other,
  including the placebo: every paired CI spans about ±1.1 Sharpe. PBO 0.971 means
  the in-sample ranking of these methods almost perfectly inverts out of
  sample. **On this system's current strategy roster, *how* signals are combined
  is not where the edge is.**
- **The robust estimator is a correctness fix, not a profitability lever.** The
  lock-out is real, and live IBKR really does ignore 3 of 11 strategies. But
  un-ignoring them via C1/C2 moved OOS Sharpe by an undetectable +0.15/+0.17.
- **The 2026-07-26 basis for `optimal` as the live default does not replicate.**
  Under the full live config, over 4 walk-forward windows, legacy `optimal` has the
  *lowest* point estimate both OOS and full-period (it lost to `confidence` by
  0.37 OOS and 0.49 full-period). Not significant either, so this refutes the
  claimed improvement without proving the opposite.
- **Why full-period looks positive while OOS is negative:** the 30% test slices
  happen to include the 2021 H1 rotation, the late-2022 bear market and mid-2024.
  No parameter was fitted on the train windows, so the full-period Sharpes
  (0.07–0.60) are not in-sample-inflated. But the pre-registered bars are OOS-only,
  and the full-period figures are context, not evidence.

### Recommendation

**No `config/live.yaml` / `config/live_alpaca.yaml` change on this evidence.**
`estimator: robust` stays shipped and off. It is the correct estimator if anyone
ever relies on `optimal` again, but there is no measured reason to switch now.
Switching live from `optimal` to `confidence` is also *not* supported: it didn't
beat the placebo either. The higher-value next question is **per-strategy
standalone OOS edge**. The ensemble is flat-to-negative out of sample regardless
of how it is combined, so the lever is which signals carry edge at all, not how
they are weighted. The new standalone signal books
(`PerformanceAttribution.get_all_signal_returns`) now record exactly that
series, in both backtest and live.

## 6. WS1: live-vs-backtest cadence sweep (same session)

Method: 3 parallel read-only code sweeps (risk/execution, attribution consumers,
strategies/data shape). An independent verifier then checked every claim
against live logs, `execution_audit.jsonl` and `api.log`, not just the code. Only
claims that survived verification are listed as confirmed.

| # | Finding | Live status | Verified impact | Action |
|---|---|---|---|---|
| 1 | `MarketRegimeDetector.retrain_frequency` (`regime_overlay`, 5 = "refit weekly") counted `detect()` **calls**, not bars. IBKR: 1 call/cycle, so a refit at least daily. **Alpaca (sleeved): RiskAgent runs once per sleeve, 5–9 calls/cycle in `api.log`, so several refits within one cycle.** | enabled on both | Refits on identical data with a fixed seed are deterministic, so labels changed only at day boundaries (2 changes in 42 Alpaca cycles). Real effects: wasted HMM fits and refits roughly daily instead of the backtest-validated every 5 bars. | **Fixed** (`src/firm/regime/detector.py`: counts new bars). `tests/test_regime_detector_cadence.py`: old code fails the 2 intraday tests and passes the daily-cadence one, so backtest behaviour is proven unchanged. Takes effect at the next service restart. |
| 2 | `rebalance_fraction: 0.7` closes 70% of the above-band gap **per cycle**. Intraday targets come from the same completed-bar panel, so live closes ~97% within 3 cycles, where the backtest (1 cycle/day) validated a multi-*day* glide. No cycle-type gate skips execution intraday. | enabled on both | Mechanism confirmed. Same-symbol, same-direction re-trades within a day appear in Alpaca's audit log, but n=11 pairs is too small to size the effect. | **Fixed 2026-09-29 at the user's request** (`ExecutionAgent._day_anchor`). A symbol's first order of the trading day (US/Eastern) closes `fraction` of the gap exactly as before. Later same-day cycles trade only toward `anchor + fraction × (target − anchor)` and skip anything under the existing dust floor (`band × close_dust_fraction`, 1% NAV live). The effect: no repeat trading for a stable target, a retry if the first order didn't fill, and a damped response to a real intraday target change. Forced full closes are unaffected, and anchors are keyed per sleeve. `tests/test_rebalance_fraction_daily.py`: the 4 intraday cases fail on the old code, while the one-call-per-day, retry, forced-close, isolation and fraction-1.0 cases pass on both, so backtests are unchanged. Caveat: the anchors are in-memory, so a mid-day restart allows at most one extra fractional step. |
| 3 | IBKR's after-hours cycle (~17:00 ET) drops today's *completed* bar (`LiveDataFeed.refresh`'s date-only cutoff), so it trades on one-day-stale signals. | IBKR only | 1 order at 17:00 ET in the entire retained log (the rest fall inside the rebalance band). | Not changed. Low impact; documented. |
| 4 | `strategy_circuit_breaker`, `hrp`, and `joint_optimizer`'s `_book_nav_history` all consume per-cycle series with `sqrt(252)` annualisation. | all **disabled** | none today | Documented. Must be fixed before any of them is enabled. `hrp` could reuse `_daily_compound`. |
| 5 | **Test suite polluted the live IBKR audit log.** `execution_safety.audit_path()` defaults to `data/execution_audit.jsonl` and only `test_live_engine.py` redirected it. `data/execution_audit.jsonl` holds 8,099 `broker_type=""` fixture records plus 870 fake `alpaca_paper` ones (e.g. AAPL qty 1, cycle 1, absent from Alpaca's own file). This session's runs added 2. | — | Audit-trail integrity: real IBKR submissions are 571 of ~18.9k lines. | **Fixed**: autouse `_isolated_execution_audit` fixture in `tests/conftest.py`, verified (a 37-test API run added 0 lines). Existing contaminated lines left in place, since the log is append-only by design. They are identifiable by `broker_type` in `{"", "alpaca_paper"}` inside IBKR's file. |

Test-infrastructure fixes found on the way to a green suite:

- **`tests/test_api.py::TestRuns` timeouts: pre-existing, fixed.** `test_cost_overrides_*` and
  `test_compare` launched default-range (6-year) synthetic runs and never waited for them.
  `routers/runs._job_manager` is a process-wide singleton that serialises runs, so those
  jobs held its lock, and `test_report_after_completion` / `test_equity_after_completion`
  hit their 120 s limit. This reproduces with `test_api.py` alone on an idle machine.
  Fixed with short windows and waiting for completion; the timeout is untouched.
  `TestRuns` went from failing (and still running after 15 min on HEAD) to 13/13 in 5 min.
- **`tests/test_investing_calendar.py`: pre-existing since the 9/22 venv rebuild, fixed.** It
  needs `bs4` from the optional `investing` extra, which was not installed. Added the
  repo's standard `pytest.importorskip` guard and installed `beautifulsoup4`: 7/7 pass.
  No live impact: both instances use `news_guard.source: forexfactory`.
- **22 setup/teardown errors in the ML inference tests: caused by this session, fixed.** The
  first version of the conftest audit-isolation fixture requested `monkeypatch`. That
  reorders fixture teardown, so `reset_cache()` ran on a still-patched lambda. The
  fixture now uses `patch.dict(os.environ)`.

Checked and fine, so coverage is visible: `max_daily_trades`/`max_daily_turnover`
(bucketed by trading day), vol/ADV/correlation lookbacks (read daily bars via
`pit_view`), kill switch and drawdown breaker (NAV peak), `news_guard`
(wall-clock), `regime_hmm` strategy refit (calendar days), sleeve series and
capital gate (already compound to daily), reflection rollup
(`get_all_daily_strategy_returns`), forming-bar exclusion (all intraday cycles
see an identical completed-bar panel), and no strategy reads wall-clock time.

## 7. Incident during this session (caused by this work, contained)

At 2026-09-28 12:06 UTC, IBKR premarket **cycle 109 failed** with
`ImportError: cannot import name 'combine_signals_optimal_robust'`. Both
`firm-api` services import from the repo checkout itself. The running process
(started Sunday 03:28 UTC) held the old `firm.agents.analysts` in memory, then
lazily imported this session's *uncommitted*, edited `research/_combine.py`
(`bull.py` imports it inside `run()`, and this was the first cycle since that
restart). The result was a mixed-version process. No orders were sent (paper only), and it was the only
failed cycle. Alpaca had not run a cycle yet.

**Containment:**
1. `git stash -u` restored the checkout to HEAD. A failed import is not cached, so
   the next cycle would retry cleanly.
2. A fresh interpreter was verified to import cleanly from HEAD.
3. At the user's instruction, both services were then stopped for the rest of
   the work session, and the work was restored into the checkout.

**Standing rule** (saved to agent memory, `feedback_never_edit_live_checkout`):
never modify tracked `src/` in the live checkout while a service is active.
Use a git worktree, or stop the services.
