# Ensemble redundancy audit (2026-09-27)

**Status:** Complete. Part D1 of the approved research plan investigating
whether the 13-(now 11-)strategy ensemble can be safely simplified.
**Scope:** correlation/redundancy audit across the real currently-enabled
strategy roster, an empirical check of `signal_combination: optimal`
(inverse-covariance weighting), and a walk-forward/PBO comparison of the
full roster vs. a leaner subset using the existing
`scripts/run_walk_forward_pbo_audit.py` harness.

**Do not confuse this with the prior "Phase 2" correlation analysis**
referenced in `docs/formal_pbo_audit.md` / `docs/remediation_progress.md`
#65 (momentum↔event_driven = 0.74, momentum/trend/multi_factor/regime_hmm
mean off-diagonal = −0.151). That analysis used the older 10-strategy
roster (no `pattern_recognition`, still included the now-decommissioned
Danelfin strategies at various points across this repo's history) over an
unrecorded/unknown date window (the script itself is gone, only the prose
survives). This audit **reruns the correlation measurement from scratch**
against the real, current 11-strategy live roster, over a fresh
(2024-01-01→2026-06-30) window, and — as instructed — treats that prior
0.74 finding as a hypothesis to confirm, not a given. **It is not
confirmed** (see below); the two analyses disagree, most likely because
they cover different strategy rosters and different, non-overlapping (or
only partially overlapping) date ranges. Correlation structure between two
strategies is not a fixed property — it is regime- and roster-dependent,
and this audit's own single 2.5-year window should be read with the same
caveat.

## Method

1. Built a scratch script (`/tmp/.../run_correlation_audit.py`, not
   checked in) that monkeypatches `firm.backtest.run.BacktestEngine` to
   capture the underlying `BacktestEngine` instance that the public
   `execute_backtest()` wrapper normally discards, so the
   `PerformanceAttribution` object's raw per-strategy daily return series
   (`get_all_strategy_returns()`) is reachable after the run — this data is
   **not** persisted to any `runs/<id>/` artifact (`report.json` only has
   collapsed per-strategy metrics; the raw daily series is discarded once
   the backtest process exits).
2. Ran one full backtest with the real current live roster (11 strategies
   from `config/live.yaml` `strategies.enabled`: `momentum`, `trend`,
   `mean_reversion`, `stat_arb`, `multi_factor`, `sentiment`,
   `event_driven`, `volatility_breakout`, `seasonality`, `regime_hmm`,
   `pattern_recognition`), the real 25-symbol live universe, `data_source:
   cache`, `signal_combination: optimal`, `allocation_method:
   conviction_weighted` (both matching the current live default), daily
   rebalance, over **2024-01-01 → 2026-06-30** (624 trading days).
3. **Scoping tradeoff, stated explicitly:** this system's own 2020-2026
   cache-backed audits (`docs/formal_pbo_audit.md`) typically run the full
   6.5-year range. This audit deliberately used a shorter, more recent
   ~2.5-year window instead. Reason: this specific 11-strategy/25-symbol
   config, run daily-rebalance with `pattern_recognition`'s CNN scoring
   and `stat_arb`'s per-pair cointegration tests, costs roughly
   **3.1-3.75 seconds of wall-clock per simulated trading day** on this
   host (measured directly: an earlier attempt at the full 2020-2026 range
   was killed by its own 1100s timeout at 356/1638 days; the 624-day
   2024-2026 run actually completed in ~39 minutes = 3.75s/day). A full
   2020-2026 walk-forward grid (2 candidates × 3 folds, each fold training
   both candidates and testing the winner) at that rate would run
   3+ hours — judged not worth the wall-clock cost for what is fundamentally
   a scoping/triage audit, not a final gate. The shorter window still spans
   a real bull run (2024), a choppy 2025, and includes the only period
   `sentiment`'s cache has any real data (from 2025-11-17) and the only
   period `pattern_recognition` was designed for. It excludes the 2020
   COVID crash and 2022 bear market that feature heavily in the longer
   audits — a real limitation, disclosed here rather than glossed over.
4. Computed the correlation matrix directly from the 11×624 return frame
   (`pd.DataFrame(...).fillna(0.0).corr()`, matching the "Phase 2"
   methodology's convention that a strategy-day with no recorded fill is a
   legitimate zero contribution, not missing data).
5. Called the actual production function, `optimal_signal_weights()`
   (`src/firm/agents/analysts/__init__.py:136`), on that same return frame
   to get a real weight vector — this is a **single, full-history**
   application of the same math the live/backtest engine calls per-symbol,
   per-cycle with a *trailing* window; it is a reasonable proxy for what
   the mechanism converges to given ample history, not a claim about any
   single day's live weights (which are also never persisted anywhere, so
   there was no way to pull a real historical live weight vector directly).
6. Reused `scripts/run_walk_forward_pbo_audit.py` unmodified, with a custom
   2-candidate `--param-grid-json`: candidate 0 = the full real 11-strategy
   roster, candidate 1 = a leaner subset (design rationale below), both
   otherwise identical (`signal_combination: optimal`,
   `allocation_method: conviction_weighted`) so only the strategy set
   differs — isolating the redundancy question from the
   already-separately-tested combination-method question
   (`docs/portfolio_construction_diagnosis.md`). Same 2024-01-01→2026-06-30
   window, 3 folds, 70/30 train/test, `selection_metric: sharpe_ratio`
   (matching the very first formal PBO audit's fold count).

## 1. Correlation matrix (real, 2024-01-01 → 2026-06-30, 624 trading days)

All 11 currently-live strategies produced usable attribution history.
Full matrix (Pearson correlation of daily attributed returns):

| | momentum | trend | mean_rev | stat_arb | multi_factor | sentiment | event_driven | vol_breakout | seasonality | regime_hmm | pattern_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **momentum** | 1.00 | -0.19 | -0.13 | 0.13 | -0.26 | -0.04 | **-0.44** | -0.00 | -0.31 | -0.33 | 0.04 |
| **trend** | -0.19 | 1.00 | 0.06 | -0.05 | 0.38 | 0.09 | -0.23 | **-0.56** | 0.05 | -0.19 | -0.19 |
| **mean_reversion** | -0.13 | 0.06 | 1.00 | 0.16 | -0.32 | 0.00 | **-0.54** | 0.40 | -0.04 | -0.11 | 0.37 |
| **stat_arb** | 0.13 | -0.05 | 0.16 | 1.00 | -0.42 | -0.41 | -0.14 | 0.08 | -0.48 | -0.28 | 0.37 |
| **multi_factor** | -0.26 | 0.38 | -0.32 | -0.42 | 1.00 | 0.23 | 0.04 | -0.18 | **0.72** | 0.21 | **-0.62** |
| **sentiment** | -0.04 | 0.09 | 0.00 | -0.41 | 0.23 | 1.00 | -0.07 | 0.05 | 0.26 | 0.20 | -0.32 |
| **event_driven** | **-0.44** | -0.23 | **-0.54** | -0.14 | 0.04 | -0.07 | 1.00 | -0.28 | -0.17 | -0.07 | -0.19 |
| **volatility_breakout** | -0.00 | **-0.56** | 0.40 | 0.08 | -0.18 | 0.05 | -0.28 | 1.00 | 0.26 | 0.27 | 0.18 |
| **seasonality** | -0.31 | 0.05 | -0.04 | -0.48 | **0.72** | 0.26 | -0.17 | 0.26 | 1.00 | **0.53** | -0.43 |
| **regime_hmm** | -0.33 | -0.19 | -0.11 | -0.28 | 0.21 | 0.20 | -0.07 | 0.27 | **0.53** | 1.00 | -0.21 |
| **pattern_recognition** | 0.04 | -0.19 | 0.37 | 0.37 | **-0.62** | -0.32 | -0.19 | 0.18 | -0.43 | -0.21 | 1.00 |

### Key finding: the prior 0.74 momentum↔event_driven claim is **not confirmed** in this fresh measurement

In this window/roster, `momentum`↔`event_driven` correlation is **-0.44** —
moderately *negative* (diversifying), the opposite sign of the earlier
claim. This is a real, direct disagreement with the prior "Phase 2"
finding, not a rounding difference. Most likely explanation: the earlier
analysis ran on a different (unknown) date range and a 10-strategy roster
that also may have still included Danelfin strategies at points in this
repo's history — `event_driven`'s reliance on a fundamentals-surprise
proxy and `momentum`'s reliance on price trend can plausibly co-move in
some regimes and diverge in others. **Practical conclusion: this specific
pairwise correlation is not stable enough across measurement windows to
be treated as a fixed structural fact about the system**, and neither
number should be used on its own to justify dropping either strategy.

### Real redundant cluster in this window: `multi_factor` / `seasonality` / `regime_hmm`

The strongest positive (same-direction, genuinely redundant) correlations
in the whole matrix are:

- `multi_factor` ↔ `seasonality`: **0.72** (the single highest off-diagonal
  correlation in the matrix)
- `seasonality` ↔ `regime_hmm`: **0.53**
- `multi_factor` ↔ `regime_hmm`: 0.21 (weaker, but same sign)

This is a real three-way cluster of same-direction bets, not a noise
artifact — 0.72 is a strong correlation for daily attributed strategy
returns in this system (compare to the -0.81 momentum↔trend correlation
the prior Phase-2 analysis found meaningfully diversifying in the opposite
direction). Their standalone performance over this window differs sharply,
which is what makes the redundancy actionable rather than merely
descriptive:

| Strategy | Sharpe | CAGR | Ann. vol |
|---|---|---|---|
| `multi_factor` | **0.80** | +3.96% | 5.0% |
| `seasonality` | 0.06 | +0.14% | 3.1% |
| `regime_hmm` | **-0.55** | -3.22% | 5.7% |

`multi_factor` is the best-performing member of the cluster by a wide
margin; `seasonality` is statistically indistinguishable from zero
(Sharpe 0.06 on 624 days is not a meaningful edge); `regime_hmm` is
negative. A strategy that is both (a) highly correlated with a materially
better-performing strategy and (b) weak-to-negative on its own is the
textbook case for "this isn't adding diversification, it's diluting the
better signal" — exactly the pattern `docs/portfolio_construction_diagnosis.md`
already flagged for the pre-fix `regime_hmm` (negative Sharpe in 6/6 of
its original test windows too, before the separation-metric fix — this
audit shows it's still negative in a fresh, independent window,
post-fix).

### Other notable pairs — mostly diversifying, not redundant

`trend`↔`volatility_breakout` (-0.56), `mean_reversion`↔`event_driven`
(-0.54), `multi_factor`↔`pattern_recognition` (-0.62), `stat_arb`↔`seasonality`
(-0.48), `stat_arb`↔`multi_factor` (-0.42), `stat_arb`↔`sentiment` (-0.41)
are all **strong but negative** — i.e., genuinely diversifying pairs, the
opposite of redundancy. These are not candidates for removal on redundancy
grounds; if anything they're evidence the ensemble's diversification
mechanism is working as intended for these pairs.

### Weakest standalone performers (not necessarily redundant, but candidates on Sharpe alone)

| Strategy | Sharpe | Note |
|---|---|---|
| `stat_arb` | **-0.93** | Worst standalone Sharpe in the roster this window |
| `mean_reversion` | -0.55 | Negative, but *not* redundant with anything (mostly negative correlations = diversifying) — a genuine diversifier having a bad window, not dead weight |
| `regime_hmm` | -0.55 | Negative **and** redundant with `seasonality`/`multi_factor` (see cluster above) — the stronger removal case of the two |
| `pattern_recognition` | -0.38 | Negative in this window; only 624 days of a strategy with a documented single-window validation history (`docs/pattern_recognition_plan.md`) — too early to treat this as a settled verdict either way |

## 2. Does `signal_combination: optimal` concentrate sensibly, or produce something degenerate?

**Empirical answer: it produces something degenerate in this window — a
near-single-strategy allocation dominated by an artifact of uneven data
history, not by genuine risk-adjusted edge.**

Real weight vector from `optimal_signal_weights()` on the full 624-day,
11-strategy return frame (L1-normalized, `effective_n = 3.07` — i.e.
despite 11 strategies, the combination behaves like roughly **3**
independent bets):

| Strategy | Weight | Standalone Sharpe | Ann. vol |
|---|---|---|---|
| **sentiment** | **54.3%** | 1.53 | **0.64%** |
| volatility_breakout | 12.1% | 0.20 | 0.98% |
| stat_arb | 9.2% | **-0.93** | 1.65% |
| seasonality | 4.3% | 0.06 | 3.11% |
| pattern_recognition | 4.1% | -0.38 | 4.17% |
| event_driven | 3.6% | 0.46 | 12.60% |
| momentum | 3.5% | 0.13 | 10.40% |
| trend | 2.8% | 0.13 | 8.87% |
| regime_hmm | 2.7% | -0.55 | 5.70% |
| mean_reversion | 2.1% | -0.55 | 8.95% |
| **multi_factor** | **1.4%** | **0.80** | 5.01% |

Two concrete problems, both directly visible in this table:

1. **The mechanism rewards low realized variance almost regardless of
   sign or quality of Sharpe.** `stat_arb` has the single worst standalone
   Sharpe in the entire roster (-0.93) yet receives the **third-largest**
   weight (9.2%) — because `w ∝ Σ⁻¹·1` is fundamentally an inverse-variance
   (min-variance) construction, not a Sharpe-maximizing one; a
   low-volatility loser and a low-volatility winner both look attractive
   to it if their correlation profile doesn't offset that. Meanwhile
   `multi_factor` — the single best standalone Sharpe in the whole roster
   (0.80) — receives the **smallest** weight of all 11 (1.4%), purely
   because its volatility (5.0%) is unremarkable relative to the two
   strategies ahead of it, not because its edge is weak.
2. **`sentiment`'s dominant 54.3% weight is very likely a data-coverage
   artifact, not genuine low-risk skill.** `sentiment`'s annualized
   volatility (0.64%) is **8-20× smaller** than every other strategy in
   the roster (next-lowest is `volatility_breakout` at 0.98%; most sit in
   the 3-13% range). `docs/formal_pbo_audit.md`'s own documented cache
   caveat explains why: the `combined/sentiment` panel only starts
   **2025-11-17**, so `sentiment` is a true, recorded zero-return
   contributor for roughly the first ~23 of this window's 30 months (no
   fill ever recorded, correctly treated as literal zero — not a bug, just
   very little real trading history within this window). A strategy that
   is mechanically flat for ~75% of the measurement period will have a
   tiny full-period variance almost by construction, and `Σ⁻¹` cannot
   distinguish "usually flat because it just started" from "usually flat
   because it is a genuinely low-risk strategy" — both look identical to
   the covariance matrix. `optimal`'s inverse-covariance weighting is
   **not equipped to handle heterogeneous strategy start dates within its
   lookback window**, and this is a real, mechanism-level finding, not
   specific to this one measurement.

**Caveat on generalizability:** this is a single full-history application
of the math, a proxy for (not identical to) the live/backtest per-cycle,
per-symbol trailing-window computation — the actual live weights evolve
day to day and would look somewhat different once `sentiment` has
accumulated more real trailing history post-2025-11-17 and less of the
zero-fill period sits inside the trailing lookback. But the underlying
mechanism-level problem (inverse-variance weighting cannot tell "genuinely
low-risk" from "recently started/thinly traded") is structural, not an
artifact of this one measurement window, and will recur for any future
strategy added with a shorter cache/data history than its peers.

## 3. Full ensemble vs. leaner subset — walk-forward + PBO

**Leaner subset design** (11 → 8 strategies), based directly on the
findings above: drop `seasonality` and `regime_hmm` (the redundant,
weak/negative members of the `multi_factor`/`seasonality`/`regime_hmm`
cluster — keeping `multi_factor`, the cluster's clear best performer) and
`stat_arb` (the single worst standalone Sharpe in the roster, -0.93, and
the strategy whose spurious inverse-variance weight in part 2 best
illustrates the mechanism's blind spot). `mean_reversion` was
deliberately **kept** despite its negative Sharpe (-0.55) — it is not
redundant with anything in this measurement (mostly negative
correlations, i.e. diversifying), so cutting it would be Sharpe-chasing on
a single window, not a redundancy-driven decision; distinguishing these
two cases was the explicit point of this audit.

Leaner subset: `momentum`, `trend`, `mean_reversion`, `multi_factor`,
`sentiment`, `event_driven`, `volatility_breakout`, `pattern_recognition`.

Both candidates run identically otherwise: `signal_combination: optimal`,
`allocation_method: conviction_weighted` (both matching current live
defaults), same 25-symbol universe, same 2024-01-01→2026-06-30 range, 3
folds, 70/30 train/test, `selection_metric: sharpe_ratio` — via
`scripts/run_walk_forward_pbo_audit.py --param-grid-json <2-candidate
grid>`, unmodified harness.

**Real per-fold results** (`scripts/run_walk_forward_pbo_audit.py --start-date
2024-01-01 --end-date 2026-06-30 --n-splits 3`, unmodified harness, fold run
ids `20260927_150634_363276_80086b86`, `20260927_152651_998122_98d20c94`,
`20260927_154914_557961_d0781d8e`):

| Fold | Train window | Test window | In-sample Sharpe: **full** (11) | In-sample Sharpe: **lean** (8) | Winner | OOS test Sharpe | OOS total return |
|---|---|---|---|---|---|---|---|
| 1 | 2024-01-01 → 2024-07-31 | 2024-08-01 → 2024-10-30 | -0.33 | **0.76** | **lean** | **+1.26** | +0.94% |
| 2 | 2024-10-30 → 2025-05-30 | 2025-05-31 → 2025-08-29 | **2.17** | 1.01 | **full** | **-3.20** | -2.33% |
| 3 | 2025-08-29 → 2026-03-29 | 2026-03-30 → 2026-06-28 | 1.00 | **1.11** | **lean** | **+4.94** | +4.51% |

Aggregate OOS (mean across the 3 walk-forward-selected test-window runs):
**Sharpe 1.00** (values 1.26 / -3.20 / 4.94), **PBO 0.705**, **Deflated
Sharpe 0.570**, **Probabilistic Sharpe 0.854**, **verdict: `fail`**.

### A genuinely mixed, honestly-reported result — not a clean win for either side

**Directionally suggestive for the leaner subset:** the walk-forward
selection procedure picked the **leaner 8-strategy subset in 2 of 3
folds** (1 and 3), and both of those OOS test windows were **strongly
positive** (+1.26, +4.94 Sharpe). The one fold where the harness picked
the **full 11-strategy roster** produced the **single worst OOS result of
the three** (-3.20 Sharpe, -2.33% return, the fold's largest drawdown at
2.81%). If this pattern held up on a larger sample, it would be a
meaningful point in favor of the leaner subset.

**But the formal statistical gate still says `fail`, and by one measure
(PBO) this is the worst result yet recorded against this system's own
audit history:** `PBO = 0.705` is nominally *worse* (higher probability of
backtest overfitting) than every previously-recorded PBO in
`docs/formal_pbo_audit.md`, including the very first, roughest audit
(0.686). At the same time, `Deflated Sharpe = 0.570` and `Probabilistic
Sharpe = 0.854` are the **best** DSR/PSR ever recorded in this system's
audit history (previous best DSR: 0.285). These two metrics disagreeing
this sharply is itself a signal to distrust the precision of either
number here, not a genuine contradiction to resolve: **PBO computed from
only 3 folds is a coarse, high-variance point estimate** (CSCV's
combinatorial-symmetry test has very little material to work with at
n=3), and **DSR mechanically improves with fewer candidates in the grid**
(this run tested only 2 candidates vs. 3-7 in prior audits — less
multiple-testing correction is needed, independent of whether the
underlying edge is real). Neither number should be compared at face value
against the historical baselines in `docs/formal_pbo_audit.md`, which used
different candidate grids, different fold counts, and a different (longer,
older) date range.

**A real methodological limitation, disclosed rather than glossed over:**
the harness only back-tests the OOS test window for the walk-forward-
selected winner each fold, not both candidates — so there is no
counterfactual for what the *full* roster would have scored in fold 1/3's
test windows, or what the *lean* subset would have scored in fold 2's. It
is therefore not possible to fully separate "the leaner subset is
genuinely more robust" from "the two good test-window periods happened to
coincide with when the lean subset was already in-sample-preferred." A
dedicated follow-up that back-tests *both* candidates on *every* fold's
test window (doubling the compute cost of this run) would close this gap
and is the natural next step if this hypothesis is worth pursuing further.

## Recommendation

**On the strategy-count/redundancy question this audit was scoped
around: inconclusive — not proven, not disproven.** The formal PBO gate
still fails, continuing this system's now well-established pattern (every
combination/architecture/roster change tested across this and prior
sessions — `zscore_demean`, concentration, `joint_optimizer`, stat_arb
pairs, seasonality overlay, macro overlay — has failed the same gate). The
directional numbers here (leaner subset selected 2/3 folds, both strongly
positive OOS; full ensemble selected once, strongly negative OOS) are the
most encouraging-looking numbers for a leaner roster seen in this
repo's audit history, but n=3 folds with no counterfactual cross-testing
is too thin a sample to justify a `config/live.yaml` change on its own.
**Do not drop any strategy from either live config based on this audit
alone.**

**Two concrete, more clearly-supported findings this audit does support,
worth acting on separately from the redundancy question:**

1. **`regime_hmm` and `seasonality` form a real, correlated cluster with
   `multi_factor` (0.72 / 0.53 correlation) in which they are both the
   weaker members** (Sharpe 0.06 and -0.55 vs. `multi_factor`'s 0.80) —
   this is a more surgical, better-targeted hypothesis than the 3-strategy
   leaner subset tested here (which also dropped `stat_arb` for an
   unrelated, Sharpe-only reason). **Recommend a dedicated follow-up**:
   re-run this same harness with a 2-candidate grid of *just* "full roster"
   vs. "drop `regime_hmm` + `seasonality` only" (keeping `stat_arb` and
   `mean_reversion` in both, since neither showed a redundancy signal),
   over a longer/more-folds range if compute allows, before making any
   config change.
2. **`signal_combination: optimal`'s inverse-covariance weighting has a
   concrete, mechanism-level blind spot around heterogeneous strategy
   history length**, independent of the redundancy question entirely: it
   parked 54.3% of its weight on `sentiment` in this window almost
   certainly because of `sentiment`'s short real cache history (starts
   2025-11-17), not genuine low-risk skill, while starving `multi_factor`
   (this window's best standalone Sharpe) of weight (1.4%, the smallest of
   all 11). This is arguably the more urgent, more clearly-evidenced
   finding in this whole audit — worth its own targeted fix (e.g. a
   minimum-trailing-history floor before a strategy participates in
   `optimal`'s weighting at full strength, analogous to how
   `strategy_circuit_breaker` already exists as separate, currently-off
   infrastructure) rather than being treated as a strategy-removal
   decision. Not implemented here — this audit's scope was measurement,
   not remediation — but flagged prominently for a follow-up.

**Bottom line:** this audit neither confirms nor refutes "the ensemble has
redundant strategies worth removing" to a standard this system would
normally require before touching `config/live.yaml` (see the repeated
`fail` verdicts throughout `docs/formal_pbo_audit.md`). It does newly and
concretely show that (a) the previously-reported 0.74 momentum/event_driven
correlation does not replicate in a fresh measurement and should not be
relied on, (b) `multi_factor`/`seasonality`/`regime_hmm` form a real,
actionable redundant cluster worth a dedicated, better-targeted follow-up,
and (c) `optimal` combination has a genuine, previously-undocumented
mechanism-level weakness around uneven strategy history length that is
probably worth fixing before any further redundancy work, since it
distorts every downstream weight regardless of which strategies are kept.
