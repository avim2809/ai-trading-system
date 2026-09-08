# Chart pattern recognition — implementation plan & progress tracker

**Status:** Phase 0 + Phase 1 complete, tested, and verified end-to-end
(backend + frontend) — not yet committed to git. ·
**Date:** 2026-09-09 · **Scope:** new `src/firm/patterns/` package feeding a
new Strategy #13 (`pattern_recognition`) into the existing 12-strategy /
8-agent pipeline, plus the frontend surfaces it touches.

This is the durable, repo-committed record of this initiative — the original
ask was a deep-dive research report + implementation plan for multi-bar chart
pattern detection (Head & Shoulders, triangles, flags, cup & handle, etc.),
which doesn't otherwise exist anywhere in this codebase or TA-Lib. That
research doc was presented in chat only (not saved anywhere), so **section 2
below preserves the load-bearing claims and design decisions from it** —
without this file, they'd be gone as soon as the conversation compacts or
ends. Sections 3–5 track what's actually been built, since it deviates from
that original chat-only plan in several concrete ways once checked against
the real codebase.

## 1. How to resume

1. Read section 4 ("Status") for what's done vs. outstanding — short version:
   Phase 0 + 1 (detection engine, Strategy #13, tests, frontend support) are
   done and verified; Phases 2-5 are not started (deliberately deferred, see
   section 2's table).
2. **Nothing has been committed to git yet** — `git status` will show
   `src/firm/patterns/`, `src/firm/strategies/pattern_recognition.py`,
   `tests/test_patterns.py` as untracked, plus modified files in
   `src/firm/strategies/__init__.py`, `src/firm/api/routers/meta.py`,
   `frontend/src/pages/{AgentInspector,NewBacktest}.tsx`, and a rebuilt
   `frontend/dist/`. All tests pass (26/26 new pattern tests, 1614/1614
   existing backend tests, 91/91 frontend tests) and it's been verified
   working live via a real Playwright browser run (see section 6) — treat it
   as ready to commit, not as unverified work-in-progress.
3. Section 3 ("Deviations") explains *why* the implementation departs from
   the original chat plan in several places — re-read it before assuming the
   original plan's pseudocode is authoritative; it wasn't checked against the
   real `BaseStrategy`/`Signal`/`PitView` contracts when written.
4. Section 5 lists every file with a one-line purpose — use it as a map
   instead of re-reading all the source top to bottom.
5. Section 6 documents four real bugs found while writing tests (a genuinely
   backwards comparison in the zigzag bootstrap phase, a systemic
   "pivots[-N:] isn't always the right window" issue across three rule
   modules, a handle-window measurement bug, and a structural characteristic
   of Double Top/Bottom's risk:reward) — worth reading before touching
   `extrema.py` or the rule modules, since the fixes are non-obvious and
   easy to accidentally revert.

## 2. Original research plan (condensed)

Full deep-dive covered: TA-Lib does not detect multi-bar chart patterns (only
1-3 bar candlesticks); raw/unscored pattern detection is close to a coin flip
(~47-51% win rate); quality-scoring every detection is the single highest-
leverage design choice (empirically-cited jump to ~65-75%+ win rate on
scored, high-quality setups); and market regime is a powerful filter on top
of that. Recommended architecture: rule-based extrema/geometry detection →
quality scoring → (optionally, later) ML confirmation → RL position sizing.

Proposed 5-phase rollout (13 weeks, illustrative — not a hard commitment):

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: extrema engine, rule detectors, scorer, signals | **Done — tested, see §4/§6** |
| 1 | Register as Strategy #13, plug into existing pipeline | **Done — tested end-to-end, see §4/§6** |
| 2 | Enhance `TechnicalAnalyst`/LLM variant with pattern-specific RAG validation | Not started — see 3.4, likely lower-value than originally scoped |
| 3 | Standalone scheduled scanner job + `/api/patterns/*` REST endpoints | Not started |
| 4 | ML training pipeline (XGBoost confirmation, CNN/GAF validator, PPO sizer) | Not started |
| 5 | React frontend `/patterns` page | Not started |

Empirical pattern algorithm reference (Lo/Mamaysky/Wang 2000 five-extrema
framework) — geometric conditions and measured-move targets for each pattern
family, as implemented (see section 5 for exact tolerances/defaults, which
were tuned during implementation and differ slightly from the original
research figures in a few places — e.g. shoulder tolerance defaulted to 5%
not 1.5%, since the tighter figure is a scoring target, not a detection gate):

- **Reversal** (`rules/reversal.py`): Head & Shoulders, Inverse H&S, Double
  Top/Bottom, Triple Top/Bottom — defined by alternating peak/trough pivots,
  confirmed on a neckline break.
- **Triangle/wedge/rectangle** (`rules/triangle.py`): Ascending/Descending/
  Symmetrical Triangle, Rising/Falling Wedge, Rectangle — classified by the
  sign/relative slope of OLS-fit resistance (peaks) and support (troughs)
  trendlines.
- **Continuation** (`rules/continuation.py`): Bull/Bear Flag, Pennant — a
  sharp flagpole move followed by a tight consolidation, classified as a
  parallel channel (flag) or converging wedge (pennant).
- **Cup & Handle / Rounding Bottom** (`rules/cup_handle.py`): a degree-2
  polynomial concave-up fit across two similar-height rim peaks, with or
  without a subsequent shallow handle pullback.

## 3. Deviations from the original plan (checked against the real codebase)

The original research doc was written/reviewed before (re-)reading this
repo's actual strategy/contracts code in depth. Once checked, several parts
of its illustrative pseudocode didn't match reality:

1. **No `pit_view.regime_state(ticker)` call.** No such method exists on
   `PitView` (`src/firm/strategies/base.py`), and no strategy reaches across
   into another strategy's output. `regime_hmm` is just one of 12 (now 13)
   parallel strategies whose `Signal`s get combined downstream (z-scored per
   strategy in `TechnicalAnalyst`, then blended across strategies via
   `agents/research/_regime_weights.py` / `_combine.py`). The pattern
   strategy follows the same convention — it does not gate on regime itself.
2. **No separate `config/patterns.yaml`.** Every existing strategy takes its
   params from a plain `dict` (`self.params`), sourced from
   `config["strategy_params"][<name>]` (see `runtime.py
   _build_categorized_strategies`) exactly like `settings.yaml`/`live.yaml`
   already do for the other 12 strategies. Inventing a parallel YAML file
   would be inconsistent with every other strategy and bypass the existing
   `/api/strategies` UI param-editing surface (`default_params` class attr).
3. **`Signal.meta`, not a new `PatternSignal` contract.** The real `Signal`
   dataclass (`firm/contracts/models.py`) is `(symbol, strategy, score,
   confidence, horizon, asof, meta: dict)`. Every existing strategy puts its
   diagnostic detail (entry/stop/target/pattern-specific fields, in this
   case) in `meta`, exactly like `regime_hmm` does with
   `{regime, posterior, separation, ...}`. A parallel `PatternSignal`
   dataclass would break the "any registered strategy automatically flows
   through `TechnicalAnalyst`" integration, since `TechnicalAnalyst.run`
   requires `strategy.generate(pit_view) -> list[Signal]` specifically.
4. **One `Signal` per symbol, not one per detected pattern.** If a symbol
   has multiple patterns detected the same bar, only the highest
   `quality_score` one becomes a `Signal` (others recorded in
   `meta["other_patterns"]` — see scanner, once written). `zscore_signals`
   (`agents/analysts/__init__.py`) z-scores by grouping on `strategy` name
   across the signals list; emitting >1 signal for the same symbol would
   double-count that symbol in the cross-sectional mean/std, a subtle
   statistical bug every other strategy avoids by construction (one row per
   symbol).
5. **ZigZag, not recursive PIP, for extrema.** The original plan's
   Perceptually-Important-Points sketch is O(n) per inserted point with an
   O(n) inner scan (~O(n·k) per call); it's also more error-prone to hand-
   verify. A percentage-reversal ZigZag (`extrema.py::zigzag_pivots`)
   guarantees strict peak/trough alternation by construction, is the
   standard technique in most real-world open-source chart-pattern scanners,
   and is what's actually implemented. A Gaussian kernel smoother
   (`extrema.py::gaussian_smooth`) is kept but only feeds the future ML
   layer (Phase 4), not current detection.
6. **Regime gating and volume/duration/follow-through scoring redesigned as
   a uniform post-detection pass**, not per-pattern-family bespoke scoring —
   see `scorer.py`. Dropped the plan's "regime_alignment" score component
   entirely (no data source for it inside a single strategy's `generate()`,
   per #1); rescaled from a 0-40 to a 0-100 rubric across five components
   (geometry/trendline-fit/volume/duration/follow-through — see `scorer.py`
   docstring for exact weights).
7. **Phase 2 (agent enhancement) is likely lower-value than originally
   scoped.** `runtime.py::_build_categorized_strategies` already routes any
   strategy not in `_FUND_STRATEGIES`/`_SENT_STRATEGIES` to
   `TechnicalAnalyst` automatically, which already does generic z-score
   aggregation over whatever strategies it's given. The pattern strategy
   gets this integration "for free" the moment it's registered — the
   LLM-prompt / RAG-validation enhancement described in the original Phase 2
   is a nice-to-have on top, not a requirement for end-to-end functioning.

## 4. Status

- [x] `_indicators.py` — ATR14 (Wilder's), trailing volume ratio
- [x] `extrema.py` — ZigZag pivot extraction (bootstrap-phase bug fixed, see
      §6.1) + `recent_pivot_windows()` retry helper (§6.2) + Gaussian kernel
      smoother (latter unused until Phase 4)
- [x] `trendline.py` — OLS line fit, 2-point line (neckline), degree-2 poly fit (cup/handle)
- [x] `confirmation.py` — shared breakout-confirmation search (newest-first, within a lookback window)
- [x] `match.py` — `PatternMatch` frozen dataclass (detection fields + scanner-filled scoring fields)
- [x] `scorer.py` — 0-100 quality rubric (geometry/trendline-fit/volume/duration/follow-through)
- [x] `rules/reversal.py` — H&S, IHS, Double Top/Bottom, Triple Top/Bottom (uses `recent_pivot_windows`)
- [x] `rules/triangle.py` — Ascending/Descending/Symmetrical Triangle, Rising/Falling Wedge, Rectangle (retrofitted with `recent_pivot_windows`, §6.2)
- [x] `rules/continuation.py` — Bull/Bear Flag, Pennant (pole-pair retry loop, §6.2; fit-window exclusion, §6.3)
- [x] `rules/cup_handle.py` — Cup & Handle, Rounding Bottom (handle-window fix, §6.3; uses `recent_pivot_windows`)
- [x] `scanner.py` — ties zigzag → the 9 rule-detector calls (6 reversal
      functions + 1 each triangle/continuation/cup_handle) → scorer → ATR
      stop floor, per symbol. Returns all confirmed matches above
      `min_score`, best first.
- [x] `strategies/pattern_recognition.py` — Strategy #13,
      `@register("pattern_recognition")`. Adjusts raw high/low by the
      adj_close/close ratio before scanning (keeps OHLC internally
      consistent across split/dividend boundaries — see its docstring).
      Emits at most one `Signal` per symbol (deviation #4), gated on both
      `min_score` and `min_risk_reward`.
- [x] Registered in `strategies/__init__.py`'s import list.
- [x] `tests/test_patterns.py` — 26 tests: zigzag correctness (incl. a
      below-threshold negative case), one positive-detection test per
      pattern (all 17 pattern names across the 4 rule families), a
      monotonic-trend negative case, degenerate-input robustness, scorer
      component sanity (clean-setup ≈100, weak-setup <15), `scan_symbol`
      min-score filtering/sort order, and 3 end-to-end tests through the
      real `BaseStrategy.generate(pit_view) -> list[Signal]` contract
      (signal emission + meta shape, empty universe, flat-data robustness)
      plus a registry-membership check. All 26 pass.
- [x] Full existing suite re-run after the change: `pytest -q
      --ignore=tests/test_api.py` → 1562 passed; `tests/test_api.py` (run
      separately per this repo's flakiness note) → 52 passed. No regressions
      — `pattern_recognition` being auto-included in every default-config
      backtest (§5.1) doesn't break anything existing.
- [x] Frontend support (user-requested follow-up, not in the original
      5-phase plan): confirmed via a dedicated Explore-agent audit that the
      frontend reads strategies generically from `GET /api/strategies`
      (`StrategyInfo[]`, no hardcoded name list/enum anywhere) — `NewBacktest`
      and `LiveConfig` both picked up `pattern_recognition` with zero code
      changes. Three real gaps fixed:
      1. `src/firm/api/routers/meta.py` — added a `pattern_recognition`
         entry to the `STRATEGY_INFO` dict (was missing, so `LiveConfig`'s
         description tooltip would've been blank).
      2. `frontend/src/pages/AgentInspector.tsx` — the only place `Signal
         .meta` is rendered at all previously special-cased just
         `llm_enhanced`/`llm_rationale`/`regime` with no generic fallback,
         so pattern_recognition's meta (the whole point of the strategy —
         pattern/entry/stop/target/quality_score/risk_reward) would've been
         silently invisible. Added a `PatternDetails` line gated on
         `sig.strategy === 'pattern_recognition'`.
      3. `frontend/src/pages/NewBacktest.tsx:389` — stale "blends the 12
         signals" copy (now 18 registered strategies, independent of this
         change) reworded to avoid hardcoding a count that will rot again.
      `npx tsc --noEmit` clean, `npx vitest run` 91/91 pass,
      `frontend/dist/` rebuilt. Verified live (not just type-checked): ran
      an isolated `firm-api` instance on an unused port, hit `/new`,
      `/live/config`, and `/inspector` with a real Playwright/Chromium
      session — screenshots confirm `pattern_recognition` renders correctly
      in all three, and clicking "Run Step" on synthetic data produced a
      genuine `symmetrical_triangle` signal on MSFT (quality 73, R:R 4.1)
      that rendered through the new `PatternDetails` line with no console
      errors. See §6.4 for a live-instance safety note from this exercise.
- [ ] Not yet committed to git — see §1.2. Commit as one "Phase 0+1
      pattern-detection engine + Strategy #13 + frontend support" change
      once the user confirms (per-user preference: commit self-contained
      tested chunks rather than leaving everything uncommitted, but only
      ever with explicit go-ahead).
- **Not planned this pass:** Phases 2 (LLM/RAG agent enhancement), 3 (live
  scheduler job + REST endpoints), 4 (ML training), 5 (dedicated `/patterns`
  frontend page) — each needs its own review/testing cycle and touches more
  sensitive surface (live scheduler, new API routes) than Phase 0/1's
  pure-addition, nothing-auto-enabled-live footprint (see §5.1).

## 5. Architecture reference

| File | Purpose |
|---|---|
| `src/firm/patterns/_indicators.py` | ATR14, trailing volume ratio |
| `src/firm/patterns/extrema.py` | `zigzag_pivots()`, `gaussian_smooth()` |
| `src/firm/patterns/trendline.py` | `fit_trendline()`, `line_through()`, `fit_poly2()` |
| `src/firm/patterns/confirmation.py` | `find_confirmation()` — shared breakout search |
| `src/firm/patterns/match.py` | `PatternMatch` frozen dataclass |
| `src/firm/patterns/scorer.py` | `PatternScore`, `score_pattern()` |
| `src/firm/patterns/rules/reversal.py` | H&S/IHS/Double/Triple Top&Bottom detectors |
| `src/firm/patterns/rules/triangle.py` | Triangle/wedge/rectangle detector |
| `src/firm/patterns/rules/continuation.py` | Flag/pennant detector |
| `src/firm/patterns/rules/cup_handle.py` | Cup & Handle / Rounding Bottom detector |
| `src/firm/patterns/scanner.py` | `scan_symbol()` — per-symbol orchestration |
| `src/firm/strategies/pattern_recognition.py` | Strategy #13, `@register("pattern_recognition")` |
| `tests/test_patterns.py` | 26 tests — see §4 |
| `src/firm/api/routers/meta.py` | +`pattern_recognition` entry in `STRATEGY_INFO` |
| `frontend/src/pages/AgentInspector.tsx` | +`PatternDetails` signal-meta rendering |
| `frontend/src/pages/NewBacktest.tsx` | stale hardcoded-count copy fix |

### 5.1 Live-safety note

`config/live.yaml` uses an **explicit `strategies.enabled` allowlist** — a
newly-registered strategy does **not** automatically run live. `config/
settings.yaml`, by contrast, defaults `strategies: []` to mean **all
registered strategies** (`runtime.py::_build_categorized_strategies`), so the
moment `pattern_recognition` is added to `strategies/__init__.py`'s import
list, it starts executing on every default-config backtest. It must
therefore never throw (wrap per-symbol detection in try/except, matching
`trend.py`/`regime_hmm.py` convention) and stay reasonably fast across a
full universe scan. It will not affect live trading unless a human
deliberately adds `pattern_recognition` to `config/live.yaml`'s
`strategies.enabled` list.

## 6. Bugs found while writing tests

Writing hand-built fixtures for every pattern (rather than trusting the
detectors' correctness by inspection) surfaced four real issues — none of
them fixture problems, all genuine implementation bugs or under-specified
behavior. Recorded here because the fixes are non-obvious and it would be
easy for a future edit to accidentally revert one of them.

### 6.1 ZigZag bootstrap phase had its anchors backwards

`extrema.py::zigzag_pivots`'s initial (`direction == 0`) phase compared
`high[i]` against the *running high* and `low[i]` against the *running low*
— e.g. `if high[i] >= anchor_high * (1 + pct)`. That's comparing a value
against a threshold derived from itself: since `anchor_high` was updated to
track the current price on every bar during bootstrap, this could only ever
fire via a single-bar jump of >=`pct`, never via a *cumulative* gradual move
(a smooth ramp of six bars at ~1.7%/bar, e.g., never fires — each bar's
comparison resets against the previous bar's now-higher anchor). The correct
check compares against the *opposite* running extreme (a confirmed downswing
means the running **high** was a peak, confirmed by checking `low[i]` against
it — not by checking `high[i]` against the running high). Several early test
fixtures "accidentally" worked around this because their anchors happened to
produce >=3% single-bar deltas; the ones with gradual multi-bar ramps
(`rising_wedge`, `bull_flag`, `bear_flag`) returned zero pivots and exposed
it. Fixed by tracking `max_idx/max_price` and `min_idx/min_price`
independently from bar 0 and cross-checking each against the other (bar 0
itself is still never emitted as a pivot — see the function's docstring).

### 6.2 `pivots[-N:]` isn't always the pattern's true last window

A pattern's last structural pivot (e.g. a Head & Shoulders' right shoulder,
or a flag's pole-end) is not always `pivots[-1]`: if the subsequent move away
from it — the breakout itself, a post-breakout bounce, a handle's own
pullback — is large enough to trigger its own zigzag confirmation, *that*
becomes the newest pivot instead, silently pushing the real pattern one or
more pivots further back in the list. Concretely: a bull flag's tight
consolidation, once the eventual breakout confirms it, gets recorded as its
own trough pivot — so `pivots[-2:]` reads as `[flag's own trough, ???]`
instead of `[flagpole_start, flagpole_end]`. Fixed generically via
`extrema.py::recent_pivot_windows(pivots, size, max_lookback)`, which yields
a few trailing windows (most recent first) instead of assuming the single
most-recent one is correct; `rules/reversal.py` (all six patterns), `rules/cup_handle.py`, and
`rules/triangle.py` (its window is adaptive — 4 to `window`=6 pivots, not a
fixed size — so it calls `recent_pivot_windows(pivots, min(window,
len(pivots)), max_lookback)` to keep the tried window size ≤ the available
pivot count) all loop over it. `rules/continuation.py` has its own version
of the same idea (`_try_flag` tried across `pivots[k:k+2]` for a few
trailing `k`), since flag/pennant detection has extra logic (the raw-bar
consolidation fit) that doesn't fit the generic helper's shape.
**Known limitation of this whole approach (not fixed, not exercised by any
current test):** the loop returns the *first* window that produces *any*
valid match, most-recent-first — it does not evaluate every candidate window
and pick the best-fitting one. If a noisy recent window happens to also
satisfy some *different* pattern's geometric conditions, that wrong-but-valid
match wins instead of falling through to the correct underlying pattern
further back. This is a real gap, but every concrete failure actually
observed while writing tests (§6.1, this section's motivating cases, §6.3)
was a *zero-matches* hard failure, not a wrong-classification one — so this
was left as a documented risk rather than chased with a speculative fix.

### 6.3 A detection window must exclude the breakout it's confirming against

Two independent instances of the same mistake: computing a "does this stay
within bounds" check over a bar range that *includes* the breakout bars
themselves, which by definition violate that exact bound.
- `rules/cup_handle.py`: the handle's high/low were originally measured over
  `[handle_start : n]` (all the way to the end of the array) — but the
  breakout bars in that range necessarily exceed the rim price, so
  `handle_high <= right_rim.price` would fail almost every time a real
  breakout happened. Fixed by finding the first bar that actually closes
  back above the rim (`first_breach`) and measuring the handle only over
  bars strictly before it.
- `rules/continuation.py`: the consolidation's upper/lower trendlines were
  originally fit over `[consol_start : n]`, which pulled the fitted slope
  toward the breakout ramp at the end — enough, in testing, to push a
  genuinely flat consolidation's fitted slope above the "is this really a
  pause" threshold and reject a valid bull flag. Fixed by fitting only over
  `[consol_start : n - confirm_lookback_bars]`, i.e. excluding the
  confirmation zone from the fit, then searching that excluded tail for the
  actual breakout.

### 6.4 Double Top/Bottom structurally cluster near 1:1 risk:reward

Not a bug — a real property worth knowing before assuming `min_risk_reward`
is mistuned. For Double Top/Bottom, the measured-move target height
(`level - avg(trough1, trough2)`) and the stop distance (`level -
min(trough1, trough2)`) are both approximately the same peak-to-trough swing
by construction, so `risk_reward` comes out near 1.0 almost regardless of how
clean the setup is — it's the pattern's own definition, not a scoring
artifact. A `min_risk_reward` default of 1.5 (this strategy's default)
correctly filters most Double Top/Bottom signals out; Flags, Cup & Handle,
and triangles/wedges don't share this property (their stop is placed at a
much tighter consolidation/shoulder boundary than their target height), so
they clear the same bar far more easily. Discovered because the first
end-to-end strategy test used a Double Bottom fixture and got zero signals
despite the raw detector firing correctly with quality_score 87 — the
fixture was swapped for a bull flag instead of loosening the filter.

### 6.5 Live-instance safety note (from the frontend verification pass)

Both this VPS's `firm-api` production instances (IBKR paper on :8000, Alpaca
paper on :8001, per `docs/project_ibkr_paper_trading_setup.md` /
`docs/project_alpaca_paper_instance_incident.md`-equivalent memory) were
already running when a plain `.venv/bin/firm-api` was launched for the
frontend smoke test — it failed to bind (`address already in use`) and
exited immediately both times, with no interference. `FIRM_AUTO_START_LIVE=1`
is set only via `ai-trading.service`'s `Environment=` directive and
`.env`'s `EnvironmentFile`, neither of which an interactively-launched
process picks up (confirmed no `dotenv`/auto-load anywhere in `src/firm/`) —
so a manually-run instance on an unused port (:8010 was used) is safe by
construction: no live auto-start, no broker connection, `live_engine_running:
false`. Worth remembering before ever launching this app directly on this
machine again: **always check `ss -ltnp` for the target port first**, and
prefer a port far from 8000/8001 (and 5173, which had an unrelated stray
node process on it already).
