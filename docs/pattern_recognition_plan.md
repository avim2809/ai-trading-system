# Chart pattern recognition — implementation plan & progress tracker

**Status:** All 5 phases complete, tested, and verified end-to-end (a
consolidated full-suite run against the final combined tree, plus a fresh
live browser pass covering every touched page). Phase 0+1 **committed**
(`53d5345`); Phases 2-5 verified and ready to commit. · **Date:** 2026-09-09
· **Scope:** new `src/firm/patterns/` package feeding a new Strategy #13
(`pattern_recognition`) into the existing 12-strategy/8-agent pipeline, an
XGBoost confirmation-classifier training pipeline, on-demand REST scan
endpoints + a `/patterns` frontend page, pattern-aware LLM validation, and
every frontend surface across all of the above.

**Final consolidated verification** (run against the complete tree, all 5
phases combined — supersedes each phase's own in-flight numbers from when
other phases were still concurrently editing the same tree, see §1.2):
`pytest -q --ignore=tests/test_api.py` → **1648 passed**; `tests/test_api.py`
(run separately per this repo's convention) → **52 passed**; frontend
`npx tsc --noEmit` → clean; `npx vitest run` → **94 passed** (19 files);
`frontend/dist/` rebuilt. A fresh live Playwright pass on a newly-started
isolated instance (port 8011, `live_engine_running: false` confirmed before
use) re-verified `/new`, `/live/config`, and `/inspector` for regressions
and exercised the new `/patterns` page end-to-end (real "Scan Now" trigger
→ 5 confirmed patterns of 5 different types across 5 synthetic symbols,
correctly scored/rendered) — zero console/page errors across all of it. Both
production instances (`:8000` IBKR, `:8001` Alpaca) confirmed unaffected
throughout, under their original unchanged PIDs.

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
   Phase 0+1 is committed (`53d5345`); Phases 2, 3, 4, 5 are all complete,
   tested, and (as of this writing) sitting uncommitted in the working tree,
   ready to commit. If you're reading this after that commit happened,
   `git log` will show it — this doc may lag one edit behind reality right
   after a commit, so cross-check `git status`/`git log` rather than trusting
   this line blindly forever.
2. Phases 2-4 were each built by an independent background subagent working
   concurrently in the *same* working tree (deliberately, to parallelize
   independent surface areas — LLM prompt, REST API, ML pipeline — that
   don't share files) — each was briefed with the real contracts/conventions
   from Phases 0-1 and told not to touch `src/firm/patterns/rules/`,
   `scanner.py`, `extrema.py`, `match.py`, `scorer.py`, or
   `strategies/pattern_recognition.py` (verified untouched by each). Their
   own individually-reported test-suite counts differ slightly from each
   other and from this doc because the tree kept changing under them
   mid-run — each confirmed *zero failures* in their own run regardless, and
   a final consolidated full-suite run (see §4) is the number that actually
   matters, not any single subagent's in-flight count.
3. Section 3 ("Deviations") explains *why* the implementation departs from
   the original chat plan in several places — re-read it before assuming the
   original plan's pseudocode is authoritative; it wasn't checked against the
   real `BaseStrategy`/`Signal`/`PitView` contracts when written.
4. Section 5 lists every file with a one-line purpose — use it as a map
   instead of re-reading all the source top to bottom.
5. Section 6 documents real bugs/design issues found across every phase
   (starting with a genuinely backwards comparison in the zigzag bootstrap
   phase) — worth reading before touching `extrema.py`, the rule modules, or
   the ML labeling code, since several fixes are non-obvious and easy to
   accidentally revert.

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
| 2 | Enhance `TechnicalAnalyst`/LLM variant with pattern-specific RAG validation | **Done — tested, see §4/§6.6** (narrower than originally scoped, see 3.7 — surgical, not a rewrite) |
| 3 | On-demand `/api/patterns/*` REST endpoints (no scheduled job — see §4/§6.7) | **Done — tested, see §4/§6.7** |
| 4 | ML training pipeline — **XGBoost confirmation only**, CNN/GAF + PPO sizer deliberately descoped (see §4/§6.8) | **Done — tested, see §4/§6.8** |
| 5 | React frontend `/patterns` page | **Done — tested end-to-end, see §4/§6.10** |

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
- [x] **Committed** as `53d5345` ("Add chart-pattern recognition (Strategy
      #13) with full test coverage") — Phase 0+1 only (patterns package,
      Strategy #13, tests, the 3 frontend fixes). Phases 2-4 below came
      after this commit and are uncommitted as of this writing.

### Phase 2 — pattern-aware LLM validation (done)

- [x] `src/firm/agents/llm/technical_analyst_llm.py` — `LLMTechnicalAnalyst
      .run()` now branches on `sig.strategy == "pattern_recognition"` to use
      a richer RAG query (`f"{pattern} chart pattern reliability breakout
      confirmation"` instead of the generic per-strategy one) and a richer
      prompt surfacing `pattern/direction/entry/stop/target/risk_reward
      /quality_score/volume_ratio` from `sig.meta`. Reuses the *exact* same
      `AnalystEnhancementResponse` JSON contract, `_call_llm`/
      `_bounded_override` mechanics, and z-score renormalization as every
      other strategy's enhancement path — only the query/prompt text
      differs by branch; the non-pattern path is byte-identical to before.
- [x] `tests/test_llm.py::TestLLMTechnicalAnalyst
      ::test_pattern_recognition_signal_uses_pattern_specific_prompt` — new
      test asserting the pattern-specific RAG query and prompt content
      (checks for the pattern name, formatted entry/stop/target/risk:reward,
      and the *absence* of the generic prompt's wording). Existing
      `test_returns_signal_set` (momentum path) untouched and still passes.
- [x] Along the way, found (and confirmed, via `git stash` A/B) a
      **pre-existing, unrelated** test-fixture gap: `tests/test_llm.py`'s
      `mock_llm_modules` fixture never registers `"firm.llm.schemas"` in
      `sys.modules`, so running that file *standalone* (before anything else
      has imported the real `firm.llm.*`) spuriously fails ~22 unrelated
      tests. Not fixed (out of scope, pre-existing, and the full suite's
      natural import order never hits it) — flagging here so it isn't
      mistaken for something Phase 2 broke if someone runs
      `pytest tests/test_llm.py` in isolation.

### Phase 3 — on-demand pattern-scan REST API (done)

- [x] `src/firm/api/routers/patterns.py` (registered in `app.py` same as
      every other router) — 4 endpoints, all synchronous, **none scheduled**:
      `POST /api/patterns/scan/trigger` (body: `PatternScanRequest` —
      symbols/asof/data_source/scan-tuning params, all defaulted from
      `PatternRecognitionStrategy.default_params`; runs a real scan via the
      real `scan_symbol` + the strategy's own `_adjusted_ohlc`, reused not
      reimplemented, and replaces an in-memory cache), `GET /api/patterns
      /scan` (cached results, filterable by pattern/min_score/direction,
      best-quality-first), `GET /api/patterns/summary` (counts by
      pattern/direction), `GET /api/patterns/{symbol}` (per-symbol, `[]` not
      404 when empty — registered *last* so it doesn't shadow the three
      fixed-path routes above it, since Starlette matches in registration
      order). Cache is process-local, in-memory, non-persistent by design —
      no new persistence layer for this pass.
- [x] `frontend/src/api/{types,client}.ts` — full typed plumbing
      (`PatternMatchRecord`/`PatternScanQuery`/`PatternScanTriggerRequest`
      /`PatternScanTriggerResponse`/`PatternSummary` + 4 `api.*` functions)
      ready for Phase 5 to consume; `npx tsc --noEmit` clean.
- [x] `tests/test_patterns_api.py` — 24 tests, all passing.
- [x] **Deliberately descoped** (see §6.7 for the full reasoning): the
      `/api/patterns/history` trade-outcome-tracking endpoint (needs new
      persistence — a separate feature) and, most importantly, **any
      scheduled/periodic scanning** — nothing wired into `app.py`'s
      lifespan or `firm.live.scheduler.TradingScheduler`; every scan is
      human/frontend-triggered only. Zero risk to the two running
      production engines by construction, not just by convention.

### Phase 4 — ML confirmation layer (XGBoost only; done)

- [x] `src/firm/patterns/ml/feature_engineering.py` — `build_features(match,
      ohlcv=None)`: one `PatternMatch` → ~47-key numeric feature dict
      (scalar fields, scale-invariant stop/target distances, unpacked
      `score_breakdown`, one-hot pattern/pattern-family encoding, optional
      OHLCV-derived context). Pure, NaN-safe function.
- [x] `src/firm/patterns/ml/labeling.py` — `label_triple_barrier` (López de
      Prado): walks forward from `confirm_index + 1` against the pattern's
      *own* entry/stop/target, `+1`/`-1`/`0` (target/stop/timeout),
      same-bar collisions resolve to stop (fail-closed, matching this
      codebase's convention elsewhere). `DEFAULT_TIMEOUT_BARS=20`,
      parameterized.
- [x] `src/firm/patterns/ml/xgb_classifier.py` — `train`/`predict_proba`/
      `predict_label`/`save`/`load`, wrapping a fixed `(-1, 0, +1)` label
      space regardless of which classes a given training slice happens to
      contain (xgboost 3.x requires contiguous `0..k-1` fit-time labels).
- [x] `scripts/train_pattern_ml.py` — end-to-end CLI (scan → feature → label
      → train → report), universe/date-range/output/scan-params
      CLI-configurable, defaults to `--data-source synthetic` so a bare
      invocation never touches real market data. See §6.8 for a real
      look-ahead/degenerate-labeling bug found and fixed while building this.
- [x] `tests/test_pattern_ml.py` — 35 tests (xgboost-dependent ones
      skip-gated so the file degrades gracefully without the extra
      installed).
- [x] `xgboost` added as an **optional** `patterns_ml` extra in
      `pyproject.toml` (same convention as the existing `report`/quantstats
      extra) — base install untouched. See §6.9 for a transitive-dependency
      issue found and corrected during install.
- [x] **Deliberately descoped**: the CNN/GAF image validator and PPO RL
      position sizer from the original research plan — both need heavy new
      dependencies (torch, stable-baselines3, gymnasium, pyts) that would be
      irresponsible to add unsupervised to this resource-constrained,
      live-trading-hosting VPS. ONNX export also skipped (plain
      pickle for now) — noted as the natural next step for low-latency
      serving if this is ever wired into a live scan path.

### Phase 5 — `/patterns` frontend page (done)

- [x] `frontend/src/pages/PatternScanner.tsx` (route `/patterns`, nav label
      "Pattern Scanner" in `Layout.tsx` right after "Agent Inspector") —
      scan-trigger form (modeled on `AgentInspector.tsx`) + 3 summary stat
      tiles + filter row (pattern/min-score/direction) + results table
      (modeled on `OrderHistory.tsx`), all against the Phase 3 API. Server-
      side filtering (each filter combination is its own query key) uses
      `placeholderData: keepPreviousData` so changing a filter doesn't blank
      the whole page back to a spinner — see §6.10 for why that mattered.
- [x] `frontend/src/pages/PatternScanner.test.tsx` + updated
      `frontend/src/test/{handlers,mockData}.ts` fixtures (verified against
      real backend curl output, not guessed).
- [x] `frontend/src/App.tsx` (+route), `frontend/src/components/Layout.tsx`
      (+nav link) — both minimal, additive diffs.
- [x] `npx tsc --noEmit` clean; `npx vitest run` 94/94 (91 baseline + 3 new);
      `frontend/dist/` rebuilt.
- [x] Live-verified twice — once by the building subagent (real trigger,
      real filter round-trip against the real backend, 375px mobile layout
      check) and again in the final consolidated pass (§ below) alongside a
      regression check of `/new`, `/live/config`, `/inspector`.

**Consolidated verification (final, against the complete 5-phase tree) —
see the status line at the top of this doc for the numbers.** This
supersedes every individual phase's own in-flight test counts, which were
each taken at a different moment while the other phases were still being
built concurrently in the same working tree (a deliberate parallelization
choice — see §1.2 — not a mistake, but it does mean no single subagent's
reported number should be treated as the final word).
- **Not yet committed to git** (Phases 2-5) — see §1.1.

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
| `src/firm/agents/llm/technical_analyst_llm.py` | +pattern-specific RAG query/prompt (Phase 2) |
| `src/firm/api/routers/patterns.py` | 4 on-demand scan endpoints (Phase 3) |
| `frontend/src/api/{types,client}.ts` | +`Pattern*` types/client functions (Phase 3) |
| `src/firm/patterns/ml/feature_engineering.py` | `build_features()` (Phase 4) |
| `src/firm/patterns/ml/labeling.py` | `label_triple_barrier()` (Phase 4) |
| `src/firm/patterns/ml/xgb_classifier.py` | train/predict/save/load wrapper (Phase 4) |
| `scripts/train_pattern_ml.py` | end-to-end training CLI (Phase 4) |
| `tests/test_llm.py` | +1 test (Phase 2) |
| `tests/test_patterns_api.py` | 24 tests (Phase 3) |
| `tests/test_pattern_ml.py` | 35 tests (Phase 4) |
| `frontend/src/pages/PatternScanner.tsx` | `/patterns` page (Phase 5) |
| `frontend/src/pages/PatternScanner.test.tsx` | 3 tests (Phase 5) |
| `frontend/src/App.tsx` | +`/patterns` route (Phase 5) |
| `frontend/src/components/Layout.tsx` | +nav link (Phase 5) |
| `frontend/src/test/{handlers,mockData}.ts` | +pattern-endpoint fixtures (Phase 5) |

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

## 6. Bugs and design notes found across every phase

§6.1-6.5 are from Phase 0/1 (writing hand-built fixtures for every pattern,
rather than trusting the detectors' correctness by inspection, surfaced real
implementation bugs, not fixture problems); §6.6-6.9 are from Phases 2-4;
§6.10 is from Phase 5. Recorded here because several of the fixes are
non-obvious and it would be easy for a future edit to accidentally revert
one of them.

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

### 6.6 Phase 2 note: the generic RAG query degenerates for this strategy

Every other strategy's generic enhancement query
(`f"academic research on {sig.strategy} strategy patterns for {sig.symbol}"`)
is at least a plausible search — "academic research on momentum strategy
patterns". For `pattern_recognition`, `sig.strategy` is always literally the
string `"pattern_recognition"`, so the same template produces "academic
research on pattern_recognition strategy patterns" for *every* detection
regardless of whether it's a cup & handle or a head & shoulders — the actual
useful query key (which specific chart pattern) was sitting unused in
`sig.meta["pattern"]` the whole time. Worth remembering if any *other* future
strategy also carries its real semantic content in `meta` rather than in its
own name.

### 6.7 Phase 3 note: route registration order matters for the catch-all

FastAPI/Starlette match path operations in registration order. `GET
/api/patterns/{symbol}` is a single-segment catch-all that would shadow
`/api/patterns/scan` and `/api/patterns/summary` (matching them with
`symbol="scan"`/`"summary"`) if registered before them. The router file
registers the three fixed-path routes first and the catch-all last,
specifically to avoid this — easy to break by innocently reordering the
functions in the file, so any future edit to that file should preserve the
order (or add an explicit test — `tests/test_patterns_api.py` does cover
this, so a reorder that breaks it should fail loudly).

### 6.8 Phase 4 note: a single-shot scan almost never has forward history to label

`firm.patterns.confirmation.find_confirmation`'s search is, by design, "most
recent breakout within `confirm_lookback_bars` of the window's *last* bar" —
exactly what a live strategy wants ("did something just confirm as of
today"). But it means a single whole-series `scan_symbol()` call over
historical data almost always returns matches with `confirm_index` in the
last 1-2 bars of the input, leaving virtually no genuine subsequent price
history to triple-barrier-label against — confirmed empirically while
building `scripts/train_pattern_ml.py`: the first version's dataset had
every label come back `0` (timeout), since there was nothing after
`confirm_index` to look at. Fixed by rolling a growing "as-of" cutoff
backward through each symbol's history (`min_window_bars`/`step_bars` in
`build_dataset()`) instead of scanning the whole series once — each rolled
cutoff is a lightweight point-in-time re-scan that can turn up patterns
confirmed well before "today", each with real forward bars already present
in the same series to label from. Correctly kept the point-in-time
distinction the rest of this codebase cares about throughout: labels use the
*full* series (the label is the answer key — knowing what actually happened
next is the entire point of supervised learning), but **features are built
only from the as-of `window`** (what a live scan on that date would actually
have seen) — mixing those two up would leak future information into the
features themselves, not just the label.

### 6.9 Phase 4 note: installing xgboost pulled in an unwanted GPU dependency

`pip install xgboost` transitively pulled in `nvidia-nccl-cu13` (~290MB), a
multi-GPU communication library irrelevant to this box's CPU-only `hist`
tree-method training. Uninstalled after confirming train/predict still work
without it; left a warning comment in both `pyproject.toml`'s `patterns_ml`
extra and `scripts/train_pattern_ml.py`'s docstring so a future
`pip install '.[patterns_ml]'` on this or another box doesn't silently
reintroduce ~290MB of unused GPU tooling onto a resource-constrained VPS.

### 6.10 Phase 5 note: server-side filtering needs `keepPreviousData`

`PatternScanner.tsx` filters server-side (`GET /patterns/scan?pattern=...
&min_score=...&direction=...`), unlike `OrderHistory.tsx`'s client-side
array filtering over one static query. That means every distinct filter
combination is its own React Query cache key, and the first time any
particular combination is selected there's no cached data for it yet. A
naive top-level `isLoading` gate (copied from `OrderHistory.tsx`, which
never has this problem since it only ever has one query) blanked the
*entire page* — form, stat tiles, other filters included — back to a bare
spinner on every new filter value, not just the results table. Fixed with
`placeholderData: keepPreviousData` (TanStack Query v5), which keeps
rendering the previous result set (with a small inline spinner next to the
match count) while the new combination loads in the background. Worth
remembering for any future page that filters server-side rather than
client-side — the `OrderHistory.tsx` loading-state pattern silently assumes
client-side filtering and doesn't generalize.
