# Chart pattern recognition — implementation plan & progress tracker

**Status:** All 5 original phases complete, tested, and committed
(`53d5345`..`da08323`). A follow-up pass then closed out every item that had
been deliberately left open: two small bug fixes, a scheduled scan job +
persistent history endpoint, real XGBoost training + ONNX export, and the
CNN/GAF validator + PPO RL position sizer that were previously descoped for
dependency reasons (`bbb72c8`, `d0e9727`, `4340b0b`, `680fa75`). A second
improvement round (§8, 2026-09-25 — uncommitted as of this writing) added
ATR-scaled zigzag/retest/confluence quality modifiers, an XGBoost+CNN
ensemble with per-model calibration wired into the live strategy, a
meta-labeling Kelly path in `TraderAgent`, and a repeatable CPCV-style
walk-forward validation harness + staged rollout gate — every new knob
defaults off pending its own walk-forward re-validation. **The CNN-toggle
re-validation itself came back negative** (8-fold walk-forward verdict=fail)
and, per the user's decision, `cnn_scoring_enabled` has been rolled back to
`false` in both live configs and hot-swapped into both running engines
(no restart) — see §8.2. ·
**Date:** 2026-09-09 · **Scope:** new `src/firm/patterns/` package feeding a
new Strategy #13 (`pattern_recognition`) into the existing 12-strategy/8-agent
pipeline, an XGBoost confirmation-classifier training pipeline with ONNX
export, an optional scheduled scan job + persistent scan history, on-demand
REST scan endpoints + a `/patterns` frontend page, pattern-aware LLM
validation, a CNN/GAF image validator, a PPO RL position sizer, and every
frontend surface across all of the above.

**Final consolidated verification for the original 5 phases** (run against
the complete tree, all 5 phases combined — supersedes each phase's own
in-flight numbers from when other phases were still concurrently editing the
same tree, see §1.2): `pytest -q --ignore=tests/test_api.py` → **1648
passed**; `tests/test_api.py` (run separately per this repo's convention) →
**52 passed**; frontend `npx tsc --noEmit` → clean; `npx vitest run` → **94
passed** (19 files); `frontend/dist/` rebuilt. A fresh live Playwright pass on
a newly-started isolated instance (port 8011, `live_engine_running: false`
confirmed before use) re-verified `/new`, `/live/config`, and `/inspector` for
regressions and exercised the new `/patterns` page end-to-end (real "Scan
Now" trigger → 5 confirmed patterns of 5 different types across 5 synthetic
symbols, correctly scored/rendered) — zero console/page errors across all of
it. Both production instances (`:8000` IBKR, `:8001` Alpaca) confirmed
unaffected throughout, under their original unchanged PIDs.

**Final consolidated verification for the follow-up pass** (§7 below,
covering small fixes + scheduled job/history + ONNX export + CNN/PPO):
`pytest -q --ignore=tests/test_api.py` → **1675 passed, 25 skipped** (the
skips are the CNN/PPO/ONNX tests correctly gating themselves out of the main
Python-3.14 venv); `tests/test_api.py` → **52 passed**; `.venv-ml/bin/pytest
tests/test_cnn_validator.py tests/test_ppo_sizer.py tests/test_pattern_ml_onnx.py`
(isolated Python 3.12 env, see §7.4) → **all passed**; frontend `npx tsc
--noEmit` → clean; `npx vitest run` → **94 passed**. Both production
instances confirmed unaffected throughout, under their original unchanged
PIDs (`3682961`/`3683272`) — nothing in this pass touches either running
engine until the final "enable live" step.

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

1. Read section 4 ("Status") for the original 5 phases and section 7 for the
   follow-up pass that closed out everything those phases had left open —
   short version: everything in both sections is done, tested, and
   committed (`53d5345`..`680fa75`; `git log` shows the exact sequence). The
   only remaining step, gated on explicit human approval each time, is
   pushing and adding `pattern_recognition` to `config/live.yaml`'s enabled
   strategies (§7.5) — cross-check `git status`/`git log`/`config/live.yaml`
   rather than trusting this line blindly forever.
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
6. Section 7 documents the follow-up pass (small fixes, scheduled job +
   history, real ML training + ONNX export, isolated-env CNN/PPO) — read
   §7.4 in particular before touching `ppo_sizer.py`'s `risk_aversion`
   default: it looks like an arbitrary tuning constant but was calibrated
   against a real training run, and a smaller value silently collapses the
   RL policy back to the exact degenerate "always bet max size" behavior it
   exists to prevent.

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
| 3 | On-demand `/api/patterns/*` REST endpoints + (follow-up, §7.2) scheduled scan job + persistent history | **Done — tested, see §4/§6.7/§7.2** |
| 4 | ML training pipeline — XGBoost confirmation + ONNX export + (follow-up, §7.3-7.4) CNN/GAF validator + PPO RL sizer | **Done — tested, see §4/§6.8/§7.3/§7.4** |
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
- [x] **Follow-up pass (§7) closed out both items originally descoped
      here:** a persistent `GET /api/patterns/history` endpoint backed by a
      new SQLite store, and an independent, default-OFF scheduled scan job.
      See §7.2 for the full writeup — it stays a completely separate
      scheduler from `firm.live.scheduler.TradingScheduler`, gated by its own
      env var, so the zero-risk-by-construction property for the two running
      production engines is preserved.

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
- [x] **Follow-up pass (§7) closed out every item originally descoped here:**
      a real training run against cached historical data, ONNX export for
      the XGBoost model, and — despite the heavy new dependencies — the
      CNN/GAF image validator and PPO RL position sizer, built inside a new
      isolated Python 3.12 environment (`.venv-ml`) that carries zero risk to
      the live `firm-api` process. See §7 for the full writeup.

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
- **Committed** as `ba7ccfe` (Phase 2), `3945c68` (Phase 3 REST API +
  Phase 5 `/patterns` frontend page, bundled in one commit), `738b5e8`
  (Phase 4). Everything originally descoped from Phases 3 and 4 was
  subsequently built in the follow-up pass documented in §7.

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
| `src/firm/live/pattern_scan_job.py` | independent scheduled scan job, default OFF (§7.2) |
| `src/firm/live/pattern_scan_history.py` | `PatternScanHistoryStore` — persistent SQLite scan history (§7.2) |
| `tests/test_pattern_scan_job.py` | 5 tests (§7.2) |
| `tests/test_pattern_ml_onnx.py` | XGBoost ONNX round-trip tests, isolated env (§7.3) |
| `src/firm/patterns/ml/cnn_validator.py` | GAF encoding + CNN validator, isolated env (§7.4) |
| `scripts/train_cnn_validator.py` | CNN training CLI, isolated env (§7.4) |
| `tests/test_cnn_validator.py` | 11 tests, isolated env (§7.4) |
| `src/firm/patterns/ml/ppo_sizer.py` | PPO position-sizing agent, isolated env (§7.4) |
| `scripts/train_pattern_ppo.py` | PPO training CLI, isolated env (§7.4) |
| `tests/test_ppo_sizer.py` | 11 tests, isolated env (§7.4) |

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

## 7. Follow-up pass — closing out everything left open (done)

The original 5 phases deliberately left several things open: two small bugs,
a scheduled scan job + persistent history (descoped from Phase 3), running
the ML pipeline for real + ONNX export (descoped from Phase 4), and the
CNN/GAF validator + PPO RL sizer (descoped from Phase 4 for dependency
reasons). This section covers all of it, plus the two final operational
steps (push, enable live).

### 7.0 Disk cleanup (prerequisite)

The root filesystem was at 94% full (1.4G free of 20G) — not enough headroom
to install a second Python interpreter plus a torch-class virtualenv.
Cleared only unambiguously-safe, regenerable items: `journalctl
--vacuum-size=100M`, `~/.cache/pip`, `~/.cache/ms-playwright` (reinstallable
via `npx playwright install chromium`), `~/.vscode-server/data
/CachedExtensionVSIXs`, already-rotated system logs, and old (non-active)
`~/.vscode-server/cli` version directories (identified by mtime, keeping the
currently-connected build untouched). Left untouched: `.venv`, IB Gateway
(`/opt/ibgateway`+`i4j_jres`), all live-trading state under `data/`
(`vectordb`, `live_state.db`, `approvals.json`, `execution_audit.jsonl`).
Freed enough to comfortably fit `.venv-ml` (currently 3.1G free).

### 7.1 Small fixes

- **Pivot-window detectors now collect every valid match, not just the
  first.** `rules/{reversal,triangle,continuation,cup_handle}.py`'s 9
  detector functions changed from returning `PatternMatch | None` on the
  first window that produced *any* valid match (see §6.2's "known
  limitation") to collecting every valid match across all tried windows into
  a `list[PatternMatch]`. `scanner.py`'s `_ALL_DETECTORS` loop changed from
  appending one candidate to extending with the whole list — it already
  sorts everything by `quality_score` afterward, so no other change was
  needed. `tests/test_patterns.py` gained
  `test_double_top_collects_all_valid_windows_best_quality_first`, which
  constructs a fixture with two valid double-top windows of different
  quality and asserts both are returned, better-quality first — closing the
  gap §6.2 flagged as a documented risk rather than a fix.
- **`tests/test_llm.py`'s `mock_llm_modules` fixture never registered
  `"firm.llm.schemas"`** (see §6.6's Phase 2 note for how this was first
  found) — one-line fix: register the real imported module in the fake
  `sys.modules` dict alongside the existing fake `.provider`/`.compression`
  /`.config`/`.exceptions` entries. `pytest tests/test_llm.py -q` now passes
  standalone, not just as part of the full suite.

### 7.2 Scheduled scan job + persistent history endpoint

- **`src/firm/live/pattern_scan_job.py`** — its own `BackgroundScheduler`
  (APScheduler), completely independent of `firm.live.scheduler
  .TradingScheduler` (never imports or touches it). One `CronTrigger(hour=16,
  minute=30, day_of_week="mon-fri", timezone="US/Eastern")` job,
  `max_instances=1, coalesce=True`, wrapped in a try/except-log-continue
  matching `scheduler.py`'s `_run_cycle_safe` convention. Reuses
  `scan_symbol` + `_adjusted_ohlc` exactly as the REST endpoint does — no new
  scanning logic. Gated by `FIRM_ENABLE_PATTERN_SCAN` (default off, same
  parsing convention as `FIRM_AUTO_START_LIVE`); wired into `app.py`'s
  `lifespan()` right after the existing auto-start-live task, stashed on
  `application.state.pattern_scan_job`, stopped on shutdown.
- **`src/firm/live/pattern_scan_history.py`** — `PatternScanHistoryStore`,
  following the exact SQLite convention from `firm.llm.cache.ResponseCache`
  (WAL mode, `threading.Lock`, idempotent `CREATE TABLE IF NOT EXISTS`, every
  read/write wrapped in try/except that logs and degrades gracefully). One
  row per persisted match (reusing `patterns.py`'s existing
  `_serialize_match` shape), plus a nullable `outcome` column updated
  in-place after each scan by re-checking any still-pending rows against
  fresh price data via the existing `label_triple_barrier` (no second
  triple-barrier implementation). DB path scoped via `FIRM_DATA_DIR` so the
  two `firm-api` instances never collide on one file.
- New `GET /api/patterns/history` endpoint (paginated, filterable),
  registered before the `/{symbol}` catch-all per §6.7's ordering rule.
  `POST /patterns/scan/trigger` now also persists into this store, not just
  the in-memory cache.
- **Bug found and fixed while building this:** `patterns.py`'s original
  history-store accessor read `FIRM_DATA_DIR` into a module-level constant
  (`_DATA_DIR = os.environ.get(...)`), evaluated once at import time — so a
  test's `monkeypatch.setenv("FIRM_DATA_DIR", ...)` had no effect and every
  test run leaked writes into the real `data/pattern_scan_history.db`.
  Fixed by reading the env var fresh inside `_history_store()` at call time;
  the leaked file was found and deleted.
- `tests/test_pattern_scan_job.py` (5 tests: env-gate parsing, app-wiring via
  `with TestClient(app) as client:`, safe run-once, pending-outcome
  resolution) + 6 new tests in `tests/test_patterns_api.py` for the history
  endpoint. Live-safety re-verified exactly as in §6.5: on an isolated port,
  with `FIRM_ENABLE_PATTERN_SCAN` unset nothing new starts; only with it set
  does the independent scheduler appear.

### 7.3 Real ML training run + ONNX export

Ran `scripts/train_pattern_ml.py --data-source cache` for real (25-symbol
universe, 2010-2026) — accuracy/AUC came back in the same ballpark as the
synthetic smoke test, not overfit-perfect or degenerate. Model saved to
`data/models/pattern_xgb.pkl` (new `.gitignore`d `data/models/` convention).

Added `xgb_classifier.export_onnx`/`load_onnx`/`predict_proba_onnx`, run and
verified inside the isolated `.venv-ml` environment (§7.4) since
`onnxruntime` has no Python 3.14 wheel either. **Bug found and fixed:**
`train()` originally fit the booster on a named-column `pandas.DataFrame` —
xgboost bakes those column names into its tree dump, and `onnxmltools`'
converter can't parse them back out (`Unable to interpret 'score_
follow_through', feature names should follow pattern 'f%d'`). Fixed by
fitting on `X.to_numpy()` instead; verified behavior-preserving by re-running
`tests/test_pattern_ml.py` (35 passed, unchanged) and re-running the real
training script (identical accuracy: 0.852/0.737/0.871). `tests/test_
pattern_ml_onnx.py` (new, skip-gated) round-trips a trained model through
ONNX and asserts matching predictions.

### 7.4 Isolated Python 3.12 environment + CNN/GAF validator + PPO RL sizer

**Environment.** This box runs Python 3.14.4, and torch/stable-baselines3/
gymnasium/onnxruntime have no Python 3.14 wheels yet (confirmed via PyPI's
JSON API). `apt-get install python3.12` isn't available in this Ubuntu
release's repos either. Built via `uv` (Astral's Python toolchain manager)
instead: `uv python install 3.12` fetches a prebuilt interpreter with no
APT/compilation involved, then `uv venv --python 3.12 .venv-ml` + `uv pip
install --python .venv-ml/bin/python ...`. **Gotcha:** uv-created venvs ship
no `pip` binary at all — `.venv-ml/bin/pip` doesn't exist, so `.venv-ml/bin/
pip list` silently no-ops instead of erroring; must use `uv pip list
--python .venv-ml/bin/python`. This environment is never imported by the
running `firm-api` process (Python 3.14) — it's used only for standalone
scripts run manually. Installing `xgboost` here pulled in the same unwanted
`nvidia-nccl-cu13` (~240MB) as §6.9; removed the same way.
`PYTHONPATH=src .venv-ml/bin/python ...` is required to invoke it directly
(the package isn't installed into `.venv-ml`, only imported via path).

**`src/firm/patterns/ml/cnn_validator.py`** — GAF (Gramian Angular Field,
Wang & Oates 2015) encodes the OHLCV window before a confirmed pattern's
breakout as an image; a small 2D CNN (`torch`) predicts the same
triple-barrier outcome the XGBoost classifier does, for direct comparison.
Independent of `xgb_classifier` in both directions — neither requires the
other installed. `export_onnx` works cleanly via `torch.onnx.export` (a
plain `nn.Module` has none of xgboost's feature-naming issue from §7.3).
`scripts/train_cnn_validator.py` mirrors `train_pattern_ml.py`'s structure.
`tests/test_cnn_validator.py` — 11 tests, all passing under `.venv-ml`.

**`src/firm/patterns/ml/ppo_sizer.py`** — a single-step "contextual bandit"
environment (not a multi-step portfolio simulation — deliberately out of
scope, see the module docstring) that learns *how much* to bet on a
confirmed pattern's context, reusing the same per-match dataset the other
two models train on. **Real design flaw found and fixed:** a purely linear
(risk-neutral) reward makes "always bet max size" reward-optimal whenever
the dataset's average outcome is positive — verified empirically, the first
trained policy's mean reward matched the always-max-size baseline exactly.
Fixed with a quadratic risk-aversion penalty on position size (standard
mean-variance / quadratic-utility sizing): `reward = pnl - cost*|size| -
risk_aversion*size**2`. **This needed a second round of calibration, not
just the mechanism:** the *default* `risk_aversion` value matters as much as
having the term at all. `0.5` looked reasonable in isolation but, checked
against a real training run over this repo's cached data (mean
`outcome*scale` ≈ 1.14 — the dataset really is that favorable on average),
still left the per-context optimum clipped to the max-size boundary for
99.4% of predictions — collapsing right back to the same degenerate policy
the penalty exists to prevent. `risk_aversion=1.0` is the smallest value
verified to actually break that saturation (real run: held-out mean reward
0.374 vs. an always-max-size baseline of 0.148, sizes ranging -0.76 to 1.0,
correlated with risk_reward). Confirmed the mechanism itself was sound
*before* re-calibrating, on a controlled synthetic dataset with a genuine
quality→outcome correlation: sizing came out strongly correlated with
quality (r=0.89), negative on low-quality contexts, near-max on high-quality
ones — exactly the intended "size by conviction" behavior. No ONNX export
here — SB3's action-sampling wrapper around the underlying network isn't a
plain forward pass, so consuming a saved policy still requires
stable-baselines3 itself, documented as a known follow-on constraint, not
solved in this pass. `scripts/train_pattern_ppo.py` mirrors the other two
scripts' CLI structure. `tests/test_ppo_sizer.py` — 11 tests, all passing
under `.venv-ml`.

**Dependency cascade found while running both training scripts against
`--data-source cache`:** `firm.runtime.load_prices` transitively imports the
full backtest engine, which pulled `pyyaml`, `pydantic`, `pyarrow`,
`pydantic-settings`, and `backtrader` into `.venv-ml` one missing-import at a
time. Installed each rather than building a lighter-weight cache loader, for
consistency with `train_pattern_ml.py`'s existing approach.

**Explicitly out of scope, same as the original plan:** wiring either model
into the live signal-generation path. Both stay standalone, human-run
research tools producing artifacts on disk (`data/models/pattern_cnn.*`,
`data/models/pattern_ppo.zip`, all `.gitignore`d).

### 7.5 Push + enable live (done)

Pushed all accumulated commits to `origin/main` (`da08323..c4aa561`).

**Live config isn't one file:** the two production instances read different
YAML files — `:8000` (IBKR paper) reads the default `config/live.yaml`;
`:8001` (Alpaca paper) reads `config/live_alpaca.yaml` via its
`FIRM_LIVE_CONFIG` env override (confirmed via `/proc/<pid>/environ`, per
the plan's explicit caveat to check this before assuming one edit covers
both). Added `pattern_recognition` to `strategies.enabled` *and*
`auto_approve` in both files (both instances run `approval_mode: full_auto`,
where `auto_approve` has no functional effect today, but kept consistent
with every other strategy's entry in case that ever changes).

Hot-swapped into both running engines immediately via `PUT /api/live/config`
(`LiveEngine.update_strategies()` → rebuilds the orchestrator, effective
next cycle, no restart) rather than waiting for the next service restart to
pick up the file change — confirmed via `GET /api/live/status` on each
(`active_strategies` includes `pattern_recognition`) and each instance's own
log line (`Live engine strategies updated: [...,
'pattern_recognition']`, `engine.py:427`). Both PIDs (`3682961` Alpaca,
`3683272` IBKR) unchanged throughout — no restart, no broker
reconnect, no interruption to either engine.

## 8. 2026-09-25 improvement round — quick-wins, validation-rigor, model-ensemble-sizing, detection-quality

A second research pass (web research on quality-scoring/validation
best-practice, run in parallel across 4 subagents, then implemented as 4
more parallel subagents with strictly disjoint file ownership + a final
integration pass by the orchestrating agent) targeting four areas: cheap
wins, rigor of the CNN-toggle validation, model ensembling/sizing, and raw
detection quality. **Every new knob introduced in this round defaults to its
old, unchanged behavior** — same convention as `cnn_scoring_enabled`
(§4/config comments): nothing here is live in `config/live.yaml` or
`config/live_alpaca.yaml` yet, pending each knob's own dedicated
walk-forward re-validation (the exact harness this round built for
`cnn_scoring_enabled` — §8.2 — is the template to repeat per-knob before
flipping any of them true).

### 8.1 Quick wins

- **`FIRM_ENABLE_PATTERN_SCAN` enabled on the Alpaca instance only**
  (`deploy/ai-trading-alpaca.service`, not the shared `.env` — see that
  file's inline comment for why one instance, not both). Deployed via
  `cp` to `/etc/systemd/system/` + `daemon-reload` + `restart`; confirmed
  healthy afterward (`GET :8001/api/health` → `live_engine_running: true`)
  and the job itself started cleanly (`PatternScanJob started: daily 16:30
  ET scan of 35 symbols (data_source=cache)` in the unit's journal). Purely
  additive — this existing, previously-unused feature (§7.2) was simply
  switched on so `pattern_scan_history.db` starts accumulating real
  rule-based `quality_score` + eventual target/stop/timeout outcome rows for
  `scripts/analyze_pattern_scan_outcomes.py` (below) to mine once enough
  history has built up. Zero effect on either live engine's real trading —
  see `pattern_scan_job.py`'s own docstring for why it's inert by
  construction.
- **`scripts/analyze_pattern_scan_outcomes.py`** (new, 30 tests) — CLI
  hit-rate/correlation report over `pattern_scan_history`'s resolved
  (non-pending) rows: win rate by pattern/direction, `quality_score`-vs-
  outcome correlation, and a plain-language summary. Deliberately reads
  the *scheduled scan's own* rule-based-only history (see §8's correction
  below), not a nonexistent "paired CNN vs. rule-based shadow" dataset.
- **ATR-scaled ZigZag threshold** — `extrema.py`'s `zigzag_pivots` gained an
  optional `threshold_fn` hook (per-bar callable, e.g. `2 * atr[i] /
  close[i]`, falling back to the fixed `pct` on any invalid per-bar value);
  dependency-free (numpy-only), backward-compatible (`threshold_fn=None` is
  byte-identical to the old fixed-`pct` algorithm). Wired through
  `scanner.py`'s new `zigzag_atr_mult` param (`None` = old behavior; a
  positive float builds the ATR-scaled closure from the already-computed
  `atr14` series) and `pattern_recognition.py`'s new `zigzag_atr_mult`
  strategy param (also `None`/off by default). **Not yet flipped on** —
  needs its own walk-forward comparison against the fixed-`pct` baseline
  before it's config-enabled anywhere.
- **Correction to an earlier assumption** (recorded honestly rather than
  silently acted on): a prior session had assumed `Signal.meta`'s
  `rule_based_quality_fraction`/`cnn_quality_fraction` pair was already
  durably logged somewhere and just needed extracting for a CNN-vs-rule
  comparison. Re-checked this round by grepping every write path for
  `Signal.meta` — blackboard signals are ephemeral, per-cycle, in-memory
  only; nothing persists them. `pattern_scan_history` (the store the
  scheduled job above writes to) is a *separate*, rule-based-only scan, not
  the live engine's actual paired CNN-vs-rule signals. There is currently no
  durable paired dataset for a real CNN-vs-rule ablation — only the
  rule-based-only history `analyze_pattern_scan_outcomes.py` reads. Flagging
  this here rather than building a script against data that doesn't exist.

### 8.2 Validation rigor

- **`scripts/validate_pattern_cnn_walkforward.py`** (new) — reuses this
  repo's existing CPCV-equivalent tooling
  (`ExperimentRunner.run_walk_forward` + `param_grid` over
  `cnn_scoring_enabled: [False, True]` + `aggregate_walk_forward`'s
  automatic PBO/DSR/PSR/verdict via `firm.eval.overfitting`) rather than
  building a bespoke CPCV harness — mirrors `scripts
  /run_walk_forward_pbo_audit.py`'s established pattern. Confirmed the
  11-strategy roster/25-symbol universe match that script's exactly, so
  reused verbatim.
- **Real run launched** (2018-01-01 to 2025-12-31, 8 folds — genuine
  multi-regime coverage, not a shortened window): a timing check found one
  3-month/11-strategy/25-symbol backtest costs ~261s, and — because
  `_compute_walk_forward_splits`' total compute is dominated by the fixed
  date range rather than `n_splits` — the full 8-fold run is a multi-hour
  job (~2.5–3.5h estimated from observed per-fold pace), launched detached
  (`nohup`, survives the building session) rather than cut short.
- **Real result (completed 2026-09-25 08:07, ran ~6h): verdict = `fail`,
  recommendation = ROLLBACK.** Candidate selection per fold (train-window
  Sharpe, `cnn_scoring_enabled=True` vs. `False`) picked `True` in only
  3/8 folds (0, 1, 3) — `true_fraction=0.375`. Aggregate out-of-sample
  stats across all 8 folds' winning candidate: `sharpe_ratio` mean **-0.563**
  (std 0.908, range -2.18..+0.78), `alpha` mean **-0.017**, `total_return`
  mean **-0.0068**. Overfitting stats: **PBO 0.446** (technically clears the
  conventional <0.5 bar, barely), but **probabilistic Sharpe 0.216** (needs
  >0.95 to say the Sharpe is statistically distinguishable from zero) and
  **deflated Sharpe ≈2.8e-6** (i.e. ~0, after correcting for the two-candidate
  selection search) — the combined verdict logic in `eval/overfitting.py`
  requires clearing both PBO *and* DSR/PSR bars, and this doesn't. This is
  a real, negative answer to the exact question `config/live.yaml`'s own
  `cnn_scoring_enabled` comment flagged as unresolved ("One walk-forward
  window, not this codebase's usual 3-window robustness check — revisit if
  live results diverge"): the original single-window positive result
  (Sharpe 0.415→0.645) did not replicate under a genuine 8-fold multi-regime
  CPCV-style audit.
- **`scripts/pattern_recognition_rollout_gate.py`'s combined verdict differs
  from Task A's own standalone one, by design.** Task A's script prints its
  own recommendation from the backtest verdict alone (ROLLBACK, above).
  The staged gate additionally requires a live-sample-size floor (§8.2's
  design) before *ever* acting on live P&L, but — this is a real gap
  surfaced by actually running it, not a hypothetical — it also declines to
  act on a *failing backtest by itself* while the live sample is thin: with
  IBKR `:8000` at 0 days of `pattern_recognition` attribution history and
  Alpaca `:8001` at 11 days (both `<20`), the gate returns **HOLD** on both
  instances (`overall_recommendation: "HOLD (insufficient live sample,
  backtest verdict=fail)"`), not ROLLBACK, per `evaluate_rollout_gate`'s
  documented judgment call for a failing/non-passing backtest with an
  as-yet-unmet sample gate. Full output: `/tmp/pattern_cnn_walkforward_audit
  .json` (Task A), `/tmp/pattern_recognition_rollout_gate_result.json`
  (this gate, re-run against the real audit file above).
- **Decision (user, 2026-09-25): rolled back.** Presented both
  recommendations (Task A's ROLLBACK vs. the staged gate's more
  conservative HOLD-pending-live-data) to the user directly rather than
  silently picking one — this is a real live-trading-behavior change, not
  a code refactor. User chose to trust the backtest audit and roll back
  now rather than wait ~9 more days for the live-sample floor. Executed:
  `cnn_scoring_enabled: false` in both `config/live.yaml` and
  `config/live_alpaca.yaml` (comments updated in place citing this section),
  then hot-swapped into both running engines via `PUT /api/live/config`
  (`strategy_params`, same mechanism as §7.5) — confirmed via `GET
  /api/live/config` on each and `state: running`/unchanged `uptime_seconds`
  on `GET /api/live/status`, i.e. no restart, no broker reconnect on either
  instance. `pattern_recognition` itself stays enabled on both (rule-based
  scoring only, its pre-CNN baseline) — only the CNN quality-scoring layer
  is off now.
- **`scripts/pattern_recognition_rollout_gate.py`** (new, 18 tests) — the
  keep/hold/rollback staged-rollout gate: requires BOTH a passing backtest
  verdict (from the file above) AND a pre-committed live sample-size floor
  (≥20 trading days / ≥30 signals) before ever acting on live P&L, refusing
  to flip either direction on a noisy small sample. Run for real against
  both live instances: `:8000` (IBKR) has 0 days of `pattern_recognition`
  attribution history; `:8001` (Alpaca) has 11 days (2026-09-10→09-24,
  cumulative return −0.61%, rough Sharpe ≈ −1.25) — both correctly returned
  **HOLD (insufficient live sample)** rather than reacting to that thin,
  noisy 11-day number. The KEEP path was separately exercised end-to-end via
  a synthetic 22-day/passing-backtest fixture override, confirming the gate
  logic itself (not just its HOLD branch) is correct.

### 8.3 Model ensembling & sizing

- **`src/firm/patterns/ml/calibration.py`** (new) — two calibration
  techniques, one per model family (different miscalibration failure modes
  — boosted-tree probability compression vs. deep-net overconfidence):
  temperature scaling (`fit_temperature`/`apply_temperature`, 1-parameter,
  fit via bounded NLL minimization) for the CNN's pre-softmax logits, Platt/
  sigmoid scaling (`fit_sigmoid_calibration`/`apply_sigmoid_calibration`, via
  an unregularized 1-feature `LogisticRegression`) for XGBoost's raw
  per-class probability. Plus `save_calibration`/`load_calibration` JSON
  sidecar persistence (schema-agnostic, caller-decided file layout).
  `firm.patterns.ml.inference.score_pattern_quality` gained an optional
  `temperature: float = 1.0` param (no-op default, byte-identical for every
  existing caller) so a fitted CNN calibration can actually be applied at
  inference time.
- **`src/firm/patterns/ml/xgb_inference.py`** (new) — live-path ONNX
  inference for the already-trained XGBoost pattern-confirmation classifier,
  mirroring `ml/inference.py`'s CNN wrapper exactly (same `is_available`/
  fail-soft/warn-once/lazy-`lru_cache` shape). **Better outcome than
  originally assumed possible:** `xgb_classifier.py`'s own ONNX path is
  documented `.venv-ml`-only (no Python 3.14 `onnxruntime` wheel at write
  time) — re-checked this round and a genuine `cp314` wheel now exists
  (confirmed importable in the main venv), so this module runs in the main
  venv, always-on, fully tested, no environment gating needed. Verified the
  on-disk model's real ONNX graph shape (`input: [None, 47]`) against
  `feature_engineering.build_features`'s dict-insertion order to pin down
  the exact, previously-undocumented 47-feature column layout the trained
  model expects (documented in this module's docstring — load-bearing for
  anyone retraining the model, since a schema drift here would silently
  produce meaningless scores with no self-detection possible).
- **XGBoost wired into `pattern_recognition.py` as a fixed-weight blend +
  agreement gate** (new `xgb_confirmation_enabled`/`xgb_blend_weight`
  /`xgb_agreement_gate`/`xgb_agreement_gate_threshold`
  /`xgb_agreement_gate_dampen` params, all off/no-op by default): for each
  symbol's best match, `build_features()` feeds the already-trained ONNX
  model, and its calibrated `P(target hit)` is blended with the
  CNN-or-rule-based quality fraction (`blend_weight * p_target + (1 -
  blend_weight) * base_quality`); when the two disagree by more than
  `xgb_agreement_gate_threshold`, the blend is *dampened*
  (`* xgb_agreement_gate_dampen`) rather than trusted outright — two
  independently-trained models agreeing is materially stronger evidence
  than either alone, so a sharp disagreement should reduce confidence, not
  just average it away. `p_target` is also written to the new
  `Signal.meta["calibrated_probability"]` key (only when the ensemble
  actually scored that match) for `TraderAgent._kelly` to consume directly
  (below) — this is the "meta-labeling" probability the Kelly formula wants,
  not a re-derivation of it.
- **PPO position sizer retirement**: no code changes (it was already
  standalone/unwired — §7.4), but the research recommendation to *not*
  invest further in it is recorded here rather than left as tacit
  knowledge — extend the existing, already-live Kelly sizer (below) instead
  of building a second, competing sizing mechanism with its own SB3 serving
  dependency.
- **`TraderAgent._kelly` extended** (`src/firm/agents/trader.py`) — new
  optional `blackboard` param (`None` = byte-identical legacy behavior for
  every existing caller/test) and a new `_signal_calibrated_edge` static
  method: for each selected symbol, looks up that symbol's live signals via
  `blackboard.get_signals_by_symbol` and, when one or more carry
  `meta["calibrated_probability"]`, converts it to a Kelly-style edge
  directly (simple-averaged across multiple agreeing strategies) instead of
  estimating `p`/`b` from realized return history — a genuine meta-labeling
  upgrade path, falling back to the original `_kelly_edge` history-based
  estimate whenever no signal for that symbol carries the convention
  (i.e. unchanged behavior for every strategy except `pattern_recognition`
  with the XGBoost ensemble enabled).

### 8.4 Detection quality

- **`scorer.py`'s two new optional components** — `breakout_distance`
  (0-10, ATR-normalized breakout-bar distance past the pattern's own
  structural level, *at the confirmation bar's own contemporaneous ATR* —
  deliberately distinct from the existing `follow_through`, which uses
  *current* ATR and can reflect drift long after confirmation) and
  `pre_breakout_compression` (0-5, credit for ATR(14) being compressed vs.
  its trailing 20-bar average just before confirmation — compressed-
  volatility breakouts are documented as more credible). Weights rebalanced
  (30/15/20/10/10/10/5 = 100, from 35/20/25/10/10) so `min_score` thresholds
  tuned against the old 5-component scale stay roughly comparable. Both are
  driven by new *optional* kwargs (`close_at_confirm`/`level_at_confirm`
  /`atr_series`/`confirm_index`) — omit all four and both components
  contribute a documented, tested 0.0, never a crash.
- **`confirmation.py`'s retest-hold-vs-fail modifier** (`retest_outcome`/
  `retest_score_modifier`) — a coarse, OHLCV-only read of whether price held
  or failed on coming back to test a just-broken level within a lookback
  window (`"no_retest"` — no pullback at all — is the common, perfectly
  fine ~50% case, not a failure). A small, bounded (`+1.0`/`-1.0`/`0.0`)
  scoring modifier, deliberately never a hard gate (hard-requiring a retest
  would forfeit the ~50% of setups that never retest with no clear benefit).
- **New `confluence.py`** — weekly-timeframe trend confirmation
  (`resample_to_weekly`/`weekly_trend_direction`/`confluence_modifier`).
  `resample_to_weekly` *unconditionally* drops the trailing (possibly
  still-forming) week — the single most important correctness property here
  (a 2025 multi-timeframe study found ~0.20 ROC-AUC inflation from exactly
  this kind of look-ahead leak) — so nothing downstream can ever see a
  partially-formed week regardless of caller/as-of context. Same
  small-bounded-modifier design as the retest modifier above, never an
  independent signal (would double the effective bet count on the same
  underlying move without separate validation).
- **Both modifiers wired into `scanner.py`** (`retest_modifier_enabled`
  /`retest_lookback_bars`/`retest_modifier_scale`,
  `confluence_modifier_enabled`/`confluence_lookback_weeks`
  /`confluence_modifier_scale` — all off/no-op by default) and
  `pattern_recognition.py`'s matching strategy params. Each modifier's
  scaled value is added to `PatternMatch.quality_score` and the result
  re-clipped to `[0, 100]`; `score_breakdown` gains `retest_outcome`
  /`retest_modifier` and/or `weekly_trend`/`confluence_modifier` keys only
  when the corresponding modifier actually ran (never present when off —
  verified by test, not just by construction).
- **Harmonic patterns (Gartley/Butterfly/Bat/Crab)**: explicitly skipped,
  as originally planned — Fibonacci-ratio-based patterns are a different
  detection paradigm (ratio-of-swing matching, not zigzag-pivot geometry
  fitting) that would need its own dedicated rule module, not a small
  addition to this round.

### 8.5 Integration pass (this agent, not a parallel subagent)

`scanner.py`/`pattern_recognition.py` were deliberately reserved from every
parallel subagent above specifically so one agent could wire everything
together without merge conflicts on a shared (non-worktree) checkout. This
pass:

- Reordered `scan_symbol` to compute `atr14` *before* `zigzag_pivots` (needed
  either way once ATR-scaled thresholds exist) and threaded the full ATR
  series + `close_at_confirm`/`level_at_confirm`/`confirm_index` into
  `_score_and_finalize`'s `score_pattern` call — previously these four new
  optional args were simply never passed, so `breakout_distance`/
  `pre_breakout_compression` silently contributed 0.0 for every real match
  (this is what caused the regressions below).
- Wired `zigzag_atr_mult`, the retest modifier, and the confluence modifier
  (needs a separate `dates` array — `pattern_recognition.py`'s adjusted
  OHLCV frame has no date column of its own, so `scan_symbol` gained a
  `dates` param threaded from the per-symbol `sym_df["date"]`) all the way
  from `PatternRecognitionStrategy.default_params` through `scan_symbol`.
- Wired the XGBoost ensemble + both calibration paths into
  `pattern_recognition.py`'s `generate()` — feature-vector construction via
  `build_features`, calibration sidecar loading, the blend/agreement-gate
  formula, and `meta["calibrated_probability"]` population (§8.3).
- **Fixed the two flagged pre-existing test regressions** (both correctly
  diagnosed by the parallel subagents as caused by `scanner.py` not yet
  passing the new scorer args, confirmed via `git stash`, and deliberately
  left for this pass): `test_patterns.py`'s clean-setup test now also
  exercises the two new components directly (was silently capped at an
  effective max of 85 — see scorer.py's own documented backward-compat
  contract; added a companion test pinning that 85-cap behavior explicitly)
  and `test_patterns_api.py`'s two hardcoded match-count/score assertions
  were re-verified against the real, now-fully-wired `scan_symbol` output
  and updated in place (8 matches not 7; MSFT's falling_wedge ≈97.4 not
  ≈98.8 — both *higher-fidelity* scores now that the two new components are
  actually fed real data, not a regression in the scoring logic itself).
- Added 6 new integration tests exercising the actual wiring end-to-end
  (not just each module in isolation, already covered by the parallel
  subagents' own unit tests): `zigzag_atr_mult` genuinely changes pivot
  detection (an absurdly large multiple confirms zero pivots), the retest/
  confluence modifiers apply their scaled adjustment and populate
  `score_breakdown` only when enabled, confluence degrades gracefully with
  no `dates` supplied, and the XGBoost ensemble's blend+gate formula and
  `calibrated_probability`/`scoring_mode` population match hand-computed
  expected values (both when enabled and confirming it's a true no-op when
  off).
- Full patterns/trader/tooling suite after integration: `pytest tests
  /test_patterns.py tests/test_patterns_api.py tests/test_pattern_ml.py
  tests/test_extrema.py tests/test_scorer.py tests/test_confirmation.py
  tests/test_confluence.py tests/test_calibration.py tests
  /test_xgb_inference.py tests/test_trader.py tests/test_trader_kelly.py
  tests/test_pattern_recognition_rollout_gate.py tests
  /test_analyze_pattern_scan_outcomes.py -q` → **289 passed, 9 skipped**
  (skips are the same `.venv-ml`-only-gated tests as every prior phase).

**Not yet done, tracked explicitly rather than silently dropped:** the
§8.2 walk-forward audit is still running as of this writing (final PBO/DSR
/PSR/verdict numbers pending); none of this round's new knobs
(`zigzag_atr_mult`, `retest_modifier_enabled`, `confluence_modifier_enabled`,
`xgb_confirmation_enabled`) have been flipped on in `config/live.yaml`
/`config/live_alpaca.yaml` yet — each needs the same walk-forward
re-validation treatment `cnn_scoring_enabled` already got before that
happens.

## 9. 2026-09-25 follow-up: guardrails, calibration wiring, golden benchmark

An adversarial review of §7-§8's history (commits `53d5345`..`a8bb88f`)
found the core detection/scoring logic causally sound and the §8.2 rollback
methodologically justified, but surfaced concrete gaps: no automated
control stopped a future config edit from re-enabling an
audited-and-failed knob; `meta["calibrated_probability"]` was populated
from raw, uncalibrated XGBoost output whenever no calibration file was
configured (which was always, since nothing ever produced one); the
walk-forward validator's own grid was self-admittedly stale; and there was
no regression test for the look-ahead invariant §5/§6 rely on being true.
This section tracks closing those gaps.

### 9.1 Small correctness fixes

- `pattern_recognition.py`'s module docstring still claimed "Registered
  but NOT added to config/live.yaml's strategies.enabled list" — false
  since §7.5/`c4aa561` (2026-09-09), never updated across two subsequent
  edits to the same file. Corrected.
- `trendline.fit_trendline`/`fit_poly2`: a degenerate flat-line input
  (`ss_tot <= 1e-12`, nothing for the fit to explain) returned `r2 = 1.0`
  ("perfect fit") rather than `0.0` — inflates `trendline_fit`/`geometry`
  scoring for a pathological flat/illiquid symbol. Fixed; new
  `tests/test_trendline.py`.
- `confluence.resample_to_weekly` never de-duplicated timestamps before
  `.agg({"volume": "sum", ...})` — a duplicate calendar date (upstream data
  bug) would silently double-count that day's volume. Fixed
  (`keep="last"`); regression test in `tests/test_confluence.py`.

### 9.2 Look-ahead / prefix-invariance regression tests (new coverage)

New `tests/test_pattern_invariance.py`. Manually re-verified the invariant
(appending observations after decision time T must not change any result
emitted at or before T) function-by-function, same conclusion as before:
holds for `zigzag_pivots`, the score components that read only up to
`confirm_index` (`volume_ratio`, `breakout_distance`,
`pre_breakout_compression`), and every rule detector.

Two genuine subtleties surfaced while writing the tests, worth recording
since they'd otherwise cause a naive "append and compare" test to fail for
the *wrong* reason:

- `_score_and_finalize` floors `stop` at `stop_atr_floor * current_atr`,
  where `current_atr = atr[-1]` — the last bar of the *whole input window*,
  not the ATR as of `confirm_index`. Intentional (the stop reflects
  *current* volatility, not volatility when the pattern formed) — no past
  emitted signal is ever altered since live trading never re-scans history
  — but `stop`/`risk_reward`/`follow_through_atr`/`quality_score` are
  consequently not prefix-invariant *by design* and had to be excluded
  from the structural-equality checks.
- `confirmation.find_confirmation` searches newest-first and returns the
  *freshest* bar still satisfying the breakout condition — so
  `confirm_index` deliberately drifts forward as more bars are appended,
  for as long as the breakout persists (the same mechanism §6.8 already
  documented for training-data collection, seen here from the opposite
  side: it's also why a bare `scan(data[:T])` vs `scan(data[:T+N])`
  comparison is the wrong invariance test for `scan_symbol` — the correct
  one truncates at the *longer* run's own reported `confirm_index` and
  checks nothing after it mattered).

### 9.3 Fail-closed rollout-gate enforcement

New `src/firm/live/pattern_ml_gate.py`. `scripts/pattern_recognition_rollout_gate.py`
was a real, tested decision function with zero callers outside its own
test — nothing stopped `cnn_scoring_enabled`/`xgb_confirmation_enabled`
from being flipped `true` without ever running it. The new module
refuses to honor either flag in `strategy_params.pattern_recognition`
unless a rollout-gate JSON with `overall_recommendation` starting `"KEEP"`
exists at `data/models/pattern_recognition.rollout_gate.json`
(`FIRM_PATTERN_ML_GATE` overrides), forcing the flag back to `False` and
logging `REFUSED: ...` with the remediation command otherwise. Wired into
`LiveTradingEngine.__init__` and `_rebuild_orchestrator` — the true funnel
for every live start/hot-swap path (systemd/API startup, `POST
/api/live/start`, `scripts/run_live_trading.py`, and all six existing
`update_*` setters) — and deliberately *not* into
`firm.runtime._build_categorized_strategies`, which the walk-forward
validator itself shares to produce the very record this gate reads (guarding
there would deadlock the harness). `_rebuild_orchestrator` now owns
`self._config` after sanitizing it, closing a race where an `update_*`
caller that stashed the raw (pre-sanitize) request into `self._config`
before or after the rebuild could have silently un-done the refusal.
Tests: `tests/test_pattern_ml_gate.py` (pure decision function + a
`LiveTradingEngine`/`update_strategy_params`/`update_strategies`
integration suite with a mocked orchestrator).

### 9.4 Calibration wiring

`meta["calibrated_probability"]` — the field `TraderAgent._signal_calibrated_edge`'s
Kelly path trusts as an already-calibrated probability — was populated
from `xgb_p_target` unconditionally whenever the XGBoost ensemble ran,
regardless of whether `xgb_calibration_path` was configured (it never was
— nothing in the repo had ever called `firm.patterns.ml.calibration.fit_sigmoid_calibration`
outside its own test, so the field was silently uncalibrated in practice).
Fixed: `pattern_recognition.py` now only populates it when a calibration
file of type `"sigmoid"` actually loaded; `xgb_p_target` remains exposed
under its own honest key either way. `TraderAgent._signal_calibrated_edge`
already handled a `None` value safely (`float(None)` raises `TypeError`,
caught, signal skipped, falls through to the return-history Kelly path) —
confirmed by reading it, no change needed there, new test added for the
explicit case (`tests/test_trader.py::test_none_probability_skipped`).

New `scripts/fit_pattern_calibration.py` — the fitting pipeline this gap
was really missing. `--source history` reads real resolved rows from
`PatternScanHistoryStore` (refuses below 20 decisive rows, same bar as
`analyze_pattern_scan_outcomes.py`'s `_MIN_RESOLVED_FOR_STATS` — as of this
writing both live DBs have **zero** resolved rows, so this path correctly
reports "not enough data" rather than fabricating a calibration).
`--source synthetic` exercises the same fit/save wiring against a labeled
sample built from `firm.data.synthetic.make_synthetic_prices` — and hit
the *exact same* §6.8 single-shot-scan-has-no-forward-history bug while
building it (every label came back `0`/timeout until the scan was rolled
backward through history the same way `build_dataset` does), which is
reassuring evidence that §6.8's fix is the correct general pattern for
this failure mode, not a one-off. Only sigmoid/Platt calibration is
supported (temperature scaling needs per-class logits nothing currently
logs — CNN calibration isn't wired by this script; use
`scripts/train_cnn_validator.py`'s own held-out split instead until a
future schema change logs CNN logits). Tests:
`tests/test_fit_pattern_calibration.py`.

### 9.5 Time-ordered train/test split

`scripts/train_pattern_ml.py`/`scripts/train_cnn_validator.py` both used a
random `sklearn.train_test_split` on data §6.8 already established is
"highly autocorrelated" (overlapping rolling-cutoff windows) — a random
shuffle split lets near-duplicate rows from adjacent cutoffs of the *same*
underlying pattern leak between train and test, inflating reported
accuracy/AUC. Both scripts now record `confirm_date` (the source panel's
actual date at `confirm_index`, positionally valid against `ohlcv` per
§6.8) in their `meta` output and split via a new `time_ordered_split`: sort
by `confirm_date`, take the trailing `--test-size` fraction as test, with
an `--embargo-bars` (default = `--timeout-bars`) calendar-day gap dropped
from the train period immediately before the cutoff, mirroring
`validate_pattern_cnn_walkforward.py`'s own `--embargo-days` convention.
Duplicated rather than shared between the two scripts (no cross-script
imports in this repo; `train_cnn_validator.py`'s `X` is an image ndarray,
`train_pattern_ml.py`'s is a feature DataFrame, so the indexing differs
slightly anyway). `--seed` stays in both CLIs for backward compatibility
but is now a no-op for splitting, documented as such. New
`tests/test_train_pattern_ml.py`/`tests/test_train_cnn_validator.py` — the
first tests of either script at all (previously only the underlying
`firm.patterns.ml` primitives they call were tested).

### 9.6 Golden pattern benchmark + classification-metrics harness

Nothing anywhere in this repo previously measured whether the rule-based
detectors' positive/negative calls, or `quality_score` as a probability
proxy, are actually any good — only whether the pipeline runs without
raising. New `src/firm/eval/classification.py` (thin `sklearn.metrics`
wrappers — precision/recall/F1/confusion-matrix, PR-AUC, Brier score,
reliability-diagram binning; exported from `firm.eval`'s `__all__`), new
`tests/pattern_fixtures.py` (a shared, cross-importable golden corpus —
positive/negative/boundary/ambiguous fixtures, honestly scoped: 15 of 17
`PATTERN_NAMES` get a clean single-pattern positive fixture reused
verbatim from `tests/test_patterns.py`'s own proven anchors; the remaining
continuum pairs — bull_flag/pennant, cup_handle/rounding_bottom — are
represented as ambiguous fixtures instead of forcing a fake clean split;
boundary coverage is 2 representative cases, not all 9 detectors'
tolerance constants — tracked as a real gap, not silently claimed as
full), and new `scripts/benchmark_pattern_detectors.py` (runs every
fixture plus a larger synthetic-GBM-noise negative sample through the
real `scan_symbol` at its own live default `min_score=60.0`, reports
per-category/per-family accuracy plus a corpus-wide binary classification
report + PR-AUC + Brier score treating `quality_score/100` as a predicted
probability).

**Two genuine findings surfaced by building this, not designed in:**

1. `tests/test_patterns.py`'s existing triple_top/triple_bottom fixtures,
   when run through the *full* `scan_symbol` pipeline (every detector,
   best-quality-score wins) rather than `detect_triple_top`/
   `detect_triple_bottom` called in isolation, confirm a genuine
   triple_top/triple_bottom (score ~90) but `head_shoulders_top`/
   `inverse_head_shoulders` score marginally higher (~95-96) on the same
   pivots and win the "best match" slot. Both are real, independently
   correct detections on the same geometry — a real-pipeline ambiguity a
   per-detector unit test structurally cannot see (moved to this corpus's
   AMBIGUOUS category rather than treated as a corpus bug).
2. **At the live production `min_score=60.0` default, ~2/3 of pure
   random-walk (GBM) noise symbols still produce a "confirmed" pattern**
   (14 of 21 in one 20-symbol/300-day run — reproduce with
   `python scripts/benchmark_pattern_detectors.py --n-negative-symbols 20`).
   This is the first quantified false-positive-rate measurement for this
   subsystem and a direct, concrete answer to "is `quality_score`
   actually meaningful" — worth a follow-up investigation in its own
   right (P1 candidate: is 60 simply too low a bar, or is `quality_score`
   not discriminating GBM-noise geometry as well as its component design
   intends).

Tests: `tests/test_eval_classification.py`, `tests/test_pattern_fixtures.py`,
`tests/test_benchmark_pattern_detectors.py` (the latter tests the pure
grading/aggregation logic against controlled inputs — deliberately not
"does the corpus pass today," since real detector behavior legitimately
shifting *is* the point of running this benchmark, not something a green
test suite should mask).

### 9.7 Live attribution finding: winner-take-all P&L crediting hides small-magnitude strategies (cross-cutting, not pattern-recognition-specific)

While chasing why the rollout gate saw zero live-attributed days for
`pattern_recognition` on the IBKR (blended) instance despite it being live
since `c4aa561` (2026-09-09), traced the actual mechanism rather than
assuming "not enough time has passed":

- **`pattern_recognition` is genuinely contributing every cycle.** The
  persisted decision log (`data/memory/decisions.jsonl`) shows it with
  real, nonzero per-symbol weight contributions in 42 of the last 100
  logged decisions, starting the day after it went live.
- **But `ExecutionAgent._dominant_strategy_by_symbol`
  (`src/firm/agents/execution.py`) tags each *executed order* with
  whichever single strategy contributed the largest-magnitude weight to
  that symbol that cycle — winner-take-all.** `PerformanceAttribution.record_trades`
  (`src/firm/portfolio/attribution.py`) only ever credits that one tagged
  strategy with 100% of the fill's shares. `pattern_recognition`'s
  contributions run ~0.004–0.02 in magnitude, consistently smaller than
  the strategies that end up dominant on the same symbols (e.g.
  `multi_factor` at 0.065 on the same symbol, same cycle) — so it has
  apparently never once been the largest contributor for a symbol that
  actually traded on IBKR. Zero dominant-strategy wins → zero fills ever
  recorded under its name → zero attributed days, regardless of how real
  or how long-running its actual contribution to the blended decision is.
- This is a genuine measurement blind spot, not specific to
  `pattern_recognition`: **any** strategy whose signal is real but
  consistently smaller in magnitude than its peers for the symbols that
  end up trading is structurally invisible to every consumer of
  per-strategy attribution — the Strategy Performance page, the rollout
  gate's live-sample check, reflection/lessons-learned. Confirmed
  `PerformanceAttribution` is a plain in-memory object with no
  persistence for blended mode (unlike sleeved mode's durable NAV
  history), so this also silently resets on every restart — a second,
  smaller compounding factor, not the root cause.

**Fix applied** (same session, both live and backtest paths — `ExecutionAgent`/
`PerformanceAttribution` are shared by `LiveTradingEngine` and
`BacktestEngine` via the same `Orchestrator.step()`):

- New `ExecutionAgent._fractional_strategy_weights_by_symbol`: maps each
  traded symbol to `{strategy: fraction}` (proportional to `abs(weight)`,
  summing to 1.0 across every strategy that contributed nonzero weight),
  attached to each order as a new `"strategy_weights"` field — additive,
  alongside the existing `"strategy"` tag, which is left untouched (still
  winner-take-all, correctly: an approval-queue order still needs exactly
  one human-facing owner).
- `PerformanceAttribution.record_trades` now splits a fill's shares
  proportionally across every strategy in `"strategy_weights"` when
  present, instead of crediting 100% to the single `"strategy"` tag.
  Backward compatible: any fill without the new field (old callers,
  hand-built test fixtures) behaves exactly as before. Conservation holds
  by construction (fractions sum to 1.0, so the split's total still equals
  the fill's real share count).
- Held/closing-position fallback (a symbol with no this-cycle
  `per_strategy` entry) now also populates `strategy_weights` (100% to the
  held-position strategy), keeping the two fields consistent in that path
  too.
- Tests: `tests/test_agents.py` (new fractional-weights-on-order cases,
  alongside the existing dominant-tag regression tests) and
  `tests/test_eval.py` (new `PerformanceAttribution` cases: proportional
  split, conservation, backward-compat fallback, and the concrete
  "smaller contributor gets a measurable return" scenario).

Not done: no attempt to make blended-mode attribution durable across
restarts (a separate, larger change); no re-check of whether Alpaca's
sleeved-mode attribution has the same winner-take-all blind spot for its
*inter-sleeve* reporting (sleeved mode capital is already segregated per
strategy, so this specific mechanism may not even apply there — not
verified this session).

### 9.8 Deferred (not done in this pass)

- The plan's secondary defense-in-depth gate check in
  `provider_utils.resolve_live_startup` (surfacing a refusal as an earlier
  HTTP 400 rather than a silent downgrade visible only in
  `GET /api/live/config` + logs) was scoped as optional and skipped — the
  primary `LiveTradingEngine` guard already covers every live path and is
  tested; a second raise-based check in a broadly-used helper wasn't worth
  the added risk for a UX nicety.
- Exhaustive per-detector boundary-tolerance fixtures (all 9 detectors'
  individual tolerance constants, not just the 2 representative cases
  above).

### 9.9 CNN/XGBoost retrain + extended walk-forward audit (2026-09-26/27, done)

Retrained both models on real cached data (2010-2025, 25 symbols) with the
new time-ordered split (§9.5): CNN — 5,869 confirmed-pattern images, train
acc 0.655, test acc 0.697; XGBoost — same dataset, train acc 0.848, test
acc 0.753, test AUC 0.873. Both exported to ONNX and smoke-tested through
the real live-inference path (`firm.patterns.ml.inference`/`xgb_inference`)
before anything else touched them. This closes the review's top finding —
the prior CNN artifact's training-data provenance was unconfirmed; this
one is fully reproducible from a known command and known data range.
Backed up the pre-existing (2026-09-20) artifacts to
`data/models/backup_20260926/` first.

Environment notes for future re-runs: `torch` (2.14.0) and `onnxruntime`
(1.30.0) both have working Python 3.14 wheels now (the repo's various
"no 3.14 wheel" claims are stale — see §9.5's `train_cnn_validator.py`
docstring update). Installed `xgboost`, `onnxmltools`, `skl2onnx`, and
`onnxscript` (torch 2.14's `torch.onnx.export` now routes through it) into
the main venv — all resolved clean wheels, no environment blockers. `pyts`
remains uninstalled and unnecessary (§9.5's numpy GASF replica covers the
only configuration actually used).

Extended `scripts/validate_pattern_cnn_walkforward.py`'s `DEFAULT_PARAM_GRID`
to 3 candidates (rule-based baseline / CNN / XGBoost, XGBoost isolated
from the CNN toggle) and generalized its recommendation logic to report
independently per flag (§ commit `15d8b17` — the old CNN-only reading
would have silently conflated a rule-based-baseline fold-win with an
XGBoost-candidate fold-win, since both have `cnn_scoring_enabled: False`).

**Result** (`data/models/pattern_recognition.walkforward_audit.json`,
2018-01-01 to 2025-12-31, 8 folds, ~4h20m wall-clock on 4 pinned cores):

| Metric | Value | Bar |
|---|---|---|
| PBO | 0.714 | fail (need < 0.50) |
| Deflated Sharpe | 1.51e-05 | fail (need > 0.95) |
| Probabilistic Sharpe | 0.445 | — |
| **Verdict** | **fail** | |

Both per-flag recommendations: **ROLLBACK** (`cnn_scoring_enabled=false`,
`xgb_confirmation_enabled=false`) — no change to production, since both
were already `false`. This is a fresh, independent confirmation with a
provenance-known CNN artifact and a real (never-before-validated) XGBoost
walk-forward result, not a re-hash of the 2026-09-25 finding: the CNN
result got *worse* on retrained data (PBO 0.714 vs. the prior 0.446), and
XGBoost — validated for the first time ever — fails the same bar just as
clearly. Ran `scripts/pattern_recognition_rollout_gate.py` against both
live instances: `overall_recommendation = "HOLD (insufficient live
sample, backtest verdict=fail)"` (IBKR 0 attributed days, Alpaca 1 —
consistent with §9.7's finding, not yet affected by that fix since it
requires a live-service restart to pick up). Confirmed end-to-end: `firm.live.pattern_ml_gate.evaluate_gate`
against this real record returns `{"allowed": False, ...}` — the fail-closed
gate (§9.3) correctly refuses either flag.

**Conclusion: neither ML enhancement is ready for live use.** Nothing in
production changes as a result (both flags were already off); this
round's value is the fresh, reproducible evidence base plus the guardrail
that now enforces it going forward.

## 10. 2026-09-27 false-positive-rate fix (Part A) + real meta-labeling infrastructure (Part B)

A "very deep research" pass (finance/quant-specialist framing) into two
findings — "the detector flags ~2/3 of random noise as a real pattern under
current settings" and "make the ML layer actually worth turning on" —
produced a research-and-engineering plan
(`/root/.claude/plans/typed-splashing-moonbeam.md`) executed with full
autonomy through Parts C/A/D1/B (pausing only before any new paid-data
spend, per Part D5). This section is the implementation log; commits
`e688f90`..`1d44093` on `main`.

### 10.1 Part A — false-positive-rate fix (7 items, all shipped)

Reframed `quality_score` from an arbitrary heuristic toward calibrated
evidence strength, with a hard, CI-enforced floor going forward:

1. **Removed double-counted scorer components** (`follow_through`/
   `breakout_distance` measured the same underlying quantity through
   different ATR denominators) and **fixed the non-discriminative
   `duration` component** (noise scored 10/10 on it essentially always) —
   `scorer.py`/`scanner.py`. Freed points redistributed toward volume
   confirmation (the only component that already discriminated noise).
2. **Statistical significance test** (`firm.patterns.significance`) —
   bootstraps a null score distribution from the symbol's own realized
   returns (reusing `firm.eval.robustness.MonteCarloAnalyzer`), reports a
   p-value instead of trusting the raw score.
3. **Benjamini-Hochberg FDR control** across every candidate a scan cycle
   produces — the missing correction for scanning ~35 symbols x 9
   detectors x multiple windows and keeping the max score per symbol.
4. **Per-pattern minimum-sample confidence discount**
   (`firm.patterns.sample_size`) — a pattern with fewer than
   `min_reliable_samples` (default 30) historical confirmations gets its
   confidence shrunk proportionally, rather than trusted at
   `rising_wedge`-level (1,351 examples) face value.
5. **Regime-aware confidence discount** — builds its own
   `MarketRegimeDetector` instance (strategies never receive
   `ctx.market_regime`) and discounts a pattern whose direction conflicts
   with the labelled regime, or any pattern in Chop, damped toward a no-op
   when the label itself was thin-margin/untrustworthy
   (`RegimeState.separation`).
6. **Golden benchmark promoted to a hard CI release gate**
   (`tests/test_benchmark_pattern_detectors.py::TestGoldenBenchmarkReleaseGate`)
   — pre-registered floors (recall=1.000 no tolerance, baseline FPR<=30%,
   Brier<0.25, PR-AUC>=0.85, significance+FDR-adjusted FPR<=15%), verified
   genuinely live (a tightened threshold was confirmed to actually fail).

**Net measured result** (`scripts/benchmark_pattern_detectors.py`, 37
fixtures + 20 synthetic-noise symbols, seed=42, reproducible):
synthetic-noise false-positive rate **70% -> 19.0%** (scorer fix alone,
items 1-2 — this is what ships live today, since items 2-5's own knobs
default off pending their own walk-forward re-validation, matching every
other optional knob in this strategy) **-> 4.8%** (with the opt-in
significance+FDR layer, items 3-4, on top). Recall stayed 100% throughout
— no positive/ambiguous/boundary fixture regressed. Brier 0.269 (worse
than an uninformative 0.25 forecaster) -> 0.073; PR-AUC 0.481 -> 0.942.

### 10.2 Part B — meta-labeling infrastructure (items 1-6 of 9, shipped; items 7-9 in progress)

De Prado's meta-labeling recipe (*Advances in Financial Machine Learning*)
was ~50% built (a real primary/secondary model split already existed) but
missing every piece that makes it statistically honest:

1. *(done via Part C — see the calibration-model-mismatch fix,
   `522c44a`.)*
2. **Separate binary act/no-act target**
   (`firm.patterns.ml.labeling.label_meta_binary` — target-hit vs.
   stop-hit-or-timeout collapsed together) trained as its OWN model
   (`firm.patterns.ml.xgb_inference.score_pattern_meta_confirmation`
   against a new `pattern_xgb_meta.onnx` artifact), not reusing the
   3-class direction model's own `p_target` as if it were a calibrated
   confidence gate — the conflation de Prado's recipe specifically warns
   against. Not yet wired into `pattern_recognition.py`'s
   `calibrated_probability` (no real trained meta artifact existed until
   item 8 below) — documented inline as a known, deliberately-deferred gap.
3. **Sample-uniqueness weighting** (`firm.patterns.ml.sample_weights`, de
   Prado ch.4) — per-symbol average uniqueness from each event's real
   `[confirm_index, exit_index]` span, feeding `xgb_classifier.train`'s
   new `sample_weight` parameter. `scripts/train_pattern_ml.py`'s own
   rolling-cutoff sampling was already documented as "highly
   autocorrelated"; every row was previously trained as an equal-weight
   i.i.d. sample regardless.
4. **Purged + embargoed K-Fold CV** (`firm.patterns.ml.purged_cv`, de
   Prado ch.7.4) — standard (non-combinatorial) Purged K-Fold, the plan's
   own sanctioned minimum bar. Operates on calendar dates
   (`confirm_date`/`exit_date`), purging train rows whose own label span
   overlaps a test fold's date range plus an embargo buffer. Surfaced and
   fixed a real latent bug along the way: `roc_auc_score` silently returns
   `nan` (not a `ValueError`) when a small CV fold's `y_test` is missing
   one of the 3 possible labels — previously would have silently corrupted
   any caller averaging it into a summary statistic.
5. **Non-transferable features removed**: raw `entry`/`stop`/`target`
   price levels (a $5 stock and a $500 stock convey nothing comparable
   through them) and `confirm_index` — a raw-time leak, since
   `build_dataset`'s rolling-cutoff construction scans a monotonically
   growing window, so this index trended upward with cutoff alone,
   unrelated to pattern quality. `confirm_index` replaced by
   `bars_since_confirm` (bars to the *end of the currently-available
   window* — stationary, and identical to what
   `PatternRecognitionStrategy.generate()` already reports live).
6. **Regime/cross-sectional/liquidity features** added to `build_features`
   (44 -> 52 columns): `dollar_volume_adv_log` (liquidity), plus — given an
   optional, positionally-aligned market-proxy window —
   `market_return_pct`/`relative_strength` and four raw regime features via
   `firm.regime.features.compute_regime_features` (deliberately not a full
   HMM refit per match — too expensive, and lets gradient boosting learn
   its own regime structure rather than a hand-collapsed label). Wired into
   BOTH real callers (`pattern_recognition.py` live, `train_pattern_ml.py`
   training), verified actually populated for real rows, not dead
   infrastructure.

**Items 7-9** (evaluate the strategy's own isolated edge on true
out-of-sample data before any portfolio-level re-audit; retrain both
models against real cached data with every fix above; gate any re-enable
through the existing fail-closed rollout gate plus a real live shadow
period) were in progress as of this writing — update this section with
the real result once that lands, honestly, whichever way the evidence
points (leave-off-with-evidence is an acceptable outcome here, same as
§9.9's conclusion was).

**Net engineering scope**: ~200 new/updated tests across 15 files, every
commit landed with a full green run of the pattern/ML-adjacent suite
(final count 571 passed, 6 skipped) plus the full repo suite checked
separately. No config file changed — every new knob in
`pattern_recognition.py` defaults off, matching this file's own
established convention throughout.
