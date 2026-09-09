# Per-strategy capital sleeves — implementation tracker

**Status:** Core mechanism built and tested (backend only) — **not enabled
on either live production instance**. `capital_allocation_mode` defaults to
`"blended"` everywhere (byte-for-byte today's existing behavior); flipping
either `:8000`/`:8001` to `"sleeved"` is an explicit, separate, later step
(see §5). · **Date:** 2026-09-09.

## 1. Why

Today all ~12 registered strategies blend into one shared portfolio: one
`PortfolioState`, one target-weight decision per cycle, one broker account.
`GET /api/live/attribution` already reports per-strategy Sharpe/return, but
it's a heuristic (dominant-strategy-wins-the-whole-order +
running-net-share-count over one shared book) — positions are never
actually split by strategy, so the numbers are an approximation, not a real
standalone track record. The user's goal: know which strategies genuinely
work well enough to eventually run with only those, and (separately) let a
specific strategy's own signal manage its own position directly rather than
being anonymously folded into one blended book. Both need the same
capability — real, per-strategy capital and positions, not reconstructed
after the fact.

Full design rationale and the research it's built on lives in the
planning session's approved plan (captured before implementation began);
this doc tracks what's actually been built, tested, and verified since.

## 2. Design summary

Each strategy gets its own **sleeve**: an independent
`bull → bear → debate → trader → risk → execution` pass against its own
`PortfolioState` (no new class — sleeves reuse `PortfolioState` directly,
since the live engine already never manually applies fills to it outside
the backtest path; this is exactly the backtest path's own mechanism, reused
per sleeve). A sleeve's capital is a **fixed initial split of total capital
at first allocation, then compounds independently** from its own trades from
that point on (like a real per-strategy sub-account) — it is *not*
re-normalized to a fraction of current total NAV every cycle, since that
would hide a strategy's own true standalone performance.

After every sleeve's own (already risk-approved) pass, its target weight is
converted to an absolute dollar position, summed across sleeves per symbol,
converted back to one combined weight against the *real* total NAV, and fed
into one final `ExecutionAgent` call against the real shared `PortfolioState`
— the only place actual broker orders are generated. Sleeves with opposing
views on the same symbol net down to one smaller real order, not two.

Bull/bear/debate/`ExecutionAgent` need **zero code changes** — they're
stateless, reused as-is per sleeve. `TraderAgent` needs **one independent
instance per sleeve** (not one shared instance called N times): it holds
real cross-cycle instance state (conviction EMA, joint_optimizer NAV
history) keyed only by symbol, so one shared instance would let sleeves
corrupt each other's smoothing/history state. `RiskAgent`'s one mutable
attribute (`_regime_detector`) is safe to share across sleeves — it caches
market-wide regime, not anything sleeve-specific.

## 3. Key files

| File | What changed |
|---|---|
| `src/firm/agents/orchestrator.py` | `capital_allocation_mode` config; `sleeve_traders` constructor param; LLM-cost safety guard (see §4); `_step_sleeved()` (the sleeved-mode pipeline); `_partition_blackboard()`, `_sleeve_capital_weights()`, `_get_or_create_sleeve_portfolio()`, `export_sleeve_portfolios()`/`restore_sleeve_portfolios()`, `get_sleeve_metrics()`. Blended mode's existing `step()` path is untouched. |
| `src/firm/runtime.py` | `build_orchestrator` constructs one `TraderAgent` per active strategy (instead of one shared instance) when `capital_allocation_mode: "sleeved"`. |
| `src/firm/live/state_store.py` | Additive per-sleeve trader-state keys (`save_sleeve_trader_state`/`load_sleeve_trader_state`, keyed by strategy) and a new `sleeve_portfolios` blob (`save_sleeve_portfolios`/`load_sleeve_portfolios`) — same generic key→blob table, no schema change. |
| `src/firm/live/engine.py` | `_load_persisted_state`/`_persist_live_state` grow a sleeve-specific block (mirrors the existing trader-state block exactly) — restores/saves every sleeve's `TraderAgent` state and virtual `PortfolioState` (cash/holdings) across restarts, since sleeves have no real broker sub-account to reconcile from. |
| `src/firm/api/routers/live.py` | `GET /api/live/attribution` prefers `Orchestrator.get_sleeve_metrics()` (exact, from each sleeve's own NAV history) over `PerformanceAttribution`'s heuristic when `capital_allocation_mode: "sleeved"`, falling back to the heuristic for any strategy without a sleeve. |
| `tests/test_sleeves.py` (new) | 21 tests: capital-weight splitting, LLM-cost-safety guard, single-sleeve-matches-blended sanity check, independent compounding, no-signal-day snapshot continuity, netting (partial and full offset), persistence round-trip, `get_sleeve_metrics`, `build_orchestrator` wiring, the `/attribution` endpoint merge. |
| `tests/test_live_engine.py` | +1 test: sleeve `TraderAgent` state and virtual portfolios both survive a simulated process restart (companion to the existing conviction-EMA persistence test). |

**No config file changes** — `config/live.yaml`/`config/live_alpaca.yaml`
are untouched. `capital_allocation_mode` defaults to `"blended"` in code
(`cfg.get("capital_allocation_mode", "blended")`), so neither live instance's
behavior changes until that key is explicitly added and set to `"sleeved"`.

## 4. LLM-cost safety (deliberate scope limit)

Bull/bear/debate can be LLM-enhanced (`LLMBullResearcher`/`LLMBearResearcher`/
`LLMDebateAgent`, one RAG+LLM call per symbol, capped by
`max_theses_per_agent`/`max_debate_symbols`), controlled by `agent_modes` in
`config/llm.yaml` — which today deliberately keeps all three at `"quant"`
("no per-symbol LLM fan-out"). Running them once per sleeve is cheap CPU in
that (current, production) configuration, but would multiply LLM call volume
roughly N-fold if any of the three were ever switched to `llm_enhanced`/
`llm_only`, since each sleeve's own top-K enhancement-budget selection has no
visibility into what other sleeves already spent that cycle.

**Cross-sleeve shared enhancement-budget coordination is not built yet.**
Instead, `Orchestrator.__init__` **refuses to construct** (`ValueError`) when
`capital_allocation_mode: "sleeved"` and any of
`bull_researcher`/`bear_researcher`/`debate` is set to `llm_enhanced`/
`llm_only` — fail loud rather than silently ship an N-fold cost surprise.
Covered by `TestSleeveLlmCostSafety` in `tests/test_sleeves.py`. Building the
real coordination (global top-K across all sleeves' candidate theses, not
per-sleeve) is a legitimate follow-up if these three roles are ever wanted
in `llm_enhanced` mode alongside sleeving — not required for today's actual
production configuration.

## 5. What's deliberately not done yet

- **Frontend.** `LiveConfig.tsx`/`AgentInspector.tsx` still show one shared
  portfolio view. Backend correctness came first; a per-sleeve NAV/position/
  PnL view is real follow-up work, not required for the mechanism to work
  correctly.
- **Live cutover.** Neither `:8000` (IBKR) nor `:8001` (Alpaca) has
  `capital_allocation_mode: "sleeved"` set. Per the original plan's migration
  section, that's a separate, later, explicitly-approved step — needs an
  isolated-port smoke test first (synthetic/cache data, confirm sleeve
  ledgers reconcile and risk caps bind correctly per sleeve), then an
  A/B-style comparison on cached historical data (`blended` vs `sleeved`,
  same date range, sanity-check total return/turnover aren't wildly
  diverged), then a real cutover with each sleeve seeded from a best-effort
  split of the existing (heuristic) attribution's per-strategy holdings at
  the moment of cutover.
- **Cross-sleeve LLM-enhancement budget coordination** (see §4) — only
  matters if `agent_modes` is ever changed from its current all-`"quant"`
  default for bull/bear/debate.

## 6. Verification so far

- `tests/test_sleeves.py` — 21 passed.
- `tests/test_live_engine.py` — 136 passed (full file, including the new
  sleeve-persistence test).
- Full main-venv suite (`pytest -q --ignore=tests/test_api.py`) — green
  alongside these changes (see the session's own verification run for the
  exact count at the time of committing).
- Both production instances (`:8000`, `:8001`) confirmed unaffected
  throughout — this work is purely additive/opt-in code, never touched
  either running process or its config.
