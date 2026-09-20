---
name: project-proactive-trading-system
description: "2026-09-20: 5-phase overnight build making the system proactive (position exits, reflection rollup, real news, CNN patterns, universe incubation) + frontend; full plan and results"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4e7f128e-31b0-4ad0-9bbd-6ac6c0d0ee53
  modified: 2026-09-20T08:44:39.780Z
---

User asked for a genuinely proactive system (news reaction, chart-pattern
reading, dynamic universe management, decisive exits, daily reflection
feeding offline analysis), then explicitly authorized full end-to-end
overnight implementation while asleep — see
[[feedback_autonomous_scope_calls]] for the exact authorization language
and how it was exercised.

**Why**: research (3 codebase-mapping agents + 1 professional-practices
research agent, cited in the plan) found most of the underlying
infrastructure already existed but was inert/buggy/missing one connecting
piece — this closed those specific gaps rather than rebuilding anything
already working. Full plan with citations:
`/root/.claude/plans/lively-stirring-lagoon.md` (session-local file, not
in the repo — read it directly if this memory's summary isn't enough).

**What shipped** (all committed on `main`, full backend suite 1996
passed/0 failed at the end, frontend 106/106 + tsc clean):

1. **Position exits**: `ExecutionAgent`'s new `close_dust_fraction` fixes
   real asymptotic stranding (a target-0 position could sit forever just
   under the rebalance band) without reintroducing a previously-reverted
   regression (a 3-window A/B once found unconditionally forcing full
   closes regressed Sharpe 3.45→0.80). New operator controls:
   `LiveTradingEngine.flatten_symbol`/`flatten_strategy` +
   `POST /api/live/positions/{symbol}/flatten` /
   `/sleeves/{strategy}/flatten` — the direct "sell all holdings on this
   position" the user asked for.
2. **Reflection**: found and fixed a real incident — the `hourly_market_hours`
   schedule (added earlier that same day) silently dropped 6 of every 7
   daily decisions from reflection (bare-date idempotency key,
   first-writer-wins). `TradingMemoryLog` now keys by `date#cycle_id` and
   aggregates a whole day into one `reflect_day()` LLM call. New
   `DailyReflectionRecommendation`: a bounded, pre-enumerated action
   (reduce_position_limit/flag_strategy_for_review/no_action), never
   auto-applied — no source found supports an LLM reflection loop
   auto-adjusting its own live config (SR 11-7 model-risk standard;
   TradeTrap arXiv 2512.02261 on LLM self-reflection failure modes:
   rationalization, hindsight bias, reward hacking). Human applies via
   `POST /api/live/recommendations/{date}/apply`, reusing the
   `sleeve_risk_overrides` mechanism from the same day's stat_arb fix.
3. **Real news**: the RAG wiring to let `sentiment`/`fundamental` LLM
   agents read real article text already existed but the `news` Chroma
   collection had zero documents — new daily ingestion job populates it
   (Alpaca only, verified end-to-end against production Massive
   credentials before enabling), with company-name/ticker anonymization
   per Glasserman & Lin (arXiv 2309.17322) to avoid LLM look-ahead
   contamination.
4. **CNN pattern scoring**: `pattern_recognition`'s validated CNN/GAF
   layer was fully built (an earlier session) but never called live. The
   on-disk model artifact validated *worse* than rule-based scoring
   (probably trained on synthetic smoke-test data — no training-run entry
   for it unlike XGBoost/PPO). Retrained on real cached history
   (`.venv-ml/bin/python scripts/train_cnn_validator.py --data-source
   cache`, ~90s on this 2-core box) with both live services stopped to
   free resources; the retrained model improved walk-forward Sharpe
   0.415→0.645, total return 4.9%→7.7%. Enabled on both instances (shared
   strategy param, not part of either instance's blended/sleeved A/B).
   `cnn_scoring_enabled` defaults off in code — model-file presence alone
   must never imply live usage, a real bug caught before it could
   silently activate on a restart.
5. **Dynamic universe incubation**: new symbols used to enter the live
   universe with real capital immediately; removal never closed the
   position. Now: `candidate`→`active` state (5-day incubation, mirrors
   the removal side's `min_dwell_days` convention),
   `RiskAgent.incubating_symbols` zeroes a candidate's committed weight
   until promoted, and dwell-removal now calls `flatten_symbol`. Alpaca
   only (IBKR stays the static control, per this session's established
   pattern for every new feature this quarter).
6. **Frontend**: flatten buttons on `LiveDashboard.tsx` (positions +
   new Strategy Sleeves table), a `RecommendationsPanel` on
   `Decisions.tsx` with human-gated Apply — the recommendation queue is
   meaningless without a review surface, so this wasn't optional polish.

**Execution notes for next time**: used parallel background agents
per-phase with disjoint file ownership (matches the pattern already in
[[feedback_multiagent_parallel_execution]] if that exists, else worth its
own note) — worked cleanly across 3 concurrent agents in the biggest
round. One agent's session got interrupted mid-task (Phase 3) and had to
be resumed via SendMessage rather than restarted from scratch — its
partial work was still on disk and correct, just needed to finish.
