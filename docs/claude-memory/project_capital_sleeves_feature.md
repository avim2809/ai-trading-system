---
name: project-capital-sleeves-feature
description: "Per-strategy capital sleeves — real per-strategy P&L and independent capital, built, A/B-verified, and live-cutover on the Alpaca instance"
metadata: 
  node_type: memory
  type: project
  originSessionId: d0819ea0-0b87-4dce-9dd6-90d93db8461e
  modified: 2026-09-10T11:46:46.448Z
---

Built at the user's request after asking how to know a "bad" strategy's real
contribution to P&L: today's `PerformanceAttribution` is a heuristic
(dominant-strategy-wins-the-whole-order + running-net-share-count over one
shared book), not exact. Full design/history: `docs/capital_sleeves_plan.md`.

**Design** (`capital_allocation_mode: "blended" | "sleeved"`, default
blended = unchanged behavior): each sleeved strategy runs its own
bull→bear→debate→trader→risk→execution pass against its own
`PortfolioState` (fixed initial capital split, compounds independently
from there — not re-normalized to a fraction of current NAV every cycle).
One `TraderAgent` instance per sleeve (it holds real cross-cycle state —
conviction EMA, NAV history — that one shared instance would let sleeves
corrupt). A final netted pass sums every sleeve's target into one real
order set — the only place actual broker orders are generated.

**A real bug found via A/B backtest** (blended vs sleeved, same cached
2024-Q1 data, before ever touching a live engine): sleeved mode's real book
had zero turnover for the whole quarter. Root cause: splitting capital
~18 ways means no single symbol's combined weight realistically clears the
5% rebalance band tuned for the *blended* book. Fixed with a separate,
smaller band for the final netted pass only (each sleeve's own internal
band stays unchanged). This is exactly why the A/B step existed — caught a
real, non-obvious bug before any live risk.

**Live cutover** (2026-09-10): Alpaca (`:8001`) only —
`config/live_alpaca.yaml` has `capital_allocation_mode: sleeved`; IBKR
(`:8000`, `config/live.yaml`) deliberately stays blended as the static
control, same A/B pattern as the sector-scanner work. Sleeves were seeded
from real broker positions + existing attribution via a one-time
`POST /api/live/sleeves/seed` call (refuses a second call — would
overwrite real history with a stale re-approximation). Verified live with
two forced real cycles: cycle 1 clean (22/22 orders); cycle 2 had 2 order
failures (`insufficient qty`, Alpaca-side) purely because the two forced
cycles were only ~2 min apart — a prior cycle's flip-order hadn't settled
yet. Not a bug; won't recur under the normal once-daily cadence.
`GET /api/live/attribution` now returns exact per-sleeve metrics for every
actively-sleeved strategy once it has ≥2 snapshots.

**How to apply:** if asked to expand sleeving to IBKR or change capital
weights, `capital_allocation_mode`/`strategy_capital_weights` are start-time
only (same treatment as `broker` — requires a restart, not a `PUT
/api/live/config` hot-swap) — check `docs/capital_sleeves_plan.md` §5 for
what's still deliberately deferred (frontend per-sleeve view, cross-sleeve
LLM-enhancement budget coordination if `agent_modes` for bull/bear/debate is
ever set to `llm_enhanced`).
