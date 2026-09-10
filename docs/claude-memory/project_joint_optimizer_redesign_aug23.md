---
name: project-joint-optimizer-redesign-aug23
description: "Combination-layer redesign (joint QP optimizer) built, validated, and honestly failed its gate on 2026-08-23/24 — both engines stay paused"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a2d3320-4708-4171-a0bb-5abcdfb5125a
  modified: 2026-08-29T22:08:04.671Z
---

Session on 2026-08-23/24 (user left mid-session with "run on your own... send
summary via discord when done, push to git once all is done" — full autonomy
granted) did two things:

1. **Shipped and validated real execution/turnover fixes** (Part 1 of that
   session): IBKR qualified-contract cache + two-stage health check,
   cycle-counter persistence (fixed a real client_order_id collision bug),
   `rebalance_band_pct`/`rebalance_fraction` no-trade-band + turnover-aware
   sizing in `ExecutionAgent` (shipped: band=0.05, fraction=0.7),
   conviction-EMA smoothing fixed to actually reach `TraderAgent` via
   `resolve_live_startup()` (was silently dropped before), LLM enhancement
   temperature pinned to cut nondeterminism. All validated via 3-window A/B,
   shipped to `config/live.yaml`/`live_alpaca.yaml`/`settings.yaml`.

2. **Built a full joint constrained portfolio optimizer** (`firm.portfolio.
   optimizer`, cvxpy QP replacing `TraderAgent`'s L1-normalize sizing +
   `RiskAgent`'s sequential clips) as `allocation_method: joint_optimizer`.
   Found and fixed two real implementation bugs during validation (an IC
   daily/annualized-IR units mismatch, and `ctx.portfolio.history` never
   being populated during backtests — only live — which made the IC-trust
   mechanism permanently inert offline). **Final verdict after 3 full
   4-fold walk-forward+PBO gate runs: `fail`** (PBO=0.421, DSR=0.00061 —
   both required thresholds unmet). This is the **4th consecutive negative
   result** this session on core profitability (after `zscore_demean`,
   the turnover-fix formal gate, and strategy concentration all failed
   too) — full writeup in `docs/formal_pbo_audit.md`/
   `docs/remediation_progress.md` #57-62.

**Why:** the user's own explicit standing decision was to stay paused and
scope a real combination-layer redesign rather than resume live or keep
doing one-knob A/B tweaks — this was that redesign, done rigorously (built,
hand-tested, gated by the same walk-forward+PBO harness before any
promotion, exactly per plan).

**How to apply:**
- `joint_optimizer` exists in the codebase, fully tested (40 tests), but is
  **off by default everywhere** — do not enable it in any config without
  new evidence; the gate result stands until someone re-validates.
- **SUPERSEDED 2026-08-30: both engines are running again**, resumed
  2026-08-24/25 by explicit user request during the PART 3 session below —
  confirmed via `docs/remediation_progress.md`, not a restart accident.
  User re-confirmed 2026-08-30: keep running. Treat "both engines stopped"
  as **historical**, not current state.
- **There was a third session (PART 3, 2026-08-24/25) this memory didn't
  originally capture** — see [[project_signal_quality_investigation_status]]
  for the full, corrected picture: it's **6 consecutive failed gates**
  across two sessions, not the 4 this memory originally described
  (`zscore_demean`, concentration, `joint_optimizer` ×this doc, then PART 3
  added stat_arb ETF pairs, a seasonality overlay, and a macro overlay — all
  `fail` too). Full detail, current data-coverage state, and the standing
  Sharadar-vs-scanner decision live in that memory now — read it first.
- The real, structural finding: execution/turnover/cost fixes reliably help
  (4/4 positive), but combination-layer changes have now failed 4/4 attempts
  using genuinely different mechanisms (demean toggle, concentration, full
  QP redesign) — this is likely a property of the **12-strategy signal
  set's quality on this specific 25-name universe/window**, not the
  combination mechanism. A future session should consider interrogating the
  strategies/data/universe themselves (per-strategy OOS stability, universe
  size/correlation, longer delisting-inclusive history) rather than another
  combination-layer mechanism — see `docs/remediation_progress.md`'s
  session-ending conclusion (#62) for the reasoning.
- A real GitHub Dependabot alert appeared after this push (1 high, 4
  moderate) — plausibly from the new `cvxpy`/`clarabel`/`scs`/`osqp`/
  `highspy` dependency chain added this session. Not investigated/fixed —
  flag to the user, don't silently upgrade/pin without asking.

Related: [[feedback_production_incident_priority]],
[[project_data_pipeline_and_backtest_reliability]].
