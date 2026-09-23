---
name: project-planning-cycle-feature
description: "Opt-in pre-open 'planning' cycle + self-consistency sampling — LIVE on IBKR since 2026-09-23, Alpaca still deliberately off — remind user to consider enabling Alpaca"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-23T13:20:29.836Z
  originSessionId: 403dab55-8f5b-43a3-afa3-7533df0c5692
---

Built same-session as the VPS migration cutover follow-up work
([[project_vps_migration_sep22]]), but a distinct feature request: the user
asked whether the system could analyze the market beyond trading hours,
with all its capabilities, to have more time to plan before open. Full
design + implementation in one plan-mode session — see git commit
"Add opt-in pre-open 'planning' cycle + self-consistency sampling" for the
complete rationale (also mirrored into `docs/PROJECT_CONTEXT.md`'s new
"Pre-open 'planning' cycle" section).

**Status as of 2026-09-23: enabled on IBKR (`config/live.yaml`), still
deliberately OFF on Alpaca (`config/live_alpaca.yaml`).** User explicitly
asked to be reminded that this is still open — surface this at the start of
a relevant future session (don't wait to be asked) rather than letting it
go stale silently:
- IBKR: `planning_cycle.enabled: true`, restarted, force-tested live
  (cycle 57) — ran clean, bypassed market-hours gate correctly, no orders
  that time (legitimate, no rebalancing signal), nothing hit the broker.
  Real scheduled run now fires automatically every trading day at 09:15 ET.
- Alpaca: left off on purpose — sleeved mode, more moving parts (the
  dry_run sleeve-commit-skip hazard below), wanted IBKR validated over a
  few real days first before touching the sleeved instance.
- **When reminding**: check how IBKR's planning cycle has actually
  performed over the intervening days (any `overnight_plan_applied`/
  `overnight_plan_discarded` alerts, any errors) before just proposing
  "enable Alpaca too" — the whole point of doing IBKR first was to have
  real behavior to check, not to rubber-stamp it after a fixed time delay.

**Two design pivots worth remembering the reasoning for, not just the
outcome** (the user pushed back and asked for honesty rather than a
pleasing answer both times):
1. Originally proposed two scheduled runs (an early "digest" + a
   close-to-open "refresh"). User asked "what's the point of the early one
   if the late one always wins?" — correct challenge: the early run's
   decision gets thrown away every night once the late one supersedes it,
   so its only real value is a fallback/visibility artifact, not a better
   plan. Dropped to a single `cron:09:15` run.
2. User asked "is there really no way to accumulate multiple runs for a
   stronger signal, be honest." I had conflated freshness (running again
   later, catches new information) with confidence (self-consistency —
   sampling the SAME decision point multiple times and aggregating to
   cancel noise). These are genuinely different mechanisms. Ended up
   building the real one (self-consistency sampling,
   `self_consistency_samples` config, `LLMAgentMixin._call_llm` samples N
   times and aggregates) as a distinct, separately-toggleable feature —
   user said "add it but keep it off for now," so it's real, tested code
   (`tests/test_llm_self_consistency.py`), just defaulted to `1` (off).

**Real hazard caught and fixed as part of this, not worked around**:
sleeved mode's `_step_sleeved` commits hypothetical fills into persisted
sleeve state *inside* `Orchestrator.step()` itself, before the caller ever
sees the orders — a naive "just don't submit the orders" implementation
would still have corrupted sleeve NAV with phantom fills on the Alpaca
instance, the exact same bug class fixed in `sleeve_reconciliation.py`
earlier the same session. Fixed via a `dry_run` param threaded into
`step()`/`_step_impl()`/`_step_sleeved()`.

**Reused existing infra rather than building new**: `ApprovalQueue` already
had everything needed (full order list + blackboard snapshot,
pending/approved/rejected/expired lifecycle) — just needed a
`source_cycle_type` tag and a per-call `expiry_minutes` override. Two other
existing mechanisms were investigated and confirmed NOT fits before landing
on this one: `extended_hours_trading` (bypasses market-hours gate for real
off-hours *submission* — opposite of what was needed) and
`/api/live/recommendations/{date}/apply` (daily-reflection risk-parameter
tuning, no symbols/quantities/blackboard at all).

Related: [[project_vps_migration_sep22]], [[feedback_autonomous_scope_calls]]
(same overnight-autonomy grant covered finishing this after the user went
to bed).
