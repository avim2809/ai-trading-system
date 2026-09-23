---
name: project-planning-cycle-feature
description: "Opt-in pre-open 'planning' cycle + self-consistency sampling — LIVE on both IBKR and Alpaca since 2026-09-23; IBKR outage fail-fast fix also shipped same day"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-23T15:41:01.964Z
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

**Status as of 2026-09-23: enabled on BOTH IBKR and Alpaca.**
- IBKR: `planning_cycle.enabled: true`, restarted, force-tested live
  (cycle 57) — ran clean, bypassed market-hours gate correctly, no orders
  that time (legitimate, no rebalancing signal), nothing hit the broker.
  Real scheduled run now fires automatically every trading day at 09:15 ET.
- Alpaca: originally left off on purpose (sleeved mode, more moving parts —
  the dry_run sleeve-commit-skip hazard below) pending a few days of IBKR
  validation first. User explicitly said "also lets enable alpaca" later
  the same session before that validation window elapsed — enabled anyway
  per direct instruction, restarted, verified clean. The Alpaca-specific
  sleeve hazard was already fixed in code before either instance went live,
  so this wasn't riskier than planned, just earlier than the original
  rollout intended.

**Same-day follow-up, also 2026-09-23**: IBKR's first real planning-cycle
run hit a `broker_unavailable` alert. Investigated rather than dismissed
(user pushed back with "this would happen every time since ibkr always
fails on the first attempt") — root cause was a genuine IBKR-side outage:
`Warning 2105: HMDS data farm connection is broken:ushmds`, which made
`IBKRProvider._get_prices`'s per-symbol retry loop burn ~8 minutes proving
the same outage across all 25 symbols before giving up (also explained a
prior ~30-hour failure streak, cycles 22-50, predating this session's other
changes). Fixed: `IBKRBroker._on_ib_error` now tracks HMDS farm health from
IBKR's own proactive errorEvent callbacks (2105 broken/2106 OK/2107
inactive-but-available), exposed via `is_historical_data_farm_broken()`;
`_get_prices` checks it before and during its per-symbol loop and bails
immediately once a farm is known broken instead of rediscovering it 25
times. Shipped, tested (`tests/test_ibkr_broker.py`,
`tests/test_ibkr_provider.py`), full suite green (2204 passed, 9 pre-
existing unrelated failures confirmed via stash-diff), committed, IBKR
service restarted between cycles (not mid-cycle) to deploy. Not yet proven
against a real live farm outage — will only be confirmed the next time
`ushmds` (or another HMDS farm) actually goes down.

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
