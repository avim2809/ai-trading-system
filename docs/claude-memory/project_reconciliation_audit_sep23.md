---
name: project-reconciliation-audit-sep23
description: "2026-09-23 end-to-end live audit: IBKR HMDS-outage fail-fast + reconciliation-blackout fix, Alpaca wash-trade fix, sleeve drift write-off mechanism built but NOT yet executed (needs user to run one command)"
metadata:
  node_type: memory
  type: project
  modified: 2026-09-23T17:28:53.638Z
  originSessionId: 403dab55-8f5b-43a3-afa3-7533df0c5692
---

Same session as [[project_vps_migration_sep22]] and [[project_planning_cycle_feature]],
later the same day (2026-09-23). User asked "check the system end to end is it
working well" after the planning-cycle rollout, and got frustrated ("this has
to conclude!!!") when each check kept surfacing a new issue — see
[[feedback_exhaustive_audit_closure]] for the standing lesson from that.

## What was actually wrong (4 distinct, real issues)

1. **IBKR HMDS farm outage (external, not fixable by us)**: `ushmds` (IBKR's
   own historical-data farm) has been flapping broken/OK for 2+ days
   (confirmed via `Warning 2105/2106` in journalctl, going back to 9/21).
   37 of the most recent 50 IBKR cycles failed with "No usable PIT prices for
   IBKR cycle" as a direct result. Nothing client-side fixes the outage
   itself — this instance uses `IBKRProvider` as its *only* price source, no
   fallback, by original design (`build_live_providers`'s own docstring).
   Fixed the fail-fast (retrying all 25 symbols out to timeout instead of
   bailing once IBKR's own farm-status callback says it's broken) earlier
   the same session — see the commit "Fail fast on broken IBKR
   historical-data farm..." — `IBKRBroker._on_ib_error` tracks farm health,
   `IBKRProvider._get_prices` checks it before/during its loop.

2. **IBKR reconciliation blackout (real bug, fixed)**: `self._portfolio` is
   never persisted across a restart and can only self-heal via
   `sync_portfolio_from_broker`, which ran *after* price resolution in
   `_run_cycle_work` — so whenever #1's outage made price resolution raise
   (74% of cycles), the sync never got a chance to run at all, leaving
   internal holdings frozen at 0 for every symbol while the broker held real
   positions. Looked catastrophic; root cause was just "never got a chance
   to sync". Fixed: moved the sync to run unconditionally *before* price
   resolution (`prices` was only ever used there for an optional NAV
   snapshot, not the correction itself). Commit `f887296`. **Verified live**:
   force-triggered a cycle post-restart, it still hit the known price outage,
   but `GET /api/live/reconciliation` read `{"status":"ok","discrepancies":[]}`
   immediately after — self-heals through the outage now.

3. **Alpaca wash-trade rejections (real bug, fixed by a subagent, deployed)**:
   `ExecutionAgent._maybe_submit_protective_order` submits a broker-side
   stop with no order-ID tracking and can never cancel it. One such stop for
   BKNG (cycle 51, 01:13) blocked that cycle's real entry order as a wash
   trade, and then silently blocked BKNG's risk-approved rebalance again in
   cycles 54 and 55, ~18 hours later, using the *same* stale resting order
   id every time. Fixed in `src/firm/brokers/alpaca.py`:
   `_submit_with_wash_trade_retry` catches error 40310000 specifically when
   it carries `existing_order_id` (distinct from the unrelated
   insufficient-qty flip-through-zero case that shares the same error code
   but has no `existing_order_id`), cancels the stale order, retries once.
   Commit `f27ebb7`. Also quieted `get_position`'s logging: a 404 (no
   position — the normal case for a fresh open, hit on every market order
   via `_plan_flip_split`) was logging a full traceback indistinguishable
   from a real crash; now a quiet INFO line, with any *other* failure still
   loud at WARNING. Deployed via `systemctl restart ai-trading-alpaca.service`.

4. **Alpaca sleeve reconciliation drift (real bug, root-caused, fix built
   but NOT YET EXECUTED)**: `apply_realized_fill`'s ongoing per-fill
   correction (built earlier the same session, see
   [[project_vps_migration_sep22]]) only reaches a fill whose *originating
   decision cycle* recorded `sleeve_decisions` — a field that doesn't exist
   for any cycle before it was added. Quantified: only 68 of 394 historical
   filled orders were traceable this way; the other 326 (82%) can never be
   individually reconstructed — that's exactly why every symbol was
   mismatched with a $12.8k cash gap. This is *not* a bug in the correction
   mechanism itself (confirmed it fires correctly for in-scope fills, e.g.
   5-6 corrections logged same day) — it's an unrecoverable historical data
   gap. Fix: `LiveTradingEngine.seed_sleeves_from_attribution(force=True)`
   (new `force` param, `POST /api/live/sleeves/seed?force=true`) re-runs the
   exact same attribution-based seed used at the original sleeved-mode
   cutover against *current* broker truth + *current* (persisted,
   cumulative) `PerformanceAttribution` state — an explicit one-time write-off
   of the legacy drift, not a precise reconstruction, logging each sleeve's
   pre-seed state at WARNING first. Bundled into commit `f887296`, tested
   (`tests/test_sleeves.py::test_force_bypasses_the_already_has_state_refusal`,
   `tests/test_api.py` force-seed case), deployed.

   **NOT YET RUN**: the actual `POST .../sleeves/seed?force=true` call against
   the live engine is blocked by the auto-mode permission classifier
   ("Modify Shared Resources") — a real, mutating write to live account
   state, correctly gated. Tried twice, both explicitly denied; did not
   attempt to work around it. **This is the one open item** — the user needs
   to either run `curl -s -X POST "http://127.0.0.1:8001/api/live/sleeves/seed?force=true"`
   themselves, or add a permission rule allowing it. Until then, Alpaca's
   `GET /api/live/reconciliation` will keep showing the same ~36-symbol
   mismatch — that's expected, not a regression, not something to
   re-investigate.

## Multi-agent sweep (user explicitly said "use multiagents")

Ran 4 things in parallel to close this out in one pass instead of trickling
findings: full test suite (2204 passed, only the 9 pre-existing/unrelated
failures — confirmed via stash-diff earlier the same session — `test_api.py`
TestRuns flakiness + network-dependent `test_investing_calendar.py`), an
ops/infra sweep (services/disk/backups/fail2ban/nginx-auth/TLS/cron — all
clean, backups verified as real files on disk not just timer state), the
Alpaca order-submission-error investigation (item 3 above), and a
scheduled-jobs/alerts/approvals/kill-switch sweep (all clean; two things that
initially looked like bugs — IBKR cycle_id gaps, "silent" reflection job —
were run to ground and confirmed to be journald-retention/already-reflected
red herrings, not defects).

**Also confirmed as part of the sweep**: `planning_cycle` is correctly wired
end-to-end on both instances but both were enabled today *after* the 09:15 ET
slot already passed — first real scheduled fire is tomorrow morning on both;
worth a quick check then, not before. `sp500_dynamic_universe` (Alpaca) is
silently running on the free GitHub fallback dataset every day because FMP's
`/stable/sp500-constituent` now needs a paid tier (HTTP 402) — not a new
bug, the fallback is working as designed, but worth knowing next time FMP
tier/cost comes up. `danelfin_dynamic_universe` reconfirmed still correctly
disabled on both configs.

Related: [[project_vps_migration_sep22]], [[project_planning_cycle_feature]],
[[feedback_exhaustive_audit_closure]], [[feedback_production_incident_priority]].
