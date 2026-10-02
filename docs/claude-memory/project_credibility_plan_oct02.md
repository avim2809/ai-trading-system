---
name: project-credibility-plan-oct02
description: 10/2 owner's external "credibility plan" (trend+carry core, DSR/PBO gates, guardrails) reconciled into PLAN.md + 56 tickets + 20 owner decisions; committed on branch plan/credibility-2026-10 (not merged), nothing implemented yet
metadata:
  type: project
---
2026-10-02: owner supplied an external research+implementation plan (freeze discovery → trial ledger + DSR/PBO/CPCV/SPA → ETF/futures data+costs → Carver EWMAC/breakout/carry core → lifecycle/monitoring → ≥6mo paper → stepped live). Reconciled against the repo via a 64-agent workflow into:
- `PLAN.md` (repo root), `plan/OWNER_DECISIONS.md` (OD-01..OD-20, each with a blank "Decision:" line), `plan/tickets/P0-01..P7-03` (56 files, ~298 agent-hours), `docs/REPO_MAP.md` (= P0-01 deliverable).
- Committed 10/2 on local branch `plan/credibility-2026-10` (not merged to main, not pushed); `docs/README.md` index line not yet added (P0-01 does it).

Key reconciliation facts: no real historical holdout exists (2024-07..2026-06 was the Gann window; everything ≤2026-09-30 declared burned, seal_date 2026-10-01 enforced by ACL + fail-closed loader since post-seal data already lives on host); new code goes under `src/firm/{validation,research,costs,signals,risk,monitoring,lifecycle,reporting}` (not top-level src/, and not src/firm/eval which is live-imported); `strategy_correlation.py` the source assumed doesn't exist; suite is 3402 tests not ~425; research agents should use a separate clone `/local/store/research/ai-trading-system` (see [[feedback-never-edit-live-checkout]]).

Biggest finding for the owner: with honest N≈460 legacy trials, DSR≥0.95 needs ~1.65/yr Sharpe vs the plan's own 0.3–0.6 expectation → Tier A practically unreachable; passive (60/40, already live on Alpaca per [[project-allocation-portfolio-build-sep30]]) is the likely honest outcome. OD-09/OD-16 are where the owner signs the N rule and gate deviations.

**Why:** owner wants to make the system credible rather than keep searching for edge (see [[project-edge-search-sep29]], [[project-eodhd-data-and-insider-verdict-sep30]]).
**How to apply:** next session, check which ODs are decided and whether the plan files were committed; start with P0 tickets only after OD-04..OD-09/OD-16; never let tickets touch the live services without the OD-03/P0-07 restart ticket.
