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

## Update 10/4 (overnight autonomous run)
- Owner decisions recorded 10/3 on branch plan/credibility-2026-10: OD-01 capital <$25k (passive-only outcome; P6-P7 not planned), OD-03 keep IBKR as control, OD-04/05/08/09/10/13/14/19 as recommended, OD-06 amended (shared GitHub user + Claude login, local pre-push guard, no machine account), OD-07 full isolation.
- Host setup done by /root/setup_research_isolation.sh: unix user `research`, ACL-denied from live data, clone at /local/store/research/ai-trading-system, ledger root /local/store/research-ledger. Stage 7 (`--guardrails <deploy dir>`: managed settings + hooks) NOT yet run; needs P0-04 merged.
- Agents run headless as `research` in tmux (claude binary copied to /opt/claude/claude); overnight supervisor /local/store/research/overnight_supervisor.sh finished 10/4 05:32 local. Results: reports in /local/store/research/reports/ (MORNING.md). Done: P0-01/02/04/06, P0-05(census, p0b/docs), P2-06, P1-01/02/03/04/05/06/07, P2-01/04/08, P3-01/03, P4-05, P5-03, P5-06. Integration branch integ/p0-p1; batch3/{x,xx,y,yy} still need a second merge. Full suite has 43 pre-existing failures on base (test_api 29, test_llm_providers 13, eodhd_s4 1) due to research-user env (can't read .env), not regressions.
- Gotcha: `claude -p` ends when the model replies without a tool call; agents that "wait for background tests" end the job early. Use /local/store/research/run_full_suite.sh (serialised, nice) and poll in foreground. Never run several full suites in parallel (load hit 24, 760MB free on the live-trading host).
**How to apply:** next session start from reports/MORNING.md; nothing is merged to main or deployed.

## Update 10/4 afternoon
- main (local, not pushed to GitHub) now holds the whole integration: ff-merged integ/p0-p1-b2 (0e540e9), signed OD-16 register (b3dd32c, owner signed 10/4), frozen config/gates.yaml + hash recorded (f153d98). Suite is fully green (3791 passed, 0 failed) after the hermetic-test fix (conftest LITELLM_MODE/fake Alpaca keys etc.).
- OD decisions: OD-09 amended to FAMILY-SCOPED N (~31 trend/carry trials, raw 463 reported alongside); row 15/OD-18 amended: forward check reported, not required before live. Still blank: OD-02, 11, 12, 15, 17, 20.
- Stage 7 guardrails (/root/setup_research_isolation.sh --guardrails <deploy>) was BLOCKED by the safety classifier (self-modification of permission settings); owner must run it. Research Stop hook also needs /local/store/research-venvs/core to exist.
- Autonomous supervisors in tmux: wave5 (P1-08, P1-09, P3-02/05/06) then `autopilot` (waves 6-8: P2-02/03/05, P3-04, P4-01, P3-09, P3-07, P4-03, P4-02) -> integ/auto, never main. Summary lands in /local/store/research/reports/AUTOPILOT.md; alerts in AUTOPILOT_ALERTS.txt. Blocked on owner: P5-02 charters + OD-17 tau -> P3-11 -> P3-08; P0-03/07, P1-10/12 (live), P4-04, P5-01, P5-04, P6/P7.

## Update 10/5 (waves merged; main @ c9a2c38)
- main (pushed to GitHub) now holds: wave 5 (P1-08 harness result PASS size 0.070 marginal, P1-09, P3-02/05/06), wave 6 (P2-02/03/05, P3-04, P4-01), waves 7+9 (P3-09 engine, P0-03, P5-05, P1-10, P1-12 = live-path code, import-guarded; services NOT restarted, owner restarts after US close 10/5), wave 8+10 (P3-07, P4-03, P4-02, P5-02 tooling). Suite 4195 passed / 0 failed.
- Gated merges: I (assistant) do them step by step: suite green + live_import_smoke before AND after + services active; no restarts. An unattended auto-merge script was blocked by the safety classifier even with an owner permission rule, so each stage is driven from the session; agents run in tmux (acceptEdits + allow-list in ~research/.claude/settings.json, edited by owner; bypass mode stays disabled).
- Running 10/5: w10b (P5-04), w11a (P4-04 then P5-01), w12a (core_v1 prereg draft). Queued: real-data steps job for P2-02/03/05 (owner OK'd, pre-seal via data_access only), dependency-vulnerability branch, S2 timer steps.
- Owner decisions 10/5: Kelly bound evaluated in the P3-08 report; tau stays 9% (gate 6 20% cap-days decides); handcrafting = one group per class; P1-08 marginal size accepted; ILS FX via BoI il_macro requires owner edit of protected config/research_freeze.yaml + /etc/claude-code deny copy (assistant cannot edit protected files; deny-rule + classifier block).
- Still owner: write charter (plan/drafts/P5-02 draft), approve prereg draft, restart services tonight, S2 timer install, monitor timer (OD-15), BoI data URL.
