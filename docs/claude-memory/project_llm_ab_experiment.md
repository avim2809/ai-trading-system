---
name: project-llm-ab-experiment
description: "LLM A/B experiment (quant-only vs llm_enhanced analysts) — arm A ended early 2026-09-08, now on arm B"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd4b929f-9803-4fd1-ab03-9b8800a53d3d
  modified: 2026-09-08T20:11:56.799Z
---

Live A/B test comparing analyst modes, tracked in `docs/llm_ab_experiment_log.md`. Toggled via `FIRM_LLM_CONFIG` on both `ai-trading.service` (IBKR paper) and `ai-trading-alpaca.service` (Alpaca paper) — both always run the same arm together so the broker comparison itself stays unconfounded.

**Arm A (quant-only, `config/llm_ab_quant.yaml`):** ran 2026-07-27 → 2026-09-08 (43 of a planned 56 days — user asked to end it 13 days early). Final snapshot: NAV -4.9% (998984 → 950221), annualized Sharpe -5.625, max drawdown 4.9%. Logged as consistent with the project's own repeated walk-forward/PBO findings ([[project_joint_optimizer_redesign_aug23]]-adjacent — the 12-strategy/25-name combination layer hasn't cleared a profitability gate in 6 independent attempts) — so this result isn't read as evidence against quant-only analysts specifically vs. arm B.

**Arm B (llm_enhanced, `config/llm_ab_llm.yaml`):** started 2026-09-08 ~05:57 UTC, commit `2083f71`. Both services restarted together, verified healthy on restart (both `broker_connected: true`, clean logs). No fixed end date set in the log for arm B (arm A's was a planned 8-week window; arm B doesn't have one recorded yet — worth asking the user if one should be set, or just eyeballing it).

**Why this matters going forward:** if asked to check on the "LLM experiment" or compare arms, check `FIRM_LLM_CONFIG` in the deployed `/etc/systemd/system/ai-trading*.service` units (not just the repo's `deploy/` copies — those are the source but the live ones are what's actually running) and pull snapshots via `scripts/snapshot_llm_ab_arm.py` against `data/live_state.db`.

**How to apply:** don't propose ending or restarting this experiment on your own judgment — arm transitions are the user's call (arm A's early end was an explicit user request, not a Claude-initiated decision). Do keep `docs/llm_ab_experiment_log.md` updated with new snapshots when asked to check status.
