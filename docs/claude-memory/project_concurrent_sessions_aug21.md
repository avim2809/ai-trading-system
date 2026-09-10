---
name: project-concurrent-sessions-aug21
description: Multiple concurrent Claude sessions doing live-trading reliability work as of 2026-08-21; both live engines paused
metadata:
  type: project
---

As of 2026-08-21, at least two other Claude Code sessions on this machine (`ai-trading-system-4f`, `ai-trading-system-53`) are concurrently working on live-trading reliability + portfolio-construction fixes in this repo:
- 4f: IBKR execution reliability fixes (src/firm/backtest/engine.py, src/firm/brokers/{base,ibkr}.py, src/firm/eval/reports.py, src/firm/live/{engine,state_store}.py), now building a live-faithful backtest baseline + turnover reporting.
- 53: also live-trading reliability + portfolio construction (src/firm/agents/{trader,risk,execution,analysts}.py, backtest/eval harness, LLM path), found bugs from real logs.
- A third, unattributed set of edits (config/settings.yaml, scripts/run_walk_forward_pbo_audit.py, src/firm/config.py, src/firm/experiments/runner.py, src/firm/scripts/run_backtest.py) toward "live-faithful backtest config" (conviction smoothing/correlation cap matching config/live.yaml) - likely session 53's, unconfirmed.

**Why:** the user apparently spun up multiple parallel Claude sessions to work this area at once, and asked that both live engines be paused during the work.

**How to apply:** Both live engines (IBKR :8000, Alpaca :8001) were paused via API (engine stopped, not the systemd service) at the user's request while this work is in progress - do NOT restart either without checking with the other sessions first, even if a task seems to call for it. If picking up related work in this area, check `git status` and coordinate via cross-session messages before editing files already in flight - this snapshot is time-bound and may be stale; verify current session activity via ListAgents rather than trusting this list.
