@AGENTS.md

# Claude Code notes (Claude-specific tail)

`AGENTS.md` above holds the pointers, the session-type scoping, the research-integrity rules and the rules of thumb (it is
CODEOWNERS-protected). This tail holds the architecture notes Claude Code sessions use; it is editable. The concise agent memory lives in
`.cursor/rules/` and the full reference in `docs/PROJECT_CONTEXT.md`.

## Pipeline

13 strategies (raw scores) → 3 analysts (sole z-score) → bull/bear → PM → risk → execution.
11 of 13 strategies are currently enabled (`ml_prediction`, `gann` permanently disabled — see `docs/PROJECT_CONTEXT.md`). Capital can run
**blended** (one shared portfolio, default) or in per-strategy sleeve mode (`capital_allocation_mode: sleeved` — independent per-strategy
capital/P&L, netted only at the final real-order pass). IBKR (`:8000`) stays blended and is the untouched control. See
`docs/capital_sleeves_plan.md`.

Alpaca (`:8001`) runs `strategy_mode: allocation` live (since 2026-09-30; `src/firm/allocation/`): the whole pipeline is bypassed for a passive
core plus satellite-sleeve allocator planned against broker positions — 60/40 SPY/IEF core (92% of NAV) plus an 8% BTC 4-week-trend
satellite, judged by its own frozen forward test (`docs/allocation_forward_test_plan.md`). The Alpaca strategy block is dormant in that mode.
See the "Allocation mode" section of `docs/PROJECT_CONTEXT.md` and `docs/allocation_deploy_runbook.md`.

## Eval & behavioural features (wired backend + React UI)

- **Overfitting**: PBO/CSCV + Deflated/Probabilistic Sharpe (`src/firm/eval/overfitting.py`); walk-forward aggregate returns an `overfitting` block.
- **Trade + robustness metrics**: profit factor / expectancy / win rate (`eval/metrics.py`) and Monte Carlo bootstrap (`eval/robustness.py`);
  both appear in `report.json`.
- **Tear-sheet**: QuantStats HTML via `GET /api/runs/{id}/tearsheet` (optional `report` extra).
- **Allocation / signal combination**: `allocation_method` (+ `kelly`/`kelly_fraction`) and `signal_combination` (`confidence` | `optimal`) —
  configurable via `RunRequest`, `config/settings.yaml`, `config/live.yaml`, and `PUT /api/live/config`.
- **News-guard blackout** (`src/firm/live/news_guard.py`) + **execution lock** `FIRM_ALLOW_TRADING` (`src/firm/live/execution_safety.py`) —
  both default OFF/fail-closed.
- **Backtest cache**: `data_source: cache` loads `combined/prices` + `combined/fundamentals` from `data/cache`.
- **UI**: `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx`.

## Claude Code specifics

- Committed `.claude/settings.json` applies to every session in a checkout, including root ops sessions, so it only denies edits to the
  guardrail files and force-pushes. Research-only restrictions (post-seal `Read` denies, the deny hook, sandbox) live in the research user's
  own root-owned settings; templates are in `deploy/`.
- Research sessions: read `docs/HOLDOUT_POLICY.md` before touching any data path, and use `firm.research.data_access` for file reads.
- Keep `docs/claude-memory/` in sync when memory changes (see `docs/claude-memory/feedback_repo_doc_and_memory_upkeep.md`). Research
  sessions never restart services or edit live config, regardless of any "run on your own" memory entry.
