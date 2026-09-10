# Agent Context (Claude Code)

Pointers for AI coding agents working in this repository. Mirrors `AGENTS.md`; the
concise agent memory lives in `.cursor/rules/` and the full reference in
`docs/PROJECT_CONTEXT.md`.

| Resource | Purpose |
|----------|---------|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, deployment, live config, REST/UI wiring, IBKR pitfalls |
| [.cursor/rules/](.cursor/rules/) | Concise agent memory (project context, live trading, strategies, IBKR, logging, frontend) |
| [docs/claude-memory/](docs/claude-memory/) | Repo mirror of Claude's persistent session memory (`MEMORY.md` index + per-topic files) — keep in sync when it changes |
| [deploy/ai-trading.service](deploy/ai-trading.service) | Production systemd unit (`firm-api` + auto-start live) |
| [config/live.yaml](config/live.yaml) | Canonical live paper trading configuration (IBKR, `:8000`, blended capital) |
| [config/live_alpaca.yaml](config/live_alpaca.yaml) | Second live paper instance (Alpaca, `:8001`, sleeved capital) |

**Production on bare-metal:** two live instances run as separate systemd
units/ports — `:8000` (IBKR, `config/live.yaml`) and `:8001` (Alpaca,
`config/live_alpaca.yaml`), each `firm-api` with `FIRM_AUTO_START_LIVE=1` and
its own `FIRM_API_PORT`/`FIRM_DATA_DIR`/`FIRM_LIVE_CONFIG`. Neither runs
`scripts/run_live_trading.py` directly.

## Pipeline

13 strategies (raw scores) → 3 analysts (sole z-score) → bull/bear → PM → risk → execution.
11 of 13 strategies are currently enabled (`ml_prediction`, `gann` permanently
disabled — see `docs/PROJECT_CONTEXT.md`). Capital can run **blended** (one
shared portfolio, default) or **sleeved** (`capital_allocation_mode:
sleeved` — independent per-strategy capital/P&L, netted only at the final
real-order pass); Alpaca (`:8001`) runs sleeved live today, IBKR stays
blended as the control. See `docs/capital_sleeves_plan.md`.

## Eval & behavioural features (wired backend + React UI)

- **Overfitting**: PBO/CSCV + Deflated/Probabilistic Sharpe (`src/firm/eval/overfitting.py`);
  walk-forward aggregate returns an `overfitting` block.
- **Trade + robustness metrics**: profit factor / expectancy / win rate (`eval/metrics.py`)
  and Monte Carlo bootstrap (`eval/robustness.py`); both appear in `report.json`.
- **Tear-sheet**: QuantStats HTML via `GET /api/runs/{id}/tearsheet` (optional `report` extra).
- **Allocation / signal combination**: `allocation_method` (+ `kelly`/`kelly_fraction`) and
  `signal_combination` (`confidence` | `optimal`) — configurable via `RunRequest`,
  `config/settings.yaml`, `config/live.yaml`, and `PUT /api/live/config`.
- **News-guard blackout** (`src/firm/live/news_guard.py`) + **execution lock**
  `FIRM_ALLOW_TRADING` (`src/firm/live/execution_safety.py`) — both default OFF/fail-closed.
- **Backtest cache**: `data_source: cache` loads `combined/prices` + `combined/fundamentals` from `data/cache`.
- **UI**: `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx`.

## Rules of thumb

- **Logging/traceability**: every module uses stdlib `logging`
  (`log = logging.getLogger(__name__)`); log decisions, fallbacks, external-I/O
  outcomes, and safety events (execution-gate blocks, news-guard, kill-switch).
  No `print`, no bare `except: pass`. See `.cursor/rules/logging.mdc`.
- Edit `config/live.yaml` for live universe/risk/strategies/params (include `risk.sector_map` for sector caps); use
  `resolve_live_startup()` in `src/firm/live/provider_utils.py` — do not duplicate YAML merge logic.
- Backtests with **Cache** data source load prices + fundamentals from `data/cache` (not live API).
- On a running engine, change behavioural knobs via engine setters
  (`update_news_guard` / `update_signal_combination` / `update_allocation`).
- Never call `IBKRBroker.connect()` from uvicorn's asyncio loop — use a sync handler
  thread or `asyncio.to_thread()`.
- **Frontend is mobile-first responsive** (usable at ~375px): grids start at
  `grid-cols-1/2` and scale with `md:`/`lg:`; header/action rows wrap
  (`flex flex-wrap … gap`, `flex-shrink-0`, `min-w-0`); tables `overflow-x-auto`.
  See `.cursor/rules/frontend.mdc`.
- Keep backend endpoints and `frontend/src/api/{types,client}.ts` in sync when extending features.
