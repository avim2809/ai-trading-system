# Agent Context

Pointers for AI coding agents working in this repository.

| Resource | Purpose |
|----------|---------|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, deployment, live config, IBKR pitfalls |
| [.cursor/rules/](.cursor/rules/) | Concise agent memory (project context, live trading, strategies, IBKR, logging, frontend) |
| [docs/claude-memory/](docs/claude-memory/) | Repo mirror of Claude's persistent session memory (`MEMORY.md` index + per-topic files) — keep in sync when it changes |
| [deploy/ai-trading.service](deploy/ai-trading.service) | Production systemd unit (`firm-api` + auto-start live) |
| [config/live.yaml](config/live.yaml) | Canonical live paper trading configuration (IBKR, `:8000`, blended capital) |
| [config/live_alpaca.yaml](config/live_alpaca.yaml) | Second live paper instance (Alpaca, `:8001`, sleeved capital) |

**Production on bare-metal:** two live instances run as separate systemd units/ports — `:8000` (IBKR, `config/live.yaml`) and `:8001` (Alpaca, `config/live_alpaca.yaml`), each `firm-api` with `FIRM_AUTO_START_LIVE=1` and its own `FIRM_API_PORT`/`FIRM_DATA_DIR`/`FIRM_LIVE_CONFIG`. Neither runs `scripts/run_live_trading.py` directly.

**Strategies & capital:** 13 strategies feed the pipeline; 11 are enabled (`ml_prediction`, `gann` permanently disabled — see `docs/PROJECT_CONTEXT.md`). Capital runs **blended** (one shared portfolio, default) or **sleeved** (`capital_allocation_mode: sleeved` — independent per-strategy capital/P&L, netted only at the final real-order pass); Alpaca runs sleeved live today, IBKR stays blended as the control. See `docs/capital_sleeves_plan.md`.

**Logging:** every module uses stdlib `logging` (`log = logging.getLogger(__name__)`); log decisions, fallbacks, I/O failures, and safety events — no `print`, no bare `except: pass`. See `.cursor/rules/logging.mdc`.

**Frontend:** the React dashboard is mobile-first responsive (usable at ~375px); grids start at `grid-cols-1/2` and scale with `md:`/`lg:`, header rows wrap. Keep `src/api/{types,client}.ts` in sync with backend. See `.cursor/rules/frontend.mdc`.

**Eval & behavioural features (wired backend + UI):** overfitting stats (PBO/DSR/PSR), trade + Monte Carlo metrics, QuantStats tear-sheet (`GET /api/runs/{id}/tearsheet`), `allocation_method`/`kelly_fraction`/`signal_combination`, news-guard blackout, and the `FIRM_ALLOW_TRADING` execution lock. Surfaced in `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx`. See `docs/PROJECT_CONTEXT.md` → "REST API & Web UI wiring". `CLAUDE.md` mirrors this file for Claude Code.
