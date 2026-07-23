# Agent Context

Pointers for AI coding agents working in this repository.

| Resource | Purpose |
|----------|---------|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, deployment, live config, IBKR pitfalls |
| [.cursor/rules/](.cursor/rules/) | Concise agent memory (project context, live trading, strategies, IBKR, logging, frontend) |
| [deploy/ai-trading.service](deploy/ai-trading.service) | Production systemd unit (`firm-api` + auto-start live) |
| [config/live.yaml](config/live.yaml) | Canonical live paper trading configuration |

**Production on bare-metal:** `ai-trading.service` runs `firm-api`, not `scripts/run_live_trading.py`. Set `FIRM_AUTO_START_LIVE=1` to boot live from `config/live.yaml`.

**Logging:** every module uses stdlib `logging` (`log = logging.getLogger(__name__)`); log decisions, fallbacks, I/O failures, and safety events — no `print`, no bare `except: pass`. See `.cursor/rules/logging.mdc`.

**Frontend:** the React dashboard is mobile-first responsive (usable at ~375px); grids start at `grid-cols-1/2` and scale with `md:`/`lg:`, header rows wrap. Keep `src/api/{types,client}.ts` in sync with backend. See `.cursor/rules/frontend.mdc`.

**Eval & behavioural features (wired backend + UI):** overfitting stats (PBO/DSR/PSR), trade + Monte Carlo metrics, QuantStats tear-sheet (`GET /api/runs/{id}/tearsheet`), `allocation_method`/`kelly_fraction`/`signal_combination`, news-guard blackout, and the `FIRM_ALLOW_TRADING` execution lock. Surfaced in `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx`. See `docs/PROJECT_CONTEXT.md` → "REST API & Web UI wiring". `CLAUDE.md` mirrors this file for Claude Code.
