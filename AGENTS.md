# Agent Context

Pointers for AI coding agents working in this repository.

| Resource | Purpose |
|----------|---------|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, deployment, live config, IBKR pitfalls |
| [.cursor/rules/](.cursor/rules/) | Concise agent memory (project context, live trading, strategies, IBKR) |
| [deploy/ai-trading.service](deploy/ai-trading.service) | Production systemd unit (`firm-api` + auto-start live) |
| [config/live.yaml](config/live.yaml) | Canonical live paper trading configuration |

**Production on bare-metal:** `ai-trading.service` runs `firm-api`, not `scripts/run_live_trading.py`. Set `FIRM_AUTO_START_LIVE=1` to boot live from `config/live.yaml`.
