# Project Context

Canonical reference for architecture, production deployment, and conventions in this repository. Cursor agent rules in `.cursor/rules/` summarize the actionable parts; this document is the full source of truth.

---

## Architecture

### Stack

| Layer | Technology | Entry point |
|-------|------------|-------------|
| API + web UI | FastAPI + React (`frontend/dist` mounted at `/`) | `firm-api` → `src/firm/api/app.py` |
| Backtests | Backtrader + PIT data store | `python scripts/run_backtest.py` |
| Live trading | `LiveTradingEngine` + APScheduler | `POST /api/live/start` or auto-start on boot |
| LLM / RAG | LiteLLM + ChromaDB | `config/llm.yaml` |

### Agent pipeline

```
12 Strategies → 3 Analysts (technical, fundamental, sentiment)
              → Bull / Bear researchers → Debate
              → Portfolio Manager → Risk Manager (veto) → Execution
```

- **Strategies** emit raw cross-sectional scores (no internal z-scoring).
- **Analysts** are the sole `zscore_signals` pass before the research layer.
- Each agent can run in quant, AI-enhanced, or AI-only mode (`config/llm.yaml` → `agent_modes`).

### Twelve strategies

| Name | Module | Notes |
|------|--------|-------|
| momentum | `momentum.py` | 12-1 month cross-sectional momentum |
| trend | `trend.py` | Moving-average trend |
| mean_reversion | `mean_reversion.py` | Short-horizon reversion |
| stat_arb | `stat_arb.py` | Log-price OLS pairs; cointegration gate; nets one signal per symbol |
| multi_factor | `multi_factor.py` | Needs fundamentals (FMP on IBKR live) |
| sentiment | `sentiment.py` | News/sentiment scores |
| event_driven | `event_driven.py` | Simplified PEAD proxy; needs fundamentals |
| ml_prediction | `ml_prediction.py` | PIT-safe ML features |
| volatility_breakout | `volatility_breakout.py` | Vol breakout |
| seasonality | `seasonality.py` | Calendar effects; TOM uses trading days (`pd.bdate_range`) |
| gann | `gann.py` | Heuristic composite (not academic Gann) |
| regime_hmm | `regime_hmm.py` | Per-symbol HMM regime overlay |

Register new strategies with `@register` in `src/firm/strategies/registry.py`.

### Key packages

```
src/firm/
  strategies/          # Signal generation
  agents/              # Analysts, researchers, PM, risk, execution
  live/                # LiveTradingEngine, scheduler, approval queue
  api/routers/live.py  # Live REST API + bootstrap
  data/providers/      # IBKR, FMP, Fallback chain
  brokers/             # Alpaca, IBKR execution adapters
```

---

## Production deployment (systemd)

On the primary bare-metal host, production runs via **systemd**, not `scripts/run_live_trading.py`.

### Services

| Unit | Role |
|------|------|
| `ibgateway.service` | IB Gateway (paper API on port 4002) |
| `ai-trading.service` | `firm-api` — API, web UI, live engine |

Template unit file: [`deploy/ai-trading.service`](../deploy/ai-trading.service)

```ini
WorkingDirectory=/local/store/git/ai-trading-system
EnvironmentFile=/local/store/git/ai-trading-system/.env
Environment=FIRM_AUTO_START_LIVE=1
ExecStart=/local/store/git/ai-trading-system/.venv/bin/firm-api
After=network.target ibgateway.service
```

### Environment variables (live / IBKR)

```env
IBKR_HOST=127.0.0.1
IBKR_PAPER_PORT=4002
IBKR_CLIENT_ID=1          # broker execution adapter
FMP_API_KEY=...           # fundamentals for multi_factor / event_driven
FIRM_AUTO_START_LIVE=1    # auto-start live from config/live.yaml on API boot
FIRM_ALLOW_TRADING=0      # execution-safety hard lock (see below); 1 arms live brokers
```

Data provider uses `client_id=2` (separate from broker `IBKR_CLIENT_ID`).

### Execution-safety hard lock (`FIRM_ALLOW_TRADING`)

`firm.live.execution_safety` gates every order in `LiveTradingEngine._execute_orders`:
a **live** broker (`ibkr` / `ibkr_live` / `alpaca_live`) will not submit unless
`FIRM_ALLOW_TRADING=1` is set in the service environment — a human-only switch on
top of the approval queue. Paper brokers ignore it. Blocked orders are surfaced
as a `live_trading_locked` alert and appended to an audit JSONL
(`data/execution_audit.jsonl`, override with `FIRM_EXECUTION_AUDIT`). Default
(unset/0) keeps live routing blocked. The module also exposes a standalone
`guard_order(order, RiskProfile, live=, confirmation=)` with hard risk caps and a
typed confirmation token for CLI/manual use.

### Auto-start flow

When `FIRM_AUTO_START_LIVE=1`:

1. `firm-api` starts uvicorn on port 8000.
2. FastAPI lifespan spawns a worker thread after ~1 s.
3. `bootstrap_live_from_yaml()` calls `resolve_live_startup()` and starts the engine.
4. Failures are logged; the API keeps running so you can start live manually.

**Critical:** IBKR `connect()` must not run on uvicorn's asyncio loop. Auto-start uses `asyncio.to_thread()` for the same reason `POST /api/live/start` runs in a sync handler (thread pool).

### Operations

```bash
sudo systemctl restart ai-trading
sudo systemctl status ai-trading
sudo journalctl -u ai-trading -f

curl -s http://127.0.0.1:8000/api/live/status | python3 -m json.tool
curl -X POST http://127.0.0.1:8000/api/live/stop
curl -X POST http://127.0.0.1:8000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{"broker":"ibkr_paper"}'
```

An empty `POST /api/live/start` body merges defaults from `config/live.yaml`.

---

## `config/live.yaml` — canonical live config

[`config/live.yaml`](../config/live.yaml) is the source of truth for paper trading on this host:

- **broker**: `ibkr_paper`
- **schedule**: `market_open`
- **approval_mode**: `full_auto` (only `full_auto` and `semi_auto` are valid)
- **universe**: 30 symbols (mega-cap, ETFs including SPY/QQQ/IWM)
- **strategies**: all 12 enabled with full auto-approve
- **initial_capital**: 1_000_000
- **strategy_params.stat_arb**: predefined pairs, `require_cointegration: true`
- **risk**: flattened into engine config (kill switch 8%, position limits, regime overlay)

### What `resolve_live_startup()` merges

Implemented in [`src/firm/live/provider_utils.py`](../src/firm/live/provider_utils.py):

| YAML key | Engine field |
|----------|--------------|
| `broker` | Broker type string |
| `schedule` | APScheduler schedule |
| `approval_mode` | Approval queue mode |
| `universe.symbols` | `symbols` |
| `strategies.enabled` | `strategies` |
| `strategies.auto_approve` | `auto_approve_strategies` |
| `strategy_params` | Per-strategy params for `build_orchestrator` |
| `initial_capital` | `initial_capital` |
| `risk.*` | Flattened into engine config (kill switch, exposure limits, `regime_overlay`, etc.) |
| `news_guard` | Macro-event blackout gate (default OFF) |
| `signal_combination` | Research combine method: `confidence` (default) or `optimal` |
| `allocation_method` / `kelly_fraction` | `TraderAgent` sizing (`kelly` uses `kelly_fraction`) |

Explicit API request fields override YAML when provided (non-null / non-empty).

### Optional behavioural knobs (all default OFF / unchanged)

| Knob | Module | Effect |
|------|--------|--------|
| `news_guard.enabled` | `firm.live.news_guard` | Holds orders inside a high-impact economic-event window (FOMC/NFP/CPI); offline fallback in `src/firm/live/data/events.csv` |
| `signal_combination.method: optimal` | `firm.agents.analysts` | Inverse-covariance signal weighting (down-weights correlated strategies) + effective-N; needs `ctx.strategy_returns` |
| `allocation_method: kelly` | `firm.agents.trader` | Fractional-Kelly sizing from per-name return history (`kelly_fraction`, default half-Kelly) |
| `FIRM_ALLOW_TRADING` | `firm.live.execution_safety` | Hard env lock; live brokers won't submit unless `=1` |

### Live start paths

| Path | When to use |
|------|-------------|
| `FIRM_AUTO_START_LIVE=1` + systemd | Production on this host |
| `POST /api/live/start` | Manual start / dashboard |
| `scripts/run_live_trading.py --config` | Standalone CLI / debugging (not production here) |

---

## REST API & Web UI wiring

The eval/behavioural features are wired end-to-end (backend endpoints + React UI). Keep both sides in sync when extending.

### Backtest & walk-forward (`src/firm/api/routers/runs.py`, `schemas.py`)

| Surface | Detail |
|---------|--------|
| `RunRequest` / `WalkForwardRequest` | Accept optional `allocation_method`, `kelly_fraction`, `signal_combination` (fall back to `settings.*`). Threaded into the flat config → `build_orchestrator`. |
| `POST /api/runs/walk_forward` | Aggregate response includes an `overfitting` block (`pbo`, `deflated_sharpe`, `probabilistic_sharpe`, `verdict`) from `ExperimentRunner._walk_forward_overfitting`. `_flatten_config` passes the three knobs through per fold. |
| `GET /api/runs/{id}/report` | Raw `report.json` — now also carries `trade_metrics` and `monte_carlo` blocks when available (from `BacktestReport.to_dict`). |
| `GET /api/runs/{id}/tearsheet` | Renders/caches a QuantStats HTML tear-sheet (`firm.eval.tearsheet`). Requires the optional `report` extra installed server-side, else returns a clear error. |

### Live config (`src/firm/api/routers/live.py`)

| Surface | Detail |
|---------|--------|
| `GET /api/live/config` | Returns `news_guard`, `signal_combination`, `allocation_method`, `kelly_fraction` (running-engine + no-engine/YAML branches). |
| `PUT /api/live/config` | Round-trips the same keys; applies via engine setters below. |
| `POST /api/live/start` (`StartRequest`) | Optional `news_guard` / `signal_combination` / `allocation_method` / `kelly_fraction` override the resolved YAML `engine_config`. |
| `GET /api/config/defaults` | Exposes `allocation_method`, `kelly_fraction`, `signal_combination` so the UI can seed controls. |

Engine setters (`src/firm/live/engine.py`), effective next cycle:

- `update_news_guard(enabled, before_min, after_min, offline)` — sets `_news_guard_*` attrs, keeps `_config['news_guard']` in sync.
- `update_signal_combination(cfg)` and `update_allocation(method, kelly_fraction)` — merge into `_config` and **rebuild the orchestrator** (researchers/TraderAgent read config at construction).

### Web UI surfaces (`frontend/src`)

| Page | Adds |
|------|------|
| `pages/RunDetail.tsx` | Trade-Level Metrics grid, Monte Carlo Robustness block, "Open Tear-Sheet ↗" button (`api.tearsheetUrl`). |
| `pages/NewBacktest.tsx` | Allocation method + Kelly fraction + signal combination controls (seeded from `/config/defaults`); walk-forward Overfitting Diagnostics panel. |
| `pages/LiveConfig.tsx` | Allocation & Signal Combination section + News-Guard blackout section (round-trip via `PUT /live/config`). |
| `api/types.ts`, `api/client.ts` | Types for the above + `tearsheetUrl(id)` helper. |

---

## Data providers (IBKR live)

[`build_live_providers()`](../src/firm/live/provider_utils.py) wires:

| Capability | IBKR live | Alpaca / other |
|------------|-----------|----------------|
| prices | `IBKRProvider` | `FallbackProvider` chain |
| sentiment | `IBKRProvider` | `FallbackProvider` |
| fundamentals | `FMPProvider` if `FMP_API_KEY` set | `FallbackProvider` |

[`filter_strategies_for_providers()`](../src/firm/live/provider_utils.py) drops `multi_factor` and `event_driven` when no fundamentals feed is available.

---

## Frontend (mobile-responsive)

The React dashboard is **mobile-first** and must stay usable at ~375px — see [`.cursor/rules/frontend.mdc`](../.cursor/rules/frontend.mdc).

- `components/Layout.tsx` is the mobile shell: slide-in sidebar + overlay (`md:hidden`/`md:static`), mobile top bar, responsive padding (`p-4 md:p-6`). Primary breakpoint is `md` (768px).
- New sections: grids start at `grid-cols-1`/`grid-cols-2` and scale with `md:`/`lg:`; header/action rows use `flex flex-wrap … gap` with `flex-shrink-0` actions and `min-w-0` text; inputs are `w-full`; tables use `overflow-x-auto`.
- Keep `src/api/types.ts` + `src/api/client.ts` (and `src/test/handlers.ts`) in sync with backend endpoints.

## Logging & traceability

Every module is traceable via stdlib `logging` — see [`.cursor/rules/logging.mdc`](../.cursor/rules/logging.mdc).

- Module header: `log = logging.getLogger(__name__)`; no `print`, no handler config in library code; `%`-style lazy args.
- Log **decisions/branches**, **fallbacks** (never a bare `except: pass`), **external I/O outcomes**, and **safety-critical events** (execution-gate blocks, news-guard blackouts, kill-switch, risk breaches) with the audit id / symbol / cycle.
- `execution_safety.py` additionally appends every decision to an immutable audit JSONL (`data/execution_audit.jsonl`).
- Reference modules: `eval/{overfitting,robustness,tearsheet,metrics}.py`, `live/{news_guard,execution_safety}.py`, `agents/research/_combine.py`, `agents/trader.py`, `experiments/runner.py`.

## Strategy conventions

1. **Raw scores only** — strategies must not call `zscore_signals`; analysts normalize.
2. **stat_arb** — log prices, OLS hedge ratio, optional Engle-Granger cointegration (`require_cointegration` default true), one net signal per symbol, predefined pairs in YAML to avoid correlation mining.
3. **seasonality** — turn-of-month uses trading sessions, not calendar days.
4. **Backtest parity** — `config/settings.yaml` mirrors `strategy_params` for backtests.

### Known simplifications vs literature

- `event_driven`: not true SUE/PEAD; earnings-proxy heuristic.
- `regime_hmm`: per-symbol HMM (unusual vs market-level regime models).
- `gann`: heuristic composite, not formal Gann analysis.

---

## IBKR threading constraint

Documented in [`src/firm/brokers/ibkr.py`](../src/firm/brokers/ibkr.py):

- `ib_async` binds Futures to the **calling thread's** event loop.
- `connect()` must run on a dedicated thread (same thread for `reqAccountSummary`, `qualifyContracts`, `reqContractDetails`).
- `is_market_open()` reads cached contract details from connect — calling ib_async from APScheduler or FastAPI worker threads without that cache can **hang indefinitely**.
- Production incident: scheduled cycle hung 24+ hours when `is_market_open()` ran on a different thread than `connect()`.

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Engine stuck | `cycle_running_seconds` in `/api/live/status`; restart service or stop/start live |
| `multi_factor` missing | `FMP_API_KEY` in `.env`; check logs for provider filter warning |
| IBKR connect fails on boot | Gateway up? `nc -zv 127.0.0.1 4002`; auto-start logs in journalctl |
| `this event loop is already running` | IBKR connect called from asyncio lifespan — must use worker thread |
| Orders queued forever | `approval_mode` must be `full_auto` or `semi_auto` |
| Only 11 strategies active | `regime_hmm` or fundamental strategies filtered — check provider keys |

---

## File map

| Path | Purpose |
|------|---------|
| `config/live.yaml` | Live paper experiment config |
| `config/settings.yaml` | Backtest defaults + `strategy_params` |
| `config/llm.yaml` | LLM providers, agent modes, RAG |
| `src/firm/live/provider_utils.py` | YAML merge, provider wiring, strategy filtering |
| `src/firm/live/news_guard.py` | Macro-event blackout pre-trade gate (+ `data/events.csv`) |
| `src/firm/live/execution_safety.py` | `FIRM_ALLOW_TRADING` live lock, `RiskProfile`, audit JSONL |
| `src/firm/eval/overfitting.py` | PBO (CSCV) / Deflated & Probabilistic Sharpe |
| `src/firm/eval/robustness.py` | Monte Carlo bootstrap (drawdowns, prob-of-loss, CI) |
| `src/firm/eval/metrics.py` | Return + trade-level metrics (profit factor, expectancy) |
| `src/firm/eval/tearsheet.py` | QuantStats HTML tear-sheet (optional `report` extra) |
| `src/firm/api/routers/live.py` | Live API, bootstrap, engine lifecycle, live-config round-trip |
| `src/firm/api/routers/runs.py` | Backtest/walk-forward launch, report, equity, tear-sheet endpoints |
| `src/firm/api/app.py` | FastAPI factory, lifespan auto-start |
| `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx` | UI for trade/MC/overfitting metrics, tear-sheet, allocation/combination, news-guard |
| `deploy/ai-trading.service` | systemd unit for production |
| `scripts/run_live_trading.py` | CLI alternative (not used in prod systemd path) |
| `AUTOMATED_TRADING_GUIDE.md` | Operator guide for paper trading |
| `DEPLOY.md` | Docker, Droplet, bare-metal deployment |

---

## Related docs

- [DEPLOY.md](../DEPLOY.md) — Docker, cloud, bare-metal setup
- [AUTOMATED_TRADING_GUIDE.md](../AUTOMATED_TRADING_GUIDE.md) — Paper trading operations
- [AGENTS.md](../AGENTS.md) — Pointer for AI agents
