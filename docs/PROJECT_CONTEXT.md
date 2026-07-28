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
| trend | `trend.py` | MA crossover strength `(fast−slow)/slow` — not direction/vol |
| mean_reversion | `mean_reversion.py` | Short-horizon reversion |
| stat_arb | `stat_arb.py` | Log-price OLS pairs; cointegration gate; nets one signal per symbol |
| multi_factor | `multi_factor.py` | Value/quality/momentum/low-vol; omits `low_vol` when fundamentals missing |
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

`firm.live.execution_safety` gates every order in `LiveTradingEngine._execute_orders`
with two independent checks, in order:

1. `guard_order(order, RiskProfile, live=False)` — a final, RiskAgent-independent
   hard cap: symbol must be in the engine's configured universe, and order notional
   must be under `2 × max_position_pct × NAV` (doubled vs. `config/live.yaml`
   `risk.max_position_pct` to allow a legitimate full-position flip). Runs with
   `require_stop=False` — this engine rebalances to target weights, not per-trade
   stops, unlike `guard_order`'s CLI/manual use (typed confirmation token,
   stop/ATR risk sizing), which stays available standalone. Failures raise an
   `order_risk_cap_blocked` alert.
2. `guard_live_submission` — a **live** broker (`ibkr` / `ibkr_live` / `alpaca_live`)
   will not submit unless `FIRM_ALLOW_TRADING=1` is set in the service environment —
   a human-only switch on top of the approval queue. Paper brokers ignore it.
   Failures raise a `live_trading_locked` alert. Default (unset/0) keeps live
   routing blocked.

Both append to the same immutable audit JSONL (`data/execution_audit.jsonl`,
override with `FIRM_EXECUTION_AUDIT`).

### Auto-start flow

When `FIRM_AUTO_START_LIVE=1`:

1. `firm-api` starts uvicorn on port 8000.
2. FastAPI lifespan spawns a worker thread after ~1 s.
3. `bootstrap_live_from_yaml()` calls `resolve_live_startup()` and starts the engine.
4. Failures are logged; the API keeps running so you can start live manually.

**Critical:** IBKR `connect()` must not run on uvicorn's asyncio loop. Auto-start uses `asyncio.to_thread()` for the same reason `POST /api/live/start` runs in a sync handler (thread pool).

### Durable live state (kill switch, portfolio history, attribution)

Three pieces of `LiveTradingEngine` state must survive a process restart
(`systemctl restart ai-trading`, a redeploy, or a crash) without an operator
having to notice and intervene:

| State | Mechanism | File | Read on startup? |
|-------|-----------|------|-------------------|
| Kill switch (`_halted`, `_peak_equity`) | JSON file | `data/kill_switch_state.json` | **Yes** — `_load_kill_switch_state()`; a halted engine restarts halted. |
| Portfolio NAV/equity-curve history | SQLite blob | `data/live_state.db` (`firm.live.state_store.LiveStateStore`) | Yes — `_load_persisted_state()`; cash/holdings still come from the broker, only history is restored. |
| Per-strategy attribution (`PerformanceAttribution`) | SQLite blob | `data/live_state.db` | Yes — same load call; restores `_strategy_returns`/`_trade_log`/`_strategy_holdings` so the `optimal` signal-combination method doesn't reset to empty history on every restart. |

Both `kill_switch_state_path` and `state_db_path` are constructor kwargs that
default to `None` (fully disk-free) — every test and any direct
`LiveTradingEngine(...)` construction stays isolated; only
`_start_live_engine()` in `src/firm/api/routers/live.py` points them at the
real production paths. `LiveStateStore` stores each piece of state as a
single JSON blob under a well-known key (not per-row upserts) — this state
is always read/written as one document, live/paper cadence is at most a few
cycles a minute, and a JSON blob of a few thousand data points is trivially
cheap to rewrite in full every cycle; `save_portfolio_history` also mirrors
the current kill-switch state into the same DB (`save_kill_switch`) purely
as an additional durable copy — the JSON file remains the actual mechanism
read on startup. Persistence runs from `_persist_cycle_result()` (after
every cycle attempt, including skipped/errored ones) and once more on
`stop()`.

**Related hardening**: wiring the durable attribution store surfaced a latent
fragility in the `record_trades()` call site — it fed order dicts straight
into `PerformanceAttribution.record_trades()`, which expects a signed
`shares` field. `ExecutionAgent`-produced orders already carry `shares`
(so this was never actually broken in the real pipeline), but any other
order source that only supplies `side` + unsigned `quantity` (a mocked
orchestrator in tests, or a future execution path) would hit a `KeyError`
inside `record_trades`, silently swallowed by a broad
`except Exception: log.debug(...)`, quietly recording nothing. Added
`LiveTradingEngine._orders_to_fills()` as a defensive normalizer at the
call site so attribution recording no longer depends on every upstream
order producer happening to include `shares`.

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

## Broker & host failover

Production today is a **single bare-metal host** running `ibgateway.service` +
`ai-trading.service` (see "Production deployment" above) — there is no standby
host or automatic multi-host failover. This section is the runbook for the
failure modes that *can* happen on that one host, split into what's already
automatic vs. what needs a human.

### Broker (IBKR) disconnects — mostly automatic

| Scenario | What happens automatically | What a human does |
|----------|------------------------------|--------------------|
| A single cycle's broker call fails (dropped socket, transient network blip) | `LiveTradingEngine` catches the `BrokerError`, calls `broker.reconnect()` inline on the cycle worker thread (`IBKRBroker`/`AlpacaBroker` use the `Broker.reconnect()` default: `disconnect()` then `connect()`), and emits a `broker_unavailable` alert noting whether the reconnect succeeded. | Nothing, unless alerts keep recurring — check `GET /api/live/alerts`. |
| Reconnect also fails, or several consecutive cycles fail | Alert escalates from `broker_unavailable` to **`broker_disconnected_sustained`** (severity `critical`) once `broker_disconnect_alert_threshold` (default 3, `config/live.yaml`) consecutive cycles have failed. `_consecutive_broker_failures` and `reconnected` are in the alert context. | IB Gateway is very likely actually down or logged out — see checklist below. A `broker_reconnected` alert fires automatically once a cycle's broker calls succeed again; no manual "un-halt" is needed (this is not the drawdown kill switch). |
| **IB Gateway's mandatory daily restart** lands mid-cycle | The stale connection can hang a blocking IBKR call with no error; `cycle_hard_timeout_seconds` (900s) releases the cycle lock so future cycles aren't blocked forever, and `cycle_watchdog_seconds` (1800s) fires an observational `cycle_watchdog_timeout` alert if a cycle is still running past that. The *next* cycle's first broker call will raise `BrokerError` (dead socket) and go through the same reconnect path above. | If cycles keep failing past the daily-restart window (a few minutes), follow the IB Gateway checklist below — this is the scenario the reconnect logic is least battle-tested against, since the whole `IB()` session is gone, not just one call. |
| Broker down while the process is fully stopped/starting | `LiveTradingEngine.start()` retries `connect()` 3× internally (`IBKRBroker.connect()`) before raising; `bootstrap_live_from_yaml()` (auto-start) logs the failure and leaves the API running without a live engine rather than crashing the whole process. | `POST /api/live/start` once IB Gateway is confirmed up (see checklist). |

**What's deliberately *not* automatic:** the drawdown kill switch does **not**
trip on a broker disconnect — a disconnect already prevents new orders from
being submitted (nothing dangerous happens while down), so halting trading on
top of that would just add another manual `kill-switch/reset` step for
operators once the broker comes back. If IB Gateway is down, no orders can be
placed anyway.

### IB Gateway down / needs restart — checklist

1. Confirm the process state: `sudo systemctl status ibgateway`.
2. Check it's actually listening: `nc -zv 127.0.0.1 4002` (paper) — see
   `AUTOMATED_TRADING_GUIDE.md` for the equivalent live port.
3. Tail its logs for a login/2FA/session-limit issue (IBKR allows only one
   active Gateway/TWS session per account — a second login elsewhere silently
   kicks this one): `sudo journalctl -u ibgateway -f`.
4. Restart it: `sudo systemctl restart ibgateway`. `Restart=always` (`setup.sh`,
   `DEPLOY.md`) means an unattended crash already restarts on its own; a
   *stuck-but-alive* process (e.g. a frozen 2FA prompt) needs the manual
   restart since systemd sees it as still running.
5. Once Gateway is confirmed up, `ai-trading.service` does **not** need a
   restart — the next scheduled cycle's reconnect logic (above) picks the
   connection back up on its own. If it doesn't within a couple of cycles
   (watch `GET /api/live/alerts` for `broker_reconnected` vs. repeated
   `broker_disconnected_sustained`), restart the app service too:
   `sudo systemctl restart ai-trading`.
6. **Verify unit naming if `ai-trading.service` never seems to wait for
   Gateway on boot**: `deploy/ai-trading.service`'s `After=`/`Wants=` and the
   actual installed IB Gateway unit name must match exactly
   (`ibgateway.service`, no hyphen, on the current production host) —
   `systemctl list-units | grep -i gateway` to check what's actually
   installed. `setup.sh`/`DEPLOY.md` install it under this same name; a
   mismatch here (e.g. a manually created `ib-gateway.service`) makes systemd
   silently skip the dependency ordering with no error.

### Host crash / process restart — what's recovered vs. lost

See "Durable live state" above for the full mechanism; summarized for
incident response:

| Survives a crash + restart | Rebuilt from the broker each cycle | Lost (acceptable) |
|---|---|---|
| Kill switch halt state (`data/kill_switch_state.json`) — a halted engine restarts halted, it does **not** silently resume | Cash, holdings, open orders — `sync_portfolio_from_broker()` treats the broker as the source of truth every cycle, not just at startup | In-flight cycle (at most one; never partially double-submits since orders aren't retried blind) |
| Portfolio NAV/equity-curve history, per-strategy attribution (`data/live_state.db`) | — | In-memory-only counters (`_cycle_count` continuity across the exact same process) |
| Pending manual approvals (`data/approvals.json`) | — | — |
| Execution audit trail (`data/execution_audit.jsonl`, append-only) | — | — |

**Recovery steps after any host/process crash:**

1. `sudo systemctl status ai-trading ibgateway` — confirm both came back
   (`Restart=always` on both units should have already done this).
2. `curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool` — process
   liveness; `broker.connected` reflects IBKR specifically (this endpoint
   deliberately stays `"ok"` even when the broker is down, so infra doesn't
   restart the API in a loop during IB Gateway's daily restart).
3. `curl -s http://127.0.0.1:8000/api/live/status | python3 -m json.tool` —
   check `halted` (was the kill switch tripped before the crash?) and
   `broker_connected`.
4. If `halted: true` and the drawdown trip was a real risk event (not a data
   glitch), investigate before resetting — `POST
   /api/live/kill-switch/reset` re-arms trading immediately.
5. Positions/cash need no manual reconciliation — the first cycle after
   restart re-syncs both from the broker automatically.

### Losing the host entirely (disk failure, VPS termination, etc.)

There is no warm standby today, so this is a manual rebuild, not a failover:

1. Provision a new host and follow `DEPLOY.md` end-to-end (or `setup.sh`) to
   install IB Gateway + `ai-trading.service`.
2. Restore `.env` (broker credentials, API keys) from your secrets backup —
   these are deliberately never committed to the repo.
3. Restore the `data/` directory from backup if you have one (kill-switch
   state, `live_state.db`, execution audit) — **optional**, not required for
   correctness: if `data/` is missing entirely, the engine starts fresh
   (un-halted, empty history) and re-syncs cash/holdings from the broker on
   the first cycle, same as any restart. Only do this if you specifically
   want to preserve halt state or historical continuity.
4. Log into IB Gateway on the new host with the same account — **IBKR allows
   only one active Gateway/TWS session per account**, so the old host's
   Gateway session must actually be down first, not just the trading process.
5. Before setting `FIRM_AUTO_START_LIVE=1` / calling `POST /api/live/start`,
   confirm via `GET /api/live/status` (or the IBKR TWS/Gateway UI directly)
   that positions match what you expect — the engine trusts the broker as
   ground truth on the very first cycle, so if the *account itself* has
   unexpected positions (e.g. you're pointed at the wrong account), it will
   adopt them silently rather than erroring.

### Monitoring recommendations (not yet automated end-to-end)

- Poll `GET /api/health` and `GET /api/live/status` externally (e.g. cron +
  curl, or a real uptime monitor) — `broker.connected=false` sustained across
  several polls is the earliest external signal of the disconnect scenarios
  above, ahead of the in-engine `broker_disconnected_sustained` alert
  threshold.
- Set `ALERT_WEBHOOK_URL` (`firm.live.notifications.build_alert_callback()`)
  to route `broker_disconnected_sustained`, `cycle_watchdog_timeout`, and
  `drawdown_breach` alerts to Slack/email/pager rather than relying on
  someone tailing `journalctl` or polling `/api/live/alerts`.
- No built-in Prometheus/Datadog exporter exists; the JSON endpoints above are
  the integration point if you wire one up.

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
| `news_guard.enabled` | `firm.live.news_guard` | Holds orders inside a high-impact economic-event window (FOMC/NFP/CPI); offline fallback in `src/firm/live/data/events.csv`. Fails **closed**: if the calendar can't be loaded at all (live fetch *and* the bundled CSV both fail), every order is held that cycle with a `critical` `news_guard_calendar_unavailable` alert rather than approved blind; landing on the bundled CSV after a live-fetch failure still succeeds but raises a `warning` `news_guard_stale_calendar` alert (age of the CSV included) since a static calendar can miss events added after it was last updated. |
| `signal_combination.method: optimal` | `firm.agents.analysts` | Inverse-covariance signal weighting (down-weights correlated strategies) + effective-N; needs `ctx.strategy_returns` |
| `strategy_circuit_breaker.enabled` | `firm.agents.research._circuit_breaker` | Damps a strategy's raw signal when its trailing realized Sharpe is persistently negative. **Disabled by default** — an A/B found the default thresholds net *hurt* portfolio Sharpe (see "Portfolio-construction diagnosis" below); opt in only for further calibration. |
| `strategy_regime_weights.enabled` | `firm.agents.research._regime_weights` | Per-strategy score multipliers conditioned on Bull/Bear/Chop regime (detected once per cycle). **Disabled by default** — calibrate via `scripts/calibrate_strategy_regime_weights.py` before enabling live. |
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
| `POST /api/runs/walk_forward` | `n_splits`/`train_pct` plus optional `param_grid` (list of config overrides) + `selection_metric` (default `sharpe_ratio`) for genuine train→select→test optimization — see below. Aggregate response includes an `overfitting` block (`pbo`, `pbo_n_folds`, `deflated_sharpe`, `probabilistic_sharpe`, `verdict`) from `ExperimentRunner._walk_forward_overfitting`. `_flatten_config` passes the three knobs through per fold. |
| `GET /api/runs/{id}/report` | Raw `report.json` — now also carries `trade_metrics` and `monte_carlo` blocks when available (from `BacktestReport.to_dict`). |
| `GET /api/runs/{id}/tearsheet` | Renders/caches a QuantStats HTML tear-sheet (`firm.eval.tearsheet`). Requires the optional `report` extra installed server-side, else returns a clear error. |

### Genuine walk-forward optimization + PBO trial semantics (`experiments/runner.py`)

`ExperimentRunner.run_walk_forward` supports two modes:

- **No `param_grid` (default)**: each fold just backtests the input config unchanged
  over its test window — a plain sequential OOS replay, not an optimization (nothing
  to select between with one candidate). Matches pre-redesign behavior exactly.
- **`param_grid` with ≥2 candidate config overrides**: each fold backtests *every*
  candidate on its **train** window, picks the best by `selection_metric` (default
  `sharpe_ratio`), and only that winner — not the base config — runs on the **test**
  window. This is genuine train→select→test optimization. Every such fold writes
  `walk_forward_selection.json` into its artifacts dir (candidates tried, the winner,
  each candidate's train-window per-period returns).

`ExperimentRunner._walk_forward_overfitting` / `eval.overfitting.walk_forward_overfitting`
then compute PBO/DSR from those genuine per-fold trials (CSCV within each fold's own
train-period returns matrix, trial Sharpes pooled across folds for DSR) instead of the
old heuristic of treating sequential OOS folds as pseudo-trials. **`pbo` is omitted
entirely** (not estimated) when no fold has real multi-candidate data — a fabricated
PBO from folds-as-trials would misrepresent what was actually tested. DSR degrades to
plain PSR in that case for the same reason (there is genuinely only one trial).

### Equity-curve / warmup-trim fix (`backtest/engine.py`, `backtest/run.py`)

Two related bugs, fixed together because the walk-forward/PBO redesign above depends
on both: (1) `BacktestEngine.generate_report()` always produced an **empty**
`report.snapshots` — `PortfolioState.record_snapshot()` is only ever called from the
live-trading path (`live/portfolio_sync.py`), never from the backtest loop — so
`build_equity_data()` (the dashboard equity curve, `final_nav`/`period` in
`report.json`, and any NAV-based OOS-return reconstruction) was silently empty for
*every* backtest. Fixed with a fallback that builds NAV-only `PortfolioSnapshot`s from
the same `detailed_returns` curve that already feeds `report.returns`. (2)
`execute_backtest()` only trimmed the pre-`start_date` warmup padding from
`report.returns`/`benchmark_returns` for non-synthetic data sources; synthetic
backtests pad ~252 calendar days of history *before* `start_date` too (for
long-lookback strategies) but were never trimmed, silently diluting every synthetic
backtest's Sharpe/vol with a block of flat, zero-return "no positions yet" days. Both
paths are now trimmed identically to `[start_date, end_date]`.

### Point-in-time universe membership (`data/universe.py`, `data/pit_store.py`, `backtest/firm_strategy.py`)

`UniverseResolver` already computed survivorship-aware membership as-of a single date
(`symbols_asof`); two gaps remained for a full backtest window: (1) feed loading only
used a `start_date` (or `end_date`) snapshot, so a symbol that **joins** the index
mid-backtest never had its price feed loaded at all, and (2) `FirmStrategy` built its
`PitViewAdapter` from the static `self.p.universe` passed in at `engine.setup()`, so
even when a feed *was* loaded, the strategy layer never saw membership change
mid-run — a delisted name kept being "tradable" forever and a newly-added name was
never tradable.

- `UniverseResolver.symbols_between(start, end)` — union of every symbol that was a
  member at *any point* within `[start, end]` (interval-overlap, not just endpoint
  snapshots — a name that both joins and leaves entirely inside the window is still
  included). `delisted_between(start, end)` returns symbols removed within the window.
- `PointInTimeDataStore.get_universe_union(start, end)` — uses `symbols_between` when a
  resolver is installed; degrades to `get_universe(start) ∪ get_universe(end)` if the
  resolver predates that method, or to the raw loaded-price symbol set with no resolver
  at all (each degradation step logs a warning).
- `execute_backtest()` (`backtest/run.py`) and `scripts/run_backtest.py` now call
  `pit_store.get_universe_union(start_date, end_date)` — instead of a single
  `get_universe(start_date)` snapshot — to decide which feeds to load. This is a
  superset; it does not by itself make anything tradable.
- `FirmStrategy.next()` resolves the actually-tradable subset **every rebalance** via
  `_active_universe(current_dt)` = `pit_store.get_universe(current_dt) ∩ data_map.keys()`,
  and only that subset is passed into `PitViewAdapter` for strategies/orchestrator to
  see. Mark-to-market pricing and short-borrow accrual, however, iterate over **all**
  loaded feeds (`self._data_map`), not just the active universe, so an already-open
  position in a name that gets delisted mid-backtest is still priced/charged correctly
  until it's closed out — only *new* entries are blocked once a symbol drops out.
- Net effect: with a real `UniverseResolver` (real `added_date`/`removed_date` data)
  installed, a backtest is free of both look-ahead survivorship bias (dead names
  correctly disappear) and missed-entrant bias (index adds correctly become tradable
  once added) without manual universe curation per fold.
- Still a fallback, not yet closed out: `build_resolver()` degrades to a static,
  always-active symbol list when no historical membership dataset is cached (see
  `longer-dataset` follow-up task) — the *engineering* is real point-in-time, the
  *default dataset* is not yet.

### Real fundamentals filing dates (`data/providers/base.py`, `edgar.py`, `fmp.py`)

Every fundamentals provider previously stamped a fundamentals row `period_end +
FUNDAMENTALS_PUBLICATION_LAG_DAYS` (45 days) — a conservative *estimate* of when a
filing became public, not the real date. `resolve_filing_date(period_end, filed,
symbol=...)` (`data/providers/base.py`) now prefers a genuine filing date when a
provider actually exposes one, falling back to the 45-day heuristic (unchanged)
otherwise:

- **`EdgarProvider`** — SEC EDGAR's XBRL `companyfacts` API tags every fact with the
  real `filed` date. `_series_by_period` now returns `(value, filed)` pairs per
  period, and `_companyfacts_to_rows` uses the **latest** `filed` date across every
  concept (revenue, net income, EPS, assets, equity, liabilities) contributing to a
  period — a 10-K/A restating one line item shouldn't make the whole row appear
  knowable earlier than its real availability.
- **`FMPProvider`** — the `/stable/income-statement` endpoint exposes `fillingDate`;
  merged into the (fiscalYear, period)-keyed ratios row and preferred over the
  heuristic.
- **`MassiveProvider`, `TwelveDataProvider`, `AlphaVantageProvider`, `FinnhubProvider`**
  — verified their fundamentals endpoints (`/stocks/financials/v1/ratios`, statistics
  snapshots, company overview) genuinely don't expose a filing/announcement date, only
  the accounting period-end — the 45-day heuristic remains the correct choice there,
  not a shortcut.

### Size/volume-aware market impact (`agents/_liquidity.py`, `backtest/firm_strategy.py`, `agents/execution.py`)

Flat-percentage transaction costs (`commission_pct`/`spread_pct`) charge the same rate
whether an order is 0.1% or 50% of a name's daily volume — unrealistic for anything but
small, liquid trades. `market_impact_coefficient` (default `0.0` = disabled; `0.005` in
`config/settings.yaml`/`config/live.yaml`) adds a square-root-law cost on top:
`impact_pct = coefficient * sqrt(participation)`, where `participation = trade notional /
trailing ADV dollars`. `agents/_liquidity.py` (`estimate_adv_dollars`, `sqrt_impact_pct`) is
shared by three call sites so they all agree on what "ADV" means:

- `RiskAgent._cap_liquidity` — the participation-rate liquidity cap (already existed).
- `backtest/firm_strategy.py` (`FirmStrategy._apply_market_impact`) — recomputes each
  traded symbol's `PercentageCommission` scheme (`broker.addcommissioninfo(comm,
  name=symbol)`) immediately before submitting that rebalance's order, so backtrader's
  real fill reflects the estimate; mirrors the identical `impact_pct` into the secondary
  `PortfolioState` book's cost calculation so both stay consistent. Refreshes (even to
  `0.0`) on every trade of a symbol once the model is enabled, so a stale large-order
  impact rate never lingers on a later, smaller/thinner-data trade.
- `agents/execution.py` (`ExecutionAgent._estimate_impact_cost`) — adds the same estimate
  into each order's `est_cost` for live pre-trade cost visibility (not an actual broker
  fee — IBKR fills at whatever the market gives; this is a modeled estimate surfaced to
  the audit trail/dashboard).

Wired through `BacktestConfig.market_impact_coefficient`, `RunRequest.market_impact_coefficient`
(API), and `frontend/src/pages/NewBacktest.tsx` (Capital & Costs section). `adv_lookback_days`
(the trailing window, default 20) is shared with `RiskAgent`'s own liquidity cap config.

### Live config (`src/firm/api/routers/live.py`)

| Surface | Detail |
|---------|--------|
| `GET /api/live/config` | Returns `news_guard`, `signal_combination`, `strategy_circuit_breaker`, `strategy_regime_weights`, `allocation_method`, `kelly_fraction` (running-engine + no-engine/YAML branches). |
| `PUT /api/live/config` | Round-trips the same keys; applies via engine setters below. |
| `POST /api/live/start` (`StartRequest`) | Optional `news_guard` / `signal_combination` / `strategy_circuit_breaker` / `strategy_regime_weights` / `allocation_method` / `kelly_fraction` override the resolved YAML `engine_config`. |
| `GET /api/config/defaults` | Exposes `allocation_method`, `kelly_fraction`, `signal_combination`, `strategy_circuit_breaker`, `strategy_regime_weights` so the UI can seed controls. |

Engine setters (`src/firm/live/engine.py`), effective next cycle:

- `update_news_guard(enabled, before_min, after_min, offline)` — sets `_news_guard_*` attrs, keeps `_config['news_guard']` in sync.
- `update_signal_combination(cfg)`, `update_strategy_circuit_breaker(cfg)`, `update_strategy_regime_weights(cfg)`, and `update_allocation(method, kelly_fraction)` — merge into `_config` and **rebuild the orchestrator** (researchers/TraderAgent read config at construction).

### Portfolio-construction diagnosis follow-up: `regime_hmm` fix + strategy circuit breaker (`regime/model.py`, `strategies/regime_hmm.py`, `agents/research/_circuit_breaker.py`)

Follow-up to `docs/portfolio_construction_diagnosis.md`, which found `regime_hmm` had
a negative Sharpe in 6/6 diagnostic windows. Two independent mechanisms were built and
A/B-tested against the same 3 historical windows used in that diagnosis:

1. **Signal-logic fix (shipped, on by default)** — `GaussianRegimeModel` now reports a
   per-label `separation` effect size (`_build_separation`): the gap between the
   labelled Bull/Bear state's mean return and its nearest-ranked neighbour, normalised
   by pooled standard deviation. A thin margin means the label is statistically
   indistinguishable from noise and prone to label-switching between refits.
   `HMMRegimeStrategy` damps (`min_state_separation` default `0.5`, floor
   `separation_damping_floor` default `0.15`) any Bull/Bear signal whose separation
   falls below threshold, rather than trading it at full confidence.
   **Result**: a controlled A/B (`min_state_separation=0.5` vs `0.0`, full 12-strategy
   pipeline, `optimal` combination, same 3 windows) flipped `regime_hmm`'s own
   attributed Sharpe from negative to positive in **all 3** tested windows
   (-0.20→+0.66, -0.78→+1.64, -2.73→+1.37). Portfolio-level Sharpe improved
   substantially in 2/3 windows but *worsened* in the third (`wf_fold1`, already
   flagged in the original diagnosis as a short/low-signal window) — likely from
   `optimal`'s inverse-covariance reweighting reacting to `regime_hmm`'s now-different
   correlation with the other 11 strategies. Net: the specific strategy-health problem
   this item was scoped to fix is resolved; broader portfolio-construction interactions
   remain an open research question (see `regime-conditional-weighting` backlog item).
2. **Generic per-strategy rolling-Sharpe circuit breaker (shipped, off by default)** —
   `agents/research/_circuit_breaker.py` damps any strategy's raw signal contribution
   (applied in `net_scores_for_blackboard`, upstream of both `confidence` and `optimal`
   combination) when its trailing realized Sharpe from `PerformanceAttribution` is
   persistently below `trigger_sharpe` (default `-0.5` over `lookback_days=60`,
   floored at `damping_floor=0.25` past `full_cutoff_sharpe=-1.5`). Complementary to
   `optimal`, which has no notion of a strategy's edge *sign* — a low-variance,
   steadily negative-mean strategy can still receive material minimum-variance weight.
   **Result**: an A/B with these exact default thresholds over the same 3 windows
   *hurt* portfolio Sharpe in all 3 — a noisy 60-day trailing Sharpe over-gated
   several volatile-but-legitimate strategies (gann, stat_arb, mean_reversion, etc.),
   most of a 12-strategy blend on any given cycle. **Left disabled by default**
   (`strategy_circuit_breaker.enabled: false` in both `config/settings.yaml` and
   `config/live.yaml`); fully wired through `RunRequest`/`WalkForwardRequest`,
   `POST/PUT /api/live/*`, and `frontend/src/pages/{NewBacktest,LiveConfig}.tsx`
   (marked "experimental" in the UI) for future recalibration/research rather than
   left unusable.

`ctx.strategy_returns` (from `PerformanceAttribution.get_all_strategy_returns()`) is
now populated unconditionally in both the backtest (`FirmStrategy.next()`) and live
(`LiveTradingEngine.run_cycle`) paths — previously gated on `signal_combination.method
== "optimal"` — since the circuit breaker needs it regardless of combination method.

### Web UI surfaces (`frontend/src`)

| Page | Adds |
|------|------|
| `pages/RunDetail.tsx` | Trade-Level Metrics grid, Monte Carlo Robustness block, "Open Tear-Sheet ↗" button (`api.tearsheetUrl`). |
| `pages/NewBacktest.tsx` | Allocation method + Kelly fraction + signal combination controls (seeded from `/config/defaults`); Strategy Circuit Breaker section (marked experimental); walk-forward Overfitting Diagnostics panel. |
| `pages/LiveConfig.tsx` | Allocation & Signal Combination section + Strategy Circuit Breaker section (experimental) + News-Guard blackout section (round-trip via `PUT /live/config`). |
| `api/types.ts`, `api/client.ts` | Types for the above + `tearsheetUrl(id)` helper. |

---

## Data providers (IBKR live)

[`build_live_providers()`](../src/firm/live/provider_utils.py) wires:

| Capability | IBKR live | Alpaca / other |
|------------|-----------|----------------|
| prices | `IBKRProvider` | `FallbackProvider` chain |
| sentiment | `IBKRProvider` | `FallbackProvider` |
| fundamentals | `FallbackProvider` chain; **cache-only** live cycles; daily refresh via APScheduler in `firm-api` (`fundamentals_refresh_hour` in `live.yaml`) | `FallbackProvider` |

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
4. **Backtest parity** — `config/settings.yaml` mirrors `strategy_params`, `risk.sector_map`, and behavioural knobs for backtests.
5. **Cached fundamentals in backtests** — when `data_source` is not `synthetic`, `load_fundamentals()` reads `data/cache` (`combined/fundamentals`) into the PIT store via `execute_backtest`, `run_backtest_from_config`, and the CLI — same panel live gets from FMP when keyed.
6. **Risk sector cap** — set `risk.sector_map` in `config/live.yaml` / `settings.yaml` so `max_sector_pct` is enforced (without it the risk agent logs a warning each rebalance).

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

## Minimum paper track record (live promotion gate)

Before promoting a strategy mix or behavioural knob from backtest research to **enabled in production `config/live.yaml`**, require a minimum paper track record on IBKR paper (same universe, costs, and pipeline as live):

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Calendar duration | ≥ **90 trading days** | Continuous run on `ai-trading.service`; gaps require restart clock |
| Realized Sharpe (daily) | ≥ **0.5** | Computed from live cycle NAV / `data/cycle_history.json`; not backtest Sharpe |
| Max drawdown | ≤ **15%** | Peak-to-trough on paper equity curve |
| Kill-switch trips | **0** unexplained trips | Manual operator resets are logged; investigate any auto trip |
| Execution gate blocks | Documented | `FIRM_ALLOW_TRADING`, news-guard, and risk-limit blocks must be understood, not ignored |
| LLM A/B (if applicable) | Arm completes runbook | See `docs/llm_ab_test_runbook.md` — do not enable LLM-heavy modes without the quant-only baseline arm |

**Process:** (1) backtest + walk-forward validation on cache data, (2) enable on paper with knob **off** or at research default, (3) observe through one full macro regime if possible, (4) only then set `enabled: true` in `live.yaml` or via `PUT /api/live/config`. Document the decision in `docs/remediation_progress.md` or an experiment log.

**Universe membership:** survivorship-aware backtests require `data/cache/combined/universe_membership` (see `scripts/import_universe_membership.py` and `docs/longer_dataset_options.md`). Paper promotion does not replace this — it validates execution, slippage, and operational risk on the *current* universe.

### Real-capital allocation gate

Separate, higher bar than the live-promotion gate above — this one gates moving
from paper to **real money**, not just enabling a knob in paper `config/live.yaml`.
Originally scoped as a flat 6-12 calendar months; revised to a trade-count/
statistical-significance bar plus tranched capital, since a fixed calendar window
doesn't itself guarantee enough trades to trust a Sharpe estimate, and an
all-or-nothing capital switch adds unnecessary risk versus starting small:

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Duration | ≥ **60 trading days** (~3 months) | Floor, not sufficient alone — see trade count below |
| Trade count | ≥ **100** executed orders across the roster | Guards against a quiet period satisfying the calendar floor with too few trades to say anything statistically |
| Realized Sharpe (daily) | **Bootstrap 90% CI lower bound > 0** | Point estimate alone (as in the paper-promotion gate) isn't enough for real capital; use `eval/robustness.py`'s Monte Carlo bootstrap against the live NAV series, not a plain in-sample Sharpe |
| Max drawdown | ≤ **15%** | Peak-to-trough on paper equity curve |
| Kill-switch trips | **0** unexplained trips | Manual operator resets are logged; investigate any auto trip |
| LLM A/B (if applicable) | Arm completes runbook | See `docs/llm_ab_test_runbook.md` |

**Initial allocation is tranched, not all-or-nothing:** fund at **10-20%** of the
intended target size first; only scale toward full size after a second
observation period (same criteria, shorter — e.g. 30 trading days) confirms
performance held with real fills. This shortens time-to-first-capital versus a
flat 6-12mo wait while keeping the actual dollars at risk small until the edge is
confirmed with real money on the line, not just simulated.

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
| `src/firm/time_utils.py` | `utcnow()` — deprecation-safe naive-UTC "now"; use instead of `datetime.utcnow()` anywhere the value may reach a PIT/pandas comparison |
| `src/firm/live/news_guard.py` | Macro-event blackout pre-trade gate (+ `data/events.csv`) |
| `src/firm/live/execution_safety.py` | `FIRM_ALLOW_TRADING` live lock, `RiskProfile`, audit JSONL |
| `src/firm/eval/overfitting.py` | PBO (CSCV) / Deflated & Probabilistic Sharpe; `walk_forward_overfitting` takes optional genuine per-fold trial returns |
| `src/firm/experiments/runner.py` | Walk-forward runner: `param_grid` → genuine per-fold train→select→test optimization + `walk_forward_selection.json` |
| `src/firm/eval/robustness.py` | Monte Carlo bootstrap (drawdowns, prob-of-loss, CI) |
| `src/firm/eval/metrics.py` | Return + trade-level metrics (profit factor, expectancy) |
| `src/firm/eval/tearsheet.py` | QuantStats HTML tear-sheet (optional `report` extra) |
| `src/firm/api/routers/live.py` | Live API, bootstrap, engine lifecycle, live-config round-trip |
| `src/firm/api/routers/runs.py` | Backtest/walk-forward launch, report, equity, tear-sheet endpoints |
| `src/firm/api/app.py` | FastAPI factory, lifespan auto-start |
| `frontend/src/pages/{RunDetail,NewBacktest,LiveConfig}.tsx` | UI for trade/MC/overfitting metrics, tear-sheet, allocation/combination, news-guard |
| `deploy/ai-trading.service` | systemd unit for production |
| `scripts/import_universe_membership.py` | Vendor membership CSV → `combined/universe_membership` |
| `scripts/etl_sharadar_to_cache.py` | Sharadar SEP/SF1 bulk CSV → combined prices/fundamentals cache |
| `scripts/calibrate_strategy_regime_weights.py` | A/B regime weights on diagnostic windows |
| `scripts/suggest_strategy_regime_weights.py` | Data-driven draft regime weight table |
| `scripts/run_walk_forward_pbo_audit.py` | Walk-forward + PBO/DSR audit CLI |
| `scripts/snapshot_llm_ab_arm.py` | Weekly LLM A/B NAV/Sharpe snapshot |
| `AUTOMATED_TRADING_GUIDE.md` | Operator guide for paper trading |
| `DEPLOY.md` | Docker, Droplet, bare-metal deployment |

---

## Related docs

- [DEPLOY.md](../DEPLOY.md) — Docker, cloud, bare-metal setup
- [AUTOMATED_TRADING_GUIDE.md](../AUTOMATED_TRADING_GUIDE.md) — Paper trading operations
- [AGENTS.md](../AGENTS.md) — Pointer for AI agents
- [longer_dataset_options.md](longer_dataset_options.md) — Vendor scoping for delisting-inclusive history
- [longer_dataset_vendor_decision.md](longer_dataset_vendor_decision.md) — Recommended vendor + approval checklist
- [formal_pbo_audit.md](formal_pbo_audit.md) — First walk-forward PBO audit results
- [strategy_regime_weights_calibration.md](strategy_regime_weights_calibration.md) — Regime weight A/B (v1/v2)
- [llm_ab_test_runbook.md](llm_ab_test_runbook.md) — Quant vs LLM paper experiment procedure
