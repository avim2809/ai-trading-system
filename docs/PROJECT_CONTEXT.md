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
13 Strategies → 3 Analysts (technical, fundamental, sentiment)
              → Bull / Bear researchers → Debate
              → Portfolio Manager → Risk Manager (veto) → Execution
```

- **Strategies** emit raw cross-sectional scores (no internal z-scoring).
- **Analysts** are the sole `zscore_signals` pass before the research layer.
- Each agent can run in quant, AI-enhanced, or AI-only mode (`config/llm.yaml` →
  `agent_modes`) — **currently live**: `fundamental_analyst`/`sentiment_analyst` are
  `llm_enhanced` (Arm B of the quant-vs-LLM A/B, started 2026-09-08 per
  `docs/llm_ab_experiment_log.md`), everything else stays `quant` (no per-symbol LLM
  fan-out for bull/bear/debate/trader/risk).
- **Per-strategy capital sleeves** (`capital_allocation_mode: "blended" | "sleeved"`,
  see "Per-strategy capital sleeves" below) — currently `sleeved` on the Alpaca instance
  only, `blended` (default, unchanged) on IBKR as the static control.

### Thirteen strategies

| Name | Module | Notes |
|------|--------|-------|
| momentum | `momentum.py` | 12-1 month cross-sectional momentum |
| trend | `trend.py` | MA crossover strength `(fast−slow)/slow` — not direction/vol |
| mean_reversion | `mean_reversion.py` | Short-horizon reversion |
| stat_arb | `stat_arb.py` | Log-price OLS pairs; cointegration gate; nets one signal per symbol |
| multi_factor | `multi_factor.py` | Value/quality/momentum/low-vol; omits `low_vol` when fundamentals missing |
| sentiment | `sentiment.py` | News/sentiment scores |
| event_driven | `event_driven.py` | Simplified PEAD proxy; needs fundamentals |
| ml_prediction | `ml_prediction.py` | PIT-safe ML features — registered but **permanently disabled** in live configs (overfit risk on the 25-name universe) |
| volatility_breakout | `volatility_breakout.py` | Vol breakout |
| seasonality | `seasonality.py` | Calendar effects; TOM uses trading days (`pd.bdate_range`) |
| gann | `gann.py` | Heuristic composite (not academic Gann) — registered but **permanently disabled** (no timing value found on any asset class tested) |
| regime_hmm | `regime_hmm.py` | Per-symbol HMM regime overlay |
| pattern_recognition | `pattern_recognition.py` | Strategy #13 (added 2026-09-09): multi-bar chart patterns (H&S, triangles, flags, cup & handle — Lo/Mamaysky/Wang 2000 geometric framework), quality-scored. Full history: `docs/pattern_recognition_plan.md` |

Of these 13, **11 are enabled** in both `config/live.yaml`/`config/live_alpaca.yaml`
(`ml_prediction`/`gann` are the two permanently-disabled exceptions above).

The registry (`src/firm/strategies/registry.py`, `list_strategies()`) actually holds
**18** `@register`-decorated classes today — the 13 above plus `investing_analyst_ratings`
and four `danelfin_*` variants. **Danelfin was fully decommissioned 2026-08-16** (user
closed the account) — its strategy/provider modules still exist under `src/firm/` but are
commented out of both live configs' `strategies.enabled`; see
`docs/danelfin_best_stocks_arm.md` for its A/B history if the vendor is ever reinstated.
`investing_analyst_ratings` is also registered but not currently enabled — see
`docs/investing_pro_integration.md`.

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
| `TraderAgent` conviction-EMA + joint-optimizer NAV history | SQLite blob | `data/live_state.db` | Yes — otherwise every restart re-enters unsmoothed at full strength, exactly what the smoothing exists to damp. |
| Cycle counter, daily trade/turnover counters | SQLite blob | `data/live_state.db` | Yes — the cycle counter keeps `client_order_id`s unique across a same-day restart (Alpaca previously rejected a regenerated duplicate id). |
| Per-sleeve `PortfolioState` + per-sleeve `TraderAgent` state (`capital_allocation_mode: "sleeved"` only) | SQLite blob | `data/live_state.db` | Yes — sleeves are virtual (no real broker sub-account to reconcile from), so without this a restart *or* a config hot-swap (`update_strategies` etc., which rebuilds the orchestrator) would silently reset every sleeve to its initial capital split. See "Per-strategy capital sleeves" above. |

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

### Live reflection (`agents/memory.py`, `TradingMemoryLog`)

Append-only JSONL decision log (`data/memory/decisions.jsonl`, path configurable via
`memory_log_path`): `store_decision()` writes a pending entry per cycle (target-weight
breakdown, not the `PerformanceAttribution` metrics above); a later `reflect()` call adds
an LLM-generated verdict/lesson once the realized and benchmark return for that decision
are known. Purely read-only/commentary today — `get_context()`/`summarize_lessons()` only
feed future prompts, nothing here adjusts weights, disables a strategy, or touches config.
`benchmark_return` (the SPY-based comparison reflection needs) was fixed 2026-09-08: it
previously collapsed to a false-precision `0.0` in a real-but-narrow case rather than the
real SPY return for that period (commits `7abdbe9`, `983c5a2`) — a cadence caveat from
that fix is still open, see `project_reflection_and_llm_fallback_fixes_sep8.md` in
`docs/claude-memory/`.

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
| **IB Gateway's mandatory daily restart** lands mid-cycle | The stale connection can hang a blocking IBKR call with no error; `cycle_hard_timeout_seconds` (900s) releases the cycle lock so future cycles aren't blocked forever, and `cycle_watchdog_seconds` (1800s) fires an observational `cycle_watchdog_timeout` alert if a cycle is still running past that. The *next* cycle's first broker call will raise `BrokerError` (dead socket) and go through the same reconnect path above. `IBKRBroker.health_check()` (called both at top-of-cycle and again immediately before order submission, 2026-08-23) is now **two-stage**: `reqCurrentTime()` catches a fully dead socket, then `reqContractDetails()` on the cached SPY contract exercises the same backend `qualifyContracts` depends on — confirmed live that a Gateway mid-reconnect-to-its-own-backend passed the old single-stage check and then hung every order's `qualifyContracts` call anyway. Orders also no longer each pay their own `qualifyContracts` round-trip: `IBKRBroker.warm_universe()` batch-qualifies the whole universe once per connect/reconnect (conId is stable, survives reconnects). | If cycles keep failing past the daily-restart window (a few minutes), follow the IB Gateway checklist below. |
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
| Portfolio NAV/equity-curve history, per-strategy attribution (`data/live_state.db`) | — | — |
| **`_cycle_count`** (`data/live_state.db`, `save_cycle_counter`/`load_cycle_counter`) — **not** acceptable to lose, despite looking like an in-memory-only counter: `client_order_id = f"c{cycle_id}-{order_index}-{symbol}-{side}"`, so a restart that reset it to 0 could regenerate an id already submitted earlier that day. Confirmed live: Alpaca rejected the repeat with `"client_order_id must be unique"` (2026-08-17), tripping the submission circuit breaker. Fixed 2026-08-23 (`docs/remediation_progress.md` #57). | — | — |
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

### Host resource exhaustion (disk / memory / CPU) — mostly automatic

Unlike a broker disconnect, the host running out of disk, memory, or CPU
headroom isn't something the trading pipeline can route around — it's the
substrate everything else runs on. `TradingScheduler` runs a
`resource_health_check` job every 15 minutes (`firm.live.scheduler.
run_resource_health_check`, stdlib-only: `shutil.disk_usage`,
`/proc/meminfo`, `os.getloadavg()`) that samples the host and pushes a real
alert through the same pipeline as a kill-switch trip or a broker outage —
log line, `GET /api/live/alerts`, and the configured webhook.

| Metric | Warn | Critical | Why here |
|---|---|---|---|
| Disk free (on the filesystem holding the repo + `data*/` dirs) | ≤5GB free or ≥85% used | ≤2GB free or ≥95% used | Both a free-GB and a used-% trigger fire independently — either alone crossing its bar is enough, so this still catches a much bigger disk that's simply filling up as well as this one's actual 20GB root volume. |
| Memory available (`MemAvailable`, not `MemFree` — accounts for reclaimable cache) | ≤15% of total | ≤8% of total | Expressed as a % of total so the same default still means something if the box is ever resized. |
| CPU load, 5-min average per core | ≥1.5 | ≥3.0 | The 5-minute figure, not 1-minute, so a brief burst doesn't page anyone; sustained overload on a 2-core box does. |

All six numbers are overridable per-instance via `HEALTH_CHECK_<NAME>` env
vars (e.g. `HEALTH_CHECK_DISK_WARN_FREE_GB`) without a code change — see the
`_HEALTH_CHECK_DEFAULTS` dict in `firm.live.scheduler` for the exact keys.
Each metric only re-alerts on a severity *change* (into a breach, an
escalation, or a recovery back to "ok" — sent as an `info`-severity
`..._recovered` alert), not every 15 minutes it stays breached, so a
sustained issue doesn't spam the webhook.

This job runs regardless of whether the trading engine itself is running or
halted — the host can run low on resources either way — and is registered
once per live instance (IBKR on :8000, Alpaca on :8001 each run their own
scheduler against the same physical disk, so a disk alert from either one
means the same underlying filesystem).

**What this does *not* cover:** per-process memory (e.g. a leak inside this
one Python process specifically, as opposed to the host overall), inode
exhaustion, and any filesystem other than the one holding the repo. None of
these have caused a real incident here; add a check if one does.

#### Disk-space crisis — runbook

What to do, in order, if a `host_disk_low` alert fires (or `df -h /` shows
the disk critically full) — reclaiming space safely without touching
anything the live engines need to keep running:

1. **Confirm severity and where the space actually went:**
   ```
   df -h /
   du -xhd1 / 2>/dev/null | sort -rh | head -10   # top-level offenders
   du -xhd1 /local/store/git/ai-trading-system 2>/dev/null | sort -rh | head -15
   ```
2. **Check the usual suspects first, cheapest/safest to clear:**
   - `journalctl --disk-usage` then `journalctl --vacuum-size=200M` (or
     `--vacuum-time=7d`) — systemd's own journal is unbounded by default on
     many installs and is pure log history, never state this system needs.
   - `/var/cache/apt` (`apt-get clean`) and `~/.cache/pip` — package-manager
     caches, always safe to clear and always safe to repopulate.
   - Stray SQLite `-wal`/`-shm` sidecar files under `data*/` that are large
     relative to their `.db` (`ls -la data*/*.db*`) — normal in small
     amounts (an open connection's not-yet-checkpointed writes); one that's
     grown to many times its `.db`'s size suggests something is holding a
     long-lived read transaction open and blocking `PRAGMA wal_checkpoint`,
     worth investigating rather than just deleting (deleting a `-wal` file
     that hasn't been checkpointed loses those writes).
   - Test-suite artifacts: a misconfigured mock in a test that patches
     `firm.config.get_settings()` without setting `cache_dir`/similarly
     path-like fields can make a real directory tree accumulate on disk on
     every test run (this has happened before — see git history for
     "disk-bloat"). `find . -maxdepth 2 -iname "*MagicMock*"` from the repo
     root is the tell; if found, it's a test bug to fix, not just a
     directory to delete.
3. **Never delete, even under pressure:** anything under `data/` or
   `data_alpaca/` that isn't a `.db-wal`/`.db-shm` sidecar —
   `kill_switch_state.json`, `live_state.db`, `approvals.json`,
   `execution_audit.jsonl`, `dynamic_universe_state.json` are exactly the
   durable state this system depends on (see "Host crash / process restart"
   above); `.env` (never committed, no other copy unless you made one).
4. **If genuinely out of easy slack**, `.venv` (~2GB) is fully
   reproducible from `pyproject.toml`/`requirements` via `setup.sh`, and
   `frontend/node_modules` (if present; the running services only need
   `frontend/dist`) is reproducible via `npm install` — both are legitimate
   to delete and rebuild if nothing above freed enough, but rebuilding
   `.venv` on this 2-core box is slow, so exhaust steps 2-3 first.
5. **Re-check `df -h /`** and confirm the `host_disk_low` alert clears (an
   `info`/`..._recovered` alert fires automatically once the next
   15-minute check sees it below the warn threshold again — no manual
   reset needed).

### Losing the host entirely (disk failure, VPS termination, etc.)

There is no warm standby today, so this is a manual rebuild, not a failover:

1. Provision a new host and follow `DEPLOY.md` end-to-end (or `setup.sh`) to
   install IB Gateway + `ai-trading.service`.
2. Restore `.env` (broker credentials, API keys) from your secrets backup —
   these are deliberately never committed to the repo.
3. Restore the `data/`/`data_alpaca/` state (kill-switch state,
   `live_state.db`, execution audit, decision memory) from
   `scripts/backup_live_state.sh`'s daily archives (`deploy/
   backup-live-state.timer`, see "Backups" below) if you have them —
   **optional**, not required for correctness: if `data/` is missing
   entirely, the engine starts fresh (un-halted, empty history) and
   re-syncs cash/holdings from the broker on the first cycle, same as any
   restart. Only do this if you specifically want to preserve halt state or
   historical continuity. **Note this backup lives on the same disk as
   everything else** — it survives an accidental `rm -rf data/`, not a
   whole-disk failure; if this scenario is a whole-disk loss, this backup
   is gone too unless you've separately copied it off-box.
4. Log into IB Gateway on the new host with the same account — **IBKR allows
   only one active Gateway/TWS session per account**, so the old host's
   Gateway session must actually be down first, not just the trading process.
5. Before setting `FIRM_AUTO_START_LIVE=1` / calling `POST /api/live/start`,
   confirm via `GET /api/live/status` (or the IBKR TWS/Gateway UI directly)
   that positions match what you expect — the engine trusts the broker as
   ground truth on the very first cycle, so if the *account itself* has
   unexpected positions (e.g. you're pointed at the wrong account), it will
   adopt them silently rather than erroring.

### Backups

`scripts/backup_live_state.sh`, run daily by `deploy/backup-live-state.timer`
(+ `.service`, both `systemctl enable --now backup-live-state.timer` once
installed like the other units in `deploy/`), tars `data/`/`data_alpaca/`
(excluding the large, fully-reproducible `vectordb`/`cache`/`logs`/`models`
subdirs) to `/local/store/backups/ai-trading-live-state/` — a different
directory tree on the **same** disk, not a different disk. It keeps the
newest 14 daily archives (`LIVE_STATE_BACKUP_RETAIN`) and each one is a few
MB, negligible next to the constraints in "Host resource exhaustion" above.

**What this protects against:** an accidental `rm -rf data/`, a bad script,
a botched manual edit — the archive is a separate, untouched copy.
**What this does *not* protect against:** losing the disk itself (hardware
failure, corrupted filesystem) — there is exactly one physical disk on this
host (`lsblk`/`df -h /` show a single `sda2` volume), so anything that takes
the disk down takes both the live data *and* this backup with it. Real
protection against that needs the backup archive copied somewhere off this
box — network storage, another host, a cloud bucket — none of which are
configured today; wiring that up is an infrastructure decision (credentials,
a destination, egress cost) for a human to make, not something to bolt on
silently. Until that exists, this on-disk backup is strictly better than
the "nothing at all" status quo, not a substitute for a real one.

### Monitoring recommendations

- **Host disk/memory/CPU is now automated** — see "Host resource
  exhaustion" above. What's still *not* automated: an external
  dead-man's-switch if the process itself stops running entirely (systemd's
  `Restart=always` + geometric backoff, see below, handles the process
  coming back; nothing external confirms it actually did).
- Poll `GET /api/health` and `GET /api/live/status` externally (e.g. cron +
  curl, or a real uptime monitor) — `broker.connected=false` sustained across
  several polls is the earliest external signal of the disconnect scenarios
  above, ahead of the in-engine `broker_disconnected_sustained` alert
  threshold.
- Set `ALERT_WEBHOOK_URL` (`firm.live.notifications.build_alert_callback()`)
  to route `broker_disconnected_sustained`, `cycle_watchdog_timeout`,
  `drawdown_breach`, and `host_disk_low`/`host_memory_low`/`host_cpu_high`
  alerts to Slack/email/pager rather than relying on someone tailing
  `journalctl` or polling `/api/live/alerts`.
- No built-in Prometheus/Datadog exporter exists; the JSON endpoints above are
  the integration point if you wire one up.
- **True redundancy would need a second host** — both live instances share
  one physical box, one disk, one kernel. Nothing in this section makes a
  hardware failure survivable; it only makes an *impending* resource problem
  visible before it becomes one, and gets a crashed process back up faster
  without hammering whatever it crashed against. A real warm/cold standby on
  separate hardware is a genuine improvement but an infrastructure decision
  (a second box, plus a real off-box backup destination — see "Durable live
  state" backups below), not something to build silently into this repo.
- `ai-trading.service`/`ai-trading-alpaca.service`'s `Restart=always` now
  backs off geometrically on repeated failures (`RestartSteps=4`,
  `RestartMaxDelaySec=160s`: 10s/20s/40s/80s, then holding at 160s) instead
  of retrying every 10s indefinitely — a single transient crash still
  recovers just as fast as before (first retry unchanged at 10s); a
  persistent failure no longer hammers whatever it's failing against (e.g.
  repeated IB Gateway login attempts) every 10s forever.

---

## `config/live.yaml` — canonical live config

[`config/live.yaml`](../config/live.yaml) is the source of truth for paper trading on this host:

- **broker**: `ibkr_paper`
- **schedule**: `hourly_market_hours` (2026-09-18) — ~7 cycles/trading day (9:30
  open, hourly on the half-hour 10:30-14:30, a 15:50 close-anchor), up from
  the prior once/day `market_open`-only cadence; registers 3 separate
  APScheduler jobs (`TradingScheduler._start_hourly_market_hours_jobs`), each
  passing an explicit `cycle_type` (`"open"|"intraday"|"close"`) through
  `LiveTradingEngine.run_cycle` to `Orchestrator.step`
- **llm_open_close_only**: `true` — LLM-enhanced agent reasoning
  (`agent_modes`) only runs on the day's open/close cycles; every intraday
  cycle forces `cache_only` (no live LLM calls) via
  `Orchestrator._apply_cycle_llm_mode`, keeping LLM cost flat despite the
  higher cadence above
- **approval_mode**: `full_auto` (only `full_auto` and `semi_auto` are valid)
- **universe**: 25 symbols (mega-cap, ETFs including SPY/QQQ/IWM)
- **strategies**: 11 of 13 enabled with full auto-approve (`ml_prediction`/`gann`
  permanently disabled; `pattern_recognition` added 2026-09-09)
- **capital_allocation_mode**: `blended` (default/unset) — IBKR is the static control for
  the capital-sleeves A/B; `config/live_alpaca.yaml` sets `sleeved` instead, see
  "Per-strategy capital sleeves" below
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
| `allocation_method: joint_optimizer` | `firm.agents.trader` / `firm.portfolio.optimizer` | Joint mean-variance-with-costs QP (`cvxpy`) replacing L1-normalize-to-full-investment sizing. **Disabled by default (not shipped anywhere) — failed its walk-forward+PBO gate** (see `docs/formal_pbo_audit.md`'s `joint_optimizer` section / `docs/remediation_progress.md` #61-62); kept as validated, tested, off-by-default infrastructure only. |
| `risk.stop_loss_overlay.enabled` | `firm.agents.risk` | Portfolio-level cycle-gated stop-loss (see "Stop-loss / trailing-stop / extended-hours orders" above). **Disabled by default** — pending backtest validation before flipping on for either instance. |
| `protective_orders` | `firm.agents.execution` | Broker-side stop/trailing-stop orders per strategy (see same section above). **Disabled by default (`{}`)** — also requires a real `broker` handle, only present on the live path (never in a backtest). |
| `extended_hours_trading.enabled` | `firm.live.scheduler` / `firm.live.engine` | Opt-in premarket/afterhours cycles + `OrderRequest.extended_hours` (see "Extended-hours trading" above). **Disabled by default (`{}`)** — not set in either shipped `config/live*.yaml`. |
| `llm_open_close_only` | `firm.agents.orchestrator` | Restricts LLM-enhanced `agent_modes` to the day's open/close cycles; every intraday cycle forces `cache_only`. **Enabled by default (`true`)** — deliberately not an opt-in, since it's what keeps the 2026-09-18 `hourly_market_hours` cadence cost-flat. |
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

### Stop-loss / trailing-stop / extended-hours orders (2026-09-18)

Two independent, both **opt-in and off by default**, layers — built after an audit found
`mean_reversion`/`stat_arb` were the worst live performers on both instances, with neither
strategy's own documented "a stop-loss is advisable" ever implemented anywhere:

- **`RiskAgent._stop_loss_overlay`** (`config: risk.stop_loss_overlay`) — portfolio-level,
  cycle-gated: forces a held symbol's *target weight* to 0 once its unrealized loss
  (`PortfolioState.unrealized_return_pct`, new avg-cost tracking) breaches `max_loss_pct`,
  for strategies listed in `strategies`. Exact in sleeved mode (the portfolio passed in
  *is* that strategy's own book); approximated in blended mode via the same
  dominant-strategy-by-symbol heuristic `PerformanceAttribution` already uses for order
  attribution.
- **`ExecutionAgent._maybe_submit_protective_order`** (`config: protective_orders`, e.g.
  `{"mean_reversion": {"stop_loss_pct": 0.07}}` or `{"trailing_stop_pct": 0.05}`) —
  broker-resident: submits a real stop/trailing-stop order (new `OrderRequest` order
  types wired into both `IBKRBroker`/`AlpacaBroker`) that can fire *between* scheduled
  cycles, not just at the next one. **Only fires on a flat -> open transition**
  (`_is_fresh_open`) — deliberately narrower than "opens or increases," since this agent
  has no broker order-ID tracking/cancellation, so re-firing on every incremental add
  would stack a new resting stop on top of each earlier one instead of replacing it.
  Requires a real `broker` handle, threaded via `context["broker"]`
  (`LiveTradingEngine` -> `Orchestrator.step`/`_step_sleeved` -> `ExecutionAgent.run`); in
  sleeved mode only the final netted `_real_execution` pass gets it, never the per-sleeve
  virtual pass (`sleeve_portfolio` has no broker sub-account).
- **Extended-hours**: `OrderRequest.extended_hours` sets Alpaca's `extended_hours` flag
  (limit orders only — other order types silently degrade to regular-hours with a logged
  warning, since Alpaca's API doesn't honor it elsewhere) or IBKR's `outsideRth` (every
  order type).

### Extended-hours trading (2026-09-19, opt-in and off by default)

Wires the `OrderRequest.extended_hours` flag above into the live path — previously it
existed on the order schema/both brokers but nothing ever set it. Config key
`extended_hours_trading` (`{"enabled": false, "premarket": {...}, "afterhours": {...}}`,
same shape read by both pieces below):

- **Schedule**: `TradingScheduler._start_extended_hours_jobs` registers up to two
  additional cron jobs — `premarket` (default `cron:08:00`) and `afterhours` (default
  `cron:17:00`) — additive to whatever the main `schedule` preset/composite already runs.
  Only registered per-session when `extended_hours_trading.enabled` **and** that
  session's own `enabled` are both true; each job passes its own explicit
  `cycle_type` ("premarket"/"afterhours") through to `LiveTradingEngine.run_cycle`.
- **Engine gate**: `LiveTradingEngine.run_cycle`'s market-hours check (`_respect_market_hours`)
  now branches on `cycle_type`: `"premarket"`/`"afterhours"` is validated against the
  configured `[start, end)` window (default 04:00-09:30 / 16:00-20:00 ET,
  `firm.live.scheduler.within_extended_hours_window`, Mon-Fri only) instead of
  `is_market_open()` — a cycle outside that window (including weekends/overnight) is
  still skipped exactly like a regular closed-market cycle. Every other `cycle_type`
  (`None`/`"open"`/`"intraday"`/`"close"`) is completely unaffected — still gated by
  `is_market_open()` as before this feature existed.
- **Order flag**: only when `run_cycle`'s gate confirms a cycle is genuinely inside its
  configured window (`CycleResult.extended_hours_cycle`) does `_execute_orders` set
  `OrderRequest.extended_hours=True` on that cycle's orders — never a blanket config
  toggle applied regardless of when a cycle actually runs, and never on the separate
  `ExecutionAgent._maybe_submit_protective_order` broker-resident stop side-channel
  above (always regular-hours).
- `extended_hours_trading` must be present in `provider_utils.py`'s allowlist tuple to
  reach both `TradingScheduler` and `LiveTradingEngine` via the systemd auto-start /
  `POST /api/live/start` path — added there in the same change as this feature.
  `config/live.yaml`/`config/live_alpaca.yaml` do **not** set it (feature off on both
  instances) — enabling it live is a separate decision.
- Both `risk.stop_loss_overlay` and `protective_orders` must be present in
  `provider_utils.py`'s allowlist tuple to actually reach the engine via the systemd
  auto-start / `POST /api/live/start` path (see "What `resolve_live_startup()` merges"
  below) — both were added there in the same change.

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

3. **Ensemble-HMM market regime detector (shipped, off by default)** —
   `regime/ensemble.py`'s `EnsembleRegimeModel` majority-votes across 5
   independently-seeded `GaussianRegimeModel` fits, swapped in via
   `MarketRegimeDetector(ensemble=True)` behind the same `fit`/`classify`
   interface. Built to test whether calming the single HMM's label noise
   would rescue `strategy_regime_weights` (item 2's sibling feature, which
   already failed its own A/B). **Result**: the ensemble does calm
   `regime_hmm`'s own attributed Sharpe (moves toward zero in every window:
   -1.455→-0.575, -1.516→-0.599, 2.046→2.016) but portfolio Sharpe got
   *worse* in 2/3 windows with `strategy_regime_weights` enabled under it —
   detector noise was not the bottleneck; the `optimal`/regime-weight
   interaction is. **Left disabled by default** on both `regime_overlay.ensemble`
   and `strategy_regime_weights.ensemble`; see `docs/regime_ensemble_scoping.md`
   for the full A/B and `scripts/calibrate_regime_ensemble.py` to reproduce.

### Per-strategy capital sleeves (`agents/orchestrator.py`, `capital_allocation_mode`)

Built 2026-09-09/10 to answer a real gap: `PerformanceAttribution`'s per-strategy P&L is
a heuristic (dominant-strategy-wins-the-whole-order + running-net-share-count over one
shared book), not exact, since every strategy blends into one portfolio. Full
design/history: `docs/capital_sleeves_plan.md`.

- **Config**: `capital_allocation_mode: "blended" | "sleeved"` (default `blended` =
  today's exact existing behavior, byte-for-byte). Start-time only — same treatment as
  `broker`, since switching it restructures the orchestrator rather than swapping one
  mutable attribute; not part of `PUT /api/live/config`.
- **Design**: each sleeved strategy runs its own independent
  bull→bear→debate→trader→risk→execution pass against its own `PortfolioState` (a fixed
  initial capital split that compounds independently from there — not re-normalized to a
  fraction of current NAV every cycle, or the sleeve's real standalone track record would
  be hidden). A separate `TraderAgent` instance per sleeve (it holds real cross-cycle
  state — conviction EMA, joint-optimizer NAV history — that one shared instance would
  let sleeves corrupt). A final pass nets every sleeve's approved target into one real
  order set against the *real* shared book — the only place actual broker orders are
  generated, so opposing sleeve views on the same symbol still net to one smaller order.
- **A real bug found via A/B backtest** (blended vs sleeved, same cached data, before any
  live exposure): splitting capital ~18 ways meant no single symbol's combined weight
  could clear the 5% rebalance band tuned for the *blended* book — the real (netted) book
  had **zero turnover for an entire quarter** even though every individual sleeve traded
  correctly. Fixed with a separate, smaller band for the final netted pass only
  (`real_rebalance_band_pct`, defaults to the shared band divided by the sleeve count);
  each sleeve's own internal execution keeps the original band unchanged.
- **Cutover procedure**: `POST /api/live/sleeves/seed` (once, right after starting a newly
  sleeved engine, before its first cycle) seeds every sleeve from real broker
  positions + existing `PerformanceAttribution` history — a best-effort approximation,
  not an exact split, since blended mode never tracked exact per-strategy positions.
  Refuses a second call once any sleeve has real state, to avoid overwriting genuine
  history with a stale re-seed.
- **Current live status**: `config/live_alpaca.yaml` sets `capital_allocation_mode:
  sleeved` (2026-09-10) — `config/live.yaml` (IBKR) has no such key, so it stays on
  `blended` as the static control, same A/B pattern as the sector-scanner work below.
  `GET /api/live/attribution` prefers exact per-sleeve metrics
  (`Orchestrator.get_sleeve_metrics()`) over the heuristic once a sleeve has ≥2 daily
  snapshots, falling back to the heuristic for any non-sleeved strategy.
- **A second real bug, found live (2026-09-19)**: the SAME per-sleeve risk check
  (`RiskAgent.run`, called once per sleeve against only that strategy's own signals) used
  the blended book's `max_position_pct`/`veto_threshold` unchanged. Fine for a strategy
  whose sleeve stays diversified, but `stat_arb` (at most 5 pairs / 10 legs,
  `strategy_params.stat_arb.max_pairs`) legitimately needs 30-90% of its OWN sleeve
  capital per leg on a typical 2-4-leg day — the shared 5%/50% envelope clipped that so
  hard it tripped the veto on **every single stat_arb sleeve decision since sleeved mode
  began on 9/10** (confirmed via journalctl: 9/9 logged decisions were vetoes, 0
  approved) — that sleeve had been silently frozen on its 9/10 cutover-seed positions for
  over a week, never once acting on a real signal, with no queryable trace (only
  free-text log lines). Fixed two ways:
  - `RiskAgent.sleeve_risk_overrides` (config key, empty by default): a per-strategy
    override of `max_position_pct`/`max_gross_exposure`/`max_net_exposure`/
    `veto_threshold`, applied only for the duration of that one `run()` call (save/
    restore, same pattern as `Orchestrator._apply_cycle_llm_mode`). `config/live_alpaca.
    yaml` sets `stat_arb: {max_position_pct: 0.50, max_gross_exposure: 2.0,
    veto_threshold: 0.95}` — this one **is** enabled live (not shipped disabled, unlike
    every other new knob this session), since it fixes a confirmed active incident.
  - `Blackboard.sleeve_decisions` (new field, sleeved mode only): every sleeve gets a
    per-cycle status entry (`"approved"|"rejected"|"no_signal"|"no_debate_results"|
    "<stage>_failed"`, plus `violations`/`actions` when relevant) — see
    `Orchestrator._step_sleeved`. Persisted via `CycleResult.sleeve_decisions` ->
    `LiveTradingEngine._persist_cycle_result` into the same `TradeHistoryStore` cycle
    records everything else already goes through, and queryable directly via
    `GET /api/live/sleeves/decisions?strategy=<name>&limit=N` (most recent first) instead
    of grepping logs to notice a sleeve has stopped trading.
- **Same pattern recurred, found live (2026-09-21)**: `event_driven` only signals on
  symbols with a recent earnings surprise (`strategies/event_driven.py`) — typically 3-4
  names out of the whole universe, so each legitimately needs 15-40%+ of the sleeve's own
  capital. Confirmed via journalctl: every logged `event_driven` sleeve decision back to
  at least 9/15 was a veto (clipping severity 76-80%) — frozen for 6+ days, never once
  acting on a real earnings signal. Same fix: `sleeve_risk_overrides.event_driven` added
  to `config/live_alpaca.yaml` with the identical envelope as `stat_arb` above. Also fixed
  the same day: `ExecutionAgent._maybe_submit_protective_order`'s `stop_price` was an
  unrounded float (`price * (1 ± stop_loss_pct)`) — Alpaca rejects any sub-penny price for
  stocks above $1 ("sub-penny increment does not fulfill minimum pricing criteria"), so
  every protective stop submitted since that feature was enabled the previous night failed
  100% of the time (3/3, zero successes ever). Now rounds to whole cents ($0.01 tick) above
  $1, sub-penny ($0.0001) below.
- **Deliberately not done yet**: a per-sleeve NAV/position/PnL frontend view (still one
  shared-portfolio UI); cross-sleeve LLM-enhancement budget coordination (only matters if
  `bull_researcher`/`bear_researcher`/`debate` are ever switched to `llm_enhanced` — the
  orchestrator refuses to construct in sleeved mode if they are, rather than silently
  multiplying LLM call volume ~N-fold).

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
| LLM A/B (if applicable) | Arm completes runbook | See `docs/llm_ab_test_runbook.md`. Currently on **Arm B** (`fundamental_analyst`/`sentiment_analyst`: `llm_enhanced`, started 2026-09-08 early per user request) — don't assume quant-only is still the live baseline |

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
| LLM A/B (if applicable) | Arm completes runbook | See `docs/llm_ab_test_runbook.md`. Currently on Arm B (`llm_enhanced`) — see the paper-promotion gate row above for detail |

**Initial allocation is tranched, not all-or-nothing:** fund at **10-20%** of the
intended target size first; only scale toward full size after a second
observation period (same criteria, shorter — e.g. 30 trading days) confirms
performance held with real fills. This shortens time-to-first-capital versus a
flat 6-12mo wait while keeping the actual dollars at risk small until the edge is
confirmed with real money on the line, not just simulated.

---

## Proactive trading system (2026-09-20)

A single-session initiative to make the system genuinely proactive rather than
periodically rebalancing: react to news, read chart patterns, actively manage
which symbols get real capital, decisively exit positions, and capture every
day's decisions for offline review. Built after a research pass (3 codebase
mapping agents + 1 professional-practices research agent) found most of the
underlying infrastructure already existed but was inert, buggy, or missing one
connecting piece — this closed those specific gaps rather than rebuilding
anything that already worked. Full plan/rationale + citations:
`/root/.claude/plans/lively-stirring-lagoon.md` (session-local, not in the repo).

| Gap | Fix | Where |
|-----|-----|-------|
| A target-0 position could get asymptotically stranded (band tolerates dust forever; `rebalance_fraction` decay never fully reaches zero) | `ExecutionAgent`'s new `close_dust_fraction`: a materially-sized position (above a floor much smaller than the band) closes in full, bypassing the band/fraction — deliberately narrower than "always force-close," which a prior 3-window A/B found regressed Sharpe 3.45→0.80 | `agents/execution.py` |
| No operator control to fully exit a position/strategy on demand | `LiveTradingEngine.flatten_symbol`/`flatten_strategy` + `POST /api/live/positions/{symbol}/flatten` / `/sleeves/{strategy}/flatten` | `live/engine.py`, `api/routers/live.py`, `frontend` LiveDashboard |
| `hourly_market_hours`'s ~7 cycles/day silently dropped 6 of every 7 decisions from reflection (bare-date, first-writer-wins key) | `TradingMemoryLog.store_decision` keys by `date#cycle_id`; new `reflect_day()` aggregates a whole day into ONE LLM reflection call (cost stays flat) | `agents/memory.py` |
| Reflection was purely read-only, no way to act on a conclusion | `DailyReflectionRecommendation` — a bounded, pre-enumerated action (`reduce_position_limit`/`flag_strategy_for_review`/`no_action`), never auto-applied (no source found supports an LLM reflection loop auto-adjusting its own config — SR 11-7, TradeTrap arXiv 2512.02261). `GET /api/memory/recommendations` + human `POST /api/live/recommendations/{date}/apply`, which reuses `RiskAgent.sleeve_risk_overrides` | `llm/schemas.py`, `api/routers/{decisions,live}.py`, `frontend` Decisions page |
| `sentiment`/`fundamental` LLM agents queried a `news` RAG collection that had zero documents — silent quant-only fallback every time | Daily `news_ingestion` scheduler job populates it via the existing `NewsIngestor`; retrieved news text anonymized (company/ticker stripped) per Glasserman & Lin (arXiv 2309.17322) before reaching any prompt | `live/news_ingestion_job.py`, `agents/llm/news_anonymizer.py` — Alpaca only, verified end-to-end against production credentials |
| `pattern_recognition`'s validated CNN/GAF layer was fully built but never called live; named rule-based patterns alone have no edge after data-snooping correction (Marshall & Cahan) while learned CNN features do (Jiang/Kelly/Xiu, JF 2023) | `firm.patterns.ml.inference` (fail-soft ONNX wrapper) scores each rule-based candidate's quality. Re-trained the model on real cached history after the on-disk artifact (probably trained on synthetic smoke-test data) validated *worse* than rule-based; the retrained one improved walk-forward Sharpe 0.415→0.645. `cnn_scoring_enabled` gate defaults **off** — model-file presence alone must never imply live usage | `patterns/ml/inference.py`, `strategies/pattern_recognition.py` — enabled on both instances after validation |
| New symbols entered the live universe with real capital immediately; removal never closed the position | `dynamic_universe_state`'s schema gains `candidate`→`active` (incubation_days, default 5, mirrors `min_dwell_days`'s removal-side convention); `RiskAgent.incubating_symbols` zeroes a candidate's target weight until promoted; `sp500_universe_sync` now calls `flatten_symbol` on dwell-removal | `live/{dynamic_universe_state,danelfin_universe_sync,sp500_universe_sync}.py`, `agents/risk.py` — Alpaca only |

**Deliberately not done**: a fully autonomous reflection→config loop (every
source found routes this through human review, not automation); after-hours
limit-order pricing for Alpaca (extended_hours_trading stayed IBKR-only, see
its own section above); performance-gated (vs. time-gated) incubation
promotion (a second overfit-prone gate wasn't worth it without a track record
on the simpler version first).

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Engine stuck | `cycle_running_seconds` in `/api/live/status`; restart service or stop/start live |
| `multi_factor` missing | `FMP_API_KEY` in `.env`; check logs for provider filter warning |
| IBKR connect fails on boot | Gateway up? `nc -zv 127.0.0.1 4002`; auto-start logs in journalctl |
| `this event loop is already running` | IBKR connect called from asyncio lifespan — must use worker thread |
| Orders queued forever | `approval_mode` must be `full_auto` or `semi_auto` |
| Fewer than 11 strategies active | `regime_hmm` or fundamental strategies filtered — check provider keys (11 is the current full baseline; see "Thirteen strategies" above) |

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
- [llm_lookahead_audit.md](llm_lookahead_audit.md) — RAG point-in-time audit + a dense-channel crash-on-None-date fix
- [regime_ensemble_scoping.md](regime_ensemble_scoping.md) — ensemble-HMM regime detector, A/B'd (shipped disabled: calms `regime_hmm`'s own noise but doesn't rescue `strategy_regime_weights`)
- [pattern_recognition_plan.md](pattern_recognition_plan.md) — chart-pattern recognition (Strategy #13): 5 build phases + a follow-up pass (scheduled scan job, ONNX export, isolated CNN/PPO env)
- [capital_sleeves_plan.md](capital_sleeves_plan.md) — per-strategy capital sleeves: design, an A/B-found-and-fixed rebalance-band bug, live cutover on Alpaca
