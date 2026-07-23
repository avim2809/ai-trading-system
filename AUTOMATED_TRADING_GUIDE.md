# Automated Paper Trading Guide

Your AI trading system can trade automatically on paper with IB Gateway. There are two main approaches:

## Option 1: Simple Monitoring Script (Easiest)

Run the automated script that monitors account performance:

```bash
cd /local/store/git/ai-trading-system

# Run until $100 profit
python3 scripts/run_until_profitable.py --profit-target 100

# Or customize:
python3 scripts/run_until_profitable.py \
  --profit-target 500 \
  --max-cycles 1000 \
  --interval 60
```

**Features:**
- Monitors account equity in real-time
- Stops when profit target reached
- Tracks positions and P&L
- Clean console logging

---

## Option 2: Full API Server with Strategy Execution (Recommended for Production)

For **full strategy execution** with all configured strategies:

### Step 1: Start the API Server

```bash
cd /local/store/git/ai-trading-system
firm-api
```

Or via systemd (production):

```bash
sudo systemctl start ai-trading
```

This starts FastAPI on `http://localhost:8000` (API under `/api/...`).

With `FIRM_AUTO_START_LIVE=1` in `.env`, live trading starts automatically from `config/live.yaml` on boot — no manual curl needed.

### Step 2: Trigger Live Trading (if not auto-started)

```bash
# Minimal start — merges universe, strategies, risk, and strategy_params from config/live.yaml
curl -X POST http://localhost:8000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{"broker": "ibkr_paper"}'

# Or override specific fields:
curl -X POST http://localhost:8000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{
    "broker": "ibkr_paper",
    "schedule": "market_open",
    "approval_mode": "full_auto",
    "symbols": ["AAPL", "MSFT", "GOOG"]
  }'
```

`approval_mode` must be exactly `"full_auto"` or `"semi_auto"` — anything else is treated as semi-auto with manual approval for all orders.

Omitted fields are filled from `config/live.yaml` via `resolve_live_startup()` in `src/firm/live/provider_utils.py`.

### Step 3: Monitor Trading

```bash
# Engine status (includes cycle_running_seconds — watch for stuck cycles)
curl http://localhost:8000/api/live/status

# Account info
curl http://localhost:8000/api/live/account

# Open positions
curl http://localhost:8000/api/live/positions

# Recent cycles
curl http://localhost:8000/api/live/cycles

# Orders
curl http://localhost:8000/api/live/orders

# Alerts (drawdown warnings, etc.)
curl http://localhost:8000/api/live/alerts
```

### Step 4: Stop Trading

```bash
curl -X POST http://localhost:8000/api/live/stop
```

---

## Comparison

| Feature | Script | API Server |
|---------|--------|-----------|
| Strategy Execution | Monitoring only | Full ML/analyst pipeline |
| Account Monitoring | Real-time | Real-time via API |
| Manual Control | Not during run | HTTP endpoints |
| Approval Queue | No | Yes (semi-auto mode) |
| Scheduled Execution | Manual interval | Market hours scheduling |
| Dashboard Ready | Console logs | Web UI at `/live` |
| Production Ready | Simple | Enterprise-grade |

---

## Configuration

Both use the same `.env` file:

```env
IBKR_HOST=127.0.0.1
IBKR_PAPER_PORT=4002
IBKR_CLIENT_ID=1
FMP_API_KEY=your_key          # required for multi_factor / event_driven on IBKR
FIRM_AUTO_START_LIVE=1        # auto-start live from live.yaml on firm-api boot
```

[`config/live.yaml`](config/live.yaml) is the canonical live config:

```yaml
broker: "ibkr_paper"
schedule: "market_open"
approval_mode: "full_auto"
strategies:
  enabled: [momentum, trend, mean_reversion, stat_arb, ...]  # all 12
  auto_approve: [momentum, trend, ...]
strategy_params:
  stat_arb:
    require_cointegration: true
    predefined_pairs:
      - ["SPY", "QQQ"]
      - ["AAPL", "MSFT"]
universe:
  symbols: [AAPL, MSFT, NVDA, ...]   # 30 symbols on this host
risk:
  kill_switch_drawdown: 0.08
  max_daily_trades: 40
  max_position_pct: 0.05
  regime_overlay:
    enabled: true
initial_capital: 1_000_000
```

**How YAML is applied:**

- `POST /api/live/start` merges missing fields from `config/live.yaml` (universe, strategies, `strategy_params`, flattened `risk:` block).
- `FIRM_AUTO_START_LIVE=1` applies the full YAML on `firm-api` boot.
- `scripts/run_live_trading.py --config config/live.yaml` is an alternate CLI path for debugging — not used when running via `ai-trading.service`.

See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) for full details.

---

## Running in Background (Production Setup)

### Using systemd (recommended)

Copy the unit from the repo:

```bash
sudo cp deploy/ai-trading.service /etc/systemd/system/ai-trading.service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading
sudo systemctl start ai-trading
sudo systemctl status ai-trading
```

The unit file runs `firm-api`, loads `.env`, sets `FIRM_AUTO_START_LIVE=1`, and starts after `ibgateway.service`.

```bash
# View logs
sudo journalctl -u ai-trading -f

# Restart after code or config changes
sudo systemctl restart ai-trading
```

### Using tmux (temporary)

```bash
tmux new-session -d -s trading -c /local/store/git/ai-trading-system "firm-api"
tmux attach-session -t trading
tmux kill-session -t trading
```

---

## Troubleshooting

### IB Gateway Connection Issues

```bash
nc -zv 127.0.0.1 4002
ps aux | grep ibgateway
sudo systemctl status ibgateway
```

### Stuck trading cycle

Check `cycle_running_seconds` in `/api/live/status`. If non-null for hours, restart:

```bash
sudo systemctl restart ai-trading
# or: curl -X POST http://localhost:8000/api/live/stop && curl -X POST .../start -d '{}'
```

### Account Access Issues

Verify credentials and paper account in IB Gateway. Data provider uses `client_id=2`; broker uses `IBKR_CLIENT_ID`.

### Strategy Issues

- Check `config/live.yaml` for enabled strategies.
- `multi_factor` and `event_driven` require `FMP_API_KEY` on IBKR live.
- Check logs: `journalctl -u ai-trading -f`

---

## Next Steps

1. **Verify connectivity:** `nc -zv 127.0.0.1 4002` and `/api/live/status`
2. **Monitor cycles:** Live Dashboard at `http://localhost:8000/live`
3. **Tune risk:** Edit `config/live.yaml` risk block, restart service
4. **Go live:** Switch `broker: "ibkr_live"` when ready (requires live Gateway connection)
