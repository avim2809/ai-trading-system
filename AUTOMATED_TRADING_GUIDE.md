# Automated Paper Trading Guide

Your AI trading system can now trade automatically on paper with IB Gateway. There are two main approaches:

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
- ✅ Monitors account equity in real-time
- ✅ Stops when profit target reached
- ✅ Tracks positions and P&L
- ✅ Clean console logging

**Example Output:**
```
✅ Connected to IB Gateway
Initial account equity: $1,000,172.16
Available buying power: $6,667,814.40

--- Cycle 1/100 ---
Equity: $1,000,172.16 | Profit/Loss: $0.00
Buying Power: $6,667,814.40
📊 Open positions: 0
```

---

## Option 2: Full API Server with Strategy Execution (Recommended for Production)

For **full strategy execution** with all your configured strategies:

### Step 1: Start the API Server

```bash
cd /local/store/git/ai-trading-system
firm-api
```

This starts a FastAPI server on `http://localhost:8000`

### Step 2: Trigger Live Trading

In another terminal:

```bash
# Start live trading with auto-approval
curl -X POST http://localhost:8000/live/start \
  -H "Content-Type: application/json" \
  -d '{
    "broker": "ibkr_paper",
    "schedule": "market_open",
    "approval_mode": "full_auto",
    "auto_approve_strategies": ["momentum", "trend", "mean_reversion"],
    "symbols": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
  }'
```

`approval_mode` must be exactly `"full_auto"` or `"semi_auto"` — anything else (including plain `"auto"`) is silently treated as "queue everything for manual approval" by the engine. There's no `strategies` selector in this request: omitting it (as above) runs **every** registered strategy in parallel; `auto_approve_strategies` only matters in `"semi_auto"` mode, where it picks which strategies' orders skip the approval queue.
```

### Step 3: Monitor Trading

```bash
# Get account info
curl http://localhost:8000/live/account

# Get open positions
curl http://localhost:8000/live/positions

# Get recent cycles
curl http://localhost:8000/live/cycles

# Get orders
curl http://localhost:8000/live/orders

# Get alerts (drawdown warnings, etc)
curl http://localhost:8000/live/alerts
```

### Step 4: Stop Trading

```bash
curl -X POST http://localhost:8000/live/stop
```

---

## Comparison

| Feature | Script | API Server |
|---------|--------|-----------|
| Strategy Execution | ❌ Monitoring only | ✅ Full ML/analyst pipeline |
| Account Monitoring | ✅ Real-time | ✅ Real-time via API |
| Manual Control | ❌ Not during run | ✅ HTTP endpoints |
| Approval Queue | ❌ No | ✅ Yes (semi-auto mode) |
| Scheduled Execution | ❌ Manual interval | ✅ Market hours scheduling |
| Dashboard Ready | ❌ Console logs | ✅ JSON responses |
| Production Ready | ⚠️ Simple | ✅ Enterprise-grade |

---

## Configuration

Both use the same `.env` file:

```env
IBKR_HOST=127.0.0.1
IBKR_PAPER_PORT=4002
IBKR_CLIENT_ID=1
```

And `config/live.yaml` documents the intended strategies/risk/universe for a run:

```yaml
broker: "ibkr_paper"
schedule: "market_open"
approval_mode: "full_auto"
strategies:
  enabled: [momentum, trend, mean_reversion]
  auto_approve: [trend]
  require_approval: [momentum, mean_reversion]
risk:
  kill_switch_drawdown: 0.10
  max_daily_trades: 50
  max_daily_turnover: 0.5
universe:
  symbols: [AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA, JPM, V, JNJ]
```

**Note:** `/live/start` does not read this file directly — it only accepts the fields shown in the curl example above, so the `risk`/`universe` sections here take effect only via `scripts/run_live_trading.py --config config/live.yaml`, which does load them (flattening `risk:` into the engine config and passing `universe.symbols`/`strategies.enabled` through).

---

## Running in Background (Production Setup)

### Using tmux (Temporary):

```bash
# Start in background
tmux new-session -d -s trading -c /local/store/git/ai-trading-system "firm-api"

# Check status
tmux list-sessions

# Attach to see logs
tmux attach-session -t trading

# Kill when done
tmux kill-session -t trading
```

### Using systemd (Permanent):

Create `/etc/systemd/system/ai-trading.service`:

```ini
[Unit]
Description=AI Trading System Paper Trading
After=network.target

[Service]
Type=simple
User=milner
WorkingDirectory=/local/store/git/ai-trading-system
ExecStart=/usr/bin/python3 -m firm.api.app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-trading
sudo systemctl start ai-trading
sudo systemctl status ai-trading

# View logs
sudo journalctl -u ai-trading -f
```

---

## Troubleshooting

### IB Gateway Connection Issues

```bash
# Test connectivity
nc -zv 127.0.0.1 4002
# Should show: Connection succeeded

# Check IB Gateway is running
ps aux | grep ibgateway
```

### Account Access Issues

```bash
# Verify credentials and paper account is active in IB Gateway
python3 << 'EOF'
from ib_async import IB
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=2)
print(f"Connected: {ib.isConnected()}")
acc_summary = ib.reqAccountSummary()
import time; time.sleep(1)
for a in ib.accountSummary():
    if 'Liquidation' in a.tag:
        print(f"{a.tag}: {a.value}")
ib.disconnect()
EOF
```

### Strategy Issues

Check `config/live.yaml` for enabled strategies. Ensure data providers are configured in `.env`:

```env
POLYGON_API_KEY=your_key
FMP_API_KEY=your_key
TIINGO_API_KEY=your_key
```

---

## Next Steps

1. **Start Simple:** Use Option 1 script to verify connectivity
2. **Test Strategies:** Start API server and monitor cycles via endpoints
3. **Fine-tune:** Adjust approval modes and risk settings in `config/live.yaml`
4. **Go Live:** When profitable on paper, switch `broker: "ibkr_live"` (requires live account connection)

Good luck! 🚀
