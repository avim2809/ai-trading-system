"""Read-only peek at the Alpaca instance's last persisted portfolio snapshot."""
import json
import sqlite3

DB = "file:/local/store/git/ai-trading-system/data_alpaca/live_state.db?mode=ro"
conn = sqlite3.connect(DB, uri=True)
row = conn.execute("SELECT value, updated_at FROM state_blobs WHERE key='portfolio_history'").fetchone()
hist = json.loads(row[0])
last = hist[-1]
print("updated_at", row[1], "snapshots", len(hist))
print("asof", last["asof"], "nav", round(last["nav"], 2), "cash", round(last["cash"], 2))
w = last.get("weights") or {}
longs = {k: v for k, v in w.items() if v > 0}
shorts = {k: v for k, v in w.items() if v < 0}
print("n_positions", len(w), "longs", len(longs), "shorts", len(shorts))
print("gross", round(sum(abs(v) for v in w.values()), 4), "net", round(sum(w.values()), 4))
print("largest", sorted(w.items(), key=lambda kv: -abs(kv[1]))[:8])
print("has SPY/IEF", {k: w.get(k) for k in ("SPY", "IEF")})
row = conn.execute("SELECT value FROM state_blobs WHERE key='allocation_state'").fetchone()
print("allocation_state present:", row is not None)
conn.close()
