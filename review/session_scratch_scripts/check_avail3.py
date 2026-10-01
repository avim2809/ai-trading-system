import sys, json
sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
import pandas as pd
from pathlib import Path
from eodhd_clean import clean_bars, equity_calendar

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs"
TICKERS = ["PALL","PPLT"]
cal = equity_calendar()
rows = {}
for t in TICKERS:
    f = EODHD / f"{t}.parquet"
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, "equity", cal)
    d2 = d[d["segment"]==d["segment"].max()]
    dvol = (d2["adjusted_close"]*d2["volume"]).astype(float)
    rows[t] = {
        "first_date": str(d["date"].min().date()), "last_date": str(d["date"].max().date()),
        "n_clean_bars": int(len(d)), "n_segments": int(d["segment"].nunique()),
        "adv20_latest_usd": float(dvol.tail(20).mean()), "clean_report": rep,
    }
print(json.dumps(rows, indent=2, default=str))
