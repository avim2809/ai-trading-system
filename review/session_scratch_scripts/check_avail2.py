import json
import sys

sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
from pathlib import Path

import pandas as pd
from eodhd_clean import clean_bars, equity_calendar

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs"
TICKERS = ["CORN","SOYB","WEAT","DBA","DBB","DBE","DBP","GSG","DJP","CPER","USO","UNG","GLD","SLV"]
cal = equity_calendar()
rows = {}
for t in TICKERS:
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        rows[t] = {"present": False}; continue
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, "equity", cal)
    if len(d)==0:
        rows[t] = {"present": True, "n_clean": 0}; continue
    d2 = d[d["segment"]==d["segment"].max()]
    dvol = (d2["adjusted_close"]*d2["volume"]).astype(float)
    rows[t] = {
        "present": True, "first_date": str(d["date"].min().date()), "last_date": str(d["date"].max().date()),
        "n_clean_bars": len(d), "n_segments": int(d["segment"].nunique()),
        "adv20_latest_usd": float(dvol.tail(20).mean()),
    }
print(json.dumps(rows, indent=2, default=str))
