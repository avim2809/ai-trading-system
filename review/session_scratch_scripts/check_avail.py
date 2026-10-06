import json
import sys

sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
from pathlib import Path

import pandas as pd
from eodhd_clean import clean_bars, cleaning_fingerprint, equity_calendar

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs"

BOND = ["SHY","IEI","IEF","TLH","TLT","VGIT","VGLT","GOVT"]
COMMOD = ["DBC","GSG","DJP","DBE","DBB","DBP","DBA","GLD","SLV","USO","UNG","CPER"]
CASH = ["BIL","SHV"]
CORE = ["SPY","IEF"]

cal = equity_calendar()
rows = {}
for t in sorted(set(BOND+COMMOD+CASH+CORE)):
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        rows[t] = {"present": False}
        continue
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, "equity", cal)
    d = d[d["segment"] == d["segment"].max()] if len(d) else d  # last segment only, for a quick ADV snapshot
    if len(d) == 0:
        rows[t] = {"present": True, "n_clean": 0}
        continue
    dvol = (d["adjusted_close"] * d["volume"]).astype(float)
    adv20 = dvol.tail(20).mean()
    rows[t] = {
        "present": True,
        "first_date": str(d["date"].min().date()),
        "last_date": str(d["date"].max().date()),
        "n_clean_bars": len(d),
        "n_segments": int(raw.shape[0] and d["segment"].nunique()),
        "adv20_latest_usd": float(adv20),
        "clean_report": rep,
    }

out = {
    "cleaning_fingerprint": cleaning_fingerprint(),
    "bond_candidates": BOND,
    "commodity_candidates": COMMOD,
    "cash_candidates": CASH,
    "core_candidates": CORE,
    "per_ticker": rows,
}
print(json.dumps(out, indent=2, default=str))
