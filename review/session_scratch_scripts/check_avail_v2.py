import sys, json
sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
import pandas as pd
from pathlib import Path
from eodhd_clean import clean_bars, equity_calendar, cleaning_fingerprint

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs_full"
BOND = ["SHY","IEF","TLT"]
COMMOD = ["DBA","DBB","DBE","USO","GLD","SLV"]
CASH = ["BIL"]
CORE = ["SPY","IEF"]
NAV = ["VFITX","VUSTX","VFISX"]

cal = equity_calendar("etfs_full")

def snap(t, asset="equity"):
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        return {"present": False}
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, asset, cal)
    if len(d) == 0:
        return {"present": True, "n_clean_bars": 0, "rep": rep}
    last_seg = d[d["segment"] == d["segment"].max()]
    out = {
        "present": True,
        "first_clean_date": str(d["date"].min().date()),
        "last_clean_date": str(d["date"].max().date()),
        "n_clean_bars": int(len(d)),
        "n_segments": int(d["segment"].nunique()),
        "clean_report": rep,
    }
    if asset != "nav":
        dvol = (last_seg["adjusted_close"] * last_seg["volume"]).astype(float)
        out["adv20_latest_usd"] = round(float(dvol.tail(20).mean()), 2)
    return out

rows = {}
for t in BOND + COMMOD + CASH + CORE:
    rows[t] = snap(t)
for t in NAV:
    rows[t] = snap(t, asset="nav")

print(json.dumps({"cleaning_fingerprint": cleaning_fingerprint(), "per_ticker": rows}, indent=2, default=str))
