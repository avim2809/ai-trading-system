import sys, json
sys.path.insert(0, sys.argv[1] + "/scripts")
import pandas as pd
from pathlib import Path
from eodhd_clean import clean_bars, equity_calendar

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs_full"
cal = equity_calendar("etfs_full")
for t in ["IEI","TLH","VGIT","VGLT","GOVT","PALL","PPLT"]:
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        print(t, "MISSING"); continue
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, "equity", cal)
    print(t, str(d["date"].min().date()), str(d["date"].max().date()), len(d))
