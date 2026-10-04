"""Mechanical metadata for config/universe_etf.yaml (P2-01). Reads ONLY: date column (full calendar, rows <= 2026-09-30),
and volume*close inside each instrument's first 3 years after inception. No return/Sharpe/correlation is computed."""
import json, re, sys
from pathlib import Path
import pandas as pd

DATA = Path(sys.argv[1])  # .../data/research/eodhd
CANDS = sys.argv[2].split(",")
man = json.load(open(DATA / "extras_manifest_full.json"))
out = {}
for s in CANDS:
    d = pd.read_parquet(DATA / "etfs_full" / f"{s}.parquet", columns=["date", "close", "volume"])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= "2026-09-30"].reset_index(drop=True)
    inc = d["date"].iloc[0]
    assert str(inc.date()) == man[f"etf:{s}"]["first"], s
    w = d[d["date"] < inc + pd.DateOffset(years=3)]
    adv = float((w["close"] * w["volume"]).median())
    out[s] = dict(inception=str(inc.date()), first_trade=str(d["date"].iloc[256].date()),
                  full_vol=str(d["date"].iloc[2520].date()) if len(d) > 2520 else None,
                  early_adv_usd=adv, rows=len(d))
print(json.dumps(out, indent=1))
