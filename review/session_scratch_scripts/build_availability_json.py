import sys, json
sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
import pandas as pd
from pathlib import Path
from eodhd_clean import clean_bars, equity_calendar, cleaning_fingerprint

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs"
BOND = ["SHY","IEI","IEF","TLH","TLT"]
COMMOD = ["DBA","DBB","DBE","USO","GLD","SLV","PALL","PPLT"]
CASH = ["BIL","SHV"]
CORE = ["SPY","IEF"]
EXCLUDED_CONSIDERED = ["VGIT","VGLT","GOVT","DBC","GSG","DJP","DBP","CORN","SOYB","WEAT","CPER","UNG"]

cal = equity_calendar()
def snap(t):
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        return {"present": False}
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, "equity", cal)
    if len(d) == 0:
        return {"present": True, "n_clean_bars": 0}
    last_seg = d[d["segment"] == d["segment"].max()]
    dvol = (last_seg["adjusted_close"] * last_seg["volume"]).astype(float)
    return {
        "present": True,
        "first_clean_date": str(d["date"].min().date()),
        "last_clean_date": str(d["date"].max().date()),
        "n_clean_bars": int(len(d)),
        "n_segments": int(d["segment"].nunique()),
        "adv20_latest_usd": round(float(dvol.tail(20).mean()), 2),
        "clean_report": rep,
    }

out = {
    "generated_at": "2026-09-30",
    "purpose": "Phase-1 data-availability snapshot for S3 (bond duration momentum + commodity dual momentum) "
               "pre-registration design. No candidate/benchmark/placebo return computed.",
    "cleaning_fingerprint": cleaning_fingerprint(),
    "bond_buckets": {t: snap(t) for t in BOND},
    "commodity_basket": {t: snap(t) for t in COMMOD},
    "cash_proxies": {t: snap(t) for t in CASH},
    "core": {t: snap(t) for t in CORE},
    "considered_and_excluded": {t: snap(t) for t in EXCLUDED_CONSIDERED},
}
print(json.dumps(out, indent=2, default=str))
