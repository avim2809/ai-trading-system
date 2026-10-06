import json
import sys

sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
from pathlib import Path

import pandas as pd
from eodhd_clean import clean_bars, cleaning_fingerprint, equity_calendar

EODHD = Path(sys.argv[1]) / "data" / "research" / "eodhd" / "etfs_full"
FRED = Path(sys.argv[1]) / "data" / "research" / "fred"
BOND = ["SHY","IEF","TLT"]
COMMOD = ["DBA","DBB","DBE","USO","GLD","SLV"]
CASH = ["BIL"]
CORE = ["SPY","IEF"]
NAV = ["VFITX"]
EXCLUDED_CONSIDERED = ["IEI","TLH","VGIT","VGLT","GOVT","PALL","PPLT","DBC","GSG","DJP","DBP","CORN","SOYB","WEAT","CPER","UNG"]

cal = equity_calendar("etfs_full")
def snap(t, asset="equity"):
    f = EODHD / f"{t}.parquet"
    if not f.exists():
        return {"present": False}
    raw = pd.read_parquet(f)
    d, rep = clean_bars(raw, asset, cal)
    if len(d) == 0:
        return {"present": True, "n_clean_bars": 0, "clean_report": rep}
    last_seg = d[d["segment"] == d["segment"].max()]
    out = {
        "present": True,
        "first_clean_date": str(d["date"].min().date()),
        "last_clean_date": str(d["date"].max().date()),
        "n_clean_bars": len(d),
        "n_segments": int(d["segment"].nunique()),
        "clean_report": rep,
    }
    if asset != "nav":
        dvol = (last_seg["adjusted_close"] * last_seg["volume"]).astype(float)
        out["adv20_latest_usd"] = round(float(dvol.tail(20).mean()), 2)
    return out

out = {
    "generated_at": "2026-09-30",
    "amendment": "protocol_amendment_1 adopted (etfs_full/, cleaning v2, BIL+DTB3 cash, VFITX BM2 pre-IEF leg)",
    "purpose": "Phase-1 data-availability snapshot for S3 (bond duration momentum + commodity dual momentum) "
               "pre-registration design, after Amendment 1 and the coordinator's review. No candidate/"
               "benchmark/placebo return computed.",
    "cleaning_fingerprint": cleaning_fingerprint(),
    "bond_buckets": {t: snap(t) for t in BOND},
    "commodity_basket": {t: snap(t) for t in COMMOD},
    "cash_proxies": {t: snap(t) for t in CASH},
    "core": {t: snap(t) for t in CORE},
    "bm2_pre_ief_bond_leg_nav": {t: snap(t, asset="nav") for t in NAV},
    "considered_and_excluded": {t: snap(t) for t in EXCLUDED_CONSIDERED},
}
fred = pd.read_parquet(FRED / "DTB3.parquet")
out["fred_dtb3"] = {
    "first_date": str(pd.to_datetime(fred["date"]).min().date()),
    "last_date": str(pd.to_datetime(fred["date"]).max().date()),
    "n_rows": len(fred),
    "used_for": "cash accrual before BIL's 2007-05-30 first bar only",
}
print(json.dumps(out, indent=2, default=str))
