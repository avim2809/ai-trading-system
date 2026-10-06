import json
import sys

sys.path.insert(0, sys.argv[1] + "/scripts")
from eodhd_clean import equity_calendar

cal = equity_calendar("etfs_full")

def midpoint(start, end):
    sub = cal[(cal >= start) & (cal <= end)]
    mid_idx = len(sub) // 2
    return str(sub[mid_idx].date()), len(sub)

bond_start, bond_end = "2002-07-26", "2026-09-29"
commod_start, commod_end = "2007-01-05", "2026-09-29"
combined_start, combined_end = "2007-01-05", "2026-09-29"

bm, bn = midpoint(bond_start, bond_end)
cm, cn = midpoint(commod_start, commod_end)
km, kn = midpoint(combined_start, combined_end)
print(json.dumps({
    "bond_midpoint": bm, "bond_n_sessions": bn, "bond_years_approx": round(bn/252,2),
    "commodity_midpoint": cm, "commodity_n_sessions": cn, "commodity_years_approx": round(cn/252,2),
    "combined_midpoint": km, "combined_n_sessions": kn,
}, indent=2))
