import json
import sys

sys.path.insert(0, sys.argv[1] + "/scripts")
from eodhd_clean import equity_calendar

cal = equity_calendar()

def midpoint(start, end):
    sub = cal[(cal >= start) & (cal <= end)]
    mid_idx = len(sub) // 2
    return str(sub[mid_idx].date()), len(sub)

bond_start, bond_end = "2007-01-11", "2026-09-29"
commod_start, commod_end = "2010-01-08", "2026-09-29"

bm, bn = midpoint(bond_start, bond_end)
cm, cn = midpoint(commod_start, commod_end)
print(json.dumps({
    "bond_midpoint": bm, "bond_n_sessions": bn,
    "commodity_midpoint": cm, "commodity_n_sessions": cn,
    "bond_months_approx": bn/21, "commodity_months_approx": cn/21,
}, indent=2))
