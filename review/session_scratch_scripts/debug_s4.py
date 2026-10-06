import logging
import sys

sys.path.insert(0, "/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/wt_S4/scripts")
sys.path.insert(0, "/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/wt_S4/src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from pathlib import Path

import numpy as np
import pandas as pd
import run_eodhd_s4_evaluation as r

panel = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/S4/panel.parquet")
built = r.build_all(panel, limit_placebo=1)

dates = built["dates"]
cand = "S4_p1_6mo_N500"
slots = built["slots_of"][cand]
target_date = pd.Timestamp("2020-03-16")
pos = int(np.searchsorted(dates.to_numpy(), np.datetime64(target_date)))
print("pos", pos, "date at pos", dates[pos])

invested = 0.0
contribs = []
ticker_of_col = {v: k for k, v in built["col_of"].items()}
for s in slots:
    if s.entry_pos <= pos <= s.exit_pos:
        invested += s.weight
        # today's return contribution for this slot
        if pos == s.entry_pos:
            rr = built["ENTRY_RET"][pos, s.col]
        else:
            rr = built["RET"][pos, s.col]
        contribs.append((ticker_of_col[s.col], s.weight, float(rr) if np.isfinite(rr) else np.nan, s.entry_pos, s.exit_pos))

print("total invested weight on", target_date, ":", invested)
print("n active slots:", len(contribs))
contribs.sort(key=lambda x: (x[1]*(x[2] if np.isfinite(x[2]) else 0)))
print("worst 15 contributors:")
for c in contribs[:15]:
    print(c)
print("best 5:")
for c in contribs[-5:]:
    print(c)

# also check net return that day matches expectation
net = built["base"][cand]
print("reported net return that day:", net[pos])
