import sys
import time

sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import run_insider_cluster_evaluation as ev

S = sys.argv[1]
benches = {s: ev.load_series(ev.EODHD / "etfs" / f"{s}.parquet") for s in ("IWC", "IWM", "IJH")}
rows = pd.read_parquet(f"{S}/runs/insider/events_evaluated.parquet")
old = pd.read_parquet(f"{S}/runs/insider/calendar_time.parquet")
cache = {t: ev.load_series(ev.price_path(t)) for t in rows["ticker"].unique()}
start = pd.Timestamp(ev.prereg.WINDOWS["primary_post_sample"]["start"])
cd = benches["IWM"]["date"]; cal = pd.DatetimeIndex(cd[cd >= start])
prep = {}; t0 = time.time(); worst = 0.0
for key in old.columns:
    h, st, b = key.split("|")
    df = rows[rows["hold_name"] == h]
    if st == "strict": df = df[df["name_ok"] == True]
    new = ev.calendar_time(df, cache, benches, b, cal, prep)
    worst = max(worst, float(np.max(np.abs(new.to_numpy() - old[key].to_numpy()))))
print(f"8 variants in {time.time()-t0:.1f}s, max abs diff vs old code {worst:.2e}")
