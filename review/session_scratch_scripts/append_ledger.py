import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, sys.argv[1] + "/scripts"); sys.path.insert(0, sys.argv[1] + "/src")
import eodhd_s3_bond_commodity_trend_preregistered_bars as prereg
import pandas as pd
import run_eodhd_s3_evaluation as ev

out_dir = Path(sys.argv[2])
report = json.loads((Path(sys.argv[3])).read_text())
frame = pd.read_parquet(out_dir / "returns_base.parquet")
rf = frame["rf"]
pbo_cols = ev.VARIANT_NAMES + ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"]
pbo_mat = frame[pbo_cols].sub(rf, axis=0).dropna()
trial_daily_sr = (pbo_mat[ev.VARIANT_NAMES].mean() / pbo_mat[ev.VARIANT_NAMES].std(ddof=1)).to_numpy()

# Cross-check against the report's own recorded pbo/dsr_trials for consistency.
assert len(trial_daily_sr) == report["dsr_trials"] == 5, (len(trial_daily_sr), report["dsr_trials"])
fp = prereg.bars_fingerprint()
assert fp == report["fingerprint"], (fp, report["fingerprint"])

LEDGER = Path(sys.argv[1]) / "docs" / "S3_trial_history.json"
ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"family": "S3", "entries": []}
ledger["entries"].append({
    "date": datetime.now(UTC).date().isoformat(), "fingerprint": fp,
    "n_trials": len(trial_daily_sr), "trials": pbo_cols[:5],
    "trial_daily_sharpes": [float(x) for x in trial_daily_sr],
    "tiers": {k: v["tier"] for k, v in report["results"].items()},
})
ledger["cumulative_trials"] = sum(e["n_trials"] for e in ledger["entries"])
LEDGER.write_text(json.dumps(ledger, indent=2))
print("wrote", LEDGER, "cumulative", ledger["cumulative_trials"])
print("trial_daily_sharpes:", [round(x,4) for x in trial_daily_sr])
