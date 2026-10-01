"""Gate check: for each Jan-1 anchor date 2018-2025, what fraction of the
top-30-by-trailing-30-day-median-dollar-volume coins at that date are now
in the delisted list? Streams one parquet file at a time, keeps only the
handful of scalar numbers needed per anchor date, to respect the RSS cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EODHD = Path("data/research/eodhd")
CRYPTO_DIR = EODHD / "crypto"

ANCHORS = [pd.Timestamp(f"{y}-01-01") for y in range(2018, 2026)]
LOOKBACK = 30

STABLE_WRAPPED_SUBSTR = (
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX",
    "UST-", "USTC", "WBTC", "WETH", "WBNB", "STETH", "WSTETH", "RETH",
    "CBETH", "SUSD", "LUSD", "MIM-", "USDE", "PYUSD", "FDUSD", "EURT",
    "EURS",
)


def is_stable_or_wrapped(code: str) -> bool:
    u = code.upper()
    return any(s in u for s in STABLE_WRAPPED_SUBSTR)


def main() -> None:
    sym = pd.read_parquet(EODHD / "crypto_symbols.parquet")
    delisted_set = set(sym.loc[sym["listing"] == "delisted", "code"])
    # restrict to USD-quoted pairs only (exclude BTC-EUR, BTC-GBP, etc. --
    # otherwise the same coin's non-USD quote pairs pollute the dollar-volume
    # ranking as if they were separate assets).
    files = sorted(f for f in CRYPTO_DIR.glob("*.parquet") if f.stem.endswith("-USD"))
    print(f"scanning {len(files)} files (USD-quoted only)")

    # per-anchor list of (code, dollar_vol)
    per_anchor: dict[pd.Timestamp, list[tuple[str, float]]] = {a: [] for a in ANCHORS}

    for i, f in enumerate(files):
        code = f.stem
        try:
            df = pd.read_parquet(f, columns=["date", "close", "volume"])
        except Exception:
            continue
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        dates = df["date"].to_numpy()
        close = df["close"].to_numpy(dtype=np.float32)
        vol = df["volume"].to_numpy(dtype=np.float32)
        dvol = close * vol
        idx = pd.DatetimeIndex(dates)
        for a in ANCHORS:
            lo = a - pd.Timedelta(days=LOOKBACK)
            mask = (idx >= lo) & (idx < a)
            if mask.sum() < LOOKBACK // 2:  # need reasonable coverage
                continue
            med = float(np.median(dvol[mask]))
            if np.isfinite(med) and med > 0:
                per_anchor[a].append((code, med))
        del df, dates, close, vol, dvol, idx
        if i % 1000 == 0:
            print(f"  ...{i}/{len(files)}")

    results = {}
    for a in ANCHORS:
        rows = per_anchor[a]
        rows_nonstable = [(c, v) for c, v in rows if not is_stable_or_wrapped(c)]
        rows_nonstable.sort(key=lambda x: -x[1])
        top30 = rows_nonstable[:30]
        n_delisted = sum(1 for c, _ in top30 if c in delisted_set)
        results[str(a.date())] = {
            "n_eligible": len(rows_nonstable),
            "top30": [c for c, _ in top30],
            "n_top30_now_delisted": n_delisted,
            "frac_top30_now_delisted": n_delisted / len(top30) if top30 else None,
        }
        print(a.date(), "n_eligible=", len(rows_nonstable), "top30 delisted now:", n_delisted, "/", len(top30))

    out = Path("gate_topvol_results.json")
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
