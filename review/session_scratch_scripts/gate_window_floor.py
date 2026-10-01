"""Design-only data-availability check (no returns): first Sunday-UTC date at
which >=20 eligible coins exist, under the S5 exclusion rule AND a
trailing-30-day median dollar volume floor of >= $1,000,000/day. Streams one
file at a time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EODHD = Path("data/research/eodhd")
CRYPTO_DIR = EODHD / "crypto"
FLOOR_USD = 1_000_000.0

EXCLUDE_EXACT = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX",
    "LUSD", "MIM", "SUSD", "USDE", "PYUSD", "FDUSD", "UST", "USTC", "EURT",
    "EURS", "USDX", "USDK", "OUSD", "USDN", "VUSD", "MUSD", "DUSD",
    "WBTC", "WETH", "WBNB", "STETH", "WSTETH", "RETH", "CBETH", "CBBTC",
    "BTCB", "RENBTC", "IBETH", "WBETH", "RSETH", "WEETH", "METH", "SFRXETH",
    "FRXETH",
    "XAU", "PAXG", "XAUT", "DJ30", "SPX",
    "BULL1", "XRPBULL", "XRPBEAR", "XRPUP", "XRPDOWN", "3X-LONG-BITCOIN-TOKEN",
}
_SUFFIX_RE = re.compile(r"^([A-Za-z-]+?)([0-9]{3,})$")


def base_symbol(code: str) -> str:
    stem = code[:-4] if code.endswith("-USD") else code
    stem = stem.upper()
    m = _SUFFIX_RE.match(stem)
    return m.group(1) if m else stem


def is_excluded(code: str) -> bool:
    return base_symbol(code) in EXCLUDE_EXACT


def main() -> None:
    files = sorted(f for f in CRYPTO_DIR.glob("*.parquet") if f.stem.endswith("-USD"))
    files = [f for f in files if not is_excluded(f.stem)]
    print(f"{len(files)} eligible-category USD-quoted files (post-exclusion, pre-liquidity-filter)")

    sundays = pd.date_range("2013-01-01", "2026-09-30", freq="W-SUN")
    counts = pd.Series(0, index=sundays, dtype=int)

    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["date", "close", "volume"])
        except Exception:
            continue
        if df.empty or len(df) < 31:
            continue
        df["date"] = pd.to_datetime(df["date"]).sort_values()
        idx = pd.DatetimeIndex(df["date"])
        dvol = (df["close"].to_numpy(dtype=np.float32) * df["volume"].to_numpy(dtype=np.float32))
        idx_vals = idx.values
        lo_all = (sundays - pd.Timedelta(days=30)).values
        starts = np.searchsorted(idx_vals, lo_all, side="left")
        ends = np.searchsorted(idx_vals, sundays.values, side="left")
        has_28d_before = np.searchsorted(idx_vals, (sundays - pd.Timedelta(days=28)).values, side="right") > 0
        ok = (ends > starts) & has_28d_before
        for j in np.where(ok)[0]:
            s, e = starts[j], ends[j]
            if e > s and np.median(dvol[s:e]) >= FLOOR_USD:
                counts.iloc[j] += 1
        del df, idx, dvol, idx_vals
        if i % 1000 == 0:
            print(f"  ...{i}/{len(files)}")

    first20 = counts[counts >= 20].index.min()
    first30 = counts[counts >= 30].index.min()
    print("FLOOR_USD =", FLOOR_USD)
    print("first Sunday with >=20 eligible (floor applied):", first20)
    print("first Sunday with >=30 eligible (floor applied):", first30)
    print(counts[counts.index >= "2014-01-01"].iloc[::13])  # every ~quarter
    counts.to_frame("n_eligible").to_csv("gate_window_floor_counts.csv")


if __name__ == "__main__":
    main()
