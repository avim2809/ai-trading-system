"""Design-only data-availability check (no returns): first Sunday-UTC date at
which >=20 (and >=30) eligible coins exist, under the S5 exclusion rule.
Streams one file at a time; keeps only a small per-week eligible-count table.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EODHD = Path("data/research/eodhd")
CRYPTO_DIR = EODHD / "crypto"

# Exact base-symbol (part before "-USD") exclusions -- stablecoins, wrapped/
# staked derivatives, tokenized non-crypto assets, leveraged tokens actually
# observed in the historical top-30-by-dollar-volume scan.
EXCLUDE_EXACT = {
    # stablecoins
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX",
    "LUSD", "MIM", "SUSD", "USDE", "PYUSD", "FDUSD", "UST", "USTC", "EURT",
    "EURS", "USDX", "USDK", "OUSD", "USDN", "VUSD", "MUSD", "DUSD",
    # wrapped / staked / liquid-restaking derivatives of another coin
    "WBTC", "WETH", "WBNB", "STETH", "WSTETH", "RETH", "CBETH", "CBBTC",
    "CBBTC32994", "BTCB", "RENBTC", "IBETH", "WBETH", "RSETH", "WEETH",
    "METH", "SFRXETH", "FRXETH",
    # tokenized non-crypto assets (metals, indices)
    "XAU", "PAXG", "XAUT", "DJ30", "SPX",
    # leveraged/inverse tokens seen in the top-30 scan
    "BULL1", "XRPBULL", "XRPBEAR", "XRPUP", "XRPDOWN", "3X-LONG-BITCOIN-TOKEN",
}
# regex for ticker-collision numeric suffixes, e.g. TAO22974 -> TAO
_SUFFIX_RE = re.compile(r"^([A-Za-z]+)[0-9]{3,}$")


def base_symbol(code: str) -> str:
    stem = code.removesuffix("-USD")
    return stem.upper()


def is_excluded(code: str) -> bool:
    b = base_symbol(code)
    if b in EXCLUDE_EXACT:
        return True
    m = _SUFFIX_RE.match(b)
    if m and m.group(1) in EXCLUDE_EXACT:
        return True
    return False


def main() -> None:
    files = sorted(f for f in CRYPTO_DIR.glob("*.parquet") if f.stem.endswith("-USD"))
    files = [f for f in files if not is_excluded(f.stem)]
    print(f"{len(files)} eligible-category USD-quoted files (post-exclusion, pre-liquidity-filter)")

    # Sunday grid from 2013-01-01 (before any real usage) to 2026-09-28 (last Sunday <= data end)
    sundays = pd.date_range("2013-01-01", "2026-09-30", freq="W-SUN")

    # per-file: a boolean "has a bar in [sunday-30, sunday)" with median $vol,
    # and "has a bar exactly at or before sunday with >=29 days of trailing
    # history" -- accumulate counts per sunday without keeping full frames.
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
        # for each sunday, eligible if there's a bar in the 30d window before
        # it AND a bar >=29 days before it (for the 28d return lookback) AND
        # median 30d $vol > 0
        lo_all = sundays - pd.Timedelta(days=30)
        # searchsorted on sorted idx
        idx_vals = idx.values
        starts = np.searchsorted(idx_vals, lo_all.values, side="left")
        ends = np.searchsorted(idx_vals, sundays.values, side="left")
        has_28d_before = np.searchsorted(idx_vals, (sundays - pd.Timedelta(days=28)).values, side="right") > 0
        ok = (ends > starts) & has_28d_before
        for j in np.where(ok)[0]:
            s, e = starts[j], ends[j]
            if e > s and np.median(dvol[s:e]) > 0:
                counts.iloc[j] += 1
        del df, idx, dvol, idx_vals
        if i % 1000 == 0:
            print(f"  ...{i}/{len(files)}")

    first20 = counts[counts >= 20].index.min()
    first30 = counts[counts >= 30].index.min()
    print("first Sunday with >=20 eligible:", first20)
    print("first Sunday with >=30 eligible:", first30)
    print(counts[counts.index >= "2014-01-01"].iloc[::26])  # every ~6 months, sanity
    counts.to_frame("n_eligible").to_csv("gate_window_counts.csv")


if __name__ == "__main__":
    main()
