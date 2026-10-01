"""One-off data-availability scan for S1 (industry/sector ETF momentum).
Looks ONLY at availability (coverage, first/last clean date, ADV20 level,
count eligible over time) -- never at returns. Not committed; output goes to
$S/runs/S1/availability.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/wt_S1")
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts"))

from eodhd_clean import clean_bars, equity_calendar, cleaning_fingerprint  # noqa: E402

EODHD = WT / "data" / "research" / "eodhd" / "etfs_full"

SECTOR_11 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
INDUSTRY = ["SMH", "SOXX", "XSD", "XBI", "IBB", "XPH", "IHI", "KBE", "KRE", "KIE", "IAI", "XHB",
            "ITB", "XRT", "XOP", "OIH", "XES", "XME", "GDX", "GDXJ", "IYT", "XTN", "ITA", "XAR",
            "IGV", "FDN", "VNQ", "IYR", "PBJ", "XHS", "IYZ", "TAN", "ICLN"]
UNIVERSE = SECTOR_11 + INDUSTRY

ADV_MIN_USD = 1_000_000  # candidate eligibility floor (reported, decided from this scan)
MIN_HISTORY_CALENDAR_DAYS = 210  # ~7 months of calendar days for a 6-1 formation window


def main() -> None:
    cal = equity_calendar()
    per_etf = {}
    all_dates = set()
    for t in UNIVERSE:
        f = EODHD / f"{t}.parquet"
        if not f.exists():
            per_etf[t] = {"status": "missing"}
            continue
        raw = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
        d, rep = clean_bars(raw, asset="equity", calendar=cal)
        if d.empty:
            per_etf[t] = {"status": "empty_after_clean", "clean_report": rep}
            continue
        d["dollar_vol"] = d["adjusted_close"].astype(float) * d["volume"].astype(float)
        adv20 = d["dollar_vol"].rolling(20, min_periods=20).median()
        n_segments = int(d["segment"].nunique())
        last_seg = d["segment"].iloc[-1]
        # "clean, usable" history = within the LAST segment only (no return computed
        # across a segment boundary), from first bar of that segment.
        last_seg_df = d[d["segment"] == last_seg]
        per_etf[t] = {
            "status": "ok",
            "first_date_any_segment": str(d["date"].iloc[0].date()),
            "last_date": str(d["date"].iloc[-1].date()),
            "n_rows_clean": int(len(d)),
            "n_segments": n_segments,
            "last_segment_first_date": str(last_seg_df["date"].iloc[0].date()),
            "last_segment_n_rows": int(len(last_seg_df)),
            "adv20_usd_at_end": None if pd.isna(adv20.iloc[-1]) else round(float(adv20.iloc[-1]), 0),
            "adv20_usd_median_last_seg": None if last_seg_df.empty else round(
                float((last_seg_df["adjusted_close"].astype(float) * last_seg_df["volume"].astype(float))
                      .rolling(20, min_periods=20).median().median()), 0),
            "clean_report": rep,
        }
        all_dates.update(d["date"].tolist())

    # Point-in-time eligibility: on each exchange session, an ETF is "eligible" once
    # (a) it has >= MIN_HISTORY_CALENDAR_DAYS of clean history within its current
    #     (post-break) segment, ending that session, and (b) its ADV20 that session
    #     is >= ADV_MIN_USD. Recomputed from the same per-day frames (availability
    #     only, no returns).
    frames = {}
    for t in UNIVERSE:
        f = EODHD / f"{t}.parquet"
        if not f.exists():
            continue
        raw = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
        d, _ = clean_bars(raw, asset="equity", calendar=cal)
        if d.empty:
            continue
        d["dollar_vol"] = d["adjusted_close"].astype(float) * d["volume"].astype(float)
        # ADV20 computed WITHIN each segment only (no lookback across a break).
        d["adv20"] = d.groupby("segment")["dollar_vol"].transform(
            lambda s: s.rolling(20, min_periods=20).median())
        # days_in_segment as of each bar's OWN segment (point-in-time: at date dt we
        # only know dt's own segment, not whether a future break will start a new one).
        seg_start_date = d.groupby("segment")["date"].transform("min")
        d["days_in_segment"] = (d["date"] - seg_start_date).dt.days
        frames[t] = d.set_index("date")

    sessions = sorted(cal)
    session_dates = pd.DatetimeIndex(sessions)
    eligible_counts = []
    # Sample monthly (month-end sessions) to keep this cheap and match the monthly
    # rebalance cadence; full daily detail is unnecessary for an availability note.
    month_ends = pd.Series(session_dates).groupby(session_dates.to_period("M")).max()
    for dt in month_ends:
        n_elig = 0
        elig_tickers = []
        for t, d in frames.items():
            if dt not in d.index:
                continue
            row = d.loc[dt]
            if row["days_in_segment"] >= MIN_HISTORY_CALENDAR_DAYS and pd.notna(row["adv20"]) and row["adv20"] >= ADV_MIN_USD:
                n_elig += 1
                elig_tickers.append(t)
        eligible_counts.append({"month_end": str(dt.date()), "n_eligible": n_elig, "tickers": elig_tickers})

    out = {
        "universe": UNIVERSE,
        "universe_sector_11": SECTOR_11,
        "universe_industry": INDUSTRY,
        "n_universe": len(UNIVERSE),
        "cleaning_fingerprint": cleaning_fingerprint(),
        "adv_min_usd_eligibility_floor": ADV_MIN_USD,
        "min_history_calendar_days": MIN_HISTORY_CALENDAR_DAYS,
        "per_etf": per_etf,
        "eligible_count_by_month_end": eligible_counts,
    }
    out_dir = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/S1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "availability.json").write_text(json.dumps(out, indent=1, default=str))
    # Summary to stdout
    n_ge9 = [(r["month_end"], r["n_eligible"]) for r in eligible_counts if r["n_eligible"] >= 9]
    print("first month with >=9 eligible:", n_ge9[0] if n_ge9 else None)
    # Confirm it never dips below 9 after that first crossing.
    if n_ge9:
        start_idx = next(i for i, r in enumerate(eligible_counts) if r["n_eligible"] >= 9)
        tail = eligible_counts[start_idx:]
        dips = [r["month_end"] for r in tail if r["n_eligible"] < 9]
        print("dips below 9 after first crossing:", dips)
        print("min eligible from first crossing onward:", min(r["n_eligible"] for r in tail),
              "over", len(tail), "months")
    print("last month:", eligible_counts[-1])
    print("wrote", out_dir / "availability.json")


if __name__ == "__main__":
    main()
