"""Shared weekly-bar helpers for Gann cycle event studies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META",
    "TSLA", "AVGO", "AMD", "CRM", "NFLX", "ADBE",
    "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "UNH", "LLY",
    "XOM", "CVX",
    "SPY", "QQQ", "IWM",
]

PivotType = Literal["high", "low"]
PivotRecord = tuple[pd.Timestamp, float, PivotType]


def normalize_price(p: float) -> float:
    if p < 500:
        return p
    if p < 5000:
        return p / 10.0
    return p / 100.0


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLC to ISO weeks (Monday–Sunday grouping)."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)

    rows: list[dict[str, Any]] = []
    for (symbol, iso_year, iso_week), g in df.groupby(["symbol", "iso_year", "iso_week"]):
        g = g.sort_values("date")
        row: dict[str, Any] = {
            "symbol": symbol,
            "week_start_date": pd.Timestamp(g["date"].iloc[0]),
            "open": float(g["open"].iloc[0]) if "open" in g.columns else float(g["adj_close"].iloc[0]),
            "high": float(g["high"].max()) if "high" in g.columns else float(g["adj_close"].max()),
            "low": float(g["low"].min()) if "low" in g.columns else float(g["adj_close"].min()),
            "adj_close": float(g["adj_close"].iloc[-1]),
        }
        if "volume" in g.columns:
            row["volume"] = float(g["volume"].sum())
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["symbol", "week_start_date"]).reset_index(drop=True)


def detect_major_pivots_weekly(
    weekly_highs: np.ndarray,
    weekly_lows: np.ndarray,
    weekly_dates: np.ndarray,
    order: int = 3,
) -> tuple[list[tuple[pd.Timestamp, float]], list[tuple[pd.Timestamp, float]]]:
    """Pivot lists as (week_start_date, price)."""
    n = len(weekly_highs)
    pivot_highs: list[tuple[pd.Timestamp, float]] = []
    pivot_lows: list[tuple[pd.Timestamp, float]] = []
    for i in range(order, n - order):
        if weekly_highs[i] == max(weekly_highs[i - order : i + order + 1]):
            pivot_highs.append((pd.Timestamp(weekly_dates[i]), float(weekly_highs[i])))
        if weekly_lows[i] == min(weekly_lows[i - order : i + order + 1]):
            pivot_lows.append((pd.Timestamp(weekly_dates[i]), float(weekly_lows[i])))
    return pivot_highs, pivot_lows


def pivots_to_records(
    pivot_highs: list[tuple[pd.Timestamp, float]],
    pivot_lows: list[tuple[pd.Timestamp, float]],
) -> list[PivotRecord]:
    records = [(d, p, "high") for d, p in pivot_highs] + [(d, p, "low") for d, p in pivot_lows]
    return sorted(records, key=lambda x: x[0])


def load_prices(cache_dir: str, start: str, end: str) -> pd.DataFrame:
    from firm.data.cache import ParquetCache

    cache = ParquetCache(cache_dir)
    prices_df = cache.get("combined/prices")
    if prices_df is None or prices_df.empty:
        print(
            "ERROR: No cached price data at combined/prices.\n"
            "Run: python scripts/fetch_data.py --symbols AAPL,MSFT,... "
            f"--start {start} --end {end}",
            file=sys.stderr,
        )
        sys.exit(1)

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    if "adj_close" not in prices_df.columns and "close" in prices_df.columns:
        prices_df["adj_close"] = prices_df["close"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return prices_df[(prices_df["date"] >= start_ts) & (prices_df["date"] <= end_ts)]


def _within_days(a: pd.Timestamp, b: pd.Timestamp, tolerance_days: int) -> bool:
    return abs((a - b).days) <= tolerance_days


def anniversary_window_active(
    asof: pd.Timestamp,
    pivot_date: pd.Timestamp,
    anniversary_days: list[int],
    tolerance_days: int,
) -> bool:
    for ann in anniversary_days:
        target = pivot_date + pd.Timedelta(days=ann)
        if _within_days(asof, target, tolerance_days):
            return True
    return False
